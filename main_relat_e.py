from __future__ import annotations

import argparse
import copy
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from losses.reconstruction import AutoencoderReconstructionLoss
from losses.relational import RelationalAlignmentLoss
from losses.relat_flow import RelatEFlowMatching
from models.fm.relat_mot import ReLaTMoT
from models.vae.relat_autoencoder import Relat3DVAE
from tools.relat_e_dataloader import get_relat_e_loaders
from tools.relat_e_scaling import MODEL_SCALE_PRESETS, apply_model_scale
from tools.relat_e_teachers import create_teacher
from tools.relat_e_utils import (
    build_condition_latents,
    init_metric_meters,
    load_relat_e_checkpoint,
    prepare_relat_e_batch,
    run_relat_e_evaluation,
    save_relat_e_checkpoint,
    set_requires_grad,
    setup_relat_e_logger,
    unwrap_model,
    warmup_teacher,
)
from tools.train_utils import init_multiprocessing, update_ema
from tools.utils import set_random_seed, setup_distibuted_training


def optimizer_step(loss, optimizers, scaler, parameter_groups, grad_clip):
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    if not isinstance(parameter_groups, (list, tuple)):
        parameter_groups = [parameter_groups]
    elif len(parameter_groups) > 0 and isinstance(parameter_groups[0], (torch.Tensor, nn.Parameter)):
        parameter_groups = [parameter_groups]
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
        for parameters in parameter_groups:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        for parameters in parameter_groups:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        for optimizer in optimizers:
            optimizer.step()


def extract_teacher_tokens(teacher, video, chunk_size):
    if not chunk_size or chunk_size <= 0 or video.size(0) <= chunk_size:
        return teacher.extract(video)
    outputs = []
    for start in range(0, video.size(0), chunk_size):
        outputs.append(teacher.extract(video[start:start + chunk_size]))
    return torch.cat(outputs, dim=0)


def compute_vae_branch(vae, recon_loss_fn, past, future, save_memory):
    past_recon, past_kl, _ = vae(x=past, mode="reconstruct")
    past_total, past_stats = recon_loss_fn(past, past_recon, past_kl)
    if save_memory:
        del past_recon

    future_recon, future_kl, _ = vae(x=future, mode="reconstruct")
    future_total, future_stats = recon_loss_fn(future, future_recon, future_kl)
    if save_memory:
        del future_recon

    cond_latent = vae(x=past, mode="extract", normalize=True, conditioning=True)
    future_latent = vae(x=future, mode="extract", normalize=True, conditioning=False)
    total = past_total + future_total
    stats = {
        "mse": 0.5 * (past_stats["mse"] + future_stats["mse"]),
        "l1": 0.5 * (past_stats["l1"] + future_stats["l1"]),
        "perceptual": 0.5 * (past_stats["perceptual"] + future_stats["perceptual"]),
        "kl": 0.5 * (past_stats["kl"] + future_stats["kl"]),
    }
    return total, stats, cond_latent, future_latent


def run(rank, cfg, ckpt_path=None):
    distributed = cfg.experiment.n_gpus > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA in this implementation.")
        setup_distibuted_training(cfg.experiment, rank)
        init_multiprocessing(rank=rank, sync_device=torch.device("cuda", rank))
        torch.cuda.set_device(rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", rank if distributed else 0)
    else:
        device = torch.device("cpu")

    set_random_seed(cfg.experiment.seed + rank)
    log_, logger = setup_relat_e_logger(cfg, rank)
    loader_cfg = copy.deepcopy(cfg)
    loader_cfg.data.clip_frames = int(cfg.data.cond_frames + max(cfg.data.pred_frames, cfg.eval.future_frames))
    train_loader, val_loader, _ = get_relat_e_loaders(rank, loader_cfg)

    rgb_vae = Relat3DVAE(cfg.rgb_vae.embed_dim, cfg.rgb_vae, bn_momentum=cfg.rgb_vae.bn_momentum).to(device)
    depth_vae = Relat3DVAE(cfg.depth_vae.embed_dim, cfg.depth_vae, bn_momentum=cfg.depth_vae.bn_momentum).to(device)

    rgb_teacher = create_teacher(cfg.teachers.rgb, device)
    depth_teacher = create_teacher(cfg.teachers.depth, device)
    rgb_teacher_dim = warmup_teacher(
        rgb_teacher,
        channels=cfg.rgb_vae.in_channels,
        frames=cfg.data.pred_frames,
        resolution=cfg.data.resolution,
        device=device,
    ) or int(cfg.teachers.rgb.target_dim)
    depth_teacher_dim = warmup_teacher(
        depth_teacher,
        channels=cfg.depth_vae.in_channels,
        frames=cfg.data.pred_frames,
        resolution=cfg.data.resolution,
        device=device,
    ) or int(cfg.teachers.depth.target_dim)

    generator = ReLaTMoT(
        input_size=cfg.generator.mot.input_size,
        in_channels=cfg.generator.mot.in_channels,
        hidden_size=cfg.generator.mot.hidden_size,
        depth=cfg.generator.mot.depth,
        num_heads=cfg.generator.mot.num_heads,
        frames=cfg.rgb_vae.frames,
        aligned_depth=cfg.generator.mot.aligned_depth,
        rgb_teacher_dim=rgb_teacher_dim,
        depth_teacher_dim=depth_teacher_dim,
        use_qknorm=cfg.generator.mot.use_qknorm,
        use_swiglu=cfg.generator.mot.use_swiglu,
        use_rmsnorm=cfg.generator.mot.use_rmsnorm,
        wo_shift=cfg.generator.mot.wo_shift,
        fused_attn=cfg.generator.mot.fused_attn,
        use_rope=cfg.generator.mot.use_rope,
        same_noise=cfg.generator.mot.same_noise,
        use_checkpoint=bool(cfg.optim.save_memory),
    ).to(device)

    ema_rgb_vae = copy.deepcopy(rgb_vae).to(device).eval()
    ema_depth_vae = copy.deepcopy(depth_vae).to(device).eval()
    ema_generator = copy.deepcopy(generator).to(device).eval()
    update_ema(ema_rgb_vae, rgb_vae, decay=0.0)
    update_ema(ema_depth_vae, depth_vae, decay=0.0)
    update_ema(ema_generator, generator, decay=0.0)
    set_requires_grad(ema_rgb_vae, False)
    set_requires_grad(ema_depth_vae, False)
    set_requires_grad(ema_generator, False)

    if distributed:
        rgb_vae = nn.parallel.DistributedDataParallel(rgb_vae, device_ids=[rank], find_unused_parameters=True)
        depth_vae = nn.parallel.DistributedDataParallel(depth_vae, device_ids=[rank], find_unused_parameters=True)
        generator = nn.parallel.DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True)

    rgb_recon_loss = AutoencoderReconstructionLoss(
        mse_weight=cfg.loss.recon.rgb.mse_weight,
        l1_weight=cfg.loss.recon.rgb.l1_weight,
        perceptual_weight=cfg.loss.recon.rgb.perceptual_weight,
        kl_weight=cfg.loss.recon.rgb.kl_weight,
        channels=cfg.rgb_vae.out_channels,
    ).to(device)
    depth_recon_loss = AutoencoderReconstructionLoss(
        mse_weight=cfg.loss.recon.depth.mse_weight,
        l1_weight=cfg.loss.recon.depth.l1_weight,
        perceptual_weight=cfg.loss.recon.depth.perceptual_weight,
        kl_weight=cfg.loss.recon.depth.kl_weight,
        channels=cfg.depth_vae.out_channels,
    ).to(device)
    relation_loss = RelationalAlignmentLoss(
        input_size=cfg.generator.mot.input_size,
        frames=cfg.rgb_vae.frames,
        beta_spatial=cfg.loss.relation.beta_spatial,
        beta_temporal=cfg.loss.relation.beta_temporal,
    ).to(device)
    flow = RelatEFlowMatching(
        sampling_timesteps=cfg.loss.flow.sampling_timesteps,
        sigma_min=cfg.loss.flow.sigma_min,
        same_noise=cfg.generator.mot.same_noise,
    ).to(device)

    opt_rgb_vae = torch.optim.AdamW(
        unwrap_model(rgb_vae).parameters(),
        lr=cfg.optim.lr_vae,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )
    opt_depth_vae = torch.optim.AdamW(
        unwrap_model(depth_vae).parameters(),
        lr=cfg.optim.lr_vae,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )
    opt_generator = torch.optim.AdamW(
        unwrap_model(generator).parameters(),
        lr=cfg.optim.lr_generator,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )

    amp_enabled = bool(cfg.optim.amp) and device.type == "cuda"
    scaler_vae = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_generator = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_step = 0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        start_step = load_relat_e_checkpoint(
            ckpt_path,
            rgb_vae=rgb_vae,
            depth_vae=depth_vae,
            generator=generator,
            ema_rgb_vae=ema_rgb_vae,
            ema_depth_vae=ema_depth_vae,
            ema_generator=ema_generator,
            opt_rgb_vae=opt_rgb_vae,
            opt_depth_vae=opt_depth_vae,
            opt_generator=opt_generator,
            map_location=device,
        )
        log_(f"Loaded checkpoint from {ckpt_path} at step {start_step}.")

    meters = init_metric_meters()
    train_iter = iter(train_loader)
    max_iter = int(cfg.optim.max_iter)
    pbar = tqdm(total=max_iter, initial=start_step, dynamic_ncols=True, disable=(rank != 0))

    for step in range(start_step, max_iter):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        prepared = prepare_relat_e_batch(batch, cfg, device)
        with torch.no_grad():
            rgb_future_teacher = extract_teacher_tokens(rgb_teacher, prepared["rgb_future"], int(cfg.optim.teacher_chunk_size))
            depth_future_teacher = extract_teacher_tokens(depth_teacher, prepared["depth_future"], int(cfg.optim.teacher_chunk_size))

        set_requires_grad(rgb_vae, True)
        set_requires_grad(depth_vae, True)
        set_requires_grad(generator, False)
        rgb_vae.train()
        depth_vae.train()
        generator.train()
        opt_rgb_vae.zero_grad(set_to_none=True)
        opt_depth_vae.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            rgb_recon_total, rgb_stats, rgb_cond, rgb_future_latent = compute_vae_branch(
                rgb_vae, rgb_recon_loss, prepared["rgb_past"], prepared["rgb_future"], bool(cfg.optim.save_memory)
            )
            if bool(cfg.optim.save_memory) and device.type == "cuda":
                torch.cuda.empty_cache()
            depth_recon_total, depth_stats, depth_cond, depth_future_latent = compute_vae_branch(
                depth_vae, depth_recon_loss, prepared["depth_past"], prepared["depth_future"], bool(cfg.optim.save_memory)
            )
            rgb_cond, depth_cond = build_condition_latents(rgb_cond, depth_cond, prepared["modality_mask"])

            flow_tuple = flow.sample_training_tuple(rgb_future_latent, depth_future_latent)
            relation_outputs = generator(
                z_rgb_t=flow_tuple["z_rgb_t"],
                z_depth_t=flow_tuple["z_depth_t"],
                cond_rgb=rgb_cond,
                cond_depth=depth_cond,
                t_rgb=flow_tuple["scaled_t_rgb"],
                t_depth=flow_tuple["scaled_t_depth"],
                return_features=True,
            )
            relation_rgb, relation_rgb_stats = relation_loss(relation_outputs["aligned_rgb"], rgb_future_teacher)
            relation_depth, relation_depth_stats = relation_loss(relation_outputs["aligned_depth"], depth_future_teacher)
            vae_rgb_total = rgb_recon_total + cfg.loss.relation.weight * relation_rgb
            vae_depth_total = depth_recon_total + cfg.loss.relation.weight * relation_depth
            vae_total = vae_rgb_total + vae_depth_total

        optimizer_step(
            vae_total,
            optimizers=[opt_rgb_vae, opt_depth_vae],
            scaler=scaler_vae,
            parameter_groups=[list(unwrap_model(rgb_vae).parameters()), list(unwrap_model(depth_vae).parameters())],
            grad_clip=cfg.optim.grad_clip,
        )
        opt_rgb_vae.zero_grad(set_to_none=True)
        opt_depth_vae.zero_grad(set_to_none=True)
        update_ema(ema_rgb_vae, unwrap_model(rgb_vae), decay=cfg.ema.decay)
        update_ema(ema_depth_vae, unwrap_model(depth_vae), decay=cfg.ema.decay)
        if bool(cfg.optim.save_memory):
            del relation_outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        set_requires_grad(rgb_vae, False)
        set_requires_grad(depth_vae, False)
        set_requires_grad(generator, True)
        opt_generator.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            rgb_cond_detached = rgb_cond.detach()
            depth_cond_detached = depth_cond.detach()
            rgb_future_detached = rgb_future_latent.detach()
            depth_future_detached = depth_future_latent.detach()
            relation_flow_tuple = flow.sample_training_tuple(
                rgb_future_detached,
                depth_future_detached,
                t_rgb=flow_tuple["t_rgb"],
                t_depth=flow_tuple["t_depth"],
                noise_rgb=flow_tuple["noise_rgb"],
                noise_depth=flow_tuple["noise_depth"],
            )
            generator_relation_outputs = generator(
                z_rgb_t=relation_flow_tuple["z_rgb_t"],
                z_depth_t=relation_flow_tuple["z_depth_t"],
                cond_rgb=rgb_cond_detached,
                cond_depth=depth_cond_detached,
                t_rgb=relation_flow_tuple["scaled_t_rgb"],
                t_depth=relation_flow_tuple["scaled_t_depth"],
                return_features=True,
            )
            generator_relation_rgb, _ = relation_loss(generator_relation_outputs["aligned_rgb"], rgb_future_teacher)
            generator_relation_depth, _ = relation_loss(generator_relation_outputs["aligned_depth"], depth_future_teacher)
            generator_relation_total = cfg.loss.relation.weight * (generator_relation_rgb + generator_relation_depth)
        if bool(cfg.optim.save_memory):
            del flow_tuple

        optimizer_step(
            generator_relation_total,
            optimizers=opt_generator,
            scaler=scaler_generator,
            parameter_groups=list(unwrap_model(generator).parameters()),
            grad_clip=cfg.optim.grad_clip,
        )
        if bool(cfg.optim.save_memory):
            del generator_relation_outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        opt_generator.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            flow_outputs = generator(
                z_rgb_t=relation_flow_tuple["z_rgb_t"],
                z_depth_t=relation_flow_tuple["z_depth_t"],
                cond_rgb=rgb_cond_detached,
                cond_depth=depth_cond_detached,
                t_rgb=relation_flow_tuple["scaled_t_rgb"],
                t_depth=relation_flow_tuple["scaled_t_depth"],
                return_features=False,
            )
            flow_total, flow_stats = flow.compute_loss(flow_outputs, relation_flow_tuple)
            flow_total = cfg.loss.flow.weight * flow_total

        optimizer_step(
            flow_total,
            optimizers=opt_generator,
            scaler=scaler_generator,
            parameter_groups=list(unwrap_model(generator).parameters()),
            grad_clip=cfg.optim.grad_clip,
        )
        opt_generator.zero_grad(set_to_none=True)
        update_ema(ema_generator, unwrap_model(generator), decay=cfg.ema.decay)
        if bool(cfg.optim.save_memory):
            del relation_flow_tuple, flow_outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        meters["loss/vae_rgb"].update(vae_rgb_total.item(), 1)
        meters["loss/vae_depth"].update(vae_depth_total.item(), 1)
        meters["loss/relation_rgb"].update(relation_rgb.item(), 1)
        meters["loss/relation_depth"].update(relation_depth.item(), 1)
        meters["loss/generator_relation"].update(generator_relation_total.item(), 1)
        meters["loss/flow_rgb"].update(flow_stats["flow_loss_rgb"].item(), 1)
        meters["loss/flow_depth"].update(flow_stats["flow_loss_depth"].item(), 1)
        meters["recon/rgb_mse"].update(rgb_stats["mse"].item(), 1)
        meters["recon/rgb_l1"].update(rgb_stats["l1"].item(), 1)
        meters["recon/rgb_perceptual"].update(rgb_stats["perceptual"].item(), 1)
        meters["recon/rgb_kl"].update(rgb_stats["kl"].item(), 1)
        meters["recon/depth_mse"].update(depth_stats["mse"].item(), 1)
        meters["recon/depth_l1"].update(depth_stats["l1"].item(), 1)
        meters["recon/depth_perceptual"].update(depth_stats["perceptual"].item(), 1)
        meters["recon/depth_kl"].update(depth_stats["kl"].item(), 1)

        if rank == 0:
            pbar.update(1)
            pbar.set_description(
                f"VAE-RGB {meters['loss/vae_rgb'].average:.3f} | VAE-Depth {meters['loss/vae_depth'].average:.3f} | Flow {0.5 * (meters['loss/flow_rgb'].average + meters['loss/flow_depth'].average):.3f}"
            )

        global_step = step + 1
        if global_step % int(cfg.optim.log_freq) == 0 and rank == 0 and logger is not None:
            logger.scalar_summary("loss/relation_rgb_spatial", relation_rgb_stats["spatial"].item(), global_step)
            logger.scalar_summary("loss/relation_rgb_temporal", relation_rgb_stats["temporal"].item(), global_step)
            logger.scalar_summary("loss/relation_depth_spatial", relation_depth_stats["spatial"].item(), global_step)
            logger.scalar_summary("loss/relation_depth_temporal", relation_depth_stats["temporal"].item(), global_step)
            for key, meter in meters.items():
                logger.scalar_summary(key, meter.average, global_step)
            meters = init_metric_meters()

        if rank == 0 and global_step % int(cfg.optim.eval_freq) == 0:
            run_relat_e_evaluation(val_loader, ema_rgb_vae, ema_depth_vae, ema_generator, flow, cfg, device, global_step, logger, log_)

        if rank == 0 and global_step % int(cfg.optim.save_freq) == 0:
            ckpt_dir = logger.logdir if logger is not None else cfg.experiment.output
            save_relat_e_checkpoint(
                os.path.join(ckpt_dir, f"relat_e_{global_step:07d}.pt"),
                rgb_vae=rgb_vae,
                depth_vae=depth_vae,
                generator=generator,
                ema_rgb_vae=ema_rgb_vae,
                ema_depth_vae=ema_depth_vae,
                ema_generator=ema_generator,
                opt_rgb_vae=opt_rgb_vae,
                opt_depth_vae=opt_depth_vae,
                opt_generator=opt_generator,
                step=global_step,
            )

    pbar.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/relat_e_rgbd.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument("--model_scale", type=str, default="", choices=sorted(list(MODEL_SCALE_PRESETS.keys()) + ["custom"]))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    resolved_scale = apply_model_scale(cfg, args.model_scale or None)
    if resolved_scale != "custom" and not str(cfg.experiment.name).endswith(f"_{resolved_scale}"):
        cfg.experiment.name = f"{cfg.experiment.name}_{resolved_scale}"
    if args.output:
        cfg.experiment.output = args.output
    if args.save_memory:
        cfg.optim.save_memory = True
        if int(cfg.optim.teacher_chunk_size) <= 0:
            cfg.optim.teacher_chunk_size = 1
    if cfg.experiment.n_gpus > 1:
        torch.multiprocessing.spawn(run, args=(cfg, args.ckpt or None), nprocs=cfg.experiment.n_gpus)
    else:
        run(rank=0, cfg=cfg, ckpt_path=args.ckpt or None)
