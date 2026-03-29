from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import torch
import torchvision
from einops import rearrange
from omegaconf import OmegaConf

from losses.lpips import LPIPS
from tools.relat_e_batch import apply_condition_mask, condition_mask, sample_condition_mode
from tools.utils import AverageMeter, Logger


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def setup_relat_e_logger(cfg, rank):
    if rank != 0:
        return print, None
    os.makedirs(cfg.experiment.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{cfg.experiment.name}_{cfg.data.name}_{cfg.experiment.seed}"
    logger = Logger(name, path=cfg.experiment.output, resume=cfg.experiment.resume)
    logger.log(OmegaConf.to_yaml(cfg))
    return logger.log, logger


def prepare_relat_e_batch(batch, cfg, device, mode=None, future_frames=None):
    rgb = batch["rgb"].to(device=device, dtype=torch.float32)
    depth = batch["depth"].to(device=device, dtype=torch.float32)
    rgb = rearrange(rgb / 127.5 - 1.0, "b t c h w -> b c t h w")
    depth = rearrange(depth / 127.5 - 1.0, "b t c h w -> b c t h w")
    cond_frames = int(cfg.data.cond_frames)
    pred_frames = int(cfg.data.pred_frames if future_frames is None else future_frames)
    mode = sample_condition_mode(cfg.partial_conditioning.probs) if mode is None else mode
    mask = condition_mask(rgb.size(0), mode, device)
    return {
        "rgb_past": rgb[:, :, :cond_frames],
        "rgb_future": rgb[:, :, cond_frames:cond_frames + pred_frames],
        "depth_past": depth[:, :, :cond_frames],
        "depth_future": depth[:, :, cond_frames:cond_frames + pred_frames],
        "mode": mode,
        "modality_mask": mask,
        "meta": batch["meta"],
    }


def build_condition_latents(rgb_cond, depth_cond, modality_mask):
    return apply_condition_mask(rgb_cond, depth_cond, modality_mask)


def warmup_teacher(teacher, channels, frames, resolution, device):
    dummy = torch.zeros(1, channels, frames, resolution, resolution, device=device)
    with torch.no_grad():
        teacher.extract(dummy)
    return teacher.output_dim


def save_relat_e_checkpoint(path, rgb_vae, depth_vae, generator, ema_rgb_vae, ema_depth_vae, ema_generator, opt_rgb_vae, opt_depth_vae, opt_generator, step):
    torch.save(
        {
            "vae_rgb_model": unwrap_model(rgb_vae).state_dict(),
            "vae_depth_model": unwrap_model(depth_vae).state_dict(),
            "generator_model": unwrap_model(generator).state_dict(),
            "ema_vae_rgb": ema_rgb_vae.state_dict(),
            "ema_vae_depth": ema_depth_vae.state_dict(),
            "ema_generator": ema_generator.state_dict(),
            "opt_rgb_vae": opt_rgb_vae.state_dict(),
            "opt_depth_vae": opt_depth_vae.state_dict(),
            "opt_generator": opt_generator.state_dict(),
            "step": step,
        },
        path,
    )


def load_relat_e_checkpoint(
    path,
    rgb_vae,
    depth_vae,
    generator,
    ema_rgb_vae=None,
    ema_depth_vae=None,
    ema_generator=None,
    opt_rgb_vae=None,
    opt_depth_vae=None,
    opt_generator=None,
    map_location="cpu",
    use_ema=False,
):
    ckpt = torch.load(path, map_location=map_location)
    if use_ema and "ema_vae_rgb" in ckpt:
        unwrap_model(rgb_vae).load_state_dict(ckpt["ema_vae_rgb"])
        unwrap_model(depth_vae).load_state_dict(ckpt["ema_vae_depth"])
        unwrap_model(generator).load_state_dict(ckpt["ema_generator"])
    else:
        unwrap_model(rgb_vae).load_state_dict(ckpt["vae_rgb_model"])
        unwrap_model(depth_vae).load_state_dict(ckpt["vae_depth_model"])
        unwrap_model(generator).load_state_dict(ckpt["generator_model"])
    if ema_rgb_vae is not None and "ema_vae_rgb" in ckpt:
        ema_rgb_vae.load_state_dict(ckpt["ema_vae_rgb"])
    if ema_depth_vae is not None and "ema_vae_depth" in ckpt:
        ema_depth_vae.load_state_dict(ckpt["ema_vae_depth"])
    if ema_generator is not None and "ema_generator" in ckpt:
        ema_generator.load_state_dict(ckpt["ema_generator"])
    if opt_rgb_vae is not None and "opt_rgb_vae" in ckpt:
        opt_rgb_vae.load_state_dict(ckpt["opt_rgb_vae"])
    if opt_depth_vae is not None and "opt_depth_vae" in ckpt:
        opt_depth_vae.load_state_dict(ckpt["opt_depth_vae"])
    if opt_generator is not None and "opt_generator" in ckpt:
        opt_generator.load_state_dict(ckpt["opt_generator"])
    return int(ckpt.get("step", 0))


def _to_display(video):
    if video.size(1) == 1:
        video = video.repeat(1, 3, 1, 1, 1)
    return ((video + 1.0) / 2.0).clamp(0.0, 1.0)


def save_paired_visualization(path, rgb_real, rgb_pred, depth_real, depth_pred, max_items=1):
    rgb_real = _to_display(rgb_real[:max_items]).permute(0, 2, 1, 3, 4)
    rgb_pred = _to_display(rgb_pred[:max_items]).permute(0, 2, 1, 3, 4)
    depth_real = _to_display(depth_real[:max_items]).permute(0, 2, 1, 3, 4)
    depth_pred = _to_display(depth_pred[:max_items]).permute(0, 2, 1, 3, 4)
    rows = []
    for b_idx in range(rgb_real.size(0)):
        rows.extend([rgb_real[b_idx], rgb_pred[b_idx], depth_real[b_idx], depth_pred[b_idx]])
    grid = torch.cat(rows, dim=0)
    torchvision.utils.save_image(grid, path, nrow=rgb_real.size(1), normalize=False)


def _psnr(reals, preds):
    mse = torch.mean((reals - preds) ** 2)
    if mse <= 0:
        return 100.0
    return 10 * torch.log10(1.0 / mse).item()


def _ssim(reals, preds):
    try:
        from skimage.metrics import structural_similarity as ssim_func
    except ImportError:
        return 0.0
    reals_np = reals.cpu().numpy()
    preds_np = preds.cpu().numpy()
    total = 0.0
    count = 0
    for b_idx in range(reals_np.shape[0]):
        for t_idx in range(reals_np.shape[2]):
            img1 = np.transpose(reals_np[b_idx, :, t_idx], (1, 2, 0))
            img2 = np.transpose(preds_np[b_idx, :, t_idx], (1, 2, 0))
            if img1.shape[-1] == 1:
                img1 = img1[..., 0]
                img2 = img2[..., 0]
                total += ssim_func(img1, img2, data_range=1.0)
            else:
                total += ssim_func(img1, img2, data_range=1.0, channel_axis=-1)
            count += 1
    return total / max(count, 1)


def _lpips(reals, preds, model):
    if model is None:
        return 0.0
    if reals.size(1) != 3:
        return 0.0
    reals_2d = reals.transpose(1, 2).flatten(0, 1) * 2.0 - 1.0
    preds_2d = preds.transpose(1, 2).flatten(0, 1) * 2.0 - 1.0
    with torch.no_grad():
        return model(reals_2d, preds_2d).mean().item()


def _fvd(reals, preds, i3d, device):
    if i3d is None:
        return 0.0
    try:
        from evals.fvd.fvd import calculate_fvd
    except ImportError:
        return 0.0
    reals_uint8 = (reals * 255).clamp(0, 255).to(torch.uint8)
    preds_uint8 = (preds * 255).clamp(0, 255).to(torch.uint8)
    if reals_uint8.size(1) == 1:
        reals_uint8 = reals_uint8.repeat(1, 3, 1, 1, 1)
        preds_uint8 = preds_uint8.repeat(1, 3, 1, 1, 1)
    reals_fvd = reals_uint8.permute(0, 2, 3, 4, 1).contiguous()
    preds_fvd = preds_uint8.permute(0, 2, 3, 4, 1).contiguous()
    value = calculate_fvd(reals_fvd.to(device), preds_fvd.to(device), i3d, device)
    return value.item() if isinstance(value, torch.Tensor) else float(value)


def collect_metrics(reals, preds, device, use_lpips=True, use_fvd=True):
    reals_01 = ((reals + 1.0) / 2.0).clamp(0.0, 1.0)
    preds_01 = ((preds + 1.0) / 2.0).clamp(0.0, 1.0)
    lpips_model = None
    i3d = None
    if use_lpips and reals.size(1) == 3:
        lpips_model = LPIPS().eval().to(device)
    if use_fvd:
        try:
            from evals.fvd.download import load_i3d_pretrained

            i3d = load_i3d_pretrained(device)
        except Exception:
            i3d = None
    return {
        "psnr": _psnr(reals_01, preds_01),
        "ssim": _ssim(reals_01, preds_01),
        "lpips": _lpips(reals_01.to(device), preds_01.to(device), lpips_model),
        "fvd": _fvd(reals_01, preds_01, i3d, device),
    }


@torch.no_grad()
def rollout_future(batch, rgb_vae, depth_vae, generator, flow, cfg, device, condition_mode="full", future_frames=None):
    future_frames = cfg.eval.future_frames if future_frames is None else future_frames
    cond_frames = int(cfg.data.cond_frames)
    pred_frames = int(cfg.data.pred_frames)
    prepared = prepare_relat_e_batch(batch, cfg, device, mode=condition_mode, future_frames=future_frames)
    rgb_context = prepared["rgb_past"]
    depth_context = prepared["depth_past"]
    rgb_preds = []
    depth_preds = []
    generator = unwrap_model(generator)
    rgb_vae = unwrap_model(rgb_vae)
    depth_vae = unwrap_model(depth_vae)
    while sum(item.size(2) for item in rgb_preds) < future_frames:
        rgb_cond = rgb_vae.encode_past(rgb_context[:, :, -cond_frames:], normalize=True)
        depth_cond = depth_vae.encode_past(depth_context[:, :, -cond_frames:], normalize=True)
        rgb_cond, depth_cond = build_condition_latents(rgb_cond, depth_cond, prepared["modality_mask"])
        latent_shape = (unwrap_model(generator).in_channels, unwrap_model(generator).ae_emb_dim)
        z_rgb, z_depth = flow.sample(
            generator=generator,
            batch_size=rgb_context.size(0),
            latent_shape=latent_shape,
            cond_rgb=rgb_cond,
            cond_depth=depth_cond,
            guidance_scale=cfg.eval.cfg_scale,
        )
        rgb_seg = rgb_vae.decode(z_rgb, num_frames=pred_frames)
        depth_seg = depth_vae.decode(z_depth, num_frames=pred_frames)
        rgb_preds.append(rgb_seg)
        depth_preds.append(depth_seg)
        rgb_context = torch.cat([rgb_context, rgb_seg], dim=2)
        depth_context = torch.cat([depth_context, depth_seg], dim=2)
    rgb_pred = torch.cat(rgb_preds, dim=2)[:, :, :future_frames]
    depth_pred = torch.cat(depth_preds, dim=2)[:, :, :future_frames]
    return {
        "rgb_real": prepared["rgb_future"][:, :, :future_frames],
        "depth_real": prepared["depth_future"][:, :, :future_frames],
        "rgb_pred": rgb_pred,
        "depth_pred": depth_pred,
    }


@torch.no_grad()
def run_relat_e_evaluation(val_loader, rgb_vae, depth_vae, generator, flow, cfg, device, step, logger, log_):
    max_samples = int(cfg.eval.eval_samples)
    rgb_reals, rgb_preds = [], []
    depth_reals, depth_preds = [], []
    seen = 0
    for batch in val_loader:
        rollout = rollout_future(
            batch,
            rgb_vae,
            depth_vae,
            generator,
            flow,
            cfg,
            device,
            condition_mode="full",
            future_frames=cfg.eval.future_frames,
        )
        rgb_reals.append(rollout["rgb_real"].cpu())
        rgb_preds.append(rollout["rgb_pred"].cpu())
        depth_reals.append(rollout["depth_real"].cpu())
        depth_preds.append(rollout["depth_pred"].cpu())
        seen += rollout["rgb_real"].size(0)
        if seen >= max_samples:
            break

    if not rgb_reals:
        return
    rgb_real = torch.cat(rgb_reals, dim=0)[:max_samples]
    rgb_pred = torch.cat(rgb_preds, dim=0)[:max_samples]
    depth_real = torch.cat(depth_reals, dim=0)[:max_samples]
    depth_pred = torch.cat(depth_preds, dim=0)[:max_samples]

    rgb_metrics = collect_metrics(rgb_real, rgb_pred, device)
    depth_metrics = collect_metrics(depth_real, depth_pred, device, use_lpips=(depth_real.size(1) == 3))
    log_(f"[Eval RGB step {step}] PSNR: {rgb_metrics['psnr']:.4f} | SSIM: {rgb_metrics['ssim']:.4f} | LPIPS: {rgb_metrics['lpips']:.4f} | FVD: {rgb_metrics['fvd']:.4f}")
    log_(f"[Eval Depth step {step}] PSNR: {depth_metrics['psnr']:.4f} | SSIM: {depth_metrics['ssim']:.4f} | LPIPS: {depth_metrics['lpips']:.4f} | FVD: {depth_metrics['fvd']:.4f}")
    if logger is not None:
        for name, value in rgb_metrics.items():
            logger.scalar_summary(f"eval/rgb_{name}", value, step)
        for name, value in depth_metrics.items():
            logger.scalar_summary(f"eval/depth_{name}", value, step)

    save_path = os.path.join(logger.logdir if logger is not None else cfg.experiment.output, f"relat_e_eval_{step:07d}.png")
    save_paired_visualization(save_path, rgb_real, rgb_pred, depth_real, depth_pred, max_items=1)


def init_metric_meters():
    keys = [
        "loss/vae_rgb",
        "loss/vae_depth",
        "loss/relation_rgb",
        "loss/relation_depth",
        "loss/generator_relation",
        "loss/flow_rgb",
        "loss/flow_depth",
        "recon/rgb_mse",
        "recon/rgb_l1",
        "recon/rgb_perceptual",
        "recon/rgb_kl",
        "recon/depth_mse",
        "recon/depth_l1",
        "recon/depth_perceptual",
        "recon/depth_kl",
    ]
    return {key: AverageMeter() for key in keys}
