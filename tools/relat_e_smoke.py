from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from losses.relational import RelationalAlignmentLoss
from losses.relat_flow import RelatEFlowMatching
from models.fm.relat_mot import ReLaTMoT
from models.vae.relat_autoencoder import RelatAutoencoder
from tools.relat_e_batch import condition_mask
from tools.relat_e_scaling import MODEL_SCALE_PRESETS, apply_model_scale
from tools.relat_e_teachers import create_teacher


def grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.abs().sum().item())
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/relat_e_rgbd.yaml")
    parser.add_argument("--model_scale", type=str, default="", choices=sorted(list(MODEL_SCALE_PRESETS.keys()) + ["custom"]))
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    apply_model_scale(cfg, args.model_scale or None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 2
    total_frames = cfg.data.cond_frames + cfg.data.pred_frames

    rgb_vae = RelatAutoencoder(cfg.rgb_vae.embed_dim, cfg.rgb_vae, bn_momentum=cfg.rgb_vae.bn_momentum).to(device)
    depth_vae = RelatAutoencoder(cfg.depth_vae.embed_dim, cfg.depth_vae, bn_momentum=cfg.depth_vae.bn_momentum).to(device)
    rgb_teacher = create_teacher(cfg.teachers.rgb, device)
    depth_teacher = create_teacher(cfg.teachers.depth, device)

    rgb_sample = torch.randn(batch, cfg.rgb_vae.in_channels, total_frames, cfg.data.resolution, cfg.data.resolution, device=device)
    depth_sample = torch.randn(batch, cfg.depth_vae.in_channels, total_frames, cfg.data.resolution, cfg.data.resolution, device=device)
    rgb_teacher.extract(rgb_sample[:, :, cfg.data.cond_frames:])
    depth_teacher.extract(depth_sample[:, :, cfg.data.cond_frames:])

    generator = ReLaTMoT(
        input_size=cfg.generator.mot.input_size,
        in_channels=cfg.generator.mot.in_channels,
        hidden_size=cfg.generator.mot.hidden_size,
        depth=cfg.generator.mot.depth,
        num_heads=cfg.generator.mot.num_heads,
        frames=cfg.rgb_vae.frames,
        aligned_depth=cfg.generator.mot.aligned_depth,
        rgb_teacher_dim=rgb_teacher.output_dim,
        depth_teacher_dim=depth_teacher.output_dim,
        use_qknorm=cfg.generator.mot.use_qknorm,
        use_swiglu=cfg.generator.mot.use_swiglu,
        use_rmsnorm=cfg.generator.mot.use_rmsnorm,
        wo_shift=cfg.generator.mot.wo_shift,
        fused_attn=cfg.generator.mot.fused_attn,
        use_rope=cfg.generator.mot.use_rope,
        same_noise=cfg.generator.mot.same_noise,
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

    rgb_past = rgb_sample[:, :, :cfg.data.cond_frames]
    depth_past = depth_sample[:, :, :cfg.data.cond_frames]
    rgb_future = rgb_sample[:, :, cfg.data.cond_frames:]
    depth_future = depth_sample[:, :, cfg.data.cond_frames:]
    rgb_cond = rgb_vae(x=rgb_past, mode="extract", normalize=True, conditioning=True)
    depth_cond = depth_vae(x=depth_past, mode="extract", normalize=True, conditioning=True)
    mask = condition_mask(batch, "full", device)
    rgb_cond = rgb_cond * mask[:, 0].view(batch, 1, 1)
    depth_cond = depth_cond * mask[:, 1].view(batch, 1, 1)
    rgb_future_latent = rgb_vae(x=rgb_future, mode="extract", normalize=True, conditioning=False)
    depth_future_latent = depth_vae(x=depth_future, mode="extract", normalize=True, conditioning=False)

    teacher_rgb = rgb_teacher.extract(rgb_future)
    teacher_depth = depth_teacher.extract(depth_future)
    flow_tuple = flow.sample_training_tuple(rgb_future_latent, depth_future_latent)

    for module in (rgb_vae, depth_vae, generator):
        module.zero_grad(set_to_none=True)
    relation_outputs = generator(
        flow_tuple["z_rgb_t"],
        flow_tuple["z_depth_t"],
        cond_rgb=rgb_cond,
        cond_depth=depth_cond,
        t_rgb=flow_tuple["scaled_t_rgb"],
        t_depth=flow_tuple["scaled_t_depth"],
        return_features=True,
    )
    relation_total = relation_loss(relation_outputs["aligned_rgb"], teacher_rgb)[0] + relation_loss(relation_outputs["aligned_depth"], teacher_depth)[0]
    relation_total.backward()
    relation_vae_grad = grad_norm(list(rgb_vae.parameters()) + list(depth_vae.parameters()))
    relation_gen_grad = grad_norm(generator.parameters())

    for module in (rgb_vae, depth_vae, generator):
        module.zero_grad(set_to_none=True)
    flow_outputs = generator(
        flow_tuple["z_rgb_t"].detach(),
        flow_tuple["z_depth_t"].detach(),
        cond_rgb=rgb_cond.detach(),
        cond_depth=depth_cond.detach(),
        t_rgb=flow_tuple["scaled_t_rgb"],
        t_depth=flow_tuple["scaled_t_depth"],
        return_features=False,
    )
    flow_total = flow.compute_loss(flow_outputs, flow_tuple)[0]
    flow_total.backward()
    flow_vae_grad = grad_norm(list(rgb_vae.parameters()) + list(depth_vae.parameters()))
    flow_gen_grad = grad_norm(generator.parameters())

    print(
        {
            "relation_vae_grad": relation_vae_grad,
            "relation_gen_grad": relation_gen_grad,
            "flow_vae_grad": flow_vae_grad,
            "flow_gen_grad": flow_gen_grad,
            "rgb_hidden_shape": tuple(relation_outputs["aligned_rgb"].shape),
            "depth_hidden_shape": tuple(relation_outputs["aligned_depth"].shape),
        }
    )


if __name__ == "__main__":
    main()
