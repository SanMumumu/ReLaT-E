from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from losses.relat_flow import RelatEFlowMatching
from models.fm.relat_mot import ReLaTMoT
from models.vae.relat_autoencoder import Relat3DVAE
from tools.relat_e_dataloader import get_relat_e_loaders
from tools.relat_e_scaling import MODEL_SCALE_PRESETS, apply_model_scale, infer_model_scale_from_state_dict
from tools.relat_e_utils import load_relat_e_checkpoint, run_relat_e_evaluation


def _safe_torch_load(path, map_location="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/relat_e_rgbd.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--model_scale", type=str, default="", choices=sorted(list(MODEL_SCALE_PRESETS.keys()) + ["custom"]))
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    ckpt_meta = _safe_torch_load(args.ckpt, map_location="cpu", weights_only=True)
    generator_state = ckpt_meta.get("ema_generator", ckpt_meta["generator_model"])
    resolved_scale = args.model_scale or infer_model_scale_from_state_dict(generator_state)
    apply_model_scale(cfg, resolved_scale)
    if args.eval_samples > 0:
        cfg.eval.eval_samples = args.eval_samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_teacher_dim = generator_state["rgb_relation_head.net.2.weight"].shape[0]
    depth_teacher_dim = generator_state["depth_relation_head.net.2.weight"].shape[0]

    rgb_vae = Relat3DVAE(cfg.rgb_vae.embed_dim, cfg.rgb_vae, bn_momentum=cfg.rgb_vae.bn_momentum).to(device)
    depth_vae = Relat3DVAE(cfg.depth_vae.embed_dim, cfg.depth_vae, bn_momentum=cfg.depth_vae.bn_momentum).to(device)
    if int(cfg.generator.mot.in_channels) != int(cfg.rgb_vae.embed_dim) or int(cfg.generator.mot.in_channels) != int(cfg.depth_vae.embed_dim):
        raise ValueError("generator.mot.in_channels must match both VAE embed_dim values.")
    if rgb_vae.latent_size != depth_vae.latent_size or rgb_vae.latent_frames != depth_vae.latent_frames:
        raise ValueError("RGB and depth VAE latent grids must match for evaluation.")
    cfg.generator.mot.input_size = rgb_vae.latent_size
    generator = ReLaTMoT(
        input_size=rgb_vae.latent_size,
        in_channels=cfg.generator.mot.in_channels,
        hidden_size=cfg.generator.mot.hidden_size,
        depth=cfg.generator.mot.depth,
        num_heads=cfg.generator.mot.num_heads,
        frames=rgb_vae.latent_frames,
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
        depth_width_ratio=getattr(cfg.generator.mot, "depth_width_ratio", 1),
    ).to(device)
    load_relat_e_checkpoint(args.ckpt, rgb_vae, depth_vae, generator, map_location=device, use_ema=True)

    flow = RelatEFlowMatching(
        sampling_timesteps=cfg.eval.sampling_timesteps,
        sigma_min=cfg.loss.flow.sigma_min,
        same_noise=cfg.generator.mot.same_noise,
    ).to(device)
    cfg.data.clip_frames = int(cfg.data.cond_frames + max(cfg.data.pred_frames, cfg.eval.future_frames))
    _, val_loader, _ = get_relat_e_loaders(0, cfg)

    def log_(message):
        print(message)

    run_relat_e_evaluation(
        val_loader,
        rgb_vae,
        depth_vae,
        generator,
        flow,
        cfg,
        device,
        step=0,
        logger=None,
        log_=log_,
        rollout=True,
    )


if __name__ == "__main__":
    main()
