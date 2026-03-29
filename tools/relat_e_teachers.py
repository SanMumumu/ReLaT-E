from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


class BaseTokenTeacher(nn.Module):
    output_dim = None

    def extract(self, video):
        raise NotImplementedError

    def forward(self, video):
        return self.extract(video)


class PatchTokenTeacher(BaseTokenTeacher):
    def __init__(self, patch_size=16, normalize="imagenet"):
        super().__init__()
        self.patch_size = int(patch_size)
        self.normalize = normalize
        self._normalizers = {
            "imagenet": Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        }
        self.output_dim = None

    def _normalize_video(self, video):
        if self.normalize == "depth":
            flat = video.transpose(1, 2).flatten(0, 1)
            mean = flat.mean(dim=(2, 3), keepdim=True)
            std = flat.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
            flat = (flat - mean) / std
            return flat
        if self.normalize in self._normalizers:
            flat = ((video + 1.0) / 2.0).transpose(1, 2).flatten(0, 1)
            return self._normalizers[self.normalize](flat)
        return video.transpose(1, 2).flatten(0, 1)

    def extract(self, video):
        b, c, t, h, w = video.shape
        frames = self._normalize_video(video)
        target_h = max(self.patch_size, h - (h % self.patch_size))
        target_w = max(self.patch_size, w - (w % self.patch_size))
        if target_h != h or target_w != w:
            frames = F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)
        patches = F.unfold(frames, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        self.output_dim = patches.size(-1)
        return patches.reshape(b, t * patches.size(1), patches.size(-1))


class WrappedVideoTeacher(BaseTokenTeacher):
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model.eval()
        self.model_name = model_name
        self.output_dim = getattr(model, "embed_dim", None)
        for param in self.model.parameters():
            param.requires_grad = False

    def extract(self, video):
        b, _, f, _, _ = video.shape
        frames_01 = (video + 1.0) / 2.0
        flat = frames_01.transpose(1, 2).flatten(0, 1)
        flat = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(flat)
        prepared = flat.reshape(b, f, flat.size(1), flat.size(2), flat.size(3)).transpose(1, 2)
        with torch.no_grad():
            if self.model_name in {"VideoMAEv2", "VideoMAE", "OminiMAE", "VJEPA", "VJEPA2"}:
                features = self.model(prepared)
            elif self.model_name in {"DINOv3"}:
                frames_2d = prepared.transpose(1, 2).flatten(0, 1)
                out = self.model.forward_features(frames_2d)
                tokens = out["x_norm_patchtokens"]
                features = tokens.reshape(b, f * tokens.size(1), tokens.size(2))
            else:
                raise NotImplementedError(f"Unsupported wrapped teacher: {self.model_name}")
        if self.output_dim is None:
            self.output_dim = features.size(-1)
        return features


def _load_rgb_backbone(name, ckpt_dir, device):
    if name == "VideoMAEv2":
        from models.ssl.videomaev2 import vit_base_patch16_224

        model = vit_base_patch16_224().to(device)
        model.from_pretrained(os.path.join(ckpt_dir, "VideoMAEv2/vit_b_k710_dl_from_giant.pth"))
        return model
    if name == "VideoMAE":
        from models.ssl.videomae import vit_base_patch16_224

        model = vit_base_patch16_224().to(device)
        model.from_pretrained(
            os.path.join(ckpt_dir, "VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth")
        )
        return model
    if name == "OminiMAE":
        from models.ssl.omini_mae import vit_base_mae_pretraining

        return vit_base_mae_pretraining().to(device)
    if name == "VJEPA":
        from models.ssl.JEPA import load_VJEPA

        return load_VJEPA(device=device, pretrained_path=os.path.join(ckpt_dir, "vjepa_l/vitl16.pth.tar"))
    if name == "VJEPA2":
        model, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
        if hasattr(model, "norm"):
            model.norm = nn.Identity()
        return model.to(device)
    if name == "DINOv3":
        model = torch.hub.load("facebookresearch/dinov3", "dinov3_vits16").to(device)
        if hasattr(model, "head"):
            model.head = nn.Identity()
        return model
    raise NotImplementedError(f"Unsupported teacher backbone: {name}")


def create_teacher(cfg, device):
    name = cfg.name
    if name == "patch_tokens":
        teacher = PatchTokenTeacher(patch_size=cfg.patch_size, normalize=cfg.normalize)
    else:
        model = _load_rgb_backbone(name, cfg.ckpt_dir, device)
        teacher = WrappedVideoTeacher(model, name)
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher
