from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


def _safe_torch_load(path, map_location="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ensure_repo_import(repo_root):
    if not repo_root:
        return
    repo_path = str(Path(repo_root).expanduser().resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


class BaseTeacher(nn.Module):
    output_dim = None

    def extract(self, video):
        raise NotImplementedError

    def forward(self, video):
        return self.extract(video)


class VideoMAEv2RGBTeacher(BaseTeacher):
    def __init__(self, checkpoint, input_size=224, num_frames=8, target_resolution=None, normalize="imagenet"):
        super().__init__()
        from models.ssl.videomaev2 import vit_base_patch16_224

        target_resolution = target_resolution or [input_size, input_size]
        self.model = vit_base_patch16_224(
            img_size=input_size,
            align_video_resolution=tuple(target_resolution),
            all_frames=num_frames,
        )
        self.model.from_pretrained(checkpoint)
        self.model.eval()
        self.output_dim = int(self.model.embed_dim)
        self.input_size = int(input_size)
        self.num_frames = int(num_frames)
        self.normalize = normalize
        self._imagenet_norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        for param in self.model.parameters():
            param.requires_grad = False

    def _prepare(self, video):
        b, c, t, _, _ = video.shape
        if c != 3:
            raise ValueError(f"VideoMAEv2 RGB teacher expects 3 channels, got {c}.")
        frames = ((video + 1.0) / 2.0).clamp(0.0, 1.0).transpose(1, 2).flatten(0, 1)
        frames = F.interpolate(frames, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        if self.normalize == "imagenet":
            frames = self._imagenet_norm(frames)
        return frames.reshape(b, t, frames.size(1), self.input_size, self.input_size).transpose(1, 2).contiguous()

    def extract(self, video):
        with torch.no_grad():
            return self.model(self._prepare(video))


class VideoDepthAnythingTeacher(BaseTeacher):
    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(self, repo_root, checkpoint, encoder="vitb", input_size=518, metric=False, feature_source="path3"):
        super().__init__()
        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported Video Depth Anything encoder: {encoder}")
        _ensure_repo_import(repo_root)
        try:
            module = importlib.import_module("video_depth_anything.video_depth")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Video Depth Anything is not importable. Set teachers.depth.repo_root to the cloned "
                "DepthAnything/Video-Depth-Anything repository."
            ) from exc

        config = dict(self.MODEL_CONFIGS[encoder])
        self.model = module.VideoDepthAnything(**config, metric=metric)
        state_dict = _safe_torch_load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.encoder = encoder
        self.input_size = int(round(input_size / 14) * 14)
        self.feature_source = feature_source
        self.output_dim = int(config["features"]) if feature_source != "prediction" else 1
        self._imagenet_norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        for param in self.model.parameters():
            param.requires_grad = False

    def _prepare(self, video):
        b, c, t, _, _ = video.shape
        frames = ((video + 1.0) / 2.0).clamp(0.0, 1.0).transpose(1, 2).flatten(0, 1)
        if c == 1:
            frames = frames.repeat(1, 3, 1, 1)
        elif c > 3:
            frames = frames[:, :3]
        frames = F.interpolate(frames, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        frames = self._imagenet_norm(frames)
        return frames.reshape(b, t, frames.size(1), self.input_size, self.input_size).contiguous()

    def _extract_prediction_tokens(self, frames):
        depth = self.model(frames)
        tokens = depth.reshape(depth.size(0), depth.size(1), -1, 1)
        return tokens.reshape(depth.size(0), depth.size(1) * tokens.size(2), 1)

    def _extract_path3_tokens(self, frames):
        b, t, _, h, w = frames.shape
        patch_h, patch_w = h // 14, w // 14
        out_features = self.model.pretrained.get_intermediate_layers(
            frames.flatten(0, 1),
            self.model.intermediate_layer_idx[self.model.encoder],
            return_class_token=True,
        )
        head = self.model.head
        out = []
        for index, feat in enumerate(out_features):
            if head.use_clstoken:
                tokens, cls_token = feat[0], feat[1]
                readout = cls_token.unsqueeze(1).expand_as(tokens)
                tokens = head.readout_projects[index](torch.cat((tokens, readout), dim=-1))
            else:
                tokens = feat[0]
            tokens = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w).contiguous()
            tokens = head.projects[index](tokens)
            tokens = head.resize_layers[index](tokens)
            out.append(tokens)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_3, _ = head.motion_modules[0](layer_3.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4), None, None, None)
        layer_3 = layer_3.permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_4, _ = head.motion_modules[1](layer_4.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4), None, None, None)
        layer_4 = layer_4.permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_1_rn = head.scratch.layer1_rn(layer_1)
        layer_2_rn = head.scratch.layer2_rn(layer_2)
        layer_3_rn = head.scratch.layer3_rn(layer_3)
        layer_4_rn = head.scratch.layer4_rn(layer_4)

        path_4 = head.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4, _ = head.motion_modules[2](path_4.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4), None, None, None)
        path_4 = path_4.permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = head.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3, _ = head.motion_modules[3](path_3.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4), None, None, None)
        path_3 = path_3.permute(0, 2, 1, 3, 4).flatten(0, 1)

        channels = path_3.size(1)
        tokens = path_3.reshape(b, t, channels, -1).permute(0, 1, 3, 2).contiguous()
        return tokens.reshape(b, t * tokens.size(2), channels)

    def extract(self, video):
        frames = self._prepare(video)
        with torch.no_grad():
            if self.feature_source == "prediction":
                return self._extract_prediction_tokens(frames)
            if self.feature_source == "path3":
                return self._extract_path3_tokens(frames)
            raise ValueError(f"Unsupported Video Depth Anything feature source: {self.feature_source}")


def create_teacher(cfg, device):
    name = str(cfg.name).lower()
    if name in {"videomaev2_distill_base", "videomaev2_rgb"}:
        teacher = VideoMAEv2RGBTeacher(
            checkpoint=cfg.checkpoint,
            input_size=int(getattr(cfg, "input_size", 224)),
            num_frames=int(getattr(cfg, "num_frames", 8)),
            target_resolution=getattr(cfg, "target_resolution", None),
            normalize=str(getattr(cfg, "normalize", "imagenet")),
        )
    elif name in {"video_depth_anything", "videodepthanything"}:
        teacher = VideoDepthAnythingTeacher(
            repo_root=str(getattr(cfg, "repo_root", "")),
            checkpoint=cfg.checkpoint,
            encoder=str(getattr(cfg, "encoder", "vitb")),
            input_size=int(getattr(cfg, "input_size", 518)),
            metric=bool(getattr(cfg, "metric", False)),
            feature_source=str(getattr(cfg, "feature_source", "path3")),
        )
    else:
        raise NotImplementedError(f"Unsupported teacher backbone: {cfg.name}")
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher
