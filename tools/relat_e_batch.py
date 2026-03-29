from __future__ import annotations

import random

import torch


CONDITION_MODES = ("full", "rgb_only", "depth_only")


def sample_condition_mode(probs, rng=None):
    rng = random if rng is None else rng
    full = float(probs.get("full", 0.0))
    rgb_only = float(probs.get("rgb_only", 0.0))
    depth_only = float(probs.get("depth_only", 0.0))
    total = full + rgb_only + depth_only
    if total <= 0:
        return "full"
    draw = rng.random() * total
    if draw < full:
        return "full"
    if draw < full + rgb_only:
        return "rgb_only"
    return "depth_only"


def condition_mask(batch_size, mode, device):
    mask = torch.ones(batch_size, 2, device=device)
    if mode == "rgb_only":
        mask[:, 1] = 0.0
    elif mode == "depth_only":
        mask[:, 0] = 0.0
    elif mode != "full":
        raise ValueError(f"Unknown conditioning mode: {mode}")
    return mask


def apply_condition_mask(cond_rgb, cond_depth, mask):
    rgb_mask = mask[:, 0].view(-1, 1, 1)
    depth_mask = mask[:, 1].view(-1, 1, 1)
    return cond_rgb * rgb_mask, cond_depth * depth_mask
