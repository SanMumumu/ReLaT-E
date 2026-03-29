from __future__ import annotations

import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from tools.data_utils import resize_crop


def _frame_paths(root, split):
    split_root = os.path.join(root, split)
    if not os.path.exists(split_root):
        raise FileNotFoundError(f"{split_root} does not exist")
    return split_root


class PairedVideoFramesDataset(Dataset):
    def __init__(
        self,
        rgb_root,
        depth_root,
        split,
        resolution,
        video_len,
        n_frames,
        depth_channels=3,
        max_size=None,
        seed=42,
        dataset_name="paired",
    ):
        self.rgb_root = _frame_paths(rgb_root, split)
        self.depth_root = _frame_paths(depth_root, split)
        self.split = split
        self.resolution = resolution
        self.video_len = video_len
        self.n_frames = n_frames
        self.depth_channels = depth_channels
        self.dataset_name = dataset_name
        self.seed = seed
        self.clips = self._discover_clips()
        if max_size is not None:
            self.clips = self.clips[:max_size]

    def _discover_clips(self):
        rgb_dirs = {
            os.path.basename(path): path
            for path in glob(os.path.join(self.rgb_root, "*"))
            if os.path.isdir(path)
        }
        depth_dirs = {
            os.path.basename(path): path
            for path in glob(os.path.join(self.depth_root, "*"))
            if os.path.isdir(path)
        }
        clip_keys = sorted(set(rgb_dirs.keys()) & set(depth_dirs.keys()))
        if not clip_keys:
            raise RuntimeError(f"No paired clips found under {self.rgb_root} and {self.depth_root}.")

        clips = []
        for key in clip_keys:
            rgb_frames = sorted(glob(os.path.join(rgb_dirs[key], "*.png")) + glob(os.path.join(rgb_dirs[key], "*.jpg")))
            depth_frames = sorted(glob(os.path.join(depth_dirs[key], "*.png")) + glob(os.path.join(depth_dirs[key], "*.jpg")))
            frame_count = min(len(rgb_frames), len(depth_frames), self.video_len)
            if frame_count < self.n_frames:
                continue
            clips.append(
                {
                    "key": key,
                    "rgb_frames": rgb_frames[:frame_count],
                    "depth_frames": depth_frames[:frame_count],
                }
            )
        if not clips:
            raise RuntimeError(f"No clips with at least {self.n_frames} paired frames were found.")
        return clips

    def __len__(self):
        return len(self.clips)

    def _load_rgb_clip(self, frame_paths):
        frames = np.stack([cv2.imread(path, cv2.IMREAD_COLOR) for path in frame_paths], axis=0)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
        frames = frames[[2, 1, 0]]
        return resize_crop(frames, self.resolution)

    def _load_depth_clip(self, frame_paths):
        frames = []
        for path in frame_paths:
            frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise RuntimeError(f"Failed to read depth frame: {path}")
            if frame.ndim == 2:
                frame = frame[..., None]
            if self.depth_channels == 1:
                frame = frame[..., :1]
            elif self.depth_channels == 3 and frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif self.depth_channels == 3 and frame.shape[-1] >= 3:
                frame = frame[..., :3]
            else:
                frame = np.repeat(frame[..., :1], self.depth_channels, axis=-1)
            frames.append(frame)
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
        if frames.size(0) == 3:
            frames = frames[[2, 1, 0]]
        return resize_crop(frames, self.resolution)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        max_start = len(clip["rgb_frames"]) - self.n_frames
        if self.split == "train":
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = max_start // 2
        rgb_paths = clip["rgb_frames"][start:start + self.n_frames]
        depth_paths = clip["depth_frames"][start:start + self.n_frames]
        rgb = self._load_rgb_clip(rgb_paths)
        depth = self._load_depth_clip(depth_paths)
        return {
            "rgb": rgb,
            "depth": depth,
            "modality_mask": torch.ones(2, dtype=torch.float32),
            "meta": {
                "dataset": self.dataset_name,
                "index": idx,
                "clip_key": clip["key"],
            },
        }
