from __future__ import annotations

import os
import math

from torch.utils.data import DataLoader

from tools.data_utils import InfiniteSampler
from tools.relat_e_datasets import PairedVideoFramesDataset


def _dataset_roots(args):
    dataset_name = args.data.name
    if dataset_name == "CITYSCAPES_RGBD":
        dataset_root = os.path.join(args.data.data_folder, "CITYSCAPES")
        return os.path.join(dataset_root, "rgb"), os.path.join(dataset_root, "depth"), "cityscapes_rgbd", args.data.video_len
    if dataset_name == "OpenDV_RGBD":
        dataset_root = os.path.join(args.data.data_folder, "OpenDV")
        return os.path.join(dataset_root, "rgb"), os.path.join(dataset_root, "depth"), "opendv_rgbd", args.data.video_len
    raise NotImplementedError(f"Unsupported multimodal dataset: {dataset_name}")


def get_relat_e_loaders(rank, args):
    rgb_root, depth_root, dataset_name, video_len = _dataset_roots(args)
    clip_frames = getattr(args.data, "clip_frames", args.data.cond_frames + args.data.pred_frames)
    common_kwargs = dict(
        resolution=args.data.resolution,
        video_len=video_len,
        n_frames=clip_frames,
        depth_channels=args.data.depth_in_channels,
        max_size=args.data.max_size,
        seed=args.experiment.seed,
        dataset_name=dataset_name,
    )
    trainset = PairedVideoFramesDataset(rgb_root, depth_root, split="train", **common_kwargs)

    try:
        valset = PairedVideoFramesDataset(rgb_root, depth_root, split="val", **common_kwargs)
    except Exception:
        valset = None

    testset = PairedVideoFramesDataset(rgb_root, depth_root, split="test", **common_kwargs)

    world_size = max(int(args.experiment.n_gpus), 1)
    per_rank_batch = max(1, math.ceil(int(args.data.batch_size) / world_size))

    train_sampler = InfiniteSampler(dataset=trainset, rank=rank, num_replicas=world_size, seed=args.experiment.seed)
    trainloader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=per_rank_batch,
        pin_memory=False,
        num_workers=args.data.num_workers,
    )

    if valset is not None:
        validationloader = DataLoader(
            valset,
            batch_size=per_rank_batch,
            pin_memory=False,
            num_workers=args.data.num_workers,
            shuffle=False,
        )
    else:
        validationloader = DataLoader(
            testset,
            batch_size=per_rank_batch,
            pin_memory=False,
            num_workers=args.data.num_workers,
            shuffle=False,
        )

    testloader = DataLoader(
        testset,
        batch_size=per_rank_batch,
        pin_memory=False,
        num_workers=args.data.num_workers,
        shuffle=False,
    )
    return trainloader, validationloader, testloader
