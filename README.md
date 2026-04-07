# ReLaT-E

## ⚙️ Installation

It is recommended to use Anaconda to create a virtual environment.

### 1. Create Environment

```bash
conda create -n ule python=3.12 -y
conda activate ule

pip install -r requirements.txt
```

## ReLaT-E Entry Points

This repository now exposes only the multimodal ReLaT-E path:

```bash
python main_relat_e.py --config configs/relat_e_rgbd.yaml
python main_relat_e.py --config configs/relat_e_rgbd.yaml --save_memory
python main_relat_e.py --config configs/relat_e_rgbd.yaml --model_scale tiny
python main_relat_e.py --config configs/relat_e_rgbd.yaml --model_scale small
python main_relat_e.py --config configs/relat_e_rgbd.yaml --model_scale base
python main_relat_e.py --config configs/relat_e_rgbd.yaml --model_scale large
python eval_relat_e.py --config configs/relat_e_rgbd.yaml --ckpt /path/to/relat_e_checkpoint.pt
python eval_relat_e.py --config configs/relat_e_rgbd.yaml --ckpt /path/to/relat_e_checkpoint.pt --model_scale large
python tools/relat_e_smoke.py --config configs/relat_e_rgbd.yaml
```

The default ReLaT-E config uses a paired RGB-D batch contract, a dual-teacher registry, and a dedicated MoT-based generator path.

The default teacher setup is:

- RGB teacher: VideoMAEv2 ViT-Base distilled from the giant checkpoint, loaded from `./ckpts/VideoMAEv2/distill/vit_b_k710_dl_from_giant.pth`
- Depth teacher: Video Depth Anything ViT-Base, loaded from `./external/Video-Depth-Anything/checkpoints/video_depth_anything_vitb.pth`

Clone the official Video Depth Anything repository under `./external/Video-Depth-Anything` or update `teachers.depth.repo_root` in [configs/relat_e_rgbd.yaml](C:/Users/wangsen/OneDrive/Desktop/ReLaT-E/configs/relat_e_rgbd.yaml).

If 24GB GPUs still run out of memory, enable `--save_memory` or set `optim.save_memory: true` in the config. This turns on block checkpointing in the MoT generator, chunks teacher feature extraction, and releases intermediate tensors earlier during the three training stages.

Generator scaling presets are:

- `tiny`: `hidden_size=384`, `depth=6`, `num_heads=6`
- `small`: `hidden_size=512`, `depth=8`, `num_heads=8`
- `base`: `hidden_size=768`, `depth=12`, `num_heads=12`
- `large`: `hidden_size=1024`, `depth=16`, `num_heads=16`

Set `generator.mot.depth_width_ratio` to make the depth stream narrower than the RGB stream. For example, `depth_width_ratio: 4` keeps RGB at `hidden_size` and uses `hidden_size / 4` for the depth stream internals.

Evaluation can infer the preset scale from the checkpoint when it matches one of the built-in presets. Use `--model_scale custom` only when you intentionally manage the generator dimensions yourself.
