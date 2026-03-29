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

If 24GB GPUs still run out of memory, enable `--save_memory` or set `optim.save_memory: true` in the config. This turns on block checkpointing in the MoT generator, chunks teacher feature extraction, and releases intermediate tensors earlier during the three training stages.

Generator scaling presets are:

- `tiny`: `hidden_size=384`, `depth=6`, `num_heads=6`
- `small`: `hidden_size=512`, `depth=8`, `num_heads=8`
- `base`: `hidden_size=768`, `depth=12`, `num_heads=12`
- `large`: `hidden_size=1024`, `depth=16`, `num_heads=16`

Evaluation can infer the preset scale from the checkpoint when it matches one of the built-in presets. Use `--model_scale custom` only when you intentionally manage the generator dimensions yourself.
