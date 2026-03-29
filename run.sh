CUDA_VISIBLE_DEVICES=0 python main_relat_e.py \
    --config configs/relat_e_rgbd.yaml \
    --model_scale base

CUDA_VISIBLE_DEVICES=0 python main_relat_e.py \
    --config configs/relat_e_rgbd.yaml \
    --model_scale small \
    --save_memory

CUDA_VISIBLE_DEVICES=0 python eval_relat_e.py \
    --config configs/relat_e_rgbd.yaml \
    --ckpt /path/to/relat_e_checkpoint.pt \
    --model_scale base
