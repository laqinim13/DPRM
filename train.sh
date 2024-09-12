#!/usr/bin/env bash
# OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# 高精度模型：HRNet
python train.py  -c "configs/coco/hrnet/w32_adam_hand_192_192.yaml" --workers=8 --batch_size=32 --gpu_id=0 --work_dir="work_space/hand"
python train.py  -c "configs/coco/hrnet/w48_adam_hand_192_192.yaml" --workers=8 --batch_size=32 --gpu_id=0 --work_dir="work_space/hand"

# 轻量化模型：LiteHRNet
#python train.py  -c "configs/coco/litehrnet/litehrnet30_hand_192_192.yaml" --workers=8 --batch_size=32 --gpu_id=0 --work_dir="work_space/hand"
#python train.py  -c "configs/coco/litehrnet/litehrnet18_hand_192_192.yaml" --workers=8 --batch_size=32 --gpu_id=0 --work_dir="work_space/hand"

# 轻量化模型：Mobilenet-v2
#python train.py  -c "configs/coco/mobilenet/mobilenetv2_hand_192_192.yaml" --workers=8 --batch_size=32 --gpu_id=0 --work_dir="work_space/hand"
