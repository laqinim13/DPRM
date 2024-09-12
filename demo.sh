#!/usr/bin/env bash

# 测试图片
python demo.py \
  -c "work_space/hand/hrnet_w32_21_192_192_custom_coco_20231007_083128_2043/w32_adam_hand_192_192.yaml"  \
  -m "work_space/hand/hrnet_w32_21_192_192_custom_coco_20231007_083128_2043/model/best_model_189_0.8570.pth" \
  --target "hand" \
  --image_dir "data/hand" \
  --out_dir "output"

# 测试视频
python demo.py \
  -c "work_space/hand/hrnet_w32_21_192_192_custom_coco_20231007_083128_2043/w32_adam_hand_192_192.yaml"  \
  -m "work_space/hand/hrnet_w32_21_192_192_custom_coco_20231007_083128_2043/model/best_model_189_0.8570.pth" \
  --target "hand" \
  --video_file "data/hand/test-video.mp4" \
  --out_dir "output"


# 测试摄像头
python demo.py \
  -c "work_space/hand/hrnet_w32_21_192_192_custom_coco_20231007_083128_2043/w32_adam_hand_192_192.yaml"  \
  -m "work_space/hand/hrnet_w32_21_192_192_custom_coco_20231007_083128_2043/model/best_model_189_0.8570.pth" \
  --target "hand" \
  --video_file 0 \
  --out_dir "output"
