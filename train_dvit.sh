#!/bin/bash
# FLOPS
## 0.7: 3.0
## 0.8: 3.5
## 0.9: 4.0
IMNET=/opt/ssd2/ILSVRC2012
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env ./main_dynamic_vit.py  --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 96 --data-path $IMNET --epochs 30 --dist-eval --distill --base_rate 0.9
