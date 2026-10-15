#!/bin/bash
# Pretrain on 67 top-ladder submissions (run pre_deep67 -> keep e0).
set -e
cd "$(dirname "$0")/../experiments/net"
export FARMCORE_TEACHER=$(cat ../../pipeline/teachers/pretrain67.txt)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
python -u bc_train.py --run pre_deep67 --epochs 2 --save-from 0 --batch 512 --warmup 500 \
  --device cuda --amp bf16 --gpu-corpus 1 -j 14 --seed 0 --val-frac 0.01 --refit-all \
  --width-mult 2 --depth-mult 3 --opt muon --wd 0.01 --dropout 0.1 --chunk-games 1500 \
  --win-w 1.5 --aug-shed 0.3 --aug-shed-max 40 --aug-shed-other 0
