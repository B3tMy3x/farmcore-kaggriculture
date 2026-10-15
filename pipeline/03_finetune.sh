#!/bin/bash
# Two fine-tunes on M & M & P & Q from the same pretrain; e7 of each goes into the soup.
set -e
cd "$(dirname "$0")/../experiments/net"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
COMMON="--batch 512 --device cuda --amp bf16 --gpu-corpus 1 --seed 0 --val-frac 0.05 --refit-all
  --width-mult 2 --depth-mult 3 --opt muon --wd 0.01 --win-w 1.5
  --aug-shed 0.3 --aug-shed-max 40 --aug-shed-other 0 --init runs/pre_deep67/e0.pt"

# deep_mm: corpus = the pretrain cache, restricted to 4 M&M submissions with --ft-subs
FARMCORE_TEACHER=$(cat ../../pipeline/teachers/pretrain67.txt) \
python -u bc_train.py --run deep_mm --epochs 12 --save-from 6 -j 4 \
  --ft-subs $(cat ../../pipeline/teachers/ft_mm4.txt) $COMMON

# deep_mm6: corpus = 6 M&M submissions only
FARMCORE_TEACHER=$(cat ../../pipeline/teachers/ft_mm6.txt) \
python -u bc_train.py --run deep_mm6 --epochs 8 --save-from 4 -j 2 $COMMON
