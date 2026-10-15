#!/bin/bash
# Soup the two fine-tunes, build the Kaggle package and check it against an in-process run.
set -e
cd "$(dirname "$0")/.."
R=experiments/net/runs
python tools/soup.py $R/deep_mm/e7.pt $R/deep_mm6/e7.pt --out $R/S_DEEP.pt
python tools/submission/build.py --ckpt $R/S_DEEP.pt --out submission --end-sell \
  --night-room --night-room-mode value
# the package weights carry the decoder flags, so validate against them
python tools/submission/validate.py --ckpt submission/weights.pt --pkg submission -n 3
