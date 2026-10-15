"""Uniform weight soup of fine-tunes that share one pretrain.

    python tools/soup.py runs/deep_mm/e7.pt runs/deep_mm6/e7.pt --out S_DEEP.pt

Floating-point tensors of `model` are averaged; integer buffers must match
exactly. Everything else (`model_cfg`, `prior`, `cfg`) is taken from the FIRST
parent: this is how the final S_DEEP was built (its weights equal the mean of
deep_mm e7 and deep_mm6 e7 to the bit, its `prior` is deep_mm's).

Souping only paid off for parents that start from the same pretrain and are
fine-tuned on the same team; seeds of one recipe and fine-tunes on different
teams did not soup (see the writeup).
"""
from __future__ import annotations

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parents", nargs="+", help="checkpoints to average (the first one supplies cfg/prior)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ws = [torch.load(p, map_location="cpu", weights_only=False) for p in a.parents]
    base = ws[0]
    for p, w in zip(a.parents[1:], ws[1:]):
        if w["model_cfg"] != base["model_cfg"]:
            raise SystemExit(f"{p}: model_cfg differs from {a.parents[0]}")
        if w["model"].keys() != base["model"].keys():
            raise SystemExit(f"{p}: state_dict keys differ from {a.parents[0]}")
    soup = {}
    for k, v in base["model"].items():
        if v.is_floating_point():
            soup[k] = sum(w["model"][k].double() for w in ws).div(len(ws)).to(v.dtype)
        else:
            for p, w in zip(a.parents[1:], ws[1:]):
                if not torch.equal(w["model"][k], v):
                    raise SystemExit(f"{p}: non-float tensor {k} differs")
            soup[k] = v.clone()
    out = {k: base[k] for k in ("model_cfg", "prior", "cfg") if k in base}
    out["model"] = soup
    out["soup_of"] = list(a.parents)
    torch.save(out, a.out)
    print(f"soup of {len(ws)} -> {a.out}")


if __name__ == "__main__":
    main()
