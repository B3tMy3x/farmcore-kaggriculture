"""Epoch selection by MONEY, not by validation.

Why. Across eight versions v15-v22 (same protocol, one teacher) the relation of
validation to money is NEGATIVE:

    whole turn                  Spearman -0.86  (p = 0.006)
    target accuracy             Spearman -0.81  (p = 0.015)
    0.5*(op_acc+tgt_acc)        Spearman -0.69  (p = 0.058)   <- used to select best.pt

So the old selection metric systematically picked the worst epoch among the good ones.
Here the epoch is selected by a short self-play rollout on fixed seeds:
the checkpoint plays itself, the score is the median of ITS money. The measurement
is deterministic (the generator is fixed per game), so 12-24 games suffice.

Cannot measure against `starter`: it takes $3.5k and does not touch the market, the gain there
is overstated threefold.

Run (checkpoints must be local):
    ./env/bin/python tools/bench/select_epoch.py f1_coord -n 16
    ./env/bin/python tools/bench/select_epoch.py f1_coord --ref runs/v23_place/best.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAY = os.path.join(ROOT, "experiments/net/play.py")


def epochs_of(run_dir):
    out = []
    for f in glob.glob(os.path.join(run_dir, "e*.pt")):
        m = re.fullmatch(r"e(\d+)\.pt", os.path.basename(f))
        if m:
            out.append((int(m.group(1)), f))
    return [f for _, f in sorted(out)]


def play(ckpt, opp, n, jobs, seed0, extra):
    cmd = [sys.executable, "-u", PLAY, "-n", str(n), "-j", str(jobs), "--ckpt", ckpt,
           "--opp-ckpt", opp, "--device", "cpu", "--seed0", str(seed0),
           "--dump", os.path.join("/tmp", "sel_" + os.path.basename(ckpt) + ".json")]
    cmd += extra
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    # Read play.py output as a STREAM, not via capture_output: an epoch takes
    # minutes, and `subprocess.run` would hand over its progress in one piece at the end --
    # exactly the 45-minute silence that got progress added to play.py.
    tag = os.path.basename(ckpt)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env,
                            cwd=ROOT, bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        del tail[:-40]
        if line:
            print(f"  [{tag}] {line}", flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"{tag}: play.py crashed\n" + "\n".join(tail))
    dump = json.load(open(cmd[cmd.index("--dump") + 1]))
    g = np.array(dump["games"])
    dead = int((g[:, 2] > 0).sum())
    ok = g[g[:, 2] == 0]
    if len(ok) == 0:
        return None
    return {"money": float(np.median(ok[:, 0])), "opp": float(np.median(ok[:, 1])),
            "wins": int((ok[:, 0] > ok[:, 1]).sum()), "n": len(ok), "dead": dead,
            "p10": float(np.percentile(ok[:, 0], 10))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name in experiments/net/runs/ or a path to the folder")
    ap.add_argument("-n", type=int, default=16, help="games per epoch")
    ap.add_argument("-j", "--jobs", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--ref", default="",
                    help="play against a FIXED checkpoint instead of "
                         "itself. Then the score is the win rate, not the bank median.")
    ap.add_argument("--epochs", default="", help="comma-separated, otherwise all")
    ap.add_argument("--dry", action="store_true", help="do not overwrite best.pt")
    a, extra = ap.parse_known_args()

    run_dir = a.run if os.path.isdir(a.run) else os.path.join(
        ROOT, "experiments/net/runs", a.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"no run folder: {run_dir}")
    files = epochs_of(run_dir)
    if a.epochs:
        want = {int(x) for x in a.epochs.split(",")}
        files = [f for f in files
                 if int(re.fullmatch(r"e(\d+)\.pt", os.path.basename(f)).group(1)) in want]
    if not files:
        raise SystemExit(f"no e*.pt in {run_dir}")
    ref = a.ref
    if ref and not os.path.isabs(ref):
        ref = os.path.join(ROOT, ref)

    print(f"{run_dir}: epochs {len(files)}, {a.n} games each "
          f"{'against ' + os.path.basename(os.path.dirname(ref)) if ref else 'self-play'}")
    rows = []
    for f in files:
        r = play(f, ref or f, a.n, a.jobs, a.seed0, extra)
        if r is None:
            print(f"  {os.path.basename(f):>8}  all games dead")
            continue
        key = r["wins"] / r["n"] if ref else r["money"]
        rows.append((key, f, r))
        mark = "  ⚠ DROPPED " + str(r["dead"]) if r["dead"] else ""
        print(f"  {os.path.basename(f):>8}  median ${r['money']:>9,.0f}  "
              f"p10 ${r['p10']:>9,.0f}  wins {r['wins']:>2}/{r['n']}{mark}", flush=True)
    if not rows:
        raise SystemExit("nothing to choose from")
    rows.sort(key=lambda t: -t[0])
    best = rows[0]
    print(f"\nbest epoch: {os.path.basename(best[1])}  "
          f"{'win rate ' + format(best[0], '.3f') if ref else '$' + format(best[0], ',.0f')}")
    if not a.dry:
        dst = os.path.join(run_dir, "best.pt")
        shutil.copyfile(best[1], dst)
        print(f"copied to {dst}")
        val = os.path.join(run_dir, "best_val.pt")
        if os.path.exists(val):
            print("  (best_val.pt — selection by the old metric, kept for comparison)")


if __name__ == "__main__":
    main()
