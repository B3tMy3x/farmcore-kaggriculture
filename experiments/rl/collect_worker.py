"""A rollout chunk as a separate process. Launched only by `collect_parallel`.

This module has NO logic of its own: the worker loads the same weights with the same
`collect.load_net`, calls the same `collect.collect_serial` and writes its
result to a file. Otherwise "parallel" would become a second way to play a game --
exactly the class of bug that bit this repo three times ("training and play must
compute ONE quantity").

The worker does NOT compute the critic estimate: weights do not change between rollout
games, so `V(s)` is just as correct computed after collection (`collect.score`), and
the value tower is not needed by the worker at all.

Progress is printed one line per game with `flush=True` (repo rule); the parent
process reads the output as a stream.
"""
from __future__ import annotations

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import torch  # noqa: E402

# ⚠ Single thread -- both for the cores (otherwise J processes each schedule over all
# cores) and for PARITY: the thread count changes the reduction order in CPU matrix
# products, and with it the last bits of the logits.
torch.set_num_threads(1)

import collect as COL  # noqa: E402


def main(task_path: str) -> int:
    with open(task_path, "rb") as f:
        task = pickle.load(f)
    cfg = task["cfg"]
    net = COL.load_net(task["weights"], cfg.rollout.device)
    # One call for the whole chunk, not per game: `collect_serial` keeps a cache
    # of opponents (checkpoint net, loaded package), and per game it would be
    # reloaded every time.
    eps = COL.collect_serial(net, task["plan"], cfg, None, progress=False,
                             offset=task["offset"], tag="__PROG__ ")
    with open(task["out"], "wb") as f:
        pickle.dump(eps, f, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
