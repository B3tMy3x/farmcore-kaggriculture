"""Build a Kaggle submission: main.py + our modules + weights, as is.

Three peculiarities of how Kaggle runs the agent (`kaggle_environments/agent.py`):
  1. the file is executed via `exec(code, {})` -- **`__file__` is not defined**, so the path to
     the weights has to be found through `sys.path`;
  2. the agent is the **last callable in the file**, so `agent` is defined
     at the very end of `main.py`, after all imports;
  3. the agent directory is put on `sys.path` only for the duration of exec and removed
     immediately -- so all imports must happen at module level, not lazily.

The package contains the SAME modules used in training and local play
(`features.py`, `masks.py`, `model.py`, `play.py`, ...), without copy-paste: otherwise the submission
and the local measurement would drift apart, which is exactly the ailment the repository
guards against with a single canonical encoding point.

Run: ./env/bin/python tools/submission/build.py --ckpt experiments/net/runs/mega_p8/e11.pt
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NET = os.path.join(ROOT, "experiments/net")
MODULES = ["model.py", "backbone.py", "utils.py", "value_head.py", "features.py",
           "masks.py", "play.py", "bc_train.py", "effects.py", "counts.py",
           # policy: turn assembly, masks before sampling and log π (R2, R3, R5)
           "policy.py",
           # envcfg reads the engine configuration (shed capacity, order limit,
           # land prices); without it the package does not import at all
           "envcfg.py"]

MAIN = '''"""Kaggriculture submission. A thin wrapper: all logic is in play.py, the very same one
we measure with locally."""
import os
import sys

import numpy as np
import torch

torch.set_num_threads(2)          # 2 vCPU on the ladder

# kaggle_environments puts the agent directory on sys.path before exec and removes it
# right after, so the import and the weights search are done here, at module level
_DIR = next((p for p in reversed(sys.path)
             if p and os.path.exists(os.path.join(p, "weights.pt"))), os.getcwd())

sys.path.insert(0, _DIR)
import play  # noqa: E402

play._load(os.path.join(_DIR, "weights.pt"))
# The package has no decoder settings: it plays with the `play.agent` defaults -- the same
# as `play.py` from the command line (market gates by sampling, units argmax, one
# pass). The former decode.json was a second path to the settings, and through it the
# ladder since 68fd64a silently played one market pass versus three in measurements.

# Market gates are sampled, so two runs of the same seed diverge. The variable
# is set only by `tools/submission/validate.py`, to check the package code against the code of
# the repository; on Kaggle it is absent and behavior does not change. The seed is set HERE, not
# in the validator: building the model in `play._load` itself draws from the generator, and a seed
# set before the file's exec would be knocked off exactly by the weight initialization.
_seed = os.environ.get("FARMCORE_SEED")
if _seed is not None:
    torch.manual_seed(int(_seed))


# IMPORTANT: agent is defined last -- kaggle_environments takes the last callable
def agent(obs, config=None):
    return play.agent(obs, config)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "raw-content/submission"))
    ap.add_argument("--name", default="submission.tar.gz")
    ap.add_argument("--mkt-order", default="", help="market block order, travels in the package")
    ap.add_argument("--gate-rate", action="store_true", help="gates by accumulated rate, travels in the package")
    # Without this flag the package with --gate-rate accumulated ALL heads, whereas the measurement
    # (`play.py --gate-rate --gate-rate-heads sell`) only the named ones: the setting
    # did not reach the package (29.09).
    ap.add_argument("--gate-rate-heads", default="",
                    help="gate accumulator heads, comma-separated (as in play.py); travels in the package")
    ap.add_argument("--dedup-targets", action="store_true", help="unit targets without repeats, travels in the package")
    ap.add_argument("--feed-guard-risk", action="store_true", help="feed rule only for escape risk")
    ap.add_argument("--feed-guard-hour", type=int, default=None,
                    help="hour of the feed rule in risk mode (default policy.FEED_GUARD_HOUR); travels in the package")
    ap.add_argument("--land-rate", action="store_true", help="land by accumulated rate, travels in the package")
    ap.add_argument("--end-haul", action="store_true", help="in the package: units haul cargo to the shed at the end of the season")
    ap.add_argument("--experts", default="",
                    help="extra experts, comma-separated (weights, `path@style`); travel inside weights.pt")
    ap.add_argument("--route", default="", help="expert router (play.DECODE_DEFAULTS['route'])")
    ap.add_argument("--router", default="", help="router file (router_train.py) -- travels in weights.pt")
    ap.add_argument("--style", default="", help="style of the main net of a style checkpoint")
    ap.add_argument("--night-room", action="store_true", help="in the package: sell the shed surplus before the night dump")
    ap.add_argument("--night-room-mode", default="", choices=["", "stock", "value"],
                    help="in the package: night_room mode (stock -- first version, value -- by price)")
    ap.add_argument("--end-sell", action="store_true", help="enable in the package the sell-off on the last turn")
    ap.add_argument("--feed-guard", action="store_true", help="enable in the package topping up feed for a hungry herd")
    a = ap.parse_args()

    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    for m in MODULES:
        shutil.copy(os.path.join(NET, m), os.path.join(a.out, m))
    # extract_features is needed by bc_train for own_flows/town_step
    shutil.copy(os.path.join(ROOT, "experiments/net/extract_features.py"), a.out)
    open(os.path.join(a.out, "main.py"), "w").write(MAIN)

    sd = torch.load(a.ckpt, map_location="cpu")
    model = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    # `cfg` and the commit travel with the weights for provenance: the architecture is single
    # (frozen 22.09), and `play._load` loads the weights strictly, without a config.
    # Before the freeze the config defined the architecture, and a bare state_dict in the package
    # silently brought up v11-v18 with a different corrector input.
    cfg = sd.get("cfg") if isinstance(sd, dict) else None
    git = sd.get("git") if isinstance(sd, dict) else None
    # `prior` is not provenance but part of the distribution: with β > 0 (H2) the net
    # plays logits plus `β·log prior`, and a package without it would either play a DIFFERENT
    # distribution or (now) loudly refuse in `policy.family`.
    prior = sd.get("prior") if isinstance(sd, dict) else None
    model_cfg = sd.get("model_cfg") if isinstance(sd, dict) else None   # big oh3 (27.09)
    # Decoder settings that already arrived with the weights (building from a package's weights.pt)
    # are kept: previously rebuilding a package from a package silently lost them if the
    # flags were not repeated. Command-line flags only add.
    dec0 = dict((sd.get("decode") or {}) if isinstance(sd, dict) else {})
    if dec0:
        print(f"decoder settings from the weights: {dec0}")
    decode = dec0
    decode.update({k: True for k, on in (("feed_guard", a.feed_guard), ("end_sell", a.end_sell), ("night_room", a.night_room), ("end_haul", a.end_haul),
                                    ("gate_rate", a.gate_rate),
                                    ("dedup_targets", a.dedup_targets),
                                    ("feed_guard_risk", a.feed_guard_risk),
                                    ("land_rate", a.land_rate)) if on})
    if a.feed_guard_hour is not None:
        decode["feed_guard_hour"] = a.feed_guard_hour
    if a.night_room_mode:
        if not a.night_room and not decode.get("night_room"):
            raise SystemExit("--night-room-mode without --night-room changes nothing")
        decode["night_room_mode"] = a.night_room_mode
    if a.gate_rate_heads:
        if not a.gate_rate:
            raise SystemExit("--gate-rate-heads without --gate-rate changes nothing")
        decode["gate_rate_heads"] = a.gate_rate_heads
    if a.mkt_order:
        decode["mkt_order"] = a.mkt_order.split(",")
    experts = []
    for ex in (a.experts.split(",") if a.experts else []):
        pth, _, sty = ex.partition("@")
        e = torch.load(pth, map_location="cpu", weights_only=False)
        experts.append({k: e[k] for k in ("model", "model_cfg", "prior", "style_teams") if k in e}
                       | {"style": sty})
    if experts and not a.route:
        raise SystemExit("--experts without --route: the experts will not play")
    if a.route:
        decode["route"] = a.route
    decode = decode or None
    print(f"decoder settings in the package: {decode}; experts {len(experts)}")
    extra = {"experts": experts} if experts else {}
    if a.style:
        extra["style"] = a.style
    if a.router:
        extra["router"] = torch.load(a.router, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and sd.get("style_teams"):
        extra["style_teams"] = sd["style_teams"]
    torch.save({"model": model, "cfg": cfg, "git": git,      # without the optimizer
                "prior": prior, "model_cfg": model_cfg, "decode": decode, **extra},
               os.path.join(a.out, "weights.pt"))

    # The package is built by copying the MODULES list, and that list is easy to
    # desynchronize from the real imports: that is how `counts.py` went missing, and the submission
    # crashed on the ladder with ImportError, returning $3 000 (the starting bank) with status
    # DONE -- i.e. silently, without a single complaint. So the package is loaded here exactly
    # as Kaggle loads it: exec of the file WITHOUT __file__, in a separate process,
    # with a sys.path that has neither the repository root nor experiments/net.
    probe = ("import sys, os\n"
             f"sys.path[:] = [p for p in sys.path if p not in ({ROOT!r}, {NET!r}, '')]\n"
             f"sys.path.insert(0, {a.out!r})\n"
             f"g = {{}}\n"
             f"exec(compile(open(os.path.join({a.out!r}, 'main.py')).read(), 'main.py', 'exec'), g)\n"
             "fn = [k for k, v in g.items() if callable(v)][-1]\n"
             "assert fn == 'agent', f'last callable = {fn}, Kaggle will take it as the agent'\n"
             "print('package loads, agent:', fn)\n")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       cwd=os.path.dirname(a.out))
    if r.returncode != 0:
        print(r.stdout + r.stderr, file=sys.stderr)
        raise SystemExit("package does not load -- archive not built")
    print(r.stdout.strip())

    shutil.rmtree(os.path.join(a.out, "__pycache__"), ignore_errors=True)  # debris from the check

    tar = os.path.join(ROOT, "raw-content", a.name)
    subprocess.run(["tar", "-czf", tar, "-C", a.out] + sorted(os.listdir(a.out)),
                   check=True)
    size = os.path.getsize(tar) / 1e6
    print(f"package: {a.out}")
    print(f"files: {sorted(os.listdir(a.out))}")
    print(f"parameters in weights: {sum(v.numel() for v in model.values()):,}")
    print(f"archive: {tar}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
