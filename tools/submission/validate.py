"""Check a submission exactly the way Kaggle runs it.

  1. the agent is loaded FROM A FILE (`exec` without `__file__`, last callable), not by import;
  2. the validation episode is the agent against a COPY OF ITSELF, as on submission;
  3. turn limit: look at `remainingOverageTime` at the end of the game, the environment
     itself subtracts from it any excess over `actTimeout`;
  4. match with the local measurement: the same seeds against `starter` must give the same
     money for the package and for `play.py` with the command-line settings (as in measurements).

Run: ./env/bin/python tools/submission/validate.py -n 3 --ckpt <same as in build.py>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from kaggle_environments import make  # noqa: E402


def run(a1, a2, seed):
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed},
               debug=False)
    # The decoder draws market gates from a Bernoulli (policy.Sampler), so two runs
    # of the same seed diverge on their own. Without fixing the generator, item 3 would always
    # print DIVERGENCE -- a check people learn to ignore.
    # The package seeds the generator itself, reading FARMCORE_SEED only AFTER the model is built
    # (building also draws from the generator); here we only pass the number.
    os.environ["FARMCORE_SEED"] = str(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    env.run([a1, a2])
    last = env.steps[-1]
    over = [last[i]["observation"].get("remainingOverageTime") for i in (0, 1)]
    # The status of the last step is always DONE: the interpreter overwrites it for everyone
    # (env_ref.py:958-963), and the reward becomes the frozen bank. A crashed
    # package would report "DONE, $3 000, margin 60" -- exactly what we once
    # saw from a build without counts.py. So find the first NON-ACTIVE status across all steps.
    st = []
    for i in (0, 1):
        bad = next((s[i]["status"] for s in env.steps[:-1]
                    if s[i]["status"] not in ("ACTIVE", "INACTIVE")), None)
        st.append(bad or last[i]["status"])
    return ([last[i].reward or 0.0 for i in (0, 1)], over, time.time() - t0, st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=4000)
    ap.add_argument("--ckpt", required=True, help="the same checkpoint that was built into the package")
    ap.add_argument("--pkg", default=os.path.join(ROOT, "raw-content/submission"),
                    help="package directory (build.py --out)")
    a = ap.parse_args()
    MAIN = os.path.join(a.pkg, "main.py")
    assert os.path.exists(MAIN), "run tools/submission/build.py first"

    print("=== 1. Kaggle validation episode: agent against a copy of itself ===")
    for i in range(min(2, a.n)):
        r, over, el, st = run(MAIN, MAIN, 9000 + i)
        print(f"  seed {9000+i}: ${r[0]:,.0f} vs ${r[1]:,.0f}  "
              f"statuses {st}  time margin {over}  {el:.0f}s")

    print("\n=== 2. from file against starter (comparison with the local measurement) ===")
    got = []
    for i in range(a.n):
        r, over, el, st = run(MAIN, "starter", a.seed0 + i)
        got.append(r[0])
        print(f"  seed {a.seed0+i}: ${r[0]:,.0f} vs ${r[1]:,.0f}  "
              f"status {st[0]}  time margin {over[0]:.1f}  {el:.0f}s")

    print("\n=== 3. the same computation via local play.py (SEPARATE process) ===")
    # Must be a separate process: on load the package does `import play`
    # and fills the global _NET, after which a local run in the same process
    # would get the PACKAGE's weights and settings -- the check would become a tautology.
    import subprocess
    code = f"""
import sys, json
sys.path.insert(0, {os.path.join(ROOT, 'experiments/net')!r})
import play, torch
from kaggle_environments import make
torch.set_num_threads(2)   # same as the package's main.py sets: the thread
                           # count changes the reduction order, and with it argmax
play._load({a.ckpt!r})
# The market order from the checkpoint `cfg` (`bc_train --mkt-order`) is put by `_load` into
# `_NET["mkt_order"]`, and the package plays with it; the defaults below erased it (None), and
# the check compared the package with a different market order (29.09).
_mo = play._NET.get("mkt_order")
# settings -- as in `play.py` from the command line (its defaults), i.e. as in
# MEASUREMENTS: otherwise the check is tautological and cannot see that the package plays differently than we measure
# (that is how the ladder since 68fd64a played one market pass versus three in measurements)
play._NET.update(play.DECODE_DEFAULTS)
play._NET["mkt_order"] = _mo
# ...and on top of them, the settings that arrived WITH THE WEIGHTS (feed rule, 28.09), as
# `play._load` and `play.py` from the command line apply them. Without this line the
# defaults silently turned the rule off, and the check compared the package with a different game.
_dec = torch.load({a.ckpt!r}, map_location="cpu").get("decode")
if _dec:
    play._NET.update(_dec)
print("decoder settings:", {{k: play._NET.get(k) for k in play.DECODE_DEFAULTS}}, file=sys.stderr)
out = []
for i in range({a.n}):
    play._reset_flow()
    env = make("kaggriculture", configuration={{"episodeSteps": 720, "seed": {a.seed0} + i}},
               debug=False)
    torch.manual_seed({a.seed0} + i)     # same seed as in run() above
    env.run([play.agent, "starter"])
    out.append(env.steps[-1][0].reward or 0.0)
print(json.dumps(out))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    local = json.loads(res.stdout.strip().splitlines()[-1])
    same = 0
    for i, r in enumerate(local):
        ok = abs(r - got[i]) < 1e-6
        same += ok
        print(f"  seed {a.seed0+i}: ${r:,.0f}  {'match' if ok else 'DIVERGENCE'}")
    print(f"\ntotal: matched {same}/{a.n} games")


if __name__ == "__main__":
    main()
