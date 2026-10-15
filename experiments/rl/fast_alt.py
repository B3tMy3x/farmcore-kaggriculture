"""Fast alternatives to per-micro-turn PPO: a shared game engine and three experiments.

Why (24.09). Per-micro-turn PPO learns noise; the main line tests PPO + sparse
exploration. Here are other ways to turn the simulator into a signal, each with
a cheap works/doesn't experiment and a control:

  es      ES on the market-head subspace (167k parameters) with common random
          numbers (PEGASUS) and antithetic pairs. Measures:
          (1) chaos — when the game diverges from the base under a σ shift;
          (2) reliability of the directional derivative: correlation across
          directions of differences F(θ+σε)−F(θ−σε) computed on TWO disjoint
          seed sets. r ≈ 0 — ES learns noise; r > 0 — there is signal.
  rpi     "Rollout improvement" labels with a DETERMINISTIC future: base up to
          turn T is the ladder policy (BC mode: units argmax, gate sampled with a
          seeded generator), at T — K different turn candidates, each played out
          by THE SAME ladder policy over m futures (own seeding of our gate
          sampling and of the opponent). Label reliability is the correlation of
          A across futures. The points file is compatible with `rollout_pi.py train`.
  groups  Sparse groups (like the main line's `rollout.explore_turns` + `crn`):
          K games of one seed, PPO-mode sampling on n_ex of our turns, the rest
          the ladder policy. Records of exploration turns and game margin are
          raw material for per-game Best-of-N / AWR.
  train   Training on points/groups with different losses (linear A·log π —
          Rollout PI; best-of-K CE — RAFT/ExIt; AWR exp(A/β)) with a KL leash to
          the start and a shuffled control. Single criterion: E_{π|candidates}[A]
          on HELD-OUT rpi points (gain over start vs control).

Speed. The game is the Rust port (`kaggriculture_rs.Env`, `clone()`) and the
native feature encoder (`encode_all`, the same `features.NATIVE` slot as
`collect_lockstep.RustEnvRunner`); the turn is the unmodified `play.agent` (torch
net at batch 1 on a core — that is the bottleneck, ~8 ms per decision on an
M-series core). Parallel via processes (`-j`), one progress line per work unit.

  ./env/bin/python -u experiments/rl/fast_alt.py es-chaos --seeds 16 -j 10
  ./env/bin/python -u experiments/rl/fast_alt.py es --dirs 24 --seeds 24 --sigma 0.01 -j 10
  ./env/bin/python -u experiments/rl/fast_alt.py rpi --seeds 40000 40040 --points 4 -j 10 --out rpi_a.pkl
  ./env/bin/python -u experiments/rl/fast_alt.py groups --seeds 50000 50030 --k 8 -j 10 --out grp_a.pkl
  ./env/bin/python -u experiments/rl/fast_alt.py train --rpi 'rpi_*.pkl' --loss lin
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import numpy as np  # noqa: E402

DEF_CKPT = os.path.join(ROOT, "experiments/net/runs/value_tower/e11.pt")
MKT = ("head_sell", "head_buy", "head_seed", "head_animal", "head_hire",
       "head_land", "head_qty", "co_mkt")
OPP_STREAM = 7
HORIZONS = (1, 6, 24, 72, 168)

# ------------------------------------------------------------------ worker

W = {}          # worker process state


def _init(ckpt, opp_ckpt, threads=1):
    import torch
    torch.set_num_threads(threads)
    import collect as C
    import play
    import policy as POL
    from collect_lockstep import RustEnvRunner
    net = C.load_net(ckpt, "cpu")
    C.install(play, net, "cpu", C.decode_mode(POL.BC), record=False,
              prior=C.load_prior(ckpt, "cpu"), who="our policy")
    play._NET["keep_obs"] = False
    play._NET["logp_now"] = False
    opp = C.fresh_play_module()
    onet = C.load_net(opp_ckpt, "cpu")
    C.install(opp, onet, "cpu", C.decode_mode(POL.BC), record=False,
              prior=C.load_prior(opp_ckpt, "cpu"), who="opponent (BC)")
    cfg = RustEnvRunner(native_encode=True)._cfg
    W.update(torch=torch, C=C, play=play, opp=opp, net=net, cfg=cfg,
             RER=RustEnvRunner, unbind=None,
             BC={k: POL.BC[k] for k in ("sample", "sample_units", "sample_qty")},
             PPO={k: POL.PPO[k] for k in ("sample", "sample_units", "sample_qty")},
             DET={"sample": False, "sample_units": 0.0, "sample_qty": 0.0},
             POL=POL, knob={},
             base={k: p.detach().clone() for k, p in net.named_parameters()
                   if k.split(".")[0] in MKT})


def _bind(env):
    """Native encoder bound to THIS engine copy (the slot is process-wide)."""
    if W["unbind"] is not None:
        W["unbind"]()
    W["unbind"] = W["RER"]._install_native(None, env)


def tseed(a, t, c=0):
    return (int(a) * 1_000_003 + int(t) * 7_919 + int(c) * 104_729) % (2 ** 63 - 1)


def new_game(seed):
    import kaggriculture_rs as K
    W["play"]._reset_flow()
    W["opp"]._reset_flow()
    env = K.Env(config={"episodeSteps": 720}, seed=seed)
    obs = env.reset(seed)
    return env, obs


def snapshot(env, obs):
    import copy
    return (env.clone(), copy.deepcopy(obs), copy.deepcopy(W["play"]._FLOW),
            copy.deepcopy(W["opp"]._FLOW))


def restore(snap):
    import copy
    env, obs, f_me, f_op = snap
    W["play"]._FLOW.clear(); W["play"]._FLOW.update(copy.deepcopy(f_me))
    W["opp"]._FLOW.clear(); W["opp"]._FLOW.update(copy.deepcopy(f_op))
    return env.clone(), copy.deepcopy(obs)


def step_us(obs_me, t, ss, mode, alt=0, record=False):
    """Our turn: seeding (ss, t, alt), decoder mode. -> (action, record|None)."""
    play, torch = W["play"], W["torch"]
    play._NET.update(W[mode])
    torch.manual_seed(tseed(ss, t, alt))
    rec = None
    if record:
        play._NET["record"] = []
    try:
        act = play.agent(obs_me, W["cfg"])
    finally:
        if record:
            r = play._NET["record"].pop()
            play._NET["record"] = None
            rec = {"enc": r["enc"], "legal0": r["legal0"], "lat": r["lat"]}
    return act, rec


def step_opp(obs_op, t, oseed):
    W["torch"].manual_seed(tseed(oseed, t, OPP_STREAM))
    return W["opp"].agent(obs_op, W["cfg"])


SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
LAND_COST = {"NE": 1000, "SW": 2000, "SE": 4000}


CROP = {  # env_ref.py:CROPS
    "WHEAT": (2, 4, 0, 6, False), "CARROT": (2, 3, 0, 4, False), "TOMATO": (8, 8, 1, 4, True),
    "STRAWBERRY": (10, 10, 2, 4, True), "MELON": (10, 12, 0, 6, False)}
ANIMAL = {"GOOSE": (4, 1), "COW": (8, 2), "SHEEP": (6, 3)}   # first_yield_day, interval
LAST_DAY = 29        # last day a harvest can still be collected and sold


def valuation(o):
    """A side's value from ITS observation -> (money, liquid, total, season).

    liquid = money + (shed + unit hands) × current market price;
    total = liquid + seeds and plants at seed price + yield on tiles at price
    + animals at purchase price + unlocked quadrants at price;
    season = liquid + what can still turn into money by season end
    (env_ref: CROPS/ANIMALS, `_daily_refresh_*`): a plant — its yield at price
    if it matures by day 29 (ongoing crops — remaining harvests of 1 unit), an
    animal — accumulated + 1.5/interval units per day to the end minus feed
    (wheat per day), seeds — seed price if the planted crop would mature, land —
    price × fraction of season left. Rough (ignores price impact of own sales)."""
    pr = o["market"]["prices"]
    f = o["farms"][o["player"]]
    day = int(o.get("step", o["day"] * 24 + o["hour"])) // 24
    money = float(f["money"])
    pv = o["private"]
    liq = money
    for k, n in pv["shed"].items():
        if k in pr:
            liq += n * pr[k]
        elif k in ANIMAL_COST:
            liq += n * ANIMAL_COST[k]
    for inv in pv.get("inventories", []) or []:
        for k, n in inv.items():
            liq += n * pr.get(k, 0)
    tot = liq + sum(n * SEED_COST[k] for k, n in pv["seeds"].items())
    sea = liq + sum(n * SEED_COST[k] for k, n in pv["seeds"].items()
                    if day + CROP[k][1] <= LAST_DAY)
    for row in f["tiles"]:
        for c in row:
            if not isinstance(c, dict):
                continue
            if c.get("kind") == "PLANT":
                cr = c.get("crop")
                tot += SEED_COST.get(cr, 0) + c.get("yield_units", 0) * pr.get(cr, 0)
                fy, my, iv, mx, ongoing = CROP[cr]
                pd = int(c.get("planted_day", day))
                if not ongoing:
                    if pd + my <= LAST_DAY:
                        sea += mx * pr.get(cr, 0)
                    elif pd + fy <= LAST_DAY:
                        sea += 0.5 * mx * pr.get(cr, 0)
                else:
                    first = pd + fy
                    done = max(0, (day - first) // iv + 1) if day >= first else 0
                    fut = sum(1 for d in range(max(day + 1, first), LAST_DAY + 1)
                              if (d - first) % iv == 0)
                    fut = max(0, min(fut, mx - done))
                    sea += (c.get("yield_units", 0) + fut) * pr.get(cr, 0)
            a = c.get("animal")
            if a:
                tot += ANIMAL_COST[a] + c.get("yield_units", 0) * pr.get(ANIMAL_PRODUCT[a], 0)
                fy, iv = ANIMAL[a]
                left = max(0, LAST_DAY - max(day, int(c.get("placed_day", day)) + fy))
                sea += (c.get("yield_units", 0) + 1.5 / iv * left) * pr.get(ANIMAL_PRODUCT[a], 0)
                sea -= max(0, LAST_DAY - day) * pr.get("WHEAT", 0)
    for q in f.get("unlocked_quadrants", []):
        tot += LAND_COST.get(q, 0)
        sea += LAND_COST.get(q, 0) * max(0, LAST_DAY - day) / 30
    return money, liq, tot, sea


def play_out(env, obs, t, me, ss, oseed, until=None, explore=(), record=False,
             trace=None, base_mode="BC", probe=None):
    """Play out from turn t to `until` (None — to the end). Our mode is the ladder
    (BC), on `explore` turns — PPO sampling. -> (env, obs, t, margin|None, records)."""
    _bind(env)
    recs = []
    while until is None or t < until:
        ex = t in explore
        a_me, rec = step_us(obs[me], t, ss, "PPO" if ex else base_mode, record=record and ex)
        if rec is not None:
            recs.append((t, rec))
        if trace is not None:
            trace.append(zlib.crc32(json.dumps(a_me, sort_keys=True, default=str).encode()))
        acts = [None, None]
        acts[me] = a_me
        acts[1 - me] = step_opp(obs[1 - me], t, oseed)
        res = env.step(acts)
        obs = res["observations"]
        t += 1
        if probe is not None and t in probe:
            a_, b_ = valuation(obs[me]), valuation(obs[1 - me])
            probe[t] = tuple(x - y for x, y in zip(a_, b_))
        if res["done"]:
            r = [float(x or 0.0) for x in res["rewards"]]
            return env, obs, t, r[me] - r[1 - me], recs
    return env, obs, t, None, recs


# ------------------------------------------------------------------ decoder knobs

KNOBS = ("sell", "buy", "seed", "animal", "hire", "land", "sellq", "buyq")


def _set_knob(knob):
    """Logit shifts for OUR side after `policy.family`: `<head>` — the gate
    (logit 0 of the head row; for land, the only one), `<head>q` — the quantity
    chain (logits 1..K: plus means more units). The opponent is untouched: its
    module sees the original `policy`, while our `play.POL` is replaced with a
    namespace holding a wrapped `family`."""
    import types
    import counts as CNT
    POL, play = W["POL"], W["play"]
    knob = {k: float(v) for k, v in (knob or {}).items() if v}
    if not knob:
        play.POL = POL
        return

    def family(out, prior=None, beta=0.0, floor=0.0):
        o = POL.family(out, prior, beta, floor)
        for k, b in knob.items():
            if k == "land":
                o["land_logits"] = o["land_logits"] + b
            elif k.endswith("q"):
                h = k[:-1]
                lg = o[f"{h}_logits"]
                kk = CNT.ALL_CAPS[h]
                lg = lg.clone(); lg[..., 1:kk] += b
                o[f"{h}_logits"] = lg
            else:
                lg = o[f"{k}_logits"].clone(); lg[..., 0] += b
                o[f"{k}_logits"] = lg
        return o

    ns = types.SimpleNamespace(**{k: getattr(POL, k) for k in dir(POL) if not k.startswith("__")})
    ns.family = family
    play.POL = ns


def knob_task(task):
    knob, seed, mode = task
    _set_knob(knob)
    me = seed % 2
    env, obs = new_game(seed)
    _, _, _, m, _ = play_out(env, obs, 0, me, seed, seed, base_mode=mode)
    _set_knob({})
    return {"knob": knob, "seed": seed, "margin": m}


def cmd_knob(a):
    """Decoder knobs: each arm vs the base on the same seeds (CRN)."""
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    arms = [{}]
    for spec in a.arms:
        d = {}
        for part in spec.split(","):
            k, v = part.split(":")
            assert k in KNOBS, k
            d[k] = float(v)
        arms.append(d)
    tasks = [(arm, s, a.mode) for arm in arms for s in seeds]
    res = run_pool(knob_task, tasks, a, "knob")
    if a.out:
        pickle.dump(res, open(a.out, "wb"))
    knob_report(res, seeds)


def knob_report(res, seeds):
    key = lambda d: json.dumps(d, sort_keys=True)
    M = {(key(r["knob"]), r["seed"]): r["margin"] for r in res}
    b = np.array([M[(key({}), s)] for s in seeds])
    print(f"\nbase: mean margin ${b.mean():,.0f}; seeds {len(seeds)}")
    for arm in sorted({key(r["knob"]) for r in res} - {key({})}):
        x = np.array([M[(arm, s)] for s in seeds]) - b
        nz = x[x != 0]
        from scipy.stats import wilcoxon
        p = wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
        print(f"  {arm:28s} Δ ${x.mean():+7,.0f} ± {x.std(ddof=1) / len(x) ** .5:5,.0f}  "
              f"better {int((x > 0).sum())}/{len(x)}  wilcoxon p {p:.3f}")


# ------------------------------------------------------------------ ES

def _perturb(dir_id, sign, sigma):
    """θ = θ0 + sign·σ·std(tensor)·ε(dir_id) on the market heads (in place)."""
    torch = W["torch"]
    net = W["net"]
    with torch.no_grad():
        g = torch.Generator().manual_seed(1_000_003 * (dir_id + 1))
        for k, p in net.named_parameters():
            if k not in W["base"]:
                continue
            b = W["base"][k]
            p.copy_(b)
            if dir_id >= 0 and sigma > 0:
                s = float(b.std()) if b.numel() > 1 else float(b.abs().mean() + 1e-2)
                p.add_(torch.randn(b.shape, generator=g) * (sign * sigma * s))


def es_task(task):
    dir_id, sign, sigma, seed, want_trace = task
    _perturb(dir_id, sign, sigma)
    me = seed % 2
    env, obs = new_game(seed)
    tr = [] if want_trace else None
    _, _, _, m, _ = play_out(env, obs, 0, me, seed, seed, trace=tr)
    _perturb(-1, 0, 0)
    return {"dir": dir_id, "sign": sign, "sigma": sigma, "seed": seed, "margin": m,
            "trace": tr}


# ------------------------------------------------------------------ rpi

def rpi_task(task):
    """Points of one game: the base advances over increasing T, at each T — a
    snapshot, K candidates, m futures. -> list of points."""
    seed, Ts, K, m, tries, bm = task[:6]
    short = int(task[6]) if len(task) > 6 else 0
    me = seed % 2
    env, obs = new_game(seed)
    t, out = 0, []
    for T in sorted(Ts):
        env, obs, t, _, _ = play_out(env, obs, t, me, seed, seed, until=T, base_mode=bm)
        snap = snapshot(env, obs)
        cands, seen = [], set()
        for j in range(tries):
            _, o = restore(snap)
            _bind(snap[0])
            mode = bm if j == 0 else "PPO"
            act, rec = step_us(o[me], T, seed, mode, alt=0 if j == 0 else 1000 + j, record=True)
            key = json.dumps(act, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            cands.append({"alt": 0 if j == 0 else 1000 + j, "mode": mode, "key": key,
                          "rec": rec, "act": act, "margin": []})
            if len(cands) >= K:
                break
        for c in cands:
            for r in range(m):
                e, o = restore(snap)
                _bind(e)
                acts = [None, None]
                acts[me] = c["act"]
                acts[1 - me] = step_opp(o[1 - me], T, seed)   # opponent's turn T is shared
                res = e.step(acts)
                o = res["observations"]
                # future r: own seeding of OUR gate sampling and of the opponent
                fs = seed * 31 + 17 + r * 7_777_777
                pr = {T + n: None for n in HORIZONS if T + n < 720}
                if T + 1 in pr:
                    a_, b_ = valuation(o[me]), valuation(o[1 - me])
                    pr[T + 1] = tuple(x - y for x, y in zip(a_, b_))
                stop = min(720, T + short + 1) if short else None
                _, _, _, mg, _ = play_out(e, o, T + 1, me, fs, fs, base_mode=bm, probe=pr,
                                          until=stop)
                c["margin"].append(mg)
                c.setdefault("probe", []).append(pr)
        for c in cands:
            c.pop("act")
        out.append({"seed": seed, "T": T, "me": me, "cands": cands})
        env, obs = restore(snap)
    return {"points": out}


# ------------------------------------------------------------------ groups

def group_task(task):
    """One game of a sparse group: env seed, sampling seed, n_ex."""
    seed, ss, n_ex, bm = task
    me = seed % 2
    ex = set(random.Random(ss * 7_919 + 13).sample(range(719), n_ex))
    env, obs = new_game(seed)
    _, _, _, mg, recs = play_out(env, obs, 0, me, ss, seed, explore=ex, record=True,
                                 base_mode=bm)
    return {"seed": seed, "ss": ss, "me": me, "margin": mg, "recs": recs}


# ------------------------------------------------------------------ pool

def run_pool(fn, tasks, a, label):
    """Tasks over processes, one unbuffered progress line per task.

    Partial results are written to `a.out + ".part"` every 10 tasks, and on
    rerun finished tasks are skipped: a cloud sandbox kills processes without
    warning (on 24.09 that lost 253 games out of 312)."""
    part = (a.out + ".part") if getattr(a, "out", "") else ""
    out = []
    if part and os.path.exists(part):
        out = pickle.load(open(part, "rb"))
        done = {repr(r["task"]) for r in out}
        tasks = [t for t in tasks if repr(t) not in done]
        print(f"resuming: done {len(out)}, left {len(tasks)}", flush=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(a.jobs, initializer=_init, initargs=(a.ckpt, a.opp_ckpt or a.ckpt)) as pool:
        for i, (t, r) in enumerate(pool.imap_unordered(_tagged, [(fn, t) for t in tasks])):
            r["task"] = t
            out.append(r)
            el = time.time() - t0
            print(f"__PROG__ {label} {i + 1}/{len(tasks)}  {el / 60:.1f} min, "
                  f"left ~{el / (i + 1) * (len(tasks) - i - 1) / 60:.1f} min", flush=True)
            if part and ((i + 1) % 10 == 0 or i + 1 == len(tasks)):
                with open(part + ".tmp", "wb") as f:
                    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(part + ".tmp", part)
    return out


def _tagged(ft):
    fn, t = ft
    return t, fn(t)


def cmd_es_chaos(a):
    """Chaos: turn of first divergence from the base and |ΔF| for different σ."""
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    sigmas = [float(s) for s in a.sigmas.split(",")]
    tasks = [(-1, 0, 0.0, s, True) for s in seeds]
    tasks += [(d, sg, sig, s, True) for sig in sigmas for d in range(a.dirs)
              for sg in (1, -1) for s in seeds]
    res = run_pool(es_task, tasks, a, "es-chaos")
    base = {r["seed"]: r for r in res if r["dir"] < 0}
    sd_seed = np.std([base[s]["margin"] for s in seeds])
    print(f"\nbase: margin spread across seeds ${sd_seed:,.0f}")
    for sig in sigmas:
        rs = [r for r in res if r["dir"] >= 0 and r["sigma"] == sig]
        dF = np.array([r["margin"] - base[r["seed"]]["margin"] for r in rs])
        div = []
        for r in rs:
            b = base[r["seed"]]["trace"]
            k = next((i for i, (x, y) in enumerate(zip(r["trace"], b)) if x != y), len(b))
            div.append(k)
        same = np.mean([d >= 719 for d in div])
        print(f"σ {sig:g}: |ΔF| median ${np.median(np.abs(dF)):,.0f}, std ${dF.std():,.0f}; "
              f"first divergence: median turn {np.median(div):.0f}, "
              f"games without divergence {same:.0%}")
    if a.out:
        pickle.dump(res, open(a.out, "wb"))


def cmd_es(a):
    """Directional-derivative reliability across two seed halves + ES step."""
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    tasks = [(-1, 0, 0.0, s, False) for s in seeds]
    tasks += [(d, sg, a.sigma, s, False) for d in range(a.dirs) for sg in (1, -1) for s in seeds]
    res = run_pool(es_task, tasks, a, "es")
    if a.out:
        pickle.dump(res, open(a.out, "wb"))
    es_report(res, seeds, a.dirs)


def es_report(res, seeds, dirs):
    M = {(r["dir"], r["sign"], r["seed"]): r["margin"] for r in res}
    D = np.array([[M[(d, 1, s)] - M[(d, -1, s)] for s in seeds] for d in range(dirs)])  # [dirs, seeds]
    h1, h2 = D[:, 0::2], D[:, 1::2]
    m1, m2 = h1.mean(1), h2.mean(1)
    r = np.corrcoef(m1, m2)[0, 1]
    # signal/noise: variance of the mean across directions vs the mean's standard error
    mean_d = D.mean(1)
    se2 = (D.var(1, ddof=1) / D.shape[1]).mean()
    sig2 = max(mean_d.var(ddof=1) - se2, 0.0)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        p = rng.permutation(dirs)
        null.append(np.corrcoef(m1, m2[p])[0, 1])
    pval = float(np.mean(np.array(null) >= r))
    print(f"\nES: directions {dirs}, seeds {len(seeds)}; |D| median ${np.median(np.abs(D)):,.0f}")
    F0 = np.array([M[(-1, 0, s)] for s in seeds]) if (-1, 0, seeds[0]) in M else None
    if F0 is not None:
        Fp = np.array([[M[(d, 1, s)] for s in seeds] for d in range(dirs)])
        Fm = np.array([[M[(d, -1, s)] for s in seeds] for d in range(dirs)])
        cur = (Fp + Fm) / 2 - F0[None]
        print(f"  (F+ + F-)/2 - F0: mean ${cur.mean():+,.0f} ± {cur.mean(1).std(ddof=1) / dirs ** .5:,.0f} "
              f"(harm of a random shift); best of 2·dirs shifts vs base ${max(Fp.mean(1).max(), Fm.mean(1).max()) - F0.mean():+,.0f}")
    print(f"  correlation of mean D across seed halves r = {r:+.3f} (permutation p = {pval:.3f})")
    print(f"  directional derivative variance: signal {sig2 ** .5:,.0f}$, mean's std error "
          f"{se2 ** .5:,.0f}$, signal share {sig2 / (sig2 + se2 + 1e-9):.2f}")
    return r, pval


def cmd_es_step(a):
    """ES step from an `es` dump: θ1 = θ0 + α·σ·Σ c_d ε_d, c = w/‖w‖, w_d — the mean
    antithetic difference along direction d. Shift norm ≈ norm of one
    perturbation (ε independent), i.e. a step "one probe perturbation long".
    `--shuffle` is the control: the same |w|, shuffled across directions."""
    import torch
    import collect as C
    res = pickle.load(open(a.dump, "rb"))
    sigma = next(r["sigma"] for r in res if r["dir"] >= 0)
    dirs = 1 + max(r["dir"] for r in res)
    seeds = sorted({r["seed"] for r in res})
    M = {(r["dir"], r["sign"], r["seed"]): r["margin"] for r in res}
    w = np.array([np.mean([M[(d, 1, s)] - M[(d, -1, s)] for s in seeds]) for d in range(dirs)])
    if a.shuffle:
        w = np.random.default_rng(a.seed).permutation(w) * np.random.default_rng(a.seed + 1).choice([-1, 1], dirs)
    c = w / np.linalg.norm(w)
    print(f"directions {dirs}, σ {sigma}, α {a.alpha}; weights c: {np.round(c, 2).tolist()}")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = dict(ck["model"])
    net = C.load_net(a.ckpt, "cpu")
    with torch.no_grad():
        delta = {k: torch.zeros_like(p) for k, p in net.named_parameters() if k.split(".")[0] in MKT}
        for d in range(dirs):
            g = torch.Generator().manual_seed(1_000_003 * (d + 1))
            for k, p in net.named_parameters():
                if k not in delta:
                    continue
                b = p.detach()
                s_ = float(b.std()) if b.numel() > 1 else float(b.abs().mean() + 1e-2)
                delta[k] += torch.randn(b.shape, generator=g) * (s_ * a.alpha * sigma * float(c[d]))
        for k, v in delta.items():
            assert k in sd, k
            sd[k] = sd[k] + v.to(sd[k].dtype)
    torch.save({"model": sd, "prior": ck.get("prior")}, a.save)
    print(f"weights: {a.save}")


def cmd_rpi(a):
    rng = random.Random(a.seeds[0] * 7919 + 3)
    tasks = [(s, tuple(sorted(rng.sample(range(a.tmin, a.tmax), a.points))), a.k, a.m,
              a.tries, a.base_mode, a.short) for s in range(a.seeds[0], a.seeds[1])]
    res = run_pool(rpi_task, tasks, a, "rpi")
    pts = [p for r in res for p in r["points"]]
    pickle.dump(pts, open(a.out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    if pts and pts[0]["cands"][0]["margin"][0] is not None:
        rpi_report(pts)
    probe_report(pts)


def probe_report(pts):
    """Reliability of a decision's advantage on local quantities n turns later:
    r(A_n future 0, A_n future 1) and the link of A_n to final margin (same future)."""
    names = ("money", "liquid", "total", "season")
    print("\nhorizon   quantity   pairs r(futures)   95% CI          |A| med   corr(A_n, A_final)")
    for n in HORIZONS:
        for q, nm in enumerate(names):
            a0, a1, f0 = [], [], []
            for p in pts:
                cs = p["cands"]
                if len(cs) < 2 or "probe" not in cs[0] or len(cs[0]["probe"]) < 2:
                    continue
                t = p["T"] + n
                if any(c["probe"][r].get(t) is None for c in cs for r in (0, 1)):
                    continue
                if len(cs[0]["probe"][0][t]) <= q:
                    continue
                V = np.array([[c["probe"][r][t][q] for r in (0, 1)] for c in cs], float)
                F = np.array([[np.nan if x is None else x for x in c["margin"][:2]] for c in cs], float)
                A = V - V.mean(0); AF = F - F.mean(0)
                a0 += list(A[:, 0]); a1 += list(A[:, 1]); f0 += list(AF[:, 0])
            if len(a0) < 10:
                continue
            a0, a1, f0 = map(np.array, (a0, a1, f0))
            r = np.corrcoef(a0, a1)[0, 1] if a0.std() > 0 and a1.std() > 0 else float("nan")
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(500):
                i = rng.integers(0, len(a0), len(a0))
                if a0[i].std() > 0 and a1[i].std() > 0:
                    bs.append(np.corrcoef(a0[i], a1[i])[0, 1])
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"),) * 2)
            ok = ~np.isnan(f0)
            cf = np.corrcoef(a0[ok], f0[ok])[0, 1] if ok.sum() > 10 and a0.std() > 0 else float("nan")
            print(f"  +{n:<4d}   {nm:9s} {len(a0):5d}   {r:+.2f}       [{lo:+.2f}, {hi:+.2f}]   "
                  f"${np.median(np.abs(a0)):7,.0f}   {cf:+.2f}")


def rpi_report(pts):
    """Label reliability: A on future 0 vs A on future 1."""
    a0, a1, spread = [], [], []
    for p in pts:
        if len(p["cands"]) < 2 or p["cands"][0]["margin"][0] is None:
            continue
        M = np.array([c["margin"] for c in p["cands"]], dtype=float)
        if M.shape[1] < 2:
            continue
        A = M - M.mean(0, keepdims=True)
        a0 += list(A[:, 0]); a1 += list(A[:, 1])
        spread.append(np.ptp(M.mean(1)))
    a0, a1 = np.array(a0), np.array(a1)
    r = np.corrcoef(a0, a1)[0, 1]
    print(f"\npoints {len(pts)}, label pairs {len(a0)}: corr(A_future0, A_future1) = {r:+.3f}; "
          f"|A| median ${np.median(np.abs(a0)):,.0f}; spread of means ${np.median(spread):,.0f}")
    # "signal" share of a label from two futures: r = var_s / (var_s + var_n)
    print(f"  label reliability, one future {r:.2f}; mean of two {2 * r / (1 + r):.2f}")
    return r


def cmd_groups(a):
    tasks = []
    for s in range(a.seeds[0], a.seeds[1]):
        for j in range(a.k):
            tasks.append((s, s * 1000 + j + 1, a.n_ex, a.base_mode))
    res = run_pool(group_task, tasks, a, "groups")
    pickle.dump(res, open(a.out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    by = {}
    for r in res:
        by.setdefault(r["seed"], []).append(r["margin"])
    within = np.mean([np.var(v, ddof=1) for v in by.values()])
    tot = np.var([m for v in by.values() for m in v], ddof=1)
    print(f"\ngroups {len(by)}: std within ${within ** .5:,.0f}, total ${tot ** .5:,.0f}, "
          f"share within {within / tot:.2f}")


# ------------------------------------------------------------------ training

def cmd_train(a):
    """Losses on the same labels; criterion — E_{π|candidates}[A] on held-out rpi."""
    import torch
    import collect as C
    import critic as CRIT
    import policy as POL
    import ppo as PPO
    from config import PPOCfg

    raw = [p for f in sorted(sum((glob.glob(x) for x in a.rpi), []))
           for p in pickle.load(open(f, "rb"))]
    raw = [x for p in raw for x in (p["points"] if "points" in p else [p])]

    def labels(p):
        """Label matrix [K, m]: final margin or probe "n:q" (quantity q of the
        value difference at turn T+n, `valuation`)."""
        if a.label == "final":
            return np.array([c["margin"] for c in p["cands"]], dtype=np.float64)
        n, q = (int(x) for x in a.label.split(":"))
        t = p["T"] + n
        return np.array([[pr[t][q] if pr.get(t) is not None else np.nan for pr in c["probe"]]
                         for c in p["cands"]], dtype=np.float64)

    pts = [p for p in raw if len(p["cands"]) > 1 and not np.isnan(labels(p)).any()]
    seeds = sorted({p["seed"] for p in pts})
    rs = random.Random(a.split_seed); rs.shuffle(seeds)
    held = set(seeds[:int(round(len(seeds) * a.held))])
    tr_pts = [p for p in pts if p["seed"] not in held]
    ho = [p for p in pts if p["seed"] in held]

    def adv(p):
        M = labels(p)
        if a.future >= 0:
            M = M[:, a.future:a.future + 1]
        return (M - M.mean(0, keepdims=True)).mean(1) / a.scale

    # training "pairs": (records, weights) per point
    items = []
    if a.groups:
        grp = [g for f in sorted(sum((glob.glob(x) for x in a.groups), [])) for g in pickle.load(open(f, "rb"))]
        by = {}
        for g in grp:
            by.setdefault(g["seed"], []).append(g)
        for s, gs in by.items():
            if s in held or len(gs) < 2:
                continue
            ms = np.array([g["margin"] for g in gs]) / 1e4
            A = ms - ms.mean()
            best = int(np.argmax(ms))
            for i, g in enumerate(gs):
                if a.loss == "bon":
                    w = 1.0 if i == best else 0.0
                elif a.loss == "awr":
                    w = float(np.exp(np.clip(A[i] / a.beta, -5, 5)))
                else:
                    w = float(A[i])
                if w == 0.0:
                    continue
                for _, rec in g["recs"]:
                    items.append(([rec], np.array([w])))
    else:
        for p in tr_pts:
            A = adv(p)
            recs = [c["rec"] for c in p["cands"]]
            if a.loss == "self":
                # null experiment: the label is OUR OWN turn in base mode (candidate
                # alt 0), as in play; a correct training path must not move money
                w = np.array([1.0 if c["alt"] == 0 else 0.0 for c in p["cands"]])
            elif a.loss == "bon":
                w = np.zeros(len(A)); w[int(np.argmax(A))] = 1.0
            elif a.loss == "awr":
                w = np.exp(np.clip(A / a.beta, -5, 5)); w = w / w.sum()
            else:
                w = A
            if a.det_logp == "drop":
                # ⚠ The alt 0 turn is recorded DETERMINISTICALLY (gate threshold,
                # median, argmax — DET/BC mode), while `log π` is computed in PPO
                # mode (τ 0.5/1, Bernoulli gate). Such a label is not a sample from
                # the loss distribution: E[∇log π(label)] ≠ 0 even on OUR OWN
                # turns, and the step systematically sharpens gates and shrinks
                # quantities (measured: det_bs512 −$1.9k over 144 games, the same
                # config on sampled labels — about zero). So only PPO samples go
                # into `log π`; alt 0 stays in the A baseline.
                ppo = np.array([c["mode"] == "PPO" for c in p["cands"]])
                if a.loss == "self":
                    w = ppo / max(1, ppo.sum())
                else:
                    w = np.where(ppo, w, 0.0)
            items.append((recs, w))
    print(f"rpi points {len(pts)}: train {len(tr_pts)}, held out {len(ho)}; "
          f"training items {len(items)} ({'groups' if a.groups else 'rpi'}, loss {a.loss})",
          flush=True)

    torch.manual_seed(a.seed)
    dev = a.device
    net = C.load_net(a.ckpt, dev)
    anchor = C.load_net(a.ckpt, dev)
    cfg = PPOCfg()
    trn = PPO.Trainer(net, anchor, CRIT.make("zero"), cfg, C.decode_mode(POL.PPO),
                      device=dev, prior=C.load_prior(a.ckpt, dev))
    params = [p for k, p in net.named_parameters()
              if p.requires_grad and (not a.mkt_only or k.split(".")[0] in MKT)]
    if a.mkt_only:
        for k, p in net.named_parameters():
            if k.split(".")[0] not in MKT:
                p.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    def batch(recs):
        x = {k: torch.from_numpy(np.stack([r["enc"][k] for r in recs])).to(dev)
             for k in recs[0]["enc"]}
        x["legal"] = torch.from_numpy(np.stack([r["legal0"] for r in recs])).to(dev)
        return x, POL.stack([r["lat"] for r in recs], dev)

    def gain(ps, shuffle=False):
        g, rsh = [], random.Random(1)
        net.eval()
        with torch.no_grad():
            for i in range(0, len(ps), 32):
                sub = ps[i:i + 32]
                recs = [c["rec"] for p in sub for c in p["cands"]]
                x, lat = batch(recs)
                lp, _, _, _, _ = trn.evaluate(x, lat, grad=False, want_kl=False)
                lp = lp.float().cpu().numpy()
                o = 0
                for p in sub:
                    n = len(p["cands"])
                    M = labels(p)
                    A = (M - M.mean(0, keepdims=True)).mean(1) / 1e4
                    if shuffle:
                        A = A[rsh.sample(range(n), n)]
                    w = np.exp(lp[o:o + n] - lp[o:o + n].max()); w /= w.sum()
                    g.append(float((w * A).sum())); o += n
        return float(np.mean(g)), float(np.std(g) / np.sqrt(len(g)))

    g0, se0 = gain(ho)
    print(f"start: E[A|π] held-out {g0 * 1e4:+.0f}$ (±{se0 * 1e4:.0f})", flush=True)
    rsh = np.random.default_rng(a.seed)
    stop = False
    for ep in range(a.epochs):
        if stop:
            break
        idx = rsh.permutation(len(items))
        net.train()
        kls = []
        for i in range(0, len(idx), a.bs):
            sub = [items[j] for j in idx[i:i + a.bs]]
            recs = [r for rr, _ in sub for r in rr]
            w = np.concatenate([ww for _, ww in sub])
            if a.shuffle_control:
                w = rsh.permutation(w)
            x, lat = batch(recs)
            lp, _, kl, _, _ = trn.evaluate(x, lat, grad=True)
            wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
            loss = -(wt * lp).sum() / len(sub) + a.kl * kl.mean()
            kls.append(float(kl.mean().detach()))
            if a.kl_stop and len(kls) >= 5 and np.mean(kls[-5:]) >= a.kl_stop:
                stop = True
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if stop:
                print(f"  KL stop: {np.mean(kls[-5:]):.3f} >= {a.kl_stop} at step {len(kls)}",
                      flush=True)
                break
        g, se = gain(ho)
        gs, _ = gain(ho, shuffle=True)
        print(f"epoch {ep + 1}: E[A|π] held-out {g * 1e4:+.0f}$ (Δ {(g - g0) * 1e4:+.0f}$ ± "
              f"{se * 1e4:.0f}); shuffled A on held-out {gs * 1e4:+.0f}$; KL to start "
              f"{np.mean(kls):.4f}", flush=True)
    if a.save:
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save({"model": sd, "prior": C.load_prior(a.ckpt, "cpu")}, a.save)
        print(f"weights: {a.save}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ckpt", default=DEF_CKPT)
        p.add_argument("--opp-ckpt", default=None)
        p.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument("--out", default="")

    p = sub.add_parser("es-chaos"); common(p)
    p.add_argument("--seed0", type=int, default=60000)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--dirs", type=int, default=1)
    p.add_argument("--sigmas", default="0.001,0.01,0.05")
    p = sub.add_parser("es"); common(p)
    p.add_argument("--seed0", type=int, default=61000)
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--dirs", type=int, default=24)
    p.add_argument("--sigma", type=float, default=0.01)
    p = sub.add_parser("es-step")
    p.add_argument("--dump", required=True)
    p.add_argument("--ckpt", default=DEF_CKPT)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", required=True)
    p = sub.add_parser("rpi"); common(p)
    p.add_argument("--seeds", type=int, nargs=2, required=True)
    p.add_argument("--points", type=int, default=4)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--tries", type=int, default=16)
    p.add_argument("--tmin", type=int, default=24)
    p.add_argument("--tmax", type=int, default=700)
    p.add_argument("--short", type=int, nargs="?", const=max(HORIZONS), default=0,
                   help="future only up to T + N (no N — up to T + max(HORIZONS)): probes without final result")
    p.add_argument("--base-mode", default="BC", choices=("BC", "DET"),
                   help="what plays the base and futures: BC — ladder policy (gate sampled), "
                        "DET — fully deterministic (gate thresholded)")
    p = sub.add_parser("groups"); common(p)
    p.add_argument("--seeds", type=int, nargs=2, required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n-ex", type=int, default=8)
    p.add_argument("--base-mode", default="DET", choices=("BC", "DET"),
                   help="outside exploration turns: DET — like the main line's `collect._DET`")
    p = sub.add_parser("knob"); common(p)
    p.add_argument("--seed0", type=int, default=62000)
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--mode", default="BC", choices=("BC", "DET"))
    p.add_argument("arms", nargs="+", help="arms like sell:0.5,buy:-0.5")
    p = sub.add_parser("rpi-report")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("train")
    p.add_argument("--rpi", nargs="+", required=True, help="rpi points (train and held-out)")
    p.add_argument("--groups", nargs="*", default=[], help="train on groups, evaluate on rpi")
    p.add_argument("--ckpt", default=DEF_CKPT)
    p.add_argument("--loss", choices=("lin", "bon", "awr", "self"), default="lin")
    p.add_argument("--beta", type=float, default=0.1, help="AWR: temperature in --scale units")
    p.add_argument("--scale", type=float, default=1e4,
                   help="advantage unit in the loss, $ (the E[A|π] metric is always in $)")
    p.add_argument("--label", default="final",
                   help="final — final margin; 'n:q' — probe at T+n, q: 0 money, 1 liquid, 2 total")
    p.add_argument("--future", type=int, default=-1, help="train on a single future (-1 — mean)")
    p.add_argument("--held", type=float, default=0.3)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mkt-only", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--kl-stop", type=float, default=0.0,
                   help="stop when the running KL to start (5 steps) reaches the threshold: "
                        "arms are compared at an EQUAL step")
    p.add_argument("--shuffle-control", action="store_true")
    p.add_argument("--det-logp", choices=("drop", "keep"), default="drop",
                   help="drop — deterministic turns (alt 0, DET/BC mode) are excluded from log π, "
                        "self trains on PPO samples; keep — behaviour before 25.09 (biased)")
    p.add_argument("--save", default="")
    a = ap.parse_args()
    if a.cmd == "rpi-report":
        pts = [p for f in a.files for p in pickle.load(open(f, "rb"))]
        pts = [x for p in pts for x in (p["points"] if "points" in p else [p])]
        rpi_report(pts)
        probe_report(pts)
        return
    {"es-chaos": cmd_es_chaos, "es": cmd_es, "rpi": cmd_rpi, "groups": cmd_groups, "knob": cmd_knob, "es-step": cmd_es_step,
     "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
