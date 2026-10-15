"""Policy improvement by rollouts with common random numbers (CRN) -- v1.

Why (24.09, branching probe). Per-micro-turn PPO learns noise here
(signal per step 0.7%, critic EV ~0.1), and the group baseline removes only 29% of the variance.
Branching a game with CRN gives a reproducible difference of outcomes for ONE decision,
with a spread 1.4-4x smaller than the noise.
Hence the Rollout PI / Expert Iteration scheme: at a game point compare several
turn candidates by rollouts to the end, and train the policy toward the best -- supervised
learning per turn, without a critic and without credit over 719 turns.

  gen    Games of our policy (PPO mode, sampling) against a net opponent
         (BC, argmax + gate sampling). At P points of the game (turn T) -- a snapshot
         (`Env.clone()` + `_FLOW` of both modules), K DIFFERENT turn candidates
         (sampling with different streams, duplicates dropped; the first is the base,
         stream 0), each played to the end on m common streams of random
         numbers. The candidate label is the margin (own - opponent) on each stream.
  train  Split by games into training/held-out. The candidate advantage is the
         mean over streams of the paired difference "candidate margin - mean margin of
         the candidates on the same stream" (CRN: the future is common to all candidates),
         in units of $10k. Step: -Σ_c A_c · log π(c|s) + kl·KL(anchor) over points.
         Check on HELD-OUT points: how much the gain expected over
         candidates E_{π|candidates}[A] grew against the starting policy, and the same
         with shuffled A (control ≈ 0).

  python -u experiments/rl/rollout_pi.py gen --seeds 30000 30016 --out pts_0.pkl
  python -u experiments/rl/rollout_pi.py train --data 'pts_*.pkl'
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import pickle
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import collect as C  # noqa: E402
import play  # noqa: E402
import policy as POL  # noqa: E402

DEF_CKPT = os.path.join(ROOT, "experiments/net/runs/value_tower/e11.pt")


# ------------------------------------------------------------------ game

def _cfg():
    import kaggle_environments as ke
    return ke.make("kaggriculture", configuration={"episodeSteps": 720},
                   debug=False).configuration


def setup(ckpt: str, opp_ckpt: str):
    net = C.load_net(ckpt, "cpu")
    C.install(play, net, "cpu", C.decode_mode(POL.PPO), record=False,
              prior=C.load_prior(ckpt, "cpu"), who="our policy (PPO mode)")
    opp = C.fresh_play_module()
    onet = C.load_net(opp_ckpt, "cpu")
    C.install(opp, onet, "cpu", C.decode_mode(POL.BC), record=False,
              prior=C.load_prior(opp_ckpt, "cpu"), who="opponent (BC)")
    return opp


def call(mod, obs, cfg, seed, t, side, stream):
    # sampling randomness is a function of (game, turn, side, stream) and nothing
    torch.manual_seed(((seed * 1_000_003 + t) * 7 + side) * 101 + stream)
    return mod.agent(obs, cfg)


def snapshot(env, obs, opp):
    return (env.clone(), copy.deepcopy(obs), copy.deepcopy(play._FLOW),
            copy.deepcopy(opp._FLOW))


def restore(snap, opp):
    env, obs, f_me, f_op = snap
    play._FLOW.clear(); play._FLOW.update(copy.deepcopy(f_me))
    opp._FLOW.clear(); opp._FLOW.update(copy.deepcopy(f_op))
    return env.clone(), copy.deepcopy(obs)


def rollout(env, obs, T, cfg, opp, me, seed, stream_at, stream_after):
    """Play out from turn T: our stream `stream_at` at T, `stream_after` afterwards.
    -> (margin of our side, own money)."""
    t, rew = T, [0.0, 0.0]
    while True:
        acts = [None, None]
        acts[me] = call(play, obs[me], cfg, seed, t, me,
                        stream_at if t == T else stream_after)
        acts[1 - me] = call(opp, obs[1 - me], cfg, seed, t, 1 - me, 0)
        res = env.step(acts)
        obs = res["observations"]
        rew = [float(x or 0.0) for x in res["rewards"]]
        t += 1
        if res["done"]:
            return rew[me] - rew[1 - me], rew[me]


def candidates(snap, opp, cfg, seed, T, me, K, tries):
    """K DIFFERENT turns at a point. The first is stream 0 (base). -> list of
    (stream, action key, turn record)."""
    out, seen = [], set()
    for j in range(tries):
        stream = 0 if j == 0 else 1000 + j
        _, obs = restore(snap, opp)
        play._NET["record"] = []
        act = call(play, obs[me], cfg, seed, T, me, stream)
        rec = play._NET["record"].pop()
        play._NET["record"] = None
        key = json.dumps(act, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append((stream, key, {"enc": rec["enc"], "legal0": rec["legal0"],
                                  "lat": rec["lat"]}))
        if len(out) >= K:
            break
    return out


def gen(a):
    import kaggriculture_rs as K
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    cfg = _cfg()
    opp = setup(a.ckpt, a.opp_ckpt or a.ckpt)
    rng = random.Random(a.seeds[0] * 7919 + 17)
    pts, t0 = [], time.time()
    seeds = list(range(a.seeds[0], a.seeds[1]))
    for gi, seed in enumerate(seeds):
        me = seed % 2
        Ts = sorted(rng.sample(range(a.tmin, a.tmax), a.points))
        play._reset_flow(); opp._reset_flow()
        env = K.Env(config={"episodeSteps": 720}, seed=seed)
        obs = env.reset(seed)
        t = 0
        for T in Ts:
            while t < T:                                   # base game, stream 0
                acts = [None, None]
                acts[me] = call(play, obs[me], cfg, seed, t, me, 0)
                acts[1 - me] = call(opp, obs[1 - me], cfg, seed, t, 1 - me, 0)
                obs = env.step(acts)["observations"]
                t += 1
            snap = snapshot(env, obs, opp)
            cands = candidates(snap, opp, cfg, seed, T, me, a.k, a.tries)
            res = []
            for stream, key, rec in cands:
                ms, own = [], []
                for r in range(a.m):
                    e, o = restore(snap, opp)
                    mg, mn = rollout(e, o, T, cfg, opp, me, seed, stream, 5000 + r)
                    ms.append(mg); own.append(mn)
                res.append({"stream": stream, "key": key, "rec": rec,
                            "margin": ms, "money": own})
            pts.append({"seed": seed, "T": T, "me": me, "cands": res})
            # the base game continues from the snapshot
            env, obs = restore(snap, opp)
            spread = np.ptp([np.mean(c["margin"]) for c in res]) if len(res) > 1 else 0.0
            print(f"__PROG__ game {gi + 1}/{len(seeds)} seed {seed} T {T}: "
                  f"candidates {len(res)}, spread of mean margins ${spread:,.0f}, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        with open(a.out + ".tmp", "wb") as f:              # dump per game
            pickle.dump(pts, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(a.out + ".tmp", a.out)
    print(f"done: points {len(pts)} in {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ training

def advantages(pt, scale=1e4):
    """Paired (CRN) candidate gain: on each stream r subtract the mean
    candidate margin on THAT SAME stream, then average over streams."""
    M = np.array([c["margin"] for c in pt["cands"]], dtype=np.float64)   # [K, m]
    return (M - M.mean(0, keepdims=True)).mean(1) / scale


def train(a):
    import critic as CRIT
    import ppo as PPO
    from config import PPOCfg
    files = sorted(sum((glob.glob(p) for p in a.data), []))
    pts = [p for f in files for p in pickle.load(open(f, "rb")) if len(p["cands"]) > 1]
    seeds = sorted({p["seed"] for p in pts})
    rs = random.Random(0); rs.shuffle(seeds)
    held = set(seeds[:int(round(len(seeds) * a.held))]) if a.held > 0 else set()
    if not held or len(held) == len(seeds):
        raise SystemExit(f"games {len(seeds)}: with --held {a.held} either training "
                         "or held-out is empty -- more games needed")
    tr = [p for p in pts if p["seed"] not in held]
    ho = [p for p in pts if p["seed"] in held]
    print(f"points {len(pts)} from {len(files)} files; training {len(tr)}, held out {len(ho)}"
          f" ({len(held)} games)", flush=True)
    A_all = np.concatenate([advantages(p) for p in pts])
    print(f"candidate gain, $10k: std {A_all.std():.3f}, mean |A| {np.abs(A_all).mean():.3f}",
          flush=True)

    dev = a.device
    torch.manual_seed(a.seed)
    net = C.load_net(a.ckpt, dev)
    anchor = C.load_net(a.ckpt, dev)
    cfg = PPOCfg()
    cfg.kl_coef = a.kl
    tr_ = PPO.Trainer(net, anchor, CRIT.make("zero"), cfg, C.decode_mode(POL.PPO),
                      device=dev, prior=C.load_prior(a.ckpt, dev))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.0)

    def batch(ps):
        recs = [c["rec"] for p in ps for c in p["cands"]]
        x = {k: torch.from_numpy(np.stack([r["enc"][k] for r in recs])).to(dev)
             for k in recs[0]["enc"]}
        x["legal"] = torch.from_numpy(np.stack([r["legal0"] for r in recs])).to(dev)
        lat = POL.stack([r["lat"] for r in recs], dev)
        sizes = [len(p["cands"]) for p in ps]
        return x, lat, sizes

    def expected_gain(ps, shuffle=False):
        """E_{π|candidates}[A] over points: π renormalized to the set of
        point candidates. -> mean over points."""
        g, rsh = [], random.Random(1)
        net.eval()
        with torch.no_grad():
            for i in range(0, len(ps), a.bs):
                sub = ps[i:i + a.bs]
                x, lat, sizes = batch(sub)
                lp, _, _, _, _ = tr_.evaluate(x, lat, grad=False, want_kl=False)
                lp = lp.float().cpu().numpy()
                o = 0
                for p, n in zip(sub, sizes):
                    A = advantages(p)
                    if shuffle:
                        A = A[rsh.sample(range(n), n)]
                    w = np.exp(lp[o:o + n] - lp[o:o + n].max()); w /= w.sum()
                    g.append(float((w * A).sum())); o += n
        return float(np.mean(g))

    base_ho = expected_gain(ho)
    print(f"start: E[A|π] held-out {base_ho:+.4f}; training {expected_gain(tr):+.4f}",
          flush=True)
    for ep in range(a.epochs):
        rs.shuffle(tr)
        net.train()
        tot = 0.0
        for i in range(0, len(tr), a.bs):
            sub = tr[i:i + a.bs]
            x, lat, sizes = batch(sub)
            A = torch.as_tensor(np.concatenate([advantages(p) for p in sub]),
                                dtype=torch.float32, device=dev)
            if a.shuffle_control:
                A = A[torch.randperm(len(A), device=dev)]
            lp, _, kl, _, _ = tr_.evaluate(x, lat, grad=True)
            loss = -(A * lp).sum() / len(sub) + a.kl * kl.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(sub)
        ho_g = expected_gain(ho)
        print(f"epoch {ep + 1}: loss {tot / len(tr):+.4f}; E[A|π] held-out {ho_g:+.4f}"
              f" (Δ vs start {ho_g - base_ho:+.4f}); shuffled control"
              f" {expected_gain(ho, shuffle=True):+.4f}", flush=True)
    if a.save:
        torch.save({"model": net.state_dict(), "prior": C.load_prior(a.ckpt, "cpu")}, a.save)
        print(f"weights: {a.save}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--ckpt", default=DEF_CKPT)
    g.add_argument("--opp-ckpt", default=None)
    g.add_argument("--seeds", type=int, nargs=2, required=True, help="[start, end)")
    g.add_argument("--points", type=int, default=3)
    g.add_argument("--k", type=int, default=4)
    g.add_argument("--m", type=int, default=2)
    g.add_argument("--tries", type=int, default=24)
    g.add_argument("--tmin", type=int, default=240)
    g.add_argument("--tmax", type=int, default=700)
    g.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--ckpt", default=DEF_CKPT)
    t.add_argument("--device", default="cpu")
    t.add_argument("--held", type=float, default=0.25)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--bs", type=int, default=16, help="points per minibatch")
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--kl", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--shuffle-control", action="store_true",
                   help="train on shuffled A (control of the whole pipeline)")
    t.add_argument("--save", default="")
    a = ap.parse_args()
    (gen if a.cmd == "gen" else train)(a)


if __name__ == "__main__":
    main()
