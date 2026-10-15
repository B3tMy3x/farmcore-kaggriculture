"""Lockstep: K games step in sync, the net is called ONCE with batch K.

Why. Measured 23.09 (A100, load 3.3 of 15, OMP_NUM_THREADS=1): one self-play
decision costs 29.95 ms, of which 15.4 ms is a batch-1 net pass on a CPU core.
The same net at batch 256 on the A100 costs 0.065 ms per state, i.e. 238x
cheaper. The GPU is not used by the current collector at all: batch 1 on the
A100 (12.4 ms) is even slower than a CPU core, and `collect_parallel` rightly
spreads games over CPU processes. For the GPU to pay off the batch must be
large, and a large batch only comes from SEVERAL games at once.

What this is NOT. A second turn implementation. There is no own decoder,
encoder or `log π` here: the game is still played by the unmodified
`play.agent`, and all this module does is intercept ONE call
`_NET["m"](...)` inside it. That works because `_NET["m"]` is a plain object
the loop puts into the play module (`collect.install`), and `play.agent` only
calls it. We swap the object, not the code.

How it works. K games run in K threads, an unmodified `play.agent` per side.
On reaching the net, a thread puts its input into `NetBatcher` and sleeps.
When all live threads are asleep, the last one does ONE pass on the
concatenated batch, slices the output per game and wakes the others.

⚠ MEASURED 23.09, AND THE MEASUREMENT DID NOT CONFIRM THIS DESIGN. A100,
15 cores, load 2.6-4.7, OMP_NUM_THREADS=1, 120-step games, Rust port env:

| K | decisions/s | in net, ms/decision | rest (python), ms/decision |
|---|---|---|---|
| 1 | 44.2 | 16.25 | 6.4 |
| 4 | 43.0 | 4.40 | 18.9 |
| 8 | 37.4 | 2.33 | 24.4 |
| 16 | 37.3 | 1.23 | 25.6 |
| 32 | 40.2 | 0.65 | 24.2 |

Batching works exactly as intended: the net pass cost falls as 1/K, from 16.25
to 0.65 ms per decision. But the python part of the turn GROWS with K — 6.4 ms
with one game vs 24-26 ms with eight or more — and eats the whole gain to zero.

The cause is the GIL, but not the obvious way. `sys.setswitchinterval` from
0.005 to 0.05 changes NOTHING (44.2/43.0/37.4/37.3 vs 44.2/42.5/37.6/37.5), so
forced switches are not the issue. Voluntary ones are: every tensor op in
`policy.family`, `Sampler`, `run_turn` and `log_prob` releases and reacquires
the GIL, and there are hundreds per turn. With one ready thread reacquiring is
free; with K ready threads it is lock contention on every op.

Consequence: batch across PROCESSES, not threads. An inference server holds the
GPU, J worker processes each play a game with unmodified code and send it
their input. But even that won't pay off while a pass costs ~12.4 ms of fixed
overhead regardless of batch, because the worker would wait those 12.4 ms
instead of computing.

Those 12.4 ms are not arithmetic but kernel launches and python inside
`model.forward`, and a CUDA graph removes them. Measured 23.09, A100, direct
`torch.cuda.CUDAGraph` capture (inductor does not build on that machine —
triton calls gcc with `-lcuda` and it fails; the graph needs no compiler):

| batch | plain pass | CUDA graph | speedup |
|---|---|---|---|
| 1 | 12.45 ms | **1.81 ms** | 6.9 |
| 8 | 13.57 | 2.67 | 5.1 |
| 32 | 13.98 | 4.10 | 3.4 |
| 64 | 14.00 | 5.98 | 2.3 |
| 128 | 13.96 | 8.79 | 1.6 |
| 256 | 16.56 | 15.64 | 1.06 |

So the fixed part drops from 12.4 to 1.8 ms, and at batch 256 the graph gives
nothing — there the GPU is finally computing, not launching. Hence CUDA graphs
are not the last step, as planned, but the FIRST: without them cross-process
batching hits the fixed cost of a pass.

What the 6.4 ms python floor per decision consists of (measured 23.09, CPU
core, per turn): `policy.log_prob` 1.60 ms, `Sampler` + `run_turn` 1.01 ms,
`features.encode` 1.55 ms, `copy.deepcopy(obs)` for recording 0.69 ms,
`policy.family` 0.30 ms, legality mask 0.09 ms, port step 0.05 ms. The first
hurts most: `log π` is computed at batch 1 only so the PPO ratio has a
behaviour-policy denominator — and the trainer recomputes it from the latent
anyway, in batch.

The module is kept working and verified (internal parity script, K = 1, 0
mismatches over 2 876 turns) as a measurement rig and as the ready half of a
future inference server: `NetBatcher` is its core, only the transport changes
(condition variable -> shared memory).

⚠ Lockstep is NOT bitwise identical to serial collection, and must not be
presented as such. Two reasons, both measured by an internal batch-parity
script: a GEMM at batch K sums in a different order than at batch 1, and there
is one generator per process that threads draw from interleaved. So lockstep is
for COLLECTING experience, where a stream of on-policy games is needed, not
per-seed reproducibility. Acceptance (`collect_eval`) and any money comparison
of versions (`tools/bench/pair.py`) stay serial: there a seed must give the
same number.

⚠ The Rust env port (`--env rust`) is verified on OUR rollouts by an internal
env-parity script: 43 130 transitions and 86 400 observations, 0 mismatches
apart from two DECLARED ones. Do not put the port into the loop without that
check.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import collect as C  # noqa: E402


# ------------------------------------------------------------------ batcher

class NetBatcher:
    """One net pass for all games waiting at the barrier.

    The barrier is "all live", not "exactly K": a game may end before its
    neighbours, and waiting for it would hang. `leave` decrements the live count
    and releases the rest if they are already waiting.
    """

    def __init__(self, net, device: str, n_live: int, timeout: float = 120.0,
                 io_device: str = "cpu"):
        self.net = net
        self.dev = torch.device(device)
        # ⚠ Input and output live on `io_device`; the net computes on `device`.
        # Otherwise each game would push its dozen batch-1 tensors over PCIe
        # (`play.agent: tt()`), and lockstep would trade one expensive pass for
        # K·12 small transfers. This way exactly one concatenated batch crosses
        # the bus each way. Invisible to the game: it gets tensors on its own
        # device as before.
        self.io = torch.device(io_device)
        self.cv = threading.Condition()
        self.pending: List[list] = []        # [key, args, kwargs, result]
        self.live = n_live
        self.timeout = timeout
        self.calls = 0                       # total net passes
        self.states = 0                      # states through them
        self.hist: Dict[int, int] = {}       # batch size histogram
        self.t_fwd = 0.0                     # seconds inside the net pass
        self.t_wait = 0.0                    # seconds asleep at the barrier, all threads
        self.err: Optional[BaseException] = None

    # --- call from a game thread ---

    def __call__(self, *args, **kwargs):
        slot = [threading.get_ident(), args, kwargs, None]
        with self.cv:
            self.pending.append(slot)
            if len(self.pending) >= self.live:
                self._flush()
            else:
                t0 = time.time()
                while slot[3] is None and self.err is None:
                    if not self.cv.wait(timeout=self.timeout):
                        self.err = RuntimeError(
                            f"lockstep hung: {len(self.pending)} at barrier with "
                            f"{self.live} live games, {time.time() - t0:.0f} s")
                        self.cv.notify_all()
                self.t_wait += time.time() - t0
            if self.err is not None:
                raise self.err
            return slot[3]

    def leave(self) -> None:
        """A game ended: fewer live games, the barrier may now be full."""
        with self.cv:
            self.live -= 1
            if self.pending and len(self.pending) >= self.live:
                self._flush()
            else:
                self.cv.notify_all()

    # --- the pass itself ---

    def _flush(self) -> None:
        """Concatenate, run, slice, wake. Called under `self.cv`."""
        batch = self.pending
        self.pending = []
        tf = time.time()
        try:
            n = len(batch)
            args0, kw0 = batch[0][1], batch[0][2]
            # Positional args are [1, ...] tensors; concatenate along axis 0.
            args = [torch.cat([b[1][i] for b in batch], 0).to(self.dev)
                    for i in range(len(args0))]
            kw = {}
            for k, v in kw0.items():
                if torch.is_tensor(v):
                    kw[k] = torch.cat([b[2][k] for b in batch], 0).to(self.dev)
                else:
                    # Non-tensor settings must match across games:
                    # if they differ, these are different passes, not one batch.
                    for b in batch:
                        if b[2][k] != v:
                            raise RuntimeError(
                                f"differing values of {k} in batch: {v!r} and {b[2][k]!r}")
                    kw[k] = v
            with torch.no_grad():
                out = self.net(*args, **kw)
            out = {k: v.to(self.io) for k, v in out.items()}
            for j, b in enumerate(batch):
                b[3] = {k: v[j:j + 1] for k, v in out.items()}
            self.calls += 1
            self.states += n
            self.hist[n] = self.hist.get(n, 0) + 1
        except BaseException as e:      # noqa: BLE001 — otherwise all threads hang
            self.err = e
        finally:
            self.t_fwd += time.time() - tf
            self.cv.notify_all()

    def stats(self) -> Dict[str, float]:
        tot = sum(self.hist.values()) or 1
        return {"passes": self.calls, "states": self.states,
                "mean batch": self.states / max(self.calls, 1),
                "batch=1 share": self.hist.get(1, 0) / tot,
                "s in net": self.t_fwd, "s asleep (sum over threads)": self.t_wait}


# ------------------------------------------------------------------ env

class RustEnvRunner:
    """A game on the Rust port. Same contract as `collect.KaggleEnvRunner`.

    The configuration is taken from `kaggle_environments` ONCE per process and
    passed to agents as is: `play.agent` reads it via `envcfg.configure`, and a
    hand-built dict would let five engine values drift from those baked into
    the weights. The port takes only `episodeSteps` from it.

    ⚠ The port does not execute agents, so it cannot report "removed from the
    game": an agent exception is caught here and counted as that agent's loss,
    as the harness would.
    """

    name = "kaggriculture_rs"
    _cfg = None

    def __init__(self, episode_steps: int = 720, native_encode: bool = False):
        self.steps = episode_steps
        # Native feature encoder (`encode.rs`). Installed into the slot
        # `features.NATIVE` for the duration of a game: `play.agent` does not
        # and should not know about it — it calls `features.encode` as before.
        self.native_encode = native_encode
        if RustEnvRunner._cfg is None:
            import kaggle_environments as ke
            RustEnvRunner._cfg = ke.make(
                "kaggriculture", configuration={"episodeSteps": episode_steps},
                debug=False).configuration

    def run(self, agents, seed: int):
        import kaggriculture_rs as K
        env = K.Env(config={"episodeSteps": self.steps}, seed=seed)
        obs = env.reset(seed)
        cfg = RustEnvRunner._cfg
        restore = self._install_native(env) if self.native_encode else None
        try:
            return self._play(env, obs, agents, cfg)
        finally:
            if restore is not None:
                restore()

    def _install_native(self, env):
        """Install `features.NATIVE` for the game. -> how to restore it.

        ⚠ The slot is process-wide, and `encode` may be called for something
        other than this game (replay parsing, a probe). So the closure CHECKS
        the observation's step against the engine's and returns None on
        mismatch — then `features.encode` computes itself, as before. Silently
        returning features of another state is exactly the class of bug we
        guard against here.
        """
        import numpy as np
        import features
        from features import PRODUCTS

        lay = {n: (o, k) for n, o, k in env.__class__.encode_layout()}
        rows = {"products": len(PRODUCTS), "crops": 5, "animals": 3}

        def native(obs, p, extra):
            o = obs.get("step")
            st = int(o) if o is not None else int(obs["day"]) * 24 + int(obs["hour"])
            if st != env.current_step():
                return None
            e = extra or {}
            ex = []
            for pref, dflt in (("pi_", 0.0), ("oppsold_", 0.0),
                               ("mysold_", 0.0), ("crowd_", 0.5)):
                ex += [float(e.get(pref + g, dflt)) for g in PRODUCTS]
            flat = np.frombuffer(env.encode_all(p, ex), dtype=np.float32)
            out = {}
            for n, (off, k) in lay.items():
                v = flat[off:off + k]
                if n in ("tiles_me", "tiles_op"):
                    v = v.reshape(-1, 40)
                elif n == "units":
                    v = v.reshape(24, k // 24)
                elif n in rows:
                    v = v.reshape(rows[n], k // rows[n])
                elif n in ("tile_mask", "unit_mask"):
                    out[n] = v.astype(bool)     # astype copies anyway
                    continue
                # ⚠ The COPY is required, and not for mutability. A
                # `np.frombuffer` slice sits at an arbitrary byte offset, while
                # the python path's `np.stack` returns a FRESH aligned array.
                # Same values, different alignment, and torch below picks a
                # different kernel — `log π` drifts by 1e-6 (caught by the
                # internal parity check: 2 693 mismatches, all in `logp`, with
                # bitwise-identical `enc`). Same pitfall as the server's flat
                # input. The copy costs a 36 KB memcpy vs the 1.5 ms the
                # encoder saves.
                # ⚠ Specifically `.copy()`: `ascontiguousarray` on an already
                # contiguous slice returns THE SAME array, alignment unchanged
                # (checked — still the same 2 693 mismatches).
                out[n] = v.copy()
            return out

        prev = features.NATIVE
        features.NATIVE = native
        return lambda: setattr(features, "NATIVE", prev)

    def _play(self, env, obs, agents, cfg):
        dead = [False, False]
        rew = [0.0, 0.0]
        for _ in range(self.steps - 1):
            acts = []
            for i in (0, 1):
                if dead[i]:
                    acts.append({"farmer": ["PASS"], "hands": [], "market": []})
                    continue
                try:
                    acts.append(agents[i](obs[i], cfg))
                except Exception:                       # noqa: BLE001
                    traceback.print_exc()
                    dead[i] = True
                    acts.append({"farmer": ["PASS"], "hands": [], "market": []})
            res = env.step(acts)
            obs = res["observations"]
            rew = [float(x or 0.0) for x in res["rewards"]]
            if res["done"]:
                break
        return rew, dead


def make_runner(kind: str, episode_steps: int = 720, native_encode: bool = False):
    if kind == "rust":
        return RustEnvRunner(episode_steps, native_encode=native_encode)
    if kind == "kaggle":
        return C.KaggleEnvRunner(episode_steps)
    raise SystemExit(f"unknown env: {kind}")


# ------------------------------------------------------------------ collection

def collect_lockstep(net, plan, cfg, env_kind: str = "rust",
                     progress: bool = True, offset: int = 0,
                     device: Optional[str] = None,
                     episode_steps: int = 720) -> List:
    """K games of the plan in one process, net in batch. -> list of `Episode`.

    `plan` is the same as for `collect.collect_serial`: a list of opponents, one
    per game. `device` overrides `cfg.rollout.device`: batched collection makes
    sense on the GPU, measuring the rest — on CPU.

    Game order in the result is restored by index: threads finish out of order,
    and `offset` sets both the seed and the parity (which seat we play), so the
    order must not be lost — seat alternation depends on it.
    """
    dev = device or cfg.rollout.device
    import policy as POL
    if C.ppo_decode(cfg) != POL.PPO or getattr(cfg.rollout, "crn", False):
        raise SystemExit("lockstep only plays `policy.PPO` without CRN")
    if int(getattr(cfg.rollout, "group_size", 1) or 1) > 1:
        raise SystemExit("lockstep does not support group collection (`rollout.group_size`): "
                         "it has one generator per process, a game's sampling seed "
                         "cannot be set — the group would be wrong")
    k = len(plan)
    if k == 0:
        return []
    # Games live on CPU, the net computes on `dev` (see `NetBatcher.io`).
    io = "cpu"
    batcher = NetBatcher(net, dev, k, io_device=io)
    # Teacher log-frequencies and opponent weights — ONCE per rollout, not per
    # game: `torch.load` of a 40 MB checkpoint for each of K games would cost
    # more than the whole batching gain. Same cache as in `collect_serial`
    # (`cache` there lives for the whole plan).
    cache: Dict[str, Any] = {}
    out: Dict[int, List] = {}
    errs: Dict[int, BaseException] = {}
    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def one(j: int, opponent):
        idx = offset + j
        try:
            # Own copy of `play.py` per game: `_NET` and `_FLOW` in it are global,
            # and a shared module would give the games one pressure accumulator.
            mod = C.fresh_play_module()
            # 720 is the same default `collect` uses for its own
            # `KaggleEnvRunner`: game length does not live in `cfg.rollout`.
            # Only speed MEASUREMENTS use shorter games, and then decisions per
            # second are comparable only with an equally shortened run: late in
            # the season there are more units and orders, and decisions cost more.
            runner = make_runner(env_kind, episode_steps)
            eps = _play_one(mod, batcher, opponent, idx, cfg, io, runner,
                            cache, lock)
            with lock:
                out[j] = eps
        except BaseException as e:                       # noqa: BLE001
            with lock:
                errs[j] = e
            traceback.print_exc()
        finally:
            batcher.leave()
            with lock:
                done[0] += 1
                if progress:
                    el = time.time() - t0
                    print(f"__PROG__ lockstep {done[0]}/{k} games, {el:.0f} s, "
                          f"mean batch {batcher.stats()['mean batch']:.1f}",
                          flush=True)

    ths = [threading.Thread(target=one, args=(j, opponent), daemon=True)
           for j, opponent in enumerate(plan)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise list(errs.values())[0]
    res = []
    for j in range(k):
        res.extend(out.get(j, []))
    st = batcher.stats()
    el = time.time() - t0
    if progress:
        print(f"lockstep: {k} games in {el:.1f} s, "
              f"net passes {st['passes']}, states {st['states']}, "
              f"mean batch {st['mean batch']:.1f}, "
              f"single-item batches {st['batch=1 share'] * 100:.1f}%", flush=True)
    # ⚠ Printed ALWAYS: without the share spent in the net, "lockstep gave
    # nothing" is indistinguishable from "lockstep gave, but something else ate it".
    print(f"  of which in net pass {st['s in net']:.1f} s ({st['s in net'] / el * 100:.1f}%), "
          f"per decision {st['s in net'] / max(st['states'], 1) * 1e3:.2f} ms",
          flush=True)
    collect_lockstep.last_stats = st
    return res


def _play_one(mod, batcher, opponent, index: int, cfg, dev: str, runner,
              cache: Dict[str, Any], cache_lock):
    """One game with its own copy of `play.py`, with the net pass intercepted.

    Mirrors `collect.play_episode` only in seat setup — but does NOT duplicate
    the decoder or labelling: the collected records go back through
    `_episodes_from`, so slicing into experience streams and reward are computed
    by one function both here and in serial collection.
    """
    import policy as POL
    rc = cfg.rollout
    decode = C.decode_mode(POL.PPO)

    def cached(key, make):
        # The lock covers both read and write: games start simultaneously, and
        # without it K threads would all load the same checkpoint at once.
        with cache_lock:
            if key not in cache:
                cache[key] = make()
            return cache[key]

    prior = cached("prior:" + (cfg.ckpt or ""),
                   lambda: C.load_prior(cfg.ckpt, dev) if cfg.ckpt else None)
    C.install(mod, batcher, dev, decode, record=True, prior=prior,
              slim=not rc.keep_obs, who=f"trained policy ({cfg.ckpt})",
              logp_every=getattr(rc, "logp_every", 1))
    # The net is replaced by the batcher: `play.agent` calls `_NET["m"](...)` and
    # does not know there is a barrier behind the object.
    mod._NET["m"] = batcher
    mod._reset_flow()

    mirror = opponent.kind == "self"
    if mirror:
        opp_agent = mod.agent
        opp_mod = None
    elif opponent.kind == "package":
        opp_mod = None
        opp_agent = cached("pkg:" + opponent.path,
                           lambda: mod._package_agent(opponent.path))
    else:
        opp_mod = C.fresh_play_module()
        C.install(opp_mod, batcher, dev, C.decode_mode(POL.BC), record=False,
                  prior=cached("prior:" + opponent.path,
                               lambda: C.load_prior(opponent.path, dev)),
                  who=f"league opponent {opponent.key} ({opponent.path})")
        opp_mod._NET["m"] = batcher
        opp_mod._reset_flow()
        opp_agent = opp_mod.agent

    me = 0 if index % 2 == 0 else 1
    agents = [mod.agent, opp_agent] if me == 0 else [opp_agent, mod.agent]
    seed = rc.seed0 + index
    # Per-game generator seeding — as in `collect.play_episode`. At K = 1 this
    # gives a BITWISE match with serial collection, which the internal parity
    # check relies on. At K > 1 there is still one generator per process and
    # threads draw from it interleaved: lockstep has no per-seed
    # reproducibility and does not claim it (see module header).
    torch.manual_seed(seed)
    rew, dead = runner.run(agents, seed)
    recs = mod._NET.pop("record", [])
    return _episodes_from(recs, rew, dead, seed, me, opponent, cfg, mirror, decode)


def _episodes_from(recs, rew, dead, seed, me, opponent, cfg, mirror, decode):
    """Turn records -> experience streams. Exactly the same parsing as in
    `collect.play_episode`; factored out here only because there it is baked
    into the function that runs the game itself."""
    import copy
    from buffer import Episode, Turn
    rc = cfg.rollout
    by_seat: Dict[int, list] = {}
    for r in recs:
        by_seat.setdefault(int(r["obs"]["player"]), []).append(r)
    seats = sorted(by_seat) if (mirror and rc.both_seats) else [me]
    kind = {"self": "mirror", "package": "package"}.get(opponent.kind, "bc")
    our = "ppo" if decode["sample_units"] > 0 else "bc"

    out = []
    for seat in seats:
        rr = sorted(by_seat.get(seat, []), key=lambda r: C._step_of(r["obs"]))
        if rc.max_turns:
            rr = rr[:rc.max_turns]
        if not rr:
            continue
        margins = [float(r["obs"]["farms"][seat]["money"]
                         - r["obs"]["farms"][1 - seat]["money"]) for r in rr]
        margins.append(float(rew[seat] - rew[1 - seat]))
        ep = Episode(seed=seed, seat=seat, opponent=opponent.key,
                     money=float(rew[seat]), opp_money=float(rew[1 - seat]),
                     dead=bool(dead[seat]), decode=our, opp_decode=kind)
        d = ep.money - ep.opp_money
        ep.outcome = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
        rws = C._turn_rewards(len(rr), ep.money, ep.opp_money, margins, cfg.reward)
        for i, r in enumerate(rr):
            ep.turns.append(Turn(
                enc=r["enc"], legal0=r["legal0"], lat=r["lat"], logp=r["logp"],
                reward=rws[i], terminated=(i == len(rr) - 1), truncated=False,
                obs=(copy.deepcopy(r["obs"]) if rc.keep_obs else None)))
        out.append(ep)
    return out
