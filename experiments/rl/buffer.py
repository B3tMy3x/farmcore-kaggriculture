"""Rollout: one turn is one transition, a game is one stream, GAE over the stream.

Our unit of experience is the whole TURN, not a single order: the net emits all ~29
factors in one pass, and the turn's `log π` is the sum over factors (`policy.log_prob`).
So PPO also works with one joint ratio per turn (H11: watch
`clip_frac`, and if large, split by factor groups).

Boundaries. A game is exactly 720 turns per side, truncation never happens: either the game
is played out or it broke and does not go into training. So `truncated` is present in the code
(GAE is incomplete without it), but in our collection it is always False -- and this
is checked, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class Turn:
    """One turn of one side. Everything without which the turn cannot be recomputed.

    `enc` and `legal0` are the net INPUT exactly as it was in the game;
    `lat` is the turn latent (the choices and all masks under which they were made). This
    triple is enough for `policy.log_prob` to give the same quantity that was
    computed in the rollout (R5).
    """

    enc: Dict[str, np.ndarray]
    legal0: np.ndarray
    lat: Dict[str, np.ndarray]
    # log π_old from the rollout is a CHECK, not training. `None` = on this turn
    # the check was not computed (`rollout.logp_every > 1`): the trainer recomputes
    # `log π_old` itself anyway, and the in-game value is only needed
    # to compare the game with the recomputation.
    logp: Optional[float]
    reward: float = 0.0         # trainer reward (not the "outcome", see RewardCfg)
    value: float = 0.0          # V(s) at collection time, from the frozen critic
    terminated: bool = False
    truncated: bool = False
    obs: Optional[dict] = None  # only with rollout.keep_obs
    # A sampled turn (`rollout.explore_turns`). False is a deterministic turn:
    # its advantage is zeroed, it does not enter the policy gradient.
    explore: bool = True


@dataclass
class Episode:
    """One side of one game -- a continuous stream of turns."""

    turns: List[Turn] = field(default_factory=list)
    seed: int = 0
    seat: int = 0
    opponent: str = ""
    money: float = 0.0
    opp_money: float = 0.0
    outcome: float = 0.0        # TRUE outcome ±1/0, separate from reward
    dead: bool = False          # removed by the environment: does not go into training
    # ⚠ How the sides were decoded. Without this field the rollout win rate is a mixture
    # of incomparable quantities: against a checkpoint we SAMPLE while it plays argmax,
    # and this asymmetry is measured here (H1/R4: -$1 784 median, 16/48), i.e.
    # equal strength gives a win rate of about 0.33, not 0.5. In mirror self-play
    # both sides are ours, and the two streams of one game give exactly +1 and -1 --
    # the win rate over them is identically 0.5, however much the policy improves.
    decode: str = "ppo"         # how we moved
    opp_decode: str = ""        # "bc" | "package" | "mirror"
    # Group collection (`rollout.group_size`). `index` is the global game number
    # in the rollout (it restores the order: with groups the environment seed
    # is shared by 2K games). `group` is the advantage group key, "environment seed:
    # side:opponent"; empty = the game is not in a group. `sample_seed` is what
    # seeded the sampling of our policy.
    index: int = -1
    group: str = ""
    sample_seed: int = -1
    # Town shops on the last turn of the stream, in order of opening (the i-th opened
    # at the end of day 3i+2). Exogenous randomness of the game: it shows where
    # the games of a group diverged by environment rather than by play.
    shops: tuple = ()

    def __len__(self) -> int:
        return len(self.turns)


def group_advantage(episodes: List["Episode"], mode: str, std: bool = False,
                    gamma: float = 1.0):
    """Group advantage per turn. -> (list of A arrays per stream,
    summary).

    A group is the streams with one `Episode.group` (one environment seed, one side,
    one opponent, different sampling seeds). The baseline is the mean of the OTHER
    group members (leave-one-out), so it does not depend on the
    stream's own actions and the gradient estimate is unbiased for any K ≥ 2.

      rloo   -- A_i = R_i - mean_{j≠i} R_j, R is the full return of the stream; one A for
               all turns of the stream.
      grpo_t -- A_{i,t} = G_{i,t} - mean_{j≠i} G_{j,t}, G_t is the return from turn t to
               the end (γ = `gamma`). Turn t is aligned by turn NUMBER in the stream; where
               other members have no turn t (different length), the baseline is over
               those that have it.

    `std` -- divide by the group std (ddof 0; for `grpo_t` -- at every turn t).
    A loner in a group (the neighbors' game was removed by the environment) gets A ≡ 0: there is nothing to
    compare it with, and this is counted in the summary, not hidden.
    """
    if mode not in ("rloo", "grpo_t"):
        raise ValueError(f"unknown group advantage {mode!r}")
    by: Dict[str, List[int]] = {}
    for i, ep in enumerate(episodes):
        if not ep.group:
            raise ValueError(
                f"stream {i} (seed {ep.seed}, side {ep.seat}) without a group: "
                f"`ppo.advantage = {mode}` requires `rollout.group_size > 1`")
        by.setdefault(ep.group, []).append(i)
    out: List[Optional[np.ndarray]] = [None] * len(episodes)
    sizes, single = [], 0
    for key, ids in by.items():
        sizes.append(len(ids))
        if len(ids) < 2:
            single += 1
            out[ids[0]] = np.zeros(len(episodes[ids[0]]), dtype=np.float64)
            continue
        rs = [np.asarray([t.reward for t in episodes[i].turns], dtype=np.float64)
              for i in ids]
        if mode == "rloo":
            if gamma != 1.0:
                R = np.array([float(np.sum(r * gamma ** np.arange(len(r))))
                              for r in rs])
            else:
                R = np.array([float(r.sum()) for r in rs])
            k = len(R)
            base = (R.sum() - R) / (k - 1)
            A = R - base
            if std:
                A = A / max(float(R.std()), 1e-8)
            for i, a in zip(ids, A):
                out[i] = np.full(len(episodes[i]), float(a), dtype=np.float64)
            continue
        # grpo_t: returns to the end, aligned by turn, NaN where there is no turn
        T = max(len(r) for r in rs)
        G = np.full((len(rs), T), np.nan)
        for j, r in enumerate(rs):
            g = np.zeros(len(r))
            acc = 0.0
            for t in range(len(r) - 1, -1, -1):
                acc = r[t] + gamma * acc
                g[t] = acc
            G[j, :len(r)] = g
        have = ~np.isnan(G)
        n_t = have.sum(0)                               # members with turn t
        s_t = np.where(have, G, 0.0).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            base = (s_t[None, :] - np.where(have, G, 0.0)) / (n_t[None, :] - 1)
            A = np.where(n_t[None, :] >= 2, G - base, 0.0)
            if std:
                mu = s_t / np.maximum(n_t, 1)
                var = np.where(have, (G - mu[None, :]) ** 2, 0.0).sum(0) \
                    / np.maximum(n_t, 1)
                A = A / np.maximum(np.sqrt(var), 1e-8)[None, :]
        for j, i in enumerate(ids):
            out[i] = A[j, :len(rs[j])].astype(np.float64)
    info = {"groups": len(by), "group_size_mean": float(np.mean(sizes)) if sizes else 0.0,
            "group_size_min": int(min(sizes)) if sizes else 0,
            "group_singletons": single}
    return out, info


def compute_gae(rewards, values, terminated, truncated, gamma=1.0,
                gae_lambda=0.95, bootstrap=0.0, horizon=0):
    """Advantages and value targets for ONE continuous stream. -> (adv, ret).

    Ported from the archive's `training/rollout.py` (with its shape and dtype checks) with
    one change of form: there `next_values` lay as a column for every transition,
    here they are taken by shifting `values`, because the stream is continuous and a second
    column cannot exist by construction.

    The distinction between terminal and truncation is essential and both sources keep it:
    at a TRUE terminal there is no bootstrap (`V(s_T) = 0`, the game ended, there will be
    no more money) and the recursion stops; at TRUNCATION there is a bootstrap, but
    the recursion stops anyway -- otherwise the advantage would leak into the cut-off
    future.

    `horizon > 0` is chunking from the archive's production `self_play_returns.py`: a chunk
    of length h, ordinary recursion inside, at the boundary a bootstrap from the FROZEN
    `values[end]`, the advantage accumulator is zeroed. ⚠ Off by default:
    the source's authors themselves write "adds bootstrap bias" and did not measure the effect,
    and our reward is terminal -- at h = 64 it does not reach the start of the game at all,
    except through the critic (R1).
    """
    v = np.asarray(values, dtype=np.float64)
    r = np.asarray(rewards, dtype=np.float64)
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    T = len(r)
    if not (len(v) == len(term) == len(trunc) == T):
        raise ValueError("GAE: stream lengths diverged")
    if T == 0:
        return np.zeros(0), np.zeros(0)
    nxt = np.concatenate([v[1:], np.array([float(bootstrap)])])
    adv = np.zeros(T, dtype=np.float64)
    size = int(horizon) if horizon and horizon > 0 else T
    for start in range(0, T, size):
        end = min(start + size, T)
        carry = 0.0
        for t in range(end - 1, start - 1, -1):
            # at a terminal there is no future; at a chunk boundary and at truncation
            # the value is taken but the recursion is cut
            boot = 0.0 if term[t] else nxt[t]
            delta = r[t] + gamma * boot - v[t]
            cont = 0.0 if (term[t] or trunc[t]) else 1.0
            carry = delta + gamma * gae_lambda * cont * carry
            adv[t] = carry
    return adv, adv + v


def _ev(target, pred) -> float:
    """1 - Var(target - estimate)/Var(target). NaN if the target does not vary."""
    var = np.asarray(target).var()
    if var <= 1e-12:
        return float("nan")
    return float(1.0 - (np.asarray(target) - np.asarray(pred)).var() / var)


class Batch:
    """The rollout laid out in tensors for the update.

    Built ONCE per iteration: advantages, targets and normalization are computed
    before epochs and minibatches, as in both sources. Recomputing them between
    epochs is not allowed -- that is no longer the same PPO.
    """

    def __init__(self, episodes: List[Episode], cfg, horizon: int = 0,
                 defer_values: bool = False):
        """`defer_values` (`ppo.prepass_value`): `t.value` is not set yet --
        the trainer's pre-pass will provide it via `set_values`, and only then
        are GAE, targets and everything depending on them computed. Until then the buffer has
        turns (`turns`, `pin`, `stack`) but no `adv`/`ret`/`values`."""
        self.episodes = [e for e in episodes if not e.dead and len(e)]
        self.turns: List[Turn] = [t for ep in self.episodes for t in ep.turns]
        self._cfg, self._horizon = cfg, horizon
        self.deferred = bool(defer_values)
        if not self.deferred:
            self._finish()

    def set_values(self, values) -> None:
        """V(s) of each turn in `turns` order -> finish GAE (`defer_values`).

        `float(v)` -- exactly how `collect.score` wrote them (`t.value = float(vv)`
        from a `float64` from `float32`), so the number in the turn is the same."""
        if not self.deferred:
            raise RuntimeError("set_values on a buffer without defer_values: V is already in the turns")
        v = np.asarray(values)
        if len(v) != len(self.turns):
            raise ValueError(f"V for {len(v)} turns, but the buffer has {len(self.turns)}")
        for t, vv in zip(self.turns, v):
            t.value = float(vv)
        self.deferred = False
        self._finish()

    def _finish(self) -> None:
        cfg, horizon = self._cfg, self._horizon
        self.turns = []
        adv_all, ret_all = [], []
        for ep in self.episodes:
            rw = [t.reward for t in ep.turns]
            vv = [t.value for t in ep.turns]
            tm = [t.terminated for t in ep.turns]
            tr = [t.truncated for t in ep.turns]
            if not tm[-1] and not tr[-1]:
                raise ValueError("the stream ends neither in a terminal nor in truncation: "
                                 "nothing to bootstrap GAE from")
            if any(tm[:-1]) or any(tr[:-1]):
                raise ValueError("terminal in the middle of a stream -- games got mixed")
            K = int(getattr(cfg, "window_k", 0) or 0)
            if K > 0:
                # Credit window (25.09): the turn's return is
                # the direct sum of ΔΦ over K turns WITHOUT bootstrap, the critic target is the same.
                # With γ = 1 and TD(λ) the critic learned on a target made of its own estimates
                # (EV 0.99 to it, EV -10 to the 24-turn sum, std V 13 versus 3.4
                # for the sum), and A = r + V(s′) - V(s) was a difference of two numbers of the
                # "to the end of the game" horizon -- noise: `snr` could not tell A from -A.
                r = np.asarray(rw, dtype=np.float64)
                c = np.concatenate([[0.0], np.cumsum(r)])
                i = np.arange(len(r)); j = np.minimum(i + K, len(r))
                g = c[j] - c[i]
                base = getattr(cfg, "window_baseline", "critic")
                if base == "turn_mean":
                    a = g.copy()          # baseline by turn number is below, over the buffer
                else:
                    a = g - np.asarray(vv, dtype=np.float64)
            else:
                a, g = compute_gae(rw, vv, tm, tr, cfg.gamma, cfg.gae_lambda,
                                   horizon=horizon)
            # The CRITIC target uses its own λ (`ppo.value_lambda`), if set. One λ
            # for both the advantage and the target forced a choice between "low
            # advantage variance" and "a target without its own estimates"; at
            # λ = 0.95 the rl1 critic learned to agree with itself (`rl1.sh`).
            vl = getattr(cfg, "value_lambda", -1.0)
            if K <= 0 and vl is not None and vl >= 0 and vl != cfg.gae_lambda:
                _, g = compute_gae(rw, vv, tm, tr, cfg.gamma, vl, horizon=horizon)
            adv_all.append(a)
            ret_all.append(g)
            self.turns += ep.turns
        mode = getattr(cfg, "advantage", "gae") or "gae"
        self.group_info = {}
        if mode != "gae":
            # The group advantage REPLACES GAE; the critic target (`ret`) stays as
            # it was, so that the critic can learn for the metrics (`ppo.advantage`).
            adv_all, self.group_info = group_advantage(
                self.episodes, mode, bool(getattr(cfg, "group_std", False)),
                cfg.gamma)
        self.adv_raw = (np.concatenate(adv_all) if adv_all else np.zeros(0))
        self.ret = (np.concatenate(ret_all) if ret_all else np.zeros(0))
        if int(getattr(cfg, "window_k", 0) or 0) > 0 and \
                getattr(cfg, "window_baseline", "critic") == "turn_mean":
            # a state-independent baseline: the window's mean return by
            # turn number in the game (over the whole buffer) -- for probes without a critic
            step = np.concatenate([np.arange(len(ep.turns)) for ep in self.episodes])
            mean_by = np.zeros(step.max() + 1)
            cnt = np.zeros(step.max() + 1)
            np.add.at(mean_by, step, self.adv_raw); np.add.at(cnt, step, 1)
            self.adv_raw = self.adv_raw - mean_by[step] / np.maximum(cnt[step], 1)
        self.values = np.array([t.value for t in self.turns], dtype=np.float64)
        # ⚠ The rollout `log π` is now NOT present for every turn. The mask must
        # travel with the array: without it a `nan` in `logp_roll` would silently poison
        # both the drift and (with `recompute_old_logp = False`) the ratio itself.
        self.logp_have = np.array([t.logp is not None for t in self.turns],
                                  dtype=bool)
        self.logp_roll = np.array([(t.logp if t.logp is not None else np.nan)
                                   for t in self.turns], dtype=np.float64)
        # explained variance is computed BEFORE normalization and over the whole buffer
        # (a rule from a previous project): after normalization it loses its meaning.
        self.explained_var = _ev(self.ret, self.values)
        # ⚠ That is not enough either. The TD(λ) target is itself BUILT from critic estimates: at λ = 0.95
        # and a reward only at the end, the terminal's contribution reaches turn t with weight
        # λ^(T-t), i.e. does not reach the start of the game at all, and `ret_t ≈ V_t`.
        # Then `explained_var` measures the critic's smoothness, not its correctness:
        # in the 23.09 smoke it gave 0.478 with `value_mse` 2.48 versus 1.00 for
        # V ≡ 0 and sign agreement 0.358. So a Monte Carlo is computed alongside
        # (λ = 1, γ = 1): the target is the sum of rewards to the end of the game, it contains
        # no critic estimates at all, and there is nothing for it to self-confirm.
        mc = []
        for ep in self.episodes:
            r = np.array([t.reward for t in ep.turns], dtype=np.float64)
            mc.append(np.cumsum(r[::-1])[::-1])
        self.ret_mc = (np.concatenate(mc) if mc else np.zeros(0))
        # ⚠ With a terminal reward the Monte Carlo target is CONSTANT within a game
        # (γ = 1, no rewards along the way), so all the variance of the denominator is
        # between games. A rollout where all games ended the same way gives a
        # near-zero denominator and a meaningless number: the 23.09 measurement on two
        # defeats in a row ($172 and $3 966 gap) gave -1277.7.
        #
        # ⚠ The threshold `std(ret_mc) > 0.1` stood here without justification, and guarded the wrong
        # quantity. The criterion is derived from the reward shape, not chosen:
        # `r_T = ±1 + α·tanh(Δ/σ)` at α = 0.3, i.e. between two WINS
        # the target differs by at most 2α = 0.6, and usually by hundredths (in the
        # 23.09 smoke two wins gave targets 1.0040 and 1.0015, Var ≈ 1.6e-6, and
        # `explained_var_mc` came out -536 264 with n = 2 "distinct" values).
        # Between a win and a defeat it differs by at least 2. So the
        # denominator is meaningful exactly when the rollout has games with
        # DIFFERENT outcome signs, and signs must be counted, not distinct numbers.
        # Fewer than two signs -- the quantity is undefined (NaN), not "very
        # bad". `n` and the spread are printed alongside: on two or three games the
        # confidence interval is wider than the quantity itself, and it must be read together
        # with them.
        ends = np.array([ep.turns[-1].reward for ep in self.episodes]
                        if self.episodes else [], dtype=np.float64)
        self.ret_mc_n = int(np.unique(np.sign(ends)).size) if len(ends) else 0
        self.ret_mc_std = float(self.ret_mc.std()) if len(self.ret_mc) else 0.0
        self.explained_var_mc = (_ev(self.ret_mc, self.values)
                                 if self.ret_mc_n >= 2 else float("nan"))
        adv = self.adv_raw.copy()
        # Sparse exploration: deterministic turns have no advantage
        # (turn probability 1), normalization is only over sampled turns.
        ex = np.array([getattr(t, "explore", True) for t in self.turns], dtype=bool)
        self.explore = ex
        if getattr(cfg, "adv_shuffle", False):
            # Noise arm (25.09): A is shuffled among sampled turns
            # WITHIN a group -- same distribution, no link to the action.
            # A run that is no better in money than such an arm learned nothing.
            rng = np.random.default_rng(int(abs(hash(tuple(
                e.seed for e in self.episodes))) % 2**32))
            gk = np.concatenate([[str(e.group or f"ep{i}")] * len(e)
                                 for i, e in enumerate(self.episodes)])
            for k in np.unique(gk):
                ix = np.flatnonzero((gk == k) & ex)
                adv[ix] = adv[rng.permutation(ix)]
        if cfg.normalize_adv and ex.sum() > 1:
            # Normalization over the WHOLE buffer once, not per minibatch: as in
            # both sources. Per minibatch the advantage sign would depend on the
            # random minibatch composition. eps via clamp_min, as in the archive.
            adv = (adv - adv[ex].mean()) / max(adv[ex].std(), 1e-8)
        adv[~ex] = 0.0
        if not ex.all() and ex.any():
            # Sparse exploration: the signal is on ~1% of turns, and a minibatch
            # mean would dilute it with zeros by a factor of N/n -- any other loss term
            # (KL to the anchor, entropy) would crush it, and AdamW would inflate
            # this mixture to an ordinary step. The scale gives the policy gradient its
            # weight back, as if the buffer held only sampled turns; the ratio
            # and its clip do not depend on the scale of A.
            if getattr(cfg, "explore_only", False):
                # Only sampled turns (25.09): the rest carry A = 0, and in a
                # minibatch of 256 with 8 sampled turns per game there turned out to be
                # 1-2 of them, and the Adam step was made by those 1-2 turns (rl4: gradient norm 17
                # with clipping at 0.5, 80% of steps clipped). Then no scale is needed.
                keep = np.flatnonzero(ex)
                self.turns = [self.turns[i] for i in keep]
                for k in ("adv_raw", "ret", "values", "logp_have", "logp_roll",
                          "explore", "ret_mc"):
                    setattr(self, k, getattr(self, k)[keep])
                self.adv = adv[keep]
                return
            adv[ex] *= len(adv) / float(ex.sum())
        self.adv = adv

    def __len__(self) -> int:
        return len(self.turns)

    # ---------------- buffer on the device (`ppo.resident_buffer`) ----------------

    _dev_x = None
    _dev_lat = None

    def pin(self, device) -> None:
        """Put the whole buffer on the device ONCE; afterwards `stack` and
        `stack_into` take a minibatch with `index_select` right there.

        ⚠ Bitwise parity. A repo rule (caught twice): what goes into torch is a
        FRESHLY ALLOCATED tensor, not a slice of someone else's buffer, otherwise a different
        alignment -> a different cuBLAS kernel -> ~1e-6 in `log π`. `index_select`
        does give a fresh tensor (or writes into a ready graph buffer) -- the same
        as `np.stack` + `.to()`. This is proved by a parity test, not by this
        comment.
        """
        import policy as POL
        if self._dev_x is not None:
            return
        t0 = self.turns[0]
        x = {k: torch.from_numpy(np.stack([t.enc[k] for t in self.turns])).to(device)
             for k in t0.enc}
        x["legal"] = torch.from_numpy(
            np.stack([t.legal0 for t in self.turns])).to(device)
        self._dev_x = x
        self._dev_lat = POL.stack([t.lat for t in self.turns], device)
        self._dev = torch.device(device)

    def pinned_bytes(self) -> int:
        if self._dev_x is None:
            return 0
        return sum(v.numel() * v.element_size()
                   for d in (self._dev_x, self._dev_lat) for v in d.values())

    def _idx_t(self, idx):
        if torch.is_tensor(idx):
            return idx.to(self._dev)
        return torch.as_tensor(idx, dtype=torch.long, device=self._dev)

    def stack(self, idx, device="cpu"):
        """Indices -> (net input, latent) as a batch for recomputation."""
        import policy as POL  # local: the path is set by collect/train
        if self._dev_x is not None and torch.device(device) == self._dev:
            it = self._idx_t(idx)
            return ({k: v.index_select(0, it) for k, v in self._dev_x.items()},
                    {k: v.index_select(0, it) for k, v in self._dev_lat.items()})
        sub = [self.turns[i] for i in idx]
        x = {k: torch.from_numpy(np.stack([t.enc[k] for t in sub])).to(device)
             for k in sub[0].enc}
        x["legal"] = torch.from_numpy(
            np.stack([t.legal0 for t in sub])).to(device)
        lat = POL.stack([t.lat for t in sub], device)
        return x, lat

    def stack_into(self, idx, x_dst, lat_dst) -> None:
        """The same, but INTO ready buffers: under a CUDA graph the input must lie at the same
        addresses from measurement to measurement.

        ⚠ The values must come out the same as in `stack`, and this is not
        self-evident: the repo twice caught a divergence of `log π` in the sixth digit
        precisely because a tensor arrived in torch with a different alignment and
        cuBLAS picked a different kernel. Here the source is a fresh `np.stack` (as in
        `stack`), and the destination is a persistent buffer of the same shape and dtype;
        bitwise parity is proved by a parity test, not
        assumed.
        """
        if self._dev_x is not None:
            it = self._idx_t(idx)
            for k, dst in x_dst.items():
                torch.index_select(self._dev_x[k], 0, it, out=dst)
            for k, dst in lat_dst.items():
                torch.index_select(self._dev_lat[k], 0, it, out=dst)
            return
        sub = [self.turns[i] for i in idx]
        for k, dst in x_dst.items():
            src = (np.stack([t.legal0 for t in sub]) if k == "legal"
                   else np.stack([t.enc[k] for t in sub]))
            dst.copy_(torch.from_numpy(src))
        for k, dst in lat_dst.items():
            dst.copy_(torch.from_numpy(
                np.stack([np.asarray(t.lat[k]) for t in sub])))
