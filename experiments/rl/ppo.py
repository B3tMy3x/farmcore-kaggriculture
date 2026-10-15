"""PPO update over a rollout of turns.

Base: `training/ppo.py` from an earlier codebase for the same game (clip, order,
metrics, shapes). On top: tricks from a previous project of ours: dual-clip, a
`target_kl` gate on EVERY minibatch, zeroing non-finite gradients, a skipped
minibatch counter, PPG. Plus our own review rules: fp32 without autocast (N3),
`log π_old` recomputed by the trainer (N3), analytic KL to the anchor over the
latent (N9).

One joint ratio per TURN. A turn is ~29 factors and the turn's `log π` is their
sum, so the ratio r = exp(Σ Δ log π_i) swings more than in single-action tasks.
Two consequences, both handled: dual-clip stops being decoration, and
`clip_frac` becomes a first-class metric (N11) — if it creeps up, the ratio
will have to be taken per factor group with a shared advantage.

⚠ **Measurement 23.09: the culprit is ONE factor, not the sum of twenty-nine.**
Smoke on `value_tower/e11`, first epoch: overall `clip_frac` 0.083-0.193, of
which group `tgt` (unit target pointer) 0.083-0.146, `op` 0.031-0.042, and
market gates, market quantities, unit quantities and land — EXACTLY 0.000.
`max |log r|` 0.37-0.61, i.e. the ratio stays below 1.85 and dual-clip at
c = 3.0 binds no turn (`ratio_gt_c` = 0.0000). Group `tgt` `clip_frac` can even
be HIGHER than overall (0.146 vs 0.135) — other factors partly cancel the
pointer's movement. The breakdown is computed by `group_logp`, and the sum of
groups is checked against the total `log π` (self-test, blocks F and E).
"""
from __future__ import annotations

import math
import os
import sys
import time as _t
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments/net"))

import policy as POL  # noqa: E402
import value_head as VH  # noqa: E402


def policy_loss(logp, logp_old, adv, clip_range, dual_clip=0.0):
    """Clipped surrogate with a lower bound for A < 0. -> (loss, kl, clip_frac, extra).

    Dual-clip (Ye et al., JueWu; verbatim from a previous project): `min(rA, clip(r)A)`
    is unbounded below — for A < 0 and large r the term rA goes to −∞ faster than
    clip(r)A "rescues" it, and min picks exactly that. The bound c·A (c > 1)
    fixes this for A < 0 only, leaving positive advantages alone.

    `approx_kl` is Schulman's k3 estimator `(r − 1) − log r`: non-negative and
    lower variance than `−log r`. Both sources use it.

    The fourth value is what `clip_frac` alone does not show (N11):
    `max |log r|` (how far the worst turn flies: the clip cuts the turn's WHOLE
    contribution, and the fraction doesn't tell whether by 0.21 or by 3), the
    fraction of turns with ratio above `dual_clip` (where the lower bound is
    load-bearing, not a safety net), and the fraction where the dual-clip bound
    actually bound the surrogate.
    """
    if not (logp.shape == logp_old.shape == adv.shape):
        raise ValueError("PPO: exactly one joint log π per turn is required")
    log_ratio = logp - logp_old
    ratio = log_ratio.exp()
    plain = torch.minimum(ratio * adv,
                          ratio.clamp(1 - clip_range, 1 + clip_range) * adv)
    obj = plain
    dual_bound = torch.zeros((), device=logp.device)
    if dual_clip and dual_clip > 1.0:
        obj = torch.where(adv < 0, torch.maximum(plain, dual_clip * adv), plain)
        dual_bound = ((adv < 0) & (dual_clip * adv > plain)).float().mean()
    approx_kl = ((ratio - 1) - log_ratio).mean()
    clip_frac = ((ratio - 1).abs() > clip_range).float().mean()
    extra = {
        "max_abs_log_ratio": log_ratio.detach().abs().max(),
        "ratio_gt_c": ((ratio > (dual_clip if dual_clip > 1.0 else float("inf")))
                       .float().mean().detach()),
        "dual_clip_frac": dual_bound.detach(),
        "clip_hi": ((ratio - 1) > clip_range).float().mean().detach(),
        "clip_lo": ((ratio - 1) < -clip_range).float().mean().detach(),
    }
    return -obj.mean(), approx_kl, clip_frac, extra


# Splitting the turn's joint `log π` into factor groups (N11). A single
# `clip_frac` over the sum of ~29 factors does not say WHICH factor drives it:
# a turn with one runaway market quantity and a turn where all unit targets
# moved by half a logit give the same ratio. Groups are computed NOT by a
# second copy of the formula but by the same `policy.log_prob` on a truncated
# latent: all factors except one group are switched off. Hence the acceptance
# check — the sum of groups must match the total `log π` bitwise (self-test,
# block F).
GROUPS = ("tgt", "op", "u_qty", "mkt_gate", "mkt_qty", "land")


def group_logp(out, lat, dec):
    """Turn `log π` per factor group. -> {name: [B]}. Sum = total `log π`.

    The latent is cut, not the logits: `log_prob` adds a factor's contribution
    only where its flag is set in the latent (`u_live`, `op_on`, `q_on`,
    `m_max > 0`, `land_ok`), so zeroing a flag disables exactly one group and
    nothing else. Temperatures stay the same as in the rollout.
    """
    su, sq = dec["sample_units"], dec["sample_qty"]
    z = lambda k: torch.zeros_like(lat[k])
    cut = lambda **kw: {**lat, **kw}
    # sample=False stops `log_prob` before market and land — nothing to switch off there
    g = {
        "tgt": POL.log_prob(out, cut(op_on=z("op_on")), False, su, 0.0),
        "op": POL.log_prob(out, cut(u_live=z("u_live")), False, su, 0.0),
        "u_qty": POL.log_prob(out, lat, False, 0.0, sq),
    }
    no_units = cut(u_live=z("u_live"), op_on=z("op_on"), q_on=z("q_on"))
    # Gates and quantities share ONE τ = sq (oh3, 25.09), and gates cannot be
    # computed separately from quantities: the market group is one — `mkt_qty`;
    # `mkt_gate` is kept at zero to preserve the old metric names.
    both = POL.log_prob(out, {**no_units, "land_ok": z("land_ok")}, True, 0.0, sq)
    g["mkt_gate"] = torch.zeros_like(both)
    g["mkt_qty"] = both
    g["land"] = POL.log_prob(out, {**no_units, "m_max": z("m_max")}, True, 0.0, sq)
    return g


# Anchor output heads read by `policy.kl` (`policy._walk`). The anchor cache
# (`ppo.anchor_cache`) stores exactly these, already mapped to the family.
KL_KEYS = ("target_logits", "op_logits", "qty_logits", "sell_logits",
           "hire_logits", "seed_logits", "buy_logits", "animal_logits",
           "land_logits")


def _as_logits(d, spec):
    """Dict of tensors -> `policy.Logits` with a family tag.

    The anchor cache stores `policy.family` output per head; `kl` accepts only
    `Logits`, and the tag must match that of the live pass.
    """
    o = POL.Logits(d)
    o.spec = spec
    return o


def value_loss(v, ret, v_old, clip=0.0):
    """MSE; with `clip > 0` — pessimistic clip around the old estimate.

    No clip by default: Andrychowicz et al. and the earlier codebase agree on
    "MSE without Huber and without value-function clipping"; only the previous
    project clipped. The flag is kept because the sources disagreed.
    """
    if clip and clip > 0:
        vc = v_old + (v - v_old).clamp(-clip, clip)
        return 0.5 * torch.maximum((v - ret).square(), (vc - ret).square()).mean()
    return 0.5 * (v - ret).square().mean()


# Order of numbers in the graph's static metrics vector. One `.cpu()` per
# minibatch instead of thirteen `float()`: each `float(tensor)` is a separate
# device sync, i.e. a full pipeline drain. Values are bitwise the same: only
# the way they are fetched changes.
MET = ("policy_loss", "value_loss", "entropy", "kl_anchor", "approx_kl",
       "clip_frac", "clip_hi", "clip_lo", "ratio_gt_c", "dual_clip_frac",
       "max_abs_log_ratio", "grad_norm", "grad_norm_value", "finite")


class _StepGraph:
    """CUDA graph for ONE update minibatch: captured once, then replayed.

    Inside: NO gradient zeroing (the graph writes them fresh, see below),
    policy forward, anchor forward under `no_grad`, `log π`, entropy, KL,
    critic head, both losses, `backward`, `nan_to_num_` on gradients and norm
    clipping over two groups. Only `AdamW.step` stays outside: it runs foreach
    and costs a handful of kernel launches, while `capturable = True` would need
    its own proof (it computes the bias correction on device, not as a Python
    float, and bitwise parity with the current step is not guaranteed).

    Why gradients are not zeroed. `zero_grad(set_to_none=True)` is called
    before capture, so `.grad` is BORN during capture, and the accumulation
    node stores a tensor there rather than adding to an existing one. On
    replay the same kernels write to the same memory — documented
    `torch.cuda.graph` behaviour ("the captured backward refills static .grad
    tensors in place"). Hence the restriction: after capture you must not call
    `zero_grad(set_to_none=True)` — `.grad` becomes `None`, the graph keeps
    writing into detached memory, and the optimizer sees nothing.

    ⚠ Shape and addresses are fixed: input is copied into THESE buffers, output
    is read from THESE. A partial minibatch doesn't fit and takes the old path.
    """

    def __init__(self, tr, x, lat, mb: int, ref=None, groups: bool = False):
        self.tr, self.mb = tr, int(mb)
        dev = tr.device
        # Group breakdown of `clip_frac` (N11) INSIDE the capture
        # (`ppo.graph_groups`): previously it ran after replay outside the
        # graph — six `log_prob` kernel launches and twelve `float()` per
        # first-epoch minibatch. In the capture it runs on EVERY minibatch (one
        # graph), but is read only in the first epoch.
        self.groups = bool(groups)
        self.names = MET + ((tuple(f"gc/{g}" for g in GROUPS)
                             + tuple(f"gm/{g}" for g in GROUPS))
                            if self.groups else ())
        self.gold = (torch.zeros(len(GROUPS), int(mb), dtype=torch.float32,
                                 device=dev) if self.groups else None)
        # Anchor cache: anchor output arrives ready in these buffers, no
        # anchor pass in the capture (`ppo.anchor_cache`).
        self.ref = (None if ref is None else
                    {k: v.clone() for k, v in ref.items()})
        # Input buffers are clones of the first real minibatch: same view,
        # dtype and device. Clone rather than `empty_like` so warmup runs on
        # meaningful numbers (on garbage `log_prob` may give inf and warmup
        # would take a different branch).
        self.x = {k: v.clone() for k, v in x.items()}
        self.lat = {k: v.clone() for k, v in lat.items()}
        f32 = dict(dtype=torch.float32, device=dev)
        self.logp_old = torch.zeros(self.mb, **f32)
        self.adv = torch.zeros(self.mb, **f32)
        self.ret = torch.zeros(self.mb, **f32)
        self.v_old = torch.zeros(self.mb, **f32)
        # Warmup on a side stream is mandatory: the first call of each kernel
        # triggers lazy init (cuBLAS, workspaces), which would end up in the
        # capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                tr.opt.zero_grad(set_to_none=True)
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        tr.opt.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out, self.met = self._body()
        # ⚠ The check is "gradients were born", not "born for all": under PPO
        # the BC auxiliary heads (`head_econ`, `head_effect`) take no part in
        # `log π` and get no gradient by construction — exactly as in the old
        # path (self-test, block E). Empty for ALL means a broken capture.
        self.with_grad = sum(1 for p in tr.params if p.grad is not None)
        if not self.with_grad:
            raise RuntimeError(
                "after graph capture no parameter has `.grad`: "
                "replay would write into nowhere")
        self.replays = 0

    def _body(self):
        tr, cfg = self.tr, self.tr.cfg
        ref = None if self.ref is None else _as_logits(self.ref, tr.fam_spec)
        lp, ent, kl, v, out = tr.evaluate(self.x, self.lat, grad=True, ref=ref)
        pl, akl, cf, ex = policy_loss(lp, self.logp_old, self.adv,
                                      cfg.clip_range, cfg.dual_clip)
        vl = (value_loss(v.float(), self.ret, self.v_old, cfg.value_clip)
              if v is not None else torch.zeros((), device=tr.device))
        ent_m, kl_m = ent.mean(), kl.mean()
        loss = (pl + cfg.value_coef * vl - cfg.ent_coef * ent_m
                + tr.kl_coef * kl_m)
        # ⚠ Non-finite minibatch. The old path checks `isfinite` BEFORE
        # `backward` and skips the whole minibatch; the graph must run the same
        # kernels every time, so the pass always runs and the decision is made
        # outside from the `finite` flag. Weights are unaffected: `opt.step` is
        # skipped and the next replay overwrites the gradients.
        finite = torch.isfinite(loss).to(torch.float32)
        loss.backward()
        for p in tr.params:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        gp, gv = tr._clip_t()
        rows = [
            pl.detach(), vl.detach(), ent_m.detach(), kl_m.detach(),
            akl.detach(), cf.detach(), ex["clip_hi"], ex["clip_lo"],
            ex["ratio_gt_c"], ex["dual_clip_frac"], ex["max_abs_log_ratio"],
            gp, gv, finite]
        if self.groups:
            # Same arithmetic as the `update` loop outside the graph: `val − gold`,
            # count of |Δ| > log(1 + clip) as a float32 sum (exact for integers
            # up to mb) and the max. Output `out` is the same as previously read
            # after replay.
            thr = math.log1p(cfg.clip_range)
            cnt, mx = [], []
            with torch.no_grad():
                g = group_logp(out, self.lat, tr.dec)
                for j, k in enumerate(GROUPS):
                    dd = (g[k].float() - self.gold[j]).abs()
                    cnt.append((dd > thr).float().sum())
                    mx.append(dd.max())
            rows += cnt + mx
        met = torch.stack(rows).float()
        return out, met

    def run(self, batch, idx, sel, logp_old, adv, ret, v_old, ref_cache=None,
            gold=None):
        """Input -> static buffers -> replay. -> (network output, CPU metrics)."""
        batch.stack_into(idx, self.x, self.lat)
        if self.ref is not None:
            for k, dst in self.ref.items():
                torch.index_select(ref_cache[k], 0, sel, out=dst)
        if self.groups and gold is not None:
            for j, g in enumerate(GROUPS):
                torch.index_select(gold[g], 0, sel, out=self.gold[j])
        self.logp_old.copy_(logp_old[sel])
        self.adv.copy_(adv[sel])
        self.ret.copy_(ret[sel])
        self.v_old.copy_(v_old[sel])
        self.graph.replay()
        self.replays += 1
        return self.out, self.met.cpu()


class _PreGraph:
    """CUDA graph of the `log π_old` PRE-pass (N3(2)) and the group breakdown.

    A separate graph, not part of the step: this pass covers the WHOLE buffer
    exactly once per update, has no gradient, and the `target_kl` gate does not
    cut it — so with a short epoch loop it is most of the update. Inside is
    exactly what the old path computed: `evaluate(grad=False, want_kl=False)`
    and, if `group_diag` is on, six `log_prob` on a truncated latent.

    Output is one [1 + 6, B] matrix: `log π` in the first row, then groups in
    `GROUPS` order. One tensor because it is consumed on device (sliced into
    the accumulator), so there is no point splitting it into seven transfers.
    """

    def __init__(self, tr, x, lat, mb: int, groups: bool, anchor: bool = False,
                 value: bool = False):
        self.tr, self.mb, self.groups = tr, int(mb), bool(groups)
        self.with_anchor = bool(anchor)
        # `ppo.prepass_value`: critic V as the LAST output row. The tower was
        # already computed in this pass (`critic.wants_value`), it just wasn't
        # read; the row is added only under the flag — without it the graph is
        # unchanged.
        self.with_value = bool(value)
        self.x = {k: v.clone() for k, v in x.items()}
        self.lat = {k: v.clone() for k, v in lat.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.res, self.ref = self._body()
        self.replays = 0

    def _body(self):
        lp, _, _, v, out0 = self.tr.evaluate(self.x, self.lat, grad=False,
                                             want_kl=False)
        rows = [lp.float()]
        if self.groups:
            g = group_logp(out0, self.lat, self.tr.dec)
            rows += [g[k].float() for k in GROUPS]
        if self.with_value:
            rows.append(v.float())
        ref = self.tr.anchor_out(self.x) if self.with_anchor else None
        return torch.stack(rows), ref

    def run(self, batch, idx):
        batch.stack_into(idx, self.x, self.lat)
        self.graph.replay()
        self.replays += 1
        return self.res, self.ref


class DualAdamW:
    """Two AdamW — policy and critic — behind one optimizer interface.

    ⚠ Why (rl1, 23.09). A single AdamW over policy and critic steps ALL
    parameters whose `.grad` is not `None`, and the step graph zeroes gradients
    IN PLACE — so policy grads are zero tensors, not `None`. The critic phase
    (`_value_phase`, epochs outside the gate) called the shared `step()`, and
    Adam moved policy weights by residual MOMENTUM: one real policy step, a
    `target_kl` gate cut, then ~118 critic steps — and every policy weight
    drifted ~30·lr in the same direction. Result: KL to anchor 14.8,
    max |log r| 17, rollout win rate 0/84 after one update. In critic warmup
    (`epochs = 0`) the same mechanism accumulated `step = 4720` with zero
    policy moments.

    `step()` — both (PPO epochs train policy and critic with one loss),
    `step_value()` — critic only. `param_groups` — concatenation, so an lr edit
    on resume reaches both groups.
    """

    def __init__(self, p_policy, p_value, value_lr=None, **kw):
        self.pol = torch.optim.AdamW(p_policy, **kw)
        # `value_lr` — the critic's own lr (`ppo.value_lr`); None or < 0 — shared
        kv = dict(kw)
        if value_lr is not None and value_lr >= 0:
            kv["lr"] = float(value_lr)
        self.val = torch.optim.AdamW(p_value, **kv) if p_value else None
        self._n_pol = len(p_policy)

    @property
    def param_groups(self):
        return self.pol.param_groups + (self.val.param_groups if self.val else [])

    @property
    def state(self):
        """Per-parameter merge, like `Optimizer.state` (read by `train_parity`)."""
        return {**self.pol.state, **(self.val.state if self.val else {})}

    def step(self) -> None:
        self.pol.step()
        if self.val is not None:
            self.val.step()

    def step_value(self) -> None:
        if self.val is not None:
            self.val.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.pol.zero_grad(set_to_none=set_to_none)
        if self.val is not None:
            self.val.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"dual": True, "policy": self.pol.state_dict(),
                "value": self.val.state_dict() if self.val else None}

    def load_state_dict(self, sd: dict, legacy_order=None) -> None:
        """Own state — as is. State of a SINGLE AdamW (pre-split) — by
        `legacy_order` indices (the old `Trainer.params`): each entry goes to
        its own optimizer. Entries with zero `exp_avg_sq` at `step > 0` are
        dropped: they are exactly the artefact of the shared optimizer (steps
        with zero gradient), and carrying them over would repeat the rl1 bug."""
        if sd.get("dual"):
            self.pol.load_state_dict(sd["policy"])
            if self.val is not None and sd.get("value") is not None:
                self.val.load_state_dict(sd["value"])
            return
        if legacy_order is None:
            raise ValueError("single-AdamW state: `legacy_order` is required")
        grp = {k: v for k, v in sd["param_groups"][0].items() if k != "params"}
        pol_ids = {id(p): j for j, p in enumerate(self.pol.param_groups[0]["params"])}
        val_ids = ({id(p): j for j, p in enumerate(self.val.param_groups[0]["params"])}
                   if self.val else {})
        out = {"pol": {}, "val": {}}
        dropped = 0
        for i, st in sd["state"].items():
            p = legacy_order[int(i)]
            if float(st["step"]) > 0 and float(st["exp_avg_sq"].abs().max()) == 0:
                dropped += 1
                continue
            if id(p) in pol_ids:
                out["pol"][pol_ids[id(p)]] = st
            elif id(p) in val_ids:
                out["val"][val_ids[id(p)]] = st
        self.pol.load_state_dict({"state": out["pol"], "param_groups": [
            {**grp, "params": list(range(len(pol_ids)))}]})
        if self.val is not None:
            self.val.load_state_dict({"state": out["val"], "param_groups": [
                {**grp, "params": list(range(len(val_ids)))}]})
        print(f"optimizer: single AdamW state split in two "
              f"(policy {len(out['pol'])}, critic {len(out['val'])}); "
              f"dropped {dropped} entries with zero moments", flush=True)


class Trainer:
    """Policy + anchor + critic -> one update per rollout."""

    def __init__(self, net, anchor, critic, cfg, decode, device="cpu",
                 optimizer=None, ppg=None, reward=None, prior=None):
        """`decode` — the rollout decoder mode (`policy.PPO`): the temperatures
        the turn was sampled with. Recomputation MUST use the same ones,
        otherwise `log π` is computed under a different distribution — the same
        class of bug as `legal`/`legal_now`.

        `reward` — the rollout's `RewardCfg`. Needed for exactly one check: GAE
        subtracts the critic estimate from the reward, and if they are on
        different scales the advantage is wrong while metrics look healthy.
        """
        self.net, self.anchor, self.critic = net, anchor, critic
        if reward is not None and getattr(critic, "trainable", False):
            self._check_scale(reward, critic)
        # Margin shaping is built for γ = 1 (`collect._turn_rewards`: adds
        # `Φ(s') − Φ(s)` without discount). For γ ≠ 1 Ng's theorem requires
        # `γ·Φ(s') − Φ(s)`, and without that factor the optimum shifts silently
        # — the same class of bug as a mismatched reward scale.
        if reward is not None and float(getattr(reward, "cash_shaping", 0.0)) \
                and float(cfg.gamma) != 1.0:
            raise ValueError(
                f"cash_shaping = {reward.cash_shaping} with gamma = {cfg.gamma}: "
                "shaping in `collect._turn_rewards` is built without discount (γ = 1). "
                "Either γ = 1, or fix the potential formula there")
        self.cfg, self.dec, self.device = cfg, dict(decode), device
        self.ppg = ppg
        # The distribution family (N2/N4) must be THE SAME as in the rollout:
        # `policy.family` is the single point where β·prior and the gate floor
        # are added to the logits, and `log_prob` without its tag fails loudly.
        # The numbers arrive in `decode` via the same merge as in play
        # (`collect.decode_mode`).
        self.prior = prior
        self.fam = (float(self.dec.get("beta", POL.FAMILY["beta"])),
                    float(self.dec.get("floor", POL.FAMILY["floor"])))
        self.fam_spec = self.fam
        self._ref = None         # anchor output cache for the update
        if anchor is not None:
            # ⚠ requires_grad is NOT turned off on the anchor, deliberately.
            # Measurement 23.09 (self-test): with `requires_grad_(False)` the
            # same input and bitwise-identical weights give logits differing by
            # 3e-6 — torch picks a different attention kernel in the corrector,
            # and KL to the anchor on the anchor itself comes out 1.3e-07
            # instead of zero. With requires_grad = True the difference is
            # exactly 0. The anchor still accumulates no gradient: its pass runs
            # under `torch.no_grad()`, and its parameters are not in the
            # optimizer (`self.params` below takes only policy and critic).
            anchor.eval()
        if getattr(cfg, "relbias_onehot", False):
            # The flag is set on BOTH networks: the anchor computes KL and its
            # output stays bitwise the same, while the collection network lives
            # in other processes and doesn't see it (forward does not depend on
            # the flag).
            import model as M
            for m_ in [net] + ([anchor] if anchor is not None else []):
                for mod in m_.modules():
                    if isinstance(mod, M.RelBias):
                        mod.onehot = True
        params = list(net.parameters())
        seen = {id(p) for p in params}
        params += [p for p in critic.params() if id(p) not in seen]
        self.params = params
        # `fused_adamw`: one kernel for the whole step instead of a foreach
        # chain. Same formula, its own op order inside the kernel; bitwise
        # parity is checked by `train_parity.py --test ppo.fused_adamw=true`,
        # not assumed.
        fused = bool(getattr(cfg, "fused_adamw", False)) and str(device).startswith("cuda")
        vset0 = {id(p) for p in critic.params()}
        self.opt = optimizer or DualAdamW(
            [p for p in params if id(p) not in vset0],
            [p for p in params if id(p) in vset0],
            value_lr=getattr(cfg, "value_lr", -1.0),
            lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay,
            **({"fused": True} if fused else {}))
        self.kl_coef = cfg.kl_coef
        self.updates = 0
        # Norm clipping over TWO groups, not one shared list. The critic has
        # its own tower (R1), its input is detached, but `clip_grad_norm_` over
        # all parameters at once computes ONE norm: a large critic gradient
        # would silently shrink the policy step. Split the same way in
        # `bc_train` (`model.policy_parameters` / `value_parameters`). If the
        # critic has no parameters (`zero`), the second group is empty and
        # behaviour is unchanged.
        vset = {id(p) for p in critic.params()}
        self.p_value = [p for p in params if id(p) in vset]
        self.p_policy = [p for p in params if id(p) not in vset]
        self._graph = None       # step CUDA graph, captured lazily
        self._pre = None         # CUDA graph of the `log π_old` pre-pass

    # ---------------- step CUDA graph ----------------

    def _graph_ok(self) -> bool:
        """Whether the graph may be captured. Refusal comes WITH A MESSAGE, not silently."""
        cfg = self.cfg
        if not getattr(cfg, "cuda_graph", False):
            return False
        why = ""
        if not str(self.device).startswith("cuda"):
            why = f"trainer is not on GPU ({self.device})"
        elif not torch.cuda.is_available():
            why = "CUDA unavailable"
        elif int(cfg.grad_accum) > 1:
            why = (f"grad_accum = {cfg.grad_accum}: the graph writes gradients "
                   "fresh, it cannot accumulate")
        elif cfg.kl_target > 0:
            why = (f"kl_target = {cfg.kl_target}: the leash coefficient changes "
                   "between updates, but is a constant in the capture")
        if why:
            print(f"⚠ training-step CUDA graph NOT enabled: {why}",
                  file=sys.stderr, flush=True)
            return False
        return True

    def _ensure_graph(self, batch, dev):
        if self._graph is not None or not self._graph_ok():
            return self._graph
        mb = int(self.cfg.minibatch)
        if len(batch) < mb:
            return None
        x, lat = batch.stack(list(range(mb)), dev)
        ref = (None if self._ref is None else
               {k: v[:mb] for k, v in self._ref.items()})
        t0 = _t.time()
        self._graph = _StepGraph(self, x, lat, mb, ref,
                                 groups=bool(getattr(self.cfg, "graph_groups", False)
                                             and self.cfg.group_diag))
        print(f"training-step CUDA graph captured in {_t.time() - t0:.1f} s "
              f"(minibatch {mb}); partial minibatches take the old path",
              flush=True)
        return self._graph

    def _ensure_pre_graph(self, batch, dev):
        if self._pre is not None or not self._graph_ok():
            return self._pre
        mb = int(self.cfg.minibatch)
        if len(batch) < mb:
            return None
        x, lat = batch.stack(list(range(mb)), dev)
        t0 = _t.time()
        self._pre = _PreGraph(self, x, lat, mb, bool(self.cfg.group_diag),
                              self._anchor_cached(),
                              value=bool(getattr(self.cfg, "prepass_value", False)))
        print(f"`log π_old` pre-pass CUDA graph captured in "
              f"{_t.time() - t0:.1f} s", flush=True)
        return self._pre

    def _zero_grad(self) -> None:
        """⚠ With a live graph — zero IN PLACE, not `set_to_none`.

        The graph was captured with `.grad = None`, so the accumulation node
        stores its own tensor there, and replay writes exactly into that
        memory. Detaching `.grad` would mean the graph writes into nowhere and
        the optimizer steps on nothing. Numbers are identical: `0 + g = g`
        bitwise.
        """
        self.opt.zero_grad(set_to_none=self._graph is None)

    @staticmethod
    def _check_scale(reward, critic):
        """Reward and estimate must live on one scale. Refusal, not a warning.

        `value_scalar` gives (P(W) − P(L)) + α·E[tanh(Δ/σ)], and the reward is
        sign(Δ)·outcome_scale + α·tanh(Δ/σ). They match only at
        `outcome_scale = 1`, the same α and the same σ the HL head was trained
        with. σ is taken not from the default but from the critic checkpoint's
        own `cfg` (`collect.ckpt_value_scale`): `bc_train.py --value-sigma`
        overrides it, and then the default would lie.
        """
        dense = (float(getattr(reward, "margin_delta_scale", 0.0) or 0.0) > 0
                 or float(getattr(reward, "phi_delta_scale", 0.0) or 0.0) > 0)
        if getattr(critic, "free_scale", False):
            return          # the scalar head learns on whatever return there is
        if dense:
            raise ValueError(
                "dense reward (`reward.margin_delta_scale` / `phi_delta_scale`) with a "
                "W/D/L+HL critic: its V is confined to ±(1+α) and describes the terminal "
                "reward. Use `--critic tower_scalar`")
        bad = []
        c_alpha = float(getattr(critic, "alpha", VH.RL_ALPHA))
        c_sigma = float(getattr(critic, "sigma", VH.RL_SIGMA))
        if float(reward.outcome_scale) != 1.0:
            bad.append(f"outcome_scale = {reward.outcome_scale} ≠ 1 "
                       "(V's outcome contribution is exactly P(W) − P(L))")
        if float(reward.cash_alpha) != c_alpha:
            bad.append(f"reward α {reward.cash_alpha} ≠ critic α {c_alpha}")
        if float(reward.cash_sigma) != c_sigma:
            bad.append(f"reward σ {reward.cash_sigma} ≠ σ the critic was trained "
                       f"with, {c_sigma}")
        if bad:
            raise ValueError(
                "reward and critic scales diverged, GAE would subtract one from "
                "the other incorrectly: " + "; ".join(bad))

    # ---------------- forward pass ----------------

    def _forward(self, net, x, want_value):
        # `vcrit` — critic-only columns; they don't feed the policy, but without
        # them the model refuses to compute the tower (zeros would be a silent
        # substitution). Ignored when `want_value=False`.
        return net(x["tiles_me"].float(), x["tiles_op"].float(),
                   x["units"].float(), x["unit_mask"],
                   x["products"].float(), x["glob"].float(),
                   crops=x["crops"].float(), animals=x["animals"].float(),
                   tile_mask=x["tile_mask"], legal=x["legal"],
                   vcrit=(x["vcrit"].float() if "vcrit" in x else None),
                   want_value=want_value)

    def _anchor_cached(self) -> bool:
        """Whether the anchor cache is on for THIS trainer (flag + anchor + leash)."""
        # With `epochs = 1` the anchor is computed exactly once per turn anyway
        # — the cache only moves it from the step to the pre-pass (measured
        # 23.09, mb 256: pre-pass 2.31 -> 3.39 min per 1e6 at the same total),
        # so it is not enabled there. Numbers don't depend on it: parity is bitwise.
        return bool(getattr(self.cfg, "anchor_cache", False)
                    and int(self.cfg.epochs) > 1
                    and self.anchor is not None and self.cfg.kl_coef > 0)

    def anchor_out(self, x):
        """Anchor output for `KL_KEYS` heads, already in the family. No gradient.

        Exactly the same expression as in `evaluate`: `family(forward(anchor))`
        under `no_grad`. A separate function so the pre-pass and the step don't
        keep two copies of the formula.
        """
        with torch.no_grad():
            ref = POL.family(self._forward(self.anchor, x, False),
                             self.prior, *self.fam)
        return {k: ref[k] for k in KL_KEYS}

    def evaluate(self, x, lat, grad=True, want_kl=None, ref=None):
        """Recompute a turn: -> (log π, entropy, KL to anchor, V, network output).

        ⚠ N3: fp32 only and NO autocast. bf16 gives |Δ log π| p99 0.295 — 6.4%
        of turns exceed clip 0.2 from precision alone, and the earlier codebase
        rejected a bf16 run on A100 for the same reason (drift max 0.169). The
        check lives here, not in a comment: people turn autocast on without
        looking.
        """
        if self.cfg.fp32 and torch.is_autocast_enabled():
            raise RuntimeError("log π recomputation under autocast: see N3, fp32 only")
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            raw = self._forward(self.net, x, self.critic.wants_value)
            # Logits are mapped to the played family ONCE, before everything
            # else: rollout sampling saw exactly these.
            out = POL.family(raw, self.prior, *self.fam)
            lp = POL.log_prob(out, lat, self.dec["sample"],
                              self.dec["sample_units"], self.dec["sample_qty"])
            # ⚠ With `ent_coef = 0` (our default: the regulariser is the leash
            # to the anchor, not entropy) entropy is needed ONLY as a metric,
            # and computing it with a graph is not just waste. `0.0 · ent` in the
            # loss is zero in value, but in the gradient it is
            # `0 · ∂ent/∂logit`, and once the entropy derivative becomes inf you
            # get nan. Exactly this class already cost us an update in which 11
            # of 275 tensors moved (`policy._safe`). No coefficient — no graph.
            if self.cfg.ent_coef:
                ent = POL.entropy(out, lat, self.dec["sample"],
                                  self.dec["sample_units"], self.dec["sample_qty"])
            else:
                with torch.no_grad():
                    ent = POL.entropy(out, lat, self.dec["sample"],
                                      self.dec["sample_units"],
                                      self.dec["sample_qty"])
            kl = torch.zeros_like(lp)
            need = self.cfg.kl_coef > 0 if want_kl is None else want_kl
            if self.anchor is not None and need:
                # `ref` given — it is the anchor cache from the pre-pass
                # (`anchor_cache`), don't compute it again
                if ref is None:
                    with torch.no_grad():
                        ref = POL.family(self._forward(self.anchor, x, False),
                                         self.prior, *self.fam)
                kl = POL.kl(out, ref, lat, self.dec["sample"],
                            self.dec["sample_units"], self.dec["sample_qty"])
            v = self.critic.head(out, x)
        return lp, ent, kl, v, out

    # ---------------- update ----------------

    def update(self, batch) -> Dict[str, float]:
        """One PPO pass over the rollout. The buffer is consumed (no longer on-policy)."""
        cfg = self.cfg
        if getattr(cfg, "step_mode", "ppo") == "kl_line":
            return self._update_line(batch)
        t_start = _t.time()
        N = len(batch)
        if N == 0:
            raise ValueError("empty rollout")
        dev = self.device
        # `ppo.prepass_value`: turn V values don't exist yet — the pre-pass
        # below provides them, and only then does the buffer finish GAE
        # (`Batch.set_values`).
        deferred = bool(getattr(batch, "deferred", False))
        if deferred != bool(getattr(cfg, "prepass_value", False)):
            raise RuntimeError(
                f"buffer {'without' if deferred else 'with'} per-turn V with "
                f"`ppo.prepass_value = {getattr(cfg, 'prepass_value', False)}`: "
                "V for GAE would be computed by either nobody or both")
        if deferred and not self.critic.wants_value:
            raise RuntimeError("`ppo.prepass_value` without a value tower: nowhere to get V from")
        v_pre = torch.empty(N, dtype=torch.float32, device=dev) if deferred else None
        if not deferred:
            adv = torch.as_tensor(batch.adv, dtype=torch.float32, device=dev)
            ret = torch.as_tensor(batch.ret, dtype=torch.float32, device=dev)
            v_old = torch.as_tensor(batch.values, dtype=torch.float32, device=dev)

        # N3(2): log π_old is recomputed HERE by the same code, not taken from
        # the rollout. Then the CPU/GPU mismatch (~1e-5) never enters the ratio,
        # and the rollout log π stays as a control: `logp_drift` must be on the
        # order of 1e-5, and if it grows — play and training have diverged.
        self.net.eval()      # dropout 0.0, but the rollout ran in eval — keep the same
        logp_old = torch.empty(N, dtype=torch.float32, device=dev)
        # The same `log π_old`, split by factor group: used for per-group
        # `clip_frac` in the FIRST epoch (N11). Computed from the network output
        # already at hand, no second pass.
        gold = ({g: torch.empty(N, dtype=torch.float32, device=dev)
                 for g in GROUPS} if cfg.group_diag else None)
        if getattr(cfg, "resident_buffer", False) and str(dev).startswith("cuda"):
            batch.pin(dev)
        # Anchor cache (`anchor_cache`): anchor output on EVERY buffer turn,
        # computed in batches of exactly `minibatch` — the same size the step
        # would use. An output row doesn't depend on its batch neighbours but
        # does depend on batch SIZE (~7e-6, a different kernel), hence the rule.
        mb = int(cfg.minibatch)
        use_ref = self._anchor_cached() and N >= mb
        self._ref = None
        if use_ref:
            self._ref = {}
        # With `epochs = 0` (critic warmup) nobody needs `log π_old`: no policy
        # epochs, no ratio. The pre-pass and graph capture are skipped then
        # (review 23.09, finding 8), `logp_drift` comes out `nan`.
        pol_on = int(cfg.epochs) > 0
        if not pol_on:
            logp_old.fill_(float("nan"))
        pre = self._ensure_pre_graph(batch, dev) if pol_on else None
        for s in range(0, N if pol_on else 0, cfg.minibatch):
            idx = list(range(s, min(s + cfg.minibatch, N)))
            if pre is not None and len(idx) == cfg.minibatch:
                # ⚠ Same arithmetic as below; output stays ON DEVICE and is
                # sliced into the accumulator — no host transfers in either
                # path.
                res, ref = pre.run(batch, idx)
                logp_old[s:s + len(idx)] = res[0]
                if gold is not None:
                    for j, g in enumerate(GROUPS):
                        gold[g][s:s + len(idx)] = res[1 + j]
                if v_pre is not None:
                    v_pre[s:s + len(idx)] = res[-1]
                if use_ref:
                    self._ref_put(ref, s, N)
                continue
            x, lat = batch.stack(idx, dev)
            lp, _, _, v0, out0 = self.evaluate(x, lat, grad=False, want_kl=False)
            logp_old[s:s + len(idx)] = lp.float()
            if v_pre is not None:
                v_pre[s:s + len(idx)] = v0.float()
            if gold is not None:
                for g, val in group_logp(out0, lat, self.dec).items():
                    gold[g][s:s + len(idx)] = val.float()
            if use_ref and len(idx) == mb:
                self._ref_put(self.anchor_out(x), s, N)
        if use_ref and N % mb:
            # Buffer tail: the anchor is computed over the LAST `mb` turns as a
            # whole, so these rows also get a batch of the same size as the step.
            # The overlap with the previous minibatch writes bitwise-identical numbers.
            idx = list(range(N - mb, N))
            if pre is not None:
                self._ref_put(pre.run(batch, idx)[1], N - mb, N)
            else:
                x, _ = batch.stack(idx, dev)
                self._ref_put(self.anchor_out(x), N - mb, N)
        if deferred:
            if not pol_on:
                # Critic warmup (`epochs = 0`): no pre-pass, V via the same
                # `evaluate` in the same chunks the pre-pass would use.
                for s in range(0, N, mb):
                    idx = list(range(s, min(s + mb, N)))
                    x, lat = batch.stack(idx, dev)
                    v0 = self.evaluate(x, lat, grad=False, want_kl=False)[3]
                    v_pre[s:s + len(idx)] = v0.float()
            # `score` wrote `float(float64(float32))`; same cast here
            batch.set_values(v_pre.double().cpu().numpy())
            adv = torch.as_tensor(batch.adv, dtype=torch.float32, device=dev)
            ret = torch.as_tensor(batch.ret, dtype=torch.float32, device=dev)
            v_old = torch.as_tensor(batch.values, dtype=torch.float32, device=dev)
        # N3 control: recomputation must match the rollout to ~1e-5. Always
        # computed, even when training uses the rollout log π: this is the only
        # place where a play/training mismatch shows up as a number.
        # Drift is over the turns where the rollout `log π` was computed at all
        # (`rollout.logp_every`). Computing it over all turns would give `nan`
        # unnoticed: `np.abs(... - nan).max()` is `nan`, and `nan` in the log
        # looks like "empty field", not "control broken".
        have = getattr(batch, "logp_have", None)
        if have is None:
            have = np.ones(N, dtype=bool)
        lo = logp_old.detach().cpu().numpy()
        drift = (float(np.abs(lo[have] - batch.logp_roll[have]).max())
                 if have.any() else float("nan"))
        drift_n = int(have.sum())
        if not cfg.recompute_old_logp:
            if not have.all():
                raise SystemExit(
                    f"`recompute_old_logp = False` with `rollout.logp_every > 1`: "
                    f"the rollout `log π` exists only for {drift_n} turns of {N}, and "
                    "with recomputation off it is exactly the ratio denominator. "
                    "Either set `logp_every = 1` back, or don't turn off "
                    "recomputation")
            # ⚠ The group breakdown stays RECOMPUTED here: the rollout doesn't
            # write `log π` per group (the record holds one joint number), and
            # there is nothing to split it with after the fact. So with
            # `recompute_old_logp = False` group `clip_frac` values use a
            # slightly different reference than the total (difference ~1e-5,
            # i.e. no effect on fractions), while the total uses the rollout
            # one. Keep that in mind if the numbers don't match perfectly.
            logp_old = torch.as_tensor(batch.logp_roll, dtype=torch.float32,
                                       device=dev)

        # Correction for collection lag (`RunCfg.overlap`): the rollout was
        # played by the PREVIOUS update's weights θ_{k−1} (behaviour policy),
        # while `logp_old` above is the current θ_k (proximal). Decoupled PPO
        # (Hilton, Cobbe, Schulman, arXiv:2110.00641): the clip holds the ratio
        # to the PROXIMAL policy, and sampling bias is removed by the weight
        # w = π_prox/π_behav, clipped from above (`behavior_rho`, like ρ̄ in
        # V-trace). The per-sample PPO objective is linear in A for w ≥ 0 —
        # w·min(rA, clip(r)A) = min(r·wA, clip(r)·wA), and likewise for the
        # dual-clip lower bound — so the weight goes into the advantage, and
        # neither `policy_loss` nor the step graph changes. The critic trains on
        # its own target as before: the one-update bias of the TD(λ) target is
        # not corrected.
        is_m = {}
        if cfg.behavior_rho > 0:
            if not have.all():
                raise SystemExit(
                    "`ppo.behavior_rho > 0` requires `rollout.logp_every = 1`: "
                    f"behaviour `log π` is recorded for {drift_n} turns of {N}")
            lb = torch.as_tensor(batch.logp_roll, dtype=torch.float32, device=dev)
            # the proximal one is the RECOMPUTED one (`lo`), even if the ratio
            # denominator came from the rollout: otherwise w ≡ 1 and the
            # correction silently switches off
            lw = torch.as_tensor(lo, dtype=torch.float32, device=dev) - lb
            w = torch.exp(lw).clamp(max=cfg.behavior_rho)
            adv = adv * w
            is_m = dict(is_w_mean=float(w.mean()), is_w_min=float(w.min()),
                        is_clip_frac=float((lw > math.log(cfg.behavior_rho))
                                           .float().mean()),
                        is_max_abs_log=float(lw.abs().max()))

        t_pre = _t.time() - t_start
        tot = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, kl_anchor=0.0,
                   approx_kl=0.0, clip_frac=0.0, clip_hi=0.0, clip_lo=0.0,
                   ratio_gt_c=0.0, dual_clip_frac=0.0)
        n_seen, grad_sum, gradv_sum, steps = 0, 0.0, 0.0, 0
        skipped, early_stop, epochs_done = 0, 0.0, 0
        # Norm clipping: how many steps it BOUND. With `max_grad_norm` below
        # the working gradient norm, clipping stops being insurance against
        # outliers and becomes constant renormalisation — step length is then
        # set by the threshold, not lr, and the mean `grad_norm` doesn't show it.
        clipped_p, clipped_v, gmax = 0, 0, 0.0
        # Per-factor-group `clip_frac`, first epoch only (there the ratio isn't
        # yet "worked up" by epochs and an oversized step is readable).
        gcf = {g: 0.0 for g in GROUPS}
        gmaxlr = {g: 0.0 for g in GROUPS}
        g_seen = 0
        last_kl = 0.0
        accum = max(1, int(cfg.grad_accum))
        stop = False
        graph = self._ensure_graph(batch, dev) if pol_on else None
        for ep in range(cfg.epochs):
            if stop:
                break
            order = torch.randperm(N).tolist()
            groups = [order[i:i + cfg.minibatch]
                      for i in range(0, N, cfg.minibatch)]
            for gi, idx in enumerate(groups):
                on_graph = graph is not None and len(idx) == cfg.minibatch
                if gi % accum == 0:
                    # A full minibatch under the graph needs no zeroing: the
                    # capture writes the gradient fresh rather than adding. The
                    # partial tail takes the old path and needs zeroing — but IN
                    # PLACE (`_zero_grad`), otherwise `.grad` detaches from the graph.
                    if not on_graph:
                        self._zero_grad()
                    end = min(gi + accum, len(groups))
                    # weight by the number of samples in the group: without it
                    # accumulation is not equivalent to one large batch
                    gsize = sum(len(g) for g in groups[gi:end])
                sel = torch.as_tensor(idx, device=dev)
                if on_graph:
                    # ⚠ The graph branch MUST compute exactly the same as the
                    # branch below and keep the same decision order:
                    # `max |log r|` and the group breakdown — before the
                    # `target_kl` gate, the gate — before the optimizer step, a
                    # non-finite minibatch — no step and no contribution to the
                    # means. This is proven not by reading but by a parity test:
                    # the same weights and data must give BITWISE the same
                    # weights after N steps. There are two substantive
                    # differences, neither affects weights:
                    # (1) the graph always runs backward, even when the old path
                    # would skip it on `isfinite` or the gate — the gradients are
                    # then just not applied;
                    # (2) thirteen `float()` replaced by one `.cpu()` over the
                    # metrics vector (`MET`), same values.
                    out_mb, met = graph.run(batch, idx, sel, logp_old, adv,
                                            ret, v_old, self._ref, gold)
                    d = dict(zip(graph.names, met.tolist()))
                    last_kl = d["approx_kl"]
                    gmax = max(gmax, d["max_abs_log_ratio"])
                    if gold is not None and ep == 0 and graph.groups:
                        for g in GROUPS:
                            gcf[g] += d[f"gc/{g}"]
                            gmaxlr[g] = max(gmaxlr[g], d[f"gm/{g}"])
                        gcf["all"] = gcf.get("all", 0.0) + d["clip_frac"] * len(idx)
                        g_seen += len(idx)
                    elif gold is not None and ep == 0:
                        with torch.no_grad():
                            for g, val in group_logp(out_mb, graph.lat,
                                                     self.dec).items():
                                dd = (val.float() - gold[g][sel])
                                gcf[g] += float((dd.abs()
                                                 > math.log1p(cfg.clip_range))
                                                .float().sum())
                                gmaxlr[g] = max(gmaxlr[g], float(dd.abs().max()))
                        gcf["all"] = gcf.get("all", 0.0) + d["clip_frac"] * len(idx)
                        g_seen += len(idx)
                    if cfg.target_kl > 0 and last_kl > cfg.kl_stop_mult * cfg.target_kl:
                        early_stop = 1.0
                        stop = True
                        break
                    if d["finite"] < 1.0:
                        skipped += 1
                        continue
                    self.opt.step()
                    grad_sum += d["grad_norm"]
                    gradv_sum += d["grad_norm_value"]
                    clipped_p += int(d["grad_norm"] > cfg.max_grad_norm)
                    clipped_v += int(d["grad_norm_value"] > cfg.max_grad_norm)
                    steps += 1
                    for name in tot:
                        tot[name] += d[name] * len(idx)
                    n_seen += len(idx)
                    continue
                x, lat = batch.stack(idx, dev)
                # A partial epoch tail computes the anchor itself, at its own
                # size: the cache was computed in `minibatch`-sized batches, and
                # using it here would switch kernels (~7e-6) vs the old path.
                ref_mb = (self._ref_rows(sel) if self._ref is not None
                          and len(idx) == cfg.minibatch else None)
                lp, ent, kl, v, out_mb = self.evaluate(x, lat, grad=True, ref=ref_mb)
                pl, akl, cf, ex = policy_loss(lp, logp_old[sel], adv[sel],
                                              cfg.clip_range, cfg.dual_clip)
                last_kl = float(akl.detach())
                gmax = max(gmax, float(ex["max_abs_log_ratio"]))
                if gold is not None and ep == 0:
                    # Group criterion: that group ALONE would push the ratio past
                    # the clip, i.e. the group's |Δ log π| exceeds
                    # log(1 + clip_range). Group fractions need not sum to the
                    # total — factors can pull in different directions;
                    # `clip_frac/all` alongside is computed over the same first
                    # epoch so there is something to compare against (the
                    # overall `clip_frac` is averaged over ALL epochs and is not
                    # comparable with the groups).
                    with torch.no_grad():
                        for g, val in group_logp(out_mb, lat, self.dec).items():
                            d = (val.float() - gold[g][sel])
                            gcf[g] += float((d.abs() > math.log1p(cfg.clip_range))
                                            .float().sum())
                            gmaxlr[g] = max(gmaxlr[g], float(d.abs().max()))
                    gcf["all"] = gcf.get("all", 0.0) + float(cf.detach()) * len(idx)
                    g_seen += len(idx)
                # The gate is checked BEFORE backward on EVERY minibatch. Pitfall
                # from a previous project: checking once per epoch, the policy
                # collapsed inside the first one (win rate 0.84 -> 0.49 in two
                # updates). Second pitfall: approx_kl is computed from the ratio
                # BEFORE the step, so the gate doesn't undo the guilty step but
                # prevents the next one — with a small rollout one minibatch
                # pushes KL several times past the threshold. So don't use
                # rollouts below ~2000 turns.
                if cfg.target_kl > 0 and last_kl > cfg.kl_stop_mult * cfg.target_kl:
                    early_stop = 1.0
                    stop = True
                    break
                vl = (value_loss(v.float(), ret[sel], v_old[sel], cfg.value_clip)
                      if v is not None else torch.zeros((), device=dev))
                ent_m, kl_m = ent.mean(), kl.mean()
                loss = (pl + cfg.value_coef * vl - cfg.ent_coef * ent_m
                        + self.kl_coef * kl_m)
                if not torch.isfinite(loss):
                    # The minibatch is skipped ENTIRELY rather than zeroed after
                    # the fact: otherwise nan poisons all update means, and the
                    # gate `last_kl > threshold` gives False on nan — early stop
                    # stays silent exactly on the broken step.
                    skipped += 1
                    continue
                (loss * (len(idx) / max(1, gsize))).backward()
                if gi + 1 == end or gi + 1 == len(groups):
                    for p in self.params:
                        if p.grad is not None:
                            # otherwise one bad step poisons ALL weights
                            # (a pitfall from earlier projects)
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0,
                                              neginf=0.0)
                    gn = self._clip()
                    self.opt.step()
                    grad_sum += float(gn[0])
                    gradv_sum += float(gn[1])
                    clipped_p += int(gn[0] > cfg.max_grad_norm)
                    clipped_v += int(gn[1] > cfg.max_grad_norm)
                    steps += 1
                for name, m in zip(tot, (pl, vl, ent_m, kl_m, akl, cf,
                                         ex["clip_hi"], ex["clip_lo"],
                                         ex["ratio_gt_c"], ex["dual_clip_frac"])):
                    tot[name] += float(m.detach()) * len(idx)
                n_seen += len(idx)
            if not stop:
                epochs_done += 1

        m = {k: (v / n_seen if n_seen else float("nan")) for k, v in tot.items()}
        m.update(# The two halves of the update separately: the `log π_old`
                 # pre-pass always covers the WHOLE buffer, while the epoch loop
                 # is cut by the `target_kl` gate at a random minibatch. As one
                 # "seconds per update" number they are indistinguishable, and
                 # speeding up one half would read as noise in the other.
                 t_logp_pass=t_pre, t_epochs=_t.time() - t_start - t_pre,
                 grad_norm=grad_sum / max(1, steps),
                 # critic gradient norm as a SEPARATE metric: it is clipped
                 # separately and grows on its own (the tower learns from
                 # scratch on the HL head, the policy from a trained anchor)
                 grad_norm_value=gradv_sum / max(1, steps),
                 optimizer_steps=steps,
                 samples=n_seen, epochs_done=epochs_done,
                 # ⚠ health metrics are ALWAYS written, not "if triggered":
                 # otherwise the metric disappears exactly in the mode it
                 # warns about
                 early_stop=early_stop, nonfinite_minibatches=skipped,
                 last_approx_kl=last_kl, kl_coef=self.kl_coef,
                 logp_drift=drift, logp_drift_n=drift_n,
                 **is_m,
                 explained_var=batch.explained_var,
                 # Monte Carlo version of the same quantity: target without
                 # critic estimates (see `buffer.Batch`). The gap between these
                 # two numbers measures how much `explained_var` confirms itself.
                 explained_var_mc=batch.explained_var_mc,
                 # The EV (Monte Carlo) denominator is across games: how many
                 # DISTINCT outcomes are in the rollout and their spread. Fewer
                 # than two — undefined (NaN), and it should only be read with
                 # these two numbers alongside.
                 explained_var_mc_n=getattr(batch, "ret_mc_n", 0),
                 ret_mc_std=getattr(batch, "ret_mc_std", float("nan")),
                 adv_std=float(batch.adv_raw.std()),
                 return_mean=float(batch.ret.mean()),
                 value_mean=float(batch.values.mean()),
                 # ⚠ Fraction of steps where norm clipping BOUND the step. At 1.0
                 # `max_grad_norm` stops being outlier insurance: it renormalises
                 # EVERY step, erasing minibatch differences in gradient
                 # magnitude. The mean `grad_norm` doesn't show this — it is
                 # reported BEFORE clipping and looks the same whether clipping
                 # binds or not.
                 grad_clipped_frac=clipped_p / max(1, steps),
                 grad_clipped_frac_value=clipped_v / max(1, steps),
                 # N11: `clip_frac` alone doesn't show whether the clip cuts at
                 # 0.21 or at 3 — it removes the turn's contribution entirely
                 # in both cases.
                 max_abs_log_ratio=gmax)
        if gold is not None and g_seen:
            for g in GROUPS:
                m[f"clip_frac/{g}"] = gcf[g] / g_seen
                m[f"max_abs_log_ratio/{g}"] = gmaxlr[g]
            m["clip_frac/all"] = gcf.get("all", 0.0) / g_seen
        if self._ref is not None:
            m["anchor_cache_mb"] = sum(v.numel() * v.element_size()
                                       for v in self._ref.values()) / 2**20
        self._ref = None     # the cache belongs to THIS buffer, not needed further
        m.update(self._value_phase(batch, ret))
        m.update(self._ppg(batch, logp_old))
        self._adapt_kl(m)
        self.updates += 1
        return m

    # ---------------- KL-sized step (`ppo.step_mode = "kl_line"`) ----------------

    def _holdout(self, batch, frac, gen):
        """Turn indices: (gradient, held out). Whole GROUPS are held out: turns
        of one game share one A (rloo), games of a group share a baseline, and a
        held-out turn from the same group would not be an independent probe.
        A turn's group is taken from its game (turn `id`), not buffer order:
        `ppo.explore_only` keeps only part of the turns in the buffer."""
        N = len(batch)
        eps = getattr(batch, "episodes", None) or []
        self._tr_half = None
        if frac <= 0:
            return np.arange(N), np.arange(0)
        ep_of = {id(t): i for i, e in enumerate(eps) for t in e.turns}
        if not eps or any(id(t) not in ep_of for t in batch.turns):
            perm = torch.randperm(N, generator=gen).numpy()
            k = max(1, int(N * frac))
            return np.sort(perm[k:]), np.sort(perm[:k])
        gkey = lambda i: str(eps[i].group or f"ep{i}")
        key = np.array([gkey(ep_of[id(t)]) for t in batch.turns])
        keys = sorted(set(key.tolist()))
        perm = torch.randperm(len(keys), generator=gen).tolist()
        held = {keys[i] for i in perm[:max(1, int(round(len(keys) * frac)))]}
        is_ho = np.array([k in held for k in key])
        tr, ho = np.flatnonzero(~is_ho), np.flatnonzero(is_ho)
        # halves of the training groups — for the cosine between their gradients
        half = set(sorted({k for k in keys if k not in held})[::2])
        self._tr_half = np.array([key[i] in half for i in tr], dtype=bool)
        return tr, ho

    def _update_line(self, batch) -> Dict[str, float]:
        """One step along the policy gradient over the whole buffer, length set by KL.

        1. `log π_old` by recomputation (as in PPO, N3(2)).
        2. Gradient of L = −mean(A·log π) + kl_coef·mean(KL to anchor)
           − ent_coef·mean(H) over turns OUTSIDE the held-out groups, accumulated.
        3. θ = θ₀ − α·g/|g|; α is chosen so that KL(π_old ‖ π_new) on the
           held-out turns (k3 estimate on played actions) falls into
           [0.5, 1.5]·`line_kl`; KL ≈ c·α², hence α ← α·√(δ/KL).
        4. On the same held-out turns — surrogate gain mean((r − 1)·A): the
           gradient didn't see them, so this measures signal in the step, not fit.
        """
        cfg = self.cfg
        t0 = _t.time()
        N = len(batch)
        if N == 0:
            raise ValueError("empty rollout")
        if self.p_value or self.critic.wants_value:
            raise SystemExit("`ppo.step_mode = kl_line` — only with `--critic zero`: "
                             "no critic step is made in this mode")
        if getattr(batch, "deferred", False):
            raise SystemExit("`kl_line` without `ppo.prepass_value`")
        dev = self.device
        if getattr(cfg, "resident_buffer", False) and str(dev).startswith("cuda"):
            batch.pin(dev)
        self.net.eval()
        mb = int(cfg.minibatch)
        adv = torch.as_tensor(batch.adv, dtype=torch.float32, device=dev)
        gen = torch.Generator().manual_seed(1_000_003 * (self.updates + 1))
        tr_idx, ho_idx = self._holdout(batch, float(cfg.line_holdout), gen)
        if len(ho_idx) > cfg.line_probe:
            pick = torch.randperm(len(ho_idx), generator=gen)[:cfg.line_probe].numpy()
            ho_idx = np.sort(ho_idx[pick])
        chunks = lambda ix: [ix[i:i + mb].tolist() for i in range(0, len(ix), mb)]

        # 1. log π_old over the whole buffer
        logp_old = torch.empty(N, dtype=torch.float32, device=dev)
        for idx in chunks(np.arange(N)):
            x, lat = batch.stack(idx, dev)
            logp_old[idx] = self.evaluate(x, lat, grad=False, want_kl=False)[0].float()
        have = getattr(batch, "logp_have", None)
        if have is None:
            have = np.ones(N, dtype=bool)
        lo = logp_old.cpu().numpy()
        drift = (float(np.abs(lo[have] - batch.logp_roll[have]).max())
                 if have.any() else float("nan"))

        # 2. gradient over turns outside the held-out set
        params = [p for p in self.p_policy if p.requires_grad]
        for p in params:
            p.grad = None
        n_tr = max(1, len(tr_idx))
        tot = dict(policy_loss=0.0, entropy=0.0, kl_anchor=0.0)
        # Cosine between gradients of the two halves of training GROUPS: shared
        # signal s and independent half-noise n give E[cos] ≈ |s|²/(|s|² + |n|²).
        # Near 0 — the step is almost pure noise, whatever KL it costs.
        self._tr_half = getattr(self, "_tr_half", None)
        halves = (self._tr_half is not None and len(self._tr_half) == len(tr_idx)
                  and self._tr_half.any() and (~self._tr_half).any())
        tr_chunks = (chunks(tr_idx[self._tr_half]) + chunks(tr_idx[~self._tr_half])
                     if halves else chunks(tr_idx))
        n_a = int(self._tr_half.sum()) if halves else 0
        g_a = None
        reg_acc = None
        done = 0
        for idx in tr_chunks:
            if halves and g_a is None and done >= n_a:
                g_a = [p.grad.detach().clone() if p.grad is not None
                       else torch.zeros_like(p) for p in params]
            x, lat = batch.stack(idx, dev)
            lp, ent, kl, _, _ = self.evaluate(x, lat, grad=True)
            a = adv[idx]
            pl = -(a * lp).sum()
            # Policy gradient goes into `.grad` (the half-cosine uses it); leash
            # and entropy separately: they are shared by both halves and would by
            # themselves add to the cosine without any reward signal.
            reg = (self.kl_coef * kl.sum() - cfg.ent_coef * ent.sum()) / n_tr
            if reg.requires_grad:
                (pl / n_tr).backward(retain_graph=True)
                rg = torch.autograd.grad(reg, params, allow_unused=True)
                if reg_acc is None:
                    reg_acc = [torch.zeros_like(p) for p in params]
                for acc, g in zip(reg_acc, rg):
                    if g is not None:
                        acc.add_(g)
            else:
                (pl / n_tr).backward()
            tot["policy_loss"] += float(pl.detach()) / n_tr
            tot["entropy"] += float(ent.detach().sum()) / n_tr
            tot["kl_anchor"] += float(kl.detach().sum()) / n_tr
            done += len(idx)
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        half_cos = float("nan")
        if g_a is not None:
            # g = g_a + g_b; the halves share the n_tr normalisation — no effect
            # on the cosine
            dot = sum(float((a.double() * (g.double() - a.double())).sum())
                      for a, g in zip(g_a, grads))
            na = math.sqrt(sum(float((a.double() ** 2).sum()) for a in g_a))
            nb = math.sqrt(sum(float(((g.double() - a.double()) ** 2).sum())
                               for a, g in zip(g_a, grads)))
            half_cos = dot / max(na * nb, 1e-30)
            del g_a
        pg_norm = math.sqrt(sum(float((g.double() ** 2).sum()) for g in grads))
        reg_norm = 0.0
        if reg_acc is not None:
            reg_norm = math.sqrt(sum(float((r.double() ** 2).sum()) for r in reg_acc))
            grads = [g + r for g, r in zip(grads, reg_acc)]
            del reg_acc
        for g in grads:
            torch.nan_to_num_(g, nan=0.0, posinf=0.0, neginf=0.0)
        gnorm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads)))
        for p in params:
            p.grad = None

        # 3. line search on KL over the held-out turns
        theta0 = [p.detach().clone() for p in params]
        dirs = [-g / max(gnorm, 1e-12) for g in grads]
        # Where the step points: against the previous step (< 0 — the policy
        # jumps back and forth) and toward the anchor (> 0 — a restoring
        # gradient, BC is near a local optimum).
        dot_f = lambda a, b: sum(float((x.double() * y.double()).sum())
                                 for x, y in zip(a, b))
        prev = getattr(self, "_line_prev_dir", None)
        cos_prev = (dot_f(dirs, prev) if prev is not None else float("nan"))
        cos_anchor = float("nan")
        if self.anchor is not None:
            ap = dict(self.anchor.named_parameters())
            names = {id(p): n for n, p in self.net.named_parameters()}
            back = [(ap[names[id(p)]].detach() - p.detach()) if id(p) in names
                    and names[id(p)] in ap else torch.zeros_like(p) for p in params]
            bn = math.sqrt(dot_f(back, back))
            if bn > 0:
                cos_anchor = dot_f(dirs, back) / bn
            del back
        self._line_prev_dir = [d.detach().clone() for d in dirs]
        ho_chunks = chunks(ho_idx)
        a_ho = adv[ho_idx.tolist()] if len(ho_idx) else None

        def set_alpha(al):
            with torch.no_grad():
                for p, p0, d in zip(params, theta0, dirs):
                    p.copy_(p0 + al * d)

        def probe():
            kl3, gain, mlr, n = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for idx in ho_chunks:
                    x, lat = batch.stack(idx, dev)
                    lp = self.evaluate(x, lat, grad=False, want_kl=False)[0].float()
                    lr_ = lp - logp_old[idx]
                    r = lr_.exp()
                    kl3 += float(((r - 1) - lr_).sum())
                    gain += float(((r - 1) * adv[idx]).sum())
                    mlr = max(mlr, float(lr_.abs().max()))
                    n += len(idx)
            n = max(1, n)
            return kl3 / n, gain / n, mlr

        delta = float(cfg.line_kl)
        al = float(getattr(self, "_line_alpha", 0.0) or cfg.line_alpha0)
        tries = []
        kl_, gain_, mlr_ = float("nan"), float("nan"), 0.0
        if len(ho_idx) and gnorm > 0:
            for _ in range(int(cfg.line_iters)):
                set_alpha(al)
                kl_, gain_, mlr_ = probe()
                tries.append((al, kl_))
                if kl_ != kl_ or kl_ == float("inf"):
                    al *= 0.1
                    continue
                if 0.5 * delta <= kl_ <= 1.5 * delta:
                    break
                al *= float(np.clip(math.sqrt(delta / max(kl_, 1e-12)), 0.1, 10.0))
            else:
                # missed the corridor: take the largest probe that didn't
                # overshoot 1.5·δ, otherwise the step is cancelled
                ok = [t for t in tries if t[1] == t[1] and t[1] <= 1.5 * delta]
                al = max(ok)[0] if ok else 0.0
                set_alpha(al)
                kl_, gain_, mlr_ = probe() if al > 0 else (0.0, 0.0, 0.0)
        else:
            set_alpha(0.0)
            al = 0.0
        if al > 0:
            self._line_alpha = al
        a_std = float(a_ho.std()) if a_ho is not None and len(a_ho) > 1 else float("nan")
        m = dict(tot,
                 approx_kl=kl_, clip_frac=0.0, early_stop=0.0,
                 logp_drift=drift, logp_drift_n=int(have.sum()),
                 grad_norm=gnorm, optimizer_steps=1 if al > 0 else 0,
                 samples=int(len(tr_idx)), epochs_done=1,
                 line_alpha=al, line_iters=len(tries), half_cos=half_cos,
                 pg_norm=pg_norm, reg_norm=reg_norm,
                 cos_prev_step=cos_prev, cos_to_anchor=cos_anchor,
                 line_tries=" ".join(f"{a:.3g}:{k:.3g}" for a, k in tries),
                 # surrogate gain on HELD-OUT groups: > 0 — the step carries a
                 # signal that generalises; divided by √KL — "benefit per unit
                 # of shift", comparable across runs
                 heldout_gain=gain_,
                 heldout_gain_per_sqrt_kl=(gain_ / math.sqrt(kl_)
                                           if kl_ == kl_ and kl_ > 0 else float("nan")),
                 heldout_n=int(len(ho_idx)), heldout_adv_std=a_std,
                 max_abs_log_ratio=mlr_,
                 explained_var=batch.explained_var,
                 explained_var_mc=batch.explained_var_mc,
                 adv_std=float(batch.adv_raw.std()),
                 return_mean=float(batch.ret.mean()),
                 t_update=_t.time() - t0, kl_coef=self.kl_coef)
        self._adapt_kl(m)
        self.updates += 1
        return m

    def _ref_put(self, ref, s, N):
        """Store the minibatch anchor output in the cache at rows s.. (cache allocated lazily)."""
        n = next(iter(ref.values())).shape[0]
        for k in KL_KEYS:
            if k not in self._ref:
                self._ref[k] = torch.empty((N,) + tuple(ref[k].shape[1:]),
                                           dtype=ref[k].dtype, device=ref[k].device)
            self._ref[k][s:s + n].copy_(ref[k])

    def _ref_rows(self, sel):
        """Anchor cache at minibatch rows -> `Logits` for `policy.kl`."""
        return _as_logits({k: v.index_select(0, sel) for k, v in self._ref.items()},
                          self.fam_spec)

    def _clip(self):
        """Norm clipping over two groups. -> (policy norm, critic norm).

        A shared norm over all parameters would silently tie the policy step to
        the critic gradient: `clip_grad_norm_` divides ALL gradients by one
        coefficient. Our critic is a separate tower on a detached input, and
        "the critic doesn't touch the policy" must hold for training too, not
        only for the forward pass (`model.policy_parameters`).
        """
        gp, gv = self._clip_t()
        return float(gp), float(gv)

    def _clip_t(self):
        """The same clipping, but as TENSORS: `float()` is a device sync,
        and inside a CUDA graph there can be none."""
        gp = nn.utils.clip_grad_norm_(self.p_policy, self.cfg.max_grad_norm)
        gv = (nn.utils.clip_grad_norm_(self.p_value, self.cfg.max_grad_norm)
              if self.p_value else torch.zeros((), device=self.device))
        return gp, gv

    # ---------------- separate critic epochs ----------------

    def _value_phase(self, batch, ret) -> Dict[str, float]:
        """Another `value_epochs` epochs on the critic ONLY, outside the policy gate.

        ⚠ Why separately. The policy loop is cut by the `target_kl` gate, and
        before this change the critic went silent with it — exactly when its
        target was most stale (KL moved => policy shifted => `V` refers to a
        different policy). Also their optimal sample reuse DIFFERS: PPG
        (arXiv:2009.04416) measures 1 policy epoch vs 6 critic epochs; tying
        them to one `epochs` number means tuning one to the other blindly.

        Why this is not PPG. PPG fixes the conflict over a SHARED trunk and so
        pays a clone loss to hold the policy. Our tower sits on `h.detach()`,
        owns no policy parameters, and the value loss ALONE gives the policy no
        non-zero gradient — a property of the construction, checked by name
        (self-test, "critic gradient does not flow into the policy"). So there is
        nothing to hold: only `p_value` gets gradients, `AdamW.step` skips
        parameters without gradient, and norm clipping runs over its own group.

        The target is THE SAME `ret` (rollout TD(λ)), not recomputed between
        epochs: recomputing would mean chasing our own tail inside one buffer,
        and that is no longer the PPO whose metrics we read.
        """
        cfg = self.cfg
        if cfg.value_epochs <= 0 or not self.critic.trainable:
            return {"value_epochs_done": 0, "value_phase_loss": float("nan")}
        N = len(batch)
        # Speed (review 23.09, finding 6): previously every minibatch ran a full
        # `evaluate` — trunk with autograd, `log π`, entropy — for the sake of a
        # tower on `h.detach()`, and the leak check did ~275 device syncs per
        # minibatch (with a live graph policy gradients are zero tensors, not
        # `None`). Now: network pass and one V head; the leak is checked ONCE,
        # on the first minibatch, with a single sync, and then `requires_grad`
        # is turned off on the policy so autograd doesn't record the trunk at
        # all. Flags are restored in `finally`.
        pol_req = [p.requires_grad for p in self.p_policy]
        try:
            return self._value_loop(batch, ret, N, pol_req)
        finally:
            for p, r in zip(self.p_policy, pol_req):
                p.requires_grad_(r)

    def _trunk_cache(self, batch, N: int, mb: int):
        """Trunk output (`h`, `pad`) on EVERY buffer turn (`ppo.value_cache`).

        In consecutive chunks of exactly `mb` — the same size the phase step
        would use (a row doesn't depend on batch neighbours, but does on size);
        the buffer tail is the last `mb` turns as a whole, as with the anchor
        cache. No gradient: the tower reads `h.detach()` anyway."""
        dev = self.device
        hc = pc = None
        starts = list(range(0, N - N % mb, mb)) + ([N - mb] if N % mb else [])
        with torch.no_grad():
            for s in starts:
                x, _ = batch.stack(list(range(s, s + mb)), dev)
                h, pad, _, _ = self.net.trunk(
                    x["tiles_me"].float(), x["tiles_op"].float(),
                    x["units"].float(), x["unit_mask"], x["products"].float(),
                    x["glob"].float(), x["crops"].float(), x["animals"].float())
                if hc is None:
                    hc = torch.empty((N,) + tuple(h.shape[1:]), dtype=h.dtype,
                                     device=h.device)
                    pc = torch.empty((N,) + tuple(pad.shape[1:]),
                                     dtype=pad.dtype, device=pad.device)
                hc[s:s + mb] = h
                pc[s:s + mb] = pad
        return hc, pc

    def _value_loop(self, batch, ret, N, pol_req) -> Dict[str, float]:
        cfg = self.cfg
        done, vl_sum, n, gv_sum = 0, 0.0, 0, 0.0
        checked = False
        mb = int(cfg.minibatch)
        # `ppo.value_cache`: trunk once per buffer, then only the tower.
        # Requires the R1 tower (`vtower`); the stub critic has no phase at all.
        cache = (self._trunk_cache(batch, N, mb)
                 if getattr(cfg, "value_cache", False) and N >= mb
                 and hasattr(self.net, "vtower") else None)
        for _ in range(int(cfg.value_epochs)):
            order = torch.randperm(N).tolist()
            for s in range(0, N, cfg.minibatch):
                idx = order[s:s + cfg.minibatch]
                if not idx:
                    continue
                x, lat = batch.stack(idx, self.device)
                sel = torch.as_tensor(idx, device=self.device)
                with torch.enable_grad():
                    if cache is not None and len(idx) == mb:
                        # the same tower call as in `FarmNet.forward`, on cache
                        # rows (`index_select` gives a fresh tensor)
                        vl_, vh_ = self.net.vtower(
                            cache[0].index_select(0, sel),
                            cache[1].index_select(0, sel),
                            x["tiles_op"].float(), x["vcrit"].float())
                        v = self.critic.scalar({"value_logits": vl_,
                                                "value_hl": vh_})
                    else:
                        raw = self._forward(self.net, x, True)
                        v = self.critic.head(POL.family(raw, self.prior, *self.fam), x)
                if v is None:
                    return {"value_epochs_done": 0,
                            "value_phase_loss": float("nan")}
                vl = 0.5 * (v.float() - ret[sel]).square().mean()
                if not torch.isfinite(vl):
                    continue
                self._zero_grad()
                vl.backward()
                for p in self.p_value:
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                # ⚠ Verify the construction, don't trust it: if the policy has a
                # gradient after the value loss, the tower is no longer separate,
                # and the phase would silently train the policy without the PPO clip.
                if not checked:
                    gs = [p.grad.abs().max() for p in self.p_policy
                          if p.grad is not None]
                    if gs and float(torch.stack(gs).max()) > 0:
                        raise RuntimeError(
                            "value loss produced a policy gradient: the tower is no "
                            "longer on `h.detach()`, separate critic epochs must "
                            "not run like this (see `_value_phase`)")
                    checked = True
                    for p in self.p_policy:
                        p.requires_grad_(False)
                gv_sum += float(nn.utils.clip_grad_norm_(self.p_value,
                                                         cfg.max_grad_norm))
                # critic ONLY: the shared `step()` would move the policy by
                # residual momentum (`DualAdamW`, the rl1 bug)
                if hasattr(self.opt, "step_value"):
                    self.opt.step_value()
                else:
                    self.opt.step()
                vl_sum += float(vl.detach()) * len(idx)
                n += len(idx)
                done += 1
        return {"value_epochs_done": int(cfg.value_epochs),
                "value_phase_steps": done,
                "value_phase_loss": vl_sum / max(1, n),
                "value_phase_grad_norm": gv_sum / max(1, done)}

    # ---------------- auxiliary phase (PPG) ----------------

    def _ppg(self, batch, logp_old) -> Dict[str, float]:
        """Critic phase with policy retention (Cobbe et al., arXiv:2009.04416).

        ⚠ Off by default: PPG fixes the policy/critic conflict over a SHARED
        trunk, while per P1/R1 our critic reads `h.detach()` and owns its tower
        — nothing to fix. Ported verbatim in case the critic ever moves onto the
        shared trunk.

        Difference from the previous project: the phase runs regardless of the
        `target_kl` gate (there, at target_kl 0.03, it NEVER ran), and the
        policy reference is taken after the policy phase, not from the rollout.
        Difference from the paper: one buffer, the current one — no memory to
        keep several rollouts of turns.
        """
        cfg_ppg = self.ppg
        if cfg_ppg is None or cfg_ppg.aux_epochs <= 0 or not self.critic.trainable:
            return {"ppg_updates": 0}
        N = len(batch)
        ret = torch.as_tensor(batch.ret, dtype=torch.float32, device=self.device)
        ref = torch.empty(N, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for s in range(0, N, self.cfg.minibatch):
                idx = list(range(s, min(s + self.cfg.minibatch, N)))
                x, lat = batch.stack(idx, self.device)
                lp, _, _, _, _ = self.evaluate(x, lat, grad=False, want_kl=False)
                ref[s:s + len(idx)] = lp.float()
        done, kl_sum, vl_sum, n = 0, 0.0, 0.0, 0
        for _ in range(cfg_ppg.aux_epochs):
            for s in range(0, N, self.cfg.minibatch):
                idx = list(range(s, min(s + self.cfg.minibatch, N)))
                x, lat = batch.stack(idx, self.device)
                sel = torch.as_tensor(idx, device=self.device)
                lp, _, _, v, _ = self.evaluate(x, lat, grad=True, want_kl=False)
                if v is None:
                    return {"ppg_updates": 0}
                vl = 0.5 * (v.float() - ret[sel]).square().mean()
                # The previous project estimates KL as 0.5·(Δ log π)² — to
                # second order that is exactly it, and it needs no second pass
                # over the distribution. For `exact_kl` see `config.PPGCfg`.
                clone = (0.5 * (lp - ref[sel]).square().mean() if not cfg_ppg.exact_kl
                         else self._clone_kl(x, lat))
                loss = vl + cfg_ppg.clone_coef * clone
                if not torch.isfinite(loss):
                    continue
                self._zero_grad()
                loss.backward()
                self._clip()
                self.opt.step()
                kl_sum += float(clone.detach()) * len(idx)
                vl_sum += float(vl.detach()) * len(idx)
                n += len(idx)
                done += 1
        return {"ppg_updates": done, "ppg_clone_kl": kl_sum / max(1, n),
                "ppg_aux_value_loss": vl_sum / max(1, n)}

    def _clone_kl(self, x, lat):
        raise NotImplementedError(
            "exact KL to the policy-phase reference needs a second weight snapshot; "
            "PPG is off for now (critic is a separate tower, R1), set exact_kl=False")

    def _adapt_kl(self, m):
        """Adapt the leash coefficient toward a target (as in KL-penalty PPO).

        Off at `kl_target = 0`: a constant coefficient is easier to measure
        pairwise, while adaptation changes the leash strength within a run and
        makes two arms incomparable.
        """
        if self.cfg.kl_target <= 0:
            return
        kl = m.get("kl_anchor", 0.0)
        if kl != kl:
            return
        if kl > self.cfg.kl_target * 1.5:
            self.kl_coef *= self.cfg.kl_adapt
        elif kl < self.cfg.kl_target / 1.5:
            self.kl_coef /= self.cfg.kl_adapt
        self.kl_coef = float(min(max(self.kl_coef, 1e-4), 100.0))
