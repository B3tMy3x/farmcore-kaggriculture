"""Turn from the network output: masks before sampling, sampling and `log π` -- one place.

Why a separate module (R2, R3, R5, 22.09). Under PPO the `log π` of the executed turn
must be computed identically in play (rollout) and in training (recomputation from the
stored state). This is the same class of bugs as `shed_at_market` and
`legal`/`legal_now`: two copies of the logic silently diverge. So there is one
function `run_turn` here, which drives the turn decision by decision, and it is given a
"picker": `Sampler` draws decisions from the net (play), `Replayer` returns
the recorded ones (recomputation). The masks each decision sees are built by one and the same
code, and `log_prob` computes the probability from the recorded masks and choices --
in a batch, identically for play (batch 1) and training. Verified by
an internal parity script.

What counts as a network action (the turn latent `lat`), in execution order:

  units k = 0..min(n, 24)-1, one by one, as the engine applies them:
    target    pointer to a tile or "stay", always;
    job       op under the legality mask -- only if the unit is at its target (target =
              its own cell or "stay"); the mask is SEQUENTIAL: seeds and
              shed after the jobs of units with lower numbers;
    how many  for PICKUP, PLACE of a good and PLACE of an animal into the shed -- a CORN
              chain with a budget cap (see below);
  market, in order of the order list (sells finance buys):
    sells     9 goods: gate, "whole stock" flag, how many;
    hire      gate, how many;
    seeds     5 crops: gate, how many;
    buys      9 goods (the engine executes only wheat and fertilizer,
              env_ref.py:598 -- the rest are closed by a rule): gate, how many;
    animals   3: gate, how many;
    land      Bernoulli.

Quantity mask by budget BEFORE sampling (R3). Until 22.09 the decoder clipped what the
net asked for: `BUY_PRODUCT` 84.9% of requested units, `BUY_SEED` 63.6%,
`PLACE` 49.2%, `HIRE` 40.0% (A9, game `n1_base`). Under PPO this is a reward for an
action that did not happen. Now each decision has its own cap m: money
after the previous orders (with sale revenue along the engine's price path), room in the
shed, contents of hands and shed, free slots out of the limit of 10 orders.
m = 0 closes the gate (no decision, zero contribution to `log π`), otherwise the chain
is closed at the step "continue past m" (counts.py).

What remains with the environment rather than a network action (not in `log π`, determined
by the latent deterministically):
  * X-then-Y rule step towards the target (100.000% of the teacher's steps are such);
  * units beyond 24 stand still (the net has nothing to control them with);
  * DROP when the shed lacks room: the engine would destroy the excess, so the
    decoder places the most expensive good by address (PLACE), and with a full shed
    stands still -- the substitution is deterministic given the state;
  * PLACE of an animal on a suitable free building: the engine ignores the quantity,
    the net does not choose it;
  * the opponent: the engine executes the orders of both sides one by one, interleaved
    (`_process_market`), and its sells/buys of the same turn move prices. Our
    cap counts the price path only over our own orders; a buy that fell short
    because of the opponent is cut off by the engine. This is environment stochasticity.

There is no tail beyond the order limit any more: slots are part of the budget, and decisions
left without a slot are closed by the mask, not cut off after sampling.
"""
from __future__ import annotations

import copy
import math
import os as _os
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import counts as CNT  # noqa: E402
import envcfg  # noqa: E402
from bc_train import UNIT_OPS, STAY, BOARD, shed_at_market  # noqa: E402
from features import MAX_UNITS  # noqa: E402
from masks import legal_ops, structure_free  # noqa: E402
from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    CROPS, ANIMALS, PRODUCTS, market_price, _hire_cost,
)

CROP_LIST = list(CROPS)
ANIMAL_LIST = list(ANIMALS)
N_OPS = len(UNIT_OPS)
# An engine rule, not a habit: it discards BUY_PRODUCT of any other good
# entirely (env_ref.py:598). Teacher 56263693 writes such orders in 69%
# of its BUY_PRODUCT (112,128 of 162,313) -- none is executed.
BUYABLE = ("WHEAT", "FERTILIZER")

# Market decisions in execution order: (head, item index).
MKT = ([("sell", i) for i in range(len(PRODUCTS))] + [("hire", 0)]
       + [("seed", i) for i in range(len(CROP_LIST))]
       + [("buy", i) for i in range(len(PRODUCTS))]
       + [("animal", i) for i in range(len(ANIMAL_LIST))])
# The default EXECUTION order of market blocks is that of teacher 56263693
# (the MKT latent layout does not depend on it). The net gets its own order ONLY via
# the `run_turn(order=...)` argument: from the checkpoint (`bc_train --mkt-order`) or
# the decoder setting `mkt_order` (play._NET / the package's decode). The environment
# variable FARMCORE_MKT_ORDER was removed 29.09: the module is shared, and it changed the order
# for the opponent too.
MARKET_ORDER = ("sell", "hire", "seed", "buy", "animal", "land")
N_MKT = len(MKT)
_MKT_SPAN = {}                       # head -> (start, end) in MKT
for _j, (_h, _i) in enumerate(MKT):
    a, _ = _MKT_SPAN.get(_h, (_j, _j))
    _MKT_SPAN[_h] = (a, _j + 1)

# Decoder modes. BC -- as played before R2/R3: gates by sampling, units argmax,
# quantity by the median, the CARE habit on. PPO -- everything by sampling, so that
# `log π` is defined for every decision; unit τ 0.5 (R4: free in
# money, τ = 1 costs -$9.8k), habits are held by a leash, not by a mask (R8).
# ⚠ Market gates in BC are by THRESHOLD since 25.09. Measurement: `corn_r8off/e11` by threshold against
# itself with gate sampling, 96 seeds 3000+, +$1,082 (median +$1,196), 68/96,
# wilcoxon p = 0.0005; on `value_tower/e11` +$1,231, 62/96, p = 0.0009.
# The earlier conclusion "threshold -$164, indistinguishable" (internal note) is withdrawn:
# `play.py --no-sample` then switched the opponent as well, and the measurement was "threshold
# against threshold".
BC = {"sample": False, "sample_units": 0.0, "sample_qty": 0.0}
PPO = {"sample": True, "sample_units": 0.5, "sample_qty": 0.5}   # one τ: units, quantities, gates, "all", land

# ---------------------------------------------------------------------------
# Distribution family (N2, N4). Fixed BEFORE the anchor: PPO computes log π,
# entropy and KL to the anchor in the family that stands at the start, and changing it
# later means moving the anchor (internal review, "Important before the anchor").
#
# beta -- logit adjustment. `bc_train.loss_fn` trains op and target with an added
#   `τ·log prior` (Menon et al., ICLR 2021), while the net played with raw logits:
#   trained one distribution, played another (N2). `prior` lies in the
#   checkpoint (`ck["prior"]`), β = 1 at τ = 1 returns the teacher's
#   distribution (op label NLL 0.141 vs 0.277 for raw logits, accuracy 0.949
#   vs 0.904).
# floor -- lower bound on the probability of saturated gates (N4): animal 96.7%
#   of states with |logit| > 9, land 92.7%, hire 90.9%, and outside the teacher's windows per
#   game 0.13 expected animal openings. Under PPO this is the absence of
#   exploration exactly where the strategic money lies.
#
# MEASURED 23.09 on `corn_r8off/e11`, head-to-head 48 games, seeds 3000-3047
# (`tools/bench/pair.py`, paired by seed; control on the same block +$318,
# 26/48, threshold $1.5k):
#   β = 1 vs β = 0, BC mode          +$3,173, 44/48, p = 8.4e-09, threshold $1.4k
#   ε = 1e-3 vs 0, BC mode             +$203, 25/48, p = 0.60,    threshold $1.7k
#   β = 1 and ε = 1e-3 vs raw        +$2,098, 36/48, p = 2.7e-04, threshold $1.5k
# That is, logit adjustment is not "neutral but honest", it is directly more profitable than
# raw logits, and the gate lower bound is free. Exploration meanwhile
# grows (probe C, internal gate-exploration script, PPO mode: expected
# openings in states with P < 0.01 per game -- animals 0.065 -> 1.57,
# land 0.041 -> 0.48, hire 0.10 -> 0.68), while behaviour barely changes
# (total animal openings 20.2 -> 21.8, land 4.04 -> 4.03).
FAMILY = {"beta": 1.0, "floor": 0.25}   # floor = tail slope k (see _floor_gate)

# Gates that get a lower bound: strategic levers (internal economics note).
# Sells and buys are left alone -- they are not saturated (buy: 0.0% of states with
# |logit| > 9, sell 53%) and make up the bulk of the turn's decisions.
FLOOR_HEADS = ("seed", "animal", "hire")     # land is a separate output


class Logits(dict):
    """Logits ALREADY brought to the played family (output of `family`).

    A separate type, not a dict key: values stay tensors and only
    tensors, so `for v in out.values(): v.requires_grad_(True)` and other
    traversals of the net output work as before. A plain `dict` of logits is not
    let through further -- see `_need_family`.
    """
    __slots__ = ("spec",)


KNEE = 4.0            # up to |logit| = 4 the family is untouched


def _floor_gate(l, k):
    """Gate logit after tail squeezing: |l| <= KNEE -- as is, beyond that slope k.

    Replaces the mixture p' = (1-ε)p + ε/2 (N4, 23.09). For the mixture the derivative of log p'
    with respect to the logit at p -> 0 is (1-ε)p(1-p)/p' -> 0: exploring saturated
    gates (animals 97.7%, land 94.4%, hire 88% of states with |logit| > 9)
    existed, but there was no gradient through it -- PPO could not reward a single such
    opening. The squeeze is monotone (threshold 0.5 = argmax does not change, the ladder plays
    as before), gives the lower bound σ(-(KNEE + k·(|l|-KNEE))) -- at k = 0.25
    and |l| = 12 this is σ(-6) ≈ 2.5e-3, of the same order as ε/2 of the mixture -- and
    the derivative k > 0 over the whole tail. k = 0 -- raw logits.
    """
    if not k:
        return l
    a = l.abs()
    return torch.sign(l) * torch.where(a <= KNEE, a, KNEE + k * (a - KNEE))


def family(out, prior=None, beta=0.0, floor=0.0, macro=None):
    """Raw network logits -> logits of the PLAYED distribution. The single point.

    Downstream (`Sampler`, `log_prob`, `entropy`, `kl`) there are no more corrections:
    sampling and recomputation must see the same numbers, otherwise this is the same
    class of bugs as `shed_at_market` and `legal`/`legal_now` -- a divergence
    invisible in validation and in money. So `Sampler`,
    `log_prob`, `entropy` and `kl` accept ONLY the output of this function: a loud
    refusal instead of a silently different distribution.

    β = 0 and ε = 0 is the former behaviour (raw logits), but the output carries the mark
    anyway: "family not chosen" and "family empty" are different things.
    """
    if isinstance(out, Logits):
        raise RuntimeError(f"family already applied: {out.spec}; a second call "
                           "would add β·prior and the gate bound twice")
    o = dict(out)
    if beta:
        if not prior:
            raise RuntimeError(
                "β > 0 but prior is not given. It lies in the checkpoint (`ck['prior']`, "
                "put there by `bc_train`) and is loaded in `play._load`. A checkpoint without "
                "it (mega_p8, base_p01 -- restored from packages, and packages "
                "before 23.09 did not carry prior) plays only with --beta 0 or after "
                "recomputing `bc_train.logit_prior` over its corpus")
        dev = out["op_logits"].device
        o["op_logits"] = out["op_logits"] + beta * prior["op"].to(dev)
        T1 = out["target_logits"].shape[-1]
        o["target_logits"] = out["target_logits"] + beta * prior["target"][:T1].to(dev)
    if macro is not None:
        # Macro-policy (26.09): per-day logit offsets. `op` -- [40] for unit
        # verbs; `sell`/`buy`/`seed`/`animal` -- to the GATE LOGIT (index 0,
        # `counts.split`) per good; `hire` and `land` -- scalars on the gates.
        # Positive = more activity. Placed BEFORE the gate bound.
        dev = o["op_logits"].device
        o["op_logits"] = o["op_logits"] + macro["op"].to(dev)
        for head in ("sell", "buy", "seed", "animal"):
            k = f"{head}_logits"; lg = o[k]
            b = macro[head].to(dev).view(*([1] * (lg.dim() - 2)), -1, 1)
            o[k] = torch.cat([lg[..., :1] + b, lg[..., 1:]], -1)
        hl = o["hire_logits"]
        o["hire_logits"] = torch.cat([hl[..., :1] + macro["hire"].to(dev), hl[..., 1:]], -1)
        o["land_logits"] = o["land_logits"] + macro["land"].to(dev)
    if floor:
        for head in FLOOR_HEADS:
            k = f"{head}_logits"
            lg = o[k]
            o[k] = torch.cat([_floor_gate(lg[..., :1], floor), lg[..., 1:]], -1)
        o["land_logits"] = _floor_gate(o["land_logits"], floor)
    o = Logits(o)
    o.spec = (float(beta), float(floor))
    return o


def _need_family(out):
    if not isinstance(out, Logits):
        raise RuntimeError(
            "logits did not pass through policy.family(out, prior, beta, floor): "
            "the distribution family (β·prior -- N2, gate lower bound -- N4) "
            "is set ONCE right after the network pass, and sampling, log π, "
            "entropy and KL must see the same logits")


def empty_lat():
    """Turn latent: choices and the masks under which they were made. Shapes are fixed
    so that turns stack into a batch for recomputation."""
    U = MAX_UNITS
    return {
        "u_live": np.zeros(U, bool), "tgt": np.full(U, STAY, np.int64),
        "u_pos": np.zeros(U, np.int64),
        "op_on": np.zeros(U, bool), "op": np.zeros(U, np.int64),
        "op_mask": np.zeros((U, N_OPS), bool),
        "q_on": np.zeros(U, bool), "q": np.ones(U, np.int64),
        "q_max": np.ones(U, np.int64),
        "m_max": np.zeros(N_MKT, np.int64), "m_gate": np.zeros(N_MKT, np.int64),
        "m_all": np.zeros(N_MKT, np.int64), "m_n": np.zeros(N_MKT, np.int64),
        "land_ok": np.zeros((), bool), "land": np.zeros((), np.int64),
    }


class Sampler:
    """Decisions from the network output (one game, batch 1).

    `sample` -- gates, the "all" flag and land are Bernoulli (else threshold 0.5);
    `sample_units` -- temperature of target and job (0 = argmax);
    `sample_qty` -- temperature of the quantity chain (0 = median).
    The generator is the global torch one: the measurement fixes it per game.
    """

    def __init__(self, out, sample=True, sample_units=0.0, sample_qty=0.0, acc=None, dedup=False,
                 land_acc=None, acc_heads=""):
        _need_family(out)
        self.o = {k: v[0].detach().float().cpu() for k, v in out.items()
                  if k.endswith("_logits")}
        self.smp, self.tu, self.tq = bool(sample), float(sample_units), float(sample_qty)
        # Gates by accumulated frequency (28.09): a per-game dict, head -> accumulated P.
        # A calibrated net that does not know the exact hour of a purchase spreads its
        # probability over many turns (0.2 × 5), and the threshold 0.5 never
        # fires: wheat seeds 61 openings at ΣP = 146 and 150 for the teacher.
        # Here a gate opens when the accumulated value reaches ≥ 0.5, and the accumulator
        # decreases by 1 -- the opening frequency equals ΣP (systematic sampling).
        self.acc = acc
        self.acc_heads = set(h for h in (acc_heads or "").split(",") if h)
        # Task distribution without repeats (28.09): the teacher-planner hands out different
        # cells to hands, while the pointer picks a target for each unit separately --
        # clone 56254996 has three times more units on an already occupied cell (443-475
        # vs 151-185 for the teacher per game), the second hand arrives at done work.
        self.dedup = bool(dedup)
        # Land by accumulated frequency (28.09): 2-3 purchases per game at any hour
        # of the right day -- P is smeared over hours, threshold 0.5 never fires
        # (clone 56407295 stays on one quarter all season, the teacher takes the 2nd and 3rd
        # on the 6th and 8th day). The accumulator is its own, per game (play._FLOW).
        self.land_acc = land_acc

    def targets(self, n_live=None):
        tl = self.o["target_logits"]
        if self.tu > 0:
            return torch.multinomial(torch.softmax(tl / self.tu, -1), 1).squeeze(-1).numpy()
        # argmax, on a tie -- a tile, not "stay" (as it was in play.py)
        tp = torch.softmax(tl, -1)
        best = tp[:, :-1].argmax(-1)
        out = torch.where(tp[:, -1] > tp[:, :-1].max(-1).values,
                          torch.full_like(best, STAY), best).numpy()
        if not self.dedup or not n_live:
            return out
        # Greedy matching: confident units choose first, the rest take the
        # best free cell; approaches to the shed may be shared.
        free_ok = {y * BOARD + x for (x, y) in envcfg.shed_tiles(BOARD)}
        n = min(int(n_live), tp.shape[0])
        order = sorted(range(n), key=lambda k: -float(tp[k, :-1].max()))
        taken = set()
        for k in order:
            if out[k] == STAY:
                continue
            pk = tp[k, :-1].clone()
            for j in taken:
                pk[j] = -1.0
            j = int(pk.argmax())
            if float(tp[k, -1]) > float(pk[j]):
                out[k] = STAY
                continue
            out[k] = j
            if j not in free_ok:
                taken.add(j)
        return out

    def op(self, k, mask):
        pr = self.o["op_logits"][k].masked_fill(~torch.from_numpy(mask), float("-inf"))
        if self.tu > 0:
            return int(torch.multinomial(torch.softmax(pr / self.tu, -1), 1))
        return int(pr.argmax())

    def argmax_op(self, k, mask):
        pr = self.o["op_logits"][k].masked_fill(~torch.from_numpy(mask), float("-inf"))
        return int(pr.argmax())

    def _count(self, cond, m, head):
        if self.tq > 0:
            c = int(torch.multinomial(CNT.pmf(cond, m, self.tq, head), 1))
        else:
            c = int(CNT.log_pmf(cond, torch.tensor(int(m)), 1.0, head).argmax())
        return int(CNT.values(head)[c])

    def qty(self, k, m):
        _, cond, _ = CNT.split(self.o["qty_logits"][k], "pickup")
        return self._count(cond, m, "pickup")

    def _bern(self, logit):
        if not self.smp:
            return int(logit.item() > 0)
        p = torch.sigmoid(logit / (self.tq if self.tq > 0 else 1.0))
        return int(torch.bernoulli(p).item())

    def _row(self, head, i):
        lg = self.o[f"{head}_logits"]
        return lg if head == "hire" else lg[i]

    def gate(self, head, i):
        logit = CNT.split(self._row(head, i), head)[0]
        if self.acc is None or self.smp or (self.acc_heads and head not in self.acc_heads):
            return self._bern(logit)
        k = (head, i)
        a = min(1.0, self.acc.get(k, 0.0) + float(torch.sigmoid(logit)))
        if a >= 0.5:
            self.acc[k] = a - 1.0
            return 1
        self.acc[k] = a
        return 0

    def allf(self, head, i):
        return self._bern(CNT.split(self._row(head, i), head)[2])

    def count(self, head, i, m):
        return self._count(CNT.split(self._row(head, i), head)[1], m, head)

    def land(self):
        lg = self.o["land_logits"][0]
        if self.land_acc is None or self.smp:
            return self._bern(lg)
        a = min(1.0, self.land_acc.get("land", 0.0) + float(torch.sigmoid(lg)))
        if a >= 0.5:
            self.land_acc["land"] = a - 1.0
            return 1
        self.land_acc["land"] = a
        return 0


class Replayer:
    """Decisions from a recorded latent -- recomputation of the same turn from the state."""

    def __init__(self, lat):
        self.lat = lat
        self.j = 0                   # index of the current market decision

    def targets(self):
        return self.lat["tgt"]

    def op(self, k, mask):
        return int(self.lat["op"][k])

    def qty(self, k, m):
        return int(self.lat["q"][k])

    def gate(self, head, i):
        return int(self.lat["m_gate"][self.j])

    def allf(self, head, i):
        return int(self.lat["m_all"][self.j])

    def count(self, head, i, m):
        return int(self.lat["m_n"][self.j])

    def land(self):
        return int(self.lat["land"])


_structure_free = structure_free     # one function with labeling (masks.py)


def _step(x, y, tgt):
    """Step towards the target by the X-then-Y rule: 100.000% of the teacher's steps are such."""
    tx, ty = tgt % BOARD, tgt // BOARD
    if x != tx:
        return ["EAST" if tx > x else "WEST"]
    return ["SOUTH" if ty > y else "NORTH"]


# The hour from which the feed rule in risk mode buys for all unfed animals.
# A decoder setting (`play.DECODE_DEFAULTS["feed_guard_hour"]`, travels in the package),
# not an environment variable: FARMCORE_FEED_GUARD_HOUR changed the hour for the opponent too
# (the module is shared) and did not reach the package (29.09).
FEED_GUARD_HOUR = 18
if _os.environ.get("FARMCORE_FEED_GUARD_HOUR"):
    raise SystemExit("FARMCORE_FEED_GUARD_HOUR was removed 29.09: the feed rule hour is a setting "
                     "of the decoder feed_guard_hour (play.py --feed-guard-hour, build.py)")


def _step_of(obs):
    return int(obs.get("step", obs.get("day", 0) * envcfg.TURNS_PER_DAY + obs.get("hour", 0)))


def _sim_turn(obs, action, want_market=False):
    """Our turn by engine code on a copy (without the opponent): -> (farm, private) after
    unit jobs and the market, BEFORE the night dump. Prices -- without the opponent's turn."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    p = obs["player"]
    farm = copy.deepcopy(obs["farms"][p])
    private = copy.deepcopy(obs.get("private") or {})
    private.setdefault("shed", {}); private.setdefault("seeds", {})
    private.setdefault("inventories", [{}])
    market = copy.deepcopy(obs["market"])
    board = len(farm["tiles"])
    ua = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
    for idx, a in enumerate(ua):
        K._apply_unit_action(farm, private, idx, a, board, obs["day"], envcfg.TURNS_PER_DAY,
                             envcfg.SHED_CAP)
    for o in action.get("market") or []:
        kind = o[0]
        if kind == "HIRE":
            K._do_hire(farm, private, board, envcfg.HAND_COST_MULT)
            continue
        if kind == "BUY_LAND":
            K._do_buy_land(farm, board)
            continue
        item, want = o[1], int(o[2])
        for _ in range(want):
            if kind == "SELL":
                px = market_price(item, market["inventory"][item], market.get("params"))
            elif kind == "BUY_PRODUCT":
                if item not in BUYABLE:
                    break
                px = market_price(item, market["inventory"][item] - 1, market.get("params"))
            elif kind == "BUY_SEED":
                px = CROPS[item]["seed"]
            else:
                px = ANIMALS[item]["cost"]
            if not K._commit_unit(kind, item, px, farm, private, market, envcfg.SHED_CAP):
                break
    return (farm, private, market) if want_market else (farm, private)


NIGHT_ROOM_MIN_GAIN = 5.0     # $: save a unit only if it is worth more than the loss from selling


def night_room_value(obs, action, info=None, min_gain=NIGHT_ROOM_MIN_GAIN):
    """Smart version of `night_room` (30.09): what burns and what to sell -- by price.

    1. The night dump is replayed in engine order (`_drop_inventories_to_shed`:
       units in turn, goods in order) -- the burning units are known; room
       freed by a sale saves them in the same order.
    2. Selling from the shed is not a loss but money now; its cost is the knocked-down price: an extra
       unit of good g moves the market, and our remaining g gets cheaper by Δq·(remainder).
    3. A unit is saved if its price now exceeds this loss by `min_gain`;
       the good with the smallest loss is sold. Fertilizer and wheat for feed -- never.
    Edits `action["market"]` in place. -> number of added sell units."""
    farm, private, market = _sim_turn(obs, action, want_market=True)
    cap = envcfg.SHED_CAP
    shed = {g: int(v) for g, v in private["shed"].items() if v > 0}
    cur, lost = sum(shed.values()), []
    for inv in private["inventories"]:
        for item, n in (inv or {}).items():
            n = int(n)
            if n <= 0:
                continue
            take = min(n, max(0, cap - cur))
            cur += take
            lost += [item] * (n - take)
    if not lost:
        return 0
    params = market.get("params")
    minv = dict(market["inventory"])
    hold = dict(shed)                       # our stock of the good: shed + hands
    for inv in private["inventories"]:
        for item, n in (inv or {}).items():
            hold[item] = hold.get(item, 0) + max(0, int(n))
    n_an = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and "animal" in t)
    free = {g: shed.get(g, 0) - (n_an if g == "WHEAT" else 0) for g in PRODUCTS if g != "FERTILIZER"}
    # Room freed by k sales saves the FIRST k burning units (engine
    # order), so a prefix is chosen: the k with the greatest total gain.
    plan, run, best_run, best_k = [], 0.0, 0.0, 0
    for item in lost:
        v = market_price(item, minv[item], params)          # price of the unit being saved
        best, best_cost = None, None
        for g, f in free.items():
            if f <= 0:
                continue
            q0 = market_price(g, minv[g], params)
            q1 = market_price(g, minv[g] + 1, params)
            cost = max(0.0, q0 - q1) * max(0, hold.get(g, 0) - 1)
            if best is None or cost < best_cost:
                best, best_cost = g, cost
        if best is None:
            break
        free[best] -= 1
        hold[best] -= 1
        minv[best] += 1
        plan.append(best)
        run += v - best_cost - min_gain
        if run > best_run:
            best_run, best_k = run, len(plan)
    add = {}
    for g in plan[:best_k]:
        add[g] = add.get(g, 0) + 1
    mk = action["market"]
    added = 0
    for g, n in add.items():
        ex = next((o for o in mk if o[0] == "SELL" and o[1] == g), None)
        if ex is not None:
            ex[2] = int(ex[2]) + n
        elif len(mk) < envcfg.MAX_ORDERS:
            mk.insert(0, ["SELL", g, n])
        else:
            continue
        added += n
    if info is not None and added:
        info["env"]["night(price): sale for the hand dump"] = info["env"].get(
            "night(price): sale for the hand dump", 0) + 1
        info["env"]["night(price): units sold"] = info["env"].get("night(price): units sold", 0) + added
    return added


def night_room(obs, action, info=None):
    """Rule 30.09 (endgame/night): last turn of the day, after the market the engine dumps
    hands into the shed, the excess beyond capacity is DESTROYED (_drop_inventories_to_shed).
    On the ladder Champion·Guild lose ~20 units of goods per game this way (days 14-28).
    If the hands do not fit into the shed after the turn -- we sell the same number of units from the shed
    in this very turn (the market goes before the night dump): the good in the shed is chosen
    by the greatest remainder above the wheat stock for feed (per head of animal).
    From the hour `night_room_hour` (< 23) the rule acts earlier too: hands anyway
    are dumped by night, while a full shed in the daytime keeps loaded units without a drop.
    Edits `action["market"]` in place. -> number of added sell units."""
    farm, private = _sim_turn(obs, action)
    hands = sum(int(v) for iv in private["inventories"] for v in (iv or {}).values() if v > 0)
    shed = {g: int(v) for g, v in private["shed"].items() if v > 0}
    over = hands + sum(shed.values()) - envcfg.SHED_CAP
    if over <= 0:
        return 0
    n_an = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and "animal" in t)
    free = {g: shed.get(g, 0) - (n_an if g == "WHEAT" else 0) for g in PRODUCTS}
    add = {}
    for _ in range(over):
        g = max(PRODUCTS, key=lambda g_: free[g_])
        if free[g] <= 0:
            break
        free[g] -= 1
        add[g] = add.get(g, 0) + 1
    mk = action["market"]
    added = 0
    for g, n in add.items():
        ex = next((o for o in mk if o[0] == "SELL" and o[1] == g), None)
        if ex is not None:
            ex[2] = int(ex[2]) + n
        elif len(mk) < envcfg.MAX_ORDERS:
            # sale goes to the head of the list: the engine executes by index, buys after
            mk.insert(0, ["SELL", g, n])
        else:
            continue
        added += n
    if info is not None and added:
        info["env"]["night: sale for the hand dump"] = info["env"].get(
            "night: sale for the hand dump", 0) + 1
        info["env"]["night: units sold"] = info["env"].get("night: units sold", 0) + added
    return added


def run_turn(obs, pick, order=None, feed_guard=False, end_sell=False, feed_guard_risk=False,
             feed_guard_hour=FEED_GUARD_HOUR, end_haul=False, night_room_on=False,
             night_room_hour=23, night_room_mode="stock"):
    """Assemble the turn. -> (action for the engine, latent, summary).

    One and the same function both in play (`pick` = Sampler) and in recomputation
    (`pick` = Replayer): the masks of each decision are built by ONE piece of code.
    Summary: `clip` -- what had to be clipped after the choice (after R3 should be
    empty; non-empty means a mask error), `env` -- environment substitutions,
    once a habit closed a job that the net would have chosen by the rules alone.
    """
    p = obs["player"]
    farm = obs["farms"][p]
    priv = obs.get("private") or {}
    shed0 = dict(priv.get("shed") or {})
    seeds = dict(priv.get("seeds") or {})
    invs = priv.get("inventories") or [{}]
    pos = [tuple(q) for q in ([farm["farmer"]] + farm["hands"])]
    tiles = farm["tiles"]
    lat = empty_lat()
    info = {"clip": {}, "env": {}}

    def clip(kind, want, got):
        if want > got:
            c = info["clip"].setdefault(kind, [0, 0, 0])
            c[0] += 1; c[1] += want - got; c[2] += want
        return got

    def env_sub(kind):
        info["env"][kind] = info["env"].get(kind, 0) + 1

    targets = (pick.targets(n_live=min(len(pos), MAX_UNITS)) if isinstance(pick, Sampler)
               else pick.targets())
    acts = []
    shed = dict(shed0)               # shed AFTER the jobs of units 0..k-1
    for k in range(len(pos)):
        if k >= MAX_UNITS:           # the net gives outputs only for MAX_UNITS units
            acts.append(["PASS"])
            continue
        x, y = pos[k]
        tgt = int(targets[k])
        if end_haul and isinstance(pick, Sampler):
            # Season end (29.09, review): goods in hands on the last turn are not
            # sold -- rn_ctl loses $358-1,398 per game. A loaded unit that has
            # no more turns until the last decision than it takes to walk to the shed
            # (+1 for the drop) goes to the nearest shed cell and drops the cargo;
            # end_sell sells the whole shed on the last turn.
            inv_h = invs[k] if k < len(invs) else {}
            load = sum(v for g_, v in (inv_h or {}).items() if g_ in PRODUCTS)
            left = envcfg.SEASON_DAYS * envcfg.TURNS_PER_DAY - 2 - _step_of(obs)
            if load > 0 and left >= 0:
                sx, sy = min(envcfg.shed_tiles(BOARD), key=lambda q: abs(q[0] - x) + abs(q[1] - y))
                dist = abs(sx - x) + abs(sy - y)
                if dist + 1 >= left:
                    lat["u_live"][k] = True
                    if dist > 0:
                        acts.append(_step(x, y, sy * BOARD + sx))
                        env_sub("season end: to the shed")
                        continue
                    room = max(0, envcfg.SHED_CAP - sum(shed.values()))
                    if room > 0:
                        if sum(inv_h.values()) <= room:
                            acts.append(["DROP"])
                        else:
                            best = max((g_ for g_ in inv_h if g_ in PRODUCTS), key=lambda g_: (
                                market_price(g_, obs["market"]["inventory"][g_],
                                             obs["market"].get("params"))))
                            acts.append(["PLACE", best, min(inv_h[best], room)])
                        env_sub("season end: drop")
                        shed = shed_at_market(shed0, acts, invs, pos, tiles)
                        continue
        lat["u_live"][k] = True
        lat["tgt"][k] = tgt
        lat["u_pos"][k] = y * BOARD + x
        if tgt != STAY and tgt != y * BOARD + x:
            acts.append(_step(x, y, tgt))
            continue
        inv_k = invs[k] if k < len(invs) else {}
        ok = legal_ops(obs, p, k, pos[k], inv_k, seeds, shed)
        mask = np.array([nm in ok for nm in UNIT_OPS], dtype=bool)
        j = pick.op(k, mask)
        lat["op_on"][k] = True
        lat["op"][k] = j
        lat["op_mask"][k] = mask
        op = UNIT_OPS[j]
        room = max(0, envcfg.SHED_CAP - sum(shed.values()))

        def qty(m):
            lat["q_on"][k] = True
            lat["q_max"][k] = max(1, m)
            n = pick.qty(k, max(1, m))
            lat["q"][k] = n
            return clip("unit quantity outside the cap", n, min(n, max(1, m)))

        if op.startswith("PLANT_"):
            c = op[len("PLANT_"):]
            seeds[c] -= 1            # the mask guarantees seeds[c] > 0
            acts.append(["PLANT", c])
        elif op.startswith("PICKUP_"):
            g = op[len("PICKUP_"):]
            acts.append(["PICKUP", g, qty(int(shed.get(g, 0)))])
        elif op.startswith("PLACE_"):
            it = op[len("PLACE_"):]
            if it in ANIMALS and _structure_free(tiles[y][x], it):
                # on a building: the engine ignores the quantity, the net does not choose it
                acts.append(["PLACE", it, 1])
            else:
                acts.append(["PLACE", it, qty(min(int(inv_k.get(it, 0)), room))])
        elif op == "DROP":
            # `_apply_unit_action`: DROP removes the WHOLE stack from the inventory, but puts into the shed
            # only what fits -- the excess is destroyed. PLACE puts
            # one good with a quantity and leaves the remainder with the unit, so when
            # there is not enough room we drop the most expensive one by address. This is a substitution
            # by the environment: it is deterministic given the latent and the state.
            inv2 = inv_k or {}
            if sum(inv2.values()) <= room:
                acts.append(["DROP"])
            elif room > 0 and inv2:
                best = max(inv2, key=lambda g: (
                    market_price(g, obs["market"]["inventory"][g],
                                 obs["market"].get("params"))
                    if g in PRODUCTS else 0.0))
                acts.append(["PLACE", best, min(inv2[best], room)])
                env_sub("DROP -> PLACE: does not fit in the shed")
            else:
                acts.append(["PASS"])
                env_sub("DROP -> PASS: shed is full")
        else:
            acts.append([op])
        # The shed is visible to the next units of the turn: the engine applies them in turn.
        # It is computed by the same function as the shed at the market moment and labeling.
        shed = shed_at_market(shed0, acts, invs, pos, tiles)

    step = int(obs.get("step", obs.get("day", 0) * envcfg.TURNS_PER_DAY + obs.get("hour", 0)))
    last_turn = step >= envcfg.SEASON_DAYS * envcfg.TURNS_PER_DAY - 2

    # ---------------- market ----------------
    # Shed at the MOMENT OF MARKET EXECUTION: the engine processes orders AFTER
    # the actions of all units (env_ref.py:935-941).
    shed_mkt = shed_at_market(shed0, acts, invs, pos, tiles)
    params = obs["market"].get("params")
    st_ = {"inv": dict(obs["market"]["inventory"]), "budget": float(farm["money"]),
           "room": max(0, envcfg.SHED_CAP - sum(shed_mkt.values())),
           "slots": envcfg.MAX_ORDERS}
    orders = []

    def decide(head, i, m, has_all=False):
        """One market decision with cap m (0 = closed). -> quantity.
        The decision index in the latent is by the MKT layout (`_MKT_SPAN`), not by call
        order: the execution order of blocks is configurable (MARKET_ORDER), the layout is not."""
        j = _MKT_SPAN[head][0] + i
        if isinstance(pick, Replayer):
            pick.j = j
        m = min(int(m), CNT.ALL_CAPS[head]) if st_["slots"] > 0 else 0
        lat["m_max"][j] = max(0, m)
        if m <= 0:
            return 0
        if not pick.gate(head, i):
            return 0
        lat["m_gate"][j] = 1
        if has_all:
            if pick.allf(head, i):
                lat["m_all"][j] = 1
                lat["m_n"][j] = m        # executed quantity (see _count_logq)
                return m
        n = pick.count(head, i, m)
        lat["m_n"][j] = n
        return clip(f"{head}: outside the cap", n, min(n, m))

    def blk_sell():
        for i, g in enumerate(PRODUCTS):
            n = decide("sell", i, int(shed_mkt.get(g, 0)), has_all=True)
            if n > 0:
                orders.append(["SELL", g, n])
                st_["slots"] -= 1
                st_["room"] += n
                for _ in range(n):       # engine price path (_commit_unit)
                    px = market_price(g, st_["inv"][g], params)
                    st_["budget"] += px
                    if px > 1:
                        st_["inv"][g] += 1

    def blk_hire():
        # hire: each hand is a separate order, the price grows by Fibonacci
        mult = envcfg.HAND_COST_MULT
        m, cost = 0, 0.0
        while m < min(st_["slots"], CNT.ALL_CAPS["hire"]):
            c = _hire_cost(farm["hires_today"] + m, mult)
            if cost + c > st_["budget"]:
                break
            cost += c
            m += 1
        n = decide("hire", 0, m)
        for j in range(n):
            st_["budget"] -= _hire_cost(farm["hires_today"] + j, mult)
            orders.append(["HIRE"])
        st_["slots"] -= n

    def blk_seed():
        for i, c in enumerate(CROP_LIST):
            price = CROPS[c]["seed"]
            n = decide("seed", i, int(st_["budget"] // max(1, price)))
            if n > 0:
                orders.append(["BUY_SEED", c, n])
                st_["budget"] -= n * price
                st_["slots"] -= 1

    def blk_buy():
        for i, g in enumerate(PRODUCTS):
            m, spent = 0, 0.0
            if g in BUYABLE:
                # the engine quotes a buy at inventory-1, each next unit is
                # more expensive; short of money for the next -- the order is cut off
                while m < min(st_["room"], CNT.ALL_CAPS["buy"]):
                    px = market_price(g, st_["inv"][g] - 1 - m, params)
                    if spent + px > st_["budget"]:
                        break
                    spent += px
                    m += 1
            n = decide("buy", i, m)
            if feed_guard and g == "WHEAT" and m > n:
                # A game rule, not teacher taste (28.09): an animal not fed
                # for two days in a row runs away (_daily_refresh_animals), FEED takes
                # one wheat from hands. Teacher 56521745 always has its own wheat,
                # and the state "shed empty, herd hungry" is absent from the corpus -- the clone
                # stood in it for days and lost the herd (seed 3021: 15 heads -> 2).
                # We buy exactly what is missing for animals unfed TODAY.
                n_feed = sum(1 for a_ in acts if a_ and a_[0] == "FEED")
                # The risk is only for an animal that will run away tonight: not fed
                # since yesterday (consecutive_unfed >= 1), or the day is running out. Before, the rule
                # bought for EVERY not-yet-fed animal every turn, and a teacher
                # feeding with its own harvest later the same day (56254996) got
                # 14,372 extra wheat vs 547 for itself.
                # The risk mode is a package setting (teacher "for the task", 56254996);
                # by default the former behaviour: for 56521745 early top-ups are needed
                # (narrowed rule on t45_big -$1.5k, 23/48).
                late = (not feed_guard_risk) or obs.get("hour", 0) >= feed_guard_hour
                hungry = sum(1 for row in tiles for tl in row
                             if isinstance(tl, dict) and tl.get("animal")
                             and not tl.get("fed_today")
                             and (late or tl.get("consecutive_unfed", 0) >= 1)) - n_feed
                have = int(shed_mkt.get("WHEAT", 0)) + sum(
                    int((iv or {}).get("WHEAT", 0)) for iv in invs) - n_feed
                need = min(m, max(0, hungry - have))
                if need > n:
                    n = need
                    env_sub("feed: wheat top-up")
            if n > 0:
                orders.append(["BUY_PRODUCT", g, n])
                st_["budget"] -= sum(market_price(g, st_["inv"][g] - 1 - t, params) for t in range(n))
                st_["inv"][g] -= n
                st_["room"] -= n
                st_["slots"] -= 1

    def blk_animal():
        for i, an in enumerate(ANIMAL_LIST):
            cost = ANIMALS[an]["cost"]
            n = decide("animal", i, min(int(st_["budget"] // cost), st_["room"]))
            if n > 0:
                orders.append(["BUY_ANIMAL", an, n])
                st_["budget"] -= n * cost
                st_["room"] -= n
                st_["slots"] -= 1

    def blk_land():
        n_unl = len(farm["unlocked_quadrants"])
        ok_land = (n_unl < 4 and st_["slots"] > 0 and st_["budget"] >= envcfg.land_price(n_unl))
        lat["land_ok"] = np.array(ok_land)
        if ok_land and pick.land():
            lat["land"] = np.array(1)
            orders.append(["BUY_LAND"])
            st_["budget"] -= envcfg.land_price(n_unl)
            st_["slots"] -= 1

    blocks = {"sell": blk_sell, "hire": blk_hire, "seed": blk_seed, "buy": blk_buy,
              "animal": blk_animal, "land": blk_land}
    if end_sell and last_turn:
        # The last decision of the season (28.09): the score is money in the bank, goods in
        # the shed after it are worth zero, and a buy will not manage to bring anything.
        # We sell the whole shed at the market moment, the other blocks are silent.
        def blk_sell_all():
            for g in PRODUCTS:
                n = int(shed_mkt.get(g, 0))
                if n > 0 and st_["slots"] > 0:
                    orders.append(["SELL", g, n])
                    st_["slots"] -= 1
            env_sub("season end: sell all")
        blocks = {k: (lambda: None) for k in blocks}
        blocks["sell"] = blk_sell_all
    order = tuple(order) if order else MARKET_ORDER
    assert sorted(order) == sorted(blocks), f"market order {order}: need exactly {sorted(blocks)}"
    for name in order:
        blocks[name]()
    if len(orders) > envcfg.MAX_ORDERS:           # slots in the caps: must not happen
        clip("tail beyond the order limit", len(orders), envcfg.MAX_ORDERS)
        orders = orders[:envcfg.MAX_ORDERS]
    action = {"farmer": acts[0] if acts else ["PASS"], "hands": acts[1:],
              "market": orders}
    if (night_room_on and isinstance(pick, Sampler) and not last_turn
            and int(obs.get("hour", step % envcfg.TURNS_PER_DAY)) >= night_room_hour):
        (night_room_value if night_room_mode == "value" else night_room)(obs, action, info)
    return action, lat, info


def stack(lats, device="cpu"):
    """List of latents -> a batch of tensors for `log_prob`."""
    return {k: torch.from_numpy(np.stack([np.asarray(l[k]) for l in lats])).to(device)
            for k in lats[0]}


def log_prob(out, lat, sample=True, sample_units=0.0, sample_qty=0.0):
    """`log π` of the turn from the network output and the latent, in a batch. -> [B] float32.

    The same function both in play (batch 1, right after sampling) and in recomputation during
    training. Temperatures are those with which the turn was chosen: a decision taken
    deterministically (argmax, median, threshold) has probability 1 and gives no
    contribution. Masks are taken from the latent -- exactly those the decision saw in play.
    Logits are already brought by `family` (β·prior and the gate lower bound), the same
    numbers the sampling saw.
    """
    _need_family(out)
    dev = out["op_logits"].device
    lat = {k: v.to(dev) for k, v in lat.items()}
    B = lat["tgt"].shape[0]
    lp = torch.zeros(B, device=dev)
    z = lambda t: torch.zeros_like(t)
    if sample_units > 0:
        lc = _unit_logc(out["target_logits"], lat["u_pos"], sample_units)
        cls = unit_class(lat["u_pos"], out["target_logits"].shape[-1]).gather(
            -1, lat["tgt"].unsqueeze(-1))
        lt = lc.gather(-1, cls).squeeze(-1)
        lp = lp + torch.where(lat["u_live"], lt, z(lt)).sum(1)
        ol = out["op_logits"].float().masked_fill(~lat["op_mask"], float("-inf"))
        lo = torch.log_softmax(ol / sample_units, -1).gather(
            -1, lat["op"].unsqueeze(-1)).squeeze(-1)
        lp = lp + torch.where(lat["op_on"], lo, z(lo)).sum(1)
    if sample_qty > 0:
        _, cond, _ = CNT.split(out["qty_logits"], "pickup")
        lq = CNT.log_prob(cond, lat["q"], lat["q_max"], sample_qty, head="pickup")
        lp = lp + torch.where(lat["q_on"], lq, z(lq)).sum(1)
    if not sample:
        return lp
    for head in ("sell", "hire", "seed", "buy", "animal"):
        a, b = _MKT_SPAN[head]
        lg = out[f"{head}_logits"].float()
        if head == "hire":
            lg = lg.unsqueeze(1)
        gate, cond, allf = CNT.split(lg, head)
        tg = sample_qty if sample_qty > 0 else 1.0        # one τ for gates and quantities
        gate = gate / tg
        allf = None if allf is None else allf / tg
        op_ = lat["m_max"][:, a:b] > 0
        g1 = lat["m_gate"][:, a:b] > 0
        lg_ = torch.where(g1, F.logsigmoid(gate), F.logsigmoid(-gate))
        lp = lp + torch.where(op_, lg_, z(lg_)).sum(1)
        need_n = op_ & g1
        if sample_qty > 0:
            # by the executed quantity: "all" and "n = m" are one order
            mm = lat["m_max"][:, a:b].clamp_min(1)
            lq = _count_logq(cond, allf, mm, sample_qty, head)
            ln = lq.gather(-1, CNT.label(head, lat["m_n"][:, a:b].clamp_min(1)).unsqueeze(-1)).squeeze(-1)
            lp = lp + torch.where(need_n, ln, z(ln)).sum(1)
        elif allf is not None:
            a1 = lat["m_all"][:, a:b] > 0
            la = torch.where(a1, F.logsigmoid(allf), F.logsigmoid(-allf))
            lp = lp + torch.where(need_n, la, z(la)).sum(1)
    ll = out["land_logits"].float().squeeze(-1) / (sample_qty if sample_qty > 0 else 1.0)
    ll = torch.where(lat["land"] > 0, F.logsigmoid(ll), F.logsigmoid(-ll))
    return lp + torch.where(lat["land_ok"], ll, z(ll))


def _bern_h(l):
    """Bernoulli entropy from a logit."""
    p, lp, lq = torch.sigmoid(l), F.logsigmoid(l), F.logsigmoid(-l)
    return -(p * lp + (1 - p) * lq)


def _bern_kl(l, r):
    """KL(Bernoulli(l) ‖ Bernoulli(r)) from logits."""
    p = torch.sigmoid(l)
    return p * (F.logsigmoid(l) - F.logsigmoid(r)) + (1 - p) * (
        F.logsigmoid(-l) - F.logsigmoid(-r))


# ⚠ Outside the mask the log-probability is −inf, and the product `p · log p` there gives
# 0 · (−inf) = nan. `torch.where` hides this in the VALUE but not in the GRADIENT:
# the derivative of a product with respect to one factor equals the other, i.e. −inf,
# and multiplying by a zero upstream gradient gives nan again. Measurement 23.09
# (our probes): the gradient norm of entropy and KL over the policy was nan, it was
# zeroed by `nan_to_num_` in the trainer, and of 275 tensors 11 moved --
# exactly the value head. So the MULTIPLIER is zeroed before the multiplication.
def _safe(logp, live):
    return torch.where(live, logp, torch.zeros_like(logp))


def _cat_h(logp):
    """Entropy of a categorical from log-probabilities (zeros outside the mask do no harm)."""
    p = logp.exp()
    return -(p * _safe(logp, p > 0)).sum(-1)


def _cat_kl(logp, logq):
    """KL(p ‖ q) from log-probabilities. Supports coincide: the mask is the same."""
    p = logp.exp()
    return (p * _safe(logp - logq, p > 0)).sum(-1)


# ---------------------------------------------------------------------------
# Probability of the EXECUTED action (25.09). The decoder chooses internal
# quantities that the environment does not see in full: from the target tile the engine gets one step
# (`_step`, the X-then-Y rule) or "work here" (STAY and one's own tile give the same
# thing); for a sale "all" and "quantity = m" give one order. Sampling still goes
# through internal quantities, the turn distribution does not change because of
# this, while `log π`, entropy and KL are computed over classes of the executed
# action: log p(step) = logsumexp over the tiles giving that step. Otherwise the gradient
# moves mass BETWEEN tiles of one step, to which the reward does not respond
# at all -- pure noise: 31% of target entropy per game in PPO mode
# (value_tower e11, seed 3000).
N_UCLS = 5                           # here, E, W, S, N
_FAR = -1e4                          # "no mass": finite, so as not to catch nan in the gradient


def unit_class(pos, T):
    """Class of the executed unit action for each target tile. pos [...] ->
    [..., T]: 0 -- here (STAY or own tile), 1 E, 2 W, 3 S, 4 N (as `_step`)."""
    t = torch.arange(T, device=pos.device)
    tx, ty = t % BOARD, t // BOARD
    p = pos.long().unsqueeze(-1)
    x, y = p % BOARD, p // BOARD
    here = (t == STAY) | (t == p)
    return torch.where(here, 0, torch.where(tx > x, 1, torch.where(
        tx < x, 2, torch.where(ty > y, 3, 4))))


def _unit_logc(logits, pos, tau):
    """log p(class of the executed action) of a unit. logits [B,U,T] -> [B,U,5].
    An empty class gets _FAR (exp gives exactly 0)."""
    lt = torch.log_softmax(logits.float() / tau, -1)
    cls = unit_class(pos, lt.shape[-1])                                # [B,U,T]
    m = cls.unsqueeze(-2) == torch.arange(N_UCLS, device=lt.device).view(-1, 1)
    x = lt.unsqueeze(-2).masked_fill(~m, _FAR)                         # [B,U,5,T]
    return torch.logsumexp(x, -1)


def _count_logq(cond, allf, m, tau, head):
    """log q(class) of the executed quantity with open gates. -> [..., C].

    Categorical over classes with cap m; if there is an "all" flag (sale), then
    q(c) = (1 − a)·p(c) + a·[value_c = m]: "all" and "n = m" are one order.
    If there is no class with value m (m off the grid), "all" is a separate order and
    its mass is not included in q over classes (log π takes it through the flag)."""
    lp = CNT.log_pmf(cond, m, tau, head)
    if allf is None:
        return lp
    la = F.logsigmoid(allf.float()).unsqueeze(-1)
    lna = F.logsigmoid(-allf.float()).unsqueeze(-1)
    v = CNT.values(head, cond.device)
    is_m = v == torch.as_tensor(m, device=cond.device).long().unsqueeze(-1)
    lp = lna + lp
    return torch.where(is_m, torch.logaddexp(la, lp), lp)


def _walk(out, lat, ref, sample, sample_units, sample_qty):
    """Walk over the turn factors once and collect H or KL. -> [B].

    One walk for both quantities: the set of factors and masks are exactly the same as in
    `log_prob` (the row order deliberately repeats it), only what is computed
    on a factor changes. `ref = None` is entropy, otherwise KL(current ‖ ref).

    This is a sum of CONDITIONAL quantities along the path recorded in the latent: which factors
    exist at all (the gate opened -- there is a quantity) depends on the turn's
    choices. For entropy this is an unbiased estimate of H(turn | s), for KL -- unbiased
    at π_new = π_old, i.e. exact at the zeroth PPO step, where the equality to zero
    at the anchor is checked (N10).
    """
    _need_family(out)
    if ref is not None:
        _need_family(ref)
    dev = out["op_logits"].device
    lat = {k: v.to(dev) for k, v in lat.items()}
    B = lat["tgt"].shape[0]
    acc = torch.zeros(B, device=dev)
    z = lambda t: torch.zeros_like(t)
    one = lambda a, b=None: (_cat_h(a) if ref is None else _cat_kl(a, b))
    bern = lambda a, b=None: (_bern_h(a) if ref is None else _bern_kl(a, b))

    def ch(cond, cref, m, tau, head):
        lp = CNT.log_pmf(cond, m, tau, head)
        return (_cat_h(lp) if ref is None else _cat_kl(lp, CNT.log_pmf(cref, m, tau, head)))

    if sample_units > 0:
        tl = _unit_logc(out["target_logits"], lat["u_pos"], sample_units)
        rl = (None if ref is None else
              _unit_logc(ref["target_logits"], lat["u_pos"], sample_units))
        v = one(tl, rl)
        acc = acc + torch.where(lat["u_live"], v, z(v)).sum(1)
        ol = out["op_logits"].float().masked_fill(~lat["op_mask"], float("-inf"))
        ol = torch.log_softmax(ol / sample_units, -1)
        rl = None
        if ref is not None:
            rl = ref["op_logits"].float().masked_fill(~lat["op_mask"], float("-inf"))
            rl = torch.log_softmax(rl / sample_units, -1)
        v = one(ol, rl)
        acc = acc + torch.where(lat["op_on"], v, z(v)).sum(1)
    if sample_qty > 0:
        _, cond, _ = CNT.split(out["qty_logits"], "pickup")
        cref = None if ref is None else CNT.split(ref["qty_logits"], "pickup")[1]
        v = ch(cond, cref, lat["q_max"], sample_qty, "pickup")
        acc = acc + torch.where(lat["q_on"], v, z(v)).sum(1)
    if not sample:
        return acc
    for head in ("sell", "hire", "seed", "buy", "animal"):
        a, b = _MKT_SPAN[head]
        lg = out[f"{head}_logits"].float()
        rg = None if ref is None else ref[f"{head}_logits"].float()
        if head == "hire":
            lg = lg.unsqueeze(1)
            rg = None if rg is None else rg.unsqueeze(1)
        gate, cond, allf = CNT.split(lg, head)
        r_gate, r_cond, r_allf = (None, None, None) if rg is None else CNT.split(rg, head)
        tg = sample_qty if sample_qty > 0 else 1.0
        gate, allf = gate / tg, (None if allf is None else allf / tg)
        if rg is not None:
            r_gate, r_allf = r_gate / tg, (None if r_allf is None else r_allf / tg)
        op_ = lat["m_max"][:, a:b] > 0
        g1 = lat["m_gate"][:, a:b] > 0
        v = bern(gate, r_gate)
        acc = acc + torch.where(op_, v, z(v)).sum(1)
        need_n = op_ & g1
        if sample_qty > 0:
            mm = lat["m_max"][:, a:b].clamp_min(1)
            lq = _count_logq(cond, allf, mm, sample_qty, head)
            v = one(lq, None if ref is None else _count_logq(r_cond, r_allf, mm, sample_qty, head))
            acc = acc + torch.where(need_n, v, z(v)).sum(1)
        elif allf is not None:
            v = bern(allf, r_allf)
            acc = acc + torch.where(need_n, v, z(v)).sum(1)
    tg = sample_qty if sample_qty > 0 else 1.0
    ll = out["land_logits"].float().squeeze(-1) / tg
    rl = None if ref is None else ref["land_logits"].float().squeeze(-1) / tg
    v = bern(ll, rl)
    return acc + torch.where(lat["land_ok"], v, z(v))


def entropy(out, lat, sample=True, sample_units=0.0, sample_qty=0.0):
    """H(turn | s) along the latent path, analytically over factors. -> [B] float32.

    Analytic, not by samples: a turn has ~29 factors, and an estimate from a single
    sample is noisier than the quantity itself. Deterministic decisions (τ = 0, median,
    threshold) give no contribution -- exactly as in `log_prob`.
    """
    return _walk(out, lat, None, sample, sample_units, sample_qty)


def kl(out, ref, lat, sample=True, sample_units=0.0, sample_qty=0.0):
    """KL(π ‖ π_ref) over the latent -- the leash to the BC anchor (R7, N9). -> [B] float32.

    Computed over THE SAME masks from the latent and at the same temperature as
    `log_prob`: the anchor must live in the same family of distributions, otherwise the
    leash pulls towards a flatter policy than the starting one. On the anchor itself
    (`ref` = the same output) it equals zero bitwise -- this is acceptance N10.
    """
    return _walk(out, lat, ref, sample, sample_units, sample_qty)


def engine_check(obs, action):
    """Execute OUR turn by engine code on a copy of the state, without the opponent.

    -> {order kind: [decisions underexecuted, units lost, requested]}.
    After R3 it must be empty: everything the net chose, the engine executes
    in full, unless the opponent moved prices in the same turn. Unit jobs
    go through `_apply_unit_action` -- so the shed at the moment of the
    market, by which the caps were computed, is checked too.
    """
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    p = obs["player"]
    farm = copy.deepcopy(obs["farms"][p])
    private = copy.deepcopy(obs.get("private") or {})
    private.setdefault("shed", {}); private.setdefault("seeds", {})
    private.setdefault("inventories", [{}])
    market = copy.deepcopy(obs["market"])
    board = len(farm["tiles"])
    day = obs["day"]
    ua = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
    for idx, a in enumerate(ua):
        K._apply_unit_action(farm, private, idx, a, board, day, envcfg.TURNS_PER_DAY,
                             envcfg.SHED_CAP)
    lost = {}
    for o in action.get("market") or []:
        kind = o[0]
        if kind == "HIRE":
            h0 = farm["hires_today"]
            K._do_hire(farm, private, board, envcfg.HAND_COST_MULT)
            got, want = farm["hires_today"] - h0, 1
        elif kind == "BUY_LAND":
            q0 = len(farm["unlocked_quadrants"])
            K._do_buy_land(farm, board)
            got, want = len(farm["unlocked_quadrants"]) - q0, 1
        else:
            item, want, got = o[1], int(o[2]), 0
            for _ in range(want):
                if kind == "SELL":
                    px = market_price(item, market["inventory"][item], market.get("params"))
                elif kind == "BUY_PRODUCT":
                    if item not in BUYABLE:
                        break
                    px = market_price(item, market["inventory"][item] - 1,
                                      market.get("params"))
                elif kind == "BUY_SEED":
                    px = CROPS[item]["seed"]
                else:
                    px = ANIMALS[item]["cost"]
                if not K._commit_unit(kind, item, px, farm, private, market,
                                      envcfg.SHED_CAP):
                    break
                got += 1
        if got < want:
            c = lost.setdefault(kind, [0, 0, 0])
            c[0] += 1; c[1] += want - got; c[2] += want
    return lost
