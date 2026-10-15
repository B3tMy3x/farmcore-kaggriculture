"""BC for FarmNet, version 2. What changed against the first one and why.

The first version was analysed in an internal net audit; each fix here closes
a specific point from it, not a matter of taste.

1. **One teacher.** The replays are episodes of submission 56263693, but v1 labeled BOTH players,
   i.e. cloned a mixture of 146 policies. Now by default only the teacher's side
   is learned (`--teacher only`).
2. **`extra` is passed into the features.** In v1 six of the eighteen goods features were
   constants, because `features.encode` was called without `extra`: external pressure,
   opponent sales and own sales, crowding. They are computed from the flow identity in
   `experiments/net/extract_features.py` (median error of the opponent supply estimate -1.4%).
3. **Class weights on the market heads.** 95% of labels there are zeros, and v1 collapsed into a
   constant (recall on nonzeros: sell 0.09, buy 0.01, seed 0.11). Weights are computed
   by a pre-pass over the data as (1/frequency)^alpha.
4. **CE is averaged first within a turn, then across turns** -- a trick from pokemoni
   (`bc_sequential_loss`): otherwise a turn with 12 units weighs twelvefold against a turn with one,
   and the net learns late days at the expense of early ones.
5. **Side weight by outcome** (1.5 win / 1.0 loss), also from pokemoni. It does NOT
   extend to the critic -- otherwise the state estimate is systematically inflated.
6. **Value: discounted outcome and Huber instead of MSE.** In v1 the head emitted the constant
   0.022 on any input. The target is `z * gamma^(n-1-i)` by the decision index within the side.
7. **Shuffle buffer.** v1 fed batches with consecutive turns of one game --
   64 neighbouring turns are one and the same moment of the game, the gradient is biased.
8. **Honest fold by episodes and metrics to a file.** Validation every epoch, a checkpoint with
   the optimizer, metrics in jsonl. Nothing was left of the first run except a line
   in a commit message.

The target pointer head is still not trained in BC: the replay has no "where the unit is going" labels.

Run: ./env/bin/python experiments/net/bc_train.py --epochs 3 -j 6 --run v2
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import features  # noqa: E402
import effects as EFF  # noqa: E402
from masks import legal_ops, structure_free  # noqa: E402
import counts as CNT  # noqa: E402
from features import MAX_UNITS, CROP_LIST  # noqa: E402
from model import FarmNet, ModelConfig, N_OPS, OP_NAMES  # noqa: E402
import value_head as VAL  # noqa: E402
from extract_features import own_flows, town_step  # noqa: E402
from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    CROPS, ANIMALS, PRODUCTS,
)

ANIMAL_LIST = list(ANIMALS)
# Directions are NOT included here: the step is computed from the target by the X-then-Y rule
# (internal movement-factorization note -- 100.000% of the teacher's steps are shortest,
# at a fork 100% along X). The op head answers only "what to do AT THE TARGET".
MOVE_NAMES = ("NORTH", "SOUTH", "EAST", "WEST")
# PICKUP and PLACE are spelled out per item by name. Previously seven goods of nine
# sat in one class PICKUP_OTHER, and the item was chosen in play.py by the rule
# "whichever is more in the shed", and PLACE_ANIMAL took the first suitable animal from the
# list -- a sheep could not be expressed when a cow was available. This is a limitation of the
# decoder, not of the engine: under PPO the policy must be able to name the item itself.
UNIT_OPS = (["PASS", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE",
             "DIG", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
            + [f"PICKUP_{g}" for g in PRODUCTS]
            + [f"PICKUP_{a}" for a in ANIMAL_LIST]
            + [f"PLACE_{a}" for a in ANIMAL_LIST]
            # Addressed drop into the shed: put N units of ONE good, leave the rest
            # in hands. Previously reduced to DROP, which dumps the whole
            # stack. For teacher 56216119 this is 1.375% of actions against 0.161% for
            # 56263693 -- he carries the good over, puts it down and KEEPS the
            # fertilizer in hands for FERTILIZE, whereas DROP would make him go for it
            # again. That also explains his 0.85% PASS against our 12.1%.
            + [f"PLACE_{g}" for g in PRODUCTS]
            + [f"PLANT_{c}" for c in CROP_LIST])
BOARD = 10
N_TILES = BOARD * BOARD
STAY = N_TILES              # the "no target" column in the pointer
OP_ID = {o: i for i, o in enumerate(UNIT_OPS)}
assert len(UNIT_OPS) == N_OPS, (len(UNIT_OPS), N_OPS)
# The composite action decomposes THIS SAME list into a verb and an object
# (model.VERB_OF/OBJ_OF). The order must match up to the class: labeling, mask and
# decoder are addressed by number, and a silent shift by one class would give a model
# that learns to water but plays digging.
assert list(OP_NAMES) == UNIT_OPS, "class order diverged from model.OP_NAMES"

# Full output over quantities instead of bins. Bins erased 56.2% of the teacher's purchases:
# _bin took the nearest edge, and quantities 1,2,3,5 fell into the class "do not buy",
# even though the most frequent quantity is 2. Caps chosen by percentiles
# (measured on 25 games): SELL 99% <= 28 (max 44), BUY 99% <= 48 (max 96),
# SEED 99% <= 7, ANIMAL max 3, HIRE per turn max 9.
# The distributions are multimodal (mode 2, peaks at 18-20), so Poisson and
# regression do not fit -- a categorical output is needed.

# The teacher is chosen by the flag `--teacher-sub`, not by editing code: otherwise two
# corpora cannot be kept side by side and compared. Measured from leaderboard metadata
# (internal rank-submissions script): 56263693 gives a median of $96,577 and 66% wins, while
# 56216119 gives $107,010 and 73% over 612 episodes against 166 opponents.
# There may be SEVERAL teachers: `FARMCORE_TEACHER=56263693,56216119`. The corpus
# is then the union of their replays, and the cache fingerprint includes the whole set, so
# corpora of different composition cannot be confused. ⚠ The literature is against pooling (one
# teacher is better, internal imitation-lessons note), and we have our own negative
# measurement on switching teachers, so the corpus composition is a hypothesis measured
# in money, not a default.
TEACHER_SUBS = tuple(int(x) for x in
                     os.environ.get("FARMCORE_TEACHER", "56263693").split(",")
                     if x.strip())
TEACHER_SUB = TEACHER_SUBS[0]                 # for tools that take a single one
# Teacher style (28.09, conditional BC): the net gets the index of the row's "author" and does not
# average the incompatible styles of different teachers. Submissions of one author are one
# style: FARMCORE_STYLE_SAME="56561966:56561951,56548499:56548059". The index
# len(STYLE_SUBS) is the "common" style (the net learns it with probability --style-drop).
_STYLE_SAME = dict((int(a), int(b)) for a, b in (
    p.split(":") for p in os.environ.get("FARMCORE_STYLE_SAME", "").split(",") if p.strip()))
STYLE_SUBS = tuple(dict.fromkeys(_STYLE_SAME.get(s, s) for s in TEACHER_SUBS))


def style_of(sub):
    sub = _STYLE_SAME.get(sub, sub)
    return STYLE_SUBS.index(sub) if sub in STYLE_SUBS else len(STYLE_SUBS)
REPLAY_DIRS = [os.path.join(ROOT, f"raw-content/replays/sub_{s}")
               for s in TEACHER_SUBS]
REPLAY_DIR = REPLAY_DIRS[0]
CACHE_DIR = os.path.join(ROOT, "raw-content/cache")

# Our own submissions. Learning from our own replays is forbidden: it locks the model onto
# its own mistakes, and any measurement "got better" stops meaning anything.
# v1 took replays by the glob `raw-content/replays/*/*.json` -- after episodes of our submission
# arrived in that same folder, such a glob would have silently started mixing them in.
# ALL our submissions, not just the first. While the teacher was hardcoded as a constant, this
# did not matter; now it is chosen via FARMCORE_TEACHER, and the protection must
# know all of them -- otherwise a typo in one digit is enough to silently start
# learning from our own mistakes.
OWN_SUBS = {
    56335130,   # heuristic prototype
    56356138,   # BC v4
    56361984,   # BC v6
    56377353,   # v11
    56384159,   # v13, coordination block
    56390142,   # v18
}


def check_not_own(replay_dir):
    sub = os.path.basename(replay_dir.rstrip("/")).replace("sub_", "")
    if sub.isdigit() and int(sub) in OWN_SUBS:
        raise SystemExit(f"refusal: {replay_dir} is our own submission, "
                         f"we do not learn from our own replays")
    return replay_dir


def _git_stamp():
    """The commit we computed with. `feature_version` covers only seven labeling
    files and changes on a comment edit; the code version is covered by nothing, and
    once this already cost a day: `n1_base` was trained with code from BEFORE A7 (`envcfg`),
    and we learned of it from a diverged cache fingerprint, not from the checkpoint."""
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=5)
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, timeout=5)
        if h.returncode:
            return _sync_stamp()
        return h.stdout.strip() + ("+dirty" if d.stdout.strip() else "")
    except Exception:                            # git may be absent altogether
        return _sync_stamp()


def _sync_stamp():
    """Code version where git is unavailable. On the box there is no `.git` (`cusync` does not
    carry it), so cusync puts a `.sync_stamp` with the local commit alongside."""
    try:
        with open(os.path.join(ROOT, ".sync_stamp")) as f:
            v = f.read().strip()
        return (v + " (synced)") if v else "no git"
    except OSError:
        return "no git"


_END_OF_LABELING = "# ===== END " + "OF LABELING"   # glued: the literal will not find itself


def _labeling_src():
    src = open(os.path.join(HERE, "bc_train.py"), "rb").read()
    cut = src.find(_END_OF_LABELING.encode())
    if cut < 0:
        raise SystemExit("bc_train.py has no end-of-labeling marker")
    return src[:cut]


def _rust_feature_src() -> bytes:
    """Sources of the native encoder -- into the feature fingerprint.

    Absent (the port was not fetched) -- we return empty: then the fingerprint is the same
    as it was before the native path, and caches built by Python remain valid.
    This is the right default answer: no Rust -- no second encoder either.
    """
    out = b""
    for rel in ("engines/kaggriculture_rs/src/encode.rs",
                "engines/kaggriculture_rs/src/model.rs"):
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            out += open(path, "rb").read()
    return out


def feature_version():
    """Fingerprint of the feature and labeling contract. If it changes, the cache invalidates itself."""
    import hashlib
    from model import (TILE_FEAT_DIM, UNIT_FEAT_DIM, PRODUCT_FEAT_DIM,
                       GLOBAL_FEAT_DIM, CROP_FEAT_DIM, ANIMAL_FEAT_DIM)
    src = open(os.path.join(HERE, "features.py"), "rb").read()
    # we hash bc_train.py itself too: all labeling (raw_name, label_of, quantity
    # classes, target, mask) lives here, and without this edits to the labeling would silently
    # not take effect -- the cache would be read as is
    # masks.py and counts.py too: the legality mask goes into the cache as the tensor `legal`
    # (further down the file), and the quantity caps set the shape of the labels. Without them an edit to the
    # mask would NOT invalidate the cache -- the model would learn by the old mask, play
    # by the new one, and the divergence would be invisible.
    lbl = (open(os.path.join(HERE, "effects.py"), "rb").read()
           # bc_train.py -- only LABELING, up to the end-of-labeling marker: below it are
           # cache reading, the loss and the training loop, and editing them does not change the cache.
           # Previously the whole file was hashed, and any loss ablation silently
           # rebuilt the cache -- the "one cache, many trainings" scheme did not work.
           + _labeling_src()
           + open(os.path.join(HERE, "masks.py"), "rb").read()
           + open(os.path.join(HERE, "counts.py"), "rb").read()
           # envcfg sets shed capacity, day length and the number of season days,
           # which go straight into the features and the mask
           + open(os.path.join(HERE, "envcfg.py"), "rb").read()
           # own_flows/town_step feed FlowTracker, and its `extra()` goes into
           # features.encode and lands in the npz. Without this an edit to the flow identity
           # would not invalidate the cache -- exactly the disease for which masks.py
           # was added here.
           + open(os.path.join(HERE, "extract_features.py"),
                  "rb").read()
           # ⚠ Rust sources of the native encoder. It computes THE SAME features
           # as `features.py`, and when the loop goes via the native path, their result
           # lands in the cache. Without these files in the fingerprint an edit in Rust
           # would silently reuse the old cache -- i.e. exactly
           # `remainingOverageTime` and `legal`/`legal_now` once again, only worse:
           # there training and play diverged, and here two encoders
           # will diverge, and there will be nothing to see it with.
           #
           # `encode.rs` -- the features themselves; `model.rs` -- tables of constants, the price curve
           # and structures it reads from. The engine as a whole (`engine.rs`,
           # `actions.rs`) does NOT go here: it defines the state, not the features,
           # and its edits invalidate the cache via the replay itself.
           + _rust_feature_src())
    key = (hashlib.md5(src + lbl).hexdigest()[:8]
           + f"-{TILE_FEAT_DIM}.{UNIT_FEAT_DIM}.{PRODUCT_FEAT_DIM}.{GLOBAL_FEAT_DIM}"
           + f".{CROP_FEAT_DIM}.{ANIMAL_FEAT_DIM}"
           + f"-{N_OPS}-{len(UNIT_OPS)}"
           + f"-caps{sorted(CNT.ALL_CAPS.items())}"
           + f"-teacher{'.'.join(str(x) for x in sorted(TEACHER_SUBS))}"
           + f"-styles{'.'.join(str(x) for x in STYLE_SUBS)}"
           + "-labels6")   # effects, economy, PICKUP quantity; without swaps and side
    return hashlib.md5(key.encode()).hexdigest()[:10]
COUNT_HEADS = tuple(CNT.CAPS)          # sell, buy, seed, animal, hire
MKT_HEADS = dict(CNT.CAPS); MKT_HEADS["land"] = 1


def raw_name(a):
    """Action name as recorded in the replay, including directions."""
    if not isinstance(a, list) or not a:
        return "PASS"
    op = a[0]
    if op in MOVE_NAMES:
        return op
    if op == "PLANT" and len(a) > 1 and a[1] in CROPS:
        return f"PLANT_{a[1]}"
    if op == "PICKUP":
        item = a[1] if len(a) > 1 else ""
        return f"PICKUP_{item}" if f"PICKUP_{item}" in OP_ID else "PASS"
    if op == "PLACE":
        item = a[1] if len(a) > 1 else ""
        # An animal is placement on a building, a good is an addressed drop into the shed
        # (engine: branch `_is_shed_adjacent`). Both are now classes of their own.
        return f"PLACE_{item}" if f"PLACE_{item}" in OP_ID else "DROP"
    return op if op in OP_ID else "PASS"


def label_of(a):
    """Work class; for step -1 (a step is no longer a class)."""
    n = raw_name(a)
    return -1 if n in MOVE_NAMES else OP_ID.get(n, 0)


def _qty(act, kind, item):
    n = 0
    for o in (act.get("market") or []):
        if isinstance(o, list) and o and o[0] == kind:
            if kind in ("HIRE", "BUY_LAND"):
                n += 1
            elif len(o) >= 3 and o[1] == item:
                try:
                    n += int(o[2])
                except (TypeError, ValueError):
                    pass
    return n


SIDE_SUB = {}                                  # (episode id, side) -> teacher's submission


def teacher_index():
    """episode id -> TUPLE of sides that one of the teachers played in it.

    There are two sides when two of our teachers met each other: then we learn from
    both, and take the episode itself once (the file lies in the directory of each of them).
    """
    out = {}
    for sub, d in zip(TEACHER_SUBS, REPLAY_DIRS):
        path = os.path.join(d, "episodes.json")
        if not os.path.exists(path):
            continue
        for e in json.load(open(path)):
            for a in e["agents"]:
                if a["submissionId"] == sub:
                    eid = str(e["id"])
                    side = a["index"] or 0
                    SIDE_SUB[(eid, side)] = sub
                    if side not in out.get(eid, ()):
                        out[eid] = out.get(eid, ()) + (side,)
    return out


TEACHER_IDX = teacher_index()


def replay_files():
    """All replays of all teachers, one per episode.

    Dedup by file name: a meeting of two of our teachers is contained by both directories
    as one and the same episode, and without dedup it would enter the corpus twice.
    """
    seen, out = set(), []
    for d in REPLAY_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            b = os.path.basename(f)
            if b == "episodes.json" or b in seen:
                continue
            seen.add(b)
            out.append(f)
    return out


class FlowTracker:
    """External pressure and accumulated sales -- what the net did not see in v1.

    Flow identity: change of market stock = opponent supply + our supply
    - town consumption + purchases. Our flows are known exactly (own private),
    the town is computed exactly (`town_step`), the remainder is the opponent.
    """

    def __init__(self):
        self.mysold = defaultdict(float)
        self.oppsold = defaultdict(float)
        self.pi_ema = defaultdict(float)
        self.day_me = defaultdict(float)
        self.day_op = defaultdict(float)
        self.last_me = defaultdict(float)
        self.last_op = defaultdict(float)
        self.day = 0
        self.crew_today = 0      # hands today (maximum over the day)
        self.crew_prev = 0       # hands yesterday -- for the cash-reserve feature (27.09)
        # --- accumulators ONLY for the critic (R1) ---
        # internal architecture audit, section 8: three quantities cannot be recovered from hidden
        # state on the opponent -- an estimate of its shed, its purchases and the pace of its money. All three
        # are finite (a sufficient statistic, not recurrence), and all three
        # are computed here. They do NOT go to the policy: the gain for actions is zero
        # (section 9), and with a batch of value features the policy had already lost $16,042.
        # The counters are SYMMETRIC between the sides, although the critic needs only the opponent's.
        # Our own side is the only way to check the estimate: for us the shed and
        # hands are visible exactly, and an internal estimator check compares the same
        # formula against them. Computing it only for the opponent would mean creating a
        # quantity with nothing to compare against.
        self.harv = {"me": defaultdict(float), "op": defaultdict(float)}
        self.bought = {"me": defaultdict(float), "op": defaultdict(float)}
        self.fed = {"me": 0.0, "op": 0.0}          # wheat fed (FEED)
        self.fert_used = {"me": 0.0, "op": 0.0}    # fertilizer spent (FERTILIZE)
        self.rate_me = 0.0                   # EMA of bank growth per day
        self.rate_op = 0.0
        self.money_me0 = None                # bank at the start of the current day
        self.money_op0 = None

    def extra(self):
        out = {}
        tot = {g: self.last_me[g] + max(0.0, self.last_op[g]) for g in PRODUCTS}
        for g in PRODUCTS:
            out[f"pi_{g}"] = self.pi_ema[g]
            out[f"oppsold_{g}"] = self.oppsold[g]
            out[f"mysold_{g}"] = self.mysold[g]
            out[f"crowd_{g}"] = (self.last_me[g] / tot[g]) if tot[g] > 1e-6 else 0.5
        out["crew_prev"] = self.crew_prev
        return out

    def hold(self, side="op"):
        """Estimate of a side's stock: how much of a good it has in hands AND in the shed.

        Stock identity (internal architecture audit, section 8): harvested on visible tiles plus
        bought minus sold minus spent on the farm. All four
        terms are observable: the opponent's field is fully visible, its sales are
        the remainder of the market identity, purchases are the same remainder with the opposite sign
        (previously it was cut by `max(0, op)` and lost), feeding and fertilizing are
        the transitions of `fed_today` and `fertilized_until_day`.

        ⚠ Fertilizer is subtracted ONLY for the opponent, and this is not a slip. Our
        "sales" (`mysold`) are not an order but the REMAINDER over own stocks
        (`own_flows`: `t0 + harvested + bought - fed - t1`): feeding is
        subtracted there explicitly, and everything else that vanished -- fertilizer spent on
        FERTILIZE, and shed overflow -- already lies inside the remainder. Subtracting it once
        more means subtracting twice: an internal estimator check showed this as
        a bias of -3 fertilizer units in the median, and at the other edge as
        +125 wheat units if feeding is conversely not subtracted. For the opponent
        `oppsold` is the remainder of the MARKET identity, and neither feeding nor fertilizer
        moves market stock, so both are subtracted there.
        """
        sold = self.oppsold if side == "op" else self.mysold
        out = {}
        for g in PRODUCTS:
            h = self.harv[side][g] + self.bought[side][g] - sold[g]
            if g == "WHEAT":
                h -= self.fed[side]
            if g == "FERTILIZER" and side == "op":
                h -= self.fert_used[side]
            out[g] = max(0.0, h)
        return out

    def critic(self):
        """Accumulators for the value tower. They do NOT go to `features.encode`."""
        return {"opp_hold": self.hold("op"), "opp_bought": dict(self.bought["op"]),
                "opp_harv": sum(self.harv["op"].values()),
                "my_harv": sum(self.harv["me"].values()),
                "opp_fed": self.fed["op"],
                "rate_me": self.rate_me, "rate_op": self.rate_op}

    def step(self, steps, k, p, day):
        """Transition (k-1) -> k from the replay."""
        self.observe(steps[k - 1][p]["observation"], steps[k][p]["observation"],
                     steps[k][p].get("action") or {}, p, k, day)

    def observe(self, o_prev, o_cur, act, p, abs_step, day):
        """The same accounting, but from a pair of live observations and an own action.

        The only place where external pressure is computed: both training and play
        must call it, otherwise features diverge between train and serve.
        """
        m_me = (o_prev.get("farms") or [{}, {}])[p].get("money")
        m_op = (o_prev.get("farms") or [{}, {}])[1 - p].get("money")
        if m_me is not None and self.money_me0 is None:
            self.money_me0, self.money_op0 = float(m_me), float(m_op)
        # Crew size by day: at night the engine disbands hands, in the morning they are hired
        # anew by Fibonacci -- "how much tomorrow morning costs" is computed from yesterday.
        if day != self.day:
            self.crew_prev, self.crew_today = self.crew_today, 0
        self.crew_today = max(self.crew_today, len(((o_prev.get("farms") or [{}, {}])[p]).get("hands") or []))
        if day != self.day:
            for g in PRODUCTS:
                self.pi_ema[g] = 0.6 * self.pi_ema[g] + 0.4 * self.day_op[g]
                self.last_me[g], self.last_op[g] = self.day_me[g], self.day_op[g]
            self.day_me.clear()
            self.day_op.clear()
            self.day = day
            # Money pace per day -- the same EMA(0.6/0.4) as for market pressure.
            # Computed at the day change, not per turn: within a day the bank jumps
            # from a single sale, and an instantaneous derivative would measure noise.
            if m_me is not None and self.money_me0 is not None:
                self.rate_me = 0.6 * self.rate_me + 0.4 * (float(m_me) - self.money_me0)
                self.rate_op = 0.6 * self.rate_op + 0.4 * (float(m_op) - self.money_op0)
                self.money_me0, self.money_op0 = float(m_me), float(m_op)
        if "market" not in o_prev or "market" not in o_cur:
            return
        fake = [{p: {"observation": o_prev}},
                {p: {"observation": o_cur, "action": act}}]
        try:
            ours, bought = own_flows(fake, 1, p)
        except Exception:
            return
        shops_k = list(o_prev["town"].get("unlocked_shops", []))
        inv_a, inv_b = o_prev["market"]["inventory"], o_cur["market"]["inventory"]
        for g in PRODUCTS:
            me = max(0.0, ours[g])
            op = (inv_b[g] - inv_a[g]) + town_step(g, shops_k, abs_step - 1) - me + bought[g]
            self.day_me[g] += me
            self.day_op[g] += op
            self.mysold[g] += me
            self.oppsold[g] += max(0.0, op)
            # A negative remainder is a PURCHASE by the opponent, not zero. Previously
            # it was cut by `max(0, op)` and lost: the internal architecture audit lists
            # it among three unrecovered quantities.
            self.bought["op"][g] += max(0.0, -op)
            self.bought["me"][g] += float(bought.get(g, 0.0))
        self._harvest(o_prev, o_cur, p, abs_step - 1)

    def _harvest(self, o_prev, o_cur, p, st):
        """How much of a good both sides removed from VISIBLE tiles in this turn.

        The opponent's field is fully visible, so its harvest is observable: `HARVEST`
        zeroes the tile's `yield_units` and puts them into the unit's hands
        (`kaggriculture_env_ref.py:446-472`). This is where the estimate of its
        stock comes from -- the only one of the three hidden quantities that cannot be obtained
        from the market identity.

        Three caveats make this an ESTIMATE rather than an exact count:
          * `_decay_plants` removes one unit every two turns after
            `max_lifespan_step` -- this is subtracted here too, by the same condition
            as in the engine;
          * `DIG` removes the plant together with the yield and looks like a harvest. For
            the teacher this is 0.3% of actions, for us less;
          * at night the engine both adds yield and kills unwatered plants, and
            disbands hands, so the transition across the day is skipped entirely.
        The error goes both ways -- units of goods per game against hundreds harvested.
        """
        day_a = o_prev.get("day", 0)
        if o_cur.get("day", day_a) != day_a:      # night: harvest and death are indistinguishable
            return
        for side, who in ((1 - p, "op"), (p, "me")):
            fa = (o_prev.get("farms") or [None, None])[side]
            fb = (o_cur.get("farms") or [None, None])[side]
            if not fa or not fb:
                continue
            ta, tb = fa["tiles"], fb["tiles"]
            for y in range(len(ta)):
                for x in range(len(ta[y])):
                    a = ta[y][x]
                    if not isinstance(a, dict):
                        continue
                    b = tb[y][x] if y < len(tb) and x < len(tb[y]) else None
                    b = b if isinstance(b, dict) else {}
                    ya = a.get("yield_units", 0) or 0
                    if a.get("kind") == "PLANT":
                        yb = b.get("yield_units", 0) if b.get("kind") == "PLANT" else 0
                        mls = a.get("max_lifespan_step", -1)
                        decay = int(mls is not None and mls >= 0 and st >= mls
                                    and (st - mls) % 2 == 0)
                        took = max(0.0, ya - (yb or 0) - decay)
                        item = a.get("crop")           # for a plant the good = the crop
                        if (b.get("kind") == "PLANT"   # fertilized -> spent a unit
                                and b.get("fertilized_until_day", -1)
                                > a.get("fertilized_until_day", -1)):
                            self.fert_used[who] += 1.0
                    elif a.get("animal"):
                        yb = b.get("yield_units", 0) if b.get("animal") else 0
                        took = max(0.0, ya - (yb or 0))
                        item = ANIMALS[a["animal"]]["product"]
                        if not a.get("fed_today") and b.get("fed_today"):
                            self.fed[who] += 1.0
                        if (a.get("fertilizer_available") and b.get("animal")
                                and not b.get("fertilizer_available")):
                            self.harv[who]["FERTILIZER"] += 1.0
                    else:
                        continue
                    if took <= 0 or item not in PRODUCTS:
                        continue
                    self.harv[who][item] += took


def split_files(files, val_frac=0.1, mod=1000):
    """Fold by episode id remainder, not by a sample from a list.

    A `random.sample(files, k)` sample shifts entirely when it is worth fetching new
    replays -- and validation numbers stop being comparable between runs.
    The id remainder is not tied to the file list: a game that landed in val stays
    in val forever.
    """
    thr = int(mod * val_frac)
    val = [f for f in files if int(os.path.basename(f)[:-5]) % mod < thr]
    train = [f for f in files if f not in set(val)]
    return train, sorted(val)


def _sides(path, teacher_only):
    eid = os.path.basename(path)[:-5]
    if not teacher_only:
        return eid, (0, 1)
    return eid, TEACHER_IDX.get(eid, (0,))


# `mdiff` is NOT included here: the final bank difference reaches $130,000, while the
# float16 ceiling is 65,504. In the cache it is stored as float32.
_FP16 = ("tiles_me", "tiles_op", "units", "products", "glob", "effect", "econ",
         "vcrit")


def cache_path(path, ver):
    return os.path.join(CACHE_DIR, ver, os.path.basename(path)[:-5] + ".npz")


def encode_replay_cached(path, teacher_only=True, gamma=0.999, ver=None, write=True):
    """Parsing with a cache: json is parsed once for the whole feature set.

    The cache is `raw-content/cache/<fingerprint>/<episode>.npz` in float16 (collate casts to
    float32 anyway). The fingerprint is computed by `feature_version()` from the source of
    `features.py` and the dimensions: edit the features -- the cache invalidates itself.
    A trick from pokemoni (`train/replay_dataset.py` caches parsed `Sample`s),
    only our unit is a game, not a side.
    """
    cp = cache_path(path, ver) if ver else None
    if cp and os.path.exists(cp):
        try:
            z = np.load(cp)
            return {k: (z[k].astype(np.float32) if k in _FP16 else z[k]) for k in z.files}
        except Exception:
            pass
    d = encode_replay(path, teacher_only, gamma)
    if d is not None and cp and write:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        # np.savez appends .npz to a name without this extension, so the temporary
        # file must carry it: otherwise os.replace looks for a nonexistent path, fails
        # in the worker, and the game is silently lost together with all its samples.
        tmp = f"{cp}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **{k: (v.astype(np.float16) if k in _FP16 else v)
                         for k, v in d.items()})
        os.replace(tmp, cp)
    return d


def shed_at_market(shed, acts, invs, pos, tiles, board=BOARD):
    """Shed at the MOMENT of order execution -- the single canonical point.

    The engine processes the actions of ALL units and only then the market
    (`kaggriculture_env_ref.py:935-941`), so by the time of a sale the shed has already been changed
    by this very turn. Three operations must be accounted for, not one:

      DROP    -- the whole stack into the shed, the excess is destroyed;
      PLACE   -- addressed drop of N units of a good (an animal goes to a suitable
                building bypassing the shed, otherwise it falls into the shed);
      PICKUP  -- the good LEAVES the shed BEFORE the market.

    Previously labeling counted only `DROP`, while the decoder in `play.py` counted `DROP` and
    `PICKUP`. The divergence: 13.2% of turns, median 4 units, maximum 16.
    Exactly the same class as `legal` versus `legal_now`: training and play compute a
    DIFFERENT quantity, and the engine is silent about it. Now both sides call this
    function, and they have nowhere left to diverge.

    The unit order is respected: the engine applies them in turn, and the room in the
    shed for the second depends on what the first did.
    """
    half = board // 2
    adj = {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
    sp = dict(shed)
    inv_l = [dict(v or {}) for v in (invs or [])]
    for k, a in enumerate(acts or []):
        if not (isinstance(a, list) and a) or k >= len(pos):
            continue
        op = a[0]
        if op not in ("DROP", "PICKUP", "PLACE"):
            continue
        x, y = pos[k]
        near = (x, y) in adj
        iv = inv_l[k] if k < len(inv_l) else {}
        if op == "DROP":
            if not near:
                continue
            for item, n in list(iv.items()):
                if n > 0:
                    room = max(0, CNT.SHED_CAP - sum(sp.values()))
                    take = min(int(n), room)
                    if take > 0:
                        sp[item] = sp.get(item, 0) + take
                del iv[item]
        elif op == "PICKUP":
            if not near or len(a) < 2:
                continue
            item = a[1]
            n = int(a[2]) if len(a) >= 3 else 1
            n = min(n, sp.get(item, 0))
            if n > 0:
                sp[item] -= n
                iv[item] = iv.get(item, 0) + n
        else:                                   # PLACE
            if len(a) < 2:
                continue
            item = a[1]
            tile = tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else None
            if item in ANIMALS and structure_free(tile, item):
                if iv.get(item, 0) > 0:         # went to a building, bypassing the shed
                    iv[item] -= 1
                continue
            if not near:
                continue
            n = int(a[2]) if len(a) >= 3 else 1
            n = min(n, iv.get(item, 0))
            room = max(0, CNT.SHED_CAP - sum(sp.values()))
            n = min(n, room)
            if n > 0:
                sp[item] = sp.get(item, 0) + n
                iv[item] -= n
    return sp


def executed_market(rep):
    """Market orders EXECUTED by the engine: step -> (Counter of player 0, Counter of player 1).

    28.09, teacher 56521745: he orders with a margin ("sell 13" with 5 in the
    shed), and the engine executes only ~30% of what was ordered, the rest it silently
    discards. The decoder, though, chooses the quantity under a cap by the real shed and
    money, so the "ordered" label is noise that the net cannot
    repeat (sell-gate recall 0.43). The game is replayed by the engine from the
    recorded turns of both sides, and the executed units are counted. A game
    that did not reproduce bitwise in money is a loud refusal.
    """
    import collections as _c
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from kaggle_environments import make
    steps = rep["steps"]
    cfg = dict(rep["configuration"]); cfg["seed"] = rep["info"]["seed"]
    cfg["episodeSteps"] = len(steps)
    cur = {"farms": None, "c": None}
    cu0, pm0, dh0, dl0 = K._commit_unit, K._process_market, K._do_hire, K._do_buy_land

    def who(farm):
        return 0 if farm is cur["farms"][0] else 1

    def cu(op, item, price, farm, private, market, *a, **k):
        ok = cu0(op, item, price, farm, private, market, *a, **k)
        if ok:
            cur["c"][who(farm)][(op, item)] += 1
        return ok

    def dh(farm, *a, **k):
        n = farm["hires_today"]
        dh0(farm, *a, **k)
        if farm["hires_today"] > n:
            cur["c"][who(farm)][("HIRE", None)] += 1

    def dl(farm, *a, **k):
        n = len(farm["unlocked_quadrants"])
        dl0(farm, *a, **k)
        if len(farm["unlocked_quadrants"]) > n:
            cur["c"][who(farm)][("BUY_LAND", None)] += 1

    def pm(state, env):
        cur["farms"] = state[0].observation.farms
        return pm0(state, env)

    K._commit_unit, K._process_market, K._do_hire, K._do_buy_land = cu, pm, dh, dl
    try:
        env = make("kaggriculture", configuration=cfg, debug=False)
        env.reset()
        out = [None]
        for t in range(1, len(steps)):
            cur["c"] = (_c.Counter(), _c.Counter())
            env.step([steps[t][q].get("action") for q in (0, 1)])
            out.append(cur["c"])
            for q in (0, 1):
                want = (steps[t][q].get("observation") or {}).get("farms")
                if want and env.state[q].observation["farms"][q]["money"] != want[q]["money"]:
                    raise RuntimeError(f"replay did not reproduce at step {t}")
    finally:
        K._commit_unit, K._process_market, K._do_hire, K._do_buy_land = cu0, pm0, dh0, dl0
    return out


def _exec_orders(c):
    """Executed -> canonical order list (`_qty` counts from it)."""
    out = []
    for (op, item), n in c.items():
        if op in ("HIRE", "BUY_LAND"):
            out += [[op]] * n
        elif n > 0:
            out.append([op, item, n])
    return out


def encode_replay(path, teacher_only=True, gamma=0.999):
    """One replay -> dense arrays.

    Labeling in two passes. The first collects positions and raw action names, the second
    assigns labels: if a unit is walking, its label becomes the JOB at the end of the walk
    segment and the TILE where it will happen. The direction does not take part in labels --
    it is computed by a rule (internal movement-factorization note). The search for the end of
    the segment is limited to a day: at night the engine disbands hands (`_end_of_day`).
    """
    eid, sides = _sides(path, teacher_only)
    try:
        rep = json.load(open(path))
    except Exception:
        return None
    steps = rep["steps"]
    rewards = rep.get("rewards") or [0, 0]
    # The market in the record is replaced with the executed one ONCE and for all readers:
    # both the labels (`_qty`) and the flow accumulator (`own_flows` takes purchases from
    # the orders) -- in play both see our orders, and they are executable by construction.
    exq = executed_market(rep)
    for t in range(1, len(steps)):
        for q in (0, 1):
            a = steps[t][q].get("action")
            if isinstance(a, dict):
                a["market"] = _exec_orders(exq[t][q])
    rows = []
    for p in sides:
        r_me, r_op = (rewards[p] or 0), (rewards[1 - p] or 0)
        vcls = 0 if r_me > r_op else (1 if r_me == r_op else 2)      # win/draw/loss
        # Final bank difference in DOLLARS and the number of decisions to the end of the game.
        # They go into the cache raw, not scaled: the scale `σ` and the discount
        # `γ` are training flags (value_head.RL_SIGMA), and tuning them by rebuilding the
        # cache would mean paying an hour of compute for each trial. Exactly the same
        # argument for which `win_w` is not baked in here.
        mdiff = float(r_me) - float(r_op)
        # The outcome weight is NOT baked into the cache: it is set in training by the flag
        # `--win-w` and computed from `value` (the same outcome class). Otherwise tuning the
        # weight would cost a cache rebuild, and the constant itself would live in two places.
        wgt = 1.0
        flow = FlowTracker()

        recs = []
        for i in range(1, len(steps)):
            obs = steps[i - 1][p]["observation"]
            if "farms" not in obs or "market" not in obs:
                continue
            act = steps[i][p].get("action") or {}
            farm = obs["farms"][p]
            pos = [tuple(farm["farmer"])] + [tuple(q) for q in farm["hands"]]
            names = [raw_name(u) for u in
                     [act.get("farmer")] + list(act.get("hands") or [])]
            recs.append((i, obs, act, pos, names))

        per_turn = [EFF.turn_effects(r[3], [r[2].get("farmer")]
                                     + list(r[2].get("hands") or [])) for r in recs]
        days = [r[1].get("day", 0) for r in recs]
        for idx, (i, obs, act, pos, names) in enumerate(recs):
            # The farm of THIS turn. Before, `farm` from the first pass was read below --
            # i.e. the last observation of the game: the shed projection (`sell_have`)
            # checked a free building for the animal PLACE against the end-of-season board.
            # Found by checking labeling against `run_turn` (an internal label-parity script).
            farm = obs["farms"][p]
            enc = features.encode(obs, p, extra=dict(
                flow.extra(), prev_act=(steps[i - 1][p].get("action") if i >= 2 else None)))
            # The critic columns are taken WITH THE SAME accumulator and BEFORE its step --
            # otherwise the tower would see the flow state after its own turn,
            # i.e. the future. One line lower, and this would have been a leak of the class
            # of `remainingOverageTime`.
            vcr = features.value_extra(obs, p, flow.critic())
            flow.step(steps, i, p, obs.get("day", 0))
            shed = (obs.get("private") or {}).get("shed") or {}
            day = obs.get("day", 0)
            n_live = min(int(enc["unit_mask"].sum()), MAX_UNITS)
            ops = np.full(MAX_UNITS, -100, dtype=np.int64)
            tgt = np.full(MAX_UNITS, -100, dtype=np.int64)
            # mask of allowed jobs: without it 30.6% of the op head's argmax land
            # on an inadmissible action, and at inference the next class is taken,
            # which the net was never trained to rank
            lm = np.zeros((MAX_UNITS, N_OPS), dtype=np.uint8)
            # The second mask is by the unit's CURRENT cell. `lm` is taken at the DESTINATION
            # tile, and this is right for the loss (the label is the job on arrival),
            # but feeding it to the net is NOT allowed: the destination is chosen by the teacher, so
            # the mask on it is a function of a future decision, not of the observable
            # state. Exactly a leak of the `remainingOverageTime` kind. Verified
            # in money: a model with `lm` on the input gives the BEST validation of all time
            # (target 0.896, whole turn 0.228) and $891 in play.
            lmn = np.zeros((MAX_UNITS, N_OPS), dtype=np.uint8)
            # how many to take from the shed: before, the number was hardwired in play.py
            # ({WHEAT: 10, FERTILIZER: 5, the rest 1/5}) and matched neither the
            # teacher nor anything in the engine. Now it is a net output.
            qty = np.full(MAX_UNITS, -100, dtype=np.int64)
            mask_pos = None
            seeds_i = (obs.get("private") or {}).get("seeds") or {}
            invs_i = (obs.get("private") or {}).get("inventories") or [{}]
            raw_acts = [act.get("farmer")] + list(act.get("hands") or [])
            # Seeds AFTER the plantings of units 0..k-1 -- as the op mask sees them in play
            # (`policy.run_turn` deducts a seed at every PLANT). The shed is sequential in the same way,
            # via `shed_at_market` over actions 0..k-1.
            seeds_seq = dict(seeds_i)
            for k in range(n_live):
                nm = names[k] if k < len(names) else "PASS"
                inv_k = invs_i[k] if k < len(invs_i) else {}
                raw = raw_acts[k] if k < len(raw_acts) else None
                # The input `legal_now` is for the CURRENT cell and as of the START of the
                # turn, for EVERY live unit: exactly `legal0` from `play.agent`.
                # Previously it was filled below `continue` and for units without an observed
                # target stayed zero (B2, internal policy-parity note: 0.46% of units).
                ok_now = legal_ops(obs, p, k, pos[k], inv_k, seeds_i, shed)
                for j, nm_op in enumerate(UNIT_OPS):
                    if nm_op in ok_now:
                        lmn[k, j] = 1
                if nm in MOVE_NAMES:
                    fin = None
                    j = idx + 1
                    while j < len(recs) and recs[j][1].get("day", 0) == day:
                        if k >= len(recs[j][4]):
                            break
                        if recs[j][4][k] not in MOVE_NAMES:
                            fin = j
                            break
                        j += 1
                    if fin is None:          # the target was not observed until the end of the day
                        continue
                    tx, ty = recs[fin][3][k]
                    tgt[k] = ty * BOARD + tx
                    # B4: the op of a walking unit is NOT labeled. The game asks for op
                    # only of a unit whose target is its own cell or "stay"
                    # (`policy.run_turn`), and masks it by the CURRENT cell.
                    # The label "job at the destination" taught the marginal
                    # P(op | s) over all destinations, while the conditional is played:
                    # P(op | s, I work here). The job on arrival is labeled
                    # by its own turn -- there the unit is already in place.
                    continue
                elif nm == "PASS":
                    ops[k] = OP_ID["PASS"]
                    tgt[k] = STAY
                    mask_pos = pos[k]
                else:
                    ops[k] = OP_ID.get(nm, 0)
                    x, y = pos[k]
                    tgt[k] = y * BOARD + x
                    mask_pos = pos[k]
                # Only units working HERE (PASS or a job
                # on their own cell) get here: exactly those of whom the game asks for op.
                # Quantity -- in the same place where the decoder asks for it: PICKUP,
                # addressed PLACE of a good and PLACE of an animal that goes to the
                # shed rather than to a free building (B5: the latter had
                # no label, although the decoder asks the `pickup` head).
                if (isinstance(raw, list) and raw
                        and (raw[0] == "PICKUP"
                             or (raw[0] == "PLACE" and len(raw) > 1
                                 and (raw[1] in PRODUCTS
                                      or (raw[1] in ANIMALS
                                          and not structure_free(
                                              farm["tiles"][pos[k][1]][pos[k][0]],
                                              raw[1])))))):
                    qty[k] = min(int(raw[2]) if len(raw) >= 3 else 1,
                                 CNT.ALL_CAPS["pickup"])
                # The op mask is SEQUENTIAL, as in play (B6): on its own
                # cell, seeds and shed after the jobs of units 0..k-1 of this turn
                # (`shed_at_market` -- the same function as in `run_turn`).
                shed_k = shed_at_market(shed, raw_acts[:k], invs_i, pos, farm["tiles"])
                ok = legal_ops(obs, p, k, mask_pos, inv_k, seeds_seq, shed_k)
                for j, nm_op in enumerate(UNIT_OPS):
                    if nm_op in ok:
                        lm[k, j] = 1
                if (isinstance(raw, list) and raw and raw[0] == "PLANT"
                        and len(raw) > 1 and seeds_seq.get(raw[1], 0) > 0):
                    seeds_seq[raw[1]] -= 1
                # A label outside the mask (0.07% of cases: the state managed to change by
                # the arrival, the engine rejected such an action) is not a decision but noise.
                # It must not be left: a class outside the mask gets logit -1e9, and CE
                # on it equals a billion -- one such sample breaks the epoch.
                if ops[k] >= 0 and not lm[k, ops[k]]:
                    ops[k] = -100
            # raw quantities: gates and the scale are built from them in the loss (counts.py)
            sell = np.array([_qty(act, "SELL", g) for g in PRODUCTS], dtype=np.int64)
            # The "sell all" flag is checked against the shed at the MOMENT OF MARKET EXECUTION:
            # orders go after unit actions, so DROP, PLACE and PICKUP
            # of this same turn are already visible to the sale. This is computed by `shed_at_market` --
            # ONE function for labeling and for the decoder (play.py calls the same one).
            shed_p = shed_at_market(shed, raw_acts, invs_i, pos, farm["tiles"])
            sell_have = np.array([shed_p.get(g, 0) for g in PRODUCTS], dtype=np.int64)
            buy = np.array([_qty(act, "BUY_PRODUCT", g) for g in PRODUCTS],
                           dtype=np.int64)
            # Round-trip trades (28.09): teacher 56521745 in one turn sells and
            # buys ONE good (88% of his buys). The engine quotes a buy at the stock
            # after the trade, so the round trip is net zero; a one-turn probe
            # (an internal market probe, variant net) confirms this. We learn the net.
            _rt = np.minimum(sell, buy)
            sell, buy = sell - _rt, buy - _rt
            eff = EFF.day_effect_map(per_turn, idx, days)
            oa = dict(obs); oa["player"] = p
            nxt = steps[i][p].get("observation") or {}
            if "farms" in nxt:
                ob_ = dict(nxt); ob_["player"] = p
                econ = EFF.econ_vector(oa, ob_)
            else:
                econ = np.zeros(EFF.ECON_DIM, dtype=np.float32)
            rows.append((enc, ops, sell, buy,
                         _qty(act, "HIRE", None),
                         1 if _qty(act, "BUY_LAND", None) else 0,
                         np.array([_qty(act, "BUY_SEED", c) for c in CROP_LIST],
                                  dtype=np.int64),
                         np.array([_qty(act, "BUY_ANIMAL", x) for x in ANIMAL_LIST],
                                  dtype=np.int64),
                         vcls, wgt, tgt, eff, econ, lm, sell_have, qty,
                         lmn, vcr, mdiff, len(recs) - 1 - idx,
                         style_of(SIDE_SUB.get((eid, p), 0))))
    del rep
    if not rows:
        return None
    keys = ("tiles_me", "tiles_op", "tile_mask", "units", "unit_mask", "products",
            "glob", "crops", "animals")
    d = {k: np.stack([r[0][k] for r in rows]) for k in keys}
    d["ops"] = np.stack([r[1] for r in rows])
    d["sell"] = np.stack([r[2] for r in rows])
    d["buy"] = np.stack([r[3] for r in rows])
    d["hire"] = np.array([r[4] for r in rows], dtype=np.int64)
    d["land"] = np.array([r[5] for r in rows], dtype=np.int64)
    d["seed"] = np.stack([r[6] for r in rows])
    d["animal"] = np.stack([r[7] for r in rows])
    d["value"] = np.array([r[8] for r in rows], dtype=np.int64)     # outcome class
    d["w"] = np.array([r[9] for r in rows], dtype=np.float32)
    d["target"] = np.stack([r[10] for r in rows])
    d["effect"] = np.stack([r[11] for r in rows])
    d["econ"] = np.stack([r[12] for r in rows])
    d["legal"] = np.stack([r[13] for r in rows])
    d["legal_now"] = np.stack([r[16] for r in rows])
    d["sell_have"] = np.stack([r[14] for r in rows])
    d["qty"] = np.stack([r[15] for r in rows])
    d["vcrit"] = np.stack([r[17] for r in rows])                    # critic input
    d["mdiff"] = np.array([r[18] for r in rows], dtype=np.float32)  # $ of outcomes
    d["to_end"] = np.array([r[19] for r in rows], dtype=np.int32)   # decisions to the end
    d["style"] = np.array([r[20] for r in rows], dtype=np.int64)    # teacher style
    return d


def _cache_one(args):
    """Parse a replay into the cache. Only the name is returned outward: the arrays stay
    in the worker, otherwise pickling them eats all the benefit of parallelism."""
    path, teacher_only, gamma, ver = args
    try:
        encode_replay_cached(path, teacher_only, gamma, ver)
        return path
    except Exception as e:                      # games must not be lost silently
        return f"ERROR {os.path.basename(path)}: {e}"


def build_cache(files, workers, teacher_only=True, gamma=0.999, ver=None):
    todo = [f for f in files if not (ver and os.path.exists(cache_path(f, ver)))]
    if not todo:
        return 0
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    bad = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, r in enumerate(ex.map(_cache_one,
                                     [(f, teacher_only, gamma, ver) for f in todo]), 1):
            if isinstance(r, str) and r.startswith("ERROR"):
                print("  " + r, flush=True)
                bad += 1
            if i % 50 == 0:
                print(f"  cache: {i}/{len(todo)} games, {i/(time.time()-t0):.2f} games/s",
                      flush=True)
    print(f"  cache ready: {len(todo)} games in {(time.time()-t0)/60:.1f} min"
          + (f", errors {bad}" if bad else ""), flush=True)
    return len(todo)


# Types for the in-memory corpus: the whole set in float16 is about 4 GB, so it
# simply lives in RAM, and batches are cut by permuting indices. The per-sample
# path gave 410 samples/s against the A100 ceiling of 7,300 -- i.e. 5.6% of the card.
CORPUS_F16 = ("tiles_me", "tiles_op", "units", "products", "glob", "crops",
              "animals", "effect", "econ", "vcrit")
CORPUS_I16 = ("ops", "target", "sell", "buy", "seed", "animal", "hire", "land",
              "value", "sell_have", "qty", "style")
# ===== END OF LABELING =====================================================
# Everything above defines the CONTENT of the cache and is part of its fingerprint
# (`feature_version`). Everything below is cache reading, the loss and training: they can be
# changed without rebuilding the cache. A function on which the npz content depends must NOT be
# moved below this line.

_MISS = object()          # cache miss in `load_corpus`


class _MultiOpt:
    """Several optimizers under one interface (Muon + AdamW, 30.09)."""
    def __init__(self, opts):
        self.opts = opts

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {f"opt{i}": o.state_dict() for i, o in enumerate(self.opts)}


class _MultiSched:
    def __init__(self, scheds):
        self.scheds = scheds

    def step(self):
        for s_ in self.scheds:
            s_.step()

# Fine-tuning on a team from the shared cache (30.09): STYLE_SUBS indices of its submissions;
# rows of other sides are dropped at load. None -- no selection.
_FT_STYLES = None

# Team styles (30.09, teacher blend): index of a submission in STYLE_SUBS -> index of the
# team; the last element is "common". None -- training without a condition.
_STYLE_MAP = None


def _fix_style(path, d, teacher_only):
    """The `style` column by CURRENT metadata. In the cache it is the index of a submission in
    STYLE_SUBS at build time; for npz carried over from an old cache this is a different
    list. The side of a row: one side -- all rows; two -- by `mdiff`."""
    if not teacher_only or "style" not in d:
        return d
    eid, sides = _sides(path, True)
    n_gen = len(STYLE_SUBS)
    st = np.full(len(d["mdiff"]), n_gen, dtype=d["style"].dtype)
    if len(sides) == 1:
        sub = SIDE_SUB.get((eid, sides[0]))
        st[:] = style_of(sub) if sub else n_gen
    else:
        r = _ep_rewards().get(eid) or {}
        md = np.asarray(d["mdiff"], dtype=np.float64)
        for sd in sides:
            sub = SIDE_SUB.get((eid, sd))
            if sub and r.get(0) is not None and r.get(1) is not None:
                st[np.abs(md - (r[sd] - r[1 - sd])) < 1.0] = style_of(sub)
    d["style"] = st
    return d


def _write_cache(path, ver, d):
    """Writing an npz exactly as in `encode_replay_cached` (atomically, via tmp)."""
    cp = cache_path(path, ver)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    tmp = f"{cp}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **{k: (v.astype(np.float16) if k in _FP16 else v)
                     for k, v in d.items()})
    os.replace(tmp, cp)


_EP_REWARDS = None


def _ep_rewards():
    """episode id -> {side: bank result} from the teachers' episodes.json."""
    global _EP_REWARDS
    if _EP_REWARDS is None:
        _EP_REWARDS = {}
        for d in REPLAY_DIRS:
            p = os.path.join(d, "episodes.json")
            if not os.path.exists(p):
                continue
            for e in json.load(open(p)):
                _EP_REWARDS[str(e["id"])] = {(a["index"] or 0): a.get("reward")
                                             for a in e["agents"]}
    return _EP_REWARDS


def _sides_ok(path, d, teacher_only):
    """Whether the sides in the labeling match those that `_sides` expects NOW (F3).

    The cache key has neither the sides mode nor the contents of episodes.json, and the npz itself
    does not record the side. But `mdiff` on each row is the side's result minus the
    opponent's: each labeled side has its own value. The set of values
    must match the one expected from the rewards in episodes.json. A draw and a game without
    a record (own corpus, no metadata) are not checked.
    """
    eid, sides = _sides(path, teacher_only)
    r = _ep_rewards().get(eid)
    if not r or any(r.get(s) is None for s in (0, 1)) or abs(r[0] - r[1]) < 1:
        return True
    want = sorted(float(r[s] - r[1 - s]) for s in sides)
    got = sorted(float(x) for x in np.unique(np.asarray(d["mdiff"], dtype=np.float64)))
    # tolerance: in the cache mdiff is float32, the reward may be fractional
    return len(got) == len(want) and all(abs(g - w) < 1.0 for g, w in zip(got, want))

CORPUS_RAW = ("tile_mask", "unit_mask", "legal", "legal_now", "w")


# ----- unit quantity label by the EXECUTABLE (`--qty-exec`, 29.09, off) ----
# The same disease as in the market before `executed_market` (internal imitation-fidelity note, section 1):
# the `qty` label is the teacher's ORDER (`PICKUP WHEAT 20`), while the engine takes
# min(order, shed stock) (`_apply_unit_action`, the PICKUP branch). The decoder
# closes classes above the stock BEFORE choosing (`policy.run_turn`: cap m = shed),
# i.e. the net learns to put mass on a class that is closed by the mask in play, and
# the choice among the open ones stays untrained. Measurement by an internal qty probe:
# for 56263693 19.6% of PICKUPs ask for more than there is (97 of 652 -- a different class).
# The fix is at cache load, the cache does not change: for rows where the op label is PICKUP
# of a good/animal or PLACE of a good, the quantity is clipped by the stock AT THE START of the turn
# (shed from products[:, 3] and glob[19:22], hands from units[:, U_INV0:]). The jobs
# of lower-numbered units of the same turn are not counted -- an approximation, not an exact recount.
# The quantity label of a unit whose op was thrown out as illegal (ops = -100)
# is dropped: the decoder does not ask it for a quantity.
def _qty_exec(d):
    from model import U_INV0
    q = np.asarray(d["qty"]).astype(np.int64).copy()
    ops = np.asarray(d["ops"])
    cap = float(features.envcfg.SHED_CAP)
    shed_p = np.rint(np.asarray(d["products"], np.float32)[:, :, P_SHED] * cap)
    shed_a = np.rint(np.asarray(d["glob"], np.float32)[:, 19:22] * 5)
    inv = np.rint(np.asarray(d["units"], np.float32)[:, :, U_INV0:U_INV0 + len(PRODUCTS)] * 20)
    keep = np.zeros_like(q, dtype=bool)
    for j, g in enumerate(PRODUCTS):
        m = (q >= 0) & (ops == OP_ID[f"PICKUP_{g}"])
        q = np.where(m, np.minimum(q, shed_p[:, None, j]), q); keep |= m
        m = (q >= 0) & (ops == OP_ID[f"PLACE_{g}"])
        q = np.where(m, np.minimum(q, inv[:, :, j]), q); keep |= m
    for j, a_ in enumerate(ANIMAL_LIST):
        m = (q >= 0) & (ops == OP_ID[f"PICKUP_{a_}"])
        q = np.where(m, np.minimum(q, shed_a[:, None, j]), q); keep |= m
        keep |= (q >= 0) & (ops == OP_ID[f"PLACE_{a_}"])      # into the shed: hands are not visible by name
    q = np.where(keep & (q >= 1), q, -100)
    d = dict(d)
    d["qty"] = q
    return d


def load_corpus(files, ver, teacher_only=True, gamma=0.999, jobs=0, device=None,
                qty_exec=False):
    """The whole cache in memory as one dict of arrays.

    `device` -- load RIGHT THERE (usually GPU memory). Then RAM holds one game at a time,
    not the whole corpus: the previous path first gathered everything in
    RAM and only then transferred, i.e. required DOUBLE the corpus volume in
    main memory. On the box that was tolerable (116 GB), on a rented machine
    it is not: there `/proc/meminfo` shows the host, not the container quota, and
    its number cannot be relied on.

    Read by THREADS: npz is unpacked with zlib, which releases the GIL, so
    threads work here. Measurement on the box: sequential loading of a corpus of
    1644 games ran at 8 MB/s with 14 free cores -- an hour just for reading, and it
    is repeated before EVERY run.
    """
    import concurrent.futures as _cf

    def read(f):
        """Only reading the npz -- the one thing the threads do. Labeling
        (`encode_replay` -> `executed_market`) replaces engine functions for the whole
        module and in threads silently mixes up the attribution of trades (review 30.09, F1)."""
        cp = cache_path(f, ver) if ver else None
        if not (cp and os.path.exists(cp)):
            return _MISS
        try:
            z = np.load(cp)
            return {k: (z[k].astype(np.float32) if k in _FP16 else z[k]) for k in z.files}
        except Exception:
            return _MISS

    stale, dropped = [], []

    def one(f, d):
        """Called in the main thread: a cache miss and a stale cache
        are labeled here, sequentially."""
        if d is _MISS:
            try:
                d = encode_replay_cached(f, teacher_only, gamma, None)
            except Exception as e:
                print(f"  ⚠ game {os.path.basename(f)} dropped: {e}", flush=True)
                dropped.append(f)
                return None
            if d is not None and ver:
                _write_cache(f, ver, d)
        if d is None:
            return None
        if not _sides_ok(f, d, teacher_only):
            # A cache with different sides: built under `--teacher both` or before
            # the game got into episodes.json (F3). Re-label from the replay.
            stale.append(f)
            d = encode_replay_cached(f, teacher_only, gamma, None) \
                if os.path.exists(f) else None
            if d is None or not _sides_ok(f, d, teacher_only):
                print(f"  ⚠ game {os.path.basename(f)} dropped: sides in the cache do not "
                      "match episodes.json, and there is no replay", flush=True)
                dropped.append(f)
                return None
            if ver:
                _write_cache(f, ver, d)
        if _STYLE_MAP is not None or _FT_STYLES is not None:
            d = _fix_style(f, d, teacher_only)
        if _FT_STYLES is not None:
            keep = np.isin(np.asarray(d["style"]), list(_FT_STYLES))
            if not keep.any():
                return None
            n_ = len(keep)
            d = {k: (v[keep] if hasattr(v, "shape") and v.ndim and v.shape[0] == n_ else v)
                 for k, v in d.items()}
        if qty_exec:
            d = _qty_exec(d)
        out = {}
        for k, v in d.items():
            if k in CORPUS_F16:
                out[k] = v.astype(np.float16)
            elif k in CORPUS_I16:
                out[k] = v.astype(np.int16)
            else:
                out[k] = v
        return out

    jobs = max(1, jobs or 1)

    def to_dev(d):
        """A game straight onto the device: nothing accumulates in RAM except that one game.
        FARMCORE_CPU_CONCAT=1 -- chunks stay in RAM, and the already
        concatenated key goes to the device (30.09: a corpus of 2,671 games, concatenation on the card = OOM)."""
        if device is None or d is None or os.environ.get("FARMCORE_CPU_CONCAT"):
            return d
        return {k: torch.from_numpy(v).to(device, non_blocking=True)
                for k, v in d.items()}

    if jobs == 1:
        parts = [x for x in (to_dev(one(f, read(f))) for f in files) if x is not None]
    else:
        # map preserves file order: the corpus must be assembled identically at
        # any number of threads, otherwise the split by games would move with it.
        # Transfer to the device is in THIS thread, not in the workers: CUDA does not like
        # foreign threads, and the transfer itself is cheaper than unpacking.
        with _cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            parts = [x for x in (to_dev(one(f, d)) for f, d in zip(files, ex.map(read, files)))
                     if x is not None]
    if stale:
        print(f"  ⚠ cache with sides not matching episodes.json: {len(stale)} games "
              f"re-labeled (example: {os.path.basename(stale[0])})", flush=True)
    if dropped:
        print(f"  ⚠ games dropped: {len(dropped)} of {len(files)}", flush=True)
        if len(dropped) > 0.05 * len(files):
            raise SystemExit(f"refusal: dropped {len(dropped)} of {len(files)} games (>5%)")
    if not parts:
        return None
    keys = list(parts[0].keys())
    if device is None:
        # one key at a time, freeing chunks: the peak is the corpus plus one key
        C = {}
        for k in keys:
            C[k] = np.concatenate([p[k] for p in parts])
            for p in parts:
                del p[k]
    else:
        # concatenation one key at a time, freeing chunks along the way: the peak on
        # the device is the corpus plus one key, not a doubled corpus
        C = {}
        for k in keys:
            if isinstance(parts[0][k], np.ndarray):          # FARMCORE_CPU_CONCAT
                C[k] = torch.from_numpy(np.concatenate([p[k] for p in parts])).to(device)
            else:
                C[k] = torch.cat([p[k] for p in parts])
            for p in parts:
                del p[k]
    n = len(C["value"])
    mb = sum((v.nbytes if hasattr(v, "nbytes") else v.numel() * v.element_size())
             for v in C.values()) / 1e6
    print(f"  corpus in memory: {n:,} samples, {mb:,.0f} MB", flush=True)
    return C


def corpus_to_device(C, device):
    """The whole corpus into GPU memory: 4-5 GB in float16 with 80 GB on the card.
    Then a batch is a gather inside the GPU, not a host->device copy
    (which ate a third of the step)."""
    out = {}
    for k, v in C.items():
        t = torch.from_numpy(v)
        out[k] = t.to(device, non_blocking=True)
    return out


def collate_idx(C, idx, device):
    """A batch by slicing the corpus: no dicts per sample, no np.stack."""
    if torch.is_tensor(C["value"]):            # the corpus is already on the device
        ix = idx if torch.is_tensor(idx) else torch.as_tensor(idx, device=device)
        g = lambda k: C[k].index_select(0, ix)
    else:
        g = lambda k: torch.from_numpy(C[k][idx]).to(device, non_blocking=True)
    x = dict(tiles_me=g("tiles_me").float(), tiles_op=g("tiles_op").float(),
             units=g("units").float(), unit_mask=g("unit_mask"),
             products=g("products").float(), glob=g("glob").float(),
             crops=g("crops").float(), animals=g("animals").float(),
             tile_mask=g("tile_mask"), vcrit=g("vcrit").float())
    if _STYLE_MAP is not None:
        x["style"] = _STYLE_MAP.to(device)[g("style").long()]
    t = dict(ops=g("ops").long(), sell=g("sell").long(), buy=g("buy").long(),
             hire=g("hire").long(), land=g("land").long(), seed=g("seed").long(),
             animal=g("animal").long(), value=g("value").long(),
             sell_have=g("sell_have").long(), qty=g("qty").long(),
             target=g("target").long(), effect=g("effect").float(),
             econ=g("econ").float(), legal=g("legal"),
             legal_now=g("legal_now"), w=g("w").float(),
             mdiff=g("mdiff").float(), to_end=g("to_end").float())
    return x, t


# ----- input augmentations (28.09) --------------------------------------------
# Only on the training batch, AFTER collate_idx: the cache does not change, validation
# (`evaluate`) and play (`play`) see a clean input. Why, in an internal
# imitation-fidelity note: clone 56407295 and the glob[18] feature.
#
# glob indices (features.encode): 16 shed fill, 18 wheat in the shed per
# animal, 19-21 animals in the shed, 27 feed deficit; products[:, 3] -- the stock of
# a good in the shed / SHED_CAP; crops[:, 17] -- the stock of a crop in the shed / 100.
G_SHED_USED, G_WHEAT_PER_ANIMAL, G_N_ANIMAL, G_FEED_DEF = 16, 18, 11, 27
P_SHED, C_SHED = 3, 17


def aug_shed(x, t, p, kmax, p_other=0.3, gen=None):
    """A consistent shed stock shift (counterfactual input).

    For a share `p` of rows, δ units are "added" to the shed: wheat -- always
    (δ ~ U{1..kmax}; our feed rule in play keeps more wheat than the
    teacher), other goods -- each with probability `p_other`. All columns
    into which the stock enters are recomputed together: glob 16/18/27, products[:,3],
    crops[:,17]. Otherwise the net would read the untouched copy and the noise would teach nothing.
    The shift is only upward and not above capacity: the label "sold n" stays executable.

    Labels that depend on the stock of a good BY RIGHT (sell/buy this good,
    seeds of this crop) are switched off on the changed rows by the masks
    `t[sell_m|buy_m|seed_m]`; the other decisions (land, hire, animals, units)
    are considered invariant to the stock -- exactly the link we break.
    """
    B = x["glob"].shape[0]
    dev = x["glob"].device
    cap = float(features.envcfg.SHED_CAP)
    nP = x["products"].shape[1]
    row = torch.rand(B, device=dev, generator=gen) < p
    pick = torch.rand(B, nP, device=dev, generator=gen) < p_other
    pick[:, 0] = True                                   # WHEAT = PRODUCTS[0]
    pick &= row.unsqueeze(1)
    delta = torch.randint(1, kmax + 1, (B, nP), device=dev, generator=gen).float() * pick
    used = (x["glob"][:, G_SHED_USED] * 100).round()
    room = (cap - used).clamp_min(0)
    tot = delta.sum(1)
    scale = torch.where(tot > room, room / tot.clamp_min(1), torch.ones_like(tot))
    delta = (delta * scale.unsqueeze(1)).floor()
    pick = delta > 0
    d_all = delta.sum(1)
    dW = delta[:, 0]
    g = x["glob"]
    shed_w = (x["products"][:, 0, P_SHED] * cap).round()
    n_anim = (g[:, G_N_ANIMAL] * 100).round()
    g[:, G_SHED_USED] = g[:, G_SHED_USED] + d_all / 100
    hit = dW > 0                          # recompute only where there was a shift
    g[:, G_WHEAT_PER_ANIMAL] = torch.where(
        hit, (shed_w + dW) / n_anim.clamp_min(1) / 10, g[:, G_WHEAT_PER_ANIMAL])
    g[:, G_FEED_DEF] = torch.where(hit, (g[:, G_FEED_DEF] - dW / 20).clamp(-1, 1),
                                   g[:, G_FEED_DEF])
    x["products"][:, :, P_SHED] = x["products"][:, :, P_SHED] + delta / cap
    nC = x["crops"].shape[1]
    x["crops"][:, :, C_SHED] = x["crops"][:, :, C_SHED] + delta[:, :nC] / 100
    keep = (~pick).float()
    t["sell_m"] = t["sell_m"] * keep if "sell_m" in t else keep
    t["buy_m"] = t["buy_m"] * keep if "buy_m" in t else keep
    t["seed_m"] = t["seed_m"] * keep[:, :nC] if "seed_m" in t else keep[:, :nC]
    return int(row.sum())


# Unit columns derived from its cell (features.encode): x, y, distance to the
# center, at the shed, on a plant, on an animal, on a closed cell.
U_POS_COLS = (2, 3, 4, 5, 11, 12, 16)


def aug_units(x, t, p, gen=None):
    """Unit positions from another row of the batch (29.09, a manual review of games).

    Clone 56266758: on d8-d11 it has 3-4 empty pastures and $2-13k, and P(sheep)
    0.002. Substitution: all clone units to the shed -- P(cow) 0.000 -> 0.997.
    The teacher buys animals in the phase of his plan (the crew gathered at the shed after
    hiring), and the net learned the link "buy when the crew is at the shed". A clone with a
    dispersed crew never gets into this window. An animal from the shed
    can be carried off by any hand, so market decisions (except sells, which depend
    on who drops what in this turn) are considered invariant to positions.

    For a share `p` of rows the positional columns of each live unit and its `legal_now`
    (legality on the cell is a position too) are taken from the unit with the same number from a
    random row of the batch. On these rows only the market without sells is learned:
    `t["mkt_only"]` = 1 (units, effects, economy and critic are off, the market is
    at weight 1 instead of the outcome weight, see `loss_fn`), `sell_m` = 0.
    """
    u = x["units"]
    B = u.shape[0]
    dev = u.device
    row = torch.rand(B, device=dev, generator=gen) < p
    src = torch.randint(0, B, (B,), device=dev, generator=gen)
    both = x["unit_mask"] & x["unit_mask"][src] & row.unsqueeze(1)          # (B, U)
    cols = torch.tensor(U_POS_COLS, device=dev)
    donor = u[src][:, :, cols]
    u[:, :, cols] = torch.where(both.unsqueeze(-1), donor, u[:, :, cols])
    ln = t["legal_now"]
    t["legal_now"] = torch.where(both.unsqueeze(-1), ln[src], ln)
    mo = row.float()
    t["mkt_only"] = torch.maximum(t["mkt_only"], mo) if "mkt_only" in t else mo
    nP = t["sell"].shape[1]
    keep = (~row).float().unsqueeze(1).expand(-1, nP)
    t["sell_m"] = t["sell_m"] * keep if "sell_m" in t else keep
    return int(row.sum())


def _unpack(d):
    keys = ("tiles_me", "tiles_op", "tile_mask", "units", "unit_mask", "products",
            "glob", "crops", "animals", "vcrit")
    for j in range(len(d["value"])):
        yield ({k: d[k][j] for k in keys}, d["ops"][j], d["sell"][j], d["buy"][j],
               int(d["hire"][j]), int(d["land"][j]), d["seed"][j], d["animal"][j],
               int(d["value"][j]), float(d["w"][j]), d["target"][j],
               d["effect"][j], d["econ"][j], d["legal"][j], d["sell_have"][j],
               d["qty"][j], d["legal_now"][j], float(d["mdiff"][j]),
               float(d["to_end"][j]))


def sample_stream(files, workers=0, teacher_only=True, gamma=0.999, shuffle=0,
                  seed=0, ver=None):
    """A sample stream with a shuffle buffer: consecutive turns of one game
    must not land in one batch."""
    rng = random.Random(seed)
    buf = []

    def emit(s):
        if shuffle <= 0:
            yield s
            return
        buf.append(s)
        if len(buf) >= shuffle:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()

    def raw():
        if workers <= 0:
            for f in files:
                d = encode_replay_cached(f, teacher_only, gamma, ver)
                if d is not None:
                    yield from _unpack(d)
            return
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
            it = iter(files)
            pending = set()
            for _ in range(workers + 3):
                f = next(it, None)
                if f is None:
                    break
                pending.add(ex.submit(encode_replay_cached, f, teacher_only, gamma, ver))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        d = fut.result()
                    except Exception:
                        d = None
                    f = next(it, None)
                    if f is not None:
                        pending.add(ex.submit(encode_replay_cached, f, teacher_only, gamma, ver))
                    if d is not None:
                        yield from _unpack(d)

    for s in raw():
        yield from emit(s)
    rng.shuffle(buf)
    yield from buf


def collate(batch, device):
    st = lambda key: torch.from_numpy(np.stack([b[0][key] for b in batch])).to(device)
    return dict(
        tiles_me=st("tiles_me").float(), tiles_op=st("tiles_op").float(),
        units=st("units").float(), unit_mask=st("unit_mask"),
        products=st("products").float(), glob=st("glob").float(),
        crops=st("crops").float(), animals=st("animals").float(),
        tile_mask=st("tile_mask"), vcrit=st("vcrit").float(),
    ), dict(
        ops=torch.from_numpy(np.stack([b[1] for b in batch])).to(device),
        sell=torch.from_numpy(np.stack([b[2] for b in batch])).to(device),
        buy=torch.from_numpy(np.stack([b[3] for b in batch])).to(device),
        hire=torch.tensor([b[4] for b in batch], device=device),
        land=torch.tensor([b[5] for b in batch], device=device),
        seed=torch.from_numpy(np.stack([b[6] for b in batch])).to(device),
        animal=torch.from_numpy(np.stack([b[7] for b in batch])).to(device),
        value=torch.tensor([b[8] for b in batch], device=device, dtype=torch.long),
        w=torch.tensor([b[9] for b in batch], device=device, dtype=torch.float32),
        target=torch.from_numpy(np.stack([b[10] for b in batch])).to(device),
        effect=torch.from_numpy(np.stack([b[11] for b in batch])).to(device).float(),
        econ=torch.from_numpy(np.stack([b[12] for b in batch])).to(device).float(),
        legal=torch.from_numpy(np.stack([b[13] for b in batch])).to(device),
        sell_have=torch.from_numpy(np.stack([b[14] for b in batch])).to(device),
        qty=torch.from_numpy(np.stack([b[15] for b in batch])).to(device),
        legal_now=torch.from_numpy(np.stack([b[16] for b in batch])).to(device),
        mdiff=torch.tensor([b[17] for b in batch], device=device, dtype=torch.float32),
        to_end=torch.tensor([b[18] for b in batch], device=device, dtype=torch.float32),
    )


def logit_prior(C, device="cpu"):
    """Logarithm of the prior frequency of op classes and pointer columns.

    Logit-adjusted loss (Menon et al., ICLR 2021): at training the log of the class
    frequency is added to the class logit, and then a plain argmax at inference is optimal
    for the balanced error. This replaces the hand-tuned `stay_bias`,
    which fixed TWO different decisions AT ONCE (the "stay" column of the pointer and
    PASS of the op head) with one number and was tuned by money.

    The correction is given ONLY to these two heads, and the reason is not technical. For op and the
    pointer a decision is made by each unit each turn, one of mutually
    exclusive ones; the PASS/STAY dominance is a label skew, not economics, and it
    must be removed. Market gates are independent "act or not" decisions with an
    asymmetric cost of error: an extra buy immediately costs money. There the base
    frequency IS the content of the decision, and the correction breaks it. Measurement: with the correction on
    gates accuracy fell to 0.1-0.3, the agent spent the bank on the first turn and
    finished the game with zero. The 0.5 threshold on a calibrated sigmoid is the
    MAP rule, one can deviate from it only by a value estimate, not by a
    hand-tuned number.

    The quantity scale is not given the correction either: the asymmetry of the thresholds P(n >= k) IS
    the quantity. Having evened it out, the net keeps all thresholds open, and the median decode
    (Frank-Hall) returns the cap -- measurement: the quantity error for buy 89.2 at cap 100.
    """
    def pri(v, n, ignore=-100):
        v = v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        v = v[v != ignore]
        c = np.bincount(v.ravel().astype(np.int64), minlength=n)[:n]
        f = np.maximum(c, 1.0) / max(1.0, c.sum())
        return torch.tensor(np.log(f), dtype=torch.float32, device=device)
    return {"op": pri(C["ops"], N_OPS),
            "target": pri(C["target"], N_TILES + 1)}


def _masked_turn_mean(ce, sel, w):
    """Mean CE first within a turn, then across turns, over the chosen decisions.

    Averaging within a turn is a trick from pokemoni (`bc_sequential_loss`): otherwise a turn
    with 12 units weighs twelvefold against a turn with one. `sel` is which decisions
    are counted at all (live, with a legal label). Turns without a single
    counted decision are dropped from the outer mean, rather than entering it
    as zero.
    """
    sel = sel.float()
    per_turn = (ce * sel).sum(1) / sel.sum(1).clamp_min(1)
    has = (sel.sum(1) > 0).float() * w
    return (per_turn * has).sum() / has.sum().clamp_min(1e-6)


def densify_mdiff(C, k):
    """Dense target of the HL head: the margin change over `k` decisions instead of the final one.

    Why. At γ = 1 the target `tanh(final difference / σ)` is CONSTANT within a game,
    i.e. N games give N labels for hundreds of thousands of states. Measurement 23.09: the tower
    on 150 self-play games memorized them by heart (train AUC 1.000 vs 0.518 on
    unseen ones, in the first third of the season 0.365 -- below a coin). The same failure and the
    same cure are described for AlphaGo ("memorised the game outcomes", fixed with
    positions FROM DIFFERENT games) and KataGo ("only one noisy binary result per
    entire game", fixed with dense auxiliary targets, 1.65x to the training
    speed) -- analysis in an internal replay-buffer note.

    Here there are as many labels as there are states. The W/D/L head stays
    on the outcome: it reads `C["value"]`, not `mdiff` -- so the sparse
    target does not go anywhere, a dense auxiliary one is added to it.

    Game boundaries are taken from `to_end` (decreasing within a game, a jump up is a
    new one), not from the file list: the corpus is already concatenated.

    ⚠ The target is computed IN THE SAME UNITS in which the column `glob[4]` lies
    (`features.encode` writes the bank difference divided by its own scale there), and is
    NOT converted to dollars. So there is no second copy of the multiplier from
    `features.py` here: if the divisor changes there, the target's scale simply changes, and
    `--value-sigma` is tuned by the printed std. An attempt to derive the multiplier from
    the data failed in substance, and this is worth remembering: the last RECORDED
    state of a game is the observation BEFORE the last turn, while `mdiff` is the difference
    AFTER it, with a whole market turn in between (at season end usually a shed dump),
    and the ratio diverged by 46% over 83 games.
    """
    md_col = np.asarray(C["glob"], dtype=np.float32)[:, 4]
    te = np.asarray(C["to_end"])
    starts = np.r_[0, np.where(np.diff(te) > 0)[0] + 1]
    ends = np.r_[starts[1:], len(te)]

    out = np.empty(len(md_col), dtype=np.float32)
    for s, e in zip(starts, ends):
        m = md_col[s:e]
        nxt = np.minimum(np.arange(e - s) + k, e - s - 1)
        out[s:e] = m[nxt] - m
    C["mdiff"] = out
    return len(starts), float(np.std(out)), float(np.percentile(np.abs(out), 95))


def value_target(t, sigma=VAL.RL_SIGMA, gamma=1.0):
    """Target of the HL head: normalized final bank difference.

    `z = tanh(Δmoney / σ) · γ^(decisions to the end)`. Computed FROM THE CACHE, not from a
    bootstrap: in the BC stage the trajectory is known to the end, and the outcome itself serves as the target.
    TD(λ) bootstrap is the PPO trainer's business (`value_head.gae`), it is absent
    here on purpose: a critic trained by bootstrap on its own errors on
    foreign trajectories is already an off-policy estimate, not labeling.

    γ = 1 by default. The horizon is fixed (720 turns), the reward is terminal,
    and a discount would multiply the target by a known function of time -- the critic would learn
    it as an "hour" feature, not as a state estimate. Credit compression under PPO
    is given by λ.
    """
    z = torch.tanh(t["mdiff"] / sigma)
    if gamma != 1.0:
        z = z * (gamma ** t["to_end"])
    return z


def _match_targets(logp, lt, moving):
    """Walking units' targets as a SET (28.09): permute labels among the walking
    units of a turn so that the total -log p is minimal (the Hungarian algorithm).

    The planner-teacher hands out cells to hands by its own logic; which exact hand
    went to a cell is often not visible from the state, and the per-unit CE penalizes it
    as an error (~3/4 of unit disagreements for 56521745 are "who and in what order").
    This does not concern units working in place and standing: they have their own label.
    logp: [B,U,T1] (detach), lt: [B,U] labels, moving: [B,U] bool -> new labels."""
    from scipy.optimize import linear_sum_assignment
    mv = moving.cpu().numpy()
    rows = np.nonzero(mv.sum(1) >= 2)[0]
    if len(rows) == 0:
        return lt.clone()
    lab = lt.cpu().numpy().copy()
    lp = logp[torch.as_tensor(rows, device=logp.device)].float().cpu().numpy()
    for n, b in enumerate(rows):
        idx = np.nonzero(mv[b])[0]
        tg = lab[b, idx]
        r, c = linear_sum_assignment(-lp[n][idx][:, tg])
        lab[b, idx[r]] = tg[c]
    out = torch.from_numpy(lab)
    return out.to(lt.device)


def loss_fn(out, t, value_coef=0.5, mkt_coef=0.6, tgt_coef=0.3,
            op_coef=0.3, eff_coef=1.0, econ_coef=0.5, prior=None, tau=1.0,
            win_w=1.5, val_sigma=VAL.RL_SIGMA, val_gamma=1.0, hl_coef=1.0,
            set_targets=False):
    """The main target is the turn result (effect map and economy), actions are an
    executive layer with a reduced weight (internal effect-loss note).

    `win_w` -- how many times more a turn from a WON game weighs than from a
    lost or drawn one. The simplest form of outcome weighting: the teacher
    errs in won games too, but in lost ones the share of errors is higher, and
    we have nothing to tell them apart one by one. ⚠ The constant itself is NOT measured (neither 1.5 nor
    1.25): it came from the first version and has never been checked in money.
    """
    Bn = t["ops"].shape[0]
    # One source of truth for the weight. Before, it lay in the cache (`t["w"]`, 1.5 for a
    # win) and was applied to op/tgt/mkt/land/eff, but NOT to value, econ and qty --
    # i.e. three heads learned on a different distribution than the rest.
    w = torch.where(t["value"] == 0,
                    torch.full_like(t["value"], 1, dtype=torch.float32) * win_w,
                    torch.ones_like(t["value"], dtype=torch.float32))
    # "Market only" rows (`aug_units`): unit positions are substituted, so
    # ONLY the market (sell/buy/seed/animal/hire/land) is learned, with weight 1; units,
    # effects, economy and critic are switched off on them.
    w_mkt = w
    vmask = None
    if "mkt_only" in t:
        mo = t["mkt_only"] > 0
        w_mkt = torch.where(mo, torch.ones_like(w), w)
        w = w * (~mo).float()
        vmask = (~mo).float()
    t = dict(t, w=w)
    live = t["ops"] != -100
    op_lg = out["op_logits"]
    if "legal" in t:                       # illegal classes outside the softmax
        op_lg = op_lg.masked_fill(t["legal"] == 0, -1e9)
    if prior is not None:              # logit adjustment instead of a threshold on PASS
        op_lg = op_lg + tau * prior["op"]
    if "legal" in t:                   # safety net: a label outside the mask is not learned
        ok_lbl = t["legal"].gather(-1, t["ops"].clamp_min(0).unsqueeze(-1)).squeeze(-1)
        t = dict(t, ops=torch.where((t["ops"] < 0) | (ok_lbl > 0), t["ops"],
                                    torch.full_like(t["ops"], -100)))
        live = t["ops"] != -100
    ce = F.cross_entropy(op_lg.reshape(-1, N_OPS), t["ops"].reshape(-1),
                         ignore_index=-100, reduction="none").reshape(Bn, -1)
    op = _masked_turn_mean(ce, live, t["w"])

    T1 = out["target_logits"].shape[-1]
    lt = t["target"]
    tl = out["target_logits"]
    # The same safety net as for `op`, but for the pointer. Illegal targets are suppressed
    # to -inf (`tile_mask`), and if the teacher's label lands on such a cell,
    # cross-entropy becomes INFINITE, and the gradient nan. The old
    # teacher never had this; 56216119 does (in the log `tgt inf`).
    # A label outside the mask is not a decision but a state divergence, and one cannot learn from it:
    # we drop it the same way we dropped illegal op.
    bad_t = torch.isinf(tl.gather(-1, lt.clamp_min(0).unsqueeze(-1)).squeeze(-1))
    lt = torch.where((lt < 0) | ~bad_t, lt, torch.full_like(lt, -100))
    live_t = lt != -100
    if prior is not None:              # the same correction for the "stay" column
        tl = tl + tau * prior["target"][:T1]
    if set_targets:
        # walking = without a job label (op is labeled only for units working here and
        # standing) and with a target cell
        moving = live_t & (t["ops"] == -100) & (lt != STAY)
        lt = _match_targets(F.log_softmax(tl.detach().float(), -1), lt, moving)
    ct = F.cross_entropy(tl.reshape(-1, T1), lt.reshape(-1),
                         ignore_index=-100, reduction="none").reshape(Bn, -1)
    tgt = _masked_turn_mean(ct, live_t, t["w"])

    def count_head(name, logits, n):
        """Gate (act or not) + CORN chain (how many), all binary.

        The chain is learned only where the gate is open: "how many" makes sense
        only conditional on acting. No class weights -- the gate has two classes and
        a dense prior, and the conditional c_k is learned only on n >= k-1 (counts.loss,
        R2: before 22.09 there was a Frank-Hall scale with non-monotone thresholds here).
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
            n = n.unsqueeze(1)
        gate_lg, ord_lg, all_lg = CNT.split(logits, name)
        gate_t = (n > 0).float()
        lg = F.binary_cross_entropy_with_logits(gate_lg, gate_t, reduction="none")
        lo = CNT.loss(ord_lg, n, head=name) * gate_t      # quantity -- with open gates
        all_t = None
        if all_lg is not None and "sell_have" in t:
            have = t["sell_have"]
            all_t = ((n > 0) & (n >= have)).float()
        # B3: the chain only where the decoder asks for it: the gate is open
        # and the "sell all stock" flag is NOT set (`policy.run_turn.decide`: with the
        # flag n = have is taken, and the chain is not read). Before, it was also learned
        # on rows with "all" (45.7% of the teachers' SELL orders), i.e. it learned the
        # marginal P(n), while P(n | not all) is played.
        if all_t is not None:
            lo = lo * (1.0 - all_t)
        l = lg + lo
        if all_t is not None:
            l = l + F.binary_cross_entropy_with_logits(all_lg, all_t,
                                                       reduction="none") * gate_t
        # ⚠ Restricting the `buy` head to wheat and fertilizer is ROLLED BACK: measurement v16
        # vs v15 gave -$18,806 of median, 8 wins of 48, p=3.3e-06. Mechanism:
        # removing the gradient for seven goods is not enough -- their `E[n]` keeps flowing into
        # the coordination block (model.py, co_mkt), and the block smears
        # untrained garbage by attention over ALL heads. It can be returned
        # only together with a mask inside the model, not by the loss alone.
        if f"{name}_m" in t:                  # shed augmentation: the label is undefined
            l = l * t[f"{name}_m"].to(l.dtype)
        l = l.mean(1)
        return (l * w_mkt).sum() / w_mkt.sum()

    sell = count_head("sell", out["sell_logits"], t["sell"])
    buy = count_head("buy", out["buy_logits"], t["buy"])
    seed = count_head("seed", out["seed_logits"], t["seed"])
    animal = count_head("animal", out["animal_logits"], t["animal"])
    hire = count_head("hire", out["hire_logits"], t["hire"])
    # PICKUP quantity: the same CORN chain, but per unit and only where the
    # teacher actually took from the shed (the gate is always open --
    # supervision goes through the only branch where the quantity is meaningful; the decoder
    # does not read this head's gate, n >= 1 by construction)
    if "qty" in t:
        qm = (t["qty"] >= 0).float()
        nq = t["qty"].clamp_min(0)
        gq, oq, _ = CNT.split(out["qty_logits"], "pickup")
        lq = (F.binary_cross_entropy_with_logits(gq, (nq > 0).float(),
                                                 reduction="none")
              + CNT.loss(oq, nq, head="pickup"))
        wq = w.view(-1, *([1] * (lq.dim() - 1)))
        qty_l = (lq * qm * wq).sum() / (qm * wq).sum().clamp_min(1.0)
    else:
        qty_l = out["qty_logits"].sum() * 0.0
    land_lg = out["land_logits"].squeeze(-1)
    land_t = (t["land"] > 0).float()
    land_l = F.binary_cross_entropy_with_logits(land_lg, land_t, reduction="none")
    if "land_m" in t:
        land_l = land_l * t["land_m"].to(land_l.dtype)
    land = (land_l * w_mkt).sum() / w_mkt.sum()
    # The outcome weight does NOT extend to the critic, and this is a fix, not taste:
    # "1.5 for a win" is a way of saying "in lost games the teacher errs
    # more often", i.e. an argument about ACTIONS. By multiplying the value loss by it
    # we systematically inflated P(win) -- exactly what the v2 docstring promised not to
    # do ("does NOT extend to the critic"), while the code did.
    val_wdl = F.cross_entropy(out["value_logits"], t["value"], reduction="none")
    val_wdl = (val_wdl.mean() if vmask is None
               else (val_wdl * vmask).sum() / vmask.sum().clamp_min(1))
    # Second head: HL-Gauss histogram over the normalized bank difference
    # (arXiv:2403.03950). The outcome gives the sign, the histogram gives the gradation, which the
    # terminal ±1 lacks altogether.
    z = value_target(t, val_sigma, val_gamma)
    val_hl = -(VAL.hl_target(z)
               * F.log_softmax(out["value_hl"].float(), -1)).sum(-1)
    val_hl = (val_hl.mean() if vmask is None
              else (val_hl * vmask).sum() / vmask.sum().clamp_min(1))
    val = val_wdl + hl_coef * val_hl
    mkt = sell + buy + seed + animal + hire + land + qty_l

    eff = F.binary_cross_entropy_with_logits(out["effect_logits"], t["effect"],
                                             reduction="none").mean((1, 2))
    eff = (eff * t["w"]).sum() / t["w"].sum().clamp_min(1e-6)
    econ_e = F.smooth_l1_loss(out["econ"], t["econ"], reduction="none").mean(-1)
    econ = (econ_e * w).sum() / w.sum().clamp_min(1e-6)

    total = (op_coef * op + tgt_coef * tgt + mkt_coef * mkt + value_coef * val
             + eff_coef * eff + econ_coef * econ)
    return total, {"op": float(op.detach()), "tgt": float(tgt.detach()),
                   "mkt": float(mkt.detach()), "val": float(val.detach()),
                   "val_wdl": float(val_wdl.detach()), "val_hl": float(val_hl.detach()),
                   "eff": float(eff.detach()), "econ": float(econ.detach()),
                   "qty": float(qty_l.detach())}


def auc(score, pos):
    """AUC without sklearn: the share of (positive, negative) pairs in the correct order.

    Via Mann-Whitney ranks, so it is computed in a sort, not in a pair
    enumeration (on 26,603 states enumeration is 350 million comparisons).
    """
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    r = np.empty(len(score), dtype=np.float64)
    r[order] = np.arange(1, len(score) + 1)
    s = np.sort(score)
    i = 0                                   # mean rank over ties
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ev(pred, true):
    """Explained variance share: 1 − Var(true − pred) / Var(true).

    Not R² and not correlation: correlation does not see shift and scale, and a critic
    that emits exactly half of the needed value is useless for GAE.
    """
    true = np.asarray(true, dtype=np.float64)
    v = true.var()
    if v < 1e-12:
        return float("nan")
    return float(1.0 - (true - np.asarray(pred, dtype=np.float64)).var() / v)


@torch.no_grad()
def evaluate(model, C, dev, batch=1024, val_sigma=VAL.RL_SIGMA, val_gamma=1.0,
             prior=None, beta=0.0):
    """Validation over the in-memory corpus.

    `val_sigma`/`val_gamma` arrive here the same as in the loss, and this is not
    carefulness but that very rule: training and evaluation must compute ONE
    quantity. With a default inside the function, tuning `--value-sigma` would silently change
    the training target while leaving EV computed on the old scale -- and the plot
    "the critic got worse" would be drawn from nothing.
    """
    model.eval()
    n = len(C["value"])
    op_ok = op_n = turn_ok = turn_n = tgt_ok = tgt_n = 0
    # Accuracy in the PLAYED family (policy.family: β·prior on op and target). The old
    # `op_acc`/`tgt_acc` are the argmax of RAW logits (β = 0), while the net plays with β = 1:
    # validation measured a different policy than the ladder (internal policy-parity note, 24.09).
    # Kept as they were so that logs of earlier runs remain comparable.
    op_ok_p = tgt_ok_p = 0
    e_tp = e_fp = e_fn = 0
    econ_err, vp, vt, vz, vzh = [], [], [], [], []
    mk = {k: [0, 0, 0, 0.0] for k in MKT_HEADS}
    for s in range(0, n, batch):
        idx = np.arange(s, min(n, s + batch))
        x, t = collate_idx(C, idx, dev)
        o = model(**x, legal=t["legal_now"])
        # the same mask as in the loss and in the decoder: masked classes get no
        # gradient, their logits are arbitrary, and without the mask argmax takes garbage
        pred = o["op_logits"].masked_fill(t["legal"] == 0, -1e9).argmax(-1)
        live = t["ops"] != -100
        op_ok += int(((pred == t["ops"]) & live).sum()); op_n += int(live.sum())
        pt = o["target_logits"].argmax(-1)
        livet = t["target"] != -100
        tgt_ok += int(((pt == t["target"]) & livet).sum()); tgt_n += int(livet.sum())
        if prior is not None and beta:
            pp = (o["op_logits"] + beta * prior["op"]).masked_fill(
                t["legal"] == 0, -1e9).argmax(-1)
            T1 = o["target_logits"].shape[-1]
            ptp = (o["target_logits"] + beta * prior["target"][:T1]).argmax(-1)
            op_ok_p += int(((pp == t["ops"]) & live).sum())
            tgt_ok_p += int(((ptp == t["target"]) & livet).sum())
        both = ((pred == t["ops"]) | ~live) & ((pt == t["target"]) | ~livet)
        turn_ok += int(both.all(-1).sum()); turn_n += len(idx)
        for k in COUNT_HEADS:
            lg = o[f"{k}_logits"]
            if lg.dim() == 2:
                lg = lg.unsqueeze(1)
            n_true = t[k] if t[k].dim() > 1 else t[k].unsqueeze(1)
            have = t["sell_have"] if k == "sell" else None
            n_pred = CNT.decode(lg.float(), k, have=have)
            tp = ((n_pred > 0) & (n_true > 0))
            mk[k][0] += int(tp.sum())                      # correctly opened the gate
            mk[k][1] += int((n_pred > 0).sum())             # opened in total
            mk[k][2] += int((n_true > 0).sum())             # needed in total
            mk[k][3] += float((n_pred - n_true).abs()[tp].sum())   # quantity error
        pl = (torch.sigmoid(o["land_logits"].squeeze(-1)) > 0.5)
        lt = t["land"] > 0
        mk["land"][0] += int((pl & lt).sum()); mk["land"][1] += int(pl.sum())
        mk["land"][2] += int(lt.sum())
        pe = (torch.sigmoid(o["effect_logits"].float()) > 0.5).float()
        te = t["effect"]
        e_tp += int((pe * te).sum()); e_fp += int((pe * (1 - te)).sum())
        e_fn += int(((1 - pe) * te).sum())
        econ_err.append(float((o["econ"].float() - t["econ"]).abs().mean()))
        pv = torch.softmax(o["value_logits"].float(), -1)
        vp.append((pv[:, 0] - pv[:, 2]).cpu().numpy()); vt.append(t["value"].cpu().numpy())
        vz.append(value_target(t, val_sigma, val_gamma).cpu().numpy())
        vzh.append(VAL.hl_expect(o["value_hl"]).cpu().numpy())
    model.train()
    vp = np.concatenate(vp); vt = np.concatenate(vt)
    vz = np.concatenate(vz); vzh = np.concatenate(vzh)
    z = np.where(vt == 0, 1.0, np.where(vt == 2, -1.0, 0.0))
    m = {"op_acc": op_ok / max(1, op_n), "tgt_acc": tgt_ok / max(1, tgt_n),
         "turn_acc": turn_ok / max(1, turn_n), "turns": turn_n,
         "val_corr": float(np.corrcoef(vp, z)[0, 1]) if vp.std() > 1e-8 else 0.0,
         # AUC of the outcome by P(W)−P(L) and the explained variance share of the HL head --
         # two R1 acceptance quantities. Validation accuracy of actions has no relation to
         # the critic's quality, yet it was used to judge everything before.
         "val_auc": auc(vp, vt == 0),
         "val_ev": ev(vzh, vz),
         "eff_f1": 2 * e_tp / max(1, 2 * e_tp + e_fp + e_fn),
         "econ_mae": float(np.mean(econ_err))}
    if prior is not None and beta:
        m["op_acc_play"] = op_ok_p / max(1, op_n)
        m["tgt_acc_play"] = tgt_ok_p / max(1, tgt_n)
    for k, (tp, npred, ntrue, abserr) in mk.items():
        m[f"{k}_prec"] = tp / max(1, npred)      # of opened gates, correct
        m[f"{k}_rec"] = tp / max(1, ntrue)       # of the needed, caught
        m[f"{k}_mae"] = abserr / max(1, tp)      # quantity error where guessed right
    return m


def main():
    # The architecture was frozen on 22.09 (internal backlog, "Architecture freeze"): there are no
    # architecture flags any more, and the training defaults are those with which
    # `mega_p8` was trained (game b7): cosine + EMA 0.999, market weight 0.6, seed 0. A forgotten
    # flag must not silently give a different net.
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=0, help="0 = all replays")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.0,
                    help="0 = scale from the batch: 3e-4 * sqrt(batch/64)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", type=int, default=50)
    ap.add_argument("-j", type=int, default=13, help="workers for building the cache")
    ap.add_argument("--run", default="v7")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--teacher", default="only", choices=("only", "both"))
    ap.add_argument("--teacher-sub", type=int, default=0,
                    help="whose corpus to take; 0 = FARMCORE_TEACHER or 56263693")
    # By default WITHOUT class weights. With exact quantities the tail is sparse,
    # and (1/f)^0.5 gave a spread of up to 606x: a class with ONE observation weighed 2.79
    # against 0.005 for "do not sell". That is optimizing noise, not fighting
    # imbalance -- and it is what made the heads pull the trigger twice as often as
    # the teacher, which we then cured with tuned thresholds.
    # If weights are needed after all, the right way to remove their skew is to
    # subtract log(w_c) from the logits at inference (logit adjustment,
    # Menon et al., ICLR 2021), not to tune a threshold.
    ap.add_argument("--eff-coef", type=float, default=1.0,
                    help="weight of the effect map. 1.0 is THREE times more than for actions "
                         "(op 0.3, target 0.3), even though the decoder does not read this "
                         "head's `pointer` at all.")
    ap.add_argument("--op-coef", type=float, default=0.3)
    ap.add_argument("--tgt-coef", type=float, default=0.3)
    ap.add_argument("--mkt-coef", type=float, default=0.6,
                    help="weight of the market group -- the SUM of seven sub-losses (sell, buy, "
                         "seed, animal, hire, land, qty). 0.6 instead of 0.3 was taken in "
                         "P7 (b2: no worse, against v47 19/48)")
    ap.add_argument("--econ-coef", type=float, default=0.5,
                    help="weight of the economy delta. The head is NOT read at inference")
    ap.add_argument("--value-coef", type=float, default=0.5,
                    help="weight of the value tower (W/D/L + HL). Not read at "
                         "inference (play.py calls the net with want_value=False), "
                         "needed under PPO. It does not affect the policy at all: the tower "
                         "input is detached, gradient clipping is per group")
    ap.add_argument("--hl-coef", type=float, default=1.0,
                    help="weight of the HL-Gauss head inside the value group. 1.0 = "
                         "outcome and gradation weigh equally")
    ap.add_argument("--value-sigma", type=float, default=VAL.RL_SIGMA,
                    help="scale of the bank difference in the critic target, $. 25,000 from a "
                         "measurement on 200 teacher games: median |Δ| $5,244, "
                         "std $23,373, 92%% of games fit in $30k")
    ap.add_argument("--value-gamma", type=float, default=1.0,
                    help="discount of the critic target by the number of decisions to the end. 1.0 "
                         "-- the horizon is fixed and the reward is terminal, the discount "
                         "merely multiplies the target by a function of time")
    ap.add_argument("--value-only", action="store_true",
                    help="train ONLY the value tower: the whole policy "
                         "is frozen. Critic warm-up before PPO (R1 item 4) and "
                         "the only way to compare critics on exactly the same "
                         "features")
    ap.add_argument("--value-dense", type=int, default=0, metavar="K",
                    help="HL head target is the margin change over K decisions instead of "
                         "the final bank difference (0 = as before). Removes the "
                         "label deficit: at γ = 1 the final target is CONSTANT "
                         "within a game, i.e. N games = N labels, and the tower "
                         "memorizes games (measurement 23.09: train AUC 1.000 vs "
                         "0.518 on unseen). The W/D/L head stays on the outcome. "
                         "⚠ A different scale: the change over 24 decisions has median "
                         "$379, std $1,104 against $3,199 for the final difference, "
                         "so --value-sigma must be set for it")
    ap.add_argument("--corpus-dir", default="",
                    help="take the corpus NOT from teacher replays but from this folder "
                         "(own self-play games in the Kaggle replay format, written "
                         "by an internal self-play corpus script). Only together with "
                         "--value-only: the policy must not be learned from own games, "
                         "it would lock onto its own mistakes (project rule), while "
                         "the critic on the contrary REQUIRES them -- on the teacher's corpus "
                         "it is anti-predictive in self-play (AUC 0.518 vs "
                         "0.764 for the plain bank difference, 48 games 23.09)")
    ap.add_argument("--win-w", type=float, default=1.5,
                    help="how many times more a turn from a WON game weighs. "
                         "1.0 = the outcome is not taken into account. ⚠ The constant was never "
                         "measured in money: 1.5 came from the first version")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="strength of logit adjustment: 1 = balanced error "
                         "(Menon et al.), 0 = plain MAP. One number for the whole "
                         "model instead of six tuned thresholds")
    # patience=0 because the measurement scripts take the last epoch, and early
    # stopping would silently compare arms at different epochs.
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of net initialization and batch order (P0)")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA of weights, decay per step (P0, taken; 0 = off). The "
                         "checkpoint's `model` holds the EMA weights -- they are played with and "
                         "validation is computed on them -- while the raw ones go to `raw`")
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--wd", type=float, default=0.01, help="weight decay (AdamW and Muon)")
    ap.add_argument("--dropout", type=float, default=0.0, help="trunk dropout (ModelConfig.dropout)")
    ap.add_argument("--ft-subs", default="",
                    help="fine-tune on a team: its submissions separated by commas (from the shared FARMCORE_TEACHER cache)")
    ap.add_argument("--chunk-games", type=int, default=0,
                    help="lazy corpus: an epoch in chunks of N games (a chunk on the card, the next one prepared in RAM)")
    ap.add_argument("--opt", default="adamw", choices=["adamw", "muon"],
                    help="muon -- torch.optim.Muon for 2D matrices (except embeddings), AdamW for the rest")
    ap.add_argument("--styles-json", default="",
                    help="JSON {submission: team}: condition on the style-team (experts)")
    ap.add_argument("--style-drop", type=float, default=0.1,
                    help="share of rows with the \"common\" style under --styles-json")
    ap.add_argument("--init", default="",
                    help="checkpoint for a warm start: of the same architecture, "
                         "loading is strict")
    ap.add_argument("--width-mult", type=int, default=1, help="model width ×N (ModelConfig.scaled)")
    ap.add_argument("--depth-mult", type=int, default=1, help="model depth ×N (ModelConfig.scaled)")
    ap.add_argument("--set-targets", action="store_true",
                    help="targets of walking units as a set (Hungarian matching in the loss)")
    ap.add_argument("--mkt-order", default="", help="execution order of the teacher's market blocks, e.g. sell,buy,animal,land,seed,hire; travels in the checkpoint")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--aug-shed", type=float, default=0.0,
                    help="share of rows with a consistent upward shed stock shift (aug_shed); 0 = off")
    ap.add_argument("--aug-shed-max", type=int, default=40, help="δ ~ U{1..K} units per good")
    ap.add_argument("--aug-shed-other", type=float, default=0.3,
                    help="shift probability for a good other than wheat")
    ap.add_argument("--save-from", type=int, default=0,
                    help="write eN.pt only from this epoch (and no best_val.pt if > 0)")
    ap.add_argument("--aug-units", type=float, default=0.0,
                    help="share of rows with unit positions from another row (aug_units); 0 = off")
    ap.add_argument("--qty-exec", action="store_true",
                    help="PICKUP/PLACE quantity label is the executable (min with the stock at the start "
                         "of the turn), not the teacher's order (_qty_exec); off = as before")
    ap.add_argument("--refit-all", action="store_true",
                    help="train on all games, including the val fold (final refit)")
    ap.add_argument("--amp", default="bf16", choices=("off", "bf16", "fp16"))
    ap.add_argument("--gpu-corpus", type=int, default=1,
                    help="keep the corpus in GPU memory (1) or in RAM (0)")
    a = ap.parse_args()

    if a.teacher_sub and a.teacher_sub != TEACHER_SUB:
        raise SystemExit(
            f"--teacher-sub {a.teacher_sub} was set after import: the corpus and the cache fingerprint "
            f"are fixed at module level. Run it like this:\n"
            f"  FARMCORE_TEACHER={a.teacher_sub} python3 experiments/net/bc_train.py ...")
    print(f"teachers: {', '.join(str(x) for x in TEACHER_SUBS)}", flush=True)
    dev = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available())
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    teacher_only = a.teacher == "only"
    for _d in REPLAY_DIRS:
        check_not_own(_d)
    # `OWN_SUBS` is above the labeling marker (an edit there changes the cache fingerprint) and
    # ends at v18; submissions after it are here (project notes, the submissions table).
    _own_late = {56428375, 56457517, 56460656, 56460657, 56467528, 56483194,
                 56545047, 56553266, 56628198, 56634494}
    if set(TEACHER_SUBS) & _own_late:
        raise SystemExit(f"refusal: {sorted(set(TEACHER_SUBS) & _own_late)} are our "
                         "own submissions, we do not learn from our own replays")
    ver = None if a.no_cache else feature_version()
    if ver and not teacher_only:
        # The sides mode is not in the fingerprint (it is above the marker, an edit would change the
        # version of all caches). Both sides go to their own folder, otherwise an `only` run
        # would silently pick up an npz with the opponent's turns (review 30.09, F3).
        ver += "-both"
    if a.corpus_dir:
        # Own games: side 0 is labeled by `_sides` (the episode is not in TEACHER_IDX ->
        # (0,)), and it is played by exactly our net in `selfplay_corpus`. There is no need
        # for a second labeling.
        if not a.value_only:
            raise SystemExit("--corpus-dir only with --value-only: the critic is learned on own "
                             "games, but NOT the policy")
        files = sorted(glob.glob(os.path.join(a.corpus_dir, "*.json")))
        if not files:
            raise SystemExit(f"in {a.corpus_dir} no games found")
        print(f"corpus of OWN games: {a.corpus_dir}, {len(files)} files.", flush=True)
        # ⚠ The fold by id remainder mod 1000 does NOT fit here: it was written for Kaggle episode ids,
        # while own games are numbered consecutively (`900000 + seed`), and the remainder
        # turns out to be a function of the seed. On 150 games from seeds 7000+ this gave
        # train 50 / val 100 -- i.e. nothing to train on, and an epoch takes 6 seconds.
        # Here the fold is positional: every k-th game goes to validation. A game
        # is pinned to a fold by file name, i.e. forever -- exactly the
        # property for which the fold is computed rather than chosen.
        k = max(2, int(round(1 / max(a.val_frac, 1e-9))))
        val_files = [f for i, f in enumerate(files) if i % k == 0]
        train_files = [f for f in files if f not in set(val_files)]
        if a.files:
            files = files[:a.files]
            val_files = [f for f in val_files if f in set(files)]
            train_files = [f for f in train_files if f in set(files)]
    else:
        files = replay_files()
        if a.files:
            files = files[:a.files]
        if teacher_only:
            # An episode without a record in episodes.json is labeled with side 0
            # (`_sides`), i.e. in half of such games the teacher's OPPONENT is cloned
            # silently. Say loudly how many such there are.
            lost = [f for f in files if os.path.basename(f)[:-5] not in TEACHER_IDX]
            if lost:
                print(f"⚠ {len(lost)} of {len(files)} games are not in the teachers' episodes.json: "
                      "side 0 is taken, i.e. at random (example: "
                      f"{os.path.basename(lost[0])})", flush=True)
        if a.ft_subs:
            global _FT_STYLES
            ft = {int(x) for x in a.ft_subs.split(",")}
            bad_ft = ft - set(TEACHER_SUBS)
            if bad_ft:
                raise SystemExit(f"--ft-subs not from FARMCORE_TEACHER: {sorted(bad_ft)}")
            _FT_STYLES = {style_of(x) for x in ft}
            files = [f for f in files
                     if any(SIDE_SUB.get((os.path.basename(f)[:-5], sd)) in ft
                            for sd in TEACHER_IDX.get(os.path.basename(f)[:-5], ()))]
            print(f"fine-tune on {sorted(ft)}: games {len(files)}", flush=True)
        train_files, val_files = split_files(files, a.val_frac)
    if a.refit_all:
        # Final refit (29.09): train on ALL games, the epoch is fixed (not by
        # validation). The val fold remains only for the log -- its numbers are inflated.
        train_files = sorted(set(train_files) | set(val_files))
        print(f"--refit-all: training on all {len(train_files)} games", flush=True)

    out_dir = os.path.join(HERE, "runs", a.run)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "metrics.jsonl"), "a")

    torch.manual_seed(a.seed)
    style_teams = []
    if a.styles_json:
        global _STYLE_MAP
        s2t = {int(k): v for k, v in json.load(open(a.styles_json)).items()}
        style_teams = sorted(set(s2t[s_] for s_ in STYLE_SUBS if s_ in s2t))
        _STYLE_MAP = torch.tensor([style_teams.index(s2t[s_]) if s_ in s2t else len(style_teams)
                                   for s_ in STYLE_SUBS] + [len(style_teams)], dtype=torch.long)
        print(f"style-teams: {len(style_teams)} ({', '.join(style_teams)}), "
              f"style-drop {a.style_drop}", flush=True)
    mcfg = ModelConfig.scaled(a.width_mult, a.depth_mult, styles=len(style_teams))
    if a.dropout:
        from dataclasses import replace as _dc_replace
        mcfg = _dc_replace(mcfg, dropout=a.dropout)
    model = FarmNet(mcfg).to(dev)
    print(f"model: width ×{a.width_mult}, depth ×{a.depth_mult} -> d_model {mcfg.d_model}, "
          f"layers {mcfg.n_layers}, heads {mcfg.n_heads}, FFN {mcfg.ffn}", flush=True)
    if a.init:
        sd = torch.load(a.init, map_location=dev)
        took = model.load_bc(sd.get("model", sd))
        print(f"warm start from {a.init}"
              + (f"; value tower initialized from BC weights: {', '.join(took)}"
                 if took else ""), flush=True)
    lr = a.lr or 3e-4 * (a.batch / 64) ** 0.5
    if a.value_only:
        # The policy is frozen COMPLETELY, not merely not optimized:
        # then autograd builds no graph for it at all (all its inputs and parameters
        # have no gradient), and the critic warm-up costs about one forward pass.
        for p_ in model.policy_parameters():
            p_.requires_grad_(False)
        opt = torch.optim.AdamW(model.value_parameters(), lr=lr, weight_decay=0.01)
        print("--value-only mode: only the value tower is learned, "
              "the policy is frozen", flush=True)
    elif a.opt == "muon":
        mu_p, ad_p = [], []
        for n_, p_ in model.named_parameters():
            if not p_.requires_grad:
                continue
            (mu_p if p_.ndim == 2 and "emb" not in n_ and ".id_" not in n_
             and not n_.startswith("id_") else ad_p).append(p_)
        opt = _MultiOpt([torch.optim.Muon(mu_p, lr=lr, weight_decay=a.wd,
                                          adjust_lr_fn="match_rms_adamw"),
                         torch.optim.AdamW(ad_p, lr=lr, weight_decay=a.wd)])
        print(f"Muon optimizer: {sum(p.numel() for p in mu_p)/1e6:.2f} M in matrices, "
              f"AdamW: {sum(p.numel() for p in ad_p)/1e6:.2f} M", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.wd)
    print(f"device {dev}, replays {len(files)} (train {len(train_files)} / "
          f"val {len(val_files)}), teacher {a.teacher}, batch {a.batch}, "
          f"lr {lr:.2e}, parameters "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M", flush=True)
    print(f"feature cache: {ver or 'off'}, code: {_git_stamp()}", flush=True)

    t0 = time.time()
    if ver:                         # without a cache labeling is in load_corpus, one by one
        build_cache(files, a.j, teacher_only, a.gamma, ver)
    print("loading corpus...", flush=True)
    on_dev = dev if (a.gpu_corpus and dev.type == "cuda") else None
    if a.chunk_games:
        # Lazy corpus (30.09): the whole train does not fit in memory. The first chunk is for
        # prior and the estimate of the number of steps; later chunks are loaded per epoch (`_batches`).
        _rs = np.random.default_rng(a.seed + 7)
        # prior and the row estimate are from the sample in RAM; epoch chunks are loaded by `_batches`
        _first = sorted(_rs.permutation(train_files)[:min(300, a.chunk_games)].tolist())
        Ctr = load_corpus(_first, ver, teacher_only, a.gamma,
                          jobs=max(1, a.j), device=None, qty_exec=a.qty_exec)
        _rows_per_game = len(Ctr["value"]) / max(1, len(_first))
        print(f"lazy corpus: {len(train_files)} games in chunks of {a.chunk_games}, "
              f"~{_rows_per_game:.0f} rows per game", flush=True)
    else:
        Ctr = load_corpus(train_files, ver, teacher_only, a.gamma,
                          jobs=max(1, a.j), device=on_dev, qty_exec=a.qty_exec)
    Cva = load_corpus(val_files, ver, teacher_only, a.gamma,
                      jobs=max(1, a.j), device=on_dev, qty_exec=a.qty_exec)
    if Cva is None:                 # happens only on toy file sets
        raise SystemExit("validation fold is empty: take more games (--files)")
    if a.value_dense:
        if on_dev is not None:
            raise SystemExit("--value-dense is incompatible with --gpu-corpus 1: "
                             "the target is computed in numpy before moving to the card")
        for C_, name in ((Ctr, "train"), (Cva, "val")):
            games, sd, p95 = densify_mdiff(C_, a.value_dense)
            print(f"dense HL target ({name}): K = {a.value_dense} decisions, "
                  f"games {games}, target std {sd:.4f}, p95 {p95:.4f} "
                  f"(in units of glob[4]; head σ {a.value_sigma})", flush=True)
    prior = logit_prior(Ctr, dev)          # prior -- from the teacher
    n_tr = (int(_rows_per_game * len(train_files)) if a.chunk_games else len(Ctr["value"]))
    # the corpus is already assembled on the device if so requested -- nothing to move
    if a.gpu_corpus and dev.type == "cuda":
        print(f"  corpus in GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB",
              flush=True)
    print(f"preparation took {(time.time()-t0)/60:.1f} min", flush=True)
    for k in COUNT_HEADS:
        v = Ctr[k]
        v = v.cpu().numpy() if torch.is_tensor(v) else v
        nz = v[v > 0]
        print(f"  {k:<7} nonzero {(v > 0).mean():.3f}, of them median "
              f"{np.median(nz) if len(nz) else 0:.0f}, 99% <= "
              f"{np.percentile(nz, 99) if len(nz) else 0:.0f}, cap {CNT.CAPS[k]}",
              flush=True)

    model.train()
    steps_per_ep = max(1, (n_tr - a.batch + 1) // a.batch)
    total_steps = max(1, steps_per_ep * a.epochs)

    def _lr_factor(st):
        warm = min(1.0, (st + 1) / max(1, a.warmup))
        if st < a.warmup:
            return warm
        frac = min(1.0, (st - a.warmup) / max(1, total_steps - a.warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * frac))
    sched = (_MultiSched([torch.optim.lr_scheduler.LambdaLR(o, _lr_factor) for o in opt.opts])
             if isinstance(opt, _MultiOpt) else torch.optim.lr_scheduler.LambdaLR(opt, _lr_factor))
    ema = None
    if a.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval()
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    step = seen = 0
    best, best_ep, since = -1.0, -1, 0
    rng = np.random.default_rng(a.seed)

    def _to_dev(C_):
        return {k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v)
                for k, v in C_.items()}

    def _batches(ep):
        """Epoch batches. Without --chunk-games -- the whole corpus, as before. With it -- games
        are shuffled and cut into chunks; a chunk on the card, the next one in the background in RAM."""
        nonlocal Ctr
        if not a.chunk_games:
            perm = rng.permutation(n_tr)
            for s_ in range(0, n_tr - a.batch + 1, a.batch):
                yield Ctr, np.sort(perm[s_:s_ + a.batch])   # sorting speeds up the gather
            return
        import concurrent.futures as _cf
        fl = rng.permutation(train_files).tolist()
        chunks = [sorted(fl[i:i + a.chunk_games]) for i in range(0, len(fl), a.chunk_games)]
        load = lambda fs: load_corpus(fs, ver, teacher_only, a.gamma, jobs=max(1, a.j),
                                      device=None, qty_exec=a.qty_exec)
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(load, chunks[0])
            for ci in range(len(chunks)):
                Ccpu = fut.result()
                if ci + 1 < len(chunks):
                    fut = ex.submit(load, chunks[ci + 1])
                Ctr = None
                torch.cuda.empty_cache()
                C_ = _to_dev(Ccpu) if dev.type == "cuda" else Ccpu
                del Ccpu
                n_c = len(C_["value"])
                print(f"  ep {ep} chunk {ci + 1}/{len(chunks)}: {len(chunks[ci])} games, "
                      f"{n_c:,} rows", flush=True)
                perm = rng.permutation(n_c)
                for s_ in range(0, n_c - a.batch + 1, a.batch):
                    yield C_, np.sort(perm[s_:s_ + a.batch])
                del C_
    amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.amp)
    use_amp = amp_dt is not None and dev.type == "cuda"
    for ep in range(a.epochs):
        t_ep = time.time()
        ep_loss = defaultdict(float)
        ep_steps = 0
        for C_cur, idx in _batches(ep):
            x, t = collate_idx(C_cur, idx, dev)
            # augmentations -- only here, on the training batch; evaluate and play are clean
            if a.aug_shed > 0:
                aug_shed(x, t, a.aug_shed, a.aug_shed_max, a.aug_shed_other)
            if a.aug_units > 0:
                aug_units(x, t, a.aug_units)
            if "style" in x and a.style_drop > 0:
                drop = torch.rand(x["style"].shape, device=x["style"].device) < a.style_drop
                x["style"] = torch.where(drop, torch.full_like(x["style"], mcfg.styles), x["style"])
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                out = model(**x, legal=t["legal_now"])
                z0 = 0.0 if a.value_only else 1.0
                loss, parts = loss_fn(out, t, prior=prior, tau=a.tau,
                                      win_w=a.win_w,
                                      eff_coef=z0 * a.eff_coef, op_coef=z0 * a.op_coef,
                                      tgt_coef=z0 * a.tgt_coef,
                                      mkt_coef=z0 * a.mkt_coef,
                                      econ_coef=z0 * a.econ_coef,
                                      value_coef=a.value_coef,
                                      val_sigma=a.value_sigma,
                                      val_gamma=a.value_gamma, hl_coef=a.hl_coef,
                                      set_targets=a.set_targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Clipping by TWO groups, not by all parameters at once. The tower input
            # is detached, so its loss does not flow into the policy -- but a common
            # norm would mix them back: a critic with a large gradient would silently
            # shrink the policy step. With separate clipping "the critic does not
            # touch the policy" holds for training too (model.policy_parameters).
            torch.nn.utils.clip_grad_norm_(model.policy_parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.value_parameters(), 1.0)
            opt.step(); sched.step()
            if ema is not None:
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),
                                         list(model.parameters()), 1.0 - a.ema)
                    for b_e, b_m in zip(ema.buffers(), model.buffers()):
                        b_e.copy_(b_m)
            step += 1; ep_steps += 1; seen += len(idx)
            for k, v in parts.items():
                ep_loss[k] += v
            if step % a.report == 0:
                el = time.time() - t_ep
                print(f"ep {ep} step {step:>6} loss {float(loss.detach()):.3f} "
                      f"(op {parts['op']:.3f} tgt {parts['tgt']:.3f} mkt {parts['mkt']:.3f} "
                      f"eff {parts['eff']:.3f} econ {parts['econ']:.3f}) "
                      f"{(ep_steps * a.batch)/el:7.0f} samples/s", flush=True)
        net = ema if ema is not None else model      # validation -- on the net that plays
        import policy as _POL        # lazily: policy imports bc_train
        m = evaluate(net, Cva, dev, val_sigma=a.value_sigma,
                     val_gamma=a.value_gamma, prior=prior,
                     beta=_POL.FAMILY["beta"])
        tr = {k: v / max(1, ep_steps) for k, v in ep_loss.items()}
        rec = {"epoch": ep, "step": step, "samples": seen,
               "minutes": (time.time() - t_ep) / 60, "train": tr, "val": m}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
        print(f"\n--- epoch {ep}: {(time.time()-t_ep)/60:.1f} min, "
              f"train op {tr['op']:.3f} tgt {tr['tgt']:.3f} eff {tr['eff']:.3f}")
        print(f"    val: op {m['op_acc']:.3f}  target {m['tgt_acc']:.3f}  "
              f"whole turn {m['turn_acc']:.3f}  effects F1 {m['eff_f1']:.3f}  "
              f"economy MAE {m['econ_mae']:.3f}")
        if "op_acc_play" in m:
            print(f"    val in the played family (β·prior): op {m['op_acc_play']:.3f}  "
                  f"target {m['tgt_acc_play']:.3f}")
        print(f"    critic: outcome AUC {m['val_auc']:.3f}  money EV "
              f"{m['val_ev']:.3f}  corr {m['val_corr']:.3f}  "
              f"(train wdl {tr.get('val_wdl', 0):.3f} hl {tr.get('val_hl', 0):.3f})")
        print("    val market (gate precision/recall, quantity error): " + "  ".join(
            f"{k} {m[k+'_prec']:.2f}/{m[k+'_rec']:.2f}/{m[k+'_mae']:.1f}"
            for k in MKT_HEADS), flush=True)
        ck = {"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep,
              "metrics": rec, "cfg": vars(a), "feature_version": ver,
              "model_cfg": asdict(mcfg), "style_teams": style_teams,
              "git": _git_stamp(),
              "prior": {k: v.cpu() for k, v in prior.items()}}
        if ema is not None:
            ck["raw"] = model.state_dict()
        if ep >= a.save_from:          # the box disk is shared and gets full (28.09, 100%)
            torch.save(ck, os.path.join(out_dir, f"e{ep}.pt"))
        score = 0.5 * (m["op_acc"] + m["tgt_acc"])
        if score > best:
            best, best_ep, since = score, ep, 0
            # The validation metric does NOT write best.pt: over v15-v22 its Spearman with
            # money is -0.69 (whole turn -0.86, target -0.81), i.e. it
            # systematically selects the worst epoch. Kept as a "did not break" cutoff
            # and as a condition for early stopping.
            if a.save_from == 0:
                torch.save(ck, os.path.join(out_dir, "best_val.pt"))
            print(f"    best by op+target: epoch {ep} ({best:.3f}) -> best_val.pt",
                  flush=True)
        else:
            since += 1
            print(f"    no improvement for {since} ep. (best {best:.3f} at epoch {best_ep})",
                  flush=True)
            if a.patience and since >= a.patience:
                print("    early stopping", flush=True)
                break
    print(f"\ntotal {(time.time()-t0)/60:.1f} min, checkpoints in {out_dir}")
    print("⚠ best.pt NOT written: the epoch is chosen by money, not by validation.",
          flush=True)


if __name__ == "__main__":
    main()