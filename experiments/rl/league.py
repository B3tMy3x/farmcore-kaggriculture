"""Opponent league: who plays against the trained policy and how that is decided.

Three categories, by quotas (the earlier project's scheme, `opponent_sampling.plan_opponents`):

  bc_anchor  the frozen BC checkpoint we started from - answers
             "did we fall apart": if it goes down against this one, the leash (R7)
             is not holding;
  latest     the latest accepted snapshot of the policy itself (mirror) - answers
             "is there growth";
  pool       PFSP by GROUPS: frozen packages with their own code (ladder
             scripts and our own earlier networks, `LeagueCfg.groups`) plus an implicit
             group `history` - accepted policy snapshots. Answers
             forgetting and the link to the ladder.

Why quotas and not one softmax over the whole pool. In an earlier project a pool member's weight was
the archetype's ladder share, and without that weight the rollout budget went to hard
but rare opponents (34.6% of games on a deck with 5.9% of the field). We have no field: our
pool holds DIFFERENT questions, not opponents of one kind, and mixing them into one
softmax would let the sampler decide which question to answer.

Why two levels inside the pool. Ladder scripts come in families: of 23
public packages scoring >= 2450 half are versions of two authors with
move agreement of 80-100%. Flat PFSP
over members would give a family a budget proportional to its number of versions. So
first a group (behaviour cluster) is chosen, then a member within it, both times
with weight `difficulty_floor + difficulty**difficulty_power` and a lower bound on
the share. A group's difficulty is the mean EMA of its members, so it does not depend on
how many of its members landed in the rollout. History snapshots are one group: they are one
policy line and are closer to each other than members of any script cluster.

Sides are balanced inside each (category, opponent) pair: the parity of the
global game number decides who moves first, exactly as in `play.py`
(`--index0`). Without this the side alone gives +$1.3k and a 58% win rate for free
(measurement protocol).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Member:
    """One pool opponent."""

    key: str                    # unique name: the difficulty EMA is kept under it
    kind: str                   # "ckpt" | "package" | "self"
    path: str = ""              # checkpoint or package main.py
    update: int = 0             # at which update it was taken (for history)
    category: str = ""
    group: str = ""             # pool group (behaviour cluster)

    def exists(self) -> bool:
        return self.kind == "self" or (bool(self.path) and os.path.exists(self.path))


def allocate(total: int, weights: List[float]) -> List[int]:
    """Integer split by quotas using the largest-remainder method.

    Hamilton, as in the earlier project: with 8 games and four equal quotas the result must be
    2/2/2/2, not "however it falls". Otherwise a category with a 25% quota in a small
    rollout shows up every other time, and its metric jumps not because of the policy.
    """
    s = sum(weights)
    if s <= 0 or total <= 0:
        return [0] * len(weights)
    exact = [total * w / s for w in weights]
    base = [int(e) for e in exact]
    rest = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: -(exact[i] - base[i]))
    for i in order[:rest]:
        base[i] += 1
    return base


def floor_mix(w: List[float], eps: float) -> List[float]:
    """Probabilities from weights with a lower bound: p' = (1 - k*eps)*p + eps.

    The bound is clipped if k*eps >= 1 (then uniform): a share cannot be
    negative, and the uniform split is the limit of "lose no one".
    """
    k = len(w)
    s = sum(w)
    if k == 0:
        return []
    p = [x / s for x in w] if s > 0 else [1.0 / k] * k
    eps = min(max(eps, 0.0), 1.0 / k)
    return [(1 - k * eps) * x + eps for x in p]


class League:
    """Pool, difficulty EMA, admission of snapshots into history."""

    HISTORY = "history"

    def __init__(self, cfg, root: str, anchor_path: str = ""):
        self.cfg = cfg
        self.root = root
        self.anchor = Member("bc_anchor", "ckpt", anchor_path, 0, "bc_anchor")
        self.latest: Optional[Member] = None
        self.history: List[Member] = []
        # Group -> member packages. Group order is as in the config.
        self.groups: Dict[str, List[Member]] = {}
        self.missing: List[str] = []
        for g, paths in (getattr(cfg, "groups", None) or {}).items():
            for p in paths:
                full = p if os.path.isabs(p) else os.path.join(root, p)
                m = Member(os.path.basename(os.path.dirname(full)), "package", full,
                           0, "pool", g)
                # WARNING: the packages are third-party public Kaggle notebook agents and are not
                # shipped; put them under `opponents/` yourself. If they are missing, the league must SAY so
                # rather than silently play without them: otherwise the run thinks it was measured against the ladder
                # while it was measured against itself.
                if m.exists():
                    self.groups.setdefault(g, []).append(m)
                else:
                    self.missing.append(p)
        keys = [m.key for ms in self.groups.values() for m in ms]
        dup = {k for k in keys if keys.count(k) > 1}
        if dup:
            # The key is the package folder name, the EMA is kept under it: two packages with the same
            # name would silently share difficulty.
            raise ValueError(f"league: duplicate package keys {sorted(dup)}")
        self.difficulty: Dict[str, float] = {}

    @property
    def frozen(self) -> List[Member]:
        """All frozen packages as a flat list (compatibility)."""
        return [m for ms in self.groups.values() for m in ms]

    # ---------------- selection ----------------

    def pool_groups(self) -> Dict[str, List[Member]]:
        """Pool groups with live members, including history if present."""
        out = {g: [m for m in ms if m.exists()] for g, ms in self.groups.items()}
        hist = [m for m in self.history if m.exists()]
        if hist:
            out[self.HISTORY] = hist
        return {g: ms for g, ms in out.items() if ms}

    def diff(self, key: str) -> float:
        return self.difficulty.get(key, self.cfg.difficulty_init)

    def weight(self, d: float) -> float:
        return self.cfg.difficulty_floor + max(d, 0.0) ** self.cfg.difficulty_power

    def group_difficulty(self, members: List[Member]) -> float:
        """Group difficulty is the mean EMA of its members (unplayed = neutral)."""
        return sum(self.diff(m.key) for m in members) / len(members)

    def group_probs(self) -> Dict[str, float]:
        gs = self.pool_groups()
        names = list(gs)
        p = floor_mix([self.weight(self.group_difficulty(gs[g])) for g in names],
                      self.cfg.pool_min_share)
        return dict(zip(names, p))

    def member_probs(self, members: List[Member]) -> List[float]:
        return floor_mix([self.weight(self.diff(m.key)) for m in members],
                         self.cfg.member_min_share)

    def expected_shares(self) -> Dict[str, float]:
        """Expected share of each member INSIDE the pool (for probes and the log)."""
        gs = self.pool_groups()
        out: Dict[str, float] = {}
        for g, pg in self.group_probs().items():
            for m, pm in zip(gs[g], self.member_probs(gs[g])):
                out[m.key] = out.get(m.key, 0.0) + pg * pm
        return out

    def plan(self, n: int, rng: random.Random, group: int = 1) -> List[Member]:
        """Opponent schedule for a rollout: a list of length n.

        WARNING: opponents are drawn in PAIRS: each drawn one plays both sides in a row
        (numbers 2k and 2k+1 - the side is set by the parity of the game number,
        `collect.play_episode`). Earlier the plan was shuffled one by one, and both
        games against one opponent could land on the same side - and the side
        alone gives +$1.3k and a 58% win rate (measurement protocol). On acceptance of
        8 games this decided admission into the league by luck of the side (review 23.09,
        finding 3). For odd n the last pair is cut off.
        """
        if group > 1:
            # Group collection (`rollout.group_size`): each drawn one plays
            # 2K games in a row - K on each side (number parity), with one
            # environment seed (`collect.group_seeds`).
            if n % (2 * group):
                raise SystemExit(f"group plan: {n} games is not divisible by "
                                 f"2*group_size = {2 * group}")
            draws = self._plan(n // (2 * group), rng)
            return [m for m in draws for _ in range(2 * group)]
        half = self._plan((n + 1) // 2, rng)
        return [m for m in half for _ in (0, 1)][:n]

    def _plan(self, n: int, rng: random.Random) -> List[Member]:
        gs = self.pool_groups()
        cats = [("bc_anchor", self.cfg.bc_anchor,
                 [self.anchor] if self.anchor.exists() else []),
                ("latest", self.cfg.latest,
                 [self.latest] if self.latest and self.latest.exists() else []),
                ("pool", self.cfg.pool, [m for ms in gs.values() for m in ms])]
        live = [(name, w, pool) for name, w, pool in cats if pool and w > 0]
        if not live:
            # Neither anchor nor packages - play mirror against ourselves. This is a
            # working mode in the first updates, not a failure.
            return [Member("self", "self", "", 0, "self") for _ in range(n)]
        counts = allocate(n, [w for _, w, _ in live])
        out: List[Member] = []
        gp = self.group_probs() if gs else {}
        for (name, _, pool), k in zip(live, counts):
            for _ in range(k):
                if name != "pool":
                    out.append(rng.choice(pool))
                    continue
                # Inside the pool - a random choice, not a Hamilton split:
                # with 4 pool games over 6 groups the split would always give the slots
                # to the same groups.
                g = rng.choices(list(gp), weights=list(gp.values()), k=1)[0]
                ms = gs[g]
                out.append(rng.choices(ms, weights=self.member_probs(ms), k=1)[0])
        rng.shuffle(out)
        return out

    # ---------------- accounting ----------------

    def record(self, key: str, loss_rate: float) -> None:
        """EMA of the share of our losses against an opponent over the past rollout.

        The first record shifts the NEUTRAL start (`difficulty_init`) by the same
        EMA instead of replacing it: one game with a loss must not immediately
        make a member the hardest in the pool.
        """
        a = self.cfg.difficulty_ema
        cur = self.difficulty.get(key, self.cfg.difficulty_init)
        self.difficulty[key] = a * cur + (1 - a) * loss_rate

    def promote(self, path: str, update: int, win_rate: float,
                honest: bool = False) -> bool:
        """Accept a policy snapshot into `latest` and history - only above the threshold.

        WARNING: `honest` is how `win_rate` was computed. `True` - by the acceptance panel
        (`collect.collect_eval`: argmax on both sides, a separate seed block).
        `False` - by the mixed rollout win rate, and this is a BIASED quantity: in the
        rollout we sample while the checkpoint opponent plays argmax, and the price of this
        asymmetry was measured - -$1 784 median, 16/48 on the very same weights
        (H1/R4). So at equal strength the rollout win rate is about 0.33, a threshold of
        0.60 demands superiority "by two thresholds", while mirror games pull
        the same number toward 0.5. The threshold is then not "strict", it is of UNKNOWN
        strictness, and because of it league difficulty grows not when the
        policy grows. Promotion by the biased win rate is allowed, but this is
        said out loud - silence is not acceptable here.

        A separate panel gives selection bias if it is also used to SELECT
        the checkpoint (the earlier project warns). Here it only admits into the league, while
        hand selection is done head-to-head by `pair.py` - these are different decisions.
        """
        ok = win_rate >= self.cfg.promote_win_rate
        if not honest:
            print(f"  WARNING: league admission by the MIXED rollout win rate "
                  f"({win_rate:.3f} vs threshold {self.cfg.promote_win_rate}): "
                  "we sample, the opponent plays argmax, bias -$1 784 / 16-48 "
                  "(H1/R4). The honest threshold is `rollout.eval_episodes > 0`",
                  flush=True)
        if not ok:
            return False
        m = Member(f"u{update:06d}", "ckpt", path, update, "pool", self.HISTORY)
        self.latest = Member("latest", "ckpt", path, update, "latest")
        self.history.append(m)
        self._prune()
        return True

    def _prune(self) -> None:
        """Keep `history_limit` snapshots, thinning by LOG-age.

        The ends are kept, the neighbour closest in log age is dropped:
        recent snapshots stay dense, old ones get sparser. Uniform
        thinning would drop exactly the segment where the policy changed
        fastest.
        """
        import math
        while len(self.history) > self.cfg.history_limit:
            ages = [math.log1p(self.history[-1].update - m.update)
                    for m in self.history]
            gaps = [(ages[i] - ages[i + 1], i + 1) for i in range(len(ages) - 2)]
            if not gaps:
                self.history.pop(0)
                continue
            self.history.pop(min(gaps)[1])

    # ---------------- state ----------------

    def state(self) -> dict:
        """Everything that must survive `--resume`: snapshots and difficulty EMA of ALL
        members, including frozen packages (group difficulty is derived from them,
        no need to store it separately)."""
        return {"latest": self.latest.__dict__ if self.latest else None,
                "history": [m.__dict__ for m in self.history],
                "difficulty": dict(self.difficulty)}

    def load_state(self, st: dict) -> None:
        if not st:
            return

        def mem(d):
            d = dict(d)
            if d.get("category") == "history":      # state before groups (23.09)

                d["category"], d["group"] = "pool", self.HISTORY
            return Member(**d)
        self.latest = Member(**st["latest"]) if st.get("latest") else None
        self.history = [mem(d) for d in st.get("history", [])]
        self.difficulty = dict(st.get("difficulty", {}))

    def dump(self, path: str) -> None:
        st = self.state()
        st["pool_groups"] = {g: [m.key for m in ms] for g, ms in self.pool_groups().items()}
        st["pool_shares"] = self.expected_shares()
        with open(path, "w") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
