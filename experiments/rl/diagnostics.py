"""Self-play diagnostics: what shows up in a rollout but not in money or in the loss.

Pure read of a collected rollout, no network passes and no edits. The layout
follows `self_play_diagnostics.py` of an earlier project: the same quantities are computed
over the whole rollout, per game quarter, on the tail and per opponent.
Quarters because a game is not homogeneous: day 0 and day 29 are different
tasks and the per-game mean mixes them.

WARNING: here `advantage` is (true outcome - estimate at collection), NOT the PPO
advantage. The PPO advantage is GAE and may include shaping. The earlier project keeps
a warning in the docstring because they are easy to confuse and the
consequence is silent: diagnostics say "the critic is good" while it trains on something else.

What we catch. The BC decoder already has measured walking share 37.6% vs 40.0% for the
teacher and `PASS` on arrival 18.3% vs 12.3% (measured on live games). Under PPO entropy grows
and these are the first to drift: a unit changes target on the way and arrives nowhere. So
walking share and `PASS` are computed here, not "when needed".
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def _slice_stats(turns, outcome) -> Dict[str, float]:
    """Quantities over one slice of turns."""

    if not turns:
        return {}
    # WARNING: `y` is the TRUE outcome +-1, while `v` after R1 lives in the reward scale
    # (+-1 + alpha*tanh(delta/sigma)). So `value_mse` contains a constant offset up to alpha^2
    # and measures not "critic error" but "critic error plus the money
    # gradation". It is comparable only with baselines in the same row
    # (computed against the same `y`), not with `ppo/value_loss`, whose
    # target is TD(lambda) in the reward scale.
    v = np.array([t.value for t in turns], dtype=np.float64)
    y = np.array(outcome, dtype=np.float64)
    st = {
        "n": float(len(turns)),
        "value_mean": float(v.mean()),
        "value_mse": float(((v - y) ** 2).mean()),
        # Three baselines, as in the earlier project: if the critic does not beat even the zero one,
        # it is not a critic. Zero is V == 0, mean is the outcome of this same slice
        # (peeking at the answer, an upper bound for "no features").
        "base_zero_mse": float((y ** 2).mean()),
        "base_mean_mse": float(((y - y.mean()) ** 2).mean()),
        "value_sign_acc": float((np.sign(v) == np.sign(y)).mean()),
        "reward_sum": float(sum(t.reward for t in turns)),
    }
    return st


def _turn_shape(turns) -> Dict[str, float]:
    """Shape of the turns themselves from the latent: walking, jobs, gates, quantities.

    Everything is taken from the latent, not from the engine action: the latent is what the
    network chose, while the action already contains environment substitutions (rule-based step, DROP).
    """
    n = len(turns)
    if not n:
        return {}
    live = op_on = q_on = gates = counts = land = 0
    for t in turns:
        lat = t.lat
        live += int(lat["u_live"].sum())
        op_on += int(lat["op_on"].sum())
        q_on += int(lat["q_on"].sum())
        gates += int((lat["m_gate"] > 0).sum())
        counts += int(((lat["m_gate"] > 0) & (lat["m_all"] == 0)).sum())
        land += int(lat["land"] > 0)
    return {
        # a unit whose target is not its own tile is walking, so there will be no job
        "walk_share": 1.0 - (op_on / max(1, live)),
        "units_per_turn": live / n,
        "ops_per_turn": op_on / n,
        "unit_counts_per_turn": q_on / n,
        "market_gates_per_turn": gates / n,
        "market_counts_per_turn": counts / n,
        "land_per_turn": land / n,
    }


def summarize(episodes, quarters: int = 4, tail: int = 64) -> Dict[str, float]:
    """Rollout summary. -> flat dict "name -> number" for jsonl/TensorBoard."""
    live = [e for e in episodes if not e.dead and len(e)]
    if not live:
        return {"diag/episodes": 0.0}
    out: Dict[str, float] = {}
    allt, ally = [], []
    quart = [[] for _ in range(quarters)]
    quart_y = [[] for _ in range(quarters)]
    tailt, taily = [], []
    by_opp: Dict[str, List] = {}
    by_opp_y: Dict[str, List] = {}
    for ep in live:
        n = len(ep)
        allt += ep.turns
        ally += [ep.outcome] * n
        for i, t in enumerate(ep.turns):
            q = min(quarters - 1, i * quarters // n)
            quart[q].append(t)
            quart_y[q].append(ep.outcome)
        tailt += ep.turns[-tail:]
        taily += [ep.outcome] * len(ep.turns[-tail:])
        by_opp.setdefault(ep.opponent, []).extend(ep.turns)
        by_opp_y.setdefault(ep.opponent, []).extend([ep.outcome] * n)

    for k, v in _slice_stats(allt, ally).items():
        out[f"diag/all/{k}"] = v
    for k, v in _turn_shape(allt).items():
        out[f"diag/all/{k}"] = v
    for q in range(quarters):
        for k, v in _slice_stats(quart[q], quart_y[q]).items():
            out[f"diag/q{q + 1}/{k}"] = v
        for k, v in _turn_shape(quart[q]).items():
            out[f"diag/q{q + 1}/{k}"] = v
    for k, v in _slice_stats(tailt, taily).items():
        out[f"diag/tail{tail}/{k}"] = v
    for key in sorted(by_opp):
        for k, v in _slice_stats(by_opp[key], by_opp_y[key]).items():
            out[f"diag/opp:{key}/{k}"] = v
    out["diag/episodes"] = float(len(live))
    return out


def report(stats: Dict[str, float]) -> str:
    """The same numbers for a human, as one table."""
    lines = []
    keys = ["n", "value_mse", "base_zero_mse", "base_mean_mse",
            "value_sign_acc", "walk_share", "ops_per_turn",
            "market_gates_per_turn"]
    groups = sorted({k.split("/")[1] for k in stats if k.startswith("diag/")
                     and len(k.split("/")) > 2})
    head = f"{'slice':16s}" + "".join(f"{k[:11]:>13s}" for k in keys)

    lines.append(head)
    for g in groups:
        row = f"{g[:16]:16s}"
        for k in keys:
            v = stats.get(f"diag/{g}/{k}")
            row += f"{v:13.3f}" if v is not None else f"{'—':>13s}"
        lines.append(row)
    return "\n".join(lines)
