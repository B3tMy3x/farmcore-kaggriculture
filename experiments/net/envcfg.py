"""Environment configuration in one place: the engine reads it from `configuration`, we did not.

The problem this fixes. `agent(obs, config=None)` ignored its second argument,
and five engine quantities were hardcoded as constants in four files: shed
capacity (100) in `counts.py`, `masks.py`, `play.py`, `features.py`; the order
limit (10) in `counts.py` and three times in `play.py`; land prices as the literal
`[1000, 2000, 4000]`; board side and day length. Today everything matches the
defaults, so the bug is invisible — and it will diverge silently as soon as the
organizers change the configuration.

⚠ The quantities fall into TWO classes, and they must not be confused.

**Baked into the weights.** Board side sets the number of tile tokens and the
width of the target pointer, shed capacity and order limit set the caps of the
quantity heads (`counts.CAPS`), day length sits in feature normalizations. If they
diverge from training, the net does not "play worse" — it computes the wrong
thing. So they get a LOUD check: a mismatch is a refusal, not a warning.

**Decoder arithmetic.** Free shed space, truncating the order list, land price,
turns left until night — computed on the fly and must follow the
configuration.

Engine: `get(cfg, "shedCapacity", 100)` etc. in `kaggriculture_env_ref.py`.
Land prices are NOT controlled by the configuration — they are the module constant
`LAND_PRICES`, so we take it from the engine itself instead of copying the numbers.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    LAND_PRICES, FARM_HAND_COST_MULT,
)

# Engine defaults. Match what the weights were trained on.
SHED_CAP = 100          # configuration.shedCapacity
MAX_ORDERS = 10         # configuration.maxMarketOrdersPerTurn
BOARD = 10              # configuration.boardSize
TURNS_PER_DAY = 24      # configuration.turnsPerDay
HAND_COST_MULT = FARM_HAND_COST_MULT   # configuration.farmHandCostMult
SEASON_DAYS = 30        # episodeSteps / turnsPerDay

_BAKED = ("SHED_CAP", "MAX_ORDERS", "BOARD", "TURNS_PER_DAY")
_TRAINED = {k: globals()[k] for k in _BAKED}
_applied = False


def _get(cfg, key, default):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            v = cfg.get(key, default)
        except TypeError:
            v = getattr(cfg, key, default)
    else:
        v = getattr(cfg, key, default)
    return default if v is None else v


def configure(cfg, strict=True):
    """Read the configuration once per process. Returns the list of mismatches.

    `strict=True` — a mismatch in a baked quantity is SystemExit. Staying silent
    is not an option: the engine does not complain about a wrong action (`Invalid /
    illegal actions are silent no-ops`), and we would learn about the mismatch only
    from money, a day later and with no explanation.
    """
    global SHED_CAP, MAX_ORDERS, BOARD, TURNS_PER_DAY, HAND_COST_MULT, SEASON_DAYS, _applied
    if _applied or cfg is None:
        return []
    SHED_CAP = int(_get(cfg, "shedCapacity", SHED_CAP))
    MAX_ORDERS = max(1, int(_get(cfg, "maxMarketOrdersPerTurn", MAX_ORDERS)))
    BOARD = int(_get(cfg, "boardSize", BOARD))
    TURNS_PER_DAY = max(1, int(_get(cfg, "turnsPerDay", TURNS_PER_DAY)))
    HAND_COST_MULT = int(_get(cfg, "farmHandCostMult", HAND_COST_MULT))
    steps = int(_get(cfg, "episodeSteps", SEASON_DAYS * TURNS_PER_DAY))
    SEASON_DAYS = max(1, steps // TURNS_PER_DAY)
    _applied = True
    bad = [f"{k}: trained {v}, configuration {globals()[k]}"
           for k, v in _TRAINED.items() if globals()[k] != v]
    if bad and strict:
        raise SystemExit(
            "environment configuration differs from the one the weights were trained on:\n  "
            + "\n  ".join(bad)
            + "\nthese quantities are baked into head shapes and feature normalizations, "
              "cannot play with them — weights for the new configuration are needed")
    return bad


def land_price(n_unlocked):
    """Price of the next quadrant. Engine: LAND_PRICES[len(unlocked) - 1]."""
    i = n_unlocked - 1
    return LAND_PRICES[i] if 0 <= i < len(LAND_PRICES) else 0


def shed_tiles(board=None):
    """The four shed access tiles. Engine: `_shed_access_tiles`."""
    b = board or BOARD
    h = b // 2
    return {(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)}
