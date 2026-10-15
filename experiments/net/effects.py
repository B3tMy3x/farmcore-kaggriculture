"""Labeling by turn outcome: effect map and table economy.

Two views of the same state change, both invariant to which unit
executed what:

  eff[100, K]  -- multi-label map: which jobs will happen on each own tile
                 by the end of the day. The horizon is one day because `_end_of_day`
                 disbands all hands, and a plan does not live past the night.
  econ[25]     -- per-turn table delta: money, shed by product, seeds, hands,
                 market stock. Market stock is also changed by the opponent -- this is
                 the only part of the target that cannot be predicted exactly in principle.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    CROPS, PRODUCTS, ANIMALS,
)

CROP_LIST = list(CROPS)
ANIMAL_LIST = list(ANIMALS)
# Effect names match the classes of the op head (bc_train.UNIT_OPS) -- otherwise
# the "execute the plan right now" branch in play.py silently does not fire: it looks up
# the effect name in the mask of allowed jobs.
EFFECTS = (["WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
            "FEED", "CARE", "COLLECT_FERTILIZER"]
           + [f"PLACE_{a}" for a in ANIMAL_LIST]
           + [f"PLANT_{c}" for c in CROP_LIST])
EFF_ID = {e: i for i, e in enumerate(EFFECTS)}
N_EFFECTS = len(EFFECTS)
BOARD = 10
N_TILES = BOARD * BOARD
# Money and market stock (9). Shed, seeds and hand count were removed from the vector: they
# follow unambiguously from the net's own decoded turn, and the head spent
# capacity restating what is already known. Money and market stock are not removed:
# they depend on prices, the opponent's simultaneous orders and town consumption.
ECON_DIM = 1 + len(PRODUCTS)
# Scales = empirical mean |delta| per component, so the loss does not give
# the whole gradient to hiring. With the old ones (money/1000, hands/1) the model learned only hands
# (84.2% explained), and predicted money WORSE than a constant zero (-1.0%).
ECON_SCALE = np.array([200.0] + [3.0] * len(PRODUCTS), dtype=np.float32)


def effect_name(action):
    """Tile effect name of an action, or None (step, PASS, shed work)."""
    if not isinstance(action, list) or not action:
        return None
    op = action[0]
    if op == "PLANT":
        c = action[1] if len(action) > 1 else None
        return f"PLANT_{c}" if c in CROPS else None
    if op == "PLACE":
        a = action[1] if len(action) > 1 else None
        return f"PLACE_{a}" if a in ANIMALS else None
    return op if op in EFF_ID else None


def turn_effects(pos, units):
    """(tile, effect) pairs of one turn from unit positions and their actions."""
    out = []
    for k, a in enumerate(units):
        if k >= len(pos):
            break
        e = effect_name(a)
        if e is None:
            continue
        x, y = pos[k]
        out.append((y * BOARD + x, EFF_ID[e]))
    return out


def day_effect_map(per_turn, i, days):
    """Multi-label map to the end of the day, starting from turn i."""
    m = np.zeros((N_TILES, N_EFFECTS), dtype=np.float32)
    d = days[i]
    for j in range(i, len(per_turn)):
        if days[j] != d:
            break
        for tile, eff in per_turn[j]:
            m[tile, eff] = 1.0
    return m


def econ_vector(obs_a, obs_b):
    """Table delta between observations before and after the turn, normalized."""
    def row(o):
        farm = o["farms"][o["player"]] if "player" in o else o["farms"][0]
        inv = o["market"]["inventory"]
        return np.array([farm["money"]] + [inv[g] for g in PRODUCTS],
                        dtype=np.float32)
    return (row(obs_b) - row(obs_a)) / ECON_SCALE
