"""Mask of allowed unit actions -- the canonical contract of the network.

It used to live in `experiments/bc/bc.py` (the boosting prototype) and was imported
from there. Moved here: the network has its own class set (27 vs 23), and keeping
the contract in a foreign module is a direct path to training/play desync.

Huang & Ontanon: the importance of masking grows with the share of invalid actions,
and here almost everything is invalid.

Masks are of two kinds (R8, 22.09), and under PPO they behave differently:

  rule     `legal_ops` -- what the engine will not execute (silent no-op:
           WATER of a watered plant, FEED of a fed animal, PICKUP from an empty shed, PLACE into a
           full one...) or what is strictly dominated by idling (FERTILIZE of a plant
           already covered until day+2 consumes the fertilizer and changes nothing). Such a
           mask stays HARD under PPO as well: sampling a no-op is noise in the
           gradient, not exploration (arXiv:2006.14171). On teacher
           56263693 (419 games, 1.76M unit actions) 0.08% of its actions
           fall under the rules, and all are engine no-ops: the state
           changed before the unit arrived (HARVEST 364, PICKUP 661,
           WATER 172, ...), FERTILIZE on a covered plant -- 8 of 46,884.
  CARE on a hungry animal that this unit will not feed -- the only prohibition
           that is absent from the engine. It stays a HARD mask, the same for
           labeling, BC play and PPO: removal was measured twice and both times cost
           money (-$24,753 on the previous architecture 21.09; -$1,538 head-to-head against
           mega_p8 on the current one, b8/b9 22.09, vs +$70 with the mask). The split
           into "rules" and "habits" from R8 is cancelled by this measurement.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    CROPS, ANIMALS, PRODUCTS,
)
import envcfg  # noqa: E402

CROP_LIST = list(CROPS)
ANIMAL_LIST = list(ANIMALS)
# goods worth taking from the shed besides wheat and fertilizer


def structure_free(tile, animal):
    """An animal will stand on this tile rather than fall into the shed (env_ref.py:392).

    One function for the decoder (`policy.run_turn`: quantity is not asked),
    labeling (`bc_train`: no quantity label is set) and the shed projection
    (`bc_train.shed_at_market`)."""
    return (isinstance(tile, dict) and tile.get("kind") == ANIMALS[animal]["structure"]
            and "animal" not in tile)


def legal_ops(obs, p, k, pos, inv_k, seeds, shed):
    farm = obs["farms"][p]
    board = len(farm["tiles"])
    half = board // 2
    x, y = pos
    tile = farm["tiles"][y][x]
    is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT"
    is_animal = isinstance(tile, dict) and tile.get("animal")
    shed_adj = (x, y) in envcfg.shed_tiles(board)
    inv_k = inv_k or {}
    # directions are not included here: the step is computed from the target by the X-then-Y rule
    # (internal movement-factorization analysis), the mask is needed only for jobs
    ok = {"PASS"}
    if is_plant and not tile["watered_today"]:
        ok.add("WATER")
    if isinstance(tile, dict) and tile.get("yield_units", 0) > 0:
        if is_animal:
            ok.add("HARVEST")
        elif is_plant and obs["day"] - tile["planted_day"] >= CROPS[tile["crop"]]["first_yield_day"]:
            ok.add("HARVEST")
    if is_plant and inv_k.get("FERTILIZER", 0) > 0:
        # The engine deducts fertilizer BEFORE checking the effect and sets
        # `fertilized_until_day = max(current, day+2)` (env_ref.py:478-481).
        # A plant already covered until day+2 or later consumes the item and does not change
        # state -- a pure loss of ~$100. The remaining-time feature is in the input
        # (features.py:62, tile[26]), but the mask did not look at it.
        if tile.get("fertilized_until_day", -1) < obs["day"] + 2:
            ok.add("FERTILIZE")
    if tile is None:
        ok.update(["BUILD_COOP", "BUILD_PASTURE"])
        for c in CROP_LIST:
            if seeds.get(c, 0) > 0:
                ok.add(f"PLANT_{c}")
    if tile is not None and tile != "LOCKED" and not is_animal:
        ok.add("DIG")
    if is_animal:
        if not tile["fed_today"] and inv_k.get("WHEAT", 0) > 0:
            ok.add("FEED")
        if tile.get("fertilizer_available"):
            ok.add("COLLECT_FERTILIZER")
        if not tile["cared_today"]:
            # CARE on a hungry animal that THIS unit cannot feed is
            # closed HERE -- in labeling, in play and under PPO. By the engine
            # rules the action is legal (it checks `fed_today` at the end of the day), but
            # money confirmed the prohibition twice: removing it cost -$24,753 on the
            # previous architecture (48 games, 21.09) and -$1,538 head-to-head against
            # mega_p8 on the current one (b8/b9, 22.09; with it +$70, 25/48).
            # So this is not a "habit on a leash" but a hard mask.
            if tile["fed_today"] or inv_k.get("WHEAT", 0) > 0:
                ok.add("CARE")
    if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and not is_animal:
        for a in ANIMAL_LIST:
            if inv_k.get(a, 0) > 0 and ANIMALS[a]["structure"] == tile["kind"]:
                ok.add(f"PLACE_{a}")
    if shed_adj:
        if inv_k:
            ok.add("DROP")
        # Addressed drop: needs the item itself in hand and room in the shed. The engine clips
        # by the remainder itself (`n = min(n, inv[item])`), but with a full shed the operation
        # is not executed at all.
        if sum(shed.values()) < envcfg.SHED_CAP:
            for g in PRODUCTS:
                if inv_k.get(g, 0) > 0:
                    ok.add(f"PLACE_{g}")
            # An animal can also be put into the shed by address: `_apply_unit_action`
            # on a building mismatch FALLS THROUGH into the shed-drop branch and
            # takes any item. Previously the mask allowed `PLACE_<animal>`
            # only on a matching free building, and the cycle was
            # asymmetric: an animal can be taken from the shed, but put back
            # by address -- no, only `DROP` of the whole stack. The same hole that was fixed
            # for `PLACE item n` of a good; on teacher actions it is not visible,
            # because the teacher does not play like that.
            for a in ANIMAL_LIST:
                if inv_k.get(a, 0) > 0:
                    ok.add(f"PLACE_{a}")
        for g in list(PRODUCTS) + ANIMAL_LIST:
            if shed.get(g, 0) > 0:
                ok.add(f"PICKUP_{g}")
    return ok


