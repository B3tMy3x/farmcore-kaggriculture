"""Kaggriculture observation -> token tensors for FarmNet.

Five token types, as in orbit-final: own tiles, opponent tiles, units, goods,
global. Dimensions are fixed in model.py and checked here with asserts.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    CROPS, ANIMALS, PRODUCTS, SHOPS, MARKET_PARAMS, MARKET_I0, market_price,
    LAND_PRICES, FARM_HAND_COST_MULT, _fib,
)
from model import (  # noqa: E402
    TILE_FEAT_DIM, UNIT_FEAT_DIM, UNIT_BASE_DIM, PRODUCT_FEAT_DIM, GLOBAL_FEAT_DIM,
    CROP_FEAT_DIM, ANIMAL_FEAT_DIM, VCRIT_FEAT_DIM, MAX_UNITS, U_FX, U_FY,
)

CROP_LIST = list(CROPS)
ANIMAL_LIST = list(ANIMALS)
# The engine has no cap on the number of hands (`_do_hire`), the hire price is mult * fib(n)
# per day: 15 hires cost $1,596, 20 about $17k. The teacher's maximum is exactly 15,
# but that is its habit, not a rule: labor is the main scarce resource of the season,
# and locking the policy into the teacher's ceiling is not allowed (under PPO the 16th hand would become
# inexpressible). We take it with a budget margin.
import envcfg  # noqa: E402

# MAX_UNITS is imported from model.py: two numbers in two files silently drift apart,
# and this is the ceiling both for the features and for the relative-shift token layout.
# Shed capacity, day length and number of season days come from `envcfg`: the engine
# reads them from `configuration`, and keeping our own copy here is a sure way to
# diverge silently (A7).
# Share of a good in a randomly opened shop: needed to compute town demand
# FORWARD, accounting for shops not yet opened (town_ahead below).
SHOP_W = {g: sum((2 if len(v) == 1 else 1) for v in SHOPS.values() if g in v) / len(SHOPS)
          for g in PRODUCTS}

# --- depth of a product "bucket" ---
# The price falls as market stock grows and hits the $1 floor. How many MORE units the
# good will take before the floor and how much money is left in it is the answer to
# the question "is it worth entering the crop at all": wool has a $7.9k pool and 59 units,
# melon $26.5k and 158, wheat is practically inexhaustible. Previously the input had
# only `drop10` (price drop over 10 units), and it lies in sign: for strawberry it is
# $19 with a pool of $3.8k, for melon $1 with a pool of $26.5k.
#
# It cannot be computed with a loop: `encode` is called 2160 times per game, and a loop to the floor is
# up to 5000 iterations per good. The table is built once per process.
_POOL_CAP = 20000
_POOL = {}


def _pool_tables(params=None):
    # The key is by CONTENT, not by id: obs brings a new dict every turn, and
    # by id the table (up to 5000 iterations per good) would be recomputed every turn,
    # risking hitting actTimeout, and after garbage collection an id could coincide
    # with another one and return the wrong table.
    key = 0 if not params else tuple(sorted(
        (g, p["base"], p["I0"], p["T"]) for g, p in params.items()))
    if key in _POOL:
        return _POOL[key]
    tab = {}
    for g in PRODUCTS:
        cum = [0.0]
        i = 0
        while i < _POOL_CAP:
            pr = market_price(g, i, params)
            if pr <= 1.0:
                break
            cum.append(cum[-1] + pr)
            i += 1
        tab[g] = (i, cum)          # floor index and cumulative revenue up to it
    _POOL[key] = tab
    return tab


def sell_revenue(g, inv, k, params=None):
    """Revenue from selling k units at market stock `inv`, with price slippage.

    Not k*price: each next unit is cheaper, and with finite melon and
    wool buckets the difference is several-fold. Computed from the same cumulative table as
    the bucket depth, i.e. for free.
    """
    floor_i, cum = _pool_tables(params)[g]
    a = max(0, min(int(inv), floor_i))
    b = max(0, min(int(inv) + int(k), floor_i))
    return cum[b] - cum[a]


def bucket_left(g, inv, params=None):
    """(units to the floor, money in the bucket) at the current market stock `inv`."""
    floor_i, cum = _pool_tables(params)[g]
    i = max(0, min(int(inv), floor_i))
    return floor_i - i, cum[floor_i] - cum[i]


def _tile_feats(t, x, y, day, half, step=0):
    f = np.zeros(TILE_FEAT_DIM, dtype=np.float32)
    if t is None:
        f[0] = 1
    elif t == "LOCKED":
        f[1] = 1
    elif isinstance(t, dict):
        k = t.get("kind")
        if k == "WEED":
            f[2] = 1
        elif k == "PLANT":
            f[3] = 1
            cd = CROPS[t["crop"]]
            age = day - t["planted_day"]
            w0 = (cd["max_yield_day"] + 1) // 2
            # crop is a category, not a scale: the ordinal code suggested that melon is
            # "five times" carrot. The ordinal is kept (the network is free to ignore it),
            # with an honest one-hot next to it.
            f[7] = (CROP_LIST.index(t["crop"]) + 1) / 5
            f[32 + CROP_LIST.index(t["crop"])] = 1.0
            f[8] = age / 16
            f[9] = float(t["watered_today"])
            f[10] = t["consecutive_unwatered"] / 2
            f[11] = t["yield_units"] / 6
            f[12] = float(age >= cd["first_yield_day"] and t["yield_units"] > 0)
            f[13] = float(w0 <= age <= cd["max_yield_day"])
            f[14] = float(t.get("fertilized_until_day", -1) >= day)
            f[26] = max(0, t.get("fertilized_until_day", -1) - day) / 5
            # --- engine edge cases (frequencies: internal edge-case analysis) ---
            # death occurs at consecutive_unwatered >= 2 in the night recomputation,
            # i.e. "not watered and already skipped a day" = will die tonight
            f[27] = float(t["consecutive_unwatered"] >= 1 and not t["watered_today"])
            # the fertilizer bonus is credited ONLY on a watered day, otherwise it burns
            f[28] = float(t.get("fertilized_until_day", -1) >= day
                          and not t["watered_today"])
            # max_lifespan_step is set at the last harvest: a day before the end
            mls = t.get("max_lifespan_step", -1)
            # Before: float(0 <= mls - step <= 24), so for a plant that is
            # ALREADY decaying (mls < step, `_decay_plants` removes one unit
            # every two turns) the feature was 0 -- exactly as for a healthy one. The
            # "run and harvest" signal did not exist. Now the sign carries "decaying",
            # the magnitude -- how close the end is; 1.0 means "no term assigned".
            f[29] = (float(np.clip((mls - step) / 24.0, -1.0, 1.0))
                     if mls >= 0 else 1.0)
            # for ongoing crops the yield is added only on its own day of the interval
            if cd["ongoing"]:
                dsf = (day + 1) - t["planted_day"] - cd["first_yield_day"]
                f[30] = float(dsf >= 0 and cd["interval"] and dsf % cd["interval"] == 0)
        elif t.get("animal"):
            f[6] = 1
            a = t["animal"]
            f[15] = (ANIMAL_LIST.index(a) + 1) / 3
            f[37 + ANIMAL_LIST.index(a)] = 1.0
            f[16] = float(t["fed_today"])
            f[17] = float(t["cared_today"])
            f[18] = t["consecutive_unfed"] / 2
            f[19] = float(t.get("fertilizer_available", False))
            f[20] = t.get("pending_care_bonus", 0) / 6
            f[11] = t["yield_units"] / 6
            a_age = day - t.get("placed_day", day)
            f[24] = a_age / 16
            f[25] = float(a_age >= ANIMALS[a]["first_yield_day"])
            # an animal runs away at consecutive_unfed >= 2; the structure remains
            f[27] = float(t["consecutive_unfed"] >= 1 and not t["fed_today"])
            # the accumulator is full: further production is lost, it must be collected
            f[31] = float(t["yield_units"] >= ANIMALS[a]["max_held"])
        else:
            f[4 if k == "COOP" else 5] = 1
    f[21] = x / 10
    f[22] = y / 10
    f[23] = (abs(x - (half - 0.5)) + abs(y - (half - 0.5))) / 10
    return f


# Keys returned by `encode`. The list is needed by the native path: it returns
# exactly them, and a mismatch in the set must be loud, not "well, something is missing".
ENCODE_KEYS = ("tiles_me", "tiles_op", "tile_mask", "units", "unit_mask",
               "products", "glob", "crops", "animals")


# Native encoder, if the loop installed one: callable(obs, p, extra) ->
# dict with the same keys. Set by `experiments/rl/collect_lockstep.RustEnvRunner`,
# the default is None, i.e. "as before".
#
# A slot, not an argument: then `play.agent` need not be touched at all -- it
# keeps calling `features.encode(obs, p, extra=...)` as before. The same trick as with
# `play._NET`: the loop supplies the object, and the play code stays single.
NATIVE = None


def _runway(farm, extra):
    """Money after topping up today's crew to yesterday's size, after also
    tomorrow's hire of the same crew, and how many of the cheapest seeds remain."""
    crew = max(len(farm["hands"]), int((extra or {}).get("crew_prev", 0)))
    hired = int(farm.get("hires_today", 0))
    fill = sum(FARM_HAND_COST_MULT * _fib(k) for k in range(hired, crew))
    tomorrow = sum(FARM_HAND_COST_MULT * _fib(k) for k in range(crew))
    left = float(farm["money"]) - fill
    seed_min = min(c["seed"] for c in CROPS.values())
    return [float(np.clip(left / 10000, -3, 3)), float(np.clip((left - tomorrow) / 10000, -3, 3)),
            min(max(left, 0.0) / seed_min, 200.0) / 200.0]


def encode(obs, p, extra=None):
    """Returns a dict of arrays. extra -- optional precomputed quantities
    (external pressure, cumulative sales), see experiments/net/extract_features.py.

    ⚠ The Python path below is the REFERENCE, and it stays the default. The native one
    (`engines/kaggriculture_rs/src/encode.rs`) is enabled via the `NATIVE` slot,
    is checked against this code element-wise over all keys on whole games
    (internal parity check), and its source is part of
    `bc_train.feature_version` -- otherwise an edit in Rust would silently reuse the
    old feature cache.
    """
    if NATIVE is not None:
        out = NATIVE(obs, p, extra)
        if out is not None:
            miss = [k for k in ENCODE_KEYS if k not in out]
            if miss:
                raise ValueError(f"native encoder did not return keys: {miss}")
            return out
    farm = obs["farms"][p]
    opp = obs["farms"][1 - p]
    board = len(farm["tiles"])
    half = board // 2
    day, hour = obs["day"], obs["hour"]
    priv = obs.get("private") or {}
    shed = priv.get("shed") or {}
    seeds = priv.get("seeds") or {}
    invs = priv.get("inventories") or [{}]
    market = obs["market"]
    shops = list(obs["town"].get("unlocked_shops", []))
    extra = extra or {}

    step = int(obs.get("step", day * 24 + hour))
    shed_tiles_set = envcfg.shed_tiles(board)
    tiles_me = np.stack([_tile_feats(farm["tiles"][y][x], x, y, day, half, step)
                         for y in range(board) for x in range(board)])
    tiles_op = np.stack([_tile_feats(opp["tiles"][y][x], x, y, day, half, step)
                         for y in range(board) for x in range(board)])
    # Mask of LEGAL POINTER TARGETS: a closed tile is a dead end (the unit arrives and can
    # only PASS), except the four shed access cells, where shed operations
    # go before the LOCKED check. At the start 75 tiles are closed, 3 of them legal,
    # so without the mask 72 pointer columns of 101 lead nowhere -- both
    # lost turns and an exploration sink under PPO.
    tile_mask = np.array([(farm["tiles"][y][x] != "LOCKED")
                          or ((x, y) in shed_tiles_set)
                          for y in range(board) for x in range(board)], dtype=bool)

    pos = [tuple(farm["farmer"])] + [tuple(q) for q in farm["hands"]]
    units = np.zeros((MAX_UNITS, UNIT_FEAT_DIM), dtype=np.float32)
    unit_mask = np.zeros(MAX_UNITS, dtype=bool)
    shed_tiles = shed_tiles_set
    for k, (x, y) in enumerate(pos[:MAX_UNITS]):
        inv = invs[k] if k < len(invs) else {}
        t = farm["tiles"][y][x]
        units[k, :UNIT_BASE_DIM] = [float(k == 0), k / MAX_UNITS, x / 10, y / 10,
                    (abs(x - (half - 0.5)) + abs(y - (half - 0.5))) / 10,
                    float((x, y) in shed_tiles),
                    # "wheat in hands" removed: it is now also in the cargo
                    # by name below (corr = 1.0000, internal feature-health check)
                    sum(inv.values()) / 50,
                    float(sum(inv.get(a, 0) for a in ANIMAL_LIST)),
                    sum(v for g, v in inv.items()
                        if g in PRODUCTS and g != "WHEAT") / 50,
                    hour / envcfg.TURNS_PER_DAY,
                    float(inv.get("FERTILIZER", 0) > 0),
                    float(isinstance(t, dict) and t.get("kind") == "PLANT"),
                    float(isinstance(t, dict) and bool(t.get("animal"))),
                    float(inv.get("GOOSE", 0) > 0), float(inv.get("COW", 0) > 0),
                    float(inv.get("SHEEP", 0) > 0), float(t == "LOCKED"),
                    # Cargo by name. Previously only wheat was by name, and
                    # the rest was two sums: a unit with six melons ($1500) and with
                    # six carrots ($210) gave the same vector, although
                    # these are different routing and DROP decisions.
                    *[inv.get(g, 0) / 20 for g in PRODUCTS],
                    # and the cargo value at current prices -- what it is carried for
                    sum(inv.get(g, 0) * market["prices"][g]
                        for g in PRODUCTS) / 3000]
        # Past action of THIS unit (28.09): its own turn is known at decision
        # time. In training it is the record of the previous turn, in play our previous turn
        # (play._FLOW). The teacher's script holds a plan between turns; without this the net
        # decides from a single frame.
        pa = (extra or {}).get("prev_act") or {}
        ul = [pa.get("farmer")] + list(pa.get("hands") or [])
        a_ = ul[k] if k < len(ul) and isinstance(ul[k], list) and ul[k] else None
        op_ = a_[0] if a_ else None
        units[k, UNIT_BASE_DIM:] = [float(op_ == "NORTH"), float(op_ == "SOUTH"),
                                    float(op_ == "EAST"), float(op_ == "WEST"),
                                    float(op_ == "PASS"),
                                    float(op_ not in (None, "PASS", "NORTH", "SOUTH", "EAST", "WEST"))]
        unit_mask[k] = True
    # The relative shift takes the unit coordinates FROM THESE SAME features
    # (model.RelBias, columns U_FX and U_FY). Permuting the columns above would break
    # it silently -- the geometry would simply become wrong, and it would not fail
    # anywhere. The check costs two comparisons per call.
    if pos:
        assert (round(float(units[0][U_FX] * 10)), round(float(units[0][U_FY] * 10))) \
            == tuple(pos[0]), "unit coordinate columns diverged from model.U_FX/U_FY"

    def town_rate(g):
        base = 0.0 if g == "FERTILIZER" else 1.0
        return base + 6.0 * sum((2 if len(SHOPS[s]) == 1 else 1)
                                for s in shops if g in SHOPS[s])

    def town_ahead(g):
        """How much the town will eat from this day to the end of the season.

        Previously this had `town_rate(g) * (30 - day)` -- demand of ONLY the already
        opened shops, stretched over the rest of the season. But the engine opens a new
        shop every three days up to eight (`_end_of_day`, MAX_SHOP_INSTANCES), and on
        day 0 for wheat the underestimate was 17.5x: 30 vs 525.
        And the first half of the season is exactly the window when melon and
        wool are laid down, i.e. the "hold or dump" decision was made on a value
        underestimated several-fold. The formula repeats `town_ahead` in
        `experiments/net/extract_features.py`, checked on replays.
        """
        known = sum((2 if len(SHOPS[s]) == 1 else 1) for s in shops if g in SHOPS[s])
        base = 0.0 if g == "FERTILIZER" else 1.0
        acc = 0.0
        for t in range(int(day), envcfg.SEASON_DAYS):
            extra_shops = max(0, min(8, t // 3) - len(shops)) * SHOP_W[g]
            acc += base + 6.0 * (known + extra_shops)
        return acc

    # The engine computes the price from `market["params"]` if they are set, and only otherwise
    # from the global MARKET_PARAMS (kaggriculture_env_ref.py:194). Previously we
    # always took the global ones: with the default configuration they coincide, but silently
    # diverge if the organizers set overrides.
    mp = market.get("params")
    shed_used = sum(shed.values())
    shed_free = max(0, envcfg.SHED_CAP - shed_used)
    products = np.zeros((len(PRODUCTS), PRODUCT_FEAT_DIM), dtype=np.float32)
    # the same will be needed by crop and animal tokens: each crop has
    # its own product (wheat, carrot, ...), each animal its own
    # (egg, milk, wool). Compute once here, not a second time below.
    pstat = {}
    for i, g in enumerate(PRODUCTS):
        iv = market["inventory"][g]
        pr = market["prices"][g]
        ta = town_ahead(g)
        rev10 = sum(market_price(g, iv + k, mp) for k in range(10)) / 10.0
        n_left, money_left = bucket_left(g, iv, mp)
        pstat[g] = (pr, iv, ta, n_left, money_left)
        # Three dead columns removed: the "animal product" flag and the
        # "wheat/fertilizer" flag are constants per row, and the row is already distinguished by
        # `id_product` (model.py); `shed_free` was repeated nine times and once more
        # in glob.
        products[i] = [pr / 300, (iv - MARKET_I0) / 1000, pr / (mp or MARKET_PARAMS)[g]["base"],
                       shed.get(g, 0) / envcfg.SHED_CAP,
                       (pr - market_price(g, iv + 1, mp)) / 20,   # marginal price shift
                       town_rate(g) / 50, ta / 600,
                       extra.get(f"pi_{g}", 0.0) / 50,
                       extra.get(f"oppsold_{g}", 0.0) / 500,
                       extra.get(f"mysold_{g}", 0.0) / 500,
                       rev10 / 300, (pr - market_price(g, iv + 10, mp)) / 300,
                       (market_price(g, int(iv - ta), mp) - pr) / 300,
                       extra.get(f"crowd_{g}", 0.5),
                       seeds.get(g, 0) / 20,
                       float(pr <= 1),
                       min(n_left, 500) / 500.0,       # units to the price floor
                       min(money_left, 30000.0) / 30000.0]  # money left in the bucket

    n_plant = n_animal = n_empty = n_weed = n_dying = 0
    n_unfed = decaying = 0
    # breakdown by crops and animals: "how much is already occupied by this" is the quantity
    # by which one decides whether to plant more, and it is in no token
    my_crop = np.zeros(len(CROP_LIST), dtype=np.float32)
    my_anim = np.zeros(len(ANIMAL_LIST), dtype=np.float32)
    free_struct = {"COOP": 0, "PASTURE": 0}     # where an animal can be placed
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                n_empty += 1
            elif isinstance(t, dict):
                if t.get("kind") == "PLANT":
                    n_plant += 1
                    # the same criterion as the tile column f[27]
                    n_dying += int(t.get("consecutive_unwatered", 0) >= 1
                                   and not t.get("watered_today"))
                    # crop that is already melting: after max_lifespan_step the engine
                    # removes one unit every two turns (`_decay_plants`)
                    mls = t.get("max_lifespan_step", -1)
                    if mls is not None and 0 <= mls <= step:
                        decaying += t.get("yield_units", 0)
                    if t.get("crop") in CROPS:
                        my_crop[CROP_LIST.index(t["crop"])] += 1
                elif t.get("kind") == "WEED":
                    n_weed += 1
                elif t.get("animal"):
                    n_animal += 1
                    my_anim[ANIMAL_LIST.index(t["animal"])] += 1
                    n_unfed += int(not t.get("fed_today"))
                elif t.get("kind") in free_struct:
                    free_struct[t["kind"]] += 1
    op_plant = op_animal = 0
    op_crop = np.zeros(len(CROP_LIST), dtype=np.float32)
    op_anim = np.zeros(len(ANIMAL_LIST), dtype=np.float32)
    for row in opp["tiles"]:
        for t in row:
            if isinstance(t, dict):
                if t.get("kind") == "PLANT":
                    op_plant += 1
                    if t.get("crop") in CROPS:
                        op_crop[CROP_LIST.index(t["crop"])] += 1
                elif t.get("animal"):
                    op_animal += 1
                    op_anim[ANIMAL_LIST.index(t["animal"])] += 1

    # --- crop and animal tokens ---
    # Previously seeds (5 "how many to buy" decisions) and animals (3) were issued by
    # one global token through heads with 500 and 300 outputs, and in unit actions
    # the item identity was baked into the CLASS NUMBER: PLANT_MELON and
    # PLANT_WHEAT are different softmax rows that borrow nothing from each other.
    # Meanwhile BUY_SEED is 21% of the teacher's turns. Now each crop and each
    # animal has its own token with its own features, and the action becomes
    # "verb + pointer to an object" (model.py).
    money = farm["money"]
    left = max(0, envcfg.SEASON_DAYS - day)
    crops_t = np.zeros((len(CROP_LIST), CROP_FEAT_DIM), dtype=np.float32)
    for i, c in enumerate(CROP_LIST):
        cd = CROPS[c]
        pr, iv, ta, n_left, money_left = pstat[c]
        crops_t[i] = [
            cd["seed"] / 300, cd["first_yield_day"] / 16, cd["max_yield_day"] / 16,
            cd["interval"] / 4, cd["max_yield"] / 6, float(cd["ongoing"]),
            pr / 300, (iv - MARKET_I0) / 1000,
            min(n_left, 500) / 500.0, min(money_left, 30000.0) / 30000.0,
            ta / 600,
            seeds.get(c, 0) / 20, my_crop[i] / 100, op_crop[i] / 100,
            min(money / max(1, cd["seed"]), 40) / 40,
            # rough return per invested dollar: yield at the current price against the
            # seed price. Exactly what was missing for strawberry with eight
            # observations to borrow from wheat with seven thousand.
            min(cd["max_yield"] * pr / max(1, cd["seed"]), 20) / 20,
            # whether what is planted today will manage to bring at least one harvest
            min(left / max(1, cd["first_yield_day"]), 2.0) / 2.0,
            shed.get(c, 0) / 100,
        ]
    animals_t = np.zeros((len(ANIMAL_LIST), ANIMAL_FEAT_DIM), dtype=np.float32)
    for i, an in enumerate(ANIMAL_LIST):
        ad = ANIMALS[an]
        pr, iv, ta, n_left, money_left = pstat[ad["product"]]
        crowd = float(sum(inv.get(an, 0) for inv in invs))
        animals_t[i] = [
            ad["cost"] / 500, float(ad["structure"] == "COOP"),
            float(ad["structure"] == "PASTURE"),
            ad["first_yield_day"] / 16, ad["interval"] / 4, ad["max_held"] / 6,
            pr / 300, (iv - MARKET_I0) / 1000,
            min(n_left, 500) / 500.0, min(money_left, 30000.0) / 30000.0,
            ta / 600,
            shed.get(an, 0) / 5, my_anim[i] / 20,
            # free buildings of the needed kind: without them a bought animal
            # has nowhere to be placed, and $300-500 lies in the shed as dead weight
            free_struct[ad["structure"]] / 20,
            crowd / 5, min(money / max(1, ad["cost"]), 20) / 20,
            min(left / max(1, ad["first_yield_day"]), 2.0) / 2.0,
            op_anim[i] / 20,
        ]
    md = farm["money"] - opp["money"]
    glob = np.array([
        # "days left" removed: (30 - day - hour/24)/30 is an exact linear
        # combination of two neighbouring columns, corr 0.9995
        day / envcfg.SEASON_DAYS, hour / envcfg.TURNS_PER_DAY,
        farm["money"] / 50000, opp["money"] / 50000, md / 50000,
        md / (20000 * math.sqrt(max(1, 29 - day))),
        len(farm["hands"]) / MAX_UNITS, len(opp["hands"]) / MAX_UNITS,
        len(farm["unlocked_quadrants"]) / 4, len(opp["unlocked_quadrants"]) / 4,
        # `opp["hires_today"]` is not needed here: `_do_hire` increments the counter and
        # adds a hand TOGETHER, `_end_of_day` zeroes both (env_ref.py:698-708,
        # 879-882), so len(hands) == hires_today identically and this was an
        # exact copy of a neighbouring column. The same was true for our own farm.

        n_plant / 100, n_animal / 100, n_empty / 100, n_weed / 100,
        op_plant / 100, op_animal / 100,
        # sh/free sum to 100, corr = 1.0000: one is enough
        shed_used / 100,
        len(shops) / 8,
        shed.get("WHEAT", 0) / max(1.0, n_animal) / 10,
        shed.get("GOOSE", 0) / 5, shed.get("COW", 0) / 5, shed.get("SHEEP", 0) / 5,
        # The town eats in portions, not continuously: shops at `step % 4 == 0`, the center at
        # `step % 24 == 0` (_town_consume). Only the continuous hour/24 was supplied,
        # from which six periodic bursts per day cannot be recovered.
        # Only the shop tick: `step % 4 == hour % 4`, period 4 -- an MLP cannot build
        # such a structure from hour/24. And "until the center tick" is `step % 24 == hour`,
        # i.e. an EXACT function of an existing feature: zero information.
        ((4 - step % 4) % 4) / 4,
        # Shops open every three days up to eight (_end_of_day). The forward demand
        # level I already fix in products, but "when it increases" is a separate
        # decision: hold the good until opening or dump now.
        # "shops expected" removed: shops open strictly every three days,
        # so min(8, day//3) is identically equal to the number already open (corr 1.0)
        ((3 - day % 3) % 3) / 3,
        # ⚠ `remainingOverageTime` REMOVED, although the constraint is real. It is not a
        # game-state feature but the TEACHER'S CLOCK: in replays there are only 5 unique
        # values, monotonically decreasing over the season (a turn took longer as the
        # farm grew), hence corr 0.64 with the opponent's animal count -- the net
        # received a game-progress indicator unrelated to the rules.
        # And in our case the reserve stays exactly 60 of 60 in every game, so at
        # inference the feature is the constant 1.0 -- a value that in training
        # meant "the very start of the season". The model played the whole season as an early
        # stage. A classic leak with a train/serve mismatch.
        FARM_HAND_COST_MULT * _fib(farm["hires_today"]) / 1000,
        (LAND_PRICES[len(farm["unlocked_quadrants"]) - 1]
         if len(farm["unlocked_quadrants"]) - 1 < len(LAND_PRICES) else 0) / 4000,
        # --- P8, measured 22.09 (internal architecture audit, games b4, b6) ---
        # "Night shed surplus" (everything in hands is poured into the shed, whatever exceeds capacity
        # is lost) stood here as column 26 and was removed at freeze: in
        # our games it is always 0, and the P8 net did not read it.
        # plants that will die tonight (the tile has the flag, but not the sum)
        n_dying / 100,
        # feed deficit: hungry today minus wheat in the shed and in hands. FEED
        # takes one wheat from a unit's hands, two days without feed -- escape
        # (`_daily_refresh_animals`). We lose 6.7 animals per game to escape, the
        # teacher 5.1; from hidden states the sum reads worse in our games.
        float(np.clip((n_unfed - shed.get("WHEAT", 0)
                       - sum(inv.get("WHEAT", 0) for inv in invs)) / 20, -1, 1)),
        # decaying crop: we lose 5.7 units per game to melting, the teacher 1.3
        decaying / 20,
        # --- cash reserve (27.09; poverty trap of clones of new teachers) ---
        # In the morning the crew is hired anew, and Fibonacci hiring can eat all
        # the money before seeds and feed. The net saw money and the price of the NEXT hand, but
        # not "what remains after the whole crew". Models without `runway` do not read these three
        # columns (`FarmNet.n_glob`), old checkpoints play as before.
        *_runway(farm, extra),
    ], dtype=np.float32)

    assert tiles_me.shape[1] == TILE_FEAT_DIM
    assert units.shape[1] == UNIT_FEAT_DIM
    assert products.shape[1] == PRODUCT_FEAT_DIM
    assert glob.shape[0] == GLOBAL_FEAT_DIM
    assert crops_t.shape[1] == CROP_FEAT_DIM
    assert animals_t.shape[1] == ANIMAL_FEAT_DIM
    return {"tiles_me": tiles_me, "tiles_op": tiles_op, "tile_mask": tile_mask,
            "units": units, "unit_mask": unit_mask,
            "products": products, "glob": glob,
            "crops": crops_t, "animals": animals_t}


# Critic column names are here, not in a comment: the permutation importance
# is printed by them in an internal critic evaluation, and the plot label
# must not be a second, separately living list.
VCRIT_NAMES = (
    [f"opp_hold_{g}" for g in PRODUCTS]          # 9: estimate of its hands + shed
    + [f"opp_bought_{g}" for g in PRODUCTS]      # 9: its purchases over the game
    + ["opp_hold_val", "my_shed_val",            # the same stock in dollars
       "rate_op", "rate_me", "rate_gap",         # bank pace per day (EMA)
       "money_me", "money_op", "md", "md_norm",
       "day", "hour",
       "opp_fed", "opp_harv", "my_harv"]
)
assert len(VCRIT_NAMES) == VCRIT_FEAT_DIM, (len(VCRIT_NAMES), VCRIT_FEAT_DIM)


def value_extra(obs, p, crit=None):
    """Columns ONLY for the value tower (R1). `crit` is `FlowTracker.critic()`.

    Why a separate function rather than columns in `glob`. A measurement (backlog, "Do
    not do"): value features fed to the policy as a batch cost **-$16,042**
    (11 wins of 48, p = 0.00022) with the BEST validation. They are inert for
    actions (internal audit: the gain in teacher NLL is within the noise
    control) and useful only to the critic, so they have exactly one path -- into the tower.

    Everything here is computable at decision time: `crit` accumulates over pairs of live
    observations and own actions (`FlowTracker.observe`), the rest
    is taken from `obs` itself. There is nothing from the future here, and there cannot be --
    the function sees only `obs` and the accumulator.

    Some of the quantities (money, gap, day, hour) are already in `glob`, i.e. in
    `h`. The duplication is deliberate: the tower reads `h` after three mixing layers,
    where permuting money moves the decision by 0.026 against 0.132 for the opponent's
    field (N5). Exactly the numbers R1 is being made for -- so
    money is fed to the tower directly, not through someone else's pooling.
    """
    crit = crit or {}
    farm = obs["farms"][p]
    opp = obs["farms"][1 - p]
    day, hour = obs["day"], obs["hour"]
    priv = obs.get("private") or {}
    shed = priv.get("shed") or {}
    prices = obs["market"]["prices"]
    hold = crit.get("opp_hold") or {}
    bought = crit.get("opp_bought") or {}
    md = farm["money"] - opp["money"]
    v = ([min(hold.get(g, 0.0), 300.0) / 100.0 for g in PRODUCTS]
         + [min(bought.get(g, 0.0), 500.0) / 500.0 for g in PRODUCTS]
         + [
             sum(hold.get(g, 0.0) * prices[g] for g in PRODUCTS) / 50000.0,
             sum(shed.get(g, 0) * prices[g] for g in PRODUCTS) / 50000.0,
             crit.get("rate_op", 0.0) / 5000.0,
             crit.get("rate_me", 0.0) / 5000.0,
             (crit.get("rate_me", 0.0) - crit.get("rate_op", 0.0)) / 5000.0,
             farm["money"] / 50000.0, opp["money"] / 50000.0, md / 50000.0,
             # the same normalized gap as in glob[5]: "how many games
             # remain to win it back"
             md / (20000 * math.sqrt(max(1, 29 - day))),
             day / envcfg.SEASON_DAYS, hour / envcfg.TURNS_PER_DAY,
             crit.get("opp_fed", 0.0) / 200.0,
             crit.get("opp_harv", 0.0) / 500.0,
             crit.get("my_harv", 0.0) / 500.0,
         ])
    out = np.asarray(v, dtype=np.float32)
    assert out.shape[0] == VCRIT_FEAT_DIM
    return out
