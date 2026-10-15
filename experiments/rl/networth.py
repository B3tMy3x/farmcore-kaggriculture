"""Farm net worth Phi(s) - potential for the local reward (25.09).

Phi = money + stocks (shed, hands, purchased animals) at SALE REVENUE
  + seeds at purchase price
  + crops: ready units plus future ones x `FUTURE`, at the same revenue
  + animals on the field: purchase price plus accumulated product
  + land at unlock price.

Revenue from n units is not n*price but sum_{k<n} price(inv + k): the price falls with
every unit sold (`market_price`, a sale increases the market
inventory). Otherwise a shed of 60 melons would be worth $15k at a real revenue of ~$4k: for
melon, milk, wool and strawberries the price drops to $1 after ~T/2 units of glut.
So "own Phi" is computed against the market inventory at the state's moment, item by
item, stocks first, then future units.

Fitness check - `chaos_probe.py` (not shipped): the sign of dPhi 24 turns
after a lost turn holds where money does not move yet (12/12 at
turn 60, 11/12 at 200, 9/12 at 500; this was for the
n*price version; the integral version has not been probed yet).

Two entry points:
  `phi_obs(obs)` - own Phi from OUR observation (shed, hands, seeds, field,
      land). This is the rollout reward (`collect.SlimRecord`); the opponent is not
      part of it (see docstring), the goal is own money at season end.
  `phi_from_step(states)` - Phi of both players from the full environment state
      (`env.steps[i]`), for probes.
Neither goes into the network input.
"""
from __future__ import annotations

import math

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {"GOOSE": {"cost": 300, "product": "EGG"}, "COW": {"cost": 400, "product": "MILK"},
           "SHEEP": {"cost": 500, "product": "WOOL"}}
LAND_PRICES = [1000, 2000, 4000]
FUTURE = 0.7      # discount on not-yet-grown units

# Market - a copy of the environment's `MARKET_PARAMS` (`kaggriculture_env_ref.py`).
MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0
MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
BASE_PRICES = {k: v["base"] for k, v in MARKET_PARAMS.items()}


def _shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "log10":  return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory):
    p = MARKET_PARAMS[item]
    base, T = p["base"], p["T"]
    if inventory < MARKET_I0:
        amp = p["below_target"] * base / _shape(p["below_func"], T, T)
        price = base + amp * _shape(p["below_func"], MARKET_I0 - inventory, T)
    else:
        amp = p["above_target"] * base / _shape(p["above_func"], T, T)
        price = base - amp * _shape(p["above_func"], inventory - MARKET_I0, T)
    return max(PRICE_FLOOR, int(round(price)))


class _Book:
    """Sale revenue: each next unit of the item is priced after the
    previous ones. `sell(item, n)` returns the revenue and shifts the inventory."""

    def __init__(self, inventory):
        self.inv = dict(inventory)

    def sell(self, item, n):
        n = int(n)
        if n <= 0 or item not in MARKET_PARAMS:
            return 0.0
        i = self.inv.get(item, MARKET_I0)
        v = 0.0
        for k in range(n):
            v += market_price(item, i + k)
        self.inv[item] = i + n
        return v


def _plant_units(tile, day):
    """(ready, future) units of a crop."""
    cd = CROPS[tile["crop"]]
    ready = int(tile.get("yield_units", 0))
    if cd["ongoing"]:
        since = day - tile["planted_day"] - cd["first_yield_day"]
        produced = 0 if since < 0 else min(cd["max_yield"], since // max(1, cd["interval"]) + 1)
        future = cd["max_yield"] - produced
    else:
        future = cd["max_yield"] - ready
        if day > tile["planted_day"] + cd["max_yield_day"]:
            future = 0
    return ready, max(0, future)


def phi_farm(farm, private, day, book: _Book):
    """Phi of one farm. `private` is None if the shed is not visible (opponent)."""
    v = float(farm["money"])
    stock = {}
    if private:
        for inv in [private.get("shed") or {}] + list(private.get("inventories") or []):
            for item, n in inv.items():
                if n <= 0:
                    continue
                if item in ANIMALS:
                    v += n * ANIMALS[item]["cost"]
                else:
                    stock[item] = stock.get(item, 0) + int(n)
        for crop, n in (private.get("seeds") or {}).items():
            v += n * CROPS[crop]["seed"]
    ready, future = {}, {}
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                r, f = _plant_units(t, day)
                c = t["crop"]
                ready[c] = ready.get(c, 0) + r
                future[c] = future.get(c, 0) + f
            elif "animal" in t:
                a = ANIMALS[t["animal"]]
                v += a["cost"]
                ready[a["product"]] = ready.get(a["product"], 0) + int(t.get("yield_units", 0))
    # revenue: stocks first, then ready on the field, then future at a discount
    for item, n in stock.items():
        v += book.sell(item, n)
    for item, n in ready.items():
        v += book.sell(item, n)
    for item, n in future.items():
        v += FUTURE * book.sell(item, n)
    v += sum(LAND_PRICES[:len(farm.get("unlocked_quadrants", ["NW"])) - 1])
    return v


def phi_obs(obs):
    """Own Phi from our observation (farm with shed, hands, seeds, land).

    The opponent is NOT part of the reward: its shed is invisible, and the "Phi margin" over the visible
    part of the opponent is twice as noisy as own Phi (p90 |delta| per turn $921 vs
    $335 in self-play 25.09) - each of its harvests moves units from the field into an
    invisible shed and jumps in the margin. The goal of a rollout with this reward is own
    money at season end (terminal: Phi(s') = own money), the opponent acts
    on it through the prices of the shared market."""
    p = int(obs["player"])
    day = int(obs.get("day", 0))
    return phi_farm(obs["farms"][p], obs.get("private") or {}, day,
                    _Book(obs["market"]["inventory"]))


def phi_margin_obs(obs):
    """Phi margin: own Phi minus the visible part of the opponent (for probes, not a reward)."""
    p = int(obs["player"])
    day = int(obs.get("day", 0))
    inv = obs["market"]["inventory"]
    return phi_obs(obs) - phi_farm(obs["farms"][1 - p], None, day, _Book(inv))


def phi_from_step(step_states, base_prices=False):
    """Phi of both players from `env.steps[i]` (full state, for probes).
    `base_prices=True` - the old estimate n*base price (for comparison)."""
    o0 = step_states[0]["observation"]
    day = int(o0.get("day", 0))
    out = []
    for p in range(2):
        priv = step_states[p]["observation"].get("private") or {}
        if base_prices:
            book = _Book({k: MARKET_I0 for k in MARKET_PARAMS})
        else:
            book = _Book(o0["market"]["inventory"])
        out.append(phi_farm(o0["farms"][p], priv, day, book))
    return out


def phi_cost_obs(obs):
    """Phi "at cost" (26.09): money + investments AT PURCHASE PRICE.
    Seeds at seed price, animals (hands, shed, field) at purchase price, crops -
    seed price while the tile is PLANT, land at unlock price. Shed and harvest count
    as ZERO: revenue is credited only by selling. dPhi_cost per turn = d(money) +
    investments of that turn; the sum over a game telescopes into end-of-game money."""

    p = int(obs["player"]); farm = obs["farms"][p]
    priv = obs.get("private") or {}
    v = float(farm["money"])
    for inv in [priv.get("shed") or {}] + list(priv.get("inventories") or []):
        for item, n in inv.items():
            if n > 0 and item in ANIMALS:
                v += n * ANIMALS[item]["cost"]
    for crop, n in (priv.get("seeds") or {}).items():
        v += n * CROPS[crop]["seed"]
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                v += CROPS[t["crop"]]["seed"]
            elif "animal" in t:
                v += ANIMALS[t["animal"]]["cost"]
    v += sum(LAND_PRICES[:len(farm.get("unlocked_quadrants", ["NW"])) - 1])
    return v
