"""Extracts features from Kaggriculture replays -> one table for LightGBM.

A row = (episode, seat, game day). Features are computed ONLY from what
this player saw at that moment: both farms, market, town, own private data.
The opponent's hidden state from the replay is used only as ground truth
for checking the estimator and does not enter the features.

Feature catalog: see the project notes.

Run: ./env/bin/python experiments/net/extract_features.py [--limit N] [-j N]
Result: raw-content/features.parquet (or .csv if pyarrow is absent)
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    market_price, MARKET_I0, MARKET_PARAMS, CROPS, ANIMALS, PRODUCTS, SHOPS,
)


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def town_step(g, shops, step):
    """Town consumption exactly on THIS turn. Checked against replays: exact."""
    n = 0
    if step % 4 == 0:
        n += sum((2 if len(SHOPS[s]) == 1 else 1) for s in shops if g in SHOPS[s])
    if step % 24 == 0 and g != "FERTILIZER":
        n += 1
    return n


def _orders(act, op, g):
    n = 0
    for o in (act.get("market") or []):
        if isinstance(o, list) and len(o) >= 3 and o[0] == op and o[1] == g:
            try:
                n += int(o[2])
            except (TypeError, ValueError):
                pass
    return n


def own_flows(steps, i, p):
    """Our flows on the transition (i-1) -> i, from own data only.

    IMPORTANT: the action recorded in steps[i] explains the transition (i-1) -> i.
    We keep the total stocks (shed + unit inventories), so DROP/PICKUP
    are internal and cancel: T(i) = T(i-1) + harvested + bought - fed - sold.
    Checked against replays: median error -1.4%, 41% exact hits.
    """
    pv = (steps[i - 1][p]["observation"].get("private") or {})
    cv = (steps[i][p]["observation"].get("private") or {})

    def total(priv):
        t = defaultdict(float)
        for g, n in (priv.get("shed") or {}).items():
            if g in PRODUCTS:
                t[g] += n
        for inv in (priv.get("inventories") or []):
            for g, n in (inv or {}).items():
                if g in PRODUCTS:
                    t[g] += n
        return t

    t0, t1 = total(pv), total(cv)
    act = steps[i][p].get("action") or {}
    farm = steps[i - 1][p]["observation"]["farms"][p]
    pos = [tuple(farm["farmer"])] + [tuple(q) for q in farm["hands"]]
    units = [act.get("farmer")] + list(act.get("hands") or [])
    harvested, fed = defaultdict(float), 0.0
    for k, a in enumerate(units):
        if not isinstance(a, list) or not a or k >= len(pos):
            continue
        x, y = pos[k]
        tile = farm["tiles"][y][x]
        if not isinstance(tile, dict):
            continue
        if a[0] == "HARVEST":
            if tile.get("kind") == "PLANT":
                harvested[tile["crop"]] += tile.get("yield_units", 0)
            elif tile.get("animal"):
                harvested[ANIMALS[tile["animal"]]["product"]] += tile.get("yield_units", 0)
        elif a[0] == "COLLECT_FERTILIZER" and tile.get("fertilizer_available"):
            harvested["FERTILIZER"] += 1
        elif a[0] == "FEED" and tile.get("animal") and not tile.get("fed_today"):
            fed += 1
    sold, bought = {}, {}
    for g in PRODUCTS:
        # the engine buys only wheat and fertilizer (_process_market: everything else
        # -> order_states = None). The teacher also sends BUY_PRODUCT for the rest -- 71%
        # of its orders are phantom, and they used to pile up in mysold_/crowd_.
        b = _orders(act, "BUY_PRODUCT", g) if g in ("WHEAT", "FERTILIZER") else 0
        bought[g] = b
        sold[g] = t0[g] + harvested[g] + b - (fed if g == "WHEAT" else 0.0) - t1[g]
    return sold, bought

TURNS = 24
DAYS = 30
SHOP_NAMES = sorted(SHOPS)
# weight of a good in the shop draw (single-good shops eat twice as much)
SHOP_W = {g: sum((2 if len(v) == 1 else 1) for k, v in SHOPS.items() if g in v) / len(SHOPS)
          for g in PRODUCTS}


def dp_dq(g, inv):
    return market_price(g, inv + 1) - market_price(g, inv)


def d2p_dq2(g, inv):
    return market_price(g, inv + 2) - 2 * market_price(g, inv + 1) + market_price(g, inv)


def units_to(g, inv, target_price):
    n, i = 0, inv
    while n < 5000 and market_price(g, i) > target_price:
        i += 1
        n += 1
    return n


def pool_left(g, inv, cap=5000):
    rev, i, n = 0, inv, 0
    while n < cap:
        p = market_price(g, i)
        if p <= 1:
            break
        rev += p
        i += 1
        n += 1
    return rev, n


def town_rate(g, shops_unlocked):
    """Exact daily town demand: center 1/day + each shop 6/day."""
    base = 0.0 if g == "FERTILIZER" else 1.0
    mult = sum((2 if len(SHOPS[s]) == 1 else 1) for s in shops_unlocked if g in SHOPS[s])
    return base + 6.0 * mult


def town_ahead(g, day, shops_unlocked):
    """How much the town will eat from this day to the end: exact for open ones + expectation."""
    acc = 0.0
    n_now = len(shops_unlocked)
    known = sum((2 if len(SHOPS[s]) == 1 else 1) for s in shops_unlocked if g in SHOPS[s])
    for t in range(day, DAYS):
        n_t = min(8, t // 3)
        extra = max(0, n_t - n_now) * SHOP_W[g]
        acc += (0.0 if g == "FERTILIZER" else 1.0) + 6.0 * (known + extra)
    return acc


def crop_forecast(tile, day):
    """(good, harvest day, expected yield) for a plant under ideal care."""
    cd = CROPS[tile["crop"]]
    age = day - tile["planted_day"]
    if cd["ongoing"]:
        hd = tile["planted_day"] + cd["max_yield_day"] + 2 * (cd["max_yield"] - 1)
        return tile["crop"], hd, cd["max_yield"]
    hd = tile["planted_day"] + cd["max_yield_day"]
    left = max(0, cd["max_yield_day"] - max(age, (cd["max_yield_day"] + 1) // 2) + 1)
    return tile["crop"], max(hd, day), min(cd["max_yield"], tile["yield_units"] + left)


def animal_forecast(tile, day):
    """Expected animal output by day until the end of the season under ideal care."""
    a = ANIMALS[tile["animal"]]
    out = defaultdict(float)
    first = tile["placed_day"] + a["first_yield_day"]
    t = max(first, day)
    while t <= DAYS - 1:
        if (t - first) % a["interval"] == 0:
            out[t] += 1 + a["interval"]  # base + accumulated care bonus
        t += 1
    return a["product"], out


def farm_stats(farm, day, prefix):
    f = {}
    kinds = defaultdict(int)
    yield_standing = 0
    at_risk_plant = at_risk_animal = 0
    watered = plants = fed = animals = 0
    care_bonus = 0
    ages = []
    for row in farm["tiles"]:
        for t in row:
            if t is None:
                kinds["empty"] += 1
            elif t == "LOCKED":
                kinds["locked"] += 1
            elif t.get("kind") == "WEED":
                kinds["weed"] += 1
            elif t.get("kind") == "PLANT":
                kinds["crop_" + t["crop"]] += 1
                plants += 1
                watered += int(t["watered_today"])
                at_risk_plant += int(t["consecutive_unwatered"] >= 1)
                yield_standing += t["yield_units"]
                ages.append(day - t["planted_day"])
            elif t.get("animal"):
                kinds["animal_" + t["animal"]] += 1
                animals += 1
                fed += int(t["fed_today"])
                at_risk_animal += int(t["consecutive_unfed"] >= 1)
                yield_standing += t["yield_units"]
                care_bonus += t.get("pending_care_bonus", 0)
            else:
                kinds["struct_empty"] += 1
    for k in ("empty", "locked", "weed", "struct_empty"):
        f[f"{prefix}_{k}"] = kinds[k]
    for c in CROPS:
        f[f"{prefix}_crop_{c}"] = kinds["crop_" + c]
    for a in ANIMALS:
        f[f"{prefix}_animal_{a}"] = kinds["animal_" + a]
    f[f"{prefix}_plants"] = plants
    f[f"{prefix}_animals"] = animals
    f[f"{prefix}_tiles_unlocked"] = 100 - kinds["locked"]
    f[f"{prefix}_busy_ratio"] = (plants + animals) / max(1, 100 - kinds["locked"])
    f[f"{prefix}_watered_ratio"] = watered / max(1, plants)
    f[f"{prefix}_fed_ratio"] = fed / max(1, animals)
    f[f"{prefix}_at_risk_plant"] = at_risk_plant
    f[f"{prefix}_at_risk_animal"] = at_risk_animal
    f[f"{prefix}_yield_standing"] = yield_standing
    f[f"{prefix}_care_bonus"] = care_bonus
    f[f"{prefix}_mean_age"] = sum(ages) / len(ages) if ages else 0.0
    f[f"{prefix}_quadrants"] = len(farm["unlocked_quadrants"])
    f[f"{prefix}_hands"] = len(farm["hands"])
    f[f"{prefix}_hires_today"] = farm["hires_today"]
    f[f"{prefix}_money"] = farm["money"]
    return f


def production_forecast(farm, day):
    """Expected harvest by good over 1/3/7 days and to the end of the season."""
    horizons = {"1": day + 1, "3": day + 3, "7": day + 7, "end": DAYS - 1}
    out = {f"{g}_{h}": 0.0 for g in PRODUCTS for h in horizons}
    first_day = {g: DAYS for g in PRODUCTS}
    for row in farm["tiles"]:
        for t in row:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                g, hd, y = crop_forecast(t, day)
                first_day[g] = min(first_day[g], hd)
                for h, lim in horizons.items():
                    if hd <= lim:
                        out[f"{g}_{h}"] += y
            elif t.get("animal"):
                g, sched = animal_forecast(t, day)
                for d2, y in sched.items():
                    first_day[g] = min(first_day[g], d2)
                    for h, lim in horizons.items():
                        if d2 <= lim:
                            out[f"{g}_{h}"] += y
    return out, first_day


def process(path):
    try:
        rep = json.load(open(path))
    except Exception:
        return []
    ep_id = rep.get("id")
    rewards = rep.get("rewards") or [None, None]
    if rewards[0] is None or rewards[1] is None:
        return []
    steps = rep["steps"]
    rows = []
    for p in (0, 1):
        o = 1 - p
        sold_cum = defaultdict(float)     # our sold units (estimate from orders)
        inv_prev = None
        pi_ema = defaultdict(float)
        opp_sold_cum = defaultdict(float)
        cusum = defaultdict(float)
        cusum_k = defaultdict(float)
        money_hist = []
        for step_i in range(0, len(steps), TURNS):
            st = steps[step_i]
            obs = st[p]["observation"]
            if "farms" not in obs:
                continue
            day = obs.get("day", step_i // TURNS)
            farms, market, town = obs["farms"], obs["market"], obs["town"]
            priv = obs.get("private") or {}
            shed = priv.get("shed", {})
            shops = list(town.get("unlocked_shops", []))
            inv = market["inventory"]

            # --- flows over the past day: ours exact, the opponent's from an identity ---
            # Checked by an internal feature check: the town model is exact,
            # the opponent supply estimate has median error -1.4%.
            day_sold_me = defaultdict(float)
            day_sold_op = defaultdict(float)
            for k in range(max(1, step_i - TURNS + 1), step_i + 1):
                if k >= len(steps):
                    break
                o_prev = steps[k - 1][p]["observation"]
                if "market" not in o_prev or "market" not in steps[k][p]["observation"]:
                    continue
                shops_k = list(o_prev["town"].get("unlocked_shops", []))
                try:
                    ours, bought = own_flows(steps, k, p)
                except Exception:
                    continue
                inv_a = o_prev["market"]["inventory"]
                inv_b = steps[k][p]["observation"]["market"]["inventory"]
                for g in PRODUCTS:
                    day_sold_me[g] += max(0.0, ours[g])
                    d = inv_b[g] - inv_a[g]
                    day_sold_op[g] += (d + town_step(g, shops_k, k - 1)
                                       - max(0.0, ours[g]) + bought[g])
            for g in PRODUCTS:
                sold_cum[g] += day_sold_me[g]
                opp_sold_cum[g] = opp_sold_cum.get(g, 0.0) + max(0.0, day_sold_op[g])
                pi_ema[g] = 0.6 * pi_ema[g] + 0.4 * day_sold_op[g]

            f = {"episode": ep_id, "seat": p, "day": day, "days_left": DAYS - 1 - day}
            f.update(farm_stats(farms[p], day, "me"))
            f.update(farm_stats(farms[o], day, "op"))
            md = farms[p]["money"] - farms[o]["money"]
            f["money_diff"] = md
            f["money_ratio"] = farms[p]["money"] / max(1.0, farms[p]["money"] + farms[o]["money"])
            money_hist.append(farms[p]["money"])
            f["money_d1"] = money_hist[-1] - money_hist[-2] if len(money_hist) > 1 else 0.0
            f["money_d3"] = money_hist[-1] - money_hist[-4] if len(money_hist) > 3 else 0.0
            # log growth rate (Bell & Cover: with equal positions the objective is E[log B])
            f["log_growth_me"] = (math.log(max(1.0, money_hist[-1]) / max(1.0, money_hist[-2]))
                                  if len(money_hist) > 1 else 0.0)
            # gap normalized by remaining time (core of the Stern formula)
            dl = max(1, DAYS - 1 - day)
            f["gap_norm"] = md / math.sqrt(dl)
            f["gap_norm_rel"] = md / max(1.0, farms[p]["money"] + farms[o]["money"])
            # marginal value of a dollar by win probability, up to sigma
            f["dwp_ddollar"] = _phi(f["gap_norm"] / 20000.0) / (20000.0 * math.sqrt(dl))
            f["wp_proxy"] = _Phi(f["gap_norm"] / 20000.0)

            # --- market: levels, derivatives, pools, depth, town ---
            shed_val_price = shed_val_mr = 0.0
            for g in PRODUCTS:
                iv = inv[g]
                pr = market["prices"][g]
                d1 = dp_dq(g, iv)
                rev, n_floor = pool_left(g, iv)
                f[f"px_{g}"] = pr
                f[f"inv_dev_{g}"] = iv - MARKET_I0
                f[f"px_rel_{g}"] = pr / MARKET_PARAMS[g]["base"]
                # Kyle's lambda and depth: computed exactly for us, not estimated
                f[f"lam_{g}"] = abs(d1)
                f[f"depth_{g}"] = 1.0 / max(1e-6, abs(d1))
                f[f"elast_{g}"] = pr / max(1.0, abs(d1))
                f[f"curv_{g}"] = d2p_dq2(g, iv)
                # self-impact of a dump: the real sale price of a batch
                rev10 = sum(market_price(g, iv + k) for k in range(10))
                f[f"avg10_{g}"] = rev10 / 10.0
                f[f"drop10_{g}"] = pr - market_price(g, iv + 10)
                f[f"tofloor_{g}"] = n_floor
                f[f"tobase_{g}"] = units_to(g, iv, MARKET_PARAMS[g]["base"])
                f[f"pool_{g}"] = rev
                f[f"townrate_{g}"] = town_rate(g, shops)
                ta = town_ahead(g, day, shops)
                f[f"townahead_{g}"] = ta
                f[f"pi_{g}"] = pi_ema[g]
                f[f"oppsold_{g}"] = opp_sold_cum.get(g, 0.0)
                f[f"mysold_{g}"] = sold_cum[g]
                tot_flow = day_sold_me[g] + max(0.0, day_sold_op[g])
                f[f"crowd_{g}"] = day_sold_me[g] / max(1e-6, tot_flow)
                # CUSUM on the opponent supply: catches the switch accumulation -> dump
                cusum[g] = max(0.0, cusum[g] + day_sold_op[g] - cusum_k[g])
                cusum_k[g] = 0.9 * cusum_k[g] + 0.1 * max(0.0, day_sold_op[g])
                f[f"cusum_{g}"] = cusum[g]
                # the price the good could have if the town ate everything
                f[f"px_if_drained_{g}"] = market_price(g, int(iv - ta))
                f[f"hold_gain_{g}"] = market_price(g, int(iv - ta)) - pr
                q = shed.get(g, 0)
                f[f"shed_{g}"] = q
                shed_val_price += q * pr
                shed_val_mr += q * (rev10 / 10.0 if q >= 10 else pr)
            f["shed_total"] = sum(shed.get(g, 0) for g in PRODUCTS)
            f["shed_fill"] = f["shed_total"] / 100.0
            f["shed_val_price"] = shed_val_price
            f["shed_val_mr"] = shed_val_mr
            # theta: unsold at the end is worth zero
            f["shed_val_at_risk"] = shed_val_price / max(1.0, DAYS - day)
            f["shops_unlocked"] = len(shops)
            f["seeds_total"] = sum((priv.get("seeds") or {}).values())
            f["wheat_feed_days"] = shed.get("WHEAT", 0) / max(1.0, f["me_animals"])

            # --- production forecast of both farms and the race ---
            fc_me, first_me = production_forecast(farms[p], day)
            fc_op, first_op = production_forecast(farms[o], day)
            for g in PRODUCTS:
                for h in ("1", "3", "7", "end"):
                    f[f"fc_me_{g}_{h}"] = fc_me[f"{g}_{h}"]
                    f[f"fc_op_{g}_{h}"] = fc_op[f"{g}_{h}"]
                total = fc_me[f"{g}_end"] + fc_op[f"{g}_end"] + f[f"shed_{g}"]
                f[f"supply_vs_pool_{g}"] = total / max(1.0, f[f"tofloor_{g}"])
                f[f"race_lead_{g}"] = first_op[g] - first_me[g]
                # opponent pipeline: its future supply is almost deterministic
                pipe_op = fc_op[f"{g}_end"]
                pipe_me = fc_me[f"{g}_end"]
                opp_shed_est = max(0.0, f[f"oppsold_{g}"] * 0.0)  # stub, see below
                f[f"pipe_op_{g}"] = pipe_op
                # town recovery capacity to the end of the game (Obizhaeva-Wang)
                f[f"recover_cap_{g}"] = f[f"townahead_{g}"] - pipe_me - pipe_op
                # residual supply index: <1 -> the town will not close without us
                f[f"rsi_{g}"] = (max(0.0, f[f"inv_dev_{g}"]) + pipe_op) / max(1.0, f[f"townahead_{g}"])
                # pressure of the opponent's forced liquidation in the endgame
                f[f"forced_liq_{g}"] = pipe_op / max(1.0, f["days_left"])
                # availability per Deaton-Laroque: the correct price argument
                f[f"avail_{g}"] = (f[f"inv_dev_{g}"] + f[f"shed_{g}"] + pipe_me + pipe_op)
                f[f"vuln_proxy_{g}"] = min(fc_op[f"{g}_3"], f[f"tofloor_{g}"]) * f[f"drop10_{g}"]

            # --- lever of buying up input resources (Salop-Scheffman) ---
            for g in ("WHEAT", "FERTILIZER"):
                iv = inv[g]
                q = 50
                cost_us = sum(market_price(g, iv - k - 1) for k in range(q))
                harm_op = (market_price(g, iv - q) - market_price(g, iv)) * max(
                    1.0, f["op_animals"] if g == "WHEAT" else f["op_plants"])
                f[f"buyout_cost_{g}"] = cost_us
                f[f"buyout_leverage_{g}"] = harm_op / max(1.0, cost_us)

            # --- resources ---
            f["tile_days_left"] = f["me_tiles_unlocked"] * f["days_left"]
            f["action_supply"] = TURNS * (1 + f["me_hands"])
            f["action_demand_est"] = 1.6 * f["me_plants"] + 3.4 * f["me_animals"]
            f["action_slack"] = f["action_supply"] - f["action_demand_est"]

            # --- targets ---
            f["margin"] = rewards[p] - rewards[o]
            f["win"] = int(rewards[p] > rewards[o])
            f["final_money"] = rewards[p]
            # --- ground truth for checking the estimator (NOT a feature) ---
            opp_priv = steps[step_i][o].get("observation", {}).get("private") or {}
            f["_true_opp_shed_total"] = sum((opp_priv.get("shed") or {}).values())
            rows.append(f)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    # default is teacher only: raw-content/replays/ may contain our
    # own episodes, and they must not be mixed into the training data
    ap.add_argument("--glob", default="raw-content/replays/sub_56263693/*.json")
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(ROOT, args.glob)))
             if not f.endswith("episodes.json")]
    if args.limit:
        files = files[:args.limit]
    print(f"replays: {len(files)}, workers: {args.j}", flush=True)

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.j) as ex:
        for r in ex.map(process, files, chunksize=1):
            rows.extend(r)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(files)}  rows: {len(rows)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "raw-content", "features.parquet")
    try:
        df.to_parquet(out)
    except Exception:
        out = out.replace(".parquet", ".csv")
        df.to_csv(out, index=False)
    print(f"done: {df.shape[0]} rows x {df.shape[1]} columns -> {out}")
    print(f"games: {df['episode'].nunique()}, wins: {df['win'].mean():.3f}")


if __name__ == "__main__":
    main()
