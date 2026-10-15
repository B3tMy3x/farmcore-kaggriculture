//! Tile features in Rust: an exact port of `features._tile_feats`.
//!
//! Why. A profile taken from a live worker (23.09, training machine): `features.encode`
//! costs 6.03 s out of 24.37, i.e. 25% of the worker's time, and 2.78 s of those
//! six is `_tile_feats`, which Python calls 200 times per decision (a hundred
//! tiles for two farms). It is the hottest spot left after
//! the environment and the network pass moved off the critical path.
//!
//! Why Rust -> Rust and not through a flat buffer, as in a previous project. There the buffer was a
//! BRIDGE between a C++ engine and a Python encoder. Here there is no bridge:
//! the state is already here, in `model::Farm`, and the encoder reads it directly.
//! What goes out is a ready array of numbers.
//!
//! ⚠ THIS IS A SECOND IMPLEMENTATION OF ONE QUANTITY -- the class of bug that
//! bit this repo three times. So: the column order, divisions and type casts here
//! repeat Python literally, including what looks redundant (for example
//! `(CROP_LIST.index(crop) + 1) / 5` next to the one-hot of the same crop);
//! everything is checked element by element by an encode parity script, and
//! `feature_version` must hash THIS file, otherwise an edit here silently
//! reuses the old feature cache.

use crate::model::{animal_rules, crop_rules, Animal, Farm, Occupant, Plant, Tile, ANIMALS, CROPS};

/// Width of the tile features. Must match `model.TILE_FEAT_DIM`; checked
/// in Python on load (`encode_parity` prints both).
pub const TILE_FEAT_DIM: usize = 40;

#[inline]
fn crop_index(name: &str) -> usize {
    CROPS.iter().position(|c| *c == name).unwrap_or(0)
}

#[inline]
fn animal_index(name: &str) -> usize {
    ANIMALS.iter().position(|a| *a == name).unwrap_or(0)
}

/// ⚠ ALL arithmetic here is in f64, and this is not pedantry. Python divides integers by
/// integers (`age / 16`, `/ 5`, `/ 3`, `/ 10`), gets float64 and rounds to
/// float32 ONCE -- on writing into the array. Computing in f32 right away, for
/// divisors that are not powers of two, gives double rounding, and the
/// last bit sometimes diverges. So everything below is computed in f64, and
/// `put` does the single cast.
#[inline]
fn put(f: &mut [f32], i: usize, v: f64) {
    f[i] = v as f32;
}

/// Python's `np.clip(v, -1.0, 1.0)`.
#[inline]
fn clip1(v: f64) -> f64 {
    if v < -1.0 {
        -1.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

fn plant_feats(f: &mut [f32], p: &Plant, day: i64, step: i64) {
    f[3] = 1.0;
    let cd = crop_rules(&p.crop);
    let age = day - p.planted_day;
    // Python: `(max_yield_day + 1) // 2` -- floor. The values are non-negative,
    // so Rust truncation gives the same; were they negative, it would diverge.
    let w0 = (cd.max_yield_day + 1).div_euclid(2);
    let ci = crop_index(&p.crop);
    // The crop is a category, not a scale. The ordinal is left next to the one-hot
    // exactly as in Python: the net is free to ignore it.
    put(f, 7, (ci as f64 + 1.0) / 5.0);
    f[32 + ci] = 1.0;
    put(f, 8, age as f64 / 16.0);
    f[9] = if p.watered_today { 1.0 } else { 0.0 };
    put(f, 10, p.consecutive_unwatered as f64 / 2.0);
    put(f, 11, p.yield_units as f64 / 6.0);
    f[12] = if age >= cd.first_yield_day && p.yield_units > 0 {
        1.0
    } else {
        0.0
    };
    f[13] = if w0 <= age && age <= cd.max_yield_day {
        1.0
    } else {
        0.0
    };
    f[14] = if p.fertilized_until_day >= day { 1.0 } else { 0.0 };
    put(f, 26, (p.fertilized_until_day - day).max(0) as f64 / 5.0);
    // Death occurs at consecutive_unwatered >= 2 in the night recomputation:
    // "not watered and already missed a day" = will die tonight.
    f[27] = if p.consecutive_unwatered >= 1 && !p.watered_today {
        1.0
    } else {
        0.0
    };
    // The fertilizer bonus accrues ONLY on a watered day, otherwise it burns off.
    f[28] = if p.fertilized_until_day >= day && !p.watered_today {
        1.0
    } else {
        0.0
    };
    // The sign carries "decays", the magnitude -- how close the end is;
    // 1.0 means "no deadline set".
    let mls = p.max_lifespan_step;
    put(
        f,
        29,
        if mls >= 0 {
            clip1((mls - step) as f64 / 24.0)
        } else {
            1.0
        },
    );
    // For ongoing crops the yield is added only on its own day of the interval.
    if cd.ongoing {
        let dsf = (day + 1) - p.planted_day - cd.first_yield_day;
        f[30] = if dsf >= 0 && cd.interval != 0 && dsf % cd.interval == 0 {
            1.0
        } else {
            0.0
        };
    }
}

fn animal_feats(f: &mut [f32], a: &Animal, day: i64) {
    f[6] = 1.0;
    let ai = animal_index(&a.animal);
    let ar = animal_rules(&a.animal);
    put(f, 15, (ai as f64 + 1.0) / 3.0);
    f[37 + ai] = 1.0;
    f[16] = if a.fed_today { 1.0 } else { 0.0 };
    f[17] = if a.cared_today { 1.0 } else { 0.0 };
    put(f, 18, a.consecutive_unfed as f64 / 2.0);
    f[19] = if a.fertilizer_available { 1.0 } else { 0.0 };
    put(f, 20, a.pending_care_bonus as f64 / 6.0);
    put(f, 11, a.yield_units as f64 / 6.0);
    let a_age = day - a.placed_day;
    put(f, 24, a_age as f64 / 16.0);
    f[25] = if a_age >= ar.first_yield_day { 1.0 } else { 0.0 };
    // An animal runs away at consecutive_unfed >= 2; the structure remains.
    f[27] = if a.consecutive_unfed >= 1 && !a.fed_today {
        1.0
    } else {
        0.0
    };
    // The accumulator is full: further production is lost, it must be collected.
    f[31] = if a.yield_units >= ar.max_held { 1.0 } else { 0.0 };
}

/// Features of ONE tile into `f` (length `TILE_FEAT_DIM`, zeroed by the caller).
pub fn tile_feats(f: &mut [f32], t: &Tile, x: usize, y: usize, day: i64, half: f64, step: i64) {
    match t {
        // ⚠ Python distinguishes `None` (column 0) and `"LOCKED"` (column 1).
        // `Tile::Locked` carries a string, but in the observation it is always "LOCKED";
        // that this is so is checked by the parity check -- no branching is added here.
        Tile::Empty => f[0] = 1.0,
        Tile::Locked(_) => f[1] = 1.0,
        Tile::Occupied(o) => match o {
            Occupant::Weed => f[2] = 1.0,
            Occupant::Plant(p) => plant_feats(f, p, day, step),
            Occupant::Coop { animal } => match animal {
                Some(a) => animal_feats(f, a, day),
                // Python: `t.get("animal")` empty -> the `else` branch, column 4/5
                // by structure kind.
                None => f[4] = 1.0,
            },
            Occupant::Pasture { animal } => match animal {
                Some(a) => animal_feats(f, a, day),
                None => f[5] = 1.0,
            },
        },
    }
    put(f, 21, x as f64 / 10.0);
    put(f, 22, y as f64 / 10.0);
    // ⚠ Python computes this in f64 (`half` is half the board minus 0.5) and only
    // then puts it into a float32 array. Do the same: adding two f32 and
    // dividing is a different rounding.
    put(
        f,
        23,
        ((x as f64 - (half - 0.5)).abs() + (y as f64 - (half - 0.5)).abs()) / 10.0,
    );
}

/// Features of all tiles of a farm: `[board*board, TILE_FEAT_DIM]` in
/// "row by row" order, as Python walks them (`for y ... for x ...`).
pub fn farm_tiles(out: &mut [f32], farm: &Farm, day: i64, board: usize, step: i64) {
    // ⚠ Python: `half = board // 2` -- INTEGER division. For a board of 10 this is
    // 5 and would coincide with 10/2, but for an odd board would diverge silently.
    let half = (board / 2) as f64;
    for y in 0..board {
        for x in 0..board {
            let a = (y * board + x) * TILE_FEAT_DIM;
            tile_feats(
                &mut out[a..a + TILE_FEAT_DIM],
                &farm.tiles[y][x],
                x,
                y,
                day,
                half,
                step,
            );
        }
    }
}

// ===================== the rest of `features.encode` =====================
//
// ⚠ Everything below is a second implementation of `features.encode`, column for column.
// The rules are the same as for `tile_feats`: arithmetic in f64, a single
// cast in `put`, the column order literal. Checked by an encode parity
// script over ALL keys, whole games at a time.
//
// `extra` (external pressure and accumulated sales, the Python `FlowTracker`)
// is passed as an argument -- 36 numbers per call. This is NOT a bridge in the sense of the previous project: no
// state structure travels through it, only already computed quantities, and when the
// tracker moves to Rust, this same argument will be filled from inside, without changing
// a single line below.

use crate::model::{Config, Inventory, Market, MarketPricing, Town, PRODUCTS};
use serde_json::Value;
use std::sync::OnceLock;

/// Four shed access cells (`envcfg.shed_tiles`, in the engine
/// `_shed_access_tiles`): shed operations on them go before the LOCKED check.
pub fn shed_tiles_of(board: usize) -> [(usize, usize); 4] {
    let h = board / 2;
    [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]
}

pub const MAX_UNITS: usize = 24;
pub const UNIT_FEAT_DIM: usize = 27;
pub const PRODUCT_FEAT_DIM: usize = 18;
pub const GLOBAL_FEAT_DIM: usize = 29;
pub const CROP_FEAT_DIM: usize = 18;
pub const ANIMAL_FEAT_DIM: usize = 18;
pub const MARKET_I0: f64 = 10000.0;
pub const LAND_PRICES: [f64; 3] = [1000.0, 2000.0, 4000.0];
pub const FARM_HAND_COST_MULT: f64 = 1.0;
const POOL_CAP: usize = 20000;

/// Offsets of the flat output. Python parses it by this same layout
/// (`features.NATIVE_LAYOUT`), and they cannot diverge: the length is checked.
pub fn layout() -> Vec<(&'static str, usize, usize)> {
    let t = 100 * TILE_FEAT_DIM;
    let mut v = Vec::new();
    let mut o = 0usize;
    for (n, k) in [
        ("tiles_me", t),
        ("tiles_op", t),
        ("tile_mask", 100),
        ("units", MAX_UNITS * UNIT_FEAT_DIM),
        ("unit_mask", MAX_UNITS),
        ("products", PRODUCTS.len() * PRODUCT_FEAT_DIM),
        ("glob", GLOBAL_FEAT_DIM),
        ("crops", CROPS.len() * CROP_FEAT_DIM),
        ("animals", ANIMALS.len() * ANIMAL_FEAT_DIM),
    ] {
        v.push((n, o, k));
        o += k;
    }
    v
}

pub fn flat_len() -> usize {
    layout().iter().map(|(_, _, k)| k).sum()
}

fn fib(n: i64) -> f64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n.max(0) {
        let t = a + b;
        a = b;
        b = t;
    }
    a as f64
}

/// `features._pool_tables`: the price-floor index and the accumulated revenue up to it.
/// The table is built once per process -- the loop to the floor is up to 5 000 iterations
/// per good, and `encode` is called thousands of times per game.
fn pool_tables(pricing: &MarketPricing) -> &'static Vec<(usize, Vec<f64>)> {
    static T: OnceLock<Vec<(usize, Vec<f64>)>> = OnceLock::new();
    T.get_or_init(|| {
        PRODUCTS
            .iter()
            .enumerate()
            .map(|(gi, _)| {
                let mut cum = vec![0.0f64];
                let mut i = 0usize;
                while i < POOL_CAP {
                    let pr = pricing.price_at(gi, i as f64);
                    if pr <= 1.0 {
                        break;
                    }
                    cum.push(cum[cum.len() - 1] + pr);
                    i += 1;
                }
                (i, cum)
            })
            .collect()
    })
}

/// `features.bucket_left` -> (units to the floor, money in the bucket).
fn bucket_left(pricing: &MarketPricing, gi: usize, inv: f64) -> (f64, f64) {
    let (floor_i, cum) = &pool_tables(pricing)[gi];
    // Python's `int()` truncates TOWARD ZERO, not down.
    let i = (inv.trunc() as i64).clamp(0, *floor_i as i64) as usize;
    ((*floor_i - i) as f64, cum[*floor_i] - cum[i])
}

fn shop_items(shops_def: &Value, name: &str) -> Vec<String> {
    shops_def[name]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// `features.SHOP_W`: the share of a good in a randomly opened shop.
fn shop_w(shops_def: &Value, g: &str) -> f64 {
    let obj = shops_def.as_object().unwrap();
    let n = obj.len() as f64;
    let s: f64 = obj
        .iter()
        .map(|(_, v)| {
            let items = v.as_array().unwrap();
            if items.iter().any(|x| x.as_str() == Some(g)) {
                if items.len() == 1 {
                    2.0
                } else {
                    1.0
                }
            } else {
                0.0
            }
        })
        .sum();
    s / n
}

fn inv_get(inv: &Inventory, k: &str) -> f64 {
    *inv.get(k).unwrap_or(&0) as f64
}

pub struct Extra<'a> {
    /// Nine numbers each in `PRODUCTS` order: pi_, oppsold_, mysold_, crowd_.
    pub pi: &'a [f64],
    pub oppsold: &'a [f64],
    pub mysold: &'a [f64],
    pub crowd: &'a [f64],
}

#[allow(clippy::too_many_arguments)]
pub fn encode_all(
    out: &mut [f32],
    cfg: &Config,
    farms: &[crate::model::Farm; 2],
    privates: &[crate::model::Private; 2],
    market: &Market,
    pricing: &MarketPricing,
    town: &Town,
    step: i64,
    player: usize,
    ex: &Extra<'_>,
    shed_tiles: &[(usize, usize)],
    shops_def: &Value,
) {
    let board = cfg.board;
    let turns = cfg.turns.max(1);
    let day = step / turns;
    let hour = step % turns;
    let season_days = (cfg.steps / turns).max(1);
    let shed_cap = cfg.cap as f64;
    let farm = &farms[player];
    let opp = &farms[1 - player];
    let priv_ = &privates[player];
    let lay = layout();
    let at = |n: &str| lay.iter().find(|(m, _, _)| *m == n).map(|(_, o, _)| *o).unwrap();

    farm_tiles(&mut out[at("tiles_me")..], farm, day, board, step);
    farm_tiles(&mut out[at("tiles_op")..], opp, day, board, step);

    // --- mask of legal pointer targets ---
    let tm = at("tile_mask");
    for y in 0..board {
        for x in 0..board {
            let locked = matches!(farm.tiles[y][x], Tile::Locked(_));
            let shed = shed_tiles.iter().any(|(sx, sy)| *sx == x && *sy == y);
            out[tm + y * board + x] = if !locked || shed { 1.0 } else { 0.0 };
        }
    }

    // --- market prices and stock: needed by units, goods and tokens ---
    let price: Vec<f64> = PRODUCTS
        .iter()
        .map(|g| market.prices.get(*g).map(|n| n.as_f64().unwrap_or(0.0)).unwrap_or(0.0))
        .collect();
    let invm: Vec<f64> = PRODUCTS
        .iter()
        .map(|g| market.inventory.get(*g).map(|n| n.as_f64().unwrap_or(0.0)).unwrap_or(0.0))
        .collect();

    // --- units ---
    let uo = at("units");
    let umo = at("unit_mask");
    let half = (board / 2) as f64;
    let mut pos: Vec<[usize; 2]> = vec![farm.farmer];
    pos.extend(farm.hands.iter().copied());
    for (k, p) in pos.iter().take(MAX_UNITS).enumerate() {
        let (x, y) = (p[0], p[1]);
        let empty = Inventory::new();
        let inv = priv_.inventories.get(k).unwrap_or(&empty);
        let t = &farm.tiles[y][x];
        let b = uo + k * UNIT_FEAT_DIM;
        let f = &mut out[b..b + UNIT_FEAT_DIM];
        let on_shed = shed_tiles.iter().any(|(sx, sy)| *sx == x && *sy == y);
        let total: f64 = inv.values().map(|v| *v as f64).sum();
        let animals_held: f64 = ANIMALS.iter().map(|a| inv_get(inv, a)).sum();
        let prod_not_wheat: f64 = PRODUCTS
            .iter()
            .filter(|g| **g != "WHEAT")
            .map(|g| inv_get(inv, g))
            .sum();
        let is_plant = matches!(t, Tile::Occupied(Occupant::Plant(_)));
        let has_animal = t.has_animal();
        put(f, 0, if k == 0 { 1.0 } else { 0.0 });
        put(f, 1, k as f64 / MAX_UNITS as f64);
        put(f, 2, x as f64 / 10.0);
        put(f, 3, y as f64 / 10.0);
        put(f, 4, ((x as f64 - (half - 0.5)).abs() + (y as f64 - (half - 0.5)).abs()) / 10.0);
        put(f, 5, if on_shed { 1.0 } else { 0.0 });
        put(f, 6, total / 50.0);
        put(f, 7, animals_held);
        put(f, 8, prod_not_wheat / 50.0);
        put(f, 9, hour as f64 / turns as f64);
        put(f, 10, if inv_get(inv, "FERTILIZER") > 0.0 { 1.0 } else { 0.0 });
        put(f, 11, if is_plant { 1.0 } else { 0.0 });
        put(f, 12, if has_animal { 1.0 } else { 0.0 });
        put(f, 13, if inv_get(inv, "GOOSE") > 0.0 { 1.0 } else { 0.0 });
        put(f, 14, if inv_get(inv, "COW") > 0.0 { 1.0 } else { 0.0 });
        put(f, 15, if inv_get(inv, "SHEEP") > 0.0 { 1.0 } else { 0.0 });
        put(f, 16, if matches!(t, Tile::Locked(_)) { 1.0 } else { 0.0 });
        for (i, g) in PRODUCTS.iter().enumerate() {
            put(f, 17 + i, inv_get(inv, g) / 20.0);
        }
        let cargo: f64 = PRODUCTS
            .iter()
            .enumerate()
            .map(|(i, g)| inv_get(inv, g) * price[i])
            .sum();
        put(f, 26, cargo / 3000.0);
        out[umo + k] = 1.0;
    }

    // --- town demand ---
    let shops = &town.unlocked_shops;
    let town_rate = |g: &str| -> f64 {
        let base = if g == "FERTILIZER" { 0.0 } else { 1.0 };
        let k: f64 = shops
            .iter()
            .filter(|s| shop_items(shops_def, s).iter().any(|i| i == g))
            .map(|s| if shop_items(shops_def, s).len() == 1 { 2.0 } else { 1.0 })
            .sum();
        base + 6.0 * k
    };
    let town_ahead = |g: &str| -> f64 {
        let known: f64 = shops
            .iter()
            .filter(|s| shop_items(shops_def, s).iter().any(|i| i == g))
            .map(|s| if shop_items(shops_def, s).len() == 1 { 2.0 } else { 1.0 })
            .sum();
        let base = if g == "FERTILIZER" { 0.0 } else { 1.0 };
        let w = shop_w(shops_def, g);
        let mut acc = 0.0;
        for t in day..season_days {
            let extra_shops = (((t / 3).min(8)) as f64 - shops.len() as f64).max(0.0) * w;
            acc += base + 6.0 * (known + extra_shops);
        }
        acc
    };

    // --- goods ---
    let po = at("products");
    let shed = &priv_.shed;
    let seeds = &priv_.seeds;
    let mut pstat: Vec<(f64, f64, f64, f64, f64)> = Vec::with_capacity(PRODUCTS.len());
    for (i, g) in PRODUCTS.iter().enumerate() {
        let pr = price[i];
        let iv = invm[i];
        let ta = town_ahead(g);
        let rev10: f64 =
            (0..10).map(|k| pricing.price_at(i, iv + k as f64)).sum::<f64>() / 10.0;
        let (n_left, money_left) = bucket_left(pricing, i, iv);
        pstat.push((pr, iv, ta, n_left, money_left));
        let base = pricing.base_of(i);
        let b = po + i * PRODUCT_FEAT_DIM;
        let f = &mut out[b..b + PRODUCT_FEAT_DIM];
        put(f, 0, pr / 300.0);
        put(f, 1, (iv - MARKET_I0) / 1000.0);
        put(f, 2, pr / base);
        put(f, 3, inv_get(shed, g) / shed_cap);
        put(f, 4, (pr - pricing.price_at(i, iv + 1.0)) / 20.0);
        put(f, 5, town_rate(g) / 50.0);
        put(f, 6, ta / 600.0);
        put(f, 7, ex.pi[i] / 50.0);
        put(f, 8, ex.oppsold[i] / 500.0);
        put(f, 9, ex.mysold[i] / 500.0);
        put(f, 10, rev10 / 300.0);
        put(f, 11, (pr - pricing.price_at(i, iv + 10.0)) / 300.0);
        // ⚠ Python's `int(iv - ta)` truncates TOWARD ZERO.
        put(f, 12, (pricing.price_at(i, (iv - ta).trunc()) - pr) / 300.0);
        put(f, 13, ex.crowd[i]);
        put(f, 14, inv_get(seeds, g) / 20.0);
        put(f, 15, if pr <= 1.0 { 1.0 } else { 0.0 });
        put(f, 16, n_left.min(500.0) / 500.0);
        put(f, 17, money_left.min(30000.0) / 30000.0);
    }

    // --- field recomputation ---
    let (mut n_plant, mut n_animal, mut n_empty, mut n_weed) = (0i64, 0i64, 0i64, 0i64);
    let (mut n_dying, mut n_unfed, mut decaying) = (0i64, 0i64, 0i64);
    let mut my_crop = [0f64; CROPS.len()];
    let mut my_anim = [0f64; ANIMALS.len()];
    let (mut free_coop, mut free_pasture) = (0f64, 0f64);
    for row in &farm.tiles {
        for t in row {
            match t {
                Tile::Empty => n_empty += 1,
                Tile::Occupied(Occupant::Plant(p)) => {
                    n_plant += 1;
                    if p.consecutive_unwatered >= 1 && !p.watered_today {
                        n_dying += 1;
                    }
                    if p.max_lifespan_step >= 0 && p.max_lifespan_step <= step {
                        decaying += p.yield_units;
                    }
                    my_crop[crop_index(&p.crop)] += 1.0;
                }
                Tile::Occupied(Occupant::Weed) => n_weed += 1,
                Tile::Occupied(Occupant::Coop { animal })
                | Tile::Occupied(Occupant::Pasture { animal }) => match animal {
                    Some(a) => {
                        n_animal += 1;
                        my_anim[animal_index(&a.animal)] += 1.0;
                        if !a.fed_today {
                            n_unfed += 1;
                        }
                    }
                    None => {
                        if matches!(t, Tile::Occupied(Occupant::Coop { .. })) {
                            free_coop += 1.0;
                        } else {
                            free_pasture += 1.0;
                        }
                    }
                },
                _ => {}
            }
        }
    }
    let (mut op_plant, mut op_animal) = (0f64, 0f64);
    let mut op_crop = [0f64; CROPS.len()];
    let mut op_anim = [0f64; ANIMALS.len()];
    for row in &opp.tiles {
        for t in row {
            match t {
                Tile::Occupied(Occupant::Plant(p)) => {
                    op_plant += 1.0;
                    op_crop[crop_index(&p.crop)] += 1.0;
                }
                Tile::Occupied(Occupant::Coop { animal: Some(a) })
                | Tile::Occupied(Occupant::Pasture { animal: Some(a) }) => {
                    op_animal += 1.0;
                    op_anim[animal_index(&a.animal)] += 1.0;
                }
                _ => {}
            }
        }
    }

    // --- crop tokens ---
    let money = farm.money;
    let left = (season_days - day).max(0) as f64;
    let co = at("crops");
    for (i, c) in CROPS.iter().enumerate() {
        let cd = crop_rules(c);
        let gi = PRODUCTS.iter().position(|p| p == c).unwrap();
        let (pr, iv, ta, n_left, money_left) = pstat[gi];
        let b = co + i * CROP_FEAT_DIM;
        let f = &mut out[b..b + CROP_FEAT_DIM];
        put(f, 0, cd.seed as f64 / 300.0);
        put(f, 1, cd.first_yield_day as f64 / 16.0);
        put(f, 2, cd.max_yield_day as f64 / 16.0);
        put(f, 3, cd.interval as f64 / 4.0);
        put(f, 4, cd.max_yield as f64 / 6.0);
        put(f, 5, if cd.ongoing { 1.0 } else { 0.0 });
        put(f, 6, pr / 300.0);
        put(f, 7, (iv - MARKET_I0) / 1000.0);
        put(f, 8, n_left.min(500.0) / 500.0);
        put(f, 9, money_left.min(30000.0) / 30000.0);
        put(f, 10, ta / 600.0);
        put(f, 11, inv_get(seeds, c) / 20.0);
        put(f, 12, my_crop[i] / 100.0);
        put(f, 13, op_crop[i] / 100.0);
        put(f, 14, (money / (cd.seed.max(1)) as f64).min(40.0) / 40.0);
        put(f, 15, ((cd.max_yield as f64 * pr) / (cd.seed.max(1)) as f64).min(20.0) / 20.0);
        put(f, 16, (left / (cd.first_yield_day.max(1)) as f64).min(2.0) / 2.0);
        put(f, 17, inv_get(shed, c) / 100.0);
    }

    // --- animal tokens ---
    let ao = at("animals");
    for (i, an) in ANIMALS.iter().enumerate() {
        let ad = animal_rules(an);
        let gi = PRODUCTS.iter().position(|p| *p == ad.product).unwrap();
        let (pr, iv, ta, n_left, money_left) = pstat[gi];
        let crowd: f64 = priv_.inventories.iter().map(|iv2| inv_get(iv2, an)).sum();
        let free = if ad.structure == "COOP" { free_coop } else { free_pasture };
        let b = ao + i * ANIMAL_FEAT_DIM;
        let f = &mut out[b..b + ANIMAL_FEAT_DIM];
        put(f, 0, ad.cost as f64 / 500.0);
        put(f, 1, if ad.structure == "COOP" { 1.0 } else { 0.0 });
        put(f, 2, if ad.structure == "PASTURE" { 1.0 } else { 0.0 });
        put(f, 3, ad.first_yield_day as f64 / 16.0);
        put(f, 4, ad.interval as f64 / 4.0);
        put(f, 5, ad.max_held as f64 / 6.0);
        put(f, 6, pr / 300.0);
        put(f, 7, (iv - MARKET_I0) / 1000.0);
        put(f, 8, n_left.min(500.0) / 500.0);
        put(f, 9, money_left.min(30000.0) / 30000.0);
        put(f, 10, ta / 600.0);
        put(f, 11, inv_get(shed, an) / 5.0);
        put(f, 12, my_anim[i] / 20.0);
        put(f, 13, free / 20.0);
        put(f, 14, crowd / 5.0);
        put(f, 15, (money / (ad.cost.max(1)) as f64).min(20.0) / 20.0);
        put(f, 16, (left / (ad.first_yield_day.max(1)) as f64).min(2.0) / 2.0);
        put(f, 17, op_anim[i] / 20.0);
    }

    // --- global token ---
    let go = at("glob");
    let md = farm.money - opp.money;
    let shed_used: f64 = shed.values().map(|v| *v as f64).sum();
    let wheat_held: f64 = priv_.inventories.iter().map(|iv2| inv_get(iv2, "WHEAT")).sum();
    let g = &mut out[go..go + GLOBAL_FEAT_DIM];
    put(g, 0, day as f64 / season_days as f64);
    put(g, 1, hour as f64 / turns as f64);
    put(g, 2, farm.money / 50000.0);
    put(g, 3, opp.money / 50000.0);
    put(g, 4, md / 50000.0);
    put(g, 5, md / (20000.0 * ((29 - day).max(1) as f64).sqrt()));
    put(g, 6, farm.hands.len() as f64 / MAX_UNITS as f64);
    put(g, 7, opp.hands.len() as f64 / MAX_UNITS as f64);
    put(g, 8, farm.unlocked_quadrants.len() as f64 / 4.0);
    put(g, 9, opp.unlocked_quadrants.len() as f64 / 4.0);
    put(g, 10, n_plant as f64 / 100.0);
    put(g, 11, n_animal as f64 / 100.0);
    put(g, 12, n_empty as f64 / 100.0);
    put(g, 13, n_weed as f64 / 100.0);
    put(g, 14, op_plant / 100.0);
    put(g, 15, op_animal / 100.0);
    put(g, 16, shed_used / 100.0);
    put(g, 17, shops.len() as f64 / 8.0);
    put(g, 18, inv_get(shed, "WHEAT") / (n_animal as f64).max(1.0) / 10.0);
    put(g, 19, inv_get(shed, "GOOSE") / 5.0);
    put(g, 20, inv_get(shed, "COW") / 5.0);
    put(g, 21, inv_get(shed, "SHEEP") / 5.0);
    put(g, 22, ((4 - step % 4) % 4) as f64 / 4.0);
    put(g, 23, ((3 - day % 3) % 3) as f64 / 3.0);
    put(g, 24, FARM_HAND_COST_MULT * fib(farm.hires_today as i64) / 1000.0);
    let q = farm.unlocked_quadrants.len() as i64 - 1;
    put(
        g,
        25,
        if q >= 0 && (q as usize) < LAND_PRICES.len() {
            LAND_PRICES[q as usize]
        } else {
            0.0
        } / 4000.0,
    );
    put(g, 26, n_dying as f64 / 100.0);
    put(
        g,
        27,
        clip1((n_unfed as f64 - inv_get(shed, "WHEAT") - wheat_held) / 20.0),
    );
    put(g, 28, decaying as f64 / 20.0);
}
