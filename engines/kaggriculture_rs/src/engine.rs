use crate::{actions, model::*, rng::PyRandom};
use indexmap::IndexMap;
use num_bigint::BigInt;
use serde::Serialize;
use serde_json::Value;
use std::sync::OnceLock;
#[derive(Clone)]
pub struct Engine {
    pub config: Config,
    pub seed: BigInt,
    pub farms: [Farm; 2],
    pub privates: [Private; 2],
    market: Market,
    pub town: Town,
    pub step: i64,
    pub done: bool,
    pricing: MarketPricing,
    prices_initialized: bool,
}
impl Engine {
    pub fn new(config: Config, seed: BigInt) -> Self {
        let params = config
            .params
            .as_ref()
            .unwrap_or(&constants()["MARKET_PARAMS"]);
        let pricing = MarketPricing::new(params);
        let market = Market {
            inventory: PRODUCTS
                .into_iter()
                .map(|s| (s.into(), number(&params[s]["I0"])))
                .collect(),
            prices: PRODUCTS
                .into_iter()
                .map(|s| (s.into(), number(&params[s]["base"])))
                .collect(),
            params: config.params.clone(),
        };
        Self {
            farms: std::array::from_fn(|_| new_farm(&config)),
            privates: std::array::from_fn(|_| new_private()),
            market,
            pricing,
            prices_initialized: false,
            town: Town {
                unlocked_shops: vec![],
            },
            step: 0,
            done: false,
            config,
            seed,
        }
    }
    /// Market parameters are immutable for an episode so compiled pricing
    /// cannot become stale. Create a new Engine to use another configuration.
    /// The engine's price curve. The features need exactly the same one: computing the price
    /// a second way is precisely the class of bug we avoid.
    pub fn pricing(&self) -> &MarketPricing {
        &self.pricing
    }
    pub fn market(&self) -> &Market {
        &self.market
    }
    pub fn observations(&self) -> Vec<Observation<'_>> {
        (0..2)
            .map(|i| Observation {
                player: i,
                step: self.step,
                day: self.step / self.config.turns,
                hour: self.step % self.config.turns,
                farms: &self.farms,
                private: &self.privates[i],
                market: &self.market,
                town: &self.town,
                remaining_overage_time: 60,
            })
            .collect()
    }
    pub fn result(&self) -> StepResult<'_> {
        StepResult {
            observations: self.observations(),
            rewards: std::array::from_fn(|i| {
                if self.done {
                    Value::from(self.farms[i].money)
                } else {
                    Value::from(0)
                }
            }),
            statuses: if self.done {
                ["DONE"; 2]
            } else {
                ["ACTIVE"; 2]
            },
            done: self.done,
        }
    }
    pub fn advance(&mut self, actions: &[Value; 2]) {
        if self.done {
            return;
        }
        let day = self.step / self.config.turns;
        for (i, a) in actions.iter().enumerate().take(2) {
            // Null has the same no-op behavior as the default PASS action.
            let main = a.get("farmer").unwrap_or(&Value::Null);
            let hands = a.get("hands").and_then(Value::as_array);
            // With only the farmer, the simultaneous seed-demand check is a
            // single lookup. Avoid allocating a map for this common case.
            if hands.is_none_or(Vec::is_empty) {
                let blocked = main.is_array()
                    && main[0] == "PLANT"
                    && main[1]
                        .as_str()
                        .is_some_and(|crop| count(&self.privates[i].seeds, crop) < 1);
                if !blocked {
                    actions::unit(
                        &mut self.farms[i],
                        &mut self.privates[i],
                        0,
                        main,
                        &self.config,
                        day,
                    );
                }
                continue;
            }
            let units = std::iter::once(main).chain(hands.into_iter().flatten());
            let mut demand: IndexMap<&str, i64> = IndexMap::new();
            for a in units.clone() {
                if a.is_array() && a[0] == "PLANT" {
                    if let Some(c) = a[1].as_str() {
                        *demand.entry(c).or_default() += 1;
                    }
                }
            }
            // Keep only oversubscribed crops; borrow action strings instead of
            // allocating a String for every requested and blocked crop.
            demand.retain(|crop, n| *n > count(&self.privates[i].seeds, crop));
            for (idx, a) in units.enumerate() {
                if a.is_array()
                    && a[0] == "PLANT"
                    && a[1].as_str().is_some_and(|s| demand.contains_key(s))
                {
                    continue;
                }
                actions::unit(
                    &mut self.farms[i],
                    &mut self.privates[i],
                    idx,
                    a,
                    &self.config,
                    day,
                );
            }
        }
        actions::market(
            &mut self.farms,
            &mut self.privates,
            &mut self.market,
            actions,
            &self.config,
            &self.pricing,
        );
        // Market orders refresh their own prices. Only scheduled inventory
        // changes and the initial base-price normalization need a final pass.
        let mut prices_dirty = !self.prices_initialized;
        if self.step % self.config.shop == 0 {
            for shop in &self.town.unlocked_shops {
                let products = constants()["SHOPS"][shop].as_array().unwrap();
                for p in products {
                    prices_dirty = true;
                    adjust_market(
                        &mut self.market,
                        p.as_str().unwrap(),
                        if products.len() == 1 { -2 } else { -1 },
                    );
                }
            }
        }
        if self.step % self.config.center == 0 {
            prices_dirty = true;
            for p in &PRODUCTS[..8] {
                adjust_market(&mut self.market, p, -1);
            }
        }
        if prices_dirty {
            actions::refresh(&mut self.market, &self.pricing);
            self.prices_initialized = true;
        }
        if (self.step + 1) % self.config.turns == 0 {
            self.end_day(day);
        } else {
            for f in &mut self.farms {
                for row in &mut f.tiles {
                    for tile in row {
                        decay_plant(tile, self.step);
                    }
                }
            }
        }
        self.step += 1;
        self.done = self.step >= self.config.steps - 1;
    }
    fn end_day(&mut self, day: i64) {
        let c = &self.config;
        // Late in an episode there may be no empty tiles and no shop draw.
        // Initializing MT19937 is then entirely unnecessary. The lazy stream
        // still consumes every original weed draw, including weed=0 and 1.
        let mut rng: Option<PyRandom> = None;
        let new_rng =
            || PyRandom::new(&((&self.seed * BigInt::from(1_000_003)) ^ BigInt::from(day)));
        for i in 0..2 {
            let f = &mut self.farms[i];
            // These updates are local to a tile, and none creates an Empty
            // tile. Fusing the scans preserves weed draws in row-major order.
            for row in &mut f.tiles {
                for tile in row {
                    decay_plant(tile, self.step);
                    if let Tile::Occupied(Occupant::Plant(t)) = tile {
                        let watered = t.watered_today;
                        t.consecutive_unwatered = if watered {
                            0
                        } else {
                            t.consecutive_unwatered + 1
                        };
                        t.watered_today = false;
                        if t.consecutive_unwatered >= 2 {
                            *tile = Tile::Occupied(Occupant::Weed);
                            continue;
                        }
                        let rules = crop_rules(&t.crop);
                        if rules.ongoing {
                            let since = day + 1 - t.planted_day - rules.first_yield_day;
                            let interval = rules.interval;
                            let max = rules.max_yield;
                            if since >= 0 && since % interval == 0 {
                                let times = since / interval + 1;
                                if times <= max {
                                    let fert = watered && t.fertilized_until_day >= day;
                                    t.yield_units =
                                        (t.yield_units + if fert { 2 } else { 1 }).min(max);
                                    if times == max {
                                        t.max_lifespan_step = (day + 2) * c.turns;
                                    }
                                }
                            }
                        }
                    } else if let Some(t) = tile.animal_mut() {
                        t.consecutive_unfed = if t.fed_today {
                            0
                        } else {
                            t.consecutive_unfed + 1
                        };
                        if t.consecutive_unfed >= 2 {
                            *tile = Tile::Occupied(if t.animal == "GOOSE" {
                                Occupant::Coop { animal: None }
                            } else {
                                Occupant::Pasture { animal: None }
                            });
                            continue;
                        }
                        let rules = animal_rules(&t.animal);
                        let since = day + 1 - t.placed_day - rules.first_yield_day;
                        if since >= 0 && since % rules.interval == 0 {
                            let bonus = if t.fed_today { t.pending_care_bonus } else { 0 };
                            t.yield_units = (t.yield_units + 1 + bonus).min(rules.max_held);
                            t.pending_care_bonus = 0;
                        }
                        if t.cared_today && t.fed_today {
                            t.pending_care_bonus += 1;
                        }
                        t.fertilizer_available = true;
                        t.fed_today = false;
                        t.cared_today = false;
                    } else if matches!(tile, Tile::Empty)
                        && rng.get_or_insert_with(new_rng).random() < c.weed
                    {
                        *tile = Tile::Occupied(Occupant::Weed);
                    }
                }
            }
            let p = &mut self.privates[i];
            for inv in &mut p.inventories {
                drop_inv(inv, &mut p.shed, c.cap);
            }
            // All inventories were drained. Reuse the farmer's allocation.
            p.inventories.truncate(1);
            if p.inventories.is_empty() {
                p.inventories.push(Inventory::new());
            }
            f.farmer = access(c.board)[0];
            f.hands.clear();
            f.hires_today = 0;
        }
        if (day + 1) % c.unlock == 0 && self.town.unlocked_shops.len() < 8 {
            static SHOPS: OnceLock<Vec<&'static str>> = OnceLock::new();
            let shops = SHOPS.get_or_init(|| {
                let mut shops: Vec<_> = constants()["SHOPS"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect();
                shops.sort_unstable();
                shops
            });
            self.town
                .unlocked_shops
                .push(shops[rng.get_or_insert_with(new_rng).below(shops.len() as u32)].to_owned());
        }
    }
}

#[inline]
fn decay_plant(tile: &mut Tile, step: i64) {
    if let Tile::Occupied(Occupant::Plant(t)) = tile {
        if t.max_lifespan_step >= 0
            && step >= t.max_lifespan_step
            && (step - t.max_lifespan_step) % 2 == 0
        {
            t.yield_units -= 1;
            if t.yield_units <= 0 {
                *tile = Tile::Occupied(Occupant::Weed);
            }
        }
    }
}

/// Borrowed views serialize directly into Python dictionaries without an
/// intermediate JSON tree. No references into engine state escape the binding.
#[derive(Serialize)]
pub struct Observation<'a> {
    pub player: usize,
    pub step: i64,
    pub day: i64,
    pub hour: i64,
    pub farms: &'a [Farm],
    pub private: &'a Private,
    pub market: &'a Market,
    pub town: &'a Town,
    #[serde(rename = "remainingOverageTime")]
    pub remaining_overage_time: i64,
}
#[derive(Serialize)]
pub struct StepResult<'a> {
    pub observations: Vec<Observation<'a>>,
    #[serde(serialize_with = "serialize_slice")]
    pub rewards: [Value; 2],
    #[serde(serialize_with = "serialize_slice")]
    pub statuses: [&'static str; 2],
    pub done: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn initial_prices_normalize_even_if_public_step_was_changed() {
        let mut params = constants()["MARKET_PARAMS"].clone();
        for item in PRODUCTS {
            params[item]["base"] = json!(2.5);
        }
        let mut config = Config::parse(&json!({})).unwrap();
        config.params = Some(params);
        config.shop = 100;
        config.center = 100;
        let mut engine = Engine::new(config, 1.into());
        engine.step = 1;
        assert!(engine.market().prices["WHEAT"].is_f64());
        let actions = [json!({"farmer": ["PASS"]}), json!({"farmer": ["PASS"]})];
        engine.advance(&actions);
        assert_eq!(engine.market().prices["WHEAT"].as_i64(), Some(2));
        let expected = serde_json::to_vec(&engine.market().prices).unwrap();
        engine.advance(&actions);
        assert_eq!(
            serde_json::to_vec(&engine.market().prices).unwrap(),
            expected
        );
    }

    #[test]
    fn daily_decay_precedes_yield_and_preserves_shop_random_stream() {
        let mut config = Config::parse(&json!({})).unwrap();
        config.turns = 1;
        config.unlock = 1;
        config.weed = 0.0; // Even impossible weeds must consume their draws.
        let mut engine = Engine::new(config, 17.into());
        for farm in &mut engine.farms {
            for row in &mut farm.tiles {
                row.fill(Tile::Occupied(Occupant::Weed));
            }
            farm.tiles[0][0] = Tile::Empty;
        }
        let plant = Plant {
            crop: "TOMATO".into(),
            planted_day: -7,
            watered_today: true,
            consecutive_unwatered: 0,
            yield_units: 1,
            max_lifespan_step: 0,
            fertilized_until_day: -1,
        };
        engine.farms[0].tiles[0][1] = Tile::Occupied(Occupant::Plant(plant.clone()));
        engine.farms[0].tiles[0][2] = Tile::Occupied(Occupant::Plant(Plant {
            yield_units: 2,
            ..plant
        }));
        let mut expected_rng = PyRandom::new(&BigInt::from(17 * 1_000_003));
        expected_rng.random();
        expected_rng.random();
        let mut shops: Vec<_> = constants()["SHOPS"].as_object().unwrap().keys().collect();
        shops.sort();
        let expected_shop = shops[expected_rng.below(shops.len() as u32)].clone();
        engine.advance(&[json!({}), json!({})]);
        assert!(matches!(
            engine.farms[0].tiles[0][1],
            Tile::Occupied(Occupant::Weed)
        ));
        let Tile::Occupied(Occupant::Plant(survivor)) = &engine.farms[0].tiles[0][2] else {
            panic!("plant with two units should survive decay");
        };
        assert_eq!(survivor.yield_units, 2); // 2 - decay + daily yield.
        assert_eq!(engine.town.unlocked_shops, vec![expected_shop]);
    }

    #[test]
    fn farmer_only_seed_check_preserves_blocking_and_inventory_side_effects() {
        let config = Config::parse(&json!({})).unwrap();
        let mut engine = Engine::new(config, 0.into());
        engine.privates[0].inventories.clear();
        engine.privates[0].seeds.insert("WHEAT".into(), 0);
        let actions = [json!({"farmer": ["PLANT", "WHEAT"]}), json!({})];
        engine.advance(&actions);
        assert!(engine.privates[0].inventories.is_empty());
        // Public state can change between calls; no seed availability is cached.
        engine.privates[0].seeds.insert("WHEAT".into(), 1);
        let [x, y] = engine.farms[0].farmer;
        engine.farms[0].tiles[y][x] = Tile::Empty;
        engine.advance(&actions);
        assert_eq!(engine.privates[0].inventories.len(), 1);
        assert_eq!(engine.privates[0].seeds["WHEAT"], 0);
        assert!(matches!(
            engine.farms[0].tiles[y][x],
            Tile::Occupied(Occupant::Plant(_))
        ));
    }
}
