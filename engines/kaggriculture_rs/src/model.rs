use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Number, Value};
use std::sync::OnceLock;
pub type Inventory = IndexMap<String, i64>;
pub const PRODUCTS: [&str; 9] = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
];
pub const CROPS: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
pub const ANIMALS: [&str; 3] = ["GOOSE", "COW", "SHEEP"];
pub fn constants() -> &'static Value {
    static C: OnceLock<Value> = OnceLock::new();
    C.get_or_init(|| serde_json::from_str(include_str!("constants.json")).unwrap())
}
pub fn ci(group: &str, item: &str, key: &str) -> i64 {
    constants()[group][item][key].as_i64().unwrap()
}
pub fn cs<'a>(group: &str, item: &str, key: &str) -> &'a str {
    constants()[group][item][key].as_str().unwrap()
}
pub fn ongoing(c: &str) -> bool {
    constants()["CROPS"][c]["ongoing"].as_bool().unwrap()
}
// Decode immutable rules once; simulation loops use typed fields instead of
// repeated JSON object lookups. JSON remains the single source of truth.
pub struct CropRules {
    pub seed: i64,
    pub first_yield_day: i64,
    pub max_yield_day: i64,
    pub interval: i64,
    pub max_yield: i64,
    pub ongoing: bool,
}
pub fn crop_rules(item: &str) -> &'static CropRules {
    static RULES: OnceLock<[CropRules; CROPS.len()]> = OnceLock::new();
    &RULES.get_or_init(|| {
        CROPS.map(|item| CropRules {
            seed: ci("CROPS", item, "seed"),
            first_yield_day: ci("CROPS", item, "first_yield_day"),
            max_yield_day: ci("CROPS", item, "max_yield_day"),
            interval: ci("CROPS", item, "interval"),
            max_yield: ci("CROPS", item, "max_yield"),
            ongoing: ongoing(item),
        })
    })[CROPS.iter().position(|name| *name == item).unwrap()]
}
pub struct AnimalRules {
    pub cost: i64,
    pub first_yield_day: i64,
    pub interval: i64,
    pub max_held: i64,
    pub structure: &'static str,
    pub product: &'static str,
}
pub fn animal_rules(item: &str) -> &'static AnimalRules {
    static RULES: OnceLock<[AnimalRules; ANIMALS.len()]> = OnceLock::new();
    &RULES.get_or_init(|| {
        ANIMALS.map(|item| AnimalRules {
            cost: ci("ANIMALS", item, "cost"),
            first_yield_day: ci("ANIMALS", item, "first_yield_day"),
            interval: ci("ANIMALS", item, "interval"),
            max_held: ci("ANIMALS", item, "max_held"),
            structure: cs("ANIMALS", item, "structure"),
            product: cs("ANIMALS", item, "product"),
        })
    })[ANIMALS.iter().position(|name| *name == item).unwrap()]
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plant {
    pub crop: String,
    pub planted_day: i64,
    pub watered_today: bool,
    pub consecutive_unwatered: i64,
    pub yield_units: i64,
    pub max_lifespan_step: i64,
    pub fertilized_until_day: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Animal {
    pub animal: String,
    pub placed_day: i64,
    pub yield_units: i64,
    pub consecutive_unfed: i64,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Occupant {
    #[serde(rename = "PLANT")]
    Plant(Plant),
    #[serde(rename = "WEED")]
    Weed,
    #[serde(rename = "COOP")]
    Coop {
        #[serde(flatten)]
        animal: Option<Animal>,
    },
    #[serde(rename = "PASTURE")]
    Pasture {
        #[serde(flatten)]
        animal: Option<Animal>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Tile {
    Empty,
    Locked(String),
    Occupied(Occupant),
}
impl Tile {
    pub fn animal_mut(&mut self) -> Option<&mut Animal> {
        match self {
            Self::Occupied(Occupant::Coop { animal } | Occupant::Pasture { animal }) => {
                animal.as_mut()
            }
            _ => None,
        }
    }
    pub fn has_animal(&self) -> bool {
        matches!(
            self,
            Self::Occupied(
                Occupant::Coop { animal: Some(_) } | Occupant::Pasture { animal: Some(_) }
            )
        )
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Farm {
    pub money: f64,
    pub tiles: Vec<Vec<Tile>>,
    #[serde(serialize_with = "serialize_slice")]
    pub farmer: [usize; 2],
    #[serde(serialize_with = "serialize_positions")]
    pub hands: Vec<[usize; 2]>,
    pub unlocked_quadrants: Vec<String>,
    pub hires_today: usize,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Private {
    pub shed: Inventory,
    pub seeds: Inventory,
    pub inventories: Vec<Inventory>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Market {
    pub inventory: IndexMap<String, Number>,
    pub prices: IndexMap<String, Number>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Town {
    pub unlocked_shops: Vec<String>,
}
#[derive(Clone)]
pub struct Config {
    pub board: usize,
    pub money: i64,
    pub steps: i64,
    pub turns: i64,
    pub cap: i64,
    pub orders: usize,
    pub hire: i64,
    pub weed: f64,
    pub unlock: i64,
    pub shop: i64,
    pub center: i64,
    pub params: Option<Value>,
}
impl Config {
    pub fn parse(v: &Value) -> Result<Self, String> {
        if !v.is_null() && !v.is_object() {
            return Err("config must be a dictionary".into());
        }
        static SCHEMA: OnceLock<Value> = OnceLock::new();
        let schema = SCHEMA.get_or_init(|| {
            serde_json::from_str(include_str!("../reference/kaggriculture.json")).unwrap()
        });
        let defaults = schema["configuration"].as_object().unwrap();
        if let Some(obj) = v.as_object() {
            for k in obj.keys() {
                if !defaults.contains_key(k)
                    && !matches!(k.as_str(), "runTimeout" | "agentTimeout" | "timeout")
                {
                    return Err(format!("unknown configuration field: {k}"));
                }
            }
        }
        // Borrow overrides/defaults; configuration parsing does not need a
        // deep-cloned intermediate JSON object on every environment creation.
        let value = |k: &str| {
            v.get(k).unwrap_or_else(|| {
                let field = &defaults[k];
                field.get("default").unwrap_or(field)
            })
        };
        let get = |k: &str, min: i64| -> Result<i64, String> {
            let n = value(k)
                .as_i64()
                .ok_or_else(|| format!("{k} must be an integer"))?;
            if n < min {
                Err(format!("{k} must be >= {min}"))
            } else {
                Ok(n)
            }
        };
        let weed = value("weedSpawnChance")
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.)
            .ok_or("invalid weedSpawnChance")?;
        let p = value("marketParams");
        if !p.is_object() {
            return Err("marketParams must be a dictionary".into());
        }
        let params = if p.as_object().unwrap().is_empty() {
            None
        } else {
            let mut all = constants()["MARKET_PARAMS"].clone();
            for (item, patch) in p.as_object().unwrap() {
                if let (Some(dest), Some(patch)) = (
                    all.get_mut(item).and_then(Value::as_object_mut),
                    patch.as_object(),
                ) {
                    for (k, v) in patch {
                        dest.insert(k.clone(), v.clone());
                    }
                }
            }
            for item in PRODUCTS {
                let x = &all[item];
                for k in ["base", "I0", "T", "below_target", "above_target"] {
                    if x[k].as_f64().filter(|n| n.is_finite()).is_none() {
                        return Err(format!("invalid market parameter {item}.{k}"));
                    }
                }
                if x["T"].as_f64().unwrap() <= 0. {
                    return Err("market T must be positive".into());
                }
            }
            Some(all)
        };
        Ok(Self {
            board: get("boardSize", 4)? as usize,
            money: get("startingMoney", 0)?,
            steps: get("episodeSteps", 1)?,
            turns: get("turnsPerDay", 1)?,
            cap: get("shedCapacity", 1)?,
            orders: get("maxMarketOrdersPerTurn", 1)? as usize,
            hire: get("farmHandCostMult", 0)?,
            weed,
            unlock: get("townShopUnlockInterval", 1)?,
            shop: get("townShopSellInterval", 1)?,
            center: get("townCenterSellInterval", 1)?,
            params,
        })
    }
}
pub fn quadr(x: usize, y: usize, n: usize) -> String {
    quadrant_name(x, y, n).into()
}
pub fn quadrant_name(x: usize, y: usize, n: usize) -> &'static str {
    match (y < n / 2, x < n / 2) {
        (true, true) => "NW",
        (true, false) => "NE",
        (false, true) => "SW",
        (false, false) => "SE",
    }
}
pub fn access(n: usize) -> [[usize; 2]; 4] {
    let h = n / 2;
    [[h - 1, h - 1], [h, h - 1], [h - 1, h], [h, h]]
}
pub fn new_farm(c: &Config) -> Farm {
    Farm {
        money: c.money as f64,
        tiles: (0..c.board)
            .map(|y| {
                (0..c.board)
                    .map(|x| {
                        if quadrant_name(x, y, c.board) == "NW" {
                            Tile::Empty
                        } else {
                            Tile::Locked("LOCKED".into())
                        }
                    })
                    .collect()
            })
            .collect(),
        farmer: access(c.board)[0],
        hands: vec![],
        unlocked_quadrants: vec!["NW".into()],
        hires_today: 0,
    }
}
pub fn new_private() -> Private {
    Private {
        shed: PRODUCTS
            .into_iter()
            .chain(ANIMALS)
            .map(|s| (s.into(), 0))
            .collect(),
        seeds: CROPS.into_iter().map(|s| (s.into(), 0)).collect(),
        inventories: vec![Inventory::new()],
    }
}
pub fn add(inv: &mut Inventory, s: &str, n: i64) {
    if let Some(value) = inv.get_mut(s) {
        *value += n;
    } else {
        inv.insert(s.into(), n);
    }
}
pub fn count(inv: &Inventory, s: &str) -> i64 {
    *inv.get(s).unwrap_or(&0)
}
pub fn take(inv: &mut Inventory, s: &str, n: i64) -> bool {
    if count(inv, s) < n {
        return false;
    }
    let value = inv.get_mut(s).unwrap();
    *value -= n;
    if *value == 0 {
        inv.shift_remove(s);
    }
    true
}
pub fn drop_inv(inv: &mut Inventory, shed: &mut Inventory, cap: i64) {
    if inv.is_empty() {
        return;
    }
    let mut used = shed.values().sum::<i64>();
    for (s, n) in inv.drain(..) {
        let k = n.min((cap - used).max(0));
        if k > 0 {
            add(shed, &s, k);
            used += k;
        }
    }
}
pub fn shape(f: &str, x: f64, t: f64) -> f64 {
    let x = x.max(0.);
    match f {
        "sq" => x * x,
        "sqrt" => x.sqrt(),
        "log" => (1. + x).ln(),
        "log10" => (1. + x).log10(),
        "hinge" if t > 0. => {
            let u = x / t;
            u + 8. * (u - 1.).max(0.).powi(2)
        }
        _ => x,
    }
}
pub fn price(params: &Value, item: &str, inv: f64) -> f64 {
    let p = &params[item];
    let base = p["base"].as_f64().unwrap();
    let i = p["I0"].as_f64().unwrap();
    let t = p["T"].as_f64().unwrap();
    let side = if inv < i { "below" } else { "above" };
    let f = p[format!("{side}_func")].as_str().unwrap_or("");
    let amp = p[format!("{side}_target")].as_f64().unwrap() * base / shape(f, t, t);
    (base + if inv < i { 1. } else { -1. } * amp * shape(f, (inv - i).abs(), t))
        .round_ties_even()
        .max(1.)
}
// Compile the immutable JSON pricing parameters once per environment. Keep the
// original arithmetic order (including the normalization division) so prices
// remain identical to the Python reference at rounding boundaries.
#[derive(Clone)]
pub struct MarketPricing {
    curves: [PriceCurve; PRODUCTS.len()],
}
#[derive(Clone, Copy)]
enum PriceShape {
    Linear,
    Square,
    Sqrt,
    Log,
    Log10,
    Hinge,
}
impl PriceShape {
    fn parse(name: &str) -> Self {
        match name {
            "sq" => Self::Square,
            "sqrt" => Self::Sqrt,
            "log" => Self::Log,
            "log10" => Self::Log10,
            "hinge" => Self::Hinge,
            _ => Self::Linear,
        }
    }
    fn evaluate(self, x: f64, t: f64) -> f64 {
        let x = x.max(0.);
        match self {
            Self::Square => x * x,
            Self::Sqrt => x.sqrt(),
            Self::Log => (1. + x).ln(),
            Self::Log10 => (1. + x).log10(),
            Self::Hinge if t > 0. => {
                let u = x / t;
                u + 8. * (u - 1.).max(0.).powi(2)
            }
            _ => x,
        }
    }
}
#[derive(Clone)]
struct PriceCurve {
    base: f64,
    initial: f64,
    threshold: f64,
    below: (PriceShape, f64),
    above: (PriceShape, f64),
}
impl MarketPricing {
    pub fn new(params: &Value) -> Self {
        Self {
            curves: std::array::from_fn(|idx| {
                let p = &params[PRODUCTS[idx]];
                let base = p["base"].as_f64().unwrap();
                let threshold = p["T"].as_f64().unwrap();
                let side = |func: &str, target: &str| {
                    let f = p[func].as_str().unwrap_or("");
                    (
                        PriceShape::parse(f),
                        p[target].as_f64().unwrap() * base / shape(f, threshold, threshold),
                    )
                };
                PriceCurve {
                    base,
                    initial: p["I0"].as_f64().unwrap(),
                    threshold,
                    below: side("below_func", "below_target"),
                    above: side("above_func", "above_target"),
                }
            }),
        }
    }
    pub fn price_at(&self, idx: usize, inv: f64) -> f64 {
        let p = &self.curves[idx];
        let below = inv < p.initial;
        let (f, amplitude) = if below { &p.below } else { &p.above };
        (p.base
            + if below { 1. } else { -1. }
                * amplitude
                * f.evaluate((inv - p.initial).abs(), p.threshold))
        .round_ties_even()
        .max(1.)
    }
    /// Base price of a good: `MARKET_PARAMS[g]["base"]`. Needed by the features
    /// (`encode`, column `pr / base`), and building a second parse of the
    /// parameters for it would mean creating a second copy of the table.
    pub fn base_of(&self, idx: usize) -> f64 {
        self.curves[idx].base
    }
    pub fn price(&self, item: &str, inv: f64) -> f64 {
        self.price_at(PRODUCTS.iter().position(|p| *p == item).unwrap(), inv)
    }
}
/// Shop table from the pinned constants. Needed by the features (`encode`:
/// `town_rate`, `town_ahead`, `SHOP_W`), and there must not be a second copy of it.
pub fn shops_def() -> &'static Value {
    &constants()["SHOPS"]
}
pub fn empty_action() -> Value {
    json!({"farmer":["PASS"]})
}

pub fn number(v: &Value) -> Number {
    match v {
        Value::Number(n) => n.clone(),
        _ => unreachable!("validated numeric market parameter"),
    }
}
pub fn market_amount(m: &Market, item: &str) -> f64 {
    m.inventory[item].as_f64().unwrap()
}
pub fn adjust_market(m: &mut Market, item: &str, delta: i64) {
    let n = m.inventory.get_mut(item).unwrap();
    *n = if let Some(i) = n.as_i64() {
        if let Some(sum) = i.checked_add(delta) {
            Number::from(sum)
        } else {
            Number::from_f64(i as f64 + delta as f64).unwrap()
        }
    } else {
        Number::from_f64(n.as_f64().unwrap() + delta as f64).unwrap()
    };
}
pub fn price_number(v: f64) -> Number {
    if v < i64::MAX as f64 {
        Number::from(v as i64)
    } else {
        Number::from_f64(v).unwrap()
    }
}

// Pythonize maps Serde tuples to Python tuples. Kaggle uses lists everywhere,
// so fixed-size arrays are serialized explicitly as sequences at the boundary.
pub fn serialize_slice<S: serde::Serializer, T: Serialize>(
    items: &[T],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    items.serialize(serializer)
}
fn serialize_positions<S: serde::Serializer>(
    items: &[[usize; 2]],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(items.len()))?;
    for pos in items {
        seq.serialize_element(pos.as_slice())?;
    }
    seq.end()
}

#[cfg(test)]
mod pricing_tests {
    use super::*;

    #[test]
    fn compiled_prices_match_original_float_bytes() {
        let mut variants = vec![constants()["MARKET_PARAMS"].clone()];
        for function in ["sq", "sqrt", "log", "log10", "hinge", "linear", "unknown"] {
            let mut params = constants()["MARKET_PARAMS"].clone();
            for item in PRODUCTS {
                params[item]["base"] = json!(2.5);
                params[item]["I0"] = json!(20.5);
                params[item]["T"] = json!(0.1);
                params[item]["below_func"] = json!(function);
                params[item]["above_func"] = json!(function);
                params[item]["above_target"] = json!(2.0);
            }
            variants.push(params);
        }
        for params in variants {
            let pricing = MarketPricing::new(&params);
            for item in PRODUCTS {
                let initial = params[item]["I0"].as_f64().unwrap();
                for n in -2000..=2000 {
                    let inventory = initial + n as f64 / 4.;
                    let expected = price(&params, item, inventory).to_le_bytes();
                    let actual = pricing.price(item, inventory).to_le_bytes();
                    assert_eq!(actual, expected, "{item} inventory={inventory}");
                }
            }
        }
    }
}
