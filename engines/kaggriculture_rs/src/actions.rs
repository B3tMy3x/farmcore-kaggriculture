use crate::model::*;
use serde_json::Value;

pub fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|f| f.is_finite() && *f < i64::MAX as f64 && *f >= i64::MIN as f64)
                .map(|f| f as i64)
        }),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}
pub fn unit(f: &mut Farm, p: &mut Private, idx: usize, action: &Value, c: &Config, day: i64) {
    let Some(a) = action.as_array().filter(|a| !a.is_empty()) else {
        return;
    };
    let op = a[0].as_str().unwrap_or("");
    let pos = if idx == 0 {
        f.farmer
    } else if let Some(x) = f.hands.get(idx - 1) {
        *x
    } else {
        return;
    };
    while p.inventories.len() <= idx {
        p.inventories.push(Inventory::new());
    }
    let inv = &mut p.inventories[idx];
    let [x, y] = pos;
    let delta = match op {
        "NORTH" => Some((0, -1)),
        "SOUTH" => Some((0, 1)),
        "EAST" => Some((1, 0)),
        "WEST" => Some((-1, 0)),
        _ => None,
    };
    if let Some((dx, dy)) = delta {
        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
        if nx >= 0 && ny >= 0 && nx < c.board as i64 && ny < c.board as i64 {
            let pos = [nx as usize, ny as usize];
            if idx == 0 {
                f.farmer = pos;
            } else {
                f.hands[idx - 1] = pos;
            }
        }
        return;
    }
    if op == "PASS" {
        return;
    }
    let adjacent = access(c.board).contains(&pos);
    let item = a.get(1).and_then(Value::as_str).unwrap_or("");
    let quantity = || a.get(2).map(py_int).unwrap_or(Some(1)).unwrap_or(0);
    let tile = &mut f.tiles[y][x];
    match op {
        "DROP" => {
            if adjacent {
                drop_inv(inv, &mut p.shed, c.cap);
            }
            return;
        }
        "PICKUP" => {
            if adjacent {
                let n = quantity().min(count(&p.shed, item));
                if n > 0 {
                    *p.shed.get_mut(item).unwrap() -= n;
                    add(inv, item, n);
                }
            }
            return;
        }
        "PLACE" => {
            let matching = ANIMALS.contains(&item)
                && matches!(&*tile,Tile::Occupied(Occupant::Coop{animal:None}) if animal_rules(item).structure=="COOP")
                || ANIMALS.contains(&item)
                    && matches!(&*tile,Tile::Occupied(Occupant::Pasture{animal:None}) if animal_rules(item).structure=="PASTURE");
            if matching {
                if take(inv, item, 1) {
                    let animal = Some(Animal {
                        animal: item.into(),
                        placed_day: day,
                        yield_units: 0,
                        consecutive_unfed: 0,
                        fed_today: false,
                        cared_today: false,
                        fertilizer_available: false,
                        pending_care_bonus: 0,
                    });
                    *tile = Tile::Occupied(if item == "GOOSE" {
                        Occupant::Coop { animal }
                    } else {
                        Occupant::Pasture { animal }
                    });
                }
                return;
            }
            if adjacent {
                let n = quantity()
                    .min(count(inv, item))
                    .min((c.cap - p.shed.values().sum::<i64>()).max(0));
                if n > 0 {
                    take(inv, item, n);
                    add(&mut p.shed, item, n);
                }
            }
            return;
        }
        _ => (),
    }
    if matches!(tile, Tile::Locked(_)) {
        return;
    }
    match op {
        "PLANT" => {
            if CROPS.contains(&item) && matches!(tile, Tile::Empty) && count(&p.seeds, item) > 0 {
                *p.seeds.get_mut(item).unwrap() -= 1;
                let rules = crop_rules(item);
                *tile = Tile::Occupied(Occupant::Plant(Plant {
                    crop: item.into(),
                    planted_day: day,
                    watered_today: false,
                    consecutive_unwatered: 1,
                    yield_units: if rules.ongoing { 0 } else { 1 },
                    max_lifespan_step: if rules.ongoing {
                        -1
                    } else {
                        (day + rules.max_yield_day + 1) * c.turns
                    },
                    fertilized_until_day: -1,
                }));
            }
        }
        "WATER" => {
            if let Tile::Occupied(Occupant::Plant(t)) = tile {
                if !t.watered_today {
                    t.watered_today = true;
                    let rules = crop_rules(&t.crop);
                    let max = rules.max_yield_day;
                    let age = day - t.planted_day;
                    if !rules.ongoing && age >= (max + 1) / 2 && age <= max {
                        t.yield_units = (t.yield_units
                            + if t.fertilized_until_day >= day { 2 } else { 1 })
                        .min(rules.max_yield);
                    }
                }
            }
        }
        "HARVEST" => {
            if let Tile::Occupied(Occupant::Plant(t)) = tile {
                let rules = crop_rules(&t.crop);
                if t.yield_units > 0 && day - t.planted_day >= rules.first_yield_day {
                    add(inv, &t.crop, t.yield_units);
                    t.yield_units = 0;
                    if !rules.ongoing {
                        *tile = Tile::Empty;
                    }
                }
            } else if let Some(t) = tile.animal_mut() {
                if t.yield_units > 0 {
                    add(inv, animal_rules(&t.animal).product, t.yield_units);
                    t.yield_units = 0;
                }
            }
        }
        "FERTILIZE" => {
            if let Tile::Occupied(Occupant::Plant(t)) = tile {
                if take(inv, "FERTILIZER", 1) {
                    t.fertilized_until_day = t.fertilized_until_day.max(day + 2);
                }
            }
        }
        "DIG" => {
            if !tile.has_animal() {
                *tile = Tile::Empty;
            }
        }
        "BUILD_COOP" | "BUILD_PASTURE" => {
            if matches!(tile, Tile::Empty) {
                *tile = Tile::Occupied(if op == "BUILD_COOP" {
                    Occupant::Coop { animal: None }
                } else {
                    Occupant::Pasture { animal: None }
                });
            }
        }
        "FEED" => {
            if let Some(t) = tile.animal_mut() {
                if !t.fed_today && take(inv, "WHEAT", 1) {
                    t.fed_today = true;
                }
            }
        }
        "CARE" => {
            if let Some(t) = tile.animal_mut() {
                t.cared_today = true;
            }
        }
        "COLLECT_FERTILIZER" => {
            if let Some(t) = tile.animal_mut() {
                if t.fertilizer_available {
                    t.fertilizer_available = false;
                    add(inv, "FERTILIZER", 1);
                }
            }
        }
        _ => (),
    }
}
struct Order<'a> {
    op: &'a str,
    item: &'a str,
    n: i64,
}
fn parse(v: &Value) -> Option<Order<'_>> {
    let a = v.as_array()?;
    let op = a.first()?.as_str()?;
    if matches!(op, "HIRE" | "BUY_LAND") {
        return Some(Order { op, item: "", n: 1 });
    }
    if !matches!(op, "SELL" | "BUY_PRODUCT" | "BUY_ANIMAL" | "BUY_SEED") {
        return None;
    }
    let n = py_int(a.get(2)?)?;
    if n <= 0 {
        return None;
    }
    Some(Order {
        op,
        item: a.get(1)?.as_str()?,
        n,
    })
}
fn hire(f: &mut Farm, p: &mut Private, c: &Config) {
    let cost = if c.hire == 0 {
        0.
    } else {
        let (mut a, mut b) = (1u128, 1u128);
        for _ in 0..f.hires_today {
            let Some(n) = a.checked_add(b) else {
                return;
            };
            a = b;
            b = n;
        }
        a as f64 * c.hire as f64
    };
    if f.money < cost {
        return;
    }
    f.money -= cost;
    f.hires_today += 1;
    let pos = access(c.board)
        .into_iter()
        .min_by_key(|pos| {
            std::iter::once(&f.farmer)
                .chain(f.hands.iter())
                .filter(|p| *p == pos)
                .count()
        })
        .unwrap();
    f.hands.push(pos);
    p.inventories.push(Inventory::new());
}
fn land(f: &mut Farm, c: &Config) {
    let n = f.unlocked_quadrants.len() - 1;
    if n >= 3 {
        return;
    }
    let cost = [1000., 2000., 4000.][n];
    if f.money < cost {
        return;
    }
    f.money -= cost;
    let q = ["NE", "SW", "SE"][n];
    f.unlocked_quadrants.push(q.into());
    for y in 0..c.board {
        for x in 0..c.board {
            if quadrant_name(x, y, c.board) == q && matches!(f.tiles[y][x], Tile::Locked(_)) {
                f.tiles[y][x] = Tile::Empty;
            }
        }
    }
}
pub fn refresh(m: &mut Market, pricing: &MarketPricing) {
    for (idx, item) in PRODUCTS.iter().enumerate() {
        let price = price_number(pricing.price_at(idx, market_amount(m, item)));
        *m.prices.get_mut(*item).unwrap() = price;
    }
}
// Fixed purchases affect only this player's money/inventory. They can finish
// independently of the opposing order, without changing any market quote.
// Keep each subtraction: money - (price * quantity) is not bit equivalent.
fn fixed_purchase(
    money: &mut f64,
    inventory: &mut Inventory,
    item: &str,
    price: f64,
    quantity: i64,
) -> i64 {
    if quantity <= 0 || *money < price {
        return 0;
    }
    let count = match inventory.get_index_of(item) {
        Some(index) => inventory.get_index_mut(index).unwrap().1,
        None => inventory.entry(item.into()).or_default(),
    };
    let mut bought = 0;
    for _ in 0..quantity.min(99_999) {
        if *money < price {
            break;
        }
        *money -= price;
        *count += 1;
        bought += 1;
    }
    bought
}
pub fn market(
    f: &mut [Farm; 2],
    p: &mut [Private; 2],
    m: &mut Market,
    actions: &[Value; 2],
    c: &Config,
    pricing: &MarketPricing,
) {
    let queues: [&[Value]; 2] = std::array::from_fn(|i| {
        actions[i]
            .get("market")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    });
    let max = queues
        .iter()
        .map(|q| q.len().min(c.orders))
        .max()
        .unwrap_or(0);
    if max == 0 {
        return;
    }
    let mut shed_used: [i64; 2] = std::array::from_fn(|j| p[j].shed.values().sum());
    for i in 0..max {
        let mut orders: [Option<Order<'_>>; 2] =
            std::array::from_fn(|j| queues[j].get(i).and_then(parse));
        for j in 0..2 {
            if let Some(o) = &orders[j] {
                match o.op {
                    "HIRE" => {
                        hire(&mut f[j], &mut p[j], c);
                        orders[j] = None;
                    }
                    "BUY_LAND" => {
                        land(&mut f[j], c);
                        orders[j] = None;
                    }
                    _ => (),
                }
            }
        }
        // Resolve fixed prices and product indices once per order. Dynamic
        // market prices still use the common inventory before either commit.
        enum Quote {
            Sell(usize),
            Buy(usize),
        }
        for j in 0..2 {
            if let Some(o) = &orders[j] {
                if o.op == "BUY_SEED" && CROPS.contains(&o.item) {
                    fixed_purchase(
                        &mut f[j].money,
                        &mut p[j].seeds,
                        o.item,
                        crop_rules(o.item).seed as f64,
                        o.n,
                    );
                    orders[j] = None;
                } else if o.op == "BUY_ANIMAL" && ANIMALS.contains(&o.item) {
                    let quantity = o.n.min((c.cap - shed_used[j]).max(0));
                    shed_used[j] += fixed_purchase(
                        &mut f[j].money,
                        &mut p[j].shed,
                        o.item,
                        animal_rules(o.item).cost as f64,
                        quantity,
                    );
                    orders[j] = None;
                }
            }
        }
        let quotes: [Option<Quote>; 2] = std::array::from_fn(|j| {
            let o = orders[j].as_ref()?;
            match o.op {
                "SELL" => PRODUCTS
                    .iter()
                    .position(|item| *item == o.item)
                    .map(Quote::Sell),
                "BUY_PRODUCT" if matches!(o.item, "WHEAT" | "FERTILIZER") => Some(Quote::Buy(
                    PRODUCTS.iter().position(|item| *item == o.item).unwrap(),
                )),
                _ => None,
            }
        });
        for _ in 0..99999 {
            let prices: [Option<f64>; 2] = std::array::from_fn(|j| {
                let o = orders[j].as_ref()?;
                if o.n <= 0 {
                    return None;
                }
                Some(match quotes[j].as_ref()? {
                    Quote::Sell(idx) => pricing.price_at(*idx, market_amount(m, o.item)),
                    Quote::Buy(idx) => pricing.price_at(*idx, market_amount(m, o.item) - 1.),
                })
            });
            let mut committed = false;
            for j in 0..2 {
                let Some(price) = prices[j] else {
                    orders[j] = None;
                    continue;
                };
                let o = orders[j].as_mut().unwrap();
                let item = o.item;
                let ok = if o.op == "SELL" {
                    if count(&p[j].shed, item) <= 0 {
                        false
                    } else {
                        *p[j].shed.get_mut(item).unwrap() -= 1;
                        shed_used[j] -= 1;
                        f[j].money += price;
                        if price > 1. {
                            adjust_market(m, item, 1);
                        }
                        true
                    }
                } else if f[j].money < price || shed_used[j] >= c.cap {
                    false
                } else {
                    f[j].money -= price;
                    add(&mut p[j].shed, item, 1);
                    shed_used[j] += 1;
                    adjust_market(m, item, -1);
                    true
                };
                if ok {
                    o.n -= 1;
                    committed = true;
                } else {
                    orders[j] = None;
                }
            }
            if !committed {
                break;
            }
        }
    }
    // Prices are read directly from inventory during matching; publish the
    // final table once after all orders have completed.
    refresh(m, pricing);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_purchases_preserve_scalar_money_bytes_and_inventory_order() {
        // At large balances subtracting 10 repeatedly differs from subtracting
        // 10 * quantity once. Also cover insufficient funds and the loop cap.
        for initial in [0., 9.5, 10., 29.75, 1e6, 2f64.powi(60)] {
            for quantity in [0, 1, 17, 100_003] {
                for existing in [false, true] {
                    let mut expected_money = initial;
                    let mut expected = Inventory::new();
                    add(&mut expected, "BEFORE", 3);
                    if existing {
                        add(&mut expected, "WHEAT", 7);
                    }
                    add(&mut expected, "AFTER", 5);
                    let mut actual = expected.clone();
                    let mut expected_bought = 0;
                    for _ in 0..quantity.min(99_999) {
                        if expected_money < 10. {
                            break;
                        }
                        expected_money -= 10.;
                        add(&mut expected, "WHEAT", 1);
                        expected_bought += 1;
                    }
                    let mut actual_money = initial;
                    let actual_bought =
                        fixed_purchase(&mut actual_money, &mut actual, "WHEAT", 10., quantity);
                    assert_eq!(actual_bought, expected_bought);
                    assert_eq!(actual_money.to_le_bytes(), expected_money.to_le_bytes());
                    assert_eq!(
                        serde_json::to_vec(&actual).unwrap(),
                        serde_json::to_vec(&expected).unwrap()
                    );
                }
            }
        }
    }
}
