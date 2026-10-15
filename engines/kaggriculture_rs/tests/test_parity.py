import random
import pytest
from checked_env import Env, BatchEnv
from oracle import Oracle, module, assert_same

PASS = {'farmer': ['PASS']}

def run_script(actions, config=None, seed=0, finish=True):
    oracle = Oracle(config, seed)
    rust = Env(config, seed)
    assert_same(rust.reset(seed), oracle.observations())
    for pair in actions:
        assert_same(rust.step(pair), oracle.step(pair))
    if finish:
        while not oracle.done:
            assert_same(rust.step([PASS, PASS]), oracle.step([PASS, PASS]))
    return oracle

@pytest.mark.parametrize('seed', [0, 1, -1, 1234567, -(2**100 + 123), 2**256 + 27])
def test_seed_and_terminal(seed):
    o = run_script([], seed=seed)
    assert o.state[0].observation.step == 719

@pytest.mark.parametrize('seed', range(100))
def test_full_games(seed):
    rng = random.Random(seed)
    cfg = {} if seed % 2 == 0 else {'turnsPerDay': 8, 'startingMoney': 20000, 'boardSize': 5 + seed % 4, 'shedCapacity': 7 + seed % 20, 'townShopUnlockInterval': 1}
    oracle, rust = Oracle(cfg, seed), Env(cfg, seed)
    products = list(module.PRODUCTS)
    crops = list(module.CROPS)
    animals = list(module.ANIMALS)
    ops = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'DIG', 'HARVEST', 'WATER', 'FEED', 'CARE', 'COLLECT_FERTILIZER', 'FERTILIZE', 'BUILD_COOP', 'BUILD_PASTURE', 'DROP', 'PASS']
    def unit(obs, position):
        x, y = position
        tile = obs['farms'][obs['player']]['tiles'][y][x]
        if rng.random() < .65:
            if tile is None:
                return ['PLANT', rng.choice(crops)]
            if isinstance(tile, dict) and tile.get('kind') == 'PLANT':
                return [rng.choice(['WATER', 'HARVEST', 'FERTILIZE', 'DIG'])]
        return rng.choice([[rng.choice(ops)], ['PLANT', rng.choice(crops)], ['PICKUP', rng.choice(products + animals), rng.randint(-1, 10)], ['PLACE', rng.choice(products + animals), rng.randint(-1, 10)]])
    while not oracle.done:
        pairs = []
        for obs in oracle.observations():
            f = obs['farms'][obs['player']]
            orders = []
            for _ in range(rng.randrange(5)):
                orders.append(rng.choice([
                    ['BUY_SEED', rng.choice(crops), rng.randrange(8)],
                    ['BUY_PRODUCT', rng.choice(products), rng.randrange(10)],
                    ['SELL', rng.choice(products), rng.randrange(10)],
                    ['BUY_ANIMAL', rng.choice(animals), rng.randrange(3)],
                    ['HIRE'], ['BUY_LAND']]))
            pairs.append({'farmer': unit(obs, f['farmer']), 'hands': [unit(obs, p) for p in f['hands']], 'market': orders})
        actual, expected = rust.step(pairs), oracle.step(pairs)
        if actual != expected:
            assert_same(actual, expected, f'seed={seed},step={oracle.state[0].observation.step}')

@pytest.mark.parametrize('crop', module.CROPS)
def test_crop_lifecycle(crop):
    cfg = {'turnsPerDay': 12, 'startingMoney': 100000, 'weedSpawnChance': 0}
    oracle, rust = Oracle(cfg), Env(cfg)
    for step in range(719):
        obs = oracle.observations()[0]
        day, hour = obs['day'], obs['hour']
        if step == 0:
            a = {'market': [['BUY_SEED', crop, 10], ['BUY_PRODUCT', 'FERTILIZER', 15]]}
        elif step == 1: a = {'farmer': ['PICKUP', 'FERTILIZER', 15]}
        elif step == 2: a = {'farmer': ['PLANT', crop]}
        elif step == 3: a = {'farmer': ['WATER']}
        else:
            op = 'PASS'
            if hour == 0 and day < 22: op = 'WATER'
            if hour == 1 and day % 3 == 0: op = 'FERTILIZE'
            if hour == 3 and day in [4, 11, 13, 16]: op = 'HARVEST'
            a = {'farmer': [op], 'market': [['SELL', crop, 100]]}
        assert_same(rust.step([a, PASS]), oracle.step([a, PASS]))

@pytest.mark.parametrize('animal', module.ANIMALS)
def test_animal_lifecycle(animal):
    cfg = {'turnsPerDay': 12, 'startingMoney': 100000, 'weedSpawnChance': 0}
    oracle, rust = Oracle(cfg), Env(cfg)
    for step in range(719):
        day, hour = divmod(step, 12)
        if step == 0: a = {'market': [['BUY_ANIMAL', animal, 1], ['BUY_PRODUCT', 'WHEAT', 50]], 'farmer': ['BUILD_COOP' if animal == 'GOOSE' else 'BUILD_PASTURE']}
        elif step == 1: a = {'farmer': ['PICKUP', animal]}
        elif step == 2: a = {'farmer': ['PLACE', animal]}
        elif hour == 3: a = {'farmer': ['PICKUP', 'WHEAT', 1]}
        elif hour == 4 and day < 25: a = {'farmer': ['FEED']}
        elif hour == 5: a = {'farmer': ['CARE']}
        elif hour == 6: a = {'farmer': ['COLLECT_FERTILIZER']}
        elif hour == 7: a = {'farmer': ['HARVEST']}
        elif hour == 8: a = {'farmer': ['DIG']}
        else: a = {'market': [['SELL', module.ANIMALS[animal]['product'], 100], ['SELL', 'FERTILIZER', 100]]}
        assert_same(rust.step([a, PASS]), oracle.step([a, PASS]))


def test_atomic_plant_and_inventory_order():
    a = [
        {'market': [['HIRE'], ['HIRE'], ['BUY_SEED', 'WHEAT', 1], ['BUY_LAND'], ['BUY_PRODUCT', 'WHEAT', 5], ['BUY_PRODUCT', 'FERTILIZER', 5]]},
        {'farmer': ['PLANT', 'WHEAT'], 'hands': [['PLANT', 'WHEAT']]},
        {'farmer': ['PICKUP', 'FERTILIZER', 5], 'hands': [['PICKUP', 'WHEAT', 5]]},
        {'market': [['BUY_PRODUCT', 'WHEAT', 9]], 'farmer': ['SOUTH']},
        {'farmer': ['NORTH'], 'hands': [['DROP']]},
        {'farmer': ['DROP']},
    ]
    o = run_script([[x, x] for x in a], {'startingMoney': 20000, 'shedCapacity': 10}, finish=False)
    assert o.state[0].observation.farms[0]['tiles'][4][4] is None
    assert o.state[0].observation.private['shed']['WHEAT'] == 10
    assert o.state[0].observation.private['shed']['FERTILIZER'] == 0

@pytest.mark.parametrize('shape', ['linear','sq','sqrt','log','log10','hinge','unknown'])
def test_market_curves(shape):
    cfg = {'startingMoney': 1000000, 'marketParams': {'WHEAT': {'I0': 20, 'T': 4, 'below_func': shape, 'above_func': shape, 'above_target': 2.0}, 'FERTILIZER': {'base': 2.5, 'T': 4}}, 'shedCapacity': 10000}
    buy={'market': [['BUY_PRODUCT', 'WHEAT', 30], ['BUY_PRODUCT', 'FERTILIZER', 10]]}
    sell={'market': [['SELL', 'WHEAT', 30], ['SELL', 'FERTILIZER', 10]]}
    run_script([[buy,buy],[sell,sell]]*10, cfg)


def test_malformed_noops():
    actions = [None, {}, {'farmer': []}, {'farmer': 'NORTH'}, {'farmer': ['NOT_AN_OP']}, {'hands': 'bad'}, {'market': [['SELL'], ['BUY_PRODUCT','WHEAT','bad'], [], ['BUY_SEED','NO_CROP',2], ['HIRE']]}, {'market': 'bad'}]
    run_script([[x,x] for x in actions])


def test_batch_isolation_and_errors():
    seeds = [0, 12, -123, 2**99]
    one = [Env(seed=s) for s in seeds]
    batches = [BatchEnv(4, num_threads=t) for t in (1,2,4)]
    for b in batches: assert b.reset(seeds) == [e.reset(s) for e,s in zip(one,seeds)]
    a = [[{'market': [['BUY_SEED','WHEAT',i+1]]},PASS] for i in range(4)]
    for _ in range(30):
        expected = [e.step(x) for e,x in zip(one,a)]
        for b in batches: assert b.step(a) == expected
    for b in batches:
        with pytest.raises(ValueError): b.step(a[:3])
        with pytest.raises(ValueError): b.step(a[:3]+[[PASS]])
        with pytest.raises(ValueError): b.reset_at([0,0],[1,2])
        with pytest.raises(ValueError): b.reset_at([4],[1])
        assert b.reset_at([2],[777]) == [Env().reset(777)]
    one[2].reset(777)
    expected=[e.step(x) for e,x in zip(one,a)]
    for b in batches: assert b.step(a)==expected
    expected[0]['observations'][0]['farms'][0]['money']=-1
    assert batches[0].step(a) == batches[1].step(a)


def test_short_terminal_and_reset():
    env=Env({'episodeSteps':2})
    result=env.step([PASS,PASS]);assert result['done']
    assert env.step([{'market':[['HIRE']]},PASS]) == result
    assert env.reset(4)[0]['step']==0

@pytest.mark.parametrize('config', [
    {'boardSize':4,'turnsPerDay':1,'episodeSteps':20,'farmHandCostMult':0},
    {'boardSize':7,'turnsPerDay':3,'episodeSteps':60,'weedSpawnChance':1.0},
    {'marketParams':{'WHEAT':{'I0':10000.5}}},
])
def test_configuration(config):
    run_script([[{'market':[['HIRE']]*12},PASS]]*10,config)


def test_batch_reset_errors_are_atomic():
    b=BatchEnv(2,num_threads=2)
    b.reset([4,5])
    one=[Env(seed=4),Env(seed=5)]
    with pytest.raises(ValueError):b.reset([99])
    with pytest.raises(ValueError):b.reset_at([0],[1,2])
    with pytest.raises(ValueError):b.reset_at([0,99],[1,2])
    assert b.step([[PASS,PASS]]*2)==[e.step([PASS,PASS])for e in one]
    assert b.reset_at([],[])==[]


def test_minimum_episode_steps():
    run_script([[PASS,PASS]], {'episodeSteps':1})


def test_drop_preserves_item_insertion_order():
    script=[
        {'market':[['BUY_PRODUCT','WHEAT',5],['BUY_PRODUCT','FERTILIZER',5]]},
        {'farmer':['PICKUP','FERTILIZER',5]},
        {'farmer':['PICKUP','WHEAT',5]},
        {'market':[['BUY_PRODUCT','WHEAT',9]]},
        {'farmer':['DROP']},
    ]
    o=run_script([[a,PASS]for a in script],{'shedCapacity':10},finish=False)
    shed=o.state[0].observation.private['shed']
    assert shed['FERTILIZER']==1 and shed['WHEAT']==9


def test_sales_at_price_floor_do_not_add_supply():
    cfg={'turnsPerDay':6,'townCenterSellInterval':100000,'townShopUnlockInterval':100000,'weedSpawnChance':0,'marketParams':{'CARROT':{'I0':10,'T':.1,'above_target':100.,'below_target':0.}}}
    oracle,rust=Oracle(cfg),Env(cfg)
    for step in range(24):
        if step==0: a={'market':[['BUY_SEED','CARROT',1]]}
        elif step==1:a={'farmer':['PLANT','CARROT']}
        elif step in [2,6,12,18]:a={'farmer':['WATER']}
        elif step==19:a={'farmer':['HARVEST']}
        elif step==20:a={'farmer':['DROP']}
        elif step==21:a={'market':[['SELL','CARROT',999]]}
        else:a=PASS
        assert_same(rust.step([a,PASS]),oracle.step([a,PASS]))
    market=oracle.state[0].observation.market
    assert market['inventory']['CARROT']==11
    assert market['prices']['CARROT']==1


def test_observation_number_types():
    rust=Env()
    obs=rust.reset(0)[0]
    assert type(obs['market']['inventory']['WHEAT']) is int
    assert type(obs['market']['prices']['WHEAT']) is int
    assert type(obs['farms'][0]['money']) is float
    r=rust.step([PASS,PASS])
    assert type(r['rewards'][0]) is int
    assert type(r['observations'][0]['market']['prices']['WHEAT']) is int
    rust=Env({'marketParams':{'WHEAT':{'I0':10000.0,'base':25.0}}})
    obs=rust.reset(0)[0]
    assert type(obs['market']['inventory']['WHEAT']) is float
    assert type(obs['market']['prices']['WHEAT']) is float
    obs=rust.step([PASS,PASS])['observations'][0]
    assert type(obs['market']['inventory']['WHEAT']) is float
    assert type(obs['market']['prices']['WHEAT']) is int


def test_market_iteration_limit():
    o=run_script([[{'market':[['BUY_SEED','WHEAT',100003]]},PASS]],{'startingMoney':3000000,'episodeSteps':2})
    assert o.state[0].observation.private['seeds']['WHEAT']==99999


def test_fertilizer_bonus_and_expiry():
    cfg={'turnsPerDay':6,'weedSpawnChance':0}
    oracle,rust=Oracle(cfg),Env(cfg)
    for step in range(20):
        if step==0:a={'market':[['BUY_SEED','WHEAT',1],['BUY_PRODUCT','FERTILIZER',1]]}
        elif step==1:a={'farmer':['PLANT','WHEAT']}
        elif step in [2,6,12,18]:a={'farmer':['WATER']}
        elif step==3:a={'farmer':['PICKUP','FERTILIZER',1]}
        elif step==4:a={'farmer':['FERTILIZE']}
        else:a=PASS
        result=rust.step([a,PASS])
        assert_same(result,oracle.step([a,PASS]))
        tile=result['observations'][0]['farms'][0]['tiles'][4][4]
        if step==4:assert tile['fertilized_until_day']==2
        if step==12:assert tile['yield_units']==3
        if step==18:assert tile['yield_units']==4


@pytest.mark.parametrize('threads', [1, 4])
def test_singleton_batch_reset_and_validation(threads):
    batch = BatchEnv(1, {'episodeSteps': 4}, num_threads=threads)
    batch.reset([2**100 + 3])
    before = batch.inner.snapshot()
    with pytest.raises(ValueError):
        batch.step([[PASS]])
    assert batch.inner.snapshot() == before
    for _ in range(5):
        batch.step([[{'market': [['BUY_SEED', 'CARROT', 2]]}, PASS]])
    batch.reset([-41])
    batch.step([[PASS, PASS]])


def test_market_refresh_after_idle_and_invalid_orders():
    config = {
        'episodeSteps': 90, 'turnsPerDay': 3, 'startingMoney': 100000,
        'townCenterSellInterval': 13, 'townShopSellInterval': 7,
        'townShopUnlockInterval': 1,
        'marketParams': {'WHEAT': {'base': 25.5, 'I0': 100.5}},
    }
    actions = [
        [PASS, PASS],
        [{'market': [['BUY_PRODUCT', 'WHEAT', 8], ['SELL', 'WHEAT', 3]]}, PASS],
        [{'market': [[], ['UNKNOWN'], ['HIRE']]}, PASS],
        [PASS, PASS],
    ] * 15
    run_script(actions, config=config, seed=-1234)
