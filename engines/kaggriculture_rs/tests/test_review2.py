"""Focused second-review regressions; checked_env audits every transition's bytes."""
import pytest
from checked_env import Env

PASS = {'farmer': ['PASS']}


@pytest.mark.parametrize('board,seed', [(4, 0), (5, -(2**129 + 31)), (7, 2**257 + 19)])
def test_mixed_lifecycles_keep_weed_rng_order(board, seed):
    """One board scan sees plants, animals, dead occupants and empty tiles."""
    env = Env({'boardSize': board, 'startingMoney': 100000,
               'turnsPerDay': 24, 'episodeSteps': 24 * 18 + 1,
               'shedCapacity': 500, 'weedSpawnChance': .31,
               'townShopUnlockInterval': 1}, seed)
    initial = {
        'market': [['BUY_LAND']] * 3 + [
            ['BUY_SEED', 'TOMATO', 1], ['BUY_SEED', 'STRAWBERRY', 1],
            ['BUY_ANIMAL', 'GOOSE', 1], ['BUY_ANIMAL', 'COW', 1],
            ['BUY_PRODUCT', 'WHEAT', 100], ['BUY_PRODUCT', 'FERTILIZER', 20]]}
    setup = [initial] + [{'farmer': a} for a in [
        ['DIG'], ['PLANT', 'TOMATO'], ['WATER'], ['EAST'], ['DIG'],
        ['BUILD_COOP'], ['PICKUP', 'GOOSE'], ['PLACE', 'GOOSE'],
        ['PICKUP', 'WHEAT', 20], ['FEED'], ['SOUTH'], ['DIG'],
        ['BUILD_PASTURE'], ['PICKUP', 'COW'], ['PLACE', 'COW'], ['FEED'],
        ['WEST'], ['DIG'], ['PLANT', 'STRAWBERRY'], ['WATER'],
        ['EAST'], ['CARE'], ['PASS']]]
    for action in setup:
        result = env.step([action, action])
    tiles = result['observations'][0]['farms'][0]['tiles']
    h = board // 2
    assert tiles[h - 1][h - 1]['crop'] == 'TOMATO'
    assert tiles[h - 1][h]['animal'] == 'GOOSE'
    assert tiles[h][h]['animal'] == 'COW'
    assert tiles[h][h - 1]['crop'] == 'STRAWBERRY'
    for day in range(1, 18):
        routine = [
            ['PICKUP', 'WHEAT', 2], ['WATER'], ['EAST'], ['FEED'],
            ['CARE'], ['HARVEST'], ['COLLECT_FERTILIZER'], ['SOUTH'],
            ['FEED'], ['CARE'], ['HARVEST'], ['COLLECT_FERTILIZER'],
            ['WEST'], ['WATER'], ['FERTILIZE'], ['HARVEST'], ['NORTH'],
            ['FERTILIZE'], ['HARVEST'], ['PASS'], ['PASS'], ['PASS'],
            ['PASS'], ['PASS']]
        # Death/removal paths coexist with still-growing crops; they must not
        # consume weed random draws or perturb the subsequent shop choice.
        if day >= 8:
            routine[3] = ['PASS']
        if day >= 12:
            routine[1] = ['PASS']
        for action in routine:
            result = env.step([{'farmer': action}, {'farmer': action}])
    assert result['done']
    tiles = result['observations'][0]['farms'][0]['tiles']
    assert tiles[h - 1][h] == {'kind': 'COOP'}
    assert tiles[h][h]['animal'] == 'COW'
    assert tiles[h - 1][h - 1] == {'kind': 'WEED'}


@pytest.mark.parametrize('initial', [20, 20.5])
def test_market_quotes_capacity_and_order_boundaries(initial):
    """Quotes must stay lockstep as sales free capacity for subsequent orders."""
    env = Env({'startingMoney': 100000, 'shedCapacity': 5,
               'episodeSteps': 40, 'turnsPerDay': 7, 'boardSize': 5,
               'maxMarketOrdersPerTurn': 4, 'townShopUnlockInterval': 1,
               'townShopSellInterval': 2, 'townCenterSellInterval': 3,
               'marketParams': {'WHEAT': {'base': 2.5, 'I0': initial, 'T': 2,
                                        'below_func': 'hinge', 'above_func': 'sq'},
                                'FERTILIZER': {'base': 1.0, 'I0': initial}}}, -79)
    actions = [
        [{'market': [['BUY_PRODUCT', 'WHEAT', 99]]},
         {'market': [['BUY_PRODUCT', 'FERTILIZER', 99]]}],
        [{'market': [['SELL', 'WHEAT', 3], ['BUY_ANIMAL', 'GOOSE', 3],
                     ['BUY_SEED', 'TOMATO', 4], ['SELL', 'WHEAT', 99],
                     ['HIRE']]},
         {'market': [['SELL', 'FERTILIZER', 3], ['BUY_PRODUCT', 'WHEAT', 3],
                     ['BUY_SEED', 'CARROT', 4], ['SELL', 'WHEAT', 99],
                     ['BUY_LAND']]}],
        [{'farmer': ['PICKUP', 'GOOSE', 3],
          'market': [['BUY_PRODUCT', 'WHEAT', 4], ['SELL', 'WHEAT', 2],
                     ['BUY_PRODUCT', 'FERTILIZER', 3], ['SELL', 'FERTILIZER', 2]]},
         {'farmer': ['PICKUP', 'FERTILIZER', 3],
          'market': [['BUY_PRODUCT', 'WHEAT', 4], ['SELL', 'WHEAT', 2],
                     ['BUY_PRODUCT', 'FERTILIZER', 3], ['SELL', 'FERTILIZER', 2]]}],
        [{'farmer': ['DROP']}, {'farmer': ['PLACE', 'FERTILIZER', 99]}],
    ]
    for pair in actions:
        env.step(pair)
    while not env.oracle.done:
        env.step([PASS, PASS])


def test_observation_output_has_independent_nested_containers():
    from copy import deepcopy
    from bytecheck import assert_bytes_equal

    env = Env({'startingMoney': 10000, 'weedSpawnChance': 0}, 33)
    env.step([{'market': [['HIRE'], ['BUY_SEED', 'WHEAT', 1]]}, PASS])
    result = env.step([{'farmer': ['PLANT', 'WHEAT']}, PASS])
    pristine = deepcopy(result)
    assert_bytes_equal(env.inner.snapshot(), pristine, 'snapshot equals returned result')
    obs0, obs1 = result['observations']
    farm = obs0['farms'][0]
    x, y = farm['farmer']
    assert isinstance(farm['money'], float)
    assert isinstance(farm['farmer'], list)
    assert isinstance(farm['hands'][0], list)
    assert isinstance(farm['tiles'][y][x]['yield_units'], int)
    assert isinstance(obs0['market']['prices']['WHEAT'], int)
    farm['money'] = -1
    farm['tiles'][y][x]['yield_units'] = -100
    farm['tiles'][y].append('bad row')
    farm['tiles'].append([])
    farm['farmer'][0] = -1
    farm['hands'][0][0] = -1
    farm['hands'].append([99, 99])
    obs0['farms'].append({})
    obs0['private']['seeds']['WHEAT'] = -1
    obs0['market']['inventory']['WHEAT'] = -1
    obs0['town']['unlocked_shops'].append('BAD')
    assert_bytes_equal(obs1, pristine['observations'][1], 'other player output independence')
    assert_bytes_equal(env.inner.snapshot(), pristine, 'snapshot isolation from output mutations')
    env.step([PASS, PASS])


def test_batch_outputs_and_reset_results_have_independent_containers():
    from copy import deepcopy
    from bytecheck import assert_bytes_equal
    from checked_env import BatchEnv

    env = BatchEnv(2, {'weedSpawnChance': 0}, num_threads=2)
    reset = env.reset([71, 71])
    original = deepcopy(reset)
    reset[0][0]['farms'][0]['farmer'][0] = -1
    assert_bytes_equal(reset[0][1], original[0][1], 'reset player isolation')
    assert_bytes_equal(reset[1], original[1], 'reset slot isolation')
    results = env.step([[{'farmer': ['BUILD_COOP']}, PASS]] * 2)
    pristine = deepcopy(results)
    x, y = results[0]['observations'][0]['farms'][0]['farmer']
    results[0]['observations'][0]['farms'][0]['tiles'][y][x]['kind'] = 'BAD'
    results[0]['observations'][0]['farms'][0]['tiles'][y].clear()
    results[0]['observations'][0]['market']['prices'].clear()
    results[0]['observations'][0]['private']['inventories'].append({'WHEAT': 999})
    assert_bytes_equal(results[0]['observations'][1], pristine[0]['observations'][1], 'batch player isolation')
    assert_bytes_equal(results[1], pristine[1], 'batch slot isolation')
    assert_bytes_equal(env.inner.snapshot(), pristine, 'batch snapshot isolation')
    env.step([[PASS, PASS]] * 2)
