"""Third-pass regression scenarios, with mandatory per-transition byte audits."""
import pytest
from checked_env import Env

PASS = {'farmer': ['PASS']}


@pytest.mark.parametrize('money', [0, 13, 1000000, 2**60])
def test_fixed_purchases_beside_dynamic_orders(money):
    env = Env({'startingMoney': money, 'shedCapacity': 3,
               'episodeSteps': 12, 'turnsPerDay': 3,
               'marketParams': {'WHEAT': {'base': 2.5, 'I0': 20.5}}}, -15)
    script = [
        [{'market': [['BUY_SEED', 'WHEAT', 100003], ['BUY_ANIMAL', 'GOOSE', 100003]]},
         {'market': [['BUY_PRODUCT', 'WHEAT', 10], ['SELL', 'WHEAT', 5]]}],
        [{'market': [['BUY_PRODUCT', 'WHEAT', 4], ['SELL', 'WHEAT', 4]]},
         {'market': [['BUY_SEED', 'CARROT', 7], ['BUY_ANIMAL', 'SHEEP', 4]]}],
        [{'farmer': ['PICKUP', 'GOOSE', 2],
          'market': [['BUY_ANIMAL', 'COW', 4], ['BUY_SEED', 'MELON', 7]]},
         {'market': [['BUY_SEED', 'UNKNOWN', 4], ['BUY_ANIMAL', 'UNKNOWN', 2]]}],
    ]
    for pair in script:
        env.step(pair)
    while not env.oracle.done:
        env.step([PASS, PASS])


@pytest.mark.parametrize('weed', [0.0, 1.0])
def test_daily_rng_with_no_draw_days_and_short_days(weed):
    env = Env({'boardSize': 4, 'turnsPerDay': 1, 'episodeSteps': 80,
               'weedSpawnChance': weed, 'townShopUnlockInterval': 1}, 2**512 + 11)
    for step in range(79):
        # With weed=1 the initial empty cells fill, then shops reach their limit:
        # most days need no RNG; digging creates a new random draw later.
        action = {'farmer': ['DIG']} if step in (19, 35, 61) else PASS
        env.step([action, PASS])


def test_farmer_only_and_phantom_hand_plant_demand():
    env = Env({'episodeSteps': 8, 'weedSpawnChance': 0})
    env.step([{'market': [['BUY_SEED', 'WHEAT', 1]]}, PASS])
    # The reference counts even an action for a non-existent hand when deciding
    # whether to block all requests for a crop.
    result = env.step([{'farmer': ['PLANT', 'WHEAT'],
                        'hands': [['PLANT', 'WHEAT']]}, PASS])
    assert result['observations'][0]['private']['seeds']['WHEAT'] == 1
    result = env.step([{'farmer': ['PLANT', 'WHEAT'], 'hands': []}, PASS])
    assert result['observations'][0]['private']['seeds']['WHEAT'] == 0
    while not env.oracle.done:
        env.step([PASS, PASS])
