import pytest
from framework import make, result
from oracle import Oracle, assert_same, module
from checked_env import Env
from bytecheck import assert_bytes_equal

@pytest.mark.parametrize('steps',[1,2,25,720])
def test_actual_framework(steps):
    config={'episodeSteps':steps}
    framework=make(config,seed=14)
    oracle=Oracle(config,seed=14)
    rust=Env(config,seed=14)
    assert_bytes_equal(rust.reset(14),result(framework)['observations'], 'framework.reset')
    while not framework.done:
        actions=[module.starter_agent(o) for o in result(framework)['observations']]
        framework.step(actions)
        expected=result(framework)
        assert_same(oracle.step(actions),expected)
        assert_bytes_equal(rust.step(actions),expected, 'framework.step')
