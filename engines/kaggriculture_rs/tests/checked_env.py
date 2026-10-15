"""Rust API with independent Python oracles, audited by fixture at test end."""
from kaggriculture_rs import Env as RustEnv, BatchEnv as RustBatchEnv
from oracle import Oracle
from bytecheck import assert_bytes_equal

TRACKED = []
TRANSITION_AUDITS = {'checks': 0, 'bytes_compared': 0}


def check_transition(actual, expected, label):
    size = assert_bytes_equal(actual, expected, label)
    TRANSITION_AUDITS['checks'] += 1
    TRANSITION_AUDITS['bytes_compared'] += size


class Env:
    def __init__(self, config=None, seed=0):
        self.inner = RustEnv(config, seed)
        self.oracle = Oracle(config, seed)
        TRACKED.append(self)
        check_transition(self.inner.snapshot(), self.oracle.result(), 'Env.initial')

    def reset(self, seed):
        observations = self.inner.reset(seed)
        config = dict(self.oracle.configuration)
        self.oracle = Oracle(config, seed)
        check_transition(observations, self.oracle.observations(), 'Env.reset')
        return observations

    def step(self, actions):
        result = self.inner.step(actions)
        expected = self.oracle.step(actions)
        check_transition(result, expected, f'Env.step-{self.oracle.state[0].observation.step}')
        return result

    def bytecheck(self, label):
        return assert_bytes_equal(self.inner.snapshot(), self.oracle.result(), label)


class BatchEnv:
    def __init__(self, num_envs, config=None, num_threads=None):
        self.inner = RustBatchEnv(num_envs, config, num_threads)
        self.config = config
        self.oracles = [Oracle(config, i) for i in range(num_envs)]
        TRACKED.append(self)
        check_transition(self.inner.snapshot(), [o.result() for o in self.oracles], 'BatchEnv.initial')

    def reset(self, seeds):
        observations = self.inner.reset(seeds)
        self.oracles = [Oracle(self.config, seed) for seed in seeds]
        check_transition(observations, [o.observations() for o in self.oracles], 'BatchEnv.reset')
        return observations

    def step(self, actions):
        results = self.inner.step(actions)
        expected = [oracle.step(pair) for oracle, pair in zip(self.oracles, actions)]
        check_transition(results, expected, 'BatchEnv.step')
        return results

    def reset_at(self, indices, seeds):
        observations = self.inner.reset_at(indices, seeds)
        for i, seed, pair in zip(indices, seeds, observations):
            self.oracles[i] = Oracle(self.config, seed)
        check_transition(observations, [self.oracles[i].observations() for i in indices], 'BatchEnv.reset_at')
        return observations

    def bytecheck(self, label):
        return assert_bytes_equal(self.inner.snapshot(), [o.result() for o in self.oracles], label)
