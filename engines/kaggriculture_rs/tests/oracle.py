"""Pinned interpreter, with only the Kaggle framework adapter replaced.

No mechanics are reimplemented here. The adapter follows reference/core.py:
initial step=0, interpreter sees previous step, then increment, terminal score
comes from the interpreter. Wall-clock agent timeouts are outside this API.
"""
import ast
import copy
import json
import random
import types
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

class Struct(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None
    def __setattr__(self, key, value):
        self[key] = value

# Execute the exact upstream seed resolver, avoiding optional Kaggle dependencies.
utils = ast.parse((ROOT / 'reference/utils.py').read_text())
resolver = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_episode_seed')
ns = {'Any': Any, 'Callable': Callable, 'random': random}
exec(compile(ast.Module(body=[resolver], type_ignores=[]), 'reference/utils.py', 'exec'), ns)
source = ast.parse((ROOT / 'reference/kaggriculture.py').read_text())
source.body = [n for n in source.body if not (isinstance(n, ast.ImportFrom) and n.module == 'kaggle_environments.utils')]
module = types.ModuleType('pinned_kaggriculture')
module.__file__ = str(ROOT / 'reference/kaggriculture.py')
module.resolve_episode_seed = ns['resolve_episode_seed']
exec(compile(source, module.__file__, 'exec'), module.__dict__)

class Oracle:
    def __init__(self, config=None, seed=0):
        spec = json.loads((ROOT / 'reference/kaggriculture.json').read_text())
        self.configuration = Struct({k: v.get('default') if isinstance(v, dict) else v for k, v in spec['configuration'].items()})
        self.configuration.update(config or {})
        self.configuration.seed = seed
        self.info = {}
        self.done = False
        self.state = [Struct(observation=Struct(step=0, remainingOverageTime=60), action={}, reward=0, status='ACTIVE') for _ in range(2)]
        module.interpreter(self.state, self)

    def observations(self):
        return copy.deepcopy([s.observation for s in self.state])

    def result(self):
        return dict(observations=self.observations(), rewards=[s.reward for s in self.state], statuses=[s.status for s in self.state], done=self.done)

    def step(self, actions):
        if self.done:
            return self.result()
        step = self.state[0].observation.step
        for s, a in zip(self.state, actions):
            s.action = a
        module.interpreter(self.state, self)
        for s in self.state:
            s.observation.step = step + 1
        self.done = all(s.status == 'DONE' for s in self.state)
        return self.result()


def assert_same(a, b, path='root'):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            assert_same(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f'{path}[{i}]')
    else:
        assert a == b, (path, a, b)
