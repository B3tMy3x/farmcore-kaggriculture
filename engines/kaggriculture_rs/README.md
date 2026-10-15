# Kaggriculture Rust engine

A deterministic simulator for self-play: Rust computes game transitions,
Python drives the policies. Independent games run in parallel in a
Rayon pool. The reference is pinned in [reference/](reference/README.md).

Latest check against the original Python: 166 Python tests, byte-for-byte
comparison on every transition. On a game strategy, Env is 14.09x faster than
the Python API and batch on four threads is 7.58x faster.

## Build

Requires Rust stable and Python 3.10+. From the project root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
maturin develop --release
```

For a wheel: `maturin build --release`. The runtime does not require Kaggle, PyTorch or
NumPy. `Cargo.lock` pins the Rust dependencies. The core can be built separately from
Python with `--no-default-features`.

## API

```python
from kaggriculture_rs import Env, BatchEnv

env = Env(config={"episodeSteps": 720}, seed=42)
observations = env.reset(42)
result = env.step([{"farmer": ["PASS"]}, {"farmer": ["PASS"]}])

batch = BatchEnv(32, num_threads=4)
observations = batch.reset(list(range(32)))
actions = [[{"farmer": ["PASS"]}, {"farmer": ["PASS"]}] for _ in range(32)]
results = batch.step(actions)
observations_for_reset_slots = batch.reset_at([3, 7], [1003, 1007])
```

- `Env(config=None, seed=0)`: `reset(seed)` returns
  two observations. In batch, the constructor sets initial seeds by index;
  for experiments call `reset(seeds)` explicitly.
- `snapshot()` returns the current result without executing a turn; batch
  returns a list of current results. The data are independent copies.
- `step` returns `observations` (two dicts), `rewards` (two numbers),
  `statuses` (two statuses) and `done`. Batch returns a list of such results
  in environment order. One call executes one joint turn of both players.
- `reset_at` returns observations in the order of the passed indices. The indices
  must be unique. Reset is allowed for an unfinished game too.
- There is no auto-reset. A repeated `step` on a finished environment returns its final
  result. Batch size and index errors are checked before any environment is modified.
- Each player sees both farms, the shared market and only its own `private`.
  Results are independent copies. The seed and the opponent's private data are not
  added to observations.
- `num_threads=None` uses the available CPU parallelism. The pool belongs to the
  batch, there is no global pool. Different environments share no mutable state.
  The thread count does not affect the result. With several batches, account for the
  total number of threads, including threads of the training library.

The action format is Kaggle dicts (`farmer`, `hands`, `market`). Quantities
and counters inside the engine use signed 64-bit integers; the seed is an integer
of arbitrary length. Game operations and their semantic
no-ops are supported, including insufficient resources, unknown commands and surplus orders.
Arbitrary Python objects are not an action format.

The configuration corresponds to the game fields of `reference/kaggriculture.json`;
`marketParams` supports partial overrides. The constructor checks
types and minimum values; `T` of the market curve must be positive.
Pass the seed as an argument/reset: the `config['seed']` field is not used.
The timeout fields do not start or interrupt agents: this API
accepts already computed actions, and `remainingOverageTime` stays 60.

`episodeSteps=720` means the initial state + **719 transitions**,
as in the pinned reference. Until the end rewards are 0, at the end they are the
money balances; this is not a normalized win reward.

## Self-play

```sh
python examples/self_play.py --games 32 --batch-size 8 --threads 4 --seed 100
```

The example runs a simple carrot policy against its copy, restarts
finished slots and prints the seed, the score and the winner. Training, tensor
encoding and GPU inference are outside this version.

## Checks and measurements

```sh
cargo test --no-default-features --lib
pytest -q tests
cargo build --release --no-default-features --bin bench-core
python benchmarks/run.py
```

After every transition/reset and at the end of every game test, a direct byte-for-byte comparison
of the canonical state with the Python reference is performed. Number types and
the order of items in the inventory are preserved.

The differential tests compare every field after every turn with the
unmodified upstream interpreter; separate tests also run the
pinned Kaggle `Environment` class to check the adapter. Covered are 100 full games, randomness,
market, plants/animals, storage, batch, reset and termination. Comparing both
observations covers the entire game state, including both private storages.

The benchmark runs each variant in a separate process. `rust-core` measures the
core without Python and observations, `rust`/`batch` the whole API including dict creation,
`python` the pinned interpreter through a test adapter with observation copying,
without running agents or the full Kaggle framework. Output is steps/s,
time and the process peak RSS. The load is PASS/PASS; the results are not a
forecast of training speed or of games with intensive trading.

[Current results and measurement history](benchmarks/README.md) separate the
core from the API.

## Layout

`src/model.rs` -- state and configuration; `actions.rs` -- operations and market;
`engine.rs` -- turn order and day change; `rng.rs` -- CPython compatibility;
`bindings.rs` -- Python API. Constants are taken from the pinned reference.

The original mechanics and the vendored Kaggle files are distributed under Apache-2.0;
the license is kept in `reference/LICENSE`.
