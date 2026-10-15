# Repeated speedup check

Apple M4 Pro, release build. Medians of three consecutive runs of each
variant. PASS/PASS load, standard configuration, 719 joint
transitions per game. API: 128 games per run, batch size 32.
Python and Rust core: 1024 games per run. Same seeds;
the checksums of the final balances matched.

| Rust mode | Steps/s | Speedup relative to the corresponding Python variant |
| --- | ---: | ---: |
| Env + Python dicts | 70,133 | 5.92× |
| BatchEnv + Python dicts, 4 threads | 53,059 | 4.48× |
| Pure engine, 1 thread | 409,013 | 7.84× |
| Pure engine, 4 threads | 1,230,409 | 23.57× |

Python with observation output: 11,854
steps/s; Python without copying observations:
52,192 steps/s.
API is compared with API, core with core. The full Kaggle framework,
policy execution and training are excluded from the measurements.

Going from 1 to 4 threads speeds up the pure Rust core about
3.01x.
In the batch API, Python dict creation and synchronization between turns
limit scaling; on PASS/PASS batch is slower than a single Env.
Games with intensive trading and real policies need a separate measurement.

To reproduce: run `python benchmarks/run.py` three times and
`python benchmarks/bench_python_core.py` three times in sequence;
take the medians of steps/s. The build is described in the root README.
