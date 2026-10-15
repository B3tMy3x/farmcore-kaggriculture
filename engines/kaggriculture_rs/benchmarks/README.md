# Release build measurements

[Repeated check: medians of three runs and a Python/Rust core comparison](speed-comparison.md).

Hardware: Apple M4 Pro, 14 logical CPUs;
macOS-26.5.1-arm64-arm-64bit-Mach-O; Python 3.13.13; rustc 1.98.1 (48a229cea 2026-09-01).

Load: PASS against PASS, standard configuration, fixed seeds.
One step is a joint transition of both players. Python API: 128 games,
batch size 32; pure core: 1024 games. In all cases 719 transitions per
game; the checksums of the final balances matched.

| Variant | Threads | Steps/s | Peak RSS, MiB |
| --- | ---: | ---: | ---: |
| Python reference, adapter | 1 | 11,527 | 26.7 |
| Rust Env + Python dicts | 1 | 66,700 | 27.4 |
| Rust BatchEnv + Python dicts | 1 | 48,575 | 31.3 |
| Rust BatchEnv + Python dicts | 4 | 54,259 | 32.0 |
| Rust core without Python | 1 | 391,530 | 2.3 |
| Rust core without Python | 4 | 1,080,466 | 2.8 |

Rust Env with dict output is about 5.8x faster than the Python adapter.
The pure core scales about 2.8x from 1 to 4 threads.
The batch API is limited by sequential creation of Python dicts and a barrier
between turns: on this light load it is slower than a single Env.

The core distributes whole games across threads; BatchEnv distributes one turn
of the batch per call. So the pure core speed is not the speed of Python
self-play. Policy/neural-net evaluation speed is not measured here.
The Python reference is the pinned interpreter through an adapter, without the overhead
of the full Kaggle framework. This is a single measurement, not a statistical
estimate; the numbers depend on the load and the machine.

To repeat: `python benchmarks/run.py` after the release build from the README.
