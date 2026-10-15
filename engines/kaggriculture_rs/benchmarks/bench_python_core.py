"""Pinned Python interpreter without materializing observation snapshots.

Matches bench-core's PASS/PASS workload and final-score checksum. Agent
execution, serialization and full Kaggle framework are excluded on both sides.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from oracle import Oracle, module
from verification import verify_records

parser = argparse.ArgumentParser()
parser.add_argument('--games', type=int, default=1024)
args = parser.parse_args()
action = {'farmer': ['PASS']}
start = time.perf_counter()
checksum = 0.0
finished = []
for seed in range(args.games):
    env = Oracle(seed=seed)
    for state in env.state:
        state.action = action
    for step in range(719):
        module.interpreter(env.state, env)
        for state in env.state:
            state.observation.step = step + 1
    checksum += env.state[0].observation.farms[0]['money']
    env.done = True
    finished.append(env)
seconds = time.perf_counter() - start
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != 'darwin':
    rss *= 1024
audit = verify_records({'seed':seed,'result':e.result()} for seed,e in enumerate(finished))
print(json.dumps(dict(bytecheck=audit, mode='python-core', threads=1, episodes=args.games,
    seconds=seconds, steps_per_second=args.games*719/seconds,
    peak_rss_mib=rss/2**20, checksum=checksum)))
