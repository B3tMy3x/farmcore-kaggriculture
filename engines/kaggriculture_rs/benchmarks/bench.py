"""Each mode runs in a fresh process so peak RSS is attributable to that mode."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from oracle import Oracle
from kaggriculture_rs import Env, BatchEnv
from verification import verify_records

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['python', 'rust', 'batch'], required=True)
p.add_argument('--games', type=int, default=128)
p.add_argument('--batch-size', type=int, default=32)
p.add_argument('--threads', type=int, default=1)
a = p.parse_args()
action = [{'farmer': ['PASS']}] * 2
start = time.perf_counter()
checksum = 0
records = []
if a.mode == 'batch':
    for offset in range(0, a.games, a.batch_size):
        width = min(a.batch_size, a.games-offset)
        e = BatchEnv(width, num_threads=a.threads)
        e.reset(list(range(offset, offset+width)))
        for _ in range(719):
            result = e.step([action]*width)
        checksum += sum(x['rewards'][0] for x in result)
        records.extend({'seed':offset+i,'result':x} for i,x in enumerate(result))
else:
    for seed in range(a.games):
        e = Oracle(seed=seed) if a.mode == 'python' else Env(seed=seed)
        for _ in range(719):
            result = e.step(action)
        checksum += result['rewards'][0]
        records.append({'seed':seed,'result':result})
seconds = time.perf_counter()-start
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != 'darwin': rss *= 1024
audit = verify_records(records)
print(json.dumps(dict(bytecheck=audit, mode=a.mode, games=a.games, threads=a.threads, batch_size=a.batch_size, seconds=seconds, steps_per_second=a.games*719/seconds, peak_rss_mib=rss/2**20, checksum=checksum)))
