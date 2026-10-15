"""Reproducible release benchmark suite; write a machine-readable report."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
from verification import verify_records

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--games', type=int, default=128)
p.add_argument('--core-games', type=int, default=1024)
p.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1))
p.add_argument('--output', default='benchmarks/results.json')
p.add_argument('--core-worker', action='store_true')
a = p.parse_args()
if a.core_worker:
    with tempfile.TemporaryDirectory() as temp:
        snapshots = Path(temp)/'snapshots.jsonl'
        r = json.loads(subprocess.check_output([str(ROOT/'target/release/bench-core'),str(a.core_games),str(a.threads),'--verify-output',str(snapshots)],text=True))
        with snapshots.open() as records:
            r['bytecheck'] = verify_records(json.loads(line) for line in records)
        r.pop('snapshot_output',None)
    rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform != 'darwin': rss *= 1024
    r.update(mode='rust-core',peak_rss_mib=rss/2**20)
    print(json.dumps(r))
    sys.exit()
results=[]
for mode, threads in [('python',1),('rust',1),('batch',1),('batch',a.threads)]:
    cmd=[sys.executable,str(ROOT/'benchmarks/bench.py'),'--mode',mode,'--games',str(a.games),'--threads',str(threads)]
    result=json.loads(subprocess.check_output(cmd,text=True,cwd=ROOT))
    results.append(result)
    print(json.dumps(result),flush=True)
for threads in [1,a.threads]:
    cmd=[sys.executable,__file__,'--core-worker','--core-games',str(a.core_games),'--threads',str(threads)]
    result=json.loads(subprocess.check_output(cmd,text=True,cwd=ROOT))
    results.append(result)
    print(json.dumps(result),flush=True)
report=dict(platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),python=sys.version,reference_commit='28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c',workload='PASS vs PASS, default configuration, explicit seeds, 719 transitions/episode',results=results)
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
