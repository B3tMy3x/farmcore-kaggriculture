"""Compare two core binaries; final byte audit is mandatory after each run."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
from verification import verify_records

p=argparse.ArgumentParser()
p.add_argument('--before',type=Path,required=True)
p.add_argument('--after',type=Path,default=Path('target/release/bench-core'))
p.add_argument('--games',type=int,default=1024)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--output',type=Path,default=Path('benchmarks/optimization-results.json'))
a=p.parse_args()
runs=[]
for repeat in range(a.repeats):
    for threads in (1,4):
        for version,binary in [('before',a.before),('after',a.after)]:
            with tempfile.TemporaryDirectory() as tmp:
                snapshot=Path(tmp)/'final.jsonl'
                command=[str(binary.resolve()),str(a.games),str(threads),'--verify-output',str(snapshot)]
                result=json.loads(subprocess.check_output(command,text=True))
                with snapshot.open() as lines:
                    result['bytecheck']=verify_records(json.loads(line) for line in lines)
                result.pop('snapshot_output',None)
            result.update(version=version,repeat=repeat+1)
            runs.append(result)
            print(json.dumps(result),flush=True)
summary=[]
for threads in (1,4):
    medians={version:statistics.median(r['steps_per_second'] for r in runs if r['version']==version and r['threads']==threads) for version in ('before','after')}
    summary.append(dict(threads=threads,**medians,speedup=medians['after']/medians['before']))
report=dict(workload='PASS/PASS, default config, identical joint actions and seeds',repeats=a.repeats,episodes_per_run=a.games,summary=summary,runs=runs)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print('SUMMARY',json.dumps(summary),flush=True)
