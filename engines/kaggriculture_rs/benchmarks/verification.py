"""Final byte audits outside benchmark timing; cached pinned PASS/PASS oracle."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from bytecheck import canonical_bytes
from oracle import Oracle, module, ROOT

VERSION = hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in [
    'reference/kaggriculture.py', 'reference/core.py', 'reference/utils.py',
    'reference/kaggriculture.json', 'tests/oracle.py', 'tests/bytecheck.py',
    'benchmarks/verification.py'])).hexdigest()[:24]
CACHE = ROOT / 'target/benchmark-oracle' / VERSION


def expected_bytes(seed):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{seed}.bin'
    if path.exists():
        return path.read_bytes()
    oracle = Oracle(seed=seed)
    for s in oracle.state:
        s.action = {'farmer':['PASS']}
    for step in range(719):
        module.interpreter(oracle.state, oracle)
        for s in oracle.state:
            s.observation.step = step + 1
    oracle.done = True
    expected = canonical_bytes(oracle.result())
    path.write_bytes(expected)
    return expected


def verify_records(records):
    count = total = 0
    for record in records:
        seed = record['seed']
        actual = canonical_bytes(record['result'])
        expected = expected_bytes(seed)
        if actual != expected:
            output = ROOT/'target/bytecheck-failures'
            output.mkdir(parents=True, exist_ok=True)
            (output/f'benchmark-{seed}.actual.json').write_bytes(actual)
            (output/f'benchmark-{seed}.expected.json').write_bytes(expected)
            raise AssertionError(f'Benchmark byte mismatch for seed {seed}; see {output}')
        count += 1
        total += len(actual)
    return {'passed':True, 'episodes':count, 'bytes_compared':total,
            'format':'kaggriculture-bytecheck-v1', 'oracle_cache_key':VERSION}
