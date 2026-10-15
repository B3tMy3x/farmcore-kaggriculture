"""Versioned canonical wire snapshots, compared as bytes, never just hashes.

Object field order is normalized. Per-unit inventory insertion order is part
of the state because it decides overflow losses, so it is encoded explicitly.
Numbers retain Python's round-trip representation (1 != 1.0, -0.0 != 0.0).
Only dict/list/JSON scalar types are accepted; tuples cannot silently become lists.
"""
import json
from pathlib import Path
import re


def canonical_bytes(value):
    def normalize(obj, path=()):
        if isinstance(obj, dict):
            if not all(isinstance(k, str) for k in obj):
                raise AssertionError(f'Non-string key at {path}')
            if len(path) >= 2 and path[-2] == 'inventories' and isinstance(path[-1], int):
                return {'$ordered_inventory': [[k, normalize(v, path + (k,))] for k, v in obj.items()]}
            return {k: normalize(v, path + (k,)) for k, v in obj.items()}
        if type(obj) is list:
            return [normalize(v, path + (i,)) for i, v in enumerate(obj)]
        if obj is None or type(obj) in (str, int, float, bool):
            return obj
        raise AssertionError(f'Unsupported snapshot type {type(obj).__name__} at {path}')
    snapshot = {'format': 'kaggriculture-bytecheck-v1', 'state': normalize(value)}
    return json.dumps(snapshot, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def assert_bytes_equal(actual, expected, label='state'):
    actual_bytes, expected_bytes = canonical_bytes(actual), canonical_bytes(expected)
    if actual_bytes == expected_bytes:
        return len(actual_bytes)
    directory = Path(__file__).resolve().parents[1] / 'target/bytecheck-failures'
    directory.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', label)[-150:]
    actual_path, expected_path = directory / f'{name}.actual.json', directory / f'{name}.expected.json'
    actual_path.write_bytes(actual_bytes)
    expected_path.write_bytes(expected_bytes)
    offset = next((i for i, (a, b) in enumerate(zip(actual_bytes, expected_bytes)) if a != b), min(len(actual_bytes), len(expected_bytes)))
    raise AssertionError(f'{label}: byte mismatch at offset {offset}; lengths {len(actual_bytes)}/{len(expected_bytes)}; '
                         f'actual={actual_bytes[max(0, offset-40):offset+80]!r}; '
                         f'expected={expected_bytes[max(0, offset-40):offset+80]!r}; '
                         f'files: {actual_path}, {expected_path}')
