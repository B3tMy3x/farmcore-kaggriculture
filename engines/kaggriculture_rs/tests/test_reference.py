import ast
import hashlib
import json
from pathlib import Path
from oracle import ROOT, module

def test_reference_hashes():
    manifest=json.loads((ROOT/'reference/UPSTREAM.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/'reference'/name).read_bytes()).hexdigest()==expected

def test_constant_tables():
    tables=json.loads((ROOT/'src/constants.json').read_text())
    for name, value in tables.items():
        assert value == getattr(module,name)
