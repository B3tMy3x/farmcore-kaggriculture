"""Mandatory final byte audit for every Env/BatchEnv created in game tests."""
import json
from pathlib import Path
import pytest
from checked_env import TRACKED, TRANSITION_AUDITS

AUDITS = []


@pytest.fixture(autouse=True)
def final_byte_audit(request):
    TRACKED.clear()
    yield
    total = 0
    try:
        for i, env in enumerate(TRACKED):
            total += env.bytecheck(f'{request.node.nodeid}-env{i}')
        if TRACKED:
            AUDITS.append({'test': request.node.nodeid, 'environments': len(TRACKED), 'bytes_compared': total, 'passed': True})
    except AssertionError:
        AUDITS.append({'test': request.node.nodeid, 'passed': False})
        raise
    finally:
        TRACKED.clear()


def pytest_sessionfinish(session, exitstatus):
    destination = Path(__file__).resolve().parents[1] / 'target/bytecheck-report.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'format': 'kaggriculture-bytecheck-v1', 'exitstatus': int(exitstatus), 'transitions': TRANSITION_AUDITS, 'audits': AUDITS}, indent=2) + '\n')


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f'Transition byte audits: {TRANSITION_AUDITS["checks"]:,}; '
                               f'{TRANSITION_AUDITS["bytes_compared"]:,} bytes compared')
    terminalreporter.write_line(f'Final byte audits: {sum(a["passed"] for a in AUDITS)}/{len(AUDITS)} game tests; '
                               f'{sum(a.get("bytes_compared", 0) for a in AUDITS):,} bytes compared')
