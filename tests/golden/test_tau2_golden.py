"""The tau2 agent's model traffic under the paper settings matches the recorded reference.

Needs tau2-bench with the SpecAgent orchestrator patch applied; skipped otherwise.
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((Path(__file__).resolve().parent / "tau2_golden.json").read_text())


def _patched_tau2() -> bool:
    if importlib.util.find_spec("tau2") is None:
        return False
    from tau2.orchestrator.orchestrator import Orchestrator
    return "take_resolved" in inspect.getsource(Orchestrator.step)


@pytest.mark.skipif(not _patched_tau2(), reason="needs tau2-bench with the orchestrator patch")
def test_tau2_paper_settings_traffic_is_unchanged() -> None:
    sys.path.insert(0, str(ROOT / "benchmarks" / "tau2"))
    import agent as tau2_agent_module
    from tau2_scenarios import run_all

    current = json.loads(json.dumps(run_all(tau2_agent_module), sort_keys=True, default=str))
    assert sorted(current) == sorted(GOLDEN)
    for name, expected in GOLDEN.items():
        got = current[name]
        for part in ("calls", "executions", "trajectory", "error", "unused"):
            assert got[part] == expected[part], (name, part)
        # Counters may only disappear with the features they counted; kept ones are unchanged.
        assert {k: v for k, v in expected["stats"].items() if k in got["stats"]} == got["stats"]
