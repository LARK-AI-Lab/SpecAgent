"""Record the golden traffic: `python tests/golden/capture.py` with the reference code on
sys.path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scenarios import run_all  # noqa: E402

from specagent.runtime.gateway import GatewayConfig, SpeculativeToolGateway  # noqa: E402


def paper_configs() -> dict[str, GatewayConfig]:
    raw = json.loads((HERE / "paper_configs.json").read_text())
    configs = {}
    for name, fields in raw.items():
        fields = {k: tuple(v) if isinstance(v, list) else v for k, v in fields.items()}
        configs[name] = GatewayConfig(weak_model="weak", strong_model="strong", **fields)
    return configs


def record() -> dict:
    return run_all(SpeculativeToolGateway, paper_configs())


if __name__ == "__main__":
    (HERE / "golden.json").write_text(json.dumps(record(), indent=1, sort_keys=True))
