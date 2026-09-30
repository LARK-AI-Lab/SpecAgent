"""The gateway's model traffic under the paper settings matches the recorded reference."""

import json
from pathlib import Path

from capture import record

GOLDEN = json.loads((Path(__file__).resolve().parent / "golden.json").read_text())


def test_paper_settings_traffic_is_unchanged() -> None:
    current = json.loads(json.dumps(record(), sort_keys=True))
    assert sorted(current) == sorted(GOLDEN)
    for name in GOLDEN:
        assert current[name] == GOLDEN[name], name
