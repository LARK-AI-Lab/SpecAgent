"""Record the golden traffic: `python tests/golden/capture.py [CONFIG ...]` with the reference
code on sys.path re-records the named paper configs (every config when none is named) and keeps
the others' recorded traffic as it is."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scenarios import run_all  # noqa: E402

from specagent.runtime.gateway import GatewayConfig, SpeculativeToolGateway  # noqa: E402

# How the reference code spells a paper config, where it differs from paper_configs.json; used
# only when recording. The paper's VitaBench run used the reference's selective_fused_rp profile,
# whose behaviour the release gateway has built in, with verify_scope="mutation" (only
# state-changing tool calls reviewed), which the release spells review_replies=False; the
# reference's payload labels are what the release calls verifier_payload_labels="legacy".
REFERENCE_FIELDS: dict[str, dict[str, Any]] = {
    "vitabench": {
        "review_replies": None,
        "verifier_payload_labels": None,
        "verify_scope": "mutation",
        "task_semantics": "stateful",
        "protocol_family": "selective_fused_rp",
        "contract_version": "rp_fused_v1",
        "verifier_contract": "rp_fused_v1",
        "fuse_verifier_correction": True,
        "corrector_multi_tool": True,
        "deliver_draft_reasoning": False,
        "deliver_correction_reasoning": False,
    },
}


def paper_configs(*, reference: bool = False) -> dict[str, GatewayConfig]:
    raw = json.loads((HERE / "paper_configs.json").read_text())
    configs = {}
    for name, fields in raw.items():
        fields = {k: tuple(v) if isinstance(v, list) else v for k, v in fields.items()}
        if reference:
            fields = {**fields, **REFERENCE_FIELDS.get(name, {})}
            fields = {k: v for k, v in fields.items() if v is not None}
        configs[name] = GatewayConfig(weak_model="weak", strong_model="strong", **fields)
    return configs


def record(*, reference: bool = False, only: tuple[str, ...] = ()) -> dict:
    configs = paper_configs(reference=reference)
    if only:
        configs = {name: configs[name] for name in only}
    return run_all(SpeculativeToolGateway, configs)


if __name__ == "__main__":
    only = tuple(sys.argv[1:])
    path = HERE / "golden.json"
    golden = json.loads(path.read_text()) if only else {}
    golden = {k: v for k, v in golden.items() if k.split("/", 1)[0] not in only}
    golden.update(record(reference=True, only=only))
    path.write_text(json.dumps(golden, indent=1, sort_keys=True))
