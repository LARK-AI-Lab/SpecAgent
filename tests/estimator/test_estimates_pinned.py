"""Pin the numeric FLOPs / HBM-read estimates for the four paper models.

``pinned_estimates.json`` holds values recorded from the estimator; any change to an estimate
fails this test. Regenerate only for an intended modeling change:

    python tests/estimator/test_estimates_pinned.py --write
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from specagent.protocol.accounting import InferencePhase, InferenceSpan, QuantityStatus
from specagent.protocol.flops import estimate_request_flops, load_arch
from specagent.protocol.hbm import _pair_from_modes, estimate_request_hbm
from specagent.protocol.resource_estimate import enrich_span, serving_spec_from_arch_id
from specagent.protocol.types import ModelRole

PINNED = Path(__file__).with_name("pinned_estimates.json")
MODELS = ("Qwen3.5-4B", "Qwen3.5-397B-A17B", "DeepSeek-V4-Flash-0731", "GLM-5")
INPUTS = (1, 500, 8000, 60000)
OUTPUTS = (0, 1, 300, 4000)


def _quantity(q: Any) -> dict[str, Any]:
    return {
        "status": q.status.value,
        "value": q.value,
        "lower": q.observed_lower,
        "upper": q.observed_upper,
    }


def _span(model: str, inp: int, out: int, cached: int, provenance: str) -> InferenceSpan:
    return InferenceSpan(
        role=ModelRole.WEAK if model == "Qwen3.5-4B" else ModelRole.STRONG,
        phase=InferencePhase.DRAFT,
        input_tokens=inp,
        output_tokens=out,
        cached_input_tokens=cached,
        duration_seconds=0.0,
        usage_status=QuantityStatus.MEASURED,
        cache_provenance=provenance,
    )


def compute() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for model in MODELS:
        arch = load_arch(model)
        spec = serving_spec_from_arch_id(model)
        for inp in INPUTS:
            for out in OUTPUTS:
                for cached in (0, inp // 2):
                    f = estimate_request_flops(
                        arch,
                        input_tokens=inp,
                        output_tokens=out,
                        compute_prefix_cached_tokens=cached,
                    )
                    h = estimate_request_hbm(
                        arch,
                        input_tokens=inp,
                        output_tokens=out,
                        compute_prefix_cached_tokens=cached,
                    )
                    read, _write = _pair_from_modes(
                        arch, input_tokens=inp, output_tokens=out, cached=cached, extra=""
                    )
                    reuse = enrich_span(_span(model, inp, out, cached, "sglang_prefix_cache"), spec)
                    enriched = enrich_span(_span(model, inp, out, cached, "openai_usage"), spec)
                    no_cache_field = enrich_span(
                        replace(
                            _span(model, inp, out, 0, "openai_usage"),
                            cached_tokens_status=QuantityStatus.UNKNOWN,
                        ),
                        spec,
                    )
                    rows[f"{model}|{inp}|{out}|{cached}"] = {
                        "flops": [f.prefill, f.decode, f.lm_head, f.total],
                        "hbm": [
                            h.read,
                            h.write,
                            h.weight_read,
                            h.kv_read,
                            h.state_read,
                            h.activation_read,
                        ],
                        "hbm_read_interval": _quantity(read),
                        "reuse_flops": _quantity(reuse.flops),
                        "reuse_hbm_read": _quantity(reuse.hbm_read_bytes),
                        "enriched_flops": _quantity(enriched.flops),
                        "enriched_hbm_read": _quantity(enriched.hbm_read_bytes),
                        "unknown_cache_flops": _quantity(no_cache_field.flops),
                        "unknown_cache_hbm_read": _quantity(no_cache_field.hbm_read_bytes),
                    }
    return rows


def test_estimates_match_pinned_values() -> None:
    expected = json.loads(PINNED.read_text())
    got = compute()
    assert set(got) == set(expected)
    mismatches = [key for key in expected if got[key] != expected[key]]
    assert not mismatches, f"{len(mismatches)} changed, first: {mismatches[:3]}"


if __name__ == "__main__" and "--write" in sys.argv:
    PINNED.write_text(json.dumps(compute(), indent=1, sort_keys=True) + "\n")
