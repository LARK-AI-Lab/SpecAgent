"""tau2 adapter onto the ResourceLedger: unknown FLOPs/HBM by default, usage never faked."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "benchmarks" / "tau2", REPO / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from metrics_interface import (  # noqa: E402
    TAU2_RESOURCE_PROVENANCE,
    make_tau2_span,
    usage_fields_from_result,
)

from specagent.protocol.accounting import (  # noqa: E402
    InferencePhase,
    QuantityStatus,
    ResourceQuantity,
)
from specagent.protocol.types import ModelRole  # noqa: E402


def test_make_tau2_span_defaults_flops_hbm_unknown_with_tau2_provenance():
    span = make_tau2_span(
        role=ModelRole.WEAK,
        phase=InferencePhase.DRAFT,
        input_tokens=3,
        output_tokens=5,
        duration_seconds=1.25,
        request_id="req",
        transport_attempt_index=1,
        parent_task_id="7",
        parent_episode_id="ep",
        usage_status=QuantityStatus.MEASURED,
    )
    assert span.flops.value is None
    assert span.flops.status.value == "unknown"
    assert span.flops.provenance == TAU2_RESOURCE_PROVENANCE
    assert span.hbm_read_bytes.value is None
    assert span.hbm_write_bytes.value is None
    assert span.transport_attempt_id == "req:1"


def test_make_tau2_span_rejects_bare_flops_including_duration_times_gpu():
    with pytest.raises(ValueError, match="bare number"):
        make_tau2_span(
            role=ModelRole.STRONG,
            phase=InferencePhase.VERIFICATION,
            input_tokens=1,
            output_tokens=1,
            duration_seconds=1.2,
            flops=1.2 * 8,
        )


def test_measured_quantity_is_preserved_when_caller_supplies_one():
    qty = ResourceQuantity(value=42, status=QuantityStatus.MEASURED, provenance="server_counter")
    span = make_tau2_span(
        role=ModelRole.WEAK,
        phase=InferencePhase.DRAFT,
        input_tokens=1,
        output_tokens=1,
        duration_seconds=0.1,
        flops=qty,
        usage_status=QuantityStatus.MEASURED,
    )
    assert span.flops.value == 42
    assert span.flops.status.value == "measured"
    assert span.flops.provenance == "server_counter"


def test_missing_usage_blob_stays_unknown_and_does_not_copy_result_tokens():
    class Result:
        input_tokens = 9
        output_tokens = 4
        cached_input_tokens = 1
        reasoning_tokens = 2
        raw_response = None

    fields = usage_fields_from_result(Result())
    assert fields["usage_status"] is QuantityStatus.UNKNOWN
    assert fields["input_tokens"] == 0
    assert fields["output_tokens"] == 0
    assert fields["cached_input_tokens"] == 0
    assert fields["reasoning_tokens"] == 0
    span = make_tau2_span(
        role=ModelRole.WEAK,
        phase=InferencePhase.DRAFT,
        duration_seconds=0.2,
        **fields,
    )
    assert span.usage_status is QuantityStatus.UNKNOWN
    assert span.input_tokens_status is QuantityStatus.UNKNOWN
    assert span.output_tokens_status is QuantityStatus.UNKNOWN
    assert span.cached_tokens_status is QuantityStatus.UNKNOWN
    assert span.reasoning_tokens_status is QuantityStatus.UNKNOWN


def test_io_measured_without_cached_or_reasoning_is_incomplete_not_measured():
    class Result:
        raw_response = {
            "_specagent_usage": {
                "input_tokens": 11,
                "output_tokens": 6,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "input_status": "measured",
                "output_status": "measured",
                "cached_status": "unknown",
                "reasoning_status": "unknown",
                "provenance": "openai_usage",
                "status": "incomplete",
            }
        }

    fields = usage_fields_from_result(Result())
    assert fields["usage_status"] is QuantityStatus.INCOMPLETE
    assert fields["input_tokens"] == 11
    assert fields["output_tokens"] == 6
    span = make_tau2_span(
        role=ModelRole.STRONG,
        phase=InferencePhase.VERIFICATION,
        duration_seconds=0.3,
        **fields,
    )
    assert span.usage_status is QuantityStatus.INCOMPLETE
    assert span.cache_provenance == "openai_usage"
    assert span.input_tokens_status is QuantityStatus.MEASURED
    assert span.output_tokens_status is QuantityStatus.MEASURED
    assert span.cached_tokens_status is QuantityStatus.UNKNOWN
    assert span.reasoning_tokens_status is QuantityStatus.UNKNOWN
