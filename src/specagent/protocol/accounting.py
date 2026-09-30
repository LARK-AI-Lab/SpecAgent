"""Token, call-count, latency, and allocated-GPU-second accounting for SpecAgent experiments.

Two deployment settings share this ledger. In the dedicated setting both models are served locally
and GPU-seconds is the primary cost. In the local--cloud setting the Strong model is a metered API,
so the number of Strong calls, Strong input tokens, and call latency are the reportable costs and
GPU-seconds is not meaningful for the Strong side. ``ResourceSummary.to_dict`` exposes all of them
so the same per-task record can be written for every integration.

FLOPs/HBM stay unknown unless an estimator or measurement writes a ResourceQuantity with
provenance and coverage. Client duration * gpu_count is allocated GPU-seconds only; it is never a
FLOPs or HBM fill-in.

Token completeness is per role. A Strong span marked ``legacy_unspecified`` does not relabel
evidenced Weak tokens as unspecified: ``weak_token_totals_status`` stays with the Weak evidence,
and mixed roles make the combined ``token_totals_status`` incomplete with an explicit reason.

``exclusive_gpu_seconds_estimated`` is a separate analyzer field. It is entered as
sum(duration_seconds * gpu_count) over charged spans with status=estimated. That formula is a
client-elapsed exclusive-occupancy *proxy*, not server exclusive service time and not a rename of
allocated GPU-seconds (``total_gpu_seconds`` / ``allocated_gpu_seconds_status``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import Any

from specagent.protocol.estimator_ids import (
    FLOPS_ESTIMATOR_ID,
    HBM_ESTIMATOR_ID,
    LEDGER_NO_ESTIMATOR,
)
from specagent.protocol.types import ModelRole

RESOURCE_SCHEMA_VERSION = "resource_ledger_v2"
ESTIMATOR_VERSION = LEDGER_NO_ESTIMATOR
EXCLUSIVE_GPU_SECONDS_PROVENANCE = (
    "client_elapsed_times_gpu_count_proxy:"
    "sum(duration_seconds * gpu_count) over charged agent spans;"
    "not_server_exclusive_service_time;"
    "not_allocated_gpu_seconds_rename;"
    "not_nsight_busy_time"
)
ALLOCATED_GPU_SECONDS_PROVENANCE = (
    "client_elapsed_times_gpu_count:"
    "sum(duration_seconds * gpu_count) over charged agent spans;"
    "allocated_occupancy_not_exclusive_service_time"
)
LEGACY_CALL_KEYS = (
    "role",
    "phase",
    "duration_seconds",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)

_STATUS_RANK = {
    "unknown": 0,
    "incomplete": 1,
    "legacy_unspecified": 2,
    "estimated": 3,
    "measured": 4,
}


class QuantityStatus(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    # Present in old logs as a bare number with no provenance.
    LEGACY_UNSPECIFIED = "legacy_unspecified"
    # Some components or token fields are known; the total is not complete.
    INCOMPLETE = "incomplete"


class CanonicalPhase(str, Enum):
    WEAK_DRAFT = "weak_draft"
    WEAK_ACT = "weak_act"
    WEAK_RETRY = "weak_retry"
    STRONG_VERIFY = "strong_verify"
    STRONG_CORRECT = "strong_correct"
    STRONG_OPENING = "strong_opening"
    STRONG_ACT = "strong_act"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


class InferencePhase(str, Enum):
    DRAFT = "draft"
    VERIFICATION = "verification"
    CORRECTION = "correction"
    ACT = "act"  # committed acting without speculation (solo baselines)
    OPENING = "opening"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


class LedgerConflictError(ValueError):
    """Same request/attempt identity with disagreeing billed fields."""


def canonical_phase(
    role: ModelRole | str,
    phase: InferencePhase | str,
    *,
    intent: str | None = None,
) -> CanonicalPhase:
    """Map (role, persisted phase, optional intent) onto the frozen reporting names.

    Persisted InferencePhase values stay stable for running code. Reporting tables
    use CanonicalPhase. Unknown combinations stay unknown rather than being coerced.
    """
    role_value = role.value if isinstance(role, ModelRole) else str(role)
    phase_value = phase.value if isinstance(phase, InferencePhase) else str(phase)
    weak_retryable = {
        InferencePhase.DRAFT.value,
        InferencePhase.ACT.value,
    }
    if (
        intent == "retry"
        and role_value == ModelRole.WEAK.value
        and phase_value in weak_retryable
    ):
        return CanonicalPhase.WEAK_RETRY
    if phase_value == InferencePhase.UNKNOWN.value:
        return CanonicalPhase.UNKNOWN
    if phase_value == InferencePhase.OPENING.value:
        return CanonicalPhase.STRONG_OPENING
    if phase_value == InferencePhase.SUMMARY.value:
        return CanonicalPhase.SUMMARY
    if phase_value == InferencePhase.VERIFICATION.value:
        return CanonicalPhase.STRONG_VERIFY
    if phase_value == InferencePhase.CORRECTION.value:
        return CanonicalPhase.STRONG_CORRECT
    if phase_value == InferencePhase.ACT.value:
        if role_value == ModelRole.WEAK.value:
            return CanonicalPhase.WEAK_ACT
        if role_value == ModelRole.STRONG.value:
            return CanonicalPhase.STRONG_ACT
        return CanonicalPhase.UNKNOWN
    if phase_value == InferencePhase.DRAFT.value:
        if role_value == ModelRole.WEAK.value:
            return CanonicalPhase.WEAK_DRAFT
        if role_value == ModelRole.STRONG.value:
            return CanonicalPhase.STRONG_ACT
        return CanonicalPhase.UNKNOWN
    return CanonicalPhase.UNKNOWN


def _finite_non_negative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return float(value) >= 0.0


def _require_finite_non_negative(value: Any, *, what: str) -> None:
    if not _finite_non_negative(value):
        raise ValueError(f"{what} must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class ResourceQuantity:
    value: float | int | None
    status: QuantityStatus = QuantityStatus.UNKNOWN
    provenance: str = "no_server_counter"
    coverage: float | None = None
    observed_subtotal: float | int | None = None
    observed_measured: float | int | None = None
    observed_estimated: float | int | None = None
    observed_lower: float | int | None = None
    observed_upper: float | int | None = None

    def __post_init__(self) -> None:
        if self.coverage is not None and not _finite_non_negative(self.coverage):
            raise ValueError("coverage must be in [0, 1] or None")
        if self.coverage is not None and float(self.coverage) > 1.0:
            raise ValueError("coverage must be in [0, 1] or None")
        complete = {QuantityStatus.MEASURED, QuantityStatus.ESTIMATED}
        if self.status in complete:
            _require_finite_non_negative(self.value, what=f"{self.status.value} value")
        elif self.status is QuantityStatus.LEGACY_UNSPECIFIED:
            _require_finite_non_negative(self.value, what="legacy_unspecified value")
        elif self.status in {
            QuantityStatus.UNKNOWN,
            QuantityStatus.NOT_APPLICABLE,
            QuantityStatus.INCOMPLETE,
        }:
            if self.value is not None:
                raise ValueError("incomplete/unknown/n/a quantities must not carry a fill-in value")
        for name in (
            "observed_subtotal",
            "observed_measured",
            "observed_estimated",
            "observed_lower",
            "observed_upper",
        ):
            observed = getattr(self, name)
            if observed is not None:
                _require_finite_non_negative(observed, what=name)
        if (
            self.observed_lower is not None
            and self.observed_upper is not None
            and float(self.observed_lower) > float(self.observed_upper)
        ):
            raise ValueError("observed_lower cannot exceed observed_upper")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status.value,
            "provenance": self.provenance,
            "coverage": self.coverage,
            "observed_subtotal": self.observed_subtotal,
            "observed_measured": self.observed_measured,
            "observed_estimated": self.observed_estimated,
            "observed_lower": self.observed_lower,
            "observed_upper": self.observed_upper,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ResourceQuantity:
        if not data:
            return unknown_quantity()
        status = QuantityStatus(str(data.get("status") or QuantityStatus.UNKNOWN.value))
        return cls(
            value=data.get("value"),
            status=status,
            provenance=str(data.get("provenance") or "no_server_counter"),
            coverage=data.get("coverage"),
            observed_subtotal=data.get("observed_subtotal"),
            observed_measured=data.get("observed_measured"),
            observed_estimated=data.get("observed_estimated"),
            observed_lower=data.get("observed_lower"),
            observed_upper=data.get("observed_upper"),
        )


def unknown_quantity(*, provenance: str = "no_server_counter") -> ResourceQuantity:
    return ResourceQuantity(
        value=None,
        status=QuantityStatus.UNKNOWN,
        provenance=provenance,
        coverage=None,
    )


def _quantity_field() -> ResourceQuantity:
    return unknown_quantity()


def _as_phase(value: Any) -> InferencePhase:
    raw = value.value if isinstance(value, InferencePhase) else str(value or "")
    try:
        return InferencePhase(raw)
    except ValueError:
        return InferencePhase.UNKNOWN


def _as_role(value: Any) -> ModelRole:
    return value if isinstance(value, ModelRole) else ModelRole(str(value))


def _as_usage_status(data: Mapping[str, Any]) -> QuantityStatus:
    raw = data.get("usage_status")
    if raw:
        return QuantityStatus(str(raw))
    return QuantityStatus.LEGACY_UNSPECIFIED


def _as_attribution(data: Mapping[str, Any]) -> str:
    if data.get("attribution"):
        return str(data["attribution"])
    phase = str(data.get("phase") or "")
    if phase == InferencePhase.SUMMARY.value and not data.get("parent_task_id"):
        return "run_unattributed"
    return "task"


def _as_field_status(data: Mapping[str, Any], key: str) -> QuantityStatus | None:
    raw = data.get(key)
    if raw is None:
        return None
    return QuantityStatus(str(raw))


@dataclass(frozen=True, slots=True)
class ParsedUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    input_status: QuantityStatus = QuantityStatus.UNKNOWN
    output_status: QuantityStatus = QuantityStatus.UNKNOWN
    cached_status: QuantityStatus = QuantityStatus.UNKNOWN
    reasoning_status: QuantityStatus = QuantityStatus.UNKNOWN
    provenance: str = "missing_usage"

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            _require_finite_non_negative(getattr(self, name), what=name)
        for name, status_name in (
            ("input_tokens", "input_status"),
            ("output_tokens", "output_status"),
            ("cached_input_tokens", "cached_status"),
            ("reasoning_tokens", "reasoning_status"),
        ):
            status = getattr(self, status_name)
            if status is QuantityStatus.UNKNOWN and getattr(self, name) != 0:
                raise ValueError("unknown token fields cannot carry fill-in values")

    @property
    def status(self) -> QuantityStatus:
        statuses = (
            self.input_status,
            self.output_status,
            self.cached_status,
            self.reasoning_status,
        )
        if all(item is QuantityStatus.UNKNOWN for item in statuses):
            return QuantityStatus.UNKNOWN
        if all(item is QuantityStatus.NOT_APPLICABLE for item in statuses):
            return QuantityStatus.NOT_APPLICABLE
        if all(item is QuantityStatus.MEASURED for item in statuses):
            return QuantityStatus.MEASURED
        if all(item is QuantityStatus.LEGACY_UNSPECIFIED for item in statuses):
            return QuantityStatus.LEGACY_UNSPECIFIED
        return QuantityStatus.INCOMPLETE

    def to_dict(self) -> dict[str, Any]:
        record = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "input_status": self.input_status.value,
            "output_status": self.output_status.value,
            "cached_status": self.cached_status.value,
            "reasoning_status": self.reasoning_status.value,
            "provenance": self.provenance,
            "status": self.status.value,
        }
        return record

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ParsedUsage:
        """Rebuild usage from to_dict()."""
        if not data:
            return cls()

        def _status(key: str, default: QuantityStatus) -> QuantityStatus:
            raw = data.get(key)
            if raw is None:
                return default
            return QuantityStatus(str(raw))

        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            cached_input_tokens=int(data.get("cached_input_tokens") or 0),
            reasoning_tokens=int(data.get("reasoning_tokens") or 0),
            input_status=_status("input_status", QuantityStatus.UNKNOWN),
            output_status=_status("output_status", QuantityStatus.UNKNOWN),
            cached_status=_status("cached_status", QuantityStatus.UNKNOWN),
            reasoning_status=_status("reasoning_status", QuantityStatus.UNKNOWN),
            provenance=str(data.get("provenance") or "missing_usage"),
        )


def token_fields_from_usage(parsed: ParsedUsage) -> dict[str, Any]:
    """Adapt ParsedUsage onto InferenceSpan token/status kwargs."""
    return {
        "input_tokens": parsed.input_tokens,
        "output_tokens": parsed.output_tokens,
        "cached_input_tokens": parsed.cached_input_tokens,
        "reasoning_tokens": parsed.reasoning_tokens,
        "usage_status": parsed.status,
        "input_tokens_status": parsed.input_status,
        "output_tokens_status": parsed.output_status,
        "cached_tokens_status": parsed.cached_status,
        "reasoning_tokens_status": parsed.reasoning_status,
        "cache_provenance": parsed.provenance,
    }


@dataclass(frozen=True, slots=True)
class InferenceSpan:
    role: ModelRole
    phase: InferencePhase
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    gpu_count: int = 1
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    committed_output_tokens: int = 0
    discarded_output_tokens: int = 0
    request_id: str | None = None
    transport_attempt_id: str | None = None
    transport_attempt_index: int = 1
    parent_request_id: str | None = None
    parent_task_id: str | None = None
    parent_episode_id: str | None = None
    provider_trace_id: str | None = None
    status: str = "ok"
    usage_status: QuantityStatus = QuantityStatus.MEASURED
    cache_provenance: str = "unspecified"
    charge_inference: bool = True
    source: str = "live"
    intent: str | None = None
    attribution: str = "task"
    flops: ResourceQuantity = field(default_factory=_quantity_field)
    hbm_read_bytes: ResourceQuantity = field(default_factory=_quantity_field)
    hbm_write_bytes: ResourceQuantity = field(default_factory=_quantity_field)
    input_tokens_status: QuantityStatus | None = None
    output_tokens_status: QuantityStatus | None = None
    cached_tokens_status: QuantityStatus | None = None
    reasoning_tokens_status: QuantityStatus | None = None

    def __post_init__(self) -> None:
        counts = (
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.reasoning_tokens,
            self.committed_output_tokens,
            self.discarded_output_tokens,
        )
        if any(not _finite_non_negative(value) for value in counts):
            raise ValueError("token counts must be non-negative")
        if self.duration_seconds < 0 or self.gpu_count < 1:
            raise ValueError("duration must be non-negative and gpu_count positive")
        if self.committed_output_tokens + self.discarded_output_tokens > self.output_tokens:
            raise ValueError("draft token disposition exceeds output_tokens")
        if self.transport_attempt_index < 1:
            raise ValueError("transport_attempt_index must be >= 1")
        pairs = (
            (self.input_tokens, self.resolved_input_status),
            (self.output_tokens, self.resolved_output_status),
            (self.cached_input_tokens, self.resolved_cached_status),
            (self.reasoning_tokens, self.resolved_reasoning_status),
        )
        for count, field_status in pairs:
            if field_status is QuantityStatus.UNKNOWN and count != 0:
                raise ValueError("unknown usage cannot carry token fill-in values")
            if field_status is QuantityStatus.NOT_APPLICABLE and count != 0:
                raise ValueError("not_applicable usage cannot carry token fill-in values")
        if self.source == "cached_replay" and self.charge_inference:
            raise ValueError("cached replay without new inference must not be charged")
        if (
            self.phase is InferencePhase.SUMMARY
            and not self.parent_task_id
            and self.attribution == "task"
        ):
            object.__setattr__(self, "attribution", "run_unattributed")

    def _inherited(self, explicit: QuantityStatus | None) -> QuantityStatus:
        return self.usage_status if explicit is None else explicit

    @property
    def resolved_input_status(self) -> QuantityStatus:
        return self._inherited(self.input_tokens_status)

    @property
    def resolved_output_status(self) -> QuantityStatus:
        return self._inherited(self.output_tokens_status)

    @property
    def resolved_cached_status(self) -> QuantityStatus:
        return self._inherited(self.cached_tokens_status)

    @property
    def resolved_reasoning_status(self) -> QuantityStatus:
        return self._inherited(self.reasoning_tokens_status)

    @property
    def gpu_seconds(self) -> float:
        return self.duration_seconds * self.gpu_count

    @property
    def canonical_phase(self) -> CanonicalPhase:
        return canonical_phase(self.role, self.phase, intent=self.intent)

    @property
    def identity_key(self) -> tuple[str, ...]:
        kind = "charged" if self.charge_inference else f"uncharged:{self.source}"
        if self.request_id and self.transport_attempt_id:
            return ("attempt_id", self.request_id, self.transport_attempt_id, kind)
        if self.request_id:
            return (
                "attempt_index",
                self.request_id,
                str(self.transport_attempt_index),
                kind,
            )
        return ("anon",)

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        return (
            self.role.value,
            self.phase.value,
            self.intent,
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.reasoning_tokens,
            self.committed_output_tokens,
            self.discarded_output_tokens,
            round(self.duration_seconds, 9),
            self.gpu_count,
            self.status,
            self.usage_status.value,
            self.resolved_input_status.value,
            self.resolved_output_status.value,
            self.resolved_cached_status.value,
            self.resolved_reasoning_status.value,
            self.charge_inference,
            self.source,
            self.attribution,
            self.parent_request_id,
            self.parent_task_id,
            self.parent_episode_id,
            self.provider_trace_id,
            self.cache_provenance,
            self.transport_attempt_index,
            self.flops.to_dict(),
            self.hbm_read_bytes.to_dict(),
            self.hbm_write_bytes.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Legacy 7-key projection. Running readers assert this exact key set."""
        return {
            "role": self.role.value,
            "phase": self.phase.value,
            "duration_seconds": self.duration_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    def to_record(self) -> dict[str, Any]:
        record = self.to_dict()
        record.update(
            {
                "schema_version": RESOURCE_SCHEMA_VERSION,
                "legacy_phase": self.phase.value,
                "canonical_phase": self.canonical_phase.value,
                "gpu_count": self.gpu_count,
                "committed_output_tokens": self.committed_output_tokens,
                "discarded_output_tokens": self.discarded_output_tokens,
                "request_id": self.request_id,
                "transport_attempt_id": self.transport_attempt_id,
                "transport_attempt_index": self.transport_attempt_index,
                "parent_request_id": self.parent_request_id,
                "parent_task_id": self.parent_task_id,
                "parent_episode_id": self.parent_episode_id,
                "provider_trace_id": self.provider_trace_id,
                "status": self.status,
                "usage_status": self.usage_status.value,
                "input_tokens_status": self.resolved_input_status.value,
                "output_tokens_status": self.resolved_output_status.value,
                "cached_tokens_status": self.resolved_cached_status.value,
                "reasoning_tokens_status": self.resolved_reasoning_status.value,
                "cache_provenance": self.cache_provenance,
                "charge_inference": self.charge_inference,
                "source": self.source,
                "intent": self.intent,
                "attribution": self.attribution,
                "flops": self.flops.to_dict(),
                "hbm_read_bytes": self.hbm_read_bytes.to_dict(),
                "hbm_write_bytes": self.hbm_write_bytes.to_dict(),
            }
        )
        return record


def span_from_legacy_call(data: Mapping[str, Any]) -> InferenceSpan:
    """Rebuild a span from ResourceSummary.calls rows and other v1 dumps.

    Missing identity/parent/FLOPs/HBM stay unknown. Bare token integers are kept
    on the token axis with usage_status=legacy_unspecified; they are not upgraded
    into measured FLOPs/HBM.
    """
    source = str(data.get("source") or "live")
    charge = data.get("charge_inference")
    if charge is None:
        charge = source != "cached_replay"
    intent = data.get("intent")
    return InferenceSpan(
        role=_as_role(data["role"]),
        phase=_as_phase(data.get("phase")),
        input_tokens=int(data.get("input_tokens") or 0),
        output_tokens=int(data.get("output_tokens") or 0),
        duration_seconds=float(data.get("duration_seconds") or 0.0),
        gpu_count=max(int(data.get("gpu_count") or 1), 1),
        cached_input_tokens=int(data.get("cached_input_tokens") or 0),
        reasoning_tokens=int(data.get("reasoning_tokens") or 0),
        committed_output_tokens=int(data.get("committed_output_tokens") or 0),
        discarded_output_tokens=int(data.get("discarded_output_tokens") or 0),
        request_id=data.get("request_id"),
        transport_attempt_id=data.get("transport_attempt_id"),
        transport_attempt_index=int(data.get("transport_attempt_index") or 1),
        parent_request_id=data.get("parent_request_id"),
        parent_task_id=data.get("parent_task_id"),
        parent_episode_id=data.get("parent_episode_id"),
        provider_trace_id=data.get("provider_trace_id"),
        status=str(data.get("status") or "ok"),
        usage_status=_as_usage_status(data),
        cache_provenance=str(data.get("cache_provenance") or "legacy_unspecified"),
        charge_inference=bool(charge),
        source=source,
        intent=None if intent in (None, "") else str(intent),
        attribution=_as_attribution(data),
        flops=ResourceQuantity.from_dict(
            data.get("flops") if isinstance(data.get("flops"), Mapping) else None
        ),
        hbm_read_bytes=ResourceQuantity.from_dict(
            data.get("hbm_read_bytes")
            if isinstance(data.get("hbm_read_bytes"), Mapping)
            else None
        ),
        hbm_write_bytes=ResourceQuantity.from_dict(
            data.get("hbm_write_bytes")
            if isinstance(data.get("hbm_write_bytes"), Mapping)
            else None
        ),
        input_tokens_status=_as_field_status(data, "input_tokens_status"),
        output_tokens_status=_as_field_status(data, "output_tokens_status"),
        cached_tokens_status=_as_field_status(data, "cached_tokens_status"),
        reasoning_tokens_status=_as_field_status(data, "reasoning_tokens_status"),
    )


def span_from_record(data: Mapping[str, Any]) -> InferenceSpan:
    return span_from_legacy_call(data)


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    total_gpu_seconds: float
    strong_gpu_seconds: float
    weak_input_tokens: int
    weak_committed_output_tokens: int
    weak_discarded_output_tokens: int
    strong_verifier_input_tokens: int
    strong_verifier_output_tokens: int
    strong_correction_input_tokens: int
    strong_correction_output_tokens: int
    strong_act_input_tokens: int
    strong_act_output_tokens: int
    cached_input_tokens: int
    weak_gpu_seconds: float = 0.0
    weak_output_tokens: int = 0
    weak_cached_input_tokens: int = 0
    weak_reasoning_tokens: int = 0
    strong_input_tokens: int = 0
    strong_output_tokens: int = 0
    strong_cached_input_tokens: int = 0
    strong_reasoning_tokens: int = 0
    llm_call_seconds: float = 0.0
    weak_call_seconds: float = 0.0
    strong_call_seconds: float = 0.0
    strong_verify_call_seconds: float = 0.0
    strong_correction_call_seconds: float = 0.0
    strong_act_call_seconds: float = 0.0
    n_calls: int = 0
    n_weak_calls: int = 0
    n_strong_calls: int = 0
    n_strong_verify_calls: int = 0
    n_strong_correction_calls: int = 0
    calls: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    resource_schema_version: str = RESOURCE_SCHEMA_VERSION
    estimator_version: str = ESTIMATOR_VERSION
    n_transport_attempts: int = 0
    n_failed_calls: int = 0
    n_cached_replays: int = 0
    n_unattributed: int = 0
    n_unknown_usage: int = 0
    n_strong_opening_calls: int = 0
    n_summary_calls: int = 0
    n_weak_retry_calls: int = 0
    strong_opening_input_tokens: int = 0
    strong_opening_output_tokens: int = 0
    summary_input_tokens: int = 0
    summary_output_tokens: int = 0
    unattributed_input_tokens: int = 0
    unattributed_output_tokens: int = 0
    token_totals_status: str = QuantityStatus.MEASURED.value
    weak_token_totals_status: str = QuantityStatus.NOT_APPLICABLE.value
    strong_token_totals_status: str = QuantityStatus.NOT_APPLICABLE.value
    token_totals_status_reason: str | None = None
    allocated_gpu_seconds_status: str = QuantityStatus.MEASURED.value
    exclusive_gpu_seconds_estimated: float = 0.0
    exclusive_gpu_seconds_status: str = QuantityStatus.ESTIMATED.value
    exclusive_gpu_seconds_provenance: str = EXCLUSIVE_GPU_SECONDS_PROVENANCE
    flops: ResourceQuantity = field(default_factory=_quantity_field)
    hbm_read_bytes: ResourceQuantity = field(default_factory=_quantity_field)
    hbm_write_bytes: ResourceQuantity = field(default_factory=_quantity_field)
    phase_counts: dict[str, int] = field(default_factory=dict)
    spans: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def strong_generation_output_tokens_act_correction(self) -> int:
        """Legacy generation total: solo act + correction, excluding opening."""
        return self.strong_act_output_tokens + self.strong_correction_output_tokens

    @property
    def strong_generation_output_tokens(self) -> int:
        """Autoregressive Strong tokens: act + correction + opening. Excludes verifier."""
        return (
            self.strong_generation_output_tokens_act_correction
            + self.strong_opening_output_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record with every reportable field, including per-call rows."""
        record: dict[str, Any] = {}
        for item in dataclass_fields(self):
            value = getattr(self, item.name)
            if item.name in {"calls", "spans"}:
                record[item.name] = list(value)
            elif item.name in {"flops", "hbm_read_bytes", "hbm_write_bytes"}:
                record[item.name] = value.to_dict()
            elif item.name == "phase_counts":
                record[item.name] = dict(value)
            else:
                record[item.name] = value
        record["strong_generation_output_tokens"] = self.strong_generation_output_tokens
        record["strong_generation_output_tokens_act_correction"] = (
            self.strong_generation_output_tokens_act_correction
        )
        return record


def _charged(span: InferenceSpan) -> bool:
    return span.charge_inference


def _has_explicit_disposition(span: InferenceSpan) -> bool:
    return span.committed_output_tokens + span.discarded_output_tokens > 0


def _committed_output(span: InferenceSpan) -> int:
    if _has_explicit_disposition(span):
        return span.committed_output_tokens
    if span.phase in (InferencePhase.DRAFT, InferencePhase.ACT):
        return span.output_tokens
    return 0


def _discarded_output(span: InferenceSpan) -> int:
    if _has_explicit_disposition(span):
        return span.discarded_output_tokens
    return 0


def _token_sum(
    spans: Iterable[InferenceSpan],
    attr: str,
    status_prop: str,
) -> int:
    total = 0
    for span in spans:
        if not _charged(span):
            continue
        field_status = getattr(span, status_prop)
        if field_status in {QuantityStatus.UNKNOWN, QuantityStatus.NOT_APPLICABLE}:
            continue
        total += int(getattr(span, attr))
    return total


def _duration_sum(spans: Iterable[InferenceSpan]) -> float:
    return sum(span.duration_seconds for span in spans if _charged(span))


def _gpu_sum(spans: Iterable[InferenceSpan]) -> float:
    return sum(span.gpu_seconds for span in spans if _charged(span))


def transport_attempt_count(spans: Iterable[InferenceSpan]) -> int:
    """Count dispatch attempts, including failed ones whose usage is unknown.

    If only the successful attempt was stored but it reports index N, N attempts
    are counted so a lost failed usage row is not silently dropped.
    """
    by_request: dict[str, list[InferenceSpan]] = {}
    anonymous = 0
    for span in spans:
        if not _charged(span):
            continue
        if not span.request_id:
            anonymous += 1
            continue
        by_request.setdefault(span.request_id, []).append(span)
    total = anonymous
    for group in by_request.values():
        total += max(len(group), max(item.transport_attempt_index for item in group))
    return total


def missing_transport_attempts(spans: Iterable[InferenceSpan]) -> int:
    charged = [span for span in spans if _charged(span)]
    return transport_attempt_count(charged) - len(charged)


def _coverage_if_unambiguous(items: list[ResourceQuantity]) -> float | None:
    coverages = [item.coverage for item in items]
    if any(coverage is None for coverage in coverages):
        return None
    unique = set(float(coverage) for coverage in coverages)
    if len(unique) != 1:
        return None
    return unique.pop()


def _item_interval(item: ResourceQuantity) -> tuple[float | None, float | None]:
    """Aggregatable [lower, upper] for one quantity.

    Complete points collapse to a degenerate interval. Incomplete ranges use
    ``observed_lower``/``observed_upper``; ``observed_estimated`` is the sum of
    estimated *points* and is never an interval upper bound.
    """
    complete = {
        QuantityStatus.MEASURED,
        QuantityStatus.ESTIMATED,
        QuantityStatus.LEGACY_UNSPECIFIED,
    }
    if item.status in complete and item.value is not None:
        value = float(item.value)
        return value, value
    if item.status is QuantityStatus.INCOMPLETE:
        if item.observed_lower is not None:
            lower = float(item.observed_lower)
        elif item.observed_subtotal is not None:
            lower = float(item.observed_subtotal)
        else:
            lower = None
        upper = None if item.observed_upper is None else float(item.observed_upper)
        return lower, upper
    return None, None


def _aggregate_interval(
    applicable: list[ResourceQuantity], *, missing_unknown: int
) -> tuple[float | None, float | None]:
    lowers: list[float] = []
    uppers: list[float] = []
    closed_upper = missing_unknown <= 0
    for item in applicable:
        lower, upper = _item_interval(item)
        if lower is not None:
            lowers.append(lower)
        if upper is None:
            closed_upper = False
        else:
            uppers.append(upper)
    observed_lower = sum(lowers) if lowers else None
    observed_upper = sum(uppers) if closed_upper and uppers else None
    return observed_lower, observed_upper


def _incomplete_merged(
    *,
    observed_name: str,
    observed_subtotal: float,
    measured_sum: float,
    estimated_sum: float,
    applicable: list[ResourceQuantity],
    missing_unknown: int,
) -> ResourceQuantity:
    observed_lower, observed_upper = _aggregate_interval(
        applicable, missing_unknown=missing_unknown
    )
    return ResourceQuantity(
        value=None,
        status=QuantityStatus.INCOMPLETE,
        provenance=f"incomplete_{observed_name}",
        coverage=None,
        observed_subtotal=observed_subtotal,
        observed_measured=measured_sum,
        observed_estimated=estimated_sum,
        observed_lower=observed_lower,
        observed_upper=observed_upper,
    )



def merge_quantities(
    quantities: Iterable[ResourceQuantity],
    *,
    observed_name: str,
    missing_unknown: int = 0,
) -> ResourceQuantity:
    """Sum quantities without upgrading status or filling unknowns with zero.

    Complete points sum to a point at the weakest status; any unknown or incomplete component
    (or ``missing_unknown`` lost attempts) makes the result incomplete with aggregated bounds.
    """
    items = list(quantities)
    applicable = [
        item for item in items if item.status is not QuantityStatus.NOT_APPLICABLE
    ]
    if not applicable and missing_unknown <= 0:
        if items:
            return ResourceQuantity(
                value=None,
                status=QuantityStatus.NOT_APPLICABLE,
                provenance="all_not_applicable",
            )
        return unknown_quantity(provenance="no_component")

    known = [
        item
        for item in applicable
        if item.status
        not in {QuantityStatus.UNKNOWN, QuantityStatus.INCOMPLETE}
        or item.observed_subtotal is not None
        or item.observed_lower is not None
    ]
    measured_sum = sum(
        float(item.value)
        for item in applicable
        if item.status is QuantityStatus.MEASURED and item.value is not None
    )
    estimated_sum = sum(
        float(item.value)
        for item in applicable
        if item.status is QuantityStatus.ESTIMATED and item.value is not None
    )
    legacy_sum = sum(
        float(item.value)
        for item in applicable
        if item.status is QuantityStatus.LEGACY_UNSPECIFIED and item.value is not None
    )
    incomplete_obs = sum(
        float(
            item.observed_subtotal
            if item.observed_subtotal is not None
            else (item.observed_lower or 0.0)
        )
        for item in applicable
        if item.status is QuantityStatus.INCOMPLETE
    )
    observed_subtotal = measured_sum + estimated_sum + legacy_sum + incomplete_obs
    has_unknown = missing_unknown > 0 or any(
        item.status in {QuantityStatus.UNKNOWN, QuantityStatus.INCOMPLETE}
        for item in applicable
    )
    ranks = [
        _STATUS_RANK[item.status.value]
        for item in applicable
        if item.status.value in _STATUS_RANK
    ]
    if missing_unknown > 0:
        ranks.append(_STATUS_RANK[QuantityStatus.INCOMPLETE.value])
    if has_unknown and known:
        return _incomplete_merged(
            observed_name=observed_name,
            observed_subtotal=observed_subtotal,
            measured_sum=measured_sum,
            estimated_sum=estimated_sum,
            applicable=applicable,
            missing_unknown=missing_unknown,
        )
    if has_unknown and not known:
        return unknown_quantity(provenance="no_server_counter")
    if not ranks:
        return unknown_quantity(provenance="no_server_counter")
    lowest = min(ranks)
    if lowest == _STATUS_RANK[QuantityStatus.MEASURED.value]:
        coverage = _coverage_if_unambiguous(applicable)
        return ResourceQuantity(
            value=measured_sum,
            status=QuantityStatus.MEASURED,
            provenance="sum_measured_components",
            coverage=coverage,
        )
    if lowest == _STATUS_RANK[QuantityStatus.ESTIMATED.value]:
        return ResourceQuantity(
            value=measured_sum + estimated_sum,
            status=QuantityStatus.ESTIMATED,
            provenance="sum_estimated_components",
            coverage=_coverage_if_unambiguous(applicable),
            observed_measured=measured_sum or None,
            observed_estimated=estimated_sum or None,
        )
    if lowest == _STATUS_RANK[QuantityStatus.LEGACY_UNSPECIFIED.value]:
        return ResourceQuantity(
            value=observed_subtotal,
            status=QuantityStatus.LEGACY_UNSPECIFIED,
            provenance="legacy_unspecified_sum",
            coverage=None,
            observed_subtotal=observed_subtotal,
            observed_measured=measured_sum or None,
            observed_estimated=estimated_sum or None,
        )
    return _incomplete_merged(
        observed_name=observed_name,
        observed_subtotal=observed_subtotal,
        measured_sum=measured_sum,
        estimated_sum=estimated_sum,
        applicable=applicable,
        missing_unknown=missing_unknown,
    )


def exclusive_gpu_seconds_estimated_from_spans(
    spans: Iterable[InferenceSpan],
) -> tuple[float, str, str]:
    """Enter exclusive GPU-seconds as an estimated duration x gpu_count proxy.

    This is the analyzer field ``exclusive_gpu_seconds_estimated``. It is not a
    rename of allocated GPU-seconds: occupancy stays on ``total_gpu_seconds``
    with ``allocated_gpu_seconds_status``.
    """
    charged = [span for span in spans if _charged(span)]
    value = sum(span.duration_seconds * span.gpu_count for span in charged)
    return (
        float(value),
        QuantityStatus.ESTIMATED.value,
        EXCLUSIVE_GPU_SECONDS_PROVENANCE,
    )


def _token_field_statuses(span: InferenceSpan) -> tuple[QuantityStatus, ...]:
    return (
        span.usage_status,
        span.resolved_input_status,
        span.resolved_output_status,
        span.resolved_cached_status,
        span.resolved_reasoning_status,
    )


def _token_totals_status_for_spans(
    spans: Iterable[InferenceSpan],
    *,
    missing_unknown: int,
) -> str:
    charged = [span for span in spans if _charged(span)]
    if not charged:
        return QuantityStatus.NOT_APPLICABLE.value
    statuses: list[QuantityStatus] = []
    for item in charged:
        statuses.extend(_token_field_statuses(item))
    if missing_unknown > 0:
        return QuantityStatus.INCOMPLETE.value
    if any(status is QuantityStatus.UNKNOWN for status in statuses):
        if any(
            status
            in {
                QuantityStatus.MEASURED,
                QuantityStatus.ESTIMATED,
                QuantityStatus.LEGACY_UNSPECIFIED,
                QuantityStatus.INCOMPLETE,
            }
            for status in statuses
        ):
            return QuantityStatus.INCOMPLETE.value
        return QuantityStatus.UNKNOWN.value
    if any(status is QuantityStatus.INCOMPLETE for status in statuses):
        return QuantityStatus.INCOMPLETE.value
    if any(status is QuantityStatus.LEGACY_UNSPECIFIED for status in statuses):
        return QuantityStatus.LEGACY_UNSPECIFIED.value
    return QuantityStatus.MEASURED.value


def _combine_role_token_status(
    weak_status: str,
    strong_status: str,
) -> tuple[str, str | None]:
    applicable = [
        status
        for status in (weak_status, strong_status)
        if status != QuantityStatus.NOT_APPLICABLE.value
    ]
    if not applicable:
        return QuantityStatus.MEASURED.value, None
    unique = set(applicable)
    if len(unique) == 1:
        return applicable[0], None
    reason = (
        "mixed_role_token_status:"
        f"weak={weak_status},strong={strong_status}"
    )
    return QuantityStatus.INCOMPLETE.value, reason


def _allocated_gpu_seconds_status(*, missing_unknown: int) -> str:
    if missing_unknown > 0:
        return QuantityStatus.INCOMPLETE.value
    return QuantityStatus.MEASURED.value


def _n_unknown_usage(spans: Iterable[InferenceSpan], *, missing_unknown: int) -> int:
    charged = [span for span in spans if _charged(span)]
    unknown_spans = sum(
        1
        for span in charged
        if span.usage_status
        in {QuantityStatus.UNKNOWN, QuantityStatus.INCOMPLETE}
        or span.resolved_input_status is QuantityStatus.UNKNOWN
        or span.resolved_output_status is QuantityStatus.UNKNOWN
        or span.resolved_cached_status is QuantityStatus.UNKNOWN
        or span.resolved_reasoning_status is QuantityStatus.UNKNOWN
    )
    return unknown_spans + missing_unknown


KNOWN_ESTIMATOR_IDS = frozenset({FLOPS_ESTIMATOR_ID, HBM_ESTIMATOR_ID})


def _summary_estimator_version(spans: Iterable[InferenceSpan]) -> str:
    """Ledger default is none until a charged span carries a known estimator id."""
    found: set[str] = set()
    for span in spans:
        for qty in (span.flops, span.hbm_read_bytes, span.hbm_write_bytes):
            if qty.status in {
                QuantityStatus.ESTIMATED,
                QuantityStatus.MEASURED,
                QuantityStatus.INCOMPLETE,
            }:
                first = str(qty.provenance or "").split(";", 1)[0]
                if first in KNOWN_ESTIMATOR_IDS:
                    found.add(first)
    if not found:
        return LEDGER_NO_ESTIMATOR
    if len(found) == 1:
        return next(iter(found))
    return "+".join(sorted(found))


class ResourceLedger:
    def __init__(self) -> None:
        self._spans: list[InferenceSpan] = []

    def add(self, span: InferenceSpan) -> bool:
        """Store a span. Exact duplicates are idempotent; conflicts raise.

        Cached replay and charged live inference use different identity keys so
        insertion order cannot drop real cost.
        """
        key = span.identity_key
        if key[0] == "anon":
            self._spans.append(span)
            return True
        for existing in self._spans:
            if existing.identity_key != key:
                continue
            if existing.fingerprint == span.fingerprint:
                return False
            raise LedgerConflictError(
                "conflicting spans for "
                f"{key}: {existing.fingerprint} vs {span.fingerprint}"
            )
        self._spans.append(span)
        return True

    @property
    def spans(self) -> tuple[InferenceSpan, ...]:
        return tuple(self._spans)

    def summarize(self) -> ResourceSummary:
        charged = [span for span in self._spans if _charged(span)]
        missing = missing_transport_attempts(charged)
        weak = [span for span in charged if span.role is ModelRole.WEAK]
        strong = [span for span in charged if span.role is ModelRole.STRONG]
        verifier = [
            span for span in strong if span.canonical_phase is CanonicalPhase.STRONG_VERIFY
        ]
        correction = [
            span
            for span in strong
            if span.canonical_phase is CanonicalPhase.STRONG_CORRECT
        ]
        act = [span for span in strong if span.canonical_phase is CanonicalPhase.STRONG_ACT]
        opening = [
            span
            for span in charged
            if span.canonical_phase is CanonicalPhase.STRONG_OPENING
        ]
        summary_spans = [
            span for span in charged if span.canonical_phase is CanonicalPhase.SUMMARY
        ]
        retries = [
            span for span in charged if span.canonical_phase is CanonicalPhase.WEAK_RETRY
        ]
        unattributed = [
            span for span in charged if span.attribution == "run_unattributed"
        ]
        failed = [span for span in charged if span.status != "ok"]
        phase_counts: dict[str, int] = {}
        for span in charged:
            name = span.canonical_phase.value
            phase_counts[name] = phase_counts.get(name, 0) + 1
        weak_token_status = _token_totals_status_for_spans(
            weak, missing_unknown=missing if weak else 0
        )
        strong_token_status = _token_totals_status_for_spans(
            strong, missing_unknown=missing if strong else 0
        )
        token_status, token_reason = _combine_role_token_status(
            weak_token_status, strong_token_status
        )
        exclusive_value, exclusive_status, exclusive_prov = (
            exclusive_gpu_seconds_estimated_from_spans(charged)
        )
        return ResourceSummary(
            total_gpu_seconds=_gpu_sum(charged),
            strong_gpu_seconds=_gpu_sum(strong),
            weak_input_tokens=_token_sum(weak, "input_tokens", "resolved_input_status"),
            weak_committed_output_tokens=sum(
                _committed_output(span)
                for span in weak
                if span.resolved_output_status
                not in {QuantityStatus.UNKNOWN, QuantityStatus.NOT_APPLICABLE}
            ),
            weak_discarded_output_tokens=sum(
                _discarded_output(span)
                for span in weak
                if span.resolved_output_status
                not in {QuantityStatus.UNKNOWN, QuantityStatus.NOT_APPLICABLE}
            ),
            strong_verifier_input_tokens=_token_sum(
                verifier, "input_tokens", "resolved_input_status"
            ),
            strong_verifier_output_tokens=_token_sum(
                verifier, "output_tokens", "resolved_output_status"
            ),
            strong_correction_input_tokens=_token_sum(
                correction, "input_tokens", "resolved_input_status"
            ),
            strong_correction_output_tokens=_token_sum(
                correction, "output_tokens", "resolved_output_status"
            ),
            strong_act_input_tokens=_token_sum(
                act, "input_tokens", "resolved_input_status"
            ),
            strong_act_output_tokens=_token_sum(
                act, "output_tokens", "resolved_output_status"
            ),
            cached_input_tokens=_token_sum(
                charged, "cached_input_tokens", "resolved_cached_status"
            ),
            weak_gpu_seconds=_gpu_sum(weak),
            weak_output_tokens=_token_sum(
                weak, "output_tokens", "resolved_output_status"
            ),
            weak_cached_input_tokens=_token_sum(
                weak, "cached_input_tokens", "resolved_cached_status"
            ),
            weak_reasoning_tokens=_token_sum(
                weak, "reasoning_tokens", "resolved_reasoning_status"
            ),
            strong_input_tokens=_token_sum(
                strong, "input_tokens", "resolved_input_status"
            ),
            strong_output_tokens=_token_sum(
                strong, "output_tokens", "resolved_output_status"
            ),
            strong_cached_input_tokens=_token_sum(
                strong, "cached_input_tokens", "resolved_cached_status"
            ),
            strong_reasoning_tokens=_token_sum(
                strong, "reasoning_tokens", "resolved_reasoning_status"
            ),
            llm_call_seconds=_duration_sum(charged),
            weak_call_seconds=_duration_sum(weak),
            strong_call_seconds=_duration_sum(strong),
            strong_verify_call_seconds=_duration_sum(verifier),
            strong_correction_call_seconds=_duration_sum(correction),
            strong_act_call_seconds=_duration_sum(act),
            n_calls=len(charged),
            n_weak_calls=len(weak),
            n_strong_calls=len(strong),
            n_strong_verify_calls=len(verifier),
            n_strong_correction_calls=len(correction),
            calls=tuple(span.to_dict() for span in charged),
            n_transport_attempts=transport_attempt_count(charged),
            n_failed_calls=len(failed),
            n_cached_replays=sum(
                1 for span in self._spans if span.source == "cached_replay"
            ),
            n_unattributed=len(unattributed),
            n_unknown_usage=_n_unknown_usage(charged, missing_unknown=missing),
            n_strong_opening_calls=len(opening),
            n_summary_calls=len(summary_spans),
            n_weak_retry_calls=len(retries),
            strong_opening_input_tokens=_token_sum(
                opening, "input_tokens", "resolved_input_status"
            ),
            strong_opening_output_tokens=_token_sum(
                opening, "output_tokens", "resolved_output_status"
            ),
            summary_input_tokens=_token_sum(
                summary_spans, "input_tokens", "resolved_input_status"
            ),
            summary_output_tokens=_token_sum(
                summary_spans, "output_tokens", "resolved_output_status"
            ),
            unattributed_input_tokens=_token_sum(
                unattributed, "input_tokens", "resolved_input_status"
            ),
            unattributed_output_tokens=_token_sum(
                unattributed, "output_tokens", "resolved_output_status"
            ),
            token_totals_status=token_status,
            weak_token_totals_status=weak_token_status,
            strong_token_totals_status=strong_token_status,
            token_totals_status_reason=token_reason,
            allocated_gpu_seconds_status=_allocated_gpu_seconds_status(
                missing_unknown=missing
            ),
            exclusive_gpu_seconds_estimated=exclusive_value,
            exclusive_gpu_seconds_status=exclusive_status,
            exclusive_gpu_seconds_provenance=exclusive_prov,
            flops=merge_quantities(
                (span.flops for span in charged),
                observed_name="flops",
                missing_unknown=missing,
            ),
            hbm_read_bytes=merge_quantities(
                (span.hbm_read_bytes for span in charged),
                observed_name="hbm_read",
                missing_unknown=missing,
            ),
            hbm_write_bytes=merge_quantities(
                (span.hbm_write_bytes for span in charged),
                observed_name="hbm_write",
                missing_unknown=missing,
            ),
            phase_counts=phase_counts,
            spans=tuple(span.to_record() for span in self._spans),
            estimator_version=_summary_estimator_version(charged),
        )
