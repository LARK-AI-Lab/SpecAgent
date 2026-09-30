"""Shared FLOPs / HBM-read enrich and rollup helper.

Adapters record spans (ParsedUsage, reservation/span identity, failed transport); this module
enriches those spans for the paper metric and rolls them up (attempt, request, task, benchmark).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from specagent.protocol.accounting import (
    CanonicalPhase,
    InferencePhase,
    InferenceSpan,
    ParsedUsage,
    QuantityStatus,
    ResourceLedger,
    ResourceQuantity,
    merge_quantities,
    token_fields_from_usage,
    unknown_quantity,
)
from specagent.protocol.cache_report import apply_sglang_cache_report_to_span
from specagent.protocol.estimator_ids import (
    FLOPS_ESTIMATOR_ID,
    HBM_ESTIMATOR_ID,
    LEDGER_NO_ESTIMATOR,
    PAPER_HBM_READ_METRIC_NAME,
    PAPER_RESOURCE_HELPER_ID,
)
from specagent.protocol.flops import estimate_span_flops, load_arch
from specagent.protocol.hbm import estimate_span_hbm
from specagent.protocol.paper import bytes_to_gb, paper_write_unknown
from specagent.protocol.types import ModelRole

PAPER_COST_SCOPE = "agent_generation_only"

COVERED_CANONICAL_PHASES = (
    CanonicalPhase.WEAK_DRAFT,
    CanonicalPhase.WEAK_RETRY,
    CanonicalPhase.STRONG_VERIFY,
    CanonicalPhase.STRONG_CORRECT,
    CanonicalPhase.STRONG_OPENING,
    CanonicalPhase.SUMMARY,
    CanonicalPhase.WEAK_ACT,
    CanonicalPhase.STRONG_ACT,
)

# Served / HF names -> architecture id (see flops._ARCH_FILES).
_ARCH_ALIASES = {
    "Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen3.5-397B-A17B": "Qwen3.5-397B-A17B",
    "Qwen3.5-397B-A17B-BF16": "Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-397B-A17B": "Qwen3.5-397B-A17B",
    "DeepSeek-V4-Flash-0731": "DeepSeek-V4-Flash-0731",
    "deepseek-v4-flash-0731": "DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "DeepSeek-V4-Flash-0731",
    "GLM-5": "GLM-5",
    "glm-5": "GLM-5",
    "zai-org/GLM-5": "GLM-5",
}

SpecLookup = Callable[[InferenceSpan], "ServingSpec | None"]


@dataclass(frozen=True, slots=True)
class ServingSpec:
    """Which architecture served a span. A model_id without an architecture stays unknown."""

    model_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceRollup:
    level: str
    key: str
    n_spans: int
    n_calls: int
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    latency_s: float
    flops: ResourceQuantity
    hbm_read_bytes: ResourceQuantity
    hbm_read_gb: ResourceQuantity
    hbm_write_bytes: ResourceQuantity
    phases: tuple[str, ...]
    model_identities: tuple[str, ...]
    estimator_version: str
    coverage: float | None
    provenance: str
    metric_name: str = PAPER_HBM_READ_METRIC_NAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "key": self.key,
            "n_spans": self.n_spans,
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "latency_s": self.latency_s,
            "flops": self.flops.to_dict(),
            "hbm_read_bytes": self.hbm_read_bytes.to_dict(),
            "hbm_read_gb": self.hbm_read_gb.to_dict(),
            "hbm_write_bytes": self.hbm_write_bytes.to_dict(),
            "phases": list(self.phases),
            "model_identities": list(self.model_identities),
            "estimator_version": self.estimator_version,
            "coverage": self.coverage,
            "provenance": self.provenance,
            "metric_name": self.metric_name,
        }

    def paper_main_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("hbm_write_bytes", None)
        payload["metric_name"] = PAPER_HBM_READ_METRIC_NAME
        return payload


def resolve_architecture_id(name: str | None) -> str | None:
    """Map a served/HF model name onto a supported architecture id, else None."""
    if not name or not str(name).strip():
        return None
    text = str(name).strip()
    if text in _ARCH_ALIASES:
        return _ARCH_ALIASES[text]
    return None


def serving_spec_from_arch_id(model_id: str) -> ServingSpec:
    return ServingSpec(model_id=load_arch(model_id).model_id)


def serving_spec_from_model_name(name: str | None) -> ServingSpec | None:
    if not name or not str(name).strip():
        return None
    resolved = resolve_architecture_id(name)
    if resolved is None:
        return ServingSpec(model_id=str(name).strip())
    return serving_spec_from_arch_id(resolved)


def serving_spec_from_tau_agent(agent: Any, span: InferenceSpan) -> ServingSpec | None:
    """Pick Weak vs Strong serving identity from the Tau agent that reserved the span."""
    role = span.role
    client = None
    agent_role = getattr(agent, "role", None) or getattr(agent, "model_role", None)
    if role is ModelRole.STRONG:
        client = getattr(agent, "strong_client", None)
        if client is None and agent_role is ModelRole.STRONG:
            client = getattr(agent, "client", None)
    else:
        client = getattr(agent, "weak_client", None)
        if client is None and agent_role is not ModelRole.STRONG:
            client = getattr(agent, "client", None)
    model = getattr(client, "model", None) if client is not None else None
    if not model:
        model = getattr(agent, "model", None)
    return serving_spec_from_model_name(model)


def spec_lookup_for_tau_agent(agent: Any) -> SpecLookup:
    return lambda span: serving_spec_from_tau_agent(agent, span)


def _estimator_ids_from(*quantities: ResourceQuantity) -> str:
    found: list[str] = []
    for qty in quantities:
        first = str(qty.provenance or "").split(";", 1)[0]
        if first in {FLOPS_ESTIMATOR_ID, HBM_ESTIMATOR_ID}:
            if first not in found:
                found.append(first)
    if not found:
        return LEDGER_NO_ESTIMATOR
    if PAPER_RESOURCE_HELPER_ID not in found:
        found.append(PAPER_RESOURCE_HELPER_ID)
    return "+".join(found)


def enrich_span(span: InferenceSpan, spec: ServingSpec | None) -> InferenceSpan:
    """Fill FLOPs + HBM-read on one request span. Write stays unknown.

    On SGLang cache-report models (cache_report.py) cached tokens count as compute reuse and
    an omitted cached_tokens is a measured zero; other usage stays as parsed. Missing usage or
    an unsupported model stays unknown/incomplete, never a fill-in.
    """
    write = paper_write_unknown()
    if spec is None or not (spec.model_id or "").strip():
        missing = unknown_quantity(provenance="missing_model_identity_no_resource_upgrade")
        return replace(
            span,
            flops=missing,
            hbm_read_bytes=missing,
            hbm_write_bytes=write,
        )
    span = apply_sglang_cache_report_to_span(span, model_id=spec.model_id)
    flops = estimate_span_flops(span, spec.model_id)
    read, _write = estimate_span_hbm(span, spec.model_id)
    return replace(
        span,
        flops=flops,
        hbm_read_bytes=read,
        hbm_write_bytes=write,
    )


def _mapping_get(blob: Any, name: str) -> Any:
    if blob is None:
        return None
    if isinstance(blob, Mapping):
        return blob.get(name)
    return getattr(blob, name, None)


def parsed_usage_from_provider_blob(usage: Any) -> ParsedUsage:
    """Map an OpenAI-style usage blob onto ParsedUsage.

    Field rules match ``openai_client.parse_usage``: a present key with value 0
    is measured zero; a missing key or missing blob is unknown. Gateway HTTP
    responses are mappings, so this reads keys rather than only attributes.
    Already-serialized ParsedUsage dicts pass through ``from_dict``.
    """
    if usage is None:
        return ParsedUsage(provenance="missing_usage")
    if isinstance(usage, ParsedUsage):
        return usage
    if isinstance(usage, Mapping) and "input_status" in usage:
        return ParsedUsage.from_dict(usage)
    if isinstance(usage, Mapping) and not usage:
        return ParsedUsage(provenance="missing_usage")

    prompt_details = _mapping_get(usage, "prompt_tokens_details")
    completion_details = _mapping_get(usage, "completion_tokens_details")
    prompt = _mapping_get(usage, "prompt_tokens")
    completion = _mapping_get(usage, "completion_tokens")
    cached_raw = _mapping_get(prompt_details, "cached_tokens")
    reasoning_raw = _mapping_get(completion_details, "reasoning_tokens")
    if reasoning_raw is None:
        reasoning_raw = _mapping_get(usage, "reasoning_tokens")

    def _count(raw: Any, *, present: bool) -> tuple[int, QuantityStatus]:
        if not present or raw is None:
            return 0, QuantityStatus.UNKNOWN
        return int(raw or 0), QuantityStatus.MEASURED

    input_tokens, input_status = _count(prompt, present=prompt is not None)
    output_tokens, output_status = _count(completion, present=completion is not None)
    cache_present = prompt_details is not None and cached_raw is not None
    cached_tokens, cached_status = _count(cached_raw, present=cache_present)
    reasoning_tokens, reasoning_status = _count(
        reasoning_raw, present=reasoning_raw is not None
    )
    if all(
        status is QuantityStatus.UNKNOWN
        for status in (input_status, output_status, cached_status, reasoning_status)
    ):
        return ParsedUsage(provenance="missing_usage")
    return ParsedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        input_status=input_status,
        output_status=output_status,
        cached_status=cached_status,
        reasoning_status=reasoning_status,
        provenance="openai_usage",
    )


def span_from_parsed_usage(
    parsed: ParsedUsage,
    *,
    role: ModelRole,
    phase: InferencePhase,
    duration_seconds: float,
    committed: bool | None = None,
    **extra: Any,
) -> InferenceSpan:
    """Build an InferenceSpan from ParsedUsage via token_fields_from_usage."""
    fields = token_fields_from_usage(parsed)
    if committed is True:
        extra.setdefault("committed_output_tokens", parsed.output_tokens)
        extra.setdefault("discarded_output_tokens", 0)
    elif committed is False:
        extra.setdefault("committed_output_tokens", 0)
        extra.setdefault("discarded_output_tokens", parsed.output_tokens)
    request_id = extra.get("request_id")
    index = int(extra.get("transport_attempt_index") or 1)
    if request_id and extra.get("transport_attempt_id") is None:
        extra["transport_attempt_id"] = f"{request_id}:{index}"
    return InferenceSpan(
        role=role,
        phase=phase,
        duration_seconds=duration_seconds,
        **fields,
        **extra,
    )


def _token_sum(spans: Sequence[InferenceSpan], field: str, status_attr: str) -> int | None:
    total = 0
    unknown = False
    for span in spans:
        status = getattr(span, status_attr)
        if status in {QuantityStatus.UNKNOWN, QuantityStatus.NOT_APPLICABLE}:
            unknown = True
            continue
        total += int(getattr(span, field))
    if unknown and total == 0:
        return None
    return total


def _model_identity(span: InferenceSpan) -> str:
    for qty in (span.flops, span.hbm_read_bytes):
        prov = str(qty.provenance or "")
        for part in prov.split(";"):
            if part.startswith("arch="):
                return part.split("=", 1)[1]
    return "unknown"


def rollup_spans(
    spans: Iterable[InferenceSpan],
    *,
    level: str,
    key: str,
) -> ResourceRollup:
    items = tuple(spans)
    charged = tuple(span for span in items if span.charge_inference)
    flops = merge_quantities(
        (span.flops for span in charged),
        observed_name="flops",
    )
    read = merge_quantities(
        (span.hbm_read_bytes for span in charged),
        observed_name="hbm_read",
    )
    write = paper_write_unknown()
    if any(
        span.flops.status is QuantityStatus.INCOMPLETE
        or span.hbm_read_bytes.status is QuantityStatus.INCOMPLETE
        for span in charged
    ) and flops.status is QuantityStatus.ESTIMATED:
        flops = ResourceQuantity(
            value=None,
            status=QuantityStatus.INCOMPLETE,
            provenance="incomplete_span_not_upgraded_to_point;" + flops.provenance,
            coverage=flops.coverage,
            observed_lower=flops.value if flops.value is not None else flops.observed_lower,
            observed_upper=flops.value if flops.value is not None else flops.observed_upper,
            observed_subtotal=flops.value if flops.value is not None else flops.observed_subtotal,
        )
    phases = tuple(dict.fromkeys(span.canonical_phase.value for span in charged))
    identities = tuple(dict.fromkeys(_model_identity(span) for span in charged))
    estimator = _estimator_ids_from(
        *(span.flops for span in charged),
        *(span.hbm_read_bytes for span in charged),
    )
    coverages = [
        qty.coverage
        for span in charged
        for qty in (span.flops, span.hbm_read_bytes)
        if qty.coverage is not None
    ]
    coverage = coverages[0] if coverages and len(set(coverages)) == 1 else None
    provenance = ";".join(
        (
            PAPER_RESOURCE_HELPER_ID,
            f"level={level}",
            "paper_write=unknown",
        )
    )
    return ResourceRollup(
        level=level,
        key=key,
        n_spans=len(charged),
        n_calls=len(charged),
        input_tokens=_token_sum(charged, "input_tokens", "resolved_input_status"),
        output_tokens=_token_sum(charged, "output_tokens", "resolved_output_status"),
        cached_input_tokens=_token_sum(
            charged, "cached_input_tokens", "resolved_cached_status"
        ),
        latency_s=sum(span.duration_seconds for span in charged),
        flops=flops,
        hbm_read_bytes=read,
        hbm_read_gb=bytes_to_gb(read),
        hbm_write_bytes=write,
        phases=phases,
        model_identities=identities,
        estimator_version=estimator,
        coverage=coverage,
        provenance=provenance,
    )


def _group_key(span: InferenceSpan, level: str) -> str:
    if level == "attempt":
        if span.request_id and span.transport_attempt_id:
            return f"{span.request_id}/{span.transport_attempt_id}"
        if span.request_id:
            return f"{span.request_id}:{span.transport_attempt_index}"
        return "attempt:anon"
    if level == "request":
        return span.request_id or "request:anon"
    if level == "task":
        return span.parent_task_id or "task:unattributed"
    return "benchmark"


def rollup_by_level(
    spans: Sequence[InferenceSpan],
    level: str,
) -> tuple[ResourceRollup, ...]:
    grouped: dict[str, list[InferenceSpan]] = {}
    for span in spans:
        grouped.setdefault(_group_key(span, level), []).append(span)
    return tuple(
        rollup_spans(group, level=level, key=key)
        for key, group in grouped.items()
    )


def rollup_hierarchy(spans: Sequence[InferenceSpan]) -> dict[str, tuple[ResourceRollup, ...]]:
    """attempt -> request -> task -> benchmark."""
    return {
        "attempt": rollup_by_level(spans, "attempt"),
        "request": rollup_by_level(spans, "request"),
        "task": rollup_by_level(spans, "task"),
        "benchmark": (rollup_spans(spans, level="benchmark", key="benchmark"),),
    }


def paper_cost_scope_payload() -> dict[str, Any]:
    """Declare what the paper agent-generation payload is and is not.

    Retriever FLOPs / HBM-read / GPU-seconds are explicit zeros: shared
    Qwen3-Embedding-8B infra is not incremental protocol cost. Judge, exclusive
    GPU-seconds, and the Tau2 user simulator stay excluded/unknown and must not
    be filled with a measured zero. Retriever wall-time still belongs in
    end-to-end wall-clock, not this payload.
    """
    return {
        "cost_scope": PAPER_COST_SCOPE,
        "out_of_scope": {
            "qwen3_embedding_8b_retriever": {
                "flops": 0,
                "hbm_read_bytes": 0,
                "gpu_seconds": 0,
                "included_in_agent_generation": False,
                "reason": (
                    "shared_retrieval_infra_across_arms;"
                    "not_incremental_protocol_cost;"
                    "wall_time_in_e2e_only"
                ),
            },
            "judge": {
                "status": "excluded",
                "value": None,
                "reason": "offline_eval_not_agent_generation;not_zero_fill",
            },
            "exclusive_gpu_seconds": {
                "status": "unknown",
                "value": None,
                "reason": "not_in_paper_agent_generation_payload;not_zero_fill",
            },
            "tau2_user_simulator": {
                "status": "excluded",
                "value": None,
                "reason": "report_in_separate_column_not_protocol_agent_generation",
            },
        },
    }


def paper_resources_from_spans(
    spans: Sequence[InferenceSpan],
    *,
    spec_lookup: SpecLookup | None = None,
) -> dict[str, Any]:
    items = tuple(spans)
    if spec_lookup is not None:
        items = tuple(enrich_span(span, spec_lookup(span)) for span in items)
    tree = rollup_hierarchy(items)
    payload = {
        "metric_name": PAPER_HBM_READ_METRIC_NAME,
        "estimator_helper": PAPER_RESOURCE_HELPER_ID,
        "batch_assumption": "batch_1_equivalent_not_measured_dram",
        "hierarchy": {
            level: [row.paper_main_dict() for row in rows]
            for level, rows in tree.items()
        },
    }
    payload.update(paper_cost_scope_payload())
    return payload


def paper_resources_from_ledger(
    ledger: ResourceLedger | None,
    *,
    spec_lookup: SpecLookup | None = None,
) -> dict[str, Any] | None:
    """Paper resource payload for a ledger.

    Does not rewrite the ledger: stored spans keep unknown FLOPs/HBM, and estimates are
    computed on copies when spec_lookup is provided.
    """
    if ledger is None:
        return None
    return paper_resources_from_spans(ledger.spans, spec_lookup=spec_lookup)


def paper_resources_from_tau_agent(agent: Any) -> dict[str, Any] | None:
    """Roll up paper estimates from a Tau agent's already-recorded spans."""
    ledger = getattr(agent, "ledger", None)
    if ledger is None:
        return None
    return paper_resources_from_ledger(
        ledger, spec_lookup=spec_lookup_for_tau_agent(agent)
    )


_GATEWAY_PHASES: dict[str, tuple[ModelRole, InferencePhase, str | None]] = {
    "weak_act": (ModelRole.WEAK, InferencePhase.DRAFT, None),
    "strong_verify": (ModelRole.STRONG, InferencePhase.VERIFICATION, None),
    "strong_correct": (ModelRole.STRONG, InferencePhase.CORRECTION, None),
}


def _gateway_role_phase(phase_name: str) -> tuple[ModelRole, InferencePhase, str | None]:
    mapped = _GATEWAY_PHASES.get(phase_name)
    if mapped is not None:
        return mapped
    role, phase, intent = ModelRole.WEAK, InferencePhase.UNKNOWN, None
    if "strong" in phase_name:
        role = ModelRole.STRONG
    return role, phase, intent
