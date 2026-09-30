"""Tau2 adapter onto the shared ResourceLedger.

Tau2 does not estimate architecture FLOPs or HBM. Every agent in this package
constructs spans through this module so:

- FLOPs / HBM stay ``unknown`` unless a caller passes a real ``ResourceQuantity``
  (measured/estimated with provenance). Bare numbers and ``duration * gpu_count``
  are rejected. specagent.protocol.resource_estimate fills paper estimates from
  the recorded token counts afterwards.
- parent task/episode and transport-attempt identity are filled in one place.
- failed HTTP attempts become spans with unknown usage, not silent holes and
  not token zeros dressed up as measured work.

This is not a parallel ledger. ``ResourceLedger`` remains the store.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from specagent.protocol.accounting import (
    InferencePhase,
    InferenceSpan,
    ParsedUsage,
    QuantityStatus,
    ResourceQuantity,
    token_fields_from_usage,
    unknown_quantity,
)
from specagent.protocol.types import ModelRole

TAU2_RESOURCE_PROVENANCE = "tau2_no_server_counter"


def tau2_unknown_quantity() -> ResourceQuantity:
    return unknown_quantity(provenance=TAU2_RESOURCE_PROVENANCE)


def _as_quantity(value: Any, *, what: str) -> ResourceQuantity:
    if value is None:
        return tau2_unknown_quantity()
    if isinstance(value, ResourceQuantity):
        return value
    if isinstance(value, (int, float)):
        raise ValueError(
            f"tau2 must not fill {what} from a bare number; "
            "duration*gpu_count is allocated GPU-seconds, not FLOPs/HBM"
        )
    if isinstance(value, Mapping):
        return ResourceQuantity.from_dict(value)
    raise TypeError(f"{what} must be a ResourceQuantity, mapping, or None")


def make_tau2_span(
    *,
    role: ModelRole,
    phase: InferencePhase,
    input_tokens: int,
    output_tokens: int,
    duration_seconds: float,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    committed_output_tokens: int = 0,
    discarded_output_tokens: int = 0,
    request_id: str | None = None,
    transport_attempt_id: str | None = None,
    transport_attempt_index: int = 1,
    parent_request_id: str | None = None,
    parent_task_id: str | None = None,
    parent_episode_id: str | None = None,
    status: str = "ok",
    usage_status: QuantityStatus = QuantityStatus.UNKNOWN,
    source: str = "live",
    intent: str | None = None,
    cache_provenance: str = "unspecified",
    input_tokens_status: QuantityStatus | None = None,
    output_tokens_status: QuantityStatus | None = None,
    cached_tokens_status: QuantityStatus | None = None,
    reasoning_tokens_status: QuantityStatus | None = None,
    flops: ResourceQuantity | Mapping[str, Any] | None = None,
    hbm_read_bytes: ResourceQuantity | Mapping[str, Any] | None = None,
    hbm_write_bytes: ResourceQuantity | Mapping[str, Any] | None = None,
    charge_inference: bool | None = None,
) -> InferenceSpan:
    """Build a tau2 span. FLOPs/HBM default to unknown; never inferred."""
    if charge_inference is None:
        charge_inference = source != "cached_replay"
    attempt_id = transport_attempt_id
    if request_id and attempt_id is None:
        attempt_id = f"{request_id}:{transport_attempt_index}"
    return InferenceSpan(
        role=role,
        phase=phase,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_seconds=duration_seconds,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        committed_output_tokens=committed_output_tokens,
        discarded_output_tokens=discarded_output_tokens,
        request_id=request_id,
        transport_attempt_id=attempt_id,
        transport_attempt_index=transport_attempt_index,
        parent_request_id=parent_request_id,
        parent_task_id=parent_task_id,
        parent_episode_id=parent_episode_id,
        status=status,
        usage_status=usage_status,
        cache_provenance=cache_provenance,
        source=source,
        intent=intent,
        charge_inference=charge_inference,
        flops=_as_quantity(flops, what="flops"),
        hbm_read_bytes=_as_quantity(hbm_read_bytes, what="hbm_read_bytes"),
        hbm_write_bytes=_as_quantity(hbm_write_bytes, what="hbm_write_bytes"),
        input_tokens_status=input_tokens_status,
        output_tokens_status=output_tokens_status,
        cached_tokens_status=cached_tokens_status,
        reasoning_tokens_status=reasoning_tokens_status,
    )


def make_tau2_failed_transport_span(
    *,
    role: ModelRole,
    phase: InferencePhase,
    duration_seconds: float,
    request_id: str | None = None,
    transport_attempt_id: str | None = None,
    transport_attempt_index: int = 1,
    parent_task_id: str | None = None,
    parent_episode_id: str | None = None,
    parent_request_id: str | None = None,
    intent: str | None = None,
    status: str = "failed",
) -> InferenceSpan:
    """One HTTP attempt that produced no usage. Tokens stay 0 with unknown status."""
    return make_tau2_span(
        role=role,
        phase=phase,
        input_tokens=0,
        output_tokens=0,
        duration_seconds=duration_seconds,
        request_id=request_id,
        transport_attempt_id=transport_attempt_id,
        transport_attempt_index=transport_attempt_index,
        parent_request_id=parent_request_id,
        parent_task_id=parent_task_id,
        parent_episode_id=parent_episode_id,
        status=status,
        usage_status=QuantityStatus.UNKNOWN,
        intent=intent,
    )


def span_ids_from_agent(agent: Any) -> dict[str, Any]:
    request_id = getattr(agent, "_request_id", None) or None
    index = int(getattr(agent, "_transport_attempt_index", 1) or 1)
    attempt_id = f"{request_id}:{index}" if request_id else None
    return {
        "request_id": request_id,
        "transport_attempt_id": attempt_id,
        "transport_attempt_index": index,
        "parent_task_id": getattr(agent, "task_id", None),
        "parent_episode_id": getattr(agent, "_attempt_id", None) or None,
    }


_ID_KEYS = (
    "request_id",
    "transport_attempt_id",
    "transport_attempt_index",
    "parent_request_id",
    "parent_task_id",
    "parent_episode_id",
)


def span_id_kwargs(ids: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _ID_KEYS:
        if ids and key in ids and ids[key] is not None:
            out[key] = ids[key]
    return out


def _attempt_key(ids: Mapping[str, Any] | None) -> str | None:
    if not ids:
        return None
    return ids.get("transport_attempt_id") or (
        f"{ids['request_id']}:{ids.get('transport_attempt_index') or 1}"
        if ids.get("request_id")
        else None
    )


def mark_span_recorded(agent: Any, ids: Mapping[str, Any] | None) -> None:
    key = _attempt_key(ids)
    if not key:
        return
    recorded = getattr(agent, "_recorded_span_keys", None)
    if recorded is None:
        agent._recorded_span_keys = set()
        recorded = agent._recorded_span_keys
    recorded.add(key)


def span_already_recorded(agent: Any, ids: Mapping[str, Any] | None) -> bool:
    key = _attempt_key(ids)
    if not key:
        return False
    return key in getattr(agent, "_recorded_span_keys", set())


def parsed_usage_from_result(result: Any) -> ParsedUsage:
    """Read provider usage. Missing blob → unknown, never measured zeros."""
    raw = getattr(result, "raw_response", None)
    blob = raw.get("_specagent_usage") if isinstance(raw, Mapping) else None
    if not blob:
        return ParsedUsage(provenance="missing_usage")
    return ParsedUsage.from_dict(blob)


def usage_fields_from_result(result: Any) -> dict[str, Any]:
    """Span kwargs for usage_status / per-field status / cache_provenance / tokens."""
    return token_fields_from_usage(parsed_usage_from_result(result))


def begin_episode(agent: Any) -> str:
    """Mint one parent_episode_id for a tau2 trial/conversation."""
    episode_id = uuid4().hex
    agent._attempt_id = episode_id
    agent._request_id = ""
    agent._transport_attempt_index = 1
    agent._last_attempt_duration = 0.0
    agent._recorded_span_keys = set()
    return episode_id


def begin_logical_request(agent: Any) -> str:
    """Mint one request_id for a logical model call (transport retries share it)."""
    request_id = uuid4().hex
    agent._request_id = request_id
    agent._transport_attempt_index = 1
    return request_id


def run_until_complete_dispatched(
    agent: Any,
    loop: Any,
    coro: Any,
    *,
    role: ModelRole,
    phase: InferencePhase,
) -> tuple[Any, float, dict[str, Any]]:
    """Run one HTTP coroutine as a new logical request; record a failed span on exception."""
    begin_logical_request(agent)
    ids = span_ids_from_agent(agent)
    started = time.perf_counter()
    try:
        result = loop.run_until_complete(coro)
    except asyncio.CancelledError:
        record_failed_transport(
            agent,
            role=role,
            phase=phase,
            duration_seconds=time.perf_counter() - started,
            ids=ids,
            status="cancelled",
        )
        raise
    except BaseException:
        record_failed_transport(
            agent,
            role=role,
            phase=phase,
            duration_seconds=time.perf_counter() - started,
            ids=ids,
        )
        raise
    duration = time.perf_counter() - started
    _attach_ids(result, ids)
    return result, duration, ids


def _attach_ids(result: Any, ids: Mapping[str, Any]) -> None:
    try:
        result._tau2_span_ids = ids
    except Exception:
        pass


def record_result_span(
    agent: Any,
    result: Any,
    *,
    role: ModelRole,
    phase: InferencePhase,
    duration_seconds: float,
    committed: bool = True,
    ids: Mapping[str, Any] | None = None,
    status: str = "ok",
) -> None:
    """Record a successful (or charged-but-dropped) attempt using provider usage."""
    ledger = getattr(agent, "ledger", None)
    if ledger is None:
        return
    resolved = dict(ids) if ids else dict(
        getattr(result, "_tau2_span_ids", None) or span_ids_from_agent(agent)
    )
    if span_already_recorded(agent, resolved):
        return
    usage = usage_fields_from_result(result)
    out = int(usage["output_tokens"] or 0)
    ledger.add(
        make_tau2_span(
            role=role,
            phase=phase,
            duration_seconds=duration_seconds,
            committed_output_tokens=out if committed else 0,
            discarded_output_tokens=0 if committed else out,
            status=status,
            **usage,
            **span_id_kwargs(resolved),
        )
    )
    mark_span_recorded(agent, resolved)


def record_failed_transport(
    agent: Any,
    *,
    role: ModelRole,
    phase: InferencePhase,
    duration_seconds: float,
    ids: Mapping[str, Any] | None = None,
    status: str = "failed",
) -> None:
    """Record one reserved attempt that produced no usage. Pass dispatch-time ids."""
    ledger = getattr(agent, "ledger", None)
    if ledger is None:
        return
    resolved = dict(ids) if ids else span_ids_from_agent(agent)
    if span_already_recorded(agent, resolved):
        return
    ledger.add(
        make_tau2_failed_transport_span(
            role=role,
            phase=phase,
            duration_seconds=duration_seconds,
            status=status,
            **span_id_kwargs(resolved),
        )
    )
    mark_span_recorded(agent, resolved)
