"""SpecAgent accounting adapter for AutomationBench: pure-python core.

This module deliberately imports only stdlib + ``specagent.protocol.*`` so the
repo test suite (``pythonpath = ["src"]``) exercises it without verifiers /
openai / datasets installed. The verifiers-bound recording clients live in
``solo_client.py`` and ``rp_client.py``; the runner in ``run.py``.

Identity scheme (existing SpecAgent span fields, no new schema):

* ``task_id``              = AutomationBench ``info["task_name"]`` (fallback
                             ``example_<example_id>``), stored on
                             ``InferenceSpan.parent_task_id``.
* ``trial_id``             = verifiers ``state["trajectory_id"]`` — unique per
                             rollout attempt — stored on
                             ``InferenceSpan.parent_episode_id``.
* ``request_id``           = ``f"{trial_id}#r{seq:04d}"`` — one per logical
                             model call (one ``get_native_response``).
* ``transport_attempt_id`` = ``f"{request_id}:{n}"`` — one per HTTP attempt,
                             index from ``transport_attempt_index``.

Honesty rules (the ResourceLedger contract):

* Missing provider usage -> ``ParsedUsage(provenance="missing_usage")`` with
  all statuses UNKNOWN. Never zero-filled.
* FLOPs / HBM / exclusive GPU-seconds stay UNKNOWN on the span; point
  estimates are produced only at rollup time via ``spec_lookup`` (model-arch
  derived) — never silently written into the ledger.
* Failed / cancelled transport attempts produce spans with ``status != "ok"``
  and unknown usage so task totals degrade to INCOMPLETE instead of
  undercounting.
* Benchmark infrastructure (assertion scorer, tool execution, world init) is
  pure Python inside the rollout — it has no inference cost and therefore no
  spans; ``paper_cost_scope_payload()`` documents it as excluded.
"""

from __future__ import annotations

import contextvars
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from specagent.protocol.accounting import (
    InferencePhase,
    InferenceSpan,
    ParsedUsage,
    QuantityStatus,
    ResourceLedger,
    missing_transport_attempts,
    transport_attempt_count,
)
from specagent.protocol.resource_estimate import (
    paper_resources_from_ledger,
    parsed_usage_from_provider_blob,
    serving_spec_from_model_name,
    span_from_parsed_usage,
)
from specagent.protocol.types import ModelRole

ADAPTER_VERSION = "automationbench_adapter"
RESOURCE_PROVENANCE = "automationbench_no_server_counter"

# This module is imported under two names (``specagent_adapter`` via the
# benchmarks dir on sys.path, and ``benchmarks.automationbench.
# specagent_adapter`` via the repo root). Both must resolve to ONE module
# object: CURRENT_REQUEST is process-global identity, and a duplicate module
# would silently split request binding from transport recording.
sys.modules.setdefault("specagent_adapter", sys.modules[__name__])
sys.modules.setdefault(
    "benchmarks.automationbench.specagent_adapter", sys.modules[__name__]
)

# is_aborted() in automationbench.export treats an assistant tail with tool
# calls as an infra cutoff only when ``steps < step_cap - 2``. Keep the same
# margin so adapter classification matches the benchmark's own export logic.
STEP_CAP_MARGIN = 2

UNRESOLVED_TOOL_CALL = "unresolved_tool_call"


@dataclass(frozen=True)
class RequestContext:
    """Identity bound to one logical model call (one get_native_response)."""

    task_id: str
    trial_id: str
    request_id: str
    attribution: str  # "task" or "run_unattributed"


CURRENT_REQUEST: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "automationbench_request", default=None
)


# ---------------------------------------------------------------------------
# State identity helpers
# ---------------------------------------------------------------------------


def info_from_state(state: Any) -> dict[str, Any]:
    """Return the task ``info`` dict; tolerates the JSON-string form."""
    if not isinstance(state, Mapping):
        return {}
    info = state.get("info") or {}
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except (json.JSONDecodeError, TypeError):
            info = {}
    return dict(info) if isinstance(info, Mapping) else {}


def task_id_from_state(state: Any) -> str:
    """Stable per-task identity: the benchmark's own ``info["task_name"]``."""
    info = info_from_state(state)
    name = info.get("task_name")
    if isinstance(name, str) and name:
        return name
    if isinstance(state, Mapping):
        example = state.get("example_id")
        if example is not None:
            return f"example_{example}"
    return "unknown_task"


def trial_id_from_state(state: Any) -> str | None:
    """Per-rollout attempt identity: verifiers' ``state["trajectory_id"]``."""
    if not isinstance(state, Mapping):
        return None
    value = state.get("trajectory_id")
    return str(value) if value else None


# ---------------------------------------------------------------------------
# Attempt recorder
# ---------------------------------------------------------------------------


class AttemptRecorder:
    """Records one InferenceSpan per HTTP transport attempt into a ledger.

    Spans carry ``parent_task_id`` (task_id) and ``parent_episode_id``
    (trial_id), so a retried task (new trajectory_id) produces a distinct
    trial rollup under the same task key. Spans are per *transport attempt*:
    a logical request retried 3 times yields 3 spans sharing ``request_id``
    with contiguous ``transport_attempt_index`` values, letting
    ``missing_transport_attempts`` detect dropped rows honestly.
    """

    def __init__(
        self,
        *,
        role: ModelRole | str,
        phase: InferencePhase = InferencePhase.ACT,
        ledger: ResourceLedger | None = None,
        intent: str | None = None,
        model: str | None = None,
    ) -> None:
        self.role = role if isinstance(role, ModelRole) else ModelRole(str(role))
        self.phase = phase
        self.intent = intent
        self.model = model
        self.ledger = ledger if ledger is not None else ResourceLedger()
        self._request_seq: dict[str, int] = {}
        self._attempt_seq: dict[str, int] = {}
        self.anomalies: list[dict[str, Any]] = []
        # request_id -> {dp_rank, route_key, route_source} rows, persisted as
        # rank_map.jsonl (sidecar; InferenceSpan schema stays untouched).
        self.rank_rows: list[dict[str, Any]] = []

    def note_rank(
        self,
        request_id: str,
        *,
        dp_rank: int | None,
        route_key: str | None = None,
        route_source: str | None = None,
    ) -> None:
        """Record which DP rank served one logical request (per-request
        attribution for the duration×GPU proxy and shard-balance audits)."""
        self.rank_rows.append(
            {
                "request_id": request_id,
                "dp_rank": dp_rank,
                "route_key": route_key,
                "route_source": route_source,
            }
        )

    def begin_request(self, state: Any) -> RequestContext:
        """Bind identity for one logical model call.

        Calls outside a rollout (state=None) are recorded as
        ``run_unattributed`` anomalies — never silently attributed to a task.
        """
        if isinstance(state, Mapping):
            task_id = task_id_from_state(state)
            trial_id = trial_id_from_state(state) or f"no-trial:{task_id}"
            attribution = "task"
        else:
            task_id = "unattributed"
            trial_id = "unattributed"
            attribution = "run_unattributed"
            self.anomalies.append(
                {"kind": "unbound_state", "detail": repr(type(state))}
            )
        seq = self._request_seq.get(trial_id, 0) + 1
        self._request_seq[trial_id] = seq
        return RequestContext(
            task_id=task_id,
            trial_id=trial_id,
            request_id=f"{trial_id}#r{seq:04d}",
            attribution=attribution,
        )

    def next_attempt_index(self, request_id: str) -> int:
        seq = self._attempt_seq.get(request_id, 0) + 1
        self._attempt_seq[request_id] = seq
        return seq

    def record_attempt(
        self,
        ctx: RequestContext,
        *,
        attempt_index: int,
        duration_seconds: float,
        parsed: ParsedUsage | None = None,
        status: str = "ok",
        provider_trace_id: str | None = None,
        role: ModelRole | str | None = None,
        phase: InferencePhase | None = None,
        intent: str | None = None,
        request_id: str | None = None,
    ) -> InferenceSpan:
        """Append one transport-attempt span. FLOPs/HBM stay UNKNOWN.

        The SpecAgent lane records several phase-labelled sub-calls under one bound
        request; ``role``/``phase``/``intent``/``request_id`` override the
        recorder defaults for those spans.
        """
        usage = parsed if parsed is not None else ParsedUsage(provenance="missing_usage")
        span = span_from_parsed_usage(
            usage,
            role=(
                (role if isinstance(role, ModelRole) else ModelRole(str(role)))
                if role is not None
                else self.role
            ),
            phase=phase if phase is not None else self.phase,
            duration_seconds=max(0.0, float(duration_seconds)),
            committed=(status == "ok") or None,
            request_id=request_id or ctx.request_id,
            transport_attempt_index=attempt_index,
            parent_task_id=ctx.task_id,
            parent_episode_id=ctx.trial_id,
            provider_trace_id=provider_trace_id,
            status=status,
            intent=intent if intent is not None else self.intent,
            attribution=ctx.attribution,
            source="live",
        )
        self.ledger.add(span)
        return span

    # -- run-level views ----------------------------------------------------
    def spans_for_trial(self, trial_id: str) -> list[InferenceSpan]:
        return [s for s in self.ledger.spans if s.parent_episode_id == trial_id]

    def transport_holes(self) -> int:
        """Attempt rows inferred missing from non-contiguous indices."""
        return missing_transport_attempts(
            [s for s in self.ledger.spans if s.charge_inference]
        )


# ---------------------------------------------------------------------------
# Provider response parsing
# ---------------------------------------------------------------------------


def parse_response_body(content: Any) -> tuple[ParsedUsage, str | None]:
    """Extract ``(usage, provider_response_id)`` from a chat.completion body.

    Returns ``missing_usage`` when the usage block is absent/malformed and
    ``None`` provider id when there is no ``id`` — never fabricates values.
    """
    if content is None:
        return ParsedUsage(provenance="missing_usage"), None
    body: Any = content
    if isinstance(content, (bytes, bytearray, str)):
        try:
            body = json.loads(content)
        except (ValueError, TypeError):
            return ParsedUsage(provenance="missing_usage"), None
    if not isinstance(body, Mapping):
        return ParsedUsage(provenance="missing_usage"), None
    usage = parsed_usage_from_provider_blob(body.get("usage"))
    provider_id = body.get("id")
    return usage, (str(provider_id) if provider_id else None)


def parsed_usage_from_response(response: Any) -> tuple[ParsedUsage, str | None]:
    """Best-effort usage extraction for either an httpx.Response-like object
    (``content`` bytes) or a pydantic ChatCompletion-like object (``usage``
    attribute / ``model_dump``)."""
    usage_attr = getattr(response, "usage", None)
    if usage_attr is not None:
        parsed = parsed_usage_from_provider_blob(
            usage_attr.model_dump() if hasattr(usage_attr, "model_dump") else usage_attr
        )
        provider_id = getattr(response, "id", None)
        return parsed, (str(provider_id) if provider_id else None)
    return parse_response_body(getattr(response, "content", None))


# ---------------------------------------------------------------------------
# Fail-closed trajectory validation
# ---------------------------------------------------------------------------


def _messages_of(row: Mapping[str, Any]) -> list[Any]:
    msgs = row.get("messages")
    if msgs is None:
        msgs = row.get("completion")
    if msgs is None:
        return []
    return list(msgs) if isinstance(msgs, (list, tuple)) else []


def _msg_get(msg: Any, key: str) -> Any:
    if isinstance(msg, Mapping):
        return msg.get(key)
    return getattr(msg, key, None)


def has_dangling_tool_call(row: Mapping[str, Any]) -> bool:
    """Last message is an assistant turn with tool_calls that were never
    answered — the trajectory was cut mid-work."""
    msgs = _messages_of(row)
    if not msgs:
        return False
    last = msgs[-1]
    return _msg_get(last, "role") == "assistant" and bool(_msg_get(last, "tool_calls"))


def validate_export_task(
    task: Mapping[str, Any], *, step_cap: int, step_cap_margin: int = STEP_CAP_MARGIN
) -> dict[str, Any]:
    """Fail-closed classification of one ``export_results`` task row."""
    flags: list[str] = []
    if task.get("error") or task.get("errors"):
        flags.append("error")
    if task.get("died_on_refusal"):
        flags.append("refusal")
    if has_dangling_tool_call(task):
        steps = int(task.get("steps") or 0)
        if steps < step_cap - step_cap_margin:
            # Cut mid-work with budget left — the benchmark's own abort
            # signature. Never report this as a completed task.
            flags.append(UNRESOLVED_TOOL_CALL)
        else:
            flags.append("tool_call_at_step_cap")
    return {"ok": not flags, "flags": flags}


def validate_rollout_output(
    output: Mapping[str, Any], *, step_cap: int, step_cap_margin: int = STEP_CAP_MARGIN
) -> dict[str, Any]:
    """Fail-closed classification of a verifiers ``RolloutOutput`` dict."""
    flags: list[str] = []
    if output.get("error"):
        flags.append("error")
    stop = output.get("stop_condition")
    if stop == "timeout_reached" or output.get("timed_out"):
        flags.append("timeout")
    if not output.get("is_completed", False):
        flags.append("incomplete")
    if output.get("is_truncated"):
        flags.append("truncated")
    if has_dangling_tool_call(output):
        trajectory = output.get("trajectory") or ()
        steps = int(output.get("steps") or len(trajectory) or 0)
        if steps and steps >= step_cap - step_cap_margin:
            flags.append("tool_call_at_step_cap")
        else:
            flags.append(UNRESOLVED_TOOL_CALL)
    return {"ok": not flags, "flags": flags}


# ---------------------------------------------------------------------------
# Run summary: joins rollout outputs, export rows and ledger spans on
# task_id / trial_id.
# ---------------------------------------------------------------------------


def _usage_totals(spans: Sequence[InferenceSpan]) -> dict[str, Any]:
    """Sum token fields over charged spans, honoring per-field status.

    A field reports ``incomplete`` when some spans are measured and some are
    unknown; ``unknown`` when no span measured it; ``measured`` when every
    charged span measured it.
    """

    def _sum(attr: str, status_attr: str) -> tuple[int, str]:
        total = 0
        saw_measured = False
        saw_unknown = False
        for span in spans:
            if not span.charge_inference:
                continue
            status = getattr(span, status_attr)
            if status in {QuantityStatus.UNKNOWN, QuantityStatus.NOT_APPLICABLE}:
                saw_unknown = True
                continue
            saw_measured = True
            total += int(getattr(span, attr))
        if saw_measured and saw_unknown:
            return total, "incomplete"
        if saw_measured:
            return total, "measured"
        return 0, "unknown"

    input_tokens, input_status = _sum("input_tokens", "resolved_input_status")
    output_tokens, output_status = _sum("output_tokens", "resolved_output_status")
    cached, cached_status = _sum("cached_input_tokens", "resolved_cached_status")
    reasoning, reasoning_status = _sum("reasoning_tokens", "resolved_reasoning_status")
    return {
        "input_tokens": input_tokens if input_status != "unknown" else None,
        "output_tokens": output_tokens if output_status != "unknown" else None,
        "cached_input_tokens": cached if cached_status != "unknown" else None,
        "reasoning_tokens": reasoning if reasoning_status != "unknown" else None,
        "input_status": input_status,
        "output_status": output_status,
        "cached_status": cached_status,
        "reasoning_status": reasoning_status,
    }


def _end_to_end_seconds(output: Mapping[str, Any]) -> float | None:
    """Per-task wall seconds (first model call -> scoring end).

    ``RolloutTiming.total`` is 0.0 when scoring never ran — treat that as
    unknown rather than a real zero.
    """
    timing = output.get("timing")
    if isinstance(timing, Mapping):
        total = timing.get("total")
        if isinstance(total, (int, float)) and total > 0:
            return float(total)
    return None


def build_run_summary(
    *,
    raw_outputs: Sequence[Mapping[str, Any]],
    export_tasks: Sequence[Mapping[str, Any]],
    recorder: AttemptRecorder,
    model: str,
    role: ModelRole | str,
    benchmark_makespan_seconds: float | None,
    step_cap: int,
) -> dict[str, Any]:
    """Join verifiers RolloutOutputs + export task rows + spans by identity.

    Join key: ``trajectory_id`` (trial_id) on the output ==
    ``parent_episode_id`` on spans; ``info.task_name`` (task_id) ==
    ``parent_task_id`` == export row ``name``.
    """
    export_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in export_tasks:
        name = row.get("name")
        if name is not None:
            export_by_name.setdefault(str(name), []).append(row)

    tasks: list[dict[str, Any]] = []
    all_flags: list[str] = []
    for output in raw_outputs:
        task_id = task_id_from_state(output)
        trial_id = trial_id_from_state(output) or ""
        spans = [s for s in recorder.ledger.spans if s.parent_episode_id == trial_id]
        export_rows = export_by_name.get(task_id, [])
        export_row = export_rows[0] if export_rows else {}

        validation = validate_rollout_output(output, step_cap=step_cap)
        export_validation = (
            validate_export_task(export_row, step_cap=step_cap)
            if export_row
            else {"ok": None, "flags": ["missing_export_row"]}
        )
        for flag in validation["flags"] + list(export_validation["flags"] or []):
            all_flags.append(f"{task_id}:{flag}")

        holes = missing_transport_attempts([s for s in spans if s.charge_inference])
        usage = _usage_totals(spans)
        tasks.append(
            {
                "task_id": task_id,
                "trial_id": trial_id or None,
                "score": output.get("reward"),
                "passed": export_row.get("passed"),
                "metrics": dict(output.get("metrics") or {}),
                "is_completed": bool(output.get("is_completed", False)),
                "stop_condition": output.get("stop_condition"),
                "end_to_end_wall_seconds": _end_to_end_seconds(output),
                "n_transport_attempts": transport_attempt_count(
                    [s for s in spans if s.charge_inference]
                ),
                "n_failed_attempts": sum(1 for s in spans if s.status != "ok"),
                "missing_transport_attempts": holes,
                "usage": usage,
                "validation": validation,
                "export_validation": export_validation,
            }
        )

    summary = recorder.ledger.summarize()
    return {
        "adapter_version": ADAPTER_VERSION,
        "cost_scope": "agent_generation_only",
        "model": model,
        "role": role.value if isinstance(role, ModelRole) else str(role),
        "n_tasks": len(tasks),
        "tasks": tasks,
        "benchmark_makespan_seconds": benchmark_makespan_seconds,
        "run_flags": all_flags,
        "anomalies": list(recorder.anomalies),
        "ledger_totals": {
            "n_transport_attempts": summary.n_transport_attempts,
            "n_failed_calls": summary.n_failed_calls,
            "n_unknown_usage": summary.n_unknown_usage,
            "n_unattributed": summary.n_unattributed,
            "token_totals_status": summary.token_totals_status,
            "missing_transport_attempts": recorder.transport_holes(),
            "llm_call_seconds": summary.llm_call_seconds,
        },
    }


# ---------------------------------------------------------------------------
# Paper-resource bridge: the analysis-side view of the ledger.
# ---------------------------------------------------------------------------


def spec_lookup_for_model(model: str | None):
    """Return a spec_lookup binding every span to the serving spec of ``model``.

    Returns None only for an empty name. Unknown names still produce a
    name-only spec: ``estimate_span_flops`` then marks FLOPs
    ``unsupported_model`` (UNKNOWN) — identity is preserved, nothing is
    fabricated.
    """
    spec = serving_spec_from_model_name(model)
    if spec is None:
        return None

    def _lookup(span: InferenceSpan):
        return spec

    return _lookup


def paper_payload(
    ledger: ResourceLedger, *, model: str | None
) -> dict[str, Any] | None:
    """Analysis-consumable resource payload.

    Spans keep UNKNOWN FLOPs/HBM in the ledger; point estimates live only in
    this derived payload when the model resolves to a registered arch.
    """
    return paper_resources_from_ledger(
        ledger, spec_lookup=spec_lookup_for_model(model)
    )
