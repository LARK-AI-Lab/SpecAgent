"""Decision-row provenance for the AutomationBench SpecAgent lane.

Pure helpers (no verifiers/openai/httpx imports), so tests can exercise the row builder under the
repository environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from specagent.protocol.side_effects import action_mutates_state
from specagent.protocol.types import PrefixVerdict
from specagent.runtime.gateway import GatewayDecision


def _verdict_brief(verdict: PrefixVerdict | None) -> dict[str, Any] | None:
    if verdict is None:
        return None
    return {
        "accepted_steps": verdict.accepted_steps,
        "reason": verdict.reason or None,
        "format_retries": verdict.format_retries,
        "diagnostic_status": verdict.diagnostic_status,
        "format_failure_kind": verdict.format_failure_kind or None,
        # Salvage audit: foreign keys dropped on the way in, and why an emitted
        # replacement_action did not survive.
        "discarded_verdict_fields": list(verdict.discarded_verdict_fields) or None,
        "reason_on_full_accept": verdict.reason_on_full_accept or None,
        "replacement_validation_failure_kind": (
            verdict.replacement_validation_failure_kind or None
        ),
        # True when a structurally valid replacement survived parsing and the
        # runtime guards; a guard-failed one is None here and named in
        # replacement_validation_failure_kind.
        "fused_proposed": verdict.replacement_action is not None,
        # Diagnostics: on parse failure keep a tail of the raw verifier output
        # so live format_failure rows can be root-caused without replay.
        "raw_tail": (verdict.raw or "")[-400:] or None
        if verdict.diagnostic_status == "format_failure"
        else None,
    }


def _accept_class(accepted: int | None, candidate_length: int) -> str | None:
    if accepted is None:
        return None
    if accepted >= candidate_length:
        return "full"
    if accepted == 0:
        return "zero"
    return "partial"


def _action_type(action: Any) -> str | None:
    if not isinstance(action, Mapping):
        return None
    if str(action.get("type") or "") == "message":
        return "reply"
    name = action.get("tool") or action.get("name")
    return str(name) if name else None


def _delivered_tool_calls(response: Mapping[str, Any] | None) -> int:
    if not isinstance(response, Mapping):
        return 0
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], Mapping):
        return 0
    message = choices[0].get("message") or {}
    return len(message.get("tool_calls") or [])


def decision_fields(decision: GatewayDecision) -> dict[str, Any]:
    """Auditable provenance for one AutomationBench request.

    Complements (never replaces) the per-attempt InferenceSpan accounting: the verdict, accept
    class, kept prefix, how the repair was routed and how it ended, format retries, withheld
    mutations, the repair's action type, and how many tool calls were released for execution.
    """
    verdict = decision.verdict
    candidate_length = decision.candidate_length
    accepted = verdict.accepted_steps if verdict is not None else None
    rejected_suffix = (
        list(decision.candidate_steps[accepted:]) if accepted is not None else []
    )
    withheld_mutations = sum(
        1
        for step in rejected_suffix
        if isinstance(step, Mapping)
        and action_mutates_state(step.get("action"), known_tools=None)
    )
    return {
        "source": decision.source,
        "boundary": decision.boundary,
        "mutating": decision.mutating,
        "verdict": _verdict_brief(verdict),
        "accept_class": _accept_class(accepted, candidate_length),
        "kept_steps": accepted,
        "candidate_length": candidate_length,
        "strong_route": decision.source in {"strong_correction", "strong_fused_correction"},
        "fused_correction": decision.source == "strong_fused_correction",
        "format_retries": verdict.format_retries if verdict is not None else 0,
        "withheld_mutations": withheld_mutations,
        "executed_rejected_steps": decision.executed_rejected_steps,
        "corrector_action_type": _action_type(decision.correction_action),
        "correction_actions": len(decision.correction_actions),
        "corrector_generation_count": decision.corrector_generation_count,
        "correction_tool_call_count": decision.correction_tool_call_count,
        "correction_suppressed_reply": decision.correction_suppressed_reply,
        "correction_withheld_replays": decision.correction_withheld_replays,
        "replacement_validation_failure_count": (
            decision.replacement_validation_failure_count
        ),
        "delivered_tool_calls": _delivered_tool_calls(decision.response),
    }
