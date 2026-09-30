"""The SpecAgent profile for AutomationBench.

Pure helpers (no verifiers/openai/httpx imports), so tests can exercise the configuration under
the repository environment.

- Irrevocable review: mutating tool calls and replies (a reply ends the task and cannot be
  unsent) face Strong verification; a batch of ``horizon`` reads commits without review, so the
  horizon is a draft batching cap, not a verification interval.
- The verdict is ``{reason, accepted_steps, replacement_action}``. A rejection at a mutation
  boundary may carry one tool-only ``replacement_action``, which commits directly; otherwise
  Strong takes one repair turn that is never re-verified. Message boundaries do not fuse: the
  replacement is tool-only and cannot repair a rejected reply.
- Hidden reasoning never enters the delivered trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from specagent.runtime.gateway import GatewayConfig

# GatewayConfig fields of the profile. Recorded in manifest.json so a resumed run is checked
# against the same configuration.
PROFILE_CONFIG: dict[str, Any] = {
    "horizon": 8,
    "fuse_correction_boundaries": ("mutation",),
    # The environment validates action arguments: a schema-invalid or unknown-tool repair call or
    # replacement is delivered and fails as an error observation, as it does in single-model runs.
    # Execution-safety guards the environment cannot soften stay on.
    "strict_action_validation": False,
    # Room for a deliberation before the verdict JSON plus the fused replacement payload.
    "verifier_max_tokens": 4096,
    # Committed reasoning describes decisions already judged: drop it from the verifier payload
    # and from the repair turn's transcript. The candidate's own reasoning still ships.
    "verifier_history_reasoning": False,
    "correction_history_reasoning": False,
    "verifier_prompt_suffix": (
        "AutomationBench policy: when rejecting a tool-call step whose correct repair is "
        "a grounded call of the same tool, emit that repaired call as replacement_action. "
        "Reserve replacement_action=null for repairs that need a new observation, repairs "
        "that are themselves a reply, or arguments that cannot be grounded in the evidence. "
        "Independently, reject a tool-call step that repeats a call already committed in "
        "confirmed_history with materially identical arguments when the earlier observation "
        "already provided the information it seeks; the correct next step is to act on that "
        "information or to reply, not to query again. "
        "Before accepting a reply that claims the task is done, check each deliverable the "
        "task requires against confirmed_history: if any required artifact, update, or "
        "notification has no executed step or observation supporting it, reject the reply "
        "(replacement_action=null) so correction can continue the missing work."
    ),
    # Show each cleared rejection reason to the Weak drafter once, on its next draft.
    "weak_review_hint": True,
    # Reads never reach the verifier, so an all-repeat read batch is withheld deterministically
    # and routed to Strong repair.
    "repeat_call_guard": True,
}

# What the manifest records: the configuration plus labels describing the protocol.
MANIFEST_PROFILE: dict[str, Any] = {
    "output_keys": ["reason", "accepted_steps", "replacement_action"],
    "correction_policy": "rp_fused",
    "review_scope": "irrevocable",
    "horizon_role": "draft_batching_cap",
    # Repair turns may carry several tool calls: the gateway validates ids and replays first,
    # then the harness executes them natively in provider order.
    "multi_call_execution": "validate_all_then_execute_native",
    **PROFILE_CONFIG,
}


def rp_fused_gateway_config(
    weak_model: str,
    strong_model: str,
    *,
    weak_extra_body: Mapping[str, Any] | None = None,
    strong_extra_body: Mapping[str, Any] | None = None,
    verifier_extra_body: Mapping[str, Any] | None = None,
) -> GatewayConfig:
    """GatewayConfig for the AutomationBench profile."""
    return GatewayConfig(
        weak_model=weak_model,
        strong_model=strong_model,
        weak_extra_body=weak_extra_body,
        strong_extra_body=strong_extra_body,
        verifier_extra_body=verifier_extra_body,
        **PROFILE_CONFIG,
    )
