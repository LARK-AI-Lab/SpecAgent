"""Context views for Weak drafting, Strong verification, and clean correction."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any

from specagent.protocol.types import AgentStep, ModelRole, TaskContext


class ContextPolicy(str, Enum):
    MIXED_THINKING = "mixed_thinking"
    ACTION_OBSERVATION_ONLY = "action_observation_only"
    PRIVATE_THINKING = "private_thinking"


def _include_reasoning(step: AgentStep, consumer: ModelRole, policy: ContextPolicy) -> bool:
    if policy is ContextPolicy.MIXED_THINKING:
        return True
    if policy is ContextPolicy.PRIVATE_THINKING:
        return step.actor is consumer
    return False


def render_step(
    step: AgentStep,
    *,
    consumer: ModelRole,
    policy: ContextPolicy,
    force_reasoning: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "actor": step.actor.value,
        "action": step.action,
        "observation": step.observation,
        "terminal": step.terminal,
    }
    if (step.metadata or {}).get("pending"):
        record["pending"] = True
    if step.reasoning and (force_reasoning or _include_reasoning(step, consumer, policy)):
        record["reasoning"] = step.reasoning
    return record


def committed_context(
    history: Iterable[AgentStep],
    *,
    consumer: ModelRole,
    policy: ContextPolicy,
) -> list[dict[str, Any]]:
    return [render_step(step, consumer=consumer, policy=policy) for step in history]


def verification_payload(
    history: Iterable[AgentStep],
    candidate: Iterable[AgentStep],
    *,
    policy: ContextPolicy,
    task: TaskContext | None = None,
) -> dict[str, Any]:
    """The verifier's view: committed history, the task, and the candidate under review.

    Field order is a cache layout: providers reuse a byte prefix, so the append-only history
    leads and the candidate, which differs on every call, is last. The task block sits behind
    the history because later user messages can be folded into its instruction.
    """
    history = list(history)
    candidate = list(candidate)
    payload: dict[str, Any] = {
        "confirmed_history": committed_context(
            history,
            consumer=ModelRole.STRONG,
            policy=policy,
        )
    }
    candidate_steps = [
        {
            "index": index,
            **render_step(
                step,
                consumer=ModelRole.STRONG,
                policy=policy,
                force_reasoning=True,
            ),
        }
        for index, step in enumerate(candidate, start=1)
    ]
    if task is not None:
        invoked = {
            str(step.action.get("tool"))
            for step in candidate
            if isinstance(step.action, dict) and step.action.get("tool") is not None
        }
        contracts = task.tool_contracts or {}
        payload["task"] = {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "system_constraints": list(task.system_constraints),
        }
        anchors: dict[str, Any] = {"task_id": task.task_id}
        workspace = (task.metadata or {}).get("workspace_dir")
        if workspace:
            anchors["workspace_dir"] = workspace
        payload["anchors"] = anchors
        payload["invoked_tool_contracts"] = {
            name: contracts[name] for name in sorted(invoked) if name in contracts
        }
    payload["proposed_trajectory"] = candidate_steps
    return payload


def correction_payload(
    history: Iterable[AgentStep],
    *,
    policy: ContextPolicy,
) -> dict[str, Any]:
    """Return only the committed prefix; rejected suffixes have no input here."""
    return {
        "confirmed_history": committed_context(
            history,
            consumer=ModelRole.STRONG,
            policy=policy,
        )
    }
