"""Pruning of verifier payload context that the verifier does not need."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Joins later user messages (or compaction summaries) onto the task instruction.
LATER_CONTEXT_SEPARATOR = "\n\n[Later user message or summary]\n"


def prune_payload(
    payload: dict[str, Any], *, history_reasoning: bool, compaction_summaries: bool,
    candidate_reasoning: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Drop context the verifier holds a better copy of, or that is stale by construction.

    Returns the pruned payload and a receipt of what was removed.

    - ``history_reasoning=False`` drops committed steps' reasoning: it describes decisions that
      have already been judged.
    - ``candidate_reasoning=False`` drops the proposed steps' reasoning: the verdict is about the
      action, and the drafter's rationale is unreviewed text arguing for the step being judged.
    - ``compaction_summaries=False`` drops compaction summaries appended to the task instruction:
      the verifier keeps the full step history, so a summary of it is redundant and stale.
    """
    removed = {
        "reasoning_steps": 0, "reasoning_chars": 0, "summaries": 0, "summary_chars": 0,
        "candidate_reasoning_steps": 0, "candidate_reasoning_chars": 0,
    }
    if not candidate_reasoning:
        proposed = []
        for step in payload.get("proposed_trajectory") or []:
            step = dict(step)
            text = step.pop("reasoning", None)
            if isinstance(text, str) and text:
                removed["candidate_reasoning_steps"] += 1
                removed["candidate_reasoning_chars"] += len(text)
            proposed.append(step)
        payload["proposed_trajectory"] = proposed
    if not history_reasoning:
        history = []
        for step in payload.get("confirmed_history") or []:
            step = dict(step)
            text = step.pop("reasoning", None)
            if isinstance(text, str) and text:
                removed["reasoning_steps"] += 1
                removed["reasoning_chars"] += len(text)
            history.append(step)
        payload["confirmed_history"] = history
    if not compaction_summaries:
        metadata = payload.get("request_evidence_metadata")
        if isinstance(metadata, Mapping):
            metadata = dict(metadata)
            records = metadata.get("later_context_records")
            if isinstance(records, list):
                kept = []
                for record in records:
                    if (
                        isinstance(record, Mapping)
                        and record.get("kind") == "compaction_summary_candidate"
                    ):
                        removed["summaries"] += 1
                        removed["summary_chars"] += len(str(record.get("text") or ""))
                        continue
                    kept.append(record)
                metadata["later_context_records"] = kept
            payload["request_evidence_metadata"] = metadata
        task = payload.get("task")
        if isinstance(task, Mapping):
            task = dict(task)
            instruction = task.get("instruction")
            if isinstance(instruction, str) and LATER_CONTEXT_SEPARATOR in instruction:
                head, *rest = instruction.split(LATER_CONTEXT_SEPARATOR)
                kept = [
                    part
                    for part in rest
                    if "was compacted into the following summary" not in part
                ]
                removed["summary_chars"] += sum(
                    len(part) for part in rest if part not in kept
                )
                task["instruction"] = LATER_CONTEXT_SEPARATOR.join([head, *kept])
            payload["task"] = task
    return payload, removed
