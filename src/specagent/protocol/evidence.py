"""Evidence-preserving verifier inputs; no tool execution or model decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from typing import Any

from jsonschema import SchemaError
from jsonschema.validators import validator_for
from referencing.exceptions import Unresolvable

TASK_SEMANTICS = ("generic", "stateful", "qa", "dialogue")


class VerifierInputError(ValueError):
    """Verification could not run on complete evidence; this is not a rejection."""


def content_hash(value: Any) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def input_token_upper_bound(messages: Sequence[Mapping[str, Any]]) -> int:
    """Conservative serialized-input size bound, NOT a measured token count.

    This deliberately avoids pretending that a provider tokenizer is available. Callers must
    treat the result as a diagnostic upper bound, not as the actual prompt-token count.
    """
    return 256 + sum(64 + len(json.dumps(m, ensure_ascii=False).encode()) for m in messages)


def check_input_budget(messages: Sequence[Mapping[str, Any]], limit: int | None) -> None:
    if limit is not None:
        if limit < 1:
            raise ValueError("verifier input token limit must be positive")
        bound = input_token_upper_bound(messages)
        if bound > limit:
            raise VerifierInputError(
                f"input_incomplete: conservative serialized-input bound {bound} exceeds "
                f"configured bound {limit}; "
                "no evidence was silently removed and no semantic verdict was produced"
            )


def _has_external_reference(schema: Any) -> bool:
    if isinstance(schema, dict):
        if any(
            isinstance(schema.get(key), str) and not schema[key].startswith("#")
            for key in ("$ref", "$dynamicRef", "$recursiveRef")
        ):
            return True
        return any(_has_external_reference(value) for value in schema.values())
    return isinstance(schema, list) and any(_has_external_reference(value) for value in schema)


def schema_check(step: Mapping[str, Any], index: int, contracts: Mapping[str, Any]) -> dict:
    """Validate one rendered step against its supplied tool contract without executing it."""
    result: dict[str, Any] = {"step_ref": f"candidate:{index}", "status": "unavailable"}
    action = step.get("action")
    if not isinstance(action, Mapping) or action.get("type") == "message":
        result["status"] = "not_applicable"
        return result
    tool = str(action.get("tool") or "")
    contract = contracts.get(tool)
    if contract is None:
        result["detail"] = "Tool is absent from the supplied tool contracts."
        return result
    function = contract.get("function", contract) if isinstance(contract, Mapping) else {}
    schema = function.get("parameters") if isinstance(function, Mapping) else None
    if not isinstance(schema, (dict, bool)):
        result["detail"] = "No argument schema was supplied; not proof of invalidity."
        return result
    if _has_external_reference(schema):
        result["detail"] = "External schema references are not fetched during verification."
        return result
    try:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        errors = sorted(
            validator.iter_errors(action.get("arguments")),
            key=lambda e: (str(list(e.absolute_path)), e.message),
        )
    except (SchemaError, ValueError, TypeError, LookupError, Unresolvable) as error:
        result["detail"] = f"Schema unavailable: {type(error).__name__}"
        return result
    result["status"] = "invalid" if errors else "valid"
    result["errors"] = [
        {"path": list(error.absolute_path), "message": error.message, "rule": error.validator}
        for error in errors
    ]
    return result


def evidence_payload(
    payload: Mapping[str, Any],
    *,
    task_semantics: str,
    tool_contracts: Mapping[str, Any],
    prior_review: Mapping[str, Any] | None = None,
    include_reasoning: bool = True,
) -> dict[str, Any]:
    """Enrich a full payload without truncation or changes to committed state."""
    if task_semantics not in TASK_SEMANTICS:
        raise ValueError(f"unknown task semantics: {task_semantics}")
    result = deepcopy(dict(payload))
    for section, prefix in (("confirmed_history", "history"), ("proposed_trajectory", "candidate")):
        if section in result:
            result[section] = [deepcopy(step) for step in result[section]]
        for index, step in enumerate(result.get(section) or [], 1):
            step["evidence_ref"] = f"{prefix}:{index}"
            if not include_reasoning:
                step.pop("reasoning", None)
    result["task_semantics"] = task_semantics
    result["input_integrity"] = {
        "status": "complete",
        "profile": "evidence_v1",
        "omissions": [],
        "scope": "runtime-supplied evidence only; not a claim of task completion",
    }
    result["available_tool_names"] = sorted(tool_contracts)
    result["schema_checks"] = [
        schema_check(step, index, tool_contracts)
        for index, step in enumerate(result.get("proposed_trajectory") or [], 1)
    ]
    if prior_review is not None:
        result["prior_review"] = deepcopy(dict(prior_review))
    # Stable task constraints precede the growing execution history for prefix caching.
    # `input_integrity` sits after confirmed_history: callers may rewrite it per call, and ahead
    # of the history it would end the shared prefix before the largest cacheable block.
    order = (
        "task",
        "task_semantics",
        "anchors",
        "available_tool_names",
        "confirmed_history",
        "input_integrity",
        "proposed_trajectory",
        "invoked_tool_contracts",
        "schema_checks",
        "prior_review",
    )
    return {
        **{key: result[key] for key in order if key in result},
        **{key: value for key, value in result.items() if key not in order},
    }
