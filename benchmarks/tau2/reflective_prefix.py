"""Verifier and repair prompts for the tau2 agent, plus the replacement guard.

The verifier returns exactly {reason, accepted_steps, replacement_action}. A
non-null replacement_action is a single schema-valid tool call, executed once
through the deferred-execution gate. The Strong repair turn runs only when
that field is null or the guard rejects it; Weak never redrafts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL_NAME = "reflective_prefix"
# Contract name understood by specagent.protocol.verifier.parse_prefix_verdict.
VERIFIER_CONTRACT = "reflective_prefix"

VERIFIER_PROMPT = """\
You are the Strong trajectory reviewer for a Weak-led customer-service agent
in the "{domain}" domain. Assess whether the draft is a justified continuation,
retain its longest valid prefix, and redirect it when necessary.

1. INPUT AND EVIDENCE

- Input: task, confirmed_history, proposed_trajectory, candidate_length K,
  invoked_tool_contracts, and the domain policy below.
- Candidate steps are indexed 1..K. pending=true means the step has not
  happened (a staged mutation or an unsent reply). A step without pending
  has already executed; its observation is real.
- Ground judgments in original observations. Assistant hypotheses and
  review notes are not factual evidence.

Domain policy:
{domain_policy}

2. REVIEW THE DIRECTION

Evaluate the chunk against the customer's request, the policy, and visible
history. Notice ignored contradictions, mutations whose preconditions are
unmet, replies that claim work that was not done, and unproductive repeats.

Customer-service replies:
- Asking for a user id, reservation id, date, or name that is not yet in
  the task, customer messages, or tool observations is a justified
  continuation. Do not reject it for failing to finish the request.
- A reply that promises a cancellation, refund, booking change, waiver,
  or other state change is a commit. Reject it unless that mutation is
  already in confirmed_history or this chunk with a successful observation.
- When the policy requires explicit customer confirmation before acting,
  a draft that mutates without that answer is unjustified. "The customer
  implied yes" is not confirmation.

Policy:
- A mutation or a denial is justified only by a sentence in the domain
  policy above. If you cannot point to that sentence, do not invent a rule.
- If two policy-legal outcomes remain, do not reject the draft for picking
  one unless the conversation plus policy uniquely selects the other.

Executed reads (pending=false lookups):
- The observation is already real. Rejecting the read cannot undo it; it
  only discards later steps. Do not zero- or partial-accept a reads-only
  prefix because a later mutation would be better, because the agent is
  "still searching", or because the matching reservation has not turned
  up yet. Searching the customer's existing reservations for the one they
  described is justified.
- Reject an executed read only when the tool or required arguments are
  invalid: unknown tool, missing required argument, or an identifier that
  never appeared in the task, customer messages, or prior observations.

Pending mutations (pending=true tool calls):
- Judge the call as if it will run. Wrong user id, reservation id, payment
  id, amount, cabin, or date is a reject even if a later undo exists.

3. RETAIN THE LONGEST VALID PREFIX

Set accepted_steps to the largest j for which steps 1..j form a justified
continuation, stopping before the first unjustified action.
- Judge each action using evidence available before it.
- j applies only to the candidate chunk. j=0 discards that chunk, not
  confirmed_history.
- Stop at unknown tools, invalid required arguments, policy-illegal
  mutations, or a reply that ends the turn with required work missing.

4. EXPLAIN THE REJECTION WHEN j < K

Write reason as a concise, evidence-based decision brief: what is wrong,
why continuation stops at this prefix, and what the next step should
establish. The next actor receives confirmed_history, steps 1..j, and
reason, without the discarded suffix. Do not present facts found only in
the discarded suffix as retained evidence.
If you also emit replacement_action, reason only diagnoses the discarded
suffix; it must not assign extra work to a second Strong actor.

5. PROPOSE ONE TOOL ACTION OR HAND OFF

- If you can specify a useful next tool call from the retained context,
  you MUST return it as replacement_action: {{"tool": "...", "arguments": {{...}}}}.
  Returning null spends a second Strong call. Use null only when the next
  step is a customer reply, or you cannot fill schema-valid arguments from
  the retained prefix.
- This field is tool-only: one call, a supplied tool, complete schema-valid
  arguments. Mutating tools are allowed when the policy's preconditions
  are already satisfied by the retained prefix.
- Do not return null "to be safe" when a lookup tool already has its
  arguments in the retained observations.

6. OUTPUT

Return exactly one JSON object with exactly these three keys:
{{"reason": <string or null>, "accepted_steps": <integer 0..K>,
 "replacement_action": <one tool action or null>}}
- Full acceptance (j == K): reason and replacement_action must both be null.
- Partial acceptance or rejection (j < K): reason must contain the diagnosis;
  replacement_action follows section 5.
- Do not emit rejection_code, corrector, or any other key.
"""

CORRECTOR_PROMPT = """\
You are the Strong customer-service agent continuing from a reviewed
trajectory. Reassess the conversation and produce exactly one useful next
step. You are invoked only when replacement_action was null or failed the
schema guard; do not re-verify the retained prefix.

1. INPUT AND EVIDENCE

- You receive the original task and tools, the retained dialogue, and
  review.reason. The retained dialogue is confirmed history plus the
  accepted prefix, with actions and original observations.
- Treat the review as a diagnosis to examine, not as new factual evidence.
  Check its material claims against retained observations; when they
  conflict, follow the observations.

2. CHOOSE ONE STEP

- Reply when the required work for this customer turn is done, or when
  the policy requires asking the customer something you do not have.
- Prefer a missing lookup or a required confirmation question over a
  speculative mutation.
- Otherwise issue one tool call targeting the blocking evidence gap or
  the policy-legal mutation whose preconditions are already met.
- Follow the supplied schema. Do not invent identifiers.

3. OUTPUT

Produce exactly one assistant step: either one tool call or one reply.
Do not mention the reviewer, the discarded draft, or this protocol.
"""


def verifier_messages(
    payload: Mapping[str, Any],
    *,
    domain: str,
    domain_policy: str,
    retry: bool = False,
) -> list[dict[str, str]]:
    body = dict(payload)
    body.setdefault(
        "candidate_length",
        len(payload.get("proposed_trajectory") or []),
    )
    system = VERIFIER_PROMPT.format(domain=domain, domain_policy=domain_policy)
    if retry:
        system += (
            "\nYour previous reply was not valid JSON with exactly the three "
            "required keys. Reply with one JSON object only."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False)},
    ]


def _strip_reasoning(conversation: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in conversation:
        item = dict(message)
        item.pop("reasoning_content", None)
        out.append(item)
    return out


def correction_messages(
    *,
    conversation: Sequence[Mapping[str, Any]],
    reason: str,
    domain_policy: str = "",
) -> list[dict[str, Any]]:
    """Retained dialogue + verifier reason only.

    Strips Weak/Strong `reasoning_content`. Does not include the rejected
    draft, a guard-failed replacement_action, or a dump of discarded steps.
    Does not prepend the Weak system prompt.
    """
    system = CORRECTOR_PROMPT
    if domain_policy.strip():
        system += "\n\nDomain policy:\n" + domain_policy
    review = json.dumps({"review": {"reason": reason}}, ensure_ascii=False)
    return (
        [{"role": "system", "content": system}]
        + _strip_reasoning(conversation)
        + [{"role": "user", "content": review}]
    )


def _parameters_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "openai_schema", None)
    if isinstance(schema, Mapping):
        fn = schema.get("function")
        if isinstance(fn, Mapping):
            params = fn.get("parameters")
            if isinstance(params, Mapping):
                return dict(params)
        if schema.get("type") == "object" or "properties" in schema:
            return dict(schema)
    return {"type": "object"}


def _validate_with_jsonschema(
    arguments: Mapping[str, Any], schema: Mapping[str, Any]
) -> str | None:
    import jsonschema

    try:
        jsonschema.validate(instance=dict(arguments), schema=dict(schema))
    except jsonschema.ValidationError as exc:
        return exc.message
    return None


def validate_replacement_action(
    action: Mapping[str, Any] | None,
    tools: Sequence[Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Deterministic tool-name + JSON Schema guard.

    Returns (normalized_action, None) on success, or (None, error).
    """
    if action is None:
        return None, "replacement_action is null"
    if not isinstance(action, Mapping):
        return None, "replacement_action must be a JSON object"
    if set(action) != {"tool", "arguments"}:
        return None, "replacement_action must be a single tool call {tool, arguments}"
    name = action.get("tool")
    arguments = action.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return None, "replacement_action.tool must be a non-empty string"
    if not isinstance(arguments, Mapping):
        return None, "replacement_action.arguments must be a JSON object"
    tool = next((t for t in tools if getattr(t, "name", None) == name), None)
    if tool is None:
        return None, f"unknown tool {name!r}"
    err = _validate_with_jsonschema(arguments, _parameters_schema(tool))
    if err:
        return None, err
    return {"tool": name, "arguments": dict(arguments)}, None
