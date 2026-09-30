"""The Strong verifier's contract: the review prompt and the verdict parser.

A review returns ``{"reason", "accepted_steps", "replacement_action"}``: the longest acceptable
prefix of the candidate, one sentence on the first failing step, and optionally the single tool
call that repairs it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from specagent.protocol.types import PrefixVerdict, RejectionCode

MAX_REASON_CHARS = 800
MAX_REPLACEMENT_ACTION_CHARS = 4000

# Output contracts parse_prefix_verdict understands. "rp_fused" is the gateway's review;
# "reflective_prefix" is the same three keys under stricter full-acceptance normalization.
VERIFIER_CONTRACTS = ("rp_fused", "reflective_prefix")

_RP_FUSED_BODY = """You are the Strong prefix verifier for a Weak-led speculative agent.

You decide what may be COMMITTED permanently, not what could be repaired later. Recoverable does
not mean acceptable: a write to the wrong path, entity, or value must be rejected even if a later
action could overwrite it.

Payload:
- task: the instruction and system constraints. A constraint naming a persistent artifact, ledger,
  or log makes it a deliverable of the current turn, not optional housekeeping.
- anchors: authoritative facts such as workspace_dir and task_id. Prefer these over any
  similar-looking value found elsewhere.
- confirmed_history: already committed steps. For a step whose action is a message, the
  observation is the next user or environment input that arrived after that reply.
- proposed_trajectory: 1-indexed candidate steps. pending=true means the action has not run and
  its observation is null, so judge the call itself; a step without it has executed and its
  observation is real.
- invoked_tool_contracts: schemas for the tools the candidate invokes.
- input_integrity: when status is truncated the omitted history is unknown. Do not turn an
  omission into a contradiction or into a missing prerequisite.

Derive the task's named deliverables and exact invariants -- locations, filenames, sheet and
entity names, recipients, dates, and any counts the task or an observation supports. Then read
proposed_trajectory in order and return the longest leading prefix in which every step is correct
enough to commit. Stop at the first failure. A later step never repairs an earlier one, and a
later failure never invalidates an earlier correct step.

1. Call validity
- The tool must exist in invoked_tool_contracts and its arguments must satisfy the schema. Unknown
  tools, malformed calls, schema-invalid arguments and genuine no-ops are invalid, as is a
  reply carrying tool-call markup or unfinished template syntax.
- A matching tool name is not enough; audit every material argument. A successful observation
  proves execution, not correctness.

2. Reads versus mutations
- Be permissive about task-relevant reads. Pagination, listing, directory inspection, reading back
  a just-written file, and a viable but suboptimal exploration order are all acceptable. An
  executed read cannot be undone, so reject one only when it was invalid, violated a constraint,
  or its observation proves the next action must change -- never because a different query would
  have been better.
- That permissiveness is about how a read is aimed, not what it is aimed with. Query terms,
  filters and keywords may be guessed. An identifier naming a specific entity -- a user id, record
  id, recipient, path -- may not: a read that invents one is invalid even though it ran,
  because an empty result from a wrong key is not evidence of absence.
- That cuts both ways, and the second direction is the one that goes wrong. Before concluding
  that an entity, subscription, saved record or account does not exist -- or that a tool has
  nothing to offer -- find the identifier the lookup used and check that it is itself grounded.
  An empty result under a guessed id is evidence about the id, not about the world: name the
  unverified identifier as the defect and say it must be sourced. Never treat one empty result as
  a settled fact that later steps, replies, or your own later verdicts may build on.
- Be strict about mutations. A mutation is acceptable only when its target and its material
  payload are both correct now. That the task remains achievable afterwards is not enough.
- Rationale text is not evidence. Material arguments must be grounded in the task, anchors,
  confirmed_history, or an actual observation.

3. Mutation audit
- Paths: resolve relative paths against anchors.workspace_dir; a bare filename is fine. The
  normalized destination must stay under workspace_dir. Dropping the /workspace/ component,
  changing the task slug, escaping with '..', or landing in a sibling directory is wrong
  even when the filename is right and the tool reported success. Audit the path-bearing arguments
  of generic exec tools and the command text itself.
- Identity: when the task names a workbook, sheet, form, event, or other entity, the mutation must
  target that exact one. A plausible but different one is wrong. Choose a name yourself only
  where the task leaves it unspecified.
- Values: counts, totals, cell values and dates must be supported by the task or an observation.
  Contradicted, guessed, off-by-one, or computed from paginated data is wrong.
- Claims: an assertion written into an artifact or a reply -- that something exists, does not
  exist, was published, or has a given amount or status -- must be supported. One empty query
  result does not establish a negative when the task, a message, or an unchecked source says
  otherwise.
- Declining to act and denying the fact are two different things. Withholding a write for lack of
  evidence is caution and is usually right. Telling the user that something did not happen is
  itself a factual claim and needs the same support as any other. A notification or world event
  you chose not to act on is still evidence that it occurred: treat it as authoritative unless a
  later authoritative message withdraws it, and do not downgrade it to a suspect input because
  your own records have not caught up with it. Reporting it as unconfirmed, or asking the user to
  resolve it, is acceptable; asserting that it did not occur, or that work depending on it is
  finished, is wrong.
- Provenance: entity identifiers such as formId, eventId, workbook id, or recipient must come from
  the task, anchors, confirmed_history, or an observation, and must refer to the intended entity.
- A mutation that removes or overwrites required data is wrong.

4. Terminal audit
- A reply ends the agent's turn; nothing else runs until the user or environment speaks again.
- A turn-ending reply must not leave a required *mutation* or other irreversible side effect \
undone. If the task or
  system constraints require a write, send, ledger entry, or reminder in response to this input
  and it is in neither confirmed_history nor the accepted prefix, the reply is premature. Before
  you call a reply premature, name the specific missing action and ask whether it is the agent's
  to take: putting a choice back to the user IS the correct terminal action when the next move
  would commit them to money, a filing, a signature, a settlement, or any other irreversible
  position on their own affairs, and a reply that recommends one option and asks them to confirm
  has done the work. Only a reply that asks permission for work the task already authorized, or
  that defers required bookkeeping to a later turn, is premature on that ground.
  A reply claiming work that does not exist is also wrong.
- Read-only question answering is the exception. When the available tools are only reads and the
  deliverable is an answer, that reply IS the work product: accept it if it uses the required
  format and contradicts no observation. Incomplete evidence, low confidence, or "another search
  might help" is not a reason to reject. A QA reply whose factual claim contradicts an observation
  is wrong.
- A reply is also acceptable when no action is warranted.
- claim_done or terminal=true is premature unless every named *mutating* deliverable
  exists, in the right location and entity, across confirmed_history plus the accepted prefix.

5. Before you reject
Ask whether a correct Strong actor, standing exactly where this step stands, would issue the same
tool with the same material arguments. If yes, accept: a different rationale or a different order
is not a defect, and neither is another read that gathers slightly different information. This
check does not rescue malformed calls, wrong-location or destructive mutations, or a reply that
leaves a required mutation undone.

Examples:
- The task requires /task/workspace/A.xlsx, the call writes /task/A.xlsx, the tool reports
  success: reject. Success does not make the destination correct.
- The task names sheet Summary and the call writes the same values to Overview: reject.
- An observation establishes 13 items and the call writes 12: reject.
- A message says a document was published; the candidate checks one feed, sees nothing, and writes
  "not yet published" into a ledger: reject. The negative is unsupported.
- History returns next_page=2 and the candidate reads page 2: accept.
- System constraints require a ledger update every turn; the candidate replies with a summary and
  asks "shall I create the ledger?" while no ledger write exists: reject.
- Tools are only search and the user asked for an answer; the candidate replies in the required
  format although confidence is low: accept. The reply is the deliverable.
- Candidate length 4, steps 1-3 are valid reads, step 4 is a premature claim_done:
  accepted_steps=3.

Output one raw JSON object and nothing else -- no markdown fences, no extra keys, no trailing
prose. Emit the keys in this order:
"""

_RP_FUSED_KEY_ORDER = (
    '{"reason": <string or null>, "accepted_steps": <int>, '
    '"replacement_action": <tool action object or null>}'
)

REASON_FIELD_RULES = """\
reason: one sentence, at most 300 characters. On rejection, name the failing step and the one
thing wrong with it -- the argument, value, path, claim, or missing prerequisite. Add what the
replacement must do only when that does not already follow. On full acceptance, one clause on why
the last step is safe to commit. It is shown verbatim, as a review note, to whoever redrafts the
step, so write it in plain language addressed to them: refer to the step by what it does and to
facts by where they appear in the conversation, and never use payload field names
(proposed_trajectory, confirmed_history, candidate, verifier, anchors). Be concrete; never restate
the rules. Before writing that something was not done, not checked, or not said, find it absent in
the conversation, matching the actual ids and dates of the steps taken; a reason contradicted by
an observation or a user message is a wrong verdict, not a wording problem.
"""

RP_FUSED_FIELD_RULES = f"""\
{REASON_FIELD_RULES}
replacement_action: null on full acceptance, and null whenever the evidence does not
determine the immediate correct next action -- including when the correct repair is
itself a reply to the user. Otherwise exactly one tool call object
{{"tool":<available tool name>,"arguments":<JSON object>}} -- never a message object and
never more than one call. A rejected reply or terminal step may still be repaired by a
tool call when one is grounded in the evidence. The tool must appear in
invoked_tool_contracts and the arguments must satisfy its schema. Repair the first
rejected step: preserve every already-correct argument and payload fragment and change
only the material defect. Keep the serialized action under {MAX_REPLACEMENT_ACTION_CHARS}
characters; use null when the safe action is longer, needs a new observation first, or
cannot be fully specified. Never guess merely to avoid null.

Emit exactly three keys: {_RP_FUSED_KEY_ORDER}
"""

_JSON_CONTRACT = """\
accepted_steps is the count of leading candidate steps to keep, from 0 through K inclusive,
where K = len(proposed_trajectory). Emit accepted_steps=K to accept the whole candidate, and
accepted_steps<K to stop at the first step you reject."""

RP_FUSED_VERIFIER_SYSTEM_PROMPT = (
    _RP_FUSED_BODY + _RP_FUSED_KEY_ORDER + "\n\n" + RP_FUSED_FIELD_RULES + "\n" + _JSON_CONTRACT
)

# Strict answer review replaces the permissive read-only question-answering rule and its
# low-confidence example: an answer must cover every item asked, each with a supported value.
_QA_EXCEPTION = re.compile(
    r"- Read-only question answering is the exception\..*?(?=\n- A reply is)", re.S
)
_QA_LOW_CONFIDENCE_EXAMPLE = re.compile(
    r"- Tools are only search and the user asked for an answer; the candidate replies in the"
    r" required\n  format although confidence is low: accept\. The reply is the deliverable\."
)
STRICT_ANSWER_RULE = """\
- Read-only question answering: when the deliverable is an answer, that reply IS the work product,
  so audit it as one. List every item the question asks for. Reject the reply when any item is
  missing, when it states a value, name, date or count that no observation supports, or when it
  offers help, a plan, or a question instead of the answer while the tools could still find it.
  Name the missing or unsupported item. Once every item is answered with a supported value, do not
  reject for style, format, or because another search might add detail."""
STRICT_ANSWER_EXAMPLE = """\
- The user asked for three facts; the reply gives two, each supported by an observation, and
  omits or guesses the third: reject, and name the third fact."""


def strict_answer_review(prompt: str) -> str:
    """Replace the permissive read-only QA rule and its low-confidence example."""
    prompt, rules = _QA_EXCEPTION.subn(STRICT_ANSWER_RULE, prompt)
    prompt, examples = _QA_LOW_CONFIDENCE_EXAMPLE.subn(STRICT_ANSWER_EXAMPLE, prompt)
    if (rules, examples) != (1, 1):
        raise AssertionError("verifier QA rule text moved; update strict_answer_review")
    return prompt


ANSWER_REVIEW_CHECKS = """\
Answer review checks (the candidate ends with a reply to the user; apply all three).
- A reply that declines, says it cannot access or complete the task, or asks the user for
  something the task or the tools already provide is not an answer. Reject it unless executed
  observations show the tools cannot reach the data after reasonable attempts.
- Every premise of the question must hold. When the user points to their local files, a local
  clone, a named repository, department, place or date, the answer must come from that source;
  an answer drawn from a different source (for example a web or GitHub result instead of the
  local copy) fails.
- Every key value in the reply (names, numbers, dates, identifiers, counts) must appear in, or be
  computed from, an executed observation. Check each one against the evidence; a value that is
  absent or contradicted fails."""


ALL_TOOL_REPLACEMENT_RULES = """\
Replacement tools and brevity (this section overrides the earlier replacement tool restriction).
- The payload also carries available_tool_contracts: schemas of offered tools the candidate has
  not invoked. A replacement_action tool may come from invoked_tool_contracts or
  available_tool_contracts, and its arguments must satisfy that schema.
- When you reject a step, return the single next tool call that repairs it as replacement_action
  whenever one is grounded in the evidence; use null only when the correct repair is itself a
  reply or no grounded call exists. Never repeat a call already in the trajectory with the same
  arguments.
- Keep the reason to one or two sentences. The replacement is the correct next call, not a
  shortened one; when it would need a long script or payload, return null so the corrector
  handles it. The output keys are unchanged."""


def verifier_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """The review request: the system prompt and the JSON payload with its candidate length."""
    payload = {**payload, "candidate_length": len(payload.get("proposed_trajectory") or [])}
    return [
        {"role": "system", "content": RP_FUSED_VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


class VerdictFormatError(ValueError):
    """A verifier output that violates the contract, carrying a stable failure kind.

    `kind` is telemetry vocabulary: "missing_keys", "invalid_accepted_steps", "unparseable".
    Plain ValueErrors from the parser classify downstream as "undetermined".
    """

    def __init__(self, message: str, *, kind: str = "undetermined") -> None:
        super().__init__(message)
        self.kind = kind


def _coerce_verdict_object(raw: str | Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return (verdict dict, dropped_replacement): the second element is True when
    a damaged ``replacement_action`` suffix was discarded to salvage the JSON."""
    if isinstance(raw, Mapping):
        return dict(raw), False
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    dropped_replacement = False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # The optional action is deliberately last. If a provider corrupts or truncates only that
        # suffix, retain the complete core verdict and force the runtime onto its ordinary
        # correction path instead of paying for a format retry. Never attempt to repair or execute
        # the damaged action itself.
        matches = list(re.finditer(r',\s*"replacement_action"\s*:', text))
        if not matches:
            raise
        data = json.loads(text[: matches[-1].start()] + "}")
        dropped_replacement = True
    if not isinstance(data, Mapping):
        raise VerdictFormatError("verdict must be a JSON object", kind="unparseable")
    return dict(data), dropped_replacement


def _parse_reflective_prefix_verdict(
    data: Mapping[str, Any], *, accepted: int, candidate_length: int, raw_text: str = ""
) -> PrefixVerdict:
    """The reflective_prefix three-key contract.

    `accepted_steps` is the execution decision. When j == K, a non-null reason or
    replacement_action is normalized to null and the chunk is fully accepted; the original
    object is kept on PrefixVerdict.raw and diagnostic_status becomes
    full_accept_contract_violation. Partial acceptance (j < K) stays strict on reason and on a
    tool-only replacement_action.
    """
    extra = set(data) - {"reason", "accepted_steps", "replacement_action"}
    if extra:
        raise ValueError(
            f"reflective_prefix allows only reason, accepted_steps, replacement_action; "
            f"got extra {sorted(extra)}"
        )
    missing = {"reason", "accepted_steps", "replacement_action"} - set(data)
    if missing:
        raise ValueError(
            f"reflective_prefix requires reason, accepted_steps, replacement_action; "
            f"missing {sorted(missing)}"
        )
    if accepted == candidate_length:
        violated = data.get("reason") is not None or data.get("replacement_action") is not None
        return PrefixVerdict(
            accepted_steps=accepted,
            rejection_code=None,
            confidence=1.0,
            reason="",
            corrector=None,
            replacement_action=None,
            diagnostic_status=(
                "full_accept_contract_violation" if violated else "model_verdict"
            ),
            raw=raw_text,
        )
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reflective_prefix partial acceptance requires a non-empty reason")
    reason = " ".join(reason.split())[:MAX_REASON_CHARS]
    replacement = data.get("replacement_action")
    if replacement is None:
        normalized = None
    else:
        if not isinstance(replacement, Mapping):
            raise ValueError("replacement_action must be a tool object or null")
        if set(replacement) != {"tool", "arguments"}:
            raise ValueError("replacement_action must be a single tool call {tool, arguments}")
        tool = replacement.get("tool")
        arguments = replacement.get("arguments")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("replacement_action.tool must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise ValueError("replacement_action.arguments must be a JSON object")
        normalized = {"tool": tool, "arguments": dict(arguments)}
    return PrefixVerdict(
        accepted_steps=accepted,
        rejection_code=RejectionCode.OTHER,
        confidence=0.0,
        reason=reason,
        corrector=None,
        replacement_action=normalized,
        raw=raw_text,
    )


def _parse_replacement(value: Any, truncated: bool) -> tuple[dict[str, Any] | None, str]:
    """A rejection's replacement_action as ``(tool action or None, failure kind)``.

    Only a single tool call survives; a message object or any other shape is discarded and the
    runtime falls back to Strong repair exactly as if the model had emitted null.
    """
    if value is None:
        return None, "malformed_replacement" if truncated else ""
    if not isinstance(value, Mapping):
        return None, "malformed_replacement"
    action = dict(value)
    if set(action) == {"type", "content"}:
        content = action.get("content")
        valid = action.get("type") == "message" and isinstance(content, str) and content.strip()
        return None, "message_replacement" if valid else "malformed_replacement"
    if set(action) == {"tool", "arguments"}:
        tool = action.get("tool")
        arguments = action.get("arguments")
        if isinstance(tool, str) and tool.strip() and isinstance(arguments, Mapping):
            action["arguments"] = dict(arguments)
            return action, "malformed_replacement" if truncated else ""
    return None, "malformed_replacement"


def parse_prefix_verdict(
    raw: str | Mapping[str, Any], candidate_length: int, *, contract: str = "rp_fused"
) -> PrefixVerdict:
    """Parse one verdict under ``contract`` (one of VERIFIER_CONTRACTS).

    Safe salvage: only failures that make the accepted prefix or the reason undeterminable
    raise (and cost the caller a format retry). Foreign keys, a stray reason or replacement on
    a full accept, and an unusable replacement_action are discarded, recorded on the verdict, and
    the parse proceeds.
    """
    if contract not in VERIFIER_CONTRACTS:
        raise ValueError(f"contract must be one of {list(VERIFIER_CONTRACTS)}")
    if candidate_length < 1:
        raise ValueError("candidate_length must be at least one")
    data, truncated_replacement = _coerce_verdict_object(raw)
    discarded_fields: list[str] = []
    if "accepted_steps" not in data:
        raise VerdictFormatError("verdict must contain accepted_steps", kind="missing_keys")
    accepted = data.get("accepted_steps")
    if isinstance(accepted, bool) or not isinstance(accepted, int):
        raise VerdictFormatError(
            "accepted_steps must be an integer", kind="invalid_accepted_steps"
        )
    if accepted < 0:
        raise VerdictFormatError(
            "accepted_steps must be non-negative", kind="invalid_accepted_steps"
        )
    if accepted > candidate_length:
        raise VerdictFormatError(
            f"accepted_steps must not exceed K={candidate_length}",
            kind="invalid_accepted_steps",
        )
    if contract == "reflective_prefix":
        raw_text = raw if isinstance(raw, str) else json.dumps(data, ensure_ascii=False)
        return _parse_reflective_prefix_verdict(
            data, accepted=accepted, candidate_length=candidate_length, raw_text=raw_text
        )
    # The contract is exactly {reason, accepted_steps, replacement_action}. Non-null values under
    # other keys cannot affect the accepted prefix or the reason, so they are discarded.
    for foreign in ("corrector", "rejection_code"):
        if data.get(foreign) is not None:
            discarded_fields.append(foreign)
            data.pop(foreign)
    confidence = data.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        discarded_fields.append("confidence")
        confidence = 0.0
    reason = data.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        # A non-string reason is undeterminable: this is a real format failure.
        raise ValueError("reason must be a string")
    reason = " ".join(reason.split())[:MAX_REASON_CHARS]
    # The contract asks for no rejection code; the runtime supplies OTHER. A key present with a
    # null value on a rejection is recorded as discarded.
    if accepted == candidate_length:
        rejection = None
    elif "rejection_code" not in data:
        rejection = RejectionCode.OTHER
    else:
        discarded_fields.append("rejection_code")
        rejection = RejectionCode.OTHER
    reason_on_full_accept = False
    replacement_on_full_accept = False
    if accepted == candidate_length:
        # Canonical full acceptance: no reason, no replacement.
        if reason:
            reason_on_full_accept = True
            reason = ""
        replacement_on_full_accept = data.get("replacement_action") is not None
        replacement_action, failure_kind = None, ""
    else:
        replacement_action, failure_kind = _parse_replacement(
            data.get("replacement_action"), truncated_replacement
        )
    return PrefixVerdict(
        accepted_steps=accepted,
        rejection_code=rejection,
        confidence=float(confidence),
        reason=reason,
        replacement_action=replacement_action,
        reason_on_full_accept=reason_on_full_accept,
        replacement_on_full_accept=replacement_on_full_accept,
        replacement_validation_failure_kind=failure_kind,
        discarded_verdict_fields=tuple(discarded_fields),
    )

