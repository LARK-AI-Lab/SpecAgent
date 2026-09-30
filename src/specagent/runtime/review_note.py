"""The review note handed to whoever redrafts a rejected step.

The structure is fixed here -- which draft was withheld, why, how much was
dropped with it, and that exactly one action is expected. Only the nouns differ
per setting (who the assistant is talking to, what kinds of argument a fix
typically changes), and those come from a ReviewNoteVocabulary.

Deliberately neutral throughout: the note reads as a pre-execution review of
the assistant's own draft, not as a hand-off between two models. The model
simply continues the conversation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Keep long observations and reasoning out of the note; the transcript the
# corrector already carries has them in full.
OBSERVATION_LIMIT = 1200


@dataclass(frozen=True)
class ReviewNoteVocabulary:
    """The benchmark-specific nouns in an otherwise fixed note."""

    # Who the assistant is talking to, as the assistant would refer to them.
    counterpart: str
    # How to describe the non-tool-call option in the one-action instruction.
    reply_clause: str
    # Example arguments a corrected version typically changes.
    fix_nouns: str
    # What a good next action makes progress towards.
    advance_clause: str
    # The counterpart, possessive, for the rule that cites what they said.
    counterpart_possessive: str = ""
    # Rules that only make sense for one runtime.
    extra_rules: tuple[str, ...] = field(default_factory=tuple)


GATEWAY_VOCABULARY = ReviewNoteVocabulary(
    counterpart="user",
    counterpart_possessive="the user's",
    reply_clause="a plain reply with no tool call when replying is the right next step",
    fix_nouns="path, target, value, content, recipient, or date",
    advance_clause="an action that advances the deliverables",
)

def review_note(
    *,
    code: str,
    reason: str,
    rejected_step: Mapping[str, Any] | None = None,
    discarded_after: int = 0,
    earlier_rejections: int = 0,
    vocabulary: ReviewNoteVocabulary = GATEWAY_VOCABULARY,
) -> str:
    """Note appended to the transcript before the corrector writes its next turn.

    The only extra information it carries is which draft was withheld, why, and
    that exactly one step is expected. Generic by design: it never mentions the
    task, and it does not restate domain rules such as how many tool calls a
    step may contain.
    """
    who = vocabulary.counterpart
    lines = [
        "Runtime note before your next step (inserted by the tool runtime, not written by "
        f"the {who}; the {who} does not see it).",
        "",
    ]
    if rejected_step is not None:
        step = dict(rejected_step)
        step.pop("index", None)
        step.pop("actor", None)
        executed = not step.get("pending")
        observation = step.get("observation")
        if isinstance(observation, str) and len(observation) > OBSERVATION_LIMIT:
            step["observation"] = observation[:OBSERVATION_LIMIT] + "...[truncated]"
        # The draft's own reasoning is dropped, not truncated. The note asks the
        # corrector not to reissue the draft; showing it the thinking that
        # produced the draft argues the other way.
        step.pop("reasoning", None)
        if not executed:
            step.pop("observation", None)
            step.pop("pending", None)
            lines.append(
                "Your draft for the next step was reviewed before execution and was not run:"
            )
        else:
            lines.append(
                "Your previous step below did run (its result is above), but on review it was "
                "judged to be on the wrong path:"
            )
        lines.append(json.dumps(step, ensure_ascii=False))
    else:
        lines.append("Your draft for the next step was withheld on review.")
    # `other` carries nothing: it is the sentinel the runtime supplies for a synthesized verdict,
    # or for a contract that does not ask the verifier to classify at all. Naming it in the note
    # spends the corrector's attention on a label with no content behind it.
    finding = "Review finding" if code in ("other", "unspecified") else f"Review finding ({code})"
    lines.append(f"{finding}: {reason.strip() or 'no details given'}")
    if discarded_after > 0:
        lines.append(f"{discarded_after} further draft step(s) that depended on it were dropped.")
    if earlier_rejections > 0:
        lines.append(
            f"{earlier_rejections} earlier draft(s) for this same step were also withheld on "
            "review; do not return to them."
        )
    lines.extend(
        [
            "",
            # "one step", not "one tool call": how many calls a step may carry is
            # the domain policy's call, and it is already stated in the system
            # prompt at the top of this same conversation.
            "Continue from here with exactly ONE step: a tool call, or "
            f"{vocabulary.reply_clause}.",
            "- Fix the draft rather than starting over. If its intent was right, issue the "
            f"corrected version ({vocabulary.fix_nouns}). If it skipped "
            "a prerequisite, do that prerequisite now.",
            "- If the draft was a reply and the finding says required work or records are "
            "missing, do that work with a tool call before replying.",
            f"- Prefer {vocabulary.advance_clause}. Gather more information only "
            "when the finding says it is missing and the conversation above does not already "
            "contain it.",
            # The reason is one sentence from a verifier that saw a payload, not the live
            # transcript; it can be wrong about what already happened. The conversation also
            # carries the drafter's committed reasoning, which nothing has reviewed, so tool
            # results and the counterpart's own words are named as the evidence instead.
            "- Check the finding against the conversation. Where a tool result or "
            f"{vocabulary.counterpart_possessive} own words contradict it, trust those; the "
            "reasoning above is not evidence and was never reviewed.",
            "- Do not repeat the draft or a materially equivalent action, and do not claim work "
            "that has not been done.",
            "- Act directly; do not explain first.",
            *vocabulary.extra_rules,
        ]
    )
    return "\n".join(lines)

