"""Model-based prefix reuse: how many input tokens of each request a radix cache can reuse.

Serving caches report reuse only partially and are affected by eviction, so the paper assigns
each request's cached prefix with a model of radix prefix caching, applied identically to every
method (appendix, "Request Shape and Prefix Reuse"). Requests are grouped into sessions that
share task, trial, model, role and serving replica; reuse never crosses sessions. Within a
session, requests are processed in order and newly processed tokens are

  * nested (Weak draft, Strong act):  max(0, I - M)
  * verify (Strong verification):     c + max(0, I - c - B_v)
  * repair (Strong repair):           n + max(0, I - n - B_c)

where I is the request's input tokens, M, B_v and B_c are running maxima of the nested input,
the verify backbone (I - c) and the repair backbone (I - n), c is the candidate length (the
same round's Weak output, or a default when unknown) and n the rejection-note length (300
tokens, since note text is not logged). Verification and repair backbones are kept separate.

The cached count from here goes into an InferenceSpan's cached_input_tokens before
specagent.protocol.flops / hbm estimate the request.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

Family = Literal["nested", "verify", "repair"]

# Assumed rejection-note length for repair prompts, and the candidate length used when no
# verify request in the batch has a known one.
NOTE_TOKENS = 300
CANDIDATE_TOKENS_FALLBACK = 1500


@dataclass(frozen=True)
class ReuseRequest:
    """One successful generation request, as the reuse model needs it.

    ``session`` must already separate task, trial, model, role and serving replica. ``order``
    sorts requests within a session (for example (round, turn, index)). ``candidate_tokens`` is
    the same-round Weak output for a verify request and is ignored for other families.
    """

    session: Hashable
    order: tuple
    family: Family
    input_tokens: int
    candidate_tokens: int | None = None


def default_candidate_tokens(requests: Sequence[ReuseRequest]) -> int:
    """Median known candidate length over the verify requests (the method's median)."""
    known = [
        r.candidate_tokens
        for r in requests
        if r.family == "verify" and r.candidate_tokens is not None
    ]
    if not known:
        return CANDIDATE_TOKENS_FALLBACK
    return int(round(statistics.median(known)))


def cached_input_tokens(
    requests: Sequence[ReuseRequest],
    *,
    candidate_default: int | None = None,
    note_tokens: int = NOTE_TOKENS,
) -> list[int]:
    """Cached prefix tokens for each request, in the order given.

    ``candidate_default`` replaces an unknown candidate length; it defaults to
    default_candidate_tokens(requests).
    """
    if candidate_default is None:
        candidate_default = default_candidate_tokens(requests)
    by_session: dict[Hashable, list[int]] = defaultdict(list)
    for index, request in enumerate(requests):
        by_session[request.session].append(index)

    cached = [0] * len(requests)
    for indices in by_session.values():
        nested_max = verify_backbone = repair_backbone = 0
        for index in sorted(indices, key=lambda i: (requests[i].order, i)):
            request = requests[index]
            total = int(request.input_tokens)
            if request.family == "verify":
                fresh = request.candidate_tokens
                fresh = candidate_default if fresh is None else fresh
                fresh = max(0, min(total, int(fresh)))
                backbone = total - fresh
                new = fresh + max(0, backbone - verify_backbone)
                verify_backbone = max(verify_backbone, backbone)
            elif request.family == "repair":
                fresh = max(0, min(total, int(note_tokens)))
                backbone = total - fresh
                new = fresh + max(0, backbone - repair_backbone)
                repair_backbone = max(repair_backbone, backbone)
            else:
                new = max(0, total - nested_max)
                nested_max = max(nested_max, total)
            cached[index] = max(0, total - new)
    return cached
