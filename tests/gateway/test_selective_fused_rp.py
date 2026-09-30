"""Focused tests for the gateway's action-step speculation protocol.

Speculation is action-step level, not turn level: one assistant turn's tool calls expand into
per-action candidate steps, reads are released without review, the first irrevocable step (a
state-changing call or a reply) triggers one Strong review under the fused three-key contract,
and one Strong repair turn may carry several tool calls.
"""

from __future__ import annotations

import json
from collections import deque

import pytest

from specagent.protocol.evidence import VerifierInputError
from specagent.protocol.types import (
    AgentStep,
    ModelRole,
    RejectionCode,
    RuntimeErrorCode,
)
from specagent.protocol.verifier import (
    ALL_TOOL_REPLACEMENT_RULES,
    ANSWER_REVIEW_CHECKS,
    STRICT_ANSWER_RULE,
    VerdictFormatError,
    parse_prefix_verdict,
)
from specagent.runtime.errors import SpecAgentRuntimeError
from specagent.runtime.gateway import (
    WEAK_BUDGET_NOTE,
    GatewayConfig,
    SpeculativeToolGateway,
    _context_overflow,
)


def completion(*calls: tuple[str, str, dict], content: str | None = None) -> dict:
    tool_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        for call_id, name, arguments in calls
    ]
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "hidden-reason",
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def rp_verdict(accepted: int, replacement: dict | None = None, **extra) -> dict:
    return completion(
        content=json.dumps(
            {
                "reason": "verifier reason",
                "accepted_steps": accepted,
                "replacement_action": replacement,
                **extra,
            }
        )
    )


class FakeBackend:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def complete(
        self,
        request,
        *,
        model,
        reasoning_effort=None,
        max_tokens=None,
        extra_body=None,
        is_cancelled=None,
    ):
        self.calls.append({"request": request, "model": model})
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def request(*, messages=None) -> dict:
    return {
        "model": "specagent",
        "messages": messages or [{"role": "user", "content": "Do the task"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "write",
                    "parameters": {"type": "object"},
                },
            },
        ],
    }


def profile(**overrides) -> GatewayConfig:
    return GatewayConfig(weak_model="weak", strong_model="strong", **overrides)


def tool_calls_of(response: dict) -> list[dict]:
    return response["choices"][0]["message"].get("tool_calls") or []


# A drafted write: the first irrevocable step, so it always ends the candidate with a review.
DRAFT_WRITE = ("w0", "write_file", {"path": "x"})


# --- candidate expansion -----------------------------------------------------


def test_three_call_turn_expands_to_three_candidate_steps() -> None:
    backend = FakeBackend(
        [
            completion(
                ("a", "read_file", {"path": "x"}),
                ("b", "read_file", {"path": "y"}),
                ("c", "read_file", {"path": "z"}),
            )
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=6))

    decision = gateway.handle(request())

    assert decision.source == "weak_read"
    assert decision.candidate_length == 3
    assert [
        (step.metadata or {}).get("tool_call_id") for step in gateway.state.candidate
    ] == ["a", "b", "c"]
    assert [c["id"] for c in tool_calls_of(decision.response)] == ["a", "b", "c"]


def test_partial_acceptance_releases_only_accepted_prefix() -> None:
    backend = FakeBackend(
        [
            completion(
                ("a", "read_file", {"path": "x"}),
                ("b", "read_file", {"path": "y"}),
                ("c", "write_file", {"path": "z"}),
            ),
            rp_verdict(1),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))

    decision = gateway.handle(request())

    assert decision.source == "weak_partial_then_correction"
    assert [c["id"] for c in tool_calls_of(decision.response)] == ["a"]
    assert gateway.state.deferred_correction
    # The rejected suffix was never delivered: b and c are gone from state.
    remaining = [
        (step.metadata or {}).get("tool_call_id")
        for step in (*gateway.state.confirmed, *gateway.state.candidate)
    ]
    assert remaining == ["a"]


def test_candidate_length_exceeds_horizon_is_validated_by_expanded_length() -> None:
    backend = FakeBackend(
        [
            completion(
                ("a", "read_file", {"path": "x"}),
                ("b", "read_file", {"path": "y"}),
                ("c", "write_file", {"path": "z"}),
            ),
            rp_verdict(3),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=2))

    decision = gateway.handle(request())

    # candidate_length=3 > horizon=2, and accepted_steps=3 is legal because the bound is the
    # expanded length, not K.
    assert decision.source == "weak_verified"
    assert decision.candidate_length == 3
    assert len(gateway.state.confirmed) == 3
    payload = json.loads(backend.calls[1]["request"]["messages"][1]["content"])
    assert payload["candidate_length"] == 3


def test_accepted_beyond_candidate_retries_once_then_fails_closed() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(2),
            rp_verdict(2),  # still invalid on the single retry
            completion(content="Final answer"),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.verdict is not None
    assert decision.verdict.diagnostic_status == "format_failure"
    assert decision.verdict.format_failure_kind == "invalid_accepted_steps"
    assert decision.verdict.format_retries == 1
    assert decision.source == "strong_correction"


def test_format_failure_kind_classifies_parse_failures() -> None:
    def kind(raw):
        with pytest.raises(VerdictFormatError) as raised:
            parse_prefix_verdict(raw, 3)
        return raised.value.kind

    # Only failures that leave the accepted prefix or reason undeterminable cost a format retry;
    # everything else is salvaged and recorded on the verdict.
    assert kind({"accepted_steps": 4, "reason": "r", "replacement_action": None}) == (
        "invalid_accepted_steps"
    )
    assert kind({"reason": "r", "replacement_action": None}) == "missing_keys"
    assert kind("[1, 2, 3]") == "unparseable"


# --- corrector multi-call ----------------------------------------------------


def test_corrector_two_reads_both_execute_once_with_observations() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "read_file", {"path": "a"}),
                ("f2", "read_file", {"path": "b"}),
            ),
            completion(content="next draft"),
            rp_verdict(1),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    delivered = tool_calls_of(decision.response)
    assert [c["id"] for c in delivered] == ["f1", "f2"]
    assert len(decision.correction_actions) == 2
    assert decision.correction_action == decision.correction_actions[0]

    # Both calls bind their own observation on the next request.
    gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "f1", "content": "obs-a"},
                {"role": "tool", "tool_call_id": "f2", "content": "obs-b"},
            ]
        )
    )
    observations = [
        step.observation
        for step in gateway.state.confirmed
        if step.actor is ModelRole.STRONG
    ]
    assert observations == ["obs-a", "obs-b"]


def test_corrector_mixed_read_mutation_batch_preserves_order() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "read_file", {"path": "a"}),
                ("f2", "write_file", {"path": "out", "data": "v"}),
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    delivered = tool_calls_of(decision.response)
    assert [(c["id"], c["function"]["name"]) for c in delivered] == [
        ("f1", "read_file"),
        ("f2", "write_file"),
    ]
    assert [action.get("tool") for action in decision.correction_actions] == [
        "read_file",
        "write_file",
    ]


def test_corrector_mutation_replay_fails_closed() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            # Identical to the mutation that already ran below.
            completion(("f1", "write_file", {"path": "out", "data": "v"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))
    gateway.state.confirmed = [
        AgentStep(
            ModelRole.WEAK,
            {"tool": "write_file", "arguments": {"path": "out", "data": "v"}},
            "written",
            metadata={"pending": False, "tool_call_id": "old"},
        )
    ]

    with pytest.raises(SpecAgentRuntimeError) as raised:
        gateway.handle(request())

    assert raised.value.code == RuntimeErrorCode.REPLAY_DIVERGENCE


def test_corrector_call_without_observation_fails_closed() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "read_file", {"path": "a"}),
                ("f2", "read_file", {"path": "b"}),
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))
    gateway.handle(request())

    with pytest.raises(SpecAgentRuntimeError) as raised:
        gateway.handle(
            request(
                messages=[
                    {"role": "user", "content": "Do the task"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "f1", "type": "function"},
                            {"id": "f2", "type": "function"},
                        ],
                    },
                    {"role": "tool", "tool_call_id": "f1", "content": "obs-a"},
                    # f2's tool result never arrived, but the client still carries the call:
                    # fail closed.
                ]
            )
        )

    assert raised.value.code == RuntimeErrorCode.MISSING_OBSERVATION


def test_corrector_reply_committed_once_without_reverify() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(content="The answer is 42."),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert decision.response["choices"][0]["message"]["content"] == "The answer is 42."
    assert [call["model"] for call in backend.calls] == ["weak", "strong", "strong"]
    assert decision.correction_actions[0]["type"] == "message"


# --- fused replacement --------------------------------------------------------


def test_rejected_terminal_with_grounded_tool_replacement_fuses() -> None:
    # A rejected reply is not exempt from fusion: when the verifier can name the grounded next
    # tool call, it commits directly instead of paying for a repair turn.
    backend = FakeBackend(
        [
            completion(content="All done, nothing left."),
            rp_verdict(
                0, replacement={"tool": "read_file", "arguments": {"path": "evidence.txt"}}
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))

    decision = gateway.handle(request())

    assert decision.source == "strong_fused_correction"
    assert decision.boundary == "message"
    delivered = tool_calls_of(decision.response)
    assert len(delivered) == 1
    assert json.loads(delivered[0]["function"]["arguments"]) == {"path": "evidence.txt"}
    # Weak draft + verifier only; no repair generation ran.
    assert [call["model"] for call in backend.calls] == ["weak", "strong"]
    assert decision.corrector_generation_count == 0
    assert len(decision.correction_actions) == 1


def test_rejected_terminal_with_null_replacement_goes_to_corrector() -> None:
    # The verifier could not name a tool action (the right repair is a reply), so the Strong
    # repair turn takes over.
    backend = FakeBackend(
        [
            completion(content="All done, nothing left."),
            rp_verdict(0),
            completion(content="The answer is 42."),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert decision.response["choices"][0]["message"]["content"] == "The answer is 42."
    assert [call["model"] for call in backend.calls] == ["weak", "strong", "strong"]
    assert decision.corrector_generation_count == 1


def test_rejected_terminal_message_shaped_replacement_is_guarded() -> None:
    # Safe salvage: a message-shaped replacement is unusable, so the parser drops it, records
    # the failure kind, and the repair runs without a wasted verifier retry. The bad action
    # never reaches the repair prompt.
    backend = FakeBackend(
        [
            completion(content="All done, nothing left."),
            rp_verdict(0, replacement={"type": "message", "content": "rewritten reply"}),
            completion(content="The answer is 42."),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert decision.verdict.format_retries == 0
    assert decision.corrector_generation_count == 1
    assert decision.replacement_validation_failure_count == 1
    assert decision.verdict.replacement_validation_failure_kind == "message_replacement"
    assert decision.verdict.replacement_action is None
    corrector_messages = backend.calls[2]["request"]["messages"]
    assert "rewritten reply" not in json.dumps(corrector_messages)
    assert decision.response["choices"][0]["message"]["content"] == "The answer is 42."
    assert not tool_calls_of(decision.response)
    assert [call["model"] for call in backend.calls] == ["weak", "strong", "strong"]


def test_replacement_guard_failure_discards_action_and_corrects() -> None:
    # A well-formed but ungrounded replacement (unknown tool) survives parsing and is dropped
    # by the runtime guard instead of a verifier retry.
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0, replacement={"tool": "missing_tool", "arguments": {}}),
            completion(content="fixed"),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert decision.verdict.format_retries == 0
    assert decision.verdict.replacement_validation_failure_kind == "unknown_tool"
    assert decision.replacement_validation_failure_count == 1
    assert decision.corrector_generation_count == 1
    corrector_messages = backend.calls[2]["request"]["messages"]
    assert "missing_tool" not in json.dumps(corrector_messages)


def test_replacement_identical_to_rejected_is_discarded() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0, replacement={"tool": "write_file", "arguments": {"path": "x"}}),
            completion(content="fixed"),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert decision.verdict.format_retries == 0
    assert decision.verdict.replacement_validation_failure_kind == "replays_rejected"
    assert decision.replacement_validation_failure_count == 1


def test_full_accept_strays_canonicalize_without_retry() -> None:
    # rp_verdict emits a non-null reason and a non-null replacement on a full accept; both are
    # canonicalized to null and recorded, no retry spent.
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(1, replacement={"tool": "read_file", "arguments": {"path": "e"}}),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "weak_verified"
    assert decision.verdict.accepted_steps == 1
    assert decision.verdict.format_retries == 0
    assert decision.verdict.reason == ""
    assert decision.verdict.replacement_action is None
    assert decision.verdict.reason_on_full_accept is True
    assert decision.verdict.replacement_on_full_accept is True
    assert decision.replacement_validation_failure_count == 0
    assert [call["model"] for call in backend.calls] == ["weak", "strong"]


@pytest.mark.parametrize(
    ("boundaries", "expect_fused"),
    [
        (("message", "mutation"), True),
        (("message",), True),
        (("mutation",), False),
        ((), False),
    ],
)
def test_terminal_replacement_respects_boundary_config(boundaries, expect_fused) -> None:
    responses = [
        completion(content="All done, nothing left."),
        rp_verdict(0, replacement={"tool": "read_file", "arguments": {"path": "e"}}),
    ]
    if not expect_fused:
        responses.append(completion(content="fixed"))
    backend = FakeBackend(responses)
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, fuse_correction_boundaries=boundaries)
    )

    decision = gateway.handle(request())

    if expect_fused:
        assert decision.source == "strong_fused_correction"
        assert len(tool_calls_of(decision.response)) == 1
    else:
        # The contract still emitted the replacement; the config chose not to use it.
        assert decision.source == "strong_correction"
        assert decision.verdict.replacement_action == {
            "tool": "read_file",
            "arguments": {"path": "e"},
        }


def test_mutation_boundary_replacement_fuses_when_enabled() -> None:
    backend = FakeBackend(
        [
            completion(("w0", "write_file", {"path": "out", "data": "v"})),
            rp_verdict(0, replacement={"tool": "read_file", "arguments": {"path": "e"}}),
        ]
    )
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, fuse_correction_boundaries=("mutation",))
    )

    decision = gateway.handle(request())

    assert decision.source == "strong_fused_correction"
    assert decision.boundary == "mutation"


def test_only_message_and_mutation_boundaries_are_fusable() -> None:
    # A horizon commit is never reviewed, so there is no verdict whose replacement could fuse.
    with pytest.raises(ValueError, match="fuse_correction_boundaries"):
        profile(fuse_correction_boundaries=("horizon",))
    with pytest.raises(ValueError, match="fuse_correction_boundaries"):
        profile(fuse_correction_boundaries=("message", "horizon"))


def test_fused_replacement_is_one_tool_call_without_corrector_call() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0, replacement={"tool": "read_file", "arguments": {"path": "y"}}),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "strong_fused_correction"
    delivered = tool_calls_of(decision.response)
    assert len(delivered) == 1
    assert json.loads(delivered[0]["function"]["arguments"]) == {"path": "y"}
    # Weak + verifier only: the replacement commits without a repair call.
    assert [call["model"] for call in backend.calls] == ["weak", "strong"]


# --- reasoning stays private --------------------------------------------------


def test_hidden_reasoning_never_enters_canonical_trajectory() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "read_file", {"path": "a"}),
                ("f2", "read_file", {"path": "b"}),
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    message = decision.response["choices"][0]["message"]
    assert "reasoning_content" not in message
    assert "reasoning" not in message
    # The private audit side keeps it: committed steps still carry reasoning.
    assert gateway.state.confirmed[-1].reasoning == "hidden-reason"


def test_draft_reasoning_stripped_but_kept_internally() -> None:
    backend = FakeBackend([completion(("w0", "read_file", {"path": "x"}))])
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))

    decision = gateway.handle(request())

    message = decision.response["choices"][0]["message"]
    assert "reasoning_content" not in message
    assert gateway.state.candidate[0].reasoning == "hidden-reason"


# --- contract strictness ------------------------------------------------------


def test_rp_fused_never_clamps_overlength_acceptance() -> None:
    with pytest.raises(ValueError, match="exceed"):
        parse_prefix_verdict(
            {"reason": "r", "accepted_steps": 4, "replacement_action": None},
            3,
            contract="rp_fused",
        )


def test_rp_fused_salvages_foreign_keys_and_bad_replacements() -> None:
    # Foreign keys cannot affect the prefix or reason: discarded and recorded, not retried.
    verdict = parse_prefix_verdict(
        {
            "reason": "r",
            "accepted_steps": 1,
            "rejection_code": "other",
            "corrector": "strong",
            "replacement_action": None,
        },
        3,
        contract="rp_fused",
    )
    assert verdict.accepted_steps == 1
    assert verdict.rejection_code is RejectionCode.OTHER
    assert verdict.corrector is None
    assert set(verdict.discarded_verdict_fields) == {"corrector", "rejection_code"}

    # Message-shaped replacement: dropped with a recorded kind, no raise.
    verdict = parse_prefix_verdict(
        {
            "reason": "r",
            "accepted_steps": 1,
            "replacement_action": {"type": "message", "content": "hi"},
        },
        3,
        contract="rp_fused",
    )
    assert verdict.replacement_action is None
    assert verdict.replacement_validation_failure_kind == "message_replacement"

    # Bad shape: same salvage path.
    verdict = parse_prefix_verdict(
        {
            "reason": "r",
            "accepted_steps": 1,
            "replacement_action": {"tool": "read_file"},
        },
        3,
        contract="rp_fused",
    )
    assert verdict.replacement_action is None
    assert verdict.replacement_validation_failure_kind == "malformed_replacement"

    # Full accept carrying a stray reason and replacement: canonicalized, flagged.
    verdict = parse_prefix_verdict(
        {
            "reason": "r",
            "accepted_steps": 3,
            "replacement_action": {"tool": "read_file", "arguments": {}},
        },
        3,
        contract="rp_fused",
    )
    assert verdict.accepted_steps == 3
    assert verdict.rejection_code is None
    assert verdict.reason == ""
    assert verdict.replacement_action is None
    assert verdict.reason_on_full_accept is True
    assert verdict.replacement_on_full_accept is True


def test_rp_fused_tool_replacement_parses() -> None:
    verdict = parse_prefix_verdict(
        {
            "reason": "Step 2 used the wrong path",
            "accepted_steps": 1,
            "replacement_action": {"tool": "read_file", "arguments": {"path": "y"}},
        },
        3,
        contract="rp_fused",
    )
    assert verdict.replacement_action == {
        "tool": "read_file",
        "arguments": {"path": "y"},
    }


# --- mixed corrector turns -----------------------------------------------------


def test_corrector_tool_calls_suppress_public_reply_but_keep_audit() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "read_file", {"path": "a"}),
                ("f2", "read_file", {"path": "b"}),
                content="I will fetch both files now.",
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    message = decision.response["choices"][0]["message"]
    assert [c["id"] for c in message["tool_calls"]] == ["f1", "f2"]
    # The tool calls are the repair; the public reply is not delivered.
    assert message["content"] is None
    # It stays audit-only: mirrored on the decision.
    assert decision.correction_suppressed_reply == "I will fetch both files now."
    # Canonical state carries only the two tool steps, no phantom message step.
    assert [
        step.action["tool"]
        for step in gateway.state.confirmed
        if step.actor is ModelRole.STRONG
    ] == ["read_file", "read_file"]


def test_corrector_reply_without_calls_still_delivers() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(content="The answer is 42."),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.response["choices"][0]["message"]["content"] == "The answer is 42."
    assert decision.correction_suppressed_reply is None
    assert decision.correction_tool_call_count == 0


# --- correction telemetry counters ----------------------------------------------


def test_correction_counters_distinguish_correction_kinds() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "read_file", {"path": "a"}),
                ("f2", "read_file", {"path": "b"}),
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.corrector_generation_count == 1
    assert decision.correction_tool_call_count == 2
    # The committed action count is len(correction_actions).
    assert len(decision.correction_actions) == 2
    assert gateway.correction_runtime_failure_count == 0
    assert gateway.correction_missing_observation_count == 0


def test_fused_replacement_is_not_a_corrector_generation() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0, replacement={"tool": "read_file", "arguments": {"path": "y"}}),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "strong_fused_correction"
    assert decision.corrector_generation_count == 0
    assert decision.correction_tool_call_count == 0
    assert len(decision.correction_actions) == 1


def test_corrector_validation_failure_counts_before_raising() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(("f1", "no_such_tool", {"path": "a"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    with pytest.raises(SpecAgentRuntimeError):
        gateway.handle(request())

    assert gateway.correction_runtime_failure_count == 1
    assert gateway.correction_missing_observation_count == 0


def test_missing_observation_counts_before_raising() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "read_file", {"path": "a"}),
                ("f2", "read_file", {"path": "b"}),
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))
    gateway.handle(request())

    with pytest.raises(SpecAgentRuntimeError):
        gateway.handle(
            request(
                messages=[
                    {"role": "user", "content": "Do the task"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "f1", "type": "function"},
                            {"id": "f2", "type": "function"},
                        ],
                    },
                    {"role": "tool", "tool_call_id": "f1", "content": "obs-a"},
                ]
            )
        )

    assert gateway.correction_missing_observation_count == 1
    assert gateway.correction_runtime_failure_count == 0


def test_corrector_undeclared_tool_is_reasked_once_when_enabled() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(("c1", "no_such_tool", {"path": "a"})),
            completion(("c2", "read_file", {"path": "a"})),
        ]
    )
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=1, corrector_reask_invalid_call=True)
    )

    decision = gateway.handle(request())

    assert gateway.corrector_reask_count == 1
    assert "not an available tool" in backend.calls[-1]["request"]["messages"][-1]["content"]
    assert [call["function"]["name"] for call in tool_calls_of(decision.response)] == [
        "read_file"
    ]


def test_corrector_undeclared_tool_still_fails_closed_by_default() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(("c1", "no_such_tool", {"path": "a"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    with pytest.raises(SpecAgentRuntimeError):
        gateway.handle(request())
    assert gateway.corrector_reask_count == 0


# --- Weak history reasoning -------------------------------------------------------


def _after_first_read() -> list[dict]:
    return [
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "w0", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"path\": \"x\"}"}}]},
        {"role": "tool", "tool_call_id": "w0", "content": "ok"},
    ]


def test_weak_history_reasoning_reattaches_weak_reasoning_when_enabled() -> None:
    backend = FakeBackend(
        [
            completion(("w0", "read_file", {"path": "x"})),
            completion(("w1", "read_file", {"path": "y"})),
        ]
    )
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, weak_history_reasoning=True)
    )
    first = gateway.handle(request())
    assert "reasoning_content" not in first.response["choices"][0]["message"]

    gateway.handle(request(messages=_after_first_read()))

    weak_view = backend.calls[1]["request"]["messages"]
    assert weak_view[1]["reasoning_content"] == "hidden-reason"


def test_weak_history_reasoning_is_off_by_default() -> None:
    backend = FakeBackend(
        [
            completion(("w0", "read_file", {"path": "x"})),
            completion(("w1", "read_file", {"path": "y"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))
    gateway.handle(request())
    gateway.handle(request(messages=_after_first_read()))
    assert "reasoning_content" not in backend.calls[1]["request"]["messages"][1]


# --- context overflow -------------------------------------------------------------


_OVERFLOW = RuntimeError(
    'upstream HTTP 400: {"message":"The input (662093 tokens) is longer than the model\'s '
    'context length (262144 tokens)."}'
)


def _overflow_transcript() -> list[dict]:
    return [
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "big", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"path\": \"huge\"}"}}]},
        {"role": "tool", "tool_call_id": "big", "content": "x" * 900_000},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "small", "type": "function",
             "function": {"name": "read_file", "arguments": "{\"path\": \"small\"}"}}]},
        {"role": "tool", "tool_call_id": "small", "content": "KEEP"},
    ]


def test_weak_overflow_drops_oversized_result_and_resamples_when_enabled() -> None:
    backend = FakeBackend([_OVERFLOW, completion(("w1", "read_file", {"path": "part"}))])
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, weak_overflow_resample=True)
    )

    decision = gateway.handle(request(messages=_overflow_transcript()))

    retried = backend.calls[1]["request"]["messages"]
    assert "too large to read" in retried[2]["content"]
    assert retried[4]["content"] == "KEEP"
    assert gateway.weak_overflow_resample_count == 1
    assert tool_calls_of(decision.response)[0]["function"]["name"] == "read_file"


def test_weak_overflow_still_fails_by_default() -> None:
    backend = FakeBackend([_OVERFLOW])
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))
    with pytest.raises(RuntimeError, match="context length"):
        gateway.handle(request(messages=_overflow_transcript()))


def _context_length_error() -> VerifierInputError:
    return VerifierInputError("upstream rejected the request context length")


def test_verifier_overflow_drops_largest_observation_when_enabled() -> None:
    backend = FakeBackend(
        [completion(content="The answer is 42."), _context_length_error(), rp_verdict(1)]
    )
    gateway = SpeculativeToolGateway(
        backend,
        # A large recent-observation cap keeps the full observation in the verifier payload.
        profile(horizon=4, verifier_recent_observation_chars=1_000_000,
                strong_overflow_drop=True),
    )
    gateway.state.confirmed = [
        AgentStep(ModelRole.WEAK, {"tool": "read_file"}, "x" * 900_000),
        AgentStep(ModelRole.WEAK, {"tool": "read_file"}, "KEEP"),
    ]

    gateway.handle(request())

    first, retried = (backend.calls[1]["request"], backend.calls[2]["request"])
    history = json.loads(retried["messages"][-1]["content"])["confirmed_history"]
    assert history[1]["observation"] == "KEEP"
    assert len(json.dumps(retried)) < len(json.dumps(first)) // 2
    assert "too large to read" in retried["messages"][-1]["content"]
    assert gateway.strong_overflow_drop_count == 1


def test_verifier_overflow_still_fails_by_default() -> None:
    backend = FakeBackend([completion(content="The answer is 42."), _context_length_error()])
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))
    with pytest.raises(VerifierInputError):
        gateway.handle(request())


def test_input_length_range_error_counts_as_context_overflow() -> None:
    error = RuntimeError('upstream HTTP 400: {"error":{"code":"invalid_parameter_error",'
                         '"message":"Range of input length should be [1, 258048]"}}')
    assert _context_overflow(error) is True


def test_prompt_too_long_error_counts_as_context_overflow() -> None:
    error = RuntimeError('upstream HTTP 400: {"error":{"message":"prompt is too long: 385469 '
                         'tokens \\u003e 262144 maximum (request id: x)"}}')
    assert _context_overflow(error) == (385469, 262144)


def test_provider_code_prompt_too_long_error_counts_as_context_overflow() -> None:
    error = RuntimeError('upstream HTTP 400: {"error":{"code":"1261","message":"Prompt 超长"}}')
    assert _context_overflow(error) is True
    assert _context_overflow(RuntimeError('upstream HTTP 429: {"error":{"code":"1113"}}')) is None


# --- verifier input ---------------------------------------------------------------


def test_empty_workspace_is_omitted_and_strict_answer_review_applies() -> None:
    backend = FakeBackend([completion(content="The answer is 42."), rp_verdict(1)])
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, workspace_dir="", strict_answer_review=True)
    )
    gateway.handle(request())
    messages = backend.calls[1]["request"]["messages"]
    payload = json.loads(messages[-1]["content"])
    assert "workspace_dir" not in payload["anchors"]
    assert STRICT_ANSWER_RULE in messages[0]["content"]


def test_reply_boundary_verifier_gets_uninvoked_tool_contracts() -> None:
    backend = FakeBackend([completion(content="The answer is 42."), rp_verdict(1)])
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, verifier_all_tool_contracts=True)
    )
    gateway.handle(request())
    messages = backend.calls[1]["request"]["messages"]
    payload = json.loads(messages[-1]["content"])
    assert sorted(payload["available_tool_contracts"]) == ["read_file", "write_file"]
    assert ALL_TOOL_REPLACEMENT_RULES in messages[0]["content"]


def test_tool_contracts_off_keeps_payload_unchanged() -> None:
    backend = FakeBackend([completion(content="The answer is 42."), rp_verdict(1)])
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))
    gateway.handle(request())
    messages = backend.calls[1]["request"]["messages"]
    payload = json.loads(messages[-1]["content"])
    assert "available_tool_contracts" not in payload
    assert ALL_TOOL_REPLACEMENT_RULES not in messages[0]["content"]


def test_rejected_reply_commits_uninvoked_tool_without_corrector() -> None:
    backend = FakeBackend([
        completion(content="I could not find it."),
        rp_verdict(0, {"tool": "read_file", "arguments": {"path": "notes.md"}}),
    ])
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, verifier_all_tool_contracts=True)
    )
    decision = gateway.handle(request())
    assert decision.source == "strong_fused_correction"
    assert [c["function"]["name"] for c in tool_calls_of(decision.response)] == ["read_file"]
    assert len(backend.calls) == 2


def test_all_tool_contracts_list_only_uninvoked_tools() -> None:
    backend = FakeBackend([completion(("w1", "write_file", {"path": "a"})), rp_verdict(1)])
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, verifier_all_tool_contracts=True)
    )
    gateway.handle(request())
    messages = backend.calls[1]["request"]["messages"]
    payload = json.loads(messages[-1]["content"])
    assert sorted(payload["available_tool_contracts"]) == ["read_file"]
    assert ALL_TOOL_REPLACEMENT_RULES in messages[0]["content"]


def _after_call(decision) -> list[dict]:
    call = tool_calls_of(decision.response)[0]
    return [
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "obs"},
    ]


def test_weak_acts_again_after_a_correction() -> None:
    backend = FakeBackend([
        completion(content="I could not find it."),
        rp_verdict(0),
        completion(("s1", "read_file", {"path": "notes.md"})),
        completion(content="The answer is 42."),
        rp_verdict(1),
    ])
    gateway = SpeculativeToolGateway(backend, profile(horizon=4))
    first = gateway.handle(request())
    second = gateway.handle(request(messages=_after_call(first)))
    assert backend.calls[3]["model"] == "weak"
    assert second.source == "weak_verified"


def test_readonly_tool_is_not_a_mutation_boundary() -> None:
    backend = FakeBackend([completion(("w1", "write_file", {"path": "a"}))])
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, readonly_tools=("write_file",))
    )
    decision = gateway.handle(request())
    assert decision.source == "weak_read"
    assert len(backend.calls) == 1


def _review_payload_with_long_read(**overrides) -> dict:
    backend = FakeBackend([
        completion(("w1", "read_file", {"path": "a"})),
        completion(content="The answer is 42."),
        rp_verdict(2),
    ])
    gateway = SpeculativeToolGateway(backend, profile(horizon=4, **overrides))
    first = gateway.handle(request())
    call = tool_calls_of(first.response)[0]
    gateway.handle(request(messages=[
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "x" * 5000},
    ]))
    return json.loads(backend.calls[2]["request"]["messages"][-1]["content"])


def test_recent_observations_keep_more_characters_when_configured() -> None:
    short = _review_payload_with_long_read()
    full = _review_payload_with_long_read(verifier_recent_observation_chars=12000)
    assert len(short["proposed_trajectory"][0]["observation"]) < 1300
    assert len(full["proposed_trajectory"][0]["observation"]) == 5000


def test_weak_step_budget_hands_the_task_to_strong_once() -> None:
    backend = FakeBackend([
        completion(("s1", "read_file", {"path": "b"})),
        completion(content="The answer is 42."),
    ])
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=4, strong_takeover_after_tool_results=1)
    )
    transcript = _after_first_read()
    first = gateway.handle(request(messages=transcript))
    assert first.source == "strong_takeover"
    assert backend.calls[0]["request"]["messages"][-1]["content"] == WEAK_BUDGET_NOTE
    second = gateway.handle(request(messages=[*transcript, *_after_call(first)[1:]]))
    assert second.source == "strong_takeover"
    # The handoff note is sent on the first Strong turn only.
    assert backend.calls[1]["request"]["messages"][-1]["role"] == "tool"
    assert [c["model"] for c in backend.calls] == ["strong", "strong"]


def test_verifier_uses_its_own_body_and_cap() -> None:
    seen = []

    class Recording(FakeBackend):
        def complete(self, request, *, model, reasoning_effort=None, max_tokens=None,
                     extra_body=None, is_cancelled=None):
            seen.append((extra_body, max_tokens))
            return super().complete(request, model=model)

    backend = Recording([
        completion(("w1", "write_file", {"path": "a"})), rp_verdict(1),
        completion(content="The answer is 42."), rp_verdict(1),
    ])
    gateway = SpeculativeToolGateway(backend, profile(
        horizon=4, verifier_max_tokens=500,
        weak_extra_body={"enable_thinking": True},
        verifier_extra_body={"enable_thinking": False}))
    first = gateway.handle(request())
    gateway.handle(request(messages=_after_call(first)))
    assert seen[0] == ({"enable_thinking": True}, None)
    assert seen[1] == ({"enable_thinking": False}, 500)
    assert seen[3] == ({"enable_thinking": False}, 500)


def test_answer_checks_only_on_reply_reviews() -> None:
    backend = FakeBackend([
        completion(("w1", "write_file", {"path": "a"})), rp_verdict(1),
        completion(content="The answer is 42."), rp_verdict(1),
    ])
    gateway = SpeculativeToolGateway(backend, profile(horizon=4, verifier_answer_checks=True))
    first = gateway.handle(request())
    gateway.handle(request(messages=_after_call(first)))
    assert ANSWER_REVIEW_CHECKS not in backend.calls[1]["request"]["messages"][0]["content"]
    assert ANSWER_REVIEW_CHECKS in backend.calls[3]["request"]["messages"][0]["content"]


def test_partially_returned_corrector_batch_tombstones_the_dropped_call() -> None:
    # The client's transcript no longer references f2 at all, so its result can never arrive:
    # it is tombstoned and the run continues.
    backend = FakeBackend([
        completion(DRAFT_WRITE),
        rp_verdict(0),
        completion(("f1", "read_file", {"path": "a"}), ("f2", "read_file", {"path": "b"})),
        completion(("w1", "write_file", {"path": "c"})),
        rp_verdict(1),
    ])
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))
    gateway.handle(request())
    decision = gateway.handle(request(messages=[
        {"role": "user", "content": "Do the task"},
        {"role": "tool", "tool_call_id": "f1", "content": "obs-a"},
    ]))
    assert decision.source == "weak_verified"
    assert gateway.correction_missing_observation_count == 1
    by_id = {
        (s.metadata or {}).get("tool_call_id"): s for s in gateway.state.confirmed
    }
    assert by_id["f1"].observation == "obs-a"
    assert "missing_observation" in by_id["f2"].observation
    assert by_id["f2"].metadata["observation_lost"] is True


def test_corrector_max_tokens_caps_strong_actor_turns() -> None:
    seen = []

    class Recording(FakeBackend):
        def complete(self, request, *, model, reasoning_effort=None, max_tokens=None,
                     extra_body=None, is_cancelled=None):
            seen.append(max_tokens)
            return super().complete(request, model=model)

    backend = Recording([
        completion(content="Draft."), rp_verdict(0),
        completion(("s1", "read_file", {"path": "a"})),
    ])
    gateway = SpeculativeToolGateway(backend, profile(horizon=4, corrector_max_tokens=4096))
    gateway.handle(request())
    assert seen[-1] == 4096


def test_append_only_reviews_share_a_growing_prefix() -> None:
    backend = FakeBackend([
        completion(("w1", "write_file", {"path": "a"})), rp_verdict(1),
        completion(("w2", "write_file", {"path": "b"})), rp_verdict(1),
    ])
    gateway = SpeculativeToolGateway(backend, profile(
        horizon=1, verifier_all_tool_contracts=True, verifier_append_only=True))
    first = gateway.handle(request())
    gateway.handle(request(messages=_after_call(first)))
    one, two = (backend.calls[i]["request"]["messages"] for i in (1, 3))
    assert one[0]["content"] == two[0]["content"]
    p1, p2 = json.loads(one[-1]["content"]), json.loads(two[-1]["content"])
    assert list(p2)[:4] == ["task", "anchors", "available_tool_contracts", "confirmed_history"]
    assert "proposed_trajectory" in list(p2)[-2:]
    assert p1["available_tool_contracts"] == p2["available_tool_contracts"]
    head = json.dumps({k: p2[k] for k in ("task", "anchors", "available_tool_contracts")})
    assert head == json.dumps({k: p1[k] for k in ("task", "anchors", "available_tool_contracts")})


# --- tool_call_id hygiene ------------------------------------------------------------


def test_duplicate_corrector_call_id_is_reminted() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("dup", "read_file", {"path": "a"}),
                ("dup", "read_file", {"path": "b"}),
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    decision = gateway.handle(request())

    ids = [call["id"] for call in tool_calls_of(decision.response)]
    assert ids[0] == "dup" and ids[1] != "dup" and len(set(ids)) == 2
    assert gateway.reminted_call_id_count == 1


def test_corrector_reusing_a_delivered_call_id_is_reminted() -> None:
    # Strong re-used the label of a call delivered earlier for a new write; the label is only
    # the model's pairing id, so the call gets a fresh one instead of failing the turn.
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(("write_45", "write_file", {"path": "out", "data": "new"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))
    gateway.state.confirmed = [
        AgentStep(
            ModelRole.STRONG,
            {"tool": "write_file", "arguments": {"path": "out", "data": "old"}},
            "written",
            metadata={"pending": False, "tool_call_id": "write_45"},
        )
    ]

    decision = gateway.handle(request())

    (call,) = tool_calls_of(decision.response)
    assert call["id"] != "write_45" and call["function"]["name"] == "write_file"
    assert gateway.state.confirmed[-1].metadata["tool_call_id"] == call["id"]


def test_identical_mutations_in_one_corrector_turn_still_fail_closed() -> None:
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("dup", "write_file", {"path": "out", "data": "v"}),
                ("dup", "write_file", {"path": "out", "data": "v"}),
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))

    with pytest.raises(SpecAgentRuntimeError) as raised:
        gateway.handle(request())

    assert raised.value.code == RuntimeErrorCode.REPLAY_DIVERGENCE


def test_dropped_corrector_calls_are_tombstoned_not_wedged() -> None:
    # A transcript reset removes the assistant message carrying the repair calls entirely, so
    # no observation can ever arrive. The pending calls are tombstoned in place and the session
    # keeps going instead of raising on every subsequent request.
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "read_file", {"path": "a"}),
                ("f2", "read_file", {"path": "b"}),
            ),
            completion(("w1", "write_file", {"path": "y"})),
            rp_verdict(1),
            completion(("w2", "write_file", {"path": "z"})),
            rp_verdict(1),
        ]
    )
    gateway = SpeculativeToolGateway(backend, profile(horizon=1))
    gateway.handle(request())

    # The client reset its transcript: both repair calls are gone entirely.
    decision = gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "user", "content": "Start over."},
            ]
        )
    )

    assert decision.source == "weak_verified"
    assert gateway.correction_missing_observation_count == 2
    lost = [
        step
        for step in gateway.state.confirmed
        if (step.metadata or {}).get("observation_lost")
    ]
    assert len(lost) == 2
    assert all(
        step.observation and "missing_observation" in step.observation
        for step in lost
    )

    # A client that re-ran the calls anyway still binds the late results: the real observation
    # replaces the tombstone.
    gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "f1", "content": "obs-a"},
                {"role": "tool", "tool_call_id": "f2", "content": "obs-b"},
            ]
        )
    )
    assert [
        step.observation
        for step in gateway.state.confirmed
        if str((step.metadata or {}).get("tool_call_id") or "") in {"f1", "f2"}
    ] == ["obs-a", "obs-b"]


# --- environment-validated mode (strict_action_validation=False) ----------------------


def _executed_write() -> AgentStep:
    return AgentStep(
        ModelRole.WEAK,
        {"tool": "write_file", "arguments": {"path": "out", "data": "v"}},
        "written",
        metadata={"pending": False, "tool_call_id": "old"},
    )


def _replayed_mutation_gateway():
    """The repair turn re-issues a mutation already executed earlier in history."""
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(("f1", "write_file", {"path": "out", "data": "v"})),
        ]
    )
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=1, strict_action_validation=False)
    )
    gateway.state.confirmed = [_executed_write()]
    return gateway, backend


def test_corrector_mutation_replay_partial_withhold_env_mode() -> None:
    """A replay mixed into a multi-call turn is withheld; the rest deliver."""
    backend = FakeBackend(
        [
            completion(DRAFT_WRITE),
            rp_verdict(0),
            completion(
                ("f1", "write_file", {"path": "out", "data": "v"}),
                ("f2", "read_file", {"path": "notes"}),
            ),
        ]
    )
    gateway = SpeculativeToolGateway(
        backend, profile(horizon=1, strict_action_validation=False)
    )
    gateway.state.confirmed = [_executed_write()]
    decision = gateway.handle(request())
    assert decision.correction_withheld_replays == 1
    delivered = tool_calls_of(decision.response)
    assert [(c["id"], c["function"]["name"]) for c in delivered] == [("f2", "read_file")]


def test_corrector_mutation_replay_persistent_env_mode_delivers_reply() -> None:
    """Replay again on the retry: the emptied turn delivers as a reply, no abort."""
    gateway, backend = _replayed_mutation_gateway()
    backend.responses.append(completion(("f2", "write_file", {"path": "out", "data": "v"})))
    decision = gateway.handle(request())
    assert decision.correction_withheld_replays == 2
    assert not tool_calls_of(decision.response)


def test_corrector_mutation_replay_retried_once_env_mode() -> None:
    """An all-replay repair turn is withheld, retried once with feedback, and the retry's clean
    call delivers -- no abort."""
    gateway, backend = _replayed_mutation_gateway()
    # The first repair turn replays; add a clean second generation.
    backend.responses.append(completion(("f2", "write_file", {"path": "out2", "data": "w"})))
    decision = gateway.handle(request())
    assert decision.source == "strong_correction"
    assert decision.correction_withheld_replays == 1
    assert [a.get("tool") for a in decision.correction_actions] == ["write_file"]
    delivered = tool_calls_of(decision.response)
    assert [c["id"] for c in delivered] == ["f2"]
