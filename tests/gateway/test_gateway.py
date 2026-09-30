"""Unit tests for the speculative tool gateway.

The gateway runs one protocol: Weak drafts, reads are released at once, and the first
state-changing call or text reply is held for one Strong review. A candidate of ``horizon`` reads
commits without review. A rejection commits the verifier's schema-valid replacement directly at a
fusable boundary, and otherwise takes one Strong repair turn that commits without re-review.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from collections import deque

import pytest

from specagent.protocol.evidence import VerifierInputError
from specagent.protocol.types import AgentStep, ModelRole
from specagent.protocol.verifier import (
    RP_FUSED_VERIFIER_SYSTEM_PROMPT,
    STRICT_ANSWER_RULE,
)
from specagent.runtime import gateway as gateway_module
from specagent.runtime.gateway import (
    HISTORY_WINDOW_STRIDE,
    QA_CORRECTION_SUFFIX,
    GatewayConfig,
    SpeculativeToolGateway,
    _verification_system_constraints,
    completion_to_sse,
)

# --- helpers -------------------------------------------------------------------------------------


def completion(*calls: tuple[str, str, dict], content: str | None = None) -> dict:
    """A chat completion carrying the given (id, tool, arguments) calls, or a text reply."""
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
                    "reasoning_content": "reason",
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def verdict(accepted: int, replacement: dict | None = None, reason: str | None = None) -> dict:
    """A verifier response in the three-key contract."""
    return completion(
        content=json.dumps(
            {"reason": reason, "accepted_steps": accepted, "replacement_action": replacement}
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
        self.calls.append(
            {
                "request": request,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
            }
        )
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
        "stream": True,
    }


def config(**overrides) -> GatewayConfig:
    return GatewayConfig(weak_model="weak", strong_model="strong", **overrides)


def tool_calls_of(response: dict) -> list[dict]:
    return response["choices"][0]["message"].get("tool_calls") or []


def after_call(decision) -> list[dict]:
    """The client transcript after it executed the delivered calls."""
    calls = tool_calls_of(decision.response)
    return [
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": None, "tool_calls": calls},
        *({"role": "tool", "tool_call_id": call["id"], "content": "obs"} for call in calls),
    ]


def verifier_payload(backend: FakeBackend, index: int) -> dict:
    return json.loads(backend.calls[index]["request"]["messages"][-1]["content"])


def models(backend: FakeBackend) -> list[str]:
    return [call["model"] for call in backend.calls]


# --- configuration -------------------------------------------------------------------------------


def test_verifier_overlay_cannot_replace_budgeted_request_payload() -> None:
    with pytest.raises(ValueError, match="verifier_extra_body cannot override"):
        config(verifier_extra_body={"messages": [{"role": "user", "content": "hidden"}]})


def test_every_phase_the_gateway_emits_has_an_extra_body_mapping() -> None:
    """A new phase that forgets the mapping would silently send no role body."""
    source = inspect.getsource(SpeculativeToolGateway._complete)
    mapped = set(re.findall(r'"(\w+)":\s*self\.config\.\w+_extra_body', source))
    emitted = set(re.findall(r'phase="(\w+)"', inspect.getsource(gateway_module)))
    assert emitted
    assert not emitted - mapped, f"phases with no extra_body mapping: {emitted - mapped}"


def test_role_extra_bodies_reach_their_own_phase_only() -> None:
    weak_body = {"chat_template_kwargs": {"enable_thinking": True}}
    verify_body = {"chat_template_kwargs": {"enable_thinking": False}}
    strong_body = {"chat_template_kwargs": {"thinking": True}}
    backend = FakeBackend(
        [
            completion(("bad", "write_file", {"path": "wrong"})),
            verdict(0),
            completion(("fix", "write_file", {"path": "right"})),
        ]
    )
    gateway = SpeculativeToolGateway(
        backend,
        config(
            weak_extra_body=weak_body,
            verifier_extra_body=verify_body,
            strong_extra_body=strong_body,
        ),
    )

    gateway.handle(request())

    assert [call["extra_body"] for call in backend.calls] == [weak_body, verify_body, strong_body]


# --- reads, boundaries and the horizon -----------------------------------------------------------


def test_read_only_call_runs_without_verifier_before_horizon() -> None:
    backend = FakeBackend([completion(("r1", "read_file", {"path": "a"}))])
    gateway = SpeculativeToolGateway(backend, config(horizon=4))

    decision = gateway.handle(request())

    assert decision.source == "weak_read"
    assert decision.candidate_length == 1
    assert models(backend) == ["weak"]


def test_weak_reasoning_effort_comes_from_config() -> None:
    backend = FakeBackend([completion(content="done"), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config(weak_reasoning_effort="high"))

    gateway.handle(request())

    assert backend.calls[0]["reasoning_effort"] == "high"


def test_mutation_is_verified_before_returning() -> None:
    backend = FakeBackend([completion(("w1", "write_file", {"path": "a"})), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config())

    decision = gateway.handle(request())

    assert decision.source == "weak_verified"
    assert decision.mutating is True
    assert decision.boundary == "mutation"
    assert decision.verdict and decision.verdict.accepted_steps == 1
    assert models(backend) == ["weak", "strong"]
    assert [call.phase for call in decision.model_calls] == ["weak_act", "strong_verify"]
    verify = backend.calls[1]
    assert verify["max_tokens"] == 1024
    assert verify["reasoning_effort"] is None
    assert verify["request"]["enable_thinking"] is False
    assert verify["request"]["response_format"] == {"type": "json_object"}
    assert decision.verdict.raw.startswith("{")


def test_horizon_of_reads_commits_without_review() -> None:
    backend = FakeBackend([completion(("r1", "read_file", {"path": "a"}))])
    gateway = SpeculativeToolGateway(backend, config(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "runtime_auto_accept"
    assert decision.boundary == "horizon"
    assert decision.verifier_accepted_steps == 1
    assert models(backend) == ["weak"]
    assert [step.metadata["tool_call_id"] for step in gateway.state.confirmed] == ["r1"]
    assert gateway.state.candidate == []


def test_parallel_reads_beyond_the_horizon_commit_whole_without_review() -> None:
    backend = FakeBackend(
        [
            completion(
                ("a", "read_file", {"path": "x"}),
                ("b", "read_file", {"path": "y"}),
                ("c", "read_file", {"path": "z"}),
            )
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(horizon=2))

    decision = gateway.handle(request())

    assert decision.source == "runtime_auto_accept"
    assert decision.candidate_length == 3
    assert [c["id"] for c in tool_calls_of(decision.response)] == ["a", "b", "c"]
    assert len(gateway.state.confirmed) == 3


def test_text_reply_is_verified_before_reaching_user() -> None:
    backend = FakeBackend([completion(content="All booked."), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config(horizon=1))

    decision = gateway.handle(request())

    assert decision.source == "weak_verified"
    assert decision.boundary == "message"
    assert models(backend) == ["weak", "strong"]
    step = gateway.state.confirmed[-1]
    assert step.action == {"type": "message", "content": "All booked."}
    assert step.metadata["pending"] is False
    payload = verifier_payload(backend, 1)
    assert payload["proposed_trajectory"][0]["pending"] is True
    assert payload["proposed_trajectory"][0]["action"]["type"] == "message"


def test_hallucinated_tool_name_is_released_without_verification() -> None:
    # ``writing`` is not among the request's tools, so it cannot execute; gating it would only
    # buy a Strong call for a runtime error the drafting model can recover from.
    backend = FakeBackend([completion(("h1", "writing", {"path": "a", "content": "x"}))])
    gateway = SpeculativeToolGateway(backend, config(horizon=4))

    decision = gateway.handle(request())

    assert decision.source == "weak_read"
    assert decision.mutating is False
    assert models(backend) == ["weak"]


def test_request_without_tools_keeps_default_gating() -> None:
    backend = FakeBackend([completion(("m1", "mystery_tool", {})), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config(horizon=4))
    req = request()
    req["tools"] = []

    decision = gateway.handle(req)

    assert decision.mutating is True
    assert models(backend) == ["weak", "strong"]


def test_search_is_a_read_until_horizon() -> None:
    backend = FakeBackend([completion(("s1", "search", {"query": "who won"}))])
    gateway = SpeculativeToolGateway(backend, config(horizon=4))
    tools = [
        {
            "type": "function",
            "function": {"name": "search", "description": "bm25", "parameters": {"type": "object"}},
        }
    ]

    decision = gateway.handle(request() | {"tools": tools})

    assert decision.source == "weak_read"
    assert decision.mutating is False
    assert models(backend) == ["weak"]


def test_api_fetch_post_is_verified_before_returning() -> None:
    backend = FakeBackend(
        [completion(("w1", "api_fetch", {"method": "POST", "url": "/messages"})), verdict(1)]
    )
    gateway = SpeculativeToolGateway(backend, config())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "api_fetch",
                "description": "http",
                "parameters": {"type": "object"},
            },
        }
    ]

    decision = gateway.handle(request() | {"tools": tools})

    assert decision.source == "weak_verified"
    assert decision.mutating is True
    assert models(backend) == ["weak", "strong"]


def test_schema_invalid_write_still_goes_to_the_model_verifier() -> None:
    backend = FakeBackend(
        [
            completion(("w", "write_file", {"content": 42})),
            verdict(0, reason="content must be a string"),
            completion(content="cannot write"),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())
    req = request()
    req["tools"][1]["function"]["parameters"] = {
        "type": "object",
        "properties": {"content": {"type": "string"}},
    }

    decision = gateway.handle(req)

    assert models(backend) == ["weak", "strong", "strong"]
    assert decision.verdict.accepted_steps == 0
    assert decision.verdict.diagnostic_status == "model_verdict"
    contracts = verifier_payload(backend, 1)["invoked_tool_contracts"]
    assert contracts["write_file"]["function"]["parameters"]["properties"]["content"] == {
        "type": "string"
    }


# --- reconciliation of observations --------------------------------------------------------------


def test_next_user_message_binds_as_observation_of_previous_reply() -> None:
    backend = FakeBackend(
        [
            completion(content="Which date?"),
            verdict(1),
            completion(("w1", "write_file", {"path": "a"})),
            verdict(1),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())
    gateway.handle(request())

    gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "assistant", "content": "Which date?"},
                {"role": "user", "content": "Next Friday"},
            ]
        )
    )

    assert gateway.state.confirmed[0].observation == "Next Friday"
    payload = verifier_payload(backend, 3)
    assert payload["task"]["instruction"] == "Do the task"
    assert payload["confirmed_history"][0]["observation"] == "Next Friday"


@pytest.mark.parametrize(
    ("raw_id", "wire_id"),
    [
        ("call_917a3ec6511f4005b9f57c6b", "call917a3ec6511f4005b9f57c6b"),
        ("call_" + "a" * 60, "call" + "a" * 36),
        ("functions.read_file:0", "functions.read_file:0"),
    ],
)
def test_sanitized_tool_result_id_binds_to_its_call(raw_id, wire_id) -> None:
    backend = FakeBackend(
        [
            completion((raw_id, "read_file", {"path": "a"})),
            completion(("w1", "write_file", {"path": "b"})),
            verdict(2),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())
    first = gateway.handle(request())
    gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": wire_id, "content": "price=5750"},
            ]
        )
    )

    step = verifier_payload(backend, 2)["proposed_trajectory"][0]
    assert step["observation"] == "price=5750"
    assert not step.get("pending", False)
    # The verifier sees the draft's reasoning; the client never does.
    assert step["reasoning"] == "reason"
    message = first.response["choices"][0]["message"]
    assert message["tool_calls"][0]["id"] == raw_id
    assert "reasoning_content" not in message


def test_normalized_id_collision_does_not_bind_even_an_exact_match() -> None:
    backend = FakeBackend(
        [
            completion(
                ("call_abc", "read_file", {"path": "a"}),
                ("callabc", "read_file", {"path": "b"}),
            )
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())
    gateway.handle(request())

    with pytest.raises(ValueError, match="ambiguous tool result"):
        gateway.handle(
            request(
                messages=[
                    {"role": "user", "content": "Do the task"},
                    {"role": "tool", "tool_call_id": "callabc", "content": "must not guess owner"},
                ]
            )
        )
    assert all(step.observation is None for step in gateway.state.candidate)
    assert len(backend.calls) == 1


def test_shortened_transcript_keeps_new_user_instruction_and_original_task() -> None:
    backend = FakeBackend(
        [
            completion(content="Which date?"),
            verdict(1),
            completion(content="Noted."),
            verdict(1),
            completion(("w1", "write_file", {"path": "a"})),
            verdict(1),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())
    gateway.handle(request())
    gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "user", "content": "Friday"},
                {"role": "user", "content": "Budget 100"},
            ]
        )
    )
    shortened = request(
        messages=[
            {"role": "user", "content": "Condensed task summary" + "x" * 13000},
            {"role": "user", "content": "NEW: do not purchase anything"},
        ]
    )

    gateway.handle(shortened)

    payload = verifier_payload(backend, 5)
    assert payload["task"]["instruction"].startswith("Do the task")
    assert "NEW: do not purchase anything" in payload["task"]["instruction"]
    before = list(gateway.state.unbound_user_messages)
    gateway.handle(shortened)
    assert gateway.state.unbound_user_messages == before


# --- verifier payload ----------------------------------------------------------------------------


def test_trimmed_history_is_marked_as_incomplete_in_verifier_payload() -> None:
    backend = FakeBackend([completion(content="answer"), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config())
    gateway.state.confirmed = [
        AgentStep(ModelRole.WEAK, {"tool": "read_file"}, "evidence " * 400) for _ in range(65)
    ]

    gateway.handle(request())

    integrity = verifier_payload(backend, 1)["input_integrity"]
    assert integrity["status"] == "truncated"
    assert integrity["profile"] == "trimmed"
    assert integrity["omissions"]
    assert integrity["source_hash"]


def test_history_window_advances_in_strides_so_the_prefix_stays_cacheable() -> None:
    """A per-step slide would rewrite the front of the payload at every boundary."""

    def first_retained(step_count: int) -> str:
        backend = FakeBackend([completion(content="answer"), verdict(1)])
        gateway = SpeculativeToolGateway(backend, config())
        gateway.state.confirmed = [
            AgentStep(ModelRole.WEAK, {"tool": "read_file"}, f"step-{index}")
            for index in range(step_count)
        ]
        gateway.handle(request())
        return verifier_payload(backend, 1)["confirmed_history"][0]["observation"]

    # Below the window nothing is dropped.
    assert first_retained(64) == "step-0"
    # Within one stride the window start holds still, so consecutive boundaries share a prefix.
    assert first_retained(96) == "step-32"
    assert first_retained(120) == "step-32"
    assert first_retained(127) == "step-32"
    # The next stride drops exactly one stride of steps.
    assert first_retained(128) == "step-64"


def test_dropped_history_steps_are_recorded_as_omissions() -> None:
    backend = FakeBackend([completion(content="answer"), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config())
    gateway.state.confirmed = [
        AgentStep(ModelRole.WEAK, {"tool": "read_file"}, f"observation {i}") for i in range(100)
    ]

    gateway.handle(request())

    payload = verifier_payload(backend, 1)
    # 100 steps, window 64, stride 32: one stride dropped, 68 kept.
    assert len(payload["confirmed_history"]) == 100 - HISTORY_WINDOW_STRIDE
    assert payload["confirmed_history"][0]["observation"] == f"observation {HISTORY_WINDOW_STRIDE}"
    assert payload["input_integrity"]["status"] == "truncated"
    assert {"section": "confirmed_history", "kind": "steps", "removed": 32} in payload[
        "input_integrity"
    ]["omissions"]


def test_input_budget_failure_does_not_release_mutation_or_call_corrector() -> None:
    backend = FakeBackend([completion(("w", "write_file", {"content": "x"}))])
    gateway = SpeculativeToolGateway(backend, config(verifier_max_input_tokens=100))
    delivered = []

    with pytest.raises(VerifierInputError):
        gateway.handle(request(), deliver=delivered.append)

    assert not delivered
    assert not gateway.state.confirmed
    assert not gateway.state.candidate
    assert models(backend) == ["weak"]


def test_history_reasoning_can_be_pruned_but_not_the_step_under_review() -> None:
    backend = FakeBackend([completion(("w", "write_file", {"path": "p"})), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config(verifier_history_reasoning=False))
    gateway.state.confirmed.append(
        AgentStep(
            actor=ModelRole.WEAK,
            reasoning="a long committed chain of thought the verifier does not need",
            action={"tool": "read_file", "arguments": {"path": "p"}},
            observation="ok",
            metadata={"pending": False},
        )
    )

    gateway.handle(request())

    payload = verifier_payload(backend, 1)
    assert all("reasoning" not in step for step in payload["confirmed_history"])
    # The step under review keeps its own reasoning.
    assert any(step.get("reasoning") for step in payload["proposed_trajectory"])
    assert payload["request_evidence_metadata"]["pruned_context"]["reasoning_steps"] == 1


def test_history_reasoning_is_kept_by_default() -> None:
    backend = FakeBackend([completion(("w", "write_file", {"path": "p"})), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config())
    gateway.state.confirmed.append(
        AgentStep(
            actor=ModelRole.WEAK,
            reasoning="kept",
            action={"tool": "read_file", "arguments": {"path": "p"}},
            observation="ok",
            metadata={"pending": False},
        )
    )

    gateway.handle(request())

    payload = verifier_payload(backend, 1)
    assert any(step.get("reasoning") == "kept" for step in payload["confirmed_history"])
    assert "request_evidence_metadata" not in payload


def test_candidate_reasoning_can_be_excluded_from_the_verifier() -> None:
    backend = FakeBackend([completion(("w", "write_file", {"path": "p"})), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config(verifier_include_reasoning=False))

    gateway.handle(request())

    payload = verifier_payload(backend, 1)
    assert payload["proposed_trajectory"]
    assert all("reasoning" not in step for step in payload["proposed_trajectory"])
    assert payload["request_evidence_metadata"]["pruned_context"]["candidate_reasoning_steps"] == 1


def test_strict_answer_review_replaces_the_permissive_qa_exception() -> None:
    backend = FakeBackend([completion(content="happy to help"), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config(strict_answer_review=True))

    gateway.handle(request(messages=[{"role": "user", "content": "When was it created?"}]))

    system = backend.calls[1]["request"]["messages"][0]["content"]
    assert STRICT_ANSWER_RULE in system
    assert "confidence is low: accept" not in system
    assert "confidence is low: accept" in RP_FUSED_VERIFIER_SYSTEM_PROMPT


def test_prompt_suffix_is_appended_to_the_verifier_system_prompt() -> None:
    backend = FakeBackend([completion(content="done"), verdict(1)])
    gateway = SpeculativeToolGateway(backend, config(verifier_prompt_suffix="Extra rule."))

    gateway.handle(request())

    assert backend.calls[1]["request"]["messages"][0]["content"].endswith("\n\nExtra rule.")


# --- system constraints shown to the verifier ----------------------------------------------------

SYSTEM_PROMPT = """You are a personal assistant.
## Tooling
- read: Read file contents
## Skills
Skills live in skills/.
### Skill Workshop
Author new skills here.
## Workspace
Your working directory is: /srv/agent/workspace
## Group Chats
### Know When to Speak
Stay quiet unless addressed.
## Red Lines
- Don't exfiltrate private data.
"""




def test_verification_system_constraints_fails_open_when_headings_are_unrecognised():
    plain = "Operating rules with no markdown headings at all.\nSecond line."
    kept, receipt = _verification_system_constraints(plain)
    assert kept == plain
    assert receipt == {"dropped_sections": 0, "dropped_chars": 0, "kept_chars": len(plain)}


INJECTED_WORKSPACE = """Preamble.
## Tooling
- read: Read file contents
## /ws/AGENTS.md
# AGENTS.md - Your Workspace
## Red Lines
- Don't exfiltrate private data.
## /ws/SOUL.md
# SOUL.md - Who You Are
You're not a chatbot.
## Silent Replies
Respond with ONLY: NO_REPLY
# Dynamic Project Context
## Messaging
Reply in current session.
"""




def test_a_retained_section_survives_a_dropped_parent():
    """Silent Replies defines the NO_REPLY contract and sits inside a dropped persona file."""
    kept, _ = _verification_system_constraints(INJECTED_WORKSPACE)
    assert "## Silent Replies" in kept and "NO_REPLY" in kept


# --- verdict contract ----------------------------------------------------------------------------


def test_invalid_verdict_fails_closed_on_unreleased_batch_only() -> None:
    backend = FakeBackend(
        [
            completion(("r1", "read_file", {"path": "a"})),
            completion(("w1", "write_file", {"path": "b"})),
            completion(content="not json"),
            completion(content="still not json"),
            completion(("fix", "write_file", {"path": "c"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(horizon=4))
    gateway.handle(request())

    decision = gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "r1", "content": "file a"},
            ]
        )
    )

    assert decision.source == "strong_correction"
    assert decision.verdict.accepted_steps == 1
    assert decision.verdict.diagnostic_status == "format_failure"
    assert decision.verdict.format_failure_kind == "unparseable"
    assert decision.verdict.format_retries == 1
    assert "not a proven task error" in decision.verdict.reason
    assert decision.executed_rejected_steps == 0
    assert [step.metadata.get("tool_call_id") for step in gateway.state.confirmed] == ["r1", "fix"]
    assert "verifier_rejected" not in gateway.state.confirmed[0].metadata
    # The retry repeats the request with a trailing format reminder.
    assert backend.calls[3]["request"]["messages"][-1]["content"].startswith("Previous output")


# --- rejection and Strong repair -----------------------------------------------------------------


def test_correction_prompt_carries_rejected_step_and_verifier_reason() -> None:
    backend = FakeBackend(
        [
            completion(("bad", "write_file", {"path": "wrong"})),
            verdict(0, reason="step 1 writes to wrong; the task names right"),
            completion(("fix", "write_file", {"path": "right"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert decision.verdict.reason == "step 1 writes to wrong; the task names right"
    note = backend.calls[2]["request"]["messages"][-1]
    # A trailing user turn, never a trailing system message, so the transcript prefix stays
    # cacheable on the Strong side.
    assert note["role"] == "user"
    assert "inserted by the tool runtime, not written by the user" in note["content"]
    assert "reviewed before execution and was not run" in note["content"]
    assert "Check the finding against the conversation" in note["content"]
    for word in ("Strong", "Weak", "verifier", "speculative"):
        assert word not in note["content"]
    assert '{"tool": "write_file", "arguments": {"path": "wrong"}}' in note["content"]
    assert "Review finding: step 1 writes to wrong; the task names right" in note["content"]
    assert "Fix the draft rather than starting over" in note["content"]
    # The clean transcript comes first, unchanged; the rejected call is not in it.
    assert backend.calls[2]["request"]["messages"][:-1] == request()["messages"]
    assert backend.calls[2]["reasoning_effort"] == "high"
    assert backend.calls[2]["max_tokens"] == 16384
    # Consumed: the next correction must not reuse a stale diagnosis.
    assert gateway.state.rejected_step is None
    assert gateway.state.rejection_reason == ""
    assert gateway.state.discarded_after_rejected == 0


def test_rejected_mutation_is_replaced_by_one_strong_call() -> None:
    backend = FakeBackend(
        [
            completion(("bad", "write_file", {"path": "wrong"})),
            verdict(0),
            completion(("fix", "write_file", {"path": "right"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert tool_calls_of(decision.response)[0]["id"] == "fix"
    assert models(backend) == ["weak", "strong", "strong"]
    assert [call.phase for call in decision.model_calls] == [
        "weak_act",
        "strong_verify",
        "strong_correct",
    ]
    assert decision.candidate_steps == (
        {
            "index": 1,
            "actor": "weak",
            "action": {"tool": "write_file", "arguments": {"path": "wrong"}},
            "observation": None,
            "terminal": False,
            "pending": True,
            "reasoning": "reason",
        },
    )
    assert decision.correction_action == {"tool": "write_file", "arguments": {"path": "right"}}


def test_rejected_text_reply_is_replaced_by_strong_reply() -> None:
    backend = FakeBackend(
        [
            completion(content="Done, I cancelled everything."),
            verdict(0),
            completion(content="I still need your confirmation before cancelling."),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert decision.response["choices"][0]["message"]["content"].startswith("I still need")
    assert models(backend) == ["weak", "strong", "strong"]
    assert decision.corrector_generation_count == 1
    assert decision.correction_actions[0]["type"] == "message"
    assert len(gateway.state.confirmed) == 1
    assert gateway.state.confirmed[0].actor is ModelRole.STRONG
    assert gateway.state.confirmed[0].action["type"] == "message"


def test_correction_reasoning_is_kept_internally_but_never_delivered() -> None:
    backend = FakeBackend(
        [
            completion(("bad", "write_file", {"path": "wrong"})),
            verdict(0),
            completion(("fix", "write_file", {"path": "right"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())

    decision = gateway.handle(request())

    message = decision.response["choices"][0]["message"]
    assert message["tool_calls"][0]["id"] == "fix"
    assert "reasoning_content" not in message and "reasoning" not in message
    assert decision.response["usage"] is not None
    assert gateway.state.confirmed[-1].reasoning == "reason"


@pytest.mark.parametrize("with_tool", [False, True])
def test_repair_turn_commits_its_action_and_appends_the_qa_note(with_tool) -> None:
    response = (
        completion(("r1", "read_file", {"path": "evidence"}))
        if with_tool
        else completion(content="Final Answer: Ada")
    )
    backend = FakeBackend([response])
    gateway = SpeculativeToolGateway(backend, config(task_semantics="qa"))

    correction = gateway._correction(request())

    delivered = correction.response["choices"][0]["message"]
    assert "reasoning_content" not in delivered
    assert delivered["tool_calls"] == response["choices"][0]["message"]["tool_calls"]
    assert delivered["content"] == response["choices"][0]["message"]["content"]
    assert len(correction.actions) == 1
    assert correction.tool_call_count == (1 if with_tool else 0)
    assert correction.suppressed_reply is None
    assert len(gateway.state.confirmed) == 1
    note = backend.calls[0]["request"]["messages"][-1]["content"]
    assert note.endswith(QA_CORRECTION_SUFFIX)


def test_strong_sees_its_own_prior_correction_reasoning() -> None:
    backend = FakeBackend(
        [
            completion(("bad", "write_file", {"path": "wrong"})),
            verdict(0),
            completion(("fix", "write_file", {"path": "right"})),
            completion(("bad2", "write_file", {"path": "wrong2"})),
            verdict(0),
            completion(("fix2", "write_file", {"path": "right2"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())
    first = gateway.handle(request())
    assert "reasoning_content" not in first.response["choices"][0]["message"]
    # The client replays Strong's turn without reasoning.
    transcript = [
        {"role": "user", "content": "Do the task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "fix",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "fix", "content": "ok"},
    ]

    second = gateway.handle(request(messages=transcript))

    assert second.source == "strong_correction"
    weak_view = backend.calls[3]["request"]["messages"]
    assert "reasoning_content" not in weak_view[1]
    strong_view = backend.calls[5]["request"]["messages"]
    assert strong_view[1]["reasoning_content"] == "reason"
    assert strong_view[-1]["role"] == "user"
    assert "reasoning_content" not in second.response["choices"][0]["message"]


def test_corrector_can_drop_every_earlier_turns_reasoning() -> None:
    gateway = SpeculativeToolGateway(FakeBackend([]), config(correction_history_reasoning=False))
    gateway.state.confirmed.append(
        AgentStep(
            actor=ModelRole.STRONG,
            reasoning="strong's own correction rationale",
            action={"tool": "write_file", "arguments": {"path": "p"}},
            observation="ok",
            metadata={"pending": False, "tool_call_id": "call_strong"},
        )
    )
    messages = [
        {"role": "user", "content": "Do the task"},
        {
            "role": "assistant",
            "reasoning_content": "weak deriving the wrong number",
            "tool_calls": [{"id": "call_weak", "function": {"name": "read_file"}}],
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_strong", "function": {"name": "write_file"}}],
        },
    ]

    rendered = gateway._with_strong_reasoning(messages)

    assert "reasoning_content" not in rendered[1]
    assert "reasoning_content" not in rendered[2]

    # On by default: earlier reasoning is kept and Strong's own is re-attached by call id.
    gateway.config = dataclasses.replace(gateway.config, correction_history_reasoning=True)
    kept = gateway._with_strong_reasoning(messages)
    assert kept[1]["reasoning_content"] == "weak deriving the wrong number"
    assert kept[2]["reasoning_content"] == "strong's own correction rationale"


def test_rejection_of_an_executed_read_triggers_correction_now() -> None:
    backend = FakeBackend(
        [
            completion(("r1", "read_file", {"path": "a"})),
            completion(("w1", "write_file", {"path": "b"})),
            verdict(0),
            completion(("fix", "read_file", {"path": "c"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(horizon=4))
    gateway.handle(request())

    decision = gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "r1", "content": "file a"},
            ]
        )
    )

    assert decision.source == "strong_correction"
    assert decision.boundary == "mutation"
    assert decision.verifier_accepted_steps == 0
    assert decision.executed_rejected_steps == 1
    assert tool_calls_of(decision.response)[0]["id"] == "fix"
    # r1 already ran in the client and stays, flagged; w1 never reached the client.
    ids = [step.metadata.get("tool_call_id") for step in gateway.state.confirmed]
    assert ids == ["r1", "fix"]
    assert gateway.state.confirmed[0].metadata["verifier_rejected"] is True
    assert gateway.state.confirmed[0].reasoning is None
    assert gateway.state.candidate == []


def test_correction_prompt_marks_executed_rejected_read_and_discarded_suffix() -> None:
    backend = FakeBackend(
        [
            completion(("r1", "read_file", {"path": "a"})),
            completion(("r2", "read_file", {"path": "b"}), ("w1", "write_file", {"path": "c"})),
            verdict(0, reason="step 1 read the wrong file"),
            completion(("fix", "read_file", {"path": "z"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(horizon=4))
    gateway.handle(request())
    followup = request(
        messages=[
            {"role": "user", "content": "Do the task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "r1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "r1", "content": "contents of a"},
        ]
    )

    decision = gateway.handle(followup)

    assert decision.source == "strong_correction"
    assert decision.executed_rejected_steps == 1
    note = backend.calls[-1]["request"]["messages"][-1]["content"]
    assert "did run (its result is above)" in note
    assert '"observation": "contents of a"' in note
    assert "2 further draft step(s) that depended on it were dropped" in note


# --- fused replacement ---------------------------------------------------------------------------


def test_fused_verdict_replaces_rejected_mutation_without_another_strong_call() -> None:
    backend = FakeBackend(
        [
            completion(("bad", "write_file", {"path": "wrong"})),
            verdict(
                0,
                {"tool": "write_file", "arguments": {"path": "right", "content": "fixed"}},
                reason="step 1 targets the wrong file",
            ),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())

    decision = gateway.handle(request())

    assert decision.source == "strong_fused_correction"
    assert [call.phase for call in decision.model_calls] == ["weak_act", "strong_verify"]
    assert decision.corrector_generation_count == 0
    assert decision.correction_tool_call_count == 0
    (call,) = tool_calls_of(decision.response)
    assert call["function"]["name"] == "write_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "right", "content": "fixed"}
    expected = {"tool": "write_file", "arguments": {"path": "right", "content": "fixed"}}
    assert decision.correction_action == expected
    assert decision.correction_actions == (expected,)
    assert gateway.state.confirmed[-1].actor is ModelRole.STRONG
    assert gateway.state.confirmed[-1].metadata["tool_call_id"] == call["id"]


@pytest.mark.parametrize("draft", ["reply", "tool"])
@pytest.mark.parametrize("text", ["I still need the date.", "NO_REPLY"])
def test_message_shaped_replacement_is_discarded_for_strong_repair(draft, text) -> None:
    drafted = (
        completion(content="Done")
        if draft == "reply"
        else completion(("bad", "write_file", {"path": "wrong"}))
    )
    backend = FakeBackend(
        [drafted, verdict(0, {"type": "message", "content": text}), completion(content="Fixed.")]
    )
    gateway = SpeculativeToolGateway(backend, config())

    decision = gateway.handle(request())

    assert decision.source == "strong_correction"
    assert [call.phase for call in decision.model_calls] == [
        "weak_act",
        "strong_verify",
        "strong_correct",
    ]
    assert decision.verdict.format_retries == 0
    assert decision.verdict.replacement_action is None
    assert decision.verdict.replacement_validation_failure_kind == "message_replacement"
    assert decision.replacement_validation_failure_count == 1
    # The discarded action never reaches the repair prompt.
    assert text not in json.dumps(backend.calls[2]["request"]["messages"])
    assert decision.response["choices"][0]["message"]["content"] == "Fixed."


def _schema_request() -> dict:
    req = request()
    req["tools"][1]["function"]["parameters"] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    return req


@pytest.mark.parametrize(
    ("replacement", "kind"),
    [
        ({"tool": "unknown_tool", "arguments": {"path": "right"}}, "unknown_tool"),
        (
            {"tool": "write_file", "arguments": {"path": "right", "content": "x" * 5000}},
            "oversized_replacement",
        ),
        ({"tool": "write_file", "arguments": {"path": 7}}, "schema_invalid"),
        ({"tool": "write_file", "arguments": {"path": "wrong"}}, "replays_rejected"),
        ({"tool": "write_file", "arguments": {"path": "done"}}, "mutation_replay"),
    ],
)
def test_unusable_fused_replacement_falls_back_to_strong_correction(replacement, kind) -> None:
    backend = FakeBackend(
        [
            completion(("bad", "write_file", {"path": "wrong"})),
            verdict(0, replacement, reason="step 1 targets the wrong file"),
            completion(("fix", "write_file", {"path": "right"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())
    # An earlier write that already ran, for the replay guard.
    gateway.state.confirmed = [
        AgentStep(
            ModelRole.WEAK,
            {"tool": "write_file", "arguments": {"path": "done"}},
            "written",
            metadata={"pending": False, "tool_call_id": "old"},
        )
    ]

    decision = gateway.handle(_schema_request())

    assert decision.source == "strong_correction"
    assert [call.phase for call in decision.model_calls] == [
        "weak_act",
        "strong_verify",
        "strong_correct",
    ]
    assert decision.verdict.format_retries == 0
    assert decision.verdict.replacement_validation_failure_kind == kind
    assert decision.verdict.replacement_action is None
    assert decision.replacement_validation_failure_count == 1
    assert decision.corrector_generation_count == 1
    assert json.dumps(replacement["arguments"]) not in json.dumps(
        backend.calls[2]["request"]["messages"]
    )


@pytest.mark.parametrize(("draft", "fused"), [("reply", False), ("write", True)])
def test_fusion_restricted_to_mutations_skips_reply_boundaries(draft, fused) -> None:
    drafted = (
        completion(content="all done")
        if draft == "reply"
        else completion(("bad", "write_file", {"path": "wrong"}))
    )
    replacement = {"tool": "write_file", "arguments": {"path": "right"}}
    responses = [drafted, verdict(0, replacement, reason="the file is not written yet")]
    if not fused:
        responses.append(completion(("fix", "write_file", {"path": "ledger"})))
    backend = FakeBackend(responses)
    gateway = SpeculativeToolGateway(backend, config(fuse_correction_boundaries=("mutation",)))

    decision = gateway.handle(request())

    if fused:
        assert decision.source == "strong_fused_correction"
        assert models(backend) == ["weak", "strong"]
        assert [step.action["arguments"]["path"] for step in gateway.state.confirmed] == ["right"]
    else:
        # The verifier still emitted the replacement; the config chose not to commit it here.
        assert decision.source == "strong_correction"
        assert decision.verdict.replacement_action == replacement
        assert [step.action["arguments"]["path"] for step in gateway.state.confirmed] == [
            "ledger"
        ]


# --- partial parallel batches --------------------------------------------------------------------


def test_partial_batch_executes_read_then_defers_correction_until_observation() -> None:
    backend = FakeBackend(
        [
            completion(
                ("read", "read_file", {"path": "a"}),
                ("bad", "write_file", {"path": "wrong"}),
            ),
            verdict(1),
            completion(("fix", "write_file", {"path": "right"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())

    first = gateway.handle(request())

    assert first.source == "weak_partial_then_correction"
    message = first.response["choices"][0]["message"]
    assert [call["id"] for call in message["tool_calls"]] == ["read"]
    # The reasoning covered the withheld call too, so neither it nor the content is released.
    assert "reasoning_content" not in message
    assert message["content"] is None
    assert gateway.state.confirmed[0].reasoning is None
    assert [step["action"]["tool"] for step in first.candidate_steps] == [
        "read_file",
        "write_file",
    ]
    assert first.correction_action is None

    second = gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "read", "content": "file contents"},
            ]
        )
    )

    assert second.source == "strong_correction"
    assert models(backend) == ["weak", "strong", "strong"]
    assert second.correction_action == {"tool": "write_file", "arguments": {"path": "right"}}
    assert gateway.state.confirmed[0].observation == "file contents"
    assert gateway.state.deferred_correction is False


def test_fused_replacement_waits_when_partial_batch_must_execute_first() -> None:
    backend = FakeBackend(
        [
            completion(("r", "read_file", {"path": "a"}), ("bad", "write_file", {"path": "wrong"})),
            verdict(1, {"tool": "write_file", "arguments": {"path": "right"}}),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())

    decision = gateway.handle(request())

    assert decision.source == "weak_partial_then_correction"
    assert gateway.state.deferred_correction is True
    assert [c["id"] for c in tool_calls_of(decision.response)] == ["r"]


# --- repair turn without strict validation -------------------------------------------------------


def test_unknown_corrector_tool_is_delivered_without_strict_validation() -> None:
    backend = FakeBackend(
        [
            completion(("w0", "write_file", {"path": "x"})),
            verdict(0),
            completion(("f1", "no_such_tool", {"path": "a"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(strict_action_validation=False))

    decision = gateway.handle(request())

    assert [c["function"]["name"] for c in tool_calls_of(decision.response)] == ["no_such_tool"]
    assert gateway.correction_runtime_failure_count == 0


# --- Weak-side knobs -----------------------------------------------------------------------------


def test_weak_review_hint_echoes_cleared_rejection_once() -> None:
    backend = FakeBackend(
        [
            completion(("w1", "write_file", {"path": "a"})),
            verdict(0, reason="Write to a conflicts with committed state."),
            completion(("c1", "write_file", {"path": "b"})),
            completion(("w2", "read_file", {"path": "c"})),
            completion(("w3", "read_file", {"path": "d"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(weak_review_hint=True))

    first = gateway.handle(request())
    assert first.source == "strong_correction"
    assert gateway.state.review_hint

    second = gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "c1", "content": "wrote b"},
            ]
        )
    )
    last = backend.calls[3]["request"]["messages"][-1]
    assert backend.calls[3]["model"] == "weak"
    assert last["role"] == "user"
    assert "That batch was rejected" in last["content"]
    assert "Write to a conflicts" in last["content"]
    # One-shot, and never part of the delivered response.
    assert gateway.state.review_hint == ""
    assert "That batch was rejected" not in json.dumps(second.response)

    gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "c1", "content": "wrote b"},
                {"role": "tool", "tool_call_id": "w2", "content": "read c"},
            ]
        )
    )
    assert "That batch was rejected" not in json.dumps(backend.calls[4]["request"]["messages"])


def test_weak_review_hint_off_by_default() -> None:
    backend = FakeBackend(
        [
            completion(("w1", "write_file", {"path": "a"})),
            verdict(0, reason="nope"),
            completion(("c1", "write_file", {"path": "b"})),
            completion(("w2", "read_file", {"path": "c"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())
    gateway.handle(request())
    assert gateway.state.review_hint == ""

    gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "c1", "content": "wrote b"},
            ]
        )
    )

    assert "That batch was rejected" not in json.dumps(backend.calls[-1]["request"]["messages"])


def test_repeat_call_guard_withholds_all_repeat_batch() -> None:
    """A draft that only replays executed calls is withheld and repaired like a rejection."""
    backend = FakeBackend(
        [
            completion(("r1", "read_file", {"path": "a"})),
            completion(("r2", "read_file", {"path": "a"})),
            completion(("c1", "write_file", {"path": "done"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(repeat_call_guard=True, horizon=8))
    assert gateway.handle(request()).source == "weak_read"

    second = gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "r1", "content": "file a"},
            ]
        )
    )

    assert second.source == "strong_correction"
    assert second.boundary == "repeat"
    assert second.repeat_rejected == 1
    assert models(backend) == ["weak", "weak", "strong"]
    assert "repeat calls already executed" in backend.calls[2]["request"]["messages"][-1]["content"]
    # The repeat was never committed: only the delivered read and the correction remain.
    assert [step.action["tool"] for step in gateway.state.confirmed] == ["read_file", "write_file"]
    assert tool_calls_of(second.response)[0]["id"] == "c1"


def test_repeat_call_guard_allows_new_calls() -> None:
    backend = FakeBackend(
        [
            completion(("r1", "read_file", {"path": "a"})),
            completion(("r2", "read_file", {"path": "b"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(repeat_call_guard=True, horizon=8))
    gateway.handle(request())

    second = gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "r1", "content": "file a"},
            ]
        )
    )

    assert second.source == "weak_read"
    assert models(backend) == ["weak", "weak"]


def test_repeat_call_guard_off_by_default() -> None:
    backend = FakeBackend(
        [
            completion(("r1", "read_file", {"path": "a"})),
            completion(("r2", "read_file", {"path": "a"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(horizon=8))
    gateway.handle(request())

    second = gateway.handle(
        request(
            messages=[
                {"role": "user", "content": "Do the task"},
                {"role": "tool", "tool_call_id": "r1", "content": "file a"},
            ]
        )
    )

    assert second.source == "weak_read"
    assert models(backend) == ["weak", "weak"]


# --- request lifecycle ---------------------------------------------------------------------------


def test_buffered_completion_converts_to_sse() -> None:
    body = completion(("r1", "read_file", {"path": "a"}))

    stream = completion_to_sse(body).decode("utf-8")

    chunks = [json.loads(line[6:]) for line in stream.split("\n\n") if line.startswith("data: {")]
    delta = chunks[0]["choices"][0]["delta"]
    assert delta["role"] == "assistant"
    assert delta["tool_calls"][0]["index"] == 0
    assert delta["tool_calls"][0]["id"] == "r1"
    assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[1]["usage"] == body["usage"]
    assert stream.endswith("data: [DONE]\n\n")


def test_identical_provider_retry_reuses_decision_without_duplicate_model_call() -> None:
    backend = FakeBackend([completion(("r1", "read_file", {"path": "a"}))])
    gateway = SpeculativeToolGateway(backend, config())
    payload = request()

    first = gateway.handle(payload)
    second = gateway.handle(payload)

    assert first.source == "weak_read"
    assert second.source == "cached_replay"
    assert second.model_calls == ()
    assert second.response == first.response
    assert len(backend.calls) == 1
    assert len(gateway.state.candidate) == 1


def test_failed_verification_rolls_back_candidate_state_before_provider_retry() -> None:
    backend = FakeBackend(
        [
            completion(("r1", "read_file", {"path": "a"})),
            completion(("w1", "write_file", {"path": "a"})),
            RuntimeError("transient strong failure"),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config(horizon=4))
    gateway.handle(request())
    retry_request = request(
        messages=[
            {"role": "user", "content": "Do the task"},
            {"role": "tool", "tool_call_id": "r1", "content": "file contents"},
        ]
    )

    with pytest.raises(RuntimeError, match="transient strong failure"):
        gateway.handle(retry_request)

    assert len(gateway.state.candidate) == 1
    assert gateway.state.candidate[0].metadata["tool_call_id"] == "r1"
    assert gateway.state.candidate[0].observation is None
    # The aborted attempt's calls stay visible for accounting.
    assert [call.phase for call in gateway.model_calls] == ["weak_act", "strong_verify"]


def test_failed_delivery_rolls_back_and_does_not_cache_undelivered_action() -> None:
    backend = FakeBackend(
        [
            completion(("lost", "write_file", {"path": "a"})),
            verdict(1),
            completion(("retry", "read_file", {"path": "a"})),
        ]
    )
    gateway = SpeculativeToolGateway(backend, config())

    def fail_delivery(decision):
        raise BrokenPipeError("client left")

    with pytest.raises(BrokenPipeError):
        gateway.handle(request(), deliver=fail_delivery)
    assert not gateway.state.confirmed
    assert not gateway.state.candidate

    retry = gateway.handle(request())

    assert retry.source == "weak_read"
    assert tool_calls_of(retry.response)[0]["id"] == "retry"


def test_failed_cached_delivery_keeps_previously_delivered_state() -> None:
    backend = FakeBackend([completion(("r1", "read_file", {"path": "a"}))])
    gateway = SpeculativeToolGateway(backend, config())
    gateway.handle(request())

    def fail_delivery(decision):
        raise BrokenPipeError("retry disconnected")

    with pytest.raises(BrokenPipeError):
        gateway.handle(request(), deliver=fail_delivery)

    assert gateway.handle(request()).source == "cached_replay"
    assert len(gateway.state.candidate) == 1
    assert len(backend.calls) == 1


def test_disconnect_during_weak_call_prevents_paid_followup_phases() -> None:
    class DisconnectingBackend(FakeBackend):
        disconnected = False

        def complete(self, *args, **kwargs):
            response = super().complete(*args, **kwargs)
            self.disconnected = True
            return response

    backend = DisconnectingBackend([completion(("w1", "write_file", {"path": "a"}))])
    gateway = SpeculativeToolGateway(backend, config())

    with pytest.raises(ConnectionAbortedError):
        gateway.handle(request(), is_cancelled=lambda: backend.disconnected)

    assert len(backend.calls) == 1
    assert not gateway.state.confirmed
    assert not gateway.state.candidate


def test_cancelled_request_is_refused_before_any_model_call() -> None:
    backend = FakeBackend([completion(("r1", "read_file", {"path": "a"}))])
    gateway = SpeculativeToolGateway(backend, config())

    with pytest.raises(ConnectionAbortedError):
        gateway.handle(request(), is_cancelled=lambda: True)

    assert backend.calls == []
