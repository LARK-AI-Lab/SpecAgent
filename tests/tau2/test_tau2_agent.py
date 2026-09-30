"""Protocol-level tests for the tau2 agents.

Model clients are replaced by scripted queues. Most tests replace the orchestrator
with a fake execute callback, so they cover drafting, verification, commit/reject
bookkeeping, and the Weak prompt (`state.raw`) contract; the `test_real_orch_*`
tests run the real tau2 Orchestrator with the SpecAgent patch applied.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "benchmarks" / "tau2", REPO / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

pytest.importorskip("tau2")

import agent as tca  # noqa: E402
from reflective_prefix import PROTOCOL_NAME, validate_replacement_action  # noqa: E402
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage  # noqa: E402
from tau2.environment.tool import Tool  # noqa: E402
from tau2.environment.toolkit import MUTATES_STATE_ATTR  # noqa: E402

from specagent.protocol.types import ModelRole  # noqa: E402


def _measured_usage(inp, out, cached=0, reasoning=0):
    return {
        "_specagent_usage": {
            "input_tokens": inp,
            "output_tokens": out,
            "cached_input_tokens": cached,
            "reasoning_tokens": reasoning,
            "input_status": "measured",
            "output_status": "measured",
            "cached_status": "measured",
            "reasoning_status": "measured",
            "provenance": "openai_usage",
            "status": "measured",
        }
    }


class FakeToolCallResult:
    def __init__(self, content="", reasoning=None, tool_calls=None):
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls or []
        self.input_tokens = 10
        self.output_tokens = 5
        self.duration_seconds = 0.01
        self.cached_input_tokens = 0
        self.reasoning_tokens = 2
        self.raw_response = _measured_usage(10, 5, 0, 2)


class FakeCompletion:
    def __init__(self, content):
        self.content = content
        self.input_tokens = 20
        self.output_tokens = 8
        self.duration_seconds = 0.01
        self.cached_input_tokens = 0
        self.reasoning_tokens = 3
        self.raw_response = _measured_usage(20, 8, 0, 3)


def user_message(content: str) -> UserMessage:
    return UserMessage(role="user", content=content)


def tool_message(id_: str, content: str) -> ToolMessage:
    return ToolMessage(id=id_, role="tool", content=content, requestor="assistant", error=False)


def call(id_, name, **arguments):
    return {"id": id_, "name": name, "arguments": json.dumps(arguments)}


def tool_turn(id_, name, **arguments):
    return FakeToolCallResult(tool_calls=[call(id_, name, **arguments)])


def verdict(accepted, reason=None, replacement=None):
    return FakeCompletion(json.dumps({
        "reason": reason, "accepted_steps": accepted, "replacement_action": replacement,
    }))


class FakeOrchestrator:
    def __init__(self):
        self.trajectory = []
        self.executed = []


def make_tool(name: str, mutates: bool) -> Tool:
    def fn(reservation_id: str = "", user_id: str = "", summary: str = "") -> str:
        """Look up or change a reservation.

        Args:
            reservation_id: Reservation identifier.
            user_id: User identifier.
            summary: Transfer summary.
        """
        return "ok"

    fn.__name__ = name
    fn.__qualname__ = name
    setattr(fn, MUTATES_STATE_ATTR, mutates)
    return Tool(fn)


class Harness:
    def __init__(self, horizon=4, task_id="fixture-0"):
        tools = [
            make_tool("get_reservation_details", mutates=False),
            make_tool("get_user_details", mutates=False),
            make_tool("cancel_reservation", mutates=True),
            # tau2 marks this GENERIC (non-mutating); the agent still holds it
            # for review by name.
            make_tool("transfer_to_human_agents", mutates=False),
        ]
        self.agent = tca.SpecAgentTau2Agent(
            tools=tools,
            domain_policy="POLICY",
            task=SimpleNamespace(id=task_id) if task_id is not None else None,
            domain="airline",
            horizon=horizon,
            weak={"base_url": "http://x/v1", "model": "weak"},
            strong={"base_url": "http://x/v1", "model": "strong"},
        )
        self.orch = FakeOrchestrator()
        self.agent.gate.bind(execute=self._execute, orchestrator=self.orch)
        self.weak_queue: list = []
        self.verify_queue: list = []
        self.correct_queue: list = []
        self.verify_requests: list = []
        self.verify_messages: list = []
        self.verify_kwargs: list = []
        self.weak_requests: list = []
        self.correct_requests: list = []
        self.agent.weak_client.complete_with_tools = self._weak
        self.agent.strong_client.complete = self._verify
        self.agent.strong_client.complete_with_tools = self._correct
        self.state = self.agent.get_init_state([])

    def _execute(self, tool_calls):
        self.orch.executed.extend(tc.name for tc in tool_calls)
        return [
            tool_message(tc.id, f"ok:{tc.name}:{json.dumps(tc.arguments)}")
            for tc in tool_calls
        ]

    async def _weak(self, messages, tools, **kwargs):
        assert self.weak_queue, "weak queue exhausted"
        self.weak_requests.append(messages)
        return self.weak_queue.pop(0)

    async def _verify(self, messages, **kwargs):
        assert self.verify_queue, "verify queue exhausted"
        self.verify_requests.append(json.loads(messages[1]["content"]))
        self.verify_messages.append(messages)
        self.verify_kwargs.append(kwargs)
        return self.verify_queue.pop(0)

    async def _correct(self, messages, tools, **kwargs):
        assert self.correct_queue, "correct queue exhausted"
        self.correct_requests.append(messages)
        return self.correct_queue.pop(0)

    def turn(self, message):
        reply, self.state = self.agent.generate_next_message(message, self.state)
        return reply

    @property
    def raw(self):
        return self.state.raw

    @property
    def stats(self):
        return self.agent.gate.stats


def _phases(agent) -> list[str]:
    return [span.phase.value for span in agent.ledger.spans]


# -- reply / read boundaries ------------------------------------------------


def test_reply_is_sent_unreviewed_and_next_user_message_becomes_its_observation():
    h = Harness()
    h.weak_queue = [FakeToolCallResult(content="Which reservation?")]

    reply = h.turn(user_message("cancel my flight"))

    assert reply.content == "Which reservation?" and reply.tool_calls is None
    assert h.verify_requests == []
    assert h.stats["auto_accepted_chunks"] == 1
    step = h.state.session.history[-1]
    assert step.metadata == {"pending": False} and step.observation is None

    h.weak_queue = [
        tool_turn("c1", "get_reservation_details", reservation_id="R1"),
        tool_turn("c2", "cancel_reservation", reservation_id="R1"),
    ]
    h.verify_queue = [verdict(2)]
    h.turn(user_message("R1"))

    assert h.state.session.history[0].observation == "R1"
    payload = h.verify_requests[0]
    assert payload["task"]["instruction"] == "cancel my flight"
    assert payload["confirmed_history"][0]["observation"] == "R1"
    assert h.state.instruction == "cancel my flight"


def test_read_only_chunk_at_horizon_commits_unreviewed_and_keeps_drafting():
    h = Harness(horizon=2)
    h.weak_queue = [
        tool_turn("c1", "get_user_details", user_id="u1"),
        tool_turn("c2", "get_reservation_details", reservation_id="R1"),
        FakeToolCallResult(content="Here is your reservation."),
    ]

    reply = h.turn(user_message("show my reservation"))

    assert reply.tool_calls is None and reply.content == "Here is your reservation."
    assert h.verify_requests == []
    assert h.orch.executed == ["get_user_details", "get_reservation_details"]
    assert [m["tool_call_id"] for m in h.raw if m.get("role") == "tool"] == ["c1", "c2"]
    assert h.stats["horizon_continuations"] == 1
    assert h.stats["auto_accepted_chunks"] == 2
    assert len(h.state.session.history) == 3


def test_a_hallucinated_tool_name_is_released_as_a_read_not_staged():
    """Classification comes from the real Tool objects' mutates-state marker, so a
    name the agent does not have is not in the held set and goes out as a read --
    the environment's own error is then its observation."""
    h = Harness(horizon=2)
    h.weak_queue = [
        tool_turn("c1", "definitely_not_a_tool", x=1),
        FakeToolCallResult(content="Sorry, let me check that another way."),
    ]

    h.turn(user_message("cancel R1"))

    assert h.orch.executed == ["definitely_not_a_tool"]
    assert h.stats["staged"] == 0
    assert h.stats["reads_executed"] == 1


def test_endless_read_only_drafting_terminates_the_turn():
    """The orchestrator's max_steps cannot see inside one turn, so the chunk loop
    needs its own bound. Distinct reads never trip the repeat budget."""
    h = Harness(horizon=1)
    n = tca._MAX_CHUNKS_PER_TURN
    h.weak_queue = [
        tool_turn(f"c{i}", "get_user_details", user_id=f"u{i}")
        for i in range(n)
    ]

    reply = h.turn(user_message("who am i?"))

    assert reply.tool_calls is None, "must never hand an executed read to the orchestrator"
    assert h.stats["read_loop_exhausted"] == 1
    assert h.stats["read_budget_escalations"] == 0
    assert len(h.orch.executed) == n


def test_spliced_read_is_ordered_before_its_result_by_timestamp():
    """Orchestrator.get_trajectory() sorts by timestamp, so the announcing assistant
    message must be stamped before the ToolMessage it produces."""
    h = Harness(horizon=1)
    h.weak_queue = [
        tool_turn("c1", "get_user_details", user_id="u1"),
        FakeToolCallResult(content="Done."),
    ]

    h.turn(user_message("who am i?"))

    traj = h.orch.trajectory
    assert len(traj) == 2, traj
    announce, result = traj
    assert announce.tool_calls[0].id == result.id
    assert announce.timestamp <= result.timestamp
    assert sorted(traj, key=lambda m: m.timestamp) == [announce, result]


def test_parallel_read_calls_become_separate_steps_with_their_own_messages():
    h = Harness(horizon=4)
    h.weak_queue = [
        FakeToolCallResult(tool_calls=[
            call("c1", "get_user_details", user_id="u1"),
            call("c2", "get_reservation_details", reservation_id="R1"),
        ]),
        FakeToolCallResult(content="Found both."),
    ]

    h.turn(user_message("show me"))

    assistant_calls = [
        [tc["id"] for tc in m["tool_calls"]] for m in h.raw
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert assistant_calls == [["c1"], ["c2"]]
    assert h.orch.executed == ["get_user_details", "get_reservation_details"]
    # One Weak call produced two steps; its tokens are charged once.
    drafts = [s for s in h.agent.ledger.spans if s.phase.value == "draft"]
    assert len(drafts) == 2


# -- verification of staged calls --------------------------------------------


def test_accepted_staged_mutation_runs_once_and_is_handed_to_orchestrator():
    h = Harness()
    h.weak_queue = [
        tool_turn("c1", "get_reservation_details", reservation_id="R1"),
        tool_turn("c2", "cancel_reservation", reservation_id="R1"),
    ]
    h.verify_queue = [verdict(2)]

    reply = h.turn(user_message("cancel R1"))

    assert reply.tool_calls[0].name == "cancel_reservation"
    assert h.orch.executed == ["get_reservation_details", "cancel_reservation"]
    assert h.agent.gate.defers(reply) is True
    assert h.agent.gate.take_resolved(reply)[0].content.startswith("ok:cancel_reservation")
    assert h.state.session.history[-1].observation.startswith("ok:cancel_reservation")
    assert h.stats["committed"] == 1
    # The verifier rules on the whole chunk, executed read included.
    assert [s["action"]["tool"] for s in h.verify_requests[0]["proposed_trajectory"]] == [
        "get_reservation_details", "cancel_reservation",
    ]
    assert h.verify_requests[0]["proposed_trajectory"][1]["pending"] is True
    # Accounting contract used by stop(): derived fields must survive to_dict().
    summary = h.agent.ledger.summarize().to_dict()
    assert summary["n_strong_verify_calls"] == 1 and summary["n_weak_calls"] == 2
    assert summary["strong_reasoning_tokens"] == 3
    json.dumps(summary)


def test_verifier_runs_at_temperature_zero_without_thinking():
    h = Harness()
    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="R1")]
    h.verify_queue = [verdict(1)]
    h.turn(user_message("cancel R1"))
    assert h.verify_kwargs[0]["temperature"] == 0.0
    assert h.verify_kwargs[0]["thinking"] is False


def test_verifier_does_not_see_the_candidates_own_reasoning():
    h = Harness()
    h.weak_queue = [FakeToolCallResult(
        tool_calls=[call("c1", "cancel_reservation", reservation_id="R1")],
        reasoning="I am sure this is right",
    )]
    h.verify_queue = [verdict(1)]
    h.turn(user_message("cancel R1"))
    assert "I am sure this is right" not in json.dumps(h.verify_messages[0])
    pruned = h.verify_requests[0]["request_evidence_metadata"]["pruned_context"]
    assert pruned["candidate_reasoning_steps"] == 1


def test_transfer_is_held_for_review_even_though_tau2_marks_it_generic():
    h = Harness()
    h.weak_queue = [tool_turn("c1", "transfer_to_human_agents", summary="x")]
    h.verify_queue = [verdict(0, reason="the policy allows this cancellation")]
    h.correct_queue = [FakeToolCallResult(content="I can cancel this for you.")]

    reply = h.turn(user_message("cancel R1"))

    assert h.orch.executed == []
    assert h.verify_requests[0]["proposed_trajectory"][0]["pending"] is True
    assert reply.content == "I can cancel this for you."
    assert h.stats["correction_reply"] == 1
    assert h.stats["rejected_reads"] == 0


def test_full_accept_executes_mutation_once():
    h = Harness()
    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="ABC")]
    h.verify_queue = [verdict(1)]
    reply = h.turn(user_message("Please cancel ABC"))
    assert reply.tool_calls and reply.tool_calls[0].name == "cancel_reservation"
    assert h.orch.executed == ["cancel_reservation"]
    assert h.stats["committed"] == 1
    assert h.stats["verifier_replacements"] == 0
    assert h.agent.gate.take_resolved(reply)[0].id == "c1"
    assert h.correct_requests == []


def test_full_accept_ignores_stray_reason_and_replacement():
    h = Harness()
    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="ABC")]
    h.verify_queue = [verdict(
        1, reason="do not cancel",
        replacement={"tool": "get_reservation_details", "arguments": {"reservation_id": "ABC"}},
    )]
    reply = h.turn(user_message("Please cancel ABC"))
    assert reply.tool_calls[0].arguments["reservation_id"] == "ABC"
    assert h.orch.executed == ["cancel_reservation"]
    assert h.stats["committed"] == 1
    assert h.stats["verifier_replacements"] == 0
    assert h.stats["full_accept_contract_violation"] == 1
    assert h.stats["verifier_format_errors"] == 0
    assert h.correct_requests == []
    assert "correction" not in _phases(h.agent)


# -- rejection: replacement or repair turn -----------------------------------


def test_replacement_is_single_validated_tool_not_a_strong_repair():
    h = Harness()
    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="WRONG")]
    h.verify_queue = [verdict(
        0, reason="wrong reservation id",
        replacement={"tool": "cancel_reservation", "arguments": {"reservation_id": "ABC"}},
    )]
    reply = h.turn(user_message("Please cancel ABC"))
    assert h.correct_requests == [], "replacement must not call the Strong repair turn"
    assert h.stats["discarded"] == 1
    assert h.stats["committed"] == 0
    assert h.stats["verifier_replacements"] == 1
    assert h.stats["corrections_executed"] == 0
    assert h.stats["corrector_strong"] == 0
    assert h.stats["reject_other"] == 1
    assert h.orch.executed == ["cancel_reservation"]
    assert reply.tool_calls[0].arguments["reservation_id"] == "ABC"
    resolved = h.agent.gate.take_resolved(reply)
    assert [r.id for r in resolved] == [reply.tool_calls[0].id]
    assert "correction" not in _phases(h.agent)
    assert "verification" in _phases(h.agent)
    assert h.state.session.history[-1].actor is ModelRole.STRONG


def test_null_replacement_calls_repair_once_without_the_rejected_draft():
    h = Harness()
    h.weak_queue = [FakeToolCallResult(
        tool_calls=[call("c1", "cancel_reservation", reservation_id="WRONG")],
        reasoning="I will cancel WRONG because the user said so",
    )]
    h.verify_queue = [verdict(0, reason="look the reservation up first")]
    h.correct_queue = [
        tool_turn("c2", "get_reservation_details", reservation_id="ABC")
    ]
    reply = h.turn(user_message("Please cancel ABC"))
    assert len(h.correct_requests) == 1
    assert h.stats["corrector_strong"] == 1
    assert h.stats["verifier_replacements"] == 0
    blob = json.dumps(h.correct_requests[0])
    assert "look the reservation up first" in blob
    assert "reasoning_content" not in blob
    assert "I will cancel WRONG" not in blob
    assert "cancel_reservation" not in blob
    assert reply.tool_calls[0].name == "get_reservation_details"
    assert h.stats["correction_read"] == 1
    assert h.agent.gate.take_resolved(reply)[0].id == "c2"
    assert "correction" in _phases(h.agent)


def test_guard_failure_drops_action_and_calls_repair_without_the_payload():
    h = Harness()
    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="WRONG")]
    h.verify_queue = [verdict(
        0, reason="need a lookup",
        replacement={"tool": "no_such_tool", "arguments": {"reservation_id": "ABC"}},
    )]
    h.correct_queue = [FakeToolCallResult(content="I will look that up.")]
    reply = h.turn(user_message("Please cancel ABC"))
    assert h.stats["replacement_guard_failed"] == 1
    assert h.stats["verifier_replacements"] == 0
    assert h.orch.executed == []
    assert "no_such_tool" not in json.dumps(h.correct_requests[0])
    assert reply.content == "I will look that up."


def test_partial_prefix_strips_rejected_suffix_and_does_not_replay_executed_read():
    h = Harness()
    h.weak_queue = [
        tool_turn("r1", "get_reservation_details", reservation_id="ABC"),
        tool_turn("c1", "cancel_reservation", reservation_id="ABC"),
    ]
    h.verify_queue = [verdict(1, reason="do not cancel yet")]
    h.correct_queue = [FakeToolCallResult(content="I found the reservation.")]
    h.turn(user_message("Please cancel ABC"))
    assert h.orch.executed == ["get_reservation_details"]
    assert h.stats["discarded"] == 1
    assert h.stats["rejected_reads"] == 0
    assert [m["tool_call_id"] for m in h.raw if m.get("role") == "tool"] == ["r1"]
    assert not any(
        (m.get("tool_calls") or [{}])[0].get("id") == "c1"
        for m in h.raw if m.get("role") == "assistant"
    )
    actors = [step.actor.value for step in h.state.session.history]
    assert actors == ["weak", "strong"]


def test_the_verifier_can_decline_to_commit_an_executed_read():
    h = Harness()
    h.weak_queue = [
        tool_turn("c1", "get_user_details", user_id="u1"),
        tool_turn("c2", "get_reservation_details", reservation_id="R1"),
        tool_turn("c3", "cancel_reservation", reservation_id="R1"),
    ]
    h.verify_queue = [verdict(1, reason="that is the wrong reservation")]
    h.correct_queue = [FakeToolCallResult(content="Which reservation did you mean?")]

    reply = h.turn(user_message("cancel R1"))

    assert reply.content == "Which reservation did you mean?"
    assert h.stats["rejected_reads"] == 1
    assert [m["tool_call_id"] for m in h.raw if m.get("role") == "tool"] == ["c1"]
    assert [step.actor.value for step in h.state.session.history] == ["weak", "strong"]


def test_parallel_repair_calls_are_all_executed_and_resolved_once():
    h = Harness()
    h.weak_queue = [tool_turn("bad", "cancel_reservation", reservation_id="X")]
    h.verify_queue = [verdict(0, reason="inspect both customer records")]
    h.correct_queue = [FakeToolCallResult(tool_calls=[
        call("r1", "get_reservation_details", reservation_id="ABC"),
        call("r2", "get_user_details", user_id="u1"),
    ])]

    repair = h.turn(user_message("Cancel ABC"))

    assert [tc.id for tc in repair.tool_calls] == ["r1", "r2"]
    assert [tm.id for tm in h.agent.gate.take_resolved(repair)] == ["r1", "r2"]
    assert h.orch.executed == ["get_reservation_details", "get_user_details"]
    assert list(h.state.session.history[-1].action) == ["parallel"]


def test_verifier_format_failure_keeps_executed_reads_and_withholds_only_the_pending_step():
    h = Harness()
    h.weak_queue = [
        tool_turn("c1", "get_reservation_details", reservation_id="R1"),
        tool_turn("c2", "cancel_reservation", reservation_id="R1"),
    ]
    h.verify_queue = [FakeCompletion("not json"), FakeCompletion("still not json")]
    h.correct_queue = [tool_turn("c3", "cancel_reservation", reservation_id="R1")]

    reply = h.turn(user_message("cancel R1"))

    assert h.stats["verifier_format_errors"] == 1
    assert h.stats["rejected_reads"] == 0, "the executed read must survive the fallback"
    assert h.stats["discarded"] == 1, "the staged call is what gets withheld"
    assert h.stats.get("reject_other", 0) == 0, "a format failure is not a semantic rejection"
    assert [m["tool_call_id"] for m in h.raw if m.get("role") == "tool"] == ["c1"]
    actors = [(s.actor.value, s.action.get("tool")) for s in h.state.session.history]
    assert actors == [("weak", "get_reservation_details"), ("strong", "cancel_reservation")]
    assert reply.tool_calls[0].id == "c3"
    # The retry keeps the payload and restates the output contract.
    first, second = h.verify_messages
    assert second[1] == first[1]
    assert second[0]["content"].startswith(first[0]["content"])
    assert "not valid JSON" in second[0]["content"]
    assert "could not produce a verdict" in h.correct_requests[0][-1]["content"]


def test_replacement_schema_guard_uses_tool_json_schema():
    tool = make_tool("get_reservation_details", mutates=False)
    ok, err = validate_replacement_action(
        {"tool": "get_reservation_details", "arguments": {"reservation_id": "ABC"}}, [tool],
    )
    assert err is None and ok is not None
    bad, err = validate_replacement_action(
        {"tool": "get_reservation_details", "arguments": {"reservation_id": 7}}, [tool],
    )
    assert bad is None and err
    unknown, err = validate_replacement_action(
        {"tool": "no_such_tool", "arguments": {"reservation_id": "ABC"}}, [tool],
    )
    assert unknown is None and err


# -- repeat budget on unreviewed chunks --------------------------------------


def test_repeat_budget_escalates_repeated_read_chunks_to_strong():
    h = Harness(horizon=1)
    # The same lookup three times: Weak is stuck, not searching.
    h.weak_queue = [
        tool_turn(f"c{i}", "get_user_details", user_id="u1")
        for i in range(1, 4)
    ]
    h.verify_queue = [verdict(0, reason="u1 was already looked up twice.")]
    h.correct_queue = [FakeToolCallResult(content="Your reservation is R1.")]

    reply = h.turn(user_message("show my reservation"))

    assert reply.content == "Your reservation is R1."
    # Exhausting the budget *asks* Strong; the third read is only withheld
    # because Strong rejected it.
    assert len(h.verify_requests) == 1
    assert h.stats["auto_accepted_chunks"] == 2
    assert h.stats["read_budget_escalations"] == 1
    assert h.orch.executed == ["get_user_details"] * 3
    assert [m["tool_call_id"] for m in h.raw if m.get("role") == "tool"] == ["c1", "c2"]
    assert "u1 was already looked up twice." in h.correct_requests[0][-1]["content"]


def test_repeat_budget_escalation_commits_the_chunk_when_strong_accepts_it():
    h = Harness(horizon=1)
    h.weak_queue = [
        *[tool_turn(f"c{i}", "get_user_details", user_id="u1")
          for i in range(1, 4)],
        FakeToolCallResult(content="Your reservation is R1."),
    ]
    h.verify_queue = [verdict(1)]

    reply = h.turn(user_message("show my reservation"))

    assert reply.content == "Your reservation is R1."
    assert h.stats["read_budget_escalations"] == 1
    assert [m["tool_call_id"] for m in h.raw if m.get("role") == "tool"] == ["c1", "c2", "c3"]
    assert h.correct_requests == []
    assert h.state.unverified_chunks == 0


def test_repeat_budget_counts_unreviewed_replies_across_turns():
    h = Harness()
    for _ in range(2):
        h.weak_queue = [FakeToolCallResult(content="I already transferred you.")]
        reply = h.turn(user_message("nothing happened, please cancel it"))
        assert reply.content == "I already transferred you."
        assert h.verify_requests == []
    assert h.stats["auto_accepted_chunks"] == 2
    assert h.stats["read_budget_escalations"] == 0

    h.weak_queue = [FakeToolCallResult(content="I already transferred you.")]
    h.verify_queue = [verdict(0, reason="Nothing was transferred.")]
    h.correct_queue = [
        tool_turn("c1", "cancel_reservation", reservation_id="R1")
    ]
    reply = h.turn(user_message("nothing happened, please cancel it"))

    assert h.stats["read_budget_escalations"] == 1
    assert h.verify_requests[0]["proposed_trajectory"][0]["pending"] is True
    assert reply.tool_calls[0].name == "cancel_reservation"


def test_verified_commit_resets_the_unreviewed_run():
    h = Harness()
    for _ in range(2):
        h.weak_queue = [FakeToolCallResult(content="Let me check that.")]
        h.turn(user_message("cancel my flight"))
    assert h.state.unverified_chunks == 1

    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="R1")]
    h.verify_queue = [verdict(1)]
    h.turn(user_message("R1"))

    assert h.state.unverified_chunks == 0
    assert h.stats["read_budget_escalations"] == 0


def test_repeat_budget_ignores_unreviewed_chunks_that_do_not_repeat():
    h = Harness()
    for text in ("Which reservation is it?", "Thanks -- one moment while I look.",
                 "That booking is basic economy.", "Shall I check the fare rules?"):
        h.weak_queue = [FakeToolCallResult(content=text)]
        assert h.turn(user_message("cancel my flight")).content == text
    assert h.verify_requests == []
    assert h.stats["read_budget_escalations"] == 0
    assert h.stats["auto_accepted_chunks"] == 4
    assert h.state.unverified_chunks == 0


def test_repeat_budget_treats_a_reworded_repeat_as_a_repeat():
    h = Harness()
    for text in ("I have already transferred you to a human agent.",
                 "You have already been transferred to a human agent."):
        h.weak_queue = [FakeToolCallResult(content=text)]
        h.turn(user_message("nothing happened"))
    assert h.verify_requests == []

    h.weak_queue = [
        FakeToolCallResult(content="A human agent has already been transferred to you.")
    ]
    h.verify_queue = [verdict(0, reason="Nothing was transferred.")]
    h.correct_queue = [FakeToolCallResult(content="Let me cancel it myself.")]
    reply = h.turn(user_message("nothing happened"))

    assert h.stats["read_budget_escalations"] == 1
    assert reply.content == "Let me cancel it myself."


# -- resilience and accounting -----------------------------------------------


def test_transient_upstream_error_is_retried_instead_of_failing_the_task(monkeypatch):
    import openai

    h = Harness()
    monkeypatch.setattr(tca, "_RETRY_BASE_DELAY", 0.0)
    attempts = {"n": 0}

    async def flaky(messages, tools, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise openai.APIConnectionError(request=SimpleNamespace(method="POST", url="http://x/v1"))
        return FakeToolCallResult(content="Recovered.")

    h.agent.weak_client.complete_with_tools = flaky
    reply = h.turn(user_message("hello"))

    assert reply.content == "Recovered."
    assert attempts["n"] == 2
    assert h.stats["transport_retries"] == 1
    drafts = [s for s in h.agent.ledger.spans if s.phase.value == "draft"]
    failed = [s for s in drafts if s.status == "failed"]
    ok = [s for s in drafts if s.status == "ok"]
    assert len(failed) == 1 and ok
    assert failed[0].usage_status.value == "unknown"
    assert failed[0].input_tokens == 0
    assert failed[0].flops.value is None
    assert failed[0].request_id == ok[0].request_id
    assert failed[0].transport_attempt_index == 1
    assert ok[0].transport_attempt_index == 2


def test_weak_tokens_are_recorded_when_verification_never_completes(monkeypatch):
    import openai

    h = Harness()
    monkeypatch.setattr(tca, "_RETRY_BASE_DELAY", 0.0)
    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="R1")]

    async def dead(messages, **kwargs):
        raise openai.APIConnectionError(request=SimpleNamespace(method="POST", url="http://x/v1"))

    h.agent.strong_client.complete = dead
    with pytest.raises(RuntimeError, match="strong verification call"):
        h.turn(user_message("cancel R1"))

    summary = h.agent.ledger.summarize().to_dict()
    assert summary["weak_output_tokens"] == 5
    assert summary["weak_discarded_output_tokens"] == 5
    assert summary["weak_committed_output_tokens"] == 0
    assert h.agent.gate._pending is None
    assert h.orch.executed == []


@pytest.mark.parametrize(
    ("error", "status"),
    [(RuntimeError("hard-fail"), "failed"), (asyncio.CancelledError(), "cancelled")],
)
def test_non_transient_model_error_records_one_span(error, status):
    h = Harness()

    async def boom(*args, **kwargs):
        raise error

    h.agent.weak_client.complete_with_tools = boom
    with pytest.raises(type(error)):
        h.turn(user_message("hello"))

    assert len(h.agent.ledger.spans) == 1
    span = h.agent.ledger.spans[0]
    assert span.status == status
    assert span.usage_status.value == "unknown"
    assert span.phase.value == "draft"


def test_tau2_spans_keep_flops_hbm_unknown_and_carry_identity():
    from specagent.protocol.accounting import LEGACY_CALL_KEYS

    h = Harness(task_id="7")
    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="R1")]
    h.verify_queue = [verdict(1)]
    h.turn(user_message("hello"))

    payload = h.agent.ledger.summarize().to_dict()
    assert payload["calls"]
    assert set(payload["calls"][0]) == set(LEGACY_CALL_KEYS)
    for key in ("flops", "hbm_read_bytes", "hbm_write_bytes"):
        assert payload[key]["value"] is None
        assert payload[key]["status"] == "unknown"
    spans = h.agent.ledger.spans
    assert {span.role for span in spans} == {ModelRole.WEAK, ModelRole.STRONG}
    assert all(span.parent_task_id == "7" for span in spans)
    assert h.agent._attempt_id
    assert all(span.parent_episode_id == h.agent._attempt_id for span in spans)
    assert all(span.request_id for span in spans)
    assert all(span.flops.provenance == "tau2_no_server_counter" for span in spans)


def test_stop_writes_the_resource_record(tmp_path, monkeypatch):
    from specagent.evals import trial_shard_io

    monkeypatch.setenv(trial_shard_io.ENV_LOCAL, str(tmp_path / "local"))
    monkeypatch.setenv(trial_shard_io.ENV_ARCHIVE, str(tmp_path / "archive"))
    monkeypatch.setenv("SPECAGENT_STATS", str(tmp_path / "stats.jsonl"))
    h = Harness(task_id="t1")
    h.weak_queue = [tool_turn("c1", "cancel_reservation", reservation_id="R1")]
    h.verify_queue = [verdict(1)]
    h.turn(user_message("cancel R1"))
    h.agent.stop(state=h.state)

    (record,) = trial_shard_io.collect_kind("stats")
    assert record["task_id"] == "t1"
    assert record["trial_id"] == h.agent._attempt_id
    assert record["protocol"] == {"name": PROTOCOL_NAME, "horizon": 4}
    assert record["gate"]["committed"] == 1
    assert record["resources"]["n_strong_verify_calls"] == 1
    assert "paper_resources" in record
    assert record["overhead_seconds"] == pytest.approx(
        record["wall_clock_seconds"] - record["llm_call_seconds"] - record["tool_seconds"]
    )


# -- single-model baseline ---------------------------------------------------


def _baseline(role="weak"):
    from baseline_agent import BaselineTau2Agent

    tools = [make_tool("get_reservation_details", mutates=False)]
    agent = BaselineTau2Agent(
        tools=tools, domain_policy="POLICY", task=SimpleNamespace(id="b1"),
        role=role, base_url="http://x/v1", model=role,
    )
    agent.get_init_state([])
    return agent


def test_weak_sees_the_baseline_system_prompt():
    """A SpecAgent-vs-baseline delta must be the protocol, not a prompt difference."""
    h = Harness()
    assert h.agent.system_prompt == _baseline().system_prompt


def test_baseline_returns_every_tool_call_and_charges_one_act_span():
    agent = _baseline(role="strong")
    requests = []

    async def complete(messages, tools, **kwargs):
        requests.append(messages)
        return FakeToolCallResult(tool_calls=[
            call("c1", "get_reservation_details", reservation_id="A"),
            call("c2", "get_reservation_details", reservation_id="B"),
        ])

    agent.client.complete_with_tools = complete
    state = agent.get_init_state([])
    reply, state = agent.generate_next_message(user_message("show A and B"), state)

    assert [tc.id for tc in reply.tool_calls] == ["c1", "c2"]
    assert requests[0][0] == {"role": "system", "content": agent.system_prompt}
    (span,) = agent.ledger.spans
    assert span.role is ModelRole.STRONG
    assert span.phase.value == "act"
    assert span.committed_output_tokens == 5
    assert span.parent_task_id == "b1"


# -- real tau2 Orchestrator + Environment (patched take_resolved) ------------


def _require_real_orch():
    from tau2.orchestrator.orchestrator import Orchestrator

    if "take_resolved" not in inspect.getsource(Orchestrator.step):
        pytest.skip("Orchestrator.step is not patched with take_resolved")


def _build_env_and_tools():
    from tau2.environment.environment import Environment
    from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

    class CountingTools(ToolKitBase):
        def __init__(self):
            super().__init__(db=None)
            self.executions: list[tuple[str, dict]] = []

        @is_tool(ToolType.READ)
        def get_reservation_details(self, reservation_id: str) -> str:
            """Look up a reservation.

            Args:
                reservation_id: Reservation identifier.
            """
            self.executions.append(("get_reservation_details", {"reservation_id": reservation_id}))
            return f"reservation {reservation_id}: confirmed"

        @is_tool(ToolType.WRITE)
        def cancel_reservation(self, reservation_id: str) -> str:
            """Cancel a reservation.

            Args:
                reservation_id: Reservation identifier.
            """
            self.executions.append(("cancel_reservation", {"reservation_id": reservation_id}))
            return f"cancelled {reservation_id}"

    tools = CountingTools()
    env = Environment(domain_name="fixture", policy="Be careful with cancellations.", tools=tools)
    return env, tools


class ScriptedUser:
    def get_init_state(self, message_history=None):
        from tau2.user.user_simulator import UserState
        return UserState(messages=[], system_messages=[])

    def set_seed(self, seed):
        return None

    def stop(self, *args, **kwargs):
        return None

    def generate_next_message(self, message, state):
        from tau2.user.user_simulator import STOP
        return UserMessage(role="user", content=STOP), state


def _run_real(weak, verify, correct):
    from tau2.data_model.tasks import InitialState, Task, UserScenario
    from tau2.orchestrator.orchestrator import Orchestrator

    env, tools = _build_env_and_tools()
    task = Task(
        id="orch-0",
        user_scenario=UserScenario(instructions="Cancel reservation ABC if eligible."),
        initial_state=InitialState(
            message_history=[UserMessage(role="user", content="Please cancel reservation ABC")]
        ),
    )
    agent = tca.SpecAgentTau2Agent(
        tools=env.get_tools(), domain_policy=env.get_policy(), task=task, domain="airline",
        weak={"base_url": "http://x/v1", "model": "weak"},
        strong={"base_url": "http://x/v1", "model": "strong"},
    )
    queues = {"weak": list(weak), "verify": list(verify), "correct": list(correct)}
    correct_requests = []

    async def _weak(messages, tools, **kwargs):
        return queues["weak"].pop(0)

    async def _verify(messages, **kwargs):
        return queues["verify"].pop(0)

    async def _correct(messages, tools, **kwargs):
        correct_requests.append(messages)
        return queues["correct"].pop(0)

    agent.weak_client.complete_with_tools = _weak
    agent.strong_client.complete = _verify
    agent.strong_client.complete_with_tools = _correct
    result = Orchestrator(
        domain="airline", agent=agent, user=ScriptedUser(), environment=env, task=task,
        max_steps=8, max_errors=5, seed=None, solo_mode=False,
    ).run()
    trajectory = list(getattr(result, "messages", None) or result.trajectory)
    assert not any(queues.values()), queues
    return agent, tools, trajectory, correct_requests


def _tool_pairs(trajectory) -> list[str]:
    calls: dict[str, str] = {}
    results: set[str] = set()
    for msg in trajectory:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                calls[tc.id] = tc.name
        elif isinstance(msg, ToolMessage):
            results.add(msg.id)
    assert set(calls) == results, f"assistant tool_calls {set(calls)} != tool messages {results}"
    return list(calls.values())


def test_real_orch_replacement_mutation_pairs_once_and_is_not_a_repair():
    _require_real_orch()
    agent, tools, trajectory, correct_requests = _run_real(
        weak=[
            tool_turn("c1", "cancel_reservation", reservation_id="WRONG"),
            FakeToolCallResult(content="Cancelled ABC."),
        ],
        verify=[verdict(
            0, reason="wrong id",
            replacement={"tool": "cancel_reservation", "arguments": {"reservation_id": "ABC"}},
        )],
        correct=[],
    )
    assert tools.executions == [("cancel_reservation", {"reservation_id": "ABC"})]
    assert agent.gate.stats["verifier_replacements"] == 1
    assert agent.gate.stats["corrections_executed"] == 0
    assert agent.gate.stats["discarded"] == 1
    assert correct_requests == []
    assert _tool_pairs(trajectory).count("cancel_reservation") == 1
    assert "correction" not in _phases(agent)


def test_real_orch_executed_read_is_not_replayed_and_unexecuted_mutation_is_dropped():
    _require_real_orch()
    agent, tools, trajectory, _ = _run_real(
        weak=[
            tool_turn("r1", "get_reservation_details", reservation_id="ABC"),
            tool_turn("c1", "cancel_reservation", reservation_id="ABC"),
        ],
        verify=[verdict(1, reason="do not mutate yet")],
        correct=[FakeToolCallResult(content="I will not cancel yet.")],
    )
    assert tools.executions == [("get_reservation_details", {"reservation_id": "ABC"})]
    assert agent.gate.stats["reads_executed"] == 1
    assert agent.gate.stats["discarded"] == 1
    assert agent.gate.stats["corrector_strong"] == 1
    assert _tool_pairs(trajectory) == ["get_reservation_details"]


def test_real_orch_null_replacement_repair_once_and_spans():
    _require_real_orch()
    agent, tools, trajectory, correct_requests = _run_real(
        weak=[
            FakeToolCallResult(
                tool_calls=[call("c1", "cancel_reservation", reservation_id="WRONG")],
                reasoning="secret weak chain of thought",
            ),
            FakeToolCallResult(content="Looked it up."),
        ],
        verify=[verdict(0, reason="lookup first")],
        correct=[FakeToolCallResult(
            tool_calls=[call("r2", "get_reservation_details", reservation_id="ABC")]
        )],
    )
    assert len(correct_requests) == 1
    blob = json.dumps(correct_requests[0])
    assert "lookup first" in blob
    assert "secret weak chain of thought" not in blob
    assert "WRONG" not in blob
    assert tools.executions == [("get_reservation_details", {"reservation_id": "ABC"})]
    assert agent.gate.stats["corrector_strong"] == 1
    assert _tool_pairs(trajectory) == ["get_reservation_details"]
    assert {"draft", "verification", "correction"} <= set(_phases(agent))
