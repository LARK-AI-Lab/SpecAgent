"""Scripted tau2 episodes that pin the tau2 agent's exact model traffic under the paper settings.

Runs the real tau2 Orchestrator (with the SpecAgent patch applied) on a two-tool fixture
domain. Every Weak, verifier, and corrector request, every tool execution, and the final
trajectory are recorded with deterministic ids.
"""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
from typing import Any
from unittest import mock

PAPER_KWARGS: dict[str, Any] = {
    "horizon": 6,
}


def _usage():
    return {"_specagent_usage": {
        "input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 0, "reasoning_tokens": 2,
        "input_status": "measured", "output_status": "measured", "cached_status": "measured",
        "reasoning_status": "measured", "provenance": "openai_usage", "status": "measured"}}


class ToolTurn:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.reasoning = "private reasoning"
        self.tool_calls = tool_calls or []
        self.input_tokens, self.output_tokens = 10, 5
        self.duration_seconds = 0.01
        self.cached_input_tokens, self.reasoning_tokens = 0, 2
        self.raw_response = _usage()


class TextTurn:
    def __init__(self, content):
        self.content = content
        self.input_tokens, self.output_tokens = 20, 8
        self.duration_seconds = 0.01
        self.cached_input_tokens, self.reasoning_tokens = 0, 0
        self.raw_response = _usage()


def call(call_id, name, **arguments):
    return {"id": call_id, "name": name, "arguments": json.dumps(arguments)}


def verdict(accepted, reason="bad step", replacement=None):
    return TextTurn(json.dumps({"reason": reason, "accepted_steps": accepted,
                                "replacement_action": replacement}))


def _env():
    from tau2.environment.environment import Environment
    from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

    class Tools(ToolKitBase):
        def __init__(self):
            super().__init__(db=None)
            self.executions: list[list] = []

        @is_tool(ToolType.READ)
        def get_order_details(self, order_id: str) -> str:
            """Look up an order.

            Args:
                order_id: Order identifier.
            """
            self.executions.append(["get_order_details", order_id])
            return f"order {order_id}: delivered"

        @is_tool(ToolType.WRITE)
        def cancel_order(self, order_id: str) -> str:
            """Cancel an order.

            Args:
                order_id: Order identifier.
            """
            self.executions.append(["cancel_order", order_id])
            return f"cancelled {order_id}"

    tools = Tools()
    return Environment(domain_name="fixture", policy="Confirm before cancelling.",
                       tools=tools), tools


class User:
    def __init__(self, replies):
        self.replies = list(replies)

    def get_init_state(self, message_history=None):
        from tau2.user.user_simulator import UserState
        return UserState(messages=[], system_messages=[])

    def set_seed(self, seed):
        return None

    def stop(self, *args, **kwargs):
        return None

    def generate_next_message(self, message, state):
        from tau2.data_model.message import UserMessage
        from tau2.user.user_simulator import STOP
        content = self.replies.pop(0) if self.replies else STOP
        return UserMessage(role="user", content=content), state


def _dump(value):
    def default(obj):
        return getattr(obj, "model_dump", lambda: str(obj))()

    return json.loads(json.dumps(value, default=default))


def run(agent_module, weak, verify, correct, replies) -> dict:
    from tau2.data_model.message import UserMessage
    from tau2.data_model.tasks import InitialState, Task, UserScenario
    from tau2.orchestrator.orchestrator import Orchestrator

    env, tools = _env()
    task = Task(id="fixture-0", user_scenario=UserScenario(instructions="Cancel order A1."),
                initial_state=InitialState(
                    message_history=[UserMessage(role="user", content="Please cancel order A1.")]))
    counter = itertools.count()
    with mock.patch("uuid.uuid4", lambda: SimpleNamespace(hex=f"{next(counter):032x}")):
        agent = agent_module.SpecAgentTau2Agent(
            tools=env.get_tools(), domain_policy=env.get_policy(), task=task, domain="retail",
            weak={"base_url": "http://weak/v1", "model": "weak"},
            strong={"base_url": "http://strong/v1", "model": "strong"}, **PAPER_KWARGS)
        record: dict[str, list] = {"weak": [], "verify": [], "correct": []}
        queues = {"weak": list(weak), "verify": list(verify), "correct": list(correct)}

        def make(role):
            async def _call(messages, tools=None, **kwargs):
                record[role].append({"messages": _dump(messages),
                                     "tools": [t["function"]["name"] if isinstance(t, dict)
                                               else getattr(t, "name", str(t))
                                               for t in (tools or [])],
                                     "kwargs": _dump(kwargs)})
                return queues[role].pop(0)
            return _call

        agent.weak_client.complete_with_tools = make("weak")
        agent.strong_client.complete = make("verify")
        agent.strong_client.complete_with_tools = make("correct")
        error = None
        try:
            result = Orchestrator(domain="retail", agent=agent, user=User(replies), environment=env,
                                  task=task, max_steps=20, max_errors=5, seed=None,
                                  solo_mode=False).run()
            trajectory = [
                {"role": m.role, "content": getattr(m, "content", None),
                 "tool_calls": [[c.name, c.arguments]
                                for c in (getattr(m, "tool_calls", None) or [])]}
                for m in (getattr(result, "messages", None) or result.trajectory)
            ]
        except Exception as exc:
            trajectory, error = None, f"{type(exc).__name__}: {exc}"
    return {"calls": record, "executions": tools.executions, "trajectory": trajectory,
            "error": error, "stats": _dump(agent.gate.stats),
            "unused": {k: len(v) for k, v in queues.items()}}


def scenarios():
    return {
        "read_then_accepted_cancel": dict(
            weak=[ToolTurn(tool_calls=[call("w1", "get_order_details", order_id="A1")]),
                  ToolTurn(tool_calls=[call("w2", "cancel_order", order_id="A1")]),
                  ToolTurn(content="Order A1 is cancelled.")],
            verify=[verdict(2, reason=None)], correct=[], replies=[]),
        "cancel_rejected_with_replacement": dict(
            weak=[ToolTurn(tool_calls=[call("w1", "cancel_order", order_id="B2")]),
                  ToolTurn(content="Done.")],
            verify=[verdict(0, "wrong order id",
                            {"tool": "cancel_order", "arguments": {"order_id": "A1"}})],
            correct=[], replies=[]),
        "cancel_rejected_corrector": dict(
            weak=[ToolTurn(tool_calls=[call("w1", "cancel_order", order_id="A1")]),
                  ToolTurn(content="Cancelled.")],
            verify=[verdict(0, "customer has not confirmed"), verdict(1, reason=None)],
            correct=[ToolTurn(content="Can you confirm you want to cancel order A1?")],
            replies=["Yes, please cancel it."]),
        "parallel_read_and_cancel_partial": dict(
            weak=[ToolTurn(tool_calls=[call("w1", "get_order_details", order_id="A1"),
                                       call("w2", "cancel_order", order_id="A1")]),
                  ToolTurn(content="Anything else?")],
            verify=[verdict(1, "confirm first")],
            correct=[ToolTurn(content="Please confirm the cancellation.")], replies=[]),
        "malformed_verdict": dict(
            weak=[ToolTurn(tool_calls=[call("w1", "cancel_order", order_id="A1")]),
                  ToolTurn(content="Done.")],
            verify=[TextTurn("not json"), TextTurn("still not json")],
            correct=[ToolTurn(content="Let me check first.")], replies=[]),
    }


def run_all(agent_module) -> dict:
    return {name: run(agent_module, **spec) for name, spec in scenarios().items()}
