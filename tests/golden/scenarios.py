"""Scripted sessions that pin the gateway's exact model traffic under the paper settings.

Each scenario drives the gateway like a client would: it sends the transcript, appends the
delivered turn and a synthetic tool result, and asks again. Every backend request (messages,
tools, model, effort, token cap, request body) and every delivered turn is recorded, with ids
made deterministic, so a refactor that changes behaviour changes the record.
"""

from __future__ import annotations

import itertools
import json
import os
from collections import deque
from pathlib import Path
from typing import Any
from unittest import mock

TOOLS = [
    {"type": "function", "function": {"name": name, "description": name,
                                      "parameters": {"type": "object",
                                                     "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}}
    for name in ("read_file", "search_docs", "write_file", "send_message")
]


_IDS = itertools.count()


def completion(*calls: tuple[str, dict], content: str | None = None) -> dict:
    tool_calls = [
        {"id": f"call_{next(_IDS)}_{name}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}
        for i, (name, arguments) in enumerate(calls)
    ]
    return {
        "id": "cmpl", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                                             "reasoning_content": "private reasoning",
                                             "tool_calls": tool_calls or None},
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


def verdict(
    accepted: int, replacement: dict | None = None, reason: str | None = "bad step"
) -> dict:
    return completion(content=json.dumps(
        {"reason": reason, "accepted_steps": accepted, "replacement_action": replacement}))


class Recorder:
    def __init__(self, responses: list[dict]):
        self.responses = deque(responses)
        self.calls: list[dict] = []

    def complete(self, request, *, model, reasoning_effort=None, max_tokens=None,
                 extra_body=None, is_cancelled=None):
        self.calls.append({"model": model, "reasoning_effort": reasoning_effort,
                           "max_tokens": max_tokens, "extra_body": extra_body,
                           "request": json.loads(json.dumps(request, default=str))})
        return self.responses.popleft()


def drive(gateway_cls, config, responses: list[dict], turns: int, user: str,
          tools: list[dict] = TOOLS, opening: tuple[dict, ...] = ()) -> dict:
    """Run a client loop for `turns` gateway requests; return the full record."""
    backend = Recorder(responses)
    counter = itertools.count()

    class _U:
        def __init__(self):
            self.hex = f"{next(counter):032x}"

    with mock.patch("specagent.runtime.gateway.uuid4", _U):
        gateway = gateway_cls(backend, config)
        messages: list[dict] = [{"role": "system", "content": "You are an agent."},
                                *opening, {"role": "user", "content": user}]
        delivered = []
        error = None
        try:
            _loop(gateway, backend, messages, delivered, turns, tools)
        except Exception as exc:  # the failure itself is part of the behaviour
            error = f"{type(exc).__name__}: {exc}"
    return {"backend_calls": backend.calls, "delivered": delivered, "error": error,
            "unused_responses": len(backend.responses)}


def _loop(gateway, backend, messages, delivered, turns, tools) -> None:
    for _ in range(turns):
        if not backend.responses:
            break
        decision = gateway.handle({"model": "specagent", "messages": list(messages),
                                   "tools": tools})
        message = decision.response["choices"][0]["message"]
        delivered.append({"source": decision.source, "boundary": decision.boundary,
                          "accepted": decision.verifier_accepted_steps,
                          "message": {k: message.get(k) for k in ("content", "tool_calls")}})
        messages.append({"role": "assistant", "content": message.get("content"),
                         **({"tool_calls": message["tool_calls"]}
                            if message.get("tool_calls") else {})})
        for call in message.get("tool_calls") or []:
            name = call["function"]["name"]
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": f"result of {name} " + "x" * 50})
        if not message.get("tool_calls"):
            messages.append({"role": "user", "content": "Please continue."})


READ = ("read_file", {"path": "a.txt"})
READ2 = ("search_docs", {"path": "b"})
WRITE = ("write_file", {"path": "out.txt"})
GOOD_WRITE = ("write_file", {"path": "fixed.txt"})

SCENARIOS: dict[str, tuple[list[dict], int]] = {
    # reads released unreviewed, then an accepted reply
    "reads_then_accepted_reply": ([completion(READ), completion(READ2),
                                   completion(content="The answer is 7."), verdict(3)], 3),
    # a write held and accepted
    "write_accepted": ([completion(READ), completion(WRITE), verdict(2)], 2),
    # a write rejected with a grounded replacement
    "write_fused_replacement": ([completion(WRITE),
                                 verdict(0, {"tool": "write_file",
                                             "arguments": {"path": "fixed.txt"}})], 1),
    # a reply rejected without replacement; the corrector issues two reads
    "reply_rejected_corrector_reads": ([completion(content="Done."), verdict(0, None),
                                        completion(READ, READ2),
                                        completion(content="Final: 7."), verdict(1)], 3),
    # a parallel read+write batch partially accepted, corrected next request
    "partial_batch": ([completion(READ, WRITE), verdict(1), completion(GOOD_WRITE)], 2),
    # a rejected write whose corrector names an undeclared tool, then a valid one
    "corrector_invalid_then_valid": ([completion(WRITE), verdict(0, None),
                                      completion(("no_such_tool", {"path": "x"})),
                                      completion(GOOD_WRITE)], 1),
    # repeated identical reads, then a reply
    "repeat_reads": ([completion(READ), completion(READ), completion(READ),
                      completion(content="Answer."), verdict(4), completion(content="Answer."),
                      verdict(1)], 5),
    # malformed verdict twice: fail closed onto the corrector
    "malformed_verdict": ([completion(WRITE), completion(content="not json"),
                           completion(content="still not json"), completion(GOOD_WRITE)], 1),
    # a rejected reply with a grounded tool replacement (fused only where the profile allows)
    "reply_rejected_with_replacement": ([completion(content="Done."),
                                         verdict(0, {"tool": "search_docs",
                                                     "arguments": {"path": "q"}}),
                                         completion(READ2), completion(content="Final."),
                                         verdict(1), verdict(1)], 3),
    # a long read-only stretch that crosses the Strong takeover threshold
    "long_reads_takeover": ([completion(("read_file", {"path": f"f{i}"})) for i in range(32)]
                            + [completion(content="Strong answer."), verdict(1),
                               completion(content="Strong answer.")] , 34),
}


# VitaBench-shaped sessions (Delivery tools, labels from the benchmark's @is_tool decorators).
# Only state-changing calls are reviewed: replies and K=6 read chunks commit unreviewed.
VITA_TOOL_CLASS_FILE = Path(__file__).resolve().parent / "vitabench_tool_classes.json"
_VITA_PARAMS = {
    "delivery_store_search_recommend": {"keywords": {"type": "array",
                                                     "items": {"type": "string"}}},
    "get_delivery_store_info": {"store_id": {"type": "string"}},
    "get_delivery_product_info": {"product_id": {"type": "string"}},
    "weather": {"city": {"type": "string"}},
    "longitude_latitude_to_distance": {"origin": {"type": "string"},
                                       "destination": {"type": "string"}},
    "create_delivery_order": {"store_id": {"type": "string"}, "product_id": {"type": "string"},
                              "address": {"type": "string"}},
    "pay_delivery_order": {"order_id": {"type": "string"}},
    "modify_delivery_order": {"order_id": {"type": "string"}, "note": {"type": "string"}},
}
VITA_TOOLS = [
    {"type": "function", "function": {"name": name, "description": name,
                                      "parameters": {"type": "object", "properties": params,
                                                     "required": sorted(params)}}}
    for name, params in _VITA_PARAMS.items()
]
VITA_OPENING = ({"role": "assistant", "content": "Hello, how can I help you?"},)

SEARCH = ("delivery_store_search_recommend", {"keywords": ["noodles"]})
STORE = ("get_delivery_store_info", {"store_id": "S1"})
PRODUCT = ("get_delivery_product_info", {"product_id": "P2"})
WEATHER = ("weather", {"city": "Beijing"})
DISTANCE = ("longitude_latitude_to_distance", {"origin": "home", "destination": "S1"})
ORDER_ARGS = {"store_id": "S1", "product_id": "P2", "address": "Room 301"}
BAD_ORDER = ("create_delivery_order", {**ORDER_ARGS, "product_id": "P9"})
ORDER = ("create_delivery_order", ORDER_ARGS)
PAY = ("pay_delivery_order", {"order_id": "O1"})
BAD_PAY = ("pay_delivery_order", {"order_id": "O7"})
NOTE = ("modify_delivery_order", {"order_id": "O1", "note": "no chili"})

VITA_SCENARIOS: dict[str, tuple[list[dict], int]] = {
    # replies commit unreviewed, with the reads drafted before them
    "vita_replies_unreviewed": ([completion(SEARCH), completion(content="Which store?"),
                                 completion(STORE), completion(content="S1 is open.")], 4),
    # reads then an order held and accepted, then a payment accepted against the history
    "vita_write_accepted": ([completion(SEARCH), completion(STORE), completion(ORDER),
                             verdict(3, reason=None), completion(PAY), verdict(1, reason=None),
                             completion(content="Your order is paid.")], 5),
    # a wrong order replaced by the verifier's grounded order
    "vita_write_fused_replacement": ([completion(STORE), completion(BAD_ORDER),
                                      verdict(1, {"tool": ORDER[0], "arguments": ORDER[1]},
                                              "wrong product"),
                                      completion(PAY), verdict(1, reason=None)], 3),
    # a wrong order repaired by a Strong turn with a read and the order
    "vita_write_strong_repair": ([completion(BAD_ORDER), verdict(0, None, "wrong product"),
                                  completion(PRODUCT, ORDER), completion(PAY),
                                  verdict(1, reason=None), completion(content="Done.")], 4),
    # a replacement that replays the rejected order is discarded; Strong replies instead
    "vita_replacement_replays_rejected": ([completion(BAD_ORDER),
                                           verdict(0, {"tool": BAD_ORDER[0],
                                                       "arguments": BAD_ORDER[1]}),
                                           completion(content="Which product did you want?"),
                                           completion(ORDER), verdict(1, reason=None)], 3),
    # K=6 reads, generic tools among them, commit unreviewed; then a reply
    "vita_horizon_reads": ([completion(SEARCH), completion(STORE), completion(WEATHER),
                            completion(DISTANCE), completion(PRODUCT), completion(STORE),
                            completion(WEATHER), completion(content="It will rain; order?")],
                           8),
    # the same read repeated: no review ever runs on a pure read loop
    "vita_read_loop": ([completion(STORE) for _ in range(9)]
                       + [completion(content="S1 sells noodles.")], 10),
    # a parallel read and order: the read is released, Strong repairs on the next request
    "vita_partial_batch": ([completion(STORE, BAD_ORDER), verdict(1, None, "wrong product"),
                            completion(ORDER), completion(PAY), verdict(1, reason=None)], 3),
    # an executed read is rejected at the order boundary; Strong repairs from the history
    "vita_executed_read_rejected": ([completion(SEARCH), completion(BAD_ORDER),
                                     verdict(0, None, "searched the wrong cuisine"),
                                     completion(STORE)], 2),
    # two rejections in one session: Strong's earlier reasoning is re-attached
    "vita_two_repairs": ([completion(BAD_ORDER), verdict(0, None, "wrong product"),
                          completion(ORDER), completion(BAD_PAY), verdict(0, None, "wrong order"),
                          completion(PAY), completion(NOTE), verdict(1, reason=None)], 4),
    # a malformed verdict fails closed onto the Strong repair
    "vita_malformed_verdict": ([completion(ORDER), completion(content="not json"),
                                completion(content="still not json"), completion(ORDER)], 1),
}

# Scenarios and environment that only one paper config runs.
CONFIG_SCENARIOS: dict[str, dict[str, tuple[list[dict], int]]] = {"vitabench": VITA_SCENARIOS}
CONFIG_ENV: dict[str, dict[str, str]] = {
    "vitabench": {"SPECAGENT_TOOL_CLASS_FILE": str(VITA_TOOL_CLASS_FILE)},
}


def run_all(gateway_cls, configs: dict[str, Any]) -> dict:
    out = {}
    for config_name, config in configs.items():
        with mock.patch.dict(os.environ):
            os.environ.pop("SPECAGENT_TOOL_CLASS_FILE", None)
            os.environ.update(CONFIG_ENV.get(config_name, {}))
            for name, (responses, turns) in SCENARIOS.items():
                out[f"{config_name}/{name}"] = drive(
                    gateway_cls, config, [json.loads(json.dumps(r)) for r in responses], turns,
                    user=f"Task for {name}",
                )
            for name, (responses, turns) in CONFIG_SCENARIOS.get(config_name, {}).items():
                out[f"{config_name}/{name}"] = drive(
                    gateway_cls, config, [json.loads(json.dumps(r)) for r in responses], turns,
                    user=f"Task for {name}", tools=VITA_TOOLS, opening=VITA_OPENING,
                )
    return out
