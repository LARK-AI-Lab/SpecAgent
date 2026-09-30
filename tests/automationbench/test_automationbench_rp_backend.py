"""Offline tests for the AutomationBench routed backend and its phase-attributed telemetry.

Pure: they run in the repository environment without verifiers/openai/httpx. The backend's
transport is an injected ``post_fn`` and the gateway runs in-process.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INTEG = ROOT / "benchmarks" / "automationbench"
for _p in (str(ROOT / "src"), str(INTEG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rp_backend as rpb  # noqa: E402
import specagent_adapter as adapter  # noqa: E402

from specagent.protocol.accounting import InferencePhase, QuantityStatus  # noqa: E402
from specagent.protocol.types import ModelRole  # noqa: E402
from specagent.runtime.gateway import GatewayConfig, SpeculativeToolGateway  # noqa: E402

WEAK_URL = "http://weak.test/v1"
STRONG_URL = "http://strong.test/v1"


def _completion(content, usage=None, rid="resp-1", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": rid,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage
        or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _backend(post_fn=None, **kw):
    kw.setdefault("weak", rpb.RoleEndpoint(base_url=WEAK_URL, api_key="k-weak"))
    kw.setdefault("strong", rpb.RoleEndpoint(base_url=STRONG_URL, api_key="k-strong"))
    return rpb.RecordingRoutedBackend(
        weak_model="weak-model",
        strong_model="strong-model",
        post_fn=post_fn or (lambda url, body, headers: rpb.PostResult(200, _completion("ok"), {})),
        retry_base_delay=0.0,
        **kw,
    )


def _state(task="task.a", trial="trial-1"):
    return {"info": {"task_name": task}, "trajectory_id": trial}


def _ctx(recorder, state):
    ctx = recorder.begin_request(state)
    token = adapter.CURRENT_REQUEST.set(ctx)
    return ctx, token


# RecordingRoutedBackend ------------------------------------------------------------


def test_backend_routes_by_model():
    seen = []

    def post(url, body, headers):
        seen.append((url, body["model"]))
        return rpb.PostResult(200, _completion("ok"), {})

    backend = _backend(post_fn=post)
    backend.complete({"messages": []}, model="weak-model")
    backend.complete({"messages": []}, model="strong-model")
    assert seen == [
        (WEAK_URL + "/chat/completions", "weak-model"),
        (STRONG_URL + "/chat/completions", "strong-model"),
    ]
    with pytest.raises(ValueError):
        backend.complete({"messages": []}, model="other")


def test_backend_applies_phase_overrides_and_auth():
    seen = []
    backend = _backend(
        post_fn=lambda url, body, headers: seen.append((body, headers))
        or rpb.PostResult(200, _completion("ok"), {})
    )
    backend.complete(
        {"messages": [], "temperature": 0.1},
        model="weak-model",
        reasoning_effort="high",
        max_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    body, headers = seen[0]
    assert body["model"] == "weak-model"
    assert body["temperature"] == 0.1
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 64
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert headers["authorization"] == "Bearer k-weak"


def test_phase_overlay_wins_over_endpoint_overlay_and_raw_authorization():
    backend = _backend(
        weak=rpb.RoleEndpoint(
            base_url=WEAK_URL,
            api_key="ignored",
            authorization="raw-token",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20},
        )
    )
    body = backend.prepare_request(
        {"messages": []},
        model="weak-model",
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["top_k"] == 20
    assert backend._headers(backend.endpoints["weak-model"])["authorization"] == "raw-token"


def test_backend_retry_records_contiguous_failed_then_ok_attempts():
    calls = []

    def flaky(url, body, headers):
        calls.append(url)
        if len(calls) == 1:
            return rpb.PostResult(500, {"error": "boom"}, {})
        return rpb.PostResult(200, _completion("ok"), {})

    backend = _backend(post_fn=flaky)
    backend.complete({"messages": []}, model="weak-model")
    attempts = backend.calls[0]["attempts"]
    assert [a["index"] for a in attempts] == [1, 2]
    assert [a["status"] for a in attempts] == ["failed", "ok"]


def test_backend_transport_exception_is_retried_then_fails_closed():
    def down(url, body, headers):
        raise OSError("connection refused")

    backend = _backend(post_fn=down, max_attempts=2)
    with pytest.raises(RuntimeError, match="upstream transport error"):
        backend.complete({"messages": []}, model="weak-model")
    attempts = backend.calls[0]["attempts"]
    assert [a["status"] for a in attempts] == ["failed", "failed"]
    assert all("OSError" in a["error"] for a in attempts)


def test_backend_non_retryable_fails_closed():
    backend = _backend(
        post_fn=lambda url, body, headers: rpb.PostResult(
            400, {"error": "bad request"}, {}
        )
    )
    with pytest.raises(RuntimeError):
        backend.complete({"messages": []}, model="weak-model")
    attempts = backend.calls[0]["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["status"] == "failed"


def test_backend_cancelled_before_upstream_call_records_cancelled_attempt():
    posts = []
    backend = _backend(
        post_fn=lambda url, body, headers: posts.append(url)
        or rpb.PostResult(200, _completion("ok"), {})
    )
    with pytest.raises(ConnectionAbortedError):
        backend.complete({"messages": []}, model="weak-model", is_cancelled=lambda: True)
    assert posts == []
    assert backend.calls[0]["attempts"][0]["status"] == "cancelled"


def test_backend_session_header_binds_trial():
    recorder = adapter.AttemptRecorder(role=ModelRole.WEAK, phase=InferencePhase.ACT)
    ctx, token = _ctx(recorder, _state(trial="trial-xyz"))
    seen = []
    try:
        backend = _backend(
            post_fn=lambda url, body, headers: seen.append(headers)
            or rpb.PostResult(200, _completion("ok"), {})
        )
        backend.complete({"messages": []}, model="weak-model")
    finally:
        adapter.CURRENT_REQUEST.reset(token)
    assert seen[0]["x-specagent-session-id"] == "trial-xyz"


def test_complete_appends_to_per_request_buffer_when_set():
    """CURRENT_CALLS isolates each request's calls from the shared buffer, so concurrent trials
    never drain each other's calls."""
    backend = _backend()
    buf_a: list = []
    buf_b: list = []
    token = rpb.CURRENT_CALLS.set(buf_a)
    try:
        backend.complete({"messages": []}, model="weak-model")
    finally:
        rpb.CURRENT_CALLS.reset(token)
    token = rpb.CURRENT_CALLS.set(buf_b)
    try:
        backend.complete({"messages": []}, model="strong-model")
    finally:
        rpb.CURRENT_CALLS.reset(token)

    assert len(buf_a) == 1 and buf_a[0]["model"] == "weak-model"
    assert len(buf_b) == 1 and buf_b[0]["model"] == "strong-model"
    assert buf_a[0] is not buf_b[0]
    # The shared debug buffer still accumulates every call.
    assert len(backend.calls) == 2

    # Unset buffer: calls land only in the shared list (direct-use path).
    backend.complete({"messages": []}, model="weak-model")
    assert len(backend.calls) == 3


def test_distinct_model_names_required_for_distinct_endpoints():
    with pytest.raises(ValueError):
        rpb.RecordingRoutedBackend(
            weak_model="same",
            weak=rpb.RoleEndpoint(base_url=WEAK_URL),
            strong_model="same",
            strong=rpb.RoleEndpoint(base_url=STRONG_URL),
        )


# drain_calls: phase-attributed span recording ------------------------------------------


def test_drain_calls_records_role_phase_and_usage():
    recorder = adapter.AttemptRecorder(role=ModelRole.WEAK, phase=InferencePhase.ACT)
    ctx, token = _ctx(recorder, _state())
    try:
        calls = [
            {
                "model": "weak-model",
                "phase": "weak_act",
                "attempts": [
                    {
                        "index": 1,
                        "duration_seconds": 1.5,
                        "status": "ok",
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                        "provider_id": "p1",
                        "headers": {},
                    }
                ],
            },
            {
                "model": "strong-model",
                "phase": "strong_verify",
                "attempts": [
                    {
                        "index": 1,
                        "duration_seconds": 4.0,
                        "status": "ok",
                        "usage": {"prompt_tokens": 21, "completion_tokens": 3},
                        "provider_id": "p2",
                        "headers": {},
                    }
                ],
            },
        ]
        row = rpb.drain_calls(recorder, ctx, calls)
    finally:
        adapter.CURRENT_REQUEST.reset(token)

    spans = recorder.ledger.spans
    assert len(spans) == 2
    assert spans[0].role is ModelRole.WEAK
    assert spans[0].phase is InferencePhase.DRAFT
    assert spans[0].request_id == f"{ctx.request_id}:weak_act:1"
    assert (spans[0].input_tokens, spans[0].output_tokens) == (11, 7)
    assert spans[1].role is ModelRole.STRONG
    assert spans[1].phase is InferencePhase.VERIFICATION
    assert spans[1].request_id == f"{ctx.request_id}:strong_verify:2"
    assert all(s.parent_task_id == "task.a" for s in spans)
    assert all(s.parent_episode_id == "trial-1" for s in spans)
    assert row["phases"] == ["weak_act", "strong_verify"]
    assert row["n_transport_calls"] == 2
    assert (row["request_id"], row["trial_id"], row["task_id"]) == (
        ctx.request_id,
        "trial-1",
        "task.a",
    )


def test_drain_calls_failed_attempt_unknown_usage_and_rank_sidecar():
    recorder = adapter.AttemptRecorder(role=ModelRole.WEAK, phase=InferencePhase.ACT)
    ctx, token = _ctx(recorder, _state())
    try:
        calls = [
            {
                "model": "weak-model",
                "phase": "weak_act",
                "attempts": [
                    {"index": 1, "duration_seconds": 0.2, "status": "failed"},
                    {
                        "index": 2,
                        "duration_seconds": 0.4,
                        "status": "ok",
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                        "headers": {
                            "X-SpecAgent-DP-Rank": "5",
                            "X-SpecAgent-Route-Key": "rk",
                            "X-SpecAgent-Route-Source": "sticky",
                        },
                    },
                ],
            }
        ]
        rpb.drain_calls(recorder, ctx, calls)
    finally:
        adapter.CURRENT_REQUEST.reset(token)

    spans = recorder.ledger.spans
    assert [s.transport_attempt_index for s in spans] == [1, 2]
    assert spans[0].status == "failed"
    assert spans[0].usage_status is QuantityStatus.UNKNOWN
    assert spans[1].status == "ok"
    assert recorder.rank_rows == [
        {
            "request_id": f"{ctx.request_id}:weak_act:1",
            "dp_rank": 5,
            "route_key": "rk",
            "route_source": "sticky",
        }
    ]


def test_drain_calls_untagged_call_falls_back_unknown_phase():
    recorder = adapter.AttemptRecorder(role=ModelRole.WEAK, phase=InferencePhase.ACT)
    ctx, token = _ctx(recorder, _state())
    try:
        calls = [
            {
                "model": "strong-model",
                "phase": "",  # a call made outside the gateway funnel: never invented
                "attempts": [{"index": 1, "duration_seconds": 0.1, "status": "ok"}],
            }
        ]
        rpb.drain_calls(recorder, ctx, calls)
    finally:
        adapter.CURRENT_REQUEST.reset(token)
    span = recorder.ledger.spans[0]
    assert span.phase is InferencePhase.UNKNOWN
    assert span.request_id.endswith(":call:1")


# In-process gateway end to end ---------------------------------------------------------


def test_gateway_handle_produces_phase_attributed_spans():
    """Weak draft, Strong verify (accepted), committed Weak response."""

    def post(url, body, headers):
        if url.startswith(WEAK_URL):
            return rpb.PostResult(
                200, _completion("final answer", rid="w1"), {}
            )
        if url.startswith(STRONG_URL):
            return rpb.PostResult(
                200,
                _completion(
                    json.dumps(
                        {"reason": None, "accepted_steps": 1, "replacement_action": None}
                    ),
                    rid="s1",
                ),
                {},
            )
        raise AssertionError(f"unexpected url {url}")

    backend = _backend(post_fn=post)
    gateway = SpeculativeToolGateway(
        backend, GatewayConfig(weak_model="weak-model", strong_model="strong-model")
    )
    recorder = adapter.AttemptRecorder(role=ModelRole.WEAK, phase=InferencePhase.ACT)
    ctx, token = _ctx(recorder, _state(task="task.gw", trial="trial-gw"))
    try:
        rpb.tag_gateway_phases(gateway)
        decision = gateway.handle(
            {"model": "weak-model", "messages": [{"role": "user", "content": "hi"}]}
        )
        row = rpb.drain_calls(recorder, ctx, backend.calls)
    finally:
        adapter.CURRENT_REQUEST.reset(token)

    # A reply is a message boundary and is always verified.
    assert [c["phase"] for c in backend.calls] == ["weak_act", "strong_verify"]
    assert decision.source == "weak_verified"
    roles = [(s.role, s.phase) for s in recorder.ledger.spans]
    assert roles == [
        (ModelRole.WEAK, InferencePhase.DRAFT),
        (ModelRole.STRONG, InferencePhase.VERIFICATION),
    ]
    assert all(s.parent_task_id == "task.gw" for s in recorder.ledger.spans)
    assert decision.response["choices"][0]["message"]["content"] == "final answer"
    assert row["phases"] == ["weak_act", "strong_verify"]


def test_gateway_backend_failure_records_failed_span():
    """A hard transport failure keeps a failed span; nothing is fabricated."""

    def post(url, body, headers):
        return rpb.PostResult(400, {"error": "bad request"}, {})

    backend = _backend(post_fn=post)
    gateway = SpeculativeToolGateway(
        backend, GatewayConfig(weak_model="weak-model", strong_model="strong-model")
    )
    recorder = adapter.AttemptRecorder(role=ModelRole.WEAK, phase=InferencePhase.ACT)
    ctx, token = _ctx(recorder, _state())
    try:
        rpb.tag_gateway_phases(gateway)
        with pytest.raises(RuntimeError):
            gateway.handle(
                {"model": "weak-model", "messages": [{"role": "user", "content": "hi"}]}
            )
        rpb.drain_calls(recorder, ctx, backend.calls)
    finally:
        adapter.CURRENT_REQUEST.reset(token)

    spans = recorder.ledger.spans
    assert len(spans) == 1
    assert spans[0].status == "failed"
    assert spans[0].phase is InferencePhase.DRAFT  # weak_act was the attempted call
    assert spans[0].usage_status is QuantityStatus.UNKNOWN
    # The gateway rolled its state back: the failed request left no trace in the protocol.
    assert not gateway.state.confirmed and not gateway.state.candidate
