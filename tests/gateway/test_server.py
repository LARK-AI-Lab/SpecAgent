"""Tests for the OpenAI-compatible gateway server (``specagent.server``)."""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from specagent import server
from specagent.runtime.gateway import GatewayConfig, SpeculativeToolGateway

PAPER_CONFIGS = Path(__file__).resolve().parents[1] / "golden" / "paper_configs.json"

ENDPOINT_ARGS = [
    "--weak-base-url", "http://weak.test/v1",
    "--weak-model", "weak",
    "--strong-base-url", "http://strong.test/v1",
    "--strong-model", "strong",
]


def _paper_setting(name: str) -> dict:
    fields = json.loads(PAPER_CONFIGS.read_text())[name]
    return {k: tuple(v) if isinstance(v, list) else v for k, v in fields.items()}


def _config(*extra: str) -> GatewayConfig:
    args = server.build_arg_parser().parse_args([*ENDPOINT_ARGS, *extra])
    return server.build_gateway_config(args)


# (a) Defaults reproduce the MCP-Atlas setting ------------------------------------------


def test_endpoint_flags_alone_yield_the_mcp_atlas_setting():
    expected = GatewayConfig(
        weak_model="weak", strong_model="strong", **_paper_setting("mcp_atlas")
    )
    assert _config() == expected


def test_mcp_atlas_setting_matches_the_recorded_paper_setting():
    assert server.MCP_ATLAS_SETTING == _paper_setting("mcp_atlas")


def test_setting_vitabench_yields_the_recorded_vitabench_setting():
    assert server.VITABENCH_SETTING == _paper_setting("vitabench")
    expected = GatewayConfig(
        weak_model="weak", strong_model="strong", **_paper_setting("vitabench")
    )
    assert _config("--setting", "vitabench") == expected


def test_explicit_flags_override_the_setting():
    config = _config("--setting", "vitabench", "--horizon", "4", "--review-replies")
    assert (config.horizon, config.review_replies, config.weak_max_tokens) == (4, True, 8192)
    assert _config("--weak-max-tokens", "100").weak_max_tokens == 100


def test_unusable_tool_class_file_stops_startup(tmp_path, monkeypatch):
    monkeypatch.setenv(server.TOOL_CLASS_FILE_ENV, "")
    bad = tmp_path / "classes.json"
    bad.write_text('{"create_delivery_order": "write"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="non-bool labels"):
        server.main([*ENDPOINT_ARGS, "--tool-class-file", str(bad)])
    assert server.os.environ[server.TOOL_CLASS_FILE_ENV] == str(bad.resolve())
    monkeypatch.setenv(server.TOOL_CLASS_FILE_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(SystemExit, match="cannot read JSON"):
        server.main(ENDPOINT_ARGS)


# (b) One flag per configuration field ------------------------------------------------------


def _config_field_names() -> list[str]:
    return [
        f.name
        for f in dataclasses.fields(GatewayConfig)
        if f.name not in ("weak_model", "strong_model")
    ]


def test_every_config_field_has_a_flag_and_help():
    parser = server.build_arg_parser()
    options = set(parser._option_string_actions)
    names = _config_field_names()
    assert set(server.FIELD_HELP) == set(names)
    for name in names:
        assert "--" + name.replace("_", "-") in options, name
        assert server.FIELD_HELP[name].strip(), name


def test_boolean_flags_have_negations():
    assert _config("--no-verifier-append-only").verifier_append_only is False
    assert _config("--no-strict-answer-review").strict_answer_review is False
    assert _config("--verifier-include-reasoning").verifier_include_reasoning is True
    assert _config("--repeat-call-guard").repeat_call_guard is True
    assert _config("--no-strict-action-validation").strict_action_validation is False


def test_comma_list_flags():
    config = _config("--readonly-tools", "list_dir, read_file,", "--fuse-correction-boundaries",
                     "mutation")
    assert config.readonly_tools == ("list_dir", "read_file")
    assert config.fuse_correction_boundaries == ("mutation",)
    assert _config("--readonly-tools", "").readonly_tools == ()


def test_json_extra_body_flags():
    config = _config(
        "--weak-extra-body", "null",
        "--verifier-extra-body", '{"thinking": {"type": "enabled"}}',
        "--strong-extra-body", '{"top_k": 20}',
    )
    assert config.weak_extra_body is None
    assert config.verifier_extra_body == {"thinking": {"type": "enabled"}}
    assert config.strong_extra_body == {"top_k": 20}


def test_json_extra_body_rejects_non_objects(capsys):
    with pytest.raises(SystemExit):
        _config("--weak-extra-body", "[1, 2]")


def test_verifier_reasoning_effort_off_sends_none():
    assert _config("--verifier-reasoning-effort", "off").verifier_reasoning_effort is None
    assert _config("--verifier-reasoning-effort", "high").verifier_reasoning_effort == "high"


def test_numeric_and_choice_flags():
    config = _config("--horizon", "3", "--task-semantics", "generic", "--workspace-dir", "/w")
    assert config.horizon == 3
    assert config.task_semantics == "generic"
    assert config.workspace_dir == "/w"
    with pytest.raises(SystemExit):
        _config("--task-semantics", "chat")


# (c) Request helpers ----------------------------------------------------------------------


def test_apply_role_extra_body_without_overlay_keeps_the_client_body():
    payload = {
        "messages": [],
        "extra_body": {"x": 1},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = server.apply_role_extra_body(payload, None)
    assert result is payload
    assert result == {"messages": [], "chat_template_kwargs": {"enable_thinking": False}}


def test_apply_role_extra_body_overlay_replaces_thinking_controls():
    payload = {
        "messages": [],
        "extra_body": {"x": 1},
        "chat_template_kwargs": {"enable_thinking": False},
        "separate_reasoning": True,
        "enable_thinking": False,
        "temperature": 0.2,
    }
    result = server.apply_role_extra_body(payload, {"thinking": {"type": "disabled"}})
    assert result == {
        "messages": [],
        "temperature": 0.2,
        "thinking": {"type": "disabled"},
    }


def test_backend_prepare_request_forces_buffered_role_payload():
    backend = server.OpenAIBackend("http://weak.test/v1/", "k", 5.0)
    assert backend.base_url == "http://weak.test/v1"
    payload = backend.prepare_request(
        {"model": "client", "stream": True, "stream_options": {"include_usage": True},
         "messages": []},
        model="weak",
        reasoning_effort="low",
        max_tokens=32,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert payload == {
        "model": "weak",
        "stream": False,
        "messages": [],
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_effort": "low",
        "max_tokens": 32,
    }


def test_routed_backend_requires_distinct_models_and_known_names():
    weak = server.OpenAIBackend("http://weak.test/v1", "k", 5.0)
    strong = server.OpenAIBackend("http://strong.test/v1", "k", 5.0)
    with pytest.raises(ValueError):
        server.RoutedOpenAIBackend(weak_model="m", weak=weak, strong_model="m", strong=strong)
    routed = server.RoutedOpenAIBackend(
        weak_model="w", weak=weak, strong_model="s", strong=strong
    )
    assert routed.prepare_request({}, model="s")["model"] == "s"
    with pytest.raises(ValueError):
        routed.prepare_request({}, model="other")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_session_identity_prefers_a_session_header():
    request = {"messages": [{"role": "system", "content": "task"}]}
    key, source = server.request_session_identity(request, {"X-SpecAgent-Session-Id": "s1"})
    assert (key, source) == (_digest("s1"), "x-specagent-session-id")
    key, source = server.request_session_identity(request, {"X-Session-Id": "s2"})
    assert (key, source) == (_digest("s2"), "x-session-id")
    both = {"x-session-id": "s2", "x-specagent-session-id": "s1"}
    assert server.request_session_identity(request, both)[1] == "x-specagent-session-id"
    # An empty header does not count.
    assert server.request_session_identity(request, {"x-session-id": ""})[1] == (
        "system_message_hash"
    )


def test_session_identity_falls_back_to_the_opening_message_hash():
    first = {
        "messages": [
            {"role": "system", "content": "task-specific stable prompt"},
            {"role": "user", "content": "start"},
        ]
    }
    later = {
        "messages": [
            *first["messages"],
            {"role": "assistant", "content": "progress"},
            {"role": "user", "content": "continue"},
        ]
    }
    key, source = server.request_session_identity(first, {})
    assert source == "system_message_hash"
    assert key == _digest("task-specific stable prompt")
    # Stable as the history grows, distinct across tasks.
    assert server.request_session_identity(later, {})[0] == key
    other = {"messages": [{"role": "system", "content": "another task"}]}
    assert server.request_session_identity(other, {})[0] != key


def test_session_identity_uses_the_user_message_without_a_system_message():
    blocks = [{"type": "text", "text": "hello"}]
    key, source = server.request_session_identity(
        {"messages": [{"role": "user", "content": blocks}]}, {}
    )
    assert source == "user_message_hash"
    assert key == _digest(json.dumps(blocks, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")))
    with pytest.raises(ValueError):
        server.request_session_identity({"messages": []}, {})


# (d) End to end over HTTP ----------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


def _upstream_completion(model: str, message: dict, usage: dict) -> dict:
    return {
        "id": f"cmpl-{model}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
        "usage": usage,
    }


class _Upstream:
    """A scripted OpenAI-compatible upstream: one queue of assistant messages per model."""

    def __init__(self) -> None:
        self.script: dict[str, list[dict]] = {"weak": [], "strong": []}
        self.requests: list[tuple[str, dict, dict]] = []
        upstream = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002
                pass

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                upstream.requests.append((self.path, body, dict(self.headers.items())))
                model = body["model"]
                usage = {
                    "prompt_tokens": 100 if model == "strong" else 10,
                    "completion_tokens": 7 if model == "strong" else 3,
                    "total_tokens": 107 if model == "strong" else 13,
                }
                payload = json.dumps(
                    _upstream_completion(model, upstream.script[model].pop(0), usage)
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def calls_for(self, model: str) -> list[dict]:
        return [body for _, body, _ in self.requests if body["model"] == model]


@pytest.fixture
def stack(tmp_path, monkeypatch):
    # Loopback traffic must not be routed through a proxy configured on the host.
    monkeypatch.setattr(
        urllib.request, "_opener", urllib.request.build_opener(urllib.request.ProxyHandler({}))
    )
    upstream = _Upstream()
    upstream.thread.start()
    args = server.build_arg_parser().parse_args(
        [
            "--weak-base-url", upstream.base_url,
            "--weak-api-key", "weak-key",
            "--weak-model", "weak",
            "--strong-base-url", upstream.base_url,
            "--strong-api-key", "strong-key",
            "--strong-model", "strong",
            "--timeout", "10",
        ]
    )
    gateway_config = server.build_gateway_config(args)
    backend = server.RoutedOpenAIBackend(
        weak_model=args.weak_model,
        weak=server.OpenAIBackend(args.weak_base_url, args.weak_api_key, args.timeout),
        strong_model=args.strong_model,
        strong=server.OpenAIBackend(args.strong_base_url, args.strong_api_key, args.timeout),
    )
    httpd = server.GatewayServer(("127.0.0.1", 0), server.Handler)
    httpd.gateway_factory = lambda: SpeculativeToolGateway(backend, gateway_config)
    httpd.gateways = {}
    httpd.gateways_lock = threading.Lock()
    httpd.telemetry_path = tmp_path / "telemetry.jsonl"
    httpd.telemetry_lock = threading.Lock()
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield httpd, upstream, backend
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)
        upstream.httpd.shutdown()
        upstream.httpd.server_close()
        assert not thread.is_alive()


def _post(httpd, body: dict, headers: dict | None = None):
    client = http.client.HTTPConnection(*httpd.server_address[:2], timeout=10)
    client.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json", **(headers or {})},
    )
    response = client.getresponse()
    try:
        return response.status, dict(response.getheaders()), response.read().decode()
    finally:
        response.close()
        client.close()


def _telemetry(httpd, count: int) -> list[dict]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if httpd.telemetry_path.exists():
            rows = [json.loads(line) for line in httpd.telemetry_path.read_text().splitlines()]
            if len(rows) >= count:
                return rows
        time.sleep(0.01)
    raise AssertionError("gateway did not record the request")


def _sse_events(body: str) -> list:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def _verdict(accepted: int, reason=None, replacement=None) -> dict:
    return {
        "role": "assistant",
        "content": json.dumps(
            {"reason": reason, "accepted_steps": accepted, "replacement_action": replacement}
        ),
    }


def test_health_and_model_listing(stack):
    httpd, _, _ = stack
    for path, expected in (
        ("/healthz", {"status": "ok"}),
        ("/v1/models", {"object": "list", "data": [{"id": "specagent", "object": "model"}]}),
    ):
        client = http.client.HTTPConnection(*httpd.server_address[:2], timeout=5)
        client.request("GET", path)
        response = client.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == expected
        client.close()
    client = http.client.HTTPConnection(*httpd.server_address[:2], timeout=5)
    client.request("POST", "/v1/other", "{}")
    response = client.getresponse()
    assert response.status == 404
    client.close()


def test_verified_reply_and_fused_repair_over_http(stack):
    httpd, upstream, _ = stack
    upstream.script["weak"] = [
        {"role": "assistant", "content": "The answer is 42.", "reasoning_content": "private"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_w1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path": "a"}'},
                }
            ],
        },
    ]
    upstream.script["strong"] = [
        _verdict(1),
        _verdict(
            0,
            reason="Step 1 writes the wrong path.",
            replacement={"tool": "write_file", "arguments": {"path": "b"}},
        ),
    ]

    # Non-streaming: Weak's reply is a message boundary; Strong accepts it.
    status, _, body = _post(
        httpd,
        {
            "model": "specagent",
            "messages": [
                {"role": "system", "content": "Answer the question."},
                {"role": "user", "content": "What is six times seven?"},
            ],
        },
    )
    assert status == 200
    message = json.loads(body)["choices"][0]["message"]
    assert message["content"] == "The answer is 42."
    # Hidden reasoning never reaches the client.
    assert "reasoning_content" not in message

    # Streaming, in a second session: Weak's write is rejected and the replacement commits.
    status, headers, body = _post(
        httpd,
        {
            "model": "specagent",
            "stream": True,
            "messages": [
                {"role": "system", "content": "Edit files."},
                {"role": "user", "content": "Write file b."},
            ],
            "tools": TOOLS,
        },
        headers={"X-SpecAgent-Session-Id": "session-2"},
    )
    assert status == 200
    assert headers["Content-Type"] == "text/event-stream"
    assert body.startswith(":")  # the keep-alive comment precedes the verified completion
    events = _sse_events(body)
    assert events[-1] == "[DONE]"
    assert body.count("data: [DONE]") == 1
    first, final = events[0], events[1]
    calls = first["choices"][0]["delta"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["write_file"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "b"}
    assert final["choices"][0]["finish_reason"] == "tool_calls"
    assert "call_w1" not in body

    # The upstream saw role-specific payloads with the configured overlays and credentials.
    weak_calls, strong_calls = upstream.calls_for("weak"), upstream.calls_for("strong")
    assert len(weak_calls) == 2 and len(strong_calls) == 2
    assert all(call["stream"] is False for call in weak_calls + strong_calls)
    assert weak_calls[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert strong_calls[0]["thinking"] == {"type": "disabled"}
    assert strong_calls[0]["response_format"] == {"type": "json_object"}
    assert strong_calls[0]["max_tokens"] == 500
    auth = {body["model"]: headers["Authorization"] for _, body, headers in upstream.requests}
    assert auth == {"weak": "Bearer weak-key", "strong": "Bearer strong-key"}
    assert {path for path, _, _ in upstream.requests} == {"/v1/chat/completions"}

    rows = _telemetry(httpd, 2)
    accepted, repaired = rows
    assert accepted["source"] == "weak_verified"
    assert accepted["delivery_status"] == "delivered"
    assert accepted["session_identity_source"] == "system_message_hash"
    assert accepted["boundary"] == "message"
    assert accepted["accepted_steps"] == 1
    assert accepted["verdict_diagnostic_status"] == "model_verdict"
    assert accepted["verification_id"]
    assert accepted["horizon"] == 6
    assert [
        (call["phase"], call["model"], call["usage"]["prompt_tokens"])
        for call in accepted["model_calls"]
    ] == [("weak_act", "weak", 10), ("strong_verify", "strong", 100)]
    assert all(call["transport_attempts"] == 1 for call in accepted["model_calls"])

    assert repaired["source"] == "strong_fused_correction"
    assert repaired["session_identity_source"] == "x-specagent-session-id"
    assert repaired["session_key"] == _digest("session-2")
    assert repaired["boundary"] == "mutation"
    assert repaired["mutating"] is True
    assert repaired["accepted_steps"] == 0
    assert repaired["verifier_accepted_steps"] == 0
    assert repaired["rejection_code"] == "other"
    assert repaired["verdict_reason"] == "Step 1 writes the wrong path."
    assert json.loads(repaired["verdict_raw"])["replacement_action"] == {
        "tool": "write_file",
        "arguments": {"path": "b"},
    }
    assert repaired["replacement_validation_failure_kind"] == ""
    assert repaired["correction_action"] == {"tool": "write_file", "arguments": {"path": "b"}}
    assert repaired["corrector_generation_count"] == 0
    assert [call["phase"] for call in repaired["model_calls"]] == ["weak_act", "strong_verify"]
    assert repaired["model_calls"][1]["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 7,
        "total_tokens": 107,
    }
    assert repaired["request_id"] != accepted["request_id"]
    assert len(httpd.gateways) == 2


def test_upstream_failure_is_reported_and_recorded(stack):
    httpd, upstream, backend = stack
    upstream.httpd.shutdown()
    upstream.httpd.server_close()
    for role_backend in backend.backends.values():
        role_backend.max_retries = 0
    status, _, body = _post(
        httpd,
        {"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-session-id": "k"},
    )
    assert status == 502
    assert json.loads(body)["error"]["type"] == "RuntimeError"
    row = _telemetry(httpd, 1)[0]
    assert row["source"] == "request_failed"
    assert row["delivery_status"] == "failed"
    assert [call["phase"] for call in row["model_calls"]] == ["weak_act"]
    assert row["model_calls"][0]["usage"] == {}
