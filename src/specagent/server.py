"""Serve the SpecAgent gateway as an OpenAI-compatible chat-completions endpoint.

Run ``python -m specagent.server --weak-base-url ... --weak-model ... --strong-base-url ...
--strong-model ...``. Every protocol flag defaults to the MCP-Atlas setting of the paper, so
endpoint and model flags alone reproduce that configuration. Each client session gets its own
gateway state; one telemetry line per request records the decision, the verdict, the repair, and
every model call with its token usage.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from specagent.http_auth import VALID_AUTH_SCHEMES, authorization_header, parse_auth_scheme
from specagent.protocol.evidence import VerifierInputError
from specagent.runtime.gateway import (
    TASK_SEMANTICS,
    GatewayConfig,
    GatewayDecision,
    SpeculativeToolGateway,
    completion_to_sse,
)

LOG = logging.getLogger("specagent.server")

# The paper's MCP-Atlas setting: every GatewayConfig field it sets away from the default. The
# CLI defaults are these values; ``verifier_extra_body`` is the thinking switch of the provider
# used for that setting and usually needs adjusting for another provider.
MCP_ATLAS_SETTING: dict[str, Any] = {
    "horizon": 6,
    "verifier_max_tokens": 500,
    "corrector_max_tokens": 4096,
    "weak_extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    "verifier_extra_body": {"thinking": {"type": "disabled"}},
    "task_semantics": "qa",
    "readonly_tools": ("cli-mcp-server_run_command",),
    "verifier_include_reasoning": False,
    "weak_history_reasoning": True,
    "verifier_recent_observation_chars": 12000,
    "verifier_append_only": True,
    "verifier_all_tool_contracts": True,
    "verifier_answer_checks": True,
    "strict_answer_review": True,
    "corrector_reask_invalid_call": True,
    "weak_overflow_resample": True,
    "strong_overflow_drop": True,
    "strong_takeover_after_tool_results": 30,
}

# One line of CLI help per GatewayConfig field; the flag is the field name with dashes.
FIELD_HELP: dict[str, str] = {
    "horizon": "Draft batching cap: a candidate of this many reads commits without review",
    "weak_reasoning_effort": "reasoning_effort sent with Weak drafts",
    "verifier_reasoning_effort": "reasoning_effort sent with Strong reviews ('off' sends none)",
    "correction_reasoning_effort": "reasoning_effort sent with Strong repair turns",
    "verifier_max_tokens": "Output cap of a Strong review",
    "corrector_max_tokens": "Output cap of a Strong repair turn",
    "verifier_max_input_tokens": "Conservative size bound on a review request (unset: none)",
    "weak_extra_body": "JSON overlay for Weak requests ('null' keeps the client's body)",
    "strong_extra_body": "JSON overlay for Strong repair requests ('null' keeps the client's body)",
    "verifier_extra_body": "JSON overlay for Strong review requests ('null' keeps the body)",
    "workspace_dir": "Workspace root shown to the verifier as an anchor (empty: none)",
    "task_semantics": "'qa' adds the read-only answer rule to the repair note",
    "readonly_tools": "Comma-separated tools treated as reads although their names look like "
    "writes",
    "verifier_include_reasoning": "Show the candidate steps' reasoning to the verifier",
    "verifier_history_reasoning": "Show committed steps' reasoning to the verifier",
    "correction_history_reasoning": "Keep earlier turns' reasoning in the repair transcript",
    "weak_history_reasoning": "Re-attach Weak's own reasoning to its earlier turns",
    "verifier_recent_observation_chars": "Observation cap for the batch under review and the "
    "latest confirmed steps (0: the common cap)",
    "verifier_append_only": "Append-only review input, so consecutive reviews share a prefix",
    "verifier_all_tool_contracts": "Give every review all offered tool schemas for replacements",
    "verifier_answer_checks": "Add the answer review checks to the review prompt",
    "strict_answer_review": "Answers must cover every asked item with a supported value",
    "verifier_prompt_suffix": "Text appended to the review system prompt",
    "fuse_correction_boundaries": "Comma-separated boundaries (message, mutation) where a "
    "verifier replacement may commit",
    "strict_action_validation": "Reject schema-invalid or unknown-tool repair calls in the gateway "
    "instead of delivering them to the environment",
    "repeat_call_guard": "Route an all-repeat drafted batch to Strong repair",
    "weak_review_hint": "Show a cleared rejection reason once to the next Weak draft",
    "corrector_reask_invalid_call": "Re-ask a repair turn once when a call is undeclared or "
    "schema-invalid",
    "weak_overflow_resample": "On a Weak context overflow, drop oversized results and re-sample",
    "strong_overflow_drop": "On a Strong context overflow, drop the largest results and retry",
    "strong_takeover_after_tool_results": "Strong owns the task once the transcript holds this "
    "many tool results (0: never)",
}

_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504, 524}

# Keys that switch a model's thinking: an overlay replaces them so they cannot leak across roles.
_THINKING_BODY_KEYS = ("chat_template_kwargs", "separate_reasoning", "enable_thinking")


def apply_role_extra_body(
    payload: dict[str, Any], extra_body: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Drop nested extra_body and, when an overlay is set, replace thinking-control keys."""
    payload.pop("extra_body", None)
    if extra_body is None:
        return payload
    for key in _THINKING_BODY_KEYS:
        payload.pop(key, None)
    payload.update(dict(extra_body))
    return payload


class OpenAIBackend:
    """One OpenAI-compatible upstream with bounded retries on transient failures."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        *,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 20.0,
        auth_scheme: str = "bearer",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.auth_scheme = parse_auth_scheme(auth_scheme)

    def prepare_request(
        self,
        request: Mapping[str, Any],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(request)
        payload["model"] = model
        payload["stream"] = False
        payload.pop("stream_options", None)
        apply_role_extra_body(payload, extra_body)
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def complete(
        self,
        request: dict[str, Any],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        payload = self.prepare_request(
            request,
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": authorization_header(self.api_key, scheme=self.auth_scheme),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SpecAgent/0.1",
        }
        attempt = 0
        while True:
            if is_cancelled is not None and is_cancelled():
                error = ConnectionAbortedError("client disconnected before upstream attempt")
                error.transport_attempts = attempt
                raise error
            started = time.perf_counter()
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, method="POST", headers=headers
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    result = json.load(response)
                LOG.info(
                    "upstream ok base_url=%s model=%s elapsed=%.2fs attempt=%d",
                    self.base_url,
                    model,
                    time.perf_counter() - started,
                    attempt,
                )
                result["_specagent_transport"] = {"attempts": attempt + 1}
                return result
            except urllib.error.HTTPError as error:
                error_body = error.read(4000).decode("utf-8", "replace")
                try:
                    error_payload = json.loads(error_body)
                    detail = error_payload.get("error", error_payload)
                except (ValueError, AttributeError):
                    detail = {}
                lower_body = error_body.lower()
                context_overflow = (
                    isinstance(detail, Mapping)
                    and any(
                        detail.get(field)
                        in {"context_length_exceeded", "max_context_length_exceeded"}
                        for field in ("code", "type")
                    )
                ) or (
                    "context" in lower_body
                    and "length" in lower_body
                    and any(word in lower_body for word in ("exceed", "too long", "maximum"))
                )
                if context_overflow:
                    failure = VerifierInputError("upstream rejected the request context length")
                    failure.transport_attempts = attempt + 1
                    raise failure from error
                LOG.warning(
                    "upstream HTTP error base_url=%s model=%s elapsed=%.2fs status=%s "
                    "attempt=%d body=%r",
                    self.base_url,
                    model,
                    time.perf_counter() - started,
                    error.code,
                    attempt,
                    error_body,
                )
                if error.code not in _RETRYABLE_HTTP_STATUS or attempt >= self.max_retries:
                    failure = RuntimeError(f"upstream HTTP {error.code}: {error_body}")
                    failure.transport_attempts = attempt + 1
                    raise failure from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                LOG.warning(
                    "upstream transport error base_url=%s model=%s elapsed=%.2fs attempt=%d "
                    "error=%r",
                    self.base_url,
                    model,
                    time.perf_counter() - started,
                    attempt,
                    error,
                )
                if attempt >= self.max_retries:
                    failure = RuntimeError(f"upstream transport error: {error}")
                    failure.transport_attempts = attempt + 1
                    raise failure from error

            delay = min(self.retry_base_delay * (2**attempt), self.retry_max_delay)
            if is_cancelled is not None and is_cancelled():
                failure = ConnectionAbortedError("client disconnected before upstream retry")
                failure.transport_attempts = attempt + 1
                raise failure
            LOG.warning(
                "retrying upstream call base_url=%s model=%s attempt=%d delay=%.1fs",
                self.base_url,
                model,
                attempt + 1,
                delay,
            )
            time.sleep(delay)
            attempt += 1


class RoutedOpenAIBackend:
    """Route model calls to role-specific OpenAI-compatible endpoints by model name."""

    def __init__(
        self,
        *,
        weak_model: str,
        weak: OpenAIBackend,
        strong_model: str,
        strong: OpenAIBackend,
    ) -> None:
        if weak_model == strong_model and weak is not strong:
            raise ValueError("different endpoints require distinct weak and strong model names")
        self.backends = {weak_model: weak, strong_model: strong}

    def _backend(self, model: str) -> OpenAIBackend:
        try:
            return self.backends[model]
        except KeyError as error:
            raise ValueError(f"no upstream endpoint configured for model {model!r}") from error

    def prepare_request(self, request: Mapping[str, Any], *, model: str, **kwargs: Any):
        return self._backend(model).prepare_request(request, model=model, **kwargs)

    def complete(self, request: dict[str, Any], *, model: str, **kwargs: Any) -> dict[str, Any]:
        return self._backend(model).complete(request, model=model, **kwargs)


class GatewayServer(ThreadingHTTPServer):
    gateway_factory: Callable[[], SpeculativeToolGateway]
    gateways: dict[str, tuple[SpeculativeToolGateway, threading.Lock]]
    gateways_lock: threading.Lock
    telemetry_path: Path
    telemetry_lock: threading.Lock
    heartbeat_interval: float = 5.0
    trace_path: Path | None = None

    def gateway_for(self, session_key: str) -> tuple[SpeculativeToolGateway, threading.Lock]:
        with self.gateways_lock:
            current = self.gateways.get(session_key)
            if current is None:
                current = (self.gateway_factory(), threading.Lock())
                self.gateways[session_key] = current
            return current


SESSION_HEADERS = ("x-specagent-session-id", "x-session-id")


def request_session_identity(
    request: Mapping[str, Any], headers: Mapping[str, str]
) -> tuple[str, str]:
    """(session key, identity source): a session header, else a hash of the first system (or,
    without one, user) message. Clients that run concurrent sessions with identical opening
    messages must send a session header."""
    lowered = {key.lower(): value for key, value in headers.items()}
    for header in SESSION_HEADERS:
        if lowered.get(header):
            material = str(lowered[header])
            return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], header
    messages = request.get("messages")
    if isinstance(messages, list):
        for role in ("system", "user"):
            for message in messages:
                if not isinstance(message, Mapping) or message.get("role") != role:
                    continue
                content = message.get("content")
                if content is None:
                    continue
                material = (
                    content
                    if isinstance(content, str)
                    else json.dumps(
                        content,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                return (
                    hashlib.sha256(material.encode("utf-8")).hexdigest()[:16],
                    f"{role}_message_hash",
                )
    raise ValueError("request has no session header, system message, or user message")


def decision_event(decision: GatewayDecision, gateway: SpeculativeToolGateway) -> dict[str, Any]:
    """The telemetry record of one decision: verdict, repair, and every model call's usage."""
    verdict = decision.verdict
    return {
        "source": decision.source,
        "horizon": gateway.config.horizon,
        "candidate_length": decision.candidate_length,
        "mutating": decision.mutating,
        "boundary": decision.boundary,
        "verifier_accepted_steps": decision.verifier_accepted_steps,
        "executed_rejected_steps": decision.executed_rejected_steps,
        "accepted_steps": verdict.accepted_steps if verdict is not None else None,
        "rejection_code": (
            verdict.rejection_code.value
            if verdict is not None and verdict.rejection_code
            else None
        ),
        "verdict_reason": verdict.reason or None if verdict is not None else None,
        "verdict_raw": verdict.raw[-4000:] if verdict is not None else None,
        "verdict_diagnostic_status": verdict.diagnostic_status if verdict is not None else None,
        "verification_id": verdict.verification_id if verdict is not None else "",
        "verifier_format_retries": verdict.format_retries if verdict is not None else 0,
        "verifier_format_failure_kind": (
            verdict.format_failure_kind if verdict is not None else ""
        ),
        # Why an emitted replacement was discarded (a guard kind), empty when it committed or
        # none was emitted.
        "replacement_validation_failure_kind": (
            verdict.replacement_validation_failure_kind if verdict is not None else ""
        ),
        "replacement_validation_failure_count": decision.replacement_validation_failure_count,
        "reason_on_full_accept": verdict.reason_on_full_accept if verdict is not None else False,
        "replacement_on_full_accept": (
            verdict.replacement_on_full_accept if verdict is not None else False
        ),
        "discarded_verdict_fields": (
            list(verdict.discarded_verdict_fields) if verdict is not None else None
        ),
        "candidate_steps": list(decision.candidate_steps) or None,
        "correction_action": decision.correction_action,
        "correction_actions": list(decision.correction_actions) or None,
        "correction_action_count": len(decision.correction_actions),
        "corrector_generation_count": decision.corrector_generation_count,
        "correction_tool_call_count": decision.correction_tool_call_count,
        "correction_suppressed_reply": (
            decision.correction_suppressed_reply[:4000]
            if decision.correction_suppressed_reply
            else None
        ),
        "correction_withheld_replays": decision.correction_withheld_replays,
        "repeat_rejected": decision.repeat_rejected,
        # Session-cumulative counters: a failed repair raises before its decision exists.
        "correction_runtime_failure_count": gateway.correction_runtime_failure_count,
        "correction_missing_observation_count": gateway.correction_missing_observation_count,
        "corrector_reask_count": gateway.corrector_reask_count,
        "weak_overflow_resample_count": gateway.weak_overflow_resample_count,
        "strong_overflow_drop_count": gateway.strong_overflow_drop_count,
        "reminted_call_id_count": gateway.reminted_call_id_count,
        "usage": decision.response.get("usage"),
        "model_calls": [
            {
                "phase": call.phase,
                "model": call.model,
                "duration_seconds": call.duration_seconds,
                "transport_attempts": call.transport_attempts,
                "usage": dict(call.usage),
            }
            for call in decision.model_calls
        ],
    }


class Handler(BaseHTTPRequestHandler):
    server: GatewayServer
    server_version = "SpecAgentGateway/0.1"

    def setup(self) -> None:
        super().setup()
        self._stream_started = False
        self._delivered = False
        self._write_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._disconnected = threading.Event()
        self._heartbeat: threading.Thread | None = None

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        LOG.info(format, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _write_sse(self, body: bytes) -> None:
        with self._write_lock:
            if self._disconnected.is_set():
                raise ConnectionAbortedError("client disconnected")
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except OSError:
                self._disconnected.set()
                raise

    def _start_stream(self) -> None:
        self.close_connection = True
        self._stream_started = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._write_sse(b": waiting for verified completion\n\n")

        def heartbeat() -> None:
            # SSE comments detect disconnects without becoming model text.
            while not self._heartbeat_stop.wait(self.server.heartbeat_interval):
                try:
                    self._write_sse(b": keep-alive\n\n")
                except OSError:
                    return

        self._heartbeat = threading.Thread(target=heartbeat, daemon=True)
        self._heartbeat.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join()

    def _deliver(self, decision: GatewayDecision) -> None:
        self._heartbeat_stop.set()
        if self._stream_started:
            self._write_sse(completion_to_sse(decision.response))
        else:
            self._send_json(200, decision.response)
        self._delivered = True

    def _record_event(
        self,
        decision: GatewayDecision,
        gateway: SpeculativeToolGateway,
        session_key: str,
        started: float,
        *,
        delivery_status: str,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "request_id": self._request_id,
            "session_key": session_key,
            "session_identity_source": self._session_identity_source,
            "delivery_status": delivery_status,
            "duration_seconds": time.perf_counter() - started,
            **decision_event(decision, gateway),
        }
        with self.server.telemetry_lock:
            with self.server.telemetry_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _record_trace(self, session_key: str, event: dict[str, Any]) -> None:
        path = self.server.trace_path
        if path is None:
            return
        record = {
            "timestamp": time.time(),
            "request_id": self._request_id,
            "session_key": session_key,
            **event,
        }
        with self.server.telemetry_lock:
            # Full benchmark context is private; never record HTTP headers or API keys.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/healthz":
            self._send_json(200, {"status": "ok"})
            return
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                200,
                {"object": "list", "data": [{"id": "specagent", "object": "model"}]},
            )
            return
        self._send_json(404, {"error": {"message": f"unknown path: {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": f"unknown path: {self.path}"}})
            return
        session_key = "unknown"
        self._session_identity_source = "unknown"
        self._request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(request, dict):
                raise ValueError("completion request must be an object")
            wants_stream = bool(request.get("stream"))
            session_key, self._session_identity_source = request_session_identity(
                request, dict(self.headers.items())
            )
            if self._session_identity_source not in SESSION_HEADERS:
                with self.server.gateways_lock:
                    is_new = session_key not in self.server.gateways
                if is_new:
                    LOG.warning(
                        "session=%s is keyed by %s; send a session header when concurrent "
                        "sessions can share their opening messages",
                        session_key,
                        self._session_identity_source,
                    )
            LOG.info(
                "request start session=%s messages=%d stream=%s",
                session_key,
                len(request.get("messages") or []),
                wants_stream,
            )
            gateway, request_lock = self.server.gateway_for(session_key)
            self.connection.settimeout(30)
            if wants_stream:
                self._start_stream()
            with request_lock:
                try:
                    if self.server.trace_path is not None:
                        gateway.trace = lambda event: self._record_trace(session_key, event)
                        self._record_trace(
                            session_key, {"event": "client_request", "request": request}
                        )
                    decision = gateway.handle(
                        request, deliver=self._deliver, is_cancelled=self._disconnected.is_set
                    )
                except Exception:
                    self._record_event(
                        GatewayDecision(
                            response={}, source="request_failed", model_calls=gateway.model_calls
                        ),
                        gateway,
                        session_key,
                        started,
                        delivery_status="failed",
                    )
                    raise
                finally:
                    gateway.trace = None
                self._record_trace(
                    session_key, {"event": "client_response", "response": decision.response}
                )
                self._record_event(
                    decision, gateway, session_key, started, delivery_status="delivered"
                )
            LOG.info(
                "request ok session=%s source=%s elapsed=%.2fs",
                session_key,
                decision.source,
                time.perf_counter() - started,
            )
        except Exception as error:  # noqa: BLE001
            LOG.error(
                "request failed session=%s elapsed=%.2fs %s: %s\n%s",
                session_key,
                time.perf_counter() - started,
                type(error).__name__,
                error,
                traceback.format_exc(),
            )
            payload = {"error": {"message": str(error), "type": type(error).__name__}}
            self._stop_heartbeat()
            if self._delivered:
                return  # A telemetry failure cannot turn an already sent response into HTTP 502.
            try:
                if self._stream_started:
                    self._write_sse(f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode())
                else:
                    self._send_json(502, payload)
            except OSError:
                pass
        finally:
            self._stop_heartbeat()


def _json_object(value: str) -> dict[str, Any] | None:
    parsed = json.loads(value)
    if parsed is not None and not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object or null")
    return parsed


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _config_fields() -> list[dataclasses.Field]:
    return [
        item
        for item in dataclasses.fields(GatewayConfig)
        if item.name not in ("weak_model", "strong_model")
    ]


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """One flag per GatewayConfig field, defaulting to the MCP-Atlas setting."""
    group = parser.add_argument_group("protocol (defaults: the paper's MCP-Atlas setting)")
    for item in _config_fields():
        flag = "--" + item.name.replace("_", "-")
        default = MCP_ATLAS_SETTING.get(item.name, item.default)
        help_text = FIELD_HELP[item.name]
        kind = str(item.type)
        if kind == "bool":
            group.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=default, help=help_text
            )
        elif kind.startswith("Mapping"):
            group.add_argument(
                flag, type=_json_object, default=default, metavar="JSON", help=help_text
            )
        elif kind.startswith("tuple"):
            group.add_argument(
                flag, type=_csv, default=default, metavar="A,B", help=help_text
            )
        elif kind.startswith("int"):
            group.add_argument(flag, type=int, default=default, help=help_text)
        elif item.name == "task_semantics":
            group.add_argument(flag, choices=TASK_SEMANTICS, default=default, help=help_text)
        else:
            group.add_argument(flag, default=default, help=help_text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m specagent.server",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    endpoints = parser.add_argument_group("endpoints")
    endpoints.add_argument("--weak-base-url", required=True, help="OpenAI-compatible Weak URL")
    endpoints.add_argument(
        "--weak-api-key",
        default=os.environ.get("SPECAGENT_WEAK_API_KEY", "none"),
        help="Weak API key; $SPECAGENT_WEAK_API_KEY when set",
    )
    endpoints.add_argument("--weak-model", required=True)
    endpoints.add_argument("--weak-auth-scheme", choices=VALID_AUTH_SCHEMES, default="bearer")
    endpoints.add_argument("--strong-base-url", required=True, help="OpenAI-compatible Strong URL")
    endpoints.add_argument(
        "--strong-api-key",
        default=os.environ.get("SPECAGENT_STRONG_API_KEY", "none"),
        help="Strong API key; $SPECAGENT_STRONG_API_KEY when set",
    )
    endpoints.add_argument("--strong-model", required=True)
    endpoints.add_argument("--strong-auth-scheme", choices=VALID_AUTH_SCHEMES, default="bearer")
    endpoints.add_argument("--timeout", type=float, default=1800.0, help="Per-call timeout (s)")
    _add_config_arguments(parser)
    serving = parser.add_argument_group("serving")
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=18081)
    serving.add_argument(
        "--telemetry", type=Path, help="JSONL of per-request decisions and model-call usage"
    )
    serving.add_argument(
        "--trace", type=Path, help="Private full request/response JSONL audit log (new file)"
    )
    serving.add_argument(
        "--log-file", type=Path, help="Defaults to the telemetry path with a .log suffix"
    )
    serving.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser


def build_gateway_config(args: argparse.Namespace) -> GatewayConfig:
    values = {item.name: getattr(args, item.name) for item in _config_fields()}
    if values["verifier_reasoning_effort"] in ("off", ""):
        values["verifier_reasoning_effort"] = None
    return GatewayConfig(weak_model=args.weak_model, strong_model=args.strong_model, **values)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    gateway_config = build_gateway_config(args)
    strong_backend = OpenAIBackend(
        args.strong_base_url, args.strong_api_key, args.timeout, auth_scheme=args.strong_auth_scheme
    )
    weak_backend = OpenAIBackend(
        args.weak_base_url, args.weak_api_key, args.timeout, auth_scheme=args.weak_auth_scheme
    )
    backend = RoutedOpenAIBackend(
        weak_model=args.weak_model,
        weak=weak_backend,
        strong_model=args.strong_model,
        strong=strong_backend,
    )
    telemetry = args.telemetry or Path(f"outputs/specagent_gateway_{uuid.uuid4().hex}.jsonl")
    telemetry = telemetry.expanduser().resolve()
    telemetry.parent.mkdir(parents=True, exist_ok=True)
    log_file = (args.log_file or telemetry.with_suffix(".log")).expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    server = GatewayServer((args.host, args.port), Handler)
    server.gateway_factory = lambda: SpeculativeToolGateway(backend, gateway_config)
    server.gateways = {}
    server.gateways_lock = threading.Lock()
    server.telemetry_path = telemetry
    server.telemetry_lock = threading.Lock()
    if args.trace:
        server.trace_path = args.trace.expanduser().resolve()
        server.trace_path.parent.mkdir(parents=True, exist_ok=True)
        if server.trace_path.exists():
            raise SystemExit("--trace must be a new file; use an isolated path for each run")
    LOG.info("SpecAgent gateway listening on http://%s:%d/v1", args.host, args.port)
    LOG.info("config=%s", json.dumps(dataclasses.asdict(gateway_config), default=list))
    LOG.info("weak_endpoint=%s strong_endpoint=%s", weak_backend.base_url, strong_backend.base_url)
    LOG.info("telemetry=%s log_file=%s", telemetry, log_file)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
