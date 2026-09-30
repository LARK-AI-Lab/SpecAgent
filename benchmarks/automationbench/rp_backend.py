"""CompletionBackend + transport telemetry for the AutomationBench SpecAgent lane.

Pure-python (stdlib + ``specagent.protocol.*`` + ``specagent_adapter``): the
HTTP transport is an injectable callable so the repo test suite exercises the
routing/telemetry path without httpx, verifiers, or live endpoints.

Design:

* ``RecordingRoutedBackend`` implements the ``CompletionBackend`` protocol the
  ``SpeculativeToolGateway`` calls. It routes each ``complete()`` to the
  weak/strong endpoint by model name and buffers per-attempt telemetry in
  ``calls`` — it does NOT write spans itself because the transport layer
  cannot see the gateway's phase labels.
* The SpecAgent client drains this request's calls after each ``handle()`` and
  records one InferenceSpan per transport attempt with the role/phase each call
  was tagged with (``drain_calls`` below).
* Retries inside ``complete()`` produce contiguous attempt indices under one
  derived ``request_id``, preserving the missing-attempt detection contract.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from specagent.protocol.accounting import ParsedUsage
from specagent.protocol.resource_estimate import (
    _gateway_role_phase,
    parsed_usage_from_provider_blob,
)

try:
    from benchmarks.automationbench.specagent_adapter import (
        CURRENT_REQUEST,
        AttemptRecorder,
        RequestContext,
    )
except ImportError:
    from specagent_adapter import (  # type: ignore[no-redef]
        CURRENT_REQUEST,
        AttemptRecorder,
        RequestContext,
    )

# The SpecAgent client wraps the gateway's ``_complete`` funnel to publish the
# in-flight phase here; the transport layer then tags each buffered call
# with its own phase, so attribution never depends on an index join with
# ``gateway.model_calls``.
CURRENT_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "automationbench_rp_phase", default=None
)

# Per-request transport buffer: the client installs a fresh list before each
# gateway.handle() so drain_calls sees exactly this request's calls. Slicing
# the shared ``self.calls`` buffer under concurrent trials counts foreign
# calls in n_transport_calls and re-records their spans under a different
# request id.
CURRENT_CALLS: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "automationbench_rp_calls", default=None
)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
SESSION_HEADER = "x-specagent-session-id"
RANK_HEADERS = {
    "rank": "x-specagent-dp-rank",
    "route_key": "x-specagent-route-key",
    "route_source": "x-specagent-route-source",
}


@dataclass(frozen=True)
class RoleEndpoint:
    """One OpenAI-compatible upstream for a gateway role."""

    base_url: str
    api_key: str | None = None
    # Full Authorization header value for non-Bearer schemes (e.g. raw
    # auth); when None the backend sends ``Bearer <api_key>``.
    authorization: str | None = None
    # Endpoint-level request overlay (sglang chat_template_kwargs etc.);
    # applied before the gateway's per-phase extra_body.
    extra_body: Mapping[str, Any] | None = None


@dataclass
class PostResult:
    status: int
    body: dict[str, Any] | None
    headers: Mapping[str, str]


PostFn = Callable[[str, dict[str, Any], dict[str, str]], PostResult]


def _default_post(url: str, body: dict[str, Any], headers: dict[str, str]) -> PostResult:
    import httpx  # lazy: only the live lane needs it

    resp = httpx.post(
        url, json=body, headers=headers, timeout=httpx.Timeout(600.0, connect=30.0)
    )
    try:
        parsed = resp.json()
    except Exception:  # noqa: BLE001
        parsed = None
    return PostResult(
        status=resp.status_code,
        body=parsed if isinstance(parsed, dict) else None,
        headers=dict(resp.headers),
    )


def _lower_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (headers or {}).items()}


class RecordingRoutedBackend:
    """Route gateway model calls to role endpoints; buffer attempt telemetry."""

    def __init__(
        self,
        *,
        weak_model: str,
        weak: RoleEndpoint,
        strong_model: str,
        strong: RoleEndpoint,
        post_fn: PostFn | None = None,
        max_attempts: int = 40,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        if weak_model == strong_model and weak is not strong:
            raise ValueError(
                "different endpoints require distinct weak and strong model names"
            )
        self.endpoints = {weak_model: weak, strong_model: strong}
        self.post_fn = post_fn or _default_post
        self.max_attempts = max(1, max_attempts)
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        # One entry per complete() call, appended in call order — the client
        # zips these with gateway.model_calls (same order) for attribution.
        self.calls: list[dict[str, Any]] = []

    def _endpoint(self, model: str) -> RoleEndpoint:
        try:
            return self.endpoints[model]
        except KeyError as error:
            raise ValueError(
                f"no upstream endpoint configured for model {model!r}"
            ) from error

    def prepare_request(
        self,
        request: Mapping[str, Any],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self._endpoint(model)
        body = dict(request)
        body["model"] = model
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        # AutomationBench flattens extra_body into the request root (sglang
        # style); the gateway's per-phase overlay wins over endpoint defaults.
        for overlay in (cfg.extra_body, extra_body):
            if overlay:
                body.update(overlay)
        return body

    def _headers(self, cfg: RoleEndpoint) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if cfg.authorization is not None:
            headers["authorization"] = cfg.authorization
        elif cfg.api_key:
            headers["authorization"] = f"Bearer {cfg.api_key}"
        ctx = CURRENT_REQUEST.get()
        if ctx is not None:
            headers[SESSION_HEADER] = ctx.trial_id
        return headers

    def complete(
        self,
        request: Mapping[str, Any],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        cfg = self._endpoint(model)
        body = self.prepare_request(
            request,
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        headers = self._headers(cfg)
        call: dict[str, Any] = {
            "model": model,
            "phase": CURRENT_PHASE.get() or "",
            "attempts": [],
        }
        self.calls.append(call)
        request_buffer = CURRENT_CALLS.get()
        if request_buffer is not None:
            request_buffer.append(call)
        attempts: list[dict[str, Any]] = call["attempts"]

        for attempt in range(1, self.max_attempts + 1):
            if is_cancelled is not None and is_cancelled():
                attempts.append(
                    {"index": attempt, "duration_seconds": 0.0, "status": "cancelled"}
                )
                raise ConnectionAbortedError("client cancelled before upstream call")
            t0 = time.monotonic()
            try:
                result = self.post_fn(url, body, headers)
            except Exception as error:  # noqa: BLE001 — transport family
                attempts.append(
                    {
                        "index": attempt,
                        "duration_seconds": time.monotonic() - t0,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if attempt >= self.max_attempts:
                    raise RuntimeError(f"upstream transport error: {error}") from error
                self._sleep_before_retry(attempt, None, is_cancelled)
                continue
            duration = time.monotonic() - t0
            if result.status == 200 and result.body is not None:
                attempts.append(
                    {
                        "index": attempt,
                        "duration_seconds": duration,
                        "status": "ok",
                        "usage": result.body.get("usage"),
                        "provider_id": result.body.get("id"),
                        "headers": result.headers,
                    }
                )
                return result.body
            detail = ""
            if isinstance(result.body, dict):
                detail = str(result.body.get("error") or result.body)[:500]
            attempts.append(
                {
                    "index": attempt,
                    "duration_seconds": duration,
                    "status": "failed",
                    "error": f"HTTP {result.status}: {detail}",
                }
            )
            if result.status not in RETRYABLE_STATUS or attempt >= self.max_attempts:
                raise RuntimeError(f"upstream HTTP {result.status}: {detail}")
            retry_after = _lower_headers(result.headers).get("retry-after")
            self._sleep_before_retry(attempt, retry_after, is_cancelled)
        raise RuntimeError("unreachable")

    def _sleep_before_retry(
        self,
        attempt: int,
        retry_after: str | None,
        is_cancelled: Callable[[], bool] | None,
    ) -> None:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except (TypeError, ValueError):
            delay = 0.0
        if delay <= 0.0:
            delay = min(
                self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay
            )
        if is_cancelled is not None and is_cancelled():
            raise ConnectionAbortedError("client disconnected before upstream retry")
        time.sleep(delay)


def tag_gateway_phases(gateway: Any) -> None:
    """Observer-only wrap of the gateway's ``_complete`` funnel.

    ``_complete`` is the single path every role call takes and receives the
    phase name; publish it on CURRENT_PHASE so the transport tags each
    buffered call with its own phase, independent of ``model_calls``.
    """
    original_complete = gateway._complete

    def _phase_tagged(request: Any, *, phase: str, **kw: Any) -> Any:
        phase_token = CURRENT_PHASE.set(phase)
        try:
            return original_complete(request, phase=phase, **kw)
        finally:
            CURRENT_PHASE.reset(phase_token)

    gateway._complete = _phase_tagged


def drain_calls(
    recorder: AttemptRecorder,
    ctx: RequestContext,
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record one InferenceSpan per buffered transport attempt.

    Each call entry carries the gateway phase captured from CURRENT_PHASE at
    transport time. Returns the decision provenance row for
    ``rp_decisions.jsonl``.
    """
    phases: list[str | None] = []
    for i, call in enumerate(calls):
        phase_name = str(call.get("phase") or "")
        phases.append(phase_name or None)
        role, phase, intent = _gateway_role_phase(phase_name)
        sub_request_id = f"{ctx.request_id}:{phase_name or 'call'}:{i + 1}"
        for att in call.get("attempts", []):
            usage = att.get("usage")
            parsed = (
                parsed_usage_from_provider_blob(usage)
                if usage is not None
                else ParsedUsage(provenance="missing_usage")
            )
            recorder.record_attempt(
                ctx,
                attempt_index=int(att.get("index") or 0),
                duration_seconds=float(att.get("duration_seconds") or 0.0),
                parsed=parsed,
                status=str(att.get("status") or "failed"),
                provider_trace_id=att.get("provider_id"),
                role=role,
                phase=phase,
                intent=intent,
                request_id=sub_request_id,
            )
            headers = _lower_headers(att.get("headers"))
            if RANK_HEADERS["rank"] in headers:
                rank_raw = headers[RANK_HEADERS["rank"]]
                try:
                    dp_rank: int | None = int(rank_raw)
                except (TypeError, ValueError):
                    dp_rank = None
                recorder.note_rank(
                    sub_request_id,
                    dp_rank=dp_rank,
                    route_key=headers.get(RANK_HEADERS["route_key"]),
                    route_source=headers.get(RANK_HEADERS["route_source"]),
                )
    return {
        "request_id": ctx.request_id,
        "trial_id": ctx.trial_id,
        "task_id": ctx.task_id,
        "phases": phases,
        "n_transport_calls": len(calls),
    }
