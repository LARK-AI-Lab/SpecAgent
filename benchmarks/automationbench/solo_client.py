"""verifiers-bound recording client for the AutomationBench single-model baselines.

Import this ONLY inside an environment with verifiers + automationbench
installed (the AutomationBench environment). The pure accounting core is
``specagent_adapter.py``; this file adds just the wiring:

* ``SpecAgentChatClient`` subclasses ``RetryingOpenAIChatCompletionsClient``
  (required: ``resolve_client`` demands a verifiers ``Client`` instance).
* ``get_native_response`` binds a ``RequestContext`` (task_id / trial_id /
  request_id) via a contextvar before delegating to the retrying loop.
* The wrapped ``client.post`` records one span per *transport attempt* — so
  retries inside the retrying loop each produce a span with a contiguous
  ``transport_attempt_index``, and a failed/cancelled attempt is a span with
  ``status != "ok"`` and unknown usage rather than a silent hole.
* ``authorization`` (optional) overrides the SDK's ``Authorization`` header
  per request for endpoints that use a non-Bearer scheme; ``extra_body`` and
  the benchmark's session-id header are left untouched.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from automationbench.clients import RetryingOpenAIChatCompletionsClient

try:
    from benchmarks.automationbench.specagent_adapter import (
        CURRENT_REQUEST,
        AttemptRecorder,
        parsed_usage_from_response,
    )
except ImportError:
    from specagent_adapter import (  # type: ignore[no-redef]
        CURRENT_REQUEST,
        AttemptRecorder,
        parsed_usage_from_response,
    )


class SpecAgentChatClient(RetryingOpenAIChatCompletionsClient):
    """Chat-completions client that records SpecAgent spans per HTTP attempt."""

    def __init__(
        self,
        client_or_config: Any,
        *,
        recorder: AttemptRecorder,
        authorization: str | None = None,
    ) -> None:
        # Attributes must exist before super().__init__ because setup_client()
        # builds the transport we wrap immediately afterwards.
        self._specagent_recorder = recorder
        self._specagent_authorization = authorization
        super().__init__(client_or_config)
        self._wrap_transport()

    def _wrap_transport(self) -> None:
        api = self._client  # AsyncOpenAI (or a test stub)
        original_post = api.post
        recorder = self._specagent_recorder

        async def recording_post(*args: Any, **kwargs: Any) -> Any:
            ctx = CURRENT_REQUEST.get()
            if ctx is None:
                # A transport call outside a bound request is an anomaly —
                # record it, then pass through (never block the benchmark).
                recorder.anomalies.append(
                    {
                        "kind": "unbound_post",
                        "path": args[0] if args else kwargs.get("path"),
                    }
                )
                return await original_post(*args, **kwargs)
            attempt_index = recorder.next_attempt_index(ctx.request_id)
            t0 = time.monotonic()
            try:
                response = await original_post(*args, **kwargs)
            except asyncio.CancelledError:
                recorder.record_attempt(
                    ctx,
                    attempt_index=attempt_index,
                    duration_seconds=time.monotonic() - t0,
                    status="cancelled",
                )
                raise
            except Exception:
                recorder.record_attempt(
                    ctx,
                    attempt_index=attempt_index,
                    duration_seconds=time.monotonic() - t0,
                    status="failed",
                )
                raise
            duration = time.monotonic() - t0
            parsed, provider_id = parsed_usage_from_response(response)
            headers = getattr(response, "headers", None) or {}
            rank_raw = headers.get("x-specagent-dp-rank")
            if rank_raw is not None:
                try:
                    dp_rank: int | None = int(rank_raw)
                except (TypeError, ValueError):
                    dp_rank = None
                recorder.note_rank(
                    ctx.request_id,
                    dp_rank=dp_rank,
                    route_key=headers.get("x-specagent-route-key"),
                    route_source=headers.get("x-specagent-route-source"),
                )
            recorder.record_attempt(
                ctx,
                attempt_index=attempt_index,
                duration_seconds=duration,
                parsed=parsed,
                status="ok",
                provider_trace_id=provider_id,
            )
            return response

        api.post = recording_post  # type: ignore[method-assign]

    async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
        state = kwargs.get("state")
        ctx = self._specagent_recorder.begin_request(state)
        if self._specagent_authorization is not None:
            headers = dict(kwargs.get("extra_headers") or {})
            headers["Authorization"] = self._specagent_authorization
            kwargs["extra_headers"] = headers
        token = CURRENT_REQUEST.set(ctx)
        try:
            return await super().get_native_response(*args, **kwargs)
        finally:
            CURRENT_REQUEST.reset(token)
