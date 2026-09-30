"""verifiers-bound client for the AutomationBench SpecAgent lane.

Import this ONLY inside an environment with verifiers + automationbench +
openai installed (the AutomationBench environment).

Instead of posting each rollout turn directly, ``get_native_response`` routes
the chat request through a per-trajectory ``SpeculativeToolGateway`` (Weak
drafting, Strong verification, Strong repair) and hands the committed
``decision.response`` back to AutomationBench, which keeps owning the tool
loop, environment state, and rubric.

Accounting: the transport layer cannot see gateway phase labels, so the
``RecordingRoutedBackend`` buffers per-attempt telemetry; after ``handle()``
returns (or raises) this client drains it and zips with
``gateway.model_calls`` (same order) to record phase-attributed spans via
``rp_backend.drain_calls``. Provenance for each boundary lands in
``self.decisions`` → ``rp_decisions.jsonl``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from automationbench.clients import RetryingOpenAIChatCompletionsClient
from openai.types.chat import ChatCompletion

from specagent.runtime.gateway import GatewayConfig, SpeculativeToolGateway

try:
    from benchmarks.automationbench.rp_backend import (
        CURRENT_CALLS,
        RecordingRoutedBackend,
        drain_calls,
        tag_gateway_phases,
    )
    from benchmarks.automationbench.specagent_adapter import (
        CURRENT_REQUEST,
        AttemptRecorder,
    )
except ImportError:
    from rp_backend import (  # type: ignore[no-redef]
        CURRENT_CALLS,
        RecordingRoutedBackend,
        drain_calls,
        tag_gateway_phases,
    )
    from specagent_adapter import (  # type: ignore[no-redef]
        CURRENT_REQUEST,
        AttemptRecorder,
    )

try:
    from benchmarks.automationbench.decision_fields import decision_fields
except ImportError:
    from decision_fields import decision_fields  # type: ignore[no-redef]


class SpecAgentGatewayClient(RetryingOpenAIChatCompletionsClient):
    """Chat-completions client that runs turns through the SpecAgent gateway."""

    def __init__(
        self,
        client_or_config: Any,
        *,
        recorder: AttemptRecorder,
        backend: RecordingRoutedBackend,
        gateway_config: GatewayConfig,
    ) -> None:
        self._recorder = recorder
        self._backend = backend
        self._gateway_config = gateway_config
        self._gateways: dict[str, SpeculativeToolGateway] = {}
        # One provenance row per get_native_response → rp_decisions.jsonl.
        self.decisions: list[dict[str, Any]] = []
        super().__init__(client_or_config)

    def _gateway_for(self, trial_id: str) -> SpeculativeToolGateway:
        """One gateway (protocol state) per trajectory, so no state crosses trials."""
        gateway = self._gateways.get(trial_id)
        if gateway is None:
            gateway = SpeculativeToolGateway(self._backend, self._gateway_config)
            tag_gateway_phases(gateway)
            self._gateways[trial_id] = gateway
        return gateway

    async def get_native_response(
        self,
        prompt: Any,
        model: str,
        sampling_args: Any,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        state = kwargs.get("state")
        ctx = self._recorder.begin_request(state)
        gateway = self._gateway_for(ctx.trial_id)
        body: dict[str, Any] = {"model": model, "messages": prompt}
        if tools:
            body["tools"] = tools
        if sampling_args:
            merged = dict(sampling_args)
            # A single role's extra_body must not leak onto the other role;
            # per-role overlays are applied per phase in the backend.
            merged.pop("extra_body", None)
            body.update(merged)
        token = CURRENT_REQUEST.set(ctx)
        # Per-request drain buffer: under --max-concurrent N, parallel trials
        # share self._backend.calls; a suffix slice would count foreign calls
        # here and duplicate their spans. The contextvar propagates into the
        # to_thread worker, so complete() appends to exactly this list.
        request_calls: list[dict[str, Any]] = []
        calls_token = CURRENT_CALLS.set(request_calls)
        decision = None
        error: Exception | None = None
        try:
            decision = await asyncio.to_thread(gateway.handle, body)
        except Exception as exc:  # noqa: BLE001 — drain telemetry, then raise
            error = exc
        finally:
            CURRENT_REQUEST.reset(token)
            CURRENT_CALLS.reset(calls_token)
            row = drain_calls(
                self._recorder,
                ctx,
                request_calls,
            )
            if decision is not None:
                row.update(self._decision_fields(decision))
            if error is not None:
                row["error"] = f"{type(error).__name__}: {error}"
            self.decisions.append(row)
        if error is not None:
            raise error
        return ChatCompletion.model_validate(decision.response)

    @staticmethod
    def _decision_fields(decision: Any) -> dict[str, Any]:
        """Provenance fields merged into the drained decision row."""
        return decision_fields(decision)
