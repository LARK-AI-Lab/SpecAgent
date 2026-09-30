"""OpenAI-compatible chat client shared by the tau2 agents.

Supports two call shapes:
- complete(): plain completion -> Completion (content is str), used by the
  Strong verifier which returns a JSON verdict string.
- complete_with_tools(): completion with `tools` -> returns content text,
  tool_calls, and usage, used by the Weak drafter, the Strong repair turn,
  and the single-model baselines.

Reasoning handling: sglang with a reasoning parser exposes reasoning in
`reasoning_content`; some backends leave <think>..</think> in content.
We surface both so the actor can pick.

Thinking control: when `thinking_kwarg` is set on the client (e.g. "thinking"
for sglang DeepSeek), a per-call `thinking` flag is sent as
`chat_template_kwargs={thinking_kwarg: bool}` plus `separate_reasoning=true`
in the request body. With `thinking_kwarg` unset no thinking control is sent
and the server default applies.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from specagent.http_auth import openai_init_kwargs, parse_auth_scheme
from specagent.protocol.accounting import ParsedUsage, QuantityStatus


@dataclass(frozen=True, slots=True)
class Completion:
    content: str | Mapping[str, Any]
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    request_snapshot: Mapping[str, Any] | None = None
    raw_response: Mapping[str, Any] | None = None

USAGE_MEASURED = QuantityStatus.MEASURED.value
USAGE_UNKNOWN = QuantityStatus.UNKNOWN.value


def parse_usage(usage: Any) -> ParsedUsage:
    """Parse provider usage without turning missing fields into measured zeros.

    Integer fields stay 0 when unknown so legacy Completion/ToolCallResult
    constructors keep working. Status/provenance tell aggregators not to treat
    those zeros as observed work.
    """
    if usage is None:
        return ParsedUsage(provenance="missing_usage")
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    cached_raw = None
    if prompt_details is not None:
        cached_raw = getattr(prompt_details, "cached_tokens", None)
    reasoning_raw = None
    if completion_details is not None:
        reasoning_raw = getattr(completion_details, "reasoning_tokens", None)
    if reasoning_raw is None:
        reasoning_raw = getattr(usage, "reasoning_tokens", None)

    def _count(raw: Any, *, present: bool) -> tuple[int, QuantityStatus]:
        if not present or raw is None:
            return 0, QuantityStatus.UNKNOWN
        return int(raw or 0), QuantityStatus.MEASURED

    input_tokens, input_status = _count(prompt, present=prompt is not None)
    output_tokens, output_status = _count(
        completion, present=completion is not None
    )
    cache_present = prompt_details is not None and cached_raw is not None
    cached_tokens, cached_status = _count(cached_raw, present=cache_present)
    reasoning_tokens, reasoning_status = _count(
        reasoning_raw,
        present=reasoning_raw is not None,
    )
    return ParsedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        input_status=input_status,
        output_status=output_status,
        cached_status=cached_status,
        reasoning_status=reasoning_status,
        provenance="openai_usage",
    )


@dataclass
class ToolCallResult:
    content: str
    reasoning: str | None
    tool_calls: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    request_snapshot: Mapping[str, Any] | None = None
    raw_response: Mapping[str, Any] | None = None
    usage_status: str = USAGE_MEASURED
    usage_provenance: str = "openai_usage"


def _usage_counts(usage: Any) -> tuple[int, int, int, int]:
    """Return (input, output, cached_input, reasoning) tokens from an OpenAI usage object.

    Backends disagree about where reasoning tokens live: the OpenAI schema puts
    them in ``completion_tokens_details.reasoning_tokens``, while some sglang
    builds expose a top-level
    ``usage.reasoning_tokens``. Read both and take whichever is populated --
    reading only one silently reports 0 for the other family, and reasoning
    tokens are part of the cost metric we report.

    Missing usage still returns integer zeros for legacy callers; use
    ``parse_usage`` to distinguish unknown from measured zero.
    """
    parsed = parse_usage(usage)
    return (
        parsed.input_tokens,
        parsed.output_tokens,
        parsed.cached_input_tokens,
        parsed.reasoning_tokens,
    )


def _attach_usage(
    raw_response: Mapping[str, Any] | None, parsed: ParsedUsage
) -> dict[str, Any] | None:
    if raw_response is None:
        return {"_specagent_usage": parsed.to_dict()}
    record = dict(raw_response)
    record["_specagent_usage"] = parsed.to_dict()
    return record


class OpenAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 600.0,
        thinking_kwarg: str | None = None,
        auth_scheme: str = "bearer",
    ) -> None:
        self.auth_scheme = parse_auth_scheme(auth_scheme)
        self.client = AsyncOpenAI(
            **openai_init_kwargs(base_url, api_key, timeout, scheme=self.auth_scheme)
        )
        self.model = model
        # Name of the chat_template_kwargs key that toggles thinking for this
        # backend (e.g. "thinking" for sglang DeepSeek). None = no thinking
        # control sent (rely on the server default).
        self.thinking_kwarg = thinking_kwarg

    def _extra_body(
        self, thinking: bool | None, extra_body: Mapping[str, Any] | None
    ) -> dict | None:
        eb: dict[str, Any] = dict(extra_body) if extra_body else {}
        if thinking is not None and self.thinking_kwarg:
            ctk = eb.get("chat_template_kwargs")
            if not isinstance(ctk, dict):
                ctk = {}
            ctk[self.thinking_kwarg] = thinking
            eb["chat_template_kwargs"] = ctk
            # separate_reasoning must be a top-level field so reasoning lands in
            # `reasoning_content` instead of being inlined into `content`.
            eb["separate_reasoning"] = True
        return eb or None

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_output_tokens: int | None = None,
        reasoning_budget: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        thinking: bool | None = None,
        temperature: float | None = None,
    ) -> Completion:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_output_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        eb = self._extra_body(thinking, extra_body)
        if eb:
            kwargs["extra_body"] = eb
        resp = await self.client.chat.completions.create(**kwargs)
        dur = time.perf_counter() - t0
        msg = resp.choices[0].message
        content = msg.content or ""
        # Reasoning models may put the final answer in
        # reasoning_content and leave content empty when the output budget is
        # tight; fall back to reasoning_content so the verifier still gets a
        # parseable string.
        if not content:
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                content = rc
        parsed = parse_usage(resp.usage)
        input_tokens, output_tokens, cached, reasoning_tokens = (
            parsed.input_tokens,
            parsed.output_tokens,
            parsed.cached_input_tokens,
            parsed.reasoning_tokens,
        )
        raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else None
        return Completion(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_seconds=dur,
            cached_input_tokens=cached,
            reasoning_tokens=reasoning_tokens,
            request_snapshot=deepcopy(kwargs),
            raw_response=_attach_usage(raw, parsed),
        )

    async def complete_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_output_tokens: int | None = None,
        reasoning_budget: int | None = None,
        thinking: bool | None = None,
    ) -> ToolCallResult:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            "max_tokens": max_output_tokens,
        }
        eb = self._extra_body(thinking, None)
        if eb:
            kwargs["extra_body"] = eb
        resp = await self.client.chat.completions.create(**kwargs)
        dur = time.perf_counter() - t0
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None)
        raw_calls = msg.tool_calls or []
        if not reasoning and content:
            # fall back to <think>..</think> extraction
            if "</think>" in content:
                reasoning = content.split("</think>")[0]
                if reasoning.startswith("<think>"):
                    reasoning = reasoning[len("<think>") :]
        tool_calls = []
        for tc in raw_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )
        parsed = parse_usage(resp.usage)
        raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else None
        return ToolCallResult(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            duration_seconds=dur,
            cached_input_tokens=parsed.cached_input_tokens,
            reasoning_tokens=parsed.reasoning_tokens,
            request_snapshot=deepcopy(kwargs),
            raw_response=_attach_usage(raw, parsed),
            usage_status=parsed.status.value,
            usage_provenance=parsed.provenance,
        )
