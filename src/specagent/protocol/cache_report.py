"""Treat SGLang-reported cached tokens as prefix-cache compute reuse.

The Qwen3.5 models were self-hosted on SGLang with ``--enable-cache-report`` and the radix
(prefix) cache enabled, so ``prompt_tokens_details.cached_tokens`` counts prefill positions the
server actually skipped. SGLang omits ``prompt_tokens_details`` when the hit count is 0, so on
these models a missing field in a live usage blob is a measured zero. Usage from other providers
is left unchanged: their cached tokens are a billing discount, and missing fields stay unknown.
"""

from __future__ import annotations

from dataclasses import replace

from specagent.protocol.accounting import (
    InferenceSpan,
    ParsedUsage,
    QuantityStatus,
    token_fields_from_usage,
)

SGLANG_PREFIX_CACHE = "sglang_prefix_cache"

# Architecture ids served by SGLang with cache reporting and prefix caching enabled.
SGLANG_CACHE_REPORT_MODELS = frozenset({"Qwen3.5-4B", "Qwen3.5-397B-A17B"})


def serving_has_sglang_cache_report(model_id: str | None) -> bool:
    return model_id is not None and str(model_id).strip() in SGLANG_CACHE_REPORT_MODELS


def apply_sglang_cache_report(
    parsed: ParsedUsage,
    *,
    model_id: str | None,
) -> ParsedUsage:
    """Mark cached tokens as compute reuse for SGLang cache-report models.

    Missing usage is never turned into a measured zero. For a live ``openai_usage`` blob an
    omitted prompt_tokens_details means cached_tokens=0.
    """
    if not serving_has_sglang_cache_report(model_id):
        return parsed
    if parsed.provenance == "missing_usage":
        return parsed
    if parsed.input_status is QuantityStatus.UNKNOWN:
        return parsed
    if parsed.cached_status is QuantityStatus.UNKNOWN:
        # Only a live OpenAI-style blob proves the field was omitted by the server; records
        # rebuilt from truncated logs (which drop usage details) stay unknown.
        if parsed.provenance != "openai_usage":
            return parsed
        return replace(
            parsed,
            cached_input_tokens=0,
            cached_status=QuantityStatus.MEASURED,
            provenance=SGLANG_PREFIX_CACHE,
        )
    if parsed.provenance == SGLANG_PREFIX_CACHE:
        return parsed
    return replace(parsed, provenance=SGLANG_PREFIX_CACHE)


def parsed_usage_from_span(span: InferenceSpan) -> ParsedUsage:
    return ParsedUsage(
        input_tokens=span.input_tokens,
        output_tokens=span.output_tokens,
        cached_input_tokens=span.cached_input_tokens,
        reasoning_tokens=span.reasoning_tokens,
        input_status=span.resolved_input_status,
        output_status=span.resolved_output_status,
        cached_status=span.resolved_cached_status,
        reasoning_status=span.resolved_reasoning_status,
        provenance=span.cache_provenance,
    )


def apply_sglang_cache_report_to_span(
    span: InferenceSpan,
    *,
    model_id: str | None,
) -> InferenceSpan:
    parsed = parsed_usage_from_span(span)
    mapped = apply_sglang_cache_report(
        parsed,
        model_id=model_id,
    )
    if mapped == parsed:
        return span
    return replace(span, **token_fields_from_usage(mapped))
