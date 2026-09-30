"""Behaviour of the FLOPs and HBM-read estimators (exact values live in the pinned test)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from specagent.protocol.accounting import (
    InferencePhase,
    InferenceSpan,
    QuantityStatus,
    ResourceLedger,
    merge_quantities,
)
from specagent.protocol.estimator_ids import (
    FLOPS_ESTIMATOR_ID,
    HBM_ESTIMATOR_ID,
    LEDGER_NO_ESTIMATOR,
)
from specagent.protocol.flops import (
    FMA,
    _full_attn_scores_from_pairs,
    _mlp_tokens,
    causal_prefill_pairs,
    estimate_request_flops,
    estimate_span_flops,
    load_arch,
    supported_models,
)
from specagent.protocol.hbm import (
    _v3_weight_read_bytes,
    estimate_request_hbm,
    estimate_span_hbm,
    hbm_read_interval,
    kv_bytes_per_token,
    recurrent_state_bytes,
    stack_weight_elems,
    unique_routed_expert_bounds,
    weight_dtype_bytes,
)
from specagent.protocol.resource_estimate import (
    enrich_span,
    resolve_architecture_id,
    serving_spec_from_model_name,
)
from specagent.protocol.types import ModelRole

WEAK = "Qwen3.5-4B"
QWEN_MOE = "Qwen3.5-397B-A17B"
DSV4 = "DeepSeek-V4-Flash-0731"
GLM = "GLM-5"


def _dense_param_flops(n_params: int, tokens: int) -> int:
    """2 * params * tokens: the naive count the estimator must not reduce to."""
    return FMA * n_params * tokens


def _span(**kwargs) -> InferenceSpan:
    payload = {
        "role": ModelRole.WEAK,
        "phase": InferencePhase.DRAFT,
        "input_tokens": 100,
        "output_tokens": 8,
        "duration_seconds": 0.1,
        "usage_status": QuantityStatus.MEASURED,
    }
    payload.update(kwargs)
    return InferenceSpan(**payload)


@pytest.fixture(scope="module")
def weak():
    return load_arch(WEAK)


@pytest.fixture(scope="module")
def moe():
    return load_arch(QWEN_MOE)


# Architecture configs -------------------------------------------------------------------


def test_only_paper_models_are_supported():
    assert set(supported_models()) == {WEAK, QWEN_MOE, DSV4, GLM}
    for name in ("Kimi-K2.6", "Qwen3.5-9B", "Qwen3.5-397B-A17B-FP8"):
        with pytest.raises(KeyError):
            load_arch(name)


def test_qwen_layouts_match_public_configs(weak, moe):
    assert (weak.num_hidden_layers, weak.n_full_attn, weak.n_linear_attn) == (32, 8, 24)
    assert weak.hidden_size == 2560 and not weak.is_moe and weak.tie_word_embeddings
    assert (moe.num_hidden_layers, moe.n_full_attn, moe.n_linear_attn) == (60, 15, 45)
    assert (moe.num_experts, moe.num_experts_per_tok) == (512, 10) and moe.is_moe
    assert weak.hf_repo == "Qwen/Qwen3.5-4B" and moe.hf_repo == "Qwen/Qwen3.5-397B-A17B"
    assert weight_dtype_bytes(weak) == 2 and weight_dtype_bytes(moe) == 2


def test_deepseek_v4_layout():
    arch = load_arch(DSV4)
    assert arch.is_deepseek_v4 and arch.is_moe
    assert (arch.num_hidden_layers, arch.num_experts, arch.num_experts_per_tok) == (43, 256, 6)
    assert arch.expert_dtype == "fp4"
    assert len(arch.compress_ratios) == 43
    assert sorted(set(arch.compress_ratios)) == [0, 4, 128]


def test_glm5_layout():
    arch = load_arch(GLM)
    assert arch.is_deepseek_v3 and arch.index_topk == 2048
    assert arch.head_dim == arch.qk_nope_head_dim + arch.rope_head_dim
    assert (arch.num_hidden_layers, arch.first_k_dense_replace) == (78, 3)


def test_served_names_resolve():
    assert resolve_architecture_id("Qwen/Qwen3.5-4B") == WEAK
    assert resolve_architecture_id("Qwen3.5-397B-A17B-BF16") == QWEN_MOE
    assert resolve_architecture_id("deepseek-v4-flash-0731") == DSV4
    assert resolve_architecture_id("glm-5") == GLM
    assert resolve_architecture_id("definitely-not-a-model") is None


# FLOPs ----------------------------------------------------------------------------------


def test_moe_is_not_dense_total_or_active_param_linear(moe):
    got = estimate_request_flops(moe, input_tokens=128, output_tokens=32).total
    total_tokens = 160
    assert got < _dense_param_flops(397_000_000_000, total_tokens)
    assert got != _dense_param_flops(17_000_000_000, total_tokens)
    assert got > _mlp_tokens(moe, total_tokens) * moe.num_hidden_layers


def test_weak_is_much_cheaper_than_strong(weak, moe):
    kwargs = {"input_tokens": 64, "output_tokens": 16}
    w = estimate_request_flops(weak, **kwargs).total
    s = estimate_request_flops(moe, **kwargs).total
    assert s / w > 3.0


def test_causal_prefill_pairs(weak):
    assert causal_prefill_pairs(4, 0) == 10
    assert causal_prefill_pairs(2, 3) == 9
    causal = _full_attn_scores_from_pairs(weak, causal_prefill_pairs(4, 0))
    rectangular = _full_attn_scores_from_pairs(weak, 16)
    assert rectangular / causal == pytest.approx(1.6)


def test_full_attn_prefill_is_superlinear(weak):
    short = estimate_request_flops(weak, input_tokens=64, output_tokens=0)
    long = estimate_request_flops(weak, input_tokens=256, output_tokens=0)
    assert 4.0 < long.prefill / short.prefill < 16.0


def test_first_token_comes_from_prefill_logits(weak):
    one = estimate_request_flops(weak, input_tokens=4, output_tokens=1)
    three = estimate_request_flops(weak, input_tokens=4, output_tokens=3)
    assert (one.decode_tokens, one.n_logits) == (0, 1)
    assert (three.decode_tokens, three.n_logits) == (2, 3)
    assert one.prefill == three.prefill
    assert one.lm_head * 3 == three.lm_head


def test_deepseek_v4_is_moe_scale_and_sparse():
    arch = load_arch(DSV4)
    got = estimate_request_flops(arch, input_tokens=128, output_tokens=32).total
    active = _dense_param_flops(17_000_000_000, 160)
    assert active / 4 < got < active * 4
    small = estimate_request_flops(arch, input_tokens=1024, output_tokens=1).total
    big = estimate_request_flops(arch, input_tokens=8192, output_tokens=1).total
    assert big / small < 32.0


def test_billing_cache_is_interval_not_skip(weak):
    span = _span(cached_input_tokens=80, cache_provenance="openai_usage")
    q = estimate_span_flops(span, WEAK)
    full = estimate_request_flops(weak, input_tokens=100, output_tokens=8).total
    skipped = estimate_request_flops(
        weak, input_tokens=100, output_tokens=8, compute_prefix_cached_tokens=80
    ).total
    assert q.status is QuantityStatus.INCOMPLETE and q.value is None
    assert "billing_cache_not_compute_reuse" in q.provenance
    assert (q.observed_lower, q.observed_upper, q.observed_subtotal) == (skipped, full, skipped)
    assert skipped < full


def test_compute_reuse_skips_prefix(weak):
    q = estimate_span_flops(
        _span(cached_input_tokens=80, cache_provenance="sglang_prefix_cache"), WEAK
    )
    skipped = estimate_request_flops(
        weak, input_tokens=100, output_tokens=8, compute_prefix_cached_tokens=80
    )
    assert q.status is QuantityStatus.ESTIMATED and q.value == skipped.total
    assert skipped.prefill_tokens == 20


def test_unknown_cache_is_interval(weak):
    q = estimate_span_flops(_span(cached_tokens_status=QuantityStatus.UNKNOWN), WEAK)
    full = estimate_request_flops(weak, input_tokens=100, output_tokens=8).total
    all_skipped = estimate_request_flops(
        weak, input_tokens=100, output_tokens=8, compute_prefix_cached_tokens=100
    ).total
    assert q.status is QuantityStatus.INCOMPLETE
    assert (q.observed_lower, q.observed_upper) == (all_skipped, full)


def test_explicit_cache_zero_is_point(weak):
    q = estimate_span_flops(_span(cached_tokens_status=QuantityStatus.MEASURED), WEAK)
    assert q.status is QuantityStatus.ESTIMATED
    assert q.value == estimate_request_flops(weak, input_tokens=100, output_tokens=8).total
    assert "cache=no_compute_reuse" in q.provenance


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_tokens": 0, "output_tokens_status": QuantityStatus.UNKNOWN},
        {"output_tokens": 0, "output_tokens_status": QuantityStatus.INCOMPLETE},
        {"usage_status": QuantityStatus.INCOMPLETE},
        {"input_tokens": 0, "output_tokens": 0, "usage_status": QuantityStatus.UNKNOWN},
    ],
)
def test_unknown_tokens_are_never_zero_filled(kwargs):
    assert estimate_span_flops(_span(**kwargs), WEAK).status is QuantityStatus.UNKNOWN


def test_unknown_tokens_give_unknown_hbm():
    for kwargs in (
        {"output_tokens": 0, "output_tokens_status": QuantityStatus.UNKNOWN},
        {"input_tokens": 0, "output_tokens": 0, "usage_status": QuantityStatus.UNKNOWN},
    ):
        read, write = estimate_span_hbm(_span(**kwargs), WEAK)
        assert read.status is QuantityStatus.UNKNOWN
        assert write.status is QuantityStatus.UNKNOWN


def test_unsupported_model_is_unknown():
    assert estimate_span_flops(_span(), "deepseek-v4-flash").status is QuantityStatus.UNKNOWN
    assert estimate_span_hbm(_span(), "Kimi-K2.6")[0].status is QuantityStatus.UNKNOWN


def test_reasoning_is_part_of_output_tokens():
    base = estimate_span_flops(_span(output_tokens=10, reasoning_tokens=7), WEAK).value
    more = estimate_span_flops(_span(output_tokens=17, reasoning_tokens=7), WEAK).value
    assert base < more


def test_intervals_and_points_aggregate():
    billing = _span(cached_input_tokens=80, cache_provenance="openai_usage", request_id="a")
    billing2 = _span(
        input_tokens=40,
        output_tokens=4,
        cached_input_tokens=10,
        cache_provenance="openai_usage",
        request_id="b",
    )
    point_span = _span(
        input_tokens=20,
        output_tokens=2,
        cached_tokens_status=QuantityStatus.MEASURED,
        request_id="c",
    )
    q1, q2, point = (estimate_span_flops(s, WEAK) for s in (billing, billing2, point_span))
    ledger = ResourceLedger()
    for span, q in ((billing, q1), (billing2, q2), (point_span, point)):
        ledger.add(replace(span, flops=q))
    summary = ledger.summarize()
    lower = q1.observed_lower + q2.observed_lower + point.value
    upper = q1.observed_upper + q2.observed_upper + point.value
    assert summary.flops.status is QuantityStatus.INCOMPLETE and summary.flops.value is None
    assert (summary.flops.observed_lower, summary.flops.observed_upper) == (lower, upper)
    assert summary.flops.observed_estimated == point.value
    merged = merge_quantities([q1, q2, point], observed_name="flops")
    assert (merged.observed_lower, merged.observed_upper) == (lower, upper)


def test_estimator_ids_reach_the_ledger_summary():
    span = _span(input_tokens=16, output_tokens=2)
    ledger = ResourceLedger()
    ledger.add(span)
    assert ledger.summarize().estimator_version == LEDGER_NO_ESTIMATOR
    read, write = estimate_span_hbm(span, WEAK)
    filled = ResourceLedger()
    filled.add(
        replace(span, flops=estimate_span_flops(span, WEAK), hbm_read_bytes=read,
                hbm_write_bytes=write)
    )
    version = filled.summarize().estimator_version
    assert FLOPS_ESTIMATOR_ID in version and HBM_ESTIMATOR_ID in version


# HBM ------------------------------------------------------------------------------------


def test_hybrid_kv_only_on_full_attention_layers(weak):
    hybrid = kv_bytes_per_token(weak, 2) * weak.n_full_attn
    assert hybrid < kv_bytes_per_token(weak, 2) * weak.num_hidden_layers
    assert recurrent_state_bytes(weak, 4) > recurrent_state_bytes(weak, 2)


def test_moe_active_weights_are_not_all_experts(moe):
    assert stack_weight_elems(moe) < stack_weight_elems(moe, routed_experts=moe.num_experts)


def test_decode_adds_weight_passes_and_write_is_unknown(weak):
    prefill = estimate_request_hbm(weak, input_tokens=32, output_tokens=1)
    decoded = estimate_request_hbm(weak, input_tokens=32, output_tokens=9)
    assert decoded.read > prefill.read and decoded.write > prefill.write
    read, write = estimate_span_hbm(_span(input_tokens=32, gpu_count=16), WEAK)
    assert read.value == estimate_request_hbm(weak, input_tokens=32, output_tokens=8).read
    assert write.status is QuantityStatus.UNKNOWN and write.value is None
    assert "paper_excludes_hbm_write" in write.provenance
    assert "no_tp" in read.provenance


def test_gated_delta_prefill_state_is_per_chunk(weak):
    one_chunk = estimate_request_hbm(weak, input_tokens=64, output_tokens=0)
    two_chunks = estimate_request_hbm(weak, input_tokens=65, output_tokens=0)
    per_touch = recurrent_state_bytes(weak, 4) * weak.n_linear_attn
    assert one_chunk.state_read == per_touch
    assert two_chunks.state_read == 2 * per_touch


def test_moe_prefill_read_is_an_expert_interval(moe):
    read, write = estimate_span_hbm(
        _span(role=ModelRole.STRONG, phase=InferencePhase.VERIFICATION, input_tokens=40),
        QWEN_MOE,
    )
    assert read.status is QuantityStatus.INCOMPLETE and read.value is None
    assert read.observed_lower < read.observed_upper
    assert "moe_prefill_unique_experts=" in read.provenance
    assert write.status is QuantityStatus.UNKNOWN
    assert unique_routed_expert_bounds(moe, 40) == (10, 400)
    assert unique_routed_expert_bounds(moe, 1000) == (10, 512)
    assert (read.observed_lower, read.observed_upper) == hbm_read_interval(
        moe, input_tokens=40, output_tokens=8
    )


def test_dense_interval_is_a_point(weak):
    lo, hi = hbm_read_interval(weak, input_tokens=100, output_tokens=8)
    assert lo == hi == estimate_request_hbm(weak, input_tokens=100, output_tokens=8).read


def test_unknown_cache_hbm_is_interval():
    read, _ = estimate_span_hbm(_span(cached_tokens_status=QuantityStatus.UNKNOWN), WEAK)
    assert read.status is QuantityStatus.INCOMPLETE
    assert read.observed_lower < read.observed_upper


def test_billing_cache_hbm_interval_survives_summarize():
    span = _span(cached_input_tokens=80, cache_provenance="openai_usage", request_id="r")
    read, write = estimate_span_hbm(span, WEAK)
    ledger = ResourceLedger()
    ledger.add(replace(span, hbm_read_bytes=read, hbm_write_bytes=write))
    summary = ledger.summarize()
    assert summary.hbm_read_bytes.status is QuantityStatus.INCOMPLETE
    assert summary.hbm_read_bytes.observed_upper == read.observed_upper
    assert summary.flops.status is QuantityStatus.UNKNOWN
    assert summary.estimator_version == HBM_ESTIMATOR_ID


def test_deepseek_v4_split_dtype_and_sublinear_kv():
    arch = load_arch(DSV4)
    got = estimate_request_hbm(arch, input_tokens=64, output_tokens=8)
    assert 0 < got.weight_read < got.read
    small = estimate_request_hbm(arch, input_tokens=2048, output_tokens=8)
    big = estimate_request_hbm(arch, input_tokens=16384, output_tokens=8)
    assert big.kv_read / small.kv_read < 64.0
    assert unique_routed_expert_bounds(arch, 1000) == (6, 256)


def test_glm5_decode_reads_topk_latent_plus_all_indexer_keys():
    arch = load_arch(GLM)
    got = estimate_request_hbm(
        arch, input_tokens=1000, output_tokens=2, compute_prefix_cached_tokens=1000
    )
    latent = (arch.kv_lora_rank + arch.rope_head_dim) * 2 * arch.num_hidden_layers
    index_keys = arch.index_head_dim * 2 * arch.num_hidden_layers
    # Fully cached prompt: both output tokens are decode steps attending to 1001 then 1002
    # positions, all under the top-2048 cap.
    assert got.kv_read == (1001 + 1002) * (latent + index_keys)


def test_glm5_all_expert_pass_matches_published_bf16_size():
    arch = load_arch(GLM)
    # zai-org/GLM-5 is ~754B parameters in BF16 (MTP layer excluded here).
    got = _v3_weight_read_bytes(arch, int(arch.num_experts))
    assert got / (754e9 * 2) == pytest.approx(1.0, abs=0.05)


def test_enrich_span_uses_sglang_cache_report_only_for_qwen():
    usage = {"cached_input_tokens": 80, "cache_provenance": "openai_usage"}
    weak = enrich_span(_span(**usage), serving_spec_from_model_name(WEAK))
    assert weak.flops.status is QuantityStatus.ESTIMATED
    glm = enrich_span(
        _span(role=ModelRole.STRONG, **usage), serving_spec_from_model_name("glm-5")
    )
    assert glm.flops.status is QuantityStatus.INCOMPLETE
    assert "billing_cache_not_compute_reuse" in glm.flops.provenance
    unknown = enrich_span(_span(**usage), serving_spec_from_model_name("some-model"))
    assert unknown.flops.status is QuantityStatus.UNKNOWN
