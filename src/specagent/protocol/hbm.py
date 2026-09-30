"""Architecture-aware, batch-1-equivalent HBM (model-memory) traffic from request token counts.

Counts algorithmic DRAM bytes for one request served alone (batch 1, no tensor-parallel
factor): weight reads per forward pass, KV-cache reads and writes, recurrent-state traffic, and
a coarse activation term. Concurrent serving shares weight reads across requests, so per-request
sums are an isolated-request equivalent, not measured DRAM traffic. The paper reports the read
side only; write is kept as an unknown (not zero) quantity.

Modeling assumptions:
  * KV cache exists only on full-attention layers (qwen3_5) or as the MLA / compressed latent
    on every layer (deepseek_v3 / deepseek_v4).
  * Gated DeltaNet layers keep a float32 recurrent state that is read and written once per
    decode step and once per prefill chunk (see GATED_DELTA_PREFILL_CHUNK_SIZE).
  * MoE prefill reads an unknown set of unique routed experts per layer. Without routing
    telemetry the estimate is the interval top_k <= U <= min(E, top_k * prefill_tokens)
    (see unique_routed_expert_bounds); decode reads top_k experts per step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from specagent.protocol.accounting import (
    InferenceSpan,
    QuantityStatus,
    ResourceQuantity,
    unknown_quantity,
)
from specagent.protocol.estimator_ids import HBM_ESTIMATOR_ID
from specagent.protocol.flops import (
    COMPUTE_REUSE_PROVENANCE,
    ModelArch,
    _generation_steps,
    _v4_compressed_pairs,
    _v4_window_pairs,
    capped_positions_sum,
    load_arch,
)
from specagent.protocol.paper import paper_write_unknown

ESTIMATOR_ID = HBM_ESTIMATOR_ID
COVERAGE = None
UNCOVERED = (
    "collective_comms",
    "padding_and_kernel_replay",
    "gated_delta_chunk_kernel_intra_chunk_bytes",
    "moe_routing_telemetry",
    "elementwise_and_norm_activation_detail",
)
ACTIVATION_BYTES = 2
# Gated DeltaNet prefill runs the chunked chunk_gated_delta_rule kernel: the float32 recurrent
# state round-trips through HBM once per chunk of positions, not once per token (intra-chunk
# work stays on chip). chunk_size=64 is the default of the reference implementation
# (torch_chunk_gated_delta_rule in transformers' Qwen3.5 modeling code). A decode forward is a
# chunk of one token, so decode keeps one state read+write per step.
GATED_DELTA_PREFILL_CHUNK_SIZE = 64
_DTYPE_BYTES = {"fp8": 1, "bf16": 2, "float32": 4}


@dataclass(frozen=True, slots=True)
class HbmBytes:
    read: int
    write: int
    weight_read: int = 0
    kv_read: int = 0
    state_read: int = 0
    activation_read: int = 0

    @property
    def total(self) -> int:
        return self.read + self.write


def _dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype!r}") from None


def weight_dtype_bytes(arch: ModelArch) -> int:
    return _dtype_bytes(arch.weight_dtype)


def _full_attn_weight_elems(arch: ModelArch) -> int:
    gate = 1 + int(arch.attn_output_gate)
    q_out = arch.num_attention_heads * gate * arch.head_dim
    kv_out = 2 * arch.num_key_value_heads * arch.head_dim
    o_in = arch.num_attention_heads * arch.head_dim
    return (
        arch.hidden_size * (q_out + kv_out)
        + o_in * arch.hidden_size
    )


def _linear_attn_weight_elems(arch: ModelArch) -> int:
    qkvz = arch.hidden_size * (2 * arch.key_dim + 2 * arch.value_dim)
    ba = arch.hidden_size * (2 * arch.linear_num_value_heads)
    conv = (arch.key_dim * 2 + arch.value_dim) * arch.linear_conv_kernel_dim
    out = arch.hidden_size * arch.value_dim
    return qkvz + ba + conv + out


def _mlp_weight_elems(arch: ModelArch, *, routed_experts: int | None = None) -> int:
    if not arch.is_moe:
        return 3 * arch.hidden_size * int(arch.intermediate_size)
    if routed_experts is None:
        routed_experts = int(arch.num_experts_per_tok)
    router = arch.hidden_size * int(arch.num_experts)
    shared = 3 * arch.hidden_size * int(arch.shared_expert_intermediate_size)
    routed = 3 * arch.hidden_size * int(arch.moe_intermediate_size) * int(routed_experts)
    return router + shared + routed


def stack_weight_elems(arch: ModelArch, *, routed_experts: int | None = None) -> int:
    """Weight elements read by one forward pass of a qwen3_5 stack.

    The vocab_size * hidden term is the lm_head matmul (one full read per pass). Untied
    input-embedding rows are charged per token in estimate_request_hbm.
    """
    mlp = _mlp_weight_elems(arch, routed_experts=routed_experts)
    total = arch.vocab_size * arch.hidden_size
    for kind in arch.layer_types:
        total += mlp
        if kind == "full_attention":
            total += _full_attn_weight_elems(arch)
        elif kind == "linear_attention":
            total += _linear_attn_weight_elems(arch)
        else:
            raise ValueError(f"unsupported layer type {kind}")
    return total


def unique_routed_expert_bounds(arch: ModelArch, prefill_tokens: int) -> tuple[int, int]:
    """Architecture-derived prefill unique-expert interval.

    Per layer, without routing telemetry:
        top_k <= U_l <= min(E, top_k * prefill_tokens)
    For Qwen3.5-397B-A17B that is 10 <= U_l <= min(512, 10 * prefill_tokens).
    This is not a serving DRAM counter and not a choice of 10 or 512 as a point.
    Decode uses top_k only. Prefill_tokens <= 0 means no prefill expert read.
    """
    if not arch.is_moe:
        return 0, 0
    topk = int(arch.num_experts_per_tok)
    n_exp = int(arch.num_experts)
    tokens = max(int(prefill_tokens), 0)
    if tokens <= 0:
        return 0, 0
    lower = topk
    upper = min(n_exp, topk * tokens)
    if lower > upper:
        upper = lower
    return lower, upper


def kv_bytes_per_token(arch: ModelArch, kv_bytes: int) -> int:
    return 2 * arch.num_key_value_heads * arch.head_dim * kv_bytes


def _v4_weight_elems(
    arch: ModelArch, routed_experts: int
) -> tuple[int, int, int, int]:
    """(fp8, bf16, fp32, fp4) element counts for one full deepseek_v4 stack.

    Dtypes follow the reference inference/model.py: fp8 default Linear, bf16 wo_a/wo_b and
    weights_proj, fp32 compressor/gate/hc params, routed experts at ``expert_dtype`` (fp4 in
    the published checkpoint). The untied head is counted at fp8.
    """
    if arch.expert_dtype not in ("fp4", "bf16"):
        raise ValueError(f"unsupported deepseek_v4 expert_dtype {arch.expert_dtype!r}")
    h = arch.hidden_size
    hd = arch.head_dim
    fp8 = arch.vocab_size * h  # untied head; embedding rows are charged per token
    bf16 = 0
    fp32 = 0
    fp4 = 0
    for ratio in arch.compress_ratios[: arch.num_hidden_layers]:
        r = int(ratio)
        fp8 += h * arch.q_lora_rank
        fp8 += arch.q_lora_rank * (arch.num_attention_heads * hd)
        fp8 += h * hd
        o_in = arch.num_attention_heads * hd // max(arch.o_groups, 1)
        bf16 += o_in * (arch.o_groups * arch.o_lora_rank)
        bf16 += (arch.o_groups * arch.o_lora_rank) * h
        if r > 0:
            coff = 1 + int(r == 4)
            fp32 += 2 * h * coff * hd
        if r == 4:
            fp8 += arch.q_lora_rank * arch.index_n_heads * arch.index_head_dim
            bf16 += h * arch.index_n_heads
            fp32 += 2 * h * 2 * arch.index_head_dim
        fp32 += h * int(arch.num_experts)
        fp8 += 3 * h * int(arch.shared_expert_intermediate_size)
        expert = 3 * h * int(arch.moe_intermediate_size) * routed_experts
        if arch.expert_dtype == "bf16":
            bf16 += expert
        else:
            fp4 += expert
        if arch.hc_mult > 0:
            mix = (2 + arch.hc_mult) * arch.hc_mult
            fp32 += 2 * mix * arch.hc_mult * h
    return fp8, bf16, fp32, fp4


def _v4_weight_read_bytes(arch: ModelArch, routed_experts: int) -> int:
    fp8, bf16, fp32, fp4 = _v4_weight_elems(arch, routed_experts)
    return fp8 + 2 * bf16 + 4 * fp32 + fp4 // 2


def _v4_request_hbm(
    arch: ModelArch,
    *,
    prefill_tokens: int,
    decode_forwards: int,
    context: int,
    kv_b: int,
    act_bytes: int,
    pref_e: int,
    dec_e: int,
) -> HbmBytes:
    """deepseek_v4 algorithmic bytes: split-dtype weights + latent KV + fp32 state.

    KV traffic is positions-based: each layer's cache holds a 512-dim latent
    per window slot and per compressed slot; ratio-4 layers additionally read
    the index cache (index_head_dim wide, all compressed slots) for scoring.
    """
    weight_reads = (
        (1 if prefill_tokens > 0 else 0) * _v4_weight_read_bytes(arch, pref_e)
        + decode_forwards * _v4_weight_read_bytes(arch, dec_e)
        # one fp8 input-embedding row per newly embedded token
        + (prefill_tokens + decode_forwards) * arch.hidden_size
    )
    kv_latent = arch.head_dim * kv_b
    idx_latent = arch.index_head_dim * kv_b
    kv_read = 0
    kv_write = prefill_tokens * arch.num_hidden_layers * kv_latent
    n_decode = decode_forwards
    win_read = _v4_window_pairs(n_decode, context, arch.sliding_window)
    for ratio in arch.compress_ratios[: arch.num_hidden_layers]:
        r = int(ratio)
        kv_read += win_read * kv_latent
        kv_write += n_decode * kv_latent
        if r == 0:
            continue
        kv_write += (prefill_tokens // r) * kv_latent
        uncapped = _v4_compressed_pairs(n_decode, context, r, 0)
        if r == 4:
            kv_read += (
                _v4_compressed_pairs(n_decode, context, r, arch.index_topk)
                * kv_latent
            )
            kv_read += uncapped * idx_latent
            kv_write += (prefill_tokens // 4) * idx_latent
            kv_write += n_decode * (kv_latent // 4 + idx_latent // 4)
        else:
            kv_read += uncapped * kv_latent
            kv_write += n_decode * kv_latent // r
    act_mult = max(arch.hc_mult, 1)
    act = (
        (prefill_tokens + decode_forwards)
        * arch.hidden_size
        * act_bytes
        * 2
        * act_mult
        * arch.num_hidden_layers
    )
    read = weight_reads + kv_read + act // 2
    write = kv_write + act // 2
    return HbmBytes(
        read=read,
        write=write,
        weight_read=weight_reads,
        kv_read=kv_read,
        state_read=0,
        activation_read=act // 2,
    )


def _v3_weight_read_bytes(arch: ModelArch, routed_experts: int) -> int:
    """One pass over a deepseek_v3 stack reading ``routed_experts`` experts per MoE layer.

    All weights BF16, as published for GLM-5. The MLA kv_b weights are read in both serving
    forms, so the count does not depend on prefill vs decode.
    """
    if arch.weight_dtype != "bf16" or arch.expert_dtype != "bf16":
        raise ValueError("deepseek_v3 weight bytes assume BF16 weights and routed experts")
    h = arch.hidden_size
    heads = arch.num_attention_heads
    attn = (
        h * arch.q_lora_rank
        + arch.q_lora_rank * heads * arch.head_dim
        + h * (arch.kv_lora_rank + arch.rope_head_dim)
        + arch.kv_lora_rank * heads * (arch.qk_nope_head_dim + arch.v_head_dim)
        + heads * arch.v_head_dim * h
    )
    n_dense = min(arch.first_k_dense_replace, arch.num_hidden_layers)
    n_moe = arch.num_hidden_layers - n_dense
    bf16 = arch.vocab_size * h + arch.num_hidden_layers * attn  # untied head only
    if arch.index_topk > 0:
        # DSA lightning indexer weights (wq_b, wk, weights_proj) on every layer.
        bf16 += arch.num_hidden_layers * (
            arch.q_lora_rank * arch.index_n_heads * arch.index_head_dim
            + h * arch.index_head_dim
            + h * arch.index_n_heads
        )
    bf16 += n_dense * 3 * h * int(arch.intermediate_size)
    bf16 += n_moe * (
        h * int(arch.num_experts) + 3 * h * int(arch.shared_expert_intermediate_size)
    )
    routed = n_moe * 3 * h * int(arch.moe_intermediate_size) * int(routed_experts)
    return 2 * bf16 + 2 * routed


def _v3_request_hbm(
    arch: ModelArch,
    *,
    prefill_tokens: int,
    decode_forwards: int,
    context: int,
    kv_b: int,
    act_bytes: int,
    pref_e: int,
    dec_e: int,
) -> HbmBytes:
    """deepseek_v3 algorithmic bytes: split-dtype weights + MLA latent KV on every layer.

    The cache holds kv_lora_rank + rope_head_dim values per token per layer, written once per
    new position and read in full by every decode step.
    """
    weight_reads = (
        (1 if prefill_tokens > 0 else 0) * _v3_weight_read_bytes(arch, pref_e)
        + decode_forwards * _v3_weight_read_bytes(arch, dec_e)
        # one bf16 input-embedding row per newly embedded token
        + (prefill_tokens + decode_forwards) * arch.hidden_size * 2
    )
    latent = (arch.kv_lora_rank + arch.rope_head_dim) * kv_b * arch.num_hidden_layers
    kv_sum = decode_forwards * context + decode_forwards * (decode_forwards + 1) // 2
    if arch.index_topk > 0:
        # DSA: each decode step reads the latent of its top-k keys only, plus the indexer key
        # of every cached position; both caches are written once per new position.
        index_keys = arch.index_head_dim * kv_b * arch.num_hidden_layers
        kv_read = (
            capped_positions_sum(context, decode_forwards, arch.index_topk) * latent
            + kv_sum * index_keys
        )
        kv_write = (prefill_tokens + decode_forwards) * (latent + index_keys)
    else:
        kv_read = kv_sum * latent
        kv_write = (prefill_tokens + decode_forwards) * latent
    act = (
        (prefill_tokens + decode_forwards)
        * arch.hidden_size
        * act_bytes
        * 2
        * arch.num_hidden_layers
    )
    return HbmBytes(
        read=weight_reads + kv_read + act // 2,
        write=kv_write + act // 2,
        weight_read=weight_reads,
        kv_read=kv_read,
        state_read=0,
        activation_read=act // 2,
    )


def recurrent_state_bytes(arch: ModelArch, state_bytes: int) -> int:
    return (
        arch.linear_num_value_heads
        * arch.linear_key_head_dim
        * arch.linear_value_head_dim
        * state_bytes
    )


def estimate_request_hbm(
    arch: ModelArch,
    *,
    input_tokens: int,
    output_tokens: int,
    compute_prefix_cached_tokens: int | None = None,
    prefill_routed_experts: int | None = None,
) -> HbmBytes:
    """Algorithmic DRAM bytes for one text request served alone. No TP factor.

    ``prefill_routed_experts`` is the number of unique routed experts read per MoE layer
    during prefill; the default is top_k (the lower end of unique_routed_expert_bounds).
    For qwen3_5 MoE it is clamped to that interval.
    """
    if min(input_tokens, output_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    kv_b = _dtype_bytes(arch.kv_dtype)
    if compute_prefix_cached_tokens is None:
        cached = 0
    else:
        cached = min(int(compute_prefix_cached_tokens), input_tokens)
    prefill_tokens = input_tokens - cached
    decode_forwards, _n_logits = _generation_steps(prefill_tokens, output_tokens)
    if arch.is_deepseek_v4 or arch.is_deepseek_v3:
        if prefill_routed_experts is not None:
            pref_e = int(prefill_routed_experts)
        else:
            pref_e = int(arch.num_experts_per_tok)
        return (_v3_request_hbm if arch.is_deepseek_v3 else _v4_request_hbm)(
            arch,
            prefill_tokens=prefill_tokens,
            decode_forwards=decode_forwards,
            context=input_tokens,
            kv_b=kv_b,
            act_bytes=ACTIVATION_BYTES,
            pref_e=pref_e,
            dec_e=int(arch.num_experts_per_tok),
        )
    w_b = weight_dtype_bytes(arch)
    if arch.is_moe:
        lo, hi = unique_routed_expert_bounds(arch, prefill_tokens)
        pref_e = lo if prefill_routed_experts is None else int(prefill_routed_experts)
        pref_e = min(max(pref_e, lo), hi if hi else pref_e)
        dec_e = int(arch.num_experts_per_tok)
        pref_w = (
            stack_weight_elems(arch, routed_experts=pref_e) if prefill_tokens > 0 else 0
        )
        dec_w = stack_weight_elems(arch, routed_experts=dec_e)
        weight_reads = (pref_w + decode_forwards * dec_w) * w_b
    else:
        weight_passes = (1 if prefill_tokens > 0 else 0) + decode_forwards
        weight_reads = weight_passes * stack_weight_elems(arch) * w_b

    if not arch.tie_word_embeddings:
        # An untied embedding lookup reads one row (hidden_size elements) per newly embedded
        # token; the full vocab_size * hidden lm_head matrix is already charged once per
        # weight pass in stack_weight_elems. Tied models share that single matrix.
        weight_reads += (prefill_tokens + decode_forwards) * arch.hidden_size * w_b

    kv_tok = kv_bytes_per_token(arch, kv_b)
    state = recurrent_state_bytes(arch, _dtype_bytes(arch.mamba_ssm_dtype))
    n_full = arch.n_full_attn
    n_linear = arch.n_linear_attn
    context = input_tokens

    kv_write = prefill_tokens * kv_tok * n_full
    kv_read = 0
    if decode_forwards > 0:
        kv_sum = decode_forwards * context + decode_forwards * (decode_forwards + 1) // 2
        kv_read = kv_sum * kv_tok * n_full
        kv_write += decode_forwards * kv_tok * n_full

    # Recurrent state: one read+write per prefill chunk and per decode forward, on every
    # linear-attention layer.
    if n_linear > 0 and prefill_tokens > 0:
        n_state_touches_prefill = math.ceil(prefill_tokens / GATED_DELTA_PREFILL_CHUNK_SIZE)
    else:
        n_state_touches_prefill = prefill_tokens
    state_rw = (n_state_touches_prefill + decode_forwards) * 2 * state * n_linear
    state_read = state_rw // 2
    act = (prefill_tokens + decode_forwards) * arch.hidden_size * ACTIVATION_BYTES * 2
    act *= arch.num_hidden_layers
    activation_read = act // 2
    read = weight_reads + kv_read + state_read + activation_read
    write = kv_write + state_rw // 2 + act // 2
    return HbmBytes(
        read=read,
        write=write,
        weight_read=weight_reads,
        kv_read=kv_read,
        state_read=state_read,
        activation_read=activation_read,
    )


def _hbm_provenance(arch: ModelArch, extra: str = "") -> str:
    if arch.is_deepseek_v4:
        layout = f"deepseek_v4_latent_kv_compressed_positions;{arch.expert_dtype}_experts"
    elif arch.is_deepseek_v3:
        layout = "mla_latent_kv_all_layers"
    else:
        layout = (
            "full_attn_kv_only;linear_recurrent_state;"
            f"gdn_prefill_chunk={GATED_DELTA_PREFILL_CHUNK_SIZE}"
        )
    parts = [
        ESTIMATOR_ID,
        f"arch={arch.model_id}",
        f"hf_repo={arch.hf_repo}",
        f"weight_dtype={arch.weight_dtype}",
        f"kv_dtype={arch.kv_dtype}",
        "no_tp",
        layout,
        "strict_batch_1",
        "paper_metric=batch_1_equivalent_estimated_model_memory_read_traffic",
        "coverage=unknown",
        "uncovered=" + ",".join(UNCOVERED),
    ]
    if extra:
        parts.append(extra)
    return ";".join(parts)


def _quantity_from_bytes(
    value: int | None,
    arch: ModelArch,
    *,
    extra: str,
    lower: int | None = None,
    upper: int | None = None,
) -> ResourceQuantity:
    if lower is not None and upper is not None and lower != upper:
        return ResourceQuantity(
            value=None,
            status=QuantityStatus.INCOMPLETE,
            provenance=_hbm_provenance(arch, extra=extra),
            coverage=COVERAGE,
            observed_subtotal=lower,
            observed_lower=lower,
            observed_upper=upper,
        )
    return ResourceQuantity(
        value=int(value or 0),
        status=QuantityStatus.ESTIMATED,
        provenance=_hbm_provenance(arch, extra=extra),
        coverage=COVERAGE,
    )


def hbm_read_interval(
    arch: ModelArch,
    *,
    input_tokens: int,
    output_tokens: int,
    compute_prefix_cached_tokens: int | None = None,
) -> tuple[int, int]:
    """(lower, upper) batch-1-equivalent HBM read bytes for one request.

    For MoE models the bounds read top_k and min(E, top_k * prefill_tokens) unique routed
    experts per layer during prefill (unique_routed_expert_bounds); dense models give a point.
    """
    if not arch.is_moe:
        read = estimate_request_hbm(
            arch,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compute_prefix_cached_tokens=compute_prefix_cached_tokens,
        ).read
        return read, read
    cached = compute_prefix_cached_tokens
    prefill_tokens = input_tokens if cached is None else max(input_tokens - int(cached), 0)
    reads = [
        estimate_request_hbm(
            arch,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compute_prefix_cached_tokens=cached,
            prefill_routed_experts=experts,
        ).read
        for experts in unique_routed_expert_bounds(arch, prefill_tokens)
    ]
    return min(reads), max(reads)


def _pair_from_modes(
    arch: ModelArch,
    *,
    input_tokens: int,
    output_tokens: int,
    cached: int | None,
    extra: str,
) -> tuple[ResourceQuantity, ResourceQuantity]:
    """(read, write) quantities for one request: a point, or an interval for MoE prefill."""
    extra_mode = extra + ";moe=dense"
    if arch.is_moe:
        prefill_tokens = input_tokens if cached is None else max(input_tokens - int(cached), 0)
        lo, hi = unique_routed_expert_bounds(arch, prefill_tokens)
        extra_mode = (
            extra
            + f";moe_prefill_unique_experts=interval:{lo}-{hi}"
            + ";phase_split=prefill_then_decode"
        )
    lower, upper = hbm_read_interval(
        arch,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        compute_prefix_cached_tokens=cached,
    )
    read = _quantity_from_bytes(
        lower if lower == upper else None,
        arch,
        extra=extra_mode + ";axis=read",
        lower=lower,
        upper=upper,
    )
    return read, paper_write_unknown()


def estimate_span_hbm(
    span: InferenceSpan, model_id: str
) -> tuple[ResourceQuantity, ResourceQuantity]:
    """(read, write) for one recorded request span; write is always unknown.

    Cache handling matches estimate_span_flops: only compute-reuse provenance skips the cached
    prefix; billing or unknown cache widens the read into an interval.
    """
    unknown = {QuantityStatus.UNKNOWN, QuantityStatus.NOT_APPLICABLE}
    blank = unknown_quantity(provenance="unknown_usage_no_hbm_upgrade")
    if span.usage_status in unknown:
        return blank, blank
    if span.resolved_input_status in unknown:
        miss = unknown_quantity(provenance="unknown_input_tokens_no_hbm_upgrade")
        return miss, miss
    if span.resolved_output_status in unknown:
        miss = unknown_quantity(provenance="unknown_output_tokens_no_hbm_upgrade")
        return miss, miss
    try:
        arch = load_arch(model_id)
    except KeyError as exc:
        miss = unknown_quantity(provenance=f"unsupported_model:{model_id}:{exc}")
        return miss, miss

    reuse_known = span.cache_provenance in COMPUTE_REUSE_PROVENANCE
    if span.resolved_cached_status in unknown:
        no_skip_r, _no_skip_w = _pair_from_modes(
            arch,
            input_tokens=span.input_tokens,
            output_tokens=span.output_tokens,
            cached=None,
            extra="unknown_cache_not_compute_reuse",
        )
        skip_r, _skip_w = _pair_from_modes(
            arch,
            input_tokens=span.input_tokens,
            output_tokens=span.output_tokens,
            cached=span.input_tokens,
            extra="unknown_cache_not_compute_reuse",
        )
        return (
            _span_interval(no_skip_r, skip_r, arch, "unknown_cache_not_compute_reuse;axis=read"),
            paper_write_unknown(),
        )
    billing_cache = span.cached_input_tokens > 0 and not reuse_known
    if billing_cache:
        no_skip_r, _no_skip_w = _pair_from_modes(
            arch,
            input_tokens=span.input_tokens,
            output_tokens=span.output_tokens,
            cached=None,
            extra="billing_cache_not_compute_reuse",
        )
        skip_r, _skip_w = _pair_from_modes(
            arch,
            input_tokens=span.input_tokens,
            output_tokens=span.output_tokens,
            cached=span.cached_input_tokens,
            extra="billing_cache_not_compute_reuse",
        )
        return (
            _span_interval(no_skip_r, skip_r, arch, "billing_cache_not_compute_reuse;axis=read"),
            paper_write_unknown(),
        )
    cached = span.cached_input_tokens if reuse_known else None
    return _pair_from_modes(
        arch,
        input_tokens=span.input_tokens,
        output_tokens=span.output_tokens,
        cached=cached,
        extra="compute_reuse" if reuse_known else "no_compute_reuse",
    )


def _bound_value(item: ResourceQuantity) -> tuple[int, int]:
    if item.status is QuantityStatus.ESTIMATED and item.value is not None:
        value = int(item.value)
        return value, value
    lower = int(item.observed_lower or item.observed_subtotal or 0)
    upper = int(item.observed_upper or lower)
    return lower, upper


def _span_interval(
    left: ResourceQuantity,
    right: ResourceQuantity,
    arch: ModelArch,
    extra: str,
) -> ResourceQuantity:
    l0, l1 = _bound_value(left)
    r0, r1 = _bound_value(right)
    lower = min(l0, l1, r0, r1)
    upper = max(l0, l1, r0, r1)
    return _quantity_from_bytes(None, arch, extra=extra, lower=lower, upper=upper)


