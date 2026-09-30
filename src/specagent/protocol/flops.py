"""Architecture-aware inference FLOPs from request token counts.

Counts algorithmic multiply-adds of the text backbone described by the architecture configs in
``architectures/`` (transcribed from each model's published config.json). Not counted: tensor-
parallel replication, vision towers, MTP layers, norms, softmax, RoPE, and collectives.

FMA=2: a multiply-add is two FLOPs, so GEMM(m, n, k) = 2 * m * n * k.

Supported architectures (``model_type``):
  * ``qwen3_5_text`` / ``qwen3_5_moe_text``: hybrid Gated DeltaNet + gated full attention,
    dense SwiGLU or MoE (router + shared expert + top-k routed experts).
  * ``deepseek_v4``: sliding-window plus compressed sparse attention with a ratio-4 indexer,
    low-rank projections, and Hyper-Connection mixing.
  * ``deepseek_v3``: MLA (multi-head form for prefill, weight-absorbed latent form for decode),
    optionally with DSA top-k sparse attention (GLM-5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import lru_cache
from importlib import resources
from typing import Any

from specagent.protocol.accounting import (
    InferenceSpan,
    QuantityStatus,
    ResourceQuantity,
    unknown_quantity,
)
from specagent.protocol.estimator_ids import FLOPS_ESTIMATOR_ID

FMA = 2
ESTIMATOR_ID = FLOPS_ESTIMATOR_ID
COVERAGE = None
UNCOVERED_OPERATORS = (
    "rmsnorm",
    "silu_elementwise",
    "softmax",
    "rope",
    "mtp",
    "vision",
    "collective_comms",
    "gated_delta_chunk_kernel",
)
# Terms of the Gated DeltaNet recurrent update, each one rank-sized multiply-add per value head.
GATED_DELTA_RECURRENT_TERMS = (
    "decay",
    "state_key_read",
    "delta",
    "rank1_update",
    "query_readout",
)
# Cache provenances that mean the reported cached prefix was actually skipped by the server
# (prefix-cache compute reuse), as opposed to a provider billing discount.
COMPUTE_REUSE_PROVENANCE = frozenset({"sglang_prefix_cache", "measured_compute_reuse"})
_ARCH_FILES = {
    "Qwen3.5-4B": "qwen3_5_4b.json",
    "Qwen3.5-397B-A17B": "qwen3_5_397b_a17b.json",
    "DeepSeek-V4-Flash-0731": "deepseek_v4_flash_0731.json",
    "GLM-5": "glm_5.json",
}
_TUPLE_FIELDS = ("layer_types", "compress_ratios")
_DOC_FIELDS = ("notes",)


def _gemm(m: int, n: int, k: int) -> int:
    return FMA * int(m) * int(n) * int(k)


def _swiglu(tokens: int, hidden: int, intermediate: int) -> int:
    # gate, up, down
    return (
        _gemm(tokens, intermediate, hidden)
        + _gemm(tokens, intermediate, hidden)
        + _gemm(tokens, hidden, intermediate)
    )


@dataclass(frozen=True, slots=True)
class ModelArch:
    """Text-backbone architecture. Field names follow the HF config keys."""

    model_id: str
    model_type: str
    hf_repo: str
    weight_dtype: str
    kv_dtype: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    hf_revision: str = ""
    # Routed-expert weight dtype when it differs from weight_dtype (deepseek_v3 / deepseek_v4).
    expert_dtype: str | None = None
    tie_word_embeddings: bool = False
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    # qwen3_5: gated full attention + Gated DeltaNet linear attention.
    attn_output_gate: bool = False
    linear_conv_kernel_dim: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    layer_types: tuple[str, ...] = ()
    mamba_ssm_dtype: str = "float32"
    # deepseek_v4 fields.
    q_lora_rank: int = 0
    o_lora_rank: int = 0
    o_groups: int = 0
    rope_head_dim: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    compress_ratios: tuple[int, ...] = ()
    sliding_window: int = 0
    hc_mult: int = 0
    # deepseek_v3 (MLA) fields. head_dim is the q/k width, qk_nope_head_dim + rope_head_dim.
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    first_k_dense_replace: int = 0

    @property
    def is_moe(self) -> bool:
        return self.model_type in ("qwen3_5_moe_text", "deepseek_v4", "deepseek_v3")

    @property
    def is_deepseek_v4(self) -> bool:
        return self.model_type == "deepseek_v4"

    @property
    def is_deepseek_v3(self) -> bool:
        return self.model_type == "deepseek_v3"

    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def n_full_attn(self) -> int:
        return sum(1 for item in self.layer_types if item == "full_attention")

    @property
    def n_linear_attn(self) -> int:
        return sum(1 for item in self.layer_types if item == "linear_attention")


def supported_models() -> tuple[str, ...]:
    return tuple(_ARCH_FILES)


def _load_arch_payload(filename: str) -> dict[str, Any]:
    package = "specagent.protocol.architectures"
    with resources.files(package).joinpath(filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=8)
def load_arch(model_id: str) -> ModelArch:
    """Load a supported architecture by id; raises KeyError for anything else."""
    filename = _ARCH_FILES.get(model_id)
    if filename is None:
        raise KeyError(f"unsupported model_id for the estimators: {model_id}")
    payload = _load_arch_payload(filename)
    known = {item.name for item in fields(ModelArch)}
    unexpected = set(payload) - known - set(_DOC_FIELDS)
    if unexpected:
        raise KeyError(f"unexpected architecture keys {sorted(unexpected)}: {model_id}")
    kwargs = {key: value for key, value in payload.items() if key in known}
    for key in _TUPLE_FIELDS:
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    arch = ModelArch(**kwargs)
    if arch.is_deepseek_v4 and len(arch.compress_ratios) < arch.num_hidden_layers:
        raise KeyError(f"deepseek_v4_compress_ratios_shorter_than_layers:{model_id}")
    if arch.is_deepseek_v3 and (
        not arch.kv_lora_rank
        or not arch.v_head_dim
        or arch.head_dim != arch.qk_nope_head_dim + arch.rope_head_dim
    ):
        raise KeyError(f"deepseek_v3_mla_dims_inconsistent:{model_id}")
    return arch


def _mlp_tokens(arch: ModelArch, tokens: int) -> int:
    if tokens <= 0:
        return 0
    if arch.is_moe:
        hidden = arch.hidden_size
        routed = _swiglu(
            tokens, hidden, int(arch.moe_intermediate_size)
        ) * int(arch.num_experts_per_tok)
        shared = _swiglu(
            tokens, hidden, int(arch.shared_expert_intermediate_size)
        )
        router = _gemm(tokens, int(arch.num_experts), hidden)
        return router + shared + routed
    return _swiglu(tokens, arch.hidden_size, int(arch.intermediate_size))


def _full_attn_proj(arch: ModelArch, tokens: int) -> int:
    if tokens <= 0:
        return 0
    gate = 1 + int(arch.attn_output_gate)
    q_out = arch.num_attention_heads * gate * arch.head_dim
    kv_out = 2 * arch.num_key_value_heads * arch.head_dim
    qkv = _gemm(tokens, q_out + kv_out, arch.hidden_size)
    out = _gemm(tokens, arch.hidden_size, arch.num_attention_heads * arch.head_dim)
    return qkv + out


def causal_prefill_pairs(new_tokens: int, cached_tokens: int) -> int:
    """Causal QK pairs for P new tokens with C cached prefix tokens.

    Uncached length N uses N(N+1)/2. Cached prefix C and suffix P uses
    P*C + P(P+1)/2.
    """
    prefix = int(new_tokens)
    cached = int(cached_tokens)
    if prefix <= 0:
        return 0
    if cached < 0:
        raise ValueError("cached_tokens must be non-negative")
    return prefix * cached + prefix * (prefix + 1) // 2


def _full_attn_scores_from_pairs(arch: ModelArch, pairs: int) -> int:
    if pairs <= 0:
        return 0
    # QK^T and attn@V, FMA=2 each.
    return 2 * _gemm(arch.num_attention_heads, pairs, arch.head_dim)


def _gated_delta_state(arch: ModelArch, tokens: int) -> int:
    """Gated DeltaNet state update, token-linear.

    Decode matches the recurrent rule terms (decay, state-key, delta, rank-1, query). Prefill
    uses the same per-token count as an approximation of the chunked kernel, whose intra-chunk
    matmuls are not modeled separately.
    """
    if tokens <= 0:
        return 0
    return (
        FMA
        * tokens
        * arch.linear_num_value_heads
        * arch.linear_key_head_dim
        * arch.linear_value_head_dim
        * len(GATED_DELTA_RECURRENT_TERMS)
    )


def _linear_attn(arch: ModelArch, tokens: int) -> int:
    if tokens <= 0:
        return 0
    qkvz = _gemm(tokens, 2 * arch.key_dim + 2 * arch.value_dim, arch.hidden_size)
    ba = _gemm(tokens, 2 * arch.linear_num_value_heads, arch.hidden_size)
    conv = _gemm(tokens, arch.key_dim * 2 + arch.value_dim, arch.linear_conv_kernel_dim)
    out = _gemm(tokens, arch.hidden_size, arch.value_dim)
    return qkvz + ba + conv + _gated_delta_state(arch, tokens) + out


def _stack_tokens(arch: ModelArch, tokens: int, attn_pairs: int) -> int:
    """One forward through every decoder layer for ``tokens`` positions."""
    if tokens <= 0:
        return 0
    total = 0
    mlp = _mlp_tokens(arch, tokens)
    linear = _linear_attn(arch, tokens)
    proj = _full_attn_proj(arch, tokens)
    scores = _full_attn_scores_from_pairs(arch, attn_pairs)
    for kind in arch.layer_types:
        total += mlp
        if kind == "full_attention":
            total += proj + scores
        elif kind == "linear_attention":
            total += linear
        else:
            raise ValueError(f"unsupported layer type {kind}")
    return total


def _decode_full_attn_scores(arch: ModelArch, prompt_len: int, n_decode: int) -> int:
    if n_decode <= 0:
        return 0
    # Each decode step t attends causally to prompt_len+t positions.
    kv_sum = n_decode * prompt_len + n_decode * (n_decode + 1) // 2
    return _full_attn_scores_from_pairs(arch, kv_sum)


def _decode_stack(arch: ModelArch, prompt_len: int, n_decode: int) -> int:
    if n_decode <= 0:
        return 0
    mlp = _mlp_tokens(arch, n_decode)
    linear = _linear_attn(arch, n_decode)
    proj = _full_attn_proj(arch, n_decode)
    scores = _decode_full_attn_scores(arch, prompt_len, n_decode)
    total = 0
    for kind in arch.layer_types:
        total += mlp
        if kind == "full_attention":
            total += proj + scores
        elif kind == "linear_attention":
            total += linear
        else:
            raise ValueError(f"unsupported layer type {kind}")
    return total


def _lm_head(arch: ModelArch, n_logits: int) -> int:
    if n_logits <= 0:
        return 0
    return _gemm(n_logits, arch.vocab_size, arch.hidden_size)


def _floor_sum_upto(n: int, r: int) -> int:
    """Sum_{k=0}^{n} floor(k/r)."""
    if n < 0 or r <= 0:
        return 0
    q, rem = divmod(n, r)
    return r * q * (q - 1) // 2 + q * (rem + 1)


def _floor_sum_range(lo: int, hi: int, r: int) -> int:
    if hi < lo:
        return 0
    return _floor_sum_upto(hi, r) - _floor_sum_upto(lo - 1, r)


def _v4_window_pairs(new_tokens: int, cached: int, window: int) -> int:
    """Sum over new tokens of min(window, position+1)."""
    if new_tokens <= 0 or window <= 0:
        return 0
    lo, hi = cached + 1, cached + new_tokens
    n_sat = max(0, hi - max(lo, window) + 1)
    partial_hi = min(hi, window - 1)
    partial = 0
    if partial_hi >= lo:
        cnt = partial_hi - lo + 1
        partial = cnt * (lo + partial_hi) // 2
    return partial + n_sat * window


def _v4_compressed_pairs(
    new_tokens: int, cached: int, ratio: int, topk: int
) -> int:
    """Compressed KV slots per token: floor((p+1)/ratio), capped at topk (0=uncapped)."""
    if new_tokens <= 0 or ratio <= 0:
        return 0
    lo, hi = cached + 1, cached + new_tokens
    if topk <= 0:
        return _floor_sum_range(lo, hi, ratio)
    cap = topk * ratio
    n_sat = max(0, hi - max(lo, cap) + 1)
    partial = _floor_sum_range(lo, min(hi, cap - 1), ratio)
    return partial + n_sat * topk


def _v4_attn_proj(arch: ModelArch, tokens: int) -> int:
    """Low-rank MLA-ish projections: wq_a/wq_b, single-latent wkv, grouped wo_a/wo_b."""
    if tokens <= 0:
        return 0
    q = _gemm(tokens, arch.q_lora_rank, arch.hidden_size) + _gemm(
        tokens, arch.num_attention_heads * arch.head_dim, arch.q_lora_rank
    )
    kv = _gemm(tokens, arch.head_dim, arch.hidden_size)
    o_in = arch.num_attention_heads * arch.head_dim // max(arch.o_groups, 1)
    o = _gemm(tokens, arch.o_groups * arch.o_lora_rank, o_in) + _gemm(
        tokens, arch.hidden_size, arch.o_groups * arch.o_lora_rank
    )
    return q + kv + o


def _v4_compressor(arch: ModelArch, tokens: int, ratio: int, head_dim: int) -> int:
    """wkv + wgate gated-pooling projections (fp32 in reference impl)."""
    coff = 1 + int(ratio == 4)
    return 2 * _gemm(tokens, coff * head_dim, arch.hidden_size)


def _v4_indexer_proj(arch: ModelArch, tokens: int) -> int:
    """Index head query proj + per-head weights + its own compressor."""
    return (
        _gemm(tokens, arch.index_n_heads * arch.index_head_dim, arch.q_lora_rank)
        + _gemm(tokens, arch.index_n_heads, arch.hidden_size)
        + _v4_compressor(arch, tokens, 4, arch.index_head_dim)
    )


def _v4_hc_mix(arch: ModelArch, tokens: int) -> int:
    """Hyper-Connection mixing: hc_pre GEMM before attn and ffn, plus comb contraction."""
    if arch.hc_mult <= 0 or tokens <= 0:
        return 0
    mix = (2 + arch.hc_mult) * arch.hc_mult
    hc_dim = arch.hc_mult * arch.hidden_size
    return 2 * (
        _gemm(tokens, mix, hc_dim)
        + _gemm(tokens, arch.hc_mult * arch.hc_mult, arch.hidden_size)
    )


def _v4_layer_flops(
    arch: ModelArch,
    tokens: int,
    *,
    win_pairs: int,
    comp_pairs: int,
    index_pairs: int,
    ratio: int,
) -> int:
    total = _mlp_tokens(arch, tokens) + _v4_attn_proj(arch, tokens)
    total += _v4_hc_mix(arch, tokens)
    if ratio > 0:
        total += _v4_compressor(arch, tokens, ratio, arch.head_dim)
    if ratio == 4:
        total += _v4_indexer_proj(arch, tokens)
        # index scores: one q.kv matmul over all compressed slots (pre-topk)
        total += _gemm(arch.index_n_heads, index_pairs, arch.index_head_dim)
    total += _full_attn_scores_from_pairs(arch, win_pairs + comp_pairs)
    return total


def _v4_stack(arch: ModelArch, tokens: int, cached: int) -> int:
    """One forward over every decoder layer for ``tokens`` at context ``cached``.

    Attention cost is linear-in-context by design: sliding window plus
    compressed slots (topk-bounded when the ratio-4 indexer selects).
    """
    if tokens <= 0:
        return 0
    win_pairs = _v4_window_pairs(tokens, cached, arch.sliding_window)
    total = 0
    for ratio in arch.compress_ratios[: arch.num_hidden_layers]:
        r = int(ratio)
        if r == 4:
            comp = _v4_compressed_pairs(tokens, cached, r, arch.index_topk)
            idx = _v4_compressed_pairs(tokens, cached, r, 0)
        elif r > 0:
            comp = _v4_compressed_pairs(tokens, cached, r, 0)
            idx = 0
        else:
            comp = idx = 0
        total += _v4_layer_flops(
            arch, tokens, win_pairs=win_pairs, comp_pairs=comp,
            index_pairs=idx, ratio=r,
        )
    return total


def _v3_attn_proj(arch: ModelArch, tokens: int, *, absorbed: bool) -> int:
    """MLA projections. Non-absorbed (prefill) up-projects the new tokens' latent into
    per-head K/V; absorbed (decode) folds those weights into the query and output instead."""
    if tokens <= 0:
        return 0
    heads = arch.num_attention_heads
    q = _gemm(tokens, arch.q_lora_rank, arch.hidden_size) + _gemm(
        tokens, heads * arch.head_dim, arch.q_lora_rank
    )
    kv_a = _gemm(tokens, arch.kv_lora_rank + arch.rope_head_dim, arch.hidden_size)
    if absorbed:
        # q_nope -> latent and latent -> v, one per head: the kv_b weights applied per query.
        kv = _gemm(tokens, heads * arch.kv_lora_rank, arch.qk_nope_head_dim) + _gemm(
            tokens, heads * arch.v_head_dim, arch.kv_lora_rank
        )
    else:
        kv = _gemm(
            tokens, heads * (arch.qk_nope_head_dim + arch.v_head_dim), arch.kv_lora_rank
        )
    o = _gemm(tokens, arch.hidden_size, heads * arch.v_head_dim)
    return q + kv_a + kv + o


def _v3_scores(arch: ModelArch, pairs: int, *, absorbed: bool) -> int:
    """QK^T and attn@V per causal pair: head width 192/128 per head, or the 576/512 latent."""
    if pairs <= 0:
        return 0
    heads = arch.num_attention_heads
    if absorbed:
        qk, av = arch.kv_lora_rank + arch.rope_head_dim, arch.kv_lora_rank
    else:
        qk, av = arch.head_dim, arch.v_head_dim
    return _gemm(heads, pairs, qk) + _gemm(heads, pairs, av)


def capped_positions_sum(start: int, n: int, cap: int) -> int:
    """sum_{i=start+1}^{start+n} min(i, cap): keys each of n new queries attends to under a
    top-``cap`` sparse selection (DSA), counting the query itself; cap <= 0 means uncapped."""
    if n <= 0:
        return 0
    lo, hi = start + 1, start + n

    def full(a: int, b: int) -> int:
        return (a + b) * (b - a + 1) // 2 if b >= a else 0

    if cap <= 0 or hi <= cap:
        return full(lo, hi)
    if lo > cap:
        return n * cap
    return full(lo, cap) + (hi - cap) * cap


def _v3_dsa_indexer(arch: ModelArch, tokens: int, pairs: int) -> int:
    """DSA lightning indexer per layer: q from the q latent, one shared key and per-head
    weights from the hidden state, then a score over every causal pair (all keys)."""
    if tokens <= 0 or arch.index_topk <= 0:
        return 0
    hi, di = arch.index_n_heads, arch.index_head_dim
    proj = (
        _gemm(tokens, hi * di, arch.q_lora_rank)
        + _gemm(tokens, di, arch.hidden_size)
        + _gemm(tokens, hi, arch.hidden_size)
    )
    return proj + _gemm(hi, pairs, di)


def _v3_stack(arch: ModelArch, tokens: int, pairs: int, *, absorbed: bool) -> int:
    """Every decoder layer: MLA, then dense SwiGLU for the first k layers and MoE after."""
    if tokens <= 0:
        return 0
    attn = _v3_attn_proj(arch, tokens, absorbed=absorbed) + _v3_scores(
        arch, pairs, absorbed=absorbed
    )
    dense = _swiglu(tokens, arch.hidden_size, int(arch.intermediate_size))
    moe = _mlp_tokens(arch, tokens)
    n_dense = min(arch.first_k_dense_replace, arch.num_hidden_layers)
    return arch.num_hidden_layers * attn + n_dense * dense + (
        arch.num_hidden_layers - n_dense
    ) * moe


def _generation_steps(prefill_tokens: int, output_tokens: int) -> tuple[int, int]:
    """Map API output_tokens onto serving forwards.

    Prefill emits the first generated token via last-position logits. Each
    later output token is one decode forward. The last output token is not
    fed back into the model.
    """
    if output_tokens <= 0:
        return 0, 0
    if prefill_tokens > 0:
        return output_tokens - 1, output_tokens
    return output_tokens, output_tokens


@dataclass(frozen=True, slots=True)
class FlopsBreakdown:
    prefill: int
    decode: int
    lm_head: int
    total: int
    cache_mode: str
    prefill_tokens: int
    decode_tokens: int
    context_tokens: int
    n_logits: int


def estimate_request_flops(
    arch: ModelArch,
    *,
    input_tokens: int,
    output_tokens: int,
    compute_prefix_cached_tokens: int | None = None,
) -> FlopsBreakdown:
    """F_prefill + F_decode + lm_head for one text request.

    ``compute_prefix_cached_tokens`` is actual reused compute prefix, not
    provider billing cache. Pass None to count a full prefill.
    """
    if min(input_tokens, output_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    if compute_prefix_cached_tokens is None:
        cached = 0
        cache_mode = "no_compute_reuse"
    else:
        if compute_prefix_cached_tokens < 0:
            raise ValueError("compute_prefix_cached_tokens must be non-negative")
        cached = min(compute_prefix_cached_tokens, input_tokens)
        cache_mode = "compute_prefix_reuse"
    prefill_tokens = input_tokens - cached
    context = input_tokens
    decode_forwards, n_logits = _generation_steps(prefill_tokens, output_tokens)
    if arch.is_deepseek_v4:
        prefill = _v4_stack(arch, prefill_tokens, cached)
        decode = _v4_stack(arch, decode_forwards, context)
    elif arch.is_deepseek_v3:
        # Prefill runs MLA in its multi-head form, decode in the weight-absorbed latent form,
        # the split SGLang and vLLM serve DeepSeek-V3 checkpoints with.
        all_pref = causal_prefill_pairs(prefill_tokens, cached)
        kv_sum = decode_forwards * context + decode_forwards * (decode_forwards + 1) // 2
        if arch.index_topk > 0:
            # DSA (GLM-5, DeepSeek-V3.2): MLA attends only to the indexer's top-k keys; the
            # indexer itself scores every causal pair.
            prefill = _v3_stack(
                arch,
                prefill_tokens,
                capped_positions_sum(cached, prefill_tokens, arch.index_topk),
                absorbed=False,
            ) + arch.num_hidden_layers * _v3_dsa_indexer(arch, prefill_tokens, all_pref)
            decode = _v3_stack(
                arch,
                decode_forwards,
                capped_positions_sum(context, decode_forwards, arch.index_topk),
                absorbed=True,
            ) + arch.num_hidden_layers * _v3_dsa_indexer(arch, decode_forwards, kv_sum)
        else:
            prefill = _v3_stack(arch, prefill_tokens, all_pref, absorbed=False)
            decode = _v3_stack(arch, decode_forwards, kv_sum, absorbed=True)
    else:
        pairs = causal_prefill_pairs(prefill_tokens, cached)
        prefill = _stack_tokens(arch, prefill_tokens, pairs)
        decode = _decode_stack(arch, prompt_len=context, n_decode=decode_forwards)
    head = _lm_head(arch, n_logits)
    total = prefill + decode + head
    return FlopsBreakdown(
        prefill=prefill,
        decode=decode,
        lm_head=head,
        total=total,
        cache_mode=cache_mode,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_forwards,
        context_tokens=context,
        n_logits=n_logits,
    )


def _flops_provenance(breakdown: FlopsBreakdown, arch: ModelArch, extra: str = "") -> str:
    if arch.is_deepseek_v4:
        attention = "deepseek_v4_sparse_attn=window_plus_compressed_topk;indexer_scores_counted"
    elif arch.is_deepseek_v3 and arch.index_topk > 0:
        attention = "mla_dsa=topk_attn_plus_dense_indexer;prefill_mha_form,decode_absorbed_latent"
    elif arch.is_deepseek_v3:
        attention = "mla=prefill_mha_form,decode_absorbed_latent"
    else:
        attention = (
            "gated_delta_decode=" + ",".join(GATED_DELTA_RECURRENT_TERMS)
            + ";gated_delta_prefill=token_linear_proxy"
        )
    parts = [
        ESTIMATOR_ID,
        f"fma={FMA}",
        f"arch={arch.model_id}",
        f"hf_repo={arch.hf_repo}",
        f"weight_dtype={arch.weight_dtype}",
        f"cache={breakdown.cache_mode}",
        "no_tp",
        "text_backbone",
        "first_token=prefill_logits",
        "last_token_not_fed_back",
        attention,
        "coverage=unknown",
        "uncovered=" + ",".join(UNCOVERED_OPERATORS),
    ]
    if extra:
        parts.append(extra)
    return ";".join(parts)


def estimated_quantity(breakdown: FlopsBreakdown, arch: ModelArch) -> ResourceQuantity:
    return ResourceQuantity(
        value=breakdown.total,
        status=QuantityStatus.ESTIMATED,
        provenance=_flops_provenance(breakdown, arch),
        coverage=COVERAGE,
    )


def _interval_quantity(
    arch: ModelArch,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    extra: str,
) -> ResourceQuantity:
    """[FLOPs if the cached prefix was skipped, FLOPs if it was recomputed]."""
    no_skip = estimate_request_flops(
        arch, input_tokens=input_tokens, output_tokens=output_tokens
    )
    skipped = estimate_request_flops(
        arch,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        compute_prefix_cached_tokens=cached_tokens,
    )
    lower, upper = sorted((skipped.total, no_skip.total))
    return ResourceQuantity(
        value=None,
        status=QuantityStatus.INCOMPLETE,
        provenance=_flops_provenance(
            no_skip,
            arch,
            extra=(
                f"{extra};"
                f"flops_if_reuse={skipped.total};"
                f"flops_if_no_reuse={no_skip.total}"
            ),
        ),
        coverage=None,
        observed_subtotal=lower,
        observed_lower=lower,
        observed_upper=upper,
    )


def estimate_span_flops(span: InferenceSpan, model_id: str) -> ResourceQuantity:
    """FLOPs for one recorded request span.

    Unknown token fields stay unknown (never zero-filled). Cached tokens count as skipped
    prefill only when ``span.cache_provenance`` is a compute-reuse provenance; a provider
    billing cache or an unknown cache field gives the interval [prefix skipped, recomputed].
    """
    unknown = {
        QuantityStatus.UNKNOWN,
        QuantityStatus.NOT_APPLICABLE,
    }
    incomplete_field = unknown | {QuantityStatus.INCOMPLETE}
    if span.usage_status in unknown:
        return unknown_quantity(provenance="unknown_usage_no_flops_upgrade")
    if span.resolved_input_status in incomplete_field:
        return unknown_quantity(provenance="unknown_input_tokens_no_flops_upgrade")
    if span.resolved_output_status in incomplete_field:
        return unknown_quantity(provenance="unknown_output_tokens_no_flops_upgrade")
    try:
        arch = load_arch(model_id)
    except KeyError as exc:
        return unknown_quantity(provenance=f"unsupported_model:{model_id}:{exc}")
    if span.resolved_cached_status in incomplete_field:
        return _interval_quantity(
            arch,
            input_tokens=span.input_tokens,
            output_tokens=span.output_tokens,
            cached_tokens=span.input_tokens,
            extra="unknown_cache_not_compute_reuse",
        )
    reuse_known = span.cache_provenance in COMPUTE_REUSE_PROVENANCE
    billing_cache = span.cached_input_tokens > 0 and not reuse_known
    if billing_cache:
        return _interval_quantity(
            arch,
            input_tokens=span.input_tokens,
            output_tokens=span.output_tokens,
            cached_tokens=span.cached_input_tokens,
            extra="billing_cache_not_compute_reuse",
        )
    compute_cached = span.cached_input_tokens if reuse_known else None
    breakdown = estimate_request_flops(
        arch,
        input_tokens=span.input_tokens,
        output_tokens=span.output_tokens,
        compute_prefix_cached_tokens=compute_cached,
    )
    return estimated_quantity(breakdown, arch)
