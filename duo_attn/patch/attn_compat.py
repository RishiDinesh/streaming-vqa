from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
    from flash_attn.bert_padding import (
        index_first_axis as _index_first_axis,
        pad_input as _pad_input,
        unpad_input as _unpad_input,
    )

    FLASH_ATTN_AVAILABLE = True
except Exception:
    _flash_attn_func = None
    _flash_attn_varlen_func = None
    _flash_attn_with_kvcache = None
    _index_first_axis = None
    _pad_input = None
    _unpad_input = None
    FLASH_ATTN_AVAILABLE = False


def _repeat_kv_for_gqa(x: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_kv_heads = x.size(2)
    if num_kv_heads == num_query_heads:
        return x
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"Cannot expand KV heads {num_kv_heads} to query heads {num_query_heads}."
        )
    repeat = num_query_heads // num_kv_heads
    return torch.repeat_interleave(x, repeats=repeat, dim=2)


def _sdpa_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    # flash-attn style input: [batch, seq, heads, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if k.size(1) != q.size(1):
        k = _repeat_kv_for_gqa(k.transpose(1, 2), q.size(1)).transpose(1, 2)
        v = _repeat_kv_for_gqa(v.transpose(1, 2), q.size(1)).transpose(1, 2)

    if softmax_scale is not None:
        q = q * float(softmax_scale)

    attn_mask = None
    sdpa_is_causal = bool(causal)
    if causal and q.size(-2) != k.size(-2):
        # FlashAttention-style causal masking for cached decoding/prefill aligns
        # each query position to the *right* edge of the KV sequence. PyTorch's
        # generic `is_causal=True` path does not represent that offset when
        # q_len < k_len, so we build the bottom-right causal mask explicitly.
        q_len = q.size(-2)
        k_len = k.size(-2)
        query_positions = torch.arange(
            k_len - q_len,
            k_len,
            device=q.device,
        )
        key_positions = torch.arange(k_len, device=q.device)
        attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attn_mask = attn_mask.view(1, 1, q_len, k_len)
        sdpa_is_causal = False

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=float(dropout_p),
        is_causal=sdpa_is_causal,
    )
    return out.transpose(1, 2)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    if FLASH_ATTN_AVAILABLE:
        return _flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            **kwargs,
        )
    return _sdpa_flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    if FLASH_ATTN_AVAILABLE:
        return _flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            **kwargs,
        )

    outputs = []
    batch_size = cu_seqlens_q.numel() - 1
    for b in range(batch_size):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        k_start = int(cu_seqlens_k[b].item())
        k_end = int(cu_seqlens_k[b + 1].item())
        q_b = q[q_start:q_end].unsqueeze(0)
        k_b = k[k_start:k_end].unsqueeze(0)
        v_b = v[k_start:k_end].unsqueeze(0)
        out_b = flash_attn_func(
            q_b,
            k_b,
            v_b,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        ).squeeze(0)
        outputs.append(out_b)
    return torch.cat(outputs, dim=0) if outputs else q.new_zeros(q.shape)


def flash_attn_with_kvcache(*args, **kwargs):
    if FLASH_ATTN_AVAILABLE:
        return _flash_attn_with_kvcache(*args, **kwargs)
    raise RuntimeError(
        "flash_attn_with_kvcache requires flash-attn. "
        "Install flash-attn or avoid static-kv-cache flash-attn paths."
    )


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if FLASH_ATTN_AVAILABLE:
        return _index_first_axis(x, indices)
    return x.index_select(0, indices)


def unpad_input(
    x: torch.Tensor,
    padding_mask: torch.Tensor,
):
    if FLASH_ATTN_AVAILABLE:
        return _unpad_input(x, padding_mask)

    batch_size, seq_len = padding_mask.shape
    flat_mask = padding_mask.reshape(-1).to(dtype=torch.bool)
    indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
    x_flat = x.reshape(batch_size * seq_len, *x.shape[2:])
    x_unpad = x_flat.index_select(0, indices)
    seqlens = padding_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=padding_mask.device
    )
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    max_seqlen = int(seqlens.max().item()) if batch_size > 0 else 0
    return x_unpad, indices, cu_seqlens, max_seqlen


def pad_input(
    x_unpad: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    if FLASH_ATTN_AVAILABLE:
        return _pad_input(x_unpad, indices, batch_size, seq_len)

    out_shape = (batch_size * seq_len,) + tuple(x_unpad.shape[1:])
    out = x_unpad.new_zeros(out_shape)
    out.index_copy_(0, indices, x_unpad)
    return out.reshape(batch_size, seq_len, *x_unpad.shape[1:])
