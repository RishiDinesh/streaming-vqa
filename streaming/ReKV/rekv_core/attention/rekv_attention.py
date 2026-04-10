import copy
import torch
import torch.nn.functional as F
from typing import Optional

from .kv_cache_manager import ContextManager
from .dot_production_attention import get_multi_stage_dot_production_attention


def _bottom_right_causal_mask(
    len_q: int,
    len_k: int,
    device: torch.device,
) -> torch.Tensor:
    q_positions = torch.arange(len_k - len_q, len_k, device=device).unsqueeze(-1)
    k_positions = torch.arange(len_k, device=device).unsqueeze(0)
    return k_positions <= q_positions


def _full_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    enable_gqa = False
    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError(
                "Grouped-query attention requires query heads to be divisible by key/value heads: "
                f"q_heads={query.shape[1]} kv_heads={key.shape[1]}"
            )
        enable_gqa = True

    len_q = query.shape[-2]
    len_k = key.shape[-2]
    if len_q == len_k:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            enable_gqa=enable_gqa,
        )

    causal_mask = _bottom_right_causal_mask(len_q, len_k, query.device).view(1, 1, len_q, len_k)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        is_causal=False,
        enable_gqa=enable_gqa,
    )


def _rekv_local_init_attention(
    Attn,
    position_bias,
    h_q: torch.Tensor,
    h_k: torch.Tensor,
    h_v: torch.Tensor,
    *,
    batch_size: int,
    query_head_count: int,
    dim_head: int,
    len_q: int,
    len_k: int,
    n_local: int,
    n_init: int,
) -> torch.Tensor:
    h_q_, h_k_, h_v_ = h_q, h_k, h_v
    if len_q + n_local < h_k_.size(-2):
        h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local :, :]
        h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local :, :]

    local_h_q, local_h_k = position_bias(h_q_, h_k_)
    local_h_v = h_v_

    if len_k > n_local:
        init_h_q = position_bias.apply_rotary_pos_emb_one_angle(h_q, n_local)
        init_h_k = h_k[:, :, :n_init, :].contiguous()
        init_h_v = h_v[:, :, :n_init, :].contiguous()
    else:
        init_h_q = h_q
        init_h_k = torch.empty(
            (batch_size, h_k.shape[1], 0, dim_head),
            device=h_k.device,
            dtype=h_k.dtype,
        )
        init_h_v = torch.empty(
            (batch_size, h_v.shape[1], 0, dim_head),
            device=h_v.device,
            dtype=h_v.dtype,
        )

    attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
    attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local)
    attn.append(
        init_h_q,
        init_h_k,
        init_h_v,
        end=True,
        sliding_window=(len_k - len_q, n_local),
        complement_sliding_window=True,
    )
    score, _ = attn.get_result()
    return score.view(batch_size, query_head_count, len_q, dim_head)


def rekv_attention_forward(
    n_local, n_init, topk, chunk_size,
    block_size, max_cached_block,
    exc_block_size, fattn,
    async_global_stream=True,
    pin_memory=False,
    *args, **kwargs
):
    Attn, _ = get_multi_stage_dot_production_attention(fattn)
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out, 
                    dim_head, num_heads, num_heads_kv,
    ):

        """ 1. Project QKV """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()      # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
            position_bias = copy.deepcopy(position_bias)
            if position_bias.inv_freq.device != h_q.device:
                position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
            if position_bias._cos_cached is not None:
                position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
            if position_bias._sin_cached is not None:
                position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias,
                n_init, n_local, 
                block_size, max_cached_block, topk, chunk_size, exc_block_size,
                fattn,
                async_global_stream,
                pin_memory,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        # NOTE: Question-answering, fall back to sliding-window attention (infinite_lm)
        if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:
            if type(past_key_value) is ContextManager:  # retrieval
                if past_key_value.retrieved_block_indices is None:  # retrieve based on global_q (question's query)
                    past_k, past_v = past_key_value.get_retrieved_kv(global_q)
                else:  # retrieve based on pre-computed retrieved_block_indices
                    past_k, past_v = past_key_value.get_retrieved_kv()
                updata_kv_cache = False  # We do not update KV cache with the input KV (h_k, h_v) because we only use it for retrieval
            else:  # sliding-window attention
                past_k = past_key_value[0]
                past_v = past_key_value[1]
                updata_kv_cache = True

            """ 2. Update KV w/ past KV cache """
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            len_k += past_k.shape[2]

            """ 3. Update KV cache """
            if updata_kv_cache:
                if len_k <= n_local + n_init:
                    h_k_cache = h_k
                    h_v_cache = h_v
                else:
                    h_k_cache = torch.cat([h_k[:,:, :n_init, :], h_k[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                    h_v_cache = torch.cat([h_v[:,:, :n_init, :], h_v[:, :, max(0, h_k.size(-2) - n_local):, :]], dim=2)
                current_key_value = (h_k_cache, h_v_cache)
            else:
                current_key_value = (past_k, past_v)

            is_retrieval_request = (
                type(past_key_value) is ContextManager and past_key_value.to_retrieve
            )
            # For A+B we want ReKV to decide retrieval exactly as in the native method.
            # Duo only applies after retrieval, when the LM attends over the assembled context.
            duo_enabled = (
                not is_retrieval_request
                and hasattr(self, "full_attention_heads")
                and getattr(self, "rekv_duo_enabled", False)
            )
            if duo_enabled:
                full_attention_heads = self.full_attention_heads > 0.5
                num_full_kv_head = int(full_attention_heads.sum().item())
                num_full_query_head = num_full_kv_head * (num_heads // num_heads_kv)
                num_streaming_query_head = num_heads - num_full_query_head
                streaming_local_window = int(
                    min(n_local, getattr(self, "recent_size", n_local))
                )

                full_score = None
                streaming_score = None

                if num_full_query_head > 0:
                    full_q = h_q[:, :num_full_query_head, :, :]
                    full_k = h_k[:, :num_full_kv_head, :, :]
                    full_v = h_v[:, :num_full_kv_head, :, :]
                    full_q, full_k = position_bias(full_q, full_k)
                    full_score = _full_causal_attention(full_q, full_k, full_v)

                if num_streaming_query_head > 0:
                    streaming_q = h_q[:, num_full_query_head:, :, :]
                    streaming_k = h_k[:, num_full_kv_head:, :, :]
                    streaming_v = h_v[:, num_full_kv_head:, :, :]
                    streaming_score = _rekv_local_init_attention(
                        Attn,
                        position_bias,
                        streaming_q,
                        streaming_k,
                        streaming_v,
                        batch_size=batch_size,
                        query_head_count=num_streaming_query_head,
                        dim_head=dim_head,
                        len_q=len_q,
                        len_k=len_k,
                        n_local=streaming_local_window,
                        n_init=n_init,
                    )

                if full_score is None:
                    score = streaming_score
                elif streaming_score is None:
                    score = full_score
                else:
                    score = torch.cat([full_score, streaming_score], dim=1)
            else:
                score = _rekv_local_init_attention(
                    Attn,
                    position_bias,
                    h_q,
                    h_k,
                    h_v,
                    batch_size=batch_size,
                    query_head_count=num_heads,
                    dim_head=dim_head,
                    len_q=len_q,
                    len_k=len_k,
                    n_local=n_local,
                    n_init=n_init,
                )

            score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
            score = score.reshape(batch_size, len_q, num_heads * dim_head) # (batch, len_q, num_heads * dim_head)
            score = attention_out(score)

            return score, current_key_value

        # NOTE: Encode video, managed by the KVCacheManager
        else:
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )
            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            o = o.reshape(batch_size, len_q, dim_head * num_heads)
            o = attention_out(o)

            return o, past_key_value

    return forward
