import inspect
import warnings

import torch

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_VARLEN_AVAILABLE = True
except ImportError:
    _flash_attn_func = None
    _flash_attn_varlen_func = None
    FLASH_ATTN_AVAILABLE = False
    FLASH_ATTN_VARLEN_AVAILABLE = False


_FALLBACK_WARNING_EMITTED = False


def _warn_fallback_once() -> None:
    global _FALLBACK_WARNING_EMITTED
    if _FALLBACK_WARNING_EMITTED:
        return
    warnings.warn(
        "flash_attn is not available; falling back to PyTorch SDPA kernels. "
        "This is expected on many AMD/ROCm setups, but it may be slower.",
        RuntimeWarning,
        stacklevel=2,
    )
    _FALLBACK_WARNING_EMITTED = True


def supports_sdpa_enable_gqa() -> bool:
    try:
        return (
            "enable_gqa"
            in inspect.signature(
                torch.nn.functional.scaled_dot_product_attention
            ).parameters
        )
    except (TypeError, ValueError):
        return False


def supports_sdpa_scale() -> bool:
    try:
        return (
            "scale"
            in inspect.signature(
                torch.nn.functional.scaled_dot_product_attention
            ).parameters
        )
    except (TypeError, ValueError):
        return False


def repeat_kv_for_gqa_if_needed(query, key, value):
    num_heads = query.size(2)
    num_kv_heads = key.size(2)
    if num_heads == num_kv_heads:
        return key, value

    repeat_num = num_heads // num_kv_heads
    key = torch.repeat_interleave(key, repeats=repeat_num, dim=2)
    value = torch.repeat_interleave(value, repeats=repeat_num, dim=2)
    return key, value


def flash_attn_func(query, key, value, *args, **kwargs):
    if FLASH_ATTN_AVAILABLE:
        return _flash_attn_func(query, key, value, *args, **kwargs)

    _warn_fallback_once()

    dropout_p = kwargs.pop("dropout_p", None)
    if dropout_p is None and len(args) > 0:
        dropout_p = args[0]
    if dropout_p is None:
        dropout_p = 0.0

    causal = kwargs.pop("causal", True)
    softmax_scale = kwargs.pop("softmax_scale", None)
    kwargs.pop("window_size", None)
    kwargs.pop("alibi_slopes", None)
    kwargs.pop("deterministic", None)

    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)
    value_t = value.transpose(1, 2)

    sdpa_kwargs = {
        "dropout_p": dropout_p,
        "is_causal": causal,
    }
    if supports_sdpa_enable_gqa():
        sdpa_kwargs["enable_gqa"] = True
    else:
        key_t, value_t = repeat_kv_for_gqa_if_needed(query, key, value)
        key_t = key_t.transpose(1, 2)
        value_t = value_t.transpose(1, 2)

    if softmax_scale is not None and supports_sdpa_scale():
        sdpa_kwargs["scale"] = softmax_scale

    output = torch.nn.functional.scaled_dot_product_attention(
        query_t,
        key_t,
        value_t,
        **sdpa_kwargs,
    )
    return output.transpose(1, 2)


def flash_attn_varlen_func(*args, **kwargs):
    if not FLASH_ATTN_VARLEN_AVAILABLE:
        raise ImportError("flash_attn_varlen_func is unavailable in this environment.")
    return _flash_attn_varlen_func(*args, **kwargs)
