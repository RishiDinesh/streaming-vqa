from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
import torch
import types
from typing import Optional

try:
    import flashinfer

    FLASHINFER_AVAILABLE = True
except ImportError:
    flashinfer = None
    FLASHINFER_AVAILABLE = False


def flashinfer_rmsnorm_forward(self, hidden_states):
    bsz, seq_len, hidden_size = hidden_states.size()
    hidden_states = flashinfer.norm.rmsnorm(
        hidden_states.view(bsz * seq_len, hidden_size),
        self.weight,
        eps=self.variance_epsilon,
    )
    return hidden_states.view(bsz, seq_len, hidden_size)


def enable_flashinfer_rmsnorm(model):
    if not FLASHINFER_AVAILABLE:
        print("flashinfer is unavailable; keeping native RMSNorm modules.")
        return model
    print("Replacing RMSNorm with Flashinfer's RMSNorm")
    for name, module in model.named_modules():
        if isinstance(module, LlamaRMSNorm):
            module.forward = types.MethodType(flashinfer_rmsnorm_forward, module)
        elif isinstance(module, MistralRMSNorm):
            module.forward = types.MethodType(flashinfer_rmsnorm_forward, module)
        elif isinstance(module, Qwen2RMSNorm):
            module.forward = types.MethodType(flashinfer_rmsnorm_forward, module)
    return model


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    offsets: torch.Tensor,
    rope_scale: float,
    rope_theta: float,
    indptr: Optional[torch.Tensor] = None,
):
    if not FLASHINFER_AVAILABLE:
        raise ImportError("flashinfer is required for apply_rope_inplace.")
    bsz, seq_len, num_heads, head_dim = q.size()
    _, _, num_kv_heads, _ = k.size()
    nnz = bsz * seq_len
    q = q.view(nnz, num_heads, head_dim)
    k = k.view(nnz, num_kv_heads, head_dim)
    if indptr is None:
        indptr = torch.tensor(
            [i * seq_len for i in range(bsz + 1)], dtype=torch.int32, device=q.device
        )
    if offsets.numel() == 1:
        offsets = offsets.expand(bsz).contiguous()
    flashinfer.rope.apply_rope_inplace(
        q,
        k,
        indptr,
        offsets,
        interleave=False,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    q = q.view(bsz, seq_len, num_heads, head_dim)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim)
    return q, k
