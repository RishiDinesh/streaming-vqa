from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
import torch
import types
from typing import Optional

try:
    import flashinfer
except Exception:
    flashinfer = None


def flashinfer_rmsnorm_forward(self, hidden_states):
    if flashinfer is None:
        raise RuntimeError("flashinfer is not available for RMSNorm replacement.")
    bsz, seq_len, hidden_size = hidden_states.size()
    hidden_states = flashinfer.norm.rmsnorm(
        hidden_states.view(bsz * seq_len, hidden_size),
        self.weight,
        eps=self.variance_epsilon,
    )
    return hidden_states.view(bsz, seq_len, hidden_size)


def enable_flashinfer_rmsnorm(model):
    if flashinfer is None:
        print("flashinfer not found; keeping standard RMSNorm implementations.")
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
    if flashinfer is None:
        # Fallback RoPE path for non-flashinfer environments (e.g., ROCm-only installs).
        bsz, seq_len, _, head_dim = q.size()
        if offsets.numel() == 1:
            offsets = offsets.expand(bsz)
        offsets = offsets.to(device=q.device)
        positions = offsets[:, None] + torch.arange(seq_len, device=q.device)[None, :]
        positions = positions.to(dtype=torch.float32)
        if rope_scale != 0 and rope_scale != 1.0:
            positions = positions / float(rope_scale)

        inv_freq = 1.0 / (
            float(rope_theta)
            ** (
                torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
                / head_dim
            )
        )
        freqs = torch.einsum("bs,d->bsd", positions, inv_freq)
        cos = torch.cos(freqs).to(dtype=q.dtype)
        sin = torch.sin(freqs).to(dtype=q.dtype)

        cos = torch.cat([cos, cos], dim=-1).unsqueeze(2)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(2)

        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        q.copy_(q_rot)
        k.copy_(k_rot)
        return q, k

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
