# ROCm Backend Audit Notes

This note tracks what is currently wrong, what is acceptable for AMD validation, and what still needs deeper work.

## What Was Wrong Before (Now Fixed)

- Treating AMD Duo fallback results as if they were equivalent to NVIDIA `block_sparse_attn` results.
- Looking only at library availability instead of recording the actual backend selected at runtime.
- Leaving `/opt/venv` versus `duo` ambiguous on MI300X even though the maintained launchers prefer `/opt/venv`.
- Running Duo-sensitive experiments without a strict mode that can fail fast when the streaming path falls back to SDPA.
- Mislabeling CUDA SDPA-fallback runs as `"rocm_baseline_duo"` — now fixed to `"sdpa_fallback_duo"` (hardware-neutral label).

## What Is Acceptable In Phase 1

- Using `/opt/venv` as the preferred MI300X runtime.
- Treating Duo on AMD as a ROCm engineering baseline when `block_sparse_attn` is unavailable.
- Using SDPA fallback for Duo only when that fallback is explicitly reported in the manifest and downstream comparisons.
- Treating `duo_plus_rekv` as an approximate multimodal hybrid rather than a literal reproduction of standalone DuoAttention paper semantics.

## Immediate Fixes Now In Tree

- Backend resolution is emitted into result/profile manifests for all methods.
- Duo methods now record:
  - full-attention backend
  - requested streaming backend
  - actual streaming backend
  - fallback reason
  - RoPE backend
  - RMSNorm backend
- Duo methods support `--duo-strict-no-sdpa-fallback` to fail if SDPA fallback would be used.
- A ROCm environment validator exists at:
  - [validate_rocm_env.py](/workspace/streaming-vqa/streaming/ReKV/validate_rocm_env.py)
- A one-video backend audit workflow exists at:
  - [run_streaming_rocm_audit_local.sh](/workspace/streaming-vqa/scripts/run_streaming_rocm_audit_local.sh)

## Longer-Term Fixes Not Implemented Yet

- Evaluate ROCm-native sparse attention alternatives that could replace the CUDA-only `block_sparse_attn` path.
- Add a true ROCm-native fast path for Duo streaming attention if a stable library or custom kernel becomes available.
- Revisit whether any flashinfer-like RMSNorm/RoPE replacement is worth supporting on ROCm for this multimodal eval path.
- If a ROCm-native sparse path is added, split result interpretation between:
  - ROCm baseline Duo
  - ROCm sparse-kernel Duo
  - NVIDIA sparse-kernel Duo

## Paper Alignment Reminder

- `rekv` is closer to paper intent on AMD today because its core retrieval/offload logic has a stable torch-based path.
- `duo_streaming` on AMD is paper-inspired but backend-divergent when it falls back away from sparse kernels.
- `duo_plus_rekv` remains an approximate multimodal hybrid and should be described that way in reports.
