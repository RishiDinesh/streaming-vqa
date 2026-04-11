# ReKV Streaming Notes

## TL;DR

- Scope: the streaming `ReKV` lane only
- Datasets:
  - `RVS-Ego`
  - `RVS-Movie`
- Model:
  - `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- Working environment during the latest audit:
  - `/opt/venv`
  - AMD `MI300X`
- Main four comparison methods:
  - `full_streaming`
  - `duo_streaming`
  - `rekv`
  - `duo_plus_rekv`
- Kept full-eval result trees:
  - `outputs/evaluations_streaming/rvs-ego/full_eval_topk64_memavg`
  - `outputs/evaluations_streaming/rvs-movie/full_eval_topk64_memavg`
- Fresh audit subset trees:
  - `outputs/evaluations_streaming/rvs-ego/audit_firstq5`
  - `outputs/evaluations_streaming/rvs-movie/audit_firstq5`

## Current Repository State

This lane is now centered on the corrected streaming comparison flow.

Still kept:
- the streaming evaluation code under `streaming/ReKV/`
- the local launcher scripts under `scripts/`
- the retained full-eval result trees
- the fresh audit subset trees and profile audit outputs

Older artifacts intentionally removed earlier:
- old subsample result trees under `outputs/evaluations_streaming/*/subsample*`
- `artifacts/streaming_rekv_duo`
- stale local launcher clutter that no longer matched the active workflow

## Streaming Semantics

All four active comparison methods are expected to preserve the same streaming contract:

- causal frame ingest only
- one sampled frame per forward pass
- shared cache state across questions in the same video
- no offline full-video prefill
- question-time frame access determined by sampled timestamps strictly earlier than `end_time`
- identical sampled-frame schedule across methods

This is now explicitly written into the run metadata through the shared `evaluation_manifest`.

## What Changed In The Audit

The latest audit fixed several correctness and comparability problems.

### 1. Shared comparison manifest

File:
- [run_eval.py](/workspace/streaming-vqa/streaming/ReKV/run_eval.py)

What changed:
- every run now stores a shared `evaluation_manifest`
- manifests record:
  - common streaming settings
  - method family
  - runtime backend path
  - Duo deploy settings
  - retrieval/offload semantics
- resume now fails if comparison-critical settings drift

Why it matters:
- speed and quality comparisons are now tied to explicit runtime metadata instead of assumptions

### 2. Shared answer flow across methods

File:
- [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)

What changed:
- all methods now use a common prompt-prefill / answer path
- ReKV methods no longer double-apply the question prompt

Why it matters:
- this removed a silent source of unfairness between methods

### 3. ReKV answer-time context fix

Files:
- [kv_cache_manager.py](/workspace/streaming-vqa/streaming/ReKV/rekv_core/attention/kv_cache_manager.py)
- [rekv_attention.py](/workspace/streaming-vqa/streaming/ReKV/rekv_core/attention/rekv_attention.py)

What changed:
- ReKV answer-time assembly now force-includes the true local recent window
- the result JSONs now expose assembled-context diagnostics

Why it matters:
- native `rekv` and `duo_plus_rekv` can no longer accidentally answer from `init + retrieved-old` without the current local context

### 4. Hybrid semantics made explicit

File:
- [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)

What changed:
- `duo_plus_rekv` is now clearly labeled as an approximate hybrid
- ReKV retrieval remains native
- Duo is applied over the assembled answer-time ReKV context

Why it matters:
- this avoids overstating the hybrid as the same thing as official standalone DuoAttention eval

### 5. ROCm-specific decode fix

File:
- [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)

What changed:
- clone `last_hidden_state` before LM-head projection on the ReKV decode path

Why it matters:
- this fixed the MI300X / ROCm inference-tensor crash seen during real subset evaluation

### 6. Plot and profile cleanup

Files:
- [plot_results.py](/workspace/streaming-vqa/streaming/ReKV/plot_results.py)
- [plot_profile.py](/workspace/streaming-vqa/streaming/ReKV/plot_profile.py)
- [profile_streaming.py](/workspace/streaming-vqa/streaming/ReKV/profile_streaming.py)
- [run_streaming_profile_local.sh](/workspace/streaming-vqa/scripts/run_streaming_profile_local.sh)

What changed:
- plots were restyled to be cleaner and easier to read
- profile outputs now include the same comparison manifest
- profile launcher supports `duo`
- profile launcher no longer blindly passes a missing feature-cache path

## Runtime / Backend Findings On MI300X

Environment findings from the live audit runs:

- `flash_attn` load path was active for all four methods
- `block_sparse_attn` was not available in this ROCm environment
- `flashinfer` was not available
- during live Duo profiling, `rocm-smi` showed about `95%` GPU usage

Interpretation:
- these runs are using real GPU execution on MI300X
- Duo comparisons here should not be described as using NVIDIA-only block-sparse kernels

## Duo Deployment Settings

The corrected streaming audit used training-aligned Duo deploy settings:

- `sink=512`
- `recent=1024`

This matters because the older reduced deployment assumption:

- `sink=256`
- `recent=512`

was useful in offline-only work but should not be treated as the default streaming comparison setting.

## How The Local Pipeline Works

### 1. Dataset loading and causal sampling

File:
- [datasets.py](/workspace/streaming-vqa/streaming/ReKV/datasets.py)

Behavior:
- loads `RVS-Ego` or `RVS-Movie`
- sorts conversations in causal order
- resolves local or HF video paths
- samples frames at the requested FPS
- uses sampled timestamps as the causal schedule

### 2. Shared visual feature cache

Files:
- [feature_cache.py](/workspace/streaming-vqa/streaming/ReKV/feature_cache.py)
- [precompute_features.py](/workspace/streaming-vqa/streaming/ReKV/precompute_features.py)

Behavior:
- precomputes per-frame OneVision visual features
- validates cache compatibility against the same sampled-frame schedule
- lets all methods reuse the same frame features fairly

### 3. Streaming runner

File:
- [run_eval.py](/workspace/streaming-vqa/streaming/ReKV/run_eval.py)

Behavior:
- resets state per video
- ingests only newly available sampled frames before each question
- answers from the current method state
- checkpoints at video and conversation boundaries
- writes method-specific and shared comparison metadata

### 4. Method implementations

File:
- [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)

Method meanings:
- `full_streaming`
  - standard streaming cache, no ReKV retrieval, no Duo sparsification
- `duo_streaming`
  - DuoAttention-only streaming baseline
- `rekv`
  - main ReKV streaming method
- `rekv_no_offload`
  - ablation
- `duo_plus_rekv`
  - native ReKV retrieval plus Duo over the final answer-time context

### 5. ReKV backend

Files:
- [rekv_attention.py](/workspace/streaming-vqa/streaming/ReKV/rekv_core/attention/rekv_attention.py)
- [kv_cache_manager.py](/workspace/streaming-vqa/streaming/ReKV/rekv_core/attention/kv_cache_manager.py)

Behavior:
- keeps a live local window
- stores older blocks for retrieval
- retrieves relevant blocks at answer time
- force-includes the local window
- records block/timestamp diagnostics

## Relation To The Papers

### ReKV paper

Paper:
- https://openreview.net/pdf/e491e49e823ff5b84c75cc8daad0e1b80ebfd206.pdf

Relevant paper idea:
- sliding-window stream ingest
- preserve old KV outside the live GPU window
- retrieve only relevant old KV blocks at question time

The repo still follows that core logic, but in a local adapted implementation rather than a literal copy of the original engineering path.

### DuoAttention paper

Paper:
- https://arxiv.org/abs/2410.10819

Relevant paper idea:
- only some heads need full-context retrieval behavior
- other heads can run sink/recent streaming attention

This repo adapts DuoAttention to OneVision/Qwen2 and also exposes the hybrid `duo_plus_rekv` path.

## Latest Audit Results

These are from the corrected first-5-video / first-question subset runs.

### `RVS-Movie / audit_firstq5`

- `full_streaming`
  - judge `0.80`
  - answer latency `0.9899s`
  - TTFT `0.0416s`
  - peak GPU `2.7916 GB`
- `duo_streaming`
  - judge `0.76`
  - answer latency `0.9314s`
  - TTFT `0.0476s`
  - peak GPU `2.8436 GB`
  - Duo window `512/1024`
- `rekv`
  - judge `0.80`
  - answer latency `0.1398s`
  - TTFT `0.0036s`
  - peak GPU `3.2408 GB`
  - avg retrieved blocks `57.2`
  - retrieval latency `0.6397s`
  - forced local window rate `1.0`
- `duo_plus_rekv`
  - judge `0.80`
  - answer latency `0.1923s`
  - TTFT `0.0037s`
  - peak GPU `3.2667 GB`
  - avg retrieved blocks `56.6`
  - retrieval latency `0.6457s`
  - forced local window rate `1.0`
  - Duo window `512/1024`

### `RVS-Ego / audit_firstq5`

- `full_streaming`
  - judge `0.80`
  - answer latency `0.7754s`
  - TTFT `0.0477s`
  - peak GPU `3.1281 GB`
- `duo_streaming`
  - judge `0.76`
  - answer latency `0.9886s`
  - TTFT `0.0527s`
  - peak GPU `3.2209 GB`
  - Duo window `512/1024`
- `rekv`
  - judge `0.76`
  - answer latency `0.3432s`
  - TTFT `0.0017s`
  - peak GPU `3.1280 GB`
  - avg retrieved blocks `64.0`
  - retrieval latency `0.6930s`
  - forced local window rate `1.0`
- `duo_plus_rekv`
  - judge `0.76`
  - answer latency `0.3599s`
  - TTFT `0.0018s`
  - peak GPU `3.1643 GB`
  - avg retrieved blocks `64.0`
  - retrieval latency `0.6702s`
  - forced local window rate `1.0`
  - Duo window `512/1024`

## How To Read Those Results

What the subset audit does support:
- the four methods are now being run under a much fairer comparison contract
- the backend path is explicit
- ReKV and hybrid both include the local recent window at answer time
- ReKV answer-time latency is much lower than the non-ReKV baselines on these subset runs

What it does not support yet:
- a clear hybrid quality win over native `rekv`
- a claim that Duo is underperforming because it was simply misconfigured in the old obvious ways

The corrected result is less exciting, but more trustworthy.

## Practical Guidance For Next Work

If you keep working only on this lane, prioritize:

1. [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)
2. [run_eval.py](/workspace/streaming-vqa/streaming/ReKV/run_eval.py)
3. [kv_cache_manager.py](/workspace/streaming-vqa/streaming/ReKV/rekv_core/attention/kv_cache_manager.py)
4. [rekv_attention.py](/workspace/streaming-vqa/streaming/ReKV/rekv_core/attention/rekv_attention.py)
5. [plot_results.py](/workspace/streaming-vqa/streaming/ReKV/plot_results.py)
6. [profile_streaming.py](/workspace/streaming-vqa/streaming/ReKV/profile_streaming.py)

Lower priority:
- broad `duo_attn/` training code
- older removed subsample narratives
- non-streaming paths elsewhere in the repo

## Launchers To Use

Primary:
- [run_streaming_full_eval_local.sh](/workspace/streaming-vqa/scripts/run_streaming_full_eval_local.sh)

Useful helpers:
- [run_streaming_subsample5_local.sh](/workspace/streaming-vqa/scripts/run_streaming_subsample5_local.sh)
- [run_streaming_subsample_matrix_local.sh](/workspace/streaming-vqa/scripts/run_streaming_subsample_matrix_local.sh)
- [run_streaming_profile_local.sh](/workspace/streaming-vqa/scripts/run_streaming_profile_local.sh)

## Current Takeaway

The safest current statement is:

- native `rekv` is still the main method to trust and extend
- `duo_plus_rekv` is now in a better correctness state and is a fairer comparison than before
- the retained full-eval `topk=64` results and the new audited subset runs still do not justify claiming a clear default hybrid win over native `rekv`
