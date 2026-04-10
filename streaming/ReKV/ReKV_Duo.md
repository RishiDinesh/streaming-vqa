# ReKV vs DuoAttention

## TL;DR
- Goal: evaluate streaming `DuoAttention`, `ReKV`, and `DuoAttention + ReKV` for streaming VideoQA on `LLaVA-OneVision 0.5B`.
- Hardware assumption: one local AMD `MI300X`, no SLURM for this lane.
- Streaming semantics used throughout:
  - causal frame ingest
  - one sampled frame per forward pass
  - shared state across questions in the same video
  - no offline full-video prefill
  - no LongBench-style tail replay
- Current promoted hybrid:
  - `duo_plus_rekv`
  - `sparsity = 0.75`
  - `deploy_sink_size = 256`
  - `deploy_recent_size = 512`
  - `retrieve_size = 32`
  - `n_local = 15000`
- Current strongest overall standalone method on the validated subsamples: `rekv`
- `duo_n_init = 601` (block-aligned sink token count: system prompt + 3 frames)
- Current status:
  - figure-ready subsample / profiling workflow is implemented
  - optimized full-eval workflow is implemented
  - safe single-GPU speed knobs are implemented for the local MI300X path
  - sink token mismatch fixed (n_init raised from 13 to 601 for duo_plus_rekv)
  - tuple KV cache patching removed from duo_plus_rekv init
  - local `duo` Conda env is installed at `/root/miniforge3/envs/duo`
  - direct local ROCm execution is validated in this environment
  - `rekv` smoke eval + judge + plots now run end to end on the 0.5B model
  - matched-topk subset tuning now supports both peak and average memory metrics
  - current active reruns are subset-only, cached, and do not rerun precompute
  - full-dataset runs were paused until subset + memory-instrumentation checks finish

## Current Direction Update

As of `2026-04-10`, the active ReKV lane should reflect the following project decisions:

- We care about **both** streaming datasets:
  - `RVS-Ego`
  - `RVS-Movie`
- The intended streaming paper story is:
  - `DuoAttention` alone is **not** expected to beat the streaming state of the art.
  - The more important claim is that `DuoAttention` can act as an **add-on to ReKV** to reduce memory while preserving most of ReKV's accuracy.
- In addition to the currently promoted hybrid setting, we want targeted subsample experiments for:
  - `duo_plus_rekv` with `sparsity = 0.50`
  - `duo_plus_rekv` with `sparsity = 0.75`
- We do **not** plan to spend compute on a `duo_streaming (s=0.0)` control unless a later review shows it is necessary.
- These subsample sweeps are mainly for checking whether higher Duo sparsity on top of ReKV gives:
  - acceptable accuracy retention relative to native `rekv`
  - lower latency at matched quality
  - better peak and/or average memory behavior

## External Paper Notes

### Original ReKV paper / repo

After reviewing the original paper and repository:

- ReKV's core design is:
  - streaming video ingest with sliding-window attention
  - preservation of old KV cache outside the live GPU window
  - question-time retrieval of relevant KV blocks
- The official repo explicitly positions offloading as part of the method:
  - old KV cache can be stored in `RAM` or `disk`
  - retrieved KV is reloaded for answer-time reasoning
- The official repo also supports both:
  - `rvs_ego`
  - `rvs_movie`

### StreamMem note about `ReKV w/o offloading`

After reviewing `StreamMem`:

- `StreamMem` reports both:
  - `ReKV`
  - `ReKV w/o offloading`
- Their text defines `ReKV w/o offloading` as a memory-constrained variant that:
  - does **not** keep old KV via CPU offloading
  - instead discards older KV and keeps only recent context as short-term memory
- In other words, this is not the full ReKV method.
  It is a constrained ablation that removes ReKV's long-range storage mechanism.

## Relevance of `ReKV w/o offloading` for this repo

Current recommendation:

- `ReKV w/o offloading` is **relevant as an optional ablation**, not as the main comparison target.
- It is useful if we want to answer:
  - whether Duo+ReKV is only helping because of a favorable memory budget
  - how much long-range memory really matters under a strict no-offload setting
  - whether Duo can provide a cleaner memory/accuracy tradeoff than a naive short-memory fallback
- It is **not** the main story for this project because:
  - our current ReKV implementation already behaves like an in-process retrieval system rather than the official CPU/disk-offload pipeline
  - the central claim we want is `ReKV + Duo` versus native `ReKV`, not versus a deliberately weakened ReKV ablation

Practical interpretation for this codebase:

- The local analogue is now implemented as:
  - `rekv_no_offload`
- Semantics:
  - disable long-range retained retrieval memory
  - keep only the recent/local live KV window plus init tokens
  - answer from short-term memory only
- It should be treated as an ablation, not as the main promoted ReKV setting.

## Immediate Next Experimental Priorities

1. Finish cached subset reruns for both `RVS-Ego` and `RVS-Movie` at:
   - matched `topk=32`
   - default `topk=64`
2. Use the new plots to compare:
   - peak GPU memory
   - average GPU memory
   - peak CPU/offloaded KV
   - average CPU/offloaded KV
   - latency / quality tradeoffs
3. Decide whether the paper story is:
   - default-configuration reproduction (`topk=64`)
   - or matched-budget Duo-vs-ReKV (`topk=32`)
4. Only after the updated subset figures look convincing, relaunch any full-dataset runs.

## Single-MI300X Speed Notes

Because this lane currently assumes access to one large local `MI300X`, the safe way to speed things up is to reduce overhead **without changing streaming semantics**.

Implemented safe speed knobs:

- `torch.cuda.empty_cache()` is no longer forced on every per-video reset.
  - This avoids repeated allocator stalls between videos.
  - It can still be re-enabled explicitly with:
    - `CLEAR_CUDA_CACHE_ON_RESET=1`
- Video decode now supports configurable CPU-side threading:
  - `VIDEO_DECODE_THREADS=<n>`
  - This only affects host-side frame decode and does not change sampled timestamps or model semantics.
- The shared feature cache remains the main paper-safe acceleration path:
  - precompute visual features once
  - reuse them across all methods
- Result logging for ReKV-based methods is derived from already-computed state.
  - No extra benchmark-time forward passes were added for the new plots/figures.
- Memory instrumentation now records both peak and average-style aggregates:
  - GPU: `peak_memory_bytes`, `avg_gpu_memory_bytes_current`
  - CPU/offload: `peak_cpu_offload_bytes`, `avg_cpu_offload_bytes_current`

What we should still avoid for correctness:

- batching multiple streamed frames into one LM ingest step
- answering multiple questions from the same video in parallel
- running multiple measured methods concurrently on the same GPU for final latency/memory numbers
- changing the feature-extraction path to an approximation that does not match online ingest

Recommended local settings for preview and full runs:

- keep `USE_FEATURE_CACHE=1`
- leave `CLEAR_CUDA_CACHE_ON_RESET=0` unless fragmentation becomes a real issue
- try `VIDEO_DECODE_THREADS=4` first, then increase if CPU decode is clearly the bottleneck

## Current Matched-TopK Findings

### `RVS-Ego / subsample5`

Matched `topk=64`:

- `rekv topk=64`: score `0.7733`, latency `0.7978s`, GPU `2.668 GB`, CPU `0.942 GB`
- `duo_plus_rekv topk=64`: score `0.7600`, latency `0.9185s`, GPU `2.671 GB`, CPU `0.938 GB`

Interpretation:

- At the default ReKV setting, `duo_plus_rekv` does **not** currently help on this subset.

Matched `topk=32`:

- `rekv topk=32`: score `0.7733`, latency `0.6350s`, GPU `2.437 GB`, CPU `0.942 GB`
- `duo_plus_rekv topk=32`: score `0.7733`, latency `0.5686s`, GPU `2.444 GB`, CPU `0.938 GB`

Interpretation:

- At a matched tuned retrieval budget, `duo_plus_rekv` improves latency while preserving quality.
- This is currently a latency-at-matched-quality story, not a peak-GPU-memory win story.

### Consequence for current reporting

- If staying closest to the original ReKV paper defaults, keep `topk=64` and treat `rekv` as the stronger default baseline.
- If the goal is a fair Duo-vs-ReKV comparison, use matched settings, especially `topk=32`.
- Because peak-only memory can hide different steady-state behavior, the pipeline now emits average-memory plots as well.
- keep `FLUSH_EVERY_CONVERSATIONS=1` so interrupted long runs can resume from inside a video
- only change `FLUSH_EVERY_VIDEOS` if we want a small I/O reduction and are comfortable with less frequent checkpointing

Practical compatibility fallbacks now present in the repo:

- local launcher scripts still prefer the `duo` Conda env, but can fall back to the current Python with a warning if `duo` is not installed and required packages are present
- video sampling still prefers `decord`, but can fall back to `imageio` when `decord` is unavailable

Current environment note:

- in this session, the local ROCm path is healthy:
  - `rocminfo` succeeds
  - `rocm-smi` sees one GPU
  - `torch.cuda.is_available()` is `True`
  - Torch allocation on `cuda:0` succeeds
- `sinfo` / SLURM are not available in this environment, so this lane should use
  the direct local ROCm path here
- the `duo` env is installed at `/root/miniforge3/envs/duo` (torch 2.4 + ROCm 6.1 — **no longer the preferred env**)
- preferred env: `/opt/venv` (torch 2.9 + ROCm 7.0) — all launcher scripts now prefer this automatically
- `PYTORCH_COMPILE=0` disables `torch.compile` if it causes issues; default is auto-enable on ROCm 7+
- if a future shell instead reports `Unable to open /dev/kfd read-write: Operation not permitted` and `torch.cuda.is_available()` is `False`, the issue is the current shell/container session, not the streaming runner logic
- use [check_mi300x_access.sh](/workspace/streaming-vqa/scripts/check_mi300x_access.sh) before launching long runs

## Session-Validated Compatibility Fixes

During the `2026-04-10` smoke pass on `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`,
several compatibility fixes were required and are now in the repo:

1. ReKV patching for the current `Qwen2Model` layout
- File: [patch.py](/workspace/streaming-vqa/streaming/ReKV/rekv_core/patch.py)
- `patch_hf` now supports:
  - bare `Qwen2Model` language-model cores without a `.model` wrapper
  - newer Qwen2 attention modules that do not expose legacy `num_heads` and
    `num_key_value_heads` attributes directly
  - newer decoder-layer forward conventions used by the current `transformers`
    stack

2. OneVision video feature extraction compatibility
- File: [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)
- The streaming feature extractor now uses `get_video_features(...)` when the
  model exposes it, instead of assuming the outer model has a direct
  `apply_pooling(...)` helper.

3. Greedy decoding compatibility for bare language-model outputs
- File: [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)
- The current OneVision stack returns `last_hidden_state` from
  `model.language_model` rather than full `logits`.
- Greedy decoding now projects through the outer model's LM head when needed.

4. Judge path compatibility
- File: [judge_results.py](/workspace/streaming-vqa/streaming/ReKV/judge_results.py)
- The judge now uses the same LM-head projection fallback as the main runner.

5. Safer batched video decode fallback
- File: [datasets.py](/workspace/streaming-vqa/streaming/ReKV/datasets.py)
- If batched `decord` reads fail for a particular mp4, the sampler now falls
  back to single-frame fetches without changing the sampled-frame schedule.

6. ROCm 7 / transformers 4.55 compatibility and 3–5x speedup (`2026-04-10`)
- Root cause: launcher scripts were activating the `duo` conda env
  (torch 2.4.1 + ROCm 6.1) instead of `/opt/venv` (torch 2.9 + ROCm 7.0).
  ROCm 7 ships tuned gfx942 (MI300X) HIP kernels and is dramatically faster.
- Measured speedups on MI300X (bfloat16, 0.5B model):
  - Vision encoder: 18.7 ms → 8.1 ms per frame (**2.3×**)
  - LM decode 1 token (3000 ctx): 60.8 ms → 11.8 ms (**5.1×**)
  - Answer generation (64 tokens): ~3.9 s → ~0.8 s
  - `torch.compile(mode='reduce-overhead')` adds a further ~1.5× on ROCm 7
- Files changed:
  - [scripts/run_streaming_subsample5_local.sh](/workspace/streaming-vqa/scripts/run_streaming_subsample5_local.sh)
  - [scripts/run_streaming_subsample_matrix_local.sh](/workspace/streaming-vqa/scripts/run_streaming_subsample_matrix_local.sh)
  - [scripts/run_streaming_full_eval_local.sh](/workspace/streaming-vqa/scripts/run_streaming_full_eval_local.sh)
  - [scripts/run_streaming_profile_local.sh](/workspace/streaming-vqa/scripts/run_streaming_profile_local.sh)
  - `activate_duo_env` now checks `/opt/venv` first; falls back to `duo` conda if not present or missing packages.
- `torch.compile` and `torch.set_float32_matmul_precision('high')` applied at startup:
  - File: [run_eval.py](/workspace/streaming-vqa/streaming/ReKV/run_eval.py)
  - Enabled automatically when ROCm 7+ is detected; can be disabled with `PYTORCH_COMPILE=0`.
- transformers 4.55 layout change: `model.language_model` is now `Qwen2Model` directly (no `.model` wrapper):
  - File: [duo_attn/patch/llava_onevision.py](/workspace/streaming-vqa/duo_attn/patch/llava_onevision.py)
  - `_get_qwen2_layers`, `get_qwen2_full_attention_heads`, `set_qwen2_full_attention_heads`,
    `map_qwen2_full_attention_heads` all now use `lang.layers if isinstance(lang, Qwen2Model) else lang.model.layers`.
  - File: [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)
  - `DuoPlusReKVStreamingMethod.__init__` uses the same `.layers` fallback for the `rekv_duo_enabled` flag loop.
- decord threaded-decoder crash fix:
  - File: [datasets.py](/workspace/streaming-vqa/streaming/ReKV/datasets.py)
  - `SampledVideo.get_frames` and `get_frame` now fall back to a single-threaded
    decord reader, then imageio, before reaching `__getitem__`/`seek_accurate` which
    crashes on some mp4s with the threaded FFmpeg decoder.

## Session Smoke Validation

A local ROCm smoke run for streaming `rekv` on the 0.5B model completed
successfully in this session:

- command shape:
  - `MAX_VIDEOS=1 MAX_CONVERSATIONS=1 VIDEO_DECODE_THREADS=1 bash scripts/run_streaming_subsample5_local.sh rekv`
- outputs:
  - JSON: [rekv_topk64_nlocal15000.json](/workspace/streaming-vqa/outputs/evaluations_streaming_smoke/rvs-ego/subsample5_1x1/rekv/rekv_topk64_nlocal15000.json)
  - plots: [plots](/workspace/streaming-vqa/outputs/evaluations_streaming_smoke/rvs-ego/subsample5_1x1/plots)
- smoke metrics:
  - `completed_videos = 1/1`
  - `total_frames_ingested = 150`
  - `avg_answer_latency_sec = 0.4832`
  - `peak_memory_bytes = 2929964032`
  - `avg_judge_score = 0.8`
- progress reporting was visibly confirmed:
  - outer `videos` bar
  - inner per-video `frames` bar
  - ETA / rate displayed by `tqdm`

## Long-Run Safety

The streaming eval runner now supports **partial-video resume** in addition to whole-video resume.

What is checkpointed:

- completed videos
- run metadata / aggregate progress
- optional `in_progress_video`
  - completed conversations so far
  - frame-ingest progress so far
  - enough metadata to replay the video stream and continue safely

How resume works:

- the output JSON is updated in place during the run
- if a run stops after some conversations inside a video, `--resume` reconstructs the stream state by:
  - resetting the method
  - replaying frames up to the last checkpointed frame index
  - continuing from the next unanswered conversation
- this avoids trying to serialize live model KV state directly

Current checkpoint defaults in the shell launchers:

- `FLUSH_EVERY_VIDEOS=1`
- `FLUSH_EVERY_CONVERSATIONS=1`

Progress reporting is now implemented for the main long-running steps:

- streaming eval:
  - outer video bar
  - inner per-video frame bar
- feature precompute:
  - outer video bar
  - inner per-video frame/batch bar
- profiling:
  - per-video frame bar
  - probe bar
- judge:
  - conversation bar

## What This Project Is
We are comparing four main streaming inference strategies on long-video streaming QA,
plus one short-memory ablation:

| Method | Idea |
| --- | --- |
| `full_streaming` | plain streaming baseline with no DuoAttention and no ReKV |
| `duo_streaming` | same streaming runner, but decoder attention uses DuoAttention |
| `rekv` | same streaming runner, but long-range memory is handled by ReKV retrieval |
| `rekv_no_offload` | ReKV-style local-window path, but old context is discarded instead of being kept for long-range retrieval |
| `duo_plus_rekv` | ReKV retrieval plus DuoAttention only in the post-retrieval LM attention path |

This project is deliberately about **streaming** behavior, so the runner never assumes the whole video is known in advance.

## Streaming Basics

### How the stream is processed
For each video:
1. Sample frames at `0.5 FPS`.
2. Ingest sampled frames causally, one frame at a time.
3. When a question arrives at time `t`, only sampled frames with `timestamp < t` are available.
4. Answer from the method’s current state.

### What “current state” means
- `full_streaming` / `duo_streaming`: the live decoder KV cache after all frames seen so far.
- `rekv`: live KV cache plus ReKV’s retained retrievable memory.
- `duo_plus_rekv`: same ReKV retrieval state, then DuoAttention in the answer-time LM path.

### A+B meaning in this repo
`duo_plus_rekv` is intentionally implemented as:
- retrieval policy = **native ReKV**
- answer attention policy = **DuoAttention post-retrieval**

That means Duo does **not** interfere with ReKV’s retrieval selection.

## Repo Map

### Main code
- [run_eval.py](/workspace/streaming-vqa/streaming/ReKV/run_eval.py): main streaming runner
- [datasets.py](/workspace/streaming-vqa/streaming/ReKV/datasets.py): `RVS-Ego` / `RVS-Movie` dataset adapters and sampling
- [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py): `full_streaming`, `duo_streaming`, `rekv`, `rekv_no_offload`, `duo_plus_rekv`
- [feature_cache.py](/workspace/streaming-vqa/streaming/ReKV/feature_cache.py): strict feature-cache metadata/load helpers
- [precompute_features.py](/workspace/streaming-vqa/streaming/ReKV/precompute_features.py): precompute shared frame features
- [judge_results.py](/workspace/streaming-vqa/streaming/ReKV/judge_results.py): post-hoc local judge rescoring
- [plot_results.py](/workspace/streaming-vqa/streaming/ReKV/plot_results.py): plot generation
- [compare_subsamples.py](/workspace/streaming-vqa/streaming/ReKV/compare_subsamples.py): cross-slice comparisons
- [profile_streaming.py](/workspace/streaming-vqa/streaming/ReKV/profile_streaming.py): fixed-horizon profiling runner
- [plot_profile.py](/workspace/streaming-vqa/streaming/ReKV/plot_profile.py): profiling-curve plotting
- [build_qualitative_bundle.py](/workspace/streaming-vqa/streaming/ReKV/build_qualitative_bundle.py): machine-readable / markdown qualitative bundle export

### Local launchers
- [run_streaming_subsample5_local.sh](/workspace/streaming-vqa/scripts/run_streaming_subsample5_local.sh): subsample runs, judge, plots, qualitative bundle
- [run_streaming_subsample_matrix_local.sh](/workspace/streaming-vqa/scripts/run_streaming_subsample_matrix_local.sh): multi-slice subsample runner with logs + `status.txt`
- [run_streaming_full_eval_local.sh](/workspace/streaming-vqa/scripts/run_streaming_full_eval_local.sh): optimized local full eval
- [run_streaming_profile_local.sh](/workspace/streaming-vqa/scripts/run_streaming_profile_local.sh): low-overhead fixed-horizon profiling lane
- [check_mi300x_access.sh](/workspace/streaming-vqa/scripts/check_mi300x_access.sh): ROCm / Torch visibility diagnostic for the current shell

## Architecture Diagram

```mermaid
flowchart TD
    A[Dataset Adapter\nRVS-Ego / RVS-Movie] --> B[Video Sample\nvideo_id, path, duration, conversations]
    B --> C[Frame Sampler\n0.5 FPS causal schedule]
    C --> D{Input Source}
    D -->|Raw frames| E[Video Decode\nSampledVideo.get_frame / get_frames]
    D -->|Cached features| F[Feature Cache\nprecompute_features.py]
    E --> G[Per-frame Visual Features]
    F --> G
    G --> H[Streaming Runner\nrun_eval.py]
    H --> I{Method Wrapper}
    I -->|full_streaming| J[Plain LM KV Cache]
    I -->|duo_streaming| K[DuoAttention KV Cache]
    I -->|rekv| L[ReKV Local KV + Retrieval Memory]
    I -->|duo_plus_rekv| M[ReKV Retrieval + DuoAttention LM]
    J --> N[Question-time Answer]
    K --> N
    L --> N
    M --> N
    N --> O[Result JSON]
    O --> P[Judge / Lexical Scoring]
    P --> Q[Plots + Comparison Bundles + Final Package]
```

## Workflow Diagram

Diagram status:
- No architecture-diagram change is required after the causality review.
- The only clarification is that question-time availability is defined by sampled timestamps
  (`timestamp < end_time`), not by a rounded FPS formula.

```mermaid
flowchart LR
    A[1. Load dataset slice] --> B[2. Build causal sampled-frame schedule]
    B --> C{Use feature cache?}
    C -->|No| D[Decode sampled frames]
    C -->|Yes| E[Load cached per-frame features]
    D --> F[Ingest one frame at a time]
    E --> F
    F --> G[Update method state]
    G --> H{Question end_time reached?}
    H -->|No| F
    H -->|Yes| I[Answer from current state]
    I --> J[Save per-conversation stats]
    J --> K{More conversations in video?}
    K -->|Yes| F
    K -->|No| L[Save per-video JSON checkpoint]
    L --> M[Aggregate metrics]
    M --> N[Judge / rescore]
    N --> O[Plots / final package]
```

## Method Architecture

```text
full_streaming
  sampled frames -> visual features -> LM KV cache -> answer

duo_streaming
  sampled frames -> visual features -> DuoAttention LM KV cache -> answer

rekv
  sampled frames -> visual features -> local KV + retrievable memory
  question -> ReKV retrieval -> answer

duo_plus_rekv
  sampled frames -> visual features -> local KV + retrievable memory
  question -> native ReKV retrieval -> DuoAttention answer-time LM
```

## Mental Model Diagram

```text
1. duo_streaming

video stream
  -> per-frame visual features
  -> live decoder attention over the stream
     - some heads keep long-range access
     - some heads use sink + recent window
  -> answer

Key idea:
  Keep streaming context live, but make attention cheaper.


2. rekv

video stream
  -> per-frame visual features
  -> local live KV + retrievable long-range memory
  -> question arrives
  -> retrieve a small relevant subset of old context
  -> answer

Key idea:
  Do not keep all old context live; retrieve what matters at question time.


3. duo_plus_rekv

video stream
  -> per-frame visual features
  -> local live KV + retrievable long-range memory
  -> question arrives
  -> native ReKV retrieval picks the relevant old context
  -> DuoAttention runs on the reduced answer-time LM context
  -> answer

Key idea:
  ReKV solves long-range memory first, then DuoAttention tries to make the
  remaining answer-time attention cheaper.
```

## What Was Implemented

### Core streaming pipeline
- Added normalized dataset adapters for:
  - `RVS-Ego`
  - `RVS-Movie`
- Added a causal streaming runner with:
  - atomic JSON checkpoints
  - `--resume`
  - `--overwrite-output`
  - `--flush-every-videos`
  - deterministic `--video-offset`
  - per-video progress output

### Methods
- Added:
  - `full_streaming`
  - `duo_streaming`
  - `rekv`
  - `rekv_no_offload`
  - `duo_plus_rekv`

### Evaluation and reporting
- Added:
  - lexical open-ended scoring bundle
  - local LLM-judge rescoring
  - per-slice plots
  - cross-slice comparison bundles
  - qualitative bundle export
  - fixed-horizon profiling outputs and plots
  - curated final package under `outputs/evaluations_streaming/final_subsample_package/`

### Figure-ready logging
- Result JSONs now preserve:
  - sample-level `extra_metadata`
  - conversation-level `extra_metadata`
  - `cpu_offload_bytes_current`
  - `cpu_offload_bytes_peak`
  - retrieved block / timestamp unions for ReKV-based answers
- These fields are intended to support:
  - CPU-offload tables
  - memory-vs-context plots
  - retrieval-timeline qualitative figures

### Safe full-eval optimization
- Added a shared feature-cache workflow so visual features can be computed once and reused across methods.
- Kept the comparison paper-safe by ensuring:
  - same sampled timestamps
  - same one-frame-per-forward LM ingest
  - same answer-time semantics

## Important Fixes

### 1. DuoAttention ROCm correctness fix
- File: [attn_compat.py](/workspace/streaming-vqa/duo_attn/patch/attn_compat.py)
- Issue: cached incremental attention used the wrong causal alignment on MI300X.
- Fix: correct bottom-right causal mask for `q_len < k_len`.
- Result: streaming Duo behavior stopped degenerating.

### 2. Streaming causality fix
- File: [run_eval.py](/workspace/streaming-vqa/streaming/ReKV/run_eval.py)
- Issue: one extra sampled frame was being ingested at question time.
- Intermediate fix: upstream-style exclusive cutoff using `int(end_time * sample_fps)`.
- Final fix after boundary review: use the actual sampled timestamps and ingest exactly
  the frames with `timestamp < end_time`.
- Reason: `int(end_time * sample_fps)` can still under-ingest by one frame when the
  sampled schedule starts at `t=0` and `end_time` falls between sample-grid points.

### 3. ReKV + Duo integration fix
- File: [rekv_attention.py](/workspace/streaming-vqa/streaming/ReKV/rekv_core/attention/rekv_attention.py)
- Fixes:
  - manual KV-head expansion for ROCm GQA compatibility
  - Duo disabled during ReKV retrieval selection
  - Duo only applied after retrieval

### 4. Feature-cache safety fix
- Initial idea: batch frames through the vision tower during precompute.
- Problem: on this ROCm stack, batched vision forwarding changed the features enough to alter predictions.
- Final safe design:
  - extract features exactly as raw single-frame ingest would
  - stack and save those features on disk
  - reuse them across methods

This is slower than the unsafe shortcut, but it preserves comparability.

### 5. DuoPlusReKV sink token and initialization fix (2026-04-09)
- Files: [methods.py](/workspace/streaming-vqa/streaming/ReKV/methods.py)

**Problem A — Sink token mismatch**:
- DuoAttention's streaming heads were trained with `sink_size=512` attention sink tokens.
- In `duo_plus_rekv`, the ReKV context manager only preserved `n_init=13` tokens
  (the system prompt) as init/sink tokens.
- At answer time, streaming heads received only 13 sink tokens instead of the 512 they
  were trained to expect.
- This caused a significant mismatch between training and inference conditions for
  streaming heads, likely degrading their attention quality.

**Fix**: `n_init` for the ReKV context manager in `duo_plus_rekv` is now computed as
`init_prompt_len + ceil((sink_size - init_prompt_len) / block_size) * block_size`.
This rounds up to block boundaries to avoid breaking the context manager's block
alignment assertions.  For the current config (`sink_size=512`, `init_prompt_len=13`,
`block_size=196`): `n_init = 13 + 3*196 = 601`.  The first 601 tokens (system prompt
+ first 3 frames) are preserved as non-retrievable attention sinks.

**Problem B — Unnecessary tuple KV cache patching**:
- `enable_duo_attention_eval` patched both the model forward (for tuple KV caches)
  and the attention weights (for head reordering).
- ReKV's `patch_hf` then overwrote the model forward and attention forward.
- The tuple KV cache patching from DuoAttention was never used in `duo_plus_rekv`;
  only the weight reordering and `full_attention_heads` buffer registration mattered.
- The leftover `old_qwen2_decoder_layer_forward` patches were harmless but added
  unnecessary complexity.

**Fix**: `DuoPlusReKVStreamingMethod.__init__` now directly calls
`_enable_qwen2_layers_duo_attention_eval` (weight reordering + head registration only)
instead of the full `enable_duo_attention_eval` (which also patches model/layer forwards
for tuple KV caches).

**Traceability**: `duo_n_init` is now recorded in `method_stats` so the effective
sink token count is visible in result JSONs.

**Impact**: All previous `duo_plus_rekv` subsample results used `n_init=13`
(system prompt only) as sink tokens.  Fresh runs with this fix will use `n_init=601`.
The existing subsample results should be considered pre-fix baselines.  Accuracy may
improve because streaming heads now see the sink context they were trained with.

## Datasets and Official Protocol

### Datasets
- `RVS-Ego`
  - 10 videos
  - 1465 QA conversations
- `RVS-Movie`
  - 22 videos
  - 1905 QA conversations
- Total
  - 32 videos
  - 3370 QA conversations

### Official validated subsample slices
- `RVS-Ego subsample5`
- `RVS-Movie subsample5_movie`

### Official subsample protocol
- `5` videos
- `3` conversations per video
- `0.5 FPS`
- `seed = 42`

### Official comparison set
- Main set:
  - `full_streaming`
  - `duo_streaming (s=0.5)`
  - `rekv`
  - one selected `duo_plus_rekv` setting for the final run

## Main Results

Important provenance note:
- The retained checked-in outputs are now the standard `topk=64` runs only.
- The promoted hybrid setting is:
  - `AB_SPARSITY=0.75`
  - `deploy_sink_size=256`
  - `deploy_recent_size=512`
  - `AB_TOPK=64`
  - `AB_N_LOCAL=15000`
- For any fresh subsample rerun, use
  [run_streaming_subsample5_local.sh](/workspace/streaming-vqa/scripts/run_streaming_subsample5_local.sh),
  which now supports:
  - `rekv`
  - `duo_plus_rekv` at `s=0.5` and `s=0.75`
  - in-place judge, plot, and qualitative bundle generation

### RVS-Ego
| Slice | Method | Judge | ROUGE-L F1 | Token F1 | Latency (s) | Peak Mem (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `subsample5_topk64_memavg` | `full_streaming` | retained | retained | retained | retained | retained |
| `subsample5_topk64_memavg` | `duo_streaming (s=0.5)` | retained | retained | retained | retained | retained |
| `subsample5_topk64_memavg` | `rekv` | retained | retained | retained | retained | retained |
| `subsample5_topk64_memavg` | `duo_plus_rekv (s=0.75, sink=256, recent=512)` | retained | retained | retained | retained | retained |

### RVS-Movie
| Slice | Method | Judge | ROUGE-L F1 | Token F1 | Latency (s) | Peak Mem (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `subsample5_movie_topk64_memavg` | `full_streaming` | retained | retained | retained | retained | retained |
| `subsample5_movie_topk64_memavg` | `duo_streaming (s=0.5)` | retained | retained | retained | retained | retained |
| `subsample5_movie_topk64_memavg` | `rekv` | retained | retained | retained | retained | retained |
| `subsample5_movie_topk64_memavg` | `duo_plus_rekv (s=0.75, sink=256, recent=512)` | retained | retained | retained | retained | retained |

## What the Current Results Support
- `rekv` is the most consistently strong method across the retained standard subsample slices.
- `duo_streaming (s=0.5)` is a meaningful streaming baseline and often lowers memory relative to `full_streaming`, but the quality-latency tradeoff is dataset-dependent.
- `duo_plus_rekv (s=0.75, sink=256, recent=512)` is the retained hybrid setting, but it is not yet a universal improvement over plain `rekv`.

## Qualitative Findings
- `duo_streaming` helped most when `full_streaming` over-interpreted scenes and hallucinated stronger events than the visuals supported.
- `rekv` tended to beat `duo_plus_rekv` when the hybrid became too generic or hedgey.
- The local judge is useful, but still coarse enough that qualitative inspection matters.

Best qualitative artifact:
- [qualitative_examples.md](/workspace/streaming-vqa/outputs/evaluations_streaming/final_subsample_package/qualitative_examples.md)

## Plot Guide

### Existing core plots
- `aggregate_comparison.png`
  - high-level per-slice comparison of quality, TTFT, answer latency, and frame-ingest latency
- `peak_memory_comparison.png`
  - direct peak-memory comparison across methods
- `quality_latency_tradeoff.png`
  - raw quality vs latency operating points
- `quality_memory_tradeoff.png`
  - raw quality vs memory operating points
- `per_conversation_metrics.png`
  - how frames ingested, TTFT, and answer latency vary over conversations inside one slice
- `efficiency_vs_context.png`
  - how latency and memory change as more frames have been processed
- `quality_vs_context.png`
  - whether quality degrades or improves later in the stream
- `question_timeline.png`
  - sanity-check that frame ingest grows causally with question time
- `rekv_retrieval_diagnostics.png`
  - ReKV-only retrieval latency and retrieved-block behavior

### New decision-focused plots
- `delta_to_baseline.png`
  - shows whether Duo helps or hurts relative to its intended baseline
  - `duo_streaming` is measured against `full_streaming`
  - `duo_plus_rekv` is measured against `rekv`
  - includes quality delta, answer-latency delta, and peak-memory delta
- `pareto_tradeoffs_with_arrows.png`
  - shows quality vs latency and quality vs memory in one figure
  - dashed arrows show the directional move from:
    - `full_streaming -> duo_streaming`
    - `rekv -> duo_plus_rekv`
  - this is the clearest “does Duo move the operating point in a useful direction?” plot
- `delta_stability.png`
  - cross-subsample stability of Duo’s effect, not just raw method scores
  - tracks:
    - `duo_streaming - full_streaming`
    - `duo_plus_rekv - rekv`
  - includes quality, latency, and memory deltas across the validated slices

## Optimized Full Eval Workflow

### Why the feature cache exists
Without caching, every method recomputes:
- video decode
- vision preprocessing
- vision tower forward
- projector / pooling

The safe optimization is to do that visual work once, save the features, and reuse them across methods.

### Cache validation result
Validated on one real `RVS-Movie` clip (`tt0090756`) with one conversation:
- raw vs cached matched exactly for:
  - `full_streaming`
  - `duo_streaming (s=0.5)`
  - `rekv`
  - `duo_plus_rekv (s=0.375)`
- ReKV retrieval stats also matched exactly.

Short-clip wall times:
- `full_streaming`: `7.63s -> 6.96s`
- `duo_streaming`: `7.92s -> 7.44s`
- `rekv`: `8.41s -> 7.85s`
- `duo_plus_rekv`: `8.39s -> 7.86s`

Interpretation:
- the per-method speedup on one short clip is modest
- the bigger win should come from reusing one precompute pass across the whole 4-method full run

## How To Run

### Environment
```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate duo
```

### Quick environment sanity check
Run the lightweight smoke test first:
```bash
python -m streaming.ReKV.smoke_test
```

If that passes, run a tiny real-data subsample check:
```bash
MAX_VIDEOS=1 MAX_CONVERSATIONS=1 bash scripts/run_streaming_subsample5_local.sh full
MAX_VIDEOS=1 MAX_CONVERSATIONS=1 bash scripts/run_streaming_subsample5_local.sh duo
MAX_VIDEOS=1 MAX_CONVERSATIONS=1 bash scripts/run_streaming_subsample5_local.sh rekv
MAX_VIDEOS=1 MAX_CONVERSATIONS=1 bash scripts/run_streaming_subsample5_local.sh ab_s0375
```

What this verifies:
- dataset loading
- frame sampling
- streaming ingest
- DuoAttention path
- ReKV path
- A+B path
- JSON writing

If you want to test the cache workflow too, run:
```bash
DATASET=rvs_ego MAX_VIDEOS=1 bash scripts/run_streaming_full_eval_local.sh precompute
DATASET=rvs_ego MAX_VIDEOS=1 MAX_CONVERSATIONS=1 bash scripts/run_streaming_full_eval_local.sh all
```

### Subsample runs
```bash
bash scripts/run_streaming_subsample5_local.sh all
DATASET=rvs_movie SUBSAMPLE_NAME=subsample5_movie bash scripts/run_streaming_subsample5_local.sh all
```

Current promoted subsample settings:
- `duo_streaming`: `s=0.5`
- `rekv`: `retrieve_size=64`, `n_local=15000`
- `duo_plus_rekv` sweep: `s=0.375`, `s=0.5`, `s=0.75`
- `rekv_no_offload`: `n_local=15000`

### Optimized full eval
Precompute once:
```bash
DATASET=rvs_ego MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf bash scripts/run_streaming_full_eval_local.sh precompute
DATASET=rvs_movie MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf bash scripts/run_streaming_full_eval_local.sh precompute
```

Run official 4-method package at paper-style `topk=64`:
```bash
DATASET=rvs_ego MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf USE_FEATURE_CACHE=1 \
OUTPUT_ROOT=outputs/evaluations_streaming/rvs-ego/full_eval_topk64_memavg \
REKV_TOPK=64 AB_TOPK=64 REKV_N_LOCAL=15000 AB_N_LOCAL=15000 \
AB_SPARSITY=0.75 AB_DEPLOY_SINK_SIZE=256 AB_DEPLOY_RECENT_SIZE=512 \
bash scripts/run_streaming_full_eval_local.sh all

DATASET=rvs_movie MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf USE_FEATURE_CACHE=1 \
OUTPUT_ROOT=outputs/evaluations_streaming/rvs-movie/full_eval_topk64_memavg \
REKV_TOPK=64 AB_TOPK=64 REKV_N_LOCAL=15000 AB_N_LOCAL=15000 \
AB_SPARSITY=0.75 AB_DEPLOY_SINK_SIZE=256 AB_DEPLOY_RECENT_SIZE=512 \
bash scripts/run_streaming_full_eval_local.sh all
```

Current promoted full-eval settings:
- `duo_streaming`: `s=0.5`
- `rekv`: `retrieve_size` controlled by `REKV_TOPK`, `n_local=15000`
- `duo_plus_rekv`: `AB_SPARSITY=0.75`, `deploy_sink_size=256`, `deploy_recent_size=512`

Resume:
```bash
DATASET=rvs_ego MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf USE_FEATURE_CACHE=1 \
OUTPUT_ROOT=outputs/evaluations_streaming/rvs-ego/full_eval_topk64_memavg \
REKV_TOPK=64 AB_TOPK=64 REKV_N_LOCAL=15000 AB_N_LOCAL=15000 \
AB_SPARSITY=0.75 AB_DEPLOY_SINK_SIZE=256 AB_DEPLOY_RECENT_SIZE=512 RESUME=1 \
bash scripts/run_streaming_full_eval_local.sh all

DATASET=rvs_movie MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf USE_FEATURE_CACHE=1 \
OUTPUT_ROOT=outputs/evaluations_streaming/rvs-movie/full_eval_topk64_memavg \
REKV_TOPK=64 AB_TOPK=64 REKV_N_LOCAL=15000 AB_N_LOCAL=15000 \
AB_SPARSITY=0.75 AB_DEPLOY_SINK_SIZE=256 AB_DEPLOY_RECENT_SIZE=512 RESUME=1 \
bash scripts/run_streaming_full_eval_local.sh all
```

Droplet-stop / restart note:
- if a full run is interrupted, keep the same:
  - `DATASET`
  - `MODEL`
  - `OUTPUT_ROOT`
  - `REKV_TOPK`, `AB_TOPK`
  - `REKV_N_LOCAL`, `AB_N_LOCAL`
  - `AB_SPARSITY`
  - `AB_DEPLOY_SINK_SIZE`, `AB_DEPLOY_RECENT_SIZE`
- then rerun the same full-eval command with `RESUME=1`
- the launcher will:
  - move past already-complete method JSONs
  - resume partially completed method JSONs
  - start only missing methods from scratch
- if `USE_FEATURE_CACHE=1`, the shared feature cache must exist on the new machine:
  - either restore `outputs/evaluations_streaming/feature_cache/...`
  - or rerun `precompute` first
- cloning the repo and pushing only JSON/plot outputs is not enough for fast resume
  unless the cache is also restored or rebuilt

Example restart sequence on a new droplet:
```bash
DATASET=rvs_ego MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf bash scripts/run_streaming_full_eval_local.sh precompute
DATASET=rvs_movie MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf bash scripts/run_streaming_full_eval_local.sh precompute

DATASET=rvs_ego MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf USE_FEATURE_CACHE=1 \
OUTPUT_ROOT=outputs/evaluations_streaming/rvs-ego/full_eval_topk64_memavg \
REKV_TOPK=64 AB_TOPK=64 REKV_N_LOCAL=15000 AB_N_LOCAL=15000 \
AB_SPARSITY=0.75 AB_DEPLOY_SINK_SIZE=256 AB_DEPLOY_RECENT_SIZE=512 RESUME=1 \
bash scripts/run_streaming_full_eval_local.sh all

DATASET=rvs_movie MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf USE_FEATURE_CACHE=1 \
OUTPUT_ROOT=outputs/evaluations_streaming/rvs-movie/full_eval_topk64_memavg \
REKV_TOPK=64 AB_TOPK=64 REKV_N_LOCAL=15000 AB_N_LOCAL=15000 \
AB_SPARSITY=0.75 AB_DEPLOY_SINK_SIZE=256 AB_DEPLOY_RECENT_SIZE=512 RESUME=1 \
bash scripts/run_streaming_full_eval_local.sh all
```

Judge only:
```bash
DATASET=rvs_ego bash scripts/run_streaming_full_eval_local.sh judge
DATASET=rvs_movie bash scripts/run_streaming_full_eval_local.sh judge
```

Run a single method:
```bash
DATASET=rvs_ego bash scripts/run_streaming_full_eval_local.sh rekv
DATASET=rvs_movie bash scripts/run_streaming_full_eval_local.sh ab
```

Optional overrides:
```bash
DATASET=rvs_movie FEATURE_BATCH_SIZE=32 bash scripts/run_streaming_full_eval_local.sh precompute
DATASET=rvs_ego USE_FEATURE_CACHE=0 bash scripts/run_streaming_full_eval_local.sh full
DATASET=rvs_ego FLUSH_EVERY_VIDEOS=5 bash scripts/run_streaming_full_eval_local.sh all
```

### Regenerate plots after a run
Per-slice plots:
```bash
python -m streaming.ReKV.plot_results \
  outputs/evaluations_streaming/rvs-ego/subsample5_topk64_memavg/full_streaming/full_streaming.json \
  outputs/evaluations_streaming/rvs-ego/subsample5_topk64_memavg/duo_streaming/duo_streaming_s05.json \
  outputs/evaluations_streaming/rvs-ego/subsample5_topk64_memavg/rekv/rekv_topk64_nlocal15000.json \
  outputs/evaluations_streaming/rvs-ego/subsample5_topk64_memavg/duo_plus_rekv/duo_plus_rekv_s075_sink256_recent512_topk64_nlocal15000.json \
  --output-dir outputs/evaluations_streaming/rvs-ego/subsample5_topk64_memavg/main_plots
```

## Output Layout

### Main package to read or push
- [final_subsample_package](/workspace/streaming-vqa/outputs/evaluations_streaming/final_subsample_package)

Best entry points:
- [final_metrics.md](/workspace/streaming-vqa/outputs/evaluations_streaming/final_subsample_package/final_metrics.md)
- [paper_story.md](/workspace/streaming-vqa/outputs/evaluations_streaming/final_subsample_package/paper_story.md)
- [qualitative_examples.md](/workspace/streaming-vqa/outputs/evaluations_streaming/final_subsample_package/qualitative_examples.md)

### Official raw subsample outputs retained
- `outputs/evaluations_streaming/rvs-ego/subsample5_topk64_memavg/`
- `outputs/evaluations_streaming/rvs-movie/subsample5_movie_topk64_memavg/`

When reading old raw subsample outputs, always inspect `run_config.sparsity` before
using A+B numbers in writeups.

## Cleanup Done
- Removed exploratory smoke-only outputs.
- Removed tuning-only output directories.
- Removed temporary cache-validation raw outputs that were not part of the final story.
- Removed temporary `full_eval` dry-run output.
- Removed `topk=32` and offset-only result trees.
- Removed Python `__pycache__` clutter from `streaming/ReKV/`.

The output tree is now centered on:
- official `topk=64` subsample outputs
- official `topk=64` full-eval outputs
- final curated package

## What Remains Deferred
- full-dataset evaluation on both datasets
- any further A+B retuning
- stronger external judge if a better scoring path becomes available

## Practical Conclusion
If work resumes later:
1. Start from the optimized full-eval workflow.
2. Keep the official method set to:
   - `full_streaming`
   - `duo_streaming (s=0.5)`
   - `rekv`
   - one chosen `duo_plus_rekv` sparsity
3. Use subsamples to compare `duo_plus_rekv` at:
   - `s=0.375`
   - `s=0.5`
   - `s=0.75`
4. Use the new delta plots to answer the project question:
   - Does Duo help vs `full_streaming`?
   - Does Duo help when added on top of `rekv`?
   - Is that effect stable across subsamples?
