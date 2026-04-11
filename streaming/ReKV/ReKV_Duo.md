# ReKV + DuoAttention Streaming Guide

Working reference for the `streaming/ReKV` module on the `exp/nv-gpu-inference` branch.

**Target:** NVIDIA SLURM cluster (Toronto CS) · CUDA · conda env at `<repo-root>/envs/duo`

---

## 1. Four Methods at a Glance

| Method | What it does | Cache semantics label |
|---|---|---|
| `full_streaming` | Plain causal KV cache; no compression | `plain_full_streaming_cache` |
| `duo_streaming` | Duo head routing: full KV for retrieval heads, sink+recent window for streaming heads | `duo_tuple_kv_compressed_streaming` |
| `rekv` | Local window on GPU; old blocks offloaded to CPU; top-k blocks retrieved at question time | `rekv_init_retrieved_old_forced_local` |
| `duo_plus_rekv` | ReKV retrieval assembles answer context; Duo head routing applied over that context | `hybrid_rekv_context_plus_duo_head_routing` |

All four methods share the same frame schedule, the same init prompt, and the same greedy decode path. Differences are only in how the KV cache is assembled at answer time.

---

## 2. Streaming Contract

These rules apply identically to all four methods. Any result that violates them is not comparable.

- Frames are sampled at `sample_fps` from native video FPS
- Only sampled frames with `timestamp < question.end_time` are visible (enforced by `bisect_left`)
- One sampled frame is ingested per forward pass
- Method state (KV cache) is shared across all questions from the same video
- No offline whole-video prefill
- If a feature cache is used, its sampling schedule must exactly match the run schedule

The contract is recorded verbatim in the `evaluation_manifest` produced by `run_eval.py`.

---

## 3. End-to-End Architecture

```
Dataset (RVS-Ego or RVS-Movie)
  → datasets.py: sort conversations by end_time, sample frames at sample_fps
  → [optional] precompute_features.py → feature_cache.py (validated schedule match)
  → methods.py: reset per video, ingest frames causally
      ↳ full_streaming:    plain KV cache
      ↳ duo_streaming:     enable_duo_attention_eval, blocksparse streaming heads
      ↳ rekv:              patch_hf → rekv_attention_forward → ContextManager
      ↳ duo_plus_rekv:     patch_hf + Duo head weights → rekv_duo_enabled=True
  → answer_question → greedy_decode_with_cache
  → run_eval.py: write result JSON with evaluation_manifest
  → [optional] judge_results.py, plot_results.py, profile_streaming.py
```

---

## 4. Module Map

| File | Purpose |
|---|---|
| `run_eval.py` | Main evaluation loop: causal ingest, question answering, checkpoint/resume, sharding |
| `methods.py` | All four method classes + shared utilities (feature extraction, greedy decode) |
| `datasets.py` | RVS dataset loading (`rvs_ego`, `rvs_movie`), frame sampling, video reader |
| `feature_cache.py` | Pre-computed feature cache: validated load, schedule-equivalence check |
| `precompute_features.py` | Batch-extract and save visual features to cache |
| `common.py` | Data structures (`StreamingVideoSample`, `StreamingConversation`) and scoring |
| `validate_runtime_env.py` | Report actual backend stack (flash-attn, blocksparse, flashinfer) |
| `judge_results.py` | LLM-based semantic scoring (0–5) for open-ended answers |
| `rescore_results.py` | Recompute token/ROUGE metrics without re-running eval |
| `plot_results.py` | Paper-style quality vs. latency/memory plots |
| `plot_profile.py` | Latency/memory curves from profiling runs |
| `compare_subsamples.py` | Cross-slice stability analysis across multiple result JSONs |
| `build_backend_audit_report.py` | Markdown comparison table across methods and backends |
| `build_qualitative_bundle.py` | JSON+markdown qualitative examples across methods |
| `profile_streaming.py` | Single-video profiling: latency/memory curves at fixed frame counts |
| `smoke_test.py` | Fast unit tests (no GPU required) for dataset loading, causal ingest, resume, plotting |
| `rekv_core/patch.py` | Patches HF model (Qwen2/Llama/Mistral) with ReKV attention forward |
| `rekv_core/attention/rekv_attention.py` | ReKV attention forward: local+init+retrieved layout, Duo gate |
| `rekv_core/attention/kv_cache_manager.py` | CPU offload engine: `ContextManager`, `MemoryUnit`, `VectorTensor` |
| `rekv_core/attention/rope.py` | `RotaryEmbeddingESM` with distance scaling |
| `rekv_core/attention/utils.py` | `repeat_kv` for GQA head expansion |
| `rekv_core/attention/dot_production_attention/` | Multi-stage dot-product attention: Torch (default) or Triton (`--rekv-fattn`) |

---

## 5. Component Notes

### 5.1 Datasets — `datasets.py`

- `rvs_ego`: annotation `ego/ego4d_oe.json`, HF repo `Becomebright/RVS`, subset `ego`
- `rvs_movie`: annotation `movie/movienet_oe.json`, subset `movie`
- Conversations sorted by `end_time` (ascending) at load time
- Videos resolved from `--video-root`, then annotation-relative path, then HF download if `--allow-hf-video-download`
- Frame decoding via `decord` (primary), falling back to `imageio`

### 5.2 Causal Frame Cutoff

`conversation_target_frame_count(end_time, sampled_timestamps_sec)` returns `bisect_left(timestamps, end_time)` — strictly fewer frames than `end_time`. This is the fairness anchor: every method sees exactly the same frames before each question.

### 5.3 Feature Cache — `feature_cache.py`

- Stores `[num_frames, n_frame_tokens, hidden_dim]` tensors in `bfloat16` on CPU
- `validate_feature_cache_payload` checks: `sample_id`, `video_id`, `sample_fps`, frame indices, timestamps (tolerance 1e-6s), tensor shape
- Version tag `FEATURE_CACHE_VERSION = "v1"` guards against stale caches
- Using a shared cache guarantees bit-identical features across methods

### 5.4 `full_streaming`

Plain causal KV cache accumulation. Model loads with `attn_implementation="eager"`. No retrieval, no Duo routing.

### 5.5 `duo_streaming`

- Loads learned head-routing weights from `--attn-dir`
- `enable_duo_attention_eval` sets `full_attention_heads` per layer; sparse streaming heads use the `blocksparse` kernel
- Use `--duo-strict-no-sdpa-fallback` to fail fast if `block_sparse_attn` is missing
- `--deploy-sink-size` and `--deploy-recent-size` control the streaming head window

### 5.6 `rekv` — primary paper-aligned method

**Patching** (`rekv_core/patch.py`): `patch_hf` replaces each attention layer's forward with `rekv_attention_forward` and installs `RotaryEmbeddingESM` on `model_core.position_bias`. Supports Qwen2 (LLaVA-OneVision backbone).

**Ingest phase** — `ContextManager.append()`:
- Local KV kept on GPU (size `n_local` tokens)
- When `n_init` tokens are accumulated, init KV is frozen; subsequent blocks offloaded to CPU as `MemoryUnit` objects
- Representative key vector per block stored in `VectorTensor` (GPU, fp32 cosine similarity)
- CUDA streams: main compute on current stream, offload/load on `GLOBAL_STREAM`

**Retrieval phase** — triggered at question time by `set_retrieval()`:
- `_calc_block_topk`: selects top-k blocks by cosine similarity to the question query
- Assembled context: `[init tokens] + [retrieved old blocks] + [local window]`

**Key parameters:**
- `--n-local`: local window size in tokens (default 15000)
- `--retrieve-size`: number of blocks to retrieve (default 64)
- `--n-frame-tokens`: tokens per video frame (default 196, must match model)
- `block_size = n_frame_tokens` (one block = one video frame)

### 5.7 `duo_plus_rekv` — approximate hybrid

- ReKV's `patch_hf` overwrites the attention forward (same as `rekv`)
- Duo head weights loaded and registered: `full_attention_heads`, `sink_size`, `recent_size` on each layer
- `rekv_duo_enabled = True` flag on each attention layer activates the hybrid branch
- During retrieval: Duo suppressed, ReKV does pure retrieval
- During answering: Duo head routing applied over the assembled ReKV context
- `duo_n_init`: extended to cover Duo's sink tokens, block-aligned to `n_frame_tokens`

**This is approximate**: streaming heads see the ReKV local window, which may differ from standalone Duo's `recent_size` window.

### 5.8 Attention Kernel — `dot_production_attention/`

- Default: `TorchMultiStageDotProductionAttention` (pure PyTorch)
- With `--rekv-fattn`: tries `TritonMultiStageDotProductionAttention`; `BLOCK_DMODEL` must be in {16, 32, 64, 128}
- GQA: KV heads expanded via `repeat_kv` (`repeat_interleave`)

---

## 6. Environment Setup (Toronto CS Cluster)

### 6.1 Conda location

Conda lives at `/u/navdeep/miniconda3` (referenced in `~/.bashrc`). Activate it in any shell with:

```bash
source /u/navdeep/miniconda3/etc/profile.d/conda.sh
```

The project env is installed as a **prefix** inside the repo (not a named env), so it stays in scratch space and doesn't touch home-dir quota:

```
<repo-root>/envs/duo/
```

Activate it:

```bash
conda activate /w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/envs/duo
```

`scripts/streaming_env.sh` does this automatically for SLURM jobs — it tries the project prefix first, then falls back to a named `duo` env.

### 6.2 First-time setup

**Step 1 — Install Miniconda** (login node, once only):

```bash
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /u/navdeep/miniconda3
source /u/navdeep/miniconda3/etc/profile.d/conda.sh
conda --version
```

**Step 2 — Build the env** (must run on a GPU compute node — `block_sparse_attn` compiles against CUDA):

```bash
# Request an interactive GPU session
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
     --time=01:00:00 --pty bash -l

# Inside the compute node:
source /u/navdeep/miniconda3/etc/profile.d/conda.sh
cd /w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;89;90" bash setup.sh 2>&1 | tee logs/setup_$(date +%Y%m%d_%H%M%S).log
```

`BLOCK_SPARSE_ATTN_CUDA_ARCHS`: A100=`80`, H100=`90`, L40/Ada=`89`. The default `"80;89;90"` covers all three. Check with `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`.

**Step 3 — Validate** (still on the compute node):

```bash
conda activate /w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/envs/duo
python -m streaming.ReKV.validate_runtime_env
```

Expected:
- `"streaming_attn_backend_actual": "blocksparse"` — Duo is paper-faithful ✓
- `"flash_attn_available": true` ✓
- `"cuda_available": true` ✓

If `streaming_attn_backend_actual` is `"sdpa"`, the `block_sparse_attn` compile failed — check `logs/setup_*.log`.

### 6.3 Disk layout

| Location | Purpose | Notes |
|---|---|---|
| `/u/navdeep/miniconda3/` | Conda base install | ~500 MB, on home fs (1 TB, 32% used) |
| `<repo>/envs/duo/` | Project Python env | ~10 GB, in scratch space |
| `<repo>/.hf_cache/` | HF model + dataset cache | Videos auto-downloaded here |
| `<repo>/outputs/` | All eval results, plots, logs | In scratch space |
| `<repo>/logs/` | SLURM job stdout logs | Named `stream-<method>-<jobid>.out` |

---

## 7. Backend Requirements

| Library | Role | Required? |
|---|---|---|
| `flash-attn` | Full-attention forward for non-ReKV paths | Strongly recommended |
| `block_sparse_attn` | Sparse streaming attention for Duo methods | Required for paper-faithful Duo |
| `flashinfer` | RMSNorm/RoPE acceleration | Optional (torch fallback) |
| `triton` | Triton kernel for ReKV inner attention | Optional (`--rekv-fattn`) |
| `decord` | Fast video frame decoding | Recommended (imageio fallback) |

---

## 8. Running Evaluations

Videos and annotations are auto-downloaded from HuggingFace (`Becomebright/RVS`) into `<repo>/.hf_cache/` on first run. No manual data prep needed.

### 8.1 Smoke test — 1 video, all 4 methods

Useful for verifying the environment end-to-end before committing to a larger run.

```bash
cd /w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
bash scripts/run_streaming_subset3_slurm.sh --max-videos 1
```

Jobs take ~5–15 min each for 1 video. Monitor:

```bash
squeue -u ${USER}
watch -n 30 'tail -n 5 logs/stream-*-sub1-*.out'
```

### 8.2 Subset — N videos, all 4 methods

```bash
# 5-video subset
bash scripts/run_streaming_subset3_slurm.sh --max-videos 5
```

### 8.3 Full eval — single GPU

```bash
sbatch --output="$(pwd)/logs/%x-%j.out" streaming/ReKV/run_eval.sh \
  --dataset rvs_ego --method rekv --retrieve-size 64 --n-local 15000
```

### 8.4 Full eval — multi-GPU sharded

```bash
NUM_CHUNKS=8 DATASET=rvs_ego METHOD=rekv \
  sbatch --array=0-7 \
    --output="$(pwd)/logs/%x-%A_%a.out" \
    scripts/run_streaming_eval_slurm_array.sh
```

---

## 9. After Jobs Finish

The submit script prints the exact commands. General pattern:

```bash
source /u/navdeep/miniconda3/etc/profile.d/conda.sh
conda activate /w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/envs/duo
cd /w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa

TS=<timestamp>          # from the submit script output
BASE=outputs/evaluations_streaming/rvs-ego/subset1   # or subset5, etc.

# Compare all four methods
python -m streaming.ReKV.compare_subsamples \
    ${BASE}/full_streaming/${TS}_results.json \
    ${BASE}/duo_streaming/${TS}_results.json \
    ${BASE}/rekv/${TS}_results.json \
    ${BASE}/duo_plus_rekv/${TS}_results.json \
    --output-dir ${BASE}/comparison/

# Plots (PNG files saved to plots/)
python -m streaming.ReKV.plot_results \
    ${BASE}/full_streaming/${TS}_results.json \
    ${BASE}/duo_streaming/${TS}_results.json \
    ${BASE}/rekv/${TS}_results.json \
    ${BASE}/duo_plus_rekv/${TS}_results.json \
    --output-dir ${BASE}/plots/
```

Outputs land in:
```
outputs/evaluations_streaming/rvs-ego/subset1/
├── full_streaming/<ts>_results.json
├── duo_streaming/<ts>_results.json
├── rekv/<ts>_results.json
├── duo_plus_rekv/<ts>_results.json
├── comparison/    ← markdown table, CSV, stability plots
└── plots/         ← aggregate_comparison.png, memory_comparison.png, ...
```

---

## 10. Cross-Method Comparability Checklist

Before comparing results across methods, verify in the manifests:

- `shared_run_settings.sample_fps` — identical
- `shared_run_settings.model` — identical
- `shared_run_settings.dataset` — identical
- `shared_run_settings.max_new_tokens` — identical
- `streaming_protocol.causal_cutoff_policy` = `"sampled_timestamps_strictly_before_end_time"`
- `streaming_protocol.question_ordering` = `"dataset_loader_sorted_by_end_time"`
- `shared_run_settings.ingest_source` — both `raw_frames` or both `cached_features` from the same cache
- For Duo methods: `method_manifest.backend_resolution.streaming_attn_backend_actual` = `"blocksparse"`

---

## 11. Results Interpretation

- `rekv` is the strongest default: memory-efficient, paper-aligned, stable
- `duo_streaming` is paper-aligned only when `streaming_attn_backend_actual = blocksparse`
- `duo_plus_rekv` is approximate: useful for ablation, not a literal paper reproduction
- `full_streaming` is the memory-unlimited upper bound

Result category labels in manifests:
- `nvidia_sparse_duo_equivalent` — `block_sparse_attn` active, paper-faithful Duo
- `sdpa_fallback_duo` — `block_sparse_attn` missing (should not occur on correctly set-up NVIDIA)

---

## 12. Official References

- ReKV paper: https://arxiv.org/abs/2503.00540 · repo: https://github.com/Becomebright/ReKV
- DuoAttention paper: https://arxiv.org/abs/2410.10819 · repo: https://github.com/mit-han-lab/duo-attention
