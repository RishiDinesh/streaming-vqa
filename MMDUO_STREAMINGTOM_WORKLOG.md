# MMDuo + StreamingTOM Integration Worklog

Date: 2026-04-13 (UTC)
Repo: `/root/streaming-vqa`

## Goal
Build and validate:
1. StreamingTOM single-video inference on RVS data using ReKV sampling utilities.
2. MMDuo dependency readiness and checkpoint compatibility.
3. A **true hybrid** `mmduo_plus_streamingtom` single-path inference (not just two separate runs).

---

## High-Level Outcomes
- StreamingTOM single-video pipeline works on GPU.
- MMDuo checkpoint + patching works in current env after fixing missing deps.
- Implemented true hybrid `mmduo_plus_streamingtom` attention path and validated on a real RVS-Ego video.
- Added single-command shell triggers for both StreamingTOM and hybrid inference.

---

## Environment + Dependency Changes (Conda env: `duo`)

### Installed/changed
- CUDA stack:
  - `torch==2.5.1+cu124`
  - `torchvision==0.20.1+cu124`
  - `torchaudio==2.5.1+cu124`
- Pinned:
  - `numpy==1.26.4`
- Added runtime deps:
  - `einops==0.8.2`
  - `av==17.0.0`
  - `open_clip_torch==3.3.0`
  - `timm==1.0.26`
  - `ftfy==6.3.1`
  - `wcwidth==0.6.0`
  - `imageio==2.37.3`
  - `imageio-ffmpeg==0.6.0`
- MMDuo dependency fix:
  - `tensor_parallel==2.0.0`
  - `setuptools<81` (installed `80.10.2`) for `pkg_resources` required by `tensor_parallel`

### Notes
- `decord` was installed and then removed due to runtime decoder instability in this environment.
- Some upstream warnings remain non-fatal (e.g., `qwen2-revise` config warnings, `pyav` warning line from upstream text).

---

## Files Added

### 1) StreamingTOM single trigger
- `/root/streaming-vqa/streaming/StreamingTom/scripts/run_streamingtom_single_trigger.sh`
- Purpose: one-command trigger for single-video StreamingTOM run with pre-set env vars.

### 2) Sequential comparison script (MMDuo then StreamingTOM)
- `/root/streaming-vqa/streaming/StreamingTom/scripts/run_mmduo_streamingtom_single.py`
- Purpose: run both methods on same sample sequentially and print both outputs.
- Important: this was an intermediate step and **not** the final hybrid method.

### 3) Final hybrid Python script (single-path)
- `/root/streaming-vqa/streaming/StreamingTom/scripts/run_mmduo_plus_streamingtom_single.py`
- Purpose: run **true** `mmduo_plus_streamingtom` in one model forward path by combining Duo routing with StreamingTOM patched attention.

### 4) Final hybrid shell trigger
- `/root/streaming-vqa/streaming/StreamingTom/scripts/run_mmduo_plus_streamingtom_single.sh`
- Purpose: one-command shell wrapper for final hybrid inference.

### 5) Environment spec
- `/root/streaming-vqa/environment.yml`
- Purpose: reproducible GPU environment file with key dependencies.

---

## Files Modified

### StreamingTOM runners + integration
1. `/root/streaming-vqa/streaming/StreamingTom/scripts/run_streamingtom_rvs_single.py`
   - Added HF dataset path support (`--hf-repo-id`, `--allow-hf-video-download`).
   - Added robust import path setup for `streamingtom-core` + `LLaVA-NeXT`.
   - Adjusted CPU dtype logic and model loading options.

2. `/root/streaming-vqa/streaming/StreamingTom/scripts/run_streamingtom_rvs_single.sh`
   - Added flexible args (local files or HF-only mode).
   - Added split model settings:
     - `PRECOMPUTE_MODEL` (ReKV feature cache model, HF format)
     - `RUN_MODEL` (StreamingTOM model, lmms-lab format)
   - Default model IDs updated to avoid prior mismatches.

3. `/root/streaming-vqa/streaming/StreamingTom/LLaVA-NeXT/llava/model/builder.py`
   - Added CPU-safe vision tower placement fallback.
   - Aligned vision tower dtype with model dtype when CUDA unavailable.

4. `/root/streaming-vqa/streaming/StreamingTom/streamingtom-core/streamingtom/modules/streamingtom_context.py`
   - Removed hardcoded `28` layer assumption; use dynamic model layer count.

5. `/root/streaming-vqa/streaming/StreamingTom/streamingtom-core/streamingtom/tasks/vision.py`
   - Removed hardcoded `28` layer checks; switched to dynamic consistency checks.

6. `/root/streaming-vqa/streaming/StreamingTom/streamingtom-core/streamingtom/tasks/query.py`
   - Removed hardcoded `28` layer assertion; dynamic checks.

### ReKV preprocessing warning fix
7. `/root/streaming-vqa/streaming/ReKV/methods.py`
   - Set `use_fast=False` in `AutoProcessor.from_pretrained(...)` to silence specific warning.

### Final hybrid attention implementation
8. `/root/streaming-vqa/streaming/StreamingTom/streamingtom-core/streamingtom/models/llava/modeling_qwen2_revise.py`
   - Added Duo-routing helpers into StreamingTOM custom attention:
     - `_supports_enable_gqa`
     - `_sdpa_attention`
     - `_duo_query_indices_from_kv`
     - `_build_duo_streaming_bool_mask`
     - `duo_mixed_attention_forward`
   - Updated `Qwen2Attention.forward` to route through `duo_mixed_attention_forward` when `duo_enable` is set, otherwise keep original path.
   - Cleaned duplicate helper block introduced during editing.

---

## MMDuo Checkpoint Verified
Provided checkpoint directory:
- `/root/streaming-vqa/outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1`

Contains:
- `config.json`
- `full_attention_heads.tsv`
- `full_attention_heads_latest.tsv`
- optimizer state files

Parsed values (from config):
- `model_name`: `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- `sink_size`: `512`
- `recent_size`: `1024`
- `max_length`: `32000`
- `num_frames`: `64`

---

## Validation & Test Runs

### A) StreamingTOM single-video success
Command family used:
- `run_streamingtom_rvs_single.sh`
- HF RVS sample (`video_index=0`)

Observed success output included:
- sample metadata printed
- answer produced:
  - "The task being performed with vegetables is chopping and grating."

### B) MMDuo readiness tests
- Initial failure: `tensor_parallel` import missing.
- After install + setuptools fix, smoke test passed:
  - Loaded mask `(24,2)`
  - Enabled Duo eval patch on Llava-OneVision model
  - `smoke_test: PASS`

### C) Sequential dual-method script test
`run_mmduo_streamingtom_single.py` ran both methods sequentially on same sample and produced two answers.

### D) Final true hybrid test (`mmduo_plus_streamingtom`)
Command:
- `run_mmduo_plus_streamingtom_single.py` on GPU with sample 0.

Observed success output:
- Header: `=== mmduo_plus_streamingtom Single-Video Run ===`
- Duo metadata printed:
  - `duo_actual_sparsity: 0.5`
  - `duo_sink_size: 512`
  - `duo_recent_size: 1024`
- Answer produced:
  - "The task being performed with vegetables is chopping and preparing them for cooking."

This confirms **single-path hybrid inference** is operational.

---

## Final Trigger Commands

### 1) StreamingTOM single-video trigger
```bash
bash /root/streaming-vqa/streaming/StreamingTom/scripts/run_streamingtom_single_trigger.sh
```

### 2) Final hybrid trigger (mmduo_plus_streamingtom)
```bash
bash /root/streaming-vqa/streaming/StreamingTom/scripts/run_mmduo_plus_streamingtom_single.sh
```

---

## Important Clarification
- `run_mmduo_streamingtom_single.py` = sequential comparison script (MMDuo run + StreamingTOM run).
- `run_mmduo_plus_streamingtom_single.py` = **true integrated hybrid** path in one patched model attention flow.

---

## Remaining Warnings (Non-blocking in tested run)
- Upstream informational warnings from LLaVA/transformers (`qwen2-revise` config notices).
- `Please install pyav to use video processing functions.` line appears in upstream logs, but run still completes successfully in this setup.
