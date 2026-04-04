# ReKV vs DuoAttention Log

## Summary
- Goal: implement a video-level streaming comparison for RVS-Ego on LLaVA-OneVision 0.5B.
- Target methods: `duo_streaming` and `rekv`.
- Hardware focus: local AMD MI300X / ROCm validation.

## Implemented So Far
- Added a dedicated streaming package under `streaming/ReKV`.
- Added normalized streaming types for videos and conversations.
- Added an RVS-Ego dataset adapter that reads `ego/ego4d_oe.json`.
- Made the RVS-Ego adapter practical for local testing:
  - filters videos before resolving paths
  - downloads only the requested HF video file instead of the whole subset
  - uses lazy frame reads instead of loading all sampled frames into memory
- Added a causal streaming runner that:
  - samples frames at `0.5 FPS`
  - reuses state across conversations in the same video
  - avoids offline full-video prefill
  - writes JSON results under `outputs/evaluations_streaming/...`
- Added local smoke-friendly runner controls:
  - `--video-id`
  - `--video-index`
  - `--max-conversations-per-video`
- Added `duo_streaming` method support using the repo’s learned 0.5B DuoAttention masks.
- Added `rekv` method support using a local vendored ReKV core.
- Patched vendored ReKV pieces to reduce CUDA-only assumptions and keep ROCm execution viable.
- Patched vendored ReKV to support the current `transformers` Qwen2 rotary embedding API on this machine.
- Added a smoke harness at `streaming/ReKV/smoke_test.py`.
- Added a SLURM-style runner wrapper at `streaming/ReKV/run_eval.sh`.
- Added a local plotting script at `streaming/ReKV/plot_results.py`.
- Added a helper launcher at `scripts/run_streaming_smoke.sh`.

## Important Design Decisions
- Streaming semantics are video-level, not per-question replay from start.
- ReKV and DuoAttention use the same sampled-frame ingest schedule.
- Ingest granularity is one sampled frame per forward pass.
- No LongBench-style tail replay in the main streaming path.
- Current scoring stored in results is `normalized_exact_match`.

## Validation Done
- `python -m compileall streaming/ReKV`
- `python -m streaming.ReKV.run_eval --help`
- `python -m streaming.ReKV.smoke_test`
- Confirmed the local `duo` environment sees one AMD GPU and bf16 support.
- Real local MI300X Duo smoke run completed on one RVS-Ego video / one conversation.
- Real local MI300X ReKV smoke run completed on the same video / conversation.
- Plot generation completed from the two real result JSON files.

## Current Results
- Smoke test result: `streaming/ReKV smoke test passed`
- Real Duo local result:
  - video: `879dd163-7588-45d1-9466-a5deabc59167`
  - conversations: `1`
  - frames ingested: `151`
  - avg frame ingest latency: `0.0787s`
  - avg TTFT: `0.0168s`
  - avg answer latency: `0.0169s`
  - peak memory: `2.13 GiB`
  - prediction was empty / immediate stop token
- Real ReKV local result:
  - same video and conversation slice as Duo
  - frames ingested: `151`
  - avg frame ingest latency: `0.0673s`
  - avg TTFT: `0.0390s`
  - avg answer latency: `0.5323s`
  - peak memory: `2.66 GiB`
  - retrieval latency: `0.2247s`
  - avg retrieved block count: `64`
  - prediction was a partial but semantically relevant description
- Plot generation completed for:
  - aggregate comparison
  - peak memory comparison
  - per-conversation metrics
  - question timeline
  - ReKV retrieval diagnostics

## Pending Next Steps
- Inspect correctness signals:
  - causality
  - frame ingest parity
  - latency and memory fields
  - ReKV retrieval diagnostics
- Improve answer-quality evaluation beyond `normalized_exact_match` for open-ended streaming QA.
- Run more than one conversation and more than one video once single-GPU runtime is acceptable.
- Add additional paper-facing comparison plots once we have multi-video results.
- If needed, tune local smoke arguments for single-GPU practicality.

## Result Paths
- Duo JSON: `outputs/evaluations_streaming/rvs-ego/duo_streaming/local_duo_smoke_results.json`
- ReKV JSON: `outputs/evaluations_streaming/rvs-ego/rekv/local_rekv_smoke_results.json`
- Plot directory: `outputs/evaluations_streaming/rvs-ego/local_smoke_plots`
