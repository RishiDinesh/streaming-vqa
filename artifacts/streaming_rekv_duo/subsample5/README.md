# Streaming ReKV vs DuoAttention: `subsample5`

This directory is the GitHub-safe artifact bundle for the first validated local MI300X streaming comparison run.

## Contents
- `results/`
  - `full_streaming.json`
  - `duo_streaming_s05.json`
  - `duo_streaming_s00.json`
  - `rekv_topk64_nlocal15000.json`
- `main_plots/`
  - paper-facing A-vs-B-vs-control plots
- `debug_plots/`
  - includes the `duo_streaming (s=0.0)` ablation and sparsity sweep curve

## Protocol
- Dataset: `RVS-Ego`
- Selection: first `5` videos in dataset order
- Conversations per video: first `3` sorted by `end_time`
- Total conversations: `15`
- Sample FPS: `0.5`
- Seed: `42`
- Model: `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`

## Methods
- `full_streaming`
- `duo_streaming_s05`
  - learned DuoAttention mask with `sparsity = 0.5`
- `duo_streaming_s00`
  - DuoAttention full-head control with `sparsity = 0.0`
- `rekv_topk64_nlocal15000`

## Headline Results
- `full_streaming`
  - avg ROUGE-L F1: `0.1843`
  - avg answer latency: `1.5985s`
  - peak memory: `5.16 GiB`
- `duo_streaming_s05`
  - avg ROUGE-L F1: `0.2049`
  - avg answer latency: `1.3657s`
  - peak memory: `3.63 GiB`
- `duo_streaming_s00`
  - avg ROUGE-L F1: `0.1902`
  - avg answer latency: `2.4286s`
  - peak memory: `5.14 GiB`
- `rekv_topk64_nlocal15000`
  - avg ROUGE-L F1: `0.1918`
  - avg answer latency: `0.9185s`
  - peak memory: `2.87 GiB`

## Notes
- All methods used the same sampled-frame ingest schedule.
- All methods processed the same `15` conversations.
- No method produced empty predictions on this batch.
