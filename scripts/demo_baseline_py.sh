#!/usr/bin/env bash

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TERM=xterm-256color
export FORCE_COLOR=1
export CLICOLOR_FORCE=1
export PY_COLORS=1

GPU_ID=0
if [[ $# -gt 0 ]]; then
    GPU_ID="$1"
    shift
fi

if [[ "$GPU_ID" != "0" && "$GPU_ID" != "1" ]]; then
    echo "Usage: bash scripts/demo_baseline_py.sh [0|1] [extra live_llava_video.py args...]" >&2
    exit 1
fi

exec env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    /opt/venv/bin/python live_llava_video.py \
    --model_name llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
    --video_path data/sample.mp4 \
    --num_frames 32 \
    --max_length 22000 \
    --max_new_tokens 128 \
    --prompt "Describe this video in detail." \
    --attention_mode baseline \
    --device cuda:0 \
    "$@"
