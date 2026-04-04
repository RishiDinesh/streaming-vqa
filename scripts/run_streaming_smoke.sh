#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_streaming_smoke.sh <annotation_path> <video_root> [max_videos]

Example:
  scripts/run_streaming_smoke.sh /path/to/ego4d_oe.json /path/to/rvs/videos 1

This submits two jobs, one for DuoAttention and one for ReKV, using:
  streaming/ReKV/run_eval.sh
EOF
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available in this environment." >&2
    echo "Use this script on the cluster login node, or submit streaming/ReKV/run_eval.sh manually." >&2
    exit 1
fi

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ANNOTATION_PATH=$1
VIDEO_ROOT=$2
MAX_VIDEOS=${3:-1}

DUO_JOB_NAME=${DUO_JOB_NAME:-stream-duo-smoke}
REKV_JOB_NAME=${REKV_JOB_NAME:-stream-rekv-smoke}
MODEL=${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}

cd "${ROOT}"

sbatch --job-name="${DUO_JOB_NAME}" streaming/ReKV/run_eval.sh \
    --annotation-path "${ANNOTATION_PATH}" \
    --video-root "${VIDEO_ROOT}" \
    --model "${MODEL}" \
    --method duo_streaming \
    --attn-dir "${ATTN_DIR}" \
    --max-videos "${MAX_VIDEOS}"

sbatch --job-name="${REKV_JOB_NAME}" streaming/ReKV/run_eval.sh \
    --annotation-path "${ANNOTATION_PATH}" \
    --video-root "${VIDEO_ROOT}" \
    --model "${MODEL}" \
    --method rekv \
    --max-videos "${MAX_VIDEOS}"
