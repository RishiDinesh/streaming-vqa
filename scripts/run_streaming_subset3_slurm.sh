#!/usr/bin/env bash
# Submit a 3-video subset evaluation for all four streaming methods on SLURM.
# One SLURM job per method (4 jobs total), each on a single GPU.
# Results land in outputs/evaluations_streaming/<dataset>/subset3/<method>/.
#
# Usage (submit from repo root):
#   bash scripts/run_streaming_subset3_slurm.sh \
#       --annotation-path /path/to/ego4d_oe.json \
#       --video-root /path/to/rvs/videos \
#       [--dataset rvs_ego|rvs_movie] \
#       [--attn-dir /path/to/duo/weights] \
#       [--model llava-hf/llava-onevision-qwen2-0.5b-ov-hf]
#
# Outputs (one JSON per method):
#   outputs/evaluations_streaming/<dataset>/subset3/full_streaming/<timestamp>.json
#   outputs/evaluations_streaming/<dataset>/subset3/duo_streaming/<timestamp>.json
#   outputs/evaluations_streaming/<dataset>/subset3/rekv/<timestamp>.json
#   outputs/evaluations_streaming/<dataset>/subset3/duo_plus_rekv/<timestamp>.json
#
# After all jobs finish:
#   python -m streaming.ReKV.compare_subsamples outputs/evaluations_streaming/<dataset>/subset3/
#
# Environment variables (optional overrides):
#   SBATCH_EXTRA_ARGS   extra #SBATCH flags (e.g. "--partition=gpu --account=myaccount")
#   MAX_NEW_TOKENS      default 256
#   SAMPLE_FPS          default 0.5
#   VIDEO_DECODE_THREADS  default 4
#   REKV_TOPK           default 64
#   REKV_N_LOCAL        default 15000
#   SPARSITY            default 0.5

set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ANNOTATION_PATH=""
VIDEO_ROOT=""
DATASET="${DATASET:-rvs_ego}"
MODEL="${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"
ATTN_DIR="${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}"
MAX_VIDEOS=3
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SAMPLE_FPS="${SAMPLE_FPS:-0.5}"
VIDEO_DECODE_THREADS="${VIDEO_DECODE_THREADS:-4}"
REKV_TOPK="${REKV_TOPK:-64}"
REKV_N_LOCAL="${REKV_N_LOCAL:-15000}"
SPARSITY="${SPARSITY:-0.5}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_streaming_subset3_slurm.sh \
      --annotation-path <path> \
      --video-root <path> \
      [--dataset rvs_ego|rvs_movie] \
      [--attn-dir <path>] \
      [--model <hf-model-id>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --annotation-path) ANNOTATION_PATH="$2"; shift 2 ;;
    --video-root)      VIDEO_ROOT="$2";      shift 2 ;;
    --dataset)         DATASET="$2";         shift 2 ;;
    --attn-dir)        ATTN_DIR="$2";        shift 2 ;;
    --model)           MODEL="$2";           shift 2 ;;
    --max-videos)      MAX_VIDEOS="$2";      shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${ANNOTATION_PATH}" || -z "${VIDEO_ROOT}" ]]; then
  echo "[error] --annotation-path and --video-root are required" >&2
  usage >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[error] sbatch is not available. Run this on the cluster login node." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
OUTPUT_BASE="${ROOT}/outputs/evaluations_streaming/${DATASET//_/-}/subset${MAX_VIDEOS}"
SUBSAMPLE_NAME="subset${MAX_VIDEOS}_${TIMESTAMP}"

# Extra sbatch flags from environment (e.g. partition, account)
declare -a SBATCH_EXTRA=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  read -r -a SBATCH_EXTRA <<< "${SBATCH_EXTRA_ARGS}"
fi

# ---------------------------------------------------------------------------
# Helper: submit one method job
# ---------------------------------------------------------------------------
submit_method() {
  local method="$1"
  local output_path="${OUTPUT_BASE}/${method}/${TIMESTAMP}_results.json"
  mkdir -p "$(dirname "${output_path}")"

  local method_args=()
  case "${method}" in
    duo_streaming)
      method_args+=(--attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}" --duo-strict-no-sdpa-fallback)
      ;;
    rekv)
      method_args+=(--retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}")
      ;;
    duo_plus_rekv)
      method_args+=(--attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}" --duo-strict-no-sdpa-fallback \
                    --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}")
      ;;
  esac

  local job_id
  job_id=$(sbatch \
    --job-name="stream-${method}-sub${MAX_VIDEOS}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=02:00:00 \
    --output="${LOG_DIR}/stream-${method}-sub${MAX_VIDEOS}-%j.out" \
    "${SBATCH_EXTRA[@]}" \
    streaming/ReKV/run_eval.sh \
      --dataset "${DATASET}" \
      --annotation-path "${ANNOTATION_PATH}" \
      --video-root "${VIDEO_ROOT}" \
      --model "${MODEL}" \
      --method "${method}" \
      --sample-fps "${SAMPLE_FPS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --video-decode-threads "${VIDEO_DECODE_THREADS}" \
      --max-videos "${MAX_VIDEOS}" \
      --subsample-name "${SUBSAMPLE_NAME}" \
      --output-path "${output_path}" \
      "${method_args[@]}" \
    | awk '{print $NF}')

  echo "  ${method}: job ${job_id} -> ${output_path}"
  echo "${job_id}"
}

# ---------------------------------------------------------------------------
# Submit all 4 methods
# ---------------------------------------------------------------------------
echo "==> Submitting ${MAX_VIDEOS}-video subset eval"
echo "    dataset:         ${DATASET}"
echo "    annotation_path: ${ANNOTATION_PATH}"
echo "    video_root:      ${VIDEO_ROOT}"
echo "    model:           ${MODEL}"
echo "    output_base:     ${OUTPUT_BASE}"
echo ""

declare -a JOB_IDS=()
for METHOD in full_streaming duo_streaming rekv duo_plus_rekv; do
  jid=$(submit_method "${METHOD}")
  JOB_IDS+=("${jid}")
done

echo ""
echo "==> Submitted 4 jobs: ${JOB_IDS[*]}"
echo ""
echo "==> Monitor:"
echo "    squeue -u \${USER}"
echo "    tail -f ${LOG_DIR}/stream-*-sub${MAX_VIDEOS}-*.out"
echo ""
echo "==> After all jobs finish, compare results:"
echo "    python -m streaming.ReKV.compare_subsamples \\"
echo "        ${OUTPUT_BASE}/full_streaming/${TIMESTAMP}_results.json \\"
echo "        ${OUTPUT_BASE}/duo_streaming/${TIMESTAMP}_results.json \\"
echo "        ${OUTPUT_BASE}/rekv/${TIMESTAMP}_results.json \\"
echo "        ${OUTPUT_BASE}/duo_plus_rekv/${TIMESTAMP}_results.json \\"
echo "        --output-dir ${OUTPUT_BASE}/comparison/"
echo ""
echo "==> To judge answers (requires LLM judge):"
for METHOD in full_streaming duo_streaming rekv duo_plus_rekv; do
  echo "    python -m streaming.ReKV.judge_results --in-place ${OUTPUT_BASE}/${METHOD}/${TIMESTAMP}_results.json"
done
