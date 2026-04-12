#!/usr/bin/env bash
# Submit a subset evaluation for all four streaming methods on SLURM.
# One job per method (4 jobs total), each on a single GPU.
# Default: 1 video (smoke test). Pass --max-videos N for larger subsets.
#
# Videos and annotations are downloaded from HuggingFace automatically.
# The HF cache is kept inside the project repo so nothing lands in home dir.
#
# Usage (from repo root on the login node):
#   bash scripts/run_streaming_subset3_slurm.sh [--max-videos 1|3|5] [--dataset rvs_ego|rvs_movie]
#
# Outputs (example for --max-videos 1):
#   outputs/evaluations_streaming/rvs-ego/subset1/full_streaming/<ts>_results.json
#   outputs/evaluations_streaming/rvs-ego/subset1/duo_streaming/<ts>_results.json
#   outputs/evaluations_streaming/rvs-ego/subset1/rekv/<ts>_results.json
#   outputs/evaluations_streaming/rvs-ego/subset1/duo_plus_rekv/<ts>_results.json
#   logs/stream-<method>-sub1-<jobid>.out
#
# Key environment overrides (all optional):
#   DATASET              rvs_ego | rvs_movie               (default: rvs_ego)
#   ATTN_DIR             path to Duo attention weights dir  (default: outputs/train/...)
#   MAX_NEW_TOKENS       tokens to generate per answer      (default: 64)
#   SAMPLE_FPS           frames/sec to sample from video    (default: 0.5)
#   VIDEO_DECODE_THREADS decord threads per job             (default: 4)
#   REKV_TOPK            ReKV top-k blocks to retrieve      (default: 64)
#   REKV_N_LOCAL         ReKV local window size in tokens   (default: 15000)
#   SPARSITY             Duo head sparsity fraction         (default: 0.5)
#   SBATCH_EXTRA_ARGS    extra sbatch flags as a string     (e.g. "--partition=gpu")

set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Defaults — match the prior successful full-eval run
# ---------------------------------------------------------------------------
DATASET="${DATASET:-rvs_ego}"
MODEL="${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"
ATTN_DIR="${ATTN_DIR:-${ROOT}/outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}"
MAX_VIDEOS=1
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SAMPLE_FPS="${SAMPLE_FPS:-0.5}"
VIDEO_DECODE_THREADS="${VIDEO_DECODE_THREADS:-1}"
REKV_TOPK="${REKV_TOPK:-64}"
REKV_N_LOCAL="${REKV_N_LOCAL:-15000}"
SPARSITY="${SPARSITY:-0.5}"

# HF cache: keep inside the project so nothing lands in home dir quota.
# Videos for 3 samples are a few GB max; scratch space has room.
HF_HOME="${ROOT}/.hf_cache"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
  echo "Usage: bash scripts/run_streaming_subset3_slurm.sh [--dataset rvs_ego|rvs_movie] [--attn-dir <path>]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)    DATASET="$2";    shift 2 ;;
    --attn-dir)   ATTN_DIR="$2";   shift 2 ;;
    --model)      MODEL="$2";      shift 2 ;;
    --max-videos) MAX_VIDEOS="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "[error] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if ! command -v sbatch >/dev/null 2>&1; then
  echo "[error] sbatch not found — run this on the cluster login node." >&2
  exit 1
fi

if [[ ! -d "${ATTN_DIR}" ]]; then
  echo "[error] Duo attention dir not found: ${ATTN_DIR}" >&2
  echo "        Set ATTN_DIR= to the correct path." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
OUTPUT_BASE="${ROOT}/outputs/evaluations_streaming/${DATASET//_/-}/subset${MAX_VIDEOS}"
SUBSAMPLE_NAME="subset${MAX_VIDEOS}_${TIMESTAMP}"

declare -a SBATCH_EXTRA=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  read -r -a SBATCH_EXTRA <<< "${SBATCH_EXTRA_ARGS}"
fi

# ---------------------------------------------------------------------------
# submit_method <method>  — submits one job, prints "<method>: job <id>" then job id
# ---------------------------------------------------------------------------
submit_method() {
  local method="$1"
  local output_path="${OUTPUT_BASE}/${method}/${TIMESTAMP}_results.json"
  mkdir -p "$(dirname "${output_path}")"

  local method_args=()
  case "${method}" in
    duo_streaming)
      method_args+=(
        --attn-dir "${ATTN_DIR}"
        --sparsity "${SPARSITY}"
        --duo-strict-no-sdpa-fallback
      )
      ;;
    rekv)
      method_args+=(
        --retrieve-size "${REKV_TOPK}"
        --n-local "${REKV_N_LOCAL}"
      )
      ;;
    duo_plus_rekv)
      method_args+=(
        --attn-dir "${ATTN_DIR}"
        --sparsity "${SPARSITY}"
        --duo-strict-no-sdpa-fallback
        --retrieve-size "${REKV_TOPK}"
        --n-local "${REKV_N_LOCAL}"
      )
      ;;
  esac

  local raw
  raw=$(sbatch \
    --job-name="stream-${method}-sub${MAX_VIDEOS}" \
    --nodes=1 \
    --ntasks=1 \
    --partition=gpunodes \
    --gres=gpu:rtx_a6000:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=02:00:00 \
    --output="${LOG_DIR}/stream-${method}-sub${MAX_VIDEOS}-%j.out" \
    "${SBATCH_EXTRA[@]}" \
    "--export=HF_HOME=${HF_HOME},TOKENIZERS_PARALLELISM=false" \
    streaming/ReKV/run_eval.sh \
      --dataset        "${DATASET}" \
      --model          "${MODEL}" \
      --allow-hf-video-download \
      --method         "${method}" \
      --sample-fps     "${SAMPLE_FPS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --video-decode-threads "${VIDEO_DECODE_THREADS}" \
      --max-videos     "${MAX_VIDEOS}" \
      --subsample-name "${SUBSAMPLE_NAME}" \
      --output-path    "${output_path}" \
      "${method_args[@]}")

  local job_id
  job_id=$(echo "${raw}" | awk '{print $NF}')
  echo "  ${method}: job ${job_id}  ->  ${output_path}"
  echo "${job_id}"
}

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
echo "==> ${MAX_VIDEOS}-video subset eval  [${TIMESTAMP}]"
echo "    dataset:    ${DATASET}"
echo "    model:      ${MODEL}"
echo "    attn_dir:   ${ATTN_DIR}"
echo "    hf_cache:   ${HF_HOME}"
echo "    output_dir: ${OUTPUT_BASE}"
echo "    logs:       ${LOG_DIR}"
echo ""

declare -a JOB_IDS=()
declare -a OUTPUT_PATHS=()
for METHOD in full_streaming duo_streaming rekv duo_plus_rekv; do
  # submit_method prints "  <method>: job <id>  ->  <path>" then prints just the id on its own line
  # Capture only the last line (job id); let the first line go to stdout
  raw_output=$(submit_method "${METHOD}" 2>&1)
  # The last line is the bare job id; preceding lines are informational
  jid=$(echo "${raw_output}" | tail -1)
  info=$(echo "${raw_output}" | head -n -1)
  echo "${info}"
  JOB_IDS+=("${jid}")
  OUTPUT_PATHS+=("${OUTPUT_BASE}/${METHOD}/${TIMESTAMP}_results.json")
done

echo ""
echo "==> Submitted jobs: ${JOB_IDS[*]}"
echo ""
cat <<MONITOR
==> Monitor progress:
    # All your jobs:
    squeue -u \${USER}

    # Live log for each method (open 4 terminals or use tmux):
    tail -f ${LOG_DIR}/stream-full_streaming-sub${MAX_VIDEOS}-*.out
    tail -f ${LOG_DIR}/stream-duo_streaming-sub${MAX_VIDEOS}-*.out
    tail -f ${LOG_DIR}/stream-rekv-sub${MAX_VIDEOS}-*.out
    tail -f ${LOG_DIR}/stream-duo_plus_rekv-sub${MAX_VIDEOS}-*.out

    # Or watch all at once (shows last 5 lines of each):
    watch -n 30 'tail -n 5 ${LOG_DIR}/stream-*-sub${MAX_VIDEOS}-*.out'

    # Check if a job has finished:
    sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS

MONITOR

cat <<NEXT
==> After all jobs finish — compare results:
    conda activate ${ROOT}/envs/duo
    cd ${ROOT}

    python -m streaming.ReKV.compare_subsamples \\
        ${OUTPUT_PATHS[0]} \\
        ${OUTPUT_PATHS[1]} \\
        ${OUTPUT_PATHS[2]} \\
        ${OUTPUT_PATHS[3]} \\
        --output-dir ${OUTPUT_BASE}/comparison/

==> Plots (auto-saved to the comparison dir):
    python -m streaming.ReKV.plot_results \\
        ${OUTPUT_PATHS[0]} \\
        ${OUTPUT_PATHS[1]} \\
        ${OUTPUT_PATHS[2]} \\
        ${OUTPUT_PATHS[3]} \\
        --output-dir ${OUTPUT_BASE}/plots/

==> Results land in:
    ${OUTPUT_BASE}/
NEXT
