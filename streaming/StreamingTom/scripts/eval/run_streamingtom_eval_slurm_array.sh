#!/usr/bin/env bash
# Multi-GPU sharded SLURM array job for StreamingTom evaluation.
# Each array task handles one contiguous chunk of videos on one GPU.
#
# Usage (submit from repo root):
#   NUM_CHUNKS=10 DATASET=rvs_ego METHOD=streamingtom \
#     sbatch --array=0-9 streaming/StreamingTom/scripts/eval/run_streamingtom_eval_slurm_array.sh
#
# Methods:
#   streamingtom | duo_plus_streamingtom

#SBATCH --job-name=streamingtom-eval-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

ROOT=${ROOT:-/root/streaming-vqa}
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="${ROOT}:${ROOT}/streaming/StreamingTom:${PYTHONPATH:-}"

# StreamingTom requires the NAMED 'duo' conda env (torch 2.5.1, transformers 4.53.3,
# LLaVA-NeXT + lmms-eval installed editable). Do NOT use activate_streaming_env here
# — it picks up the project-local envs/duo (the ReKV env) instead.
CONDA_INIT_SCRIPT=""
for _candidate in \
  "/u/navdeep/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "/root/miniconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${_candidate}" ]]; then
    CONDA_INIT_SCRIPT="${_candidate}"
    break
  fi
done
if [[ -z "${CONDA_INIT_SCRIPT}" ]]; then
  echo "[error] Cannot find conda init script" >&2; exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_INIT_SCRIPT}"
# StreamingTom env is installed in scratch space to avoid home quota limits.
DUO_ST_ENV="${ROOT}/envs/duo-st"
if [[ -d "${DUO_ST_ENV}" ]]; then
  conda activate "${DUO_ST_ENV}"
else
  # Fallback to named 'duo' env if scratch env not yet created.
  conda activate duo
fi
echo "[env] Using python: $(which python)"

cd "${ROOT}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required. Submit with sbatch --array=..." >&2
  exit 1
fi

DATASET=${DATASET:-rvs_ego}
MODEL=${MODEL:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
METHOD=${METHOD:-streamingtom}
HF_REPO_ID=${HF_REPO_ID:-Becomebright/RVS}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-1}
NUM_CHUNKS=${NUM_CHUNKS:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/outputs/evaluations_streaming/${DATASET//_/-}/full_eval/run2}
RESUME=${RESUME:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

STREAMINGTOM_ROOT=${STREAMINGTOM_ROOT:-streaming/StreamingTom}
DUO_ATTN_DIR=${DUO_ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
DUO_HEADS_FILE=${DUO_HEADS_FILE:-${DUO_ATTN_DIR}/full_attention_heads_latest.tsv}
DUO_THRESHOLD=${DUO_THRESHOLD:-0.5}
DUO_SPARSITY=${DUO_SPARSITY:-0.75}
DUO_SINK_SIZE=${DUO_SINK_SIZE:-256}
DUO_RECENT_SIZE=${DUO_RECENT_SIZE:-512}

CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
if (( CHUNK_INDEX >= NUM_CHUNKS )); then
  echo "SLURM_ARRAY_TASK_ID=${CHUNK_INDEX} must be smaller than NUM_CHUNKS=${NUM_CHUNKS}" >&2
  exit 1
fi

OUTPUT_PATH="${OUTPUT_ROOT}/${METHOD}/chunk_$(printf '%03d' "${CHUNK_INDEX}").json"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "[array task ${CHUNK_INDEX}/${NUM_CHUNKS}] dataset=${DATASET} method=${METHOD} output=${OUTPUT_PATH}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --hf-repo-id "${HF_REPO_ID}"
  --allow-hf-video-download
  --model "${MODEL}"
  --method "${METHOD}"
  --sample-fps "${SAMPLE_FPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --video-decode-threads "${VIDEO_DECODE_THREADS}"
  --num-chunks "${NUM_CHUNKS}"
  --chunk-index "${CHUNK_INDEX}"
  --output-path "${OUTPUT_PATH}"
  --streamingtom-root "${STREAMINGTOM_ROOT}"
)

if [[ "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

if [[ "${METHOD}" == "duo_plus_streamingtom" ]]; then
  COMMON_ARGS+=(
    --duo-attn-dir "${DUO_ATTN_DIR}"
    --duo-heads-file "${DUO_HEADS_FILE}"
    --duo-threshold "${DUO_THRESHOLD}"
    --duo-sparsity "${DUO_SPARSITY}"
    --duo-sink-size "${DUO_SINK_SIZE}"
    --duo-recent-size "${DUO_RECENT_SIZE}"
  )
fi

python streaming/StreamingTom/scripts/eval/run_eval.py \
  "${COMMON_ARGS[@]}" \
  ${EXTRA_ARGS}
