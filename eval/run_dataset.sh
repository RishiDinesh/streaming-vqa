#!/usr/bin/env bash
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=rishidinesh@cs.toronto.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch eval/run_dataset.sh <dataset> [pretrained] [attn_dir] [extra launcher args...]

Supported datasets:
  egoschema_subset
  longvideobench_i
  longvideobench_v
  mlvu_dev
  videomme

You can also provide DATASET, PRETRAINED, and ATTN_DIR as environment variables.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_FROM_SCRIPT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-${ROOT_FROM_SCRIPT}}}
PARENT_DIR=$(cd -- "${ROOT}/.." && pwd)
LAUNCHER_PATH="${ROOT}/eval/launch_duo_lmms_eval.py"

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_ASYNC_ERROR_HANDLING=1

DEFAULT_PYTHON_BIN="${PARENT_DIR}/.conda/envs/duo/bin/python"
PYTHON_BIN=${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Could not find the duo environment python at ${PYTHON_BIN}." >&2
    echo "Set PYTHON_BIN explicitly to the python inside the 'duo' Conda environment." >&2
    exit 1
fi

if [[ ! -f "${LAUNCHER_PATH}" ]]; then
    echo "Launcher not found at ${LAUNCHER_PATH}." >&2
    exit 1
fi

DATASET=${DATASET:-}
if [[ -z "${DATASET}" ]]; then
    if [[ $# -eq 0 ]]; then
        usage >&2
        exit 1
    fi
    DATASET=$1
    shift
fi

DATASET_KEY=${DATASET,,}
DATASET_KEY=${DATASET_KEY//-/_}

REQUIRES_LONGVIDEOBENCH_TOKEN=0
ALLOW_DEPLOY_OVERRIDES=1
DEFAULT_DECODING_SIMULATION_LENGTH=256

case "${DATASET_KEY}" in
    egoschema|egoschema_subset)
        DATASET_NAME=egoschema_subset
        TASK=egoschema_subset
        ;;
    longvideobench_i|longvideobench_val_i)
        DATASET_NAME=longvideobench_i
        TASK=longvideobench_val_i
        REQUIRES_LONGVIDEOBENCH_TOKEN=1
        ALLOW_DEPLOY_OVERRIDES=0
        ;;
    longvideobench_v|longvideobench_val_v)
        DATASET_NAME=longvideobench_v
        TASK=longvideobench_val_v
        REQUIRES_LONGVIDEOBENCH_TOKEN=1
        ;;
    mlvu|mlvu_dev)
        DATASET_NAME=mlvu_dev
        TASK=mlvu_dev
        ;;
    videomme)
        DATASET_NAME=videomme
        TASK=videomme
        ;;
    *)
        echo "Unsupported dataset '${DATASET}'. Expected one of: egoschema_subset, longvideobench_i, longvideobench_v, mlvu_dev, videomme." >&2
        exit 1
        ;;
esac

if [[ "${REQUIRES_LONGVIDEOBENCH_TOKEN}" == "1" ]]; then
    "${PYTHON_BIN}" - <<'PY'
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

repo_id = "longvideobench/LongVideoBench"
token = get_token()

if not token:
    print(
        "No Hugging Face token found for LongVideoBench.\n"
        "This dataset is gated, so you must both request access on the dataset page\n"
        "and log in in the same environment used by the SLURM job.\n\n"
        "Login command:\n"
        "  python -m huggingface_hub.commands.huggingface_cli login\n\n"
        "Verification command:\n"
        "  python -m huggingface_hub.commands.huggingface_cli whoami",
        file=sys.stderr,
    )
    raise SystemExit(1)

try:
    HfApi(token=token).dataset_info(repo_id)
except Exception as exc:
    print(
        f"Authenticated access check failed for gated dataset '{repo_id}'.\n"
        "Make sure your Hugging Face account has been granted access and that the\n"
        "token belongs to that same account.\n\n"
        f"Original error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
fi

PRETRAINED=${PRETRAINED:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
ATTN_DIR=${ATTN_DIR:-}
if [[ $# -gt 0 ]]; then
    PRETRAINED=$1
    shift
fi
if [[ $# -gt 0 && "$1" != --* ]]; then
    ATTN_DIR=$1
    shift
fi

MODE=${MODE:-auto}
BATCH_SIZE=${BATCH_SIZE:-1}
FPS=${FPS:-auto}
NUM_PROCESSES=${NUM_PROCESSES:-1}
NUM_MACHINES=${NUM_MACHINES:-${SLURM_NNODES:-1}}
VIDEO_DECODE_BACKEND=${VIDEO_DECODE_BACKEND:-decord}
DEVICE_MAP=${DEVICE_MAP:-}
SPARSITY=${SPARSITY:-}
THRESHOLD=${THRESHOLD:-}
LIMIT=${LIMIT:-}
OUTPUT_NAME=${OUTPUT_NAME:-}
LOG_SAMPLES=${LOG_SAMPLES:-1}
MAX_FRAMES_NUM=${MAX_FRAMES_NUM:-64}
CAP_FPS_SAMPLING=${CAP_FPS_SAMPLING:-1}
DECODING_SIMULATION_LENGTH=${DECODING_SIMULATION_LENGTH:-${DEFAULT_DECODING_SIMULATION_LENGTH}}
DEPLOY_SINK_SIZE=${DEPLOY_SINK_SIZE:-}
DEPLOY_RECENT_SIZE=${DEPLOY_RECENT_SIZE:-}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
SAME_NETWORK=${SAME_NETWORK:-1}

if [[ -z "${ATTN_DIR}" && "${MODE}" == "duo" ]]; then
    PRETRAINED_LOWER=${PRETRAINED,,}
    case "${PRETRAINED_LOWER}" in
        *0.5b*|*0p5b*)
            ATTN_DIR="outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632"
            ;;
        *7b*)
            ATTN_DIR="outputs/train/7b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5"
            ;;
    esac
fi

if [[ "${NUM_PROCESSES}" == "1" && "${NUM_MACHINES}" -gt 1 ]]; then
    NUM_PROCESSES=${NUM_MACHINES}
fi

CMD=(
    "${PYTHON_BIN}"
    "${LAUNCHER_PATH}"
    --pretrained "${PRETRAINED}"
    --task "${TASK}"
    --mode "${MODE}"
    --batch-size "${BATCH_SIZE}"
    --fps "${FPS}"
    --num-processes "${NUM_PROCESSES}"
    --num-machines "${NUM_MACHINES}"
    --max-frames-num "${MAX_FRAMES_NUM}"
    --decoding-simulation-length "${DECODING_SIMULATION_LENGTH}"
    --video-decode-backend "${VIDEO_DECODE_BACKEND}"
    --sparsity "${SPARSITY}"
)

if [[ -n "${DEVICE_MAP}" ]]; then
    CMD+=(--device-map "${DEVICE_MAP}")
fi
if [[ "${ALLOW_DEPLOY_OVERRIDES}" != "0" && -n "${DEPLOY_SINK_SIZE}" ]]; then
    CMD+=(--deploy-sink-size "${DEPLOY_SINK_SIZE}")
fi
if [[ "${ALLOW_DEPLOY_OVERRIDES}" != "0" && -n "${DEPLOY_RECENT_SIZE}" ]]; then
    CMD+=(--deploy-recent-size "${DEPLOY_RECENT_SIZE}")
fi
if [[ -n "${ATTN_DIR}" ]]; then
    CMD+=(--attn-dir "${ATTN_DIR}")
fi
if [[ -n "${THRESHOLD}" ]]; then
    CMD+=(--threshold "${THRESHOLD}")
fi
if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
fi
if [[ -n "${OUTPUT_NAME}" ]]; then
    CMD+=(--output-name "${OUTPUT_NAME}")
fi
if [[ "${LOG_SAMPLES}" != "0" ]]; then
    CMD+=(--log-samples)
fi
if [[ "${CAP_FPS_SAMPLING}" != "0" ]]; then
    CMD+=(--cap-fps-sampling)
fi
if [[ "${SAME_NETWORK}" != "0" ]]; then
    CMD+=(--same-network)
fi
CMD+=("$@")

echo "Working directory: ${ROOT}"
echo "Dataset: ${DATASET_NAME} -> ${TASK}"
echo "Python: ${PYTHON_BIN}"
echo "Running command: ${CMD[*]}"

if [[ "${NUM_MACHINES}" -gt 1 ]]; then
    if [[ -z "${MASTER_ADDR:-}" ]]; then
        if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
            MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
        else
            MASTER_ADDR=$(hostname -I | awk '{print $1}')
        fi
    fi
    export MASTER_ADDR
    export MASTER_PORT=${MAIN_PROCESS_PORT}
    echo "Distributed eval setup: machines=${NUM_MACHINES}, processes=${NUM_PROCESSES}, master=${MASTER_ADDR}:${MASTER_PORT}, max_frames_num=${MAX_FRAMES_NUM}, cap_fps_sampling=${CAP_FPS_SAMPLING}, decoding_simulation_length=${DECODING_SIMULATION_LENGTH}"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun --nodes="${NUM_MACHINES}" --ntasks="${NUM_MACHINES}" --ntasks-per-node=1 --kill-on-bad-exit=1 "${CMD[@]}"
    else
        "${CMD[@]}"
    fi
else
    "${CMD[@]}"
fi
