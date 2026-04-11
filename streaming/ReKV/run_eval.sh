#!/usr/bin/env bash
# Single-GPU SLURM eval job for streaming ReKV methods.
#
# Always submit from the repo root with an absolute --output path:
#   sbatch --output="$(pwd)/logs/%x-%j.out" streaming/ReKV/run_eval.sh --method rekv ...
#
# Or use the submit wrapper which handles all of this:
#   bash scripts/run_streaming_subset3_slurm.sh
#   bash scripts/run_streaming_eval_slurm_array.sh
#
#SBATCH --job-name=streaming-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "${ROOT}/logs"

# HF_HOME: keep model/dataset cache inside the project if set by caller,
# otherwise fall back to ~/.cache/huggingface (safe on Toronto cluster /h).
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"

python -m streaming.ReKV.run_eval "$@"
