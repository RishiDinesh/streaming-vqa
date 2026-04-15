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

# Hardcoded: BASH_SOURCE[0] is unreliable when SLURM copies the script to spool.
ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
mkdir -p "${ROOT}/logs"

# Always keep HF cache inside the project scratch space, not home dir quota.
export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"

# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"

python -m streaming.ReKV.run_eval "$@"
