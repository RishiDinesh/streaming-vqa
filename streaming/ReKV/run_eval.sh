#!/usr/bin/env bash
# Single-GPU SLURM eval job for streaming ReKV methods.
# Submit from the repo root:
#   sbatch streaming/ReKV/run_eval.sh --dataset rvs_ego --method rekv [...]
#
# The --output log path is set on the sbatch command line by the submit wrapper
# so it resolves to an absolute path under <repo-root>/logs/.
# If submitting this script directly, pass --output yourself:
#   sbatch --output=/abs/path/logs/%x-%j.out streaming/ReKV/run_eval.sh ...
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

# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

python -m streaming.ReKV.run_eval "$@"
