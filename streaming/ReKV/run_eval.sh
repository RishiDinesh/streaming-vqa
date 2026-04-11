#!/usr/bin/env bash
#SBATCH --job-name=streaming-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "${ROOT}/logs"

# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

python -m streaming.ReKV.run_eval "$@"
