#!/usr/bin/env bash
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

source /root/miniforge3/etc/profile.d/conda.sh
conda activate duo

cd "${ROOT}"

python -m streaming.ReKV.run_eval "$@"
