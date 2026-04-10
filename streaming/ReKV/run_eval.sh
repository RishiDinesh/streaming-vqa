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

activate_duo_env() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate duo
        return
    fi

    local candidate
    for candidate in \
        "/root/miniforge3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh" \
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
    do
        if [[ -f "${candidate}" ]]; then
            # shellcheck disable=SC1090
            source "${candidate}"
            conda activate duo
            return
        fi
    done

    echo "Could not locate Conda activation script for the 'duo' environment." >&2
    exit 1
}

activate_duo_env

cd "${ROOT}"

python -m streaming.ReKV.run_eval "$@"
