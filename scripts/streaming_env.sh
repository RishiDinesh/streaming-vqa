#!/usr/bin/env bash
# Environment activation for NVIDIA SLURM cluster.
# Activates the 'duo' conda environment which must have all dependencies installed.
#
# Priority order:
#   1. Project-local env at <repo-root>/envs/duo  (created by setup.sh — preferred)
#   2. Named conda env 'duo' in the user's default conda prefix
#   3. Current Python (if all required packages are importable) — last resort
#
# The project-local env keeps all packages inside scratch space and avoids
# touching other users' environments or the home-directory quota.

activate_streaming_env() {
  local repo_root
  repo_root=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  local project_env_prefix="${repo_root}/envs/duo"

  _init_conda() {
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      return 0
    fi
    local candidate
    for candidate in \
      "${HOME}/miniforge3/etc/profile.d/conda.sh" \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "/root/miniforge3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh" \
      "/usr/local/miniconda3/etc/profile.d/conda.sh"
    do
      if [[ -f "${candidate}" ]]; then
        # shellcheck disable=SC1090
        source "${candidate}"
        return 0
      fi
    done
    return 1
  }

  # --- 1. Project-local prefix env (preferred) ---
  if [[ -d "${project_env_prefix}" ]]; then
    if _init_conda; then
      conda activate "${project_env_prefix}"
      echo "[env] Activated project-local env: ${project_env_prefix}" >&2
      return 0
    fi
  fi

  # --- 2. Named 'duo' env in default conda prefix ---
  if _init_conda; then
    if conda activate duo 2>/dev/null; then
      echo "[env] Activated named conda env 'duo'" >&2
      return 0
    fi
  fi

  # --- 3. Current Python sanity check ---
  if python - <<'PY' >/dev/null 2>&1
import importlib
for name in ("torch", "numpy", "matplotlib", "transformers", "tqdm", "duo_attn"):
    importlib.import_module(name)
PY
  then
    echo "[warn] conda 'duo' env not found; using current Python: $(command -v python)" >&2
    echo "[warn] Run 'bash setup.sh' from the repo root to create the project-local env." >&2
    return 0
  fi

  echo "[error] Could not activate 'duo' conda env and current Python is missing required packages." >&2
  echo "[error] Run 'bash setup.sh' from the repo root to set up the environment." >&2
  exit 1
}
