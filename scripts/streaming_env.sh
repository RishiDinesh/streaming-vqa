#!/usr/bin/env bash

activate_streaming_env() {
  local _check_pkgs='
import importlib
for name in ("torch", "numpy", "matplotlib", "transformers", "tqdm", "duo_attn"):
    importlib.import_module(name)
'
  local _preference=${STREAMING_ENV_PREFERENCE:-duo}

  _activate_conda_duo() {
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate duo
      echo "[env] Using conda env 'duo'" >&2
      return 0
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
        echo "[env] Using conda env 'duo'" >&2
        return 0
      fi
    done
    return 1
  }

  _activate_opt_venv() {
    local label=$1
    if [[ -x "/opt/venv/bin/python" ]]; then
      if /opt/venv/bin/python - <<PY >/dev/null 2>&1
${_check_pkgs}
PY
      then
        export PATH="/opt/venv/bin:${PATH}"
        echo "[env] Using /opt/venv (${label})" >&2
        return 0
      fi
    fi
    return 1
  }

  case "${_preference}" in
    duo)
      _activate_conda_duo && return
      ;;
    rocm)
      _activate_opt_venv "preferred ROCm runtime" && return
      ;;
    auto)
      _activate_conda_duo && return
      if command -v rocm-smi >/dev/null 2>&1 || [[ -e /dev/kfd ]]; then
        _activate_opt_venv "auto-selected ROCm runtime" && return
      fi
      ;;
    *)
      echo "[warn] Unknown STREAMING_ENV_PREFERENCE=${_preference}; falling back to auto" >&2
      _activate_conda_duo && return
      if command -v rocm-smi >/dev/null 2>&1 || [[ -e /dev/kfd ]]; then
        _activate_opt_venv "auto-selected ROCm runtime" && return
      fi
      ;;
  esac

  _activate_opt_venv "fallback runtime" && return

  if python - <<'PY' >/dev/null 2>&1
import importlib
for name in ("torch", "numpy", "matplotlib", "transformers", "tqdm"):
    importlib.import_module(name)
PY
  then
    echo "[warn] Could not locate preferred streaming environment; using current Python: $(command -v python)" >&2
    return
  fi

  echo "Could not locate a suitable Python environment with required packages." >&2
  exit 1
}
