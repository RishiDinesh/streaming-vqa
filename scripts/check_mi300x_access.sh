#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

echo "== MI300X Access Check =="
echo "cwd: ${ROOT}"
echo

echo "== Environment =="
env | rg '^(CUDA|HIP|HSA|ROCR|ROCM|AMD|GPU|SLURM|SINGULARITY|APPTAINER)' | sort || true
echo

echo "== User / Groups =="
id
echo

echo "== Sysfs DRM entries =="
ls -l /sys/class/drm 2>/dev/null || true
echo

echo "== Device nodes =="
ls -l /dev/kfd /dev/dri 2>/dev/null || true
echo

echo "== rocminfo =="
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | sed -n '1,80p' || true
else
  echo "rocminfo_unavailable"
fi
echo

echo "== rocm-smi =="
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || true
else
  echo "rocm_smi_unavailable"
fi
echo

echo "== Torch visibility =="
python - <<'PY'
import os
import torch

print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("hip_version:", getattr(torch.version, "hip", None))
print("cuda_version:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"device[{idx}]:", torch.cuda.get_device_name(idx))
PY
echo

echo "== Direct /dev/kfd open test =="
python - <<'PY'
try:
    fd = open("/dev/kfd", "rb+")
except Exception as exc:
    print(type(exc).__name__, exc)
else:
    print("open_ok")
    fd.close()
PY
echo

echo "== Summary =="
python - <<'PY'
from pathlib import Path
import torch

kfd = Path("/dev/kfd")
dri = Path("/dev/dri")
if not kfd.exists():
    print("Missing /dev/kfd: GPU device nodes are not exposed in this session.")
elif not dri.exists():
    print("Missing /dev/dri: render nodes are not exposed in this session.")
else:
    try:
        open("/dev/kfd", "rb+").close()
    except Exception as exc:
        print(f"/dev/kfd exists but cannot be opened: {type(exc).__name__}: {exc}")
        print("This usually means the session/container lacks GPU device permissions or passthrough.")
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print("Torch can see the GPU. The MI300X path looks healthy.")
        else:
            print("ROCm device nodes are accessible, but torch still cannot see a GPU.")
            print("This points to a ROCm/PyTorch runtime mismatch rather than a missing device mount.")
PY
