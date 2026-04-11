# Multi-modal DuoAttention (MMDA): Adapting DuoAttention for Efficient Multimodal Long-Context Video Question Answering

Link to original [[paper](https://arxiv.org/abs/2410.10819)] 

## Objective
We adapt the idea of DuoAttention to the multi-modal setting, specifically for video question answering tasks. By identifying retrieval heads that are crucial for processing long contexts and streaming heads that focus on recent tokens and attention sinks, we apply a full KV cache to retrieval heads while using a light-weight, constant-length KV cache for streaming heads. This approach significantly reduce both pre-filling and decoding memory and latency in VLMs, enabling efficient long-context video question answering without compromising accuracy.

## Flowchart
![image](images/duo_atten_crop.png)

## Environment Setup (AMD/ROCm, tested on MI300X VF)

This setup was validated in this repo on:
- Ubuntu 24.04.2
- ROCm runtime with `gfx942` (AMD Instinct MI300X VF)
- Conda env: `duo`

```bash
conda create -yn duo python=3.10
conda activate duo

# PyTorch ROCm build (ROCm 6.1 wheel line)
pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/rocm6.1

# Core dependencies used by training/eval/scripts
pip install \
  transformers==4.45.2 \
  opencv-python \
  decord \
  accelerate \
  sentencepiece \
  datasets \
  wandb \
  zstandard \
  matplotlib \
  huggingface_hub==0.25.2 \
  tensor_parallel==2.0.0 \
  ninja \
  packaging \
  tqdm \
  requests \
  pandas \
  seaborn \
  jieba \
  fuzzywuzzy \
  rouge

# tensor_parallel currently imports pkg_resources
pip install "setuptools<81"

# Install this repo
pip install -e .
```

### AMD Notes on Optional Acceleration Libraries

- The currently documented `flash_attn`, `flashinfer`, and `block_sparse_attn` install paths are CUDA-specific.
- On ROCm machines, those CUDA wheels fail at runtime (missing `libcudart.so.12`), and source builds are not wired in this repo yet.
- The repo now includes ROCm-safe fallbacks:
  - `flash_attn` API calls fallback to PyTorch SDPA kernels.
  - `flashinfer` RMSNorm/RoPE hooks are optional.
  - `blocksparse` streaming attention falls back to `sdpa` when unavailable.

### Install LMMs-Eval (eval only)

This project uses a patched fork of `lmms-eval` for evaluation. To setup:

```bash
cd /path/to/streaming-vqa
rm -rf lmms-eval
git clone -b duo_attn https://github.com/RishiDinesh/lmms-eval.git
cd lmms-eval
git checkout 146a2836a3e34347d611a348592f25ac22958589
pip install -e .
cd ..
```

### Install Block Sparse Streaming Attention (CUDA-only today)

This project uses a patched fork of Block Sparse Attention with the required `setup.py` changes already applied. To setup:

```bash
cd /path/to/streaming-vqa
rm -rf Block-Sparse-Attention
git clone -b mm_duo_attn https://github.com/RishiDinesh/Block-Sparse-Attention.git
cd Block-Sparse-Attention
git checkout 8e73b82a47de87dba3110e40c38c35f41c3f5d0d
```

This fork targets NVIDIA cluster GPUs through `setup.py`.
- 86 corresponds to RTX A2000 / A4000 / A4500 / A5000 / A6000 class,
- 89 corresponds to Ada / RTX 4090. 

Then install it from inside the `Block-Sparse-Attention` directory:

```bash
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="86;89"
export MAX_JOBS=4
python setup.py install
cd ..
```

## Verify Installation
To verify that the AMD environment installation was successful, run:
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

import transformers
import decord
import cv2
import datasets
import duo_attn
print("core deps ok")

for name in ("flash_attn", "flashinfer", "block_sparse_attn"):
    try:
        __import__(name)
        print(f"{name}: available")
    except Exception as exc:
        print(f"{name}: not available ({type(exc).__name__})")
PY
```
