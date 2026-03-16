# Multi-modal DuoAttention (MMDA): Adapting DuoAttention for Efficient Multimodal Long-Context Video Question Answering

Link to original [[paper](https://arxiv.org/abs/2410.10819)] 

## Objective
We adapt the idea of DuoAttention to the multi-modal setting, specifically for video question answering tasks. By identifying retrieval heads that are crucial for processing long contexts and streaming heads that focus on recent tokens and attention sinks, we apply a full KV cache to retrieval heads while using a light-weight, constant-length KV cache for streaming heads. This approach significantly reduce both pre-filling and decoding memory and latency in VLMs, enabling efficient long-context video question answering without compromising accuracy.

## Environment Setup

```bash
conda create -yn duo python=3.10
conda activate duo

# Install CUDA toolkit
conda install -y \
  -c "nvidia/label/cuda-12.4.1" \
  -c conda-forge \
  cuda-toolkit

# Install PyTorch with CUDA 12.4 support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install transformers==4.45.2 opencv-python decord accelerate sentencepiece datasets wandb zstandard matplotlib huggingface_hub==0.25.2 tensor_parallel==2.0.0 ninja packaging

# Install FlashAttention
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl"

# Install FlashInfer
pip install "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl#sha256=d7605fbe3f14ef7f36e702f627c1f06e5a32495b5ebfe34313c3fb15f3e4eb06"

# Install DuoAttention
pip install -e .
```

### Install Block Sparse Streaming Attention

First, clone the Block Sparse Attention repository and navigate into it:

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd  Block-Sparse-Attention
```

Then, modify the following two functions in `setup.py` to ensure that the appropriate CUDA architectures are targeted for compilation (these are the GPUs on Slurm).
- 86 corresponds to RTX A2000 / A4000 / A4500 / A5000 / A6000 class,
- 89 corresponds to Ada / RTX 4090. 

```python
def cuda_archs() -> str:
    # CUDA 12.4 cluster build: target Ampere workstation + Ada
    return os.getenv("BLOCK_SPARSE_ATTN_CUDA_ARCHS", "86;89").split(";")
```

```python
def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    """
    CUDA 12.4-friendly gencodes for this cluster.

    Supported here:
      - 80 : A100
      - 86 : RTX A2000 / A4000 / A4500 / A5000 / A6000 class
      - 89 : Ada / RTX 4090
      - 90 : H100/H200

    We intentionally do NOT emit 100 / 110 / 120 for CUDA 12.4.
    We also only emit PTX for the newest arch we actually emitted.
    """
    emitted_archs = []

    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
        emitted_archs.append("80")

    if "86" in archs:
        cc_flag += ["-gencode", "arch=compute_86,code=sm_86"]
        emitted_archs.append("86")

    # Ada support
    if bare_metal_version >= Version("11.8") and "89" in archs:
        cc_flag += ["-gencode", "arch=compute_89,code=sm_89"]
        emitted_archs.append("89")

    # Hopper support
    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
        emitted_archs.append("90")

    # PTX only for the newest arch we actually emitted
    if emitted_archs:
        newest = max(emitted_archs, key=int)
        cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]

    return cc_flag
```
Then run the following command to install Block Sparse Streaming Attention (inside the `Block-Sparse-Attention` directory):

```bash
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="86;89"
export MAX_JOBS=4
python setup.py install
```

## Verify Installation
To verify that the installation was successful, you can run the following command in the root directory of the project:
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
import flash_attn
print("flash_attn ok")
import flashinfer
print("flashinfer ok")
import block_sparse_attn
import block_sparse_attn_cuda
print("block_sparse_attn:", block_sparse_attn.__file__)
print("block_sparse_attn_cuda:", block_sparse_attn_cuda.__file__)
print("block_sparse_attn ok")
PY
```