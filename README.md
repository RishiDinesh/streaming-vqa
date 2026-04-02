# Multi-modal DuoAttention (MMDA): Adapting DuoAttention for Efficient Multimodal Long-Context Video Question Answering

Link to original [[paper](https://arxiv.org/abs/2410.10819)] 

## Objective
We adapt the idea of DuoAttention to the multi-modal setting, specifically for video question answering tasks. By identifying retrieval heads that are crucial for processing long contexts and streaming heads that focus on recent tokens and attention sinks, we apply a full KV cache to retrieval heads while using a light-weight, constant-length KV cache for streaming heads. This approach significantly reduce both pre-filling and decoding memory and latency in VLMs, enabling efficient long-context video question answering without compromising accuracy.

## Flowchart
![image](images/duo_atten_crop.png)

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

### Install Block Sparse Streaming Attention

This project uses a patched fork of Block Sparse Attention with the required `setup.py` changes already applied. To setup:

```bash
cd /path/to/streaming-vqa
rm -rf Block-Sparse-Attention
git clone -b mm_duo_attn https://github.com/RishiDinesh/Block-Sparse-Attention.git
cd Block-Sparse-Attention
git checkout 8e73b82a47de87dba3110e40c38c35f41c3f5d0d
```

This fork targets the cluster GPUs through `setup.py`.
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
