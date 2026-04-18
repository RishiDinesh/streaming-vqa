#!/usr/bin/env python
from __future__ import annotations

import importlib


def main() -> None:
    import torch

    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    for module_name in (
        "flash_attn",
        "flashinfer",
        "block_sparse_attn",
        "block_sparse_attn_cuda",
    ):
        module = importlib.import_module(module_name)
        print(f"{module_name}: {getattr(module, '__file__', '<builtin>')}")

    print("installation ok")


if __name__ == "__main__":
    main()
