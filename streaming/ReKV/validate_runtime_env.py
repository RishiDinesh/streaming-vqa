#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .methods import (
    collect_runtime_backend_info,
    resolve_device,
    resolve_dtype,
    resolve_duo_backend_stack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the current accelerator/runtime backend and report the actual "
            "streaming ReKV/Duo backend stack that will be used."
        )
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--duo-strict-no-sdpa-fallback", action="store_true")
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    runtime_backend = collect_runtime_backend_info(device, dtype)
    duo_backend = resolve_duo_backend_stack(streaming_attn_backend_requested="blocksparse")

    payload = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "torch_runtime": {
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "method_backend_resolution": {
            "shared_runtime_backend": runtime_backend,
            "full_streaming": {
                "backend_resolution": {
                    "attention_module_load_path": runtime_backend.get("attention_module_load_path"),
                    "flash_attn_available": runtime_backend.get("flash_attn_available"),
                    "flashinfer_available": runtime_backend.get("flashinfer_available"),
                    "block_sparse_attn_available": runtime_backend.get("block_sparse_attn_available"),
                }
            },
            "duo_streaming": {
                "backend_resolution": duo_backend,
                "duo_backend_policy": {
                    "strict_no_sdpa_fallback": bool(args.duo_strict_no_sdpa_fallback),
                },
            },
            "rekv": {
                "backend_resolution": {
                    "attention_module_load_path": runtime_backend.get("attention_module_load_path"),
                    "rekv_dot_backend_requested": "torch",
                    "rekv_dot_backend_actual": "torch",
                }
            },
            "duo_plus_rekv": {
                "backend_resolution": {
                    **duo_backend,
                    "rekv_dot_backend_requested": "torch",
                    "rekv_dot_backend_actual": "torch",
                },
                "duo_backend_policy": {
                    "strict_no_sdpa_fallback": bool(args.duo_strict_no_sdpa_fallback),
                },
            },
        },
        "warnings": [],
    }

    if (
        args.duo_strict_no_sdpa_fallback
        and duo_backend.get("streaming_attn_backend_actual") == "sdpa"
    ):
        raise SystemExit(
            "Duo strict mode requested, but the current environment resolves the Duo "
            f"streaming backend to SDPA fallback: {duo_backend.get('streaming_attn_fallback_reason')}"
        )

    if duo_backend.get("streaming_attn_backend_actual") == "sdpa":
        payload["warnings"].append(
            "Duo streaming attention resolves to SDPA fallback. "
            "block_sparse_attn is not installed or not detected. "
            "Install block_sparse_attn (see README) for paper-faithful native sparse Duo streaming on NVIDIA. "
            "Results labeled 'sdpa_fallback_duo' are a valid baseline but not equivalent to the DuoAttention paper kernel."
        )
    if not runtime_backend.get("flashinfer_available"):
        payload["warnings"].append(
            "flashinfer is unavailable; standard transformer RMSNorm/RoPE paths will be used."
        )
    return payload


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve(strict=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved runtime validation summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
