import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ratio-sweep VNBench validation using all-full baseline for "
            "Llava-OneVision DuoAttention."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="./datasets/vnbench/videos",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="./datasets/vnbench/anno.jsonl",
    )
    parser.add_argument("--num_frames", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "Must be 1. The patched reordered Qwen2 eval attention path does not "
            "apply attention_mask, so padded batches (>1) can skew metrics."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate for answer decoding.",
    )
    parser.add_argument("--disable_video_chat_template", action="store_true")
    parser.add_argument(
        "--video_answer_prefix",
        type=str,
        default="",
        help=(
            "Optional answer prefix appended to the generation prompt before "
            "decoding. Leave empty to let the model generate the full answer."
        ),
    )

    parser.add_argument("--attn_load_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sparsity", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help=(
            "Path to save a line plot of ratio-vs-accuracy for the two run "
            "directions plus the dotted full-baseline accuracy."
        ),
    )
    return parser.parse_args()


def ensure_default_output_paths(args: argparse.Namespace) -> None:
    exp_name = os.path.basename(os.path.normpath(args.attn_load_dir))
    val_dir = Path("outputs") / "validation" / f"{exp_name}_val"
    val_dir.mkdir(parents=True, exist_ok=True)

    if args.output_json is None:
        args.output_json = str(val_dir / "retrieval_pool_ratio_ablation.json")
    if args.output_csv is None:
        args.output_csv = str(val_dir / "retrieval_pool_ratio_ablation.csv")
    if args.output_plot is None:
        args.output_plot = str(val_dir / "retrieval_pool_ratio_ablation_plot.png")


def resolve_device_and_dtype(args: argparse.Namespace) -> Tuple[torch.device, torch.dtype]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        raise ValueError(f"Unsupported device: {args.device}")

    if args.dtype == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    return device, dtype


def init_distributed(device: torch.device) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return False, rank, world_size, local_rank

    if device.type != "cuda":
        raise ValueError("Distributed validation currently requires CUDA.")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return True, rank, world_size, local_rank


def cleanup_distributed(use_distributed: bool) -> None:
    if use_distributed and dist.is_initialized():
        dist.destroy_process_group()
