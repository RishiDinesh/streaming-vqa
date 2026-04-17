#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.eval.efficiency.benchmark_dynamic_llava import move_batch_to_device
from duo_attn.eval.efficiency.utils import bench_func
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.train import build_llava_video_inputs_embeds
from duo_attn.utils import (
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)


@dataclass
class SweepPoint:
    target_context: int
    actual_context: int
    prefix_seconds: float
    prefix_ratio: float
    num_frames: int


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO_PATH = Path("data") / "benchmark_sample.mp4"
MODEL_SCALE_TO_REPO = {
    "0.5b": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    "7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone LLaVA efficiency suite for context sweep, prefill chunk "
            "sweep, and plot regeneration using the full-model multimodal "
            "benchmark flow."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    context_parser = subparsers.add_parser("context", help="Run context sweep.")
    add_shared_args(context_parser)
    context_parser.add_argument("--max_context", type=int, default=32000)
    context_parser.add_argument("--num_points", type=int, default=5)
    context_parser.add_argument("--target_contexts", type=int, nargs="+", default=None)
    context_parser.add_argument("--decode_tokens", type=int, default=100)

    prefill_parser = subparsers.add_parser(
        "prefill", help="Run fixed-context prefill chunk sweep over multimodal embeddings."
    )
    add_shared_args(prefill_parser)
    prefill_parser.add_argument("--target_context", type=int, default=32000)
    prefill_parser.add_argument(
        "--prefill_chunk_sizes",
        type=int,
        nargs="+",
        default=[4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000],
    )

    context_plot_parser = subparsers.add_parser(
        "context-plot", help="Regenerate context sweep plot from JSON."
    )
    context_plot_parser.add_argument("--input_json", type=str, required=True)
    context_plot_parser.add_argument("--output_plot", type=str, default=None)

    prefill_plot_parser = subparsers.add_parser(
        "prefill-plot", help="Regenerate prefill chunk sweep plot from JSON."
    )
    prefill_plot_parser.add_argument("--input_json", type=str, required=True)
    prefill_plot_parser.add_argument("--config_json", type=str, default=None)
    prefill_plot_parser.add_argument("--output_plot", type=str, default=None)
    prefill_plot_parser.add_argument("--title", type=str, default=None)

    return parser.parse_args()


def add_shared_args(parser):
    parser.add_argument(
        "--model_scale",
        "--model-scale",
        dest="model_scale",
        choices=tuple(MODEL_SCALE_TO_REPO),
        default=None,
        help="Model scale to load from Hugging Face. One of: 0.5b, 7b.",
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        dest="model_name",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--video_path",
        "--video-path",
        dest="video_path",
        type=str,
        default=str(DEFAULT_VIDEO_PATH),
        help=(
            "Path to the benchmark video. Defaults to "
            f"{DEFAULT_VIDEO_PATH}."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--attn_load_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.")
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=32000)
    parser.add_argument("--max_num_frames", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument(
        "--baseline_attn_impl",
        "--baseline-attn-impl",
        dest="baseline_attn_impl",
        choices=("default", "eager"),
        default="default",
        help=(
            "Attention implementation for baseline runs. 'default' keeps the "
            "model's stock optimized attention path; 'eager' forces eager attention."
        ),
    )
    parser.add_argument("--skip_plot", action="store_true")


def resolve_prompt(args) -> str:
    if args.prompt_file and os.path.isfile(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    return args.prompt


def resolve_model_name(model_scale: Optional[str], model_name: Optional[str]) -> str:
    if model_scale:
        return MODEL_SCALE_TO_REPO[model_scale]
    if model_name:
        return model_name
    allowed = ", ".join(MODEL_SCALE_TO_REPO)
    raise ValueError(
        f"Provide --model-scale ({allowed}) or the legacy --model-name."
    )


def resolve_input_video_path(video_path: Optional[str]) -> str:
    raw_path = video_path or str(DEFAULT_VIDEO_PATH)
    expanded_path = Path(os.path.expanduser(os.path.expandvars(raw_path)))

    candidates: List[Path] = []
    if expanded_path.is_absolute():
        candidates.append(expanded_path)
    else:
        candidates.append(Path.cwd() / expanded_path)
        repo_relative_path = REPO_ROOT / expanded_path
        if repo_relative_path not in candidates:
            candidates.append(repo_relative_path)

    for candidate in candidates:
        resolved_path = candidate.resolve()
        if resolved_path.is_file():
            return str(resolved_path)

    checked_paths = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        "Video file not found. Checked: "
        f"{checked_paths}"
    )


def infer_video_token(processor) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        for attr in ("video_token", "vision_token", "image_token"):
            token = getattr(tokenizer, attr, None)
            if token:
                return token
        extra_tokens = getattr(tokenizer, "additional_special_tokens", None)
        if extra_tokens:
            for token in extra_tokens:
                low = token.lower()
                if "video" in low or "vision" in low:
                    return token
            for token in extra_tokens:
                if "image" in token.lower():
                    return token
    return "<video>"


def build_prompt_text(processor, prompt_text: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{infer_video_token(processor)}\n{prompt_text}"


def to_pil_frames(frames) -> List[Image.Image]:
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    pil_frames: List[Image.Image] = []
    for frame in frames:
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame_uint8).convert("RGB"))
    return pil_frames


def sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("Video has zero frames.")
    if num_frames <= 1:
        return [0]
    if total_frames == 1:
        return [0] * num_frames
    return (
        torch.linspace(0, total_frames - 1, steps=num_frames).round().long().tolist()
    )


def load_video_frames(
    video_path: str,
    num_frames: int,
    prefix_ratio: float,
) -> Tuple[List[Image.Image], float]:
    errors: List[str] = []
    for loader in (
        load_video_frames_decord,
        load_video_frames_torchvision,
        load_video_frames_opencv,
    ):
        try:
            return loader(video_path, num_frames, prefix_ratio)
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError(
        f"Unable to decode '{video_path}' with decord/torchvision/opencv. "
        + "; ".join(errors)
    )


def load_video_frames_decord(
    video_path: str,
    num_frames: int,
    prefix_ratio: float,
) -> Tuple[List[Image.Image], float]:
    from decord import VideoReader, cpu

    reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError("decord found zero frames.")
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        raise ValueError("decord returned non-positive FPS.")

    capped_frames = max(1, min(total_frames, int(math.ceil(total_frames * prefix_ratio))))
    indices = sample_frame_indices(capped_frames, num_frames)
    frames = reader.get_batch(indices).asnumpy()
    duration_seconds = capped_frames / fps
    return to_pil_frames(frames), duration_seconds


def load_video_frames_torchvision(
    video_path: str,
    num_frames: int,
    prefix_ratio: float,
) -> Tuple[List[Image.Image], float]:
    from torchvision.io import read_video

    video, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
    if video.ndim != 4 or video.shape[0] == 0:
        raise ValueError("torchvision read_video returned no frames.")

    total_frames = int(video.shape[0])
    fps = float(info.get("video_fps", 0.0))
    if fps <= 0:
        raise ValueError("torchvision returned non-positive FPS.")

    capped_frames = max(1, min(total_frames, int(math.ceil(total_frames * prefix_ratio))))
    indices = torch.tensor(sample_frame_indices(capped_frames, num_frames), dtype=torch.long)
    sampled = video.index_select(0, indices)
    frames = sampled.permute(0, 2, 3, 1).cpu().numpy()
    duration_seconds = capped_frames / fps
    return to_pil_frames(frames), duration_seconds


def load_video_frames_opencv(
    video_path: str,
    num_frames: int,
    prefix_ratio: float,
) -> Tuple[List[Image.Image], float]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0:
        raise ValueError("OpenCV found zero frames.")
    if fps <= 0:
        raise ValueError("OpenCV returned non-positive FPS.")

    capped_frames = max(1, min(total_frames, int(math.ceil(total_frames * prefix_ratio))))
    wanted = set(sample_frame_indices(capped_frames, num_frames))
    frames = []
    frame_idx = 0
    while frame_idx < capped_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        frame_idx += 1
    capture.release()

    if len(frames) != len(wanted):
        raise ValueError("OpenCV failed to gather all requested frames.")

    duration_seconds = capped_frames / fps
    return to_pil_frames(np.stack(frames)), duration_seconds


def encode_video_prompt(
    processor,
    frames: Sequence[Image.Image],
    prompt_text: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    full_text = build_prompt_text(processor, prompt_text)
    attempts = [
        {"text": full_text, "videos": [frames]},
        {"text": [full_text], "videos": [frames]},
        {"text": full_text, "videos": frames},
        {"text": [full_text], "videos": frames},
    ]
    last_error = None
    for kwargs in attempts:
        for with_length_controls in (True, False):
            try:
                processor_kwargs: Dict[str, Any] = {
                    "return_tensors": "pt",
                    **kwargs,
                }
                if with_length_controls:
                    processor_kwargs["truncation"] = True
                    processor_kwargs["max_length"] = max_length
                outputs = processor(**processor_kwargs)
                if outputs.get("input_ids", None) is not None:
                    return dict(outputs)
            except Exception as exc:
                last_error = exc
    raise RuntimeError("Unable to encode video/text prompt.") from last_error


def move_prefill_kwargs_to_model(
    model,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    return move_batch_to_device(
        dict(batch),
        device=model_device,
        model_dtype=model_dtype,
    )


def run_full_model_prefill(
    model,
    prefill_kwargs: Dict[str, Any],
    past_key_values=None,
):
    return model(
        **prefill_kwargs,
        past_key_values=past_key_values,
        use_cache=True,
    )

def build_prefill_chunk_plan(
    seq_len: int,
    requested_chunk_size: int,
) -> List[Tuple[int, int]]:
    if seq_len <= 0:
        raise ValueError("Chunked prefill expects a positive sequence length.")

    chunk_size = max(1, int(requested_chunk_size))
    chunks: List[Tuple[int, int]] = []
    start = 0
    while start < seq_len:
        end = min(seq_len, start + chunk_size)
        chunks.append((start, end))
        start = end

    return chunks


def get_context_targets(args) -> List[int]:
    if args.target_contexts:
        return sorted(int(x) for x in args.target_contexts if int(x) > 0)
    step = args.max_context / float(args.num_points)
    return [int(round(step * idx)) for idx in range(1, args.num_points + 1)]


def get_prefix_ratio(target_context: int, max_context: int) -> float:
    return min(1.0, max(1e-6, float(target_context) / float(max_context)))


def get_sequence_length_for_frames(
    processor,
    video_path: str,
    prompt_text: str,
    max_length: int,
    num_frames: int,
    prefix_ratio: float,
) -> Tuple[int, float]:
    frames, prefix_seconds = load_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        prefix_ratio=prefix_ratio,
    )
    outputs = encode_video_prompt(
        processor=processor,
        frames=frames,
        prompt_text=prompt_text,
        max_length=max_length,
    )
    input_ids = outputs["input_ids"]
    seq_len = int(input_ids.shape[-1])
    return seq_len, prefix_seconds


def find_sweep_points(
    processor,
    video_path: str,
    prompt_text: str,
    max_length: int,
    max_context: int,
    max_num_frames: int,
    targets: Sequence[int],
) -> List[SweepPoint]:
    sweep_points: List[SweepPoint] = []
    for target in targets:
        prefix_ratio = get_prefix_ratio(target, max_context)
        low = 1
        high = 1
        best_high: Optional[Tuple[int, int, float]] = None

        while high <= max_num_frames:
            seq_len, prefix_seconds = get_sequence_length_for_frames(
                processor=processor,
                video_path=video_path,
                prompt_text=prompt_text,
                max_length=max_length,
                num_frames=high,
                prefix_ratio=prefix_ratio,
            )
            if seq_len >= target or seq_len >= max_length:
                best_high = (high, seq_len, prefix_seconds)
                break
            low = high + 1
            high *= 2

        if best_high is None:
            final_seq_len, prefix_seconds = get_sequence_length_for_frames(
                processor=processor,
                video_path=video_path,
                prompt_text=prompt_text,
                max_length=max_length,
                num_frames=max_num_frames,
                prefix_ratio=prefix_ratio,
            )
            sweep_points.append(
                SweepPoint(
                    target_context=target,
                    actual_context=final_seq_len,
                    prefix_seconds=prefix_seconds,
                    prefix_ratio=prefix_ratio,
                    num_frames=max_num_frames,
                )
            )
            continue

        best_num_frames, best_seq_len, best_prefix_seconds = best_high
        left = max(1, low)
        right = best_num_frames
        while left <= right:
            mid = (left + right) // 2
            seq_len, prefix_seconds = get_sequence_length_for_frames(
                processor=processor,
                video_path=video_path,
                prompt_text=prompt_text,
                max_length=max_length,
                num_frames=mid,
                prefix_ratio=prefix_ratio,
            )
            if seq_len >= target:
                best_num_frames = mid
                best_seq_len = seq_len
                best_prefix_seconds = prefix_seconds
                right = mid - 1
            else:
                left = mid + 1

        sweep_points.append(
            SweepPoint(
                target_context=target,
                actual_context=best_seq_len,
                prefix_seconds=best_prefix_seconds,
                prefix_ratio=prefix_ratio,
                num_frames=best_num_frames,
            )
        )
    return sweep_points


def configure_model_for_mode(
    model_name: str,
    mode: str,
    attn_load_dir: Optional[str],
    threshold: float,
    sparsity: float,
    sink_size_override: Optional[int],
    recent_size_override: Optional[int],
    baseline_attn_impl: str = "default",
):
    print(f"Loading {mode} model: {model_name}")
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    if mode == "duo":
        load_kwargs["attn_implementation"] = "eager"
    elif baseline_attn_impl == "eager":
        load_kwargs["attn_implementation"] = "eager"

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        **load_kwargs,
    )
    model.eval()
    resolved_attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    print(f"[{mode}] Loaded attention implementation: {resolved_attn_impl}")

    resolved_sparsity = 0.0
    if mode == "duo":
        if attn_load_dir is None:
            raise ValueError("--attn_load_dir is required for DuoAttention mode.")
        full_attention_heads, sink_size, recent_size = load_attn_pattern(attn_load_dir)
        if sink_size_override is not None:
            sink_size = sink_size_override
        if recent_size_override is not None:
            recent_size = recent_size_override
        full_attention_heads, resolved_sparsity = sparsify_attention_heads(
            full_attention_heads,
            threshold,
            sparsity,
        )
        print(f"DuoAttention true sparsity: {resolved_sparsity}")
        enable_duo_attention_eval(model, full_attention_heads, sink_size, recent_size)

    device = torch.device("cuda")
    model = model.to(device)
    return model, resolved_sparsity


def run_benchmark_with_decode_steps(
    model,
    prefill_kwargs: Dict[str, Any],
    decode_tokens: int,
) -> Dict[str, Any]:
    print("\n--- Pre-filling benchmark ---")
    torch.cuda.reset_peak_memory_stats()

    def prefill_func():
        with torch.no_grad():
            _ = run_full_model_prefill(model, prefill_kwargs, past_key_values=None)

    ctx_latency, ctx_memory = bench_func(prefill_func, num_steps=10, num_warmup_steps=3)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = run_full_model_prefill(model, prefill_kwargs, past_key_values=None)
    prefill_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory usage in the pre-filling stage: {prefill_peak_memory:.2f} MB")

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    decode_steps = max(1, int(decode_tokens))

    print("\n--- Decoding benchmark ---")
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        current_past_key_values = past_key_values
        current_token = pred_token_idx
        for _ in range(decode_steps):
            outputs = model(
                input_ids=current_token,
                past_key_values=current_past_key_values,
                use_cache=True,
            )
            current_past_key_values = outputs.past_key_values
            current_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    end.record()
    torch.cuda.synchronize()

    decode_total_ms = float(start.elapsed_time(end))
    gen_latency = decode_total_ms / decode_steps
    gen_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(
        f"Autoregressive decode time for {decode_steps} tokens: "
        f"{decode_total_ms:.2f} ms"
    )
    print(f"Per-token decoding latency: {gen_latency:.2f} ms")
    print(f"Peak memory usage in the decoding stage: {gen_memory:.2f} MB")

    return {
        "ctx_latency": ctx_latency,
        "ctx_memory": ctx_memory,
        "gen_latency": gen_latency,
        "gen_memory": gen_memory,
        "prefill_total_ms": None,
    }


def benchmark_one_point(
    model,
    processor,
    video_path: str,
    prompt_text: str,
    max_length: int,
    decode_tokens: int,
    sweep_point: SweepPoint,
) -> Dict[str, Any]:
    frames, _ = load_video_frames(
        video_path=video_path,
        num_frames=sweep_point.num_frames,
        prefix_ratio=sweep_point.prefix_ratio,
    )
    batch = encode_video_prompt(
        processor=processor,
        frames=frames,
        prompt_text=prompt_text,
        max_length=max_length,
    )
    prefill_kwargs = move_prefill_kwargs_to_model(model, batch)

    result = run_benchmark_with_decode_steps(model, prefill_kwargs, decode_tokens)
    result["decode_tokens"] = int(decode_tokens)
    return result


def safe_benchmark_one_point(*args, **kwargs) -> Dict[str, Any]:
    try:
        result = benchmark_one_point(*args, **kwargs)
        result["oom"] = False
        return result
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"OOM while benchmarking point: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "ctx_latency": None,
            "ctx_memory": None,
            "gen_latency": None,
            "gen_memory": None,
            "prefill_total_ms": None,
            "decode_tokens": None,
            "oom": True,
            "error": str(exc),
        }


def run_prefill_only(
    model,
    inputs_embeds: torch.Tensor,
    prefill_chunk_size: int,
):
    if not torch.is_tensor(inputs_embeds) or inputs_embeds.ndim != 3:
        raise ValueError(
            "Prefill chunk sweep expects batched inputs_embeds with shape [B, L, H]."
        )

    seq_len = int(inputs_embeds.shape[1])
    chunk_plan = build_prefill_chunk_plan(seq_len, prefill_chunk_size)
    num_chunks = len(chunk_plan)

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    past_key_values = None
    start.record()
    pbar = tqdm(chunk_plan, leave=False)
    for s, e in pbar:
        chunk = inputs_embeds[:, s:e, :]
        with torch.no_grad():
            outputs = model.language_model(
                inputs_embeds=chunk,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        mem_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        pbar.set_description(
            f"Prefill chunk sweep ({e}/{seq_len}, Mem: {mem_gb:.2f} GB)"
        )

    end.record()
    torch.cuda.synchronize()

    prefill_total_ms = float(start.elapsed_time(end))
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    print(f"Peak prefill ctx_memory: {peak_memory_mb:.2f} MB (max allocated GPU memory)")
    return {
        "ctx_latency": prefill_total_ms / max(1, num_chunks),
        "ctx_memory": peak_memory_mb,
        "prefill_total_ms": prefill_total_ms,
        "num_chunks": num_chunks,
        "effective_first_chunk_size": int(chunk_plan[0][1] - chunk_plan[0][0]),
        "multimodal_prefix_context": None,
        "oom": False,
    }


def safe_run_prefill_only(
    model,
    inputs_embeds: torch.Tensor,
    prefill_chunk_size: int,
):
    try:
        return run_prefill_only(model, inputs_embeds, prefill_chunk_size)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"OOM while benchmarking chunk size {prefill_chunk_size}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "ctx_latency": None,
            "ctx_memory": None,
            "prefill_total_ms": None,
            "num_chunks": None,
            "effective_first_chunk_size": None,
            "multimodal_prefix_context": None,
            "oom": True,
            "error": str(exc),
        }


def write_csv(rows: Sequence[Dict[str, Any]], output_path: Path):
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_context_summary(rows: Sequence[Dict[str, Any]], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            label = (
                f"{row['mode']} | target={row['target_context']} | "
                f"actual={row['actual_context']} | frames={row['num_frames']}"
            )
            print(label, file=f)
            print(f"  prefill_latency_ms={row['ctx_latency']}", file=f)
            print(f"  prefill_memory_mb={row['ctx_memory']}", file=f)
            print(f"  generation_latency_ms={row['gen_latency']}", file=f)
            print(f"  generation_memory_mb={row['gen_memory']}", file=f)
            if row.get("attn_implementation"):
                print(f"  attn_implementation={row['attn_implementation']}", file=f)
            print(f"  prefix_seconds={row['prefix_seconds']:.2f} oom={row['oom']}", file=f)
            if row.get("error"):
                print(f"  error={row['error']}", file=f)
            print("", file=f)


def write_prefill_summary(rows: Sequence[Dict[str, Any]], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            label = (
                f"{row['mode']} | chunk={row['prefill_chunk_size']} | "
                f"context={row['actual_context']} | frames={row['num_frames']}"
            )
            print(label, file=f)
            print(f"  prefill_total_ms={row['prefill_total_ms']}", file=f)
            print(f"  prefill_latency_ms_per_chunk={row['ctx_latency']}", file=f)
            print(f"  prefill_memory_mb={row['ctx_memory']}", file=f)
            if row.get("attn_implementation"):
                print(f"  attn_implementation={row['attn_implementation']}", file=f)
            print(f"  num_chunks={row['num_chunks']} oom={row['oom']}", file=f)
            if row.get("effective_first_chunk_size") is not None:
                print(
                    f"  effective_first_chunk_size={row['effective_first_chunk_size']}",
                    file=f,
                )
            if row.get("multimodal_prefix_context") is not None:
                print(
                    f"  multimodal_prefix_context={row['multimodal_prefix_context']}",
                    file=f,
                )
            if row.get("error"):
                print(f"  error={row['error']}", file=f)
            print("", file=f)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_chunk_label(chunk_size: int) -> str:
    if chunk_size % 1000 == 0:
        return f"{chunk_size // 1000}K"
    return f"{chunk_size / 1000:.1f}K"

def infer_prefill_title(config: Optional[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "Prefill Chunk Sweep"
    sweep_point = config.get("sweep_point", {}) if isinstance(config, dict) else {}
    target_context = sweep_point.get("target_context") or rows[0].get("target_context")
    model_name = ""
    if isinstance(config, dict):
        model_name = str(config.get("model_name", "")).rstrip("/").split("/")[-1]
    context_label = (
        f"{int(target_context) // 1000}K Context" if target_context else "Fixed Context"
    )
    if model_name:
        return f"{model_name} | {context_label}"
    return f"Prefill Chunk Sweep | {context_label}"


def build_mode_rows(
    rows: Sequence[Dict[str, Any]],
    chunk_sizes: Sequence[int],
    mode: str,
) -> List[Optional[Dict[str, Any]]]:
    row_map: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("mode")) != mode:
            continue
        chunk_size = int(row["prefill_chunk_size"])
        row_map[chunk_size] = row
    return [row_map.get(chunk_size) for chunk_size in chunk_sizes]


def metric_values_with_oom_bars(
    mode_rows: Sequence[Optional[Dict[str, Any]]],
    metric: str,
    divisor: float,
):
    finite_vals = [
        float(row[metric]) / divisor
        for row in mode_rows
        if row is not None and row.get(metric) is not None
    ]
    oom_bar_height = max(finite_vals) * 1.05 if finite_vals else 1.0

    values = []
    for row in mode_rows:
        if row is None:
            values.append(np.nan)
        elif row.get(metric) is not None:
            values.append(float(row[metric]) / divisor)
        elif row.get("oom"):
            values.append(oom_bar_height)
        else:
            values.append(np.nan)
    return values


def annotate_oom(ax, bars, mode_rows):
    for bar, row in zip(bars, mode_rows):
        if row is not None and row.get("oom"):
            bar.set_facecolor("#d9d9d9")
            bar.set_hatch("//")
            bar.set_linewidth(1.5)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                "OOM",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )


def annotate_values(ax, bars, mode_rows, metric: str, divisor: float):
    heights = [bar.get_height() for bar in bars if np.isfinite(bar.get_height())]
    if not heights:
        return

    ymax = max(heights)
    offset = max(ymax * 0.025, 0.03)

    for bar, row in zip(bars, mode_rows):
        if row is None or row.get(metric) is None or row.get("oom"):
            continue

        value = float(row[metric]) / divisor
        label = f"{int(round(value))}" if divisor == 1.0 else f"{value:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_prefill_results(rows: Sequence[Dict[str, Any]], output_plot: Path, title: str):
    if not rows:
        raise ValueError("No rows to plot.")

    chunk_sizes = sorted({int(row["prefill_chunk_size"]) for row in rows})
    labels = [format_chunk_label(chunk_size) for chunk_size in chunk_sizes]
    target_context = rows[0].get("target_context")
    context_text = (
        f"{int(target_context) // 1000}K Context / Prefill Chunk Size"
        if target_context is not None
        else "Fixed Context / Prefill Chunk Size"
    )

    baseline_rows = build_mode_rows(rows, chunk_sizes, mode="baseline")
    duo_rows = build_mode_rows(rows, chunk_sizes, mode="duo")

    x = np.arange(len(chunk_sizes))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    plot_specs = (
        (
            axes[0],
            "prefill_total_ms",
            "Latency (s)",
            f"Prefill Total Time ({context_text})",
            1000.0,
        ),
        (
            axes[1],
            "ctx_memory",
            "Memory (GB)",
            f"Peak Prefill Memory ({context_text})",
            1024.0,
        ),
    )

    for ax, metric, ylabel, subtitle, divisor in plot_specs:
        baseline_vals = metric_values_with_oom_bars(baseline_rows, metric, divisor)
        duo_vals = metric_values_with_oom_bars(duo_rows, metric, divisor)

        baseline_bars = ax.bar(
            x - width / 2,
            baseline_vals,
            width,
            label="Baseline",
            color="#d9d9d9",
            edgecolor="#333333",
        )
        duo_bars = ax.bar(
            x + width / 2,
            duo_vals,
            width,
            label="DuoAttention",
            color="#b22234",
            edgecolor="#333333",
        )

        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        annotate_values(ax, baseline_bars, baseline_rows, metric, divisor)
        annotate_values(ax, duo_bars, duo_rows, metric, divisor)
        annotate_oom(ax, baseline_bars, baseline_rows)
        annotate_oom(ax, duo_bars, duo_rows)

    axes[0].set_xlabel("Prefill Chunk Size")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[1].set_xlabel("Prefill Chunk Size")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[0].legend(loc="upper left")
    fig.suptitle(title, fontsize=18, y=0.99)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_context_results(rows: Sequence[Dict[str, Any]], output_path: Path):
    labels = [f"{row['actual_context'] // 1000}K" for row in rows if row["mode"] == "baseline"]
    x = np.arange(len(labels))
    width = 0.36

    baseline_rows = [row for row in rows if row["mode"] == "baseline"]
    duo_rows = [row for row in rows if row["mode"] == "duo"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for ax, metric, ylabel, title in (
        (axes[0], "ctx_latency", "Latency (ms)", "Prefill Latency by Context Length"),
        (axes[1], "ctx_memory", "Memory (MB)", "Prefill Memory by Context Length"),
    ):
        baseline_vals = metric_values_with_oom_bars(baseline_rows, metric, 1.0)
        duo_vals = metric_values_with_oom_bars(duo_rows, metric, 1.0)
        bars1 = ax.bar(
            x - width / 2,
            baseline_vals,
            width,
            label="Baseline",
            color="#d9d9d9",
            edgecolor="#333333",
        )
        bars2 = ax.bar(
            x + width / 2,
            duo_vals,
            width,
            label="DuoAttention",
            color="#b22234",
            edgecolor="#333333",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        annotate_oom(ax, bars1, baseline_rows)
        annotate_oom(ax, bars2, duo_rows)

    axes[1].set_xlabel("Context Length")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_context_command(args):
    os.makedirs(args.output_dir, exist_ok=True)
    args.model_name = resolve_model_name(args.model_scale, args.model_name)
    print(f"Using model: {args.model_name}")
    args.video_path = resolve_input_video_path(args.video_path)
    print(f"Using video: {args.video_path}")
    seed_everything(args.seed)

    prompt_text = resolve_prompt(args)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    targets = get_context_targets(args)

    print("Calibrating frame counts for target contexts...")
    sweep_points = find_sweep_points(
        processor=processor,
        video_path=args.video_path,
        prompt_text=prompt_text,
        max_length=args.max_length,
        max_context=args.max_context,
        max_num_frames=args.max_num_frames,
        targets=targets,
    )

    for point in sweep_points:
        print(
            "Target context "
            f"{point.target_context} -> actual {point.actual_context} "
            f"using {point.num_frames} frames over {point.prefix_seconds:.2f}s "
            f"(ratio={point.prefix_ratio:.3f})"
        )

    rows: List[Dict[str, Any]] = []
    for mode in ("baseline", "duo"):
        if mode == "duo" and args.attn_load_dir is None:
            print("Skipping duo mode because --attn_load_dir was not provided.")
            continue

        model, mode_sparsity = configure_model_for_mode(
            model_name=args.model_name,
            mode=mode,
            attn_load_dir=args.attn_load_dir,
            threshold=args.threshold,
            sparsity=args.sparsity,
            sink_size_override=args.sink_size,
            recent_size_override=args.recent_size,
            baseline_attn_impl=args.baseline_attn_impl,
        )
        mode_attn_impl = getattr(model.config, "_attn_implementation", "unknown")

        try:
            for point in sweep_points:
                print(
                    f"\n[{mode}] Benchmarking target={point.target_context} "
                    f"actual={point.actual_context} frames={point.num_frames}"
                )
                result = safe_benchmark_one_point(
                    model=model,
                    processor=processor,
                    video_path=args.video_path,
                    prompt_text=prompt_text,
                    max_length=args.max_length,
                    decode_tokens=args.decode_tokens,
                    sweep_point=point,
                )
                rows.append(
                    {
                        "mode": mode,
                        "target_context": point.target_context,
                        "actual_context": point.actual_context,
                        "prefix_seconds": round(point.prefix_seconds, 3),
                        "prefix_ratio": round(point.prefix_ratio, 6),
                        "num_frames": point.num_frames,
                        "sparsity": mode_sparsity if mode == "duo" else 0.0,
                        "attn_implementation": mode_attn_impl,
                        **result,
                    }
                )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "context_sweep_results.json"
    csv_path = output_dir / "context_sweep_results.csv"
    plot_path = output_dir / "context_sweep_plot.png"
    summary_path = output_dir / "context_sweep_summary.txt"
    sweep_path = output_dir / "context_sweep_points.json"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    sweep_path.write_text(
        json.dumps([asdict(point) for point in sweep_points], indent=2),
        encoding="utf-8",
    )
    write_csv(rows, csv_path)
    write_context_summary(rows, summary_path)

    print(f"Saved sweep points to {sweep_path}")
    print(f"Saved JSON results to {json_path}")
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved summary to {summary_path}")

    if rows and not args.skip_plot:
        try:
            plot_context_results(rows, plot_path)
            print(f"Saved plot to {plot_path}")
        except Exception as exc:
            print(
                "Plot generation failed after results were saved. "
                f"Error: {exc}"
            )


def run_prefill_command(args):
    os.makedirs(args.output_dir, exist_ok=True)
    args.model_name = resolve_model_name(args.model_scale, args.model_name)
    print(f"Using model: {args.model_name}")
    args.video_path = resolve_input_video_path(args.video_path)
    print(f"Using video: {args.video_path}")

    seed_everything(args.seed)
    prompt_text = resolve_prompt(args)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Calibrating a fixed context near {args.target_context} tokens...")
    sweep_point = find_sweep_points(
        processor=processor,
        video_path=args.video_path,
        prompt_text=prompt_text,
        max_length=args.max_length,
        max_context=args.target_context,
        max_num_frames=args.max_num_frames,
        targets=[args.target_context],
    )[0]
    print(
        f"Locked context: target={sweep_point.target_context} "
        f"actual={sweep_point.actual_context} frames={sweep_point.num_frames} "
        f"prefix_seconds={sweep_point.prefix_seconds:.2f}"
    )

    frames, _ = load_video_frames(
        video_path=args.video_path,
        num_frames=sweep_point.num_frames,
        prefix_ratio=sweep_point.prefix_ratio,
    )
    batch = encode_video_prompt(
        processor=processor,
        frames=frames,
        prompt_text=prompt_text,
        max_length=args.max_length,
    )

    rows: List[Dict[str, Any]] = []
    for mode in ("baseline", "duo"):
        if mode == "duo" and args.attn_load_dir is None:
            print("Skipping duo mode because --attn_load_dir was not provided.")
            continue

        model, mode_sparsity = configure_model_for_mode(
            model_name=args.model_name,
            mode=mode,
            attn_load_dir=args.attn_load_dir,
            threshold=args.threshold,
            sparsity=args.sparsity,
            sink_size_override=args.sink_size,
            recent_size_override=args.recent_size,
            baseline_attn_impl=args.baseline_attn_impl,
        )
        mode_attn_impl = getattr(model.config, "_attn_implementation", "unknown")

        prefill_kwargs = None
        inputs_embeds = None
        try:
            prefill_kwargs = move_prefill_kwargs_to_model(model, batch)
            actual_seq_len = int(prefill_kwargs["input_ids"].shape[1])
            with torch.no_grad():
                inputs_embeds = build_llava_video_inputs_embeds(model, prefill_kwargs)
            prefill_kwargs = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{mode}] Fixed sequence length: {actual_seq_len}")

            for chunk_size in args.prefill_chunk_sizes:
                print(f"\n[{mode}] Benchmarking prefill_chunk_size={chunk_size}")
                result = safe_run_prefill_only(model, inputs_embeds, chunk_size)
                rows.append(
                    {
                        "mode": mode,
                        "target_context": sweep_point.target_context,
                        "actual_context": actual_seq_len,
                        "prefix_seconds": round(sweep_point.prefix_seconds, 3),
                        "prefix_ratio": round(sweep_point.prefix_ratio, 6),
                        "num_frames": sweep_point.num_frames,
                        "sparsity": mode_sparsity if mode == "duo" else 0.0,
                        "attn_implementation": mode_attn_impl,
                        "prefill_chunk_size": int(chunk_size),
                        "requested_prefill_chunk_size": int(chunk_size),
                        **result,
                    }
                )
        finally:
            if inputs_embeds is not None:
                del inputs_embeds
            if prefill_kwargs is not None:
                del prefill_kwargs
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "prefill_chunk_sweep_results.json"
    csv_path = output_dir / "prefill_chunk_sweep_results.csv"
    summary_path = output_dir / "prefill_chunk_sweep_summary.txt"
    config_path = output_dir / "prefill_chunk_sweep_config.json"
    plot_path = output_dir / "prefill_chunk_sweep_plot.png"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "sweep_point": asdict(sweep_point),
                "prefill_chunk_sizes": args.prefill_chunk_sizes,
                "video_path": args.video_path,
                "model_scale": args.model_scale,
                "model_name": args.model_name,
                "baseline_attn_impl": args.baseline_attn_impl,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(rows, csv_path)
    write_prefill_summary(rows, summary_path)

    print(f"Saved JSON results to {json_path}")
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved config to {config_path}")

    if rows and not args.skip_plot:
        try:
            title = infer_prefill_title(load_json(config_path), rows)
            plot_prefill_results(rows, plot_path, title)
            print(f"Saved plot to {plot_path}")
        except Exception as exc:
            print(
                "Plot generation failed after results were saved. "
                f"Error: {exc}"
            )


def run_context_plot_command(args):
    input_json = Path(args.input_json)
    rows = load_json(input_json)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No rows found in {input_json}")
    output_plot = (
        Path(args.output_plot)
        if args.output_plot
        else input_json.with_name("context_sweep_plot.png")
    )
    plot_context_results(rows, output_plot)
    print(f"Loaded {len(rows)} rows from {input_json}")
    print(f"Saved plot to {output_plot}")


def run_prefill_plot_command(args):
    input_json = Path(args.input_json)
    rows = load_json(input_json)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of rows in {input_json}")

    config = None
    if args.config_json:
        config_path = Path(args.config_json)
    else:
        config_path = input_json.with_name("prefill_chunk_sweep_config.json")
    if config_path.exists():
        loaded_config = load_json(config_path)
        if isinstance(loaded_config, dict):
            config = loaded_config

    output_plot = (
        Path(args.output_plot)
        if args.output_plot
        else input_json.with_name("prefill_chunk_sweep_plot.png")
    )
    title = args.title or infer_prefill_title(config, rows)
    plot_prefill_results(rows, output_plot, title)
    print(f"Loaded {len(rows)} rows from {input_json}")
    print(f"Saved plot to {output_plot}")


def main():
    args = parse_args()
    if args.command == "context":
        run_context_command(args)
    elif args.command == "prefill":
        run_prefill_command(args)
    elif args.command == "context-plot":
        run_context_plot_command(args)
    elif args.command == "prefill-plot":
        run_prefill_plot_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
