import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.eval.efficiency.benchmark_dynamic_llava import (
    move_batch_to_device,
)
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
from duo_attn.train import build_llava_video_inputs_embeds
from duo_attn.utils import (
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)
from duo_attn.eval.efficiency.utils import bench_func


@dataclass
class SweepPoint:
    target_context: int
    actual_context: int
    prefix_seconds: float
    prefix_ratio: float
    num_frames: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep LLaVA-OneVision context lengths on one video and compare "
            "baseline vs DuoAttention."
        )
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--attn_load_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.")
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=32000)
    parser.add_argument(
        "--max_context",
        type=int,
        default=32000,
        help="Largest target context length in tokens.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=5,
        help="Number of evenly spaced context targets up to max_context.",
    )
    parser.add_argument(
        "--target_contexts",
        type=int,
        nargs="+",
        default=None,
        help="Explicit target context lengths. Overrides --num_points.",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=512,
        help="Upper bound for frame-count search.",
    )
    parser.add_argument(
        "--decode_tokens",
        type=int,
        default=100,
        help="Number of decode steps to benchmark after prefill.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip plot generation and only write JSON/CSV/summary outputs.",
    )
    parser.add_argument(
        "--plot_only_json",
        type=str,
        default=None,
        help="Load existing JSON results and only regenerate the plot.",
    )
    return parser.parse_args()


def resolve_prompt(args) -> str:
    if args.prompt_file and os.path.isfile(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    return args.prompt


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
):
    print(f"Loading {mode} model: {model_name}")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

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
    else:
        enable_tuple_kv_cache(model)

    device = torch.device("cuda")
    model = model.to(device)
    return model, resolved_sparsity


def run_benchmark_with_decode_steps(
    model,
    inputs_embeds: torch.Tensor,
    decode_tokens: int,
) -> Dict[str, Any]:
    print("\n--- Pre-filling benchmark ---")
    torch.cuda.reset_peak_memory_stats()

    def prefill_func():
        with torch.no_grad():
            _ = model.language_model(
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                use_cache=True,
            )

    ctx_latency, ctx_memory = bench_func(prefill_func, num_steps=10, num_warmup_steps=3)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
        )
    prefill_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory usage in the pre-filling stage: {prefill_peak_memory:.2f} MB")

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    print("\n--- Decoding benchmark ---")

    def decode_func():
        with torch.no_grad():
            _ = model.language_model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

    gen_latency, gen_memory = bench_func(
        decode_func,
        num_steps=max(1, int(decode_tokens)),
        num_warmup_steps=min(10, max(1, int(decode_tokens))),
    )

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
    batch = move_batch_to_device(
        batch,
        device=torch.device("cuda"),
        model_dtype=torch.bfloat16,
    )
    with torch.no_grad():
        inputs_embeds = build_llava_video_inputs_embeds(model, batch)

    result = run_benchmark_with_decode_steps(model, inputs_embeds, decode_tokens)
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


def plot_results(rows: Sequence[Dict[str, Any]], output_path: str):
    labels = [f"{row['actual_context'] // 1000}K" for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    baseline_rows = [row for row in rows if row["mode"] == "baseline"]
    duo_rows = [row for row in rows if row["mode"] == "duo"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for ax, metric, ylabel, title in (
        (
            axes[0],
            "ctx_latency",
            "Latency (ms)",
            "Prefill Latency by Context Length",
        ),
        (
            axes[1],
            "ctx_memory",
            "Memory (MB)",
            "Prefill Memory by Context Length",
        ),
    ):
        baseline_vals = [
            np.nan if row.get(metric) is None else float(row[metric]) for row in baseline_rows
        ]
        duo_vals = [np.nan if row.get(metric) is None else float(row[metric]) for row in duo_rows]
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

        for bars, mode_rows in ((bars1, baseline_rows), (bars2, duo_rows)):
            for bar, row in zip(bars, mode_rows):
                if row.get("oom"):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        ax.get_ylim()[1] * 0.05,
                        "OOM",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=10,
                        fontweight="bold",
                    )

    axes[1].set_xlabel("Context Length")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: Sequence[Dict[str, Any]], output_path: str):
    import csv

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(rows: Sequence[Dict[str, Any]], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            label = (
                f"{row['mode']} | target={row['target_context']} | "
                f"actual={row['actual_context']} | frames={row['num_frames']}"
            )
            print(label, file=f)
            print(
                f"  prefill_latency_ms={row['ctx_latency']}",
                file=f,
            )
            print(
                f"  prefill_memory_mb={row['ctx_memory']}",
                file=f,
            )
            print(
                f"  generation_latency_ms={row['gen_latency']}",
                file=f,
            )
            print(
                f"  generation_memory_mb={row['gen_memory']}",
                file=f,
            )
            print(
                f"  prefix_seconds={row['prefix_seconds']:.2f} oom={row['oom']}",
                file=f,
            )
            print("", file=f)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "context_sweep_results.json")
    csv_path = os.path.join(args.output_dir, "context_sweep_results.csv")
    plot_path = os.path.join(args.output_dir, "context_sweep_plot.png")
    summary_path = os.path.join(args.output_dir, "context_sweep_summary.txt")
    sweep_path = os.path.join(args.output_dir, "context_sweep_points.json")
    if args.plot_only_json:
        with open(args.plot_only_json, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not rows:
            raise ValueError(f"No rows found in {args.plot_only_json}")
        plot_results(rows, plot_path)
        print(f"Saved plot to {plot_path}")
        return

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
        )

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
                row = {
                    "mode": mode,
                    "target_context": point.target_context,
                    "actual_context": point.actual_context,
                    "prefix_seconds": round(point.prefix_seconds, 3),
                    "prefix_ratio": round(point.prefix_ratio, 6),
                    "num_frames": point.num_frames,
                    "sparsity": mode_sparsity if mode == "duo" else 0.0,
                    **result,
                }
                rows.append(row)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump([asdict(point) for point in sweep_points], f, indent=2)

    write_csv(rows, csv_path)
    write_summary(rows, summary_path)

    print(f"Saved sweep points to {sweep_path}")
    print(f"Saved JSON results to {json_path}")
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved summary to {summary_path}")

    if rows and not args.skip_plot:
        try:
            plot_results(rows, plot_path)
            print(f"Saved plot to {plot_path}")
        except Exception as exc:
            print(
                "Plot generation failed after results were saved. "
                f"Error: {exc}"
            )
            print(
                "You can regenerate the plot later with "
                f"--plot_only_json {json_path}"
            )


if __name__ == "__main__":
    main()


# python -u duo_attn/eval/efficiency/benchmark_context_sweep_llava.py \
#   --model_name /root/streaming-vqa/models/llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
#   --video_path /some/real/video.mp4 \
#   --output_dir /root/streaming-vqa/untracked/context_sweep_13min \
#   --plot_only_json /root/streaming-vqa/untracked/context_sweep_13min/context_sweep_results.json


# python -u duo_attn/eval/efficiency/benchmark_context_sweep_llava.py  \
#      --model_name /root/streaming-vqa/models/llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
#      --video_path /root/streaming-vqa/data/sample.mp4 \
#      --attn_load_dir /root/streaming-vqa/outputs/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1 \
#      --prompt_file /root/streaming-vqa/long_prompt.txt \
#      --max_length 32000 \
#      --max_context 32000 \
#      --output_dir /root/streaming-vqa/untracked/context_sweep_13min \
#      --target_contexts 4000 8000 12000 16000 20000 24000 28000 32000 \
#      --skip_plot