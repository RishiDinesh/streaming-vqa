import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from duo_attn.eval.efficiency.benchmark_context_sweep_llava import (
    SweepPoint,
    configure_model_for_mode,
    encode_video_prompt,
    find_sweep_points,
    load_video_frames,
    resolve_prompt,
)
from duo_attn.eval.efficiency.benchmark_dynamic_llava import move_batch_to_device
from duo_attn.train import build_llava_video_inputs_embeds
from duo_attn.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fix LLaVA-OneVision context length and sweep prefill chunk size for "
            "baseline vs DuoAttention."
        )
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--attn_load_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.")
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument(
        "--target_context",
        type=int,
        default=32000,
        help="Desired fixed context length in tokens.",
    )
    parser.add_argument("--max_length", type=int, default=32000)
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=512,
        help="Upper bound for frame-count search during context calibration.",
    )
    parser.add_argument(
        "--prefill_chunk_sizes",
        type=int,
        nargs="+",
        default=[4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000],
        help="Prefill chunk sizes to benchmark.",
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
    return parser.parse_args()


def run_prefill_only(model, inputs_embeds: torch.Tensor, prefill_chunk_size: int):
    seq_len = int(inputs_embeds.shape[1])
    chunk_size = max(1, int(prefill_chunk_size))
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    past_key_values = None
    start.record()
    pbar = tqdm(range(num_chunks), leave=False)
    for i in pbar:
        s = i * chunk_size
        e = min((i + 1) * chunk_size, seq_len)
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
        "oom": False,
    }


def safe_run_prefill_only(model, inputs_embeds: torch.Tensor, prefill_chunk_size: int):
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
            "oom": True,
            "error": str(exc),
        }


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
                f"{row['mode']} | chunk={row['prefill_chunk_size']} | "
                f"context={row['actual_context']} | frames={row['num_frames']}"
            )
            print(label, file=f)
            print(f"  prefill_total_ms={row['prefill_total_ms']}", file=f)
            print(f"  prefill_latency_ms_per_chunk={row['ctx_latency']}", file=f)
            print(f"  prefill_memory_mb={row['ctx_memory']}", file=f)
            print(f"  num_chunks={row['num_chunks']} oom={row['oom']}", file=f)
            print("", file=f)


def plot_results(rows: Sequence[Dict[str, Any]], output_path: str):
    import matplotlib.pyplot as plt
    import numpy as np

    chunk_sizes = sorted({int(row["prefill_chunk_size"]) for row in rows})
    labels = [f"{chunk // 1000}K" for chunk in chunk_sizes]
    x = np.arange(len(chunk_sizes))
    width = 0.36

    row_map = {
        (str(row["mode"]), int(row["prefill_chunk_size"])): row
        for row in rows
    }
    baseline_rows = [row_map.get(("baseline", chunk)) for chunk in chunk_sizes]
    duo_rows = [row_map.get(("duo", chunk)) for chunk in chunk_sizes]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    plot_specs = (
        (
            axes[0],
            "prefill_total_ms",
            "Latency (s)",
            "Prefill Total Time by Prefill Chunk Size",
            1000.0,
        ),
        (
            axes[1],
            "ctx_memory",
            "Memory (GB)",
            "Prefill Peak Memory by Prefill Chunk Size",
            1024.0,
        ),
    )

    for ax, metric, ylabel, title, divisor in plot_specs:
        baseline_vals = [
            np.nan
            if row is None or row.get(metric) is None
            else float(row[metric]) / divisor
            for row in baseline_rows
        ]
        duo_vals = [
            np.nan
            if row is None or row.get(metric) is None
            else float(row[metric]) / divisor
            for row in duo_rows
        ]

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
            ymin, ymax = ax.get_ylim()
            oom_y = ymax * 0.05 if ymax > 0 else 0.5
            for bar, row in zip(bars, mode_rows):
                if row is not None and row.get("oom"):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        oom_y,
                        "OOM",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=10,
                        fontweight="bold",
                    )

    axes[1].set_xlabel("Prefill Chunk Size")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

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
        )

        try:
            batch_on_device = move_batch_to_device(
                dict(batch),
                device=torch.device("cuda"),
                model_dtype=torch.bfloat16,
            )
            with torch.no_grad():
                inputs_embeds = build_llava_video_inputs_embeds(model, batch_on_device)

            actual_seq_len = int(inputs_embeds.shape[1])
            print(f"[{mode}] Fixed sequence length: {actual_seq_len}")

            for chunk_size in args.prefill_chunk_sizes:
                print(f"\n[{mode}] Benchmarking prefill_chunk_size={chunk_size}")
                result = safe_run_prefill_only(model, inputs_embeds, chunk_size)
                row = {
                    "mode": mode,
                    "target_context": sweep_point.target_context,
                    "actual_context": actual_seq_len,
                    "prefix_seconds": round(sweep_point.prefix_seconds, 3),
                    "prefix_ratio": round(sweep_point.prefix_ratio, 6),
                    "num_frames": sweep_point.num_frames,
                    "sparsity": mode_sparsity if mode == "duo" else 0.0,
                    "prefill_chunk_size": int(chunk_size),
                    **result,
                }
                rows.append(row)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    json_path = os.path.join(args.output_dir, "prefill_chunk_sweep_results.json")
    csv_path = os.path.join(args.output_dir, "prefill_chunk_sweep_results.csv")
    summary_path = os.path.join(args.output_dir, "prefill_chunk_sweep_summary.txt")
    plot_path = os.path.join(args.output_dir, "prefill_chunk_sweep_plot.png")
    config_path = os.path.join(args.output_dir, "prefill_chunk_sweep_config.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sweep_point": asdict(sweep_point),
                "prefill_chunk_sizes": args.prefill_chunk_sizes,
                "video_path": args.video_path,
                "model_name": args.model_name,
            },
            f,
            indent=2,
        )

    write_csv(rows, csv_path)
    write_summary(rows, summary_path)

    print(f"Saved JSON results to {json_path}")
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved config to {config_path}")

    if rows and not args.skip_plot:
        try:
            plot_results(rows, plot_path)
            print(f"Saved plot to {plot_path}")
        except Exception as exc:
            print(
                "Plot generation failed after results were saved. "
                f"Error: {exc}"
            )


if __name__ == "__main__":
    main()
