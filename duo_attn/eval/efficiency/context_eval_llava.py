#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoProcessor

from duo_attn.eval.efficiency.prefill_eval_llava import (
    configure_model_for_mode,
    find_sweep_points,
    get_context_targets,
    load_json,
    resolve_prompt,
    safe_benchmark_one_point,
    write_context_summary,
    write_csv,
)
from duo_attn.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone LLaVA context-length sweep benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a context-length sweep.")
    run_parser.add_argument("--model_name", type=str, required=True)
    run_parser.add_argument("--video_path", type=str, required=True)
    run_parser.add_argument("--output_dir", type=str, required=True)
    run_parser.add_argument("--attn_load_dir", type=str, default=None)
    run_parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this video in detail.",
    )
    run_parser.add_argument("--prompt_file", type=str, default=None)
    run_parser.add_argument("--max_length", type=int, default=32000)
    run_parser.add_argument("--max_context", type=int, default=32000)
    run_parser.add_argument("--max_num_frames", type=int, default=512)
    run_parser.add_argument("--num_points", type=int, default=5)
    run_parser.add_argument("--target_contexts", type=int, nargs="+", default=None)
    run_parser.add_argument("--decode_tokens", type=int, default=100)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--threshold", type=float, default=0.5)
    run_parser.add_argument("--sparsity", type=float, default=0.5)
    run_parser.add_argument("--sink_size", type=int, default=None)
    run_parser.add_argument("--recent_size", type=int, default=None)
    run_parser.add_argument("--skip_plot", action="store_true")

    plot_parser = subparsers.add_parser(
        "plot", help="Regenerate a context sweep plot from saved JSON."
    )
    plot_parser.add_argument("--input_json", type=str, required=True)
    plot_parser.add_argument("--output_plot", type=str, default=None)
    plot_parser.add_argument("--model_name", type=str, default=None)
    plot_parser.add_argument("--title", type=str, default=None)

    return parser.parse_args()


def format_context_label(context_len: int) -> str:
    if context_len >= 1_000_000:
        if context_len % 1_000_000 == 0:
            return f"{context_len // 1_000_000}M"
        return f"{context_len / 1_000_000:.1f}M"
    if context_len >= 1000:
        if context_len % 1000 == 0:
            return f"{context_len // 1000}K"
        return f"{context_len / 1000:.1f}K"
    return str(context_len)


def infer_model_heading(model_name: Optional[str]) -> str:
    if not model_name:
        return "LLaVA-OneVision Context Sweep"
    short_name = str(model_name).rstrip("/").split("/")[-1]
    return f"{short_name} Context Sweep"


def build_mode_rows(
    rows: Sequence[Dict[str, Any]],
    context_points: Sequence[int],
    mode: str,
) -> List[Optional[Dict[str, Any]]]:
    row_map: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("mode")) != mode:
            continue
        context_len = int(row["actual_context"])
        row_map[context_len] = row
    return [row_map.get(context_len) for context_len in context_points]


def metric_values_with_oom_bars(
    mode_rows: Sequence[Optional[Dict[str, Any]]],
    metric: str,
    divisor: float,
) -> List[float]:
    finite_vals = [
        float(row[metric]) / divisor
        for row in mode_rows
        if row is not None and row.get(metric) is not None
    ]
    oom_bar_height = max(finite_vals) * 1.05 if finite_vals else 1.0

    values: List[float] = []
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


def annotate_values(ax, bars, mode_rows, metric: str, divisor: float) -> None:
    heights = [bar.get_height() for bar in bars if np.isfinite(bar.get_height())]
    if not heights:
        return

    ymax = max(heights)
    offset = max(ymax * 0.025, 0.8 if divisor == 1.0 else 0.03)

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


def annotate_oom(ax, bars, mode_rows) -> None:
    for bar, row in zip(bars, mode_rows):
        if row is None or not row.get("oom"):
            continue
        bar.set_facecolor("#f2f2f2")
        bar.set_hatch("//")
        bar.set_linestyle("--")
        bar.set_linewidth(1.5)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.5,
            "OOM",
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
            fontweight="bold",
        )


def plot_context_results(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    if not rows:
        raise ValueError("No rows to plot.")

    context_points = sorted({int(row["actual_context"]) for row in rows})
    labels = [format_context_label(context_len) for context_len in context_points]
    baseline_rows = build_mode_rows(rows, context_points, mode="baseline")
    duo_rows = build_mode_rows(rows, context_points, mode="duo")

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    plot_specs = (
        (
            axes[0],
            "ctx_latency",
            "Latency (ms)",
            "Context Length",
            "Prefill Latency by Context Length",
            1.0,
        ),
        (
            axes[1],
            "ctx_memory",
            "Memory (GB)",
            "Context Length",
            "Peak Prefill Memory by Context Length",
            1024.0,
        ),
    )

    for ax, metric, ylabel, xlabel, subtitle, divisor in plot_specs:
        baseline_vals = metric_values_with_oom_bars(baseline_rows, metric, divisor)
        duo_vals = metric_values_with_oom_bars(duo_rows, metric, divisor)

        baseline_bars = ax.bar(
            x - width / 2,
            baseline_vals,
            width,
            label="Full Attention",
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
        ax.set_xlabel(xlabel)
        ax.set_title(subtitle)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        annotate_values(ax, baseline_bars, baseline_rows, metric, divisor)
        annotate_values(ax, duo_bars, duo_rows, metric, divisor)
        annotate_oom(ax, baseline_bars, baseline_rows)
        annotate_oom(ax, duo_bars, duo_rows)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[0].legend(loc="upper left")
    fig.suptitle(title, fontsize=18, y=0.99)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_context_command(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

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
                rows.append(
                    {
                        "mode": mode,
                        "target_context": point.target_context,
                        "actual_context": point.actual_context,
                        "prefix_seconds": round(point.prefix_seconds, 3),
                        "prefix_ratio": round(point.prefix_ratio, 6),
                        "num_frames": point.num_frames,
                        "sparsity": mode_sparsity if mode == "duo" else 0.0,
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
            plot_context_results(rows, plot_path, infer_model_heading(args.model_name))
            print(f"Saved plot to {plot_path}")
        except Exception as exc:
            print(
                "Plot generation failed after results were saved. "
                f"Error: {exc}"
            )


def run_plot_command(args):
    input_json = Path(args.input_json)
    rows = load_json(input_json)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No rows found in {input_json}")

    output_plot = (
        Path(args.output_plot)
        if args.output_plot
        else input_json.with_name("context_sweep_plot.png")
    )
    title = args.title or infer_model_heading(args.model_name)
    plot_context_results(rows, output_plot, title)
    print(f"Loaded {len(rows)} rows from {input_json}")
    print(f"Saved plot to {output_plot}")


def main():
    args = parse_args()
    if args.command == "run":
        run_context_command(args)
    elif args.command == "plot":
        run_plot_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
