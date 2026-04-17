#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from duo_attn.eval.efficiency.context_eval_llava import (
    annotate_oom_points,
    annotate_points,
    build_mode_rows,
    format_context_label,
    metric_values_with_oom_markers,
)


DEFAULT_0P5B_JSON = Path("outputs/benchmarking/llava-ov-0p5b/context_sweep_results.json")
DEFAULT_7B_JSON = Path("outputs/benchmarking/llava-ov-7b/context_sweep_results.json")
DEFAULT_OUTPUT_PLOT = Path("outputs/benchmarking/llava_context_memory_prefill_grid.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 2x2 grid of LLaVA-OneVision context sweep memory and "
            "prefill latency results for the 0.5B and 7B models."
        )
    )
    parser.add_argument("--input_0p5b_json", type=Path, default=DEFAULT_0P5B_JSON)
    parser.add_argument("--input_7b_json", type=Path, default=DEFAULT_7B_JSON)
    parser.add_argument("--output_plot", type=Path, default=DEFAULT_OUTPUT_PLOT)
    return parser.parse_args()


def load_rows(path: Path) -> Sequence[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def plot_metric_panel(
    ax,
    rows: Sequence[Dict[str, Any]],
    metric: str,
    ylabel: str,
    divisor: float,
    show_legend: bool = False,
):
    context_points = sorted({int(row["actual_context"]) for row in rows})
    labels = [format_context_label(context_len) for context_len in context_points]
    baseline_rows = build_mode_rows(rows, context_points, mode="baseline")
    duo_rows = build_mode_rows(rows, context_points, mode="duo")

    x = np.arange(len(labels))
    width = 0.36

    baseline_vals, baseline_oom, _ = metric_values_with_oom_markers(
        baseline_rows, metric, divisor
    )
    duo_vals, duo_oom, _ = metric_values_with_oom_markers(
        duo_rows, metric, divisor
    )

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
        label="MMDA",
        color="#b22234",
        edgecolor="#333333",
    )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if divisor != 1.0:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    annotate_points(ax, x - width / 2, baseline_vals, baseline_oom, divisor)
    annotate_points(ax, x + width / 2, duo_vals, duo_oom, divisor)
    annotate_oom_points(ax, baseline_bars, baseline_vals, baseline_oom)
    annotate_oom_points(ax, duo_bars, duo_vals, duo_oom)

    if show_legend:
        ax.legend(loc="upper left")


def main():
    args = parse_args()
    rows_0p5b = load_rows(args.input_0p5b_json)
    rows_7b = load_rows(args.input_7b_json)

    fig, axes = plt.subplots(2, 2, figsize=(24, 7))

    plot_metric_panel(
        axes[0, 0],
        rows_0p5b,
        metric="ctx_memory",
        ylabel="Memory (GB)",
        divisor=1024.0,
        show_legend=True,
    )
    plot_metric_panel(
        axes[0, 1],
        rows_7b,
        metric="ctx_memory",
        ylabel="Memory (GB)",
        divisor=1024.0,
    )
    plot_metric_panel(
        axes[1, 0],
        rows_0p5b,
        metric="ctx_latency",
        ylabel="Prefill Latency (ms)",
        divisor=1.0,
    )
    plot_metric_panel(
        axes[1, 1],
        rows_7b,
        metric="ctx_latency",
        ylabel="Prefill Latency (ms)",
        divisor=1.0,
    )

    axes[0, 0].set_title("Llava-onevision 0.5B")
    axes[0, 1].set_title("Llava-onevision 7B")
    axes[1, 0].set_xlabel("Context Length")
    axes[1, 1].set_xlabel("Context Length")

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {args.output_plot}")


if __name__ == "__main__":
    main()
