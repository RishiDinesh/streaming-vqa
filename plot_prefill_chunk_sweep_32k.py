#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_JSON = Path(
    "/root/streaming-vqa/untracked/prefill_chunk_sweep_32k/prefill_chunk_sweep_results.json"
)
DEFAULT_CONFIG_JSON = Path(
    "/root/streaming-vqa/untracked/prefill_chunk_sweep_32k/prefill_chunk_sweep_config.json"
)
DEFAULT_OUTPUT_PLOT = Path(
    "/root/streaming-vqa/untracked/prefill_chunk_sweep_32k/prefill_chunk_sweep_plot.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the saved 32K prefill chunk sweep benchmark results."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        default=DEFAULT_RESULTS_JSON,
        help="Path to the saved prefill chunk sweep JSON results.",
    )
    parser.add_argument(
        "--config_json",
        type=Path,
        default=DEFAULT_CONFIG_JSON,
        help="Optional config JSON used to infer a title/context label.",
    )
    parser.add_argument(
        "--output_plot",
        type=Path,
        default=DEFAULT_OUTPUT_PLOT,
        help="Where to save the PNG plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title override.",
    )
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_chunk_label(chunk_size: int) -> str:
    if chunk_size % 1000 == 0:
        return f"{chunk_size // 1000}K"
    return f"{chunk_size / 1000:.1f}K"


def infer_title(config: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "Prefill Chunk Sweep"

    sweep_point = config.get("sweep_point", {}) if isinstance(config, dict) else {}
    target_context = sweep_point.get("target_context") or rows[0].get("target_context")
    model_name = ""
    if isinstance(config, dict):
        model_name = str(config.get("model_name", "")).rstrip("/").split("/")[-1]

    context_label = f"{int(target_context) // 1000}K Context" if target_context else "Fixed Context"
    if model_name:
        return f"{model_name} | {context_label}"
    return f"Prefill Chunk Sweep | {context_label}"


def build_mode_rows(rows: Sequence[Dict[str, Any]], chunk_sizes: Sequence[int], mode: str):
    row_map: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("mode")) != mode:
            continue
        chunk_size = int(row["prefill_chunk_size"])
        if chunk_size in row_map:
            raise ValueError(
                f"Duplicate row for mode={mode!r}, prefill_chunk_size={chunk_size}."
            )
        row_map[chunk_size] = row
    return [row_map.get(chunk_size) for chunk_size in chunk_sizes]


def metric_values(mode_rows: Sequence[Dict[str, Any]], metric: str, divisor: float):
    values = []
    for row in mode_rows:
        if row is None or row.get(metric) is None:
            values.append(np.nan)
            continue
        values.append(float(row[metric]) / divisor)
    return values


def annotate_oom(ax, bars, mode_rows):
    _, ymax = ax.get_ylim()
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


def plot_results(rows: Sequence[Dict[str, Any]], output_plot: Path, title: str):
    if not rows:
        raise ValueError("No rows to plot.")

    chunk_sizes = sorted({int(row["prefill_chunk_size"]) for row in rows})
    labels = [format_chunk_label(chunk_size) for chunk_size in chunk_sizes]
    target_context = rows[0].get("target_context")
    if target_context is not None:
        context_text = f"{int(target_context) // 1000}K Context / Prefill Chunk Size"
    else:
        context_text = "Fixed Context / Prefill Chunk Size"
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
            f"Peak Prefill ctx_memory ({context_text})",
            1024.0,
        ),
    )

    for ax, metric, ylabel, subtitle, divisor in plot_specs:
        baseline_vals = metric_values(baseline_rows, metric, divisor)
        duo_vals = metric_values(duo_rows, metric, divisor)

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


def main():
    args = parse_args()
    rows = load_json(args.input_json)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of rows in {args.input_json}")

    config: Dict[str, Any] = {}
    if args.config_json.exists():
        loaded_config = load_json(args.config_json)
        if isinstance(loaded_config, dict):
            config = loaded_config

    title = args.title or infer_title(config, rows)
    plot_results(rows, args.output_plot, title)
    print(f"Loaded {len(rows)} rows from {args.input_json}")
    print(f"Saved plot to {args.output_plot}")


if __name__ == "__main__":
    main()
