#!/usr/bin/env python3
"""Plot evaluation accuracy vs. KV cache budget from lmms-eval result JSON files."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError as exc:
    if exc.name == "matplotlib":
        raise SystemExit(
            "matplotlib is required to render the evaluation plot. "
            "Install it in the active environment and rerun this script."
        ) from exc
    raise


SPARSITY_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
KV_BUDGET_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
X_AXIS_PADDING = 0.04
MODEL_ORDER = ["0.5B", "7B"]
MODEL_COLORS = {
    "0.5B": "#ff7f0e",
    "7B": "#1f77b4",
}


@dataclass(frozen=True)
class DatasetSpec:
    task_name: str
    label: str
    metric_key: str
    scale: float
    results_subdir: str


@dataclass(frozen=True)
class ResultPoint:
    dataset: str
    model: str
    sparsity: float
    accuracy: float
    timestamp: str
    source_path: Path


DATASET_SPECS = {
    "videomme": DatasetSpec(
        task_name="videomme",
        label="Video-MME",
        metric_key="videomme_perception_score,none",
        scale=1.0,
        results_subdir="videomme",
    ),
    "mlvu_dev": DatasetSpec(
        task_name="mlvu_dev",
        label="MLVU Dev",
        metric_key="mlvu_percetion_score,none",
        scale=1.0,
        results_subdir="mlvu",
    ),
    "egoschema_subset": DatasetSpec(
        task_name="egoschema_subset",
        label="EgoSchema",
        metric_key="score,none",
        scale=100.0,
        results_subdir="egoschema",
    ),
    "longvideobench_val_v": DatasetSpec(
        task_name="longvideobench_val_v",
        label="LongVideoBench-V",
        metric_key="lvb_acc,none",
        scale=100.0,
        results_subdir="longvideobench-v",
    ),
    "longvideobench_val_i": DatasetSpec(
        task_name="longvideobench_val_i",
        label="LongVideoBench-I",
        metric_key="lvb_acc,none",
        scale=100.0,
        results_subdir="longvideobench-i",
    ),
}

DEFAULT_DATASETS = [
    "videomme",
    "mlvu_dev",
    "egoschema_subset",
    "longvideobench_val_v",
]
DEFAULT_RESULTS_DIR = Path("outputs/evaluations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read lmms-eval result JSON files and plot accuracy vs. KV cache budget "
            "in a paper-style multi-panel matplotlib figure."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Evaluation root directory. Expected layout: "
            "outputs/evaluations/<dataset>/<config>/<model>/<timestamp>_results.json"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=sorted(DATASET_SPECS),
        help="Datasets to include, ordered left-to-right in the subplot grid.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_accuracy_vs_kv_budget.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--title",
        default="Accuracy vs. KV Cache Budget",
        help="Figure title.",
    )
    parser.add_argument(
        "--y-axis-0-100",
        action="store_true",
        help=(
            "Use a fixed 0-100 y-axis for every subplot. By default, each "
            "complete subplot uses its own tight min/max range."
        ),
    )
    return parser.parse_args()


def list_result_files(results_dir: Path, datasets: list[str]) -> list[Path]:
    result_files: list[Path] = []
    for dataset in datasets:
        dataset_dir = results_dir / DATASET_SPECS[dataset].results_subdir
        if not dataset_dir.exists():
            continue

        for result_path in dataset_dir.rglob("*_results.json"):
            relative_path = result_path.relative_to(results_dir)
            if len(relative_path.parts) < 4:
                continue
            result_files.append(result_path)

    return sorted(result_files)


def resolve_results_dir(requested_dir: Path, datasets: list[str]) -> Path:
    requested_dir = requested_dir.expanduser().resolve()
    if requested_dir.exists() and list_result_files(requested_dir, datasets):
        return requested_dir

    raise FileNotFoundError(
        "Could not find any *_results.json files under "
        f"{requested_dir} using the expected layout "
        "outputs/evaluations/<dataset>/<config>/<model>/<timestamp>_results.json"
    )


def infer_model_label(result_path: Path) -> str | None:
    text = result_path.as_posix().lower()
    if "0.5b" in text or "0p5b" in text:
        return "0.5B"
    if re.search(r"(^|[^0-9])7b([^0-9]|$)", text):
        return "7B"
    return None


def parse_sparsity(result_path: Path) -> float | None:
    experiment_dir = result_path.parents[1].name
    match = re.search(r"(?:^|-)sp(\d+)(?:$|-)", experiment_dir)
    if match is None:
        return None

    value = float(match.group(1))
    if value > 1.0:
        value /= 100.0
    return round(value, 4)


def parse_timestamp(result_path: Path) -> str:
    match = re.match(r"(\d{8}_\d{6})_results\.json$", result_path.name)
    if match is not None:
        return match.group(1)
    return result_path.name


def load_result_point(result_path: Path) -> ResultPoint | None:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Warning: could not parse {result_path}: {exc}", file=sys.stderr)
        return None

    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        print(f"Warning: missing results payload in {result_path}", file=sys.stderr)
        return None

    dataset_name = next(iter(results.keys()))
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        return None

    task_payload = results.get(dataset_name, {})
    metric_value = task_payload.get(spec.metric_key)
    if not isinstance(metric_value, (int, float)):
        print(
            f"Warning: metric {spec.metric_key!r} not found in {result_path}",
            file=sys.stderr,
        )
        return None

    model = infer_model_label(result_path)
    if model is None:
        print(
            f"Warning: could not infer model size from {result_path}",
            file=sys.stderr,
        )
        return None

    sparsity = parse_sparsity(result_path)
    if sparsity is None:
        print(
            f"Warning: could not infer sparsity from {result_path}",
            file=sys.stderr,
        )
        return None

    return ResultPoint(
        dataset=dataset_name,
        model=model,
        sparsity=sparsity,
        accuracy=float(metric_value) * spec.scale,
        timestamp=parse_timestamp(result_path),
        source_path=result_path,
    )


def load_latest_points(
    results_dir: Path,
    datasets: list[str],
) -> dict[tuple[str, str, float], ResultPoint]:
    latest_points: dict[tuple[str, str, float], ResultPoint] = {}

    for result_path in list_result_files(results_dir, datasets):
        point = load_result_point(result_path)
        if point is None:
            continue

        key = (point.dataset, point.model, point.sparsity)
        previous = latest_points.get(key)
        if previous is None or point.timestamp > previous.timestamp:
            latest_points[key] = point

    return latest_points


def build_dataset_series(
    latest_points: dict[tuple[str, str, float], ResultPoint],
    datasets: list[str],
) -> dict[str, dict[str, list[ResultPoint]]]:
    series = {
        dataset: {model: [] for model in MODEL_ORDER}
        for dataset in datasets
    }

    for point in latest_points.values():
        if point.dataset in series and point.model in series[point.dataset]:
            series[point.dataset][point.model].append(point)

    for dataset in series:
        for model in MODEL_ORDER:
            series[dataset][model].sort(key=lambda point: point.sparsity)

    return series


def warn_on_missing_points(series: dict[str, dict[str, list[ResultPoint]]]) -> None:
    expected = set(SPARSITY_VALUES)
    for dataset, model_points in series.items():
        label = DATASET_SPECS[dataset].label
        for model, points in model_points.items():
            found = {point.sparsity for point in points}
            missing = sorted(expected - found)
            if not points:
                print(
                    f"Warning: no data found for {label} / {model}.",
                    file=sys.stderr,
                )
                continue
            if missing:
                missing_text = ", ".join(f"{value:g}" for value in missing)
                print(
                    f"Warning: {label} / {model} is missing sparsity values: "
                    f"{missing_text}",
                    file=sys.stderr,
                )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def is_complete_series(points: list[ResultPoint]) -> bool:
    found = {point.sparsity for point in points}
    expected = set(SPARSITY_VALUES)
    return found == expected and len(points) == len(SPARSITY_VALUES)


def draw_placeholder(ax: plt.Axes, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=11,
        color="#666666",
        transform=ax.transAxes,
    )


def round_down(value: float, step: float) -> float:
    return math.floor(value / step) * step


def round_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def compute_smooth_ylim(ys: list[float]) -> tuple[float, float]:
    y_min = min(ys)
    y_max = max(ys)

    if math.isclose(y_min, y_max):
        pad = max(1.0, abs(y_min) * 0.02)
    else:
        # Use half the observed range as padding on each side so small local
        # reversals read as a smooth overall trend rather than a sharp zig-zag.
        pad = max(1.0, 0.5 * (y_max - y_min))

    lower = max(0.0, round_down(y_min - pad, 0.5))
    upper = min(100.0, round_up(y_max + pad, 0.5))

    if math.isclose(lower, upper):
        upper = min(100.0, lower + 1.0)

    return lower, upper


def plot_series(
    series: dict[str, dict[str, list[ResultPoint]]],
    datasets: list[str],
    title: str,
    y_axis_0_100: bool,
) -> plt.Figure:
    width = max(16, 4.1 * len(datasets))
    height = 7.0
    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(datasets),
        figsize=(width, height),
        sharex=True,
        squeeze=False,
    )

    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            dataset_spec = DATASET_SPECS[dataset]
            points = sorted(
                series[dataset][model],
                key=lambda point: 1.0 - point.sparsity,
            )

            if row_idx == 0:
                ax.set_title(dataset_spec.label, fontsize=12, pad=12)

            ax.text(
                0.02,
                0.96,
                model,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=MODEL_COLORS[model],
            )

            ax.set_xlim(-X_AXIS_PADDING, 1.0 + X_AXIS_PADDING)
            ax.set_xticks(KV_BUDGET_TICKS)
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
            if y_axis_0_100:
                ax.set_ylim(0.0, 100.0)

            if not points:
                draw_placeholder(ax, "No data found")
            elif not is_complete_series(points):
                draw_placeholder(
                    ax,
                    f"Incomplete data\n(found {len(points)}/{len(SPARSITY_VALUES)} points)",
                )
            else:
                xs = [1.0 - point.sparsity for point in points]
                ys = [point.accuracy for point in points]

                ax.plot(
                    xs,
                    ys,
                    color=MODEL_COLORS[model],
                    marker="o",
                    linewidth=2.2,
                    markersize=6,
                )

                if not y_axis_0_100:
                    ax.set_ylim(*compute_smooth_ylim(ys))

            style_axis(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=model,
        )
        for model in MODEL_ORDER
    ]

    fig.suptitle(title, fontsize=15, y=0.98)
    fig.supxlabel("KV Cache Budget", fontsize=12, y=0.06)
    fig.supylabel("Accuracy (%)", fontsize=12, x=0.04)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.tight_layout(rect=(0.04, 0.08, 1.0, 0.89))
    return fig


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir, args.datasets)
    latest_points = load_latest_points(results_dir, args.datasets)
    if not latest_points:
        raise RuntimeError(f"No supported dataset results found under {results_dir}")

    series = build_dataset_series(latest_points, args.datasets)
    warn_on_missing_points(series)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_series(series, args.datasets, args.title, args.y_axis_0_100)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
