#!/usr/bin/env python3
"""Plot a sink/recent DuoAttention sweep from lmms-eval result JSON files."""

from __future__ import annotations

import argparse
import csv
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
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
except ModuleNotFoundError as exc:
    if exc.name in {"matplotlib", "numpy"}:
        raise SystemExit(
            "matplotlib and numpy are required to render the sink/recent sweep plot."
        ) from exc
    raise


DEFAULT_SINK_VALUES = [64, 128, 256, 512]
DEFAULT_RECENT_VALUES = [128, 256, 512, 1024]
DEFAULT_RESULTS_DIR = Path("outputs/evaluations/egoschema-sweep")
DEFAULT_OUTPUT = Path("egoschema_sink_recent_sweep.png")
DEFAULT_CSV_OUTPUT = Path("egoschema_sink_recent_sweep.csv")
ANCHOR_POINT = (256, 512)
TRAINING_POINT = (512, 1024)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    metric_key: str
    scale: float


@dataclass(frozen=True)
class ResultPoint:
    dataset: str
    model: str
    accuracy: float
    timestamp: str
    source_path: Path
    mode: str
    sparsity: float | None
    max_frames_num: int | None
    decoding_simulation_length: int | None
    deploy_sink_size: int | None
    deploy_recent_size: int | None


DATASET_SPECS = {
    "videomme": DatasetSpec(
        label="Video-MME",
        metric_key="videomme_perception_score,none",
        scale=1.0,
    ),
    "mlvu_dev": DatasetSpec(
        label="MLVU Dev",
        metric_key="mlvu_percetion_score,none",
        scale=1.0,
    ),
    "egoschema_subset": DatasetSpec(
        label="EgoSchema",
        metric_key="score,none",
        scale=100.0,
    ),
    "longvideobench_val_v": DatasetSpec(
        label="LongVideoBench-V",
        metric_key="lvb_acc,none",
        scale=100.0,
    ),
    "longvideobench_val_i": DatasetSpec(
        label="LongVideoBench-I",
        metric_key="lvb_acc,none",
        scale=100.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read lmms-eval result JSON files, extract deploy-time DuoAttention "
            "sink/recent overrides, and render an annotated heatmap."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Sweep directory to scan recursively for *_results.json files. "
            "Expected layout: outputs/evaluations/egoschema-sweep/<config>/<model>/..."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path for the heatmap.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Output CSV path for the filtered summary table.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SPECS),
        default="egoschema_subset",
        help="Dataset to plot.",
    )
    parser.add_argument(
        "--model",
        choices=["0.5B", "7B"],
        default="0.5B",
        help="Model size label to include.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Required DuoAttention sparsity for included sweep points.",
    )
    parser.add_argument(
        "--max-frames-num",
        type=int,
        default=64,
        help="Required max_frames_num value for included sweep points.",
    )
    parser.add_argument(
        "--decoding-simulation-length",
        type=int,
        default=256,
        help="Required decoding_simulation_length value for included sweep points.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional explicit figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI.",
    )
    return parser.parse_args()


def list_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.expanduser().resolve().rglob("*_results.json"))


def parse_model_args(model_args: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_part in model_args.split(","):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(value)


def infer_model_label(*texts: str) -> str | None:
    for text in texts:
        normalized = text.lower()
        if "0.5b" in normalized or "0p5b" in normalized:
            return "0.5B"
        if re.search(r"(^|[^0-9])7b([^0-9]|$)", normalized):
            return "7B"
    return None


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
        return None

    dataset = next(iter(results))
    spec = DATASET_SPECS.get(dataset)
    if spec is None:
        return None

    metric_value = results.get(dataset, {}).get(spec.metric_key)
    if not isinstance(metric_value, (int, float)):
        print(
            f"Warning: metric {spec.metric_key!r} not found in {result_path}",
            file=sys.stderr,
        )
        return None

    config_payload = payload.get("config", {})
    model_args = parse_model_args(str(config_payload.get("model_args", "")))
    pretrained = model_args.get("pretrained", "")
    model = infer_model_label(pretrained, result_path.as_posix())
    if model is None:
        print(
            f"Warning: could not infer model size from {result_path}",
            file=sys.stderr,
        )
        return None

    attn_dir = model_args.get("attn_dir") or model_args.get("attn_load_dir")
    mode = "duo" if attn_dir else "baseline"

    return ResultPoint(
        dataset=dataset,
        model=model,
        accuracy=float(metric_value) * spec.scale,
        timestamp=parse_timestamp(result_path),
        source_path=result_path.resolve(),
        mode=mode,
        sparsity=parse_optional_float(model_args.get("sparsity")),
        max_frames_num=parse_optional_int(model_args.get("max_frames_num")),
        decoding_simulation_length=parse_optional_int(
            model_args.get("decoding_simulation_length")
        ),
        deploy_sink_size=parse_optional_int(model_args.get("deploy_sink_size")),
        deploy_recent_size=parse_optional_int(model_args.get("deploy_recent_size")),
    )


def matches_common_filters(point: ResultPoint, args: argparse.Namespace) -> bool:
    if point.dataset != args.dataset:
        return False
    if point.model != args.model:
        return False
    if point.max_frames_num != args.max_frames_num:
        return False
    if point.decoding_simulation_length != args.decoding_simulation_length:
        return False
    return True


def matches_duo_filters(point: ResultPoint, args: argparse.Namespace) -> bool:
    if not matches_common_filters(point, args):
        return False
    if point.mode != "duo":
        return False
    if point.sparsity is None or not math.isclose(point.sparsity, args.sparsity, abs_tol=1e-6):
        return False
    return True


def latest_by_key(points: list[ResultPoint], key_func) -> dict[tuple[object, ...], ResultPoint]:
    latest: dict[tuple[object, ...], ResultPoint] = {}
    for point in points:
        key = key_func(point)
        previous = latest.get(key)
        if previous is None or point.timestamp > previous.timestamp:
            latest[key] = point
    return latest


def compute_color_limits(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 100.0

    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        pad = max(1.0, abs(vmin) * 0.05)
        return max(0.0, vmin - pad), min(100.0, vmax + pad)

    pad = max(0.5, 0.1 * (vmax - vmin))
    return max(0.0, vmin - pad), min(100.0, vmax + pad)


def write_csv_summary(csv_output: Path, points: list[ResultPoint]) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "model",
                "sparsity",
                "max_frames_num",
                "decoding_simulation_length",
                "sink_size",
                "recent_size",
                "accuracy",
                "timestamp",
                "result_path",
            ],
        )
        writer.writeheader()
        for point in sorted(
            points,
            key=lambda item: (item.deploy_sink_size or -1, item.deploy_recent_size or -1),
        ):
            writer.writerow(
                {
                    "dataset": point.dataset,
                    "model": point.model,
                    "sparsity": point.sparsity,
                    "max_frames_num": point.max_frames_num,
                    "decoding_simulation_length": point.decoding_simulation_length,
                    "sink_size": point.deploy_sink_size,
                    "recent_size": point.deploy_recent_size,
                    "accuracy": f"{point.accuracy:.6f}",
                    "timestamp": point.timestamp,
                    "result_path": str(point.source_path),
                }
            )


def add_highlight(
    ax: plt.Axes,
    sink_values: list[int],
    recent_values: list[int],
    *,
    sink_size: int,
    recent_size: int,
    edgecolor: str,
    label: str,
) -> Line2D | None:
    if sink_size not in sink_values or recent_size not in recent_values:
        return None

    row_idx = sink_values.index(sink_size)
    col_idx = recent_values.index(recent_size)
    ax.add_patch(
        Rectangle(
            (col_idx - 0.5, row_idx - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor=edgecolor,
            linewidth=2.6,
        )
    )
    return Line2D(
        [0],
        [0],
        color=edgecolor,
        linewidth=2.6,
        label=label,
    )


def plot_heatmap(
    *,
    points: list[ResultPoint],
    args: argparse.Namespace,
    baseline_point: ResultPoint | None,
) -> plt.Figure:
    sink_values = sorted(set(DEFAULT_SINK_VALUES) | {point.deploy_sink_size for point in points if point.deploy_sink_size is not None})
    recent_values = sorted(set(DEFAULT_RECENT_VALUES) | {point.deploy_recent_size for point in points if point.deploy_recent_size is not None})

    matrix = np.full((len(sink_values), len(recent_values)), np.nan, dtype=float)
    point_lookup = {
        (point.deploy_sink_size, point.deploy_recent_size): point
        for point in points
    }

    for row_idx, sink_size in enumerate(sink_values):
        for col_idx, recent_size in enumerate(recent_values):
            point = point_lookup.get((sink_size, recent_size))
            if point is not None:
                matrix[row_idx, col_idx] = point.accuracy

    valid_values = [point.accuracy for point in points]
    vmin, vmax = compute_color_limits(valid_values)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(recent_values)))
    ax.set_xticklabels([str(value) for value in recent_values])
    ax.set_yticks(range(len(sink_values)))
    ax.set_yticklabels([str(value) for value in sink_values])
    ax.set_xlabel("Recent Size")
    ax.set_ylabel("Sink Size")

    for row_idx, sink_size in enumerate(sink_values):
        for col_idx, recent_size in enumerate(recent_values):
            value = matrix[row_idx, col_idx]
            if math.isnan(value):
                label = "NA"
                color = "#666666"
            else:
                label = f"{value:.1f}"
                color = "white" if value >= 0.5 * (vmin + vmax) else "#1f1f1f"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if not math.isnan(value) else "normal",
                color=color,
            )

    ax.set_xticks(np.arange(-0.5, len(recent_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sink_values), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = []
    anchor_handle = add_highlight(
        ax,
        sink_values,
        recent_values,
        sink_size=ANCHOR_POINT[0],
        recent_size=ANCHOR_POINT[1],
        edgecolor="#d62728",
        label="Target operating point (256, 512)",
    )
    if anchor_handle is not None:
        legend_handles.append(anchor_handle)

    training_handle = add_highlight(
        ax,
        sink_values,
        recent_values,
        sink_size=TRAINING_POINT[0],
        recent_size=TRAINING_POINT[1],
        edgecolor="#2ca02c",
        label="Training-window point (512, 1024)",
    )
    if training_handle is not None:
        legend_handles.append(training_handle)

    dataset_label = DATASET_SPECS[args.dataset].label
    title = args.title or (
        f"{dataset_label} Sink/Recent Sweep ({args.model}, sparsity={args.sparsity:g})"
    )
    subtitle = (
        f"max_frames_num={args.max_frames_num}, "
        f"decoding_simulation_length={args.decoding_simulation_length}"
    )
    if baseline_point is not None:
        subtitle += f", baseline={baseline_point.accuracy:.1f}%"

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=10, color="#444444")

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Accuracy (%)")

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
        )

    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.92))
    return fig


def main() -> None:
    args = parse_args()
    result_files = list_result_files(args.results_dir)
    if not result_files:
        raise FileNotFoundError(
            f"Could not find any *_results.json files under {args.results_dir.expanduser().resolve()}"
        )

    loaded_points = [point for point in (load_result_point(path) for path in result_files) if point is not None]

    skipped_missing_overrides = 0
    duo_candidates: list[ResultPoint] = []
    baseline_candidates: list[ResultPoint] = []
    for point in loaded_points:
        if point.mode == "baseline":
            if matches_common_filters(point, args):
                baseline_candidates.append(point)
            continue

        if not matches_duo_filters(point, args):
            continue

        if point.deploy_sink_size is None or point.deploy_recent_size is None:
            skipped_missing_overrides += 1
            continue

        duo_candidates.append(point)

    latest_duo = latest_by_key(
        duo_candidates,
        key_func=lambda point: (point.deploy_sink_size, point.deploy_recent_size),
    )
    latest_baseline = latest_by_key(
        baseline_candidates,
        key_func=lambda point: (point.dataset, point.model),
    )

    filtered_points = list(latest_duo.values())
    if not filtered_points:
        raise RuntimeError(
            "No DuoAttention sweep points matched the requested filters after "
            "requiring deploy_sink_size and deploy_recent_size in config.model_args."
        )

    expected_pairs = {
        (sink_size, recent_size)
        for sink_size in DEFAULT_SINK_VALUES
        for recent_size in DEFAULT_RECENT_VALUES
        if recent_size >= sink_size
    }
    found_pairs = {
        (point.deploy_sink_size, point.deploy_recent_size)
        for point in filtered_points
    }
    missing_pairs = sorted(expected_pairs - found_pairs)

    if skipped_missing_overrides:
        print(
            f"Warning: skipped {skipped_missing_overrides} DuoAttention results that matched the "
            "dataset/model/sparsity filters but did not record deploy overrides in config.model_args.",
            file=sys.stderr,
        )
    if missing_pairs:
        missing_text = ", ".join(f"({sink},{recent})" for sink, recent in missing_pairs)
        print(
            f"Warning: missing sink/recent pairs for the default 13-point grid: {missing_text}",
            file=sys.stderr,
        )

    baseline_point = latest_baseline.get((args.dataset, args.model))

    csv_output = args.csv_output.expanduser().resolve()
    write_csv_summary(csv_output, filtered_points)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_heatmap(points=filtered_points, args=args, baseline_point=baseline_point)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved CSV summary to {csv_output}")
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
