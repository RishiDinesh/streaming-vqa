#!/usr/bin/env python3
"""Plot side-by-side sink/recent DuoAttention sweep heatmaps from lmms-eval JSON."""

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
except ModuleNotFoundError as exc:
    if exc.name in {"matplotlib", "numpy"}:
        raise SystemExit(
            "matplotlib and numpy are required to render the sink/recent sweep plot."
        ) from exc
    raise


DEFAULT_SINK_VALUES = [64, 128, 256, 512]
DEFAULT_RECENT_VALUES = [128, 256, 512, 1024]
DEFAULT_COMPARISON_SPARSITIES = [1.0, 0.5]
DEFAULT_RESULTS_DIR_CANDIDATES = [
    Path("evaluations/egoschema-7b-sweep"),
    Path("outputs/evaluations/egoschema-7b-sweep"),
]
HEATMAP_VMIN = 35.0
HEATMAP_VMAX = 65.0


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


def resolve_default_results_dir() -> Path:
    for candidate in DEFAULT_RESULTS_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_RESULTS_DIR_CANDIDATES[0]


def parse_sparsity_arg(raw_value: str) -> float:
    value = raw_value.strip().lower()
    if value.startswith("sp"):
        percent_text = value[2:].replace("p", ".")
        return float(percent_text) / 100.0
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    return float(value)


def format_sparsity_tag(value: float) -> str:
    percent = value * 100.0
    rounded_percent = round(percent)
    if math.isclose(percent, rounded_percent, abs_tol=1e-6):
        return f"sp{rounded_percent}"
    formatted = f"{percent:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"sp{formatted}"


def format_sparsity_comparison_tag(sparsities: list[float]) -> str:
    return "_vs_".join(format_sparsity_tag(value) for value in sparsities)


def default_output_path_for_sparsities(sparsities: list[float]) -> Path:
    return Path(
        f"egoschema_7b_sink_recent_sweep_{format_sparsity_comparison_tag(sparsities)}.png"
    )


def default_csv_path_for_sparsities(sparsities: list[float]) -> Path:
    return Path(
        f"egoschema_7b_sink_recent_sweep_{format_sparsity_comparison_tag(sparsities)}.csv"
    )


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
        default=resolve_default_results_dir(),
        help=(
            "Sweep directory to scan recursively for *_results.json files. "
            "Defaults to the first existing path among "
            "evaluations/egoschema-7b-sweep and "
            "outputs/evaluations/egoschema-7b-sweep."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path for the comparison heatmap. "
            "Defaults to egoschema_7b_sink_recent_sweep_<left-tag>_vs_<right-tag>.png."
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help=(
            "Output CSV path for the filtered comparison summary table. "
            "Defaults to egoschema_7b_sink_recent_sweep_<left-tag>_vs_<right-tag>.csv."
        ),
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
        default="7B",
        help="Model size label to include.",
    )
    parser.add_argument(
        "--sparsities",
        nargs=2,
        type=parse_sparsity_arg,
        default=DEFAULT_COMPARISON_SPARSITIES,
        metavar=("LEFT_SPARSITY", "RIGHT_SPARSITY"),
        help=(
            "Ordered DuoAttention sparsities to compare, left-to-right. "
            "Accepts values like 1.0, 0.5, sp100, or sp50. "
            "Defaults to sp100 on the left and sp50 on the right."
        ),
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
        help="Optional explicit figure title. Defaults to a dataset/model comparison title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI.",
    )
    return parser.parse_args()


def list_result_files(results_dir: Path) -> list[Path]:
    root = results_dir.expanduser().resolve()
    return sorted(root.rglob("*_results.json"))


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


def matches_duo_filters(
    point: ResultPoint,
    args: argparse.Namespace,
    sparsity: float,
) -> bool:
    if not matches_common_filters(point, args):
        return False
    if point.mode != "duo":
        return False
    if point.sparsity is None or not math.isclose(point.sparsity, sparsity, abs_tol=1e-6):
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
            key=lambda item: (
                item.sparsity if item.sparsity is not None else -1.0,
                item.deploy_sink_size or -1,
                item.deploy_recent_size or -1,
            ),
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


def build_axis_values(points_by_sparsity: dict[float, list[ResultPoint]]) -> tuple[list[int], list[int]]:
    sink_values = set(DEFAULT_SINK_VALUES)
    recent_values = set(DEFAULT_RECENT_VALUES)

    for points in points_by_sparsity.values():
        sink_values.update(
            point.deploy_sink_size
            for point in points
            if point.deploy_sink_size is not None
        )
        recent_values.update(
            point.deploy_recent_size
            for point in points
            if point.deploy_recent_size is not None
        )

    return sorted(sink_values), sorted(recent_values)


def build_accuracy_matrix(
    points: list[ResultPoint],
    sink_values: list[int],
    recent_values: list[int],
) -> np.ndarray:
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

    return matrix


def plot_heatmaps(
    *,
    points_by_sparsity: dict[float, list[ResultPoint]],
    args: argparse.Namespace,
) -> plt.Figure:
    sink_values, recent_values = build_axis_values(points_by_sparsity)
    panel_count = len(args.sparsities)

    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(7.2 * panel_count, 5.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")

    image = None
    for ax, sparsity in zip(axes_flat, args.sparsities):
        matrix = build_accuracy_matrix(
            points_by_sparsity[sparsity],
            sink_values,
            recent_values,
        )
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(
            masked,
            cmap=cmap,
            vmin=HEATMAP_VMIN,
            vmax=HEATMAP_VMAX,
            aspect="auto",
        )

        ax.set_xticks(range(len(recent_values)))
        ax.set_xticklabels([str(value) for value in recent_values])
        ax.set_yticks(range(len(sink_values)))
        ax.set_yticklabels([str(value) for value in sink_values])
        ax.set_xlabel("Recent Size")
        ax.set_title(format_sparsity_tag(sparsity), fontsize=12, pad=10)

        for row_idx, sink_size in enumerate(sink_values):
            for col_idx, recent_size in enumerate(recent_values):
                value = matrix[row_idx, col_idx]
                if math.isnan(value):
                    label = "NA"
                    color = "#666666"
                else:
                    label = f"{value:.1f}"
                    color = (
                        "white"
                        if value >= 0.5 * (HEATMAP_VMIN + HEATMAP_VMAX)
                        else "#1f1f1f"
                    )
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

    axes_flat[0].set_ylabel("Sink Size")
    for ax in axes_flat[1:]:
        ax.set_ylabel("")

    colorbar = fig.colorbar(image, ax=axes_flat.tolist(), pad=0.02)
    colorbar.set_label("Accuracy (%)")

    title = (
        args.title
        or f"{DATASET_SPECS[args.dataset].label} {args.model} DuoAttention Sink/Recent Sweep"
    )
    fig.suptitle(title, fontsize=14)
    return fig


def collect_latest_duo_points(
    loaded_points: list[ResultPoint],
    args: argparse.Namespace,
    sparsity: float,
) -> tuple[list[ResultPoint], int]:
    skipped_missing_overrides = 0
    duo_candidates: list[ResultPoint] = []

    for point in loaded_points:
        if not matches_duo_filters(point, args, sparsity):
            continue

        if point.deploy_sink_size is None or point.deploy_recent_size is None:
            skipped_missing_overrides += 1
            continue

        duo_candidates.append(point)

    latest_duo = latest_by_key(
        duo_candidates,
        key_func=lambda point: (point.deploy_sink_size, point.deploy_recent_size),
    )
    return list(latest_duo.values()), skipped_missing_overrides


def warn_on_missing_pairs(points: list[ResultPoint], sparsity: float) -> None:
    expected_pairs = {
        (sink_size, recent_size)
        for sink_size in DEFAULT_SINK_VALUES
        for recent_size in DEFAULT_RECENT_VALUES
    }
    found_pairs = {
        (point.deploy_sink_size, point.deploy_recent_size)
        for point in points
    }
    missing_pairs = sorted(expected_pairs - found_pairs)
    if not missing_pairs:
        return

    missing_text = ", ".join(f"({sink},{recent})" for sink, recent in missing_pairs)
    print(
        f"Warning: missing sink/recent pairs for {format_sparsity_tag(sparsity)} "
        f"in the default 16-point grid: {missing_text}",
        file=sys.stderr,
    )


def main() -> None:
    args = parse_args()
    result_files = list_result_files(args.results_dir)
    if not result_files:
        raise FileNotFoundError(
            "Could not find any *_results.json files under "
            f"{args.results_dir.expanduser().resolve()}."
        )

    loaded_points = [
        point for point in (load_result_point(path) for path in result_files) if point is not None
    ]

    points_by_sparsity: dict[float, list[ResultPoint]] = {}
    all_filtered_points: list[ResultPoint] = []
    for sparsity in args.sparsities:
        filtered_points, skipped_missing_overrides = collect_latest_duo_points(
            loaded_points,
            args,
            sparsity,
        )
        if not filtered_points:
            raise RuntimeError(
                "No DuoAttention sweep points matched the requested filters after "
                "requiring deploy_sink_size and deploy_recent_size in "
                f"config.model_args for {format_sparsity_tag(sparsity)}."
            )

        if skipped_missing_overrides:
            print(
                f"Warning: skipped {skipped_missing_overrides} DuoAttention results that matched "
                f"the dataset/model/{format_sparsity_tag(sparsity)} filters but did not record "
                "deploy overrides in config.model_args.",
                file=sys.stderr,
            )
        warn_on_missing_pairs(filtered_points, sparsity)

        points_by_sparsity[sparsity] = filtered_points
        all_filtered_points.extend(filtered_points)

    csv_output = (
        args.csv_output or default_csv_path_for_sparsities(args.sparsities)
    ).expanduser().resolve()
    write_csv_summary(csv_output, all_filtered_points)

    output_path = (
        args.output or default_output_path_for_sparsities(args.sparsities)
    ).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_heatmaps(points_by_sparsity=points_by_sparsity, args=args)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved CSV summary to {csv_output}")
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
