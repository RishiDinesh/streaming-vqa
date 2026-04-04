#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_results import aggregate_quality_key, aggregate_quality_label, color_for_payload, display_label, method_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple subsample result JSONs and summarize cross-slice stability."
    )
    parser.add_argument("result_paths", nargs="+", help="One or more *_results.json files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the summary markdown/csv/json and stability plots.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_source_path"] = str(path)
    return payload


def slice_name(payload: dict) -> str:
    run_config = payload.get("run_config", {})
    return str(run_config.get("subsample_name") or f"offset{run_config.get('video_offset', 0)}")


def slice_sort_key(name: str) -> tuple[int, int | str, int | str, str]:
    subsample_match = re.fullmatch(r"subsample(\d+)(?:[_-]offset(\d+))?", name)
    if subsample_match:
        base_count = int(subsample_match.group(1))
        offset = int(subsample_match.group(2) or 0)
        return (0, base_count, offset, name)

    offset_match = re.fullmatch(r"offset(\d+)", name)
    if offset_match:
        return (1, int(offset_match.group(1)), name, name)

    suffixed_offset = re.search(r"(?:^|[_-])offset(\d+)$", name)
    if suffixed_offset:
        return (2, int(suffixed_offset.group(1)), name, name)

    return (3, name, name, name)


def collect_rows(results: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in results:
        agg = payload.get("aggregate_metrics", {})
        rows.append(
            {
                "dataset": payload.get("run_config", {}).get("dataset"),
                "slice_name": slice_name(payload),
                "method": method_family(payload),
                "display_label": display_label(payload),
                "primary_quality_metric": agg.get("primary_quality_metric"),
                "primary_quality_score": agg.get("primary_quality_score"),
                "avg_answer_latency_sec": agg.get("avg_answer_latency_sec"),
                "avg_frame_ingest_latency_sec": agg.get("avg_frame_ingest_latency_sec"),
                "avg_ttft_sec": agg.get("avg_ttft_sec"),
                "peak_memory_bytes": agg.get("peak_memory_bytes"),
                "total_frames_ingested": agg.get("total_frames_ingested"),
                "total_conversations_answered": agg.get("total_conversations_answered"),
                "source_path": payload.get("_source_path"),
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    headers = [
        "dataset",
        "slice_name",
        "display_label",
        "primary_quality_score",
        "avg_answer_latency_sec",
        "peak_memory_bytes",
        "total_conversations_answered",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_stability_report(rows: list[dict[str, object]]) -> dict[str, object]:
    by_dataset_method: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["display_label"]))
        by_dataset_method.setdefault(key, []).append(row)

    report: dict[str, object] = {"groups": []}
    for (dataset, label), group in sorted(by_dataset_method.items()):
        ordered = sorted(group, key=lambda item: slice_sort_key(str(item["slice_name"])))
        quality_values = [item["primary_quality_score"] for item in ordered if item["primary_quality_score"] is not None]
        latency_values = [item["avg_answer_latency_sec"] for item in ordered if item["avg_answer_latency_sec"] is not None]
        memory_values = [item["peak_memory_bytes"] for item in ordered if item["peak_memory_bytes"] is not None]
        report["groups"].append(
            {
                "dataset": dataset,
                "display_label": label,
                "slice_names": [item["slice_name"] for item in ordered],
                "quality_range": (
                    max(quality_values) - min(quality_values) if len(quality_values) >= 2 else 0.0
                ),
                "latency_range_sec": (
                    max(latency_values) - min(latency_values) if len(latency_values) >= 2 else 0.0
                ),
                "peak_memory_range_bytes": (
                    max(memory_values) - min(memory_values) if len(memory_values) >= 2 else 0
                ),
                "source_paths": [item["source_path"] for item in ordered],
            }
        )
    return report


def plot_slice_stability(results: list[dict], output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for payload in results:
        key = (str(payload.get("run_config", {}).get("dataset")), display_label(payload))
        grouped.setdefault(key, []).append(payload)

    for (_, label), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda payload: slice_sort_key(slice_name(payload)))
        xs = [slice_name(payload) for payload in ordered]
        quality_key = aggregate_quality_key(ordered[0])
        color = color_for_payload(ordered[0])
        axes[0].plot(
            xs,
            [payload.get("aggregate_metrics", {}).get(quality_key) for payload in ordered],
            marker="o",
            label=label,
            color=color,
        )
        axes[1].plot(
            xs,
            [payload.get("aggregate_metrics", {}).get("avg_answer_latency_sec") for payload in ordered],
            marker="o",
            label=label,
            color=color,
        )
        axes[2].plot(
            xs,
            [
                (payload.get("aggregate_metrics", {}).get("peak_memory_bytes") or 0) / (1024 ** 3)
                for payload in ordered
            ],
            marker="o",
            label=label,
            color=color,
        )

    axes[0].set_title("Cross-Slice Stability")
    axes[0].set_ylabel(aggregate_quality_label(results[0]) if results else "Quality")
    axes[1].set_ylabel("Avg Answer Latency (s)")
    axes[2].set_ylabel("Peak Memory (GB)")
    axes[2].set_xlabel("Slice")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    output_path = output_dir / "slice_stability.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [load_payload(Path(path).expanduser().resolve()) for path in args.result_paths]
    rows = collect_rows(results)
    write_summary_csv(rows, output_dir / "summary.csv")
    write_summary_markdown(rows, output_dir / "summary.md")
    stability_report = build_stability_report(rows)
    with open(output_dir / "stability_report.json", "w", encoding="utf-8") as handle:
        json.dump(stability_report, handle, indent=2)
    plot_path = plot_slice_stability(results, output_dir)
    manifest = {
        "summary_csv": str(output_dir / "summary.csv"),
        "summary_md": str(output_dir / "summary.md"),
        "stability_report": str(output_dir / "stability_report.json"),
        "slice_stability_plot": str(plot_path),
    }
    with open(output_dir / "compare_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved comparison summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
