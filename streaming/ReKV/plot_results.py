#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paper-friendly plots from streaming/ReKV result JSON files."
    )
    parser.add_argument("result_paths", nargs="+", help="One or more *_results.json files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated plots. Defaults to a plots/ sibling next to the first result file.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_source_path"] = str(path)
    return payload


def method_label(payload: dict) -> str:
    return payload["run_config"]["method"]


def maybe_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024 ** 3)


def plot_aggregate_comparison(results: list[dict], output_dir: Path) -> Path:
    labels = [method_label(item) for item in results]
    metrics = [
        ("Normalized EM", "normalized_exact_match"),
        ("Avg TTFT (s)", "avg_ttft_sec"),
        ("Avg Answer Latency (s)", "avg_answer_latency_sec"),
        ("Avg Frame Ingest (s)", "avg_frame_ingest_latency_sec"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for axis, (title, key) in zip(axes, metrics):
        values = [item["aggregate_metrics"].get(key) for item in results]
        axis.bar(labels, values, color=colors[: len(labels)])
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "aggregate_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_memory_comparison(results: list[dict], output_dir: Path) -> Path:
    labels = [method_label(item) for item in results]
    values = [maybe_gb(item["aggregate_metrics"].get("peak_memory_bytes")) for item in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"][: len(labels)])
    ax.set_title("Peak GPU Memory")
    ax.set_ylabel("GB")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "peak_memory_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def flatten_conversations(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for video in payload.get("videos", []):
        for idx, conversation in enumerate(video.get("conversations", [])):
            rows.append(
                {
                    "video_id": video["video_id"],
                    "conversation_index": idx,
                    "question": conversation["question"],
                    "end_time": float(conversation["end_time"]),
                    "frames_ingested": int(conversation["num_frames_ingested_before_answer"]),
                    "ttft_sec": conversation["method_stats"].get("ttft_sec"),
                    "answer_latency_sec": conversation["method_stats"].get("answer_latency_sec"),
                    "retrieval_latency_sec": conversation["method_stats"].get("retrieval_latency_sec"),
                    "avg_retrieved_block_count": conversation["method_stats"].get("avg_retrieved_block_count"),
                }
            )
    return rows


def plot_per_conversation(results: list[dict], output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colors = {"duo_streaming": "#1f77b4", "rekv": "#ff7f0e"}

    for payload in results:
        label = method_label(payload)
        rows = flatten_conversations(payload)
        xs = list(range(len(rows)))
        axes[0].plot(xs, [row["frames_ingested"] for row in rows], marker="o", label=label, color=colors.get(label))
        axes[1].plot(xs, [row["ttft_sec"] for row in rows], marker="o", label=label, color=colors.get(label))
        axes[2].plot(xs, [row["answer_latency_sec"] for row in rows], marker="o", label=label, color=colors.get(label))

    axes[0].set_ylabel("Frames Ingested")
    axes[0].set_title("Per-Conversation Streaming Progress")
    axes[1].set_ylabel("TTFT (s)")
    axes[2].set_ylabel("Answer Latency (s)")
    axes[2].set_xlabel("Conversation Index")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    out_path = output_dir / "per_conversation_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_question_timeline(results: list[dict], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"duo_streaming": "#1f77b4", "rekv": "#ff7f0e"}

    for payload in results:
        label = method_label(payload)
        rows = flatten_conversations(payload)
        ax.step(
            [row["end_time"] for row in rows],
            [row["frames_ingested"] for row in rows],
            where="post",
            marker="o",
            label=label,
            color=colors.get(label),
        )

    ax.set_title("Frames Ingested by Question Time")
    ax.set_xlabel("Question End Time (s)")
    ax.set_ylabel("Frames Ingested So Far")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / "question_timeline.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_rekv_retrieval(results: list[dict], output_dir: Path) -> Path | None:
    rekv_payload = next((payload for payload in results if method_label(payload) == "rekv"), None)
    if rekv_payload is None:
        return None

    rows = flatten_conversations(rekv_payload)
    if not any(row["retrieval_latency_sec"] is not None for row in rows):
        return None

    xs = list(range(len(rows)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(xs, [row["retrieval_latency_sec"] for row in rows], marker="o", color="#ff7f0e")
    axes[0].set_ylabel("Retrieval Latency (s)")
    axes[0].set_title("ReKV Retrieval Diagnostics")
    axes[1].plot(xs, [row["avg_retrieved_block_count"] for row in rows], marker="o", color="#ff7f0e")
    axes[1].set_ylabel("Avg Retrieved Blocks")
    axes[1].set_xlabel("Conversation Index")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "rekv_retrieval_diagnostics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    result_paths = [Path(path).expanduser().resolve() for path in args.result_paths]
    results = [load_result(path) for path in result_paths]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = result_paths[0].parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = {
        "aggregate_comparison": str(plot_aggregate_comparison(results, output_dir)),
        "peak_memory_comparison": str(plot_memory_comparison(results, output_dir)),
        "per_conversation_metrics": str(plot_per_conversation(results, output_dir)),
        "question_timeline": str(plot_question_timeline(results, output_dir)),
    }
    rekv_plot = plot_rekv_retrieval(results, output_dir)
    if rekv_plot is not None:
        generated["rekv_retrieval_diagnostics"] = str(rekv_plot)

    summary_path = output_dir / "plot_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(generated, handle, indent=2)
    print(f"Saved plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
