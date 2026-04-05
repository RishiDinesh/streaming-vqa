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


def method_family(payload: dict) -> str:
    return str(payload.get("run_config", {}).get("method", "unknown"))


def _format_display_value(value: int | float | None) -> str | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric.is_integer():
        return f"{numeric:.1f}" if abs(numeric) < 10 else str(int(numeric))
    return f"{numeric:.3g}"


def display_label(payload: dict) -> str:
    run_config = payload.get("run_config", {})
    method = method_family(payload)
    if method == "duo_streaming":
        sparsity = run_config.get("sparsity")
        if sparsity is None:
            sparsity = duo_display_sparsity(payload)
        formatted = _format_display_value(sparsity)
        return f"duo_streaming (s={formatted})" if formatted is not None else "duo_streaming"
    if method == "rekv":
        topk = _format_display_value(run_config.get("retrieve_size"))
        n_local = _format_display_value(run_config.get("n_local"))
        parts: list[str] = []
        if topk is not None:
            parts.append(f"topk={topk}")
        if n_local is not None:
            parts.append(f"n_local={n_local}")
        return f"rekv ({','.join(parts)})" if parts else "rekv"
    if method == "duo_plus_rekv":
        topk = _format_display_value(run_config.get("retrieve_size"))
        sparsity = _format_display_value(run_config.get("sparsity"))
        parts: list[str] = []
        if topk is not None:
            parts.append(f"topk={topk}")
        if sparsity is not None:
            parts.append(f"s={sparsity}")
        return f"duo_plus_rekv ({','.join(parts)})" if parts else "duo_plus_rekv"
    return method


def maybe_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024 ** 3)


def duo_display_sparsity(payload: dict) -> float | None:
    sparsity = payload.get("run_config", {}).get("sparsity")
    if sparsity is not None:
        return float(sparsity)
    for video in payload.get("videos", []):
        for conversation in video.get("conversations", []):
            method_stats = conversation.get("method_stats", {})
            if method_stats.get("actual_sparsity") is not None:
                return float(method_stats["actual_sparsity"])
    return None


def sort_key(payload: dict) -> tuple[int, float, str]:
    method = method_family(payload)
    family_order = {"full_streaming": 0, "duo_streaming": 1, "rekv": 2, "duo_plus_rekv": 3}.get(method, 99)
    duo_priority = 0.0
    if method == "duo_streaming":
        sparsity = duo_display_sparsity(payload)
        duo_priority = -sparsity if sparsity is not None else 0.0
    return (family_order, duo_priority, display_label(payload))


def color_for_method(label: str) -> str:
    palette = {
        "full_streaming": "#4c78a8",
        "duo_streaming": "#f58518",
        "rekv": "#54a24b",
        "duo_plus_rekv": "#b279a2",
    }
    return palette.get(label, "#9c755f")


def color_for_payload(payload: dict) -> str:
    method = method_family(payload)
    if method == "duo_streaming" and (duo_display_sparsity(payload) or 0.5) <= 0.0:
        return "#e45756"
    return color_for_method(method)


def marker_for_payload(payload: dict) -> str:
    method = method_family(payload)
    if method == "full_streaming":
        return "o"
    if method == "duo_streaming":
        return "s" if (duo_display_sparsity(payload) or 0.5) <= 0.0 else "D"
    if method == "rekv":
        return "^"
    if method == "duo_plus_rekv":
        return "P"
    return "o"


def ordered_results(results: list[dict]) -> list[dict]:
    return sorted(results, key=sort_key)


def aggregate_quality_key(payload: dict) -> str:
    aggregate_metrics = payload.get("aggregate_metrics", {})
    if aggregate_metrics.get("primary_quality_metric"):
        return str(aggregate_metrics["primary_quality_metric"])
    if aggregate_metrics.get("avg_rouge_l_f1") is not None:
        return "avg_rouge_l_f1"
    if aggregate_metrics.get("avg_token_f1") is not None:
        return "avg_token_f1"
    return "normalized_exact_match"


def aggregate_quality_label(payload: dict) -> str:
    key = aggregate_quality_key(payload)
    labels = {
        "avg_judge_score": "Avg Judge Score",
        "avg_rouge_l_f1": "Avg ROUGE-L F1",
        "avg_token_f1": "Avg Token F1",
        "normalized_exact_match": "Normalized EM",
    }
    return labels.get(key, key)


def plot_aggregate_comparison(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    labels = [display_label(item) for item in results]
    primary_quality_key = aggregate_quality_key(results[0]) if results else "normalized_exact_match"
    primary_quality_title = aggregate_quality_label(results[0]) if results else "Primary Quality"
    metrics = [
        (primary_quality_title, primary_quality_key),
        ("Avg TTFT (s)", "avg_ttft_sec"),
        ("Avg Answer Latency (s)", "avg_answer_latency_sec"),
        ("Avg Frame Ingest (s)", "avg_frame_ingest_latency_sec"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for axis, (title, key) in zip(axes, metrics):
        values = [item["aggregate_metrics"].get(key) for item in results]
        axis.bar(labels, values, color=[color_for_payload(item) for item in results])
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out_path = output_dir / "aggregate_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_memory_comparison(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    labels = [display_label(item) for item in results]
    values = [maybe_gb(item["aggregate_metrics"].get("peak_memory_bytes")) for item in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=[color_for_payload(item) for item in results])
    ax.set_title("Peak GPU Memory")
    ax.set_ylabel("GB")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    out_path = output_dir / "peak_memory_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_quality_latency_tradeoff(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    for payload in results:
        label = display_label(payload)
        aggregate_metrics = payload["aggregate_metrics"]
        quality_key = aggregate_quality_key(payload)
        x = aggregate_metrics.get("avg_answer_latency_sec")
        y = aggregate_metrics.get(quality_key)
        if x is None or y is None:
            continue
        ax.scatter(
            [x],
            [y],
            s=110,
            color=color_for_payload(payload),
            marker=marker_for_payload(payload),
            label=label,
        )
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")

    ax.set_title("Quality vs Answer Latency")
    ax.set_xlabel("Avg Answer Latency (s)")
    ax.set_ylabel(aggregate_quality_label(results[0]) if results else "Quality")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "quality_latency_tradeoff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_quality_memory_tradeoff(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    for payload in results:
        label = display_label(payload)
        aggregate_metrics = payload["aggregate_metrics"]
        quality_key = aggregate_quality_key(payload)
        x = maybe_gb(aggregate_metrics.get("peak_memory_bytes"))
        y = aggregate_metrics.get(quality_key)
        if x is None or y is None:
            continue
        ax.scatter(
            [x],
            [y],
            s=110,
            color=color_for_payload(payload),
            marker=marker_for_payload(payload),
            label=label,
        )
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")

    ax.set_title("Quality vs Peak GPU Memory")
    ax.set_xlabel("Peak GPU Memory (GB)")
    ax.set_ylabel(aggregate_quality_label(results[0]) if results else "Quality")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "quality_memory_tradeoff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _aggregate_metric_value(payload: dict, key: str) -> float | None:
    value = payload.get("aggregate_metrics", {}).get(key)
    if value is None:
        return None
    return float(value)


def _find_baseline_payload(results: list[dict], method_name: str) -> dict | None:
    matches = [payload for payload in results if method_family(payload) == method_name]
    if not matches:
        return None
    return matches[0]


def plot_delta_to_baseline(results: list[dict], output_dir: Path) -> Path | None:
    results = ordered_results(results)
    full_payload = _find_baseline_payload(results, "full_streaming")
    rekv_payload = _find_baseline_payload(results, "rekv")
    comparisons: list[tuple[str, dict, dict]] = []

    for payload in results:
        method = method_family(payload)
        if method == "duo_streaming" and full_payload is not None:
            comparisons.append((f"{display_label(payload)} - full_streaming", payload, full_payload))
        elif method == "duo_plus_rekv" and rekv_payload is not None:
            comparisons.append((f"{display_label(payload)} - rekv", payload, rekv_payload))

    if not comparisons:
        return None

    quality_key = aggregate_quality_key(results[0])
    metric_specs = [
        (aggregate_quality_label(results[0]), quality_key, 1.0),
        ("Avg Answer Latency Delta (s)", "avg_answer_latency_sec", 1.0),
        ("Peak Memory Delta (GB)", "peak_memory_bytes", 1024 ** 3),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels = [label for label, _, _ in comparisons]
    colors = [color_for_payload(payload) for _, payload, _ in comparisons]
    for axis, (title, metric_key, scale) in zip(axes, metric_specs):
        deltas: list[float] = []
        for _, payload, baseline in comparisons:
            value = _aggregate_metric_value(payload, metric_key)
            baseline_value = _aggregate_metric_value(baseline, metric_key)
            if value is None or baseline_value is None:
                deltas.append(0.0)
            else:
                deltas.append((value - baseline_value) / scale)

        axis.axhline(0.0, color="#666666", linewidth=1.0)
        axis.bar(labels, deltas, color=colors)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out_path = output_dir / "delta_to_baseline.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_pareto_with_arrows(results: list[dict], output_dir: Path) -> Path | None:
    results = ordered_results(results)
    if not results:
        return None

    quality_key = aggregate_quality_key(results[0])
    full_payload = _find_baseline_payload(results, "full_streaming")
    rekv_payload = _find_baseline_payload(results, "rekv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metric_specs = [
        ("avg_answer_latency_sec", "Avg Answer Latency (s)"),
        ("peak_memory_bytes", "Peak GPU Memory (GB)"),
    ]

    for axis, (metric_key, x_label) in zip(axes, metric_specs):
        for payload in results:
            x_value = _aggregate_metric_value(payload, metric_key)
            y_value = _aggregate_metric_value(payload, quality_key)
            if x_value is None or y_value is None:
                continue
            if metric_key == "peak_memory_bytes":
                x_value /= 1024 ** 3
            axis.scatter(
                [x_value],
                [y_value],
                s=120,
                color=color_for_payload(payload),
                marker=marker_for_payload(payload),
                label=display_label(payload),
            )
            axis.annotate(display_label(payload), (x_value, y_value), xytext=(5, 5), textcoords="offset points")

        arrow_pairs: list[tuple[dict | None, dict | None]] = [
            (full_payload, next((payload for payload in results if method_family(payload) == "duo_streaming"), None)),
            (rekv_payload, next((payload for payload in results if method_family(payload) == "duo_plus_rekv"), None)),
        ]
        for start_payload, end_payload in arrow_pairs:
            if start_payload is None or end_payload is None:
                continue
            start_x = _aggregate_metric_value(start_payload, metric_key)
            end_x = _aggregate_metric_value(end_payload, metric_key)
            start_y = _aggregate_metric_value(start_payload, quality_key)
            end_y = _aggregate_metric_value(end_payload, quality_key)
            if None in {start_x, end_x, start_y, end_y}:
                continue
            if metric_key == "peak_memory_bytes":
                start_x /= 1024 ** 3
                end_x /= 1024 ** 3
            axis.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#666666", "lw": 1.2},
            )

        axis.set_xlabel(x_label)
        axis.set_ylabel(aggregate_quality_label(results[0]))
        axis.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_title("Pareto View: Quality vs Latency")
    axes[1].set_title("Pareto View: Quality vs Memory")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    out_path = output_dir / "pareto_tradeoffs_with_arrows.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def flatten_conversations(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for video in payload.get("videos", []):
        for idx, conversation in enumerate(video.get("conversations", [])):
            scores = conversation.get("scores", {})
            rows.append(
                {
                    "video_id": video["video_id"],
                    "conversation_index": idx,
                    "question": conversation["question"],
                    "end_time": float(conversation["end_time"]),
                    "frames_ingested": int(conversation["num_frames_ingested_before_answer"]),
                    "peak_memory_bytes": conversation["method_stats"].get("peak_memory_bytes"),
                    "ttft_sec": conversation["method_stats"].get("ttft_sec"),
                    "answer_latency_sec": conversation["method_stats"].get("answer_latency_sec"),
                    "retrieval_latency_sec": conversation["method_stats"].get("retrieval_latency_sec"),
                    "avg_retrieved_block_count": conversation["method_stats"].get("avg_retrieved_block_count"),
                    "normalized_exact_match": scores.get(
                        "normalized_exact_match",
                        conversation.get("normalized_exact_match"),
                    ),
                    "token_f1": scores.get("token_f1"),
                    "rouge_l_f1": scores.get("rouge_l_f1"),
                    "judge_score": scores.get("judge_score"),
                }
            )
    return rows


def plot_per_conversation(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for payload in results:
        label = display_label(payload)
        rows = flatten_conversations(payload)
        xs = list(range(len(rows)))
        color = color_for_payload(payload)
        marker = marker_for_payload(payload)
        axes[0].plot(xs, [row["frames_ingested"] for row in rows], marker=marker, label=label, color=color)
        axes[1].plot(xs, [row["ttft_sec"] for row in rows], marker=marker, label=label, color=color)
        axes[2].plot(xs, [row["answer_latency_sec"] for row in rows], marker=marker, label=label, color=color)

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


def plot_efficiency_vs_context(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for payload in results:
        label = display_label(payload)
        rows = flatten_conversations(payload)
        xs = [row["frames_ingested"] for row in rows]
        color = color_for_payload(payload)
        marker = marker_for_payload(payload)
        axes[0].plot(xs, [row["answer_latency_sec"] for row in rows], marker=marker, label=label, color=color)
        axes[1].plot(
            xs,
            [maybe_gb(row["peak_memory_bytes"]) for row in rows],
            marker=marker,
            label=label,
            color=color,
        )

    axes[0].set_title("Latency and Memory vs Processed Frames")
    axes[0].set_ylabel("Answer Latency (s)")
    axes[1].set_ylabel("Peak GPU Memory (GB)")
    axes[1].set_xlabel("Frames Ingested Before Answer")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    out_path = output_dir / "efficiency_vs_context.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_quality_vs_context(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    for payload in results:
        label = display_label(payload)
        rows = flatten_conversations(payload)
        xs = [row["frames_ingested"] for row in rows]
        if any(row["judge_score"] is not None for row in rows):
            quality_key = "judge_score"
        elif any(row["rouge_l_f1"] is not None for row in rows):
            quality_key = "rouge_l_f1"
        else:
            quality_key = "normalized_exact_match"
        ys = [row[quality_key] for row in rows]
        ax.plot(
            xs,
            ys,
            marker=marker_for_payload(payload),
            label=label,
            color=color_for_payload(payload),
        )

    ax.set_title("Quality vs Processed Frames")
    ax.set_xlabel("Frames Ingested Before Answer")
    ax.set_ylabel("Quality")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / "quality_vs_context.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_question_timeline(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    for payload in results:
        label = display_label(payload)
        rows = flatten_conversations(payload)
        ax.step(
            [row["end_time"] for row in rows],
            [row["frames_ingested"] for row in rows],
            where="post",
            marker=marker_for_payload(payload),
            label=label,
            color=color_for_payload(payload),
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
    results = ordered_results(results)
    retrieval_payloads = [
        payload for payload in results if method_family(payload) in {"rekv", "duo_plus_rekv"}
    ]
    if not retrieval_payloads:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    any_rows = False
    for payload in retrieval_payloads:
        rows = flatten_conversations(payload)
        if not any(row["retrieval_latency_sec"] is not None for row in rows):
            continue
        any_rows = True
        xs = list(range(len(rows)))
        label = display_label(payload)
        axes[0].plot(
            xs,
            [row["retrieval_latency_sec"] for row in rows],
            marker=marker_for_payload(payload),
            color=color_for_payload(payload),
            label=label,
        )
        axes[1].plot(
            xs,
            [row["avg_retrieved_block_count"] for row in rows],
            marker=marker_for_payload(payload),
            color=color_for_payload(payload),
            label=label,
        )
    if not any_rows:
        plt.close(fig)
        return None

    axes[0].set_ylabel("Retrieval Latency (s)")
    axes[0].set_title("ReKV Retrieval Diagnostics")
    axes[1].set_ylabel("Avg Retrieved Blocks")
    axes[1].set_xlabel("Conversation Index")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    out_path = output_dir / "rekv_retrieval_diagnostics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_auto_sweeps(results: list[dict], output_dir: Path) -> list[Path]:
    candidate_keys = [
        ("sparsity", "Sparsity"),
        ("retrieve_size", "Retrieve Size"),
        ("n_local", "Local Window"),
        ("sample_fps", "Sample FPS"),
    ]
    relevant_keys = {
        "full_streaming": {"sample_fps"},
        "duo_streaming": {"sample_fps", "sparsity"},
        "rekv": {"sample_fps", "retrieve_size", "n_local"},
        "duo_plus_rekv": {"sample_fps", "retrieve_size", "n_local", "sparsity"},
    }
    generated: list[Path] = []
    for key, title in candidate_keys:
        method_to_points: dict[str, list[tuple[float, dict]]] = {}
        for payload in results:
            method = method_family(payload)
            if key not in relevant_keys.get(method, set()):
                continue
            value = payload.get("run_config", {}).get(key)
            if value is None:
                continue
            method_to_points.setdefault(method, []).append((float(value), payload))
        method_to_points = {
            label: sorted(points, key=lambda item: item[0])
            for label, points in method_to_points.items()
            if len({point[0] for point in points}) >= 2
        }
        if not method_to_points:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        for label, points in method_to_points.items():
            xs = [point[0] for point in points]
            quality_key = aggregate_quality_key(points[0][1])
            color = color_for_method(label)
            axes[0].plot(
                xs,
                [point[1]["aggregate_metrics"].get(quality_key) for point in points],
                marker="o",
                label=label,
                color=color,
            )
            axes[1].plot(
                xs,
                [point[1]["aggregate_metrics"].get("avg_answer_latency_sec") for point in points],
                marker="o",
                label=label,
                color=color,
            )
            axes[2].plot(
                xs,
                [maybe_gb(point[1]["aggregate_metrics"].get("peak_memory_bytes")) for point in points],
                marker="o",
                label=label,
                color=color,
            )

        axes[0].set_title(f"{title} Sweep")
        axes[0].set_ylabel("Quality")
        axes[1].set_ylabel("Latency (s)")
        axes[2].set_ylabel("Peak GPU Memory (GB)")
        axes[2].set_xlabel(title)
        for axis in axes:
            axis.grid(True, linestyle="--", alpha=0.3)
            axis.legend()
        fig.tight_layout()
        out_path = output_dir / f"{key}_sweep_curves.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        generated.append(out_path)
    return generated


def main() -> int:
    args = parse_args()
    result_paths = [Path(path).expanduser().resolve() for path in args.result_paths]
    results = ordered_results([load_result(path) for path in result_paths])

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = result_paths[0].parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = {
        "aggregate_comparison": str(plot_aggregate_comparison(results, output_dir)),
        "peak_memory_comparison": str(plot_memory_comparison(results, output_dir)),
        "quality_latency_tradeoff": str(plot_quality_latency_tradeoff(results, output_dir)),
        "quality_memory_tradeoff": str(plot_quality_memory_tradeoff(results, output_dir)),
        "per_conversation_metrics": str(plot_per_conversation(results, output_dir)),
        "efficiency_vs_context": str(plot_efficiency_vs_context(results, output_dir)),
        "quality_vs_context": str(plot_quality_vs_context(results, output_dir)),
        "question_timeline": str(plot_question_timeline(results, output_dir)),
    }
    delta_plot = plot_delta_to_baseline(results, output_dir)
    if delta_plot is not None:
        generated["delta_to_baseline"] = str(delta_plot)
    pareto_plot = plot_pareto_with_arrows(results, output_dir)
    if pareto_plot is not None:
        generated["pareto_tradeoffs_with_arrows"] = str(pareto_plot)
    rekv_plot = plot_rekv_retrieval(results, output_dir)
    if rekv_plot is not None:
        generated["rekv_retrieval_diagnostics"] = str(rekv_plot)
    sweep_plots = plot_auto_sweeps(results, output_dir)
    if sweep_plots:
        generated["sweep_curves"] = [str(path) for path in sweep_plots]
    generated["series_labels"] = [display_label(payload) for payload in results]
    generated["series_sources"] = {
        display_label(payload): payload["_source_path"] for payload in results
    }

    summary_path = output_dir / "plot_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(generated, handle, indent=2)
    print(f"Saved plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
