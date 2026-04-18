#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from textwrap import fill
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "figure.facecolor": "#f7f5ef",
        "axes.facecolor": "#fffdf8",
        "axes.edgecolor": "#d8d0c2",
        "axes.titleweight": "semibold",
        "axes.labelcolor": "#2a2724",
        "xtick.color": "#3b352f",
        "ytick.color": "#3b352f",
        "grid.color": "#d8d0c2",
        "grid.alpha": 0.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.facecolor": "#fffdf8",
        "legend.edgecolor": "#ddd3c4",
    }
)


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


def _method_manifest(payload: dict) -> dict:
    return dict(payload.get("evaluation_manifest", {}).get("method_manifest", {}) or {})


def duo_deploy_config(payload: dict) -> dict:
    return dict(_method_manifest(payload).get("duo_deploy_config", {}) or {})


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
        config = duo_deploy_config(payload)
        window_class = config.get("deploy_window_class")
        deploy_parts: list[str] = []
        if window_class and window_class != "training_aligned":
            sink = _format_display_value(config.get("deploy_sink_size"))
            recent = _format_display_value(config.get("deploy_recent_size"))
            if sink is not None:
                deploy_parts.append(f"sink={sink}")
            if recent is not None:
                deploy_parts.append(f"recent={recent}")
        parts = [f"s={formatted}"] if formatted is not None else []
        parts.extend(deploy_parts)
        return f"duo_streaming ({','.join(parts)})" if parts else "duo_streaming"
    if method == "rekv":
        topk = _format_display_value(run_config.get("retrieve_size"))
        n_local = _format_display_value(run_config.get("n_local"))
        parts: list[str] = []
        if topk is not None:
            parts.append(f"topk={topk}")
        if n_local is not None:
            parts.append(f"n_local={n_local}")
        return f"rekv ({','.join(parts)})" if parts else "rekv"
    if method == "rekv_no_offload":
        n_local = _format_display_value(run_config.get("n_local"))
        return f"rekv_no_offload (n_local={n_local})" if n_local is not None else "rekv_no_offload"
    if method == "duo_plus_rekv":
        topk = _format_display_value(run_config.get("retrieve_size"))
        sparsity = _format_display_value(run_config.get("sparsity"))
        config = duo_deploy_config(payload)
        window_class = config.get("deploy_window_class")
        parts: list[str] = []
        if topk is not None:
            parts.append(f"topk={topk}")
        if sparsity is not None:
            parts.append(f"s={sparsity}")
        if window_class and window_class != "training_aligned":
            sink = _format_display_value(config.get("deploy_sink_size"))
            recent = _format_display_value(config.get("deploy_recent_size"))
            if sink is not None:
                parts.append(f"sink={sink}")
            if recent is not None:
                parts.append(f"recent={recent}")
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
    family_order = {
        "full_streaming": 0,
        "duo_streaming": 1,
        "rekv": 2,
        "rekv_no_offload": 3,
        "duo_plus_rekv": 4,
    }.get(method, 99)
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
        "rekv_no_offload": "#72b7b2",
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
    if method == "rekv_no_offload":
        return "X"
    if method == "duo_plus_rekv":
        return "P"
    return "o"


def ordered_results(results: list[dict]) -> list[dict]:
    return sorted(results, key=sort_key)


def wrapped_display_label(payload: dict, width: int = 18) -> str:
    label = display_label(payload).replace(" (", "\n(")
    return fill(label, width=width, break_long_words=False, break_on_hyphens=False)


def _style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.grid(True, axis=grid_axis, linestyle="--", linewidth=0.8, alpha=0.45)
    axis.set_axisbelow(True)
    for spine_name in ("left", "bottom"):
        spine = axis.spines.get(spine_name)
        if spine is not None:
            spine.set_color("#d8d0c2")


def _annotate_bars(axis, bars, values: list[float | None]) -> None:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return
    offset = max(max(abs(value) for value in finite_values) * 0.02, 0.01)
    for bar, value in zip(bars, values):
        if value is None:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + offset,
            f"{float(value):.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#3b352f",
        )


def _scatter_with_label(axis, x: float, y: float, payload: dict, label: str) -> None:
    axis.scatter(
        [x],
        [y],
        s=135,
        color=color_for_payload(payload),
        marker=marker_for_payload(payload),
        edgecolors="#fffdf8",
        linewidths=1.2,
        zorder=3,
        label=label,
    )
    axis.annotate(
        label,
        (x, y),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=8.5,
        color="#2a2724",
        bbox={"boxstyle": "round,pad=0.18", "fc": "#fffdf8", "ec": "#e1d8cb", "alpha": 0.9},
    )


def _line_kwargs(payload: dict) -> dict:
    return {
        "marker": marker_for_payload(payload),
        "markersize": 6.5,
        "linewidth": 2.0,
        "color": color_for_payload(payload),
        "label": display_label(payload),
    }


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
    labels = [wrapped_display_label(item) for item in results]
    primary_quality_key = aggregate_quality_key(results[0]) if results else "normalized_exact_match"
    primary_quality_title = aggregate_quality_label(results[0]) if results else "Primary Quality"
    metrics = [
        (primary_quality_title, primary_quality_key),
        ("Avg TTFT (s)", "avg_ttft_sec"),
        ("Avg Answer Latency (s)", "avg_answer_latency_sec"),
        ("Avg Frame Ingest (s)", "avg_frame_ingest_latency_sec"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.flatten()
    for axis, (title, key) in zip(axes, metrics):
        values = [item["aggregate_metrics"].get(key) for item in results]
        plotted_values = [float(value) if value is not None else 0.0 for value in values]
        bars = axis.bar(
            labels,
            plotted_values,
            color=[color_for_payload(item) for item in results],
            edgecolor="#fffdf8",
            linewidth=1.2,
            alpha=0.92,
        )
        axis.set_title(title)
        _style_axis(axis, grid_axis="y")
        axis.tick_params(axis="x", rotation=0, labelsize=9)
        _annotate_bars(axis, bars, [float(value) if value is not None else None for value in values])

    fig.suptitle("Streaming Evaluation Overview", fontsize=15, fontweight="semibold", y=1.01)
    fig.tight_layout()
    out_path = output_dir / "aggregate_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_memory_comparison(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    labels = [wrapped_display_label(item) for item in results]
    values = [maybe_gb(item["aggregate_metrics"].get("peak_memory_bytes")) for item in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels,
        [value if value is not None else 0.0 for value in values],
        color=[color_for_payload(item) for item in results],
        edgecolor="#fffdf8",
        linewidth=1.2,
    )
    ax.set_title("Peak GPU Memory")
    ax.set_ylabel("GB")
    _style_axis(ax, grid_axis="y")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    _annotate_bars(ax, bars, values)
    fig.tight_layout()
    out_path = output_dir / "peak_memory_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_avg_memory_comparison(results: list[dict], output_dir: Path) -> Path | None:
    results = ordered_results(results)
    values = [maybe_gb(item["aggregate_metrics"].get("avg_gpu_memory_bytes_current")) for item in results]
    if not any(value is not None for value in values):
        return None

    labels = [display_label(item) for item in results]
    wrapped_labels = [wrapped_display_label(item) for item in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        wrapped_labels,
        [value if value is not None else 0.0 for value in values],
        color=[color_for_payload(item) for item in results],
        edgecolor="#fffdf8",
        linewidth=1.2,
    )
    ax.set_title("Avg GPU Memory")
    ax.set_ylabel("GB")
    _style_axis(ax, grid_axis="y")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    _annotate_bars(ax, bars, values)
    fig.tight_layout()
    out_path = output_dir / "avg_memory_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_cpu_offload_comparison(results: list[dict], output_dir: Path) -> Path | None:
    results = ordered_results(results)
    values = [maybe_gb(item["aggregate_metrics"].get("peak_cpu_offload_bytes")) for item in results]
    if not any(value is not None for value in values):
        return None

    wrapped_labels = [wrapped_display_label(item) for item in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        wrapped_labels,
        [value if value is not None else 0.0 for value in values],
        color=[color_for_payload(item) for item in results],
        edgecolor="#fffdf8",
        linewidth=1.2,
    )
    ax.set_title("Peak CPU / Offloaded KV")
    ax.set_ylabel("GB")
    _style_axis(ax, grid_axis="y")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    _annotate_bars(ax, bars, values)
    fig.tight_layout()
    out_path = output_dir / "peak_cpu_offload_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_avg_cpu_offload_comparison(results: list[dict], output_dir: Path) -> Path | None:
    results = ordered_results(results)
    values = [maybe_gb(item["aggregate_metrics"].get("avg_cpu_offload_bytes_current")) for item in results]
    if not any(value is not None for value in values):
        return None

    wrapped_labels = [wrapped_display_label(item) for item in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        wrapped_labels,
        [value if value is not None else 0.0 for value in values],
        color=[color_for_payload(item) for item in results],
        edgecolor="#fffdf8",
        linewidth=1.2,
    )
    ax.set_title("Avg CPU / Offloaded KV")
    ax.set_ylabel("GB")
    _style_axis(ax, grid_axis="y")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    _annotate_bars(ax, bars, values)
    fig.tight_layout()
    out_path = output_dir / "avg_cpu_offload_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_quality_latency_tradeoff(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for payload in results:
        label = display_label(payload)
        aggregate_metrics = payload["aggregate_metrics"]
        quality_key = aggregate_quality_key(payload)
        x = aggregate_metrics.get("avg_answer_latency_sec")
        y = aggregate_metrics.get(quality_key)
        if x is None or y is None:
            continue
        _scatter_with_label(ax, float(x), float(y), payload, label)

    ax.set_title("Quality vs Answer Latency")
    ax.set_xlabel("Avg Answer Latency (s)")
    ax.set_ylabel(aggregate_quality_label(results[0]) if results else "Quality")
    _style_axis(ax, grid_axis="both")
    fig.tight_layout()
    out_path = output_dir / "quality_latency_tradeoff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_quality_memory_tradeoff(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for payload in results:
        label = display_label(payload)
        aggregate_metrics = payload["aggregate_metrics"]
        quality_key = aggregate_quality_key(payload)
        x = maybe_gb(aggregate_metrics.get("peak_memory_bytes"))
        y = aggregate_metrics.get(quality_key)
        if x is None or y is None:
            continue
        _scatter_with_label(ax, float(x), float(y), payload, label)

    ax.set_title("Quality vs Peak GPU Memory")
    ax.set_xlabel("Peak GPU Memory (GB)")
    ax.set_ylabel(aggregate_quality_label(results[0]) if results else "Quality")
    _style_axis(ax, grid_axis="both")
    fig.tight_layout()
    out_path = output_dir / "quality_memory_tradeoff.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_quality_avg_memory_tradeoff(results: list[dict], output_dir: Path) -> Path | None:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    any_points = False
    for payload in results:
        label = display_label(payload)
        aggregate_metrics = payload["aggregate_metrics"]
        quality_key = aggregate_quality_key(payload)
        x = maybe_gb(aggregate_metrics.get("avg_gpu_memory_bytes_current"))
        y = aggregate_metrics.get(quality_key)
        if x is None or y is None:
            continue
        any_points = True
        _scatter_with_label(ax, float(x), float(y), payload, label)

    if not any_points:
        plt.close(fig)
        return None

    ax.set_title("Quality vs Avg GPU Memory")
    ax.set_xlabel("Avg GPU Memory (GB)")
    ax.set_ylabel(aggregate_quality_label(results[0]) if results else "Quality")
    _style_axis(ax, grid_axis="both")
    fig.tight_layout()
    out_path = output_dir / "quality_avg_memory_tradeoff.png"
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

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
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
        bars = axis.bar(labels, deltas, color=colors, edgecolor="#fffdf8", linewidth=1.2)
        axis.set_title(title)
        _style_axis(axis, grid_axis="y")
        axis.tick_params(axis="x", rotation=0, labelsize=9)
        _annotate_bars(axis, bars, deltas)

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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
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
            _scatter_with_label(axis, float(x_value), float(y_value), payload, display_label(payload))

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
        _style_axis(axis, grid_axis="both")

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
                    "current_memory_bytes": conversation["method_stats"].get("current_memory_bytes"),
                    "peak_memory_bytes": conversation["method_stats"].get("peak_memory_bytes"),
                    "cpu_offload_bytes_current": conversation["method_stats"].get("cpu_offload_bytes_current"),
                    "cpu_offload_bytes_peak": conversation["method_stats"].get("cpu_offload_bytes_peak"),
                    "ttft_sec": conversation["method_stats"].get("ttft_sec"),
                    "answer_latency_sec": conversation["method_stats"].get("answer_latency_sec"),
                    "retrieval_latency_sec": conversation["method_stats"].get("retrieval_latency_sec"),
                    "avg_retrieved_block_count": conversation["method_stats"].get("avg_retrieved_block_count"),
                    "retrieved_block_indices_union": conversation["method_stats"].get(
                        "retrieved_block_indices_union", []
                    ),
                    "retrieved_timestamps_sec_union": conversation["method_stats"].get(
                        "retrieved_timestamps_sec_union", []
                    ),
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
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for payload in results:
        label = display_label(payload)
        rows = flatten_conversations(payload)
        xs = list(range(len(rows)))
        color = color_for_payload(payload)
        marker = marker_for_payload(payload)
        axes[0].plot(xs, [row["frames_ingested"] for row in rows], **_line_kwargs(payload))
        axes[1].plot(xs, [row["ttft_sec"] for row in rows], **_line_kwargs(payload))
        axes[2].plot(xs, [row["answer_latency_sec"] for row in rows], **_line_kwargs(payload))

    axes[0].set_ylabel("Frames Ingested")
    axes[0].set_title("Per-Conversation Streaming Progress")
    axes[1].set_ylabel("TTFT (s)")
    axes[2].set_ylabel("Answer Latency (s)")
    axes[2].set_xlabel("Conversation Index")
    for axis in axes:
        _style_axis(axis, grid_axis="both")
        axis.legend()
    fig.tight_layout()
    out_path = output_dir / "per_conversation_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_efficiency_vs_context(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for payload in results:
        label = display_label(payload)
        rows = flatten_conversations(payload)
        xs = [row["frames_ingested"] for row in rows]
        axes[0].plot(xs, [row["answer_latency_sec"] for row in rows], **_line_kwargs(payload))
        axes[1].plot(
            xs,
            [maybe_gb(row["peak_memory_bytes"]) for row in rows],
            **_line_kwargs(payload),
        )
        axes[2].plot(
            xs,
            [maybe_gb(row["cpu_offload_bytes_current"]) for row in rows],
            **_line_kwargs(payload),
        )

    axes[0].set_title("Latency and Memory vs Processed Frames")
    axes[0].set_ylabel("Answer Latency (s)")
    axes[1].set_ylabel("Peak GPU Memory (GB)")
    axes[2].set_ylabel("CPU / Offloaded KV (GB)")
    axes[2].set_xlabel("Frames Ingested Before Answer")
    for axis in axes:
        _style_axis(axis, grid_axis="both")
        axis.legend()
    fig.tight_layout()
    out_path = output_dir / "efficiency_vs_context.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_quality_vs_context(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(11, 5.5))
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
        ax.plot(xs, ys, **_line_kwargs(payload))

    ax.set_title("Quality vs Processed Frames")
    ax.set_xlabel("Frames Ingested Before Answer")
    ax.set_ylabel("Quality")
    _style_axis(ax, grid_axis="both")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / "quality_vs_context.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_question_timeline(results: list[dict], output_dir: Path) -> Path:
    results = ordered_results(results)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for payload in results:
        label = display_label(payload)
        rows = flatten_conversations(payload)
        ax.step(
            [row["end_time"] for row in rows],
            [row["frames_ingested"] for row in rows],
            where="post",
            linewidth=2.0,
            markersize=6.5,
            marker=marker_for_payload(payload),
            label=label,
            color=color_for_payload(payload),
        )

    ax.set_title("Frames Ingested by Question Time")
    ax.set_xlabel("Question End Time (s)")
    ax.set_ylabel("Frames Ingested So Far")
    _style_axis(ax, grid_axis="both")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / "question_timeline.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_rekv_retrieval(results: list[dict], output_dir: Path) -> Path | None:
    results = ordered_results(results)
    retrieval_payloads = [
        payload for payload in results if method_family(payload) in {"rekv", "duo_plus_rekv", "rekv_no_offload"}
    ]
    if not retrieval_payloads:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    any_rows = False
    for payload in retrieval_payloads:
        rows = flatten_conversations(payload)
        if not any(row["retrieval_latency_sec"] is not None for row in rows):
            continue
        any_rows = True
        xs = list(range(len(rows)))
        label = display_label(payload)
        axes[0].plot(xs, [row["retrieval_latency_sec"] for row in rows], **_line_kwargs(payload))
        axes[1].plot(xs, [row["avg_retrieved_block_count"] for row in rows], **_line_kwargs(payload))
    if not any_rows:
        plt.close(fig)
        return None

    axes[0].set_ylabel("Retrieval Latency (s)")
    axes[0].set_title("ReKV Retrieval Diagnostics")
    axes[1].set_ylabel("Avg Retrieved Blocks")
    axes[1].set_xlabel("Conversation Index")
    for axis in axes:
        _style_axis(axis, grid_axis="both")
        axis.legend()
    fig.tight_layout()
    out_path = output_dir / "rekv_retrieval_diagnostics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_retrieval_timeline(results: list[dict], output_dir: Path) -> Path | None:
    results = ordered_results(results)
    retrieval_payloads = [
        payload for payload in results if method_family(payload) in {"rekv", "duo_plus_rekv"}
    ]
    if not retrieval_payloads:
        return None

    fig, axes = plt.subplots(len(retrieval_payloads), 1, figsize=(11, 4.2 * len(retrieval_payloads)))
    if len(retrieval_payloads) == 1:
        axes = [axes]

    any_points = False
    for axis, payload in zip(axes, retrieval_payloads):
        rows = flatten_conversations(payload)
        for row_idx, row in enumerate(rows):
            retrieved_timestamps = [float(value) for value in row["retrieved_timestamps_sec_union"]]
            if retrieved_timestamps:
                any_points = True
                axis.scatter(
                    retrieved_timestamps,
                    [row_idx] * len(retrieved_timestamps),
                    s=26,
                    color=color_for_payload(payload),
                    alpha=0.82,
                    edgecolors="#fffdf8",
                    linewidths=0.5,
                )
            axis.scatter(
                [row["end_time"]],
                [row_idx],
                s=60,
                color="#333333",
                marker="x",
            )
        axis.set_title(f"Retrieved Context Timeline: {display_label(payload)}")
        axis.set_ylabel("Conversation Index")
        _style_axis(axis, grid_axis="both")

    if not any_points:
        plt.close(fig)
        return None

    axes[-1].set_xlabel("Timestamp (s)")
    fig.tight_layout()
    out_path = output_dir / "retrieval_timeline.png"
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
        "rekv_no_offload": {"sample_fps", "n_local"},
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

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        for label, points in method_to_points.items():
            xs = [point[0] for point in points]
            quality_key = aggregate_quality_key(points[0][1])
            color = color_for_method(label)
            axes[0].plot(
                xs,
                [point[1]["aggregate_metrics"].get(quality_key) for point in points],
                marker="o",
                markersize=6,
                linewidth=2.0,
                label=label,
                color=color,
            )
            axes[1].plot(
                xs,
                [point[1]["aggregate_metrics"].get("avg_answer_latency_sec") for point in points],
                marker="o",
                markersize=6,
                linewidth=2.0,
                label=label,
                color=color,
            )
            axes[2].plot(
                xs,
                [maybe_gb(point[1]["aggregate_metrics"].get("peak_memory_bytes")) for point in points],
                marker="o",
                markersize=6,
                linewidth=2.0,
                label=label,
                color=color,
            )

        axes[0].set_title(f"{title} Sweep")
        axes[0].set_ylabel("Quality")
        axes[1].set_ylabel("Latency (s)")
        axes[2].set_ylabel("Peak GPU Memory (GB)")
        axes[2].set_xlabel(title)
        for axis in axes:
            _style_axis(axis, grid_axis="both")
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
    avg_memory_plot = plot_avg_memory_comparison(results, output_dir)
    if avg_memory_plot is not None:
        generated["avg_memory_comparison"] = str(avg_memory_plot)
    avg_memory_tradeoff_plot = plot_quality_avg_memory_tradeoff(results, output_dir)
    if avg_memory_tradeoff_plot is not None:
        generated["quality_avg_memory_tradeoff"] = str(avg_memory_tradeoff_plot)
    delta_plot = plot_delta_to_baseline(results, output_dir)
    if delta_plot is not None:
        generated["delta_to_baseline"] = str(delta_plot)
    cpu_offload_plot = plot_cpu_offload_comparison(results, output_dir)
    if cpu_offload_plot is not None:
        generated["peak_cpu_offload_comparison"] = str(cpu_offload_plot)
    avg_cpu_offload_plot = plot_avg_cpu_offload_comparison(results, output_dir)
    if avg_cpu_offload_plot is not None:
        generated["avg_cpu_offload_comparison"] = str(avg_cpu_offload_plot)
    pareto_plot = plot_pareto_with_arrows(results, output_dir)
    if pareto_plot is not None:
        generated["pareto_tradeoffs_with_arrows"] = str(pareto_plot)
    rekv_plot = plot_rekv_retrieval(results, output_dir)
    if rekv_plot is not None:
        generated["rekv_retrieval_diagnostics"] = str(rekv_plot)
    retrieval_timeline_plot = plot_retrieval_timeline(results, output_dir)
    if retrieval_timeline_plot is not None:
        generated["retrieval_timeline"] = str(retrieval_timeline_plot)
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
