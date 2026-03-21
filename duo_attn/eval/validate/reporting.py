import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def sort_rows_for_output(rows: List[Dict[str, object]]) -> None:
    run_order = {
        "streaming_first": 0,
        "retrieval_first": 1,
    }
    rows.sort(
        key=lambda row: (
            int(row.get("ratio_percent", 0)),
            run_order.get(str(row.get("run_name", "")), 99),
        )
    )


def sort_detailed_iterations_for_output(iterations: List[Dict[str, object]]) -> None:
    run_order = {
        "streaming_first": 0,
        "retrieval_first": 1,
    }
    iterations.sort(
        key=lambda item: (
            int(item.get("ratio_percent", 0)),
            run_order.get(str(item.get("run_name", "")), 99),
        )
    )


def build_report(
    *,
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    warnings: Sequence[str],
    ratios: Sequence[int],
    baseline_accuracy: float,
    baseline_exact: int,
    baseline_total: int,
    streaming_pool_size: int,
    retrieval_pool_size: int,
    actual_sparsity: float,
    sink_size: int,
    recent_size: int,
    num_layers: int,
    num_heads: int,
) -> Dict[str, object]:
    streaming_rows = [r for r in rows if r["run_name"] == "streaming_first"]
    retrieval_rows = [r for r in rows if r["run_name"] == "retrieval_first"]

    streaming_max_abs_drop = max(
        (abs(float(r["delta_vs_full_baseline"])) for r in streaming_rows),
        default=0.0,
    )

    streaming_100 = next(
        (r for r in streaming_rows if int(r["ratio_percent"]) == 100),
        None,
    )
    retrieval_100 = next(
        (r for r in retrieval_rows if int(r["ratio_percent"]) == 100),
        None,
    )

    streaming_drop_at_100 = (
        baseline_accuracy - float(streaming_100["accuracy"]) if streaming_100 else 0.0
    )
    retrieval_run_drop_at_100 = (
        baseline_accuracy - float(retrieval_100["accuracy"]) if retrieval_100 else 0.0
    )
    retrieval_minus_streaming_drop_at_100 = (
        retrieval_run_drop_at_100 - streaming_drop_at_100
    )

    return {
        "summary": {
            "baseline_accuracy_full": baseline_accuracy,
            "baseline_exact_match_count": baseline_exact,
            "baseline_processed_samples": baseline_total,
            "ratios": list(ratios),
            "streaming_pool_size": int(streaming_pool_size),
            "retrieval_pool_size": int(retrieval_pool_size),
            "actual_sparsity": float(actual_sparsity),
            "sink_size": int(sink_size),
            "recent_size": int(recent_size),
            "num_layers": int(num_layers),
            "num_heads_per_layer": int(num_heads),
            "total_heads": int(num_layers * num_heads),
        },
        "diagnostics": {
            "streaming_run_max_abs_drop": float(streaming_max_abs_drop),
            "retrieval_run_drop_at_100": float(retrieval_run_drop_at_100),
            "retrieval_minus_streaming_drop_at_100": float(
                retrieval_minus_streaming_drop_at_100
            ),
        },
        "results": list(rows),
        "warnings": list(warnings),
        "args": vars(args),
    }


def write_csv(output_csv: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_name",
        "ratio_percent",
        "pool_size",
        "num_heads_set_streaming",
        "accuracy",
        "delta_vs_full_baseline",
        "exact_match_count",
        "processed_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_accuracy_plot(
    output_plot: str,
    rows: Sequence[Dict[str, object]],
    baseline_accuracy: float,
) -> bool:
    if not rows:
        return False

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Warning: matplotlib is not installed; skipping accuracy plot output. "
            "Install matplotlib to enable --output_plot."
        )
        return False

    run_to_label = {
        "streaming_first": "Streaming-first",
        "retrieval_first": "Retrieval-first",
    }
    run_to_style = {
        "streaming_first": {"color": "#1f77b4", "marker": "o"},
        "retrieval_first": {"color": "#d62728", "marker": "s"},
    }

    points_by_run: Dict[str, List[tuple[int, float]]] = {
        run_name: [] for run_name in run_to_label
    }
    for row in rows:
        run_name = str(row.get("run_name", ""))
        if run_name not in points_by_run:
            continue
        ratio = int(row["ratio_percent"])
        accuracy_pct = 100.0 * float(row["accuracy"])
        points_by_run[run_name].append((ratio, accuracy_pct))

    if all(len(points) == 0 for points in points_by_run.values()):
        return False

    output_path = Path(output_plot)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for run_name, points in points_by_run.items():
        if not points:
            continue
        points = sorted(points, key=lambda pair: pair[0])
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(
            xs,
            ys,
            label=run_to_label[run_name],
            linewidth=2.0,
            markersize=5.0,
            **run_to_style[run_name],
        )

    baseline_pct = 100.0 * float(baseline_accuracy)
    ax.axhline(
        y=baseline_pct,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=f"Baseline all-full ({baseline_pct:.2f}%)",
    )

    ax.set_xlim(0, 100)
    ax.set_xticks(list(range(0, 101, 10)))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Ablated Ratio (%)")
    ax.set_ylabel("Task Accuracy (%)")
    ax.set_title("VNBench Secret-Word Ratio Sweep")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def write_outputs(
    args: argparse.Namespace,
    report: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    baseline_accuracy: float,
    detailed_iterations: Optional[Sequence[Dict[str, object]]] = None,
    emit_path_logs: bool = True,
    checkpoint_tag: Optional[str] = None,
) -> None:
    wrote_any = False

    if args.output_json is not None:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        wrote_any = True
        if emit_path_logs:
            print(f"Saved JSON report to {output_json_path}")

        detailed_json_path = output_json_path.with_name(
            "retrieval_pool_ratio_ablation_detailed.json"
        )
        detailed_report = dict(report)
        detailed_report["detailed_results"] = list(detailed_iterations or [])
        with detailed_json_path.open("w", encoding="utf-8") as f:
            json.dump(detailed_report, f, indent=2)
        wrote_any = True
        if emit_path_logs:
            print(f"Saved detailed JSON report to {detailed_json_path}")

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        wrote_any = True
        if emit_path_logs:
            print(f"Saved CSV report to {args.output_csv}")

    if args.output_plot:
        wrote_plot = write_accuracy_plot(
            output_plot=args.output_plot,
            rows=rows,
            baseline_accuracy=baseline_accuracy,
        )
        if wrote_plot:
            wrote_any = True
            if emit_path_logs:
                print(f"Saved accuracy plot to {args.output_plot}")

    if checkpoint_tag is not None and wrote_any:
        print(f"Updated output artifacts after {checkpoint_tag}")
