#!/usr/bin/env python3
"""Summarize per-task-type MLVU accuracy from lmms-eval sample logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-task-type accuracy for an lmms-eval MLVU run. "
            "The input can be a run directory, a results.json file, or a "
            "samples_mlvu*.jsonl file."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to an MLVU run directory, results.json, or samples_mlvu*.jsonl file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of a text table.",
    )
    return parser.parse_args()


def resolve_samples_path(input_path: Path) -> Path:
    path = input_path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        candidates = sorted(path.glob("*samples_mlvu*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No samples_mlvu*.jsonl file found in {path}")
        if len(candidates) > 1:
            raise ValueError(
                "Found multiple MLVU sample logs in "
                f"{path}: {', '.join(candidate.name for candidate in candidates)}"
            )
        return candidates[0]

    if path.name.endswith(".jsonl"):
        return path

    if path.name.endswith("_results.json"):
        parent = path.parent
        stem = path.name[: -len("_results.json")]
        sibling = parent / f"{stem}_samples_mlvu_dev.jsonl"
        if sibling.exists():
            return sibling

        candidates = sorted(parent.glob("*samples_mlvu*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find a sibling samples_mlvu*.jsonl next to {path}"
            )
        if len(candidates) > 1:
            raise ValueError(
                "Found multiple sibling MLVU sample logs next to "
                f"{path}: {', '.join(candidate.name for candidate in candidates)}"
            )
        return candidates[0]

    raise ValueError(
        "Unsupported input path. Expected a directory, *_results.json, or "
        "samples_mlvu*.jsonl file."
    )


def load_records(samples_path: Path) -> Iterable[dict]:
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {samples_path}"
                ) from exc


def summarize(samples_path: Path) -> dict:
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in load_records(samples_path):
        payload = record.get("mlvu_percetion_score")
        if not isinstance(payload, dict):
            continue

        task_type = payload.get("task_type")
        pred_answer = payload.get("pred_answer")
        answer = payload.get("answer")
        if task_type is None or pred_answer is None or answer is None:
            continue

        by_task[task_type]["total"] += 1
        by_task[task_type]["correct"] += int(pred_answer == answer)

    if not by_task:
        raise ValueError(
            f"No valid MLVU records with mlvu_percetion_score found in {samples_path}"
        )

    rows = []
    total_correct = 0
    total_count = 0
    macro_sum = 0.0
    for task_type in sorted(by_task):
        correct = by_task[task_type]["correct"]
        total = by_task[task_type]["total"]
        accuracy = correct / total if total else 0.0
        rows.append(
            {
                "task_type": task_type,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "accuracy_pct": accuracy * 100.0,
            }
        )
        total_correct += correct
        total_count += total
        macro_sum += accuracy

    overall_micro = total_correct / total_count if total_count else 0.0
    overall_macro = macro_sum / len(rows) if rows else 0.0

    return {
        "samples_file": str(samples_path),
        "task_type_results": rows,
        "overall": {
            "correct": total_correct,
            "total": total_count,
            "micro_accuracy": overall_micro,
            "micro_accuracy_pct": overall_micro * 100.0,
            "macro_accuracy": overall_macro,
            "macro_accuracy_pct": overall_macro * 100.0,
        },
    }


def print_table(summary: dict) -> None:
    rows = summary["task_type_results"]
    name_width = max(len("task_type"), max(len(row["task_type"]) for row in rows))
    count_width = max(len("correct"), max(len(str(row["correct"])) for row in rows))
    total_width = max(len("total"), max(len(str(row["total"])) for row in rows))

    print(f"Samples: {summary['samples_file']}")
    print(
        f"{'task_type':<{name_width}}  {'correct':>{count_width}}  "
        f"{'total':>{total_width}}  accuracy"
    )
    print(
        f"{'-' * name_width}  {'-' * count_width}  "
        f"{'-' * total_width}  --------"
    )
    for row in rows:
        print(
            f"{row['task_type']:<{name_width}}  "
            f"{row['correct']:>{count_width}}  "
            f"{row['total']:>{total_width}}  "
            f"{row['accuracy_pct']:7.2f}%"
        )

    overall = summary["overall"]
    print()
    print(
        "Overall micro accuracy: "
        f"{overall['correct']}/{overall['total']} = "
        f"{overall['micro_accuracy_pct']:.2f}%"
    )
    print(f"Overall macro accuracy: {overall['macro_accuracy_pct']:.2f}%")


def main() -> None:
    args = parse_args()
    samples_path = resolve_samples_path(args.input_path)
    summary = summarize(samples_path)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_table(summary)


if __name__ == "__main__":
    main()
