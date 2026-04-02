#!/usr/bin/env python3
"""Summarize LongVideoBench accuracy by duration group from lmms-eval sample logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute LongVideoBench accuracy grouped by duration_group. "
            "The input can be a run directory, a results.json file, or a "
            "samples_longvideobench*.jsonl file."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        help=(
            "Path(s) to a LongVideoBench run directory, *_results.json, "
            "or samples_longvideobench*.jsonl file."
        ),
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
        candidates = sorted(path.glob("*samples_longvideobench*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No samples_longvideobench*.jsonl file found in {path}")
        if len(candidates) > 1:
            raise ValueError(
                "Found multiple LongVideoBench sample logs in "
                f"{path}: {', '.join(candidate.name for candidate in candidates)}"
            )
        return candidates[0]

    if path.name.endswith(".jsonl"):
        return path

    if path.name.endswith("_results.json"):
        parent = path.parent
        stem = path.name[: -len("_results.json")]
        candidates = sorted(parent.glob(f"{stem}_samples_longvideobench*.jsonl"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                "Found multiple matching sibling LongVideoBench sample logs next to "
                f"{path}: {', '.join(candidate.name for candidate in candidates)}"
            )

        candidates = sorted(parent.glob("*samples_longvideobench*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find a sibling samples_longvideobench*.jsonl next to {path}"
            )
        if len(candidates) > 1:
            raise ValueError(
                "Found multiple sibling LongVideoBench sample logs next to "
                f"{path}: {', '.join(candidate.name for candidate in candidates)}"
            )
        return candidates[0]

    raise ValueError(
        "Unsupported input path. Expected a directory, *_results.json, or "
        "samples_longvideobench*.jsonl file."
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
    by_duration = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in load_records(samples_path):
        payload = record.get("lvb_acc")
        if not isinstance(payload, dict):
            continue

        duration_group = payload.get("duration_group")
        answer = payload.get("answer")
        parsed_pred = payload.get("parsed_pred")
        if duration_group is None or answer is None or parsed_pred is None:
            continue

        by_duration[duration_group]["total"] += 1
        by_duration[duration_group]["correct"] += int(parsed_pred == answer)

    if not by_duration:
        raise ValueError(
            f"No valid LongVideoBench records with lvb_acc found in {samples_path}"
        )

    rows = []
    total_correct = 0
    total_count = 0
    macro_sum = 0.0
    for duration_group in sorted(by_duration):
        correct = by_duration[duration_group]["correct"]
        total = by_duration[duration_group]["total"]
        accuracy = correct / total if total else 0.0
        rows.append(
            {
                "duration_group": duration_group,
                "duration_group_seconds": duration_group,
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
        "duration_group_results": rows,
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
    rows = summary["duration_group_results"]
    duration_width = max(len("duration_s"), max(len(str(row["duration_group"])) for row in rows))
    count_width = max(len("correct"), max(len(str(row["correct"])) for row in rows))
    total_width = max(len("total"), max(len(str(row["total"])) for row in rows))

    print(f"Samples: {summary['samples_file']}")
    print(
        f"{'duration_s':<{duration_width}}  {'correct':>{count_width}}  "
        f"{'total':>{total_width}}  accuracy"
    )
    print(
        f"{'-' * duration_width}  {'-' * count_width}  "
        f"{'-' * total_width}  --------"
    )
    for row in rows:
        print(
            f"{row['duration_group']:<{duration_width}}  "
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
    summaries = [summarize(resolve_samples_path(path)) for path in args.input_path]

    if args.json:
        output = summaries[0] if len(summaries) == 1 else summaries
        print(json.dumps(output, indent=2, sort_keys=True))
        return

    for index, summary in enumerate(summaries):
        if index:
            print()
        print_table(summary)


if __name__ == "__main__":
    main()
