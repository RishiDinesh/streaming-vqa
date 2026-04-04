#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import aggregate_score_bundles, open_ended_score_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill richer open-ended scoring into existing streaming result JSON files."
    )
    parser.add_argument("result_paths", nargs="+", help="One or more *_results.json files.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Update the input files in place. Otherwise writes *_rescored.json next to each input.",
    )
    return parser.parse_args()


def rescore_payload(payload: dict) -> dict:
    score_bundles: list[dict[str, float]] = []
    for video in payload.get("videos", []):
        for conversation in video.get("conversations", []):
            prediction = conversation.get("prediction", "")
            reference = conversation.get("reference_answer", "")
            scores = open_ended_score_bundle(prediction, reference)
            conversation["scores"] = scores
            conversation["normalized_exact_match"] = scores["normalized_exact_match"]
            score_bundles.append(scores)

    aggregate_metrics = dict(payload.get("aggregate_metrics", {}))
    aggregate_metrics.update(aggregate_score_bundles(score_bundles))
    if aggregate_metrics.get("avg_normalized_exact_match") is not None:
        aggregate_metrics["normalized_exact_match"] = aggregate_metrics["avg_normalized_exact_match"]
    aggregate_metrics["evaluation_mode"] = "open_ended_bundle"
    payload["aggregate_metrics"] = aggregate_metrics
    return payload


def main() -> int:
    args = parse_args()
    for raw_path in args.result_paths:
        path = Path(raw_path).expanduser().resolve()
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rescored = rescore_payload(payload)
        if args.in_place:
            output_path = path
        else:
            output_path = path.with_name(f"{path.stem}_rescored{path.suffix}")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(rescored, handle, indent=2)
        print(f"Saved rescored results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
