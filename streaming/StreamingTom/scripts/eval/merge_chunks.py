#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from streaming.ReKV.run_eval import normalize_result_payload_schema, summarize_aggregate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge chunked StreamingTom eval outputs into run2-style merged/comparison/plots artifacts."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["streamingtom", "duo_plus_streamingtom"],
    )
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-plots-judge", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def merge_method(run_root: Path, method: str) -> Path:
    method_dir = run_root / method
    chunk_paths = sorted(method_dir.glob("chunk_*.json"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found for method={method} under {method_dir}")

    chunks = [_load_json(path) for path in chunk_paths]
    run_config = dict(chunks[0].get("run_config", {}))
    evaluation_manifest = dict(chunks[0].get("evaluation_manifest", {}))

    videos: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for chunk in chunks:
        for video in chunk.get("videos", []):
            sample_id = str(video.get("sample_id"))
            if sample_id in seen_sample_ids:
                continue
            seen_sample_ids.add(sample_id)
            videos.append(video)

    merged_payload = {
        "run_config": run_config,
        "evaluation_manifest": evaluation_manifest,
        "aggregate_metrics": summarize_aggregate_metrics(videos),
        "videos": videos,
    }
    merged_payload = normalize_result_payload_schema(merged_payload)

    merged_path = run_root / "merged" / f"{method}.json"
    _save_json(merged_path, merged_payload)
    return merged_path


def _run_python_module(module_name: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module_name, *args]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve(strict=False)
    run_root.mkdir(parents=True, exist_ok=True)

    merged_paths = [str(merge_method(run_root, method)) for method in args.methods]
    print("Merged methods:")
    for path in merged_paths:
        print(f"  {path}")

    if not args.skip_compare:
        _run_python_module(
            "streaming.ReKV.compare_subsamples",
            [*merged_paths, "--output-dir", str(run_root / "comparison")],
        )

    if not args.skip_plots:
        _run_python_module(
            "streaming.ReKV.plot_results",
            [*merged_paths, "--output-dir", str(run_root / "plots")],
        )

    if not args.skip_plots_judge:
        _run_python_module(
            "streaming.ReKV.plot_results",
            [*merged_paths, "--output-dir", str(run_root / "plots_judge")],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
