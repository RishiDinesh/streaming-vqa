#!/usr/bin/env python3
"""Merge StreamingTom chunk results and combine with existing 4-method full-eval results.

Usage
-----
All 6 methods, full comparison + plots:

    python streaming/merge_all_results.py \\
        --rekv-results-dir   outputs/evaluations_streaming/rvs-ego/full_eval/run1 \\
        --st-results-dir     outputs/evaluations_streaming/rvs-ego/full_eval/run2 \\
        --output-dir         outputs/evaluations_streaming/rvs-ego/full_eval/merged_all

ST-only merge (no existing ReKV results):

    python streaming/merge_all_results.py \\
        --st-results-dir  outputs/evaluations_streaming/rvs-ego/full_eval/run2 \\
        --output-dir      outputs/evaluations_streaming/rvs-ego/full_eval/merged_st_only

ReKV-only merge:

    python streaming/merge_all_results.py \\
        --rekv-results-dir  outputs/evaluations_streaming/rvs-ego/full_eval/run1 \\
        --output-dir        outputs/evaluations_streaming/rvs-ego/full_eval/merged_rekv_only

Output layout
-------------
<output-dir>/
  merged/
    full_streaming.json
    duo_streaming.json
    rekv.json
    duo_plus_rekv.json
    streamingtom.json
    duo_plus_streamingtom.json
  comparison/      <- compare_subsamples.py output (summary.md, CSV, stability)
  plots/           <- plot_results.py output (PNG charts)
  plots_judge/     <- same plots after judge scoring (if --run-judge)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from streaming.ReKV.run_eval import normalize_result_payload_schema, summarize_aggregate_metrics

# ── All 6 canonical method names ─────────────────────────────────────────────
REKV_METHODS = ["full_streaming", "duo_streaming", "rekv", "duo_plus_rekv"]
ST_METHODS = ["streamingtom", "duo_plus_streamingtom"]
ALL_METHODS = REKV_METHODS + ST_METHODS


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_module(module: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *extra_args]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ── ReKV method: find the single result JSON under <dir>/<method>/ ────────────
def _find_rekv_result(results_dir: Path, method: str) -> Path | None:
    method_dir = results_dir / method
    if not method_dir.is_dir():
        return None
    # Accept chunk_*.json (sharded) or *_results.json (single-file run)
    chunks = sorted(method_dir.glob("chunk_*.json"))
    if chunks:
        return None  # sharded — use merge path
    singles = sorted(method_dir.glob("*_results.json"))
    if singles:
        return singles[-1]  # most recent
    # flat dir with result JSONs
    any_json = sorted(method_dir.glob("*.json"))
    if any_json:
        return any_json[-1]
    return None


def _find_rekv_shards(results_dir: Path, method: str) -> list[Path]:
    method_dir = results_dir / method
    if not method_dir.is_dir():
        # Maybe shards under results_dir/shards/chunk*/method/
        shard_pattern = list(results_dir.glob(f"shards/chunk*/{method}/*.json"))
        if shard_pattern:
            return sorted(shard_pattern)
        return []
    return sorted(method_dir.glob("chunk_*.json"))


def merge_rekv_method(results_dir: Path, method: str) -> dict[str, Any] | None:
    """Load a ReKV method result, handling both single-file and sharded layouts."""
    shards = _find_rekv_shards(results_dir, method)
    if shards:
        print(f"  [rekv] {method}: merging {len(shards)} shards from {results_dir / method}")
        chunks = [_load_json(p) for p in shards]
        run_config = dict(chunks[0].get("run_config", {}))
        evaluation_manifest = dict(chunks[0].get("evaluation_manifest", {}))
        videos: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in chunks:
            for video in chunk.get("videos", []):
                sid = str(video.get("sample_id"))
                if sid not in seen:
                    seen.add(sid)
                    videos.append(video)
        payload: dict[str, Any] = {
            "run_config": run_config,
            "evaluation_manifest": evaluation_manifest,
            "aggregate_metrics": summarize_aggregate_metrics(videos),
            "videos": videos,
        }
        return normalize_result_payload_schema(payload)

    single = _find_rekv_result(results_dir, method)
    if single:
        print(f"  [rekv] {method}: loading single result {single}")
        return normalize_result_payload_schema(_load_json(single))

    print(f"  [rekv] {method}: no results found under {results_dir}")
    return None


def merge_st_method(st_dir: Path, method: str) -> dict[str, Any] | None:
    """Merge StreamingTom chunk results for one method."""
    method_dir = st_dir / method
    chunk_paths = sorted(method_dir.glob("chunk_*.json"))
    if not chunk_paths:
        print(f"  [st]   {method}: no chunk files found under {method_dir}")
        return None

    print(f"  [st]   {method}: merging {len(chunk_paths)} chunks from {method_dir}")
    chunks = [_load_json(p) for p in chunk_paths]
    run_config = dict(chunks[0].get("run_config", {}))
    evaluation_manifest = dict(chunks[0].get("evaluation_manifest", {}))
    videos: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        for video in chunk.get("videos", []):
            sid = str(video.get("sample_id"))
            if sid not in seen:
                seen.add(sid)
                videos.append(video)

    payload: dict[str, Any] = {
        "run_config": run_config,
        "evaluation_manifest": evaluation_manifest,
        "aggregate_metrics": summarize_aggregate_metrics(videos),
        "videos": videos,
    }
    return normalize_result_payload_schema(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge streaming eval results for all 6 methods and produce comparison + plots."
        )
    )
    parser.add_argument(
        "--rekv-results-dir",
        type=Path,
        default=None,
        help=(
            "Root directory of the 4 existing ReKV method results. "
            "Expected layout: <dir>/<method>/ with chunk_*.json or *_results.json files."
        ),
    )
    parser.add_argument(
        "--st-results-dir",
        type=Path,
        default=None,
        help=(
            "Root directory of the 2 StreamingTom method chunk results. "
            "Expected layout: <dir>/streamingtom/chunk_*.json "
            "and <dir>/duo_plus_streamingtom/chunk_*.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for merged JSONs, comparison, and plot artifacts.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Subset of methods to process. Defaults to all available based on "
            "--rekv-results-dir and --st-results-dir."
        ),
    )
    parser.add_argument(
        "--run-judge",
        action="store_true",
        help="Run judge scoring on merged results and regenerate plots_judge.",
    )
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.rekv_results_dir is None and args.st_results_dir is None:
        print("[error] At least one of --rekv-results-dir or --st-results-dir is required.", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser().resolve(strict=False)
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Determine which methods to process
    requested_methods: list[str]
    if args.methods:
        requested_methods = args.methods
    else:
        requested_methods = []
        if args.rekv_results_dir is not None:
            requested_methods.extend(REKV_METHODS)
        if args.st_results_dir is not None:
            requested_methods.extend(ST_METHODS)

    print(f"[merge] Processing methods: {requested_methods}")

    merged_paths: list[str] = []

    for method in requested_methods:
        out_path = merged_dir / f"{method}.json"

        payload: dict[str, Any] | None = None

        if method in REKV_METHODS and args.rekv_results_dir is not None:
            rekv_dir = args.rekv_results_dir.expanduser().resolve(strict=False)
            payload = merge_rekv_method(rekv_dir, method)
        elif method in ST_METHODS and args.st_results_dir is not None:
            st_dir = args.st_results_dir.expanduser().resolve(strict=False)
            payload = merge_st_method(st_dir, method)
        else:
            print(f"  [skip] {method}: no source directory provided")
            continue

        if payload is None:
            print(f"  [warn] {method}: skipping — no results found")
            continue

        n_videos = len(payload.get("videos", []))
        n_convs = sum(len(v.get("conversations", [])) for v in payload.get("videos", []))
        agg = payload.get("aggregate_metrics", {})
        print(
            f"  [ok]  {method}: {n_videos} videos, {n_convs} conversations | "
            f"rouge_l={agg.get('avg_rouge_l_f1')} "
            f"judge={agg.get('avg_judge_score')} "
            f"latency={agg.get('avg_answer_latency_sec')}"
        )
        _save_json(out_path, payload)
        merged_paths.append(str(out_path))

    if not merged_paths:
        print("[error] No merged result files produced.", file=sys.stderr)
        return 1

    print(f"\n[merge] Merged {len(merged_paths)} method(s) → {merged_dir}")
    for p in merged_paths:
        print(f"  {p}")

    # ── compare_subsamples ───────────────────────────────────────────────────
    if not args.skip_compare:
        compare_dir = output_dir / "comparison"
        compare_dir.mkdir(parents=True, exist_ok=True)
        _run_module(
            "streaming.ReKV.compare_subsamples",
            [*merged_paths, "--output-dir", str(compare_dir)],
        )

    # ── plot_results ─────────────────────────────────────────────────────────
    if not args.skip_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        _run_module(
            "streaming.ReKV.plot_results",
            [*merged_paths, "--output-dir", str(plots_dir)],
        )

    # ── judge + judge plots ───────────────────────────────────────────────────
    if args.run_judge:
        _run_module(
            "streaming.ReKV.judge_results",
            ["--in-place", *merged_paths],
        )
        plots_judge_dir = output_dir / "plots_judge"
        plots_judge_dir.mkdir(parents=True, exist_ok=True)
        _run_module(
            "streaming.ReKV.plot_results",
            [*merged_paths, "--output-dir", str(plots_judge_dir)],
        )

    print(f"\n[merge] Done. Artifacts under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
