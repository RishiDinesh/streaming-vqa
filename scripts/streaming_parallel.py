#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RVS_REPO_ID = "Becomebright/RVS"
FEATURE_CACHE_VERSION = "v1"

RVS_DATASET_CONFIGS = {
    "rvs_ego": {
        "subset": "ego",
        "annotation": "ego/ego4d_oe.json",
        "label": "RVS-Ego",
    },
    "rvs_movie": {
        "subset": "movie",
        "annotation": "movie/movienet_oe.json",
        "label": "RVS-Movie",
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")


def feature_cache_path(cache_root: Path, sample_id: str) -> Path:
    return cache_root / "videos" / f"{slugify(sample_id)}.pt"


def write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(output_path)


def write_feature_cache_manifest(cache_root: Path, payload: dict[str, Any]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(payload, cache_root / "manifest.json")


def aggregate_score_bundles(score_bundles: list[dict[str, float]]) -> dict[str, float | None]:
    if not score_bundles:
        return {}
    metric_keys = sorted(score_bundles[0].keys())
    aggregated: dict[str, float | None] = {}
    for key in metric_keys:
        values = [bundle.get(key) for bundle in score_bundles if bundle.get(key) is not None]
        aggregated[f"avg_{key}"] = float(sum(values) / len(values)) if values else None
    if aggregated.get("avg_judge_score") is not None:
        aggregated["primary_quality_metric"] = "avg_judge_score"
        aggregated["primary_quality_score"] = aggregated["avg_judge_score"]
    elif aggregated.get("avg_rouge_l_f1") is not None:
        aggregated["primary_quality_metric"] = "avg_rouge_l_f1"
        aggregated["primary_quality_score"] = aggregated["avg_rouge_l_f1"]
    elif aggregated.get("avg_token_f1") is not None:
        aggregated["primary_quality_metric"] = "avg_token_f1"
        aggregated["primary_quality_score"] = aggregated["avg_token_f1"]
    return aggregated


def summarize_aggregate_metrics(video_results: list[dict[str, Any]]) -> dict[str, Any]:
    score_bundles: list[dict[str, float]] = []
    ttfts: list[float] = []
    answer_latencies: list[float] = []
    frame_ingest_latencies: list[float] = []
    peak_memories: list[int] = []
    total_frames_ingested = 0

    for video in video_results:
        runtime_stats = video.get("runtime_stats", {})
        avg_ingest = runtime_stats.get("avg_frame_ingest_latency_sec")
        frames_ingested = int(runtime_stats.get("frames_ingested", 0) or 0)
        if avg_ingest is not None and frames_ingested > 0:
            frame_ingest_latencies.extend([float(avg_ingest)] * frames_ingested)
        total_frames_ingested += frames_ingested

        for conversation in video.get("conversations", []):
            scores = conversation.get("scores")
            if scores:
                score_bundles.append(scores)
            method_stats = conversation.get("method_stats", {})
            if method_stats.get("ttft_sec") is not None:
                ttfts.append(float(method_stats["ttft_sec"]))
            if method_stats.get("answer_latency_sec") is not None:
                answer_latencies.append(float(method_stats["answer_latency_sec"]))
            if method_stats.get("peak_memory_bytes") is not None:
                peak_memories.append(int(method_stats["peak_memory_bytes"]))

    aggregate_metrics = {
        "avg_ttft_sec": float(sum(ttfts) / len(ttfts)) if ttfts else None,
        "avg_answer_latency_sec": (
            float(sum(answer_latencies) / len(answer_latencies)) if answer_latencies else None
        ),
        "avg_frame_ingest_latency_sec": (
            float(sum(frame_ingest_latencies) / len(frame_ingest_latencies))
            if frame_ingest_latencies
            else None
        ),
        "peak_memory_bytes": max(peak_memories) if peak_memories else None,
        "total_frames_ingested": total_frames_ingested,
        "total_conversations_answered": len(score_bundles),
        "evaluation_mode": "open_ended_bundle",
    }
    aggregate_metrics.update(aggregate_score_bundles(score_bundles))
    if aggregate_metrics.get("avg_normalized_exact_match") is not None:
        aggregate_metrics["normalized_exact_match"] = aggregate_metrics["avg_normalized_exact_match"]
    return aggregate_metrics


def build_result_payload(
    *,
    run_config: dict[str, Any],
    video_results: list[dict[str, Any]],
    started_at_utc: str,
    total_requested_videos: int,
    status: str,
) -> dict[str, Any]:
    return {
        "run_config": run_config,
        "run_state": {
            "status": status,
            "started_at_utc": started_at_utc,
            "updated_at_utc": utc_now_iso(),
            "completed_videos": len(video_results),
            "total_requested_videos": total_requested_videos,
            "completed_sample_ids": [video["sample_id"] for video in video_results],
        },
        "aggregate_metrics": summarize_aggregate_metrics(video_results),
        "videos": video_results,
    }


def add_dataset_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="rvs_ego", choices=sorted(RVS_DATASET_CONFIGS))
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--hf-repo-id", default=DEFAULT_RVS_REPO_ID)
    parser.add_argument("--allow-hf-video-download", action="store_true")
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-index", type=int, default=None)
    parser.add_argument("--video-id", default=None)


def resolve_annotation_file(args: argparse.Namespace) -> Path:
    if args.dataset not in RVS_DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if args.annotation_path:
        path = Path(args.annotation_path).expanduser().resolve(strict=False)
        if not path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required when --annotation-path is not provided."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            filename=str(RVS_DATASET_CONFIGS[args.dataset]["annotation"]),
        )
    )


def load_filtered_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    annotation_file = resolve_annotation_file(args)
    with open(annotation_file, "r", encoding="utf-8") as handle:
        raw_records = json.load(handle)

    if args.video_id is not None:
        raw_records = [record for record in raw_records if str(record["video_id"]) == args.video_id]
    if args.video_index is not None:
        raw_records = raw_records[args.video_index : args.video_index + 1]
    elif args.video_offset > 0:
        raw_records = raw_records[args.video_offset:]
    if args.max_videos is not None:
        raw_records = raw_records[: args.max_videos]
    return raw_records


def sample_ids_for_records(raw_records: list[dict[str, Any]]) -> list[str]:
    return [f"{str(record['video_id'])}-{index}" for index, record in enumerate(raw_records)]


def command_count_samples(args: argparse.Namespace) -> int:
    print(len(load_filtered_records(args)))
    return 0


def load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Shard output not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object payload at {path}")
    return payload


def merge_run_config(
    shard_payloads: list[dict[str, Any]],
    *,
    video_offset: int,
    max_videos: int | None,
) -> dict[str, Any]:
    base_run_config = dict(shard_payloads[0].get("run_config", {}))
    ignored_keys = {"video_offset", "max_videos"}

    for payload in shard_payloads[1:]:
        other_run_config = payload.get("run_config", {})
        for key in set(base_run_config) | set(other_run_config):
            if key in ignored_keys:
                continue
            if base_run_config.get(key) != other_run_config.get(key):
                raise ValueError(
                    "Shard run_config mismatch for "
                    f"{key}: {base_run_config.get(key)!r} vs {other_run_config.get(key)!r}"
                )

    base_run_config["video_offset"] = video_offset
    base_run_config["max_videos"] = max_videos
    return base_run_config


def command_merge_results(args: argparse.Namespace) -> int:
    shard_paths = [Path(path) for path in args.shard_path]
    if not shard_paths:
        raise ValueError("At least one --shard-path is required.")

    shard_payloads = [load_payload(path) for path in shard_paths]
    merged_videos: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    started_at_values: list[datetime] = []

    for path, payload in zip(shard_paths, shard_payloads):
        run_state = payload.get("run_state", {})
        status = str(run_state.get("status", ""))
        if status != "completed":
            raise ValueError(f"Shard output is not completed: {path} (status={status!r})")
        started_at_utc = run_state.get("started_at_utc")
        if started_at_utc:
            started_at_values.append(datetime.fromisoformat(str(started_at_utc)))
        for video in payload.get("videos", []):
            sample_id = str(video.get("sample_id"))
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id across shards: {sample_id}")
            seen_sample_ids.add(sample_id)
            merged_videos.append(video)

    if len(merged_videos) != args.total_requested_videos:
        raise ValueError(
            "Shard coverage mismatch: expected "
            f"{args.total_requested_videos} videos, found {len(merged_videos)} across shards."
        )

    started_at_utc = (
        min(started_at_values).astimezone(timezone.utc).isoformat()
        if started_at_values
        else utc_now_iso()
    )
    merged_run_config = merge_run_config(
        shard_payloads,
        video_offset=args.video_offset,
        max_videos=args.max_videos,
    )
    payload = build_result_payload(
        run_config=merged_run_config,
        video_results=merged_videos,
        started_at_utc=started_at_utc,
        total_requested_videos=args.total_requested_videos,
        status="completed",
    )
    output_path = Path(args.output_path)
    write_json_atomic(payload, output_path)
    print(
        f"Merged {len(merged_videos)} videos from {len(shard_paths)} shard(s) into {output_path}"
    )
    return 0


def command_write_cache_manifest(args: argparse.Namespace) -> int:
    raw_records = load_filtered_records(args)
    sample_ids = sample_ids_for_records(raw_records)
    cache_root = Path(args.feature_cache_root).expanduser().resolve(strict=False)
    cached_sample_ids = [
        sample_id for sample_id in sample_ids if feature_cache_path(cache_root, sample_id).is_file()
    ]
    manifest = {
        "cache_version": FEATURE_CACHE_VERSION,
        "created_at_utc": utc_now_iso(),
        "dataset": args.dataset,
        "model": args.model,
        "sample_fps": float(args.sample_fps),
        "feature_dtype": "bfloat16",
        "feature_batch_size": int(args.feature_batch_size),
        "num_cached_videos": len(cached_sample_ids),
        "cached_sample_ids": cached_sample_ids,
    }
    write_feature_cache_manifest(cache_root, manifest)
    print(
        f"Wrote cache manifest for {len(cached_sample_ids)}/{len(sample_ids)} videos to {cache_root}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Helper utilities for launcher-level data parallelism in the streaming "
            "evaluation scripts."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    count_parser = subparsers.add_parser(
        "count-samples",
        help="Count videos after applying the same dataset filters used by the streaming runner.",
    )
    add_dataset_selection_args(count_parser)
    count_parser.set_defaults(func=command_count_samples)

    merge_parser = subparsers.add_parser(
        "merge-results",
        help="Merge completed per-shard streaming eval JSONs into one final result file.",
    )
    merge_parser.add_argument("--output-path", required=True)
    merge_parser.add_argument("--shard-path", action="append", required=True)
    merge_parser.add_argument("--video-offset", type=int, default=0)
    merge_parser.add_argument("--max-videos", type=int, default=None)
    merge_parser.add_argument("--total-requested-videos", type=int, required=True)
    merge_parser.set_defaults(func=command_merge_results)

    manifest_parser = subparsers.add_parser(
        "write-cache-manifest",
        help="Rebuild the feature-cache manifest after parallel precompute shards finish.",
    )
    add_dataset_selection_args(manifest_parser)
    manifest_parser.add_argument("--model", required=True)
    manifest_parser.add_argument("--sample-fps", type=float, required=True)
    manifest_parser.add_argument("--feature-batch-size", type=int, required=True)
    manifest_parser.add_argument("--feature-cache-root", required=True)
    manifest_parser.set_defaults(func=command_write_cache_manifest)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
