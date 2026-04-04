#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .common import aggregate_score_bundles, open_ended_score_bundle
from .datasets import RVS_DATASET_CONFIGS, build_dataset_from_args, sample_video_frames
from .feature_cache import (
    FEATURE_CACHE_VERSION,
    load_cached_feature_video,
    load_feature_cache_manifest,
)
from .methods import DEFAULT_DUO_ATTN_DIR, build_method_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal streaming RVS evaluation for LLaVA-OneVision 0.5B using "
            "either DuoAttention or ReKV."
        )
    )
    parser.add_argument("--dataset", default="rvs_ego", choices=sorted(RVS_DATASET_CONFIGS))
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--hf-repo-id", default="Becomebright/RVS")
    parser.add_argument("--allow-hf-video-download", action="store_true")
    parser.add_argument(
        "--model",
        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    )
    parser.add_argument(
        "--method",
        default="duo_streaming",
        choices=["full_streaming", "duo_streaming", "rekv", "duo_plus_rekv"],
    )
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--video-index", type=int, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--subsample-name", default=None)
    parser.add_argument("--max-conversations-per-video", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--feature-cache-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--flush-every-videos", type=int, default=1)
    parser.add_argument("--disable-progress-bar", action="store_true")

    parser.add_argument("--attn-dir", default=DEFAULT_DUO_ATTN_DIR)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--deploy-sink-size", type=int, default=None)
    parser.add_argument("--deploy-recent-size", type=int, default=None)

    parser.add_argument("--n-local", type=int, default=15000)
    parser.add_argument("--retrieve-size", type=int, default=64)
    parser.add_argument("--retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--n-frame-tokens", type=int, default=196)
    parser.add_argument("--rekv-fattn", action="store_true")
    parser.add_argument("--disable-rekv-pin-memory", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slugify(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(args.model)
    base_dir = Path("outputs") / "evaluations_streaming" / args.dataset.replace("_", "-")
    if args.subsample_name:
        base_dir = base_dir / slugify(args.subsample_name)
    elif args.max_videos is not None:
        base_dir = base_dir / f"subsample{args.max_videos}"
    return base_dir / args.method / model_slug / f"{timestamp}_results.json"


def conversation_target_frame_count(end_time_sec: float, sample_fps: float) -> int:
    if end_time_sec <= 0:
        return 0
    return max(int(end_time_sec * sample_fps), 0)


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "annotation_path": args.annotation_path,
        "video_root": args.video_root,
        "hf_repo_id": args.hf_repo_id,
        "allow_hf_video_download": args.allow_hf_video_download,
        "model": args.model,
        "method": args.method,
        "sample_fps": args.sample_fps,
        "max_videos": args.max_videos,
        "video_offset": args.video_offset,
        "video_index": args.video_index,
        "video_id": args.video_id,
        "subsample_name": args.subsample_name,
        "max_conversations_per_video": args.max_conversations_per_video,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "attn_dir": args.attn_dir,
        "sparsity": args.sparsity,
        "threshold": args.threshold,
        "deploy_sink_size": args.deploy_sink_size,
        "deploy_recent_size": args.deploy_recent_size,
        "n_local": args.n_local,
        "retrieve_size": args.retrieve_size,
        "retrieve_chunk_size": args.retrieve_chunk_size,
        "n_frame_tokens": args.n_frame_tokens,
        "rekv_fattn": args.rekv_fattn,
        "disable_rekv_pin_memory": args.disable_rekv_pin_memory,
        "feature_cache_root": args.feature_cache_root,
        "ingest_source": "cached_features" if args.feature_cache_root else "raw_frames",
    }


def write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(output_path)


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


def validate_resume_payload(existing_payload: dict[str, Any], run_config: dict[str, Any]) -> None:
    existing_run_config = existing_payload.get("run_config", {})
    mismatches: list[str] = []
    for key, current_value in run_config.items():
        if existing_run_config.get(key) != current_value:
            mismatches.append(
                f"{key}: existing={existing_run_config.get(key)!r} current={current_value!r}"
            )
    if mismatches:
        mismatch_text = "\n".join(mismatches[:12])
        raise ValueError(
            "Existing output file does not match the current run configuration.\n"
            f"{mismatch_text}"
        )


def evaluate_samples(
    *,
    samples,
    method,
    sample_fps: float,
    run_config: dict[str, Any],
    existing_videos: list[dict[str, Any]] | None = None,
    total_requested_videos: int | None = None,
    started_at_utc: str | None = None,
    checkpoint_path: Path | None = None,
    flush_every_videos: int = 1,
    show_progress_bar: bool = True,
    feature_cache_root: Path | None = None,
) -> dict[str, Any]:
    video_results: list[dict[str, Any]] = list(existing_videos or [])
    completed_sample_ids = {video["sample_id"] for video in video_results}
    pending_samples = [sample for sample in samples if sample.sample_id not in completed_sample_ids]
    if total_requested_videos is None:
        total_requested_videos = len(samples)
    if started_at_utc is None:
        started_at_utc = utc_now_iso()

    iterable = pending_samples
    progress = None
    if show_progress_bar and tqdm is not None:
        progress = tqdm(
            pending_samples,
            total=len(pending_samples),
            desc=f"{run_config['method']} videos",
            unit="video",
        )
        iterable = progress

    newly_completed = 0
    for sample in iterable:
        ingest_source = "raw_frames"
        sampled_frame_indices_total: list[int]
        sampled_timestamps_total: list[float]
        sampled_native_fps: float
        sampled_base_fps: int
        sampled_total_frames: int
        feature_cache_path: str | None = None
        cached_video = None
        sampled_video = None
        if feature_cache_root is not None:
            cached_video = load_cached_feature_video(
                feature_cache_root,
                sample_id=sample.sample_id,
                video_id=sample.video_id,
                sample_fps=sample_fps,
            )
            ingest_source = "cached_features"
            feature_cache_path = cached_video.cache_path
            sampled_frame_indices_total = list(cached_video.sampled_frame_indices)
            sampled_timestamps_total = list(cached_video.sampled_timestamps_sec)
            sampled_native_fps = cached_video.native_fps
            sampled_base_fps = cached_video.sampling_base_fps
            sampled_total_frames = len(cached_video.sampled_frame_indices)
        else:
            sampled_video = sample_video_frames(
                sample.video_path,
                sample_fps,
                duration_sec=sample.duration,
            )
            sampled_frame_indices_total = list(sampled_video.sampled_frame_indices)
            sampled_timestamps_total = list(sampled_video.sampled_timestamps_sec)
            sampled_native_fps = sampled_video.native_fps
            sampled_base_fps = sampled_video.sampling_base_fps
            sampled_total_frames = len(sampled_video.sampled_frame_indices)

        method.reset(
            {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "video_path": sample.video_path,
                "duration": sample.duration,
                "ingest_source": ingest_source,
                "feature_cache_path": feature_cache_path,
            }
        )

        ingested_until_idx = -1
        conversation_results: list[dict[str, Any]] = []

        for conversation in sample.conversations:
            target_frame_count = min(
                conversation_target_frame_count(conversation.end_time, sample_fps),
                sampled_total_frames,
            )
            new_indices = list(range(ingested_until_idx + 1, target_frame_count))
            ingest_records: list[dict[str, Any]] = []
            if cached_video is not None:
                for idx in new_indices:
                    ingest_record = method.ingest_features(
                        cached_video.get_feature(idx),
                        cached_video.sampled_timestamps_sec[idx],
                    )
                    ingest_records.append(ingest_record)
                    ingested_until_idx = idx
            elif sampled_video is not None and new_indices:
                decoded_frames = sampled_video.get_frames(new_indices)
                for batch_offset, idx in enumerate(new_indices):
                    ingest_record = method.ingest_frame(
                        decoded_frames[batch_offset],
                        sampled_video.sampled_timestamps_sec[idx],
                    )
                    ingest_records.append(ingest_record)
                    ingested_until_idx = idx

            answer = method.answer_question(
                conversation.question,
                metadata={
                    "start_time": conversation.start_time,
                    "end_time": conversation.end_time,
                },
            )
            score_bundle = open_ended_score_bundle(answer.prediction, conversation.answer)
            conversation_results.append(
                {
                    "question": conversation.question,
                    "reference_answer": conversation.answer,
                    "prediction": answer.prediction,
                    "normalized_exact_match": score_bundle["normalized_exact_match"],
                    "scores": score_bundle,
                    "start_time": conversation.start_time,
                    "end_time": conversation.end_time,
                    "num_frames_ingested_before_answer": method.frames_ingested,
                    "sampled_timestamps_sec_so_far": list(method.ingested_timestamps_sec),
                    "new_frame_timestamps_sec": [
                        float(record["timestamp_sec"]) for record in ingest_records
                    ],
                    "method_stats": answer.stats,
                }
            )

        video_results.append(
            {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "video_path": sample.video_path,
                "duration": sample.duration,
                "sample_fps": sample_fps,
                "native_fps": sampled_native_fps,
                "sampling_base_fps": sampled_base_fps,
                "num_sampled_frames_total": sampled_total_frames,
                "sampled_frame_indices_total": sampled_frame_indices_total,
                "sampled_timestamps_sec_total": sampled_timestamps_total,
                "ingest_source": ingest_source,
                "feature_cache_path": feature_cache_path,
                "conversations": conversation_results,
                "runtime_stats": method.get_runtime_stats(),
            }
        )
        newly_completed += 1

        if progress is not None:
            progress.set_postfix_str(
                f"video_id={sample.video_id} convs={len(conversation_results)} completed={len(video_results)}/{total_requested_videos}"
            )
        else:
            print(
                f"[video {len(video_results)}/{total_requested_videos}] "
                f"{sample.video_id} convs={len(conversation_results)}"
            )

        if checkpoint_path is not None and flush_every_videos > 0 and newly_completed % flush_every_videos == 0:
            checkpoint_payload = build_result_payload(
                run_config=run_config,
                video_results=video_results,
                started_at_utc=started_at_utc,
                total_requested_videos=total_requested_videos,
                status="in_progress",
            )
            write_json_atomic(checkpoint_payload, checkpoint_path)
            print(
                f"[checkpoint] saved {len(video_results)}/{total_requested_videos} videos to {checkpoint_path}"
            )

    if progress is not None:
        progress.close()
    return build_result_payload(
        run_config=run_config,
        video_results=video_results,
        started_at_utc=started_at_utc,
        total_requested_videos=total_requested_videos,
        status="completed",
    )


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    run_config = build_run_config(args)
    feature_cache_root = Path(args.feature_cache_root) if args.feature_cache_root else None
    if feature_cache_root is not None:
        cache_manifest = load_feature_cache_manifest(feature_cache_root)
        cache_dataset = str(cache_manifest.get("dataset"))
        cache_model = str(cache_manifest.get("model"))
        cache_sample_fps = float(cache_manifest.get("sample_fps"))
        cache_version = str(cache_manifest.get("cache_version"))
        if cache_version != FEATURE_CACHE_VERSION:
            raise ValueError(
                f"Feature cache version mismatch: expected {FEATURE_CACHE_VERSION}, got {cache_version}"
            )
        if cache_dataset != args.dataset:
            raise ValueError(
                f"Feature cache dataset mismatch: expected {args.dataset!r}, got {cache_dataset!r}"
            )
        if cache_model != args.model:
            raise ValueError(
                f"Feature cache model mismatch: expected {args.model!r}, got {cache_model!r}"
            )
        if abs(cache_sample_fps - float(args.sample_fps)) > 1e-9:
            raise ValueError(
                f"Feature cache sample_fps mismatch: expected {args.sample_fps}, got {cache_sample_fps}"
            )

    dataset = build_dataset_from_args(args)
    samples = dataset.load(
        video_id=args.video_id,
        video_index=args.video_index,
        video_offset=args.video_offset,
        max_videos=args.max_videos,
    )
    if not samples:
        raise ValueError("No samples matched the requested dataset/video filters.")

    if args.max_conversations_per_video is not None:
        samples = [
            type(sample)(
                sample_id=sample.sample_id,
                video_id=sample.video_id,
                video_path=sample.video_path,
                duration=sample.duration,
                conversations=sample.conversations[: args.max_conversations_per_video],
                extra_metadata=sample.extra_metadata,
            )
            for sample in samples
        ]

    existing_videos: list[dict[str, Any]] = []
    started_at_utc = utc_now_iso()
    checkpoint_path = Path(args.output_path) if args.output_path else default_output_path(args)

    if checkpoint_path.exists():
        if args.resume:
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                existing_payload = json.load(handle)
            validate_resume_payload(existing_payload, run_config)
            existing_videos = existing_payload.get("videos", [])
            started_at_utc = (
                existing_payload.get("run_state", {}).get("started_at_utc") or started_at_utc
            )
            completed_sample_ids = {video["sample_id"] for video in existing_videos}
            if len(completed_sample_ids) == len(samples):
                print(f"All requested videos already completed in {checkpoint_path}")
                return existing_payload
            print(
                f"Resuming {run_config['method']} from {checkpoint_path}: "
                f"{len(existing_videos)}/{len(samples)} videos already completed."
            )
        elif not args.overwrite_output:
            raise FileExistsError(
                f"Output already exists: {checkpoint_path}. "
                "Use --resume to continue it or --overwrite-output to replace it."
            )

    method = build_method_from_args(args)

    return evaluate_samples(
        samples=samples,
        method=method,
        sample_fps=args.sample_fps,
        run_config=run_config,
        existing_videos=existing_videos,
        total_requested_videos=len(samples),
        started_at_utc=started_at_utc,
        checkpoint_path=checkpoint_path,
        flush_every_videos=args.flush_every_videos,
        show_progress_bar=not args.disable_progress_bar,
        feature_cache_root=feature_cache_root,
    )


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    if args.resume and args.output_path is None:
        raise ValueError("--resume requires an explicit --output-path so the runner knows which file to continue.")
    if args.flush_every_videos <= 0:
        raise ValueError("--flush-every-videos must be >= 1.")

    output_path = Path(args.output_path) if args.output_path else default_output_path(args)
    args.output_path = str(output_path)
    if args.feature_cache_root is not None:
        args.feature_cache_root = str(Path(args.feature_cache_root).expanduser().resolve(strict=False))
    results = run_eval(args)
    write_json_atomic(results, Path(args.output_path))
    print(
        f"Saved results to {args.output_path} "
        f"({results.get('run_state', {}).get('completed_videos')}/"
        f"{results.get('run_state', {}).get('total_requested_videos')} videos)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
