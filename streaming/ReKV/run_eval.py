#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from bisect import bisect_left
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


def parse_args() -> argparse.Namespace:
    from .methods import DEFAULT_DUO_ATTN_DIR

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
        choices=["full_streaming", "duo_streaming", "rekv", "rekv_no_offload", "duo_plus_rekv"],
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
    parser.add_argument("--video-decode-threads", type=int, default=4)
    parser.add_argument("--clear-cuda-cache-on-reset", action="store_true")
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
    parser.add_argument("--flush-every-conversations", type=int, default=1)
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


def conversation_target_frame_count(
    end_time_sec: float,
    sampled_timestamps_sec: list[float],
) -> int:
    if end_time_sec <= 0:
        return 0
    # Strict causal availability: ingest exactly the sampled frames whose
    # timestamps are strictly earlier than the question cutoff.
    return bisect_left(sampled_timestamps_sec, float(end_time_sec))


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
        "video_decode_threads": args.video_decode_threads,
        "clear_cuda_cache_on_reset": args.clear_cuda_cache_on_reset,
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


def validate_comparison_run_config(run_config: dict[str, Any]) -> None:
    if float(run_config["sample_fps"]) <= 0:
        raise ValueError("sample_fps must be > 0 for comparable streaming evaluation.")
    if int(run_config["max_new_tokens"]) <= 0:
        raise ValueError("max_new_tokens must be > 0 for comparable streaming evaluation.")
    if int(run_config["video_decode_threads"]) <= 0:
        raise ValueError("video_decode_threads must be > 0.")
    if run_config["method"] in {"duo_streaming", "duo_plus_rekv"}:
        deploy_sink_size = run_config.get("deploy_sink_size")
        deploy_recent_size = run_config.get("deploy_recent_size")
        if deploy_sink_size is not None and int(deploy_sink_size) <= 0:
            raise ValueError("deploy_sink_size must be > 0 when provided.")
        if deploy_recent_size is not None and int(deploy_recent_size) <= 0:
            raise ValueError("deploy_recent_size must be > 0 when provided.")
        sparsity = run_config.get("sparsity")
        if sparsity is not None and not (0.0 <= float(sparsity) <= 1.0):
            raise ValueError("sparsity must be in [0, 1].")
    if run_config["method"] in {"rekv", "rekv_no_offload", "duo_plus_rekv"}:
        if int(run_config["n_local"]) <= 0:
            raise ValueError("n_local must be > 0 for ReKV-based methods.")
        if int(run_config["retrieve_size"]) <= 0:
            raise ValueError("retrieve_size must be > 0 for ReKV-based methods.")
        if int(run_config["retrieve_chunk_size"]) <= 0:
            raise ValueError("retrieve_chunk_size must be > 0 for ReKV-based methods.")
        if int(run_config["n_frame_tokens"]) <= 0:
            raise ValueError("n_frame_tokens must be > 0 for ReKV-based methods.")


def normalize_feature_cache_manifest(cache_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_version": cache_manifest.get("cache_version"),
        "dataset": cache_manifest.get("dataset"),
        "model": cache_manifest.get("model"),
        "sample_fps": cache_manifest.get("sample_fps"),
        "feature_token_count": cache_manifest.get("feature_token_count"),
        "num_videos": cache_manifest.get("num_videos"),
        "created_at_utc": cache_manifest.get("created_at_utc"),
    }


def build_evaluation_manifest(
    *,
    run_config: dict[str, Any],
    method_manifest: dict[str, Any],
    feature_cache_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "comparison_contract_version": "v1",
        "comparison_scope": "streaming_vqa_cross_method",
        "shared_run_settings": {
            "dataset": run_config.get("dataset"),
            "model": run_config.get("model"),
            "sample_fps": run_config.get("sample_fps"),
            "max_new_tokens": run_config.get("max_new_tokens"),
            "seed": run_config.get("seed"),
            "dtype_request": run_config.get("dtype"),
            "device_request": run_config.get("device"),
            "video_decode_threads": run_config.get("video_decode_threads"),
            "feature_cache_root": run_config.get("feature_cache_root"),
            "ingest_source": run_config.get("ingest_source"),
            "annotation_path": run_config.get("annotation_path"),
            "video_root": run_config.get("video_root"),
            "hf_repo_id": run_config.get("hf_repo_id"),
            "allow_hf_video_download": run_config.get("allow_hf_video_download"),
            "max_videos": run_config.get("max_videos"),
            "video_offset": run_config.get("video_offset"),
            "video_index": run_config.get("video_index"),
            "video_id": run_config.get("video_id"),
            "subsample_name": run_config.get("subsample_name"),
            "max_conversations_per_video": run_config.get("max_conversations_per_video"),
        },
        "streaming_protocol": {
            "causal_cutoff_policy": "sampled_timestamps_strictly_before_end_time",
            "frame_ingest_policy": "one_sampled_frame_per_forward_pass",
            "question_ordering": "dataset_loader_sorted_by_start_time_then_end_time",
            "shared_state_across_questions": True,
            "offline_full_video_prefill": False,
            "resume_requires_run_config_match": True,
            "feature_cache_requires_schedule_equivalence": True,
        },
        "method_manifest": method_manifest,
        "feature_cache_manifest": (
            normalize_feature_cache_manifest(feature_cache_manifest)
            if feature_cache_manifest is not None
            else None
        ),
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
    gpu_memory_currents: list[int] = []
    peak_memories: list[int] = []
    cpu_offload_currents: list[int] = []
    cpu_offload_peaks: list[int] = []
    retrieval_latencies: list[float] = []
    retrieved_block_counts: list[float] = []
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
            if method_stats.get("current_memory_bytes") is not None:
                gpu_memory_currents.append(int(method_stats["current_memory_bytes"]))
            if method_stats.get("peak_memory_bytes") is not None:
                peak_memories.append(int(method_stats["peak_memory_bytes"]))
            if method_stats.get("cpu_offload_bytes_current") is not None:
                cpu_offload_currents.append(int(method_stats["cpu_offload_bytes_current"]))
            if method_stats.get("cpu_offload_bytes_peak") is not None:
                cpu_offload_peaks.append(int(method_stats["cpu_offload_bytes_peak"]))
            if method_stats.get("retrieval_latency_sec") is not None:
                retrieval_latencies.append(float(method_stats["retrieval_latency_sec"]))
            if method_stats.get("avg_retrieved_block_count") is not None:
                retrieved_block_counts.append(float(method_stats["avg_retrieved_block_count"]))

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
        "avg_retrieval_latency_sec": (
            float(sum(retrieval_latencies) / len(retrieval_latencies)) if retrieval_latencies else None
        ),
        "avg_retrieved_block_count": (
            float(sum(retrieved_block_counts) / len(retrieved_block_counts))
            if retrieved_block_counts
            else None
        ),
        "avg_gpu_memory_bytes_current": (
            float(sum(gpu_memory_currents) / len(gpu_memory_currents))
            if gpu_memory_currents
            else None
        ),
        "peak_memory_bytes": max(peak_memories) if peak_memories else None,
        "avg_cpu_offload_bytes_current": (
            float(sum(cpu_offload_currents) / len(cpu_offload_currents))
            if cpu_offload_currents
            else None
        ),
        "peak_cpu_offload_bytes": max(cpu_offload_peaks) if cpu_offload_peaks else None,
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
    evaluation_manifest: dict[str, Any],
    video_results: list[dict[str, Any]],
    started_at_utc: str,
    total_requested_videos: int,
    status: str,
    in_progress_video: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_config": run_config,
        "evaluation_manifest": evaluation_manifest,
        "run_state": {
            "status": status,
            "started_at_utc": started_at_utc,
            "updated_at_utc": utc_now_iso(),
            "completed_videos": len(video_results),
            "total_requested_videos": total_requested_videos,
            "completed_sample_ids": [video["sample_id"] for video in video_results],
            "in_progress_sample_id": (
                str(in_progress_video["sample_id"]) if in_progress_video is not None else None
            ),
        },
        "aggregate_metrics": summarize_aggregate_metrics(video_results),
        "videos": video_results,
        "in_progress_video": in_progress_video,
    }


def _cumulative_ingest_latency_sec(runtime_stats: dict[str, Any]) -> float:
    cumulative = runtime_stats.get("cumulative_frame_ingest_latency_sec")
    if cumulative is not None:
        return float(cumulative)
    avg_latency = runtime_stats.get("avg_frame_ingest_latency_sec")
    frames_ingested = int(runtime_stats.get("frames_ingested", 0) or 0)
    if avg_latency is None or frames_ingested <= 0:
        return 0.0
    return float(avg_latency) * frames_ingested


def build_video_runtime_stats(
    *,
    method,
    frames_ingested_total: int,
    cumulative_ingest_latency_sec: float,
    last_ingested_timestamp_sec: float | None,
) -> dict[str, Any]:
    runtime_stats = dict(method.get_runtime_stats())
    runtime_stats["frames_ingested"] = int(frames_ingested_total)
    runtime_stats["avg_frame_ingest_latency_sec"] = (
        float(cumulative_ingest_latency_sec / frames_ingested_total)
        if frames_ingested_total > 0
        else None
    )
    runtime_stats["cumulative_frame_ingest_latency_sec"] = float(cumulative_ingest_latency_sec)
    runtime_stats["last_ingested_timestamp_sec"] = last_ingested_timestamp_sec
    return runtime_stats


def build_video_result(
    *,
    sample,
    sample_fps: float,
    ingest_source: str,
    feature_cache_path: str | None,
    sampled_native_fps: float,
    sampled_base_fps: int,
    sampled_total_frames: int,
    sampled_frame_indices_total: list[int],
    sampled_timestamps_total: list[float],
    conversation_results: list[dict[str, Any]],
    runtime_stats: dict[str, Any],
) -> dict[str, Any]:
    frames_ingested = int(runtime_stats.get("frames_ingested", 0) or 0)
    return {
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
        "extra_metadata": sample.extra_metadata,
        "conversations": conversation_results,
        "runtime_stats": runtime_stats,
        "checkpoint_state": {
            "completed_conversations": len(conversation_results),
            "frames_ingested": frames_ingested,
            "ingested_until_frame_index": (frames_ingested - 1) if frames_ingested > 0 else -1,
            "last_ingested_timestamp_sec": runtime_stats.get("last_ingested_timestamp_sec"),
        },
    }


def _replay_frames(
    *,
    method,
    cached_video,
    sampled_video,
    sampled_timestamps_total: list[float],
    end_frame_count: int,
) -> int:
    if end_frame_count <= 0:
        return -1
    replay_indices = list(range(end_frame_count))
    if cached_video is not None:
        for idx in replay_indices:
            method.ingest_features(
                cached_video.get_feature(idx),
                sampled_timestamps_total[idx],
            )
    elif sampled_video is not None:
        decoded_frames = sampled_video.get_frames(replay_indices)
        for batch_offset, idx in enumerate(replay_indices):
            method.ingest_frame(
                decoded_frames[batch_offset],
                sampled_timestamps_total[idx],
            )
    return end_frame_count - 1


def validate_resume_payload(
    existing_payload: dict[str, Any],
    run_config: dict[str, Any],
    evaluation_manifest: dict[str, Any] | None = None,
) -> None:
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
    if evaluation_manifest is not None:
        existing_manifest = existing_payload.get("evaluation_manifest")
        if existing_manifest != evaluation_manifest:
            raise ValueError(
                "Existing output file does not match the current evaluation manifest. "
                "Resume is only supported when comparison-critical settings are identical."
            )


def evaluate_samples(
    *,
    samples,
    method,
    sample_fps: float,
    run_config: dict[str, Any],
    evaluation_manifest: dict[str, Any],
    existing_videos: list[dict[str, Any]] | None = None,
    total_requested_videos: int | None = None,
    started_at_utc: str | None = None,
    checkpoint_path: Path | None = None,
    flush_every_videos: int = 1,
    flush_every_conversations: int = 1,
    show_progress_bar: bool = True,
    feature_cache_root: Path | None = None,
    existing_in_progress_video: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video_results: list[dict[str, Any]] = list(existing_videos or [])
    completed_sample_ids = {video["sample_id"] for video in video_results}
    in_progress_sample_id = (
        str(existing_in_progress_video["sample_id"])
        if existing_in_progress_video is not None
        else None
    )
    pending_samples = [
        sample
        for sample in samples
        if sample.sample_id not in completed_sample_ids
        and sample.sample_id != in_progress_sample_id
    ]
    if existing_in_progress_video is not None:
        matching_samples = [sample for sample in samples if sample.sample_id == in_progress_sample_id]
        if not matching_samples:
            raise ValueError(
                f"Resume payload references an in-progress sample that is not present in this run: {in_progress_sample_id}"
            )
        pending_samples = [matching_samples[0], *pending_samples]
    if total_requested_videos is None:
        total_requested_videos = len(samples)
    if started_at_utc is None:
        started_at_utc = utc_now_iso()

    progress = None
    if show_progress_bar and tqdm is not None:
        progress = tqdm(
            total=total_requested_videos,
            initial=len(video_results),
            desc=f"{run_config['method']} videos",
            unit="video",
        )

    newly_completed = 0
    for sample in pending_samples:
        partial_video_result = (
            dict(existing_in_progress_video)
            if existing_in_progress_video is not None and sample.sample_id == in_progress_sample_id
            else None
        )
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
                decode_threads=run_config.get("video_decode_threads", 1),
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
                "extra_metadata": sample.extra_metadata,
            }
        )

        conversation_results: list[dict[str, Any]] = []
        video_frames_ingested_total = 0
        video_cumulative_ingest_latency_sec = 0.0
        last_ingested_timestamp_sec: float | None = None
        starting_conversation_index = 0
        ingested_until_idx = -1

        if partial_video_result is not None:
            conversation_results = list(partial_video_result.get("conversations", []))
            starting_conversation_index = len(conversation_results)
            partial_runtime_stats = dict(partial_video_result.get("runtime_stats", {}))
            video_frames_ingested_total = int(partial_runtime_stats.get("frames_ingested", 0) or 0)
            if video_frames_ingested_total <= 0 and conversation_results:
                video_frames_ingested_total = int(
                    conversation_results[-1].get("num_frames_ingested_before_answer", 0) or 0
                )
            video_cumulative_ingest_latency_sec = _cumulative_ingest_latency_sec(partial_runtime_stats)
            last_ingested_timestamp_sec = partial_runtime_stats.get("last_ingested_timestamp_sec")
            expected_sample_id = str(partial_video_result.get("sample_id"))
            if expected_sample_id != sample.sample_id:
                raise ValueError(
                    "Resume payload sample mismatch for in-progress video: "
                    f"expected {sample.sample_id!r}, got {expected_sample_id!r}"
                )
            ingested_until_idx = _replay_frames(
                method=method,
                cached_video=cached_video,
                sampled_video=sampled_video,
                sampled_timestamps_total=sampled_timestamps_total,
                end_frame_count=video_frames_ingested_total,
            )
            if progress is not None:
                progress.write(
                    f"[resume] restored partial video {sample.video_id}: "
                    f"{starting_conversation_index}/{len(sample.conversations)} conversations, "
                    f"{video_frames_ingested_total} frames"
                )

        frame_progress = None
        if show_progress_bar and tqdm is not None:
            frame_progress = tqdm(
                total=sampled_total_frames,
                initial=max(video_frames_ingested_total, 0),
                desc=f"{sample.video_id} frames",
                unit="frame",
                leave=False,
            )

        for conversation_index, conversation in enumerate(
            sample.conversations[starting_conversation_index:],
            start=starting_conversation_index,
        ):
            target_frame_count = min(
                conversation_target_frame_count(
                    conversation.end_time,
                    sampled_timestamps_total,
                ),
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
                    video_frames_ingested_total += 1
                    video_cumulative_ingest_latency_sec += float(ingest_record["ingest_latency_sec"])
                    last_ingested_timestamp_sec = float(ingest_record["timestamp_sec"])
                    if frame_progress is not None:
                        frame_progress.update(1)
            elif sampled_video is not None and new_indices:
                decoded_frames = sampled_video.get_frames(new_indices)
                for batch_offset, idx in enumerate(new_indices):
                    ingest_record = method.ingest_frame(
                        decoded_frames[batch_offset],
                        sampled_video.sampled_timestamps_sec[idx],
                    )
                    ingest_records.append(ingest_record)
                    ingested_until_idx = idx
                    video_frames_ingested_total += 1
                    video_cumulative_ingest_latency_sec += float(ingest_record["ingest_latency_sec"])
                    last_ingested_timestamp_sec = float(ingest_record["timestamp_sec"])
                    if frame_progress is not None:
                        frame_progress.update(1)

            if frame_progress is not None:
                frame_progress.set_postfix_str(
                    f"conv={conversation_index + 1}/{len(sample.conversations)} "
                    f"cutoff={conversation.end_time:.1f}s"
                )

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
                    "extra_metadata": conversation.extra_metadata,
                    "method_stats": answer.stats,
                }
            )

            if (
                checkpoint_path is not None
                and flush_every_conversations > 0
                and (conversation_index + 1) % flush_every_conversations == 0
            ):
                partial_video_payload = build_video_result(
                    sample=sample,
                    sample_fps=sample_fps,
                    ingest_source=ingest_source,
                    feature_cache_path=feature_cache_path,
                    sampled_native_fps=sampled_native_fps,
                    sampled_base_fps=sampled_base_fps,
                    sampled_total_frames=sampled_total_frames,
                    sampled_frame_indices_total=sampled_frame_indices_total,
                    sampled_timestamps_total=sampled_timestamps_total,
                    conversation_results=conversation_results,
                    runtime_stats=build_video_runtime_stats(
                        method=method,
                        frames_ingested_total=video_frames_ingested_total,
                        cumulative_ingest_latency_sec=video_cumulative_ingest_latency_sec,
                        last_ingested_timestamp_sec=last_ingested_timestamp_sec,
                    ),
                )
                checkpoint_payload = build_result_payload(
                    run_config=run_config,
                    evaluation_manifest=evaluation_manifest,
                    video_results=video_results,
                    started_at_utc=started_at_utc,
                    total_requested_videos=total_requested_videos,
                    status="in_progress",
                    in_progress_video=partial_video_payload,
                )
                write_json_atomic(checkpoint_payload, checkpoint_path)
                partial_message = (
                    f"[checkpoint] saved partial {sample.video_id}: "
                    f"{conversation_index + 1}/{len(sample.conversations)} conversations"
                )
                if progress is not None:
                    progress.write(partial_message)
                else:
                    print(partial_message)

        if frame_progress is not None:
            frame_progress.close()

        video_results.append(
            build_video_result(
                sample=sample,
                sample_fps=sample_fps,
                ingest_source=ingest_source,
                feature_cache_path=feature_cache_path,
                sampled_native_fps=sampled_native_fps,
                sampled_base_fps=sampled_base_fps,
                sampled_total_frames=sampled_total_frames,
                sampled_frame_indices_total=sampled_frame_indices_total,
                sampled_timestamps_total=sampled_timestamps_total,
                conversation_results=conversation_results,
                runtime_stats=build_video_runtime_stats(
                    method=method,
                    frames_ingested_total=video_frames_ingested_total,
                    cumulative_ingest_latency_sec=video_cumulative_ingest_latency_sec,
                    last_ingested_timestamp_sec=last_ingested_timestamp_sec,
                ),
            )
        )
        existing_in_progress_video = None
        newly_completed += 1

        if progress is not None:
            progress.update(1)
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
                evaluation_manifest=evaluation_manifest,
                video_results=video_results,
                started_at_utc=started_at_utc,
                total_requested_videos=total_requested_videos,
                status="in_progress",
                in_progress_video=None,
            )
            write_json_atomic(checkpoint_payload, checkpoint_path)
            message = (
                f"[checkpoint] saved {len(video_results)}/{total_requested_videos} videos to {checkpoint_path}"
            )
            if progress is not None:
                progress.write(message)
            else:
                print(message)

    if progress is not None:
        progress.close()
    return build_result_payload(
        run_config=run_config,
        evaluation_manifest=evaluation_manifest,
        video_results=video_results,
        started_at_utc=started_at_utc,
        total_requested_videos=total_requested_videos,
        status="completed",
        in_progress_video=None,
    )


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    from .methods import build_method_from_args

    run_config = build_run_config(args)
    validate_comparison_run_config(run_config)
    feature_cache_root = Path(args.feature_cache_root) if args.feature_cache_root else None
    cache_manifest: dict[str, Any] | None = None
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
    existing_in_progress_video: dict[str, Any] | None = None
    started_at_utc = utc_now_iso()
    checkpoint_path = Path(args.output_path) if args.output_path else default_output_path(args)

    if checkpoint_path.exists() and not args.resume and not args.overwrite_output:
            raise FileExistsError(
                f"Output already exists: {checkpoint_path}. "
                "Use --resume to continue it or --overwrite-output to replace it."
            )

    method = build_method_from_args(args)
    evaluation_manifest = build_evaluation_manifest(
        run_config=run_config,
        method_manifest=method.get_evaluation_manifest(),
        feature_cache_manifest=cache_manifest,
    )

    if checkpoint_path.exists() and args.resume:
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            existing_payload = json.load(handle)
        validate_resume_payload(existing_payload, run_config, evaluation_manifest)
        existing_videos = existing_payload.get("videos", [])
        existing_in_progress_video = existing_payload.get("in_progress_video")
        started_at_utc = (
            existing_payload.get("run_state", {}).get("started_at_utc") or started_at_utc
        )
        completed_sample_ids = {video["sample_id"] for video in existing_videos}
        if len(completed_sample_ids) == len(samples) and existing_in_progress_video is None:
            print(f"All requested videos already completed in {checkpoint_path}")
            return existing_payload
        print(
            f"Resuming {run_config['method']} from {checkpoint_path}: "
            f"{len(existing_videos)}/{len(samples)} videos already completed"
            + (
                f", partial video={existing_in_progress_video.get('video_id')}"
                if existing_in_progress_video is not None
                else "."
            )
        )

    return evaluate_samples(
        samples=samples,
        method=method,
        sample_fps=args.sample_fps,
        run_config=run_config,
        evaluation_manifest=evaluation_manifest,
        existing_videos=existing_videos,
        total_requested_videos=len(samples),
        started_at_utc=started_at_utc,
        checkpoint_path=checkpoint_path,
        flush_every_videos=args.flush_every_videos,
        flush_every_conversations=args.flush_every_conversations,
        show_progress_bar=not args.disable_progress_bar,
        feature_cache_root=feature_cache_root,
        existing_in_progress_video=existing_in_progress_video,
    )


def _apply_rocm_speedups() -> None:
    """Apply safe MI300X / ROCm performance knobs at process startup.

    These are always-on optimisations that do not change numerical output:
    - torch.set_float32_matmul_precision('high'): lets the matmul unit pick
      TF32 for intermediate accumulation (bfloat16 weights are unaffected).
    - PYTORCH_COMPILE=1 env var: wraps the language model with
      torch.compile(mode='reduce-overhead', dynamic=True) which fuses ops and
      enables kernel caching.  Adds ~60s of one-time warm-up but cuts per-token
      decode latency by ~30-50% on ROCm 7 / gfx942.  Disable with
      PYTORCH_COMPILE=0 if you hit a compilation error.
    """
    import os

    torch.set_float32_matmul_precision("high")

    compile_flag = os.environ.get("PYTORCH_COMPILE", "").strip()
    if compile_flag == "0":
        return
    # Only enable when explicitly requested or when running on ROCm 7+.
    hip_ver = getattr(torch.version, "hip", None) or ""
    rocm7_or_newer = hip_ver.startswith("7.")
    if compile_flag == "1" or rocm7_or_newer:
        # Patch methods module so build_method_from_args compiles the LM after load.
        from . import methods as _methods
        _orig_build = _methods.build_method_from_args

        def _compiling_build(args):  # type: ignore[override]
            m = _orig_build(args)
            # All method classes store the LlavaOnevision model as self.model
            llava = getattr(m, "model", None)
            lang = getattr(llava, "language_model", None)
            if lang is not None and not getattr(lang, "_rekv_compiled", False):
                try:
                    compiled = torch.compile(lang, mode="reduce-overhead", dynamic=True)
                    llava.language_model = compiled  # type: ignore[attr-defined]
                    compiled._rekv_compiled = True  # type: ignore[attr-defined]
                    print("[rocm] torch.compile(reduce-overhead) applied to language model")
                except Exception as exc:
                    print(f"[rocm] torch.compile skipped: {exc}")
            return m

        _methods.build_method_from_args = _compiling_build


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    _apply_rocm_speedups()
    if args.resume and args.output_path is None:
        raise ValueError("--resume requires an explicit --output-path so the runner knows which file to continue.")
    if args.flush_every_videos <= 0:
        raise ValueError("--flush-every-videos must be >= 1.")
    if args.flush_every_conversations <= 0:
        raise ValueError("--flush-every-conversations must be >= 1.")

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
