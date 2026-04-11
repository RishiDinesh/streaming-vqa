#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import RVS_DATASET_CONFIGS, build_dataset_from_args, sample_video_frames
from .feature_cache import (
    FEATURE_CACHE_VERSION,
    load_cached_feature_video,
    load_feature_cache_manifest,
)
from .methods import DEFAULT_DUO_ATTN_DIR, build_method_from_args
from .run_eval import (
    build_evaluation_manifest,
    seed_everything,
    validate_comparison_run_config,
    write_json_atomic,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile streaming latency/memory/offload curves on a single video using "
            "fixed probe horizons."
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
        default="rekv",
        choices=["full_streaming", "duo_streaming", "rekv", "rekv_no_offload", "duo_plus_rekv"],
    )
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--video-index", type=int, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-decode-threads", type=int, default=4)
    parser.add_argument("--clear-cuda-cache-on-reset", action="store_true")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--feature-cache-root", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--probe-frame-counts",
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated cumulative frame counts to probe.",
    )
    parser.add_argument(
        "--profiling-question",
        default="What is happening in the video so far?",
    )
    parser.add_argument("--attn-dir", default=DEFAULT_DUO_ATTN_DIR)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--deploy-sink-size", type=int, default=None)
    parser.add_argument("--deploy-recent-size", type=int, default=None)
    parser.add_argument(
        "--duo-strict-no-sdpa-fallback",
        action="store_true",
        help=(
            "Fail fast for Duo-based methods if streaming attention would fall back to SDPA."
        ),
    )
    parser.add_argument("--n-local", type=int, default=15000)
    parser.add_argument("--retrieve-size", type=int, default=64)
    parser.add_argument("--retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--n-frame-tokens", type=int, default=196)
    parser.add_argument("--rekv-fattn", action="store_true")
    parser.add_argument("--disable-rekv-pin-memory", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_probe_frame_counts(raw_value: str) -> list[int]:
    values: set[int] = set()
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 0:
            raise ValueError("--probe-frame-counts must contain only non-negative integers.")
        values.add(value)
    if not values:
        raise ValueError("--probe-frame-counts must contain at least one frame count.")
    return sorted(values)


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
        "video_offset": args.video_offset,
        "video_index": args.video_index,
        "video_id": args.video_id,
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
        "duo_strict_no_sdpa_fallback": args.duo_strict_no_sdpa_fallback,
        "n_local": args.n_local,
        "retrieve_size": args.retrieve_size,
        "retrieve_chunk_size": args.retrieve_chunk_size,
        "n_frame_tokens": args.n_frame_tokens,
        "rekv_fattn": args.rekv_fattn,
        "disable_rekv_pin_memory": args.disable_rekv_pin_memory,
        "feature_cache_root": args.feature_cache_root,
        "ingest_source": "cached_features" if args.feature_cache_root else "raw_frames",
    }


def validate_feature_cache(args: argparse.Namespace) -> tuple[Path | None, dict[str, Any] | None]:
    if args.feature_cache_root is None:
        return None, None

    cache_root = Path(args.feature_cache_root).expanduser().resolve(strict=False)
    manifest = load_feature_cache_manifest(cache_root)
    if str(manifest.get("cache_version")) != FEATURE_CACHE_VERSION:
        raise ValueError(
            f"Feature cache version mismatch: expected {FEATURE_CACHE_VERSION}, "
            f"got {manifest.get('cache_version')}"
        )
    if str(manifest.get("dataset")) != args.dataset:
        raise ValueError(
            f"Feature cache dataset mismatch: expected {args.dataset!r}, got {manifest.get('dataset')!r}"
        )
    if str(manifest.get("model")) != args.model:
        raise ValueError(
            f"Feature cache model mismatch: expected {args.model!r}, got {manifest.get('model')!r}"
        )
    if abs(float(manifest.get("sample_fps")) - float(args.sample_fps)) > 1e-9:
        raise ValueError(
            f"Feature cache sample_fps mismatch: expected {args.sample_fps}, got {manifest.get('sample_fps')}"
        )
    return cache_root, manifest


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    run_config = build_run_config(args)
    validate_comparison_run_config(run_config)
    cache_root, cache_manifest = validate_feature_cache(args)
    probe_frame_counts = parse_probe_frame_counts(args.probe_frame_counts)

    dataset = build_dataset_from_args(args)
    samples = dataset.load(
        video_id=args.video_id,
        video_index=args.video_index,
        video_offset=args.video_offset,
        max_videos=1,
    )
    if not samples:
        raise ValueError("No samples matched the requested dataset/video filters.")
    sample = samples[0]

    feature_cache_path: str | None = None
    cached_video = None
    sampled_video = None
    if cache_root is not None:
        cached_video = load_cached_feature_video(
            cache_root,
            sample_id=sample.sample_id,
            video_id=sample.video_id,
            sample_fps=args.sample_fps,
        )
        feature_cache_path = cached_video.cache_path
        sampled_timestamps_total = list(cached_video.sampled_timestamps_sec)
        sampled_total_frames = len(cached_video.sampled_frame_indices)
        sampled_frame_indices_total = list(cached_video.sampled_frame_indices)
        sampled_native_fps = cached_video.native_fps
        sampled_base_fps = cached_video.sampling_base_fps
        ingest_source = "cached_features"
    else:
        sampled_video = sample_video_frames(
            sample.video_path,
            args.sample_fps,
            duration_sec=sample.duration,
            decode_threads=args.video_decode_threads,
        )
        sampled_timestamps_total = list(sampled_video.sampled_timestamps_sec)
        sampled_total_frames = len(sampled_video.sampled_frame_indices)
        sampled_frame_indices_total = list(sampled_video.sampled_frame_indices)
        sampled_native_fps = sampled_video.native_fps
        sampled_base_fps = sampled_video.sampling_base_fps
        ingest_source = "raw_frames"

    clipped_probe_counts = sorted(
        {
            min(frame_count, sampled_total_frames)
            for frame_count in probe_frame_counts
            if frame_count <= sampled_total_frames
        }
    )
    if sampled_total_frames not in clipped_probe_counts:
        clipped_probe_counts.append(sampled_total_frames)
    if not clipped_probe_counts:
        raise ValueError("No valid probe frame counts remain after clipping to the sampled video length.")

    method = build_method_from_args(args)
    evaluation_manifest = build_evaluation_manifest(
        run_config=run_config,
        method_manifest=method.get_evaluation_manifest(),
        feature_cache_manifest=cache_manifest,
    )
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

    ingested_until_idx = -1
    probes: list[dict[str, Any]] = []
    frame_progress = None
    probe_progress = None
    if not args.disable_progress_bar and tqdm is not None:
        frame_progress = tqdm(
            total=sampled_total_frames,
            desc=f"{sample.video_id} frames",
            unit="frame",
            leave=False,
        )
        probe_progress = tqdm(
            total=len(clipped_probe_counts),
            desc=f"{args.method} probes",
            unit="probe",
        )
    for probe_index, target_frame_count in enumerate(clipped_probe_counts):
        new_indices = list(range(ingested_until_idx + 1, target_frame_count))
        if cached_video is not None:
            for idx in new_indices:
                method.ingest_features(
                    cached_video.get_feature(idx),
                    cached_video.sampled_timestamps_sec[idx],
                )
                ingested_until_idx = idx
                if frame_progress is not None:
                    frame_progress.update(1)
        elif sampled_video is not None and new_indices:
            decoded_frames = sampled_video.get_frames(new_indices)
            for batch_offset, idx in enumerate(new_indices):
                method.ingest_frame(
                    decoded_frames[batch_offset],
                    sampled_video.sampled_timestamps_sec[idx],
                )
                ingested_until_idx = idx
                if frame_progress is not None:
                    frame_progress.update(1)

        answer = method.answer_question(
            args.profiling_question,
            metadata={
                "profiling_probe": True,
                "probe_index": probe_index,
                "target_frame_count": target_frame_count,
            },
        )
        last_timestamp = (
            float(sampled_timestamps_total[target_frame_count - 1]) if target_frame_count > 0 else 0.0
        )
        probes.append(
            {
                "probe_index": probe_index,
                "target_frame_count": int(target_frame_count),
                "ingested_frame_count": int(method.frames_ingested),
                "target_timestamp_sec": last_timestamp,
                "sampled_timestamps_sec_so_far": list(method.ingested_timestamps_sec),
                "prediction": answer.prediction,
                "method_stats": answer.stats,
            }
        )
        if frame_progress is not None:
            frame_progress.set_postfix_str(
                f"probe={probe_index + 1}/{len(clipped_probe_counts)}"
            )
        if probe_progress is not None:
            probe_progress.update(1)
            probe_progress.set_postfix_str(
                f"frames={target_frame_count}/{sampled_total_frames}"
            )

    if frame_progress is not None:
        frame_progress.close()
    if probe_progress is not None:
        probe_progress.close()

    payload = {
        "run_config": run_config,
        "evaluation_manifest": evaluation_manifest,
        "profile_config": {
            "profiling_question": args.profiling_question,
            "probe_frame_counts": clipped_probe_counts,
            "created_at_utc": utc_now_iso(),
        },
        "run_state": {
            "status": "completed",
            "created_at_utc": utc_now_iso(),
        },
        "video_profile": {
            "sample_id": sample.sample_id,
            "video_id": sample.video_id,
            "video_path": sample.video_path,
            "duration": sample.duration,
            "sample_fps": args.sample_fps,
            "native_fps": sampled_native_fps,
            "sampling_base_fps": sampled_base_fps,
            "num_sampled_frames_total": sampled_total_frames,
            "sampled_frame_indices_total": sampled_frame_indices_total,
            "sampled_timestamps_sec_total": sampled_timestamps_total,
            "ingest_source": ingest_source,
            "feature_cache_path": feature_cache_path,
            "extra_metadata": sample.extra_metadata,
            "profiling_question": args.profiling_question,
            "probes": probes,
            "runtime_stats": method.get_runtime_stats(),
        },
    }

    output_path = Path(args.output_path).expanduser().resolve(strict=False)
    write_json_atomic(payload, output_path)
    print(f"Saved profiling results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
