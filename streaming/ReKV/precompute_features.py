#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .datasets import RVS_DATASET_CONFIGS, build_dataset_from_args, sample_video_frames
from .feature_cache import (
    FEATURE_CACHE_VERSION,
    default_feature_cache_root,
    feature_cache_path,
    write_feature_cache_manifest,
)
from .methods import FullStreamingMethod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute per-frame LLaVA-OneVision visual features for streaming eval "
            "so decoder-side method comparisons can reuse the same cached inputs."
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
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--video-index", type=int, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--feature-cache-root", default=None)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = parse_args()
    cache_root = (
        Path(args.feature_cache_root).expanduser().resolve(strict=False)
        if args.feature_cache_root
        else default_feature_cache_root(args.dataset, args.model, args.sample_fps).resolve(strict=False)
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
    if args.feature_batch_size <= 0:
        raise ValueError("--feature-batch-size must be >= 1.")

    extractor = FullStreamingMethod(
        pretrained=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=1,
    )
    videos_dir = cache_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    iterable = samples
    progress = None
    if not args.disable_progress_bar and tqdm is not None:
        progress = tqdm(samples, total=len(samples), desc="feature cache", unit="video")
        iterable = progress

    cached_sample_ids: list[str] = []
    for sample in iterable:
        output_path = feature_cache_path(cache_root, sample.sample_id)
        if output_path.is_file() and not args.overwrite_existing:
            cached_sample_ids.append(sample.sample_id)
            if progress is not None:
                progress.set_postfix_str(f"skip {sample.video_id}")
            continue

        sampled_video = sample_video_frames(
            sample.video_path,
            args.sample_fps,
            duration_sec=sample.duration,
        )

        frame_feature_batches: list[torch.Tensor] = []
        total_sampled_frames = len(sampled_video.sampled_frame_indices)
        for start_idx in range(0, total_sampled_frames, args.feature_batch_size):
            end_idx = min(start_idx + args.feature_batch_size, total_sampled_frames)
            sampled_indices = list(range(start_idx, end_idx))
            frame_batch = sampled_video.get_frames(sampled_indices)
            with torch.inference_mode():
                batch_features = extractor.extract_frame_features_batch(frame_batch)
            frame_feature_batches.append(
                batch_features.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            )

        features = torch.cat(frame_feature_batches, dim=0) if frame_feature_batches else torch.empty((0, 0, 0), dtype=torch.bfloat16)
        payload = {
            "cache_version": FEATURE_CACHE_VERSION,
            "created_at_utc": utc_now_iso(),
            "sample_id": sample.sample_id,
            "video_id": sample.video_id,
            "video_path": sample.video_path,
            "duration": float(sample.duration),
            "sample_fps": float(args.sample_fps),
            "native_fps": float(sampled_video.native_fps),
            "sampling_base_fps": int(sampled_video.sampling_base_fps),
            "num_source_frames": int(sampled_video.num_source_frames),
            "sampled_frame_indices": [int(value) for value in sampled_video.sampled_frame_indices],
            "sampled_timestamps_sec": [float(value) for value in sampled_video.sampled_timestamps_sec],
            "num_frame_tokens": int(features.shape[1]) if features.ndim == 3 and features.shape[0] > 0 else 0,
            "hidden_size": int(features.shape[2]) if features.ndim == 3 and features.shape[0] > 0 else 0,
            "feature_dtype": str(features.dtype).replace("torch.", ""),
            "features": features,
        }
        torch.save(payload, output_path)
        cached_sample_ids.append(sample.sample_id)
        if progress is not None:
            progress.set_postfix_str(
                f"{sample.video_id} frames={total_sampled_frames} tokens={payload['num_frame_tokens']}"
            )

    if progress is not None:
        progress.close()

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
    print(f"Saved feature cache for {len(cached_sample_ids)} videos to {cache_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
