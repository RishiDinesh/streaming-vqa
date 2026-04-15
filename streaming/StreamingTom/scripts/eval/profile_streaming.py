#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from streaming.ReKV.datasets import RVS_DATASET_CONFIGS, build_dataset_from_args, sample_video_frames

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_eval import build_method_from_args, build_run_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile StreamingTom latency/memory curves on one video using probe frame counts."
        )
    )
    parser.add_argument("--dataset", default="rvs_ego", choices=sorted(RVS_DATASET_CONFIGS))
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--hf-repo-id", default="Becomebright/RVS")
    parser.add_argument("--allow-hf-video-download", action="store_true")
    parser.add_argument("--model", default="lmms-lab/llava-onevision-qwen2-0.5b-ov")
    parser.add_argument("--method", default="streamingtom", choices=["streamingtom", "duo_plus_streamingtom"])
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--video-index", type=int, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-decode-threads", type=int, default=1)
    parser.add_argument("--clear-cuda-cache-on-reset", action="store_true")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--probe-frame-counts", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--profiling-question", default="What is happening in the video so far?")
    parser.add_argument("--streamingtom-root", default="streaming/StreamingTom")
    parser.add_argument("--duo-attn-dir", default="outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632")
    parser.add_argument("--duo-heads-file", default="outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632/full_attention_heads_latest.tsv")
    parser.add_argument("--duo-threshold", type=float, default=0.5)
    parser.add_argument("--duo-sparsity", type=float, default=0.75)
    parser.add_argument("--duo-sink-size", type=int, default=256)
    parser.add_argument("--duo-recent-size", type=int, default=512)
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    probes = parse_probe_frame_counts(args.probe_frame_counts)

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

    sampled_video = sample_video_frames(
        sample.video_path,
        args.sample_fps,
        duration_sec=sample.duration,
        decode_threads=args.video_decode_threads,
    )

    method = build_method_from_args(args)

    probe_records: list[dict[str, Any]] = []
    for frame_count in probes:
        target = min(int(frame_count), len(sampled_video.sampled_frame_indices))
        method.reset(
            {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "video_path": sample.video_path,
                "duration": sample.duration,
                "ingest_source": "raw_frames",
                "feature_cache_path": None,
                "extra_metadata": sample.extra_metadata,
            }
        )

        if target > 0:
            frames = sampled_video.get_frames(list(range(target)))
            for idx in range(target):
                method.ingest_frame(frames[idx], sampled_video.sampled_timestamps_sec[idx])

        answer = method.answer_question(args.profiling_question)
        probe_records.append(
            {
                "ingested_frame_count": int(target),
                "elapsed_profile_sec": float(time.perf_counter()),
                "prediction": answer.prediction,
                "method_stats": answer.stats,
            }
        )

    payload = {
        "run_config": build_run_config(args),
        "evaluation_manifest": {
            "comparison_contract_version": "v1",
            "comparison_scope": "streaming_vqa_cross_method",
            "method_manifest": method.get_evaluation_manifest(),
        },
        "video_profile": {
            "sample_id": sample.sample_id,
            "video_id": sample.video_id,
            "video_path": sample.video_path,
            "sample_fps": float(args.sample_fps),
            "num_sampled_frames_total": int(len(sampled_video.sampled_frame_indices)),
            "probes": probe_records,
        },
    }

    output_path = Path(args.output_path).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved profile to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
