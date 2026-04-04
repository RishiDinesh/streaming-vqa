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

from .common import normalized_exact_match
from .datasets import RVSEgoDataset, sample_video_frames
from .methods import DEFAULT_DUO_ATTN_DIR, build_method_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal streaming RVS-Ego evaluation for LLaVA-OneVision 0.5B using "
            "either DuoAttention or ReKV."
        )
    )
    parser.add_argument("--dataset", default="rvs_ego", choices=["rvs_ego"])
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
        choices=["duo_streaming", "rekv"],
    )
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-index", type=int, default=None)
    parser.add_argument("--video-id", default=None)
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


def default_output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(args.model)
    return (
        Path("outputs")
        / "evaluations_streaming"
        / "rvs-ego"
        / args.method
        / model_slug
        / f"{timestamp}_results.json"
    )


def evaluate_samples(
    *,
    samples,
    method,
    sample_fps: float,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    video_results: list[dict[str, Any]] = []
    aggregate_scores: list[float] = []
    ttfts: list[float] = []
    answer_latencies: list[float] = []
    frame_ingest_latencies: list[float] = []
    peak_memories: list[int] = []
    total_frames_ingested = 0

    for sample in samples:
        sampled_video = sample_video_frames(sample.video_path, sample_fps)
        method.reset(
            {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "video_path": sample.video_path,
                "duration": sample.duration,
            }
        )

        ingested_until_idx = -1
        conversation_results: list[dict[str, Any]] = []

        for conversation in sample.conversations:
            new_indices = [
                idx
                for idx, timestamp_sec in enumerate(sampled_video.sampled_timestamps_sec)
                if ingested_until_idx < idx and timestamp_sec <= conversation.end_time
            ]
            ingest_records: list[dict[str, Any]] = []
            for idx in new_indices:
                ingest_record = method.ingest_frame(
                    sampled_video.get_frame(idx),
                    sampled_video.sampled_timestamps_sec[idx],
                )
                ingest_records.append(ingest_record)
                frame_ingest_latencies.append(float(ingest_record["ingest_latency_sec"]))
                total_frames_ingested += 1
                ingested_until_idx = idx

            answer = method.answer_question(
                conversation.question,
                metadata={
                    "start_time": conversation.start_time,
                    "end_time": conversation.end_time,
                },
            )
            exact_match = normalized_exact_match(answer.prediction, conversation.answer)
            aggregate_scores.append(exact_match)
            if answer.stats.get("ttft_sec") is not None:
                ttfts.append(float(answer.stats["ttft_sec"]))
            if answer.stats.get("answer_latency_sec") is not None:
                answer_latencies.append(float(answer.stats["answer_latency_sec"]))
            if answer.stats.get("peak_memory_bytes") is not None:
                peak_memories.append(int(answer.stats["peak_memory_bytes"]))

            conversation_results.append(
                {
                    "question": conversation.question,
                    "reference_answer": conversation.answer,
                    "prediction": answer.prediction,
                    "normalized_exact_match": exact_match,
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
                "native_fps": sampled_video.native_fps,
                "num_sampled_frames_total": len(sampled_video.sampled_frame_indices),
                "sampled_frame_indices_total": sampled_video.sampled_frame_indices,
                "sampled_timestamps_sec_total": sampled_video.sampled_timestamps_sec,
                "conversations": conversation_results,
                "runtime_stats": method.get_runtime_stats(),
            }
        )

    aggregate_metrics = {
        "normalized_exact_match": (
            float(sum(aggregate_scores) / len(aggregate_scores)) if aggregate_scores else None
        ),
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
        "total_conversations_answered": len(aggregate_scores),
        "evaluation_mode": "normalized_exact_match",
    }

    return {
        "run_config": run_config,
        "aggregate_metrics": aggregate_metrics,
        "videos": video_results,
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    dataset = RVSEgoDataset(
        annotation_path=args.annotation_path,
        video_root=args.video_root,
        hf_repo_id=args.hf_repo_id,
        allow_hf_video_download=args.allow_hf_video_download,
    )
    samples = dataset.load(
        video_id=args.video_id,
        video_index=args.video_index,
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

    method = build_method_from_args(args)

    return evaluate_samples(
        samples=samples,
        method=method,
        sample_fps=args.sample_fps,
        run_config={
            "dataset": args.dataset,
            "annotation_path": args.annotation_path,
            "video_root": args.video_root,
            "hf_repo_id": args.hf_repo_id,
            "allow_hf_video_download": args.allow_hf_video_download,
            "model": args.model,
            "method": args.method,
            "sample_fps": args.sample_fps,
            "max_videos": args.max_videos,
            "video_index": args.video_index,
            "video_id": args.video_id,
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
        },
    )


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    results = run_eval(args)
    output_path = Path(args.output_path) if args.output_path else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
