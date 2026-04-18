#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .plot_results import display_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a machine-readable and markdown qualitative bundle from result JSON files."
    )
    parser.add_argument("result_paths", nargs="+", help="One or more streaming result JSON files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_source_path"] = str(path)
    return payload


def slice_name(payload: dict[str, Any]) -> str:
    run_config = payload.get("run_config", {})
    return str(run_config.get("subsample_name") or f"offset{run_config.get('video_offset', 0)}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = [load_payload(Path(path).expanduser().resolve()) for path in args.result_paths]
    grouped: dict[tuple[str, str, str, float, str], dict[str, Any]] = {}

    for payload in payloads:
        dataset = str(payload.get("run_config", {}).get("dataset"))
        subslice = slice_name(payload)
        label = display_label(payload)
        for video in payload.get("videos", []):
            for conversation in video.get("conversations", []):
                key = (
                    dataset,
                    subslice,
                    str(video.get("video_id")),
                    float(conversation.get("end_time", 0.0)),
                    str(conversation.get("question", "")),
                )
                entry = grouped.setdefault(
                    key,
                    {
                        "dataset": dataset,
                        "slice_name": subslice,
                        "video_id": str(video.get("video_id")),
                        "sample_id": str(video.get("sample_id")),
                        "question": str(conversation.get("question", "")),
                        "reference_answer": str(conversation.get("reference_answer", "")),
                        "end_time": float(conversation.get("end_time", 0.0)),
                        "conversation_extra_metadata": conversation.get("extra_metadata", {}),
                        "methods": {},
                    },
                )
                entry["methods"][label] = {
                    "prediction": str(conversation.get("prediction", "")),
                    "scores": conversation.get("scores", {}),
                    "judge": conversation.get("judge"),
                    "method_stats": conversation.get("method_stats", {}),
                    "sampled_timestamps_sec_so_far": conversation.get(
                        "sampled_timestamps_sec_so_far", []
                    ),
                    "source_path": payload["_source_path"],
                }

    bundle = sorted(
        grouped.values(),
        key=lambda item: (
            item["dataset"],
            item["slice_name"],
            item["video_id"],
            item["end_time"],
            item["question"],
        ),
    )

    json_path = output_dir / "qualitative_bundle.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)

    lines: list[str] = ["# Qualitative Bundle", ""]
    for index, item in enumerate(bundle, start=1):
        lines.append(
            f"## {index}. {item['dataset']} / {item['slice_name']} / {item['video_id']}"
        )
        lines.append(f"- Question: `{item['question']}`")
        lines.append(f"- Reference: `{item['reference_answer']}`")
        lines.append(f"- End time: `{item['end_time']}`")
        for label, method_entry in sorted(item["methods"].items()):
            method_stats = method_entry.get("method_stats", {})
            scores = method_entry.get("scores", {})
            lines.append(f"- {label}: `{method_entry['prediction']}`")
            lines.append(
                "  "
                + f"judge={scores.get('judge_score')} rouge_l_f1={scores.get('rouge_l_f1')} "
                + f"latency={method_stats.get('answer_latency_sec')} "
                + f"cpu_offload_bytes={method_stats.get('cpu_offload_bytes_current')} "
                + f"retrieved_timestamps={method_stats.get('retrieved_timestamps_sec_union', [])}"
            )
        lines.append("")

    md_path = output_dir / "qualitative_bundle.md"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    manifest = {
        "qualitative_bundle_json": str(json_path),
        "qualitative_bundle_md": str(md_path),
        "num_items": len(bundle),
        "sources": [payload["_source_path"] for payload in payloads],
    }
    manifest_path = output_dir / "qualitative_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved qualitative bundle to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
