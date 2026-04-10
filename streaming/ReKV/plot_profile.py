#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_results import color_for_payload, display_label, marker_for_payload, maybe_gb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render profiling curves from streaming/ReKV/profile_streaming.py outputs."
    )
    parser.add_argument("profile_paths", nargs="+", help="One or more profiling JSON files.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_profile(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_source_path"] = str(path)
    return payload


def main() -> int:
    args = parse_args()
    profile_paths = [Path(path).expanduser().resolve() for path in args.profile_paths]
    profiles = [load_profile(path) for path in profile_paths]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = profile_paths[0].parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    metric_specs = [
        ("answer_latency_sec", "Answer Latency (s)"),
        ("ttft_sec", "TTFT (s)"),
        ("peak_memory_bytes", "Peak GPU Memory (GB)"),
        ("cpu_offload_bytes_current", "CPU / Offloaded KV (GB)"),
        ("retrieval_latency_sec", "Retrieval Latency (s)"),
    ]

    for payload in profiles:
        probes = payload.get("video_profile", {}).get("probes", [])
        if not probes:
            continue
        xs = [int(probe["ingested_frame_count"]) for probe in probes]
        label = display_label(payload)
        for axis, (metric_key, y_label) in zip(axes, metric_specs):
            ys = [probe.get("method_stats", {}).get(metric_key) for probe in probes]
            if metric_key.endswith("_bytes"):
                ys = [maybe_gb(value) for value in ys]
            axis.plot(
                xs,
                ys,
                marker=marker_for_payload(payload),
                color=color_for_payload(payload),
                label=label,
            )
            axis.set_ylabel(y_label)
            axis.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_title("Streaming Profiling Curves")
    axes[-1].set_xlabel("Frames Ingested")
    for axis in axes:
        axis.legend()

    fig.tight_layout()
    profile_plot_path = output_dir / "profiling_curves.png"
    fig.savefig(profile_plot_path, dpi=200)
    plt.close(fig)

    manifest = {
        "profiling_curves": str(profile_plot_path),
        "sources": [payload["_source_path"] for payload in profiles],
        "labels": [display_label(payload) for payload in profiles],
    }
    manifest_path = output_dir / "profile_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved profiling plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
