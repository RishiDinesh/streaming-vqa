#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_results import (
    color_for_payload,
    display_label,
    marker_for_payload,
    maybe_gb,
    wrapped_display_label,
)


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

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
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
                markersize=6.5,
                linewidth=2.0,
                color=color_for_payload(payload),
                label=label,
            )
            axis.set_ylabel(y_label)
            axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)
            axis.set_axisbelow(True)

    axes[0].set_title("Streaming Profiling Curves")
    axes[-1].set_xlabel("Frames Ingested")
    for axis in axes:
        axis.legend(title="Methods", fontsize=9, title_fontsize=9)

    manifest_summaries = []
    for payload in profiles:
        method_manifest = payload.get("evaluation_manifest", {}).get("method_manifest", {})
        backend = method_manifest.get("kernel_backend_path", {})
        duo_backend = method_manifest.get("duo_attention_backend")
        rekv_config = method_manifest.get("rekv_config", {})
        manifest_summaries.append(
            " | ".join(
                part
                for part in [
                    wrapped_display_label(payload, width=24).replace("\n", " "),
                    f"attn={backend.get('attention_module_load_path')}" if backend.get("attention_module_load_path") else None,
                    f"duo={duo_backend}" if duo_backend else None,
                    f"rekv_dot={rekv_config.get('dot_backend_actual')}" if rekv_config.get("dot_backend_actual") else None,
                ]
                if part
            )
        )
    if manifest_summaries:
        fig.text(
            0.5,
            0.005,
            "\n".join(manifest_summaries),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4a433d",
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
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
