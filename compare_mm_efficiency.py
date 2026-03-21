#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Optional


def parse_result(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    data: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def get_float(data: Dict[str, str], key: str) -> Optional[float]:
    raw = data.get(key)
    if raw is None:
        return None
    token = raw.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def fmt(v: Optional[float], unit: str = "") -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}{unit}"


def pct_delta(base: Optional[float], other: Optional[float]) -> str:
    if base is None or other is None or base == 0:
        return "N/A"
    return f"{((other - base) / base) * 100.0:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dynamic/static efficiency benchmark outputs."
    )
    parser.add_argument(
        "--dynamic-baseline",
        type=Path,
        default=Path("./untracked/mm_efficiency_compare/dynamic_baseline/benchmark_result.txt"),
    )
    parser.add_argument(
        "--dynamic-duo",
        type=Path,
        default=Path("./untracked/mm_efficiency_compare/dynamic_duo/benchmark_result.txt"),
    )
    parser.add_argument(
        "--static-full-proxy",
        type=Path,
        default=Path("./untracked/mm_efficiency_compare/static_full_proxy/benchmark_result.txt"),
    )
    parser.add_argument(
        "--static-duo",
        type=Path,
        default=Path("./untracked/mm_efficiency_compare/static_duo/benchmark_result.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./untracked/mm_efficiency_compare/comparison_report.md"),
    )
    args = parser.parse_args()

    db = parse_result(args.dynamic_baseline)
    dd = parse_result(args.dynamic_duo)
    sf = parse_result(args.static_full_proxy)
    sd = parse_result(args.static_duo)

    lines = []
    lines.append("# Multimodal Efficiency Comparison")
    lines.append("")
    lines.append("## Dynamic (Standard vs Duo)")
    lines.append("")
    lines.append("| Metric | Baseline | Duo | Delta (Duo vs Baseline) |")
    lines.append("|---|---:|---:|---:|")
    for key, unit in [
        ("Average context time", " ms"),
        ("Average generation time", " ms"),
        ("Peak context memory usage", " MB"),
        ("Peak generation memory usage", " MB"),
    ]:
        b = get_float(db, key)
        d = get_float(dd, key)
        lines.append(f"| {key} | {fmt(b)}{unit} | {fmt(d)}{unit} | {pct_delta(b, d)} |")

    lines.append("")
    lines.append("## Static (Full-Head Proxy vs Duo)")
    lines.append("")
    lines.append("> `static_full_proxy` is Duo with `--sparsity 0` (all heads full).")
    lines.append("")
    lines.append("| Metric | Full Proxy | Duo | Delta (Duo vs Proxy) |")
    lines.append("|---|---:|---:|---:|")
    for key, unit in [
        ("Average context time", " ms"),
        ("Average generation time", " ms"),
        ("Peak context memory usage", " MB"),
        ("Peak generation memory usage", " MB"),
        ("KV cache memory usage", " MB"),
    ]:
        b = get_float(sf, key)
        d = get_float(sd, key)
        lines.append(f"| {key} | {fmt(b)}{unit} | {fmt(d)}{unit} | {pct_delta(b, d)} |")

    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append(f"- dynamic baseline: `{args.dynamic_baseline}`")
    lines.append(f"- dynamic duo: `{args.dynamic_duo}`")
    lines.append(f"- static full proxy: `{args.static_full_proxy}`")
    lines.append(f"- static duo: `{args.static_duo}`")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
