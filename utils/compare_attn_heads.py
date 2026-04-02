#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


ATTENTION_PATTERN_FILENAME = "full_attention_heads.tsv"
ATTENTION_CONFIG_FILENAME = "config.json"


def normalize_attention_dir(attn_path: str) -> Path:
    raw_path = Path(attn_path).expanduser()
    attn_dir = raw_path.parent if raw_path.name == ATTENTION_PATTERN_FILENAME else raw_path
    attn_dir = attn_dir.resolve(strict=False)

    if not attn_dir.exists():
        raise FileNotFoundError(f"Attention pattern directory does not exist: {attn_dir}")
    if not attn_dir.is_dir():
        raise NotADirectoryError(
            f"Attention pattern path must resolve to a directory: {attn_dir}"
        )

    pattern_path = attn_dir / ATTENTION_PATTERN_FILENAME
    config_path = attn_dir / ATTENTION_CONFIG_FILENAME
    if not pattern_path.is_file():
        raise FileNotFoundError(f"Missing attention pattern file: {pattern_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing DuoAttention config file: {config_path}")

    return attn_dir


def load_attn_pattern(attn_path: str) -> tuple[Path, list[list[float]], dict]:
    attn_dir = normalize_attention_dir(attn_path)
    full_attention_heads: list[list[float]] = []
    with open(attn_dir / ATTENTION_PATTERN_FILENAME, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            row = [min(1.0, max(0.0, float(value))) for value in stripped.split("\t")]
            full_attention_heads.append(row)

    if not full_attention_heads:
        raise ValueError(
            f"Attention pattern file is empty: {attn_dir / ATTENTION_PATTERN_FILENAME}"
        )
    with open(attn_dir / ATTENTION_CONFIG_FILENAME, "r", encoding="utf-8") as f:
        config = json.load(f)
    return attn_dir, full_attention_heads, config


def flatten(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def shape_of(matrix: list[list[float]]) -> tuple[int, int]:
    rows = len(matrix)
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("Attention pattern matrix is ragged")
    return rows, cols


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty list")

    sorted_values = sorted(values)
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]

    position = (len(sorted_values) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value

    weight = position - lower_index
    return lower_value * (1.0 - weight) + upper_value * weight


def sparsify_attention_heads(
    full_attention_heads: list[list[float]],
    *,
    seed: int,
    threshold: float | None = None,
    sparsity: float | None = None,
) -> tuple[list[list[int]], float, float]:
    rng = random.Random(seed)
    heads = [
        [value + rng.uniform(0.0, 1e-6) for value in row]
        for row in full_attention_heads
    ]

    # Match repo behavior by adding a tiny amount of noise before thresholding.
    if sparsity is not None:
        threshold = quantile(flatten(heads), sparsity)
        if sparsity >= 1:
            threshold = 2.0
        if sparsity <= 0:
            threshold = -1.0
    elif threshold is None:
        raise ValueError("Either threshold or sparsity must be provided")

    binary_heads = [
        [1 if value >= threshold else 0 for value in row]
        for row in heads
    ]
    flat_binary_heads = flatten(binary_heads)
    actual_sparsity = 1.0 - (sum(flat_binary_heads) / len(flat_binary_heads))
    return binary_heads, actual_sparsity, float(threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two DuoAttention full_attention_heads patterns after applying "
            "the same sparsification settings."
        )
    )
    parser.add_argument(
        "attn_a",
        help="First experiment directory under outputs/ or direct path to full_attention_heads.tsv",
    )
    parser.add_argument(
        "attn_b",
        help="Second experiment directory under outputs/ or direct path to full_attention_heads.tsv",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Quantile sparsity used by DuoAttention sparsification (default: 0.5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold to use instead of sparsity. Ignored when --sparsity is provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic tie-breaking noise (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sparsity = args.sparsity
    if sparsity is None and args.threshold is None:
        sparsity = 0.5
    if args.threshold is not None and "--sparsity" not in sys.argv:
        sparsity = None

    attn_dir_a, heads_a, config_a = load_attn_pattern(args.attn_a)
    attn_dir_b, heads_b, config_b = load_attn_pattern(args.attn_b)

    shape_a = shape_of(heads_a)
    shape_b = shape_of(heads_b)
    if shape_a != shape_b:
        raise ValueError(
            f"Attention head shapes do not match: {shape_a} vs {shape_b}"
        )

    binary_a, actual_sparsity_a, threshold_a = sparsify_attention_heads(
        heads_a,
        seed=args.seed,
        threshold=args.threshold,
        sparsity=sparsity,
    )
    binary_b, actual_sparsity_b, threshold_b = sparsify_attention_heads(
        heads_b,
        seed=args.seed,
        threshold=args.threshold,
        sparsity=sparsity,
    )

    flat_a = flatten(binary_a)
    flat_b = flatten(binary_b)
    total = len(flat_a)
    matched = sum(1 for left, right in zip(flat_a, flat_b) if left == right)
    both_zero = sum(1 for left, right in zip(flat_a, flat_b) if left == 0 and right == 0)
    both_one = sum(1 for left, right in zip(flat_a, flat_b) if left == 1 and right == 1)
    mismatched = total - matched
    match_pct = 100.0 * matched / total
    mismatch_pct = 100.0 * mismatched / total
    mismatched_heads = [
        (layer_idx, head_idx)
        for layer_idx, (row_a, row_b) in enumerate(zip(binary_a, binary_b))
        for head_idx, (value_a, value_b) in enumerate(zip(row_a, row_b))
        if value_a != value_b
    ]

    print(f"A: {attn_dir_a}")
    print(f"B: {attn_dir_b}")
    print(f"Shape: {shape_a}")
    if sparsity is not None:
        print(f"Requested sparsity: {sparsity}")
    if args.threshold is not None and sparsity is None:
        print(f"Requested threshold: {args.threshold}")
    print(f"A learned threshold: {threshold_a:.6f}")
    print(f"B learned threshold: {threshold_b:.6f}")
    print(f"A actual sparsity: {actual_sparsity_a:.6f}")
    print(f"B actual sparsity: {actual_sparsity_b:.6f}")
    print(
        "Config sink/recent: "
        f"A=({config_a.get('sink_size')}, {config_a.get('recent_size')}), "
        f"B=({config_b.get('sink_size')}, {config_b.get('recent_size')})"
    )
    print(f"Matching heads: {matched}/{total}")
    print(f"Mismatch heads: {mismatched}/{total}")
    print(f"Both 0: {both_zero}")
    print(f"Both 1: {both_one}")
    print(f"Match %: {match_pct:.4f}")
    print(f"Mismatch %: {mismatch_pct:.4f}")
    print(f"Mismatched heads (layer, head): {mismatched_heads}")


if __name__ == "__main__":
    main()
