import hashlib
import math
import random
from typing import List, Sequence, Tuple

import numpy as np


def build_pool(mask: np.ndarray, select_value: float) -> List[Tuple[int, int]]:
    pool: List[Tuple[int, int]] = []
    for layer_idx in range(mask.shape[0]):
        for head_idx in range(mask.shape[1]):
            if mask[layer_idx, head_idx] == select_value:
                pool.append((int(layer_idx), int(head_idx)))
    return pool


def deterministic_pool_order(
    pool: Sequence[Tuple[int, int]],
    seed: int,
    seed_offset: int,
) -> List[Tuple[int, int]]:
    ordered = sorted(pool)
    rng = random.Random(seed + seed_offset)
    rng.shuffle(ordered)
    return ordered


def ablate_pool_ratio(
    full_mask: np.ndarray,
    ordered_pool: Sequence[Tuple[int, int]],
    ratio_percent: int,
) -> Tuple[np.ndarray, int]:
    ablated = full_mask.copy()
    pool_size = len(ordered_pool)
    num_to_stream = int(math.floor(((ratio_percent / 100.0) * pool_size) + 0.5))

    for layer_idx, head_idx in ordered_pool[:num_to_stream]:
        ablated[layer_idx, head_idx] = 0.0

    return ablated, num_to_stream


def build_fixed_ratios() -> List[int]:
    return list(range(0, 101, 10))


def mask_cache_key(mask: np.ndarray) -> str:
    mask_u8 = mask.astype(np.uint8, copy=False)
    return hashlib.sha1(mask_u8.tobytes()).hexdigest()
