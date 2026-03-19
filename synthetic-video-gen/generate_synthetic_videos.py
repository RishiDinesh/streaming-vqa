#!/usr/bin/env python3
"""
VideoNIAH Synthetic Generator — VNBench-Style + DuoAttention-Aligned

Generates videos where MULTIPLE secret words are burned as subtitles at
different temporal positions, directly extending the VNBench ret_edit format
while aligning with DuoAttention paper implementation details:

  Paper params (Table 6, Appendix A.4):
    - 10 passkeys per sample
    - depth ratios in [0.05, 0.5] (first half, forces long-range retrieval)
    - context fills model max length via 20 intervals
    - 32-word passkeys using NATO alphabet

  VNBench ret_edit style:
    - White text on grey background bar
    - Bottom-centered, fixed position
    - Sans-serif font (OpenSans only)
    - Font size ~6% of video height

Designed for LLaVA-OneVision 0.5B (32k context, ~196 tokens/frame).
At 64 sampled frames → need ~163 frames to fill 32k.
At 1 fps → ~2.7 min of video. We generate 3-10 min videos.
"""

import os
import json
import random
import argparse
import glob

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Please install required packages: pip install opencv-python numpy pillow")
    import sys
    sys.exit(1)


# ── Secret Word Pool (500+ diverse words) ────────────────────────────────────
# Categories: Names, Animals, Objects, Places, Foods, Colors, Science, Nature
SECRET_WORDS = [
    # Names (100)
    "Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Helen",
    "Ivan", "Jack", "Kate", "Leo", "Mike", "Nick", "Oscar", "Paul",
    "Quinn", "Rose", "Sam", "Tom", "Uma", "Vince", "Wendy", "Xena",
    "Yvonne", "Zack", "Amy", "Ben", "Clara", "Dan", "Emma", "Fred",
    "Gina", "Henry", "Iris", "Jade", "Kyle", "Lucy", "Max", "Nina",
    "Owen", "Pam", "Ray", "Sara", "Tina", "Uri", "Vera", "Walt",
    "Yuri", "Zoe", "Amber", "Blake", "Chloe", "Derek", "Elena", "Felix",
    "Hana", "Igor", "Jules", "Kira", "Liam", "Maya", "Noah", "Olive",
    "Petra", "Reese", "Stella", "Theo", "Vivian", "Wyatt", "Axel", "Briar",
    "Cyrus", "Diana", "Elton", "Flora", "Grant", "Holly", "Jasper", "Kenji",
    "Luna", "Marco", "Nadia", "Orion", "Penny", "Rufus", "Sage", "Tobias",
    "Ursa", "Violet", "Wren", "Xavier", "Yara", "Zelda", "Arlo", "Blythe",
    "Cruz", "Daphne", "Ezra", "Fern",
    # Animals (80)
    "Falcon", "Tiger", "Dolphin", "Eagle", "Panda", "Wolf", "Raven", "Cobra",
    "Otter", "Jaguar", "Crane", "Bison", "Gecko", "Heron", "Koala", "Lemur",
    "Manta", "Newt", "Okapi", "Parrot", "Quail", "Robin", "Shark", "Toucan",
    "Viper", "Walrus", "Yak", "Zebra", "Alpaca", "Badger", "Condor", "Dingo",
    "Ermine", "Ferret", "Gorilla", "Hyena", "Iguana", "Jackal", "Kiwi", "Lynx",
    "Moose", "Narwhal", "Osprey", "Puffin", "Quetzal", "Raptor", "Stork", "Tapir",
    "Urchin", "Vulture", "Wombat", "Xerus", "Giraffe", "Pelican", "Mantis", "Beetle",
    "Coyote", "Donkey", "Finch", "Grouse", "Hornet", "Impala", "Jacana", "Kestrel",
    "Lobster", "Marmot", "Numbat", "Oriole", "Pigeon", "Rabbit", "Salmon", "Turtle",
    "Vervet", "Weasel", "Xenops", "Gopher", "Hermit", "Ibis", "Jerboa", "Lark",
    # Objects (80)
    "Anchor", "Basket", "Candle", "Dagger", "Engine", "Flagon", "Goblet", "Hammer",
    "Inkwell", "Jacket", "Kettle", "Lantern", "Mirror", "Needle", "Obelisk", "Pillar",
    "Quiver", "Ribbon", "Scepter", "Trumpet", "Umbrella", "Vessel", "Widget", "Zipper",
    "Anvil", "Beacon", "Chisel", "Drum", "Easel", "Funnel", "Garnet", "Hatchet",
    "Ivory", "Jewel", "Knapsack", "Locket", "Mortar", "Nugget", "Orchid", "Prism",
    "Quartz", "Ratchet", "Saddle", "Thimble", "Urn", "Vial", "Wrench", "Xylophone",
    "Buckle", "Compass", "Decanter", "Emblem", "Feather", "Gauntlet", "Hourglass",
    "Insignia", "Javelin", "Keystone", "Lattice", "Medallion", "Nozzle", "Pendant",
    "Relic", "Spindle", "Talisman", "Utensil", "Valve", "Wagon", "Abacus",
    "Bellows", "Caliper", "Dowel", "Eyelet", "Flint", "Gimbal", "Harness", "Ingot",
    # Nature (60)
    "Aurora", "Blizzard", "Canyon", "Delta", "Eclipse", "Fjord", "Glacier", "Horizon",
    "Island", "Jungle", "Karst", "Lagoon", "Monsoon", "Nebula", "Oasis", "Prairie",
    "Quasar", "Ravine", "Summit", "Tsunami", "Volcano", "Wetland", "Zenith", "Bamboo",
    "Cedar", "Daisy", "Elm", "Ficus", "Grove", "Hazel", "Ivy", "Juniper",
    "Kelp", "Lotus", "Maple", "Nettle", "Oak", "Palm", "Reed", "Spruce",
    "Thicket", "Tundra", "Willow", "Acacia", "Birch", "Clover", "Dune", "Estuary",
    "Geyser", "Heath", "Inlet", "Jasmine", "Kindle", "Lichen", "Moss",
    "Marigold", "Poplar", "Redwood", "Savanna", "Terrace",
    # Foods (50)
    "Almond", "Biscuit", "Cashew", "Dumpling", "Espresso", "Fondue", "Granola",
    "Hazelnut", "Icing", "Jambalaya", "Kumquat", "Lychee", "Mango", "Nougat",
    "Pretzel", "Quinoa", "Raisin", "Sorbet", "Truffle", "Vanilla", "Waffle",
    "Apricot", "Brioche", "Clementine", "Focaccia", "Ginger", "Hummus",
    "Kale", "Lemon", "Muffin", "Nutmeg", "Papaya", "Rhubarb", "Saffron", "Tapioca",
    "Wasabi", "Arugula", "Basil", "Cinnamon", "Dill", "Fennel", "Garlic", "Honey",
    "Jalapeno", "Lavender", "Mint", "Pepper", "Sesame", "Thyme", "Turnip",
    # Science / Abstract (80)
    "Atom", "Binary", "Cipher", "Dynamo", "Electron", "Fractal", "Genome", "Helix",
    "Isotope", "Joule", "Kelvin", "Lambda", "Matrix", "Neutron", "Omega", "Photon",
    "Quantum", "Reactor", "Sigma", "Tensor", "Uranium", "Vector", "Waveform", "Xenon",
    "Alpha", "Beta", "Carbon", "Doppler", "Entropy", "Flux", "Gamma", "Hadron",
    "Inertia", "Kinetic", "Muon", "Nucleus", "Orbit", "Plasma", "Quarks",
    "Radium", "Syntax", "Theorem", "Upsilon", "Vertex", "Wavelength", "Axiom", "Boson",
    "Cosine", "Decibel", "Epsilon", "Faraday", "Gauss", "Hertz", "Impulse", "Kinase",
    "Laser", "Magnet", "Newton", "Optics", "Proton", "Reflex", "Scalar", "Torque",
    "Unity", "Vortex", "Weber", "Ampere", "Boron", "Cobalt", "Diode", "Ether",
    "Fusion", "Gravity", "Hybrid", "Iodine", "Krypton", "Lithium", "Manganese", "Neon",
]

# Remove duplicates
SECRET_WORDS = list(dict.fromkeys(SECRET_WORDS))

ORDINALS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
]


# ── VNBench-Style Fixed Rendering ────────────────────────────────────────────

def get_video_info(video_path):
    """Return dict with fps, total_frames, duration, width, height or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    if duration <= 0:
        return None
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "width": width,
        "height": height,
    }


def _load_vnbench_font(height):
    """
    Load a sans-serif TTF font at ~6% of video height (matching VNBench).
    Falls back to PIL default if no font is found.
    """
    font_size = max(14, int(height * 0.06))
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "fonts", "OpenSans.ttf")
    try:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    except Exception:
        pass
    return ImageFont.load_default()


def burn_subtitle_vnbench(frame, text, font):
    """
    Burns a VNBench-style subtitle onto a frame:
      - Full-width semi-dark grey background bar
      - White text, horizontally centered, vertically centered in the bar
      - Bar sits at the bottom of the frame (~85% down)
    """
    h, w = frame.shape[:2]

    # Convert BGR → RGB for PIL
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # Measure text — use (0, 0) as anchor and get the real bbox offsets
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]   # actual rendered width
    text_h = bbox[3] - bbox[1]   # actual rendered height

    # Bar height = text height + vertical padding
    pad_y = max(4, int(text_h * 0.30))
    bar_h = text_h + 2 * pad_y

    # Bar sits at 85% down (bottom edge of bar)
    bar_top = int(h * 0.85) - bar_h
    bar_bottom = bar_top + bar_h

    # Draw full-width grey bar (matching VNBench)
    draw.rectangle([0, bar_top, w, bar_bottom], fill=(80, 80, 80))

    # Centre text inside the bar, compensating for PIL bbox offset
    text_x = (w - text_w) // 2 - bbox[0]
    text_y = bar_top + pad_y - bbox[1]

    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── Core Generation ──────────────────────────────────────────────────────────

def sample_depth_ratios(num_needles, min_ratio, max_ratio, num_intervals=20):
    """
    Mirrors duo_attn/data.py: create `num_intervals` evenly spaced depth
    positions, then randomly select `num_needles` of them (sorted).
    """
    intervals = np.linspace(min_ratio, max_ratio, num_intervals)
    chosen_indices = np.random.choice(len(intervals), size=num_needles, replace=False)
    chosen = np.sort(intervals[chosen_indices])
    return chosen.tolist()


def generate_video_multi_needle(
    input_video_path,
    output_video_path,
    needles,              # list of (secret_word, ratio, duration_sec)
    target_duration=None,
    num_frames=64,
):
    """
    Reads input video, optionally crops to target_duration, burns each needle's
    secret word subtitle (VNBench-style) for its duration window,
    and writes the output video.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = 0
    end_frame = total_frames

    if target_duration and target_duration < (total_frames / fps):
        target_frames = int(target_duration * fps)
        max_start = total_frames - target_frames
        start_frame = random.randint(0, max(0, max_start))
        end_frame = start_frame + target_frames

    # Load VNBench-style font once
    font = _load_vnbench_font(height)

    out_total_frames = end_frame - start_frame
    if out_total_frames <= 1:
        loader_sampled_frames = [0] * num_frames
    else:
        loader_sampled_frames = np.round(np.linspace(0, out_total_frames - 1, num_frames)).astype(int).tolist()

    # Build list of (frame_start, frame_end, subtitle_text) for each needle
    needle_windows = []
    actual_needle_times = []
    real_idx = 0
    for word, ratio, dur_sec in needles:
        ordinal = ORDINALS[real_idx] if real_idx < len(ORDINALS) else f"#{real_idx+1}"
        subtitle = f"The {ordinal} secret word is: {word}"
        real_idx += 1

        sampled_idx_mapped = int(round(ratio * (num_frames - 1)))
        target_frame_in_out = loader_sampled_frames[sampled_idx_mapped]
        
        half_dur_frames = int((dur_sec / 2) * fps)
        f_start = target_frame_in_out - half_dur_frames + start_frame
        f_end = target_frame_in_out + half_dur_frames + start_frame

        f_start = max(start_frame, min(f_start, end_frame - 1))
        f_end = max(f_start + 1, min(f_end, end_frame))

        needle_windows.append((f_start, f_end, subtitle))
        actual_needle_times.append(round(target_frame_in_out / fps, 2))

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_idx = start_frame
    while current_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if current frame falls within any needle window
        for f_start, f_end, subtitle in needle_windows:
            if f_start <= current_idx < f_end:
                frame = burn_subtitle_vnbench(frame, subtitle, font)
                break

        out.write(frame)
        current_idx += 1

    cap.release()
    out.release()

    actual_duration = (current_idx - start_frame) / fps
    return actual_duration, actual_needle_times


# ── Dataset Generation ───────────────────────────────────────────────────────

def generate_dataset(
    num_videos,
    source_videos_dir,
    output_dir,
    min_duration,
    max_duration,
    num_length_intervals,
    num_needles,
    needle_duration,
    min_depth_ratio,
    max_depth_ratio,
    depth_num_intervals,
    num_frames=64,
):
    os.makedirs(output_dir, exist_ok=True)
    videos_output_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_output_dir, exist_ok=True)

    # Discover source videos
    video_files = []
    for ext in ["*.mp4", "*.webm", "*.avi", "*.mkv"]:
        video_files.extend(
            glob.glob(os.path.join(source_videos_dir, "**", ext), recursive=True)
        )

    if not video_files:
        print(f"ERROR: No videos found in {source_videos_dir}.")
        return

    print(f"Found {len(video_files)} source videos.")
    print(f"Secret word pool size: {len(SECRET_WORDS)}")

    # Linearly-spaced target durations (mirrors duo_attn/data.py)
    duration_intervals = np.linspace(min_duration, max_duration, num_length_intervals)
    videos_per_interval = num_videos // num_length_intervals
    remainder = num_videos % num_length_intervals

    annotations = []
    success_count = 0
    fail_count = 0

    print(f"Generating {num_videos} multi-needle (ret_edit) videos...")
    print(f"  Duration range: {min_duration}s – {max_duration}s  ({num_length_intervals} intervals)")
    print(f"  Needles per video: {num_needles}")
    print(f"  Needle duration: {needle_duration}s each (fixed, VNBench-style)")
    print(f"  Depth ratio: {min_depth_ratio} – {max_depth_ratio}  ({depth_num_intervals} intervals)")

    for interval_idx, target_dur in enumerate(duration_intervals):
        n_this_interval = videos_per_interval + (1 if interval_idx < remainder else 0)

        for j in range(n_this_interval):
            # Pick a source video long enough
            random.shuffle(video_files)
            src_video = None
            info = None
            for vf in video_files:
                info = get_video_info(vf)
                if info and info["duration"] >= target_dur:
                    src_video = vf
                    break

            if src_video is None:
                for vf in video_files:
                    info = get_video_info(vf)
                    if info and info["duration"] >= 3.0:
                        src_video = vf
                        target_dur_actual = info["duration"]
                        break
                if src_video is None:
                    fail_count += 1
                    continue
                target_dur_actual = info["duration"]
            else:
                target_dur_actual = target_dur

            # ── Sample depth ratios (DuoAttention-style) ─────────────────
            depth_ratios = sample_depth_ratios(
                num_needles, min_depth_ratio, max_depth_ratio, depth_num_intervals
            )

            # Pick unique secret words for all needles
            chosen_words = random.sample(SECRET_WORDS, num_needles)

            # Pass ratio to guarantee exact sampled frame alignment
            needles = []
            for word, ratio in zip(chosen_words, depth_ratios):
                needles.append((word, ratio, needle_duration))

            # Randomly pick which needle the question will ask about
            ask_idx = random.randint(0, num_needles - 1)
            ordinal = ORDINALS[ask_idx] if ask_idx < len(ORDINALS) else f"#{ask_idx+1}"
            gt_word = chosen_words[ask_idx]

            # Build 4-choice options: gt + 3 distractors
            other_needle_words = [w for i, w in enumerate(chosen_words) if i != ask_idx]
            pool_words = [w for w in SECRET_WORDS if w not in chosen_words]
            distractor_pool = other_needle_words + pool_words
            random.shuffle(distractor_pool)
            distractors = distractor_pool[:3]

            options = [gt_word] + distractors
            random.shuffle(options)
            gt_idx = options.index(gt_word)
            gt_option_letter = chr(ord("A") + gt_idx)

            # Generate video
            video_id = success_count + fail_count
            output_filename = f"synth_{video_id:04d}_ret_edit1.mp4"
            output_path = os.path.join(videos_output_dir, output_filename)

            try:
                actual_duration, needle_times = generate_video_multi_needle(
                    src_video,
                    output_path,
                    needles,
                    target_dur_actual,
                    num_frames=num_frames,
                )

                annotations.append(
                    {
                        "video": f"./videos/{output_filename}",
                        "question": f"What is the {ordinal} secret word?",
                        "options": options,
                        "needle_time": needle_times,
                        "needle_words": chosen_words,
                        "ask_index": ask_idx,
                        "gt": gt_word,
                        "gt_option": gt_option_letter,
                        "length": round(actual_duration, 2),
                        "num_needles": num_needles,
                        "type": "ret_edit1",
                        "try": 0,
                    }
                )

                success_count += 1
                if success_count % 50 == 0:
                    print(f"  Generated {success_count}/{num_videos} videos...")

            except Exception as e:
                print(f"  FAILED {output_filename}: {e}")
                fail_count += 1

    # Save annotations
    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nDone! Generated {success_count} videos. (Failed: {fail_count})")
    print(f"Annotations: {ann_path}")

    if annotations:
        lengths = [a["length"] for a in annotations]
        print(f"  Min length: {min(lengths):.1f}s")
        print(f"  Max length: {max(lengths):.1f}s")
        print(f"  Mean length: {np.mean(lengths):.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate VideoNIAH multi-needle ret_edit1 synthetic videos "
                    "(VNBench-style, DuoAttention-aligned)."
    )
    parser.add_argument("--num_videos", type=int, default=2000)
    parser.add_argument("--source_videos_dir", type=str, default="source_videos/extracted")
    parser.add_argument("--output_dir", type=str, default="synthetic_dataset")

    # Video length parameters — long enough to fill 32k context at ~196 tok/frame
    parser.add_argument("--min_duration", type=float, default=180.0,
                        help="Minimum video duration in seconds (3 min)")
    parser.add_argument("--max_duration", type=float, default=600.0,
                        help="Maximum video duration in seconds (10 min)")
    parser.add_argument("--num_length_intervals", type=int, default=20,
                        help="Number of linearly-spaced length intervals (matches data.py)")

    # Multi-needle parameters — aligned with DuoAttention paper
    parser.add_argument("--num_needles", type=int, default=10,
                        help="Number of secret word needles per video (DuoAttention=10)")
    parser.add_argument("--needle_duration", type=float, default=2.0,
                        help="How many seconds each subtitle is visible (VNBench default)")

    # Depth ratios — biased to first half for harder retrieval
    parser.add_argument("--min_depth_ratio", type=float, default=0.05,
                        help="Earliest allowed needle position (fraction of video)")
    parser.add_argument("--max_depth_ratio", type=float, default=0.5,
                        help="Latest allowed needle position (first half only)")
    parser.add_argument("--depth_num_intervals", type=int, default=20,
                        help="Number of linspace depth intervals to sample from")
    parser.add_argument("--num_frames", type=int, default=64,
                        help="Number of sampled frames. Ensures needles are placed on exact frames that will be sampled.")

    args = parser.parse_args()

    generate_dataset(
        num_videos=args.num_videos,
        source_videos_dir=args.source_videos_dir,
        output_dir=args.output_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_length_intervals=args.num_length_intervals,
        num_needles=args.num_needles,
        needle_duration=args.needle_duration,
        min_depth_ratio=args.min_depth_ratio,
        max_depth_ratio=args.max_depth_ratio,
        depth_num_intervals=args.depth_num_intervals,
        num_frames=args.num_frames,
    )
