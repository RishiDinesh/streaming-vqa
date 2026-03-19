# Dynamic Synthetic Dataloader Setup & Instructions

We have upgraded the training dataloader to generate multi-needle video data dynamically on-the-fly. This replaces the need to pre-render and store gigabytes of MP4s.

## Step 1: Prep the Node
First, download the raw unedited video pool directly to the cluster (500 videos, ~76.5s average length). We have a pre-configured script for this.
```bash
pip install gdown
./duo_attn/scripts/download_from_gdrive.sh
```
This will automatically pull the ZIP from Google Drive and neatly extract exactly 500 numbered `.mp4` background videos into `video_ds/unedited_500` at the project root.

## Step 2: Training Command
Run your standard training cluster script (`run_train.sh`) but ensure it points to the unedited source root and includes the dynamic flags. 

```bash
# Example snippet inside run_train.sh
python -m duo_attn.train \
    --video_dataset_name dynamic_synthetic \
    --video_root video_ds/unedited_500 \
    --num_needles 5 \
    --min_needle_depth_ratio 0.2 \
    --max_needle_depth_ratio 0.8 \
    # ... [your other args like --num_frames 64, --model_name, etc]
```

## Configurable Parameters

*   `--video_root`: Must point to the unedited background videos folder (`video_ds/unedited_500`).
*   `--num_needles`: Defines how many distinct secret words are inserted into the video sample. (Default: `5`)
*   `--min_needle_depth_ratio` / `--max_needle_depth_ratio`: Controls the valid temporal interval where needles drop. e.g. `0.2` to `0.8` restricts needles from popping up in the first and last 20% of the parsed frames. 
*   `--frame_idx`: **[Optional]** Allows you to explicitly lock the exact frame insertions rather than relying on randomized depth ratios. Pass a space-separated list of exact frame indices matching your `num_needles` count. *(Example: If `--num_frames 64`, `--num_needles 3`, you can pass `--frame_idx 10 32 50` to force needles strictly onto the 11th, 33rd, and 51st sampled frames).*

## Core Design Choices
1. **On-The-Fly Burning**: Needles are burned onto the raw images natively during `__getitem__`. This allows us to train on "infinite" random variations and random words without ever touching disk storage or `ffmpeg`.
2. **Perfect Frame Alignment**: By burning the text onto the array *after* the initial 64/128 chunk is parsed across the timeline, the needle is guaranteed to perfectly map to exactly the frames the Vision processor actually sees. This completely squashes "flicker" or desync bugs.
3. **Dynamic Prompt Templates**: The expected Target Text dynamically adjusts based on the requested index. Rather than stripping context to `"The secret word is: TARGET"`, the answer generator is perfectly aware of its placement, generating exact target alignments like `"The second secret word is: TARGET"`.
4. **VNBench Consistency**: The visual subtitles rigorously obey the exact VNBench specifications (`f"The {ordinal} secret word is {word}"` visually overlaid on an 80-80-80 RGB grey box mounted at `85%` screen depth).
