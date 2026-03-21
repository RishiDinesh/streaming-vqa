import os
import sys
import argparse
import cv2
import numpy as np

# Add the parent directory to sys.path so we can import duo_attn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duo_attn.data import DynamicSyntheticVideoQADataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, default="datasets/extracted")
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--num_needles", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="datasets/verification_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing DynamicSyntheticVideoQADataset...")
    dataset = DynamicSyntheticVideoQADataset(
        video_root=args.video_root,
        num_frames=args.num_frames,
        num_needles=args.num_needles,
        # We don't strictly need to download the full processor model weights just to verify the frames
        # but the dataset init loads AutoProcessor by default. Let's let it load.
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf"
    )

    print(f"Dataset initialized with {len(dataset.video_files)} background videos.")

    # Monkey patch to extract the raw PIL frames before they get destroyed by the processor
    original_build_model_inputs = dataset._build_model_inputs
    def patched_build_model_inputs(frames, prefix_text, full_text):
        out = original_build_model_inputs(frames, prefix_text, full_text)
        out["raw_pil_frames"] = frames
        out["full_text"] = full_text
        return out
    
    dataset._build_model_inputs = patched_build_model_inputs

    print(f"\nGenerating {args.num_videos} samples dynamically from the dataloader...")
    
    for i in range(args.num_videos):
        print(f"\n--- Generating Sample {i+1} ---")
        sample = dataset[i]
        frames = sample["raw_pil_frames"]
        text = sample["full_text"]
        
        # Save frames as mp4
        output_path = os.path.join(args.output_dir, f"sample_{i+1:03d}.mp4")
        
        # Assume 1 fps for the output visualization so it's easy to read
        fps = 1
        width, height = frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        expected_needle_frames = 0
        for frame_img in frames:
            # Convert PIL RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            # Simple heuristic check if frame has subtitle bar 
            # (look at bottom 15% for exact grey color (80,80,80) in RGB -> (80,80,80) in BGR)
            bar_region = frame_bgr[int(height*0.85)-30:int(height*0.85), :]
            if np.mean(bar_region == [80, 80, 80]) > 0.1: # if >10% of pixels match grey bar
                expected_needle_frames += 1
                
        out.release()
        
        print(f"Saved: {output_path}")
        print(f"QA Text Output:\n{text}")
        print(f"Frames with detected subtitle bar: {expected_needle_frames} / {args.num_needles}")
        
    print(f"\nVerification complete. Visual samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
