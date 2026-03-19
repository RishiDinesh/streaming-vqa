import os, glob, subprocess

# Crop top 150 VNBench videos and add to unedited_500
files = glob.glob("source_videos/extracted/VNBench_new/*_cnt_edit1.mp4")
out_dir = "source_videos/unedited_500"

durations = []
import cv2
for f in files:
    cap = cv2.VideoCapture(f)
    if cap.isOpened():
        durations.append((f, cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS)>0 else 0))
    cap.release()

# Take all 150 videos
to_crop = [f for f, d in durations]
print(f"Cropping {len(to_crop)} VNBench videos...")

start_idx = 501
for i, f in enumerate(to_crop):
    out_path = os.path.join(out_dir, f"{start_idx + i}.mp4")
    # Crop bottom 20% to remove any text
    cmd = ["ffmpeg", "-y", "-i", f, "-vf", "crop=iw:ih*0.8:0:0", "-c:a", "copy", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if (i+1) % 10 == 0:
        print(f"Cropped {i+1}/{len(to_crop)}")

print("Done cropping VNBench videos.")
