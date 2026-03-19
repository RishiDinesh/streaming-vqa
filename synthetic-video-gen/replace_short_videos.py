import os, glob, cv2

# Get durations of all 500 in unedited_500
files = glob.glob("source_videos/unedited_500/*.mp4")
durations = []
for f in files:
    cap = cv2.VideoCapture(f)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        durations.append((f, cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps if fps>0 else 0))
    cap.release()

# Sort from shortest to longest
durations.sort(key=lambda x: x[1])

# Delete the 350 shortest videos to push the average up to ~60s overall once refilled
to_delete = durations[:350]

for f, d in to_delete:
    os.remove(f)

print(f"Deleted the shortest {len(to_delete)} videos.")
print(f"Remaining videos average length: {sum(d for f, d in durations[350:])/150:.1f}s")
