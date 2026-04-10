python live_llava_video_debug.py \
  --video_path data/samplee_45min.mp4 \
  --num_frames 100 \
  --max_length 32000 \
  --max_new_tokens 256 \
  --disable_auto_frame_backoff \
  --prompt "whats in the video, explain in detail" \
  --stream_every 1 \
  --report_json outputs/live_llava_video_debug_report.json
