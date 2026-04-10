python -u duo_attn/eval/efficiency/context_eval_llava.py run \
  --model_name /root/streaming-vqa/models/llava-hf-llava-onevision-qwen2-7b-ov-hf   \
  --video_path /root/streaming-vqa/data/samplee.mp4   \
  --output_dir /root/streaming-vqa/untracked/context_eval_fig9_7b   \
  --attn_load_dir /root/streaming-vqa/outputs/7b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1_20260320_154812   \
  --prompt "Describe this video in detail."   \
  --max_length 32000   \
  --max_context 32000   \
  --target_contexts 4000 8000 12000 16000 24000 32000   \
  --decode_tokens 100   \
  --sparsity 0.5


# python -m duo_attn.eval.efficiency.context_eval_llava plot \
#   --input_json untracked/context_sweep_32k/context_sweep_results.json
