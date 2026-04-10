python -m duo_attn.eval.efficiency.context_eval_llava run \
  --model_name models/llava-hf-llava-onevision-qwen2-0.5b-ov-hf \
  --video_path data/samplee.mp4 \
  --output_dir untracked/0.5b/context_sweep_32k \
  --attn_load_dir outputs/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1 \
  --prompt_file long_prompt.txt \
  --max_length 32000 \
  --max_context 32000 \
  --target_contexts 4000 8000 12000 16000 20000 24000 28000 32000


# python -m duo_attn.eval.efficiency.context_eval_llava plot \
#   --input_json untracked/context_sweep_32k/context_sweep_results.json
