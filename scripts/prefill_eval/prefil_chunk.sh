python -u -m duo_attn.eval.efficiency.prefill_eval_llava prefill \
  --model_name models/llava-hf-llava-onevision-qwen2-7b-ov-hf \
  --video_path data/samplee.mp4 \
  --output_dir untracked/7b/prefill_chunk_sweep_32k \
  --attn_load_dir outputs/7b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1_20260320_154812 \
  --prompt_file long_prompt.txt \
  --target_context 32000 \
  --max_length 32000 \
  --max_num_frames 512 \
  --prefill_chunk_sizes 4000 8000 12000 16000 20000 24000 28000 32000 \
  --threshold 0.5 \
  --sparsity 0.5

# python -u -m duo_attn.eval.efficiency.prefill_eval_llava context \
#   --model_name models/llava-hf-llava-onevision-qwen2-0.5b-ov-hf \
#   --video_path data/samplee.mp4 \
#   --output_dir untracked/context_sweep_32k \
#   --attn_load_dir outputs/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1 \
#   --prompt_file long_prompt.txt \
#   --max_length 32000 \
#   --max_context 32000 \
#   --target_contexts 4000 8000 12000 16000 20000 24000 28000 32000
