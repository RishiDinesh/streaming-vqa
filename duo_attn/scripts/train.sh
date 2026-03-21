export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8

model_name=${1}
ctx_len_min=${2}
ctx_len_max=${3}
reg_weight=${4}
lr=${5}
num_passkey=${6}
output_root=attn_patterns/${model_name}

torchrun --nnodes 1 --nproc_per_node 8 \
    duo_attn/train.py \
    --model_name models/${model_name} \
    --batch_size 1 \
    --max_length ${ctx_len_max} \
    --dataset_name "datasets/booksum.jsonl.zst" \
    --sink_size 128 \
    --recent_size 256 \
    --num_steps 2000 \
    --lr ${lr} \
    --reg_weight ${reg_weight} \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min ${ctx_len_min} \
    --context_length_max ${ctx_len_max} \
    --context_lengths_num_intervals 50 \
    --depth_ratio_num_intervals 1000 \
    --gradient_accumulation_steps 1 \
    --num_passkey ${num_passkey} \
    --dataset_format "multiple_passkey" \
    --output_dir ${output_root}
