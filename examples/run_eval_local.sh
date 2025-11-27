# BF16 KV
# vllm (pretrained=/storage/yiliu7/qwen/Qwen3-8B/,tensor_parallel_size=1,max_model_len=8192,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8954|±  |0.0084|
# |     |       |strict-match    |     5|exact_match|↑  |0.8961|±  |0.0084|
# qdq
# vllm (pretrained=/storage/yiliu7/qwen/Qwen3-8B/,tensor_parallel_size=1,max_model_len=8192,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8870|±  |0.0087|
# |     |       |strict-match    |     5|exact_match|↑  |0.8878|±  |0.0087|

# - quantp with 8 unit scale
# vllm (pretrained=/storage/yiliu7/qwen/Qwen3-8B/,tensor_parallel_size=1,max_model_len=8192,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: auto
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7301|±  |0.0122|
# |     |       |strict-match    |     5|exact_match|↑  |0.7597|±  |0.0118|

# !/bin/bash

if [ $# -gt 0 ] && [ "$1" == "--model_path" ]; then
    model_path=$2
else
    model_path="/storage/yiliu7/qwen/Qwen3-8B/"
fi

if [ $# -eq 4 ] && [ "$3" == "--tp_size" ]; then
    tp_size=$4
else
    tp_size=1
fi

model_name=$(basename ${model_path})
output_dir="${model_name}-tp${tp_size}-gsm8k-acc"
#limit=None

mkdir -p ${output_dir}
VLLM_USE_DEEP_GEMM=0 \
VLLM_LOGGING_LEVEL=DEBUG  \
VLLM_ENABLE_V1_MULTIPROCESSING=0  \
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
lm_eval --model vllm \
  --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=8192,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048" \
  --tasks gsm8k  \
    --batch_size 'auto' --log_samples --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log.txt
  
  