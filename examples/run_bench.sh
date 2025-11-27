model_path="deepseek-ai/DeepSeek-V2-Lite-Chat"
model_path="/home/yiliu7/workspace/models/deepseek-ai/Deepseek-v2-lite"
# model_path=inc-res/quantized_model_ds_mxfp8/

# VLLM_ENABLE_AR_EXT=1 \
# VLLM_AR_MXFP4_MODULAR_MOE=1 \
# VLLM_ENABLE_AR_EXT=1 \
# VLLM_MXFP4_PRE_UNPACK_TO_FP8=0 \
# VLLM_ENABLE_STATIC_MOE=0 \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_ENABLE_V1_MULTIPROCESSING=1 \
# vllm bench throughput \
#   --model $model_path \
#   --dataset-name sonnet \
#   --dataset-path ../../vllm/benchmarks/sonnet.txt \
#   --num-prompts 10
# - BS 10
# VLLM_MXFP4_PRE_UNPACK_TO_FP8=1 \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
# Throughput: 0.40 requests/s, 276.36 total tokens/s, 60.52 output tokens/s
# Total num prompt tokens:  5350
# Total num output tokens:  1500

# - BS 512
# Throughput: 12.86 requests/s, 8879.82 total tokens/s, 1929.68 output tokens/s
# Total num prompt tokens:  276610
# Total num output tokens:  76800
# run_bench.sh: line 62: syntax erro

# VLLM_MXFP4_PRE_UNPACK_TO_FP8=0 \
# VLLM_MXFP4_PRE_UNPACK_WEIGHTS=1 \
# Throughput: 3.39 requests/s, 2324.83 total tokens/s, 509.09 output tokens/s
# Total num prompt tokens:  5350
# Total num output tokens:  1500

# - BS 512
# Throughput: 32.56 requests/s, 22471.77 total tokens/s, 4883.37 output tokens/s
# Total num prompt tokens:  276610
# Total num output tokens:  76800

# model_path=quantized_model_qwen_mxfp4
# VLLM_LOGGING_LEVEL=DEBUG  \
model_path="/storage/yiliu7/Yi30/Qwen2.5-0.5B-Instruct-FP8_STATIC-fp8-attn-ar-llmc"
# VLLM_MLA_DISABLE=1 \
VLLM_TORCH_PROFILER_RECORD_SHAPES=1 \
VLLM_TORCH_PROFILER_WITH_FLOPS=1 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1 \
VLLM_TORCH_PROFILER_DIR=./vllm_profiler/Qwen2.5-0.5B-ar-llmc-kv-fp8-qbf16-2nd-prof \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
vllm bench throughput \
  --model $model_path \
  --enforce-eager \
  --dataset-name sonnet \
  --dataset-path /home/yiliu7/workspace/vllm/benchmarks/sonnet.txt \
  --num-prompts 32 \
  --profile \
  --input-len 64 \
  --tensor_parallel_size 1 \
  --output-len 32 \
  --max_model_len 512 \
  --kv_cache_dtype "fp8" \
    --enforce-eager 
  
# #!/bin/bash

# # Model Benchmarking Script
# # Usage: ./run_bench.sh

# # Define the model path
# # MODEL_PATH="/path/to/quantized_model"

# # Define the dataset
# DATASET_NAME="sonnet"
# DATASET_PATH="../../vllm/benchmarks/sonnet.txt"

# # Define the list of num-prompts to benchmark
# # NUM_PROMPTS_LIST=(32 64 128 256 512)
# NUM_PROMPTS_LIST=(4)

# Loop through each num-prompts value and run the benchmark
# for NUM_PROMPTS in "${NUM_PROMPTS_LIST[@]}"; do
#     echo "Benchmarking with num-prompts: $NUM_PROMPTS"

#     VLLM_ENABLE_AR_EXT=1 \
#     VLLM_AR_MXFP4_MODULAR_MOE=1 \
#     VLLM_ENABLE_AR_EXT=1 \
#     VLLM_MXFP4_PRE_UNPACK_TO_FP8=1 \
#     VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0 \
#     VLLM_ENABLE_STATIC_MOE=0 \
#     VLLM_USE_DEEP_GEMM=0 \
#     VLLM_ENABLE_V1_MULTIPROCESSING=1 \
#     vllm bench throughput \
#       --model $model_path \
#       --dataset-name $DATASET_NAME \
#       --dataset-path $DATASET_PATH \
#       --num-prompts $NUM_PROMPTS 

#     echo "Completed benchmarking with num-prompts: $NUM_PROMPTS"
#     echo "---------------------------------------------"
# done

# echo "All benchmarks completed."
  
# #   --profile
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,enable_expert_parallel=True" \
#   --tasks $task_name  \
#     --batch_size 16 \
#     --limit 256 \
#     --log_samples \
#     --seed 42 \
#     --profile \
#     --output_path ${output_dir} \
#     --show_config 2>&1 | tee ${output_dir}/log.txt
  
