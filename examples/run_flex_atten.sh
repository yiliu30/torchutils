model_name=/storage/yiliu7/qwen/Qwen3-8B/
# VLLM_USE_DEEP_GEMM=0 \
# VLLM_LOGGING_LEVEL=DEBUG  \
# VLLM_ENABLE_V1_MULTIPROCESSING=0  \
# VLLM_ATTENTION_BACKEND=FLEX_ATTENTION  python \
#     generate.py \
#     --model ${model_name} \
#     --gpu-memory-utilization  0.8 
    
    

# model_path="/data5/yliu7/tmp/Qwen2.5-0.5B-W4A16-G128"
# model_path="/data5/yliu7/tmp/Meta-Llama-3.1-8B-Instruct-W4A16-G128"
# model_path="/data5/yliu7/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct-AR-W4G128"
tp_size=1
VLLM_USE_DEEP_GEMM=0 \
VLLM_LOGGING_LEVEL=DEBUG  \
VLLM_ENABLE_V1_MULTIPROCESSING=0  \
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
vllm serve $model_name \
    --max-model-len 8192 \
    --max-num-batched-tokens 32768 \
    --tensor-parallel-size $tp_size \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --port 8099 \
    --no-enable-prefix-caching \
    --trust-remote-code  2>&1 | tee $log_file
    
