model_path="/home/yiliu7/workspace/llm-compressor/experimental/Llama-3.2-1B-Instruct-attention-fp8-head"
model_path="/home/yiliu7/workspace/llm-compressor/examples/quantization_kv_cache/Qwen2.5-0.5B-Instruct-FP8-KV"
model_path="/storage/yiliu7/Yi30/Qwen2.5-0.5B-Instruct-FP8_STATIC-fp8-attn-ar-llmc"
# disbale quantize query
# VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=1
model_path="/storage/yiliu7/Yi30/Qwen2.5-0.5B-Instruct-FP8_STATIC-fp8-attn-ar-llmc"
# VLLM_MLA_DISABLE=1 \
VLLM_TORCH_PROFILER_RECORD_SHAPES=1 \
VLLM_TORCH_PROFILER_WITH_FLOPS=1 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_ATTENTION_BACKEND=FLASHINFER \
VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION=0 \
VLLM_TORCH_PROFILER_DIR=./vllm_profiler/Qwen2.5-0.5B-ar-llmc-qkv-fp8-prof \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    nsys profile -o ./vllm_profiler/report-qkvfp8.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node \
    python /home/yiliu7/workspace/vllm/examples/offline_inference/basic/generate.py \
    --model ${model_path} \
    --tensor_parallel_size 1 \
    --max-tokens 16 \
    --max-num-seqs 4  \
    --gpu_memory_utilization 0.2 \
    --no-enable-prefix-caching \
    --kv_cache_dtype "fp8" \
    --enforce-eager 