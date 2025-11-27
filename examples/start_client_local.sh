# OPENAI_API_KEY=None  python -m gpt_oss.evals --model /data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/  --eval aime25 --base-url http://localhost:8099/v1 --reasoning-effort low 


#  curl http://127.0.0.1:8088/metrics

export no_proxy="localhost, 127.0.0.1, ::1"
task_name=lambada_openai
# hellaswag,gsm8k,mmlupro
# task_name="arc_challenge,hellaswag,lambada_openai,mmlu,truthfulqa_mc1,winogrande"
task_name=gsm8k
# task_name=mmlu
batch_size=8
# LIMIT=128
timestamp=$(date +%Y%m%d_%H%M%S)
EVAL_LOG_NAME="eval__${timestamp}"
max_length=8192
max_gen_toks=4096

mkdir -p benchmark_logs
model_path=/home/yliu7/workspace/inc/3rd-party/llm-compressor/examples/quantization_non_uniform/Llama-3.2-1B-Instruct-NVFP4-FP8-Dynamic
model_path="/data5/yliu7/HF_HOME/qwen_moe_skip_lm_head"
model_path=/data5/yliu7/HF_HOME/ByteDance-Seed/Seed-OSS-36B-Instruct
model_path=/data5/yliu7/HF_HOME/GLM-4.5-Air-w8afp8-llmc/GLM-4.5-Air-w8afp8
# model_path=/data5/yliu7/HF_HOME/Llama-3.2-1B-Instruct-NVFPP_B16/
# model_path=/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/
model_path=/data5/yliu7/HF_HOME/Yi30/gpt-oss-20b-BF16-MXFP8/
model_path=/data5/yliu7/HF_HOME/Yi30/gpt-oss-120b-BF16-unsloth-MXFP8
model_path=/data6/yiliu4/unsloth-gpt-oss-120b-BF16-ar-MXFP4/
model_path="/data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/"
model_path="/data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/"
model_path="/data5/yliu7/tmp/Qwen2.5-0.5B-W4A16-G128"
model_path="/data5/yliu7/tmp/Meta-Llama-3.1-8B-Instruct-W4A16-G128"
model_path="/data5/yliu7/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct-AR-W4G128"
model_path="/data5/yliu7/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct-AR-W4G128"
model_path=/storage/yiliu7/qwen/Qwen3-8B/
# model_path=/data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/
# model_path=/data6/yiliu4/unsloth-gpt-oss-120b-BF16-ar-MXFP4/
# model_path=/data5/yliu7/HF_HOME/Yi30/unsloth-gpt-oss-20b-BF16-MXFP4
# model_path=/models/DeepSeek-V2-Lite-Chat/
port=8099
# HF_ALLOW_CODE_EVAL=1 \
# lm_eval --model local-completions \
#     --tasks $task_name \
#     --model_args model=${model_path},base_url=http://127.0.0.1:${port}/v1/completions,num_concurrent=8,max_retriess=1000,max_length=${max_length},max_gen_toks=${max_gen_toks} \
#     --batch_size ${batch_size}  \
#     --gen_kwargs="max_gen_toks=${max_gen_toks}" \
#     --confirm_run_unsafe_code \
#     --log_samples \
#     --include_path /home/yliu7/workspace/inc/3rd-party/vllm/examples/offline_inference/basic/gpt_oss_gsm8k/ \
#     --output_path "benchmark_logs/$EVAL_LOG_NAME" \
#     2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"

# --include_path /home/yliu7/workspace/inc/3rd-party/vllm/examples/offline_inference/basic/gpt_oss_gsm8k/ \



# --include_path /home/yliu7/workspace/inc/3rd-party/vllm/examples/offline_inference/basic/gpt_oss_gsm8k/ \

    # --apply_chat_template \

# --use_cache "benchmark_logs/$EVAL_LOG_NAME" \

# arc_challenge,hellaswag,lambada_openai,mmlu,truthfulqa_mc1,winogrande
HF_ALLOW_CODE_EVAL=1 \
lm_eval --model  local-completions  \
    --tasks  $task_name \
    --model_args model=${model_path},base_url=http://127.0.0.1:${port}/v1/completions,num_concurrent=1500,max_retries=1000,timeout=6000,max_length=${max_length},max_gen_toks=${max_gen_toks},add_bos_token=True \
    --batch_size 128  \
    --gen_kwargs="max_gen_toks=${max_gen_toks}" \
    --confirm_run_unsafe_code \
    --log_samples \
    --output_path "benchmark_logs/$EVAL_LOG_NAME" \
    2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"



# curl -X POST http://127.0.0.1:8099/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "/data5/yliu7/tmp/Qwen2.5-0.5B-W4A16-G128",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'


# HF_ALLOW_CODE_EVAL=1 \
# lm_eval --model local-chat-completions \
#     --tasks $task_name \
#     --apply_chat_template \
#     --model_args model=${model_path},base_url=http://127.0.0.1:${port}/v1/chat/completions,num_concurrent=1500,max_retries=1000,timeout=600,max_length=${max_length},max_gen_toks=${max_gen_toks} \
#     --batch_size 1  \
#     --gen_kwargs="max_gen_toks=${max_gen_toks}" \
#     --confirm_run_unsafe_code \
#     --log_samples \
#     --limit 64 \
#     --output_path "benchmark_logs/$EVAL_LOG_NAME" \
#     --use_cache "benchmark_logs/$EVAL_LOG_NAME" \
#     2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"


# HF_ALLOW_CODE_EVAL=1 \
# lm_eval --model local-chat-completions \
#     --tasks $task_name \
#     --apply_chat_template \
#     --model_args model=${model_path},base_url=http://127.0.0.1:${port}/v1/chat/completions,num_concurrent=512,max_retries=100,timeout=10000,max_length=${max_length},max_gen_toks=${max_gen_toks} \
#     --batch_size 1  \
#     --gen_kwargs="max_gen_toks=${max_gen_toks}" \
#     --confirm_run_unsafe_code \
#     --log_samples \
#     --limit 128 \
#     --include_path /home/yliu7/workspace/inc/3rd-party/vllm/examples/offline_inference/basic/gpt_oss_gsm8k/ \
#     --output_path "benchmark_logs/$EVAL_LOG_NAME" \
#     --use_cache "benchmark_logs/$EVAL_LOG_NAME" \
#     2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"

# --use_cache /home/yliu7/workspace/inc/3rd-party/vllm/examples/offline_inference/basic/benchmark_logs/eval_gsm8k_20251001_223738_rank0.db \
# For next token tasks, we need use local-completions.
# task_name=mmlu
# HF_ALLOW_CODE_EVAL=1 \
# lm_eval --model local-completions \
#     --tasks $task_name \
#     --model_args model=${model_path},base_url=http://127.0.0.1:${port}/v1/completions,num_concurrent=1500,max_retries=100,timeout=6000,max_length=${max_length},max_gen_toks=${max_gen_toks} \
#     --batch_size ${batch_size}  \
#     --gen_kwargs="max_gen_toks=${max_gen_toks}" \
#     --confirm_run_unsafe_code \
#     --limit 8 \
#     --log_samples \
#     --include_path /home/yliu7/workspace/inc/3rd-party/vllm/examples/offline_inference/basic/gpt_oss_gsm8k/ \
#     --output_path "benchmark_logs/$EVAL_LOG_NAME" \
#     --use_cache "benchmark_logs/$EVAL_LOG_NAME" \
#     2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"




# curl -X POST http://127.0.0.1:8099/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "/data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'

# curl -X POST http://127.0.0.1:8000/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "/models/Qwen3-0.6B/",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'


# batch_size=4
# lm_eval --model local-completions \
#     --tasks $task_name \
#     --model_args model=${model_path},base_url=http://127.0.0.1:8687/v1/completions,max_concurrent=1,max_length=${max_length},max_gen_toks=${max_gen_toks} \
#     --batch_size ${batch_size}  \
#     --gen_kwargs="max_length=${max_length},max_gen_toks=${max_gen_toks}" \
#     --confirm_run_unsafe_code \
#     --log_samples \
#     --limit 16 \
#     --output_path "benchmark_logs/$EVAL_LOG_NAME" \
#     2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"

# curl -X POST http://127.0.0.1:8687/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "seed_oss",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'

