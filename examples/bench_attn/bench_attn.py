import torch
import time
from transformers import AutoConfig

# Assuming the import path works in your environment
from transformers.models.qwen3.modular_qwen3 import Qwen3Attention
from loguru import logger

# --- 1. Setup Configuration ---
model_name = "/data/yiliu/models/Qwen/Qwen3-8B"
# model_name = "/data5/yliu7/HF_HOME/Qwen/Qwen3-8B/"
# Load config
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.use_cache = False  # Disable KV cache for training/benchmarking logic


# Benchmarking Parameters
batch_size = 8
seq_len = 2048
hidden_size = config.hidden_size
num_heads = config.num_attention_heads
head_dim = hidden_size // num_heads

dtype = torch.bfloat16
device = "xpu"


def benchmark_implementation(impl_name, forward_only=False, num_steps=10):
    mode_str = "Forward Only" if forward_only else "Forward + Backward"
    logger.info(f"\n--- Benchmarking: {impl_name} [{mode_str}] ---")

    # Update config implementation
    config._attn_implementation = impl_name

    try:
        attn_mod = Qwen3Attention(config, layer_idx=0).to(device).to(dtype)
    except Exception as e:
        logger.info(f"Skipping {impl_name}: Could not initialize. Error: {e}")
        return

    # 2. Prepare Data
    # Only require gradients if we are testing backward pass
    req_grad = not forward_only

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=req_grad)

    # Generate Position Embeddings (RoPE) - Cos/Sin tuple
    cos = torch.randn(1, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(1, seq_len, head_dim, device=device, dtype=dtype)
    position_embeddings = (cos, sin)

    attention_mask = None

    # Helper function to run one step
    def run_step():
        if forward_only:
            with torch.no_grad():
                output, _ = attn_mod(hidden_states, position_embeddings, attention_mask=attention_mask)
            return output
        else:
            output, _ = attn_mod(hidden_states, position_embeddings, attention_mask=attention_mask)
            loss = output.mean()
            loss.backward()
            # Zero gradients
            hidden_states.grad = None
            for param in attn_mod.parameters():
                param.grad = None
            return output

    # 3. Warmup
    logger.info("  Warming up...")
    try:
        for _ in range(10):
            run_step()
    except Exception as e:
        logger.info(f"  Error during warmup: {e}")
        return

    torch.xpu.synchronize()

    # 4. Benchmark Loop
    logger.info("  Running benchmark...")
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)

    start_event.record()
    from auto_round.compressors.profiler_wrapper import XPUTorchProfilerWrapper as Profiler
    # profiler = Profiler(worker_name=f"attn_{impl_name}_{'fwd' if forward_only else 'fwd_bwd'}", local_rank=0)
    # profiler.start()
    for _ in range(num_steps):
        
        run_step()
        # profiler.step()
    # profiler.stop()

    end_event.record()
    torch.xpu.synchronize()

    # 5. Stats
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_per_step = elapsed_time_ms / num_steps

    logger.info(f"  > Average Time: {avg_time_per_step:.2f} ms")

    # Approx TFLOPS Calculation
    # Base FLOPs for Forward pass (Linear Projections + Attention)
    # Projections (Q,K,V,O): 4 projections * 2 (mul+add) * B * S * H^2 -> 8 B S H^2
    # Attention (Matmuls): ~ 4 * B * S^2 * H
    fwd_flops = (8 * batch_size * seq_len * hidden_size**2) + (4 * batch_size * seq_len**2 * hidden_size)

    # Backward pass is roughly 2x Forward pass cost
    total_flops_per_step = fwd_flops if forward_only else (fwd_flops * 3)

    tflops = (total_flops_per_step / (avg_time_per_step / 1000)) / 1e12
    # logger.info(f"  > Approx TFLOPS: {tflops:.2f}")


# --- Run Benchmarks ---
implementations = [
    # "flash_attention_2",
    "sdpa",
    "flex_attention",
    "eager",
]

# Test Forward Only
# logger.info("=== MODE: FORWARD ONLY ===")
# for impl in implementations:
#     torch.xpu.empty_cache()
#     benchmark_implementation(impl, forward_only=True)

# Test Forward + Backward
logger.info("\n=== MODE: FORWARD + BACKWARD ===")
for impl in implementations:
    torch.xpu.empty_cache()
    benchmark_implementation(impl, forward_only=False)
