# Frequently Asked Questions (FAQ) & Troubleshooting

### Q1: How does InferenceOS prevent Out-Of-Memory (OOM) errors?
InferenceOS uses a multi-layered approach:
1. **Pre-flight Memory Planner**: Computes weight and KV cache footprints before execution and offloads excess layers to CPU RAM.
2. **Dynamic Microbatch Scheduler**: Scales prompt prefill batch sizes dynamically based on free VRAM headroom.
3. **Live Hysteresis Layer Migration**: Dynamically migrates layers from GPU to CPU during active execution if system VRAM pressure spikes.

### Q2: Is InferenceOS fully compatible with `llama.cpp` GGUF models?
Yes! InferenceOS uses `llama.cpp` C++ kernels under the hood and is 100% binary compatible with standard GGUF model files.

### Q3: How do I run the OpenAI API server?
Use the CLI `serve` subcommand:
```bash
inferenceos serve models/llama-3-8b.Q4_K_M.gguf --port 11434
```

### Q4: Which GPU backends are supported?
NVIDIA CUDA, AMD ROCm/HIP, Vulkan (cross-platform for AMD/Intel/NVIDIA), Apple Metal, and generic CPU vector extensions (AVX2, AVX-512, AMX, ARM NEON).

### Q5: How do I run system diagnostics?
Run the built-in diagnostic doctor:
```bash
inferenceos doctor
```
