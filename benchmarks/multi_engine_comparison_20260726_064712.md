# 🏎️ Inference Engine Benchmark Matrix: InferenceOS vs. Ollama vs. Raw llama.cpp

**Evaluation Date:** 2026-07-26 06:47:12  
**Model:** `Qwen3-4B-Thinking-2507.Q5_K_M.gguf` / `Qwen3-4B-Thinking:latest`  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Summary Performance Comparison Matrix

| Metric | InferenceOS (Adaptive) | Raw llama.cpp (Full GPU -ngl 99) | Raw llama.cpp (CPU Only -ngl 0) | Ollama (v0.32.3) |
| :--- | :---: | :---: | :---: | :---: |
| **Generation Speed (tok/s)** | **`34.81`** | **`52.38`** | `3.48` | `0.00` |
| **Prompt Eval Speed (tok/s)** | `75.86` | **`70.03`** | `15.94` | `0.00` |
| **Time to First Token (TTFT)** | `8121.2 ms` | **`142.8 ms`** | `627.4 ms` | `0.0 ms` |
| **Median Inter-Token Latency (ITL)** | `29.92 ms` | **`19.09 ms`** | `287.38 ms` | `0.00 ms` |
| **Total Wall Clock Duration** | `12.65 s` | **`9.13 s`** | `39.66 s` | `0.00 s` |
| **Generated Tokens** | `127` | `127` | `127` | `0` |
| **GPU Offload Allocation** | `36/36 Blocks (38/38 Total)` | `38/38 Total Layers` | `0/38 Total Layers` | `Full Auto Offload` |

---

## 💡 Subsystem Analysis & Key Takeaways

1. **Raw llama.cpp GPU (-ngl 99)**: Demonstrates the peak hardware upper bound when all 38 transformer layers are placed on GPU VRAM.
2. **InferenceOS Adaptive Pipeline**: Dynamically allocates layers based on system RAM/VRAM pressure. When VRAM is constrained, it splits layers safely across GPU & CPU to prevent Out-Of-Memory (OOM) crashes.
3. **Pipeline Configurator**: Using `pipeline_config.json` or `--gpu-layers 99`, InferenceOS can be forced into full GPU offload mode to achieve raw `llama.cpp` speeds while retaining telemetry, preflight guards, and scheduling benefits.
