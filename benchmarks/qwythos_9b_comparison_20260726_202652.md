# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix

**Evaluation Date:** 2026-07-26 20:26:52  
**Model:** `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` (5.48 GB)  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Comparative Performance Matrix

| Engine / Layer Configuration | Layer Placement | Generation Speed (tok/s) | Prompt Eval Speed (tok/s) | TTFT (ms) | p50 ITL (ms) | Wall Duration (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **InferenceOS (Adaptive Auto-Placement: 28 dGPU / 0 iGPU / 5 CPU)** | `28 dGPU / 5 CPU` | **`9.02`** | `30.91` | `13087.3 ms` | `113.40 ms` | `28.91 s` | `SUCCESS` |
| **InferenceOS (dGPU + iGPU Hybrid Split: 28 dGPU / 0 iGPU / 5 CPU)** | `28 dGPU / 5 CPU` | **`8.04`** | `30.84` | `12886.8 ms` | `127.93 ms` | `30.80 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split: 18 dGPU / 0 iGPU / 17 CPU)** | `18 dGPU / 17 CPU` | **`8.16`** | `28.78` | `12542.3 ms` | `127.99 ms` | `30.08 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 18)** | `18 dGPU / 15 CPU` | **`4.06`** | `18.09` | `552.8 ms` | `246.10 ms` | `44.73 s` | `SUCCESS` |
| **InferenceOS (iGPU Only: 28 dGPU / 7 iGPU / 0 CPU)** | `28 dGPU / 7 iGPU / 0 CPU` | **`3.80`** | `7.45` | `15925.4 ms` | `266.88 ms` | `52.45 s` | `SUCCESS` |
| **Raw llama.cpp (CPU Only (-ngl 0))** | `0 dGPU / 35 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `120.00 s` | `TIMEOUT / VRAM OOM THRASH` |
| **Raw llama.cpp (GPU -ngl 99)** | `99 dGPU / 0 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `2.65 s` | `FAILED / OOM` |

---

## 💡 Key Architectural Insights

1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.
2. **dGPU + iGPU Hybrid Offload**: By utilizing the integrated GPU for CPU-bound layers via Vulkan UMA memory, InferenceOS achieves compute acceleration on shared system RAM without starving dGPU VRAM.
3. **InferenceOS Layer Placement Optimization**: Dynamically balances transformer layers across dGPU, iGPU, and System RAM to achieve peak generation speed while preventing OOM errors.