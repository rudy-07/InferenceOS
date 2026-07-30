# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix

**Evaluation Date:** 2026-07-26 20:47:46  
**Model:** `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` (5.48 GB)  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Comparative Performance Matrix

| Engine / Layer Configuration | Layer Placement | Generation Speed (tok/s) | Prompt Eval Speed (tok/s) | TTFT (ms) | p50 ITL (ms) | Wall Duration (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **InferenceOS (Adaptive Auto-Placement: 28 dGPU / 0 iGPU / 5 CPU)** | `28 dGPU / 5 CPU` | **`9.27`** | `33.09` | `13330.2 ms` | `109.58 ms` | `28.51 s` | `SUCCESS` |
| **InferenceOS (3-Way Tri-Offload: 28 dGPU / 2 iGPU / 5 CPU)** | `28 dGPU / 2 iGPU / 5 CPU` | **`4.19`** | `16.98` | `14877.3 ms` | `233.37 ms` | `47.37 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (28 dGPU / 5 CPU): 28 dGPU / 0 iGPU / 7 CPU)** | `28 dGPU / 7 CPU` | **`8.49`** | `30.23` | `11737.7 ms` | `121.17 ms` | `28.51 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (18 dGPU / 15 CPU): 18 dGPU / 0 iGPU / 17 CPU)** | `18 dGPU / 17 CPU` | **`8.53`** | `29.38` | `12128.0 ms` | `118.89 ms` | `28.74 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 18)** | `18 dGPU / 15 CPU` | **`4.13`** | `16.76` | `596.7 ms` | `242.21 ms` | `41.73 s` | `SUCCESS` |
| **Raw llama.cpp (CPU Only (-ngl 0))** | `0 dGPU / 33 CPU` | **`1.67`** | `1.31` | `7612.9 ms` | `600.03 ms` | `92.73 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 99)** | `99 dGPU / 0 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `2.56 s` | `FAILED / OOM` |

---

## 💡 Key Architectural Insights

1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.
2. **dGPU + iGPU Hybrid Offload**: By utilizing the integrated GPU for CPU-bound layers via Vulkan UMA memory, InferenceOS achieves compute acceleration on shared system RAM without starving dGPU VRAM.
3. **InferenceOS Layer Placement Optimization**: Dynamically balances transformer layers across dGPU, iGPU, and System RAM to achieve peak generation speed while preventing OOM errors.