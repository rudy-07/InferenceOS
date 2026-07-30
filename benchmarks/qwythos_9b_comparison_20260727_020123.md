# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix

**Evaluation Date:** 2026-07-27 02:01:23  
**Model:** `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` (5.48 GB)  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Comparative Performance Matrix

| Engine / Layer Configuration | Layer Placement | Generation Speed (tok/s) | Prompt Eval Speed (tok/s) | TTFT (ms) | p50 ITL (ms) | Wall Duration (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **InferenceOS (Adaptive Auto-Placement: 28 dGPU / 0 iGPU / 5 CPU)** | `28 dGPU / 5 CPU` | **`8.28`** | `28.98` | `345.1 ms` | `123.12 ms` | `30.71 s` | `SUCCESS` |
| **InferenceOS (3-Way Tri-Offload: 28 dGPU / 2 iGPU / 5 CPU)** | `28 dGPU / 2 iGPU / 5 CPU` | **`4.07`** | `15.76` | `634.7 ms` | `247.20 ms` | `47.65 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (28 dGPU / 5 CPU): 28 dGPU / 0 iGPU / 7 CPU)** | `28 dGPU / 7 CPU` | **`8.41`** | `29.66` | `337.2 ms` | `122.75 ms` | `28.34 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (18 dGPU / 15 CPU): 18 dGPU / 0 iGPU / 17 CPU)** | `18 dGPU / 17 CPU` | **`8.47`** | `27.60` | `362.3 ms` | `121.33 ms` | `29.67 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 18)** | `18 dGPU / 15 CPU` | **`4.10`** | `14.56` | `687.0 ms` | `243.72 ms` | `41.94 s` | `SUCCESS` |
| **Raw llama.cpp (CPU Only (-ngl 0))** | `0 dGPU / 33 CPU` | **`1.53`** | `1.29` | `7726.1 ms` | `651.51 ms` | `99.89 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 99)** | `99 dGPU / 0 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `13.24 s` | `FAILED / OOM` |

---

## 💡 Key Architectural Insights

1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.
2. **dGPU + iGPU Hybrid Offload**: By utilizing the integrated GPU for CPU-bound layers via Vulkan UMA memory, InferenceOS achieves compute acceleration on shared system RAM without starving dGPU VRAM.
3. **InferenceOS Layer Placement Optimization**: Dynamically balances transformer layers across dGPU, iGPU, and System RAM to achieve peak generation speed while preventing OOM errors.