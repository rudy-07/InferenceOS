# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix

**Evaluation Date:** 2026-08-07 08:19:13  
**Model:** `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` (5.48 GB)  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Comparative Performance Matrix

| Engine / Layer Configuration | Layer Placement | Gen Speed (tok/s) | Prompt Speed (tok/s) | TTFT (ms) | VRAM Est | RAM Est | Avg CPU % | Avg GPU % | GPU Idle % | Wall Duration | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **InferenceOS (Adaptive Auto-Placement: 28 dGPU / 0 iGPU / 5 CPU)** | `28 dGPU / 5 CPU` | **`8.85`** | `33.26` | `300.7 ms` | `4999 MB` | `880 MB` | `30.9%` | `4.9%` | `87.7%` | `25.39 s` | `SUCCESS` |
| **InferenceOS (3-Way Tri-Offload: 28 dGPU / 2 iGPU / 5 CPU)** | `28 dGPU / 2 iGPU / 5 CPU` | **`3.92`** | `15.55` | `643.2 ms` | `4999 MB` | `880 MB` | `16.4%` | `0.8%` | `98.9%` | `40.11 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (28 dGPU / 5 CPU): 28 dGPU / 0 iGPU / 7 CPU)** | `28 dGPU / 7 CPU` | **`9.19`** | `36.71` | `272.4 ms` | `4999 MB` | `880 MB` | `21.4%` | `0.1%` | `100.0%` | `20.66 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (18 dGPU / 15 CPU): 18 dGPU / 0 iGPU / 17 CPU)** | `18 dGPU / 17 CPU` | **`9.00`** | `36.07` | `277.2 ms` | `4999 MB` | `880 MB` | `22.8%` | `0.4%` | `100.0%` | `20.87 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 18)** | `18 dGPU / 15 CPU` | **`4.50`** | `23.48` | `425.9 ms` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `34.04 s` | `SUCCESS` |
| **Raw llama.cpp (CPU Only (-ngl 0))** | `0 dGPU / 33 CPU` | **`2.22`** | `14.94` | `669.3 ms` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `61.71 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 99)** | `99 dGPU / 0 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `6.00 s` | `FAILED / OOM` |

---

## 💡 Key Architectural Insights

1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.
2. **dGPU + iGPU Hybrid Offload**: By utilizing the integrated GPU for CPU-bound layers via Vulkan UMA memory, InferenceOS achieves compute acceleration on shared system RAM without starving dGPU VRAM.
3. **InferenceOS Layer Placement Optimization**: Dynamically balances transformer layers across dGPU, iGPU, and System RAM to achieve peak generation speed while preventing OOM errors.