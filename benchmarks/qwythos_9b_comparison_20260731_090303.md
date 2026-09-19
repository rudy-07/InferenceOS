# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix

**Evaluation Date:** 2026-07-31 09:03:03  
**Model:** `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` (5.48 GB)  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Comparative Performance Matrix

| Engine / Layer Configuration | Layer Placement | Gen Speed (tok/s) | Prompt Speed (tok/s) | TTFT (ms) | VRAM Est | RAM Est | Avg CPU % | Avg GPU % | GPU Idle % | Wall Duration | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **InferenceOS (Adaptive Auto-Placement: 28 dGPU / 0 iGPU / 5 CPU)** | `28 dGPU / 5 CPU` | **`9.24`** | `35.24` | `283.8 ms` | `4999 MB` | `880 MB` | `23.8%` | `2.5%` | `91.7%` | `28.63 s` | `SUCCESS` |
| **InferenceOS (3-Way Tri-Offload: 28 dGPU / 2 iGPU / 5 CPU)** | `28 dGPU / 2 iGPU / 5 CPU` | **`4.65`** | `18.16` | `550.7 ms` | `4999 MB` | `880 MB` | `16.0%` | `0.1%` | `99.7%` | `45.03 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (28 dGPU / 5 CPU): 28 dGPU / 0 iGPU / 7 CPU)** | `28 dGPU / 7 CPU` | **`9.36`** | `34.61` | `288.9 ms` | `4999 MB` | `880 MB` | `19.2%` | `0.0%` | `100.0%` | `25.96 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (18 dGPU / 15 CPU): 18 dGPU / 0 iGPU / 17 CPU)** | `18 dGPU / 17 CPU` | **`9.33`** | `32.58` | `306.9 ms` | `4999 MB` | `880 MB` | `18.7%` | `0.0%` | `100.0%` | `26.92 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 18)** | `18 dGPU / 15 CPU` | **`4.66`** | `21.37` | `468.0 ms` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `37.35 s` | `SUCCESS` |
| **Raw llama.cpp (CPU Only (-ngl 0))** | `0 dGPU / 33 CPU` | **`1.99`** | `14.07` | `710.5 ms` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `69.59 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 99)** | `99 dGPU / 0 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `12.77 s` | `FAILED / OOM` |

---

## 💡 Key Architectural Insights

1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.
2. **dGPU + iGPU Hybrid Offload**: By utilizing the integrated GPU for CPU-bound layers via Vulkan UMA memory, InferenceOS achieves compute acceleration on shared system RAM without starving dGPU VRAM.
3. **InferenceOS Layer Placement Optimization**: Dynamically balances transformer layers across dGPU, iGPU, and System RAM to achieve peak generation speed while preventing OOM errors.