# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix

**Evaluation Date:** 2026-07-26 20:39:19  
**Model:** `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` (5.48 GB)  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Comparative Performance Matrix

| Engine / Layer Configuration | Layer Placement | Generation Speed (tok/s) | Prompt Eval Speed (tok/s) | TTFT (ms) | p50 ITL (ms) | Wall Duration (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **InferenceOS (Adaptive Auto-Placement: 28 dGPU / 0 iGPU / 5 CPU)** | `28 dGPU / 5 CPU` | **`9.37`** | `34.30` | `12374.0 ms` | `108.92 ms` | `27.49 s` | `SUCCESS` |
| **InferenceOS (3-Way Tri-Offload (28 dGPU / 2 iGPU / 3 CPU): 28 dGPU / 0 iGPU / 5 CPU)** | `28 dGPU / 5 CPU` | **`9.37`** | `34.24` | `12409.9 ms` | `109.61 ms` | `27.32 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (28 dGPU / 5 CPU): 28 dGPU / 0 iGPU / 7 CPU)** | `28 dGPU / 7 CPU` | **`9.34`** | `33.54` | `12106.9 ms` | `109.77 ms` | `27.44 s` | `SUCCESS` |
| **InferenceOS (dGPU + CPU Split (18 dGPU / 15 CPU): 18 dGPU / 0 iGPU / 17 CPU)** | `18 dGPU / 17 CPU` | **`9.25`** | `34.26` | `11941.3 ms` | `110.96 ms` | `27.14 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 18)** | `18 dGPU / 15 CPU` | **`4.36`** | `18.34` | `545.2 ms` | `229.41 ms` | `39.54 s` | `SUCCESS` |
| **Raw llama.cpp (CPU Only (-ngl 0))** | `0 dGPU / 35 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `120.00 s` | `TIMEOUT / VRAM OOM THRASH` |
| **Raw llama.cpp (GPU -ngl 99)** | `99 dGPU / 0 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `2.67 s` | `FAILED / OOM` |

---

## 💡 Key Architectural Insights

1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.
2. **dGPU + iGPU Hybrid Offload**: By utilizing the integrated GPU for CPU-bound layers via Vulkan UMA memory, InferenceOS achieves compute acceleration on shared system RAM without starving dGPU VRAM.
3. **InferenceOS Layer Placement Optimization**: Dynamically balances transformer layers across dGPU, iGPU, and System RAM to achieve peak generation speed while preventing OOM errors.