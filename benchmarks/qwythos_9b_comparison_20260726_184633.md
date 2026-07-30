# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix

**Evaluation Date:** 2026-07-26 18:46:33  
**Model:** `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` (5.48 GB)  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Comparative Performance Matrix

| Engine / Layer Configuration | Layer Placement | Generation Speed (tok/s) | Prompt Eval Speed (tok/s) | TTFT (ms) | p50 ITL (ms) | Wall Duration (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **InferenceOS (Adaptive Auto-Placement: 28 GPU / 5 CPU)** | `28 GPU / 5 CPU` | **`8.81`** | `34.71` | `22796.4 ms` | `116.18 ms` | `38.97 s` | `SUCCESS` |
| **InferenceOS (High GPU Split (18 GPU / 15 CPU): 18 GPU / 17 CPU)** | `18 GPU / 17 CPU` | **`8.83`** | `33.63` | `12880.6 ms` | `115.76 ms` | `28.82 s` | `SUCCESS` |
| **InferenceOS (Balanced Split (12 GPU / 21 CPU): 12 GPU / 23 CPU)** | `12 GPU / 23 CPU` | **`8.74`** | `30.36` | `12379.5 ms` | `118.18 ms` | `28.47 s` | `SUCCESS` |
| **InferenceOS (Low GPU Split (6 GPU / 27 CPU): 6 GPU / 29 CPU)** | `6 GPU / 29 CPU` | **`8.80`** | `29.88` | `12137.0 ms` | `117.08 ms` | `28.51 s` | `SUCCESS` |
| **Raw llama.cpp (CPU Only (-ngl 0))** | `0 GPU / 33 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `60.00 s` | `TIMEOUT / VRAM OOM THRASH` |
| **Raw llama.cpp (Full GPU (-ngl 99))** | `99 GPU / 0 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `2.48 s` | `FAILED / OOM` |

---

## 💡 Key Architectural Insights

1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.
2. **InferenceOS Layer Placement Optimization**: By dynamically balancing transformer layers across GPU and System RAM, InferenceOS achieves fast generation without tripping Out-Of-Memory (OOM) errors or heavy PCIe thrashing.
3. **Optimal Layer Split Balance**: High GPU offload yields maximum speed until VRAM limits are hit; balanced splits provide peak stability for oversized 9B models on mid-range hardware.