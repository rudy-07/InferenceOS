# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix

**Evaluation Date:** 2026-07-26 19:55:35  
**Model:** `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf` (5.48 GB)  
**Test Prompt:** `"Explain what artificial intelligence is in one paragraph."`  
**Parameters:** `n_predict=128`, `n_ctx=2048`, `temp=0.0`, `seed=42`

---

## 📊 Comparative Performance Matrix

| Engine / Layer Configuration | Layer Placement | Generation Speed (tok/s) | Prompt Eval Speed (tok/s) | TTFT (ms) | p50 ITL (ms) | Wall Duration (s) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **InferenceOS (Adaptive Auto-Placement: 28 GPU / 5 CPU)** | `28 GPU / 5 CPU` | **`9.00`** | `33.16` | `13183.8 ms` | `114.73 ms` | `29.04 s` | `SUCCESS` |
| **InferenceOS (18 GPU / 17 CPU: 18 GPU / 17 CPU)** | `18 GPU / 17 CPU` | **`9.04`** | `29.07` | `12255.9 ms` | `114.56 ms` | `28.35 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 18)** | `18 GPU / 15 CPU` | **`4.67`** | `20.30` | `492.6 ms` | `214.18 ms` | `37.70 s` | `SUCCESS` |
| **InferenceOS (12 GPU / 23 CPU: 12 GPU / 23 CPU)** | `12 GPU / 23 CPU` | **`9.02`** | `33.12` | `12121.4 ms` | `113.86 ms` | `27.91 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 12)** | `12 GPU / 21 CPU` | **`3.54`** | `13.06` | `765.7 ms` | `282.70 ms` | `47.32 s` | `SUCCESS` |
| **InferenceOS (6 GPU / 29 CPU: 6 GPU / 29 CPU)** | `6 GPU / 29 CPU` | **`9.16`** | `28.97` | `11928.3 ms` | `113.34 ms` | `27.64 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 6)** | `6 GPU / 27 CPU` | **`2.86`** | `8.13` | `1230.3 ms` | `349.36 ms` | `56.98 s` | `SUCCESS` |
| **Raw llama.cpp (CPU Only (-ngl 0))** | `0 GPU / 33 CPU` | **`2.32`** | `2.54` | `3938.2 ms` | `430.79 ms` | `64.43 s` | `SUCCESS` |
| **Raw llama.cpp (GPU -ngl 99)** | `99 GPU / 0 CPU` | **`0.00`** | `0.00` | `0.0 ms` | `0.00 ms` | `2.53 s` | `FAILED / OOM` |

---

## 💡 Key Architectural Insights

1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.
2. **InferenceOS Layer Placement Optimization**: By dynamically balancing transformer layers across GPU and System RAM, InferenceOS achieves fast generation without tripping Out-Of-Memory (OOM) errors or heavy PCIe thrashing.
3. **Optimal Layer Split Balance**: High GPU offload yields maximum speed until VRAM limits are hit; balanced splits provide peak stability for oversized 9B models on mid-range hardware.