# Changelog

All notable changes to the **InferenceOS** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-30

### Added
- **Hardware Profiler (`profiler/`)**:
  - Multi-vendor automated detection for NVIDIA (`pynvml`), AMD (`amdsmi`), Apple Metal, and Intel Arc.
  - Interconnect bandwidth benchmarking (PCIe Gen3/4/5 and unified bus speeds).
  - Automated `hardware_profile.json` output containing backend hints and quantization recommendations.
- **Adaptive Inference Runtime (`inference_runtime/`, `orchestrator/`)**:
  - Process-managed dynamic C++ runtime execution wrapper.
  - Streaming stdout parser capturing prompt t/s, eval t/s, TTFT, RAM, and VRAM footprints.
  - Dynamic backend selector with automatic fallback (`cuda` -> `vulkan` -> `cpu`).
- **Dynamic Microbatch Scheduler (`scheduler/microbatch_scheduler.py`)**:
  - Multi-heuristic prefill batch sizing (VRAM headroom, prompt TPS, GPU occupancy, PCIe boundaries).
  - Out-of-memory prevention safety margins.
- **KV Cache Manager (`kv_manager/`)**:
  - Attention-sink eviction policies: H2O (Heavy-Hitter Oracle) and StreamingLLM.
  - KV cache quantization options (FP16, Q8_0, Q4_0).
- **Layer Placement & Live Migration (`layer_placement/`, `layer_migration/`, `igpu_support/`)**:
  - Integer linear programming cost model for optimal layer offloading.
  - Live hysteresis-controlled layer migration between CPU, dGPU, and iGPU under memory pressure.
  - Integrated GPU memory contention modeling.
- **Runtime Learning & Knowledge Base (`runtime_learning/`, `runtime_kb/`)**:
  - SQLite persistent execution database capturing per-run performance metrics.
  - Predictive recommendation engine for microbatch and thread allocations.
- **OpenAI & Ollama Compatible Server (`server/`)**:
  - Dual REST API server supporting `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and Ollama `/api/*` endpoints.
  - Server-Sent Events (SSE) streaming output.
- **Rich 21-Command CLI Suite (`cli/`)**:
  - Subcommands: `chat`, `run`, `serve`, `benchmark`, `optimize`, `profile`, `hardware`, `doctor`, `inspect`, `models`, `config`, `plugins`, `cache`, `logs`, `telemetry`, `monitor`, `stats`, `placement`, `update`, `version`, `reset`.
- **Comprehensive Documentation Suite (`docs/`)**:
  - Architecture breakdown, execution flow diagrams, llama.cpp fork attribution, scheduler guide, memory policies, developer tutorials, and benchmarks.
