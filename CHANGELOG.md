# Changelog

All notable changes to the **InferenceOS** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-01

### Added
- **System 1 Non-Autoregressive Reflex Runtime Engine (`inference_runtime/system1_engine.py`)**:
  - Sub-150ms non-autoregressive decision & reflex inference (10x faster than autoregressive decoding).
  - Native runtime integration for Laya mmBERT architectures (`laya/agent`) and Kev pointer-head classification heads.
  - Zero KV cache footprint (0 bytes KV cache allocation during System 1 classification).
  - Hardware acceleration placement across DirectML, CUDA, and pinned CPU memory.
  - Multi-type output parsing: compound multiple choice, numerical scoring, binary validation, and non-empty fallback (`noul`).
- **System 1 + System 2 Kahneman Hybrid Symbiosis Cognitive Loop (`orchestrator/hybrid_orchestrator.py`)**:
  - Dynamic bidirectional fast/slow cognitive loop implementing Kahneman's dual-system cognitive architecture.
  - Confidence & uncertainty gating against configurable threshold $\tau$ (`--tau`, default 0.85).
  - System 2 Teacher Escalation: direct in-memory GGUF session reuse under memory pressure, eliminating dual GPU context contention.
  - DAgger (Dataset Aggregation) continuous learning export (`/dagger`) producing structured JSONL traces for offline student distillation.
  - Server REST API routes: `/v1/system1/decide`, `/v1/system1/models`, `/v1/hybrid/deliberate`.
- **Next-Gen Modern TUI Revamp (`cli/tui/`)**:
  - **Dynamic Slash Command Autocompleter (`cli/tui/completer.py`)**: Interactive floating popover listing all commands (`/model`, `/mode`, `/tau`, `/stats`, `/dagger`, `/clear`, `/theme`, `/help`, `/exit`), subcommands, and dynamic argument completions.
  - **Interactive Modal Model & Mode Pickers (`cli/tui/interactive_picker.py`)**: Split-view terminal modal with live fuzzy search, keyboard navigation (Up/Down/Enter/Esc), metadata inspector panel, and instant hotkey selection (`Ctrl+O` / `Ctrl+T`).
  - **Docked Bottom Status Ribbon (`cli/tui/bottom_toolbar.py`)**: Persistent terminal status bar showing active System 1/2 models, cognitive mode (`system2`, `system1_reflex`, `symbiosis`), latency, confidence, $\tau$ threshold, and GPU/RAM memory stats.
  - **Deliberation Flowcards & Visual Markdown Panels (`cli/tui/chat_renderer.py`)**: Distinct card styling for User queries, System 1 Reflexes (cyan badges, confidence bars, timing), and System 2 Reasoning Chains (purple accents, eval t/s, TTFT).
  - **Crash-Proof Terminal Engine**: UTF-16LE BOM stripping (`\ufeff`), buffer safety on `Ctrl+C` (clears input buffer without dropping session), and collision-free mode toggling on `Ctrl+T`.
- **New CLI Command (`cli/commands/decide.py`)**:
  - `inferenceos decide`: Execute ultra-fast non-autoregressive decision inference on System 1 models with confidence reporting.

### Changed
- **CLI Commands (`cli/commands/chat.py`, `run.py`, `models_cmd.py`)**: Added `--mode`, `--system1-model`, `--system2-model`, and `--tau` parameters.
- **Model Registry (`cli/core/model_registry.py`)**: Universal discovery for HuggingFace directories, SafeTensors, ONNX, and PyTorch models alongside GGUF binaries.
- **Production Server (`server/api/routes/systemone.py`)**: Mounted System 1 and Hybrid Symbiosis endpoints in FastAPI server.

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
