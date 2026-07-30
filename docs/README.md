# 📚 InferenceOS Documentation Hub

Welcome to the official technical documentation for **InferenceOS** — the hardware-agnostic, adaptive operating system for local Large Language Model (LLM) inference.

---

## 🧭 Navigation Index

### 1. Architecture & Design Philosophy
- **[Architecture Overview](architecture/overview.md)**: Control plane, system layers, and design philosophy.
- **[Execution Lifecycle & Flow](architecture/execution_flow.md)**: End-to-end trace from CLI launch to token generation.
- **[llama.cpp Fork & Synergy](architecture/llama_cpp_fork.md)**: Why `llama.cpp` was forked, what InferenceOS extends, and upstream alignment.

### 2. Runtime & Backend Management
- **[Adaptive Runtime Engine](runtime/adaptive_engine.md)**: `AdaptiveEngine`, process isolation, and streaming output parsers.
- **[Multi-Vendor Backend Selector](runtime/backend_selector.md)**: CUDA, ROCm, Vulkan, Metal, and CPU backend resolution.
- **[Async Pipeline & Prefetching](runtime/async_pipeline.md)**: CUDA streams, layer prefetching, and multi-worker dispatching.

### 3. Schedulers
- **[Dynamic Microbatch Scheduler](scheduler/microbatch_scheduler.md)**: Hardware-aware prefill sizing and safety scoring matrix.
- **[Context Window Scheduler](scheduler/context_scheduler.md)**: Memory-bounded dynamic context window scaling.
- **[Tiered Memory Scheduler](scheduler/memory_scheduler.md)**: Layer offloading policies across VRAM, RAM, and Swap.

### 4. Memory & Cache Subsystems
- **[Unified Memory Budget Planner](memory/unified_memory_budget.md)**: Unified memory bounds and safety allocations.
- **[Intelligent KV Cache Manager](memory/kv_cache_manager.md)**: H2O attention sinks, StreamingLLM, and FP16/Q8_0/Q4_0 quantization.
- **[Live Hysteresis Layer Migration](memory/layer_migration.md)**: Real-time layer movement under dynamic memory pressure & iGPU contention.

### 5. Runtime Learning & Intelligence
- **[Runtime Learning Engine](learning/runtime_learning.md)**: SQLite execution database, ML prediction models, and telemetry feedback.
- **[Runtime Knowledge Base](learning/runtime_kb.md)**: Hardware-model graph, fingerprint matching, and historical lookups.

### 6. Optimizer & Performance Intelligence
- **[Automatic Performance Optimizer](optimizer/runtime_optimizer.md)**: 5-step automated placement search, fingerprinting, and profile caching.
- **[Performance Intelligence & Health](optimizer/performance_intelligence.md)**: Regression detection, baseline metrics, OOM sensors, and thermal throttling alerts.

### 7. API Server
- **[OpenAI REST API Reference](server/openai_api.md)**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/models`.
- **[Ollama REST API Reference](server/ollama_api.md)**: `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, and `/api/ps`.

### 8. Interfaces & Configuration
- **[CLI Command Reference](cli/command_reference.md)**: Complete guide to all 21 subcommands with usage examples.
- **[Configuration Settings Reference](configuration/settings_reference.md)**: Environment variables, `pipeline_config.json`, and default settings.

### 9. Developer & Extension Guides
- **[Codebase Architecture Guide](developer/architecture_guide.md)**: Subsystem directory breakdown and internal dependency graph.
- **[Tutorial: Adding a Custom Scheduler](developer/adding_scheduler.md)**: Step-by-step guide to writing a custom scheduler.
- **[Tutorial: Adding a Custom Backend](developer/adding_backend.md)**: Step-by-step guide to integrating a new hardware backend.
- **[Release Workflow & CI Guide](developer/release_process.md)**: Build, test, and release process documentation.

### 10. Performance & FAQ
- **[Benchmarking Guide](performance/benchmarks.md)**: Benchmark methodology, reproducibility guidelines, and throughput metrics.
- **[Troubleshooting FAQ](faq/faq.md)**: Common errors, driver checks, and runtime solutions.
