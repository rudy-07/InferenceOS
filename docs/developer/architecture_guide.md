# Developer Architecture & Codebase Layout

This guide provides developers and contributors with a map of the InferenceOS repository.

---

## Directory Responsibilities

```
InferenceOS/
├── cli/                        # 21 Subcommand entry points and Rich TUI displays
│   ├── commands/               # Individual command handlers (run, serve, chat, etc.)
│   ├── main.py                 # Main argument parser and command dispatcher
│   └── tui/                    # Terminal user interface screens and monitors
├── docs/                       # Complete Markdown documentation suite
├── examples/                   # Executable python example scripts
├── igpu_support/               # Integrated GPU contention models and profiling
├── inference_runtime/          # Adaptive Runtime Engine, ProcessManager, StatsCollector
├── kv_manager/                 # KV cache eviction policies, compression, quantization
├── layer_migration/            # Live hysteresis-controlled layer migration engine
├── layer_placement/            # Integer linear programming layer placement cost model
├── memory_budget/              # Pre-flight memory budget planner and allocator
├── optimizer/                  # Automatic Performance Optimizer (APO) and search matrix
├── orchestrator/               # AdaptiveEngine, GGUF binary metadata reader
├── performance_intelligence/   # PIE engine, regression detection, baseline scoring
├── profiler/                   # Hardware profiler backends (NVIDIA, AMD, Apple, Intel)
├── runtime_health/             # Health monitor, OOM alert sensors, thermal checks
├── runtime_kb/                 # Runtime Knowledge Base (RKB) and fingerprint graph
├── runtime_learning/           # SQLite telemetry store and predictive ML engine
├── scheduler/                  # Dynamic Microbatch, Context, and Memory Schedulers
├── server/                     # FastAPI OpenAI and Ollama compatible REST API server
└── tests/                      # Pytest unit and integration test suite
```
