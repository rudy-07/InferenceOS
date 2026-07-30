# Adaptive Runtime Engine (`inference_runtime/`)

The `inference_runtime` package coordinates system execution, process isolation, streaming output parsing, and runtime statistics aggregation.

---

## Core Classes

### `AdaptiveEngine` (`orchestrator/engine.py`)
Central orchestrator interface:
- **`load_model(model_dir, target_model_name, desired_n_ctx)`**: Inspects GGUF metadata, runs the GGUF selector, computes memory plans, and initializes C++ backend libraries via ctypes bindings.

### `InferenceSession` (`inference_runtime/inference_session.py`)
Session abstraction managing a single inference run:
- Resolves execution backend via `BackendSelector`.
- Constructs CLI arguments via `ArgumentBuilder`.
- Spawns subprocess via `ProcessManager`.
- Collects real-time OS utilization metrics via `StatsCollector`.
- Records post-inference execution telemetry into `RuntimeLearningEngine` and `RuntimeKnowledgeBase`.

### `ProcessManager` (`inference_runtime/process_manager.py`)
Subprocess lifecycle manager providing non-blocking asynchronous I/O:
- Runs concurrent background reader threads for `stdout` (token streaming) and `stderr` (metrics parsing).
- Collects per-token arrival timestamps for $p50$, $p95$, and $p99$ latency calculation.
- Supports synchronous (`wait_sync`) and asynchronous (`stream_async`) execution.

### `StatsCollector` (`inference_runtime/stats_collector.py`)
Real-time statistics collector:
- Monitors CPU utilization, GPU utilization, VRAM usage, and RAM usage.
- Detects pipeline stalls when CPU utilization > 95% while GPU utilization < 10%.
- Parses `stderr` timing output (`prompt eval time`, `eval time`, `t/s`, `TTFT`).

---

## Code Example: Using `InferenceSession`

```python
from pathlib import Path
from inference_runtime import InferenceSession, RuntimeConfig
from layer_placement import PlacementPlan

# Initialize session
session = InferenceSession(
    model_path=Path("models/llama-3-8b.Q4_K_M.gguf"),
    plan=placement_plan,
    config=RuntimeConfig(threads=8, use_flash_attn=True),
    hw_profile=hw_profile,
)

# Stream execution
def on_token(token_text: str):
    print(token_text, end="", flush=True)

result = session.run("Explain quantum computing", on_token=on_token)

print(f"\nPrompt TPS: {result.stats.prompt_eval_tps:.2f} t/s")
print(f"Eval TPS:   {result.stats.eval_tps:.2f} t/s")
print(f"TTFT:       {result.stats.time_to_first_token_ms:.2f} ms")
```
