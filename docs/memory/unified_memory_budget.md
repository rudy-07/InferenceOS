# Unified Memory Budget Planner (`memory_budget/`)

The `memory_budget` module computes pre-flight memory budgets, partitioning system VRAM and RAM into dedicated pools for model weights, KV caches, activation buffers, and operating system safety headroom.

---

## Memory Allocation Partitioning

```
+-------------------------------------------------------------------+
|                        Total System Memory                        |
+-----------------------------------+-------------------------------+
|             GPU VRAM              |           Host RAM            |
+-----------------+-----------------+---------------+---------------+
| Model Weights   | KV Cache        | Model Weights | OS & App      |
| (GPU Layers)    | (Q8_0 / FP16)   | (CPU Layers)  | Buffer        |
+-----------------+-----------------+---------------+---------------+
```

---

## Key Classes

- **`MemoryBudgetManager` (`manager.py`)**: Computes component-level memory allocations based on system health and memory pressure levels.
- **`MemoryPlanner` (`planner.py`)**: Evaluates model weight sizes vs VRAM boundaries to calculate optimal layer offload ratios.
- **`PreflightGuard` (`allocator.py`)**: Validates allocation plans prior to C++ engine launch to guarantee zero OOM failures.
