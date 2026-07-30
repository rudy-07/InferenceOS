# Inference Execution Lifecycle & Flow

This document details the step-by-step execution lifecycle when an inference request is processed by InferenceOS.

---

## Complete Lifecycle Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as CLI / REST Server
    participant Engine as AdaptiveEngine
    participant Profiler as Hardware Profiler
    participant Placement as Layer Placement Engine
    participant Microbatch as Microbatch Scheduler
    participant KV as KV Cache Manager
    participant Runtime as InferenceSession / ProcessManager
    participant Backend as llama.cpp C++ Binary
    participant Learning as Runtime Learning DB

    Client->>API: Submit Prompt / Request
    API->>Engine: load_model(model_path)
    
    rect rgb(240, 248, 255)
        note right of Engine: Step 1: System Hardware Profiling
        Engine->>Profiler: Run Hardware Profiler
        Profiler-->>Engine: hardware_profile.json
    end

    rect rgb(250, 240, 230)
        note right of Engine: Step 2: Layer Placement & Memory Planning
        Engine->>Placement: Calculate Layer Offload Plan
        Placement-->>Engine: PlacementPlan (GPU layers, CPU layers, PCIe boundaries)
    end

    rect rgb(240, 255, 240)
        note right of Engine: Step 3: Backend & Process Launch
        Engine->>Runtime: Initialize InferenceSession
        Runtime->>Backend: Launch C++ Subprocess (--ngl, -t, -c, --flash-attn)
    end

    rect rgb(255, 240, 245)
        note right of Engine: Step 4: Pre-Inference Scheduling
        Runtime->>Microbatch: Select Optimal Microbatch Size
        Microbatch-->>Runtime: SchedulingDecision (e.g. 1024 tokens)
        Runtime->>KV: Configure KV Quantization & Eviction Policy
    end

    rect rgb(255, 255, 240)
        note right of Engine: Step 5: Streaming Execution & Telemetry
        Runtime->>Backend: Pass Prompt to Standard Input
        loop Token Generation Stream
            Backend-->>Runtime: Output Generated Token
            Runtime-->>Client: Stream SSE / TUI Token Chunk
        end
    end

    rect rgb(235, 245, 255)
        note right of Engine: Step 6: Telemetry Finalization & Learning
        Runtime->>Learning: Log Run Telemetry (TTFT, t/s, VRAM, RAM)
        Learning-->>Engine: Update Local SQLite Database
    end

    Runtime-->>Client: Return Complete InferenceResult
```

---

## Detailed Step Breakdown

### Step 1: System Hardware Profiling
1. The hardware profiler inspects the host system.
2. Interconnect speeds (PCIe bandwidth, unified RAM bus) and GPU VRAM capacities are measured.
3. System ISA capabilities (AVX2, AVX-512, AMX, Metal) are recorded in `hardware_profile.json`.

### Step 2: Layer Placement & Memory Planning
1. `GGUFParser` reads the model header (layer count, hidden dimension, head count, quantization BPW).
2. `PlacementEngine` calculates the memory required per layer.
3. An integer linear programming cost model assigns $N_{gpu}$ layers to GPU VRAM and $N_{cpu}$ layers to host RAM such that available VRAM is never exceeded.

### Step 3: Backend & Process Launch
1. `BackendSelector` matches system capabilities against backend hints (`cuda`, `rocm`, `vulkan`, `metal`, `cpu`).
2. `ArgumentBuilder` constructs the exact CLI flags (`-ngl`, `-t`, `-c`, `-b`, `--flash-attn`).
3. `ProcessManager` spawns the `llama.exe` binary in an isolated subprocess.

### Step 4: Pre-Inference Dynamic Scheduling
1. `MicrobatchScheduler` evaluates candidate prefill batch sizes against remaining VRAM headroom.
2. `IntelligentKVManager` initializes attention-sink eviction windows (H2O or StreamingLLM) if context exceeds cache bounds.

### Step 5: Streaming Execution & Telemetry
1. Prompt tokens are ingested by the execution backend.
2. Standard output is consumed character-by-character by a dedicated background thread.
3. Per-token arrival timestamps are recorded to compute latency percentiles ($p50$, $p95$, $p99$).
4. System utilization (CPU %, GPU %, VRAM MB) is sampled at configurable intervals.

### Step 6: Telemetry Finalization & Learning
1. Subprocess completion status is verified.
2. Detailed run stats (`prompt_eval_tps`, `eval_tps`, `time_to_first_token_ms`) are parsed from `stderr`.
3. Execution details are persisted into `runtime_learning.db` to train predictive recommendation models for subsequent runs.
