# Performance Intelligence Engine & Health Monitor (`performance_intelligence/`, `runtime_health/`)

The **Performance Intelligence Engine (PIE)** and **Runtime Health Monitor** continuously assess system health, track baseline metrics, detect throughput regressions, and emit thermal/OOM alert events.

---

## Performance Score & Metrics (`performance_intelligence/engine.py`)

PIE evaluates overall execution performance into a 0 - 100 runtime score based on:
1. **Prompt Ingestion Throughput (`prompt_tps`)**: Tokens/sec during prefill.
2. **Evaluation Generation Throughput (`eval_tps`)**: Tokens/sec during generation.
3. **Time To First Token (`TTFT`)**: Initial response latency in milliseconds.
4. **GPU Occupancy & VRAM Efficiency**: Percentage of peak hardware utilization achieved.

---

## System Health Sensors (`runtime_health/sensors.py`)

The health sensor monitors live system metrics:
- **VRAM Pressure Sensor**: Warns when free VRAM falls below 10%.
- **Thermal Sensor**: Flags thermal throttling on CPU/GPU.
- **OOM Alert Trigger**: Prevents new process spawning when memory pressure is critical.
