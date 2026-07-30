# Runtime Learning Engine (`runtime_learning/`)

The **Runtime Learning Engine** makes InferenceOS self-improving. It records execution telemetry into a local SQLite database (`runtime_learning.db`) and fits machine learning regression models to predict optimal microbatch sizes, thread counts, and layer placements for future inference runs.

---

## SQLite Database Schema (`database.py`)

The feedback database stores historical records across runs:

```sql
CREATE TABLE IF NOT EXISTS execution_history (
    id TEXT PRIMARY KEY,
    timestamp REAL,
    model_name TEXT,
    num_layers INTEGER,
    hidden_size INTEGER,
    backend TEXT,
    gpu_name TEXT,
    n_gpu_layers INTEGER,
    n_cpu_layers INTEGER,
    microbatch_size INTEGER,
    context_length INTEGER,
    prompt_tps REAL,
    eval_tps REAL,
    ttft_ms REAL,
    gpu_vram_used_mb REAL,
    success INTEGER,
    warnings TEXT
);
```

---

## Predictive Recommendation Engine (`prediction.py`, `recommendation.py`)

1. **Feature Vector Extraction**: Extract hardware fingerprint (GPU name, VRAM, RAM, PCIe BW) + model parameters ($N_{\text{layers}}$, $D_{\text{hidden}}$, quantization) + requested context length.
2. **Regression Scoring**: Evaluates past successful runs to fit a predictive score:
   $$\text{Score} = w_1 \cdot \text{eval\_tps} + w_2 \cdot \text{prompt\_tps} - w_3 \cdot \text{ttft\_ms} - w_4 \cdot \text{OOM\_risk}$$
3. **Recommendation**: `recommend_microbatch()` returns the predicted highest-throughput batch size $B_{opt}$, which is passed directly into `MicrobatchScheduler`.
