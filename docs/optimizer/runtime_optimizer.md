# Automatic Performance Optimizer (APO) (`optimizer/`)

The **Automatic Performance Optimizer (APO)** executes a 5-step search matrix to find, benchmark, validate, and cache the fastest layer offloading and thread configuration for any model on your hardware.

---

## 5-Step Optimization Pipeline

```mermaid
graph TD
    Step1["1. Hardware & GGUF Fingerprinting"] --> Step2["2. Layer Offload Plan Candidates (ILP)"]
    Step2 --> Step3["3. Matrix Benchmarking (Microbatch x Threads)"]
    Step3 --> Step4["4. Confidence Scoring & Validation"]
    Step4 --> Step5["5. Profile Persistence (JSON Storage)"]
```

---

## CLI Integration

Run automated optimization for any model file via CLI:

```bash
inferenceos optimize models/llama-3-8b.Q4_K_M.gguf
```

Cached profiles are stored under `.inferenceos/profiles/` and re-used automatically on future runs.
