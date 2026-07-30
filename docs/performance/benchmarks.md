# Benchmarking Methodology & Performance Guide

This document specifies the benchmarking methodology, hardware measurement guidelines, and throughput metrics used to evaluate InferenceOS performance.

---

## Performance Metrics

1. **Prompt Ingestion Throughput (`prompt_tps`)**:
   $$\text{Prompt TPS} = \frac{N_{\text{prompt\_tokens}}}{T_{\text{prefill\_seconds}}}$$
   Measures how fast the model ingests input context. Highly dependent on the Dynamic Microbatch Scheduler setting.

2. **Evaluation Generation Throughput (`eval_tps`)**:
   $$\text{Eval TPS} = \frac{N_{\text{gen\_tokens}}}{T_{\text{decode\_seconds}}}$$
   Measures token generation speed during token-by-token decoding.

3. **Time To First Token (`TTFT`)**:
   $$\text{TTFT} = T_{\text{first\_token\_received}} - T_{\text{request\_submitted}}$$
   Measures user-perceived responsiveness.

---

## Running Benchmarks

Execute the automated benchmarking suite via CLI:

```bash
inferenceos benchmark models/llama-3-8b.Q4_K_M.gguf -t 8 -g 28
```

Or run the comparison benchmark scripts:

```bash
python run_benchmark.py
python run_ollama_comparison.py
```
