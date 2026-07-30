"""
run_ollama_comparison.py
-------------------------
Benchmark script to run Ollama and raw llama.cpp under reproducible conditions
identical to InferenceOS, generating side-by-side metric comparison reports (Markdown, JSON, CSV).
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
import platform
import csv
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "Qwen3-4B-Thinking:latest"

PROMPT = "Explain what artificial intelligence is in one paragraph."
N_PREDICT = 128
CONTEXT_LENGTH = 2048
TEMPERATURE = 0.0
SEED = 42

def find_latest_inferenceos_json() -> Path | None:
    """Find the most recent e2e_benchmark_*.json output from InferenceOS pipeline."""
    if not BENCHMARKS_DIR.exists():
        return None
    files = list(BENCHMARKS_DIR.glob("e2e_benchmark_*.json"))
    if not files:
        return None
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0]

def run_raw_llamacpp_benchmark(n_gpu_layers: int = 99) -> dict:
    from run_e2e_integration import find_llama_cli, discover_gguf_models, MODELS_DIR
    cli_path = find_llama_cli()
    models = discover_gguf_models(MODELS_DIR)
    if not cli_path or not models:
        print(f"[Warning] Cannot run raw llama.cpp benchmark: CLI or model missing.")
        return {}

    model_path = models[0][0]
    mode_title = f"Full GPU (-ngl {n_gpu_layers})" if n_gpu_layers > 0 else "CPU Only (-ngl 0)"
    print("============================================================")
    print(f"       Running Raw llama.cpp Benchmark [{mode_title}]        ")
    print("============================================================")
    print(f"Executable:      {cli_path.name}")
    print(f"Model:           {model_path.name}")
    print(f"GPU Offload:     -ngl {n_gpu_layers}")
    print("============================================================\n")

    cmd = [
        str(cli_path),
        "completion",
        "-m", str(model_path),
        "-p", PROMPT,
        "-ngl", str(n_gpu_layers),
        "-c", str(CONTEXT_LENGTH),
        "-n", str(N_PREDICT),
        "-t", "6",
        "--temp", str(TEMPERATURE),
        "-no-cnv",
    ]

    start_time = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout, stderr = proc.communicate()
    end_time = time.perf_counter()
    total_wall_sec = end_time - start_time

    prompt_tps = None
    gen_tps = None
    prompt_eval_ms = None
    gen_eval_ms = None
    gen_tokens = N_PREDICT
    prompt_tokens = 0

    for line in stderr.splitlines():
        if "prompt eval time" in line:
            try:
                sub_parts = line.split("prompt eval time =")[1].split("/")
                prompt_eval_ms = float(sub_parts[0].replace("ms", "").strip())
                prompt_tokens = int(sub_parts[1].split("tokens")[0].strip())
                prompt_tps = float(line.split("tokens per second")[0].split(",")[-1].strip())
            except Exception:
                pass
        elif "eval time =" in line and "prompt" not in line:
            try:
                sub_parts = line.split("eval time =")[1].split("/")
                gen_eval_ms = float(sub_parts[0].replace("ms", "").strip())
                gen_tokens = int(sub_parts[1].split("runs")[0].split("tokens")[0].strip())
                gen_tps = float(line.split("tokens per second")[0].split(",")[-1].strip())
            except Exception:
                pass

    return {
        "engine": f"Raw llama.cpp ({'GPU -ngl 99' if n_gpu_layers > 0 else 'CPU -ngl 0'})",
        "n_gpu_layers": n_gpu_layers,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prompt_eval_ms": prompt_eval_ms or 0.0,
        "prompt_tps": round(prompt_tps, 2) if prompt_tps else 0.0,
        "gen_eval_ms": gen_eval_ms or 0.0,
        "generation_tps": round(gen_tps, 2) if gen_tps else 0.0,
        "ttft_ms": round(prompt_eval_ms, 1) if prompt_eval_ms else 0.0,
        "p50_itl_ms": round(gen_eval_ms / max(1, gen_tokens), 2) if gen_eval_ms else 0.0,
        "total_wall_sec": round(total_wall_sec, 2),
        "response_text": stdout.strip()
    }

def run_ollama_benchmark() -> dict:
    print("============================================================")
    print("           Running Ollama Reproducible Benchmark            ")
    print("============================================================")
    print(f"Model:           {MODEL_NAME}")
    print(f"Endpoint:        {OLLAMA_URL}")
    print(f"Prompt:          \"{PROMPT}\"")
    print(f"Max Predict:     {N_PREDICT}")
    print(f"Context Length:  {CONTEXT_LENGTH}")
    print(f"Temperature:     {TEMPERATURE}")
    print(f"Seed:            {SEED}")
    print("============================================================\n")

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "stream": True,
        "options": {
            "num_predict": N_PREDICT,
            "num_ctx": CONTEXT_LENGTH,
            "temperature": TEMPERATURE,
            "seed": SEED,
        }
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    token_timestamps = []
    token_strings = []
    first_token_time = None
    prompt_tokens = 0
    gen_tokens = 0
    prompt_eval_duration_ns = 0
    eval_duration_ns = 0
    total_duration_ns = 0
    load_duration_ns = 0

    print("Executing Ollama inference stream...")
    start_time = time.perf_counter()

    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if not line.strip():
                    continue
                now = time.perf_counter()
                chunk = json.loads(line.decode("utf-8"))
                
                response_text = chunk.get("response", "")
                if response_text:
                    if first_token_time is None:
                        first_token_time = now
                    token_timestamps.append(now)
                    token_strings.append(response_text)
                    gen_tokens += 1
                    sys.stdout.write(response_text)
                    sys.stdout.flush()

                if chunk.get("done", False):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)
                    total_duration_ns = chunk.get("total_duration", 0)
                    load_duration_ns = chunk.get("load_duration", 0)
                    if not gen_tokens:
                        gen_tokens = chunk.get("eval_count", 0)

        end_time = time.perf_counter()
        total_wall_sec = end_time - start_time
        print("\n")

    except Exception as e:
        print(f"\n[Warning] Ollama inference failed: {e}")
        return {
            "engine": "Ollama",
            "generation_tps": 0.0,
            "prompt_tps": 0.0,
            "ttft_ms": 0.0,
            "p50_itl_ms": 0.0,
            "total_wall_sec": 0.0,
            "gen_tokens": 0,
            "prompt_tokens": 0,
            "error": str(e)
        }

    ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    
    itls = []
    if len(token_timestamps) > 1:
        for i in range(1, len(token_timestamps)):
            itls.append((token_timestamps[i] - token_timestamps[i-1]) * 1000.0)

    p50_itl_ms = float(np.median(itls)) if itls else 0.0
    p95_itl_ms = float(np.percentile(itls, 95)) if itls else 0.0

    prompt_tps = (prompt_tokens / (prompt_eval_duration_ns / 1e9)) if prompt_eval_duration_ns > 0 else 0.0
    gen_tps = (gen_tokens / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else (gen_tokens / total_wall_sec)

    return {
        "engine": "Ollama",
        "version": "0.32.3",
        "model_name": MODEL_NAME,
        "prompt": PROMPT,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prompt_eval_ms": prompt_eval_duration_ns / 1e6,
        "prompt_tps": round(prompt_tps, 2),
        "gen_eval_ms": eval_duration_ns / 1e6,
        "generation_tps": round(gen_tps, 2),
        "ttft_ms": round(ttft_ms, 2),
        "p50_itl_ms": round(p50_itl_ms, 2),
        "p95_itl_ms": round(p95_itl_ms, 2),
        "total_wall_sec": round(total_wall_sec, 2),
        "load_duration_ms": load_duration_ns / 1e6,
        "response_text": "".join(token_strings)
    }

def compare_and_export(ollama_metrics: dict, raw_gpu_metrics: dict, raw_cpu_metrics: dict, inferenceos_json_path: Path):
    with open(inferenceos_json_path, "r", encoding="utf-8") as f:
        ios_data = json.load(f)

    ios_telemetry = ios_data.get("telemetry", {})
    ios_throughput = ios_telemetry.get("throughput", {})
    ios_latency = ios_telemetry.get("latency", {})
    ios_util = ios_telemetry.get("utilization", {})
    ios_placement = ios_data.get("placement_plan", {})
    ios_validation = ios_data.get("validation_accuracy", {})
    ios_model = ios_data.get("model", {})

    ios_metrics = {
        "engine": "InferenceOS (Adaptive)",
        "version": "1.0-adaptive",
        "model_name": ios_model.get("name", "Qwen3-4B-Thinking-2507.Q5_K_M.gguf"),
        "prompt_tokens": ios_throughput.get("prompt_tokens", 0),
        "gen_tokens": ios_throughput.get("generation_tokens", 0),
        "prompt_eval_ms": ios_throughput.get("prompt_eval_ms", 0.0),
        "prompt_tps": ios_throughput.get("prompt_tps", 0.0),
        "gen_eval_ms": ios_throughput.get("generation_eval_ms", 0.0),
        "generation_tps": ios_throughput.get("generation_tps", 0.0),
        "ttft_ms": ios_latency.get("ttft_ms", 0.0),
        "p50_itl_ms": ios_latency.get("p50_itl_ms", 0.0),
        "p95_itl_ms": ios_latency.get("p95_itl_ms", 0.0),
        "total_wall_sec": ios_throughput.get("total_wall_ms", 0.0) / 1000.0,
        "gpu_offload_ratio": ios_placement.get("gpu_offload_ratio", 0.0),
        "n_gpu_layers": ios_placement.get("n_gpu_layers", 0),
        "n_cpu_layers": ios_placement.get("n_cpu_layers", 0),
        "actual_vram_mb": ios_validation.get("actual_vram_mb", 0.0),
        "actual_ram_mb": ios_validation.get("actual_ram_mb", 0.0),
        "avg_gpu_util_pct": ios_util.get("avg_gpu_pct", 0.0),
        "avg_cpu_util_pct": ios_util.get("avg_cpu_pct", 0.0),
    }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = BENCHMARKS_DIR / f"multi_engine_comparison_{timestamp_str}.md"
    json_path = BENCHMARKS_DIR / f"multi_engine_comparison_{timestamp_str}.json"
    csv_path = BENCHMARKS_DIR / f"multi_engine_comparison_{timestamp_str}.csv"

    md_content = f"""# 🏎️ Inference Engine Benchmark Matrix: InferenceOS vs. Ollama vs. Raw llama.cpp

**Evaluation Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model:** `Qwen3-4B-Thinking-2507.Q5_K_M.gguf` / `Qwen3-4B-Thinking:latest`  
**Test Prompt:** `"{PROMPT}"`  
**Parameters:** `n_predict={N_PREDICT}`, `n_ctx={CONTEXT_LENGTH}`, `temp={TEMPERATURE}`, `seed={SEED}`

---

## 📊 Summary Performance Comparison Matrix

| Metric | InferenceOS (Adaptive) | Raw llama.cpp (Full GPU -ngl 99) | Raw llama.cpp (CPU Only -ngl 0) | Ollama (v0.32.3) |
| :--- | :---: | :---: | :---: | :---: |
| **Generation Speed (tok/s)** | **`{ios_metrics['generation_tps']:.2f}`** | **`{raw_gpu_metrics.get('generation_tps', 0.0):.2f}`** | `{raw_cpu_metrics.get('generation_tps', 0.0):.2f}` | `{ollama_metrics['generation_tps']:.2f}` |
| **Prompt Eval Speed (tok/s)** | `{ios_metrics['prompt_tps']:.2f}` | **`{raw_gpu_metrics.get('prompt_tps', 0.0):.2f}`** | `{raw_cpu_metrics.get('prompt_tps', 0.0):.2f}` | `{ollama_metrics['prompt_tps']:.2f}` |
| **Time to First Token (TTFT)** | `{ios_metrics['ttft_ms']:.1f} ms` | **`{raw_gpu_metrics.get('ttft_ms', 0.0):.1f} ms`** | `{raw_cpu_metrics.get('ttft_ms', 0.0):.1f} ms` | `{ollama_metrics['ttft_ms']:.1f} ms` |
| **Median Inter-Token Latency (ITL)** | `{ios_metrics['p50_itl_ms']:.2f} ms` | **`{raw_gpu_metrics.get('p50_itl_ms', 0.0):.2f} ms`** | `{raw_cpu_metrics.get('p50_itl_ms', 0.0):.2f} ms` | `{ollama_metrics['p50_itl_ms']:.2f} ms` |
| **Total Wall Clock Duration** | `{ios_metrics['total_wall_sec']:.2f} s` | **`{raw_gpu_metrics.get('total_wall_sec', 0.0):.2f} s`** | `{raw_cpu_metrics.get('total_wall_sec', 0.0):.2f} s` | `{ollama_metrics['total_wall_sec']:.2f} s` |
| **Generated Tokens** | `{ios_metrics['gen_tokens']}` | `{raw_gpu_metrics.get('gen_tokens', 0)}` | `{raw_cpu_metrics.get('gen_tokens', 0)}` | `{ollama_metrics['gen_tokens']}` |
| **GPU Offload Allocation** | `{ios_metrics['n_gpu_layers']}/36 Blocks ({ios_metrics['n_gpu_layers']+2}/38 Total)` | `38/38 Total Layers` | `0/38 Total Layers` | `Full Auto Offload` |

---

## 💡 Subsystem Analysis & Key Takeaways

1. **Raw llama.cpp GPU (-ngl 99)**: Demonstrates the peak hardware upper bound when all 38 transformer layers are placed on GPU VRAM.
2. **InferenceOS Adaptive Pipeline**: Dynamically allocates layers based on system RAM/VRAM pressure. When VRAM is constrained, it splits layers safely across GPU & CPU to prevent Out-Of-Memory (OOM) crashes.
3. **Pipeline Configurator**: Using `pipeline_config.json` or `--gpu-layers 99`, InferenceOS can be forced into full GPU offload mode to achieve raw `llama.cpp` speeds while retaining telemetry, preflight guards, and scheduling benefits.
"""

    md_path.write_text(md_content, encoding="utf-8")

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": PROMPT,
        "settings": {
            "n_predict": N_PREDICT,
            "n_ctx": CONTEXT_LENGTH,
            "temperature": TEMPERATURE,
            "seed": SEED,
        },
        "engines": {
            "inferenceos_adaptive": ios_metrics,
            "raw_llamacpp_gpu": raw_gpu_metrics,
            "raw_llamacpp_cpu": raw_cpu_metrics,
            "ollama": ollama_metrics,
        }
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Engine", "Gen_TPS", "Prompt_TPS", "TTFT_ms", "p50_ITL_ms",
            "Total_Wall_Sec", "Gen_Tokens", "Prompt_Tokens"
        ])
        for eng in [ios_metrics, raw_gpu_metrics, raw_cpu_metrics, ollama_metrics]:
            writer.writerow([
                eng.get("engine", "Unknown"), eng.get("generation_tps", 0.0), eng.get("prompt_tps", 0.0),
                eng.get("ttft_ms", 0.0), eng.get("p50_itl_ms", 0.0), eng.get("total_wall_sec", 0.0),
                eng.get("gen_tokens", 0), eng.get("prompt_tokens", 0)
            ])

    print("\n============================================================")
    print("      Multi-Engine Comparative Reports Generated            ")
    print("============================================================")
    print(f"  Markdown Report: {md_path}")
    print(f"  JSON Artifact:   {json_path}")
    print(f"  CSV Summary:     {csv_path}")
    print("============================================================\n")

    try:
        print(md_content)
    except UnicodeEncodeError:
        print(md_content.encode("utf-8", errors="replace").decode("utf-8"))

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    latest_ios_json = find_latest_inferenceos_json()
    if not latest_ios_json:
        print("❌ Error: No InferenceOS e2e_benchmark_*.json report found in benchmarks/. Run run_e2e_integration.py first.")
        sys.exit(1)

    print(f"Loaded latest InferenceOS benchmark metrics from: {latest_ios_json.name}")
    
    # Run Ollama and raw llama.cpp benchmarks
    ollama_metrics = run_ollama_benchmark()
    raw_gpu_metrics = run_raw_llamacpp_benchmark(n_gpu_layers=99)
    raw_cpu_metrics = run_raw_llamacpp_benchmark(n_gpu_layers=0)

    compare_and_export(ollama_metrics, raw_gpu_metrics, raw_cpu_metrics, latest_ios_json)

if __name__ == "__main__":
    main()
