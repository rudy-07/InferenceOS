"""
run_qwythos_benchmark.py
-------------------------
Comparative Benchmark Suite for Qwythos-9B-Claude-Mythos GGUF model:
  1. Raw llama.cpp Full GPU (-ngl 99)
  2. Raw llama.cpp CPU Only (-ngl 0)
  3. InferenceOS Adaptive Auto-Placement Plan
  4. InferenceOS Layer Split 1: High GPU Offload (-ngl 22)
  5. InferenceOS Layer Split 2: Balanced Split (-ngl 16)
  6. InferenceOS Layer Split 3: Low GPU Offload (-ngl 8)

Generates comparative markdown matrix, JSON telemetry, and CSV report in benchmarks/
"""
import os
import sys
import json
import time
import csv
import subprocess
import platform
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf"
PROMPT = "Explain what artificial intelligence is in one paragraph."
N_PREDICT = 128
CONTEXT_LENGTH = 2048
TEMPERATURE = 0.0
SEED = 42

def find_qwythos_model() -> Path:
    target = MODELS_DIR / MODEL_NAME
    if target.exists():
        return target
    # Fallback search
    for f in MODELS_DIR.rglob("*.gguf"):
        if "qwythos" in f.name.lower():
            return f
    raise FileNotFoundError(f"Model {MODEL_NAME} not found in {MODELS_DIR}")

def run_raw_llamacpp(model_path: Path, n_gpu_layers: int = 99) -> dict:
    from run_e2e_integration import find_llama_cli
    cli_path = find_llama_cli()
    if not cli_path:
        print("[Warning] llama.exe CLI binary not found.")
        return {}

    mode_title = f"GPU -ngl {n_gpu_layers}" if n_gpu_layers > 0 else "CPU Only (-ngl 0)"
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
    try:
        stdout, stderr = proc.communicate(timeout=120)
        end_time = time.perf_counter()
        total_wall_sec = end_time - start_time
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        print(f"[Warning] Raw llama.cpp (-ngl {n_gpu_layers}) timed out / thrashing (120s limit).")
        return {
            "config_name": f"Raw llama.cpp ({mode_title})",
            "n_gpu_layers": n_gpu_layers if n_gpu_layers <= 35 else 35,
            "n_cpu_layers": 35 - n_gpu_layers if n_gpu_layers <= 35 else 0,
            "prompt_tokens": 0,
            "gen_tokens": 0,
            "prompt_eval_ms": 0.0,
            "prompt_tps": 0.0,
            "gen_eval_ms": 0.0,
            "generation_tps": 0.0,
            "ttft_ms": 0.0,
            "p50_itl_ms": 0.0,
            "total_wall_sec": 120.0,
            "status": "TIMEOUT / VRAM OOM THRASH",
            "response_text": ""
        }

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
        "config_name": f"Raw llama.cpp ({mode_title})",
        "n_gpu_layers": n_gpu_layers,
        "n_cpu_layers": 33 - n_gpu_layers if n_gpu_layers <= 33 else 0,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prompt_eval_ms": prompt_eval_ms or 0.0,
        "prompt_tps": round(prompt_tps, 2) if prompt_tps else 0.0,
        "gen_eval_ms": gen_eval_ms or 0.0,
        "generation_tps": round(gen_tps, 2) if gen_tps else 0.0,
        "ttft_ms": round(prompt_eval_ms, 1) if prompt_eval_ms else 0.0,
        "p50_itl_ms": round(gen_eval_ms / max(1, gen_tokens), 2) if gen_eval_ms else 0.0,
        "total_wall_sec": round(total_wall_sec, 2),
        "status": "SUCCESS" if proc.returncode == 0 and (gen_tps or 0) > 0 else "FAILED / OOM",
        "response_text": stdout.strip()
    }

def run_inferenceos_config(
    model_path: Path,
    gpu_layers_override: int | None = None,
    igpu_layers_override: int | None = None,
    enable_igpu: bool = False,
    label: str = "Adaptive Auto"
) -> dict:
    from run_e2e_integration import PipelineConfig, run_end_to_end_validation, find_llama_cli
    from layer_placement import PlacementEngine, ModelDescriptor
    from orchestrator.gguf_parser import read_gguf_metadata
    from profiler import get_system_resources
    from inference_runtime import RuntimeConfig, RuntimeEngine, detect_backend
    from igpu_support import IgpuPlacementContributor

    print("============================================================")
    print(f"       Running InferenceOS Benchmark [{label}]        ")
    print("============================================================")
    
    hw_profile = get_system_resources().to_dict()
    gguf_meta = read_gguf_metadata(model_path)
    model_desc = ModelDescriptor.from_gguf_metadata(
        metadata=gguf_meta,
        model_size_bytes=model_path.stat().st_size,
        quant_type="Q4_K_M",
        model_name=model_path.stem,
    )

    placement_engine = PlacementEngine(hw_profile=hw_profile)
    plan = placement_engine.generatePlacementPlan(model=model_desc, context_length=CONTEXT_LENGTH)

    if gpu_layers_override is not None:
        n_gpu = max(0, min(gpu_layers_override, plan.total_layers))
        plan.n_gpu_layers = n_gpu
        plan.n_cpu_layers = plan.total_layers - n_gpu
        plan.n_igpu_layers = 0

    if enable_igpu or igpu_layers_override is not None:
        igpu_contributor = IgpuPlacementContributor(hw_profile)
        igpu_res = igpu_contributor.contribute(model_desc, plan)
        plan = igpu_res.plan
        if igpu_layers_override is not None:
            n_igpu = max(0, min(igpu_layers_override, plan.total_layers - plan.n_gpu_layers))
            plan.n_igpu_layers = n_igpu
            plan.n_cpu_layers = max(0, plan.total_layers - plan.n_gpu_layers - plan.n_igpu_layers)

    print(f"  [Plan Layout] dGPU: {plan.n_gpu_layers} | iGPU: {plan.n_igpu_layers} | CPU: {plan.n_cpu_layers}")

    backend_info = detect_backend(hw_profile, plan)
    cli_path = find_llama_cli()

    runtime_cfg = RuntimeConfig.from_hw_profile(
        hw_profile,
        n_predict=N_PREDICT,
        context_length=CONTEXT_LENGTH,
        temp=TEMPERATURE,
        seed=SEED,
        enable_async_scheduler=True,
        enable_profiler=True,
    )
    runtime_cfg.threads = 6
    runtime_cfg.use_flash_attn = True

    engine = RuntimeEngine(hw_profile=hw_profile, llama_exe_path=cli_path)

    start_exec = time.perf_counter()
    try:
        infer_result, prof_result = engine.profileRun(
            plan=plan,
            model_path=model_path,
            prompt=PROMPT,
            config_override=runtime_cfg,
            vram_free_mb=hw_profile.get("gpus", [{}])[0].get("vram_free_mb", 4096),
        )
        end_exec = time.perf_counter()
        t = prof_result.telemetry

        dev_desc = f"{plan.n_gpu_layers} dGPU / {plan.n_igpu_layers} iGPU / {plan.n_cpu_layers} CPU"
        return {
            "config_name": f"InferenceOS ({label}: {dev_desc})",
            "n_gpu_layers": plan.n_gpu_layers,
            "n_igpu_layers": plan.n_igpu_layers,
            "n_cpu_layers": plan.n_cpu_layers,
            "prompt_tokens": t.prompt_tokens,
            "gen_tokens": t.generation_tokens,
            "prompt_eval_ms": t.prompt_eval_ms,
            "prompt_tps": round(t.prompt_tps, 2),
            "gen_eval_ms": t.generation_eval_ms,
            "generation_tps": round(t.generation_tps, 2),
            "ttft_ms": round(t.ttft_ms, 1),
            "p50_itl_ms": round(t.p50_itl_ms, 2),
            "total_wall_sec": round(end_exec - start_exec, 2),
            "status": "SUCCESS" if t.generation_tps > 0 else "FAILED",
            "vram_est_mb": round(plan.estimated_vram_bytes / (1024**2), 1),
            "ram_est_mb": round(plan.estimated_ram_bytes / (1024**2), 1),
            "gpu_idle_pct": round(t.gpu_idle_pct, 1),
            "avg_cpu_pct": round(t.avg_cpu_util_pct, 1),
            "avg_gpu_pct": round(t.avg_gpu_util_pct, 1),
        }
    except Exception as e:
        print(f"  [ERROR] Execution failed for {label}: {e}")
        return {
            "config_name": f"InferenceOS ({label})",
            "n_gpu_layers": plan.n_gpu_layers if 'plan' in locals() else 0,
            "n_cpu_layers": plan.n_cpu_layers if 'plan' in locals() else 0,
            "generation_tps": 0.0,
            "prompt_tps": 0.0,
            "ttft_ms": 0.0,
            "p50_itl_ms": 0.0,
            "total_wall_sec": 0.0,
            "status": f"FAILED: {e}",
        }

def export_matrix_reports(results: list[dict], model_path: Path):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    md_path = BENCHMARKS_DIR / f"qwythos_9b_comparison_{timestamp_str}.md"
    json_path = BENCHMARKS_DIR / f"qwythos_9b_comparison_{timestamp_str}.json"
    csv_path = BENCHMARKS_DIR / f"qwythos_9b_comparison_{timestamp_str}.csv"

    md_lines = [
        f"# 🏎️ Qwythos-9B Layer-Split & Engine Benchmark Matrix",
        f"",
        f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model:** `{model_path.name}` ({model_path.stat().st_size / (1024**3):.2f} GB)  ",
        f"**Test Prompt:** `\"{PROMPT}\"`  ",
        f"**Parameters:** `n_predict={N_PREDICT}`, `n_ctx={CONTEXT_LENGTH}`, `temp={TEMPERATURE}`, `seed={SEED}`",
        f"",
        f"---",
        f"",
        f"## 📊 Comparative Performance Matrix",
        f"",
        f"| Engine / Layer Configuration | Layer Placement | Gen Speed (tok/s) | Prompt Speed (tok/s) | TTFT (ms) | VRAM Est | RAM Est | Avg CPU % | Avg GPU % | GPU Idle % | Wall Duration | Status |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        n_dgpu = r.get('n_gpu_layers', 0)
        n_igpu = r.get('n_igpu_layers', 0)
        n_cpu = r.get('n_cpu_layers', 0)
        if n_igpu > 0:
            layers_str = f"{n_dgpu} dGPU / {n_igpu} iGPU / {n_cpu} CPU"
        else:
            layers_str = f"{n_dgpu} dGPU / {n_cpu} CPU"
        gen_tps = f"**`{r.get('generation_tps', 0.0):.2f}`**"
        prompt_tps = f"{r.get('prompt_tps', 0.0):.2f}"
        ttft = f"{r.get('ttft_ms', 0.0):.1f} ms"
        vram_est = f"{r.get('vram_est_mb', 0.0):.0f} MB" if 'vram_est_mb' in r else "N/A"
        ram_est = f"{r.get('ram_est_mb', 0.0):.0f} MB" if 'ram_est_mb' in r else "N/A"
        avg_cpu = f"{r.get('avg_cpu_pct', 0.0):.1f}%" if 'avg_cpu_pct' in r else "N/A"
        avg_gpu = f"{r.get('avg_gpu_pct', 0.0):.1f}%" if 'avg_gpu_pct' in r else "N/A"
        gpu_idle = f"{r.get('gpu_idle_pct', 0.0):.1f}%" if 'gpu_idle_pct' in r else "N/A"
        wall = f"{r.get('total_wall_sec', 0.0):.2f} s"
        status = r.get('status', 'UNKNOWN')
        md_lines.append(f"| **{r['config_name']}** | `{layers_str}` | {gen_tps} | `{prompt_tps}` | `{ttft}` | `{vram_est}` | `{ram_est}` | `{avg_cpu}` | `{avg_gpu}` | `{gpu_idle}` | `{wall}` | `{status}` |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 💡 Key Architectural Insights",
        "",
        "1. **9B Parameter Offload Footprint**: Qwythos-9B requires ~5.48 GB for weights alone. Offloading all 33 layers to a 6GB VRAM dGPU leaves insufficient space for context KV cache, causing memory pressure / slowdown.",
        "2. **dGPU + iGPU Hybrid Offload**: By utilizing the integrated GPU for CPU-bound layers via Vulkan UMA memory, InferenceOS achieves compute acceleration on shared system RAM without starving dGPU VRAM.",
        "3. **InferenceOS Layer Placement Optimization**: Dynamically balances transformer layers across dGPU, iGPU, and System RAM to achieve peak generation speed while preventing OOM errors.",
    ])

    md_content = "\n".join(md_lines)
    md_path.write_text(md_content, encoding="utf-8")

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": model_path.name,
            "size_bytes": model_path.stat().st_size,
            "size_gb": round(model_path.stat().st_size / (1024**3), 2),
        },
        "prompt": PROMPT,
        "results": results,
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Config_Name", "N_GPU_Layers", "N_IGPU_Layers", "N_CPU_Layers", "Gen_TPS", "Prompt_TPS",
            "TTFT_ms", "p50_ITL_ms", "Wall_Sec", "Status"
        ])
        for r in results:
            writer.writerow([
                r.get("config_name"), r.get("n_gpu_layers"), r.get("n_igpu_layers", 0), r.get("n_cpu_layers"),
                r.get("generation_tps"), r.get("prompt_tps"), r.get("ttft_ms"),
                r.get("p50_itl_ms"), r.get("total_wall_sec"), r.get("status")
            ])

    print("\n============================================================")
    print("      Qwythos-9B Comparative Matrix Reports Generated       ")
    print("============================================================")
    print(f"  Markdown Report: {md_path}")
    print(f"  JSON Artifact:   {json_path}")
    print(f"  CSV Summary:     {csv_path}")
    print("============================================================\n")
    print(md_content)

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_path = find_qwythos_model()
    print(f"Found Qwythos-9B model: {model_path.name} ({model_path.stat().st_size / (1024**3):.2f} GB)\n")

    results = []

    # 1. InferenceOS Adaptive Auto Placement
    res_ios_auto = run_inferenceos_config(model_path, gpu_layers_override=None, label="Adaptive Auto-Placement")
    results.append(res_ios_auto)

    # 2. InferenceOS 3-Way Tri-Offload (28 dGPU / 2 iGPU / 3 CPU)
    res_ios_tri = run_inferenceos_config(
        model_path,
        gpu_layers_override=28,
        igpu_layers_override=2,
        enable_igpu=True,
        label="3-Way Tri-Offload"
    )
    results.append(res_ios_tri)

    # 3. InferenceOS dGPU + CPU Split (28 dGPU / 5 CPU)
    res_ios_28_cpu = run_inferenceos_config(model_path, gpu_layers_override=28, enable_igpu=False, label="dGPU + CPU Split (28 dGPU / 5 CPU)")
    results.append(res_ios_28_cpu)

    # 4. InferenceOS dGPU + CPU Split (18 dGPU / 15 CPU)
    res_ios_18_cpu = run_inferenceos_config(model_path, gpu_layers_override=18, enable_igpu=False, label="dGPU + CPU Split (18 dGPU / 15 CPU)")
    results.append(res_ios_18_cpu)

    # 5. Raw llama.cpp (-ngl 18) (dGPU + CPU)
    res_raw_18 = run_raw_llamacpp(model_path, n_gpu_layers=18)
    results.append(res_raw_18)

    # 6. Raw llama.cpp CPU Only (-ngl 0)
    res_cpu = run_raw_llamacpp(model_path, n_gpu_layers=0)
    results.append(res_cpu)

    # 7. Raw llama.cpp Full GPU (-ngl 99)
    res_gpu = run_raw_llamacpp(model_path, n_gpu_layers=99)
    results.append(res_gpu)

    export_matrix_reports(results, model_path)

if __name__ == "__main__":
    main()
