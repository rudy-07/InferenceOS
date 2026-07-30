"""
run_benchmark.py
----------------
Benchmark script for InferenceOS.
Loads GGUF model, runs inference on Vulkan GPU backend, and measures performance metrics.
"""
import os
import sys
import time
import json
import subprocess
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
BUILD_DIR = PROJECT_ROOT / "build"
PROFILE_JSON = PROJECT_ROOT / "hardware_profile.json"


def find_llama_cli() -> Path | None:
    """Find the compiled llama-app / llama-cli executable."""
    system = platform.system()
    # llama-app monolith produces bin/llama.exe; older builds use llama-cli.exe
    names = (
        ["llama.exe", "llama-app.exe", "llama-cli.exe"]
        if system == "Windows"
        else ["llama", "llama-app", "llama-cli"]
    )

    for binary_name in names:
        candidates = [
            BUILD_DIR / "bin" / binary_name,
            BUILD_DIR / "bin" / "Release" / binary_name,
        ]
        for c in candidates:
            if c.exists():
                return c

    # Fallback: walk build/bin for any runnable llama binary
    bin_dir = BUILD_DIR / "bin"
    if bin_dir.exists():
        for f in bin_dir.rglob("llama*.exe" if system == "Windows" else "llama*"):
            if f.is_file() and f.suffix not in (".a", ".lib"):
                return f

    return None


def run_vulkan_benchmark(
    model_path: Path,
    prompt: str = "Explain the advantage of Vulkan GPU acceleration for AI inference in 3 concise bullet points.",
    n_gpu_layers: int = 99,
    n_predict: int = 256,
    n_ctx: int = 2048,
    threads: int = 6,
) -> dict:
    cli_path = find_llama_cli()
    if not cli_path:
        raise FileNotFoundError(f"llama-cli executable not found under {BUILD_DIR}. Please run build_engine first.")

    print(f"\n============================================================")
    print(f"       InferenceOS Vulkan GPU Inference Benchmark")
    print(f"============================================================")
    print(f"Model:           {model_path.name} ({model_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"Backend:         Vulkan GPU")
    print(f"GPU Offload:     -ngl {n_gpu_layers}")
    print(f"Context Window:  {n_ctx} tokens")
    print(f"CPU Threads:     {threads}")
    print(f"Executable:      {cli_path.name}")
    print(f"============================================================\n")

    # New llama.exe monolith uses subcommands: `llama.exe cli <flags>`
    # stdin=DEVNULL sends EOF immediately → exits after one response (no REPL loop)
    cmd = [
        str(cli_path),
        "cli",                  # subcommand
        "-m", str(model_path),
        "-p", prompt,
        "-ngl", str(n_gpu_layers),
        "-c", str(n_ctx),
        "-n", str(n_predict),
        "-t", str(threads),
        "--temp", "0.7",
        "-no-cnv",              # disable conversation / system prompt injection
    ]

    print("Running inference command...")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,  # send EOF → non-interactive single-shot mode
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    stdout_chunks = []
    stderr_chunks = []
    
    # Read output
    stdout, stderr = process.communicate()
    end_time = time.time()
    total_duration = end_time - start_time

    # Parse llama-cli metrics from stderr output
    # llama.cpp prints timing statistics in stderr like:
    # llama_perf_sampler_print:    sample time =     5.23 ms /   128 runs   (    0.04 ms per token, 24474.19 tokens per second)
    # llama_perf_context_print: prompt eval time =   150.40 ms /    32 tokens (    4.70 ms per token,   212.77 tokens per second)
    # llama_perf_context_print:        eval time =  2400.10 ms /   128 runs   (   18.75 ms per token,    53.33 tokens per second)
    
    metrics = {
        "model_name": model_path.name,
        "model_size_gb": round(model_path.stat().st_size / (1024**3), 2),
        "total_duration_sec": round(total_duration, 2),
        "prompt_eval_ms": None,
        "prompt_eval_tokens": None,
        "prompt_eval_tps": None,
        "eval_ms": None,
        "eval_tokens": None,
        "eval_tps": None,
        "raw_response": stdout.strip(),
        "exit_code": process.returncode
    }

    for line in stderr.splitlines():
        if "prompt eval time" in line:
            # Parse prompt eval timing
            parts = line.split("=")
            try:
                # Extract ms, tokens, tps
                # line example: llama_perf_context_print: prompt eval time = 150.40 ms / 32 tokens ( 4.70 ms per token, 212.77 tokens per second)
                sub_parts = line.split("prompt eval time =")[1].split("/")
                ms_val = float(sub_parts[0].replace("ms", "").strip())
                tokens_part = sub_parts[1].split("tokens")[0].strip()
                tokens_val = int(tokens_part)
                tps_val = float(line.split("tokens per second")[0].split(",")[-1].strip())
                metrics["prompt_eval_ms"] = ms_val
                metrics["prompt_eval_tokens"] = tokens_val
                metrics["prompt_eval_tps"] = tps_val
            except Exception:
                pass
        elif "eval time =" in line and "prompt" not in line:
            # Parse generation eval timing
            try:
                sub_parts = line.split("eval time =")[1].split("/")
                ms_val = float(sub_parts[0].replace("ms", "").strip())
                tokens_part = sub_parts[1].split("runs")[0].split("tokens")[0].strip()
                tokens_val = int(tokens_part)
                tps_val = float(line.split("tokens per second")[0].split(",")[-1].strip())
                metrics["eval_ms"] = ms_val
                metrics["eval_tokens"] = tokens_val
                metrics["eval_tps"] = tps_val
            except Exception:
                pass

    return metrics, stderr


import argparse
from orchestrator.gguf_selector import GGUFSelector, QUANT_BPW
from orchestrator.memory_planner import MemoryPlanner
from orchestrator.gguf_parser import read_gguf_metadata

# Phase 3 + 4 imports (graceful fallback if not yet built)
try:
    from layer_placement import PlacementEngine, ModelDescriptor
    from layer_placement.model_descriptor import infer_quant_type_from_filename
    from inference_runtime import RuntimeEngine, RuntimeConfig
    _PHASE4_AVAILABLE = True
except ImportError:
    _PHASE4_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="InferenceOS Dynamic Hardware-Aware Inference Benchmark")
    parser.add_argument("--model", type=str, default="auto", help="Base model name or 'auto' to select best quantization")
    parser.add_argument("--prompt", type=str, default="Explain the advantage of Vulkan GPU acceleration for AI inference in 3 concise bullet points.", help="Prompt text for inference")
    parser.add_argument("--ctx", type=int, default=4096, help="Desired context length")
    # --- Phase 4 flags ---
    parser.add_argument("--use-runtime", action="store_true",
                        help="Route through Phase 4 RuntimeEngine (PlacementEngine plan + RuntimeEngine execution). "
                             "Default: legacy run_vulkan_benchmark path.")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of benchmark runs when --use-runtime is active. Default 1.")
    parser.add_argument("--warmup-runs", type=int, default=0,
                        help="Warmup runs excluded from stats when --use-runtime is active. Default 0.")
    parser.add_argument("--predict", type=int, default=256,
                        help="Tokens to generate (Phase 4 runtime only). Default 256.")
    parser.add_argument("--threads", type=int, default=-1,
                        help="CPU threads (-1 = auto). Phase 4 runtime only.")
    args = parser.parse_args()

    if not PROFILE_JSON.exists():
        print(f"Error: Hardware profile not found at {PROFILE_JSON}. Please run profiler first.", file=sys.stderr)
        sys.exit(1)

    with open(PROFILE_JSON, "r", encoding="utf-8") as f:
        hw_profile = json.load(f)

    # Dynamic selector and memory planner initialization
    selector = GGUFSelector(hw_profile)
    planner = MemoryPlanner(hw_profile)

    model_file = selector.select_best_model(MODELS_DIR, target_model_name=args.model)
    if not model_file:
        print(f"Error: No GGUF models found in {MODELS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[InferenceOS Configurator] Dynamically selected optimal model quantization: {model_file.name}")

    # Read model metadata dynamically from binary GGUF header
    params = read_gguf_metadata(model_file)
    model_size_mb = model_file.stat().st_size / (1024 * 1024)

    # Extract quantization BPW rating
    quant_bpw = 4.5
    stem = model_file.stem.upper()
    for q, bpw in sorted(QUANT_BPW.items(), key=lambda x: len(x[0]), reverse=True):
        if q in stem or q.replace("_", "") in stem:
            quant_bpw = bpw
            break

    # Calculate optimal memory & GPU layer plan
    plan = planner.calculate_plan(
        model_size_mb=model_size_mb,
        num_layers=params["num_layers"],
        hidden_size=params["hidden_size"],
        num_heads=params["num_heads"],
        num_kv_heads=params["num_kv_heads"],
        quant_bpw=quant_bpw,
        desired_n_ctx=min(args.ctx, params.get("max_context_length", 4096))
    )

    print(f"[InferenceOS Configurator] Computed Dynamic Hardware Plan:")
    print(f"  - GPU Layer Offload (-ngl): {plan.n_gpu_layers} / {params['num_layers']} ({plan.offload_ratio*100:.1f}%)")
    print(f"  - Context Window (-c):     {plan.n_ctx} tokens")
    print(f"  - Estimated VRAM Footprint: {plan.estimated_vram_mb:.1f} MB")
    print(f"  - Estimated RAM Footprint:  {plan.estimated_ram_mb:.1f} MB\n")

    # -------------------------------------------------------------------------
    # Phase 4: RuntimeEngine path (--use-runtime)
    # -------------------------------------------------------------------------
    if getattr(args, "use_runtime", False):
        if not _PHASE4_AVAILABLE:
            print("Error: Phase 3/4 modules (layer_placement, inference_runtime) not found.",
                  file=sys.stderr)
            sys.exit(1)

        print("\n[InferenceOS Phase 4] Using RuntimeEngine execution path.")

        # Build Phase 3 ModelDescriptor from GGUF metadata
        quant_type = infer_quant_type_from_filename(model_file.name) or "Q4_K_M"
        model_descriptor = ModelDescriptor.from_gguf_metadata(
            params,
            model_size_bytes=model_file.stat().st_size,
            quant_type=quant_type,
            model_name=model_file.stem,
        )

        # Run Phase 3 optimizer
        placement_engine = PlacementEngine(hw_profile=hw_profile)
        p3_plan = placement_engine.generatePlacementPlan(
            model_descriptor, context_length=args.ctx
        )
        print(placement_engine.generateTextReport(p3_plan))

        # Build Phase 4 RuntimeConfig
        rt_config = RuntimeConfig(
            n_predict=args.predict,
            context_length=args.ctx,
            threads=args.threads,
            use_flash_attn=True,
        )

        engine = RuntimeEngine(hw_profile=hw_profile, config=rt_config)

        if args.n_runs > 1 or args.warmup_runs > 0:
            # Multi-run benchmark mode
            bench = engine.benchmark(
                plan=p3_plan,
                model_path=model_file,
                n_runs=args.n_runs,
                warmup_runs=args.warmup_runs,
                prompt=args.prompt,
            )
            print(bench.report())
        else:
            # Single-run inference with streaming output
            print("\n[InferenceOS Phase 4] Running single inference...")
            result = engine.executePlan(
                plan=p3_plan,
                model_path=model_file,
                prompt=args.prompt,
                on_token=lambda t: print(t, end="", flush=True),
            )
            print("\n")
            stats = result.stats
            print("\n============================================================")
            print("               Phase 4 Runtime Metrics")
            print("============================================================")
            print(f"  Model:                  {model_file.name}")
            print(f"  Backend:                {result.backend.upper()}")
            print(f"  GPU Layers:             {stats.n_gpu_layers}")
            print(f"  Total Wall Time:        {stats.total_wall_ms / 1000:.2f} s")
            if stats.prompt_eval_tps > 0:
                print(f"  Prompt Processing:      {stats.prompt_eval_tps:.2f} tok/s ({stats.prompt_eval_ms:.1f} ms)")
            if stats.eval_tps > 0:
                print(f"  Generation Speed:       {stats.eval_tps:.2f} tok/s ({stats.eval_ms:.1f} ms)")
            print(f"  p95 Token Latency:      {stats.p95_latency_ms:.2f} ms")
            print(f"  Avg GPU Util:           {stats.avg_gpu_util_pct:.1f}%")
            print(f"  Avg CPU Util:           {stats.avg_cpu_util_pct:.1f}%")
            print(f"  Pipeline Stalls:        {stats.pipeline_stalls}")
            print(f"  Estimated Transfer:     {stats.estimated_transfer_ms:.2f} ms")
            print(f"  Compute Time:           {stats.compute_time_ms:.2f} ms")
            print(f"  Status:                 {'SUCCESS' if result.success else 'FAILED'}")
            print("============================================================\n")
        return

    # -------------------------------------------------------------------------
    # Legacy path: run_vulkan_benchmark (default, unchanged)
    # -------------------------------------------------------------------------
    metrics, stderr = run_vulkan_benchmark(
        model_path=model_file,
        prompt=args.prompt,
        n_gpu_layers=plan.n_gpu_layers,
        n_ctx=plan.n_ctx
    )

    print("\n------------------------------------------------------------")
    print("                Generated Model Output")
    print("------------------------------------------------------------")
    print(metrics["raw_response"])
    print("------------------------------------------------------------")

    print("\n============================================================")
    print("               Inference Metrics Summary")
    print("============================================================")
    print(f"  Model Name:             {metrics['model_name']}")
    print(f"  Total Duration:         {metrics['total_duration_sec']} s")
    if metrics["prompt_eval_tps"]:
        print(f"  Prompt Processing:      {metrics['prompt_eval_tps']:.2f} tokens/sec ({metrics['prompt_eval_tokens']} tokens in {metrics['prompt_eval_ms']:.1f} ms)")
    if metrics["eval_tps"]:
        print(f"  Generation Speed:       {metrics['eval_tps']:.2f} tokens/sec ({metrics['eval_tokens']} tokens in {metrics['eval_ms']:.1f} ms)")
    print(f"  Execution Status:       {'SUCCESS' if metrics['exit_code'] == 0 else 'FAILED'}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
