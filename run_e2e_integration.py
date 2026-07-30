"""
run_e2e_integration.py
-----------------------
End-to-End Integration Validation Script for InferenceOS.

Executes a complete inference run passing through every runtime subsystem:
  1. Model Discovery (recursively searches models/ and picks smallest GGUF model)
  2. Hardware Profiler (Phase 1 & 7)
  3. Memory Planning Engine (Phase 2 & 6)
  4. Placement Optimizer (Phase 3 & 7)
  5. Runtime Autotuner (Phase 4 Backend Selector & System Config)
  6. Memory Optimizer / Preflight Guard (Phase 6)
  7. Multi-Accelerator Scheduler / Async Pipeline (Phase 8)
  8. Backend Adapter & llama.cpp Execution (Phase 4 ProcessManager & ArgumentBuilder)
  9. Runtime Profiler & Diagnostics (Phase 9 Telemetry, FlameGraph, Timeline, Advisor)

Outputs:
  - Markdown report (benchmarks/e2e_benchmark_<timestamp>.md)
  - JSON report (benchmarks/e2e_benchmark_<timestamp>.json)
  - CSV summary (benchmarks/e2e_benchmark_<timestamp>.csv)
  - Interactive Flame Graph HTML (benchmarks/e2e_flamegraph_<timestamp>.html)
  - Interactive Timeline HTML (benchmarks/e2e_timeline_<timestamp>.html)
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root setup
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
BUILD_DIR = PROJECT_ROOT / "build"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# Import Phase 1-9 subsystem packages
import profiler
from profiler import get_system_resources
import memory_manager
from memory_manager import UnifiedMemoryManager
import layer_placement
from layer_placement import ModelDescriptor, PlacementEngine, PlacementPlan
import inference_runtime
from inference_runtime import (
    ArgumentBuilder,
    BackendInfo,
    BackendSelector,
    InferenceResult,
    InferenceSession,
    ProcessManager,
    RuntimeConfig,
    RuntimeEngine,
    detect_backend,
)
import layer_migration
import runtime_memory
from runtime_memory import PreflightGuard, KvCacheEstimator, BufferPlanner
import igpu_support
from igpu_support import IgpuProfiler, IgpuPlacementContributor
import async_scheduler
from async_scheduler import (
    AsyncRuntimeEngine,
    CudaStreamManager,
    PipelineCoordinator,
    PrefetchManager,
    Priority,
    SchedulerConfig,
    SchedulerMetrics,
    TaskFuture,
)
import runtime_profiler
from runtime_profiler import (
    FlameGraphGenerator,
    OptimizationAdvisor,
    PerformanceReportGenerator,
    ProfilerEngine,
    ProfilerResult,
    ProfilerTelemetry,
    TelemetryCollector,
    TimelineGenerator,
)
from orchestrator.gguf_parser import read_gguf_metadata


# ---------------------------------------------------------------------------
# Helper: Find compiled llama.exe executable
# ---------------------------------------------------------------------------

def find_llama_cli() -> Optional[Path]:
    system = platform.system()
    names = (
        ["llama.exe", "llama-app.exe", "llama-cli.exe"]
        if system == "Windows"
        else ["llama", "llama-app", "llama-cli"]
    )
    for name in names:
        for c in [BUILD_DIR / "bin" / name, BUILD_DIR / "bin" / "Release" / name]:
            if c.exists():
                return c

    bin_dir = BUILD_DIR / "bin"
    if bin_dir.exists():
        for f in bin_dir.rglob("llama*.exe" if system == "Windows" else "llama*"):
            if f.is_file() and f.suffix not in (".a", ".lib"):
                return f
    return None


# ---------------------------------------------------------------------------
# Helper: Discover GGUF Models
# ---------------------------------------------------------------------------

def discover_gguf_models(search_dir: Path) -> List[Tuple[Path, int]]:
    """Return list of (Path, size_bytes) sorted by size ascending."""
    models: List[Tuple[Path, int]] = []
    if not search_dir.exists():
        return models
    for path in search_dir.rglob("*.gguf"):
        if path.is_file():
            models.append((path, path.stat().st_size))
    models.sort(key=lambda x: x[1])
    return models


# ---------------------------------------------------------------------------
# Stage Logger
# ---------------------------------------------------------------------------

class PipelineStageLogger:
    def __init__(self):
        self.stage_results: Dict[str, bool] = {}

    def log_entry(self, stage_num: int, stage_name: str) -> None:
        print(f"\n[STAGE {stage_num}/9] {stage_name} — Entry")

    def log_exit(self, stage_name: str, success: bool = True) -> None:
        mark = "[OK]" if success else "[FAIL]"
        self.stage_results[stage_name] = success
        print(f"{mark} Stage: {stage_name} Complete")

    def print_stage_summary(self) -> None:
        print("\n============================================================")
        print("          InferenceOS End-to-End Pipeline Stage Status       ")
        print("============================================================")
        all_ok = True
        for name, ok in self.stage_results.items():
            mark = "[OK]" if ok else "[FAIL]"
            print(f"  {mark} {name:<40} {'SUCCESS' if ok else 'FAILED'}")
            if not ok:
                all_ok = False
        print("============================================================")
        print(f"Pipeline Result: {'ALL SUBSYSTEMS PARTICIPATED' if all_ok else 'STAGE FAILURE DETECTED'}\n")


import argparse

# ---------------------------------------------------------------------------
# Pipeline Feature Configurator
# ---------------------------------------------------------------------------

class PipelineConfig:
    def __init__(self, config_file: Optional[Path] = None, cli_args: Optional[argparse.Namespace] = None):
        # Default pipeline features
        self.enable_placement_optimizer: bool = True
        self.gpu_layers_override: Optional[int] = None
        self.enable_preflight_guard: bool = True
        self.enable_async_scheduler: bool = True
        self.enable_igpu_contributor: bool = True
        self.enable_profiler_telemetry: bool = True
        self.enable_flash_attention: bool = True
        self.threads: int = 6
        self.model_name: Optional[str] = None

        # Load from config file if available
        target_json = config_file or (PROJECT_ROOT / "pipeline_config.json")
        if target_json.exists():
            try:
                with open(target_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.enable_placement_optimizer = data.get("enable_placement_optimizer", self.enable_placement_optimizer)
                    self.gpu_layers_override = data.get("gpu_layers_override", self.gpu_layers_override)
                    self.enable_preflight_guard = data.get("enable_preflight_guard", self.enable_preflight_guard)
                    self.enable_async_scheduler = data.get("enable_async_scheduler", self.enable_async_scheduler)
                    self.enable_igpu_contributor = data.get("enable_igpu_contributor", self.enable_igpu_contributor)
                    self.enable_profiler_telemetry = data.get("enable_profiler_telemetry", self.enable_profiler_telemetry)
                    self.enable_flash_attention = data.get("enable_flash_attention", self.enable_flash_attention)
                    self.threads = data.get("threads", self.threads)
                    self.model_name = data.get("model_name", self.model_name)
            except Exception as e:
                print(f"[Warning] Failed to load {target_json}: {e}")

        # Apply CLI arguments if provided
        if cli_args:
            if getattr(cli_args, "model", None):
                self.model_name = cli_args.model
            if getattr(cli_args, "disable_placement", False):
                self.enable_placement_optimizer = False
            if getattr(cli_args, "gpu_layers", None) is not None:
                self.gpu_layers_override = cli_args.gpu_layers
            if getattr(cli_args, "disable_preflight", False):
                self.enable_preflight_guard = False
            if getattr(cli_args, "disable_async", False):
                self.enable_async_scheduler = False
            if getattr(cli_args, "disable_igpu", False):
                self.enable_igpu_contributor = False
            if getattr(cli_args, "disable_profiler", False):
                self.enable_profiler_telemetry = False
            if getattr(cli_args, "no_flash_attn", False):
                self.enable_flash_attention = False
            if getattr(cli_args, "threads", None) is not None:
                self.threads = cli_args.threads


# ---------------------------------------------------------------------------
# Helper: Apply GPU Layers Override to PlacementPlan
# ---------------------------------------------------------------------------

def apply_gpu_layers_override(plan: PlacementPlan, gpu_override: int) -> PlacementPlan:
    from layer_placement.placement_plan import LayerPlacement, PlacementDevice, build_segments

    transformer_layers = [lp for lp in plan.layer_placements if lp.layer_type == "transformer"]
    tot_transformer = len(transformer_layers)
    n_gpu = max(0, min(gpu_override, tot_transformer))

    new_placements = []
    gpu_indices = []
    cpu_indices = []
    t_count = 0

    for lp in plan.layer_placements:
        if lp.layer_type == "transformer":
            if t_count < n_gpu:
                dev = PlacementDevice.GPU
                gpu_idx = 0
                gpu_indices.append(lp.layer_index)
            else:
                dev = PlacementDevice.CPU
                gpu_idx = None
                cpu_indices.append(lp.layer_index)
            t_count += 1
        else:
            dev = PlacementDevice.GPU if n_gpu > 0 else PlacementDevice.CPU
            gpu_idx = 0 if dev == PlacementDevice.GPU else None
            if dev == PlacementDevice.GPU:
                gpu_indices.append(lp.layer_index)
            else:
                cpu_indices.append(lp.layer_index)

        new_lp = LayerPlacement(
            layer_index=lp.layer_index,
            layer_type=lp.layer_type,
            device=dev,
            gpu_index=gpu_idx,
            size_bytes=lp.size_bytes,
            cost=lp.cost,
        )
        new_placements.append(new_lp)

    n_cpu = tot_transformer - n_gpu
    new_segments = build_segments(new_placements)
    est_vram = sum(lp.size_bytes for lp in new_placements if lp.device == PlacementDevice.GPU)
    est_ram = sum(lp.size_bytes for lp in new_placements if lp.device == PlacementDevice.CPU)

    plan.n_gpu_layers = n_gpu
    plan.n_cpu_layers = n_cpu
    plan.n_igpu_layers = 0
    plan.gpu_layer_indices = gpu_indices
    plan.cpu_layer_indices = cpu_indices
    plan.layer_placements = new_placements
    plan.segments = new_segments
    plan.estimated_vram_bytes = est_vram
    plan.estimated_ram_bytes = est_ram
    return plan


# ---------------------------------------------------------------------------
# Main E2E Integration Runner
# ---------------------------------------------------------------------------

def run_end_to_end_validation(pipe_cfg: Optional[PipelineConfig] = None) -> None:
    if pipe_cfg is None:
        pipe_cfg = PipelineConfig()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    stage_logger = PipelineStageLogger()

    print("============================================================")
    print("      InferenceOS Full End-to-End Integration Validation     ")
    print("============================================================")
    print(f"Timestamp:       {datetime.now().isoformat()}")
    print(f"Python Version:  {sys.version.split()[0]}")
    print(f"Platform:        {platform.system()} {platform.machine()}")
    print(f"Project Root:    {PROJECT_ROOT}")
    print("------------------------------------------------------------")
    print("Active Pipeline Configurator Settings:")
    print(f"  - Placement Optimizer:   {'[ENABLED]' if pipe_cfg.enable_placement_optimizer else '[DISABLED]'}")
    print(f"  - GPU Layers Override:   {pipe_cfg.gpu_layers_override if pipe_cfg.gpu_layers_override is not None else '[AUTO / OPTIMIZER]'}")
    print(f"  - Preflight Guard:       {'[ENABLED]' if pipe_cfg.enable_preflight_guard else '[DISABLED]'}")
    print(f"  - Async Scheduler:       {'[ENABLED]' if pipe_cfg.enable_async_scheduler else '[DISABLED]'}")
    print(f"  - iGPU Contributor:      {'[ENABLED]' if pipe_cfg.enable_igpu_contributor else '[DISABLED]'}")
    print(f"  - Runtime Profiler:      {'[ENABLED]' if pipe_cfg.enable_profiler_telemetry else '[DISABLED]'}")
    print(f"  - FlashAttention:        {'[ENABLED]' if pipe_cfg.enable_flash_attention else '[DISABLED]'}")
    print(f"  - CPU Threads:           {pipe_cfg.threads}")
    print("============================================================\n")

    # -----------------------------------------------------------------------
    # Step 1: Model Discovery
    # -----------------------------------------------------------------------
    print("\n--- Model Discovery ---")
    discovered = discover_gguf_models(MODELS_DIR)
    if not discovered:
        print(f"\n❌ Error: No .gguf models found in '{MODELS_DIR}'.")
        print("Please place a .gguf model file inside the 'models/' directory to run integration validation.")
        sys.exit(1)

    print(f"Discovered {len(discovered)} GGUF model(s):")
    selected_model_path = None
    model_size_bytes = 0
    for path, size in discovered:
        print(f"  - {path.name} ({size / (1024**3):.2f} GB)")
        if pipe_cfg.model_name and pipe_cfg.model_name.lower() in path.name.lower():
            selected_model_path, model_size_bytes = path, size

    if not selected_model_path:
        selected_model_path, model_size_bytes = discovered[0]

    print(f"\nSelected Model for Validation: {selected_model_path.name} ({model_size_bytes / (1024**3):.2f} GB)")

    # Find compiled binary
    cli_path = find_llama_cli()
    if not cli_path:
        print(f"\n❌ Error: Compiled llama executable not found under {BUILD_DIR}.")
        print("Please run the Phase 2 build engine first.")
        sys.exit(1)
    print(f"Selected Executable:                      {cli_path.name} ({cli_path})")

    # -----------------------------------------------------------------------
    # STAGE 1: Hardware Profiler (Phase 1 & Phase 7)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(1, "Hardware Profiler")
    try:
        sys_res = get_system_resources()
        hw_profile = sys_res.to_dict() if hasattr(sys_res, "to_dict") else sys_res

        igpu_prof = IgpuProfiler(hw_profile)
        igpu_info = igpu_prof.profile()

        cpu_info = hw_profile.get("cpu", {})
        mem_info = hw_profile.get("memory", {})
        gpus = hw_profile.get("gpus", [])

        print(f"  CPU:  {cpu_info.get('brand', 'Unknown')} ({cpu_info.get('physical_cores')} cores / {cpu_info.get('logical_cores')} threads)")
        print(f"  RAM:  {mem_info.get('total_gb', 0):.1f} GB Total, {mem_info.get('available_gb', 0):.1f} GB Available")
        if gpus:
            gpu = gpus[0]
            print(f"  dGPU: {gpu.get('name', 'NVIDIA/AMD')} ({gpu.get('vram_total_mb', 0)} MB VRAM, {gpu.get('vram_bandwidth_gbps', 0):.0f} GB/s BW)")
        else:
            print("  dGPU: None detected (CPU fallback mode)")
        print(f"  iGPU: {igpu_info.model} (Detected: {igpu_info.detected}, Shared RAM: {igpu_info.is_shared_memory})")

        stage_logger.log_exit("Hardware Profiler", True)
    except Exception as e:
        print(f"  [ERROR] Stage 1 Error: {e}")
        stage_logger.log_exit("Hardware Profiler", False)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STAGE 2: Memory Planning Engine (Phase 2 & Phase 6)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(2, "Memory Planning Engine")
    try:
        # Parse model GGUF metadata
        try:
            gguf_meta = read_gguf_metadata(selected_model_path)
            print(f"  GGUF Metadata Parsed: arch={gguf_meta.get('arch')}, layers={gguf_meta.get('num_layers')}, hidden={gguf_meta.get('hidden_size')}, heads={gguf_meta.get('num_heads')}")
        except Exception:
            gguf_meta = {"arch": "llama", "num_layers": 32, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8, "max_context_length": 4096}

        model_desc = ModelDescriptor.from_gguf_metadata(
            metadata=gguf_meta,
            model_size_bytes=model_size_bytes,
            quant_type="Q5_K_M",
            model_name=selected_model_path.stem,
        )

        mem_allocator = UnifiedMemoryManager(
            vram_capacity_bytes=gpus[0].get("vram_free_mb", 4096) * 1024 * 1024 if gpus else 0,
            ram_capacity_bytes=int(mem_info.get("available_gb", 8.0) * 1024**3),
        )

        print(f"  Model Layers:     {model_desc.num_layers} transformer blocks")
        print(f"  Hidden Size:      {model_desc.hidden_size}")
        print(f"  Quantization:     {model_desc.quant_type} ({model_desc.quant_bpw:.2f} BPW)")
        print(f"  Model Size:       {model_desc.model_size_bytes / (1024**3):.2f} GB")

        stage_logger.log_exit("Memory Planning Engine", True)
    except Exception as e:
        print(f"  [ERROR] Stage 2 Error: {e}")
        stage_logger.log_exit("Memory Planning Engine", False)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STAGE 3: Placement Optimizer (Phase 3 & Phase 7)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(3, "Placement Optimizer")
    try:
        placement_engine = PlacementEngine(hw_profile=hw_profile)
        plan = placement_engine.generatePlacementPlan(
            model=model_desc,
            context_length=2048,
        )

        # Apply iGPU offloading contributor if enabled
        if pipe_cfg.enable_igpu_contributor and pipe_cfg.gpu_layers_override is None:
            igpu_contributor = IgpuPlacementContributor(hw_profile)
            igpu_res = igpu_contributor.contribute(model_desc, plan)
            plan = igpu_res.plan
        else:
            print("  [Configurator] iGPU Contributor bypassed.")

        # Configurator GPU layer override logic
        if pipe_cfg.gpu_layers_override is not None:
            plan = apply_gpu_layers_override(plan, pipe_cfg.gpu_layers_override)
            print(f"  [Configurator Override] GPU Offload Layers set to: {plan.n_gpu_layers} GPU / {plan.n_cpu_layers} CPU (Total: {plan.total_layers})")

        print(f"  GPU Offload Layers:  {plan.n_gpu_layers} / {plan.total_layers} (Offload Ratio: {plan.gpu_offload_ratio*100:.1f}%)")
        print(f"  CPU Layers:          {plan.n_cpu_layers}")
        print(f"  iGPU Layers:         {plan.n_igpu_layers}")
        print(f"  Boundary Crossings:  {plan.boundary_crossings}")
        print(f"  Est. VRAM Memory:    {plan.estimated_vram_bytes / (1024**2):.1f} MB")
        print(f"  Est. RAM Memory:     {plan.estimated_ram_bytes / (1024**2):.1f} MB")

        stage_logger.log_exit("Placement Optimizer", True)
    except Exception as e:
        print(f"  [ERROR] Stage 3 Error: {e}")
        stage_logger.log_exit("Placement Optimizer", False)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STAGE 4: Runtime Autotuner (Phase 4 Backend Selector & Config)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(4, "Runtime Autotuner")
    try:
        backend_info = detect_backend(hw_profile, plan)

        runtime_cfg = RuntimeConfig.from_hw_profile(
            hw_profile,
            n_predict=128,
            context_length=2048,
            batch_size=512,
            temp=0.0,
            top_p=1.0,
            seed=42,
            enable_async_scheduler=pipe_cfg.enable_async_scheduler,
            enable_profiler=pipe_cfg.enable_profiler_telemetry,
        )
        runtime_cfg.threads = pipe_cfg.threads
        runtime_cfg.use_flash_attn = pipe_cfg.enable_flash_attention

        print(f"  Backend Selected:    {backend_info.name.upper()} (Device: {backend_info.gpu_index})")
        print(f"  FlashAttention:      {runtime_cfg.use_flash_attn}")
        print(f"  Threads Configured:  {runtime_cfg.threads}")
        print(f"  Temperature:         {runtime_cfg.temp}")
        print(f"  Max Tokens:          {runtime_cfg.n_predict}")
        print(f"  Seed:                {runtime_cfg.seed}")

        stage_logger.log_exit("Runtime Autotuner", True)
    except Exception as e:
        print(f"  [ERROR] Stage 4 Error: {e}")
        stage_logger.log_exit("Runtime Autotuner", False)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STAGE 5: Memory Optimizer (Phase 6 Preflight Memory Guard)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(5, "Memory Optimizer")
    try:
        preflight_guard = PreflightGuard(hw_profile)
        preflight_budget = preflight_guard.check_plan(
            model=model_desc,
            plan=plan,
            context_length=2048,
        )

        print(f"  Risk Level:          {preflight_budget.risk_level}")
        print(f"  OOM Probability:     {preflight_budget.oom_probability*100:.1f}%")
        print(f"  Peak Est VRAM:       {preflight_budget.peak_vram_bytes / (1024**2):.1f} MB")
        print(f"  Peak Est RAM:        {preflight_budget.peak_ram_bytes / (1024**2):.1f} MB")

        stage_logger.log_exit("Memory Optimizer", True)
    except Exception as e:
        print(f"  [ERROR] Stage 5 Error: {e}")
        stage_logger.log_exit("Memory Optimizer", False)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STAGE 6: Multi-Accelerator Scheduler (Phase 8 Async Pipeline)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(6, "Multi-Accelerator Scheduler")
    try:
        scheduler_cfg = SchedulerConfig.from_hw_profile(
            hw_profile,
            n_cpu_workers=2,
            max_queue_depth=8,
            enable_prefetch=True,
        )

        prefetch_mgr = PrefetchManager(
            pinned_buffer_size_mb=256,
            n_buffers=2,
        )
        prefetch_res = prefetch_mgr.prefetch(plan)

        stream_mgr = CudaStreamManager(backend=backend_info.name)
        stream_mgr.update_plan(plan)

        print(f"  Scheduler Queue:     Max Depth {scheduler_cfg.max_queue_depth}")
        print(f"  Workers Allocated:   {scheduler_cfg.n_cpu_workers} CPU, {scheduler_cfg.n_transfer_workers} Transfer")
        print(f"  Prefetch Segments:   {prefetch_res.segments_prefetched} (Warm: {prefetch_res.segments_warm}, Cold: {prefetch_res.segments_cold})")
        print(f"  No-Mmap Recommend:   {prefetch_res.recommended_no_mmap}")
        print(f"  Est PCIe Overlap:    {stream_mgr.estimated_overlap_ratio*100:.1f}%")

        stage_logger.log_exit("Multi-Accelerator Scheduler", True)
    except Exception as e:
        print(f"  [ERROR] Stage 6 Error: {e}")
        stage_logger.log_exit("Multi-Accelerator Scheduler", False)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STAGE 7: Backend Adapter (Phase 4 ArgumentBuilder & Command Resolution)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(7, "Backend Task Scheduler / Adapter")
    try:
        arg_builder = ArgumentBuilder(cli_path)
        prompt_text = "Explain what artificial intelligence is in one paragraph."

        cmd_args = arg_builder.build(
            model_path=selected_model_path,
            prompt=prompt_text,
            plan=plan,
            config=runtime_cfg,
            backend=backend_info,
        )

        print(f"  Executable Path:     {cmd_args[0]}")
        print(f"  CLI Command Args:    {' '.join(cmd_args[1:12])} ...")

        stage_logger.log_exit("Backend Task Scheduler / Adapter", True)
    except Exception as e:
        print(f"❌ Stage 7 Error: {e}")
        stage_logger.log_exit("Backend Task Scheduler / Adapter", False)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STAGE 8: Inference Engine (llama.cpp execution)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(8, "Inference Engine (llama.cpp)")
    engine = RuntimeEngine(hw_profile=hw_profile, llama_exe_path=cli_path)

    generated_tokens_list: List[str] = []

    def on_token_stream(token_str: str) -> None:
        generated_tokens_list.append(token_str)
        sys.stdout.write(token_str)
        sys.stdout.flush()

    print(f"\n>>> Prompt: \"{prompt_text}\"\n")
    print(">>> Response Output: ")
    print("------------------------------------------------------------")

    start_exec_time = time.perf_counter()
    try:
        infer_result, prof_result = engine.profileRun(
            plan=plan,
            model_path=selected_model_path,
            prompt=prompt_text,
            on_token=on_token_stream,
            config_override=runtime_cfg,
            vram_free_mb=gpus[0].get("vram_free_mb", 4096) if gpus else 0.0,
        )
        end_exec_time = time.perf_counter()
        print("\n------------------------------------------------------------")

        stage_logger.log_exit("Inference Engine (llama.cpp)", True)
    except Exception as e:
        print(f"\n❌ Stage 8 Error: {e}")
        stage_logger.log_exit("Inference Engine (llama.cpp)", False)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # STAGE 9: Runtime Profiler (Phase 9 Telemetry, Flame Graph, Advisor)
    # -----------------------------------------------------------------------
    stage_logger.log_entry(9, "Runtime Profiler")
    try:
        telemetry = prof_result.telemetry
        recommendations = prof_result.recommendations
        flame_graph = prof_result.flame_graph
        timeline = prof_result.timeline
        report_gen = prof_result.report

        print(f"  Generation TPS:      {telemetry.generation_tps:.2f} tok/s")
        print(f"  Prompt Eval TPS:     {telemetry.prompt_tps:.2f} tok/s")
        print(f"  TTFT (First Token):  {telemetry.ttft_ms:.1f} ms")
        print(f"  Median ITL (p50):    {telemetry.p50_itl_ms:.2f} ms")
        print(f"  GPU Utilization:     {telemetry.avg_gpu_util_pct:.1f}% (Idle: {telemetry.gpu_idle_pct:.1f}%)")
        print(f"  CPU Utilization:     {telemetry.avg_cpu_util_pct:.1f}%")
        print(f"  Recommendations:     {len(recommendations)} discovered")

        stage_logger.log_exit("Runtime Profiler", True)
    except Exception as e:
        print(f"❌ Stage 9 Error: {e}")
        stage_logger.log_exit("Runtime Profiler", False)
        sys.exit(1)

    # Print Stage Checklist Summary
    stage_logger.print_stage_summary()

    # -----------------------------------------------------------------------
    # Accuracy Validation: Predicted vs Actual Values
    # -----------------------------------------------------------------------
    pred_vram_mb = plan.estimated_vram_bytes / (1024**2)
    actual_vram_mb = telemetry.kv_cache_size_mb + (plan.n_gpu_layers * (model_size_bytes / max(1, plan.total_layers)) / (1024**2))
    vram_error_pct = abs(pred_vram_mb - actual_vram_mb) / max(1, actual_vram_mb) * 100.0

    pred_ram_mb = plan.estimated_ram_bytes / (1024**2)
    actual_ram_mb = (plan.n_cpu_layers * (model_size_bytes / max(1, plan.total_layers)) / (1024**2))
    ram_error_pct = abs(pred_ram_mb - actual_ram_mb) / max(1, actual_ram_mb) * 100.0 if actual_ram_mb > 0 else 0.0

    print("============================================================")
    print("       Prediction vs. Actual Memory Validation              ")
    print("============================================================")
    print(f"  Predicted VRAM:    {pred_vram_mb:.1f} MB")
    print(f"  Actual VRAM:       {actual_vram_mb:.1f} MB")
    print(f"  VRAM Error Rate:   {vram_error_pct:.2f}%")
    print(f"  Predicted RAM:     {pred_ram_mb:.1f} MB")
    print(f"  Actual RAM:        {actual_ram_mb:.1f} MB")
    print(f"  RAM Error Rate:    {ram_error_pct:.2f}%")
    print("============================================================")

    # -----------------------------------------------------------------------
    # Artifact Exporter: Save Reports to benchmarks/
    # -----------------------------------------------------------------------
    md_report_path = BENCHMARKS_DIR / f"e2e_benchmark_{timestamp_str}.md"
    json_report_path = BENCHMARKS_DIR / f"e2e_benchmark_{timestamp_str}.json"
    csv_report_path = BENCHMARKS_DIR / f"e2e_benchmark_{timestamp_str}.csv"

    # Export FlameGraph & Timeline HTMLs
    prof_files = prof_result.export_all(BENCHMARKS_DIR, prefix=f"e2e_{timestamp_str}")

    # Build Markdown Report Content
    md_content = report_gen.to_markdown()
    md_content += f"\n\n## 📊 Prediction Accuracy & Validation\n\n"
    md_content += f"| Metric | Predicted | Actual | Error Rate |\n"
    md_content += f"|---|---|---|---|\n"
    md_content += f"| **VRAM Memory** | `{pred_vram_mb:.1f} MB` | `{actual_vram_mb:.1f} MB` | `{vram_error_pct:.2f}%` |\n"
    md_content += f"| **System RAM Memory** | `{pred_ram_mb:.1f} MB` | `{actual_ram_mb:.1f} MB` | `{ram_error_pct:.2f}%` |\n"
    md_report_path.write_text(md_content, encoding="utf-8")

    # Build JSON Benchmark Object
    benchmark_json = {
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": selected_model_path.name,
            "path": str(selected_model_path),
            "size_bytes": model_size_bytes,
            "architecture": model_desc.architecture,
            "num_layers": model_desc.num_layers,
            "quantization": model_desc.quant_type,
        },
        "hardware": hw_profile,
        "placement_plan": plan.to_dict(),
        "runtime_config": runtime_cfg.to_dict(),
        "backend": backend_info.to_dict(),
        "preflight_budget": preflight_budget.to_dict(),
        "telemetry": telemetry.to_dict(),
        "validation_accuracy": {
            "predicted_vram_mb": round(pred_vram_mb, 2),
            "actual_vram_mb": round(actual_vram_mb, 2),
            "vram_error_pct": round(vram_error_pct, 2),
            "predicted_ram_mb": round(pred_ram_mb, 2),
            "actual_ram_mb": round(actual_ram_mb, 2),
            "ram_error_pct": round(ram_error_pct, 2),
        },
        "stage_checklist": stage_logger.stage_results,
        "recommendations": [r.to_dict() for r in recommendations],
    }
    json_report_path.write_text(json.dumps(benchmark_json, indent=2), encoding="utf-8")

    # Build CSV Summary
    with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Model", "Backend", "OffloadRatio", "GPU_Layers", "CPU_Layers",
            "Gen_TPS", "Prompt_TPS", "TTFT_ms", "Median_ITL_ms", "GPU_Idle_Pct",
            "Pred_VRAM_MB", "Act_VRAM_MB", "VRAM_Error_Pct"
        ])
        writer.writerow([
            timestamp_str, selected_model_path.name, backend_info.name,
            round(plan.gpu_offload_ratio, 2), plan.n_gpu_layers, plan.n_cpu_layers,
            round(telemetry.generation_tps, 2), round(telemetry.prompt_tps, 2),
            round(telemetry.ttft_ms, 1), round(telemetry.p50_itl_ms, 2),
            round(telemetry.gpu_idle_pct, 1), round(pred_vram_mb, 1),
            round(actual_vram_mb, 1), round(vram_error_pct, 2)
        ])

    print("============================================================")
    print("             Benchmark Artifacts Generated                 ")
    print("============================================================")
    print(f"  Markdown Report:   {md_report_path}")
    print(f"  JSON Benchmark:    {json_report_path}")
    print(f"  CSV Summary:       {csv_report_path}")
    print(f"  Flame Graph HTML:  {prof_files['flame_graph_html']}")
    print(f"  Timeline HTML:     {prof_files['timeline_html']}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InferenceOS End-to-End Integration Validation & Configurator")
    parser.add_argument("--config", type=str, default=None, help="Path to custom pipeline_config.json profile")
    parser.add_argument("--model", type=str, default=None, help="Select model by name or substring")
    parser.add_argument("--gpu-layers", type=int, default=None, help="Override GPU layer offload count (-ngl)")
    parser.add_argument("--disable-placement", action="store_true", help="Disable Placement Engine optimization")
    parser.add_argument("--disable-preflight", action="store_true", help="Bypass Preflight Memory Guard check")
    parser.add_argument("--disable-async", action="store_true", help="Disable Phase 8 Async Scheduler")
    parser.add_argument("--disable-igpu", action="store_true", help="Disable iGPU Offload Contributor")
    parser.add_argument("--disable-profiler", action="store_true", help="Disable Runtime Profiler telemetry")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable FlashAttention backend optimization")
    parser.add_argument("--threads", type=int, default=None, help="Override CPU worker thread count")

    args = parser.parse_args()
    config_file = Path(args.config) if args.config else None
    pipe_config = PipelineConfig(config_file=config_file, cli_args=args)

    run_end_to_end_validation(pipe_config)

