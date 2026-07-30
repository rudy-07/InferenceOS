"""
placement_engine.py
--------------------
Top-level facade for the Phase 3 Automatic Layer Placement Engine.

Public API (camelCase + snake_case aliases):

    generatePlacementPlan(model, context_length) -> PlacementPlan
    estimatePerformance(plan, tokens_per_second_gpu_baseline) -> dict
    estimateMemoryUsage(model, context_length) -> dict

This module orchestrates the full pipeline:
  1. Reads hardware constraints from Phase 1 profiler or hardware_profile.json
  2. Constructs the cost model with real bandwidth, VRAM, and CPU parameters
  3. Runs the two-phase optimizer to produce a PlacementPlan
  4. Exposes performance and memory estimation APIs for pre-flight checks

This phase COMPUTES plans only. It does NOT execute layer loading or call
any llama.cpp functions. Plans are data structures that future phases consume.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

from .cost_model import CostModel, CostWeights, HardwareContext
from .model_descriptor import ModelDescriptor, LAYER_TYPE_TRANSFORMER
from .optimizer import OptimizerConfig, PlacementOptimizer
from .placement_plan import PlacementPlan
from .report_generator import ReportGenerator, generate_json_report, generate_text_report

# Phase 7: iGPU support (optional — graceful fallback if module unavailable)
try:
    from igpu_support import IgpuPlacementContributor as _IgpuContributor
    _IGPU_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IGPU_AVAILABLE = False
    _IgpuContributor = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# PlacementEngine
# ---------------------------------------------------------------------------

class PlacementEngine:
    """
    Automatic layer placement engine for InferenceOS Phase 3.

    Determines the optimal distribution of model transformer layers between
    GPU VRAM and CPU RAM without manual ``--gpu-layers`` tuning.

    Parameters
    ----------
    hw_profile : dict, optional
        Hardware profile dict (from ``hardware_profile.json`` or
        ``profiler.get_system_resources()``). If None, attempts to load
        from the default ``hardware_profile.json`` in the project root.
    optimizer_config : OptimizerConfig, optional
        Hyperparameters for the simulated annealing optimizer. If None,
        uses sensible production defaults.
    gpu_model_name : str, optional
        Human-readable GPU label for report generation (e.g. "AMD RX 5600M").
        Auto-detected from hw_profile if not supplied.

    Examples
    --------
    >>> from layer_placement import PlacementEngine, ModelDescriptor
    >>> from orchestrator.gguf_parser import read_gguf_metadata
    >>> from pathlib import Path
    >>>
    >>> meta = read_gguf_metadata(Path("models/llama-3-8b-q4_k_m.gguf"))
    >>> model = ModelDescriptor.from_gguf_metadata(
    ...     meta, model_size_bytes=4_500_000_000, quant_type="Q4_K_M",
    ...     model_name="Llama-3-8B-Q4_K_M"
    ... )
    >>> engine = PlacementEngine()
    >>> plan = engine.generatePlacementPlan(model, context_length=4096)
    >>> print(engine.generateTextReport(plan))
    """

    def __init__(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        gpu_model_name: Optional[str] = None,
    ) -> None:
        self.hw_profile = hw_profile or self._load_default_hw_profile()
        self.optimizer_config = optimizer_config or OptimizerConfig()

        # Build hardware context from profile
        self.hardware_context = HardwareContext.from_hw_profile(self.hw_profile)

        # Build cost model
        cost_weights = (
            self.optimizer_config.cost_weights
            if self.optimizer_config.cost_weights is not None
            else CostWeights()
        )
        self.cost_model = CostModel(self.hardware_context, cost_weights)

        # GPU label for reports
        if gpu_model_name is not None:
            self._gpu_label = gpu_model_name
        else:
            self._gpu_label = self._detect_gpu_label()

        # Optimizer instance
        self._optimizer = PlacementOptimizer(self.cost_model, self.optimizer_config)

    # ---------------------------------------------------------------------------
    # Core API 1: generatePlacementPlan / generate_placement_plan
    # ---------------------------------------------------------------------------

    def generatePlacementPlan(
        self,
        model: ModelDescriptor,
        context_length: int = 4096,
        force_gpu_only: bool = False,
        force_cpu_only: bool = False,
        primary_gpu_index: int = 0,
        enable_igpu: bool = True,
    ) -> PlacementPlan:
        """
        Compute an optimized layer placement plan for the given model.

        Parameters
        ----------
        model : ModelDescriptor
            Full model descriptor including per-layer weight and KV cache sizes.
        context_length : int
            Inference context window length. Affects KV cache memory estimates.
            Clipped to model.max_context_length if larger.
        force_gpu_only : bool
            If True, force all layers onto GPU regardless of VRAM capacity.
            The plan will be marked infeasible if it exceeds available VRAM.
        force_cpu_only : bool
            If True, force all layers onto CPU (useful for benchmarking or
            CPU-only deployments).
        primary_gpu_index : int
            Index of the primary GPU to place GPU layers on.
        enable_igpu : bool
            Phase 7: If True, apply the iGPU placement overlay after the base
            optimization. Has no effect when no iGPU is present or
            suitability is DISABLED. Default True.

        Returns
        -------
        PlacementPlan
            Optimized placement plan. Check ``plan.is_feasible`` and
            ``plan.warnings`` before consuming the plan.
            When iGPU layers are assigned, ``plan.n_igpu_layers > 0``.
        """
        # Clip context to model maximum
        context_length = min(context_length, model.max_context_length)

        # Resolve available memory
        vram_available = self._get_vram_available_bytes(primary_gpu_index)
        ram_available = self._get_ram_available_bytes()

        # Apply forcing overrides
        if force_cpu_only:
            vram_available = 0
        if force_gpu_only:
            vram_available = max(vram_available, model.model_size_bytes * 2)

        # Phase 3: base GPU/CPU optimization
        plan = self._optimizer.optimize(
            model_name=model.name,
            architecture=model.architecture,
            layers=model.layers,
            vram_available_bytes=vram_available,
            ram_available_bytes=ram_available,
            context_length=context_length,
            gpu_index=primary_gpu_index,
        )

        # Phase 7: iGPU post-optimization overlay
        if enable_igpu and _IGPU_AVAILABLE and not force_gpu_only and not force_cpu_only:
            try:
                contributor = _IgpuContributor(self.hw_profile)
                result = contributor.contribute(model, plan)
                plan = result.plan
                # Propagate any iGPU warnings into the plan
                if result.warnings:
                    plan.warnings.extend(
                        [w for w in result.warnings if w not in plan.warnings]
                    )
            except Exception as exc:  # pragma: no cover
                # Non-fatal: iGPU overlay failure falls back to original plan
                plan.warnings.append(f"[Phase 7] iGPU overlay skipped: {exc}")

        return plan

    generate_placement_plan = generatePlacementPlan

    # ---------------------------------------------------------------------------
    # Core API 2: estimatePerformance / estimate_performance
    # ---------------------------------------------------------------------------

    def estimatePerformance(
        self,
        plan: PlacementPlan,
        tokens_per_second_gpu_baseline: float = 50.0,
    ) -> Dict[str, Any]:
        """
        Estimate inference throughput metrics for a given placement plan.

        This is a model-based estimate, not a benchmark. It uses hardware
        parameters (GPU TFLOPS, CPU GFLOPS, PCIe bandwidth) to predict
        relative performance vs. an all-GPU baseline.

        Parameters
        ----------
        plan : PlacementPlan
            The placement plan to evaluate.
        tokens_per_second_gpu_baseline : float
            Expected tokens/second for a fully GPU-offloaded run of this model.
            Defaults to 50 tok/s (representative of mid-range GPU inference).

        Returns
        -------
        dict with keys:
            estimated_tokens_per_second : float
            gpu_utilization_ratio : float  (0.0–1.0)
            pcie_transfer_overhead_ms_per_token : float
            compute_score : float  (1.0 = ideal all-GPU)
            bottleneck : str  ("NONE", "VRAM_PRESSURE", "RAM_PRESSURE",
                               "PCIE_BANDWIDTH", "CPU_COMPUTE", "MEMORY_OOM")
        """
        hw = self.hardware_context

        # Compute score: 0.0 = all CPU, 1.0 = all GPU
        gpu_ratio = plan.gpu_offload_ratio
        cpu_throughput_ratio = (hw.cpu_gflops / 1000.0) / max(hw.gpu_tflops, 1e-9)
        cpu_throughput_ratio = min(cpu_throughput_ratio, 1.0)

        # Weighted throughput (GPU layers at full speed, CPU layers at reduced speed)
        effective_compute_ratio = (
            gpu_ratio * 1.0 + (1.0 - gpu_ratio) * cpu_throughput_ratio
        )

        # PCIe penalty per boundary crossing
        # Approximate activation size per token = hidden_size × 2 bytes (fp16)
        # We don't have hidden_size here, so use a fixed representative value
        activation_bytes = 8192 * 2  # 8192-dim fp16 typical hidden size
        xfer_time_sec = (
            plan.boundary_crossings * activation_bytes
            / max(hw.pcie_bandwidth_bytes_per_sec, 1)
        )
        pcie_overhead_ms = xfer_time_sec * 1000.0

        # GPU-only token time at baseline
        baseline_ms_per_token = 1000.0 / max(tokens_per_second_gpu_baseline, 1e-9)

        # Adjusted token time
        adjusted_ms = (
            baseline_ms_per_token / max(effective_compute_ratio, 1e-6)
            + pcie_overhead_ms
        )
        estimated_tps = 1000.0 / max(adjusted_ms, 1e-6)

        # Bottleneck classification
        vram_util = plan.estimated_vram_bytes / max(hw.vram_total_bytes, 1)
        ram_util = plan.estimated_ram_bytes / max(hw.ram_total_bytes, 1)

        if not plan.is_feasible:
            bottleneck = "MEMORY_OOM"
        elif vram_util > 0.92:
            bottleneck = "VRAM_PRESSURE"
        elif ram_util > 0.85:
            bottleneck = "RAM_PRESSURE"
        elif plan.boundary_crossings >= 4:
            bottleneck = "PCIE_BANDWIDTH"
        elif gpu_ratio < 0.5:
            bottleneck = "CPU_COMPUTE"
        else:
            bottleneck = "NONE"

        return {
            "estimated_tokens_per_second": round(estimated_tps, 2),
            "gpu_utilization_ratio": round(gpu_ratio, 4),
            "pcie_transfer_overhead_ms_per_token": round(pcie_overhead_ms, 4),
            "compute_score": round(effective_compute_ratio, 4),
            "bottleneck": bottleneck,
            "boundary_crossings": plan.boundary_crossings,
            "gpu_tflops": round(hw.gpu_tflops, 2),
            "cpu_gflops": round(hw.cpu_gflops, 2),
            "cpu_throughput_ratio": round(cpu_throughput_ratio, 4),
            "is_feasible": plan.is_feasible,
        }

    estimate_performance = estimatePerformance

    # ---------------------------------------------------------------------------
    # Core API 3: estimateMemoryUsage / estimate_memory_usage
    # ---------------------------------------------------------------------------

    def estimateMemoryUsage(
        self,
        model: ModelDescriptor,
        context_length: int = 4096,
    ) -> Dict[str, Any]:
        """
        Estimate total memory usage for a model without computing a placement plan.

        Uses the Phase 2 peak memory formula:
            peak = model_weights + kv_cache(all_layers) + workspace_scratch

        This is useful for a quick pre-flight check before calling
        ``generatePlacementPlan()``.

        Parameters
        ----------
        model : ModelDescriptor
            Model descriptor.
        context_length : int
            Context window for KV cache estimate.

        Returns
        -------
        dict with keys:
            model_weights_bytes, kv_cache_bytes, workspace_bytes,
            total_vram_bytes (ideal all-GPU), total_ram_bytes (all-CPU fallback),
            peak_total_bytes, fits_fully_in_vram, fits_fully_in_ram,
            model_size_gb, kv_cache_gb, peak_total_gb
        """
        context_length = min(context_length, model.max_context_length)

        model_weights_bytes = model.model_size_bytes
        kv_cache_bytes = model.kv_cache_bytes_per_token() * context_length
        workspace_bytes = max(512 * 1024 * 1024, int(model_weights_bytes * 0.10))
        peak_total_bytes = model_weights_bytes + kv_cache_bytes + workspace_bytes

        vram_available = self._get_vram_available_bytes()
        ram_available = self._get_ram_available_bytes()

        _gb = lambda b: round(b / (1024 ** 3), 3)
        _mb = lambda b: round(b / (1024 * 1024), 1)

        return {
            "model_weights_bytes": model_weights_bytes,
            "model_weights_gb": _gb(model_weights_bytes),
            "kv_cache_bytes": kv_cache_bytes,
            "kv_cache_gb": _gb(kv_cache_bytes),
            "workspace_bytes": workspace_bytes,
            "workspace_mb": _mb(workspace_bytes),
            "peak_total_bytes": peak_total_bytes,
            "peak_total_gb": _gb(peak_total_bytes),
            "vram_available_bytes": vram_available,
            "vram_available_gb": _gb(vram_available),
            "ram_available_bytes": ram_available,
            "ram_available_gb": _gb(ram_available),
            "fits_fully_in_vram": peak_total_bytes <= vram_available,
            "fits_fully_in_ram": peak_total_bytes <= ram_available,
            "requires_split": peak_total_bytes > vram_available,
        }

    estimate_memory_usage = estimateMemoryUsage

    # ---------------------------------------------------------------------------
    # Report generation helpers
    # ---------------------------------------------------------------------------

    def generateTextReport(self, plan: PlacementPlan) -> str:
        """Generate a human-readable ASCII text report for a plan."""
        return generate_text_report(plan, gpu_model_name=self._gpu_label)

    generate_text_report = generateTextReport

    def generateJsonReport(self, plan: PlacementPlan, indent: int = 2) -> str:
        """Generate a machine-readable JSON report for a plan."""
        return generate_json_report(plan, indent=indent)

    generate_json_report = generateJsonReport

    def generateReport(self, plan: PlacementPlan, format: str = "text") -> str:
        """
        Generate a report in the requested format.

        Parameters
        ----------
        plan : PlacementPlan
            Placement plan to document.
        format : str
            ``"text"`` (default) or ``"json"``.

        Returns
        -------
        str
            Formatted report string.
        """
        if format == "json":
            return self.generateJsonReport(plan)
        return self.generateTextReport(plan)

    generate_report = generateReport

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _load_default_hw_profile(self) -> Dict[str, Any]:
        """Try to load hardware_profile.json from the project root."""
        candidate = Path(__file__).parent.parent / "hardware_profile.json"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)

        # Try live profiler if file missing
        try:
            from profiler import get_system_resources
            sys_res = get_system_resources()
            return sys_res.to_dict()
        except Exception:
            pass

        # Final fallback: minimal safe defaults
        return {
            "gpus": [{"vram_total_bytes": 4 * 1024 ** 3, "bandwidth": 10.0}],
            "ram": {"total_bytes": 16 * 1024 ** 3},
            "memory": {"total_bytes": 16 * 1024 ** 3, "available_gb": 8.0},
            "cpu": {"logical_cores": 4, "base_freq_mhz": 2000.0, "isa_extensions": ["avx2"]},
            "interconnects": [],
            "inference_hints": {},
        }

    def _get_vram_available_bytes(self, primary_gpu_index: int = 0) -> int:
        """Compute available VRAM in bytes for the target primary GPU, reserving backend overhead buffer."""
        gpus = self.hw_profile.get("gpus", [])
        if not gpus:
            igpus = self.hw_profile.get("igpus", [])
            gpus = igpus

        if not gpus:
            return 0

        target_gpu = gpus[min(primary_gpu_index, len(gpus) - 1)]
        free_mb = target_gpu.get("vram_free_mb", target_gpu.get("vram_total_mb", 0))
        if free_mb == 0:
            free_mb = int(target_gpu.get("vram_total_mb", 0) * 0.85)

        total_free_bytes = int(free_mb * 1024 * 1024)

        # Reserve dynamic overhead (12% of VRAM, bounded between 512 MB and 1024 MB)
        # for CUDA/ROCm/Vulkan backend context, compute graph buffer, KV cache, and workspace headroom
        backend_overhead_bytes = min(1024 * 1024 * 1024, max(512 * 1024 * 1024, int(total_free_bytes * 0.12)))
        return max(0, total_free_bytes - backend_overhead_bytes)

    def _get_ram_available_bytes(self) -> int:
        """Compute available RAM in bytes, reserving OS overhead."""
        ram = self.hw_profile.get("ram", self.hw_profile.get("memory", {}))
        available_gb = float(ram.get("available_gb", 0.0))
        if available_gb > 0:
            available_bytes = int(available_gb * 1024 ** 3)
        else:
            total_bytes = int(ram.get("total_bytes", 16 * 1024 ** 3))
            available_bytes = int(total_bytes * 0.7)  # assume 70% available

        # Reserve 1 GB for OS and other processes
        os_reserve = 1 * 1024 ** 3
        return max(0, available_bytes - os_reserve)

    def _detect_gpu_label(self) -> str:
        """Extract a human-friendly GPU model name from the hw_profile."""
        gpus = self.hw_profile.get("gpus", [])
        if gpus:
            return str(gpus[0].get("model", "GPU"))
        igpus = self.hw_profile.get("igpus", [])
        if igpus:
            return str(igpus[0].get("model", "iGPU"))
        return "GPU"
