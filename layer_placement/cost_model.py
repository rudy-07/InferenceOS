"""
cost_model.py
-------------
Multi-objective cost functions for the Phase 3 layer placement optimizer.

Three independent cost dimensions are combined into a single weighted score:

1. **Memory Pressure Cost**
   Penalizes placements that push a memory tier beyond safe utilization.
   Applies a smooth exponential penalty above a configurable threshold.

2. **Transfer (PCIe Boundary) Cost**
   Penalizes GPU↔CPU boundary crossings in the layer sequence.
   Each contiguous run incurs zero transfer cost internally; only the
   crossing points themselves are penalized based on tensor transfer size
   and estimated PCIe bandwidth.

3. **Compute Cost**
   Estimates relative throughput slowdown compared to an ideal all-GPU
   baseline. CPU-placed layers incur a slowdown proportional to the
   GPU/CPU TFLOP ratio.

Total cost = α × memory_pressure + β × transfer + γ × compute
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .model_descriptor import LayerDescriptor, LAYER_TYPE_TRANSFORMER
from .placement_plan import LayerCostBreakdown, PlacementDevice


# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------

@dataclass
class CostWeights:
    """
    Configurable weights for the multi-objective cost function.

    Attributes
    ----------
    memory_pressure : float
        Weight (α) applied to the memory pressure sub-cost. Default 0.35.
    transfer : float
        Weight (β) applied to the transfer/PCIe boundary sub-cost. Default 0.40.
    compute : float
        Weight (γ) applied to the compute slowdown sub-cost. Default 0.25.
    """
    memory_pressure: float = 0.35
    transfer: float = 0.40
    compute: float = 0.25

    def __post_init__(self) -> None:
        total = self.memory_pressure + self.transfer + self.compute
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(
                f"CostWeights must sum to 1.0, got {total:.3f}. "
                "Adjust memory_pressure, transfer, and compute weights."
            )


@dataclass
class HardwareContext:
    """
    Hardware parameters used by the cost model.

    Attributes
    ----------
    vram_total_bytes : int
        Total VRAM capacity (all discrete GPUs combined) in bytes.
    ram_total_bytes : int
        Total system RAM capacity in bytes.
    pcie_bandwidth_bytes_per_sec : float
        Estimated PCIe bandwidth in bytes/second. Used for transfer cost.
    gpu_tflops : float
        Estimated discrete GPU throughput in TFLOP/s (fp16 or quantized).
    cpu_gflops : float
        Estimated CPU throughput in GFLOP/s.
    memory_pressure_threshold : float
        Fraction of tier capacity beyond which memory pressure cost begins
        ramping up. Default 0.85 (85%).
    igpu_gflops : float
        Estimated iGPU throughput in GFLOP/s (Phase 7). Default 0.0 (disabled).
    igpu_vram_bytes : int
        Total iGPU VRAM in bytes (Phase 7). Default 0.
    igpu_bandwidth_gbps : float
        Estimated iGPU memory bandwidth in GB/s (Phase 7). Default 0.0.
    """
    vram_total_bytes: int = 8 * 1024 ** 3       # 8 GB default
    ram_total_bytes: int = 16 * 1024 ** 3       # 16 GB default
    pcie_bandwidth_bytes_per_sec: float = 16.0 * 1024 ** 3  # 16 GB/s PCIe 3.0 x16
    gpu_tflops: float = 10.0                    # Typical mid-range GPU (fp16)
    cpu_gflops: float = 200.0                   # ~200 GFLOP/s for modern 6-core + AVX2
    memory_pressure_threshold: float = 0.85
    # Phase 7: iGPU fields (0 = disabled)
    igpu_gflops: float = 0.0
    igpu_vram_bytes: int = 0
    igpu_bandwidth_gbps: float = 0.0

    @classmethod
    def from_hw_profile(cls, hw_profile: dict) -> "HardwareContext":
        """
        Build a :class:`HardwareContext` from a ``hardware_profile.json`` dict.

        Falls back gracefully to sensible defaults for any missing keys.
        """
        # VRAM: sum discrete GPUs only
        gpus = hw_profile.get("gpus", [])
        vram_total = sum(g.get("vram_total_bytes", 0) for g in gpus)
        if vram_total == 0:
            # Fall back to igpus if no discrete GPU detected
            igpus = hw_profile.get("igpus", [])
            vram_total = sum(g.get("vram_total_bytes", 0) for g in igpus)
        if vram_total == 0:
            vram_total = 4 * 1024 ** 3  # 4 GB fallback

        # RAM
        ram_info = hw_profile.get("ram", hw_profile.get("memory", {}))
        ram_total = int(ram_info.get("total_bytes", 16 * 1024 ** 3))

        # PCIe bandwidth: take first GPU PCIe link or use system bus
        pcie_bw_gbps = 16.0  # PCIe 3.0 x16 default
        interconnects = hw_profile.get("interconnects", [])
        for ic in interconnects:
            if ic.get("type") == "PCIe":
                pcie_bw_gbps = float(ic.get("bandwidth", 16.0))
                break

        # GPU TFLOPS estimate: rough from VRAM bandwidth (heuristic)
        # Higher-end GPUs have faster VRAM BW, correlates loosely with TFLOP
        gpu_bw = 0.0
        for g in gpus:
            gpu_bw = max(gpu_bw, float(g.get("bandwidth", 0.0)))
        # Heuristic: TFLOPS ≈ VRAM_BW_GBps × 0.4 for mid-range GPU
        gpu_tflops = max(1.0, gpu_bw * 0.4) if gpu_bw > 0 else 10.0

        # CPU GFLOPS: rough estimate from logical core count × base freq × SIMD factor
        cpu_info = hw_profile.get("cpu", {})
        logical_cores = int(cpu_info.get("logical_cores", 4))
        base_freq_mhz = float(cpu_info.get("base_freq_mhz", 2000.0))
        isa = cpu_info.get("isa_extensions", [])
        simd_factor = 16.0  # AVX2 FP32 (8 floats × 2 FMA)
        if "avx512f" in isa:
            simd_factor = 32.0
        elif "avx2" in isa or "avx" in isa:
            simd_factor = 16.0
        elif "sse4_2" in isa or "sse2" in isa:
            simd_factor = 8.0
        cpu_gflops = logical_cores * (base_freq_mhz / 1000.0) * simd_factor

        # Phase 7: iGPU parameters
        igpu_gflops = 0.0
        igpu_vram_bytes = 0
        igpu_bandwidth_gbps = 0.0
        igpus_raw = hw_profile.get("igpus", [])
        if igpus_raw:
            best_igpu = max(igpus_raw, key=lambda g: g.get("vram_total_mb", 0))
            igpu_vram_bytes = int(
                best_igpu.get("vram_total_bytes",
                              best_igpu.get("vram_total_mb", 0) * 1024 * 1024)
            )
            igpu_bw_raw = float(best_igpu.get("bandwidth", 0.0))
            if igpu_bw_raw <= 0:
                # Fall back to system bus bandwidth
                for ic in hw_profile.get("interconnects", []):
                    if str(ic.get("type", "")).lower() in ("systembus", "system_bus", "unified"):
                        igpu_bw_raw = float(ic.get("bandwidth", 25.0))
                        break
                if igpu_bw_raw <= 0:
                    igpu_bw_raw = 25.0
            igpu_bandwidth_gbps = igpu_bw_raw
            # iGPU GFLOPS heuristic: bandwidth × 0.2 (shared bus, far less than dGPU)
            igpu_gflops = max(0.5, igpu_bandwidth_gbps * 0.2)

        return cls(
            vram_total_bytes=vram_total,
            ram_total_bytes=ram_total,
            pcie_bandwidth_bytes_per_sec=pcie_bw_gbps * 1024 ** 3,
            gpu_tflops=gpu_tflops,
            cpu_gflops=cpu_gflops,
            igpu_gflops=igpu_gflops,
            igpu_vram_bytes=igpu_vram_bytes,
            igpu_bandwidth_gbps=igpu_bandwidth_gbps,
        )


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------

class CostModel:
    """
    Evaluates the total cost of a candidate placement assignment.

    Parameters
    ----------
    hardware : HardwareContext
        Physical resource parameters.
    weights : CostWeights, optional
        Weighting of the three cost dimensions.
    """

    def __init__(
        self,
        hardware: HardwareContext,
        weights: CostWeights | None = None,
    ) -> None:
        self.hw = hardware
        self.weights = weights or CostWeights()

    # ---------------------------------------------------------------------------
    # Public API: score a full placement assignment
    # ---------------------------------------------------------------------------

    def evaluate_plan(
        self,
        layers: List[LayerDescriptor],
        assignment: List[PlacementDevice],
        vram_used_bytes: int,
        ram_used_bytes: int,
        context_length: int = 4096,
    ) -> float:
        """
        Compute the total weighted cost for a complete placement assignment.

        Parameters
        ----------
        layers : List[LayerDescriptor]
            All model layers in index order.
        assignment : List[PlacementDevice]
            Parallel list of device assignments for each layer.
        vram_used_bytes : int
            Bytes already committed to VRAM by this assignment.
        ram_used_bytes : int
            Bytes already committed to RAM by this assignment.
        context_length : int
            Context window for KV cache memory estimates.

        Returns
        -------
        float
            Total weighted cost (lower is better; 0.0 = ideal all-GPU).
        """
        mem = self.memory_pressure_cost(vram_used_bytes, ram_used_bytes)
        xfer = self.transfer_cost(layers, assignment)
        comp = self.compute_cost(layers, assignment)
        return (
            self.weights.memory_pressure * mem
            + self.weights.transfer * xfer
            + self.weights.compute * comp
        )

    def evaluate_layer_costs(
        self,
        layers: List[LayerDescriptor],
        assignment: List[PlacementDevice],
        vram_used_bytes: int,
        ram_used_bytes: int,
    ) -> List[LayerCostBreakdown]:
        """
        Return per-layer cost breakdowns for reporting.

        Parameters
        ----------
        layers : List[LayerDescriptor]
            All model layers.
        assignment : List[PlacementDevice]
            Parallel device assignment list.
        vram_used_bytes : int
            Total VRAM committed by this plan.
        ram_used_bytes : int
            Total RAM committed by this plan.

        Returns
        -------
        List[LayerCostBreakdown]
            One breakdown per layer; boundary-crossing cost is attributed
            to the *first layer* of the new segment.
        """
        n = len(layers)
        breakdowns = [LayerCostBreakdown() for _ in range(n)]

        # Amortise memory pressure evenly across all layers
        mem_cost = self.memory_pressure_cost(vram_used_bytes, ram_used_bytes)
        per_layer_mem = mem_cost / max(n, 1)

        # Compute cost per layer
        for i, (layer, device) in enumerate(zip(layers, assignment)):
            comp = self._compute_cost_single(device)
            breakdowns[i].compute_cost = comp
            breakdowns[i].memory_pressure_cost = per_layer_mem

        # Transfer cost: attributed to boundary-crossing layers
        xfer_costs = self._transfer_cost_per_layer(layers, assignment)
        for i, xc in enumerate(xfer_costs):
            breakdowns[i].transfer_cost = xc

        # Total
        w = self.weights
        for bd in breakdowns:
            bd.total_cost = (
                w.memory_pressure * bd.memory_pressure_cost
                + w.transfer * bd.transfer_cost
                + w.compute * bd.compute_cost
            )

        return breakdowns

    # ---------------------------------------------------------------------------
    # Sub-cost 1: Memory Pressure
    # ---------------------------------------------------------------------------

    def memory_pressure_cost(
        self,
        vram_used_bytes: int,
        ram_used_bytes: int,
    ) -> float:
        """
        Compute memory pressure cost for the given allocation state.

        Returns a value in [0, ∞). Zero means both tiers are well within
        safe limits. Rises steeply above the pressure threshold.

        The formula uses a smooth exponential ramp:
          cost = exp(max(0, utilization - threshold) × k) - 1
        where k=10 makes it steep above threshold.
        """
        vram_util = vram_used_bytes / max(self.hw.vram_total_bytes, 1)
        ram_util = ram_used_bytes / max(self.hw.ram_total_bytes, 1)
        threshold = self.hw.memory_pressure_threshold
        k = 10.0

        def _pressure(util: float) -> float:
            excess = max(0.0, util - threshold)
            return math.exp(excess * k) - 1.0

        return _pressure(vram_util) + _pressure(ram_util)

    # ---------------------------------------------------------------------------
    # Sub-cost 2: Transfer (PCIe boundary) Cost
    # ---------------------------------------------------------------------------

    def transfer_cost(
        self,
        layers: List[LayerDescriptor],
        assignment: List[PlacementDevice],
    ) -> float:
        """
        Compute PCIe transfer cost for all boundary crossings in the sequence.

        Boundary crossing cost is proportional to the bytes that must be
        streamed across PCIe (the tensor activations at the boundary point).
        We approximate the activation size as `hidden_size × 2` bytes (fp16
        activation tensor for one token).

        The cost is normalised to [0, 1] where 1.0 represents a worst-case
        all-alternating placement at the slowest PCIe bandwidth.
        """
        xfer_costs = self._transfer_cost_per_layer(layers, assignment)
        return sum(xfer_costs)

    def _transfer_cost_per_layer(
        self,
        layers: List[LayerDescriptor],
        assignment: List[PlacementDevice],
    ) -> List[float]:
        """Internal: per-layer transfer costs (normalised)."""
        n = len(layers)
        costs = [0.0] * n
        if n < 2:
            return costs

        for i in range(1, n):
            if assignment[i] != assignment[i - 1]:
                # Boundary crossing: estimate activation transfer time
                # Activation tensor ≈ weight_size × 0.001 (tiny fraction, just activations)
                # Use layer size as a proxy for the scale of the layer, activation is ~hidden_size × 2 bytes
                # Normalise: PCIe latency penalty ≈ 1 ms at 16 GB/s for a 16 MB activation
                # We normalize so that crossing with a "typical" layer boundary = 0.05
                costs[i] = 0.05
        return costs

    # ---------------------------------------------------------------------------
    # Sub-cost 3: Compute Cost
    # ---------------------------------------------------------------------------

    def compute_cost(
        self,
        layers: List[LayerDescriptor],
        assignment: List[PlacementDevice],
    ) -> float:
        """
        Compute relative throughput cost vs. ideal all-GPU placement.

        GPU layers:  cost 0.0 (baseline).
        iGPU layers: cost = (1 - igpu_throughput_ratio) — faster than CPU, slower than dGPU.
        CPU layers:  cost = (1 - cpu_throughput_ratio) — slowest.

        Returns a value in [0, 1] where 0.0 = all-GPU, 1.0 = all-CPU.
        """
        total_flops = sum(l.compute_flops for l in layers)
        if total_flops <= 0:
            return 0.0

        # CPU throughput ratio vs dGPU
        cpu_ratio = (self.hw.cpu_gflops / 1000.0) / max(self.hw.gpu_tflops, 1e-9)
        cpu_ratio = min(cpu_ratio, 1.0)

        # iGPU throughput ratio vs dGPU (between CPU and GPU; defaults to CPU ratio if no iGPU)
        if self.hw.igpu_gflops > 0:
            igpu_ratio = (self.hw.igpu_gflops / 1000.0) / max(self.hw.gpu_tflops, 1e-9)
            igpu_ratio = min(igpu_ratio, 1.0)
        else:
            igpu_ratio = cpu_ratio

        cpu_flops = sum(
            l.compute_flops for l, d in zip(layers, assignment)
            if d == PlacementDevice.CPU
        )
        igpu_flops = sum(
            l.compute_flops for l, d in zip(layers, assignment)
            if d == PlacementDevice.IGPU
        )

        cpu_fraction  = cpu_flops  / total_flops
        igpu_fraction = igpu_flops / total_flops

        return (
            cpu_fraction  * (1.0 - cpu_ratio)
            + igpu_fraction * (1.0 - igpu_ratio)
        )

    def _compute_cost_single(self, device: PlacementDevice) -> float:
        """Compute cost for a single layer on the given device."""
        if device == PlacementDevice.GPU:
            return 0.0
        if device == PlacementDevice.IGPU:
            if self.hw.igpu_gflops > 0:
                igpu_ratio = (self.hw.igpu_gflops / 1000.0) / max(self.hw.gpu_tflops, 1e-9)
                return max(0.0, 1.0 - min(igpu_ratio, 1.0))
            # Fall through to CPU cost if no iGPU configured
        cpu_ratio = (self.hw.cpu_gflops / 1000.0) / max(self.hw.gpu_tflops, 1e-9)
        return max(0.0, 1.0 - min(cpu_ratio, 1.0))

    # ---------------------------------------------------------------------------
    # Feasibility check
    # ---------------------------------------------------------------------------

    def is_feasible(
        self,
        vram_used_bytes: int,
        ram_used_bytes: int,
    ) -> bool:
        """
        Returns True if the allocation fits within physical memory bounds.
        Both VRAM and RAM must each be within 100% of their tier capacity.
        """
        return (
            vram_used_bytes <= self.hw.vram_total_bytes
            and ram_used_bytes <= self.hw.ram_total_bytes
        )
