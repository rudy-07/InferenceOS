"""
optimization_advisor.py
------------------------
Optimization recommendation engine for Phase 9 Runtime Profiler.

Analyzes ProfilerTelemetry to discover bottlenecks (GPU idle, PCIe transfer
saturation, CPU thread starvation, VRAM headroom wastage, KV cache growth) and
generates concrete, quantitative recommendations with estimated performance impact.

Example Output
--------------
  Bottleneck:  GPU Idle 27% (waiting on CPU boundary transfer)
  Action:      Move Layer 61 to GPU
  Impact:      +9.2% tok/s
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from .telemetry_collector import ProfilerTelemetry


# ---------------------------------------------------------------------------
# Severity Level
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Impact severity of a discovered bottleneck."""
    CRITICAL = auto()  # > 25% performance degradation or OOM risk
    WARNING  = auto()  # 10%–25% degradation
    INFO     = auto()  # < 10% degradation / tuning hint


# ---------------------------------------------------------------------------
# OptimizationRecommendation
# ---------------------------------------------------------------------------

@dataclass
class OptimizationRecommendation:
    """
    Actionable recommendation produced by the advisor.

    Attributes
    ----------
    title : str
        Short title of the recommendation (e.g. "Move Layer 61 to GPU").
    bottleneck : str
        Description of the observed bottleneck (e.g. "GPU idle 27%").
    action : str
        Concrete configuration / flag change required.
    estimated_improvement : str
        Human-readable improvement estimate (e.g. "+9.2% tok/s", "-18ms ITL").
    estimated_tps_delta_pct : float
        Numerical estimated tok/s increase percentage (+9.2).
    severity : Severity
        Impact level (CRITICAL, WARNING, INFO).
    category : str
        Category: "layer_placement", "memory", "prefetch", "quantization", "threads".
    """
    title: str
    bottleneck: str
    action: str
    estimated_improvement: str
    estimated_tps_delta_pct: float
    severity: Severity = Severity.WARNING
    category: str = "layer_placement"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "bottleneck": self.bottleneck,
            "action": self.action,
            "estimated_improvement": self.estimated_improvement,
            "estimated_tps_delta_pct": round(self.estimated_tps_delta_pct, 2),
            "severity": self.severity.name,
            "category": self.category,
        }

    def summary_line(self) -> str:
        return f"[{self.severity.name}] {self.bottleneck} ➔ {self.action} ({self.estimated_improvement})"


# ---------------------------------------------------------------------------
# OptimizationAdvisor
# ---------------------------------------------------------------------------

class OptimizationAdvisor:
    """
    Expert system that evaluates telemetry and produces recommendations.

    Parameters
    ----------
    telemetry : ProfilerTelemetry
        Telemetry snapshot collected by TelemetryCollector.
    hw_profile : dict, optional
        Hardware profile for host limits.
    vram_free_mb : float, optional
        Available VRAM headroom in MB.
    """

    def __init__(
        self,
        telemetry: ProfilerTelemetry,
        hw_profile: Optional[Dict[str, Any]] = None,
        vram_free_mb: float = 0.0,
    ) -> None:
        self.telemetry = telemetry
        self.hw_profile = hw_profile or {}
        self.vram_free_mb = vram_free_mb

    def analyze(self) -> List[OptimizationRecommendation]:
        """
        Run heuristic diagnostic checks against telemetry data.

        Returns a list of prioritized OptimizationRecommendation objects.
        """
        recs: List[OptimizationRecommendation] = []

        self._check_gpu_idle_stalls(recs)
        self._check_pcie_transfer_bottlenecks(recs)
        self._check_vram_headroom(recs)
        self._check_kv_cache_growth(recs)
        self._check_cpu_thread_starvation(recs)
        self._check_prefetch_opportunities(recs)

        # Sort recommendations by estimated TPS impact (highest impact first)
        recs.sort(key=lambda r: r.estimated_tps_delta_pct, reverse=True)
        return recs

    # ---------------------------------------------------------------------------
    # Heuristic Diagnostic Checks
    # ---------------------------------------------------------------------------

    def _check_gpu_idle_stalls(self, recs: List[OptimizationRecommendation]) -> None:
        """Check if GPU is frequently idle while waiting for CPU / PCIe."""
        t = self.telemetry
        if t.n_gpu_layers == 0:
            return

        idle_pct = t.gpu_idle_pct
        if idle_pct >= 15.0:
            # Estimate how many additional layers could be offloaded to GPU
            # Assume ~150 MB per layer for 7B Q4 model
            layer_vram_mb = 150.0
            possible_layers = int(self.vram_free_mb // layer_vram_mb) if self.vram_free_mb > 0 else 1

            if t.n_cpu_layers > 0 and possible_layers > 0:
                target_layer = t.n_gpu_layers
                delta_tps = (idle_pct * 0.35)  # Recover ~35% of idle time into tok/s
                recs.append(
                    OptimizationRecommendation(
                        title=f"Offload Layer {target_layer} to GPU",
                        bottleneck=f"GPU idle {idle_pct:.0f}% (waiting on CPU layer computation)",
                        action=f"Move Layer {target_layer} to GPU (--gpu-layers {t.n_gpu_layers + 1})",
                        estimated_improvement=f"+{delta_tps:.1f}% tok/s",
                        estimated_tps_delta_pct=delta_tps,
                        severity=Severity.CRITICAL if idle_pct >= 25.0 else Severity.WARNING,
                        category="layer_placement",
                    )
                )

    def _check_pcie_transfer_bottlenecks(self, recs: List[OptimizationRecommendation]) -> None:
        """Check if PCIe transfers are saturating bandwidth."""
        t = self.telemetry
        pcie_mb = t.pcie_bytes_transferred / (1024 * 1024)

        if pcie_mb > 100.0 and t.pcie_bandwidth_gbps > 0:
            transfer_ms = (pcie_mb / (t.pcie_bandwidth_gbps * 1024)) * 1000.0
            if transfer_ms > 20.0:
                delta_tps = min(15.0, (transfer_ms / max(1.0, t.generation_eval_ms)) * 100.0)
                recs.append(
                    OptimizationRecommendation(
                        title="Quantize Boundary Layers to Q4_0",
                        bottleneck=f"High PCIe transfer overhead ({pcie_mb:.1f} MB across boundaries)",
                        action="Set boundary layer quantization to Q4_0 or enable pinned buffer",
                        estimated_improvement=f"+{delta_tps:.1f}% tok/s (-{transfer_ms:.0f}ms PCIe delay)",
                        estimated_tps_delta_pct=delta_tps,
                        severity=Severity.WARNING,
                        category="quantization",
                    )
                )

    def _check_vram_headroom(self, recs: List[OptimizationRecommendation]) -> None:
        """Check if excess VRAM is wasted while layers remain on CPU."""
        t = self.telemetry
        if t.n_cpu_layers > 0 and self.vram_free_mb > 1024.0:
            # We have > 1 GB of free VRAM sitting unused!
            layers_to_move = min(t.n_cpu_layers, int(self.vram_free_mb // 250.0))
            if layers_to_move > 0:
                delta_tps = layers_to_move * 3.5  # ~3.5% per layer offloaded
                recs.append(
                    OptimizationRecommendation(
                        title=f"Utilize Unused VRAM ({self.vram_free_mb / 1024:.1f} GB Free)",
                        bottleneck=f"{self.vram_free_mb:.0f} MB VRAM unallocated while {t.n_cpu_layers} layers execute on CPU",
                        action=f"Increase GPU layer offload count to {t.n_gpu_layers + layers_to_move}",
                        estimated_improvement=f"+{delta_tps:.1f}% tok/s",
                        estimated_tps_delta_pct=delta_tps,
                        severity=Severity.WARNING,
                        category="memory",
                    )
                )

    def _check_kv_cache_growth(self, recs: List[OptimizationRecommendation]) -> None:
        """Check KV cache context ceiling and growth rate risks."""
        t = self.telemetry
        if t.context_utilization_pct > 80.0:
            recs.append(
                OptimizationRecommendation(
                    title="Enable KV Cache Quantization (K4_0 / V4_0)",
                    bottleneck=f"High context window usage ({t.context_utilization_pct:.1f}% of {t.context_length})",
                    action="Pass --ctk k4_0 --ctv k4_0 to reduce KV cache memory by 50%",
                    estimated_improvement="-50% KV cache memory size (prevents OOM)",
                    estimated_tps_delta_pct=5.0,
                    severity=Severity.CRITICAL if t.context_utilization_pct > 92.0 else Severity.WARNING,
                    category="memory",
                )
            )

    def _check_cpu_thread_starvation(self, recs: List[OptimizationRecommendation]) -> None:
        """Check if CPU utilization is high or thread count sub-optimal."""
        t = self.telemetry
        phys_cores = self.hw_profile.get("cpu", {}).get("physical_cores", 8)
        if t.avg_cpu_util_pct > 95.0 and t.n_cpu_layers > 0:
            recs.append(
                OptimizationRecommendation(
                    title="Restrict llama.cpp Thread Pool",
                    bottleneck=f"CPU utilization saturated at {t.avg_cpu_util_pct:.0f}%",
                    action=f"Set --threads {max(1, phys_cores // 2)} to leave cores for OS and async transfers",
                    estimated_improvement="+4.5% tok/s (reduces core contention)",
                    estimated_tps_delta_pct=4.5,
                    severity=Severity.INFO,
                    category="threads",
                )
            )

    def _check_prefetch_opportunities(self, recs: List[OptimizationRecommendation]) -> None:
        """Check if async prefetching could hide transfer latency."""
        t = self.telemetry
        if t.n_gpu_layers > 0 and t.n_cpu_layers > 0 and t.pcie_bytes_transferred > 0:
            recs.append(
                OptimizationRecommendation(
                    title="Enable Phase 8 Async Prefetching",
                    bottleneck="Synchronous layer boundary transfer stalls GPU execution",
                    action="Set enable_async_scheduler=True and prefetch_lookahead=1",
                    estimated_improvement="+8.0% tok/s (overlaps transfer with compute)",
                    estimated_tps_delta_pct=8.0,
                    severity=Severity.INFO,
                    category="prefetch",
                )
            )
