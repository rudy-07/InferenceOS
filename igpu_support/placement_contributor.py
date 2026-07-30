"""
placement_contributor.py
------------------------
iGPU-aware placement plan overlay for Phase 7 Integrated GPU Support.

The IgpuPlacementContributor is the primary entry point for Phase 7. It
orchestrates the full iGPU analysis pipeline and applies an iGPU assignment
overlay on top of an existing GPU/CPU placement plan from Phase 3.

Strategy
--------
Phase 3 already solved the optimal GPU/CPU split via simulated annealing.
Phase 7 does not re-run the optimizer. Instead it:

  1. Profiles the iGPU (IgpuProfiler)
  2. Estimates bus contention (ContentionModel)
  3. Evaluates overall suitability (SuitabilityEvaluator)
  4. If DISABLED → return the plan unchanged
  5. If LIGHT or FULL → identify candidate layers from CPU-placed layers
     that are suitable for iGPU according to WorkloadClassifier
  6. Budget-check each candidate against remaining iGPU VRAM
  7. Reassign qualifying layers from CPU → IGPU in a copy of the plan
  8. Return IgpuPlacementResult with the annotated plan and diagnostics

Why steal from CPU, not GPU?
-----------------------------
The discrete GPU is already optimally used by Phase 3. Taking layers away
from the dGPU would waste dedicated VRAM and TFLOPS. The iGPU is instead
given CPU-bound layers that would otherwise run on the host processor,
potentially saving CPU cycles and reducing RAM bus pressure from CPU compute.

VRAM budget management
-----------------------
The contributor maintains a running VRAM budget and only assigns layers
that fit within `igpu_profile.vram_bytes × _IGPU_BUDGET_FRACTION (0.70)`.
A 30% safety margin is kept for KV cache allocations and OS overhead.

Plan immutability
-----------------
The contributor never modifies the original PlacementPlan. It creates
new LayerPlacement objects with device=IGPU and assembles a new
PlacementPlan with updated field values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from layer_placement.model_descriptor import ModelDescriptor
from layer_placement.placement_plan import (
    LayerPlacement,
    PlacementDevice,
    PlacementPlan,
    build_segments,
)

from .contention_model import ContentionEstimate, ContentionModel
from .igpu_profiler import IgpuProfile, IgpuProfiler
from .suitability_evaluator import SuitabilityEvaluator, SuitabilityResult
from .workload_classifier import IgpuWorkloadSuitability, WorkloadClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum fraction of iGPU VRAM to use (leave 30% for KV + OS)
_IGPU_BUDGET_FRACTION = 0.70


# ---------------------------------------------------------------------------
# IgpuPlacementResult
# ---------------------------------------------------------------------------

@dataclass
class IgpuPlacementResult:
    """
    Result of the iGPU placement contributor.

    Attributes
    ----------
    plan : PlacementPlan
        Updated placement plan with iGPU-assigned layers. The original plan
        is not modified — this is a new immutable copy.
    igpu_layer_indices : List[int]
        Layer indices (within the full layer list) now assigned to the iGPU.
    igpu_vram_used_bytes : int
        Total VRAM bytes committed to the iGPU.
    workload_suitability : str
        The suitability level applied: ``"FULL"``, ``"LIGHT"``, or ``"DISABLED"``.
    igpu_profile : IgpuProfile
        The iGPU profile used for this decision.
    contention : ContentionEstimate
        The bus contention estimate used for this decision.
    suitability_result : SuitabilityResult
        Full suitability evaluation with reasoning.
    warnings : List[str]
        Non-fatal warnings about the iGPU assignment.
    igpu_enabled : bool
        Convenience flag: True when at least one layer was assigned to iGPU.
    """
    plan: PlacementPlan
    igpu_layer_indices: List[int]
    igpu_vram_used_bytes: int
    workload_suitability: str
    igpu_profile: IgpuProfile
    contention: ContentionEstimate
    suitability_result: SuitabilityResult
    warnings: List[str] = field(default_factory=list)

    @property
    def igpu_enabled(self) -> bool:
        """True when at least one layer is assigned to the iGPU."""
        return len(self.igpu_layer_indices) > 0

    def summary(self) -> str:
        """One-line human-readable summary."""
        if not self.igpu_enabled:
            return (f"iGPU: DISABLED ({self.igpu_profile.model}) — "
                    f"{'; '.join(self.suitability_result.reasons[:1])}")
        return (
            f"iGPU: {self.workload_suitability} mode ({self.igpu_profile.model}) — "
            f"{len(self.igpu_layer_indices)} layers, "
            f"{self.igpu_vram_used_bytes // (1024**2)} MB VRAM used"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "igpu_enabled": self.igpu_enabled,
            "workload_suitability": self.workload_suitability,
            "igpu_layer_count": len(self.igpu_layer_indices),
            "igpu_layer_indices": self.igpu_layer_indices,
            "igpu_vram_used_bytes": self.igpu_vram_used_bytes,
            "igpu_vram_used_mb": self.igpu_vram_used_bytes // (1024 * 1024),
            "igpu_profile": self.igpu_profile.to_dict(),
            "contention": self.contention.to_dict(),
            "suitability": self.suitability_result.to_dict(),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# IgpuPlacementContributor
# ---------------------------------------------------------------------------

class IgpuPlacementContributor:
    """
    Computes which layers should run on the iGPU and produces an updated
    :class:`PlacementPlan`.

    This contributor runs *after* the Phase 3 optimizer and applies a
    post-optimization overlay: it only re-assigns CPU-placed layers to the
    iGPU, never touching dGPU-placed layers.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile from Phase 1 profiler.
    """

    def __init__(self, hw_profile: Dict[str, Any]) -> None:
        self.hw_profile = hw_profile

    def contribute(
        self,
        model: ModelDescriptor,
        plan: PlacementPlan,
    ) -> IgpuPlacementResult:
        """
        Apply iGPU placement overlay to the given plan.

        Parameters
        ----------
        model : ModelDescriptor
            Full model descriptor (used to look up layer types).
        plan : PlacementPlan
            Existing GPU/CPU placement plan from Phase 3.

        Returns
        -------
        IgpuPlacementResult
            Updated plan + diagnostics.  ``result.plan`` is the new plan;
            ``result.igpu_enabled`` indicates whether any iGPU work was assigned.
        """
        # Step 1: Profile the iGPU
        igpu_profile = IgpuProfiler(self.hw_profile).profile()

        # Step 2: Estimate bus contention
        contention = ContentionModel(self.hw_profile).estimate(igpu_profile)

        # Step 3: Evaluate suitability
        suitability_result = SuitabilityEvaluator().evaluate(igpu_profile, contention)
        suitability = suitability_result.suitability

        # Step 4: If DISABLED, return plan unchanged
        if suitability == IgpuWorkloadSuitability.DISABLED:
            return IgpuPlacementResult(
                plan=plan,
                igpu_layer_indices=[],
                igpu_vram_used_bytes=0,
                workload_suitability=str(suitability),
                igpu_profile=igpu_profile,
                contention=contention,
                suitability_result=suitability_result,
                warnings=igpu_profile.warnings + suitability_result.reasons,
            )

        # Step 5: Identify candidate layers (CPU-placed) and budget-check them
        classifier = WorkloadClassifier(
            suitability=suitability,
            igpu_vram_bytes=igpu_profile.vram_bytes,
            effective_bw_gbps=contention.effective_igpu_bandwidth_gbps,
        )

        igpu_budget_bytes = int(igpu_profile.vram_bytes * _IGPU_BUDGET_FRACTION)
        igpu_used_bytes = 0
        igpu_layer_indices: List[int] = []
        new_placements: List[LayerPlacement] = []
        warnings: List[str] = list(igpu_profile.warnings)

        for lp in plan.layer_placements:
            # Only steal from CPU-placed layers
            if lp.device != PlacementDevice.CPU:
                new_placements.append(lp)
                continue

            # Check workload suitability and VRAM budget
            fits_in_budget = (igpu_used_bytes + lp.size_bytes) <= igpu_budget_bytes
            is_suitable = classifier.is_suitable_for_igpu(lp.layer_type, lp.size_bytes)

            if is_suitable and fits_in_budget:
                # Reassign to iGPU
                new_lp = LayerPlacement(
                    layer_index=lp.layer_index,
                    layer_type=lp.layer_type,
                    device=PlacementDevice.IGPU,
                    gpu_index=igpu_profile.vulkan_device_index,
                    size_bytes=lp.size_bytes,
                    cost=lp.cost,
                )
                new_placements.append(new_lp)
                igpu_layer_indices.append(lp.layer_index)
                igpu_used_bytes += lp.size_bytes
            else:
                new_placements.append(lp)

        # Step 6: Warn if nothing was assigned (suitability said LIGHT/FULL but
        # no layers actually passed the budget/type checks)
        if not igpu_layer_indices:
            warnings.append(
                f"iGPU suitability is {suitability} but no CPU-placed layers "
                "passed the VRAM budget or workload type filters. "
                "All layers remain on CPU."
            )

        # Step 7: Rebuild plan with iGPU assignments
        updated_plan = self._rebuild_plan(plan, new_placements, igpu_layer_indices, igpu_used_bytes)

        return IgpuPlacementResult(
            plan=updated_plan,
            igpu_layer_indices=igpu_layer_indices,
            igpu_vram_used_bytes=igpu_used_bytes,
            workload_suitability=str(suitability),
            igpu_profile=igpu_profile,
            contention=contention,
            suitability_result=suitability_result,
            warnings=warnings,
        )

    # Alias
    contribute_plan = contribute

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_plan(
        self,
        original: PlacementPlan,
        new_placements: List[LayerPlacement],
        igpu_indices: List[int],
        igpu_vram_bytes: int,
    ) -> PlacementPlan:
        """Build a new PlacementPlan with iGPU placements incorporated."""
        # Count layers by device
        n_gpu  = sum(1 for lp in new_placements if lp.device == PlacementDevice.GPU
                     and lp.layer_type == "transformer")
        n_igpu = sum(1 for lp in new_placements if lp.device == PlacementDevice.IGPU)
        n_cpu  = sum(1 for lp in new_placements if lp.device == PlacementDevice.CPU
                     and lp.layer_type == "transformer")

        # Rebuild segments (contiguous device runs)
        new_segments = build_segments(new_placements)

        # Build new plan — use dataclass copy approach to preserve all other fields
        new_plan = PlacementPlan(
            model_name=original.model_name,
            architecture=original.architecture,
            total_layers=original.total_layers,
            n_gpu_layers=n_gpu,
            n_cpu_layers=n_cpu,
            n_igpu_layers=n_igpu,
            gpu_layer_indices=[
                lp.layer_index for lp in new_placements
                if lp.device == PlacementDevice.GPU
            ],
            igpu_layer_indices=igpu_indices,
            cpu_layer_indices=[
                lp.layer_index for lp in new_placements
                if lp.device == PlacementDevice.CPU
            ],
            layer_placements=new_placements,
            segments=new_segments,
            total_cost=original.total_cost,
            optimizer_iterations=original.optimizer_iterations,
            estimated_vram_bytes=original.estimated_vram_bytes,
            estimated_igpu_vram_bytes=igpu_vram_bytes,
            estimated_ram_bytes=original.estimated_ram_bytes,
            estimated_peak_bytes=original.estimated_peak_bytes + igpu_vram_bytes,
            context_length=original.context_length,
            is_feasible=original.is_feasible,
            warnings=original.warnings,
        )
        return new_plan


def apply_igpu_placement(
    model: ModelDescriptor,
    plan: PlacementPlan,
    hw_profile: Dict[str, Any],
) -> IgpuPlacementResult:
    """
    Module-level convenience wrapper.

    Parameters
    ----------
    model : ModelDescriptor
        Full model descriptor.
    plan : PlacementPlan
        Existing Phase 3 placement plan.
    hw_profile : dict
        Hardware profile dict.

    Returns
    -------
    IgpuPlacementResult
    """
    return IgpuPlacementContributor(hw_profile).contribute(model, plan)
