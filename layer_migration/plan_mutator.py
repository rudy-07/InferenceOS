"""
plan_mutator.py
---------------
Fast greedy PlacementPlan mutations for real-time layer migration.

Unlike the Phase 3 optimizer (which runs Simulated Annealing — too slow for
live migration decisions), PlanMutator uses O(n) greedy reassignment:

  migrate_layers_to_cpu(plan, n)
      Move the last ``n`` GPU transformer layers to CPU.
      Always produces a valid plan. Never modifies the original (immutable).

  restore_layers_to_gpu(plan, n, vram_available_bytes)
      Move up to ``n`` CPU transformer layers back to GPU, respecting the
      available VRAM budget. Layers are restored front-to-back.

  clamp_plan(plan, n_gpu_layers)
      Force an exact GPU layer count. Used for OOM emergency resets.

Key invariants preserved after any mutation
-------------------------------------------
  - layer_placements list is sorted by layer_index
  - n_gpu_layers + n_cpu_layers == total transformer layers
  - segments are rebuilt from scratch (via build_segments)
  - estimated_vram_bytes and estimated_ram_bytes are updated
  - layer_placement.device and .gpu_index are consistent
  - The original plan is never modified (copy-on-write)
"""
from __future__ import annotations

import copy
from typing import List, Optional

from layer_placement.placement_plan import (
    LayerPlacement,
    PlacementDevice,
    PlacementPlan,
    build_segments,
)


class PlanMutator:
    """
    Applies fast greedy layer reassignments to a :class:`PlacementPlan`.

    All methods return a *new* PlacementPlan; the original is never mutated.

    Parameters
    ----------
    min_gpu_layers : int
        Floor on GPU layer count. Mutations will not reduce GPU layers
        below this value. Default 0.
    gpu_index : int
        Target GPU device index for restored GPU layers. Default 0.
    """

    def __init__(self, min_gpu_layers: int = 0, gpu_index: int = 0) -> None:
        self._min_gpu = max(0, min_gpu_layers)
        self._gpu_index = gpu_index

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def migrate_layers_to_cpu(
        self,
        plan: PlacementPlan,
        n_layers: int,
    ) -> PlacementPlan:
        """
        Move the last ``n_layers`` GPU transformer layers to CPU.

        The "last" GPU layers are chosen because removing them from the GPU
        tail minimises boundary crossings (llama.cpp processes layers
        sequentially, so contiguous GPU runs are cheapest).

        Parameters
        ----------
        plan : PlacementPlan
            Current active plan. Not modified.
        n_layers : int
            Number of GPU transformer layers to move to CPU.
            Clamped to ``(plan.n_gpu_layers - min_gpu_layers)``.

        Returns
        -------
        PlacementPlan
            New plan with ``n_layers`` fewer GPU layers.
        """
        if n_layers <= 0:
            return copy.deepcopy(plan)

        current_gpu = plan.n_gpu_layers
        effective_n = min(n_layers, max(0, current_gpu - self._min_gpu))
        if effective_n <= 0:
            return copy.deepcopy(plan)

        new_placements = self._copy_placements(plan.layer_placements)

        # Find GPU transformer layers from the tail and demote them to CPU
        demoted = 0
        for lp in reversed(new_placements):
            if demoted >= effective_n:
                break
            if lp.layer_type == "transformer" and lp.device == PlacementDevice.GPU:
                lp.device = PlacementDevice.CPU
                lp.gpu_index = None
                demoted += 1

        return self._rebuild_plan(plan, new_placements)

    def restore_layers_to_gpu(
        self,
        plan: PlacementPlan,
        n_layers: int,
        vram_available_bytes: int = 0,
    ) -> PlacementPlan:
        """
        Move up to ``n_layers`` CPU transformer layers back to GPU.

        Layers are restored front-to-back (lowest index first) to maximise
        contiguous GPU segments and minimise boundary crossings.

        Parameters
        ----------
        plan : PlacementPlan
            Current active plan. Not modified.
        n_layers : int
            Target number of CPU layers to restore to GPU.
        vram_available_bytes : int
            Available VRAM headroom. Restoration stops when this budget
            would be exceeded. 0 = no budget constraint (restore freely).

        Returns
        -------
        PlacementPlan
            New plan with up to ``n_layers`` more GPU layers.
        """
        if n_layers <= 0:
            return copy.deepcopy(plan)

        new_placements = self._copy_placements(plan.layer_placements)
        restored = 0
        vram_consumed = 0

        for lp in new_placements:
            if restored >= n_layers:
                break
            if lp.layer_type != "transformer" or lp.device != PlacementDevice.CPU:
                continue
            # Check VRAM budget
            if vram_available_bytes > 0:
                if vram_consumed + lp.size_bytes > vram_available_bytes:
                    break  # Would exceed available VRAM
                vram_consumed += lp.size_bytes

            lp.device = PlacementDevice.GPU
            lp.gpu_index = self._gpu_index
            restored += 1

        return self._rebuild_plan(plan, new_placements)

    def clamp_plan(
        self,
        plan: PlacementPlan,
        n_gpu_layers: int,
    ) -> PlacementPlan:
        """
        Force the plan to have exactly ``n_gpu_layers`` GPU transformer layers.

        Used for OOM emergency resets where a specific target is needed.
        If ``n_gpu_layers`` is less than the current count, excess GPU layers
        are moved to CPU (from the tail). If greater, CPU layers are moved to
        GPU (from the front, unconstrained by VRAM budget).

        Parameters
        ----------
        plan : PlacementPlan
            Current active plan. Not modified.
        n_gpu_layers : int
            Exact target GPU layer count (clamped to valid range).

        Returns
        -------
        PlacementPlan
            New plan with exactly ``n_gpu_layers`` GPU transformer layers.
        """
        target = max(self._min_gpu, min(n_gpu_layers, plan.n_gpu_layers + plan.n_cpu_layers))
        current = plan.n_gpu_layers

        if target == current:
            return copy.deepcopy(plan)
        elif target < current:
            return self.migrate_layers_to_cpu(plan, current - target)
        else:
            return self.restore_layers_to_gpu(plan, target - current, vram_available_bytes=0)

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _copy_placements(placements: List[LayerPlacement]) -> List[LayerPlacement]:
        """Deep-copy the placement list so the original is never mutated."""
        return [
            LayerPlacement(
                layer_index=lp.layer_index,
                layer_type=lp.layer_type,
                device=lp.device,
                gpu_index=lp.gpu_index,
                size_bytes=lp.size_bytes,
                cost=copy.copy(lp.cost),
            )
            for lp in placements
        ]

    @staticmethod
    def _rebuild_plan(
        original: PlacementPlan,
        new_placements: List[LayerPlacement],
    ) -> PlacementPlan:
        """
        Reconstruct a PlacementPlan from a mutated placement list.

        Recomputes:
          - n_gpu_layers, n_cpu_layers
          - gpu_layer_indices, cpu_layer_indices
          - segments (via build_segments)
          - estimated_vram_bytes, estimated_ram_bytes
          - estimated_peak_bytes
        """
        gpu_layers = [
            lp for lp in new_placements
            if lp.device == PlacementDevice.GPU and lp.layer_type == "transformer"
        ]
        cpu_layers = [
            lp for lp in new_placements
            if lp.device == PlacementDevice.CPU and lp.layer_type == "transformer"
        ]

        gpu_indices = [lp.layer_index for lp in gpu_layers]
        cpu_indices = [lp.layer_index for lp in cpu_layers]

        vram_bytes = sum(lp.size_bytes for lp in new_placements if lp.device == PlacementDevice.GPU)
        ram_bytes = sum(lp.size_bytes for lp in new_placements if lp.device == PlacementDevice.CPU)
        peak_bytes = vram_bytes + ram_bytes

        # Preserve KV cache estimates from original (unchanged by layer migration)
        # KV cache is rebuilt by llama.cpp from context, not from weights
        kv_adjustment = original.estimated_peak_bytes - (
            original.estimated_vram_bytes + original.estimated_ram_bytes
        )
        if kv_adjustment < 0:
            kv_adjustment = 0

        new_segments = build_segments(new_placements)

        return PlacementPlan(
            model_name=original.model_name,
            architecture=original.architecture,
            total_layers=original.total_layers,
            n_gpu_layers=len(gpu_layers),
            n_cpu_layers=len(cpu_layers),
            gpu_layer_indices=gpu_indices,
            cpu_layer_indices=cpu_indices,
            layer_placements=new_placements,
            segments=new_segments,
            total_cost=original.total_cost,       # Preserved (not re-optimized)
            optimizer_iterations=original.optimizer_iterations,
            estimated_vram_bytes=vram_bytes,
            estimated_ram_bytes=ram_bytes,
            estimated_peak_bytes=peak_bytes + kv_adjustment,
            context_length=original.context_length,
            is_feasible=True,                     # After migration, plan is always feasible
            warnings=list(original.warnings),
        )
