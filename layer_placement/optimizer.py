"""
optimizer.py
------------
Non-greedy layer placement optimizer for Phase 3 of InferenceOS.

Uses a two-phase approach to avoid the greedy all-or-nothing trap:

Phase A — Capacity-Aware Initialization
    Fills GPU from both ends of the transformer stack simultaneously
    (embedding + lm_head on GPU first, then transformer blocks inward),
    stopping when VRAM headroom is exhausted. Produces a valid but potentially
    suboptimal starting solution.

Phase B — Simulated Annealing Refinement
    Iteratively proposes single-layer swaps (GPU→CPU or CPU→GPU). A swap is
    accepted if it reduces total plan cost, or with decreasing probability for
    slightly worse moves (temperature schedule). Runs for max_iterations or
    until convergence. Deterministic given a fixed random seed.

Why simulated annealing over pure greedy:
    A greedy packing may fit exactly 48/80 layers and discard 32 to CPU even
    though moving 2 transformer layers from GPU to CPU frees just enough room
    to pull back a critical shared layer (e.g. lm_head) onto GPU, reducing
    boundary crossings and lowering total cost. SA explores this trade-space.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .cost_model import CostModel, CostWeights, HardwareContext
from .model_descriptor import LayerDescriptor, LAYER_TYPE_TRANSFORMER
from .placement_plan import (
    LayerPlacement,
    PlacementDevice,
    PlacementPlan,
    PlacementSegment,
    build_segments,
)


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """
    Hyperparameters controlling the simulated annealing optimizer.

    Attributes
    ----------
    max_iterations : int
        Maximum number of SA refinement iterations. Default 500.
    initial_temperature : float
        Starting temperature for acceptance probability. Default 1.0.
    cooling_rate : float
        Multiplicative cooling factor per iteration (< 1.0). Default 0.995.
    random_seed : int
        Fixed seed for reproducibility. Default 42.
    cost_weights : CostWeights
        Objective function weights. See :class:`CostWeights`.
    vram_safety_margin : float
        Fraction of VRAM to leave as headroom (prevents 100% utilisation).
        Default 0.05 (5%).
    ram_safety_margin : float
        Fraction of RAM to leave as headroom.
        Default 0.10 (10%).
    prefer_gpu_for_shared_layers : bool
        If True, embedding and lm_head layers are anchored to GPU when any
        VRAM is available. These layers are accessed every token and benefit
        most from high-bandwidth VRAM. Default True.
    """
    max_iterations: int = 500
    initial_temperature: float = 1.0
    cooling_rate: float = 0.995
    random_seed: int = 42
    cost_weights: CostWeights = field(default_factory=CostWeights)
    vram_safety_margin: float = 0.05
    ram_safety_margin: float = 0.10
    prefer_gpu_for_shared_layers: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_bytes(
    layers: List[LayerDescriptor],
    assignment: List[PlacementDevice],
    device: PlacementDevice,
    context_length: int = 4096,
) -> int:
    """
    Count total bytes committed to the given device tier (weights + KV cache).

    Parameters
    ----------
    layers : List[LayerDescriptor]
        All model layers.
    assignment : List[PlacementDevice]
        Parallel device assignment.
    device : PlacementDevice
        Target device to tally.
    context_length : int
        Context window length for KV cache calculation.

    Returns
    -------
    int
        Total bytes on the device (weight bytes + KV cache bytes for
        transformer layers on that device).
    """
    weight_bytes = sum(
        l.size_bytes for l, d in zip(layers, assignment) if d == device
    )
    kv_bytes = sum(
        l.kv_cache_bytes_per_token * context_length
        for l, d in zip(layers, assignment)
        if d == device and l.layer_type == LAYER_TYPE_TRANSFORMER
    )
    return weight_bytes + kv_bytes


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

class PlacementOptimizer:
    """
    Two-phase layer placement optimizer.

    Parameters
    ----------
    cost_model : CostModel
        Evaluates candidate placement costs.
    config : OptimizerConfig
        Hyperparameter configuration.
    """

    def __init__(
        self,
        cost_model: CostModel,
        config: Optional[OptimizerConfig] = None,
    ) -> None:
        self.cost_model = cost_model
        self.config = config or OptimizerConfig()
        self._rng = random.Random(self.config.random_seed)

    def optimize(
        self,
        model_name: str,
        architecture: str,
        layers: List[LayerDescriptor],
        vram_available_bytes: int,
        ram_available_bytes: int,
        context_length: int = 4096,
        gpu_index: int = 0,
    ) -> PlacementPlan:
        """
        Compute an optimal layer placement plan.

        Parameters
        ----------
        model_name : str
            Human-readable model identifier for the plan.
        architecture : str
            Architecture family string (e.g. "llama").
        layers : List[LayerDescriptor]
            Ordered list of all model layers (embedding + transformers + lm_head).
        vram_available_bytes : int
            Bytes of VRAM available for allocation (free after OS/backend overhead).
        ram_available_bytes : int
            Bytes of RAM available for allocation.
        context_length : int
            Context window length used for KV cache estimates.
        gpu_index : int
            Primary GPU device index to assign GPU layers to.

        Returns
        -------
        PlacementPlan
            Optimized placement plan with per-layer decisions, segments, costs.
        """
        # Apply safety margins to effective capacities
        cfg = self.config
        effective_vram = int(vram_available_bytes * (1.0 - cfg.vram_safety_margin))
        effective_ram = int(ram_available_bytes * (1.0 - cfg.ram_safety_margin))

        # Phase A: capacity-aware greedy initialization
        assignment = self._phase_a_initialize(
            layers=layers,
            effective_vram=effective_vram,
            effective_ram=effective_ram,
            context_length=context_length,
        )

        # Phase B: simulated annealing refinement
        assignment, iterations_run = self._phase_b_refine(
            layers=layers,
            assignment=assignment,
            effective_vram=effective_vram,
            effective_ram=effective_ram,
            context_length=context_length,
        )

        # Build the result plan
        return self._build_plan(
            model_name=model_name,
            architecture=architecture,
            layers=layers,
            assignment=assignment,
            vram_available_bytes=vram_available_bytes,
            ram_available_bytes=ram_available_bytes,
            context_length=context_length,
            gpu_index=gpu_index,
            iterations_run=iterations_run,
        )

    # ---------------------------------------------------------------------------
    # Phase A: Capacity-Aware Greedy Initialization
    # ---------------------------------------------------------------------------

    def _phase_a_initialize(
        self,
        layers: List[LayerDescriptor],
        effective_vram: int,
        effective_ram: int,
        context_length: int,
    ) -> List[PlacementDevice]:
        """
        Greedy initialization that fills VRAM from both ends simultaneously:
          - Shared layers (embedding, lm_head) anchored to GPU first.
          - Transformer blocks filled from the first index upward.
          - Remaining blocks fall back to CPU.

        This produces a valid starting assignment that the SA phase can then
        refine without violating hard memory constraints.
        """
        n = len(layers)
        # Start everything on CPU
        assignment = [PlacementDevice.CPU] * n

        vram_used = 0
        cfg = self.config

        # Step 1: Anchor shared layers to GPU (embedding + lm_head) if possible
        # and if VRAM is available at all.
        if effective_vram > 0:
            for i, layer in enumerate(layers):
                if layer.layer_type != LAYER_TYPE_TRANSFORMER:
                    # Shared layers: prefer GPU if space allows
                    if cfg.prefer_gpu_for_shared_layers and vram_used + layer.size_bytes <= effective_vram:
                        assignment[i] = PlacementDevice.GPU
                        vram_used += layer.size_bytes

        # Step 2: Fill transformer blocks onto GPU from the start
        for i, layer in enumerate(layers):
            if layer.layer_type != LAYER_TYPE_TRANSFORMER:
                continue  # Already handled above
            layer_vram_cost = layer.size_bytes + layer.kv_cache_bytes_per_token * context_length
            if vram_used + layer_vram_cost <= effective_vram:
                assignment[i] = PlacementDevice.GPU
                vram_used += layer_vram_cost

        return assignment

    # ---------------------------------------------------------------------------
    # Phase B: Simulated Annealing Refinement
    # ---------------------------------------------------------------------------

    def _phase_b_refine(
        self,
        layers: List[LayerDescriptor],
        assignment: List[PlacementDevice],
        effective_vram: int,
        effective_ram: int,
        context_length: int,
    ) -> Tuple[List[PlacementDevice], int]:
        """
        Improve the greedy initialization via simulated annealing.

        At each iteration:
        1. Pick a random transformer layer.
        2. Propose flipping it to the opposite device.
        3. Check the flip is capacity-feasible.
        4. Accept unconditionally if cost improves; otherwise accept with
           probability exp(-Δcost / temperature).

        Returns the final assignment and the number of iterations executed.
        """
        cfg = self.config
        temperature = cfg.initial_temperature
        rng = self._rng

        # Only consider swappable layers (transformer blocks with both options valid)
        swappable_indices = [i for i, l in enumerate(layers) if l.layer_type == LAYER_TYPE_TRANSFORMER]

        if not swappable_indices:
            return assignment, 0

        # Current cost baseline
        vram_used = _count_bytes(layers, assignment, PlacementDevice.GPU, context_length)
        ram_used = _count_bytes(layers, assignment, PlacementDevice.CPU, context_length)
        current_cost = self.cost_model.evaluate_plan(
            layers, assignment, vram_used, ram_used, context_length
        )

        best_assignment = list(assignment)
        best_cost = current_cost

        for iteration in range(cfg.max_iterations):
            if temperature < 1e-6:
                break

            # Pick a random transformer layer to swap
            idx = rng.choice(swappable_indices)
            layer = layers[idx]
            current_device = assignment[idx]
            proposed_device = (
                PlacementDevice.CPU if current_device == PlacementDevice.GPU
                else PlacementDevice.GPU
            )

            # Compute layer cost in KV cache terms
            layer_vram_cost = layer.size_bytes + layer.kv_cache_bytes_per_token * context_length

            # Check feasibility of proposed swap
            if proposed_device == PlacementDevice.GPU:
                new_vram = vram_used + layer_vram_cost
                new_ram = ram_used - layer_vram_cost
                if new_vram > effective_vram:
                    temperature *= cfg.cooling_rate
                    continue  # VRAM would overflow — reject
            else:
                new_vram = vram_used - layer_vram_cost
                new_ram = ram_used + layer_vram_cost
                if new_ram > effective_ram:
                    temperature *= cfg.cooling_rate
                    continue  # RAM would overflow — reject

            new_vram = max(0, new_vram)
            new_ram = max(0, new_ram)

            # Tentative assignment
            assignment[idx] = proposed_device
            proposed_cost = self.cost_model.evaluate_plan(
                layers, assignment, new_vram, new_ram, context_length
            )

            delta = proposed_cost - current_cost

            # Acceptance decision
            import math
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                # Accept
                vram_used = new_vram
                ram_used = new_ram
                current_cost = proposed_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = list(assignment)
            else:
                # Reject: revert
                assignment[idx] = current_device

            temperature *= cfg.cooling_rate

        return best_assignment, cfg.max_iterations

    # ---------------------------------------------------------------------------
    # Plan construction
    # ---------------------------------------------------------------------------

    def _build_plan(
        self,
        model_name: str,
        architecture: str,
        layers: List[LayerDescriptor],
        assignment: List[PlacementDevice],
        vram_available_bytes: int,
        ram_available_bytes: int,
        context_length: int,
        gpu_index: int,
        iterations_run: int,
    ) -> PlacementPlan:
        """Assemble the final :class:`PlacementPlan` from the optimized assignment."""

        vram_used = _count_bytes(layers, assignment, PlacementDevice.GPU, context_length)
        ram_used = _count_bytes(layers, assignment, PlacementDevice.CPU, context_length)

        # Per-layer cost breakdowns
        breakdowns = self.cost_model.evaluate_layer_costs(
            layers, assignment, vram_used, ram_used
        )

        # Build LayerPlacement list
        layer_placements: List[LayerPlacement] = []
        gpu_layer_indices: List[int] = []
        cpu_layer_indices: List[int] = []
        n_gpu_transformer = 0
        n_cpu_transformer = 0

        for i, (layer, device, bd) in enumerate(zip(layers, assignment, breakdowns)):
            lp = LayerPlacement(
                layer_index=layer.layer_index,
                layer_type=layer.layer_type,
                device=device,
                gpu_index=gpu_index if device == PlacementDevice.GPU else None,
                size_bytes=layer.size_bytes,
                cost=bd,
            )
            layer_placements.append(lp)

            if device == PlacementDevice.GPU:
                gpu_layer_indices.append(layer.layer_index)
                if layer.layer_type == LAYER_TYPE_TRANSFORMER:
                    n_gpu_transformer += 1
            else:
                cpu_layer_indices.append(layer.layer_index)
                if layer.layer_type == LAYER_TYPE_TRANSFORMER:
                    n_cpu_transformer += 1

        # Segments
        segments = build_segments(layer_placements)

        # Total cost
        total_cost = self.cost_model.evaluate_plan(
            layers, assignment, vram_used, ram_used, context_length
        )

        # Feasibility
        feasible = self.cost_model.is_feasible(vram_used, ram_used)
        warnings: List[str] = []

        vram_util = vram_used / max(self.cost_model.hw.vram_total_bytes, 1)
        ram_util = ram_used / max(self.cost_model.hw.ram_total_bytes, 1)

        if not feasible:
            warnings.append(
                f"Plan exceeds physical memory: VRAM {vram_used/(1024**3):.2f} GB / "
                f"{vram_available_bytes/(1024**3):.2f} GB available, "
                f"RAM {ram_used/(1024**3):.2f} GB / "
                f"{ram_available_bytes/(1024**3):.2f} GB available."
            )
        if vram_util > 0.90:
            warnings.append(f"VRAM utilisation is high ({vram_util*100:.1f}%). Consider reducing context length.")
        if ram_util > 0.85:
            warnings.append(f"RAM utilisation is high ({ram_util*100:.1f}%). System may page under load.")
        if n_gpu_transformer == 0 and vram_available_bytes > 0:
            warnings.append("No transformer layers placed on GPU despite VRAM being available. Model may be too large.")

        # Workspace scratch buffer (from Phase 2 formula)
        workspace_bytes = max(512 * 1024 * 1024, int(
            sum(l.size_bytes for l in layers) * 0.10
        ))
        peak_bytes = vram_used + ram_used + workspace_bytes

        return PlacementPlan(
            model_name=model_name,
            architecture=architecture,
            total_layers=len(layers),
            n_gpu_layers=n_gpu_transformer,
            n_cpu_layers=n_cpu_transformer,
            gpu_layer_indices=gpu_layer_indices,
            cpu_layer_indices=cpu_layer_indices,
            layer_placements=layer_placements,
            segments=segments,
            total_cost=total_cost,
            optimizer_iterations=iterations_run,
            estimated_vram_bytes=vram_used,
            estimated_ram_bytes=ram_used,
            estimated_peak_bytes=peak_bytes,
            context_length=context_length,
            is_feasible=feasible,
            warnings=warnings,
        )
