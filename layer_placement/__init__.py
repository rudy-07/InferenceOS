"""
InferenceOS Layer Placement Engine — Phase 3 Public API

Exposes the complete public interface for automatic GPU/CPU layer placement:

    PlacementEngine           — main facade class
    generatePlacementPlan()   — compute optimal placement
    estimatePerformance()     — predict throughput
    estimateMemoryUsage()     — pre-flight memory check
    ModelDescriptor           — model input descriptor
    LayerDescriptor           — per-layer descriptor
    PlacementPlan             — optimization result
    LayerPlacement            — per-layer placement decision
    PlacementSegment          — contiguous same-device layer run
    PlacementDevice           — GPU / CPU enum
    OptimizerConfig           — tuning parameters for the SA optimizer
    CostWeights               — multi-objective cost function weights
    HardwareContext           — hardware parameters consumed by cost model
    generate_text_report()    — ASCII report helper
    generate_json_report()    — JSON report helper
"""
from __future__ import annotations

from .cost_model import CostWeights, HardwareContext
from .model_descriptor import LayerDescriptor, ModelDescriptor, infer_quant_type_from_filename
from .optimizer import OptimizerConfig
from .placement_engine import PlacementEngine
from .placement_plan import (
    LayerCostBreakdown,
    LayerPlacement,
    PlacementDevice,
    PlacementPlan,
    PlacementSegment,
    build_segments,
)
from .report_generator import generate_json_report, generate_text_report

__all__ = [
    # Facade
    "PlacementEngine",
    # Data types
    "ModelDescriptor",
    "LayerDescriptor",
    "PlacementPlan",
    "LayerPlacement",
    "PlacementSegment",
    "PlacementDevice",
    "LayerCostBreakdown",
    # Configuration
    "OptimizerConfig",
    "CostWeights",
    "HardwareContext",
    # Helpers
    "generate_text_report",
    "generate_json_report",
    "build_segments",
    "infer_quant_type_from_filename",
]
