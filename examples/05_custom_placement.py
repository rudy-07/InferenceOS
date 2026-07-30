"""
05_custom_placement.py
----------------------
Demonstrates integer linear programming placement and manual layer device overrides.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_placement import PlacementEngine, ModelDescriptor


def main() -> None:
    print("InferenceOS Custom Placement Example\n")

    hw_profile = {
        "gpus": [
            {
                "model": "NVIDIA RTX 3080",
                "vendor": "nvidia",
                "vram_total_mb": 10240,
                "vram_free_mb": 6000,
                "bandwidth": 760.0,
            }
        ],
        "ram": {"total_bytes": 32 * 1024**3, "available_gb": 24.0},
        "interconnects": [{"type": "PCIe", "bandwidth": 32.0}],
    }

    descriptor = ModelDescriptor.from_gguf_metadata(
        metadata={
            "arch": "llama",
            "num_layers": 80,
            "hidden_size": 8192,
            "num_heads": 64,
            "num_kv_heads": 8,
            "max_context_length": 4096,
        },
        model_size_bytes=40 * 1024**3,
        quant_type="Q4_K_M",
        model_name="Llama-3-70B-Instruct",
    )

    engine = PlacementEngine(hw_profile=hw_profile)
    plan = engine.generate_placement_plan(descriptor, context_length=4096)

    print("Computed Placement Plan:")
    print(f"  Model:                {plan.model_name}")
    print(f"  Total Layers:         {plan.total_layers}")
    print(f"  GPU Offloaded Layers: {plan.n_gpu_layers}")
    print(f"  CPU Layers:           {plan.n_cpu_layers}")
    print(f"  Estimated VRAM:       {plan.estimated_vram_bytes / (1024**3):.2f} GB")
    print(f"  Estimated RAM:        {plan.estimated_ram_bytes / (1024**3):.2f} GB")
    print(f"  PCIe Crossings:       {plan.boundary_crossings}")


if __name__ == "__main__":
    main()
