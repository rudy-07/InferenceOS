"""
01_simple_inference.py
----------------------
Basic single-shot streaming inference example using InferenceOS Python API.
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

from profiler.hardware_profiler import get_system_resources
from layer_placement import PlacementEngine, ModelDescriptor
from inference_runtime import RuntimeConfig


def main() -> None:
    print("InferenceOS Simple Inference Example")

    # 1. Profile System Hardware
    sys_res = get_system_resources()
    hw_profile = sys_res.to_dict()
    print(f"Detected GPUs: {len(hw_profile.get('gpus', []))}")

    # 2. Define Model Path
    model_path = PROJECT_ROOT / "models" / "sample_model.gguf"
    if not model_path.exists():
        print(f"Notice: Model file {model_path} does not exist (placeholder example).")

    # 3. Create Model Descriptor & Placement Plan
    descriptor = ModelDescriptor.from_gguf_metadata(
        metadata={
            "arch": "llama",
            "num_layers": 32,
            "hidden_size": 4096,
            "num_heads": 32,
            "num_kv_heads": 8,
            "max_context_length": 4096,
        },
        model_size_bytes=4 * 1024**3,
        quant_type="Q4_K_M",
        model_name="Llama-3-8B",
    )

    placement_engine = PlacementEngine(hw_profile=hw_profile)
    plan = placement_engine.generate_placement_plan(descriptor, context_length=4096)
    print(f"Layer Placement: {plan.n_gpu_layers} GPU layers, {plan.n_cpu_layers} CPU layers")

    # 4. Configure Session
    config = RuntimeConfig(
        threads=8,
        context_length=4096,
        use_flash_attn=True,
    )

    print("Initialization complete. Ready to run InferenceSession.")


if __name__ == "__main__":
    main()
