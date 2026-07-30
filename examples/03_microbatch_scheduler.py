"""
03_microbatch_scheduler.py
--------------------------
Demonstrates how the Dynamic Microbatch Scheduler evaluates VRAM headroom and candidate scoring matrices.
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

from scheduler import MicrobatchScheduler, SchedulerConfig


def main() -> None:
    print("InferenceOS Dynamic Microbatch Scheduler Example\n")

    # Configure scheduler with verbose CLI output
    scheduler = MicrobatchScheduler(config=SchedulerConfig(verbose=True, safety_margin=0.15))

    # Evaluate candidate batch sizes for a 32-layer 7B model
    decision = scheduler.select_microbatch(
        model_metadata={"hidden_size": 4096, "num_layers": 32},
        context_length=4096,
        vram_free_mb=8192.0,
        ram_free_mb=24576.0,
        gpu_utilization=15.0,
    )

    print("\n--- Final Scheduling Decision ---")
    print(f"Selected Microbatch Size: {decision.microbatch}")
    print(f"Decision Confidence:      {decision.confidence:.2%}")
    print(f"Estimated Prefill VRAM:   {decision.estimated_vram_gb:.2f} GB")
    print("Reasoning:")
    for r in decision.reasoning:
        print(f"  • {r}")


if __name__ == "__main__":
    main()
