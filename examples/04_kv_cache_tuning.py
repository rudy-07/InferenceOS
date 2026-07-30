"""
04_kv_cache_tuning.py
---------------------
Demonstrates KV Cache quantization and attention-sink eviction policy selection.
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

from kv_manager import IntelligentKVManager, KVPolicyConfig


def main() -> None:
    print("InferenceOS KV Cache Tuning Example\n")

    kv_manager = IntelligentKVManager()
    config = KVPolicyConfig(
        enabled=True,
        compression_enabled=True,
        eviction_enabled=True,
        compression_mode="Q8_0",
        eviction_policy="h2o",
    )

    decision = kv_manager.evaluate_kv_state(
        model_metadata={"hidden_size": 4096, "num_layers": 32},
        context_length=16384,
        vram_free_mb=4096.0,
        vram_total_mb=12288.0,
        pressure_level="Medium",
        config_override=config,
    )

    print(f"Selected Strategy: {decision.compression_strategy}")
    print(f"Eviction Action:   {decision.eviction_action}")
    print(f"Estimated KV Size: {decision.current_kv_gb:.2f} GB")
    print(f"Memory Saved:      {decision.memory_saved_mb:.1f} MB")


if __name__ == "__main__":
    main()
