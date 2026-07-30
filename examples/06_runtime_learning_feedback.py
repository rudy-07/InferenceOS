"""
06_runtime_learning_feedback.py
--------------------------------
Demonstrates logging execution telemetry into the local SQLite database and querying predictive recommendations.
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

from runtime_learning import RuntimeLearningEngine


def main() -> None:
    print("InferenceOS Runtime Learning Feedback Example\n")

    engine = RuntimeLearningEngine()

    hw_profile = {
        "gpus": [{"name": "NVIDIA RTX 4090", "vram_total_mb": 24576}],
        "cpu": {"physical_cores": 16},
    }

    model_metadata = {
        "model_name": "Llama-3-8B",
        "num_layers": 32,
        "hidden_size": 4096,
    }

    # Query recommendation based on past runs
    rec = engine.get_recommendation(
        model_metadata=model_metadata,
        hw_profile=hw_profile,
        requested_context=4096,
        backend="cuda",
    )

    print("Runtime Learning Recommendation:")
    print(f"  Recommended Microbatch: {rec.recommended_microbatch}")
    print(f"  Confidence Score:       {rec.confidence:.2%}")
    print(f"  Expected Prompt TPS:    {rec.expected_prompt_tps:.1f} t/s")
    print(f"  Expected Eval TPS:      {rec.expected_eval_tps:.1f} t/s")


if __name__ == "__main__":
    main()
