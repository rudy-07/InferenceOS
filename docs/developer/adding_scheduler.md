# Tutorial: Adding a Custom Scheduler

This tutorial explains how to write and register a custom scheduler in InferenceOS.

---

## Step 1: Define Your Scheduler Class

Create a new file under `scheduler/custom_scheduler.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class CustomSchedulingDecision:
    setting_value: int
    confidence: float
    reason: str

class CustomScheduler:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def evaluate(self, model_metadata: Dict[str, Any], hw_profile: Dict[str, Any]) -> CustomSchedulingDecision:
        # Your custom heuristic logic here
        return CustomSchedulingDecision(
            setting_value=4,
            confidence=0.90,
            reason="Custom heuristic satisfied."
        )
```

---

## Step 2: Register in `InferenceSession`

Open `inference_runtime/inference_session.py`:
1. Import your custom scheduler.
2. Instantiate it in `InferenceSession.__init__()`.
3. Invoke it inside `InferenceSession.run()` before process launch.

---

## Step 3: Add Unit Tests

Create `tests/test_custom_scheduler.py` to test your heuristics under mock hardware profiles.
