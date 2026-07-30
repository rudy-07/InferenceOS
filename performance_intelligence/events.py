"""
events.py
---------
Performance events for Performance Intelligence Engine in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PerformanceEvent:
    event_type: str = "PerformanceEvent"
    message: str = ""
    timestamp: float = field(default_factory=time.time)
