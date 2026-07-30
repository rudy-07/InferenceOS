"""
policies.py
-----------
Modular memory budget policy implementations for InferenceOS.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


class BalancedBudgetPolicy:
    name = "Balanced"
    reserve_ratio = 0.15


class ConservativeBudgetPolicy:
    name = "Conservative"
    reserve_ratio = 0.25


class AggressiveBudgetPolicy:
    name = "Aggressive"
    reserve_ratio = 0.08


class AdaptiveBudgetPolicy:
    name = "Adaptive"
    reserve_ratio = 0.15


class ServerBudgetPolicy:
    name = "Server"
    reserve_ratio = 0.12


class LowMemoryBudgetPolicy:
    name = "Low Memory"
    reserve_ratio = 0.30
