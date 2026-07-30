"""
profiles.py
-----------
Optimization profile manager for Automatic Performance Optimizer in InferenceOS.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .interfaces import HardwareFingerprintData, ModelFingerprintData, OptimizationProfile
from .storage import OptimizationStorageEngine

logger = logging.getLogger("InferenceOS.APO.ProfileManager")


class ProfileManager:
    """
    Manages loading, saving, versioning, and invalidation of optimization profiles.
    """

    def __init__(self, storage: Optional[OptimizationStorageEngine] = None) -> None:
        self.storage = storage or OptimizationStorageEngine()

    def get_valid_profile(
        self,
        model_fp: ModelFingerprintData,
        hw_fp: HardwareFingerprintData,
    ) -> Optional[OptimizationProfile]:
        """
        Fetch valid optimization profile for (model, hardware) combination.
        """
        prof = self.storage.load_profile(model_fp.get_fingerprint_hash(), hw_fp.gpu_name)
        if prof and prof.is_valid:
            return prof
        return None

    def save_profile(self, profile: OptimizationProfile) -> None:
        self.storage.save_profile(profile)

    def invalidate_profile(self, model_hash: str, gpu_name: str) -> None:
        prof = self.storage.load_profile(model_hash, gpu_name)
        if prof:
            prof.is_valid = False
            self.storage.save_profile(prof)

    def list_profiles(self) -> List[OptimizationProfile]:
        return self.storage.fetch_all_profiles()
