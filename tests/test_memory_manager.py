"""
test_memory_manager.py
----------------------
Unit tests for InferenceOS Phase 2 Unified Memory Manager.

Tests:
  - Allocation in Tier 1 (VRAM) and Tier 2 (RAM)
  - Rejection of Tier 3 (Storage) for active inference memory
  - Fallback allocation when VRAM is exhausted
  - Tensor transfers between VRAM and RAM
  - Tensor deallocation & memory recovery
  - Peak memory estimation & KV growth estimation
  - Memory accounting and fragmentation tracking
"""
from __future__ import annotations

import pytest
from memory_manager import MemoryTier, TensorMetadata, UnifiedMemoryManager


class TestUnifiedMemoryManager:
    def test_allocation_in_preferred_vram_tier(self):
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=8 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        tensor = mm.allocateTensor(
            tensor_id="model.layer.0.weight",
            size_bytes=512 * 1024 * 1024,  # 512 MB
            preferred_location=MemoryTier.TIER_1_VRAM,
        )

        assert tensor.current_location == MemoryTier.TIER_1_VRAM
        assert tensor.is_in_preferred_location
        assert mm.accounting.allocated_vram_bytes == 512 * 1024 * 1024
        assert mm.accounting.allocated_ram_bytes == 0

    def test_allocation_fallback_to_ram_when_vram_full(self):
        # 1 GB VRAM, 16 GB RAM
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=1 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        # Allocate 800 MB in VRAM
        mm.allocateTensor("t1", 800 * 1024 * 1024, preferred_location=MemoryTier.TIER_1_VRAM)

        # Attempt to allocate 500 MB in VRAM -> should fall back to RAM (Tier 2)
        t2 = mm.allocateTensor("t2", 500 * 1024 * 1024, preferred_location=MemoryTier.TIER_1_VRAM)

        assert t2.current_location == MemoryTier.TIER_2_RAM
        assert not t2.is_in_preferred_location
        assert mm.accounting.allocated_ram_bytes == 500 * 1024 * 1024

    def test_rejection_of_storage_tier_for_active_inference(self):
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=8 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        # Attempting to set allowed_locations exclusively to STORAGE must fail
        with pytest.raises(ValueError, match="Disk storage is forbidden"):
            mm.allocateTensor(
                "t_disk",
                100 * 1024 * 1024,
                preferred_location=MemoryTier.FUTURE_TIER_STORAGE,
                allowed_locations=[MemoryTier.FUTURE_TIER_STORAGE],
            )

    def test_tensor_movement_vram_to_ram(self):
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=8 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        t = mm.allocateTensor("t1", 1024 * 1024 * 1024, preferred_location=MemoryTier.TIER_1_VRAM)
        assert t.current_location == MemoryTier.TIER_1_VRAM
        assert mm.accounting.allocated_vram_bytes == 1024 * 1024 * 1024

        # Move to RAM
        moved = mm.moveTensor("t1", MemoryTier.TIER_2_RAM)
        assert moved.current_location == MemoryTier.TIER_2_RAM
        assert mm.accounting.allocated_vram_bytes == 0
        assert mm.accounting.allocated_ram_bytes == 1024 * 1024 * 1024

    def test_rejection_of_tensor_movement_to_storage(self):
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=8 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        mm.allocateTensor("t1", 100 * 1024 * 1024)

        with pytest.raises(ValueError, match="Disk storage is not active inference memory"):
            mm.moveTensor("t1", MemoryTier.FUTURE_TIER_STORAGE)

    def test_free_tensor_reclaims_memory(self):
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=8 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        mm.allocateTensor("t1", 500 * 1024 * 1024)
        assert mm.accounting.allocated_vram_bytes == 500 * 1024 * 1024

        freed = mm.freeTensor("t1")
        assert freed is True
        assert mm.accounting.allocated_vram_bytes == 0
        assert "t1" not in mm.tensors

    def test_estimate_peak_memory(self):
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=8 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        # 4 GB model, 4096 context
        peak = mm.estimatePeakMemory(
            model_size_bytes=4 * 1024 * 1024 * 1024,
            context_len=4096,
            n_layers=32,
            n_heads=32,
            head_dim=128,
        )

        assert peak["model_size_bytes"] == 4 * 1024 * 1024 * 1024
        assert peak["kv_cache_bytes"] > 0
        assert peak["peak_total_bytes"] > peak["model_size_bytes"]
        assert "fits_in_vram" in peak

    def test_estimate_kv_growth(self):
        mm = UnifiedMemoryManager()

        kv_info = mm.estimateKVGrowth(
            context_len=4096,
            n_layers=32,
            n_heads=32,
            head_dim=128,
            element_size_bytes=2,
            batch_size=1,
        )

        # 2 * 32 * 32 * 128 * 2 = 524,288 bytes per token = 512 KB per token
        assert kv_info["bytes_per_token"] == 524288
        assert kv_info["total_kv_bytes"] == 524288 * 4096
        assert kv_info["total_kv_gb"] == 2.0  # 2 GB for 4096 context

    def test_memory_accounting_telemetry(self):
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=8 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        mm.allocateTensor("t1", 1024 * 1024 * 1024)
        mm.allocateTensor("t2", 500 * 1024 * 1024, preferred_location=MemoryTier.TIER_2_RAM)

        state = mm.getMemoryState()
        assert state["vram"]["allocated_mb"] == 1024.0
        assert state["ram"]["allocated_mb"] == 500.0
        assert "fragmentation" in state["vram"]
        assert "fragmentation" in state["ram"]
        assert state["tensors_count"] == 2

    def test_camel_case_aliases(self):
        mm = UnifiedMemoryManager(
            vram_capacity_bytes=8 * 1024 * 1024 * 1024,
            ram_capacity_bytes=16 * 1024 * 1024 * 1024,
        )

        t = mm.allocateTensor("t1", 100 * 1024 * 1024)
        assert t.tensor_id == "t1"

        t_moved = mm.moveTensor("t1", MemoryTier.TIER_2_RAM)
        assert t_moved.current_location == MemoryTier.TIER_2_RAM

        freed = mm.freeTensor("t1")
        assert freed is True
