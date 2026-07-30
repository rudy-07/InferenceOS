"""
unified_memory_manager.py
--------------------------
Core Unified Memory Manager for InferenceOS Phase 2.

Presents GPU VRAM (Tier 1) and System RAM (Tier 2) as a single hierarchical
memory system. Excludes persistent disk storage from active inference memory.
Tracks tensor metadata, allocations, transfers, deallocations, peak memory,
and KV cache memory growth.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from profiler import get_system_resources
from .accounting import MemoryAccounting
from .memory_tier import MemoryTier
from .tensor import TensorMetadata


class UnifiedMemoryManager:
    """
    Unified Hierarchical Memory Manager for InferenceOS.
    """

    def __init__(
        self,
        vram_capacity_bytes: Optional[int] = None,
        ram_capacity_bytes: Optional[int] = None,
    ) -> None:
        # Initialize physical capacities from Phase 1 profiler if not supplied
        if vram_capacity_bytes is None or ram_capacity_bytes is None:
            try:
                sys_res = get_system_resources()
                if vram_capacity_bytes is None:
                    # Sum VRAM across discrete and integrated GPUs
                    vram_capacity_bytes = sum(g.vram_total_bytes for g in sys_res.gpus + sys_res.igpus)
                    if vram_capacity_bytes == 0:
                        vram_capacity_bytes = 4 * 1024 * 1024 * 1024  # 4GB fallback
                if ram_capacity_bytes is None:
                    ram_capacity_bytes = sys_res.ram.total_bytes
                    if ram_capacity_bytes == 0:
                        ram_capacity_bytes = 16 * 1024 * 1024 * 1024  # 16GB fallback
            except Exception:
                vram_capacity_bytes = vram_capacity_bytes or 8 * 1024 * 1024 * 1024
                ram_capacity_bytes = ram_capacity_bytes or 16 * 1024 * 1024 * 1024

        self.vram_capacity_bytes: int = vram_capacity_bytes
        self.ram_capacity_bytes: int = ram_capacity_bytes

        self.accounting: MemoryAccounting = MemoryAccounting(
            max_vram_bytes=self.vram_capacity_bytes,
            max_ram_bytes=self.ram_capacity_bytes,
        )

        self.tensors: Dict[str, TensorMetadata] = {}

    # ---------------------------------------------------------------------------
    # Core API 1: allocateTensor / allocate_tensor
    # ---------------------------------------------------------------------------

    def allocateTensor(
        self,
        tensor_id: str,
        size_bytes: int,
        preferred_location: MemoryTier | str = MemoryTier.TIER_1_VRAM,
        allowed_locations: Optional[List[MemoryTier | str]] = None,
        dtype: str = "f16",
        shape: Optional[Tuple[int, ...]] = None,
    ) -> TensorMetadata:
        """
        Allocates a tensor in the hierarchical memory system.
        Selects preferred_location if sufficient space exists, otherwise falls
        back to allowed_locations in priority order. Strictly rejects Disk Storage.
        """
        if tensor_id in self.tensors:
            raise ValueError(f"Tensor '{tensor_id}' is already allocated.")

        if isinstance(preferred_location, str):
            preferred_location = MemoryTier(preferred_location)

        if allowed_locations is None:
            allowed_locations = [MemoryTier.TIER_1_VRAM, MemoryTier.TIER_2_RAM]
        else:
            allowed_locations = [MemoryTier(loc) if isinstance(loc, str) else loc for loc in allowed_locations]

        # Enforce rule: Persistent storage (Disk) cannot be used for active inference memory
        valid_allowed = [loc for loc in allowed_locations if loc.is_active_inference_memory]
        if not valid_allowed:
            raise ValueError(f"No valid active inference memory tier specified in allowed_locations ({allowed_locations}). Disk storage is forbidden for active inference.")

        # Determine target tier (Preferred tier if it fits, else next allowed tier)
        target_tier: Optional[MemoryTier] = None

        candidate_tiers = []
        if preferred_location in valid_allowed:
            candidate_tiers.append(preferred_location)
        for loc in valid_allowed:
            if loc not in candidate_tiers:
                candidate_tiers.append(loc)

        for tier in candidate_tiers:
            if self._has_capacity(tier, size_bytes):
                target_tier = tier
                break

        if target_tier is None:
            raise MemoryError(
                f"Out of Memory: Cannot allocate tensor '{tensor_id}' of size {size_bytes / (1024**2):.2f} MB in VRAM or RAM."
            )

        # Record allocation in accounting system
        self.accounting.allocate(target_tier, tensor_id, size_bytes)

        metadata = TensorMetadata(
            tensor_id=tensor_id,
            size_bytes=size_bytes,
            current_location=target_tier,
            preferred_location=preferred_location,
            allowed_locations=valid_allowed,
            access_frequency=1.0,
            dtype=dtype,
            shape=shape,
        )

        self.tensors[tensor_id] = metadata
        return metadata

    allocate_tensor = allocateTensor

    # ---------------------------------------------------------------------------
    # Core API 2: moveTensor / move_tensor
    # ---------------------------------------------------------------------------

    def moveTensor(
        self,
        tensor_id: str,
        target_location: MemoryTier | str,
    ) -> TensorMetadata:
        """
        Transfers a tensor between memory tiers (e.g. VRAM <-> RAM).
        Rejects movements to persistent disk storage.
        """
        if tensor_id not in self.tensors:
            raise KeyError(f"Tensor '{tensor_id}' not found.")

        if isinstance(target_location, str):
            target_location = MemoryTier(target_location)

        if not target_location.is_active_inference_memory:
            raise ValueError(f"Cannot move active tensor to non-inference tier '{target_location}'. Disk storage is not active inference memory.")

        meta = self.tensors[tensor_id]
        current_loc = meta.current_location

        if current_loc == target_location:
            meta.record_access()
            return meta

        if target_location not in meta.allowed_locations:
            raise ValueError(f"Target location '{target_location}' is not in allowed_locations for tensor '{tensor_id}'.")

        if not self._has_capacity(target_location, meta.size_bytes):
            raise MemoryError(f"Insufficient memory in target tier '{target_location}' to move tensor '{tensor_id}' ({meta.size_bytes / (1024**2):.2f} MB).")

        # Deallocate from current tier and allocate in target tier
        self.accounting.deallocate(current_loc, tensor_id, meta.size_bytes)
        self.accounting.allocate(target_location, tensor_id, meta.size_bytes)

        meta.current_location = target_location
        meta.record_access()
        return meta

    move_tensor = moveTensor

    # ---------------------------------------------------------------------------
    # Core API 3: freeTensor / free_tensor
    # ---------------------------------------------------------------------------

    def freeTensor(self, tensor_id: str) -> bool:
        """
        Deallocates a tensor and frees its memory capacity.
        """
        if tensor_id not in self.tensors:
            return False

        meta = self.tensors.pop(tensor_id)
        self.accounting.deallocate(meta.current_location, tensor_id, meta.size_bytes)
        return True

    free_tensor = freeTensor

    # ---------------------------------------------------------------------------
    # Core API 4: estimatePeakMemory / estimate_peak_memory
    # ---------------------------------------------------------------------------

    def estimatePeakMemory(
        self,
        model_size_bytes: int,
        context_len: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 128,
        batch_size: int = 1,
        element_size_bytes: int = 2,
    ) -> Dict[str, Any]:
        """
        Estimates total peak memory requirement (Model weights + KV Cache + Workspace scratch buffer).
        """
        kv_growth = self.estimateKVGrowth(
            context_len=context_len,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            element_size_bytes=element_size_bytes,
            batch_size=batch_size,
        )
        total_kv_bytes = kv_growth["total_kv_bytes"]

        # Workspace scratch buffer estimation (~512MB default or 10% of model size)
        workspace_bytes = max(512 * 1024 * 1024, int(model_size_bytes * 0.10))

        peak_total_bytes = model_size_bytes + total_kv_bytes + workspace_bytes

        # Tier breakdown estimation
        vram_allocable = self.vram_capacity_bytes
        fits_in_vram = peak_total_bytes <= vram_allocable

        estimated_vram_bytes = min(peak_total_bytes, vram_allocable)
        estimated_ram_bytes = max(0, peak_total_bytes - estimated_vram_bytes)

        _mb = lambda b: round(b / (1024 * 1024), 2)
        _gb = lambda b: round(b / (1024 ** 3), 2)

        return {
            "model_size_bytes": model_size_bytes,
            "model_size_gb": _gb(model_size_bytes),
            "kv_cache_bytes": total_kv_bytes,
            "kv_cache_gb": _gb(total_kv_bytes),
            "workspace_bytes": workspace_bytes,
            "workspace_mb": _mb(workspace_bytes),
            "peak_total_bytes": peak_total_bytes,
            "peak_total_gb": _gb(peak_total_bytes),
            "fits_in_vram": fits_in_vram,
            "estimated_vram_gb": _gb(estimated_vram_bytes),
            "estimated_ram_gb": _gb(estimated_ram_bytes),
        }

    estimate_peak_memory = estimatePeakMemory

    # ---------------------------------------------------------------------------
    # Core API 5: estimateKVGrowth / estimate_kv_growth
    # ---------------------------------------------------------------------------

    def estimateKVGrowth(
        self,
        context_len: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 128,
        element_size_bytes: int = 2,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Estimates KV cache memory growth per token and total context window memory usage.
        Formula:
          bytes_per_token = 2 * n_layers * n_heads * head_dim * element_size_bytes * batch_size
        """
        # Key + Value vectors per layer per token
        bytes_per_token = 2 * n_layers * n_heads * head_dim * element_size_bytes * batch_size
        total_kv_bytes = bytes_per_token * context_len

        _mb = lambda b: round(b / (1024 * 1024), 2)

        return {
            "bytes_per_token": bytes_per_token,
            "kb_per_token": round(bytes_per_token / 1024, 2),
            "total_kv_bytes": total_kv_bytes,
            "total_kv_mb": _mb(total_kv_bytes),
            "total_kv_gb": round(total_kv_bytes / (1024 ** 3), 2),
            "context_length": context_len,
            "batch_size": batch_size,
        }

    estimate_kv_growth = estimateKVGrowth

    # ---------------------------------------------------------------------------
    # Accounting & Telemetry API
    # ---------------------------------------------------------------------------

    def getMemoryState(self) -> Dict[str, Any]:
        """
        Returns full accounting telemetry snapshot including allocated tensors graph.
        """
        state = self.accounting.to_dict()
        state["tensors_count"] = len(self.tensors)
        state["tensors"] = {tid: meta.to_dict() for tid, meta in self.tensors.items()}
        return state

    get_memory_state = getMemoryState

    # ---------------------------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------------------------

    def _has_capacity(self, tier: MemoryTier, size_bytes: int) -> bool:
        if tier == MemoryTier.TIER_1_VRAM:
            free_space = self.vram_capacity_bytes - self.accounting.allocated_vram_bytes - self.accounting.reserved_vram_bytes
            return size_bytes <= free_space
        elif tier == MemoryTier.TIER_2_RAM:
            free_space = self.ram_capacity_bytes - self.accounting.allocated_ram_bytes - self.accounting.reserved_ram_bytes
            return size_bytes <= free_space
        return False
