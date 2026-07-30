"""
memory_planner.py
-----------------
Hardware-aware memory planner. Computes bytes-per-layer, calculates max
GPU offload layers (n_gpu_layers), and scales the context window (n_ctx)
to prevent Out-Of-Memory (OOM) errors based on the hardware_profile.json.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryPlan:
    n_gpu_layers: int
    n_ctx: int
    offload_ratio: float
    estimated_vram_mb: float
    estimated_ram_mb: float


class MemoryPlanner:
    def __init__(self, hw_profile: dict[str, Any]):
        self.hw = hw_profile
        
    def _get_vram_free_mb(self) -> float:
        """Returns the total free VRAM across all GPUs in MB."""
        gpus = self.hw.get("gpus", [])
        if not gpus:
            return 0.0
        # If there are multiple GPUs, we aggregate free VRAM.
        # For simplicity, assuming a single unified pool (e.g. LLAMA_SPLIT_MODE_LAYER).
        return sum(gpu.get("vram_free_mb", 0.0) for gpu in gpus)
        
    def _get_ram_free_mb(self) -> float:
        """Returns the total free system RAM in MB."""
        return self.hw.get("memory", {}).get("available_gb", 0.0) * 1024

    def calculate_plan(
        self,
        model_size_mb: float,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        quant_bpw: float, # Bits per weight (e.g. 4.5 for Q4_K_M)
        desired_n_ctx: int = 4096
    ) -> MemoryPlan:
        """
        Calculate an execution plan that fits within available VRAM and RAM.
        """
        vram_mb = self._get_vram_free_mb()
        ram_mb = self._get_ram_free_mb()
        
        # 1. Base model memory
        bytes_per_layer = (model_size_mb * 1024 * 1024) / num_layers
        
        # 2. KV Cache memory estimation per token per layer
        # Formula: 2 (K and V) * 2 bytes (fp16) * hidden_size * (num_kv_heads / num_heads)
        bytes_per_token_per_layer = 4 * hidden_size * (num_kv_heads / num_heads)
        total_kv_bytes_per_token = bytes_per_token_per_layer * num_layers
        
        # Calculate how many layers can fit in VRAM
        # Reserve 500MB VRAM for CUDA context/backend overhead
        usable_vram_bytes = max(0.0, (vram_mb - 500) * 1024 * 1024)
        
        if usable_vram_bytes == 0:
            # CPU only
            n_gpu_layers = 0
            offload_ratio = 0.0
            estimated_vram = 0.0
        else:
            # How many layers can we fit?
            # We assume KV cache is distributed proportionally to the offloaded layers.
            bytes_per_layer_with_ctx = bytes_per_layer + (bytes_per_token_per_layer * desired_n_ctx)
            
            n_gpu_layers = math.floor(usable_vram_bytes / bytes_per_layer_with_ctx)
            
            # Clamp to actual number of layers
            n_gpu_layers = min(n_gpu_layers, num_layers)
            
            offload_ratio = n_gpu_layers / num_layers
            estimated_vram = (n_gpu_layers * bytes_per_layer_with_ctx) / (1024 * 1024)
            
        # 3. Check if we have enough system RAM for remaining CPU layers (if any)
        remaining_layers = num_layers - n_gpu_layers
        if remaining_layers > 0:
            bytes_per_layer_with_ctx = bytes_per_layer + (bytes_per_token_per_layer * desired_n_ctx)
            estimated_ram = (remaining_layers * bytes_per_layer_with_ctx) / (1024 * 1024)
            
            # Reserve 2GB for OS overhead from available RAM, clamping usable_ram to at least 512MB
            usable_ram = max(512.0, ram_mb - 2048.0)
            
            if estimated_ram > usable_ram:
                # OOM risk! Scale down context window for remaining CPU layers
                usable_ram_bytes = usable_ram * 1024 * 1024
                static_ram_bytes = remaining_layers * bytes_per_layer
                remaining_for_kv = usable_ram_bytes - static_ram_bytes
                
                if remaining_for_kv <= 0:
                    raise MemoryError("Insufficient System RAM to load remaining model layers even with 0 context window.")
                    
                safe_n_ctx = math.floor(remaining_for_kv / (remaining_layers * bytes_per_token_per_layer))
                desired_n_ctx = max(512, min(desired_n_ctx, safe_n_ctx))
                estimated_ram = (static_ram_bytes + (remaining_layers * bytes_per_token_per_layer * desired_n_ctx)) / (1024 * 1024)
        else:
            estimated_ram = 0.0

        return MemoryPlan(
            n_gpu_layers=n_gpu_layers,
            n_ctx=desired_n_ctx,
            offload_ratio=offload_ratio,
            estimated_vram_mb=estimated_vram,
            estimated_ram_mb=estimated_ram,
        )
