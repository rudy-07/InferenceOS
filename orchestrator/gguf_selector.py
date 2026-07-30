"""
gguf_selector.py
----------------
Dynamically selects the highest quality GGUF quantization that fits within
the available system and GPU memory based on the hardware profile.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path


# BPW (Bits Per Weight) mapping for common quantizations
QUANT_BPW = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q8_1": 8.5,
    "Q6_K": 6.66,
    "Q5_K_M": 5.5,
    "Q5_K_S": 5.5,
    "Q5_0": 5.5,
    "Q5_1": 5.5,
    "Q4_K_M": 4.5,
    "Q4_K_S": 4.5,
    "Q4_0": 4.5,
    "Q4_1": 4.5,
    "IQ4_NL": 4.5,
    "IQ4_XS": 4.25,
    "Q3_K_L": 4.27,
    "Q3_K_M": 3.91,
    "Q3_K_S": 3.5,
    "IQ3_M": 3.7,
    "IQ3_XXS": 3.06,
    "Q2_K": 2.62,
    "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31,
    "IQ1_M": 1.75,
    "IQ1_S": 1.56,
}


class GGUFSelector:
    def __init__(self, hw_profile: dict[str, Any]):
        self.hw = hw_profile

    def get_recommended_quant(self) -> str:
        """
        Returns the optimal fallback quantization from the hardware profile.
        """
        return self.hw.get("inference_hints", {}).get("recommended_quant", "Q4_K_M")
        
    def select_best_model(self, model_dir: Path, target_model_name: str | None = None) -> Path | None:
        """
        Given a directory and optional base model name (e.g. 'llama-3-8b'), finds the highest
        quality GGUF file that fits in our system's memory.
        """
        if not model_dir.exists():
            return None
            
        candidates = []
        if target_model_name and target_model_name.lower() != "auto":
            candidates = list(model_dir.glob(f"*{target_model_name}*.gguf"))
            
        if not candidates:
            candidates = list(model_dir.glob("*.gguf"))
            
        if not candidates:
            return None
            
        # Get free memory (VRAM + RAM)
        vram_gb = sum(g.get("vram_free_mb", 0) for g in self.hw.get("gpus", [])) / 1024.0
        ram_gb = self.hw.get("memory", {}).get("available_gb", 0)
        
        # Reserve 2GB for OS and 0.5GB for backend overhead
        usable_gb = max(0.0, (vram_gb + ram_gb) - 2.5)
        
        best_file = None
        best_bpw = 0.0
        
        for candidate in candidates:
            name = candidate.stem.upper()
            quant = None
            
            # Sort QUANT_BPW keys by length descending to match longest pattern first (e.g. Q5_K_M before Q5_0)
            sorted_quants = sorted(QUANT_BPW.keys(), key=len, reverse=True)
            for q in sorted_quants:
                if q in name or q.replace("_", "") in name:
                    quant = q
                    break
                    
            if not quant:
                quant = "Q4_0"
                
            bpw = QUANT_BPW[quant]
            size_gb = candidate.stat().st_size / (1024**3)
            
            if size_gb <= usable_gb and bpw > best_bpw:
                best_bpw = bpw
                best_file = candidate
                
        # If no file strictly fits, pick the smallest one
        if not best_file:
            best_file = min(candidates, key=lambda p: p.stat().st_size)
            
        return best_file
