"""
engine.py
---------
The AdaptiveEngine acts as the central interface for InferenceOS.
It loads the hardware profile, initializes the C++ backend via ctypes,
selects the optimal GGUF, computes the memory plan, and manages inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orchestrator.backend_wrapper import load_library_from_build_result, LlamaLibrary
from orchestrator.gguf_selector import GGUFSelector, QUANT_BPW
from orchestrator.gguf_parser import read_gguf_metadata
from orchestrator.memory_planner import MemoryPlanner, MemoryPlan


class AdaptiveEngine:
    def __init__(self, hw_profile_path: Path | None = None, build_result_path: Path | None = None):
        if hw_profile_path is None:
            hw_profile_path = Path(__file__).parent.parent / "hardware_profile.json"
            
        if not hw_profile_path.exists():
            raise FileNotFoundError(
                f"Hardware profile not found at {hw_profile_path}. "
                "Please run the Phase 1 hardware profiler first."
            )
            
        with open(hw_profile_path, "r", encoding="utf-8") as f:
            self.hw_profile: dict[str, Any] = json.load(f)
            
        self.selector = GGUFSelector(self.hw_profile)
        self.planner = MemoryPlanner(self.hw_profile)
        
        # Load the C++ backend library
        self.lib: LlamaLibrary = load_library_from_build_result(build_result_path)
        
        # We don't initialize the backend until load_model is called.
        self._is_initialized = False

    def load_model(self, model_dir: Path, target_model_name: str | None = None, desired_n_ctx: int = 4096) -> bool:
        """
        Dynamically selects the best GGUF, calculates memory constraints, and initializes the backend.
        """
        model_path = self.selector.select_best_model(model_dir, target_model_name)
        if not model_path:
            raise FileNotFoundError(f"No GGUF models matching '{target_model_name}' found in {model_dir}")
            
        print(f"[AdaptiveEngine] Selected optimal model: {model_path.name}")
        
        # Read real metadata from GGUF binary header
        params = read_gguf_metadata(model_path)
        num_layers = params["num_layers"]
        hidden_size = params["hidden_size"]
        num_heads = params["num_heads"]
        num_kv_heads = params["num_kv_heads"]
        
        # Detect BPW from filename
        quant_bpw = 4.5
        stem = model_path.stem.upper()
        for q, bpw in sorted(QUANT_BPW.items(), key=lambda x: len(x[0]), reverse=True):
            if q in stem or q.replace("_", "") in stem:
                quant_bpw = bpw
                break

        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        plan = self.planner.calculate_plan(
            model_size_mb=model_size_mb,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            quant_bpw=quant_bpw,
            desired_n_ctx=desired_n_ctx
        )
        
        print(f"[AdaptiveEngine] Memory Plan:")
        print(f"  - Offloading {plan.n_gpu_layers}/{num_layers} layers to GPU ({plan.offload_ratio*100:.1f}%)")
        print(f"  - Safe context window scaled to: {plan.n_ctx} tokens")
        print(f"  - Estimated VRAM footprint: {plan.estimated_vram_mb:.1f} MB")
        print(f"  - Estimated RAM footprint: {plan.estimated_ram_mb:.1f} MB")
        
        # Initialize backend
        if not self._is_initialized:
            print("[AdaptiveEngine] Initializing llama.cpp backend via ctypes...")
            self.lib.backend_init()
            
            # Optional: NUMA init for CPU-heavy deployments
            if plan.offload_ratio < 1.0:
                self.lib.numa_init(1) # GGML_NUMA_STRATEGY_DISTRIBUTE
                
            self._is_initialized = True
            
        print("[AdaptiveEngine] Model loaded and ready for inference.")
        return True

    def close(self) -> None:
        if self._is_initialized:
            self.lib.backend_free()
            self._is_initialized = False

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
