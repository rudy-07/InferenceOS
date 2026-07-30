"""
Tests for orchestrator/memory_planner.py
"""
import pytest
from orchestrator.memory_planner import MemoryPlanner, MemoryPlan

@pytest.fixture
def mock_hw_profile():
    return {
        "memory": {
            "available_gb": 16.0  # 16384 MB
        },
        "gpus": [
            {
                "vram_free_mb": 8192.0 # 8GB VRAM
            }
        ]
    }

def test_memory_planner_fits_in_vram(mock_hw_profile):
    planner = MemoryPlanner(mock_hw_profile)
    
    # Simulate a small model: 2GB size, 32 layers.
    plan = planner.calculate_plan(
        model_size_mb=2048,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        quant_bpw=4.5,
        desired_n_ctx=2048
    )
    
    # 8GB VRAM, minus 500MB overhead = 7692MB usable
    # Model is ~2GB, KV cache for 2048 ctx is small. Should fit 100% in VRAM.
    assert plan.n_gpu_layers == 32
    assert plan.offload_ratio == 1.0
    assert plan.n_ctx == 2048

def test_memory_planner_partial_offload(mock_hw_profile):
    planner = MemoryPlanner(mock_hw_profile)
    
    # Simulate a huge model: 12GB size, 40 layers.
    plan = planner.calculate_plan(
        model_size_mb=12288,
        num_layers=40,
        hidden_size=8192,
        num_heads=64,
        num_kv_heads=8,
        quant_bpw=4.5,
        desired_n_ctx=4096
    )
    
    # VRAM is 8GB, overhead 500MB -> 7692MB usable
    # Total model size is 12GB. So partial offload.
    assert 0 < plan.n_gpu_layers < 40
    assert 0.0 < plan.offload_ratio < 1.0

def test_memory_planner_oom_scaling_down_context():
    # Tiny amount of RAM to force context scaling
    tiny_hw = {
        "memory": {"available_gb": 3.0}, # 3072 MB total, 2048 MB reserved -> 1024 MB usable RAM
        "gpus": [{"vram_free_mb": 500.0}] # 0 usable VRAM after 500MB overhead
    }
    planner = MemoryPlanner(tiny_hw)
    
    # Model is 512MB, leaves 512MB for KV cache.
    plan = planner.calculate_plan(
        model_size_mb=512,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        quant_bpw=4.5,
        desired_n_ctx=100000 # Unrealistic desired ctx
    )
    
    assert plan.n_gpu_layers == 0
    assert plan.n_ctx < 100000
    assert plan.n_ctx > 0
    # Should estimate RAM footprint inside safe limits
    assert plan.estimated_ram_mb <= 1024
