"""
Tests for orchestrator/gguf_selector.py
"""
import pytest
from unittest.mock import patch
from pathlib import Path
from orchestrator.gguf_selector import GGUFSelector

@pytest.fixture
def mock_hw_profile():
    return {
        "memory": {
            "available_gb": 8.0 
        },
        "gpus": [
            {
                "vram_free_mb": 4096.0 # 4GB VRAM
            }
        ]
    }

def test_select_best_model_picks_highest_bpw_that_fits(tmp_path, mock_hw_profile):
    selector = GGUFSelector(mock_hw_profile)
    
    # 8GB RAM + 4GB VRAM = 12GB total. 
    # Usable = 12 - 2.5 = 9.5GB
    
    # Create fake files
    model_name = "llama-3-8b"
    file1 = tmp_path / f"{model_name}-q4_k_m.gguf" # ~4.8 BPW
    file2 = tmp_path / f"{model_name}-q8_0.gguf"   # ~8.5 BPW
    file3 = tmp_path / f"{model_name}-f16.gguf"    # ~16.0 BPW
    
    file1.touch()
    file2.touch()
    file3.touch()
    
    # Mock stat to return fake sizes
    def mock_stat(*args, **kwargs):
        path = args[0]
        import os
        class FakeStat:
            def __init__(self, size):
                self.st_size = size
        if "q4_k_m" in path.name:
            return FakeStat(5 * 1024**3)
        elif "q8_0" in path.name:
            return FakeStat(8 * 1024**3)
        elif "f16" in path.name:
            return FakeStat(16 * 1024**3)
        return os.stat(path)
        
    with patch("pathlib.Path.stat", autospec=True, side_effect=mock_stat):
        best_file = selector.select_best_model(tmp_path, model_name)
    
    # Should pick Q8_0 (8GB fits in 9.5GB usable, and has higher BPW than Q4_K_M)
    assert best_file is not None
    assert best_file.name == f"{model_name}-q8_0.gguf"

def test_select_best_model_picks_smallest_if_none_fit(tmp_path, mock_hw_profile):
    # Overwrite HW profile to have very little memory
    mock_hw_profile["memory"]["available_gb"] = 2.0
    mock_hw_profile["gpus"][0]["vram_free_mb"] = 0.0
    # usable = max(0, 2 - 2.5) = 0.0 GB
    
    selector = GGUFSelector(mock_hw_profile)
    
    model_name = "llama-3-8b"
    file1 = tmp_path / f"{model_name}-q4_k_m.gguf"
    file2 = tmp_path / f"{model_name}-q8_0.gguf"
    
    file1.touch()
    file2.touch()
    
    def mock_stat(*args, **kwargs):
        path = args[0]
        import os
        class FakeStat:
            def __init__(self, size):
                self.st_size = size
        if "q4_k_m" in path.name:
            return FakeStat(5 * 1024**3)
        elif "q8_0" in path.name:
            return FakeStat(8 * 1024**3)
        return os.stat(path)
        
    with patch("pathlib.Path.stat", autospec=True, side_effect=mock_stat):
        best_file = selector.select_best_model(tmp_path, model_name)
    
    # None fit strictly, so it should pick the smallest (Q4_K_M)
    assert best_file is not None
    assert best_file.name == f"{model_name}-q4_k_m.gguf"
