"""
run_multiformat_benchmark.py
----------------------------
Comprehensive end-to-end benchmark across all supported model formats using real SLMs:
  1. GGUF        (Qwen3-0.6B-Q8_0.gguf)
  2. ONNX        (tinystories_8m.onnx)
  3. OBX         (tinystories_8m.obx)
  4. SafeTensors (pythia-14m / model.safetensors)
  5. PyTorch     (TinyStories-8M / pytorch_model.bin)

Measures:
  - Tokens Generated
  - Generation Speed (tokens/sec)
  - Time To First Token (TTFT ms)
  - Total Wall-Clock Latency (ms)
  - Generated Output Quality / Text Coherence
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Dict, Any

from inference_runtime.multiformat_engine import MultiFormatRuntimeEngine
from inference_runtime.runtime_config import RuntimeConfig
import profiler

def run_benchmark():
    sys_res = profiler.get_system_resources()
    hw = sys_res.to_dict() if hasattr(sys_res, "to_dict") else sys_res

    cfg = RuntimeConfig(n_predict=32, temp=0.7)
    engine = MultiFormatRuntimeEngine(hw_profile=hw, config=cfg)

    benchmarks = [
        {
            "format": "GGUF",
            "name": "Qwen3-0.6B-Q8_0.gguf (600M)",
            "path": Path("models/Qwen3-0.6B-Q8_0.gguf"),
            "prompt": "Explain gravity in one sentence.",
            "max_tokens": 32,
        },
        {
            "format": "ONNX",
            "name": "tinystories_8m.onnx (8M)",
            "path": Path("models/test_formats/tinystories_8m.onnx"),
            "prompt": "Once upon a time, there was a little dog named Rover who loved to run.",
            "max_tokens": 32,
        },
        {
            "format": "OBX",
            "name": "tinystories_8m.obx (8M)",
            "path": Path("models/test_formats/tinystories_8m.obx"),
            "prompt": "Once upon a time, there was a clever little cat named Whiskers.",
            "max_tokens": 32,
        },
        {
            "format": "SafeTensors",
            "name": "pythia-14m (14M)",
            "path": Path(r"C:\Users\gamin\.cache\huggingface\hub\models--EleutherAI--pythia-14m\snapshots\cf967c0a9a04383db6f7b1108d86b2962634b4ac\model.safetensors"),
            "prompt": "Artificial intelligence will transform the future of science by",
            "max_tokens": 32,
        },
        {
            "format": "PyTorch (.bin)",
            "name": "TinyStories-8M / pytorch_model.bin (8M)",
            "path": Path(r"C:\Users\gamin\.cache\huggingface\hub\models--roneneldan--TinyStories-8M\snapshots\8612e3b15c66ffa94eaa6ee0de5c96edd2d630af\pytorch_model.bin"),
            "prompt": "Once upon a time, Lily found a shining star in the grass.",
            "max_tokens": 32,
        },
        {
            "format": "PyTorch (.pth)",
            "name": "tinystories_8m.pth (8M)",
            "path": Path("models/test_formats/tinystories_8m.pth"),
            "prompt": "Once upon a time, there was a tiny blue bird.",
            "max_tokens": 32,
        },
        {
            "format": "Pickle (.pkl)",
            "name": "tinystories_8m.pkl (8M)",
            "path": Path("models/test_formats/tinystories_8m.pkl"),
            "prompt": "Once upon a time, there was a little cat named Kitty.",
            "max_tokens": 32,
        },
        {
            "format": "Pickle (.pickle)",
            "name": "tinystories_8m.pickle (8M)",
            "path": Path("models/test_formats/tinystories_8m.pickle"),
            "prompt": "Once upon a time, there was a playful puppy in the garden.",
            "max_tokens": 32,
        },
        {
            "format": "TFLite (.tflite)",
            "name": "text_classification.tflite",
            "path": Path("models/test_formats/text_classification.tflite"),
            "prompt": "This movie was absolutely wonderful and inspiring!",
            "max_tokens": 32,
        },
        {
            "format": "OpenVINO (.xml)",
            "name": "tinystories_8m.xml (8M)",
            "path": Path("models/test_formats/tinystories_8m.xml"),
            "prompt": "Once upon a time, there was a tiny kitten.",
            "max_tokens": 32,
        },
        {
            "format": "Keras 3 (.keras)",
            "name": "text_model.keras",
            "path": Path("models/test_formats/text_model.keras"),
            "prompt": "InferenceOS is blazing fast and lightweight.",
            "max_tokens": 32,
        },
        {
            "format": "Keras Legacy (.h5)",
            "name": "text_model.h5",
            "path": Path("models/test_formats/text_model.h5"),
            "prompt": "Legacy H5 model evaluation on InferenceOS.",
            "max_tokens": 32,
        },
        {
            "format": "PyTorch INT8",
            "name": "tinystories_8m_int8.pt (8M)",
            "path": Path("models/test_formats/tinystories_8m_int8.pt"),
            "prompt": "Once upon a time, there was a little bird.",
            "max_tokens": 32,
        },
    ]

    results = []
    print("=" * 80)
    print("InferenceOS Multi-Format SLM Benchmark Suite")
    print("=" * 80)

    for b in benchmarks:
        p = b["path"]
        fmt = b["format"]
        name = b["name"]

        if not p.exists():
            print(f"[-] Skipping {fmt} ({name}): File not found at {p}")
            continue

        print(f"\n[+] Running Benchmark for {fmt} ({name})...")
        t0 = time.perf_counter()
        res = engine.execute(
            model_path=p,
            prompt=b["prompt"],
            max_tokens=b["max_tokens"],
            temp=0.7,
        )
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        stats = res.stats
        tokens = stats.tokens_generated
        tps = stats.eval_tps
        ttft = stats.prompt_eval_ms
        backend = res.backend

        # Clean preview text
        sample_output = res.generated_text.strip().replace("\n", " ")[:100]

        record = {
            "format": fmt,
            "model_name": name,
            "backend": backend,
            "tokens_generated": tokens,
            "tokens_per_sec": round(tps, 2),
            "ttft_ms": round(ttft, 2),
            "total_latency_ms": round(total_time_ms, 2),
            "output_preview": sample_output,
            "success": res.success,
        }
        results.append(record)

        print(f"    Backend:     {backend}")
        print(f"    Tokens:      {tokens} tokens")
        print(f"    TPS:         {tps:.2f} tok/s")
        print(f"    TTFT:        {ttft:.2f} ms")
        print(f"    Total Wall:  {total_time_ms:.2f} ms")
        print(f"    Output:      \"{sample_output}...\"")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Format':<12} | {'Model':<30} | {'Backend':<28} | {'Tokens':<6} | {'TPS':<8} | {'TTFT (ms)':<9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['format']:<12} | {r['model_name']:<30} | {r['backend']:<28} | {r['tokens_generated']:<6} | {r['tokens_per_sec']:<8.2f} | {r['ttft_ms']:<9.2f}")
    print("=" * 80)

    # Save to json report
    report_path = Path("benchmarks/multiformat_benchmark_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full benchmark results saved to {report_path.resolve()}")

if __name__ == "__main__":
    run_benchmark()
