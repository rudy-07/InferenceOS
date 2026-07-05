# InferenceOS

A hardware-agnostic, adaptive LLM inference engine built on a `llama.cpp` fork.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Hardware Profiler | ✅ Complete |
| 2 | Dynamic Compilation | 🔲 Planned |
| 3 | Adaptive Orchestrator | 🔲 Planned |
| 4 | Advanced Features | 🔲 Planned |

## Phase 1 — Hardware Profiler

Generates a `hardware_profile.json` describing the system's capabilities and providing inference hints for downstream phases.

### Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Run the Profiler

```bash
# Write profile to hardware_profile.json (default)
python -m profiler.hardware_profiler --verbose

# Write to a custom path and also print to stdout
python -m profiler.hardware_profiler -o my_profile.json --stdout --verbose

# Print to stdout only (no file written)
python -m profiler.hardware_profiler --no-file --stdout
```

### Run Tests

```bash
pytest tests/ -v
```

### Output Example

```json
{
  "schema_version": "1.0",
  "timestamp": "2025-01-15T12:00:00Z",
  "os": {
    "system": "Windows",
    "release": "11",
    "machine": "AMD64"
  },
  "cpu": {
    "brand": "AMD Ryzen 9 7950X",
    "physical_cores": 16,
    "logical_cores": 32,
    "isa_extensions": ["avx", "avx2", "avx512f", "fma"]
  },
  "memory": {
    "total_gb": 64.0,
    "available_gb": 48.2
  },
  "gpus": [
    {
      "vendor": "nvidia",
      "name": "NVIDIA GeForce RTX 4090",
      "vram_total_mb": 24576,
      "backend_hint": "cuda"
    }
  ],
  "inference_hints": {
    "recommended_backend": "cuda",
    "recommended_quant": "Q8_0",
    "max_gpu_layers": -1,
    "parallelism_threads": 32
  }
}
```

## Project Structure

```
InferenceOS/
├── profiler/
│   ├── hardware_profiler.py      # Main entry point
│   ├── cpu_profiler.py
│   ├── memory_profiler.py
│   ├── gpu_profiler.py
│   └── gpu_backends/
│       ├── nvidia_backend.py     # pynvml → nvidia-smi → GPUtil
│       ├── amd_backend.py        # amdsmi → rocm-smi
│       ├── apple_backend.py      # system_profiler → Metal/PyObjC
│       └── intel_backend.py      # oneAPI → wmic/PowerShell → lshw
├── tests/
│   └── test_profiler.py
├── requirements.txt
└── hardware_profile.json         # Generated (gitignored)
```
