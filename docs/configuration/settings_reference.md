# Configuration Settings Reference (`configuration/`)

This document lists all configuration parameters, environment variables, and `pipeline_config.json` options in InferenceOS.

---

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `INFERENCEOS_HOME` | Root directory for configuration, profiles, and caches | `~/.inferenceos` |
| `INFERENCEOS_BACKEND` | Force execution backend (`cuda`, `rocm`, `vulkan`, `metal`, `cpu`) | `auto` |
| `INFERENCEOS_THREADS` | Number of CPU compute threads | System physical cores |
| `INFERENCEOS_PORT` | REST API Server port number | `11434` |
| `CUDA_VISIBLE_DEVICES` | Target NVIDIA GPU indices | System default |
| `HIP_VISIBLE_DEVICES` | Target AMD GPU indices | System default |

---

## Configuration File (`pipeline_config.json`)

```json
{
  "runtime": {
    "threads": 8,
    "context_length": 4096,
    "batch_size": 512,
    "use_flash_attn": true,
    "use_mmap": true,
    "use_mlock": false
  },
  "schedulers": {
    "enable_dynamic_microbatch": true,
    "microbatch_safety_margin": 0.15,
    "enable_dynamic_context": true
  },
  "memory": {
    "vram_safety_margin": 0.15,
    "ram_safety_margin": 0.10,
    "kv_quantization": "Q8_0",
    "kv_eviction_policy": "h2o"
  },
  "learning": {
    "enable_runtime_learning": true,
    "db_path": "runtime_learning.db"
  }
}
```
