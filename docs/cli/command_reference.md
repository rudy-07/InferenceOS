# CLI Command Reference (`cli/`)

InferenceOS provides a 21-command terminal interface accessible via `inferenceos <command>`.

---

## Command Matrix

### 1. `chat`
Launch an interactive terminal UI chat session with live streaming.
```bash
inferenceos chat models/llama-3-8b.gguf --theme nord
```

### 2. `run`
Execute a single prompt against a model with live token output and metrics.
```bash
inferenceos run models/llama-3-8b.gguf --prompt "Explain quantum computing" -b vulkan -t 8
```

### 3. `serve`
Launch local OpenAI and Ollama REST API server.
```bash
inferenceos serve models/llama-3-8b.gguf --host 0.0.0.0 --port 11434
```

### 4. `benchmark`
Run end-to-end performance benchmark matrix across batch sizes and thread counts.
```bash
inferenceos benchmark models/llama-3-8b.gguf -g 28 -t 8
```

### 5. `optimize`
Automated 5-step hardware detection, placement generation, matrix benchmark, and caching.
```bash
inferenceos optimize models/llama-3-8b.gguf
```

### 6. `profile`
Profile execution and generate interactive flamegraph HTML reports.
```bash
inferenceos profile models/llama-3-8b.gguf
```

### 7. `hardware`
Inspect CPU, GPU, iGPU, VRAM, RAM, and interconnect bandwidth.
```bash
inferenceos hardware
```

### 8. `doctor`
Run diagnostic health checks across drivers, SDKs, and executable binaries.
```bash
inferenceos doctor
```

### 9. `inspect`
Inspect GGUF binary metadata, tensor shapes, and layer counts.
```bash
inferenceos inspect models/llama-3-8b.gguf
```

### 10. `models`
Register, list, remove, search, or tag models in the local library.
```bash
inferenceos models list
inferenceos models add --nickname llama3 --path /path/to/llama.gguf
```

### 11. `config`
View, edit, or interactively configure runtime settings.
```bash
inferenceos config runtime.threads 8
inferenceos config --interactive
```

### 12. `plugins`
List and manage loaded extension plugins.
```bash
inferenceos plugins
```

### 13. `cache`
View or clear placement and benchmark optimization caches.
```bash
inferenceos cache clear
```

### 14. `logs`
View and tail recent runtime execution logs.
```bash
inferenceos logs -n 100
```

### 15. `telemetry`
Launch historical metrics & telemetry dashboard UI.
```bash
inferenceos telemetry
```

### 16. `monitor`
Launch HTOP-style live process & hardware resource monitor.
```bash
inferenceos monitor
```

### 17. `stats`
Display aggregated performance summary statistics.
```bash
inferenceos stats
```

### 18. `placement`
Compute and visualize layer placement across CPU/dGPU/iGPU.
```bash
inferenceos placement models/llama-3-8b.gguf
```

### 19. `update`
Check engine binaries and C++ build updates.
```bash
inferenceos update
```

### 20. `version`
Display InferenceOS release version and system build information.
```bash
inferenceos version
```

### 21. `reset`
Reset settings, profiles, and caches back to default state.
```bash
inferenceos reset
```
