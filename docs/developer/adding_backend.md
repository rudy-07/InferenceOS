# Tutorial: Adding a New Execution Backend

This guide outlines how to integrate a new hardware backend (e.g. WebGPU, Ascend NPU, or Qualcomm NPU) into InferenceOS.

---

## Architecture Requirements

To add a new backend, you need to update three subsystems:

1. **Hardware Profiler (`profiler/gpu_backends/`)**: Create a backend profiler class (e.g. `npu_backend.py`) that queries hardware VRAM, core counts, and driver presence.
2. **Backend Selector (`inference_runtime/backend_selector.py`)**: Add vendor matching rules in `BackendSelector.detect_backend()`.
3. **Argument Builder (`inference_runtime/argument_builder.py`)**: Add command-line flag translation mapping backend names to C++ runtime flags.
