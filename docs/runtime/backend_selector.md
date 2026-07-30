# Multi-Vendor Backend Selector (`inference_runtime/backend_selector.py`)

The `BackendSelector` module resolves the optimal hardware backend (`cuda`, `rocm`, `vulkan`, `metal`, or `cpu`) based on system hardware profiles, GPU vendor signatures, driver capabilities, and model layer offload plans.

---

## Resolution Flow Chart

```mermaid
flowchart TD
    Start["Request Backend Resolution"] --> CheckOverride{"Force Backend Override Specified?"}
    CheckOverride -- Yes --> UseOverride["Return Overridden Backend Info"]
    CheckOverride -- No --> CheckHint{"Inference Hints Present in Profile?"}
    
    CheckHint -- Yes --> UseHint["Use Hinted Backend (e.g., Vulkan / CUDA)"]
    CheckHint -- No --> InspectVendor{"Inspect System GPU Vendors"}
    
    InspectVendor -- NVIDIA GPU --> SelectCUDA["Select CUDA Backend"]
    InspectVendor -- AMD GPU --> SelectVulkanHIP["Select Vulkan / HIP Backend"]
    InspectVendor -- Apple Silicon --> SelectMetal["Select Metal Backend"]
    InspectVendor -- Intel Arc GPU --> SelectVulkanOneAPI["Select Vulkan Backend"]
    InspectVendor -- No GPU Detected --> SelectCPU["Select CPU Backend"]
    
    UseOverride --> ResolveSplitMode
    UseHint --> ResolveSplitMode
    SelectCUDA --> ResolveSplitMode
    SelectVulkanHIP --> ResolveSplitMode
    SelectMetal --> ResolveSplitMode
    SelectVulkanOneAPI --> ResolveSplitMode
    SelectCPU --> ResolveSplitMode
    
    ResolveSplitMode{"Placement Has PCIe Boundaries?"} -- Yes --> SetLayerSplit["set split_mode = 'layer'"]
    ResolveSplitMode -- No --> SetNoneSplit["set split_mode = 'none'"]
```

---

## Backend Feature Matrix

| Backend | Vendor Target | Environment Variable | FlashAttention | Split Mode |
| :--- | :--- | :--- | :---: | :---: |
| `cuda` | NVIDIA GPUs | `CUDA_VISIBLE_DEVICES` | ✅ | `layer` / `row` |
| `rocm` | AMD Radeon / Instinct | `HIP_VISIBLE_DEVICES` | ✅ | `layer` |
| `vulkan` | AMD, Intel, NVIDIA | `GGML_VULKAN_DEVICE` | 🟡 Basic | `layer` |
| `metal` | Apple M-Series | Default | ✅ | `none` (Unified) |
| `cpu` | Generic x86_64 / ARM64 | `OMP_NUM_THREADS` | ❌ | `none` |
