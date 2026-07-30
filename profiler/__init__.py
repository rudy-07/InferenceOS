"""
InferenceOS Hardware Profiler and Resource Manager Package (Phase 1)
"""
from .hardware_profiler import (
    estimateBandwidth,
    estimate_bandwidth,
    getAvailableMemory,
    getCPUs,
    getGPUs,
    getSystemResources,
    get_available_memory,
    get_cpus,
    get_gpus,
    get_system_resources,
    run_profiler,
)
from .resource_model import (
    BaseResource,
    CPUResource,
    GPUResource,
    InterconnectResource,
    RAMResource,
    StorageResource,
    SystemResources,
)

__all__ = [
    "getSystemResources",
    "getAvailableMemory",
    "getGPUs",
    "getCPUs",
    "estimateBandwidth",
    "get_system_resources",
    "get_available_memory",
    "get_gpus",
    "get_cpus",
    "estimate_bandwidth",
    "run_profiler",
    "BaseResource",
    "CPUResource",
    "RAMResource",
    "GPUResource",
    "StorageResource",
    "InterconnectResource",
    "SystemResources",
]
