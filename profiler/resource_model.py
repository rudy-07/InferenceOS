"""
resource_model.py
------------------
Unified hardware resource data models for InferenceOS Phase 1.

Every hardware resource (CPU, RAM, GPU, Storage, Interconnect) inherits from
BaseResource and exposes standard metrics:
  - capacity
  - available
  - bandwidth
  - latency
  - utilization
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseResource:
    """
    Base abstraction for all physical compute, memory, storage, and transport resources.
    
    Attributes
    ----------
    capacity : float
        Total capacity unit (e.g. bytes for memory/storage, core count or Hz for CPU).
    available : float
        Currently available capacity unit.
    bandwidth : float
        Bandwidth in GB/s (0.0 if unavailable or non-applicable).
    latency : float
        Access or transport latency in microseconds (0.0 if unavailable or non-applicable).
    utilization : float
        Current resource utilization percentage (0.0 to 100.0).
    """
    capacity: float = 0.0
    available: float = 0.0
    bandwidth: float = 0.0
    latency: float = 0.0
    utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to dictionary."""
        return asdict(self)


@dataclass
class CPUResource(BaseResource):
    """
    CPU compute resource representation.
    """
    brand: str = "Unknown CPU"
    architecture: str = "x86_64"
    physical_cores: int = 1
    logical_cores: int = 1
    base_freq_mhz: float = 0.0
    cache_l1_kb: float = 0.0
    cache_l2_kb: float = 0.0
    cache_l3_mb: float = 0.0
    isa_extensions: List[str] = field(default_factory=list)
    numa_nodes: int = 1
    numa_topology: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RAMResource(BaseResource):
    """
    System RAM memory resource representation.
    """
    total_bytes: int = 0
    free_bytes: int = 0
    available_bytes: int = 0
    total_gb: float = 0.0
    free_gb: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0


@dataclass
class GPUResource(BaseResource):
    """
    GPU compute and memory resource representation (Discrete or Integrated).
    """
    global_index: int = 0
    vendor: str = "unknown"  # nvidia, amd, intel, apple
    model: str = "Unknown GPU"
    uuid: str = ""
    is_integrated: bool = False
    vram_total_bytes: int = 0
    vram_available_bytes: int = 0
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    vram_used_mb: int = 0
    shared_memory_bytes: int = 0
    compute_capability: str = "unknown"
    backend_hint: str = "cpu"  # cuda, rocm, metal, vulkan
    pcie_gen: Optional[int] = None
    pcie_lanes: Optional[int] = None
    driver_version: str = "unknown"


@dataclass
class StorageResource(BaseResource):
    """
    Storage drive resource representation (SSD/HDD).
    """
    device: str = ""
    mountpoint: str = ""
    fstype: str = ""
    media_type: str = "SSD"  # SSD, HDD, NVMe, Unknown
    total_bytes: int = 0
    free_bytes: int = 0
    used_bytes: int = 0
    total_gb: float = 0.0
    free_gb: float = 0.0
    read_speed_mbps: float = 0.0
    write_speed_mbps: float = 0.0


@dataclass
class InterconnectResource(BaseResource):
    """
    Interconnect bus resource representation (PCIe, NVLink, System Bus).
    """
    name: str = "System Bus"
    type: str = "PCIe"  # PCIe, NVLink, SystemBus, Unified
    source: str = "CPU"
    destination: str = "RAM"


@dataclass
class SystemResources:
    """
    Aggregate unified hardware model container.
    """
    cpus: List[CPUResource] = field(default_factory=list)
    gpus: List[GPUResource] = field(default_factory=list)
    igpus: List[GPUResource] = field(default_factory=list)
    ram: RAMResource = field(default_factory=RAMResource)
    storage: List[StorageResource] = field(default_factory=list)
    interconnects: List[InterconnectResource] = field(default_factory=list)
    os: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    timestamp: str = ""
    inference_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire SystemResources graph into serializable dict."""
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "os": self.os,
            "cpu": self.cpus[0].to_dict() if self.cpus else {},
            "cpus": [c.to_dict() for c in self.cpus],
            "ram": self.ram.to_dict(),
            "memory": self.ram.to_dict(),  # Alias for backward compatibility
            "gpus": [g.to_dict() for g in self.gpus],
            "igpus": [g.to_dict() for g in self.igpus],
            "storage": [s.to_dict() for s in self.storage],
            "interconnects": [i.to_dict() for i in self.interconnects],
            "inference_hints": self.inference_hints,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize SystemResources graph to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
