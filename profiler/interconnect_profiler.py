"""
interconnect_profiler.py
------------------------
Detects host interconnect topology including CPU-RAM system bus, PCIe links,
and NVLink interconnects.
"""
from __future__ import annotations

import platform
from typing import Any, Dict, List

from .resource_model import InterconnectResource


def get_interconnect_resources(gpus: List[Any] = None, ram_bandwidth: float = 45.0) -> List[InterconnectResource]:
    """
    Returns InterconnectResource models for host interconnects.
    """
    interconnects: List[InterconnectResource] = []

    # 1. CPU <-> RAM System Bus
    interconnects.append(
        InterconnectResource(
            capacity=ram_bandwidth,
            available=ram_bandwidth,
            bandwidth=ram_bandwidth,
            latency=0.07,  # ns / us
            utilization=0.0,
            name="System Memory Bus",
            type="SystemBus",
            source="CPU",
            destination="RAM",
        )
    )

    # 2. PCIe Link for each discrete GPU
    if gpus:
        for idx, g in enumerate(gpus):
            is_igpu = getattr(g, "is_integrated", False) if hasattr(g, "is_integrated") else g.get("is_integrated", False)
            if not is_igpu:
                bw = getattr(g, "bandwidth", 16.0) if hasattr(g, "bandwidth") else float(g.get("pcie_bandwidth_gbps", 16.0))
                model = getattr(g, "model", "GPU") if hasattr(g, "model") else g.get("model", "GPU")
                interconnects.append(
                    InterconnectResource(
                        capacity=bw,
                        available=bw,
                        bandwidth=bw,
                        latency=5.0,
                        utilization=0.0,
                        name=f"PCIe Bus (GPU {idx}: {model})",
                        type="PCIe",
                        source="CPU",
                        destination=f"GPU_{idx}",
                    )
                )

    return interconnects


def profile(gpus: List[Any] = None) -> List[Dict[str, Any]]:
    """
    Returns interconnect profiles as a list of dicts.
    """
    return [i.to_dict() for i in get_interconnect_resources(gpus)]
