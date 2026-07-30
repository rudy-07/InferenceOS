"""
storage_profiler.py
--------------------
Detects storage devices (SSD vs HDD), capacity, available free space,
and estimated read/write speeds.
"""
from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Dict, List

from .resource_model import StorageResource


def _detect_media_type(device: str, mountpoint: str) -> str:
    """
    Detects media type: NVMe, SSD, or HDD across Windows, Linux, and macOS.
    """
    sys_name = platform.system()

    if sys_name == "Windows":
        try:
            drive_letter = mountpoint.rstrip("\\").rstrip(":")
            if drive_letter:
                cmd = f"Get-Partition -DriveLetter {drive_letter} | Get-Disk | Select-Object BusType, MediaType | ConvertTo-Csv -NoTypeInformation"
                out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, timeout=5)
                out_upper = out.upper()
                if "NVME" in out_upper:
                    return "NVMe"
                if "SSD" in out_upper:
                    return "SSD"
                if "HDD" in out_upper:
                    return "HDD"
        except Exception:
            pass
        return "SSD"

    elif sys_name == "Linux":
        try:
            # Extract base device name, e.g., /dev/sda1 -> sda, /dev/nvme0n1p1 -> nvme0n1
            dev_basename = os.path.basename(device)
            base_dev = dev_basename.rstrip("0123456789p")
            if "nvme" in dev_basename:
                return "NVMe"
            rotational_file = f"/sys/block/{base_dev}/queue/rotational"
            if os.path.exists(rotational_file):
                with open(rotational_file) as f:
                    rot = f.read().strip()
                    return "HDD" if rot == "1" else "SSD"
        except Exception:
            pass
        return "SSD"

    elif sys_name == "Darwin":
        try:
            if "nvme" in device.lower() or "apple" in device.lower():
                return "NVMe"
        except Exception:
            pass
        return "SSD"

    return "SSD"


def _estimate_speeds(media_type: str) -> tuple[float, float, float]:
    """
    Returns (read_speed_mbps, write_speed_mbps, latency_ms) for media type.
    """
    if media_type == "NVMe":
        return 3500.0, 3000.0, 0.1
    elif media_type == "SSD":
        return 550.0, 500.0, 0.5
    else:  # HDD
        return 150.0, 130.0, 10.0


def get_storage_resources() -> List[StorageResource]:
    """
    Discovers all physical mount points and returns StorageResource models.
    """
    import psutil

    resources: List[StorageResource] = []
    seen_mounts = set()

    try:
        partitions = psutil.disk_partitions(all=False)
    except Exception:
        partitions = []

    for part in partitions:
        mount = part.mountpoint
        if mount in seen_mounts or "cdrom" in part.opts or not part.fstype:
            continue
        seen_mounts.add(mount)

        try:
            usage = psutil.disk_usage(mount)
        except Exception:
            continue

        _gb = lambda b: round(b / (1024 ** 3), 2)
        media_type = _detect_media_type(part.device, mount)
        read_mbps, write_mbps, latency_ms = _estimate_speeds(media_type)

        bw_gbps = round(read_mbps / 1024.0, 2)
        total_bytes = usage.total
        free_bytes = usage.free
        used_bytes = usage.used
        percent_used = float(usage.percent)

        resources.append(
            StorageResource(
                capacity=float(total_bytes),
                available=float(free_bytes),
                bandwidth=bw_gbps,
                latency=latency_ms,
                utilization=percent_used,
                device=part.device,
                mountpoint=mount,
                fstype=part.fstype,
                media_type=media_type,
                total_bytes=total_bytes,
                free_bytes=free_bytes,
                used_bytes=used_bytes,
                total_gb=_gb(total_bytes),
                free_gb=_gb(free_bytes),
                read_speed_mbps=read_mbps,
                write_speed_mbps=write_mbps,
            )
        )

    if not resources:
        # Fallback empty drive
        resources.append(StorageResource(device="default", mountpoint="/"))

    return resources


def profile() -> List[Dict[str, Any]]:
    """
    Returns storage profiles as a list of dicts.
    """
    return [s.to_dict() for s in get_storage_resources()]
