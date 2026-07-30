"""
hardware.py
-----------
Command handler for 'inferenceos hardware'.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import profiler
from igpu_support import IgpuProfiler


def handle_hardware_command() -> None:
    """Display comprehensive system hardware inventory."""
    console = Console()
    console.print("[bold cyan]Querying System Hardware Resources...[/bold cyan]")

    sys_res = profiler.get_system_resources()
    hw = sys_res.to_dict() if hasattr(sys_res, "to_dict") else sys_res

    cpu = hw.get("cpu", {})
    mem = hw.get("memory", {})
    gpus = hw.get("gpus", [])

    igpu_prof = IgpuProfiler(hw)
    igpu_info = igpu_prof.profile()

    table = Table(title="InferenceOS Hardware Inventory", box=None, expand=True)
    table.add_column("Subsystem", style="bold cyan", width=18)
    table.add_column("Details & Capabilities", style="white")

    table.add_row("CPU Processor", f"{cpu.get('brand', 'Unknown')} ({cpu.get('physical_cores')} cores / {cpu.get('logical_cores')} threads)")
    table.add_row("System RAM", f"{mem.get('total_gb', 0):.1f} GB Total | {mem.get('available_gb', 0):.1f} GB Available")

    if gpus:
        gpu = gpus[0]
        table.add_row("Discrete GPU", f"{gpu.get('name')} ({gpu.get('vram_total_mb')} MB VRAM, {gpu.get('vram_bandwidth_gbps', 0):.0f} GB/s BW)")
    else:
        table.add_row("Discrete GPU", "None detected (CPU execution fallback)")

    table.add_row("Integrated GPU", f"{igpu_info.model} (Detected: {igpu_info.detected}, Shared RAM: {igpu_info.is_shared_memory})")

    console.print(Panel(table, border_style="cyan"))
