"""
monitor_ui.py
--------------
HTOP-like Live Monitor Screen for InferenceOS.

Displays continuous real-time process monitoring:
  - CPU core load bars & memory footprint
  - GPU engine utilization & VRAM allocation
  - Active inference threads and llama.cpp subprocess status
  - Real-time token streaming rate & latency stats
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import psutil
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from .themes import get_theme
from cli.core.config_manager import get_config_manager
import profiler


class MonitorInterface:
    """
    Renders HTOP-style terminal monitoring dashboard.
    """

    def __init__(self, theme_name: str = "nord", refresh_rate_hz: float = 2.0) -> None:
        self.console = Console()
        self.theme_mgr = get_theme(theme_name)
        self.config_mgr = get_config_manager()
        self.refresh_rate = max(0.5, refresh_rate_hz)

    def render_frame(self) -> Panel:
        tm = self.theme_mgr
        c_primary = tm.color("primary")
        c_secondary = tm.color("secondary")
        c_success = tm.color("success")
        c_warning = tm.color("warning")

        # Query System Resources
        cpu_percent = psutil.cpu_percent(percpu=True)
        mem = psutil.virtual_memory()

        # Build CPU Core Bars Table
        cpu_table = Table(box=None, expand=True, padding=(0, 1))
        cpu_table.add_column("CORE", style="cyan", width=8)
        cpu_table.add_column("LOAD BAR", justify="left")
        cpu_table.add_column("UTIL", justify="right", style="bold green", width=8)

        for i, pct in enumerate(cpu_percent[:8]):
            bar_len = int(pct / 5.0)
            bar_str = "█" * bar_len + "░" * (20 - bar_len)
            cpu_table.add_row(f"CPU {i}", f"[{c_success}]{bar_str}[/{c_success}]", f"{pct:.1f}%")

        # Memory & GPU Overview Table
        sys_table = Table(box=None, expand=True, padding=(0, 1))
        sys_table.add_column("RESOURCE", style="bold cyan")
        sys_table.add_column("USAGE", style="bold green")

        sys_table.add_row("RAM Used", f"{mem.used / (1024**3):.2f} / {mem.total / (1024**3):.2f} GB ({mem.percent}%)")
        sys_table.add_row("RAM Available", f"{mem.available / (1024**3):.2f} GB")

        # Check llama.exe active processes
        llama_procs: List[psutil.Process] = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
            try:
                name = p.info["name"] or ""
                if "llama" in name.lower() or "python" in name.lower():
                    llama_procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        proc_table = Table(title="Active Inference Subprocesses & Workers", box=None, expand=True)
        proc_table.add_column("PID", style="cyan", width=8)
        proc_table.add_column("NAME", style="bold white", width=18)
        proc_table.add_column("CPU %", style="bold green", justify="right")
        proc_table.add_column("RAM (MB)", style="yellow", justify="right")
        proc_table.add_column("STATUS", style="magenta")

        for p in llama_procs[:5]:
            try:
                mem_mb = (p.info["memory_info"].rss / (1024**2)) if p.info["memory_info"] else 0
                proc_table.add_row(
                    str(p.info["pid"]),
                    str(p.info["name"])[:18],
                    f"{p.info['cpu_percent'] or 0.0:.1f}%",
                    f"{mem_mb:.1f}",
                    "RUNNING",
                )
            except Exception:
                continue

        if not llama_procs:
            proc_table.add_row("-", "No active llama procs", "0.0%", "0.0", "IDLE")

        main_grid = Table(box=None, expand=True)
        main_grid.add_column("CPU & SYSTEM HARDWARE", ratio=1)
        main_grid.add_column("PROCESSES & RUNTIME", ratio=1)
        main_grid.add_row(cpu_table, proc_table)

        header = Text("📊 InferenceOS Runtime Monitor ── (Press Ctrl+C to exit)", style=f"bold {c_primary}")
        return Panel(main_grid, title=header, border_style=c_primary, padding=(1, 1))

    def run(self) -> None:
        """Run continuous live monitoring loop."""
        self.console.clear()
        try:
            with Live(self.render_frame(), console=self.console, refresh_per_second=int(self.refresh_rate)) as live:
                while True:
                    time.sleep(1.0 / self.refresh_rate)
                    live.update(self.render_frame())
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Exited InferenceOS monitor mode.[/yellow]")
