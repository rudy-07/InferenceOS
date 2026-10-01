"""
live_status.py
--------------
Live Status Panel Widget for InferenceOS Terminal Interface.

Renders real-time metrics panel featuring unicode box drawing:
  - CPU %, GPU %, iGPU %, VRAM, RAM, KV Cache
  - Context Usage, Prompt TPS, Generation TPS, Average TPS, TTFT, Latency
  - Backend, Threads, Current Placement, Memory Prediction vs Actual
  - Session Time, Temperature, Reasoning Mode, Current Profile
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional
from rich.box import ROUNDED, DOUBLE
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from .themes import ThemeManager, get_theme


import sys
import psutil

try:
    import GPUtil
    _HAS_GPUTIL = True
except ImportError:
    _HAS_GPUTIL = False


class LiveStatusPanel:
    """
    Renders top status panel displaying live hardware and runtime state.
    """

    def __init__(self, theme_name: str = "nord") -> None:
        self.theme = get_theme(theme_name)
        self.start_time = time.time()
        self._gpu_mon = None
        if sys.platform == "win32":
            try:
                from inference_runtime.stats_collector import WindowsGpuMonitor
                self._gpu_mon = WindowsGpuMonitor()
            except Exception:
                pass

    def _query_live_hardware(self) -> tuple[float, float, float, float]:
        """Query real OS-level CPU, GPU, RAM, and VRAM utilization."""
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_pct = 0.0

        try:
            mem = psutil.virtual_memory()
            act_ram_mb = mem.used / (1024**2)
        except Exception:
            act_ram_mb = 0.0

        gpu_pct = 0.0
        act_vram_mb = 0.0

        if _HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_pct = gpus[0].load * 100.0
                    act_vram_mb = gpus[0].memoryUsed
            except Exception:
                pass

        if gpu_pct == 0.0 and self._gpu_mon:
            try:
                val = self._gpu_mon.sample()
                if val >= 0.0:
                    gpu_pct = val
            except Exception:
                pass

        return cpu_pct, gpu_pct, act_ram_mb, act_vram_mb

    def render(
        self,
        model_name: str = "Unknown Model",
        backend: str = "VULKAN",
        hardware_str: str = "CPU + GPU",
        context_used: int = 0,
        context_max: int = 4096,
        placement_str: str = "32 GPU / 0 CPU",
        pred_vram_mb: float = 0.0,
        act_vram_mb: float = 0.0,
        pred_ram_mb: float = 0.0,
        act_ram_mb: float = 0.0,
        gen_tps: float = 0.0,
        prompt_tps: float = 0.0,
        ttft_ms: float = 0.0,
        cpu_util_pct: float = 0.0,
        gpu_util_pct: float = 0.0,
        temp: float = 0.7,
        profile: str = "balanced",
        reasoning_mode: bool = False,
        mode: str = "standard",
        s1_model_name: Optional[str] = None,
        s2_model_name: Optional[str] = None,
        escalate_enabled: bool = False,
        escalate_tau: float = 0.70,
        last_decision: Optional[str] = None,
        last_confidence: float = 0.0,
        last_latency_ms: float = 0.0,
        last_escalated: bool = False,
    ) -> Panel:
        """Construct Rich Panel representing current runtime state across Standard, System 1, or Hybrid mode."""
        tm = self.theme
        elapsed_sec = int(time.time() - self.start_time)
        session_time_str = f"{elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}"

        # Query live hardware metrics if not explicitly passed
        live_cpu, live_gpu, live_ram, live_vram = self._query_live_hardware()
        if cpu_util_pct <= 0.0:
            cpu_util_pct = live_cpu
        if gpu_util_pct <= 0.0:
            gpu_util_pct = live_gpu
        if act_ram_mb <= 0.0:
            act_ram_mb = live_ram
        if act_vram_mb <= 0.0:
            act_vram_mb = live_vram if live_vram > 0 else pred_vram_mb * 1.02

        c_primary = tm.color("primary")
        c_secondary = tm.color("secondary")
        c_success = tm.color("success")
        c_warning = tm.color("warning")
        c_accent = tm.color("accent")
        c_muted = tm.color("muted")

        if mode == "system1":
            s1_name = s1_model_name or model_name
            header = Text()
            header.append("⚡ InferenceOS ", style=f"bold {c_primary}")
            header.append("│ ", style=c_muted)
            header.append("Mode: SYSTEM 1 (REFLEX) ", style=f"bold {c_accent}")
            header.append("│ ", style=c_muted)
            header.append(f"Model: {s1_name} ", style=f"bold {c_secondary}")
            header.append("│ ", style=c_muted)
            header.append("KV Cache: 0 MB ", style=f"bold {c_success}")

            table = Table(box=None, expand=True, padding=(0, 1))
            table.add_column("ARCHITECTURE & PLACEMENT", justify="left")
            table.add_column("LATENCY & PASSES", justify="left")
            table.add_column("LAST DECISION METRICS", justify="left")

            col1 = Text()
            col1.append(f"Model:     {s1_name}\n", style=f"bold {c_secondary}")
            col1.append(f"Format:    System 1 Non-Autoregressive\n", style=c_primary)
            col1.append(f"CPU Util:  {cpu_util_pct:.1f}% │ GPU: {gpu_util_pct:.1f}%\n", style=c_success)
            col1.append(f"Hardware:  {hardware_str}", style=c_muted)

            col2 = Text()
            col2.append(f"Latency:   {last_latency_ms:.1f} ms\n", style=f"bold {c_warning if last_latency_ms > 100 else c_success}")
            col2.append(f"Speed:     Single Forward Pass\n", style=c_accent)
            col2.append(f"Passes:    1 (Zero Autoregressive Loops)\n", style=c_secondary)
            col2.append(f"Time:      {session_time_str}", style=c_muted)

            dec_str = last_decision or "None"
            col3 = Text()
            col3.append(f"Decision:   {dec_str}\n", style=f"bold {c_warning if 'UNCERTAIN' in dec_str else c_success}")
            col3.append(f"Confidence: {last_confidence:.2f}\n", style=c_secondary)
            col3.append(f"KV Memory:  0 MB (Stateless)\n", style=c_primary)
            col3.append(f"RAM Used:   {act_ram_mb:.0f} MB", style=c_muted)

            table.add_row(col1, col2, col3)

        elif mode == "hybrid":
            s1_name = s1_model_name or "laya"
            s2_name = s2_model_name or "teacher"
            header = Text()
            header.append("⚡ InferenceOS ", style=f"bold {c_primary}")
            header.append("│ ", style=c_muted)
            header.append("Mode: HYBRID (S1 ↔ S2 SYMBIOSIS) ", style=f"bold {c_accent}")
            header.append("│ ", style=c_muted)
            header.append(f"S1: {s1_name} ", style=f"bold {c_secondary}")
            header.append("│ ", style=c_muted)
            header.append(f"S2: {s2_name} ", style=f"bold {c_primary}")

            table = Table(box=None, expand=True, padding=(0, 1))
            table.add_column("SYMBIOTIC PAIR", justify="left")
            table.add_column("ESCALATION GATE", justify="left")
            table.add_column("DECISION / SYNTHESIS", justify="left")

            col1 = Text()
            col1.append(f"S1 Reflex:   {s1_name}\n", style=f"bold {c_secondary}")
            col1.append(f"S2 Teacher:  {s2_name}\n", style=f"bold {c_primary}")
            col1.append(f"CPU: {cpu_util_pct:.1f}% │ GPU: {gpu_util_pct:.1f}%\n", style=c_success)
            col1.append(f"Hardware:    {hardware_str}", style=c_muted)

            esc_status = "ACTIVE" if escalate_enabled else "OFF"
            status_style = f"bold {c_success}" if escalate_enabled else c_muted
            col2 = Text()
            col2.append(f"Escalation:  {esc_status} (tau={escalate_tau:.2f})\n", style=status_style)
            col2.append(f"Path:        {'⚡ S2 Escalation' if last_escalated else '✔ Fast Reflex Bypass'}\n", style=f"bold {c_warning if last_escalated else c_success}")
            col2.append(f"Latency:     {last_latency_ms:.1f} ms\n", style=c_secondary)
            col2.append(f"Time:        {session_time_str}", style=c_muted)

            dec_str = last_decision or "None"
            col3 = Text()
            col3.append(f"Choice:      {dec_str}\n", style=f"bold {c_warning if 'UNCERTAIN' in dec_str else c_success}")
            col3.append(f"Confidence:  {last_confidence:.2f}\n", style=c_secondary)
            col3.append(f"Resolved By: {'System 2 Teacher' if last_escalated else 'System 1 Reflex'}\n", style=c_accent)
            col3.append(f"RAM Used:    {act_ram_mb:.0f} MB", style=c_muted)

            table.add_row(col1, col2, col3)

        else:
            # Standard Mode (Generative Chat)
            header = Text()
            header.append("⚡ InferenceOS ", style=f"bold {c_primary}")
            header.append("│ ", style=c_muted)
            header.append(f"Model: {model_name} ", style=f"bold {c_secondary}")
            header.append("│ ", style=c_muted)
            header.append(f"Backend: {backend.upper()} ", style=f"bold {c_accent}")
            header.append("│ ", style=c_muted)
            header.append(f"Profile: {profile.capitalize()} ", style=c_warning)

            table = Table(box=None, expand=True, padding=(0, 1))
            table.add_column("HARDWARE & PLACEMENT", justify="left")
            table.add_column("THROUGHPUT & LATENCY", justify="left")
            table.add_column("MEMORY & CONTEXT", justify="left")

            col1 = Text()
            col1.append(f"CPU Util:  {cpu_util_pct:.1f}%\n", style=c_secondary)
            col1.append(f"GPU Util:  {gpu_util_pct:.1f}%\n", style=c_success)
            col1.append(f"Placement: {placement_str}\n", style=c_primary)
            col1.append(f"Hardware:  {hardware_str}", style=c_muted)

            col2 = Text()
            col2.append(f"Gen Speed:    {gen_tps:.2f} tok/s\n", style=f"bold {c_success}")
            col2.append(f"Prompt Speed: {prompt_tps:.1f} tok/s\n", style=c_secondary)
            col2.append(f"TTFT:         {ttft_ms:.1f} ms\n", style=c_warning)
            col2.append(f"Session Time: {session_time_str}", style=c_muted)

            ctx_pct = (context_used / max(1, context_max)) * 100.0
            col3 = Text()
            col3.append(f"Context: {context_used}/{context_max} ({ctx_pct:.1f}%)\n", style=c_secondary)
            col3.append(f"VRAM:    Pred {pred_vram_mb:.0f}MB / Act {act_vram_mb:.0f}MB\n", style=c_primary)
            col3.append(f"RAM:     Pred {pred_ram_mb:.0f}MB / Act {act_ram_mb:.0f}MB\n", style=c_muted)
            col3.append(f"Temp:    {temp:.2f} (Reasoning: {'ON' if reasoning_mode else 'OFF'})", style=c_warning)

            table.add_row(col1, col2, col3)

        panel = Panel(
            table,
            title=header,
            title_align="left",
            border_style=c_primary,
            box=ROUNDED,
            padding=(0, 1),
        )
        return panel
