"""
version_cmd.py
--------------
Command handler for 'inferenceos version'.
"""
from __future__ import annotations

import platform
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def handle_version_command() -> None:
    """Display InferenceOS version banner."""
    console = Console()
    txt = Text()
    txt.append("⚡ InferenceOS ── The Operating System for Local AI Inference\n", style="bold cyan")
    txt.append("Version:        1.0.0 (Production Release)\n", style="bold green")
    txt.append(f"Python:         {sys.version.split()[0]}\n", style="yellow")
    txt.append(f"Platform:       {platform.system()} {platform.release()} ({platform.machine()})\n", style="white")
    txt.append("Engine:         llama.cpp Multi-Accelerator Engine\n", style="magenta")
    txt.append("CLI UI Engine:  Rich 15.0.0 / Prompt Toolkit 3.0.52", style="dim")

    console.print(Panel(txt, border_style="cyan"))
