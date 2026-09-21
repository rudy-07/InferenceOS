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

    try:
        import rich
        rich_ver = getattr(rich, "__version__", "13+")
    except Exception:
        rich_ver = "13+"

    try:
        import prompt_toolkit
        pt_ver = getattr(prompt_toolkit, "__version__", "3.0+")
    except Exception:
        pt_ver = "3.0+"

    txt = Text()
    txt.append("⚡ InferenceOS ── The Operating System for Local AI Inference\n", style="bold cyan")
    txt.append("Version:        1.0.0 (Production Release)\n", style="bold green")
    txt.append(f"Python:         {sys.version.split()[0]}\n", style="yellow")
    txt.append(f"Platform:       {platform.system()} {platform.release()} ({platform.machine()})\n", style="white")
    txt.append("Engine:         llama.cpp Multi-Accelerator Engine\n", style="magenta")
    txt.append(f"CLI UI Engine:  Rich {rich_ver} / Prompt Toolkit {pt_ver}", style="dim")

    console.print(Panel(txt, border_style="cyan"))
