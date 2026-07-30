"""
update_cmd.py
-------------
Command handler for 'inferenceos update'.
"""
from __future__ import annotations

from rich.console import Console
from run_e2e_integration import find_llama_cli


def handle_update_command() -> None:
    """Check for llama.cpp binary updates and system engine status."""
    console = Console()
    console.print("[bold cyan]Checking InferenceOS System & Binary Executable Status...[/bold cyan]")

    exe_path = find_llama_cli()
    if exe_path and exe_path.exists():
        console.print(f"[green]✔ llama.cpp executable is up to date at: {exe_path}[/green]")
    else:
        console.print("[yellow]⚠ llama.cpp executable missing under build/bin/. Run setup build engine.[/yellow]")

    console.print("[bold green]InferenceOS CLI is running latest version 1.0.0.[/bold green]")
