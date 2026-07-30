"""
cache_cmd.py
------------
Command handler for 'inferenceos cache [action]'.
"""
from __future__ import annotations

import shutil
from rich.console import Console
from cli.core.config_manager import get_config_manager


def handle_cache_command(action: str = "view") -> None:
    """Manage runtime placement & benchmark cache."""
    console = Console()
    config_mgr = get_config_manager()
    cache_dir = config_mgr.cache_dir

    if action.lower() in ("clear", "clean", "purge"):
        for item in cache_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        console.print("[green]✔ Placement and benchmark cache cleared successfully.[/green]")
    else:
        files = list(cache_dir.glob("*"))
        console.print(f"[bold cyan]InferenceOS Cache Directory ({cache_dir}):[/bold cyan]")
        for f in files:
            console.print(f"  • {f.name} ({f.stat().st_size} bytes)")
        if not files:
            console.print("  [dim]Cache is empty.[/dim]")
