"""
logs_cmd.py
-----------
Command handler for 'inferenceos logs'.
"""
from __future__ import annotations

from rich.console import Console
from cli.core.config_manager import get_config_manager


def handle_logs_command(lines: int = 50) -> None:
    """View recent runtime logs."""
    console = Console()
    config_mgr = get_config_manager()
    log_dir = config_mgr.logs_dir

    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        console.print("[yellow]No log files found in ~/.inferenceos/logs/[/yellow]")
        return

    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    console.print(f"[bold cyan]Showing last {lines} lines of {latest_log.name}:[/bold cyan]\n")

    try:
        content = latest_log.read_text(encoding="utf-8").splitlines()
        for l in content[-lines:]:
            console.print(l)
    except Exception as e:
        console.print(f"[red]Failed to read log file: {e}[/red]")
