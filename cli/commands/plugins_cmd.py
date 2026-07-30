"""
plugins_cmd.py
--------------
Command handler for 'inferenceos plugins'.
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table
from cli.core.plugin_manager import get_plugin_manager


def handle_plugins_command() -> None:
    """List loaded InferenceOS plugins."""
    console = Console()
    mgr = get_plugin_manager()
    plugins = mgr.list_plugins()

    table = Table(title="InferenceOS Plugins Ecosystem", box=None, expand=True)
    table.add_column("Plugin Name", style="bold cyan")
    table.add_column("Version", style="yellow")
    table.add_column("Status", style="bold green")
    table.add_column("Path", style="dim")

    for p in plugins:
        table.add_row(
            p["name"],
            p["version"],
            "[green]ENABLED[/green]" if p["enabled"] else "[red]DISABLED[/red]",
            p["path"],
        )

    if not plugins:
        table.add_row("No plugins installed", "-", "-", "Plugins path: ~/.inferenceos/plugins/")

    console.print(table)
