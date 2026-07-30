"""
config_cmd.py
-------------
Command handler for 'inferenceos config [key] [val] --interactive'.
"""
from __future__ import annotations

from typing import Optional
from rich.console import Console
from cli.core.config_manager import get_config_manager
from cli.tui.settings_menu import InteractiveSettingsMenu


def handle_config_command(
    key: Optional[str] = None,
    value: Optional[str] = None,
    interactive: bool = False,
) -> None:
    """View or edit configuration parameters."""
    console = Console()
    menu = InteractiveSettingsMenu()

    if interactive:
        menu.run_interactive()
        return

    config_mgr = get_config_manager()

    if key and value is not None:
        # Heuristic value conversion
        val: Any = value
        if value.lower() == "true":
            val = True
        elif value.lower() == "false":
            val = False
        elif value.isdigit():
            val = int(value)

        config_mgr.set(key, val)
        console.print(f"[green]✔ Config updated: {key} = {val}[/green]")
    elif key:
        val = config_mgr.get(key)
        console.print(f"[cyan]{key} = [bold green]{val}[/bold green][/cyan]")
    else:
        menu.display()
