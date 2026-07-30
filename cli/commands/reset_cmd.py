"""
reset_cmd.py
------------
Command handler for 'inferenceos reset'.
"""
from __future__ import annotations

from rich.console import Console
from cli.core.config_manager import get_config_manager
from cli.core.telemetry_store import get_telemetry_store


def handle_reset_command() -> None:
    """Reset configuration, profiles, and telemetry back to default states."""
    console = Console()
    config_mgr = get_config_manager()
    config_mgr.reset_to_defaults()

    telemetry_store = get_telemetry_store()
    telemetry_store.clear_history()

    console.print("[bold green]✔ InferenceOS configuration, active profiles, and telemetry history have been reset to factory defaults.[/bold green]")
