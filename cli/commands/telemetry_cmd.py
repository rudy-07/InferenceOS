"""
telemetry_cmd.py
----------------
Command handler for 'inferenceos telemetry'.
"""
from __future__ import annotations

from cli.tui.telemetry_ui import TelemetryInterface


def handle_telemetry_command() -> None:
    """Launch telemetry dashboard."""
    ui = TelemetryInterface()
    ui.display()
