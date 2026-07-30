"""
monitor_cmd.py
--------------
Command handler for 'inferenceos monitor'.
"""
from __future__ import annotations

from cli.tui.monitor_ui import MonitorInterface


def handle_monitor_command() -> None:
    """Launch HTOP-style live monitor."""
    ui = MonitorInterface()
    ui.run()
