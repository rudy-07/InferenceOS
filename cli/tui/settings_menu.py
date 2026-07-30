"""
settings_menu.py
----------------
Interactive Settings Menu for InferenceOS.

Provides interactive terminal navigation and value editing across 18 configuration categories:
  - Hardware, Runtime, Placement, Memory, Backend, Scheduler, KV Cache,
    Sampling, Generation, Telemetry, Monitoring, Logging, Network,
    Experimental, Appearance, Profiles, Keyboard Shortcuts.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from .themes import get_theme
from cli.core.config_manager import get_config_manager
from cli.core.profile_manager import get_profile_manager


CATEGORIES = [
    ("Hardware", "hardware"),
    ("Runtime", "runtime"),
    ("Placement", "placement"),
    ("Memory", "memory"),
    ("Backend", "runtime.backend"),
    ("Scheduler", "runtime"),
    ("KV Cache", "memory.kv_cache_compression"),
    ("Sampling", "sampling"),
    ("Generation", "generation"),
    ("Telemetry", "telemetry"),
    ("Monitoring", "monitoring"),
    ("Logging", "logging"),
    ("Network", "network"),
    ("Experimental", "experimental"),
    ("Appearance", "appearance"),
    ("Profiles", "profiles"),
    ("Keyboard Shortcuts", "shortcuts"),
]


class InteractiveSettingsMenu:
    """
    Renders and manages interactive settings configuration.
    """

    def __init__(self, theme_name: str = "nord") -> None:
        self.console = Console()
        self.config_mgr = get_config_manager()
        self.profile_mgr = get_profile_manager()
        self.theme_mgr = get_theme(theme_name)

    def display(self) -> None:
        """Display non-blocking formatted settings overview or interactive edit prompt."""
        self.console.clear()
        tm = self.theme_mgr
        header = Text("⚙ InferenceOS Settings & Runtime Configuration", style=f"bold {tm.color('primary')}")

        table = Table(box=None, expand=True)
        table.add_column("Category", style=f"bold {tm.color('secondary')}", width=22)
        table.add_column("Key Settings & Current Values", style="white")

        full_config = self.config_mgr.to_dict()

        for label, section_key in CATEGORIES:
            if "." in section_key:
                val = self.config_mgr.get(section_key)
                summary_str = f"{section_key} = {val}"
            else:
                sec_dict = full_config.get(section_key, {})
                if isinstance(sec_dict, dict):
                    items = [f"{k}={v}" for k, v in list(sec_dict.items())[:4]]
                    summary_str = ", ".join(items)
                else:
                    summary_str = str(sec_dict)

            table.add_row(f"❯ {label}", summary_str)

        panel = Panel(table, title=header, border_style=tm.color("primary"), padding=(1, 2))
        self.console.print(panel)

    def run_interactive(self) -> None:
        """Interactive loop allowing category selection and value updates."""
        while True:
            self.display()
            self.console.print("\n[bold cyan]Select Category by Name/Number (or 'q' to exit edit mode):[/bold cyan] ", end="")
            choice = input().strip()

            if choice.lower() in ("q", "quit", "exit"):
                break

            # Process selection
            matched_cat = None
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(CATEGORIES):
                    matched_cat = CATEGORIES[idx]
            else:
                for c in CATEGORIES:
                    if choice.lower() in c[0].lower():
                        matched_cat = c
                        break

            if matched_cat:
                self._edit_category(matched_cat)
            else:
                self.console.print("[red]Invalid selection. Press Enter to retry...[/red]")
                input()

    def _edit_category(self, category: tuple[str, str]) -> None:
        label, key_path = category
        self.console.clear()
        self.console.print(f"[bold cyan]Category: {label}[/bold cyan]\n")

        curr_val = self.config_mgr.get(key_path) if "." in key_path else self.config_mgr.to_dict().get(key_path, {})

        if isinstance(curr_val, dict):
            table = Table(box=None)
            table.add_column("Setting Key", style="cyan")
            table.add_column("Current Value", style="bold green")
            for k, v in curr_val.items():
                table.add_row(k, str(v))
            self.console.print(table)

            self.console.print("\n[bold yellow]Enter setting key to edit (or press Enter to return):[/bold yellow] ", end="")
            sub_key = input().strip()
            if sub_key and sub_key in curr_val:
                self.console.print(f"New value for {sub_key}: ", end="")
                new_val_str = input().strip()
                # Type conversion heuristic
                parsed_val: Any = new_val_str
                if new_val_str.lower() == "true":
                    parsed_val = True
                elif new_val_str.lower() == "false":
                    parsed_val = False
                elif new_val_str.isdigit():
                    parsed_val = int(new_val_str)
                else:
                    try:
                        parsed_val = float(new_val_str)
                    except ValueError:
                        pass

                self.config_mgr.set(f"{key_path}.{sub_key}", parsed_val)
                self.console.print(f"[green]Updated {key_path}.{sub_key} = {parsed_val}[/green]")
                input("Press Enter to continue...")
        else:
            self.console.print(f"Current Value: {curr_val}")
            self.console.print("New Value: ", end="")
            new_val_str = input().strip()
            if new_val_str:
                self.config_mgr.set(key_path, new_val_str)
                self.console.print(f"[green]Updated {key_path} = {new_val_str}[/green]")
                input("Press Enter to continue...")
