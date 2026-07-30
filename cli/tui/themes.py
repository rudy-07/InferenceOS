"""
themes.py
---------
Terminal design aesthetics and theme engine for InferenceOS.

Provides curated visual color schemes for Rich terminal rendering:
  - Dark
  - Light
  - Minimal
  - Cyberpunk
  - Nord
  - Catppuccin
  - Dracula
  - Gruvbox
  - Monochrome
  - Custom user themes
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from rich.theme import Theme
from rich.style import Style


THEMES_REGISTRY: Dict[str, Dict[str, str]] = {
    "nord": {
        "primary": "#88c0d0",
        "secondary": "#81a1c1",
        "accent": "#5e81ac",
        "success": "#a3be8c",
        "warning": "#ebcb8b",
        "error": "#bf616a",
        "info": "#b48ead",
        "border": "#4c566a",
        "background": "#2e3440",
        "text": "#d8dee9",
        "muted": "#4c566a",
        "header": "bold #88c0d0",
        "metric_value": "bold #a3be8c",
        "metric_label": "#81a1c1",
    },
    "cyberpunk": {
        "primary": "#ff007f",
        "secondary": "#00f0ff",
        "accent": "#ffe600",
        "success": "#00ff66",
        "warning": "#ffaa00",
        "error": "#ff0033",
        "info": "#9900ff",
        "border": "#ff007f",
        "background": "#0d0221",
        "text": "#ffffff",
        "muted": "#555577",
        "header": "bold #ff007f",
        "metric_value": "bold #00f0ff",
        "metric_label": "#ffe600",
    },
    "dracula": {
        "primary": "#bd93f9",
        "secondary": "#ff79c6",
        "accent": "#8be9fd",
        "success": "#50fa7b",
        "warning": "#f1fa8c",
        "error": "#ff5555",
        "info": "#ffb86c",
        "border": "#6272a4",
        "background": "#282a36",
        "text": "#f8f8f2",
        "muted": "#6272a4",
        "header": "bold #bd93f9",
        "metric_value": "bold #50fa7b",
        "metric_label": "#8be9fd",
    },
    "catppuccin": {
        "primary": "#cba6f7",
        "secondary": "#89b4fa",
        "accent": "#f5e0dc",
        "success": "#a6e3a1",
        "warning": "#f9e2af",
        "error": "#f38ba8",
        "info": "#74c7ec",
        "border": "#585b70",
        "background": "#1e1e2e",
        "text": "#cdd6f4",
        "muted": "#6c7086",
        "header": "bold #cba6f7",
        "metric_value": "bold #a6e3a1",
        "metric_label": "#89b4fa",
    },
    "gruvbox": {
        "primary": "#fe8019",
        "secondary": "#83a598",
        "accent": "#fabd2f",
        "success": "#b8bb26",
        "warning": "#fe8019",
        "error": "#fb4934",
        "info": "#d3869b",
        "border": "#665c54",
        "background": "#282828",
        "text": "#ebdbb2",
        "muted": "#928374",
        "header": "bold #fe8019",
        "metric_value": "bold #b8bb26",
        "metric_label": "#83a598",
    },
    "dark": {
        "primary": "#1e90ff",
        "secondary": "#00bfff",
        "accent": "#7b68ee",
        "success": "#3cb371",
        "warning": "#ffa500",
        "error": "#ff4500",
        "info": "#da70d6",
        "border": "#444444",
        "background": "#121212",
        "text": "#ffffff",
        "muted": "#888888",
        "header": "bold #1e90ff",
        "metric_value": "bold #3cb371",
        "metric_label": "#00bfff",
    },
    "light": {
        "primary": "#005fb8",
        "secondary": "#0078d4",
        "accent": "#404040",
        "success": "#107c41",
        "warning": "#797704",
        "error": "#c42b1c",
        "info": "#5c2d91",
        "border": "#cccccc",
        "background": "#ffffff",
        "text": "#000000",
        "muted": "#666666",
        "header": "bold #005fb8",
        "metric_value": "bold #107c41",
        "metric_label": "#0078d4",
    },
    "minimal": {
        "primary": "#ffffff",
        "secondary": "#cccccc",
        "accent": "#aaaaaa",
        "success": "#ffffff",
        "warning": "#cccccc",
        "error": "#ffffff",
        "info": "#aaaaaa",
        "border": "#444444",
        "background": "#000000",
        "text": "#ffffff",
        "muted": "#666666",
        "header": "bold #ffffff",
        "metric_value": "bold #ffffff",
        "metric_label": "#cccccc",
    },
    "monochrome": {
        "primary": "#ffffff",
        "secondary": "#dddddd",
        "accent": "#bbbbbb",
        "success": "#ffffff",
        "warning": "#ffffff",
        "error": "#ffffff",
        "info": "#ffffff",
        "border": "#ffffff",
        "background": "#000000",
        "text": "#ffffff",
        "muted": "#777777",
        "header": "bold #ffffff",
        "metric_value": "bold #ffffff",
        "metric_label": "#dddddd",
    },
}


class ThemeManager:
    """
    Manages active Rich theme definitions and palette lookups.
    """

    def __init__(self, theme_name: str = "nord", custom_themes_dir: Optional[Path] = None) -> None:
        self.theme_name = theme_name.lower()
        self.custom_dir = custom_themes_dir or (Path.home() / ".inferenceos" / "themes")
        self.palette = self._resolve_palette(self.theme_name)

    def _resolve_palette(self, name: str) -> Dict[str, str]:
        if name in THEMES_REGISTRY:
            return THEMES_REGISTRY[name]

        # Check custom theme json file in ~/.inferenceos/themes/<name>.json
        if self.custom_dir and self.custom_dir.exists():
            theme_file = self.custom_dir / f"{name}.json"
            if theme_file.exists():
                try:
                    with open(theme_file, "r", encoding="utf-8") as f:
                        custom_palette = json.load(f)
                    base = THEMES_REGISTRY["nord"].copy()
                    base.update(custom_palette)
                    return base
                except Exception:
                    pass

        return THEMES_REGISTRY["nord"]

    def set_theme(self, name: str) -> None:
        """Switch active theme."""
        self.theme_name = name.lower()
        self.palette = self._resolve_palette(self.theme_name)

    def get_rich_theme(self) -> Theme:
        """Construct Rich Theme object from active palette."""
        p = self.palette
        return Theme({
            "primary": p["primary"],
            "secondary": p["secondary"],
            "accent": p["accent"],
            "success": p["success"],
            "warning": p["warning"],
            "error": p["error"],
            "info": p["info"],
            "border": p["border"],
            "text": p["text"],
            "muted": p["muted"],
            "header": p["header"],
            "metric.val": p["metric_value"],
            "metric.lbl": p["metric_label"],
        })

    def color(self, key: str, default: str = "white") -> str:
        """Get color code by key."""
        return self.palette.get(key, default)


_theme_manager_instance: Optional[ThemeManager] = None


def get_theme(name: str = "nord") -> ThemeManager:
    global _theme_manager_instance
    if _theme_manager_instance is None or _theme_manager_instance.theme_name != name.lower():
        _theme_manager_instance = ThemeManager(name)
    return _theme_manager_instance
