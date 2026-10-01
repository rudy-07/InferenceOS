"""
bottom_toolbar.py
-----------------
Docked Status Ribbon Widget for InferenceOS prompt_toolkit Session.

Renders a live, dynamic status bar anchored at the bottom of the prompt:
- Active Mode badge (Standard Chat, System 1 Reflex, Hybrid Symbiosis)
- Active Model nickname(s) and hardware backend
- Real-time VRAM allocation and generation/reflex metrics
- Hotkey reminders (^O Models, ^M Mode, ^C Cancel, Tab Complete)
"""
from __future__ import annotations

import html
from typing import Any, Callable, Dict, Optional
from prompt_toolkit.formatted_text import HTML, FormattedText


class BottomToolbarBuilder:
    """
    Constructs styled HTML/ANSI toolbar for prompt_toolkit sessions.
    """

    def __init__(self, state_provider: Callable[[], Dict[str, Any]]) -> None:
        self.state_provider = state_provider

    def __call__(self) -> HTML:
        """Invoked by prompt_toolkit on every cursor render."""
        try:
            try:
                state = self.state_provider()
            except Exception:
                state = {}

            mode = str(state.get("mode", "standard")).lower()
            s1_model = html.escape(str(state.get("s1_model_name") or "None"))
            s2_model = html.escape(str(state.get("s2_model_name") or "None"))
            s1_device = html.escape(str(state.get("s1_device", "auto")).upper())
            backend = html.escape(str(state.get("backend", "AUTO")).upper())
            vram_used_mb = float(state.get("vram_used_mb") or 0.0)
            vram_total_mb = float(state.get("vram_total_mb") or 6144.0)
            last_tps = float(state.get("last_tps") or 0.0)
            last_latency_ms = float(state.get("last_latency_ms") or 0.0)
            tau = float(state.get("tau") or 0.85)

            # Mode Badge
            if mode == "system1":
                mode_badge = '<style bg="#d79921" fg="#282828"><b> ⚡ SYSTEM 1 REFLEX </b></style>'
                model_info = f'<b>S1:</b> {s1_model} ({s1_device})'
                perf_info = f'<b>Reflex:</b> {last_latency_ms:.1f}ms | <b>0 KV</b>'
            elif mode == "hybrid":
                mode_badge = '<style bg="#b16286" fg="#ffffff"><b> ⚡🧠 HYBRID SYMBIOSIS </b></style>'
                model_info = f'<b>S1:</b> {s1_model} ({s1_device}) │ <b>S2:</b> {s2_model} │ <b>τ:</b> {tau:.2f}'
                if last_tps > 0:
                    perf_info = f'<b>Speed:</b> {last_tps:.1f} t/s │ <b>Lat:</b> {last_latency_ms:.1f}ms'
                else:
                    perf_info = f'<b>Reflex:</b> {last_latency_ms:.1f}ms'
            else:
                mode_badge = '<style bg="#458588" fg="#ffffff"><b> 💬 STANDARD CHAT </b></style>'
                model_info = f'<b>Model:</b> {s2_model} │ <b>Backend:</b> {backend}'
                perf_info = f'<b>Speed:</b> {last_tps:.1f} t/s' if last_tps > 0 else '<b>Ready</b>'

            # Memory / VRAM
            vram_str = f'{vram_used_mb / 1024:.1f}/{vram_total_mb / 1024:.1f}GB' if vram_total_mb > 0 else f'{vram_used_mb:.0f}MB'
            vram_info = f'<b>VRAM:</b> {vram_str}'

            # Hotkeys breadcrumbs
            hotkeys = '<style fg="#928374">[<b>^O</b> Models │ <b>^T</b> Mode │ <b>Tab</b> Complete │ <b>^C</b> Cancel]</style>'

            html_text = f" {mode_badge}  {model_info} │ {vram_info} │ {perf_info}  {hotkeys} "
            return HTML(html_text)
        except Exception:
            return HTML(" <style bg=\"#458588\" fg=\"#ffffff\"><b> InferenceOS </b></style> [<b>^O</b> Models │ <b>^T</b> Mode] ")

