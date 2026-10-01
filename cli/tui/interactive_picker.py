"""
interactive_picker.py
---------------------
Modern Arrow-Key Interactive Modal Selector for InferenceOS TUI.

Provides rich split-view interactive menus:
- Real-time search/filter bar (type letters to filter candidates)
- Arrow-key navigation (Up/Down or j/k)
- Live preview card showing model architecture, layers, estimated VRAM, and target hardware
- Dedicated pickers for:
    1. Models (/model, /s1, /s2, Ctrl+O)
    2. Cognitive Modes (/mode, Ctrl+M)
    3. Hardware Devices (/s1_device)
    4. Performance Profiles (/profile)
- Non-interactive TTY fallback for automated tests and pipes
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from rich.box import ROUNDED, DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout


def _read_key() -> str:
    """Read a single keypress cross-platform (Windows & Unix)."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            # Extended key (arrows, navigation, function keys)
            ext = msvcrt.getch()
            if ext == b"H":
                return "up"
            elif ext == b"P":
                return "down"
            elif ext == b"K":
                return "left"
            elif ext == b"M":
                return "right"
            elif ext == b"G":
                return "home"
            elif ext == b"O":
                return "end"
            elif ext == b"I":
                return "page_up"
            elif ext == b"Q":
                return "page_down"
            return "unknown"
        elif ch in (b"\r", b"\n"):
            return "enter"
        elif ch == b"\x1b":
            return "escape"
        elif ch == b"\x08":
            return "backspace"
        elif ch == b"\x03":
            return "ctrl_c"
        try:
            return ch.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    else:
        import termios
        import tty
        import select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    seq = sys.stdin.read(1)
                    if seq == "[":
                        r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r2:
                            code = sys.stdin.read(1)
                            if code == "A":
                                return "up"
                            elif code == "B":
                                return "down"
                            elif code == "C":
                                return "right"
                            elif code == "D":
                                return "left"
                            elif code == "H":
                                return "home"
                            elif code == "F":
                                return "end"
                            elif code in ("5", "6", "3"):
                                sys.stdin.read(1)  # consume ~
                                if code == "5":
                                    return "page_up"
                                elif code == "6":
                                    return "page_down"
                return "escape"
            elif ch in ("\r", "\n"):
                return "enter"
            elif ch in ("\x7f", "\x08"):
                return "backspace"
            elif ch == "\x03":
                return "ctrl_c"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class InteractivePicker:
    """
    Renders an interactive selection modal with live search and details preview.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def select(
        self,
        title: str,
        items: List[Dict[str, Any]],
        current_value: Optional[str] = None,
        searchable: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Interactive loop for selecting from a list of structured item dictionaries.

        Each item dict should ideally have:
            - "label": str (Display name)
            - "value": Any (Returned value)
            - "badge": Optional[str] (e.g. "[S1]", "[GGUF]")
            - "details": Optional[Dict[str, str]] (Key-value pairs for preview card)
        """
        if not items:
            return None

        # Check for interactive TTY
        if not sys.stdin.isatty():
            # In automated / piped environments, return current_value match or first item
            if current_value:
                for it in items:
                    if str(it.get("value")) == str(current_value) or str(it.get("label")) == str(current_value):
                        return it
            return items[0]

        query = ""
        selected_idx = 0

        # Try to initially focus the item matching current_value
        if current_value:
            for idx, it in enumerate(items):
                if str(it.get("value")) == str(current_value) or str(it.get("label")) == str(current_value):
                    selected_idx = idx
                    break

        # Flush any stale keystrokes before opening modal
        if sys.platform == "win32":
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()
            except Exception:
                pass

        try:
            self.console.show_cursor(False)
            while True:
                # 1. Filter items based on search query
                if searchable and query:
                    q_lower = query.lower()
                    filtered = [
                        it for it in items
                        if q_lower in it.get("label", "").lower()
                        or q_lower in str(it.get("badge", "")).lower()
                        or q_lower in str(it.get("value", "")).lower()
                    ]
                else:
                    filtered = list(items)

                if selected_idx >= len(filtered):
                    selected_idx = max(0, len(filtered) - 1)

                # 2. Render UI
                self.console.clear()

                # Header Panel
                header_text = Text()
                header_text.append(f"❯ {title}\n", style="bold cyan")
                if searchable:
                    header_text.append("Filter: ", style="dim")
                    header_text.append(f"{query}█", style="bold yellow")
                else:
                    header_text.append("Use [↑/↓] or [j/k] to navigate, [1-9] to select, [Enter] to confirm, [Esc/q] to cancel", style="dim")

                self.console.print(Panel(header_text, box=ROUNDED, border_style="cyan"))

                # Split-view table
                split_table = Table(box=None, show_header=False, expand=True, pad_edge=False)
                split_table.add_column("List", ratio=3)
                split_table.add_column("Details", ratio=2)

                # Left side: items list
                list_table = Table(box=ROUNDED, border_style="bright_black", expand=True)
                list_table.add_column("Model / Option", style="bold")
                list_table.add_column("Tag", justify="right", style="cyan")

                if not filtered:
                    list_table.add_row("[dim italic]No matching items (press Backspace)[/dim italic]", "")
                else:
                    # Show window of items around selected_idx
                    max_show = 12
                    start_i = max(0, selected_idx - max_show // 2)
                    end_i = min(len(filtered), start_i + max_show)
                    if end_i - start_i < max_show:
                        start_i = max(0, end_i - max_show)

                    for i in range(start_i, end_i):
                        it = filtered[i]
                        is_active = (i == selected_idx)
                        prefix = "❯ " if is_active else "  "
                        label = it.get("label", str(it.get("value", "")))
                        badge = it.get("badge", "")

                        if is_active:
                            row_label = f"[bold green]{prefix}{label}[/bold green]"
                            row_badge = f"[bold green]{badge}[/bold green]"
                        else:
                            row_label = f"{prefix}{label}"
                            row_badge = f"[dim]{badge}[/dim]"

                        list_table.add_row(row_label, row_badge)

                # Right side: detail card for currently selected item
                details_content = Text()
                if filtered and 0 <= selected_idx < len(filtered):
                    active_item = filtered[selected_idx]
                    details = active_item.get("details", {})
                    if details:
                        for k, v in details.items():
                            details_content.append(f"{k}: ", style="bold cyan")
                            details_content.append(f"{v}\n", style="white")
                    else:
                        details_content.append(active_item.get("label", ""), style="bold yellow")
                        details_content.append("\n\nReady for execution.")
                else:
                    details_content.append("[dim]Select an option to view hardware and runtime specifications.[/dim]")

                details_panel = Panel(
                    details_content,
                    title="[bold yellow]Specification Preview[/bold yellow]",
                    box=ROUNDED,
                    border_style="yellow",
                    expand=True,
                )

                split_table.add_row(list_table, details_panel)
                self.console.print(split_table)

                # Footer navigation hints
                if searchable:
                    self.console.print(
                        "[dim] [↑/↓ / j/k] Navigate  •  [Enter] Confirm  •  [Backspace] Delete Filter  •  [Esc] Cancel[/dim]"
                    )
                else:
                    self.console.print(
                        "[dim] [↑/↓ / j/k] Navigate  •  [1-9] Quick Jump  •  [Enter] Confirm  •  [Esc / q] Cancel[/dim]"
                    )

                # 3. Read keypress
                try:
                    key = _read_key()
                except (KeyboardInterrupt, Exception):
                    return None

                if not key or key in ("escape", "ctrl_c"):
                    return None
                elif key in ("enter", "\r", "\n"):
                    if filtered and 0 <= selected_idx < len(filtered):
                        return filtered[selected_idx]
                    # If query returned no items, don't abruptly close on enter
                    continue
                elif key in ("up", "k"):
                    if filtered:
                        selected_idx = (selected_idx - 1) % len(filtered)
                elif key in ("down", "j"):
                    if filtered:
                        selected_idx = (selected_idx + 1) % len(filtered)
                elif key in ("home", "page_up"):
                    if filtered:
                        selected_idx = 0 if key == "home" else max(0, selected_idx - 6)
                elif key in ("end", "page_down"):
                    if filtered:
                        selected_idx = len(filtered) - 1 if key == "end" else min(len(filtered) - 1, selected_idx + 6)
                elif key == "backspace":
                    if searchable and query:
                        query = query[:-1]
                        selected_idx = 0
                elif len(key) == 1 and key.isprintable():
                    if not searchable:
                        if key in ("q", "Q"):
                            return None
                        elif key.isdigit() and 1 <= int(key) <= len(filtered):
                            selected_idx = int(key) - 1
                    else:
                        query += key
                        selected_idx = 0
        finally:
            self.console.show_cursor(True)


# ---------------------------------------------------------------------------
# High-Level Domain Selectors
# ---------------------------------------------------------------------------

def pick_model(
    model_registry: Any,
    filter_kind: Optional[str] = None,
    current_path: Optional[Union[str, Path]] = None,
    console: Optional[Console] = None,
) -> Optional[Path]:
    """
    Interactive modal to select a model from the registry.
    """
    raw_models = model_registry.list_models()
    items: List[Dict[str, Any]] = []

    current_str = str(current_path) if current_path else ""

    for m in raw_models:
        fmt = str(m.get("format", "")).lower()
        is_s1 = (fmt == "system1")

        if filter_kind == "system1" and not is_s1:
            continue
        if filter_kind == "system2" and is_s1:
            continue

        nick = m.get("nickname", "Unnamed")
        loc = m.get("location", "")
        sz_bytes = m.get("size_bytes", 0)
        sz_gb = sz_bytes / (1024**3)

        badge = "[S1 Reflex]" if is_s1 else f"[S2 {fmt.upper()}]"

        details = {
            "Nickname": nick,
            "Cognitive Paradigm": "System 1 Non-Autoregressive Reflex" if is_s1 else "System 2 Autoregressive LLM",
            "Format": fmt.upper(),
            "File Size": f"{sz_gb:.2f} GB ({sz_bytes:,} bytes)",
            "KV Cache Footprint": "0 MB (Non-autoregressive)" if is_s1 else "Dynamic (~256 - 1,024 MB)",
            "Target Hardware": "DirectML / CUDA GPU or CPU Pinned" if is_s1 else "Vulkan / CUDA GPU Offloaded",
            "File Path": str(loc),
        }

        items.append({
            "label": nick,
            "value": loc,
            "badge": badge,
            "details": details,
        })

    title = f"Model Selector ({filter_kind.upper() if filter_kind else 'ALL REGISTERED'})"
    picker = InteractivePicker(console)
    chosen = picker.select(title, items, current_value=current_str)

    if chosen and chosen.get("value"):
        return Path(chosen["value"])
    return None


def pick_mode(
    current_mode: str = "standard",
    console: Optional[Console] = None,
) -> Optional[str]:
    """
    Interactive modal to switch cognitive mode.
    """
    items = [
        {
            "label": "Standard Chat",
            "value": "standard",
            "badge": "[Autoregressive LLM]",
            "details": {
                "Mode": "Standard Chat",
                "Cognitive Architecture": "System 2 Sequential Autoregressive Generation",
                "Supported Models": "GGUF, SafeTensors, ONNX, PyTorch",
                "Hardware Allocation": "Offload layers to GPU VRAM via Vulkan/CUDA",
                "Best Used For": "Deep conversational reasoning, code generation, creative writing",
            },
        },
        {
            "label": "System 1 Reflex",
            "value": "system1",
            "badge": "[Sub-150ms ⚡]",
            "details": {
                "Mode": "System 1 Reflex",
                "Cognitive Architecture": "Non-Autoregressive Single Forward-Pass Encoder",
                "KV Cache Footprint": "0 MB (Zero KV Cache)",
                "Supported Models": "Laya (mmBERT encoder heads), Kev (Qwen LoRA pointer heads)",
                "Hardware Allocation": "100% Whole-model placement on GPU or CPU",
                "Best Used For": "Sub-150ms immediate decisions, safety guardrails, ranking, choice evaluation",
            },
        },
        {
            "label": "Hybrid Symbiosis ⚡🧠",
            "value": "hybrid",
            "badge": "[Kahneman Fast/Slow]",
            "details": {
                "Mode": "Hybrid Symbiosis",
                "Cognitive Architecture": "Bi-directional Kahneman Fast/Slow Cognitive Loop",
                "Reflex Stage": "Instant sub-150ms System 1 pass evaluated first",
                "Escalation Stage": "Auto-escalates to System 2 Teacher if confidence < tau (default: 0.85)",
                "Hardware Topology": "Heterogeneous Symbiosis (S2 on Vulkan GPU + S1 on CPU or DirectML)",
                "Best Used For": "Real-time robotics, UI automation, agentic workflows with safety guarantees",
            },
        },
    ]

    picker = InteractivePicker(console)
    chosen = picker.select("Cognitive Execution Mode", items, current_value=current_mode, searchable=False)
    if chosen:
        return chosen["value"]
    return None


def pick_device(
    current_device: str = "auto",
    console: Optional[Console] = None,
) -> Optional[str]:
    """
    Interactive modal to switch System 1 hardware device.
    """
    items = [
        {
            "label": "Auto-Placement",
            "value": "auto",
            "badge": "[Recommended]",
            "details": {
                "Device": "Automatic Resolution",
                "Strategy": "Selects GPU (CUDA / DirectML) if VRAM available, else CPU pinned RAM",
                "Contention Protection": "Balances VRAM headroom between System 1 and System 2",
            },
        },
        {
            "label": "GPU Acceleration",
            "value": "gpu",
            "badge": "[DirectML / CUDA]",
            "details": {
                "Device": "Dedicated GPU VRAM",
                "Backend": "DirectML (AMD Radeon / Intel) or native CUDA (NVIDIA)",
                "Latency Profile": "Peak Reflex Speed (~130ms on RX 5600M, ~15-35ms on NVIDIA)",
            },
        },
        {
            "label": "CPU Pinned Threads",
            "value": "cpu",
            "badge": "[Zero Contention]",
            "details": {
                "Device": "Multi-threaded CPU RAM",
                "Backend": "PyTorch CPU with AVX / OpenMP vectorization",
                "Advantage": "Leaves 100% of GPU compute and VRAM for heavy System 2 generation",
            },
        },
    ]

    picker = InteractivePicker(console)
    chosen = picker.select("System 1 Hardware Placement Device", items, current_value=current_device, searchable=False)
    if chosen:
        return chosen["value"]
    return None


def pick_profile(
    current_profile: str = "balanced",
    console: Optional[Console] = None,
) -> Optional[str]:
    """
    Interactive modal to switch performance profile.
    """
    items = [
        {
            "label": "Balanced",
            "value": "balanced",
            "badge": "[Default]",
            "details": {
                "Profile": "Balanced Performance",
                "KV Cache": "Moderate compression",
                "Latency": "Balanced TTFT and generation speed",
            },
        },
        {
            "label": "Low-Latency",
            "value": "low-latency",
            "badge": "[Fast TTFT]",
            "details": {
                "Profile": "Low Latency Reflex",
                "Priority": "Instant Time-To-First-Token (TTFT)",
                "Batch Size": "Optimized for single-user conversational responsiveness",
            },
        },
        {
            "label": "High-Throughput",
            "value": "high-throughput",
            "badge": "[Max TPS]",
            "details": {
                "Profile": "High Throughput Generation",
                "Priority": "Max continuous tokens per second",
                "VRAM Utilization": "Aggressive layer offloading and caching",
            },
        },
    ]

    picker = InteractivePicker(console)
    chosen = picker.select("Performance Profile", items, current_value=current_profile, searchable=False)
    if chosen:
        return chosen["value"]
    return None
