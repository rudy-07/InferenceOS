"""
completer.py
------------
Dynamic Floating Slash-Command and Argument Completer for InferenceOS TUI.

Provides rich autocomplete suggestions for prompt_toolkit:
- Live floating dropdown menu under cursor when typing '/'
- Fuzzy command matching (e.g. '/m' -> '/mode', '/model', '/models', '/memory')
- Sub-argument completions (e.g. '/mode standard', '/s1_device gpu', '/escalate on')
- Dynamic model nickname completions queried from ModelRegistry for '/model', '/s1', '/s2'
- Clean category tags and descriptive subtitles
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


# Command definition registry with metadata and category tags
COMMAND_METADATA: Dict[str, Dict[str, Any]] = {
    "/mode": {
        "desc": "Switch cognitive mode (standard, system1, hybrid)",
        "category": "Cognition",
        "sub_args": ["standard", "system1", "hybrid"],
        "sub_meta": {
            "standard": "Autoregressive LLM chat (System 2)",
            "system1": "Fast reflex decision agent (0 KV cache)",
            "hybrid": "Kahneman Fast/Slow deliberate symbiosis ⚡",
        },
    },
    "/model": {
        "desc": "Select active model from interactive modal or query",
        "category": "Model",
        "dynamic_models": "all",
    },
    "/s1": {
        "desc": "Select System 1 reflex model (Laya / Kev)",
        "category": "Model",
        "dynamic_models": "system1",
    },
    "/s1_device": {
        "desc": "Configure System 1 execution hardware [auto|gpu|cpu]",
        "category": "Hardware",
        "sub_args": ["auto", "gpu", "cpu"],
        "sub_meta": {
            "auto": "VRAM-aware automatic hardware placement",
            "gpu": "DirectML / CUDA GPU acceleration",
            "cpu": "Multi-threaded CPU pinned RAM",
        },
    },
    "/device": {
        "desc": "Alias for /s1_device",
        "category": "Hardware",
        "sub_args": ["auto", "gpu", "cpu"],
        "sub_meta": {
            "auto": "VRAM-aware automatic hardware placement",
            "gpu": "DirectML / CUDA GPU acceleration",
            "cpu": "Multi-threaded CPU pinned RAM",
        },
    },
    "/s2": {
        "desc": "Select System 2 teacher model (GGUF / Multi-format)",
        "category": "Model",
        "dynamic_models": "system2",
    },
    "/models": {
        "desc": "List registered models grouped by S1 and S2",
        "category": "Model",
    },
    "/escalate": {
        "desc": "Toggle System 2 escalation in Hybrid mode [on|off]",
        "category": "Cognition",
        "sub_args": ["on", "off"],
        "sub_meta": {
            "on": "Escalate to teacher if confidence < tau",
            "off": "Reflex-only execution without escalation",
        },
    },
    "/tau": {
        "desc": "Set escalation confidence threshold (e.g. 0.85)",
        "category": "Cognition",
        "sub_args": ["0.70", "0.80", "0.85", "0.90", "0.95"],
        "sub_meta": {
            "0.70": "Permissive reflex (infrequent escalation)",
            "0.80": "Balanced confidence gate",
            "0.85": "Recommended default threshold",
            "0.90": "Strict confidence gate",
            "0.95": "High-certainty escalation filter",
        },
    },
    "/teacher": {
        "desc": "Set System 2 teacher model or remote URL endpoint",
        "category": "Cognition",
        "dynamic_models": "system2",
    },
    "/dagger": {
        "desc": "Manage DAgger dataset logging [status|on|off|<path>]",
        "category": "Distillation",
        "sub_args": ["status", "on", "off"],
        "sub_meta": {
            "status": "Check active DAgger logging state & file",
            "on": "Enable logging teacher deliberations",
            "off": "Disable DAgger dataset logging",
        },
    },
    "/decide": {
        "desc": "Interactively formulate a decision on custom criteria",
        "category": "Reflex",
    },
    "/guardrail": {
        "desc": "Run sub-15ms System 1 safety guardrail check",
        "category": "Safety",
    },
    "/tool": {
        "desc": "Evaluate options list via fast System 1 tool",
        "category": "Reflex",
    },
    "/backend": {
        "desc": "Switch inference backend (auto, vulkan, cuda, cpu)",
        "category": "Hardware",
        "sub_args": ["auto", "vulkan", "cuda", "cpu"],
        "sub_meta": {
            "auto": "Select optimal hardware backend",
            "vulkan": "AMD/Intel/Cross-vendor Vulkan backend",
            "cuda": "NVIDIA CUDA acceleration",
            "cpu": "CPU fallback compute",
        },
    },
    "/profile": {
        "desc": "Switch active performance profile",
        "category": "Hardware",
        "sub_args": ["low-latency", "balanced", "high-throughput"],
        "sub_meta": {
            "low-latency": "Prioritize instant token response time",
            "balanced": "Balanced memory and generation speed",
            "high-throughput": "Maximize batch/concurrency tokens/s",
        },
    },
    "/theme": {
        "desc": "Switch console visual theme",
        "category": "UI",
        "sub_args": ["nord", "cyberpunk", "dracula", "catppuccin", "gruvbox", "monokai", "dark", "light", "minimal"],
    },
    "/doctor": {
        "desc": "Run hardware, driver, and environment health checks",
        "category": "Diagnostics",
    },
    "/benchmark": {
        "desc": "Run performance benchmark on active model",
        "category": "Diagnostics",
    },
    "/inspect": {
        "desc": "Inspect model architecture, parameters, and metadata",
        "category": "Model",
    },
    "/health": {
        "desc": "Display real-time thermal, GPU, and memory pressure",
        "category": "Diagnostics",
    },
    "/kv": {
        "desc": "Inspect KV cache compression and memory efficiency",
        "category": "Memory",
    },
    "/budget": {
        "desc": "Inspect dynamic memory budget allocations",
        "category": "Memory",
    },
    "/memory": {
        "desc": "Display VRAM/RAM allocation details",
        "category": "Hardware",
    },
    "/hardware": {
        "desc": "Display hardware device inventory",
        "category": "Hardware",
    },
    "/placement": {
        "desc": "Display layer placement distribution",
        "category": "Hardware",
    },
    "/reload": {
        "desc": "Reload model and placement plan",
        "category": "System",
    },
    "/reset": {
        "desc": "Reset chat context and session statistics",
        "category": "Chat",
    },
    "/history": {
        "desc": "Show current session message and decision history",
        "category": "Chat",
    },
    "/save": {
        "desc": "Save conversation session to file",
        "category": "Chat",
    },
    "/load": {
        "desc": "Load conversation session from file",
        "category": "Chat",
    },
    "/export": {
        "desc": "Export session as Markdown / JSON",
        "category": "Chat",
        "sub_args": ["md", "json"],
    },
    "/context": {
        "desc": "Show context usage & window size",
        "category": "Memory",
    },
    "/stats": {
        "desc": "Display session inference statistics",
        "category": "Diagnostics",
    },
    "/telemetry": {
        "desc": "Display telemetry summary & metrics",
        "category": "Diagnostics",
    },
    "/system": {
        "desc": "Display system resource overview",
        "category": "System",
    },
    "/settings": {
        "desc": "Open interactive settings & configuration menu",
        "category": "System",
    },
    "/config": {
        "desc": "Display current configuration parameters",
        "category": "System",
    },
    "/clear": {
        "desc": "Clear terminal screen",
        "category": "UI",
    },
    "/help": {
        "desc": "Display interactive help menu & available slash commands",
        "category": "Help",
    },
    "/quit": {
        "desc": "Exit InferenceOS interactive session",
        "category": "System",
    },
    "/exit": {
        "desc": "Exit InferenceOS interactive session",
        "category": "System",
    },
}


class InferenceOSCompleter(Completer):
    """
    Rich auto-completer for InferenceOS slash commands, sub-arguments, and registered models.
    """

    def __init__(self, registry_getter: Optional[Callable[[], Any]] = None) -> None:
        self.registry_getter = registry_getter

    def get_completions(self, document: Document, complete_event: Any) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip("\ufeff\u200b\ufffe")
        stripped = text.strip()

        # If empty or not starting with slash, don't show slash command menu
        if not text.startswith("/"):
            return

        parts = text.split(maxsplit=1)
        cmd_part = parts[0]
        has_space = len(parts) > 1 or text.endswith(" ")

        # Case 1: Still typing the slash command itself (e.g. "/m" or "/mode")
        if not has_space:
            match_str = cmd_part.lower()
            for cmd, info in sorted(COMMAND_METADATA.items()):
                if cmd.lower().startswith(match_str) or match_str == "/":
                    meta = f"[{info['category']}] {info['desc']}"
                    # Append a trailing space to convenience completion
                    insert_text = f"{cmd} "
                    yield Completion(
                        text=insert_text,
                        start_position=-len(cmd_part),
                        display=cmd,
                        display_meta=meta,
                    )
            return

        # Case 2: Command is complete, user is typing sub-arguments (e.g. "/mode hy" or "/s1_device g")
        cmd_key = cmd_part.lower()
        if cmd_key not in COMMAND_METADATA:
            return

        info = COMMAND_METADATA[cmd_key]
        arg_prefix = parts[1] if len(parts) > 1 else ""

        # Sub-arguments defined in metadata (e.g. 'standard', 'hybrid', 'gpu')
        if "sub_args" in info:
            sub_args = info["sub_args"]
            sub_meta = info.get("sub_meta", {})
            for arg in sub_args:
                if arg.lower().startswith(arg_prefix.lower()):
                    meta = sub_meta.get(arg, f"{cmd_key} {arg}")
                    yield Completion(
                        text=arg,
                        start_position=-len(arg_prefix),
                        display=arg,
                        display_meta=meta,
                    )

        # Dynamic model suggestions from ModelRegistry (for /model, /s1, /s2, /teacher)
        if "dynamic_models" in info and self.registry_getter:
            try:
                reg = self.registry_getter()
                if reg:
                    models = reg.list_models()
                    target_kind = info["dynamic_models"]
                    for m in models:
                        fmt = str(m.get("format", "")).lower()
                        is_s1 = fmt == "system1"
                        if target_kind == "system1" and not is_s1:
                            continue
                        if target_kind == "system2" and is_s1:
                            continue

                        nick = m.get("nickname", "")
                        if not nick:
                            continue

                        if nick.lower().startswith(arg_prefix.lower()):
                            sz_gb = m.get("size_bytes", 0) / (1024**3)
                            kind_str = "S1 Reflex" if is_s1 else "S2 LLM"
                            meta = f"[{kind_str}] {fmt.upper()} • {sz_gb:.2f} GB"
                            yield Completion(
                                text=nick,
                                start_position=-len(arg_prefix),
                                display=nick,
                                display_meta=meta,
                            )
            except Exception:
                pass
