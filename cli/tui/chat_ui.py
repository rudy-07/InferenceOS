"""
chat_ui.py
-----------
Interactive TUI Chat Interface for InferenceOS.

Renders streaming conversation, live top status panel, slash command palette,
and prompt input with rich markdown formatting and history.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .live_status import LiveStatusPanel
from .themes import get_theme
from cli.core.config_manager import get_config_manager
from cli.core.model_registry import get_model_registry
from cli.core.profile_manager import get_profile_manager

import profiler
from layer_placement import ModelDescriptor, PlacementEngine
from inference_runtime import InferenceSession, RuntimeConfig


SLASH_COMMANDS = {
    "/help": "Display interactive help menu & available slash commands",
    "/model": "Display or change current GGUF model",
    "/models": "List registered models in library",
    "/backend": "Switch inference backend (auto, vulkan, cuda, cpu)",
    "/reload": "Reload model and placement plan",
    "/reset": "Reset chat context and session statistics",
    "/history": "Show current session message history",
    "/save": "Save conversation session to file",
    "/load": "Load conversation session from file",
    "/export": "Export session as Markdown / JSON",
    "/context": "Show context usage & window size",
    "/stats": "Display session inference statistics",
    "/telemetry": "Display telemetry summary & metrics",
    "/system": "Display system resource overview",
    "/hardware": "Display hardware device inventory",
    "/memory": "Display VRAM/RAM allocation details",
    "/placement": "Display layer placement distribution",
    "/profile": "Switch active performance profile",
    "/config": "Display current configuration parameters",
    "/theme": "Switch console visual theme",
    "/clear": "Clear terminal screen",
    "/quit": "Exit InferenceOS chat mode",
}


class ChatInterface:
    """
    Manages interactive TUI chat session.
    """

    def __init__(self, model_query: Optional[str] = None, theme_name: str = "nord") -> None:
        self.console = Console()
        self.theme_mgr = get_theme(theme_name)
        self.config_mgr = get_config_manager()
        self.model_registry = get_model_registry()
        self.profile_mgr = get_profile_manager()
        self.live_panel = LiveStatusPanel(theme_name)

        self.history: List[Dict[str, str]] = []
        self.model_path: Optional[Path] = None
        self.session: Optional[InferenceSession] = None
        self.hw_profile: Dict[str, Any] = {}
        self.plan: Optional[Any] = None

        # Resolve initial model
        if model_query:
            self.model_path = self.model_registry.resolve_model_path(model_query)

        if not self.model_path:
            # Auto discover from models registry, prioritizing GGUF for native chat engine
            models = self.model_registry.list_models()
            ggufs = [m for m in models if str(m.get("format", "")).lower() == "gguf" and Path(m.get("location", "")).exists()]
            if ggufs:
                self.model_path = Path(ggufs[0]["location"])
            elif models:
                self.model_path = Path(models[0]["location"])

    def _initialize_engine(self) -> bool:
        """Initialize hardware profile, placement plan, and InferenceSession."""
        if not self.model_path or not self.model_path.exists():
            self.console.print(f"[bold red]Error: No valid model file selected or found.[/bold red]")
            return False

        try:
            self.console.print(f"[cyan]Initializing InferenceOS Runtime Engine for [bold]{self.model_path.name}[/bold]...[/cyan]")
            sys_res = profiler.get_system_resources()
            self.hw_profile = sys_res.to_dict() if hasattr(sys_res, "to_dict") else sys_res

            # Generate placement plan
            from orchestrator.gguf_parser import read_gguf_metadata
            try:
                gguf_meta = read_gguf_metadata(self.model_path)
            except Exception:
                gguf_meta = {"arch": "llama", "num_layers": 32, "max_context_length": 4096}

            model_desc = ModelDescriptor.from_gguf_metadata(
                metadata=gguf_meta,
                model_size_bytes=self.model_path.stat().st_size,
                model_name=self.model_path.stem,
            )

            placement_engine = PlacementEngine(hw_profile=self.hw_profile)
            ctx_len = self.config_mgr.get("memory.context_length", 4096)
            self.plan = placement_engine.generatePlacementPlan(model=model_desc, context_length=ctx_len)

            # Runtime config
            cfg_kwargs = self.config_mgr.get_runtime_config_kwargs()
            runtime_cfg = RuntimeConfig.from_hw_profile(self.hw_profile, **cfg_kwargs)

            # Override GPU layers if specified in config
            override_gpu = self.config_mgr.get("runtime.gpu_layers")
            if override_gpu is not None:
                from run_e2e_integration import apply_gpu_layers_override
                self.plan = apply_gpu_layers_override(self.plan, override_gpu)

            self.session = InferenceSession(
                model_path=self.model_path,
                plan=self.plan,
                config=runtime_cfg,
                hw_profile=self.hw_profile,
            )
            return True

        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize runtime session: {e}[/bold red]")
            return False

    def run(self) -> None:
        """Launch main interactive chat loop."""
        self.console.clear()
        self.console.print(self._build_welcome_banner())

        if not self._initialize_engine():
            self.console.print("[yellow]Type /model to select a model or /help for commands.[/yellow]")

        while True:
            try:
                # Print live status panel
                self._render_top_panel()

                # User Prompt Input
                prompt = self.console.input("\n[bold cyan]InferenceOS ❯ [/bold cyan]").strip()

                if not prompt:
                    continue

                if prompt.startswith("/"):
                    if not self._handle_slash_command(prompt):
                        break
                    continue

                if not self.session:
                    self.console.print("[red]No active model session. Use /model <path> to load a model.[/red]")
                    continue

                # Run inference stream
                self._process_user_prompt(prompt)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[bold yellow]Exiting InferenceOS chat session. Goodbye![/bold yellow]")
                break

    def _render_top_panel(self) -> None:
        """Render live top status panel."""
        if not self.session or not self.plan:
            return

        model_name = self.model_path.name if self.model_path else "No Model"
        backend = self.session.backend_info.name if self.session else "AUTO"
        hardware_str = f"CPU ({self.hw_profile.get('cpu', {}).get('logical_cores', 4)}c)"
        if self.hw_profile.get("gpus"):
            hardware_str += f" + {self.hw_profile['gpus'][0].get('name', 'GPU')}"

        placement_str = f"{self.plan.n_gpu_layers} GPU / {self.plan.n_cpu_layers} CPU"
        pred_vram = self.plan.estimated_vram_bytes / (1024**2)
        pred_ram = self.plan.estimated_ram_bytes / (1024**2)

        last_res = self.session.last_result
        gen_tps = last_res.stats.eval_tps if last_res else 0.0
        prompt_tps = last_res.stats.prompt_eval_tps if last_res else 0.0
        ttft = last_res.stats.prompt_eval_ms if last_res else 0.0

        ctx_used = int(sum(len(m.get("content", "").split()) * 1.3 for m in self.history))

        panel = self.live_panel.render(
            model_name=model_name,
            backend=backend,
            hardware_str=hardware_str,
            context_used=ctx_used,
            context_max=self.plan.context_length if self.plan else 4096,
            placement_str=placement_str,
            pred_vram_mb=pred_vram,
            act_vram_mb=0.0,  # Auto queries real live OS VRAM
            pred_ram_mb=pred_ram,
            act_ram_mb=0.0,   # Auto queries real live OS RAM
            gen_tps=gen_tps,
            prompt_tps=prompt_tps,
            ttft_ms=ttft,
            cpu_util_pct=0.0, # Auto queries real live OS CPU %
            gpu_util_pct=0.0, # Auto queries real live OS GPU %
            profile=self.config_mgr.get("profiles.active_profile", "balanced"),
        )
        self.console.print(panel)

    def _process_user_prompt(self, prompt: str) -> None:
        """Stream response for prompt."""
        self.history.append({"role": "user", "content": prompt})

        self.console.print("\n[bold green]Assistant ❯ [/bold green]", end="")

        accumulated_text = ""

        def on_token(text: str) -> None:
            nonlocal accumulated_text
            accumulated_text += text
            sys.stdout.write(text)
            sys.stdout.flush()

        try:
            res = self.session.run(prompt, on_token=on_token)
            print()  # Newline after stream
            self.history.append({"role": "assistant", "content": res.generated_text})

            # Record telemetry
            from cli.core.telemetry_store import get_telemetry_store
            get_telemetry_store().record_run({
                "model_name": self.model_path.name if self.model_path else "unknown",
                "backend": res.backend,
                "generation_tps": res.stats.eval_tps,
                "prompt_tps": res.stats.prompt_eval_tps,
                "ttft_ms": res.stats.prompt_eval_ms,
                "tokens_generated": res.stats.tokens_generated,
            })

            # Display clean inline runtime stats footer
            st = res.stats
            self.console.print(
                f"[dim]⚡ Gen: {st.eval_tps:.2f} tok/s │ Prompt: {st.prompt_eval_tps:.1f} tok/s │ "
                f"TTFT: {st.prompt_eval_ms:.1f} ms │ Tokens: {st.tokens_generated} │ Backend: {res.backend.upper()}[/dim]"
            )

        except Exception as e:
            self.console.print(f"\n[bold red]Inference Error: {e}[/bold red]")

    def _handle_slash_command(self, cmd_line: str) -> bool:
        """
        Execute slash command. Returns False if user requested quit.
        """
        parts = cmd_line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/quit", "/exit"):
            return False

        elif cmd == "/clear":
            self.console.clear()

        elif cmd == "/help":
            self._print_help_table()

        elif cmd == "/models":
            models = self.model_registry.list_models()
            table = Table(title="InferenceOS Registered Models", box=None)
            table.add_column("Nickname", style="bold cyan")
            table.add_column("Path", style="dim")
            table.add_column("Size (GB)", justify="right")
            for m in models:
                sz_gb = m.get("size_bytes", 0) / (1024**3)
                table.add_row(m["nickname"], m["location"], f"{sz_gb:.2f}")
            self.console.print(table)

        elif cmd == "/model":
            if args:
                new_path = self.model_registry.resolve_model_path(" ".join(args))
                if new_path:
                    self.model_path = new_path
                    self._initialize_engine()
                else:
                    self.console.print(f"[red]Could not resolve model: {' '.join(args)}[/red]")
            else:
                self.console.print(f"[cyan]Current Model: [bold]{self.model_path}[/bold][/cyan]")

        elif cmd == "/backend":
            if args:
                be = args[0].lower()
                self.config_mgr.set("runtime.backend", be)
                self.console.print(f"[green]Backend set to: {be}. Re-initializing engine...[/green]")
                self._initialize_engine()
            else:
                self.console.print(f"[cyan]Current Backend: [bold]{self.config_mgr.get('runtime.backend')}[/bold][/cyan]")

        elif cmd == "/profile":
            if args:
                prof_key = args[0].lower()
                if self.profile_mgr.activate_profile(prof_key):
                    self.console.print(f"[green]Activated profile: {prof_key}[/green]")
                    self._initialize_engine()
                else:
                    self.console.print(f"[red]Unknown profile: {prof_key}[/red]")
            else:
                self.console.print(f"[cyan]Active Profile: [bold]{self.config_mgr.get('profiles.active_profile')}[/bold][/cyan]")

        elif cmd == "/theme":
            if args:
                theme_name = args[0].lower()
                self.theme_mgr = get_theme(theme_name)
                self.config_mgr.set("appearance.theme", theme_name)
                self.console.print(f"[green]Switched theme to: {theme_name}[/green]")
            else:
                self.console.print(f"[cyan]Active Theme: [bold]{self.theme_mgr.theme_name}[/bold][/cyan]")

        elif cmd == "/history":
            self.console.print("[bold yellow]Session Message History:[/bold yellow]")
            for msg in self.history:
                role = "User" if msg["role"] == "user" else "Assistant"
                self.console.print(f"[bold]{role}:[/bold] {msg['content'][:100]}...")

        elif cmd == "/reset":
            self.history = []
            self.console.print("[green]Chat context reset.[/green]")

        elif cmd in ("/hardware", "/system"):
            table = Table(title="Hardware System Profile", box=None)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="bold green")
            cpu = self.hw_profile.get("cpu", {})
            table.add_row("CPU Brand", str(cpu.get("brand", "Unknown")))
            table.add_row("Cores / Threads", f"{cpu.get('physical_cores')} cores / {cpu.get('logical_cores')} threads")
            gpus = self.hw_profile.get("gpus", [])
            if gpus:
                table.add_row("GPU Model", str(gpus[0].get("name")))
                table.add_row("VRAM Total", f"{gpus[0].get('vram_total_mb')} MB")
            self.console.print(table)

        elif cmd == "/placement":
            if self.plan:
                self.console.print(f"[bold cyan]Layer Placement Distribution:[/bold cyan]")
                self.console.print(f"  GPU Layers: {self.plan.n_gpu_layers} / {self.plan.total_layers}")
                self.console.print(f"  CPU Layers: {self.plan.n_cpu_layers}")
                self.console.print(f"  Boundary Crossings: {self.plan.boundary_crossings}")

        else:
            self.console.print(f"[red]Unknown command: {cmd}. Type /help for assistance.[/red]")

        return True

    def _print_help_table(self) -> None:
        table = Table(title="InferenceOS Command Palette & Slash Commands", box=None)
        table.add_column("Command", style="bold cyan")
        table.add_column("Description", style="white")
        for k, v in SLASH_COMMANDS.items():
            table.add_row(k, v)
        self.console.print(table)

    def _build_welcome_banner(self) -> Panel:
        tm = self.theme_mgr
        banner = Text()
        banner.append("  ██╗███╗   ██╗███████╗███████╗██████╗ ███████╗███╗   ██╗██████╗███████╗██████╗ ██╗███████╗\n", style=tm.color("primary"))
        banner.append("  ██║████╗  ██║██╔════╝██╔════╝██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝██╔══██╗██║██╔════╝\n", style=tm.color("primary"))
        banner.append("  ██║██╔██╗ ██║█████╗  █████╗  ██████╔╝█████╗  ██╔██╗ ██║██║     █████╗  ██║  ██║██║███████╗\n", style=tm.color("secondary"))
        banner.append("  ██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══╝  ██║  ██║██║╚════██║\n", style=tm.color("secondary"))
        banner.append("  ██║██║ ╚████║██║     ███████╗██║  ██║███████╗██║ ╚████║╚██████╗███████╗██████╔╝██║███████║\n", style=tm.color("accent"))
        banner.append("  ╚═╝╚═╝  ╚═══╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═════╝ ╚═╝╚══════╝\n", style=tm.color("accent"))
        banner.append("\n          The Operating System for Local AI Inference ── v1.0.0\n", style=f"bold {tm.color('success')}")
        banner.append("          Type /help for slash commands palette ── Type /quit to exit\n", style=tm.color("muted"))

        return Panel(banner, border_style=tm.color("primary"))
