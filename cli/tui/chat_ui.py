"""
chat_ui.py
-----------
Interactive TUI Interface for InferenceOS.

Supports:
1. Standard Mode: Autoregressive text generation & streaming chat (GGUF, Multi-Format).
2. System 1 Mode: Fast non-autoregressive decision reflex inference (Laya, Kev) with 0 KV cache.
3. Hybrid Symbiosis Mode: Bi-directional Kahneman Fast/Slow cognitive loop:
   - Instant 15ms reflex forward pass
   - Automatic System 2 teacher escalation on low confidence (< tau) or uncertainty
   - Teacher reasoning synthesis and rationale display
4. Interactive Model Selection:
   - Dual-model assignment (System 1 Reflex + System 2 Teacher)
   - Interactive numbered model pickers (/model, /s1, /s2, /mode)
   - Real-time escalation toggle (/escalate on|off, /tau <float>, /teacher <model>)
   - Instant safety guardrails (/guardrail <action>) and option tool (/tool <opts>)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .live_status import LiveStatusPanel
from .themes import get_theme
from .completer import InferenceOSCompleter
from .interactive_picker import pick_model, pick_mode, pick_device, pick_profile
from .bottom_toolbar import BottomToolbarBuilder
from .chat_renderer import ChatRenderer
from cli.core.config_manager import get_config_manager
from cli.core.model_registry import get_model_registry
from cli.core.profile_manager import get_profile_manager

import profiler
from layer_placement import ModelDescriptor, PlacementEngine
from inference_runtime import InferenceSession, RuntimeConfig, MultiFormatRuntimeEngine
from orchestrator.model_parser import detect_model_format, ModelFormat
from orchestrator.hybrid_orchestrator import HybridOrchestrator, DEFAULT_ESCALATE_TAU


SLASH_COMMANDS = {
    "/help": "Display interactive help menu & available slash commands",
    "/mode": "Switch TUI mode (standard, system1, hybrid) or open mode menu",
    "/s1": "Select or change System 1 reflex model (e.g. laya:v14s, kev)",
    "/s1_device": "Configure System 1 execution hardware device [auto|gpu|cpu]",
    "/device": "Configure System 1 execution hardware device [auto|gpu|cpu]",
    "/s2": "Select or change System 2 teacher model (e.g. Qwen3-0.6B-Q8_0.gguf)",
    "/model": "Select active model from interactive numbered menu or query",
    "/models": "List registered models grouped by System 1 and System 2",
    "/escalate": "Toggle System 2 escalation [on|off] in Hybrid mode",
    "/tau": "Set escalation confidence threshold (e.g. /tau 0.85)",
    "/teacher": "Set System 2 teacher model or remote URL endpoint",
    "/dagger": "Manage DAgger dataset logging [status|on|off|<path>]",
    "/decide": "Interactively formulate a decision on custom criteria",
    "/guardrail": "Run instant 15ms System 1 safety guardrail on an action",
    "/tool": "Evaluate options list via System 1 fast decision tool",
    "/settings": "Open interactive settings & runtime configuration menu",
    "/doctor": "Run hardware, driver, and environment health checks",
    "/benchmark": "Run performance benchmark on active model",
    "/inspect": "Inspect model architecture, parameters, and metadata",
    "/health": "Display real-time thermal, GPU, and memory pressure",
    "/kv": "Inspect KV cache compression and memory efficiency",
    "/budget": "Inspect dynamic memory budget allocations",
    "/backend": "Switch inference backend (auto, vulkan, cuda, cpu)",
    "/reload": "Reload model and placement plan",
    "/reset": "Reset chat context and session statistics",
    "/history": "Show current session message and decision history",
    "/save": "Save conversation session to file",
    "/load": "Load conversation session from file",
    "/export": "Export session as Markdown / JSON",
    "/context": "Show context usage & window size",
    "/stats": "Display session inference statistics",
    "/status": "Display session inference statistics (alias for /stats)",
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
    Manages interactive TUI session across Standard, System 1, and Hybrid modes.
    """

    def __init__(self, model_query: Optional[str] = None, theme_name: str = "nord") -> None:
        self.console = Console()
        self.theme_mgr = get_theme(theme_name)
        self.config_mgr = get_config_manager()
        self.model_registry = get_model_registry()
        self.profile_mgr = get_profile_manager()
        self.live_panel = LiveStatusPanel(theme_name)

        self.history: List[Dict[str, str]] = []
        self.mode = "standard"  # "standard", "system1", "hybrid"

        self.model_path: Optional[Path] = None
        self.s1_model_path: Optional[Path] = None
        self.s2_model_path: Optional[Path] = None
        self.s2_base_url: Optional[str] = None

        self.escalate = True
        self.escalate_tau = DEFAULT_ESCALATE_TAU
        self.s1_device = "auto"  # "auto", "gpu", "cpu"
        self.dagger_log_path: Optional[Path] = None

        self.session: Optional[InferenceSession] = None
        self.hw_profile: Dict[str, Any] = {}
        self.plan: Optional[Any] = None

        self.orchestrator = HybridOrchestrator()

        self.last_decision: Optional[str] = None
        self.last_confidence: float = 0.0
        self.last_latency_ms: float = 0.0
        self.last_escalated: bool = False

        self.renderer = ChatRenderer(self.console)
        self.completer = InferenceOSCompleter(registry_getter=lambda: self.model_registry)
        self.prompt_session = None
        self._init_prompt_session()

        # Auto-discover available models from registry
        self._discover_default_models()

        # Resolve requested model if specified
        if model_query:
            resolved = self.model_registry.resolve_model_path(model_query)
            if resolved and resolved.exists():
                fmt = detect_model_format(resolved)
                if fmt == ModelFormat.SYSTEM1:
                    self.s1_model_path = resolved
                    self.model_path = resolved
                    self.mode = "system1"
                else:
                    self.s2_model_path = resolved
                    self.model_path = resolved
                    self.mode = "standard"

    def _init_prompt_session(self) -> None:
        """Initialize prompt_toolkit session with floating completions, bottom toolbar, and shortcuts."""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
            from prompt_toolkit.history import InMemoryHistory
            from prompt_toolkit.styles import Style

            kb = KeyBindings()

            @kb.add("c-o")
            def _kb_model(event):
                event.app.exit(result="/model")

            @kb.add("c-t")
            def _kb_mode(event):
                event.app.exit(result="/mode")

            @kb.add("c-l")
            def _kb_clear(event):
                event.app.exit(result="/clear")

            @kb.add("c-c")
            def _kb_ctrl_c(event):
                buf = event.app.current_buffer
                if buf.text:
                    buf.text = ""
                else:
                    event.app.exit(result="/exit")

            def _get_toolbar_state():
                act_vram = 0.0
                if self.live_panel:
                    try:
                        _, _, act_vram, _ = self.live_panel._query_live_hardware()
                    except Exception:
                        pass
                if act_vram == 0.0 and self.plan:
                    act_vram = self.plan.estimated_vram_bytes / (1024**2)

                last_res = self.session.last_result if self.session else None
                return {
                    "mode": self.mode,
                    "s1_model_name": self.s1_model_path.name if self.s1_model_path else None,
                    "s2_model_name": self.s2_model_path.name if self.s2_model_path else ("Remote" if self.s2_base_url else "Rule-Teacher"),
                    "s1_device": self.s1_device,
                    "backend": self.session.backend_info.name if self.session else ("SYSTEM1" if self.mode != "standard" else "AUTO"),
                    "vram_used_mb": act_vram,
                    "vram_total_mb": 6144.0,
                    "last_tps": last_res.stats.eval_tps if last_res else 0.0,
                    "last_latency_ms": self.last_latency_ms,
                    "tau": self.escalate_tau,
                }

            self.bottom_toolbar = BottomToolbarBuilder(_get_toolbar_state)
            pt_style = Style.from_dict({
                "s1": "#d79921 bold",
                "hybrid": "#b16286 bold",
                "std": "#458588 bold",
                "prompt": "#83a598 bold",
            })

            try:
                self.prompt_session = PromptSession(
                    completer=self.completer,
                    complete_while_typing=True,
                    bottom_toolbar=self.bottom_toolbar,
                    key_bindings=kb,
                    auto_suggest=AutoSuggestFromHistory(),
                    history=InMemoryHistory(),
                    style=pt_style,
                )
            except Exception:
                from prompt_toolkit.input import DummyInput
                from prompt_toolkit.output import DummyOutput
                self.prompt_session = PromptSession(
                    completer=self.completer,
                    complete_while_typing=True,
                    bottom_toolbar=self.bottom_toolbar,
                    key_bindings=kb,
                    auto_suggest=AutoSuggestFromHistory(),
                    history=InMemoryHistory(),
                    style=pt_style,
                    input=DummyInput(),
                    output=DummyOutput(),
                )
        except Exception:
            self.prompt_session = None

    def _discover_default_models(self) -> None:
        """Scan registry to pre-populate default System 1 and System 2 models."""
        models = self.model_registry.list_models()

        # Find System 1 models
        s1_candidates = [
            m for m in models
            if str(m.get("format", "")).lower() == "system1" and Path(m.get("location", "")).exists()
        ]
        if s1_candidates:
            self.s1_model_path = Path(s1_candidates[0]["location"])

        # Find System 2 models (GGUF first, then other multi-formats)
        ggufs = [
            m for m in models
            if str(m.get("format", "")).lower() == "gguf" and Path(m.get("location", "")).exists()
        ]
        other_s2 = [
            m for m in models
            if str(m.get("format", "")).lower() in ("safetensors", "onnx", "pytorch") and Path(m.get("location", "")).exists()
        ]
        if ggufs:
            self.s2_model_path = Path(ggufs[0]["location"])
            self.model_path = self.s2_model_path
        elif other_s2:
            self.s2_model_path = Path(other_s2[0]["location"])
            self.model_path = self.s2_model_path
        elif s1_candidates:
            self.model_path = self.s1_model_path
            self.mode = "system1"

    def _initialize_engine(self) -> bool:
        """Initialize hardware profile, placement plan, and InferenceSession for active model."""
        if self.mode == "system1":
            if not self.s1_model_path or not self.s1_model_path.exists():
                self.console.print(f"[bold red]Error: No valid System 1 model selected. Use /s1 to choose one.[/bold red]")
                return False
            self.console.print(f"[yellow]✔ System 1 Reflex Engine ready: [bold]{self.s1_model_path.name}[/bold][/yellow]")
            return True

        if self.mode == "hybrid":
            if not self.s1_model_path or not self.s1_model_path.exists():
                self.console.print(f"[bold yellow]Warning: No System 1 model selected for Hybrid mode. Use /s1 to set reflex model.[/bold yellow]")
            if not self.s2_model_path and not self.s2_base_url:
                self.console.print(f"[bold yellow]Notice: No System 2 teacher model selected. Heuristic rule teacher active. Use /s2 to set local teacher.[/bold yellow]")
            self.console.print(f"[magenta]✔ Hybrid Symbiosis Engine ready: S1={self.s1_model_path.name if self.s1_model_path else 'None'} | S2={self.s2_model_path.name if self.s2_model_path else 'Rule-Teacher'}[/magenta]")
            return True

        # Standard Mode (Generative Chat)
        active_p = self.s2_model_path or self.model_path
        if not active_p or not active_p.exists():
            self.console.print(f"[bold red]Error: No valid model file selected or found. Use /model to select one.[/bold red]")
            return False

        try:
            self.console.print(f"[cyan]Initializing InferenceOS Runtime Engine for [bold]{active_p.name}[/bold]...[/cyan]")
            sys_res = profiler.get_system_resources()
            self.hw_profile = sys_res.to_dict() if hasattr(sys_res, "to_dict") else sys_res

            fmt = detect_model_format(active_p)
            if fmt == ModelFormat.GGUF:
                from orchestrator.gguf_parser import read_gguf_metadata
                try:
                    gguf_meta = read_gguf_metadata(active_p)
                except Exception:
                    gguf_meta = {"arch": "llama", "num_layers": 32, "max_context_length": 4096}

                model_desc = ModelDescriptor.from_gguf_metadata(
                    metadata=gguf_meta,
                    model_size_bytes=active_p.stat().st_size,
                    model_name=active_p.stem,
                )

                placement_engine = PlacementEngine(hw_profile=self.hw_profile)
                ctx_len = self.config_mgr.get("memory.context_length", 4096)
                self.plan = placement_engine.generatePlacementPlan(model=model_desc, context_length=ctx_len)

                cfg_kwargs = self.config_mgr.get_runtime_config_kwargs()
                runtime_cfg = RuntimeConfig.from_hw_profile(self.hw_profile, **cfg_kwargs)

                override_gpu = self.config_mgr.get("runtime.gpu_layers")
                if override_gpu is not None:
                    from run_e2e_integration import apply_gpu_layers_override
                    self.plan = apply_gpu_layers_override(self.plan, override_gpu)

                self.session = InferenceSession(
                    model_path=active_p,
                    plan=self.plan,
                    config=runtime_cfg,
                    hw_profile=self.hw_profile,
                )
            else:
                self.session = None

            return True

        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize runtime session: {e}[/bold red]")
            return False

    def run(self) -> None:
        """Launch main interactive TUI loop."""
        self.console.clear()
        self.console.print(self._build_welcome_banner())
        self._initialize_engine()
        self._render_top_panel()

        while True:
            try:
                # Dynamic mode prompt prefix
                if self.mode == "system1":
                    pt_prompt = [("class:s1", "\nInferenceOS [System 1 Reflex ⚡] ❯ ")]
                    raw_prompt = "\nInferenceOS [System 1 Reflex ⚡] ❯ "
                elif self.mode == "hybrid":
                    pt_prompt = [("class:hybrid", "\nInferenceOS [Hybrid Symbiosis ⚡🧠] ❯ ")]
                    raw_prompt = "\nInferenceOS [Hybrid Symbiosis ⚡🧠] ❯ "
                else:
                    pt_prompt = [("class:std", "\nInferenceOS [Standard Chat] ❯ ")]
                    raw_prompt = "\nInferenceOS [Standard Chat] ❯ "

                if sys.stdin.isatty() and self.prompt_session is not None:
                    prompt = self.prompt_session.prompt(pt_prompt).strip()
                else:
                    prompt = self.console.input(raw_prompt).strip()

                prompt = prompt.lstrip("\ufeff\u200b\ufffe").strip()

                if not prompt:
                    continue

                if prompt.startswith("/"):
                    if not self._handle_slash_command(prompt):
                        break
                    continue

                self._process_user_prompt(prompt)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[bold yellow]Exiting InferenceOS interactive session. Goodbye![/bold yellow]")
                break

    def _render_top_panel(self) -> None:
        """Render live top status panel adapted for active mode."""
        model_name = self.model_path.name if self.model_path else "No Model"
        backend = self.session.backend_info.name if self.session else ("SYSTEM1" if self.mode != "standard" else "AUTO")

        hardware_str = f"CPU ({self.hw_profile.get('cpu', {}).get('logical_cores', 4)}c)"
        if self.hw_profile.get("gpus"):
            hardware_str += f" + {self.hw_profile['gpus'][0].get('name', 'GPU')}"

        placement_str = f"{self.plan.n_gpu_layers} GPU / {self.plan.n_cpu_layers} CPU" if self.plan else "Direct Allocation"
        pred_vram = (self.plan.estimated_vram_bytes / (1024**2)) if self.plan else 0.0
        pred_ram = (self.plan.estimated_ram_bytes / (1024**2)) if self.plan else 0.0

        last_res = self.session.last_result if self.session else None
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
            act_vram_mb=0.0,
            pred_ram_mb=pred_ram,
            act_ram_mb=0.0,
            gen_tps=gen_tps,
            prompt_tps=prompt_tps,
            ttft_ms=ttft,
            cpu_util_pct=0.0,
            gpu_util_pct=0.0,
            profile=self.config_mgr.get("profiles.active_profile", "balanced"),
            mode=self.mode,
            s1_model_name=self.s1_model_path.name if self.s1_model_path else None,
            s2_model_name=self.s2_model_path.name if self.s2_model_path else ("Remote" if self.s2_base_url else "Rule-Teacher"),
            escalate_enabled=self.escalate,
            escalate_tau=self.escalate_tau,
            last_decision=self.last_decision,
            last_confidence=self.last_confidence,
            last_latency_ms=self.last_latency_ms,
            last_escalated=self.last_escalated,
        )
        self.console.print(panel)

    def _parse_prompt_to_state_and_questions(self, prompt: str) -> tuple[str, dict]:
        """
        Intelligently parse user input in System 1 or Hybrid mode.
        Supports:
        1. Pure JSON input (e.g. {"state": "...", "questions": {...}})
        2. Explicit options in brackets (e.g. "Options: [PROCEED, CANCEL, WAIT] Is cart valid?")
        3. Freeform text prompt (defaults to YES, NO, UNCERTAIN)
        """
        trimmed = prompt.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                data = json.loads(trimmed)
                if isinstance(data, dict) and "questions" in data:
                    return data.get("state", prompt), data["questions"]
            except Exception:
                pass

        # Check for State: ... Questions: ... pattern (even if single-line or no newline)
        import re
        sq_match = re.search(r'state\s*:\s*(.*?)(?:[\.\n\s]*questions?\s*:\s*(.*))?$', prompt, re.IGNORECASE | re.DOTALL)
        if sq_match and sq_match.group(2):
            raw_state = sq_match.group(1).strip()
            raw_q = sq_match.group(2).strip()
            opt_inside = re.search(r'(?:-\s*)?(\w+)?\s*:\s*\[(.*?)\]', raw_q)
            if opt_inside:
                qid = opt_inside.group(1) or "decision"
                raw_opts = opt_inside.group(2).split(",")
                opts = [o.strip().strip("'\"") for o in raw_opts if o.strip()]
                if opts:
                    crit = {opt: f"Select {opt}" for opt in opts}
                    return raw_state, {
                        qid: {
                            "type": "choice",
                            "instructions": {"goal": f"Choose {qid} for: {raw_state}"},
                            "criteria": crit,
                        }
                    }

        opt_match = re.search(r'(?:options|choices|\w+)\s*:\s*\[(.*?)\]', prompt, re.IGNORECASE)
        if opt_match:
            raw_opts = opt_match.group(1).split(",")
            opts = [o.strip().strip("'\"") for o in raw_opts if o.strip()]
            if opts:
                clean_goal = prompt[:opt_match.start()] + prompt[opt_match.end():]
                clean_goal = clean_goal.strip(" -:\n") or prompt
                crit = {opt: f"Select {opt}" for opt in opts}
                questions = {
                    "decision": {
                        "type": "choice",
                        "instructions": {"goal": clean_goal},
                        "criteria": crit,
                    }
                }
                return clean_goal, questions

        questions = {
            "decision": {
                "type": "choice",
                "instructions": {"goal": prompt},
                "criteria": {
                    "YES": "Affirmative / Proceed",
                    "NO": "Negative / Reject",
                    "UNCERTAIN": "Needs further context",
                },
            }
        }
        return prompt, questions

    def _process_user_prompt(self, prompt: str) -> None:
        """Route user input according to active operation mode."""
        if not prompt or not prompt.strip():
            return
        self.history.append({"role": "user", "content": prompt})

        # -------------------------------------------------------------------
        # Mode 1: System 1 (Non-Autoregressive Decision Reflex)
        # -------------------------------------------------------------------
        if self.mode == "system1":
            if not self.s1_model_path or not self.s1_model_path.exists():
                self.console.print("[red]No System 1 model selected. Use /s1 to pick a model.[/red]")
                return

            state_val, questions = self._parse_prompt_to_state_and_questions(prompt)

            self.console.print(f"\n[bold yellow]⚡ System 1 Reflex Evaluation...[/bold yellow]")
            try:
                s1_dev_arg = None if self.s1_device == "auto" else self.s1_device
                res = self.orchestrator.s1_engine.predict(
                    model_path=self.s1_model_path,
                    state=state_val,
                    questions=questions,
                    device=s1_dev_arg,
                )
                ans = res["answers"].get("decision", {}) or next(iter(res["answers"].values()), {})
                chosen = ans.get("choice", "")
                conf = ans.get("confidence", 0.0)
                lat = res.get("latency_ms", 0.0)

                self.last_decision = chosen
                self.last_confidence = conf
                self.last_latency_ms = lat
                self.last_escalated = False

                self.history.append({"role": "assistant", "content": f"Decision: {chosen} (Confidence: {conf:.2f} | Latency: {lat:.1f}ms)"})
                self.console.print(f"[bold green]• Decision:[/bold green] [bold yellow]{chosen}[/bold yellow] (Confidence: [bold]{conf:.2f}[/bold] | Latency: [bold]{lat:.1f}ms[/bold] | Passes: 1)")
                self.renderer.render_system1_card(res, state_summary=state_val)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]⚡ System 1 reflex halted by user (Ctrl+C).[/yellow]")
                return
            except Exception as e:
                self.console.print(f"[bold red]System 1 Reflex Error: {e}[/bold red]")
            return

        # -------------------------------------------------------------------
        # Mode 2: Hybrid (Bi-Directional S1 Reflex + S2 Teacher Symbiosis)
        # -------------------------------------------------------------------
        if self.mode == "hybrid":
            if not self.s1_model_path or not self.s1_model_path.exists():
                self.console.print("[red]No System 1 model selected for Hybrid mode. Use /s1 to set model.[/red]")
                return

            state_val, questions = self._parse_prompt_to_state_and_questions(prompt)

            self.console.print(f"\n[bold magenta]⚡ Symbiosis Loop: Running System 1 reflex...[/bold magenta]")
            try:
                s1_dev_arg = None if self.s1_device == "auto" else self.s1_device
                res = self.orchestrator.decide_with_escalation(
                    model_path=self.s1_model_path,
                    state=state_val,
                    questions=questions,
                    escalate=self.escalate,
                    escalate_tau=self.escalate_tau,
                    system2_model=self.s2_model_path,
                    system2_base_url=self.s2_base_url,
                    dagger_log_path=self.dagger_log_path,
                    s1_device=s1_dev_arg,
                    system2_session=self.session,
                )

                ans = res["answers"].get("decision", {}) or next(iter(res["answers"].values()), {})
                chosen = ans.get("choice", "")
                conf = ans.get("confidence", 0.0)
                lat = res.get("latency_ms", 0.0)
                is_esc = res.get("escalated", False)

                self.last_decision = chosen
                self.last_confidence = conf
                self.last_latency_ms = lat
                self.last_escalated = is_esc

                self.history.append({"role": "assistant", "content": f"Decision: {chosen} (Confidence: {conf:.2f} | Escalated: {is_esc})"})

                if is_esc:
                    self.console.print(f"[bold magenta]⚡ Escalated to System 2 Teacher[/bold magenta] [dim]Reason: {res.get('escalate_reason')}[/dim]")
                    self.console.print(f"[bold magenta]• Choice (System 2):[/bold magenta] [bold yellow]{chosen}[/bold yellow] (Confidence: [bold]{conf:.2f}[/bold] | S1: {res.get('s1_latency_ms', 0):.1f}ms | S2: {res.get('s2_latency_ms', 0):.1f}ms)")
                    if ans.get("rationale"):
                        self.console.print(f"  [dim italic]Rationale: {ans['rationale']}[/dim italic]")
                else:
                    self.console.print(f"[bold green]✔ Fast Reflex Path ({lat:.1f}ms)[/bold green] [dim]Confidence {conf:.2f} >= tau={self.escalate_tau:.2f}[/dim]")
                    self.console.print(f"[bold green]• Choice (System 1):[/bold green] [bold yellow]{chosen}[/bold yellow] (Confidence: [bold]{conf:.2f}[/bold])")

                self.renderer.render_hybrid_flowcard(res, state_summary=state_val)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]⚡ Hybrid Symbiosis halted by user (Ctrl+C).[/yellow]")
                return
            except Exception as e:
                self.console.print(f"[bold red]Hybrid Symbiosis Error: {e}[/bold red]")
            return

        # -------------------------------------------------------------------
        # Mode 3: Standard (Autoregressive LLM Chat)
        # -------------------------------------------------------------------
        if not self.session:
            if not self._initialize_engine():
                self.console.print("[red]No active model session. Use /model to select a model or /mode to change mode.[/red]")
                return

        self.console.print("\n[bold green]Assistant ❯ [/bold green]", end="")
        accumulated_text = ""

        def on_token(text: str) -> None:
            nonlocal accumulated_text
            accumulated_text += text
            sys.stdout.write(text)
            sys.stdout.flush()

        try:
            res = self.session.run(prompt, on_token=on_token)
            print()
            self.history.append({"role": "assistant", "content": res.generated_text})

            from cli.core.telemetry_store import get_telemetry_store
            get_telemetry_store().record_run({
                "model_name": self.model_path.name if self.model_path else "unknown",
                "backend": res.backend,
                "generation_tps": res.stats.eval_tps,
                "prompt_tps": res.stats.prompt_eval_tps,
                "ttft_ms": res.stats.prompt_eval_ms,
                "tokens_generated": res.stats.tokens_generated,
            })

            st = res.stats
            self.renderer.render_streaming_pill(st, res.backend)

        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]⚡ Streaming halted by user (Ctrl+C). Returning to prompt.[/bold yellow]")
            if accumulated_text:
                self.history.append({"role": "assistant", "content": accumulated_text})
        except Exception as e:
            self.console.print(f"\n[bold red]Inference Error: {e}[/bold red]")

    def _select_model_interactive(self, filter_format: Optional[str] = None) -> Optional[Path]:
        """Render modern arrow-key modal selector for registered models."""
        return pick_model(
            self.model_registry,
            filter_kind=filter_format,
            current_path=self.model_path,
            console=self.console,
        )

    def _select_mode_interactive(self) -> None:
        """Display interactive mode switch modal."""
        chosen = pick_mode(current_mode=self.mode, console=self.console)
        if chosen:
            self.mode = chosen
            self.console.print(f"[green]✔ Switched to {chosen.upper()} Mode.[/green]")
            self._initialize_engine()

    def _handle_slash_command(self, cmd_line: str) -> bool:
        """
        Execute slash command. Returns False if user requested quit.
        """
        clean_line = cmd_line.lstrip("\ufeff\u200b\ufffe").strip()
        parts = clean_line.split()
        if not parts:
            return True
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/quit", "/exit"):
            return False

        elif cmd == "/clear":
            self.console.clear()

        elif cmd == "/help":
            self._print_help_table()

        elif cmd == "/mode":
            if args:
                target_mode = args[0].lower()
                mode_aliases = {
                    "standard": "standard", "s2": "standard", "chat": "standard",
                    "system1": "system1", "s1": "system1", "reflex": "system1",
                    "hybrid": "hybrid", "symbiosis": "hybrid", "fast-slow": "hybrid",
                }
                normalized_mode = mode_aliases.get(target_mode)
                if normalized_mode:
                    self.mode = normalized_mode
                    self.console.print(f"[green]✔ Switched to {normalized_mode.upper()} mode.[/green]")
                    self._initialize_engine()
                else:
                    self.console.print("[yellow]Invalid mode. Choose: standard, system1, or hybrid.[/yellow]")
            else:
                self._select_mode_interactive()

        elif cmd == "/s1":
            if args:
                p = self.model_registry.resolve_model_path(" ".join(args))
                if p and p.exists():
                    self.s1_model_path = p
                    self.console.print(f"[green]✔ Set System 1 model to: [bold]{p.name}[/bold][/green]")
                else:
                    self.console.print(f"[red]Could not resolve System 1 model '{' '.join(args)}'[/red]")
            else:
                chosen = self._select_model_interactive(filter_format="system1")
                if chosen:
                    self.s1_model_path = chosen
                    self.console.print(f"[green]✔ Set System 1 model to: [bold]{chosen.name}[/bold][/green]")

        elif cmd in ("/s1_device", "/device"):
            if args:
                dev = args[0].lower()
                if dev in ("gpu", "cuda", "directml", "dml"):
                    self.s1_device = "gpu"
                    self.console.print("[green]✔ System 1 device set to: [bold]GPU[/bold] (DirectML / CUDA)[/green]")
                    self.orchestrator.s1_engine._agents.clear()
                elif dev in ("cpu", "ram"):
                    self.s1_device = "cpu"
                    self.console.print("[green]✔ System 1 device set to: [bold]CPU[/bold] (Heterogeneous parallel mode)[/green]")
                    self.orchestrator.s1_engine._agents.clear()
                elif dev in ("auto", "default"):
                    self.s1_device = "auto"
                    self.console.print("[green]✔ System 1 device set to: [bold]AUTO[/bold] (Adaptive hardware topology)[/green]")
                    self.orchestrator.s1_engine._agents.clear()
                else:
                    self.console.print("[yellow]Invalid device. Choose: gpu, cpu, or auto.[/yellow]")
            else:
                if sys.stdin.isatty():
                    chosen_dev = pick_device(current_device=self.s1_device, console=self.console)
                    if chosen_dev:
                        self.s1_device = chosen_dev
                        self.console.print(f"[green]✔ System 1 device set to: [bold]{chosen_dev.upper()}[/bold][/green]")
                        self.orchestrator.s1_engine._agents.clear()
                else:
                    self.console.print(f"[cyan]Current System 1 Device: [bold]{self.s1_device.upper()}[/bold][/cyan]")

        elif cmd == "/s2":
            if args:
                p = self.model_registry.resolve_model_path(" ".join(args))
                if p and p.exists():
                    self.s2_model_path = p
                    self.console.print(f"[green]✔ Set System 2 teacher model to: [bold]{p.name}[/bold][/green]")
                else:
                    self.console.print(f"[red]Could not resolve System 2 model '{' '.join(args)}'[/red]")
            else:
                chosen = self._select_model_interactive(filter_format="system2")
                if chosen:
                    self.s2_model_path = chosen
                    self.console.print(f"[green]✔ Set System 2 teacher model to: [bold]{chosen.name}[/bold][/green]")

        elif cmd == "/model":
            if args:
                p = self.model_registry.resolve_model_path(" ".join(args))
                if p and p.exists():
                    self.model_path = p
                    fmt = detect_model_format(p)
                    if fmt == ModelFormat.SYSTEM1:
                        self.s1_model_path = p
                        self.mode = "system1"
                    else:
                        self.s2_model_path = p
                        self.mode = "standard"
                    self._initialize_engine()
                else:
                    self.console.print(f"[red]Could not resolve model '{' '.join(args)}'[/red]")
            else:
                chosen = self._select_model_interactive()
                if chosen:
                    self.model_path = chosen
                    fmt = detect_model_format(chosen)
                    if fmt == ModelFormat.SYSTEM1:
                        self.s1_model_path = chosen
                        self.mode = "system1"
                    else:
                        self.s2_model_path = chosen
                        self.mode = "standard"
                    self._initialize_engine()

        elif cmd == "/models":
            models = self.model_registry.list_models()
            table = Table(title="InferenceOS Registered Models", box=None)
            table.add_column("Type", style="bold magenta")
            table.add_column("Nickname", style="bold cyan")
            table.add_column("Format", style="yellow")
            table.add_column("Size (GB)", justify="right")
            table.add_column("Path", style="dim")

            for m in models:
                sz_gb = m.get("size_bytes", 0) / (1024**3)
                fmt_str = str(m.get("format", "")).upper()
                type_badge = "⚡ System 1" if fmt_str == "SYSTEM1" else "🧠 System 2"
                table.add_row(type_badge, m["nickname"], fmt_str, f"{sz_gb:.2f}", m.get("location", "")[:45])
            self.console.print(table)

        elif cmd == "/escalate":
            if args:
                val = args[0].lower()
                self.escalate = val in ("on", "true", "1", "yes")
                self.console.print(f"[green]✔ System 2 Escalation set to: {'ON' if self.escalate else 'OFF'}[/green]")
            else:
                self.escalate = not self.escalate
                self.console.print(f"[green]✔ Toggled System 2 Escalation to: {'ON' if self.escalate else 'OFF'}[/green]")

        elif cmd == "/tau":
            if args:
                try:
                    self.escalate_tau = max(0.0, min(1.0, float(args[0])))
                    self.console.print(f"[green]✔ Escalation threshold tau set to: [bold]{self.escalate_tau:.2f}[/bold][/green]")
                except ValueError:
                    self.console.print("[red]Invalid tau value. Provide a float between 0.0 and 1.0.[/red]")
            else:
                self.console.print(f"[cyan]Current escalation threshold tau: [bold]{self.escalate_tau:.2f}[/bold][/cyan]")

        elif cmd == "/teacher":
            if args:
                val = " ".join(args)
                if val.startswith("http://") or val.startswith("https://"):
                    self.s2_base_url = val
                    self.console.print(f"[green]✔ Set remote System 2 teacher endpoint to: {val}[/green]")
                else:
                    p = self.model_registry.resolve_model_path(val)
                    if p and p.exists():
                        self.s2_model_path = p
                        self.console.print(f"[green]✔ Set local System 2 teacher model to: {p.name}[/green]")
                    else:
                        self.console.print(f"[red]Could not resolve teacher model: {val}[/red]")
            else:
                t_str = self.s2_base_url or (self.s2_model_path.name if self.s2_model_path else "Rule-based fallback")
                self.console.print(f"[cyan]Current System 2 Teacher: [bold]{t_str}[/bold][/cyan]")

        elif cmd == "/dagger":
            if args:
                sub = args[0].lower()
                if sub in ("off", "disable", "stop"):
                    self.dagger_log_path = None
                    self.console.print("[yellow]✔ DAgger distillation logging disabled.[/yellow]")
                elif sub in ("on", "enable", "start"):
                    p = Path(args[1]) if len(args) > 1 else Path("logs/dagger_distill.jsonl")
                    self.dagger_log_path = p
                    p.parent.mkdir(parents=True, exist_ok=True)
                    self.console.print(f"[green]✔ DAgger distillation logging active: [bold]{p}[/bold][/green]")
                elif sub == "status":
                    if self.dagger_log_path:
                        cnt = 0
                        if self.dagger_log_path.exists():
                            with open(self.dagger_log_path, "r", encoding="utf-8") as f:
                                cnt = sum(1 for _ in f)
                        self.console.print(f"[cyan]DAgger Log: [bold]{self.dagger_log_path}[/bold] ({cnt} escalation records captured)[/cyan]")
                    else:
                        self.console.print("[dim]DAgger distillation logging is currently OFF.[/dim]")
                else:
                    self.dagger_log_path = Path(args[0])
                    self.dagger_log_path.parent.mkdir(parents=True, exist_ok=True)
                    self.console.print(f"[green]✔ DAgger log path set to: [bold]{self.dagger_log_path}[/bold][/green]")
            else:
                if self.dagger_log_path:
                    self.console.print(f"[cyan]DAgger Log: [bold]{self.dagger_log_path}[/bold][/cyan]")
                else:
                    self.console.print("[cyan]DAgger distillation logging is currently OFF. Use '/dagger on [path]' to enable.[/cyan]")

        elif cmd == "/guardrail":
            if not args:
                self.console.print("[yellow]Usage: /guardrail <command or action to verify>[/yellow]")
            else:
                if not self.s1_model_path or not self.s1_model_path.exists():
                    self.console.print("[red]No System 1 model selected. Use /s1 to choose one first.[/red]")
                else:
                    action = " ".join(args)
                    self.console.print(f"[cyan]Running System 1 safety guardrail check on '{action}'...[/cyan]")
                    try:
                        res = self.orchestrator.system1_guardrail(
                            model_path=self.s1_model_path,
                            proposed_action=action,
                        )
                        status_style = "bold green" if res["allowed"] else "bold red"
                        verdict = "✔ ALLOWED (Safe)" if res["allowed"] else "✖ BLOCKED (Destructive / Risky)"
                        self.console.print(f"[{status_style}]Verdict: {verdict}[/{status_style}] (Safety Score: [bold]{res['safety_score']:.3f}[/bold] | Latency: {res['latency_ms']:.1f}ms)")
                    except Exception as e:
                        self.console.print(f"[red]Guardrail evaluation error: {e}[/red]")

        elif cmd == "/tool":
            if not args:
                self.console.print("[yellow]Usage: /tool option1, option2, option3...[/yellow]")
            else:
                if not self.s1_model_path or not self.s1_model_path.exists():
                    self.console.print("[red]No System 1 model selected. Use /s1 to choose one first.[/red]")
                else:
                    raw_opts = " ".join(args).split(",")
                    opts = [o.strip() for o in raw_opts if o.strip()]
                    self.console.print(f"[cyan]Evaluating options across System 1: {opts}...[/cyan]")
                    try:
                        res = self.orchestrator.system1_tool_decide(
                            model_path=self.s1_model_path,
                            state="User tool evaluation request",
                            options=opts,
                        )
                        self.console.print(f"[bold green]• Selected Choice:[/bold green] [bold yellow]{res['chosen']}[/bold yellow] (Confidence: [bold]{res['confidence']:.2f}[/bold] | Latency: {res['latency_ms']:.1f}ms)")
                    except Exception as e:
                        self.console.print(f"[red]Tool evaluation error: {e}[/red]")

        elif cmd == "/decide":
            goal = " ".join(args) if args else "Evaluate current context"
            options_input = self.console.input("[bold cyan]Enter comma-separated options (or Enter for YES/NO/UNCERTAIN): [/bold cyan]").strip()
            if options_input:
                opts = [o.strip() for o in options_input.split(",") if o.strip()]
                crit = {opt: f"Select {opt}" for opt in opts}
            else:
                crit = {"YES": "Affirmative", "NO": "Negative", "UNCERTAIN": "Need more info"}

            questions = {
                "decision": {
                    "type": "choice",
                    "instructions": {"goal": goal},
                    "criteria": crit,
                }
            }

            if not self.s1_model_path or not self.s1_model_path.exists():
                self.console.print("[red]No System 1 model selected. Use /s1 to select one first.[/red]")
                return True

            self.console.print(f"[cyan]Evaluating decision for goal: '{goal}'...[/cyan]")
            try:
                res = self.orchestrator.decide_with_escalation(
                    model_path=self.s1_model_path,
                    state=goal,
                    questions=questions,
                    escalate=self.escalate,
                    escalate_tau=self.escalate_tau,
                    system2_model=self.s2_model_path,
                )
                ans = res["answers"].get("decision", {})
                self.console.print(f"[bold green]Result:[/bold green] [bold yellow]{ans.get('choice')}[/bold yellow] (Confidence: {ans.get('confidence', 0):.2f})")
            except Exception as e:
                self.console.print(f"[red]Decision evaluation error: {e}[/red]")

        elif cmd == "/settings":
            try:
                from cli.tui.settings_menu import InteractiveSettingsMenu
                menu = InteractiveSettingsMenu(theme_name=self.theme_mgr.theme_name)
                menu.run_interactive()
                self.console.clear()
            except Exception as e:
                self.console.print(f"[red]Settings menu error: {e}[/red]")

        elif cmd == "/doctor":
            from cli.commands.doctor import handle_doctor_command
            self.console.print("\n[bold cyan]Running InferenceOS Diagnostic Health Checks...[/bold cyan]\n")
            try:
                handle_doctor_command()
            except Exception as e:
                self.console.print(f"[red]Doctor diagnostic error: {e}[/red]")

        elif cmd == "/benchmark":
            from cli.commands.benchmark import handle_benchmark_command
            m = args[0] if args else (self.model_path.name if self.model_path else None)
            if not m:
                self.console.print("[red]No model selected for benchmark. Usage: /benchmark [model][/red]")
            else:
                self.console.print(f"\n[bold cyan]Running benchmark for '{m}'...[/bold cyan]\n")
                try:
                    handle_benchmark_command(model_query=m)
                except Exception as e:
                    self.console.print(f"[red]Benchmark error: {e}[/red]")

        elif cmd == "/inspect":
            from cli.commands.inspect_cmd import handle_inspect_command
            m = args[0] if args else (self.model_path.name if self.model_path else None)
            if not m:
                self.console.print("[red]No model selected to inspect. Usage: /inspect [model][/red]")
            else:
                try:
                    handle_inspect_command(model_query=m)
                except Exception as e:
                    self.console.print(f"[red]Inspect error: {e}[/red]")

        elif cmd == "/health":
            from cli.health_cli import handle_health_cli
            try:
                handle_health_cli(args)
            except Exception as e:
                self.console.print(f"[red]Health monitor error: {e}[/red]")

        elif cmd == "/kv":
            from cli.kv_cli import handle_kv_cli
            try:
                handle_kv_cli(args)
            except Exception as e:
                self.console.print(f"[red]KV manager error: {e}[/red]")

        elif cmd == "/budget":
            from cli.budget_cli import handle_budget_cli
            try:
                handle_budget_cli(args)
            except Exception as e:
                self.console.print(f"[red]Budget manager error: {e}[/red]")

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
                if sys.stdin.isatty():
                    active_prof = self.config_mgr.get("profiles.active_profile", "balanced")
                    chosen_prof = pick_profile(current_profile=active_prof, console=self.console)
                    if chosen_prof:
                        if self.profile_mgr.activate_profile(chosen_prof):
                            self.console.print(f"[green]✔ Activated profile: [bold]{chosen_prof}[/bold][/green]")
                            self._initialize_engine()
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
            if not self.history:
                self.console.print("[dim]Session history is currently empty.[/dim]")
            else:
                self.console.print(f"[bold yellow]Session Message History ({len(self.history)} entries):[/bold yellow]")
                for idx, msg in enumerate(self.history, 1):
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    snippet = str(msg.get("content", ""))[:120].replace("\n", " ")
                    self.console.print(f"  [dim]#{idx:02d}[/dim] [bold cyan]{role}:[/bold cyan] {snippet}...")

        elif cmd == "/reset":
            self.history = []
            self.last_decision = None
            self.last_confidence = 0.0
            self.last_latency_ms = 0.0
            self.last_escalated = False
            self.console.print("[green]Chat context and reflex session statistics reset.[/green]")

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
                self.console.print("[yellow]No active placement plan (Stateless or System 1 mode).[/yellow]")

        elif cmd == "/reload":
            self.console.print("[cyan]Reloading engine and placement plan...[/cyan]")
            if self._initialize_engine():
                self.console.print("[green]✔ Engine and placement plan reloaded successfully.[/green]")
            else:
                self.console.print("[red]Failed to reload engine.[/red]")

        elif cmd in ("/stats", "/status"):
            table = Table(title="InferenceOS Active Session Statistics", box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold green")
            table.add_row("Active Mode", self.mode.upper())
            table.add_row("System 1 Model", self.s1_model_path.name if self.s1_model_path else "None")
            table.add_row("System 2 Model", self.s2_model_path.name if self.s2_model_path else "None")
            table.add_row("Escalation", "Enabled" if self.escalate else "Disabled")
            table.add_row("Escalation Tau", f"{self.escalate_tau:.2f}")
            table.add_row("Last Decision", str(self.last_decision))
            table.add_row("Last Confidence", f"{self.last_confidence:.2f}")
            table.add_row("Last Latency", f"{self.last_latency_ms:.1f} ms")
            self.console.print(table)

        elif cmd == "/context":
            ctx_max = self.plan.context_length if self.plan else 4096
            ctx_used = int(sum(len(m.get("content", "").split()) * 1.3 for m in self.history))
            pct = min(100.0, (ctx_used / max(1, ctx_max)) * 100)
            headroom = max(0, ctx_max - ctx_used)
            bar_len = int(pct / 5.0)
            bar_str = "█" * bar_len + "░" * (20 - bar_len)

            table = Table(title="Context Window Utilization", box=None)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="bold green")
            table.add_row("Tokens Estimated", str(ctx_used))
            table.add_row("Maximum Capacity", f"{ctx_max} tokens")
            table.add_row("Headroom Remaining", f"{headroom} tokens")
            table.add_row("Utilization", f"[{bar_str}] {pct:.1f}%")
            self.console.print(table)

        elif cmd == "/memory":
            table = Table(title="InferenceOS Memory Footprint", box=None)
            table.add_column("Component", style="cyan")
            table.add_column("Allocation", style="bold green")
            if self.plan:
                table.add_row("Estimated Model VRAM", f"{self.plan.estimated_vram_bytes / (1024**2):.1f} MB")
                table.add_row("Estimated System RAM", f"{self.plan.estimated_ram_bytes / (1024**2):.1f} MB")
            mem = self.hw_profile.get("memory", {})
            if mem:
                table.add_row("System Total RAM", f"{mem.get('total_gb', 0):.1f} GB")
                table.add_row("System Available RAM", f"{mem.get('available_gb', 0):.1f} GB")
            gpus = self.hw_profile.get("gpus", [])
            if gpus:
                table.add_row("GPU VRAM Total", f"{gpus[0].get('vram_total_mb', 0)} MB")
            self.console.print(table)

        elif cmd == "/config":
            table = Table(title="InferenceOS Active Configuration", box=None)
            table.add_column("Parameter", style="cyan")
            table.add_column("Current Setting", style="bold green")
            table.add_row("Active Mode", self.mode.upper())
            table.add_row("Backend", str(self.config_mgr.get("runtime.backend", "auto")))
            table.add_row("Threads", str(self.config_mgr.get("runtime.threads", 6)))
            table.add_row("Context Length", str(self.config_mgr.get("memory.context_length", 4096)))
            table.add_row("Batch Size", str(self.config_mgr.get("memory.batch_size", 512)))
            table.add_row("Sampling Temp", str(self.config_mgr.get("sampling.temp", 0.7)))
            table.add_row("Escalation Tau", str(self.escalate_tau))
            table.add_row("Active Profile", str(self.config_mgr.get("profiles.active_profile", "balanced")))
            self.console.print(table)

        elif cmd == "/telemetry":
            from cli.core.telemetry_store import get_telemetry_store
            store = get_telemetry_store()
            summary = store.get_summary_stats()
            table = Table(title="InferenceOS Telemetry Summary", box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold green")
            table.add_row("Total Recorded Runs", str(summary.get("total_runs", 0)))
            table.add_row("Avg Generation TPS", f"{summary.get('avg_generation_tps', 0.0):.2f} tok/s")
            table.add_row("Avg Prompt TPS", f"{summary.get('avg_prompt_tps', 0.0):.1f} tok/s")
            table.add_row("Avg TTFT", f"{summary.get('avg_ttft_ms', 0.0):.1f} ms")
            self.console.print(table)

        elif cmd == "/save":
            if not self.history:
                self.console.print("[yellow]Notice: Conversation history is empty, nothing to save.[/yellow]")
            else:
                out_path = Path(args[0]) if args else (self.config_mgr.sessions_dir / f"session_{int(time.time())}.json")
                try:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    data = {
                        "model": str(self.model_path) if self.model_path else "none",
                        "mode": self.mode,
                        "s1_model": str(self.s1_model_path) if self.s1_model_path else None,
                        "s2_model": str(self.s2_model_path) if self.s2_model_path else None,
                        "escalate": self.escalate,
                        "escalate_tau": self.escalate_tau,
                        "last_decision": self.last_decision,
                        "timestamp": time.time(),
                        "messages": self.history,
                    }
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    self.console.print(f"[green]✔ Conversation saved to: {out_path}[/green]")
                except Exception as e:
                    self.console.print(f"[red]Failed to save session: {e}[/red]")

        elif cmd == "/load":
            if not args:
                self.console.print("[yellow]Usage: /load <path_to_session.json>[/yellow]")
            else:
                load_path = Path(" ".join(args))
                if not load_path.exists():
                    self.console.print(f"[red]Error: File not found: {load_path}[/red]")
                else:
                    try:
                        with open(load_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        self.history = data.get("messages", [])
                        self.mode = data.get("mode", self.mode)
                        if "escalate" in data:
                            self.escalate = data["escalate"]
                        if "escalate_tau" in data:
                            self.escalate_tau = data["escalate_tau"]
                        self.console.print(f"[green]✔ Loaded {len(self.history)} messages from {load_path}[/green]")
                    except Exception as e:
                        self.console.print(f"[red]Failed to load session: {e}[/red]")

        elif cmd == "/export":
            if not self.history:
                self.console.print("[yellow]Notice: Conversation history is empty, nothing to export.[/yellow]")
            else:
                arg_val = args[0] if args else "md"
                is_json = arg_val.lower() == "json" or arg_val.lower().endswith(".json")
                if is_json:
                    exp_path = Path(arg_val) if arg_val.lower().endswith(".json") else Path(f"chat_export_{int(time.time())}.json")
                    try:
                        exp_path.parent.mkdir(parents=True, exist_ok=True)
                        data = {
                            "export_time": time.time(),
                            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "mode": self.mode,
                            "s1_model": str(self.s1_model_path) if self.s1_model_path else None,
                            "s2_model": str(self.s2_model_path) if self.s2_model_path else None,
                            "history": self.history,
                        }
                        with open(exp_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        self.console.print(f"[green]✔ Conversation exported to JSON: [bold]{exp_path.resolve()}[/bold][/green]")
                    except Exception as e:
                        self.console.print(f"[red]Failed to export chat to JSON: {e}[/red]")
                else:
                    exp_path = Path(arg_val) if arg_val.lower().endswith((".md", ".txt")) else Path(f"chat_export_{int(time.time())}.md")
                    try:
                        exp_path.parent.mkdir(parents=True, exist_ok=True)
                        lines = [
                            "# InferenceOS Chat Session Export",
                            f"- Mode: `{self.mode.upper()}`",
                            f"- Model: `{self.model_path.name if self.model_path else 'none'}`",
                            f"- Date: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
                            f"- Total Turns: `{len(self.history)}`\n",
                            "---",
                        ]
                        for msg in self.history:
                            role_hdr = "### User" if msg.get("role") == "user" else "### Assistant"
                            lines.append(f"\n{role_hdr}\n\n{msg.get('content', '')}\n")
                        exp_path.write_text("\n".join(lines), encoding="utf-8")
                        self.console.print(f"[green]✔ Conversation exported to Markdown: [bold]{exp_path.resolve()}[/bold][/green]")
                    except Exception as e:
                        self.console.print(f"[red]Failed to export chat to Markdown: {e}[/red]")

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

        s1_name = self.s1_model_path.name if self.s1_model_path else "None"
        s2_name = self.s2_model_path.name if self.s2_model_path else "None"
        banner.append("          Active Mode: ", style=tm.color("muted"))
        banner.append(self.mode.upper(), style=f"bold {tm.color('primary')}")
        banner.append(" │ S1: ", style=tm.color("muted"))
        banner.append(s1_name, style=f"bold {tm.color('warning')}")
        banner.append(" │ S2: ", style=tm.color("muted"))
        banner.append(s2_name, style=f"bold {tm.color('accent')}")
        banner.append("\n")
        banner.append("          Type /mode to change mode ── /models to list models ── /help for all commands\n", style=tm.color("muted"))

        return Panel(banner, border_style=tm.color("primary"))
