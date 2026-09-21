"""
run.py
------
Command handler for 'inferenceos run <model> [options]'.
Supports GGUF, ONNX, SafeTensors, PyTorch (.pt/.pth), Pickle (.pkl), and OBX.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from rich.console import Console

from cli.core.config_manager import get_config_manager
from cli.core.model_registry import get_model_registry
import profiler
from layer_placement import ModelDescriptor, PlacementEngine
from inference_runtime import InferenceSession, RuntimeConfig, MultiFormatRuntimeEngine
from orchestrator.model_parser import detect_model_format, ModelFormat, read_model_metadata


def handle_run_command(
    model_query: str,
    prompt: Optional[str] = None,
    backend: Optional[str] = None,
    threads: Optional[int] = None,
    temp: Optional[float] = None,
    max_tokens: Optional[int] = None,
    speculative: bool = False,
    spec_mode: Optional[str] = None,
    draft_tokens: Optional[int] = None,
    draft_model: Optional[str] = None,
) -> None:
    """Execute single prompt or interactive run across any supported model format."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console = Console()
    registry = get_model_registry()
    config_mgr = get_config_manager()

    model_path = registry.resolve_model_path(model_query)
    if not model_path or not model_path.exists():
        console.print(f"[bold red]Error: Could not resolve model '{model_query}'[/bold red]")
        sys.exit(1)

    fmt = detect_model_format(model_path)
    console.print(f"[cyan]Loading InferenceOS Runtime for [bold yellow]{fmt.value.upper()}[/bold yellow] model [bold]{model_path.name}[/bold]...[/cyan]")

    sys_res = profiler.get_system_resources()
    hw_profile = sys_res.to_dict() if hasattr(sys_res, "to_dict") else sys_res

    cfg_kwargs = config_mgr.get_runtime_config_kwargs()
    if backend:
        cfg_kwargs["force_backend"] = backend
    if threads:
        cfg_kwargs["threads"] = threads
    if temp is not None:
        cfg_kwargs["temp"] = temp
    if max_tokens:
        cfg_kwargs["n_predict"] = max_tokens
    if speculative:
        cfg_kwargs["enable_speculative_decoding"] = True
    if spec_mode:
        cfg_kwargs["spec_mode"] = spec_mode
    if draft_tokens:
        cfg_kwargs["spec_draft_tokens"] = draft_tokens
    if draft_model:
        cfg_kwargs["spec_eagle_draft_model"] = draft_model

    runtime_cfg = RuntimeConfig.from_hw_profile(hw_profile, **cfg_kwargs)

    input_prompt = prompt
    if not input_prompt:
        console.print("[yellow]No --prompt specified. Enter prompt below:[/yellow]")
        input_prompt = console.input("[bold cyan]Prompt > [/bold cyan]").strip()

    if not input_prompt:
        console.print("[red]Empty prompt. Exiting.[/red]")
        sys.exit(0)

    console.print(f"\n[bold green]Response:[/bold green]")

    def on_token(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    if fmt == ModelFormat.GGUF:
        try:
            meta = read_model_metadata(model_path)
        except Exception:
            meta = {"arch": "llama", "num_layers": 32, "max_context_length": 4096}

        model_desc = ModelDescriptor.from_gguf_metadata(
            metadata=meta,
            model_size_bytes=model_path.stat().st_size,
            model_name=model_path.stem,
        )

        placement_engine = PlacementEngine(hw_profile=hw_profile)
        plan = placement_engine.generatePlacementPlan(model=model_desc, context_length=4096)

        session = InferenceSession(
            model_path=model_path,
            plan=plan,
            config=runtime_cfg,
            hw_profile=hw_profile,
        )
        res = session.run(input_prompt, on_token=on_token)
    else:
        # Multi-format runner: ONNX, SafeTensors, PyTorch, Pickle, OBX
        engine = MultiFormatRuntimeEngine(hw_profile=hw_profile, config=runtime_cfg)
        res = engine.execute(
            model_path=model_path,
            prompt=input_prompt,
            on_token=on_token,
            max_tokens=max_tokens or runtime_cfg.n_predict,
            temp=temp or runtime_cfg.temp,
        )

    print()
    backend_str = res.backend.upper() if res.backend else "UNKNOWN"
    console.print(f"\n[dim]Stats: {res.stats.eval_tps:.2f} tok/s | TTFT: {res.stats.prompt_eval_ms:.1f}ms | Backend: {backend_str}[/dim]")
