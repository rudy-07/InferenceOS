"""
inspect_cmd.py
--------------
Universal model inspector for 'inferenceos inspect <model>'.
Supports GGUF, SafeTensors, ONNX, PyTorch (.pt/.pth), Pickle (.pkl), and OBX.
"""
from __future__ import annotations

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.core.model_registry import get_model_registry
from orchestrator.model_parser import read_model_metadata, detect_model_format, ModelFormat


def handle_inspect_command(model_query: str) -> None:
    """Inspect model architecture, parameters, and metadata across any format."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    console = Console()
    registry = get_model_registry()

    model_path = registry.resolve_model_path(model_query)
    if not model_path or not model_path.exists():
        console.print(f"[bold red]Error: Could not resolve model '{model_query}'[/bold red]")
        sys.exit(1)

    fmt = detect_model_format(model_path)
    if fmt == ModelFormat.UNKNOWN or fmt.value == "unknown":
        console.print(f"[yellow]Notice: '{model_path.name}' does not match standard model signatures (Format: UNKNOWN).[/yellow]")
    console.print(f"[cyan]Inspecting [bold yellow]{fmt.value.upper()}[/bold yellow] file: [bold]{model_path}[/bold]...[/cyan]")

    try:
        meta = read_model_metadata(model_path)
    except Exception as e:
        console.print(f"[yellow]Warning: Error parsing model metadata ({e}). Showing basic stats.[/yellow]")
        meta = {
            "format": fmt.value,
            "arch": "unknown",
            "num_layers": 1,
            "hidden_size": 512,
            "num_heads": 8,
            "num_kv_heads": 8,
            "max_context_length": 2048,
            "total_params": 0,
            "tensor_count": 0,
            "tensors": [],
            "format_details": {},
        }

    sz_mb = model_path.stat().st_size / (1024 * 1024)
    sz_str = f"{sz_mb:.2f} MB" if sz_mb < 1024 else f"{sz_mb / 1024:.2f} GB"
    total_params = meta.get("total_params", 0)
    params_str = f"{total_params:,}" if total_params > 0 else "N/A"

    table = Table(title=f"Model Architecture Overview -- {model_path.name}", box=None, expand=True)
    table.add_column("Property", style="bold cyan", width=28, no_wrap=True)
    table.add_column("Value", style="bold green")

    table.add_row("Format", fmt.value.upper())
    table.add_row("File Location", str(model_path))
    table.add_row("File Size", sz_str)
    table.add_row("Architecture", str(meta.get("arch", "unknown")))
    table.add_row("Total Parameters", params_str)
    table.add_row("Transformer / Graph Layers", str(meta.get("num_layers", "1")))
    table.add_row("Hidden Dimension", str(meta.get("hidden_size", "512")))
    table.add_row("Attention Heads", str(meta.get("num_heads", "8")))
    table.add_row("KV Heads", str(meta.get("num_kv_heads", "8")))
    table.add_row("Max Context Length", str(meta.get("max_context_length", "2048")))

    details = meta.get("format_details", {})

    # Format-specific attributes
    if fmt.value == "onnx" or (fmt.value == "obx" and "opset_version" in details):
        if details.get("opset_version"):
            table.add_row("ONNX Opset Version", str(details["opset_version"]))
        if details.get("node_count"):
            table.add_row("Graph Node Count", str(details["node_count"]))
        if details.get("inputs"):
            inps = ", ".join([f"{i['name']} {i['shape']}" for i in details["inputs"]])
            table.add_row("Graph Inputs", inps)
        if details.get("outputs"):
            outs = ", ".join([f"{o['name']} {o['shape']}" for o in details["outputs"]])
            table.add_row("Graph Outputs", outs)
        if details.get("top_operators"):
            top_ops = ", ".join([f"{op}:{cnt}" for op, cnt in details["top_operators"][:5]])
            table.add_row("Top Operators", top_ops)

    elif fmt.value == "safetensors":
        table.add_row("Total Tensors", str(meta.get("tensor_count", 0)))
        if details.get("has_external_config"):
            table.add_row("HF Config", "Found adjacent config.json")

    elif fmt.value == "obx":
        table.add_row("Underlying Type", str(details.get("underlying_type", "unknown")))

    console.print(Panel(table, border_style="cyan"))

    # Tensors preview table if available
    tensors = meta.get("tensors", [])
    if tensors and isinstance(tensors, list) and isinstance(tensors[0], dict) and "name" in tensors[0]:
        t_table = Table(title="Tensor Preview (Sample)", box=None, expand=True)
        t_table.add_column("Tensor Name", style="white")
        t_table.add_column("Shape", style="bold yellow")
        t_table.add_column("Dtype", style="magenta")
        t_table.add_column("Params", justify="right", style="green")

        for t in tensors[:12]:
            t_table.add_row(
                str(t.get("name", "")),
                str(t.get("shape", "-")),
                str(t.get("dtype", "-")),
                f"{t.get('params', 0):,}" if "params" in t else "-",
            )
        console.print(Panel(t_table, border_style="dim"))
