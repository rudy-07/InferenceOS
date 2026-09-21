"""
models_cmd.py
-------------
Command handler for 'inferenceos models [action] [options]'.
"""
from __future__ import annotations

from typing import List, Optional
from rich.console import Console
from rich.table import Table
from cli.core.model_registry import get_model_registry


def handle_models_command(
    action: str = "list",
    target: Optional[str] = None,
    nickname: Optional[str] = None,
    path: Optional[str] = None,
    backend: str = "auto",
    context: int = 4096,
    tags: Optional[List[str]] = None,
    query: Optional[str] = None,
) -> None:
    """Manage multi-format model library (GGUF, ONNX, SafeTensors, PyTorch, Pickle, OBX)."""
    console = Console()
    registry = get_model_registry()

    act = action.lower()

    if act == "list":
        models = registry.list_models()
        table = Table(title="InferenceOS Multi-Format Model Library", box=None, expand=True)
        table.add_column("Nickname", style="bold cyan")
        table.add_column("Format", style="bold yellow")
        table.add_column("Location", style="dim")
        table.add_column("Backend", style="magenta")
        table.add_column("Context", justify="right", style="yellow")
        table.add_column("Size (GB)", justify="right", style="green")

        for m in models:
            sz_gb = m.get("size_bytes", 0) / (1024**3)
            fmt = str(m.get("format", "unknown")).upper()
            table.add_row(
                m["nickname"],
                fmt,
                m["location"],
                m.get("backend", "auto").upper(),
                str(m.get("context", 4096)),
                f"{sz_gb:.2f}",
            )
        console.print(table)

    elif act == "discover":
        console.print("[cyan]Scanning workspace, sibling repos (KaptaanLM, Qwen-Coder), and caches for models...[/cyan]")
        discovered = registry.auto_discover()
        registry.save()
        console.print(f"[green]✔ Discovered {len(discovered)} model file(s).[/green]")

        # Show breakdown by format
        models = registry.list_models()
        fmt_counts = {}
        for m in models:
            f = str(m.get("format", "unknown")).upper()
            fmt_counts[f] = fmt_counts.get(f, 0) + 1

        summary_table = Table(title="Model Library by Format", box=None)
        summary_table.add_column("Format", style="bold yellow")
        summary_table.add_column("Count", justify="right", style="bold green")
        for f, count in sorted(fmt_counts.items()):
            summary_table.add_row(f, str(count))
        console.print(summary_table)

    elif act == "add":
        nick = nickname or target
        if not nick or not path:
            console.print("[red]Error: 'models add' requires --nickname (or target) and --path[/red]")
            return
        if registry.add_model(nick, path, backend=backend, context=context, tags=tags):
            console.print(f"[green]✔ Model '{nick}' registered successfully![/green]")

    elif act in ("remove", "delete", "rm"):
        rem_target = target or nickname or query
        if not rem_target:
            console.print("[red]Error: Specify model nickname or path to remove (e.g. 'models rm <nickname>')[/red]")
            return
        if registry.remove_model(rem_target):
            console.print(f"[green]✔ Model '{rem_target}' removed from registry.[/green]")
        else:
            console.print(f"[red]Model '{rem_target}' not found in registry.[/red]")

    elif act in ("search", "find"):
        q = target or query or nickname or ""
        results = registry.search_models(q)
        console.print(f"[bold cyan]Found {len(results)} model(s) matching '{q}':[/bold cyan]")
        for m in results:
            fmt = str(m.get("format", "unknown")).upper()
            console.print(f"  • [[bold yellow]{fmt}[/bold yellow]] [bold green]{m['nickname']}[/bold green] ── {m['location']}")
