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
    nickname: Optional[str] = None,
    path: Optional[str] = None,
    backend: str = "auto",
    context: int = 4096,
    tags: Optional[List[str]] = None,
    query: Optional[str] = None,
) -> None:
    """Manage GGUF model library."""
    console = Console()
    registry = get_model_registry()

    act = action.lower()

    if act == "list":
        models = registry.list_models()
        table = Table(title="InferenceOS Model Library", box=None, expand=True)
        table.add_column("Nickname", style="bold cyan")
        table.add_column("Location", style="dim")
        table.add_column("Backend", style="magenta")
        table.add_column("Context", justify="right", style="yellow")
        table.add_column("Size (GB)", justify="right", style="green")

        for m in models:
            sz_gb = m.get("size_bytes", 0) / (1024**3)
            table.add_row(
                m["nickname"],
                m["location"],
                m.get("backend", "auto").upper(),
                str(m.get("context", 4096)),
                f"{sz_gb:.2f}",
            )
        console.print(table)

    elif act == "add":
        if not nickname or not path:
            console.print("[red]Error: 'models add' requires --nickname and --path[/red]")
            return
        if registry.add_model(nickname, path, backend=backend, context=context, tags=tags):
            console.print(f"[green]✔ Model '{nickname}' registered successfully![/green]")

    elif act in ("remove", "delete", "rm"):
        target = nickname or query
        if not target:
            console.print("[red]Error: Specify model nickname or path to remove[/red]")
            return
        if registry.remove_model(target):
            console.print(f"[green]✔ Model '{target}' removed from registry.[/green]")
        else:
            console.print(f"[red]Model '{target}' not found in registry.[/red]")

    elif act in ("search", "find"):
        q = query or nickname or ""
        results = registry.search_models(q)
        console.print(f"[bold cyan]Found {len(results)} model(s) matching '{q}':[/bold cyan]")
        for m in results:
            console.print(f"  • [bold green]{m['nickname']}[/bold green] ── {m['location']}")
