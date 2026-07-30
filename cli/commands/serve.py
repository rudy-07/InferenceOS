"""
serve.py
--------
Command handler for 'inferenceos serve [model] [options]'.
Launches the production-grade OpenAI-compatible InferenceOS server.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from rich.console import Console

from server.server_runner import run_server
from server.runtime_adapter import get_runtime_adapter


def handle_serve_command(
    model_query: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 11434,
    backend: Optional[str] = None,
    tls_cert: Optional[str] = None,
    tls_key: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Launch InferenceOS HTTP OpenAI API inference server."""
    console = Console()

    # Pre-load model if specified
    if model_query:
        try:
            adapter = get_runtime_adapter()
            adapter.load_model_if_needed(model_query)
        except Exception as e:
            console.print(f"[bold yellow]Warning: Could not pre-load model '{model_query}': {e}[/bold yellow]")

    auth_keys = [api_key] if api_key else []

    try:
        run_server(
            host=host,
            port=port,
            backend=backend,
            tls_cert=tls_cert,
            tls_key=tls_key,
            auth_keys=auth_keys,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]InferenceOS server stopped cleanly.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Server error: {e}[/bold red]")
        sys.exit(1)
