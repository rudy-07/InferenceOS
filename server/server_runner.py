"""
server_runner.py
----------------
Server runner that renders the InferenceOS Server banner and launches uvicorn.
"""
from __future__ import annotations

import sys
import uvicorn
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from server.config import ServerConfig
from server.api.app import create_app
from server.runtime_adapter.adapter import get_runtime_adapter


def print_server_banner(config: ServerConfig) -> None:
    """Render official InferenceOS Server startup banner."""
    console = Console()
    adapter = get_runtime_adapter(config)
    hw = adapter.hw_profile
    
    gpus = hw.get("gpus", [])
    igpus = hw.get("igpus", [])
    cpu_info = hw.get("cpu", {})

    hw_lines = []
    for g in gpus:
        name = g.get("model") or g.get("name") or "Discrete GPU"
        hw_lines.append(name)
    for ig in igpus:
        name = ig.get("model") or ig.get("name") or "Integrated GPU"
        hw_lines.append(name)
    
    cpu_name = cpu_info.get("brand") or cpu_info.get("brand_raw") or cpu_info.get("name") or "CPU Processor"
    hw_lines.append(cpu_name)
    
    backend_name = config.backend or "AUTO (Vulkan / CUDA)"

    banner_text = (
        f"[bold cyan]InferenceOS Server v1.0[/bold cyan]\n\n"
        f"[bold white]Address[/bold white]\n"
        f"http://{config.host}:{config.port}\n\n"
        f"[bold white]Backend[/bold white]\n"
        f"[bold magenta]{backend_name.upper()}[/bold magenta]\n\n"
        f"[bold white]Detected Hardware[/bold white]\n"
    )
    for hw_item in hw_lines:
        banner_text += f"[green]{hw_item}[/green]\n"

    banner_text += (
        f"\n[bold green]Ready[/bold green]\n\n"
        f"[dim]{adapter.get_loaded_models_count()} models loaded[/dim]"
    )

    console.print(Panel(banner_text, border_style="cyan", title="⚡ InferenceOS Core Serve"))


def run_server(
    host: str = "0.0.0.0",
    port: int = 11434,
    backend: Optional[str] = None,
    tls_cert: Optional[str] = None,
    tls_key: Optional[str] = None,
    auth_keys: Optional[list] = None,
    config: Optional[ServerConfig] = None,
) -> None:
    """Launch InferenceOS HTTP server via Uvicorn."""
    srv_config = config or ServerConfig(
        host=host,
        port=port,
        backend=backend,
        tls_cert=tls_cert,
        tls_key=tls_key,
        auth_api_keys=auth_keys or [],
    )

    app = create_app(srv_config)
    print_server_banner(srv_config)

    uvicorn_kwargs = {
        "app": app,
        "host": srv_config.host,
        "port": srv_config.port,
        "log_level": "error" if srv_config.log_level == "ERROR" else "info",
    }
    if srv_config.tls_cert and srv_config.tls_key:
        uvicorn_kwargs["ssl_certfile"] = srv_config.tls_cert
        uvicorn_kwargs["ssl_keyfile"] = srv_config.tls_key

    uvicorn.run(**uvicorn_kwargs)
