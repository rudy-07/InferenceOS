"""
main.py
-------
Main Command Line Interface orchestrator for InferenceOS.

Provides full discoverability, interactive help, consistent styling, fast startup,
and orchestration across all 21 CLI commands.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path for top-level module resolution
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.commands.chat import handle_chat_command
from cli.commands.run import handle_run_command
from cli.commands.serve import handle_serve_command
from cli.commands.benchmark import handle_benchmark_command
from cli.commands.optimize import handle_optimize_command
from cli.commands.profile import handle_profile_command
from cli.commands.hardware import handle_hardware_command
from cli.commands.doctor import handle_doctor_command
from cli.commands.inspect_cmd import handle_inspect_command
from cli.commands.models_cmd import handle_models_command
from cli.commands.config_cmd import handle_config_command
from cli.commands.plugins_cmd import handle_plugins_command
from cli.commands.cache_cmd import handle_cache_command
from cli.commands.logs_cmd import handle_logs_command
from cli.commands.telemetry_cmd import handle_telemetry_command
from cli.commands.monitor_cmd import handle_monitor_command
from cli.commands.stats_cmd import handle_stats_command
from cli.commands.placement_cmd import handle_placement_command
from cli.commands.update_cmd import handle_update_command
from cli.commands.version_cmd import handle_version_command
from cli.commands.reset_cmd import handle_reset_command


COMMAND_DESCRIPTIONS = {
    "chat": "Launch premium interactive terminal UI chat session for a model",
    "run": "Execute prompt against model with live token streaming output",
    "serve": "Launch local HTTP API inference server endpoint",
    "benchmark": "Run complete end-to-end performance benchmark suite",
    "optimize": "Automated 5-step hardware detection, placement generation, benchmarking, and caching",
    "profile": "Profile execution and generate interactive flamegraph & timeline HTMLs",
    "hardware": "Inspect CPU, GPU, iGPU, VRAM, RAM, and interconnect capabilities",
    "doctor": "Run diagnostic health checks across drivers, SDKs, and executables",
    "inspect": "Inspect GGUF file architecture, metadata, and tensor shapes",
    "models": "Register, list, remove, search, and tag models in library",
    "config": "View, edit, or interactively configure runtime settings",
    "plugins": "List, inspect, and manage loaded InferenceOS plugins",
    "cache": "View or clean placement and benchmark optimization caches",
    "logs": "View and tail recent runtime execution logs",
    "telemetry": "Launch historical metrics & telemetry dashboard UI",
    "monitor": "Launch HTOP-style live process & hardware monitor screen",
    "stats": "Display aggregated performance summary statistics",
    "placement": "Compute and visualize layer placement across CPU/dGPU/iGPU",
    "update": "Check system engine binaries and build updates",
    "version": "Display InferenceOS release version banner and environment info",
    "reset": "Reset configuration, profiles, and caches back to defaults",
}


def print_custom_help() -> None:
    """Render rich formatted CLI help table with examples and suggestions."""
    console = Console()
    console.print("\n[bold cyan]⚡ InferenceOS ── The Operating System for Local AI Inference[/bold cyan]\n")

    table = Table(box=None, expand=True)
    table.add_column("Command", style="bold green", width=16)
    table.add_column("Description", style="white")

    for cmd, desc in COMMAND_DESCRIPTIONS.items():
        table.add_row(f"inferenceos {cmd}", desc)

    console.print(Panel(table, title="Available Subcommands", border_style="cyan"))

    examples_table = Table(title="Usage Examples", box=None, expand=True)
    examples_table.add_column("Command Example", style="bold yellow")
    examples_table.add_column("Description", style="dim")

    examples_table.add_row("inferenceos chat model.gguf", "Launch interactive chat mode")
    examples_table.add_row("inferenceos run model.gguf --prompt 'Explain quantum mechanics'", "Single-shot streaming prompt run")
    examples_table.add_row("inferenceos optimize model.gguf", "Find and cache fastest placement configuration")
    examples_table.add_row("inferenceos doctor", "Run system diagnostic check")
    examples_table.add_row("inferenceos monitor", "Launch live htop-style process monitor")
    examples_table.add_row("inferenceos config --interactive", "Launch terminal settings menu")

    console.print(examples_table)
    console.print("\n[dim]Run 'inferenceos <command> --help' for detailed command options.[/dim]\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferenceos",
        description="InferenceOS ── The Operating System for Local AI Inference",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show CLI help summary")
    parser.add_argument("-v", "--version", action="store_true", help="Show version information")

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # chat
    p_chat = subparsers.add_parser("chat", help="Launch interactive chat session")
    p_chat.add_argument("model", nargs="?", default=None, help="Model nickname, path, or query")
    p_chat.add_argument("--theme", type=str, default="nord", help="Console theme")

    # run
    p_run = subparsers.add_parser("run", help="Run single-shot prompt")
    p_run.add_argument("model", help="Model nickname, path, or query")
    p_run.add_argument("-p", "--prompt", type=str, default=None, help="Input prompt")
    p_run.add_argument("-b", "--backend", type=str, default=None, help="Force backend (auto, vulkan, cuda, cpu)")
    p_run.add_argument("-t", "--threads", type=int, default=None, help="Thread count")
    p_run.add_argument("--temp", type=float, default=None, help="Sampling temperature")
    p_run.add_argument("-n", "--max-tokens", type=int, default=None, help="Max tokens to generate")

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch API server")
    p_serve.add_argument("model", nargs="?", default=None, help="Optional model nickname or path to pre-load")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    p_serve.add_argument("--port", type=int, default=11434, help="Port number")
    p_serve.add_argument("-b", "--backend", type=str, default=None, help="Backend choice (vulkan, cuda, cpu)")
    p_serve.add_argument("--tls-cert", type=str, default=None, help="TLS Certificate file path")
    p_serve.add_argument("--tls-key", type=str, default=None, help="TLS Private Key file path")
    p_serve.add_argument("--api-key", type=str, default=None, help="Require API Key for authentication")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark suite")
    p_bench.add_argument("model", nargs="?", default=None, help="Model name or query")
    p_bench.add_argument("-g", "--gpu-layers", type=int, default=None, help="Override GPU offload layers")
    p_bench.add_argument("-t", "--threads", type=int, default=None, help="Override thread count")

    # optimize
    p_opt = subparsers.add_parser("optimize", help="Run placement optimizer & benchmark search")
    p_opt.add_argument("model", help="Model nickname or path")

    # profile
    p_prof = subparsers.add_parser("profile", help="Run profiler & flamegraph generator")
    p_prof.add_argument("model", help="Model nickname or path")

    # hardware
    subparsers.add_parser("hardware", help="Inspect hardware inventory")

    # doctor
    subparsers.add_parser("doctor", help="Run system diagnostics")

    # inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect GGUF file architecture")
    p_insp.add_argument("model", help="Model nickname or path")

    # models
    p_models = subparsers.add_parser("models", help="Model registry manager")
    p_models.add_argument("action", nargs="?", default="list", choices=["list", "add", "remove", "rm", "search"], help="Action to perform")
    p_models.add_argument("--nickname", type=str, default=None, help="Model nickname")
    p_models.add_argument("--path", type=str, default=None, help="Path to GGUF model")
    p_models.add_argument("--backend", type=str, default="auto", help="Default backend")
    p_models.add_argument("--context", type=int, default=4096, help="Default context length")
    p_models.add_argument("--query", type=str, default=None, help="Search query")

    # config
    p_cfg = subparsers.add_parser("config", help="View or edit configuration")
    p_cfg.add_argument("key", nargs="?", default=None, help="Configuration path key (e.g. 'runtime.threads')")
    p_cfg.add_argument("value", nargs="?", default=None, help="New setting value")
    p_cfg.add_argument("-i", "--interactive", action="store_true", help="Launch interactive settings menu")

    # plugins
    subparsers.add_parser("plugins", help="List plugins")

    # cache
    p_cache = subparsers.add_parser("cache", help="Manage cache")
    p_cache.add_argument("action", nargs="?", default="view", choices=["view", "clear", "clean"], help="Cache action")

    # logs
    p_logs = subparsers.add_parser("logs", help="View logs")
    p_logs.add_argument("-n", "--lines", type=int, default=50, help="Number of log lines")

    # telemetry
    subparsers.add_parser("telemetry", help="Launch telemetry dashboard")

    # monitor
    subparsers.add_parser("monitor", help="Launch live htop monitor")

    # stats
    subparsers.add_parser("stats", help="Show performance statistics")

    # placement
    p_place = subparsers.add_parser("placement", help="Compute layer placement plan")
    p_place.add_argument("model", help="Model nickname or path")

    # update
    subparsers.add_parser("update", help="Check for engine updates")

    # version
    subparsers.add_parser("version", help="Display version info")

    # reset
    subparsers.add_parser("reset", help="Reset settings to defaults")

    return parser


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point dispatcher."""
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.version:
        handle_version_command()
        return

    if args.help or not args.command:
        print_custom_help()
        return

    cmd = args.command.lower()

    if cmd == "chat":
        handle_chat_command(model_query=args.model, theme=args.theme)
    elif cmd == "run":
        handle_run_command(
            model_query=args.model,
            prompt=args.prompt,
            backend=args.backend,
            threads=args.threads,
            temp=args.temp,
            max_tokens=args.max_tokens,
        )
    elif cmd == "serve":
        handle_serve_command(
            model_query=args.model,
            host=args.host,
            port=args.port,
            backend=args.backend,
            tls_cert=args.tls_cert,
            tls_key=args.tls_key,
            api_key=args.api_key,
        )
    elif cmd == "benchmark":
        handle_benchmark_command(model_query=args.model, gpu_layers=args.gpu_layers, threads=args.threads)
    elif cmd == "optimize":
        handle_optimize_command(model_query=args.model)
    elif cmd == "profile":
        handle_profile_command(model_query=args.model)
    elif cmd == "hardware":
        handle_hardware_command()
    elif cmd == "doctor":
        handle_doctor_command()
    elif cmd == "inspect":
        handle_inspect_command(model_query=args.model)
    elif cmd == "models":
        handle_models_command(
            action=args.action,
            nickname=args.nickname,
            path=args.path,
            backend=args.backend,
            context=args.context,
            query=args.query,
        )
    elif cmd == "config":
        handle_config_command(key=args.key, value=args.value, interactive=args.interactive)
    elif cmd == "plugins":
        handle_plugins_command()
    elif cmd == "cache":
        handle_cache_command(action=args.action)
    elif cmd == "logs":
        handle_logs_command(lines=args.lines)
    elif cmd == "telemetry":
        handle_telemetry_command()
    elif cmd == "monitor":
        handle_monitor_command()
    elif cmd == "stats":
        handle_stats_command()
    elif cmd == "placement":
        handle_placement_command(model_query=args.model)
    elif cmd == "update":
        handle_update_command()
    elif cmd == "version":
        handle_version_command()
    elif cmd == "reset":
        handle_reset_command()
    else:
        Console().print(f"[bold red]Unknown command: {cmd}[/bold red]")
        print_custom_help()


if __name__ == "__main__":
    cli_main()
