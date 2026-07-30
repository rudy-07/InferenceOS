"""
health_cli.py
------------
CLI commands interface for Runtime Health Monitor in InferenceOS.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from runtime_health import RuntimeHealthMonitor


def handle_health_cli(args_list: Optional[List[str]] = None) -> int:
    """
    Handle CLI commands for Runtime Health Monitor.
    """
    parser = argparse.ArgumentParser(prog="inferenceos health", description="InferenceOS Runtime Health Monitor CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    subparsers.add_parser("show", help="Display current runtime health")
    subparsers.add_parser("monitor", help="Live runtime metrics summary")

    parsed = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    monitor = RuntimeHealthMonitor()

    if parsed.subcommand in (None, "show"):
        health = monitor.observe_health()
        print(health.format_cli_output())
        return 0

    elif parsed.subcommand == "monitor":
        health = monitor.observe_health()
        print("=== InferenceOS Live Health Monitor ===")
        print(f"Status           : {health.status}")
        print(f"GPU Utilization  : {health.gpu_utilization_pct:.1f}%")
        print(f"GPU VRAM Used    : {health.gpu_memory_gb:.2f} GB")
        print(f"Temperature      : {health.temperature_c:.1f}°C")
        print(f"Memory Pressure  : {health.memory_pressure}")
        print(f"Stability        : {health.performance_stability}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(handle_health_cli())
