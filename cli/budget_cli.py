"""
budget_cli.py
------------
CLI commands interface for Adaptive Memory Budget Manager in InferenceOS.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from memory_budget import MemoryBudgetManager


def handle_budget_cli(args_list: Optional[List[str]] = None) -> int:
    """
    Handle CLI commands for Adaptive Memory Budget Manager.
    """
    parser = argparse.ArgumentParser(prog="inferenceos budget", description="InferenceOS Adaptive Memory Budget Manager CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    subparsers.add_parser("show", help="Display current component memory budgets")
    subparsers.add_parser("history", help="Display memory budget evolution history")

    parsed = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    manager = MemoryBudgetManager()

    if parsed.subcommand in (None, "show"):
        decision = manager.compute_budgets()
        print(decision.format_cli_output())
        return 0

    elif parsed.subcommand == "history":
        history = manager.get_history()
        print("=== Memory Budget Evolution History ===")
        if not history:
            print("No budget decisions recorded yet.")
        else:
            for idx, d in enumerate(history, 1):
                print(f"{idx}. Policy: {d.active_policy}, Total Budget: {d.total_vram_budget_gb:.2f} GB, Reserve: {d.budgets.safety_reserve_mb:.0f} MB")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(handle_budget_cli())
