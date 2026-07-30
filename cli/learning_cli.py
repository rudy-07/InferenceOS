"""
learning_cli.py
---------------
CLI commands interface for Runtime Learning Engine and Adaptive Runtime Intelligence in InferenceOS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from runtime_learning import RuntimeLearningEngine


def handle_learning_cli(args_list: Optional[List[str]] = None) -> int:
    """
    Handle CLI commands for runtime learning.

    Subcommands:
      show    - Display learned knowledge summary
      stats   - Display learning statistics
      reset   - Reset knowledge database
      export  - Export knowledge database to JSON
      import  - Import knowledge database from JSON
    """
    parser = argparse.ArgumentParser(prog="inferenceos learning", description="InferenceOS Runtime Learning Engine CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Subcommand: show
    subparsers.add_parser("show", help="Show learned runtime knowledge")

    # Subcommand: stats
    subparsers.add_parser("stats", help="Show learning statistics")

    # Subcommand: reset
    subparsers.add_parser("reset", help="Reset learning database")

    # Subcommand: export
    export_parser = subparsers.add_parser("export", help="Export learning database to JSON file")
    export_parser.add_argument("path", type=str, help="Target JSON file path")

    # Subcommand: import
    import_parser = subparsers.add_parser("import", help="Import learning database from JSON file")
    import_parser.add_argument("path", type=str, help="Source JSON file path")

    parsed = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    engine = RuntimeLearningEngine()

    if parsed.subcommand in (None, "show"):
        stats = engine.get_statistics()
        print("=== InferenceOS Adaptive Runtime Intelligence ===")
        print(f"Total Executions Recorded : {stats.total_executions}")
        print(f"Unique Models Learned     : {stats.unique_models_learned}")
        print(f"Unique Hardware Learned   : {stats.unique_hardware_learned}")
        print(f"Avg Throughput Gain       : +{stats.avg_throughput_gain_pct:.1f}%")
        print(f"Database Storage Size     : {stats.database_size_bytes} bytes")
        return 0

    elif parsed.subcommand == "stats":
        stats = engine.get_statistics()
        print("=== Runtime Learning Statistics ===")
        print(f"Total Executions  : {stats.total_executions}")
        print(f"Successful        : {stats.total_successful}")
        print(f"Failed            : {stats.total_failed}")
        print(f"Models Learned    : {stats.unique_models_learned}")
        print(f"Hardware Learned  : {stats.unique_hardware_learned}")
        print(f"Regressions       : {stats.regressions_detected}")
        return 0

    elif parsed.subcommand == "reset":
        engine.reset_database()
        print("Runtime Learning database successfully reset.")
        return 0

    elif parsed.subcommand == "export":
        target = Path(parsed.path)
        engine.export_database(target)
        print(f"Learning database successfully exported to {target.resolve()}")
        return 0

    elif parsed.subcommand == "import":
        source = Path(parsed.path)
        if not source.exists():
            print(f"Error: file not found at {source}")
            return 1
        engine.import_database(source)
        print(f"Learning database successfully imported from {source.resolve()}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(handle_learning_cli())
