"""
kv_cli.py
---------
CLI commands interface for Intelligent KV Manager in InferenceOS.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from kv_manager import IntelligentKVManager, KVQuantizer


def handle_kv_cli(args_list: Optional[List[str]] = None) -> int:
    """
    Handle CLI commands for Intelligent KV Manager.

    Subcommands:
      show     - Display current KV cache status
      stats    - Display compression and memory savings statistics
      policies - Display available compression and eviction policies
      reset    - Reset runtime KV statistics
    """
    parser = argparse.ArgumentParser(prog="inferenceos kv", description="InferenceOS Intelligent KV Manager CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Subcommand: show
    subparsers.add_parser("show", help="Show current KV cache status")

    # Subcommand: stats
    subparsers.add_parser("stats", help="Show KV compression and memory saved statistics")

    # Subcommand: policies
    subparsers.add_parser("policies", help="Show available compression and eviction policies")

    # Subcommand: reset
    subparsers.add_parser("reset", help="Reset runtime KV statistics")

    parsed = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    manager = IntelligentKVManager()

    if parsed.subcommand in (None, "show"):
        stats = manager.get_statistics()
        print("=== InferenceOS Intelligent KV Manager ===")
        print(f"Total Evaluations : {stats.total_evaluations}")
        print(f"Memory Saved      : {stats.total_memory_saved_mb:.2f} MB")
        print(f"Total Evictions   : {stats.total_evictions} tokens")
        print(f"Active Policy     : {stats.active_policy}")
        return 0

    elif parsed.subcommand == "stats":
        stats = manager.get_statistics()
        print("=== KV Cache Statistics ===")
        print(f"Evaluations Executed   : {stats.total_evaluations}")
        print(f"Total Memory Saved     : {stats.total_memory_saved_mb:.2f} MB")
        print(f"Evicted Tokens Count   : {stats.total_evictions}")
        print(f"Avg Compression Ratio  : {stats.avg_compression_ratio:.2f}x")
        return 0

    elif parsed.subcommand == "policies":
        print("=== Intelligent KV Manager Available Policies ===")
        print("Compression Modes : Disabled, Lossless (BF16), Balanced (INT8), Aggressive (INT4), Adaptive")
        print("Eviction Policies : LRU, FIFO, LFU, Adaptive, AttentionAware")
        print("Quantization Formats:")
        for fmt_name, fmt_info in KVQuantizer.FORMATS.items():
            print(f"  - {fmt_name:6s}: {fmt_info.bits_per_element} bits/element, Ratio: {fmt_info.memory_ratio:.2f}x, Quality: {int(fmt_info.quality_retention*100)}%")
        return 0

    elif parsed.subcommand == "reset":
        manager.reset()
        print("Intelligent KV Manager statistics successfully reset.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(handle_kv_cli())
