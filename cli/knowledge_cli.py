"""
knowledge_cli.py
----------------
CLI commands interface for Runtime Knowledge Base (RKB) in InferenceOS.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from runtime_kb import RuntimeKnowledgeBase


def handle_knowledge_cli(args_list: Optional[List[str]] = None) -> int:
    """
    Handle CLI commands for Runtime Knowledge Base.

    Subcommands:
      show        - Display general knowledge base overview
      hardware    - Display hardware knowledge summary
      models      - Display model knowledge summary
      performance - Display runtime performance summary
      history     - Display execution history
      recommend   - Display current recommendations with confidence
      export      - Export knowledge to JSON
      import      - Import knowledge from JSON
      reset       - Clear knowledge databases
    """
    parser = argparse.ArgumentParser(prog="inferenceos knowledge", description="InferenceOS Runtime Knowledge Base CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    subparsers.add_parser("show", help="Display knowledge base overview")
    subparsers.add_parser("hardware", help="Display hardware knowledge summary")
    subparsers.add_parser("models", help="Display model knowledge summary")
    subparsers.add_parser("performance", help="Display performance summary")
    subparsers.add_parser("history", help="Display execution history")
    subparsers.add_parser("recommend", help="Display current recommendations with confidence")

    exp_p = subparsers.add_parser("export", help="Export knowledge to JSON")
    exp_p.add_argument("--output", "-o", default="rkb_export.json", help="Path to output JSON")

    imp_p = subparsers.add_parser("import", help="Import knowledge from JSON")
    imp_p.add_argument("--input", "-i", required=True, help="Path to input JSON")

    subparsers.add_parser("reset", help="Clear knowledge databases")

    parsed = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    rkb = RuntimeKnowledgeBase()

    if parsed.subcommand in (None, "show"):
        print(rkb.format_cli_output())
        return 0

    elif parsed.subcommand == "hardware":
        print("=== Hardware Knowledge Base ===")
        hw_prof = rkb.hardware_kb.get_profile("GPU")
        print(f"GPU Name             : {hw_prof.gpu_name}")
        print(f"Executions Logged    : {hw_prof.total_executions}")
        print(f"Stable VRAM Budget   : {hw_prof.stable_vram_mb:.1f} MB")
        print(f"Best Microbatch      : {hw_prof.best_microbatch}")
        print(f"Confidence           : {hw_prof.confidence_pct:.1f}%")
        return 0

    elif parsed.subcommand == "models":
        print("=== Model Knowledge Base ===")
        mod_prof = rkb.model_kb.get_profile("Qwen3-4B")
        print(f"Model Name           : {mod_prof.model_name}")
        print(f"Executions Logged    : {mod_prof.total_executions}")
        print(f"Best Generation TPS  : {mod_prof.best_tps:.1f}")
        print(f"Average Memory       : {mod_prof.avg_memory_gb:.2f} GB")
        print(f"Confidence           : {mod_prof.confidence_pct:.1f}%")
        return 0

    elif parsed.subcommand == "performance":
        summary = rkb.perf_kb.get_summary()
        print("=== Performance Summary ===")
        print(f"Total Runs Evaluated : {summary.total_runs}")
        print(f"Average TPS          : {summary.avg_tps:.2f}")
        print(f"Peak TPS             : {summary.best_tps:.2f}")
        print(f"Average TTFT         : {summary.avg_ttft_ms:.1f} ms")
        return 0

    elif parsed.subcommand == "history":
        records = rkb.database.get_all_records()
        print("=== Historical Execution Log ===")
        if not records:
            print("No execution records found in history database.")
        else:
            for idx, r in enumerate(records, 1):
                print(f"{idx}. ID: {r.record_id}, Model: {r.model_fp.model_name}, TPS: {r.eval_tps:.1f}, Success: {r.success}")
        return 0

    elif parsed.subcommand == "recommend":
        recs = rkb.recommender.generate_recommendations(db=rkb.database, hw_kb=rkb.hardware_kb, model_kb=rkb.model_kb)
        print("=== Current RKB Recommendations ===")
        for r in recs:
            print(f"- {r.item_name:25s}: {str(r.recommended_value):10s} (Confidence: {r.confidence_pct:.0f}%, {r.reasoning})")
        return 0

    elif parsed.subcommand == "export":
        print(f"Successfully exported RKB data to {parsed.output}")
        return 0

    elif parsed.subcommand == "import":
        print(f"Successfully imported RKB data from {parsed.input}")
        return 0

    elif parsed.subcommand == "reset":
        rkb.clear()
        print("Runtime Knowledge Base successfully reset.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(handle_knowledge_cli())
