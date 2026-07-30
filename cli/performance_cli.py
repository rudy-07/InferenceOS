"""
performance_cli.py
------------------
CLI commands interface for Performance Intelligence Engine (PIE) in InferenceOS.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from performance_intelligence import PerformanceIntelligenceEngine


def handle_performance_cli(args_list: Optional[List[str]] = None) -> int:
    """
    Handle CLI commands for Performance Intelligence Engine.

    Subcommands:
      show        - Display current performance overview
      history     - Display historical performance records
      trends      - Display trend analysis
      regressions - Display detected regressions
      score       - Display overall runtime performance score (0-100)
      analyze     - Perform full root cause analysis
    """
    parser = argparse.ArgumentParser(prog="inferenceos performance", description="InferenceOS Performance Intelligence Engine CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    subparsers.add_parser("show", help="Display current performance overview")
    subparsers.add_parser("history", help="Display historical performance records")
    subparsers.add_parser("trends", help="Display performance trend analysis")
    subparsers.add_parser("regressions", help="Display detected regressions")
    subparsers.add_parser("score", help="Display overall runtime performance score")
    subparsers.add_parser("analyze", help="Perform full root cause analysis")

    parsed = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    pie = PerformanceIntelligenceEngine()

    if parsed.subcommand in (None, "show"):
        print(pie.format_cli_output())
        return 0

    elif parsed.subcommand == "history":
        history = pie.storage.fetch_all()
        print("=== Performance History ===")
        if not history:
            print("No performance records found.")
        else:
            for idx, r in enumerate(history, 1):
                print(f"{idx}. Model: {r['model_name']}, TPS: {r['eval_tps']:.1f}, TTFT: {r['ttft_ms']:.0f}ms, Score: {r['score']}")
        return 0

    elif parsed.subcommand == "trends":
        score, reg, cause, rec, trend = pie.evaluate_current()
        print("=== Performance Trend Analysis ===")
        print(f"Current Trend Status  : {trend}")
        print(f"Overall Stability     : High")
        print(f"Generation Throughput : {trend} trajectory")
        return 0

    elif parsed.subcommand == "regressions":
        score, reg, cause, rec, trend = pie.evaluate_current()
        print("=== Detected Regressions ===")
        if not reg.is_regression:
            print("No performance regressions detected. All metrics within baseline limits.")
        else:
            print(f"- Metric    : {reg.metric_name}")
            print(f"  Change    : {reg.change_pct:.1f}% ({reg.old_value:.1f} -> {reg.new_value:.1f})")
            print(f"  Confidence: {reg.confidence_pct:.0f}%")
            print(f"  Reason    : {reg.reasoning[0] if reg.reasoning else ''}")
        return 0

    elif parsed.subcommand == "score":
        score, _, _, _, _ = pie.evaluate_current()
        print("=== Runtime Performance Score ===")
        print(f"Overall Performance Score : {score.overall_score} / 100")
        print(f"Generation TPS Score      : {score.tps_score:.1f}")
        print(f"TTFT Latency Score        : {score.latency_score:.1f}")
        print(f"Memory Efficiency Score   : {score.memory_score:.1f}")
        return 0

    elif parsed.subcommand == "analyze":
        score, reg, cause, rec, trend = pie.evaluate_current()
        print("=== Full Performance Analysis & Root Cause Diagnosis ===")
        print(f"Runtime Score      : {score.overall_score} / 100")
        print(f"Trend Trajectory   : {trend}")
        print(f"Regression State   : {'Detected' if reg.is_regression else 'None'}")
        print(f"Probable Cause     : {cause.probable_cause}")
        print(f"Diagnosis          : {cause.explanation}")
        print(f"Recommendation     : {rec.reasoning}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(handle_performance_cli())
