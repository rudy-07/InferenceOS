"""
optimizer_cli.py
----------------
CLI commands interface for Automatic Performance Optimizer (APO) and Model Fingerprinting in InferenceOS.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from optimizer import AutomaticPerformanceOptimizer, FingerprintEngine


def handle_optimizer_cli(args_list: Optional[List[str]] = None) -> int:
    """
    Handle CLI commands for APO and Model Fingerprinting.
    """
    parser = argparse.ArgumentParser(prog="inferenceos optimize", description="InferenceOS Automatic Performance Optimizer CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    opt_p = subparsers.add_parser("run", help="Optimize model runtime configuration")
    opt_p.add_argument("--force", "-f", action="store_true", help="Force re-optimization")
    opt_p.add_argument("--goal", "-g", default="Balanced", help="Optimization goal (Balanced, Throughput, Memory, Latency)")

    subparsers.add_parser("status", help="Show optimization status")
    subparsers.add_parser("history", help="Show optimization history")
    subparsers.add_parser("profiles", help="List stored optimization profiles")

    parsed = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    apo = AutomaticPerformanceOptimizer()

    if parsed.subcommand in (None, "run"):
        force_flag = getattr(parsed, "force", False)
        goal_val = getattr(parsed, "goal", "Balanced")
        profile = apo.get_or_create_profile(config_override=None)
        if force_flag:
            profile = apo.run_optimization(
                model_fp=FingerprintEngine.compute_model_fingerprint(),
                hw_fp=FingerprintEngine.compute_hardware_fingerprint(),
                goal=goal_val,
                verbose=True,
            )
        print(profile.format_cli_output())
        return 0

    elif parsed.subcommand == "status":
        profiles = apo.list_profiles()
        print("=== APO Optimization Status ===")
        print(f"Stored Profiles Count : {len(profiles)}")
        print(f"Active Status        : Ready")
        return 0

    elif parsed.subcommand == "history":
        history = apo.history_store.get_history()
        print("=== APO Optimization History ===")
        if not history:
            print("No optimization runs recorded yet.")
        else:
            for idx, entry in enumerate(history, 1):
                print(f"{idx}. Model: {entry.profile.model_name}, Goal: {entry.profile.goal}, TPS: {entry.profile.expected_tps:.1f}")
        return 0

    elif parsed.subcommand == "profiles":
        profiles = apo.list_profiles()
        print("=== Stored Optimization Profiles ===")
        if not profiles:
            print("No stored profiles found.")
        else:
            for p in profiles:
                print(f"- {p.model_name:15s} (Hash: {p.model_hash[:8]}, Placement: {p.best_candidate.gpu_layers} GPU, TPS: {p.expected_tps:.1f})")
        return 0

    return 0


def handle_fingerprint_cli(args_list: Optional[List[str]] = None) -> int:
    """
    Handle CLI commands for Model Fingerprinting.
    """
    parser = argparse.ArgumentParser(prog="inferenceos fingerprint", description="InferenceOS Model Fingerprinting CLI")
    parser.add_argument("model_path", nargs="?", default=None, help="Path to model file")
    parser.add_argument("--verify", action="store_true", help="Verify model identity")

    parsed = parser.parse_args(args_list if args_list is not None else sys.argv[1:])
    fp = FingerprintEngine.compute_model_fingerprint(model_path=parsed.model_path)

    print("=== InferenceOS Model Fingerprint ===")
    print(f"Model Name  : {fp.model_name}")
    print(f"SHA256 Hash : {fp.sha256_hash}")
    print(f"Parameters  : {fp.param_count_b:.1f}B")
    print(f"Quantization: {fp.quantization}")
    print(f"Layers      : {fp.n_layers}")
    print(f"Hidden Dim  : {fp.hidden_dim}")
    print(f"Context Max : {fp.context_capability}")
    return 0


if __name__ == "__main__":
    sys.exit(handle_optimizer_cli())
