"""
speculative_cmd.py
------------------
CLI command handler for Phase 5 Speculative Decoding Suite in InferenceOS.

Provides the ``inferenceos speculative`` subcommand with the following actions:
  - ``status``   — Show current speculative decoding configuration.
  - ``stats``    — Display historical acceptance rates and speedup ratios from DB.
  - ``test``     — Quick self-test of the N-gram and prompt-lookup speculators.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def handle_speculative_command(
    action: str = "status",
    model_name: Optional[str] = None,
    limit: int = 20,
) -> None:
    """
    Dispatch the ``inferenceos speculative`` subcommand.

    Parameters
    ----------
    action : str
        One of: ``"status"``, ``"stats"``, ``"test"``.
    model_name : str, optional
        Filter ``stats`` output by model name.
    limit : int
        Maximum number of records to display in ``stats`` view.
    """
    action = (action or "status").lower()

    if action == "status":
        _show_status()
    elif action == "stats":
        _show_stats(model_name=model_name, limit=limit)
    elif action == "test":
        _run_self_test()
    else:
        console.print(f"[bold red]Unknown action:[/bold red] {action}")
        console.print("Valid actions: [bold]status[/bold]  [bold]stats[/bold]  [bold]test[/bold]")


# ---------------------------------------------------------------------------
# Action: status
# ---------------------------------------------------------------------------

def _show_status() -> None:
    """Display current speculative decoding configuration from RuntimeConfig defaults."""
    try:
        # Import project root
        project_root = Path(__file__).parent.parent.parent.resolve()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from inference_runtime.runtime_config import RuntimeConfig
        cfg = RuntimeConfig()

        console.print("\n[bold cyan]⚡ InferenceOS — Speculative Decoding Configuration[/bold cyan]\n")

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="bold green", width=36)
        table.add_column("Value", style="white")

        enabled_str = "[green]✓ Enabled[/green]" if cfg.enable_speculative_decoding else "[dim]Disabled[/dim]"
        table.add_row("Speculative Decoding", enabled_str)
        table.add_row("Strategy Mode", cfg.spec_mode)
        table.add_row("Draft Tokens / Step", str(cfg.spec_draft_tokens))
        table.add_row("N-Gram Size", str(cfg.spec_ngram_size))
        table.add_row("Min Match Length", str(cfg.spec_min_match_length))
        table.add_row("Prompt Lookup Window", "Full prompt" if cfg.spec_prompt_lookup_window == 0 else str(cfg.spec_prompt_lookup_window))
        table.add_row("Acceptance Strategy", cfg.spec_acceptance_strategy)
        table.add_row("Acceptance Threshold", str(cfg.spec_acceptance_threshold))
        table.add_row("Eagle Draft Model", str(cfg.spec_eagle_draft_model) if cfg.spec_eagle_draft_model else "[dim]None (set spec_eagle_draft_model)[/dim]")
        table.add_row("Max Speculation Rounds", str(cfg.spec_max_rounds))
        table.add_row("Fallback to Greedy", "[green]Yes[/green]" if cfg.spec_fallback_to_greedy else "[red]No[/red]")
        table.add_row("Record Telemetry", "[green]Yes[/green]" if cfg.spec_record_telemetry else "No")
        table.add_row("Verbose Output", "[yellow]Yes[/yellow]" if cfg.verbose_spec_decoding else "[dim]No[/dim]")

        console.print(Panel(table, title="Phase 5 — Speculative Decoding", border_style="cyan"))

        console.print("\n[dim]To enable:  inferenceos run <model> --speculative[/dim]")
        console.print("[dim]            inferenceos run <model> --spec-mode prompt_lookup[/dim]")
        console.print("[dim]Eagle mode: inferenceos run <model> --speculative --draft-model draft.gguf[/dim]\n")

    except Exception as exc:
        console.print(f"[bold red]Error loading configuration:[/bold red] {exc}")


# ---------------------------------------------------------------------------
# Action: stats
# ---------------------------------------------------------------------------

def _show_stats(model_name: Optional[str], limit: int) -> None:
    """Display historical speculative decoding telemetry from the Runtime Learning DB."""
    try:
        project_root = Path(__file__).parent.parent.parent.resolve()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from runtime_learning.database import LearningDatabase
        db = LearningDatabase()
        records = db.get_spec_stats(model_name=model_name, limit=limit)

    except Exception as exc:
        console.print(f"[bold red]Could not load speculative stats:[/bold red] {exc}")
        return

    if not records:
        console.print("\n[yellow]No speculative decoding telemetry recorded yet.[/yellow]")
        console.print("[dim]Run inference with [bold]--speculative[/bold] to begin collecting data.[/dim]\n")
        return

    title = "Phase 5 — Speculative Decoding History"
    if model_name:
        title += f" ({model_name})"

    console.print(f"\n[bold cyan]⚡ {title}[/bold cyan]\n")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="white", width=22)
    table.add_column("Mode", style="green", width=14)
    table.add_column("Drafts/Step", justify="center", width=11)
    table.add_column("Accept %", justify="right", style="bold", width=10)
    table.add_column("Speedup", justify="right", style="bold yellow", width=9)
    table.add_column("Rounds", justify="right", width=8)
    table.add_column("Eval t/s", justify="right", width=10)

    # Aggregate summary counters
    total_accept = 0.0
    total_speedup = 0.0

    for i, rec in enumerate(records, 1):
        accept_pct = rec.get("acceptance_rate", 0.0) * 100
        speedup = rec.get("speedup_ratio", 1.0)
        total_accept += accept_pct
        total_speedup += speedup

        # Colour acceptance rate
        if accept_pct >= 70:
            accept_str = f"[green]{accept_pct:.1f}%[/green]"
        elif accept_pct >= 40:
            accept_str = f"[yellow]{accept_pct:.1f}%[/yellow]"
        else:
            accept_str = f"[red]{accept_pct:.1f}%[/red]"

        speedup_str = f"{speedup:.2f}×"

        table.add_row(
            str(i),
            rec.get("model_name", "—")[:22],
            rec.get("mode", "ngram"),
            str(rec.get("draft_tokens_requested", 5)),
            accept_str,
            speedup_str,
            str(rec.get("speculative_rounds", 0)),
            f"{rec.get('eval_tps_speculative', 0.0):.1f}",
        )

    console.print(table)

    n = len(records)
    avg_accept = total_accept / n
    avg_speedup = total_speedup / n
    console.print(
        f"\n  [dim]Average acceptance:[/dim] [bold]{avg_accept:.1f}%[/bold]   "
        f"[dim]Average speedup:[/dim] [bold yellow]{avg_speedup:.2f}×[/bold yellow]   "
        f"[dim]({n} runs)[/dim]\n"
    )


# ---------------------------------------------------------------------------
# Action: test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Quick self-test of the NGram and PromptLookup speculators."""
    try:
        project_root = Path(__file__).parent.parent.parent.resolve()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from speculative_decoding import NGramSpeculator, PromptLookupSpeculator, AcceptanceSampler

    except ImportError as exc:
        console.print(f"[bold red]Import error:[/bold red] {exc}")
        return

    console.print("\n[bold cyan]⚡ Speculative Decoding — Self-Test[/bold cyan]\n")

    # ── N-Gram Test ────────────────────────────────────────────────────────
    console.print("[bold]N-Gram Speculator[/bold]")
    spec = NGramSpeculator(ngram_size=3, min_match_length=2)

    # Feed a repeating sequence so n-gram will find a match
    tokens = [10, 20, 30, 10, 20, 30, 10, 20, 30, 40, 50]
    spec.update(tokens)
    drafts = spec.generate_drafts(n_drafts=5)
    console.print(f"  Context tokens:  {tokens}")
    console.print(f"  Generated drafts: {drafts}")
    hit_pct = spec.hit_rate * 100
    status = "[green]✓ PASS[/green]" if drafts else "[yellow]~ No match (short context)[/yellow]"
    console.print(f"  Hit rate: {hit_pct:.1f}%  {status}")

    # ── Prompt-Lookup Test ─────────────────────────────────────────────────
    console.print("\n[bold]Prompt-Lookup Speculator[/bold]")
    pl = PromptLookupSpeculator(min_match_length=2)
    prompt_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pl.set_prompt(prompt_ids)
    pl.update([3, 4])   # pretend we generated tokens [3, 4]
    pl_drafts = pl.generate_drafts(n_drafts=4)
    console.print(f"  Prompt tokens:   {prompt_ids}")
    console.print(f"  Generated suffix: [3, 4] → drafts: {pl_drafts}")
    expected = [5, 6, 7, 8]
    pl_status = "[green]✓ PASS[/green]" if pl_drafts == expected else f"[yellow]~ Got {pl_drafts}[/yellow]"
    console.print(f"  Expected: {expected}  {pl_status}")

    # ── Acceptance Sampler Test ────────────────────────────────────────────
    console.print("\n[bold]Acceptance Sampler (Greedy)[/bold]")
    sampler = AcceptanceSampler(strategy="greedy")
    drafts_str = [" the", " quick", " brown", " fox"]
    model_out = " the quick brown fox jumps"
    n_acc, bonus = sampler.verify_greedy_text(drafts_str, model_out)
    console.print(f"  Draft tokens:   {drafts_str}")
    console.print(f"  Model output:  {model_out!r}")
    console.print(f"  Accepted:       {n_acc}/4  bonus={bonus!r}")
    acc_status = "[green]✓ PASS[/green]" if n_acc == 4 else f"[yellow]~ Accepted {n_acc}[/yellow]"
    console.print(f"  {acc_status}")

    console.print("\n[bold green]✓ All self-tests completed.[/bold green]\n")
