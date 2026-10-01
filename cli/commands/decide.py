"""
decide.py
---------
Command handler for 'inferenceos decide <model> [options]'.
Direct CLI tool for non-autoregressive System 1 decisions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from cli.core.model_registry import get_model_registry
from orchestrator.hybrid_orchestrator import HybridOrchestrator, DEFAULT_ESCALATE_TAU
from orchestrator.model_parser import detect_model_format, ModelFormat


def _robust_parse_json(raw: str) -> Any:
    """Robustly parse JSON or Python dict syntax across Windows PowerShell/cmd escaping quirks."""
    cleaned = raw.strip()
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (cleaned.startswith('"') and cleaned.endswith('"')):
        # Only unwrap if valid JSON structure inside
        inner = cleaned[1:-1].strip()
        if (inner.startswith("{") and inner.endswith("}")) or (inner.startswith("[") and inner.endswith("]")):
            cleaned = inner

    # 1. Direct JSON parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Unescape escaped double/single quotes from shell arguments
    try:
        unescaped = cleaned.replace(r'\"', '"').replace(r"\'", "'")
        return json.loads(unescaped)
    except Exception:
        pass

    # 3. AST literal eval (Python dictionary syntax)
    try:
        import ast
        return ast.literal_eval(cleaned)
    except Exception:
        pass

    try:
        import ast
        return ast.literal_eval(cleaned.replace(r'\"', '"'))
    except Exception:
        pass

    # 4. Replace single quotes with double quotes
    try:
        import re
        fixed = re.sub(r"'([^']*)'", r'"\1"', cleaned)
        return json.loads(fixed)
    except Exception:
        pass

    # 5. Clean stray PowerShell quote-escaping backslashes (e.g. {\ action\: {\type\: ...}})
    try:
        import re
        import yaml
        ps_cleaned = re.sub(r'\\+', '', cleaned)
        res = yaml.safe_load(ps_cleaned)
        if isinstance(res, (dict, list)):
            return res
    except Exception:
        pass

    # 6. PyYAML relaxed parsing (handles unquoted keys/values when shell strips quotes)
    try:
        import yaml
        res = yaml.safe_load(cleaned)
        if isinstance(res, (dict, list)):
            return res
    except Exception:
        pass

    raise ValueError(f"Invalid JSON/dict format: {raw[:100]}")


def handle_decide_command(
    model_query: str,
    state: Union[str, List[str], None] = None,
    questions: Union[str, List[str], None] = None,
    max_options: int = 60,
    as_json: bool = False,
    escalate: bool = False,
    escalate_tau: float = DEFAULT_ESCALATE_TAU,
    system2: Optional[str] = None,
    system2_url: Optional[str] = None,
    dagger_log: Optional[str] = None,
) -> None:
    """Execute decision inference against a System 1 model with optional System 2 escalation."""
    console = Console()
    registry = get_model_registry()

    if isinstance(state, (list, tuple)):
        state = " ".join(state) if state else None
    if isinstance(questions, (list, tuple)):
        questions = " ".join(questions) if questions else None

    model_path = registry.resolve_model_path(model_query)
    if not model_path or not model_path.exists():
        console.print(f"[bold red]Error: Could not resolve model '{model_query}'[/bold red]")
        sys.exit(1)

    fmt = detect_model_format(model_path)
    if fmt != ModelFormat.SYSTEM1:
        console.print(f"[bold yellow]Warning: Model '{model_path.name}' is format '{fmt.value}'. 'decide' is optimized for System 1 models.[/bold yellow]")

    # Parse state: file path or string
    state_payload = state or ""
    if state and Path(state).exists() and Path(state).is_file():
        try:
            with open(state, "r", encoding="utf-8") as f:
                content = f.read().strip()
                state_payload = json.loads(content) if (content.startswith("{") or content.startswith("[")) else content
        except Exception:
            state_payload = state
    elif state and (state.strip().startswith("{") or state.strip().startswith("[")):
        try:
            state_payload = _robust_parse_json(state)
        except Exception:
            pass

    # Parse questions: file path or JSON string
    questions_payload = {}
    if questions:
        if Path(questions).exists() and Path(questions).is_file():
            try:
                with open(questions, "r", encoding="utf-8") as f:
                    questions_payload = json.load(f)
            except Exception as e:
                console.print(f"[red]Error reading questions file '{questions}': {e}[/red]")
                sys.exit(1)
        else:
            try:
                questions_payload = _robust_parse_json(questions)
            except Exception as e:
                console.print(f"[red]Error parsing questions JSON: {e}[/red]")
                sys.exit(1)
    else:
        # Default decision question
        questions_payload = {
            "decision": {
                "type": "choice",
                "instructions": {"goal": str(state_payload)[:300] if state_payload else "Evaluate input"},
                "criteria": {
                    "YES": "Affirmative / Proceed",
                    "NO": "Negative / Reject",
                    "UNCERTAIN": "Needs further context",
                },
            }
        }

    orchestrator = HybridOrchestrator()
    try:
        res = orchestrator.decide_with_escalation(
            model_path=model_path,
            state=state_payload,
            questions=questions_payload,
            escalate=escalate,
            escalate_tau=escalate_tau,
            system2_model=system2,
            system2_base_url=system2_url,
            dagger_log_path=dagger_log,
            max_options=max_options,
        )

        if as_json:
            print(json.dumps(res, indent=2))
            return

        is_escalated = res.get("escalated", False)
        if is_escalated:
            console.print(
                f"\n[bold cyan]System 1 + System 2 Hybrid Decision[/bold cyan] "
                f"[dim](Total: {res['latency_ms']:.1f}ms | S1: {res.get('s1_latency_ms', 0):.1f}ms | S2: {res.get('s2_latency_ms', 0):.1f}ms | Backend: {res.get('backend')})[/dim]"
            )
            console.print(f"[bold magenta]⚡ Escalated to System 2 Teacher[/bold magenta] [dim]Reason: {res.get('escalate_reason')}[/dim]\n")
        else:
            console.print(
                f"\n[bold cyan]System 1 Decision Results[/bold cyan] "
                f"[dim](Latency: {res['latency_ms']:.1f}ms | Passes: {res.get('passes', 1)} | Backend: {res.get('backend', 'system1')})[/dim]\n"
            )

        for qid, ans in res["answers"].items():
            t = ans.get("type", "choice")
            conf = ans.get("confidence", 0.0)
            is_s2_ans = ans.get("system2", False)
            badge = "[bold magenta][System 2][/bold magenta] " if is_s2_ans else ""

            if t == "choice":
                chosen = ans.get("choice", "")
                console.print(f"[bold green]• {qid}:[/bold green] {badge}[bold yellow]{chosen}[/bold yellow] (Confidence: [bold]{conf:.2f}[/bold])")
                if is_s2_ans and ans.get("rationale"):
                    console.print(f"  [dim italic]Rationale: {ans['rationale']}[/dim italic]")

                probs = ans.get("probabilities", {})
                if probs:
                    table = Table(box=None, show_header=False, pad_edge=False)
                    table.add_column("Option", style="bold")
                    table.add_column("Prob", justify="right", style="cyan")
                    table.add_column("Bar", style="green")

                    for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:8]:
                        bar_len = int(round(v * 20))
                        bar = "#" * bar_len + "-" * (20 - bar_len)
                        table.add_row(f"  {k}", f"{v:5.2f}", f"[{bar}]")
                    console.print(table)
            elif t == "score":
                console.print(f"[bold green]• {qid}:[/bold green] {badge}Score = [bold yellow]{ans.get('score', 0):.2f}[/bold yellow] (Confidence: {conf:.2f})")
            elif t == "noul":
                console.print(f"[bold green]• {qid}:[/bold green] {badge}P(true) = [bold yellow]{ans.get('noul', 0.0):.2f}[/bold yellow]")

        if res.get("generated_text"):
            console.print(f"\n[bold green]System 2 Generated Synthesis:[/bold green]\n{res['generated_text']}\n")

        if is_escalated and dagger_log:
            console.print(f"[dim]DAgger distillation sample appended to '{dagger_log}'[/dim]")

    except Exception as e:
        console.print(f"[bold red]System 1 execution error: {e}[/bold red]")
        sys.exit(1)
