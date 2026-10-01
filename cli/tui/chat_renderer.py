"""
chat_renderer.py
----------------
Visual Formatting & Deliberation Flowcard Renderer for InferenceOS TUI.

Provides sleek terminal cards and charts inspired by modern Gen-AI CLIs:
- Rich message cards with rounded borders and theme accent rails
- Dynamic probability bar charts for System 1 choices
- Multi-stage Kahneman deliberation flowcards for Hybrid Symbiosis
- Markdown rendering with code syntax highlighting and token performance pills
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from rich.box import ROUNDED, DOUBLE, SIMPLE
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class ChatRenderer:
    """
    Renders high-aesthetic terminal message cards, flowcharts, and telemetry pills.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def render_user_prompt(self, prompt: str) -> None:
        """Render user input bubble with modern accent rail."""
        txt = Text()
        txt.append("❯ User ", style="bold cyan")
        txt.append(f"({time.strftime('%H:%M:%S')})\n", style="dim")
        txt.append(prompt, style="white")

        self.console.print(Panel(txt, box=ROUNDED, border_style="cyan", padding=(0, 1)))

    def render_system1_card(
        self,
        res: Dict[str, Any],
        state_summary: Optional[str] = None,
    ) -> None:
        """Render structured card for System 1 reflex decisions."""
        if not isinstance(res, dict):
            res = {}
        latency = float(res.get("latency_ms") or 0.0)
        answers = res.get("answers") or {}

        card_table = Table(box=None, show_header=False, expand=True, pad_edge=False)
        card_table.add_column("Key", style="bold yellow", ratio=1)
        card_table.add_column("Value", ratio=3)

        if state_summary:
            card_table.add_row("Task State", f"[white]{state_summary}[/white]")

        for qid, ans in answers.items():
            if not isinstance(ans, dict):
                continue
            q_type = ans.get("type", "choice")
            conf_val = ans.get("confidence")
            conf = (float(conf_val) if conf_val is not None else 0.0) * 100.0

            if q_type == "choice":
                chosen = ans.get("choice", "N/A")
                val_text = f"[bold green]{chosen}[/bold green] [dim](Confidence: {conf:.1f}%)[/dim]\n"

                # Probability bars
                probs = ans.get("probabilities") or {}
                if isinstance(probs, dict) and probs:
                    bar_table = Table(box=None, show_header=False, pad_edge=False)
                    bar_table.add_column("Option", style="bold")
                    bar_table.add_column("Prob", justify="right", style="cyan")
                    bar_table.add_column("Bar", style="green")

                    for opt, p in sorted(probs.items(), key=lambda kv: -(float(kv[1]) if kv[1] is not None else 0.0)):
                        try:
                            p_val = float(p) if p is not None else 0.0
                        except (ValueError, TypeError):
                            p_val = 0.0
                        bar_len = max(0, min(24, int(round(p_val * 24))))
                        bar = "█" * bar_len + "░" * (24 - bar_len)
                        bar_table.add_row(f"  {opt}", f"{p_val * 100.0:5.1f}%", f"[{bar}]")
                    card_table.add_row(f"Decision ({qid})", val_text)
                    card_table.add_row("Probabilities", bar_table)
                else:
                    card_table.add_row(f"Decision ({qid})", val_text)

            elif q_type == "score":
                score = float(ans.get("score") or 0.0)
                card_table.add_row(f"Score ({qid})", f"[bold cyan]{score:.2f} / 5.0[/bold cyan] [dim](Confidence: {conf:.1f}%)[/dim]")

            elif q_type == "noul":
                prob = float(ans.get("noul") or 0.0)
                status_str = "[bold green]PASS / SAFE[/bold green]" if prob >= 0.70 else "[bold red]FAIL / UNSAFE[/bold red]"
                card_table.add_row(f"Safety Gate ({qid})", f"{status_str} [dim]({prob * 100.0:.1f}% threshold score)[/dim]")

        header_title = f"[bold yellow]⚡ System 1 Reflex[/bold yellow] [dim]• Latency: {latency:.1f}ms • 0 KV Cache[/dim]"
        self.console.print(Panel(card_table, title=header_title, box=ROUNDED, border_style="yellow", padding=(0, 1)))

    def render_hybrid_flowcard(
        self,
        res: Dict[str, Any],
        state_summary: Optional[str] = None,
    ) -> None:
        """Render dual-stage Kahneman deliberation flowcard for Hybrid mode."""
        if not isinstance(res, dict):
            res = {}
        s1_ans = res.get("answers") or {}
        escalated = bool(res.get("escalated", False))
        esc_reason = res.get("escalation_reason", "")
        total_lat = float(res.get("total_latency_ms") or res.get("latency_ms") or 0.0)
        s1_lat = float(res.get("system1_latency_ms") or res.get("latency_ms") or 0.0)
        s2_lat = float(res.get("system2_latency_ms") or 0.0)
        gen_text = res.get("generated_text")

        content = Table(box=None, show_header=False, expand=True, pad_edge=False)
        content.add_column("Section", ratio=1)

        # Stage 1: Reflex evaluation summary
        reflex_text = Text()
        reflex_text.append("⚡ Stage 1: System 1 Fast Reflex\n", style="bold yellow")
        for qid, ans in s1_ans.items():
            if isinstance(ans, dict):
                c = ans.get("choice", "N/A")
                conf_val = ans.get("confidence")
                conf = (float(conf_val) if conf_val is not None else 0.0) * 100.0
                reflex_text.append(f"  • {qid}: ", style="bold")
                reflex_text.append(f"{c} ", style="green")
                reflex_text.append(f"(Confidence: {conf:.1f}%)\n", style="dim")

        content.add_row(reflex_text)

        # Stage 2: Deliberation status
        if escalated:
            handoff = Text()
            handoff.append("\n🧠 Stage 2: System 2 Teacher Deliberation Escalated\n", style="bold magenta")
            handoff.append(f"  Reason: {esc_reason}\n", style="italic white")
            content.add_row(handoff)

            if gen_text:
                resp_panel = Panel(
                    Markdown(str(gen_text)),
                    title="[bold green]Teacher Synthesized Response[/bold green]",
                    box=ROUNDED,
                    border_style="green",
                )
                content.add_row(resp_panel)
        else:
            bypass = Text()
            bypass.append("\n✔ System 1 Confidence Passed (No Teacher Escalation Needed)\n", style="bold green")
            content.add_row(bypass)

        # Footer latency breakdown
        perf_text = Text()
        perf_text.append(f"\n⏱ Total: {total_lat:.1f}ms ", style="bold white")
        perf_text.append(f"(S1 Reflex: {s1_lat:.1f}ms", style="dim")
        if escalated and s2_lat > 0:
            perf_text.append(f" + S2 Teacher: {s2_lat:.1f}ms", style="dim")
        perf_text.append(")", style="dim")
        content.add_row(perf_text)

        title = "[bold magenta]⚡🧠 Kahneman Cognitive Symbiosis Loop[/bold magenta]"
        self.console.print(Panel(content, title=title, box=ROUNDED, border_style="magenta", padding=(0, 1)))

    def render_streaming_pill(self, st: Any, backend: str) -> None:
        """Render performance stats pill after assistant token streaming."""
        gen_tps = getattr(st, "eval_tps", 0.0) if not isinstance(st, dict) else st.get("eval_tps", 0.0)
        prompt_tps = getattr(st, "prompt_eval_tps", 0.0) if not isinstance(st, dict) else st.get("prompt_eval_tps", 0.0)
        ttft = getattr(st, "prompt_eval_ms", 0.0) if not isinstance(st, dict) else st.get("prompt_eval_ms", 0.0)
        tokens = getattr(st, "tokens_generated", 0) if not isinstance(st, dict) else st.get("tokens_generated", 0)

        pill = Text()
        pill.append("⚡ ", style="bold yellow")
        pill.append(f"Gen: {float(gen_tps or 0.0):.2f} tok/s  │  ", style="bold green")
        pill.append(f"Prompt: {float(prompt_tps or 0.0):.1f} tok/s  │  ", style="cyan")
        pill.append(f"TTFT: {float(ttft or 0.0):.1f} ms  │  ", style="yellow")
        pill.append(f"Tokens: {int(tokens or 0)}  │  ", style="white")
        pill.append(f"Backend: {str(backend or 'AUTO').upper()}", style="magenta")

        self.console.print(Panel(pill, box=ROUNDED, border_style="bright_black", padding=(0, 1)))
