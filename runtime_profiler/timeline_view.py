"""
timeline_view.py
----------------
Timeline view generator for Phase 9 Runtime Profiler.

Generates interactive Gantt charts and timeline event logs showing:
  - Token arrival events (TTFT and inter-token generation timestamps).
  - 3-stage pipeline activities (Prepare, Transfer, Compute).
  - GPU active vs. idle intervals.
  - PCIe boundary transfer channels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .telemetry_collector import ProfilerTelemetry


# ---------------------------------------------------------------------------
# TimelineEvent
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """
    Individual event block on the timeline chart.

    Attributes
    ----------
    name : str
        Event title (e.g., "Token #3", "GPU Layer 0-48").
    track : str
        Track name ("Tokens", "GPU Compute", "PCIe Bus", "CPU Prepare").
    start_ms : float
        Start timestamp in milliseconds relative to run start.
    duration_ms : float
        Duration in milliseconds.
    category : str
        Category for coloring: "gpu", "cpu", "pcie", "token", "stall".
    details : dict
        Optional metadata dict.
    """
    name: str
    track: str
    start_ms: float
    duration_ms: float
    category: str = "gpu"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "track": self.track,
            "start_ms": round(self.start_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "end_ms": round(self.start_ms + self.duration_ms, 3),
            "category": self.category,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# TimelineGenerator
# ---------------------------------------------------------------------------

class TimelineGenerator:
    """
    Generates interactive timeline Gantt charts from ProfilerTelemetry data.
    """

    TRACK_COLORS = {
        "token": "#38bdf8",     # Sky Blue
        "gpu": "#10b981",       # Emerald Green
        "cpu": "#6366f1",       # Indigo
        "pcie": "#f59e0b",      # Amber
        "stall": "#ef4444",     # Red
    }

    def __init__(self, telemetry: ProfilerTelemetry) -> None:
        self.telemetry = telemetry
        self.events = self._build_events()

    def _build_events(self) -> List[TimelineEvent]:
        """Construct chronologically ordered timeline events."""
        t = self.telemetry
        events: List[TimelineEvent] = []

        curr_time = 0.0

        # Stage 1: Prepare (Prompt Prefill)
        prep_dur = max(0.1, t.prompt_eval_ms if t.prompt_eval_ms > 0 else 10.0)
        events.append(
            TimelineEvent(
                name="Prompt Tokenization & Prefill",
                track="CPU Prepare",
                start_ms=curr_time,
                duration_ms=prep_dur,
                category="cpu",
                details={"tokens": t.prompt_tokens},
            )
        )
        curr_time += prep_dur

        # Inter-token generation events
        if t.inter_token_latencies_ms:
            for idx, itl in enumerate(t.inter_token_latencies_ms, start=1):
                # Token arrival event
                events.append(
                    TimelineEvent(
                        name=f"Token #{idx}",
                        track="Token Arrival",
                        start_ms=curr_time,
                        duration_ms=2.0,
                        category="token",
                        details={"itl_ms": round(itl, 2)},
                    )
                )

                # GPU compute event vs PCIe transfer event
                if t.n_gpu_layers > 0:
                    gpu_dur = itl * (1.0 - (t.gpu_idle_pct / 100.0))
                    idle_dur = itl - gpu_dur

                    if gpu_dur > 0:
                        events.append(
                            TimelineEvent(
                                name=f"GPU Compute #{idx}",
                                track="GPU Execution",
                                start_ms=curr_time,
                                duration_ms=gpu_dur,
                                category="gpu",
                            )
                        )

                    if idle_dur > 1.0:
                        events.append(
                            TimelineEvent(
                                name=f"PCIe / CPU Wait #{idx}",
                                track="PCIe Bus",
                                start_ms=curr_time + gpu_dur,
                                duration_ms=idle_dur,
                                category="stall" if idle_dur > 15.0 else "pcie",
                            )
                        )

                curr_time += itl
        else:
            # Synthetic generation block if no per-token timestamps
            gen_dur = max(0.1, t.generation_eval_ms if t.generation_eval_ms > 0 else 50.0)
            events.append(
                TimelineEvent(
                    name=f"Generation ({t.generation_tokens} tokens)",
                    track="GPU Execution" if t.n_gpu_layers > 0 else "CPU Execution",
                    start_ms=curr_time,
                    duration_ms=gen_dur,
                    category="gpu" if t.n_gpu_layers > 0 else "cpu",
                )
            )

        return events

    def to_json(self, indent: int = 2) -> str:
        """Return JSON list of all timeline events."""
        return json.dumps([e.to_dict() for e in self.events], indent=indent)

    def to_html(self) -> str:
        """Generate interactive self-contained HTML Gantt timeline view."""
        events_json = self.to_json(indent=2)
        total_span = max(1.0, max(e.start_ms + e.duration_ms for e in self.events))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InferenceOS Phase 9 — Execution Timeline</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #0b132b;
            color: #f1f5f9;
            margin: 0;
            padding: 24px;
        }}
        h1 {{
            font-size: 24px;
            color: #38bdf8;
            margin-bottom: 8px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .timeline-card {{
            background: #1c2541;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        }}
        .track-row {{
            margin-bottom: 20px;
        }}
        .track-label {{
            font-size: 13px;
            font-weight: 600;
            color: #94a3b8;
            margin-bottom: 6px;
        }}
        .track-bar-container {{
            position: relative;
            height: 32px;
            background: #0b132b;
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid #334155;
        }}
        .event-box {{
            position: absolute;
            height: 100%;
            border-radius: 2px;
            padding: 4px 6px;
            font-size: 11px;
            font-weight: 500;
            color: #ffffff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            box-sizing: border-box;
            cursor: pointer;
        }}
        .event-box:hover {{
            filter: brightness(1.2);
            border: 1px solid #ffffff;
        }}
        .tooltip {{
            position: fixed;
            background: #090d16;
            border: 1px solid #38bdf8;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⏱️ InferenceOS Execution Timeline</h1>
            <div style="font-size:13px; color:#94a3b8">Total Span: {total_span:.1f} ms &bull; Tokens: {self.telemetry.generation_tokens}</div>
        </div>
    </div>

    <div class="timeline-card" id="timeline-container">
        <!-- Rendered by JS -->
    </div>

    <div id="tooltip" class="tooltip"></div>

    <script>
        const events = {events_json};
        const totalSpan = {total_span};
        const container = document.getElementById('timeline-container');
        const tooltip = document.getElementById('tooltip');

        const colors = {json.dumps(self.TRACK_COLORS)};

        // Group by track
        const tracks = {{}};
        events.forEach(e => {{
            if (!tracks[e.track]) tracks[e.track] = [];
            tracks[e.track].push(e);
        }});

        Object.keys(tracks).forEach(trackName => {{
            const row = document.createElement('div');
            row.className = 'track-row';

            const label = document.createElement('div');
            label.className = 'track-label';
            label.innerText = trackName;
            row.appendChild(label);

            const barContainer = document.createElement('div');
            barContainer.className = 'track-bar-container';

            tracks[trackName].forEach(ev => {{
                const leftPct = (ev.start_ms / totalSpan) * 100;
                const widthPct = Math.max(0.2, (ev.duration_ms / totalSpan) * 100);

                const box = document.createElement('div');
                box.className = 'event-box';
                box.style.left = leftPct + '%';
                box.style.width = widthPct + '%';
                box.style.backgroundColor = colors[ev.category] || '#64748b';
                box.innerText = ev.name;

                box.addEventListener('mouseenter', (e) => {{
                    tooltip.style.display = 'block';
                    tooltip.innerHTML = `<b>${{ev.name}}</b><br/>Start: ${{ev.start_ms.toFixed(1)}} ms<br/>Duration: ${{ev.duration_ms.toFixed(1)}} ms`;
                }});

                box.addEventListener('mousemove', (e) => {{
                    tooltip.style.left = (e.clientX + 12) + 'px';
                    tooltip.style.top = (e.clientY + 12) + 'px';
                }});

                box.addEventListener('mouseleave', () => {{
                    tooltip.style.display = 'none';
                }});

                barContainer.appendChild(box);
            }});

            row.appendChild(barContainer);
            container.appendChild(row);
        }});
    </script>
</body>
</html>
"""
