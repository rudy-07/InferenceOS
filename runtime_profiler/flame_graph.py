"""
flame_graph.py
--------------
Flame graph generator for Phase 9 Runtime Profiler.

Generates:
  1. Interactive HTML/SVG Flame Graphs (self-contained, color-coded by device/stage).
  2. Structured JSON Flame Graph (for speedscope, Chrome trace, or programmatic inspection).
  3. Folded stack text representation (compatible with Brendan Gregg's flamegraph.pl).

Hierarchy
---------
InferenceSession
  ├── Stage: Prepare (CPU tokenize & prompt)
  ├── Stage: Transfer (PCIe segment prefetch)
  ├── Stage: Compute (Transformer layer execution)
  │    ├── Segment 0 (GPU Layers 0-48) -> GEMM, Attention
  │    └── Segment 1 (CPU Layers 49-72) -> GEMM, KV Cache Update
  └── Stage: Sampling (Logits & Top-P)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .telemetry_collector import ProfilerTelemetry


# ---------------------------------------------------------------------------
# FlameNode
# ---------------------------------------------------------------------------

@dataclass
class FlameNode:
    """
    Node in the flame graph tree.

    Attributes
    ----------
    name : str
        Node display label (e.g., "GPU Compute", "Layer 0-48").
    value : float
        Total duration / cost in milliseconds.
    category : str
        Category for color scheme: "gpu", "cpu", "pcie", "kv", "stage", "root".
    children : List[FlameNode]
        Child call stack nodes.
    """
    name: str
    value: float
    category: str = "stage"
    children: List[FlameNode] = field(default_factory=list)

    def add_child(self, name: str, value: float, category: str = "stage") -> FlameNode:
        child = FlameNode(name=name, value=value, category=category)
        self.children.append(child)
        return child

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 3),
            "category": self.category,
            "children": [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# FlameGraphGenerator
# ---------------------------------------------------------------------------

class FlameGraphGenerator:
    """
    Generates interactive flame graphs from ProfilerTelemetry data.
    """

    # Category color palette (Hex codes)
    COLORS = {
        "root": "#1f2937",     # Dark Slate
        "stage": "#3b82f6",    # Blue
        "gpu": "#10b981",      # Emerald Green
        "cpu": "#6366f1",      # Indigo
        "pcie": "#f59e0b",     # Amber / Orange
        "kv": "#8b5cf6",       # Purple
        "sample": "#ec4899",   # Pink
    }

    def __init__(self, telemetry: ProfilerTelemetry) -> None:
        self.telemetry = telemetry
        self.root = self._build_stack_tree()

    def _build_stack_tree(self) -> FlameNode:
        """Construct the hierarchical FlameNode tree from telemetry."""
        t = self.telemetry
        tot_ms = max(0.1, t.total_wall_ms if t.total_wall_ms > 0 else (t.prompt_eval_ms + t.generation_eval_ms))

        root = FlameNode(
            name=f"InferenceSession ({t.model_name or 'model'})",
            value=tot_ms,
            category="root",
        )

        # Stage 1: Prepare (Prompt Eval)
        prep_ms = max(0.01, t.prompt_eval_ms if t.prompt_eval_ms > 0 else tot_ms * 0.1)
        prep_node = root.add_child("Stage: Prepare (Prompt Prefill)", prep_ms, "cpu")
        prep_node.add_child("Tokenize & Prefix Scan", prep_ms * 0.3, "cpu")
        prep_node.add_child("Prompt Layer Execution", prep_ms * 0.7, "cpu" if t.n_gpu_layers == 0 else "gpu")

        # Stage 2: Transfer (PCIe / Prefetch)
        xfr_ms = max(0.01, (t.pcie_bytes_transferred / (1024**2)) * 0.1 if t.pcie_bytes_transferred > 0 else tot_ms * 0.05)
        xfr_node = root.add_child("Stage: Transfer (PCIe DMA Prefetch)", xfr_ms, "pcie")
        xfr_node.add_child("Boundary Buffer Copy", xfr_ms * 0.8, "pcie")
        xfr_node.add_child("CUDA Stream Sync", xfr_ms * 0.2, "pcie")

        # Stage 3: Compute (Generation Eval)
        comp_ms = max(0.01, t.generation_eval_ms if t.generation_eval_ms > 0 else tot_ms * 0.8)
        comp_node = root.add_child("Stage: Compute (Autoregressive Generation)", comp_ms, "stage")

        # Layer segments inside compute
        if t.layer_timings:
            # Group by device
            gpu_timings = [lt for lt in t.layer_timings if "GPU" in lt.device and "IGPU" not in lt.device]
            cpu_timings = [lt for lt in t.layer_timings if "CPU" in lt.device]
            igpu_timings = [lt for lt in t.layer_timings if "IGPU" in lt.device]

            if gpu_timings:
                gpu_tot = sum(lt.compute_ms for lt in gpu_timings)
                n_g = len(gpu_timings)
                g_node = comp_node.add_child(f"dGPU Block ({n_g} layers)", gpu_tot, "gpu")
                g_node.add_child("Tensor Core GEMM (Q4/Q8)", gpu_tot * 0.65, "gpu")
                g_node.add_child("FlashAttention-2 Kernel", gpu_tot * 0.35, "gpu")

            if igpu_timings:
                igpu_tot = sum(lt.compute_ms for lt in igpu_timings)
                n_ig = len(igpu_timings)
                ig_node = comp_node.add_child(f"iGPU Block ({n_ig} layers)", igpu_tot, "gpu")
                ig_node.add_child("Vulkan Compute Shader", igpu_tot * 0.8, "gpu")
                ig_node.add_child("Shared Memory Sync", igpu_tot * 0.2, "gpu")

            if cpu_timings:
                cpu_tot = sum(lt.compute_ms for lt in cpu_timings)
                n_c = len(cpu_timings)
                c_node = comp_node.add_child(f"CPU Block ({n_c} layers)", cpu_tot, "cpu")
                c_node.add_child("AVX-512 / AMX Matrix Engine", cpu_tot * 0.70, "cpu")
                c_node.add_child("KV Cache Allocation", cpu_tot * 0.30, "kv")

        else:
            # Fallback if no per-layer timings available
            gpu_share = (t.n_gpu_layers / max(1, t.total_layers)) if t.total_layers > 0 else 0.5
            comp_node.add_child(f"dGPU Layers ({t.n_gpu_layers})", comp_ms * gpu_share, "gpu")
            comp_node.add_child(f"CPU Layers ({t.n_cpu_layers})", comp_ms * (1.0 - gpu_share), "cpu")

        # Stage 4: Sampling
        sample_ms = tot_ms * 0.05
        sample_node = root.add_child("Stage: Sampling & Post-Process", sample_ms, "sample")
        sample_node.add_child("Top-P / Temperature Filter", sample_ms * 0.6, "sample")
        sample_node.add_child("Tokenizer Decode", sample_ms * 0.4, "sample")

        return root

    def to_json(self, indent: int = 2) -> str:
        """Export flame graph tree as formatted JSON string."""
        return json.dumps(self.root.to_dict(), indent=indent)

    def to_folded_text(self) -> str:
        """
        Export call stacks in folded text format:
        ``InferenceSession;Stage:Compute;dGPU Block 1000``
        """
        lines: List[str] = []

        def _traverse(node: FlameNode, stack: List[str]):
            current_stack = stack + [node.name]
            if not node.children:
                path_str = ";".join(current_stack)
                lines.append(f"{path_str} {int(node.value * 1000)}")  # microsecond counts
            else:
                for child in node.children:
                    _traverse(child, current_stack)

        _traverse(self.root, [])
        return "\n".join(lines)

    def to_html(self) -> str:
        """
        Generate a fully self-contained, interactive HTML/SVG flame graph document.
        """
        tree_json = self.to_json(indent=2)
        colors_json = json.dumps(self.COLORS, indent=2)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InferenceOS Phase 9 — Flame Graph ({self.telemetry.model_name})</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{
            font-size: 24px;
            margin: 0;
            color: #38bdf8;
        }}
        .meta-pills {{
            display: flex;
            gap: 12px;
        }}
        .pill {{
            background: #1e293b;
            border: 1px solid #475569;
            border-radius: 16px;
            padding: 4px 12px;
            font-size: 13px;
            color: #cbd5e1;
        }}
        #flamegraph-container {{
            background: #1e293b;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
            overflow-x: auto;
        }}
        .flame-row {{
            display: flex;
            height: 28px;
            margin-bottom: 2px;
            gap: 2px;
        }}
        .flame-box {{
            height: 100%;
            border-radius: 3px;
            box-sizing: border-box;
            padding: 4px 8px;
            font-size: 12px;
            font-weight: 500;
            color: #ffffff;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            cursor: pointer;
            transition: opacity 0.15s ease, transform 0.15s ease;
        }}
        .flame-box:hover {{
            opacity: 0.85;
            transform: scaleY(1.05);
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
            box-shadow: 0 4px 12px rgba(0,0,0,0.8);
        }}
        .legend {{
            display: flex;
            gap: 16px;
            margin-top: 16px;
            font-size: 13px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🔥 InferenceOS Flame Graph</h1>
            <div style="font-size:13px; color:#94a3b8; margin-top:4px;">
                Model: <b>{self.telemetry.model_name or 'N/A'}</b> &bull; Backend: <b>{self.telemetry.backend.upper()}</b>
            </div>
        </div>
        <div class="meta-pills">
            <div class="pill">Wall Time: <b>{self.telemetry.total_wall_ms:.1f} ms</b></div>
            <div class="pill">Gen TPS: <b>{self.telemetry.generation_tps:.1f} tok/s</b></div>
            <div class="pill">GPU Idle: <b>{self.telemetry.gpu_idle_pct:.0f}%</b></div>
        </div>
    </div>

    <div id="flamegraph-container">
        <!-- Rendered by JS below -->
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:#10b981"></div> GPU Compute</div>
        <div class="legend-item"><div class="legend-color" style="background:#6366f1"></div> CPU Compute</div>
        <div class="legend-item"><div class="legend-color" style="background:#f59e0b"></div> PCIe Transfer</div>
        <div class="legend-item"><div class="legend-color" style="background:#8b5cf6"></div> KV Cache</div>
        <div class="legend-item"><div class="legend-color" style="background:#ec4899"></div> Sampling</div>
    </div>

    <div id="tooltip" class="tooltip"></div>

    <script>
        const treeData = {tree_json};
        const colors = {colors_json};
        const container = document.getElementById('flamegraph-container');
        const tooltip = document.getElementById('tooltip');

        // Flatten tree into levels (rows)
        const levels = [];

        function traverse(node, depth, leftPct, widthPct) {{
            if (!levels[depth]) levels[depth] = [];
            levels[depth].push({{
                name: node.name,
                value: node.value,
                category: node.category,
                leftPct: leftPct,
                widthPct: widthPct
            }});

            if (node.children && node.children.length > 0) {{
                let currentLeft = leftPct;
                const totalChildVal = node.children.reduce((acc, c) => acc + c.value, 0);
                node.children.forEach(child => {{
                    const childWidth = totalChildVal > 0 ? (child.value / totalChildVal) * widthPct : 0;
                    traverse(child, depth + 1, currentLeft, childWidth);
                    currentLeft += childWidth;
                }});
            }}
        }}

        traverse(treeData, 0, 0, 100);

        // Render rows top-down
        levels.forEach((row, depth) => {{
            const rowDiv = document.createElement('div');
            rowDiv.className = 'flame-row';

            row.forEach(box => {{
                if (box.widthPct <= 0.05) return; // skip tiny boxes
                const boxDiv = document.createElement('div');
                boxDiv.className = 'flame-box';
                boxDiv.style.width = box.widthPct + '%';
                boxDiv.style.backgroundColor = colors[box.category] || '#475569';
                boxDiv.innerText = `${{box.name}} (${{box.value.toFixed(1)}} ms)`;

                boxDiv.addEventListener('mouseenter', (e) => {{
                    tooltip.style.display = 'block';
                    tooltip.innerHTML = `<b>${{box.name}}</b><br/>Duration: ${{box.value.toFixed(2)}} ms (${{box.widthPct.toFixed(1)}}% of parent)`;
                }});

                boxDiv.addEventListener('mousemove', (e) => {{
                    tooltip.style.left = (e.clientX + 12) + 'px';
                    tooltip.style.top = (e.clientY + 12) + 'px';
                }});

                boxDiv.addEventListener('mouseleave', () => {{
                    tooltip.style.display = 'none';
                }});

                rowDiv.appendChild(boxDiv);
            }});

            container.appendChild(rowDiv);
        }});
    </script>
</body>
</html>
"""
