"""
system1_engine.py
-----------------
Native System 1 (Non-Autoregressive) Decision Engine for InferenceOS.

Executes single-forward-pass decision models (Laya, Kev, Jev API contract)
achieving 15–40 ms latency with calibrated probabilities over typed questions
(choice, noul, score) without autoregressive token generation or KV cache overhead.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from .inference_session import InferenceResult
from .runtime_config import RuntimeConfig
from .stats_collector import RuntimeStats
from orchestrator.model_parser import detect_model_format, read_model_metadata, ModelFormat

_KNOWN_FALLBACK_PATHS = [
    Path(r"D:\Projects\llm\.venv\Lib\site-packages"),
    Path(r"D:\Projects\llm\kev"),
    Path(r"D:\Projects\llm\laya-browser\code\apps"),
]


class System1RuntimeEngine:
    """
    High-performance engine executing non-autoregressive decision models.
    Supports coarse-to-fine chunking for wide option lists (>60 options),
    calibrated confidence scoring, and TileLang kernel acceleration.
    """

    def __init__(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        config: Optional[RuntimeConfig] = None,
    ) -> None:
        self.hw_profile = hw_profile or {}
        self.config = config or RuntimeConfig()
        self._agents: Dict[str, Any] = {}
        self._agent_metadata: Dict[str, Dict[str, Any]] = {}

    def _resolve_device(self, preferred_device: Optional[Union[str, Any]] = None) -> Union[str, Any]:
        """Resolve optimal device from config, parameter, and system resources."""
        target = preferred_device or self.config.force_backend
        if target:
            b = str(target).lower()
            if "cuda" in b and _HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            if "directml" in b or "dml" in b or "gpu" in b:
                try:
                    import torch_directml
                    return torch_directml.device()
                except Exception:
                    pass
                if _HAS_TORCH and torch.cuda.is_available():
                    return "cuda"
            if "mps" in b and _HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if "cpu" in b:
                return "cpu"

        # Auto resolution
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        if _HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_agent(self, model_path: Union[str, Path], device: Optional[Union[str, Any]] = None) -> Any:
        """
        Load or retrieve cached System 1 agent (Laya or Kev).
        """
        str_path = str(model_path)
        resolved_device = self._resolve_device(preferred_device=device)
        dev_str = str(resolved_device)
        cache_key = f"{str_path}@{dev_str}"

        if cache_key in self._agents:
            return self._agents[cache_key]
        if str_path in self._agents:
            return self._agents[str_path]

        try:
            resolved_path = str(Path(model_path).resolve())
            res_cache_key = f"{resolved_path}@{dev_str}"
            if res_cache_key in self._agents:
                return self._agents[res_cache_key]
            if resolved_path in self._agents:
                return self._agents[resolved_path]
        except Exception:
            resolved_path = str_path
            res_cache_key = cache_key

        p = Path(resolved_path)
        meta = read_model_metadata(p)
        self._agent_metadata[resolved_path] = meta
        kind = meta.get("system1_kind", "laya")

        if kind == "laya":
            try:
                import laya
            except ImportError:
                # Try fallback import via known paths
                for sp in _KNOWN_FALLBACK_PATHS:
                    if sp.exists() and str(sp) not in sys.path:
                        sys.path.insert(0, str(sp))
                import laya

            enable_fast = (os.environ.get("LAYA_FAST", "1") == "1") and (str(resolved_device) == "cuda")
            try:
                agent = laya.load(resolved_path, device=resolved_device, fast=enable_fast)
            except Exception:
                # Fall back to stock cpu load if device init hit errors
                agent = laya.load(resolved_path, device="cpu", fast=False)

            # Apply trained head length
            if getattr(agent, "cfg", None) and agent.cfg.get("head_max_len_train"):
                agent.cfg["head_max_len"] = agent.cfg["head_max_len_train"]

            self._agents[cache_key] = agent
            self._agents[res_cache_key] = agent
            return agent

        elif kind == "kev":
            # Kev loader (Qwen LoRA pointer head)
            try:
                for sp in _KNOWN_FALLBACK_PATHS:
                    if sp.exists() and str(sp) not in sys.path:
                        sys.path.insert(0, str(sp))
                import kev
                from kev.predictors import LocalPredictor
                agent = LocalPredictor(resolved_path, device=device)
                self._agents[resolved_path] = agent
                return agent
            except Exception as e:
                raise RuntimeError(f"Kev model requested but 'kev' package could not be initialized: {e}")

        else:
            raise ValueError(f"Unsupported System 1 model kind: {kind}")

    def _call_predict(self, agent: Any, state: Any, qs: Dict[str, Any]) -> Dict[str, Any]:
        """Unified predictor supporting Laya (.predict), Kev (__call__), and mock agents."""
        if hasattr(agent, "predict"):
            return agent.predict(state, qs)
        # Callable Kev LocalPredictor
        record = {"state": state, "questions": qs}
        raw = agent(record)
        if "answers" in raw:
            return raw
        answers = {}
        raw_probs = raw.get("probabilities", {})
        for qid, q in qs.items():
            q_type = q.get("type", "choice")
            qp = raw_probs.get(qid, {})
            if q_type == "choice":
                winner = max(qp, key=qp.get) if qp else ""
                conf = qp.get(winner, 0.0) if winner else 0.0
                answers[qid] = {
                    "type": "choice",
                    "choice": winner,
                    "probabilities": qp,
                    "confidence": round(conf, 4),
                }
            elif q_type == "score":
                score_val = sum(i * p for i, p in enumerate(qp.values())) if qp else 0.0
                answers[qid] = {
                    "type": "score",
                    "score": round(score_val, 2),
                    "probabilities": qp,
                    "confidence": max(qp.values()) if qp else 0.0,
                }
            elif q_type == "noul":
                p_true = qp.get("true", qp.get(True, 0.0))
                answers[qid] = {
                    "type": "noul",
                    "noul": round(p_true, 4),
                    "confidence": max(p_true, 1 - p_true),
                }
        return {
            "answers": answers,
            "usage": {"input_tokens": raw.get("input_tokens", 0)},
            "latency_ms": raw.get("latency_ms", 0.0),
        }

    def compact(self, v: Any, fmt: str = "v3") -> Any:
        """
        Compact verbose element criteria dicts into short strings
        to conserve option token budget.
        """
        if isinstance(v, dict) and "element" in v:
            max_char = 50 if fmt == "v3" else 1000
            s = str(v["element"])[:max_char]
            if v.get("role"):
                s += f" ({v['role']})"
            if v.get("current_value"):
                s += f" = {str(v['current_value'])[:30]!r}"
            for k in ("checked", "selected", "expanded"):
                if k in v:
                    s += f" {k}={v[k]}"
            return s
        return v

    def predict(
        self,
        model_path: Union[str, Path],
        state: Union[str, Dict[str, Any]],
        questions: Dict[str, Any],
        max_options: int = 60,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute single forward-pass decision inference.
        Includes coarse-to-fine chunking for questions with > max_options candidates.
        """
        str_path = str(model_path)
        if str_path in self._agents:
            resolved_path = str_path
        else:
            try:
                resolved_path = str(Path(model_path).resolve())
            except Exception:
                resolved_path = str_path

        agent = self.get_agent(resolved_path, device=kwargs.get("device"))
        meta = self._agent_metadata.get(resolved_path) or self._agent_metadata.get(str_path, {})
        cfg = getattr(agent, "cfg", {}) if hasattr(agent, "cfg") else {}
        fmt = meta.get("laya_fmt") or (cfg.get("laya_fmt", "v3") if isinstance(cfg, dict) else "v3")

        t0 = time.perf_counter()

        # Preprocess text state
        if isinstance(state, dict) and isinstance(state.get("page"), dict) and isinstance(state["page"].get("text"), str):
            cutoff = 1200 if fmt == "v3" else 1500
            state = {
                **state,
                "page": {**state["page"], "text": state["page"]["text"][:cutoff]},
            }
        elif isinstance(state, str):
            state = {"text": state[:2000]}

        # Prepare questions and detect wide choices (> max_options)
        qs: Dict[str, Any] = {}
        plan: Dict[str, Tuple[Dict[str, Any], List[List[str]]]] = {}

        for qid, q in questions.items():
            q_copy = dict(q)
            if "instructions" not in q_copy or not q_copy["instructions"]:
                goal_text = prompt_text if "prompt_text" in locals() else (state if isinstance(state, str) else "Evaluate candidate")
                q_copy["instructions"] = {"goal": str(goal_text)[:500]}

            q_type = q_copy.get("type", "choice")
            if q_type == "choice":
                if isinstance(q_copy.get("criteria"), dict):
                    q_copy["criteria"] = {k: self.compact(v, fmt=fmt) for k, v in q_copy["criteria"].items()}
                elif isinstance(q_copy.get("criteria"), list):
                    q_copy["criteria"] = {str(i): self.compact(v, fmt=fmt) for i, v in enumerate(q_copy["criteria"])}
            elif q_type == "score":
                if isinstance(q_copy.get("criteria"), list):
                    q_copy["criteria"] = [str(self.compact(v, fmt=fmt)) for v in q_copy["criteria"]]
                elif isinstance(q_copy.get("criteria"), dict):
                    q_copy["criteria"] = [str(v) for v in q_copy["criteria"].values()]

            keys = list(q_copy["criteria"]) if q_type == "choice" and isinstance(q_copy.get("criteria"), dict) else []
            if len(keys) <= max_options:
                qs[qid] = q_copy
                continue

            # Split wide choices into coarse chunks
            n_chunks = -(-len(keys) // max_options)
            chunks = [keys[i::n_chunks] for i in range(n_chunks)]
            plan[qid] = (q_copy, chunks)
            for ci, ch in enumerate(chunks):
                qs[f"{qid}__chunk{ci}"] = {**q_copy, "criteria": {k: q_copy["criteria"][k] for k in ch}}

        # Pass 1: standard forward pass
        r = self._call_predict(agent, state, qs)
        passes = 1

        # Pass 2: championship round if any question was chunked
        if plan:
            chunk_ans = {
                qid: [r["answers"].pop(f"{qid}__chunk{ci}") for ci in range(len(chunks))]
                for qid, (q, chunks) in plan.items()
            }
            finals = {
                qid: {**q, "criteria": {a["choice"]: q["criteria"][a["choice"]] for a in chunk_ans[qid]}}
                for qid, (q, _) in plan.items()
            }
            r2 = self._call_predict(agent, state, finals)
            passes = 2
            if "usage" in r and "usage" in r2:
                r["usage"]["input_tokens"] += r2["usage"].get("input_tokens", 0)

            for qid, (q, chunks) in plan.items():
                fa = r2["answers"][qid]
                probs: Dict[str, float] = {}
                for ca, ch in zip(chunk_ans[qid], chunks):
                    pf = fa["probabilities"].get(ca["choice"], 0.0)
                    for k in ch:
                        probs[k] = pf * ca["probabilities"].get(k, 0.0)

                tot = sum(probs.values()) or 1.0
                probs = {k: round(v / tot, 6) for k, v in probs.items()}
                chosen = max(probs, key=probs.get) if probs else fa.get("choice", "")
                r["answers"][qid] = {
                    "type": "choice",
                    "choice": chosen,
                    "probabilities": probs,
                    "confidence": fa.get("confidence", 0.9),
                    "action": fa.get("action", {}),
                    "coarse_to_fine": {
                        "chunks": len(chunks),
                        "winners": [a["choice"] for a in chunk_ans[qid]],
                    },
                }

        latency_ms = (time.perf_counter() - t0) * 1000.0
        model_name = Path(model_path).stem

        return {
            "model": model_name,
            "answers": r.get("answers", {}),
            "usage": r.get("usage", {"input_tokens": 0}),
            "passes": passes,
            "latency_ms": round(latency_ms, 2),
            "backend": f"system1-{meta.get('system1_kind', 'laya')} ({agent.device})",
            "success": True,
        }


class System1Runner:
    """
    Runner wrapper adapting System1RuntimeEngine into the universal
    InferenceOS MultiFormatRuntimeEngine execution contract.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        hw_profile: Optional[Dict[str, Any]] = None,
        config: Optional[RuntimeConfig] = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.hw_profile = hw_profile or {}
        self.config = config or RuntimeConfig()
        self.engine = System1RuntimeEngine(hw_profile=self.hw_profile, config=self.config)

    def run(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        max_tokens: int = 128,
        temp: float = 0.0,
        **kwargs,
    ) -> InferenceResult:
        """
        Execute prompt against System 1 model.
        Accepts either a JSON-encoded request payload or a freeform prompt.
        """
        start_t = time.perf_counter()

        # Parse prompt: test if JSON request containing {"state": ..., "questions": ...}
        state: Union[str, Dict[str, Any]] = prompt
        questions: Dict[str, Any] = {}

        is_json = False
        try:
            trimmed = prompt.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                data = json.loads(trimmed)
                if isinstance(data, dict) and "questions" in data:
                    is_json = True
                    state = data.get("state", "")
                    questions = data["questions"]
        except Exception:
            pass

        if not is_json or not questions:
            # Construct default decision question
            questions = {
                "decision": {
                    "type": "choice",
                    "instructions": {"goal": prompt},
                    "criteria": {
                        "ACCEPT": "Accept / True / Yes",
                        "REJECT": "Reject / False / No",
                        "UNCERTAIN": "Needs further investigation",
                    },
                },
                "confidence_score": {
                    "type": "score",
                    "instructions": {"goal": prompt},
                    "criteria": ["Very Low", "Low", "Medium", "High", "Very High"],
                },
            }

        try:
            res = self.engine.predict(
                model_path=self.model_path,
                state=state,
                questions=questions,
            )

            # Format human-readable output
            lines = [f"[System 1 Decision Engine] (Latency: {res['latency_ms']:.1f}ms, Passes: {res['passes']})"]
            for qid, ans in res["answers"].items():
                t = ans.get("type", "choice")
                conf = ans.get("confidence", 0.0)
                if t == "choice":
                    chosen = ans.get("choice", "")
                    lines.append(f"  * {qid}: {chosen} (Confidence: {conf:.2f})")
                    probs = ans.get("probabilities", {})
                    sorted_p = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
                    for k, v in sorted_p:
                        bar_len = int(round(v * 16))
                        bar = "#" * bar_len + "-" * (16 - bar_len)
                        lines.append(f"      [{bar}] {v:5.2f}  {k}")
                elif t == "score":
                    lines.append(f"  * {qid}: score = {ans.get('score', 0):.2f} (Confidence: {conf:.2f})")
                elif t == "noul":
                    lines.append(f"  * {qid}: P(true) = {ans.get('noul', 0.0):.2f}")

            formatted_output = "\n".join(lines)

            # Token streaming simulation if on_token provided
            if on_token:
                for chunk in formatted_output.split("\n"):
                    on_token(chunk + "\n")

            total_wall_ms = (time.perf_counter() - start_t) * 1000.0
            input_tokens = res["usage"].get("input_tokens", 64)

            stats = RuntimeStats(
                prompt_eval_tps=input_tokens / (max(0.001, res["latency_ms"]) / 1000.0),
                eval_tps=1000.0 / max(0.001, res["latency_ms"]),  # decisions per second
                tokens_generated=len(res["answers"]),
                prompt_eval_ms=res["latency_ms"],
                eval_ms=res["latency_ms"],
                total_wall_ms=total_wall_ms,
            )

            return InferenceResult(
                generated_text=formatted_output,
                stats=stats,
                raw_stderr="",
                exit_code=0,
                success=True,
                backend=res["backend"],
                system1_answers=res["answers"],
            )

        except Exception as e:
            total_wall_ms = (time.perf_counter() - start_t) * 1000.0
            return InferenceResult(
                generated_text=f"System 1 execution error: {e}",
                stats=RuntimeStats(total_wall_ms=total_wall_ms),
                raw_stderr=str(e),
                exit_code=1,
                success=False,
                backend="system1-error",
            )
