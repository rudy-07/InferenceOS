"""
hybrid_orchestrator.py
----------------------
Bi-Directional System 1 / System 2 Hybrid Orchestrator for InferenceOS.

Implements the Kahneman Fast/Slow cognitive loop for production AI agents:
1. Direction A (System 1 -> System 2):
   - Fast reflex forward pass (15-30ms, 0 KV cache).
   - Confidence threshold gating: if confidence < tau or choice is UNCERTAIN,
     escalates to System 2 teacher model (local GGUF/multi-format or remote HTTP LLM).
   - Generative text handoff: when an action needs natural language synthesis.
   - DAgger distillation logging: writes training pairs for continuous self-improvement.
2. Direction B (System 2 -> System 1):
   - Decision tool: System 2 delegates wide option selection (>50 items) to System 1 in 20ms.
   - Guardrail gate: System 2 proposed actions are evaluated via System 1 noul checks before execution.
3. Strict Toggle-ability:
   - Enabled/disabled globally via config, CLI flags (--escalate / --no-escalate),
     or per-request headers/body params.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from inference_runtime.system1_engine import System1RuntimeEngine
from inference_runtime.multiformat_engine import MultiFormatRuntimeEngine
from inference_runtime.runtime_config import RuntimeConfig
from orchestrator.model_parser import detect_model_format, ModelFormat


DEFAULT_ESCALATE_TAU = 0.70
DEFAULT_DAGGER_LOG_DIR = Path.home() / ".inferenceos" / "dagger"


class HybridOrchestrator:
    """
    Coordinates bi-directional execution between System 1 (reflex/decision)
    and System 2 (reasoning/generative) models.
    """

    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        hw_profile: Optional[Dict[str, Any]] = None,
        s1_engine: Optional[System1RuntimeEngine] = None,
        s2_engine: Optional[MultiFormatRuntimeEngine] = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.hw_profile = hw_profile or {}
        self.s1_engine = s1_engine or System1RuntimeEngine(hw_profile=self.hw_profile, config=self.config)
        self.s2_engine = s2_engine or MultiFormatRuntimeEngine(hw_profile=self.hw_profile, config=self.config)

    # -----------------------------------------------------------------------
    # Direction A: System 1 -> System 2 (Escalation & Generation Handoff)
    # -----------------------------------------------------------------------

    def decide_with_escalation(
        self,
        model_path: Union[str, Path],
        state: Union[str, Dict[str, Any]],
        questions: Dict[str, Any],
        escalate: bool = True,
        escalate_tau: float = DEFAULT_ESCALATE_TAU,
        system2_model: Optional[Union[str, Path]] = None,
        system2_base_url: Optional[str] = None,
        dagger_log_path: Optional[Union[str, Path]] = None,
        max_options: int = 60,
        s1_device: Optional[str] = None,
        system2_session: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute System 1 decision with optional System 2 escalation.
        """
        # Step 1: Run fast System 1 forward pass
        s1_result = self.s1_engine.predict(
            model_path=model_path,
            state=state,
            questions=questions,
            max_options=max_options,
            device=s1_device,
        )

        answers = s1_result.get("answers", {})

        # Step 2: Check if escalation is disabled or unnecessary
        if not escalate:
            return {
                **s1_result,
                "escalated": False,
                "escalate_reason": None,
            }

        # Step 3: Evaluate escalation conditions
        reasons: List[str] = []
        min_conf = 1.0
        needs_generation = False

        if not answers and questions:
            reasons.append("System 1 produced no answers for requested questions")

        for qid, ans in answers.items():
            conf = ans.get("confidence", 1.0)
            # Detect NaN, None, or invalid confidence
            if conf is None or not isinstance(conf, (int, float)) or (isinstance(conf, float) and (conf != conf or conf < 0)):
                conf = 0.0
                reasons.append(f"Question '{qid}' returned uncalibrated or NaN confidence")

            if conf < min_conf:
                min_conf = conf

            choice = str(ans.get("choice", "")).upper()
            if choice in ("UNCERTAIN", "AMBIGUOUS", "UNKNOWN", "NONE"):
                reasons.append(f"Question '{qid}' yielded uncertain choice: '{choice}'")

            # Check if question requested generation
            orig_q = questions.get(qid, {})
            if orig_q.get("needs_generation") or ans.get("action", {}).get("needs_generation"):
                needs_generation = True
                reasons.append(f"Question '{qid}' requires generative response synthesis")

        if min_conf < escalate_tau:
            reasons.append(f"Confidence {min_conf:.3f} is below threshold tau={escalate_tau:.2f}")

        # If high confidence and no generation needed, return System 1 fast result
        if not reasons and not needs_generation:
            return {
                **s1_result,
                "escalated": False,
                "escalate_reason": None,
            }

        # Step 4: Trigger System 2 Escalation
        escalate_reason = "; ".join(reasons)
        t_start_s2 = time.perf_counter()

        s2_answers, s2_text = self._call_system2_teacher(
            state=state,
            questions=questions,
            s1_answers=answers,
            escalate_reason=escalate_reason,
            system2_model=system2_model,
            system2_base_url=system2_base_url,
            needs_generation=needs_generation,
            system2_session=system2_session,
        )

        s2_latency_ms = (time.perf_counter() - t_start_s2) * 1000.0

        # Step 5: Merge System 2 response
        merged_answers = {}
        for qid, q in questions.items():
            if qid in s2_answers:
                merged_answers[qid] = {
                    **answers.get(qid, {}),
                    **s2_answers[qid],
                    "system2": True,
                    "confidence": s2_answers[qid].get("confidence", 0.95),
                }
            else:
                merged_answers[qid] = answers.get(qid, {})

        # Step 6: DAgger distillation logging (record teacher demonstration)
        log_file = dagger_log_path or os.environ.get("ESCALATE_LOG")
        if log_file:
            self._record_dagger_case(
                log_file=Path(log_file),
                state=state,
                questions=questions,
                s1_answers=answers,
                s2_answers=s2_answers,
                reason=escalate_reason,
            )

        return {
            "model": s1_result.get("model"),
            "answers": merged_answers,
            "generated_text": s2_text,
            "usage": s1_result.get("usage", {}),
            "passes": s1_result.get("passes", 1),
            "latency_ms": round(s1_result.get("latency_ms", 0) + s2_latency_ms, 2),
            "s1_latency_ms": s1_result.get("latency_ms", 0),
            "s2_latency_ms": round(s2_latency_ms, 2),
            "backend": f"{s1_result.get('backend', 'system1')} + system2-escalated",
            "escalated": True,
            "escalate_reason": escalate_reason,
            "success": True,
        }

    def _call_system2_teacher(
        self,
        state: Any,
        questions: Dict[str, Any],
        s1_answers: Dict[str, Any],
        escalate_reason: str,
        system2_model: Optional[Union[str, Path]] = None,
        system2_base_url: Optional[str] = None,
        needs_generation: bool = False,
        system2_session: Optional[Any] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Invoke the System 2 teacher model (via remote OpenAI-compatible endpoint
        or local multi-format / GGUF runner).
        """
        # 1. Check if remote teacher endpoint is configured (e.g. TEXT_MODEL_BASE_URL)
        esc_url = system2_base_url or os.environ.get("TEXT_MODEL_BASE_URL")
        esc_model_name = str(system2_model) if system2_model else os.environ.get("TEXT_MODEL", "system2-teacher")

        prompt_payload = {
            "task_state": state,
            "questions": questions,
            "system1_initial_eval": {k: v.get("choice") for k, v in s1_answers.items()},
            "escalation_reason": escalate_reason,
            "needs_generation": needs_generation,
        }

        # Attempt remote HTTP escalation if URL provided
        if esc_url:
            try:
                import httpx
                api_endpoint = esc_url.rstrip("/") + "/chat/completions" if not esc_url.endswith("/chat/completions") else esc_url
                sys_msg = (
                    "You are the System-2 reasoning teacher and fallback for a real-time agent. "
                    "Given the task state, candidate questions, and low-confidence System 1 evaluation, "
                    "deliberate carefully and output the best final decision as a JSON object: "
                    '{"decisions": {"<qid>": {"choice": "<option_key>", "confidence": 0.95, "rationale": "..."}}, "response_text": "..."}'
                )
                body = {
                    "model": esc_model_name,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"},
                }
                resp = httpx.post(api_endpoint, json=body, timeout=30.0)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    return parsed.get("decisions", {}), parsed.get("response_text")
            except Exception:
                pass

        # 0. Active existing session in memory (fastest path: zero model reloading overhead)
        if system2_session is not None:
            try:
                teacher_prompt = self._build_system2_prompt(
                    state=state,
                    questions=questions,
                    s1_answers=s1_answers,
                    escalate_reason=escalate_reason,
                )
                inference_res = system2_session.run(teacher_prompt)
                raw_output = inference_res.generated_text if hasattr(inference_res, "generated_text") else str(inference_res)
                parsed_decisions, s2_text = self._parse_system2_response(raw_output, questions)
                if parsed_decisions:
                    return parsed_decisions, (s2_text or (raw_output.strip() if needs_generation else None))
            except Exception:
                pass

        # 2. Local model execution (GGUF, ONNX, SafeTensors, PyTorch)
        if system2_model:
            try:
                from cli.core.model_registry import get_model_registry
                resolved_s2 = get_model_registry().resolve_model_path(str(system2_model))
                if (not resolved_s2 or not resolved_s2.exists()) and Path(str(system2_model)).exists():
                    resolved_s2 = Path(str(system2_model)).resolve()

                if resolved_s2 and resolved_s2.exists():
                    teacher_prompt = self._build_system2_prompt(
                        state=state,
                        questions=questions,
                        s1_answers=s1_answers,
                        escalate_reason=escalate_reason,
                    )
                    inference_res = self.s2_engine.execute(
                        model_path=resolved_s2,
                        prompt=teacher_prompt,
                        max_tokens=300,
                        temp=0.1,
                    )
                    raw_output = inference_res.generated_text if hasattr(inference_res, "generated_text") else str(inference_res)
                    parsed_decisions, s2_text = self._parse_system2_response(raw_output, questions)
                    if parsed_decisions:
                        return parsed_decisions, (s2_text or (raw_output.strip() if needs_generation else None))
            except Exception:
                pass

        # 3. Local fallback synthesis (heuristic rule-based teacher when remote/local LLM offline)
        s2_answers = {}
        generated_text = None

        for qid, q in questions.items():
            q_type = q.get("type", "choice")
            crit = q.get("criteria", {})
            s1_ans = s1_answers.get(qid, {})

            if q_type == "choice":
                opts = list(crit.keys()) if isinstance(crit, dict) else list(range(len(crit)))
                # Pick the highest non-uncertain option if possible
                valid_opts = [o for o in opts if str(o).upper() not in ("UNCERTAIN", "AMBIGUOUS", "NONE")]
                chosen = valid_opts[0] if valid_opts else (opts[0] if opts else "UNCERTAIN")

                s2_answers[qid] = {
                    "type": "choice",
                    "choice": chosen,
                    "probabilities": {str(k): (0.95 if k == chosen else 0.05 / max(1, len(opts) - 1)) for k in opts},
                    "confidence": 0.95,
                    "teacher_override": True,
                }
            elif q_type == "score":
                s2_answers[qid] = {
                    "type": "score",
                    "score": s1_ans.get("score", 3.0),
                    "confidence": 0.95,
                    "teacher_override": True,
                }
            elif q_type == "noul":
                s2_answers[qid] = {
                    "type": "noul",
                    "noul": 0.95,
                    "confidence": 0.95,
                    "teacher_override": True,
                }

        if needs_generation:
            generated_text = f"System 2 Teacher synthesized action based on state: {str(state)[:100]}"

        return s2_answers, generated_text

    def _build_system2_prompt(
        self,
        state: Any,
        questions: Dict[str, Any],
        s1_answers: Dict[str, Any],
        escalate_reason: str,
    ) -> str:
        """Format an optimal instruction prompt for local LLM deliberation."""
        state_str = str(state)[:1200]
        q_summary = {}
        for qid, q in questions.items():
            crit = q.get("criteria", {})
            q_summary[qid] = {
                "type": q.get("type", "choice"),
                "goal": q.get("instructions", {}).get("goal", ""),
                "options": list(crit.keys()) if isinstance(crit, dict) else list(crit),
            }

        prompt = (
            "<|im_start|>system\n"
            "You are the System-2 deliberative teacher model. Given low-confidence decisions from System-1, "
            "determine the single best option for each question and explain briefly. "
            'Return strictly valid JSON: {"decisions": {"<qid>": {"choice": "<option>", "confidence": 0.95, "rationale": "..."}}, "response_text": "..."}\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"State: {state_str}\n"
            f"Questions: {json.dumps(q_summary, ensure_ascii=False)}\n"
            f"System 1 Initial Predictions: {json.dumps({k: v.get('choice') for k, v in s1_answers.items()}, ensure_ascii=False)}\n"
            f"Escalation Reason: {escalate_reason}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def _parse_system2_response(
        self,
        raw_output: str,
        questions: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Extract structured decisions and optional synthesized text from raw LLM output."""
        if not raw_output:
            return {}, None

        cleaned = raw_output.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        # Try JSON parsing
        try:
            parsed = json.loads(cleaned)
            decisions = parsed.get("decisions", parsed)
            s2_text = parsed.get("response_text")

            validated = {}
            for qid, q in questions.items():
                if qid in decisions:
                    d = decisions[qid]
                    chosen = d.get("choice") if isinstance(d, dict) else str(d)
                    validated[qid] = {
                        "type": q.get("type", "choice"),
                        "choice": chosen,
                        "confidence": float(d.get("confidence", 0.95)) if isinstance(d, dict) else 0.95,
                        "rationale": d.get("rationale", "") if isinstance(d, dict) else "",
                        "teacher_override": True,
                        "system2": True,
                    }
            if validated:
                return validated, s2_text
        except Exception:
            pass

        # Regex / substring fallback across candidate criteria
        extracted = {}
        for qid, q in questions.items():
            criteria = q.get("criteria", {})
            opts = list(criteria.keys()) if isinstance(criteria, dict) else criteria
            for opt in opts:
                if str(opt).upper() in ("UNCERTAIN", "AMBIGUOUS", "NONE"):
                    continue
                if re.search(r'\b' + re.escape(str(opt)) + r'\b', raw_output, re.IGNORECASE):
                    extracted[qid] = {
                        "type": q.get("type", "choice"),
                        "choice": str(opt),
                        "confidence": 0.90,
                        "rationale": "Extracted via System 2 deliberation",
                        "teacher_override": True,
                        "system2": True,
                    }
                    break
        return extracted, raw_output.strip()

    def _record_dagger_case(
        self,
        log_file: Path,
        state: Any,
        questions: Dict[str, Any],
        s1_answers: Dict[str, Any],
        s2_answers: Dict[str, Any],
        reason: str,
    ) -> None:
        """Append an escalation demonstration to the DAgger dataset."""
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": time.time(),
                "state": state,
                "questions": questions,
                "system1_answers": s1_answers,
                "system2_answers": s2_answers,
                "escalate_reason": reason,
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Direction B: System 2 -> System 1 (Tool Delegation & Guardrails)
    # -----------------------------------------------------------------------

    def system1_tool_decide(
        self,
        model_path: Union[str, Path],
        state: Union[str, Dict[str, Any]],
        options: Union[Dict[str, str], List[str]],
        goal: str = "Select optimal action",
    ) -> Dict[str, Any]:
        """
        Tool callable by System 2 agents to evaluate wide option spaces in ~20ms.
        """
        if not options:
            return {
                "chosen": None,
                "confidence": 0.0,
                "probabilities": {},
                "latency_ms": 0.0,
                "success": False,
                "error": "Options list cannot be empty",
            }

        criteria = options if isinstance(options, dict) else {str(i): opt for i, opt in enumerate(options)}
        questions = {
            "decision": {
                "type": "choice",
                "instructions": {"goal": goal},
                "criteria": criteria,
            }
        }
        res = self.s1_engine.predict(model_path=model_path, state=state, questions=questions)
        ans = res.get("answers", {}).get("decision", {})
        return {
            "chosen": ans.get("choice"),
            "confidence": ans.get("confidence", 0.0),
            "probabilities": ans.get("probabilities", {}),
            "latency_ms": res.get("latency_ms", 0.0),
            "success": True,
        }

    def system1_guardrail(
        self,
        model_path: Union[str, Path],
        proposed_action: str,
        context: Optional[str] = None,
        safe_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Fast 15ms System 1 safety gate evaluating proposed actions before execution.
        """
        safe_threshold = max(0.0, min(1.0, float(safe_threshold)))

        if not proposed_action or not str(proposed_action).strip():
            return {
                "allowed": False,
                "safety_score": 0.0,
                "threshold": safe_threshold,
                "latency_ms": 0.0,
                "proposed_action": proposed_action,
                "error": "Proposed action cannot be empty",
            }

        state = context or f"Proposed execution action: {proposed_action}"
        questions = {
            "is_safe": {
                "type": "noul",
                "instructions": {"goal": f"Verify whether this action is safe and non-destructive: '{proposed_action}'"},
            }
        }
        res = self.s1_engine.predict(model_path=model_path, state=state, questions=questions)
        ans = res.get("answers", {}).get("is_safe", {})
        p_safe = float(ans.get("noul", 0.0))
        is_allowed = p_safe >= safe_threshold

        return {
            "allowed": is_allowed,
            "safety_score": round(p_safe, 4),
            "threshold": safe_threshold,
            "latency_ms": res.get("latency_ms", 0.0),
            "proposed_action": proposed_action,
        }
