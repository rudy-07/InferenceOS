"""
eagle_draft_engine.py
---------------------
Eagle-2 / external draft GGUF model engine for Phase 5 Speculative Decoding.

This module wraps a secondary (smaller) GGUF model that acts as the speculative
draft model.  It is designed to work with:

  - Eagle-2 draft models (lightweight distilled heads trained to predict the
    base model's next-token distribution).
  - Any small GGUF model used as a draft model (e.g. Qwen3-0.6B drafting for
    Qwen3-14B, Llama-3.2-1B drafting for Llama-3.1-70B).

Architecture in InferenceOS context
-------------------------------------
Since InferenceOS uses llama.cpp as an **external subprocess** rather than an
in-process library (due to its hardware-agnostic multi-vendor build system),
the Eagle draft engine runs the draft model as a **separate llama.cpp subprocess
call** with a short ``--n-predict`` count (``draft_tokens``).

This is architecturally different from in-process Eagle implementations, but
provides:
  - Full hardware backend compatibility (CUDA / Vulkan / Metal / CPU)
  - Consistent resource management via the existing InferenceSession framework
  - Zero dependency on Python model-loading libraries (no transformers/ctransformers)

Integration approach
--------------------
The ``EagleDraftEngine`` is instantiated by the ``SpeculativeOrchestrator`` and
generates draft sequences by calling ``_run_draft_subprocess()`` — a slim wrapper
around the existing ``ProcessManager`` that runs the draft model against the
current context and returns the generated token string.

The base model's ``InferenceSession`` then runs a **verification pass** using
``--n-predict <n_accepted + 1>`` to produce the authoritative output.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("InferenceOS.SpeculativeDecoding.EagleDraftEngine")


class EagleDraftEngine:
    """
    External draft model engine for Eagle-2 / small GGUF speculative decoding.

    Parameters
    ----------
    draft_model_path : str or Path
        Filesystem path to the draft GGUF model file.
    llama_exe_path : Path
        Path to the llama.cpp CLI executable (shared with base model sessions).
    hw_profile : dict
        Hardware profile for backend detection and thread allocation.
    draft_tokens : int
        Number of tokens the draft model generates per speculation step.
    backend : str
        Inference backend for the draft model (``"auto"``, ``"cuda"``, etc.).
    n_gpu_layers : int
        Number of draft model layers to offload to GPU.  -1 = auto (all layers
        that fit in remaining VRAM after base model).
    verbose : bool
        Log draft subprocess output for debugging.
    """

    def __init__(
        self,
        draft_model_path: str,
        llama_exe_path: Path,
        hw_profile: Dict[str, Any],
        draft_tokens: int = 5,
        backend: str = "auto",
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        self.draft_model_path = Path(draft_model_path)
        self.llama_exe_path = llama_exe_path
        self.hw_profile = hw_profile
        self.draft_tokens = draft_tokens
        self.backend = backend
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        # Validate draft model exists
        if not self.draft_model_path.exists():
            raise FileNotFoundError(
                f"Eagle draft model not found: {self.draft_model_path}\n"
                "Download a compatible small GGUF model and set "
                "'spec_eagle_draft_model' in RuntimeConfig."
            )

        # Derive thread count from hw_profile
        cpu_info = hw_profile.get("cpu", {})
        self._threads = int(cpu_info.get("physical_cores", 4))

        # Telemetry
        self.total_draft_calls: int = 0
        self.total_draft_tokens_generated: int = 0
        self.total_draft_latency_ms: float = 0.0

        logger.info(
            "EagleDraftEngine: draft_model=%s  draft_tokens=%d  backend=%s",
            self.draft_model_path.name, self.draft_tokens, self.backend,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate_drafts(
        self,
        context_text: str,
        n_drafts: Optional[int] = None,
    ) -> str:
        """
        Run the draft model on ``context_text`` and return draft output text.

        Parameters
        ----------
        context_text : str
            The current conversation context (prompt + generated text so far).
            Passed as the prompt to the draft model subprocess.
        n_drafts : int, optional
            Override ``self.draft_tokens`` for this call.

        Returns
        -------
        str
            The raw text output from the draft model (up to ``n_drafts`` tokens).
            Empty string if the draft subprocess fails or times out.
        """
        n = n_drafts or self.draft_tokens
        self.total_draft_calls += 1

        t_start = time.perf_counter()
        try:
            draft_text = self._run_draft_subprocess(context_text, n)
        except Exception as exc:
            logger.warning("EagleDraftEngine subprocess failed: %s", exc)
            draft_text = ""
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        self.total_draft_latency_ms += elapsed_ms
        if draft_text:
            # Rough token count from word boundaries (not exact but useful for stats)
            self.total_draft_tokens_generated += max(1, len(draft_text.split()))

        if self.verbose:
            logger.debug(
                "EagleDraft: latency=%.1fms  output=%r",
                elapsed_ms, draft_text[:40],
            )

        return draft_text

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def avg_draft_latency_ms(self) -> float:
        return self.total_draft_latency_ms / max(1, self.total_draft_calls)

    def get_stats(self) -> dict:
        return {
            "draft_model": self.draft_model_path.name,
            "total_draft_calls": self.total_draft_calls,
            "total_draft_tokens_generated": self.total_draft_tokens_generated,
            "avg_draft_latency_ms": round(self.avg_draft_latency_ms, 2),
        }

    # ------------------------------------------------------------------
    # Internal subprocess execution
    # ------------------------------------------------------------------

    def _run_draft_subprocess(self, context_text: str, n_drafts: int) -> str:
        """
        Launch a llama.cpp subprocess with the draft model and return its output.

        Uses a minimal argument set: only the essential flags needed for fast
        draft generation (no stats, no repetition penalty, greedy sampling).
        """
        # Resolve GPU layer count
        n_gpu = self.n_gpu_layers
        if n_gpu < 0:
            # Auto: offload as many layers as possible given remaining VRAM
            gpus = self.hw_profile.get("gpus", [])
            if gpus:
                free_mb = float(gpus[0].get("vram_free_mb", 2048.0))
                # Rough heuristic: 1 draft layer ≈ 40MB (typical small model)
                n_gpu = max(0, int(free_mb / 40))
            else:
                n_gpu = 0

        cmd = [
            str(self.llama_exe_path),
            "--model", str(self.draft_model_path),
            "--prompt", context_text[-2048:],   # truncate very long contexts
            "--n-predict", str(n_drafts),
            "--threads", str(self._threads),
            "--n-gpu-layers", str(n_gpu),
            "--temp", "0.0",    # greedy for deterministic drafts
            "--repeat-penalty", "1.0",
            "--no-display-prompt",
            "--log-disable",
        ]

        # Backend-specific flags
        if self.backend == "vulkan":
            cmd.append("--device-vulkan")
        elif self.backend == "metal":
            pass   # Metal is auto-detected by llama.cpp on macOS
        # cuda: no extra flag needed; handled by env-var GGML_CUDA_VISIBLE_DEVICES

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10.0,   # draft must be fast; 10s hard timeout
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            if self.verbose:
                logger.debug("Draft subprocess stderr: %s", result.stderr[:200])
            return ""

        return result.stdout.strip()
