"""
argument_builder.py
--------------------
Translates a Phase 3 PlacementPlan + RuntimeConfig + BackendInfo into a
complete, validated llama.exe CLI argument list.

This is the sole translation layer between InferenceOS's internal data
structures and the llama.cpp subprocess invocation. All CLI flag logic
lives here so other modules stay clean of string manipulation.

Argument mapping overview
--------------------------
  PlacementPlan                     → -ngl, -c, --split-mode
  RuntimeConfig.n_predict           → -n
  RuntimeConfig.threads             → -t
  RuntimeConfig.batch_size          → -b / -ub
  RuntimeConfig.temp                → --temp
  RuntimeConfig.top_p               → --top-p
  RuntimeConfig.top_k               → --top-k
  RuntimeConfig.repeat_penalty      → --repeat-penalty
  RuntimeConfig.seed                → --seed
  RuntimeConfig.use_flash_attn      → --flash-attn
  RuntimeConfig.use_mmap=False      → --no-mmap
  RuntimeConfig.use_mlock           → --mlock
  RuntimeConfig.numa_strategy > 0   → --numa distribute/isolate/numactl
  BackendInfo.extra_flags           → appended verbatim
"""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend_selector import BackendInfo
from .runtime_config import RuntimeConfig
from layer_placement.placement_plan import PlacementPlan


# NUMA strategy index → llama.cpp flag value
_NUMA_FLAGS: Dict[int, str] = {
    1: "distribute",
    2: "isolate",
    3: "numactl",
}


class ArgumentBuilder:
    """
    Builds a complete ``llama.exe`` CLI argument list from structured inputs.

    Parameters
    ----------
    llama_exe_path : Path
        Absolute path to the compiled ``llama.exe`` (or ``llama`` on Linux/macOS).
    """

    def __init__(self, llama_exe_path: Path) -> None:
        if not llama_exe_path.exists():
            raise FileNotFoundError(
                f"llama executable not found at: {llama_exe_path}\n"
                "Please run the build engine first (setup phase)."
            )
        self.llama_exe_path = llama_exe_path

    def build(
        self,
        model_path: Path,
        prompt: str,
        plan: PlacementPlan,
        config: RuntimeConfig,
        backend: BackendInfo,
        log_file: Optional[Path] = None,
        microbatch: Optional[int] = None,
    ) -> List[str]:
        """
        Build the complete subprocess argument list.

        Parameters
        ----------
        model_path : Path
            Absolute path to the ``.gguf`` model file.
        prompt : str
            Input prompt string for inference.
        plan : PlacementPlan
            Optimized placement plan from Phase 3.
        config : RuntimeConfig
            Runtime configuration parameters.
        backend : BackendInfo
            Resolved backend from :class:`BackendSelector`.
        log_file : Path, optional
            If provided, adds ``--log-file <path>`` so llama.cpp writes its
            timing statistics to a parseable file (in addition to stderr).
        microbatch : int, optional
            Dynamically scheduled microbatch size (prefill ubatch size).

        Returns
        -------
        List[str]
            Complete argument list starting with the executable path.
        """
        args: List[str] = [str(self.llama_exe_path)]

        # llama.exe uses "completion" subcommand for text generation
        args.append("completion")

        # ---- Core model ----
        args.extend(["-m", str(Path(model_path).resolve())])

        # ---- Prompt ----
        # Use -p for inline prompt; send EOF via stdin=DEVNULL for non-interactive
        args.extend(["-p", prompt])
        args.append("--no-display-prompt")

        # ---- GPU offload ----
        args.extend(["-ngl", str(backend.n_gpu_layers)])

        # ---- Context window ----
        effective_ctx = min(config.context_length, plan.context_length)
        args.extend(["-c", str(effective_ctx)])

        # ---- Token generation limit ----
        args.extend(["-n", str(config.n_predict)])

        # ---- Threading ----
        args.extend(["-t", str(config.threads)])

        # ---- Batch size (prompt + generation) ----
        effective_ubatch = microbatch if (microbatch and microbatch > 0) else config.batch_size
        effective_batch = max(config.batch_size, effective_ubatch)
        args.extend(["-b", str(effective_batch)])
        # Ubatch = micro-batch size used during prompt processing
        args.extend(["-ub", str(effective_ubatch)])

        # ---- Sampling ----
        args.extend(["--temp", str(config.temp)])
        args.extend(["--top-p", str(config.top_p)])
        args.extend(["--top-k", str(config.top_k)])
        args.extend(["--repeat-penalty", str(config.repeat_penalty)])

        if config.seed != -1:
            args.extend(["--seed", str(config.seed)])

        # ---- Memory / transfer tuning ----
        # Flash attention reduces KV cache copy operations (pinned-memory equivalent)
        if config.use_flash_attn and backend.supports_flash_attn:
            args.extend(["--flash-attn", "on"])

        # mmap: default on; disable for fully RAM-resident (pinned) loading
        if not config.use_mmap:
            args.append("--no-mmap")

        # mlock: pin model weights in physical RAM (prevent swap)
        if config.use_mlock:
            args.append("--mlock")

        # ---- NUMA strategy ----
        if config.numa_strategy > 0:
            numa_val = _NUMA_FLAGS.get(config.numa_strategy, "distribute")
            args.extend(["--numa", numa_val])

        # ---- Split mode (multi-segment plans) ----
        if backend.split_mode != "none":
            args.extend(["--split-mode", backend.split_mode])

        # ---- Disable conversation mode (single-shot inference) ----
        args.append("-no-cnv")

        # ---- Backend-specific extra flags ----
        args.extend(backend.extra_flags)

        # ---- Log file for stats parsing ----
        if log_file is not None:
            args.extend(["--log-file", str(log_file)])

        return args

    def build_benchmark_args(
        self,
        model_path: Path,
        plan: PlacementPlan,
        config: RuntimeConfig,
        backend: BackendInfo,
        n_predict: int = 128,
        log_file: Optional[Path] = None,
        microbatch: Optional[int] = None,
    ) -> List[str]:
        """
        Build arguments for a benchmark-mode run.
        """
        benchmark_prompt = (
            "The transformer architecture revolutionized natural language processing by "
            "introducing self-attention mechanisms that allow models to"
        )

        import copy
        bench_config = copy.copy(config)
        bench_config.n_predict = n_predict
        bench_config.seed = 42  # deterministic sampling for reproducibility

        return self.build(
            model_path=model_path,
            prompt=benchmark_prompt,
            plan=plan,
            config=bench_config,
            backend=backend,
            log_file=log_file,
            microbatch=microbatch,
        )

    def describe(
        self,
        model_path: Path,
        prompt: str,
        plan: PlacementPlan,
        config: RuntimeConfig,
        backend: BackendInfo,
    ) -> str:
        """
        Return a human-readable description of the launch configuration,
        useful for logging without logging the full argument list.
        """
        lines = [
            "InferenceOS Execution Configuration",
            "─" * 50,
            f"  Model:        {model_path.name}",
            f"  Backend:      {backend.name.upper()}",
            f"  GPU Layers:   {backend.n_gpu_layers} / {plan.n_gpu_layers + plan.n_cpu_layers}",
            f"  Context:      {min(config.context_length, plan.context_length)} tokens",
            f"  Predict:      {config.n_predict} tokens",
            f"  Threads:      {config.threads}",
            f"  Batch:        {config.batch_size}",
            f"  Flash Attn:   {config.use_flash_attn and backend.supports_flash_attn}",
            f"  mmap:         {config.use_mmap}",
            f"  mlock:        {config.use_mlock}",
            f"  Split Mode:   {backend.split_mode}",
            f"  Boundaries:   {plan.boundary_crossings}",
            f"  Temp:         {config.temp}",
        ]
        return "\n".join(lines)
