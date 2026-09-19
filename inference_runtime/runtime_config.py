"""
runtime_config.py
-----------------
Configuration dataclass for the Phase 4 Inference Execution Runtime.

Encapsulates every tunable parameter that controls how llama.exe is
invoked — generation settings, memory/transfer tuning flags, pipeline
options, and backend overrides. Designed to be constructed once per
session and treated as immutable during a run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RuntimeConfig:
    """
    Complete configuration for a single inference execution session.

    Generation Parameters
    ---------------------
    n_predict : int
        Maximum number of tokens to generate. Default 256.
    context_length : int
        Context window size in tokens. Clipped to model maximum at runtime.
        Default 4096.
    threads : int
        CPU threads to use for inference. ``-1`` = auto-detect physical cores.
        Default -1.
    batch_size : int
        Prompt processing batch size (tokens per batch). Default 512.
    temp : float
        Sampling temperature. 0.0 = greedy, higher = more random. Default 0.7.
    top_p : float
        Nucleus sampling probability threshold. Default 0.9.
    top_k : int
        Top-K sampling cutoff. Default 40.
    repeat_penalty : float
        Repetition penalty applied to recent tokens. Default 1.1.
    seed : int
        RNG seed. ``-1`` = random each run. Default -1.

    Memory / Transfer Tuning
    ------------------------
    use_flash_attn : bool
        Enable Flash Attention (``--flash-attn``) to reduce KV cache memory
        copy operations. Requires compatible llama.cpp build. Default True.
    use_mmap : bool
        Use memory-mapped file I/O (``--mmap``) for model loading. Set False
        (``--no-mmap``) to load the model fully into RAM, which enables OS
        pinned-memory behaviour. Default True.
    use_mlock : bool
        Lock model weights in physical RAM (``--mlock``) preventing swap.
        Eliminates page-fault latency during inference. Default False (requires
        elevated permissions on some systems).
    numa_strategy : int
        NUMA memory policy for multi-socket systems.
        ``0`` = disabled, ``1`` = distribute, ``2`` = isolate, ``3`` = numactl.
        Default 0.

    Pipeline / Streaming
    --------------------
    async_streaming : bool
        Enable background thread token streaming with ``on_token`` callbacks.
        Default True.
    stats_sample_interval_ms : float
        Interval in milliseconds between OS-level CPU/GPU utilization samples
        collected by the stats background thread. Default 250 ms.
    process_timeout_sec : float
        Maximum wall-clock seconds to wait for the subprocess to finish before
        forcibly terminating it. Default 600 s (10 minutes).

    Backend Override
    ----------------
    force_backend : str, optional
        Override auto-detected backend. One of ``"cuda"``, ``"vulkan"``,
        ``"metal"``, ``"cpu"``. ``None`` = auto-detect from hardware profile.

    Benchmark Mode
    --------------
    benchmark_warmup_tokens : int
        Tokens generated during warmup runs (excluded from statistics).
        Default 32.
    """

    # Generation
    n_predict: int = 4096
    context_length: int = 4096
    threads: int = -1
    batch_size: int = 512
    temp: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = -1

    # Memory / transfer tuning
    use_flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False
    numa_strategy: int = 0

    # Pipeline
    async_streaming: bool = True
    stats_sample_interval_ms: float = 250.0
    process_timeout_sec: float = 600.0

    # Backend override (None = auto)
    force_backend: Optional[str] = None

    # Benchmark
    benchmark_warmup_tokens: int = 32

    # Phase 6 Pre-flight Memory Optimization
    run_preflight_check: bool = False
    block_on_critical_oom: bool = True
    auto_resize_context: bool = True

    # Phase 8 Async Scheduler
    enable_async_scheduler: bool = False
    scheduler_n_cpu_workers: int = -1   # -1 = auto (physical_cores // 2)
    scheduler_queue_depth: int = 8
    scheduler_prefetch: bool = True
    scheduler_benchmark_mode: bool = False

    # Phase 9 Runtime Profiler
    enable_profiler: bool = False
    profiler_sample_interval_ms: float = 100.0
    export_profile_html: bool = False

    # Dynamic Microbatch Scheduler
    enable_dynamic_microbatch: bool = True
    min_microbatch: int = 128
    max_microbatch: int = 2048
    microbatch_safety_margin: float = 0.15
    microbatch_aggressiveness: float = 1.0
    microbatch_optimization_goal: str = "throughput"  # throughput | latency | balanced
    manual_microbatch: Optional[int] = None
    verbose_microbatch_scheduler: bool = False

    # Dynamic Context Scheduler
    enable_dynamic_context: bool = True
    min_context: int = 512
    max_context: int = 131072
    context_safety_margin: float = 0.15
    context_optimization_goal: str = "maximum_context"  # maximum_context | maximum_speed | balanced
    manual_context: Optional[int] = None
    verbose_context_scheduler: bool = False

    # Adaptive Memory Scheduler
    enable_adaptive_memory_scheduler: bool = True
    vram_safety_margin: float = 0.15
    ram_safety_margin: float = 0.15
    memory_strategy: str = "auto"  # auto | conservative | balanced | aggressive
    oom_prevention: str = "strict"  # strict | balanced | disabled
    verbose_memory_scheduler: bool = False

    # Runtime Learning Engine & ARTI
    enable_runtime_learning: bool = True
    verbose_learning_engine: bool = False

    # Intelligent KV Manager
    enable_kv_manager: bool = True
    kv_compression_enabled: bool = True
    kv_eviction_enabled: bool = True
    kv_compression_mode: str = "adaptive"  # disabled | lossless | balanced | aggressive | adaptive
    kv_eviction_policy: str = "adaptive"    # lru | fifo | lfu | adaptive
    verbose_kv_manager: bool = False

    # Runtime Health Monitor & Memory Budget Manager
    enable_health_monitor: bool = True
    enable_memory_budget_manager: bool = True
    budget_policy_mode: str = "adaptive"  # adaptive | balanced | conservative | aggressive | server | low_memory
    verbose_health_monitor: bool = False
    verbose_budget_manager: bool = False

    # Runtime Knowledge Base (RKB)
    enable_rkb: bool = True
    rkb_dir: str = "~/.inferenceos/knowledge"
    verbose_rkb: bool = False

    # Automatic Performance Optimizer (APO)
    enable_apo: bool = True
    apo_goal: str = "Balanced"  # Balanced | Max Throughput | Lowest Latency | Lowest Memory | Max Stability
    force_optimization: bool = False
    optimization_dir: str = "~/.inferenceos/optimization"
    verbose_apo: bool = False

    # Performance Intelligence Engine (PIE)
    enable_pie: bool = True
    pie_dir: str = "~/.inferenceos/performance"
    verbose_pie: bool = False

    # ── Phase 5: Speculative Decoding Suite ───────────────────────────────────
    # All flags are opt-in (disabled by default). Enable via CLI flag
    # (--speculative) or by setting enable_speculative_decoding=True here.
    enable_speculative_decoding: bool = False
    # Strategy: "ngram" | "prompt_lookup" | "eagle" | "auto"
    #   "ngram"         — N-gram suffix match from generated context (zero VRAM)
    #   "prompt_lookup" — Suffix match against input prompt (great for RAG/code)
    #   "eagle"         — External draft GGUF model (requires spec_eagle_draft_model)
    #   "auto"          — Uses eagle if a draft model is set, else falls back to ngram
    spec_mode: str = "auto"
    spec_draft_tokens: int = 5               # draft tokens per speculation step
    spec_ngram_size: int = 3                 # n-gram window size (2–8)
    spec_min_match_length: int = 3           # min suffix match length to trigger drafts
    spec_prompt_lookup_window: int = 0       # 0 = entire prompt; >0 = token window
    spec_acceptance_threshold: float = 0.0  # min acceptance probability (0 = greedy)
    spec_acceptance_strategy: str = "greedy" # "greedy" | "speculative"
    spec_eagle_draft_model: Optional[str] = None  # path to draft GGUF model
    spec_max_rounds: int = 8                 # safety cap on speculation cycles
    spec_fallback_to_greedy: bool = True     # fall back if no drafts generated
    spec_record_telemetry: bool = True       # write acceptance stats to Runtime Learning DB
    verbose_spec_decoding: bool = False


    def __post_init__(self) -> None:
        # Auto-resolve thread count from physical cores
        if self.threads == -1:
            try:
                import psutil
                self.threads = psutil.cpu_count(logical=False) or os.cpu_count() or 4
            except ImportError:
                self.threads = os.cpu_count() or 4

        # Clamp values to sane ranges
        self.threads = max(1, self.threads)
        self.n_predict = max(1, self.n_predict)
        self.context_length = max(128, self.context_length)
        self.batch_size = max(1, self.batch_size)
        self.temp = max(0.0, self.temp)
        self.top_p = max(0.0, min(1.0, self.top_p))
        self.top_k = max(0, self.top_k)
        self.repeat_penalty = max(1.0, self.repeat_penalty)
        self.stats_sample_interval_ms = max(50.0, self.stats_sample_interval_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of all config values."""
        return {
            "n_predict": self.n_predict,
            "context_length": self.context_length,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "temp": self.temp,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
            "use_flash_attn": self.use_flash_attn,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "numa_strategy": self.numa_strategy,
            "async_streaming": self.async_streaming,
            "stats_sample_interval_ms": self.stats_sample_interval_ms,
            "process_timeout_sec": self.process_timeout_sec,
            "force_backend": self.force_backend,
            "benchmark_warmup_tokens": self.benchmark_warmup_tokens,
            "run_preflight_check": self.run_preflight_check,
            "block_on_critical_oom": self.block_on_critical_oom,
            "auto_resize_context": self.auto_resize_context,
            # Phase 8
            "enable_async_scheduler": self.enable_async_scheduler,
            "scheduler_n_cpu_workers": self.scheduler_n_cpu_workers,
            "scheduler_queue_depth": self.scheduler_queue_depth,
            "scheduler_prefetch": self.scheduler_prefetch,
            "scheduler_benchmark_mode": self.scheduler_benchmark_mode,
            # Phase 9
            "enable_profiler": self.enable_profiler,
            "profiler_sample_interval_ms": self.profiler_sample_interval_ms,
            "export_profile_html": self.export_profile_html,
            # Microbatch Scheduler
            "enable_dynamic_microbatch": self.enable_dynamic_microbatch,
            "min_microbatch": self.min_microbatch,
            "max_microbatch": self.max_microbatch,
            "microbatch_safety_margin": self.microbatch_safety_margin,
            "microbatch_aggressiveness": self.microbatch_aggressiveness,
            "microbatch_optimization_goal": self.microbatch_optimization_goal,
            "manual_microbatch": self.manual_microbatch,
            "verbose_microbatch_scheduler": self.verbose_microbatch_scheduler,
            # Context Scheduler
            "enable_dynamic_context": self.enable_dynamic_context,
            "min_context": self.min_context,
            "max_context": self.max_context,
            "context_safety_margin": self.context_safety_margin,
            "context_optimization_goal": self.context_optimization_goal,
            "manual_context": self.manual_context,
            "verbose_context_scheduler": self.verbose_context_scheduler,
            # Memory Scheduler
            "enable_adaptive_memory_scheduler": self.enable_adaptive_memory_scheduler,
            "vram_safety_margin": self.vram_safety_margin,
            "ram_safety_margin": self.ram_safety_margin,
            "memory_strategy": self.memory_strategy,
            "oom_prevention": self.oom_prevention,
            "verbose_memory_scheduler": self.verbose_memory_scheduler,
            # Runtime Learning Engine & ARTI
            "enable_runtime_learning": self.enable_runtime_learning,
            "verbose_learning_engine": self.verbose_learning_engine,
            # Intelligent KV Manager
            "enable_kv_manager": self.enable_kv_manager,
            "kv_compression_enabled": self.kv_compression_enabled,
            "kv_eviction_enabled": self.kv_eviction_enabled,
            "kv_compression_mode": self.kv_compression_mode,
            "kv_eviction_policy": self.kv_eviction_policy,
            "verbose_kv_manager": self.verbose_kv_manager,
            # Runtime Health Monitor & Memory Budget Manager
            "enable_health_monitor": self.enable_health_monitor,
            "enable_memory_budget_manager": self.enable_memory_budget_manager,
            "budget_policy_mode": self.budget_policy_mode,
            "verbose_health_monitor": self.verbose_health_monitor,
            "verbose_budget_manager": self.verbose_budget_manager,
            # Runtime Knowledge Base (RKB)
            "enable_rkb": self.enable_rkb,
            "rkb_dir": self.rkb_dir,
            "verbose_rkb": self.verbose_rkb,
            # Automatic Performance Optimizer (APO)
            "enable_apo": self.enable_apo,
            "apo_goal": self.apo_goal,
            "force_optimization": self.force_optimization,
            "optimization_dir": self.optimization_dir,
            "verbose_apo": self.verbose_apo,
            # Performance Intelligence Engine (PIE)
            "enable_pie": self.enable_pie,
            "pie_dir": self.pie_dir,
            "verbose_pie": self.verbose_pie,
            # Phase 5: Speculative Decoding
            "enable_speculative_decoding": self.enable_speculative_decoding,
            "spec_mode": self.spec_mode,
            "spec_draft_tokens": self.spec_draft_tokens,
            "spec_ngram_size": self.spec_ngram_size,
            "spec_min_match_length": self.spec_min_match_length,
            "spec_prompt_lookup_window": self.spec_prompt_lookup_window,
            "spec_acceptance_threshold": self.spec_acceptance_threshold,
            "spec_acceptance_strategy": self.spec_acceptance_strategy,
            "spec_eagle_draft_model": self.spec_eagle_draft_model,
            "spec_max_rounds": self.spec_max_rounds,
            "spec_fallback_to_greedy": self.spec_fallback_to_greedy,
            "spec_record_telemetry": self.spec_record_telemetry,
            "verbose_spec_decoding": self.verbose_spec_decoding,
        }

    @classmethod
    def from_hw_profile(cls, hw_profile: dict, **overrides: Any) -> "RuntimeConfig":
        """
        Construct a :class:`RuntimeConfig` with sensible defaults derived
        from a hardware profile dict.

        Physical core count is read from ``hw_profile["cpu"]["physical_cores"]``
        if present. All other fields use defaults unless overridden via kwargs.

        Parameters
        ----------
        hw_profile : dict
            Hardware profile from ``hardware_profile.json`` or
            ``profiler.get_system_resources()``.
        **overrides
            Keyword arguments passed directly to the :class:`RuntimeConfig`
            constructor, taking precedence over auto-detected values.

        Returns
        -------
        RuntimeConfig
            Populated configuration instance.
        """
        cpu = hw_profile.get("cpu", {})
        physical_cores = int(cpu.get("physical_cores", -1))
        isa = cpu.get("isa_extensions", [])

        # Enable mlock only when we have plenty of RAM (> 24 GB)
        ram = hw_profile.get("ram", hw_profile.get("memory", {}))
        total_ram_gb = float(ram.get("total_gb", 0.0))
        auto_mlock = total_ram_gb >= 24.0

        # NUMA: enable distribute if multiple nodes detected
        numa_nodes = int(cpu.get("numa_nodes", 1))
        auto_numa = 1 if numa_nodes > 1 else 0

        kwargs: Dict[str, Any] = {
            "threads": physical_cores,
            "use_mlock": auto_mlock,
            "numa_strategy": auto_numa,
        }
        kwargs.update(overrides)
        return cls(**kwargs)
