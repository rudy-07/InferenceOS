"""
config_manager.py
------------------
Centralized configuration manager for InferenceOS.

Manages configuration files under ~/.inferenceos/:
  - ~/.inferenceos/config/global_config.json
  - ~/.inferenceos/profiles/
  - ~/.inferenceos/hardware/
  - ~/.inferenceos/cache/
  - ~/.inferenceos/logs/
  - ~/.inferenceos/telemetry/
  - ~/.inferenceos/sessions/
  - ~/.inferenceos/themes/
  - ~/.inferenceos/plugins/
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


DEFAULT_CONFIG: Dict[str, Any] = {
    "hardware": {
        "auto_detect": True,
        "device_priorities": ["cuda", "vulkan", "rocm", "metal", "cpu"],
        "igpu_enable": True,
        "vram_limit_mb": 0,  # 0 = autodetect
    },
    "runtime": {
        "backend": "auto",
        "threads": 6,
        "gpu_layers": None,  # None = auto placement
        "use_flash_attn": True,
        "use_paged_kv": True,
        "cpu_affinity": False,
        "numa": False,
        "tensor_split": None,
    },
    "placement": {
        "enable_auto_placement": True,
        "fallback_cpu": True,
        "max_boundary_crossings": 2,
    },
    "memory": {
        "context_length": 4096,
        "batch_size": 512,
        "microbatch": 128,
        "kv_cache_compression": False,
        "auto_resize_context": True,
    },
    "sampling": {
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "temp": 0.7,
        "repeat_penalty": 1.1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "mirostat": 0,
        "grammar": None,
        "reasoning_budget": 0,
        "seed": -1,
    },
    "generation": {
        "n_predict": 4096,
        "streaming": True,
        "async_streaming": True,
        "process_timeout_sec": 600,
    },
    "telemetry": {
        "enabled": True,
        "sample_interval_ms": 200,
        "export_flamegraph": True,
        "export_timeline": True,
    },
    "monitoring": {
        "refresh_rate_hz": 2.0,
    },
    "logging": {
        "level": "INFO",
        "log_to_file": True,
    },
    "appearance": {
        "theme": "nord",
        "show_status_bar": True,
        "animation_speed": 1.0,
        "code_highlighting": True,
        "markdown_render": True,
    },
    "profiles": {
        "active_profile": "balanced",
    },
}


class ConfigManager:
    """
    Manages global configuration file loading, saving, and section lookups.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or (Path.home() / ".inferenceos")
        self.config_dir = self.base_dir / "config"
        self.profiles_dir = self.base_dir / "profiles"
        self.hardware_dir = self.base_dir / "hardware"
        self.cache_dir = self.base_dir / "cache"
        self.logs_dir = self.base_dir / "logs"
        self.telemetry_dir = self.base_dir / "telemetry"
        self.sessions_dir = self.base_dir / "sessions"
        self.themes_dir = self.base_dir / "themes"
        self.plugins_dir = self.base_dir / "plugins"

        self.global_config_path = self.config_dir / "global_config.json"
        self._config_data: Dict[str, Any] = {}

        self._ensure_directories()
        self.load()

    def _ensure_directories(self) -> None:
        """Create all required InferenceOS directory structures."""
        for d in [
            self.base_dir,
            self.config_dir,
            self.profiles_dir,
            self.hardware_dir,
            self.cache_dir,
            self.logs_dir,
            self.telemetry_dir,
            self.sessions_dir,
            self.themes_dir,
            self.plugins_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load global_config.json or create default if not present."""
        if not self.global_config_path.exists():
            self._config_data = json.loads(json.dumps(DEFAULT_CONFIG))
            self.save()
            return

        try:
            with open(self.global_config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Deep merge defaults with loaded
            self._config_data = json.loads(json.dumps(DEFAULT_CONFIG))
            self._deep_update(self._config_data, loaded)
            if self.get("generation.n_predict", 512) <= 512:
                self.set("generation.n_predict", 4096, save_now=True)
        except Exception as e:
            print(f"[Warning] Failed to parse {self.global_config_path}: {e}. Resetting to defaults.")
            self._config_data = json.loads(json.dumps(DEFAULT_CONFIG))
            self.save()

    def save(self) -> None:
        """Save current configuration data to disk."""
        try:
            with open(self.global_config_path, "w", encoding="utf-8") as f:
                json.dump(self._config_data, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save config to {self.global_config_path}: {e}")

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for k, v in source.items():
            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                self._deep_update(target[k], v)
            else:
                target[k] = v

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config item using dot notation (e.g. 'runtime.threads').
        """
        keys = path.split(".")
        curr: Any = self._config_data
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr

    def set(self, path: str, value: Any, save_now: bool = True) -> None:
        """
        Set config item using dot notation (e.g. 'runtime.threads', 8).
        """
        keys = path.split(".")
        curr = self._config_data
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = value

        if save_now:
            self.save()

    def reset_to_defaults(self) -> None:
        """Reset configuration back to defaults."""
        self._config_data = json.loads(json.dumps(DEFAULT_CONFIG))
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        """Return raw dictionary of configuration."""
        return json.loads(json.dumps(self._config_data))

    def get_runtime_config_kwargs(self) -> Dict[str, Any]:
        """Convert configuration section into RuntimeConfig dictionary kwargs."""
        return {
            "force_backend": None if self.get("runtime.backend") == "auto" else self.get("runtime.backend"),
            "threads": self.get("runtime.threads", 6),
            "use_flash_attn": self.get("runtime.use_flash_attn", True),
            "context_length": self.get("memory.context_length", 4096),
            "batch_size": self.get("memory.batch_size", 512),
            "n_predict": self.get("generation.n_predict", 512),
            "temp": self.get("sampling.temp", 0.7),
            "top_p": self.get("sampling.top_p", 0.95),
            "top_k": self.get("sampling.top_k", 40),
            "repeat_penalty": self.get("sampling.repeat_penalty", 1.1),
            "seed": self.get("sampling.seed", -1),
            "async_streaming": self.get("generation.async_streaming", True),
            "enable_profiler": self.get("telemetry.enabled", True),
        }


_config_manager_instance: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager()
    return _config_manager_instance
