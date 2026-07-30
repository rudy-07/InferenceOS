"""
profile_manager.py
-------------------
Manages pre-packaged and user-created profiles for InferenceOS.

Built-in Profiles:
  - Balanced
  - Maximum Speed
  - Maximum Quality
  - Ultra Low Memory
  - Laptop
  - Desktop
  - Server
  - Reasoning
  - Coding
  - Creative
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from .config_manager import ConfigManager, get_config_manager


BUILTIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "name": "Balanced",
        "description": "Default optimal balance between inference speed, output quality, and memory headroom.",
        "settings": {
            "runtime.use_flash_attn": True,
            "runtime.threads": 6,
            "memory.context_length": 4096,
            "memory.batch_size": 512,
            "sampling.temp": 0.7,
            "sampling.top_p": 0.95,
            "sampling.top_k": 40,
            "sampling.min_p": 0.05,
        },
    },
    "maximum_speed": {
        "name": "Maximum Speed",
        "description": "Optimized for maximum token generation throughput (highest TPS).",
        "settings": {
            "runtime.use_flash_attn": True,
            "runtime.use_paged_kv": True,
            "runtime.threads": 8,
            "memory.context_length": 2048,
            "memory.batch_size": 1024,
            "sampling.temp": 0.5,
            "sampling.top_p": 0.9,
            "sampling.top_k": 20,
        },
    },
    "maximum_quality": {
        "name": "Maximum Quality",
        "description": "Optimized for precision output quality and extended context windows.",
        "settings": {
            "runtime.use_flash_attn": True,
            "runtime.threads": 8,
            "memory.context_length": 8192,
            "memory.batch_size": 512,
            "sampling.temp": 0.6,
            "sampling.top_p": 0.95,
            "sampling.top_k": 50,
            "sampling.repeat_penalty": 1.15,
        },
    },
    "ultra_low_memory": {
        "name": "Ultra Low Memory",
        "description": "Minimizes VRAM/RAM consumption to prevent OOM errors on constrained devices.",
        "settings": {
            "runtime.use_flash_attn": True,
            "runtime.threads": 4,
            "memory.context_length": 2048,
            "memory.batch_size": 256,
            "memory.microbatch": 64,
            "placement.fallback_cpu": True,
        },
    },
    "laptop": {
        "name": "Laptop",
        "description": "Balanced profile tuned for laptop thermals, battery efficiency, and iGPU offloading.",
        "settings": {
            "hardware.igpu_enable": True,
            "runtime.threads": 4,
            "memory.context_length": 4096,
            "memory.batch_size": 256,
            "telemetry.sample_interval_ms": 500,
        },
    },
    "desktop": {
        "name": "Desktop",
        "description": "Designed for high-performance desktop dGPUs with ample power budget.",
        "settings": {
            "hardware.igpu_enable": True,
            "runtime.threads": 8,
            "memory.context_length": 8192,
            "memory.batch_size": 512,
        },
    },
    "server": {
        "name": "Server",
        "description": "High-concurrency server profile with NUMA awareness and large context handling.",
        "settings": {
            "runtime.numa": True,
            "runtime.cpu_affinity": True,
            "runtime.threads": 16,
            "memory.context_length": 16384,
            "memory.batch_size": 2048,
        },
    },
    "reasoning": {
        "name": "Reasoning",
        "description": "Tuned for analytical tasks, math, logic, and chain-of-thought models.",
        "settings": {
            "sampling.temp": 0.2,
            "sampling.top_p": 0.95,
            "sampling.top_k": 20,
            "sampling.min_p": 0.05,
            "sampling.reasoning_budget": 1024,
            "generation.n_predict": 2048,
        },
    },
    "coding": {
        "name": "Coding",
        "description": "Deterministic sampling tuned for code generation, syntax accuracy, and autocomplete.",
        "settings": {
            "sampling.temp": 0.1,
            "sampling.top_p": 0.9,
            "sampling.top_k": 10,
            "sampling.repeat_penalty": 1.05,
            "memory.context_length": 8192,
            "generation.n_predict": 1024,
        },
    },
    "creative": {
        "name": "Creative",
        "description": "High-temperature, diverse sampling for storytelling, brainstorming, and roleplay.",
        "settings": {
            "sampling.temp": 0.9,
            "sampling.top_p": 0.98,
            "sampling.top_k": 80,
            "sampling.min_p": 0.02,
            "sampling.presence_penalty": 0.2,
            "sampling.frequency_penalty": 0.2,
            "generation.n_predict": 1024,
        },
    },
}


class ProfileManager:
    """
    Manages active runtime profiles and custom profile persistence.
    """

    def __init__(self, config_mgr: Optional[ConfigManager] = None) -> None:
        self.config_mgr = config_mgr or get_config_manager()
        self.profiles_dir = self.config_mgr.profiles_dir
        self._ensure_profiles_dir()

    def _ensure_profiles_dir(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Return list of all available builtin and custom profiles."""
        result: List[Dict[str, Any]] = []
        active = self.config_mgr.get("profiles.active_profile", "balanced").lower()

        # Add builtins
        for key, p in BUILTIN_PROFILES.items():
            item = {
                "key": key,
                "name": p["name"],
                "description": p["description"],
                "type": "builtin",
                "is_active": (key == active),
                "settings": p["settings"],
            }
            result.append(item)

        # Add custom profiles from ~/.inferenceos/profiles/*.json
        for f in self.profiles_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                key = f.stem.lower()
                result.append({
                    "key": key,
                    "name": data.get("name", f.stem),
                    "description": data.get("description", "Custom user profile"),
                    "type": "custom",
                    "is_active": (key == active),
                    "settings": data.get("settings", {}),
                })
            except Exception:
                continue

        return result

    def get_profile(self, name_or_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve profile by key name."""
        key = name_or_key.lower().replace(" ", "_")

        if key in BUILTIN_PROFILES:
            b = BUILTIN_PROFILES[key]
            return {
                "key": key,
                "name": b["name"],
                "description": b["description"],
                "type": "builtin",
                "settings": b["settings"],
            }

        custom_path = self.profiles_dir / f"{key}.json"
        if custom_path.exists():
            try:
                with open(custom_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "key": key,
                    "name": data.get("name", key),
                    "description": data.get("description", "Custom profile"),
                    "type": "custom",
                    "settings": data.get("settings", {}),
                }
            except Exception:
                return None

        return None

    def create_custom_profile(
        self, key: str, name: str, description: str, settings: Dict[str, Any]
    ) -> bool:
        """Create and persist a custom user profile."""
        clean_key = key.lower().replace(" ", "_")
        target_path = self.profiles_dir / f"{clean_key}.json"
        data = {
            "name": name,
            "description": description,
            "settings": settings,
        }
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"[Error] Failed to create profile {key}: {e}")
            return False

    def activate_profile(self, name_or_key: str) -> bool:
        """Apply profile settings to global config and record active_profile."""
        prof = self.get_profile(name_or_key)
        if not prof:
            return False

        settings = prof["settings"]
        for path, val in settings.items():
            self.config_mgr.set(path, val, save_now=False)

        self.config_mgr.set("profiles.active_profile", prof["key"], save_now=True)
        return True


_profile_manager_instance: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
    global _profile_manager_instance
    if _profile_manager_instance is None:
        _profile_manager_instance = ProfileManager()
    return _profile_manager_instance
