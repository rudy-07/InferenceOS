"""
model_registry.py
------------------
Model Library & Registry Manager for InferenceOS.

Persists registered GGUF models, nicknames, locations, default backends,
profiles, tags, and notes to ~/.inferenceos/config/models.json.
Also performs auto-discovery of .gguf files in local directories.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from .config_manager import ConfigManager, get_config_manager


class ModelRegistry:
    """
    Manages the GGUF model library and nicknames.
    """

    def __init__(self, config_mgr: Optional[ConfigManager] = None) -> None:
        self.config_mgr = config_mgr or get_config_manager()
        self.registry_file = self.config_mgr.config_dir / "models.json"
        self._models: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load registered models from disk."""
        if not self.registry_file.exists():
            self._models = {}
            self.auto_discover()
            self.save()
            return

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                self._models = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load model registry from {self.registry_file}: {e}")
            self._models = {}

        self.auto_discover()

    def save(self) -> None:
        """Save registered models to disk."""
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._models, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save model registry: {e}")

    def auto_discover(self, root_dir: Optional[Path] = None) -> List[Path]:
        """Auto-discover .gguf model files in standard directories."""
        discovered_paths: List[Path] = []
        search_dirs = [
            root_dir,
            Path.cwd() / "models",
            Path.home() / ".inferenceos" / "models",
        ]
        valid_dirs = [d for d in search_dirs if d and d.exists()]

        for search_dir in valid_dirs:
            for gguf_file in search_dir.rglob("*.gguf"):
                if gguf_file.is_file():
                    discovered_paths.append(gguf_file)
                    nick = gguf_file.stem.lower()
                    if nick not in self._models:
                        self._models[nick] = {
                            "nickname": nick,
                            "location": str(gguf_file.resolve()),
                            "backend": "auto",
                            "context": 4096,
                            "profile": "balanced",
                            "tags": ["auto-discovered"],
                            "notes": f"Auto-discovered in {search_dir}",
                            "size_bytes": gguf_file.stat().st_size,
                        }

        return discovered_paths

    def add_model(
        self,
        nickname: str,
        location: Union[str, Path],
        backend: str = "auto",
        context: int = 4096,
        profile: str = "balanced",
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> bool:
        """Register a model in the library."""
        path = Path(location).resolve()
        if not path.exists():
            print(f"[Error] Model file does not exist: {path}")
            return False

        nick = nickname.lower().strip()
        size_bytes = path.stat().st_size if path.exists() else 0

        self._models[nick] = {
            "nickname": nick,
            "location": str(path),
            "backend": backend,
            "context": context,
            "profile": profile,
            "tags": tags or [],
            "notes": notes,
            "size_bytes": size_bytes,
        }
        self.save()
        return True

    def remove_model(self, nickname_or_path: str) -> bool:
        """Remove a model entry from registry."""
        query = nickname_or_path.lower().strip()
        if query in self._models:
            del self._models[query]
            self.save()
            return True

        for k, v in list(self._models.items()):
            if v.get("location", "").lower() == query:
                del self._models[k]
                self.save()
                return True
        return False

    def get_model(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Find a model by nickname, partial name, or file path.
        """
        clean_q = query.lower().strip()
        if clean_q in self._models:
            return self._models[clean_q]

        # Search by location match
        for k, v in self._models.items():
            if v.get("location", "").lower() == clean_q or Path(v.get("location", "")).name.lower() == clean_q:
                return v

        # Substring match
        for k, v in self._models.items():
            if clean_q in k or clean_q in Path(v.get("location", "")).name.lower():
                return v

        # Direct file path check
        p = Path(query)
        if p.exists() and p.is_file() and p.suffix.lower() == ".gguf":
            return {
                "nickname": p.stem,
                "location": str(p.resolve()),
                "backend": "auto",
                "context": 4096,
                "profile": "balanced",
                "tags": ["unregistered"],
                "notes": "Direct path invocation",
                "size_bytes": p.stat().st_size,
            }

        return None

    def list_models(self) -> List[Dict[str, Any]]:
        """Return list of all registered models."""
        return list(self._models.values())

    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """Search models by tag, name, or note content."""
        q = query.lower()
        results: List[Dict[str, Any]] = []
        for m in self._models.values():
            if (
                q in m["nickname"].lower()
                or q in m["location"].lower()
                or q in m["notes"].lower()
                or any(q in t.lower() for t in m["tags"])
            ):
                results.append(m)
        return results

    def resolve_model_path(self, query: str) -> Optional[Path]:
        """Resolve query string (nickname, path, or filename) to existing Path."""
        model_info = self.get_model(query)
        if model_info:
            p = Path(model_info["location"])
            if p.exists():
                return p

        # Check raw query path directly
        raw_p = Path(query)
        if raw_p.exists() and raw_p.is_file():
            return raw_p.resolve()

        return None


_model_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _model_registry_instance
    if _model_registry_instance is None:
        _model_registry_instance = ModelRegistry()
    return _model_registry_instance
