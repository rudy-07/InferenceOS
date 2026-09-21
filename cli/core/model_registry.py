"""
model_registry.py
------------------
Model Library & Registry Manager for InferenceOS.

Persists registered models, nicknames, locations, default backends,
formats, profiles, tags, and notes to ~/.inferenceos/config/models.json.
Also performs universal auto-discovery of .gguf, .onnx, .safetensors,
.pt, .pth, .bin, .pkl, .pickle, and .obx files in local directories,
sibling projects (KaptaanLM, Qwen-Coder), and Hugging Face cache.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .config_manager import ConfigManager, get_config_manager
from orchestrator.model_parser import detect_model_format, ModelFormat

SUPPORTED_EXTENSIONS: Set[str] = {
    ".gguf",
    ".safetensors",
    ".onnx",
    ".ort",
    ".pt",
    ".pth",
    ".bin",
    ".pkl",
    ".pickle",
    ".joblib",
    ".obx",
    ".tflite",
    ".h5",
    ".keras",
}

IGNORE_DIR_PATTERNS = {
    ".git",
    "node_modules",
    "build",
    "cmakefiles",
    "__pycache__",
    ".venv",
    "site-packages",
}


class ModelRegistry:
    """
    Manages the multi-format model library and nicknames.
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

        # Backfill missing formats for existing models
        for k, v in self._models.items():
            if "format" not in v or not v["format"]:
                loc = v.get("location")
                if loc and Path(loc).exists():
                    v["format"] = detect_model_format(Path(loc)).value
                else:
                    v["format"] = "unknown"

        self.auto_discover()

    def save(self) -> None:
        """Save registered models to disk."""
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._models, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save model registry: {e}")

    def auto_discover(self, root_dir: Optional[Path] = None) -> List[Path]:
        """Auto-discover model files across standard directories, sibling repos, and caches."""
        discovered_paths: List[Path] = []
        project_root = Path(__file__).parent.parent.parent.resolve()
        parent_dir = project_root.parent

        search_dirs = [
            root_dir,
            project_root / "models",
            Path.home() / ".inferenceos" / "models",
            parent_dir / "KaptaanLM",
            parent_dir / "Qwen-Coder",
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "kaggle",
        ]
        valid_dirs = [d for d in search_dirs if d and d.exists()]

        for search_dir in valid_dirs:
            for file_path in search_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                ext = file_path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                # Skip build / git / cmake artifacts
                lower_parts = [p.lower() for p in file_path.parts]
                if any(ignored in lower_parts for ignored in IGNORE_DIR_PATTERNS):
                    continue

                # Skip non-model bin files
                if ext == ".bin" and ("cmake" in file_path.name.lower() or file_path.stat().st_size < 1000):
                    continue

                # Skip empty files
                if file_path.stat().st_size == 0:
                    continue

                discovered_paths.append(file_path)

                # Generate clean nickname
                fmt = detect_model_format(file_path)
                stem = file_path.stem.lower()

                # Prefix based on source
                if "huggingface" in lower_parts:
                    # e.g. models--EleutherAI--pythia-14m -> hf:pythia-14m
                    repo_part = [p for p in lower_parts if p.startswith("models--")]
                    if repo_part:
                        model_id = repo_part[0].replace("models--", "").split("--")[-1]
                        nick = f"hf:{model_id}" if stem == "model" else f"hf:{model_id}:{stem}"
                    else:
                        nick = f"hf:{stem}"
                elif "kaptaanlm" in lower_parts:
                    nick = f"kaptaan:{stem}"
                elif "qwen-coder" in lower_parts:
                    nick = f"qwen:{stem}"
                elif "test_formats" in lower_parts:
                    nick = f"test:{stem}"
                else:
                    nick = stem

                if nick not in self._models:
                    self._models[nick] = {
                        "nickname": nick,
                        "location": str(file_path.resolve()),
                        "format": fmt.value,
                        "backend": "auto",
                        "context": 4096,
                        "profile": "balanced",
                        "tags": ["auto-discovered", fmt.value],
                        "notes": f"Auto-discovered in {search_dir.name}",
                        "size_bytes": file_path.stat().st_size,
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
        fmt = detect_model_format(path)

        if isinstance(tags, str):
            all_tags = [t.strip() for t in tags.split(",") if t.strip()]
        else:
            all_tags = list(tags or [])
        if fmt.value not in all_tags:
            all_tags.append(fmt.value)

        self._models[nick] = {
            "nickname": nick,
            "location": str(path),
            "format": fmt.value,
            "backend": backend,
            "context": context,
            "profile": profile,
            "tags": all_tags,
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

    def get_model(self, query: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Find a model by nickname, partial name, or file path.
        """
        if not query:
            return None
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
        if p.exists() and p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            fmt = detect_model_format(p)
            return {
                "nickname": p.stem,
                "location": str(p.resolve()),
                "format": fmt.value,
                "backend": "auto",
                "context": 4096,
                "profile": "balanced",
                "tags": ["unregistered", fmt.value],
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
                or q in m.get("format", "").lower()
                or q in m["notes"].lower()
                or any(q in t.lower() for t in m.get("tags", []))
            ):
                results.append(m)
        return results

    def resolve_model_path(self, query: Optional[str]) -> Optional[Path]:
        """Resolve query string (nickname, path, or filename) to existing Path."""
        if not query:
            return None
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
