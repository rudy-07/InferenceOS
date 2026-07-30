"""
plugin_manager.py
------------------
Dynamic Plugin Architecture for InferenceOS.

Loads third-party python plugins from ~/.inferenceos/plugins/ and dispatches
hooks for custom commands, schedulers, backends, telemetry providers, and themes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from .config_manager import ConfigManager, get_config_manager


class PluginManifest:
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        entry_point: str = "main.py",
        enabled: bool = True,
        path: Optional[Path] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.entry_point = entry_point
        self.enabled = enabled
        self.path = path


class PluginManager:
    """
    Manages discovery, loading, enabling, and hook execution for plugins.
    """

    def __init__(self, config_mgr: Optional[ConfigManager] = None) -> None:
        self.config_mgr = config_mgr or get_config_manager()
        self.plugins_dir = self.config_mgr.plugins_dir
        self.loaded_plugins: Dict[str, PluginManifest] = {}
        self.hooks: Dict[str, List[Callable[..., Any]]] = {
            "on_startup": [],
            "on_inference_start": [],
            "on_inference_end": [],
            "on_token": [],
            "custom_commands": [],
        }
        self.discover_and_load()

    def discover_and_load(self) -> None:
        """Scan plugins directory and load all active plugins."""
        if not self.plugins_dir.exists():
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            return

        for item in self.plugins_dir.iterdir():
            manifest_file = item / "plugin.json" if item.is_dir() else None
            py_file = item if item.is_file() and item.suffix == ".py" else None

            if manifest_file and manifest_file.exists():
                self._load_from_dir(item, manifest_file)
            elif py_file:
                self._load_single_file(py_file)

    def _load_from_dir(self, plugin_dir: Path, manifest_path: Path) -> None:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            manifest = PluginManifest(
                name=data.get("name", plugin_dir.name),
                version=data.get("version", "1.0.0"),
                description=data.get("description", ""),
                author=data.get("author", ""),
                entry_point=data.get("entry_point", "main.py"),
                enabled=data.get("enabled", True),
                path=plugin_dir,
            )

            if not manifest.enabled:
                self.loaded_plugins[manifest.name] = manifest
                return

            entry_path = plugin_dir / manifest.entry_point
            if entry_path.exists():
                self._import_module_and_register(manifest.name, entry_path)
                self.loaded_plugins[manifest.name] = manifest
        except Exception as e:
            print(f"[Warning] Failed to load plugin dir {plugin_dir.name}: {e}")

    def _load_single_file(self, py_file: Path) -> None:
        name = py_file.stem
        manifest = PluginManifest(
            name=name,
            version="1.0.0",
            description="Single script plugin",
            enabled=True,
            path=py_file,
        )
        try:
            self._import_module_and_register(name, py_file)
            self.loaded_plugins[name] = manifest
        except Exception as e:
            print(f"[Warning] Failed to load plugin script {py_file.name}: {e}")

    def _import_module_and_register(self, plugin_name: str, file_path: Path) -> None:
        spec = importlib.util.spec_from_file_location(f"inferenceos_plugin_{plugin_name}", file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Register hooks if defined in module
            if hasattr(module, "register_plugin"):
                module.register_plugin(self)

    def register_hook(self, hook_name: str, callback: Callable[..., Any]) -> None:
        """Register a plugin hook callback."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)

    def trigger_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """Dispatch a hook to all registered callbacks."""
        results = []
        for cb in self.hooks.get(hook_name, []):
            try:
                res = cb(*args, **kwargs)
                results.append(res)
            except Exception as e:
                print(f"[Warning] Error executing plugin hook '{hook_name}': {e}")
        return results

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return list of discovered plugins and status."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "author": p.author,
                "enabled": p.enabled,
                "path": str(p.path) if p.path else "",
            }
            for p in self.loaded_plugins.values()
        ]


_plugin_manager_instance: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    global _plugin_manager_instance
    if _plugin_manager_instance is None:
        _plugin_manager_instance = PluginManager()
    return _plugin_manager_instance
