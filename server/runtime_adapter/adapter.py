"""
adapter.py
----------
Runtime Adapter connecting HTTP endpoints and SessionManager to the underlying
InferenceOS Runtime Engine, Placement Engine, and Model Registry.
"""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from cli.core.model_registry import get_model_registry, ModelRegistry
import profiler
from layer_placement import ModelDescriptor, PlacementEngine
from layer_placement.placement_plan import PlacementPlan
from inference_runtime import InferenceSession, RuntimeConfig, RuntimeEngine, InferenceResult
from orchestrator.gguf_parser import read_gguf_metadata
from orchestrator.model_parser import read_model_metadata, detect_model_format, ModelFormat
from server.config import ServerConfig
from server.models.openai import ChatMessage, ChatCompletionRequest, CompletionRequest
from server.sessions.session import Session


class RuntimeAdapter:
    """
    Adapter bridging OpenAI API requests & Session state with InferenceOS runtime.
    Features lazy loading, cached placement plans, and shared weight references.
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self.config = config or ServerConfig()
        self.registry = get_model_registry()
        
        # Resolve system hardware profile
        sys_res = profiler.get_system_resources()
        self.hw_profile = sys_res.to_dict() if hasattr(sys_res, "to_dict") else sys_res
        
        self.placement_engine = PlacementEngine(hw_profile=self.hw_profile)
        self.runtime_engine = RuntimeEngine(hw_profile=self.hw_profile)
        
        self._lock = threading.Lock()
        # Model cache holding: path, descriptor, placement plan, meta, loaded_count
        self._model_cache: Dict[str, Dict[str, Any]] = {}

    def get_registered_models(self) -> List[Dict[str, Any]]:
        """Return list of models registered in InferenceOS registry + loaded status."""
        reg_models = self.registry.list_models()
        results = []
        for m in reg_models:
            nick = m.get("nickname", "").lower()
            is_loaded = nick in self._model_cache
            item = dict(m)
            item["loaded"] = is_loaded
            item["status"] = "loaded" if is_loaded else "registered"
            results.append(item)
        return results

    def get_loaded_models_count(self) -> int:
        """Return number of currently loaded models in weight cache."""
        with self._lock:
            return len(self._model_cache)

    def load_model_if_needed(self, model_query: str) -> Dict[str, Any]:
        """
        Lazily load model and calculate optimal hardware placement.
        Caches placement plan for future requests to prevent duplicate loads.
        """
        with self._lock:
            clean_name = model_query.lower().strip()
            if clean_name in self._model_cache:
                self._model_cache[clean_name]["last_used"] = time.time()
                return self._model_cache[clean_name]

            model_path = self.registry.resolve_model_path(model_query)
            if not model_path or not model_path.exists():
                raise FileNotFoundError(f"Model '{model_query}' not found in registry or disk.")

            fmt = detect_model_format(model_path)
            try:
                meta = read_model_metadata(model_path)
            except Exception:
                meta = {"arch": "llama", "num_layers": 32, "max_context_length": self.config.default_context_length}

            model_desc = ModelDescriptor.from_gguf_metadata(
                metadata=meta,
                model_size_bytes=model_path.stat().st_size,
                model_name=model_path.stem,
            )

            context_len = min(
                getattr(model_desc, "max_context_length", 4096),
                self.config.default_context_length
            )

            plan: PlacementPlan = self.placement_engine.generatePlacementPlan(
                model=model_desc,
                context_length=context_len
            )

            cache_entry = {
                "name": clean_name,
                "model_path": model_path,
                "descriptor": model_desc,
                "plan": plan,
                "meta": meta,
                "loaded_time": time.time(),
                "last_used": time.time(),
            }

            # Enforce max models cached limit
            if len(self._model_cache) >= self.config.max_models:
                oldest = min(self._model_cache.keys(), key=lambda k: self._model_cache[k]["last_used"])
                del self._model_cache[oldest]

            self._model_cache[clean_name] = cache_entry
            return cache_entry

    def format_chat_prompt(self, messages: List[ChatMessage], model_name: str = "") -> str:
        """
        Format list of ChatMessages into appropriate prompt template matching model architecture.
        """
        clean_name = model_name.lower()
        
        # 1. Llama-3 template
        if "llama-3" in clean_name or "llama3" in clean_name:
            parts = ["<|begin_of_text|>"]
            for msg in messages:
                role = msg.role or "user"
                content = msg.text_content()
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            if not messages or messages[-1].role != "assistant":
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        # 2. Mistral / Llama-2 INST template
        if "mistral" in clean_name or "llama-2" in clean_name or "llama2" in clean_name:
            parts = []
            sys_text = ""
            user_text = ""
            for msg in messages:
                role = msg.role or "user"
                content = msg.text_content()
                if role == "system":
                    sys_text = f"<<SYS>>\n{content}\n<</SYS>>\n\n"
                elif role == "user":
                    user_text = content
                    parts.append(f"[INST] {sys_text}{user_text} [/INST]")
                    sys_text = ""
                elif role == "assistant":
                    parts.append(f" {content} ")
            if not messages or messages[-1].role != "assistant":
                if not parts:
                    parts.append("[INST] Hello [/INST]")
            return "".join(parts)

        # 3. Universal / ChatML fallback
        formatted_parts = []
        for msg in messages:
            role = msg.role or "user"
            content = msg.text_content()
            formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if not messages or messages[-1].role != "assistant":
            formatted_parts.append("<|im_start|>assistant\n")

        return "".join(formatted_parts)

    def build_runtime_config(
        self,
        request_overrides: Dict[str, Any],
        session_overrides: Optional[Dict[str, Any]] = None,
    ) -> RuntimeConfig:
        """
        Build per-request RuntimeConfig applying session and request-level overrides.
        """
        combined = {}
        if session_overrides:
            combined.update(session_overrides)
        if request_overrides:
            combined.update(request_overrides)

        cfg_kwargs = {}
        if self.config.backend:
            cfg_kwargs["force_backend"] = self.config.backend
        if "backend" in combined and combined["backend"]:
            cfg_kwargs["force_backend"] = combined["backend"]
        if "temperature" in combined and combined["temperature"] is not None:
            cfg_kwargs["temp"] = combined["temperature"]
        if "top_p" in combined and combined["top_p"] is not None:
            cfg_kwargs["top_p"] = combined["top_p"]
        if "top_k" in combined and combined["top_k"] is not None:
            cfg_kwargs["top_k"] = combined["top_k"]
        if "max_tokens" in combined and combined["max_tokens"]:
            cfg_kwargs["n_predict"] = combined["max_tokens"]
        if "seed" in combined and combined["seed"] is not None:
            cfg_kwargs["seed"] = combined["seed"]

        return RuntimeConfig.from_hw_profile(self.hw_profile, **cfg_kwargs)

    def execute_chat(
        self,
        model_query: str,
        messages: List[ChatMessage],
        request_overrides: Dict[str, Any],
        session: Optional[Session] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> InferenceResult:
        """
        Execute chat inference using lazy-loaded model and cached placement plan.
        """
        model_cache = self.load_model_if_needed(model_query)
        plan: PlacementPlan = model_cache["plan"]
        model_path: Path = model_cache["model_path"]

        session_overrides = session.config_overrides if session else {}
        runtime_cfg = self.build_runtime_config(request_overrides, session_overrides)

        # Build full prompt
        if session:
            # If session is used, prompt contains full updated session messages
            prompt = self.format_chat_prompt(session.messages, model_name=model_query)
        else:
            prompt = self.format_chat_prompt(messages, model_name=model_query)

        # Route execution based on model format
        fmt = detect_model_format(model_path)
        if fmt == ModelFormat.GGUF:
            with InferenceSession(
                model_path=model_path,
                plan=plan,
                config=runtime_cfg,
                hw_profile=self.hw_profile,
            ) as inf_session:
                result = inf_session.run(prompt, on_token=on_token)
        else:
            from inference_runtime import MultiFormatRuntimeEngine
            engine = MultiFormatRuntimeEngine(hw_profile=self.hw_profile, config=runtime_cfg)
            result = engine.execute(
                model_path=model_path,
                prompt=prompt,
                on_token=on_token,
                plan=plan,
            )

        # Update placement info on session
        if session:
            session.placement_info = {
                "backend": result.backend,
                "n_gpu_layers": plan.n_gpu_layers,
                "summary": plan.summary(),
            }
            stats_dict = result.stats.to_dict() if hasattr(result.stats, "to_dict") else {}
            session.update_stats(stats_dict, new_tokens=getattr(result.stats, "tokens_generated", 0))

        return result


_adapter_instance: Optional[RuntimeAdapter] = None


def get_runtime_adapter(config: Optional[ServerConfig] = None) -> RuntimeAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = RuntimeAdapter(config=config)
    return _adapter_instance
