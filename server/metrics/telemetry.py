"""
telemetry.py
------------
Telemetry collector and Prometheus/JSON metrics formatter for InferenceOS Server.
"""
from __future__ import annotations

import time
import threading
import psutil
from typing import Any, Dict, List, Optional
import profiler


class ServerTelemetryCollector:
    """
    Central server metrics collector tracking requests, latencies, TPS,
    utilization, active sessions, and errors.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.server_start_time = time.time()
        
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        self.total_tokens_generated = 0
        self.total_prompt_tokens = 0
        self.total_reasoning_tokens = 0
        
        self.latencies_ms: List[float] = []
        self.ttft_ms: List[float] = []
        self.generation_tps_list: List[float] = []
        self.prompt_tps_list: List[float] = []
        
        self.error_counts: Dict[str, int] = {}

    def record_request_start(self) -> float:
        with self._lock:
            self.total_requests += 1
        return time.perf_counter()

    def record_request_success(
        self,
        start_time_perf: float,
        prompt_tokens: int = 0,
        gen_tokens: int = 0,
        ttft_ms: Optional[float] = None,
        prompt_tps: Optional[float] = None,
        gen_tps: Optional[float] = None,
        reasoning_tokens: int = 0,
    ) -> None:
        latency_ms = (time.perf_counter() - start_time_perf) * 1000.0
        with self._lock:
            self.successful_requests += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_tokens_generated += gen_tokens
            self.total_reasoning_tokens += reasoning_tokens
            
            self.latencies_ms.append(latency_ms)
            if len(self.latencies_ms) > 1000:
                self.latencies_ms.pop(0)

            if ttft_ms is not None:
                self.ttft_ms.append(ttft_ms)
                if len(self.ttft_ms) > 1000:
                    self.ttft_ms.pop(0)

            if gen_tps is not None and gen_tps > 0:
                self.generation_tps_list.append(gen_tps)
                if len(self.generation_tps_list) > 1000:
                    self.generation_tps_list.pop(0)

            if prompt_tps is not None and prompt_tps > 0:
                self.prompt_tps_list.append(prompt_tps)
                if len(self.prompt_tps_list) > 1000:
                    self.prompt_tps_list.pop(0)

    def record_request_error(self, error_type: str) -> None:
        with self._lock:
            self.failed_requests += 1
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_metrics_summary(self, active_sessions: int = 0, loaded_models: int = 0) -> Dict[str, Any]:
        """Get structured metrics dictionary."""
        with self._lock:
            avg_latency = (sum(self.latencies_ms) / len(self.latencies_ms)) if self.latencies_ms else 0.0
            avg_ttft = (sum(self.ttft_ms) / len(self.ttft_ms)) if self.ttft_ms else 0.0
            avg_gen_tps = (sum(self.generation_tps_list) / len(self.generation_tps_list)) if self.generation_tps_list else 0.0
            avg_prompt_tps = (sum(self.prompt_tps_list) / len(self.prompt_tps_list)) if self.prompt_tps_list else 0.0

            sys_res = profiler.get_system_resources()
            res_dict = sys_res.to_dict() if hasattr(sys_res, "to_dict") else {}
            
            cpu_util = psutil.cpu_percent()
            ram_mem = psutil.virtual_memory()

            gpu_util = 0.0
            gpus = res_dict.get("gpus", [])
            if gpus and isinstance(gpus, list):
                gpu_util = gpus[0].get("utilization_pct", 0.0)

            return {
                "uptime_seconds": round(time.time() - self.server_start_time, 2),
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "active_sessions": active_sessions,
                "loaded_models": loaded_models,
                "total_tokens_generated": self.total_tokens_generated,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_reasoning_tokens": self.total_reasoning_tokens,
                "avg_latency_ms": round(avg_latency, 2),
                "avg_ttft_ms": round(avg_ttft, 2),
                "avg_generation_tps": round(avg_gen_tps, 2),
                "avg_prompt_tps": round(avg_prompt_tps, 2),
                "cpu_utilization_pct": cpu_util,
                "ram_used_gb": round(ram_mem.used / (1024 ** 3), 2),
                "ram_total_gb": round(ram_mem.total / (1024 ** 3), 2),
                "gpu_utilization_pct": gpu_util,
                "errors": dict(self.error_counts),
            }

    def format_prometheus_metrics(self, active_sessions: int = 0, loaded_models: int = 0) -> str:
        """Format metrics as Prometheus plain text format."""
        summary = self.get_metrics_summary(active_sessions=active_sessions, loaded_models=loaded_models)
        lines = [
            "# HELP inferenceos_uptime_seconds Server uptime in seconds.",
            "# TYPE inferenceos_uptime_seconds gauge",
            f"inferenceos_uptime_seconds {summary['uptime_seconds']}",
            "# HELP inferenceos_requests_total Total number of HTTP requests.",
            "# TYPE inferenceos_requests_total counter",
            f"inferenceos_requests_total {summary['total_requests']}",
            "# HELP inferenceos_requests_success_total Total successful requests.",
            "# TYPE inferenceos_requests_success_total counter",
            f"inferenceos_requests_success_total {summary['successful_requests']}",
            "# HELP inferenceos_requests_failed_total Total failed requests.",
            "# TYPE inferenceos_requests_failed_total counter",
            f"inferenceos_requests_failed_total {summary['failed_requests']}",
            "# HELP inferenceos_active_sessions Active session count.",
            "# TYPE inferenceos_active_sessions gauge",
            f"inferenceos_active_sessions {summary['active_sessions']}",
            "# HELP inferenceos_loaded_models Loaded models count.",
            "# TYPE inferenceos_loaded_models gauge",
            f"inferenceos_loaded_models {summary['loaded_models']}",
            "# HELP inferenceos_tokens_generated_total Total generated tokens.",
            "# TYPE inferenceos_tokens_generated_total counter",
            f"inferenceos_tokens_generated_total {summary['total_tokens_generated']}",
            "# HELP inferenceos_avg_generation_tps Average generation throughput.",
            "# TYPE inferenceos_avg_generation_tps gauge",
            f"inferenceos_avg_generation_tps {summary['avg_generation_tps']}",
            "# HELP inferenceos_cpu_utilization_percent Current CPU utilization percent.",
            "# TYPE inferenceos_cpu_utilization_percent gauge",
            f"inferenceos_cpu_utilization_percent {summary['cpu_utilization_pct']}",
            "# HELP inferenceos_gpu_utilization_percent Current GPU utilization percent.",
            "# TYPE inferenceos_gpu_utilization_percent gauge",
            f"inferenceos_gpu_utilization_percent {summary['gpu_utilization_pct']}",
        ]
        return "\n".join(lines) + "\n"


_telemetry_instance: Optional[ServerTelemetryCollector] = None


def get_telemetry_collector() -> ServerTelemetryCollector:
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = ServerTelemetryCollector()
    return _telemetry_instance
