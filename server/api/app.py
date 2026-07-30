"""
app.py
------
FastAPI application creation factory with middleware and route registration.
"""
from __future__ import annotations

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.config import ServerConfig
from server.middleware.auth import auth_middleware
from server.middleware.logging import logging_middleware
from server.middleware.error_handler import global_exception_handler
from server.queue.request_queue import get_queue_manager
from server.sessions.session_manager import get_session_manager
from server.runtime_adapter.adapter import get_runtime_adapter

from server.api.routes.health import router as health_router
from server.api.routes.models import router as models_router
from server.api.routes.chat import router as chat_router
from server.api.routes.completions import router as completions_router
from server.api.routes.embeddings import router as embeddings_router
from server.api.routes.responses import router as responses_router
from server.api.routes.sessions import router as sessions_router
from server.api.routes.metrics import router as metrics_router
from server.api.routes.ollama import router as ollama_router


from contextlib import asynccontextmanager


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Create and configure the InferenceOS FastAPI application."""
    cfg = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        get_queue_manager(cfg)
        get_session_manager(cfg)
        get_runtime_adapter(cfg)
        yield

    app = FastAPI(
        title="InferenceOS Server",
        description="Production-grade OpenAI-compatible local inference server.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # State injection
    app.state.config = cfg

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware registration
    @app.middleware("http")
    async def custom_auth_middleware(request, call_next):
        return await auth_middleware(request, call_next, cfg)

    @app.middleware("http")
    async def custom_logging_middleware(request, call_next):
        return await logging_middleware(request, call_next)

    # Exception handler
    app.add_exception_handler(Exception, global_exception_handler)

    # Route registration
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(chat_router)
    app.include_router(completions_router)
    app.include_router(embeddings_router)
    app.include_router(responses_router)
    app.include_router(sessions_router)
    app.include_router(metrics_router)
    app.include_router(ollama_router)

    return app
