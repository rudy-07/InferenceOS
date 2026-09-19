"""
session_manager.py
-------------------
Thread-safe Session Manager supporting LRU eviction, idle expiration,
and disk persistence for InferenceOS Server.
"""
from __future__ import annotations

import json
import time
import uuid
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from server.config import ServerConfig
from server.sessions.session import Session
from server.models.openai import ChatMessage


class SessionManager:
    """
    Manages active sessions, LRU cleanup, and session persistence.
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self.config = config or ServerConfig()
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        
        self.persistence_dir = Path.home() / ".inferenceos" / "sessions"
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        model: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        initial_messages: Optional[List[ChatMessage]] = None,
    ) -> Session:
        """Create and register a new session."""
        with self._lock:
            self.cleanup_expired()
            
            sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
            if sid in self._sessions:
                # Re-use existing if explicit ID passed
                session = self._sessions[sid]
                session.touch()
                return session

            # LRU eviction if max sessions reached
            if len(self._sessions) >= self.config.max_sessions:
                self._evict_lru()

            messages: List[ChatMessage] = []
            if system_prompt:
                messages.append(ChatMessage(role="system", content=system_prompt))
            if initial_messages:
                messages.extend(initial_messages)

            session = Session(
                session_id=sid,
                model=model,
                messages=messages,
                config_overrides=config_overrides or {},
            )
            self._sessions[sid] = session
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if self._is_expired(session):
                    del self._sessions[session_id]
                    return None
                session.touch()
            return session

    def list_sessions(self) -> List[Session]:
        """List all active unexpired sessions."""
        with self._lock:
            self.cleanup_expired()
            return sorted(self._sessions.values(), key=lambda s: s.last_access_time, reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Destroy a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def reset_session(self, session_id: str, system_prompt: Optional[str] = None) -> Optional[Session]:
        """Clear conversation history for a session."""
        session = self.get_session(session_id)
        if session:
            session.reset_history(system_prompt=system_prompt)
        return session

    def save_session(self, session_id: str, file_path: Optional[str] = None) -> Optional[Path]:
        """Persist session state to disk."""
        session = self.get_session(session_id)
        if not session:
            return None

        target_path = Path(file_path) if file_path else (self.persistence_dir / f"{session_id}.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
            return target_path.resolve()
        except Exception as e:
            print(f"[Error] Failed to save session {session_id}: {e}")
            return None

    def load_session(self, file_path_or_id: str) -> Optional[Session]:
        """Restore session from disk."""
        path = Path(file_path_or_id)
        if not path.exists():
            path = self.persistence_dir / f"{file_path_or_id}.json"
        
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            session = Session.from_dict(data)
            with self._lock:
                self._sessions[session.session_id] = session
            return session
        except Exception as e:
            print(f"[Error] Failed to load session from {path}: {e}")
            return None

    def cleanup_expired(self) -> int:
        """Remove idle or expired sessions."""
        now = time.time()
        expired_ids = [
            sid for sid, sess in self._sessions.items()
            if self._is_expired(sess, now=now)
        ]
        for sid in expired_ids:
            del self._sessions[sid]
        return len(expired_ids)

    def _is_expired(self, session: Session, now: Optional[float] = None) -> bool:
        n = now or time.time()
        idle = (n - session.last_access_time) > self.config.idle_timeout_sec
        age = (n - session.creation_time) > self.config.session_timeout_sec
        return idle or age

    def _evict_lru(self) -> None:
        """Evict the least recently used session."""
        if not self._sessions:
            return
        lru_sid = min(self._sessions.keys(), key=lambda k: self._sessions[k].last_access_time)
        del self._sessions[lru_sid]


_session_manager_instance: Optional[SessionManager] = None


def get_session_manager(config: Optional[ServerConfig] = None) -> SessionManager:
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager(config=config)
    return _session_manager_instance
