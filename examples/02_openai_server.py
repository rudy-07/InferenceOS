"""
02_openai_server.py
-------------------
Example script demonstrating how to launch and query the InferenceOS OpenAI API server.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("InferenceOS OpenAI & Ollama REST API Server Launcher")

    print("Configured server at http://127.0.0.1:11434")
    print("Endpoints enabled:")
    print("  - POST http://localhost:11434/v1/chat/completions (OpenAI Chat)")
    print("  - POST http://localhost:11434/v1/completions      (OpenAI Legacy)")
    print("  - POST http://localhost:11434/api/generate        (Ollama Generate)")
    print("  - POST http://localhost:11434/api/chat            (Ollama Chat)")

    # Server runner import
    try:
        from server.server_runner import ServerConfig, run_server
        print("Server package loaded successfully.")
    except ImportError as e:
        print(f"Notice: FastAPI / uvicorn optional server dependency: {e}")


if __name__ == "__main__":
    main()
