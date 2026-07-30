"""
chat.py
-------
Command handler for 'inferenceos chat [model]'.
"""
from __future__ import annotations
from typing import Optional
from cli.tui.chat_ui import ChatInterface


def handle_chat_command(model_query: Optional[str] = None, theme: str = "nord") -> None:
    """Launch interactive chat interface."""
    chat_ui = ChatInterface(model_query=model_query, theme_name=theme)
    chat_ui.run()
