"""
In-process session memory for multi-turn chat (used by agent graph + API).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def append_turn(self, session_id: str, user: str, assistant: str) -> None:
        sid = (session_id or "default").strip() or "default"
        self._sessions[sid].append({"user": user, "assistant": assistant})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        sid = (session_id or "default").strip() or "default"
        return list(self._sessions.get(sid, []))

    def clear_session(self, session_id: str) -> None:
        sid = (session_id or "default").strip() or "default"
        self._sessions.pop(sid, None)


SESSION_STORE = SessionStore()
