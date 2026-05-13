from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from .session_store import SESSION_STORE
except ImportError:
    from session_store import SESSION_STORE


class MemoryTool:
    def read_session(self, session_id: str, limit: int = 12) -> str:
        hist = SESSION_STORE.get_history(session_id or "default")
        if not hist:
            return ""
        lines: List[str] = []
        for turn in hist[-limit:]:
            u = (turn.get("user") or "").strip()
            a = (turn.get("assistant") or "").strip()
            if u:
                lines.append(f"User: {u}")
            if a:
                lines.append(f"Assistant: {a}")
        return "\n".join(lines).strip()


class RetrieverTool:
    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    def query_all(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        items = self._retriever.query_hybrid(query, top_k=top_k, filters=filters or {})
        out: List[Dict[str, Any]] = []
        for it in items:
            out.append(
                {
                    "id": it.id,
                    "score": float(it.score),
                    "document": it.document,
                    "metadata": dict(it.metadata or {}),
                }
            )
        return out


def build_tools(retriever: Any) -> Dict[str, Any]:
    return {"memory": MemoryTool(), "retriever": RetrieverTool(retriever)}
