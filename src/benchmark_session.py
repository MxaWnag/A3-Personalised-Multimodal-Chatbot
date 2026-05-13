"""
Build prior-turn context for benchmark items (follow-up questions).
"""

from __future__ import annotations

from typing import Any, Dict, List


def _format_history(turns: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for t in turns:
        u = (t.get("user") or "").strip()
        a = (t.get("assistant") or "").strip()
        if u:
            parts.append(f"User: {u}")
        if a:
            parts.append(f"Assistant: {a}")
    return "\n".join(parts).strip()


def benchmark_session(item: Dict[str, Any]) -> str:
    prior = item.get("prior_turn") or []
    if isinstance(prior, list) and prior:
        return _format_history(prior)
    return ""
