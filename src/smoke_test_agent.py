"""Lightweight smoke: agent loads and V0 ask returns expected keys."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp_agent import MVPAgent  # noqa: E402


def main() -> None:
    agent = MVPAgent(ROOT / "data")
    r = agent.ask("What is a project baseline in one sentence?", variant="v0", session_id="smoke")
    required = {
        "answer",
        "variant",
        "route",
        "retrieved_ids",
        "retrieved_items",
        "tool_trace",
        "stop_reason",
        "latency_ms",
        "tool_calls",
    }
    missing = required - set(r.keys())
    assert not missing, f"missing keys: {missing}"
    assert isinstance(r.get("tool_trace"), list)
    print("smoke_ok", {"keys": sorted(r.keys()), "stop_reason": r.get("stop_reason")})


if __name__ == "__main__":
    main()
