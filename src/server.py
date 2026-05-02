import time
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .mvp_agent import MVPAgent


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    variant: str = Field(default="v2", pattern="^(v0|v1|v2|v3|v4)$")
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    route: str
    retrieved_ids: List[str]
    latency_ms: float
    tool_calls: int
    llm_used: bool
    llm_available: bool
    history_len: int
    retrieved_image_sources: List[str]
    reasoning_trace: List[str]


class ResetRequest(BaseModel):
    session_id: str


ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="A3 Personalised Multimodal Chatbot API", version="0.1.0")
_AGENT: Optional[MVPAgent] = None
_AGENT_LOCK = threading.Lock()

# Minimal in-memory conversation store for follow-up context.
SESSION_STORE: Dict[str, List[Dict[str, str]]] = {}


def get_agent() -> MVPAgent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    with _AGENT_LOCK:
        if _AGENT is None:
            _AGENT = MVPAgent(ROOT / "data")
    return _AGENT


def _build_contextual_message(message: str, session_id: str) -> str:
    history = SESSION_STORE.get(session_id, [])
    if not history:
        return message
    recent_turns = history[-4:]
    serialized = " ".join([f"User: {t['user']} Assistant: {t['assistant']}" for t in recent_turns])
    return f"Conversation context: {serialized}\nCurrent query: {message}"


@app.get("/health")
def health():
    agent = _AGENT
    if agent is None:
        return {
            "status": "ok",
            "agent_ready": False,
            "llm_available": False,
            "doc_count": 0,
            "index_stats": {},
            "timestamp": time.time(),
        }
    return {
        "status": "ok",
        "agent_ready": True,
        "llm_available": agent.llm_available,
        "doc_count": len(agent.docs),
        "index_stats": agent.index_stats,
        "timestamp": time.time(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    agent = get_agent()
    session_id = req.session_id or str(uuid.uuid4())
    contextual_message = _build_contextual_message(req.message, session_id)
    out = agent.ask(contextual_message, variant=req.variant)
    retrieved_image_sources = []
    for item in out.get("retrieved_items", []):
        md = item.get("metadata", {})
        if md.get("modality") == "image":
            src = md.get("source_image") or md.get("source")
            if src:
                retrieved_image_sources.append(src)

    reasoning_trace = [
        f"variant={req.variant}",
        f"route={out.get('route', 'none')}",
        f"retriever_backend={out.get('retriever_backend', 'none')}",
        f"active_filters={out.get('active_filters', {})}",
        f"retrieval_note={out.get('retrieval_note', '')}",
        f"expanded_query={out.get('expanded_query', req.message)}",
        f"retrieved_topk={out.get('retrieved_ids', [])[:5]}",
        f"llm_available={bool(out.get('llm_available', False))}",
        f"llm_used={bool(out.get('llm_used', False))}",
        f"alignment_ready={bool(out.get('alignment_ready', False))}",
        f"clip_ready={bool(out.get('clip_ready', False))}",
    ]

    SESSION_STORE.setdefault(session_id, []).append({"user": req.message, "assistant": out["answer"]})
    return ChatResponse(
        session_id=session_id,
        answer=out["answer"],
        route=out["route"],
        retrieved_ids=out["retrieved_ids"],
        latency_ms=out["latency_ms"],
        tool_calls=out["tool_calls"],
        llm_used=bool(out.get("llm_used", False)),
        llm_available=bool(out.get("llm_available", False)),
        history_len=len(SESSION_STORE[session_id]),
        retrieved_image_sources=retrieved_image_sources,
        reasoning_trace=reasoning_trace,
    )


@app.post("/reset")
def reset(req: ResetRequest):
    SESSION_STORE.pop(req.session_id, None)
    return {"ok": True, "session_id": req.session_id}
