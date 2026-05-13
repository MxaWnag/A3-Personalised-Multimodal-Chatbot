import json
import logging
import time
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .mvp_agent import MVPAgent
from .session_store import SESSION_STORE


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    variant: str = Field(default="v2", pattern="^(v0|v1|v2|v3|v4)$")
    session_id: Optional[str] = None
    align_recovery: bool = False

    @field_validator("variant", mode="before")
    @classmethod
    def _lower_variant(cls, v: object) -> str:
        return str(v or "v2").strip().lower()


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
    stop_reason: Optional[str] = None
    verify_pass: Optional[bool] = None


class ResetRequest(BaseModel):
    session_id: str


ROOT = Path(__file__).resolve().parents[1]
_AGENT: Optional[MVPAgent] = None
_AGENT_LOCK = threading.Lock()
_LOG = logging.getLogger("a3.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index + agent before accepting traffic (first run can take several minutes)."""
    get_agent()
    yield


app = FastAPI(
    title="A3 Personalised Multimodal Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)


def get_agent() -> MVPAgent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    with _AGENT_LOCK:
        if _AGENT is None:
            _AGENT = MVPAgent(ROOT / "data")
    return _AGENT


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
    variant = req.variant
    align = bool(req.align_recovery) and variant == "v4"
    try:
        out = agent.ask(req.message, variant=variant, session_id=session_id, align_recovery=align)
    except Exception:
        _LOG.exception("chat_failed session_id=%s variant=%s", session_id, variant)
        raise HTTPException(status_code=500, detail="chat_failed") from None

    retrieved_image_sources: List[str] = []
    for item in out.get("retrieved_items", []) or []:
        md = item.get("metadata") or {}
        if md.get("modality") == "image":
            src = md.get("source_image") or md.get("source")
            if src:
                retrieved_image_sources.append(str(src))

    reasoning_trace: List[str] = []
    for step in out.get("tool_trace") or []:
        reasoning_trace.append(json.dumps(step, ensure_ascii=False))
    reasoning_trace.append(f"stop_reason={out.get('stop_reason', '')}")
    reasoning_trace.append(f"verify_pass={out.get('verify_pass')}")

    hist = SESSION_STORE.get_history(session_id)
    return ChatResponse(
        session_id=session_id,
        answer=out["answer"],
        route=str(out.get("route", "none")),
        retrieved_ids=out.get("retrieved_ids", []) or [],
        latency_ms=float(out["latency_ms"]),
        tool_calls=int(out.get("tool_calls", 0)),
        llm_used=bool(out.get("llm_used", False)),
        llm_available=bool(out.get("llm_available", False)),
        history_len=len(hist),
        retrieved_image_sources=retrieved_image_sources,
        reasoning_trace=reasoning_trace,
        stop_reason=out.get("stop_reason"),
        verify_pass=out.get("verify_pass"),
    )


@app.post("/reset")
def reset(req: ResetRequest):
    SESSION_STORE.clear_session(req.session_id)
    return {"ok": True, "session_id": req.session_id}
