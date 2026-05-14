import json
import os
import urllib.error
import urllib.request

import streamlit as st


API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
# First /chat may cold-start the agent (index + models) and Ollama can be slow on CPU/GPU.
CHAT_TIMEOUT_S = float(os.getenv("API_CHAT_TIMEOUT_S", "3600"))
HEALTH_TIMEOUT_S = float(os.getenv("STREAMLIT_HEALTH_TIMEOUT_S", "30"))


def post_json(url: str, payload: dict, timeout: float = 30.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def compact_answer_view(answer: str) -> str:
    """
    Show concise user-facing answer; detailed reasoning remains in sidebar.
    """
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    if not lines:
        return "No answer generated."
    # Prefer Direct Answer section if present.
    for i, ln in enumerate(lines):
        if ln.lower().startswith("### direct answer"):
            if i + 1 < len(lines):
                return lines[i + 1]
    # Fallback: first sentence-like line, truncated.
    first = lines[0]
    return first[:220] + ("..." if len(first) > 220 else "")


st.set_page_config(page_title="A3 Multimodal Chatbot", layout="wide")
st.title("A3 Personalised Multimodal Chatbot")
st.caption("LangGraph-style agent + hybrid retrieval + memory")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "variant" not in st.session_state:
    st.session_state.variant = "v2"
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "last_images" not in st.session_state:
    st.session_state.last_images = []

col1, col2 = st.columns([2, 1])
with col1:
    st.session_state.variant = st.selectbox("System variant", ["v0", "v1", "v2", "v3", "v4"], index=2)
with col2:
    if st.button("Reset session"):
        if st.session_state.session_id:
            try:
                post_json(f"{API_BASE}/reset", {"session_id": st.session_state.session_id})
            except Exception:
                pass
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

try:
    health = get_json(f"{API_BASE}/health", timeout=HEALTH_TIMEOUT_S)
    st.success(
        f"Server OK | docs={health['doc_count']} | llm_available={health['llm_available']}",
        icon="✅",
    )
except Exception as e:
    st.error(f"Cannot reach server at {API_BASE}. Start backend first. ({e})", icon="🚨")
    st.stop()

# Sidebar must run AFTER chat_input updates last_trace / last_images (same run), otherwise
# the first reply shows empty trace until the next interaction.

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            st.caption(msg["meta"])

if prompt := st.chat_input("Ask a course question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload: dict = {"message": prompt, "variant": st.session_state.variant}
                if st.session_state.session_id:
                    payload["session_id"] = st.session_state.session_id
                out = post_json(f"{API_BASE}/chat", payload, timeout=CHAT_TIMEOUT_S)
                st.session_state.session_id = out["session_id"]
                answer = out["answer"]
                concise = compact_answer_view(answer)
                meta = (
                    f"route={out['route']} | retrieved={out['retrieved_ids'][:5]} | "
                    f"latency_ms={out['latency_ms']:.2f} | tool_calls={out['tool_calls']} | "
                    f"llm_used={out['llm_used']} | llm_available={out['llm_available']}"
                )
                st.markdown(concise)
                with st.expander("Show full structured response"):
                    st.markdown(answer)
                st.caption(meta)
                st.session_state.last_trace = out.get("reasoning_trace", [])
                st.session_state.last_images = out.get("retrieved_image_sources", [])
                st.session_state.messages.append({"role": "assistant", "content": concise, "meta": meta})
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as e:
                if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
                    st.session_state.messages.pop()
                st.error(
                    f"Failed to call backend API: {e}. "
                    f"If this was a timeout, increase API_CHAT_TIMEOUT_S (current {CHAT_TIMEOUT_S:.0f}s) or wait for Ollama/model load."
                )

with st.sidebar:
    st.subheader("Agent Reasoning Trace")
    if st.session_state.last_trace:
        for step in st.session_state.last_trace:
            st.code(step, language="text")
    else:
        st.caption("No trace yet. Ask a question to see steps.")

    st.subheader("Retrieved Image Evidence")
    if st.session_state.last_images:
        for src in st.session_state.last_images:
            st.markdown(f"- `{src}`")
    else:
        st.caption("No image evidence retrieved yet.")
