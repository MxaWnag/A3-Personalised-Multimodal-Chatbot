from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict, total=False):
    session_id: str
    user_message: str
    variant: str
    align_recovery: bool
    conversation_summary: str
    plan_json: Dict[str, Any]
    retrieved_items: List[Dict[str, Any]]
    evidence_text: str
    sources: str
    answer_text: str
    verify_json: Dict[str, Any]
    tool_trace: List[Dict[str, Any]]
    stop_reason: str
    filters: Dict[str, Any]
    top_k: int
    route: str
    rewrite_count: int
    retrieve_recovery_count: int


def compile_langgraph(agent: Any) -> Any:
    """
    V4 (align_recovery=True): matches report diagram —
    filters → plan → retrieve → answer → verify,
    then either finalize, or rewrite→verify loop (citation), or retrieve_recovery→answer→verify (retrieval),
    with per-path retry caps (exhausted → finalize).
    v1–v3 or align_recovery=False: verify fail → finalize (no loops).
    """

    max_rewrite = max(0, int(os.getenv("V4_MAX_REWRITE_ATTEMPTS", "2")))
    max_retrieve_recovery = max(0, int(os.getenv("V4_MAX_RETRIEVE_RECOVERY", "2")))

    def prepare(state: AgentState) -> AgentState:
        user_message = state["user_message"]
        session_id = state.get("session_id") or "default"
        agent._prepare_filters(user_message)
        filters = dict(agent.current_filters)
        summary = agent.tools["memory"].read_session(session_id)
        trace = list(state.get("tool_trace", []))
        trace.append({"node": "prepare", "filters": filters, "session_chars": len(summary)})
        return {
            "conversation_summary": summary,
            "tool_trace": trace,
            "filters": filters,
            "top_k": 5,
            "rewrite_count": int(state.get("rewrite_count") or 0),
            "retrieve_recovery_count": int(state.get("retrieve_recovery_count") or 0),
        }

    def plan(state: AgentState) -> AgentState:
        variant = (state.get("variant") or "v2").lower()
        plan_notes_hint = " Prefer hybrid multimodal retrieval (text + image/caption) for this turn." if variant == "v4" else ""
        plan_out = agent.planner.plan(
            user_message=state["user_message"],
            conversation_summary=state.get("conversation_summary", ""),
            course_context=agent._course_context() + plan_notes_hint,
        )
        trace = list(state.get("tool_trace", []))
        trace.append(
            {
                "node": "plan",
                "route": "hybrid" if variant == "v4" else "router",
                "retrieve_queries": plan_out.get("retrieve_queries", []),
            }
        )
        return {"plan_json": plan_out, "tool_trace": trace}

    def retrieve(state: AgentState) -> AgentState:
        plan_out = state.get("plan_json") or {}
        queries = plan_out.get("retrieve_queries") or [state["user_message"]]
        variant = (state.get("variant") or "v2").lower()
        filters = state.get("filters") or {}
        top_k = int(state.get("top_k", 5))
        items: List[Dict[str, Any]] = []
        for q in queries:
            items.extend(agent._retrieve_items_as_dicts(str(q), variant, filters, top_k=top_k))
        items = agent._dedupe_items(items)
        evidence = agent._format_evidence(items)
        sources = ",".join([str(it["id"]) for it in items[:20]])
        trace = list(state.get("tool_trace", []))
        backend = "search_hybrid" if variant == "v4" else "search_variant"
        trace.append({"node": "retrieve", "search": backend, "n_queries": len(queries), "n_items": len(items)})
        return {
            "retrieved_items": items,
            "evidence_text": evidence,
            "sources": sources,
            "tool_trace": trace,
            "route": "langgraph",
        }

    def answer(state: AgentState) -> AgentState:
        ans = agent.composer.compose(
            conversation_summary=state.get("conversation_summary", ""),
            user_message=state["user_message"],
            retrieved_cards=state.get("evidence_text", ""),
            sources=state.get("sources", ""),
        )
        trace = list(state.get("tool_trace", []))
        trace.append({"node": "answer", "chars": len(ans)})
        return {"answer_text": ans, "tool_trace": trace}

    def verify(state: AgentState) -> AgentState:
        v = agent.verifier.verify(
            answer=state.get("answer_text", ""),
            evidence_text=state.get("evidence_text", ""),
            user_message=state["user_message"],
            sources=state.get("sources", ""),
        )
        trace = list(state.get("tool_trace", []))
        trace.append(
            {
                "node": "verify",
                "pass": bool(v.get("pass")),
                "failure_kind": v.get("failure_kind"),
                "rewrite_allowed": v.get("rewrite_allowed"),
                "retrieval_retry_allowed": v.get("retrieval_retry_allowed"),
            }
        )
        return {"verify_json": v, "tool_trace": trace}

    def rewrite(state: AgentState) -> AgentState:
        v = state.get("verify_json") or {}
        feedback = json.dumps(v, ensure_ascii=False)[:8000]
        new_ans = agent.composer.compose_citation_rewrite(
            conversation_summary=state.get("conversation_summary", ""),
            user_message=state["user_message"],
            retrieved_cards=state.get("evidence_text", ""),
            sources=state.get("sources", ""),
            previous_answer=state.get("answer_text", ""),
            verify_feedback=feedback,
        )
        rw_c = int(state.get("rewrite_count") or 0) + 1
        trace = list(state.get("tool_trace", []))
        trace.append({"node": "rewrite", "rewrite_count": rw_c})
        return {"answer_text": new_ans, "rewrite_count": rw_c, "tool_trace": trace}

    def retrieve_recovery(state: AgentState) -> AgentState:
        v = state.get("verify_json") or {}
        sugg = v.get("suggested_followups") or []
        if not sugg:
            plan_out = state.get("plan_json") or {}
            sugg = plan_out.get("retrieve_queries") or [state["user_message"]]
        filters = state.get("filters") or {}
        variant = (state.get("variant") or "v2").lower()
        top_k = int(state.get("top_k", 5))
        new_items: List[Dict[str, Any]] = []
        for q in sugg[:3]:
            new_items.extend(agent._retrieve_items_as_dicts(str(q), variant, filters, top_k=top_k))
        old = state.get("retrieved_items") or []
        merged = agent._dedupe_items(old + new_items)
        evidence = agent._format_evidence(merged)
        sources = ",".join([str(it["id"]) for it in merged[:20]])
        rr_c = int(state.get("retrieve_recovery_count") or 0) + 1
        trace = list(state.get("tool_trace", []))
        trace.append({"node": "retrieve_recovery", "search": "search_hybrid", "retrieve_recovery_count": rr_c, "queries": sugg[:3]})
        return {
            "retrieved_items": merged,
            "evidence_text": evidence,
            "sources": sources,
            "retrieve_recovery_count": rr_c,
            "tool_trace": trace,
        }

    def finalize(state: AgentState) -> AgentState:
        v = state.get("verify_json") or {}
        variant = (state.get("variant") or "v2").lower()
        align = bool(state.get("align_recovery"))
        rw_c = int(state.get("rewrite_count") or 0)
        rr_c = int(state.get("retrieve_recovery_count") or 0)

        if v.get("pass"):
            stop = "verify_pass"
        elif variant != "v4" or not align:
            stop = "verify_fail"
        elif rw_c > 0 or rr_c > 0:
            stop = "verify_fail_after_self_correct"
        else:
            stop = "verify_fail"

        trace = list(state.get("tool_trace", []))
        trace.append(
            {
                "node": "finalize",
                "stop_reason": stop,
                "rewrite_count": rw_c,
                "retrieve_recovery_count": rr_c,
            }
        )
        return {"stop_reason": stop, "tool_trace": trace}

    def route_after_verify(state: AgentState) -> Literal["finalize", "rewrite", "retrieve_recovery"]:
        v = state.get("verify_json") or {}
        if v.get("pass"):
            return "finalize"

        variant = (state.get("variant") or "v2").lower()
        if variant != "v4" or not state.get("align_recovery"):
            return "finalize"

        fk = str(v.get("failure_kind") or "mixed").lower()
        rw_ok = bool(v.get("rewrite_allowed"))
        rr_ok = bool(v.get("retrieval_retry_allowed"))
        rw_c = int(state.get("rewrite_count") or 0)
        rr_c = int(state.get("retrieve_recovery_count") or 0)
        exhausted_rw = rw_c >= max_rewrite
        exhausted_rr = rr_c >= max_retrieve_recovery

        if exhausted_rw and exhausted_rr:
            return "finalize"

        # Citation loop: citation or mixed while rewrite budget remains
        if fk in ("citation", "mixed") and rw_ok and not exhausted_rw:
            return "rewrite"

        # Retrieval recovery: pure retrieval, or mixed after rewrite budget exhausted
        if fk == "retrieval" and rr_ok and not exhausted_rr:
            return "retrieve_recovery"
        if fk == "mixed" and exhausted_rw and rr_ok and not exhausted_rr:
            return "retrieve_recovery"

        # Citation-only failure but rewrite not allowed / exhausted → try retrieval if allowed (optional bridge)
        if fk == "citation" and rr_ok and not exhausted_rr and exhausted_rw:
            return "retrieve_recovery"

        return "finalize"

    g = StateGraph(AgentState)
    g.add_node("prepare", prepare)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("verify", verify)
    g.add_node("rewrite", rewrite)
    g.add_node("retrieve_recovery", retrieve_recovery)
    g.add_node("finalize", finalize)
    g.set_entry_point("prepare")
    g.add_edge("prepare", "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"finalize": "finalize", "rewrite": "rewrite", "retrieve_recovery": "retrieve_recovery"},
    )
    g.add_edge("rewrite", "verify")
    g.add_edge("retrieve_recovery", "answer")
    g.add_edge("finalize", END)
    return g.compile()
