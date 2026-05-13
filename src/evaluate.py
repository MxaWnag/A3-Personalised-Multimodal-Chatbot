import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

try:
    from .benchmark_session import benchmark_session
    from .mvp_agent import MVPAgent
    from .session_store import SESSION_STORE
except ImportError:
    from benchmark_session import benchmark_session
    from mvp_agent import MVPAgent
    from session_store import SESSION_STORE


BENCHMARK_LEGACY: List[Dict[str, Any]] = [
    {"id": "f1", "family": "factual", "query": "What is earned value (EV) in project control?", "gold_sources": ["ENGG4800_Lecture_8_Week_8_S1_2026.pdf"], "keywords": ["earned value", "budgeted cost", "work performed"]},
    {"id": "f2", "family": "factual", "query": "What is the baseline plan used for in project monitoring?", "gold_sources": ["ENGG4800_Lecture_8_Week_8_S1_2026.pdf"], "keywords": ["baseline", "measuring performance", "compare"]},
    {"id": "f3", "family": "factual", "query": "What are control charts used to monitor?", "gold_sources": ["ENGG4800_Lecture_8_Week_8_S1_2026.pdf"], "keywords": ["control chart", "milestone", "monitor"]},
    {"id": "f4", "family": "factual", "query": "In Applied Class 7, what is a risk register?", "gold_sources": ["ENGG4800_Applied Class 7.pdf"], "keywords": ["risk register", "risk", "information"]},
    {"id": "c1", "family": "cross_modal", "query": "Which image document is about a multimodal RAG pipeline diagram?", "gold_docs": ["i_infs7205_w08_001"], "keywords": ["pipeline", "multimodal", "rag"]},
    {"id": "c2", "family": "cross_modal", "query": "Find the image related to high-dimensional indexing structure comparison.", "gold_docs": ["i_infs7205_w06_001"], "keywords": ["indexing", "high-dimensional", "vector"]},
    {"id": "c3", "family": "cross_modal", "query": "Which figure shows route, memory, retrieve, and answer nodes?", "gold_docs": ["i_infs7205_w08_001"], "keywords": ["route", "memory", "retrieve", "answer"]},
    {"id": "c4", "family": "cross_modal", "query": "Locate the practical class visual workflow example from ENGG4800.", "gold_docs": ["i_engg4800_w08_001"], "keywords": ["practical", "workflow", "engg4800"]},
    {"id": "m1", "family": "multi_hop", "query": "Summarize how baseline planning and earned value work together for control.", "gold_sources": ["ENGG4800_Lecture_8_Week_8_S1_2026.pdf"], "keywords": ["baseline", "earned value", "control"]},
    {"id": "m2", "family": "multi_hop", "query": "Compare tracking Gantt charts and control charts for monitoring progress.", "gold_sources": ["ENGG4800_Lecture_8_Week_8_S1_2026.pdf"], "keywords": ["gantt", "control chart", "monitoring"]},
    {"id": "m3", "family": "multi_hop", "query": "Connect risk management concepts in Applied Class 7 with project control needs.", "gold_sources": ["ENGG4800_Applied Class 7.pdf", "ENGG4800_Lecture_8_Week_8_S1_2026.pdf"], "keywords": ["risk", "control", "project"]},
    {"id": "m4", "family": "multi_hop", "query": "What trade-offs are discussed around schedule tracking and budget performance?", "gold_sources": ["ENGG4800_Lecture_8_Week_8_S1_2026.pdf"], "keywords": ["schedule", "budget", "performance"]},
    {"id": "p1", "family": "follow_up", "query": "Based on previous answer, give me a concise revision tip for weak topic evaluation design.", "gold_sources": ["INFS7205_Lecture8_Multimodal RAG.pdf", "INFS7205_Lecture6_High Dimensional Indexing-v1.pdf"], "keywords": ["evaluation", "retrieval", "metric"]},
    {"id": "p2", "family": "follow_up", "query": "I have only 2 hours and prefer concise summaries. What should I revise first?", "gold_sources": ["ENGG4800_Lecture_8_Week_8_S1_2026.pdf"], "keywords": ["concise", "prioritize", "revision"]},
    {"id": "p3", "family": "follow_up", "query": "Recommend a study sequence combining indexing and multimodal RAG topics.", "gold_sources": ["INFS7205_Lecture6_High Dimensional Indexing-v1.pdf", "INFS7205_Lecture8_Multimodal RAG.pdf"], "keywords": ["indexing", "multimodal", "sequence"]},
    {"id": "p4", "family": "follow_up", "query": "Given my style, rewrite the explanation into short bullet points.", "gold_sources": ["INFS7205_Lecture8_Multimodal RAG.pdf"], "keywords": ["bullet", "concise", "style"]},
]


def load_benchmark(root: Path) -> List[Dict[str, Any]]:
    path = root / "data" / "benchmark_course_assistant.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return list(BENCHMARK_LEGACY)


def _metadata_by_id(retrieved_items: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in retrieved_items or []:
        rid = it.get("id")
        if rid:
            out[str(rid)] = dict(it.get("metadata") or {})
    return out


def _canonical_doc_ids(rid: str, metadata: Dict[str, Any], agent: MVPAgent) -> set:
    ids = {str(rid)}
    for key in ("base_id", "related_text_id"):
        val = metadata.get(key)
        if val:
            ids.add(str(val))
    srid = str(rid)
    if "::" in srid:
        ids.add(srid.split("::", 1)[0])
    for cid in list(ids):
        doc = agent.doc_map.get(cid, {})
        if doc.get("id"):
            ids.add(str(doc["id"]))
        sf = doc.get("source_file")
        if sf:
            ids.add(str(sf))
    return ids


def recall_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    agent: Optional[MVPAgent] = None,
    retrieved_items: Optional[List[Dict[str, Any]]] = None,
) -> float:
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    top = retrieved_ids[:5]
    if agent is None:
        return len(set(top).intersection(gold)) / max(1, len(gold))
    meta_by_id = _metadata_by_id(retrieved_items)
    hits = 0
    for gid in gold:
        for rid in top:
            canon = _canonical_doc_ids(str(rid), meta_by_id.get(str(rid), {}), agent)
            if gid in canon:
                hits += 1
                break
    return hits / max(1, len(gold))


def recall_at_k_source(
    retrieved_ids: List[str],
    gold_sources: List[str],
    agent: MVPAgent,
    retrieved_items: Optional[List[Dict[str, Any]]] = None,
) -> float:
    if not gold_sources:
        return 0.0
    want = set(gold_sources)
    got: set = set()
    meta_by_id = _metadata_by_id(retrieved_items)
    for rid in retrieved_ids:
        doc = agent.doc_map.get(rid, {})
        src = doc.get("source_file")
        meta = meta_by_id.get(str(rid), {})
        src = src or meta.get("source_file") or meta.get("source") or meta.get("source_pdf")
        if src in want:
            got.add(str(src))
    return len(got) / max(1, len(want))


def task_success(answer: str, keywords: List[str]) -> float:
    ans = answer.lower()
    if not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k.lower() in ans)
    return hit / len(keywords)


def evidence_consistency(answer: str, retrieved_ids: List[str]) -> float:
    if not retrieved_ids:
        return 0.0
    ans = answer.lower()
    cited = sum(1 for rid in retrieved_ids[:3] if f"[{str(rid).lower()}]" in ans)
    return cited / min(3, len(retrieved_ids))


def reciprocal_rank(
    retrieved_ids: List[str],
    gold_doc_ids: List[str],
    agent: Optional[MVPAgent] = None,
    retrieved_items: Optional[List[Dict[str, Any]]] = None,
) -> float:
    if not gold_doc_ids:
        return 0.0
    gold = set(gold_doc_ids)
    if agent is None:
        for i, rid in enumerate(retrieved_ids, start=1):
            if rid in gold:
                return 1.0 / i
        return 0.0
    meta_by_id = _metadata_by_id(retrieved_items)
    for i, rid in enumerate(retrieved_ids, start=1):
        canon = _canonical_doc_ids(str(rid), meta_by_id.get(str(rid), {}), agent)
        if gold.intersection(canon):
            return 1.0 / i
    return 0.0


def reciprocal_rank_source(retrieved_ids: List[str], gold_sources: List[str], agent: MVPAgent) -> float:
    if not gold_sources:
        return 0.0
    gold = set(gold_sources)
    for i, rid in enumerate(retrieved_ids, start=1):
        doc = agent.doc_map.get(rid, {})
        src = doc.get("source_file")
        if src in gold:
            return 1.0 / i
    return 0.0


def token_usage_proxy(text: str) -> int:
    return len(text.split())


def _replay_prior_turns(item: Dict[str, Any], session_id: str) -> None:
    SESSION_STORE.clear_session(session_id)
    prior = item.get("prior_turn") or []
    if isinstance(prior, list):
        for t in prior:
            SESSION_STORE.append_turn(session_id, str(t.get("user", "")), str(t.get("assistant", "")))


def evaluate_variant(agent: MVPAgent, variant: str, benchmark: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    vf = (variant or "v2").lower()
    for item in benchmark:
        sid = f"eval::{item.get('id', 'x')}"
        msg = str(item.get("query", ""))
        if vf == "v0":
            pre = benchmark_session(item)
            if pre:
                msg = f"{pre}\n\nCurrent question: {item.get('query', '')}"
        else:
            _replay_prior_turns(item, sid)
        align = vf == "v4"
        out = agent.ask(msg, variant=vf, session_id=sid, align_recovery=align)

        retrieved_ids = list(out.get("retrieved_ids") or [])
        retrieved_items = list(out.get("retrieved_items") or [])

        if vf == "v0":
            r5 = 0.0
            rr = 0.0
        elif "gold_docs" in item:
            r5 = recall_at_k(retrieved_ids, item["gold_docs"], agent, retrieved_items)
            rr = reciprocal_rank(retrieved_ids, item["gold_docs"], agent, retrieved_items)
        else:
            r5 = recall_at_k_source(retrieved_ids, item.get("gold_sources", []) or [], agent, retrieved_items)
            rr = reciprocal_rank_source(retrieved_ids, item.get("gold_sources", []) or [], agent)

        kw = item.get("keywords") or []
        keyword_score = task_success(out["answer"], kw if isinstance(kw, list) else [])
        ev_score = evidence_consistency(out["answer"], retrieved_ids)
        ts = 0.7 * keyword_score + 0.3 * ev_score

        rows.append(
            {
                "id": item.get("id", ""),
                "family": item.get("family", ""),
                "recall5": r5,
                "mrr": rr,
                "keyword_score": keyword_score,
                "evidence_score": ev_score,
                "success": ts,
                "latency_ms": out["latency_ms"],
                "tool_calls": out["tool_calls"],
                "token_usage_proxy": token_usage_proxy(out["answer"]),
                "llm_used": int(bool(out.get("llm_used", False))),
                "llm_available": int(bool(out.get("llm_available", False))),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fam: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        fam.setdefault(str(r["family"]), []).append(r)
    summary = {
        "overall": {
            "Recall@5": mean(r["recall5"] for r in rows),
            "MRR": mean(r["mrr"] for r in rows),
            "KeywordScore": mean(r["keyword_score"] for r in rows),
            "EvidenceScore": mean(r["evidence_score"] for r in rows),
            "TaskSuccess": mean(r["success"] for r in rows),
            "LatencyMs": mean(r["latency_ms"] for r in rows),
            "ToolCalls": mean(r["tool_calls"] for r in rows),
            "TokenUsageProxy": mean(r["token_usage_proxy"] for r in rows),
            "LLMUsedRate": mean(r["llm_used"] for r in rows),
            "LLMAvailable": mean(r["llm_available"] for r in rows),
        },
        "by_family": {},
    }
    for k, vs in fam.items():
        summary["by_family"][k] = {
            "Recall@5": mean(r["recall5"] for r in vs),
            "MRR": mean(r["mrr"] for r in vs),
            "KeywordScore": mean(r["keyword_score"] for r in vs),
            "EvidenceScore": mean(r["evidence_score"] for r in vs),
            "TaskSuccess": mean(r["success"] for r in vs),
            "LatencyMs": mean(r["latency_ms"] for r in vs),
            "ToolCalls": mean(r["tool_calls"] for r in vs),
            "TokenUsageProxy": mean(r["token_usage_proxy"] for r in vs),
            "LLMUsedRate": mean(r["llm_used"] for r in vs),
            "LLMAvailable": mean(r["llm_available"] for r in vs),
        }
    return summary


def print_summary(name: str, summary: Dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    o = summary["overall"]
    print(
        (
            "Overall | Recall@5={:.3f} | MRR={:.3f} | TaskSuccess={:.3f} | "
            "Keyword={:.3f} | Evidence={:.3f} | LatencyMs={:.2f} | ToolCalls={:.2f} | TokenProxy={:.1f} | "
            "LLMUsedRate={:.2f} | LLMAvailable={:.2f}"
        ).format(
            o["Recall@5"],
            o["MRR"],
            o["TaskSuccess"],
            o["KeywordScore"],
            o["EvidenceScore"],
            o["LatencyMs"],
            o["ToolCalls"],
            o["TokenUsageProxy"],
            o["LLMUsedRate"],
            o["LLMAvailable"],
        )
    )
    for fam, vals in summary["by_family"].items():
        print(
            (
                "  - {} | Recall@5={:.3f} | MRR={:.3f} | TaskSuccess={:.3f} | "
                "Keyword={:.3f} | Evidence={:.3f} | LatencyMs={:.2f} | ToolCalls={:.2f} | TokenProxy={:.1f} | "
                "LLMUsedRate={:.2f} | LLMAvailable={:.2f}"
            ).format(
                fam,
                vals["Recall@5"],
                vals["MRR"],
                vals["TaskSuccess"],
                vals["KeywordScore"],
                vals["EvidenceScore"],
                vals["LatencyMs"],
                vals["ToolCalls"],
                vals["TokenUsageProxy"],
                vals["LLMUsedRate"],
                vals["LLMAvailable"],
            )
        )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "family",
        "recall5",
        "mrr",
        "keyword_score",
        "evidence_score",
        "success",
        "latency_ms",
        "tool_calls",
        "token_usage_proxy",
        "llm_used",
        "llm_available",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    agent = MVPAgent(root / "data")
    benchmark = load_benchmark(root)
    variants = [
        ("V0_plain_llm", "v0"),
        ("V1_rag_no_memory", "v1"),
        ("V2_agent_router_memory", "v2"),
        ("V3_agent_router_memory_aligned", "v3"),
        ("V4_agent_router_memory_clip", "v4"),
    ]
    print(f"LLM health check | available={agent.llm_available} | model={agent.llm.model}")
    print(f"Benchmark items: {len(benchmark)}")
    for name, vid in variants:
        rows = evaluate_variant(agent, vid, benchmark)
        summary = summarize(rows)
        print_summary(name, summary)
        write_csv(root / "results" / f"{name}.csv", rows)
    print("\nSaved per-query results to results/*.csv")


if __name__ == "__main__":
    main()
