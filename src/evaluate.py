from pathlib import Path
from statistics import mean
import csv

from mvp_agent import MVPAgent


BENCHMARK = [
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


def recall_at_k(retrieved_ids, gold_ids):
    if not gold_ids:
        return 0.0
    hit = len(set(retrieved_ids).intersection(set(gold_ids)))
    return hit / len(set(gold_ids))


def recall_at_k_source(retrieved_ids, gold_sources, agent):
    if not gold_sources:
        return 0.0
    got = set()
    for rid in retrieved_ids:
        doc = agent.doc_map.get(rid, {})
        src = doc.get("source_file")
        if src in gold_sources:
            got.add(src)
    return len(got) / len(set(gold_sources))


def task_success(answer: str, keywords):
    ans = answer.lower()
    if not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k.lower() in ans)
    return hit / len(keywords)


def evidence_consistency(answer: str, retrieved_ids):
    """
    Score whether answer references retrieved evidence IDs.
    This discourages ungrounded fluent responses.
    """
    if not retrieved_ids:
        return 0.0
    ans = answer.lower()
    cited = sum(1 for rid in retrieved_ids[:3] if f"[{rid.lower()}]" in ans)
    return cited / min(3, len(retrieved_ids))


def reciprocal_rank(retrieved_ids, gold_doc_ids):
    if not gold_doc_ids:
        return 0.0
    gold = set(gold_doc_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def token_usage_proxy(text: str):
    # Lightweight token proxy for local benchmarking.
    return len(text.split())


def evaluate_variant(agent: MVPAgent, variant: str):
    rows = []
    for item in BENCHMARK:
        out = agent.ask(item["query"], variant=variant)
        if variant == "v0":
            r5 = 0.0
            rr = 0.0
        elif "gold_docs" in item:
            r5 = recall_at_k(out["retrieved_ids"][:5], item["gold_docs"])
            rr = reciprocal_rank(out["retrieved_ids"][:5], item["gold_docs"])
        else:
            r5 = recall_at_k_source(out["retrieved_ids"][:5], item.get("gold_sources", []), agent)
            rr = 0.0

        keyword_score = task_success(out["answer"], item["keywords"])
        ev_score = evidence_consistency(out["answer"], out["retrieved_ids"])
        # Strict answer-quality score: semantics + grounding.
        ts = 0.7 * keyword_score + 0.3 * ev_score

        rows.append(
            {
                "id": item["id"],
                "family": item["family"],
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


def summarize(rows):
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)
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


def print_summary(name, summary):
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


def write_csv(path: Path, rows):
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


def main():
    root = Path(__file__).resolve().parents[1]
    agent = MVPAgent(root / "data")
    variants = [
        ("V0_plain_llm", "v0"),
        ("V1_rag_no_memory", "v1"),
        ("V2_agent_router_memory", "v2"),
        ("V3_agent_router_memory_aligned", "v3"),
        ("V4_agent_router_memory_clip", "v4"),
    ]
    print(f"LLM health check | available={agent.llm_available} | model={agent.llm.model}")
    for name, vid in variants:
        rows = evaluate_variant(agent, vid)
        summary = summarize(rows)
        print_summary(name, summary)
        write_csv(root / "results" / f"{name}.csv", rows)
    print("\nSaved per-query results to results/*.csv")


if __name__ == "__main__":
    main()
