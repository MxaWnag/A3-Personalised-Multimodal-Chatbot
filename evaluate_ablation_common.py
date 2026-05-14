from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ablation_report_v0_v4 import render_v0_v4_markdown_report  # noqa: E402

from benchmark_session import benchmark_session  # noqa: E402
from evaluate import (  # noqa: E402
    evidence_consistency,
    load_benchmark,
    recall_at_k,
    recall_at_k_source,
    reciprocal_rank,
    reciprocal_rank_source,
    task_success,
    token_usage_proxy,
)
from mvp_agent import MVPAgent  # noqa: E402
from session_store import SESSION_STORE  # noqa: E402


def _replay_session(item: Dict[str, Any], session_id: str) -> None:
    SESSION_STORE.clear_session(session_id)
    for t in item.get("prior_turn") or []:
        if isinstance(t, dict):
            SESSION_STORE.append_turn(session_id, str(t.get("user", "")), str(t.get("assistant", "")))


def ask_item(agent: MVPAgent, item: Dict[str, Any], variant: str, align_recovery: bool) -> Dict[str, Any]:
    vf = (variant or "v2").lower()
    sid = f"abl::{item.get('id', 'x')}::{vf}"
    msg = str(item.get("query", ""))
    if vf == "v0":
        pre = benchmark_session(item)
        if pre:
            msg = f"{pre}\n\nCurrent question: {item.get('query', '')}"
    else:
        _replay_session(item, sid)
    align = bool(align_recovery) and vf == "v4"
    return agent.ask(msg, variant=vf, session_id=sid, align_recovery=align)


def score_row(agent: MVPAgent, item: Dict[str, Any], out: Dict[str, Any], variant: str) -> Dict[str, Any]:
    vf = variant.lower()
    retrieved_ids = list(out.get("retrieved_ids") or [])
    retrieved_items = list(out.get("retrieved_items") or [])

    if vf == "v0":
        r5 = 0.0
        rr = 0.0
    elif "gold_docs" in item:
        r5 = recall_at_k(retrieved_ids, item["gold_docs"], agent, retrieved_items)
        rr = reciprocal_rank(retrieved_ids, item["gold_docs"], agent, retrieved_items)
    else:
        r5 = recall_at_k_source(retrieved_ids, item.get("gold_sources") or [], agent, retrieved_items)
        rr = reciprocal_rank_source(retrieved_ids, item.get("gold_sources") or [], agent)

    kw = item.get("keywords") or []
    if not isinstance(kw, list):
        kw = []
    keyword_score = task_success(out["answer"], kw)
    ev_score = evidence_consistency(out["answer"], retrieved_ids)
    success = 0.7 * keyword_score + 0.3 * ev_score

    return {
        "id": item.get("id", ""),
        "family": item.get("family", ""),
        "variant": vf,
        "recall5": r5,
        "mrr": rr,
        "keyword_score": keyword_score,
        "evidence_score": ev_score,
        "success": success,
        "latency_ms": out["latency_ms"],
        "tool_calls": out["tool_calls"],
        "token_usage_proxy": token_usage_proxy(out["answer"]),
        "llm_used": int(bool(out.get("llm_used", False))),
        "retriever_backend": str(out.get("retriever_backend", "")),
    }


def write_ablation_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "family",
        "variant",
        "recall5",
        "mrr",
        "keyword_score",
        "evidence_score",
        "success",
        "latency_ms",
        "tool_calls",
        "token_usage_proxy",
        "llm_used",
        "retriever_backend",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return mean(float(r[key]) for r in rows)


def summarize_family(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    fam: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        fam.setdefault(str(r["family"]), []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for k, vs in fam.items():
        out[k] = {
            "Recall@5": mean(x["recall5"] for x in vs),
            "MRR": mean(x["mrr"] for x in vs),
            "TaskSuccess": mean(x["success"] for x in vs),
            "LatencyMs": mean(x["latency_ms"] for x in vs),
            "ToolCalls": mean(x["tool_calls"] for x in vs),
        }
    return out


def run_pairs(
    tag: str,
    pairs: List[Tuple[str, str, str]],
    align_recovery: bool = True,
    csv_suffix: str = "",
) -> None:
    root = ROOT
    agent = MVPAgent(root / "data")
    benchmark = load_benchmark(root)
    print(json.dumps({"tag": tag, "n_items": len(benchmark), "llm": agent.llm_available, "csv_suffix": csv_suffix}, indent=2))
    all_rows: List[Dict[str, Any]] = []
    for csv_name, label, vid in pairs:
        rows_out: List[Dict[str, Any]] = []
        for item in benchmark:
            out = ask_item(agent, item, vid, align_recovery=align_recovery)
            row = score_row(agent, item, out, vid)
            rows_out.append(row)
            all_rows.append(row)
        base = Path(csv_name)
        out_name = f"{base.stem}{csv_suffix}{base.suffix}" if csv_suffix else csv_name
        write_ablation_csv(root / "results" / out_name, rows_out)
        print(f"Wrote results/{out_name} ({len(rows_out)} rows) [{label}]")
    print("By family (all variants combined):")
    print(json.dumps(summarize_family(all_rows), indent=2))
