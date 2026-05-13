from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

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


def render_v0_v4_markdown_report(
    v0_csv: Path,
    v4_csv: Path,
    out_md: Path,
    *,
    workflow_note: str = "",
) -> None:
    """Build V0 vs V4 comparison tables from two per-variant CSVs."""

    def load_rows(p: Path) -> List[Dict[str, str]]:
        with p.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    v0_rows = load_rows(v0_csv)
    v4_rows = load_rows(v4_csv)
    fams = sorted({str(r.get("family", "")) for r in v0_rows + v4_rows if r.get("family")})

    def pack(rows: List[Dict[str, str]]) -> Dict[str, float]:
        return {
            "Recall@5": _mean(rows, "recall5"),
            "MRR": _mean(rows, "mrr"),
            "TaskSuccess": _mean(rows, "success"),
            "LatencyMs": _mean(rows, "latency_ms"),
            "ToolCalls": _mean(rows, "tool_calls"),
        }

    o0 = pack(v0_rows)
    o4 = pack(v4_rows)

    llm_v4_rate = _mean(v4_rows, "llm_used")
    llm_note = ""
    if llm_v4_rate < 0.5:
        llm_note = (
            "\n- **Note (this run)**: V4 rows show low `llm_used` (Ollama/model likely unavailable); "
            "planner/verifier/composer used JSON/text fallbacks. Re-run with a live LLM for report-grade answer quality.\n"
        )

    def row_line(metric: str, k: str) -> str:
        a, b = o0[k], o4[k]
        d = b - a
        if k == "LatencyMs":
            return f"| {metric} | {a:.1f} | {b:.1f} | {d:+.1f} |"
        return f"| {metric} | {a:.3f} | {b:.3f} | {d:+.3f} |"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = [
        "# System Comparison: V0 (Plain LLM) vs V4 (Hybrid Multimodal Agent) — Version 2",
        "",
        f"Generated: {ts}",
        "Benchmark: `data/benchmark_course_assistant.json` (40 items; four query families × 10)",
        "",
        "## Version 2 changes vs earlier report",
        "- **V4 LangGraph** follows the report workflow: `prepare → plan → retrieve (hybrid) → answer → verify`, then **citation rewrite** (`rewrite → verify`) and/or **retrieve recovery** (`retrieve_recovery → answer → verify`) when `align_recovery=true`, with budgets `V4_MAX_REWRITE_ATTEMPTS` / `V4_MAX_RETRIEVE_RECOVERY` (default 2 each).",
        "- **V0** unchanged: plain LLM, no retrieval.",
        "",
    ]
    if workflow_note:
        lines.extend(["", workflow_note.strip(), ""])
    lines.extend(
        [
            "## Hypothesis",
            "The hybrid multimodal agent (v4) improves grounded answer quality and retrieval on course materials",
            "versus a plain LLM baseline (v0) with no retrieval.",
            "",
            "## Setup",
            "- **V0**: plain LLM only (`plain_llm`); no Chroma retrieval.",
            "- **V4**: hybrid multimodal RAG + verifier + optional **rewrite** / **retrieve_recovery** loops (`align_recovery=true` during this ablation).",
            "- **Metrics**: Recall@5 and MRR (retrieval; v0 is always 0); TaskSuccess = 0.7×keyword + 0.3×evidence;",
            "  **Latency** (milliseconds) and **ToolCalls** (efficiency). Retrieval scores use Chroma ids and metadata.",
            f"- **Artifacts**: `{v0_csv.name}`, `{v4_csv.name}`.",
            llm_note,
            "",
            "## Overall",
            "",
            "| Metric | V0 | V4 | Δ (V4−V0) |",
            "|--------|----|----|-------------|",
            row_line("Recall@5", "Recall@5"),
            row_line("MRR", "MRR"),
            row_line("TaskSuccess", "TaskSuccess"),
            row_line("Latency (ms)", "LatencyMs"),
            row_line("ToolCalls", "ToolCalls"),
            "",
            "## By query family",
            "",
        ]
    )

    for fam in fams:
        r0 = [r for r in v0_rows if str(r.get("family", "")) == fam]
        r4 = [r for r in v4_rows if str(r.get("family", "")) == fam]
        p0, p4 = pack(r0), pack(r4)
        lines.append(f"### {fam}")
        lines.append("")
        lines.append("| Metric | V0 | V4 | Δ (V4−V0) |")
        lines.append("|--------|----|----|-------------|")
        for label, k in [
            ("Recall@5", "Recall@5"),
            ("MRR", "MRR"),
            ("TaskSuccess", "TaskSuccess"),
            ("Latency (ms)", "LatencyMs"),
            ("ToolCalls", "ToolCalls"),
        ]:
            a, b = p0[k], p4[k]
            d = b - a
            if k == "LatencyMs":
                lines.append(f"| {label} | {a:.1f} | {b:.1f} | {d:+.1f} |")
            else:
                lines.append(f"| {label} | {a:.3f} | {b:.3f} | {d:+.3f} |")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "- **V0 retrieval metrics** are zero by design; compare TaskSuccess, latency, and tool cost for the system-level baseline.",
            "- **V4** may show higher ToolCalls when verify/rewrite/retrieve_recovery loops run.",
            "- **Follow-up**: `prior_turn` replayed into `SESSION_STORE` before the follow-up query.",
            "",
            "## Artifacts",
            f"- `results/{v0_csv.name}`",
            f"- `results/{v4_csv.name}`",
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {out_md}")


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
