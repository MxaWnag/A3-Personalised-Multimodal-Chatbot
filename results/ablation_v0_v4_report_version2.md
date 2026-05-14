# System Comparison: V0 (Plain LLM) vs V4 (Hybrid Multimodal Agent) — Version 2

Generated: 2026-05-14 18:01 UTC
Benchmark: `data/benchmark_course_assistant.json` (40 items; four query families × 10)

## Version 2 changes vs earlier report
- **V4 LangGraph** follows the report workflow: `prepare → plan → retrieve (hybrid) → answer → verify`, then **citation rewrite** (`rewrite → verify`) and/or **retrieve recovery** (`retrieve_recovery → answer → verify`) when `align_recovery=true`, with budgets `V4_MAX_REWRITE_ATTEMPTS` / `V4_MAX_RETRIEVE_RECOVERY` (default 2 each).
- **V0** unchanged: plain LLM, no retrieval.


Regenerated from CSVs via ablation_report_v0_v4.py (no full eval). For a paired re-run use evaluate_ablation_v0_v4.py --version2.

## Hypothesis
The hybrid multimodal agent (v4) improves grounded answer quality and retrieval on course materials
versus a plain LLM baseline (v0) with no retrieval.

## Setup
- **V0**: plain LLM only (`plain_llm`); no Chroma retrieval.
- **V4**: hybrid multimodal RAG + verifier + optional **rewrite** / **retrieve_recovery** loops (`align_recovery=true` during this ablation).
- **Metrics**: Recall@5 and MRR (retrieval; v0 is always 0); TaskSuccess = 0.7×keyword + 0.3×evidence;
  **Latency** (milliseconds) and **ToolCalls** (efficiency). Retrieval scores use Chroma ids and metadata.
- **Artifacts**: `ablation_v0_v4_V0_plain_llm_v2.csv`, `ablation_v0_v4_V4_hybrid_rag_v2.csv`.

- **Note (this run)**: V4 rows show low `llm_used` (Ollama/model likely unavailable); planner/verifier/composer used JSON/text fallbacks. Re-run with a live LLM for report-grade answer quality.


## Overall

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.762 | +0.762 |
| MRR | 0.000 | 0.225 | +0.225 |
| TaskSuccess | 0.542 | 0.724 | +0.182 |
| Latency (ms) | 13408.8 | 183.0 | -13225.8 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

## By query family

### cross_modal

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.900 | +0.900 |
| MRR | 0.000 | 0.900 | +0.900 |
| TaskSuccess | 0.443 | 0.901 | +0.458 |
| Latency (ms) | 11002.3 | 38.4 | -10963.9 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

### factual

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.900 | +0.900 |
| MRR | 0.000 | 0.000 | +0.000 |
| TaskSuccess | 0.560 | 0.860 | +0.300 |
| Latency (ms) | 13675.5 | 622.3 | -13053.2 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

### follow_up

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.550 | +0.550 |
| MRR | 0.000 | 0.000 | +0.000 |
| TaskSuccess | 0.467 | 0.487 | +0.020 |
| Latency (ms) | 13971.8 | 35.0 | -13936.8 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

### multi_hop

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.700 | +0.700 |
| MRR | 0.000 | 0.000 | +0.000 |
| TaskSuccess | 0.700 | 0.650 | -0.050 |
| Latency (ms) | 14985.5 | 36.2 | -14949.3 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

## Interpretation
- **V0 retrieval metrics** are zero by design; compare TaskSuccess, latency, and tool cost for the system-level baseline.
- **V4** may show higher ToolCalls when verify/rewrite/retrieve_recovery loops run.
- **Follow-up**: `prior_turn` replayed into `SESSION_STORE` before the follow-up query.

## Artifacts
- `results/ablation_v0_v4_V0_plain_llm_v2.csv`
- `results/ablation_v0_v4_V4_hybrid_rag_v2.csv`
