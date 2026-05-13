# System Comparison: V0 (Plain LLM) vs V4 (Hybrid Multimodal Agent) — Version 2

Generated: 2026-05-13 12:51 UTC
Benchmark: `data/benchmark_course_assistant.json` (40 items; four query families × 10)

## Version 2 changes vs earlier report
- **V4 LangGraph** follows the report workflow: `prepare → plan → retrieve (hybrid) → answer → verify`, then **citation rewrite** (`rewrite → verify`) and/or **retrieve recovery** (`retrieve_recovery → answer → verify`) when `align_recovery=true`, with budgets `V4_MAX_REWRITE_ATTEMPTS` / `V4_MAX_RETRIEVE_RECOVERY` (default 2 each).
- **V0** unchanged: plain LLM, no retrieval.

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
| TaskSuccess | 0.022 | 0.724 | +0.703 |
| Latency (ms) | 0.1 | 402.2 | +402.1 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

## By query family

### cross_modal

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.900 | +0.900 |
| MRR | 0.000 | 0.900 | +0.900 |
| TaskSuccess | 0.064 | 0.901 | +0.837 |
| Latency (ms) | 0.1 | 36.0 | +36.0 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

### factual

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.900 | +0.900 |
| MRR | 0.000 | 0.000 | +0.000 |
| TaskSuccess | 0.000 | 0.860 | +0.860 |
| Latency (ms) | 0.1 | 1510.0 | +1509.9 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

### follow_up

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.550 | +0.550 |
| MRR | 0.000 | 0.000 | +0.000 |
| TaskSuccess | 0.023 | 0.487 | +0.463 |
| Latency (ms) | 0.1 | 32.6 | +32.6 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

### multi_hop

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-------------|
| Recall@5 | 0.000 | 0.700 | +0.700 |
| MRR | 0.000 | 0.000 | +0.000 |
| TaskSuccess | 0.000 | 0.650 | +0.650 |
| Latency (ms) | 0.1 | 30.1 | +30.0 |
| ToolCalls | 0.000 | 6.000 | +6.000 |

## Interpretation
- **V0 retrieval metrics** are zero by design; compare TaskSuccess, latency, and tool cost for the system-level baseline.
- **V4** may show higher ToolCalls when verify/rewrite/retrieve_recovery loops run.
- **Follow-up**: `prior_turn` replayed into `SESSION_STORE` before the follow-up query.

## Artifacts
- `results/ablation_v0_v4_V0_plain_llm_v2.csv`
- `results/ablation_v0_v4_V4_hybrid_rag_v2.csv`
