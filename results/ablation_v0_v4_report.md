# System Comparison: V0 (Plain LLM) vs V4 (Hybrid Multimodal Agent)

Generated: 2026-05-12 15:25 UTC
Benchmark: `data/benchmark_course_assistant.json` (40 items; four query families × 10)

## Hypothesis
The final hybrid multimodal agent (v4) improves grounded answer quality and retrieval on
course materials versus a plain LLM baseline (v0) with no retrieval.

## Setup
- **V0**: plain LLM only (`plain_llm`); no Chroma retrieval.
- **V4**: full hybrid multimodal agent (retrieval + bounded verify recovery).
- **Metrics**: Recall@5 and MRR (retrieval; v0 is always 0); TaskSuccess = 0.7×keyword + 0.3×evidence;
  LatencyMs and ToolCalls (efficiency). Retrieval scores use Chroma ids and metadata.
- **LLM available during run**: True
- **Follow-up items**: `prior_turn` is replayed into session memory before the follow-up query.

## Overall

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-----------|
| Recall@5 | 0.000 | 0.900 | +0.900 |
| MRR | 0.000 | 0.877 | +0.877 |
| TaskSuccess | 0.465 | 0.779 | +0.314 |
| LatencyMs | 13662.573 | 25517.757 | +11855.184 |
| ToolCalls | 0.000 | 3.475 | +3.475 |

## By query family

### factual

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-----------|
| Recall@5 | 0.000 | 1.000 | +1.000 |
| MRR | 0.000 | 0.933 | +0.933 |
| TaskSuccess | 0.472 | 0.702 | +0.229 |
| LatencyMs | 15272.638 | 22451.990 | +7179.352 |
| ToolCalls | 0.000 | 3.000 | +3.000 |

### cross_modal

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-----------|
| Recall@5 | 0.000 | 0.800 | +0.800 |
| MRR | 0.000 | 0.800 | +0.800 |
| TaskSuccess | 0.257 | 0.793 | +0.537 |
| LatencyMs | 7907.137 | 17066.640 | +9159.504 |
| ToolCalls | 0.000 | 3.000 | +3.000 |

### multi_hop

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-----------|
| Recall@5 | 0.000 | 0.950 | +0.950 |
| MRR | 0.000 | 0.950 | +0.950 |
| TaskSuccess | 0.636 | 0.919 | +0.283 |
| LatencyMs | 17172.145 | 26813.558 | +9641.413 |
| ToolCalls | 0.000 | 3.000 | +3.000 |

### follow_up

| Metric | V0 | V4 | Δ (V4−V0) |
|--------|----|----|-----------|
| Recall@5 | 0.000 | 0.850 | +0.850 |
| MRR | 0.000 | 0.825 | +0.825 |
| TaskSuccess | 0.496 | 0.703 | +0.207 |
| LatencyMs | 14298.372 | 35738.840 | +21440.469 |
| ToolCalls | 0.000 | 4.900 | +4.900 |

## Interpretation
- **V0 retrieval metrics** are zero by design; compare TaskSuccess, latency, and tool cost for the system-level baseline.
- **Factual / multi-hop**: v4 should improve keyword grounding when answers depend on lecture PDFs.
- **Cross-modal**: v4 should dominate Recall@5 / MRR on figure-linked gold docs.
- **Follow-up**: session replay via benchmark `prior_turn`.

## Artifacts
- `results/ablation_v0_v4_V0_plain_llm.csv`
- `results/ablation_v0_v4_V4_hybrid_rag.csv`
