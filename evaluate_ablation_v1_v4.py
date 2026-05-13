"""Ablation: V1 text-only RAG vs V4 hybrid."""

from evaluate_ablation_common import run_pairs

if __name__ == "__main__":
    run_pairs(
        "ablation_v1_v4",
        [
            ("ablation_v1_v4_V1_text_only_rag.csv", "V1_text_only_rag", "v1"),
            ("ablation_v1_v4_V4_hybrid_rag.csv", "V4_hybrid_rag", "v4"),
        ],
        align_recovery=True,
    )
