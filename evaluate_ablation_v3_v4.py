"""Ablation: V3 image-only CLIP vs V4 hybrid."""

from evaluate_ablation_common import run_pairs

if __name__ == "__main__":
    run_pairs(
        "ablation_v3_v4",
        [
            ("ablation_v3_v4_V3_image_only_clip.csv", "V3_image_only_clip", "v3"),
            ("ablation_v3_v4_V4_hybrid_rag.csv", "V4_hybrid_rag", "v4"),
        ],
        align_recovery=True,
    )
