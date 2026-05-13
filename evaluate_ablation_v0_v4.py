"""Ablation: V0 plain LLM vs V4 hybrid (LangGraph + verify + optional self-correct loops)."""

import sys

from evaluate_ablation_common import ROOT, render_v0_v4_markdown_report, run_pairs

if __name__ == "__main__":
    version2 = "--version2" in sys.argv
    suffix = "_v2" if version2 else ""
    run_pairs(
        "ablation_v0_v4",
        [
            ("ablation_v0_v4_V0_plain_llm.csv", "V0_plain_llm", "v0"),
            ("ablation_v0_v4_V4_hybrid_rag.csv", "V4_hybrid_rag", "v4"),
        ],
        align_recovery=True,
        csv_suffix=suffix,
    )
    if version2:
        render_v0_v4_markdown_report(
            ROOT / "results" / f"ablation_v0_v4_V0_plain_llm{suffix}.csv",
            ROOT / "results" / f"ablation_v0_v4_V4_hybrid_rag{suffix}.csv",
            ROOT / "results" / "ablation_v0_v4_report_version2.md",
        )
