from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    raw = _strip_code_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _try_json(text: str) -> Dict[str, Any] | None:
    if not (text or "").strip():
        return None
    try:
        return _safe_json_loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


class LlmPlanner:
    def __init__(self, llm_invoke: Callable[[str, str], str]) -> None:
        self._llm = llm_invoke

    def plan(
        self,
        *,
        user_message: str,
        conversation_summary: str,
        course_context: str,
    ) -> Dict[str, Any]:
        system = (
            "You are a careful planner for a course assistant with multimodal retrieval (text + images). "
            "Return STRICT JSON only."
        )
        user = f"""Task: propose retrieval queries and a short plan. Every user turn runs vector retrieval; queries should cover the information need.

Course context (may be empty):
{course_context}

Conversation so far (may be empty):
{conversation_summary}

User message:
{user_message}

Return JSON with keys:
- retrieve_queries: array of 1-4 short English search strings (standalone; no pronouns like "it/that slide")
- plan_notes: 1-3 sentences on what evidence you need and any modality hints (figure/table/text)
"""
        raw = self._llm(system, user)
        data = _try_json(raw)
        if data is None:
            q = user_message.strip()[:200] or "course materials"
            return {
                "retrieve_queries": [q],
                "plan_notes": "(planner fallback: empty or non-JSON LLM output)",
            }
        queries = data.get("retrieve_queries") or []
        if not isinstance(queries, list):
            queries = []
        queries = [str(x).strip() for x in queries if str(x).strip()][:4]
        if not queries:
            queries = [user_message.strip()[:200] or "course materials"]
        return {
            "retrieve_queries": queries,
            "plan_notes": str(data.get("plan_notes", "")).strip(),
        }


class LlmComposer:
    def __init__(self, llm_invoke: Callable[[str, str], str]) -> None:
        self._llm = llm_invoke

    def compose(
        self,
        *,
        conversation_summary: str,
        user_message: str,
        retrieved_cards: str,
        sources: str,
    ) -> str:
        system = (
            "You are a course teaching assistant. Use ONLY the provided evidence cards for factual course claims. "
            "If evidence is insufficient for a course question, say what is missing. "
            "For greetings or non-course small talk, reply briefly without inventing lecture facts not in the cards. "
            "Output Markdown with headings: ### Direct Answer, ### Evidence, ### Follow-ups."
        )
        user = f"""Conversation so far:
{conversation_summary}

User message:
{user_message}

Evidence cards:
{retrieved_cards}

Allowed source ids (comma-separated):
{sources}

Write the answer now.
"""
        text = self._llm(system, user).strip()
        if not text:
            clip = (retrieved_cards or "")[:2000]
            return (
                "### Direct Answer\n"
                "The model endpoint returned no text; showing a raw evidence excerpt instead.\n\n"
                "### Evidence\n"
                f"{clip}\n\n"
                "### Follow-ups\n"
                "- Start the LLM service and retry.\n"
            )
        return text

    def compose_citation_rewrite(
        self,
        *,
        conversation_summary: str,
        user_message: str,
        retrieved_cards: str,
        sources: str,
        previous_answer: str,
        verify_feedback: str,
    ) -> str:
        """Regenerate answer over the SAME evidence; fix ids/citations only (no new facts)."""
        system = (
            "You are revising a course assistant answer for CITATION accuracy only. "
            "Use ONLY the same evidence cards; do not add new claims. "
            "Every factual statement must be tied to allowed source ids using [id] in ### Evidence. "
            "Output Markdown with headings: ### Direct Answer, ### Evidence, ### Follow-ups."
        )
        user = f"""Conversation so far:
{conversation_summary}

User message:
{user_message}

Evidence cards (unchanged):
{retrieved_cards}

Allowed source ids (comma-separated):
{sources}

Previous answer (to fix):
{previous_answer}

Verifier feedback (JSON or text):
{verify_feedback}

Rewrite the answer now with correct citations to the allowed ids only.
"""
        text = self._llm(system, user).strip()
        if not text:
            return (
                "### Direct Answer\n"
                "(Rewrite skipped: empty LLM output; keeping prior structure.)\n\n"
                "### Evidence\n"
                f"{previous_answer}\n\n"
                "### Follow-ups\n"
                "- Retry when the LLM endpoint is available.\n"
            )
        return text


class LlmVerifier:
    def __init__(self, llm_invoke: Callable[[str, str], str]) -> None:
        self._llm = llm_invoke

    def verify(
        self,
        *,
        answer: str,
        evidence_text: str,
        user_message: str,
        sources: str,
    ) -> Dict[str, Any]:
        system = "You are a strict verifier for grounded answers. Return STRICT JSON only."
        user = f"""Check whether the answer is supported by the evidence cards.

User message:
{user_message}

Evidence cards:
{evidence_text}

Answer:
{answer}

Allowed source ids:
{sources}

Return JSON:
{{
  "pass": true/false,
  "failure_kind": "none" | "citation" | "retrieval" | "mixed",
  "rewrite_allowed": true/false,
  "retrieval_retry_allowed": true/false,
  "reasons": ["..."],
  "unsupported_claims": ["claims not supported by evidence or bad [id] citations"],
  "suggested_followups": ["short hybrid search queries; use when evidence is missing/too thin"]
}}

Rules:
- failure_kind "citation": answer content mostly ok but [id] citations wrong/missing vs allowed ids.
- failure_kind "retrieval": evidence cards are insufficient to answer; need more hybrid retrieval.
- failure_kind "mixed": both issues.
- If pass is true, set failure_kind to "none" and both *_allowed to false.
"""
        raw = self._llm(system, user)
        data = _try_json(raw)
        if data is None:
            return normalize_verify_json(
                {
                    "pass": False,
                    "failure_kind": "retrieval",
                    "rewrite_allowed": False,
                    "retrieval_retry_allowed": False,
                    "reasons": ["Verifier LLM returned empty or non-JSON output."],
                    "unsupported_claims": [],
                    "suggested_followups": [],
                }
            )
        return normalize_verify_json(data)


def normalize_verify_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize verifier output; infer missing routing fields."""
    pass_ = bool(data.get("pass", False))
    reasons = data.get("reasons") or []
    unsupported = data.get("unsupported_claims") or []
    suggested = data.get("suggested_followups") or []
    fk_raw = str(data.get("failure_kind", "") or "").lower().strip()
    if pass_:
        return {
            "pass": True,
            "failure_kind": "none",
            "rewrite_allowed": False,
            "retrieval_retry_allowed": False,
            "reasons": reasons,
            "unsupported_claims": unsupported,
            "suggested_followups": suggested,
        }
    if fk_raw in ("citation", "retrieval", "mixed"):
        fk = fk_raw
    else:
        has_cit = bool(unsupported) or any("citation" in str(r).lower() or "id" in str(r).lower() for r in reasons)
        has_ret = bool(suggested) or any(
            k in " ".join(str(r).lower() for r in reasons) for k in ("insufficient", "missing evidence", "thin", "retrieve")
        )
        if has_cit and has_ret:
            fk = "mixed"
        elif has_cit:
            fk = "citation"
        elif has_ret:
            fk = "retrieval"
        else:
            fk = "mixed"
    rw = bool(data.get("rewrite_allowed", fk in ("citation", "mixed")))
    rr = bool(data.get("retrieval_retry_allowed", fk in ("retrieval", "mixed")))
    return {
        "pass": False,
        "failure_kind": fk,
        "rewrite_allowed": rw,
        "retrieval_retry_allowed": rr,
        "reasons": reasons,
        "unsupported_claims": unsupported,
        "suggested_followups": suggested if isinstance(suggested, list) else [],
    }
