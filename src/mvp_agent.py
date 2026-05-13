import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .agent_graph import compile_langgraph
    from .agent_policy import DeepSeekComposer, DeepSeekPlanner, DeepSeekVerifier
    from .agent_tools import build_tools
    from .pipeline.offline_index import build_offline_index
    from .pipeline.retrieval import ChromaRetriever
    from .pipeline.router import route_query
    from .session_store import SESSION_STORE
except ImportError:
    from agent_graph import compile_langgraph
    from agent_policy import DeepSeekComposer, DeepSeekPlanner, DeepSeekVerifier
    from agent_tools import build_tools
    from pipeline.offline_index import build_offline_index
    from pipeline.retrieval import ChromaRetriever
    from pipeline.router import route_query
    from session_store import SESSION_STORE


class LLMClient:
    def __init__(self, model: Optional[str] = None, endpoint: Optional[str] = None) -> None:
        self.model = (model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")).strip()
        self.endpoint = (endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")).strip()
        self.last_call_used_llm = False

    def is_available(self, timeout_s: float = 2.0) -> bool:
        try:
            tags_endpoint = self.endpoint.rsplit("/", 1)[0] + "/tags"
            req = urllib.request.Request(tags_endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models", [])
            return bool(models)
        except Exception:
            return False

    def generate(self, prompt: str, timeout_s: Optional[float] = None) -> str:
        if timeout_s is None:
            timeout_s = float(os.getenv("OLLAMA_GENERATE_TIMEOUT_S", "300"))
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = (data.get("response") or "").strip()
            self.last_call_used_llm = bool(text)
            return text
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            self.last_call_used_llm = False
            return ""


class MVPAgent:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.index_dir = self.data_dir / "chroma"
        self.index_stats = build_offline_index(self.data_dir, self.index_dir)
        self.retriever = ChromaRetriever(self.index_dir)
        self.docs = self._load_docs()
        self.doc_map = {d["id"]: d for d in self.docs}
        self.profile: Dict[str, Any] = {}
        profile_path = self.data_dir / "user_profile.json"
        if profile_path.exists():
            with profile_path.open("r", encoding="utf-8") as f:
                self.profile = json.load(f)
        self.llm = LLMClient()
        self.llm_available = self.llm.is_available()
        self.current_filters: Dict[str, Any] = {}
        self.retrieval_note = ""
        self.tools = build_tools(self.retriever)
        self.planner = DeepSeekPlanner(self._llm_chat)
        self.composer = DeepSeekComposer(self._llm_chat)
        self.verifier = DeepSeekVerifier(self._llm_chat)
        self._graph_app = compile_langgraph(self)

    def _llm_chat(self, system: str, user: str) -> str:
        prompt = f"{system.strip()}\n\n{user.strip()}"
        return self.llm.generate(prompt)

    def _load_docs(self) -> List[Dict]:
        docs = []
        for path in [self.data_dir / "sample" / "text_docs.jsonl", self.data_dir / "sample" / "image_docs.jsonl"]:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    docs.append(json.loads(line))
        return docs

    def _profile_text_optional(self) -> str:
        if not self.profile:
            return ""
        ap = self.profile.get("academic_profile") or {}
        parts = []
        if ap.get("study_style"):
            parts.append(f"study_style={ap.get('study_style')}")
        if ap.get("weak_topics"):
            parts.append("weak_topics=" + ", ".join(ap.get("weak_topics") or []))
        return "; ".join(parts)

    def _course_context(self) -> str:
        return f"Indexed courses include INFS7205 and ENGG4800; total manifest docs: {len(self.docs)}."

    def _parse_query_filters(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        filters: Dict[str, Any] = {}
        if "infs7205" in q:
            filters["course"] = "INFS7205"
        elif "engg4800" in q:
            filters["course"] = "ENGG4800"
        m = re.search(r"week\s*([0-9]{1,2})", q)
        if m:
            filters["week"] = f"week{m.group(1)}"
        return filters

    def _prepare_filters(self, query: str) -> None:
        filters = self._parse_query_filters(query)
        self.retrieval_note = ""
        if not filters:
            self.current_filters = {}
            return
        exact = [
            d
            for d in self.docs
            if (not filters.get("course") or d.get("course") == filters["course"])
            and (not filters.get("week") or str(d.get("week", "")).lower() == str(filters["week"]).lower())
        ]
        if exact:
            self.current_filters = filters
            return
        if filters.get("course") and filters.get("week"):
            self.current_filters = {"course": filters["course"]}
            self.retrieval_note = (
                f"No exact materials found for {filters['course']} {filters['week']}; "
                f"using nearest available weeks in {filters['course']}."
            )
            return
        self.current_filters = {}

    def _build_plain_llm_answer(self, query: str) -> str:
        prompt = (
            "You are an academic assistant. Answer from general knowledge only. "
            "No course documents were retrieved; do not fabricate specific filenames, "
            "week labels, or claims that you looked up materials.\n"
            f"Query: {query}\n"
            "Respond concisely under 120 words.\n"
            "Output format:\n"
            "### Direct Answer\n"
            "### Key Points\n"
            "### Evidence Used\n"
            "### Recommended Next Step\n"
        )
        llm_answer = self.llm.generate(prompt)
        if llm_answer:
            return llm_answer
        return (
            "### Direct Answer\n"
            "The language model endpoint is unavailable or returned an empty response.\n\n"
            "### Key Points\n"
            "- Plain-LLM mode uses no retrieval.\n\n"
            "### Evidence Used\n"
            "- None.\n\n"
            "### Recommended Next Step\n"
            "- Start Ollama or use a RAG variant for template fallback on evidence.\n"
        )

    def _retrieve(self, query: str, route: str, variant: str, top_k: int = 5):
        vf = (variant or "v2").lower()
        if vf == "v1":
            rows = self.retriever.query_text(query, top_k=top_k, filters=self.current_filters)
            return rows, "text_only"
        if vf == "v3":
            rows = self.retriever.query_image(query, top_k=top_k, filters=self.current_filters)
            return rows, "image_only_clip"
        if vf == "v4":
            rows = self.retriever.query_hybrid(query, top_k=top_k, filters=self.current_filters)
            return rows, "hybrid_text_image_caption"
        if route == "text":
            rows = self.retriever.query_text(query, top_k=top_k, filters=self.current_filters)
            return rows, "text_only"
        if route == "image":
            rows = self.retriever.query_image(query, top_k=top_k, filters=self.current_filters)
            return rows, "image_only_clip"
        rows = self.retriever.query_hybrid(query, top_k=top_k, filters=self.current_filters)
        return rows, "hybrid_text_image_caption"

    def _retrieve_items_as_dicts(
        self, query: str, variant: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        saved = dict(self.current_filters)
        self.current_filters = dict(filters or {})
        try:
            vf = (variant or "v2").lower()
            route = "fixed_text" if vf == "v1" else route_query(query)
            expanded = query
            if vf in {"v2", "v4"} and route in {"hybrid"}:
                extra = self._profile_text_optional()
                if extra:
                    expanded = f"{query}. {extra}"
            rows, _backend = self._retrieve(expanded, route, vf, top_k=top_k)
            return [
                {"id": r.id, "score": float(r.score), "document": r.document, "metadata": dict(r.metadata or {})}
                for r in rows
            ]
        finally:
            self.current_filters = saved

    @staticmethod
    def _dedupe_items(items: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for it in sorted(items, key=lambda x: -float(x.get("score", 0.0))):
            rid = str(it.get("id", ""))
            if rid and rid not in seen:
                seen[rid] = it
        return list(seen.values())[:limit]

    @staticmethod
    def _format_evidence(items: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for it in items[:10]:
            rid = it.get("id", "")
            doc = (it.get("document") or "").strip()
            if len(doc) > 700:
                doc = doc[:700] + "…"
            lines.append(f"[{rid}] (score={float(it.get('score', 0)):.3f})\n{doc}")
        return "\n\n---\n\n".join(lines) if lines else "(no evidence retrieved)"

    def ask(
        self,
        query: str,
        variant: str = "v2",
        session_id: Optional[str] = None,
        align_recovery: bool = False,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        vf = (variant or "v2").lower()
        sid = (session_id or "default").strip() or "default"

        if vf == "v0":
            self._prepare_filters(query)
            answer = self._build_plain_llm_answer(query)
            out = {
                "variant": vf,
                "route": "none",
                "answer": answer,
                "retrieved_ids": [],
                "retrieved_items": [],
                "latency_ms": (time.perf_counter() - start) * 1000,
                "tool_calls": 0,
                "llm_used": self.llm.last_call_used_llm,
                "llm_available": self.llm_available,
                "alignment_ready": False,
                "clip_ready": True,
                "expanded_query": query,
                "retriever_backend": "plain_llm",
                "retrieval_note": self.retrieval_note,
                "active_filters": self.current_filters,
                "index_stats": self.index_stats,
                "tool_trace": [{"node": "v0_plain", "stop_reason": "plain_llm"}],
                "verify_pass": None,
                "stop_reason": "plain_llm",
            }
            SESSION_STORE.append_turn(sid, query, answer)
            return out

        self._prepare_filters(query)
        initial: Dict[str, Any] = {
            "session_id": sid,
            "user_message": query,
            "variant": vf,
            "align_recovery": bool(align_recovery),
            "tool_trace": [],
            "rewrite_count": 0,
            "retrieve_recovery_count": 0,
        }
        final = self._graph_app.invoke(initial)
        answer = (final.get("answer_text") or "").strip() or "(empty answer)"
        trace = final.get("tool_trace") or []
        items = final.get("retrieved_items") or []
        retrieved_ids = [str(it.get("id")) for it in items if it.get("id")]
        vj = final.get("verify_json") or {}
        stop = final.get("stop_reason") or "unknown"

        out = {
            "variant": vf,
            "route": final.get("route", "langgraph"),
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "retrieved_items": items,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "tool_calls": len(trace),
            "llm_used": self.llm.last_call_used_llm,
            "llm_available": self.llm_available,
            "alignment_ready": vf == "v3",
            "clip_ready": True,
            "expanded_query": query,
            "retriever_backend": "langgraph",
            "retrieval_note": self.retrieval_note,
            "active_filters": self.current_filters,
            "index_stats": self.index_stats,
            "tool_trace": trace,
            "verify_pass": bool(vj.get("pass")) if vj else None,
            "stop_reason": stop,
        }
        SESSION_STORE.append_turn(sid, query, answer)
        return out


def interactive_main() -> None:
    root = Path(__file__).resolve().parents[1]
    agent = MVPAgent(root / "data")
    print("MVP Agent (LangGraph for v1–v4). Type 'exit' to quit.")
    print(f"LLM available: {agent.llm_available} (model={agent.llm.model})")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        v = input("Variant [v0|v1|v2|v3|v4]: ").strip().lower() or "v2"
        result = agent.ask(q, variant=v, session_id="cli")
        print("\n---")
        print(f"Route: {result['route']}")
        print(f"Stop: {result.get('stop_reason')}")
        print(f"Retrieved: {result['retrieved_ids'][:8]}")
        print(f"Latency(ms): {result['latency_ms']:.2f}")
        print(result["answer"])


if __name__ == "__main__":
    interactive_main()
