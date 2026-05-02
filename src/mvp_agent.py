import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .pipeline.offline_index import build_offline_index
    from .pipeline.retrieval import ChromaRetriever
    from .pipeline.router import route_query
except ImportError:
    from pipeline.offline_index import build_offline_index
    from pipeline.retrieval import ChromaRetriever
    from pipeline.router import route_query


class LLMClient:
    def __init__(self, model: str = "llama3.1:8b", endpoint: str = "http://localhost:11434/api/generate") -> None:
        self.model = model
        self.endpoint = endpoint
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
        with (self.data_dir / "user_profile.json").open("r", encoding="utf-8") as f:
            self.profile = json.load(f)
        self.llm = LLMClient()
        self.llm_available = self.llm.is_available()
        self.current_filters = {}
        self.retrieval_note = ""

    def _load_docs(self) -> List[Dict]:
        docs = []
        for path in [self.data_dir / "sample" / "text_docs.jsonl", self.data_dir / "sample" / "image_docs.jsonl"]:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    docs.append(json.loads(line))
        return docs

    def _profile_text(self) -> str:
        pref = self.profile.get("preferences", {})
        likes = ", ".join(pref.get("likes", []))
        dislikes = ", ".join(pref.get("dislikes", []))
        style = pref.get("typical_trip_style", "")
        return f"User likes: {likes}. Dislikes: {dislikes}. Style: {style}."

    def _parse_query_filters(self, query: str) -> Dict:
        q = query.lower()
        filters = {}
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
            and (not filters.get("week") or str(d.get("week", "")).lower() == filters["week"].lower())
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

    def _build_answer(self, query: str, evidence_docs: List[Dict], use_memory: bool) -> str:
        evidence_items = []
        for d in evidence_docs[:3]:
            did = d.get("id", "")
            text = re.sub(r"\s+", " ", d.get("document", "")).strip()[:260]
            evidence_items.append(f"- [{did}] {text}")
        evidence_block = "\n".join(evidence_items) if evidence_items else "- No retrieved evidence."
        memory_text = self._profile_text() if use_memory else ""
        prompt = (
            "You are an academic assistant. Respond concisely under 120 words.\n"
            f"Query: {query}\n"
            f"User profile: {memory_text}\n"
            f"Retrieval note: {self.retrieval_note}\n"
            f"Evidence:\n{evidence_block}\n"
            "Output format:\n"
            "### Direct Answer\n"
            "### Key Points\n"
            "### Evidence Used\n"
            "### Recommended Next Step\n"
        )
        llm_answer = self.llm.generate(prompt)
        if llm_answer:
            return llm_answer
        quick = " ".join([e.split("] ", 1)[-1][:120] for e in evidence_items[:2]]) if evidence_items else "No evidence."
        profile_line = f"- {memory_text}" if memory_text else "- No personalised profile used for this turn."
        note_line = f"- {self.retrieval_note}" if self.retrieval_note else "- No retrieval caveat."
        return (
            "### Direct Answer\n"
            f"Here is a concise summary based on nearest relevant slides: {quick}\n\n"
            "### Key Points\n"
            "- The response is grounded in retrieved evidence.\n"
            "- Retrieval uses ChromaDB multimodal collections with routing.\n"
            f"{profile_line}\n"
            f"{note_line}\n\n"
            "### Evidence Used\n"
            f"{evidence_block}\n\n"
            "### Recommended Next Step\n"
            "- Ask for a focused follow-up by topic, week, or comparison."
        )

    def _retrieve(self, query: str, route: str, variant: str):
        if variant == "v1":
            rows = self.retriever.query_text(query, top_k=5, filters=self.current_filters)
            return rows, "text_only"
        if variant == "v3":
            rows = self.retriever.query_image(query, top_k=5, filters=self.current_filters)
            return rows, "image_only_clip"
        if variant == "v4":
            rows = self.retriever.query_hybrid(query, top_k=5, filters=self.current_filters)
            return rows, "hybrid_text_image_caption"
        # v2: router driven
        if route == "text":
            rows = self.retriever.query_text(query, top_k=5, filters=self.current_filters)
            return rows, "text_only"
        if route == "image":
            rows = self.retriever.query_image(query, top_k=5, filters=self.current_filters)
            return rows, "image_only_clip"
        rows = self.retriever.query_hybrid(query, top_k=5, filters=self.current_filters)
        return rows, "hybrid_text_image_caption"

    def ask(self, query: str, variant: str = "v2") -> Dict:
        start = time.perf_counter()
        self._prepare_filters(query)

        if variant == "v0":
            answer = f"### Direct Answer\n{query}\n\n### Key Points\n- Baseline without retrieval.\n\n### Evidence Used\n- None.\n\n### Recommended Next Step\n- Try v2/v4 for grounded retrieval."
            return {
                "variant": variant,
                "route": "none",
                "answer": answer,
                "retrieved_ids": [],
                "latency_ms": (time.perf_counter() - start) * 1000,
                "tool_calls": 0,
                "llm_used": False,
                "llm_available": self.llm_available,
                "alignment_ready": False,
                "clip_ready": True,
                "expanded_query": query,
                "retriever_backend": "none",
                "retrieval_note": self.retrieval_note,
                "active_filters": self.current_filters,
                "index_stats": self.index_stats,
            }

        route = "fixed_text" if variant == "v1" else route_query(query)
        expanded_query = query
        tool_calls = 1
        if variant in {"v2", "v4"} and route in {"hybrid"}:
            expanded_query = f"{query}. {self._profile_text()}"
            tool_calls += 1
        rows, backend = self._retrieve(expanded_query, route, variant)
        tool_calls += 1
        retrieved_ids = [r.id for r in rows]
        docs = [{"id": r.id, "document": r.document, "metadata": r.metadata} for r in rows]
        answer = self._build_answer(query, docs, use_memory=(variant in {"v2", "v4"}))

        return {
            "variant": variant,
            "route": route,
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "tool_calls": tool_calls,
            "llm_used": self.llm.last_call_used_llm,
            "llm_available": self.llm_available,
            "alignment_ready": variant == "v3",
            "clip_ready": True,
            "expanded_query": expanded_query,
            "retriever_backend": backend,
            "retrieval_note": self.retrieval_note,
            "active_filters": self.current_filters,
            "index_stats": self.index_stats,
            "retrieved_items": docs,
        }


def interactive_main() -> None:
    root = Path(__file__).resolve().parents[1]
    agent = MVPAgent(root / "data")
    print("Minimal MVP Agent. Type 'exit' to quit.")
    print(f"LLM available: {agent.llm_available} (model={agent.llm.model})")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        v = input("Variant [v0|v1|v2|v3|v4]: ").strip().lower() or "v2"
        result = agent.ask(q, variant=v)
        print("\n---")
        print(f"Route: {result['route']}")
        print(f"Retrieved: {result['retrieved_ids']}")
        print(f"Latency(ms): {result['latency_ms']:.2f}")
        print(f"Tool calls: {result['tool_calls']}")
        print(result["answer"])


if __name__ == "__main__":
    interactive_main()
