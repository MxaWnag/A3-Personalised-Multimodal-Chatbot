from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List

import chromadb

from .embeddings import EmbeddingModels


@dataclass
class RetrievedItem:
    id: str
    score: float
    document: str
    metadata: Dict


class ChromaRetriever:
    def __init__(self, persist_dir: Path):
        chroma_host = os.getenv("CHROMA_HOST", "").strip()
        chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
        if chroma_host:
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.models = EmbeddingModels()
        self.text_collection = self.client.get_collection("text_chunks")
        self.image_clip_collection = self.client.get_collection("image_clip")
        self.image_caption_collection = self.client.get_collection("image_caption")

    @staticmethod
    def _metadata_where(filters: Dict):
        clauses = []
        if filters.get("course"):
            clauses.append({"course": {"$eq": filters["course"]}})
        if filters.get("week"):
            clauses.append({"week": {"$eq": filters["week"]}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _flatten_result(result) -> List[RetrievedItem]:
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        out = []
        for i, doc_id in enumerate(ids):
            # chroma distance: lower is better. convert to similarity-ish score.
            dist = float(dists[i]) if i < len(dists) else 1.0
            out.append(
                RetrievedItem(
                    id=doc_id,
                    score=1.0 / (1.0 + dist),
                    document=docs[i] if i < len(docs) else "",
                    metadata=metas[i] if i < len(metas) else {},
                )
            )
        return out

    def query_text(self, query: str, top_k: int = 5, filters: Dict = None) -> List[RetrievedItem]:
        emb = self.models.embed_text(query)
        res = self.text_collection.query(query_embeddings=[emb], n_results=top_k, where=self._metadata_where(filters or {}))
        return self._flatten_result(res)

    def query_image(self, query: str, top_k: int = 5, filters: Dict = None) -> List[RetrievedItem]:
        emb = self.models.embed_clip_text(query)
        res = self.image_clip_collection.query(query_embeddings=[emb], n_results=top_k, where=self._metadata_where(filters or {}))
        return self._flatten_result(res)

    def query_caption(self, query: str, top_k: int = 5, filters: Dict = None) -> List[RetrievedItem]:
        emb = self.models.embed_text(query)
        res = self.image_caption_collection.query(
            query_embeddings=[emb], n_results=top_k, where=self._metadata_where(filters or {})
        )
        return self._flatten_result(res)

    def query_hybrid(self, query: str, top_k: int = 5, filters: Dict = None) -> List[RetrievedItem]:
        text = self.query_text(query, top_k=top_k, filters=filters)
        image = self.query_image(query, top_k=top_k, filters=filters)
        caption = self.query_caption(query, top_k=top_k, filters=filters)
        merged = {}
        for src_name, items, w in [("text", text, 0.45), ("image", image, 0.35), ("caption", caption, 0.2)]:
            for it in items:
                base = merged.get(it.id)
                if base is None:
                    merged[it.id] = RetrievedItem(it.id, w * it.score, it.document, dict(it.metadata))
                    merged[it.id].metadata["retrieval_source"] = src_name
                else:
                    base.score += w * it.score
        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)[:top_k]
        return ranked
