import json
import os
from pathlib import Path
from typing import Dict, List

import chromadb

from .captioning import caption_with_llava
from .chunking import chunk_text
from .embeddings import EmbeddingModels


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _ensure_collection(client, name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)


def build_offline_index(
    data_dir: Path,
    persist_dir: Path,
    llava_endpoint: str = "http://localhost:11434/api/generate",
    llava_model: str = "llava:7b",
) -> Dict:
    chroma_host = os.getenv("CHROMA_HOST", "").strip()
    chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
    if chroma_host:
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
    models = EmbeddingModels()

    text_collection = _ensure_collection(client, "text_chunks")
    image_clip_collection = _ensure_collection(client, "image_clip")
    image_caption_collection = _ensure_collection(client, "image_caption")

    # Fast path: skip expensive rebuild if index already exists.
    try:
        existing_count = int(text_collection.count())
    except Exception:
        existing_count = 0
    if existing_count > 0:
        return {
            "text_chunks": existing_count,
            "image_docs": int(image_clip_collection.count()),
            "persist_dir": str(persist_dir),
            "rebuilt": False,
        }

    text_docs = _load_jsonl(data_dir / "sample" / "text_docs.jsonl")
    image_docs = _load_jsonl(data_dir / "sample" / "image_docs.jsonl")

    # Text chunk indexing.
    text_count = 0
    for d in text_docs:
        source_text = d.get("content", "")
        chunks = chunk_text(source_text, chunk_size=900, overlap=150)
        if not chunks:
            continue
        for idx, chunk in enumerate(chunks, start=1):
            doc_id = f"{d['id']}::chunk{idx}"
            metadata = {
                "modality": "text",
                "course": d.get("course", ""),
                "week": d.get("week", ""),
                "source_file": d.get("source_file", ""),
                "base_id": d["id"],
            }
            text_collection.add(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[models.embed_text(chunk)],
                metadatas=[metadata],
            )
            text_count += 1

    # Image dual-indexing.
    image_count = 0
    for d in image_docs:
        caption = d.get("caption", "").strip()
        src = d.get("source_image") or d.get("source") or ""
        image_path = (data_dir / src).resolve() if src else None
        if (not caption) and image_path and image_path.exists():
            generated = caption_with_llava(image_path, endpoint=llava_endpoint, model=llava_model)
            if generated:
                caption = generated
        if not caption:
            caption = f"{d.get('title','image')} {' '.join(d.get('topic_tags', []))}".strip()

        img_id = d["id"]
        metadata = {
            "modality": "image",
            "course": d.get("course", ""),
            "week": d.get("week", ""),
            "source_image": src,
            "related_text_id": d.get("related_text_id", ""),
        }
        # Caption text index.
        image_caption_collection.add(
            ids=[img_id],
            documents=[caption],
            embeddings=[models.embed_text(caption)],
            metadatas=[metadata],
        )
        # CLIP image index (or text proxy fallback).
        clip_emb = models.embed_clip_image_or_text_proxy(image_path, caption) if image_path else models.embed_clip_text(caption)
        image_clip_collection.add(
            ids=[img_id],
            documents=[caption],
            embeddings=[clip_emb],
            metadatas=[metadata],
        )
        image_count += 1

    return {
        "text_chunks": text_count,
        "image_docs": image_count,
        "persist_dir": str(persist_dir),
        "rebuilt": True,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    stats = build_offline_index(
        data_dir=root / "data",
        persist_dir=root / "data" / "chroma",
    )
    print(json.dumps(stats, indent=2))
