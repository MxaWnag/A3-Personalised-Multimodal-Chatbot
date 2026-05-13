import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb

try:
    from .captioning import caption_with_llava
    from .chunking import chunk_text
    from .embeddings import EmbeddingModels
except ImportError:
    from captioning import caption_with_llava
    from chunking import chunk_text
    from embeddings import EmbeddingModels


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


def _delete_collection_safe(client, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass


def _infer_course_week_from_stem(stem: str) -> Tuple[str, str]:
    u = stem.upper()
    course = ""
    if "INFS7205" in u:
        course = "INFS7205"
    elif "ENGG4800" in u:
        course = "ENGG4800"
    week = ""
    m = re.search(r"(?i)lecture\s*[_ ]?\s*(\d{1,2})", stem)
    if m:
        week = f"week{int(m.group(1))}"
    else:
        m = re.search(r"(?i)week\s*[_ ]?\s*(\d{1,2})", stem)
        if m:
            week = f"week{int(m.group(1))}"
    return course, week


def _pdf_id_prefix(pdf_path: Path) -> str:
    stem = pdf_path.stem
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    digest = hashlib.sha256(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]
    base = f"{safe[:48]}_{digest}" if safe else digest
    return f"slides_{base}"


def _iter_raw_pdfs(raw_dir: Path) -> List[Path]:
    if not raw_dir.is_dir():
        return []
    return sorted({p.resolve() for p in raw_dir.rglob("*.pdf") if p.is_file()})


def _flush_text_batch(
    text_collection,
    models: EmbeddingModels,
    batch_ids: List[str],
    batch_docs: List[str],
    batch_meta: List[Dict],
) -> None:
    if not batch_ids:
        return
    embs = [models.embed_text(d) for d in batch_docs]
    text_collection.add(ids=batch_ids, documents=batch_docs, embeddings=embs, metadatas=batch_meta)
    batch_ids.clear()
    batch_docs.clear()
    batch_meta.clear()


def _flush_image_batch(
    clip_col,
    cap_col,
    models: EmbeddingModels,
    b_ids: List[str],
    b_caps: List[str],
    b_clip_embs: List[List[float]],
    b_meta: List[Dict],
) -> None:
    if not b_ids:
        return
    cap_embs = [models.embed_text(c) for c in b_caps]
    cap_col.add(ids=b_ids, documents=b_caps, embeddings=cap_embs, metadatas=b_meta)
    clip_col.add(ids=b_ids, documents=b_caps, embeddings=b_clip_embs, metadatas=b_meta)
    b_ids.clear()
    b_caps.clear()
    b_clip_embs.clear()
    b_meta.clear()


def _index_from_jsonl(
    data_dir: Path,
    models: EmbeddingModels,
    text_collection,
    image_clip_collection,
    image_caption_collection,
    llava_endpoint: str,
    llava_model: str,
    text_batch_size: int = 32,
) -> Tuple[int, int]:
    text_docs = _load_jsonl(data_dir / "sample" / "text_docs.jsonl")
    image_docs = _load_jsonl(data_dir / "sample" / "image_docs.jsonl")

    text_count = 0
    t_ids: List[str] = []
    t_docs: List[str] = []
    t_meta: List[Dict] = []

    for d in text_docs:
        source_text = d.get("content", "")
        chunks = chunk_text(source_text, chunk_size=900, overlap=150)
        if not chunks:
            continue
        for idx, chunk in enumerate(chunks, start=1):
            doc_id = f"{d['id']}::chunk{idx}"
            metadata = {
                "modality": "text",
                "course": str(d.get("course", "") or ""),
                "week": str(d.get("week", "") or ""),
                "source_file": str(d.get("source_file", "") or ""),
                "base_id": str(d["id"]),
                "source": "jsonl",
            }
            t_ids.append(doc_id)
            t_docs.append(chunk)
            t_meta.append(metadata)
            text_count += 1
            if len(t_ids) >= text_batch_size:
                _flush_text_batch(text_collection, models, t_ids, t_docs, t_meta)
    _flush_text_batch(text_collection, models, t_ids, t_docs, t_meta)

    image_count = 0
    i_ids: List[str] = []
    i_caps: List[str] = []
    i_clip: List[List[float]] = []
    i_meta: List[Dict] = []

    for d in image_docs:
        caption = d.get("caption", "").strip()
        src = d.get("source_image") or d.get("source") or ""
        image_path = (data_dir / src).resolve() if src else None
        if (not caption) and image_path and image_path.exists():
            generated = caption_with_llava(image_path, endpoint=llava_endpoint, model=llava_model)
            if generated:
                caption = generated
        if not caption:
            caption = f"{d.get('title', 'image')} {' '.join(d.get('topic_tags', []))}".strip()

        img_id = d["id"]
        metadata = {
            "modality": "image",
            "course": str(d.get("course", "") or ""),
            "week": str(d.get("week", "") or ""),
            "source_image": str(src or ""),
            "related_text_id": str(d.get("related_text_id", "") or ""),
            "source": "jsonl",
        }
        clip_emb = (
            models.embed_clip_image_or_text_proxy(image_path, caption)
            if image_path
            else models.embed_clip_text(caption)
        )
        i_ids.append(img_id)
        i_caps.append(caption)
        i_clip.append(clip_emb)
        i_meta.append(metadata)
        image_count += 1
        if len(i_ids) >= 32:
            _flush_image_batch(image_clip_collection, image_caption_collection, models, i_ids, i_caps, i_clip, i_meta)
    _flush_image_batch(image_clip_collection, image_caption_collection, models, i_ids, i_caps, i_clip, i_meta)

    return text_count, image_count


def _index_from_raw_pdfs(
    data_dir: Path,
    models: EmbeddingModels,
    text_collection,
    image_clip_collection,
    image_caption_collection,
    raw_subdir: str = "raw_data",
    render_zoom: float = 150.0 / 72.0,
    text_batch_size: int = 32,
    image_batch_size: int = 8,
) -> Tuple[int, int, int]:
    """Index slide PDFs: per-page raster -> CLIP; per-page text -> text_chunks."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF (fitz) not installed; skip raw_data PDF indexing. pip install PyMuPDF")
        return 0, 0, 0

    from PIL import Image

    raw_dir = data_dir / raw_subdir
    pdfs = _iter_raw_pdfs(raw_dir)
    if not pdfs:
        return 0, 0, 0

    text_added = 0
    slides_added = 0
    t_ids: List[str] = []
    t_docs: List[str] = []
    t_meta: List[Dict] = []

    i_ids: List[str] = []
    i_caps: List[str] = []
    i_clip: List[List[float]] = []
    i_meta: List[Dict] = []

    for pdf_path in pdfs:
        prefix = _pdf_id_prefix(pdf_path)
        course, week = _infer_course_week_from_stem(pdf_path.stem)
        pdf_name = pdf_path.name
        doc = fitz.open(pdf_path)
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_num = page_index + 1
                page_text = (page.get_text("text") or "").strip()
                chunks = chunk_text(page_text, chunk_size=900, overlap=150) if page_text else []

                base_id = f"{prefix}_p{page_num:04d}"
                for c_idx, chunk in enumerate(chunks, start=1):
                    doc_id = f"{base_id}_c{c_idx}"
                    metadata = {
                        "modality": "text",
                        "course": course,
                        "week": week,
                        "source_file": pdf_name,
                        "base_id": base_id,
                        "page": str(page_num),
                        "source": "pdf_page_text",
                    }
                    t_ids.append(doc_id)
                    t_docs.append(chunk)
                    t_meta.append(metadata)
                    text_added += 1
                    if len(t_ids) >= text_batch_size:
                        _flush_text_batch(text_collection, models, t_ids, t_docs, t_meta)

                mat = fitz.Matrix(render_zoom, render_zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                caption = page_text[:2000] if page_text else f"Slide {page_num} ({pdf_name})"
                clip_emb = models.embed_clip_pil(pil_img, text_proxy=caption)
                slide_id = base_id
                meta_img = {
                    "modality": "image",
                    "course": course,
                    "week": week,
                    "source_image": pdf_name,
                    "related_text_id": base_id,
                    "page": str(page_num),
                    "source": "pdf_page_raster",
                }
                i_ids.append(slide_id)
                i_caps.append(caption)
                i_clip.append(clip_emb)
                i_meta.append(meta_img)
                slides_added += 1
                if len(i_ids) >= image_batch_size:
                    _flush_image_batch(
                        image_clip_collection, image_caption_collection, models, i_ids, i_caps, i_clip, i_meta
                    )
        finally:
            doc.close()

    _flush_text_batch(text_collection, models, t_ids, t_docs, t_meta)
    _flush_image_batch(image_clip_collection, image_caption_collection, models, i_ids, i_caps, i_clip, i_meta)

    return text_added, slides_added, len(pdfs)


def build_offline_index(
    data_dir: Path,
    persist_dir: Path,
    llava_endpoint: Optional[str] = None,
    llava_model: Optional[str] = None,
    rebuild: bool = False,
    raw_subdir: str = "raw_data",
) -> Dict:
    if llava_endpoint is None:
        llava_endpoint = (
            os.getenv("OLLAVA_ENDPOINT", "").strip()
            or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate").strip()
        )
    if llava_model is None:
        llava_model = os.getenv("OLLAVA_MODEL", "llava:7b").strip()

    env_rebuild = os.getenv("A3_REBUILD_INDEX", "").strip().lower() in ("1", "true", "yes")
    rebuild = bool(rebuild or env_rebuild)

    chroma_host = os.getenv("CHROMA_HOST", "").strip()
    chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
    if chroma_host:
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
    models = EmbeddingModels()

    if rebuild:
        for name in ("text_chunks", "image_clip", "image_caption"):
            _delete_collection_safe(client, name)

    text_collection = _ensure_collection(client, "text_chunks")
    image_clip_collection = _ensure_collection(client, "image_clip")
    image_caption_collection = _ensure_collection(client, "image_caption")

    if not rebuild:
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

    jsonl_text, jsonl_images = _index_from_jsonl(
        data_dir,
        models,
        text_collection,
        image_clip_collection,
        image_caption_collection,
        llava_endpoint,
        llava_model,
    )
    pdf_text, pdf_slides, pdf_files = _index_from_raw_pdfs(
        data_dir,
        models,
        text_collection,
        image_clip_collection,
        image_caption_collection,
        raw_subdir=raw_subdir,
    )

    text_total = jsonl_text + pdf_text
    image_total = jsonl_images + pdf_slides

    return {
        "text_chunks": text_total,
        "image_docs": image_total,
        "jsonl_text_chunks": jsonl_text,
        "jsonl_image_docs": jsonl_images,
        "pdf_text_chunks": pdf_text,
        "pdf_slide_pages": pdf_slides,
        "pdf_files_indexed": pdf_files,
        "persist_dir": str(persist_dir),
        "rebuilt": True,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build Chroma indexes from sample JSONL and raw_data PDF slides.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing text_chunks / image_clip / image_caption collections and rebuild from scratch.",
    )
    parser.add_argument(
        "--raw-subdir",
        default="raw_data",
        help="Directory under data/ containing slide PDFs (default: raw_data).",
    )
    args = parser.parse_args()
    stats = build_offline_index(
        data_dir=root / "data",
        persist_dir=root / "data" / "chroma",
        rebuild=args.rebuild,
        raw_subdir=args.raw_subdir,
    )
    print(json.dumps(stats, indent=2))
