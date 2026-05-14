import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Bump when PDF / JSONL indexing semantics change so existing Chroma data is invalidated.
INDEX_PIPELINE_VERSION = "pdf3way-manifest-v1"

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


def _manifest_path(persist_dir: Path) -> Path:
    return persist_dir / "index_manifest.json"


def _read_index_manifest(persist_dir: Path) -> Dict:
    path = _manifest_path(persist_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_index_manifest(persist_dir: Path, fingerprint: str, stats: Dict) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "pipeline_version": INDEX_PIPELINE_VERSION,
        "source_fingerprint": fingerprint,
        "updated_unix": int(time.time()),
        "text_chunks": stats.get("text_chunks"),
        "image_docs": stats.get("image_docs"),
        "pdf_files_indexed": stats.get("pdf_files_indexed"),
        "jsonl_text_chunks": stats.get("jsonl_text_chunks"),
        "jsonl_image_docs": stats.get("jsonl_image_docs"),
    }
    _manifest_path(persist_dir).write_text(json.dumps(doc, indent=2), encoding="utf-8")


def compute_source_fingerprint(
    data_dir: Path,
    raw_subdir: str = "raw_data",
    llava_model: str = "",
) -> str:
    """Stable hash over pipeline version, env flags, and mtimes/sizes of all index inputs."""
    parts: List[str] = [
        INDEX_PIPELINE_VERSION,
        f"raw_subdir={raw_subdir}",
        f"A3_SKIP_PDF_LLAVA={os.getenv('A3_SKIP_PDF_LLAVA', '').strip().lower()}",
        f"OLLAVA_MODEL={llava_model}",
    ]
    file_rows: List[str] = []
    raw_dir = data_dir / raw_subdir
    if raw_dir.is_dir():
        for p in sorted({q.resolve() for q in raw_dir.rglob("*.pdf") if q.is_file()}):
            try:
                st = p.stat()
                rel = p.relative_to(data_dir).as_posix()
                file_rows.append(f"{rel}|{st.st_mtime_ns}|{st.st_size}")
            except (OSError, ValueError):
                continue
    for rel in ("sample/text_docs.jsonl", "sample/image_docs.jsonl"):
        p = data_dir / rel
        if p.is_file():
            try:
                st = p.stat()
                file_rows.append(f"{rel}|{st.st_mtime_ns}|{st.st_size}")
            except OSError:
                continue
    parts.extend(sorted(file_rows))
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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
    embs = models.embed_texts(batch_docs)
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
    b_meta: List[Dict],
    *,
    b_clip_embs: Optional[List[List[float]]] = None,
    b_clip_pils: Optional[List[Any]] = None,
    b_clip_pil_proxies: Optional[List[str]] = None,
) -> None:
    if not b_ids:
        return
    if b_clip_pils is not None and len(b_clip_pils) == len(b_ids):
        clip_embs = models.embed_clip_pils(b_clip_pils, b_clip_pil_proxies)
        b_clip_pils.clear()
        if b_clip_pil_proxies is not None:
            b_clip_pil_proxies.clear()
    elif b_clip_embs is not None and len(b_clip_embs) == len(b_ids):
        clip_embs = b_clip_embs[:]
        b_clip_embs.clear()
    else:
        return
    cap_embs = models.embed_texts(b_caps)
    clip_col.add(ids=b_ids, documents=b_caps, embeddings=clip_embs, metadatas=b_meta)
    cap_col.add(ids=b_ids, documents=b_caps, embeddings=cap_embs, metadatas=b_meta)
    b_ids.clear()
    b_caps.clear()
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
    print(
        f"[a3:index] JSONL: {len(text_docs)} text rows, {len(image_docs)} image rows → encoding + Chroma flush …",
        flush=True,
    )

    text_count = 0
    t_ids: List[str] = []
    t_docs: List[str] = []
    t_meta: List[Dict] = []

    for d in tqdm(text_docs, desc="JSONL text", unit="doc"):
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
    print(f"[a3:index] JSONL text done: {text_count} chunk(s); starting image rows …", flush=True)

    image_count = 0
    i_ids: List[str] = []
    i_caps: List[str] = []
    i_clip: List[List[float]] = []
    i_meta: List[Dict] = []

    for d in tqdm(image_docs, desc="JSONL images", unit="img"):
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
            _flush_image_batch(
                image_clip_collection,
                image_caption_collection,
                models,
                i_ids,
                i_caps,
                i_meta,
                b_clip_embs=i_clip,
            )
    _flush_image_batch(
        image_clip_collection,
        image_caption_collection,
        models,
        i_ids,
        i_caps,
        i_meta,
        b_clip_embs=i_clip,
    )
    print(f"[a3:index] JSONL images done: {image_count} doc(s).", flush=True)

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
    llava_endpoint: str = "http://localhost:11434/api/generate",
    llava_model: str = "llava:7b",
) -> Tuple[int, int, int]:
    """Index slide PDFs (three paths): page raster -> CLIP; page text -> text_chunks; page raster -> LLaVA -> caption text index."""
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

    total_pages = 0
    for p in pdfs:
        doc = fitz.open(p)
        try:
            total_pages += len(doc)
        finally:
            doc.close()

    skip_pdf_llava = os.getenv("A3_SKIP_PDF_LLAVA", "").strip().lower() in ("1", "true", "yes")
    llava_timeout_s = float(os.getenv("A3_LLAVA_TIMEOUT_S", "120"))
    print(
        f"[a3:index] raw_data PDFs: {len(pdfs)} file(s), {total_pages} page(s); per page: text+CLIP"
        f"{' + LLaVA caption' if not skip_pdf_llava else ' (LLaVA skipped)'} "
        f"(LLaVA timeout {llava_timeout_s:.0f}s/page). CLIP batched every {image_batch_size} page(s).",
        flush=True,
    )

    text_added = 0
    slides_added = 0
    t_ids: List[str] = []
    t_docs: List[str] = []
    t_meta: List[Dict] = []

    i_ids: List[str] = []
    i_caps: List[str] = []
    i_pils: List[Any] = []
    i_clip_proxies: List[str] = []
    i_meta: List[Dict] = []

    with tqdm(total=total_pages, desc="PDF pages", unit="page") as page_pbar:
        for pdf_path in pdfs:
            prefix = _pdf_id_prefix(pdf_path)
            course, week = _infer_course_week_from_stem(pdf_path.stem)
            pdf_name = pdf_path.name
            doc = fitz.open(pdf_path)
            n_pages = len(doc)
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
                    fallback_caption = page_text[:2000] if page_text else f"Slide {page_num} ({pdf_name})"
                    llava_caption = ""
                    if not skip_pdf_llava:
                        tmp_path: Optional[Path] = None
                        try:
                            fd, tmp_name = tempfile.mkstemp(suffix=".png", prefix="a3_pdf_page_")
                            os.close(fd)
                            tmp_path = Path(tmp_name)
                            pil_img.save(tmp_path, format="PNG")
                            if page_num == 1 or page_num % 5 == 0:
                                print(
                                    f"[a3:index]     → LLaVA {llava_model} {pdf_name} p.{page_num}/{n_pages} "
                                    f"(≤{llava_timeout_s:.0f}s) …",
                                    flush=True,
                                )
                            llava_caption = caption_with_llava(
                                tmp_path,
                                endpoint=llava_endpoint,
                                model=llava_model,
                                timeout_s=llava_timeout_s,
                            )
                        except OSError:
                            llava_caption = ""
                        finally:
                            if tmp_path is not None:
                                try:
                                    tmp_path.unlink(missing_ok=True)
                                except OSError:
                                    pass
                    caption = llava_caption.strip() if llava_caption.strip() else fallback_caption
                    slide_id = base_id
                    meta_img = {
                        "modality": "image",
                        "course": course,
                        "week": week,
                        "source_image": pdf_name,
                        "related_text_id": base_id,
                        "page": str(page_num),
                        "source": "pdf_page_raster",
                        "caption_source": "llava" if (llava_caption and llava_caption.strip()) else "fallback_text",
                    }
                    i_ids.append(slide_id)
                    i_caps.append(caption)
                    i_pils.append(pil_img)
                    i_clip_proxies.append(fallback_caption)
                    i_meta.append(meta_img)
                    slides_added += 1
                    if len(i_ids) >= image_batch_size:
                        _flush_image_batch(
                            image_clip_collection,
                            image_caption_collection,
                            models,
                            i_ids,
                            i_caps,
                            i_meta,
                            b_clip_pils=i_pils,
                            b_clip_pil_proxies=i_clip_proxies,
                        )
                    page_pbar.update(1)
            finally:
                doc.close()
            print(f"[a3:index]   finished {pdf_name} ({n_pages} pages).", flush=True)

    _flush_text_batch(text_collection, models, t_ids, t_docs, t_meta)
    _flush_image_batch(
        image_clip_collection,
        image_caption_collection,
        models,
        i_ids,
        i_caps,
        i_meta,
        b_clip_pils=i_pils if i_ids else None,
        b_clip_pil_proxies=i_clip_proxies if i_ids else None,
    )

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
    print("[a3:index] Chroma client ready; computing source fingerprint …", flush=True)
    models = EmbeddingModels()
    models.log_devices_once()
    print(
        "[a3:index] JSONL + PDF indexing. Tip: A3_EMBED_DEVICE=cuda|cpu|mps, A3_EMBED_BATCH_SIZE=32.",
        flush=True,
    )

    fingerprint = compute_source_fingerprint(data_dir, raw_subdir, llava_model=llava_model)
    manifest = _read_index_manifest(persist_dir)

    text_collection = _ensure_collection(client, "text_chunks")
    image_clip_collection = _ensure_collection(client, "image_clip")
    image_caption_collection = _ensure_collection(client, "image_caption")

    try:
        tc = int(text_collection.count())
        ic_clip = int(image_clip_collection.count())
        ic_cap = int(image_caption_collection.count())
    except Exception:
        tc, ic_clip, ic_cap = 0, 0, 0

    manifest_ok = (
        (not rebuild)
        and manifest.get("pipeline_version") == INDEX_PIPELINE_VERSION
        and manifest.get("source_fingerprint") == fingerprint
        and tc > 0
        and ic_clip > 0
        and ic_cap > 0
    )

    if manifest_ok:
        print("[a3:index] Manifest matches sources; skipping full rebuild.", flush=True)
        return {
            "text_chunks": tc,
            "image_docs": ic_clip,
            "persist_dir": str(persist_dir),
            "rebuilt": False,
            "source_fingerprint": fingerprint,
            "index_manifest_ok": True,
        }

    print("[a3:index] Full rebuild: clearing Chroma collections …", flush=True)
    for name in ("text_chunks", "image_clip", "image_caption"):
        _delete_collection_safe(client, name)
    text_collection = _ensure_collection(client, "text_chunks")
    image_clip_collection = _ensure_collection(client, "image_clip")
    image_caption_collection = _ensure_collection(client, "image_caption")

    print("[a3:index] Indexing sample JSONL …", flush=True)
    jsonl_text, jsonl_images = _index_from_jsonl(
        data_dir,
        models,
        text_collection,
        image_clip_collection,
        image_caption_collection,
        llava_endpoint,
        llava_model,
    )
    print(
        f"[a3:index] JSONL indexed: text_chunks={jsonl_text}, image_docs={jsonl_images}. "
        "Starting raw_data PDFs …",
        flush=True,
    )
    pdf_text, pdf_slides, pdf_files = _index_from_raw_pdfs(
        data_dir,
        models,
        text_collection,
        image_clip_collection,
        image_caption_collection,
        raw_subdir=raw_subdir,
        llava_endpoint=llava_endpoint,
        llava_model=llava_model,
    )

    text_total = jsonl_text + pdf_text
    image_total = jsonl_images + pdf_slides

    stats = {
        "text_chunks": text_total,
        "image_docs": image_total,
        "jsonl_text_chunks": jsonl_text,
        "jsonl_image_docs": jsonl_images,
        "pdf_text_chunks": pdf_text,
        "pdf_slide_pages": pdf_slides,
        "pdf_files_indexed": pdf_files,
        "persist_dir": str(persist_dir),
        "rebuilt": True,
        "source_fingerprint": fingerprint,
        "index_manifest_ok": True,
    }
    _write_index_manifest(persist_dir, fingerprint, stats)
    print(
        f"[a3:index] Done. text_chunks={text_total}, image_entries={image_total}, "
        f"pdf_files={pdf_files}. Writing manifest.",
        flush=True,
    )
    return stats


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
