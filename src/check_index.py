#!/usr/bin/env python3
"""
Verify Chroma indexes after offline_index: collection counts, optional metadata breakdown,
and a tiny retrieval smoke test (requires sentence-transformers for query embedding).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import chromadb


def _client(persist_dir: Path):
    chroma_host = os.getenv("CHROMA_HOST", "").strip()
    chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=chroma_port)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _aggregate_meta_field(collection, field: str, batch: int = 2000) -> Counter:
    c: Counter = Counter()
    offset = 0
    while True:
        res = collection.get(include=["metadatas"], limit=batch, offset=offset)
        ids = res.get("ids") or []
        if not ids:
            break
        for m in res.get("metadatas") or []:
            if not m:
                continue
            c[str(m.get(field, "") or "(empty)")] += 1
        offset += len(ids)
        if len(ids) < batch:
            break
    return c


def _smoke_query_text(collection, models) -> bool:
    try:
        emb = models.embed_text("project management baseline")
        r = collection.query(query_embeddings=[emb], n_results=1, include=["distances"])
        return bool(r.get("ids") and r["ids"][0])
    except Exception as e:
        print(f"  smoke query failed: {e}")
        return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check Chroma index status (same env as offline_index / retriever).")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=root / "data" / "chroma",
        help="Chroma persist directory when CHROMA_HOST is unset (default: <repo>/data/chroma).",
    )
    parser.add_argument("--min-text", type=int, default=0, help="Fail if text_chunks count below this.")
    parser.add_argument("--min-clip", type=int, default=0, help="Fail if image_clip count below this.")
    parser.add_argument("--no-smoke", action="store_true", help="Skip retrieval smoke test.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()

    out: dict = {"ok": True, "errors": []}

    try:
        client = _client(args.persist_dir)
    except Exception as e:
        print(f"ERROR: cannot connect to Chroma: {e}", file=sys.stderr)
        return 1

    names = ("text_chunks", "image_clip", "image_caption")
    stats = {}
    for name in names:
        try:
            col = client.get_collection(name)
            n = int(col.count())
            stats[name] = {"count": n}
        except Exception as e:
            stats[name] = {"count": 0, "error": str(e)}
            out["ok"] = False
            out["errors"].append(f"missing or broken collection {name}: {e}")

    if not args.json:
        host = os.getenv("CHROMA_HOST", "").strip() or "(local file)"
        port = os.getenv("CHROMA_PORT", "8001")
        print(f"Chroma backend: host={host!r} port={port}")
        if not os.getenv("CHROMA_HOST", "").strip():
            print(f"Persist dir: {args.persist_dir.resolve()}")
        print()

    for name in names:
        row = stats[name]
        if "error" in row:
            print(f"  {name}: ERROR — {row['error']}")
        else:
            print(f"  {name}: {row['count']} items")

    text_n = stats.get("text_chunks", {}).get("count", 0)
    clip_n = stats.get("image_clip", {}).get("count", 0)

    if text_n < args.min_text:
        out["ok"] = False
        out["errors"].append(f"text_chunks count {text_n} < --min-text {args.min_text}")
    if clip_n < args.min_clip:
        out["ok"] = False
        out["errors"].append(f"image_clip count {clip_n} < --min-clip {args.min_clip}")

    if not out["errors"] and stats.get("text_chunks", {}).get("count", 0) == 0:
        out["ok"] = False
        out["errors"].append("text_chunks is empty (run offline_index with data)")

    if not args.json and "error" not in stats.get("text_chunks", {}):
        try:
            tc = client.get_collection("text_chunks")
            by_source = _aggregate_meta_field(tc, "source")
            if by_source:
                print("\ntext_chunks by metadata['source']:")
                for k, v in sorted(by_source.items(), key=lambda x: -x[1]):
                    print(f"    {k!r}: {v}")
        except Exception as e:
            print(f"\n  (could not scan text metadata: {e})")

    if not args.json and "error" not in stats.get("image_clip", {}):
        try:
            ic = client.get_collection("image_clip")
            by_source = _aggregate_meta_field(ic, "source")
            if by_source:
                print("\nimage_clip by metadata['source']:")
                for k, v in sorted(by_source.items(), key=lambda x: -x[1]):
                    print(f"    {k!r}: {v}")
        except Exception as e:
            print(f"\n  (could not scan image metadata: {e})")

    smoke_ok = None
    if not args.no_smoke and "error" not in stats.get("text_chunks", {}) and text_n > 0:
        if not args.json:
            print("\nSmoke test: query text_chunks with embedding …")
        try:
            try:
                from pipeline.embeddings import EmbeddingModels
            except ImportError:
                from src.pipeline.embeddings import EmbeddingModels

            models = EmbeddingModels()
            tc = client.get_collection("text_chunks")
            smoke_ok = _smoke_query_text(tc, models)
            if not args.json:
                print(f"  {'OK' if smoke_ok else 'FAILED'} (top-1 hit)")
            if not smoke_ok:
                out["ok"] = False
                out["errors"].append("text smoke query returned no hits")
        except Exception as e:
            smoke_ok = False
            out["ok"] = False
            out["errors"].append(f"smoke test exception: {e}")
            if not args.json:
                print(f"  FAILED: {e}")

    out["collections"] = stats
    out["smoke_query_text"] = smoke_ok

    if args.json:
        print(json.dumps(out, indent=2))
    elif out["errors"]:
        print("\nProblems:")
        for e in out["errors"]:
            print(f"  - {e}")
        print("\nRESULT: NOT OK")
        return 2

    if not args.json:
        print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
