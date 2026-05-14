import hashlib
import os
from pathlib import Path
from typing import Any, List, Optional


def _hash_vector(text: str, dim: int = 384) -> List[float]:
    # Deterministic fallback when embedding models are unavailable.
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        b = seed[i % len(seed)]
        out.append((b / 255.0) - 0.5)
    return out


def _resolve_embed_device() -> str:
    """Pick torch device for SentenceTransformer (cuda / mps / cpu)."""
    pref = os.getenv("A3_EMBED_DEVICE", "auto").strip().lower()
    if pref in ("cuda", "cpu", "mps"):
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _encode_batch_size(default: int = 32) -> int:
    try:
        return max(1, int(os.getenv("A3_EMBED_BATCH_SIZE", str(default)).strip()))
    except ValueError:
        return default


class EmbeddingModels:
    def __init__(self) -> None:
        self.text_model = None
        self.clip_model = None
        self._device: Optional[str] = None

    @property
    def device_label(self) -> str:
        if self._device:
            return self._device
        return _resolve_embed_device()

    def _load_text_model(self):
        if self.text_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            dev = _resolve_embed_device()
            self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
            self._device = str(self.text_model.device)
        except Exception:
            self.text_model = None

    def _load_clip_model(self):
        if self.clip_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            dev = _resolve_embed_device()
            self.clip_model = SentenceTransformer("clip-ViT-B-32", device=dev)
            if self._device is None:
                self._device = str(self.clip_model.device)
        except Exception:
            self.clip_model = None

    def log_devices_once(self) -> None:
        """Load both models and print which device is used (call after construction)."""
        self._load_text_model()
        self._load_clip_model()
        tdev = getattr(self.text_model, "device", None) if self.text_model is not None else "unavailable"
        cdev = getattr(self.clip_model, "device", None) if self.clip_model is not None else "unavailable"
        print(f"[a3:embed] MiniLM device={tdev}; CLIP device={cdev}", flush=True)

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._load_text_model()
        if not texts:
            return []
        if self.text_model is None:
            return [_hash_vector(t) for t in texts]
        bs = min(_encode_batch_size(64), len(texts))
        arr = self.text_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=bs,
            show_progress_bar=False,
        )
        return arr.tolist()

    def embed_clip_text(self, text: str) -> List[float]:
        return self.embed_clip_texts([text])[0]

    def embed_clip_texts(self, texts: List[str]) -> List[List[float]]:
        self._load_clip_model()
        if not texts:
            return []
        if self.clip_model is None:
            return [_hash_vector(f"clip:{t}", dim=512) for t in texts]
        bs = min(_encode_batch_size(64), len(texts))
        arr = self.clip_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=bs,
            show_progress_bar=False,
        )
        return arr.tolist()

    def embed_clip_image_or_text_proxy(self, image_path: Path, text_proxy: str) -> List[float]:
        # sentence-transformers CLIP can encode images, but keep robust fallback.
        self._load_clip_model()
        if self.clip_model is None:
            return _hash_vector(f"img:{image_path}:{text_proxy}", dim=512)
        try:
            from PIL import Image

            img = Image.open(image_path)
            return self.embed_clip_pils([img], [text_proxy])[0]
        except Exception:
            return self.embed_clip_text(text_proxy)

    def embed_clip_pil(self, img: Any, text_proxy: str = "") -> List[float]:
        """Encode a PIL Image (e.g. rendered PDF page) with CLIP; fallback to text_proxy or hash."""
        return self.embed_clip_pils([img], [text_proxy] if text_proxy else None)[0]

    def embed_clip_pils(
        self,
        images: List[Any],
        text_proxies: Optional[List[str]] = None,
    ) -> List[List[float]]:
        """Batch CLIP image encoding (faster than one forward per page on GPU)."""
        self._load_clip_model()
        if not images:
            return []
        if self.clip_model is None:
            proxies = text_proxies or [""] * len(images)
            return [_hash_vector(f"clip:pil:{p}", dim=512) for p in proxies]
        bs = min(_encode_batch_size(32), len(images))
        try:
            arr = self.clip_model.encode(
                images,
                normalize_embeddings=True,
                batch_size=bs,
                show_progress_bar=False,
            )
            return arr.tolist()
        except Exception:
            out: List[List[float]] = []
            for i, img in enumerate(images):
                proxy = (text_proxies[i] if text_proxies and i < len(text_proxies) else "") or ""
                try:
                    row = self.clip_model.encode(
                        [img],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    out.append(row[0].tolist())
                except Exception:
                    out.append(self.embed_clip_text(proxy) if proxy else _hash_vector("clip:pil:fail", dim=512))
            return out
