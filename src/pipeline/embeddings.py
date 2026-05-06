import hashlib
from pathlib import Path
from typing import Any, List


def _hash_vector(text: str, dim: int = 384) -> List[float]:
    # Deterministic fallback when embedding models are unavailable.
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        b = seed[i % len(seed)]
        out.append((b / 255.0) - 0.5)
    return out


class EmbeddingModels:
    def __init__(self) -> None:
        self.text_model = None
        self.clip_model = None

    def _load_text_model(self):
        if self.text_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.text_model = None

    def _load_clip_model(self):
        if self.clip_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.clip_model = SentenceTransformer("clip-ViT-B-32")
        except Exception:
            self.clip_model = None

    def embed_text(self, text: str) -> List[float]:
        self._load_text_model()
        if self.text_model is None:
            return _hash_vector(text)
        return self.text_model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_clip_text(self, text: str) -> List[float]:
        self._load_clip_model()
        if self.clip_model is None:
            return _hash_vector(f"clip:{text}", dim=512)
        return self.clip_model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_clip_image_or_text_proxy(self, image_path: Path, text_proxy: str) -> List[float]:
        # sentence-transformers CLIP can encode images, but keep robust fallback.
        self._load_clip_model()
        if self.clip_model is None:
            return _hash_vector(f"img:{image_path}:{text_proxy}", dim=512)
        try:
            from PIL import Image

            img = Image.open(image_path)
            emb = self.clip_model.encode([img], normalize_embeddings=True)[0]
            return emb.tolist()
        except Exception:
            return self.embed_clip_text(text_proxy)

    def embed_clip_pil(self, img: Any, text_proxy: str = "") -> List[float]:
        """Encode a PIL Image (e.g. rendered PDF page) with CLIP; fallback to text_proxy or hash."""
        self._load_clip_model()
        if self.clip_model is None:
            return _hash_vector(f"clip:pil:{text_proxy}", dim=512)
        try:
            return self.clip_model.encode([img], normalize_embeddings=True)[0].tolist()
        except Exception:
            if text_proxy:
                return self.embed_clip_text(text_proxy)
            return _hash_vector("clip:pil:fail", dim=512)
