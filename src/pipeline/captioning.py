import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def caption_with_llava(
    image_path: Path,
    endpoint: str = "http://localhost:11434/api/generate",
    model: str = "llava:7b",
    timeout_s: float = 40.0,
) -> str:
    if not image_path.exists():
        return ""
    try:
        raw = image_path.read_bytes()
        payload = {
            "model": model,
            "prompt": "Describe this image for retrieval. Mention key objects, scene type, and context in 1-2 sentences.",
            "images": [base64.b64encode(raw).decode("utf-8")],
            "stream": False,
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""
