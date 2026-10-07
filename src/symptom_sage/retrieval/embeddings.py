"""Text embedders: an offline hashing embedder and an OpenAI-compatible HTTP embedder."""
from __future__ import annotations

from typing import Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from symptom_sage.lexicon import find_symptoms


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _with_symptoms(text: str) -> str:
    """Add canonical symptom tokens, so aliases ("short of breath") meet ("shortness of breath")."""
    canon = " ".join("SYM_" + s.replace(" ", "_") for s in sorted(find_symptoms(text)))
    return f"{text} {canon} {canon}"


class HashingEmbedder:
    """Offline. Word 1-2 grams plus canonical symptom tokens, hashed to 4096 dimensions, L2 norm."""

    name = "hashing-4096"
    dim = 4096

    def __init__(self) -> None:
        self._vec = HashingVectorizer(n_features=self.dim, ngram_range=(1, 2), alternate_sign=False, norm="l2",
                                      lowercase=False, token_pattern=r"(?u)\b\w[\w-]*\b")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform([_with_symptoms(t) for t in texts]).toarray().astype(np.float32)


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", dim: int = 1536,
                 client: httpx.Client | None = None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; use SYMPTOM_SAGE_EMBEDDER=hashing for offline runs")
        self.name, self.dim = f"openai:{model}", dim
        self._model = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=60.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            r = self._client.post(self._url, json={"model": self._model, "input": texts[i:i + 256]},
                                  headers=self._headers)
            r.raise_for_status()
            out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
