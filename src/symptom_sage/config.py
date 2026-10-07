"""Settings from environment variables. Every default works offline with no key."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default="", repr=False)
    pinecone_api_key: str = field(default="", repr=False)
    llm: str = "fake"  # fake | openai
    embedder: str = "hashing"  # hashing | openai
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    openai_base_url: str = "https://api.openai.com/v1"
    index_dir: Path = Path("index")
    index_backend: str = "local"  # local | pinecone
    pinecone_index: str = "symptom-sage-cases"
    top_k: int = 5
    min_score: float = 0.05
    whisper_model: str = "base"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            openai_api_key=_get("OPENAI_API_KEY"),
            pinecone_api_key=_get("PINECONE_API_KEY"),
            llm=_get("SYMPTOM_SAGE_LLM", "fake").lower(),
            embedder=_get("SYMPTOM_SAGE_EMBEDDER", "hashing").lower(),
            chat_model=_get("SYMPTOM_SAGE_CHAT_MODEL", "gpt-4o-mini"),
            embed_model=_get("SYMPTOM_SAGE_EMBED_MODEL", "text-embedding-3-small"),
            openai_base_url=_get("SYMPTOM_SAGE_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            index_dir=Path(_get("SYMPTOM_SAGE_INDEX_DIR", "index")),
            index_backend=_get("SYMPTOM_SAGE_INDEX_BACKEND", "local").lower(),
            pinecone_index=_get("SYMPTOM_SAGE_PINECONE_INDEX", "symptom-sage-cases"),
            top_k=int(_get("SYMPTOM_SAGE_TOP_K", "5")),
            min_score=float(_get("SYMPTOM_SAGE_MIN_SCORE", "0.05")),
            whisper_model=_get("SYMPTOM_SAGE_WHISPER_MODEL", "base"),
        )
        for name, value, allowed in (("SYMPTOM_SAGE_LLM", s.llm, ("fake", "openai")),
                                     ("SYMPTOM_SAGE_EMBEDDER", s.embedder, ("hashing", "openai")),
                                     ("SYMPTOM_SAGE_INDEX_BACKEND", s.index_backend, ("local", "pinecone"))):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if s.top_k < 1:
            raise ValueError("SYMPTOM_SAGE_TOP_K must be >= 1")
        return s
