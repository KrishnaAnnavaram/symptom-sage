"""Embeddings, vector indexes and the hybrid retriever."""
from __future__ import annotations

from symptom_sage.config import Settings
from symptom_sage.extraction import CaseRecord
from symptom_sage.retrieval.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from symptom_sage.retrieval.hybrid import HybridRetriever
from symptom_sage.retrieval.index import LocalIndex


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "openai":
        return OpenAIEmbedder(settings.embed_model, settings.openai_api_key, settings.openai_base_url)
    return HashingEmbedder()


def case_doc(rec: CaseRecord) -> dict:
    """Index metadata. ``text`` holds the assessment; the plan is left out on purpose."""
    return {"text": rec.document(), "category": rec.category, "condition": rec.condition or "",
            "symptoms": rec.symptoms, "age": rec.age, "gender": rec.gender}


def build_index(records: list[CaseRecord], embedder: Embedder, index: LocalIndex | None = None) -> LocalIndex:
    """Create or update a local index. A record with a known case id replaces the old one."""
    index = index or LocalIndex(embedder.dim, embedder.name)
    usable = [r for r in records if r.condition]
    if usable:
        docs = [case_doc(r) for r in usable]
        index.upsert([r.case_id for r in usable], embedder.embed([d["text"] for d in docs]), docs)
    return index


__all__ = ["HybridRetriever", "LocalIndex", "build_index", "case_doc", "make_embedder"]
