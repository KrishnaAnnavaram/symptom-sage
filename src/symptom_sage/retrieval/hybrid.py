"""Hybrid retrieval: BM25 over the case documents plus dense cosine, fused by reciprocal rank."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from symptom_sage.lexicon import find_symptoms
from symptom_sage.retrieval.embeddings import Embedder
from symptom_sage.retrieval.index import LocalIndex

RRF_K = 60
STOP = frozenset("a an the and or of to in on for with i my me have has had is am are was it this that".split())


def tokenize(text: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9][a-z0-9-]*", text.lower()) if t not in STOP]
    return toks + ["sym_" + s.replace(" ", "_") for s in sorted(find_symptoms(text))]


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tf = [Counter(tokenize(docs[i])) for i in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for c in self.tf for t in c)
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        q = tokenize(query)
        out = {}
        for i, tf, ln in zip(self.ids, self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out[i] = s
        return out


@dataclass(frozen=True)
class Hit:
    case_id: str
    score: float  # fused RRF score
    dense: float
    bm25: float
    doc: dict


class HybridRetriever:
    def __init__(self, index: LocalIndex, embedder: Embedder):
        self.index, self.embedder = index, embedder
        self.bm25 = BM25({cid: index.docs[cid]["text"] for cid in index.ids})

    def search(self, query: str, k: int = 5, exclude: set[str] | None = None) -> list[Hit]:
        exclude = exclude or set()
        n = len(self.index)
        dense = self.index.query(self.embedder.embed([query])[0], n, exclude=exclude)
        bm = self.bm25.scores(query)
        bm_rank = sorted((c for c in bm if c not in exclude), key=lambda c: -bm[c])
        fused: dict[str, float] = {}
        for r, (cid, _, _) in enumerate(dense):
            fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + r + 1)
        for r, cid in enumerate(bm_rank):
            if bm[cid] > 0:
                fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + r + 1)
        dense_score = {cid: s for cid, s, _ in dense}
        top = sorted(fused, key=lambda c: (-fused[c], c))[:k]
        return [Hit(c, fused[c], dense_score.get(c, 0.0), bm.get(c, 0.0), self.index.docs[c]) for c in top]


class DenseRetriever:
    """Dense-only retrieval for remote indexes (for example Pinecone) that hold no local BM25 text."""

    def __init__(self, index, embedder: Embedder):
        self.index, self.embedder = index, embedder

    def search(self, query: str, k: int = 5, exclude: set[str] | None = None) -> list[Hit]:
        res = self.index.query(self.embedder.embed([query])[0], k, exclude=exclude)
        return [Hit(cid, s, s, 0.0, doc) for cid, s, doc in res]
