"""Vector indexes with idempotent upserts keyed by the case id.

* :class:`LocalIndex` - numpy in memory, saved as ``vectors.npy`` + ``docs.json``.
* :class:`PineconeIndex` - optional extra ``pinecone``. It creates the index only if it is
  missing and upserts in batches. The full retrieval document is in the metadata, so the
  assessment reaches the prompt.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, docs: list[dict]) -> int: ...

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float, dict]]: ...

    def __len__(self) -> int: ...


class LocalIndex:
    def __init__(self, dim: int, embedder_name: str):
        self.dim, self.embedder_name = dim, embedder_name
        self.ids: list[str] = []
        self.docs: dict[str, dict] = {}
        self._vecs = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def upsert(self, ids: list[str], vectors: np.ndarray, docs: list[dict]) -> int:
        if vectors.shape != (len(ids), self.dim):
            raise ValueError(f"vectors must have shape ({len(ids)}, {self.dim}), got {vectors.shape}")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in one upsert")
        pos = {i: n for n, i in enumerate(self.ids)}
        new_rows = []
        for i, v, d in zip(ids, vectors, docs):
            if i in pos:
                self._vecs[pos[i]] = v
            else:
                self.ids.append(i)
                new_rows.append(v)
            self.docs[i] = d
        if new_rows:
            self._vecs = np.vstack([self._vecs, np.asarray(new_rows, dtype=np.float32)])
        return len(ids)

    def query(self, vector: np.ndarray, k: int, exclude: set[str] | None = None) -> list[tuple[str, float, dict]]:
        if not self.ids:
            return []
        sims = self._vecs @ vector.astype(np.float32)
        order = np.argsort(-sims, kind="stable")
        out = []
        for n in order:
            cid = self.ids[n]
            if exclude and cid in exclude:
                continue
            out.append((cid, float(sims[n]), self.docs[cid]))
            if len(out) == k:
                break
        return out

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self._vecs)
        meta = {"dim": self.dim, "embedder": self.embedder_name, "ids": self.ids, "docs": self.docs}
        (folder / "docs.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: str | Path, embedder_name: str | None = None) -> "LocalIndex":
        folder = Path(folder)
        meta = json.loads((folder / "docs.json").read_text(encoding="utf-8"))
        if embedder_name and meta["embedder"] != embedder_name:
            raise ValueError(f"index was built with {meta['embedder']}, not {embedder_name}; build it again")
        idx = cls(meta["dim"], meta["embedder"])
        idx.ids, idx.docs = meta["ids"], meta["docs"]
        idx._vecs = np.load(folder / "vectors.npy")
        return idx


class PineconeIndex:  # pragma: no cover - needs the optional extra and a network service
    def __init__(self, name: str, dim: int, api_key: str, cloud: str = "aws", region: str = "us-east-1"):
        try:
            from pinecone import Pinecone, ServerlessSpec  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError('Pinecone needs the "pinecone" extra: pip install -e ".[pinecone]"') from exc
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")
        pc = Pinecone(api_key=api_key)
        if name not in pc.list_indexes().names():
            pc.create_index(name=name, dimension=dim, metric="cosine", spec=ServerlessSpec(cloud=cloud, region=region))
        self._index = pc.Index(name)
        self.dim = dim

    def __len__(self) -> int:
        return int(self._index.describe_index_stats().get("total_vector_count", 0))

    def upsert(self, ids: list[str], vectors: np.ndarray, docs: list[dict]) -> int:
        items = [{"id": i, "values": v.tolist(), "metadata": d} for i, v, d in zip(ids, vectors, docs)]
        for s in range(0, len(items), 100):
            self._index.upsert(vectors=items[s:s + 100])
        return len(items)

    def query(self, vector: np.ndarray, k: int, exclude: set[str] | None = None) -> list[tuple[str, float, dict]]:
        res = self._index.query(vector=vector.tolist(), top_k=k + len(exclude or ()), include_metadata=True)
        out = [(m["id"], float(m["score"]), dict(m["metadata"])) for m in res["matches"]
               if not exclude or m["id"] not in exclude]
        return out[:k]
