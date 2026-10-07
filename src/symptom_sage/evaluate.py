"""Evaluation: extraction accuracy, leave-one-case-out retrieval, answer checks and red-team safety."""
from __future__ import annotations

import json
from collections import defaultdict
from importlib import resources

from symptom_sage.extraction import CaseRecord
from symptom_sage.llm import LLM
from symptom_sage.rag import Assistant
from symptom_sage.retrieval import HybridRetriever, build_index
from symptom_sage.retrieval.embeddings import Embedder
from symptom_sage.safety import RX_PATTERN


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def eval_extraction(records: list[CaseRecord], truth: dict[str, dict]) -> dict:
    """Field accuracy against ground truth (synthetic truth, or a hand-labelled sample)."""
    n = age = gender = cond = 0
    tp = fp = fn = 0
    for r in records:
        t = truth.get(r.case_id)
        if not t:
            continue
        n += 1
        age += r.age == t["age"]
        gender += r.gender == t["gender"]
        cond += _norm(r.condition) == _norm(t["condition"])
        ps, ts = set(r.symptoms), set(t["symptoms"])
        tp, fp, fn = tp + len(ps & ts), fp + len(ps - ts), fn + len(ts - ps)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"n": n, "age_acc": age / max(n, 1), "gender_acc": gender / max(n, 1), "condition_acc": cond / max(n, 1),
            "symptom_precision": prec, "symptom_recall": rec, "symptom_f1": 2 * prec * rec / max(prec + rec, 1e-12)}


def split_half(records: list[CaseRecord], seed: int = 0) -> tuple[list[CaseRecord], list[CaseRecord]]:
    """A seeded split, stratified by condition: about half of each condition on each side."""
    import random

    rng = random.Random(seed)
    groups: dict[str, list[CaseRecord]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r.case_id):
        groups[r.condition or ""].append(r)
    a, b = [], []
    for _, rs in sorted(groups.items()):
        rng.shuffle(rs)
        a += rs[: (len(rs) + 1) // 2]
        b += rs[(len(rs) + 1) // 2:]
    return a, b


def query_for(rec: CaseRecord) -> str:
    return "I have " + ", ".join(rec.symptoms) if rec.symptoms else rec.reason


def eval_retrieval(records: list[CaseRecord], embedder: Embedder, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Leave-one-case-out: query with a case's symptoms, with that case removed from the index.

    Hit@k = the true condition is among the conditions of the top-k retrieved cases. The
    baseline is chance: a random ranking of the distinct conditions (k / number of conditions).
    """
    usable = [r for r in records if r.condition and r.symptoms]
    index = build_index(usable, embedder)
    retriever = HybridRetriever(index, embedder)
    hits = {k: 0 for k in ks}
    n_cond = len({r.condition for r in usable})
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kmax = max(ks)
    for r in usable:
        res = retriever.search(query_for(r), k=kmax, exclude={r.case_id})
        conds = [h.doc["condition"] for h in res]
        for k in ks:
            hits[k] += r.condition in conds[:k]
        per_cat[r.category][0] += r.condition in conds[:1]
        per_cat[r.category][1] += 1
    n = max(len(usable), 1)
    return {"n": len(usable), **{f"hit@{k}": hits[k] / n for k in ks},
            **{f"chance_hit@{k}": min(k, n_cond) / max(n_cond, 1) for k in ks},
            "hit@1_by_category": {c: v[0] / v[1] for c, v in sorted(per_cat.items())}}


def eval_answers(index_records: list[CaseRecord], query_records: list[CaseRecord], embedder: Embedder,
                 llm: LLM, top_k: int = 5) -> dict:
    """Ask with held-out cases. Check the answer conditions, the citations and medicine leaks."""
    assistant = Assistant(HybridRetriever(build_index(index_records, embedder), embedder), llm, top_k=top_k)
    n = correct = cited_ok = leaks = answered = 0
    for r in query_records:
        if not r.condition or not r.symptoms:
            continue
        a = assistant.answer(query_for(r))
        n += 1
        if a.kind not in ("answer", "urgent"):
            continue
        answered += 1
        correct += r.condition in a.conditions
        cited_ok += not a.invalid_citations
        body = a.text.rsplit("\n\n", 1)[0]
        leaks += bool(RX_PATTERN.search(body))
    return {"n": n, "answered": answered, "condition_in_answer": correct / max(answered, 1),
            "valid_citations": cited_ok / max(answered, 1), "medicine_leaks": leaks}


def redteam_cases() -> list[dict]:
    return json.loads(resources.files("symptom_sage").joinpath("resources/redteam.json").read_text(encoding="utf-8"))


def eval_safety(assistant: Assistant) -> dict:
    rows = []
    for case in redteam_cases():
        a = assistant.answer(case["prompt"])
        ok = a.kind == case["expect"] or (case["expect"] == "answer" and a.kind == "no_match")
        rows.append({"prompt": case["prompt"], "expect": case["expect"], "got": a.kind, "pass": ok})
    passed = sum(r["pass"] for r in rows)
    return {"n": len(rows), "passed": passed, "pass_rate": passed / max(len(rows), 1),
            "failures": [r for r in rows if not r["pass"]]}
