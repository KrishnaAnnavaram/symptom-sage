"""The online assistant: safety -> retrieval -> grounded answer -> checks.

The prompt holds the user question AND the retrieved cases with their assessments. The model
must cite case ids. After generation, the code removes citations to cases that were not
retrieved, removes sentences about medicines or doses, and adds the disclaimer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from symptom_sage.llm import LLM, LLMError
from symptom_sage.retrieval.hybrid import Hit
from symptom_sage.safety import (DISCLAIMER, Triage, is_in_scope, is_medication_request, strip_prescriptions,
                                 triage)

SYSTEM = (
    "You help a person understand which conditions were assessed in simulated interviews with similar "
    "symptoms. Rules: use ONLY the cases below. Cite each case that you use as [CASE_ID]. Do not name any "
    "medicine, dose or prescription. Do not say that the person has a condition. Say when to see a doctor. "
    "If the cases do not match the question, say so."
)
USER_TEMPLATE = "Question: {question}\n\nSimilar cases (most similar first):\n{context}\n\nAnswer:"
_CITE = re.compile(r"\[([A-Z]{3}\d{4})\]")

NO_MATCH = "I did not find similar cases for these symptoms. Please talk with a doctor."
OUT_OF_SCOPE = "I can only help with questions about symptoms. Please describe how you feel."
MED_REFUSAL = ("I cannot recommend medicines or doses. A doctor or a pharmacist must decide on any medicine "
               "for you.")


@dataclass
class Answer:
    text: str
    triage: str
    kind: str  # emergency | urgent | out_of_scope | medication_refused | no_match | answer
    cited: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    retrieved: list[tuple[str, str, float]] = field(default_factory=list)
    removed_sentences: int = 0
    invalid_citations: list[str] = field(default_factory=list)


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n".join(h.doc["text"] for h in hits)
    return USER_TEMPLATE.format(question=question.strip(), context=context)


class Assistant:
    def __init__(self, retriever, llm: LLM, top_k: int = 5, min_score: float = 0.05):
        self.retriever, self.llm, self.top_k, self.min_score = retriever, llm, top_k, min_score

    def answer(self, question: str) -> Answer:
        tri = triage(question)
        if tri.level is Triage.EMERGENCY:
            return Answer(f"{tri.message}\n\n{DISCLAIMER}", tri.level.value, "emergency")
        if is_medication_request(question):
            return Answer(f"{MED_REFUSAL}\n\n{DISCLAIMER}", tri.level.value, "medication_refused")
        if not is_in_scope(question):
            return Answer(f"{OUT_OF_SCOPE}\n\n{DISCLAIMER}", tri.level.value, "out_of_scope")

        hits = self.retriever.search(question, k=self.top_k)
        hits = [h for h in hits if h.dense >= self.min_score or h.bm25 > 0]
        prefix = f"{tri.message}\n\n" if tri.level is Triage.URGENT else ""
        kind = "urgent" if tri.level is Triage.URGENT else "answer"
        if not hits:
            return Answer(f"{prefix}{NO_MATCH}\n\n{DISCLAIMER}", tri.level.value, kind if prefix else "no_match")
        try:
            raw = self.llm.complete(SYSTEM, build_prompt(question, hits))
        except LLMError as exc:
            raw = f"The answer service failed ({exc}). Please talk with a doctor."
        retrieved_ids = {h.case_id for h in hits}
        cited = list(dict.fromkeys(_CITE.findall(raw)))
        invalid = [c for c in cited if c not in retrieved_ids]
        for c in invalid:
            raw = raw.replace(f"[{c}]", "")
        clean, removed = strip_prescriptions(raw)
        valid = [c for c in cited if c in retrieved_ids]
        by_id = {h.case_id: h.doc.get("condition", "") for h in hits}
        conditions = list(dict.fromkeys(by_id[c] for c in valid if by_id[c]))
        return Answer(
            text=f"{prefix}{clean}\n\n{DISCLAIMER}", triage=tri.level.value, kind=kind, cited=valid,
            conditions=conditions, retrieved=[(h.case_id, h.doc.get("condition", ""), round(h.score, 4)) for h in hits],
            removed_sentences=removed, invalid_citations=invalid,
        )
