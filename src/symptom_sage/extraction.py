"""Schema-validated case extraction from transcripts.

Two extractors give the same :class:`CaseRecord`:

* :class:`RuleExtractor` - offline, deterministic, from patient and doctor turns.
* :class:`LLMExtractor` - JSON output, validated with pydantic. One retry with the error.
  A case that fails twice becomes an :class:`ExtractionFailure`; the run does not stop.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from symptom_sage.lexicon import find_symptoms
from symptom_sage.llm import LLM, LLMError
from symptom_sage.transcripts import CATEGORIES, Transcript

Gender = Literal["female", "male", "other", "unknown"]
_EMPTY = {"", "na", "n/a", "none", "not specified", "unknown", "null"}


class CaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[A-Z]{3}\d{4}$")
    category: str
    age: int | None = Field(default=None, ge=0, le=120)
    gender: Gender = "unknown"
    reason: str = Field(default="", max_length=300)
    symptoms: list[str] = Field(default_factory=list)
    condition: str | None = Field(default=None, max_length=120)
    plan: str | None = Field(default=None, max_length=400, description="clinician plan; never shown to patients")
    extractor: str = "rules"

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"category must be one of {sorted(CATEGORIES)}")
        return v

    @field_validator("condition", "plan", mode="before")
    @classmethod
    def _empty(cls, v):
        if isinstance(v, str) and v.strip().lower() in _EMPTY:
            return None
        return v.strip().lower() if isinstance(v, str) and v else v

    @field_validator("gender", mode="before")
    @classmethod
    def _gender(cls, v):
        v = (v or "unknown").strip().lower() if isinstance(v, str) else "unknown"
        return {"f": "female", "woman": "female", "m": "male", "man": "male"}.get(v, v)

    @field_validator("symptoms", mode="before")
    @classmethod
    def _symptoms(cls, v):
        if isinstance(v, str):
            v = [s for s in re.split(r"[,;]", v)]
        out: list[str] = []
        for s in v or []:
            canon = sorted(find_symptoms(str(s)))
            out += canon if canon else ([str(s).strip().lower()] if str(s).strip().lower() not in _EMPTY else [])
        return sorted(set(out))

    def document(self) -> str:
        """The retrieval text. It holds the assessment, but never the plan."""
        who = f"{self.age}-year-old {self.gender}" if self.age is not None else f"{self.gender} patient"
        cond = self.condition or "not stated"
        return (f"[{self.case_id}] {CATEGORIES[self.category]} case. Patient: {who}. "
                f"Symptoms: {', '.join(self.symptoms) or 'none recorded'}. Assessment: {cond}.")


@dataclass(frozen=True)
class ExtractionFailure:
    case_id: str
    error: str


_AGE = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|yo\b|y/o)", re.IGNORECASE)
_NUM = re.compile(r"\b(\d{1,3})\b")
_GENDER = re.compile(r"\b(?:i'm|i am)\s+an?\s+(woman|female|man|male|lady|gentleman)\b", re.IGNORECASE)
_COND = re.compile(r"(?:could be|likely|probably|most likely|suspect)\s+(?:a |an |some )?([a-z0-9' -]{3,60}?)\s*(?:[.,;]|$)",
                   re.IGNORECASE)
_PLAN = re.compile(r"we will plan:\s*([^.]+)", re.IGNORECASE)


class RuleExtractor:
    name = "rules"

    def extract(self, t: Transcript) -> CaseRecord:
        patient = t.text("P", tags=False)
        age = None
        m = _AGE.search(patient)
        if m:
            age = int(m.group(1))
        else:  # "how old are you?" followed by "Um 52."
            for i, turn in enumerate(t.turns[:-1]):
                asks = re.search(r"how old|\bage\b", turn.text.lower()) and "?" in turn.text
                if turn.speaker == "D" and asks and t.turns[i + 1].speaker == "P":
                    n = _NUM.search(t.turns[i + 1].text)
                    if n and 0 < int(n.group(1)) <= 110:
                        age = int(n.group(1))
                        break
        g = _GENDER.search(patient)
        gender = {"woman": "female", "lady": "female", "man": "male", "gentleman": "male"}.get(
            g.group(1).lower(), g.group(1).lower()) if g else "unknown"
        first = next((x.text for x in t.turns if x.speaker == "P"), "")
        doctor_tail = " ".join(x.text for x in t.turns if x.speaker == "D")[-1200:]
        cm = _COND.search(doctor_tail)
        pm = _PLAN.search(doctor_tail)
        return CaseRecord(case_id=t.case_id, category=t.category, age=age if age is not None and age <= 120 else None,
                          gender=gender, reason=first[:300], symptoms=sorted(find_symptoms(patient)),
                          condition=cm.group(1) if cm else None, plan=pm.group(1) if pm else None,
                          extractor=self.name)


SYSTEM_PROMPT = (
    "You extract facts from a simulated doctor-patient interview. Reply with one JSON object only, with "
    'these keys: "age" (integer or null), "gender" ("female", "male", "other" or "unknown"), "reason" '
    '(the reason for the visit, max 300 characters), "symptoms" (list of short symptom names that the '
    'patient HAS; leave out denied symptoms), "condition" (the doctor\'s most likely assessment or null), '
    '"plan" (the doctor\'s plan or null). Use null when the interview does not say it. Do not guess.'
)


class LLMExtractor:
    def __init__(self, llm: LLM, max_chars: int = 24000):
        self.llm = llm
        self.name = f"llm:{llm.name}"
        self.max_chars = max_chars

    def _parse(self, t: Transcript, raw: str) -> CaseRecord:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("reply is not a JSON object")
        return CaseRecord(case_id=t.case_id, category=t.category, extractor=self.name,
                          **{k: data.get(k) for k in ("age", "gender", "reason", "symptoms", "condition", "plan")
                             if data.get(k) is not None})

    def extract(self, t: Transcript) -> CaseRecord | ExtractionFailure:
        user = t.text()[: self.max_chars]
        error = ""
        for attempt in range(2):
            prompt = user if attempt == 0 else f"{user}\n\nYour last reply was invalid: {error}. Reply with valid JSON."
            try:
                return self._parse(t, self.llm.complete(SYSTEM_PROMPT, prompt, json_mode=True))
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            except LLMError as exc:
                return ExtractionFailure(t.case_id, str(exc))
        return ExtractionFailure(t.case_id, error)
