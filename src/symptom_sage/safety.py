"""The safety layer: red-flag triage, scope checks, the output filter and the disclaimer.

The rules run BEFORE retrieval. An emergency message never goes to the retriever or the LLM.
The output filter runs AFTER generation and removes any sentence that names a medicine or a dose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from symptom_sage.lexicon import find_symptoms

DISCLAIMER = ("symptom-sage is a research demo built on simulated interviews. It is not medical advice and it "
              "cannot diagnose you. If you are worried, contact a doctor. In an emergency, call your local "
              "emergency number.")


class Triage(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"


@dataclass(frozen=True)
class RedFlag:
    name: str
    level: Triage
    all_of: tuple[str, ...]
    any_of: tuple[str, ...]
    message: str


EMERGENCY_MSG = "Call your local emergency number now (for example 911 in the US or 112 in the EU)."
RED_FLAGS: tuple[RedFlag, ...] = (
    RedFlag("possible heart attack", Triage.EMERGENCY, ("chest pain",), ("sweating", "shortness of breath", "fainting",
            "dizziness"), "Chest pain with these symptoms can be a heart attack. " + EMERGENCY_MSG),
    RedFlag("possible stroke", Triage.EMERGENCY, (), ("face drooping", "slurred speech", "weakness on one side"),
            "These can be signs of a stroke. " + EMERGENCY_MSG),
    RedFlag("possible anaphylaxis", Triage.EMERGENCY, (), ("throat swelling",),
            "Swelling of the throat, tongue or lips can be a severe allergic reaction. " + EMERGENCY_MSG),
    RedFlag("possible meningitis", Triage.EMERGENCY, ("stiff neck", "fever"), (),
            "A stiff neck with fever can be meningitis. " + EMERGENCY_MSG),
    RedFlag("sudden severe headache", Triage.EMERGENCY, (), ("worst headache",),
            "A sudden, very bad headache needs emergency care. " + EMERGENCY_MSG),
    RedFlag("severe bleeding", Triage.EMERGENCY, (), ("severe bleeding",), "Heavy bleeding needs emergency care. "
            + EMERGENCY_MSG),
    RedFlag("suicidal thoughts", Triage.EMERGENCY, (), ("suicidal thoughts",),
            "You do not have to go through this alone. Call or text 988 (US) or your local crisis line, or call "
            "your local emergency number now."),
    RedFlag("coughing blood", Triage.URGENT, (), ("coughing blood",),
            "Coughing up blood needs a doctor today. Go to urgent care or call your doctor now."),
    RedFlag("blood in stool", Triage.URGENT, (), ("blood in stool",),
            "Blood in the stool needs a doctor soon. Call your doctor today."),
    RedFlag("fainting", Triage.URGENT, (), ("fainting",), "Fainting needs a doctor today. Call your doctor now."),
)

_MED_REQUEST = re.compile(
    r"\b(dose|dosage|how (?:much|many) .*\b(?:take|use)|prescri\w*|which (?:drug|medicine|medication|pill|antibiotic)"
    r"|what (?:drug|medicine|medication|pill|antibiotic)|can i take|should i take|safe to take|mg\b)", re.IGNORECASE)
_MEDICAL_WORDS = re.compile(r"\b(pain|hurts?|sick|ill|symptoms?|ache|sore|doctor|feel|feeling|unwell|condition|"
                            r"injur\w*|disease|infection|bleed\w*|breath\w*|cough\w*|fever\w*|rash)\b", re.IGNORECASE)
DRUG_WORDS = (
    "amoxicillin", "azithromycin", "doxycycline", "ibuprofen", "paracetamol", "acetaminophen", "aspirin", "naproxen",
    "salbutamol", "albuterol", "prednisone", "prednisolone", "omeprazole", "pantoprazole", "metformin", "warfarin",
    "apixaban", "rivaroxaban", "metoprolol", "bisoprolol", "nitroglycerin", "codeine", "morphine", "antibiotics?",
    "steroids?", "inhaler", "antihistamines?", "proton pump inhibitor",
)
RX_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?\s?(?:mg|mcg|µg|ml|g|units?)\b|tablets?|capsules?|dos(?:e|age)s?|"
                          r"prescri\w*|twice (?:a|per) day|once daily|" + "|".join(DRUG_WORDS) + r")", re.IGNORECASE)


@dataclass(frozen=True)
class TriageResult:
    level: Triage
    flags: tuple[str, ...]
    message: str | None


def triage(text: str) -> TriageResult:
    sym = find_symptoms(text)
    hits = [f for f in RED_FLAGS if all(s in sym for s in f.all_of) and (not f.any_of or any(s in sym for s in f.any_of))]
    if not hits:
        return TriageResult(Triage.ROUTINE, (), None)
    level = Triage.EMERGENCY if any(h.level is Triage.EMERGENCY for h in hits) else Triage.URGENT
    top = [h for h in hits if h.level is level]
    return TriageResult(level, tuple(h.name for h in hits), " ".join(dict.fromkeys(h.message for h in top)))


def is_medication_request(text: str) -> bool:
    return bool(_MED_REQUEST.search(text))


def is_in_scope(text: str) -> bool:
    return bool(find_symptoms(text)) or bool(_MEDICAL_WORDS.search(text))


def strip_prescriptions(answer: str) -> tuple[str, int]:
    """Remove each sentence or list line that names a medicine, a dose or a prescription."""
    parts = re.split(r"(?<=[.!?])\s+|\n", answer)
    keep = [p for p in parts if p.strip() and not RX_PATTERN.search(p)]
    removed = sum(1 for p in parts if p.strip()) - len(keep)
    return "\n".join(keep), removed
