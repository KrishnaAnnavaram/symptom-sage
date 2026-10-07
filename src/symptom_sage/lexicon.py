"""A symptom lexicon with aliases and simple negation.

``find_symptoms("I have a cough but no fever")`` -> ``{"cough"}``. A symptom inside a negation
window ("no", "not", "denies", "haven't had" ... up to 4 words before) is not counted.
"""
from __future__ import annotations

import re
from functools import lru_cache

SYMPTOMS: dict[str, tuple[str, ...]] = {
    "cough": ("cough", "coughing"),
    "fever": ("fever", "feverish", "high temperature", "chills"),
    "shortness of breath": ("shortness of breath", "short of breath", "trouble breathing", "breathless",
                            "difficulty breathing", "can't catch my breath"),
    "wheeze": ("wheeze", "wheezing", "wheezy"),
    "chest pain": ("chest pain", "pain in my chest", "chest tightness", "tight chest"),
    "sore throat": ("sore throat", "throat is sore", "scratchy throat"),
    "runny nose": ("runny nose", "congestion", "stuffy nose", "blocked nose"),
    "sputum": ("sputum", "phlegm", "coughing up mucus"),
    "coughing blood": ("coughing up blood", "blood in my sputum", "hemoptysis"),
    "loss of smell": ("loss of smell", "can't smell", "lost my sense of smell", "loss of taste"),
    "fatigue": ("fatigue", "tired", "exhausted", "no energy"),
    "headache": ("headache", "head hurts", "head pain"),
    "muscle aches": ("muscle aches", "body aches", "achy", "myalgia"),
    "nausea": ("nausea", "nauseous", "feel sick to my stomach"),
    "vomiting": ("vomiting", "throwing up", "threw up", "vomited"),
    "diarrhea": ("diarrhea", "diarrhoea", "loose stools", "watery stools"),
    "abdominal pain": ("abdominal pain", "stomach pain", "belly pain", "tummy pain", "pain in my stomach"),
    "heartburn": ("heartburn", "acid reflux", "burning in my chest", "sour taste"),
    "rash": ("rash", "red patches", "hives", "spots on my skin"),
    "itching": ("itching", "itchy", "itch"),
    "joint pain": ("joint pain", "joints hurt", "joint ache"),
    "knee pain": ("knee pain", "pain in my knee", "knee hurts", "sore knee"),
    "back pain": ("back pain", "lower back", "pain in my back", "back hurts"),
    "shoulder pain": ("shoulder pain", "pain in my shoulder", "shoulder hurts"),
    "swelling": ("swelling", "swollen"),
    "stiffness": ("stiffness", "stiff"),
    "palpitations": ("palpitations", "heart racing", "heart is pounding", "fluttering"),
    "dizziness": ("dizziness", "dizzy", "lightheaded", "light-headed"),
    "leg swelling": ("leg swelling", "swollen legs", "swollen ankles"),
    "fainting": ("fainted", "fainting", "passed out", "blacked out"),
    "sweating": ("sweating", "sweaty", "night sweats"),
    "weakness on one side": ("weakness on one side", "one side of my body is weak", "arm went weak"),
    "slurred speech": ("slurred speech", "trouble speaking", "words came out wrong"),
    "face drooping": ("face drooping", "face is drooping", "drooping face"),
    "throat swelling": ("throat swelling", "throat is closing", "tongue swelling", "lips swelling"),
    "stiff neck": ("stiff neck", "neck stiffness"),
    "severe bleeding": ("severe bleeding", "won't stop bleeding", "bleeding heavily"),
    "suicidal thoughts": ("suicidal", "kill myself", "end my life", "want to die"),
    "worst headache": ("worst headache", "thunderclap headache", "sudden severe headache"),
    "blood in stool": ("blood in my stool", "bloody stool", "black stool", "rectal bleeding"),
}
NEGATIONS = ("no", "not", "never", "without", "denies", "deny", "haven't", "hasn't", "don't", "doesn't",
             "didn't", "nor", "neither")


@lru_cache(maxsize=1)
def _patterns() -> list[tuple[str, re.Pattern[str]]]:
    out = []
    for name, aliases in SYMPTOMS.items():
        body = "|".join(re.escape(a) for a in sorted(set(aliases) | {name}, key=len, reverse=True))
        out.append((name, re.compile(rf"(?<![a-z])(?:{body})(?![a-z])", re.IGNORECASE)))
    return out


def _negated(text: str, start: int) -> bool:
    before = re.split(r"[.;!?,]|\bbut\b", text[:start].lower())[-1]
    words = re.findall(r"[a-z']+", before)[-4:]
    return any(w in NEGATIONS for w in words)


def find_symptoms(text: str) -> set[str]:
    """Return the canonical symptoms that the text states (and does not deny)."""
    found = set()
    for name, pat in _patterns():
        for m in pat.finditer(text or ""):
            if not _negated(text, m.start()):
                found.add(name)
                break
    return found
