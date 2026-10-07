"""Seeded synthetic OSCE-style interviews with known ground truth.

The interviews are invented. They follow the format of the public simulated-interview
dataset (``D:`` / ``P:`` turns, a category prefix in the case id) so the same code reads both.
Each case has a known age, gender, symptom set and condition, so extraction and retrieval
can be scored with no manual labels.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from symptom_sage.lexicon import SYMPTOMS

# condition -> (category, core symptoms, optional symptoms, clinician plan)
CONDITIONS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str]] = {
    "community-acquired pneumonia": ("RES", ("cough", "fever", "sputum"), ("shortness of breath", "fatigue", "chest pain"),
                                     "chest X-ray, oxygen saturation, consider antibiotics"),
    "asthma exacerbation": ("RES", ("wheeze", "shortness of breath"), ("cough", "chest pain"),
                            "peak flow, review inhaler use and triggers"),
    "viral upper respiratory infection": ("RES", ("sore throat", "runny nose"), ("cough", "headache", "fatigue"),
                                          "rest, fluids, review if worse after 10 days"),
    "covid-19": ("RES", ("fever", "cough", "loss of smell"), ("fatigue", "muscle aches", "headache"),
                 "antigen or PCR test, isolation advice"),
    "knee osteoarthritis": ("MSK", ("knee pain", "stiffness"), ("swelling",),
                            "knee X-ray, physiotherapy, weight-bearing exercise"),
    "mechanical low back pain": ("MSK", ("back pain",), ("stiffness",),
                                 "check red flags, stay active, physiotherapy"),
    "rotator cuff tendinopathy": ("MSK", ("shoulder pain",), ("stiffness",),
                                  "shoulder exam, physiotherapy"),
    "gastroenteritis": ("GAS", ("diarrhea", "vomiting"), ("nausea", "fever", "abdominal pain"),
                        "oral rehydration, stool test if prolonged"),
    "gastro-oesophageal reflux": ("GAS", ("heartburn",), ("nausea", "chest pain"),
                                  "lifestyle changes, trial of acid suppression"),
    "stable angina": ("CAR", ("chest pain",), ("shortness of breath", "sweating"),
                      "ECG, troponin if pain at rest, cardiology referral"),
    "atrial fibrillation": ("CAR", ("palpitations",), ("dizziness", "shortness of breath", "fatigue"),
                            "ECG, rate control, stroke risk score"),
    "contact dermatitis": ("DER", ("rash", "itching"), ("swelling",),
                           "avoid the trigger, emollients"),
}
DISTRACTORS = ("headache", "fatigue", "nausea", "cough", "muscle aches", "runny nose")
NAMES = ["Alex", "Sam", "Jordan", "Robin", "Kim", "Lee", "Taylor", "Morgan", "Jamie", "Casey"]


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    category: str
    age: int
    gender: str
    symptoms: tuple[str, ...]
    condition: str
    plan: str
    transcript: str


def _say(symptom: str, rng: random.Random) -> str:
    alias = rng.choice(SYMPTOMS[symptom])
    return rng.choice([f"I've had {alias}", f"there's {alias}", f"I noticed {alias}", f"I get {alias}"])


def make_case(case_id: str, condition: str, rng: random.Random) -> SyntheticCase:
    cat, core, optional, plan = CONDITIONS[condition]
    kept = [s for s in core if rng.random() >= 0.25] or [rng.choice(core)]  # a patient can forget a symptom
    symptoms = kept + [s for s in optional if rng.random() < 0.5]
    if rng.random() < 0.4:  # and mention one that does not belong to the condition
        extra = [s for s in DISTRACTORS if s not in symptoms]
        symptoms.append(rng.choice(extra))
    age = rng.randint(18, 85)
    gender = rng.choice(["female", "male"])
    pronoun = "woman" if gender == "female" else "man"
    absent = rng.sample([s for s in ("fever", "cough", "rash", "vomiting", "chest pain", "dizziness")
                         if s not in symptoms], k=2)
    turns = [
        "D: Hello, what brings you in today?",
        f"P: Hi. {_say(symptoms[0], rng).capitalize()} for about {rng.randint(2, 14)} days now.",
        "D: OK. How old are you, and how do you describe yourself?",
        f"P: I'm {age} years old, I'm a {pronoun}.",
        "D: Any other symptoms?",
    ]
    rest = symptoms[1:]
    rng.shuffle(rest)
    turns.append("P: " + (". Also ".join(_say(s, rng) for s in rest) + "." if rest else "Nothing else really."))
    turns.append(f"D: Do you have any {absent[0]} or {absent[1]}?")
    turns.append(f"P: No, no {absent[0]} and I don't have {absent[1]}.")
    turns.append(f"D: Thank you, {rng.choice(NAMES)}. Based on what you said, it sounds like this could be "
                 f"{condition}. We will plan: {plan}.")
    turns.append("P: OK, thank you.")
    return SyntheticCase(case_id, cat, age, gender, tuple(sorted(set(symptoms))), condition, plan,
                         "\n\n".join(turns) + "\n")


def make_cases(n_per_condition: int = 8, seed: int = 21) -> list[SyntheticCase]:
    rng = random.Random(seed)
    counters: dict[str, int] = {}
    cases = []
    for _ in range(n_per_condition):
        for cond, (cat, *_rest) in CONDITIONS.items():
            counters[cat] = counters.get(cat, 0) + 1
            cases.append(make_case(f"{cat}{counters[cat]:04d}", cond, rng))
    return cases


def write_cases(cases: list[SyntheticCase], out_dir: str | Path) -> Path:
    """Write ``<case_id>.txt`` files and ``truth.jsonl`` (ground truth for evaluation)."""
    import json

    out = Path(out_dir)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    with (out / "truth.jsonl").open("w", encoding="utf-8") as fh:
        for c in cases:
            (out / "transcripts" / f"{c.case_id}.txt").write_text(c.transcript, encoding="utf-8")
            fh.write(json.dumps({k: v for k, v in c.__dict__.items() if k != "transcript"}) + "\n")
    return out
