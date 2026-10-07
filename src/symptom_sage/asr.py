"""Speech recognition and its evaluation with word and character error rates.

The prototype compared whole-transcript embeddings, which are close for any two related
texts. Here the metric is WER (and CER) after the same normalization on both sides: speaker
tags, case, punctuation and filler words are removed.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

FILLERS = frozenset({"um", "uh", "uhm", "ah", "ahh", "er", "erm", "hmm", "mm"})


def normalize_for_wer(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?m)^\s*[dp]\s*:", " ", text)  # speaker tags at the start of a line
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if w not in FILLERS]


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_for_wer(reference), normalize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = " ".join(normalize_for_wer(reference)), " ".join(normalize_for_wer(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(list(ref), list(hyp)) / len(ref)


@dataclass(frozen=True)
class AsrScore:
    case_id: str
    category: str
    wer: float
    cer: float
    ref_words: int


def score_corpus(pairs: list[tuple[str, str, str, str]]) -> dict:
    """pairs: (case_id, category, reference, hypothesis). Returns per-case and pooled results.

    The pooled WER weights each case by its reference length (total errors / total words).
    """
    scores, errs, words = [], defaultdict(int), defaultdict(int)
    for cid, cat, ref, hyp in pairs:
        r, h = normalize_for_wer(ref), normalize_for_wer(hyp)
        e = edit_distance(r, h)
        scores.append(AsrScore(cid, cat, e / max(len(r), 1), cer(ref, hyp), len(r)))
        for key in (cat, "ALL"):
            errs[key] += e
            words[key] += len(r)
    pooled = {k: errs[k] / max(words[k], 1) for k in words}
    return {"cases": scores, "pooled_wer": pooled}


class Transcriber(Protocol):
    def transcribe(self, audio_path: str | Path) -> str: ...


class FasterWhisperTranscriber:
    """faster-whisper (optional extra ``asr``). The model loads once."""

    def __init__(self, model: str = "base", device: str = "auto", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError('transcription needs the "asr" extra: pip install -e ".[asr]"') from exc
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str | Path) -> str:
        segments, _ = self._model.transcribe(str(audio_path), language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments)
