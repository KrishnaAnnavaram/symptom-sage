"""Read speaker-tagged interview transcripts (``D:`` doctor, ``P:`` patient).

A turn can span several lines. The case category comes from the file name prefix
(``RES0001.txt`` -> ``RES``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = {
    "RES": "respiratory", "MSK": "musculoskeletal", "GAS": "gastrointestinal",
    "CAR": "cardiac", "DER": "dermatology", "GEN": "general",
}
_TURN = re.compile(r"^\s*([DP])\s*:\s*(.*)$")
_CASE_ID = re.compile(r"^([A-Z]{3})\d{4}$")


@dataclass(frozen=True)
class Turn:
    speaker: str  # "D" or "P"
    text: str


@dataclass(frozen=True)
class Transcript:
    case_id: str
    category: str
    turns: tuple[Turn, ...]

    def text(self, speaker: str | None = None, tags: bool = True) -> str:
        sel = [t for t in self.turns if speaker is None or t.speaker == speaker]
        return "\n".join(f"{t.speaker}: {t.text}" if tags else t.text for t in sel)


def parse_transcript(case_id: str, raw: str) -> Transcript:
    m = _CASE_ID.match(case_id)
    if not m:
        raise ValueError(f"case id {case_id!r} must look like RES0001")
    if m.group(1) not in CATEGORIES:
        raise ValueError(f"unknown category prefix {m.group(1)!r}; expected one of {sorted(CATEGORIES)}")
    turns: list[list[str]] = []
    for line in raw.splitlines():
        tm = _TURN.match(line)
        if tm:
            turns.append([tm.group(1), tm.group(2).strip()])
        elif line.strip() and turns:
            turns[-1][1] = (turns[-1][1] + " " + line.strip()).strip()
    turns = [t for t in turns if t[1]]
    if not turns:
        raise ValueError(f"{case_id}: no 'D:' or 'P:' turns found")
    return Transcript(case_id, m.group(1), tuple(Turn(s, t) for s, t in turns))


BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"
REPLACEMENT = "\N{REPLACEMENT CHARACTER}"


def read_text_robust(path: str | Path) -> str:
    """Decode UTF-8 or UTF-16 text. Some public transcript files are UTF-16 with a damaged BOM."""
    data = Path(path).read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16")
    elif data.count(b"\x00") > len(data) // 4:  # UTF-16 LE without a usable BOM
        start = max(data.find(b"\x00") - 1, 0)
        body = data[start:]
        text = body[: len(body) // 2 * 2].decode("utf-16-le", errors="replace")
    else:
        text = data.decode("utf-8", errors="replace")
    return text.replace(BOM, "").replace(REPLACEMENT, "")


def load_transcripts(folder: str | Path) -> list[Transcript]:
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"{folder}: no .txt transcripts")
    return [parse_transcript(f.stem, read_text_robust(f)) for f in files]
