"""Read and write case records (JSONL) and extraction failures."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from symptom_sage.extraction import CaseRecord, ExtractionFailure


def write_records(path: str | Path, records: list[CaseRecord]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in sorted(records, key=lambda r: r.case_id):
            fh.write(r.model_dump_json() + "\n")
    return path


def write_failures(path: str | Path, failures: list[ExtractionFailure]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for f in failures:
            fh.write(json.dumps({"case_id": f.case_id, "error": f.error}) + "\n")
    return path


def read_records(path: str | Path) -> list[CaseRecord]:
    path = Path(path)
    out, errors, seen = [], [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = CaseRecord.model_validate_json(line)
        except ValidationError as exc:
            errors.append(f"line {n}: {exc.errors()[0]['msg']}")
            continue
        if rec.case_id in seen:
            errors.append(f"line {n}: duplicate case_id {rec.case_id}")
            continue
        seen.add(rec.case_id)
        out.append(rec)
    if errors:
        raise ValueError(f"{path}: {len(errors)} invalid line(s): " + "; ".join(errors[:5]))
    return out


def read_truth(path: str | Path) -> dict[str, dict]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["case_id"]: r for r in rows}
