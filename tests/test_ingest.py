import json

import pytest

from symptom_sage.asr import cer, normalize_for_wer, score_corpus, wer
from symptom_sage.extraction import CaseRecord, ExtractionFailure, LLMExtractor, RuleExtractor
from symptom_sage.lexicon import find_symptoms
from symptom_sage.llm import LLMError, ScriptedLLM
from symptom_sage.store import read_records, write_records
from symptom_sage.transcripts import load_transcripts, parse_transcript, read_text_robust

RAW = """D: What brings you in today?

P: I've had a cough for a week and I'm
short of breath when I walk.

D: How old are you?

P: Um 52.

D: Any fever?

P: No fever.

D: It sounds like this could be asthma exacerbation. We will plan: peak flow.
"""


def test_parse_joins_continuation_lines():
    t = parse_transcript("RES0001", RAW)
    assert t.category == "RES" and t.turns[1].text.endswith("short of breath when I walk.")
    with pytest.raises(ValueError):
        parse_transcript("XYZ0001", RAW)
    with pytest.raises(ValueError):
        parse_transcript("RES0002", "no turns here")


def test_read_utf16_with_damaged_bom(tmp_path):
    p = tmp_path / "RES0009.txt"
    p.write_bytes(b"\xef\xbf\xbd\xef\xbf\xbd" + "D: Hi?\n\nP: I have a cough.\n".encode("utf-16-le"))
    assert "P: I have a cough." in read_text_robust(p)
    assert load_transcripts(tmp_path)[0].turns[1].text == "I have a cough."


def test_symptoms_with_aliases_and_negation():
    assert find_symptoms("I'm short of breath, no fever, and I have a cough") == {"shortness of breath", "cough"}
    assert find_symptoms("I don't have chest pain but I feel dizzy") == {"dizziness"}
    assert find_symptoms("Any time I walk I'm breathless") == {"shortness of breath"}


def test_rule_extractor():
    r = RuleExtractor().extract(parse_transcript("RES0001", RAW))
    assert r.age == 52 and r.condition == "asthma exacerbation" and r.plan == "peak flow"
    assert r.symptoms == ["cough", "shortness of breath"]


def test_rule_extractor_matches_synthetic_truth(cases, records):
    for c, r in zip(cases, records):
        assert (r.age, r.gender, r.condition) == (c.age, c.gender, c.condition)


def test_document_holds_assessment_but_not_plan(records):
    doc = records[0].document()
    assert "Assessment:" in doc and records[0].condition in doc
    assert records[0].plan and records[0].plan not in doc


def test_schema_normalizes_values():
    r = CaseRecord(case_id="GAS0001", category="GAS", gender="F", condition="  NA ", symptoms="vomiting; loose stools")
    assert r.gender == "female" and r.condition is None and r.symptoms == ["diarrhea", "vomiting"]
    with pytest.raises(ValueError):
        CaseRecord(case_id="GAS0001", category="XXX")
    with pytest.raises(ValueError):
        CaseRecord(case_id="GAS0001", category="GAS", age=300)


def test_llm_extractor_retries_once_then_fails_softly():
    t = parse_transcript("RES0001", RAW)
    good = json.dumps({"age": 52, "gender": "unknown", "symptoms": ["cough"], "condition": "Asthma exacerbation"})
    llm = ScriptedLLM(["not json", good])
    r = LLMExtractor(llm).extract(t)
    assert isinstance(r, CaseRecord) and r.condition == "asthma exacerbation"
    assert "invalid" in llm.prompts[1][1]
    bad = LLMExtractor(ScriptedLLM(["{", '{"age": 999}'])).extract(t)
    assert isinstance(bad, ExtractionFailure)
    fenced = LLMExtractor(ScriptedLLM(["```json\n" + good + "\n```"])).extract(t)
    assert isinstance(fenced, CaseRecord)


def test_llm_errors_do_not_stop_the_run():
    class Down:
        name = "down"

        def complete(self, *a, **k):
            raise LLMError("timeout")

    assert isinstance(LLMExtractor(Down()).extract(parse_transcript("RES0001", RAW)), ExtractionFailure)


def test_store_round_trip(tmp_path, records):
    path = write_records(tmp_path / "cases.jsonl", records)
    assert read_records(path) == sorted(records, key=lambda r: r.case_id)
    path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_records(path)


def test_wer_ignores_tags_case_and_fillers():
    ref = "D: How are you?\nP: Um, I have a cough."
    assert wer(ref, "how are you i have a cough") == 0.0
    assert normalize_for_wer("P: Uh, HELLO!") == ["hello"]
    assert wer("a b c d", "a x c") == 0.5
    assert cer("abc", "abd") == pytest.approx(1 / 3)


def test_pooled_wer_by_category():
    res = score_corpus([("RES0001", "RES", "a b c d", "a b c d"), ("MSK0001", "MSK", "a b", "a")])
    assert res["pooled_wer"]["RES"] == 0.0 and res["pooled_wer"]["MSK"] == 0.5
    assert res["pooled_wer"]["ALL"] == pytest.approx(1 / 6)
