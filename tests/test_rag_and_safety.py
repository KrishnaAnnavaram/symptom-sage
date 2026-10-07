import json

import httpx
import numpy as np
import pytest

from symptom_sage.config import Settings
from symptom_sage.evaluate import eval_answers, eval_extraction, eval_retrieval, eval_safety, split_half
from symptom_sage.llm import OpenAIChatLLM, ScriptedLLM, TemplateLLM
from symptom_sage.rag import MED_REFUSAL, Assistant, build_prompt
from symptom_sage.retrieval import build_index
from symptom_sage.retrieval.embeddings import HashingEmbedder, OpenAIEmbedder
from symptom_sage.retrieval.hybrid import BM25
from symptom_sage.retrieval.index import LocalIndex
from symptom_sage.safety import DISCLAIMER, Triage, strip_prescriptions, triage


def test_prompt_contains_question_and_assessments(retriever):
    hits = retriever.search("I have a cough, fever and phlegm", k=3)
    prompt = build_prompt("I have a cough, fever and phlegm", hits)
    assert prompt.startswith("Question: I have a cough, fever and phlegm")
    assert all(h.doc["condition"] in prompt for h in hits)


def test_llm_sees_the_question(retriever):
    llm = ScriptedLLM(["See a doctor."])
    Assistant(retriever, llm).answer("my knee hurts and is stiff")
    assert "my knee hurts and is stiff" in llm.prompts[0][1]


def test_answer_cites_only_retrieved_cases(retriever):
    llm = ScriptedLLM(["Similar to [MSK0001] and [ZZZ9999]. Take ibuprofen 400 mg twice a day. See a doctor."])
    a = Assistant(retriever, llm).answer("my knee hurts and is stiff")
    assert "ZZZ9999" in a.invalid_citations and "[ZZZ9999]" not in a.text
    assert "ibuprofen" not in a.text and a.removed_sentences == 1
    assert a.text.endswith(DISCLAIMER)


def test_emergency_skips_retrieval_and_llm(retriever):
    llm = ScriptedLLM([])
    a = Assistant(retriever, llm).answer("I have chest pain and I'm sweating")
    assert a.kind == "emergency" and "emergency number" in a.text and llm.prompts == []


def test_medication_requests_are_refused(retriever):
    a = Assistant(retriever, ScriptedLLM([])).answer("What dose of amoxicillin should I take for my cough?")
    assert a.kind == "medication_refused" and MED_REFUSAL in a.text


def test_triage_rules_and_negation():
    assert triage("stiff neck and fever").level is Triage.EMERGENCY
    assert triage("I don't have a fever, my neck is stiff").level is Triage.ROUTINE
    assert triage("I coughed up blood").level is Triage.ROUTINE  # not an alias: no flag
    assert triage("I am coughing up blood").level is Triage.URGENT


def test_strip_prescriptions():
    text, n = strip_prescriptions("This can be reflux.\nTake omeprazole 20 mg daily.\nSee a doctor.")
    assert n == 1 and "omeprazole" not in text and "reflux" in text


def test_red_team_set_passes(retriever):
    res = eval_safety(Assistant(retriever, TemplateLLM()))
    assert res["n"] >= 20 and res["pass_rate"] == 1.0, res["failures"]


def test_index_upsert_is_idempotent(records):
    emb = HashingEmbedder()
    idx = build_index(records, emb)
    n = len(idx)
    build_index(records, emb, idx)
    assert len(idx) == n
    with pytest.raises(ValueError):
        idx.upsert(["A", "A"], np.zeros((2, emb.dim), dtype=np.float32), [{}, {}])


def test_index_save_load_and_embedder_check(tmp_path, records):
    emb = HashingEmbedder()
    idx = build_index(records, emb)
    idx.save(tmp_path)
    again = LocalIndex.load(tmp_path, emb.name)
    q = emb.embed(["cough fever"])[0]
    assert again.query(q, 3)[0][0] == idx.query(q, 3)[0][0]
    with pytest.raises(ValueError):
        LocalIndex.load(tmp_path, "openai:text-embedding-3-small")


def test_bm25_prefers_matching_documents():
    bm = BM25({"a": "cough fever sputum", "b": "knee pain stiffness", "c": "rash itching"})
    s = bm.scores("I have knee pain")
    assert max(s, key=s.get) == "b" and s["c"] == 0


def test_retrieval_beats_chance(records):
    res = eval_retrieval(records, HashingEmbedder())
    assert res["hit@1"] > 3 * res["chance_hit@1"] and res["hit@5"] >= res["hit@1"]


def test_answers_and_extraction_eval(cases, records):
    a, b = split_half(records)
    assert {r.condition for r in a} == {r.condition for r in b}
    res = eval_answers(a, b, HashingEmbedder(), TemplateLLM())
    assert res["valid_citations"] == 1.0 and res["medicine_leaks"] == 0
    truth = {c.case_id: {"age": c.age, "gender": c.gender, "condition": c.condition, "symptoms": list(c.symptoms)}
             for c in cases}
    ex = eval_extraction(records, truth)
    assert ex["condition_acc"] == 1.0 and ex["symptom_f1"] > 0.9


def test_openai_adapters_use_mock_transport():
    seen = {}

    def chat(request):
        seen["chat"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm = OpenAIChatLLM("gpt-4o-mini", "k", client=httpx.Client(transport=httpx.MockTransport(chat)))
    assert llm.complete("s", "u", json_mode=True) == "{}"
    assert seen["chat"]["response_format"] == {"type": "json_object"} and seen["chat"]["temperature"] == 0

    def emb(request):
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [3.0, 4.0]} for i in range(n)]})

    e = OpenAIEmbedder("m", "k", dim=2, client=httpx.Client(transport=httpx.MockTransport(emb)))
    assert np.allclose(e.embed(["a", "b"]), [[0.6, 0.8], [0.6, 0.8]])


def test_settings_validation(monkeypatch):
    monkeypatch.setenv("SYMPTOM_SAGE_LLM", "gpt")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("SYMPTOM_SAGE_LLM", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    assert "secret-value" not in repr(Settings.from_env())
