import json

from symptom_sage.cli import main


def test_full_cli_flow(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SYMPTOM_SAGE_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.delenv("SYMPTOM_SAGE_LLM", raising=False)
    monkeypatch.delenv("SYMPTOM_SAGE_EMBEDDER", raising=False)
    monkeypatch.delenv("SYMPTOM_SAGE_INDEX_BACKEND", raising=False)
    data = tmp_path / "data"
    assert main(["synth", "--out", str(data), "--per-condition", "4"]) == 0
    assert main(["extract", "--transcripts", str(data / "transcripts"), "--out", str(data / "cases.jsonl")]) == 0
    assert main(["index", "--cases", str(data / "cases.jsonl")]) == 0
    assert main(["index", "--cases", str(data / "cases.jsonl")]) == 0  # second run: same size
    out = capsys.readouterr().out
    assert out.count("index ") == 2 and "48 cases" in out
    assert main(["ask", "I get heartburn after meals", "--json"]) == 0
    ans = json.loads(capsys.readouterr().out)
    assert ans["kind"] == "answer" and "gastro-oesophageal reflux" in ans["conditions"]
    assert main(["eval", "all", "--cases", str(data / "cases.jsonl"), "--truth", str(data / "truth.jsonl")]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["safety"]["pass_rate"] == 1.0 and res["extraction"]["n"] == 48


def test_asr_eval_from_hypothesis_files(tmp_path, capsys):
    ref, hyp = tmp_path / "ref", tmp_path / "hyp"
    ref.mkdir()
    hyp.mkdir()
    (ref / "RES0001.txt").write_text("D: How are you?\n\nP: I have a cough.\n")
    (hyp / "RES0001.txt").write_text("how are you i have a cold")
    assert main(["asr-eval", "--ref-dir", str(ref), "--hyp-dir", str(hyp)]) == 0
    assert json.loads(capsys.readouterr().out)["pooled_wer"]["ALL"] == 1 / 7


def test_ask_without_index_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMPTOM_SAGE_INDEX_DIR", str(tmp_path / "none"))
    assert main(["ask", "cough"]) == 2


def test_demo(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    assert "SYNTHETIC DATA" in capsys.readouterr().out
