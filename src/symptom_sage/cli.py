"""The ``symptom-sage`` command. Research demo only, not medical advice."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from symptom_sage.config import Settings
from symptom_sage.extraction import ExtractionFailure, LLMExtractor, RuleExtractor
from symptom_sage.llm import LLM, OpenAIChatLLM, TemplateLLM
from symptom_sage.rag import Assistant
from symptom_sage.retrieval import HybridRetriever, LocalIndex, build_index, make_embedder
from symptom_sage.safety import DISCLAIMER
from symptom_sage.store import read_records, read_truth, write_failures, write_records
from symptom_sage.transcripts import load_transcripts


def make_llm(s: Settings) -> LLM:
    if s.llm == "openai":
        return OpenAIChatLLM(s.chat_model, s.openai_api_key, s.openai_base_url)
    return TemplateLLM()


def make_assistant(s: Settings) -> Assistant:
    embedder = make_embedder(s)
    if s.index_backend == "pinecone":
        from symptom_sage.retrieval.hybrid import DenseRetriever
        from symptom_sage.retrieval.index import PineconeIndex

        retriever = DenseRetriever(PineconeIndex(s.pinecone_index, embedder.dim, s.pinecone_api_key), embedder)
    else:
        if not (s.index_dir / "docs.json").exists():
            raise FileNotFoundError(f"no index in {s.index_dir}; run `symptom-sage index` first")
        retriever = HybridRetriever(LocalIndex.load(s.index_dir, embedder.name), embedder)
    return Assistant(retriever, make_llm(s), top_k=s.top_k, min_score=s.min_score)


def _synth(a) -> int:
    from symptom_sage.synthetic import make_cases, write_cases

    out = write_cases(make_cases(a.per_condition, a.seed), a.out)
    print(f"wrote {out / 'transcripts'} and {out / 'truth.jsonl'}")
    return 0


def _extract(a) -> int:
    s = Settings.from_env()
    extractor = LLMExtractor(make_llm(s)) if a.extractor == "llm" else RuleExtractor()
    records, failures = [], []
    for t in load_transcripts(a.transcripts):
        r = extractor.extract(t)
        (failures if isinstance(r, ExtractionFailure) else records).append(r)
    write_records(a.out, records)
    if failures:
        write_failures(Path(a.out).with_suffix(".failures.jsonl"), failures)
    no_cond = sum(1 for r in records if not r.condition)
    print(f"{len(records)} records -> {a.out}; {len(failures)} failures; {no_cond} without a condition (not indexed)")
    return 0


def _index(a) -> int:
    s = Settings.from_env()
    embedder = make_embedder(s)
    records = read_records(a.cases)
    if s.index_backend == "pinecone":
        from symptom_sage.retrieval import case_doc
        from symptom_sage.retrieval.index import PineconeIndex

        usable = [r for r in records if r.condition]
        docs = [case_doc(r) for r in usable]
        idx = PineconeIndex(s.pinecone_index, embedder.dim, s.pinecone_api_key)
        n = idx.upsert([r.case_id for r in usable], embedder.embed([d["text"] for d in docs]), docs)
        print(f"upserted {n} cases into Pinecone index {s.pinecone_index}")
        return 0
    folder = Path(a.index_dir or s.index_dir)
    existing = LocalIndex.load(folder, embedder.name) if (folder / "docs.json").exists() else None
    idx = build_index(records, embedder, existing)
    idx.save(folder)
    print(f"index {folder}: {len(idx)} cases ({embedder.name})")
    return 0


def _print_answer(ans, as_json: bool) -> None:
    if as_json:
        print(json.dumps(ans.__dict__, ensure_ascii=False, indent=2))
    else:
        print(ans.text)
        if ans.retrieved:
            print("\nretrieved: " + ", ".join(f"{c} ({cond})" for c, cond, _ in ans.retrieved))


def _ask(a) -> int:
    _print_answer(make_assistant(Settings.from_env()).answer(a.question), a.json)
    return 0


def _chat(_a) -> int:  # pragma: no cover - interactive
    assistant = make_assistant(Settings.from_env())
    print(DISCLAIMER + "\nType a message, or an empty line to stop.")
    while True:
        q = input("> ").strip()
        if not q:
            return 0
        _print_answer(assistant.answer(q), False)


def _eval(a) -> int:
    from symptom_sage import evaluate as ev

    s = Settings.from_env()
    out: dict = {}
    if a.what in ("extraction", "all"):
        if not a.truth:
            raise ValueError("--truth is necessary for the extraction evaluation")
        out["extraction"] = ev.eval_extraction(read_records(a.cases), read_truth(a.truth))
    if a.what in ("retrieval", "all"):
        out["retrieval"] = ev.eval_retrieval(read_records(a.cases), make_embedder(s))
    if a.what in ("answers", "all"):
        idx_recs, q_recs = ev.split_half(read_records(a.cases))
        out["answers"] = ev.eval_answers(idx_recs, q_recs, make_embedder(s), make_llm(s), s.top_k)
    if a.what in ("safety", "all"):
        recs = read_records(a.cases)
        emb = make_embedder(s)
        out["safety"] = ev.eval_safety(Assistant(HybridRetriever(build_index(recs, emb), emb), make_llm(s)))
    print(json.dumps(out, indent=2))
    return 0


def _asr(a) -> int:
    from symptom_sage.asr import FasterWhisperTranscriber, score_corpus

    s = Settings.from_env()
    refs = {t.case_id: t for t in load_transcripts(a.ref_dir)}
    pairs = []
    if a.audio_dir:
        tr = FasterWhisperTranscriber(s.whisper_model)
        for audio in sorted(Path(a.audio_dir).glob("*.mp3")):
            if audio.stem in refs:
                hyp = tr.transcribe(audio)
                if a.save_dir:
                    Path(a.save_dir).mkdir(parents=True, exist_ok=True)
                    (Path(a.save_dir) / f"{audio.stem}.txt").write_text(hyp, encoding="utf-8")
                pairs.append((audio.stem, refs[audio.stem].category, refs[audio.stem].text(), hyp))
    else:
        for hyp_file in sorted(Path(a.hyp_dir).glob("*.txt")):
            if hyp_file.stem in refs:
                ref = refs[hyp_file.stem]
                pairs.append((ref.case_id, ref.category, ref.text(), hyp_file.read_text(encoding="utf-8")))
    if not pairs:
        raise ValueError("no hypothesis matched a reference transcript")
    res = score_corpus(pairs)
    print(json.dumps({"n": len(pairs), "pooled_wer": res["pooled_wer"],
                      "mean_cer": sum(c.cer for c in res["cases"]) / len(res["cases"])}, indent=2))
    return 0


def _demo(a) -> int:
    from symptom_sage import evaluate as ev
    from symptom_sage.synthetic import make_cases, write_cases

    root = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="symptom-sage-demo-"))
    write_cases(make_cases(8, 21), root / "data")
    extractor = RuleExtractor()
    records = [extractor.extract(t) for t in load_transcripts(root / "data" / "transcripts")]
    write_records(root / "data" / "cases.jsonl", records)
    truth = read_truth(root / "data" / "truth.jsonl")
    emb = make_embedder(Settings())
    report = {
        "extraction": ev.eval_extraction(records, truth),
        "retrieval": ev.eval_retrieval(records, emb),
        "answers": ev.eval_answers(*ev.split_half(records), emb, TemplateLLM()),
    }
    assistant = Assistant(HybridRetriever(build_index(records, emb), emb), TemplateLLM())
    report["safety"] = ev.eval_safety(assistant)
    print("SYNTHETIC DATA - these numbers do not describe real patients.\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "failures"} for k, v in report.items()}, indent=2))
    for q in ("I have a cough, fever and green phlegm", "I have crushing chest pain and I'm sweating"):
        print(f"\n> {q}")
        _print_answer(assistant.answer(q), False)
    print(f"\n(files in {root})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="symptom-sage", description=DISCLAIMER)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write synthetic interviews and their ground truth")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--per-condition", type=int, default=8)
    s.add_argument("--seed", type=int, default=21)
    s.set_defaults(fn=_synth)

    s = sub.add_parser("extract", help="extract validated case records from transcripts")
    s.add_argument("--transcripts", required=True)
    s.add_argument("--out", default="data/cases.jsonl")
    s.add_argument("--extractor", choices=("rules", "llm"), default="rules")
    s.set_defaults(fn=_extract)

    s = sub.add_parser("index", help="build or update the case index (idempotent)")
    s.add_argument("--cases", default="data/cases.jsonl")
    s.add_argument("--index-dir")
    s.set_defaults(fn=_index)

    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_ask)

    s = sub.add_parser("chat", help="interactive chat in the terminal")
    s.set_defaults(fn=_chat)

    s = sub.add_parser("eval", help="run evaluations")
    s.add_argument("what", choices=("extraction", "retrieval", "answers", "safety", "all"))
    s.add_argument("--cases", default="data/cases.jsonl")
    s.add_argument("--truth")
    s.set_defaults(fn=_eval)

    s = sub.add_parser("asr-eval", help="WER and CER of transcripts against the references")
    s.add_argument("--ref-dir", required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--hyp-dir", help="existing ASR output (.txt per case)")
    g.add_argument("--audio-dir", help="audio (.mp3 per case); needs the asr extra")
    s.add_argument("--save-dir", help="where to save new transcripts (with --audio-dir)")
    s.set_defaults(fn=_asr)

    s = sub.add_parser("demo", help="offline end-to-end demo on synthetic interviews")
    s.add_argument("--out")
    s.set_defaults(fn=_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
