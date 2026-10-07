import pytest

from symptom_sage.extraction import RuleExtractor
from symptom_sage.retrieval import HybridRetriever, build_index
from symptom_sage.retrieval.embeddings import HashingEmbedder
from symptom_sage.synthetic import make_cases
from symptom_sage.transcripts import parse_transcript


@pytest.fixture(scope="session")
def cases():
    return make_cases(n_per_condition=6, seed=3)


@pytest.fixture(scope="session")
def records(cases):
    ex = RuleExtractor()
    return [ex.extract(parse_transcript(c.case_id, c.transcript)) for c in cases]


@pytest.fixture(scope="session")
def retriever(records):
    emb = HashingEmbedder()
    return HybridRetriever(build_index(records, emb), emb)
