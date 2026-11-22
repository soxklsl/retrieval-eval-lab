import pytest

from retrieval_eval import evaluate, rank, validate

DATA = {
    "documents": [
        {"id": "a", "text": "python backup"},
        {"id": "b", "text": "nginx proxy"},
    ],
    "queries": [{"id": "q", "text": "backup", "relevant": ["a"]}],
}


@pytest.mark.parametrize("method", ["bm25", "overlap"])
def test_exact(method):
    result = evaluate(DATA, method, 1)
    assert result["recall_at_k"] == result["mrr_at_k"] == 1


def test_no_match():
    assert rank("unknown", DATA["documents"]) == []


def test_partial_recall():
    data = dict(DATA, queries=[{"id": "q", "text": "backup", "relevant": ["a", "b"]}])
    assert evaluate(data, "bm25", 1)["recall_at_k"] == 0.5


def test_tie_stable():
    docs = [{"id": "b", "text": "same"}, {"id": "a", "text": "same"}]
    assert rank("same", docs) == ["a", "b"]


def test_invalid_k():
    with pytest.raises(ValueError):
        rank("backup", DATA["documents"], k=0)


def test_unknown_relevance():
    with pytest.raises(ValueError):
        validate(dict(DATA, queries=[{"id": "q", "text": "x", "relevant": ["z"]}]))


def test_duplicate_ids():
    with pytest.raises(ValueError):
        validate(dict(DATA, documents=DATA["documents"] * 2))
