"""Evaluate lexical RAG retrieval offline with per-query relevance judgments."""

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


def rank(query, documents, method="bm25", k=3):
    if k < 1 or method not in {"bm25", "overlap"}:
        raise ValueError("Choose k > 0 and method bm25 or overlap")
    terms = set(tokenize(query))
    tokens = {doc["id"]: Counter(tokenize(doc["text"])) for doc in documents}
    average = sum(sum(counts.values()) for counts in tokens.values()) / max(
        len(tokens), 1
    )
    scores = []
    for identifier, counts in tokens.items():
        score = 0.0
        for term in terms:
            frequency = counts[term]
            if not frequency:
                continue
            if method == "overlap":
                score += 1
                continue
            df = sum(term in other for other in tokens.values())
            idf = math.log(1 + (len(tokens) - df + 0.5) / (df + 0.5))
            score += (
                idf
                * frequency
                * 2.5
                / (frequency + 1.5 * (0.25 + 0.75 * sum(counts.values()) / average))
            )
        if score > 0:
            scores.append((identifier, score))
    return [
        identifier
        for identifier, _ in sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:k]
    ]


def validate(data):
    documents, queries = data["documents"], data["queries"]
    if not documents or not queries:
        raise ValueError("Documents and queries must be nonempty")
    for rows in (documents, queries):
        ids = [row["id"] for row in rows]
        if any(not isinstance(value, str) or not value for value in ids) or len(
            ids
        ) != len(set(ids)):
            raise ValueError("IDs must be unique nonempty strings")
        if any(not isinstance(row["text"], str) for row in rows):
            raise ValueError("Text must be a string")
    ids = {doc["id"] for doc in documents}
    for query in queries:
        relevant = query["relevant"]
        if not isinstance(relevant, list) or not relevant or not set(relevant) <= ids:
            raise ValueError("Each query needs known relevant document IDs")


def evaluate(data, method, k):
    validate(data)
    rows = []
    for query in data["queries"]:
        start = time.perf_counter()
        retrieved = rank(query["text"], data["documents"], method, k)
        elapsed = (time.perf_counter() - start) * 1000
        relevant = set(query["relevant"])
        hits = relevant & set(retrieved)
        reciprocal = next(
            (1 / i for i, doc in enumerate(retrieved, 1) if doc in relevant), 0
        )
        rows.append(
            dict(
                id=query["id"],
                retrieved=retrieved,
                recall=len(hits) / len(relevant),
                reciprocal_rank=reciprocal,
                latency_ms=round(elapsed, 4),
            )
        )
    return dict(
        method=method,
        k=k,
        recall_at_k=sum(r["recall"] for r in rows) / len(rows),
        mrr_at_k=sum(r["reciprocal_rank"] for r in rows) / len(rows),
        queries=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0)
    args = parser.parse_args()
    try:
        if not 0 <= args.min_recall <= 1:
            raise ValueError("min-recall must be between 0 and 1")
        data = json.loads(args.dataset.read_text())
        reports = [evaluate(data, method, args.k) for method in ("overlap", "bm25")]
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Invalid input: {error}\n")
    print(json.dumps(reports, indent=2))
    return int(reports[1]["recall_at_k"] < args.min_recall)


if __name__ == "__main__":
    raise SystemExit(main())
