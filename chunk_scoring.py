#!/usr/bin/env python3
"""Chunk-level scoring helpers for TF-IDF, keyword, and label ratios."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize_code(text: str) -> List[str]:
    if not text:
        return []
    return [token.lower() for token in TOKEN_RE.findall(text)]


def load_keywords(keyword_path: str | Path) -> set[str]:
    path = Path(keyword_path)
    keywords: set[str] = set()
    if not path.exists():
        print(f"WARNING: keyword file not found: {path}. Continuing with an empty keyword set.")
        return keywords
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            keyword = line.strip().lower()
            if keyword:
                keywords.add(keyword)
    return keywords


def min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [0.0 for _ in values]
    span = maximum - minimum
    return [(value - minimum) / span for value in values]


def compute_tfidf_raw_scores(texts: Sequence[str]) -> List[float]:
    tokenized_docs = [tokenize_code(text) for text in texts]
    if not tokenized_docs:
        return []

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    num_docs = len(tokenized_docs)
    idf = {
        token: math.log((1.0 + num_docs) / (1.0 + freq)) + 1.0
        for token, freq in document_frequency.items()
    }

    scores: List[float] = []
    for tokens in tokenized_docs:
        if not tokens:
            scores.append(0.0)
            continue
        token_counts = Counter(tokens)
        total_tokens = sum(token_counts.values())
        if total_tokens <= 0:
            scores.append(0.0)
            continue
        score = 0.0
        for token, count in token_counts.items():
            score += (count / total_tokens) * idf.get(token, 0.0)
        scores.append(score)
    return scores


def compute_keyword_raw_scores(texts: Sequence[str], keywords: set[str]) -> List[float]:
    if not texts:
        return []
    scores: List[float] = []
    for text in texts:
        tokens = set(tokenize_code(text))
        scores.append(float(len(tokens & keywords)))
    return scores


def compute_label_raw_scores(records: Sequence[Dict]) -> List[float]:
    scores: List[float] = []
    for record in records:
        labels = record.get("line_labels") or []
        if not labels:
            scores.append(0.0)
            continue
        diff_count = sum(1 for value in labels if int(value) == 1)
        chunk_len = float(len(labels))
        if chunk_len <= 0:
            scores.append(0.0)
            continue
        target_raw = record.get("function_target", 0)
        try:
            target = int(target_raw)
        except (TypeError, ValueError):
            target = 0
        ratio = diff_count / chunk_len
        if target == 1:
            scores.append(ratio * 100.0)  # Amplify positive labels so the normalized label component remains visible.
        else:
            scores.append(ratio * -1.0)  # Keep negative labels small to avoid adding noise.
    return scores


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def augment_chunk_metadata(
    metadata_path: str | Path,
    keyword_path: str | Path,
    tfidf_weight: float = 1.0,
    keyword_weight: float = 1.0,
    label_weight: float = 1.0,
) -> None:
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    if not records:
        return

    texts = [record.get("code_clean", "") or "" for record in records]
    keywords = load_keywords(keyword_path)

    raw_tfidf_scores = compute_tfidf_raw_scores(texts)
    raw_keyword_scores = compute_keyword_raw_scores(texts, keywords)
    raw_label_scores = compute_label_raw_scores(records)

    normalized_tfidf_scores = min_max_normalize(raw_tfidf_scores)
    normalized_keyword_scores = min_max_normalize(raw_keyword_scores)
    normalized_label_scores = min_max_normalize(raw_label_scores)

    for record, raw_tfidf, raw_keyword, raw_label, norm_tfidf, norm_keyword, norm_label in zip(
        records,
        raw_tfidf_scores,
        raw_keyword_scores,
        raw_label_scores,
        normalized_tfidf_scores,
        normalized_keyword_scores,
        normalized_label_scores,
    ):
        record["score_tfidf_raw"] = raw_tfidf
        record["score_keyword_raw"] = raw_keyword
        record["score_label_raw"] = raw_label
        record["score_tfidf_norm"] = norm_tfidf
        record["score_keyword_norm"] = norm_keyword
        record["score_label_norm"] = norm_label
        record["score_combined"] = weighted_average(
            [norm_tfidf, norm_keyword, norm_label],
            [tfidf_weight, keyword_weight, label_weight],
        )
        record["score_combined_tfidf_keyword"] = weighted_average(
            [norm_tfidf, norm_keyword],
            [tfidf_weight, keyword_weight],
        )
        record["score_combined_tfidf_label"] = weighted_average(
            [norm_tfidf, norm_label],
            [tfidf_weight, label_weight],
        )
        record["score_combined_keyword_label"] = weighted_average(
            [norm_keyword, norm_label],
            [keyword_weight, label_weight],
        )

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
