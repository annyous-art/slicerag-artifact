#!/usr/bin/env python3
"""Summarize SliceRAG query retrieval quality before LLM prompting."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROLE_COLUMNS = [
    "vulnerable_changed",
    "fixed_changed",
    "safe_background",
    "vulnerable_context",
    "unknown",
]


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evidence_role(match: dict) -> str:
    role = match.get("evidence_role")
    if role:
        return str(role)

    chunk_label = match.get("index_chunk_label", match.get("chunk_label"))
    function_target = match.get("function_target")
    chunk_label = safe_int(chunk_label)
    function_target = safe_int(function_target)
    if function_target == 1 and chunk_label == 1:
        return "vulnerable_changed"
    if function_target == 0 and chunk_label == 1:
        return "fixed_changed"
    if function_target == 0:
        return "safe_background"
    if function_target == 1:
        return "vulnerable_context"
    return "unknown"


def evidence_polarity(match: dict):
    polarity = match.get("evidence_polarity")
    if polarity is not None:
        return safe_int(polarity)

    role = evidence_role(match)
    if role == "vulnerable_changed":
        return 1
    if role in ("fixed_changed", "safe_background"):
        return 0
    return None


def normalize_cwe(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def flatten_matches(query_chunks: list[dict]) -> list[dict]:
    matches: list[dict] = []
    for chunk in query_chunks:
        for match in chunk.get("index_matches") or []:
            if isinstance(match, dict):
                matches.append(match)
    return sorted(matches, key=lambda item: safe_int(item.get("index_rank"), 999999))


def score_of_match(match: dict) -> float:
    return safe_float(match.get("faiss_score", match.get("index_score")), 0.0)


def target_agrees_with_polarity(target, polarity) -> str:
    target = safe_int(target)
    if target is None or polarity is None:
        return "unknown"
    return "yes" if target == polarity else "no"


def summarize_record(record: dict) -> dict:
    query_chunks = record.get("query_chunks") or []
    matches = flatten_matches(query_chunks)
    target = safe_int(record.get("query_target"))
    query_project = record.get("query_project", "")
    query_cwe = normalize_cwe(record.get("query_cwe", ""))
    query_func = record.get("query_func") or ""

    role_counts = Counter(evidence_role(match) for match in matches)
    polarity_counts = Counter(evidence_polarity(match) for match in matches)
    scores = [score_of_match(match) for match in matches]
    sorted_scores = sorted(scores, reverse=True)
    top1 = max(matches, key=score_of_match) if matches else {}
    top1_role = evidence_role(top1) if top1 else "none"
    top1_polarity = evidence_polarity(top1) if top1 else None

    chunk_margins = [safe_float(chunk.get("pos_neg_margin"), 0.0) for chunk in query_chunks]
    top_pos_scores = [safe_float(chunk.get("top_pos_score"), 0.0) for chunk in query_chunks]
    top_neg_scores = [safe_float(chunk.get("top_neg_score"), 0.0) for chunk in query_chunks]

    same_project = 0
    same_cwe = 0
    for match in matches:
        if query_project and match.get("project") == query_project:
            same_project += 1
        match_cwe = normalize_cwe(match.get("cwe"))
        if query_cwe and match_cwe and query_cwe == match_cwe:
            same_cwe += 1

    row = {
        "query_idx": record.get("query_idx"),
        "query_func_id": record.get("query_func_id"),
        "target": target,
        "project": query_project,
        "cwe": query_cwe,
        "func_char_len": len(query_func),
        "func_line_count": len(query_func.splitlines()),
        "num_selected_query_chunks": len(query_chunks),
        "num_index_matches": len(matches),
        "top1_score": score_of_match(top1) if top1 else 0.0,
        "top2_score": sorted_scores[1] if len(sorted_scores) > 1 else 0.0,
        "score_gap_top1_top2": (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0,
        "mean_match_score": mean(scores) if scores else 0.0,
        "top_pos_score_max": max(top_pos_scores) if top_pos_scores else 0.0,
        "top_neg_score_max": max(top_neg_scores) if top_neg_scores else 0.0,
        "pos_neg_margin_max": max(chunk_margins) if chunk_margins else 0.0,
        "pos_neg_margin_mean": mean(chunk_margins) if chunk_margins else 0.0,
        "positive_evidence_count": polarity_counts.get(1, 0),
        "negative_evidence_count": polarity_counts.get(0, 0),
        "unknown_evidence_count": polarity_counts.get(None, 0),
        "top1_evidence_role": top1_role,
        "top1_evidence_polarity": top1_polarity if top1_polarity is not None else "",
        "top1_polarity_agrees_target": target_agrees_with_polarity(target, top1_polarity),
        "same_project_match_count": same_project,
        "same_cwe_match_count": same_cwe,
    }
    for role in ROLE_COLUMNS:
        row[f"role_count_{role}"] = role_counts.get(role, 0)
    return row


def write_outputs(rows: list[dict], csv_path: Path, summary_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "num_samples": len(rows),
        "target_counts": dict(Counter(str(row["target"]) for row in rows)),
        "top1_agreement_counts": dict(Counter(row["top1_polarity_agrees_target"] for row in rows)),
        "top1_role_counts": dict(Counter(row["top1_evidence_role"] for row in rows)),
        "avg_num_index_matches": mean(row["num_index_matches"] for row in rows) if rows else 0.0,
        "avg_positive_evidence_count": mean(row["positive_evidence_count"] for row in rows) if rows else 0.0,
        "avg_negative_evidence_count": mean(row["negative_evidence_count"] for row in rows) if rows else 0.0,
        "avg_pos_neg_margin_max": mean(row["pos_neg_margin_max"] for row in rows) if rows else 0.0,
        "avg_pos_neg_margin_mean": mean(row["pos_neg_margin_mean"] for row in rows) if rows else 0.0,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SliceRAG query retrieval quality.")
    parser.add_argument("--query-results", required=True, help="Path to query_results.jsonl")
    parser.add_argument("--output-csv", required=True, help="Path for per-sample retrieval quality CSV")
    parser.add_argument("--summary-json", required=True, help="Path for aggregate summary JSON")
    args = parser.parse_args()

    rows = []
    with Path(args.query_results).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(summarize_record(json.loads(line)))

    write_outputs(rows, Path(args.output_csv), Path(args.summary_json))
    print(f"Wrote retrieval quality CSV: {args.output_csv}")
    print(f"Wrote retrieval quality summary: {args.summary_json}")


if __name__ == "__main__":
    main()
