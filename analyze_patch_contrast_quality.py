#!/usr/bin/env python3
"""Analyze patch-aware contrastive retrieval quality before prompting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def iter_matches(row: dict):
    for chunk in row.get("query_chunks", []) or []:
        if not isinstance(chunk, dict):
            continue
        for match in chunk.get("patch_pair_matches", []) or []:
            if isinstance(match, dict):
                yield chunk, match


def best_match(row: dict):
    matches = list(iter_matches(row))
    if not matches:
        return None, None
    matches.sort(
        key=lambda item: (
            -float(item[1].get("pair_relevance", 0.0) or 0.0),
            -float(item[1].get("abs_contrast_margin", 0.0) or 0.0),
            int(item[1].get("patch_pair_rank", 999) or 999),
        )
    )
    return matches[0]


def prediction_from_margin(margin: float | None):
    if margin is None:
        return None
    return 1 if margin > 0 else 0


def binary_metrics(rows: list[dict]) -> dict:
    known = [row for row in rows if row["pred"] is not None and row["target"] is not None]
    tp = sum(1 for row in known if row["pred"] == 1 and row["target"] == 1)
    fp = sum(1 for row in known if row["pred"] == 1 and row["target"] == 0)
    tn = sum(1 for row in known if row["pred"] == 0 and row["target"] == 0)
    fn = sum(1 for row in known if row["pred"] == 0 and row["target"] == 1)
    total = len(known)
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    yes_rate = (tp + fp) / total if total else 0.0
    return {
        "known": total,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": yes_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze patch contrast retrieval margin quality.")
    parser.add_argument("--query-results", required=True, help="query_patch_contrast_results.jsonl")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    rows = []
    with Path(args.query_results).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk, match = best_match(row)
            target = safe_int(row.get("query_target"))
            margin = safe_float(match.get("contrast_margin")) if match else None
            pred = prediction_from_margin(margin)
            rows.append(
                {
                    "idx": row.get("query_idx"),
                    "target": target,
                    "project": row.get("project", ""),
                    "cwe": row.get("cwe", ""),
                    "pred": pred,
                    "correct": int(pred == target) if pred is not None and target is not None else "",
                    "top_closer_side": match.get("closer_side") if match else "",
                    "top_margin": margin,
                    "top_abs_margin": safe_float(match.get("abs_contrast_margin")) if match else None,
                    "top_pair_relevance": safe_float(match.get("pair_relevance")) if match else None,
                    "top_vuln_score": safe_float(match.get("vulnerable_side_score")) if match else None,
                    "top_fixed_score": safe_float(match.get("fixed_side_score")) if match else None,
                    "top_pair_project": match.get("project") if match else "",
                    "top_pair_cwe": match.get("cwe") if match else "",
                    "top_pair_id": match.get("pair_id") if match else "",
                    "focus_start_line": chunk.get("start_line") if chunk else "",
                    "focus_end_line": chunk.get("end_line") if chunk else "",
                    "focus_code": chunk.get("code_clean", "") if chunk else "",
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["idx"])
        writer.writeheader()
        writer.writerows(rows)

    summary = binary_metrics(rows)
    summary["total_rows"] = len(rows)
    summary["unknown_predictions"] = sum(1 for row in rows if row["pred"] is None)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
