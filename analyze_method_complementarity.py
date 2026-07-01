#!/usr/bin/env python3
"""Analyze samples uniquely corrected by RAG/ICL methods versus a baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_METHODS = [
    "author_no_yes",
    "current_yes_no",
    "old_rag",
    "positive_weighted_rag",
    "balanced_grouped_no_label_bc",
    "icl",
]

NUMERIC_FEATURES = [
    "char_len",
    "token_len_approx",
    "line_count",
    "macro_count",
    "preproc_count",
    "call_count",
    "unique_call_count",
    "loop_count",
    "cond_count",
    "ptr_count",
    "memory_api_count",
    "string_api_count",
    "bounds_keyword_count",
    "error_handling_keyword_count",
    "lifecycle_keyword_count",
    "null_keyword_count",
    "retrieval_top1_score",
    "retrieval_score_gap_top1_top2",
    "retrieval_mean_match_score",
    "retrieval_top_pos_score_max",
    "retrieval_top_neg_score_max",
    "retrieval_pos_neg_margin_max",
    "retrieval_pos_neg_margin_mean",
    "retrieval_positive_evidence_count",
    "retrieval_negative_evidence_count",
    "retrieval_unknown_evidence_count",
    "retrieval_same_project_match_count",
    "retrieval_same_cwe_match_count",
    "method_yes_rate",
    "method_pred_entropy",
    "icl_message_chars",
    "old_rag_message_chars",
    "positive_weighted_rag_message_chars",
    "balanced_grouped_no_label_bc_message_chars",
]

CATEGORICAL_FEATURES = [
    "project",
    "cwe",
    "retrieval_top1_evidence_role",
    "retrieval_top1_evidence_polarity",
]


def safe_float(value: Any):
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any):
    try:
        if value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def summarize_numeric(rows: list[dict], features: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for feature in features:
        values = [safe_float(row.get(feature)) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            continue
        summary[feature] = {
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def summarize_categorical(rows: list[dict], features: list[str], topn: int) -> dict[str, list[dict[str, Any]]]:
    summary = {}
    for feature in features:
        counts = Counter(str(row.get(feature, "")) for row in rows)
        counts.pop("", None)
        summary[feature] = [
            {"value": value, "count": count, "ratio": count / len(rows) if rows else 0.0}
            for value, count in counts.most_common(topn)
        ]
    return summary


def rate(rows: list[dict], feature: str) -> float:
    values = [safe_float(row.get(feature)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else 0.0


def classify_group(row: dict, baseline: str, method: str) -> str | None:
    target = safe_int(row.get("target"))
    base_pred = safe_int(row.get(f"{baseline}_pred"))
    method_pred = safe_int(row.get(f"{method}_pred"))
    if target not in (0, 1) or base_pred not in (0, 1) or method_pred not in (0, 1):
        return None
    base_correct = base_pred == target
    method_correct = method_pred == target
    if base_correct and method_correct:
        return "both_correct"
    if base_correct and not method_correct:
        return "baseline_only"
    if not base_correct and method_correct:
        return "method_only"
    return "both_wrong"


def summarize_groups(rows: list[dict], baseline: str, method: str, topn: int) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group = classify_group(row, baseline, method)
        if group:
            groups[group].append(row)

    summary: dict[str, Any] = {
        "baseline": baseline,
        "method": method,
        "counts": {name: len(groups.get(name, [])) for name in ["both_correct", "baseline_only", "method_only", "both_wrong"]},
    }
    total = sum(summary["counts"].values())
    summary["oracle_accuracy"] = (
        (summary["counts"]["both_correct"] + summary["counts"]["baseline_only"] + summary["counts"]["method_only"]) / total
        if total
        else 0.0
    )
    summary["groups"] = {}
    for name in ["both_correct", "baseline_only", "method_only", "both_wrong"]:
        group_rows = groups.get(name, [])
        summary["groups"][name] = {
            "n": len(group_rows),
            "target_positive_rate": rate(group_rows, "target"),
            "baseline_yes_rate": rate(group_rows, f"{baseline}_pred"),
            "method_yes_rate": rate(group_rows, f"{method}_pred"),
            "numeric": summarize_numeric(group_rows, NUMERIC_FEATURES),
            "categorical": summarize_categorical(group_rows, CATEGORICAL_FEATURES, topn),
        }
    return summary


def write_group_rows(path: Path, rows: list[dict], baseline: str, method: str) -> None:
    selected_columns = [
        "sample_key",
        "idx",
        "target",
        "project",
        "cwe",
        "char_len",
        "line_count",
        "memory_api_count",
        "string_api_count",
        "bounds_keyword_count",
        "error_handling_keyword_count",
        "lifecycle_keyword_count",
        "retrieval_top1_score",
        "retrieval_pos_neg_margin_max",
        "retrieval_top1_evidence_role",
        f"{baseline}_pred",
        f"{method}_pred",
        "zero_shot_pred",
        "icl_pred",
        "old_rag_pred",
        "positive_weighted_rag_pred",
        "balanced_grouped_no_label_bc_pred",
        "icl_message_chars",
        "method_pred_entropy",
    ]
    selected_columns = list(dict.fromkeys(selected_columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze method complementarity against a baseline.")
    parser.add_argument("--features-csv", default="outdir/analysis/per_sample_feature_table.csv")
    parser.add_argument("--baseline", default="zero_shot")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--output-dir", default="outdir/complementarity")
    parser.add_argument("--topn", type=int, default=12)
    args = parser.parse_args()

    with Path(args.features_csv).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for method in args.methods:
        if f"{method}_pred" not in rows[0]:
            print(f"Warning: skipping missing method: {method}")
            continue
        summary = summarize_groups(rows, args.baseline, method, args.topn)
        all_summaries[method] = summary

        method_dir = output_dir / f"{args.baseline}_vs_{method}"
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        for group_name in ["both_correct", "baseline_only", "method_only", "both_wrong"]:
            group_rows = [row for row in rows if classify_group(row, args.baseline, method) == group_name]
            write_group_rows(method_dir / f"{group_name}.csv", group_rows, args.baseline, method)

    (output_dir / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("method\tboth_correct\tbaseline_only\tmethod_only\tboth_wrong\toracle_accuracy")
    for method, summary in all_summaries.items():
        counts = summary["counts"]
        print(
            f"{method}\t{counts['both_correct']}\t{counts['baseline_only']}\t"
            f"{counts['method_only']}\t{counts['both_wrong']}\t{summary['oracle_accuracy']:.4f}"
        )
    print(f"Wrote complementarity analysis to: {output_dir}")


if __name__ == "__main__":
    main()
