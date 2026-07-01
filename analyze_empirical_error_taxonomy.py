#!/usr/bin/env python3
"""Build error-taxonomy tables for the SliceRAG empirical study.

The goal is not to invent a perfect automatic oracle. This script creates a
consistent, auditable taxonomy from existing per-sample predictions, retrieval
features, keyword features, and optional repair-verifier outputs so that the
remaining manual inspection can focus on representative groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METHOD_COLUMNS = [
    "zero_shot_pred",
    "yes_only_pred",
    "author_no_yes_pred",
    "old_rag_pred",
    "positive_weighted_rag_pred",
    "patch_contrast_pred",
    "best_ensemble_pred",
]


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "idx",
        "target",
        "base_pred",
        "error_type",
        "taxonomy",
        "dominant_pattern",
        "yes_vote_count",
        "known_vote_count",
        "project",
        "cwe",
        "char_len",
        "line_count",
        "retrieval_top1_score",
        "retrieval_pos_neg_margin_max",
    ]
    fields = sorted({key for row in rows for key in row})
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl_by_idx(path: Path) -> dict[str, dict]:
    rows = {}
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = row.get("idx", row.get("query_idx"))
            if idx is not None:
                rows[str(idx)] = row
    return rows


def length_bucket(char_len: int) -> str:
    if char_len < 2000:
        return "short_<2k"
    if char_len < 5000:
        return "medium_2k_5k"
    if char_len < 10000:
        return "long_5k_10k"
    return "very_long_10k_plus"


def dominant_pattern(row: dict) -> str:
    counts = {
        "bounds": safe_int(row.get("bounds_keyword_count"), 0) or 0,
        "error_handling": safe_int(row.get("error_handling_keyword_count"), 0) or 0,
        "lifecycle": safe_int(row.get("lifecycle_keyword_count"), 0) or 0,
        "null": safe_int(row.get("null_keyword_count"), 0) or 0,
        "memory_api": safe_int(row.get("memory_api_count"), 0) or 0,
    }
    best, value = max(counts.items(), key=lambda item: item[1])
    return best if value > 0 else "low_keyword_signal"


def vote_counts(row: dict) -> tuple[int, int]:
    known = yes = 0
    for column in METHOD_COLUMNS:
        pred = safe_int(row.get(column))
        if pred in (0, 1):
            known += 1
            yes += int(pred == 1)
    return yes, known


def method_pattern(row: dict) -> str:
    yes, known = vote_counts(row)
    if known == 0:
        return "no_known_method_votes"
    if yes == known:
        return "all_methods_yes"
    if yes == 0:
        return "all_methods_no"
    if yes >= max(1, known - 1):
        return "near_consensus_yes"
    if yes <= 1:
        return "near_consensus_no"
    return "method_disagreement"


def retrieval_bucket(row: dict) -> str:
    top1 = safe_float(row.get("retrieval_top1_score"))
    margin = safe_float(row.get("retrieval_pos_neg_margin_max"))
    if math.isnan(top1):
        return "retrieval_missing"
    if not math.isnan(margin):
        if abs(margin) < 0.01:
            return "retrieval_low_margin"
        if margin > 0:
            return "retrieval_pos_margin"
        return "retrieval_neg_margin"
    if top1 >= 0.9:
        return "retrieval_high_similarity_unknown_margin"
    return "retrieval_low_similarity_unknown_margin"


def repair_signal(row: dict, repair_rows: dict[str, dict]) -> str:
    repair = repair_rows.get(str(row.get("idx")))
    if not repair:
        return "no_repair_verifier_candidate"
    pred = safe_int(repair.get("repair_verifier_pred"))
    present = repair.get("repair_present")
    rtype = safe_str(repair.get("repair_type") or "unknown")
    if pred == 0 and present is True:
        return f"repair_present_override_candidate:{rtype}"
    if pred == 1:
        return f"repair_checked_absent_or_irrelevant:{rtype}"
    return f"repair_unknown:{rtype}"


def taxonomy_for_row(row: dict, base_pred_column: str, repair_rows: dict[str, dict]) -> tuple[str, str]:
    target = safe_int(row.get("target"))
    pred = safe_int(row.get(base_pred_column))
    if target not in (0, 1) or pred not in (0, 1):
        return "unknown", "unknown_prediction"
    if target == pred:
        return "correct", "correct"

    pattern = dominant_pattern(row)
    methods = method_pattern(row)
    retrieval = retrieval_bucket(row)
    repair = repair_signal(row, repair_rows)
    char_len = safe_int(row.get("char_len"), 0) or 0

    if target == 0 and pred == 1:
        if methods in {"all_methods_yes", "near_consensus_yes"}:
            return "FP", "yes_prior_consensus_false_positive"
        if repair.startswith("repair_present_override_candidate"):
            return "FP", "repair_present_but_not_applied_or_policy_blocked"
        if pattern in {"bounds", "memory_api", "error_handling", "lifecycle", "null"}:
            return "FP", f"risky_pattern_false_alarm:{pattern}"
        if retrieval in {"retrieval_low_margin", "retrieval_pos_margin"}:
            return "FP", f"retrieval_misleading_or_ambiguous:{retrieval}"
        if char_len >= 5000:
            return "FP", "long_function_false_positive"
        return "FP", "other_false_positive"

    if target == 1 and pred == 0:
        if methods in {"all_methods_no", "near_consensus_no"}:
            return "FN", "conservative_consensus_false_negative"
        if char_len >= 5000:
            return "FN", "long_context_false_negative"
        if pattern in {"lifecycle", "error_handling", "bounds", "null", "memory_api"}:
            return "FN", f"semantic_pattern_false_negative:{pattern}"
        if retrieval in {"retrieval_low_margin", "retrieval_neg_margin"}:
            return "FN", f"retrieval_misleading_or_ambiguous:{retrieval}"
        return "FN", "other_false_negative"

    return "unknown", "unknown"


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)
    out = []
    for key, group_rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        targets = Counter(safe_str(row.get("target")) for row in group_rows)
        preds = Counter(safe_str(row.get("base_pred")) for row in group_rows)
        out.append(
            {
                **{group_keys[i]: key[i] for i in range(len(group_keys))},
                "n": len(group_rows),
                "target_0": targets.get("0", 0),
                "target_1": targets.get("1", 0),
                "pred_0": preds.get("0", 0),
                "pred_1": preds.get("1", 0),
                "avg_char_len": round(sum(safe_int(row.get("char_len"), 0) or 0 for row in group_rows) / len(group_rows), 2),
                "avg_yes_vote_count": round(sum(safe_int(row.get("yes_vote_count"), 0) or 0 for row in group_rows) / len(group_rows), 2),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze empirical error taxonomy.")
    parser.add_argument("--predictions-csv", default="outdir/ensemble_error_patch_contrast/per_sample_predictions.csv")
    parser.add_argument("--base-pred-column", default="best_ensemble_pred")
    parser.add_argument("--repair-verifier-jsonl", default="")
    parser.add_argument("--output-dir", default="outdir/empirical_study/error_taxonomy")
    args = parser.parse_args()

    rows = read_csv(Path(args.predictions_csv))
    repair_rows = load_jsonl_by_idx(Path(args.repair_verifier_jsonl)) if args.repair_verifier_jsonl else {}
    out_rows = []

    for row in rows:
        out = dict(row)
        target = safe_int(row.get("target"))
        pred = safe_int(row.get(args.base_pred_column))
        yes, known = vote_counts(row)
        error_type, taxonomy = taxonomy_for_row(row, args.base_pred_column, repair_rows)
        out["base_pred"] = pred if pred in (0, 1) else ""
        out["error_type"] = error_type
        out["taxonomy"] = taxonomy
        out["dominant_pattern"] = dominant_pattern(row)
        out["method_pattern"] = method_pattern(row)
        out["retrieval_bucket"] = retrieval_bucket(row)
        out["repair_signal"] = repair_signal(row, repair_rows)
        out["length_bucket"] = length_bucket(safe_int(row.get("char_len"), 0) or 0)
        out["yes_vote_count"] = yes
        out["known_vote_count"] = known
        out["base_correct"] = int(target == pred) if target in (0, 1) and pred in (0, 1) else ""
        out_rows.append(out)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_sample_error_taxonomy.csv", out_rows)
    write_csv(output_dir / "taxonomy_summary.csv", summarize(out_rows, ["error_type", "taxonomy"]))
    write_csv(output_dir / "taxonomy_by_project.csv", summarize(out_rows, ["error_type", "project"]))
    write_csv(output_dir / "taxonomy_by_cwe.csv", summarize(out_rows, ["error_type", "cwe"]))
    write_csv(output_dir / "taxonomy_by_pattern.csv", summarize(out_rows, ["error_type", "dominant_pattern"]))
    write_csv(output_dir / "taxonomy_by_length.csv", summarize(out_rows, ["error_type", "length_bucket"]))

    summary = {
        "predictions_csv": args.predictions_csv,
        "base_pred_column": args.base_pred_column,
        "repair_verifier_jsonl": args.repair_verifier_jsonl,
        "n": len(out_rows),
        "error_type_counts": dict(Counter(row["error_type"] for row in out_rows).most_common()),
        "taxonomy_counts": dict(Counter(row["taxonomy"] for row in out_rows).most_common()),
        "method_pattern_counts": dict(Counter(row["method_pattern"] for row in out_rows).most_common()),
        "retrieval_bucket_counts": dict(Counter(row["retrieval_bucket"] for row in out_rows).most_common()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
