#!/usr/bin/env python3
"""Diagnose leakage-free repair verifier candidate filtering.

This script inspects query_patch_contrast_results.jsonl and the prediction CSV
before running the verifier. It reports whether retrieved patch pairs contain
repair-signature fields and which filter removes the candidates.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
STOP_IDENTIFIERS = {
    "if",
    "else",
    "for",
    "while",
    "switch",
    "case",
    "return",
    "sizeof",
    "static",
    "const",
    "struct",
    "int",
    "char",
    "void",
    "long",
    "short",
    "unsigned",
    "signed",
    "bool",
    "true",
    "false",
    "NULL",
    "nullptr",
}


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def code_identifiers(text: Any) -> set[str]:
    return {token for token in IDENT_RE.findall(safe_str(text)) if token not in STOP_IDENTIFIERS}


def code_calls(text: Any) -> set[str]:
    return {token for token in CALL_RE.findall(safe_str(text)) if token not in STOP_IDENTIFIERS}


def normalize_cwe_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value if str(item)}
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return {str(item) for item in parsed if str(item)}
    except Exception:
        pass
    return {part.strip() for part in re.split(r"[|,]", text) if part.strip()}


def load_predictions(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idx = safe_str(row.get("idx")).strip()
            if idx:
                rows[idx] = row
    return rows


def load_query_results(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                idx = safe_str(row.get("query_idx")).strip()
                if idx:
                    rows[idx] = row
    return rows


def flatten_pairs(row: dict) -> list[dict]:
    out = []
    for chunk in row.get("query_chunks", []):
        if not isinstance(chunk, dict):
            continue
        for pair in chunk.get("patch_pair_matches", []):
            if not isinstance(pair, dict):
                continue
            item = dict(pair)
            item["focus_code"] = chunk.get("code_clean", "")
            out.append(item)
    return out


def same_as_query_pair(pair: dict, query_idx: str) -> bool:
    return query_idx in {safe_str(pair.get("vuln_idx")), safe_str(pair.get("fixed_idx"))}


def pair_features(query_row: dict, pair: dict) -> dict:
    repair_text = "\n".join(
        [
            safe_str(pair.get("repair_signature")),
            safe_str(pair.get("patch_diff")),
            safe_str(pair.get("fixed_changed_code")),
            safe_str(pair.get("vuln_changed_code")),
        ]
    )
    focus_tokens = code_identifiers(pair.get("focus_code"))
    repair_tokens = code_identifiers(repair_text)
    focus_calls = code_calls(pair.get("focus_code"))
    repair_calls = code_calls(repair_text)
    query_cwes = normalize_cwe_set(query_row.get("cwe"))
    return {
        "closer_side": safe_str(pair.get("closer_side")),
        "pair_relevance": safe_float(pair.get("pair_relevance"), 0.0) or 0.0,
        "abs_contrast_margin": safe_float(pair.get("abs_contrast_margin"), 0.0) or 0.0,
        "has_real_repair": as_bool(pair.get("has_pair_diff")) or bool(pair.get("patch_diff")),
        "has_fixed_added_lines": as_bool(pair.get("has_fixed_added_lines")) or bool(pair.get("repair_added_lines")),
        "same_project": safe_str(query_row.get("project")) == safe_str(pair.get("project")) and safe_str(pair.get("project")) != "",
        "same_cwe": bool(query_cwes & normalize_cwe_set(pair.get("cwe"))),
        "repair_token_overlap": len(focus_tokens & repair_tokens),
        "repair_call_overlap": len(focus_calls & repair_calls),
        "has_repair_signature": bool(pair.get("repair_signature")),
        "has_patch_diff": bool(pair.get("patch_diff")),
        "has_added_lines_field": bool(pair.get("repair_added_lines")),
        "pair_id": pair.get("pair_id"),
        "project": pair.get("project"),
        "cwe": pair.get("cwe"),
        "repair_signal": pair.get("repair_signal"),
    }


def first_failure(feat: dict, args) -> str:
    if feat["pair_relevance"] < args.min_relevance:
        return "min_relevance"
    if feat["abs_contrast_margin"] < args.min_abs_margin:
        return "min_abs_margin"
    if args.require_real_repair and not feat["has_real_repair"]:
        return "require_real_repair"
    if args.require_fixed_added_lines and not feat["has_fixed_added_lines"]:
        return "require_fixed_added_lines"
    if args.require_same_cwe and not feat["same_cwe"]:
        return "require_same_cwe"
    if args.require_same_project and not feat["same_project"]:
        return "require_same_project"
    if feat["repair_token_overlap"] < args.min_repair_token_overlap:
        return "min_repair_token_overlap"
    if args.mode == "fixed_closer" and feat["closer_side"] != "fixed":
        return "mode_fixed_closer"
    if args.mode == "fixed_retrieved" and feat["closer_side"] not in {"fixed", "vulnerable"}:
        return "mode_fixed_retrieved"
    if args.mode == "both_sides":
        # Cannot fully infer both-side retrieval from compact fields here, so leave pass-through.
        return "pass"
    if args.mode == "fixed_closer_both_sides" and feat["closer_side"] != "fixed":
        return "mode_fixed_closer_both_sides"
    return "pass"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose repair query result filtering.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--query-results", required=True)
    parser.add_argument("--base-pred-column", default="best_ensemble_pred")
    parser.add_argument(
        "--mode",
        choices=["fixed_closer", "fixed_retrieved", "both_sides", "fixed_closer_both_sides"],
        default="fixed_closer",
    )
    parser.add_argument("--min-relevance", type=float, default=0.0)
    parser.add_argument("--min-abs-margin", type=float, default=0.0)
    parser.add_argument("--require-real-repair", action="store_true")
    parser.add_argument("--require-fixed-added-lines", action="store_true")
    parser.add_argument("--require-same-cwe", action="store_true")
    parser.add_argument("--require-same-project", action="store_true")
    parser.add_argument("--min-repair-token-overlap", type=int, default=0)
    parser.add_argument("--sample-failures", type=int, default=5)
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions_csv))
    query_results = load_query_results(Path(args.query_results))

    summary = Counter()
    pair_field_counts = Counter()
    first_failure_counts = Counter()
    best_possible_counts = Counter()
    examples = []

    for idx, query_row in sorted(query_results.items(), key=lambda item: safe_int(item[0], item[0])):
        pred_row = predictions.get(idx)
        if pred_row is None:
            summary["missing_prediction"] += 1
            continue
        base_pred = safe_int(pred_row.get(args.base_pred_column))
        if base_pred != 1:
            summary["base_not_yes"] += 1
            continue
        summary["base_yes"] += 1
        pairs = [p for p in flatten_pairs(query_row) if not same_as_query_pair(p, idx)]
        if not pairs:
            summary["no_pairs"] += 1
            continue
        summary["rows_with_pairs"] += 1

        row_pass = False
        row_failures = Counter()
        feats = []
        for pair in pairs:
            feat = pair_features(query_row, pair)
            feats.append(feat)
            for key in [
                "has_real_repair",
                "has_fixed_added_lines",
                "same_project",
                "same_cwe",
                "has_repair_signature",
                "has_patch_diff",
                "has_added_lines_field",
            ]:
                if feat[key]:
                    pair_field_counts[key] += 1
            if feat["repair_token_overlap"] > 0:
                pair_field_counts["repair_token_overlap_gt0"] += 1
            if feat["repair_call_overlap"] > 0:
                pair_field_counts["repair_call_overlap_gt0"] += 1
            fail = first_failure(feat, args)
            row_failures[fail] += 1
            if fail == "pass":
                row_pass = True

        if row_pass:
            summary["rows_passing"] += 1
        else:
            summary["rows_failing"] += 1
            dominant = row_failures.most_common(1)[0][0] if row_failures else "unknown"
            first_failure_counts[dominant] += 1
            if len(examples) < args.sample_failures:
                best = sorted(
                    feats,
                    key=lambda feat: (
                        -int(feat["has_real_repair"]),
                        -int(feat["has_fixed_added_lines"]),
                        -feat["repair_token_overlap"],
                        -feat["pair_relevance"],
                    ),
                )[0]
                examples.append({"idx": idx, "target_eval_only": query_row.get("query_target"), "dominant_failure": dominant, "best_pair": best})

        if any(feat["has_real_repair"] for feat in feats):
            best_possible_counts["row_has_real_repair"] += 1
        if any(feat["repair_token_overlap"] > 0 for feat in feats):
            best_possible_counts["row_has_token_overlap"] += 1
        if any(feat["has_real_repair"] and feat["repair_token_overlap"] > 0 for feat in feats):
            best_possible_counts["row_has_real_repair_and_overlap"] += 1
        if any(feat["closer_side"] == "fixed" for feat in feats):
            best_possible_counts["row_has_fixed_closer"] += 1

    output = {
        "summary": dict(summary.most_common()),
        "pair_field_counts": dict(pair_field_counts.most_common()),
        "row_best_possible_counts": dict(best_possible_counts.most_common()),
        "dominant_failure_counts": dict(first_failure_counts.most_common()),
        "sample_failures": examples,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
