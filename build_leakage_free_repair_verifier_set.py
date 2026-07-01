#!/usr/bin/env python3
"""Build leakage-free repair-presence verifier candidates.

This script does not use the target label to select candidates and does not use
the target sample's own paired diff as verifier evidence. It joins first-stage
predictions with patch-contrast retrieval results, then emits only first-stage
YES candidates whose retrieved training patch evidence suggests that a
repair-side comparison may be useful.
"""

from __future__ import annotations

import argparse
import csv
import ast
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


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default=None):
    try:
        return int(value)
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
            if not line.strip():
                continue
            row = json.loads(line)
            idx = safe_str(row.get("query_idx")).strip()
            if idx:
                rows[idx] = row
    return rows


def truncate(text: Any, max_chars: int) -> str:
    text = safe_str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n/* ... truncated ... */"


def pair_side_score(pair: dict, side: str) -> float | None:
    if side == "fixed":
        return safe_float(pair.get("fixed_side_score"))
    if side == "vulnerable":
        return safe_float(pair.get("vulnerable_side_score"))
    return safe_float(pair.get("pair_relevance"))


def flatten_pairs(row: dict) -> list[dict]:
    out = []
    for chunk in row.get("query_chunks", []):
        if not isinstance(chunk, dict):
            continue
        for pair in chunk.get("patch_pair_matches", []):
            if not isinstance(pair, dict):
                continue
            item = dict(pair)
            item["focus_rank"] = chunk.get("query_rank")
            item["focus_start_line"] = chunk.get("start_line")
            item["focus_end_line"] = chunk.get("end_line")
            item["focus_code"] = chunk.get("code_clean", "")
            out.append(item)
    return out


def same_as_query_pair(pair: dict, query_idx: str) -> bool:
    return query_idx in {
        safe_str(pair.get("vuln_idx")),
        safe_str(pair.get("fixed_idx")),
    }


def select_pairs(
    query_row: dict,
    mode: str,
    max_pairs: int,
    min_relevance: float,
    min_abs_margin: float,
    exclude_same_idx: bool,
    require_real_repair: bool,
    require_fixed_added_lines: bool,
    require_same_cwe: bool,
    require_same_project: bool,
    min_repair_token_overlap: int,
) -> list[dict]:
    query_idx = safe_str(query_row.get("query_idx"))
    query_project = safe_str(query_row.get("project"))
    query_cwes = normalize_cwe_set(query_row.get("cwe"))
    pairs = []
    seen_pair_ids = set()
    for pair in flatten_pairs(query_row):
        if exclude_same_idx and same_as_query_pair(pair, query_idx):
            continue
        pair_key = safe_str(pair.get("pair_id") or pair.get("vector_id"))
        if pair_key in seen_pair_ids:
            continue
        seen_pair_ids.add(pair_key)
        relevance = safe_float(pair.get("pair_relevance"), 0.0) or 0.0
        abs_margin = safe_float(pair.get("abs_contrast_margin"), 0.0) or 0.0
        closer_side = safe_str(pair.get("closer_side"))
        fixed_score = safe_float(pair.get("fixed_side_score"))
        vuln_score = safe_float(pair.get("vulnerable_side_score"))
        same_project = query_project != "" and query_project == safe_str(pair.get("project"))
        same_cwe = bool(query_cwes & normalize_cwe_set(pair.get("cwe")))
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
        token_overlap = len(focus_tokens & repair_tokens)
        call_overlap = len(focus_calls & repair_calls)
        has_real_repair = as_bool(pair.get("has_pair_diff")) or bool(pair.get("patch_diff"))
        has_fixed_added = as_bool(pair.get("has_fixed_added_lines")) or bool(pair.get("repair_added_lines"))

        if relevance < min_relevance or abs_margin < min_abs_margin:
            continue
        if require_real_repair and not has_real_repair:
            continue
        if require_fixed_added_lines and not has_fixed_added:
            continue
        if require_same_cwe and not same_cwe:
            continue
        if require_same_project and not same_project:
            continue
        if token_overlap < min_repair_token_overlap:
            continue
        if mode == "fixed_closer" and closer_side != "fixed":
            continue
        if mode == "fixed_retrieved" and fixed_score is None:
            continue
        if mode == "both_sides" and (fixed_score is None or vuln_score is None):
            continue
        if mode == "fixed_closer_both_sides" and (
            closer_side != "fixed" or fixed_score is None or vuln_score is None
        ):
            continue
        pair["same_project"] = same_project
        pair["same_cwe"] = same_cwe
        pair["repair_token_overlap"] = token_overlap
        pair["repair_call_overlap"] = call_overlap
        pair["has_real_repair"] = has_real_repair
        pair["has_fixed_added_lines"] = has_fixed_added
        pairs.append(pair)

    pairs.sort(
        key=lambda pair: (
            safe_str(pair.get("closer_side")) != "fixed",
            -int(bool(pair.get("same_cwe"))),
            -int(bool(pair.get("same_project"))),
            -safe_int(pair.get("repair_call_overlap"), 0),
            -safe_int(pair.get("repair_token_overlap"), 0),
            -int(bool(pair.get("has_fixed_added_lines"))),
            -(safe_float(pair.get("pair_relevance"), 0.0) or 0.0),
            -(safe_float(pair.get("abs_contrast_margin"), 0.0) or 0.0),
            safe_int(pair.get("vector_id"), 10**12),
        )
    )
    return pairs[:max_pairs]


def build_repair_evidence(pairs: list[dict], max_evidence_chars: int) -> str:
    blocks = []
    for i, pair in enumerate(pairs, start=1):
        blocks.extend(
            [
                f"Retrieved training patch pair {i}:",
                f"pair_id: {pair.get('pair_id', '')}",
                f"project: {pair.get('project', '')}",
                f"CWE: {pair.get('cwe', '')}",
                f"same project as target: {pair.get('same_project', '')}",
                f"same CWE as target: {pair.get('same_cwe', '')}",
                f"repair signal: {pair.get('repair_signal', '')}",
                f"repair token overlap with target focus: {pair.get('repair_token_overlap', '')}",
                f"repair call overlap with target focus: {pair.get('repair_call_overlap', '')}",
                f"target-to-vulnerable score: {pair.get('vulnerable_side_score')}",
                f"target-to-fixed score: {pair.get('fixed_side_score')}",
                f"contrast margin (vulnerable - fixed): {pair.get('contrast_margin')}",
                f"closer side: {pair.get('closer_side', '')}",
                "",
                "Repair signature:",
                truncate(pair.get("repair_signature", ""), max_evidence_chars),
                "",
                "Patch diff summary:",
                truncate(pair.get("patch_diff", ""), max_evidence_chars),
                "",
                "Vulnerable-side changed region before fix:",
                truncate(pair.get("vuln_changed_code", ""), max_evidence_chars),
                "",
                "Fixed-side changed region after repair:",
                truncate(pair.get("fixed_changed_code", ""), max_evidence_chars),
                "",
            ]
        )
    return "\n".join(blocks).strip()


def best_pair_summary(pairs: list[dict]) -> dict:
    if not pairs:
        return {}
    pair = pairs[0]
    return {
        "top_pair_id": pair.get("pair_id", ""),
        "top_pair_project": pair.get("project", ""),
        "top_pair_cwe": pair.get("cwe", ""),
        "top_pair_closer_side": pair.get("closer_side", ""),
        "top_pair_relevance": pair.get("pair_relevance", ""),
        "top_pair_abs_margin": pair.get("abs_contrast_margin", ""),
        "top_pair_contrast_margin": pair.get("contrast_margin", ""),
        "top_pair_vuln_idx": pair.get("vuln_idx", ""),
        "top_pair_fixed_idx": pair.get("fixed_idx", ""),
        "top_pair_same_project": pair.get("same_project", ""),
        "top_pair_same_cwe": pair.get("same_cwe", ""),
        "top_pair_repair_signal": pair.get("repair_signal", ""),
        "top_pair_repair_token_overlap": pair.get("repair_token_overlap", ""),
        "top_pair_repair_call_overlap": pair.get("repair_call_overlap", ""),
        "top_pair_has_real_repair": pair.get("has_real_repair", ""),
        "top_pair_has_fixed_added_lines": pair.get("has_fixed_added_lines", ""),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free repair-presence verifier candidates.")
    parser.add_argument("--predictions-csv", default="outdir/ensemble_error_patch_contrast/per_sample_predictions.csv")
    parser.add_argument("--query-results", default="outdir/query_patch_contrast_test870_vuln_margin/query_patch_contrast_results.jsonl")
    parser.add_argument("--output", default="outdir/repair_presence_verifier/leakage_free_candidates.jsonl")
    parser.add_argument("--summary-json", default="outdir/repair_presence_verifier/leakage_free_candidates_summary.json")
    parser.add_argument("--base-pred-column", default="best_ensemble_pred")
    parser.add_argument(
        "--mode",
        choices=["fixed_closer", "fixed_retrieved", "both_sides", "fixed_closer_both_sides"],
        default="fixed_closer",
    )
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--min-relevance", type=float, default=0.0)
    parser.add_argument("--min-abs-margin", type=float, default=0.0)
    parser.add_argument("--require-real-repair", action="store_true")
    parser.add_argument("--require-fixed-added-lines", action="store_true")
    parser.add_argument("--require-same-cwe", action="store_true")
    parser.add_argument("--require-same-project", action="store_true")
    parser.add_argument("--min-repair-token-overlap", type=int, default=0)
    parser.add_argument("--max-function-chars", type=int, default=50000)
    parser.add_argument("--max-evidence-chars", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--exclude-same-idx", action="store_true", default=True)
    parser.add_argument("--allow-same-idx", dest="exclude_same_idx", action="store_false")
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions_csv))
    query_results = load_query_results(Path(args.query_results))

    candidates = []
    skip_counts = Counter()
    trigger_counts = Counter()

    for idx, query_row in sorted(query_results.items(), key=lambda item: safe_int(item[0], item[0])):
        pred_row = predictions.get(idx)
        if pred_row is None:
            skip_counts["missing_prediction"] += 1
            continue
        base_pred = safe_int(pred_row.get(args.base_pred_column))
        if base_pred != 1:
            skip_counts["base_not_yes"] += 1
            continue
        selected_pairs = select_pairs(
            query_row,
            args.mode,
            args.max_pairs,
            args.min_relevance,
            args.min_abs_margin,
            args.exclude_same_idx,
            args.require_real_repair,
            args.require_fixed_added_lines,
            args.require_same_cwe,
            args.require_same_project,
            args.min_repair_token_overlap,
        )
        if not selected_pairs:
            skip_counts["no_trigger_pair"] += 1
            continue

        trigger_counts[selected_pairs[0].get("closer_side", "unknown")] += 1
        target_func = truncate(query_row.get("query_func", ""), args.max_function_chars)
        repair_evidence = build_repair_evidence(selected_pairs, args.max_evidence_chars)
        row = {
            "idx": idx,
            "target": query_row.get("query_target"),
            "project": query_row.get("project", ""),
            "cwe": query_row.get("cwe", ""),
            "target_func": target_func,
            "repair_evidence": repair_evidence,
            "paired_diff": repair_evidence,
            "repair_signal": "retrieved_training_patch",
            "primary_category": "leakage_free_candidate",
            "base_pred_column": args.base_pred_column,
            "base_pred": base_pred,
            "trigger_mode": args.mode,
            "trigger_pair_count": len(selected_pairs),
            "focus_code": selected_pairs[0].get("focus_code", ""),
            "focus_start_line": selected_pairs[0].get("focus_start_line", ""),
            "focus_end_line": selected_pairs[0].get("focus_end_line", ""),
            "retrieved_patch_pairs": selected_pairs,
            **best_pair_summary(selected_pairs),
        }
        candidates.append(row)
        if args.limit and len(candidates) >= args.limit:
            break

    write_jsonl(Path(args.output), candidates)
    target_counts = Counter(str(row.get("target")) for row in candidates)
    summary = {
        "predictions_csv": args.predictions_csv,
        "query_results": args.query_results,
        "base_pred_column": args.base_pred_column,
        "mode": args.mode,
        "max_pairs": args.max_pairs,
        "min_relevance": args.min_relevance,
        "min_abs_margin": args.min_abs_margin,
        "require_real_repair": args.require_real_repair,
        "require_fixed_added_lines": args.require_fixed_added_lines,
        "require_same_cwe": args.require_same_cwe,
        "require_same_project": args.require_same_project,
        "min_repair_token_overlap": args.min_repair_token_overlap,
        "exclude_same_idx": args.exclude_same_idx,
        "num_query_rows": len(query_results),
        "num_prediction_rows": len(predictions),
        "num_candidates": len(candidates),
        "candidate_target_counts_for_evaluation_only": dict(target_counts.most_common()),
        "skip_counts": dict(skip_counts.most_common()),
        "trigger_top_closer_side_counts": dict(trigger_counts.most_common()),
        "top_pair_same_project_counts": dict(Counter(str(row.get("top_pair_same_project")) for row in candidates).most_common()),
        "top_pair_same_cwe_counts": dict(Counter(str(row.get("top_pair_same_cwe")) for row in candidates).most_common()),
        "top_pair_repair_signal_counts": dict(Counter(str(row.get("top_pair_repair_signal")) for row in candidates).most_common(20)),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote candidates: {args.output}")
    print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()
