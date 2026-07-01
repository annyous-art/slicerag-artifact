#!/usr/bin/env python3
"""Compute PrimeVul-style pair-wise metrics for per-sample predictions.

Pair-wise outcomes for a vulnerable/fixed pair:
- P-C: vulnerable predicted YES and fixed predicted NO.
- P-V: both predicted YES.
- P-B: both predicted NO.
- P-R: vulnerable predicted NO and fixed predicted YES.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        text = str(value).strip().upper()
        if text == "YES":
            return 1
        if text == "NO":
            return 0
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv_by_idx(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row["idx"]): row for row in reader}


def adjacent_pairs(
    rows: list[dict[str, Any]],
    require_same_project: bool = False,
    require_same_commit: bool = False,
    require_same_cve: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pairs: list[dict[str, Any]] = []
    stats = {
        "candidate_adjacent_pairs": 0,
        "skipped_incomplete": 0,
        "skipped_missing_target_pair": 0,
        "skipped_project_mismatch": 0,
        "skipped_commit_mismatch": 0,
        "skipped_cve_mismatch": 0,
    }
    for pair_id, start in enumerate(range(0, len(rows), 2)):
        pair_rows = rows[start : start + 2]
        if len(pair_rows) != 2:
            stats["skipped_incomplete"] += 1
            continue
        stats["candidate_adjacent_pairs"] += 1
        vuln = next((row for row in pair_rows if safe_int(row.get("target")) == 1), None)
        fixed = next((row for row in pair_rows if safe_int(row.get("target")) == 0), None)
        if vuln is None or fixed is None:
            stats["skipped_missing_target_pair"] += 1
            continue
        if require_same_project and vuln.get("project") != fixed.get("project"):
            stats["skipped_project_mismatch"] += 1
            continue
        if require_same_commit and vuln.get("commit_id") != fixed.get("commit_id"):
            stats["skipped_commit_mismatch"] += 1
            continue
        if require_same_cve and str(vuln.get("cve")) != str(fixed.get("cve")):
            stats["skipped_cve_mismatch"] += 1
            continue
        pairs.append(
            {
                "pair_id": pair_id,
                "vuln_idx": str(vuln.get("idx")),
                "fixed_idx": str(fixed.get("idx")),
                "project": vuln.get("project") or fixed.get("project"),
                "cwe": vuln.get("cwe") or fixed.get("cwe"),
                "commit_id_match": vuln.get("commit_id") == fixed.get("commit_id"),
                "cve_match": str(vuln.get("cve")) == str(fixed.get("cve")),
                "file_name_match": vuln.get("file_name") == fixed.get("file_name"),
            }
        )
    return pairs, stats


def outcome(vuln_pred: int, fixed_pred: int) -> str:
    if vuln_pred == 1 and fixed_pred == 0:
        return "P-C"
    if vuln_pred == 1 and fixed_pred == 1:
        return "P-V"
    if vuln_pred == 0 and fixed_pred == 0:
        return "P-B"
    if vuln_pred == 0 and fixed_pred == 1:
        return "P-R"
    raise ValueError(f"Unexpected predictions: {vuln_pred}, {fixed_pred}")


def pct(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PrimeVul-style pair-wise metrics.")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--pred-columns",
        nargs="*",
        help="Prediction columns to evaluate. Defaults to all columns ending with _pred.",
    )
    parser.add_argument("--require-same-project", action="store_true")
    parser.add_argument("--require-same-commit", action="store_true")
    parser.add_argument("--require-same-cve", action="store_true")
    args = parser.parse_args()

    data_rows = load_jsonl(Path(args.data_jsonl))
    pred_rows = load_csv_by_idx(Path(args.predictions_csv))
    pairs, pair_build_stats = adjacent_pairs(
        data_rows,
        require_same_project=args.require_same_project,
        require_same_commit=args.require_same_commit,
        require_same_cve=args.require_same_cve,
    )
    if not pred_rows:
        raise SystemExit("No prediction rows loaded.")

    first_pred_row = next(iter(pred_rows.values()))
    pred_columns = args.pred_columns or [
        col for col in first_pred_row if col.endswith("_pred") and col not in {"target"}
    ]

    method_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for method in pred_columns:
        counts = {"P-C": 0, "P-V": 0, "P-B": 0, "P-R": 0}
        known = 0
        unknown = 0
        for pair in pairs:
            vuln_row = pred_rows.get(pair["vuln_idx"])
            fixed_row = pred_rows.get(pair["fixed_idx"])
            vuln_pred = safe_int(vuln_row.get(method)) if vuln_row else None
            fixed_pred = safe_int(fixed_row.get(method)) if fixed_row else None
            if vuln_pred not in (0, 1) or fixed_pred not in (0, 1):
                unknown += 1
                continue
            known += 1
            label = outcome(vuln_pred, fixed_pred)
            counts[label] += 1
            pair_rows.append(
                {
                    **pair,
                    "method": method.removesuffix("_pred"),
                    "vuln_pred": vuln_pred,
                    "fixed_pred": fixed_pred,
                    "pair_outcome": label,
                }
            )

        method_rows.append(
            {
                "method": method.removesuffix("_pred"),
                "known_pairs": known,
                "unknown_pairs": unknown,
                "P-C": counts["P-C"],
                "P-V": counts["P-V"],
                "P-B": counts["P-B"],
                "P-R": counts["P-R"],
                "P-C_rate": round(pct(counts["P-C"], known), 6),
                "P-V_rate": round(pct(counts["P-V"], known), 6),
                "P-B_rate": round(pct(counts["P-B"], known), 6),
                "P-R_rate": round(pct(counts["P-R"], known), 6),
            }
        )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0].keys()))
        writer.writeheader()
        writer.writerows(method_rows)

    detail_csv = out_csv.with_name(out_csv.stem + "_pairs.csv")
    if pair_rows:
        with detail_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)

    summary = {
        "data_jsonl": args.data_jsonl,
        "predictions_csv": args.predictions_csv,
        "num_data_rows": len(data_rows),
        "num_adjacent_pairs": len(pairs),
        "pair_build_stats": pair_build_stats,
        "pair_filters": {
            "require_same_project": args.require_same_project,
            "require_same_commit": args.require_same_commit,
            "require_same_cve": args.require_same_cve,
        },
        "num_prediction_rows": len(pred_rows),
        "pred_columns": pred_columns,
        "method_metrics": method_rows,
        "pair_detail_csv": str(detail_csv),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
