#!/usr/bin/env python3
"""Apply repair-presence verifier outputs as a targeted YES->NO override.

The verifier is intentionally not a standalone vulnerability classifier. It is
used only for first-stage YES predictions: when the verifier says the concrete
patch repair is already present, this script flips that sample to NO and
recomputes full-test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONFIDENCE_RANK = {
    "": 0,
    "unknown": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


def split_type_list(values: list[str] | None) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        for part in str(value).split("|"):
            part = part.strip()
            if part:
                out.add(part)
    return out


def repair_type_parts(value: Any) -> set[str]:
    return split_type_list([safe_str(value)])


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_verifier_outputs(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = row.get("idx", row.get("query_idx"))
            if idx is not None:
                rows[str(idx)] = row
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(pairs: list[tuple[int | None, int | None]]) -> dict:
    tp = fp = tn = fn = unknown = 0
    for target, pred in pairs:
        if target not in (0, 1) or pred not in (0, 1):
            unknown += 1
            continue
        if target == 1 and pred == 1:
            tp += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 0 and pred == 0:
            tn += 1
        elif target == 1 and pred == 0:
            fn += 1
    known = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "known": known,
        "unknown": unknown,
        "accuracy": (tp + tn) / known if known else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": (tp + fp) / known if known else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def round_floats(row: dict) -> dict:
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()}


def repair_type_policy_allows(repair_type: Any, allow_types: set[str], exclude_types: set[str]) -> bool:
    parts = repair_type_parts(repair_type)
    if allow_types and not (parts & allow_types):
        return False
    if exclude_types and (parts & exclude_types):
        return False
    return True


def verifier_skip_reason(
    verifier_row: dict,
    min_confidence: str,
    require_repair_present: bool,
    allow_types: set[str],
    exclude_types: set[str],
) -> str:
    verifier_pred = safe_int(verifier_row.get("repair_verifier_pred"))
    if verifier_pred != 0:
        return "verifier_not_no"

    confidence = safe_str(verifier_row.get("confidence") or "unknown").lower()
    if CONFIDENCE_RANK.get(confidence, 0) < CONFIDENCE_RANK[min_confidence]:
        return "low_confidence"

    if require_repair_present and verifier_row.get("repair_present") is not True:
        return "repair_not_present"

    if not repair_type_policy_allows(verifier_row.get("repair_type"), allow_types, exclude_types):
        return "repair_type_policy"

    return ""


def verifier_allows_override(
    verifier_row: dict,
    min_confidence: str,
    require_repair_present: bool,
    allow_types: set[str],
    exclude_types: set[str],
) -> bool:
    return verifier_skip_reason(verifier_row, min_confidence, require_repair_present, allow_types, exclude_types) == ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply repair-presence verifier as a YES->NO override.")
    parser.add_argument("--predictions-csv", default="outdir/ensemble_error_patch_contrast/per_sample_predictions.csv")
    parser.add_argument("--verifier-jsonl", default="outdir/repair_presence_verifier/repair_presence_outputs.jsonl")
    parser.add_argument("--base-pred-column", default="best_ensemble_pred")
    parser.add_argument("--output-csv", default="outdir/repair_presence_verifier/applied_predictions.csv")
    parser.add_argument("--summary-json", default="outdir/repair_presence_verifier/applied_summary.json")
    parser.add_argument("--min-confidence", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--require-repair-present", action="store_true")
    parser.add_argument(
        "--allow-repair-types",
        nargs="*",
        default=[],
        help=(
            "Only allow YES->NO override when repair_type contains one of these atomic types. "
            "Compound types like added_guard|error_handling match either atomic part. "
            "If omitted, all types are allowed unless excluded."
        ),
    )
    parser.add_argument(
        "--exclude-repair-types",
        nargs="*",
        default=[],
        help=(
            "Block YES->NO override when repair_type contains any of these atomic types. "
            "Useful for high-ambiguity repair families such as bounds_or_shape_check."
        ),
    )
    args = parser.parse_args()
    allow_types = split_type_list(args.allow_repair_types)
    exclude_types = split_type_list(args.exclude_repair_types)

    rows = load_csv(Path(args.predictions_csv))
    verifier_rows = load_verifier_outputs(Path(args.verifier_jsonl))

    out_rows = []
    before_pairs = []
    after_pairs = []
    override_rows = []
    skipped_rows = []
    skip_reason_counts = Counter()

    for row in rows:
        idx = str(row.get("idx", "")).strip()
        target = safe_int(row.get("target"))
        base_pred = safe_int(row.get(args.base_pred_column))
        final_pred = base_pred
        verifier_row = verifier_rows.get(idx)
        repair_override = False
        repair_skip_reason = ""

        if base_pred == 1 and verifier_row:
            repair_skip_reason = verifier_skip_reason(
                verifier_row,
                args.min_confidence,
                args.require_repair_present,
                allow_types,
                exclude_types,
            )
            if repair_skip_reason == "":
                final_pred = 0
                repair_override = True
            else:
                skip_reason_counts[repair_skip_reason] += 1

        out = dict(row)
        out["base_pred_column"] = args.base_pred_column
        out["base_pred"] = base_pred
        out["repair_verifier_available"] = int(verifier_row is not None)
        out["repair_verifier_pred"] = "" if not verifier_row else verifier_row.get("repair_verifier_pred", "")
        out["repair_present"] = "" if not verifier_row else verifier_row.get("repair_present", "")
        out["repair_type"] = "" if not verifier_row else verifier_row.get("repair_type", "")
        out["repair_confidence"] = "" if not verifier_row else verifier_row.get("confidence", "")
        out["repair_evidence"] = "" if not verifier_row else verifier_row.get("evidence", "")
        out["repair_reason"] = "" if not verifier_row else verifier_row.get("reason", "")
        out["repair_policy_skip_reason"] = repair_skip_reason
        out["repair_override"] = int(repair_override)
        out["repair_applied_pred"] = final_pred
        out["repair_applied_correct"] = int(final_pred == target) if final_pred in (0, 1) and target in (0, 1) else ""
        out_rows.append(out)

        before_pairs.append((target, base_pred))
        after_pairs.append((target, final_pred))
        if repair_override:
            override_rows.append(out)
        elif base_pred == 1 and verifier_row:
            skipped_rows.append(out)

    before = round_floats(metrics(before_pairs))
    after = round_floats(metrics(after_pairs))
    override_metrics = round_floats(metrics([(safe_int(row.get("target")), safe_int(row.get("repair_applied_pred"))) for row in override_rows]))
    summary = {
        "base_pred_column": args.base_pred_column,
        "min_confidence": args.min_confidence,
        "require_repair_present": args.require_repair_present,
        "allow_repair_types": sorted(allow_types),
        "exclude_repair_types": sorted(exclude_types),
        "before": before,
        "after": after,
        "delta": {
            key: round(after[key] - before[key], 6)
            for key in ["accuracy", "precision", "recall", "f1", "yes_rate"]
        },
        "verifier_rows": len(verifier_rows),
        "rows_with_verifier": sum(1 for row in out_rows if row["repair_verifier_available"] == 1),
        "override_count": len(override_rows),
        "skipped_yes_with_verifier_count": len(skipped_rows),
        "override_target_counts": dict(Counter(safe_str(row.get("target")) for row in override_rows).most_common()),
        "override_repair_type_counts": dict(Counter(safe_str(row.get("repair_type") or "unknown") for row in override_rows).most_common()),
        "override_confidence_counts": dict(Counter(safe_str(row.get("repair_confidence") or "unknown") for row in override_rows).most_common()),
        "skipped_policy_reason_counts": dict(skip_reason_counts.most_common()),
        "skipped_repair_type_counts": dict(Counter(safe_str(row.get("repair_type") or "unknown") for row in skipped_rows).most_common()),
        "override_metrics_on_overridden_rows": override_metrics,
    }

    write_csv(Path(args.output_csv), out_rows)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote predictions: {args.output_csv}")
    print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()
