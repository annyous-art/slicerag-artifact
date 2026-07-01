#!/usr/bin/env python3
"""Merge multiple repair verifier outputs into one JSONL for apply step."""

from __future__ import annotations

import argparse
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

DEFAULT_TYPE_PRIORITY = {
    "api_replacement": 100,
    "state_or_lifetime_repair": 90,
    "error_handling": 80,
    "null_check": 75,
    "added_guard": 70,
    "bounds_or_shape_check": 60,
    "unknown": 0,
}


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def count_value(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return safe_str(value)


def load_jsonl(path: Path, source_name: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["_verifier_source"] = source_name
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_priority(values: list[str] | None) -> dict[str, int]:
    priority = dict(DEFAULT_TYPE_PRIORITY)
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Priority must use type=number format: {item}")
        key, value = item.split("=", 1)
        priority[key.strip()] = int(value.strip())
    return priority


def repair_type_parts(value: Any) -> set[str]:
    return {part.strip() for part in safe_str(value).split("|") if part.strip()}


def type_priority(row: dict, priority: dict[str, int]) -> int:
    parts = repair_type_parts(row.get("repair_type")) or {"unknown"}
    return max(priority.get(part, 0) for part in parts)


def merge_key(row: dict, priority: dict[str, int]) -> tuple:
    pred = safe_int(row.get("repair_verifier_pred"))
    is_override = int(pred == 0 and row.get("repair_present") is True)
    confidence = safe_str(row.get("confidence") or "unknown").lower()
    return (
        is_override,
        CONFIDENCE_RANK.get(confidence, 0),
        type_priority(row, priority),
        int(bool(row.get("strict_structured_repair_present"))),
    )


def idx_of(row: dict) -> str:
    return safe_str(row.get("idx", row.get("query_idx"))).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge repair verifier outputs by idx.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--type-priority",
        nargs="*",
        default=[],
        help="Optional type priority overrides, e.g. api_replacement=120 bounds_or_shape_check=40",
    )
    args = parser.parse_args()

    priority = parse_priority(args.type_priority)
    all_rows = []
    for input_path in args.inputs:
        path = Path(input_path)
        all_rows.extend(load_jsonl(path, path.name))

    by_idx: dict[str, list[dict]] = {}
    skipped_missing_idx = 0
    for row in all_rows:
        idx = idx_of(row)
        if not idx:
            skipped_missing_idx += 1
            continue
        by_idx.setdefault(idx, []).append(row)

    merged = []
    conflict_rows = []
    for idx, rows in by_idx.items():
        selected = max(rows, key=lambda row: merge_key(row, priority))
        selected = dict(selected)
        selected["_merged_verifier_count"] = len(rows)
        selected["_merged_available_types"] = sorted(
            {part for row in rows for part in repair_type_parts(row.get("repair_type"))}
        )
        selected["_merged_sources"] = sorted({safe_str(row.get("_verifier_source")) for row in rows})
        if len({safe_int(row.get("repair_verifier_pred")) for row in rows}) > 1:
            conflict_rows.append(selected)
        merged.append(selected)

    merged.sort(key=lambda row: safe_int(idx_of(row), 10**18))
    write_jsonl(Path(args.output), merged)

    summary = {
        "input_files": args.inputs,
        "input_rows": len(all_rows),
        "unique_idx": len(by_idx),
        "output_rows": len(merged),
        "skipped_missing_idx": skipped_missing_idx,
        "conflict_idx_count": len(conflict_rows),
        "selected_repair_type_counts": dict(Counter(count_value(row.get("repair_type")) for row in merged).most_common()),
        "selected_pred_counts": dict(Counter(count_value(row.get("repair_verifier_pred")) for row in merged).most_common()),
        "selected_confidence_counts": dict(Counter(count_value(row.get("confidence")) for row in merged).most_common()),
        "source_counts": dict(Counter(count_value(row.get("_verifier_source")) for row in merged).most_common()),
        "type_priority": priority,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote merged verifier outputs: {args.output}")
    print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()
