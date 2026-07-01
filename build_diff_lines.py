#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


def iter_jsonl_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_adjacent_pairs(path: Path) -> Iterable[tuple[dict, dict]]:
    records = list(iter_jsonl_records(path))
    for i in range(0, len(records) - 1, 2):
        yield records[i], records[i + 1]


def split_pair_by_target(left: dict, right: dict) -> tuple[dict | None, dict | None]:
    left_target = int(left.get("target", -1))
    right_target = int(right.get("target", -1))
    if left_target == 1 and right_target == 0:
        return left, right
    if left_target == 0 and right_target == 1:
        return right, left
    return None, None


def changed_line_maps(vulnerable_func: str, fixed_func: str) -> tuple[dict[str, str], dict[str, str]]:
    vulnerable_lines = vulnerable_func.splitlines()
    fixed_lines = fixed_func.splitlines()
    matcher = SequenceMatcher(None, vulnerable_lines, fixed_lines)
    deleted_or_changed: dict[str, str] = {}
    added_or_changed: dict[str, str] = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for idx in range(i1, i2):
            deleted_or_changed[str(idx + 1)] = vulnerable_lines[idx]
        for idx in range(j1, j2):
            added_or_changed[str(idx + 1)] = fixed_lines[idx]

    return deleted_or_changed, added_or_changed


def build_diff_report(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for left, right in iter_adjacent_pairs(input_path):
            vulnerable_record, fixed_record = split_pair_by_target(left, right)
            if vulnerable_record is None or fixed_record is None:
                continue

            deleted_lines, added_lines = changed_line_maps(
                vulnerable_record.get("func", "") or "",
                fixed_record.get("func", "") or "",
            )

            payload = {
                "idx_vul": int(vulnerable_record["idx"]),
                "idx_novul": int(fixed_record["idx"]),
                "deleted_lines": deleted_lines,
                "added_lines": added_lines,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract diff line numbers from adjacent paired PrimeVul JSONL records.")
    parser.add_argument("--input", type=Path, required=True, help="PrimeVul paired JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output diff-line JSONL file.")
    args = parser.parse_args()
    build_diff_report(args.input, args.output)


if __name__ == "__main__":
    main()
