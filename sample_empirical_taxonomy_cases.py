#!/usr/bin/env python3
"""Sample representative cases from empirical error-taxonomy results."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl_by_idx(path: Path) -> dict[str, dict]:
    rows = {}
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
    preferred = [
        "idx",
        "target",
        "base_pred",
        "error_type",
        "taxonomy",
        "dominant_pattern",
        "project",
        "cwe",
        "char_len",
        "yes_vote_count",
        "known_vote_count",
        "func_excerpt",
    ]
    fields = sorted({key for row in rows for key in row})
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def excerpt(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n/* ... truncated ... */"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample representative taxonomy cases for manual validation.")
    parser.add_argument("--taxonomy-csv", default="outdir/empirical_study/error_taxonomy/per_sample_error_taxonomy.csv")
    parser.add_argument("--data-jsonl", default="data/primevul_test_paired_labeled.jsonl")
    parser.add_argument("--output-dir", default="outdir/empirical_study/error_taxonomy/manual_samples")
    parser.add_argument("--per-taxonomy", type=int, default=10)
    parser.add_argument("--max-func-chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    random.seed(args.seed)
    taxonomy_rows = read_csv(Path(args.taxonomy_csv))
    data_rows = load_jsonl_by_idx(Path(args.data_jsonl))

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in taxonomy_rows:
        if row.get("error_type") == "correct":
            continue
        groups[row.get("taxonomy", "unknown")].append(row)

    all_samples = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for taxonomy, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        rows = list(rows)
        random.shuffle(rows)
        selected = rows[: args.per_taxonomy]
        enriched = []
        for row in selected:
            idx = safe_str(row.get("idx"))
            data = data_rows.get(idx, {})
            out = dict(row)
            out["func"] = safe_str(data.get("func"))
            out["func_excerpt"] = excerpt(out["func"], args.max_func_chars)
            out["commit_message"] = data.get("commit_message", "")
            out["cve_desc"] = data.get("cve_desc", "")
            enriched.append(out)
            all_samples.append(out)

        safe_name = taxonomy.replace(":", "__").replace("/", "_")
        write_csv(output_dir / f"{safe_name}.csv", enriched)
        write_jsonl(output_dir / f"{safe_name}.jsonl", enriched)

    write_csv(output_dir / "all_manual_samples.csv", all_samples)
    write_jsonl(output_dir / "all_manual_samples.jsonl", all_samples)
    summary = {
        "taxonomy_csv": args.taxonomy_csv,
        "data_jsonl": args.data_jsonl,
        "per_taxonomy": args.per_taxonomy,
        "num_taxonomies": len(groups),
        "num_samples": len(all_samples),
        "taxonomy_counts": {taxonomy: len(rows) for taxonomy, rows in sorted(groups.items())},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
