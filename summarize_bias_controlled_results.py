#!/usr/bin/env python3
"""Summarize bias-controlled prompt metrics into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "variant",
        "model",
        "total_records",
        "known",
        "unknown",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "yes_rate",
        "tp",
        "fp",
        "tn",
        "fn",
        "input",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize bias-controlled metrics.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.input_dir).glob("*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = dict(data)
        row["path"] = str(path)
        rows.append(row)

    rows.sort(key=lambda row: (row.get("variant", ""), row.get("model", "")))
    write_csv(Path(args.output_csv), rows)
    summary = {
        "input_dir": args.input_dir,
        "num_metrics": len(rows),
        "rows": rows,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
