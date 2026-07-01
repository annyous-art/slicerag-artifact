#!/usr/bin/env python3
"""Evaluate YES/NO vulnerability classification outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


YES_RE = re.compile(r"\bYES\b", re.IGNORECASE)
NO_RE = re.compile(r"\bNO\b", re.IGNORECASE)
OPTION_ONE_RE = re.compile(
    r"^\s*(?:answer\s*[:：]?\s*|option\s*)?[\(\uff08]?\s*1\s*[\)\uff09]?\s*(?:\s|[:：.)\uff09、，,;\-]|$)",
    re.IGNORECASE,
)
OPTION_TWO_RE = re.compile(
    r"^\s*(?:answer\s*[:：]?\s*|option\s*)?[\(\uff08]?\s*2\s*[\)\uff09]?\s*(?:\s|[:：.)\uff09、，,;\-]|$)",
    re.IGNORECASE,
)


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_prediction(response: Any):
    if response is None:
        return None
    text = str(response).strip()
    if OPTION_ONE_RE.search(text):
        return 1
    if OPTION_TWO_RE.search(text):
        return 0
    first_line = text.splitlines()[0].strip() if text else ""
    if OPTION_ONE_RE.search(first_line):
        return 1
    if OPTION_TWO_RE.search(first_line):
        return 0
    yes = YES_RE.search(text)
    no = NO_RE.search(text)
    if yes and not no:
        return 1
    if no and not yes:
        return 0
    if yes and no:
        return 1 if yes.start() < no.start() else 0
    return None


def get_target(record: dict):
    if record.get("query_target") is not None:
        return safe_int(record.get("query_target"))
    return safe_int(record.get("target"))


def get_idx(record: dict):
    return record.get("query_idx", record.get("idx"))


def compute_metrics(records: list[dict]) -> dict:
    tp = fp = tn = fn = unknown = 0
    for record in records:
        target = get_target(record)
        pred = parse_prediction(record.get("response"))
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

    total_known = tp + fp + tn + fn
    accuracy = (tp + tn) / total_known if total_known else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "total_records": len(records),
        "known_predictions": total_known,
        "unknown_predictions": unknown,
        "coverage": total_known / len(records) if records else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "yes_rate": (tp + fp) / total_known if total_known else 0.0,
        "directional_failure_index": recall - specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def add_predictions(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        target = get_target(record)
        pred = parse_prediction(record.get("response"))
        rows.append(
            {
                "idx": get_idx(record),
                "sample_key": record.get("sample_key", ""),
                "target": target,
                "prediction": pred if pred is not None else "",
                "correct": int(pred == target) if pred in (0, 1) and target in (0, 1) else "",
                "prompt_mode": record.get("prompt_mode", ""),
                "func_char_len": record.get("func_char_len", ""),
                "prompt_tokens": record.get("prompt_tokens", ""),
                "completion_tokens": record.get("completion_tokens", ""),
                "total_tokens": record.get("total_tokens", ""),
                "response": record.get("response", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prompt JSONL outputs.")
    parser.add_argument("--input", required=True, help="Prompt output JSONL")
    parser.add_argument("--summary-json", required=True, help="Metrics output JSON")
    parser.add_argument("--predictions-csv", default="", help="Optional per-sample prediction CSV")
    args = parser.parse_args()

    records = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    metrics = compute_metrics(records)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.predictions_csv:
        write_csv(Path(args.predictions_csv), add_predictions(records))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
