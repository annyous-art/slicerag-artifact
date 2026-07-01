#!/usr/bin/env python3
"""Sample disagreement cases for a second-stage vulnerability verifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GROUPS = [
    "icl_method_only=outdir/complementarity_zero_shot/zero_shot_vs_icl/method_only.csv",
    "icl_baseline_only=outdir/complementarity_zero_shot/zero_shot_vs_icl/baseline_only.csv",
    "positive_weighted_rag_method_only=outdir/complementarity_zero_shot/zero_shot_vs_positive_weighted_rag/method_only.csv",
    "positive_weighted_rag_baseline_only=outdir/complementarity_zero_shot/zero_shot_vs_positive_weighted_rag/baseline_only.csv",
]

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CALL_EXCLUDE = {"if", "for", "while", "switch", "return", "sizeof", "case", "do"}


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_cwe(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def parse_group(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("group must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("group name cannot be empty")
    return name, Path(path)


def load_primevul(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record.get("idx"))] = record
    return records


def function_signature(func: str) -> str:
    for line in (func or "").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:240]
    return ""


def top_calls(func: str, limit: int = 20) -> list[str]:
    calls = [name for name in CALL_RE.findall(func or "") if name not in CALL_EXCLUDE]
    return [name for name, _count in Counter(calls).most_common(limit)]


def changed_line_excerpt(record: dict, limit: int = 8) -> list[dict[str, Any]]:
    lines = (record.get("func") or "").splitlines()
    labels = record.get("labels") or []
    excerpts = []
    for line_no, (line, label) in enumerate(zip(lines, labels), start=1):
        if safe_int(label, 0) == 1 and line.strip():
            excerpts.append({"line_no": line_no, "code": line.strip()[:240]})
            if len(excerpts) >= limit:
                break
    return excerpts


def load_group_rows(group_name: str, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"Warning: missing group file {group_name}: {path}")
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["verifier_group"] = group_name
            row["source_csv"] = str(path)
            rows.append(row)
    return rows


def enrich_row(row: dict, primevul_by_idx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    idx = str(row.get("idx", ""))
    source = primevul_by_idx.get(idx, {})
    func = source.get("func", "")
    enriched = {
        "sample_key": row.get("sample_key") or f"idx:{idx}",
        "idx": idx,
        "verifier_group": row.get("verifier_group", ""),
        "source_csv": row.get("source_csv", ""),
        "project": source.get("project", row.get("project", "")),
        "commit_id": source.get("commit_id", ""),
        "target": safe_int(source.get("target", row.get("target"))),
        "cwe": normalize_cwe(source.get("cwe", row.get("cwe", ""))),
        "cve": normalize_cwe(source.get("cve", "")),
        "file_name": source.get("file_name", ""),
        "func_hash": source.get("func_hash", ""),
        "func": func,
        "func_signature": function_signature(func),
        "func_char_len": len(func),
        "func_line_count": len(func.splitlines()),
        "top_calls": top_calls(func),
        "changed_line_excerpt_for_analysis_only": changed_line_excerpt(source),
    }
    for key, value in row.items():
        if key not in enriched:
            enriched[key] = value
    return enriched


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_key",
        "idx",
        "verifier_group",
        "target",
        "project",
        "cwe",
        "func_char_len",
        "func_line_count",
        "func_signature",
        "zero_shot_pred",
        "icl_pred",
        "old_rag_pred",
        "positive_weighted_rag_pred",
        "balanced_grouped_no_label_bc_pred",
        "retrieval_top1_evidence_role",
        "retrieval_pos_neg_margin_max",
        "method_pred_entropy",
        "top_calls",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["top_calls"] = "|".join(row.get("top_calls") or [])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample verifier disagreement cases.")
    parser.add_argument("--data", default="data/primevul_test_paired_labeled.jsonl")
    parser.add_argument("--group", action="append", type=parse_group, default=[])
    parser.add_argument("--samples-per-group", type=int, default=50)
    parser.add_argument("--all", action="store_true", help="Use every row from each group instead of sampling.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate by idx after sampling.")
    parser.add_argument("--no-shuffle-output", action="store_true", help="Keep sampled rows grouped by source file.")
    parser.add_argument("--output-jsonl", default="outdir/verifier_sets/verifier_disagreement_sample.jsonl")
    parser.add_argument("--output-csv", default="outdir/verifier_sets/verifier_disagreement_sample.csv")
    parser.add_argument("--summary-json", default="outdir/verifier_sets/verifier_disagreement_sample_summary.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    groups = args.group or [parse_group(item) for item in DEFAULT_GROUPS]
    primevul_by_idx = load_primevul(Path(args.data))

    sampled_rows = []
    summary = {"seed": args.seed, "samples_per_group": args.samples_per_group, "all": args.all, "groups": {}}
    for group_name, path in groups:
        rows = load_group_rows(group_name, path)
        if args.all or args.samples_per_group <= 0 or args.samples_per_group >= len(rows):
            chosen = list(rows)
        else:
            chosen = rng.sample(rows, args.samples_per_group)
        enriched = [enrich_row(row, primevul_by_idx) for row in chosen]
        sampled_rows.extend(enriched)
        summary["groups"][group_name] = {
            "source_csv": str(path),
            "available": len(rows),
            "selected": len(enriched),
            "target_counts": dict(Counter(str(row.get("target")) for row in enriched)),
        }

    if args.dedupe:
        deduped = {}
        group_sources = defaultdict(list)
        for row in sampled_rows:
            idx = str(row.get("idx"))
            group_sources[idx].append(row.get("verifier_group", ""))
            deduped.setdefault(idx, row)
        sampled_rows = list(deduped.values())
        for row in sampled_rows:
            row["verifier_group_sources"] = "|".join(sorted(set(group_sources[str(row.get("idx"))])))

    if not args.no_shuffle_output:
        rng.shuffle(sampled_rows)

    summary["total_selected"] = len(sampled_rows)
    summary["target_counts"] = dict(Counter(str(row.get("target")) for row in sampled_rows))
    summary["group_counts"] = dict(Counter(row.get("verifier_group", "") for row in sampled_rows))

    write_jsonl(Path(args.output_jsonl), sampled_rows)
    write_csv(Path(args.output_csv), sampled_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote JSONL: {args.output_jsonl}")
    print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
