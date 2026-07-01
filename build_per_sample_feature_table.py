#!/usr/bin/env python3
"""Build a per-sample feature table for SliceRAG routing/error analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_retrieval_quality import summarize_record
from evaluate_prompt_outputs import parse_prediction


DEFAULT_PREDICTIONS = {
    "zero_shot": "outdir/baseline_fewshot_sensitivity_glm51_test870/glm-5.1_std_cls_logprobsFalse_fewshotegFalse_none.jsonl",
    "author_no_yes": "outdir/baseline_fewshot_sensitivity_glm51_test870/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_author_no_yes.jsonl",
    "yes_only": "outdir/baseline_fewshot_sensitivity_glm51_test870/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_yes_only.jsonl",
    "no_only": "outdir/baseline_fewshot_sensitivity_glm51_test870/glm-5.1_std_cls_logprobsFalse_fewshotegTrue_no_only.jsonl",
    "icl": "outdir/glm-5.1_std_cls_fewshotegTrue_ICL_870.jsonl",
    "old_rag": "outdir/glm-5.1_std_cls_fewshotegTrue_old_RAG_combine_prompt.jsonl",
    "positive_weighted_rag": "outdir/prompt_positive_weighted_rag_combine_glm51_test870/glm-5.1_std_cls_inline_function_label_rag_fewshotegTrue_cls0.jsonl",
    "balanced_grouped_no_label_bc": "outdir/prompt_baseline_compatible_glm51_test870/glm-5.1_std_cls_grouped_no_label_baseline_compatible_fewshotegTrue_cls0.jsonl",
}


MEMORY_APIS = {
    "malloc", "calloc", "realloc", "free", "alloca", "memcpy", "memmove", "memset",
    "memcmp", "kmalloc", "kcalloc", "kzalloc", "vmalloc", "kfree", "copy_from_user",
    "copy_to_user", "new", "delete",
}
STRING_APIS = {
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp", "strlen",
    "strnlen", "strchr", "strrchr", "strstr", "sprintf", "snprintf", "vsnprintf",
    "sscanf", "strtok", "strdup", "strndup",
}
BOUNDS_WORDS = {
    "len", "length", "size", "sizeof", "offset", "index", "idx", "bound", "bounds",
    "limit", "capacity", "count", "num", "nbytes", "columns", "rows",
}
ERROR_WORDS = {
    "error", "err", "fail", "failed", "goto", "cleanup", "exception", "errno",
    "invalid", "return", "null", "NULL", "nullptr",
}
LIFECYCLE_WORDS = {
    "free", "delete", "release", "destroy", "cleanup", "refcount", "reference",
    "retain", "put", "get", "close", "unlock",
}
CALL_EXCLUDE = {
    "if", "for", "while", "switch", "return", "sizeof", "case", "do",
}


IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|->|[{}()[\].,;:+*/%&|^!~=<>?-]")


def safe_int(value: Any, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_cwe(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def make_sample_key(record: dict) -> str:
    idx = record.get("query_idx", record.get("idx"))
    if idx is not None:
        return f"idx:{idx}"
    sample_key = record.get("sample_key")
    if sample_key:
        return str(sample_key)
    project = record.get("query_project", record.get("project"))
    commit_id = record.get("query_commit_id", record.get("commit_id"))
    if project and commit_id:
        return f"{project}_{commit_id}"
    func_id = record.get("query_func_id")
    if func_id:
        return str(func_id)
    return f"idx:{idx}"


def code_features(func: str) -> dict[str, Any]:
    func = func or ""
    lines = func.splitlines()
    identifiers = IDENT_RE.findall(func)
    identifier_counts = Counter(identifiers)
    calls = [name for name in CALL_RE.findall(func) if name not in CALL_EXCLUDE]
    lower_tokens = [token.lower() for token in identifiers]
    lower_counts = Counter(lower_tokens)

    def count_words(words: set[str]) -> int:
        return sum(lower_counts.get(word.lower(), 0) for word in words)

    return {
        "char_len": len(func),
        "token_len_approx": len(CODE_TOKEN_RE.findall(func)),
        "line_count": len(lines),
        "macro_count": sum(1 for line in lines if line.lstrip().startswith("#define")),
        "preproc_count": sum(1 for line in lines if line.lstrip().startswith("#")),
        "call_count": len(calls),
        "unique_call_count": len(set(calls)),
        "loop_count": len(re.findall(r"\b(for|while|do)\b", func)),
        "cond_count": len(re.findall(r"\b(if|else|switch|case)\b", func)) + func.count("?"),
        "ptr_count": func.count("->") + func.count("*") + func.count("&"),
        "memory_api_count": sum(identifier_counts.get(api, 0) for api in MEMORY_APIS),
        "string_api_count": sum(identifier_counts.get(api, 0) for api in STRING_APIS),
        "bounds_keyword_count": count_words(BOUNDS_WORDS),
        "error_handling_keyword_count": count_words(ERROR_WORDS),
        "lifecycle_keyword_count": count_words(LIFECYCLE_WORDS),
        "null_keyword_count": lower_counts.get("null", 0) + lower_counts.get("nullptr", 0),
    }


def load_samples(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = make_sample_key(record)
            duplicate_counts[key] += 1
            if duplicate_counts[key] > 1:
                key = f"{key}#dup{duplicate_counts[key]}"
            func = record.get("func", "")
            row = {
                "sample_key": key,
                "idx": record.get("idx", ""),
                "project": record.get("project", ""),
                "commit_id": record.get("commit_id", ""),
                "target": safe_int(record.get("target")),
                "cwe": normalize_cwe(record.get("cwe")),
                "cve": normalize_cwe(record.get("cve")),
                "func_hash": record.get("func_hash", ""),
                "file_name": record.get("file_name", ""),
                "label_changed_line_count": sum(1 for label in record.get("labels", []) if safe_int(label, 0) == 1),
            }
            row.update(code_features(func))
            rows[key] = row
    return rows


def message_chars(record: dict) -> int:
    total = 0
    for message in record.get("messages") or []:
        if isinstance(message, dict):
            total += len(str(message.get("content", "")))
    return total


def load_prediction_file(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = make_sample_key(record)
            duplicate_counts[key] += 1
            if duplicate_counts[key] > 1:
                key = f"{key}#dup{duplicate_counts[key]}"
            pred = parse_prediction(record.get("response"))
            target = safe_int(record.get("query_target", record.get("target")))
            rows[key] = {
                "pred": pred,
                "correct": int(pred == target) if pred in (0, 1) and target in (0, 1) else "",
                "response": str(record.get("response", ""))[:200],
                "prompt_mode": record.get("prompt_mode", ""),
                "prompt_tokens": record.get("prompt_tokens", ""),
                "completion_tokens": record.get("completion_tokens", ""),
                "total_tokens": record.get("total_tokens", ""),
                "message_chars": message_chars(record),
            }
    return rows


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--prediction must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("prediction NAME cannot be empty")
    return name, Path(path)


def add_prediction_columns(rows: dict[str, dict[str, Any]], name: str, path: Path) -> None:
    if not path.exists():
        print(f"Warning: prediction file missing for {name}: {path}")
        return
    predictions = load_prediction_file(path)
    for key, row in rows.items():
        item = predictions.get(key)
        if item is None:
            row[f"{name}_pred"] = ""
            row[f"{name}_correct"] = ""
            row[f"{name}_prompt_tokens"] = ""
            row[f"{name}_total_tokens"] = ""
            row[f"{name}_message_chars"] = ""
            continue
        row[f"{name}_pred"] = item["pred"] if item["pred"] is not None else ""
        row[f"{name}_correct"] = item["correct"]
        row[f"{name}_prompt_mode"] = item["prompt_mode"]
        row[f"{name}_prompt_tokens"] = item["prompt_tokens"]
        row[f"{name}_completion_tokens"] = item["completion_tokens"]
        row[f"{name}_total_tokens"] = item["total_tokens"]
        row[f"{name}_message_chars"] = item["message_chars"]


def add_retrieval_columns(rows: dict[str, dict[str, Any]], query_results_path: Path) -> None:
    if not query_results_path.exists():
        print(f"Warning: query results missing: {query_results_path}")
        return
    with query_results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = make_sample_key(record)
            row = rows.get(key)
            if row is None:
                continue
            retrieval = summarize_record(record)
            for col, value in retrieval.items():
                if col in {"target", "project", "cwe"}:
                    continue
                row[f"retrieval_{col}"] = value


def add_disagreement_columns(rows: dict[str, dict[str, Any]], method_names: list[str]) -> None:
    for row in rows.values():
        preds = [row.get(f"{name}_pred") for name in method_names]
        known = [safe_int(pred) for pred in preds if safe_int(pred) in (0, 1)]
        row["method_known_pred_count"] = len(known)
        row["method_yes_count"] = sum(1 for pred in known if pred == 1)
        row["method_no_count"] = sum(1 for pred in known if pred == 0)
        row["method_yes_rate"] = (row["method_yes_count"] / len(known)) if known else ""
        row["method_disagree"] = int(len(set(known)) > 1) if known else ""
        if known:
            p_yes = row["method_yes_count"] / len(known)
            p_no = 1.0 - p_yes
            entropy = 0.0
            for p in (p_yes, p_no):
                if p > 0:
                    entropy -= p * math.log2(p)
            row["method_pred_entropy"] = entropy
        else:
            row["method_pred_entropy"] = ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]], method_names: list[str]) -> None:
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "target_counts": dict(Counter(str(row.get("target")) for row in rows)),
    }
    for name in method_names:
        known = [row for row in rows if safe_int(row.get(f"{name}_pred")) in (0, 1)]
        correct = [row for row in known if safe_int(row.get(f"{name}_correct")) == 1]
        yes_count = sum(1 for row in known if safe_int(row.get(f"{name}_pred")) == 1)
        prompt_tokens = [safe_float(row.get(f"{name}_prompt_tokens"), None) for row in known]
        prompt_tokens = [value for value in prompt_tokens if value is not None]
        message_chars = [safe_float(row.get(f"{name}_message_chars"), None) for row in known]
        message_chars = [value for value in message_chars if value is not None]
        summary[name] = {
            "known": len(known),
            "accuracy": len(correct) / len(known) if known else 0.0,
            "yes_rate": yes_count / len(known) if known else 0.0,
            "avg_prompt_tokens": mean(prompt_tokens) if prompt_tokens else 0.0,
            "avg_message_chars": mean(message_chars) if message_chars else 0.0,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-sample SliceRAG feature table.")
    parser.add_argument("--data", default="data/primevul_test_paired_labeled.jsonl")
    parser.add_argument("--query-results", default="query_balanced_chunk6_stride2_test870/query_results.jsonl")
    parser.add_argument("--output-csv", default="outdir/analysis/per_sample_feature_table.csv")
    parser.add_argument("--summary-json", default="outdir/analysis/per_sample_feature_table_summary.json")
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_arg,
        default=[],
        help="Additional or overriding prediction file, format NAME=PATH.",
    )
    parser.add_argument("--no-default-predictions", action="store_true")
    args = parser.parse_args()

    rows_by_key = load_samples(Path(args.data))
    predictions: dict[str, Path] = {}
    if not args.no_default_predictions:
        predictions.update({name: Path(path) for name, path in DEFAULT_PREDICTIONS.items()})
    predictions.update(dict(args.prediction))

    for name, path in predictions.items():
        add_prediction_columns(rows_by_key, name, path)
    add_retrieval_columns(rows_by_key, Path(args.query_results))
    add_disagreement_columns(rows_by_key, list(predictions.keys()))

    rows = list(rows_by_key.values())
    write_csv(Path(args.output_csv), rows)
    write_summary(Path(args.summary_json), rows, list(predictions.keys()))
    print(f"Wrote feature table: {args.output_csv}")
    print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()
