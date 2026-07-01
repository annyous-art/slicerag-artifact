#!/usr/bin/env python3
"""Analyze ensemble and error patterns across SliceRAG prediction files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from evaluate_prompt_outputs import parse_prediction, safe_int


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--prediction must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("prediction NAME cannot be empty")
    return name, Path(path)


def load_csv_predictions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idx = str(row.get("idx", "")).strip()
            if not idx:
                continue
            target = safe_int(row.get("target"))
            pred = safe_int(row.get("prediction"))
            rows[idx] = {
                "idx": idx,
                "target": target,
                "prediction": pred if pred in (0, 1) else None,
                "response": row.get("response", ""),
                "prompt_tokens": safe_int(row.get("prompt_tokens"), 0),
                "total_tokens": safe_int(row.get("total_tokens"), 0),
            }
    return rows


def load_jsonl_predictions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = row.get("query_idx", row.get("idx"))
            if idx is None:
                continue
            idx = str(idx)
            target = row.get("query_target", row.get("target"))
            pred = parse_prediction(row.get("response"))
            rows[idx] = {
                "idx": idx,
                "target": safe_int(target),
                "prediction": pred if pred in (0, 1) else None,
                "response": row.get("response", ""),
                "prompt_tokens": safe_int(row.get("prompt_tokens"), 0),
                "total_tokens": safe_int(row.get("total_tokens"), 0),
            }
    return rows


def load_predictions(path: Path) -> dict[str, dict]:
    if path.suffix.lower() == ".csv":
        return load_csv_predictions(path)
    if path.suffix.lower() == ".jsonl":
        return load_jsonl_predictions(path)
    raise ValueError(f"Unsupported prediction file type: {path}")


def load_features(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    features = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idx = str(row.get("idx", "")).strip()
            if idx:
                features[idx] = row
    return features


def metrics_from_pairs(pairs: list[tuple[int, int | None]]) -> dict:
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
    accuracy = (tp + tn) / known if known else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    yes_rate = (tp + fp) / known if known else 0.0
    return {
        "known": known,
        "unknown": unknown,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": yes_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def round_metrics(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "name",
        "strategy",
        "base",
        "method",
        "known",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "yes_rate",
        "tp",
        "fp",
        "tn",
        "fn",
        "both_correct",
        "base_only",
        "method_only",
        "both_wrong",
        "oracle_accuracy",
    ]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_common_indices(predictions: dict[str, dict[str, dict]]) -> list[str]:
    common = None
    for rows in predictions.values():
        keys = set(rows)
        common = keys if common is None else common & keys
    return sorted(common or [])


def target_for_idx(idx: str, predictions: dict[str, dict[str, dict]]) -> int | None:
    targets = [rows[idx].get("target") for rows in predictions.values() if idx in rows]
    targets = [target for target in targets if target in (0, 1)]
    if not targets:
        return None
    count = Counter(targets)
    return count.most_common(1)[0][0]


def ensemble_pred(values: list[int | None], strategy: str) -> int | None:
    values = [value for value in values if value in (0, 1)]
    if not values:
        return None
    if strategy == "or":
        return 1 if any(value == 1 for value in values) else 0
    if strategy == "and":
        return 1 if all(value == 1 for value in values) else 0
    if strategy == "majority":
        return 1 if sum(values) >= ((len(values) / 2) + 0.000001) else 0
    if strategy == "majority_tie_yes":
        return 1 if sum(values) >= (len(values) / 2) else 0
    raise ValueError(f"Unknown strategy: {strategy}")


def pair_complementarity(base_name: str, method_name: str, indices: list[str], predictions: dict[str, dict[str, dict]]) -> dict:
    both_correct = base_only = method_only = both_wrong = 0
    for idx in indices:
        target = target_for_idx(idx, predictions)
        base_pred = predictions[base_name][idx]["prediction"]
        method_pred = predictions[method_name][idx]["prediction"]
        base_correct = base_pred == target
        method_correct = method_pred == target
        if base_correct and method_correct:
            both_correct += 1
        elif base_correct and not method_correct:
            base_only += 1
        elif method_correct and not base_correct:
            method_only += 1
        else:
            both_wrong += 1
    total = both_correct + base_only + method_only + both_wrong
    return {
        "base": base_name,
        "method": method_name,
        "both_correct": both_correct,
        "base_only": base_only,
        "method_only": method_only,
        "both_wrong": both_wrong,
        "oracle_accuracy": (both_correct + base_only + method_only) / total if total else 0.0,
    }


def make_per_sample_rows(indices: list[str], predictions: dict[str, dict[str, dict]], features: dict[str, dict], best_strategy: dict | None) -> list[dict]:
    rows = []
    method_names = list(predictions)
    for idx in indices:
        target = target_for_idx(idx, predictions)
        row = {
            "idx": idx,
            "target": target,
        }
        for name in method_names:
            pred = predictions[name][idx]["prediction"]
            row[f"{name}_pred"] = pred
            row[f"{name}_correct"] = int(pred == target) if pred in (0, 1) and target in (0, 1) else ""
        if best_strategy:
            values = [predictions[name][idx]["prediction"] for name in best_strategy["methods"]]
            pred = ensemble_pred(values, best_strategy["strategy"])
            row["best_ensemble_pred"] = pred
            row["best_ensemble_correct"] = int(pred == target) if pred in (0, 1) and target in (0, 1) else ""
        for key in [
            "project",
            "cwe",
            "char_len",
            "token_len_approx",
            "line_count",
            "memory_api_count",
            "bounds_keyword_count",
            "error_handling_keyword_count",
            "lifecycle_keyword_count",
            "null_keyword_count",
            "retrieval_top1_score",
            "retrieval_pos_neg_margin_max",
        ]:
            if idx in features and key in features[idx]:
                row[key] = features[idx][key]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze method ensembles and errors.")
    parser.add_argument("--prediction", action="append", type=parse_prediction_arg, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features-csv", default="")
    parser.add_argument("--base-method", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {name: load_predictions(path) for name, path in args.prediction}
    features = load_features(Path(args.features_csv)) if args.features_csv else {}
    indices = get_common_indices(predictions)
    if not indices:
        raise RuntimeError("No common idx values across prediction files.")

    method_rows = []
    for name, rows in predictions.items():
        pairs = [(target_for_idx(idx, predictions), rows[idx]["prediction"]) for idx in indices]
        method_rows.append(round_metrics({"name": name, **metrics_from_pairs(pairs)}))

    pair_rows = []
    names = list(predictions)
    base_names = [args.base_method] if args.base_method else names
    for base_name in base_names:
        if base_name not in predictions:
            continue
        for method_name in names:
            if base_name == method_name:
                continue
            pair_rows.append(round_metrics(pair_complementarity(base_name, method_name, indices, predictions)))

    ensemble_rows = []
    strategies = ["or", "and", "majority", "majority_tie_yes"]
    for size in range(2, len(names) + 1):
        for method_tuple in combinations(names, size):
            for strategy in strategies:
                pairs = []
                for idx in indices:
                    target = target_for_idx(idx, predictions)
                    values = [predictions[name][idx]["prediction"] for name in method_tuple]
                    pairs.append((target, ensemble_pred(values, strategy)))
                ensemble_rows.append(
                    round_metrics(
                        {
                            "name": "+".join(method_tuple),
                            "strategy": strategy,
                            "methods": "|".join(method_tuple),
                            **metrics_from_pairs(pairs),
                        }
                    )
                )

    best_strategy = None
    if ensemble_rows:
        best = max(ensemble_rows, key=lambda row: (row.get("f1", 0), row.get("accuracy", 0), row.get("precision", 0)))
        best_strategy = {
            "strategy": best["strategy"],
            "methods": str(best["methods"]).split("|"),
        }

    per_sample_rows = make_per_sample_rows(indices, predictions, features, best_strategy)

    write_csv(output_dir / "method_metrics.csv", method_rows)
    write_csv(output_dir / "pair_complementarity.csv", pair_rows)
    write_csv(output_dir / "ensemble_metrics.csv", sorted(ensemble_rows, key=lambda row: (-row.get("f1", 0), -row.get("accuracy", 0), row.get("name", ""))))
    write_csv(output_dir / "per_sample_predictions.csv", per_sample_rows)

    summary = {
        "num_common_indices": len(indices),
        "methods": names,
        "method_metrics": method_rows,
        "best_ensemble": max(ensemble_rows, key=lambda row: (row.get("f1", 0), row.get("accuracy", 0))) if ensemble_rows else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
