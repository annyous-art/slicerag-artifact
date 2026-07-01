#!/usr/bin/env python3
"""Summarize multiple full-function classifier runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SAMPLE_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "yes_rate",
    "directional_failure_index",
]

PAIR_KEYS = ["P-C_rate", "P-V_rate", "P-B_rate", "P-R_rate"]


def read_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values) if values else 0.0,
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full-function classifier metrics.")
    parser.add_argument("--runs-root", required=True, help="Directory containing seed run subdirectories")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke_* directories in the aggregate")
    args = parser.parse_args()

    run_dirs = sorted(path for path in Path(args.runs_root).iterdir() if (path / "metrics.json").exists())
    if not args.include_smoke:
        run_dirs = [path for path in run_dirs if not path.name.startswith("smoke_")]
    if not run_dirs:
        raise SystemExit(f"No metrics.json files found under {args.runs_root}")

    rows: list[dict[str, Any]] = []
    metrics_by_key: dict[str, list[float]] = {}
    for run_dir in run_dirs:
        metrics = read_metrics(run_dir / "metrics.json")
        config = metrics.get("run_config", {})
        sample = metrics.get("sample_metrics", {})
        pairwise = metrics.get("pairwise_metrics", {})
        row = {
            "run": run_dir.name,
            "seed": config.get("seed", ""),
            "known": sample.get("known", ""),
            "known_pairs": pairwise.get("known_pairs", ""),
        }
        for key in SAMPLE_KEYS:
            out_key = f"sample_{key}"
            row[out_key] = sample.get(key, "")
            if isinstance(sample.get(key), (int, float)):
                metrics_by_key.setdefault(out_key, []).append(float(sample[key]))
        for key in PAIR_KEYS:
            out_key = f"pair_{key}"
            row[out_key] = pairwise.get(key, "")
            if isinstance(pairwise.get(key), (int, float)):
                metrics_by_key.setdefault(out_key, []).append(float(pairwise[key]))
        rows.append(row)

    summary = {
        "runs_root": args.runs_root,
        "num_runs": len(rows),
        "runs": rows,
        "aggregate": {key: metric_stats(values) for key, values in sorted(metrics_by_key.items())},
    }

    write_csv(Path(args.output_csv), rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
