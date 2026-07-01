#!/usr/bin/env python3
"""Fine-tune a full-function vulnerability classifier.

This script is a supervised upper-bound style baseline for the SliceRAG study:
it uses only the full function body as model input, trains on the training split,
and evaluates on the paired PrimeVul-style test split with both sample-level and
adjacent-pair metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        text = str(value).strip().upper()
        if text == "YES":
            return 1
        if text == "NO":
            return 0
    return None


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target = safe_int(row.get("target"))
            if target not in (0, 1):
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_adjacent_pairs_for_validation(
    rows: list[dict[str, Any]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_ratio <= 0:
        return rows, []
    groups = [rows[start : start + 2] for start in range(0, len(rows), 2)]
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_group_count = max(1, int(round(len(groups) * val_ratio)))
    val_groups = groups[:val_group_count]
    train_groups = groups[val_group_count:]
    train_rows = [row for group in train_groups for row in group]
    val_rows = [row for group in val_groups for row in group]
    return train_rows, val_rows


def normalize_cwe(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


class FunctionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            str(row.get("func", "")),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(row["target"]), dtype=torch.long)
        item["row_index"] = torch.tensor(index, dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def metric_dict(targets: list[int], preds: list[int]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for target, pred in zip(targets, preds):
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
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "known": known,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "yes_rate": (tp + fp) / known if known else 0.0,
        "directional_failure_index": recall - specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def clone_prediction_rows_with_threshold(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    cloned = []
    for row in rows:
        new_row = dict(row)
        pred = 1 if float(new_row["prob_yes"]) >= threshold else 0
        target = int(new_row["target"])
        new_row["prediction"] = pred
        new_row["correct"] = int(pred == target)
        cloned.append(new_row)
    return cloned


def metrics_from_prediction_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    thresholded = clone_prediction_rows_with_threshold(rows, threshold)
    targets = [int(row["target"]) for row in thresholded]
    preds = [int(row["prediction"]) for row in thresholded]
    return metric_dict(targets, preds)


def score_for_selection(
    metric_name: str,
    sample_metrics: dict[str, Any],
    pair_metrics: dict[str, Any] | None = None,
) -> float:
    if metric_name == "val_f1":
        return float(sample_metrics["f1"])
    if metric_name == "val_accuracy":
        return float(sample_metrics["accuracy"])
    if metric_name == "val_balanced_accuracy":
        return (float(sample_metrics["recall"]) + float(sample_metrics["specificity"])) / 2.0
    if metric_name == "val_pc":
        if pair_metrics is None:
            return 0.0
        return float(pair_metrics["P-C_rate"])
    raise ValueError(f"Unsupported selection metric: {metric_name}")


def choose_threshold(
    prediction_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    mode: str,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    if mode == "fixed_0.5":
        threshold = 0.5
        sample_metrics = metrics_from_prediction_rows(prediction_rows, threshold)
        pair_metrics, _ = compute_adjacent_pair_metrics(data_rows, clone_prediction_rows_with_threshold(prediction_rows, threshold))
        return threshold, sample_metrics, pair_metrics

    best_threshold = 0.5
    best_sample: dict[str, Any] | None = None
    best_pair: dict[str, Any] | None = None
    best_score = -1.0
    count = int(round((max_threshold - min_threshold) / step)) + 1
    for index in range(count):
        threshold = round(min_threshold + index * step, 10)
        thresholded = clone_prediction_rows_with_threshold(prediction_rows, threshold)
        sample_metrics = metrics_from_prediction_rows(prediction_rows, threshold)
        pair_metrics, _ = compute_adjacent_pair_metrics(data_rows, thresholded)
        score = score_for_selection(mode, sample_metrics, pair_metrics)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_sample = sample_metrics
            best_pair = pair_metrics

    assert best_sample is not None and best_pair is not None
    return best_threshold, best_sample, best_pair


def pair_outcome(vuln_pred: int, fixed_pred: int) -> str:
    if vuln_pred == 1 and fixed_pred == 0:
        return "P-C"
    if vuln_pred == 1 and fixed_pred == 1:
        return "P-V"
    if vuln_pred == 0 and fixed_pred == 0:
        return "P-B"
    if vuln_pred == 0 and fixed_pred == 1:
        return "P-R"
    raise ValueError(f"Unexpected pair predictions: {vuln_pred}, {fixed_pred}")


def compute_adjacent_pair_metrics(
    data_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_by_row = {int(row["row_id"]): row for row in prediction_rows}
    counts = Counter({"P-C": 0, "P-V": 0, "P-B": 0, "P-R": 0})
    pair_rows: list[dict[str, Any]] = []
    skipped = Counter()

    for pair_id, start in enumerate(range(0, len(data_rows), 2)):
        pair = data_rows[start : start + 2]
        if len(pair) != 2:
            skipped["incomplete_pair"] += 1
            continue
        vuln_pos = next((start + offset for offset, row in enumerate(pair) if safe_int(row.get("target")) == 1), None)
        fixed_pos = next((start + offset for offset, row in enumerate(pair) if safe_int(row.get("target")) == 0), None)
        if vuln_pos is None or fixed_pos is None:
            skipped["missing_target_pair"] += 1
            continue
        vuln_pred_row = pred_by_row.get(vuln_pos)
        fixed_pred_row = pred_by_row.get(fixed_pos)
        if not vuln_pred_row or not fixed_pred_row:
            skipped["missing_prediction"] += 1
            continue
        vuln_pred = safe_int(vuln_pred_row.get("prediction"))
        fixed_pred = safe_int(fixed_pred_row.get("prediction"))
        if vuln_pred not in (0, 1) or fixed_pred not in (0, 1):
            skipped["unknown_prediction"] += 1
            continue
        label = pair_outcome(vuln_pred, fixed_pred)
        counts[label] += 1
        vuln = data_rows[vuln_pos]
        fixed = data_rows[fixed_pos]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "vuln_row_id": vuln_pos,
                "fixed_row_id": fixed_pos,
                "vuln_idx": vuln.get("idx", ""),
                "fixed_idx": fixed.get("idx", ""),
                "project": vuln.get("project") or fixed.get("project", ""),
                "cwe": normalize_cwe(vuln.get("cwe") or fixed.get("cwe")),
                "vuln_pred": vuln_pred,
                "fixed_pred": fixed_pred,
                "vuln_prob_yes": vuln_pred_row.get("prob_yes", ""),
                "fixed_prob_yes": fixed_pred_row.get("prob_yes", ""),
                "pair_outcome": label,
            }
        )

    known_pairs = sum(counts.values())
    summary = {
        "num_data_rows": len(data_rows),
        "candidate_adjacent_pairs": math.ceil(len(data_rows) / 2),
        "known_pairs": known_pairs,
        "unknown_pairs": int(sum(skipped.values())),
        "skip_counts": dict(skipped),
        "P-C": counts["P-C"],
        "P-V": counts["P-V"],
        "P-B": counts["P-B"],
        "P-R": counts["P-R"],
        "P-C_rate": counts["P-C"] / known_pairs if known_pairs else 0.0,
        "P-V_rate": counts["P-V"] / known_pairs if known_pairs else 0.0,
        "P-B_rate": counts["P-B"] / known_pairs if known_pairs else 0.0,
        "P-R_rate": counts["P-R"] / known_pairs if known_pairs else 0.0,
    }
    return summary, pair_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    model,
    loader: DataLoader,
    rows: list[dict[str, Any]],
    device: torch.device,
    fp16: bool,
    threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    targets: list[int] = []
    preds: list[int] = []

    with torch.no_grad():
        progress = tqdm(loader, desc="eval", leave=False)
        for batch in progress:
            row_indices = batch.pop("row_index").tolist()
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            use_amp = fp16 and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
            logits = outputs.logits.detach().float().cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
            batch_preds = (probs[:, 1] >= threshold).astype(int).tolist()
            batch_targets = labels.detach().cpu().tolist()
            targets.extend(int(item) for item in batch_targets)
            preds.extend(int(item) for item in batch_preds)

            for local_idx, row_id in enumerate(row_indices):
                row = rows[int(row_id)]
                logit_no = float(logits[local_idx][0])
                logit_yes = float(logits[local_idx][1])
                prob_no = float(probs[local_idx][0])
                prob_yes = float(probs[local_idx][1])
                pred = int(batch_preds[local_idx])
                target = int(row["target"])
                prediction_rows.append(
                    {
                        "row_id": int(row_id),
                        "idx": row.get("idx", ""),
                        "target": target,
                        "prediction": pred,
                        "correct": int(pred == target),
                        "prob_no": prob_no,
                        "prob_yes": prob_yes,
                        "logit_no": logit_no,
                        "logit_yes": logit_yes,
                        "project": row.get("project", ""),
                        "cwe": normalize_cwe(row.get("cwe")),
                        "func_char_len": len(str(row.get("func", ""))),
                    }
                )

    return metric_dict(targets, preds), sorted(prediction_rows, key=lambda item: item["row_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a full-function code classifier.")
    parser.add_argument("--train", required=True, help="Training JSONL with func and target fields")
    parser.add_argument("--test", required=True, help="Test JSONL with func and target fields")
    parser.add_argument("--model", required=True, help="HF model name/path, e.g. microsoft/graphcodebert-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--val", default="", help="Optional validation JSONL. If omitted, split train by pairs.")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Pair-level validation ratio from train split.")
    parser.add_argument(
        "--threshold-mode",
        default="fixed_0.5",
        choices=["fixed_0.5", "val_f1", "val_accuracy", "val_balanced_accuracy", "val_pc"],
        help="How to select the final decision threshold.",
    )
    parser.add_argument(
        "--selection-metric",
        default="val_f1",
        choices=["val_f1", "val_accuracy", "val_balanced_accuracy", "val_pc"],
        help="Validation metric for selecting the best epoch.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(Path(args.train), args.max_train_samples or None)
    if args.val:
        val_rows = load_jsonl(Path(args.val), args.max_val_samples or None)
    else:
        train_rows, val_rows = split_adjacent_pairs_for_validation(train_rows, args.val_ratio, args.seed)
        if args.max_val_samples:
            val_rows = val_rows[: args.max_val_samples]
    test_rows = load_jsonl(Path(args.test), args.max_test_samples or None)
    if not train_rows or not test_rows:
        raise SystemExit("Empty train/test rows after loading.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token or tokenizer.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_dataset = FunctionDataset(train_rows, tokenizer, args.max_length)
    val_dataset = FunctionDataset(val_rows, tokenizer, args.max_length) if val_rows else None
    test_dataset = FunctionDataset(test_rows, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_update_steps = max(1, update_steps_per_epoch * args.epochs)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    run_config = {
        "train": args.train,
        "test": args.test,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "max_length": args.max_length,
        "val": args.val,
        "val_ratio": args.val_ratio,
        "threshold_mode": args.threshold_mode,
        "selection_metric": args.selection_metric,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_step": args.threshold_step,
        "seed": args.seed,
        "device": str(device),
        "fp16": args.fp16,
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "num_test_rows": len(test_rows),
        "train_label_counts": dict(Counter(int(row["target"]) for row in train_rows)),
        "val_label_counts": dict(Counter(int(row["target"]) for row in val_rows)),
        "test_label_counts": dict(Counter(int(row["target"]) for row in test_rows)),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    log_path = output_dir / "train_log.jsonl"
    global_step = 0
    best_record: dict[str, Any] | None = None
    best_model_dir = output_dir / "best_model"
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen_batches = 0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch.pop("row_index")
            batch = {key: value.to(device) for key, value in batch.items()}
            use_amp = args.fp16 and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            seen_batches += 1

            should_step = step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(loss=total_loss / max(1, seen_batches), lr=scheduler.get_last_lr()[0])

        if val_loader is not None:
            _, val_prediction_rows = evaluate(model, val_loader, val_rows, device, args.fp16, threshold=0.5)
            selected_threshold, val_metrics, val_pair_metrics = choose_threshold(
                val_prediction_rows,
                val_rows,
                args.threshold_mode,
                args.threshold_min,
                args.threshold_max,
                args.threshold_step,
            )
            selection_score = score_for_selection(args.selection_metric, val_metrics, val_pair_metrics)
        else:
            selected_threshold = 0.5
            val_metrics = {}
            val_pair_metrics = {}
            selection_score = 0.0

        eval_metrics, _ = evaluate(model, test_loader, test_rows, device, args.fp16, threshold=selected_threshold)
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": total_loss / max(1, seen_batches),
            "selected_threshold": selected_threshold,
            "selection_score": selection_score,
            "val": val_metrics,
            "val_pairwise": val_pair_metrics,
            "eval": eval_metrics,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")
        print(json.dumps(epoch_record, indent=2))

        if val_loader is not None and (best_record is None or selection_score > float(best_record["selection_score"])):
            if best_model_dir.exists():
                shutil.rmtree(best_model_dir)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            best_record = epoch_record

    final_threshold = 0.5
    if best_record is not None:
        final_threshold = float(best_record["selected_threshold"])
        model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
        model.to(device)
    elif val_loader is not None:
        final_threshold = float(epoch_record["selected_threshold"])

    final_metrics, prediction_rows = evaluate(model, test_loader, test_rows, device, args.fp16, threshold=final_threshold)
    pair_summary, pair_rows = compute_adjacent_pair_metrics(test_rows, prediction_rows)

    predictions_csv = output_dir / "predictions.csv"
    write_csv(predictions_csv, prediction_rows)
    write_csv(output_dir / "pairwise_pairs.csv", pair_rows)
    write_csv(
        output_dir / "pairwise_metrics.csv",
        [
            {
                "method": "fine_tuned_full_function_classifier",
                "known_pairs": pair_summary["known_pairs"],
                "unknown_pairs": pair_summary["unknown_pairs"],
                "P-C": pair_summary["P-C"],
                "P-V": pair_summary["P-V"],
                "P-B": pair_summary["P-B"],
                "P-R": pair_summary["P-R"],
                "P-C_rate": pair_summary["P-C_rate"],
                "P-V_rate": pair_summary["P-V_rate"],
                "P-B_rate": pair_summary["P-B_rate"],
                "P-R_rate": pair_summary["P-R_rate"],
            }
        ],
    )

    summary = {
        "run_config": run_config,
        "best_validation_record": best_record,
        "final_threshold": final_threshold,
        "sample_metrics": final_metrics,
        "pairwise_metrics": pair_summary,
        "predictions_csv": str(predictions_csv),
        "pairwise_pairs_csv": str(output_dir / "pairwise_pairs.csv"),
        "pairwise_metrics_csv": str(output_dir / "pairwise_metrics.csv"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "pairwise_summary.json").write_text(json.dumps(pair_summary, indent=2), encoding="utf-8")

    if args.save_model:
        model_dir = output_dir / "model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
