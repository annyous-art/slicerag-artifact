#!/usr/bin/env python3
"""Train and evaluate a lightweight SliceRAG router/stacking classifier."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_METHODS = [
    "zero_shot",
    "author_no_yes",
    "yes_only",
    "no_only",
    "icl",
    "old_rag",
    "positive_weighted_rag",
    "balanced_grouped_no_label_bc",
]

DEFAULT_FEATURES = [
    # Code length and complexity.
    "char_len",
    "token_len_approx",
    "line_count",
    "macro_count",
    "preproc_count",
    "call_count",
    "unique_call_count",
    "loop_count",
    "cond_count",
    "ptr_count",
    "memory_api_count",
    "string_api_count",
    "bounds_keyword_count",
    "error_handling_keyword_count",
    "lifecycle_keyword_count",
    "null_keyword_count",
    # Retrieval quality. Do not include target-agreement fields.
    "retrieval_num_selected_query_chunks",
    "retrieval_num_index_matches",
    "retrieval_top1_score",
    "retrieval_top2_score",
    "retrieval_score_gap_top1_top2",
    "retrieval_mean_match_score",
    "retrieval_top_pos_score_max",
    "retrieval_top_neg_score_max",
    "retrieval_pos_neg_margin_max",
    "retrieval_pos_neg_margin_mean",
    "retrieval_positive_evidence_count",
    "retrieval_negative_evidence_count",
    "retrieval_unknown_evidence_count",
    "retrieval_top1_evidence_role",
    "retrieval_top1_evidence_polarity",
    "retrieval_same_project_match_count",
    "retrieval_same_cwe_match_count",
    "retrieval_role_count_vulnerable_changed",
    "retrieval_role_count_fixed_changed",
    "retrieval_role_count_safe_background",
    "retrieval_role_count_vulnerable_context",
    "retrieval_role_count_unknown",
    # Model predictions and prompt sizes.
    "zero_shot_pred",
    "author_no_yes_pred",
    "yes_only_pred",
    "no_only_pred",
    "icl_pred",
    "old_rag_pred",
    "positive_weighted_rag_pred",
    "balanced_grouped_no_label_bc_pred",
    "zero_shot_message_chars",
    "author_no_yes_message_chars",
    "yes_only_message_chars",
    "no_only_message_chars",
    "icl_message_chars",
    "old_rag_message_chars",
    "positive_weighted_rag_message_chars",
    "balanced_grouped_no_label_bc_message_chars",
    "old_rag_prompt_tokens",
    "positive_weighted_rag_prompt_tokens",
    "balanced_grouped_no_label_bc_prompt_tokens",
    "method_known_pred_count",
    "method_yes_count",
    "method_no_count",
    "method_yes_rate",
    "method_disagree",
    "method_pred_entropy",
    # Dataset descriptors available for analysis. Use --drop-categorical-analysis-fields
    # if you want a more deployment-realistic setting without project/CWE.
    "project",
    "cwe",
]


LEAKAGE_PATTERNS = [
    re.compile(r"(^|_)target$"),
    re.compile(r"_correct$"),
    re.compile(r"agrees_target$"),
    re.compile(r"response$"),
]


def safe_int(value: Any, default=None):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def metric_dict(y_true, y_pred) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "yes_rate": float(np.mean(y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def is_leaky_feature(name: str) -> bool:
    return any(pattern.search(name) for pattern in LEAKAGE_PATTERNS)


def select_features(df: pd.DataFrame, requested: list[str], drop_categorical_analysis_fields: bool) -> list[str]:
    features = []
    for col in requested:
        if col not in df.columns:
            continue
        if is_leaky_feature(col):
            continue
        if drop_categorical_analysis_fields and col in {"project", "cwe"}:
            continue
        features.append(col)
    return features


def maybe_add_diagnostic_features(features: list[str], include_diff_label_features: bool) -> list[str]:
    if include_diff_label_features and "label_changed_line_count" not in features:
        return features + ["label_changed_line_count"]
    return features


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical = []
    numeric = []
    for col in features:
        if df[col].dtype == object:
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


def coerce_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError("Input feature table must contain a target column.")
    df = df[df["target"].isin([0, 1])].copy()

    # Empty strings from CSV become NaN in many fields. Prediction columns should be numeric when possible.
    for col in df.columns:
        if col in {"sample_key", "idx", "project", "commit_id", "cwe", "cve", "func_hash", "file_name"}:
            continue
        if df[col].dtype == object:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() > 0:
                df[col] = numeric
    return df


def build_model(model_name: str, numeric_features: list[str], categorical_features: list[str], random_state: int) -> Pipeline:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if model_name == "logistic":
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    if model_name == "logistic":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def choose_threshold(y_true: np.ndarray, prob: np.ndarray, mode: str) -> float:
    if mode == "fixed_0.5":
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (prob >= threshold).astype(int)
        if mode == "fold_f1":
            score = f1_score(y_true, pred, zero_division=0)
        elif mode == "fold_accuracy":
            score = accuracy_score(y_true, pred)
        else:
            raise ValueError(f"Unsupported threshold mode: {mode}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def transformed_feature_names(model: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    names: list[str] = []
    if numeric_features:
        names.extend([f"num__{name}" for name in numeric_features])
    if categorical_features:
        cat_pipe = preprocessor.named_transformers_["cat"]
        onehot = cat_pipe.named_steps["onehot"]
        names.extend([f"cat__{name}" for name in onehot.get_feature_names_out(categorical_features)])
    return names


def extract_importance(model: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    feature_names = transformed_feature_names(model, numeric_features, categorical_features)
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        importance = np.abs(values)
        signed = values
    elif hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        signed = importance
    else:
        importance = np.zeros(len(feature_names))
        signed = importance
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
            "signed_value": signed,
        }
    ).sort_values("importance", ascending=False)


def evaluate_methods(df: pd.DataFrame, method_names: list[str]) -> dict[str, dict[str, Any]]:
    results = {}
    y = df["target"].astype(int).to_numpy()
    for name in method_names:
        col = f"{name}_pred"
        if col not in df.columns:
            continue
        pred = pd.to_numeric(df[col], errors="coerce")
        mask = pred.isin([0, 1]).to_numpy()
        if not mask.any():
            continue
        results[name] = {
            "n": int(mask.sum()),
            **metric_dict(y[mask], pred[mask].astype(int).to_numpy()),
        }
    return results


def oracle_metrics(df: pd.DataFrame, method_names: list[str]) -> dict[str, Any]:
    y = df["target"].astype(int).to_numpy()
    correct_any = np.zeros(len(df), dtype=bool)
    covered = np.zeros(len(df), dtype=bool)
    for name in method_names:
        col = f"{name}_pred"
        if col not in df.columns:
            continue
        pred = pd.to_numeric(df[col], errors="coerce")
        mask = pred.isin([0, 1]).to_numpy()
        covered |= mask
        correct_any |= mask & (pred.fillna(-1).astype(int).to_numpy() == y)
    return {
        "covered": int(covered.sum()),
        "oracle_correct": int(correct_any.sum()),
        "oracle_accuracy": float(correct_any.sum() / covered.sum()) if covered.any() else 0.0,
    }


def train_oof(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    folds: int,
    random_state: int,
    threshold_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], Pipeline, pd.DataFrame, list[str], list[str]]:
    y = df["target"].astype(int).to_numpy()
    numeric_features, categorical_features = split_feature_types(df, features)
    oof_prob = np.zeros(len(df), dtype=float)
    oof_pred = np.zeros(len(df), dtype=int)
    fold_metrics = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(skf.split(df[features], y), start=1):
        model = build_model(model_name, numeric_features, categorical_features, random_state + fold)
        model.fit(df.iloc[train_idx][features], y[train_idx])
        train_prob = model.predict_proba(df.iloc[train_idx][features])[:, 1]
        threshold = choose_threshold(y[train_idx], train_prob, threshold_mode)
        prob = model.predict_proba(df.iloc[test_idx][features])[:, 1]
        pred = (prob >= threshold).astype(int)
        oof_prob[test_idx] = prob
        oof_pred[test_idx] = pred
        fold_metrics.append({"fold": fold, "threshold": threshold, **metric_dict(y[test_idx], pred)})

    final_model = build_model(model_name, numeric_features, categorical_features, random_state)
    final_model.fit(df[features], y)
    importance = extract_importance(final_model, numeric_features, categorical_features)
    return oof_pred, oof_prob, fold_metrics, final_model, importance, numeric_features, categorical_features


def write_oof_predictions(path: Path, df: pd.DataFrame, oof_pred: np.ndarray, oof_prob: np.ndarray, method_names: list[str]) -> None:
    out = df[["sample_key", "idx", "target", "project", "cwe", "char_len", "line_count"]].copy()
    out["router_pred"] = oof_pred
    out["router_prob_yes"] = oof_prob
    out["router_correct"] = (out["router_pred"].astype(int) == out["target"].astype(int)).astype(int)
    for name in method_names:
        for suffix in ("pred", "correct"):
            col = f"{name}_{suffix}"
            if col in df.columns:
                out[col] = df[col]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight router over SliceRAG per-sample features.")
    parser.add_argument("--features-csv", default="outdir/analysis/per_sample_feature_table.csv")
    parser.add_argument("--output-dir", default="outdir/router")
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threshold", choices=["fixed_0.5", "fold_f1", "fold_accuracy"], default="fold_f1")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument(
        "--drop-categorical-analysis-fields",
        action="store_true",
        help="Drop project/CWE features for a more deployment-realistic setting.",
    )
    parser.add_argument(
        "--include-diff-label-features",
        action="store_true",
        help="Include diff-derived label features such as label_changed_line_count. Diagnostic only.",
    )
    args = parser.parse_args()

    df = coerce_dataframe(Path(args.features_csv))
    requested_features = maybe_add_diagnostic_features(args.features, args.include_diff_label_features)
    features = select_features(df, requested_features, args.drop_categorical_analysis_fields)
    if not features:
        raise ValueError("No usable features selected.")

    oof_pred, oof_prob, fold_metrics, _model, importance, numeric_features, categorical_features = train_oof(
        df=df,
        features=features,
        model_name=args.model,
        folds=args.folds,
        random_state=args.seed,
        threshold_mode=args.threshold,
    )
    y = df["target"].astype(int).to_numpy()
    router_metrics = metric_dict(y, oof_pred)
    method_metrics = evaluate_methods(df, args.methods)
    oracle = oracle_metrics(df, args.methods)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.model}_metrics.json"
    importance_path = output_dir / f"{args.model}_feature_importance.csv"
    predictions_path = output_dir / f"{args.model}_oof_predictions.csv"

    metrics = {
        "model": args.model,
        "folds": args.folds,
        "seed": args.seed,
        "threshold": args.threshold,
        "num_rows": int(len(df)),
        "num_features": int(len(features)),
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "router_oof": router_metrics,
        "fold_metrics": fold_metrics,
        "method_baselines": method_metrics,
        "method_oracle": oracle,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    importance.to_csv(importance_path, index=False)
    write_oof_predictions(predictions_path, df, oof_pred, oof_prob, args.methods)

    print(json.dumps({"router_oof": router_metrics, "method_baselines": method_metrics, "method_oracle": oracle}, indent=2, ensure_ascii=False))
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote feature importance: {importance_path}")
    print(f"Wrote OOF predictions: {predictions_path}")


if __name__ == "__main__":
    main()
