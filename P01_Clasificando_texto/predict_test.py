import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEST_PATH = DATA_DIR / "test.parquet"
TRAIN_PATH = DATA_DIR / "train.parquet"
MODEL_PATH = DATA_DIR / "text_classifier.joblib"
PREDS_PATH = DATA_DIR / "test_predictions.parquet"
METRICS_PATH = DATA_DIR / "test_metrics.json"

FEATURE_COLS = ["actividad", "oficio", "tareas"]
TARGET_COL = "cod_final"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    print(f"Loading test data from {TEST_PATH}...")
    test_df = pd.read_parquet(TEST_PATH)
    X_test = test_df[FEATURE_COLS].fillna("").astype(str)
    y_true = test_df[TARGET_COL].astype(str)

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)
    elapsed = time.perf_counter() - t0

    # Top-1 vs Top-2 decision margin as a confidence proxy
    sorted_scores = np.sort(scores, axis=1)
    top1_score = sorted_scores[:, -1]
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    results_df = test_df.copy()
    results_df["pred_cod_final"] = y_pred
    results_df["correct"] = y_true.values == y_pred
    results_df["top1_score"] = np.round(top1_score, 4)
    results_df["margin"] = np.round(margin, 4)
    results_df.to_parquet(PREDS_PATH, index=False)

    # Core metrics (evaluated over classes present in y_true / union with y_pred)
    test_labels = sorted(y_true.unique())
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=test_labels, average="macro", zero_division=0))
    macro_f1_all = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=test_labels, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, labels=test_labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=test_labels, average="macro", zero_division=0))

    print("=" * 80)
    print("TEST SET EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Test rows           : {len(test_df):,}")
    print(f"Inference time      : {elapsed:.2f}s ({len(test_df) / elapsed:,.0f} rows/s)")
    print(f"Correct predictions : {int(results_df['correct'].sum()):,} / {len(test_df):,}")
    print(f"Errors              : {int((~results_df['correct']).sum()):,}")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Macro F1 (77 test classes) : {macro_f1:.4f}")
    if abs(macro_f1_all - macro_f1) > 1e-6:
        print(f"Macro F1 (all predicted)   : {macro_f1_all:.4f}")
    print(f"Weighted F1         : {weighted_f1:.4f}")
    print(f"Macro Precision     : {macro_prec:.4f}")
    print(f"Macro Recall        : {macro_rec:.4f}")
    print(f"Saved predictions -> {PREDS_PATH}")
    print()

    # Per-class report
    report_dict = classification_report(
        y_true, y_pred, labels=test_labels, output_dict=True, zero_division=0
    )
    per_class = (
        pd.DataFrame({k: v for k, v in report_dict.items() if k in test_labels})
        .T.rename_axis("cod_final")
        .reset_index()
    )
    per_class["support"] = per_class["support"].astype(int)

    if TRAIN_PATH.exists():
        train_counts = pd.read_parquet(TRAIN_PATH, columns=[TARGET_COL])[TARGET_COL].value_counts()
        per_class["train_support"] = per_class["cod_final"].map(train_counts).fillna(0).astype(int)

        # Performance by train support bucket
        bins = [0, 25, 100, 500, 10000]
        labels = ["<=25", "26-100", "101-500", ">500"]
        per_class["train_bucket"] = pd.cut(per_class["train_support"], bins=bins, labels=labels)
        bucket_summary = (
            per_class.groupby("train_bucket", observed=False)
            .agg(
                n_classes=("cod_final", "count"),
                test_rows=("support", "sum"),
                mean_f1=("f1-score", "mean"),
                mean_recall=("recall", "mean"),
            )
            .reset_index()
        )
        print("--- Performance by Training Class Size ---")
        print(bucket_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    print("--- Worst 15 Classes (by F1-score) ---")
    cols_show = ["cod_final", "precision", "recall", "f1-score", "support"]
    if "train_support" in per_class.columns:
        cols_show.append("train_support")
    print(
        per_class.sort_values(["f1-score", "support"], ascending=[True, False])
        .head(15)[cols_show]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print()

    print("--- Best 10 Classes (with test support >= 20) ---")
    print(
        per_class[per_class["support"] >= 20]
        .sort_values(["f1-score", "support"], ascending=[False, False])
        .head(10)[cols_show]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print()

    # Most frequent confusion pairs
    errors_df = results_df[~results_df["correct"]]
    conf_pairs = (
        errors_df.groupby([TARGET_COL, "pred_cod_final"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    print("--- Top 15 Confusion Pairs (True -> Predicted) ---")
    print(conf_pairs.to_string(index=False))
    print()

    # Accuracy on complete vs partially empty text fields
    has_empty = (
        test_df["oficio"].fillna("").str.strip().eq("")
        | test_df["tareas"].fillna("").str.strip().eq("")
    )
    print("--- Accuracy: Complete vs. Empty 'oficio'/'tareas' ---")
    print(
        f"Complete text ({int((~has_empty).sum()):,} rows): "
        f"{results_df.loc[~has_empty, 'correct'].mean():.4f}"
    )
    print(
        f"Empty oficio/tareas ({int(has_empty.sum()):,} rows): "
        f"{results_df.loc[has_empty, 'correct'].mean():.4f}"
    )
    print()

    # Confidence (margin) buckets
    results_df["margin_bucket"] = pd.qcut(
        results_df["margin"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )
    margin_summary = (
        results_df.groupby("margin_bucket", observed=False)
        .agg(
            rows=("correct", "count"),
            min_margin=("margin", "min"),
            max_margin=("margin", "max"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    print("--- Accuracy by Decision Margin Quartile ---")
    print(margin_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # Sample misclassifications
    print("--- Sample Misclassified Rows (10 examples) ---")
    sample_cols = ["id", "actividad", "oficio", "tareas", TARGET_COL, "pred_cod_final", "margin"]
    with pd.option_context("display.max_columns", None, "display.width", 160, "display.max_colwidth", 42):
        print(errors_df.head(10)[sample_cols].to_string(index=False))

    metrics_out = {
        "n_test_rows": int(len(test_df)),
        "n_test_classes": int(len(test_labels)),
        "n_correct": int(results_df["correct"].sum()),
        "n_errors": int((~results_df["correct"]).sum()),
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "inference_seconds": round(elapsed, 4),
    }
    METRICS_PATH.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"\nSaved test metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
