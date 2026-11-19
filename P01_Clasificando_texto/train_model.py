import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.parquet"
MODEL_PATH = DATA_DIR / "text_classifier.joblib"
METRICS_PATH = DATA_DIR / "train_cv_metrics.json"

FEATURE_COLS = ["actividad", "oficio", "tareas"]
TARGET_COL = "cod_final"


def build_pipeline(
    c_value: float = 0.5,
    class_weight: str | None = None,
) -> Pipeline:
    """Build a multi-field TF-IDF (word + char_wb n-grams) + LinearSVC pipeline.

    Each text column ('actividad', 'oficio', 'tareas') is vectorized independently
    so the classifier can learn field-specific weights (e.g., a word appearing in
    'actividad' carries different signal than in 'oficio' or 'tareas').
    """
    transformers = []
    for col in FEATURE_COLS:
        transformers.append(
            (
                f"{col}_word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
                col,
            )
        )
        transformers.append(
            (
                f"{col}_char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
                col,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        transformer_weights={
            "actividad_word": 1.5,
            "actividad_char": 1.2,
            "oficio_word": 1.0,
            "oficio_char": 0.8,
            "tareas_word": 1.0,
            "tareas_char": 0.8,
        },
        remainder="drop",
    )

    classifier = LinearSVC(
        C=c_value,
        class_weight=class_weight,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("clf", classifier),
        ]
    )


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    X = df[FEATURE_COLS].fillna("").astype(str)
    y = df[TARGET_COL].astype(str)
    return X, y


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    c_value: float = 0.5,
    class_weight: str | None = None,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.empty(len(y), dtype=object)
    fold_accs = []
    fold_macro_f1s = []
    fold_weighted_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = build_pipeline(c_value=c_value, class_weight=class_weight)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        oof_preds[val_idx] = preds

        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
        fold_accs.append(acc)
        fold_macro_f1s.append(macro_f1)
        fold_weighted_f1s.append(weighted_f1)
        print(
            f"  Fold {fold}/{n_splits} | "
            f"Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}"
        )

    oof_acc = float(accuracy_score(y, oof_preds))
    oof_macro_f1 = float(f1_score(y, oof_preds, average="macro", zero_division=0))
    oof_weighted_f1 = float(f1_score(y, oof_preds, average="weighted", zero_division=0))

    return {
        "n_splits": n_splits,
        "c_value": c_value,
        "class_weight": class_weight,
        "cv_accuracy_mean": float(np.mean(fold_accs)),
        "cv_accuracy_std": float(np.std(fold_accs)),
        "cv_macro_f1_mean": float(np.mean(fold_macro_f1s)),
        "cv_macro_f1_std": float(np.std(fold_macro_f1s)),
        "cv_weighted_f1_mean": float(np.mean(fold_weighted_f1s)),
        "oof_accuracy": oof_acc,
        "oof_macro_f1": oof_macro_f1,
        "oof_weighted_f1": oof_weighted_f1,
    }


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    print(f"Loading training data from {TRAIN_PATH}...")
    X, y = load_training_data(TRAIN_PATH)
    print(f"Loaded {len(X):,} rows, {y.nunique()} classes.")

    print("\nRunning 5-fold Stratified Cross-Validation...")
    t0 = time.perf_counter()
    cv_metrics = evaluate_cv(X, y, n_splits=5, c_value=0.5, class_weight=None)
    cv_elapsed = time.perf_counter() - t0

    print("\n--- 5-Fold Stratified CV Summary ---")
    print(
        f"Accuracy    : {cv_metrics['cv_accuracy_mean']:.4f} "
        f"(+/- {cv_metrics['cv_accuracy_std']:.4f}) | OOF: {cv_metrics['oof_accuracy']:.4f}"
    )
    print(
        f"Macro F1    : {cv_metrics['cv_macro_f1_mean']:.4f} "
        f"(+/- {cv_metrics['cv_macro_f1_std']:.4f}) | OOF: {cv_metrics['oof_macro_f1']:.4f}"
    )
    print(f"Weighted F1 : {cv_metrics['cv_weighted_f1_mean']:.4f} | OOF: {cv_metrics['oof_weighted_f1']:.4f}")
    print(f"CV elapsed  : {cv_elapsed:.1f}s")

    print("\nTraining final model on full train.parquet...")
    t1 = time.perf_counter()
    final_model = build_pipeline(c_value=0.5, class_weight=None)
    final_model.fit(X, y)
    fit_elapsed = time.perf_counter() - t1

    n_features = final_model.named_steps["features"].transform(X.iloc[:1]).shape[1]
    print(f"Final model trained in {fit_elapsed:.1f}s ({n_features:,} TF-IDF features).")

    joblib.dump(final_model, MODEL_PATH, compress=3)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved model -> {MODEL_PATH} ({model_size_mb:.2f} MB)")

    cv_metrics["n_train_rows"] = int(len(X))
    cv_metrics["n_classes"] = int(y.nunique())
    cv_metrics["n_features"] = int(n_features)
    cv_metrics["fit_seconds"] = round(fit_elapsed, 2)
    METRICS_PATH.write_text(json.dumps(cv_metrics, indent=2), encoding="utf-8")
    print(f"Saved CV metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
