from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FILES = ("train", "test")
TEXT_COLS = ("actividad", "oficio", "tareas")
TARGET_COL = "cod_final"
ID_COL = "id"


def format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def inspect_dataframe(name: str, path: Path, df: pd.DataFrame) -> None:
    print("=" * 80)
    print(f"DATASET: {name.upper()} ({path.name})")
    print("=" * 80)

    file_size = path.stat().st_size
    mem_usage = int(df.memory_usage(deep=True).sum())
    print(f"File size on disk : {format_bytes(file_size)}")
    print(f"In-memory size    : {format_bytes(mem_usage)}")
    print(f"Shape             : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print()

    # Column schema, nulls, blanks, cardinality
    schema_rows = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = (null_count / len(df) * 100) if len(df) else 0.0
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            blank_count = int(series.dropna().astype(str).str.strip().eq("").sum())
        else:
            blank_count = 0
        schema_rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "nulls": null_count,
                "null_%": round(null_pct, 2),
                "empty_str": blank_count,
                "nunique": int(series.nunique(dropna=True)),
            }
        )
    print("--- Columns, Types & Missing Values ---")
    print(pd.DataFrame(schema_rows).to_string(index=False))
    print()

    # Duplicate checks
    print("--- Duplicate Checks ---")
    print(f"Exact duplicate rows             : {int(df.duplicated().sum()):,}")
    if ID_COL in df.columns:
        print(f"Duplicate '{ID_COL}' values           : {int(df[ID_COL].duplicated().sum()):,}")
    present_text_cols = [c for c in TEXT_COLS if c in df.columns]
    if present_text_cols:
        dup_features = int(df.duplicated(subset=present_text_cols).sum())
        print(f"Duplicate text-feature rows      : {dup_features:,}")
        if TARGET_COL in df.columns and dup_features > 0:
            label_counts = df.groupby(present_text_cols, dropna=False)[TARGET_COL].nunique()
            conflicting_groups = int((label_counts > 1).sum())
            print(f"Text groups with conflicting labels: {conflicting_groups:,}")
    print()

    # Text length statistics
    if present_text_cols:
        print("--- Text Length Statistics (characters / words) ---")
        text_stats = []
        for col in present_text_cols:
            s = df[col].fillna("").astype(str).str.strip()
            char_len = s.str.len()
            word_len = s.str.split().str.len()
            text_stats.append(
                {
                    "column": col,
                    "char_min": int(char_len.min()),
                    "char_med": round(float(char_len.median()), 1),
                    "char_mean": round(float(char_len.mean()), 1),
                    "char_p95": round(float(char_len.quantile(0.95)), 1),
                    "char_max": int(char_len.max()),
                    "word_med": round(float(word_len.median()), 1),
                    "word_mean": round(float(word_len.mean()), 1),
                    "word_max": int(word_len.max()),
                }
            )
        print(pd.DataFrame(text_stats).to_string(index=False))
        print()

    # Target distribution
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts(dropna=False)
        print(f"--- Target Distribution ('{TARGET_COL}') ---")
        print(f"Total unique classes : {len(vc):,}")
        print(f"Classes with == 1 row: {int((vc == 1).sum()):,}")
        print(f"Classes with < 5 rows: {int((vc < 5).sum()):,}")
        print(f"Classes with < 10 rows: {int((vc < 10).sum()):,}")
        print(
            f"Imbalance ratio (max/min): {vc.iloc[0]:,} / {vc.iloc[-1]:,} "
            f"= {vc.iloc[0] / vc.iloc[-1]:.1f}x"
        )
        top10 = pd.DataFrame(
            {"count": vc.head(10), "pct": (vc.head(10) / len(df) * 100).round(2)}
        )
        print("\nTop 10 classes:")
        print(top10.to_string())

        bottom10 = pd.DataFrame(
            {"count": vc.tail(10), "pct": (vc.tail(10) / len(df) * 100).round(3)}
        )
        print("\nBottom 10 classes:")
        print(bottom10.to_string())
        print()

    # Sample rows
    print("--- Sample Rows (first 5) ---")
    with pd.option_context("display.max_columns", None, "display.width", 140, "display.max_colwidth", 50):
        print(df.head(5).to_string(index=False))
    print()


def compare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("=" * 80)
    print("TRAIN vs. TEST COMPARISON")
    print("=" * 80)

    total_rows = len(train_df) + len(test_df)
    print(
        f"Split ratio          : train {len(train_df):,} ({len(train_df)/total_rows*100:.1f}%) "
        f"/ test {len(test_df):,} ({len(test_df)/total_rows*100:.1f}%)"
    )
    print(f"Columns identical    : {list(train_df.columns) == list(test_df.columns)}")
    print(f"Dtypes identical     : {train_df.dtypes.equals(test_df.dtypes)}")

    if ID_COL in train_df.columns and ID_COL in test_df.columns:
        id_overlap = len(set(train_df[ID_COL]).intersection(set(test_df[ID_COL])))
        print(f"Overlapping '{ID_COL}'s     : {id_overlap:,}")

    present_text_cols = [c for c in TEXT_COLS if c in train_df.columns and c in test_df.columns]
    if present_text_cols:
        train_keys = set(train_df[present_text_cols].fillna("").itertuples(index=False, name=None))
        test_keys = list(test_df[present_text_cols].fillna("").itertuples(index=False, name=None))
        overlapping_test_rows = sum(1 for k in test_keys if k in train_keys)
        print(
            f"Test rows with exact text match in train: "
            f"{overlapping_test_rows:,} ({overlapping_test_rows/len(test_df)*100:.2f}%)"
        )

    if TARGET_COL in train_df.columns and TARGET_COL in test_df.columns:
        train_classes = set(train_df[TARGET_COL].dropna().unique())
        test_classes = set(test_df[TARGET_COL].dropna().unique())
        only_in_test = sorted(test_classes - train_classes)
        only_in_train = sorted(train_classes - test_classes)
        print(f"Classes in train     : {len(train_classes):,}")
        print(f"Classes in test      : {len(test_classes):,}")
        print(f"Classes ONLY in test : {len(only_in_test)} {only_in_test[:15]}")
        print(f"Classes ONLY in train: {len(only_in_train)} {only_in_train[:15]}")

        # Compare top 10 class proportions to check stratification
        train_pct = (train_df[TARGET_COL].value_counts(normalize=True) * 100).round(2)
        test_pct = (test_df[TARGET_COL].value_counts(normalize=True) * 100).round(2)
        comp = pd.DataFrame({"train_%": train_pct, "test_%": test_pct}).fillna(0)
        comp["diff_%"] = (comp["test_%"] - comp["train_%"]).round(2)
        print("\nTop 10 class share comparison (stratification check):")
        print(comp.sort_values("train_%", ascending=False).head(10).to_string())
    print()


def main() -> None:
    datasets: dict[str, pd.DataFrame] = {}
    for name in FILES:
        path = DATA_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Expected parquet file not found: {path}")
        df = pd.read_parquet(path)
        datasets[name] = df
        inspect_dataframe(name, path, df)

    if "train" in datasets and "test" in datasets:
        compare_datasets(datasets["train"], datasets["test"])


if __name__ == "__main__":
    main()
