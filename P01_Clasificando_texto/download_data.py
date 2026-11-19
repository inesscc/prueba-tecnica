import argparse
import os
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
OUTPUT_DIR = BASE_DIR / "data"
TABLES = ("train", "test")


def get_db_config() -> dict:
    load_dotenv(ENV_PATH)

    def _get(key: str) -> str | None:
        return os.getenv(key) or os.getenv(key.upper())

    required_keys = ("host", "port", "user", "password", "database")
    config = {k: _get(k) for k in required_keys}

    missing = [k for k, v in config.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required environment variables in {ENV_PATH}: {', '.join(missing)}"
        )

    config["port"] = int(config["port"])
    return config


def fetch_table(conn: mysql.connector.MySQLConnection, table_name: str) -> pd.DataFrame:
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT * FROM `{table_name}`")
        rows = cursor.fetchall()
        columns = list(cursor.column_names)
        return pd.DataFrame(rows, columns=columns)
    finally:
        cursor.close()


def download_tables(output_format: str = "parquet") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = get_db_config()

    print(f"Connecting to MySQL database '{config['database']}' at {config['host']}:{config['port']}...")
    conn = mysql.connector.connect(**config)
    try:
        for table in TABLES:
            print(f"Downloading table '{table}'...")
            df = fetch_table(conn, table)

            output_path = OUTPUT_DIR / f"{table}.{output_format}"
            if output_format == "parquet":
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False, encoding="utf-8")

            print(
                f"Saved '{table}' ({len(df):,} rows, {len(df.columns)} columns) -> {output_path}"
            )
    finally:
        conn.close()
        print("Database connection closed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download train and test tables from MySQL to parquet or csv files."
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output file format (default: parquet).",
    )
    args = parser.parse_args()
    download_tables(output_format=args.format)


if __name__ == "__main__":
    main()
