import os
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_PATH = OUTPUT_DIR / "noticias.parquet"
TABLE_NAME = "noticias"


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


def download_noticias() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = get_db_config()

    print(
        f"Connecting to MySQL database '{config['database']}' "
        f"at {config['host']}:{config['port']}..."
    )
    conn = mysql.connector.connect(**config)
    try:
        cursor = conn.cursor()
        try:
            print(f"Downloading table '{TABLE_NAME}'...")
            cursor.execute(f"SELECT * FROM `{TABLE_NAME}`")
            rows = cursor.fetchall()
            columns = list(cursor.column_names)
            df = pd.DataFrame(rows, columns=columns)
        finally:
            cursor.close()
    finally:
        conn.close()
        print("Database connection closed.")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(
        f"Saved '{TABLE_NAME}' ({len(df):,} rows, {len(df.columns)} columns: "
        f"{list(df.columns)}) -> {OUTPUT_PATH}"
    )
    return df


if __name__ == "__main__":
    download_noticias()
