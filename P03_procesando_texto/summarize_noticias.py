import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "noticias.parquet"
OUTPUT_PATH = DATA_DIR / "noticias_resumenes.parquet"
OUTPUT_CSV_PATH = DATA_DIR / "noticias_resumenes.csv"

OPENROUTER_URL = "https://openrouter.ai/api/v1/responses"
MODEL = "google/gemini-2.5-flash-lite"
MAX_WORDS = 50
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 5


def get_api_key() -> str:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("API_KEY") or os.getenv("api_key") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(f"Missing API_KEY in {ENV_PATH}")
    return api_key.strip()


def summarize_article(titular: str, cuerpo: str, api_key: str) -> tuple[str, dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = (
        "Resume la siguiente noticia en español en un solo párrafo claro y objetivo. "
        f"El resumen NO debe superar las {MAX_WORDS} palabras (apunta a entre 30 y 45 palabras). "
        "Devuelve únicamente el texto del resumen, sin encabezados, comillas ni viñetas.\n\n"
        f"Titular: {titular}\n"
        f"Cuerpo: {cuerpo}"
    )

    payload = {
        "model": MODEL,
        "input": prompt,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 120,
        "temperature": 0.2,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            retry_after = response.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * (2 ** (attempt - 1))
            print(f"  HTTP {response.status_code} (intento {attempt}/{MAX_RETRIES}), reintentando en {wait_s:.1f}s...")
            time.sleep(wait_s)
            continue
        response.raise_for_status()
        data = response.json()
        resumen = data["output"][0]["content"][0]["text"].strip()
        usage = data.get("usage", {})
        return resumen, usage

    raise RuntimeError("Failed to get a valid response from OpenRouter after retries.")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    api_key = get_api_key()
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} articles from {INPUT_PATH}")
    print(f"Model: {MODEL} | Reasoning: disabled (effort='none')\n")

    resumenes: list[str] = []
    total_reasoning_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, row in df.iterrows():
        titular = str(row["titular"])
        cuerpo = str(row["cuerpo"])

        resumen, usage = summarize_article(titular, cuerpo, api_key)
        word_count = len(resumen.split())

        # Safety retry if the model ever exceeds MAX_WORDS
        if word_count > MAX_WORDS:
            resumen, usage = summarize_article(
                titular,
                f"Acorta este resumen a menos de 40 palabras sin perder la idea central: {resumen}",
                api_key,
            )
            word_count = len(resumen.split())

        resumenes.append(resumen)

        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
        total_reasoning_tokens += reasoning_tokens

        print(f"[{idx + 1}/{len(df)}] ({word_count} palabras | reasoning_tokens={reasoning_tokens})")
        print(f"  Titular : {titular}")
        print(f"  Resumen : {resumen}\n")

        if idx < len(df) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    df["resumen"] = resumenes
    df.to_parquet(OUTPUT_PATH, index=False)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    word_counts = df["resumen"].str.split().str.len()
    print("=" * 80)
    print(f"Saved summaries -> {OUTPUT_PATH}")
    print(f"Saved CSV       -> {OUTPUT_CSV_PATH}")
    print(
        f"Word count check: min={int(word_counts.min())}, "
        f"max={int(word_counts.max())}, "
        f"all <= {MAX_WORDS}: {bool((word_counts <= MAX_WORDS).all())}"
    )
    print(
        f"Total tokens: input={total_input_tokens:,}, "
        f"output={total_output_tokens:,}, "
        f"reasoning={total_reasoning_tokens:,}"
    )


if __name__ == "__main__":
    main()
