# AGENTS.md

Technical exam for INE (data scientist). The task statement is in `README.md`. Submit through a pull request from a fork to `master` of `inesscc/prueba-tecnica`.

## Delivery conventions (from README)

- **Branch:** `diego_quintana` (format `nombre_apellido`). Create it before the first commit; `master` stays identical to upstream.
- **Commit message:** exactly `Diego Quintana` (format `nombre apellido`, separated by a space, no prefix and no body).
- **Pull request:** from the `diego_quintana` branch of the fork to `master` of `inesscc/prueba-tecnica`.
- **Files:** `.ipynb`, `.py` or `.R`. The README asks for a single file `nombre_apellido.<ext>`; the author decided to submit one notebook per task (`01_`…`04_`). Each notebook states which exam question it answers.
- **Python:** include `requirements.txt` with the PR. Regenerate it from `pyproject.toml` after any dependency change:
  `uv export --package prueba-tecnica --no-hashes --no-dev --no-emit-workspace --no-annotate --format requirements-txt -o requirements.txt`

## Repository rules

- Leave other candidates' files (`ignacio_agloni.py`, `klaus_lehmann.py`) untouched: they belong to upstream.
- Notebook content (markdown and comments) is written in Spanish, for the evaluators.
- The OpenRouter key goes only in `.env` (gitignored; template in `example.env`). Commit outputs and code that read the key with `os.environ`.
- `data/` is gitignored: it holds local copies of the tables.

## Database gotchas

- The MySQL database is shared and slow: a full `SELECT *` on `train` fails with error 3024 (statement timeout). Download in `id` ranges (see `descargar_tabla` in `01_clasificador_texto.ipynb`).
- It limits simultaneous connections (error 1203): connect through `conectar()`, which retries with backoff, and close each connection with `with`.
