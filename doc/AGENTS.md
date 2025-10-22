# Repository Guidelines

## Project Structure & Module Organization
Source code for agents and utilities lives under `src/`, with entry points such as `milvus_client.py`, `embedding_manager.py`, and `nl_query_agent.py`. Configuration defaults sit in `src/config.py`, while runnable scripts (`concept_search_app.py`, `natural_language_agent.py`, `run_benchmark.py`) stay at the repository root for quick invocation. Tests reside in `test/`, and CSV references or docs stay alongside this guide at the top level.

## Build, Test, and Development Commands
Use `uv sync` to install or refresh dependencies defined in `pyproject.toml`/`uv.lock`. Run the main agent locally via `uv run python natural_language_agent.py`. Lint with `uv run ruff check src test`, format with `uv run black src test`, and execute the full suite through `uv run pytest test`.

## Coding Style & Naming Conventions
Apply 4-space indentation, maintain type hints where practical, and include docstrings for public entry points. Respect the 120-character line length enforced by Ruff and Black, keeping imports sorted automatically. Modules use lowercase with underscores, classes use PascalCase, and functions plus variables remain snake_case.

## Testing Guidelines
Pytest drives verification; add new modules as `test_<feature>.py` under `test/`. Prefer factories or fixtures over direct Milvus calls, and assert on both vector results and metadata when covering complex flows. Run `uv run pytest` before submitting changes.

## Commit & Pull Request Guidelines
Follow Conventional Commit prefixes (e.g., `feat: add hybrid search fallback`) and keep each commit focused. Pull requests should reference the originating issue, summarize test evidence (`uv run pytest`, manual checks), and surface logs or screenshots for Milvus interactions. Highlight any schema or configuration adjustments explicitly.

## Configuration & Security Notes
Populate `.env` with connection settings (`MILVUS_HOST`, `METADATA_COLLECTION`, `EMBEDDING_MODEL`) instead of hardcoding values. Consult `DATABASE_INFO.md` for sample expectations, and never commit real credentials. Tune embedding experiments by toggling `EMBEDDING_MODEL` or `USE_FP16` through environment variables to keep benchmarks reproducible.
