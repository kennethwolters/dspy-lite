# Agent Guidance

## Project Snapshot

- `dspy-lite` is a slim fork of DSPy that preserves the public `import dspy` API.
- The main fork delta is replacing `litellm` with `litelm` while keeping DSPy modules, signatures, adapters, optimizers, evaluation, retrievers, and streaming behavior compatible.
- `numpy` is optional and belongs behind the `embeddings` extra. Core package paths should keep working without importing `numpy`.
- The package is published as `dspy-lite`, but the runtime package directory remains `dspy/`.
- Keep public project claims aligned with `README.md`, `pyproject.toml`, and `dspy/__metadata__.py` when changing version, dependency, or packaging behavior.

## Guidance Location

- Durable repository guidance belongs in `AGENTS.md`.
- Do not add new canonical instructions to `CLAUDE.md`; if a local `CLAUDE.md` appears, migrate durable content here and remove the Claude-specific file from the tracked work.
- Keep public-facing text limited to this repository's public facts. Do not include private project names, private issue IDs, internal URLs, credentials, or unrelated workflow details in issues, PRs, commits, or docs.

## Dependency Policy

- Prefer exact pins for direct dependencies in `pyproject.toml`.
- Do not use `tool.uv.exclude-newer`; reproducibility should come from explicit pins plus `uv.lock`.
- `litelm` is intentionally sourced through the maintainer-owned git source for local resolution.
- When adding optional functionality, put provider- or feature-specific dependencies behind extras instead of the core dependency list.

## Development Commands

- Check lock consistency: `uv lock --check`
- Lint package code: `uvx ruff check dspy/`
- Run the CI-shaped test suite: `uv run --locked --extra dev --extra embeddings pytest tests/ -x --timeout=30 -q`
- For focused work, run the narrowest relevant pytest target first, then the CI-shaped suite before publishing code changes.

## Areas To Treat Carefully

- `dspy/clients/lm.py` and `dspy/clients/base_lm.py` contain most LM routing, response processing, cache, reasoning, and streaming integration behavior.
- `dspy/clients/__init__.py` configures DSPy cache behavior and suppresses `litelm` logging by default.
- `dspy/retrievers/embeddings.py`, `dspy/clients/embedding.py`, `dspy/predict/knn.py`, and test utilities must keep optional-`numpy` behavior explicit.
- Stale expected-fail tests should be removed once the underlying `litelm` support exists and the behavior is covered by passing tests.
