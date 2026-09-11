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
- Lint package and maintenance code: `uv run --locked --extra dev ruff check dspy/ scripts/`
- Run the CI-shaped test suite: `uv run --locked --extra dev --extra embeddings pytest tests/ -x --timeout=30 -q`
- For focused work, run the narrowest relevant pytest target first, then the CI-shaped suite before publishing code changes.

## Safe Upstream Synchronization

Use `scripts/sync_upstream.py` for future stable DSPy imports. It applies an upstream tag-to-tag delta only inside a disposable worktree; it never commits, pushes, merges, tags, publishes, or modifies the primary checkout.

1. Ensure both refs exist locally. Fetching is never implicit; request it explicitly with `--fetch-upstream` before the subcommand if needed.
2. Generate the read-only risk report:

   ```bash
   uv run python scripts/sync_upstream.py analyze --from 3.3.1 --to 3.4.0
   ```

3. Create the isolated candidate:

   ```bash
   uv run python scripts/sync_upstream.py candidate --from 3.3.1 --to 3.4.0
   ```

   The command applies only the `dspy/` and `tests/` delta with `git apply --3way`, performs token-aware Python identifier rewrites from LiteLLM to LiteLM, and reports conflicts without resolving them. It aborts before creating a worktree when more than 500 files change; inspect the analysis before explicitly raising `--max-changed-files`. Exit status 2 means the candidate was created with conflicts requiring manual review.

4. Inspect the printed `/tmp` report and resolve every semantic conflict, especially LM, streaming, cache, resource, provider-capability, and dependency boundaries. A clean Git merge does not waive manual review of these paths.
5. Verify progressively:

   ```bash
   uv run python scripts/sync_upstream.py verify --worktree /tmp/dspy-lite-sync-3.4.0/worktree --level static
   uv run python scripts/sync_upstream.py verify --worktree /tmp/dspy-lite-sync-3.4.0/worktree --level focused
   uv run python scripts/sync_upstream.py verify --worktree /tmp/dspy-lite-sync-3.4.0/worktree --level full
   uv run python scripts/sync_upstream.py verify --worktree /tmp/dspy-lite-sync-3.4.0/worktree --level all-extras
   ```

6. Commit and push only after manually reviewing the candidate diff. Promotion is intentionally ordinary Git, outside the tool.
7. Cleanup refuses dirty worktrees by default. To abandon one deliberately:

   ```bash
   uv run python scripts/sync_upstream.py cleanup --worktree /tmp/dspy-lite-sync-3.4.0/worktree --force
   ```

Run `uv run python scripts/sync_upstream.py --help` and each subcommand's `--help` for path, branch, report, and cleanup options. Keep generated patches/reports under `/tmp`; do not commit them.

## Areas To Treat Carefully

- `dspy/clients/lm.py` and `dspy/clients/base_lm.py` contain most LM routing, response processing, cache, reasoning, and streaming integration behavior.
- `dspy/clients/__init__.py` configures DSPy cache behavior and suppresses `litelm` logging by default.
- `dspy/retrievers/embeddings.py`, `dspy/clients/embedding.py`, `dspy/predict/knn.py`, and test utilities must keep optional-`numpy` behavior explicit.
- Stale expected-fail tests should be removed once the underlying `litelm` support exists and the behavior is covered by passing tests.
