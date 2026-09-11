#!/usr/bin/env python3
"""Prepare an upstream DSPy synchronization without touching the primary checkout.

The default workflow is intentionally split into read-only analysis, disposable
candidate creation, and explicit verification. This tool never pushes, merges,
tags, publishes, or commits.
"""

import argparse
import ast
import json
import re
import shlex
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

SYNC_PATHS = ("dspy", "tests")
ANALYSIS_PATHS = (*SYNC_PATHS, "pyproject.toml")
SENSITIVE_PATHS = (
    "dspy/clients/lm.py",
    "dspy/clients/base_lm.py",
    "dspy/clients/cache.py",
    "dspy/clients/disk_serialization.py",
    "dspy/adapters/types/image.py",
    "dspy/adapters/types/audio.py",
    "dspy/adapters/types/file.py",
    "pyproject.toml",
)
NUMPY_IMPORT_ALLOWLIST = {
    "dspy/clients/embedding.py",
    "dspy/dsp/colbertv2.py",
    "dspy/predict/knn.py",
    "dspy/retrievers/embeddings.py",
    "dspy/teleprompt/simba.py",
    "dspy/utils/dummies.py",
}
FOCUSED_TESTS = (
    "tests/metadata/test_metadata.py",
    "tests/adapters/test_resource_loading.py",
    "tests/clients/test_cache.py",
    "tests/clients/test_disk_serialization.py",
    "tests/clients/test_lm.py",
)


class SyncError(RuntimeError):
    """An expected safety or synchronization failure."""


@dataclass(frozen=True)
class AnalysisResult:
    from_ref: str
    to_ref: str
    from_sha: str
    to_sha: str
    changed_files: list[str]
    added_files: list[str]
    deleted_files: list[str]
    sensitive_files: list[str]
    litellm_occurrences: list[str]
    numpy_imports: list[str]
    dependency_diff: str
    report_path: Path
    json_path: Path


@dataclass(frozen=True)
class CandidateResult:
    worktree: Path
    branch: str
    conflicts: list[str]
    rewritten_files: list[str]
    report_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    findings: list[Finding]
    commands: list[str]
    report_path: Path

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    text: bool = True,
    input_data: str | bytes | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        input=input_data,
        capture_output=True,
        text=text,
    )
    if check and result.returncode:
        stderr = result.stderr.strip() if text else result.stderr.decode(errors="replace").strip()
        raise SyncError(f"Command failed ({shlex.join(command)}): {stderr}")
    return result


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return _run(("git", *args), cwd=repo, check=check).stdout.strip()


def _repo_root(repo: Path) -> Path:
    return Path(_git(repo.resolve(), "rev-parse", "--show-toplevel")).resolve()


def _resolve_ref(repo: Path, ref: str) -> str:
    result = _run(("git", "rev-parse", "--verify", f"{ref}^{{commit}}"), cwd=repo, check=False)
    if result.returncode:
        raise SyncError(f"Git ref does not resolve to a commit: {ref}")
    return result.stdout.strip()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not slug or slug in {".", ".."}:
        raise SyncError(f"Cannot derive a safe name from ref: {value!r}")
    return slug


def _temporary_roots() -> tuple[Path, ...]:
    roots = {Path(tempfile.gettempdir()).resolve()}
    conventional_tmp = Path("/tmp")
    if conventional_tmp.is_dir():
        roots.add(conventional_tmp.resolve())
    return tuple(sorted(roots))


def _require_temporary_path(path: Path) -> Path:
    resolved = path.resolve()
    roots = _temporary_roots()
    if not any(resolved.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise SyncError(f"Safety check failed: path must be under a temporary root ({allowed}): {resolved}")
    return resolved


def _git_grep(repo: Path, ref: str, pattern: str, *paths: str, extended: bool = False) -> list[str]:
    args = ["git", "grep", "-n", "-I"]
    if extended:
        args.append("-E")
    args.extend((pattern, ref, "--", *paths))
    result = _run(args, cwd=repo, check=False)
    if result.returncode not in (0, 1):
        raise SyncError(f"git grep failed: {result.stderr.strip()}")
    return result.stdout.splitlines()


def _changed_file_data(
    repo: Path,
    from_sha: str,
    to_sha: str,
    paths: Sequence[str] = SYNC_PATHS,
) -> tuple[list[str], list[str], list[str]]:
    output = _git(repo, "diff", "--name-status", "--find-renames", from_sha, to_sha, "--", *paths)
    changed: set[str] = set()
    added: set[str] = set()
    deleted: set[str] = set()
    for line in output.splitlines():
        fields = line.split("\t")
        status = fields[0]
        paths = fields[1:]
        changed.update(paths)
        if status.startswith("A"):
            added.update(paths)
        elif status.startswith("D"):
            deleted.update(paths)
        elif status.startswith("R") and len(paths) == 2:
            deleted.add(paths[0])
            added.add(paths[1])
    return sorted(changed), sorted(added), sorted(deleted)


def analyze(repo: Path, from_ref: str, to_ref: str, output_dir: Path) -> AnalysisResult:
    """Analyze two refs and write reports; do not mutate Git state."""
    repo = _repo_root(repo)
    output_dir = _require_temporary_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from_sha = _resolve_ref(repo, from_ref)
    to_sha = _resolve_ref(repo, to_ref)
    changed, added, deleted = _changed_file_data(repo, from_sha, to_sha, ANALYSIS_PATHS)
    sensitive = sorted(path for path in changed if path in SENSITIVE_PATHS)
    litellm = _git_grep(repo, to_sha, "litellm", "dspy", "pyproject.toml")
    numpy_imports = _git_grep(repo, to_sha, r"^(import numpy|from numpy)", "dspy", extended=True)
    dependency_diff = _git(repo, "diff", "--no-ext-diff", from_sha, to_sha, "--", "pyproject.toml")

    slug = f"{_safe_slug(from_ref)}-to-{_safe_slug(to_ref)}"
    report_path = output_dir / f"analysis-{slug}.md"
    json_path = output_dir / f"analysis-{slug}.json"
    result = AnalysisResult(
        from_ref=from_ref,
        to_ref=to_ref,
        from_sha=from_sha,
        to_sha=to_sha,
        changed_files=changed,
        added_files=added,
        deleted_files=deleted,
        sensitive_files=sensitive,
        litellm_occurrences=litellm,
        numpy_imports=numpy_imports,
        dependency_diff=dependency_diff,
        report_path=report_path,
        json_path=json_path,
    )
    payload = asdict(result)
    payload["report_path"] = str(report_path)
    payload["json_path"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_path.write_text(
        "\n".join(
            (
                f"# Upstream sync analysis: `{from_ref}` → `{to_ref}`",
                "",
                f"- From SHA: `{from_sha}`",
                f"- To SHA: `{to_sha}`",
                f"- Changed files: **{len(changed)}**",
                f"- Added files: **{len(added)}**",
                f"- Deleted files: **{len(deleted)}**",
                f"- Sensitive changed files: **{len(sensitive)}**",
                f"- `litellm` occurrences at target: **{len(litellm)}**",
                f"- direct NumPy imports at target: **{len(numpy_imports)}**",
                "",
                "## Sensitive paths requiring manual review",
                "",
                *(f"- `{path}`" for path in sensitive),
                "",
                "## Provider-boundary occurrences",
                "",
                *(f"- `{line}`" for line in litellm),
                "",
                "## Direct NumPy imports",
                "",
                *(f"- `{line}`" for line in numpy_imports),
                "",
                "## Upstream dependency metadata delta (review only; not automatically applied)",
                "",
                "```diff",
                dependency_diff,
                "```",
                "",
                f"Machine-readable report: `{json_path}`",
                "",
            )
        )
    )
    return result


def rewrite_provider_identifiers(path: Path) -> bool:
    """Rewrite LiteLLM identifiers in Python tokens without touching strings/comments."""
    original = path.read_bytes()
    replacements = (("LiteLLM", "LiteLM"), ("LITELLM", "LITELM"), ("litellm", "litelm"))
    tokens = []
    changed = False
    try:
        token_stream = tokenize.tokenize(iter(original.splitlines(keepends=True)).__next__)
        for token in token_stream:
            if token.type == tokenize.NAME:
                value = token.string
                for old, new in replacements:
                    value = value.replace(old, new)
                if value != token.string:
                    token = tokenize.TokenInfo(token.type, value, token.start, token.end, token.line)
                    changed = True
            tokens.append(token)
    except (SyntaxError, tokenize.TokenError) as exc:
        raise SyncError(f"Cannot tokenize {path}: {exc}") from exc
    if changed:
        path.write_bytes(tokenize.untokenize(tokens))
    return changed


def _metadata_path(worktree: Path) -> Path:
    return worktree.parent / f".{worktree.name}.dspy-sync.json"


def _registered_worktrees(repo: Path) -> set[Path]:
    output = _git(repo, "worktree", "list", "--porcelain")
    return {
        Path(line.removeprefix("worktree ")).resolve()
        for line in output.splitlines()
        if line.startswith("worktree ")
    }


def create_candidate(
    repo: Path,
    *,
    from_ref: str,
    to_ref: str,
    base_ref: str,
    branch: str,
    worktree: Path,
    report_dir: Path,
    max_changed_files: int = 500,
) -> CandidateResult:
    """Create and adapt an isolated worktree candidate. Never commit or push."""
    repo = _repo_root(repo)
    worktree = _require_temporary_path(worktree)
    report_dir = _require_temporary_path(report_dir)
    if worktree.exists():
        raise SyncError(f"Candidate path already exists: {worktree}")
    if worktree.is_relative_to(repo):
        raise SyncError("Candidate worktree must be outside the primary repository")
    if _run(("git", "show-ref", "--verify", f"refs/heads/{branch}"), cwd=repo, check=False).returncode == 0:
        raise SyncError(f"Candidate branch already exists: {branch}")

    from_sha = _resolve_ref(repo, from_ref)
    to_sha = _resolve_ref(repo, to_ref)
    base_sha = _resolve_ref(repo, base_ref)
    changed, _, _ = _changed_file_data(repo, from_sha, to_sha)
    if len(changed) > max_changed_files:
        raise SyncError(
            f"Upstream delta changes {len(changed)} files, exceeding the safety limit of {max_changed_files}; "
            "inspect analyze output and raise --max-changed-files explicitly"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    patch_path = report_dir / f"upstream-{_safe_slug(from_ref)}-to-{_safe_slug(to_ref)}.patch"
    patch = _run(
        ("git", "diff", "--binary", "--full-index", from_sha, to_sha, "--", *SYNC_PATHS),
        cwd=repo,
        text=False,
    ).stdout
    patch_path.write_bytes(patch)

    _git(repo, "worktree", "add", "-b", branch, str(worktree), base_ref)
    metadata_path = _metadata_path(worktree)
    metadata = {
        "schema": 1,
        "repository": str(repo),
        "worktree": str(worktree),
        "branch": branch,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "from_ref": from_ref,
        "from_sha": from_sha,
        "to_ref": to_ref,
        "to_sha": to_sha,
        "patch": str(patch_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if patch:
        apply_result = _run(("git", "apply", "--3way", "--index", str(patch_path)), cwd=worktree, check=False)
        conflicts = _git(worktree, "diff", "--name-only", "--diff-filter=U").splitlines()
        if apply_result.returncode and not conflicts:
            raise SyncError(f"Patch application failed without merge conflicts: {apply_result.stderr.strip()}")
    else:
        conflicts = []

    rewritten: list[str] = []
    conflict_set = set(conflicts)
    for relative in changed:
        path = worktree / relative
        if relative.endswith(".py") and relative not in conflict_set and path.is_file():
            if rewrite_provider_identifiers(path):
                rewritten.append(relative)
    if rewritten:
        _git(worktree, "add", "--", *rewritten)

    dependency_diff = _git(repo, "diff", "--no-ext-diff", from_sha, to_sha, "--", "pyproject.toml")
    report_path = report_dir / f"candidate-{_safe_slug(to_ref)}.md"
    report_path.write_text(
        "\n".join(
            (
                f"# Sync candidate: `{to_ref}`",
                "",
                f"- Worktree: `{worktree}`",
                f"- Branch: `{branch}`",
                f"- Upstream range: `{from_sha}` → `{to_sha}`",
                f"- Patch: `{patch_path}`",
                f"- Unresolved conflicts: **{len(conflicts)}**",
                f"- Token-aware provider rewrites: **{len(rewritten)}**",
                "",
                "## Conflicts requiring manual resolution",
                "",
                *(f"- `{path}`" for path in conflicts),
                "",
                "## Mechanically rewritten Python files",
                "",
                *(f"- `{path}`" for path in rewritten),
                "",
                "## Upstream dependency metadata delta (not applied)",
                "",
                "```diff",
                dependency_diff,
                "```",
                "",
                "No commits, pushes, merges, tags, or releases were performed.",
                "",
            )
        )
    )
    return CandidateResult(worktree, branch, conflicts, rewritten, report_path, metadata_path)


def _python_imports(path: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return [], []
    imports: list[tuple[str, int]] = []
    dynamic_imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module, node.lineno))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            function_name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            module = node.args[0].value
            if function_name in {"import_module", "require"} and isinstance(module, str):
                dynamic_imports.append((module, node.lineno))
    return imports, dynamic_imports


def _function_keyword_default(path: Path, function_name: str, argument_name: str) -> object:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            positional = node.args.posonlyargs + node.args.args
            positional_defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
            pairs = list(zip(positional, positional_defaults, strict=True)) + list(
                zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
            )
            for argument, default in pairs:
                if argument.arg == argument_name and default is not None:
                    return ast.literal_eval(default)
    raise SyncError(f"Could not find a literal default for {function_name}(..., {argument_name}=...) in {path}")


def verify_static(worktree: Path) -> list[Finding]:
    """Check fork invariants without importing project code or using the network."""
    worktree = _repo_root(worktree)
    findings: list[Finding] = []
    conflicts = _git(worktree, "diff", "--name-only", "--diff-filter=U").splitlines()
    findings.extend(Finding("error", "merge-conflict", "Unresolved merge conflict", path) for path in conflicts)

    for path in sorted((worktree / "dspy").rglob("*.py")):
        relative = path.relative_to(worktree).as_posix()
        if relative == "dspy/__metadata__.py":
            continue
        imports, dynamic_imports = _python_imports(path)
        for module, line in imports:
            if module == "litellm" or module.startswith("litellm."):
                findings.append(Finding("error", "litellm-import", f"Forbidden LiteLLM import at line {line}", relative))
            if (module == "numpy" or module.startswith("numpy.")) and relative not in NUMPY_IMPORT_ALLOWLIST:
                findings.append(Finding("error", "numpy-boundary", f"Direct NumPy import outside allowlist at line {line}", relative))
        for module, line in dynamic_imports:
            if module == "litellm" or module.startswith("litellm."):
                findings.append(Finding("error", "litellm-dynamic-import", f"Forbidden dynamic LiteLLM import at line {line}", relative))

    pyproject = worktree / "pyproject.toml"
    if pyproject.is_file():
        # Keep this maintenance tool stdlib-only on Python 3.10, where tomllib is unavailable.
        dependency_pattern = re.compile(
            r"(?im)^[^#\n]*[\"']litellm(?:\[[^\"']+\])?\s*(?:[<>=!~;]|[\"'])"
        )
        pyproject_content = pyproject.read_text()
        for match in dependency_pattern.finditer(pyproject_content):
            line = pyproject_content.count("\n", 0, match.start()) + 1
            findings.append(
                Finding("error", "litellm-dependency", f"Forbidden LiteLLM dependency at line {line}", "pyproject.toml")
            )

    cache_path = worktree / "dspy/clients/cache.py"
    clients_init = worktree / "dspy/clients/__init__.py"
    for path, function in ((cache_path, "__init__"), (clients_init, "configure_cache")):
        relative = path.relative_to(worktree).as_posix()
        if not path.is_file():
            findings.append(Finding("error", "missing-cache-boundary", "Required cache boundary file is missing", relative))
            continue
        try:
            value = _function_keyword_default(path, function, "restrict_pickle")
            if value is not True:
                findings.append(
                    Finding("error", "unsafe-cache-default", f"{function} restrict_pickle default is not True", relative)
                )
        except (SyncError, SyntaxError, ValueError) as exc:
            findings.append(Finding("error", "cache-default-unknown", str(exc), relative))

    required_regressions = {
        "tests/adapters/test_resource_loading.py": ("does_not_read_local_file",),
        "tests/clients/test_cache.py": ("malicious_pickle",),
        "tests/clients/test_lm.py": ("gpt-5.chat", "gpt-5.4"),
    }
    for relative, needles in required_regressions.items():
        path = worktree / relative
        if not path.is_file():
            findings.append(Finding("error", "missing-regression-file", "Required fork regression file is missing", relative))
            continue
        content = path.read_text()
        for needle in needles:
            if needle not in content:
                findings.append(Finding("error", "missing-regression", f"Required regression marker is missing: {needle}", relative))

    diff_check = _run(("git", "diff", "--check"), cwd=worktree, check=False)
    if diff_check.returncode:
        findings.append(Finding("error", "diff-check", diff_check.stdout.strip() or diff_check.stderr.strip()))
    return findings


def verify_candidate(worktree: Path, level: str, report_dir: Path | None = None) -> VerificationResult:
    worktree = _repo_root(worktree)
    findings = verify_static(worktree)
    commands: list[str] = []
    if not any(finding.severity == "error" for finding in findings) and level != "static":
        if level == "focused":
            existing = [path for path in FOCUSED_TESTS if (worktree / path).is_file()]
            command = ["uv", "run", "--locked", "--extra", "dev", "--extra", "embeddings", "pytest", *existing, "-x", "--timeout=30", "-q"]
            command_text = shlex.join(command)
            commands.append(command_text)
            result = _run(command, cwd=worktree, check=False)
            if result.returncode:
                findings.append(Finding("error", "focused-tests", f"Command failed: {command_text}\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"))
        else:
            extras = ["--all-extras"] if level == "all-extras" else ["--extra", "dev", "--extra", "embeddings"]
            commands_to_run = [
                ["uv", "run", "--locked", "--extra", "dev", "ruff", "check", "dspy/", "scripts/"],
                ["uv", "run", "--locked", *extras, "pytest", "tests/", "-x", "--timeout=60", "-q", *( ["--extra"] if level == "all-extras" else [])],
                ["uv", "lock", "--check"],
            ]
            for command in commands_to_run:
                command_text = shlex.join(command)
                commands.append(command_text)
                result = _run(command, cwd=worktree, check=False)
                if result.returncode:
                    findings.append(Finding("error", "verification-command", f"Command failed: {command_text}\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"))
                    break

    output_dir = _require_temporary_path(report_dir or (Path(tempfile.gettempdir()) / "dspy-lite-sync-reports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"verification-{_safe_slug(worktree.name)}.json"
    report_path.write_text(
        json.dumps(
            {
                "worktree": str(worktree),
                "level": level,
                "passed": not any(finding.severity == "error" for finding in findings),
                "commands": commands,
                "findings": [asdict(finding) for finding in findings],
            },
            indent=2,
        )
        + "\n"
    )
    return VerificationResult(findings, commands, report_path)


def cleanup_candidate(
    repo: Path,
    worktree: Path,
    *,
    force: bool = False,
    delete_branch: bool = False,
    force_delete_branch: bool = False,
) -> None:
    """Remove a generated candidate only after validating its external metadata."""
    repo = _repo_root(repo)
    worktree = _require_temporary_path(worktree)
    metadata_path = _metadata_path(worktree)
    if not metadata_path.is_file():
        raise SyncError(f"Candidate metadata is missing; refusing cleanup: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if Path(metadata.get("repository", "")).resolve() != repo or Path(metadata.get("worktree", "")).resolve() != worktree:
        raise SyncError("Candidate metadata does not match the requested repository/worktree")
    if worktree not in _registered_worktrees(repo):
        raise SyncError(f"Path is not a registered worktree for this repository: {worktree}")
    status = _git(worktree, "status", "--porcelain=v1")
    if status and not force:
        raise SyncError("Candidate has uncommitted changes; inspect them or pass --force to discard explicitly")
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(worktree))
    _git(repo, *args)
    metadata_path.unlink()
    if delete_branch:
        delete_flag = "-D" if force_delete_branch else "-d"
        _git(repo, "branch", delete_flag, metadata["branch"])


def _default_output(to_ref: str) -> Path:
    root = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    return root / f"dspy-lite-sync-{_safe_slug(to_ref)}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository checkout (default: current directory)")
    parser.add_argument(
        "--fetch-upstream",
        action="store_true",
        help="Explicitly fetch upstream tags before running (mutates remote refs; never implied)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Read-only comparison and risk report")
    analyze_parser.add_argument("--from", dest="from_ref", required=True)
    analyze_parser.add_argument("--to", dest="to_ref", required=True)
    analyze_parser.add_argument("--output", type=Path)

    candidate_parser = subparsers.add_parser("candidate", help="Create an isolated worktree and apply the upstream delta")
    candidate_parser.add_argument("--from", dest="from_ref", required=True)
    candidate_parser.add_argument("--to", dest="to_ref", required=True)
    candidate_parser.add_argument("--base", default="main")
    candidate_parser.add_argument("--branch")
    candidate_parser.add_argument("--worktree", type=Path)
    candidate_parser.add_argument("--output", type=Path)
    candidate_parser.add_argument(
        "--max-changed-files",
        type=int,
        default=500,
        help="Abort before creating a worktree when the upstream delta exceeds this size (default: 500)",
    )

    verify_parser = subparsers.add_parser("verify", help="Verify fork invariants and optionally run tests")
    verify_parser.add_argument("--worktree", type=Path, required=True)
    verify_parser.add_argument("--level", choices=("static", "focused", "full", "all-extras"), default="focused")
    verify_parser.add_argument("--output", type=Path)

    cleanup_parser = subparsers.add_parser("cleanup", help="Safely remove a generated candidate worktree")
    cleanup_parser.add_argument("--worktree", type=Path, required=True)
    cleanup_parser.add_argument("--force", action="store_true", help="Discard uncommitted candidate changes")
    cleanup_parser.add_argument("--delete-branch", action="store_true")
    cleanup_parser.add_argument("--force-delete-branch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        repo = _repo_root(args.repo)
        if args.fetch_upstream:
            _git(repo, "fetch", "upstream", "--tags")
        if args.command == "analyze":
            output = args.output or _default_output(args.to_ref)
            result = analyze(repo, args.from_ref, args.to_ref, output)
            print(f"Analysis: {result.report_path}")
            print(f"Changed files: {len(result.changed_files)}; sensitive: {len(result.sensitive_files)}")
        elif args.command == "candidate":
            output = args.output or _default_output(args.to_ref)
            worktree = args.worktree or (output / "worktree")
            branch = args.branch or f"sync/candidate-{_safe_slug(args.to_ref)}"
            result = create_candidate(
                repo,
                from_ref=args.from_ref,
                to_ref=args.to_ref,
                base_ref=args.base,
                branch=branch,
                worktree=worktree,
                report_dir=output,
                max_changed_files=args.max_changed_files,
            )
            print(f"Candidate worktree: {result.worktree}")
            print(f"Candidate branch:   {result.branch}")
            print(f"Conflicts:          {len(result.conflicts)}")
            print(f"Report:             {result.report_path}")
            if result.conflicts:
                print("Resolve conflicts manually, then run verify.")
                return 2
        elif args.command == "verify":
            result = verify_candidate(args.worktree, args.level, args.output)
            print(f"Verification report: {result.report_path}")
            for finding in result.findings:
                location = f" ({finding.path})" if finding.path else ""
                print(f"{finding.severity.upper()} {finding.code}{location}: {finding.message}")
            if not result.passed:
                return 1
            print(f"Verification passed at level: {args.level}")
        elif args.command == "cleanup":
            cleanup_candidate(
                repo,
                args.worktree,
                force=args.force,
                delete_branch=args.delete_branch,
                force_delete_branch=args.force_delete_branch,
            )
            print(f"Removed candidate worktree: {args.worktree}")
        return 0
    except SyncError as exc:
        parser.exit(1, f"sync-upstream: error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
