import subprocess
from pathlib import Path

import pytest

from scripts.sync_upstream import (
    SyncError,
    analyze,
    cleanup_candidate,
    create_candidate,
    rewrite_provider_identifiers,
    verify_static,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_repo(tmp_path: Path, *, conflict: bool = False) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "tests@example.com")
    git(repo, "config", "user.name", "Tests")
    (repo / "dspy").mkdir()
    (repo / "tests").mkdir()
    (repo / "dspy" / "base.py").write_text("VALUE = 'old'\n")
    (repo / "tests" / "test_base.py").write_text("def test_old():\n    assert True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "old upstream")
    git(repo, "tag", "1.0")

    git(repo, "switch", "-c", "upstream-next")
    (repo / "dspy" / "base.py").write_text("VALUE = 'upstream'\n")
    (repo / "dspy" / "provider.py").write_text(
        "import litellm\n\n# litellm in comments stays\nLABEL = 'litellm in strings stays'\n\ndef get_litellm():\n    return litellm\n"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", "new upstream")
    git(repo, "tag", "2.0")

    git(repo, "switch", "main")
    (repo / "FORK_MARKER").write_text("preserve me\n")
    if conflict:
        (repo / "dspy" / "base.py").write_text("VALUE = 'fork'\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "fork delta")
    return repo


def test_rewrite_provider_identifiers_only_changes_python_names(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text(
        "import litellm\n"
        "from litellm.types import LiteLLMResponse\n"
        "# litellm comment\n"
        "message = 'litellm string'\n"
        "value = get_litellm()\n"
    )

    assert rewrite_provider_identifiers(source)
    assert source.read_text() == (
        "import litelm\n"
        "from litelm.types import LiteLMResponse\n"
        "# litellm comment\n"
        "message = 'litellm string'\n"
        "value = get_litelm()\n"
    )


def test_analyze_is_read_only_and_writes_report_under_requested_directory(tmp_path):
    repo = make_repo(tmp_path)
    before_status = git(repo, "status", "--porcelain=v1")
    before_refs = git(repo, "show-ref")
    output = tmp_path / "reports"

    result = analyze(repo, "1.0", "2.0", output)

    assert result.from_sha == git(repo, "rev-parse", "1.0^{commit}")
    assert result.to_sha == git(repo, "rev-parse", "2.0^{commit}")
    assert "dspy/provider.py" in result.changed_files
    assert result.report_path.is_file()
    assert result.json_path.is_file()
    assert git(repo, "status", "--porcelain=v1") == before_status
    assert git(repo, "show-ref") == before_refs


def test_candidate_is_isolated_preserves_fork_files_and_runs_token_codemod(tmp_path):
    repo = make_repo(tmp_path)
    worktree = tmp_path / "candidate"

    result = create_candidate(
        repo,
        from_ref="1.0",
        to_ref="2.0",
        base_ref="main",
        branch="sync/candidate-2.0",
        worktree=worktree,
        report_dir=tmp_path / "reports",
    )

    assert result.conflicts == []
    assert (worktree / "FORK_MARKER").read_text() == "preserve me\n"
    provider = (worktree / "dspy" / "provider.py").read_text()
    assert "import litelm" in provider
    assert "get_litelm" in provider
    assert "'litellm in strings stays'" in provider
    assert not (repo / "dspy" / "provider.py").exists()
    assert git(repo, "branch", "--show-current") == "main"


def test_candidate_refuses_unexpectedly_large_delta_before_creating_worktree(tmp_path):
    repo = make_repo(tmp_path)
    worktree = tmp_path / "candidate"

    with pytest.raises(SyncError, match="exceeding the safety limit"):
        create_candidate(
            repo,
            from_ref="1.0",
            to_ref="2.0",
            base_ref="main",
            branch="sync/candidate-2.0",
            worktree=worktree,
            report_dir=tmp_path / "reports",
            max_changed_files=1,
        )

    assert not worktree.exists()
    assert "sync/candidate-2.0" not in git(repo, "branch", "--list")


def test_candidate_preserves_semantic_conflicts_for_manual_resolution(tmp_path):
    repo = make_repo(tmp_path, conflict=True)
    worktree = tmp_path / "candidate"

    result = create_candidate(
        repo,
        from_ref="1.0",
        to_ref="2.0",
        base_ref="main",
        branch="sync/candidate-2.0",
        worktree=worktree,
        report_dir=tmp_path / "reports",
    )

    assert result.conflicts == ["dspy/base.py"]
    assert "<<<<<<<" in (worktree / "dspy" / "base.py").read_text()


def test_static_verification_accepts_current_fork_invariants():
    repo = Path(__file__).resolve().parents[2]

    assert verify_static(repo) == []


def test_static_verification_detects_dynamic_litellm_import(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "dspy" / "dynamic.py").write_text("import importlib\nimportlib.import_module('litellm.types')\n")

    findings = verify_static(repo)

    assert any(finding.code == "litellm-dynamic-import" for finding in findings)


def test_cleanup_refuses_dirty_candidate_without_force(tmp_path):
    repo = make_repo(tmp_path)
    worktree = tmp_path / "candidate"
    create_candidate(
        repo,
        from_ref="1.0",
        to_ref="2.0",
        base_ref="main",
        branch="sync/candidate-2.0",
        worktree=worktree,
        report_dir=tmp_path / "reports",
    )

    with pytest.raises(SyncError, match="uncommitted changes"):
        cleanup_candidate(repo, worktree)

    cleanup_candidate(repo, worktree, force=True)
    assert not worktree.exists()
    assert "candidate" not in git(repo, "worktree", "list", "--porcelain")
