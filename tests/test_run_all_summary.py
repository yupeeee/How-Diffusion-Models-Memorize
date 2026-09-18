"""Author-run checks for metadata-only run summaries and shell integration."""

import os
import re
from pathlib import Path
import shutil

import pytest

from scripts import run_all_summary as summary


STAMP = 1_700_000_000_000_000_000


def _artifact(path, content=b"artifact", *, stamp=STAMP):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, ns=(stamp, stamp))
    return path


def test_summary_counts_final_sizes_of_new_and_updated_files_only(tmp_path):
    root = tmp_path / "project"
    unchanged = _artifact(root / "logs/reused.bin", b"reused cache")
    same_size = _artifact(root / "outputs/updated.csv", b"old")
    resized = _artifact(root / "data/resized.bin", b"ab")
    deleted = _artifact(root / "checkpoints/deleted.bin", b"deleted")
    staged_copy = _artifact(root / "figures/preserved.pdf", b"preserved publication")
    state = tmp_path / "before.json"
    summary.write_snapshot(root, state)

    _artifact(root / "figures/new.pdf", b"12345", stamp=STAMP + 1_000)
    _artifact(same_size, b"new", stamp=STAMP + 1_000)
    _artifact(resized, b"1234567", stamp=STAMP)  # Size alone also identifies an update.
    deleted.unlink()
    replacement = tmp_path / "replacement.pdf"
    shutil.copy2(staged_copy, replacement)
    os.replace(replacement, staged_copy)  # New inode/ctime with preserved content+mtime.

    result = summary.report_snapshot(state)

    assert result["produced_bytes"] == 15
    assert result["produced_files"] == 3
    assert result["new_files"] == 1
    assert result["updated_files"] == 2
    assert result["deleted_files"] == 1
    assert result["scope_roots"] == list(summary.ROOT_NAMES)
    assert unchanged.read_bytes() == b"reused cache"
    assert staged_copy.stat().st_mtime_ns == STAMP


def test_summary_skips_links_and_transaction_files_but_includes_archived_artifacts(tmp_path):
    root = tmp_path / "project"
    outside = _artifact(tmp_path / "outside/large.bin", b"outside source")
    (root / "outputs").mkdir(parents=True)
    (root / "logs").symlink_to(outside.parent, target_is_directory=True)
    (root / "outputs/linked.bin").symlink_to(outside)
    (root / "outputs/linked_directory").symlink_to(outside.parent, target_is_directory=True)
    for relative in (
        "outputs/.lock", "outputs/.locks/held", "outputs/.figure_locks/held",
        "outputs/run.lock", "outputs/run.tmp", "outputs/run.transaction.json",
        "outputs/run.pending.json", "outputs/.run.stage-123/partial.csv",
        "figures/.run.backup-123/old.pdf", "data/.run.tmp-123/partial.bin",
    ):
        _artifact(root / relative)
    _artifact(root / "outputs/.archives/previous/scalars.csv", b"archived")
    _artifact(root / "figures/current.pdf", b"current")
    _artifact(root / "untracked/outside_scope.bin", b"not an artifact root")

    files = summary.snapshot_files(root)

    assert set(files) == {"outputs/.archives/previous/scalars.csv", "figures/current.pdf"}
    assert files["outputs/.archives/previous/scalars.csv"] == {"size": 8, "mtime_ns": STAMP}
    assert files["figures/current.pdf"] == {"size": 7, "mtime_ns": STAMP}


def test_snapshot_state_inside_tracked_root_is_excluded_from_accounting(tmp_path):
    root = tmp_path / "project"
    _artifact(root / "outputs/scalars.csv", b"saved")
    state = root / "outputs/run-summary.json"
    summary.write_snapshot(root, state)

    result = summary.report_snapshot(state)

    assert result["produced_files"] == result["produced_bytes"] == 0
    assert result["new_files"] == result["updated_files"] == result["deleted_files"] == 0
    assert "outputs/run-summary.json" not in summary.snapshot_files(root, exclude=(state,))


def test_empty_run_reports_zero_bytes_without_creating_artifact_roots(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    state = tmp_path / "empty-state.json"
    summary.write_snapshot(root, state)

    result = summary.report_snapshot(state)

    assert result["produced_bytes"] == result["produced_files"] == 0
    assert not list(root.iterdir())
    assert summary.format_bytes(0) == "0 B"
    assert summary.format_bytes(1023) == "1023 B"
    assert summary.format_bytes(1024) == "1.00 KiB"


def test_summary_rejects_malformed_state_instead_of_printing_partial_totals(tmp_path, capsys):
    state = tmp_path / "invalid.json"
    state.write_text('{"files":')

    with pytest.raises(summary.SummaryError):
        summary.report_snapshot(state)

    assert capsys.readouterr().out == ""


def test_scan_error_invalidates_whole_report_without_partial_totals(tmp_path, monkeypatch, capsys):
    root = tmp_path / "project"
    _artifact(root / "logs/existing.bin", b"cached")
    state = tmp_path / "before.json"
    summary.write_snapshot(root, state)
    _artifact(root / "figures/new.pdf", b"new output")
    original_scandir = summary.os.scandir

    def deny_logs(directory):
        if Path(directory) == root / "logs":
            raise PermissionError("fixture metadata denied")
        return original_scandir(directory)

    monkeypatch.setattr(summary.os, "scandir", deny_logs)
    with pytest.raises(summary.SummaryError, match="could not be read completely"):
        summary.report_snapshot(state)
    assert summary.main(["report", "--state", str(state)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "run_all summary unavailable" in captured.err
    assert "fixture metadata denied" in captured.err


def test_cli_snapshot_is_silent_and_report_prints_one_total(tmp_path, capsys):
    root = tmp_path / "project"
    root.mkdir()
    state = tmp_path / "before.json"
    assert summary.main(["snapshot", "--project-root", str(root), "--state", str(state)]) == 0
    assert capsys.readouterr().out == ""
    _artifact(root / "outputs/new.csv", b"1234")

    assert summary.main(["report", "--state", str(state)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "Total produced file size: 4 B (4 bytes; 1 new or updated files)\n"
    assert captured.err == ""


@pytest.mark.parametrize("mode", ((), ("--plot",), ("--recompute-experiments",)))
def test_run_all_success_reports_elapsed_time_and_produced_files_once(tmp_path, mode):
    from tests.test_theory_plot_cli import _run_all_stub

    artifact = tmp_path / "pipeline/outputs/created.bin"
    result, calls = _run_all_stub(
        tmp_path, (*mode, "--model", "sdv1", "--scheduler", "ddim"),
        RUN_ALL_ARTIFACT_PATH=str(artifact),
    )

    assert result.returncode == 0, result.stderr
    assert calls
    assert artifact.read_bytes() == b"fixture artifact"
    assert (tmp_path / "summary_calls.log").read_text().splitlines() == ["snapshot", "report"]
    assert result.stdout.count("Experiment matrix complete (1 configurations).") == 1
    assert result.stdout.count("Total elapsed time:") == 1
    assert re.search(r"Total elapsed time: \d{2,}:\d{2}:\d{2} \(\d+ seconds\)", result.stdout)
    assert result.stdout.count(
        "Total produced file size: 16 B (16 bytes; 1 new or updated files)"
    ) == 1
    assert result.stdout.index("Experiment matrix complete") < result.stdout.index("Total elapsed time:")
    assert result.stdout.index("Total elapsed time:") < result.stdout.index("Total produced file size:")


@pytest.mark.parametrize("mode,environment,expected_status", (
    ((), {"FAIL_WRAPPER": "generate.sh", "FAIL_MODEL": "sdv1"}, 17),
    (("--plot",), {"FAIL_WRAPPER": "theory_validation.sh", "FAIL_MODEL": "sdv1"}, 17),
    ((), {"CUDA_PREFLIGHT_EXIT_CODE": "1"}, 1),
))
def test_run_all_failure_preserves_status_and_omits_completion_summary(
    tmp_path, mode, environment, expected_status,
):
    from tests.test_theory_plot_cli import _run_all_stub

    result, _ = _run_all_stub(
        tmp_path, (*mode, "--model", "sdv1", "--scheduler", "ddim"), **environment,
    )

    assert result.returncode == expected_status
    assert (tmp_path / "summary_calls.log").read_text().splitlines() == ["snapshot"]
    assert "Experiment matrix complete" not in result.stdout
    assert "Total elapsed time:" not in result.stdout
    assert "Total produced file size:" not in result.stdout
