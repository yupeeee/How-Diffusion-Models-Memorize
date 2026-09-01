"""Focused output-contract tests for the Webster download entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import download_webster


class _SuccessfulSummary:
    exit_code = 2

    def __str__(self) -> str:
        raise AssertionError("the full preparation summary must never be printed")


def test_success_prints_no_preparation_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def prepare(root: Path, **options: int) -> _SuccessfulSummary:
        assert root == tmp_path
        assert options
        return _SuccessfulSummary()

    monkeypatch.setattr(download_webster, "prepare_webster_dataset", prepare)

    assert download_webster.main(["--root", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_fatal_error_remains_concise_and_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(root: Path, **options: int) -> object:
        del root, options
        raise RuntimeError("synthetic recovery failure")

    monkeypatch.setattr(download_webster, "prepare_webster_dataset", fail)

    assert download_webster.main(["--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Webster recovery failed: RuntimeError: synthetic recovery failure\n"
    )
