"""Analysis-only/plot-only option propagation for the corrected figure contract."""

from __future__ import annotations

import pytest

from tests.test_theory_plot_cli import (
    _run_all_stub,
    _option,
    scalar_bundle as scalar_bundle,
)
from utils.experiments.theory.contracts import numerical_config


@pytest.mark.parametrize("mode", ("--plot", "--recompute-experiments", None))
def test_diagnostics_are_forwarded_only_to_theory(tmp_path, mode):
    args = ([mode] if mode else []) + [
        "--include-diagnostics",
        "--model",
        "sdv2",
        "--scheduler",
        "ddim",
    ]
    result, calls = _run_all_stub(tmp_path, args)
    assert result.returncode == 0, result.stderr
    renders = [
        c
        for c in calls
        if c[0] == "theory_validation.sh" and "--validate-only" not in c
    ]
    assert len(renders) == 1 and "--include-diagnostics" in renders[0]
    assert all(
        "--include-diagnostics" not in c
        for c in calls
        if c[0] != "theory_validation.sh"
    )
    if mode == "--recompute-experiments":
        assert len(calls) == 1


def test_explicit_tolerance_has_analysis_identity_and_reaches_plot_preflight(tmp_path):
    base = dict(model_name="sdv1", scheduler_name="ddim")
    assert "target_error_tolerance" not in numerical_config(**base)
    assert (
        numerical_config(**base, target_error_tolerance=0)["target_error_tolerance"]
        == 0
    )
    result, calls = _run_all_stub(
        tmp_path,
        [
            "--plot",
            "--model",
            "sdv1",
            "--scheduler",
            "ddim",
            "--target-error-tolerance",
            "1.25",
        ],
    )
    assert result.returncode == 0, result.stderr
    for call in calls:
        if call[0] == "theory_validation.sh":
            assert _option(call, "--target-error-tolerance") == "1.25"
        else:
            assert "--target-error-tolerance" not in call


@pytest.mark.parametrize("value", ("-1", "nan", "inf", "nonsense"))
def test_bad_tolerance_fails_before_any_stage(tmp_path, value):
    result, calls = _run_all_stub(
        tmp_path, ["--recompute-experiments", "--target-error-tolerance", value]
    )
    assert result.returncode != 0
    assert calls == []


def test_cli_diagnostics_renderer_receives_saved_scalar_bundle(
    scalar_bundle, monkeypatch
):
    from scripts import theory_validation
    from utils.experiments.theory import plotting

    calls = []
    monkeypatch.setattr(
        plotting,
        "render_bundle",
        lambda bundle, **options: calls.append((bundle, options)),
    )
    assert (
        theory_validation.main(
            ["--bundle", str(scalar_bundle), "--plot", "--include-diagnostics"]
        )
        == 0
    )
    assert calls == [(scalar_bundle.resolve(), {"include_diagnostics": True})]
