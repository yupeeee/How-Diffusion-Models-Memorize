"""Scalar plotting isolation and protected upstream orchestration contracts."""

from __future__ import annotations

import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from utils.experiments.theory.plotting import (
    KNOWN_STATEMENTS,
    DEFAULT_FIGURES,
    DIAGNOSTIC_FIGURES,
    FIGURE_ORDER,
    build_figure,
    render_bundle,
    trajectory_segments,
    validate_bundle,
)
from utils.experiments.theory.contracts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    TheoryError,
)

ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def scalar_bundle(tmp_path):
    bundle = tmp_path / "copied-scalars"
    (bundle / "plotdata").mkdir(parents=True)
    rows = []
    for index in ("0001", "0012"):
        for seed in (0, 1):
            for step in (0, 1, 2):
                rows.append(
                    dict(
                        run_id="synthetic",
                        original_index=index,
                        record_id=f"r-{index}",
                        target_id=f"target-{index}",
                        seed=seed,
                        step_index=step,
                        include_prompt=True,
                        snr=0.01 * 10**step,
                        unconditional_target_error_rmse=2 + seed + step,
                        conditional_target_error_rmse=0.5 + seed + step,
                        joint_target_error_rmse=2 + seed + step,
                        a_parallel=-1 + seed,
                        off_target=0.4 + step,
                        unconditional_center_rmse=0.3 + step + seed,
                        predicted_shift_rmse=0.8 + step,
                        matched_shift_rmse=0.8 + step,
                        candidate_condition_margin_rmse=-0.7 + step,
                        candidate_log_probability_gain=-0.5 + step,
                        candidate_variation_l2=0.2 + step,
                        candidate_variation_error_l2=0.01,
                        candidate_condition_status=(
                            "estimated_not_met" if step == 0 else "estimated_met"
                        ),
                        candidate_gain_status="negative" if step == 0 else "positive",
                        feedback_eligible=step < 2,
                        feedback_status="available" if step < 2 else "not_applicable",
                        joint_rmse=2 + seed + step,
                        terminal_A_rmse=0.5 + seed,
                        terminal_B_rmse=8 + seed,
                        terminal_clean_applicable=True,
                        terminal_rmse=3 + seed,
                        terminal_sscd=0.4 + seed * 0.7,
                    )
                )
    frame = pd.DataFrame(rows)
    registry = {
        "statements": [
            dict(
                semantic_id=key,
                evidence_classification="finite_noise_observation",
                main_figure=(
                    {"id": key, "evidence_classification": "finite_noise_observation"}
                    if key in DEFAULT_FIGURES
                    else None
                ),
                diagnostic_figure=(
                    {
                        "id": (
                            "terminal_terms" if key == "terminal_reproduction" else key
                        ),
                        "evidence_classification": "finite_noise_observation",
                    }
                    if key in {"target_injection", "terminal_reproduction"}
                    else None
                ),
            )
            for key in sorted(KNOWN_STATEMENTS)
        ]
    }
    (bundle / "statement_registry.json").write_text(json.dumps(registry))
    for key in FIGURE_ORDER:
        data = frame
        if key in {"initial_recovery", "target_injection"}:
            data = frame.loc[frame.step_index.eq(0)]
        elif key == "unconditional_center":
            data = frame.loc[frame.step_index.eq(0)].drop_duplicates(["run_id", "seed"])
        elif key == "terminal_terms":
            data = frame.loc[frame.step_index.eq(2)]
        data.to_parquet(bundle / "plotdata" / f"{key}.parquet", index=False)
    frame.loc[frame.step_index.eq(0)].to_parquet(
        bundle / "initial_metrics.parquet", index=False
    )
    frame.loc[frame.step_index.eq(2)].to_parquet(
        bundle / "endpoint_metrics.parquet", index=False
    )
    frame[["run_id", "original_index", "record_id"]].drop_duplicates().to_parquet(
        bundle / "source_records.parquet", index=False
    )
    frame[["step_index", "snr"]].drop_duplicates().to_csv(
        bundle / "schedule.csv", index=False
    )
    pd.DataFrame(columns=["original_index", "reason"]).to_csv(
        bundle / "failed.csv", index=False
    )
    for name in (
        "center_metadata",
        "support_metadata",
        "summary",
        "applicability_report",
    ):
        (bundle / f"{name}.json").write_text("{}")
    (bundle / "summaries").mkdir()
    for name in (
        "synchronization",
        "paired_synchronization",
        "phases",
        "initial_by_prompt",
        "feedback_by_step",
        "feedback_by_prompt",
    ):
        pd.DataFrame({"fixture": [True]}).to_parquet(
            bundle / "summaries" / f"{name}.parquet", index=False
        )
    manifest = dict(
        complete=True,
        schema_version=SCHEMA_VERSION,
        formula_version=FORMULA_VERSION,
        analysis_hash="f" * 64,
        config=dict(
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=-1.0,
            num_inference_steps=3,
            num_seeds=2,
            center="reference-initial",
        ),
    )
    manifest["numerical_files"] = {
        str(path.relative_to(bundle)): _hash(path)
        for path in bundle.rglob("*")
        if path.is_file()
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


def _numerical_snapshot(bundle):
    return {
        str(p.relative_to(bundle)): (_hash(p), p.stat().st_mtime_ns)
        for p in bundle.rglob("*")
        if p.is_file() and "figures" not in p.parts
    }


def _paper_numerical_snapshot(bundle):
    paths = [bundle / "analysis_config.json", bundle / "analysis_summary.json"]
    paths.extend(path for path in (bundle / "plot_inputs").rglob("*") if path.is_file())
    return {
        path.relative_to(bundle).as_posix(): (_hash(path), path.stat().st_mtime_ns)
        for path in paths
        if path.is_file()
    }


def test_render_inventory_parity_and_immutable_scalar_sources(scalar_bundle):
    before = _numerical_snapshot(scalar_bundle)
    first = render_bundle(scalar_bundle)
    pngs = {p.name: _hash(p) for p in (scalar_bundle / "figures").glob("*.png")}
    second = render_bundle(scalar_bundle)
    assert before == _numerical_snapshot(scalar_bundle)
    assert pngs == {p.name: _hash(p) for p in (scalar_bundle / "figures").glob("*.png")}
    assert first.keys() == second.keys()
    assert set(first["figures"]) == DEFAULT_FIGURES
    assert {p.name for p in (scalar_bundle / "figures").iterdir()} == {
        "manifest.json",
        "applicability.json",
    } | {f"{key}.{ext}" for key in DEFAULT_FIGURES for ext in ("png", "pdf", "json")}
    for key, item in first["figures"].items():
        assert item["data_sha256"] == _hash(
            scalar_bundle / "plotdata" / f"{key}.parquet"
        )
        assert item["evidence_classification"]
        assert item["main_axes_inches"] == [4, 4]
        assert item["excluded_rows"] == (4 if key == "posterior_feedback" else 0)
        assert item["visual_subsampling"] is False
        assert item["xlim"] == second["figures"][key]["xlim"]
        assert item["ylim"] == second["figures"][key]["ylim"]


@pytest.mark.parametrize("mode", ("--plot", "--validate-only"))
def test_copied_scalar_cli_never_imports_tensor_or_model_layers(
    tmp_path, monkeypatch, mode
):
    from scripts import theory_validation
    from tests.test_theory_paper_plotting import compact_fixture

    scalar_bundle = compact_fixture(tmp_path / "paper")

    blocked = (
        "torch",
        "diffusers",
        "transformers",
        "utils.models",
        "utils.data.selection",
        "utils.experiments.cache",
        "utils.experiments.proximity",
        "utils.experiments.theory.reduce",
        "utils.experiments.theory.cache_reader",
        "utils.experiments.theory.support",
        "utils.experiments.theory.feedback",
        "utils.experiments.theory.posterior",
    )

    class BlockImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == p or fullname.startswith(p + ".") for p in blocked):
                raise AssertionError(f"Plot-only imported {fullname}")

    guard = BlockImports()
    removed = {
        key: value
        for key, value in tuple(sys.modules.items())
        if any(key == p or key.startswith(p + ".") for p in blocked)
    }
    for key in removed:
        monkeypatch.delitem(sys.modules, key)
    sys.meta_path.insert(0, guard)
    before = _paper_numerical_snapshot(scalar_bundle)
    try:
        assert theory_validation.main(["--bundle", str(scalar_bundle), mode]) == 0
    finally:
        sys.meta_path.remove(guard)
    assert before == _paper_numerical_snapshot(scalar_bundle)


@pytest.mark.parametrize(
    "broken", ("hash", "schema", "missing", "unhashed_figure", "unsafe")
)
def test_preflight_rejects_all_bad_scalar_bundles_before_figures(scalar_bundle, broken):
    manifest_path = scalar_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if broken == "hash":
        (scalar_bundle / "summary.json").write_text('{"tampered": true}')
    elif broken == "schema":
        manifest["schema_version"] = -1
    elif broken == "missing":
        (scalar_bundle / "plotdata" / "terminal_terms.parquet").unlink()
    elif broken == "unhashed_figure":
        del manifest["numerical_files"]["plotdata/terminal_terms.parquet"]
    elif broken == "unsafe":
        manifest["numerical_files"]["../escape.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TheoryError):
        render_bundle(scalar_bundle)
    assert not (scalar_bundle / "figures").exists()


def test_preflight_rejects_missing_seed_even_with_rehashed_table(scalar_bundle):
    path = scalar_bundle / "plotdata" / "unconditional_center.parquet"
    frame = pd.read_parquet(path)
    frame.loc[~(frame.original_index.eq("0001") & frame.seed.eq(1))].to_parquet(
        path, index=False
    )
    manifest = json.loads((scalar_bundle / "manifest.json").read_text())
    manifest["numerical_files"][str(path.relative_to(scalar_bundle))] = _hash(path)
    (scalar_bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(
        TheoryError, match="seed bank|identities differ|scalar observations"
    ):
        validate_bundle(scalar_bundle)


def test_trajectory_curves_preserve_seed_order_and_missing_steps():
    frame = pd.DataFrame(
        dict(
            run_id=["r"] * 6,
            original_index=["01"] * 6,
            seed=[0, 1, 0, 1, 0, 1],
            step_index=[0, 0, 2, 1, 3, 3],
            snr=[1, 1, 3, 2, 4, 4],
            y=[10, 20, 12, 21, 13, 23],
            terminal_sscd=[0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
        )
    )
    segments = trajectory_segments(frame, "y")
    assert [p[:, 1].tolist() for p, _ in segments] == [[10], [12, 13], [20, 21], [23]]


def test_style_signed_axes_raw_scores_and_center_seed_count(scalar_bundle):
    from matplotlib import pyplot as plt

    manifest = validate_bundle(scalar_bundle)
    entry = {"evidence_classification": "empirical_support_surrogate"}
    posterior = pd.read_parquet(
        scalar_bundle / "plotdata" / "posterior_feedback.parquet"
    )
    fig, audit = build_figure(posterior, "posterior_feedback", entry, manifest)
    try:
        np.testing.assert_equal(fig.get_size_inches(), [4, 4])
        assert audit["xlim"][0] < 0 and audit["ylim"][0] < 0
        assert fig.axes[0].collections[0].norm.vmin == 0
        assert fig.axes[0].collections[0].norm.vmax == 1
        assert (
            posterior.terminal_sscd.max() > 1
        )  # color normalization never alters saved scores
        assert fig.axes[-1].collections[-1].get_alpha() == 1
    finally:
        plt.close(fig)
    center = pd.read_parquet(
        scalar_bundle / "plotdata" / "unconditional_center.parquet"
    )
    fig, audit = build_figure(center, "unconditional_center", entry, manifest)
    try:
        assert audit["initial_effective_seed_count"] == 2
        assert audit["xscale"] == "linear"
        assert audit["xlim"][0] == 0
        assert len(fig.axes) == 1
        assert audit["colorbar_label"] is None
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [0, 0.5, 1])
    finally:
        plt.close(fig)


def _run_all_stub(tmp_path, arguments=(), **environment):
    project = tmp_path / "pipeline"
    project.mkdir()
    runner = project / "run_all.sh"
    runner.write_bytes((ROOT / "run_all.sh").read_bytes())
    fake = """#!/usr/bin/env bash
set -Eeuo pipefail
wrapper_name="$(basename -- "$0")"
{ printf '%s' "$wrapper_name"; printf '\\t%s' "$@"; printf '\\n'; } >> "$RUN_ALL_LOG"
if [[ "$wrapper_name" == download_webster.sh ]]; then exit "$DOWNLOAD_EXIT_CODE"; fi
if [[ "$wrapper_name" == "$FAIL_WRAPPER" && " $* " == *" --model $FAIL_MODEL "* ]]; then exit 17; fi
"""
    for name in (
        "download_webster",
        "generate",
        "sscd",
        "compute_proximity",
        "theory_validation",
    ):
        wrapper = project / f"{name}.sh"
        wrapper.write_text(fake)
        wrapper.chmod(0o755)
    # Stub only the CUDA availability preflight; stage commands remain logged
    # independently. These routing fixtures must never import a numerical stack.
    interpreter = project / "python-preflight"
    interpreter.write_text('#!/bin/sh\ncat >/dev/null\nexit "${CUDA_PREFLIGHT_EXIT_CODE:-0}"\n')
    interpreter.chmod(0o755)
    log = tmp_path / "calls.log"
    result = subprocess.run(
        ["bash", str(runner), *arguments],
        cwd=tmp_path,
        env={
            **os.environ,
            "RUN_ALL_LOG": str(log),
            "PYTHON": str(interpreter),
            "DOWNLOAD_EXIT_CODE": "0",
            "FAIL_WRAPPER": "",
            "FAIL_MODEL": "",
            **environment,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    calls = (
        [line.split("\t") for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, calls


def _option(call, name):
    return call[call.index(name) + 1]


@pytest.mark.parametrize(
    "mode", ("normal", "plot", "overwrite", "download", "recompute-experiments")
)
def test_root_matrix_preserves_upstream_and_reduces_once(tmp_path, mode):
    result, calls = _run_all_stub(
        tmp_path, () if mode == "normal" else (f"--{mode}",), DOWNLOAD_EXIT_CODE="2"
    )
    assert result.returncode == 0, result.stderr
    assert len([c for c in calls if c[0] == "download_webster.sh"]) == (
        mode == "download"
    )
    theory = [
        c
        for c in calls
        if c[0] == "theory_validation.sh" and "--validate-only" not in c
    ]
    assert [(_option(c, "--model"), _option(c, "--scheduler")) for c in theory] == [
        ("sdv1", "ddim"),
        ("sdv1", "ddpm"),
        ("sdv2", "ddim"),
        ("realvis", "ddim"),
    ]
    assert all("--center" not in c for c in theory)  # inherit saved plot settings; analysis owns defaults
    if mode != "plot":
        assert all(_option(c, "--device") == "auto" for c in theory)
    if mode == "recompute-experiments":
        assert len(calls) == 4
        assert all(
            c[0] == "theory_validation.sh" and "--recompute-experiments" in c
            for c in calls
        )
        assert all("--overwrite" not in c for c in calls)
    elif mode == "plot":
        assert len(calls) == 16
        assert all(
            c[0] == "theory_validation.sh"
            and "--validate-only" in c
            and "--validate-proximity" in c
            for c in calls[:4]
        )
        assert all("--device" not in c and "--overwrite" not in c for c in calls)
        assert {c[0] for c in calls} == {"theory_validation.sh", "compute_proximity.sh"}
        assert all("--plot" in c for c in calls[4:])
    else:
        for wrapper in ("generate.sh", "sscd.sh", "compute_proximity.sh"):
            upstream = [c for c in calls if c[0] == wrapper]
            assert len(upstream) == 8
            assert [_option(c, "--seed-start") for c in upstream] == ["20", "0"] * 4
            assert all(
                ("--overwrite" in c)
                == (wrapper == "compute_proximity.sh" or mode == "overwrite")
                for c in upstream
            )
        assert all(
            ("--recompute-experiments" in c) == (mode == "overwrite") for c in theory
        )


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (("--scheduler", "ddpm"), [("sdv1", "ddpm")]),
        (("--model", "sdv1"), [("sdv1", "ddim"), ("sdv1", "ddpm")]),
        (("--model", "sdv2"), [("sdv2", "ddim")]),
        (
            ("--scheduler=ddim",),
            [("sdv1", "ddim"), ("sdv2", "ddim"), ("realvis", "ddim")],
        ),
    ],
)
def test_root_matrix_filters_only_requested_axes(tmp_path, arguments, expected):
    result, calls = _run_all_stub(tmp_path, ("--plot", *arguments))
    assert result.returncode == 0, result.stderr
    plotted = [c for c in calls if c[0] == "theory_validation.sh" and "--plot" in c]
    assert [
        (_option(c, "--model"), _option(c, "--scheduler")) for c in plotted
    ] == expected


@pytest.mark.parametrize(
    "arguments",
    [
        ("--plot", "--download"),
        ("--plot", "--overwrite"),
        ("--plot", "--recompute-experiments"),
        ("--recompute-experiments", "--download"),
        ("--recompute-experiments", "--overwrite"),
        ("--model", "sdv2", "--scheduler", "ddpm"),
        ("--model=realvis", "--scheduler=ddpm"),
        ("--model", "other"),
        ("--scheduler", "other"),
        ("--N", "0"),
        ("--T", "-1"),
        ("--N", str(2**63)),
        ("--g", "nan"),
        ("--g", "1e999"),
        ("--selection-strategy", "spearman"),
        ("--selection-strategy=all",),
        ("--center", "wrong"),
        ("--no-mu", "--use-mu"),
        ("--no-mu",),
        ("--center", "zero"),
        ("--center=zero",),
        ("--use-mu",),
        ("--center", "zero", "--cached-baseline", "/tmp/old"),
        ("--evaluation-source", "trajectory"),
        ("--num-loss-seeds=0",),
        ("--loss-seed", "-1"),
        ("--num-baseline-seeds", "1000"),
        ("--device", "bad"),
        ("--download", "--direct-workers", "65"),
        ("--download", "--direct-workers", "1", "--per-host-concurrency", "2"),
        ("--g",),
    ],
)
def test_root_invalid_options_fail_before_any_stage(tmp_path, arguments):
    result, calls = _run_all_stub(tmp_path, arguments)
    assert result.returncode == 2, result.stderr
    assert not calls


@pytest.mark.parametrize(
    "failed_model, failed_pairs",
    [("sdv1", ("sdv1:ddim", "sdv1:ddpm")),
     ("sdv2", ("sdv2:ddim",)), ("realvis", ("realvis:ddim",))],
)
def test_root_preflight_whole_matrix_precedes_every_plot(tmp_path, failed_model, failed_pairs):
    result, calls = _run_all_stub(
        tmp_path, ("--plot",), FAIL_WRAPPER="theory_validation.sh", FAIL_MODEL=failed_model
    )
    assert result.returncode == 17
    assert len(calls) == 4
    assert all("--validate-only" in c for c in calls)
    assert not any("--plot" in c for c in calls)
    assert [(_option(c, "--model"), _option(c, "--scheduler")) for c in calls] == [
        ("sdv1", "ddim"), ("sdv1", "ddpm"), ("sdv2", "ddim"), ("realvis", "ddim"),
    ]
    assert f"plot preflight failed for {len(failed_pairs)}/4 configurations" in result.stderr
    assert "no figures were changed" in result.stderr
    for pair in failed_pairs:
        assert f"  {pair}\n" in result.stderr
    assert "--model and --scheduler" in result.stderr


@pytest.mark.parametrize(
    "wrapper",
    ("generate.sh", "sscd.sh", "compute_proximity.sh", "theory_validation.sh"),
)
def test_root_stops_after_failed_stage(tmp_path, wrapper):
    result, calls = _run_all_stub(tmp_path, FAIL_WRAPPER=wrapper, FAIL_MODEL="sdv1")
    assert result.returncode == 17
    assert calls[-1][0] == wrapper
    assert all("--model" not in c or _option(c, "--model") == "sdv1" for c in calls)


@pytest.mark.parametrize("exit_code", ("1", "17"))
def test_root_stops_after_failed_download(tmp_path, exit_code):
    result, calls = _run_all_stub(
        tmp_path, ("--download",), DOWNLOAD_EXIT_CODE=exit_code
    )
    assert result.returncode == int(exit_code)
    assert len(calls) == 1 and calls[0][0] == "download_webster.sh"


def test_root_preserves_nondefault_guidance_seeds_and_upstream_overwrite_scope(
    tmp_path,
):
    result, calls = _run_all_stub(
        tmp_path,
        (
            "--model",
            "sdv1",
            "--scheduler",
            "ddim",
            "--g",
            "-2",
            "--N",
            "3",
            "--T",
            "4",
            "--center",
            "reference-initial",
        ),
    )
    assert result.returncode == 0, result.stderr
    assert all(
        _option(c, "--g") == "-2"
        and _option(c, "--N") == "3"
        and _option(c, "--T") == "4"
        for c in calls
    )
    assert [_option(c, "--seed-start") for c in calls if c[0] == "generate.sh"] == [
        "3",
        "0",
    ]
    assert _option(calls[-1], "--center") == "reference-initial"
    assert all("--overwrite" not in c for c in calls if c[0] != "compute_proximity.sh")


def test_wrapper_honors_python_without_model_preflight(tmp_path):
    log = tmp_path / "python.log"
    interpreter = tmp_path / "python-stub"
    interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$PYTHON_LOG"\n')
    interpreter.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "theory_validation.sh"),
            "--bundle",
            "/tmp/scalar",
            "--plot",
        ],
        env={**os.environ, "PYTHON": str(interpreter), "PYTHON_LOG": str(log)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    args = log.read_text().splitlines()
    assert args == [
        "-B",
        str(ROOT / "scripts/theory_validation.py"),
        "--bundle",
        "/tmp/scalar",
        "--plot",
    ]


def test_root_saved_baseline_alias_never_runs_baseline_inference(tmp_path):
    result, calls = _run_all_stub(
        tmp_path,
        (
            "--model",
            "sdv1",
            "--scheduler",
            "ddim",
            "--use-mu",
            "--cached-baseline",
            "/tmp/existing-baseline",
        ),
    )
    assert result.returncode == 0, result.stderr
    assert _option(calls[-1], "--center") == "cached-baseline"
    assert _option(calls[-1], "--cached-baseline") == "/tmp/existing-baseline"
    assert {c[0] for c in calls} == {
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "theory_validation.sh",
    }


def test_normal_cli_runs_paper_and_plot_reuses_compact_inputs(tmp_path, monkeypatch):
    from scripts import theory_validation
    from tests.test_theory_paper_plotting import compact_fixture
    from utils.experiments.theory.paper_contracts import render_saved_paper
    import types

    paper = compact_fixture(tmp_path / "paper")
    called = []

    def run_paper(root, **kwargs):
        called.append(kwargs)
        render_saved_paper(paper)
        return paper

    monkeypatch.setitem(
        sys.modules,
        "utils.experiments.theory.paper_reduce",
        types.SimpleNamespace(run_paper=run_paper),
    )
    before = _paper_numerical_snapshot(paper)
    assert theory_validation.main(["--N", "2"]) == 0
    pngs = {p.relative_to(paper).as_posix(): _hash(p) for p in paper.rglob("*.png")}
    assert pngs
    assert called[0]["recompute"] is False
    assert called[0]["device"] == "auto"
    assert theory_validation.main(["--bundle", str(paper), "--plot"]) == 0
    assert before == _paper_numerical_snapshot(paper)
    assert pngs == {
        p.relative_to(paper).as_posix(): _hash(p) for p in paper.rglob("*.png")
    }


@pytest.fixture
def saved_initial_mean_cli(tmp_path, monkeypatch):
    """Exercise CLI inheritance while replacing measurement and rendering calls."""
    from scripts import theory_validation
    from utils.experiments.theory.contracts import numerical_config
    from utils.experiments.theory import paper_contracts
    import types

    science = numerical_config(
        model_name="sdv1", scheduler_name="ddim", mean_source="reference-initial",
        num_mean_samples=12000, mean_seed=29, reference_snr_decades=5.,
    )
    paper = paper_contracts.PaperPaths.build(tmp_path, **science).output_directory
    paper.mkdir(parents=True)
    (paper / "run_config.json").write_text(json.dumps({"scientific_config": science}))
    calls = []

    def run_paper(root, **kwargs):
        assert root == tmp_path
        calls.append(("analysis", kwargs))
        return paper

    def validate_paper_bundle(bundle, **kwargs):
        assert bundle == paper
        calls.append(("validate", kwargs["expected_config"]))

    def render_saved_paper(bundle, **kwargs):
        assert bundle == paper
        calls.append(("plot", kwargs["expected_config"]))

    monkeypatch.setattr(theory_validation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "utils.experiments.theory.paper_reduce",
                        types.SimpleNamespace(run_paper=run_paper))
    monkeypatch.setattr(paper_contracts, "validate_paper_bundle", validate_paper_bundle)
    monkeypatch.setattr(paper_contracts, "render_saved_paper", render_saved_paper)
    return types.SimpleNamespace(main=theory_validation.main, paper=paper,
                                 science=science, calls=calls)


@pytest.mark.parametrize("arguments, expected_source, count, seed, decades, recompute", [
    ([], "reference-min-snr", 12000, 29, 5., False),
    (["--recompute-experiments"], "reference-min-snr", 12000, 29, 5., True),
    (["--mean-source", "reference-initial"], "reference-initial", 12000, 29, 5., False),
    (["--mean-source", "reference-min-snr", "--num-mean-samples", "16000",
      "--mean-seed", "7", "--reference-snr-decades", "8"],
     "reference-min-snr", 16000, 7, 8., False),
])
def test_analysis_migrates_initial_mean_default_and_preserves_draw_settings(
    saved_initial_mean_cli, arguments, expected_source, count, seed, decades, recompute
):
    case = saved_initial_mean_cli
    before = (case.paper / "run_config.json").read_bytes()
    assert case.main(arguments) == 0
    assert [name for name, _ in case.calls] == ["analysis"]
    settings = case.calls[0][1]
    assert settings["mean_source"] == expected_source
    assert settings["num_mean_samples"] == count
    assert settings["mean_seed"] == seed
    assert settings["reference_snr_decades"] == decades
    assert settings["recompute"] is recompute
    assert settings["center"] == "reference-initial"
    assert (case.paper / "run_config.json").read_bytes() == before


@pytest.mark.parametrize("portable", [False, True])
def test_plot_keeps_saved_initial_mean_instead_of_adopting_analysis_default(
    saved_initial_mean_cli, portable
):
    case = saved_initial_mean_cli
    before = (case.paper / "run_config.json").read_bytes()
    arguments = ["--plot"] + (["--bundle", str(case.paper)] if portable else [])
    assert case.main(arguments) == 0
    assert [name for name, _ in case.calls] == ["validate", "plot"]
    assert all(settings == case.science for _, settings in case.calls)
    assert (case.paper / "run_config.json").read_bytes() == before


def test_plot_rejects_reinterpreting_saved_initial_mean_as_minimum_snr(
    saved_initial_mean_cli, capsys
):
    case = saved_initial_mean_cli
    assert case.main(["--plot", "--mean-source", "reference-min-snr"]) == 1
    assert case.calls == []
    error = capsys.readouterr().err
    assert "Explicit plot settings conflict" in error and "mean_source" in error
    assert "--mean-source reference-min-snr" in error


def test_preflight_rejects_missing_registered_figure(scalar_bundle):
    path = scalar_bundle / "statement_registry.json"
    registry = json.loads(path.read_text())
    registry["statements"].pop()
    path.write_text(json.dumps(registry))
    manifest_path = scalar_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["numerical_files"][path.name] = _hash(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TheoryError, match="inventory"):
        render_bundle(scalar_bundle)
    assert not (scalar_bundle / "figures").exists()


def test_aggregate_curve_does_not_bridge_invalid_steps(scalar_bundle):
    from matplotlib import pyplot as plt

    frame = pd.read_parquet(
        scalar_bundle / "plotdata" / "target_synchronization.parquet"
    )
    for column in ("conditional_target_error_rmse", "unconditional_target_error_rmse"):
        frame.loc[frame.step_index.eq(1), column] = np.nan
    fig, audit = build_figure(
        frame,
        "target_synchronization",
        {"evidence_classification": "finite_noise_observation"},
        validate_bundle(scalar_bundle),
    )
    try:
        curves = [line for line in fig.axes[0].lines if "; " in line.get_label()]
        assert len(curves) == 4
        assert all(np.isnan(line.get_ydata()[1]) for line in curves)
        assert audit["excluded_rows"] == 4
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("device", ("auto", "cpu", "cuda", "cuda:2"))
def test_theory_cli_passes_device_to_reducer_without_resolution(
    tmp_path, monkeypatch, device
):
    from scripts import theory_validation
    import types

    called = []

    def reducer(root, **kwargs):
        called.append(kwargs)
        return tmp_path

    monkeypatch.setitem(
        sys.modules,
        "utils.experiments.theory.paper_reduce",
        types.SimpleNamespace(run_paper=reducer),
    )
    assert theory_validation.main(["--device", device]) == 0
    assert len(called) == 1
    assert called[0]["device"] == device


@pytest.mark.parametrize("device", ("auto", "cuda", "cuda:2"))
def test_root_forwards_selected_device_to_theory_recomputation(tmp_path, device):
    result, calls = _run_all_stub(
        tmp_path,
        (
            "--model",
            "sdv1",
            "--scheduler",
            "ddim",
            "--recompute-experiments",
            "--device",
            device,
        ),
    )
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert calls[0][0] == "theory_validation.sh"
    assert _option(calls[0], "--device") == device


def _rehash_scalar(bundle, relative):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["numerical_files"][relative] = _hash(bundle / relative)
    path.write_text(json.dumps(manifest))


def test_initial_recovery_uses_equal_zero_including_axes(scalar_bundle):
    from matplotlib import pyplot as plt

    frame = pd.read_parquet(scalar_bundle / "plotdata/initial_recovery.parquet")
    frame.loc[0, "conditional_target_error_rmse"] = 19.0
    fig, audit = build_figure(
        frame,
        "initial_recovery",
        {"evidence_classification": "finite_noise_observation"},
        validate_bundle(scalar_bundle),
    )
    try:
        assert audit["xlim"] == audit["ylim"]
        assert audit["xlim"][0] == 0 and audit["xlim"][1] > 19
        assert fig.axes[0].get_aspect() == 1.0
        assert audit["plotted_rows"] == len(frame)
        assert audit["legend_labels"] == ["Equal target errors"]
    finally:
        plt.close(fig)


def test_initial_baseline_needs_no_sscd_and_rejects_pseudoreplication(scalar_bundle):
    from matplotlib import pyplot as plt

    frame = pd.read_parquet(
        scalar_bundle / "plotdata/unconditional_center.parquet"
    ).drop(columns="terminal_sscd")
    manifest = validate_bundle(scalar_bundle)
    fig, audit = build_figure(frame, "unconditional_center", {}, manifest)
    try:
        assert len(fig.axes) == 1
        assert audit["plotted_rows"] == 2
        assert audit["colorbar_label"] is None
    finally:
        plt.close(fig)
    with pytest.raises(TheoryError, match="unique evaluation seed"):
        build_figure(pd.concat([frame, frame]), "unconditional_center", {}, manifest)


def test_feedback_keeps_signed_unresolved_and_underflow_signs(scalar_bundle):
    from matplotlib import pyplot as plt

    frame = pd.read_parquet(scalar_bundle / "plotdata/posterior_feedback.parquet")
    frame.loc[
        0, ["candidate_condition_margin_rmse", "candidate_log_probability_gain"]
    ] = [-2, -100]
    frame.loc[0, "candidate_condition_status"] = "numerically_unresolved"
    frame.loc[1, "candidate_log_probability_gain"] = 0.0
    frame.loc[1, "candidate_gain_status"] = (
        "positive"  # Saturated but strictly ordered odds.
    )
    frame.loc[3, "candidate_condition_margin_rmse"] = np.nan
    fig, audit = build_figure(
        frame, "posterior_feedback", {}, validate_bundle(scalar_bundle)
    )
    try:
        assert audit["xscale"] == "linear" and audit["yscale"] == "symlog"
        assert audit["symlog_linthresh"] == 1e-3
        assert audit["xlim"][0] < -2 and audit["ylim"][0] < -100
        assert len(fig.axes[0].lines) == 2  # Zero lines only; no equality diagonal.
        hollow = fig.axes[0].collections[1]
        assert len(hollow.get_offsets()) == 1 and len(hollow.get_facecolors()) == 0
        assert audit["conditional_gain_sign_counts"]["positive"] == 4
        assert audit["condition_unresolved_count"] == 1
        assert audit["unplottable_count"] == 5
        assert audit["eligible_unplottable_count"] == 1
        assert "candidate distribution" in audit["caption"]
    finally:
        plt.close(fig)


def test_synchronization_keeps_empty_cohort_and_both_wrong_branches(scalar_bundle):
    from matplotlib import pyplot as plt

    frame = pd.read_parquet(scalar_bundle / "plotdata/target_synchronization.parquet")
    frame["terminal_sscd"] = 0.6
    frame["conditional_target_error_rmse"] = 3.0
    frame["unconditional_target_error_rmse"] = 3.0
    frame["joint_target_error_rmse"] = 3.0
    fig, audit = build_figure(
        frame, "target_synchronization", {}, validate_bundle(scalar_bundle)
    )
    try:
        curves = [line for line in fig.axes[0].lines if "; " in line.get_label()]
        assert len(curves) == 2
        assert {line.get_linestyle() for line in curves} == {"-", "--"}
        assert all(np.all(line.get_ydata() == 3.0) for line in curves)
        assert audit["outcome_groups"]["SSCD > 0.75"] == {
            "prompt_count": 0,
            "seed_count": 0,
        }
        assert audit["colorbar_label"] is None and len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_missing_variation_cannot_be_relabelled_as_new_feedback(scalar_bundle):
    relative = "plotdata/posterior_feedback.parquet"
    frame = pd.read_parquet(scalar_bundle / relative)
    frame.drop(columns="candidate_variation_l2").to_parquet(
        scalar_bundle / relative, index=False
    )
    _rehash_scalar(scalar_bundle, relative)
    with pytest.raises(
        TheoryError, match="Equation-15 V.*run_all.sh --recompute-experiments"
    ):
        render_bundle(scalar_bundle)
    assert not (scalar_bundle / "figures").exists()


def test_optional_diagnostics_and_managed_obsolete_archival(scalar_bundle):
    first = render_bundle(scalar_bundle, include_diagnostics=True)
    assert set(first["figures"]) == DEFAULT_FIGURES | DIAGNOSTIC_FIGURES
    figures = scalar_bundle / "figures"
    unrelated = figures / "my-comparison.png"
    unrelated.write_bytes(b"user-owned")
    old = figures / "matched_displacement.pdf"
    old.write_bytes(b"previous-pipeline-plot")
    manifest_path = figures / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["figures"]["matched_displacement"] = {
        "formats": {"pdf": {"path": old.name}}
    }
    manifest_path.write_text(json.dumps(manifest))
    second = render_bundle(scalar_bundle)
    assert set(second["figures"]) == DEFAULT_FIGURES
    assert unrelated.read_bytes() == b"user-owned"
    assert not old.exists()
    assert [
        p.read_bytes() for p in (figures / "archive").rglob("matched_displacement.pdf")
    ] == [b"previous-pipeline-plot"]
    assert not (figures / "terminal_terms.png").exists()
    assert not (figures / "target_injection.pdf").exists()
    assert "matched_displacement" not in second["requested_figures"]


def test_inapplicable_terminal_is_reported_without_substitute(scalar_bundle):
    relative = "plotdata/terminal_terms.parquet"
    frame = pd.read_parquet(scalar_bundle / relative)
    frame["terminal_clean_applicable"] = False
    frame["terminal_clean_status"] = "non-clean final destination"
    frame.to_parquet(scalar_bundle / relative, index=False)
    _rehash_scalar(scalar_bundle, relative)
    result = render_bundle(scalar_bundle, include_diagnostics=True)
    assert "terminal_terms" in result["unavailable_figures"]
    assert result["unavailable_figures"]["terminal_terms"]["status"] == "not_applicable"
    assert "terminal_terms" not in result["figures"]
    assert not (scalar_bundle / "figures/terminal_reproduction.png").exists()
    assert not (scalar_bundle / "figures/terminal_terms.png").exists()


def test_render_failure_preserves_previous_figures_and_manifest(
    scalar_bundle, monkeypatch
):
    from utils.experiments.theory import plotting

    render_bundle(scalar_bundle)
    before = {
        p.name: p.read_bytes()
        for p in (scalar_bundle / "figures").iterdir()
        if p.is_file()
    }
    build = plotting.build_figure

    def fail_later(frame, figure_id, entry, manifest):
        if figure_id == "posterior_feedback":
            raise RuntimeError("synthetic rendering failure")
        return build(frame, figure_id, entry, manifest)

    monkeypatch.setattr(plotting, "build_figure", fail_later)
    with pytest.raises(RuntimeError, match="synthetic rendering failure"):
        render_bundle(scalar_bundle)
    assert before == {
        p.name: p.read_bytes()
        for p in (scalar_bundle / "figures").iterdir()
        if p.is_file()
    }
    assert not list(scalar_bundle.glob(".figure-stage-*"))


def test_unmanaged_destination_is_never_overwritten(scalar_bundle):
    figures = scalar_bundle / "figures"
    figures.mkdir()
    user_image = figures / "initial_recovery.png"
    user_image.write_bytes(b"user-image")
    with pytest.raises(TheoryError, match="unrelated/unmanaged"):
        render_bundle(scalar_bundle)
    assert user_image.read_bytes() == b"user-image"
    assert not (figures / "manifest.json").exists()


def test_recompute_command_preserves_config_and_quotes_paths():
    import shlex
    from utils.experiments.theory.plotting import _recompute_command

    config = {
        "model_name": "sdv1",
        "scheduler_name": "ddim",
        "center": "cached-baseline",
        "cached_baseline": "/tmp/center with 'quotes'; $(false).pt",
        "target_error_tolerance": 0.4,
    }
    tokens = shlex.split(_recompute_command({"config": config}))
    assert tokens[tokens.index("--cached-baseline") + 1] == config["cached_baseline"]
    assert tokens[tokens.index("--center") + 1] == "cached-baseline"
    assert tokens[tokens.index("--target-error-tolerance") + 1] == "0.4"


@pytest.mark.parametrize(
    "figure_id, fields, expected",
    [
        (
            "posterior_feedback",
            {
                "feedback_eligible": [False, False],
                "candidate_condition_status": ["not_applicable", "not_applicable"],
            },
            "not_applicable",
        ),
        (
            "posterior_feedback",
            {
                "feedback_eligible": [False, False],
                "candidate_condition_status": ["unavailable", "not_applicable"],
            },
            "unavailable",
        ),
        (
            "posterior_feedback",
            {
                "feedback_eligible": [True, False],
                "candidate_condition_status": [
                    "numerically_unresolved",
                    "not_applicable",
                ],
            },
            "numerically_unresolved",
        ),
        (
            "terminal_terms",
            {
                "terminal_clean_structural": [False],
                "terminal_clean_applicable": [False],
            },
            "not_applicable",
        ),
        (
            "terminal_terms",
            {"terminal_clean_structural": [True], "terminal_clean_applicable": [False]},
            "numerically_unresolved",
        ),
    ],
)
def test_unavailable_figure_report_preserves_distinct_statuses(
    figure_id, fields, expected
):
    from utils.experiments.theory.plotting import _unavailable_figure_status

    assert (
        _unavailable_figure_status(figure_id, pd.DataFrame(fields))["status"]
        == expected
    )


@pytest.mark.parametrize(
    "guidance, expected", [(0.5, "omitted"), (1.0, "shown"), (7.5, "shown")]
)
def test_terminal_tolerance_region_requires_nonnegative_amplification(
    scalar_bundle, guidance, expected
):
    from matplotlib import pyplot as plt

    manifest = validate_bundle(scalar_bundle)
    manifest["config"]["guidance_scale"] = guidance
    manifest["config"]["target_error_tolerance"] = 2.0
    frame = pd.read_parquet(scalar_bundle / "plotdata/terminal_terms.parquet")
    frame["latent_dimension"] = 4
    frame["terminal_B_rmse"] = (guidance - 1) * 2.0
    fig, audit = build_figure(frame, "terminal_terms", {}, manifest)
    try:
        assert audit["terminal_tolerance_region_status"] == expected
        references = [
            item for item in audit["references"] if item["kind"] == "tolerance"
        ]
        assert bool(references) == (expected == "shown")
        if guidance < 1:
            assert audit["ylim"][0] < frame.terminal_B_rmse.min()
            assert "guidance_below_one" in audit["terminal_tolerance_region_reason"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("figure_id", ["unconditional_center", "target_injection"])
@pytest.mark.parametrize(
    "center,symbol,caption_phrase",
    [
        ("zero", "0", "literal zero vector"),
        (
            "cached-baseline",
            "mu_saved",
            "independently estimated cached model-output center",
        ),
    ],
)
def test_optional_center_formulas_define_the_actual_center(
    scalar_bundle, figure_id, center, symbol, caption_phrase
):
    from matplotlib import pyplot as plt

    manifest = validate_bundle(scalar_bundle)
    manifest["config"]["center"] = center
    manifest["config"]["cached_baseline"] = "/tmp/independent-center.pt"
    frame = pd.read_parquet(scalar_bundle / "plotdata" / f"{figure_id}.parquet")
    fig, audit = build_figure(frame, figure_id, {}, manifest)
    try:
        assert audit["center_symbol"] == symbol
        assert "mu_ref" not in str(audit["formulas"])
        assert caption_phrase in audit["caption"]
        assert symbol in audit["formulas"]["x"]
        if center == "cached-baseline":
            assert audit["center_source"] == "/tmp/independent-center.pt"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_root_rejects_cpu_numerical_execution_before_stages(tmp_path, device):
    result, calls = _run_all_stub(tmp_path, ["--recompute-experiments", "--device", device])
    assert result.returncode != 0 and calls == []
    assert "requires CUDA" in result.stderr


def test_root_cuda_preflight_failure_starts_no_scientific_stage(tmp_path):
    result, calls = _run_all_stub(tmp_path, ["--recompute-experiments"], CUDA_PREFLIGHT_EXIT_CODE="1")
    assert result.returncode != 0 and calls == []


def test_plot_skips_cuda_preflight(tmp_path):
    result, calls = _run_all_stub(tmp_path, ["--plot"], CUDA_PREFLIGHT_EXIT_CODE="1")
    assert result.returncode == 0, result.stderr
    assert calls and all("--plot" in call or "--validate-only" in call for call in calls)
