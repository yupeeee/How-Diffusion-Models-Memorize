"""Candidate-suite selection and propagation must remain explicit and reproducible."""

import pytest
from tests.test_theory_plot_cli import _run_all_stub, _option


@pytest.mark.parametrize("mode", ["--plot", "--recompute-experiments", None])
@pytest.mark.parametrize("suite", ["main", "candidates"])
def test_suite_reaches_only_theory_and_every_preflight(tmp_path, mode, suite):
    args = ([mode] if mode else []) + ["--figure-suite", suite]
    result, calls = _run_all_stub(tmp_path, args)
    assert result.returncode == 0, result.stderr
    theory = [c for c in calls if c[0] == "theory_validation.sh"]
    assert len(theory) == (8 if mode == "--plot" else 4)
    assert all(_option(c, "--figure-suite") == suite for c in theory)
    assert all(
        "--figure-suite" not in c for c in calls if c[0] != "theory_validation.sh"
    )
    if mode == "--recompute-experiments":
        assert len(calls) == 4


def test_omitted_suite_remains_omitted_for_saved_selection(tmp_path):
    result, calls = _run_all_stub(tmp_path, ["--plot"])
    assert result.returncode == 0, result.stderr
    assert all("--figure-suite" not in c for c in calls)


def test_invalid_suite_fails_before_any_stage(tmp_path):
    result, calls = _run_all_stub(tmp_path, ["--plot", "--figure-suite", "best"])
    assert result.returncode != 0
    assert calls == []


def test_cli_omitted_suite_loads_exact_saved_candidate_bundle(tmp_path, monkeypatch):
    import json
    from scripts import theory_validation
    from utils.common.io import atomic_write_json, file_sha256
    from utils.experiments.theory import candidate_contracts, candidate_plotting
    from utils.experiments.theory import candidate_reduce, reduce
    from utils.experiments.theory.contracts import numerical_config
    import torch

    config = numerical_config(
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=12,
        num_seeds=3,
    )
    bundle = candidate_contracts.candidate_parent(tmp_path, **config) / ("a" * 64)
    bundle.mkdir(parents=True)
    names = [
        "initial.parquet",
        "endpoint.parquet",
        "reference_initial.parquet",
        "reference_snr.parquet",
        "bank_geometry.parquet",
        "registry.json",
    ]
    # This test isolates routing/validation; actual scalar rendering has its own
    # all-design fixture. The routing layer checks these declared bytes only.
    for name in names:
        (bundle / name).write_text("{}")
    atomic_write_json(
        bundle / "analysis_manifest.json",
        {
            "schema_version": 1,
            "analysis_hash": bundle.name,
            "endpoint_complete": True,
            "complete": True,
            "config": config,
            "source_metadata_files": {},
            "source_marker_inventory": {},
            "numerical_files": {name: file_sha256(bundle / name) for name in names},
        },
    )
    candidate_contracts.publish_candidate_index(tmp_path, config, bundle)
    monkeypatch.setattr(theory_validation, "PROJECT_ROOT", tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Plot-only invoked scientific computation or a tensor loader"
        )

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(candidate_reduce, "run_candidates", forbidden)
    monkeypatch.setattr(reduce, "run_theory", forbidden)
    calls = []
    monkeypatch.setattr(
        candidate_plotting, "render_candidates", lambda b: calls.append(b)
    )
    monkeypatch.setattr(candidate_plotting, "render_candidate_index", lambda *a: None)
    assert theory_validation.main(["--plot", "--T", "12", "--N", "3"]) == 0
    assert calls == [bundle]
    saved = json.loads((bundle.parent.parent / "theory_suite.json").read_text())
    assert saved["runs"][0]["figure_suite"] == "candidates"
