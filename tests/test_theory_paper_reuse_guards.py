"""Saved-paper reuse must honor scientific provenance and required figure inputs."""

from pathlib import Path

import pytest

from tests.test_theory_paper_plotting import compact_fixture
from utils.common.io import atomic_write_json, canonical_hash, file_sha256, read_json
from utils.experiments.theory import (
    candidate_contracts,
    candidate_reduce,
    paper_contracts,
    paper_plotting,
    paper_reduce,
    reduce as base_reduce,
)
from utils.experiments.theory.contracts import TheoryError, numerical_config
from utils.experiments.theory.paper_registry import paper_registry


CONFIG = numerical_config(model_name="sdv1", scheduler_name="ddim", num_seeds=2)


def _source_manifest():
    source_directory = Path(paper_reduce.__file__).parent
    return {
        "complete": True,
        "source_code": {
            name: file_sha256(source_directory / name)
            for name in paper_reduce.ENDPOINT_SOURCES | {"candidate_integration.py"}
        },
    }


def _forbidden(*args, **kwargs):
    raise AssertionError(
        "Saved endpoint reuse unexpectedly started scientific analysis"
    )


@pytest.mark.parametrize("missing", [None, "all", "candidate_feedback.py"])
def test_automatic_source_requires_complete_endpoint_provenance(missing):
    manifest = _source_manifest()
    assert paper_reduce._source_recipe_current(manifest)
    assert paper_reduce._source_recipe_current(manifest, diagnostics=True)
    if missing is None:
        del manifest["source_code"]
    elif missing == "all":
        manifest["source_code"] = {}
    else:
        del manifest["source_code"][missing]
    assert not paper_reduce._source_recipe_current(manifest)
    assert not paper_reduce._source_recipe_current(manifest, diagnostics=True)


@pytest.mark.parametrize("integration_state", ["changed", "missing"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_source_selection_only_requires_current_integration_for_diagnostics(
    tmp_path, monkeypatch, integration_state, diagnostics
):
    saved, rebuilt, base = (tmp_path / name for name in ("saved", "rebuilt", "base"))
    current = _source_manifest()
    stale = _source_manifest()
    if integration_state == "changed":
        stale["source_code"]["candidate_integration.py"] = "0" * 64
    else:
        del stale["source_code"]["candidate_integration.py"]
    monkeypatch.setattr(
        candidate_contracts, "find_candidate_bundle", lambda *a, **k: saved
    )
    monkeypatch.setattr(
        candidate_contracts, "validate_candidate_sources", lambda *a: None
    )
    monkeypatch.setattr(
        candidate_contracts,
        "validate_candidate_bundle",
        lambda path, **kwargs: stale if path == saved else current,
    )
    calls = []

    def rebuild_base(root, **options):
        assert root == tmp_path
        assert options["defer_path_integration"] is True
        assert options["prepare_candidate_endpoints"] is True
        calls.append("base")
        return base

    def rebuild_candidates(root, **options):
        assert root == tmp_path
        assert options["base_bundle"] == base
        assert options["endpoint_only"] is False
        assert options["render"] is False
        calls.append("candidates")
        return rebuilt

    monkeypatch.setattr(
        base_reduce, "run_theory", rebuild_base if diagnostics else _forbidden
    )
    monkeypatch.setattr(
        candidate_reduce,
        "run_candidates",
        rebuild_candidates if diagnostics else _forbidden,
    )
    bundle, manifest, mode = paper_reduce._source_bundle(
        tmp_path, CONFIG, source_analysis=None, diagnostics=diagnostics, device="cpu"
    )
    if diagnostics:
        assert (bundle, manifest, mode) == (
            rebuilt,
            current,
            "protected_cache_analysis",
        )
        assert calls == ["base", "candidates"]
    else:
        assert (bundle, manifest, mode) == (saved, stale, "validated_current_analysis")
        assert not calls


def test_direct_integral_recipe_is_required_without_diagnostics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from utils.experiments.theory import cache_reader, direct_reduce, four_stage_reduce
    # The precision wrapper validates source metadata before consulting the
    # existing resumable direct stage; this fixture isolates that routing.
    monkeypatch.setattr(cache_reader, "discover_sources", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(four_stage_reduce, "prepare_four_stage_primary", lambda *a, **k: None)
    from tests.test_theory_direct_figures import direct_config
    calls = []
    def require_direct(*args, **options):
        calls.append(options)
        raise TheoryError("original variation integration required")
    monkeypatch.setattr(direct_reduce, "run_direct_analysis", require_direct)
    monkeypatch.setattr(paper_plotting, "render_paper", _forbidden)
    with pytest.raises(TheoryError, match="original variation integration required"):
        paper_reduce.run_paper(tmp_path, diagnostics=False, **direct_config())
    assert len(calls) == 1


@pytest.mark.parametrize("category", ["main", "appendix"])
def test_preflight_blocks_unavailable_required_figures(tmp_path, category):
    bundle = compact_fixture(tmp_path / "paper")
    entry = next(e for e in paper_registry() if e["category"] == category and not e.get("allow_unavailable", False))
    summary = read_json(bundle / "summary.json")
    summary["figures"][entry["stem"]] = {
        "status": "unavailable",
        "reason": "missing_saved_measurements",
    }
    atomic_write_json(bundle / "summary.json", summary)
    with pytest.raises(TheoryError, match="Required paper figure.*unavailable"):
        paper_contracts.load_paper_inputs(bundle)


def test_preflight_allows_named_inapplicability_and_optional_unavailable(tmp_path):
    bundle = compact_fixture(tmp_path / "paper", diagnostics=True)
    entries = paper_registry(diagnostics=True)
    core = next(e for e in entries if e["category"] == "main")
    optional = next(e for e in entries if e["category"] == "diagnostics" and e.get("allow_unavailable", True))
    summary = read_json(bundle / "summary.json")
    summary["figures"][core["stem"]] = {
        "status": "not_applicable",
        "reason": "no_positive_destination_noise",
    }
    summary["figures"][optional["stem"]] = {
        "status": "unavailable",
        "reason": "path_integration_not_requested",
    }
    atomic_write_json(bundle / "summary.json", summary)
    _, loaded, _, frames = paper_contracts.load_paper_inputs(bundle, diagnostics=True)
    assert core["stem"] not in frames
    assert optional["stem"] not in frames
    assert loaded["figures"][core["stem"]]["reason"] == "no_positive_destination_noise"


def test_preflight_requires_registry_columns_even_with_consistent_saved_schema(
    tmp_path,
):
    bundle = compact_fixture(tmp_path / "paper")
    stem = "posterior_feedback_over_time"
    summary = read_json(bundle / "summary.json")
    spec = summary["plot_data"][stem]
    path = bundle / spec["path"]
    frame = paper_contracts.read_plot_table(path, spec).drop(columns="denominator_weight")
    replacement = paper_contracts.write_plot_table(frame, path)
    summary["plot_data"][stem] = {"path": spec["path"], **replacement}
    summary["numerical_files"][spec["path"]] = replacement["sha256"]
    atomic_write_json(bundle / "summary.json", summary)
    with pytest.raises(
        TheoryError, match="Missing required paper columns.*denominator_weight"
    ):
        paper_contracts.load_paper_inputs(bundle)


def test_preflight_rejects_noncanonical_table_even_with_valid_contents(tmp_path):
    bundle = compact_fixture(tmp_path / "paper")
    stem = "posterior_feedback_over_time"
    summary = read_json(bundle / "summary.json")
    spec = summary["plot_data"][stem]
    copied = "plot_data/copy.csv"
    (bundle / copied).write_bytes((bundle / spec["path"]).read_bytes())
    spec["path"] = copied
    atomic_write_json(bundle / "summary.json", summary)
    with pytest.raises(TheoryError, match="Noncanonical compact paper input"):
        paper_contracts.load_paper_inputs(bundle)
