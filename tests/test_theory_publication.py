"""Replacement publication and immutable-cache contracts for theory analysis."""

from pathlib import Path

import pytest
import pandas as pd

from utils.common.io import atomic_write_json, file_sha256, read_json
from utils.experiments.theory import reduce as reducer
from utils.experiments.theory.contracts import TheoryError, find_analysis_bundle
from utils.experiments.theory.plotting import (
    DEFAULT_FIGURES,
    render_bundle,
    validate_bundle,
)
from tests.test_theory_reduction import (
    CONFIG,
    _forbidden,
    _reduce,
    saved_cache as saved_cache,
)


def files_under(root):
    return {
        p.relative_to(root).as_posix(): (file_sha256(p), p.stat().st_mtime_ns)
        for p in Path(root).rglob("*")
        if p.is_file()
    }


def protected_files(fixture):
    roots = [paths.run_directory for paths in fixture["paths"].values()]
    roots.append(fixture["proximity"])
    return {str(root): files_under(root) for root in roots}


def test_analysis_and_plot_preserve_protected_caches(saved_cache):
    protected = protected_files(saved_cache)
    bundle = _reduce(saved_cache)
    assert protected_files(saved_cache) == protected
    result = render_bundle(bundle)
    assert set(result["figures"]) == DEFAULT_FIGURES
    assert protected_files(saved_cache) == protected


def test_failed_forced_recompute_preserves_valid_result_then_resumes(
    saved_cache, monkeypatch
):
    bundle = _reduce(saved_cache)
    render_bundle(bundle)
    published = files_under(bundle)
    index_path = bundle.parent / "index.json"
    index_before = index_path.read_bytes()
    protected = protected_files(saved_cache)
    selected = sorted(saved_cache["selection"].included_indices)
    original = reducer._record_metrics
    calls = []

    def interrupted(sources, record, *args, **kwargs):
        calls.append(record.original_index)
        if record.original_index == selected[-1]:
            raise RuntimeError("forced replacement interrupted")
        return original(sources, record, *args, **kwargs)

    monkeypatch.setattr(reducer, "_record_metrics", interrupted)
    with pytest.raises(TheoryError, match="no complete manifest"):
        _reduce(saved_cache, recompute=True)
    assert calls == selected
    assert files_under(bundle) == published
    assert index_path.read_bytes() == index_before
    assert find_analysis_bundle(saved_cache["root"], **CONFIG) == bundle
    validate_bundle(bundle)
    stage = bundle.parent / (".recompute-" + bundle.name)
    assert stage.is_dir() and not (stage / "manifest.json").exists()
    shard_before = files_under(stage / "trajectory_metrics")
    assert sum(name.endswith(".parquet") for name in shard_before) == len(selected) - 1
    calls.clear()

    def resumed(sources, record, *args, **kwargs):
        calls.append(record.original_index)
        return original(sources, record, *args, **kwargs)

    monkeypatch.setattr(reducer, "_record_metrics", resumed)
    assert _reduce(saved_cache, recompute=True) == bundle
    assert calls == [selected[-1]]
    assert not stage.exists()
    assert find_analysis_bundle(saved_cache["root"], **CONFIG) == bundle
    validate_bundle(bundle)
    for name, (digest, _) in shard_before.items():
        assert file_sha256(bundle / "trajectory_metrics" / name) == digest
    archived = list((bundle.parent / "archived_analyses").iterdir())
    assert len(archived) == 1
    assert files_under(archived[0]) == published
    assert protected_files(saved_cache) == protected
    # A normal resume of the fully published replacement performs no records.
    monkeypatch.setattr(reducer, "_record_metrics", _forbidden)
    assert _reduce(saved_cache) == bundle


def test_failed_atomic_directory_swap_restores_previous_publication(
    saved_cache, monkeypatch
):
    bundle = _reduce(saved_cache)
    before = files_under(bundle)
    index_path = bundle.parent / "index.json"
    index_before = index_path.read_bytes()
    original_replace = reducer.os.replace

    def fail_stage_swap(source, destination):
        if (
            Path(source).name == ".recompute-" + bundle.name
            and Path(destination) == bundle
        ):
            raise OSError("synthetic directory publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(reducer.os, "replace", fail_stage_swap)
    with pytest.raises(OSError, match="publication failure"):
        _reduce(saved_cache, recompute=True)
    assert files_under(bundle) == before
    assert index_path.read_bytes() == index_before
    validate_bundle(bundle)
    assert find_analysis_bundle(saved_cache["root"], **CONFIG) == bundle


@pytest.mark.parametrize(
    "missing", ["candidate_variation_l2", "candidate_variation_error_l2"]
)
def test_plot_rejects_missing_equation15_columns_with_exact_recompute_command(
    saved_cache, missing
):
    bundle = _reduce(saved_cache)
    relative = "plotdata/posterior_feedback.parquet"
    path = bundle / relative
    pd.read_parquet(path).drop(columns=[missing]).to_parquet(path, index=False)
    manifest = read_json(bundle / "manifest.json")
    # Keep the file's hash valid to exercise schema rejection, not tamper checks.
    manifest["numerical_files"][relative] = file_sha256(path)
    atomic_write_json(bundle / "manifest.json", manifest)
    with pytest.raises(TheoryError) as caught:
        render_bundle(bundle)
    message = str(caught.value)
    assert "projected branch-gap error and directional positive-part V" in message and missing in message
    assert (
        "./run_all.sh --recompute-experiments --model sdv1 --scheduler ddim --g 7.5 --T 3 --N 3"
        in message
    )
    assert not (bundle / "figures").exists()


def test_plot_rejects_old_schema_before_creating_figures(saved_cache):
    bundle = _reduce(saved_cache)
    manifest = read_json(bundle / "manifest.json")
    manifest["schema_version"] -= 1
    atomic_write_json(bundle / "manifest.json", manifest)
    with pytest.raises(TheoryError, match="Old feedback scalars lack projected branch-gap error and directional positive-part V"):
        render_bundle(bundle)
    assert not (bundle / "figures").exists()
