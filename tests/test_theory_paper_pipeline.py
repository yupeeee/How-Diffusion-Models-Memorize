"""A real tiny-cache paper publication and immutable-source integration test."""

import pandas as pd
import pytest
import torch

from utils.common.io import file_sha256, read_json
from utils.experiments.theory import (
    cache_reader,
    candidate_contracts,
    candidate_reduce,
    candidate_summaries,
    paper_feedback,
    paper_measurements,
    paper_plotting,
    paper_reduce,
    reduce as base_reduce,
    support,
)
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.paper_contracts import (
    load_paper_inputs,
    render_saved_paper,
)
from utils.experiments.theory.paper_registry import paper_registry
from tests.test_theory_candidate_pipeline import protected_snapshot
from tests.test_theory_reduction import CONFIG, _reduce, saved_cache as saved_cache


def _snapshot(root, names=None):
    paths = [root / name for name in names] if names is not None else root.rglob("*")
    return {
        str(path.relative_to(root)): (file_sha256(path), path.stat().st_mtime_ns)
        for path in paths
        if path.is_file()
    }


def _forbidden(*args, **kwargs):
    raise AssertionError("Saved paper operation reopened scientific/raw inputs")


def test_saved_cache_paper_publication_reuse_and_failure_preserve_contracts(
    saved_cache, monkeypatch
):
    """One setup covers the whole role transaction without repeated raw analysis."""
    root = saved_cache["root"]
    protected_before = protected_snapshot(saved_cache)
    base = _reduce(saved_cache, device="cpu")
    source = candidate_reduce.run_candidates(
        root, base_bundle=base, render=False, device="cpu", **CONFIG
    )
    source_manifest = read_json(source / "analysis_manifest.json")
    scientific_names = set(source_manifest["numerical_files"]) | {
        "analysis_manifest.json"
    }
    source_before = _snapshot(source, scientific_names)
    output = paper_reduce.run_paper(
        root, source_analysis=source, source_logs=root / "logs", device="cpu", **CONFIG
    )
    assert output == root / "outputs/sdv1_ddim_g7.5_T3_N3/theory/experiment_S0_N3"
    config, summary, audit, frames = load_paper_inputs(output, expected_config=CONFIG)
    assert summary["complete"] and not audit["blocking"]
    expected_rows = len(saved_cache["selection"].included_indices) * CONFIG["num_seeds"]
    assert summary["counts"] == {
        "initial_rows": expected_rows,
        "terminal_rows": expected_rows,
        "trajectory_rows": expected_rows * CONFIG["num_inference_steps"],
    }
    core = [
        entry for entry in paper_registry() if not entry.get("conditional_terminal")
    ]
    assert len(core) == 12
    assert sum(entry["category"] == "main" for entry in core) == 4
    manifest = read_json(output / "figure_manifest.json")
    for entry in core:
        status = summary["figures"][entry["stem"]]["status"]
        assert status in {"available", "complete", "alias"}
        if status == "alias":
            assert summary["figures"][entry["stem"]]["reason"]
            continue
        for extension, relative in entry["outputs"].items():
            path = output / relative
            assert path.is_file() and path.suffix == "." + extension
            assert file_sha256(path) == manifest["files"][relative]
    terminal = pd.read_csv(output / "terminal.csv")
    assert len(terminal) == expected_rows
    assert terminal.terminal_accounting_status.eq("available").all()
    assert terminal.terminal_accounting_numeric_pass.all()
    assert audit["terminal_supplement"]["records_loaded"] == len(
        saved_cache["selection"].included_indices
    )
    assert set(frames["initial_recovery"].terminal_sscd > 0.75) == {True, False}
    assert protected_snapshot(saved_cache) == protected_before
    assert _snapshot(source, scientific_names) == source_before

    # Plot-only uses compact CSVs even when every scientific entry point fails.
    numerical_before = _snapshot(output, summary["numerical_files"])
    images_before = {
        name: file_sha256(output / name)
        for name in manifest["files"]
        if name.endswith(".png")
    }
    with monkeypatch.context() as patch:
        for module, name in (
            (torch, "load"),
            (pd, "read_parquet"),
            (cache_reader, "load_record"),
            (candidate_contracts, "read_tables"),
            (candidate_reduce, "run_candidates"),
            (candidate_summaries, "build_candidate_summaries"),
            (base_reduce, "run_theory"),
            (support.FiniteSupport, "evaluate"),
            (paper_measurements, "build_nonfeedback_plot_inputs"),
            (paper_measurements, "extend_terminal_accounting"),
            (paper_feedback, "build_feedback_plot_inputs"),
        ):
            patch.setattr(module, name, _forbidden)
        render_saved_paper(output, expected_config=CONFIG)
        # Automatic cache validation must return before any scientific builder.
        patch.setattr(paper_plotting, "render_paper", lambda *args, **kwargs: None)
        assert paper_reduce.run_paper(root, device="cpu", **CONFIG) == output
    assert _snapshot(output, summary["numerical_files"]) == numerical_before
    assert {name: file_sha256(output / name) for name in images_before} == images_before
    assert protected_snapshot(saved_cache) == protected_before

    # An explicit scalar refresh must reuse the independently audited terminal CSV.
    original_extend = paper_measurements.extend_terminal_accounting
    receipts = []

    def counted_extend(*args, **kwargs):
        result, receipt = original_extend(*args, **kwargs)
        receipts.append(receipt)
        return result, receipt

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", _forbidden)
        patch.setattr(cache_reader, "load_record", _forbidden)
        patch.setattr(support.FiniteSupport, "evaluate", _forbidden)
        patch.setattr(paper_measurements, "extend_terminal_accounting", counted_extend)
        # The saved-only renderer was tested above; avoid a third identical export.
        patch.setattr(paper_plotting, "render_paper", lambda *args, **kwargs: None)
        assert (
            paper_reduce.run_paper(
                root,
                source_analysis=source,
                source_logs=root / "logs",
                device="cpu",
                **CONFIG,
            )
            == output
        )
    assert receipts[0]["status"] == "reused_saved_accounting"
    assert receipts[0]["records_loaded"] == 0
    assert (
        load_paper_inputs(output, expected_config=CONFIG)[0]["scientific_hash"]
        == config["scientific_hash"]
    )

    # Both preflight mismatch and post-measurement audit failure preserve the role.
    active_before = _snapshot(output)
    with pytest.raises(
        TheoryError, match="Active paper center/tolerance/configuration differs"
    ):
        paper_reduce.run_paper(
            root, source_analysis=source, device="cpu", center="zero", **CONFIG
        )
    assert _snapshot(output) == active_before
    original_builder = paper_measurements.build_nonfeedback_plot_inputs

    def blocking_builder(*args, **kwargs):
        values, metadata = original_builder(*args, **kwargs)
        metadata["audit"]["blocking"] = True
        metadata["audit"]["injected_reason"] = "integration-test correctness failure"
        return values, metadata

    with monkeypatch.context() as patch:
        patch.setattr(
            paper_measurements, "build_nonfeedback_plot_inputs", blocking_builder
        )
        with pytest.raises(TheoryError, match="correctness audit blocked publication"):
            paper_reduce.run_paper(root, source_analysis=source, device="cpu", **CONFIG)
    assert _snapshot(output) == active_before
    failures = list(
        (output.parent / ".failed-attempts" / output.name).glob("*/audit.json")
    )
    assert len(failures) == 1 and read_json(failures[0])["blocking"]
    assert protected_snapshot(saved_cache) == protected_before
    assert _snapshot(source, scientific_names) == source_before
