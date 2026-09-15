"""Fresh paper entry uses one protected trajectory pass and no path quadrature."""

from pathlib import Path

import pandas as pd

from tests.test_theory_candidate_pipeline import protected_snapshot
from tests.test_theory_reduction import CONFIG, saved_cache as saved_cache
from utils.common.io import file_sha256, read_json
from utils.experiments.theory import (
    candidate_integration,
    candidate_reduce,
    paper_reduce,
    reduce as base_reduce,
)
from utils.experiments.theory.contracts import analysis_parent
from utils.experiments.theory.paper_contracts import load_paper_inputs
from utils.experiments.theory.paper_registry import paper_registry


def _forbidden(*args, **kwargs):
    raise AssertionError(
        "Fresh endpoint paper performed a second trajectory pass or quadrature"
    )


def test_fresh_paper_entry_fuses_endpoints_without_quadrature(saved_cache, monkeypatch):
    root = saved_cache["root"]
    backing_parent = analysis_parent(root, **CONFIG)
    assert not backing_parent.exists()
    assert not list((root / "outputs").glob("*/theory_candidates/*"))
    before = protected_snapshot(saved_cache)
    loaded = []
    original_load = base_reduce.load_record

    def counted_load(paths, record, **options):
        if not options.get("initial_only"):
            loaded.append(record.original_index)
        return original_load(paths, record, **options)

    monkeypatch.setattr(base_reduce, "load_record", counted_load)
    monkeypatch.setattr(candidate_reduce, "load_record", _forbidden)
    monkeypatch.setattr(base_reduce, "proposition5_feedback", _forbidden)
    monkeypatch.setattr(candidate_integration, "integration_metrics", _forbidden)

    # No explicit saved source: this is the normal paper entry from protected inputs.
    output = paper_reduce.run_paper(root, device="cpu", **CONFIG)
    config, summary, audit, frames = load_paper_inputs(output, expected_config=CONFIG)
    assert output == root / "outputs/sdv1_ddim_g7.5_T3_N3/theory/experiment_S0_N3"
    assert sorted(loaded) == sorted(saved_cache["selection"].included_indices)
    assert summary["complete"] and not audit["blocking"]
    assert config["source_analysis"]["mode"] == "protected_cache_analysis"
    source = Path(config["source_analysis"]["path"])
    candidate_manifest = read_json(source / "analysis_manifest.json")
    assert candidate_manifest["endpoint_complete"]
    assert not candidate_manifest["complete"]
    assert candidate_manifest["integration_status"] == "pending"
    assert "integration_workers" not in candidate_manifest["execution"]
    base = Path(candidate_manifest["base_bundle"])
    base_manifest = read_json(base / "manifest.json")
    assert base.parent == backing_parent
    assert base_manifest["defer_path_integration"]
    assert base_manifest["prepare_candidate_endpoints"]
    assert not base_manifest["path_integration_complete"]
    assert base_manifest["candidate_endpoint_extension"]["complete"]
    assert not list((root / "outputs").glob("*/theory_v2"))

    expected_rows = len(saved_cache["selection"].included_indices) * CONFIG["num_seeds"]
    assert summary["counts"] == {
        "initial_rows": expected_rows,
        "terminal_rows": expected_rows,
        "trajectory_rows": expected_rows * CONFIG["num_inference_steps"],
    }
    terminal = pd.read_csv(output / "terminal.csv")
    assert terminal.terminal_accounting_status.eq("available").all()
    assert terminal.terminal_accounting_numeric_pass.all()
    assert set(frames["initial_recovery"].terminal_sscd > 0.75) == {False, True}
    published = read_json(output / "figure_manifest.json")
    for entry in paper_registry():
        metadata = summary["figures"][entry["stem"]]
        if metadata["status"] in {"not_applicable", "alias"}:
            assert metadata["reason"]
            continue
        assert metadata["status"] in {"available", "complete"}
        assert entry["stem"] in frames
        for relative in entry["outputs"].values():
            assert file_sha256(output / relative) == published["files"][relative]
    assert protected_snapshot(saved_cache) == before
