"""Four-stage publication and compact-only rendering contracts.

Fixtures represent independently measured scalars; learned probes are mocked so
these publication regressions never download or evaluate checkpoint weights.
"""
from pathlib import Path

import pandas as pd
import pytest
import torch

from utils.common.io import file_sha256, read_json
from utils.experiments.theory import direct_reduce, direct_figures, numerical_reduce, evidence_reduce, evidence_figures, four_stage_reduce, four_stage_figures, counterfactual_probes, gaussian_controls, paper_reduce, paper_plotting
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.paper_contracts import PaperPaths, load_paper_inputs, render_saved_paper
from utils.experiments.theory.paper_registry import measurement_registry, paper_registry
from tests.test_theory_four_stage_figures import four_stage_tables as direct_tables, four_stage_config as direct_config


def _snapshot(root):
    return {str(path.relative_to(root)): (file_sha256(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def _forbidden(*args, **kwargs):
    raise AssertionError("Saved paper operation reopened scientific/raw inputs")


def measured_fixture(root):
    tables = direct_tables()
    laws = {str(value) for frame in tables.values() if "reference_law_hash" in frame
            for value in frame.reference_law_hash.dropna().unique() if value != "fixed_pair_target"}
    law = next(iter(laws), "fixture-law")
    return {"tables": tables, "files": {}, "directory": root / "measurements",
            "provenance": {"analysis_hash": "fixture-analysis", "reference_law_hash": law,
                           "reference_law": {"reference_law_scope": "synthetic finite law"},
                           "probe_stage": {"complete": True}, "record_audits": {"complete_records": 2}}}


def mock_supplements(monkeypatch):
    # Publication fixtures contain saved scalars only; CUDA routing is checked
    # without allocating device tensors or executing any scientific worker.
    from utils.experiments.theory import reduce as reducer
    monkeypatch.setattr(reducer, "_resolve_theory_devices", lambda request: (torch.device("cuda:0"),))
    monkeypatch.setattr(gaussian_controls, "run_gaussian_control_analysis", lambda *a, result, **k: result)
    monkeypatch.setattr(four_stage_reduce, "prepare_four_stage_primary", lambda *a, **k: None)
    monkeypatch.setattr(four_stage_reduce, "run_four_stage_analysis", lambda *a, result, **k: result)
    monkeypatch.setattr(counterfactual_probes, "run_counterfactual_analysis", lambda *a, result, **k: result)


def test_direct_publication_plot_isolation_and_failure_preserve_active_role(tmp_path, monkeypatch):
    config = direct_config()
    mock_supplements(monkeypatch)
    protected = tmp_path / "logs" / "keep.pt"
    protected.parent.mkdir()
    protected.write_bytes(b"protected source sentinel")
    protected_before = (file_sha256(protected), protected.stat().st_mtime_ns)
    measurements = measured_fixture(tmp_path)
    monkeypatch.setattr(numerical_reduce, "run_precision_analysis", lambda *a, **k: measurements)
    output = paper_reduce.run_paper(tmp_path, device="cuda:0", **config)
    assert output == PaperPaths.build(tmp_path, **config).output_directory
    _, summary, audit, frames = load_paper_inputs(output, expected_config=config)
    assert summary["complete"] and not audit["blocking"]
    main = [entry for entry in paper_registry() if entry["category"] == "main"]
    assert len(main) == 6
    manifest = read_json(output / "figure_manifest.json")
    assert manifest["manuscript_extension_required"] is True
    for entry in main:
        assert summary["figures"][entry["stem"]]["status"] == "available"
        for relative in entry["outputs"].values():
            assert file_sha256(output / relative) == manifest["files"][relative]
    assert set(frames) == {entry["stem"] for entry in main}
    assert "final_reproduction_bound" not in frames
    assert "final_reproduction_bound" in summary["figures"]
    assert "final_reproduction_bound" in summary["plot_data"]
    assert {entry["stem"] for entry in measurement_registry()} <= set(summary["plot_data"])
    image_paths = {path for path in manifest["files"] if Path(path).suffix in {".png", ".pdf"}}
    assert len(image_paths) == 12 and all(Path(path).parent.as_posix() == "figures" for path in image_paths)
    assert "figures/guidance_scale_vs_loss.png" in image_paths
    assert not manifest.get("timestep_figures")
    saved_registry = read_json(output / "registry.json")
    assert len(saved_registry["figures"]) == 6
    assert len(saved_registry["measurement_inventory"]) >= 20
    before = {name: file_sha256(output / name) for name in summary["numerical_files"]}
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", _forbidden)
        patch.setattr(pd, "read_parquet", _forbidden)
        patch.setattr(direct_reduce, "run_direct_analysis", _forbidden)
        patch.setattr(evidence_reduce, "run_evidence_analysis", _forbidden)
        patch.setattr(numerical_reduce, "run_precision_analysis", _forbidden)
        patch.setattr(evidence_figures, "build_evidence_plot_inputs", _forbidden)
        patch.setattr(four_stage_figures, "build_four_stage_plot_inputs", _forbidden)
        patch.setattr(four_stage_reduce, "prepare_four_stage_primary", _forbidden)
        patch.setattr(four_stage_reduce, "run_four_stage_analysis", _forbidden)
        patch.setattr(counterfactual_probes, "run_counterfactual_analysis", _forbidden)
        patch.setattr(gaussian_controls, "run_gaussian_control_analysis", _forbidden)
        patch.setattr(direct_figures, "build_direct_plot_inputs", _forbidden)
        render_saved_paper(output, expected_config=config)
    assert {name: file_sha256(output / name) for name in before} == before
    assert (file_sha256(protected), protected.stat().st_mtime_ns) == protected_before
    active_before = _snapshot(output)
    original = four_stage_figures.build_four_stage_plot_inputs

    def block(*args, **kwargs):
        frames, metadata = original(*args, **kwargs)
        metadata["audit"]["blocking"] = True
        return frames, metadata
    monkeypatch.setattr(four_stage_figures, "build_four_stage_plot_inputs", block)
    with pytest.raises(TheoryError, match="correctness audit blocked publication"):
        paper_reduce.run_paper(tmp_path, device="cuda:0", **config)
    assert _snapshot(output) == active_before
    with pytest.raises(TheoryError, match="scientific configuration differs"):
        paper_reduce.run_paper(tmp_path, device="cuda:0", **{**config, "num_loss_seeds": 128})
    assert _snapshot(output) == active_before


def test_previous_behavioral_bundle_cannot_supply_missing_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(direct_reduce, "run_direct_analysis", _forbidden)
    with pytest.raises(TheoryError, match="cannot supply the new measured losses"):
        paper_reduce.run_paper(tmp_path, source_analysis=tmp_path / "old-candidates", **direct_config())
