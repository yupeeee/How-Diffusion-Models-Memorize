"""Fresh paper routing requires shared direct measurements, including original V."""
from utils.common.io import atomic_write_json
from utils.experiments.theory import numerical_reduce, evidence_reduce, paper_reduce, paper_plotting
from utils.experiments.theory.paper_contracts import load_paper_inputs
from tests.test_theory_four_stage_figures import four_stage_config as direct_config
from tests.test_theory_paper_pipeline import measured_fixture, mock_supplements


def test_fresh_paper_entry_uses_one_shared_direct_stage(tmp_path, monkeypatch):
    calls = []
    mock_supplements(monkeypatch)
    def measure(root, **options):
        calls.append(options)
        assert options["config"] == direct_config()
        return measured_fixture(tmp_path)
    monkeypatch.setattr(numerical_reduce, "run_precision_analysis", measure)
    monkeypatch.setattr(paper_reduce, "_source_bundle", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Default direct suite invoked the old endpoint-only pipeline")))
    def publish_manifest(stage, **options):
        # Exercise role publication while replacing only the expensive figure
        # exports. The retirement contract still needs the new ownership file.
        atomic_write_json(stage / "figure_manifest.json", {"files": {}, "figures": [], "complete": True})
    monkeypatch.setattr(paper_plotting, "render_paper", publish_manifest)
    output = paper_reduce.run_paper(tmp_path, device="cpu", probe_batch_size=4, **direct_config())
    assert len(calls) == 1 and calls[0]["probe_batch_size"] == 4
    saved, summary, _, frames = load_paper_inputs(output, expected_config=direct_config())
    assert saved["source_analysis"]["mode"] == "preserved_numerical_and_evidence_measurements"
    assert "branch_gap_posterior_response" in frames
    assert summary["counts"]["trajectory_rows"] == len(measured_fixture(tmp_path)["tables"]["trajectory"])
    assert not list((tmp_path / "outputs").glob("*/theory_v2"))
