"""Whole-suite stage, population, provenance and plot-only regression fixtures."""

import pytest

from utils.common.io import file_sha256, read_json
from utils.experiments.theory import candidate_reduce as discovery
from utils.experiments.theory.candidate_contracts import (
    find_candidate_bundle,
    read_tables,
    remember_suite,
    selected_suite,
    validate_candidate_bundle,
)
from tests.test_theory_reduction import CONFIG, _reduce, saved_cache as saved_cache


def protected_snapshot(fixture):
    roots = [p.run_directory for p in fixture["paths"].values()] + [
        fixture["proximity"]
    ]
    # Include the frozen selection alongside raw generation/SSCD/proximity.
    roots += [fixture["root"] / "data"]
    return {
        str(p): (file_sha256(p), p.stat().st_mtime_ns)
        for root in roots
        if root.exists()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_candidate_endpoint_then_integration_reuses_one_raw_pass(
    saved_cache, monkeypatch
):
    base = _reduce(saved_cache)
    before = protected_snapshot(saved_cache)
    main_before = {str(p): file_sha256(p) for p in base.rglob("*") if p.is_file()}
    original = discovery.load_record
    calls = []

    def counted(paths, record, **kwargs):
        calls.append(record.original_index)
        return original(paths, record, **kwargs)

    monkeypatch.setattr(discovery, "load_record", counted)
    endpoint = discovery.run_candidates(
        saved_cache["root"],
        base_bundle=base,
        render=False,
        endpoint_only=True,
        device="cpu",
        **CONFIG,
    )
    endpoint_manifest = validate_candidate_bundle(endpoint)
    assert endpoint_manifest["endpoint_complete"] and not endpoint_manifest["complete"]
    assert sorted(calls) == sorted(saved_cache["selection"].included_indices)
    assert selected_suite(saved_cache["root"], CONFIG) == "main"

    def forbidden(*args, **kwargs):
        raise AssertionError("Integration/resume reread protected trajectories")

    monkeypatch.setattr(discovery, "load_record", forbidden)
    complete = discovery.run_candidates(
        saved_cache["root"], base_bundle=base, render=False, device="cpu", **CONFIG
    )
    assert complete == endpoint
    manifest = validate_candidate_bundle(complete, require_complete=True)
    assert manifest["complete"] and manifest["integration_status"] == "complete"
    assert find_candidate_bundle(saved_cache["root"], **CONFIG) == complete
    assert selected_suite(saved_cache["root"], CONFIG) == "candidates"
    assert selected_suite(saved_cache["root"], CONFIG, "main") == "main"
    data = read_tables(complete)
    assert (
        len(data["initial"])
        == len(saved_cache["selection"].included_indices) * CONFIG["num_seeds"]
    )
    assert (
        len(data["trajectory"]) == len(data["initial"]) * CONFIG["num_inference_steps"]
    )
    assert (
        not data["trajectory"]
        .duplicated(["run_id", "original_index", "target_id", "seed", "step_index"])
        .any()
    )
    assert data["reference_initial"].seed.nunique() == CONFIG["num_seeds"]
    assert data["reference_snr"].snr_grid_index.nunique() == 49
    assert data["dose"]["lambda"].map(len).eq(22).all()
    assert protected_snapshot(saved_cache) == before
    assert {
        str(p): file_sha256(p) for p in base.rglob("*") if p.is_file()
    } == main_before
    assert (
        discovery.run_candidates(
            saved_cache["root"], base_bundle=base, render=False, device="cpu", **CONFIG
        )
        == complete
    )


def test_failed_candidate_stage_preserves_main_and_resumes(saved_cache, monkeypatch):
    base = _reduce(saved_cache)
    before = protected_snapshot(saved_cache)
    original = discovery._endpoint_record
    selected = sorted(saved_cache["selection"].included_indices)

    def interrupted(sources, record, *args, **kwargs):
        if record.original_index == selected[-1]:
            raise RuntimeError("synthetic candidate interruption")
        return original(sources, record, *args, **kwargs)

    monkeypatch.setattr(discovery, "_endpoint_record", interrupted)
    with pytest.raises(Exception, match="candidate endpoints records failed"):
        discovery.run_candidates(
            saved_cache["root"], base_bundle=base, render=False, device="cpu", **CONFIG
        )
    assert selected_suite(saved_cache["root"], CONFIG) == "main"
    assert read_json(base / "manifest.json")["complete"]
    calls = []

    def resumed(sources, record, *args, **kwargs):
        calls.append(record.original_index)
        return original(sources, record, *args, **kwargs)

    monkeypatch.setattr(discovery, "_endpoint_record", resumed)
    result = discovery.run_candidates(
        saved_cache["root"], base_bundle=base, render=False, device="cpu", **CONFIG
    )
    assert calls == [selected[-1]]
    assert read_json(result / "analysis_manifest.json")["complete"]
    assert protected_snapshot(saved_cache) == before


def test_suite_selection_is_scoped_to_exact_configuration(tmp_path):
    bundle = tmp_path / "outputs" / "synthetic" / "main"
    bundle.mkdir(parents=True)
    assert selected_suite(tmp_path, CONFIG) == "main"
    remember_suite(tmp_path, CONFIG, "candidates", bundle)
    assert selected_suite(tmp_path, CONFIG) == "candidates"
    assert selected_suite(tmp_path, CONFIG | {"guidance_scale": 2.0}) == "main"
    remember_suite(tmp_path, CONFIG, "main", bundle)
    assert selected_suite(tmp_path, CONFIG) == "main"


def test_rendering_manifests_are_excluded_from_scientific_hashes(tmp_path):
    from utils.common.io import atomic_write_json

    atomic_write_json(tmp_path / "numerical_audit.json", {"count": 3})
    atomic_write_json(tmp_path / "candidate_figure_manifest.json", {"style": 1})
    atomic_write_json(tmp_path / "candidate_render_audit.json", {"valid": True})
    atomic_write_json(tmp_path / "PF/PF01.json", {"plot": True})
    before = discovery._scalar_hashes(tmp_path)
    atomic_write_json(tmp_path / "candidate_figure_manifest.json", {"style": 2})
    assert discovery._scalar_hashes(tmp_path) == before
    assert set(before) == {"numerical_audit.json"}


def test_summary_interruption_never_publishes_complete(tmp_path, monkeypatch):
    def interrupted(bundle):
        assert not read_json(bundle / "analysis_manifest.json")["complete"]
        raise RuntimeError("synthetic summary interruption")

    monkeypatch.setattr(discovery, "read_tables", interrupted)
    with pytest.raises(RuntimeError, match="synthetic summary interruption"):
        discovery._summarize(tmp_path, {}, complete=True, integration_status="complete")
    assert not read_json(tmp_path / "analysis_manifest.json")["complete"]


@pytest.mark.parametrize("empirical_receipt", [False, True])
def test_candidate_worker_preserves_saved_empirical_masses_exactly(tmp_path, monkeypatch, empirical_receipt):
    import torch
    weights = torch.tensor([.7, .2999999999999999], dtype=torch.float64)
    tensors = {"support.pt": torch.tensor([[0.], [4.]], dtype=torch.float64),
               "center.pt": torch.tensor([1.2], dtype=torch.float64),
               "worker_schedule.pt": {"saved": True},
               "evaluation_initial.pt": torch.zeros((2, 1), dtype=torch.float64)}
    support_metadata = {"atom_ids": ["a", "b"], "aliases": {"a": 0, "duplicate-a": 0, "b": 1},
                        "weights": weights.tolist(), "tensor_sha256": "saved-atoms"}
    if empirical_receipt:
        weights = torch.tensor([.75, .25], dtype=torch.float64)
        support_metadata.update(weights=weights.tolist(), source_record_multiplicities=[3, 1])
        support_metadata["aliases"]["another-a"] = 0
    cache_identity = {"auxiliary_files": {name: name + "-hash" for name in discovery.AUXILIARY},
                      "recorded_diffusers_version": "saved-version"}
    monkeypatch.setattr(discovery, "read_json", lambda path: support_metadata if path.name == "support_metadata.json" else cache_identity)
    monkeypatch.setattr(discovery, "file_sha256", lambda path: path.name + "-hash")
    monkeypatch.setattr(discovery, "safe_torch_load", lambda path: tensors[path.name])
    monkeypatch.setattr(discovery, "SchedulerAdapter", lambda *args, **kwargs: "saved-adapter")
    bank, center, schedule, adapter, initial = discovery._worker_context(tmp_path, "cpu", 1, 2)
    assert torch.equal(bank.weights, weights)
    assert torch.equal(bank.log_weights, weights.log())
    assert bank.aliases == support_metadata["aliases"]
    if empirical_receipt:
        assert bank.metadata()["source_record_multiplicities"] == [3, 1]
    assert torch.equal(bank.weights @ bank.flat, weights @ tensors["support.pt"])
    assert torch.equal(center, tensors["center.pt"]) and schedule is tensors["worker_schedule.pt"]
    assert adapter == "saved-adapter" and initial is tensors["evaluation_initial.pt"]


def test_candidate_segment_bank_identity_pins_masses_and_aliases_without_chunk_sizes():
    metadata = {"tensor_sha256": "same-atoms", "weights": [.75, .25],
                "atom_ids": ["a", "b"], "aliases": {"a": 0, "duplicate-a": 0, "b": 1},
                "candidate_chunk": 1, "query_chunk": 2}
    original = discovery._support_bank_hash(metadata)
    assert discovery._support_bank_hash(metadata | {"weights": [.5, .5]}) != original
    assert discovery._support_bank_hash(metadata | {"aliases": {"a": 0, "b": 1}}) != original
    assert discovery._support_bank_hash(metadata | {"candidate_chunk": 512, "query_chunk": 64}) == original
