"""Regression specifications for source-record empirical reference priors."""
from types import SimpleNamespace

import pytest
import torch

from utils.experiments.theory import reduce
from utils.experiments.theory.reference_law import ReferenceLaw, build_reference_law, reference_law_from_atoms
from utils.experiments.theory.support import FiniteSupport


@pytest.mark.parametrize("candidate_chunk", [1, 2, 8])
def test_collapsed_posterior_equals_equal_record_gaussian_mixture(candidate_chunk):
    atoms = torch.tensor([[1., -2.], [1., -2.], [4., 3.]], dtype=torch.float64)
    support = FiniteSupport.from_candidates(
        zip(["a", "b", "c"], atoms), candidate_chunk=candidate_chunk, query_chunk=1,
    )
    query = torch.tensor([[0., 1.], [2., -1.]], dtype=torch.float64)
    alpha, sigma = .6, .8
    # The independent oracle keeps all three source records and their equal priors.
    logits = -(query[:, None] - alpha * atoms[None]).square().sum(-1) / (2 * sigma**2)
    probability = logits.softmax(-1)
    actual = support.evaluate(query, "a", alpha, sigma, include_mean=True)
    torch.testing.assert_close(actual["posterior_mean"], probability @ atoms)
    torch.testing.assert_close(actual["target_log_probability"].exp(), probability[:, :2].sum(-1))
    torch.testing.assert_close(
        support.posterior_mean(query, 0., 1.), atoms.mean(0).expand_as(query),
    )
    assert support.weights.tolist() == [2 / 3, 1 / 3]
    assert support.metadata()["weight_definition"] == "equal_mass_per_source_record_aggregated_by_exact_latent"


def test_empirical_receipts_do_not_mislabel_explicit_uniform_weights():
    explicit = reference_law_from_atoms([torch.tensor([0.]), torch.tensor([1.])],
                                        ["a", "b"], weights=[3., 3.])
    assert explicit.metadata["weight_definition"] == "predeclared_explicit_weights"
    assert explicit.metadata["source_record_count"] is None
    with pytest.raises(ValueError, match="unique source atom IDs"):
        reference_law_from_atoms([torch.tensor([0.]), torch.tensor([0.])], ["a", "a"])
    with pytest.raises(ValueError, match="multiplicities differ"):
        FiniteSupport(torch.tensor([[0.], [1.]]), ["a", "c"],
                      {"a": 0, "b": 0, "c": 1}, weights=[.5, .5], source_record_counts=[1, 1])
    with pytest.raises(ValueError, match="weights differ"):
        FiniteSupport(torch.tensor([[0.], [1.]]), ["a", "c"],
                      {"a": 0, "b": 0, "c": 1}, weights=[.5, .5], source_record_counts=[2, 1])


def test_cached_factory_keeps_aggregated_masses_aliases_and_weighted_mean(monkeypatch):
    support = FiniteSupport.from_candidates([
        ("a", torch.tensor([2., 1.])), ("b", torch.tensor([2., 1.])),
        ("c", torch.tensor([8., 1.])),
    ])
    rows = [{"candidate_id": key, "prompt_raw": key} for key in support.aliases]
    monkeypatch.setattr(reduce, "build_support", lambda *args, **kwargs: (support, support.metadata(), rows))
    sources = SimpleNamespace(runs={"experiment": {"scientific_config": {}}})
    law = build_reference_law(sources, {"mean_source": "cached-targets", "candidate_chunk_size": 7,
                                        "query_chunk_size": 3})
    assert torch.equal(law.support.weights, support.weights)
    assert law.support.aliases == support.aliases
    assert law.support.candidate_chunk == 7 and law.support.query_chunk == 3
    assert law.metadata["source_record_multiplicities"] == [2, 1]
    assert law.metadata["source_record_count"] == 3
    assert law.metadata["reference_law_scope"] == "D_K_all_compatible_complete_cached_source_records_preselection_empirical_multiplicity"
    torch.testing.assert_close(law.mean_vector, torch.tensor([4., 1.], dtype=torch.float64))
    assert law.metadata["mean_sha256"] == law.metadata["candidate_mean_sha256"]
    assert law.to("cpu").support.metadata() == law.support.metadata()
    # Existing worker payload readers may retain counts only in the law receipt.
    worker_support = FiniteSupport(support.atoms, support.atom_ids, support.aliases,
                                   weights=support.weights)
    worker_support.weights = support.weights.clone()
    worker_support.log_weights = worker_support.weights.log()
    worker_law = ReferenceLaw(worker_support, law.metadata, law.law_hash)
    restored = worker_law.to("cpu")
    assert restored.support.source_record_counts == [2, 1]
    assert torch.equal(restored.support.weights, support.weights)


def test_build_support_counts_all_complete_records_before_selection(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    target_directory = directory / "targets"
    target_directory.mkdir(parents=True)
    paths = SimpleNamespace(run_directory=directory, run_config=directory / "run.json",
                            schedule=directory / "schedule.pt", target_latent_directory=target_directory,
                            target_latent_path=lambda index: target_directory / f"{index}.pt")
    paths.run_config.write_text("fixture")
    paths.schedule.write_text("fixture")
    values = {"01": torch.tensor([2., 1.]), "02": torch.tensor([2., 1.]),
              "03": torch.tensor([8., 1.])}
    records = []
    for index in values:
        paths.target_latent_path(index).write_text("fixture")
        marker = directory / f"record-{index}.json"
        marker.write_text("fixture")
        records.append(SimpleNamespace(original_index=index, marker_path=marker, metadata={
            "scientific_config_hash": "run", "record_id": index, "prompt_raw": index,
            "target_image_sha256": "duplicate" if index != "03" else "different",
            "tensor_file_sha256": {"target_latent": f"{index}.pt"},
        }))
    science = {"latent_shape": [2]}
    run = {"scientific_config": science, "scientific_config_hash": "run"}
    sources = SimpleNamespace(root=tmp_path, experiment=paths, runs={"experiment": run},
                              selection=SimpleNamespace(included_indices={"03"}),
                              metadata_files={}, marker_inventory={})
    monkeypatch.setattr(reduce, "GenerationPaths", lambda value: paths)
    monkeypatch.setattr(reduce, "read_json", lambda path: run)
    monkeypatch.setattr(reduce, "require_generation_run", lambda path: run)
    monkeypatch.setattr(reduce, "list_completed_records", lambda path: records)
    monkeypatch.setattr(reduce, "file_sha256", lambda path: path.name)
    monkeypatch.setattr(reduce, "safe_torch_load", lambda path: values[path.stem])
    support, metadata, membership = reduce.build_support(sources)
    assert support.weights.tolist() == [2 / 3, 1 / 3]
    assert metadata["membership_source_count"] == metadata["source_record_count"] == 3
    assert metadata["source_record_multiplicities"] == [2, 1]
    assert membership.included.tolist() == [False, False, True]
    assert membership.source_record_prior_mass.tolist() == [1 / 3] * 3
    assert membership.support_atom_source_count.tolist() == [2, 2, 1]
    assert membership.support_atom_prior_mass.tolist() == [2 / 3, 2 / 3, 1 / 3]
    assert metadata["candidate_mean_kind"] == "mu_K_empirical_mean_of_declared_source_records"
