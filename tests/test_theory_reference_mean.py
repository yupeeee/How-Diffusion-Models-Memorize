"""Regression specifications for the independent CUDA reference-mean estimator."""
from types import SimpleNamespace
import math

import pytest
import torch

from utils.common.io import atomic_torch_save, atomic_write_json, file_sha256
from utils.experiments.theory import reference_mean as mean
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.support import _tensor_hash
from utils.experiments.theory.reference_law import reference_law_from_atoms


@pytest.fixture
def mean_sources(tmp_path, monkeypatch):
    science = {"model_id": "synthetic/model", "model_revision": "fixed-revision",
               "native_prediction_type": "epsilon", "inference_dtype": "float32",
               "latent_shape": [1, 1, 2], "seeds": [0, 1, 2]}
    schedule = {"timesteps": torch.tensor([9, 3]),
                "alphas_cumprod_t": torch.tensor([.04, .64], dtype=torch.float64)}
    path = tmp_path / "schedule.pt"
    atomic_torch_save(schedule, path)
    sources = SimpleNamespace(root=tmp_path, experiment=SimpleNamespace(schedule=path),
        selected=[SimpleNamespace(original_index="prompt1")],
        runs={"experiment": {"scientific_config": science},
              "reference": {"scientific_config": {"seeds": [3, 4, 5]}}})
    monkeypatch.setattr(mean, "_source_code", lambda: {"reference_mean.py": "source"})
    monkeypatch.setattr(mean, "_runtime", lambda: {"torch": "runtime", "torch_cuda": "cuda-runtime"})
    law = reference_law_from_atoms([torch.tensor([[[-1., 0.]]]), torch.tensor([[[2., 1.]]])],
                                   ["a", "b"], weights=[1., 3.])
    return sources, schedule, law


def test_default_mean_plan_has_exactly_ten_thousand_independent_minimum_snr_draws(mean_sources):
    sources, schedule, law = mean_sources
    plan = mean.plan_mean(sources, {}, schedule, law)
    assert plan["identity"]["num_mean_samples"] == 10_000
    assert len(plan["tasks"]) == math.ceil(10_000 / mean.LOGICAL_SHARD_SIZE)
    assert sum(task["count"] for task in plan["tasks"]) == 10_000
    seeds = [seed for task in plan["tasks"] for seed in task["identity"]["draw_seeds"]]
    assert len(set(seeds)) == 10_000 and not set(seeds) & set(range(6))
    assert all((1 << 61) <= seed < (1 << 62) for seed in seeds)
    bank = plan["identity"]["bank"]
    snr = float(mean.reference_grid(.04 / .96, 6.0)[0])
    assert bank["level"] == dict(source_range="analytical", grid_index=0, step_index=None,
                                  timestep=None, alpha=math.sqrt(snr / (1 + snr)),
                                  sigma=1 / math.sqrt(1 + snr), snr=snr)
    assert bank["initial_level"] == dict(source_range="native", grid_index=None, step_index=0,
                                          timestep=9, alpha=.2, sigma=math.sqrt(.96), snr=.04 / .96)
    assert bank["initial_snr"] == .04 / .96 and bank["estimation_snr"] == snr
    assert bank["mean_source"] == "reference-min-snr"
    assert bank["source"] == "minimum_snr_unconditional_reference_monte_carlo"
    assert bank["reference_snr_decades"] == 6.0
    assert bank["reference_grid_definition"]["points"] == 97
    sources.selected *= 40
    assert mean.plan_mean(sources, {}, schedule, law) == plan  # Never one pass per prompt.
    assert mean.plan_mean(sources, {"probe_batch_size": 1, "device": "cuda:7"}, schedule, law) == plan
    larger = mean.plan_mean(sources, {"num_mean_samples": 10_128}, schedule, law)
    assert larger["tasks"][:-2] == plan["tasks"][:-1]  # Completed full shards remain reusable.


def test_mean_modes_and_sweep_decades_have_distinct_cache_and_draw_identities(mean_sources):
    sources, schedule, law = mean_sources
    config = {"num_mean_samples": 4}
    minimum = mean.plan_mean(sources, config, schedule, law)
    initial = mean.plan_mean(sources, {**config, "mean_source": "reference-initial"}, schedule, law)
    different_grid = mean.plan_mean(sources, {**config, "reference_snr_decades": 4.0}, schedule, law)
    bank = initial["identity"]["bank"]
    assert bank["source"] == "initial_unconditional_reference_monte_carlo"
    assert bank["level"] == bank["initial_level"]
    assert bank["estimation_snr"] == bank["initial_snr"] == .04 / .96
    assert len({plan["estimator_hash"] for plan in (minimum, initial, different_grid)}) == 3
    assert len({plan["tasks"][0]["task_hash"] for plan in (minimum, initial, different_grid)}) == 3
    assert different_grid["identity"]["bank"]["estimation_snr"] == float(mean.reference_grid(.04 / .96, 4.0)[0])
    assert all(plan["identity"]["bank"]["initial_level"] == bank["initial_level"]
               for plan in (minimum, different_grid))


def test_mean_minimum_snr_uses_the_shared_reference_grid_endpoint(mean_sources, monkeypatch):
    sources, schedule, law = mean_sources
    actual_grid = mean.reference_grid
    calls = []

    def shared_grid(initial_snr, decades):
        calls.append((initial_snr, decades))
        return actual_grid(initial_snr, decades)

    monkeypatch.setattr(mean, "reference_grid", shared_grid)
    plan = mean.plan_mean(sources, {"num_mean_samples": 2, "reference_snr_decades": 3.5}, schedule, law)
    assert calls == [(.04 / .96, 3.5)]
    assert plan["identity"]["bank"]["level"]["snr"] == float(actual_grid(.04 / .96, 3.5)[0])


def test_reference_mean_identity_uses_exact_atom_law_and_ignores_attached_comparison_mean(mean_sources):
    sources, schedule, law = mean_sources
    original = mean.plan_mean(sources, {"num_mean_samples": 4}, schedule, law)
    law.metadata["theory_mean"] = {"source": "irrelevant_comparison_center", "vector_sha256": "other"}
    assert mean.plan_mean(sources, {"num_mean_samples": 4}, schedule, law) == original
    changed = reference_law_from_atoms(law.support.atoms, law.support.atom_ids, weights=[3., 1.])
    updated = mean.plan_mean(sources, {"num_mean_samples": 4}, schedule, changed)
    assert updated["estimator_hash"] != original["estimator_hash"]
    assert updated["identity"]["bank"]["reference_atom_law_hash"] != original["identity"]["bank"]["reference_atom_law_hash"]


@pytest.mark.parametrize("mean_source, allow_compute", [
    ("reference-min-snr", True), ("reference-min-snr", False),
    ("reference-initial", True), ("reference-initial", False), ("cached-targets", False),
])
def test_reference_factory_separates_chunk_execution_options_before_estimator_paths(
        mean_sources, monkeypatch, mean_source, allow_compute):
    from utils.experiments.theory import reduce
    from utils.experiments.theory.contracts import numerical_config
    from utils.experiments.theory.paper_contracts import PaperPaths
    from utils.experiments.theory.reference_law import build_reference_law

    sources, _, atom_law = mean_sources
    scientific = numerical_config(model_name="sdv1", scheduler_name="ddim", num_seeds=3,
                                  num_inference_steps=2, mean_source=mean_source,
                                  num_mean_samples=257, mean_seed=19)
    supplied = {**scientific, "candidate_chunk_size": 31, "query_chunk_size": 7}
    original = dict(supplied)
    support_calls, estimate_calls = [], []
    selected_mean = torch.tensor([[[.25, -.5]]], dtype=torch.float64)

    def build_support(received_sources, **options):
        assert received_sources is sources
        support_calls.append(dict(options))
        return atom_law.support, {"candidate_chunk": 31, "query_chunk": 7, "source": "fixture"}, [
            {"candidate_id": "a", "prompt_raw": "first"},
            {"candidate_id": "b", "prompt_raw": "second"},
        ]

    def estimate(received_sources, received_config, **options):
        assert received_sources is sources
        # Exercise the real strict path/config boundary, not a permissive mock:
        # forwarding candidate/query chunk keys here caused the reported crash.
        paths = PaperPaths.build(sources.root, **received_config)
        assert paths == PaperPaths.build(sources.root, **scientific)
        assert received_config == scientific
        assert received_config["num_mean_samples"] == 257 and received_config["mean_seed"] == 19
        assert options["reference_law"].support.candidate_chunk == 31
        assert options["reference_law"].support.query_chunk == 7
        assert options["reference_law"].estimated_mean_vector is None
        assert options["device"] == "cuda:3" and options["batch_size"] == 5
        assert options["allow_compute"] is allow_compute
        estimate_calls.append(dict(received_config))
        return {"vector": selected_mean.clone(), "directory": paths.output_directory / "synthetic-mean",
                "metadata": {"source": mean.MEAN_SOURCES[mean_source],
                             "estimator_hash": "estimate", "vector_sha256": _tensor_hash(selected_mean),
                             "sample_count": 257, "mean_seed": 19}}

    monkeypatch.setattr(reduce, "build_support", build_support)
    monkeypatch.setattr(mean, "estimate_reference_mean", estimate)
    law = build_reference_law(sources, supplied, device="cpu", allow_mean_compute=allow_compute,
                              mean_device="cuda:3", mean_batch_size=5)
    assert support_calls == [{"candidate_chunk_size": 31, "query_chunk_size": 7, "device": "cpu"}]
    assert law.support.candidate_chunk == 31 and law.support.query_chunk == 7
    assert supplied == original  # Splitting execution options must not mutate caller configuration.
    if mean_source in mean.MEAN_SOURCES:
        assert estimate_calls == [scientific]
        torch.testing.assert_close(law.theory_mean_vector, selected_mean, rtol=0, atol=0)
        assert law.theory_mean_metadata["sample_count"] == 257
        assert law.theory_mean_metadata["mean_seed"] == 19
    else:
        assert estimate_calls == []
        assert law.estimated_mean_vector is None
        torch.testing.assert_close(law.theory_mean_vector, law.mean_vector, rtol=0, atol=0)
        assert law.theory_mean_metadata["source"] == "declared_finite_bank_mean"


def test_mean_seed_domain_is_deterministic_prefix_stable_and_avoids_explicit_collisions():
    context = {"mean_seed": 7, "checkpoint": "fixed"}
    initial = mean.draw_seeds(context, 4)
    assert mean.draw_seeds(context, 8)[:4] == initial
    blocked = mean.draw_seeds(context, 4, forbidden=[initial[0]])
    assert initial[0] not in blocked and len(set(blocked)) == 4
    assert mean.draw_seeds({**context, "mean_seed": 8}, 4) != initial


@pytest.mark.parametrize("options", [{"num_mean_samples": 1}, {"num_mean_samples": True},
                                      {"mean_seed": -1}, {"mean_seed": False}])
def test_mean_plan_rejects_invalid_fixed_design(mean_sources, options):
    sources, schedule, law = mean_sources
    with pytest.raises(TheoryError, match="must be an integer"):
        mean.plan_mean(sources, options, schedule, law)


def test_mean_plan_requires_a_valid_initial_native_label(mean_sources):
    sources, schedule, law = mean_sources
    with pytest.raises(TheoryError, match="positive initial native"):
        mean.plan_mean(sources, {}, {**schedule, "alphas_cumprod_t": torch.tensor([0., .5])}, law)
    original = mean.plan_mean(sources, {}, schedule, law)
    changed = mean.plan_mean(sources, {"mean_seed": 1}, schedule, law)
    assert original["estimator_hash"] != changed["estimator_hash"]


@pytest.mark.parametrize("options, message", [
    ({"mean_source": "cached-targets"}, "mean_source"),
    ({"mean_source": "unknown"}, "mean_source"),
    ({"reference_snr_decades": 0.0}, "decades"),
    ({"reference_snr_decades": 12.1}, "decades"),
    ({"reference_snr_decades": math.nan}, "decades"),
])
def test_mean_plan_rejects_invalid_estimation_noise_configuration(mean_sources, options, message):
    sources, schedule, law = mean_sources
    with pytest.raises(TheoryError, match=message):
        mean.plan_mean(sources, {"num_mean_samples": 2, **options}, schedule, law)


def test_mean_plan_rejects_underflow_before_constructing_the_reference_grid(mean_sources, monkeypatch):
    sources, schedule, law = mean_sources
    smallest = float.fromhex("0x0.0000000000001p-1022")
    schedule = {**schedule, "alphas_cumprod_t": torch.tensor([smallest, .5], dtype=torch.float64)}

    def unexpected_grid(*args):
        raise AssertionError("Underflow must be rejected before the geometric grid is constructed")

    monkeypatch.setattr(mean, "reference_grid", unexpected_grid)
    with pytest.raises(TheoryError, match="positive and finite"):
        mean.plan_mean(sources, {"num_mean_samples": 2}, schedule, law)


@pytest.mark.parametrize("mode", ["reference-min-snr", "reference-initial"])
def test_mean_vector_cache_is_full_shape_hash_checked_and_readable_without_cuda(mean_sources, tmp_path, mode):
    sources, schedule, law = mean_sources
    plan = mean.plan_mean(sources, {"num_mean_samples": 2, "mean_source": mode}, schedule, law)
    directory = tmp_path / "complete"
    vector = torch.tensor([[[1., -2.]]], dtype=torch.float64)
    files = {"mean.pt": atomic_torch_save(vector, directory / "mean.pt"),
             "uncertainty.pt": atomic_torch_save({"coordinate_variance": 2 * torch.ones_like(vector),
                                                  "coordinate_standard_error": torch.ones_like(vector)}, directory / "uncertainty.pt")}
    marker = {"schema_version": mean.SCHEMA_VERSION, "complete": True,
              "identity": plan["identity"], "estimator_hash": plan["estimator_hash"],
              "sample_count": 2, "mean_seed": 0,
              **{name: plan["identity"]["bank"][name] for name in mean.LEVEL_RECEIPT_FIELDS},
              "reference_atom_law_hash": plan["identity"]["bank"]["reference_atom_law_hash"],
              "reference_atom_law": plan["identity"]["bank"]["reference_atom_law"],
              "latent_dimension": 2, "split_half_counts": [1, 1],
              "mean_norm_l2": math.sqrt(5), "mean_norm_rmse": math.sqrt(5 / 2),
              "mean_mc_standard_error_l2": math.sqrt(2), "mean_mc_standard_error_rmse": 1.,
              "mean_mc_standard_error_max_coordinate": 1., "split_half_difference_rmse": 2.,
              "vector_sha256": _tensor_hash(vector), "files": files}
    atomic_write_json(directory / "complete.json", marker)
    loaded = mean._load_completed(directory, plan["identity"])
    torch.testing.assert_close(loaded["vector"], vector)
    assert loaded["vector"].device.type == "cpu"
    for name in ("mean_seed", *mean.LEVEL_RECEIPT_FIELDS, "reference_atom_law_hash", "mean_mc_standard_error_rmse"):
        atomic_write_json(directory / "complete.json", {key: value for key, value in marker.items() if key != name})
        assert mean._load_completed(directory, plan["identity"]) is None
    other_mode = "reference-initial" if mode == "reference-min-snr" else "reference-min-snr"
    for changed in ({"mean_source": other_mode}, {"source": mean.MEAN_SOURCES[other_mode]},
                    {"estimation_snr": marker["estimation_snr"] * 2}, {"reference_snr_decades": 5.0},
                    {"reference_grid_definition": {**marker["reference_grid_definition"], "grid_sha256": "other"}}):
        atomic_write_json(directory / "complete.json", marker | changed)
        assert mean._load_completed(directory, plan["identity"]) is None
    atomic_write_json(directory / "complete.json", marker | {"mean_mc_standard_error_rmse": -1.})
    assert mean._load_completed(directory, plan["identity"]) is None
    atomic_write_json(directory / "complete.json", marker | {"vector_sha256": "wrong-vector"})
    assert mean._load_completed(directory, plan["identity"]) is None
    atomic_write_json(directory / "complete.json", marker)
    (directory / "mean.pt").write_bytes(b"interrupted mean vector")
    assert mean._load_completed(directory, plan["identity"]) is None


def test_mean_gpu_math_has_no_cpu_fallback():
    with pytest.raises(TheoryError, match="requires CUDA"):
        mean.gaussian_draws([11, 12], [1, 2], "cpu")
    with pytest.raises(TheoryError, match="requires CUDA"):
        mean.shard_moments(torch.ones(2, 1, 2, dtype=torch.float64), 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mean estimator arithmetic")
def test_mean_cuda_moments_give_unbiased_variance_standard_error_and_split_difference():
    values = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], device="cuda", dtype=torch.float64)
    parts = [mean.shard_moments(values[:2], 0), mean.shard_moments(values[2:], 2)]
    estimates, metadata = mean.combine_moments(parts, [2, 2])
    torch.testing.assert_close(estimates["mean"], torch.tensor([4., 5.], device="cuda", dtype=torch.float64))
    torch.testing.assert_close(estimates["coordinate_variance"], torch.full((2,), 20 / 3, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(estimates["coordinate_standard_error"], torch.full((2,), math.sqrt(5 / 3), device="cuda", dtype=torch.float64))
    assert metadata["mean_mc_standard_error_rmse"] == pytest.approx(math.sqrt(5 / 3))
    assert metadata["mean_norm_rmse"] == pytest.approx(math.sqrt(41 / 2))
    assert metadata["split_half_difference_rmse"] == pytest.approx(2.)
    assert metadata["split_half_counts"] == [2, 2]
    assert all(value.device.type == "cuda" for value in estimates.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for stable mean uncertainty")
def test_mean_uncertainty_does_not_cancel_small_variance_around_large_mean():
    offsets = torch.tensor([[-3., -3.], [-1., -1.], [1., 1.], [3., 3.]], device="cuda", dtype=torch.float64)
    values = 1e12 + offsets
    parts = [mean.shard_moments(values[:2], 0), mean.shard_moments(values[2:], 2)]
    estimates, metadata = mean.combine_moments(parts, [2, 2])
    torch.testing.assert_close(estimates["coordinate_variance"], torch.full((2,), 20 / 3, device="cuda", dtype=torch.float64))
    assert metadata["mean_mc_standard_error_rmse"] == pytest.approx(math.sqrt(5 / 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for analytical posterior means")
def test_first_step_reference_mean_single_atom_has_exact_mean_and_zero_mc_variance():
    atom = torch.tensor([[[2., -3.]]], device="cuda", dtype=torch.float64)
    law = reference_law_from_atoms([atom], ["only"], device="cuda")
    seeds = mean.draw_seeds({"mean_seed": 5}, 9)
    samples = mean.gaussian_draws(seeds, list(atom.shape), "cuda")
    batches = list(mean.posterior_batches(samples, support=law.support,
                   level={"alpha": .2, "sigma": math.sqrt(.96)}, batch_size=4))
    posteriors = torch.cat([value for _, _, value, _ in batches])
    estimates, metadata = mean.combine_moments([mean.shard_moments(posteriors, 0)], [9])
    torch.testing.assert_close(estimates["mean"], atom, rtol=0, atol=0)
    assert estimates["coordinate_variance"].eq(0).all()
    assert metadata["mean_mc_standard_error_rmse"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for independent Gaussian bank")
def test_mean_draws_are_cuda_generated_and_independent_of_batch_partition():
    seeds = mean.draw_seeds({"mean_seed": 9}, 7)
    whole = mean.gaussian_draws(seeds, [1, 2, 2], "cuda:0")
    parts = torch.cat([mean.gaussian_draws(seeds[:2], [1, 2, 2], "cuda:0"),
                       mean.gaussian_draws(seeds[2:], [1, 2, 2], "cuda:0")])
    torch.testing.assert_close(whole, parts, rtol=0, atol=0)
    assert whole.device.type == "cuda" and whole.dtype == torch.float64
    if torch.cuda.device_count() > 1:
        other = mean.gaussian_draws(seeds, [1, 2, 2], "cuda:1")
        torch.testing.assert_close(whole, other.to(whole.device), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for synthetic estimator worker")
@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("mode", ["reference-min-snr", "reference-initial"])
def test_reference_mean_worker_uses_selected_analytical_posterior_only_and_resumes(mean_sources, tmp_path, monkeypatch, batch_size, mode):
    from utils.experiments.theory import direct_probes
    sources, schedule, law = mean_sources
    count = mean.LOGICAL_SHARD_SIZE + 3
    plan = mean.plan_mean(sources, {"num_mean_samples": count, "mean_source": mode}, schedule, law)
    calls, loads = [], []
    original_load, original_batches = mean._load_reference, mean.posterior_batches

    def load_reference(*args, **kwargs):
        loads.append(1)
        return original_load(*args, **kwargs)

    def posteriors(samples, **kwargs):
        assert kwargs["level"] == plan["identity"]["bank"]["level"]
        assert kwargs["level"]["snr"] == plan["identity"]["bank"]["estimation_snr"]
        for start, stop, posterior, retries in original_batches(samples, **kwargs):
            calls.append(samples[start:stop].clone())
            yield start, stop, posterior, retries

    def forbidden(*args, **kwargs):
        raise AssertionError("Analytical reference-mean estimation must never load a learned checkpoint")

    monkeypatch.setattr(mean, "_load_reference", load_reference)
    monkeypatch.setattr(mean, "posterior_batches", posteriors)
    monkeypatch.setattr(direct_probes, "_load_replica", forbidden)
    root = tmp_path / "moments"
    inputs = mean._persist_reference(root, law, plan["identity"]["bank"]["reference_atom_law"])
    kwargs = dict(inputs=inputs, schedule_path=sources.experiment.schedule,
                  tasks=plan["tasks"], directory=root, device="cuda:0", batch_size=batch_size, worker_count=1)
    result = mean._worker(**kwargs)
    assert not result["failures"] and len(loads) == 1
    assert sum(len(batch) for batch in calls) == count
    assert all(batch.device.type == "cuda" for batch in calls)
    assert all(mean._completed_shard(root, task) is not None for task in plan["tasks"])
    assert all(mean._completed_shard(root, task)["execution"]["network_observations"] == 0 for task in plan["tasks"])
    hashes = {task["task_hash"]: file_sha256(mean._shard_paths(root, task)[0]) for task in plan["tasks"]}
    before = len(calls)
    resumed = mean._worker(**kwargs)
    assert not resumed["failures"] and len(calls) == before and len(loads) == 1
    assert hashes == {task["task_hash"]: file_sha256(mean._shard_paths(root, task)[0]) for task in plan["tasks"]}
