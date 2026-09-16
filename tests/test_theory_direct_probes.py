"""Synthetic direct-probe contracts. Written for author execution, not run here."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from utils.common.io import atomic_torch_save, atomic_write_json, file_sha256, read_json
from utils.experiments.cache import CompletedGenerationRecord, GenerationPaths
from utils.experiments.theory import direct_probes as probes
from utils.experiments.theory.direct_probe_math import (
    conditional_forward_metrics,
    conditional_gaussian_metrics,
    gaussian_reference_metrics,
    marginal_forward_metrics,
    pinsker_quantities,
    summarize_forward_losses,
    summarize_gaussian_conditional,
)
from utils.experiments.theory.reference_law import reference_law_from_atoms
from utils.models.loading import RuntimeSelection
from utils.models.probe_loading import load_denoiser_components


def test_forward_draw_loss_sums_coordinates_and_only_averages_draws_once():
    target = torch.zeros(1, 2, 2, dtype=torch.float64)
    noise = torch.tensor([1.0, 2.0], dtype=torch.float64)[:, None, None, None].expand(2, 1, 2, 2)
    a, s = 0.6, 0.8
    values = conditional_forward_metrics(target, noise, s * noise, torch.zeros_like(noise), a, s)
    torch.testing.assert_close(values["loss_squared_l2"], torch.tensor([4.0, 16.0], dtype=torch.float64))
    assert values["identity_numeric_pass"].all()
    fields = dict(run_id="r", original_index="0001", record_id="p", target_id="image", target_latent_sha256="latent",
                  step_index=0, timestep=10, alpha=a, sigma=s, snr=(a / s) ** 2, input_source="forward_target")
    fields.update(pinsker_quantities(target, a, s))
    rows = probes._rows(values, 2, fields)
    for index, row in enumerate(rows):
        row["draw_index"] = index
    summary = summarize_forward_losses(pd.DataFrame(rows)).iloc[0]
    assert summary.draw_count == 2
    assert summary.loss_sum_squared_l2 == 20
    assert summary.loss_mean_squared_l2 == 10
    assert summary.loss_over_snr_mse == pytest.approx(10 / (4 * (a / s) ** 2))
    assert summary.loss_variance_squared_l2 == 72
    assert summary.loss_mc_se_squared_l2 == 6
    assert summary.forward_clean_error_mean_squared_l2 == pytest.approx(float(values["clean_error_squared_l2"].mean()))


def test_gaussian_summary_distinguishes_mean_norm_from_root_mean_square():
    identity = dict(run_id="run", original_index="0001", record_id="pair", target_id="image",
                    step_index=0, timestep=9, snr=0.25, latent_dimension=4, input_source="gaussian_probe")
    observations = pd.DataFrame([
        identity | {"seed": seed, "conditional_error_l2": error, "conditional_error_squared_l2": error**2}
        for seed, error in enumerate((1.0, 3.0))
    ])
    row = summarize_gaussian_conditional(observations).iloc[0]
    assert row.gaussian_seed_count == 2
    assert row.gaussian_mean_error_l2 == 2
    assert row.gaussian_mean_squared_error_l2 == 5
    assert row.gaussian_root_mean_squared_error_l2 == pytest.approx(np.sqrt(5))
    assert row.gaussian_mean_error_rmse == 1
    assert row.gaussian_root_mean_squared_error_rmse == pytest.approx(np.sqrt(5) / 2)
    assert (row.gaussian_error_q25_rmse, row.gaussian_error_median_rmse, row.gaussian_error_q75_rmse) == (0.75, 1.0, 1.25)
    with pytest.raises(ValueError, match="Duplicate Gaussian seed"):
        summarize_gaussian_conditional(pd.concat([observations, observations.iloc[:1]], ignore_index=True))


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_native_conversion_uses_unscaled_conceptual_input_once(prediction_type):
    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.seen = []

        def forward(self, sample, timestep, **kwargs):
            self.seen.append(sample.detach().clone())
            return (self.native.expand_as(sample),)

    class Scheduler:
        config = {"prediction_type": prediction_type}
        alphas_cumprod = torch.tensor([0.36], dtype=torch.float64)

        def scale_model_input(self, sample, timestep):
            return sample * 7

        def step(self, *args, **kwargs):
            raise AssertionError("Gaussian/forward probes must never update a state")

    denoiser = Denoiser()
    z = torch.ones(2, 1, 2, 2, dtype=torch.float64)
    components = SimpleNamespace(device=torch.device("cpu"), inference_dtype=torch.float64, unet=denoiser)
    epsilon, precision = probes._prediction_microbatch(z, condition=torch.ones(1, 2, 3, dtype=torch.float64), timestep=0, components=components, scheduler=Scheduler())
    torch.testing.assert_close(denoiser.seen[0], z * 7)
    expected = torch.full_like(z, 2 if prediction_type == "epsilon" else 0.6 * 2 + 0.8)
    torch.testing.assert_close(epsilon, expected)
    assert precision["prediction_conversion_dtype"] == "float64"
    assert precision["input_quantization_l2"].eq(0).all()


def test_forward_expected_error_and_gaussian_error_are_distinct_input_laws():
    # Symmetric two-point quadrature has exact Gaussian first and second moments
    # for this linear denoiser. It is not a claim about arbitrary finite samples.
    target = torch.full((1, 2, 2), 2.0, dtype=torch.float64)
    noise = torch.tensor([-1.0, 1.0], dtype=torch.float64)[:, None, None, None].expand(2, 1, 2, 2)
    a, s = 0.6, 0.8
    forward = conditional_forward_metrics(target, noise, a * target + s * noise, torch.zeros_like(noise), a, s)
    gaussian = conditional_gaussian_metrics(target, noise, torch.zeros_like(noise), a, s)
    assert float(forward["clean_error_squared_l2"].mean()) == pytest.approx(4 * (s / a) ** 2)
    assert float(gaussian["conditional_error_squared_l2"].mean()) == pytest.approx(4 / a ** 2 + 16)
    assert float(forward["clean_error_squared_l2"].mean()) != float(gaussian["conditional_error_squared_l2"].mean())


def test_pinsker_keeps_small_variance_term_and_only_clips_probability_bound():
    target = torch.zeros(1, 2, 2, dtype=torch.float64)
    alpha = 1e-4
    sigma = np.sqrt(1 - alpha ** 2)
    value = pinsker_quantities(target, alpha, sigma)
    assert value["KL_target_to_gaussian"] > 0
    assert value["pinsker_tv_bound"] > 0
    huge = pinsker_quantities(torch.full_like(target, 1e6), 0.6, 0.8)
    assert huge["KL_target_to_gaussian"] > 1
    assert huge["pinsker_tv_bound"] == 1
    exact = pinsker_quantities(target, 0.6, 0.8)
    expected = 0.5 * 4 * (0.64 - 1 - np.log(0.64))
    assert exact["KL_target_to_gaussian"] == pytest.approx(expected)


def test_marginal_excess_is_direct_nonnegative_and_cross_term_is_retained():
    noise = torch.ones(3, 1, 2, 2, dtype=torch.float64)
    z = noise * 0.8
    posterior = torch.zeros_like(noise)
    predicted = torch.full_like(noise, 0.5)
    values = marginal_forward_metrics(noise, z, predicted, posterior, 0.6, 0.8)
    assert values["excess_loss_squared_l2"].ge(0).all()
    assert values["identity_numeric_pass"].all()
    torch.testing.assert_close(values["loss_decomposition_residual_squared_l2"], torch.zeros(3, dtype=torch.float64))
    # The optimal noise residual is nonzero in this deliberately finite sample.
    z = torch.zeros_like(noise)
    values = marginal_forward_metrics(noise, z, predicted, posterior, 0.6, 0.8)
    assert values["cross_term_squared_l2"].ne(0).all()
    assert not torch.equal(values["total_loss_squared_l2"] - values["optimal_loss_squared_l2"], values["excess_loss_squared_l2"])


def test_gaussian_reference_uses_declared_nonzero_mean_and_keeps_log_bound_overflow():
    z = torch.ones(2, 1, 2, 2, dtype=torch.float64)
    posterior = z * 3
    mean = z[0] * 2
    values = gaussian_reference_metrics(z, torch.zeros_like(z), posterior, mean, 0.6, 0.8, 1000)
    expected = (z / 0.6 - mean).flatten(1).norm(dim=1)
    torch.testing.assert_close(values["learned_mean_error_l2"], expected)
    # d=4: the zero baseline differs from the declared nonzero mean and is
    # normalized once, while the original mean-centered decomposition remains.
    for name, raw in (("learned_zero_error", 2 / 0.6),
                      ("reference_zero_error", 6.0), ("mean_norm", 4.0)):
        torch.testing.assert_close(values[name + "_l2"], torch.full((2,), raw, dtype=torch.float64))
        torch.testing.assert_close(values[name + "_rmse"], torch.full((2,), raw / 2, dtype=torch.float64))
    assert not torch.equal(values["learned_zero_error_l2"], values["learned_mean_error_l2"])
    assert not torch.equal(values["reference_zero_error_l2"], values["reference_mean_error_l2"])
    serialized = probes._rows(values, 2, {"input_source": "gaussian_probe"})
    assert all(row["mean_norm_rmse"] == 2.0 for row in serialized)
    assert values["baseline_slack_l2"].ge(-1e-12).all()
    assert values["baseline_vector_residual_l2"].lt(1e-12).all()
    assert values["reference_bound_overflow"].all()
    assert torch.isfinite(values["reference_bound_log_l2"]).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"))])
def test_gaussian_zero_and_mean_baselines_agree_only_for_zero_mean_on_device(device):
    z = torch.tensor([[1.0, -2.0], [-3.0, 4.0]], device=device, dtype=torch.float64)
    posterior = z * 0.25
    values = gaussian_reference_metrics(z, torch.zeros_like(z), posterior, z[0] * 0, 0.6, 0.8, 2.0)
    for name in ("learned", "reference"):
        torch.testing.assert_close(values[name + "_zero_error_l2"], values[name + "_mean_error_l2"])
        torch.testing.assert_close(values[name + "_zero_error_rmse"], values[name + "_mean_error_rmse"])
        assert values[name + "_zero_error_rmse"].device == z.device
    assert values["mean_norm_l2"].eq(0).all()
    assert values["mean_norm_rmse"].device == z.device


def test_streams_are_domain_separated_order_and_batch_invariant():
    def seeds(purpose, pair="p", count=9):
        return [probes.derive_draw_seed(0, checkpoint="immutable-checkpoint", pair=pair, timestep=42, purpose=purpose, draw_index=i, forbidden=range(40)) for i in range(count)]
    forward = seeds("forward_target")
    marginal = seeds("forward_marginal_noise")
    atoms = seeds("forward_marginal_atom")
    assert len(set(forward + marginal + atoms + seeds("forward_target", pair="other"))) == 36
    assert not set(forward) & set(range(40))
    assert forward[:4] == seeds("forward_target", count=4)
    whole = probes.draw_noise(forward, (1, 2, 2))
    chunks = torch.cat([probes.draw_noise(forward[i:i + 2], (1, 2, 2)) for i in range(0, 9, 2)])
    torch.testing.assert_close(whole, chunks, rtol=0, atol=0)
    torch.testing.assert_close(probes.draw_noise(list(reversed(forward)), (1, 2, 2)).flip(0), whole, rtol=0, atol=0)


def test_cuda_oom_retries_same_logical_slice_without_gpu_execution(monkeypatch):
    samples = torch.arange(8, dtype=torch.float64).reshape(8, 1, 1, 1)
    calls = []

    def attempted(batch, **kwargs):
        calls.append(batch.flatten().tolist())
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError("synthetic allocation failure")
        return batch.clone(), {}

    monkeypatch.setattr(probes, "_prediction_microbatch", attempted)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    batches = list(probes.prediction_batches(samples, condition=None, timestep=0, components=SimpleNamespace(device="cuda:0"), scheduler=None, batch_size=8))
    assert calls[:3] == [list(range(8)), list(range(4)), [0, 1]]
    assert [(a, b) for a, b, _, _ in batches] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    torch.testing.assert_close(torch.cat([v for _, _, v, _ in batches]), samples)


@pytest.mark.parametrize("kind", ["single_item_oom", "unrelated_runtime"])
def test_microbatch_retry_is_bounded_and_does_not_hide_other_errors(monkeypatch, kind):
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        if kind == "single_item_oom":
            raise torch.cuda.OutOfMemoryError("still too large")
        raise RuntimeError("invalid checkpoint")

    monkeypatch.setattr(probes, "_prediction_microbatch", failed)
    with pytest.raises(RuntimeError):
        list(probes.prediction_batches(torch.zeros(1, 1, 1, 1), condition=None, timestep=0, components=SimpleNamespace(device="cuda:0"), scheduler=None, batch_size=1))
    assert len(calls) == 1


@pytest.fixture
def probe_sources(tmp_path):
    paths = GenerationPaths(tmp_path / "logs" / "synthetic" / "experiment_S0_N2")
    paths.create()
    schedule = {
        "timesteps": torch.tensor([9, 5, 1]),
        "alphas_cumprod_t": torch.tensor([0.04, 0.36, 0.81]),
        "alpha_t": torch.tensor([0.2, 0.6, 0.9]),
        "sigma_t": torch.tensor([np.sqrt(0.96), 0.8, np.sqrt(0.19)]),
        "init_noise_sigma": 1.0,
    }
    atomic_torch_save(schedule, paths.schedule)
    science = {"model_id": "synthetic/model", "model_revision": "a" * 40,
               "native_prediction_type": "epsilon", "stored_prediction_type": "epsilon",
               "inference_dtype": "float32", "latent_shape": [1, 2, 2], "seeds": [0, 1],
               "scientific_tensor_storage": {"dtype": "float32"}}
    records, targets = [], []
    for index in ("0001", "0002"):
        target = torch.full((1, 2, 2), float(int(index)))
        targets.append(target)
        digest = atomic_torch_save(target, paths.target_latent_path(index))
        paths.latent_path(index).write_bytes(b"initial-only fixture")
        paths.noise_prediction_path(index).write_bytes(b"initial-only predictions")
        metadata = {"record_id": "record-" + index, "target_image_sha256": "image-" + index,
                    "tensor_file_sha256": {"target_latent": digest}, "prompt_raw": " shared prompt\n",
                    "scientific_config_hash": "run"}
        atomic_write_json(paths.record_path(index), metadata)
        records.append(CompletedGenerationRecord(index, int(index), paths.record_path(index), metadata))
    sources = SimpleNamespace(root=tmp_path, experiment=paths, selected=records,
                              config={"num_seeds": 2, "num_inference_steps": 3},
                              runs={"experiment": {"scientific_config": science, "scientific_config_hash": "run"},
                                    "reference": {"scientific_config": {"seeds": [2, 3]}}},
                              proximity=pd.DataFrame([{"original_index": r.original_index, "seed": seed, "sscd": 0.2 + 0.6 * seed} for r in records for seed in (0, 1)]))
    law = reference_law_from_atoms(targets, ["a", "b"], weights=[1, 3])
    return sources, records, law, schedule


def test_task_plan_shares_unconditional_bank_and_preserves_independent_recipes(probe_sources):
    sources, records, law, schedule = probe_sources
    config = {"num_loss_seeds": 3, "num_unconditional_loss_seeds": 4, "mean_source": "cached-targets"}
    tasks = probes.plan_probe_tasks(sources, records, law, schedule, config)
    counts = pd.Series([t["table"] for t in tasks]).value_counts().to_dict()
    assert counts == {"gaussian_reference": 3, "forward_loss_draws": 2, "gaussian_conditional": 2}
    assert all(t["record_index"] is None for t in tasks if t["table"] == "gaussian_reference")
    assert all(t["pair"]["target_id"].startswith("image-") for t in tasks if t["record_index"] is not None)
    modified = reference_law_from_atoms(law.support.atoms, ["a", "b"], weights=[3, 1])
    changed = probes.plan_probe_tasks(sources, records, modified, schedule, config)
    first = {t["task_hash"] for t in tasks if t["table"] != "gaussian_reference"}
    assert first == {t["task_hash"] for t in changed if t["table"] != "gaussian_reference"}
    assert {t["task_hash"] for t in tasks if t["table"] == "gaussian_reference"}.isdisjoint({t["task_hash"] for t in changed if t["table"] == "gaussian_reference"})
    expanded = probes.plan_probe_tasks(sources, records, law, schedule, config | {"loss_timesteps": "saved"})
    assert sum(t["table"] == "forward_loss_draws" for t in expanded) == 6
    assert sum(t["table"] == "gaussian_reference" for t in expanded) == 3
    assert sum(t["table"] == "forward_unconditional_loss" for t in expanded) == 3
    assert all("feedback.py" not in str(t["identity"].get("reference_sources", {})) for t in expanded)


def test_law_and_source_worker_inputs_are_files_not_tensor_ipc(probe_sources, tmp_path):
    sources, records, law, _ = probe_sources
    descriptor = probes._persist_worker_inputs(tmp_path / "cache", sources, records, law)
    assert all(isinstance(value, str) for value in descriptor.values())
    restored, reconstructed, schedule = probes._read_worker_inputs(descriptor)
    torch.testing.assert_close(reconstructed.support.weights, law.support.weights, rtol=0, atol=0)
    torch.testing.assert_close(reconstructed.mean_vector, law.mean_vector, rtol=0, atol=0)
    assert reconstructed.law_hash == law.law_hash
    assert [r.original_index for r in restored.selected] == [r.original_index for r in records]
    assert len(schedule["timesteps"]) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for learned-probe numerical worker")
@pytest.mark.parametrize("batch_size", [1, 8])
def test_worker_uses_fixed_gaussian_inputs_all_native_levels_and_resumes(probe_sources, tmp_path, monkeypatch, batch_size):
    sources, records, law, schedule = probe_sources
    law = law.to("cuda:0")
    config = {"num_loss_seeds": 3, "num_unconditional_loss_seeds": 4, "loss_timesteps": "saved", "mean_source": "cached-targets"}
    tasks = probes.plan_probe_tasks(sources, records, law, schedule, config)
    monkeypatch.setattr(probes, "_read_worker_inputs", lambda inputs, **kwargs: (sources, law, schedule))
    gaussian = probes.make_initial_noise([0, 1], [1, 2, 2]).to("cuda:0")
    cached_reads = []

    def initial(sources, record, bank, schedule):
        cached_reads.append(record.original_index)
        return {"state": gaussian, "conditional": torch.zeros_like(gaussian), "unconditional": torch.zeros_like(gaussian),
                "compatible": True, "status": "matched_saved_standard_gaussian_with_recorded_quantization"}

    monkeypatch.setattr(probes, "_cached_initial", initial)
    replica_calls, encoded, evaluated = [], [], []
    components = SimpleNamespace(device=torch.device("cuda:0"), inference_dtype=torch.float32, tokenizer=None, text_encoder=None, unet=None,
                                 device_metadata={"parameter_dtypes": {"unet": ["float32"]}}, package_versions={"torch": "synthetic"})

    def replica(*args):
        replica_calls.append(1)
        return components, SimpleNamespace()

    def encode(prompt, *args):
        encoded.append(prompt)
        return prompt

    def evaluate(samples, *, condition, timestep, **kwargs):
        evaluated.append((condition, timestep, samples.clone()))
        return torch.zeros_like(samples), {"input_quantization_l2": torch.zeros(len(samples), device=samples.device), "prediction_conversion_dtype": "float64"}

    monkeypatch.setattr(probes, "_load_replica", replica)
    monkeypatch.setattr(probes, "encode_prompt_condition", encode)
    monkeypatch.setattr(probes, "_prediction_microbatch", evaluate)
    destination = tmp_path / "cache"
    result = probes._run_probe_worker(inputs={}, tasks=tasks, output_directory=destination, device="cuda:0", batch_size=batch_size, worker_count=1)
    assert not result["failures"]
    assert len(replica_calls) == 1
    assert sorted(encoded) == ["", " shared prompt\n"]
    shared = [t for t in tasks if t["table"] == "gaussian_reference"]
    for task in shared:
        data, _ = probes._task_paths(destination, task)
        frame = pd.read_parquet(data)
        assert frame.seed.tolist() == [0, 1]
        assert frame.input_source.eq("gaussian_probe").all()
        assert frame.reference_law_hash.eq(law.law_hash).all()
        assert frame.terminal_sscd.isna().all()
    for timestep in (5, 1):
        gaussian_batches = [z for prompt, t, z in evaluated if prompt == "" and t == timestep
                            and any(torch.equal(z[0], row.double()) for row in gaussian)]
        assert gaussian_batches
        torch.testing.assert_close(torch.cat(gaussian_batches), gaussian.double(), rtol=0, atol=0)
    initial_conditional = next(t for t in tasks if t["table"] == "gaussian_conditional" and t["level"]["step_index"] == 0)
    frame = pd.read_parquet(probes._task_paths(destination, initial_conditional)[0])
    assert frame.terminal_sscd_matched_initialization.all()
    before = len(evaluated)
    repeated = probes._run_probe_worker(inputs={}, tasks=list(reversed(tasks)), output_directory=destination, device="cuda:0", batch_size=1, worker_count=1)
    assert not repeated["failures"] and len(evaluated) == before
    assert len(replica_calls) == 1
    # A damaged scalar task is recomputed without discarding completed siblings.
    broken = next(t for t in tasks if t["table"] == "forward_loss_draws")
    path, marker = probes._task_paths(destination, broken)
    preserved = {str(probes._task_paths(destination, t)[0]): file_sha256(probes._task_paths(destination, t)[0]) for t in tasks if t != broken}
    path.write_bytes(b"interrupted derived shard")
    repaired = probes._run_probe_worker(inputs={}, tasks=tasks, output_directory=destination, device="cuda:0", batch_size=8, worker_count=1)
    assert not repaired["failures"]
    assert probes._valid_task(destination, broken)
    assert read_json(marker)["rows"] == 3
    assert preserved == {name: file_sha256(name) for name in preserved}


def test_probe_loader_is_opt_in_pinned_denoiser_only():
    loaded = []

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            loaded.append(("tokenizer", model_id, kwargs))
            return cls()

    class Component(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            loaded.append((kwargs["subfolder"], model_id, kwargs))
            return cls()

    runtime = RuntimeSelection(torch.device("cpu"), torch.float32, None)
    result = load_denoiser_components(model_id="synthetic/model", model_revision="a" * 40, runtime=runtime,
                                     component_classes={"tokenizer": Tokenizer, "text_encoder": Component, "unet": Component})
    assert [item[0] for item in loaded] == ["tokenizer", "text_encoder", "unet"]
    assert all(item[2]["revision"] == "a" * 40 for item in loaded)
    assert not result.unet.training and not result.text_encoder.training
    assert all(not p.requires_grad for p in result.unet.parameters())
    assert result.device_metadata["parameter_dtypes"]["unet"] == ["float32"]
    assert not hasattr(result, "vae")


def test_nonunit_cached_initialization_is_not_rescaled_into_a_theorem_probe(monkeypatch):
    gaussian = torch.ones(2, 1, 2, 2)
    original = gaussian * 1.7
    record = SimpleNamespace(original_index="pair")
    monkeypatch.setattr(probes, "load_record", lambda *a, **k: (original.clone(), torch.zeros_like(original), torch.zeros_like(original), torch.zeros(1, 2, 2)))
    result = probes._cached_initial(SimpleNamespace(experiment=None), record, gaussian, {"init_noise_sigma": 1.7})
    assert not result["compatible"]
    assert "fresh_probe_required" in result["status"]
    torch.testing.assert_close(result["state"], original, rtol=0, atol=0)
    torch.testing.assert_close(gaussian, torch.ones_like(gaussian), rtol=0, atol=0)


@pytest.mark.parametrize("count,batch_size", [(20, 8), (2, 1), (16, 8), (3, 8)])
def test_cached_prediction_batches_keep_original_bank_after_consumer_rebinding(count, batch_size):
    gaussian = torch.arange(count * 4, dtype=torch.float32).reshape(count, 1, 2, 2) / 7
    samples = gaussian.half().double()
    original = (gaussian * 3 + 1).half()
    epsilon = original.clone()
    batches = probes.cached_prediction_batches(samples, epsilon, gaussian, batch_size=batch_size)
    observed, ranges = [], []
    # Reuse the consumer variable exactly as the worker does. This exposed the
    # old generator-expression closure bug only after its first microbatch.
    for start, stop, epsilon, precision in batches:
        ranges.append((start, stop))
        observed.append(epsilon)
        assert epsilon.shape == (stop - start, 1, 2, 2)
        assert precision["input_quantization_l2"].shape == (stop - start,)
        assert precision["conversion_input_quantization_l2"].shape == (stop - start,)
    assert ranges == [(start, min(count, start + batch_size)) for start in range(0, count, batch_size)]
    torch.testing.assert_close(torch.cat(observed), original.double(), rtol=0, atol=0)
    torch.testing.assert_close(original, (gaussian * 3 + 1).half(), rtol=0, atol=0)


def test_cached_prediction_batches_reject_incomplete_bank_before_reduction():
    with pytest.raises(probes.TheoryError, match="identical batch/latent shapes"):
        list(probes.cached_prediction_batches(torch.zeros(20, 1, 2, 2), torch.zeros(8, 1, 2, 2),
                                             torch.zeros(20, 1, 2, 2), batch_size=8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU probe reductions")
def test_probe_predictions_and_metric_tensors_remain_on_cuda():
    class Denoiser(torch.nn.Module):
        def forward(self, sample, timestep, **kwargs):
            return (torch.zeros_like(sample),)

    class Scheduler:
        config = {"prediction_type": "epsilon"}
        alphas_cumprod = torch.tensor([0.36], dtype=torch.float64)

        def scale_model_input(self, sample, timestep):
            return sample

    samples = torch.arange(8, dtype=torch.float64).reshape(2, 1, 2, 2) / 10
    components = SimpleNamespace(device=torch.device("cuda:0"), inference_dtype=torch.float32, unet=Denoiser().cuda())
    prediction, precision = probes._prediction_microbatch(
        samples, condition=torch.ones(1, 2, 3, device="cuda:0"), timestep=0,
        components=components, scheduler=Scheduler(),
    )
    assert prediction.device.type == "cuda"
    assert all(value.device.type == "cuda" for value in precision.values() if isinstance(value, torch.Tensor))
    cuda_samples = samples.cuda()
    values = probes.conditional_gaussian_metrics(torch.zeros_like(samples[0]), cuda_samples, prediction, .6, .8)
    assert all(value.device.type == "cuda" for value in values.values() if isinstance(value, torch.Tensor))
    expected = probes.conditional_gaussian_metrics(torch.zeros_like(samples[0]), samples, prediction.cpu(), .6, .8)
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value.cpu(), expected[name])
    batches = list(probes.cached_prediction_batches(cuda_samples, prediction, cuda_samples, batch_size=1))
    for _, _, epsilon, cached_precision in batches:
        assert epsilon.device.type == "cuda"
        assert all(value.device.type == "cuda" for value in cached_precision.values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"))])
def test_reference_mean_comparison_adds_exact_bank_offset_to_the_analytical_bound(device):
    z = torch.tensor([[1., -2.], [-3., 4.]], dtype=torch.float64, device=device)
    prediction = torch.zeros_like(z)
    posterior = torch.zeros_like(z)  # A declared singleton law at the origin.
    bank_mean = torch.zeros(2, dtype=torch.float64, device=device)
    selected_mean = torch.tensor([3., 4.], dtype=torch.float64, device=device)
    values = gaussian_reference_metrics(z, prediction, posterior, selected_mean,
                                        .6, .8, 0., bank_mean=bank_mean)
    bank_values = gaussian_reference_metrics(z, prediction, posterior, bank_mean, .6, .8, 0.)
    assert values["bank_reference_bound_l2"].eq(0).all()
    assert torch.isneginf(values["bank_reference_bound_log_l2"]).all()
    torch.testing.assert_close(values["reference_bound_l2"], torch.full((2,), 5., device=device, dtype=torch.float64))
    torch.testing.assert_close(values["reference_bound_log_l2"], values["reference_bound_l2"].log())
    torch.testing.assert_close(values["reference_mean_error_l2"], values["mean_offset_l2"])
    assert values["reference_to_bank_mean_error_l2"].eq(0).all()
    assert values["bank_mean_norm_l2"].eq(0).all()
    assert values["mean_norm_l2"].eq(5).all()
    for field in ("unconditional_reference_error_l2", "learned_zero_error_l2", "reference_zero_error_l2"):
        torch.testing.assert_close(values[field], bank_values[field], rtol=0, atol=0)
    assert values["baseline_slack_l2"].ge(-1e-12).all()
    assert values["baseline_squared_identity_residual"].abs().lt(1e-12).all()
    assert values["reference_bound_mean_offset_included"] is True
    assert values["mean_offset_rmse"].device == z.device
