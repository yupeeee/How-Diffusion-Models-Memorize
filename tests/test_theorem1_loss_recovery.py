"""Focused model-free contracts for the pair-level Theorem 1 experiment."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PathCollection, QuadMesh
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import theorem1_loss_recovery as experiment


EXPECTED_COLUMNS = (
    "record_id",
    "selection_strategy",
    "selection_hash",
    "model_name",
    "scheduler_name",
    "guidance_scale",
    "num_inference_steps",
    "timestep",
    "alpha_t",
    "sigma_t",
    "snr_t",
    "latent_dimension",
    "num_loss_seeds",
    "loss_seed",
    "generation_seeds",
    "num_generation_seeds",
    "conditional_loss",
    "normalized_loss_mse",
    "normalized_loss_rmse",
    "recovery_mse",
    "recovery_rmse",
    "mean_target_sscd",
    "status",
    "error",
)
SELECTION_HASH = "c" * 64


def _selection(
    *,
    included: Sequence[str],
    excluded: Sequence[str] = (),
    strategy: str = "gmm",
    selection_hash: str = SELECTION_HASH,
) -> SimpleNamespace:
    return SimpleNamespace(
        included_indices=frozenset(included),
        excluded_indices=frozenset(excluded),
        selection_strategy=strategy,
        sha256=selection_hash,
    )


def _prediction_samples(
    args: Sequence[object], kwargs: Mapping[str, object]
) -> torch.Tensor:
    assert not args
    value = kwargs["samples"]
    if not isinstance(value, torch.Tensor):
        raise AssertionError("conditional predictor did not receive a sample tensor")
    return value


def test_cli_schema_defaults_and_output_names_are_exact() -> None:
    parser = experiment.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"--help", "-h"}
    }
    assert options == {
        "--model",
        "--selection-strategy",
        "--scheduler",
        "--g",
        "--T",
        "--num-loss-seeds",
        "--loss-seed",
        "--N",
        "--sample-batch-size",
        "--max-records",
        "--output-dir",
        "--device",
        "--plot",
    }

    arguments = parser.parse_args([])
    assert arguments.model == "sdv1"
    assert arguments.selection_strategy == "spearman"
    assert arguments.scheduler == "ddim"
    assert arguments.g == pytest.approx(7.5)
    assert arguments.num_inference_steps == 50
    assert arguments.num_loss_seeds == 20
    assert arguments.loss_seed == 0
    assert arguments.num_seeds == 20
    assert arguments.sample_batch_size > 0
    assert arguments.max_records is None
    assert arguments.output_dir is None
    assert arguments.device == "auto"
    assert arguments.plot is False
    assert parser.parse_args(["--plot"]).plot is True
    assert (
        parser.parse_args(["--selection-strategy", "spearman"]).selection_strategy
        == "spearman"
    )
    assert (
        parser.parse_args(["--selection-strategy", "gmm-evidence"])
        .selection_strategy
        == "gmm-evidence"
    )
    assert parser.parse_args(["--device", "CUDA:2"]).device == "cuda:2"
    expected_output_directories = {
        model_name: (
            experiment.ROOT
            / "outputs"
            / f"{model_name}_ddim_g7.5_T50_N20"
            / "theorem1_loss_recovery"
            / "gmm"
            / SELECTION_HASH
        )
        for model_name in ("sdv1", "sdv2", "realvis")
    }
    assert {
        model_name: experiment._default_output_directory(
            model_name, "ddim", 7.5, 50, 20, "gmm", SELECTION_HASH
        )
        for model_name in expected_output_directories
    } == expected_output_directories
    assert len(set(expected_output_directories.values())) == 3
    assert experiment._default_output_directory(
        "sdv1", "ddpm", 3.25, 12, 3, "spearman", "d" * 64
    ) == (
        experiment.ROOT
        / "outputs"
        / "sdv1_ddpm_g3.25_T12_N3"
        / "theorem1_loss_recovery"
        / "spearman"
        / ("d" * 64)
    )
    assert experiment._default_output_directory(
        "sdv1", "ddpm", 3.25, 12, 3, "gmm-evidence", "e" * 64
    ) == (
        experiment.ROOT
        / "outputs"
        / "sdv1_ddpm_g3.25_T12_N3"
        / "theorem1_loss_recovery"
        / "gmm-evidence"
        / ("e" * 64)
    )

    smoke = parser.parse_args(
        [
            "--max-records",
            "2",
            "--scheduler",
            "ddpm",
            "--g",
            "3.25",
            "--T",
            "12",
            "--num-loss-seeds",
            "2",
            "--N",
            "2",
        ]
    )
    assert (smoke.scheduler, smoke.g, smoke.num_inference_steps) == (
        "ddpm",
        pytest.approx(3.25),
        12,
    )
    assert (
        smoke.max_records,
        smoke.num_loss_seeds,
        smoke.num_seeds,
    ) == (2, 2, 2)
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-loss-seeds", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--scheduler", "euler"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--g", "nan"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--T", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss-seed", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--N", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-loss", "2"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--plot-only"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "cuda:-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--selection-strategy", "kmeans"])

    assert experiment.CSV_COLUMNS == EXPECTED_COLUMNS
    assert experiment.CSV_NAME == "theorem1_loss_recovery.csv"
    assert experiment.FIGURE_FILENAMES == (
        "theorem1_loss_recovery.png",
        "theorem1_loss_recovery.pdf",
    )


def test_run_experiment_requires_explicit_device() -> None:
    assert (
        inspect.signature(experiment.run_experiment).parameters["device"].default
        is inspect.Parameter.empty
    )


def test_output_directory_accepts_only_the_csv_and_both_figure_formats(
    tmp_path: Path,
) -> None:
    output = tmp_path / "theorem1"
    output.mkdir()
    for filename in (experiment.CSV_NAME, *experiment.FIGURE_FILENAMES):
        (output / filename).write_bytes(b"existing artifact")

    assert experiment._prepare_output_directory(output) == output.resolve()

    unexpected = output / "unrelated.log"
    unexpected.write_text("unrelated", encoding="utf-8")
    with pytest.raises(
        experiment.ExperimentError,
        match="output directory contains unexpected artifacts: unrelated.log",
    ):
        experiment._prepare_output_directory(output)

    invalid_output = tmp_path / "invalid-theorem1"
    invalid_output.mkdir()
    invalid_name = invalid_output / experiment.FIGURE_FILENAMES[0]
    invalid_name.mkdir()
    with pytest.raises(
        experiment.ExperimentError,
        match=(
            "output directory contains unexpected artifacts: theorem1_loss_recovery.png"
        ),
    ):
        experiment._prepare_output_directory(invalid_output)


def test_nondefault_generation_contract_uses_exact_sampler_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule_path = tmp_path / "schedule.pt"
    schedule_path.write_bytes(b"cached schedule")
    paths = SimpleNamespace(
        schedule=schedule_path,
        run_directory=tmp_path / "logs/sdv1_ddpm_g3.25_T12_N3/experiment_S0_N3",
    )
    path_requests: list[tuple[Path, dict[str, object]]] = []

    def fake_generation_paths(root: Path, **kwargs: object) -> SimpleNamespace:
        path_requests.append((root, dict(kwargs)))
        return paths

    scheduler_config = {
        "_class_name": "DDPMScheduler",
        "prediction_type": "v_prediction",
    }
    science = {
        "model_cli_name": "sdv1",
        "dataset_model": "sdv1",
        "model_id": "synthetic-model",
        "guidance_scale": 3.25,
        "num_inference_steps": 12,
        "num_seeds": 3,
        "seeds": [0, 1, 2],
        "trajectory_order": "noise_to_image",
        "stored_prediction_type": "epsilon",
        "target_latent_definition": experiment.TARGET_LATENT_DEFINITION,
        "target_preprocessing": {"policy": "test"},
        "latent_shape": [4, 8, 8],
        "scientific_tensor_storage": {
            "dtype": "float16",
            "device": "cpu",
            "layout": "contiguous",
        },
        "scheduler": {"name": "ddpm", "config": scheduler_config},
    }
    schedule_payload = {
        "timesteps": torch.arange(11, -1, -1, dtype=torch.int64),
        "init_noise_sigma": 1.0,
        "scheduler_config": scheduler_config,
    }
    sscd_requests: list[tuple[object, Mapping[str, object], str, tuple[int, ...]]] = []

    def fake_sscd_configuration(
        sscd_paths: object,
        *,
        science: Mapping[str, object],
        scientific_hash: str,
        seeds: Sequence[int],
    ) -> tuple[None, None, str]:
        sscd_requests.append(
            (sscd_paths, science, scientific_hash, tuple(int(seed) for seed in seeds))
        )
        return None, None, "synthetic missing SSCD"

    monkeypatch.setattr(experiment, "generation_paths", fake_generation_paths)
    monkeypatch.setattr(
        experiment,
        "require_generation_run",
        lambda observed: (
            {
                "scientific_config": science,
                "scientific_config_hash": "science-hash",
            }
            if observed is paths
            else None
        ),
    )
    monkeypatch.setattr(
        experiment,
        "get_model_spec",
        lambda _name: SimpleNamespace(
            dataset_model="sdv1",
            model_id="synthetic-model",
            resolution=64,
        ),
    )
    monkeypatch.setattr(
        experiment,
        "target_preprocessing_policy",
        lambda _resolution: {"policy": "test"},
    )
    monkeypatch.setattr(
        experiment,
        "safe_torch_load",
        lambda observed: (
            schedule_payload
            if observed == schedule_path
            else pytest.fail(f"unexpected tensor load: {observed}")
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_load_sscd_configuration",
        fake_sscd_configuration,
    )

    contract = experiment._load_generation_contract(
        tmp_path,
        "sdv1",
        "ddpm",
        3.25,
        12,
        3,
    )

    assert path_requests == [
        (
            tmp_path.resolve(),
            {
                "model_name": "sdv1",
                "scheduler_name": "ddpm",
                "guidance_scale": 3.25,
                "num_inference_steps": 12,
                "num_seeds": 3,
                "seed_start": 0,
            },
        )
    ]
    assert contract.paths is paths
    assert contract.seeds == (0, 1, 2)
    assert contract.stored_dtype is torch.float16
    assert contract.scheduler_config == scheduler_config
    assert contract.sscd_error == "synthetic missing SSCD"
    assert len(sscd_requests) == 1
    assert sscd_requests[0][1:] == (
        science,
        "science-hash",
        (0, 1, 2),
    )


def test_mean_sscd_does_not_depend_on_preview_sensitive_generation_marker_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = tmp_path / "7.json"
    score_path = tmp_path / "7.pt"
    marker_path.write_text("{}", encoding="utf-8")
    score_path.write_bytes(b"fixed score tensor")
    configuration_hash = "b" * 64
    scientific_hash = "a" * 64
    latent_hash = "d" * 64
    score_hash = "e" * 64
    metadata = {
        "original_index": "7",
        "record_id": "sdv1-0007",
        "source_row_number": 8,
        "prompt_raw": "selected prompt",
        "target_image_sha256": "f" * 64,
    }
    marker = {
        **metadata,
        "model_id": "model",
        "model_revision": "revision",
        "num_seeds": 2,
        "seeds": [0, 1],
        "similarity": experiment.SCORE_DEFINITION,
        "sscd_configuration_hash": configuration_hash,
        "generation_scientific_config_hash": scientific_hash,
        "generation_latent_sha256": latent_hash,
        "generation_record_sha256": "outdated-preview-audit-hash",
        "score_sha256": score_hash,
    }
    contract = SimpleNamespace(
        sscd_error=None,
        sscd_configuration={"seeds": [0, 1]},
        sscd_configuration_hash=configuration_hash,
        scientific_hash=scientific_hash,
        science={"model_id": "model", "model_revision": "revision"},
        sscd_paths=SimpleNamespace(
            marker_path=lambda index: (
                marker_path if index == "7" else pytest.fail("wrong marker index")
            ),
            score_path=lambda index: (
                score_path if index == "7" else pytest.fail("wrong score index")
            ),
        ),
        paths=SimpleNamespace(
            record_path=lambda _index: pytest.fail(
                "preview-sensitive generation record must not be hashed"
            )
        ),
    )
    monkeypatch.setattr(experiment, "read_json", lambda path: marker)
    monkeypatch.setattr(
        experiment,
        "file_sha256",
        lambda path: (
            score_hash
            if path == score_path
            else pytest.fail(f"unexpected hash request: {path}")
        ),
    )
    monkeypatch.setattr(
        experiment,
        "safe_torch_load",
        lambda path: (
            torch.tensor([0.2, 0.6], dtype=torch.float32)
            if path == score_path
            else pytest.fail(f"unexpected tensor load: {path}")
        ),
    )

    result = experiment._load_mean_target_sscd(
        contract=contract,
        metadata=metadata,
        generation_seeds=(0, 1),
        generation_marker={"tensor_file_sha256": {"latent": latent_hash}},
    )

    assert result == pytest.approx(0.4)


def test_active_scheduler_uses_requested_name_and_inference_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_timesteps = torch.arange(11, -1, -1, dtype=torch.int64)
    timestep_calls: list[tuple[int, torch.device]] = []

    def set_timesteps(count: int, *, device: torch.device) -> None:
        timestep_calls.append((count, device))
        scheduler.timesteps = expected_timesteps.to(device=device)

    scheduler = SimpleNamespace(
        timesteps=torch.empty(0, dtype=torch.int64),
        init_noise_sigma=1.0,
        set_timesteps=set_timesteps,
    )
    scheduler_config = {
        "_class_name": "DDPMScheduler",
        "prediction_type": "sample",
    }
    original_scheduler = object()
    build_requests: list[tuple[object, str]] = []

    def fake_build_scheduler(original: object, name: str) -> SimpleNamespace:
        build_requests.append((original, name))
        return SimpleNamespace(scheduler=scheduler, config=scheduler_config)

    monkeypatch.setattr(experiment, "build_scheduler", fake_build_scheduler)
    contract = SimpleNamespace(
        schedule_payload={"timesteps": expected_timesteps},
        scheduler_config=scheduler_config,
        init_noise_sigma=1.0,
    )
    components = SimpleNamespace(
        original_scheduler=original_scheduler,
        device=torch.device("cpu"),
    )

    active = experiment._validate_active_scheduler(
        components,
        contract,
        "ddpm",
        12,
    )

    assert active is scheduler
    assert build_requests == [(original_scheduler, "ddpm")]
    assert timestep_calls == [(12, torch.device("cpu"))]


def test_conditional_loss_uses_sum_then_mean_then_square_root_and_checks_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.tensor([[[1.0, -1.0], [0.5, 2.0]]], dtype=torch.float32)
    loss_noise = torch.tensor(
        [
            [[[1.0, 2.0], [0.0, 0.0]]],
            [[[-1.0, 0.0], [2.0, 2.0]]],
        ],
        dtype=torch.float32,
    )
    predictor_inputs: list[torch.Tensor] = []

    def zero_prediction(*args: object, **kwargs: object) -> torch.Tensor:
        samples = _prediction_samples(args, kwargs)
        predictor_inputs.append(samples.detach().cpu().clone())
        return torch.zeros_like(samples)

    monkeypatch.setattr(experiment, "predict_conditional_epsilon", zero_prediction)
    components = SimpleNamespace(
        device=torch.device("cpu"),
        inference_dtype=torch.float32,
        unet=object(),
    )
    alpha_t, sigma_t, snr_t = 0.6, 0.8, 0.5625
    result = experiment._measure_conditional_loss(
        target_latent=target,
        loss_noise=loss_noise,
        condition=torch.ones((1, 1, 1), dtype=torch.float32),
        timestep=876,
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        snr_t=snr_t,
        components=components,
        scheduler=object(),
        sample_batch_size=2,
    )

    expected_q = alpha_t * target.unsqueeze(0) + sigma_t * loss_noise
    assert len(predictor_inputs) == 1
    torch.testing.assert_close(predictor_inputs[0], expected_q)
    per_seed_sum = loss_noise.double().square().flatten(1).sum(1)
    assert per_seed_sum.tolist() == [5.0, 9.0]
    expected_mse = 7.0 / (target.numel() * snr_t)
    expected_rmse = math.sqrt(expected_mse)
    assert result.conditional_loss == pytest.approx(7.0)
    assert result.normalized_loss_mse == pytest.approx(expected_mse)
    assert result.normalized_loss_rmse == pytest.approx(expected_rmse)
    mean_seedwise_rmse = float(
        torch.sqrt(per_seed_sum / (target.numel() * snr_t)).mean()
    )
    assert expected_rmse != pytest.approx(mean_seedwise_rmse)

    first_reconstruction = (
        expected_q[0] - sigma_t * torch.zeros_like(loss_noise[0])
    ) / alpha_t
    identity_lhs = float(
        (first_reconstruction - target).double().square().sum() / target.numel()
    )
    identity_rhs = float(per_seed_sum[0] / (target.numel() * snr_t))
    assert identity_lhs == pytest.approx(identity_rhs)

    with pytest.raises(
        (ValueError, RuntimeError), match="identity|SNR|snr|alpha|sigma"
    ):
        experiment._measure_conditional_loss(
            target_latent=target,
            loss_noise=loss_noise,
            condition=torch.ones((1, 1, 1), dtype=torch.float32),
            timestep=876,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            snr_t=0.5,
            components=components,
            scheduler=object(),
            sample_batch_size=2,
        )


def test_generation_initial_latents_match_canonical_sampler_and_recovery_has_no_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_noise = torch.tensor(
        [
            [[[0.25, 0.50]]],
            [[[0.50, 1.00]]],
        ],
        dtype=torch.float32,
    )
    noise_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def canonical_noise(
        seeds: Sequence[int], latent_shape: Sequence[int]
    ) -> torch.Tensor:
        noise_calls.append(
            (
                tuple(int(seed) for seed in seeds),
                tuple(int(size) for size in latent_shape),
            )
        )
        return stored_noise.clone()

    monkeypatch.setattr(experiment, "make_initial_noise", canonical_noise)
    reconstructed = experiment._reconstruct_generation_initial_latents(
        seeds=(3, 7),
        latent_shape=(1, 1, 2),
        stored_dtype=torch.float32,
        init_noise_sigma=1.0,
    )
    assert noise_calls == [((3, 7), (1, 1, 2))]
    torch.testing.assert_close(reconstructed, stored_noise)

    predictor_inputs: list[torch.Tensor] = []

    def zero_prediction(*args: object, **kwargs: object) -> torch.Tensor:
        samples = _prediction_samples(args, kwargs)
        predictor_inputs.append(samples.detach().cpu().clone())
        return torch.zeros_like(samples)

    def forbidden_step(*_: object, **__: object) -> None:
        raise AssertionError("the loss-recovery experiment must not denoise")

    monkeypatch.setattr(experiment, "predict_conditional_epsilon", zero_prediction)
    target = torch.zeros((1, 1, 2), dtype=torch.float32)
    components = SimpleNamespace(
        device=torch.device("cpu"),
        inference_dtype=torch.float32,
        unet=object(),
    )
    result = experiment._measure_recovery(
        target_latent=target,
        generation_initial_latents=reconstructed,
        condition=torch.ones((1, 1, 1), dtype=torch.float32),
        timestep=876,
        alpha_t=0.5,
        sigma_t=math.sqrt(0.75),
        components=components,
        scheduler=SimpleNamespace(step=forbidden_step),
        sample_batch_size=2,
    )

    assert len(predictor_inputs) == 1
    torch.testing.assert_close(predictor_inputs[0], stored_noise)
    per_seed_mse = torch.tensor([0.625, 2.5], dtype=torch.float64)
    expected_mse = float(per_seed_mse.mean())
    expected_rmse = math.sqrt(expected_mse)
    assert result.recovery_mse == pytest.approx(expected_mse)
    assert result.recovery_rmse == pytest.approx(expected_rmse)
    assert expected_rmse != pytest.approx(float(torch.sqrt(per_seed_mse).mean()))


def test_cuda_oom_halving_preserves_loss_and_recovery_sample_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.zeros((1, 1, 1), dtype=torch.float32)
    samples = torch.arange(1, 6, dtype=torch.float32).reshape(5, 1, 1, 1)
    components = SimpleNamespace(
        device=torch.device("cuda:0"),
        inference_dtype=torch.float32,
    )
    attempted_batches: list[torch.Tensor] = []
    successful_samples: list[torch.Tensor] = []
    cache_releases: list[None] = []

    def oom_above_two(
        conceptual_samples: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        attempted_batches.append(conceptual_samples.detach().clone())
        if conceptual_samples.shape[0] > 2:
            raise RuntimeError("synthetic CUDA out of memory")
        successful_samples.append(conceptual_samples.detach().clone())
        return torch.zeros_like(conceptual_samples)

    monkeypatch.setattr(experiment, "_prediction_batch", oom_above_two)
    monkeypatch.setattr(
        experiment.torch.cuda,
        "empty_cache",
        lambda: cache_releases.append(None),
    )

    loss = experiment._measure_conditional_loss(
        target_latent=target,
        loss_noise=samples,
        condition=torch.ones((1, 1, 1), dtype=torch.float32),
        timestep=876,
        alpha_t=0.6,
        sigma_t=0.8,
        snr_t=0.5625,
        components=components,
        scheduler=object(),
        sample_batch_size=4,
    )

    assert [batch.shape[0] for batch in attempted_batches] == [4, 2, 2, 1]
    torch.testing.assert_close(
        torch.cat(successful_samples),
        0.8 * samples,
    )
    assert loss.conditional_loss == pytest.approx(11.0)
    assert loss.normalized_loss_mse == pytest.approx(11.0 / 0.5625)
    assert loss.normalized_loss_rmse == pytest.approx(math.sqrt(11.0 / 0.5625))

    attempted_batches.clear()
    successful_samples.clear()
    recovery = experiment._measure_recovery(
        target_latent=target,
        generation_initial_latents=samples,
        condition=torch.ones((1, 1, 1), dtype=torch.float32),
        timestep=876,
        alpha_t=0.6,
        sigma_t=0.8,
        components=components,
        scheduler=object(),
        sample_batch_size=4,
    )

    assert [batch.shape[0] for batch in attempted_batches] == [4, 2, 2, 1]
    torch.testing.assert_close(torch.cat(successful_samples), samples)
    assert recovery.recovery_mse == pytest.approx(11.0 / 0.36)
    assert recovery.recovery_rmse == pytest.approx(math.sqrt(11.0 / 0.36))
    assert len(cache_releases) == 2


def test_non_cuda_out_of_memory_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def fail(samples: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        attempts.append(samples.shape[0])
        raise RuntimeError("synthetic out of memory")

    monkeypatch.setattr(experiment, "_prediction_batch", fail)
    monkeypatch.setattr(
        experiment.torch.cuda,
        "empty_cache",
        lambda: pytest.fail("CPU failure must not touch the CUDA allocator"),
    )
    components = SimpleNamespace(device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="synthetic out of memory"):
        tuple(
            experiment._prediction_batches(
                torch.zeros((3, 1, 1, 1)),
                condition=torch.zeros((1, 1, 1)),
                timestep=1,
                components=components,
                scheduler=object(),
                initial_batch_size=3,
            )
        )
    assert attempts == [3]


def test_unused_vae_is_offloaded_after_validation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfers: list[dict[str, object]] = []
    cache_releases: list[None] = []

    class FakeVAE:
        def to(self, **kwargs: object) -> FakeVAE:
            transfers.append(dict(kwargs))
            return self

    components = SimpleNamespace(
        device=torch.device("cuda:2"),
        vae=FakeVAE(),
    )
    monkeypatch.setattr(
        experiment.torch.cuda,
        "empty_cache",
        lambda: cache_releases.append(None),
    )

    experiment._offload_unused_vae(components)

    assert transfers == [{"device": torch.device("cpu"), "dtype": torch.float32}]
    assert cache_releases == [None]


def test_loss_noise_stream_is_reproducible_and_independent_of_generation_seeds() -> (
    None
):
    generation_seeds = tuple(range(20))
    first_seeds = experiment._loss_sample_seeds(
        0,
        3,
        forbidden_seeds=generation_seeds,
    )
    repeated_seeds = experiment._loss_sample_seeds(
        0,
        3,
        forbidden_seeds=generation_seeds,
    )
    different_seeds = experiment._loss_sample_seeds(
        1,
        3,
        forbidden_seeds=generation_seeds,
    )

    assert first_seeds == repeated_seeds
    assert len(first_seeds) == len(set(first_seeds)) == 3
    assert set(first_seeds).isdisjoint(generation_seeds)
    assert different_seeds != first_seeds
    assert set(different_seeds).isdisjoint(generation_seeds)

    first_noise = experiment._make_loss_noise(
        num_loss_seeds=3,
        latent_shape=(1, 2, 2),
        loss_seed=0,
        forbidden_seeds=generation_seeds,
    )
    repeated_noise = experiment._make_loss_noise(
        num_loss_seeds=3,
        latent_shape=(1, 2, 2),
        loss_seed=0,
        forbidden_seeds=generation_seeds,
    )
    canonical_loss_noise = experiment.make_initial_noise(first_seeds, (1, 2, 2))
    canonical_generation_noise = experiment.make_initial_noise((0, 1, 2), (1, 2, 2))
    torch.testing.assert_close(first_noise, repeated_noise)
    torch.testing.assert_close(first_noise, canonical_loss_noise)
    assert not torch.equal(first_noise, canonical_generation_noise)


class _FakeProgress:
    instances: list[_FakeProgress] = []

    def __init__(self, iterable: object = None, **kwargs: object) -> None:
        self.iterable = iterable
        self.kwargs = kwargs
        self.updated = 0
        self.postfixes: list[dict[str, object]] = []
        type(self).instances.append(self)

    def __enter__(self) -> _FakeProgress:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __iter__(self):  # type: ignore[no-untyped-def]
        if self.iterable is None:
            return
        for value in self.iterable:  # type: ignore[union-attr]
            yield value
            self.updated += 1

    def update(self, amount: int = 1) -> None:
        self.updated += amount

    def set_postfix(
        self,
        ordered_dict: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self.postfixes.append(values)

    def close(self) -> None:
        return None


class _EvalModule:
    def __init__(self, **config: object) -> None:
        self.config = SimpleNamespace(**config)
        self.training = True

    def eval(self) -> _EvalModule:
        self.training = False
        return self


def test_mocked_realvis_pair_run_keeps_alias_streams_seed_alignment_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "record_id": "realisticvision-0000",
            "model_name": "realisticvision",
            "prompt": "first prompt",
            "prompt_raw": "first prompt",
            "image": object(),
            "original_index": "100",
            "source_row_number": 10,
            "target_image_sha256": "a" * 64,
        },
        {
            "record_id": "realisticvision-discarded",
            "model_name": "realisticvision",
            "prompt": "discarded prompt",
            "prompt_raw": "discarded prompt",
            "image": object(),
            "original_index": "101",
            "source_row_number": 11,
            "target_image_sha256": "b" * 64,
        },
        {
            "record_id": "realisticvision-0001",
            "model_name": "realisticvision",
            "prompt": "third prompt",
            "prompt_raw": "third prompt",
            "image": object(),
            "original_index": "102",
            "source_row_number": 12,
            "target_image_sha256": "c" * 64,
        },
    ]
    accessed_dataset_positions: list[int] = []

    class FakeDataset:
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, object]:
            accessed_dataset_positions.append(index)
            return dict(records[index])

        def iter_metadata(self):  # type: ignore[no-untyped-def]
            for record in records:
                metadata = dict(record)
                metadata["image_available"] = True
                metadata["image"] = None
                yield metadata

    alpha_bar = torch.tensor(
        [0.9, 0.8, 0.7, 0.36, 0.2, 0.1, 0.05, 0.01], dtype=torch.float64
    )
    scheduler = SimpleNamespace(
        alphas_cumprod=alpha_bar,
        timesteps=torch.tensor([3, 2, 1], dtype=torch.int64),
        init_noise_sigma=1.0,
        config=SimpleNamespace(num_train_timesteps=len(alpha_bar)),
        set_timesteps=lambda *_args, **_kwargs: None,
    )
    components = SimpleNamespace(
        device=torch.device("cpu"),
        inference_dtype=torch.float32,
        unet=_EvalModule(in_channels=1, sample_size=2),
        vae=_EvalModule(),
        tokenizer=object(),
        text_encoder=_EvalModule(),
        scheduler=scheduler,
        device_metadata={},
    )
    generation_contract = SimpleNamespace(
        latent_shape=(1, 2, 2),
        stored_dtype=torch.float32,
        init_noise_sigma=1.0,
        seeds=tuple(range(3)),
        scheduler_config={},
        science={
            "model_id": "SG161222/Realistic_Vision_V2.0",
            "model_revision": "a" * 40,
            "vae_id": "stabilityai/sd-vae-ft-mse",
            "vae_revision": "b" * 40,
        },
        paths=SimpleNamespace(),
        sscd_paths=SimpleNamespace(),
    )

    requested_dataset_models: list[str] = []

    def fake_dataset(_root: Path, dataset_model: str, **_kwargs: object) -> FakeDataset:
        requested_dataset_models.append(dataset_model)
        return FakeDataset()

    monkeypatch.setattr(experiment, "WebsterDataset", fake_dataset)
    selection_requests: list[dict[str, object]] = []

    def load_selection(_root: Path, **kwargs: object) -> SimpleNamespace:
        selection_requests.append(dict(kwargs))
        return _selection(
            included=("100", "102"),
            excluded=("101",),
            strategy="spearman",
        )

    monkeypatch.setattr(experiment, "_load_frozen_selection", load_selection)
    loaded_contract_arguments: list[tuple[object, ...]] = []

    def load_contract(*args: object, **_kwargs: object) -> object:
        loaded_contract_arguments.append(args)
        return generation_contract

    monkeypatch.setattr(experiment, "_load_generation_contract", load_contract)
    active_scheduler_arguments: list[tuple[object, ...]] = []

    def validate_scheduler(*args: object, **_kwargs: object) -> object:
        active_scheduler_arguments.append(args)
        return scheduler

    monkeypatch.setattr(experiment, "_validate_active_scheduler", validate_scheduler)
    loaded_runtimes: list[object] = []

    def load_components(*_args: object, **kwargs: object) -> object:
        loaded_runtimes.append(kwargs["runtime"])
        resolver = kwargs["revision_resolver"]
        assert callable(resolver)
        assert resolver("SG161222/Realistic_Vision_V2.0") == "a" * 40
        return components

    monkeypatch.setattr(experiment, "load_model_components", load_components)
    preflight_steps: list[int] = []
    monkeypatch.setattr(
        experiment,
        "preflight_model_components",
        lambda *_args, **kwargs: preflight_steps.append(
            int(kwargs["num_inference_steps"])
        ),
    )
    component_lifecycle: list[str] = []
    monkeypatch.setattr(
        experiment,
        "_validate_loaded_components",
        lambda *_args, **_kwargs: component_lifecycle.append("validated"),
    )
    monkeypatch.setattr(
        experiment,
        "_offload_unused_vae",
        lambda *_args, **_kwargs: component_lifecycle.append("offloaded"),
    )
    monkeypatch.setattr(
        experiment,
        "encode_prompt_condition",
        lambda *_args, **_kwargs: torch.ones((1, 1, 1), dtype=torch.float32),
    )

    validated_pairs: list[str] = []

    def validate_pair(
        contract: object,
        metadata: Mapping[str, object],
    ) -> Mapping[str, object]:
        assert contract is generation_contract
        record_id = str(metadata["record_id"])
        validated_pairs.append(record_id)
        return {"validated_record_id": record_id}

    monkeypatch.setattr(experiment, "_validate_generation_pair", validate_pair)

    def load_target(
        *,
        metadata: Mapping[str, object],
        contract: object,
        generation_marker: Mapping[str, object],
    ) -> torch.Tensor:
        assert contract is generation_contract
        assert generation_marker["validated_record_id"] == metadata["record_id"]
        if metadata["record_id"] == records[2]["record_id"]:
            raise RuntimeError("synthetic target failure")
        return torch.ones((1, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(experiment, "_load_target_latent", load_target)
    _FakeProgress.instances.clear()

    loss_noises: list[torch.Tensor] = []
    recovery_latents: list[torch.Tensor] = []
    requested_recovery_seeds: list[tuple[int, ...]] = []
    requested_sscd_seeds: list[tuple[int, ...]] = []
    generation_initial = torch.full((3, 1, 2, 2), 99.0, dtype=torch.float32)

    def reconstruct(
        seeds: Sequence[int], *_args: object, **_kwargs: object
    ) -> torch.Tensor:
        requested_recovery_seeds.append(tuple(int(seed) for seed in seeds))
        return generation_initial.clone()

    monkeypatch.setattr(
        experiment, "_reconstruct_generation_initial_latents", reconstruct
    )

    def make_loss_noise(*_args: object, **kwargs: object) -> torch.Tensor:
        count = int(kwargs["num_loss_seeds"])
        return torch.arange(count * 4, dtype=torch.float32).reshape(count, 1, 2, 2)

    monkeypatch.setattr(experiment, "_make_loss_noise", make_loss_noise)

    def measure_loss(*_args: object, **kwargs: object) -> experiment.LossMeasurement:
        noise = kwargs["loss_noise"]
        if not isinstance(noise, torch.Tensor):
            raise AssertionError("run did not pass pair loss-noise samples")
        loss_noises.append(noise.detach().cpu().clone())
        return experiment.LossMeasurement(8.0, 2.0, math.sqrt(2.0))

    def measure_recovery(
        *_args: object, **kwargs: object
    ) -> experiment.RecoveryMeasurement:
        latents = kwargs["generation_initial_latents"]
        if not isinstance(latents, torch.Tensor):
            raise AssertionError("run did not pass generation initial latents")
        recovery_latents.append(latents.detach().cpu().clone())
        return experiment.RecoveryMeasurement(3.0, math.sqrt(3.0))

    monkeypatch.setattr(experiment, "_measure_conditional_loss", measure_loss)
    monkeypatch.setattr(experiment, "_measure_recovery", measure_recovery)

    def target_sscd(
        *,
        contract: object,
        metadata: Mapping[str, object],
        generation_seeds: Sequence[int],
        generation_marker: Mapping[str, object],
    ) -> float:
        assert contract is generation_contract
        assert generation_marker["validated_record_id"] == metadata["record_id"]
        requested_sscd_seeds.append(tuple(int(seed) for seed in generation_seeds))
        return 0.4

    monkeypatch.setattr(experiment, "_load_mean_target_sscd", target_sscd)

    plotted: list[tuple[Path, Path]] = []

    def fake_plot(csv_path: Path, output_directory: Path, **_: object) -> None:
        assert Path(csv_path).is_file()
        output_path = Path(output_directory)
        plotted.append((Path(csv_path), output_path))
        for figure_path in experiment._figure_paths(output_path):
            figure_path.write_bytes(b"synthetic figure")

    monkeypatch.setattr(experiment, "plot_saved_results", fake_plot)
    output = tmp_path / "loss-recovery"
    experiment.run_experiment(
        model_name="realvis",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=12,
        num_loss_seeds=2,
        loss_seed=7,
        num_seeds=3,
        selection_strategy="spearman",
        sample_batch_size=2,
        max_records=2,
        output_dir=output,
        device="cpu",
        progress_factory=_FakeProgress,
    )

    frame = pd.read_csv(output / experiment.CSV_NAME, keep_default_na=False)
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 2
    assert requested_dataset_models == ["realisticvision"]
    assert accessed_dataset_positions == [0, 2]
    assert selection_requests == [
        {
            "model_name": "realvis",
            "scheduler_name": "ddpm",
            "guidance_scale": 3.25,
            "num_inference_steps": 12,
            "num_seeds": 3,
            "selection_strategy": "spearman",
        }
    ]
    assert loaded_contract_arguments == [
        (experiment.ROOT, "realvis", "ddpm", 3.25, 12, 3)
    ]
    assert active_scheduler_arguments == [(components, generation_contract, "ddpm", 12)]
    assert preflight_steps == [12]
    assert len(loaded_runtimes) == 1
    assert loaded_runtimes[0].device == torch.device("cpu")
    assert loaded_runtimes[0].inference_dtype is torch.float32
    assert component_lifecycle == ["validated", "offloaded"]
    assert validated_pairs == [
        "realisticvision-0000",
        "realisticvision-0001",
    ]
    assert frame["record_id"].tolist() == [
        "realisticvision-0000",
        "realisticvision-0001",
    ]
    assert frame["model_name"].tolist() == ["realvis", "realvis"]
    assert frame["selection_strategy"].tolist() == ["spearman", "spearman"]
    assert frame["selection_hash"].tolist() == [SELECTION_HASH, SELECTION_HASH]
    assert frame["scheduler_name"].tolist() == ["ddpm", "ddpm"]
    assert frame["guidance_scale"].tolist() == [3.25, 3.25]
    assert frame["num_inference_steps"].tolist() == [12, 12]
    first, second = frame.iloc[0], frame.iloc[1]
    assert first["status"] == "ok" and first["error"] == ""
    assert second["status"] == "error"
    assert "synthetic target failure" in str(second["error"])
    assert int(first["timestep"]) == 3
    assert float(first["alpha_t"]) == pytest.approx(0.6)
    assert float(first["sigma_t"]) == pytest.approx(0.8)
    assert float(first["snr_t"]) == pytest.approx(0.5625)
    assert int(first["num_loss_seeds"]) == 2
    assert int(first["loss_seed"]) == 7
    assert str(first["generation_seeds"]) == "0,1,2"
    assert int(first["num_generation_seeds"]) == 3
    assert float(first["conditional_loss"]) == pytest.approx(8.0)
    assert float(first["normalized_loss_mse"]) == pytest.approx(2.0)
    assert float(first["normalized_loss_rmse"]) == pytest.approx(math.sqrt(2.0))
    assert float(first["recovery_mse"]) == pytest.approx(3.0)
    assert float(first["recovery_rmse"]) == pytest.approx(math.sqrt(3.0))
    assert float(first["mean_target_sscd"]) == pytest.approx(0.4)

    assert requested_recovery_seeds == [(0, 1, 2)]
    assert requested_sscd_seeds == [(0, 1, 2)]
    assert len(loss_noises) == len(recovery_latents) == 1
    assert loss_noises[0].shape == (2, 1, 2, 2)
    assert recovery_latents[0].shape == (3, 1, 2, 2)
    assert not torch.equal(loss_noises[0], recovery_latents[0])
    torch.testing.assert_close(recovery_latents[0], generation_initial)

    assert len(_FakeProgress.instances) == 1
    progress = _FakeProgress.instances[0]
    assert progress.kwargs["total"] == 2
    assert progress.kwargs["desc"] == "Theorem 1 loss–recovery"
    assert progress.kwargs["unit"] == "pair"
    assert progress.kwargs["dynamic_ncols"] is True
    assert progress.kwargs["leave"] is True
    assert progress.updated == 2
    completed_postfixes = [
        values for values in progress.postfixes if "status" in values
    ]
    assert [values["record_id"] for values in completed_postfixes] == [
        "realisticvision-0000",
        "realisticvision-0001",
    ]
    assert [values["status"] for values in completed_postfixes] == ["ok", "error"]
    assert {
        values.get("phase") for values in progress.postfixes if "phase" in values
    } == {
        "loading-target",
        "encoding-prompt",
        "conditional-loss",
        "recovery",
        "sscd",
    }
    assert plotted == [(output / experiment.CSV_NAME, output)]
    assert {path.name for path in output.iterdir()} == {
        experiment.CSV_NAME,
        *experiment.FIGURE_FILENAMES,
    }


def test_worker_setup_failure_still_writes_all_pair_rows_and_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        {"record_id": "sdv1-0000", "original_index": "10"},
        {"record_id": "sdv1-0001", "original_index": "11"},
    )

    class FakeDataset:
        def iter_metadata(self):  # type: ignore[no-untyped-def]
            yield from records

    contract = SimpleNamespace(
        seeds=tuple(range(2)),
        latent_shape=(4, 8, 8),
        science={
            "model_id": "model",
            "model_revision": "a" * 40,
            "vae_id": "vae",
            "vae_revision": "b" * 40,
        },
    )
    monkeypatch.setattr(
        experiment, "WebsterDataset", lambda *_args, **_kwargs: FakeDataset()
    )
    monkeypatch.setattr(
        experiment,
        "_load_frozen_selection",
        lambda *_args, **_kwargs: _selection(included=("10", "11")),
    )
    monkeypatch.setattr(
        experiment,
        "_load_generation_contract",
        lambda *_args, **_kwargs: contract,
    )

    def fail_model_load(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic model initialization failure")

    monkeypatch.setattr(experiment, "load_model_components", fail_model_load)
    plotted: list[tuple[Path, Path]] = []

    def fake_plot(csv_path: Path, output_directory: Path, **_: object) -> None:
        output_path = Path(output_directory)
        plotted.append((Path(csv_path), output_path))
        for figure_path in experiment._figure_paths(output_path):
            figure_path.write_bytes(b"synthetic figure")

    monkeypatch.setattr(experiment, "plot_saved_results", fake_plot)
    _FakeProgress.instances.clear()
    output = tmp_path / "setup-failure"

    exit_code = experiment.run_experiment(
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_loss_seeds=3,
        loss_seed=17,
        num_seeds=2,
        selection_strategy="gmm",
        sample_batch_size=2,
        max_records=None,
        output_dir=output,
        device="cpu",
        progress_factory=_FakeProgress,
    )

    assert exit_code == 1
    frame = pd.read_csv(output / experiment.CSV_NAME)
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert frame["record_id"].tolist() == ["sdv1-0000", "sdv1-0001"]
    assert frame["selection_strategy"].tolist() == ["gmm", "gmm"]
    assert frame["selection_hash"].tolist() == [SELECTION_HASH, SELECTION_HASH]
    assert frame["scheduler_name"].tolist() == ["ddim", "ddim"]
    assert frame["guidance_scale"].tolist() == [7.5, 7.5]
    assert frame["num_inference_steps"].tolist() == [50, 50]
    assert frame["status"].tolist() == ["error", "error"]
    assert frame["latent_dimension"].tolist() == [256, 256]
    assert frame["generation_seeds"].astype(str).tolist() == ["0,1", "0,1"]
    assert frame["timestep"].isna().all()
    assert frame["conditional_loss"].isna().all()
    assert all(
        "synthetic model initialization failure" in message
        for message in frame["error"].astype(str)
    )
    assert plotted == [(output / experiment.CSV_NAME, output)]
    assert {path.name for path in output.iterdir()} == {
        experiment.CSV_NAME,
        *experiment.FIGURE_FILENAMES,
    }
    assert len(_FakeProgress.instances) == 1
    progress = _FakeProgress.instances[0]
    assert progress.updated == 2
    assert [postfix["record_id"] for postfix in progress.postfixes] == [
        "sdv1-0000",
        "sdv1-0001",
    ]
    assert all(postfix["status"] == "error" for postfix in progress.postfixes)
    assert all(postfix["phase"] == "worker-failure" for postfix in progress.postfixes)


def test_multi_gpu_shards_whole_pairs_and_restores_metadata_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(
        (position, {"record_id": f"pair-{position}"}) for position in range(5)
    )
    devices = tuple(torch.device(f"cuda:{index}") for index in range(3))
    progress_lock = experiment.tqdm.get_lock()
    spawn_context = SimpleNamespace(RLock=lambda: progress_lock)
    dataset = object()
    contract = object()
    progress_factory = object()
    observed: list[
        tuple[
            int, str, str, float, int, tuple[int, ...], tuple[int, ...], int, int, bool
        ]
    ] = []

    class ImmediateFuture:
        def __init__(self, value: object) -> None:
            self.value = value

        def result(self) -> object:
            return self.value

    class ImmediateExecutor:
        def __init__(
            self,
            *,
            max_workers: int,
            mp_context: object,
            initializer: object,
            initargs: tuple[object, ...],
        ) -> None:
            assert max_workers == 2
            assert mp_context is spawn_context
            assert initializer is experiment._install_tqdm_lock
            assert initargs == (progress_lock,)

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *_arguments: object) -> None:
            return None

        def submit(
            self, function: object, *arguments: object, **kwargs: object
        ) -> ImmediateFuture:
            assert function is fake_pair_shard
            assert not arguments
            assert "dataset" not in kwargs
            assert "contract" not in kwargs
            assert "progress_factory" not in kwargs
            return ImmediateFuture(fake_pair_shard(**kwargs))

    def fake_pair_shard(**kwargs: object) -> experiment._PairShardResult:
        worker_index = int(kwargs["worker_index"])
        selected = kwargs["device"]
        shard = kwargs["entries"]
        seeds = kwargs["generation_seeds"]
        assert isinstance(selected, torch.device)
        assert isinstance(shard, Sequence)
        assert isinstance(seeds, tuple)
        positions = tuple(int(entry[0]) for entry in shard)
        is_local = "dataset" in kwargs
        if is_local:
            assert kwargs["dataset"] is dataset
            assert kwargs["contract"] is contract
            assert kwargs["progress_factory"] is progress_factory
        observed.append(
            (
                worker_index,
                str(selected),
                str(kwargs["scheduler_name"]),
                float(kwargs["guidance_scale"]),
                int(kwargs["num_inference_steps"]),
                positions,
                tuple(int(seed) for seed in seeds),
                int(kwargs["num_loss_seeds"]),
                int(kwargs["loss_seed"]),
                is_local,
            )
        )
        rows = tuple(
            (
                position,
                {
                    "record_id": f"pair-{position}",
                    "status": "error" if position == 3 else "ok",
                },
            )
            for position in positions
        )
        return experiment._PairShardResult(rows, int(3 in positions))

    monkeypatch.setattr(
        experiment.multiprocessing,
        "get_context",
        lambda method: spawn_context if method == "spawn" else None,
    )
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(experiment, "_run_pair_shard", fake_pair_shard)

    result = experiment._run_pair_shards(
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=12,
        num_loss_seeds=7,
        loss_seed=31,
        generation_seeds=(2, 5, 11),
        sample_batch_size=4,
        entries=entries,
        devices=devices,
        dataset=dataset,
        contract=contract,
        progress_factory=progress_factory,
    )

    assert observed == [
        (1, "cuda:1", "ddpm", 3.25, 12, (1, 4), (2, 5, 11), 7, 31, False),
        (2, "cuda:2", "ddpm", 3.25, 12, (2,), (2, 5, 11), 7, 31, False),
        (0, "cuda:0", "ddpm", 3.25, 12, (0, 3), (2, 5, 11), 7, 31, True),
    ]
    assert [position for position, _ in result.indexed_rows] == list(range(5))
    assert [row["record_id"] for _, row in result.indexed_rows] == [
        f"pair-{position}" for position in range(5)
    ]
    assert result.failed_count == 1


def test_multi_gpu_worker_crash_becomes_source_ordered_error_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_lock = experiment.tqdm.get_lock()
    spawn_context = SimpleNamespace(RLock=lambda: progress_lock)

    class FailedFuture:
        def result(self) -> object:
            raise RuntimeError("synthetic worker crash")

    class ImmediateExecutor:
        def __init__(
            self,
            *,
            max_workers: int,
            mp_context: object,
            initializer: object,
            initargs: tuple[object, ...],
        ) -> None:
            assert max_workers == 1
            assert mp_context is spawn_context
            assert initializer is experiment._install_tqdm_lock
            assert initargs == (progress_lock,)

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *_arguments: object) -> None:
            return None

        def submit(self, *_arguments: object, **_kwargs: object) -> FailedFuture:
            return FailedFuture()

    def local_pair_shard(**kwargs: object) -> experiment._PairShardResult:
        assert kwargs["worker_index"] == 0
        return experiment._PairShardResult(
            ((0, {"record_id": "pair-0", "status": "ok"}),),
            0,
        )

    monkeypatch.setattr(
        experiment.multiprocessing,
        "get_context",
        lambda method: spawn_context if method == "spawn" else None,
    )
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(experiment, "_run_pair_shard", local_pair_shard)
    _FakeProgress.instances.clear()

    result = experiment._run_pair_shards(
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=12,
        num_loss_seeds=2,
        loss_seed=0,
        generation_seeds=(0, 1),
        sample_batch_size=2,
        entries=(
            (0, {"record_id": "pair-0"}),
            (1, {"record_id": "pair-1"}),
        ),
        devices=(torch.device("cuda:0"), torch.device("cuda:1")),
        dataset=object(),
        contract=SimpleNamespace(latent_shape=(4, 8, 8)),
        progress_factory=_FakeProgress,
    )

    assert [position for position, _row in result.indexed_rows] == [0, 1]
    assert result.indexed_rows[0][1] == {
        "record_id": "pair-0",
        "status": "ok",
    }
    failed_row = result.indexed_rows[1][1]
    assert failed_row["record_id"] == "pair-1"
    assert failed_row["status"] == "error"
    assert failed_row["latent_dimension"] == 256
    assert failed_row["scheduler_name"] == "ddpm"
    assert failed_row["guidance_scale"] == pytest.approx(3.25)
    assert failed_row["num_inference_steps"] == 12
    assert "worker on cuda:1 failed" in str(failed_row["error"])
    assert "synthetic worker crash" in str(failed_row["error"])
    assert result.failed_count == 1
    assert len(_FakeProgress.instances) == 1
    progress = _FakeProgress.instances[0]
    assert progress.updated == 1
    assert progress.kwargs["total"] == 1
    assert progress.kwargs["desc"] == (
        experiment.PROGRESS_DESCRIPTION + " [cuda:1 shard 2/2]"
    )
    assert progress.postfixes[-1]["phase"] == "worker-failure"
    assert progress.postfixes[-1]["status"] == "error"


def test_merge_rejects_missing_or_duplicate_pair_positions() -> None:
    first = experiment._PairShardResult(((0, {"status": "ok"}),), 0)
    duplicate = experiment._PairShardResult(((0, {"status": "ok"}),), 0)
    with pytest.raises(experiment.ExperimentError, match="duplicate or missing"):
        experiment._merge_pair_shard_results((first, duplicate), (0, 2))

    noncontiguous = experiment._merge_pair_shard_results(
        (
            experiment._PairShardResult(((4, {"record_id": "second"}),), 0),
            experiment._PairShardResult(((1, {"record_id": "first"}),), 0),
        ),
        (1, 4),
    )
    assert [position for position, _row in noncontiguous.indexed_rows] == [1, 4]


def _row(
    record_id: str,
    *,
    normalized_loss_rmse: float,
    recovery_rmse: float,
    mean_target_sscd: float,
    status: str = "ok",
    error: str = "",
    num_loss_seeds: int = 2,
    generation_seeds: str = "0,1",
) -> dict[str, object]:
    alpha_t = 0.1
    sigma_t = math.sqrt(1.0 - alpha_t**2)
    snr_t = alpha_t**2 / sigma_t**2
    normalized_loss_mse = normalized_loss_rmse**2
    recovery_mse = recovery_rmse**2
    return {
        "record_id": record_id,
        "selection_strategy": "gmm",
        "selection_hash": SELECTION_HASH,
        "model_name": "sdv1",
        "scheduler_name": "ddim",
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "timestep": 981,
        "alpha_t": alpha_t,
        "sigma_t": sigma_t,
        "snr_t": snr_t,
        "latent_dimension": 4,
        "num_loss_seeds": num_loss_seeds,
        "loss_seed": 123,
        "generation_seeds": generation_seeds,
        "num_generation_seeds": len(generation_seeds.split(",")),
        "conditional_loss": normalized_loss_mse * 4 * snr_t,
        "normalized_loss_mse": normalized_loss_mse,
        "normalized_loss_rmse": normalized_loss_rmse,
        "recovery_mse": recovery_mse,
        "recovery_rmse": recovery_rmse,
        "mean_target_sscd": mean_target_sscd,
        "status": status,
        "error": error,
    }


def _offset_tuples(collection: PathCollection) -> set[tuple[float, float]]:
    offsets = np.asarray(collection.get_offsets(), dtype=float)
    if offsets.size == 0:
        return set()
    return {(float(x_value), float(y_value)) for x_value, y_value in offsets}


def test_plot_reloads_csv_and_draws_pair_scatter_and_binned_median(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_rows = [
        _row(
            f"sdv1-{index:04d}",
            normalized_loss_rmse=float(index + 1),
            recovery_rmse=1.0 / float(index + 1),
            mean_target_sscd=0.1 + 0.8 * index / 19.0,
        )
        for index in range(20)
    ]
    excluded_rows = [
        _row(
            "sdv1-error",
            normalized_loss_rmse=3.0,
            recovery_rmse=0.3,
            mean_target_sscd=0.5,
            status="error",
            error="synthetic failure",
        ),
        _row(
            "sdv1-zero-x",
            normalized_loss_rmse=0.0,
            recovery_rmse=0.2,
            mean_target_sscd=0.5,
        ),
        _row(
            "sdv1-nan-y",
            normalized_loss_rmse=2.0,
            recovery_rmse=math.nan,
            mean_target_sscd=0.5,
        ),
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(
        [*excluded_rows, *reversed(valid_rows)], columns=EXPECTED_COLUMNS
    ).to_csv(csv_path, index=False)

    real_read_csv = pd.read_csv
    reads: list[Path] = []

    def read_csv_spy(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(experiment.pd, "read_csv", read_csv_spy)
    real_subplots = experiment.plt.subplots
    captured: dict[str, object] = {}
    savefig_calls: list[dict[str, object]] = []

    def subplots_spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        figure, axis = real_subplots(*args, **kwargs)
        real_savefig = figure.savefig
        real_tight_layout = figure.tight_layout

        def savefig_spy(*save_args: object, **save_kwargs: object) -> object:
            savefig_calls.append(dict(save_kwargs))
            return real_savefig(*save_args, **save_kwargs)

        def tight_layout_spy(*layout_args: object, **layout_kwargs: object) -> object:
            captured["tight_layout_calls"] = (
                int(captured.get("tight_layout_calls", 0)) + 1
            )
            return real_tight_layout(*layout_args, **layout_kwargs)

        monkeypatch.setattr(figure, "savefig", savefig_spy)
        monkeypatch.setattr(figure, "tight_layout", tight_layout_spy)
        captured.update(
            figure=figure,
            axis=axis,
            font_family=tuple(experiment.matplotlib.rcParams["font.family"]),
            mathtext_fontset=experiment.matplotlib.rcParams["mathtext.fontset"],
            default_font_size=experiment.matplotlib.rcParams["font.size"],
        )
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", subplots_spy)
    experiment.plot_saved_results(csv_path, tmp_path)

    assert reads == [csv_path]
    png_path, pdf_path = experiment._figure_paths(tmp_path)
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF-")
    figure = captured["figure"]
    axis = captured["axis"]
    np.testing.assert_allclose(figure.get_size_inches(), [4.0, 4.0])
    assert captured["font_family"] == ("STIXGeneral",)
    assert captured["mathtext_fontset"] == "stix"
    assert captured["default_font_size"] == 15
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert axis.get_title() == ""
    assert axis.xaxis.label.get_fontsize() == 15
    assert axis.yaxis.label.get_fontsize() == 15
    assert axis.xaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
    assert all(label.get_fontsize() == 12 for label in axis.get_yticklabels())

    pair_collections = [
        collection
        for collection in axis.collections
        if isinstance(collection, PathCollection)
        and np.asarray(collection.get_offsets()).shape[0] == 20
    ]
    assert len(pair_collections) == 1
    scatter = pair_collections[0]
    assert scatter.cmap.name == "viridis"
    assert scatter.norm.vmin == pytest.approx(0.0)
    assert scatter.norm.vmax == pytest.approx(1.0)
    assert scatter.norm.clip is True
    assert scatter.get_alpha() == pytest.approx(experiment.SCATTER_ALPHA)
    assert scatter.get_alpha() != experiment.COLORBAR_ALPHA
    assert scatter.get_array() is not None
    np.testing.assert_allclose(
        np.sort(np.asarray(scatter.get_array(), dtype=float)),
        np.linspace(0.1, 0.9, 20),
    )
    observed_points = _offset_tuples(scatter)
    expected_points = {(float(index), 1.0 / index) for index in range(1, 21)}
    assert len(observed_points) == len(expected_points)
    for expected in expected_points:
        assert any(np.allclose(expected, observed) for observed in observed_points)

    trend = [line for line in axis.lines if line.get_label() == "Binned median"]
    assert len(trend) == 1
    trend_line = trend[0]
    assert str(trend_line.get_color()).lower() in {"black", "k", "#000000"}
    assert trend_line.get_marker() == "o"
    expected_x = np.arange(1.5, 20.0, 2.0)
    expected_y = np.asarray(
        [(1.0 / left + 1.0 / (left + 1)) / 2.0 for left in range(1, 21, 2)]
    )
    np.testing.assert_allclose(trend_line.get_xdata(), expected_x)
    np.testing.assert_allclose(trend_line.get_ydata(), expected_y)
    assert len(axis.lines) == 1

    assert len(axis.texts) == 0
    assert axis.get_xlabel() == (
        r"$\sqrt{\mathcal{L}^{\star}_{T,c} / "
        r"[d\,(\alpha_T^2/\sigma_T^2)]}$"
    )
    assert r"\frac" not in axis.get_xlabel()
    assert "SNR" not in axis.get_xlabel()
    assert axis.get_ylabel() == (
        r"$\sqrt{\mathbb{E}_{\mathbf{x}_T\sim"
        r"\mathcal{N}(\mathbf{0},\mathbf{I})}"
        r"[\|\hat{\mathbf{x}}_{0\mid T,c}-\mathbf{x}^{\star}\|_2^2/d]}$"
    )
    assert "recovery_rmse" not in axis.get_ylabel()

    legend = axis.get_legend()
    assert legend is not None
    assert all(item.get_fontsize() == 10 for item in legend.get_texts())

    colorbar_axes = [plot_axis for plot_axis in figure.axes if plot_axis is not axis]
    assert len(colorbar_axes) == 1
    colorbar_axis = colorbar_axes[0]
    assert colorbar_axis.get_ylabel() == "SSCD"
    np.testing.assert_allclose(colorbar_axis.get_ylim(), [0.0, 1.0])
    assert colorbar_axis.yaxis.label.get_fontsize() == 15
    assert colorbar_axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert all(label.get_fontsize() == 12 for label in colorbar_axis.get_yticklabels())
    meshes = [
        collection
        for collection in colorbar_axis.collections
        if isinstance(collection, QuadMesh)
    ]
    assert len(meshes) == 1
    colorbar_mesh = meshes[0]
    assert colorbar_mesh.get_alpha() == pytest.approx(1.0)
    figure.canvas.draw()
    np.testing.assert_allclose(colorbar_mesh.get_facecolors()[:, 3], 1.0)
    assert captured["tight_layout_calls"] == 1
    assert savefig_calls == [
        {
            "format": "png",
            "bbox_inches": "tight",
            "pad_inches": 0.05,
            "dpi": 300,
        },
        {
            "format": "pdf",
            "bbox_inches": "tight",
            "pad_inches": 0.05,
        },
    ]


def test_plot_reduces_bins_and_warns_when_two_bins_are_impossible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(
        [
            _row(
                "sdv1-0000",
                normalized_loss_rmse=1.0,
                recovery_rmse=2.0,
                mean_target_sscd=0.3,
            )
        ],
        columns=EXPECTED_COLUMNS,
    ).to_csv(csv_path, index=False)

    real_subplots = experiment.plt.subplots
    captured: dict[str, object] = {}

    def subplots_spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        figure, axis = real_subplots(*args, **kwargs)
        captured.update(figure=figure, axis=axis)
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", subplots_spy)
    experiment.plot_saved_results(csv_path, tmp_path)

    axis = captured["axis"]
    assert not [line for line in axis.lines if line.get_label() == "Binned median"]
    captured_output = capsys.readouterr()
    assert "warning" in (captured_output.out + captured_output.err).lower()


def test_failed_figure_staging_preserves_both_outputs_and_removes_staged_files(
    tmp_path: Path,
) -> None:
    destinations = experiment._figure_paths(tmp_path)
    original_contents = (b"original PNG", b"original PDF")
    for destination, contents in zip(destinations, original_contents, strict=True):
        destination.write_bytes(contents)

    def fail_on_pdf(destination: Path, **options: object) -> None:
        Path(destination).write_bytes(b"staged figure")
        if options["format"] == "pdf":
            raise KeyboardInterrupt("synthetic PDF failure")

    figure = SimpleNamespace(savefig=fail_on_pdf)
    with pytest.raises(KeyboardInterrupt, match="synthetic PDF failure"):
        experiment._atomic_save_figures(figure, destinations)

    assert tuple(destination.read_bytes() for destination in destinations) == (
        original_contents
    )
    assert set(tmp_path.iterdir()) == set(destinations)


def test_partial_figure_install_rolls_back_both_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destinations = experiment._figure_paths(tmp_path)
    original_contents = (b"original PNG", b"original PDF")
    for destination, contents in zip(destinations, original_contents, strict=True):
        destination.write_bytes(contents)

    def save_figure(destination: Path, **options: object) -> None:
        Path(destination).write_bytes(f"new {options['format']}".encode())

    real_replace = experiment.os.replace
    failed = False

    def fail_second_install(source: Path, destination: Path) -> None:
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not failed
            and destination_path == destinations[1]
            and source_path.name.startswith(f".{destinations[1].name}.")
        ):
            failed = True
            raise OSError("synthetic second-install failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(experiment.os, "replace", fail_second_install)
    figure = SimpleNamespace(savefig=save_figure)
    with pytest.raises(OSError, match="synthetic second-install failure"):
        experiment._atomic_save_figures(figure, destinations)

    assert tuple(destination.read_bytes() for destination in destinations) == (
        original_contents
    )
    assert set(tmp_path.iterdir()) == set(destinations)


def test_plot_uses_saved_csv_without_loading_diffusion_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "plot"
    output.mkdir()
    csv_path = output / experiment.CSV_NAME
    pd.DataFrame(
        [
            _row(
                "sdv1-0000",
                normalized_loss_rmse=1.0,
                recovery_rmse=2.0,
                mean_target_sscd=0.3,
            ),
            _row(
                "sdv1-0001",
                normalized_loss_rmse=2.0,
                recovery_rmse=1.0,
                mean_target_sscd=0.6,
            ),
        ],
        columns=EXPECTED_COLUMNS,
    ).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("plot mode must not load or run the diffusion model")

    monkeypatch.setattr(experiment, "load_model_components", forbidden)
    monkeypatch.setattr(experiment, "preflight_model_components", forbidden)
    monkeypatch.setattr(experiment, "WebsterDataset", forbidden)
    monkeypatch.setattr(experiment, "run_experiment", forbidden)
    monkeypatch.setattr(experiment, "resolve_devices", forbidden)
    monkeypatch.setattr(
        experiment,
        "_load_frozen_selection",
        lambda *_args, **kwargs: _selection(
            included=("0", "1"),
            strategy=str(kwargs["selection_strategy"]),
            selection_hash=(
                SELECTION_HASH if kwargs["selection_strategy"] == "gmm" else "d" * 64
            ),
        ),
    )
    with pytest.raises(
        experiment.ExperimentError,
        match="saved CSV differs from requested configuration at: loss_seed",
    ):
        experiment.main(
            [
                "--plot",
                "--selection-strategy",
                "gmm",
                "--N",
                "2",
                "--num-loss-seeds",
                "2",
                "--loss-seed",
                "124",
                "--device",
                "cuda:999",
                "--output-dir",
                str(output),
            ]
        )
    with pytest.raises(
        experiment.ExperimentError,
        match=(
            "saved CSV differs from requested configuration at: "
            "selection_strategy, selection_hash"
        ),
    ):
        experiment.main(
            [
                "--plot",
                "--selection-strategy",
                "spearman",
                "--N",
                "2",
                "--num-loss-seeds",
                "2",
                "--loss-seed",
                "123",
                "--output-dir",
                str(output),
            ]
        )
    assert (
        experiment.main(
            [
                "--plot",
                "--selection-strategy",
                "gmm",
                "--N",
                "2",
                "--num-loss-seeds",
                "2",
                "--loss-seed",
                "123",
                "--device",
                "cuda:999",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert csv_path.read_bytes() == original_csv
    assert {path.name for path in output.iterdir()} == {
        experiment.CSV_NAME,
        *experiment.FIGURE_FILENAMES,
    }


def test_source_has_no_previous_experiment_or_reverse_trajectory_logic() -> None:
    source_path = Path(experiment.__file__)
    source = source_path.read_text(encoding="utf-8")
    lowered = source.lower()
    tree = ast.parse(source, filename=str(source_path))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    for forbidden in (
        "theorem1_finite_t_recovery",
        "theorem1_conditional_recovery",
        "failure_probability",
        "theorem_bound",
        "markov_term",
        "pinsker",
        "noise_level_index",
        "recovery_rmse_threshold",
        "num_noise_levels",
        "num_generation_samples",
        "num_loss_samples",
        "seedwise",
        "95th percentile",
        "np.percentile",
    ):
        assert forbidden not in lowered
    assert "step" not in called_attributes
    assert "fill_between" not in called_attributes
    assert "axline" not in called_attributes
    assert "polyfit" not in called_attributes
    assert "linregress" not in called_names
    assert not called_names.intersection(
        {"sample_trajectory", "run_batched_sampling", "sample_diffusion"}
    )
    assert ".svg" not in lowered
    assert "generation_seed" not in experiment.CSV_COLUMNS
    assert "seed_sscd" not in experiment.CSV_COLUMNS
