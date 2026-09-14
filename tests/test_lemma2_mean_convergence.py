"""Offline contracts for cached, prompt-level centered Lemma 2 evaluation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import math
from pathlib import Path
import warnings
from types import SimpleNamespace
from unittest.mock import Mock
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import (
    LineCollection,
    PathCollection,
    PolyCollection,
    QuadMesh,
)
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import lemma2_mean_convergence as experiment
from utils.common.io import canonical_hash
from utils.data.selection import TargetPairSelection
from utils.experiments.cache import CompletedGenerationRecord, GenerationPaths


EXPECTED_COLUMNS = (
    "selection_strategy",
    "selection_hash",
    "model_name",
    "scheduler_name",
    "guidance_scale",
    "num_inference_steps",
    "evaluation_source",
    "centering_mode",
    "num_baseline_seeds",
    "evaluation_generation_scientific_config_hash",
    "evaluation_schedule_sha256",
    "baseline_generation_scientific_config_hash",
    "baseline_mu_hat_sha256",
    "record_id",
    "original_index",
    "generation_seed",
    "step_index",
    "timestep",
    "alpha_t",
    "sigma_t",
    "snr_t",
    "latent_dimension",
    "centered_distance_rmse",
    "target_sscd",
    "trajectory_sha256",
    "is_initial_timestep",
    "status",
    "error",
)

SELECTION_HASH = "a" * 64
EVALUATION_HASH = "b" * 64
BASELINE_HASH = "c" * 64
MU_HASH = "d" * 64
SCHEDULE_HASH = "9" * 64
OTHER_SCHEDULE_HASH = "0" * 64
NUM_BASELINE_SEEDS = 3
LATENT_HASHES = {"10": "1" * 64, "11": "2" * 64}
PREDICTION_HASHES = {"10": "3" * 64, "11": "4" * 64}
TARGET_SSCD = {
    ("10", 0): 0.10,
    ("10", 1): 0.30,
    ("11", 0): 0.70,
    ("11", 1): 0.90,
}


def _selection(root: Path) -> TargetPairSelection:
    rows = [
        {
            "original_index": "10",
            "record_id": "record-10",
            "source_row_number": 10,
            "seed": 0,
            "prompt": "first selected prompt",
            "target_image_sha256": "5" * 64,
            "include_prompt": True,
            "selection_strategy": "gmm",
            "selection_hash": SELECTION_HASH,
        },
        {
            "original_index": "11",
            "record_id": "record-11",
            "source_row_number": 11,
            "seed": 0,
            "prompt": "second selected prompt",
            "target_image_sha256": "6" * 64,
            "include_prompt": True,
            "selection_strategy": "gmm",
            "selection_hash": SELECTION_HASH,
        },
        {
            "original_index": "12",
            "record_id": "record-12",
            "source_row_number": 12,
            "seed": 0,
            "prompt": "excluded prompt",
            "target_image_sha256": "7" * 64,
            "include_prompt": False,
            "selection_strategy": "gmm",
            "selection_hash": SELECTION_HASH,
        },
    ]
    return TargetPairSelection(
        root=root,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        selection_strategy="gmm",
        frame=pd.DataFrame(rows),
        configuration={"selection_hash": SELECTION_HASH},
        sha256=SELECTION_HASH,
    )


def _science() -> dict[str, object]:
    spec = experiment.get_model_spec("sdv1")
    scheduler_config = {
        "num_train_timesteps": 10,
        "prediction_type": "epsilon",
    }
    return {
        "schema_version": 1,
        "sampler_contract_version": 2,
        "selection_policy": "all_available_canonical_pairs_label_independent",
        "model_cli_name": "sdv1",
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "model_revision": "model-revision",
        "vae_id": spec.vae_id or spec.model_id,
        "vae_revision": "vae-revision",
        "scheduler": {
            "name": "ddim",
            "class": "DDIMScheduler",
            "config": scheduler_config,
        },
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "guidance_scale": 7.5,
        "num_inference_steps": 3,
        "num_seeds": 2,
        "seeds": [0, 1],
        "latent_shape": [1, 1, 2],
        "trajectory_order": "noise_to_image",
        "scientific_tensor_storage": {
            "dtype": "float32",
            "device": "cpu",
            "layout": "contiguous",
        },
    }


def _marker(index: str) -> dict[str, object]:
    science = _science()
    row = {
        "10": ("record-10", 10, "first selected prompt", "5" * 64),
        "11": ("record-11", 11, "second selected prompt", "6" * 64),
    }[index]
    return {
        "original_index": index,
        "record_id": row[0],
        "source_row_number": row[1],
        "prompt_raw": row[2],
        "target_image_sha256": row[3],
        "model_cli_name": "sdv1",
        "dataset_model": science["dataset_model"],
        "model_id": science["model_id"],
        "model_revision": science["model_revision"],
        "vae_id": science["vae_id"],
        "vae_revision": science["vae_revision"],
        "scheduler_name": "ddim",
        "scheduler_class": "DDIMScheduler",
        "guidance_scale": 7.5,
        "num_inference_steps": 3,
        "num_seeds": 2,
        "seeds": [0, 1],
        "latent_shape": [1, 1, 2],
        "inference_dtype": "float32",
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
        "scientific_config_hash": EVALUATION_HASH,
        "latent_path": f"latent/{index}.pt",
        "noise_prediction_path": f"noise_pred/{index}.pt",
        "schedule_path": "schedule.pt",
        "tensor_shapes": {
            "latent": [2, 4, 1, 1, 2],
            "unconditional_noise_predictions": [2, 3, 1, 1, 2],
            "conditional_noise_predictions": [2, 3, 1, 1, 2],
        },
        "tensor_dtypes": {
            "latent": "float32",
            "unconditional_noise_predictions": "float32",
            "conditional_noise_predictions": "float32",
        },
        "tensor_file_sha256": {
            "latent": LATENT_HASHES[index],
            "noise_prediction": PREDICTION_HASHES[index],
            "target_latent": "8" * 64,
        },
    }


def _completed(root: Path, index: str) -> CompletedGenerationRecord:
    marker = _marker(index)
    return CompletedGenerationRecord(
        original_index=index,
        source_row_number=int(marker["source_row_number"]),
        marker_path=root / "record" / f"{index}.json",
        metadata=marker,
    )


def _identity(root: Path) -> experiment.GenerationIdentity:
    records = {index: _completed(root, index) for index in ("10", "11")}
    return experiment.GenerationIdentity(
        paths=GenerationPaths(root / "logs" / "synthetic" / "experiment_S0_N2"),
        science=_science(),
        scientific_hash=EVALUATION_HASH,
        schedule_sha256=SCHEDULE_HASH,
        seeds=(0, 1),
        latent_shape=(1, 1, 2),
        latent_dimension=2,
        stored_dtype=torch.float32,
        records_by_index=records,
    )


def _schedule() -> dict[str, object]:
    alpha = torch.tensor([0.10, 0.25, 0.50], dtype=torch.float32)
    sigma = torch.sqrt(1.0 - alpha.square()).contiguous()
    return {
        "timesteps": torch.tensor([9, 6, 3], dtype=torch.int64),
        "alpha_t": alpha,
        "sigma_t": sigma,
        "alphas_cumprod_t": alpha.square().contiguous(),
        "init_noise_sigma": 1.0,
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "scheduler_name": "ddim",
        "scheduler_class": "DDIMScheduler",
        "scheduler_config": {
            "num_train_timesteps": 10,
            "prediction_type": "epsilon",
        },
        "trajectory_order": "noise_to_image",
    }


def _contract(root: Path) -> experiment.GenerationContract:
    identity = _identity(root)
    payload = _schedule()
    alpha = payload["alpha_t"]
    sigma = payload["sigma_t"]
    assert isinstance(alpha, torch.Tensor)
    assert isinstance(sigma, torch.Tensor)
    return experiment.GenerationContract(
        identity=identity,
        schedule_payload=payload,
        scheduler_config={
            "num_train_timesteps": 10,
            "prediction_type": "epsilon",
        },
        init_noise_sigma=1.0,
        timesteps=payload["timesteps"],
        alpha_t=alpha,
        sigma_t=sigma,
        snr_t=alpha.double().square().div(sigma.double().square()).contiguous(),
    )


def _scheduler_science(scheduler_name: str) -> dict[str, object]:
    science = _science()
    scheduler = science["scheduler"]
    assert isinstance(scheduler, dict)
    return {
        **science,
        "scheduler": {
            **scheduler,
            "name": scheduler_name,
            "class": f"{scheduler_name.upper()}Scheduler",
        },
    }


def _scheduler_identity(
    root: Path, scheduler_name: str
) -> experiment.GenerationIdentity:
    return replace(
        _identity(root),
        science=_scheduler_science(scheduler_name),
    )


def _scheduler_schedule(scheduler_name: str) -> dict[str, object]:
    return {
        **_schedule(),
        "scheduler_name": scheduler_name,
        "scheduler_class": f"{scheduler_name.upper()}Scheduler",
    }


def _scheduler_contract(
    root: Path, scheduler_name: str
) -> experiment.GenerationContract:
    return replace(
        _contract(root),
        identity=_scheduler_identity(root, scheduler_name),
        schedule_payload=_scheduler_schedule(scheduler_name),
    )


def _scheduler_selection(root: Path, scheduler_name: str) -> TargetPairSelection:
    return replace(_selection(root), scheduler_name=scheduler_name)


def _scheduler_marker(index: str, scheduler_name: str) -> dict[str, object]:
    return {
        **_marker(index),
        "scheduler_name": scheduler_name,
        "scheduler_class": f"{scheduler_name.upper()}Scheduler",
    }


def _baseline_contract(
    mu_hat: torch.Tensor | None = None,
    *,
    source_schedule_sha256: str = SCHEDULE_HASH,
) -> experiment.BaselineContract:
    return experiment.BaselineContract(
        artifact=object(),
        mu_hat=mu_hat,
        mu_hat_sha256=MU_HASH,
        source_scientific_hash=BASELINE_HASH,
        source_schedule_sha256=source_schedule_sha256,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        baseline_seeds=(2, 3, 4),
        timestep=9,
        alpha_t=float(torch.tensor(0.10, dtype=torch.float32)),
        sigma_t=float(torch.sqrt(torch.tensor(0.99, dtype=torch.float32))),
        latent_shape=(1, 1, 2),
        latent_dimension=2,
    )


def _mu_centering_contract(
    mu_hat: torch.Tensor | None = None,
    *,
    source_schedule_sha256: str = SCHEDULE_HASH,
) -> experiment.CenteringContract:
    baseline = _baseline_contract(
        mu_hat,
        source_schedule_sha256=source_schedule_sha256,
    )
    return experiment.CenteringContract(
        mode=experiment.CENTERING_MU_HAT,
        center=mu_hat,
        baseline=baseline,
    )


def _zero_centering_contract(
    center: torch.Tensor | None = None,
) -> experiment.CenteringContract:
    return experiment.CenteringContract(
        mode=experiment.CENTERING_ZERO,
        center=center,
        baseline=None,
    )


def _trajectory_hash(index: str) -> str:
    return canonical_hash(
        {
            "latent": LATENT_HASHES[index],
            "noise_prediction": PREDICTION_HASHES[index],
        }
    )


def _measurements(root: Path) -> tuple[experiment._PromptMeasurement, ...]:
    return (
        experiment._PromptMeasurement(
            position=0,
            original_index="10",
            trajectory_sha256=_trajectory_hash("10"),
            values=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
            error="",
        ),
        experiment._PromptMeasurement(
            position=1,
            original_index="11",
            trajectory_sha256=_trajectory_hash("11"),
            values=np.asarray([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype=np.float64),
            error="",
        ),
    )


def _rows(root: Path) -> list[dict[str, object]]:
    selection = _selection(root)
    rows, failed = experiment._expand_measurements(
        _measurements(root),
        experiment._included_prompt_rows(selection),
        selection=selection,
        contract=_contract(root),
        centering=_mu_centering_contract(None),
    )
    assert failed == 0
    return experiment._attach_target_sscd(rows, TARGET_SSCD)


def _gaussian_measurements() -> tuple[experiment._GaussianBatchMeasurement, ...]:
    return (
        experiment._GaussianBatchMeasurement(
            entries=((0, 0), (1, 1)),
            values=np.asarray([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], dtype=np.float64),
            error="",
        ),
    )


def _gaussian_rows(root: Path) -> list[dict[str, object]]:
    rows, failed = experiment._expand_gaussian_measurements(
        _gaussian_measurements(),
        selection=_selection(root),
        contract=_contract(root),
        centering=_mu_centering_contract(None),
    )
    assert failed == 0
    return rows


def _both_rows(root: Path) -> list[dict[str, object]]:
    return [*_gaussian_rows(root), *_rows(root)]


def test_public_schema_and_cli_keep_generation_N_and_baseline_B() -> None:
    parser = experiment.build_parser()
    defaults = parser.parse_args([])
    arguments = parser.parse_args(
        ["--N", "7", "--T", "11", "--num-baseline-seeds", "13", "--device", "cpu"]
    )

    assert experiment.CSV_COLUMNS == EXPECTED_COLUMNS
    assert arguments.num_seeds == 7
    assert arguments.num_baseline_seeds == 13
    assert defaults.num_baseline_seeds == 1000
    assert defaults.selection_strategy == "gmm"
    assert (
        parser.parse_args(["--selection-strategy", "gmm"]).selection_strategy == "gmm"
    )
    for unsupported in ("spearman", "gmm-evidence", "all"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--selection-strategy", unsupported])
    assert defaults.evaluation_source == "both"
    assert defaults.use_mu is False
    assert parser.parse_args(["--use-mu"]).use_mu is True
    assert arguments.num_inference_steps == 11
    assert arguments.device == "cpu"
    assert "cached experiment generation seeds" in parser.format_help()
    assert "(default: gmm)" in parser.format_help()
    assert "(default: both)" in parser.format_help()
    assert parser.parse_args(["--evaluation-source", "gaussian"]).evaluation_source == (
        "gaussian"
    )


def test_evaluation_source_resolution_is_explicit_and_canonical() -> None:
    assert experiment._evaluation_sources("gaussian") == ("gaussian",)
    assert experiment._evaluation_sources("trajectory") == ("trajectory",)
    assert experiment._evaluation_sources("both") == ("gaussian", "trajectory")
    with pytest.raises(experiment.ExperimentError, match="unsupported evaluation"):
        experiment._evaluation_sources("legacy")


def test_trajectory_source_remains_cache_only() -> None:
    source = inspect.getsource(experiment._compute_rows)

    for forbidden in (
        "load_model_components",
        "predict_conditional_epsilon",
        "encode_prompt_condition",
        "make_initial_noise",
        "build_scheduler",
        "compile_loaded_unet",
        "_compute_gaussian_rows",
    ):
        assert forbidden not in source
    distance_source = inspect.getsource(experiment._centered_trajectory_distances)
    assert "latents[:, step_index]" in distance_source
    assert "unconditional_epsilon[:, step_index]" in distance_source


def test_selection_returns_every_included_prompt_in_canonical_order(
    tmp_path: Path,
) -> None:
    rows = experiment._included_prompt_rows(_selection(tmp_path))

    assert [row["original_index"] for row in rows] == ["10", "11"]
    assert [row["record_id"] for row in rows] == ["record-10", "record-11"]


def test_ddim_and_ddpm_are_supported() -> None:
    for scheduler_name in ("ddim", "ddpm"):
        experiment._validate_scientific_request(scheduler_name)
    with pytest.raises(experiment.ExperimentError, match="ddim, ddpm"):
        experiment._validate_scientific_request("euler")
    assert (
        experiment.build_parser().parse_args(["--scheduler", "ddpm"]).scheduler
        == "ddpm"
    )


def test_generation_identity_is_json_only_and_indexes_completion_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "run" / "experiment_S0_N2")
    paths.run_directory.mkdir(parents=True)
    paths.schedule.write_bytes(b"schedule")
    records = [_completed(tmp_path, "10"), _completed(tmp_path, "11")]
    monkeypatch.setattr(experiment, "generation_paths", lambda *_args, **_kwargs: paths)
    monkeypatch.setattr(
        experiment,
        "require_generation_run",
        lambda _paths: {
            "scientific_config": _science(),
            "scientific_config_hash": EVALUATION_HASH,
        },
    )
    monkeypatch.setattr(experiment, "list_completed_records", lambda _paths: records)
    monkeypatch.setattr(experiment, "file_sha256", lambda _path: SCHEDULE_HASH)
    monkeypatch.setattr(
        experiment,
        "safe_torch_load",
        lambda *_args, **_kwargs: pytest.fail("identity load deserialized a tensor"),
    )

    identity = experiment._load_generation_identity(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
    )

    assert identity.seeds == (0, 1)
    assert tuple(identity.records_by_index) == ("10", "11")
    assert identity.stored_dtype == torch.float32
    assert identity.schedule_sha256 == SCHEDULE_HASH


def test_schedule_file_sha256_requires_a_safe_regular_file(tmp_path: Path) -> None:
    paths = GenerationPaths(tmp_path / "generation")
    paths.run_directory.mkdir(parents=True)
    payload = b"physical schedule bytes"
    paths.schedule.write_bytes(payload)

    assert (
        experiment._schedule_file_sha256(paths) == hashlib.sha256(payload).hexdigest()
    )

    target = tmp_path / "other-schedule.pt"
    target.write_bytes(payload)
    paths.schedule.unlink()
    paths.schedule.symlink_to(target)
    with pytest.raises(experiment.ExperimentError, match="missing or unsafe"):
        experiment._schedule_file_sha256(paths)


def test_schedule_contract_loads_every_cached_ddim_timestep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity(tmp_path)
    identity.paths.run_directory.mkdir(parents=True)
    identity.paths.schedule.touch()
    monkeypatch.setattr(
        experiment, "_schedule_file_sha256", lambda _paths: SCHEDULE_HASH
    )
    monkeypatch.setattr(experiment, "safe_torch_load", lambda _path: _schedule())

    contract = experiment._load_generation_contract(identity)

    assert contract.timesteps.tolist() == [9, 6, 3]
    assert contract.num_inference_steps == 3
    torch.testing.assert_close(
        contract.snr_t,
        contract.alpha_t.double().square() / contract.sigma_t.double().square(),
    )


def test_ddpm_generation_schedule_and_active_scheduler_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "run" / "experiment_S0_N2")
    path_builder = Mock(return_value=paths)
    monkeypatch.setattr(experiment, "generation_paths", path_builder)
    monkeypatch.setattr(
        experiment,
        "require_generation_run",
        lambda _paths: {
            "scientific_config": _scheduler_science("ddpm"),
            "scientific_config_hash": EVALUATION_HASH,
        },
    )
    monkeypatch.setattr(
        experiment,
        "list_completed_records",
        lambda _paths: [_completed(tmp_path, "10"), _completed(tmp_path, "11")],
    )
    monkeypatch.setattr(
        experiment,
        "_schedule_file_sha256",
        lambda _paths: SCHEDULE_HASH,
    )
    monkeypatch.setattr(
        experiment,
        "safe_torch_load",
        lambda _path: _scheduler_schedule("ddpm"),
    )

    identity = experiment._load_generation_identity(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
    )
    contract = experiment._load_generation_contract(identity)

    assert identity.scheduler_name == contract.scheduler_name == "ddpm"
    assert path_builder.call_args.kwargs["scheduler_name"] == "ddpm"
    assert contract.schedule_payload["scheduler_name"] == "ddpm"

    class ActiveDDPM:
        init_noise_sigma = 1.0

        def __init__(self) -> None:
            self.timesteps = contract.timesteps.clone()
            self.calls: list[tuple[int, torch.device]] = []

        def set_timesteps(self, steps: int, *, device: torch.device) -> None:
            self.calls.append((steps, device))

    active = ActiveDDPM()
    original_scheduler = object()
    builder = Mock(
        return_value=SimpleNamespace(
            scheduler=active,
            config=contract.scheduler_config,
        )
    )
    coefficients = Mock(
        return_value=(
            contract.alpha_t.clone(),
            contract.sigma_t.clone(),
            contract.alpha_t.square(),
        )
    )
    monkeypatch.setattr(experiment, "build_scheduler", builder)
    monkeypatch.setattr(experiment, "schedule_alpha_sigma", coefficients)
    components = SimpleNamespace(
        original_scheduler=original_scheduler,
        device=torch.device("cpu"),
    )

    assert experiment._validate_active_scheduler(components, contract) is active
    builder.assert_called_once_with(original_scheduler, "ddpm")
    assert active.calls == [(3, torch.device("cpu"))]
    assert torch.equal(coefficients.call_args.args[1], contract.timesteps)
    assert coefficients.call_args.kwargs == {"dtype": torch.float32}


@pytest.mark.parametrize(
    ("timesteps", "message"),
    (
        ((9, 6, -1), "strictly descending"),
        ((9, 9, 3), "strictly descending"),
        ((9, 6, 7), "strictly descending"),
        ((10, 6, 3), "outside the training schedule"),
    ),
)
def test_schedule_contract_rejects_invalid_timestep_order_or_bounds(
    timesteps: tuple[int, int, int],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity(tmp_path)
    payload = _schedule()
    payload["timesteps"] = torch.tensor(timesteps, dtype=torch.int64)
    monkeypatch.setattr(
        experiment, "_schedule_file_sha256", lambda _paths: SCHEDULE_HASH
    )
    monkeypatch.setattr(experiment, "safe_torch_load", lambda _path: payload)

    with pytest.raises(experiment.ExperimentError, match=message):
        experiment._load_generation_contract(identity)


def test_schedule_contract_rejects_digest_change_during_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity(tmp_path)
    observed_hashes = iter((SCHEDULE_HASH, OTHER_SCHEDULE_HASH))
    monkeypatch.setattr(
        experiment, "_schedule_file_sha256", lambda _paths: next(observed_hashes)
    )
    monkeypatch.setattr(experiment, "safe_torch_load", lambda _path: _schedule())

    with pytest.raises(experiment.ExperimentError, match="changed while loading"):
        experiment._load_generation_contract(identity)


def _baseline_artifact(
    mu_hat: torch.Tensor | None,
    *,
    source_schedule_sha256: str = SCHEDULE_HASH,
) -> SimpleNamespace:
    science = _science()
    saved_norm = 0.0 if mu_hat is None else float(mu_hat.square().mean().sqrt().item())
    return SimpleNamespace(
        mu_hat=mu_hat,
        mu_hat_sha256=MU_HASH,
        source_scientific_config_hash=BASELINE_HASH,
        source_schedule_sha256=source_schedule_sha256,
        baseline_seed_start=2,
        baseline_seeds=(2, 3, 4),
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        timestep=9,
        alpha_t=float(torch.tensor(0.10, dtype=torch.float32)),
        sigma_t=float(torch.sqrt(torch.tensor(0.99, dtype=torch.float32))),
        latent_shape=(1, 1, 2),
        metadata={
            "model_name": "sdv1",
            "model_id": science["model_id"],
            "model_revision": science["model_revision"],
            "timestep": 9,
            "alpha_T": float(torch.tensor(0.10, dtype=torch.float32)),
            "sigma_T": float(torch.sqrt(torch.tensor(0.99, dtype=torch.float32))),
            "baseline_seed_start": 2,
            "baseline_seeds": [2, 3, 4],
            "num_baseline_seeds": NUM_BASELINE_SEEDS,
            "mu_hat_norm_rmse": saved_norm,
        },
    )


def test_baseline_loader_uses_disjoint_B_seed_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mu_hat = torch.tensor([[[0.25, -0.5]]], dtype=torch.float64)
    calls: list[dict[str, object]] = []

    def load(_root: object, **kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return _baseline_artifact(mu_hat)

    monkeypatch.setattr(experiment, "load_unconditional_baseline", load)
    baseline = experiment._load_baseline_contract(
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        identity=_identity(tmp_path),
        load_tensor=True,
    )

    assert baseline.mu_hat is mu_hat
    assert baseline.baseline_seeds == (2, 3, 4)
    assert calls[0]["num_baseline_seeds"] == NUM_BASELINE_SEEDS
    assert calls[0]["load_tensor"] is True


def test_baseline_loader_rejects_a_different_source_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment,
        "load_unconditional_baseline",
        lambda *_args, **_kwargs: _baseline_artifact(
            None, source_schedule_sha256=OTHER_SCHEDULE_HASH
        ),
    )

    with pytest.raises(experiment.ExperimentError, match="schedule SHA-256"):
        experiment._load_baseline_contract(
            project_root=tmp_path,
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=3,
            num_seeds=2,
            num_baseline_seeds=NUM_BASELINE_SEEDS,
            identity=_identity(tmp_path),
            load_tensor=False,
        )


def test_plot_baseline_load_does_not_deserialize_mu_hat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def load(_root: object, **kwargs: object) -> SimpleNamespace:
        seen.append(bool(kwargs["load_tensor"]))
        return _baseline_artifact(None)

    monkeypatch.setattr(experiment, "load_unconditional_baseline", load)
    baseline = experiment._load_baseline_contract(
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        identity=_identity(tmp_path),
        load_tensor=False,
    )

    assert seen == [False]
    assert baseline.mu_hat is None


def test_trajectory_hash_combines_marker_pinned_files() -> None:
    marker = _marker("10")

    assert experiment._trajectory_sha256(marker) == _trajectory_hash("10")
    marker["tensor_file_sha256"] = {
        "latent": "not-a-hash",
        "noise_prediction": PREDICTION_HASHES["10"],
    }
    with pytest.raises(experiment.ExperimentError, match="hashes are invalid"):
        experiment._trajectory_sha256(marker)


@pytest.mark.parametrize("verify_file_hashes", (False, True))
def test_selected_record_validation_pins_identity_science_and_hash_policy(
    verify_file_hashes: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def validate(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(dict(kwargs))
        return SimpleNamespace(valid=True, metadata=_marker("10"), errors=())

    monkeypatch.setattr(experiment, "validate_generation_record", validate)
    prompt = experiment._included_prompt_rows(_selection(tmp_path))[0]
    marker = experiment._validate_generation_prompt(
        prompt,
        identity=_identity(tmp_path),
        verify_file_hashes=verify_file_hashes,
    )

    assert marker["record_id"] == "record-10"
    assert calls[0]["tensor_names"] == ("latent", "noise_prediction")
    assert calls[0]["load_tensors"] is False
    assert calls[0]["require_preview"] is False
    assert calls[0]["verify_file_hashes"] is verify_file_hashes
    assert calls[0]["expected_record_identity"] == {
        "record_id": "record-10",
        "source_row_number": 10,
        "prompt_raw": "first selected prompt",
        "target_image_sha256": "5" * 64,
    }


def test_generation_marker_scientific_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = _marker("10")
    marker["num_inference_steps"] = 99
    monkeypatch.setattr(
        experiment,
        "validate_generation_record",
        lambda *_args, **_kwargs: SimpleNamespace(
            valid=True, metadata=marker, errors=()
        ),
    )
    prompt = experiment._included_prompt_rows(_selection(tmp_path))[0]

    with pytest.raises(experiment.ExperimentError, match="num_inference_steps"):
        experiment._validate_generation_prompt(
            prompt,
            identity=_identity(tmp_path),
            verify_file_hashes=True,
        )


def _cached_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = torch.tensor(
        [
            [[[[1.0, 2.0]]], [[[2.0, 3.0]]], [[[3.0, 4.0]]], [[[4.0, 5.0]]]],
            [[[[5.0, 6.0]]], [[[6.0, 7.0]]], [[[7.0, 8.0]]], [[[8.0, 9.0]]]],
        ],
        dtype=torch.float32,
    ).contiguous()
    unconditional = torch.tensor(
        [
            [[[[0.1, 0.2]]], [[[0.3, 0.4]]], [[[0.5, 0.6]]]],
            [[[[0.7, 0.8]]], [[[0.9, 1.0]]], [[[1.1, 1.2]]]],
        ],
        dtype=torch.float32,
    ).contiguous()
    conditional = (unconditional + 10.0).contiguous()
    return latents, unconditional, conditional


def test_cached_loader_reads_full_latent_and_prediction_trajectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path)
    prompt = experiment._included_prompt_rows(_selection(tmp_path))[0]
    latents, unconditional, conditional = _cached_tensors()
    seen: list[Path] = []
    monkeypatch.setattr(
        experiment,
        "_validate_generation_prompt",
        lambda *_args, **_kwargs: _marker("10"),
    )

    def load(path: Path) -> object:
        seen.append(path)
        if path.parent.name == "latent":
            return latents
        return (unconditional, conditional)

    monkeypatch.setattr(experiment, "safe_torch_load", load)
    observed_latents, observed_empty, observed_hash = (
        experiment._load_cached_trajectory(prompt, contract)
    )

    assert observed_latents is latents
    assert observed_empty is unconditional
    assert observed_hash == _trajectory_hash("10")
    assert [path.parent.name for path in seen] == ["latent", "noise_pred"]


def test_gaussian_formula_reuses_one_standard_noise_batch_across_all_timesteps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract(tmp_path)
    raw = torch.tensor(
        [[[[1.0, 2.0]]], [[[3.0, 4.0]]]], dtype=torch.float32
    ).contiguous()
    center = torch.tensor([[[0.25, -0.75]]], dtype=torch.float64)
    noise_calls: list[tuple[int, ...]] = []
    prediction_calls: list[tuple[int, torch.Tensor]] = []

    def make_noise(seeds: Any, latent_shape: Any) -> torch.Tensor:
        noise_calls.append(tuple(seeds))
        assert tuple(latent_shape) == contract.latent_shape
        return raw.clone()

    epsilon_by_timestep = {9: 0.1, 6: 0.2, 3: 0.3}

    def predict(**kwargs: Any) -> torch.Tensor:
        timestep = int(kwargs["timestep"])
        samples = kwargs["conversion_sample"]
        prediction_calls.append((timestep, samples.detach().cpu().clone()))
        return torch.full_like(samples, epsilon_by_timestep[timestep])

    monkeypatch.setattr(experiment, "make_initial_noise", make_noise)
    monkeypatch.setattr(experiment, "predict_conditional_epsilon", predict)
    runtime = experiment._GaussianRuntime(
        contract=contract,
        components=SimpleNamespace(
            device=torch.device("cpu"),
            inference_dtype=torch.float32,
            unet=object(),
        ),
        scheduler=object(),
        empty_condition=torch.zeros((1, 1, 1), dtype=torch.float32),
    )

    observed = experiment._measure_gaussian_batch(runtime, ((0, 0), (1, 1)), center)

    assert noise_calls == [(0, 1)]
    assert [timestep for timestep, _sample in prediction_calls] == [9, 6, 3]
    for _timestep, sample in prediction_calls:
        torch.testing.assert_close(sample, raw, rtol=0.0, atol=0.0)
    assert observed.error == ""
    assert observed.values is not None
    expected = np.empty((2, 3), dtype=np.float64)
    for seed_position in range(2):
        for step_index, timestep in enumerate((9, 6, 3)):
            estimate = (
                raw[seed_position].double()
                - float(contract.sigma_t[step_index])
                * float(
                    torch.tensor(epsilon_by_timestep[timestep], dtype=torch.float32)
                )
            ) / float(contract.alpha_t[step_index])
            expected[seed_position, step_index] = float(
                (estimate - center).square().mean().sqrt()
            )
    np.testing.assert_array_equal(observed.values, expected)


def test_centered_formula_uses_each_cached_prompt_seed_timestep_cell(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    latents, unconditional, _conditional = _cached_tensors()
    mu_hat = torch.tensor([[[0.25, -0.75]]], dtype=torch.float64)

    observed = experiment._centered_trajectory_distances(
        latents,
        unconditional,
        mu_hat,
        contract=contract,
        device="cpu",
    )
    expected = np.empty((2, 3), dtype=np.float64)
    for seed_position in range(2):
        for step_index in range(3):
            estimate = (
                latents[seed_position, step_index].double()
                - float(contract.sigma_t[step_index])
                * unconditional[seed_position, step_index].double()
            ) / float(contract.alpha_t[step_index])
            expected[seed_position, step_index] = float(
                (estimate - mu_hat).square().mean().sqrt()
            )

    assert observed.shape == (2, 3)
    assert observed.dtype == np.float64
    np.testing.assert_array_equal(observed, expected)


def test_ddpm_cached_computation_reloads_for_plot_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _scheduler_contract(tmp_path, "ddpm")
    selection = _scheduler_selection(tmp_path, "ddpm")
    latents, unconditional, _conditional = _cached_tensors()
    center = torch.zeros(contract.latent_shape, dtype=torch.float64)
    distances = experiment._centered_trajectory_distances(
        latents,
        unconditional,
        center,
        contract=contract,
        device="cpu",
    )
    measurements = tuple(
        replace(measurement, values=distances.copy())
        for measurement in _measurements(tmp_path)
    )
    rows, failed = experiment._expand_measurements(
        measurements,
        experiment._included_prompt_rows(selection),
        selection=selection,
        contract=contract,
        centering=_zero_centering_contract(),
    )
    rows = experiment._attach_target_sscd(rows, TARGET_SSCD)

    assert failed == 0
    assert {row["scheduler_name"] for row in rows} == {"ddpm"}
    assert sum(bool(row["is_initial_timestep"]) for row in rows) == 4

    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)

    def validate_record(_paths: Path, index: str, **_kwargs: object) -> object:
        return SimpleNamespace(
            valid=True,
            metadata=_scheduler_marker(str(index), "ddpm"),
            errors=(),
        )

    monkeypatch.setattr(experiment, "validate_generation_record", validate_record)
    monkeypatch.setattr(
        experiment,
        "load_model_components",
        lambda *_args, **_kwargs: pytest.fail("DDPM plot loaded a model"),
    )
    render = Mock()
    monkeypatch.setattr(experiment, "_render_figure", render)

    experiment.plot_saved_results(
        csv_path,
        tmp_path,
        selection=selection,
        identity=contract.identity,
        centering=_zero_centering_contract(),
        evaluation_source="trajectory",
    )

    render.assert_called_once()
    plotted = render.call_args.kwargs["frame"]
    assert plotted["scheduler_name"].eq("ddpm").all()
    np.testing.assert_array_equal(
        plotted["centered_distance_rmse"].to_numpy().reshape(2, 2, 3),
        np.broadcast_to(distances, (2, *distances.shape)),
    )


def test_centered_formula_rejects_incomplete_or_nonfinite_cache(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    latents, unconditional, _conditional = _cached_tensors()
    mu_hat = torch.zeros((1, 1, 2), dtype=torch.float64)

    with pytest.raises(experiment.ExperimentError, match="tensor contract"):
        experiment._centered_trajectory_distances(
            latents[:, :-1],
            unconditional,
            mu_hat,
            contract=contract,
            device="cpu",
        )
    changed = unconditional.clone()
    changed[0, 2, 0, 0, 0] = math.nan
    with pytest.raises(experiment.ExperimentError, match="tensor contract"):
        experiment._centered_trajectory_distances(
            latents,
            changed,
            mu_hat,
            contract=contract,
            device="cpu",
        )


def test_expand_gaussian_measurements_emits_exact_N_times_T_grid_without_prompts(
    tmp_path: Path,
) -> None:
    rows, failed = experiment._expand_gaussian_measurements(
        _gaussian_measurements(),
        selection=_selection(tmp_path),
        contract=_contract(tmp_path),
        centering=_mu_centering_contract(None),
    )

    assert failed == 0
    assert len(rows) == 6
    assert all(tuple(row) == EXPECTED_COLUMNS for row in rows)
    assert [
        (row["evaluation_source"], row["generation_seed"], row["step_index"])
        for row in rows
    ] == [
        ("gaussian", 0, 0),
        ("gaussian", 0, 1),
        ("gaussian", 0, 2),
        ("gaussian", 1, 0),
        ("gaussian", 1, 1),
        ("gaussian", 1, 2),
    ]
    assert {row["record_id"] for row in rows} == {""}
    assert {row["original_index"] for row in rows} == {""}
    assert {row["trajectory_sha256"] for row in rows} == {""}
    assert [row["centered_distance_rmse"] for row in rows] == [
        1.1,
        1.2,
        1.3,
        2.1,
        2.2,
        2.3,
    ]


def test_expand_measurements_emits_complete_canonical_P_times_N_times_T_grid(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    rows, failed = experiment._expand_measurements(
        tuple(reversed(_measurements(tmp_path))),
        experiment._included_prompt_rows(selection),
        selection=selection,
        contract=_contract(tmp_path),
        centering=_mu_centering_contract(None),
    )

    assert failed == 0
    assert len(rows) == 12
    assert all(tuple(row) == EXPECTED_COLUMNS for row in rows)
    assert [
        (row["record_id"], row["generation_seed"], row["step_index"]) for row in rows
    ] == [
        ("record-10", 0, 0),
        ("record-10", 0, 1),
        ("record-10", 0, 2),
        ("record-10", 1, 0),
        ("record-10", 1, 1),
        ("record-10", 1, 2),
        ("record-11", 0, 0),
        ("record-11", 0, 1),
        ("record-11", 0, 2),
        ("record-11", 1, 0),
        ("record-11", 1, 1),
        ("record-11", 1, 2),
    ]
    assert [row["centered_distance_rmse"] for row in rows] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
    ]
    assert sum(bool(row["is_initial_timestep"]) for row in rows) == 4


def test_expand_measurements_preserves_one_error_row_per_prompt_seed_timestep(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    measurements = list(_measurements(tmp_path))
    measurements[1] = experiment._PromptMeasurement(
        position=1,
        original_index="11",
        trajectory_sha256="",
        values=None,
        error="ExperimentError: invalid cache",
    )
    rows, failed = experiment._expand_measurements(
        measurements,
        experiment._included_prompt_rows(selection),
        selection=selection,
        contract=_contract(tmp_path),
        centering=_mu_centering_contract(None),
    )
    failures = [row for row in rows if row["status"] == "error"]

    assert failed == 6
    assert len(rows) == 12
    assert len(failures) == 6
    assert {row["record_id"] for row in failures} == {"record-11"}
    assert all(math.isnan(float(row["centered_distance_rmse"])) for row in failures)


def test_single_device_gaussian_compute_loads_once_and_tracks_exact_N_times_T(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(tmp_path)
    contract = _contract(tmp_path)
    centering = _zero_centering_contract(
        torch.zeros(contract.latent_shape, dtype=torch.float64)
    )
    builds: list[torch.device] = []
    batches: list[tuple[tuple[int, int], ...]] = []

    class Progress:
        def __init__(self, **options: object) -> None:
            self.options = options
            self.n = 0

        def __enter__(self) -> Progress:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def update(self, amount: int) -> None:
            self.n += amount

    instances: list[Progress] = []

    def progress_factory(**kwargs: object) -> Progress:
        progress = Progress(**kwargs)
        instances.append(progress)
        return progress

    def build(_contract: Any, *, device: torch.device, worker_count: int) -> object:
        assert worker_count == 1
        builds.append(device)
        return object()

    def measure(
        _runtime: object,
        entries: tuple[tuple[int, int], ...],
        _center: torch.Tensor,
    ) -> experiment._GaussianBatchMeasurement:
        batches.append(entries)
        values = np.asarray(
            [[seed + 0.1, seed + 0.2, seed + 0.3] for _position, seed in entries],
            dtype=np.float64,
        )
        return experiment._GaussianBatchMeasurement(entries, values, "")

    monkeypatch.setattr(experiment, "_build_gaussian_runtime", build)
    monkeypatch.setattr(experiment, "_measure_gaussian_batch_safely", measure)
    rows, failed = experiment._compute_gaussian_rows(
        selection=selection,
        contract=contract,
        centering=centering,
        devices=(torch.device("cpu"),),
        progress_factory=progress_factory,
    )

    assert failed == 0
    assert len(rows) == 6
    assert builds == [torch.device("cpu")]
    assert batches == [((0, 0), (1, 1))]
    assert len(instances) == 1
    assert instances[0].options["total"] == 6
    assert instances[0].options["unit"] == "observation"
    assert instances[0].n == 6


def test_single_device_compute_has_one_exact_P_times_N_times_T_progress_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(tmp_path)
    prompts = experiment._included_prompt_rows(selection)

    class Progress:
        def __init__(self, **options: object) -> None:
            self.options = options
            self.n = 0
            self.closed = False

        def __enter__(self) -> Progress:
            return self

        def __exit__(self, *_args: object) -> None:
            self.closed = True

        def update(self, amount: int) -> None:
            self.n += amount

    instances: list[Progress] = []

    def progress_factory(**kwargs: object) -> Progress:
        value = Progress(**kwargs)
        instances.append(value)
        return value

    seen_devices: list[torch.device] = []

    def measure(
        position: int,
        prompt: Any,
        **kwargs: object,
    ) -> experiment._PromptMeasurement:
        seen_devices.append(kwargs["device"])
        return _measurements(tmp_path)[position]

    monkeypatch.setattr(experiment, "_measure_prompt_safely", measure)
    rows, failed = experiment._compute_rows(
        prompts,
        selection=selection,
        contract=_contract(tmp_path),
        centering=_mu_centering_contract(torch.zeros((1, 1, 2), dtype=torch.float64)),
        devices=(torch.device("cpu"),),
        progress_factory=progress_factory,
    )

    assert failed == 0
    assert len(rows) == 12
    assert seen_devices == [torch.device("cpu"), torch.device("cpu")]
    assert len(instances) == 1
    assert instances[0].options["total"] == 12
    assert instances[0].options["unit"] == "observation"
    assert instances[0].n == 12
    assert instances[0].closed


def test_canonical_measurements_reorders_workers_and_rejects_missing() -> None:
    values = (
        experiment._PromptMeasurement(1, "11", "a" * 64, np.ones((2, 3)), ""),
        experiment._PromptMeasurement(0, "10", "b" * 64, np.ones((2, 3)), ""),
    )

    assert [
        item.position for item in experiment._canonical_measurements(values, 2)
    ] == [
        0,
        1,
    ]
    with pytest.raises(experiment.ExperimentError, match="missing"):
        experiment._canonical_measurements(values[:1], 2)


def test_plot_validation_enforces_source_specific_grids_and_current_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(tmp_path)
    identity = _identity(tmp_path)
    centering = _mu_centering_contract(None)
    calls: list[str] = []

    def validate(prompt: Any, **_kwargs: object) -> dict[str, object]:
        index = str(prompt["original_index"])
        calls.append(index)
        return _marker(index)

    monkeypatch.setattr(experiment, "_validate_generation_prompt", validate)
    both = pd.DataFrame(_both_rows(tmp_path), columns=EXPECTED_COLUMNS)
    validated = experiment._validated_plot_frame(
        both,
        selection=selection,
        identity=identity,
        centering=centering,
        evaluation_source="both",
    )

    assert len(validated) == 18
    assert validated["evaluation_source"].tolist()[:6] == ["gaussian"] * 6
    assert validated["evaluation_source"].tolist()[6:] == ["trajectory"] * 12
    assert calls == ["10", "11"]

    changed_score = both.copy()
    changed_score.loc[
        changed_score["evaluation_source"].eq("trajectory")
        & changed_score["record_id"].eq("record-10")
        & changed_score["generation_seed"].eq(0)
        & changed_score["step_index"].eq(1),
        "target_sscd",
    ] = 0.10000000000000002
    with pytest.raises(
        experiment.ExperimentError,
        match="same-seed endpoint target_sscd differs across trajectory timesteps",
    ):
        experiment._validated_plot_frame(
            changed_score,
            selection=selection,
            identity=identity,
            centering=centering,
            evaluation_source="both",
        )

    gaussian = both.iloc[:6].copy()
    monkeypatch.setattr(
        experiment,
        "_validate_generation_prompt",
        lambda *_args, **_kwargs: pytest.fail(
            "Gaussian-only plot validation inspected prompt trajectories"
        ),
    )
    experiment._validated_plot_frame(
        gaussian,
        selection=selection,
        identity=identity,
        centering=centering,
        evaluation_source="gaussian",
    )

    incomplete = both.drop(columns="evaluation_source")
    with pytest.raises(experiment.ExperimentError, match="columns.*schema"):
        experiment._validated_plot_frame(
            incomplete,
            selection=selection,
            identity=identity,
            centering=centering,
            evaluation_source="both",
        )
    with pytest.raises(experiment.ExperimentError, match="rows for"):
        experiment._validated_plot_frame(
            both,
            selection=selection,
            identity=identity,
            centering=centering,
            evaluation_source="gaussian",
        )
    changed = gaussian.copy()
    changed.loc[0, "record_id"] = "prompt-should-be-empty"
    with pytest.raises(experiment.ExperimentError, match="record_id"):
        experiment._validated_plot_frame(
            changed,
            selection=selection,
            identity=identity,
            centering=centering,
            evaluation_source="gaussian",
        )


def test_plot_validation_requires_exact_P_times_N_times_T_grid_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(_rows(tmp_path), columns=EXPECTED_COLUMNS)
    calls: list[tuple[str, bool]] = []

    def validate(prompt: Any, **kwargs: object) -> dict[str, object]:
        calls.append(
            (str(prompt["original_index"]), bool(kwargs["verify_file_hashes"]))
        )
        return _marker(str(prompt["original_index"]))

    monkeypatch.setattr(experiment, "_validate_generation_prompt", validate)
    validated = experiment._validated_plot_frame(
        frame,
        selection=_selection(tmp_path),
        identity=_identity(tmp_path),
        centering=_mu_centering_contract(None),
    )

    assert len(validated) == 12
    assert calls == [("10", False), ("11", False)]
    with pytest.raises(experiment.ExperimentError, match="rows for"):
        experiment._validated_plot_frame(
            frame.iloc[:-1],
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
        )
    changed = frame.copy()
    changed.loc[6:, "trajectory_sha256"] = "e" * 64
    with pytest.raises(experiment.ExperimentError, match="trajectory_sha256"):
        experiment._validated_plot_frame(
            changed,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
        )

    changed = frame.copy()
    changed["evaluation_schedule_sha256"] = OTHER_SCHEDULE_HASH
    with pytest.raises(experiment.ExperimentError, match="evaluation_schedule_sha256"):
        experiment._validated_plot_frame(
            changed,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
        )

    with pytest.raises(experiment.ExperimentError, match="schedule SHA-256"):
        experiment._validated_plot_frame(
            frame,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(
                None, source_schedule_sha256=OTHER_SCHEDULE_HASH
            ),
        )


def test_plot_validation_rejects_noncanonical_prompt_seed_or_step_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment,
        "_validate_generation_prompt",
        lambda prompt, **_kwargs: _marker(str(prompt["original_index"])),
    )
    frame = pd.DataFrame(_rows(tmp_path), columns=EXPECTED_COLUMNS)
    for column, first, second, pattern in (
        ("record_id", 0, 6, "record_id"),
        ("generation_seed", 0, 3, "generation_seed"),
        ("step_index", 0, 1, "step_index"),
    ):
        changed = frame.copy()
        changed.loc[first, column], changed.loc[second, column] = (
            changed.loc[second, column],
            changed.loc[first, column],
        )
        with pytest.raises(experiment.ExperimentError, match=pattern):
            experiment._validated_plot_frame(
                changed,
                selection=_selection(tmp_path),
                identity=_identity(tmp_path),
                centering=_mu_centering_contract(None),
            )


def test_plot_validation_rejects_invalid_cached_schedule_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment,
        "_validate_generation_prompt",
        lambda prompt, **_kwargs: _marker(str(prompt["original_index"])),
    )
    base = pd.DataFrame(_rows(tmp_path), columns=EXPECTED_COLUMNS)

    negative = base.copy()
    negative.loc[negative["step_index"] == 1, "timestep"] = -1
    with pytest.raises(experiment.ExperimentError, match="strictly descending"):
        experiment._validated_plot_frame(
            negative,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
        )

    duplicate = base.copy()
    duplicate.loc[duplicate["step_index"] == 1, "timestep"] = 9
    with pytest.raises(experiment.ExperimentError, match="strictly descending"):
        experiment._validated_plot_frame(
            duplicate,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
        )

    ascending = base.copy()
    ascending.loc[ascending["step_index"] == 2, "timestep"] = 7
    with pytest.raises(experiment.ExperimentError, match="strictly descending"):
        experiment._validated_plot_frame(
            ascending,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
        )

    out_of_range = base.copy()
    out_of_range.loc[out_of_range["step_index"] == 0, "timestep"] = 10
    with pytest.raises(experiment.ExperimentError, match="training schedule"):
        experiment._validated_plot_frame(
            out_of_range,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
        )

    inconsistent = base.copy()
    selected = inconsistent["step_index"] == 1
    inconsistent.loc[selected, "sigma_t"] = 0.5
    inconsistent.loc[selected, "snr_t"] = (
        inconsistent.loc[selected, "alpha_t"].astype(float).pow(2) / 0.25
    )
    with pytest.raises(experiment.ExperimentError, match="coefficient identity"):
        experiment._validated_plot_frame(
            inconsistent,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
        )


def test_multi_cuda_gaussian_path_shards_seeds_and_restores_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialized_devices: list[str] = []
    submitted: list[tuple[tuple[int, int], ...]] = []

    class ImmediateFuture:
        def __init__(self, result: experiment._GaussianBatchMeasurement) -> None:
            self._result = result

        def result(self) -> experiment._GaussianBatchMeasurement:
            return self._result

    class Executor:
        def __init__(self, **kwargs: object) -> None:
            initargs = kwargs["initargs"]
            assert isinstance(initargs, tuple)
            initialized_devices.append(str(initargs[2]))

        def __enter__(self) -> Executor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self,
            _function: object,
            entries: tuple[tuple[int, int], ...],
        ) -> ImmediateFuture:
            submitted.append(entries)
            values = np.asarray(
                [[seed + 0.1, seed + 0.2, seed + 0.3] for _position, seed in entries],
                dtype=np.float64,
            )
            return ImmediateFuture(
                experiment._GaussianBatchMeasurement(entries, values, "")
            )

    class Progress:
        def __init__(self, **kwargs: object) -> None:
            self.total = int(kwargs["total"])
            self.n = 0

        def __enter__(self) -> Progress:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def update(self, amount: int) -> None:
            self.n += amount

    progress = Progress(total=0)
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        experiment, "as_completed", lambda futures: tuple(reversed(tuple(futures)))
    )

    def progress_factory(**kwargs: object) -> Progress:
        nonlocal progress
        progress = Progress(**kwargs)
        return progress

    contract = _contract(tmp_path)
    rows, failed = experiment._compute_gaussian_rows(
        selection=_selection(tmp_path),
        contract=contract,
        centering=_zero_centering_contract(
            torch.zeros(contract.latent_shape, dtype=torch.float64)
        ),
        devices=(torch.device("cuda:0"), torch.device("cuda:1")),
        progress_factory=progress_factory,
    )

    assert failed == 0
    assert initialized_devices == ["cuda:0", "cuda:1"]
    assert submitted == [((0, 0),), ((1, 1),)]
    assert progress.total == 6
    assert progress.n == 6
    assert [row["generation_seed"] for row in rows] == [0, 0, 0, 1, 1, 1]


def test_multi_cuda_path_round_robins_prompts_and_restores_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(tmp_path)
    prompts = experiment._included_prompt_rows(selection)
    initialized_devices: list[str] = []
    submitted: list[int] = []

    class ImmediateFuture:
        def __init__(self, result: experiment._PromptMeasurement) -> None:
            self._result = result

        def result(self) -> experiment._PromptMeasurement:
            return self._result

    class Executor:
        def __init__(self, **kwargs: object) -> None:
            initargs = kwargs["initargs"]
            assert isinstance(initargs, tuple)
            initialized_devices.append(str(initargs[2]))

        def __enter__(self) -> Executor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self, _function: object, position: int, _prompt: object
        ) -> ImmediateFuture:
            submitted.append(position)
            return ImmediateFuture(_measurements(tmp_path)[position])

    class Progress:
        def __init__(self, **kwargs: object) -> None:
            self.total = int(kwargs["total"])
            self.n = 0

        def __enter__(self) -> Progress:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def update(self, amount: int) -> None:
            self.n += amount

    progress = Progress(total=0)
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        experiment, "as_completed", lambda futures: tuple(reversed(tuple(futures)))
    )

    def progress_factory(**kwargs: object) -> Progress:
        nonlocal progress
        progress = Progress(**kwargs)
        return progress

    rows, failed = experiment._compute_rows(
        prompts,
        selection=selection,
        contract=_contract(tmp_path),
        centering=_mu_centering_contract(torch.zeros((1, 1, 2), dtype=torch.float64)),
        devices=(torch.device("cuda:0"), torch.device("cuda:1")),
        progress_factory=progress_factory,
    )

    assert failed == 0
    assert initialized_devices == ["cuda:0", "cuda:1"]
    assert submitted == [0, 1]
    assert progress.total == 12
    assert progress.n == 12
    assert [row["record_id"] for row in rows[:7]] == [
        "record-10",
        "record-10",
        "record-10",
        "record-10",
        "record-10",
        "record-10",
        "record-11",
    ]


def test_run_both_uses_one_progress_bar_and_canonical_source_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection(tmp_path)
    identity = _identity(tmp_path)
    contract = _contract(tmp_path)
    centering = _zero_centering_contract(
        torch.zeros(contract.latent_shape, dtype=torch.float64)
    )
    output = tmp_path / "output"
    output.mkdir()
    seen_progress: list[object] = []
    written: dict[str, Any] = {}

    class Progress:
        def __init__(self, **kwargs: object) -> None:
            self.options = kwargs
            self.n = 0

        def __enter__(self) -> Progress:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def update(self, amount: int) -> None:
            self.n += amount

    instances: list[Progress] = []

    def progress_factory(**kwargs: object) -> Progress:
        progress = Progress(**kwargs)
        instances.append(progress)
        return progress

    monkeypatch.setattr(
        experiment, "_load_frozen_selection", lambda *_args, **_kwargs: selection
    )
    monkeypatch.setattr(
        experiment, "_load_generation_identity", lambda *_args, **_kwargs: identity
    )
    monkeypatch.setattr(
        experiment, "_load_centering_contract", lambda **_kwargs: centering
    )
    monkeypatch.setattr(
        experiment, "_load_generation_contract", lambda _identity: contract
    )
    monkeypatch.setattr(
        experiment, "_requested_output_directory", lambda **_kwargs: output
    )
    monkeypatch.setattr(
        experiment, "_prepare_output_directory", lambda _path, _source: output
    )
    monkeypatch.setattr(
        experiment, "resolve_devices", lambda _device: (torch.device("cpu"),)
    )
    load_target_sscd = Mock(return_value=TARGET_SSCD)
    monkeypatch.setattr(
        experiment,
        "_load_target_sscd_lookup",
        load_target_sscd,
    )

    def gaussian(**kwargs: object) -> tuple[list[dict[str, object]], int]:
        seen_progress.append(kwargs["progress"])
        kwargs["progress"].update(6)
        return _gaussian_rows(tmp_path), 0

    def trajectory(
        _prompts: Any, **kwargs: object
    ) -> tuple[list[dict[str, object]], int]:
        seen_progress.append(kwargs["progress"])
        kwargs["progress"].update(12)
        return _rows(tmp_path), 0

    monkeypatch.setattr(experiment, "_compute_gaussian_rows", gaussian)
    monkeypatch.setattr(experiment, "_compute_rows", trajectory)

    def write(_path: Path, rows: Any, columns: Any) -> None:
        written["sources"] = [row["evaluation_source"] for row in rows]
        written["target_sscd"] = [row["target_sscd"] for row in rows]
        written["columns"] = tuple(columns)

    monkeypatch.setattr(experiment, "atomic_write_csv", write)
    monkeypatch.setattr(
        experiment, "plot_saved_results", lambda *_args, **_kwargs: None
    )

    result = experiment.run_experiment(
        model_name="sdv1",
        selection_strategy="gmm",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        output_dir=output,
        evaluation_source="both",
        device="cpu",
        progress_factory=progress_factory,
    )

    assert result == 0
    assert len(instances) == 1
    assert instances[0].options["total"] == 18
    assert instances[0].n == 18
    assert seen_progress == [instances[0], instances[0]]
    assert written["sources"] == ["gaussian"] * 6 + ["trajectory"] * 12
    assert all(math.isnan(score) for score in written["target_sscd"][:6])
    assert written["target_sscd"][6:] == [
        score for score in TARGET_SSCD.values() for _step in range(3)
    ]
    load_target_sscd.assert_called_once()
    assert written["columns"] == EXPECTED_COLUMNS


@pytest.mark.parametrize("use_mu", (False, True))
def test_plot_only_never_loads_schedule_trajectory_or_device(
    use_mu: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection(tmp_path)
    identity = _identity(tmp_path)
    baseline = _baseline_contract(None)
    output = tmp_path / "output"
    output.mkdir()
    (output / experiment.CSV_NAME).touch()
    baseline_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        experiment, "_load_frozen_selection", lambda *_args, **_kwargs: selection
    )
    monkeypatch.setattr(
        experiment, "_load_generation_identity", lambda *_args, **_kwargs: identity
    )

    def load_baseline(**kwargs: object) -> experiment.BaselineContract:
        baseline_calls.append(dict(kwargs))
        return baseline

    monkeypatch.setattr(experiment, "_load_baseline_contract", load_baseline)
    monkeypatch.setattr(
        experiment, "_prepare_output_directory", lambda _path, _source: output
    )
    monkeypatch.setattr(
        experiment, "_remove_figure_outputs", lambda _path, _source: None
    )
    monkeypatch.setattr(
        experiment, "plot_saved_results", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        experiment,
        "_load_generation_contract",
        lambda *_args, **_kwargs: pytest.fail("plot mode loaded schedule.pt"),
    )
    monkeypatch.setattr(
        experiment,
        "_build_gaussian_runtime",
        lambda *_args, **_kwargs: pytest.fail("plot mode loaded a model"),
    )
    monkeypatch.setattr(
        experiment,
        "_load_cached_trajectory",
        lambda *_args, **_kwargs: pytest.fail("plot mode loaded trajectories"),
    )
    monkeypatch.setattr(
        experiment,
        "resolve_devices",
        lambda *_args, **_kwargs: pytest.fail("plot mode selected devices"),
    )
    arguments = SimpleNamespace(
        model="sdv1",
        scheduler="ddim",
        g=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        selection_strategy="gmm",
        output_dir=output,
        use_mu=use_mu,
        evaluation_source="both",
    )

    assert experiment._plot_only(arguments) == 0
    assert [call["load_tensor"] for call in baseline_calls] == (
        [False] if use_mu else []
    )


def test_trajectory_figure_has_sscd_colored_curves_bands_and_marker_free_median(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(_rows(tmp_path), columns=EXPECTED_COLUMNS)
    captured: dict[str, Any] = {}

    def save(figure: Any, destinations: Any) -> None:
        captured["figure"] = figure
        captured["destinations"] = tuple(destinations)

    monkeypatch.setattr(experiment, "_atomic_save_figures", save)
    destinations = (tmp_path / "figure.png", tmp_path / "figure.pdf")
    experiment._render_figure(
        frame=frame,
        destinations=destinations,
        centering_mode=experiment.CENTERING_MU_HAT,
    )
    figure = captured["figure"]
    axis = figure.axes[0]

    assert figure.get_size_inches().tolist() == [4.0, 4.0]
    assert axis.get_xscale() == "log"
    assert isinstance(axis.xaxis.get_major_locator(), LogLocator)
    assert isinstance(axis.xaxis.get_major_formatter(), LogFormatterMathtext)
    assert isinstance(axis.xaxis.get_minor_formatter(), NullFormatter)
    major_ticks = axis.xaxis.get_major_locator().tick_values(0.0058, 585.862)
    assert np.isclose(major_ticks, 1e-2).any()
    assert axis.get_xlabel() == r"$\mathrm{SNR}_t$"
    assert r"\widehat{\boldsymbol{\mu}}" in axis.get_ylabel()
    assert r"\|" in axis.get_ylabel()
    assert r"\|_2" not in axis.get_ylabel()
    assert axis.get_title() == ""
    assert axis.get_legend() is None
    assert len(figure.axes) == 2
    assert (
        len([item for item in axis.collections if isinstance(item, PolyCollection)])
        == 3
    )
    assert not any(isinstance(item, PathCollection) for item in axis.collections)
    observations = [
        item for item in axis.collections if isinstance(item, LineCollection)
    ]
    assert len(observations) == 1
    assert len(observations[0].get_segments()) == 4
    assert all(
        np.all(np.diff(segment[:, 0]) >= 0.0)
        for segment in observations[0].get_segments()
    )
    np.testing.assert_allclose(observations[0].get_array(), tuple(TARGET_SSCD.values()))
    assert observations[0].get_cmap().name == "viridis"
    assert observations[0].norm.vmin == 0.0
    assert observations[0].norm.vmax == 1.0
    assert observations[0].get_alpha() == experiment.OBSERVATION_LINE_ALPHA
    assert len(axis.lines) == 1
    assert axis.lines[0].get_color() == "black"
    assert axis.lines[0].get_marker() == "None"
    assert axis.xaxis.label.get_fontsize() == 15
    assert axis.yaxis.label.get_fontsize() == 15
    assert all(tick.get_fontsize() == 12 for tick in axis.get_xticklabels())
    colorbar_axis = figure.axes[1]
    assert colorbar_axis.get_ylabel() == "SSCD"
    assert colorbar_axis.yaxis.label.get_fontsize() == 15
    colorbar_mesh = [
        item for item in colorbar_axis.collections if isinstance(item, QuadMesh)
    ]
    assert len(colorbar_mesh) == 1
    assert colorbar_mesh[0].get_alpha() == 1.0
    assert experiment.PLOT_STYLE["legend.fontsize"] == 10
    assert captured["destinations"] == destinations


def test_gaussian_figure_has_one_neutral_curve_per_seed_and_no_colorbar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        experiment,
        "_atomic_save_figures",
        lambda figure, _destinations: captured.setdefault("figure", figure),
    )
    experiment._render_figure(
        frame=pd.DataFrame(_gaussian_rows(tmp_path), columns=EXPECTED_COLUMNS),
        destinations=(tmp_path / "gaussian.png", tmp_path / "gaussian.pdf"),
        centering_mode=experiment.CENTERING_ZERO,
    )

    figure = captured["figure"]
    axis = figure.axes[0]
    observations = [
        item for item in axis.collections if isinstance(item, LineCollection)
    ]
    assert len(figure.axes) == 1
    assert len(observations) == 1
    assert len(observations[0].get_segments()) == 2
    assert observations[0].get_array() is None
    assert len(axis.lines) == 1
    assert axis.lines[0].get_marker() == "None"
    assert axis.get_legend() is None


def test_plot_saved_results_writes_primary_gaussian_and_trajectory_suffixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(_both_rows(tmp_path), columns=EXPECTED_COLUMNS).to_csv(
        csv_path, index=False
    )
    monkeypatch.setattr(
        experiment,
        "_validate_generation_prompt",
        lambda prompt, **_kwargs: _marker(str(prompt["original_index"])),
    )
    rendered: list[tuple[str, tuple[str, ...]]] = []

    def render(*, frame: pd.DataFrame, destinations: Any, **_kwargs: object) -> None:
        rendered.append(
            (
                str(frame["evaluation_source"].iloc[0]),
                tuple(path.name for path in destinations),
            )
        )

    monkeypatch.setattr(experiment, "_render_figure", render)
    experiment.plot_saved_results(
        csv_path,
        tmp_path,
        selection=_selection(tmp_path),
        identity=_identity(tmp_path),
        centering=_mu_centering_contract(None),
        evaluation_source="both",
    )

    assert rendered == [
        ("gaussian", experiment.FIGURE_FILENAMES),
        ("trajectory", experiment.TRAJECTORY_FIGURE_FILENAMES),
    ]
    assert tuple(path.name for path in experiment._figure_paths(tmp_path, "both")) == (
        *experiment.FIGURE_FILENAMES,
        *experiment.TRAJECTORY_FIGURE_FILENAMES,
    )


def test_plot_reader_preserves_sscd_across_mixed_type_chunk_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gaussian = dict(_gaussian_rows(tmp_path)[0])
    filler = dict(_rows(tmp_path)[0])
    crossing = dict(filler)
    score_text = "0.06689324975013733"
    crossing.update(record_id="crossing-record", generation_seed=15)
    rows = (
        [gaussian] * 1_000
        + [filler] * 31_750
        + [{**crossing, "target_sscd": score_text}] * 50
        + [filler] * 32_737
    )
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    captured: dict[str, pd.DataFrame] = {}

    def validate(frame: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        captured["frame"] = frame
        return frame

    monkeypatch.setattr(experiment, "_validated_plot_frame", validate)
    monkeypatch.setattr(experiment, "_render_figure", lambda **_kwargs: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.DtypeWarning)
        experiment.plot_saved_results(
            csv_path,
            tmp_path,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
            evaluation_source="both",
        )

    parsed = captured["frame"]
    assert parsed["target_sscd"].dtype == np.dtype(np.float64)
    assert parsed.loc[:999, "target_sscd"].isna().all()
    crossing_scores = parsed.loc[
        parsed["record_id"].eq("crossing-record"), "target_sscd"
    ]
    assert crossing_scores.nunique(dropna=False) == 1
    assert crossing_scores.iloc[0].hex() == float(score_text).hex()


def test_plot_saved_results_requires_current_csv_without_cache_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = pd.DataFrame(_both_rows(tmp_path), columns=EXPECTED_COLUMNS).drop(
        columns="target_sscd"
    )
    csv_path = tmp_path / experiment.CSV_NAME
    incomplete.to_csv(csv_path, index=False)
    original = csv_path.read_bytes()
    for forbidden in (
        "_load_target_sscd_lookup",
        "load_model_components",
        "_build_gaussian_runtime",
        "_load_cached_trajectory",
        "_render_figure",
    ):
        monkeypatch.setattr(
            experiment,
            forbidden,
            lambda *_args, _name=forbidden, **_kwargs: pytest.fail(
                f"invalid CSV invoked {_name}"
            ),
        )
    with pytest.raises(experiment.ExperimentError, match="columns.*schema"):
        experiment.plot_saved_results(
            csv_path,
            tmp_path,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_mu_centering_contract(None),
            evaluation_source="both",
        )
    assert csv_path.read_bytes() == original


def test_output_directory_allows_only_requested_source_figures(tmp_path: Path) -> None:
    output = tmp_path / "evaluation_gaussian"
    output.mkdir()
    (output / experiment.CSV_NAME).touch()
    for filename in experiment.FIGURE_FILENAMES:
        (output / filename).touch()
    assert experiment._prepare_output_directory(output, "gaussian") == output.resolve()

    (output / experiment.TRAJECTORY_FIGURE_FILENAMES[0]).touch()
    with pytest.raises(experiment.ExperimentError, match="unexpected artifacts"):
        experiment._prepare_output_directory(output, "gaussian")


def test_default_outputs_separate_zero_and_mu_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    zero_output = experiment._default_output_directory(
        "sdv1",
        "ddim",
        7.5,
        3,
        2,
        NUM_BASELINE_SEEDS,
        False,
        "gmm",
        SELECTION_HASH,
    )
    mu_output = experiment._default_output_directory(
        "sdv1",
        "ddim",
        7.5,
        3,
        2,
        NUM_BASELINE_SEEDS,
        True,
        "gmm",
        SELECTION_HASH,
    )
    base = tmp_path / "outputs" / "sdv1_ddim_g7.5_T3_N2" / "lemma2_mean_convergence"

    assert zero_output == (base / "centering_zero" / SELECTION_HASH / "evaluation_both")
    assert mu_output == (
        base
        / "centering_mu_hat"
        / "baseline_S2_N3"
        / SELECTION_HASH
        / "evaluation_both"
    )


def test_zero_centering_never_loads_or_requires_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment,
        "_load_baseline_contract",
        lambda **_kwargs: pytest.fail("zero mode loaded a baseline artifact"),
    )
    compute = experiment._load_centering_contract(
        use_mu=False,
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        identity=_identity(tmp_path),
        load_tensor=True,
    )
    plot_only = experiment._load_centering_contract(
        use_mu=False,
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        identity=_identity(tmp_path),
        load_tensor=False,
    )

    assert compute.mode == experiment.CENTERING_ZERO
    assert compute.baseline is None
    assert compute.num_baseline_seeds == 0
    assert compute.baseline_scientific_hash == ""
    assert compute.baseline_mu_hat_sha256 == ""
    assert compute.center is not None
    assert compute.center.dtype == torch.float64
    assert compute.center.is_contiguous()
    assert torch.count_nonzero(compute.center).item() == 0
    assert plot_only.center is None


def test_use_mu_loads_the_saved_baseline_only_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mu_hat = torch.tensor([[[0.25, -0.5]]], dtype=torch.float64)
    baseline = _baseline_contract(mu_hat)
    calls: list[dict[str, object]] = []

    def load(**kwargs: object) -> experiment.BaselineContract:
        calls.append(dict(kwargs))
        return baseline

    monkeypatch.setattr(experiment, "_load_baseline_contract", load)
    centering = experiment._load_centering_contract(
        use_mu=True,
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        identity=_identity(tmp_path),
        load_tensor=True,
    )

    assert centering.mode == experiment.CENTERING_MU_HAT
    assert centering.center is mu_hat
    assert centering.baseline is baseline
    assert calls == [
        {
            "project_root": tmp_path,
            "model_name": "sdv1",
            "scheduler_name": "ddim",
            "guidance_scale": 7.5,
            "num_inference_steps": 3,
            "num_seeds": 2,
            "num_baseline_seeds": NUM_BASELINE_SEEDS,
            "identity": _identity(tmp_path),
            "load_tensor": True,
        }
    ]


def test_zero_centering_is_exactly_zero_and_not_merely_a_mode_label(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    exact = _zero_centering_contract(
        torch.zeros(contract.latent_shape, dtype=torch.float64)
    )
    experiment._validate_centering_against_schedule(exact, contract)

    nonzero = _zero_centering_contract(
        torch.full(
            contract.latent_shape,
            torch.finfo(torch.float64).tiny,
            dtype=torch.float64,
        )
    )
    with pytest.raises(experiment.ExperimentError, match="exact zero tensor"):
        experiment._validate_centering_against_schedule(nonzero, contract)


def test_zero_mode_logging_is_explicit_and_has_no_baseline_provenance(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    rows, failed = experiment._expand_measurements(
        _measurements(tmp_path),
        experiment._included_prompt_rows(selection),
        selection=selection,
        contract=_contract(tmp_path),
        centering=_zero_centering_contract(),
    )

    assert failed == 0
    assert {row["centering_mode"] for row in rows} == {experiment.CENTERING_ZERO}
    assert {row["num_baseline_seeds"] for row in rows} == {0}
    assert {row["baseline_generation_scientific_config_hash"] for row in rows} == {""}
    assert {row["baseline_mu_hat_sha256"] for row in rows} == {""}


def test_plot_validation_rejects_a_csv_from_the_other_centering_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(_rows(tmp_path), columns=EXPECTED_COLUMNS)
    monkeypatch.setattr(
        experiment,
        "_validate_generation_prompt",
        lambda prompt, **_kwargs: _marker(str(prompt["original_index"])),
    )

    with pytest.raises(experiment.ExperimentError, match="centering_mode"):
        experiment._validated_plot_frame(
            frame,
            selection=_selection(tmp_path),
            identity=_identity(tmp_path),
            centering=_zero_centering_contract(),
        )


def test_zero_mode_figure_ylabel_is_the_uncentered_norm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(_rows(tmp_path), columns=EXPECTED_COLUMNS)
    captured: dict[str, Any] = {}

    def save(figure: Any, _destinations: Any) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(experiment, "_atomic_save_figures", save)
    experiment._render_figure(
        frame=frame,
        destinations=(tmp_path / "figure.png", tmp_path / "figure.pdf"),
        centering_mode=experiment.CENTERING_ZERO,
    )
    ylabel = captured["figure"].axes[0].get_ylabel()

    assert ylabel == (r"$\|\widehat{\mathbf{x}}_{0\mid t,\emptyset}\|" r"/\sqrt{d}$")
    assert r"\boldsymbol{\mu}" not in ylabel


def test_zero_centering_computes_each_estimate_norm(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    latents, unconditional, _conditional = _cached_tensors()
    observed = experiment._centered_trajectory_distances(
        latents,
        unconditional,
        torch.zeros(contract.latent_shape, dtype=torch.float64),
        contract=contract,
        device="cpu",
    )
    expected = np.empty((2, 3), dtype=np.float64)
    for seed_position in range(2):
        for step_index in range(3):
            estimate = (
                latents[seed_position, step_index].double()
                - float(contract.sigma_t[step_index])
                * unconditional[seed_position, step_index].double()
            ) / float(contract.alpha_t[step_index])
            expected[seed_position, step_index] = float(estimate.square().mean().sqrt())

    np.testing.assert_array_equal(observed, expected)


def test_atomic_figure_writer_uses_moderate_dpi_for_png_and_pdf(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def savefig(destination: Path, **options: object) -> None:
        calls.append(options)
        destination.write_bytes(b"figure")

    destinations = (tmp_path / "figure.png", tmp_path / "figure.pdf")
    experiment._atomic_save_figures(SimpleNamespace(savefig=savefig), destinations)

    assert experiment.FIGURE_DPI == 150
    assert calls == [
        {
            "format": figure_format,
            "bbox_inches": "tight",
            "pad_inches": 0.05,
            "dpi": 150,
        }
        for figure_format in ("png", "pdf")
    ]
    assert all(destination.read_bytes() == b"figure" for destination in destinations)
    assert sorted(tmp_path.iterdir()) == sorted(destinations)
