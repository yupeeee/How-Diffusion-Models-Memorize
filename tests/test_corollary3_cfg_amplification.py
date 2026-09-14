from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
import importlib.util
import inspect
from pathlib import Path
from unittest.mock import Mock
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
import numpy as np
import pandas as pd
import pytest
import torch

from utils.data.selection import TargetPairSelection
from utils.experiments.cache import CompletedGenerationRecord, GenerationPaths
from utils.experiments.sscd import SCORE_DEFINITION, SSCDPaths


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "corollary3_cfg_amplification.py"
SPEC = importlib.util.spec_from_file_location("corollary3_cfg_amplification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)

SELECTION_HASH = "a" * 64
SCIENTIFIC_HASH = "b" * 64
SSCD_HASH = "c" * 64


BASELINE_HASH = "8" * 64
BASELINE_SCIENTIFIC_HASH = "7" * 64
BASELINE_SCHEDULE_HASH = "6" * 64
NUM_BASELINE_SEEDS = 3


def _mu_hat() -> torch.Tensor:
    return torch.tensor([[[0.25, -0.50]]], dtype=torch.float64)


def _baseline(
    root: Path, contract: experiment.GenerationContract, *, load_tensor: bool = True
) -> experiment.UnconditionalBaselineArtifact:
    return experiment.UnconditionalBaselineArtifact(
        mu_hat=_mu_hat() if load_tensor else None,
        mu_hat_sha256=BASELINE_HASH,
        metadata={
            "model_id": contract.science["model_id"],
            "model_revision": contract.science["model_revision"],
            "baseline_seed_start": len(contract.seeds),
            "baseline_seeds": list(
                range(
                    len(contract.seeds),
                    len(contract.seeds) + NUM_BASELINE_SEEDS,
                )
            ),
            "num_baseline_seeds": NUM_BASELINE_SEEDS,
        },
        metadata_path=root / "metadata.json",
        tensor_path=root / "mu_hat.pt",
        source_scientific_config_hash=BASELINE_SCIENTIFIC_HASH,
        source_schedule_sha256=BASELINE_SCHEDULE_HASH,
        baseline_seed_start=len(contract.seeds),
        baseline_seeds=tuple(
            range(len(contract.seeds), len(contract.seeds) + NUM_BASELINE_SEEDS)
        ),
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        timestep=contract.timestep,
        alpha_t=contract.alpha_t,
        sigma_t=contract.sigma_t,
        latent_shape=contract.latent_shape,
    )


def _selection(
    root: Path, prompt_count: int = 2, scheduler_name: str = "ddim"
) -> TargetPairSelection:
    rows: list[dict[str, object]] = []
    for position in range(prompt_count):
        for selection_seed in (20, 21):
            rows.append(
                {
                    "original_index": str(10 + position),
                    "record_id": f"record-{position}",
                    "source_row_number": position + 1,
                    "seed": selection_seed,
                    "prompt": f"prompt {position}",
                    "target_image_sha256": str(position + 1) * 64,
                    "include_prompt": True,
                    "selection_strategy": "gmm",
                    "selection_hash": SELECTION_HASH,
                }
            )
    return TargetPairSelection(
        root=root,
        model_name="sdv1",
        scheduler_name=scheduler_name,
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        selection_strategy="gmm",
        frame=pd.DataFrame(rows),
        configuration={"selection_hash": SELECTION_HASH},
        sha256=SELECTION_HASH,
    )


def _record(
    root: Path, position: int, scheduler_name: str = "ddim"
) -> CompletedGenerationRecord:
    index = str(10 + position)
    metadata = {
        "original_index": index,
        "record_id": f"record-{position}",
        "source_row_number": position + 1,
        "prompt_raw": f"prompt {position}",
        "target_image_sha256": str(position + 1) * 64,
        "model_cli_name": "sdv1",
        "dataset_model": "synthetic-dataset",
        "model_id": "model",
        "model_revision": "model-revision",
        "native_prediction_type": "epsilon",
        "target_preprocessing": {"synthetic": True},
        "scientific_config_hash": SCIENTIFIC_HASH,
        "scheduler_name": scheduler_name,
        "guidance_scale": 7.5,
        "num_inference_steps": 3,
        "num_seeds": 2,
        "seeds": [0, 1],
        "trajectory_order": "noise_to_image",
        "stored_prediction_type": "epsilon",
        "target_latent_definition": experiment.TARGET_LATENT_DEFINITION,
        "tensor_file_sha256": {
            "latent": "d" * 64,
            "noise_prediction": "e" * 64,
            "target_latent": "f" * 64,
        },
    }
    return CompletedGenerationRecord(
        original_index=index,
        source_row_number=position + 1,
        marker_path=root / "record" / f"{index}.json",
        metadata=metadata,
    )


def _contract(
    root: Path, prompt_count: int = 2, scheduler_name: str = "ddim"
) -> experiment.GenerationContract:
    run = root / "logs" / f"synthetic_{scheduler_name}" / "experiment_S0_N2"
    records = {
        str(10 + i): _record(run, i, scheduler_name) for i in range(prompt_count)
    }
    return experiment.GenerationContract(
        paths=GenerationPaths(run),
        sscd_paths=SSCDPaths(run),
        scheduler_name=scheduler_name,
        science={
            "model_cli_name": "sdv1",
            "model_id": "model",
            "dataset_model": "synthetic-dataset",
            "model_revision": "model-revision",
            "vae_id": "vae",
            "vae_revision": "vae-revision",
            "num_inference_steps": 3,
            "scheduler": {
                "name": scheduler_name,
                "config": {"num_train_timesteps": 10},
            },
            "native_prediction_type": "epsilon",
            "target_preprocessing": {"synthetic": True},
        },
        scientific_hash=SCIENTIFIC_HASH,
        schedule_sha256=BASELINE_SCHEDULE_HASH,
        seeds=(0, 1),
        latent_shape=(1, 1, 2),
        latent_dimension=2,
        stored_dtype=torch.float32,
        init_noise_sigma=1.0,
        timesteps=torch.tensor([9, 5, 1], dtype=torch.int64),
        alpha_values=torch.tensor([0.5, 0.7, 0.9], dtype=torch.float64),
        sigma_values=torch.tensor(
            [np.sqrt(0.75), np.sqrt(0.51), np.sqrt(0.19)], dtype=torch.float64
        ),
        snr_values=torch.tensor(
            [1.0 / 3.0, 0.49 / 0.51, 0.81 / 0.19], dtype=torch.float64
        ),
        records_by_index=records,
        sscd_configuration={"seeds": [0, 1]},
        sscd_configuration_hash=SSCD_HASH,
    )


def _row(selection: TargetPairSelection, position: int = 0) -> dict[str, object]:
    return dict(selection.prompt_frame.iloc[position])


def _values(
    position: int = 0,
    source: str = experiment.EVALUATION_GAUSSIAN,
) -> np.ndarray:
    base = np.asarray(
        [
            [7.0 + position, 0.10, 0.20, 0.30, 0.40, 0.15],
            [7.2 + position, 0.11, 0.21, 0.31, 0.41, 0.85],
        ],
        dtype=np.float64,
    )
    values = np.repeat(base[:, None, :], 3, axis=1)
    values[:, :, 0] += np.arange(3, dtype=np.float64)[None, :] * 0.05
    values[:, :, 1:5] += np.arange(3, dtype=np.float64)[None, :, None] * 0.01
    if source == experiment.EVALUATION_TRAJECTORY:
        values[:, :, :5] += 0.02
    return values


def _centering(
    contract: experiment.GenerationContract,
    *,
    use_mu: bool = True,
    load_tensor: bool = True,
) -> experiment.CenteringContract:
    if not use_mu:
        return experiment.CenteringContract(
            mode=experiment.CENTERING_ZERO,
            value=(
                torch.zeros(contract.latent_shape, dtype=torch.float64)
                if load_tensor
                else None
            ),
            baseline_generation_scientific_config_hash="",
            baseline_schedule_sha256="",
            baseline_mu_hat_sha256="",
            num_baseline_seeds=0,
        )
    return experiment.CenteringContract(
        mode=experiment.CENTERING_MU_HAT,
        value=_mu_hat() if load_tensor else None,
        baseline_generation_scientific_config_hash=BASELINE_SCIENTIFIC_HASH,
        baseline_schedule_sha256=BASELINE_SCHEDULE_HASH,
        baseline_mu_hat_sha256=BASELINE_HASH,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
    )


def _csv_rows(
    selection: TargetPairSelection,
    contract: experiment.GenerationContract,
    *,
    use_mu: bool = True,
    evaluation_source: str = experiment.EVALUATION_BOTH,
) -> list[dict[str, object]]:
    sources = experiment._evaluation_sources(evaluation_source)
    prompts = experiment._included_prompt_rows(selection)
    measurements = tuple(
        experiment._PromptMeasurement(
            position=position,
            original_index=str(prompt["original_index"]),
            values_by_source={source: _values(position, source) for source in sources},
            errors_by_source={source: "" for source in sources},
        )
        for position, prompt in enumerate(prompts)
    )
    rows: list[dict[str, object]] = []
    for source in sources:
        for position, prompt in enumerate(prompts):
            rows.extend(
                experiment._rows_for_measurement(
                    measurements[position],
                    prompt,
                    contract,
                    _centering(contract, use_mu=use_mu),
                    source,
                )
            )
    return rows


def test_cli_and_schema_are_exact() -> None:
    parser = experiment.build_parser()
    defaults = parser.parse_args([])
    custom = parser.parse_args(["--N", "7", "--num-baseline-seeds", "13", "--use-mu"])
    assert defaults.scheduler == "ddim"
    assert defaults.g == 7.5
    assert defaults.num_inference_steps == 50
    assert defaults.num_seeds == 20
    assert defaults.num_baseline_seeds == 1000
    assert defaults.selection_strategy == "gmm"
    assert (
        parser.parse_args(["--selection-strategy", "gmm"]).selection_strategy == "gmm"
    )
    for unsupported in ("spearman", "gmm-evidence", "all"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--selection-strategy", unsupported])
    assert defaults.use_mu is False
    assert defaults.evaluation_source == "both"
    assert custom.num_seeds == 7
    assert custom.num_baseline_seeds == 13
    assert custom.use_mu is True
    assert defaults.device == "auto"
    assert defaults.plot is False
    assert "(default: gmm)" in parser.format_help()
    assert experiment.CSV_COLUMNS == (
        "record_id",
        "generation_seed",
        "evaluation_source",
        "step_index",
        "timestep",
        "alpha_t",
        "sigma_t",
        "snr_t",
        "guidance_scale",
        "centering_mode",
        "baseline_generation_scientific_config_hash",
        "baseline_schedule_sha256",
        "baseline_mu_hat_sha256",
        "num_baseline_seeds",
        "fitted_guidance_scale",
        "residual_rmse",
        "guided_target_rmse",
        "conditional_recovery_rmse",
        "unconditional_rmse",
        "target_sscd",
        "status",
        "error",
    )
    assert experiment.FIGURE_SIZE == (4.0, 4.0)
    assert experiment.FIGURE_PAD_INCHES == 0.05


@pytest.mark.parametrize(
    ("scheduler", "guidance"),
    [("euler", 7.5), ("ddim", 7.5000001), ("ddpm", 1.0)],
)
def test_scientific_request_requires_supported_scheduler_and_exact_g(
    scheduler: str, guidance: float
) -> None:
    with pytest.raises(experiment.ExperimentError):
        experiment._validate_scientific_request(scheduler, guidance)


@pytest.mark.parametrize("scheduler", ("ddim", "ddpm"))
def test_scientific_request_accepts_both_cached_schedulers(scheduler: str) -> None:
    experiment._validate_scientific_request(scheduler, 7.5)


@pytest.mark.parametrize(
    ("scheduler_name", "scheduler_class"),
    (("ddim", "DDIMScheduler"), ("ddpm", "DDPMScheduler")),
)
def test_generation_contract_loader_pins_physical_schedule_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scheduler_name: str,
    scheduler_class: str,
) -> None:
    spec = experiment.get_model_spec("sdv1")
    scheduler_config = {"num_train_timesteps": 1000}
    science = {
        "model_cli_name": "sdv1",
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "model_revision": "model-revision",
        "vae_id": "vae",
        "vae_revision": "vae-revision",
        "guidance_scale": 7.5,
        "num_inference_steps": 3,
        "num_seeds": 2,
        "seeds": [0, 1],
        "selection_policy": experiment.GENERATION_SELECTION_POLICY,
        "trajectory_order": "noise_to_image",
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "target_latent_definition": experiment.TARGET_LATENT_DEFINITION,
        "target_preprocessing": experiment.target_preprocessing_policy(spec.resolution),
        "latent_shape": [1, 1, 2],
        "scientific_tensor_storage": {
            "dtype": "float32",
            "device": "cpu",
            "layout": "contiguous",
        },
        "scheduler": {
            "name": scheduler_name,
            "class": scheduler_class,
            "config": scheduler_config,
        },
    }
    run = tmp_path / "experiment_S0_N2"
    paths = GenerationPaths(run)
    paths.schedule.parent.mkdir(parents=True)
    paths.schedule.write_bytes(b"physical schedule")
    alpha = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    schedule = {
        "timesteps": torch.tensor([999, 666, 333], dtype=torch.int64),
        "alpha_t": alpha,
        "sigma_t": (1.0 - alpha.square()).sqrt().contiguous(),
        "alphas_cumprod_t": alpha.square().contiguous(),
        "init_noise_sigma": 1.0,
        "scheduler_name": scheduler_name,
        "scheduler_class": scheduler_class,
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
        "scheduler_config": scheduler_config,
    }
    monkeypatch.setattr(experiment, "generation_paths", Mock(return_value=paths))
    monkeypatch.setattr(
        experiment,
        "require_generation_run",
        Mock(
            return_value={
                "scientific_config": science,
                "scientific_config_hash": SCIENTIFIC_HASH,
            }
        ),
    )
    monkeypatch.setattr(
        experiment, "file_sha256", Mock(return_value=BASELINE_SCHEDULE_HASH)
    )
    monkeypatch.setattr(experiment, "safe_torch_load", Mock(return_value=schedule))
    monkeypatch.setattr(
        experiment,
        "list_completed_records",
        Mock(return_value=[_record(run, 0, scheduler_name)]),
    )
    sscd_loader = Mock(return_value=({}, SSCD_HASH))
    monkeypatch.setattr(experiment, "_load_sscd_configuration", sscd_loader)

    contract = experiment._load_generation_contract(
        tmp_path,
        model_name="sdv1",
        scheduler_name=scheduler_name,
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
    )

    assert contract.schedule_sha256 == BASELINE_SCHEDULE_HASH
    assert contract.timestep == 999
    assert contract.scheduler_name == scheduler_name
    sscd_loader.assert_called_once_with(
        SSCDPaths(run),
        science=science,
        scientific_hash=SCIENTIFIC_HASH,
        seeds=(0, 1),
    )


def _load_contract_with_schedule_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    timesteps: tuple[int, int, int],
    num_train_timesteps: int = 1000,
    physical_hashes: tuple[str, str] = (
        BASELINE_SCHEDULE_HASH,
        BASELINE_SCHEDULE_HASH,
    ),
) -> experiment.GenerationContract:
    spec = experiment.get_model_spec("sdv1")
    scheduler_config = {"num_train_timesteps": num_train_timesteps}
    science = {
        "model_cli_name": "sdv1",
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "guidance_scale": 7.5,
        "num_inference_steps": 3,
        "num_seeds": 2,
        "seeds": [0, 1],
        "selection_policy": experiment.GENERATION_SELECTION_POLICY,
        "trajectory_order": "noise_to_image",
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "target_latent_definition": experiment.TARGET_LATENT_DEFINITION,
        "target_preprocessing": experiment.target_preprocessing_policy(spec.resolution),
        "latent_shape": [1, 1, 2],
        "scientific_tensor_storage": {
            "dtype": "float32",
            "device": "cpu",
            "layout": "contiguous",
        },
        "scheduler": {
            "name": "ddim",
            "class": "DDIMScheduler",
            "config": scheduler_config,
        },
    }
    run = tmp_path / "experiment_S0_N2"
    paths = GenerationPaths(run)
    paths.schedule.parent.mkdir(parents=True)
    paths.schedule.write_bytes(b"physical schedule")
    alpha = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    schedule = {
        "timesteps": torch.tensor(timesteps, dtype=torch.int64),
        "alpha_t": alpha,
        "sigma_t": (1.0 - alpha.square()).sqrt().contiguous(),
        "alphas_cumprod_t": alpha.square().contiguous(),
        "init_noise_sigma": 1.0,
        "scheduler_name": "ddim",
        "scheduler_class": "DDIMScheduler",
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
        "scheduler_config": scheduler_config,
    }
    monkeypatch.setattr(experiment, "generation_paths", Mock(return_value=paths))
    monkeypatch.setattr(
        experiment,
        "require_generation_run",
        Mock(
            return_value={
                "scientific_config": science,
                "scientific_config_hash": SCIENTIFIC_HASH,
            }
        ),
    )
    monkeypatch.setattr(experiment, "file_sha256", Mock(side_effect=physical_hashes))
    monkeypatch.setattr(experiment, "safe_torch_load", Mock(return_value=schedule))
    monkeypatch.setattr(
        experiment, "list_completed_records", Mock(return_value=[_record(run, 0)])
    )
    monkeypatch.setattr(
        experiment, "_load_sscd_configuration", Mock(return_value=({}, SSCD_HASH))
    )
    return experiment._load_generation_contract(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
    )


@pytest.mark.parametrize(
    "timesteps",
    [(999, 333, 666), (999, 666, 666)],
)
def test_generation_contract_rejects_non_descending_ddim_timesteps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timesteps: tuple[int, int, int],
) -> None:
    with pytest.raises(experiment.ExperimentError, match="strictly descending"):
        _load_contract_with_schedule_variant(
            monkeypatch,
            tmp_path,
            timesteps=timesteps,
        )


def test_generation_contract_rejects_timestep_outside_training_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(
        experiment.ExperimentError, match="outside the training schedule"
    ):
        _load_contract_with_schedule_variant(
            monkeypatch,
            tmp_path,
            timesteps=(1000, 666, 333),
        )


def test_generation_contract_rejects_schedule_hash_changed_while_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(experiment.ExperimentError, match="changed while loading"):
        _load_contract_with_schedule_variant(
            monkeypatch,
            tmp_path,
            timesteps=(999, 666, 333),
            physical_hashes=(BASELINE_SCHEDULE_HASH, "5" * 64),
        )


@pytest.mark.parametrize("scheduler_name", ("ddim", "ddpm"))
def test_active_scheduler_rejects_changed_saved_coefficients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduler_name: str
) -> None:
    contract = _contract(tmp_path, scheduler_name=scheduler_name)
    scheduler_config = {"num_train_timesteps": 10}
    contract = replace(
        contract,
        science={**contract.science, "scheduler": {"config": scheduler_config}},
    )
    scheduler = SimpleNamespace(
        timesteps=contract.timesteps.clone(),
        init_noise_sigma=1.0,
        set_timesteps=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        experiment,
        "build_scheduler",
        Mock(
            return_value=SimpleNamespace(
                scheduler=scheduler,
                config=scheduler_config,
            )
        ),
    )
    active = (
        contract.alpha_values.clone(),
        contract.sigma_values.clone(),
        contract.alpha_values.float().square(),
    )
    coefficients = Mock(return_value=active)
    monkeypatch.setattr(experiment, "schedule_alpha_sigma", coefficients)
    components = SimpleNamespace(original_scheduler=object(), device="cpu")

    assert experiment._validate_active_scheduler(components, contract) is scheduler
    experiment.build_scheduler.assert_called_once_with(
        components.original_scheduler, scheduler_name
    )
    coefficient_call = coefficients.call_args
    assert coefficient_call.args[0] is scheduler
    assert torch.equal(coefficient_call.args[1], contract.timesteps)
    assert coefficient_call.kwargs == {"dtype": torch.float32}

    changed_alpha = active[0].clone()
    changed_alpha[1] = torch.nextafter(
        changed_alpha[1], torch.tensor(float("inf"), dtype=changed_alpha.dtype)
    )
    coefficients.return_value = (changed_alpha, active[1], active[2])
    with pytest.raises(experiment.ExperimentError, match="coefficients"):
        experiment._validate_active_scheduler(components, contract)


def test_shared_baseline_loader_uses_reference_seeds_and_metadata_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract(tmp_path)
    expected = _baseline(tmp_path, contract, load_tensor=False)
    loader = Mock(return_value=expected)
    monkeypatch.setattr(experiment, "load_unconditional_baseline", loader)

    observed = experiment._load_shared_baseline(
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        contract=contract,
        load_tensor=False,
    )

    assert observed is expected
    loader.assert_called_once_with(
        experiment.ROOT,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=3,
        num_seeds=2,
        num_baseline_seeds=NUM_BASELINE_SEEDS,
        load_tensor=False,
    )
    assert observed.baseline_seed_start == 2
    assert observed.num_baseline_seeds == NUM_BASELINE_SEEDS
    assert observed.baseline_seeds == (2, 3, 4)
    assert observed.mu_hat is None

    loader.return_value = replace(expected, baseline_seed_start=0)
    with pytest.raises(experiment.ExperimentError, match="baseline_seed_start"):
        experiment._load_shared_baseline(
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=3,
            num_seeds=2,
            num_baseline_seeds=NUM_BASELINE_SEEDS,
            contract=contract,
            load_tensor=False,
        )

    loader.return_value = replace(expected, num_baseline_seeds=2)
    with pytest.raises(experiment.ExperimentError, match="num_baseline_seeds"):
        experiment._load_shared_baseline(
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=3,
            num_seeds=2,
            num_baseline_seeds=NUM_BASELINE_SEEDS,
            contract=contract,
            load_tensor=False,
        )

    wrong_metadata = dict(expected.metadata)
    wrong_metadata["num_baseline_seeds"] = 2
    loader.return_value = replace(expected, metadata=wrong_metadata)
    with pytest.raises(experiment.ExperimentError, match="metadata num_baseline_seeds"):
        experiment._load_shared_baseline(
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=3,
            num_seeds=2,
            num_baseline_seeds=NUM_BASELINE_SEEDS,
            contract=contract,
            load_tensor=False,
        )

    loader.return_value = replace(expected, source_schedule_sha256="5" * 64)
    with pytest.raises(experiment.ExperimentError, match="schedule SHA-256"):
        experiment._load_shared_baseline(
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=3,
            num_seeds=2,
            num_baseline_seeds=NUM_BASELINE_SEEDS,
            contract=contract,
            load_tensor=False,
        )


def test_zero_centering_is_exact_float64_and_never_loads_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract(tmp_path)
    baseline_loader = Mock(side_effect=AssertionError("zero mode read baseline"))
    monkeypatch.setattr(experiment, "_load_shared_baseline", baseline_loader)
    options = {
        "use_mu": False,
        "model_name": "sdv1",
        "scheduler_name": "ddim",
        "guidance_scale": 7.5,
        "num_inference_steps": 3,
        "num_seeds": 2,
        "num_baseline_seeds": NUM_BASELINE_SEEDS,
        "contract": contract,
    }

    compute = experiment._load_centering_contract(**options, load_tensor=True)
    assert compute.mode == experiment.CENTERING_ZERO
    assert compute.value is not None
    assert compute.value.dtype == torch.float64
    assert compute.value.device.type == "cpu"
    assert tuple(compute.value.shape) == contract.latent_shape
    assert torch.equal(compute.value, torch.zeros_like(compute.value))
    assert compute.num_baseline_seeds == 0
    assert compute.baseline_generation_scientific_config_hash == ""
    assert compute.baseline_schedule_sha256 == ""
    assert compute.baseline_mu_hat_sha256 == ""

    plot = experiment._load_centering_contract(**options, load_tensor=False)
    assert plot.mode == experiment.CENTERING_ZERO
    assert plot.value is None
    baseline_loader.assert_not_called()


def test_no_intercept_lstsq_solves_all_seed_responses_jointly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_direction = torch.tensor(
        [[[1.0, 2.0], [-1.0, 0.5]]],
        dtype=torch.float64,
    )
    target_vector = target_direction.flatten()
    coefficients = torch.tensor([2.0, -1.25, 4.5], dtype=torch.float64)
    raw_residuals = torch.tensor(
        [
            [1.0, -0.5, 0.25, 2.0],
            [-1.5, 0.75, 2.0, -0.5],
            [0.5, 1.5, -2.5, 1.0],
        ],
        dtype=torch.float64,
    )
    residuals = (
        raw_residuals
        - ((raw_residuals @ target_vector) / target_vector.square().sum())[:, None]
        * target_vector
    )
    guided_direction = (coefficients[:, None] * target_vector + residuals).reshape(
        3, *target_direction.shape
    )
    real_lstsq = torch.linalg.lstsq
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def tracked_lstsq(
        design_matrix: torch.Tensor,
        response_matrix: torch.Tensor,
    ) -> object:
        calls.append((design_matrix.clone(), response_matrix.clone()))
        return real_lstsq(design_matrix, response_matrix)

    monkeypatch.setattr(experiment.torch.linalg, "lstsq", tracked_lstsq)

    fitted = experiment._solve_no_intercept_lstsq(
        target_direction,
        guided_direction,
    )

    assert len(calls) == 1
    design_matrix, response_matrix = calls[0]
    assert design_matrix.dtype == response_matrix.dtype == torch.float64
    assert design_matrix.shape == (target_vector.numel(), 1)
    assert response_matrix.shape == (target_vector.numel(), len(coefficients))
    torch.testing.assert_close(design_matrix, target_vector[:, None])
    torch.testing.assert_close(response_matrix, guided_direction.flatten(1).T)
    torch.testing.assert_close(fitted, coefficients, rtol=1e-13, atol=1e-13)
    solved_residuals = guided_direction.flatten(1) - fitted[:, None] * target_vector
    assert bool((solved_residuals.norm(dim=1) > 0.0).all())
    assert not torch.allclose(fitted, torch.full_like(fitted, 7.5))


def test_no_intercept_lstsq_rejects_nonfinite_rhs_before_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_direction = torch.ones((1, 2, 2), dtype=torch.float64)
    guided_direction = torch.zeros((2, 1, 2, 2), dtype=torch.float64)
    guided_direction[1, 0, 0, 0] = torch.nan
    solver = Mock(side_effect=AssertionError("non-finite RHS reached solver"))
    monkeypatch.setattr(experiment.torch.linalg, "lstsq", solver)

    with pytest.raises(
        experiment.ExperimentError,
        match="response matrix contains non-finite values",
    ):
        experiment._solve_no_intercept_lstsq(
            target_direction,
            guided_direction,
        )

    solver.assert_not_called()


@pytest.mark.parametrize("target_magnitude", (1.0, 10.0))
def test_measurement_lstsq_guard_is_stable_at_realistic_latent_dimension(
    tmp_path: Path,
    target_magnitude: float,
) -> None:
    latent_shape = (4, 64, 64)
    contract = replace(
        _contract(tmp_path),
        latent_shape=latent_shape,
        latent_dimension=int(np.prod(latent_shape)),
    )
    generator = torch.Generator().manual_seed(123)
    target = target_magnitude * torch.randn(
        latent_shape,
        generator=generator,
        dtype=torch.float64,
    )
    raw_residual = torch.randn(
        latent_shape,
        generator=generator,
        dtype=torch.float64,
    )
    target_vector = target.flatten()
    orthogonal_residual = (
        raw_residual
        - ((raw_residual.flatten() @ target_vector) / target_vector.square().sum())
        * target
    )
    xhat_c = torch.stack(
        (
            target,
            target + 1e-12 * orthogonal_residual,
        )
    )
    xhat_empty = torch.zeros_like(xhat_c)
    x_t = torch.zeros_like(xhat_c)
    epsilon_c = (x_t - contract.alpha_t * xhat_c) / contract.sigma_t
    epsilon_empty = (x_t - contract.alpha_t * xhat_empty) / contract.sigma_t

    result = experiment._measure_values(
        x_t,
        epsilon_empty,
        epsilon_c,
        target,
        torch.zeros(latent_shape, dtype=torch.float64),
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        contract=contract,
        device="cpu",
        centering_mode=experiment.CENTERING_ZERO,
    )

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:, 0], 7.5, rtol=0.0, atol=2e-13)
    assert 0.0 <= result[0, 1] < 1e-12
    assert 0.0 < result[1, 1] < 1e-10


def test_measurement_rejects_obviously_wrong_lstsq_solution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    target = torch.tensor([[[1.0, -0.5]]], dtype=torch.float64)
    xhat_c = target.unsqueeze(0).expand(len(contract.seeds), -1, -1, -1)
    xhat_empty = torch.zeros_like(xhat_c)
    x_t = torch.zeros_like(xhat_c)
    epsilon_c = (x_t - contract.alpha_t * xhat_c) / contract.sigma_t
    epsilon_empty = (x_t - contract.alpha_t * xhat_empty) / contract.sigma_t
    solver = Mock(
        return_value=SimpleNamespace(
            solution=torch.zeros((1, len(contract.seeds)), dtype=torch.float64)
        )
    )
    monkeypatch.setattr(experiment.torch.linalg, "lstsq", solver)

    with pytest.raises(
        experiment.ExperimentError,
        match="projection residual is not orthogonal",
    ):
        experiment._measure_values(
            x_t,
            epsilon_empty,
            epsilon_c,
            target,
            torch.zeros(contract.latent_shape, dtype=torch.float64),
            torch.tensor([0.2, 0.8], dtype=torch.float64),
            contract=contract,
            device="cpu",
            centering_mode=experiment.CENTERING_ZERO,
        )

    solver.assert_called_once()


def test_measurement_uses_requested_no_intercept_projection_and_four_rmses(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    target = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    x_t = torch.tensor([[[[0.4, -0.2]]], [[[0.7, 0.3]]]], dtype=torch.float32)
    xhat_c = torch.tensor([[[[1.1, 0.2]]], [[[0.8, -0.4]]]], dtype=torch.float64)
    xhat_empty = torch.tensor([[[[0.1, 0.3]]], [[[-0.2, 0.5]]]], dtype=torch.float64)
    sigma = contract.sigma_t
    alpha = contract.alpha_t
    epsilon_c = (x_t.double() - alpha * xhat_c) / sigma
    epsilon_empty = (x_t.double() - alpha * xhat_empty) / sigma

    mu_hat = _mu_hat()
    result = experiment._measure_values(
        x_t,
        epsilon_empty.float(),
        epsilon_c.float(),
        target,
        mu_hat,
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        contract=contract,
        device="cpu",
        centering_mode=experiment.CENTERING_MU_HAT,
    )
    recovered_c = (x_t.double() - sigma * epsilon_c.float().double()) / alpha
    recovered_empty = (x_t.double() - sigma * epsilon_empty.float().double()) / alpha
    guided = 7.5 * recovered_c + (1.0 - 7.5) * recovered_empty
    target_direction = target.double() - mu_hat
    guided_direction = guided - mu_hat
    target_squared = target_direction.flatten().square().sum()
    expected_fit = (
        guided_direction.flatten(1) @ target_direction.flatten()
    ) / target_squared
    expected_residual = (
        guided_direction - expected_fit.view(-1, 1, 1, 1) * target_direction
    )
    assert result[:, 0] == pytest.approx(expected_fit.numpy())
    assert result[:, 1] == pytest.approx(
        expected_residual.flatten(1).square().mean(1).sqrt().numpy()
    )
    assert result[:, 2] == pytest.approx(
        (guided - (mu_hat + 7.5 * target_direction))
        .flatten(1)
        .square()
        .mean(1)
        .sqrt()
        .numpy()
    )
    assert result[:, 3] == pytest.approx(
        (recovered_c - target.double()).flatten(1).square().mean(1).sqrt().numpy()
    )
    assert result[:, 4] == pytest.approx(
        (recovered_empty - mu_hat).flatten(1).square().mean(1).sqrt().numpy()
    )
    assert result[:, 5] == pytest.approx([0.2, 0.8])


def test_measurement_uses_xstar_and_origin_in_default_zero_mode(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    target = torch.tensor([[[1.0, -0.5]]], dtype=torch.float32)
    x_t = torch.tensor([[[[0.4, -0.2]]], [[[0.7, 0.3]]]], dtype=torch.float32)
    xhat_c = torch.tensor([[[[1.1, 0.2]]], [[[0.8, -0.4]]]], dtype=torch.float64)
    xhat_empty = torch.tensor([[[[0.1, 0.3]]], [[[-0.2, 0.5]]]], dtype=torch.float64)
    epsilon_c = (x_t.double() - contract.alpha_t * xhat_c) / contract.sigma_t
    epsilon_empty = (x_t.double() - contract.alpha_t * xhat_empty) / contract.sigma_t
    center = torch.zeros(contract.latent_shape, dtype=torch.float64)

    result = experiment._measure_values(
        x_t,
        epsilon_empty.float(),
        epsilon_c.float(),
        target,
        center,
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        contract=contract,
        device="cpu",
        centering_mode=experiment.CENTERING_ZERO,
    )
    recovered_c = (
        x_t.double() - contract.sigma_t * epsilon_c.float().double()
    ) / contract.alpha_t
    recovered_empty = (
        x_t.double() - contract.sigma_t * epsilon_empty.float().double()
    ) / contract.alpha_t
    guided = 7.5 * recovered_c + (1.0 - 7.5) * recovered_empty
    target_d = target.double()
    fitted = (guided.flatten(1) @ target_d.flatten()) / target_d.square().sum()
    residual = guided - fitted.view(-1, 1, 1, 1) * target_d

    assert result[:, 0] == pytest.approx(fitted.numpy())
    assert result[:, 1] == pytest.approx(
        residual.flatten(1).square().mean(1).sqrt().numpy()
    )
    assert result[:, 2] == pytest.approx(
        (guided - 7.5 * target_d).flatten(1).square().mean(1).sqrt().numpy()
    )
    assert result[:, 3] == pytest.approx(
        (recovered_c - target_d).flatten(1).square().mean(1).sqrt().numpy()
    )
    assert result[:, 4] == pytest.approx(
        recovered_empty.flatten(1).square().mean(1).sqrt().numpy()
    )


@pytest.mark.parametrize("scheduler_name", ("ddim", "ddpm"))
@pytest.mark.parametrize(
    ("use_mu", "centering_mode"),
    [
        (False, experiment.CENTERING_ZERO),
        (True, experiment.CENTERING_MU_HAT),
    ],
)
def test_all_timestep_gaussian_limit_recovers_exact_corollary(
    tmp_path: Path,
    use_mu: bool,
    centering_mode: str,
    scheduler_name: str,
) -> None:
    contract = _contract(tmp_path, scheduler_name=scheduler_name)
    center = _mu_hat() if use_mu else torch.zeros_like(_mu_hat())
    target = torch.tensor([[[1.25, -0.75]]], dtype=torch.float64)
    probes = torch.tensor(
        [[[[0.3, -0.9]]], [[[-1.1, 0.7]]]],
        dtype=torch.float64,
    )
    samples = probes[:, None].expand(
        len(contract.seeds),
        contract.num_inference_steps,
        *contract.latent_shape,
    )
    alpha = contract.alpha_values.view(1, -1, 1, 1, 1)
    sigma = contract.sigma_values.view(1, -1, 1, 1, 1)
    epsilon_empty = (samples - alpha * center) / sigma
    epsilon_c = (samples - alpha * target) / sigma
    scores = torch.tensor([0.2, 0.8], dtype=torch.float64)

    values = experiment._measure_all_steps(
        probes,
        epsilon_empty,
        epsilon_c,
        target,
        center,
        scores,
        contract=contract,
        device="cpu",
        centering_mode=centering_mode,
    )

    assert values.shape == (2, 3, 6)
    np.testing.assert_allclose(values[:, :, 0], 7.5, rtol=0.0, atol=1e-13)
    assert np.max(values[:, :, 1:5]) < 1e-13
    np.testing.assert_array_equal(
        values[:, :, 5], np.broadcast_to(scores.numpy()[:, None], (2, 3))
    )


def test_measurement_rejects_zero_target_direction_from_mu_hat(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    mu_hat = _mu_hat()
    x_t = torch.zeros((2, *contract.latent_shape), dtype=torch.float32)
    epsilon = torch.zeros_like(x_t)
    with pytest.raises(experiment.ExperimentError, match="target direction"):
        experiment._measure_values(
            x_t,
            epsilon,
            epsilon,
            mu_hat.float(),
            mu_hat,
            torch.tensor([0.2, 0.8], dtype=torch.float64),
            contract=contract,
            device="cpu",
            centering_mode=experiment.CENTERING_MU_HAT,
        )


def test_cfg_identity_is_independently_checked() -> None:
    shape = (2, 1, 1, 2)
    x_t = torch.ones(shape, dtype=torch.float64)
    empty = torch.full(shape, 0.1, dtype=torch.float64)
    conditional = torch.full(shape, 0.2, dtype=torch.float64)
    xhat_empty = (x_t - 0.8 * empty) / 0.6
    xhat_c = (x_t - 0.8 * conditional) / 0.6
    xhat_g = 7.5 * xhat_c + (1.0 - 7.5) * xhat_empty
    experiment._verify_cfg_identity(
        x_t,
        empty,
        conditional,
        xhat_empty,
        xhat_c,
        xhat_g,
        alpha_t=0.6,
        sigma_t=0.8,
        guidance_scale=7.5,
    )
    with pytest.raises(experiment.ExperimentError, match="CFG"):
        experiment._verify_cfg_identity(
            x_t,
            empty,
            conditional,
            xhat_empty,
            xhat_c,
            xhat_g + 1e-4,
            alpha_t=0.6,
            sigma_t=0.8,
            guidance_scale=7.5,
        )


def test_source_never_fits_tautological_cfg_branch_difference() -> None:
    source = inspect.getsource(experiment)
    assert "xhat_g - xhat_empty" not in source
    assert "xhat_c - xhat_empty" not in source
    assert "torch.linalg.lstsq(design_matrix, response_matrix)" in source
    assert "target_direction.flatten()[:, None]" in source
    assert "guided_direction.flatten(start_dim=1).T" in source
    required_gaussian_probe_symbols = (
        "load_model_components",
        "build_scheduler",
        "predict_conditional_epsilon",
        "encode_prompt_condition",
        "make_initial_noise",
    )
    assert all(name in source for name in required_gaussian_probe_symbols)


@pytest.mark.parametrize("scheduler_name", ("ddim", "ddpm"))
def test_generation_marker_pins_contract_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scheduler_name: str,
) -> None:
    selection = _selection(tmp_path, scheduler_name=scheduler_name)
    contract = _contract(tmp_path, scheduler_name=scheduler_name)
    metadata = contract.records_by_index["10"].metadata
    validator = Mock(
        return_value=SimpleNamespace(valid=True, metadata=metadata, errors=())
    )
    monkeypatch.setattr(experiment, "validate_generation_record", validator)

    observed = experiment._validate_generation_prompt(
        _row(selection),
        contract=contract,
        tensor_names=("latent",),
    )

    assert observed["scheduler_name"] == scheduler_name
    assert validator.call_args.kwargs["expected_scientific_hash"] == SCIENTIFIC_HASH
    assert validator.call_args.kwargs["tensor_names"] == ("latent",)

    wrong = dict(metadata)
    wrong["scheduler_name"] = "ddpm" if scheduler_name == "ddim" else "ddim"
    validator.return_value = SimpleNamespace(
        valid=True,
        metadata=wrong,
        errors=(),
    )
    with pytest.raises(experiment.ExperimentError, match="scheduler_name"):
        experiment._validate_generation_prompt(
            _row(selection), contract=contract, tensor_names=("latent",)
        )


def test_initial_latent_sentinel_is_bitwise_and_checks_only_one_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract(tmp_path)
    selection = _selection(tmp_path)
    reconstructed = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]]], dtype=torch.float32)
    trajectory = torch.zeros((2, 4, 1, 1, 2), dtype=torch.float32)
    trajectory[:, 0] = reconstructed
    validate = Mock(return_value=contract.records_by_index["10"].metadata)
    monkeypatch.setattr(experiment, "_validate_generation_prompt", validate)
    monkeypatch.setattr(experiment, "safe_torch_load", Mock(return_value=trajectory))
    monkeypatch.setattr(
        experiment, "_reconstruct_initial_latents", Mock(return_value=reconstructed)
    )
    experiment._validate_initial_latent_sentinel(_row(selection), contract)
    assert validate.call_args.kwargs["tensor_names"] == ("latent",)

    changed = trajectory.clone()
    changed[0, 0, 0, 0, 0] = torch.nextafter(
        changed[0, 0, 0, 0, 0], torch.tensor(float("inf"))
    )
    monkeypatch.setattr(experiment, "safe_torch_load", Mock(return_value=changed))
    with pytest.raises(experiment.ExperimentError, match="bitwise"):
        experiment._validate_initial_latent_sentinel(_row(selection), contract)


def test_trajectory_loader_uses_every_latent_and_both_prediction_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract(tmp_path)
    prompt = _row(_selection(tmp_path))
    latents = torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 1, 2)
    empty = torch.full((2, 3, 1, 1, 2), 1.0)
    conditional = torch.full_like(empty, 2.0)
    validate = Mock(return_value=contract.records_by_index["10"].metadata)
    monkeypatch.setattr(experiment, "_validate_generation_prompt", validate)
    monkeypatch.setattr(
        experiment,
        "safe_torch_load",
        Mock(side_effect=[latents, (empty, conditional)]),
    )
    loaded_latents, loaded_empty, loaded_conditional = (
        experiment._load_cached_trajectory(prompt, contract)
    )
    assert torch.equal(loaded_latents, latents)
    assert torch.equal(loaded_empty, empty)
    assert torch.equal(loaded_conditional, conditional)
    assert validate.call_args.kwargs["tensor_names"] == (
        "latent",
        "noise_prediction",
    )


def test_same_seed_target_sscd_is_not_averaged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract(tmp_path)
    prompt = _row(_selection(tmp_path))
    marker = dict(contract.records_by_index["10"].metadata)
    score_path = contract.sscd_paths.score_path("10")
    sscd_marker = {
        "original_index": "10",
        "record_id": "record-0",
        "source_row_number": 1,
        "prompt_raw": "prompt 0",
        "target_image_sha256": "1" * 64,
        "model_id": "model",
        "model_revision": "model-revision",
        "vae_id": "vae",
        "vae_revision": "vae-revision",
        "terminal_latent_index": -1,
        "score_shape": [2],
        "score_dtype": "float32",
        "num_seeds": 2,
        "seeds": [0, 1],
        "similarity": SCORE_DEFINITION,
        "sscd_configuration_hash": SSCD_HASH,
        "generation_scientific_config_hash": SCIENTIFIC_HASH,
        "generation_latent_sha256": "d" * 64,
        "score_sha256": "9" * 64,
    }
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(experiment, "read_json", Mock(return_value=sscd_marker))
    monkeypatch.setattr(experiment, "file_sha256", Mock(return_value="9" * 64))
    monkeypatch.setattr(
        experiment,
        "safe_torch_load",
        Mock(return_value=torch.tensor([0.17, 0.83], dtype=torch.float32)),
    )
    scores = experiment._load_target_sscd(
        prompt, contract=contract, generation_marker=marker
    )
    assert scores.tolist() == pytest.approx([0.17, 0.83])
    assert scores[0] != scores[1]
    assert score_path.name == "10.pt"


class _Progress:
    instances: list[_Progress] = []

    def __init__(self, **options: object) -> None:
        self.options = options
        self.count = 0
        self.__class__.instances.append(self)

    def __enter__(self) -> _Progress:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def update(self, amount: int) -> None:
        self.count += amount


def test_cpu_compute_has_one_accurate_progress_and_complete_error_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = _selection(tmp_path)
    prompts = experiment._included_prompt_rows(selection)
    contract = _contract(tmp_path)

    def measure(
        position: int, prompt: dict[str, object], **_kwargs: object
    ) -> experiment._PromptMeasurement:
        if position == 1:
            return experiment._PromptMeasurement(
                position,
                str(prompt["original_index"]),
                {experiment.EVALUATION_TRAJECTORY: None},
                {experiment.EVALUATION_TRAJECTORY: "synthetic failure"},
            )
        return experiment._PromptMeasurement(
            position,
            str(prompt["original_index"]),
            {
                experiment.EVALUATION_TRAJECTORY: _values(
                    position,
                    experiment.EVALUATION_TRAJECTORY,
                )
            },
            {experiment.EVALUATION_TRAJECTORY: ""},
        )

    monkeypatch.setattr(experiment, "_measure_prompt_safely", measure)
    _Progress.instances.clear()
    rows, failed = experiment._compute_rows(
        prompts,
        contract=contract,
        centering=_centering(contract),
        evaluation_source=experiment.EVALUATION_TRAJECTORY,
        devices=(torch.device("cpu"),),
        progress_factory=_Progress,
    )
    assert len(_Progress.instances) == 1
    assert _Progress.instances[0].options["total"] == 12
    assert _Progress.instances[0].count == 12
    assert len(rows) == 12
    assert failed == 1
    assert {row["centering_mode"] for row in rows} == {"mu_hat"}
    assert {row["num_baseline_seeds"] for row in rows} == {NUM_BASELINE_SEEDS}
    assert [row["status"] for row in rows] == ["ok"] * 6 + ["error"] * 6
    assert all(tuple(row) == experiment.CSV_COLUMNS for row in rows)


def test_compact_results_restore_source_order() -> None:
    first = experiment._PromptMeasurement(
        1,
        "11",
        {experiment.EVALUATION_GAUSSIAN: _values(1)},
        {experiment.EVALUATION_GAUSSIAN: ""},
    )
    second = experiment._PromptMeasurement(
        0,
        "10",
        {experiment.EVALUATION_GAUSSIAN: _values(0)},
        {experiment.EVALUATION_GAUSSIAN: ""},
    )
    assert experiment._canonical_measurements((first, second), 2) == (second, first)
    with pytest.raises(experiment.ExperimentError, match="duplicate"):
        experiment._canonical_measurements((second, second), 2)


def test_multi_cuda_uses_whole_prompt_device_shards_and_parent_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = _selection(tmp_path, prompt_count=3)
    prompts = experiment._included_prompt_rows(selection)
    contract = _contract(tmp_path, prompt_count=3)
    calls: list[tuple[int, str]] = []

    class Executor:
        def __init__(self, **options: object) -> None:
            self.options = options

        def __enter__(self) -> Executor:
            initializer = self.options["initializer"]
            initializer(*self.options["initargs"])
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self, function: object, position: int, prompt: dict[str, object]
        ) -> Future:
            future: Future = Future()
            calls.append((position, str(experiment._WORKER_DEVICE)))
            future.set_result(function(position, prompt))
            return future

    monkeypatch.setattr(experiment, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        experiment.multiprocessing, "get_context", Mock(return_value=None)
    )
    monkeypatch.setattr(experiment.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(experiment, "configure_worker_cpu_threads", Mock())
    monkeypatch.setattr(
        experiment,
        "_reconstruct_initial_latents",
        Mock(return_value=torch.zeros((2, 1, 1, 2))),
    )

    def measure(
        position: int, prompt: dict[str, object], **_kwargs: object
    ) -> experiment._PromptMeasurement:
        return experiment._PromptMeasurement(
            position,
            str(prompt["original_index"]),
            {
                experiment.EVALUATION_TRAJECTORY: _values(
                    position,
                    experiment.EVALUATION_TRAJECTORY,
                )
            },
            {experiment.EVALUATION_TRAJECTORY: ""},
        )

    monkeypatch.setattr(experiment, "_measure_prompt_safely", measure)
    _Progress.instances.clear()
    rows, failed = experiment._compute_rows(
        prompts,
        contract=contract,
        centering=_centering(contract),
        evaluation_source=experiment.EVALUATION_TRAJECTORY,
        devices=(torch.device("cuda:0"), torch.device("cuda:1")),
        progress_factory=_Progress,
    )
    assert calls == [(0, "cuda:0"), (2, "cuda:0"), (1, "cuda:1")]
    assert [row["record_id"] for row in rows] == (
        ["record-0"] * 6 + ["record-1"] * 6 + ["record-2"] * 6
    )
    assert failed == 0
    assert _Progress.instances[0].options["total"] == 18
    assert _Progress.instances[0].count == 18


def test_csv_validation_requires_exact_canonical_prompt_seed_grid_and_provenance(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    contract = _contract(tmp_path)
    frame = pd.DataFrame(_csv_rows(selection, contract), columns=experiment.CSV_COLUMNS)
    validated = experiment._validated_plot_frame(
        frame,
        selection=selection,
        contract=contract,
        centering=_centering(contract, load_tensor=False),
    )
    assert len(validated) == 24
    for mutate in (
        "evaluation_source",
        "generation_seed",
        "step_index",
        "timestep",
        "guidance_scale",
    ):
        broken = frame.copy()
        broken.loc[0, mutate] = "wrong" if mutate == "evaluation_source" else 99
        with pytest.raises(experiment.ExperimentError):
            experiment._validated_plot_frame(
                broken,
                selection=selection,
                contract=contract,
                centering=_centering(contract, load_tensor=False),
            )
    missing = frame.iloc[:-1]
    with pytest.raises(experiment.ExperimentError, match="source-prompt-seed-timestep"):
        experiment._validated_plot_frame(
            missing,
            selection=selection,
            contract=contract,
            centering=_centering(contract, load_tensor=False),
        )

    wrong_baseline_count = frame.copy()
    wrong_baseline_count.loc[0, "num_baseline_seeds"] += 1
    with pytest.raises(experiment.ExperimentError, match="num_baseline_seeds"):
        experiment._validated_plot_frame(
            wrong_baseline_count,
            selection=selection,
            contract=contract,
            centering=_centering(contract, load_tensor=False),
        )

    wrong_baseline_hash = frame.copy()
    wrong_baseline_hash.loc[0, "baseline_mu_hat_sha256"] = "9" * 64
    with pytest.raises(experiment.ExperimentError, match="baseline_mu_hat_sha256"):
        experiment._validated_plot_frame(
            wrong_baseline_hash,
            selection=selection,
            contract=contract,
            centering=_centering(contract, load_tensor=False),
        )

    zero_frame = pd.DataFrame(
        _csv_rows(selection, contract, use_mu=False), columns=experiment.CSV_COLUMNS
    )
    zero_centering = _centering(contract, use_mu=False, load_tensor=False)
    zero_validated = experiment._validated_plot_frame(
        zero_frame,
        selection=selection,
        contract=contract,
        centering=zero_centering,
    )
    assert set(zero_validated["centering_mode"]) == {experiment.CENTERING_ZERO}
    assert set(zero_validated["num_baseline_seeds"]) == {0}
    assert set(zero_validated["baseline_mu_hat_sha256"]) == {""}
    assert set(zero_validated["baseline_schedule_sha256"]) == {""}
    assert set(zero_validated["baseline_generation_scientific_config_hash"]) == {""}
    with pytest.raises(experiment.ExperimentError, match="centering_mode"):
        experiment._validated_plot_frame(
            zero_frame,
            selection=selection,
            contract=contract,
            centering=_centering(contract, load_tensor=False),
        )


@pytest.mark.parametrize("scheduler_name", ("ddim", "ddpm"))
@pytest.mark.parametrize(
    ("metric", "ylabel_fragment", "expected_line_count"),
    (
        (experiment.COEFFICIENT_METRIC, r"\widehat{g}_t", 2),
        (experiment.RESIDUAL_METRIC, r"\mathbf{r}_t", 1),
    ),
)
def test_figure_uses_log_snr_bands_viridis_and_opaque_colorbar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scheduler_name: str,
    metric: str,
    ylabel_fragment: str,
    expected_line_count: int,
) -> None:
    selection = _selection(tmp_path, scheduler_name=scheduler_name)
    contract = _contract(tmp_path, scheduler_name=scheduler_name)
    frame = pd.DataFrame(
        _csv_rows(
            selection,
            contract,
            evaluation_source=experiment.EVALUATION_GAUSSIAN,
        ),
        columns=experiment.CSV_COLUMNS,
    )
    # Distinguish all three quantities so plotting the direct error cannot pass.
    frame["fitted_guidance_scale"] = 2.0 + np.arange(len(frame), dtype=float)
    frame["residual_rmse"] = 0.25 + 0.125 * np.arange(len(frame), dtype=float)
    frame["guided_target_rmse"] = 1000.0
    captured: dict[str, object] = {}

    def save(figure: object, destinations: object) -> None:
        captured["figure"] = figure
        captured["destinations"] = destinations
        assert tuple(figure.get_size_inches()) == pytest.approx((4.0, 4.0))
        assert len(figure.axes) == 2
        axis, colorbar_axis = figure.axes
        assert axis.get_xlabel() == r"$\mathrm{SNR}_t$"
        assert axis.get_xscale() == "log"
        assert axis.get_title() == ""
        assert ylabel_fragment in axis.get_ylabel()
        assert r"\|_2" not in axis.get_ylabel()
        assert r"\boldsymbol{\mu}" in axis.get_ylabel()
        assert axis.xaxis.label.get_fontsize() == 15
        assert axis.yaxis.label.get_fontsize() == 15
        assert isinstance(axis.xaxis.get_major_locator(), experiment.LogLocator)
        assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
        assert all(label.get_fontsize() == 12 for label in axis.get_yticklabels())

        expected = frame.groupby("snr_t", sort=True)[metric]
        assert len(axis.lines) == expected_line_count
        median = axis.lines[0]
        np.testing.assert_allclose(median.get_xdata(), expected.median().index)
        np.testing.assert_allclose(median.get_ydata(), expected.median())
        assert median.get_marker() in (None, "None", "")

        observations = [
            item for item in axis.collections if isinstance(item, LineCollection)
        ]
        assert len(observations) == 1
        observation = observations[0]
        assert not any(isinstance(item, PathCollection) for item in axis.collections)
        assert observation.get_alpha() == pytest.approx(
            experiment.OBSERVATION_LINE_ALPHA
        )
        np.testing.assert_allclose(
            observation.get_linewidths(), experiment.OBSERVATION_LINE_WIDTH
        )
        assert observation.norm.vmin == 0.0
        assert observation.norm.vmax == 1.0
        assert observation.norm.clip is True
        assert observation.get_cmap().name == "viridis"
        expected_segments = []
        expected_colors = []
        for (_record_id, _seed), group in frame.groupby(
            ["record_id", "generation_seed"], sort=False
        ):
            ordered = group.sort_values("snr_t", kind="stable")
            expected_segments.append(ordered[["snr_t", metric]].to_numpy(dtype=float))
            expected_colors.append(float(ordered["target_sscd"].iloc[0]))
        assert len(observation.get_segments()) == 4
        for actual, expected_segment in zip(
            observation.get_segments(), expected_segments, strict=True
        ):
            np.testing.assert_allclose(actual, expected_segment)
            assert np.all(np.diff(actual[:, 0]) > 0.0)
        np.testing.assert_allclose(observation.get_array(), expected_colors)

        bands = [
            item for item in axis.collections if not isinstance(item, LineCollection)
        ]
        assert len(bands) == 3
        for band, (lower_q, upper_q, alpha) in zip(
            bands, experiment.PERCENTILE_BANDS, strict=True
        ):
            assert band.get_alpha() == pytest.approx(alpha)
            vertices = band.get_paths()[0].vertices
            for quantile in (lower_q, upper_q):
                for x, y in expected.quantile(quantile).items():
                    assert np.any(np.all(np.isclose(vertices, (x, y)), axis=1))

        if metric == experiment.COEFFICIENT_METRIC:
            np.testing.assert_allclose(
                axis.lines[1].get_ydata(), experiment.REQUIRED_GUIDANCE_SCALE
            )
            assert axis.lines[1].get_linestyle() == "--"
            assert axis.get_legend().get_texts()[0].get_text() == "$g=7.5$"
            assert axis.get_legend().get_texts()[0].get_fontsize() == 10
        else:
            assert axis.get_legend() is None
        assert colorbar_axis.get_ylabel() == "SSCD"
        assert colorbar_axis.get_ylim() == pytest.approx((0.0, 1.0))
        assert colorbar_axis.collections[-1].get_alpha() == 1.0

    monkeypatch.setattr(experiment, "_atomic_save_figures", save)
    experiment._render_figure(
        frame,
        (tmp_path / "a.png", tmp_path / "a.pdf"),
        centering_mode=experiment.CENTERING_MU_HAT,
        evaluation_source=experiment.EVALUATION_GAUSSIAN,
        metric=metric,
    )
    assert captured["destinations"] == (tmp_path / "a.png", tmp_path / "a.pdf")
    plt.close(captured["figure"])


@pytest.mark.parametrize(
    ("metric", "ylabel"),
    (
        (experiment.COEFFICIENT_METRIC, r"$\widehat{g}_t$"),
        (experiment.RESIDUAL_METRIC, r"$\|\mathbf{r}_t\|/\sqrt{d}$"),
    ),
)
def test_zero_centered_figure_ylabel_omits_mu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metric: str, ylabel: str
) -> None:
    selection = _selection(tmp_path)
    contract = _contract(tmp_path)
    frame = pd.DataFrame(
        _csv_rows(
            selection,
            contract,
            use_mu=False,
            evaluation_source=experiment.EVALUATION_TRAJECTORY,
        ),
        columns=experiment.CSV_COLUMNS,
    )
    captured: dict[str, object] = {}

    def save(figure: object, _destinations: object) -> None:
        captured["figure"] = figure
        assert len(figure.axes) == 2
        axis, _colorbar_axis = figure.axes
        assert axis.get_ylabel() == ylabel
        assert all(r"\boldsymbol{\mu}" not in axis.get_ylabel() for axis in figure.axes)

    monkeypatch.setattr(experiment, "_atomic_save_figures", save)
    experiment._render_figure(
        frame,
        (tmp_path / "a.png", tmp_path / "a.pdf"),
        centering_mode=experiment.CENTERING_ZERO,
        evaluation_source=experiment.EVALUATION_TRAJECTORY,
        metric=metric,
    )
    plt.close(captured["figure"])


def test_default_output_is_baseline_count_and_selection_hash_scoped() -> None:
    result = experiment._default_output_directory(
        "sdv1",
        "ddim",
        7.5,
        50,
        20,
        NUM_BASELINE_SEEDS,
        experiment.CENTERING_MU_HAT,
        "gmm",
        SELECTION_HASH,
    )
    assert result.parts[-5:] == (
        "corollary3_cfg_amplification",
        "centering_mu_hat",
        "baseline_S20_N3",
        SELECTION_HASH,
        "evaluation_both",
    )
    assert result.name == "evaluation_both"
    assert result != experiment._default_output_directory(
        "sdv1",
        "ddim",
        7.5,
        50,
        20,
        4,
        experiment.CENTERING_MU_HAT,
        "gmm",
        SELECTION_HASH,
    )
    zero = experiment._default_output_directory(
        "sdv1",
        "ddim",
        7.5,
        50,
        20,
        NUM_BASELINE_SEEDS,
        experiment.CENTERING_ZERO,
        "gmm",
        SELECTION_HASH,
    )
    assert zero.parts[-4:] == (
        "corollary3_cfg_amplification",
        "centering_zero",
        SELECTION_HASH,
        "evaluation_both",
    )
    assert zero == experiment._default_output_directory(
        "sdv1",
        "ddim",
        7.5,
        50,
        20,
        999,
        experiment.CENTERING_ZERO,
        "gmm",
        SELECTION_HASH,
    )
    assert zero != result


def test_plot_only_does_not_resolve_device_or_read_scientific_tensors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = _selection(tmp_path)
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        experiment, "_load_frozen_selection", Mock(return_value=selection)
    )
    monkeypatch.setattr(
        experiment, "_load_generation_contract", Mock(return_value=contract)
    )
    monkeypatch.setattr(
        experiment, "_requested_output_directory", Mock(return_value=tmp_path)
    )
    monkeypatch.setattr(
        experiment, "_prepare_output_directory", lambda value: Path(value)
    )
    baseline_loader = Mock(
        return_value=_baseline(tmp_path, contract, load_tensor=False)
    )
    monkeypatch.setattr(experiment, "_load_shared_baseline", baseline_loader)
    monkeypatch.setattr(experiment, "_remove_figure_outputs", Mock())
    plot = Mock()
    monkeypatch.setattr(experiment, "plot_saved_results", plot)
    monkeypatch.setattr(
        experiment,
        "resolve_devices",
        Mock(side_effect=AssertionError("plot-only resolved a device")),
    )
    monkeypatch.setattr(
        experiment,
        "_validate_initial_latent_sentinel",
        Mock(side_effect=AssertionError("plot-only read a trajectory")),
    )
    assert (
        experiment.main(
            [
                "--plot",
                "--use-mu",
                "--num-baseline-seeds",
                str(NUM_BASELINE_SEEDS),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert baseline_loader.call_args.kwargs["load_tensor"] is False
    assert baseline_loader.call_args.kwargs["num_baseline_seeds"] == NUM_BASELINE_SEEDS
    assert plot.call_args.kwargs["centering"].mode == experiment.CENTERING_MU_HAT
    assert plot.call_args.kwargs["centering"].value is None
    plot.assert_called_once()


def test_zero_plot_only_never_reads_baseline_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = _selection(tmp_path)
    contract = _contract(tmp_path)
    baseline_loader = Mock(side_effect=AssertionError("zero plot read baseline"))
    monkeypatch.setattr(experiment, "_load_shared_baseline", baseline_loader)
    monkeypatch.setattr(
        experiment, "_load_frozen_selection", Mock(return_value=selection)
    )
    monkeypatch.setattr(
        experiment, "_load_generation_contract", Mock(return_value=contract)
    )
    monkeypatch.setattr(
        experiment, "_requested_output_directory", Mock(return_value=tmp_path)
    )
    monkeypatch.setattr(
        experiment, "_prepare_output_directory", lambda value: Path(value)
    )
    monkeypatch.setattr(experiment, "_remove_figure_outputs", Mock())
    plot = Mock()
    monkeypatch.setattr(experiment, "plot_saved_results", plot)

    assert experiment.main(["--plot", "--output-dir", str(tmp_path)]) == 0
    baseline_loader.assert_not_called()
    centering = plot.call_args.kwargs["centering"]
    assert centering.mode == experiment.CENTERING_ZERO
    assert centering.value is None
    assert centering.num_baseline_seeds == 0
    plot.assert_called_once()


def test_normal_run_never_reuses_existing_compact_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = _selection(tmp_path)
    contract = _contract(tmp_path)
    baseline_loader = Mock(return_value=_baseline(tmp_path, contract))
    monkeypatch.setattr(experiment, "_load_shared_baseline", baseline_loader)
    rows = _csv_rows(selection, contract)
    monkeypatch.setattr(
        experiment, "_load_frozen_selection", Mock(return_value=selection)
    )
    monkeypatch.setattr(
        experiment, "_load_generation_contract", Mock(return_value=contract)
    )
    monkeypatch.setattr(
        experiment, "_requested_output_directory", Mock(return_value=tmp_path)
    )
    monkeypatch.setattr(
        experiment, "_prepare_output_directory", lambda value: Path(value)
    )
    monkeypatch.setattr(
        experiment, "resolve_devices", Mock(return_value=(torch.device("cpu"),))
    )
    sentinel = Mock()
    compute = Mock(return_value=(rows, 0))
    write = Mock()
    plot = Mock()
    monkeypatch.setattr(experiment, "_validate_initial_latent_sentinel", sentinel)
    monkeypatch.setattr(experiment, "_compute_rows", compute)
    monkeypatch.setattr(experiment, "atomic_write_csv", write)
    monkeypatch.setattr(experiment, "plot_saved_results", plot)
    (tmp_path / experiment.CSV_NAME).write_text("preexisting", encoding="utf-8")
    assert (
        experiment.run_experiment(
            model_name="sdv1",
            selection_strategy="gmm",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=3,
            num_seeds=2,
            num_baseline_seeds=NUM_BASELINE_SEEDS,
            use_mu=True,
            output_dir=tmp_path,
            device="cpu",
        )
        == 0
    )
    sentinel.assert_called_once()
    assert baseline_loader.call_args.kwargs["load_tensor"] is True
    assert baseline_loader.call_args.kwargs["num_baseline_seeds"] == NUM_BASELINE_SEEDS
    assert compute.call_args.kwargs["centering"].mode == experiment.CENTERING_MU_HAT
    assert (
        compute.call_args.kwargs["centering"].value
        is baseline_loader.return_value.mu_hat
    )
    compute.assert_called_once()
    write.assert_called_once()
    plot.assert_called_once()


def test_zero_normal_run_never_reads_baseline_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection = _selection(tmp_path)
    contract = _contract(tmp_path)
    baseline_loader = Mock(side_effect=AssertionError("zero compute read baseline"))
    monkeypatch.setattr(experiment, "_load_shared_baseline", baseline_loader)
    monkeypatch.setattr(
        experiment, "_load_frozen_selection", Mock(return_value=selection)
    )
    monkeypatch.setattr(
        experiment, "_load_generation_contract", Mock(return_value=contract)
    )
    monkeypatch.setattr(
        experiment, "_requested_output_directory", Mock(return_value=tmp_path)
    )
    monkeypatch.setattr(
        experiment, "_prepare_output_directory", lambda value: Path(value)
    )
    monkeypatch.setattr(
        experiment, "resolve_devices", Mock(return_value=(torch.device("cpu"),))
    )
    monkeypatch.setattr(experiment, "_validate_initial_latent_sentinel", Mock())
    compute = Mock(return_value=(_csv_rows(selection, contract, use_mu=False), 0))
    monkeypatch.setattr(experiment, "_compute_rows", compute)
    monkeypatch.setattr(experiment, "atomic_write_csv", Mock())
    monkeypatch.setattr(experiment, "plot_saved_results", Mock())

    assert (
        experiment.run_experiment(
            model_name="sdv1",
            selection_strategy="gmm",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=3,
            num_seeds=2,
            num_baseline_seeds=NUM_BASELINE_SEEDS,
            output_dir=tmp_path,
            device="cpu",
        )
        == 0
    )
    baseline_loader.assert_not_called()
    centering = compute.call_args.kwargs["centering"]
    assert centering.mode == experiment.CENTERING_ZERO
    assert centering.value is not None
    assert centering.value.dtype == torch.float64
    assert torch.equal(centering.value, torch.zeros_like(centering.value))


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
