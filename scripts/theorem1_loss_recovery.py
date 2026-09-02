#!/usr/bin/env python3
"""Measure pair-specific conditional loss and one-prediction latent recovery."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common.cli import (  # noqa: E402
    MAX_SEED,
    generation_run_name,
    nonnegative_integer,
    positive_integer,
)
from utils.common.io import (  # noqa: E402
    atomic_write_csv,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.data.webster import WebsterDataset  # noqa: E402
from utils.experiments.cache import (  # noqa: E402
    GenerationPaths,
    generation_paths,
    safe_index,
    require_generation_run,
    validate_generation_record,
)
from utils.experiments.sscd import (  # noqa: E402
    SCORE_DEFINITION,
    SSCDPaths,
    sscd_configuration_hash,
)
from utils.metrics.sscd import validate_sscd_scores  # noqa: E402
from utils.models.latent import (  # noqa: E402
    TARGET_LATENT_DEFINITION,
    encode_target_latent,
    target_preprocessing_policy,
)
from utils.models.loading import (  # noqa: E402
    load_model_components,
    preflight_model_components,
)
from utils.models.prediction_conversion import (  # noqa: E402
    alpha_sigma_for_timestep,
)
from utils.models.registry import get_model_spec, model_names  # noqa: E402
from utils.models.sampling import (  # noqa: E402
    encode_prompt_condition,
    make_initial_noise,
    predict_conditional_epsilon,
    validate_latent_shape,
)
from utils.models.schedulers import build_scheduler  # noqa: E402


CSV_NAME = "theorem1_loss_recovery.csv"
FIGURE_NAME = "theorem1_loss_recovery.pdf"
CSV_COLUMNS = (
    "record_id",
    "model_name",
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

SCHEDULER_NAME = "ddim"
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
CANONICAL_GENERATION_SEEDS = tuple(range(20))
DEFAULT_SAMPLE_BATCH_SIZE = 8
PROGRESS_DESCRIPTION = "Theorem 1 loss–recovery"
COLORBAR_LABEL = "SSCD"
SSCD_COLOR_RANGE = (0.0, 1.0)
FIGURE_SIZE = (4.0, 4.0)
TEXT_FONT_SIZE = 15
AXIS_NUMBER_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
SCATTER_ALPHA = 0.68
COLORBAR_ALPHA = 1.0
FIGURE_PAD_INCHES = 0.05
EXPERIMENT_DIRECTORY = "theorem1_loss_recovery"
_LOSS_STREAM_DOMAIN = b"theorem1-pair-loss-noise-v1"

PLOT_STYLE = {
    "figure.figsize": FIGURE_SIZE,
    "font.family": "STIXGeneral",
    "font.size": TEXT_FONT_SIZE,
    "mathtext.fontset": "stix",
    "axes.labelsize": TEXT_FONT_SIZE,
    "axes.titlesize": TEXT_FONT_SIZE,
    "figure.titlesize": TEXT_FONT_SIZE,
    "xtick.labelsize": AXIS_NUMBER_FONT_SIZE,
    "ytick.labelsize": AXIS_NUMBER_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "legend.title_fontsize": LEGEND_FONT_SIZE,
}


class ExperimentError(RuntimeError):
    """Report an incompatible cache or invalid experiment measurement."""


@dataclass(frozen=True, slots=True)
class TerminalParameters:
    timestep: int
    alpha_t: float
    sigma_t: float
    snr_t: float


@dataclass(frozen=True, slots=True)
class LossMeasurement:
    conditional_loss: float
    normalized_loss_mse: float
    normalized_loss_rmse: float


@dataclass(frozen=True, slots=True)
class RecoveryMeasurement:
    recovery_mse: float
    recovery_rmse: float


@dataclass(frozen=True, slots=True)
class GenerationContract:
    paths: GenerationPaths
    sscd_paths: SSCDPaths
    configuration: Mapping[str, Any]
    science: Mapping[str, Any]
    scientific_hash: str
    latent_shape: tuple[int, int, int]
    stored_dtype: torch.dtype
    latent_dtype: torch.dtype
    init_noise_sigma: float
    seeds: tuple[int, ...]
    generation_seeds: tuple[int, ...]
    schedule_payload: Mapping[str, Any]
    scheduler_config: Mapping[str, Any]
    sscd_configuration: Mapping[str, Any] | None
    sscd_configuration_hash: str | None
    sscd_error: str | None


def _generation_seed_list(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "generation seeds must be a comma-separated list of integers"
        )
    seeds: list[int] = []
    for part in parts:
        try:
            seed = int(part.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "generation seeds must be a comma-separated list of integers"
            ) from error
        if seed < 0 or seed > MAX_SEED:
            raise argparse.ArgumentTypeError(
                f"generation seeds must be between 0 and {MAX_SEED}"
            )
        if seed in seeds:
            raise argparse.ArgumentTypeError("generation seeds must be unique")
        seeds.append(seed)
    return tuple(seeds)


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately small public command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pair-specific terminal conditional loss against initial "
            "conditional clean-latent recovery."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model", choices=model_names(), default="sdv1")
    parser.add_argument(
        "--num-loss-seeds",
        type=positive_integer,
        default=20,
        metavar="K",
    )
    parser.add_argument(
        "--loss-seed",
        type=nonnegative_integer,
        default=0,
        metavar="SEED",
    )
    parser.add_argument(
        "--generation-seeds",
        type=_generation_seed_list,
        default=CANONICAL_GENERATION_SEEDS,
        metavar="S0,S1,...",
    )
    parser.add_argument(
        "--max-generation-seeds",
        type=positive_integer,
        default=None,
        metavar="S",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=positive_integer,
        default=DEFAULT_SAMPLE_BATCH_SIZE,
        metavar="BATCH",
    )
    parser.add_argument(
        "--max-records",
        type=positive_integer,
        default=None,
        metavar="PAIRS",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "explicit final output directory; by default, results are placed "
            "inside the selected model's existing output experiment folder"
        ),
    )
    parser.add_argument(
        "--plot",
        "--plot-only",
        dest="plot_only",
        action="store_true",
        help="regenerate the PDF from the saved CSV without computation",
    )
    return parser


def _dtype_from_name(value: object) -> torch.dtype:
    names = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return names[str(value)]
    except KeyError as error:
        raise ExperimentError(f"unsupported cached latent dtype: {value!r}") from error


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ExperimentError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ExperimentError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ExperimentError(f"{label} must be finite")
    return result


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExperimentError(f"{label} must be an integer sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ExperimentError(f"{label} must be an integer sequence")
        try:
            selected = int(item)
        except (TypeError, ValueError) as error:
            raise ExperimentError(f"{label} must be an integer sequence") from error
        if selected != item:
            raise ExperimentError(f"{label} must be an integer sequence")
        result.append(selected)
    return tuple(result)


def _normalize_scheduler_config(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentError("scheduler configuration must be a mapping")
    result = {str(key): item for key, item in value.items()}
    defaults = result.get("_use_default_values")
    if isinstance(defaults, Sequence) and not isinstance(defaults, (str, bytes)):
        result["_use_default_values"] = sorted(str(item) for item in defaults)
    return result


def _load_sscd_configuration(
    paths: SSCDPaths,
    *,
    science: Mapping[str, Any],
    scientific_hash: str,
    seeds: tuple[int, ...],
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    try:
        if not paths.config_json.is_file() or paths.config_json.is_symlink():
            raise ExperimentError("target-specific SSCD configuration is missing")
        configuration = read_json(paths.config_json)
        stored_hash = configuration.get("configuration_hash")
        if (
            not isinstance(stored_hash, str)
            or stored_hash != sscd_configuration_hash(configuration)
        ):
            raise ExperimentError("target-specific SSCD configuration hash differs")
        expected = {
            "generation_scientific_config_hash": scientific_hash,
            "num_seeds": len(seeds),
            "seeds": list(seeds),
            "model_id": science.get("model_id"),
            "model_revision": science.get("model_revision"),
            "vae_id": science.get("vae_id"),
            "vae_revision": science.get("vae_revision"),
            "score_definition": SCORE_DEFINITION,
        }
        wrong = [
            key
            for key, expected_value in expected.items()
            if configuration.get(key) != expected_value
        ]
        if wrong:
            raise ExperimentError(
                "target-specific SSCD configuration differs at: " + ", ".join(wrong)
            )
        return configuration, stored_hash, None
    except Exception as error:
        return None, None, str(error)


def _load_generation_contract(
    root: str | Path,
    model_name: str,
) -> GenerationContract:
    """Validate the canonical 20-seed generation and SSCD cache."""

    project = Path(root).expanduser().resolve()
    paths = generation_paths(
        project,
        model_name=model_name,
        scheduler_name=SCHEDULER_NAME,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_seeds=len(CANONICAL_GENERATION_SEEDS),
        seed_start=0,
    )
    configuration = require_generation_run(paths)
    science = configuration.get("scientific_config")
    if not isinstance(science, Mapping):
        raise ExperimentError("generation cache has no scientific configuration")
    scientific_hash = configuration.get("scientific_config_hash")
    if not isinstance(scientific_hash, str):
        raise ExperimentError("generation cache has no scientific hash")

    spec = get_model_spec(model_name)
    expected = {
        "model_cli_name": model_name,
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "num_seeds": len(CANONICAL_GENERATION_SEEDS),
        "seeds": list(CANONICAL_GENERATION_SEEDS),
        "trajectory_order": "noise_to_image",
        "stored_prediction_type": "epsilon",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
        "target_preprocessing": target_preprocessing_policy(spec.resolution),
    }
    scheduler_metadata = science.get("scheduler")
    if not isinstance(scheduler_metadata, Mapping):
        raise ExperimentError("generation cache scheduler metadata is invalid")
    wrong = [
        key
        for key, expected_value in expected.items()
        if science.get(key) != expected_value
    ]
    if scheduler_metadata.get("name") != SCHEDULER_NAME:
        wrong.append("scheduler")
    if wrong:
        raise ExperimentError(
            "canonical generation cache differs at: " + ", ".join(sorted(set(wrong)))
        )

    latent_shape = _integer_tuple(science.get("latent_shape"), "latent shape")
    if len(latent_shape) != 3 or any(size <= 0 for size in latent_shape):
        raise ExperimentError("latent shape must contain three positive dimensions")
    seeds = _integer_tuple(science.get("seeds"), "generation seeds")
    if seeds != CANONICAL_GENERATION_SEEDS:
        raise ExperimentError(
            "canonical generation cache must contain seeds 0 through 19"
        )
    storage = science.get("scientific_tensor_storage")
    if not isinstance(storage, Mapping):
        raise ExperimentError("generation tensor storage metadata is invalid")
    stored_dtype = _dtype_from_name(storage.get("dtype"))
    if storage.get("device") != "cpu" or storage.get("layout") != "contiguous":
        raise ExperimentError("generation tensors must be contiguous CPU tensors")

    if not paths.schedule.is_file() or paths.schedule.is_symlink():
        raise ExperimentError("saved generation schedule is missing")
    schedule_payload = safe_torch_load(paths.schedule)
    if not isinstance(schedule_payload, Mapping):
        raise ExperimentError("saved generation schedule must be a mapping")
    saved_timesteps = schedule_payload.get("timesteps")
    if (
        not isinstance(saved_timesteps, torch.Tensor)
        or saved_timesteps.ndim != 1
        or saved_timesteps.numel() != NUM_INFERENCE_STEPS
    ):
        raise ExperimentError("saved generation timesteps are invalid")
    if _integer_tuple(saved_timesteps.tolist(), "saved timesteps")[0] < 0:
        raise ExperimentError("saved generation timestep is invalid")
    init_noise_sigma = _number(
        schedule_payload.get("init_noise_sigma"),
        "saved scheduler initial-noise scale",
    )
    if not math.isclose(init_noise_sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ExperimentError(
            "canonical initial-noise reconstruction requires init_noise_sigma = 1"
        )
    scheduler_config = _normalize_scheduler_config(
        schedule_payload.get("scheduler_config")
    )
    science_scheduler_config = _normalize_scheduler_config(
        scheduler_metadata.get("config")
    )
    if canonical_hash(scheduler_config) != canonical_hash(science_scheduler_config):
        raise ExperimentError("saved and configured scheduler settings differ")

    sscd_paths = SSCDPaths(paths.run_directory)
    sscd_configuration, sscd_hash, sscd_error = _load_sscd_configuration(
        sscd_paths,
        science=science,
        scientific_hash=scientific_hash,
        seeds=seeds,
    )
    return GenerationContract(
        paths=paths,
        sscd_paths=sscd_paths,
        configuration=configuration,
        science=science,
        scientific_hash=scientific_hash,
        latent_shape=(latent_shape[0], latent_shape[1], latent_shape[2]),
        stored_dtype=stored_dtype,
        latent_dtype=stored_dtype,
        init_noise_sigma=init_noise_sigma,
        seeds=seeds,
        generation_seeds=seeds,
        schedule_payload=schedule_payload,
        scheduler_config=scheduler_config,
        sscd_configuration=sscd_configuration,
        sscd_configuration_hash=sscd_hash,
        sscd_error=sscd_error,
    )


def _revision_resolver(contract: GenerationContract) -> Callable[[str], str]:
    science = contract.science
    revisions = {
        str(science["model_id"]): str(science["model_revision"]),
        str(science["vae_id"]): str(science["vae_revision"]),
    }

    def resolve(repository_id: str) -> str:
        try:
            return revisions[repository_id]
        except KeyError as error:
            raise ExperimentError(
                f"generation cache has no pinned revision for {repository_id}"
            ) from error

    return resolve


def _validate_loaded_components(
    components: Any,
    contract: GenerationContract,
) -> None:
    science = contract.science
    expected = {
        "model_id": science.get("model_id"),
        "model_revision": science.get("model_revision"),
        "vae_id": science.get("vae_id"),
        "vae_revision": science.get("vae_revision"),
    }
    observed = {
        "model_id": getattr(components, "model_id", None),
        "model_revision": getattr(components, "model_revision", None),
        "vae_id": getattr(components, "vae_id", None),
        "vae_revision": getattr(components, "vae_revision", None),
    }
    wrong = [key for key, value in expected.items() if observed.get(key) != value]
    if wrong:
        raise ExperimentError(
            "loaded components differ from generation cache at: " + ", ".join(wrong)
        )
    actual_shape = validate_latent_shape(
        components.unet,
        expected_shape=contract.latent_shape,
    )
    if actual_shape != contract.latent_shape:
        raise ExperimentError("loaded UNet latent shape differs from generation cache")


def _set_scheduler_timesteps(scheduler: Any, device: torch.device) -> None:
    setter = getattr(scheduler, "set_timesteps", None)
    if not callable(setter):
        raise ExperimentError("DDIM scheduler has no timestep interface")
    try:
        setter(NUM_INFERENCE_STEPS, device=device)
    except TypeError:
        setter(NUM_INFERENCE_STEPS)


def _validate_active_scheduler(
    components: Any,
    contract: GenerationContract,
) -> Any:
    """Rebuild DDIM from the loaded checkpoint and match the cached schedule."""

    result = build_scheduler(components.original_scheduler, SCHEDULER_NAME)
    scheduler = result.scheduler
    _set_scheduler_timesteps(scheduler, torch.device(components.device))
    active_timesteps = torch.as_tensor(scheduler.timesteps).detach().cpu().long()
    saved_timesteps = (
        torch.as_tensor(contract.schedule_payload["timesteps"]).detach().cpu().long()
    )
    if not torch.equal(active_timesteps, saved_timesteps):
        raise ExperimentError("active DDIM timesteps differ from generation cache")
    active_config = _normalize_scheduler_config(result.config)
    if canonical_hash(active_config) != canonical_hash(contract.scheduler_config):
        raise ExperimentError("active DDIM configuration differs from generation cache")
    active_sigma = _number(
        getattr(scheduler, "init_noise_sigma", None),
        "active scheduler initial-noise scale",
    )
    if not math.isclose(
        active_sigma,
        contract.init_noise_sigma,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ExperimentError(
            "active scheduler initial-noise scale differs from generation cache"
        )
    return scheduler


def _terminal_parameters(scheduler: Any) -> TerminalParameters:
    timesteps = torch.as_tensor(getattr(scheduler, "timesteps", None))
    if timesteps.ndim != 1 or timesteps.numel() < 1:
        raise ExperimentError("active DDIM scheduler has no inference timesteps")
    timestep = int(timesteps[0].detach().cpu().item())
    alpha, sigma = alpha_sigma_for_timestep(
        scheduler,
        timestep,
        device="cpu",
        dtype=torch.float64,
    )
    alpha_t = float(alpha.item())
    sigma_t = float(sigma.item())
    if alpha_t <= 0.0 or sigma_t <= 0.0:
        raise ExperimentError("terminal diffusion coefficients must be positive")
    snr_t = alpha_t * alpha_t / (sigma_t * sigma_t)
    if not math.isfinite(snr_t) or snr_t <= 0.0:
        raise ExperimentError("terminal SNR must be finite and positive")
    return TerminalParameters(timestep, alpha_t, sigma_t, snr_t)


def _selected_generation_seeds(
    requested: Sequence[int],
    maximum: int | None,
    available: Sequence[int],
) -> tuple[int, ...]:
    selected = tuple(int(seed) for seed in requested)
    if not selected or len(set(selected)) != len(selected):
        raise ExperimentError("generation seeds must be non-empty and unique")
    available_values = tuple(int(seed) for seed in available)
    missing = [seed for seed in selected if seed not in available_values]
    if missing:
        raise ExperimentError(
            "requested generation seeds are absent from the canonical cache: "
            + ",".join(str(seed) for seed in missing)
        )
    if maximum is not None:
        selected = selected[:maximum]
    if not selected:
        raise ExperimentError("at least one generation seed is required")
    return selected


def _loss_sample_seeds(
    loss_seed: int,
    count: int,
    forbidden_seeds: Sequence[int] = (),
) -> tuple[int, ...]:
    """Derive a deterministic, domain-separated seed for each loss draw."""

    if isinstance(loss_seed, bool) or not 0 <= loss_seed <= MAX_SEED:
        raise ExperimentError("loss seed is outside the supported seed domain")
    if isinstance(count, bool) or count <= 0:
        raise ExperimentError("loss sample count must be positive")
    forbidden = {int(seed) for seed in forbidden_seeds}
    selected: list[int] = []
    for index in range(count):
        counter = 0
        while True:
            digest = hashlib.sha256()
            digest.update(_LOSS_STREAM_DOMAIN)
            digest.update(loss_seed.to_bytes(8, "big", signed=False))
            digest.update(index.to_bytes(8, "big", signed=False))
            digest.update(counter.to_bytes(8, "big", signed=False))
            candidate = int.from_bytes(digest.digest()[:8], "big") & MAX_SEED
            if candidate not in forbidden and candidate not in selected:
                selected.append(candidate)
                break
            counter += 1
    return tuple(selected)


def _make_loss_noise(
    *,
    num_loss_seeds: int,
    latent_shape: Sequence[int],
    loss_seed: int,
    forbidden_seeds: Sequence[int] = (),
) -> torch.Tensor:
    seeds = _loss_sample_seeds(
        loss_seed,
        num_loss_seeds,
        forbidden_seeds=forbidden_seeds,
    )
    return make_initial_noise(seeds, latent_shape).float().contiguous()


def _reconstruct_generation_initial_latents(
    seeds: Sequence[int],
    latent_shape: Sequence[int],
    stored_dtype: torch.dtype,
    init_noise_sigma: float,
) -> torch.Tensor:
    """Reproduce the scheduler-scaled cached x_T states in seed order."""

    raw_noise = make_initial_noise(seeds, latent_shape)
    reconstructed = raw_noise.to(dtype=stored_dtype).mul(float(init_noise_sigma))
    if not bool(torch.isfinite(reconstructed).all().item()):
        raise ExperimentError("reconstructed generation initial latents are non-finite")
    return reconstructed.float().cpu().contiguous()


def _prediction_batch(
    conceptual_samples: torch.Tensor,
    *,
    condition: torch.Tensor,
    timestep: int,
    components: Any,
    scheduler: Any,
) -> torch.Tensor:
    conceptual = conceptual_samples.to(
        device=components.device,
        dtype=torch.float32,
    )
    model_samples = conceptual.to(dtype=components.inference_dtype)
    predictions = predict_conditional_epsilon(
        samples=model_samples,
        timestep=timestep,
        condition=condition,
        unet=components.unet,
        scheduler=scheduler,
        conversion_sample=conceptual,
    )
    if (
        not isinstance(predictions, torch.Tensor)
        or predictions.shape != conceptual.shape
    ):
        raise ExperimentError("conditional epsilon prediction has an invalid shape")
    if not bool(torch.isfinite(predictions).all().item()):
        raise ExperimentError("conditional epsilon prediction is non-finite")
    return predictions.detach().float().cpu().contiguous()


def _measure_conditional_loss(
    *,
    target_latent: torch.Tensor,
    loss_noise: torch.Tensor,
    condition: torch.Tensor,
    timestep: int,
    alpha_t: float,
    sigma_t: float,
    snr_t: float,
    components: Any,
    scheduler: Any,
    sample_batch_size: int,
) -> LossMeasurement:
    """Average SUM-reduced conditional epsilon losses over Q samples."""

    if (
        not isinstance(target_latent, torch.Tensor)
        or target_latent.ndim != 3
        or not target_latent.is_floating_point()
    ):
        raise ExperimentError("target latent must have shape [C, H, W]")
    if (
        not isinstance(loss_noise, torch.Tensor)
        or loss_noise.ndim != 4
        or tuple(loss_noise.shape[1:]) != tuple(target_latent.shape)
        or loss_noise.shape[0] < 1
    ):
        raise ExperimentError("loss noise has an invalid shape")
    dimension = target_latent.numel()
    target = target_latent.detach().float().cpu().contiguous()
    noise = loss_noise.detach().float().cpu().contiguous()
    losses: list[torch.Tensor] = []
    identity_prediction: torch.Tensor | None = None
    identity_corruption: torch.Tensor | None = None

    for start in range(0, noise.shape[0], sample_batch_size):
        stop = min(start + sample_batch_size, noise.shape[0])
        batch_noise = noise[start:stop]
        corrupted = alpha_t * target.unsqueeze(0) + sigma_t * batch_noise
        predictions = _prediction_batch(
            corrupted,
            condition=condition,
            timestep=timestep,
            components=components,
            scheduler=scheduler,
        )
        residual = batch_noise.double() - predictions.double()
        losses.append(residual.square().flatten(1).sum(dim=1))
        if identity_prediction is None:
            identity_prediction = predictions[0]
            identity_corruption = corrupted[0]

    per_sample_loss = torch.cat(losses)
    conditional_loss = float(per_sample_loss.mean().item())
    normalized_loss_mse = conditional_loss / (dimension * snr_t)
    normalized_loss_rmse = math.sqrt(normalized_loss_mse)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (
            conditional_loss,
            normalized_loss_mse,
            normalized_loss_rmse,
        )
    ):
        raise ExperimentError("conditional loss measurement is invalid")

    if identity_prediction is None or identity_corruption is None:
        raise ExperimentError("no forward-corruption sample was checked")
    recovered = (
        identity_corruption.double() - sigma_t * identity_prediction.double()
    ) / alpha_t
    identity_lhs = float(
        (recovered - target.double()).square().sum().item() / dimension
    )
    identity_rhs = float(per_sample_loss[0].item() / (dimension * snr_t))
    half_precision = components.inference_dtype in {torch.float16, torch.bfloat16}
    relative_tolerance = 5e-4 if half_precision else 2e-5
    absolute_tolerance = 5e-5 if half_precision else 2e-6
    if not math.isclose(
        identity_lhs,
        identity_rhs,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise ExperimentError(
            "forward-corruption recovery identity failed: "
            f"recovery_mse={identity_lhs:.9g}, "
            f"loss_over_d_snr={identity_rhs:.9g}"
        )
    return LossMeasurement(
        conditional_loss,
        normalized_loss_mse,
        normalized_loss_rmse,
    )


def _measure_recovery(
    *,
    target_latent: torch.Tensor,
    generation_initial_latents: torch.Tensor,
    condition: torch.Tensor,
    timestep: int,
    alpha_t: float,
    sigma_t: float,
    components: Any,
    scheduler: Any,
    sample_batch_size: int,
) -> RecoveryMeasurement:
    """Average one-prediction clean-latent recovery MSE over P samples."""

    if (
        not isinstance(generation_initial_latents, torch.Tensor)
        or generation_initial_latents.ndim != 4
        or tuple(generation_initial_latents.shape[1:]) != tuple(target_latent.shape)
        or generation_initial_latents.shape[0] < 1
    ):
        raise ExperimentError("generation initial latents have an invalid shape")
    target = target_latent.detach().float().cpu().contiguous()
    initial = generation_initial_latents.detach().float().cpu().contiguous()
    dimension = target.numel()
    squared_errors: list[torch.Tensor] = []
    for start in range(0, initial.shape[0], sample_batch_size):
        stop = min(start + sample_batch_size, initial.shape[0])
        samples = initial[start:stop]
        predictions = _prediction_batch(
            samples,
            condition=condition,
            timestep=timestep,
            components=components,
            scheduler=scheduler,
        )
        recovered = (samples.double() - sigma_t * predictions.double()) / alpha_t
        squared_errors.append(
            (recovered - target.double()).square().flatten(1).sum(dim=1).div(dimension)
        )
    recovery_mse = float(torch.cat(squared_errors).mean().item())
    recovery_rmse = math.sqrt(recovery_mse)
    if not all(
        math.isfinite(value) and value >= 0.0 for value in (recovery_mse, recovery_rmse)
    ):
        raise ExperimentError("recovery measurement is invalid")
    return RecoveryMeasurement(recovery_mse, recovery_rmse)


def _pair_identity(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "record_id": metadata.get("record_id"),
        "source_row_number": metadata.get("source_row_number"),
        "prompt_raw": metadata.get("prompt_raw"),
        "target_image_sha256": metadata.get("target_image_sha256"),
    }


def _validate_generation_pair(
    contract: GenerationContract,
    metadata: Mapping[str, object],
) -> Mapping[str, Any]:
    index = safe_index(metadata.get("original_index"))
    validation = validate_generation_record(
        contract.paths,
        index,
        expected_scientific_hash=contract.scientific_hash,
        expected_record_identity=_pair_identity(metadata),
        load_tensors=False,
    )
    if not validation.valid or validation.metadata is None:
        raise ExperimentError(
            f"invalid generation cache for {metadata.get('record_id')}: "
            + "; ".join(validation.errors)
        )
    marker = validation.metadata
    if marker.get("model_cli_name") != contract.science.get("model_cli_name"):
        raise ExperimentError("generation model name differs from canonical cache")
    if marker.get("seeds") != list(contract.seeds):
        raise ExperimentError(
            "generation record seed order differs from canonical cache"
        )
    return marker


def _load_or_encode_target_latent(
    *,
    item: Mapping[str, object],
    metadata: Mapping[str, object],
    contract: Any,
    components: Any,
) -> torch.Tensor:
    target_path_method = getattr(contract.paths, "target_latent_path", None)
    if callable(target_path_method):
        marker = _validate_generation_pair(contract, metadata)
        index = safe_index(metadata.get("original_index"))
        target = safe_torch_load(target_path_method(index))
        if not isinstance(target, torch.Tensor):
            raise ExperimentError("cached target latent is not a tensor")
        expected_shape = tuple(contract.latent_shape)
        if (
            tuple(target.shape) != expected_shape
            or target.dtype is not torch.float32
            or target.device.type != "cpu"
            or not target.is_contiguous()
            or target.requires_grad
            or not bool(torch.isfinite(target).all().item())
        ):
            raise ExperimentError("cached target latent violates its tensor contract")
        if marker.get("target_latent_definition") != TARGET_LATENT_DEFINITION:
            raise ExperimentError("cached target latent definition differs")
        return target.detach().clone().contiguous()

    image = item.get("image")
    if image is None:
        raise ExperimentError("paired target image is unavailable")
    return encode_target_latent(
        image,
        components.vae,
        components.device,
        resolution=components.spec.resolution if hasattr(components, "spec") else 512,
        expected_channels=int(contract.latent_shape[0]),
        encoding_dtype=torch.float32,
    )


def _load_mean_target_sscd(
    *,
    contract: GenerationContract,
    metadata: Mapping[str, object],
    generation_seeds: Sequence[int],
) -> float:
    """Load exact target-specific scores and select the recovery seed positions."""

    if contract.sscd_error is not None:
        raise ExperimentError(contract.sscd_error)
    configuration = contract.sscd_configuration
    configuration_hash = contract.sscd_configuration_hash
    if configuration is None or configuration_hash is None:
        raise ExperimentError("target-specific SSCD configuration is unavailable")
    requested = tuple(int(seed) for seed in generation_seeds)
    configured = _integer_tuple(configuration.get("seeds"), "SSCD seeds")
    missing = [seed for seed in requested if seed not in configured]
    if missing:
        raise ExperimentError(
            "requested target-specific SSCD seeds are missing: "
            + ",".join(str(seed) for seed in missing)
        )

    generation_marker = _validate_generation_pair(contract, metadata)
    index = safe_index(metadata.get("original_index"))
    marker_path = contract.sscd_paths.marker_path(index)
    score_path = contract.sscd_paths.score_path(index)
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ExperimentError("target-specific SSCD marker is missing or unsafe")
    if not score_path.is_file() or score_path.is_symlink():
        raise ExperimentError("target-specific SSCD tensor is missing or unsafe")
    marker = read_json(marker_path)
    expected = {
        "original_index": index,
        "record_id": metadata.get("record_id"),
        "source_row_number": metadata.get("source_row_number"),
        "prompt_raw": metadata.get("prompt_raw"),
        "target_image_sha256": metadata.get("target_image_sha256"),
        "model_id": contract.science.get("model_id"),
        "model_revision": contract.science.get("model_revision"),
        "num_seeds": len(configured),
        "seeds": list(configured),
        "similarity": SCORE_DEFINITION,
        "sscd_configuration_hash": configuration_hash,
        "generation_record_sha256": file_sha256(contract.paths.record_path(index)),
        "generation_latent_sha256": generation_marker.get("tensor_file_sha256", {}).get(
            "latent"
        )
        if isinstance(generation_marker.get("tensor_file_sha256"), Mapping)
        else None,
    }
    wrong = [
        key
        for key, expected_value in expected.items()
        if marker.get(key) != expected_value
    ]
    if wrong:
        raise ExperimentError(
            "target-specific SSCD record differs at: " + ", ".join(wrong)
        )
    if marker.get("score_sha256") != file_sha256(score_path):
        raise ExperimentError("target-specific SSCD tensor hash differs")
    scores = validate_sscd_scores(safe_torch_load(score_path), len(configured))
    positions = {seed: position for position, seed in enumerate(configured)}
    selected = scores[[positions[seed] for seed in requested]].double()
    result = float(selected.mean().item())
    if not math.isfinite(result):
        raise ExperimentError("mean target-specific SSCD is non-finite")
    return result


def _validate_item_identity(
    item: Mapping[str, object],
    metadata: Mapping[str, object],
    dataset_model_name: str,
) -> None:
    prompt = item.get("prompt")
    expected = {
        "record_id": metadata.get("record_id"),
        "original_index": metadata.get("original_index"),
        "source_row_number": metadata.get("source_row_number"),
    }
    wrong = [
        key
        for key, expected_value in expected.items()
        if item.get(key) != expected_value
    ]
    if (
        item.get("model_name") != dataset_model_name
        or metadata.get("model_name") != dataset_model_name
    ):
        wrong.append("model_name")
    if prompt != metadata.get("prompt_raw") or not isinstance(prompt, str):
        wrong.append("prompt_raw")
    target_hash = item.get("target_image_sha256")
    if not (
        isinstance(target_hash, str)
        and len(target_hash) == 64
        and all(character in "0123456789abcdef" for character in target_hash)
    ):
        wrong.append("target_image_sha256")
    if wrong:
        raise ExperimentError(
            "Webster pair identity differs at: " + ", ".join(sorted(set(wrong)))
        )


def _base_row(
    *,
    record_id: str,
    model_name: str,
    terminal: TerminalParameters,
    latent_dimension: int,
    num_loss_seeds: int,
    loss_seed: int,
    generation_seeds: Sequence[int],
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "model_name": model_name,
        "timestep": terminal.timestep,
        "alpha_t": terminal.alpha_t,
        "sigma_t": terminal.sigma_t,
        "snr_t": terminal.snr_t,
        "latent_dimension": latent_dimension,
        "num_loss_seeds": num_loss_seeds,
        "loss_seed": loss_seed,
        "generation_seeds": ",".join(str(seed) for seed in generation_seeds),
        "num_generation_seeds": len(generation_seeds),
        "conditional_loss": math.nan,
        "normalized_loss_mse": math.nan,
        "normalized_loss_rmse": math.nan,
        "recovery_mse": math.nan,
        "recovery_rmse": math.nan,
        "mean_target_sscd": math.nan,
        "status": "error",
        "error": "",
    }


def _prepare_output_directory(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    allowed = {CSV_NAME, FIGURE_NAME}
    unexpected = sorted(
        item.name for item in path.iterdir() if item.name not in allowed
    )
    if unexpected:
        raise ExperimentError(
            "output directory contains unexpected artifacts: " + ", ".join(unexpected)
        )
    return path


def _default_output_directory(model_name: str) -> Path:
    run_name = generation_run_name(
        model_name,
        SCHEDULER_NAME,
        GUIDANCE_SCALE,
        NUM_INFERENCE_STEPS,
        len(CANONICAL_GENERATION_SEEDS),
        0,
    )
    return ROOT / "outputs" / run_name / EXPERIMENT_DIRECTORY


def _binned_medians(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    maximum_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x_values) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    order = np.argsort(x_values, kind="stable")
    sorted_x = np.asarray(x_values, dtype=float)[order]
    sorted_y = np.asarray(y_values, dtype=float)[order]
    bins = np.array_split(np.arange(len(sorted_x)), min(maximum_bins, len(sorted_x)))
    median_x = np.asarray([np.median(sorted_x[index]) for index in bins], dtype=float)
    median_y = np.asarray([np.median(sorted_y[index]) for index in bins], dtype=float)
    return median_x, median_y


def _atomic_save_figure(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        figure.savefig(
            temporary,
            format="pdf",
            bbox_inches="tight",
            pad_inches=FIGURE_PAD_INCHES,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _render_pair_figure(
    *,
    valid: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
    colors: np.ndarray,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    try:
        color_norm = Normalize(
            vmin=SSCD_COLOR_RANGE[0],
            vmax=SSCD_COLOR_RANGE[1],
            clip=True,
        )
        scatter = axis.scatter(
            x_values,
            y_values,
            c=colors,
            cmap="viridis",
            norm=color_norm,
            s=22,
            alpha=SCATTER_ALPHA,
            edgecolors="none",
        )
        colorbar_mappable = ScalarMappable(norm=scatter.norm, cmap=scatter.cmap)
        colorbar_mappable.set_array(colors)
        colorbar = figure.colorbar(colorbar_mappable, ax=axis)
        if colorbar.solids is not None:
            colorbar.solids.set_alpha(COLORBAR_ALPHA)
        colorbar.set_label(COLORBAR_LABEL, fontsize=TEXT_FONT_SIZE)
        colorbar.ax.tick_params(labelsize=AXIS_NUMBER_FONT_SIZE)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0, 3.0, 6.0), numticks=8)
        )
        axis.xaxis.set_major_formatter(
            LogFormatterSciNotation(
                base=10,
                labelOnlyBase=False,
                minor_thresholds=(math.inf, math.inf),
            )
        )
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(
            axis="both",
            which="both",
            labelsize=AXIS_NUMBER_FONT_SIZE,
        )
        if len(valid) == 0:
            axis.set_xlim(1e-3, 1.0)
            axis.set_ylim(1e-3, 1.0)
        axis.set_xlabel(
            r"$\sqrt{\mathcal{L}^{\star}_{T,c} / "
            r"[d\,(\alpha_T^2/\sigma_T^2)]}$"
        )
        axis.set_ylabel(
            r"$\sqrt{\mathbb{E}_{\mathbf{x}_T\sim"
            r"\mathcal{N}(\mathbf{0},\mathbf{I})}"
            r"[\|\hat{\mathbf{x}}_{0\mid T,c}-\mathbf{x}^{\star}\|_2^2/d]}$"
        )
        axis.grid(True, which="both", alpha=0.18, linewidth=0.6)

        median_x, median_y = _binned_medians(x_values, y_values)
        if len(median_x) >= 2:
            axis.plot(
                median_x,
                median_y,
                color="black",
                linewidth=1.35,
                marker="o",
                markersize=3.5,
                label="Binned median",
            )
            axis.legend(frameon=False, loc="best", fontsize=LEGEND_FONT_SIZE)
        else:
            print(
                "WARNING: fewer than two binned medians are available; "
                "omitting the Binned median line.",
                file=sys.stderr,
            )

        figure.tight_layout()
        _atomic_save_figure(figure, destination)
    finally:
        plt.close(figure)


def plot_saved_results(
    csv_path: str | Path,
    figure_path: str | Path,
) -> None:
    """Reload the sole CSV and render the sole pair-level figure."""

    source = Path(csv_path)
    destination = Path(figure_path)
    frame = pd.read_csv(source)
    if tuple(frame.columns) != CSV_COLUMNS:
        raise ExperimentError("saved CSV columns do not match the experiment schema")
    if frame["record_id"].astype(str).duplicated().any():
        raise ExperimentError("saved CSV contains duplicate prompt-target pairs")

    x_numeric = pd.to_numeric(frame["normalized_loss_rmse"], errors="coerce")
    y_numeric = pd.to_numeric(frame["recovery_rmse"], errors="coerce")
    color_numeric = pd.to_numeric(frame["mean_target_sscd"], errors="coerce")
    valid_mask = (
        frame["status"].eq("ok")
        & np.isfinite(x_numeric)
        & np.isfinite(y_numeric)
        & np.isfinite(color_numeric)
        & x_numeric.gt(0.0)
        & y_numeric.gt(0.0)
    )
    valid = frame.loc[valid_mask].copy()
    x_values = x_numeric.loc[valid_mask].to_numpy(dtype=float)
    y_values = y_numeric.loc[valid_mask].to_numpy(dtype=float)
    colors = color_numeric.loc[valid_mask].to_numpy(dtype=float)
    with matplotlib.rc_context(PLOT_STYLE):
        _render_pair_figure(
            valid=valid,
            x_values=x_values,
            y_values=y_values,
            colors=colors,
            destination=destination,
        )


def run_experiment(
    *,
    model_name: str,
    num_loss_seeds: int,
    loss_seed: int,
    generation_seeds: Sequence[int],
    max_generation_seeds: int | None,
    sample_batch_size: int,
    max_records: int | None,
    output_dir: str | Path,
    progress_factory: Callable[..., Any] = tqdm,
) -> int:
    """Run one conditional prediction under Q and P for every Webster pair."""

    output = _prepare_output_directory(output_dir)
    spec = get_model_spec(model_name)
    dataset = WebsterDataset(
        ROOT,
        spec.dataset_model,
        recovered_only=True,
        defer_image_validation=True,
    )
    metadata_rows = list(dataset.iter_metadata())
    if max_records is not None:
        metadata_rows = metadata_rows[:max_records]
    contract = _load_generation_contract(ROOT, model_name)
    selected_seeds = _selected_generation_seeds(
        generation_seeds,
        max_generation_seeds,
        contract.generation_seeds,
    )
    resolver = (
        _revision_resolver(contract)
        if isinstance(getattr(contract, "science", None), Mapping)
        else None
    )
    components = load_model_components(spec, revision_resolver=resolver)
    scheduler = _validate_active_scheduler(components, contract)
    preflight_model_components(
        components,
        scheduler=scheduler,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )
    _validate_loaded_components(components, contract)
    terminal = _terminal_parameters(scheduler)
    generation_initial_latents = _reconstruct_generation_initial_latents(
        selected_seeds,
        contract.latent_shape,
        contract.stored_dtype,
        contract.init_noise_sigma,
    )
    loss_noise = _make_loss_noise(
        num_loss_seeds=num_loss_seeds,
        latent_shape=contract.latent_shape,
        loss_seed=loss_seed,
        forbidden_seeds=selected_seeds,
    )
    latent_dimension = math.prod(contract.latent_shape)

    rows: list[dict[str, object]] = []
    failed = 0
    progress = progress_factory(
        total=len(metadata_rows),
        desc=PROGRESS_DESCRIPTION,
        dynamic_ncols=True,
        leave=True,
    )
    try:
        for position, metadata in enumerate(metadata_rows):
            record_id = str(metadata.get("record_id", f"pair-{position}"))
            progress.set_postfix(record_id=record_id)
            row = _base_row(
                record_id=record_id,
                model_name=model_name,
                terminal=terminal,
                latent_dimension=latent_dimension,
                num_loss_seeds=num_loss_seeds,
                loss_seed=loss_seed,
                generation_seeds=selected_seeds,
            )
            try:
                item = dataset[position]
                if not isinstance(item, Mapping):
                    raise ExperimentError("Webster dataset item is not a mapping")
                _validate_item_identity(item, metadata, spec.dataset_model)
                pair_metadata = dict(metadata)
                pair_metadata["target_image_sha256"] = item["target_image_sha256"]
                target_latent = _load_or_encode_target_latent(
                    item=item,
                    metadata=pair_metadata,
                    contract=contract,
                    components=components,
                )
                prompt = str(item["prompt"])
                condition = encode_prompt_condition(
                    prompt,
                    components.tokenizer,
                    components.text_encoder,
                    components.device,
                    components.inference_dtype,
                )
                loss = _measure_conditional_loss(
                    target_latent=target_latent,
                    loss_noise=loss_noise,
                    condition=condition,
                    timestep=terminal.timestep,
                    alpha_t=terminal.alpha_t,
                    sigma_t=terminal.sigma_t,
                    snr_t=terminal.snr_t,
                    components=components,
                    scheduler=scheduler,
                    sample_batch_size=sample_batch_size,
                )
                recovery = _measure_recovery(
                    target_latent=target_latent,
                    generation_initial_latents=generation_initial_latents,
                    condition=condition,
                    timestep=terminal.timestep,
                    alpha_t=terminal.alpha_t,
                    sigma_t=terminal.sigma_t,
                    components=components,
                    scheduler=scheduler,
                    sample_batch_size=sample_batch_size,
                )
                mean_sscd = _load_mean_target_sscd(
                    contract=contract,
                    metadata=pair_metadata,
                    generation_seeds=selected_seeds,
                )
                row.update(
                    {
                        "latent_dimension": target_latent.numel(),
                        "conditional_loss": loss.conditional_loss,
                        "normalized_loss_mse": loss.normalized_loss_mse,
                        "normalized_loss_rmse": loss.normalized_loss_rmse,
                        "recovery_mse": recovery.recovery_mse,
                        "recovery_rmse": recovery.recovery_rmse,
                        "mean_target_sscd": mean_sscd,
                        "status": "ok",
                        "error": "",
                    }
                )
            except Exception as error:
                failed += 1
                row["status"] = "error"
                row["error"] = (
                    " ".join(str(error).splitlines()).strip() or type(error).__name__
                )
            finally:
                rows.append(row)
                progress.update(1)
    finally:
        progress.close()

    csv_path = output / CSV_NAME
    figure_path = output / FIGURE_NAME
    atomic_write_csv(csv_path, rows, CSV_COLUMNS)
    plot_saved_results(csv_path, figure_path)
    return int(failed > 0)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    requested_output = (
        _default_output_directory(arguments.model)
        if arguments.output_dir is None
        else arguments.output_dir
    )
    output = _prepare_output_directory(requested_output)
    csv_path = output / CSV_NAME
    figure_path = output / FIGURE_NAME
    if arguments.plot_only:
        if not csv_path.is_file():
            raise ExperimentError(f"saved CSV is missing: {csv_path}")
        plot_saved_results(csv_path, figure_path)
        return 0
    return run_experiment(
        model_name=arguments.model,
        num_loss_seeds=arguments.num_loss_seeds,
        loss_seed=arguments.loss_seed,
        generation_seeds=arguments.generation_seeds,
        max_generation_seeds=arguments.max_generation_seeds,
        sample_batch_size=arguments.sample_batch_size,
        max_records=arguments.max_records,
        output_dir=output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
