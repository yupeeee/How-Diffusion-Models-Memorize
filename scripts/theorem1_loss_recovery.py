#!/usr/bin/env python3
"""Compare initial conditional recovery with selected cached generation trajectories.

The theorem's Gaussian statement is evaluated with one fixed bank of standard
Gaussian samples reused across timesteps. An explicitly separate trajectory
diagnostic evaluates the exact cached ``x_t`` and conditional epsilon at each
saved scheduler step; it is not used as a substitute for the theorem's
independent-Gaussian experiment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import math
import multiprocessing
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
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
    SCHEDULER_CHOICES,
    device_argument,
    finite_float,
    generation_run_name,
    nonnegative_integer,
    positive_integer,
    validate_seed_block,
)
from utils.common.io import (  # noqa: E402
    CacheIOError,
    atomic_write_csv,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.data.selection import (  # noqa: E402
    DEFAULT_SELECTION_STRATEGY,
    SELECTION_STRATEGIES,
    TargetPairSelection,
    TargetPairSelectionError,
    load_target_pair_selection,
)
from utils.data.webster import WebsterDataset  # noqa: E402
from utils.experiments.cache import (  # noqa: E402
    GenerationCacheError,
    GenerationPaths,
    generation_paths,
    safe_index,
    require_generation_run,
    validate_generation_record,
)
from utils.experiments.plotting import FIGURE_DPI  # noqa: E402
from utils.experiments.sscd import (  # noqa: E402
    SCORE_DEFINITION,
    SSCDPaths,
    sscd_configuration_hash,
)
from utils.metrics.sscd import (  # noqa: E402
    SSCD_SCORE_RANGE_TOLERANCE,
    validate_sscd_scores,
)
from utils.models.latent import (  # noqa: E402
    TARGET_LATENT_DEFINITION,
    target_preprocessing_policy,
)
from utils.models.devices import (  # noqa: E402
    DeviceSelectionError,
    configure_worker_cpu_threads,
    resolve_devices,
    round_robin_shard,
    worker_count_for_tasks,
)
from utils.models.loading import (  # noqa: E402
    compile_loaded_unet,
    load_model_components,
    preflight_model_components,
    select_runtime,
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
GAUSSIAN_FIGURE_FILENAMES = (
    "theorem1_loss_recovery.png",
    "theorem1_loss_recovery.pdf",
)
TRAJECTORY_FIGURE_FILENAMES = (
    "theorem1_loss_recovery_trajectory.png",
    "theorem1_loss_recovery_trajectory.pdf",
)
# Recognized only to remove obsolete figures; these outputs are never generated.
_RETIRED_FIGURE_FILENAMES = (
    "theorem1_loss_recovery_noise_sweep.png",
    "theorem1_loss_recovery_noise_sweep.pdf",
)
FIGURE_FILENAMES = (
    *GAUSSIAN_FIGURE_FILENAMES,
    *TRAJECTORY_FIGURE_FILENAMES,
)
CSV_COLUMNS = (
    "record_id",
    "selection_strategy",
    "selection_hash",
    "model_name",
    "scheduler_name",
    "guidance_scale",
    "num_inference_steps",
    "evaluation_generation_scientific_config_hash",
    "evaluation_schedule_sha256",
    "evaluation_source",
    "step_index",
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
    "trajectory_sha256",
    "status",
    "error",
)

DEFAULT_SAMPLE_BATCH_SIZE = 8
PROGRESS_DESCRIPTION = "Theorem 1 loss–recovery"
COLORBAR_LABEL = "SSCD"
SSCD_COLOR_RANGE = (0.0, 1.0)
FIGURE_SIZE = (4.0, 4.0)
TEXT_FONT_SIZE = 15
AXIS_NUMBER_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
OBSERVATION_LINE_ALPHA = 0.68
OBSERVATION_LINE_WIDTH = 0.75
COLORBAR_ALPHA = 1.0
FIGURE_PAD_INCHES = 0.05
EXPERIMENT_DIRECTORY = "theorem1_loss_recovery"
_LOSS_STREAM_DOMAIN = b"theorem1-pair-loss-noise-v1"
GAUSSIAN_SOURCE = "gaussian"
TRAJECTORY_SOURCE = "trajectory"
EVALUATION_SOURCES = (GAUSSIAN_SOURCE, TRAJECTORY_SOURCE)
EVALUATION_SOURCE_CHOICES = (*EVALUATION_SOURCES, "both")

INITIAL_LOSS_XLABEL = r"$\sqrt{\mathcal{L}_T(c)/[d(\alpha_T^2/\sigma_T^2)]}$"
SWEEP_LOSS_XLABEL = r"$\sqrt{\mathcal{L}_t(c)/[d(\alpha_t^2/\sigma_t^2)]}$"
INITIAL_RECOVERY_YLABEL = (
    r"$\sqrt{\mathbb{E}_{\mathbf{x}_T,\boldsymbol{\xi}}"
    r"[\|\widehat{\mathbf{x}}_{0\mid T,c}(\mathbf{x}_T)-"
    r"\mathbf{x}^{\star}\|^2]/d}$"
)

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


def _install_tqdm_lock(lock: Any) -> None:
    """Install the one inter-process lock shared by all progress bars."""

    tqdm.set_lock(lock)


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
class _PairShardResult:
    indexed_rows: tuple[tuple[int, dict[str, object]], ...]
    failed_count: int


@dataclass(frozen=True, slots=True)
class GenerationContract:
    paths: GenerationPaths
    sscd_paths: SSCDPaths
    science: Mapping[str, Any]
    scientific_hash: str
    latent_shape: tuple[int, int, int]
    stored_dtype: torch.dtype
    init_noise_sigma: float
    seeds: tuple[int, ...]
    schedule_sha256: str
    schedule_payload: Mapping[str, Any]
    scheduler_config: Mapping[str, Any]
    parameters: tuple[TerminalParameters, ...]
    sscd_configuration: Mapping[str, Any] | None
    sscd_configuration_hash: str | None
    sscd_error: str | None


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately small public command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pair-specific conditional loss and one-prediction clean-"
            "latent recovery at every saved diffusion timestep."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model", choices=model_names(), default="sdv1")
    parser.add_argument(
        "--selection-strategy",
        choices=SELECTION_STRATEGIES,
        default=DEFAULT_SELECTION_STRATEGY,
        help="frozen GMM-only prompt selection (default: gmm)",
    )
    parser.add_argument(
        "--evaluation-source",
        choices=EVALUATION_SOURCE_CHOICES,
        default="both",
        help=(
            "evaluate fixed independent Gaussian samples, exact cached trajectory "
            "states, or both (default: both)"
        ),
    )
    parser.add_argument("--scheduler", choices=SCHEDULER_CHOICES, default="ddim")
    parser.add_argument("--g", type=finite_float, default=7.5, metavar="SCALE")
    parser.add_argument(
        "--T",
        dest="num_inference_steps",
        type=positive_integer,
        default=50,
        metavar="STEPS",
    )
    parser.add_argument(
        "--num-loss-seeds",
        type=positive_integer,
        default=20,
        metavar="DRAWS",
        help="independent forward-corruption draws per pair and timestep",
    )
    parser.add_argument(
        "--N",
        dest="num_seeds",
        type=positive_integer,
        default=20,
        metavar="SEEDS",
    )
    parser.add_argument(
        "--loss-seed",
        type=nonnegative_integer,
        default=0,
        metavar="SEED",
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
        "--device",
        type=device_argument,
        default="auto",
        metavar="DEVICE",
        help=(
            "auto uses every CUDA device visible to PyTorch; cpu, mps, "
            "cuda, or cuda:N selects one device (default: auto)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute requested prompt measurements even when a valid log cache exists",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="regenerate source-specific PNG and PDF figures from the saved CSV",
    )
    return parser


def _load_frozen_selection(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str,
) -> TargetPairSelection:
    """Load the exact frozen selection requested by the experiment."""

    try:
        return load_target_pair_selection(
            root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            selection_strategy=selection_strategy,
        )
    except TargetPairSelectionError as error:
        raise ExperimentError(str(error)) from error


def _selected_dataset_entries(
    metadata_rows: Sequence[Mapping[str, object]],
    selection: TargetPairSelection,
    max_records: int | None,
) -> tuple[tuple[int, dict[str, object]], ...]:
    """Keep selected pairs while retaining their Webster dataset positions."""

    included = set(selection.included_indices)
    known = included | set(selection.excluded_indices)
    indexed: list[tuple[int, str, dict[str, object]]] = []
    observed: set[str] = set()
    for dataset_position, metadata in enumerate(metadata_rows):
        if not isinstance(metadata, Mapping):
            raise ExperimentError("Webster metadata row is not a mapping")
        try:
            original_index = safe_index(metadata.get("original_index"))
        except GenerationCacheError as error:
            raise ExperimentError(
                f"Webster metadata at position {dataset_position} has an invalid "
                "original_index"
            ) from error
        if original_index in observed:
            raise ExperimentError(
                f"Webster metadata repeats original_index {original_index}"
            )
        observed.add(original_index)
        indexed.append((dataset_position, original_index, dict(metadata)))

    unknown = sorted(observed - known)
    if unknown:
        raise ExperimentError(
            "Webster prompts are absent from the frozen selection: "
            + ", ".join(unknown[:5])
        )
    missing = sorted(included - observed)
    if missing:
        raise ExperimentError(
            "Selected prompts are absent from the recovered Webster dataset: "
            + ", ".join(missing[:5])
        )
    selected = tuple(
        (dataset_position, metadata)
        for dataset_position, original_index, metadata in indexed
        if original_index in included
    )
    if max_records is not None:
        selected = selected[:max_records]
    if not selected:
        raise ExperimentError(
            "frozen selection contains no recovered prompt-target pairs"
        )
    return selected


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _requested_sources(evaluation_source: str) -> tuple[str, ...]:
    if evaluation_source == "both":
        return EVALUATION_SOURCES
    if evaluation_source in EVALUATION_SOURCES:
        return (evaluation_source,)
    raise ExperimentError(f"unsupported evaluation source: {evaluation_source!r}")


def _saved_schedule_tensor(
    payload: Mapping[str, object],
    name: str,
    *,
    length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = payload.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (length,)
        or value.dtype is not dtype
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
    ):
        raise ExperimentError(f"saved schedule {name} violates its tensor contract")
    return value


def _saved_schedule_parameters(
    payload: Mapping[str, object], num_inference_steps: int
) -> tuple[TerminalParameters, ...]:
    timesteps = _saved_schedule_tensor(
        payload, "timesteps", length=num_inference_steps, dtype=torch.int64
    )
    alpha = _saved_schedule_tensor(
        payload, "alpha_t", length=num_inference_steps, dtype=torch.float32
    ).double()
    sigma = _saved_schedule_tensor(
        payload, "sigma_t", length=num_inference_steps, dtype=torch.float32
    ).double()
    cumulative = _saved_schedule_tensor(
        payload,
        "alphas_cumprod_t",
        length=num_inference_steps,
        dtype=torch.float32,
    ).double()
    if bool((timesteps < 0).any()) or (
        num_inference_steps > 1 and not bool((timesteps[:-1] > timesteps[1:]).all())
    ):
        raise ExperimentError("saved scheduler timesteps must be strictly descending")
    if bool((alpha <= 0).any()) or bool((sigma <= 0).any()):
        raise ExperimentError("saved scheduler coefficients must be positive")
    if not (
        torch.allclose(alpha.square(), cumulative, rtol=1e-5, atol=1e-6)
        and torch.allclose(sigma.square(), 1.0 - cumulative, rtol=1e-5, atol=1e-6)
    ):
        raise ExperimentError("saved scheduler coefficients are inconsistent")
    snr = alpha.square().div(sigma.square())
    if not bool(torch.isfinite(snr).all()) or bool((snr <= 0).any()):
        raise ExperimentError("saved alpha_t^2/sigma_t^2 values are invalid")
    return tuple(
        TerminalParameters(
            int(timesteps[index]),
            float(alpha[index]),
            float(sigma[index]),
            float(snr[index]),
        )
        for index in range(num_inference_steps)
    )


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
        if not isinstance(stored_hash, str) or stored_hash != sscd_configuration_hash(
            configuration
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
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> GenerationContract:
    """Validate the experiment generation and SSCD cache for seeds 0..N-1."""

    project = Path(root).expanduser().resolve()
    try:
        _, num_seeds = validate_seed_block(0, num_seeds)
    except ValueError as error:
        raise ExperimentError(str(error)) from error
    expected_seeds = tuple(range(num_seeds))
    paths = generation_paths(
        project,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
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
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "seeds": list(expected_seeds),
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
    if scheduler_metadata.get("name") != scheduler_name:
        wrong.append("scheduler")
    if wrong:
        raise ExperimentError(
            "experiment generation cache differs at: " + ", ".join(sorted(set(wrong)))
        )

    latent_shape = _integer_tuple(science.get("latent_shape"), "latent shape")
    if len(latent_shape) != 3 or any(size <= 0 for size in latent_shape):
        raise ExperimentError("latent shape must contain three positive dimensions")
    seeds = _integer_tuple(science.get("seeds"), "generation seeds")
    if seeds != expected_seeds:
        raise ExperimentError("experiment generation cache has the wrong seed block")
    storage = science.get("scientific_tensor_storage")
    if not isinstance(storage, Mapping):
        raise ExperimentError("generation tensor storage metadata is invalid")
    stored_dtype = _dtype_from_name(storage.get("dtype"))
    if storage.get("device") != "cpu" or storage.get("layout") != "contiguous":
        raise ExperimentError("generation tensors must be contiguous CPU tensors")

    if not paths.schedule.is_file() or paths.schedule.is_symlink():
        raise ExperimentError("saved generation schedule is missing")
    schedule_sha256 = file_sha256(paths.schedule)
    schedule_payload = safe_torch_load(paths.schedule)
    if not isinstance(schedule_payload, Mapping):
        raise ExperimentError("saved generation schedule must be a mapping")
    parameters = _saved_schedule_parameters(schedule_payload, num_inference_steps)
    init_noise_sigma = _number(
        schedule_payload.get("init_noise_sigma"),
        "saved scheduler initial-noise scale",
    )
    if not math.isclose(init_noise_sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ExperimentError("Theorem 1 requires generation init_noise_sigma = 1")
    scheduler_config = _normalize_scheduler_config(
        schedule_payload.get("scheduler_config")
    )
    science_scheduler_config = _normalize_scheduler_config(
        scheduler_metadata.get("config")
    )
    if canonical_hash(scheduler_config) != canonical_hash(science_scheduler_config):
        raise ExperimentError("saved and configured scheduler settings differ")
    if file_sha256(paths.schedule) != schedule_sha256:
        raise ExperimentError("saved generation schedule changed while loading")

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
        science=science,
        scientific_hash=scientific_hash,
        latent_shape=(latent_shape[0], latent_shape[1], latent_shape[2]),
        stored_dtype=stored_dtype,
        init_noise_sigma=init_noise_sigma,
        seeds=seeds,
        schedule_sha256=schedule_sha256,
        schedule_payload=schedule_payload,
        scheduler_config=scheduler_config,
        parameters=parameters,
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


def _offload_unused_vae(components: Any) -> None:
    """Release the validated VAE because this experiment reads cached targets."""

    device = torch.device(components.device)
    if device.type == "cpu":
        return
    move = getattr(components.vae, "to", None)
    if not callable(move):
        raise ExperimentError("loaded VAE has no device-transfer interface")
    try:
        move(device=torch.device("cpu"), dtype=torch.float32)
    except Exception as error:
        raise ExperimentError(f"cannot offload unused VAE: {error}") from error
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _set_scheduler_timesteps(
    scheduler: Any,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    setter = getattr(scheduler, "set_timesteps", None)
    if not callable(setter):
        raise ExperimentError("active scheduler has no timestep interface")
    setter(num_inference_steps, device=device)


def _validate_active_scheduler(
    components: Any,
    contract: GenerationContract,
    scheduler_name: str,
    num_inference_steps: int,
) -> Any:
    """Rebuild the requested scheduler and match the cached schedule."""

    result = build_scheduler(components.original_scheduler, scheduler_name)
    scheduler = result.scheduler
    _set_scheduler_timesteps(
        scheduler, num_inference_steps, torch.device(components.device)
    )
    active_timesteps = torch.as_tensor(scheduler.timesteps).detach().cpu().long()
    saved_timesteps = (
        torch.as_tensor(contract.schedule_payload["timesteps"]).detach().cpu().long()
    )
    if not torch.equal(active_timesteps, saved_timesteps):
        raise ExperimentError("active scheduler timesteps differ from generation cache")
    if len(contract.parameters) != num_inference_steps:
        raise ExperimentError("saved scheduler parameter count differs")
    for index, expected in enumerate(contract.parameters):
        alpha, sigma = alpha_sigma_for_timestep(
            scheduler,
            expected.timestep,
            device="cpu",
            dtype=torch.float64,
        )
        if not (
            math.isclose(float(alpha), expected.alpha_t, rel_tol=1e-5, abs_tol=1e-6)
            and math.isclose(float(sigma), expected.sigma_t, rel_tol=1e-5, abs_tol=1e-6)
        ):
            raise ExperimentError(
                f"active scheduler coefficients differ at step {index}"
            )
    active_config = _normalize_scheduler_config(result.config)
    if canonical_hash(active_config) != canonical_hash(contract.scheduler_config):
        raise ExperimentError(
            "active scheduler configuration differs from generation cache"
        )
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
        raise ExperimentError("active scheduler has no inference timesteps")
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


def _make_gaussian_recovery_samples(
    seeds: Sequence[int], latent_shape: Sequence[int]
) -> torch.Tensor:
    """Create standard-Gaussian theorem probes without cache-dtype rounding."""

    samples = make_initial_noise(seeds, latent_shape).float().cpu().contiguous()
    if not bool(torch.isfinite(samples).all().item()):
        raise ExperimentError("Gaussian recovery samples are non-finite")
    return samples


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


def _is_cuda_out_of_memory(error: RuntimeError, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return (
        isinstance(error, out_of_memory_type) or "out of memory" in str(error).lower()
    )


def _prediction_batches(
    conceptual_samples: torch.Tensor,
    *,
    condition: torch.Tensor,
    timestep: int,
    components: Any,
    scheduler: Any,
    initial_batch_size: int,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield predictions in order, reducing only CUDA-OOM batch attempts."""

    if (
        isinstance(initial_batch_size, bool)
        or not isinstance(initial_batch_size, int)
        or initial_batch_size <= 0
    ):
        raise ExperimentError("sample batch size must be a positive integer")
    device = torch.device(components.device)
    position = 0
    batch_size = min(initial_batch_size, conceptual_samples.shape[0])
    while position < conceptual_samples.shape[0]:
        stop = min(position + batch_size, conceptual_samples.shape[0])
        current_size = stop - position
        retry_after_oom = False
        try:
            predictions = _prediction_batch(
                conceptual_samples[position:stop],
                condition=condition,
                timestep=timestep,
                components=components,
                scheduler=scheduler,
            )
        except RuntimeError as error:
            if not _is_cuda_out_of_memory(error, device) or current_size == 1:
                raise
            batch_size = max(1, current_size // 2)
            retry_after_oom = True
        if retry_after_oom:
            torch.cuda.empty_cache()
            continue
        yield position, stop, predictions
        position = stop


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
    corrupted_samples = alpha_t * target.unsqueeze(0) + sigma_t * noise
    losses: list[torch.Tensor] = []
    identity_prediction: torch.Tensor | None = None
    identity_corruption: torch.Tensor | None = None

    for start, stop, predictions in _prediction_batches(
        corrupted_samples,
        condition=condition,
        timestep=timestep,
        components=components,
        scheduler=scheduler,
        initial_batch_size=sample_batch_size,
    ):
        batch_noise = noise[start:stop]
        residual = batch_noise.double() - predictions.double()
        losses.append(residual.square().flatten(1).sum(dim=1))
        if identity_prediction is None:
            identity_prediction = predictions[0]
            identity_corruption = corrupted_samples[0]

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
    for start, stop, predictions in _prediction_batches(
        initial,
        condition=condition,
        timestep=timestep,
        components=components,
        scheduler=scheduler,
        initial_batch_size=sample_batch_size,
    ):
        samples = initial[start:stop]
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


def _validate_cached_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    label: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype is not dtype
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ExperimentError(f"{label} violates its tensor contract")
    return value


def _trajectory_sha256(marker: Mapping[str, object]) -> str:
    hashes = marker.get("tensor_file_sha256")
    if not isinstance(hashes, Mapping):
        raise ExperimentError("generation marker has no tensor-file hashes")
    latent_hash = hashes.get("latent")
    prediction_hash = hashes.get("noise_prediction")
    if not _is_sha256(latent_hash) or not _is_sha256(prediction_hash):
        raise ExperimentError("generation trajectory hashes are invalid")
    return canonical_hash(
        {
            "latent": str(latent_hash),
            "noise_prediction": str(prediction_hash),
        }
    )


def _load_cached_conditional_trajectory(
    *,
    metadata: Mapping[str, object],
    contract: GenerationContract,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load exact cached x_t and conditional epsilon trajectories once."""

    index = safe_index(metadata.get("original_index"))
    validation = validate_generation_record(
        contract.paths,
        index,
        expected_scientific_hash=contract.scientific_hash,
        expected_record_identity=_pair_identity(metadata),
        load_tensors=False,
        tensor_names=("latent", "noise_prediction"),
        require_preview=False,
        verify_file_hashes=True,
    )
    if not validation.valid or validation.metadata is None:
        raise ExperimentError(
            f"invalid cached trajectory for {metadata.get('record_id')}: "
            + "; ".join(validation.errors)
        )
    marker = validation.metadata
    latent = _validate_cached_tensor(
        safe_torch_load(contract.paths.latent_path(index)),
        shape=(
            len(contract.seeds),
            len(contract.parameters) + 1,
            *contract.latent_shape,
        ),
        dtype=contract.stored_dtype,
        label=f"cached latent trajectory for {index}",
    )
    predictions = safe_torch_load(contract.paths.noise_prediction_path(index))
    if not isinstance(predictions, tuple) or len(predictions) != 2:
        raise ExperimentError(
            f"cached noise prediction for {index} is not a two-tensor tuple"
        )
    prediction_shape = (
        len(contract.seeds),
        len(contract.parameters),
        *contract.latent_shape,
    )
    _validate_cached_tensor(
        predictions[0],
        shape=prediction_shape,
        dtype=contract.stored_dtype,
        label=f"cached unconditional epsilon trajectory for {index}",
    )
    conditional = _validate_cached_tensor(
        predictions[1],
        shape=prediction_shape,
        dtype=contract.stored_dtype,
        label=f"cached conditional epsilon trajectory for {index}",
    )
    return latent, conditional, _trajectory_sha256(marker)


def _measure_trajectory_recovery(
    *,
    target_latent: torch.Tensor,
    latent_trajectory: torch.Tensor,
    conditional_epsilon: torch.Tensor,
    step_index: int,
    alpha_t: float,
    sigma_t: float,
) -> RecoveryMeasurement:
    """Aggregate recovery from cached latent[:, k] and epsilon[:, k]."""

    if not 0 <= step_index < conditional_epsilon.shape[1]:
        raise ExperimentError("trajectory step index is outside the cached prediction")
    samples = latent_trajectory[:, step_index].double()
    predictions = conditional_epsilon[:, step_index].double()
    target = target_latent.detach().double().cpu().contiguous()
    if tuple(samples.shape) != (latent_trajectory.shape[0], *target.shape):
        raise ExperimentError("cached trajectory sample shape differs from target")
    if predictions.shape != samples.shape:
        raise ExperimentError("cached conditional prediction shape differs from x_t")
    dimension = target.numel()
    recovered = (samples - sigma_t * predictions) / alpha_t
    recovery_mse = float(
        (recovered - target.unsqueeze(0)).square().flatten(1).sum(1).mean().item()
        / dimension
    )
    recovery_rmse = math.sqrt(recovery_mse)
    if not all(
        math.isfinite(value) and value >= 0.0 for value in (recovery_mse, recovery_rmse)
    ):
        raise ExperimentError("trajectory recovery measurement is invalid")

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
        tensor_names=("target_latent",),
        require_preview=False,
        verify_file_hashes=True,
    )
    if not validation.valid or validation.metadata is None:
        raise ExperimentError(
            f"invalid generation cache for {metadata.get('record_id')}: "
            + "; ".join(validation.errors)
        )
    marker = validation.metadata
    if marker.get("model_cli_name") != contract.science.get("model_cli_name"):
        raise ExperimentError("generation model name differs from experiment cache")
    if marker.get("seeds") != list(contract.seeds):
        raise ExperimentError(
            "generation record seed order differs from experiment cache"
        )
    return marker


def _load_target_latent(
    *,
    metadata: Mapping[str, object],
    contract: GenerationContract,
    generation_marker: Mapping[str, Any],
) -> torch.Tensor:
    index = safe_index(metadata.get("original_index"))
    target = safe_torch_load(contract.paths.target_latent_path(index))
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
    if generation_marker.get("target_latent_definition") != TARGET_LATENT_DEFINITION:
        raise ExperimentError("cached target latent definition differs")
    return target.detach().clone().contiguous()


def _load_mean_target_sscd(
    *,
    contract: GenerationContract,
    metadata: Mapping[str, object],
    generation_seeds: Sequence[int],
    generation_marker: Mapping[str, Any],
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
        "generation_scientific_config_hash": contract.scientific_hash,
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
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    scientific_hash: str,
    schedule_sha256: str,
    evaluation_source: str,
    step_index: int,
    parameters: TerminalParameters,
    latent_dimension: int,
    num_loss_seeds: int,
    loss_seed: int,
    generation_seeds: Sequence[int],
    trajectory_sha256: str = "",
) -> dict[str, object]:
    if evaluation_source not in EVALUATION_SOURCES:
        raise ExperimentError(f"unsupported evaluation source: {evaluation_source!r}")
    return {
        "record_id": record_id,
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "evaluation_generation_scientific_config_hash": scientific_hash,
        "evaluation_schedule_sha256": schedule_sha256,
        "evaluation_source": evaluation_source,
        "step_index": step_index,
        "timestep": parameters.timestep,
        "alpha_t": parameters.alpha_t,
        "sigma_t": parameters.sigma_t,
        "snr_t": parameters.snr_t,
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
        "trajectory_sha256": trajectory_sha256,
        "status": "error",
        "error": "",
    }


def _prepare_output_directory(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    allowed = {CSV_NAME, *FIGURE_FILENAMES, *_RETIRED_FIGURE_FILENAMES}
    unexpected = sorted(
        item.name
        for item in path.iterdir()
        if item.name not in allowed or item.is_symlink() or not item.is_file()
    )
    if unexpected:
        raise ExperimentError(
            "output directory contains unexpected artifacts: " + ", ".join(unexpected)
        )
    _remove_noise_sweep_outputs(path)
    return path


def _default_output_directory(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str,
    selection_hash: str,
    evaluation_source: str,
) -> Path:
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ExperimentError(f"unsupported selection strategy: {selection_strategy!r}")
    if not _is_sha256(selection_hash):
        raise ExperimentError("selection hash must be a lowercase SHA-256 digest")
    _requested_sources(evaluation_source)
    run_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        0,
    )
    return (
        ROOT
        / "outputs"
        / run_name
        / EXPERIMENT_DIRECTORY
        / selection_hash
        / f"evaluation_{evaluation_source}"
    )


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


def _figure_paths(
    output_directory: str | Path,
    evaluation_source: str | None = None,
) -> tuple[Path, ...]:
    output = Path(output_directory)
    if evaluation_source is None or evaluation_source == "both":
        filenames = FIGURE_FILENAMES
    elif evaluation_source == GAUSSIAN_SOURCE:
        filenames = GAUSSIAN_FIGURE_FILENAMES
    elif evaluation_source == TRAJECTORY_SOURCE:
        filenames = TRAJECTORY_FIGURE_FILENAMES
    else:
        raise ExperimentError(f"unsupported evaluation source: {evaluation_source!r}")
    return tuple(output / filename for filename in filenames)


def _remove_noise_sweep_outputs(output_directory: str | Path) -> None:
    """Remove only the two obsolete Gaussian-sweep figures, never measurements."""
    paths = tuple(Path(output_directory) / name for name in _RETIRED_FIGURE_FILENAMES)
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ExperimentError(f"obsolete noise-sweep destination is unsafe: {path}")
    for path in paths:
        path.unlink(missing_ok=True)


def _remove_figure_outputs(output_directory: str | Path) -> None:
    for destination in _figure_paths(output_directory):
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
        elif destination.exists():
            raise ExperimentError(
                f"figure destination is not a regular file: {destination}"
            )


def _atomic_save_figures(figure: Any, destinations: Sequence[Path]) -> None:
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path | None, Path]] = []
    committed = False
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure_format = destination.suffix.removeprefix(".").lower()
            if figure_format not in {"png", "pdf"}:
                raise ExperimentError(
                    f"unsupported Theorem 1 figure format: {destination.suffix!r}"
                )
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            staged.append((temporary, destination))
            save_options: dict[str, object] = {
                "format": figure_format,
                "bbox_inches": "tight",
                "pad_inches": FIGURE_PAD_INCHES,
                "dpi": FIGURE_DPI,
            }
            figure.savefig(temporary, **save_options)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        for _temporary, destination in staged:
            if destination.exists() and not (
                destination.is_file() or destination.is_symlink()
            ):
                raise ExperimentError(
                    f"Theorem 1 figure destination is not a regular file: {destination}"
                )
        for temporary, destination in staged:
            backup = None
            if destination.is_file() or destination.is_symlink():
                backup = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.tmp"
                )
                os.replace(destination, backup)
            backups.append((backup, destination))
            os.replace(temporary, destination)
        committed = True
    except BaseException:
        for backup, destination in reversed(backups):
            if destination.is_file() or destination.is_symlink():
                destination.unlink()
            elif destination.exists():
                raise ExperimentError(
                    "cannot restore Theorem 1 figure because its destination "
                    f"became unsafe: {destination}"
                )
            if backup is not None:
                os.replace(backup, destination)
        raise
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)
        if committed:
            for backup, _destination in backups:
                if backup is not None:
                    backup.unlink(missing_ok=True)


def _source_ylabel(evaluation_source: str) -> str:
    if evaluation_source == GAUSSIAN_SOURCE:
        return (
            r"$\sqrt{\mathbb{E}_{\mathbf{x}_T,\boldsymbol{\xi}}"
            r"[\|\widehat{\mathbf{x}}_{0\mid t,c}(\mathbf{x}_T)-"
            r"\mathbf{x}^{\star}\|^2]/d}$"
        )
    if evaluation_source == TRAJECTORY_SOURCE:
        return (
            r"$\sqrt{\mathbb{E}_{\mathbf{x}_T,\boldsymbol{\xi}}"
            r"[\|\widehat{\mathbf{x}}_{0\mid t,c}-"
            r"\mathbf{x}^{\star}\|^2]/d}$"
        )
    raise ExperimentError(f"unsupported evaluation source: {evaluation_source!r}")


def _set_recovery_axis_scale(
    axis: Any, values: np.ndarray, *, coordinate: str = "y"
) -> None:
    """Use logarithmic RMSE scaling without hiding exact zero observations."""

    if coordinate not in {"x", "y"}:
        raise ExperimentError("RMSE axis coordinate must be x or y")
    set_scale = axis.set_xscale if coordinate == "x" else axis.set_yscale

    flattened = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(flattened).all() or np.any(flattened < 0.0):
        raise ExperimentError("recovery-axis values must be finite and nonnegative")
    positive = flattened[flattened > 0.0]
    if positive.size == flattened.size:
        set_scale("log")
    elif positive.size:
        set_scale(
            "symlog",
            linthresh=max(float(positive.min()) / 10.0, np.finfo(float).tiny),
        )
    else:
        set_scale("linear")


def _add_sscd_colorbar(
    figure: Any,
    axis: Any,
    color_norm: Normalize,
    scores: np.ndarray,
) -> None:
    mappable = ScalarMappable(norm=color_norm, cmap="viridis")
    mappable.set_array(scores)
    colorbar = figure.colorbar(mappable, ax=axis)
    if colorbar.solids is not None:
        colorbar.solids.set_alpha(COLORBAR_ALPHA)
    colorbar.set_label(COLORBAR_LABEL, fontsize=TEXT_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=AXIS_NUMBER_FONT_SIZE)


def _render_initial_loss_recovery(
    *, valid: pd.DataFrame, destinations: Sequence[Path]
) -> None:
    """Compare independent forward loss and Gaussian recovery only at initialization."""
    if (
        valid.empty
        or not valid["evaluation_source"].eq(GAUSSIAN_SOURCE).all()
        or not valid["step_index"].eq(0).all()
        or valid["record_id"].duplicated().any()
    ):
        raise ExperimentError(
            "initial loss-recovery figure requires one Gaussian initial row per prompt"
        )
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    try:
        color_norm = Normalize(
            vmin=SSCD_COLOR_RANGE[0], vmax=SSCD_COLOR_RANGE[1], clip=True
        )
        x_values = valid["normalized_loss_rmse"].to_numpy(dtype=float)
        y_values = valid["recovery_rmse"].to_numpy(dtype=float)
        scores = valid["mean_target_sscd"].to_numpy(dtype=float)
        # Prompts are independent observations at one timestep, not a trajectory.
        axis.scatter(
            x_values,
            y_values,
            c=scores,
            cmap="viridis",
            norm=color_norm,
            s=18,
            alpha=OBSERVATION_LINE_ALPHA,
            linewidths=0.0,
            zorder=2,
        )
        _add_sscd_colorbar(figure, axis, color_norm, scores)
        _set_recovery_axis_scale(axis, x_values, coordinate="x")
        _set_recovery_axis_scale(axis, y_values)
        # Initial-prompt RMSE ranges can span less than a decade. Keep labels
        # sparse enough to remain legible at the standard four-inch figure size.
        for coordinate, scale in (
            (axis.xaxis, axis.get_xscale()),
            (axis.yaxis, axis.get_yscale()),
        ):
            if scale == "log":
                coordinate.set_major_locator(
                    LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=7)
                )
                coordinate.set_major_formatter(
                    LogFormatterSciNotation(
                        base=10,
                        labelOnlyBase=False,
                        minor_thresholds=(math.inf, math.inf),
                    )
                )
                coordinate.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="both", which="both", labelsize=AXIS_NUMBER_FONT_SIZE)
        axis.set_xlabel(INITIAL_LOSS_XLABEL)
        axis.set_ylabel(INITIAL_RECOVERY_YLABEL)
        axis.grid(True, which="both", alpha=0.18, linewidth=0.6)
        # The theorem is not a finite-noise RMSE equality; do not imply y = x.
        figure.tight_layout()
        _atomic_save_figures(figure, destinations)
    finally:
        plt.close(figure)


def _render_pair_figure(
    *,
    valid: pd.DataFrame,
    evaluation_source: str,
    destinations: Sequence[Path],
    initial: pd.DataFrame | None = None,
) -> None:
    """Show selected cached trajectories with the unchanged initial scatter on top."""
    if evaluation_source != TRAJECTORY_SOURCE:
        raise ExperimentError(
            "trajectory figure requires cached-trajectory measurements"
        )
    initial_points = (
        valid.loc[valid["step_index"].eq(0)].copy() if initial is None else initial
    )
    if (
        initial_points.empty
        or initial_points["record_id"].duplicated().any()
        or not initial_points["step_index"].eq(0).all()
        or set(initial_points["record_id"]) != set(valid["record_id"])
    ):
        raise ExperimentError(
            "initial scatter must match the selected trajectory prompts"
        )

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    try:
        color_norm = Normalize(
            vmin=SSCD_COLOR_RANGE[0],
            vmax=SSCD_COLOR_RANGE[1],
            clip=True,
        )
        observation_segments: list[np.ndarray] = []
        observation_colors: list[float] = []
        for (_record_id, _loss_seed), group in valid.groupby(
            ["record_id", "loss_seed"], sort=False
        ):
            # Loss may be nonmonotonic: preserve the actual timestep path.
            ordered = group.sort_values("step_index", kind="stable")
            observation_segments.append(
                ordered[["normalized_loss_rmse", "recovery_rmse"]].to_numpy(dtype=float)
            )
            observation_colors.append(float(ordered["mean_target_sscd"].iloc[0]))
        observations = LineCollection(
            observation_segments,
            cmap="viridis",
            norm=color_norm,
            linewidths=OBSERVATION_LINE_WIDTH,
            alpha=OBSERVATION_LINE_ALPHA,
            zorder=2,
        )
        observations.set_array(np.asarray(observation_colors, dtype=float))
        axis.add_collection(observations)
        # Use the exact primary-figure coordinates, not snapped trajectory starts.
        # Opaque markers above the lines reveal both overplotting and small
        # differences caused by cached precision at the initial timestep.
        axis.scatter(
            initial_points["normalized_loss_rmse"].to_numpy(dtype=float),
            initial_points["recovery_rmse"].to_numpy(dtype=float),
            c=initial_points["mean_target_sscd"].to_numpy(dtype=float),
            cmap="viridis",
            norm=color_norm,
            s=18,
            alpha=1.0,
            linewidths=0.0,
            zorder=3,
        )
        _add_sscd_colorbar(
            figure, axis, color_norm, np.asarray(observation_colors, dtype=float)
        )

        _set_recovery_axis_scale(
            axis,
            np.concatenate(
                [valid["normalized_loss_rmse"], initial_points["normalized_loss_rmse"]]
            ),
            coordinate="x",
        )
        _set_recovery_axis_scale(
            axis,
            np.concatenate([valid["recovery_rmse"], initial_points["recovery_rmse"]]),
        )
        if axis.get_xscale() == "log":
            axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=7))
            axis.xaxis.set_major_formatter(
                LogFormatterSciNotation(
                    base=10,
                    labelOnlyBase=False,
                    minor_thresholds=(math.inf, math.inf),
                )
            )
            axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="both", which="both", labelsize=AXIS_NUMBER_FONT_SIZE)
        axis.set_xlabel(SWEEP_LOSS_XLABEL)
        axis.set_ylabel(_source_ylabel(evaluation_source))
        axis.grid(True, which="both", alpha=0.18, linewidth=0.6)

        # There is no shared x coordinate per timestep: each prompt has its
        # own forward loss. A normalized-loss reference would just compare x
        # against itself, so show prompt curves and their initial scatter only.
        axis.autoscale_view()
        figure.tight_layout()
        _atomic_save_figures(figure, destinations)
    finally:
        plt.close(figure)


def _expected_csv_configuration(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_loss_seeds: int,
    loss_seed: int,
    num_seeds: int,
    selection_strategy: str,
    selection_hash: str,
    evaluation_source: str,
    scientific_hash: str,
    schedule_sha256: str,
) -> dict[str, object]:
    return {
        "selection_strategy": selection_strategy,
        "selection_hash": selection_hash,
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_loss_seeds": num_loss_seeds,
        "loss_seed": loss_seed,
        "generation_seeds": ",".join(str(seed) for seed in range(num_seeds)),
        "num_generation_seeds": num_seeds,
        "evaluation_generation_scientific_config_hash": scientific_hash,
        "evaluation_schedule_sha256": schedule_sha256,
        "evaluation_sources": _requested_sources(evaluation_source),
    }


def _validate_csv_configuration(
    frame: pd.DataFrame,
    expected: Mapping[str, object],
) -> None:
    if frame.empty:
        raise ExperimentError("saved CSV has no prompt-target rows")
    mismatched: list[str] = []
    for column, expected_value in expected.items():
        if column == "evaluation_sources":
            continue
        values = frame[column]
        if isinstance(expected_value, (int, float)):
            observed = pd.to_numeric(values, errors="coerce")
            matches = observed.notna() & observed.eq(expected_value)
        else:
            matches = values.astype(str).eq(str(expected_value))
        if not bool(matches.all()):
            mismatched.append(column)
    expected_sources = tuple(str(value) for value in expected["evaluation_sources"])
    observed_sources = tuple(
        source
        for source in EVALUATION_SOURCES
        if source in set(frame["evaluation_source"])
    )
    if observed_sources != expected_sources:
        mismatched.append("evaluation_source")
    if mismatched:
        raise ExperimentError(
            "saved CSV differs from requested configuration at: "
            + ", ".join(mismatched)
        )


def _validate_complete_grid(
    frame: pd.DataFrame,
    *,
    expected_sources: Sequence[str] | None = None,
) -> pd.DataFrame:
    if tuple(frame.columns) != CSV_COLUMNS:
        raise ExperimentError("saved CSV columns do not match the experiment schema")
    if frame.empty:
        raise ExperimentError("saved CSV has no prompt-target rows")

    validated = frame.copy()
    string_columns = (
        "record_id",
        "selection_strategy",
        "selection_hash",
        "model_name",
        "scheduler_name",
        "evaluation_generation_scientific_config_hash",
        "evaluation_schedule_sha256",
        "evaluation_source",
        "status",
    )
    for column in string_columns:
        validated[column] = validated[column].fillna("").astype(str)
    numeric_columns = (
        "num_inference_steps",
        "step_index",
        "timestep",
        "alpha_t",
        "sigma_t",
        "snr_t",
        "latent_dimension",
        "num_loss_seeds",
        "loss_seed",
        "num_generation_seeds",
        "conditional_loss",
        "normalized_loss_mse",
        "normalized_loss_rmse",
        "recovery_mse",
        "recovery_rmse",
        "mean_target_sscd",
    )
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    inference_steps = validated["num_inference_steps"].dropna().unique()
    if len(inference_steps) != 1 or int(inference_steps[0]) != inference_steps[0]:
        raise ExperimentError("saved CSV has inconsistent inference-step counts")
    step_count = int(inference_steps[0])
    if step_count <= 0:
        raise ExperimentError("saved CSV inference-step count must be positive")
    requested = (
        tuple(str(source) for source in expected_sources)
        if expected_sources is not None
        else tuple(
            source
            for source in EVALUATION_SOURCES
            if source in set(validated["evaluation_source"])
        )
    )
    if not requested or any(source not in EVALUATION_SOURCES for source in requested):
        raise ExperimentError("saved CSV has invalid evaluation sources")
    if set(validated["evaluation_source"]) != set(requested):
        raise ExperimentError("saved CSV evaluation-source grid is incomplete")
    if validated["record_id"].eq("").any():
        raise ExperimentError("saved CSV has an empty record_id")
    key_columns = ["record_id", "step_index", "evaluation_source"]
    if validated.duplicated(key_columns).any():
        raise ExperimentError("saved CSV repeats a prompt-timestep-source row")

    expected_steps = set(range(step_count))
    for (record_id, source), group in validated.groupby(
        ["record_id", "evaluation_source"], sort=False
    ):
        observed_steps = set(group["step_index"].dropna().astype(int))
        if len(group) != step_count or observed_steps != expected_steps:
            raise ExperimentError(
                "saved CSV has an incomplete timestep grid for "
                f"{record_id}/{source}; recompute the experiment"
            )
    record_sources = validated.groupby("record_id", sort=False)[
        "evaluation_source"
    ].agg(lambda values: set(values))
    if any(value != set(requested) for value in record_sources):
        raise ExperimentError("saved CSV has an incomplete source grid")

    schedule = (
        validated[["step_index", "timestep", "alpha_t", "sigma_t", "snr_t"]]
        .drop_duplicates()
        .sort_values("step_index", kind="stable")
    )
    if len(schedule) != step_count or schedule["step_index"].tolist() != list(
        range(step_count)
    ):
        raise ExperimentError("saved CSV does not define one schedule row per timestep")
    if not np.all(np.diff(schedule["timestep"].to_numpy(dtype=float)) < 0.0):
        raise ExperimentError("saved CSV timesteps are not strictly descending")
    coefficients = schedule[["alpha_t", "sigma_t", "snr_t"]].to_numpy(dtype=float)
    if not np.isfinite(coefficients).all() or np.any(coefficients <= 0.0):
        raise ExperimentError("saved CSV schedule coefficients are invalid")
    expected_snr = coefficients[:, 0] ** 2 / coefficients[:, 1] ** 2
    if not np.allclose(coefficients[:, 2], expected_snr, rtol=1e-9, atol=1e-12):
        raise ExperimentError("saved CSV alpha_t^2/sigma_t^2 values are inconsistent")
    if not validated["selection_hash"].map(_is_sha256).all():
        raise ExperimentError("saved CSV selection hash is invalid")
    if not validated["evaluation_schedule_sha256"].map(_is_sha256).all():
        raise ExperimentError("saved CSV schedule hash is invalid")
    if (
        not validated["evaluation_generation_scientific_config_hash"]
        .map(_is_sha256)
        .all()
    ):
        raise ExperimentError("saved CSV generation hash is invalid")

    gaussian = validated["evaluation_source"].eq(GAUSSIAN_SOURCE)
    trajectory = validated["evaluation_source"].eq(TRAJECTORY_SOURCE)
    trajectory_hash = validated["trajectory_sha256"].fillna("").astype(str)
    if bool((gaussian & trajectory_hash.ne("")).any()):
        raise ExperimentError(
            "Gaussian rows must not claim cached-trajectory provenance"
        )
    successful = validated["status"].eq("ok")
    if bool((trajectory & successful & ~trajectory_hash.map(_is_sha256)).any()):
        raise ExperimentError("trajectory rows have invalid trajectory provenance")

    if set(requested) == set(EVALUATION_SOURCES):
        shared_metrics = (
            "conditional_loss",
            "normalized_loss_mse",
            "normalized_loss_rmse",
            "mean_target_sscd",
        )
        ordered = validated.sort_values(
            ["record_id", "step_index", "evaluation_source"], kind="stable"
        )
        for metric in shared_metrics:
            comparison = ordered.pivot(
                index=["record_id", "step_index"],
                columns="evaluation_source",
                values=metric,
            )
            if not np.allclose(
                comparison[GAUSSIAN_SOURCE].to_numpy(dtype=float),
                comparison[TRAJECTORY_SOURCE].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise ExperimentError(
                    f"saved CSV does not reuse {metric} across evaluation sources"
                )
    return validated


def plot_saved_results(
    csv_path: str | Path,
    output_directory: str | Path,
    *,
    evaluation_source: str | None = None,
    expected_configuration: Mapping[str, object] | None = None,
) -> None:
    """Reload selected measurements for the initial scatter and trajectory overlay."""

    source = Path(csv_path)
    frame = pd.read_csv(source)
    expected_sources = (
        _requested_sources(evaluation_source) if evaluation_source is not None else None
    )
    validated = _validate_complete_grid(frame, expected_sources=expected_sources)
    if expected_configuration is not None:
        _validate_csv_configuration(validated, expected_configuration)
    if not validated["status"].eq("ok").all():
        failed = int((~validated["status"].eq("ok")).sum())
        raise ExperimentError(
            f"saved CSV contains {failed} failed observations; refusing to plot "
            "an incomplete Theorem 1 trend"
        )

    requested = expected_sources or tuple(
        source_name
        for source_name in EVALUATION_SOURCES
        if source_name in set(validated["evaluation_source"])
    )
    initial = (
        validated.loc[
            validated["evaluation_source"].eq(GAUSSIAN_SOURCE)
            & validated["step_index"].eq(0)
        ].copy()
        if GAUSSIAN_SOURCE in requested
        else None
    )
    metric_columns = (
        "snr_t",
        "normalized_loss_rmse",
        "recovery_rmse",
        "mean_target_sscd",
    )
    for source_name in requested:
        selected = validated.loc[validated["evaluation_source"].eq(source_name)].copy()
        # A [0, 1] colorbar does not make SSCD a probability. Keep valid
        # negative cosine scores; Normalize clips only their displayed colors.
        valid_domains = {
            "snr_t": selected["snr_t"].gt(0.0),
            "normalized_loss_rmse": selected["normalized_loss_rmse"].ge(0.0),
            "recovery_rmse": selected["recovery_rmse"].ge(0.0),
            "mean_target_sscd": selected["mean_target_sscd"].between(
                -1.0 - SSCD_SCORE_RANGE_TOLERANCE,
                1.0 + SSCD_SCORE_RANGE_TOLERANCE,
                inclusive="both",
            ),
        }
        invalid_details = []
        for column in metric_columns:
            invalid = ~(
                np.isfinite(selected[column].to_numpy(dtype=float))
                & valid_domains[column]
            )
            if bool(invalid.any()):
                first = selected.loc[invalid].iloc[0]
                invalid_details.append(
                    f"{column}: {int(invalid.sum())} invalid rows "
                    f"(record_id={first['record_id']}, "
                    f"timestep={first['timestep']}, value={first[column]!r})"
                )
        if invalid_details:
            raise ExperimentError(
                f"saved CSV contains invalid {source_name} plotting measurements: "
                + "; ".join(invalid_details)
            )
        with matplotlib.rc_context(PLOT_STYLE):
            if source_name == GAUSSIAN_SOURCE:
                assert initial is not None
                _render_initial_loss_recovery(
                    valid=initial,
                    destinations=_figure_paths(output_directory, GAUSSIAN_SOURCE),
                )
            else:
                _render_pair_figure(
                    valid=selected,
                    evaluation_source=source_name,
                    destinations=_figure_paths(output_directory, source_name),
                    initial=initial,
                )
    _remove_noise_sweep_outputs(output_directory)


@dataclass(frozen=True, slots=True)
class _MeasurementCache:
    directory: Path
    configuration: Mapping[str, Any]
    configuration_hash: str
    pair_fingerprints: Mapping[str, str | None]


def _default_log_directory(
    contract: GenerationContract,
    selection_hash: str,
    num_loss_seeds: int,
    loss_seed: int,
    evaluation_source: str,
) -> Path:
    if not _is_sha256(selection_hash):
        raise ExperimentError("selection hash must be a lowercase SHA-256 digest")
    _requested_sources(evaluation_source)
    return (
        contract.paths.run_directory
        / EXPERIMENT_DIRECTORY
        / selection_hash
        / f"loss_S{loss_seed}_N{num_loss_seeds}"
        / f"evaluation_{evaluation_source}"
    )


def _pair_cache_fingerprint(
    contract: GenerationContract, metadata: Mapping[str, object]
) -> str:
    """Check completed-cache provenance without reading trajectory tensors."""
    index = safe_index(metadata.get("original_index"))
    validation = validate_generation_record(
        contract.paths,
        index,
        expected_scientific_hash=contract.scientific_hash,
        expected_record_identity=_pair_identity(metadata),
        load_tensors=False,
        tensor_names=("target_latent", "latent", "noise_prediction"),
        require_preview=False,
        verify_file_hashes=False,
    )
    if not validation.valid or validation.metadata is None:
        raise ExperimentError(
            "measurement source is invalid: " + "; ".join(validation.errors)
        )
    if contract.sscd_error is not None:
        raise ExperimentError(contract.sscd_error)
    marker = validation.metadata
    sscd_marker_path = contract.sscd_paths.marker_path(index)
    if sscd_marker_path.is_symlink() or not sscd_marker_path.is_file():
        raise ExperimentError("measurement SSCD marker is missing or unsafe")
    sscd_marker = read_json(sscd_marker_path)
    if (
        marker.get("seeds") != list(contract.seeds)
        or sscd_marker.get("sscd_configuration_hash")
        != contract.sscd_configuration_hash
        or sscd_marker.get("target_image_sha256") != metadata.get("target_image_sha256")
        or not _is_sha256(sscd_marker.get("score_sha256"))
    ):
        raise ExperimentError("measurement source identity differs")
    stats = {}
    for name, path in (
        ("target", contract.paths.target_latent_path(index)),
        ("trajectory", contract.paths.latent_path(index)),
        ("predictions", contract.paths.noise_prediction_path(index)),
        ("sscd", contract.sscd_paths.score_path(index)),
    ):
        if path.is_symlink() or not path.is_file():
            raise ExperimentError(f"measurement source is missing or unsafe: {path}")
        stat = path.stat()
        stats[name] = [stat.st_size, stat.st_mtime_ns]
    return canonical_hash(
        {
            "pair": _pair_identity(metadata),
            "tensor_file_sha256": marker["tensor_file_sha256"],
            "sscd_marker": {
                key: sscd_marker.get(key)
                for key in (
                    "record_id",
                    "source_row_number",
                    "prompt_raw",
                    "target_image_sha256",
                    "generation_scientific_config_hash",
                    "generation_latent_sha256",
                    "sscd_configuration_hash",
                    "score_sha256",
                    "seeds",
                )
            },
            "source_stats": stats,
        }
    )


def _measurement_configuration(
    contract: GenerationContract, expected_configuration: Mapping[str, object]
) -> dict[str, object]:
    return {
        "measurement_definition": "conditional_loss_and_recovery_v1",
        "loss_stream": _LOSS_STREAM_DOMAIN.decode("ascii"),
        "csv_configuration": dict(expected_configuration),
        "sscd_configuration_hash": contract.sscd_configuration_hash,
        "latent_dimension": math.prod(contract.latent_shape),
        "schedule": [
            {
                "step_index": step,
                "timestep": value.timestep,
                "alpha_t": value.alpha_t,
                "sigma_t": value.sigma_t,
                "snr_t": value.snr_t,
            }
            for step, value in enumerate(contract.parameters)
        ],
    }


def _cache_selection_metadata(
    selection: TargetPairSelection,
) -> dict[str, dict[str, object]]:
    """Use frozen, validated target hashes without opening training images."""
    return {
        str(row["original_index"]): {
            "original_index": str(row["original_index"]),
            "record_id": str(row["record_id"]),
            "source_row_number": int(row["source_row_number"]),
            "prompt_raw": str(row["prompt"]),
            "target_image_sha256": str(row["target_image_sha256"]),
        }
        for row in selection.prompt_frame.to_dict("records")
    }


def _prepare_measurement_cache(
    contract: GenerationContract,
    expected_configuration: Mapping[str, object],
    entries: Sequence[tuple[int, Mapping[str, object]]],
    *,
    overwrite: bool,
) -> tuple[_MeasurementCache, bool]:
    sources = tuple(expected_configuration["evaluation_sources"])
    source = "both" if len(sources) == 2 else str(sources[0])
    directory = _default_log_directory(
        contract,
        str(expected_configuration["selection_hash"]),
        int(expected_configuration["num_loss_seeds"]),
        int(expected_configuration["loss_seed"]),
        source,
    )
    for path in (directory, directory / "record"):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ExperimentError(f"unsafe measurement cache directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    configuration = _measurement_configuration(contract, expected_configuration)
    configuration_hash = canonical_hash(configuration)
    config_path = directory / "run_config.json"
    new_cache = not config_path.exists()
    if config_path.is_symlink() or (config_path.exists() and not config_path.is_file()):
        raise ExperimentError(f"unsafe measurement cache configuration: {config_path}")
    if not new_cache and not overwrite:
        stored = read_json(config_path)
        if (
            stored.get("configuration_hash") != configuration_hash
            or canonical_hash(stored.get("configuration")) != configuration_hash
        ):
            raise ExperimentError(
                f"Theorem 1 measurement cache configuration differs: {directory}; "
                "pass --overwrite to recompute"
            )
    atomic_write_json(
        config_path,
        {
            "configuration_hash": configuration_hash,
            "configuration": configuration,
        },
    )
    fingerprints: dict[str, str | None] = {}
    for _, metadata in entries:
        index = safe_index(metadata.get("original_index"))
        try:
            fingerprints[index] = _pair_cache_fingerprint(contract, metadata)
        except (
            CacheIOError,
            GenerationCacheError,
            ExperimentError,
            OSError,
            ValueError,
        ):
            # Preserve the existing per-prompt error logging in the compute path.
            fingerprints[index] = None
        if overwrite:
            _remove_measurement_file(directory / "record" / f"{index}.json")
    if overwrite:
        _remove_measurement_file(directory / CSV_NAME)
    return _MeasurementCache(
        directory, configuration, configuration_hash, fingerprints
    ), new_cache


def _remove_measurement_file(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ExperimentError(f"unsafe measurement cache file: {path}")
    path.unlink(missing_ok=True)


def _validated_cached_pair_rows(
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
    cache: _MeasurementCache,
) -> list[dict[str, object]]:
    if not rows or any(set(row) != set(CSV_COLUMNS) for row in rows):
        raise ExperimentError("cached measurement row schema differs")
    expected = cache.configuration["csv_configuration"]
    frame = _validate_complete_grid(
        pd.DataFrame(rows, columns=CSV_COLUMNS),
        expected_sources=expected["evaluation_sources"],
    )
    _validate_csv_configuration(frame, expected)
    if (
        not frame["record_id"].eq(str(metadata["record_id"])).all()
        or not frame["status"].eq("ok").all()
        or not frame["latent_dimension"]
        .eq(cache.configuration["latent_dimension"])
        .all()
    ):
        raise ExperimentError(
            "cached measurements are incomplete or belong to a different pair"
        )
    schedule = pd.DataFrame(cache.configuration["schedule"])
    for column in ("timestep", "alpha_t", "sigma_t", "snr_t"):
        values = frame["step_index"].map(schedule.set_index("step_index")[column])
        if not np.allclose(frame[column], values, rtol=1e-12, atol=1e-14):
            raise ExperimentError(
                f"cached measurement {column} differs from the current schedule"
            )
    metrics = (
        "conditional_loss",
        "normalized_loss_mse",
        "normalized_loss_rmse",
        "recovery_mse",
        "recovery_rmse",
        "mean_target_sscd",
    )
    if not np.isfinite(frame[list(metrics)].to_numpy(dtype=float)).all():
        raise ExperimentError("cached measurement metrics are non-finite")
    if (frame[list(metrics[:-1])].to_numpy(dtype=float) < 0).any():
        raise ExperimentError("cached measurement metrics are negative")
    tolerance = SSCD_SCORE_RANGE_TOLERANCE
    if not frame["mean_target_sscd"].between(-1 - tolerance, 1 + tolerance).all():
        raise ExperimentError("cached SSCD metrics are invalid")
    for mse, rmse in (
        ("normalized_loss_mse", "normalized_loss_rmse"),
        ("recovery_mse", "recovery_rmse"),
    ):
        if not np.allclose(frame[mse], frame[rmse] ** 2, rtol=1e-9, atol=1e-12):
            raise ExperimentError("cached measurement MSE/RMSE values are inconsistent")
    normalization = frame["conditional_loss"] / (
        frame["latent_dimension"] * frame["snr_t"]
    )
    if not np.allclose(
        frame["normalized_loss_mse"], normalization, rtol=1e-9, atol=1e-12
    ):
        raise ExperimentError("cached conditional loss normalization is inconsistent")
    frame["error"] = ""
    frame["trajectory_sha256"] = frame["trajectory_sha256"].fillna("")
    frame["generation_seeds"] = str(expected["generation_seeds"])
    return frame.sort_values(
        ["step_index", "evaluation_source"], kind="stable"
    ).to_dict("records")


def _load_pair_checkpoint(
    cache: _MeasurementCache, metadata: Mapping[str, object]
) -> list[dict[str, object]] | None:
    index = safe_index(metadata.get("original_index"))
    fingerprint = cache.pair_fingerprints.get(index)
    path = cache.directory / "record" / f"{index}.json"
    if fingerprint is None or not path.is_file() or path.is_symlink():
        return None
    try:
        saved = read_json(path)
        if (
            saved.get("configuration_hash") != cache.configuration_hash
            or saved.get("source_fingerprint") != fingerprint
            or saved.get("rows_sha256") != canonical_hash(saved.get("rows"))
        ):
            return None
        return _validated_cached_pair_rows(saved["rows"], metadata, cache)
    except (CacheIOError, ExperimentError, KeyError, TypeError, ValueError, OSError):
        return None


def _save_pair_checkpoint(
    cache: _MeasurementCache,
    metadata: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    index = safe_index(metadata.get("original_index"))
    fingerprint = cache.pair_fingerprints.get(index)
    if (
        fingerprint is None
        or not rows
        or any(row.get("status") != "ok" for row in rows)
    ):
        return
    expected = cache.configuration["csv_configuration"]
    stamped = [
        {
            **row,
            "selection_strategy": expected["selection_strategy"],
            "selection_hash": expected["selection_hash"],
        }
        for row in rows
    ]
    # Validate before a success marker can become authoritative.
    stamped = _validated_cached_pair_rows(stamped, metadata, cache)
    path = cache.directory / "record" / f"{index}.json"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ExperimentError(f"unsafe measurement checkpoint: {path}")
    atomic_write_json(
        path,
        {
            "configuration_hash": cache.configuration_hash,
            "source_fingerprint": fingerprint,
            "rows_sha256": canonical_hash(stamped),
            "rows": stamped,
        },
    )


def _adopt_existing_measurements(
    csv_path: Path,
    cache: _MeasurementCache,
    contract: GenerationContract,
    entries: Sequence[tuple[int, Mapping[str, object]]],
) -> int:
    """Bootstrap only compatible current-schema results, never an older CSV schema."""
    if csv_path.is_symlink() or not csv_path.is_file():
        return 0
    try:
        frame = pd.read_csv(
            csv_path, keep_default_na=False, float_precision="round_trip"
        )
    except (OSError, ValueError):
        return 0
    if tuple(frame.columns) != CSV_COLUMNS:
        return 0
    adopted = 0
    for _, metadata in entries:
        index = safe_index(metadata.get("original_index"))
        if cache.pair_fingerprints.get(index) is None:
            continue
        rows = frame.loc[frame["record_id"].eq(str(metadata["record_id"]))]
        try:
            validated = _validated_cached_pair_rows(
                rows.to_dict("records"), metadata, cache
            )
            marker = read_json(contract.paths.record_path(index))
            trajectory = rows.loc[rows["evaluation_source"].eq(TRAJECTORY_SOURCE)]
            if not trajectory["trajectory_sha256"].eq(_trajectory_sha256(marker)).all():
                continue
            mean_sscd = _load_mean_target_sscd(
                contract=contract,
                metadata=metadata,
                generation_seeds=contract.seeds,
                generation_marker=marker,
            )
            if not np.allclose(
                rows["mean_target_sscd"].astype(float),
                mean_sscd,
                rtol=1e-10,
                atol=1e-12,
            ):
                continue
            _save_pair_checkpoint(cache, metadata, validated)
            adopted += 1
        except (
            CacheIOError,
            GenerationCacheError,
            ExperimentError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ):
            continue
    return adopted


def _validated_plot_log_rows(
    directory: Path,
    contract: GenerationContract,
    expected_configuration: Mapping[str, object],
    selection: TargetPairSelection,
    max_records: int | None,
) -> pd.DataFrame:
    configuration = _measurement_configuration(contract, expected_configuration)
    configuration_hash = canonical_hash(configuration)
    path = directory / "run_config.json"
    if path.is_symlink() or not path.is_file():
        raise ExperimentError(
            f"measurement cache configuration is missing or unsafe: {path}"
        )
    saved = read_json(path)
    if (
        saved.get("configuration_hash") != configuration_hash
        or canonical_hash(saved.get("configuration")) != configuration_hash
    ):
        raise ExperimentError(
            "measurement cache configuration differs; rerun without --plot"
        )
    frame = pd.read_csv(
        directory / CSV_NAME, keep_default_na=False, float_precision="round_trip"
    )
    validated = _validate_complete_grid(
        frame, expected_sources=expected_configuration["evaluation_sources"]
    )
    _validate_csv_configuration(validated, expected_configuration)
    requested = [
        metadata
        for index, metadata in _cache_selection_metadata(selection).items()
        if index in selection.included_indices
    ]
    if max_records is not None:
        requested = requested[:max_records]
    if set(validated["record_id"]) != {str(row["record_id"]) for row in requested}:
        raise ExperimentError(
            "measurement log prompt coverage differs; rerun without --plot"
        )
    try:
        fingerprints = {
            str(metadata["original_index"]): _pair_cache_fingerprint(contract, metadata)
            for metadata in requested
        }
    except (
        CacheIOError,
        GenerationCacheError,
        ExperimentError,
        OSError,
        ValueError,
    ) as error:
        raise ExperimentError(
            "measurement sources changed; rerun without --plot"
        ) from error
    cache = _MeasurementCache(
        directory, configuration, configuration_hash, fingerprints
    )
    for metadata in requested:
        checkpoint = _load_pair_checkpoint(cache, metadata)
        if checkpoint is None:
            raise ExperimentError(
                "measurement checkpoint is missing or stale; rerun without --plot"
            )
        rows = validated.loc[validated["record_id"].eq(str(metadata["record_id"]))]
        normalized = _validated_cached_pair_rows(
            rows.to_dict("records"), metadata, cache
        )
        if canonical_hash(normalized) != canonical_hash(checkpoint):
            raise ExperimentError(
                "measurement aggregate differs from checkpoints; rerun without --plot"
            )
    return frame


def _cell_key(
    pair_position: int,
    step_index: int,
    source_index: int,
    *,
    num_inference_steps: int,
    num_sources: int,
) -> int:
    return (
        pair_position * num_inference_steps * num_sources
        + step_index * num_sources
        + source_index
    )


def _error_text(error: BaseException) -> str:
    return " ".join(str(error).splitlines()).strip() or type(error).__name__


def _run_pair_shard_unisolated(
    *,
    project_root: Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_loss_seeds: int,
    loss_seed: int,
    generation_seeds: tuple[int, ...],
    evaluation_source: str,
    sample_batch_size: int,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    device: torch.device,
    worker_index: int,
    worker_count: int,
    dataset: Any | None = None,
    contract: GenerationContract | None = None,
    progress_factory: Callable[..., Any] = tqdm,
    measurement_cache: _MeasurementCache | None = None,
) -> _PairShardResult:
    """Process a complete pair-by-timestep-by-source shard on one device."""

    configure_worker_cpu_threads(worker_count)
    sources = _requested_sources(evaluation_source)
    spec = get_model_spec(model_name)
    active_dataset = dataset
    if active_dataset is None:
        active_dataset = WebsterDataset(
            project_root,
            spec.dataset_model,
            recovered_only=True,
            defer_image_validation=True,
        )
    active_contract = contract
    if active_contract is None:
        active_contract = _load_generation_contract(
            project_root,
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            len(generation_seeds),
        )
    selected_seeds = tuple(int(seed) for seed in generation_seeds)
    if selected_seeds != tuple(active_contract.seeds):
        raise ExperimentError(
            "worker generation seeds differ from the experiment generation cache"
        )
    if len(active_contract.parameters) != num_inference_steps:
        raise ExperimentError(
            "worker schedule does not contain all requested timesteps"
        )

    concrete_device = torch.device(device)
    if concrete_device.type == "cuda":
        if concrete_device.index is None:
            raise ExperimentError("worker CUDA devices must have a concrete index")
        torch.cuda.set_device(concrete_device)
    runtime = select_runtime(concrete_device, warn_without_cuda=False)
    components = load_model_components(
        spec,
        runtime=runtime,
        revision_resolver=_revision_resolver(active_contract),
    )
    scheduler = _validate_active_scheduler(
        components, active_contract, scheduler_name, num_inference_steps
    )
    preflight_model_components(
        components,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
    )
    _validate_loaded_components(components, active_contract)
    _offload_unused_vae(components)
    components = compile_loaded_unet(components)

    gaussian_latents = None
    if GAUSSIAN_SOURCE in sources:
        gaussian_latents = _make_gaussian_recovery_samples(
            selected_seeds,
            active_contract.latent_shape,
        )
    loss_noise = _make_loss_noise(
        num_loss_seeds=num_loss_seeds,
        latent_shape=active_contract.latent_shape,
        loss_seed=loss_seed,
        forbidden_seeds=selected_seeds,
    )
    latent_dimension = math.prod(active_contract.latent_shape)
    cells_per_pair = num_inference_steps * len(sources)

    progress_arguments: dict[str, object] = {
        "total": len(entries) * cells_per_pair,
        "desc": PROGRESS_DESCRIPTION,
        "unit": "observation",
        "dynamic_ncols": True,
        "leave": True,
    }
    if worker_count > 1:
        progress_arguments["desc"] = (
            f"{PROGRESS_DESCRIPTION} [{concrete_device} "
            f"shard {worker_index + 1}/{worker_count}]"
        )
        progress_arguments["position"] = worker_index
    progress = progress_factory(**progress_arguments)
    indexed_rows: list[tuple[int, dict[str, object]]] = []
    failed = 0
    completed = 0
    try:
        for pair_position, metadata in entries:
            pair_row_start = len(indexed_rows)
            record_id = str(metadata.get("record_id", f"pair-{pair_position}"))
            target_latent: torch.Tensor | None = None
            condition: torch.Tensor | None = None
            mean_sscd = math.nan
            trajectory_latents: torch.Tensor | None = None
            trajectory_predictions: torch.Tensor | None = None
            trajectory_hash = ""
            trajectory_error: Exception | None = None
            setup_error: Exception | None = None
            try:
                progress.set_postfix(
                    record_id=record_id,
                    phase="loading-pair",
                    completed=completed,
                    failed=failed,
                )
                item = active_dataset[pair_position]
                if not isinstance(item, Mapping):
                    raise ExperimentError("Webster dataset item is not a mapping")
                _validate_item_identity(item, metadata, spec.dataset_model)
                pair_metadata = dict(metadata)
                pair_metadata["target_image_sha256"] = item["target_image_sha256"]
                generation_marker = _validate_generation_pair(
                    active_contract, pair_metadata
                )
                target_latent = _load_target_latent(
                    metadata=pair_metadata,
                    contract=active_contract,
                    generation_marker=generation_marker,
                )
                condition = encode_prompt_condition(
                    str(item["prompt"]),
                    components.tokenizer,
                    components.text_encoder,
                    components.device,
                    components.inference_dtype,
                )
                mean_sscd = _load_mean_target_sscd(
                    contract=active_contract,
                    metadata=pair_metadata,
                    generation_seeds=selected_seeds,
                    generation_marker=generation_marker,
                )
                if TRAJECTORY_SOURCE in sources:
                    try:
                        (
                            trajectory_latents,
                            trajectory_predictions,
                            trajectory_hash,
                        ) = _load_cached_conditional_trajectory(
                            metadata=pair_metadata,
                            contract=active_contract,
                        )
                    except Exception as error:
                        trajectory_error = error
            except Exception as error:
                setup_error = error

            for step_index, parameters in enumerate(active_contract.parameters):
                loss: LossMeasurement | None = None
                loss_error: Exception | None = setup_error
                if setup_error is None:
                    assert target_latent is not None and condition is not None
                    try:
                        progress.set_postfix(
                            record_id=record_id,
                            step=step_index,
                            phase="conditional-loss",
                            completed=completed,
                            failed=failed,
                        )
                        loss = _measure_conditional_loss(
                            target_latent=target_latent,
                            loss_noise=loss_noise,
                            condition=condition,
                            timestep=parameters.timestep,
                            alpha_t=parameters.alpha_t,
                            sigma_t=parameters.sigma_t,
                            snr_t=parameters.snr_t,
                            components=components,
                            scheduler=scheduler,
                            sample_batch_size=sample_batch_size,
                        )
                    except Exception as error:
                        loss_error = error

                for source_index, source_name in enumerate(sources):
                    row = _base_row(
                        record_id=record_id,
                        model_name=model_name,
                        scheduler_name=scheduler_name,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        scientific_hash=active_contract.scientific_hash,
                        schedule_sha256=active_contract.schedule_sha256,
                        evaluation_source=source_name,
                        step_index=step_index,
                        parameters=parameters,
                        latent_dimension=latent_dimension,
                        num_loss_seeds=num_loss_seeds,
                        loss_seed=loss_seed,
                        generation_seeds=selected_seeds,
                        trajectory_sha256=(
                            trajectory_hash if source_name == TRAJECTORY_SOURCE else ""
                        ),
                    )
                    row_error = loss_error
                    recovery: RecoveryMeasurement | None = None
                    if row_error is None:
                        assert loss is not None and target_latent is not None
                        try:
                            progress.set_postfix(
                                record_id=record_id,
                                step=step_index,
                                source=source_name,
                                phase="recovery",
                                completed=completed,
                                failed=failed,
                            )
                            if source_name == GAUSSIAN_SOURCE:
                                assert (
                                    gaussian_latents is not None
                                    and condition is not None
                                )
                                recovery = _measure_recovery(
                                    target_latent=target_latent,
                                    generation_initial_latents=gaussian_latents,
                                    condition=condition,
                                    timestep=parameters.timestep,
                                    alpha_t=parameters.alpha_t,
                                    sigma_t=parameters.sigma_t,
                                    components=components,
                                    scheduler=scheduler,
                                    sample_batch_size=sample_batch_size,
                                )
                            else:
                                if trajectory_error is not None:
                                    raise trajectory_error
                                assert trajectory_latents is not None
                                assert trajectory_predictions is not None
                                recovery = _measure_trajectory_recovery(
                                    target_latent=target_latent,
                                    latent_trajectory=trajectory_latents,
                                    conditional_epsilon=trajectory_predictions,
                                    step_index=step_index,
                                    alpha_t=parameters.alpha_t,
                                    sigma_t=parameters.sigma_t,
                                )
                        except Exception as error:
                            row_error = error
                    if loss is not None:
                        row.update(
                            {
                                "conditional_loss": loss.conditional_loss,
                                "normalized_loss_mse": loss.normalized_loss_mse,
                                "normalized_loss_rmse": loss.normalized_loss_rmse,
                                "mean_target_sscd": mean_sscd,
                            }
                        )
                    if row_error is None and recovery is not None:
                        row.update(
                            {
                                "recovery_mse": recovery.recovery_mse,
                                "recovery_rmse": recovery.recovery_rmse,
                                "status": "ok",
                                "error": "",
                            }
                        )
                    else:
                        failed += 1
                        assert row_error is not None
                        row["error"] = _error_text(row_error)
                    key = _cell_key(
                        pair_position,
                        step_index,
                        source_index,
                        num_inference_steps=num_inference_steps,
                        num_sources=len(sources),
                    )
                    indexed_rows.append((key, row))
                    completed += 1
                    progress.set_postfix(
                        record_id=record_id,
                        step=step_index,
                        source=source_name,
                        status=row["status"],
                        completed=completed,
                        failed=failed,
                        refresh=False,
                    )
                    progress.update(1)
            if measurement_cache is not None:
                pair_rows = [row for _, row in indexed_rows[pair_row_start:]]
                if all(row["status"] == "ok" for row in pair_rows):
                    expected_fingerprint = measurement_cache.pair_fingerprints.get(
                        safe_index(metadata.get("original_index"))
                    )
                    if expected_fingerprint is not None and (
                        _pair_cache_fingerprint(active_contract, metadata)
                        != expected_fingerprint
                    ):
                        raise ExperimentError(
                            "Theorem 1 source changed during computation"
                        )
                    _save_pair_checkpoint(measurement_cache, metadata, pair_rows)
    finally:
        progress.close()
    return _PairShardResult(tuple(indexed_rows), failed)


def _failed_pair_shard_result(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_loss_seeds: int,
    loss_seed: int,
    generation_seeds: Sequence[int],
    evaluation_source: str,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    device: torch.device,
    worker_index: int,
    worker_count: int,
    error: Exception,
    contract: GenerationContract,
    progress_factory: Callable[..., Any],
) -> _PairShardResult:
    """Represent a shard-wide failure as its complete requested observation grid."""

    concrete_device = torch.device(device)
    sources = _requested_sources(evaluation_source)
    message = (
        f"Theorem 1 worker on {concrete_device} failed before returning results: "
        f"{_error_text(error)}"
    )
    cells_per_pair = num_inference_steps * len(sources)
    progress_arguments: dict[str, object] = {
        "total": len(entries) * cells_per_pair,
        "desc": PROGRESS_DESCRIPTION,
        "unit": "observation",
        "dynamic_ncols": True,
        "leave": True,
    }
    if worker_count > 1:
        progress_arguments["desc"] = (
            f"{PROGRESS_DESCRIPTION} [{concrete_device} "
            f"shard {worker_index + 1}/{worker_count}]"
        )
        progress_arguments["position"] = worker_index
    progress = progress_factory(**progress_arguments)
    indexed_rows: list[tuple[int, dict[str, object]]] = []
    try:
        for pair_position, metadata in entries:
            record_id = str(metadata.get("record_id", f"pair-{pair_position}"))
            for step_index, parameters in enumerate(contract.parameters):
                for source_index, source_name in enumerate(sources):
                    row = _base_row(
                        record_id=record_id,
                        model_name=model_name,
                        scheduler_name=scheduler_name,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        scientific_hash=contract.scientific_hash,
                        schedule_sha256=contract.schedule_sha256,
                        evaluation_source=source_name,
                        step_index=step_index,
                        parameters=parameters,
                        latent_dimension=math.prod(contract.latent_shape),
                        num_loss_seeds=num_loss_seeds,
                        loss_seed=loss_seed,
                        generation_seeds=generation_seeds,
                    )
                    row["error"] = message
                    key = _cell_key(
                        pair_position,
                        step_index,
                        source_index,
                        num_inference_steps=num_inference_steps,
                        num_sources=len(sources),
                    )
                    indexed_rows.append((key, row))
                    progress.set_postfix(
                        record_id=record_id,
                        step=step_index,
                        source=source_name,
                        phase="worker-failure",
                        status="error",
                        refresh=False,
                    )
                    progress.update(1)
    finally:
        progress.close()
    return _PairShardResult(tuple(indexed_rows), len(indexed_rows))


def _run_pair_shard(
    *,
    project_root: Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_loss_seeds: int,
    loss_seed: int,
    generation_seeds: tuple[int, ...],
    evaluation_source: str,
    sample_batch_size: int,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    device: torch.device,
    worker_index: int,
    worker_count: int,
    dataset: Any | None = None,
    contract: GenerationContract | None = None,
    progress_factory: Callable[..., Any] = tqdm,
    measurement_cache: _MeasurementCache | None = None,
) -> _PairShardResult:
    """Run a whole-pair shard without allowing setup failure to drop rows."""

    active_contract = contract
    try:
        if active_contract is None:
            active_contract = _load_generation_contract(
                project_root,
                model_name,
                scheduler_name,
                guidance_scale,
                num_inference_steps,
                len(generation_seeds),
            )
        return _run_pair_shard_unisolated(
            project_root=project_root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_loss_seeds=num_loss_seeds,
            loss_seed=loss_seed,
            generation_seeds=generation_seeds,
            evaluation_source=evaluation_source,
            sample_batch_size=sample_batch_size,
            entries=entries,
            device=device,
            worker_index=worker_index,
            worker_count=worker_count,
            dataset=dataset,
            contract=active_contract,
            progress_factory=progress_factory,
            measurement_cache=measurement_cache,
        )
    except Exception as error:
        if active_contract is None:
            raise
        return _failed_pair_shard_result(
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_loss_seeds=num_loss_seeds,
            loss_seed=loss_seed,
            generation_seeds=generation_seeds,
            evaluation_source=evaluation_source,
            entries=entries,
            device=device,
            worker_index=worker_index,
            worker_count=worker_count,
            error=error,
            contract=active_contract,
            progress_factory=progress_factory,
        )


def _merge_pair_shard_results(
    results: Sequence[_PairShardResult],
    expected_positions: Sequence[int],
) -> _PairShardResult:
    indexed_rows = tuple(
        indexed_row for result in results for indexed_row in result.indexed_rows
    )
    observed_positions = tuple(sorted(position for position, _ in indexed_rows))
    expected = tuple(sorted(int(position) for position in expected_positions))
    if len(expected) != len(set(expected)) or observed_positions != expected:
        raise ExperimentError(
            "Theorem 1 workers returned duplicate or missing observations"
        )
    return _PairShardResult(
        tuple(sorted(indexed_rows, key=lambda indexed_row: indexed_row[0])),
        sum(result.failed_count for result in results),
    )


def _run_pair_shards(
    *,
    project_root: Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_loss_seeds: int,
    loss_seed: int,
    generation_seeds: tuple[int, ...],
    evaluation_source: str,
    sample_batch_size: int,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    devices: Sequence[torch.device],
    dataset: Any,
    contract: GenerationContract,
    progress_factory: Callable[..., Any],
    measurement_cache: _MeasurementCache | None = None,
) -> _PairShardResult:
    if not entries:
        return _PairShardResult((), 0)
    worker_count = worker_count_for_tasks(devices, len(entries))
    active_devices = tuple(devices[:worker_count])
    shards = tuple(
        round_robin_shard(
            entries,
            worker_index=worker_index,
            worker_count=worker_count,
        )
        for worker_index in range(worker_count)
    )
    common_arguments = {
        "project_root": project_root,
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_loss_seeds": num_loss_seeds,
        "loss_seed": loss_seed,
        "generation_seeds": generation_seeds,
        "evaluation_source": evaluation_source,
        "sample_batch_size": sample_batch_size,
        "worker_count": worker_count,
        "measurement_cache": measurement_cache,
    }
    if worker_count == 1:
        return _run_pair_shard(
            **common_arguments,
            entries=shards[0],
            device=active_devices[0],
            worker_index=0,
            dataset=dataset,
            contract=contract,
            progress_factory=progress_factory,
        )
    if not all(selected.type == "cuda" for selected in active_devices):
        raise ExperimentError("multiple Theorem 1 workers require CUDA devices")

    print(
        "Theorem 1 worker plan: "
        + ", ".join(
            f"{selected}={len(shards[index])} pairs"
            for index, selected in enumerate(active_devices)
        )
        + "; each progress bar and ETA is shard-local; all loss and generation "
        "seeds for one pair stay together."
    )
    context = multiprocessing.get_context("spawn")
    progress_lock = context.RLock()
    tqdm.set_lock(progress_lock)
    with ProcessPoolExecutor(
        max_workers=worker_count - 1,
        mp_context=context,
        initializer=_install_tqdm_lock,
        initargs=(progress_lock,),
    ) as executor:
        futures = [
            (
                worker_index,
                active_devices[worker_index],
                shards[worker_index],
                executor.submit(
                    _run_pair_shard,
                    **common_arguments,
                    entries=shards[worker_index],
                    device=active_devices[worker_index],
                    worker_index=worker_index,
                    contract=contract,
                ),
            )
            for worker_index in range(1, worker_count)
        ]
        local = _run_pair_shard(
            **common_arguments,
            entries=shards[0],
            device=active_devices[0],
            worker_index=0,
            dataset=dataset,
            contract=contract,
            progress_factory=progress_factory,
        )
        results = [local]
        for worker_index, selected, shard, future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                results.append(
                    _failed_pair_shard_result(
                        model_name=model_name,
                        scheduler_name=scheduler_name,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        num_loss_seeds=num_loss_seeds,
                        loss_seed=loss_seed,
                        generation_seeds=generation_seeds,
                        evaluation_source=evaluation_source,
                        entries=shard,
                        device=selected,
                        worker_index=worker_index,
                        worker_count=worker_count,
                        error=error,
                        contract=contract,
                        progress_factory=progress_factory,
                    )
                )
    sources = _requested_sources(evaluation_source)
    expected_cells = tuple(
        _cell_key(
            pair_position,
            step_index,
            source_index,
            num_inference_steps=num_inference_steps,
            num_sources=len(sources),
        )
        for pair_position, _metadata in entries
        for step_index in range(num_inference_steps)
        for source_index in range(len(sources))
    )
    return _merge_pair_shard_results(results, expected_cells)


def run_experiment(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_loss_seeds: int,
    loss_seed: int,
    num_seeds: int,
    selection_strategy: str,
    evaluation_source: str,
    sample_batch_size: int,
    max_records: int | None,
    output_dir: str | Path | None,
    device: str | torch.device,
    progress_factory: Callable[..., Any] = tqdm,
    overwrite: bool = False,
) -> int:
    """Reuse completed prompt checkpoints; compute and log only pending prompts."""

    sources = _requested_sources(evaluation_source)
    if not isinstance(overwrite, bool):
        raise ExperimentError("overwrite must be a boolean")

    selection = _load_frozen_selection(
        ROOT,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        selection_strategy=selection_strategy,
    )
    spec = get_model_spec(model_name)
    dataset = WebsterDataset(
        ROOT,
        spec.dataset_model,
        recovered_only=True,
        defer_image_validation=True,
    )
    metadata_rows = list(dataset.iter_metadata())
    selected_metadata = _cache_selection_metadata(selection)
    metadata_rows = [
        {
            **metadata,
            "target_image_sha256": selected_metadata[str(metadata["original_index"])][
                "target_image_sha256"
            ],
        }
        for metadata in metadata_rows
    ]
    entries = _selected_dataset_entries(metadata_rows, selection, max_records)
    requested_output = (
        _default_output_directory(
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
            selection.selection_strategy,
            selection.sha256,
            evaluation_source,
        )
        if output_dir is None
        else output_dir
    )
    output = _prepare_output_directory(requested_output)
    contract = _load_generation_contract(
        ROOT,
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    expected_configuration = _expected_csv_configuration(
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_loss_seeds=num_loss_seeds,
        loss_seed=loss_seed,
        num_seeds=num_seeds,
        selection_strategy=selection.selection_strategy,
        selection_hash=selection.sha256,
        evaluation_source=evaluation_source,
        scientific_hash=contract.scientific_hash,
        schedule_sha256=contract.schedule_sha256,
    )
    cache, new_cache = _prepare_measurement_cache(
        contract,
        expected_configuration,
        entries,
        overwrite=overwrite,
    )
    if new_cache and not overwrite:
        # Preserve every already computed prompt before --max-records writes a subset.
        bootstrap_entries = _selected_dataset_entries(metadata_rows, selection, None)
        fingerprints = dict(cache.pair_fingerprints)
        for _, metadata in bootstrap_entries:
            index = str(metadata["original_index"])
            if index not in fingerprints:
                try:
                    fingerprints[index] = _pair_cache_fingerprint(contract, metadata)
                except (
                    CacheIOError,
                    GenerationCacheError,
                    ExperimentError,
                    OSError,
                    ValueError,
                ):
                    fingerprints[index] = None
        cache = _MeasurementCache(
            cache.directory, cache.configuration, cache.configuration_hash, fingerprints
        )
        adopted = _adopt_existing_measurements(
            output / CSV_NAME, cache, contract, bootstrap_entries
        )
        if adopted:
            print(
                f"Theorem 1: imported {adopted} completed prompts from the existing CSV."
            )
    cached = {
        position: rows
        for position, metadata in entries
        if (rows := _load_pair_checkpoint(cache, metadata)) is not None
    }
    pending = tuple(entry for entry in entries if entry[0] not in cached)
    print(
        f"Theorem 1 cache: {len(cached)}/{len(entries)} prompts reused; "
        f"{len(pending)} pending. Logs: {cache.directory}"
    )
    computed: dict[int, dict[str, object]] = {}
    if pending:
        try:
            devices = resolve_devices(device)
        except DeviceSelectionError as error:
            raise ExperimentError(str(error)) from error
        result = _run_pair_shards(
            project_root=ROOT,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_loss_seeds=num_loss_seeds,
            loss_seed=loss_seed,
            generation_seeds=contract.seeds,
            evaluation_source=evaluation_source,
            sample_batch_size=sample_batch_size,
            entries=pending,
            devices=devices,
            dataset=dataset,
            contract=contract,
            progress_factory=progress_factory,
            measurement_cache=cache,
        )
        computed = dict(result.indexed_rows)
    rows = []
    for position, metadata in entries:
        # Durable completed prompts survive a later shard/model failure.
        pair_rows = cached.get(position) or _load_pair_checkpoint(cache, metadata)
        if pair_rows is None:
            pair_rows = [
                computed[
                    _cell_key(
                        position,
                        step,
                        source_index,
                        num_inference_steps=num_inference_steps,
                        num_sources=len(sources),
                    )
                ]
                for step in range(num_inference_steps)
                for source_index in range(len(sources))
            ]
            _save_pair_checkpoint(cache, metadata, pair_rows)
        for saved_row in pair_rows:
            rows.append(
                {
                    **saved_row,
                    "selection_strategy": selection.selection_strategy,
                    "selection_hash": selection.sha256,
                }
            )
    failed_count = sum(row["status"] != "ok" for row in rows)
    csv_path = cache.directory / CSV_NAME
    atomic_write_csv(csv_path, rows, CSV_COLUMNS)
    # Keep the current output CSV as a derived, portable companion to the figures.
    atomic_write_csv(output / CSV_NAME, rows, CSV_COLUMNS)
    print(f"Theorem 1 measurements: {csv_path}")
    try:
        plot_saved_results(
            csv_path,
            output,
            evaluation_source=evaluation_source,
            expected_configuration=_expected_csv_configuration(
                model_name=model_name,
                scheduler_name=scheduler_name,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_loss_seeds=num_loss_seeds,
                loss_seed=loss_seed,
                num_seeds=num_seeds,
                selection_strategy=selection.selection_strategy,
                selection_hash=selection.sha256,
                evaluation_source=evaluation_source,
                scientific_hash=contract.scientific_hash,
                schedule_sha256=contract.schedule_sha256,
            ),
        )
    except BaseException:
        _remove_figure_outputs(output)
        raise
    return int(failed_count > 0)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.plot:
        selection = _load_frozen_selection(
            ROOT,
            model_name=arguments.model,
            scheduler_name=arguments.scheduler,
            guidance_scale=arguments.g,
            num_inference_steps=arguments.num_inference_steps,
            num_seeds=arguments.num_seeds,
            selection_strategy=arguments.selection_strategy,
        )
        contract = _load_generation_contract(
            ROOT,
            arguments.model,
            arguments.scheduler,
            arguments.g,
            arguments.num_inference_steps,
            arguments.num_seeds,
        )
        requested_output = (
            _default_output_directory(
                arguments.model,
                arguments.scheduler,
                arguments.g,
                arguments.num_inference_steps,
                arguments.num_seeds,
                selection.selection_strategy,
                selection.sha256,
                arguments.evaluation_source,
            )
            if arguments.output_dir is None
            else arguments.output_dir
        )
        output = _prepare_output_directory(requested_output)
        log_directory = _default_log_directory(
            contract,
            selection.sha256,
            arguments.num_loss_seeds,
            arguments.loss_seed,
            arguments.evaluation_source,
        )
        csv_path = log_directory / CSV_NAME
        if not csv_path.is_file() or csv_path.is_symlink():
            raise ExperimentError(
                f"saved Theorem 1 log CSV is missing or unsafe: {csv_path}; "
                "run without --plot once to resume or import completed measurements"
            )
        expected_configuration = _expected_csv_configuration(
            model_name=arguments.model,
            scheduler_name=arguments.scheduler,
            guidance_scale=arguments.g,
            num_inference_steps=arguments.num_inference_steps,
            num_loss_seeds=arguments.num_loss_seeds,
            loss_seed=arguments.loss_seed,
            num_seeds=arguments.num_seeds,
            selection_strategy=selection.selection_strategy,
            selection_hash=selection.sha256,
            evaluation_source=arguments.evaluation_source,
            scientific_hash=contract.scientific_hash,
            schedule_sha256=contract.schedule_sha256,
        )
        frame = _validated_plot_log_rows(
            log_directory,
            contract,
            expected_configuration,
            selection,
            arguments.max_records,
        )
        plot_saved_results(
            csv_path,
            output,
            evaluation_source=arguments.evaluation_source,
            expected_configuration=expected_configuration,
        )
        atomic_write_csv(output / CSV_NAME, frame.to_dict("records"), CSV_COLUMNS)
        return 0
    return run_experiment(
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_loss_seeds=arguments.num_loss_seeds,
        loss_seed=arguments.loss_seed,
        num_seeds=arguments.num_seeds,
        selection_strategy=arguments.selection_strategy,
        evaluation_source=arguments.evaluation_source,
        sample_batch_size=arguments.sample_batch_size,
        max_records=arguments.max_records,
        output_dir=arguments.output_dir,
        device=arguments.device,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
