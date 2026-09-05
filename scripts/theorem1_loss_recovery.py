#!/usr/bin/env python3
"""Measure pair-specific conditional loss and one-prediction latent recovery."""

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
    atomic_write_csv,
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
from utils.experiments.sscd import (  # noqa: E402
    SCORE_DEFINITION,
    SSCDPaths,
    sscd_configuration_hash,
)
from utils.metrics.sscd import validate_sscd_scores  # noqa: E402
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
FIGURE_FILENAMES = (
    "theorem1_loss_recovery.png",
    "theorem1_loss_recovery.pdf",
)
CSV_COLUMNS = (
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
    schedule_payload: Mapping[str, Any]
    scheduler_config: Mapping[str, Any]
    sscd_configuration: Mapping[str, Any] | None
    sscd_configuration_hash: str | None
    sscd_error: str | None


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
        "--selection-strategy",
        choices=SELECTION_STRATEGIES,
        default=DEFAULT_SELECTION_STRATEGY,
        help=(
            "frozen prompt-selection strategy: GMM posterior, GMM evidence, or "
            "Spearman (default: gmm)"
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
        metavar="K",
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
        "--plot",
        action="store_true",
        help="regenerate the PNG and PDF from the saved CSV without computation",
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
    schedule_payload = safe_torch_load(paths.schedule)
    if not isinstance(schedule_payload, Mapping):
        raise ExperimentError("saved generation schedule must be a mapping")
    saved_timesteps = schedule_payload.get("timesteps")
    if (
        not isinstance(saved_timesteps, torch.Tensor)
        or saved_timesteps.ndim != 1
        or saved_timesteps.numel() != num_inference_steps
    ):
        raise ExperimentError("saved generation timesteps are invalid")
    if _integer_tuple(saved_timesteps.tolist(), "saved timesteps")[0] < 0:
        raise ExperimentError("saved generation timestep is invalid")
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
    terminal: TerminalParameters | None,
    latent_dimension: int | None,
    num_loss_seeds: int,
    loss_seed: int,
    generation_seeds: Sequence[int],
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "timestep": terminal.timestep if terminal is not None else math.nan,
        "alpha_t": terminal.alpha_t if terminal is not None else math.nan,
        "sigma_t": terminal.sigma_t if terminal is not None else math.nan,
        "snr_t": terminal.snr_t if terminal is not None else math.nan,
        "latent_dimension": (
            latent_dimension if latent_dimension is not None else math.nan
        ),
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
    allowed = {CSV_NAME, *FIGURE_FILENAMES}
    unexpected = sorted(
        item.name
        for item in path.iterdir()
        if item.name not in allowed or item.is_symlink() or not item.is_file()
    )
    if unexpected:
        raise ExperimentError(
            "output directory contains unexpected artifacts: " + ", ".join(unexpected)
        )
    return path


def _default_output_directory(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str,
    selection_hash: str,
) -> Path:
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ExperimentError(f"unsupported selection strategy: {selection_strategy!r}")
    if not (
        isinstance(selection_hash, str)
        and len(selection_hash) == 64
        and all(character in "0123456789abcdef" for character in selection_hash)
    ):
        raise ExperimentError("selection hash must be a lowercase SHA-256 digest")
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
        / selection_strategy
        / selection_hash
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


def _figure_paths(output_directory: str | Path) -> tuple[Path, ...]:
    output = Path(output_directory)
    return tuple(output / filename for filename in FIGURE_FILENAMES)


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
            }
            if figure_format == "png":
                save_options["dpi"] = 300
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


def _render_pair_figure(
    *,
    valid: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
    colors: np.ndarray,
    destinations: Sequence[Path],
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
    }


def _validate_csv_configuration(
    frame: pd.DataFrame,
    expected: Mapping[str, object],
) -> None:
    if frame.empty:
        raise ExperimentError("saved CSV has no prompt-target rows")
    mismatched: list[str] = []
    for column, expected_value in expected.items():
        values = frame[column]
        if isinstance(expected_value, (int, float)):
            observed = pd.to_numeric(values, errors="coerce")
            matches = observed.notna() & observed.eq(expected_value)
        else:
            matches = values.astype(str).eq(str(expected_value))
        if not bool(matches.all()):
            mismatched.append(column)
    if mismatched:
        raise ExperimentError(
            "saved CSV differs from requested configuration at: "
            + ", ".join(mismatched)
        )


def plot_saved_results(
    csv_path: str | Path,
    output_directory: str | Path,
    *,
    expected_configuration: Mapping[str, object] | None = None,
) -> None:
    """Reload the sole CSV and render the pair-level PNG and PDF figures."""

    source = Path(csv_path)
    destinations = _figure_paths(output_directory)
    frame = pd.read_csv(source)
    if tuple(frame.columns) != CSV_COLUMNS:
        raise ExperimentError("saved CSV columns do not match the experiment schema")
    if frame["record_id"].astype(str).duplicated().any():
        raise ExperimentError("saved CSV contains duplicate prompt-target pairs")
    if expected_configuration is not None:
        _validate_csv_configuration(frame, expected_configuration)

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
            destinations=destinations,
        )


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
    sample_batch_size: int,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    device: torch.device,
    worker_index: int,
    worker_count: int,
    dataset: Any | None = None,
    contract: GenerationContract | None = None,
    progress_factory: Callable[..., Any] = tqdm,
) -> _PairShardResult:
    """Load one model and process complete prompt-target pairs on one device."""

    configure_worker_cpu_threads(worker_count)
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
    terminal = _terminal_parameters(scheduler)
    generation_initial_latents = _reconstruct_generation_initial_latents(
        selected_seeds,
        active_contract.latent_shape,
        active_contract.stored_dtype,
        active_contract.init_noise_sigma,
    )
    loss_noise = _make_loss_noise(
        num_loss_seeds=num_loss_seeds,
        latent_shape=active_contract.latent_shape,
        loss_seed=loss_seed,
        forbidden_seeds=selected_seeds,
    )
    latent_dimension = math.prod(active_contract.latent_shape)

    progress_arguments: dict[str, object] = {
        "total": len(entries),
        "desc": PROGRESS_DESCRIPTION,
        "unit": "pair",
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
    try:
        for position, metadata in entries:
            record_id = str(metadata.get("record_id", f"pair-{position}"))
            progress.set_postfix(
                record_id=record_id,
                phase="loading-target",
                completed=len(indexed_rows),
                failed=failed,
            )
            row = _base_row(
                record_id=record_id,
                model_name=model_name,
                scheduler_name=scheduler_name,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                terminal=terminal,
                latent_dimension=latent_dimension,
                num_loss_seeds=num_loss_seeds,
                loss_seed=loss_seed,
                generation_seeds=selected_seeds,
            )
            try:
                item = active_dataset[position]
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
                prompt = str(item["prompt"])
                progress.set_postfix(
                    record_id=record_id,
                    phase="encoding-prompt",
                    completed=len(indexed_rows),
                    failed=failed,
                )
                condition = encode_prompt_condition(
                    prompt,
                    components.tokenizer,
                    components.text_encoder,
                    components.device,
                    components.inference_dtype,
                )
                progress.set_postfix(
                    record_id=record_id,
                    phase="conditional-loss",
                    completed=len(indexed_rows),
                    failed=failed,
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
                progress.set_postfix(
                    record_id=record_id,
                    phase="recovery",
                    completed=len(indexed_rows),
                    failed=failed,
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
                progress.set_postfix(
                    record_id=record_id,
                    phase="sscd",
                    completed=len(indexed_rows),
                    failed=failed,
                )
                mean_sscd = _load_mean_target_sscd(
                    contract=active_contract,
                    metadata=pair_metadata,
                    generation_seeds=selected_seeds,
                    generation_marker=generation_marker,
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
                indexed_rows.append((position, row))
                progress.set_postfix(
                    record_id=record_id,
                    status=row["status"],
                    completed=len(indexed_rows),
                    failed=failed,
                    refresh=False,
                )
                progress.update(1)
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
    entries: Sequence[tuple[int, Mapping[str, object]]],
    device: torch.device,
    worker_index: int,
    worker_count: int,
    error: Exception,
    contract: GenerationContract | None,
    progress_factory: Callable[..., Any],
) -> _PairShardResult:
    """Represent one shard-wide failure as one explicit row per pair."""

    concrete_device = torch.device(device)
    message = " ".join(str(error).splitlines()).strip() or type(error).__name__
    message = (
        f"Theorem 1 worker on {concrete_device} failed before returning pair results: "
        f"{message}"
    )
    latent_dimension = (
        math.prod(contract.latent_shape) if contract is not None else None
    )
    progress_arguments: dict[str, object] = {
        "total": len(entries),
        "desc": PROGRESS_DESCRIPTION,
        "unit": "pair",
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
        for position, metadata in entries:
            record_id = str(metadata.get("record_id", f"pair-{position}"))
            row = _base_row(
                record_id=record_id,
                model_name=model_name,
                scheduler_name=scheduler_name,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                terminal=None,
                latent_dimension=latent_dimension,
                num_loss_seeds=num_loss_seeds,
                loss_seed=loss_seed,
                generation_seeds=generation_seeds,
            )
            row["error"] = message
            indexed_rows.append((position, row))
            progress.set_postfix(
                record_id=record_id,
                phase="worker-failure",
                status="error",
                completed=len(indexed_rows),
                failed=len(indexed_rows),
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
    sample_batch_size: int,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    device: torch.device,
    worker_index: int,
    worker_count: int,
    dataset: Any | None = None,
    contract: GenerationContract | None = None,
    progress_factory: Callable[..., Any] = tqdm,
) -> _PairShardResult:
    """Run a whole-pair shard without allowing setup failure to drop rows."""

    try:
        return _run_pair_shard_unisolated(
            project_root=project_root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_loss_seeds=num_loss_seeds,
            loss_seed=loss_seed,
            generation_seeds=generation_seeds,
            sample_batch_size=sample_batch_size,
            entries=entries,
            device=device,
            worker_index=worker_index,
            worker_count=worker_count,
            dataset=dataset,
            contract=contract,
            progress_factory=progress_factory,
        )
    except Exception as error:
        return _failed_pair_shard_result(
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_loss_seeds=num_loss_seeds,
            loss_seed=loss_seed,
            generation_seeds=generation_seeds,
            entries=entries,
            device=device,
            worker_index=worker_index,
            worker_count=worker_count,
            error=error,
            contract=contract,
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
            "Theorem 1 workers returned duplicate or missing prompt-target pairs"
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
    sample_batch_size: int,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    devices: Sequence[torch.device],
    dataset: Any,
    contract: GenerationContract,
    progress_factory: Callable[..., Any],
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
        "sample_batch_size": sample_batch_size,
        "worker_count": worker_count,
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
                        entries=shard,
                        device=selected,
                        worker_index=worker_index,
                        worker_count=worker_count,
                        error=error,
                        contract=contract,
                        progress_factory=progress_factory,
                    )
                )
    return _merge_pair_shard_results(
        results,
        tuple(position for position, _metadata in entries),
    )


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
    sample_batch_size: int,
    max_records: int | None,
    output_dir: str | Path | None,
    device: str | torch.device,
    progress_factory: Callable[..., Any] = tqdm,
) -> int:
    """Run the pair-level experiment for every selected Webster pair."""

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
    selected_seeds = contract.seeds
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
        generation_seeds=selected_seeds,
        sample_batch_size=sample_batch_size,
        entries=entries,
        devices=devices,
        dataset=dataset,
        contract=contract,
        progress_factory=progress_factory,
    )
    rows = []
    for _, saved_row in result.indexed_rows:
        row = dict(saved_row)
        row["selection_strategy"] = selection.selection_strategy
        row["selection_hash"] = selection.sha256
        rows.append(row)

    csv_path = output / CSV_NAME
    atomic_write_csv(csv_path, rows, CSV_COLUMNS)
    try:
        plot_saved_results(
            csv_path,
            output,
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
            ),
        )
    except BaseException:
        _remove_figure_outputs(output)
        raise
    return int(result.failed_count > 0)


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
        requested_output = (
            _default_output_directory(
                arguments.model,
                arguments.scheduler,
                arguments.g,
                arguments.num_inference_steps,
                arguments.num_seeds,
                selection.selection_strategy,
                selection.sha256,
            )
            if arguments.output_dir is None
            else arguments.output_dir
        )
        output = _prepare_output_directory(requested_output)
        csv_path = output / CSV_NAME
        if not csv_path.is_file():
            raise ExperimentError(f"saved CSV is missing: {csv_path}")
        plot_saved_results(
            csv_path,
            output,
            expected_configuration=_expected_csv_configuration(
                model_name=arguments.model,
                scheduler_name=arguments.scheduler,
                guidance_scale=arguments.g,
                num_inference_steps=arguments.num_inference_steps,
                num_loss_seeds=arguments.num_loss_seeds,
                loss_seed=arguments.loss_seed,
                num_seeds=arguments.num_seeds,
                selection_strategy=selection.selection_strategy,
                selection_hash=selection.sha256,
            ),
        )
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
        sample_batch_size=arguments.sample_batch_size,
        max_records=arguments.max_records,
        output_dir=arguments.output_dir,
        device=arguments.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
