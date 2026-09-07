#!/usr/bin/env python3
"""Evaluate Corollary 3 from frozen, same-seed generation caches."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common.cli import (  # noqa: E402
    SCHEDULER_CHOICES,
    device_argument,
    finite_float,
    generation_run_name,
    positive_integer,
    validate_seed_block,
)
from utils.common.io import (  # noqa: E402
    CacheIOError,
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
from utils.experiments.cache import (  # noqa: E402
    GENERATION_SELECTION_POLICY,
    CompletedGenerationRecord,
    GenerationCacheError,
    GenerationPaths,
    generation_paths,
    list_completed_records,
    require_generation_run,
    safe_index,
    validate_generation_record,
)
from utils.experiments.sscd import (  # noqa: E402
    SCORE_DEFINITION,
    SSCDPaths,
    sscd_configuration_hash,
)
from utils.experiments.unconditional_baseline import (  # noqa: E402
    DEFAULT_NUM_BASELINE_SEEDS,
    UnconditionalBaselineArtifact,
    UnconditionalBaselineError,
    load_unconditional_baseline,
)
from utils.metrics.sscd import SSCDMetricError, validate_sscd_scores  # noqa: E402
from utils.models.devices import (  # noqa: E402
    DeviceSelectionError,
    configure_worker_cpu_threads,
    resolve_devices,
    round_robin_shard,
    worker_count_for_tasks,
)
from utils.models.latent import (  # noqa: E402
    TARGET_LATENT_DEFINITION,
    target_preprocessing_policy,
)
from utils.models.registry import get_model_spec, model_names  # noqa: E402
from utils.models.sampling import make_initial_noise  # noqa: E402


CSV_NAME = "corollary3_cfg_amplification.csv"
FIGURE_FILENAMES = (
    "corollary3_cfg_amplification.png",
    "corollary3_cfg_amplification.pdf",
)
CSV_COLUMNS = tuple(
    "record_id generation_seed timestep alpha_t sigma_t snr_t guidance_scale "
    "centering_mode baseline_generation_scientific_config_hash "
    "baseline_schedule_sha256 baseline_mu_hat_sha256 num_baseline_seeds "
    "fitted_guidance_scale residual_rmse guided_target_rmse "
    "conditional_recovery_rmse unconditional_rmse target_sscd status error".split()
)

REQUIRED_SCHEDULER = "ddim"
REQUIRED_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_SEEDS = 20
EXPERIMENT_DIRECTORY = "corollary3_cfg_amplification"
PROGRESS_DESCRIPTION = "Corollary 3 cached prompt-seed observations"
CENTERING_ZERO = "zero"
CENTERING_MU_HAT = "mu_hat"
CENTERING_MODES = (CENTERING_ZERO, CENTERING_MU_HAT)

FIGURE_SIZE = (4.0, 4.0)
TEXT_FONT_SIZE = 15
AXIS_NUMBER_FONT_SIZE = 12
SCATTER_ALPHA = 0.68
COLORBAR_ALPHA = 1.0
COLORBAR_LABEL = "SSCD"
SSCD_COLOR_RANGE = (0.0, 1.0)
FIGURE_PAD_INCHES = 0.05
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
}

CFG_RTOL = 1e-12
CFG_ATOL = 1e-12
PROJECTION_RTOL = 1e-10
PROJECTION_ATOL = 1e-12


class ExperimentError(RuntimeError):
    """Report invalid provenance, cache tensors, measurements, or saved output."""


@dataclass(frozen=True, slots=True)
class GenerationContract:
    """Validated generation, schedule, target-latent, and SSCD provenance."""

    paths: GenerationPaths
    sscd_paths: SSCDPaths
    science: Mapping[str, Any]
    scientific_hash: str
    schedule_sha256: str
    seeds: tuple[int, ...]
    latent_shape: tuple[int, int, int]
    latent_dimension: int
    stored_dtype: torch.dtype
    init_noise_sigma: float
    timestep: int
    alpha_t: float
    sigma_t: float
    snr_t: float
    records_by_index: Mapping[str, CompletedGenerationRecord]
    sscd_configuration: Mapping[str, Any]
    sscd_configuration_hash: str


@dataclass(frozen=True, slots=True)
class CenteringContract:
    """Active center and mode-specific reusable-baseline provenance."""

    mode: str
    value: torch.Tensor | None
    baseline_generation_scientific_config_hash: str
    baseline_schedule_sha256: str
    baseline_mu_hat_sha256: str
    num_baseline_seeds: int


@dataclass(frozen=True, slots=True)
class _PromptMeasurement:
    """Compact process-safe metrics for all seeds of one prompt."""

    position: int
    original_index: str
    values: np.ndarray | None
    error: str


_WORKER_CONTRACT: GenerationContract | None = None
_WORKER_DEVICE: torch.device | None = None
_WORKER_INITIAL_LATENTS: torch.Tensor | None = None
_WORKER_CENTER: torch.Tensor | None = None
_WORKER_CENTERING_MODE: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Corollary 3 at the cached initial DDIM timestep for every "
            "frozen-selected prompt and generation seed. The default uses exact "
            "float64 mu = 0; --use-mu opts into the saved shared baseline."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model", choices=model_names(), default="sdv1")
    parser.add_argument(
        "--selection-strategy",
        choices=SELECTION_STRATEGIES,
        default=DEFAULT_SELECTION_STRATEGY,
        help="frozen prompt-selection strategy (default: spearman)",
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
        "--N",
        dest="num_seeds",
        type=positive_integer,
        default=DEFAULT_NUM_SEEDS,
        metavar="SEEDS",
        help="experiment-cache seeds 0..N-1 (default: 20)",
    )
    parser.add_argument(
        "--num-baseline-seeds",
        type=positive_integer,
        default=DEFAULT_NUM_BASELINE_SEEDS,
        metavar="SEEDS",
        help=(
            "dedicated baseline Gaussian seeds N..N+B-1 used only with --use-mu "
            "(default: 1000)"
        ),
    )
    parser.add_argument(
        "--use-mu",
        action="store_true",
        help=(
            "center on the saved shared mu_hat baseline; absent, use exact float64 "
            "mu = 0 without reading a baseline artifact"
        ),
    )
    parser.add_argument(
        "--device",
        type=device_argument,
        default="auto",
        help=(
            "auto uses every CUDA device visible to PyTorch; cpu, mps, cuda, "
            "or cuda:N selects one device (default: auto)"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "validate and replot the saved CSV without reading cached trajectory, "
            "prediction, target-latent, or baseline tensors"
        ),
    )
    return parser


def _validate_scientific_request(scheduler_name: str, guidance_scale: float) -> None:
    if scheduler_name != REQUIRED_SCHEDULER:
        raise ExperimentError("Corollary 3 requires --scheduler ddim")
    if not math.isclose(
        float(guidance_scale),
        REQUIRED_GUIDANCE_SCALE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ExperimentError("Corollary 3 requires --g 7.5")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExperimentError(f"{label} must be an integer sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ExperimentError(f"{label} must be an integer sequence")
        try:
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as error:
            raise ExperimentError(f"{label} must be an integer sequence") from error
        if integer != item:
            raise ExperimentError(f"{label} must be an integer sequence")
        result.append(integer)
    return tuple(result)


def _dtype_from_name(value: object) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[str(value)]
    except KeyError as error:
        raise ExperimentError(
            f"unsupported generation tensor dtype: {value!r}"
        ) from error


def _normalize_scheduler_config(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentError("scheduler configuration must be a mapping")
    result = {str(key): item for key, item in value.items()}
    defaults = result.get("_use_default_values")
    if isinstance(defaults, Sequence) and not isinstance(defaults, (str, bytes)):
        result["_use_default_values"] = sorted(str(item) for item in defaults)
    return result


def _num_train_timesteps(config: Mapping[str, object]) -> int:
    value = config.get("num_train_timesteps")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExperimentError(
            "scheduler num_train_timesteps must be a positive integer"
        )
    return value


def _schedule_file_sha256(paths: GenerationPaths) -> str:
    if not paths.schedule.is_file() or paths.schedule.is_symlink():
        raise ExperimentError("saved generation schedule is missing or unsafe")
    try:
        digest = file_sha256(paths.schedule)
    except OSError as error:
        raise ExperimentError(
            f"cannot hash saved generation schedule: {error}"
        ) from error
    if not _is_sha256(digest):
        raise ExperimentError("saved generation schedule hash is invalid")
    return digest


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


def _included_prompt_rows(
    selection: TargetPairSelection,
) -> tuple[dict[str, object], ...]:
    required = {
        "original_index",
        "record_id",
        "source_row_number",
        "prompt",
        "target_image_sha256",
        "include_prompt",
    }
    missing = sorted(required - set(selection.prompt_frame.columns))
    if missing:
        raise ExperimentError(
            "frozen prompt selection is missing: " + ", ".join(missing)
        )
    included = selection.prompt_frame.loc[
        selection.prompt_frame["include_prompt"].astype(bool)
    ]
    if included.empty:
        raise ExperimentError("frozen selection contains no included prompts")
    rows: list[dict[str, object]] = []
    observed: set[str] = set()
    for raw in included.to_dict(orient="records"):
        try:
            index = safe_index(raw.get("original_index"))
        except GenerationCacheError as error:
            raise ExperimentError(
                "frozen selection contains an unsafe index"
            ) from error
        if index in observed:
            raise ExperimentError(f"frozen selection repeats prompt {index}")
        observed.add(index)
        row = dict(raw)
        row["original_index"] = index
        rows.append(row)
    return tuple(rows)


def _prompt_identity(prompt_row: Mapping[str, object]) -> dict[str, object]:
    return {
        "record_id": prompt_row.get("record_id"),
        "source_row_number": int(prompt_row["source_row_number"]),
        "prompt_raw": prompt_row.get("prompt"),
        "target_image_sha256": prompt_row.get("target_image_sha256"),
    }


def _validated_schedule_tensor(
    schedule: Mapping[str, object], name: str, *, length: int, dtype: torch.dtype
) -> torch.Tensor:
    value = schedule.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (length,)
        or value.dtype != dtype
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
    ):
        raise ExperimentError(f"saved schedule {name} violates its tensor contract")
    return value


def _load_sscd_configuration(
    paths: SSCDPaths,
    *,
    science: Mapping[str, Any],
    scientific_hash: str,
    seeds: tuple[int, ...],
) -> tuple[Mapping[str, Any], str]:
    if not paths.config_json.is_file() or paths.config_json.is_symlink():
        raise ExperimentError("target-specific SSCD configuration is missing or unsafe")
    try:
        configuration = read_json(paths.config_json)
    except CacheIOError as error:
        raise ExperimentError(str(error)) from error
    stored_hash = configuration.get("configuration_hash")
    if not _is_sha256(stored_hash) or stored_hash != sscd_configuration_hash(
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
    wrong = [key for key, item in expected.items() if configuration.get(key) != item]
    if wrong:
        raise ExperimentError(
            "target-specific SSCD configuration differs at: " + ", ".join(wrong)
        )
    return configuration, str(stored_hash)


def _load_generation_contract(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> GenerationContract:
    _validate_scientific_request(scheduler_name, guidance_scale)
    try:
        seed_start, seed_count = validate_seed_block(0, num_seeds)
        paths = generation_paths(
            root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=seed_count,
            seed_start=seed_start,
        )
        configuration = require_generation_run(paths)
    except (CacheIOError, GenerationCacheError, ValueError) as error:
        raise ExperimentError(str(error)) from error
    science = configuration.get("scientific_config")
    scientific_hash = configuration.get("scientific_config_hash")
    if not isinstance(science, Mapping) or not _is_sha256(scientific_hash):
        raise ExperimentError("generation run scientific provenance is invalid")

    spec = get_model_spec(model_name)
    seeds = tuple(range(seed_count))
    expected = {
        "model_cli_name": model_name,
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "guidance_scale": REQUIRED_GUIDANCE_SCALE,
        "num_inference_steps": num_inference_steps,
        "num_seeds": seed_count,
        "seeds": list(seeds),
        "selection_policy": GENERATION_SELECTION_POLICY,
        "trajectory_order": "noise_to_image",
        "stored_prediction_type": "epsilon",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
        "target_preprocessing": target_preprocessing_policy(spec.resolution),
    }
    wrong = [key for key, item in expected.items() if science.get(key) != item]
    scheduler = science.get("scheduler")
    if (
        not isinstance(scheduler, Mapping)
        or scheduler.get("name") != REQUIRED_SCHEDULER
    ):
        wrong.append("scheduler")
    if wrong:
        raise ExperimentError(
            "experiment generation cache differs at: " + ", ".join(sorted(set(wrong)))
        )

    latent_values = _integer_tuple(science.get("latent_shape"), "latent shape")
    if len(latent_values) != 3 or any(value <= 0 for value in latent_values):
        raise ExperimentError("generation latent shape must contain positive [C, H, W]")
    latent_shape = (latent_values[0], latent_values[1], latent_values[2])
    if _integer_tuple(science.get("seeds"), "generation seeds") != seeds:
        raise ExperimentError("generation cache has the wrong seed order")
    storage = science.get("scientific_tensor_storage")
    if not isinstance(storage, Mapping):
        raise ExperimentError("generation tensor storage metadata is invalid")
    stored_dtype = _dtype_from_name(storage.get("dtype"))
    if storage.get("device") != "cpu" or storage.get("layout") != "contiguous":
        raise ExperimentError("generation tensors must be contiguous CPU tensors")

    schedule_sha256 = _schedule_file_sha256(paths)
    try:
        schedule = safe_torch_load(paths.schedule)
    except CacheIOError as error:
        raise ExperimentError(str(error)) from error
    if not isinstance(schedule, Mapping):
        raise ExperimentError("saved generation schedule must be a mapping")
    timesteps = _validated_schedule_tensor(
        schedule, "timesteps", length=num_inference_steps, dtype=torch.int64
    )
    alphas = _validated_schedule_tensor(
        schedule, "alpha_t", length=num_inference_steps, dtype=torch.float32
    )
    sigmas = _validated_schedule_tensor(
        schedule, "sigma_t", length=num_inference_steps, dtype=torch.float32
    )
    cumulative = _validated_schedule_tensor(
        schedule,
        "alphas_cumprod_t",
        length=num_inference_steps,
        dtype=torch.float32,
    )
    if (
        bool((timesteps < 0).any())
        or bool((alphas <= 0).any())
        or bool((sigmas <= 0).any())
    ):
        raise ExperimentError("saved initial schedule values are invalid")
    if len(timesteps) > 1 and not bool((timesteps[:-1] > timesteps[1:]).all()):
        raise ExperimentError("saved DDIM timesteps must be strictly descending")
    if not (
        torch.allclose(alphas.square(), cumulative, rtol=1e-5, atol=1e-6)
        and torch.allclose(sigmas.square(), 1.0 - cumulative, rtol=1e-5, atol=1e-6)
    ):
        raise ExperimentError("saved schedule coefficients are inconsistent")
    try:
        init_noise_sigma = float(schedule.get("init_noise_sigma"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ExperimentError(
            "saved scheduler initial-noise scale is invalid"
        ) from error
    if not math.isfinite(init_noise_sigma) or init_noise_sigma <= 0.0:
        raise ExperimentError("saved scheduler initial-noise scale is invalid")
    if not math.isclose(init_noise_sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ExperimentError("Corollary 3 requires DDIM init_noise_sigma = 1")
    schedule_expected = {
        "scheduler_name": REQUIRED_SCHEDULER,
        "scheduler_class": scheduler.get("class")
        if isinstance(scheduler, Mapping)
        else None,
        "native_prediction_type": science.get("native_prediction_type"),
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
    }
    schedule_wrong = [
        key for key, item in schedule_expected.items() if schedule.get(key) != item
    ]
    if schedule_wrong:
        raise ExperimentError(
            "saved generation schedule differs at: " + ", ".join(schedule_wrong)
        )
    scheduler_config = _normalize_scheduler_config(schedule.get("scheduler_config"))
    scientific_scheduler_config = _normalize_scheduler_config(scheduler.get("config"))
    if canonical_hash(scheduler_config) != canonical_hash(scientific_scheduler_config):
        raise ExperimentError("saved and configured scheduler settings differ")
    num_train_timesteps = _num_train_timesteps(scheduler_config)
    if bool((timesteps >= num_train_timesteps).any()):
        raise ExperimentError("saved DDIM timestep is outside the training schedule")
    if _schedule_file_sha256(paths) != schedule_sha256:
        raise ExperimentError("saved generation schedule SHA-256 changed while loading")

    try:
        records = list_completed_records(paths)
    except (CacheIOError, GenerationCacheError) as error:
        raise ExperimentError(str(error)) from error
    records_by_index = {record.original_index: record for record in records}
    if len(records_by_index) != len(records):
        raise ExperimentError("generation completion records repeat an index")
    sscd_paths = SSCDPaths(paths.run_directory)
    sscd_configuration, sscd_hash = _load_sscd_configuration(
        sscd_paths,
        science=science,
        scientific_hash=str(scientific_hash),
        seeds=seeds,
    )

    alpha_t = float(alphas[0].double().item())
    sigma_t = float(sigmas[0].double().item())
    snr_t = alpha_t * alpha_t / (sigma_t * sigma_t)
    if not math.isfinite(snr_t) or snr_t <= 0.0:
        raise ExperimentError("saved initial alpha_t^2/sigma_t^2 is invalid")
    return GenerationContract(
        paths=paths,
        sscd_paths=sscd_paths,
        science=science,
        scientific_hash=str(scientific_hash),
        schedule_sha256=schedule_sha256,
        seeds=seeds,
        latent_shape=latent_shape,
        latent_dimension=math.prod(latent_shape),
        stored_dtype=stored_dtype,
        init_noise_sigma=init_noise_sigma,
        timestep=int(timesteps[0]),
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        snr_t=snr_t,
        records_by_index=records_by_index,
        sscd_configuration=sscd_configuration,
        sscd_configuration_hash=sscd_hash,
    )


def _load_shared_baseline(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    contract: GenerationContract,
    load_tensor: bool,
) -> UnconditionalBaselineArtifact:
    try:
        baseline = load_unconditional_baseline(
            ROOT,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            num_baseline_seeds=num_baseline_seeds,
            load_tensor=load_tensor,
        )
    except UnconditionalBaselineError as error:
        raise ExperimentError(str(error)) from error

    expected_source_seeds = tuple(range(num_seeds, num_seeds + num_baseline_seeds))
    wrong: list[str] = []
    if baseline.source_seed_start != num_seeds:
        wrong.append("source_seed_start")
    if baseline.num_baseline_seeds != num_baseline_seeds:
        wrong.append("num_baseline_seeds")
    if baseline.source_seeds != expected_source_seeds:
        wrong.append("source_seeds")
    if set(baseline.source_seeds).intersection(contract.seeds):
        wrong.append("source_seed_overlap")
    if baseline.timestep != contract.timestep:
        wrong.append("timestep")
    if baseline.latent_shape != contract.latent_shape:
        wrong.append("latent_shape")
    if not math.isclose(baseline.alpha_t, contract.alpha_t, rel_tol=0.0, abs_tol=0.0):
        wrong.append("alpha_T")
    if not math.isclose(baseline.sigma_t, contract.sigma_t, rel_tol=0.0, abs_tol=0.0):
        wrong.append("sigma_T")
    for name in ("model_id", "model_revision"):
        if baseline.metadata.get(name) != contract.science.get(name):
            wrong.append(name)
    metadata_source_seeds = _integer_tuple(
        baseline.metadata.get("baseline_seeds"), "baseline metadata seeds"
    )
    if baseline.metadata.get("baseline_seed_start") != num_seeds:
        wrong.append("metadata baseline_seed_start")
    if baseline.metadata.get("num_baseline_seeds") != num_baseline_seeds:
        wrong.append("metadata num_baseline_seeds")
    if metadata_source_seeds != expected_source_seeds:
        wrong.append("metadata baseline_seeds")
    if wrong:
        raise ExperimentError(
            "unconditional baseline differs from the experiment at: " + ", ".join(wrong)
        )
    if not _is_sha256(baseline.mu_hat_sha256):
        raise ExperimentError("unconditional baseline mu_hat hash is invalid")
    if not _is_sha256(baseline.source_scientific_config_hash):
        raise ExperimentError(
            "unconditional baseline source scientific hash is invalid"
        )
    if not _is_sha256(baseline.source_schedule_sha256):
        raise ExperimentError("unconditional baseline schedule hash is invalid")
    if baseline.source_schedule_sha256 != contract.schedule_sha256:
        raise ExperimentError(
            "unconditional baseline and experiment schedule SHA-256 digests differ"
        )
    if load_tensor:
        if baseline.mu_hat is None:
            raise ExperimentError("unconditional baseline mu_hat tensor was not loaded")
        _validate_tensor(
            baseline.mu_hat,
            shape=contract.latent_shape,
            dtype=torch.float64,
            label="unconditional baseline mu_hat",
        )
    elif baseline.mu_hat is not None:
        raise ExperimentError("metadata-only baseline load deserialized mu_hat")
    return baseline


def _load_centering_contract(
    *,
    use_mu: bool,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    contract: GenerationContract,
    load_tensor: bool,
) -> CenteringContract:
    """Resolve exact zero centering or the validated saved shared baseline."""

    if not use_mu:
        value = (
            torch.zeros(contract.latent_shape, dtype=torch.float64)
            if load_tensor
            else None
        )
        centering = CenteringContract(
            mode=CENTERING_ZERO,
            value=value,
            baseline_generation_scientific_config_hash="",
            baseline_schedule_sha256="",
            baseline_mu_hat_sha256="",
            num_baseline_seeds=0,
        )
    else:
        baseline = _load_shared_baseline(
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            num_baseline_seeds=num_baseline_seeds,
            contract=contract,
            load_tensor=load_tensor,
        )
        centering = CenteringContract(
            mode=CENTERING_MU_HAT,
            value=baseline.mu_hat,
            baseline_generation_scientific_config_hash=(
                baseline.source_scientific_config_hash
            ),
            baseline_schedule_sha256=baseline.source_schedule_sha256,
            baseline_mu_hat_sha256=baseline.mu_hat_sha256,
            num_baseline_seeds=num_baseline_seeds,
        )
    return _validate_centering_contract(
        centering, contract=contract, tensor_required=load_tensor
    )


def _validate_centering_contract(
    centering: CenteringContract,
    *,
    contract: GenerationContract,
    tensor_required: bool | None,
) -> CenteringContract:
    if centering.mode not in CENTERING_MODES:
        raise ExperimentError(f"unsupported centering mode: {centering.mode!r}")
    hashes = (
        centering.baseline_generation_scientific_config_hash,
        centering.baseline_schedule_sha256,
        centering.baseline_mu_hat_sha256,
    )
    if centering.mode == CENTERING_ZERO:
        if hashes != ("", "", "") or centering.num_baseline_seeds != 0:
            raise ExperimentError("zero centering must not contain baseline provenance")
    elif (
        not all(_is_sha256(value) for value in hashes)
        or isinstance(centering.num_baseline_seeds, bool)
        or centering.num_baseline_seeds <= 0
    ):
        raise ExperimentError("mu_hat centering has invalid baseline provenance")
    elif centering.baseline_schedule_sha256 != contract.schedule_sha256:
        raise ExperimentError(
            "mu_hat baseline and experiment schedule SHA-256 digests differ"
        )
    if tensor_required is True and centering.value is None:
        raise ExperimentError("active Corollary 3 center tensor was not loaded")
    if tensor_required is False and centering.value is not None:
        raise ExperimentError("plot-only centering deserialized a tensor")
    if centering.value is not None:
        _validate_tensor(
            centering.value,
            shape=contract.latent_shape,
            dtype=torch.float64,
            label="active Corollary 3 center",
        )
        if centering.mode == CENTERING_ZERO and bool(
            torch.count_nonzero(centering.value)
        ):
            raise ExperimentError("zero centering tensor must be exactly zero")
    return centering


def _validate_generation_prompt(
    prompt_row: Mapping[str, object],
    *,
    contract: GenerationContract,
    tensor_names: Sequence[str],
) -> Mapping[str, Any]:
    index = safe_index(prompt_row.get("original_index"))
    if index not in contract.records_by_index:
        raise ExperimentError(f"selected prompt has no generation record: {index}")
    validation = validate_generation_record(
        contract.paths,
        index,
        expected_scientific_hash=contract.scientific_hash,
        expected_record_identity=_prompt_identity(prompt_row),
        load_tensors=False,
        tensor_names=tensor_names,
        require_preview=False,
        verify_file_hashes=True,
    )
    if not validation.valid or validation.metadata is None:
        raise ExperimentError(
            f"invalid cached record for {index}: " + "; ".join(validation.errors)
        )
    marker = validation.metadata
    expected = {
        "model_cli_name": contract.science.get("model_cli_name"),
        "dataset_model": contract.science.get("dataset_model"),
        "model_id": contract.science.get("model_id"),
        "model_revision": contract.science.get("model_revision"),
        "native_prediction_type": contract.science.get("native_prediction_type"),
        "target_preprocessing": contract.science.get("target_preprocessing"),
        "scientific_config_hash": contract.scientific_hash,
        "scheduler_name": REQUIRED_SCHEDULER,
        "guidance_scale": REQUIRED_GUIDANCE_SCALE,
        "num_inference_steps": contract.science.get("num_inference_steps"),
        "num_seeds": len(contract.seeds),
        "seeds": list(contract.seeds),
        "trajectory_order": "noise_to_image",
        "stored_prediction_type": "epsilon",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
    }
    wrong = [key for key, item in expected.items() if marker.get(key) != item]
    if wrong:
        raise ExperimentError(
            f"generation marker for {index} differs at: " + ", ".join(wrong)
        )
    return marker


def _validate_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    label: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ExperimentError(f"cached {label} violates its tensor contract")
    return value


def _reconstruct_initial_latents(contract: GenerationContract) -> torch.Tensor:
    """Repeat the sampler's CPU-noise, dtype-cast, then scheduler-scale order."""

    raw = make_initial_noise(contract.seeds, contract.latent_shape)
    reconstructed = raw.to(dtype=contract.stored_dtype).mul(contract.init_noise_sigma)
    return _validate_tensor(
        reconstructed.contiguous(),
        shape=(len(contract.seeds), *contract.latent_shape),
        dtype=contract.stored_dtype,
        label="reconstructed initial latent",
    )


def _validate_initial_latent_sentinel(
    prompt_row: Mapping[str, object], contract: GenerationContract
) -> None:
    """Pin reconstruction against one fully hash-validated cached x_T bundle."""

    _validate_generation_prompt(prompt_row, contract=contract, tensor_names=("latent",))
    index = safe_index(prompt_row.get("original_index"))
    try:
        value = safe_torch_load(contract.paths.latent_path(index))
    except CacheIOError as error:
        raise ExperimentError(str(error)) from error
    trajectory = _validate_tensor(
        value,
        shape=(
            len(contract.seeds),
            int(contract.science["num_inference_steps"]) + 1,
            *contract.latent_shape,
        ),
        dtype=contract.stored_dtype,
        label="sentinel latent trajectory",
    )
    reconstructed = _reconstruct_initial_latents(contract)
    if not torch.equal(trajectory[:, 0], reconstructed):
        raise ExperimentError(
            "canonical reconstructed x_T is not bitwise equal to cached latent[:, 0]"
        )


def _load_prediction_and_target(
    prompt_row: Mapping[str, object], contract: GenerationContract
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Mapping[str, Any],
]:
    marker = _validate_generation_prompt(
        prompt_row,
        contract=contract,
        tensor_names=("noise_prediction", "target_latent"),
    )
    index = safe_index(prompt_row.get("original_index"))
    try:
        prediction_value = safe_torch_load(contract.paths.noise_prediction_path(index))
        target_value = safe_torch_load(contract.paths.target_latent_path(index))
    except CacheIOError as error:
        raise ExperimentError(str(error)) from error
    if not isinstance(prediction_value, tuple) or len(prediction_value) != 2:
        raise ExperimentError("cached noise prediction is not a two-tensor tuple")
    prediction_shape = (
        len(contract.seeds),
        int(contract.science["num_inference_steps"]),
        *contract.latent_shape,
    )
    unconditional = _validate_tensor(
        prediction_value[0],
        shape=prediction_shape,
        dtype=contract.stored_dtype,
        label="unconditional epsilon predictions",
    )
    conditional = _validate_tensor(
        prediction_value[1],
        shape=prediction_shape,
        dtype=contract.stored_dtype,
        label="conditional epsilon predictions",
    )
    target = _validate_tensor(
        target_value,
        shape=contract.latent_shape,
        dtype=torch.float32,
        label="target latent",
    )
    return unconditional[:, 0], conditional[:, 0], target, marker


def _load_target_sscd(
    prompt_row: Mapping[str, object],
    *,
    contract: GenerationContract,
    generation_marker: Mapping[str, Any],
) -> torch.Tensor:
    """Load one fully validated score for each exact generation-seed position."""

    index = safe_index(prompt_row.get("original_index"))
    marker_path = contract.sscd_paths.marker_path(index)
    score_path = contract.sscd_paths.score_path(index)
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ExperimentError("target-specific SSCD marker is missing or unsafe")
    if not score_path.is_file() or score_path.is_symlink():
        raise ExperimentError("target-specific SSCD tensor is missing or unsafe")
    try:
        marker = read_json(marker_path)
    except CacheIOError as error:
        raise ExperimentError(str(error)) from error
    configured_seeds = _integer_tuple(
        contract.sscd_configuration.get("seeds"), "SSCD seeds"
    )
    if configured_seeds != contract.seeds:
        raise ExperimentError("target-specific SSCD seed order differs from generation")
    latent_hashes = generation_marker.get("tensor_file_sha256")
    expected = {
        "original_index": index,
        "record_id": prompt_row.get("record_id"),
        "source_row_number": int(prompt_row["source_row_number"]),
        "prompt_raw": prompt_row.get("prompt"),
        "target_image_sha256": prompt_row.get("target_image_sha256"),
        "model_id": contract.science.get("model_id"),
        "model_revision": contract.science.get("model_revision"),
        "vae_id": contract.science.get("vae_id"),
        "vae_revision": contract.science.get("vae_revision"),
        "terminal_latent_index": -1,
        "score_shape": [len(contract.seeds)],
        "score_dtype": "float32",
        "num_seeds": len(contract.seeds),
        "seeds": list(contract.seeds),
        "similarity": SCORE_DEFINITION,
        "sscd_configuration_hash": contract.sscd_configuration_hash,
        "generation_scientific_config_hash": contract.scientific_hash,
        "generation_latent_sha256": latent_hashes.get("latent")
        if isinstance(latent_hashes, Mapping)
        else None,
    }
    wrong = [key for key, item in expected.items() if marker.get(key) != item]
    if wrong:
        raise ExperimentError(
            "target-specific SSCD record differs at: " + ", ".join(wrong)
        )
    expected_score_hash = marker.get("score_sha256")
    if (
        not _is_sha256(expected_score_hash)
        or file_sha256(score_path) != expected_score_hash
    ):
        raise ExperimentError("target-specific SSCD tensor hash differs")
    try:
        scores = validate_sscd_scores(safe_torch_load(score_path), len(contract.seeds))
    except (CacheIOError, SSCDMetricError) as error:
        raise ExperimentError(str(error)) from error
    seed_positions = {seed: position for position, seed in enumerate(configured_seeds)}
    aligned = scores[[seed_positions[seed] for seed in contract.seeds]]
    return aligned.double().contiguous()


def _verify_cfg_identity(
    x_t: torch.Tensor,
    epsilon_empty: torch.Tensor,
    epsilon_c: torch.Tensor,
    xhat_empty: torch.Tensor,
    xhat_c: torch.Tensor,
    xhat_g: torch.Tensor,
    *,
    alpha_t: float,
    sigma_t: float,
    guidance_scale: float,
) -> None:
    epsilon_g = guidance_scale * epsilon_c + (1.0 - guidance_scale) * epsilon_empty
    independent = (x_t - sigma_t * epsilon_g) / alpha_t
    if not torch.allclose(xhat_g, independent, rtol=CFG_RTOL, atol=CFG_ATOL):
        raise ExperimentError(
            "cached estimates violate the exact CFG clean-estimate identity"
        )
    branch_identity = guidance_scale * xhat_c + (1.0 - guidance_scale) * xhat_empty
    if not torch.equal(xhat_g, branch_identity):
        raise ExperimentError("guided clean estimate was not constructed by exact CFG")


def _measure_values(
    x_t: torch.Tensor,
    epsilon_empty: torch.Tensor,
    epsilon_c: torch.Tensor,
    target: torch.Tensor,
    center: torch.Tensor,
    target_sscd: torch.Tensor,
    *,
    contract: GenerationContract,
    device: str | torch.device,
    centering_mode: str,
) -> np.ndarray:
    if centering_mode not in CENTERING_MODES:
        raise ExperimentError(f"unsupported centering mode: {centering_mode!r}")
    _validate_tensor(
        center,
        shape=contract.latent_shape,
        dtype=torch.float64,
        label="active Corollary 3 center",
    )
    selected = torch.device(device)
    reduction_device = torch.device("cpu") if selected.type == "mps" else selected
    with torch.inference_mode():
        x_t_d = x_t.to(device=reduction_device, dtype=torch.float64)
        empty_epsilon_d = epsilon_empty.to(device=reduction_device, dtype=torch.float64)
        conditional_epsilon_d = epsilon_c.to(
            device=reduction_device, dtype=torch.float64
        )
        target_d = target.to(device=reduction_device, dtype=torch.float64)
        center_d = center.to(device=reduction_device, dtype=torch.float64)
        if centering_mode == CENTERING_ZERO and bool(torch.count_nonzero(center_d)):
            raise ExperimentError("zero centering tensor must be exactly zero")
        xhat_empty = (x_t_d - contract.sigma_t * empty_epsilon_d) / contract.alpha_t
        xhat_c = (x_t_d - contract.sigma_t * conditional_epsilon_d) / contract.alpha_t
        xhat_g = (
            REQUIRED_GUIDANCE_SCALE * xhat_c
            + (1.0 - REQUIRED_GUIDANCE_SCALE) * xhat_empty
        )
        _verify_cfg_identity(
            x_t_d,
            empty_epsilon_d,
            conditional_epsilon_d,
            xhat_empty,
            xhat_c,
            xhat_g,
            alpha_t=contract.alpha_t,
            sigma_t=contract.sigma_t,
            guidance_scale=REQUIRED_GUIDANCE_SCALE,
        )

        if centering_mode == CENTERING_ZERO:
            target_direction = target_d
            guided_direction = xhat_g
            guided_reference = REQUIRED_GUIDANCE_SCALE * target_d
            unconditional_error = xhat_empty
        else:
            target_direction = target_d - center_d
            guided_direction = xhat_g - center_d
            guided_reference = center_d + REQUIRED_GUIDANCE_SCALE * target_direction
            unconditional_error = xhat_empty - center_d
        flat_target_direction = target_direction.flatten()
        target_squared = torch.dot(flat_target_direction, flat_target_direction)
        if not bool(torch.isfinite(target_squared)) or float(target_squared) <= 0.0:
            raise ExperimentError(
                "target direction from the active center has zero or invalid squared norm"
            )
        flat_guided_direction = guided_direction.flatten(start_dim=1)
        fitted = (flat_guided_direction @ flat_target_direction) / target_squared
        residual = guided_direction - fitted.view(-1, 1, 1, 1) * target_direction

        flat_residual = residual.flatten(start_dim=1)
        orthogonality = flat_residual @ flat_target_direction
        orthogonal_bound = PROJECTION_ATOL + PROJECTION_RTOL * (
            flat_residual.norm(dim=1) * flat_target_direction.norm()
        )
        if bool((orthogonality.abs() > orthogonal_bound).any()):
            raise ExperimentError("no-intercept projection residual is not orthogonal")
        lhs = flat_guided_direction.square().sum(dim=1)
        rhs = flat_residual.square().sum(dim=1) + fitted.square() * target_squared
        if not torch.allclose(lhs, rhs, rtol=PROJECTION_RTOL, atol=PROJECTION_ATOL):
            raise ExperimentError(
                "no-intercept projection violates Pythagorean identity"
            )
        guided_error = xhat_g - guided_reference
        guided_lhs = guided_error.flatten(start_dim=1).square().sum(dim=1)
        guided_rhs = (
            flat_residual.square().sum(dim=1)
            + (fitted - REQUIRED_GUIDANCE_SCALE).square() * target_squared
        )
        if not torch.allclose(
            guided_lhs, guided_rhs, rtol=PROJECTION_RTOL, atol=PROJECTION_ATOL
        ):
            raise ExperimentError(
                "guided-target error violates the projection decomposition"
            )
        residual_rmse = flat_residual.square().mean(dim=1).sqrt()
        guided_target_rmse = (
            guided_error.flatten(start_dim=1).square().mean(dim=1).sqrt()
        )
        conditional_rmse = (
            (xhat_c - target_d).flatten(start_dim=1).square().mean(dim=1).sqrt()
        )
        unconditional_rmse = (
            unconditional_error.flatten(start_dim=1).square().mean(dim=1).sqrt()
        )
        values = torch.stack(
            (
                fitted,
                residual_rmse,
                guided_target_rmse,
                conditional_rmse,
                unconditional_rmse,
                target_sscd.to(device=reduction_device, dtype=torch.float64),
            ),
            dim=1,
        ).cpu()
    if tuple(values.shape) != (len(contract.seeds), 6) or not bool(
        torch.isfinite(values).all()
    ):
        raise ExperimentError("Corollary 3 measurements are non-finite or incomplete")
    if bool((values[:, 1:5] < 0.0).any()):
        raise ExperimentError("Corollary 3 RMSE measurements must be nonnegative")
    return values.numpy().copy()


def _error_message(error: Exception) -> str:
    detail = " ".join(str(error).splitlines()).strip()
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _measure_prompt_compact(
    position: int,
    prompt_row: Mapping[str, object],
    *,
    contract: GenerationContract,
    device: str | torch.device,
    center: torch.Tensor,
    centering_mode: str,
    initial_latents: torch.Tensor | None = None,
) -> _PromptMeasurement:
    x_t = (
        initial_latents
        if initial_latents is not None
        else _reconstruct_initial_latents(contract)
    )
    epsilon_empty, epsilon_c, target, marker = _load_prediction_and_target(
        prompt_row, contract
    )
    target_sscd = _load_target_sscd(
        prompt_row, contract=contract, generation_marker=marker
    )
    values = _measure_values(
        x_t,
        epsilon_empty,
        epsilon_c,
        target,
        center,
        target_sscd,
        contract=contract,
        device=device,
        centering_mode=centering_mode,
    )
    return _PromptMeasurement(
        position=position,
        original_index=safe_index(prompt_row.get("original_index")),
        values=values,
        error="",
    )


def _measure_prompt_safely(
    position: int,
    prompt_row: Mapping[str, object],
    *,
    contract: GenerationContract,
    device: str | torch.device,
    center: torch.Tensor,
    centering_mode: str,
    initial_latents: torch.Tensor | None = None,
) -> _PromptMeasurement:
    try:
        return _measure_prompt_compact(
            position,
            prompt_row,
            contract=contract,
            device=device,
            center=center,
            centering_mode=centering_mode,
            initial_latents=initial_latents,
        )
    except Exception as error:
        return _PromptMeasurement(
            position=position,
            original_index=str(prompt_row.get("original_index", "")),
            values=None,
            error=_error_message(error),
        )


def _rows_for_measurement(
    measurement: _PromptMeasurement,
    prompt_row: Mapping[str, object],
    contract: GenerationContract,
    centering: CenteringContract,
) -> list[dict[str, object]]:
    _validate_centering_contract(centering, contract=contract, tensor_required=None)
    rows: list[dict[str, object]] = []
    if measurement.values is not None and tuple(measurement.values.shape) != (
        len(contract.seeds),
        6,
    ):
        raise ExperimentError("compact prompt measurement has an invalid shape")
    for seed_position, seed in enumerate(contract.seeds):
        common: dict[str, object] = {
            "record_id": str(prompt_row.get("record_id", "")),
            "generation_seed": seed,
            "timestep": contract.timestep,
            "alpha_t": contract.alpha_t,
            "sigma_t": contract.sigma_t,
            "snr_t": contract.snr_t,
            "guidance_scale": REQUIRED_GUIDANCE_SCALE,
            "centering_mode": centering.mode,
            "baseline_generation_scientific_config_hash": (
                centering.baseline_generation_scientific_config_hash
            ),
            "baseline_schedule_sha256": centering.baseline_schedule_sha256,
            "baseline_mu_hat_sha256": centering.baseline_mu_hat_sha256,
            "num_baseline_seeds": centering.num_baseline_seeds,
        }
        if measurement.values is None:
            common.update(
                {
                    "fitted_guidance_scale": math.nan,
                    "residual_rmse": math.nan,
                    "guided_target_rmse": math.nan,
                    "conditional_recovery_rmse": math.nan,
                    "unconditional_rmse": math.nan,
                    "target_sscd": math.nan,
                    "status": "error",
                    "error": measurement.error,
                }
            )
        else:
            fitted, residual, guided, conditional, unconditional, score = (
                measurement.values[seed_position]
            )
            common.update(
                {
                    "fitted_guidance_scale": float(fitted),
                    "residual_rmse": float(residual),
                    "guided_target_rmse": float(guided),
                    "conditional_recovery_rmse": float(conditional),
                    "unconditional_rmse": float(unconditional),
                    "target_sscd": float(score),
                    "status": "ok",
                    "error": "",
                }
            )
        rows.append(common)
    return rows


def _initialize_prompt_worker(
    contract: GenerationContract,
    center: torch.Tensor,
    centering_mode: str,
    device_string: str,
    worker_count: int,
) -> None:
    global _WORKER_CONTRACT, _WORKER_DEVICE, _WORKER_INITIAL_LATENTS
    global _WORKER_CENTER, _WORKER_CENTERING_MODE
    configure_worker_cpu_threads(worker_count)
    _WORKER_CONTRACT = contract
    _WORKER_DEVICE = torch.device(device_string)
    if _WORKER_DEVICE.type == "cuda":
        torch.cuda.set_device(_WORKER_DEVICE)
    _WORKER_INITIAL_LATENTS = _reconstruct_initial_latents(contract)
    _WORKER_CENTER = _validate_tensor(
        center,
        shape=contract.latent_shape,
        dtype=torch.float64,
        label="active Corollary 3 center",
    )
    if centering_mode not in CENTERING_MODES:
        raise ExperimentError(f"unsupported centering mode: {centering_mode!r}")
    _WORKER_CENTERING_MODE = centering_mode


def _prompt_worker_task(
    position: int, prompt_row: Mapping[str, object]
) -> _PromptMeasurement:
    if (
        _WORKER_CONTRACT is None
        or _WORKER_DEVICE is None
        or _WORKER_INITIAL_LATENTS is None
        or _WORKER_CENTER is None
        or _WORKER_CENTERING_MODE is None
    ):
        raise ExperimentError("Corollary 3 worker was not initialized")
    return _measure_prompt_safely(
        position,
        prompt_row,
        contract=_WORKER_CONTRACT,
        device=_WORKER_DEVICE,
        center=_WORKER_CENTER,
        centering_mode=_WORKER_CENTERING_MODE,
        initial_latents=_WORKER_INITIAL_LATENTS,
    )


def _canonical_measurements(
    measurements: Sequence[_PromptMeasurement], prompt_count: int
) -> tuple[_PromptMeasurement, ...]:
    by_position: dict[int, _PromptMeasurement] = {}
    for result in measurements:
        if result.position in by_position or not 0 <= result.position < prompt_count:
            raise ExperimentError(
                "worker results contain duplicate or invalid positions"
            )
        by_position[result.position] = result
    missing = sorted(set(range(prompt_count)) - set(by_position))
    if missing:
        raise ExperimentError("worker results are missing prompt positions")
    return tuple(by_position[position] for position in range(prompt_count))


def _compute_rows(
    prompt_rows: Sequence[Mapping[str, object]],
    *,
    contract: GenerationContract,
    centering: CenteringContract,
    devices: Sequence[torch.device],
    progress_factory: Callable[..., Any] = tqdm,
) -> tuple[list[dict[str, object]], int]:
    _validate_centering_contract(centering, contract=contract, tensor_required=True)
    if centering.value is None:
        raise ExperimentError("active Corollary 3 center tensor was not loaded")
    if not prompt_rows:
        raise ExperimentError("no selected prompts were supplied")
    if not devices:
        raise ExperimentError("at least one execution device is required")
    worker_count = worker_count_for_tasks(devices, len(prompt_rows))
    use_parallel = worker_count > 1 and all(
        device.type == "cuda" and device.index is not None
        for device in devices[:worker_count]
    )
    measurements: list[_PromptMeasurement] = []
    with progress_factory(
        total=len(prompt_rows) * len(contract.seeds),
        desc=PROGRESS_DESCRIPTION,
        unit="observation",
        dynamic_ncols=True,
    ) as progress:
        if not use_parallel:
            initial_latents = _reconstruct_initial_latents(contract)
            selected_device = devices[0]
            for position, prompt_row in enumerate(prompt_rows):
                measurements.append(
                    _measure_prompt_safely(
                        position,
                        prompt_row,
                        contract=contract,
                        device=selected_device,
                        center=centering.value,
                        centering_mode=centering.mode,
                        initial_latents=initial_latents,
                    )
                )
                progress.update(len(contract.seeds))
        else:
            context = multiprocessing.get_context("spawn")
            futures: dict[
                Future[_PromptMeasurement], tuple[int, Mapping[str, object]]
            ] = {}
            with ExitStack() as stack:
                for worker_index in range(worker_count):
                    device = devices[worker_index]
                    executor = stack.enter_context(
                        ProcessPoolExecutor(
                            max_workers=1,
                            mp_context=context,
                            initializer=_initialize_prompt_worker,
                            initargs=(
                                contract,
                                centering.value,
                                centering.mode,
                                str(device),
                                worker_count,
                            ),
                        )
                    )
                    indexed = tuple(enumerate(prompt_rows))
                    for position, prompt_row in round_robin_shard(
                        indexed,
                        worker_index=worker_index,
                        worker_count=worker_count,
                    ):
                        future = executor.submit(
                            _prompt_worker_task, position, prompt_row
                        )
                        futures[future] = (position, prompt_row)
                for future in as_completed(futures):
                    position, prompt_row = futures[future]
                    try:
                        result = future.result()
                    except BaseException as error:
                        result = _PromptMeasurement(
                            position=position,
                            original_index=str(prompt_row.get("original_index", "")),
                            values=None,
                            error=_error_message(
                                error
                                if isinstance(error, Exception)
                                else RuntimeError(str(error))
                            ),
                        )
                    measurements.append(result)
                    progress.update(len(contract.seeds))

    canonical = _canonical_measurements(measurements, len(prompt_rows))
    rows: list[dict[str, object]] = []
    failed = 0
    for position, measurement in enumerate(canonical):
        expected_index = safe_index(prompt_rows[position].get("original_index"))
        if measurement.original_index != expected_index:
            raise ExperimentError("worker result prompt identity differs")
        rows.extend(
            _rows_for_measurement(
                measurement,
                prompt_rows[position],
                contract,
                centering,
            )
        )
        failed += int(measurement.values is None)
    return rows, failed


def _default_output_directory(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    centering_mode: str,
    selection_strategy: str,
    selection_hash: str,
) -> Path:
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ExperimentError(f"unsupported selection strategy: {selection_strategy!r}")
    if not _is_sha256(selection_hash):
        raise ExperimentError("selection hash must be a lowercase SHA-256 digest")
    if centering_mode not in CENTERING_MODES:
        raise ExperimentError(f"unsupported centering mode: {centering_mode!r}")
    run_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        0,
    )
    experiment_root = ROOT / "outputs" / run_name / EXPERIMENT_DIRECTORY
    if centering_mode == CENTERING_ZERO:
        return experiment_root / "centering_zero" / selection_strategy / selection_hash
    if isinstance(num_baseline_seeds, bool) or num_baseline_seeds <= 0:
        raise ExperimentError("num_baseline_seeds must be positive with --use-mu")
    return (
        experiment_root
        / "centering_mu_hat"
        / f"baseline_S{num_seeds}_N{num_baseline_seeds}"
        / selection_strategy
        / selection_hash
    )


def _requested_output_directory(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    centering_mode: str,
    selection_strategy: str,
    selection_hash: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return _default_output_directory(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        num_baseline_seeds,
        centering_mode,
        selection_strategy,
        selection_hash,
    )


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


def _figure_paths(output_directory: str | Path) -> tuple[Path, ...]:
    output = Path(output_directory)
    return tuple(output / filename for filename in FIGURE_FILENAMES)


def _remove_figure_outputs(output_directory: str | Path) -> None:
    for destination in _figure_paths(output_directory):
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
        elif destination.exists():
            raise ExperimentError(f"figure destination is unsafe: {destination}")


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
                    f"unsupported figure format: {destination.suffix!r}"
                )
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            staged.append((temporary, destination))
            options: dict[str, object] = {
                "format": figure_format,
                "bbox_inches": "tight",
                "pad_inches": FIGURE_PAD_INCHES,
            }
            if figure_format == "png":
                options["dpi"] = 300
            figure.savefig(temporary, **options)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        for _temporary, destination in staged:
            if destination.exists() and not (
                destination.is_file() or destination.is_symlink()
            ):
                raise ExperimentError(f"figure destination is unsafe: {destination}")
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


def _numeric_series(
    frame: pd.DataFrame, name: str, *, integer: bool = False
) -> pd.Series:
    values = pd.to_numeric(frame[name], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ExperimentError(f"saved CSV {name} must be finite and numeric")
    if integer and not values.map(lambda value: float(value).is_integer()).all():
        raise ExperimentError(f"saved CSV {name} must contain integers")
    return values.astype(np.int64) if integer else values.astype(np.float64)


def _validated_plot_frame(
    frame: pd.DataFrame,
    *,
    selection: TargetPairSelection,
    contract: GenerationContract,
    centering: CenteringContract,
) -> pd.DataFrame:
    _validate_centering_contract(centering, contract=contract, tensor_required=None)
    if tuple(frame.columns) != CSV_COLUMNS:
        raise ExperimentError("saved CSV columns do not match the experiment schema")
    prompts = _included_prompt_rows(selection)
    expected_count = len(prompts) * len(contract.seeds)
    if len(frame) != expected_count:
        raise ExperimentError(
            f"saved CSV has {len(frame)}/{expected_count} prompt-seed rows"
        )
    validated = frame.copy()
    validated["record_id"] = validated["record_id"].astype(str)
    text_constants = {
        "centering_mode": centering.mode,
        "baseline_generation_scientific_config_hash": (
            centering.baseline_generation_scientific_config_hash
        ),
        "baseline_schedule_sha256": centering.baseline_schedule_sha256,
        "baseline_mu_hat_sha256": centering.baseline_mu_hat_sha256,
    }
    for name, expected in text_constants.items():
        if not validated[name].astype(str).eq(expected).all():
            raise ExperimentError(f"saved CSV {name} differs from requested provenance")
    for name in ("generation_seed", "timestep", "num_baseline_seeds"):
        validated[name] = _numeric_series(validated, name, integer=True)
    if not validated["num_baseline_seeds"].eq(centering.num_baseline_seeds).all():
        raise ExperimentError(
            "saved CSV num_baseline_seeds differs from the requested centering mode"
        )
    for name in (
        "alpha_t",
        "sigma_t",
        "snr_t",
        "guidance_scale",
        "fitted_guidance_scale",
        "residual_rmse",
        "guided_target_rmse",
        "conditional_recovery_rmse",
        "unconditional_rmse",
        "target_sscd",
    ):
        validated[name] = _numeric_series(validated, name)
    if not validated["status"].astype(str).eq("ok").all():
        raise ExperimentError("saved CSV contains failed observations")
    if not validated["error"].fillna("").astype(str).str.strip().eq("").all():
        raise ExperimentError("saved CSV status/error fields are inconsistent")
    for name in (
        "residual_rmse",
        "guided_target_rmse",
        "conditional_recovery_rmse",
        "unconditional_rmse",
    ):
        if bool((validated[name] < 0.0).any()):
            raise ExperimentError(f"saved CSV {name} must be nonnegative")
    if bool(
        (
            (validated["target_sscd"] < -1.000001)
            | (validated["target_sscd"] > 1.000001)
        ).any()
    ):
        raise ExperimentError("saved CSV target_sscd is outside [-1, 1]")

    expected_records = np.repeat(
        np.asarray([str(row["record_id"]) for row in prompts], dtype=object),
        len(contract.seeds),
    )
    expected_seeds = np.tile(np.asarray(contract.seeds, dtype=np.int64), len(prompts))
    if not np.array_equal(
        validated["record_id"].to_numpy(dtype=object), expected_records
    ):
        raise ExperimentError(
            "saved CSV records are not the canonical frozen selection"
        )
    if not np.array_equal(validated["generation_seed"].to_numpy(), expected_seeds):
        raise ExperimentError(
            "saved CSV does not contain the complete ordered seed grid"
        )
    constants = {
        "timestep": contract.timestep,
        "alpha_t": contract.alpha_t,
        "sigma_t": contract.sigma_t,
        "snr_t": contract.snr_t,
        "guidance_scale": REQUIRED_GUIDANCE_SCALE,
    }
    for name, expected in constants.items():
        observed = validated[name].to_numpy(dtype=float)
        if not np.allclose(observed, float(expected), rtol=1e-13, atol=0.0):
            raise ExperimentError(f"saved CSV {name} differs from requested provenance")
    return validated


def _render_figure(
    frame: pd.DataFrame,
    destinations: Sequence[Path],
    *,
    centering_mode: str,
) -> None:
    if centering_mode not in CENTERING_MODES:
        raise ExperimentError(f"unsupported centering mode: {centering_mode!r}")
    with matplotlib.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        try:
            norm = Normalize(
                vmin=SSCD_COLOR_RANGE[0], vmax=SSCD_COLOR_RANGE[1], clip=True
            )
            colors = frame["target_sscd"].to_numpy(dtype=float)
            scatter = axis.scatter(
                frame["fitted_guidance_scale"].to_numpy(dtype=float),
                frame["residual_rmse"].to_numpy(dtype=float),
                c=colors,
                cmap="viridis",
                norm=norm,
                s=22,
                alpha=SCATTER_ALPHA,
                edgecolors="none",
            )
            mappable = ScalarMappable(norm=scatter.norm, cmap=scatter.cmap)
            mappable.set_array(colors)
            colorbar = figure.colorbar(mappable, ax=axis)
            if colorbar.solids is not None:
                colorbar.solids.set_alpha(COLORBAR_ALPHA)
            colorbar.set_label(COLORBAR_LABEL, fontsize=TEXT_FONT_SIZE)
            colorbar.ax.tick_params(labelsize=AXIS_NUMBER_FONT_SIZE)
            axis.axvline(
                REQUIRED_GUIDANCE_SCALE,
                color="black",
                linewidth=0.8,
                linestyle="--",
                alpha=0.75,
            )
            axis.set_xlabel(r"$\widehat{g}$")
            if centering_mode == CENTERING_MU_HAT:
                axis.set_ylabel(
                    r"$\|(\widehat{\mathbf{x}}_{0\mid T,g}-\widehat{\boldsymbol{\mu}})-"
                    r"\widehat{g}(\mathbf{x}^{\star}-\widehat{\boldsymbol{\mu}})"
                    r"\|_2/\sqrt{d}$"
                )
            else:
                axis.set_ylabel(
                    r"$\|\widehat{\mathbf{x}}_{0\mid T,g}-"
                    r"\widehat{g}\mathbf{x}^{\star}\|_2/\sqrt{d}$"
                )
            axis.tick_params(axis="both", which="both", labelsize=AXIS_NUMBER_FONT_SIZE)
            axis.grid(True, which="both", alpha=0.18, linewidth=0.6)
            figure.tight_layout()
            _atomic_save_figures(figure, destinations)
        finally:
            plt.close(figure)


def plot_saved_results(
    csv_path: str | Path,
    output_directory: str | Path,
    *,
    selection: TargetPairSelection,
    contract: GenerationContract,
    centering: CenteringContract,
) -> None:
    source = Path(csv_path)
    if source.is_symlink() or not source.is_file():
        raise ExperimentError(f"saved CSV is missing or unsafe: {source}")
    try:
        frame = pd.read_csv(
            source,
            dtype={
                "record_id": str,
                "centering_mode": str,
                "baseline_generation_scientific_config_hash": str,
                "baseline_schedule_sha256": str,
                "baseline_mu_hat_sha256": str,
            },
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ExperimentError(f"cannot read saved CSV {source}: {error}") from error
    validated = _validated_plot_frame(
        frame,
        selection=selection,
        contract=contract,
        centering=centering,
    )
    _render_figure(
        validated,
        _figure_paths(output_directory),
        centering_mode=centering.mode,
    )


def run_experiment(
    *,
    model_name: str,
    selection_strategy: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    use_mu: bool = False,
    output_dir: str | Path | None,
    device: str | torch.device = "auto",
    progress_factory: Callable[..., Any] = tqdm,
) -> int:
    _validate_scientific_request(scheduler_name, guidance_scale)
    selection = _load_frozen_selection(
        ROOT,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        selection_strategy=selection_strategy,
    )
    contract = _load_generation_contract(
        ROOT,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
    )
    centering = _load_centering_contract(
        use_mu=use_mu,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        num_baseline_seeds=num_baseline_seeds,
        contract=contract,
        load_tensor=True,
    )
    if centering.value is None:
        raise ExperimentError("active Corollary 3 center tensor was not loaded")
    print(f"Centering mode: {centering.mode}")
    prompts = _included_prompt_rows(selection)
    missing = [
        str(row["original_index"])
        for row in prompts
        if str(row["original_index"]) not in contract.records_by_index
    ]
    if missing:
        raise ExperimentError(
            "selected prompts are absent from generation cache: "
            + ", ".join(missing[:5])
        )
    requested = _requested_output_directory(
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        num_baseline_seeds=num_baseline_seeds,
        centering_mode=centering.mode,
        selection_strategy=selection.selection_strategy,
        selection_hash=selection.sha256,
        output_dir=output_dir,
    )
    output = _prepare_output_directory(requested)
    csv_path = output / CSV_NAME
    try:
        devices = resolve_devices(device)
    except DeviceSelectionError as error:
        raise ExperimentError(str(error)) from error
    _validate_initial_latent_sentinel(prompts[0], contract)
    rows, failed_prompts = _compute_rows(
        prompts,
        contract=contract,
        centering=centering,
        devices=devices,
        progress_factory=progress_factory,
    )
    atomic_write_csv(csv_path, rows, CSV_COLUMNS)
    if failed_prompts:
        _remove_figure_outputs(output)
        return 1
    try:
        plot_saved_results(
            csv_path,
            output,
            selection=selection,
            contract=contract,
            centering=centering,
        )
    except BaseException:
        _remove_figure_outputs(output)
        raise
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    _validate_scientific_request(arguments.scheduler, arguments.g)
    if not arguments.plot:
        return run_experiment(
            model_name=arguments.model,
            selection_strategy=arguments.selection_strategy,
            scheduler_name=arguments.scheduler,
            guidance_scale=arguments.g,
            num_inference_steps=arguments.num_inference_steps,
            num_seeds=arguments.num_seeds,
            num_baseline_seeds=arguments.num_baseline_seeds,
            use_mu=arguments.use_mu,
            output_dir=arguments.output_dir,
            device=arguments.device,
        )

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
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_seeds=arguments.num_seeds,
    )
    centering = _load_centering_contract(
        use_mu=arguments.use_mu,
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_seeds=arguments.num_seeds,
        num_baseline_seeds=arguments.num_baseline_seeds,
        contract=contract,
        load_tensor=False,
    )
    print(f"Centering mode: {centering.mode}")
    requested = _requested_output_directory(
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_seeds=arguments.num_seeds,
        num_baseline_seeds=arguments.num_baseline_seeds,
        centering_mode=centering.mode,
        selection_strategy=selection.selection_strategy,
        selection_hash=selection.sha256,
        output_dir=arguments.output_dir,
    )
    output = _prepare_output_directory(requested)
    _remove_figure_outputs(output)
    plot_saved_results(
        output / CSV_NAME,
        output,
        selection=selection,
        contract=contract,
        centering=centering,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExperimentError, CacheIOError, GenerationCacheError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
