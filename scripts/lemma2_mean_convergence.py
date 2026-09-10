#!/usr/bin/env python3
"""Evaluate Lemma 2 with Gaussian probes and cached-trajectory diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import ExitStack
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
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
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
from utils.experiments.plotting import FIGURE_DPI  # noqa: E402
from utils.experiments.sscd import (  # noqa: E402
    SCORE_DEFINITION,
    SSCDPaths,
    sscd_configuration_hash,
)
from utils.experiments.unconditional_baseline import (  # noqa: E402
    DEFAULT_NUM_BASELINE_SEEDS,
    load_unconditional_baseline,
)
from utils.metrics.sscd import (  # noqa: E402
    SSCDMetricError,
    SSCD_SCORE_RANGE_TOLERANCE,
    validate_sscd_scores,
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
from utils.models.prediction_conversion import schedule_alpha_sigma  # noqa: E402
from utils.models.registry import get_model_spec, model_names  # noqa: E402
from utils.models.sampling import (  # noqa: E402
    encode_prompt_condition,
    make_initial_noise,
    predict_conditional_epsilon,
    validate_latent_shape,
)
from utils.models.schedulers import build_scheduler  # noqa: E402


CSV_NAME = "lemma2_mean_convergence.csv"
FIGURE_FILENAMES = (
    "lemma2_mean_convergence.png",
    "lemma2_mean_convergence.pdf",
)
TRAJECTORY_FIGURE_FILENAMES = (
    "lemma2_mean_convergence_trajectory.png",
    "lemma2_mean_convergence_trajectory.pdf",
)
CSV_COLUMNS = tuple(
    "selection_strategy selection_hash model_name scheduler_name guidance_scale "
    "num_inference_steps evaluation_source centering_mode num_baseline_seeds "
    "evaluation_generation_scientific_config_hash "
    "evaluation_schedule_sha256 "
    "baseline_generation_scientific_config_hash baseline_mu_hat_sha256 "
    "record_id original_index generation_seed step_index timestep alpha_t sigma_t "
    "snr_t latent_dimension centered_distance_rmse target_sscd trajectory_sha256 "
    "is_initial_timestep status error".split()
)


DEFAULT_NUM_SEEDS = 20
DEFAULT_GAUSSIAN_BATCH_SIZE = 8
EXPERIMENT_DIRECTORY = "lemma2_mean_convergence"
PROGRESS_DESCRIPTION = "Lemma 2 source-seed-timestep observations"
CENTERING_ZERO = "zero"
CENTERING_MU_HAT = "mu_hat"
SOURCE_GAUSSIAN = "gaussian"
SOURCE_TRAJECTORY = "trajectory"
EVALUATION_SOURCE_CHOICES = (SOURCE_GAUSSIAN, SOURCE_TRAJECTORY, "both")

FIGURE_SIZE = (4.0, 4.0)
TEXT_FONT_SIZE = 15
AXIS_NUMBER_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
OBSERVATION_LINE_ALPHA = 0.68
OBSERVATION_LINE_WIDTH = 0.75
COLORBAR_ALPHA = 1.0
COLORBAR_LABEL = "SSCD"
SSCD_COLOR_RANGE = (0.0, 1.0)
FIGURE_PAD_INCHES = 0.05
PERCENTILE_BANDS = (
    (0.05, 0.95, 0.12),
    (0.25, 0.75, 0.18),
    (0.40, 0.60, 0.26),
)
PERCENTILE_BAND_COLOR = "0.35"
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
    """Report invalid provenance, cached trajectories, or saved output."""


@dataclass(frozen=True, slots=True)
class GenerationIdentity:
    """Tensor-free identity of the seeds-0 experiment-generation cache."""

    paths: GenerationPaths
    science: Mapping[str, Any]
    scientific_hash: str
    schedule_sha256: str
    seeds: tuple[int, ...]
    latent_shape: tuple[int, int, int]
    latent_dimension: int
    stored_dtype: torch.dtype
    records_by_index: Mapping[str, CompletedGenerationRecord]

    @property
    def scheduler_name(self) -> str:
        """Return the scheduler name pinned by validated generation metadata."""

        scheduler = self.science.get("scheduler")
        name = scheduler.get("name") if isinstance(scheduler, Mapping) else None
        if name not in SCHEDULER_CHOICES:
            raise ExperimentError("experiment scheduler metadata is invalid")
        return str(name)


@dataclass(frozen=True, slots=True)
class GenerationContract:
    """Generation identity plus its exact deserialized scheduler schedule."""

    identity: GenerationIdentity
    schedule_payload: Mapping[str, Any]
    scheduler_config: Mapping[str, Any]
    init_noise_sigma: float
    timesteps: torch.Tensor
    alpha_t: torch.Tensor
    sigma_t: torch.Tensor
    snr_t: torch.Tensor

    @property
    def paths(self) -> GenerationPaths:
        return self.identity.paths

    @property
    def science(self) -> Mapping[str, Any]:
        return self.identity.science

    @property
    def scientific_hash(self) -> str:
        return self.identity.scientific_hash

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.identity.seeds

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self.identity.latent_shape

    @property
    def latent_dimension(self) -> int:
        return self.identity.latent_dimension

    @property
    def stored_dtype(self) -> torch.dtype:
        return self.identity.stored_dtype

    @property
    def records_by_index(self) -> Mapping[str, CompletedGenerationRecord]:
        return self.identity.records_by_index

    @property
    def num_inference_steps(self) -> int:
        return int(self.timesteps.numel())

    @property
    def scheduler_name(self) -> str:
        return self.identity.scheduler_name


@dataclass(frozen=True, slots=True)
class BaselineContract:
    """Validated shared unconditional baseline used for centering."""

    artifact: Any
    mu_hat: torch.Tensor | None
    mu_hat_sha256: str
    source_scientific_hash: str
    source_schedule_sha256: str
    num_baseline_seeds: int
    baseline_seeds: tuple[int, ...]
    timestep: int
    alpha_t: float
    sigma_t: float
    latent_shape: tuple[int, int, int]
    latent_dimension: int


@dataclass(frozen=True, slots=True)
class CenteringContract:
    """The requested zero or saved-model-mean centering convention."""

    mode: str
    center: torch.Tensor | None
    baseline: BaselineContract | None

    @property
    def num_baseline_seeds(self) -> int:
        return 0 if self.baseline is None else self.baseline.num_baseline_seeds

    @property
    def baseline_scientific_hash(self) -> str:
        return "" if self.baseline is None else self.baseline.source_scientific_hash

    @property
    def baseline_mu_hat_sha256(self) -> str:
        return "" if self.baseline is None else self.baseline.mu_hat_sha256


@dataclass(frozen=True, slots=True)
class _SSCDContract:
    """Validated local target-specific SSCD cache provenance."""

    paths: SSCDPaths
    configuration: Mapping[str, Any]
    configuration_hash: str


@dataclass(frozen=True, slots=True)
class _PromptMeasurement:
    """Compact result for every generation seed and scheduler step of one prompt."""

    position: int
    original_index: str
    trajectory_sha256: str
    values: np.ndarray | None
    error: str


@dataclass(frozen=True, slots=True)
class _GaussianRuntime:
    """One device's validated unconditional inference state."""

    contract: GenerationContract
    components: Any
    scheduler: Any
    empty_condition: torch.Tensor


@dataclass(frozen=True, slots=True)
class _GaussianBatchMeasurement:
    """Distances for a canonical batch of Gaussian evaluation seeds."""

    entries: tuple[tuple[int, int], ...]
    values: np.ndarray | None
    error: str


_WORKER_CONTRACT: GenerationContract | None = None
_WORKER_CENTER: torch.Tensor | None = None
_WORKER_DEVICE: torch.device | None = None
_GAUSSIAN_WORKER_RUNTIME: _GaussianRuntime | None = None
_GAUSSIAN_WORKER_CENTER: torch.Tensor | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI while retaining the established run arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate unconditional clean estimates from Gaussian probes, cached "
            "selected-prompt trajectories, or both, centered on zero or, with "
            "--use-mu, the shared model-implied mean."
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
    parser.add_argument("--scheduler", choices=SCHEDULER_CHOICES, default="ddim")
    parser.add_argument("--g", type=finite_float, default=7.5, metavar="SCALE")
    parser.add_argument(
        "--T",
        type=positive_integer,
        default=50,
        dest="num_inference_steps",
        metavar="STEPS",
        help="number of cached scheduler timesteps to evaluate (default: 50)",
    )
    parser.add_argument(
        "--N",
        type=positive_integer,
        default=DEFAULT_NUM_SEEDS,
        dest="num_seeds",
        metavar="SEEDS",
        help="cached experiment generation seeds 0..N-1 (default: 20)",
    )
    parser.add_argument(
        "--num-baseline-seeds",
        type=positive_integer,
        default=DEFAULT_NUM_BASELINE_SEEDS,
        metavar="SEEDS",
        help=("dedicated Gaussian seed count used only with --use-mu (default: 1000)"),
    )
    parser.add_argument(
        "--use-mu",
        action="store_true",
        help=(
            "center on the saved model-implied mu_hat; by default center on the "
            "exact zero tensor and do not require a baseline artifact"
        ),
    )
    parser.add_argument(
        "--evaluation-source",
        choices=EVALUATION_SOURCE_CHOICES,
        default="both",
        help=(
            "evaluate prompt-independent Gaussian probes, cached CFG trajectories, "
            "or both (default: both)"
        ),
    )
    parser.add_argument(
        "--device",
        type=device_argument,
        default="auto",
        help=(
            "auto shards Gaussian inference and cached prompt reductions across "
            "visible CUDA devices; "
            "cpu, mps, cuda, or cuda:N selects one device (default: auto)"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "validate metadata and replot the saved CSV without deserializing "
            "generation trajectories or, with --use-mu, the baseline tensor"
        ),
    )
    return parser


def _validate_scientific_request(scheduler_name: str) -> None:
    if scheduler_name not in SCHEDULER_CHOICES:
        choices = ", ".join(SCHEDULER_CHOICES)
        raise ExperimentError(f"Lemma 2 scheduler must be one of: {choices}")


def _evaluation_sources(value: str) -> tuple[str, ...]:
    """Resolve one CLI source request to its canonical CSV/figure order."""

    if value == SOURCE_GAUSSIAN:
        return (SOURCE_GAUSSIAN,)
    if value == SOURCE_TRAJECTORY:
        return (SOURCE_TRAJECTORY,)
    if value == "both":
        return (SOURCE_GAUSSIAN, SOURCE_TRAJECTORY)
    raise ExperimentError(f"unsupported evaluation source: {value!r}")


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
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[str(value)]
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
    path = paths.schedule
    if not path.is_file() or path.is_symlink():
        raise ExperimentError("saved experiment schedule is missing or unsafe")
    try:
        digest = file_sha256(path)
    except OSError as error:
        raise ExperimentError(
            f"cannot hash saved experiment schedule: {error}"
        ) from error
    if not _is_sha256(digest):
        raise ExperimentError("saved experiment schedule hash is invalid")
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
    """Load the exact frozen prompt selection requested by the experiment."""

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
    """Return every included frozen prompt once, in canonical selection order."""

    required = {
        "original_index",
        "record_id",
        "source_row_number",
        "prompt",
        "target_image_sha256",
        "include_prompt",
    }
    frame = selection.prompt_frame
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ExperimentError(
            "frozen prompt selection is missing: " + ", ".join(missing)
        )
    included_flags = frame["include_prompt"]
    if included_flags.isna().any() or not pd.api.types.is_bool_dtype(
        included_flags.dtype
    ):
        raise ExperimentError("frozen selection include_prompt must contain booleans")
    included = frame.loc[included_flags]
    if included.empty:
        raise ExperimentError("frozen selection contains no included prompts")

    rows: list[dict[str, object]] = []
    observed_indices: set[str] = set()
    observed_records: set[str] = set()
    observed_source_rows: set[int] = set()
    for raw in included.to_dict(orient="records"):
        try:
            index = safe_index(raw.get("original_index"))
        except GenerationCacheError as error:
            raise ExperimentError(
                "frozen selection contains an unsafe original index"
            ) from error
        record_id = str(raw.get("record_id", "")).strip()
        source_row = raw.get("source_row_number")
        if not record_id:
            raise ExperimentError(f"frozen prompt {index} has no record_id")
        if (
            isinstance(source_row, bool)
            or not isinstance(source_row, int)
            or source_row < 0
        ):
            raise ExperimentError(f"frozen prompt {index} has an invalid source row")
        if not isinstance(raw.get("prompt"), str):
            raise ExperimentError(f"frozen prompt {index} has invalid prompt text")
        if not _is_sha256(raw.get("target_image_sha256")):
            raise ExperimentError(f"frozen prompt {index} has an invalid target hash")
        if index in observed_indices:
            raise ExperimentError(f"frozen selection repeats prompt {index}")
        if record_id in observed_records:
            raise ExperimentError(f"frozen selection repeats record_id {record_id}")
        if source_row in observed_source_rows:
            raise ExperimentError(f"frozen selection repeats source row {source_row}")
        observed_indices.add(index)
        observed_records.add(record_id)
        observed_source_rows.add(source_row)
        row = dict(raw)
        row["original_index"] = index
        row["record_id"] = record_id
        rows.append(row)
    return tuple(rows)


def _prompt_identity(prompt_row: Mapping[str, object]) -> dict[str, object]:
    return {
        "record_id": prompt_row.get("record_id"),
        "source_row_number": int(prompt_row["source_row_number"]),
        "prompt_raw": prompt_row.get("prompt"),
        "target_image_sha256": prompt_row.get("target_image_sha256"),
    }


def _load_generation_identity(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> GenerationIdentity:
    """Validate run JSON and completion markers without loading tensors."""

    _validate_scientific_request(scheduler_name)
    project = Path(root).expanduser().resolve()
    try:
        seed_start, seed_count = validate_seed_block(0, num_seeds)
        paths = generation_paths(
            project,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=seed_count,
            seed_start=seed_start,
        )
        configuration = require_generation_run(paths)
        completed_records = list_completed_records(paths)
    except (CacheIOError, GenerationCacheError, ValueError) as error:
        raise ExperimentError(str(error)) from error
    science = configuration.get("scientific_config")
    scientific_hash = configuration.get("scientific_config_hash")
    if not isinstance(science, Mapping) or not _is_sha256(scientific_hash):
        raise ExperimentError("experiment generation provenance is invalid")

    spec = get_model_spec(model_name)
    expected_seeds = tuple(range(seed_count))
    expected = {
        "model_cli_name": model_name,
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "guidance_scale": float(guidance_scale),
        "num_inference_steps": num_inference_steps,
        "num_seeds": seed_count,
        "seeds": list(expected_seeds),
        "selection_policy": GENERATION_SELECTION_POLICY,
        "trajectory_order": "noise_to_image",
        "stored_prediction_type": "epsilon",
    }
    wrong = [
        key
        for key, expected_value in expected.items()
        if science.get(key) != expected_value
    ]
    scheduler_metadata = science.get("scheduler")
    if not isinstance(scheduler_metadata, Mapping):
        wrong.append("scheduler")
    elif scheduler_metadata.get("name") != scheduler_name:
        wrong.append("scheduler")
    if wrong:
        raise ExperimentError(
            "experiment generation cache differs at: " + ", ".join(sorted(set(wrong)))
        )

    seeds = _integer_tuple(science.get("seeds"), "experiment generation seeds")
    if seeds != expected_seeds:
        raise ExperimentError("experiment generation cache has the wrong seed order")
    shape_values = _integer_tuple(science.get("latent_shape"), "latent shape")
    if len(shape_values) != 3 or any(value <= 0 for value in shape_values):
        raise ExperimentError("generation latent shape must contain positive [C, H, W]")
    latent_shape = (shape_values[0], shape_values[1], shape_values[2])
    storage = science.get("scientific_tensor_storage")
    if not isinstance(storage, Mapping):
        raise ExperimentError("generation tensor storage metadata is invalid")
    stored_dtype = _dtype_from_name(storage.get("dtype"))
    if storage.get("device") != "cpu" or storage.get("layout") != "contiguous":
        raise ExperimentError("generation tensors must be contiguous CPU tensors")
    records_by_index = {record.original_index: record for record in completed_records}
    if len(records_by_index) != len(completed_records):
        raise ExperimentError("generation completion records repeat an index")
    schedule_sha256 = _schedule_file_sha256(paths)
    return GenerationIdentity(
        paths=paths,
        science=science,
        scientific_hash=str(scientific_hash),
        schedule_sha256=schedule_sha256,
        seeds=seeds,
        latent_shape=latent_shape,
        latent_dimension=math.prod(latent_shape),
        stored_dtype=stored_dtype,
        records_by_index=records_by_index,
    )


def _load_sscd_contract(identity: GenerationIdentity) -> _SSCDContract:
    """Load only the existing local SSCD metadata for this generation run."""

    paths = SSCDPaths(identity.paths.run_directory)
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
        "generation_scientific_config_hash": identity.scientific_hash,
        "num_seeds": len(identity.seeds),
        "seeds": list(identity.seeds),
        "model_id": identity.science.get("model_id"),
        "model_revision": identity.science.get("model_revision"),
        "vae_id": identity.science.get("vae_id"),
        "vae_revision": identity.science.get("vae_revision"),
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
    return _SSCDContract(
        paths=paths,
        configuration=configuration,
        configuration_hash=str(stored_hash),
    )


def _load_target_sscd(
    prompt_row: Mapping[str, object],
    *,
    identity: GenerationIdentity,
    contract: _SSCDContract,
    generation_marker: Mapping[str, Any],
) -> torch.Tensor:
    """Load the validated endpoint target score for every generation seed."""

    index = safe_index(prompt_row.get("original_index"))
    marker_path = contract.paths.marker_path(index)
    score_path = contract.paths.score_path(index)
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ExperimentError("target-specific SSCD marker is missing or unsafe")
    if not score_path.is_file() or score_path.is_symlink():
        raise ExperimentError("target-specific SSCD tensor is missing or unsafe")
    try:
        marker = read_json(marker_path)
    except CacheIOError as error:
        raise ExperimentError(str(error)) from error
    configured_seeds = _integer_tuple(contract.configuration.get("seeds"), "SSCD seeds")
    if configured_seeds != identity.seeds:
        raise ExperimentError("target-specific SSCD seed order differs from generation")
    latent_hashes = generation_marker.get("tensor_file_sha256")
    expected = {
        "original_index": index,
        "record_id": prompt_row.get("record_id"),
        "source_row_number": int(prompt_row["source_row_number"]),
        "prompt_raw": prompt_row.get("prompt"),
        "target_image_sha256": prompt_row.get("target_image_sha256"),
        "model_id": identity.science.get("model_id"),
        "model_revision": identity.science.get("model_revision"),
        "vae_id": identity.science.get("vae_id"),
        "vae_revision": identity.science.get("vae_revision"),
        "terminal_latent_index": -1,
        "score_shape": [len(identity.seeds)],
        "score_dtype": "float32",
        "num_seeds": len(identity.seeds),
        "seeds": list(identity.seeds),
        "similarity": SCORE_DEFINITION,
        "sscd_configuration_hash": contract.configuration_hash,
        "generation_scientific_config_hash": identity.scientific_hash,
        "generation_latent_sha256": (
            latent_hashes.get("latent") if isinstance(latent_hashes, Mapping) else None
        ),
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
    expected_score_hash = marker.get("score_sha256")
    try:
        observed_score_hash = file_sha256(score_path)
    except OSError as error:
        raise ExperimentError(
            f"cannot hash target-specific SSCD tensor: {error}"
        ) from error
    if (
        not _is_sha256(expected_score_hash)
        or observed_score_hash != expected_score_hash
    ):
        raise ExperimentError("target-specific SSCD tensor hash differs")
    try:
        scores = validate_sscd_scores(safe_torch_load(score_path), len(identity.seeds))
    except (CacheIOError, SSCDMetricError) as error:
        raise ExperimentError(str(error)) from error
    return scores.double().contiguous()


def _load_target_sscd_lookup(
    prompt_rows: Sequence[Mapping[str, object]],
    *,
    identity: GenerationIdentity,
) -> dict[tuple[str, int], float]:
    """Join each selected prompt and seed to its existing endpoint SSCD score."""

    contract = _load_sscd_contract(identity)
    lookup: dict[tuple[str, int], float] = {}
    for prompt in prompt_rows:
        index = safe_index(prompt.get("original_index"))
        generation_marker = _validate_generation_prompt(
            prompt,
            identity=identity,
            verify_file_hashes=False,
        )
        scores = _load_target_sscd(
            prompt,
            identity=identity,
            contract=contract,
            generation_marker=generation_marker,
        )
        for seed, score in zip(identity.seeds, scores.tolist(), strict=True):
            key = (index, seed)
            if key in lookup:
                raise ExperimentError(
                    "target-specific SSCD lookup repeats a prompt seed"
                )
            lookup[key] = float(score)
    return lookup


def _attach_target_sscd(
    rows: Sequence[Mapping[str, object]],
    lookup: Mapping[tuple[str, int], float],
) -> list[dict[str, object]]:
    """Attach target scores to trajectory rows and explicit NaN to Gaussian rows."""

    attached: list[dict[str, object]] = []
    for row in rows:
        values = dict(row)
        source = str(values.get("evaluation_source", ""))
        if source == SOURCE_GAUSSIAN:
            score = math.nan
        elif source == SOURCE_TRAJECTORY:
            index = safe_index(values.get("original_index"))
            try:
                seed = int(values.get("generation_seed"))
            except (TypeError, ValueError, OverflowError) as error:
                raise ExperimentError(
                    "trajectory row has an invalid generation seed"
                ) from error
            try:
                score = float(lookup[(index, seed)])
            except KeyError as error:
                raise ExperimentError(
                    f"target-specific SSCD is missing for prompt {index}, seed {seed}"
                ) from error
            if not math.isfinite(score):
                raise ExperimentError(
                    "target-specific SSCD lookup contains non-finite data"
                )
        else:
            raise ExperimentError(f"unsupported evaluation source: {source!r}")
        values["target_sscd"] = score
        attached.append({column: values[column] for column in CSV_COLUMNS})
    return attached


def _validated_schedule_tensor(
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
        or value.dtype != dtype
        or value.device.type != "cpu"
        or not value.is_contiguous()
        or value.requires_grad
        or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
    ):
        raise ExperimentError(f"saved schedule {name} violates its tensor contract")
    return value


def _load_generation_contract(identity: GenerationIdentity) -> GenerationContract:
    """Deserialize and validate the exact cached scheduler schedule."""

    if _schedule_file_sha256(identity.paths) != identity.schedule_sha256:
        raise ExperimentError("saved experiment schedule SHA-256 changed")
    try:
        schedule = safe_torch_load(identity.paths.schedule)
    except CacheIOError as error:
        raise ExperimentError(str(error)) from error
    if not isinstance(schedule, Mapping):
        raise ExperimentError("saved experiment schedule must be a mapping")
    steps = int(identity.science["num_inference_steps"])
    timesteps = _validated_schedule_tensor(
        schedule, "timesteps", length=steps, dtype=torch.int64
    )
    if bool((timesteps < 0).any()) or (
        steps > 1 and not bool((timesteps[:-1] > timesteps[1:]).all())
    ):
        raise ExperimentError(
            "saved scheduler timesteps must be strictly descending and nonnegative"
        )
    alpha_t = _validated_schedule_tensor(
        schedule, "alpha_t", length=steps, dtype=torch.float32
    )
    sigma_t = _validated_schedule_tensor(
        schedule, "sigma_t", length=steps, dtype=torch.float32
    )
    cumulative_t = _validated_schedule_tensor(
        schedule, "alphas_cumprod_t", length=steps, dtype=torch.float32
    )
    if bool((alpha_t <= 0).any()) or bool((sigma_t <= 0).any()):
        raise ExperimentError("saved scheduler coefficients must be positive")
    if not (
        torch.allclose(alpha_t.square(), cumulative_t, rtol=1e-5, atol=1e-6)
        and torch.allclose(sigma_t.square(), 1.0 - cumulative_t, rtol=1e-5, atol=1e-6)
    ):
        raise ExperimentError("saved scheduler coefficients are inconsistent")
    try:
        init_noise_sigma = float(schedule.get("init_noise_sigma"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ExperimentError(
            "saved scheduler initial-noise scale is invalid"
        ) from error
    if not math.isclose(init_noise_sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ExperimentError("Lemma 2 requires generation init_noise_sigma = 1")
    scheduler_metadata = identity.science.get("scheduler")
    if not isinstance(scheduler_metadata, Mapping):
        raise ExperimentError("experiment scheduler metadata is invalid")
    expected = {
        "scheduler_name": identity.scheduler_name,
        "scheduler_class": scheduler_metadata.get("class"),
        "native_prediction_type": identity.science.get("native_prediction_type"),
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
    }
    wrong = [
        key
        for key, expected_value in expected.items()
        if schedule.get(key) != expected_value
    ]
    if wrong:
        raise ExperimentError(
            "saved experiment schedule differs at: " + ", ".join(wrong)
        )
    scheduler_config = _normalize_scheduler_config(schedule.get("scheduler_config"))
    science_config = _normalize_scheduler_config(scheduler_metadata.get("config"))
    if canonical_hash(scheduler_config) != canonical_hash(science_config):
        raise ExperimentError("saved and configured scheduler settings differ")
    num_train_timesteps = _num_train_timesteps(scheduler_config)
    if bool((timesteps >= num_train_timesteps).any()):
        raise ExperimentError(
            "saved scheduler timestep is outside the training schedule"
        )
    if _schedule_file_sha256(identity.paths) != identity.schedule_sha256:
        raise ExperimentError("saved experiment schedule SHA-256 changed while loading")
    snr_t = alpha_t.double().square().div(sigma_t.double().square()).contiguous()
    if not bool(torch.isfinite(snr_t).all()) or bool((snr_t <= 0).any()):
        raise ExperimentError("saved alpha_t^2/sigma_t^2 values are invalid")
    return GenerationContract(
        identity=identity,
        schedule_payload=schedule,
        scheduler_config=scheduler_config,
        init_noise_sigma=init_noise_sigma,
        timesteps=timesteps,
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        snr_t=snr_t,
    )


def _load_baseline_contract(
    *,
    project_root: str | Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    identity: GenerationIdentity,
    load_tensor: bool,
) -> BaselineContract:
    """Load and cross-check the shared, disjoint Gaussian baseline."""

    try:
        artifact = load_unconditional_baseline(
            project_root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            num_baseline_seeds=num_baseline_seeds,
            load_tensor=load_tensor,
        )
    except Exception as error:
        raise ExperimentError(f"cannot load unconditional baseline: {error}") from error

    mu_hash = getattr(artifact, "mu_hat_sha256", None)
    source_hash = getattr(artifact, "source_scientific_config_hash", None)
    source_schedule_hash = getattr(artifact, "source_schedule_sha256", None)
    if (
        not _is_sha256(mu_hash)
        or not _is_sha256(source_hash)
        or not _is_sha256(source_schedule_hash)
    ):
        raise ExperimentError("unconditional baseline hashes are invalid")
    if source_schedule_hash != identity.schedule_sha256:
        raise ExperimentError("baseline and experiment schedule SHA-256 digests differ")
    baseline_seeds = _integer_tuple(
        getattr(artifact, "baseline_seeds", None), "baseline source seeds"
    )
    expected_baseline_seeds = tuple(range(num_seeds, num_seeds + num_baseline_seeds))
    if (
        getattr(artifact, "baseline_seed_start", None) != num_seeds
        or getattr(artifact, "num_baseline_seeds", None) != num_baseline_seeds
        or baseline_seeds != expected_baseline_seeds
    ):
        raise ExperimentError(
            "unconditional baseline must use dedicated seeds N..N+B-1"
        )
    if set(baseline_seeds).intersection(identity.seeds):
        raise ExperimentError("baseline and experiment generation seeds overlap")
    try:
        timestep = int(getattr(artifact, "timestep"))
        alpha_t = float(getattr(artifact, "alpha_t"))
        sigma_t = float(getattr(artifact, "sigma_t"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ExperimentError(
            "unconditional baseline coefficients are invalid"
        ) from error
    if not (
        math.isfinite(alpha_t)
        and alpha_t > 0.0
        and math.isfinite(sigma_t)
        and sigma_t > 0.0
    ):
        raise ExperimentError("unconditional baseline coefficients are invalid")
    shape_values = _integer_tuple(
        getattr(artifact, "latent_shape", None), "baseline latent shape"
    )
    if tuple(shape_values) != identity.latent_shape:
        raise ExperimentError("baseline latent shape differs from experiment model")

    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ExperimentError("unconditional baseline metadata is invalid")
    required = {
        "model_name",
        "model_id",
        "model_revision",
        "timestep",
        "alpha_T",
        "sigma_T",
        "baseline_seed_start",
        "baseline_seeds",
        "num_baseline_seeds",
        "mu_hat_norm_rmse",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ExperimentError(
            "unconditional baseline metadata is missing: " + ", ".join(missing)
        )
    if metadata.get("model_name") != model_name:
        raise ExperimentError("unconditional baseline model differs")
    provenance_wrong = [
        name
        for name in ("model_id", "model_revision")
        if metadata.get(name) != identity.science.get(name)
    ]
    if provenance_wrong:
        raise ExperimentError(
            "unconditional baseline model provenance differs at: "
            + ", ".join(provenance_wrong)
        )
    metadata_seeds = _integer_tuple(
        metadata["baseline_seeds"], "baseline metadata seeds"
    )
    try:
        metadata_seed_start = int(metadata["baseline_seed_start"])
        metadata_seed_count = int(metadata["num_baseline_seeds"])
        saved_norm = float(metadata["mu_hat_norm_rmse"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ExperimentError("unconditional baseline summary is invalid") from error
    if (
        metadata_seed_start != num_seeds
        or metadata_seed_count != num_baseline_seeds
        or metadata_seeds != expected_baseline_seeds
        or not math.isfinite(saved_norm)
        or saved_norm < 0.0
    ):
        raise ExperimentError("unconditional baseline summary is invalid")

    mu_hat = getattr(artifact, "mu_hat", None)
    if load_tensor:
        if (
            not isinstance(mu_hat, torch.Tensor)
            or tuple(mu_hat.shape) != identity.latent_shape
            or mu_hat.dtype != torch.float64
            or mu_hat.device.type != "cpu"
            or not mu_hat.is_contiguous()
            or mu_hat.requires_grad
            or not bool(torch.isfinite(mu_hat).all())
        ):
            raise ExperimentError("saved mu_hat violates its tensor contract")
        observed_norm = float(mu_hat.square().mean().sqrt().item())
        if not math.isclose(observed_norm, saved_norm, rel_tol=1e-12, abs_tol=1e-12):
            raise ExperimentError("saved mu_hat norm differs from baseline metadata")
    elif mu_hat is not None:
        raise ExperimentError("plot-only baseline loading deserialized mu_hat")
    return BaselineContract(
        artifact=artifact,
        mu_hat=mu_hat,
        mu_hat_sha256=str(mu_hash),
        source_scientific_hash=str(source_hash),
        source_schedule_sha256=str(source_schedule_hash),
        num_baseline_seeds=num_baseline_seeds,
        baseline_seeds=baseline_seeds,
        timestep=timestep,
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        latent_shape=(shape_values[0], shape_values[1], shape_values[2]),
        latent_dimension=math.prod(shape_values),
    )


def _validate_baseline_against_schedule(
    baseline: BaselineContract, contract: GenerationContract
) -> None:
    if baseline.timestep != int(contract.timesteps[0]):
        raise ExperimentError(
            "baseline timestep differs from the initial cached scheduler timestep"
        )
    if baseline.source_schedule_sha256 != contract.identity.schedule_sha256:
        raise ExperimentError("baseline and experiment schedule SHA-256 digests differ")
    for name, observed, expected in (
        ("alpha_T", baseline.alpha_t, float(contract.alpha_t[0])),
        ("sigma_T", baseline.sigma_t, float(contract.sigma_t[0])),
    ):
        if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ExperimentError(f"baseline {name} differs from experiment schedule")


def _load_centering_contract(
    *,
    use_mu: bool,
    project_root: str | Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    identity: GenerationIdentity,
    load_tensor: bool,
) -> CenteringContract:
    """Create an exact-zero center or load the requested saved mean."""

    if not use_mu:
        center = (
            torch.zeros(identity.latent_shape, dtype=torch.float64)
            if load_tensor
            else None
        )
        return CenteringContract(
            mode=CENTERING_ZERO,
            center=center,
            baseline=None,
        )
    baseline = _load_baseline_contract(
        project_root=project_root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        num_baseline_seeds=num_baseline_seeds,
        identity=identity,
        load_tensor=load_tensor,
    )
    return CenteringContract(
        mode=CENTERING_MU_HAT,
        center=baseline.mu_hat,
        baseline=baseline,
    )


def _validate_centering_against_schedule(
    centering: CenteringContract, contract: GenerationContract
) -> None:
    if centering.mode == CENTERING_ZERO:
        if centering.baseline is not None:
            raise ExperimentError("zero centering unexpectedly has a baseline")
        if centering.center is None:
            raise ExperimentError("zero centering did not construct its center")
        center = _validate_tensor(
            centering.center,
            shape=contract.latent_shape,
            dtype=torch.float64,
            label="zero center",
        )
        if bool(torch.count_nonzero(center)):
            raise ExperimentError("zero centering must use the exact zero tensor")
        return
    if centering.mode != CENTERING_MU_HAT or centering.baseline is None:
        raise ExperimentError("invalid centering contract")
    _validate_baseline_against_schedule(centering.baseline, contract)


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


def _revision_resolver(contract: GenerationContract) -> Callable[[str], str]:
    revisions = {
        str(contract.science["model_id"]): contract.science.get("model_revision"),
        str(contract.science["vae_id"]): contract.science.get("vae_revision"),
    }

    def resolve(repository_id: str) -> str:
        revision = revisions.get(repository_id)
        if not isinstance(revision, str) or not revision:
            raise ExperimentError(
                f"generation cache has no pinned revision for {repository_id}"
            )
        return revision

    return resolve


def _validate_loaded_components(components: Any, contract: GenerationContract) -> None:
    expected = {
        "model_id": contract.science.get("model_id"),
        "model_revision": contract.science.get("model_revision"),
        "vae_id": contract.science.get("vae_id"),
        "vae_revision": contract.science.get("vae_revision"),
    }
    observed = {
        "model_id": getattr(components, "model_id", None),
        "model_revision": getattr(components, "model_revision", None),
        "vae_id": getattr(components, "vae_id", None),
        "vae_revision": getattr(components, "vae_revision", None),
    }
    wrong = [name for name, value in expected.items() if observed.get(name) != value]
    if getattr(components, "inference_dtype", None) != contract.stored_dtype:
        wrong.append("inference_dtype")
    if getattr(components, "package_versions", None) != contract.science.get(
        "package_versions"
    ):
        wrong.append("package_versions")
    if wrong:
        raise ExperimentError(
            "loaded components differ from the generation cache at: "
            + ", ".join(sorted(set(wrong)))
        )
    if (
        validate_latent_shape(components.unet, expected_shape=contract.latent_shape)
        != contract.latent_shape
    ):
        raise ExperimentError(
            "loaded UNet latent shape differs from the generation cache"
        )


def _offload_unused_vae(components: Any) -> None:
    selected = torch.device(components.device)
    if selected.type == "cpu":
        return
    move = getattr(components.vae, "to", None)
    if not callable(move):
        raise ExperimentError("loaded VAE has no device-transfer interface")
    move(device=torch.device("cpu"), dtype=torch.float32)
    if selected.type == "cuda":
        torch.cuda.empty_cache()


def _validate_active_scheduler(components: Any, contract: GenerationContract) -> Any:
    result = build_scheduler(components.original_scheduler, contract.scheduler_name)
    scheduler = result.scheduler
    setter = getattr(scheduler, "set_timesteps", None)
    if not callable(setter):
        raise ExperimentError("active scheduler has no timestep interface")
    setter(contract.num_inference_steps, device=torch.device(components.device))
    active_timesteps = torch.as_tensor(scheduler.timesteps).detach().cpu().long()
    if not torch.equal(active_timesteps, contract.timesteps):
        raise ExperimentError(
            "active scheduler timesteps differ from the saved generation schedule"
        )
    active_coefficients = schedule_alpha_sigma(
        scheduler, active_timesteps, dtype=torch.float32
    )
    saved_coefficients = (contract.alpha_t, contract.sigma_t)
    if any(
        not torch.equal(active, saved)
        for active, saved in zip(
            active_coefficients[:2], saved_coefficients, strict=True
        )
    ):
        raise ExperimentError(
            "active scheduler coefficients differ from the saved generation schedule"
        )
    if canonical_hash(_normalize_scheduler_config(result.config)) != canonical_hash(
        contract.scheduler_config
    ):
        raise ExperimentError(
            "active scheduler configuration differs from the saved generation schedule"
        )
    try:
        active_sigma = float(getattr(scheduler, "init_noise_sigma"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ExperimentError(
            "active scheduler initial-noise scale is invalid"
        ) from error
    if not math.isclose(
        active_sigma, contract.init_noise_sigma, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ExperimentError(
            "active scheduler initial-noise scale differs from the generation cache"
        )
    return scheduler


def _build_gaussian_runtime(
    contract: GenerationContract,
    *,
    device: str | torch.device,
    worker_count: int,
) -> _GaussianRuntime:
    configure_worker_cpu_threads(worker_count)
    concrete_device = torch.device(device)
    if concrete_device.type == "cuda":
        if concrete_device.index is None:
            raise ExperimentError(
                "Gaussian CUDA workers require concrete device indices"
            )
        torch.cuda.set_device(concrete_device)
    runtime = select_runtime(concrete_device, warn_without_cuda=False)
    components = load_model_components(
        get_model_spec(str(contract.science["model_cli_name"])),
        runtime=runtime,
        revision_resolver=_revision_resolver(contract),
    )
    scheduler = _validate_active_scheduler(components, contract)
    preflight_model_components(
        components,
        scheduler=scheduler,
        num_inference_steps=contract.num_inference_steps,
        active_cuda_index=(
            concrete_device.index if concrete_device.type == "cuda" else None
        ),
    )
    _validate_loaded_components(components, contract)
    _offload_unused_vae(components)
    components = compile_loaded_unet(components)
    empty_condition = encode_prompt_condition(
        "",
        components.tokenizer,
        components.text_encoder,
        components.device,
        components.inference_dtype,
    )
    return _GaussianRuntime(
        contract=contract,
        components=components,
        scheduler=scheduler,
        empty_condition=empty_condition,
    )


def _gaussian_seed_batches(
    entries: Sequence[tuple[int, int]],
    batch_size: int = DEFAULT_GAUSSIAN_BATCH_SIZE,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ExperimentError("Gaussian sample batch size must be a positive integer")
    return tuple(
        tuple(entries[start : start + batch_size])
        for start in range(0, len(entries), batch_size)
    )


def _measure_gaussian_batch(
    runtime: _GaussianRuntime,
    entries: Sequence[tuple[int, int]],
    center: torch.Tensor,
) -> _GaussianBatchMeasurement:
    canonical_entries = tuple((int(position), int(seed)) for position, seed in entries)
    if not canonical_entries:
        raise ExperimentError("Gaussian seed batch must not be empty")
    contract = runtime.contract
    center = _validate_tensor(
        center,
        shape=contract.latent_shape,
        dtype=torch.float64,
        label="centering tensor",
    )
    seeds = tuple(seed for _position, seed in canonical_entries)
    raw = make_initial_noise(seeds, contract.latent_shape)
    expected_shape = (len(seeds), *contract.latent_shape)
    if (
        not isinstance(raw, torch.Tensor)
        or tuple(raw.shape) != expected_shape
        or raw.dtype != torch.float32
        or raw.device.type != "cpu"
        or not raw.is_contiguous()
        or raw.requires_grad
        or not bool(torch.isfinite(raw).all())
    ):
        raise ExperimentError("generated Gaussian evaluation noise is invalid")

    conceptual_cpu = raw.mul(contract.init_noise_sigma).contiguous()
    selected = torch.device(runtime.components.device)
    conceptual = conceptual_cpu.to(device=selected, dtype=torch.float32)
    model_samples = conceptual.to(dtype=runtime.components.inference_dtype)
    values = np.empty(
        (len(canonical_entries), contract.num_inference_steps), dtype=np.float64
    )
    with torch.inference_mode():
        for step_index in range(contract.num_inference_steps):
            timestep = int(contract.timesteps[step_index])
            epsilon = predict_conditional_epsilon(
                samples=model_samples,
                timestep=timestep,
                condition=runtime.empty_condition,
                unet=runtime.components.unet,
                scheduler=runtime.scheduler,
                conversion_sample=conceptual,
            )
            if (
                not isinstance(epsilon, torch.Tensor)
                or tuple(epsilon.shape) != expected_shape
                or epsilon.device != conceptual.device
                or epsilon.dtype != conceptual.dtype
                or not bool(torch.isfinite(epsilon).all())
            ):
                raise ExperimentError("unconditional Gaussian epsilon is invalid")
            estimate = (
                conceptual_cpu.double()
                - float(contract.sigma_t[step_index]) * epsilon.detach().cpu().double()
            ).div(float(contract.alpha_t[step_index]))
            distances = (
                (estimate - center.unsqueeze(0))
                .square()
                .flatten(start_dim=1)
                .mean(dim=1)
                .sqrt()
            )
            if not bool(torch.isfinite(distances).all()) or bool(
                (distances < 0.0).any()
            ):
                raise ExperimentError(
                    f"Gaussian centered distances are invalid at step {step_index}"
                )
            values[:, step_index] = distances.numpy()
    if not values.flags.c_contiguous or not np.isfinite(values).all():
        raise ExperimentError("Gaussian evaluation returned invalid distances")
    return _GaussianBatchMeasurement(canonical_entries, values, "")


def _is_cuda_out_of_memory(error: RuntimeError, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return (
        isinstance(error, out_of_memory_type)
        or "out of memory" in str(error).casefold()
    )


def _measure_gaussian_batch_adaptive(
    runtime: _GaussianRuntime,
    entries: Sequence[tuple[int, int]],
    center: torch.Tensor,
) -> _GaussianBatchMeasurement:
    canonical = tuple(entries)
    try:
        return _measure_gaussian_batch(runtime, canonical, center)
    except RuntimeError as error:
        device = torch.device(runtime.components.device)
        if len(canonical) == 1 or not _is_cuda_out_of_memory(error, device):
            raise
        torch.cuda.empty_cache()
        middle = len(canonical) // 2
        left = _measure_gaussian_batch_adaptive(runtime, canonical[:middle], center)
        right = _measure_gaussian_batch_adaptive(runtime, canonical[middle:], center)
        if left.error or right.error or left.values is None or right.values is None:
            raise ExperimentError("adaptive Gaussian batch returned an invalid result")
        return _GaussianBatchMeasurement(
            entries=left.entries + right.entries,
            values=np.concatenate((left.values, right.values), axis=0),
            error="",
        )


def _measure_gaussian_batch_safely(
    runtime: _GaussianRuntime,
    entries: Sequence[tuple[int, int]],
    center: torch.Tensor,
) -> _GaussianBatchMeasurement:
    canonical = tuple(entries)
    try:
        return _measure_gaussian_batch_adaptive(runtime, canonical, center)
    except Exception as error:
        return _GaussianBatchMeasurement(
            entries=canonical,
            values=None,
            error=_error_message(error),
        )


def _initialize_gaussian_worker(
    contract: GenerationContract,
    center: torch.Tensor,
    device_string: str,
    worker_count: int,
) -> None:
    global _GAUSSIAN_WORKER_CENTER, _GAUSSIAN_WORKER_RUNTIME
    _GAUSSIAN_WORKER_CENTER = _validate_tensor(
        center,
        shape=contract.latent_shape,
        dtype=torch.float64,
        label="centering tensor",
    )
    _GAUSSIAN_WORKER_RUNTIME = _build_gaussian_runtime(
        contract,
        device=device_string,
        worker_count=worker_count,
    )


def _gaussian_worker_task(
    entries: tuple[tuple[int, int], ...],
) -> _GaussianBatchMeasurement:
    if _GAUSSIAN_WORKER_RUNTIME is None or _GAUSSIAN_WORKER_CENTER is None:
        raise ExperimentError("Lemma 2 Gaussian worker was not initialized")
    return _measure_gaussian_batch_safely(
        _GAUSSIAN_WORKER_RUNTIME,
        entries,
        _GAUSSIAN_WORKER_CENTER,
    )


def _trajectory_sha256(marker: Mapping[str, object]) -> str:
    """Hash the marker-pinned latent and noise-prediction file digests."""

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


def _validate_generation_prompt(
    prompt_row: Mapping[str, object],
    *,
    identity: GenerationIdentity,
    verify_file_hashes: bool,
) -> Mapping[str, Any]:
    """Validate one selected prompt's marker and requested trajectory artifacts."""

    index = safe_index(prompt_row.get("original_index"))
    if index not in identity.records_by_index:
        raise ExperimentError(f"selected prompt has no generation record: {index}")
    validation = validate_generation_record(
        identity.paths,
        index,
        expected_scientific_hash=identity.scientific_hash,
        expected_record_identity=_prompt_identity(prompt_row),
        load_tensors=False,
        tensor_names=("latent", "noise_prediction"),
        require_preview=False,
        verify_file_hashes=verify_file_hashes,
    )
    if not validation.valid or validation.metadata is None:
        raise ExperimentError(
            f"invalid cached record for {index}: " + "; ".join(validation.errors)
        )
    marker = validation.metadata
    scheduler = identity.science.get("scheduler")
    scheduler_class = scheduler.get("class") if isinstance(scheduler, Mapping) else None
    expected = {
        "model_cli_name": identity.science.get("model_cli_name"),
        "dataset_model": identity.science.get("dataset_model"),
        "model_id": identity.science.get("model_id"),
        "model_revision": identity.science.get("model_revision"),
        "vae_id": identity.science.get("vae_id"),
        "vae_revision": identity.science.get("vae_revision"),
        "scheduler_name": identity.scheduler_name,
        "scheduler_class": scheduler_class,
        "guidance_scale": identity.science.get("guidance_scale"),
        "num_inference_steps": identity.science.get("num_inference_steps"),
        "num_seeds": len(identity.seeds),
        "seeds": list(identity.seeds),
        "latent_shape": list(identity.latent_shape),
        "inference_dtype": str(identity.stored_dtype).removeprefix("torch."),
        "native_prediction_type": identity.science.get("native_prediction_type"),
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
        "scientific_config_hash": identity.scientific_hash,
        "latent_path": f"latent/{index}.pt",
        "noise_prediction_path": f"noise_pred/{index}.pt",
        "schedule_path": "schedule.pt",
    }
    wrong = [key for key, value in expected.items() if marker.get(key) != value]
    if wrong:
        raise ExperimentError(
            f"generation marker for {index} differs at: " + ", ".join(wrong)
        )
    expected_shapes = {
        "latent": [
            len(identity.seeds),
            int(identity.science["num_inference_steps"]) + 1,
            *identity.latent_shape,
        ],
        "unconditional_noise_predictions": [
            len(identity.seeds),
            int(identity.science["num_inference_steps"]),
            *identity.latent_shape,
        ],
        "conditional_noise_predictions": [
            len(identity.seeds),
            int(identity.science["num_inference_steps"]),
            *identity.latent_shape,
        ],
    }
    dtype_name = str(identity.stored_dtype).removeprefix("torch.")
    expected_dtypes = {
        "latent": dtype_name,
        "unconditional_noise_predictions": dtype_name,
        "conditional_noise_predictions": dtype_name,
    }
    shapes = marker.get("tensor_shapes")
    dtypes = marker.get("tensor_dtypes")
    if not isinstance(shapes, Mapping) or any(
        shapes.get(name) != value for name, value in expected_shapes.items()
    ):
        raise ExperimentError(f"generation marker for {index} has wrong tensor shapes")
    if not isinstance(dtypes, Mapping) or any(
        dtypes.get(name) != value for name, value in expected_dtypes.items()
    ):
        raise ExperimentError(f"generation marker for {index} has wrong tensor dtypes")
    _trajectory_sha256(marker)
    return marker


def _load_cached_trajectory(
    prompt_row: Mapping[str, object],
    contract: GenerationContract,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load full cached x_t and unconditional-epsilon trajectories once."""

    marker = _validate_generation_prompt(
        prompt_row,
        identity=contract.identity,
        verify_file_hashes=True,
    )
    index = safe_index(prompt_row.get("original_index"))
    try:
        latent_value = safe_torch_load(contract.paths.latent_path(index))
        prediction_value = safe_torch_load(contract.paths.noise_prediction_path(index))
    except CacheIOError as error:
        raise ExperimentError(str(error)) from error
    latent = _validate_tensor(
        latent_value,
        shape=(
            len(contract.seeds),
            contract.num_inference_steps + 1,
            *contract.latent_shape,
        ),
        dtype=contract.stored_dtype,
        label=f"latent trajectory for {index}",
    )
    if not isinstance(prediction_value, tuple) or len(prediction_value) != 2:
        raise ExperimentError(
            f"cached noise prediction for {index} is not a two-tensor tuple"
        )
    prediction_shape = (
        len(contract.seeds),
        contract.num_inference_steps,
        *contract.latent_shape,
    )
    unconditional = _validate_tensor(
        prediction_value[0],
        shape=prediction_shape,
        dtype=contract.stored_dtype,
        label=f"unconditional epsilon trajectory for {index}",
    )
    _validate_tensor(
        prediction_value[1],
        shape=prediction_shape,
        dtype=contract.stored_dtype,
        label=f"conditional epsilon trajectory for {index}",
    )
    return latent, unconditional, _trajectory_sha256(marker)


def _centered_trajectory_distances(
    latents: torch.Tensor,
    unconditional_epsilon: torch.Tensor,
    center: torch.Tensor,
    *,
    contract: GenerationContract,
    device: str | torch.device,
) -> np.ndarray:
    """Evaluate cached latent[:, k] with cached epsilon_empty[:, k] for every k."""

    _validate_tensor(
        latents,
        shape=(
            len(contract.seeds),
            contract.num_inference_steps + 1,
            *contract.latent_shape,
        ),
        dtype=contract.stored_dtype,
        label="latent trajectory",
    )
    _validate_tensor(
        unconditional_epsilon,
        shape=(
            len(contract.seeds),
            contract.num_inference_steps,
            *contract.latent_shape,
        ),
        dtype=contract.stored_dtype,
        label="unconditional epsilon trajectory",
    )
    _validate_tensor(
        center,
        shape=contract.latent_shape,
        dtype=torch.float64,
        label="centering tensor",
    )
    selected = torch.device(device)
    reduction_device = torch.device("cpu") if selected.type == "mps" else selected
    values = np.empty(
        (len(contract.seeds), contract.num_inference_steps), dtype=np.float64
    )
    center_batch = center.to(device=reduction_device, dtype=torch.float64).unsqueeze(0)
    with torch.inference_mode():
        for step_index in range(contract.num_inference_steps):
            alpha = float(contract.alpha_t[step_index])
            sigma = float(contract.sigma_t[step_index])
            # Generation stores x_t before the kth scheduler step at latent[:, k]
            # and its corresponding empty-branch epsilon at prediction[:, k].
            x_t = latents[:, step_index].to(
                device=reduction_device, dtype=torch.float64
            )
            epsilon = unconditional_epsilon[:, step_index].to(
                device=reduction_device, dtype=torch.float64
            )
            estimate = (x_t - sigma * epsilon) / alpha
            distances = (
                (estimate - center_batch)
                .square()
                .flatten(start_dim=1)
                .mean(dim=1)
                .sqrt()
            )
            if not bool(torch.isfinite(distances).all()) or bool(
                (distances < 0.0).any()
            ):
                raise ExperimentError(
                    f"centered distances are invalid at step {step_index}"
                )
            values[:, step_index] = distances.cpu().numpy()
    if not values.flags.c_contiguous or not np.isfinite(values).all():
        raise ExperimentError("centered prompt trajectory is incomplete")
    return values


def _error_message(error: Exception) -> str:
    detail = " ".join(str(error).splitlines()).strip()
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _measure_prompt_compact(
    position: int,
    prompt_row: Mapping[str, object],
    *,
    contract: GenerationContract,
    center: torch.Tensor,
    device: str | torch.device,
) -> _PromptMeasurement:
    latents, unconditional, trajectory_hash = _load_cached_trajectory(
        prompt_row, contract
    )
    values = _centered_trajectory_distances(
        latents,
        unconditional,
        center,
        contract=contract,
        device=device,
    )
    return _PromptMeasurement(
        position=position,
        original_index=safe_index(prompt_row.get("original_index")),
        trajectory_sha256=trajectory_hash,
        values=values,
        error="",
    )


def _measure_prompt_safely(
    position: int,
    prompt_row: Mapping[str, object],
    *,
    contract: GenerationContract,
    center: torch.Tensor,
    device: str | torch.device,
) -> _PromptMeasurement:
    try:
        return _measure_prompt_compact(
            position,
            prompt_row,
            contract=contract,
            center=center,
            device=device,
        )
    except Exception as error:
        return _PromptMeasurement(
            position=position,
            original_index=str(prompt_row.get("original_index", "")),
            trajectory_sha256="",
            values=None,
            error=_error_message(error),
        )


def _initialize_prompt_worker(
    contract: GenerationContract,
    center: torch.Tensor,
    device_string: str,
    worker_count: int,
) -> None:
    global _WORKER_CONTRACT, _WORKER_CENTER, _WORKER_DEVICE
    configure_worker_cpu_threads(worker_count)
    selected = torch.device(device_string)
    if selected.type == "cuda":
        if selected.index is None:
            raise ExperimentError("worker CUDA device must have a concrete index")
        torch.cuda.set_device(selected)
    _WORKER_CONTRACT = contract
    _WORKER_CENTER = _validate_tensor(
        center,
        shape=contract.latent_shape,
        dtype=torch.float64,
        label="centering tensor",
    )
    _WORKER_DEVICE = selected


def _prompt_worker_task(
    position: int, prompt_row: Mapping[str, object]
) -> _PromptMeasurement:
    if _WORKER_CONTRACT is None or _WORKER_CENTER is None or _WORKER_DEVICE is None:
        raise ExperimentError("Lemma 2 cache worker was not initialized")
    return _measure_prompt_safely(
        position,
        prompt_row,
        contract=_WORKER_CONTRACT,
        center=_WORKER_CENTER,
        device=_WORKER_DEVICE,
    )


def _canonical_measurements(
    measurements: Sequence[_PromptMeasurement], prompt_count: int
) -> tuple[_PromptMeasurement, ...]:
    by_position: dict[int, _PromptMeasurement] = {}
    for result in measurements:
        if result.position in by_position or not 0 <= result.position < prompt_count:
            raise ExperimentError(
                "prompt results contain duplicate or invalid positions"
            )
        by_position[result.position] = result
    missing = sorted(set(range(prompt_count)) - set(by_position))
    if missing:
        raise ExperimentError("prompt results are missing canonical positions")
    return tuple(by_position[position] for position in range(prompt_count))


def _expand_measurements(
    measurements: Sequence[_PromptMeasurement],
    prompt_rows: Sequence[Mapping[str, object]],
    *,
    selection: TargetPairSelection,
    contract: GenerationContract,
    centering: CenteringContract,
) -> tuple[list[dict[str, object]], int]:
    """Emit the canonical prompt-major, seed-major, timestep-major CSV grid."""

    canonical = _canonical_measurements(measurements, len(prompt_rows))
    common = {
        "selection_strategy": selection.selection_strategy,
        "selection_hash": selection.sha256,
        "model_name": selection.model_name,
        "scheduler_name": contract.scheduler_name,
        "guidance_scale": float(selection.guidance_scale),
        "num_inference_steps": contract.num_inference_steps,
        "evaluation_source": SOURCE_TRAJECTORY,
        "centering_mode": centering.mode,
        "num_baseline_seeds": centering.num_baseline_seeds,
        "evaluation_generation_scientific_config_hash": contract.scientific_hash,
        "evaluation_schedule_sha256": contract.identity.schedule_sha256,
        "baseline_generation_scientific_config_hash": centering.baseline_scientific_hash,
        "baseline_mu_hat_sha256": centering.baseline_mu_hat_sha256,
        "latent_dimension": contract.latent_dimension,
    }
    rows: list[dict[str, object]] = []
    failed = 0
    expected_shape = (len(contract.seeds), contract.num_inference_steps)
    for position, measurement in enumerate(canonical):
        prompt = prompt_rows[position]
        expected_index = safe_index(prompt.get("original_index"))
        if measurement.original_index != expected_index:
            raise ExperimentError("prompt result identity differs from selection order")
        if measurement.error:
            if measurement.values is not None or measurement.trajectory_sha256:
                raise ExperimentError("failed prompt result contains measurements")
        elif (
            not isinstance(measurement.values, np.ndarray)
            or measurement.values.shape != expected_shape
            or measurement.values.dtype != np.float64
            or not measurement.values.flags.c_contiguous
            or not np.isfinite(measurement.values).all()
            or bool((measurement.values < 0.0).any())
            or not _is_sha256(measurement.trajectory_sha256)
        ):
            raise ExperimentError("successful prompt result is invalid")
        for seed_position, seed in enumerate(contract.seeds):
            for step_index in range(contract.num_inference_steps):
                error = measurement.error
                if error:
                    failed += 1
                values = {
                    **common,
                    "record_id": str(prompt["record_id"]),
                    "original_index": expected_index,
                    "generation_seed": seed,
                    "step_index": step_index,
                    "timestep": int(contract.timesteps[step_index]),
                    "alpha_t": float(contract.alpha_t[step_index]),
                    "sigma_t": float(contract.sigma_t[step_index]),
                    "snr_t": float(contract.snr_t[step_index]),
                    "centered_distance_rmse": (
                        math.nan
                        if error
                        else float(measurement.values[seed_position, step_index])
                    ),
                    "target_sscd": math.nan,
                    "trajectory_sha256": measurement.trajectory_sha256,
                    "is_initial_timestep": step_index == 0,
                    "status": "error" if error else "ok",
                    "error": error,
                }
                rows.append({column: values[column] for column in CSV_COLUMNS})
    return rows, failed


def _expand_gaussian_measurements(
    measurements: Sequence[_GaussianBatchMeasurement],
    *,
    selection: TargetPairSelection,
    contract: GenerationContract,
    centering: CenteringContract,
) -> tuple[list[dict[str, object]], int]:
    """Emit one canonical seed-major, timestep-major Gaussian grid."""

    expected_entries = tuple(enumerate(contract.seeds))
    by_position: dict[int, tuple[np.ndarray | None, str]] = {}
    for measurement in measurements:
        if not measurement.entries:
            raise ExperimentError("Gaussian result contains an empty seed batch")
        expected_shape = (len(measurement.entries), contract.num_inference_steps)
        if measurement.error:
            if measurement.values is not None:
                raise ExperimentError("failed Gaussian result contains measurements")
        elif (
            not isinstance(measurement.values, np.ndarray)
            or measurement.values.shape != expected_shape
            or measurement.values.dtype != np.float64
            or not measurement.values.flags.c_contiguous
            or not np.isfinite(measurement.values).all()
            or bool((measurement.values < 0.0).any())
        ):
            raise ExperimentError("successful Gaussian result is invalid")
        for local_position, (position, seed) in enumerate(measurement.entries):
            if (
                position in by_position
                or not 0 <= position < len(expected_entries)
                or expected_entries[position] != (position, seed)
            ):
                raise ExperimentError(
                    "Gaussian results contain duplicate or invalid seed positions"
                )
            value = (
                None
                if measurement.values is None
                else measurement.values[local_position].copy()
            )
            by_position[position] = (value, measurement.error)
    if set(by_position) != set(range(len(expected_entries))):
        raise ExperimentError("Gaussian results are missing canonical seed positions")

    common = {
        "selection_strategy": selection.selection_strategy,
        "selection_hash": selection.sha256,
        "model_name": selection.model_name,
        "scheduler_name": contract.scheduler_name,
        "guidance_scale": float(selection.guidance_scale),
        "num_inference_steps": contract.num_inference_steps,
        "evaluation_source": SOURCE_GAUSSIAN,
        "centering_mode": centering.mode,
        "num_baseline_seeds": centering.num_baseline_seeds,
        "evaluation_generation_scientific_config_hash": contract.scientific_hash,
        "evaluation_schedule_sha256": contract.identity.schedule_sha256,
        "baseline_generation_scientific_config_hash": centering.baseline_scientific_hash,
        "baseline_mu_hat_sha256": centering.baseline_mu_hat_sha256,
        "latent_dimension": contract.latent_dimension,
    }
    rows: list[dict[str, object]] = []
    failed = 0
    for position, seed in expected_entries:
        values, error = by_position[position]
        for step_index in range(contract.num_inference_steps):
            if error:
                failed += 1
            row = {
                **common,
                "record_id": "",
                "original_index": "",
                "generation_seed": seed,
                "step_index": step_index,
                "timestep": int(contract.timesteps[step_index]),
                "alpha_t": float(contract.alpha_t[step_index]),
                "sigma_t": float(contract.sigma_t[step_index]),
                "snr_t": float(contract.snr_t[step_index]),
                "centered_distance_rmse": (
                    math.nan if error else float(values[step_index])
                ),
                "target_sscd": math.nan,
                "trajectory_sha256": "",
                "is_initial_timestep": step_index == 0,
                "status": "error" if error else "ok",
                "error": error,
            }
            rows.append({column: row[column] for column in CSV_COLUMNS})
    return rows, failed


def _compute_gaussian_rows(
    *,
    selection: TargetPairSelection,
    contract: GenerationContract,
    centering: CenteringContract,
    devices: Sequence[torch.device],
    progress_factory: Callable[..., Any] = tqdm,
    progress: Any | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Evaluate the same N standard-Gaussian inputs at every saved timestep."""

    if not devices:
        raise ExperimentError("at least one execution device is required")
    if centering.center is None:
        raise ExperimentError("centering mode did not construct its tensor")
    center = _validate_tensor(
        centering.center,
        shape=contract.latent_shape,
        dtype=torch.float64,
        label="centering tensor",
    )
    entries = tuple(enumerate(contract.seeds))
    worker_count = worker_count_for_tasks(devices, len(entries))
    active_devices = tuple(devices[:worker_count])
    use_parallel = worker_count > 1
    if use_parallel and not all(
        device.type == "cuda" and device.index is not None for device in active_devices
    ):
        raise ExperimentError("multiple Gaussian workers require concrete CUDA devices")

    measurements: list[_GaussianBatchMeasurement] = []
    cells_per_seed = contract.num_inference_steps
    with ExitStack() as progress_stack:
        if progress is None:
            progress = progress_stack.enter_context(
                progress_factory(
                    total=len(entries) * cells_per_seed,
                    desc=PROGRESS_DESCRIPTION,
                    unit="observation",
                    dynamic_ncols=True,
                )
            )
        if not use_parallel:
            runtime = _build_gaussian_runtime(
                contract,
                device=active_devices[0],
                worker_count=1,
            )
            for batch in _gaussian_seed_batches(entries):
                measurement = _measure_gaussian_batch_safely(runtime, batch, center)
                measurements.append(measurement)
                progress.update(len(batch) * cells_per_seed)
        else:
            print(
                "Lemma 2 Gaussian-worker plan: "
                + ", ".join(
                    f"{device}={len(round_robin_shard(entries, worker_index=i, worker_count=worker_count))} seeds"
                    for i, device in enumerate(active_devices)
                )
                + "; one parent progress bar covers all observations"
            )
            context = multiprocessing.get_context("spawn")
            futures: dict[
                Future[_GaussianBatchMeasurement],
                tuple[tuple[int, int], ...],
            ] = {}
            with ExitStack() as stack:
                for worker_index, selected_device in enumerate(active_devices):
                    executor = stack.enter_context(
                        ProcessPoolExecutor(
                            max_workers=1,
                            mp_context=context,
                            initializer=_initialize_gaussian_worker,
                            initargs=(
                                contract,
                                center,
                                str(selected_device),
                                worker_count,
                            ),
                        )
                    )
                    shard = round_robin_shard(
                        entries,
                        worker_index=worker_index,
                        worker_count=worker_count,
                    )
                    for batch in _gaussian_seed_batches(shard):
                        futures[executor.submit(_gaussian_worker_task, batch)] = batch
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        measurement = future.result()
                    except BaseException as error:
                        wrapped = ExperimentError(
                            "Lemma 2 Gaussian worker failed: "
                            + _error_message(
                                error
                                if isinstance(error, Exception)
                                else RuntimeError(str(error))
                            )
                        )
                        measurement = _GaussianBatchMeasurement(
                            entries=batch,
                            values=None,
                            error=_error_message(wrapped),
                        )
                    measurements.append(measurement)
                    progress.update(len(batch) * cells_per_seed)
    return _expand_gaussian_measurements(
        measurements,
        selection=selection,
        contract=contract,
        centering=centering,
    )


def _compute_rows(
    prompt_rows: Sequence[Mapping[str, object]],
    *,
    selection: TargetPairSelection,
    contract: GenerationContract,
    centering: CenteringContract,
    devices: Sequence[torch.device],
    progress_factory: Callable[..., Any] = tqdm,
    progress: Any | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Load every selected trajectory and compute all P*N*T observations."""

    if not prompt_rows:
        raise ExperimentError("no selected prompts were supplied")
    if not devices:
        raise ExperimentError("at least one execution device is required")
    if centering.center is None:
        raise ExperimentError("centering mode did not construct its tensor")
    _validate_tensor(
        centering.center,
        shape=contract.latent_shape,
        dtype=torch.float64,
        label="centering tensor",
    )
    worker_count = worker_count_for_tasks(devices, len(prompt_rows))
    active_devices = tuple(devices[:worker_count])
    use_parallel = worker_count > 1 and all(
        device.type == "cuda" and device.index is not None for device in active_devices
    )
    cells_per_prompt = len(contract.seeds) * contract.num_inference_steps
    measurements: list[_PromptMeasurement] = []
    with ExitStack() as progress_stack:
        if progress is None:
            progress = progress_stack.enter_context(
                progress_factory(
                    total=len(prompt_rows) * cells_per_prompt,
                    desc=PROGRESS_DESCRIPTION,
                    unit="observation",
                    dynamic_ncols=True,
                )
            )
        if not use_parallel:
            for position, prompt in enumerate(prompt_rows):
                measurements.append(
                    _measure_prompt_safely(
                        position,
                        prompt,
                        contract=contract,
                        center=centering.center,
                        device=active_devices[0],
                    )
                )
                progress.update(cells_per_prompt)
        else:
            print(
                "Lemma 2 cache-worker plan: "
                + ", ".join(
                    f"{device}={len(round_robin_shard(prompt_rows, worker_index=i, worker_count=worker_count))} prompts"
                    for i, device in enumerate(active_devices)
                )
                + "; one parent progress bar covers all observations"
            )
            context = multiprocessing.get_context("spawn")
            futures: dict[
                Future[_PromptMeasurement],
                tuple[int, Mapping[str, object], torch.device],
            ] = {}
            indexed = tuple(enumerate(prompt_rows))
            with ExitStack() as stack:
                for worker_index, selected_device in enumerate(active_devices):
                    executor = stack.enter_context(
                        ProcessPoolExecutor(
                            max_workers=1,
                            mp_context=context,
                            initializer=_initialize_prompt_worker,
                            initargs=(
                                contract,
                                centering.center,
                                str(selected_device),
                                worker_count,
                            ),
                        )
                    )
                    for position, prompt in round_robin_shard(
                        indexed,
                        worker_index=worker_index,
                        worker_count=worker_count,
                    ):
                        future = executor.submit(_prompt_worker_task, position, prompt)
                        futures[future] = (position, prompt, selected_device)
                for future in as_completed(futures):
                    position, prompt, selected_device = futures[future]
                    try:
                        measurement = future.result()
                    except BaseException as error:
                        wrapped = ExperimentError(
                            f"Lemma 2 worker on {selected_device} failed: "
                            + _error_message(
                                error
                                if isinstance(error, Exception)
                                else RuntimeError(str(error))
                            )
                        )
                        measurement = _PromptMeasurement(
                            position=position,
                            original_index=str(prompt.get("original_index", "")),
                            trajectory_sha256="",
                            values=None,
                            error=_error_message(wrapped),
                        )
                    measurements.append(measurement)
                    progress.update(cells_per_prompt)
    return _expand_measurements(
        measurements,
        prompt_rows,
        selection=selection,
        contract=contract,
        centering=centering,
    )


def _default_output_directory(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    use_mu: bool,
    selection_strategy: str,
    selection_hash: str,
    evaluation_source: str = "both",
) -> Path:
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ExperimentError(f"unsupported selection strategy: {selection_strategy!r}")
    if not _is_sha256(selection_hash):
        raise ExperimentError("selection hash must be a lowercase SHA-256 digest")
    _evaluation_sources(evaluation_source)
    run_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        0,
    )
    base = ROOT / "outputs" / run_name / EXPERIMENT_DIRECTORY
    if use_mu:
        base = (
            base / "centering_mu_hat" / f"baseline_S{num_seeds}_N{num_baseline_seeds}"
        )
    else:
        base /= "centering_zero"

    return base / selection_hash / f"evaluation_{evaluation_source}"


def _requested_output_directory(
    *,
    model_name: str,
    use_mu: bool,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int,
    evaluation_source: str,
    selection: TargetPairSelection,
    output_dir: str | Path | None,
) -> Path:
    if (
        selection.model_name != model_name
        or selection.scheduler_name != scheduler_name
        or selection.guidance_scale != guidance_scale
        or selection.num_inference_steps != num_inference_steps
        or selection.num_seeds != num_seeds
    ):
        raise ExperimentError("frozen selection differs from the requested run")
    _evaluation_sources(evaluation_source)
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return _default_output_directory(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        num_baseline_seeds,
        use_mu,
        selection.selection_strategy,
        selection.sha256,
        evaluation_source,
    )


def _source_figure_filenames(source: str) -> tuple[str, ...]:
    if source == SOURCE_GAUSSIAN:
        return FIGURE_FILENAMES
    if source == SOURCE_TRAJECTORY:
        return TRAJECTORY_FIGURE_FILENAMES
    raise ExperimentError(f"unsupported figure source: {source!r}")


def _figure_paths(
    output_directory: str | Path, evaluation_source: str = "both"
) -> tuple[Path, ...]:
    output = Path(output_directory)
    return tuple(
        output / filename
        for source in _evaluation_sources(evaluation_source)
        for filename in _source_figure_filenames(source)
    )


def _prepare_output_directory(
    output_dir: str | Path, evaluation_source: str = "both"
) -> Path:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    allowed = {
        CSV_NAME,
        *(figure_path.name for figure_path in _figure_paths(path, evaluation_source)),
    }
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


def _remove_figure_outputs(
    output_directory: str | Path, evaluation_source: str = "both"
) -> None:
    for destination in _figure_paths(output_directory, evaluation_source):
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
                    f"unsupported Lemma 2 figure format: {destination.suffix!r}"
                )
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            staged.append((temporary, destination))
            options: dict[str, object] = {
                "format": figure_format,
                "bbox_inches": "tight",
                "pad_inches": FIGURE_PAD_INCHES,
                "dpi": FIGURE_DPI,
            }
            figure.savefig(temporary, **options)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        for _temporary, destination in staged:
            if destination.exists() and not (
                destination.is_file() or destination.is_symlink()
            ):
                raise ExperimentError(
                    f"figure destination is not a regular file: {destination}"
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
                    "cannot restore figure because its destination became unsafe: "
                    f"{destination}"
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


def _boolean_series(series: pd.Series, label: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    if not normalized.isin(("true", "false")).all():
        raise ExperimentError(f"saved CSV {label} must contain booleans")
    return normalized.eq("true")


def _numeric_series(
    frame: pd.DataFrame, name: str, *, integer: bool = False
) -> pd.Series:
    values = pd.to_numeric(frame[name], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ExperimentError(f"saved CSV {name} must be finite and numeric")
    if integer and not values.map(lambda value: float(value).is_integer()).all():
        raise ExperimentError(f"saved CSV {name} must contain integers")
    return values.astype(np.int64) if integer else values.astype(np.float64)


def _require_constant_text(frame: pd.DataFrame, name: str, expected: str) -> None:
    if not frame[name].astype(str).eq(expected).all():
        raise ExperimentError(f"saved CSV {name} differs from requested provenance")


def _validated_plot_frame(
    frame: pd.DataFrame,
    *,
    selection: TargetPairSelection,
    identity: GenerationIdentity,
    centering: CenteringContract,
    evaluation_source: str = SOURCE_TRAJECTORY,
) -> pd.DataFrame:
    """Validate provenance and each requested source's exact canonical grid."""

    if tuple(frame.columns) != CSV_COLUMNS:
        raise ExperimentError("saved CSV columns do not match the experiment schema")
    requested_sources = _evaluation_sources(evaluation_source)
    if selection.scheduler_name != identity.scheduler_name:
        raise ExperimentError(
            "frozen selection scheduler differs from generation cache"
        )
    if centering.mode == CENTERING_ZERO:
        if centering.baseline is not None:
            raise ExperimentError("zero centering unexpectedly has a baseline")
    elif centering.mode == CENTERING_MU_HAT:
        if centering.baseline is None:
            raise ExperimentError("mu_hat centering is missing its baseline")
        if centering.baseline.source_schedule_sha256 != identity.schedule_sha256:
            raise ExperimentError(
                "baseline and experiment schedule SHA-256 digests differ"
            )
    else:
        raise ExperimentError(f"invalid centering mode: {centering.mode!r}")

    prompts = _included_prompt_rows(selection)
    seed_count = len(identity.seeds)
    step_count = int(identity.science["num_inference_steps"])
    expected_count = sum(
        seed_count * step_count * (len(prompts) if source == SOURCE_TRAJECTORY else 1)
        for source in requested_sources
    )
    if len(frame) != expected_count:
        raise ExperimentError(
            f"saved CSV has {len(frame)}/{expected_count} rows for "
            f"--evaluation-source {evaluation_source}"
        )

    validated = frame.copy()
    for name in (
        "evaluation_source",
        "record_id",
        "original_index",
        "trajectory_sha256",
    ):
        validated[name] = validated[name].astype(str)
    if (
        not validated["evaluation_source"]
        .isin((SOURCE_GAUSSIAN, SOURCE_TRAJECTORY))
        .all()
    ):
        raise ExperimentError(
            "saved CSV evaluation_source must contain only gaussian or trajectory"
        )
    for name in (
        "num_inference_steps",
        "num_baseline_seeds",
        "generation_seed",
        "step_index",
        "timestep",
        "latent_dimension",
    ):
        validated[name] = _numeric_series(validated, name, integer=True)
    for name in (
        "guidance_scale",
        "alpha_t",
        "sigma_t",
        "snr_t",
        "centered_distance_rmse",
    ):
        validated[name] = _numeric_series(validated, name)
    target_sscd = pd.to_numeric(validated["target_sscd"], errors="coerce").astype(
        np.float64
    )
    gaussian_rows = validated["evaluation_source"].eq(SOURCE_GAUSSIAN)
    trajectory_rows = validated["evaluation_source"].eq(SOURCE_TRAJECTORY)
    if not target_sscd.loc[gaussian_rows].isna().all():
        raise ExperimentError(
            "saved CSV Gaussian rows must not invent target-specific SSCD values"
        )
    trajectory_scores = target_sscd.loc[trajectory_rows].to_numpy(dtype=float)
    if not np.isfinite(trajectory_scores).all() or bool(
        (
            (trajectory_scores < -1.0 - SSCD_SCORE_RANGE_TOLERANCE)
            | (trajectory_scores > 1.0 + SSCD_SCORE_RANGE_TOLERANCE)
        ).any()
    ):
        raise ExperimentError(
            "saved CSV trajectory target_sscd must be finite and inside [-1, 1]"
        )
    validated["target_sscd"] = target_sscd
    flags = _boolean_series(
        validated["is_initial_timestep"],
        "is_initial_timestep",
    )
    validated["is_initial_timestep"] = flags
    statuses = validated["status"].astype(str)
    if not statuses.eq("ok").all():
        raise ExperimentError(
            f"saved CSV contains {int((~statuses.eq('ok')).sum())} failed observations"
        )
    if not validated["error"].fillna("").astype(str).str.strip().eq("").all():
        raise ExperimentError("saved CSV status/error fields are inconsistent")
    if not (validated["centered_distance_rmse"] >= 0.0).all():
        raise ExperimentError("saved CSV centered distances must be nonnegative")

    constants = {
        "centering_mode": centering.mode,
        "selection_strategy": selection.selection_strategy,
        "selection_hash": selection.sha256,
        "model_name": selection.model_name,
        "scheduler_name": identity.scheduler_name,
        "evaluation_generation_scientific_config_hash": identity.scientific_hash,
        "evaluation_schedule_sha256": identity.schedule_sha256,
        "baseline_generation_scientific_config_hash": centering.baseline_scientific_hash,
        "baseline_mu_hat_sha256": centering.baseline_mu_hat_sha256,
    }
    for name, expected in constants.items():
        _require_constant_text(validated, name, str(expected))
    if not np.allclose(
        validated["guidance_scale"].to_numpy(),
        float(selection.guidance_scale),
        rtol=0.0,
        atol=0.0,
    ):
        raise ExperimentError("saved CSV guidance scale differs from requested run")
    if not validated["num_inference_steps"].eq(step_count).all():
        raise ExperimentError("saved CSV num_inference_steps differs from --T")
    if not validated["num_baseline_seeds"].eq(centering.num_baseline_seeds).all():
        raise ExperimentError(
            "saved CSV num_baseline_seeds differs from the requested centering mode"
        )
    if not validated["latent_dimension"].eq(identity.latent_dimension).all():
        raise ExperimentError(
            "saved CSV latent dimension differs from the generation contract"
        )

    trajectory_hashes: list[str] = []
    if SOURCE_TRAJECTORY in requested_sources:
        for prompt in prompts:
            marker = _validate_generation_prompt(
                prompt,
                identity=identity,
                verify_file_hashes=False,
            )
            trajectory_hashes.append(_trajectory_sha256(marker))

    expected_sources: list[str] = []
    expected_records: list[str] = []
    expected_indices: list[str] = []
    expected_hashes: list[str] = []
    expected_seeds: list[int] = []
    expected_steps: list[int] = []
    for source in requested_sources:
        if source == SOURCE_GAUSSIAN:
            for seed in identity.seeds:
                for step_index in range(step_count):
                    expected_sources.append(SOURCE_GAUSSIAN)
                    expected_records.append("")
                    expected_indices.append("")
                    expected_hashes.append("")
                    expected_seeds.append(seed)
                    expected_steps.append(step_index)
        else:
            for prompt, trajectory_hash in zip(prompts, trajectory_hashes, strict=True):
                for seed in identity.seeds:
                    for step_index in range(step_count):
                        expected_sources.append(SOURCE_TRAJECTORY)
                        expected_records.append(str(prompt["record_id"]))
                        expected_indices.append(str(prompt["original_index"]))
                        expected_hashes.append(trajectory_hash)
                        expected_seeds.append(seed)
                        expected_steps.append(step_index)

    for name, observed, expected in (
        (
            "evaluation_source",
            validated["evaluation_source"].tolist(),
            expected_sources,
        ),
        ("record_id", validated["record_id"].tolist(), expected_records),
        (
            "original_index",
            validated["original_index"].tolist(),
            expected_indices,
        ),
        (
            "trajectory_sha256",
            validated["trajectory_sha256"].tolist(),
            expected_hashes,
        ),
        (
            "generation_seed",
            validated["generation_seed"].tolist(),
            expected_seeds,
        ),
        ("step_index", validated["step_index"].tolist(), expected_steps),
    ):
        if observed != expected:
            raise ExperimentError(
                f"saved CSV {name} is not canonical for --evaluation-source "
                f"{evaluation_source}"
            )
    expected_step_array = np.asarray(expected_steps, dtype=np.int64)
    if not np.array_equal(flags.to_numpy(dtype=bool), expected_step_array == 0):
        raise ExperimentError("saved CSV initial-timestep flags are incomplete")
    score_counts = (
        validated.loc[trajectory_rows]
        .groupby(["record_id", "generation_seed"], sort=False)["target_sscd"]
        .nunique(dropna=False)
    )
    if not score_counts.eq(1).all():
        raise ExperimentError(
            "same-seed endpoint target_sscd differs across trajectory timesteps"
        )

    reference = validated.iloc[:step_count]
    scheduler = identity.science.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise ExperimentError("experiment scheduler metadata is invalid")
    num_train_timesteps = _num_train_timesteps(
        _normalize_scheduler_config(scheduler.get("config"))
    )
    reference_timesteps = reference["timestep"].to_numpy(dtype=np.int64)
    if (
        bool((reference_timesteps < 0).any())
        or bool((reference_timesteps >= num_train_timesteps).any())
        or (
            step_count > 1
            and not bool((reference_timesteps[:-1] > reference_timesteps[1:]).all())
        )
    ):
        raise ExperimentError(
            "saved CSV scheduler timesteps must be strictly descending and inside "
            "the training schedule"
        )
    reference_values = {
        name: reference[name].to_numpy()
        for name in ("timestep", "alpha_t", "sigma_t", "snr_t")
    }
    for block_start in range(0, expected_count, step_count):
        block = validated.iloc[block_start : block_start + step_count]
        for name, expected in reference_values.items():
            if not np.array_equal(block[name].to_numpy(), expected):
                raise ExperimentError(f"saved CSV {name} differs across source grids")
    alpha = validated["alpha_t"].to_numpy(dtype=float)
    sigma = validated["sigma_t"].to_numpy(dtype=float)
    snr = validated["snr_t"].to_numpy(dtype=float)
    if (alpha <= 0.0).any() or (sigma <= 0.0).any():
        raise ExperimentError("saved CSV diffusion coefficients must be positive")
    if not np.allclose(alpha * alpha + sigma * sigma, 1.0, rtol=1e-5, atol=1e-6):
        raise ExperimentError(
            "saved CSV alpha_t and sigma_t violate the scheduler coefficient identity"
        )
    if not np.allclose(snr, alpha * alpha / (sigma * sigma), rtol=1e-12, atol=0.0):
        raise ExperimentError("saved CSV snr_t differs from alpha_t^2/sigma_t^2")
    if centering.baseline is not None:
        initial = reference.iloc[0]
        if int(initial["timestep"]) != centering.baseline.timestep:
            raise ExperimentError(
                "saved CSV initial timestep differs from the baseline"
            )
        for name, observed, expected in (
            ("alpha_t", float(initial["alpha_t"]), centering.baseline.alpha_t),
            ("sigma_t", float(initial["sigma_t"]), centering.baseline.sigma_t),
        ):
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ExperimentError(
                    f"saved CSV initial {name} differs from the baseline"
                )
    return validated


def _render_figure(
    *, frame: pd.DataFrame, destinations: Sequence[Path], centering_mode: str
) -> None:
    if centering_mode not in (CENTERING_ZERO, CENTERING_MU_HAT):
        raise ExperimentError(f"invalid centering mode: {centering_mode!r}")
    sources = tuple(frame["evaluation_source"].astype(str).unique())
    if len(sources) != 1 or sources[0] not in (SOURCE_GAUSSIAN, SOURCE_TRAJECTORY):
        raise ExperimentError("each Lemma 2 figure requires exactly one valid source")
    source = sources[0]
    summaries: list[dict[str, float]] = []
    for step_index, group in frame.groupby("step_index", sort=True):
        values = group["centered_distance_rmse"].to_numpy(dtype=float)
        summaries.append(
            {
                "step_index": float(step_index),
                "snr_t": float(group["snr_t"].iloc[0]),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "q40": float(np.quantile(values, 0.40)),
                "q60": float(np.quantile(values, 0.60)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
            }
        )
    summaries.sort(key=lambda item: item["snr_t"])
    x_values = np.asarray([item["snr_t"] for item in summaries], dtype=float)
    medians = np.asarray([item["median"] for item in summaries], dtype=float)
    bands = (
        (
            np.asarray([item["q05"] for item in summaries], dtype=float),
            np.asarray([item["q95"] for item in summaries], dtype=float),
        ),
        (
            np.asarray([item["q25"] for item in summaries], dtype=float),
            np.asarray([item["q75"] for item in summaries], dtype=float),
        ),
        (
            np.asarray([item["q40"] for item in summaries], dtype=float),
            np.asarray([item["q60"] for item in summaries], dtype=float),
        ),
    )
    group_columns = (
        ["generation_seed"]
        if source == SOURCE_GAUSSIAN
        else ["record_id", "generation_seed"]
    )
    observation_segments: list[np.ndarray] = []
    observation_scores: list[float] = []
    for _key, group in frame.groupby(group_columns, sort=False):
        ordered = group.sort_values("snr_t", kind="stable")
        observation_segments.append(
            ordered[["snr_t", "centered_distance_rmse"]].to_numpy(dtype=float)
        )
        if source == SOURCE_TRAJECTORY:
            scores = ordered["target_sscd"].to_numpy(dtype=float)
            if not np.isfinite(scores).all() or not np.equal(scores, scores[0]).all():
                raise ExperimentError(
                    "each trajectory curve requires one finite same-seed SSCD score"
                )
            observation_scores.append(float(scores[0]))
    with matplotlib.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        try:
            for (lower, upper), (_lower_q, _upper_q, alpha) in zip(
                bands, PERCENTILE_BANDS, strict=True
            ):
                axis.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=PERCENTILE_BAND_COLOR,
                    alpha=alpha,
                    linewidth=0.0,
                )
            if source == SOURCE_TRAJECTORY:
                color_norm = Normalize(
                    vmin=SSCD_COLOR_RANGE[0],
                    vmax=SSCD_COLOR_RANGE[1],
                    clip=True,
                )
                observations = LineCollection(
                    observation_segments,
                    cmap="viridis",
                    norm=color_norm,
                    linewidths=OBSERVATION_LINE_WIDTH,
                    alpha=OBSERVATION_LINE_ALPHA,
                    zorder=2,
                )
                observations.set_array(np.asarray(observation_scores, dtype=float))
                axis.add_collection(observations)
                colorbar_mappable = ScalarMappable(norm=color_norm, cmap="viridis")
                colorbar_mappable.set_array(np.asarray(observation_scores, dtype=float))
                colorbar = figure.colorbar(colorbar_mappable, ax=axis)
                if colorbar.solids is not None:
                    colorbar.solids.set_alpha(COLORBAR_ALPHA)
                colorbar.set_label(COLORBAR_LABEL, fontsize=TEXT_FONT_SIZE)
                colorbar.ax.tick_params(labelsize=AXIS_NUMBER_FONT_SIZE)
            else:
                observations = LineCollection(
                    observation_segments,
                    colors="0.55",
                    linewidths=OBSERVATION_LINE_WIDTH,
                    alpha=OBSERVATION_LINE_ALPHA,
                    zorder=2,
                )
                axis.add_collection(observations)
            axis.plot(
                x_values,
                medians,
                color="black",
                linewidth=1.35,
                zorder=3,
            )
            axis.set_xscale("log")
            axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=7))
            axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlabel(r"$\alpha_t^2/\sigma_t^2$")
            if centering_mode == CENTERING_MU_HAT:
                ylabel = (
                    r"$\|\widehat{\mathbf{x}}_{0\mid t,\emptyset}"
                    r"-\widehat{\boldsymbol{\mu}}\|/\sqrt{d}$"
                )
            else:
                ylabel = (
                    r"$\|\widehat{\mathbf{x}}_{0\mid t,\emptyset}\|"
                    r"/\sqrt{d}$"
                )
            axis.set_ylabel(ylabel)
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
    identity: GenerationIdentity,
    centering: CenteringContract,
    evaluation_source: str = SOURCE_TRAJECTORY,
) -> None:
    """Reload the sole CSV, validate requested sources, and replace their figures."""

    source = Path(csv_path)
    if source.is_symlink() or not source.is_file():
        raise ExperimentError(f"saved CSV is missing or unsafe: {source}")
    try:
        frame = pd.read_csv(
            source,
            dtype={
                "selection_hash": str,
                "evaluation_source": str,
                "evaluation_generation_scientific_config_hash": str,
                "evaluation_schedule_sha256": str,
                "baseline_generation_scientific_config_hash": str,
                "baseline_mu_hat_sha256": str,
                "record_id": str,
                "original_index": str,
                # Missing Gaussian scores and numeric trajectory scores cross
                # parser chunks; one float64 dtype keeps round-trip bits uniform.
                "target_sscd": np.float64,
                "trajectory_sha256": str,
            },
            na_values={"target_sscd": ("", "nan")},
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ExperimentError(f"cannot read saved CSV {source}: {error}") from error

    validated = _validated_plot_frame(
        frame,
        selection=selection,
        identity=identity,
        centering=centering,
        evaluation_source=evaluation_source,
    )
    for source_name in _evaluation_sources(evaluation_source):
        source_frame = validated.loc[
            validated["evaluation_source"].eq(source_name)
        ].copy()
        _render_figure(
            frame=source_frame,
            destinations=_figure_paths(output_directory, source_name),
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
    output_dir: str | Path | None,
    use_mu: bool = False,
    evaluation_source: str = "both",
    device: str | torch.device = "auto",
    progress_factory: Callable[..., Any] = tqdm,
) -> int:
    """Compute every cell in the requested Gaussian and/or trajectory grids."""

    _validate_scientific_request(scheduler_name)
    sources = _evaluation_sources(evaluation_source)
    selection = _load_frozen_selection(
        ROOT,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        selection_strategy=selection_strategy,
    )
    prompts = _included_prompt_rows(selection)
    identity = _load_generation_identity(
        ROOT,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
    )
    target_sscd_lookup: Mapping[tuple[str, int], float] = {}
    if SOURCE_TRAJECTORY in sources:
        target_sscd_lookup = _load_target_sscd_lookup(
            prompts,
            identity=identity,
        )
    centering = _load_centering_contract(
        use_mu=use_mu,
        project_root=ROOT,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        num_baseline_seeds=num_baseline_seeds,
        identity=identity,
        load_tensor=True,
    )
    contract = _load_generation_contract(identity)
    _validate_centering_against_schedule(centering, contract)
    requested = _requested_output_directory(
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        use_mu=use_mu,
        num_seeds=num_seeds,
        num_baseline_seeds=num_baseline_seeds,
        evaluation_source=evaluation_source,
        selection=selection,
        output_dir=output_dir,
    )
    output = _prepare_output_directory(requested, evaluation_source)
    try:
        devices = resolve_devices(device)
    except DeviceSelectionError as error:
        raise ExperimentError(str(error)) from error

    cells_per_seed = contract.num_inference_steps
    progress_total = sum(
        len(contract.seeds)
        * cells_per_seed
        * (len(prompts) if source == SOURCE_TRAJECTORY else 1)
        for source in sources
    )
    rows: list[dict[str, object]] = []
    failed = 0
    with progress_factory(
        total=progress_total,
        desc=PROGRESS_DESCRIPTION,
        unit="observation",
        dynamic_ncols=True,
    ) as progress:
        for source in sources:
            if source == SOURCE_GAUSSIAN:
                source_rows, source_failed = _compute_gaussian_rows(
                    selection=selection,
                    contract=contract,
                    centering=centering,
                    devices=devices,
                    progress_factory=progress_factory,
                    progress=progress,
                )
            else:
                source_rows, source_failed = _compute_rows(
                    prompts,
                    selection=selection,
                    contract=contract,
                    centering=centering,
                    devices=devices,
                    progress_factory=progress_factory,
                    progress=progress,
                )
            rows.extend(source_rows)
            failed += source_failed

    rows = _attach_target_sscd(rows, target_sscd_lookup)
    csv_path = output / CSV_NAME
    atomic_write_csv(csv_path, rows, CSV_COLUMNS)
    if failed:
        _remove_figure_outputs(output, evaluation_source)
        print(
            f"Lemma 2 saved {failed}/{len(rows)} failed observations to {csv_path}",
            file=sys.stderr,
        )
        return 1
    try:
        plot_saved_results(
            csv_path,
            output,
            selection=selection,
            identity=identity,
            centering=centering,
            evaluation_source=evaluation_source,
        )
    except BaseException:
        _remove_figure_outputs(output, evaluation_source)
        raise
    print(f"CSV: {csv_path}")
    for figure_path in _figure_paths(output, evaluation_source):
        print(f"Figure: {figure_path}")
    return 0


def _plot_only(arguments: argparse.Namespace) -> int:
    """Plot from CSV/JSON metadata without tensors, models, or device selection."""

    _validate_scientific_request(arguments.scheduler)
    selection = _load_frozen_selection(
        ROOT,
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_seeds=arguments.num_seeds,
        selection_strategy=arguments.selection_strategy,
    )
    identity = _load_generation_identity(
        ROOT,
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_seeds=arguments.num_seeds,
    )
    centering = _load_centering_contract(
        use_mu=arguments.use_mu,
        project_root=ROOT,
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_seeds=arguments.num_seeds,
        num_baseline_seeds=arguments.num_baseline_seeds,
        identity=identity,
        load_tensor=False,
    )
    requested = _requested_output_directory(
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        use_mu=arguments.use_mu,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_seeds=arguments.num_seeds,
        num_baseline_seeds=arguments.num_baseline_seeds,
        evaluation_source=arguments.evaluation_source,
        selection=selection,
        output_dir=arguments.output_dir,
    )
    output = _prepare_output_directory(requested, arguments.evaluation_source)
    _remove_figure_outputs(output, arguments.evaluation_source)
    plot_saved_results(
        output / CSV_NAME,
        output,
        selection=selection,
        identity=identity,
        centering=centering,
        evaluation_source=arguments.evaluation_source,
    )
    for figure_path in _figure_paths(output, arguments.evaluation_source):
        print(f"Figure: {figure_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.plot:
        return _plot_only(arguments)
    return run_experiment(
        model_name=arguments.model,
        selection_strategy=arguments.selection_strategy,
        use_mu=arguments.use_mu,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.num_inference_steps,
        num_seeds=arguments.num_seeds,
        num_baseline_seeds=arguments.num_baseline_seeds,
        output_dir=arguments.output_dir,
        evaluation_source=arguments.evaluation_source,
        device=arguments.device,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExperimentError, CacheIOError, GenerationCacheError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
