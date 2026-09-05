"""Validate cached target latents and report their empirical moments."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from utils.common.io import (
    CacheIOError,
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)

from .cache import (
    CompletedGenerationRecord,
    GENERATION_SCHEMA_VERSION,
    GENERATION_SELECTION_POLICY,
    SAMPLER_CONTRACT_VERSION,
    GenerationCacheError,
    GenerationPaths,
    list_completed_records,
    require_generation_run,
)

STATISTICS_SCHEMA_VERSION = 1
STATISTICS_DIRECTORY_NAME = "target_latent_statistics"
TARGET_LATENT_MARKER_FINGERPRINT_FIELD = (
    "generation_target_latent_marker_fingerprint_sha256"
)


class TargetLatentStatisticsError(RuntimeError):
    """A target-latent population is absent, unsafe, or inconsistent."""


@dataclass(frozen=True, slots=True)
class TargetLatentStatisticsResult:
    """Paths and exact JSON-ready values for one completed report."""

    report_path: Path
    mean_path: Path
    population_std_path: Path
    report: Mapping[str, Any]


def target_latent_marker_fingerprint(
    records: Sequence[CompletedGenerationRecord],
) -> str:
    """Hash only marker fields that identify the target-latent population."""

    identities: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.original_index):
        image_hash = record.metadata.get("target_image_sha256")
        tensor_hashes = record.metadata.get("tensor_file_sha256")
        target_hash = (
            tensor_hashes.get("target_latent")
            if isinstance(tensor_hashes, Mapping)
            else None
        )
        if not _is_sha256(image_hash) or not _is_sha256(target_hash):
            raise TargetLatentStatisticsError(
                f"target latent {record.original_index} has invalid marker hashes"
            )
        identities.append(
            {
                "original_index": record.original_index,
                "target_image_sha256": image_hash,
                "tensor_file_sha256": {"target_latent": target_hash},
            }
        )
    return canonical_hash({"records": identities})


@dataclass(slots=True)
class _MomentAccumulator:
    """Accumulate row, channel, and coordinate moments in float64."""

    latent_shape: tuple[int, int, int]
    record_count: int
    element_sum: torch.Tensor
    element_squared_sum: torch.Tensor
    channel_sum: torch.Tensor
    channel_squared_sum: torch.Tensor
    coordinate_sum: torch.Tensor
    coordinate_squared_sum: torch.Tensor

    @classmethod
    def create(cls, latent_shape: tuple[int, int, int]) -> _MomentAccumulator:
        channels = latent_shape[0]
        return cls(
            latent_shape=latent_shape,
            record_count=0,
            element_sum=torch.zeros((), dtype=torch.float64),
            element_squared_sum=torch.zeros((), dtype=torch.float64),
            channel_sum=torch.zeros(channels, dtype=torch.float64),
            channel_squared_sum=torch.zeros(channels, dtype=torch.float64),
            coordinate_sum=torch.zeros(latent_shape, dtype=torch.float64),
            coordinate_squared_sum=torch.zeros(latent_shape, dtype=torch.float64),
        )

    def update(self, latent: torch.Tensor) -> None:
        values = latent.to(dtype=torch.float64)
        self.record_count += 1
        self.element_sum += values.sum()
        self.element_squared_sum += values.square().sum()
        flattened_channels = values.flatten(start_dim=1)
        self.channel_sum += flattened_channels.sum(dim=1)
        self.channel_squared_sum += flattened_channels.square().sum(dim=1)
        self.coordinate_sum += values
        self.coordinate_squared_sum += values.square()

    def finalize(self) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
        if self.record_count < 1:
            raise TargetLatentStatisticsError("target-latent population is empty")
        dimensions = math.prod(self.latent_shape)
        element_count = self.record_count * dimensions
        spatial_dimensions = math.prod(self.latent_shape[1:])
        channel_element_count = self.record_count * spatial_dimensions

        element_mean = self.element_sum / element_count
        raw_second_moment = self.element_squared_sum / element_count
        element_variance = torch.clamp(
            raw_second_moment - element_mean.square(), min=0.0
        )
        mean_latent = self.coordinate_sum / self.record_count
        coordinate_second_moment = self.coordinate_squared_sum / self.record_count
        coordinate_variance = torch.clamp(
            coordinate_second_moment - mean_latent.square(), min=0.0
        )
        coordinate_std = coordinate_variance.sqrt()
        mean_vector_squared_norm_per_dimension = mean_latent.square().mean()
        mean_variance_per_dimension = coordinate_variance.mean()

        channel_mean = self.channel_sum / channel_element_count
        channel_second_moment = self.channel_squared_sum / channel_element_count
        channel_variance = torch.clamp(
            channel_second_moment - channel_mean.square(), min=0.0
        )
        decomposition_residual = (
            raw_second_moment
            - mean_vector_squared_norm_per_dimension
            - mean_variance_per_dimension
        )

        metrics: dict[str, object] = {
            "record_count": self.record_count,
            "scalar_element_count": element_count,
            "element_mean": _finite_float(element_mean, "element mean"),
            "element_population_variance": _finite_float(
                element_variance, "element population variance"
            ),
            "element_population_std": _finite_float(
                element_variance.sqrt(), "element population standard deviation"
            ),
            "mean_squared_l2_per_dimension": _finite_float(
                raw_second_moment, "mean squared L2 norm per dimension"
            ),
            "root_mean_squared_l2_per_dimension": _finite_float(
                raw_second_moment.sqrt(), "root mean squared L2 norm per dimension"
            ),
            "mean_vector_rms": _finite_float(
                mean_vector_squared_norm_per_dimension.sqrt(), "mean-vector RMS"
            ),
            "mean_vector_max_abs": _finite_float(
                mean_latent.abs().max(), "maximum absolute mean-vector coordinate"
            ),
            "mean_variance_per_dimension": _finite_float(
                mean_variance_per_dimension, "mean coordinate variance"
            ),
            "root_mean_variance_per_dimension": _finite_float(
                mean_variance_per_dimension.sqrt(), "root mean coordinate variance"
            ),
            "moment_decomposition_residual": _finite_float(
                decomposition_residual, "moment decomposition residual"
            ),
            "per_channel_element_mean": _finite_float_list(
                channel_mean, "per-channel means"
            ),
            "per_channel_element_population_std": _finite_float_list(
                channel_variance.sqrt(), "per-channel population standard deviations"
            ),
            "per_channel_mean_squared_l2_per_dimension": _finite_float_list(
                channel_second_moment, "per-channel second moments"
            ),
        }
        return (
            metrics,
            mean_latent.contiguous(),
            coordinate_std.contiguous(),
        )


def compute_target_latent_statistics(
    project_root: str | Path,
    generation_run: str | Path,
) -> TargetLatentStatisticsResult:
    """Validate and summarize all completed target latents in one run.

    Every completed prompt--target record receives one unit of weight in the
    primary population, matching the empirical paired-data distribution.  A
    deduplicated target-image population is reported separately as an audit.
    Only ``target_latent`` files are loaded; trajectory and noise caches are
    never read by this diagnostic.
    """

    root = Path(project_root).expanduser().resolve()
    run_directory = _resolve_run_directory(root, generation_run)
    paths = GenerationPaths(run_directory)
    try:
        configuration = require_generation_run(paths)
        records = list_completed_records(paths)
    except (CacheIOError, GenerationCacheError) as error:
        raise TargetLatentStatisticsError(str(error)) from error
    if not records:
        raise TargetLatentStatisticsError(
            f"generation run has no completed target latents: {run_directory}"
        )
    marker_fingerprint = target_latent_marker_fingerprint(records)

    science = configuration.get("scientific_config")
    if not isinstance(science, Mapping):
        raise TargetLatentStatisticsError("generation scientific_config is missing")
    science_hash = configuration.get("scientific_config_hash")
    if not _is_sha256(science_hash):
        raise TargetLatentStatisticsError(
            "generation scientific_config_hash is invalid"
        )
    latent_shape = _latent_shape(science.get("latent_shape"))
    target_definition = science.get("target_latent_definition")
    if not isinstance(target_definition, str) or not target_definition:
        raise TargetLatentStatisticsError("target latent definition is missing")

    paired = _MomentAccumulator.create(latent_shape)
    unique = _MomentAccumulator.create(latent_shape)
    image_value_digests: dict[str, str] = {}
    label_counts: Counter[str] = Counter()
    input_identities: list[dict[str, object]] = []

    progress = tqdm(
        records,
        total=len(records),
        desc="[Target latents] Statistics",
        unit="latent",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    )
    for record in progress:
        try:
            latent, image_hash, file_hash, value_digest = _validated_target_latent(
                paths,
                record.original_index,
                record.metadata,
                science_hash,
                target_definition,
                latent_shape,
            )
            paired.update(latent)
            previous_digest = image_value_digests.get(image_hash)
            if previous_digest is None:
                image_value_digests[image_hash] = value_digest
                unique.update(latent)
            elif previous_digest != value_digest:
                raise TargetLatentStatisticsError(
                    "the same target image hash maps to different latent values"
                )
            label = str(record.metadata.get("webster_overfit_type") or "UNKNOWN")
            label_counts[label] += 1
            input_identities.append(
                {
                    "original_index": record.original_index,
                    "source_row_number": record.source_row_number,
                    "target_image_sha256": image_hash,
                    "target_latent_file_sha256": file_hash,
                    "target_latent_value_sha256": value_digest,
                }
            )
        except (
            CacheIOError,
            GenerationCacheError,
            OSError,
            TargetLatentStatisticsError,
        ) as error:
            raise TargetLatentStatisticsError(
                f"target latent {record.original_index} is invalid: {error}"
            ) from error

    paired_metrics, paired_mean, paired_coordinate_std = paired.finalize()
    unique_metrics, _, _ = unique.finalize()
    artifact_directory = run_directory / STATISTICS_DIRECTORY_NAME
    _prepare_artifact_directory(artifact_directory)
    mean_path = artifact_directory / "mean.pt"
    population_std_path = artifact_directory / "population_std.pt"
    report_path = artifact_directory / "report.json"
    mean_hash = atomic_torch_save(paired_mean, mean_path)
    population_std_hash = atomic_torch_save(paired_coordinate_std, population_std_path)

    population = _population_scope(paths, paired.record_count)
    scaling_factor = _vae_scaling_factor_metadata(science)
    report: dict[str, object] = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "artifact": "target_latent_statistics",
        "run_directory": _relative(run_directory, root),
        "population_scope": "completed_recovered_prompt_target_rows_in_generation_run",
        "primary_weighting": "one_unit_per_completed_prompt_target_record",
        "audit_weighting": "one_unit_per_unique_target_image_sha256",
        "model_manifest_rows": population["model_manifest_rows"],
        "available_paired_image_rows": population["available_paired_image_rows"],
        "completed_target_latent_rows": paired.record_count,
        "unreported_model_manifest_rows": population["unreported_model_manifest_rows"],
        "unique_target_image_count": unique.record_count,
        "duplicate_prompt_target_record_count": paired.record_count
        - unique.record_count,
        "counts_by_webster_label": dict(sorted(label_counts.items())),
        "latent_shape": list(latent_shape),
        "dimensions_per_latent": math.prod(latent_shape),
        "input_dtype": "float32",
        "accumulation_dtype": "float64",
        "standard_deviation_correction": 0,
        "target_latent_definition": target_definition,
        "target_preprocessing": science.get("target_preprocessing"),
        "model_id": science.get("model_id"),
        "model_revision": science.get("model_revision"),
        "vae_id": science.get("vae_id"),
        "vae_revision": science.get("vae_revision"),
        "vae_scaling_factor": scaling_factor,
        "scientific_config_hash": science_hash,
        TARGET_LATENT_MARKER_FINGERPRINT_FIELD: marker_fingerprint,
        "input_identity_sha256": canonical_hash(
            {
                "scientific_config_hash": science_hash,
                "records": input_identities,
            }
        ),
        "equation_mapping": {
            "empirical_vector_mean": "target_latent_statistics/mean.pt",
            "mean_squared_l2_per_dimension": (
                "paired_record_weighted.mean_squared_l2_per_dimension"
            ),
        },
        "paired_record_weighted": paired_metrics,
        "unique_target_image_weighted": unique_metrics,
        "coordinate_artifacts": {
            "mean": {
                "path": _relative(mean_path, root),
                "dtype": "float64",
                "shape": list(latent_shape),
                "sha256": mean_hash,
            },
            "population_std": {
                "path": _relative(population_std_path, root),
                "dtype": "float64",
                "shape": list(latent_shape),
                "standard_deviation_correction": 0,
                "sha256": population_std_hash,
            },
        },
    }
    atomic_write_json(report_path, report)
    return TargetLatentStatisticsResult(
        report_path=report_path,
        mean_path=mean_path,
        population_std_path=population_std_path,
        report=report,
    )


def format_target_latent_statistics(report: Mapping[str, object]) -> str:
    """Format the primary population as four concise, unambiguous lines."""

    metrics = report.get("paired_record_weighted")
    if not isinstance(metrics, Mapping):
        raise TargetLatentStatisticsError("statistics report has no primary metrics")
    shape = _latent_shape(report.get("latent_shape"))
    completed = _nonnegative_integer(
        report.get("completed_target_latent_rows"), "completed target latent rows"
    )
    total = report.get("model_manifest_rows")
    total_text = str(total) if _is_nonnegative_integer(total) else "?"
    unique = _nonnegative_integer(
        report.get("unique_target_image_count"), "unique target image count"
    )
    mean = _metric_float(metrics, "element_mean")
    std = _metric_float(metrics, "element_population_std")
    second_moment = _metric_float(metrics, "mean_squared_l2_per_dimension")
    rms = _metric_float(metrics, "root_mean_squared_l2_per_dimension")
    mean_vector_rms = _metric_float(metrics, "mean_vector_rms")
    mean_vector_max = _metric_float(metrics, "mean_vector_max_abs")
    channel_mean = _metric_float_sequence(metrics, "per_channel_element_mean")
    channel_std = _metric_float_sequence(metrics, "per_channel_element_population_std")
    return "\n".join(
        (
            "Target latent population: "
            f"{completed}/{total_text} recovered/completed rows, "
            f"{unique} unique images, shape={shape}, population correction=0",
            "Target latent moments: "
            f"mean={mean:.10f} | std={std:.10f} | "
            f"E[||x_0||^2]/d={second_moment:.10f} | RMS={rms:.10f}",
            "Target latent mean vector: "
            f"RMS={mean_vector_rms:.10f} | max_abs={mean_vector_max:.10f}",
            "Target latent channels: "
            f"mean={_format_vector(channel_mean)} | "
            f"std={_format_vector(channel_std)}",
        )
    )


def _validated_target_latent(
    paths: GenerationPaths,
    original_index: str,
    metadata: Mapping[str, object],
    science_hash: str,
    target_definition: str,
    latent_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, str, str, str]:
    if metadata.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise TargetLatentStatisticsError("generation schema differs")
    if metadata.get("sampler_contract_version") != SAMPLER_CONTRACT_VERSION:
        raise TargetLatentStatisticsError("sampler contract differs")
    if metadata.get("selection_policy") != GENERATION_SELECTION_POLICY:
        raise TargetLatentStatisticsError("generation selection policy differs")
    if metadata.get("scientific_config_hash") != science_hash:
        raise TargetLatentStatisticsError("scientific configuration hash differs")
    if metadata.get("target_latent_definition") != target_definition:
        raise TargetLatentStatisticsError("target latent definition differs")
    if metadata.get("latent_shape") != list(latent_shape):
        raise TargetLatentStatisticsError("marker latent shape differs")
    expected_relative = f"target_latent/{original_index}.pt"
    if metadata.get("target_latent_path") != expected_relative:
        raise TargetLatentStatisticsError("marker target latent path differs")

    shapes = metadata.get("tensor_shapes")
    dtypes = metadata.get("tensor_dtypes")
    hashes = metadata.get("tensor_file_sha256")
    if not isinstance(shapes, Mapping) or shapes.get("target_latent") != list(
        latent_shape
    ):
        raise TargetLatentStatisticsError("marker target latent tensor shape differs")
    if not isinstance(dtypes, Mapping) or dtypes.get("target_latent") != "float32":
        raise TargetLatentStatisticsError("marker target latent tensor dtype differs")
    expected_file_hash = (
        hashes.get("target_latent") if isinstance(hashes, Mapping) else None
    )
    if not _is_sha256(expected_file_hash):
        raise TargetLatentStatisticsError("marker target latent SHA-256 is invalid")
    image_hash = metadata.get("target_image_sha256")
    if not _is_sha256(image_hash):
        raise TargetLatentStatisticsError("marker target image SHA-256 is invalid")

    target_path = paths.target_latent_path(original_index)
    if target_path.is_symlink() or not target_path.is_file():
        raise TargetLatentStatisticsError("target latent is not a regular file")
    observed_file_hash = file_sha256(target_path)
    if observed_file_hash != expected_file_hash:
        raise TargetLatentStatisticsError("target latent SHA-256 differs")
    payload = safe_torch_load(target_path)
    if not isinstance(payload, torch.Tensor):
        raise TargetLatentStatisticsError("target latent payload is not a tensor")
    if tuple(payload.shape) != latent_shape:
        raise TargetLatentStatisticsError(
            f"target latent shape differs: expected {latent_shape}, got {tuple(payload.shape)}"
        )
    if payload.dtype != torch.float32:
        raise TargetLatentStatisticsError("target latent dtype is not float32")
    if payload.device.type != "cpu" or not payload.is_contiguous():
        raise TargetLatentStatisticsError(
            "target latent is not a contiguous CPU tensor"
        )
    if not bool(torch.isfinite(payload).all()):
        raise TargetLatentStatisticsError("target latent contains non-finite values")
    value_digest = hashlib.sha256(payload.numpy().tobytes(order="C")).hexdigest()
    return payload, image_hash, observed_file_hash, value_digest


def _resolve_run_directory(root: Path, generation_run: str | Path) -> Path:
    candidate = Path(generation_run).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink():
        raise TargetLatentStatisticsError(
            f"generation run may not be a symlink: {candidate}"
        )
    resolved = candidate.resolve()
    logs = (root / "logs").resolve()
    try:
        resolved.relative_to(logs)
    except ValueError as error:
        raise TargetLatentStatisticsError(
            f"generation run must be beneath {logs}: {resolved}"
        ) from error
    if resolved == logs or resolved.is_symlink() or not resolved.is_dir():
        raise TargetLatentStatisticsError(
            f"generation run is not a regular run directory: {resolved}"
        )
    return resolved


def _prepare_artifact_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise TargetLatentStatisticsError(
            f"unsafe target-latent statistics directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _population_scope(paths: GenerationPaths, completed: int) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "model_manifest_rows": None,
        "available_paired_image_rows": None,
        "unreported_model_manifest_rows": None,
    }
    if paths.summary_json.is_symlink() or not paths.summary_json.is_file():
        return result
    try:
        summary = read_json(paths.summary_json)
    except CacheIOError:
        return result
    total = summary.get("model_manifest_rows")
    available = summary.get("available_paired_image_rows")
    if _is_nonnegative_integer(total) and total >= completed:
        result["model_manifest_rows"] = total
        result["unreported_model_manifest_rows"] = total - completed
    if _is_nonnegative_integer(available) and available >= completed:
        result["available_paired_image_rows"] = available
    return result


def _vae_scaling_factor_metadata(science: Mapping[str, object]) -> dict[str, object]:
    recorded = science.get("vae_scaling_factor")
    if _is_positive_finite_float(recorded):
        return {"value": float(recorded), "source": "generation_scientific_config"}

    model_id = science.get("model_id")
    vae_id = science.get("vae_id")
    vae_revision = science.get("vae_revision")
    if not all(
        isinstance(value, str) and value for value in (model_id, vae_id, vae_revision)
    ):
        return {"value": None, "source": "not_recorded_in_generation_config"}
    try:
        from huggingface_hub import hf_hub_download

        subfolder = "vae" if vae_id == model_id else None
        config_path = hf_hub_download(
            repo_id=vae_id,
            filename="config.json",
            subfolder=subfolder,
            revision=vae_revision,
            local_files_only=True,
        )
        vae_config = read_json(config_path)
        value = vae_config.get("scaling_factor")
        if _is_positive_finite_float(value):
            return {
                "value": float(value),
                "source": "pinned_local_vae_config",
                "repository": vae_id,
                "revision": vae_revision,
            }
    except Exception:
        # Scaling metadata is best-effort and strictly cache-only. Missing
        # local Hub metadata must not invalidate moments computed from pinned
        # target tensors.
        pass
    return {"value": None, "source": "not_recorded_in_generation_config"}


def _latent_shape(value: object) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TargetLatentStatisticsError("latent shape must be [C, H, W]")
    dimensions = tuple(value)
    if len(dimensions) != 3 or not all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in dimensions
    ):
        raise TargetLatentStatisticsError("latent shape must be [C, H, W]")
    return dimensions


def _finite_float(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise TargetLatentStatisticsError(f"{name} is not finite")
    return number


def _finite_float_list(value: torch.Tensor, name: str) -> list[float]:
    result = [_finite_float(item, name) for item in value.tolist()]
    if not result:
        raise TargetLatentStatisticsError(f"{name} is empty")
    return result


def _metric_float(metrics: Mapping[str, object], key: str) -> float:
    if key not in metrics:
        raise TargetLatentStatisticsError(f"statistics report is missing {key}")
    return _finite_float(metrics[key], key)


def _metric_float_sequence(
    metrics: Mapping[str, object], key: str
) -> tuple[float, ...]:
    value = metrics.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TargetLatentStatisticsError(f"statistics report is missing {key}")
    result = tuple(_finite_float(item, key) for item in value)
    if not result:
        raise TargetLatentStatisticsError(f"statistics report has empty {key}")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if not _is_nonnegative_integer(value):
        raise TargetLatentStatisticsError(f"{name} is invalid")
    return value


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_finite_float(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _format_vector(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.10f}" for value in values) + "]"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())
