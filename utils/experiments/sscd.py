"""Full-resolution SSCD scoring for every completed generation record."""

from __future__ import annotations

import multiprocessing
import os
import re
import traceback
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from utils.common.cli import validate_seed_block
from utils.common.io import (
    CacheIOError,
    atomic_torch_save,
    atomic_write_frame_csv,
    atomic_write_frame_parquet,
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.metrics.sscd import (
    SSCD_CHECKPOINT_URL,
    SSCD_FEATURE_DIMENSION,
    SSCD_INPUT_SIZE,
    SSCD_MODEL_NAME,
    SSCDCheckpoint,
    SSCDMetricError,
    compute_sscd_scores,
    ensure_sscd_checkpoint,
    load_sscd_model,
    sscd_preprocessing_hash,
    sscd_preprocessing_policy,
    validate_sscd_scores,
)
from utils.models.devices import (
    DeviceSelectionError,
    configure_worker_cpu_threads,
    resolve_devices,
    round_robin_shard,
    worker_count_for_tasks,
)

from .cache import (
    CompletedGenerationRecord,
    GenerationCacheError,
    GenerationPaths,
    generation_paths,
    list_completed_records,
    require_generation_run,
    safe_index,
    validate_generation_record,
)

SSCD_SCHEMA_VERSION = 1
SCORE_DEFINITION = "cosine_similarity_of_l2_normalized_sscd_descriptors"
SSCD_SELECTION_POLICY = "completed_generation_cache_records"
_GENERATION_CACHE_NAMESPACE = re.compile(
    r"(?:experiment|reference|seed)_S(?P<start>\d+)_N(?P<count>\d+)\Z"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

MANIFEST_COLUMNS = (
    "original_index",
    "record_id",
    "source_row_number",
    "prompt_raw",
    "webster_overfit_type",
    "recovery_status",
    "recovery_method",
    "score_path",
    "target_image_sha256",
    "generation_latent_sha256",
    "num_seeds",
    "sscd_min",
    "sscd_max",
    "sscd_mean",
    "sscd_median",
    "sscd_std",
    "model_id",
    "model_revision",
    "vae_id",
    "vae_revision",
    "sscd_model_name",
    "sscd_checkpoint_sha256",
    "completed_or_resumed",
)


class SSCDEvaluationError(RuntimeError):
    """An SSCD cache cannot be validated or completed safely."""


def _install_tqdm_lock(lock: Any) -> None:
    """Install the one inter-process lock shared by all progress bars."""

    tqdm.set_lock(lock)


def _stable_generation_run_path(value: object, *, configuration: bool) -> object:
    """Return the logical run identity for a current role-scoped cache path."""

    if not isinstance(value, str):
        return value
    path = PurePosixPath(value)
    run = path.parent if configuration else path
    namespace = _GENERATION_CACHE_NAMESPACE.fullmatch(run.name)
    if namespace is None:
        return value
    seed_start = int(namespace.group("start"))
    seed_count = namespace.group("count")
    if seed_start == 0:
        logical_run = run.parent
    else:
        suffix = f"_N{seed_count}"
        parent_name = run.parent.name
        if not parent_name.endswith(suffix):
            return value
        logical_run = run.parent.with_name(
            parent_name[: -len(suffix)] + f"_S{seed_start}_N{seed_count}"
        )
    return str(logical_run / path.name) if configuration else str(logical_run)


def sscd_configuration_hash(configuration: Mapping[str, object]) -> str:
    """Hash the scientific SSCD contract independently of cache placement."""

    values = dict(configuration)
    values.pop("configuration_hash", None)
    if "generation_run_path" in values:
        values["generation_run_path"] = _stable_generation_run_path(
            values["generation_run_path"], configuration=False
        )
    if "generation_run_config_path" in values:
        values["generation_run_config_path"] = _stable_generation_run_path(
            values["generation_run_config_path"], configuration=True
        )
    return canonical_hash(values)


@dataclass(frozen=True, slots=True)
class SSCDPaths:
    """Files owned by SSCD inside one generation run."""

    run_directory: Path

    @property
    def score_directory(self) -> Path:
        return self.run_directory / "sscd"

    @property
    def record_directory(self) -> Path:
        return self.run_directory / "sscd_record"

    @property
    def traceback_directory(self) -> Path:
        return self.run_directory / "sscd_tracebacks"

    @property
    def stale_directory(self) -> Path:
        return self.run_directory / "sscd_stale"

    @property
    def config_json(self) -> Path:
        return self.run_directory / "sscd_config.json"

    @property
    def manifest_parquet(self) -> Path:
        return self.run_directory / "sscd_manifest.parquet"

    @property
    def manifest_csv(self) -> Path:
        return self.run_directory / "sscd_manifest.csv"

    @property
    def failed_csv(self) -> Path:
        return self.run_directory / "sscd_failed.csv"

    @property
    def summary_json(self) -> Path:
        return self.run_directory / "sscd_summary.json"

    def score_path(self, index: object) -> Path:
        return self.score_directory / f"{safe_index(index)}.pt"

    def marker_path(self, index: object) -> Path:
        return self.record_directory / f"{safe_index(index)}.json"

    def create(self) -> None:
        for path in (
            self.score_directory,
            self.record_directory,
        ):
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_dir():
                    raise SSCDEvaluationError(f"unsafe SSCD cache directory: {path}")
                continue
            path.mkdir(parents=True)


@dataclass(frozen=True, slots=True)
class SSCDResult:
    """Result paths and status from one cache-based evaluation."""

    summary_path: Path
    completed_prompt_count: int
    failed_prompt_count: int
    complete_cache_hit: bool

    @property
    def exit_code(self) -> int:
        return int(self.failed_prompt_count > 0)


@dataclass(slots=True)
class _Runtime:
    checkpoint: SSCDCheckpoint
    descriptor: Any
    vae: Any
    device: torch.device


@dataclass(frozen=True, slots=True)
class _ShardResult:
    rows: tuple[dict[str, object], ...]
    failures: tuple[dict[str, object], ...]
    newly_computed: int


def run_sscd(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
    device: str | torch.device,
    overwrite: bool,
) -> SSCDResult:
    """Score every completed record, reusing valid per-prompt score tensors."""

    if not isinstance(overwrite, bool):
        raise SSCDEvaluationError("overwrite must be a boolean")
    started = datetime.now(UTC)
    project = Path(root).expanduser().resolve()
    try:
        seed_start, num_seeds = validate_seed_block(seed_start, num_seeds)
    except ValueError as error:
        raise SSCDEvaluationError(str(error)) from error
    generation = generation_paths(
        project,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        seed_start=seed_start,
    )
    run_configuration = require_generation_run(generation)
    seed_values = _validate_invocation(
        run_configuration,
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
    )
    records = list_completed_records(generation)
    if not records:
        raise SSCDEvaluationError(
            f"generation run has no completed records: {generation.record_directory}"
        )
    paths = SSCDPaths(generation.run_directory)
    configuration = _load_or_create_configuration(
        project, paths, run_configuration, generation, seed_values
    )
    configuration_hash = str(configuration["configuration_hash"])
    if not overwrite:
        cached = _complete_cache_result(
            project,
            generation,
            paths,
            records,
            run_configuration,
            configuration,
            seed_values,
        )
        if cached is not None:
            return cached

    paths.create()
    try:
        devices = resolve_devices(device)
    except DeviceSelectionError as error:
        raise SSCDEvaluationError(str(error)) from error

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    pending: list[CompletedGenerationRecord] = []
    validation_progress = tqdm(
        records,
        total=len(records),
        desc="[SSCD] Validating cache",
        unit="prompt",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    )
    for record in validation_progress:
        status = "validating"
        validation_progress.set_postfix(
            id=record.original_index,
            status=status,
            resumed=len(rows),
            pending=len(pending),
            failed=len(failures),
        )
        try:
            scores = _validated_cached_scores(
                project,
                generation,
                paths,
                record,
                run_configuration,
                configuration_hash,
                seed_values,
                overwrite,
            )
            if scores is None:
                pending.append(record)
                status = "pending"
            else:
                rows.append(
                    _manifest_row(
                        project, record, scores, paths, configuration, "resumed"
                    )
                )
                status = "resumed"
        except Exception as error:
            failures.append(
                _failure_row(project, paths, record, error, traceback.format_exc())
            )
            status = "error"
        finally:
            validation_progress.set_postfix(
                id=record.original_index,
                status=status,
                resumed=len(rows),
                pending=len(pending),
                failed=len(failures),
                refresh=False,
            )

    computed = _compute_pending_records(
        project,
        generation,
        paths,
        tuple(pending),
        run_configuration,
        configuration,
        seed_values,
        devices,
        overwrite,
    )
    rows.extend(computed.rows)
    failures.extend(computed.failures)
    newly_computed = computed.newly_computed

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if not manifest.empty:
        manifest = manifest.sort_values(
            ["source_row_number", "original_index"], kind="stable"
        ).reset_index(drop=True)
    failure_columns = (
        "original_index",
        "record_id",
        "source_row_number",
        "webster_overfit_type",
        "exception_type",
        "exception_message",
        "traceback_path",
    )
    failed = pd.DataFrame(failures, columns=failure_columns)
    if not failed.empty:
        failed = failed.sort_values(
            ["source_row_number", "original_index"], kind="stable"
        ).reset_index(drop=True)
    atomic_write_frame_parquet(manifest, paths.manifest_parquet)
    atomic_write_frame_csv(manifest, paths.manifest_csv)
    atomic_write_frame_csv(failed, paths.failed_csv)
    summary = _summary(
        project,
        generation,
        paths,
        manifest,
        failures,
        configuration,
        len(records),
        newly_computed,
        started,
        overwrite,
    )
    worker_count = worker_count_for_tasks(devices, len(pending))
    summary.update(
        {
            "requested_device": str(device),
            "execution_devices": [str(selected) for selected in devices[:worker_count]],
            "worker_count": worker_count,
        }
    )
    atomic_write_json(paths.summary_json, summary)
    return SSCDResult(paths.summary_json, len(rows), len(failures), False)


def _complete_cache_result(
    project: Path,
    generation: GenerationPaths,
    paths: SSCDPaths,
    records: Sequence[CompletedGenerationRecord],
    run_configuration: Mapping[str, Any],
    configuration: Mapping[str, Any],
    seed_values: Sequence[int],
) -> SSCDResult | None:
    """Return without tensor I/O only for one fully published SSCD population."""

    aggregate_paths = (
        paths.config_json,
        paths.manifest_parquet,
        paths.manifest_csv,
        paths.failed_csv,
    )
    try:
        if (
            generation.summary_json.is_symlink()
            or not generation.summary_json.is_file()
            or generation.summary_json.stat().st_size <= 0
        ):
            return None
        for path in (paths.summary_json, *aggregate_paths):
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                return None
        summary_stat = paths.summary_json.stat()
        for path in aggregate_paths:
            artifact_stat = path.stat()
            if (
                artifact_stat.st_mtime_ns > summary_stat.st_mtime_ns
                or artifact_stat.st_ctime_ns > summary_stat.st_ctime_ns
            ):
                return None
        summary = read_json(paths.summary_json)
        generation_summary = read_json(generation.summary_json)
    except (CacheIOError, OSError, TypeError, ValueError):
        return None

    record_count = len(records)
    num_seeds = len(seed_values)
    science_hash = run_configuration.get("scientific_config_hash")
    configuration_hash = configuration.get("configuration_hash")
    expected_summary = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "generation_run_path": _relative(generation.run_directory, project),
        "generation_scientific_config_hash": science_hash,
        "generation_record_count": record_count,
        "selected_prompt_count": record_count,
        "completed_prompt_count": record_count,
        "failed_prompt_count": 0,
        "total_sscd_scores": record_count * num_seeds,
        "num_seeds": num_seeds,
        "selection_policy": SSCD_SELECTION_POLICY,
        "sscd_checkpoint_path": configuration.get("sscd_checkpoint_path"),
        "sscd_checkpoint_sha256": configuration.get("sscd_checkpoint_sha256"),
        "preprocessing": configuration.get("sscd_preprocessing"),
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        return None
    stored_configuration_hash = summary.get("sscd_configuration_hash")
    if (
        stored_configuration_hash is not None
        and stored_configuration_hash != configuration_hash
    ):
        return None
    expected_generation_summary = {
        "schema_version": 1,
        "outcome": "completed",
        "selected_rows": record_count,
        "completed_rows": record_count,
        "failed_rows": 0,
        "scientific_config_hash": science_hash,
    }
    if any(
        generation_summary.get(key) != value
        for key, value in expected_generation_summary.items()
    ):
        return None

    expected_indices = {record.original_index for record in records}
    if len(expected_indices) != record_count:
        return None
    score_indices = _exact_artifact_indices(paths.score_directory, ".pt")
    marker_indices = _exact_artifact_indices(paths.record_directory, ".json")
    generation_indices = _exact_artifact_indices(generation.record_directory, ".json")
    if (
        score_indices != expected_indices
        or marker_indices != expected_indices
        or generation_indices != expected_indices
    ):
        return None

    for record in records:
        try:
            generation_validation = validate_generation_record(
                generation,
                record.original_index,
                expected_scientific_hash=str(science_hash),
                expected_record_identity=_generation_record_identity(record),
                load_tensors=False,
                tensor_names=("latent",),
                require_preview=False,
                verify_file_hashes=False,
            )
            if not generation_validation.valid:
                return None
            marker_path = paths.marker_path(record.original_index)
            score_path = paths.score_path(record.original_index)
            marker = read_json(marker_path)
            if not _complete_sscd_marker_matches(
                marker,
                record,
                configuration,
                seed_values,
            ):
                return None
            if not _generation_inputs_are_published(
                project,
                generation,
                record,
                publication_marker=marker_path,
                verify_changed_files=False,
            ):
                return None
            marker_stat = marker_path.stat()
            score_stat = score_path.stat()
            if (
                marker_stat.st_size <= 0
                or marker_stat.st_mtime_ns > summary_stat.st_mtime_ns
                or marker_stat.st_ctime_ns > summary_stat.st_ctime_ns
                or score_stat.st_size <= 0
                or score_stat.st_mtime_ns > marker_stat.st_mtime_ns
                or score_stat.st_ctime_ns > marker_stat.st_ctime_ns
            ):
                return None
            pending = generation.pending_record_path(record.original_index)
            if pending.exists() or pending.is_symlink():
                return None
        except (
            CacheIOError,
            GenerationCacheError,
            OSError,
            SSCDEvaluationError,
            TypeError,
            ValueError,
        ):
            return None
    return SSCDResult(paths.summary_json, record_count, 0, True)


def _exact_artifact_indices(directory: Path, suffix: str) -> set[str] | None:
    """Return exact regular nonempty artifact coverage for one cache directory."""

    if directory.is_symlink() or not directory.is_dir():
        return None
    indices: set[str] = set()
    try:
        for path in directory.iterdir():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix != suffix
                or path.stat().st_size <= 0
            ):
                return None
            index = safe_index(path.stem)
            if index in indices:
                return None
            indices.add(index)
    except (GenerationCacheError, OSError, ValueError):
        return None
    return indices


def _validated_cached_scores(
    project: Path,
    generation: GenerationPaths,
    paths: SSCDPaths,
    record: CompletedGenerationRecord,
    run_configuration: Mapping[str, Any],
    configuration_hash: str,
    seed_values: Sequence[int],
    overwrite: bool,
) -> torch.Tensor | None:
    generation_validation = validate_generation_record(
        generation,
        record.original_index,
        expected_scientific_hash=str(run_configuration["scientific_config_hash"]),
        expected_record_identity=_generation_record_identity(record),
        load_tensors=False,
        tensor_names=("latent",),
        require_preview=False,
        verify_file_hashes=False,
    )
    if not generation_validation.valid:
        raise SSCDEvaluationError(
            "invalid completed generation record: "
            + "; ".join(generation_validation.errors)
        )
    if record.metadata.get("num_seeds") != len(seed_values) or record.metadata.get(
        "seeds"
    ) != list(seed_values):
        raise SSCDEvaluationError(
            "generation record seed block differs from generation configuration"
        )
    if overwrite:
        _safe_existing_score_artifacts(paths, record.original_index)
        return None
    scores = _load_cached_scores(paths, record, configuration_hash, seed_values)
    if scores is None:
        return None
    _generation_inputs_are_published(
        project,
        generation,
        record,
        publication_marker=paths.marker_path(record.original_index),
        verify_changed_files=True,
    )
    return scores


def _compute_pending_records(
    project: Path,
    generation: GenerationPaths,
    paths: SSCDPaths,
    records: Sequence[CompletedGenerationRecord],
    run_configuration: Mapping[str, Any],
    configuration: Mapping[str, Any],
    seed_values: Sequence[int],
    devices: Sequence[torch.device],
    overwrite: bool,
) -> _ShardResult:
    if not records:
        return _ShardResult((), (), 0)
    worker_count = worker_count_for_tasks(devices, len(records))
    active_devices = tuple(devices[:worker_count])
    shards = tuple(
        round_robin_shard(
            records,
            worker_index=worker_index,
            worker_count=worker_count,
        )
        for worker_index in range(worker_count)
    )
    if worker_count == 1:
        return _score_record_shard(
            project,
            generation,
            paths,
            shards[0],
            run_configuration,
            configuration,
            tuple(seed_values),
            active_devices[0],
            0,
            worker_count,
            overwrite,
        )
    if not all(selected.type == "cuda" for selected in active_devices):
        raise SSCDEvaluationError("multiple SSCD workers require concrete CUDA devices")

    print(
        "SSCD worker plan: "
        + "; ".join(
            f"{selected}={len(shard)} pending prompt(s)"
            for selected, shard in zip(active_devices, shards, strict=True)
        )
        + "."
    )

    context = multiprocessing.get_context("spawn")
    progress_lock = context.RLock()
    tqdm.set_lock(progress_lock)
    futures: list[tuple[torch.device, Any]] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_install_tqdm_lock,
        initargs=(progress_lock,),
    ) as executor:
        for worker_index, (selected, shard) in enumerate(
            zip(active_devices, shards, strict=True)
        ):
            future = executor.submit(
                _score_record_shard,
                project,
                generation,
                paths,
                shard,
                run_configuration,
                configuration,
                tuple(seed_values),
                selected,
                worker_index,
                worker_count,
                overwrite,
            )
            futures.append((selected, future))

        results: list[_ShardResult] = []
        for selected, future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                raise SSCDEvaluationError(
                    f"SSCD worker on {selected} failed: {error}"
                ) from error

    rows = sorted(
        (row for result in results for row in result.rows),
        key=lambda row: (row["source_row_number"], str(row["original_index"])),
    )
    failures = sorted(
        (failure for result in results for failure in result.failures),
        key=lambda row: (row["source_row_number"], str(row["original_index"])),
    )
    return _ShardResult(
        tuple(rows),
        tuple(failures),
        sum(result.newly_computed for result in results),
    )


def _score_record_shard(
    project: Path,
    generation: GenerationPaths,
    paths: SSCDPaths,
    records: Sequence[CompletedGenerationRecord],
    run_configuration: Mapping[str, Any],
    configuration: Mapping[str, Any],
    seed_values: Sequence[int],
    device: torch.device,
    progress_position: int,
    worker_count: int,
    overwrite: bool,
) -> _ShardResult:
    configure_worker_cpu_threads(worker_count)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    runtime: _Runtime | None = None
    runtime_failure: Exception | None = None
    newly_computed = 0
    progress = tqdm(
        records,
        total=len(records),
        desc=(
            f"[SSCD {device} shard {progress_position + 1}/{worker_count}] "
            "Computing prompts"
        ),
        unit="prompt",
        dynamic_ncols=True,
        leave=True,
        disable=False,
        position=progress_position,
    )
    for record in progress:
        status = "validating-latent"
        progress.set_postfix(
            id=record.original_index,
            status=status,
            completed=newly_computed,
            failed=len(failures),
        )
        try:
            _validate_generation_latent(
                generation,
                record,
                run_configuration,
            )
            if runtime is None:
                if runtime_failure is not None:
                    raise SSCDEvaluationError(
                        f"model runtime is unavailable on {device}: {runtime_failure}"
                    ) from runtime_failure
                try:
                    runtime = _load_runtime(
                        project, run_configuration, configuration, device
                    )
                except Exception as error:
                    runtime_failure = error
                    raise
            status = "scoring"
            progress.set_postfix(
                id=record.original_index,
                status=status,
                completed=newly_computed,
                failed=len(failures),
            )
            scores = _compute_record_scores(
                project,
                generation,
                paths,
                record,
                runtime,
                configuration,
                seed_values,
                overwrite,
            )
            newly_computed += 1
            rows.append(
                _manifest_row(project, record, scores, paths, configuration, "computed")
            )
            status = "computed"
        except Exception as error:
            failures.append(
                _failure_row(project, paths, record, error, traceback.format_exc())
            )
            status = "error"
        finally:
            progress.set_postfix(
                id=record.original_index,
                status=status,
                completed=newly_computed,
                failed=len(failures),
                refresh=False,
            )
    return _ShardResult(tuple(rows), tuple(failures), newly_computed)


def _generation_record_identity(
    record: CompletedGenerationRecord,
) -> dict[str, object]:
    return {
        "record_id": record.metadata.get("record_id"),
        "source_row_number": record.source_row_number,
        "prompt_raw": record.metadata.get("prompt_raw"),
        "target_image_sha256": record.metadata.get("target_image_sha256"),
    }


def _validate_generation_latent(
    generation: GenerationPaths,
    record: CompletedGenerationRecord,
    run_configuration: Mapping[str, Any],
) -> None:
    """Physically verify only the generation artifact consumed by SSCD."""

    validation = validate_generation_record(
        generation,
        record.original_index,
        expected_scientific_hash=str(run_configuration["scientific_config_hash"]),
        expected_record_identity=_generation_record_identity(record),
        load_tensors=False,
        tensor_names=("latent",),
        require_preview=False,
        verify_file_hashes=True,
    )
    if not validation.valid:
        raise SSCDEvaluationError(
            "invalid generation latent for SSCD: " + "; ".join(validation.errors)
        )


def _generation_latent_sha256(record: CompletedGenerationRecord) -> str:
    hashes = record.metadata.get("tensor_file_sha256")
    digest = hashes.get("latent") if isinstance(hashes, Mapping) else None
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise SSCDEvaluationError("generation latent SHA-256 is missing or invalid")
    return digest


def _generation_inputs_are_published(
    project: Path,
    generation: GenerationPaths,
    record: CompletedGenerationRecord,
    *,
    publication_marker: Path,
    verify_changed_files: bool,
) -> bool:
    """Validate SSCD's latent and target inputs without reading stable files."""

    if publication_marker.is_symlink() or not publication_marker.is_file():
        raise SSCDEvaluationError(
            f"unsafe SSCD publication marker: {publication_marker}"
        )
    publication_stat = publication_marker.stat()
    if publication_stat.st_size <= 0:
        raise SSCDEvaluationError(
            f"empty SSCD publication marker: {publication_marker}"
        )

    dataset_model = record.metadata.get("dataset_model")
    if not isinstance(dataset_model, str) or safe_index(dataset_model) != dataset_model:
        raise SSCDEvaluationError("generation dataset model is missing or unsafe")
    target_path = (
        project
        / "data"
        / "webster"
        / dataset_model
        / "images"
        / f"{safe_index(record.original_index)}.png"
    ).resolve()
    observed_target_path = record.metadata.get("target_image_path")
    if not isinstance(observed_target_path, str) or (
        Path(observed_target_path).expanduser().resolve() != target_path
    ):
        raise SSCDEvaluationError("generation target image path is noncanonical")

    scientific_inputs = (
        (
            "generation latent",
            generation.latent_path(record.original_index),
            _generation_latent_sha256(record),
        ),
        (
            "target image",
            target_path,
            record.metadata.get("target_image_sha256"),
        ),
    )
    for label, path, expected_hash in scientific_inputs:
        if (
            not isinstance(expected_hash, str)
            or _SHA256.fullmatch(expected_hash) is None
        ):
            raise SSCDEvaluationError(f"{label} SHA-256 is missing or invalid")
        if path.is_symlink() or not path.is_file():
            raise SSCDEvaluationError(f"{label} is missing or unsafe: {path}")
        artifact_stat = path.stat()
        if artifact_stat.st_size <= 0:
            raise SSCDEvaluationError(f"{label} is empty: {path}")
        changed_since_publication = (
            artifact_stat.st_mtime_ns > publication_stat.st_mtime_ns
            or artifact_stat.st_ctime_ns > publication_stat.st_ctime_ns
        )
        if not changed_since_publication:
            continue
        if not verify_changed_files:
            return False
        if file_sha256(path) != expected_hash:
            raise SSCDEvaluationError(f"{label} SHA-256 differs")
    return True


def _complete_sscd_marker_matches(
    marker: Mapping[str, Any],
    record: CompletedGenerationRecord,
    configuration: Mapping[str, Any],
    seed_values: Sequence[int],
) -> bool:
    """Check one fully published marker's scientific and tensor contract."""

    generation_marker_hash = marker.get("generation_record_sha256")
    expected = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "original_index": record.original_index,
        "record_id": record.metadata.get("record_id"),
        "source_row_number": record.source_row_number,
        "num_seeds": len(seed_values),
        "seeds": list(seed_values),
        "terminal_latent_index": -1,
        "similarity": SCORE_DEFINITION,
        "score_shape": [len(seed_values)],
        "score_dtype": "float32",
        "sscd_configuration_hash": configuration.get("configuration_hash"),
        "generation_latent_sha256": _generation_latent_sha256(record),
        "generation_scientific_config_hash": record.metadata.get(
            "scientific_config_hash"
        ),
        "target_image_sha256": record.metadata.get("target_image_sha256"),
        "model_id": configuration.get("model_id"),
        "model_revision": configuration.get("model_revision"),
        "vae_id": configuration.get("vae_id"),
        "vae_revision": configuration.get("vae_revision"),
        "decode_dtype": configuration.get("decode_dtype"),
        "sscd_model_name": configuration.get("sscd_model_name"),
        "sscd_checkpoint_path": configuration.get("sscd_checkpoint_path"),
        "sscd_checkpoint_url": configuration.get("sscd_checkpoint_url"),
        "sscd_checkpoint_sha256": configuration.get("sscd_checkpoint_sha256"),
        "sscd_feature_dimension": configuration.get("sscd_feature_dimension"),
        "sscd_input_size": configuration.get("sscd_input_size"),
        "sscd_preprocessing": configuration.get("sscd_preprocessing"),
        "sscd_preprocessing_hash": configuration.get("sscd_preprocessing_hash"),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        return False
    return (
        isinstance(marker.get("created_at"), str)
        and bool(marker.get("created_at"))
        and isinstance(marker.get("completed_at"), str)
        and bool(marker.get("completed_at"))
        and isinstance(marker.get("score_sha256"), str)
        and _SHA256.fullmatch(str(marker["score_sha256"])) is not None
        and isinstance(generation_marker_hash, str)
        and _SHA256.fullmatch(generation_marker_hash) is not None
    )


def _failure_row(
    project: Path,
    paths: SSCDPaths,
    record: CompletedGenerationRecord,
    error: Exception,
    formatted_traceback: str,
) -> dict[str, object]:
    trace_path = paths.traceback_directory / f"{record.original_index}.txt"
    atomic_write_text(trace_path, formatted_traceback)
    return {
        "original_index": record.original_index,
        "record_id": record.metadata.get("record_id"),
        "source_row_number": record.source_row_number,
        "webster_overfit_type": record.metadata.get("webster_overfit_type"),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback_path": _relative(trace_path, project),
    }


def _load_or_create_configuration(
    project: Path,
    paths: SSCDPaths,
    run: Mapping[str, Any],
    generation: GenerationPaths,
    seed_values: Sequence[int],
) -> dict[str, Any]:
    science = _science(run)
    num_seeds = len(seed_values)
    score_order = _score_order(seed_values)
    if paths.config_json.exists() or paths.config_json.is_symlink():
        if paths.config_json.is_symlink() or not paths.config_json.is_file():
            raise SSCDEvaluationError(f"unsafe SSCD configuration: {paths.config_json}")
        configuration = read_json(paths.config_json)
        stored_hash = configuration.get("configuration_hash")
        if stored_hash != sscd_configuration_hash(configuration):
            raise SSCDEvaluationError("SSCD configuration hash differs")
        expected = {
            "schema_version": SSCD_SCHEMA_VERSION,
            "generation_run_path": _relative(generation.run_directory, project),
            "generation_run_config_path": _relative(generation.run_config, project),
            "generation_scientific_config_hash": run["scientific_config_hash"],
            "selection_policy": SSCD_SELECTION_POLICY,
            "num_seeds": num_seeds,
            "seeds": list(seed_values),
            "terminal_latent_index": -1,
            "model_id": science["model_id"],
            "model_revision": science["model_revision"],
            "vae_id": science["vae_id"],
            "vae_revision": science["vae_revision"],
            "decode_dtype": "float32",
            "score_definition": SCORE_DEFINITION,
            "score_tensor_schema": {
                "shape": ["num_seeds"],
                "dtype": "float32",
                "device": "cpu",
                "contiguous": True,
                "order": score_order,
            },
            "duplicates_images_or_features": False,
            "feature_normalization": "explicit_l2_p2_dim1",
            "sscd_model_name": SSCD_MODEL_NAME,
            "sscd_checkpoint_path": "checkpoints/sscd/sscd_disc_large.torchscript.pt",
            "sscd_checkpoint_url": SSCD_CHECKPOINT_URL,
            "sscd_feature_dimension": SSCD_FEATURE_DIMENSION,
            "sscd_input_size": SSCD_INPUT_SIZE,
            "sscd_preprocessing": sscd_preprocessing_policy(),
            "sscd_preprocessing_hash": sscd_preprocessing_hash(),
        }
        wrong = [
            key for key, value in expected.items() if configuration.get(key) != value
        ]
        checkpoint_hash = configuration.get("sscd_checkpoint_sha256")
        if not (
            isinstance(checkpoint_hash, str)
            and len(checkpoint_hash) == 64
            and all(character in "0123456789abcdef" for character in checkpoint_hash)
        ):
            wrong.append("sscd_checkpoint_sha256")
        if wrong:
            raise SSCDEvaluationError(
                "existing SSCD scientific contract differs at: " + ", ".join(wrong)
            )
        return configuration

    checkpoint = ensure_sscd_checkpoint(project)
    configuration: dict[str, Any] = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "generation_run_path": _relative(generation.run_directory, project),
        "generation_run_config_path": _relative(generation.run_config, project),
        "generation_scientific_config_hash": run["scientific_config_hash"],
        "selection_policy": SSCD_SELECTION_POLICY,
        "num_seeds": num_seeds,
        "seeds": list(seed_values),
        "terminal_latent_index": -1,
        "model_id": science["model_id"],
        "model_revision": science["model_revision"],
        "vae_id": science["vae_id"],
        "vae_revision": science["vae_revision"],
        "decode_dtype": "float32",
        "score_definition": SCORE_DEFINITION,
        "score_tensor_schema": {
            "shape": ["num_seeds"],
            "dtype": "float32",
            "device": "cpu",
            "contiguous": True,
            "order": score_order,
        },
        "duplicates_images_or_features": False,
        "feature_normalization": "explicit_l2_p2_dim1",
        "sscd_model_name": SSCD_MODEL_NAME,
        "sscd_checkpoint_path": _relative(checkpoint.path, project),
        "sscd_checkpoint_url": checkpoint.url,
        "sscd_checkpoint_sha256": checkpoint.sha256,
        "sscd_feature_dimension": SSCD_FEATURE_DIMENSION,
        "sscd_input_size": SSCD_INPUT_SIZE,
        "sscd_preprocessing": sscd_preprocessing_policy(),
        "sscd_preprocessing_hash": sscd_preprocessing_hash(),
    }
    configuration["configuration_hash"] = sscd_configuration_hash(configuration)
    atomic_write_json(paths.config_json, configuration)
    return configuration


def _load_cached_scores(
    paths: SSCDPaths,
    record: CompletedGenerationRecord,
    configuration_hash: str,
    seed_values: Sequence[int],
) -> torch.Tensor | None:
    existing = _safe_existing_score_artifacts(paths, record.original_index)
    if not existing:
        return None
    try:
        return _read_cached_scores(
            paths,
            record,
            configuration_hash,
            seed_values,
        )
    except (
        CacheIOError,
        OSError,
        SSCDMetricError,
        SSCDEvaluationError,
        TypeError,
        ValueError,
    ):
        _quarantine_score_artifacts(paths, record.original_index)
        return None


def _read_cached_scores(
    paths: SSCDPaths,
    record: CompletedGenerationRecord,
    configuration_hash: str,
    seed_values: Sequence[int],
) -> torch.Tensor | None:
    num_seeds = len(seed_values)
    marker_path = paths.marker_path(record.original_index)
    score_path = paths.score_path(record.original_index)
    if not marker_path.exists() and not score_path.exists():
        return None
    if not marker_path.is_file() or marker_path.is_symlink():
        raise SSCDEvaluationError(f"unsafe SSCD marker: {marker_path}")
    if not score_path.is_file() or score_path.is_symlink():
        raise SSCDEvaluationError(f"SSCD score is missing: {score_path}")
    marker = read_json(marker_path)
    expected = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "original_index": record.original_index,
        "record_id": record.metadata.get("record_id"),
        "source_row_number": record.source_row_number,
        "num_seeds": num_seeds,
        "seeds": list(seed_values),
        "sscd_configuration_hash": configuration_hash,
        "generation_latent_sha256": _generation_latent_sha256(record),
        "generation_scientific_config_hash": record.metadata.get(
            "scientific_config_hash"
        ),
        "target_image_sha256": record.metadata.get("target_image_sha256"),
        "score_shape": [num_seeds],
        "score_dtype": "float32",
        "terminal_latent_index": -1,
        "similarity": SCORE_DEFINITION,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise SSCDEvaluationError(
                f"cached SSCD {key} differs for {record.original_index}"
            )
    if marker.get("score_sha256") != file_sha256(score_path):
        raise SSCDEvaluationError(
            f"cached SSCD hash differs for {record.original_index}"
        )
    try:
        scores = safe_torch_load(score_path)
    except CacheIOError as error:
        raise SSCDEvaluationError(str(error)) from error
    return validate_sscd_scores(scores, num_seeds)


def _safe_existing_score_artifacts(paths: SSCDPaths, index: object) -> tuple[Path, ...]:
    """Return regular derived artifacts while rejecting unsafe filesystem entries."""

    original_index = safe_index(index)
    labeled = (
        ("score", paths.score_path(original_index)),
        ("marker", paths.marker_path(original_index)),
    )
    existing: list[Path] = []
    for label, path in labeled:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise SSCDEvaluationError(f"unsafe SSCD {label}: {path}")
        existing.append(path)
    return tuple(existing)


def _quarantine_score_artifacts(paths: SSCDPaths, index: object) -> tuple[Path, ...]:
    """Move one stale regular score/marker pair into a run-local bundle."""

    original_index = safe_index(index)
    sources = _safe_existing_score_artifacts(paths, original_index)
    if not sources:
        return ()
    parent = paths.stale_directory
    if parent.exists() or parent.is_symlink():
        if parent.is_symlink() or not parent.is_dir():
            raise SSCDEvaluationError(f"unsafe SSCD stale directory: {parent}")
    else:
        parent.mkdir(parents=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = uuid.uuid4().hex
    destination = parent / f"{stamp}_{original_index}_{suffix}"
    staging = parent / f".{stamp}_{original_index}_{suffix}.pending"
    staging.mkdir()
    moved: list[tuple[Path, Path]] = []
    try:
        for source in sources:
            target = staging / source.name
            os.replace(source, target)
            moved.append((source, target))
        os.replace(staging, destination)
    except Exception:
        for source, target in reversed(moved):
            if (target.exists() or target.is_symlink()) and not (
                source.exists() or source.is_symlink()
            ):
                os.replace(target, source)
        try:
            staging.rmdir()
        except OSError:
            pass
        raise
    return tuple(destination / source.name for source in sources)


def _load_runtime(
    project: Path,
    run_configuration: Mapping[str, Any],
    configuration: Mapping[str, Any],
    device: torch.device,
) -> _Runtime:
    from utils.models.loading import load_vae_from_generation_config, select_runtime

    concrete_device = torch.device(device)
    if concrete_device.type == "cuda":
        if concrete_device.index is None:
            raise SSCDEvaluationError("worker CUDA device must have a concrete index")
        torch.cuda.set_device(concrete_device)
    selected = select_runtime(concrete_device)
    checkpoint_path = project / str(configuration["sscd_checkpoint_path"])
    checkpoint = SSCDCheckpoint(
        SSCD_MODEL_NAME,
        str(configuration["sscd_checkpoint_url"]),
        checkpoint_path,
        str(configuration["sscd_checkpoint_sha256"]),
    )
    descriptor = load_sscd_model(checkpoint, selected.device)
    loaded = load_vae_from_generation_config(run_configuration, runtime=selected)
    return _Runtime(checkpoint, descriptor, loaded.vae, selected.device)


def _compute_record_scores(
    project: Path,
    generation: GenerationPaths,
    paths: SSCDPaths,
    record: CompletedGenerationRecord,
    runtime: _Runtime,
    configuration: Mapping[str, Any],
    seed_values: Sequence[int],
    overwrite: bool,
) -> torch.Tensor:
    from utils.models.latent import decode_generated_latents

    num_seeds = len(seed_values)
    trajectory = safe_torch_load(generation.latent_path(record.original_index))
    if not isinstance(trajectory, torch.Tensor) or trajectory.shape[0] != num_seeds:
        raise SSCDEvaluationError("generation trajectory has an invalid seed dimension")
    generated = decode_generated_latents(trajectory[:, -1], runtime.vae, runtime.device)
    target_path = Path(str(record.metadata["target_image_path"])).expanduser().resolve()
    expected_target = (
        project
        / "data"
        / "webster"
        / str(record.metadata.get("dataset_model"))
        / "images"
        / f"{record.original_index}.png"
    ).resolve()
    if target_path != expected_target:
        raise SSCDEvaluationError("target image path is noncanonical")
    if not target_path.is_file() or target_path.is_symlink():
        raise SSCDEvaluationError(f"target image is missing: {target_path}")
    expected_target_hash = record.metadata.get("target_image_sha256")
    if (
        not isinstance(expected_target_hash, str)
        or file_sha256(target_path) != expected_target_hash
    ):
        raise SSCDEvaluationError("target image SHA-256 differs")
    with Image.open(target_path) as source:
        target = ImageOps.exif_transpose(source).convert("RGB")
    scores = compute_sscd_scores(target, generated, runtime.descriptor, runtime.device)
    score_path = paths.score_path(record.original_index)
    marker_path = paths.marker_path(record.original_index)
    existing = _safe_existing_score_artifacts(paths, record.original_index)
    if not overwrite and existing:
        raise SSCDEvaluationError("refusing to overwrite an incomplete SSCD cache")
    score_hash = atomic_torch_save(scores, score_path)
    metadata = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "original_index": record.original_index,
        "record_id": record.metadata.get("record_id"),
        "source_row_number": record.source_row_number,
        "prompt_raw": record.metadata.get("prompt_raw"),
        "webster_overfit_type": record.metadata.get("webster_overfit_type"),
        "generation_record_path": str(record.marker_path),
        "generation_record_sha256": file_sha256(record.marker_path),
        "generation_latent_path": str(generation.latent_path(record.original_index)),
        "generation_latent_sha256": _generation_latent_sha256(record),
        "generation_scientific_config_hash": record.metadata.get(
            "scientific_config_hash"
        ),
        "target_image_path": str(target_path),
        "target_image_sha256": record.metadata.get("target_image_sha256"),
        "score_path": str(score_path),
        "score_sha256": score_hash,
        "score_shape": [num_seeds],
        "score_dtype": "float32",
        "num_seeds": num_seeds,
        "seeds": list(seed_values),
        "terminal_latent_index": -1,
        "similarity": SCORE_DEFINITION,
        "sscd_configuration_hash": configuration["configuration_hash"],
        "sscd_model_name": SSCD_MODEL_NAME,
        "sscd_checkpoint_path": configuration["sscd_checkpoint_path"],
        "sscd_checkpoint_url": configuration["sscd_checkpoint_url"],
        "sscd_checkpoint_sha256": configuration["sscd_checkpoint_sha256"],
        "sscd_feature_dimension": SSCD_FEATURE_DIMENSION,
        "sscd_input_size": SSCD_INPUT_SIZE,
        "sscd_preprocessing": sscd_preprocessing_policy(),
        "sscd_preprocessing_hash": sscd_preprocessing_hash(),
        "model_id": record.metadata.get("model_id"),
        "model_revision": record.metadata.get("model_revision"),
        "vae_id": record.metadata.get("vae_id"),
        "vae_revision": record.metadata.get("vae_revision"),
        "decode_dtype": "float32",
    }
    atomic_write_json(marker_path, metadata)
    return scores


def _manifest_row(
    project: Path,
    record: CompletedGenerationRecord,
    scores: torch.Tensor,
    paths: SSCDPaths,
    configuration: Mapping[str, Any],
    status: str,
) -> dict[str, object]:
    values = scores.double()
    metadata = record.metadata
    return {
        "original_index": record.original_index,
        "record_id": metadata.get("record_id"),
        "source_row_number": record.source_row_number,
        "prompt_raw": metadata.get("prompt_raw"),
        "webster_overfit_type": metadata.get("webster_overfit_type"),
        "recovery_status": metadata.get("recovery_status"),
        "recovery_method": metadata.get("recovery_method"),
        "score_path": _relative(paths.score_path(record.original_index), project),
        "target_image_sha256": metadata.get("target_image_sha256"),
        "generation_latent_sha256": metadata.get("tensor_file_sha256", {}).get(
            "latent"
        ),
        "num_seeds": len(scores),
        "sscd_min": float(values.min()),
        "sscd_max": float(values.max()),
        "sscd_mean": float(values.mean()),
        "sscd_median": float(values.median()),
        "sscd_std": float(values.std(unbiased=False)),
        "model_id": metadata.get("model_id"),
        "model_revision": metadata.get("model_revision"),
        "vae_id": metadata.get("vae_id"),
        "vae_revision": metadata.get("vae_revision"),
        "sscd_model_name": SSCD_MODEL_NAME,
        "sscd_checkpoint_sha256": configuration["sscd_checkpoint_sha256"],
        "completed_or_resumed": status,
    }


def _summary(
    project: Path,
    generation: GenerationPaths,
    paths: SSCDPaths,
    manifest: pd.DataFrame,
    failures: list[dict[str, object]],
    configuration: Mapping[str, Any],
    generation_count: int,
    newly_computed: int,
    started: datetime,
    overwrite: bool,
) -> dict[str, object]:
    scores: list[float] = []
    for index in manifest.get("original_index", pd.Series(dtype=str)).astype(str):
        value = safe_torch_load(paths.score_path(index))
        if isinstance(value, torch.Tensor):
            scores.extend(value.double().tolist())
    series = pd.Series(scores, dtype="float64")
    return {
        "schema_version": SSCD_SCHEMA_VERSION,
        "outcome": (
            "completed"
            if len(manifest) == generation_count and not failures
            else "incomplete"
        ),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": (datetime.now(UTC) - started).total_seconds(),
        "generation_run_path": _relative(generation.run_directory, project),
        "generation_scientific_config_hash": configuration[
            "generation_scientific_config_hash"
        ],
        "sscd_configuration_hash": configuration["configuration_hash"],
        "generation_record_count": generation_count,
        "selected_prompt_count": generation_count,
        "completed_prompt_count": len(manifest),
        "newly_computed_prompt_count": newly_computed,
        "resumed_prompt_count": len(manifest) - newly_computed,
        "overwrite": overwrite,
        "failed_prompt_count": len(failures),
        "total_sscd_scores": len(scores),
        "num_seeds": configuration["num_seeds"],
        "selection_policy": SSCD_SELECTION_POLICY,
        "sscd_checkpoint_path": configuration["sscd_checkpoint_path"],
        "sscd_checkpoint_sha256": configuration["sscd_checkpoint_sha256"],
        "preprocessing": configuration["sscd_preprocessing"],
        "scores_all": {
            "scores": len(scores),
            "min": None if series.empty else float(series.min()),
            "max": None if series.empty else float(series.max()),
            "mean": None if series.empty else float(series.mean()),
            "median": None if series.empty else float(series.median()),
            "std": None if series.empty else float(series.std(ddof=0)),
        },
    }


def _validate_invocation(
    run: Mapping[str, Any],
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    seeds: int,
    seed_start: int,
) -> tuple[int, ...]:
    science = _science(run)
    expected = {
        "model_cli_name": model,
        "num_inference_steps": steps,
        "num_seeds": seeds,
        "guidance_scale": float(guidance),
    }
    for key, value in expected.items():
        if science.get(key) != value:
            raise SSCDEvaluationError(f"generation {key} differs")
    scheduler_value = science.get("scheduler")
    observed = (
        scheduler_value.get("name") if isinstance(scheduler_value, Mapping) else None
    )
    if observed != scheduler:
        raise SSCDEvaluationError("generation scheduler differs")
    expected_seeds = tuple(range(seed_start, seed_start + seeds))
    observed_seeds = science.get("seeds")
    if (
        isinstance(observed_seeds, (str, bytes))
        or not isinstance(observed_seeds, Sequence)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in observed_seeds
        )
    ):
        raise SSCDEvaluationError("generation seeds are invalid")
    seed_values = tuple(observed_seeds)
    if seed_values != expected_seeds:
        raise SSCDEvaluationError("generation seeds differ")
    return seed_values


def _score_order(seed_values: Sequence[int]) -> str:
    """Describe tensor position order while preserving the seed-zero contract."""

    values = tuple(seed_values)
    if values == tuple(range(len(values))):
        return "seed 0 through seed N - 1"
    return "same order as explicit seeds"


def _science(run: Mapping[str, Any]) -> Mapping[str, Any]:
    science = run.get("scientific_config")
    if not isinstance(science, Mapping):
        raise SSCDEvaluationError("generation scientific configuration is missing")
    return science


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())
