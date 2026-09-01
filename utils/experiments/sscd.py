"""Full-resolution SSCD scoring for every completed generation record."""

from __future__ import annotations

import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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
    compute_sscd_scores,
    ensure_sscd_checkpoint,
    load_sscd_model,
    sscd_preprocessing_hash,
    sscd_preprocessing_policy,
    validate_sscd_scores,
)

from .cache import (
    CompletedGenerationRecord,
    GenerationPaths,
    generation_paths,
    list_completed_records,
    require_generation_run,
    safe_index,
    validate_generation_record,
)

SSCD_SCHEMA_VERSION = 1
SCORE_DEFINITION = "cosine_similarity_of_l2_normalized_sscd_descriptors"
SSCD_SELECTION_POLICY = "all_completed_generation_records"
# Kept exact because it is part of the supplied reference configuration hash.
_SSCD_CONFIGURATION_RECORD_POLICY = "completed_generation_cache_records"

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
    def stale_directory(self) -> Path:
        return self.run_directory / "stale" / "sscd"

    @property
    def traceback_directory(self) -> Path:
        return self.run_directory / "sscd_tracebacks"

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
            self.stale_directory,
            self.traceback_directory,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class SSCDResult:
    """Result paths and status from one cache-based evaluation."""

    summary_path: Path
    completed_prompt_count: int
    failed_prompt_count: int

    @property
    def exit_code(self) -> int:
        return int(self.failed_prompt_count > 0)


@dataclass(slots=True)
class _Runtime:
    checkpoint: SSCDCheckpoint
    descriptor: Any
    vae: Any
    device: torch.device


def run_sscd(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int = 0,
) -> SSCDResult:
    """Score every completed record, reusing valid per-prompt score tensors."""

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
    paths = SSCDPaths(generation.run_directory)
    paths.create()
    configuration = _load_or_create_configuration(
        project, paths, run_configuration, generation, seed_values
    )
    configuration_hash = str(configuration["configuration_hash"])

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    runtime: _Runtime | None = None
    newly_computed = 0
    for record in tqdm(
        records,
        total=len(records),
        desc="[SSCD] Scoring prompts",
        unit="prompt",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    ):
        try:
            generation_validation = validate_generation_record(
                generation,
                record.original_index,
                expected_scientific_hash=str(
                    run_configuration["scientific_config_hash"]
                ),
                load_tensors=False,
            )
            if not generation_validation.valid:
                raise SSCDEvaluationError(
                    "invalid completed generation record: "
                    + "; ".join(generation_validation.errors)
                )
            if (
                record.metadata.get("num_seeds") != len(seed_values)
                or record.metadata.get("seeds") != list(seed_values)
            ):
                raise SSCDEvaluationError(
                    "generation record seed block differs from generation configuration"
                )
            scores = _load_cached_scores(
                paths, generation, record, configuration_hash, seed_values
            )
            status = "resumed"
            if scores is None:
                runtime = runtime or _load_runtime(project, run_configuration, configuration)
                scores = _compute_record_scores(
                    generation, paths, record, runtime, configuration, seed_values
                )
                newly_computed += 1
                status = "computed"
            rows.append(_manifest_row(project, record, scores, paths, configuration, status))
        except Exception as error:
            trace_path = paths.traceback_directory / f"{record.original_index}.txt"
            atomic_write_text(trace_path, traceback.format_exc())
            failures.append(
                {
                    "original_index": record.original_index,
                    "record_id": record.metadata.get("record_id"),
                    "source_row_number": record.source_row_number,
                    "webster_overfit_type": record.metadata.get("webster_overfit_type"),
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback_path": _relative(trace_path, project),
                }
            )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    failure_columns = (
        "original_index", "record_id", "source_row_number", "webster_overfit_type",
        "exception_type", "exception_message", "traceback_path",
    )
    failed = pd.DataFrame(failures, columns=failure_columns)
    atomic_write_frame_parquet(manifest, paths.manifest_parquet)
    atomic_write_frame_csv(manifest, paths.manifest_csv)
    atomic_write_frame_csv(failed, paths.failed_csv)
    summary = _summary(
        project, generation, paths, manifest, failures, configuration,
        len(records), newly_computed, started,
    )
    atomic_write_json(paths.summary_json, summary)
    return SSCDResult(paths.summary_json, len(rows), len(failures))


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
    if paths.config_json.is_file():
        configuration = read_json(paths.config_json)
        stored_hash = configuration.get("configuration_hash")
        values = dict(configuration)
        values.pop("configuration_hash", None)
        if stored_hash != canonical_hash(values):
            raise SSCDEvaluationError("SSCD configuration hash differs")
        expected = {
            "schema_version": SSCD_SCHEMA_VERSION,
            "generation_run_path": _relative(generation.run_directory, project),
            "generation_run_config_path": _relative(generation.run_config, project),
            "generation_scientific_config_hash": run["scientific_config_hash"],
            "selection_policy": _SSCD_CONFIGURATION_RECORD_POLICY,
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
            key for key, value in expected.items()
            if configuration.get(key) != value
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
                "existing SSCD scientific contract differs at: "
                + ", ".join(wrong)
            )
        return configuration

    checkpoint = ensure_sscd_checkpoint(project)
    configuration: dict[str, Any] = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "generation_run_path": _relative(generation.run_directory, project),
        "generation_run_config_path": _relative(generation.run_config, project),
        "generation_scientific_config_hash": run["scientific_config_hash"],
        "selection_policy": _SSCD_CONFIGURATION_RECORD_POLICY,
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
            "shape": ["num_seeds"], "dtype": "float32", "device": "cpu",
            "contiguous": True, "order": score_order,
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
    configuration["configuration_hash"] = canonical_hash(configuration)
    atomic_write_json(paths.config_json, configuration)
    return configuration


def _load_cached_scores(
    paths: SSCDPaths,
    generation: GenerationPaths,
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
        "generation_record_sha256": file_sha256(record.marker_path),
        "generation_latent_sha256": file_sha256(
            generation.latent_path(record.original_index)
        ),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise SSCDEvaluationError(
                f"cached SSCD {key} differs for {record.original_index}"
            )
    if marker.get("score_sha256") != file_sha256(score_path):
        raise SSCDEvaluationError(f"cached SSCD hash differs for {record.original_index}")
    try:
        scores = safe_torch_load(score_path)
    except CacheIOError as error:
        raise SSCDEvaluationError(str(error)) from error
    return validate_sscd_scores(scores, num_seeds)


def _load_runtime(
    project: Path,
    run_configuration: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> _Runtime:
    from utils.models.loading import load_vae_from_generation_config, select_runtime

    selected = select_runtime()
    checkpoint_path = project / str(configuration["sscd_checkpoint_path"])
    checkpoint = SSCDCheckpoint(
        SSCD_MODEL_NAME,
        str(configuration["sscd_checkpoint_url"]),
        checkpoint_path,
        str(configuration["sscd_checkpoint_sha256"]),
    )
    descriptor = load_sscd_model(checkpoint, selected.device)
    loaded = load_vae_from_generation_config(
        run_configuration, runtime=selected
    )
    return _Runtime(checkpoint, descriptor, loaded.vae, selected.device)


def _compute_record_scores(
    generation: GenerationPaths,
    paths: SSCDPaths,
    record: CompletedGenerationRecord,
    runtime: _Runtime,
    configuration: Mapping[str, Any],
    seed_values: Sequence[int],
) -> torch.Tensor:
    from utils.models.latent import decode_generated_latents

    num_seeds = len(seed_values)
    trajectory = safe_torch_load(generation.latent_path(record.original_index))
    if not isinstance(trajectory, torch.Tensor) or trajectory.shape[0] != num_seeds:
        raise SSCDEvaluationError("generation trajectory has an invalid seed dimension")
    generated = decode_generated_latents(trajectory[:, -1], runtime.vae, runtime.device)
    target_path = Path(str(record.metadata["target_image_path"])).expanduser().resolve()
    project = generation.run_directory.parent.parent.resolve()
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
    if score_path.exists() or marker_path.exists():
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
        "generation_latent_sha256": file_sha256(
            generation.latent_path(record.original_index)
        ),
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
) -> dict[str, object]:
    scores: list[float] = []
    for index in manifest.get("original_index", pd.Series(dtype=str)).astype(str):
        value = safe_torch_load(paths.score_path(index))
        if isinstance(value, torch.Tensor):
            scores.extend(value.double().tolist())
    series = pd.Series(scores, dtype="float64")
    return {
        "schema_version": SSCD_SCHEMA_VERSION,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": (datetime.now(UTC) - started).total_seconds(),
        "generation_run_path": _relative(generation.run_directory, project),
        "generation_scientific_config_hash": configuration[
            "generation_scientific_config_hash"
        ],
        "generation_record_count": generation_count,
        "selected_prompt_count": generation_count,
        "completed_prompt_count": len(manifest),
        "newly_computed_prompt_count": newly_computed,
        "resumed_prompt_count": len(manifest) - newly_computed,
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
    run: Mapping[str, Any], model: str, scheduler: str, guidance: float,
    steps: int, seeds: int, seed_start: int,
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
    observed = scheduler_value.get("name") if isinstance(scheduler_value, Mapping) else None
    if observed != scheduler:
        raise SSCDEvaluationError("generation scheduler differs")
    expected_seeds = tuple(range(seed_start, seed_start + seeds))
    observed_seeds = science.get("seeds")
    if (
        isinstance(observed_seeds, (str, bytes))
        or not isinstance(observed_seeds, Sequence)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in observed_seeds)
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
