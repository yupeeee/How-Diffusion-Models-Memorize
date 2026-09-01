"""Paths, validation, resume decisions, and markers for generation caches."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from utils.common.cli import generation_run_name
from utils.common.io import (
    CacheIOError,
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)

GENERATION_SCHEMA_VERSION = 1
SAMPLER_CONTRACT_VERSION = 2
GENERATION_SELECTION_POLICY = "all_available_canonical_pairs_label_independent"


class GenerationCacheError(RuntimeError):
    """A generation cache is missing, unsafe, or scientifically incompatible."""


@dataclass(frozen=True, slots=True)
class GenerationPaths:
    """The stable on-disk layout of one reusable trajectory run."""

    run_directory: Path

    @property
    def image_directory(self) -> Path:
        return self.run_directory / "image"

    @property
    def latent_directory(self) -> Path:
        return self.run_directory / "latent"

    @property
    def noise_prediction_directory(self) -> Path:
        return self.run_directory / "noise_pred"

    @property
    def target_latent_directory(self) -> Path:
        return self.run_directory / "target_latent"

    @property
    def record_directory(self) -> Path:
        return self.run_directory / "record"

    @property
    def stale_directory(self) -> Path:
        return self.run_directory / "stale"

    @property
    def traceback_directory(self) -> Path:
        return self.run_directory / "tracebacks"

    @property
    def run_config(self) -> Path:
        return self.run_directory / "run_config.json"

    @property
    def schedule(self) -> Path:
        return self.run_directory / "schedule.pt"

    @property
    def manifest_parquet(self) -> Path:
        return self.run_directory / "manifest.parquet"

    @property
    def manifest_csv(self) -> Path:
        return self.run_directory / "manifest.csv"

    @property
    def skipped_csv(self) -> Path:
        return self.run_directory / "skipped.csv"

    @property
    def failed_csv(self) -> Path:
        return self.run_directory / "failed.csv"

    @property
    def summary_json(self) -> Path:
        return self.run_directory / "summary.json"

    def create(self) -> None:
        for path in (
            self.image_directory,
            self.latent_directory,
            self.noise_prediction_directory,
            self.target_latent_directory,
            self.record_directory,
            self.stale_directory,
            self.traceback_directory,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def image_path(self, index: object) -> Path:
        return self.image_directory / f"{safe_index(index)}.png"

    def latent_path(self, index: object) -> Path:
        return self.latent_directory / f"{safe_index(index)}.pt"

    def noise_prediction_path(self, index: object) -> Path:
        return self.noise_prediction_directory / f"{safe_index(index)}.pt"

    def target_latent_path(self, index: object) -> Path:
        return self.target_latent_directory / f"{safe_index(index)}.pt"

    def record_path(self, index: object) -> Path:
        return self.record_directory / f"{safe_index(index)}.json"

    def pending_record_path(self, index: object) -> Path:
        return self.record_directory / f".{safe_index(index)}.pending"


@dataclass(frozen=True, slots=True)
class CompletedGenerationRecord:
    """One validated public generation completion marker."""

    original_index: str
    source_row_number: int
    marker_path: Path
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CacheValidation:
    """A resume check and its precise validation errors."""

    valid: bool
    errors: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DiskEstimate:
    """Raw scientific tensor bytes required for unfinished records."""

    remaining_records: int
    bytes_per_record: int
    estimated_remaining_bytes: int
    safety_margin_bytes: int
    free_bytes: int

    @property
    def sufficient(self) -> bool:
        return self.free_bytes >= self.estimated_remaining_bytes + self.safety_margin_bytes


def generation_paths(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int = 0,
) -> GenerationPaths:
    """Resolve one generation run beneath ``logs``."""

    project = Path(root).expanduser().resolve()
    name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
    )
    return GenerationPaths(project / "logs" / name)


def require_generation_run(paths: GenerationPaths) -> dict[str, Any]:
    """Load the immutable generation configuration or raise a useful error."""

    if not paths.run_config.is_file():
        raise GenerationCacheError(f"generation run is missing: {paths.run_directory}")
    try:
        configuration = read_json(paths.run_config)
    except CacheIOError as error:
        raise GenerationCacheError(str(error)) from error
    science = configuration.get("scientific_config")
    if not isinstance(science, Mapping):
        raise GenerationCacheError("generation run has no scientific_config")
    expected = configuration.get("scientific_config_hash")
    observed = canonical_hash(science)
    if expected != observed:
        raise GenerationCacheError("generation scientific_config_hash differs")
    return configuration


def list_completed_records(paths: GenerationPaths) -> list[CompletedGenerationRecord]:
    """List only public ``record/*.json`` markers in source-row order."""

    directory = paths.record_directory
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise GenerationCacheError(f"unsafe record directory: {directory}")
    records: list[CompletedGenerationRecord] = []
    seen_indices: set[str] = set()
    seen_rows: set[int] = set()
    for marker in directory.iterdir():
        if marker.suffix != ".json":
            continue
        if marker.is_symlink() or not marker.is_file():
            raise GenerationCacheError(f"unsafe completion marker: {marker}")
        metadata = read_json(marker)
        index = safe_index(metadata.get("original_index"))
        row = metadata.get("source_row_number")
        if marker.stem != index:
            raise GenerationCacheError(f"marker filename/index mismatch: {marker}")
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise GenerationCacheError(f"invalid source row in {marker}")
        if index in seen_indices or row in seen_rows:
            raise GenerationCacheError(f"duplicate completion identity: {marker}")
        seen_indices.add(index)
        seen_rows.add(row)
        records.append(CompletedGenerationRecord(index, row, marker, metadata))
    return sorted(records, key=lambda item: (item.source_row_number, item.original_index))


def validate_generation_record(
    paths: GenerationPaths,
    index: object,
    *,
    expected_scientific_hash: str | None = None,
    expected_record_identity: Mapping[str, object] | None = None,
    load_tensors: bool = True,
) -> CacheValidation:
    """Validate a completion marker and all scientific tensor identities."""

    original_index = safe_index(index)
    marker = paths.record_path(original_index)
    if not marker.is_file() or marker.is_symlink():
        return CacheValidation(False, ("completion marker is missing",))
    try:
        metadata = read_json(marker)
    except CacheIOError as error:
        return CacheValidation(False, (str(error),))
    errors = _validate_record_metadata(metadata, original_index, expected_scientific_hash)
    if expected_record_identity is not None:
        for key in (
            "record_id",
            "source_row_number",
            "prompt_raw",
            "target_image_sha256",
        ):
            if metadata.get(key) != expected_record_identity.get(key):
                errors.append(f"marker {key} differs from current Webster metadata")
    tensor_paths = {
        "latent": paths.latent_path(original_index),
        "noise_prediction": paths.noise_prediction_path(original_index),
        "target_latent": paths.target_latent_path(original_index),
    }
    expected_hashes = metadata.get("tensor_file_sha256")
    for name, tensor_path in tensor_paths.items():
        if not tensor_path.is_file() or tensor_path.is_symlink():
            errors.append(f"{name} file is missing")
            continue
        expected_hash = (
            expected_hashes.get(name) if isinstance(expected_hashes, Mapping) else None
        )
        if not _is_sha256(expected_hash) or file_sha256(tensor_path) != expected_hash:
            errors.append(f"{name} SHA-256 differs")
            continue
        if load_tensors:
            try:
                payload = safe_torch_load(tensor_path)
                errors.extend(_validate_tensor_schema(name, payload, metadata))
            except CacheIOError as error:
                errors.append(str(error))
    preview = paths.image_path(original_index)
    expected_preview = metadata.get("preview_image_sha256")
    if not preview.is_file() or preview.is_symlink():
        errors.append("preview image is missing")
    elif not _is_sha256(expected_preview) or file_sha256(preview) != expected_preview:
        errors.append("preview image SHA-256 differs")
    return CacheValidation(not errors, tuple(errors), metadata)


def save_generation_tensors(
    paths: GenerationPaths,
    index: object,
    *,
    latents: torch.Tensor,
    unconditional_predictions: torch.Tensor,
    conditional_predictions: torch.Tensor,
    target_latent: torch.Tensor,
) -> dict[str, str]:
    """Write the three scientific files exactly once for an unfinished record."""

    original_index = safe_index(index)
    destinations = (
        paths.latent_path(original_index),
        paths.noise_prediction_path(original_index),
        paths.target_latent_path(original_index),
    )
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise GenerationCacheError(
            f"refusing to overwrite an existing scientific tensor for {original_index}"
        )
    latent_hash = atomic_torch_save(latents, destinations[0])
    prediction_hash = atomic_torch_save(
        (unconditional_predictions, conditional_predictions), destinations[1]
    )
    target_hash = atomic_torch_save(target_latent, destinations[2])
    return {
        "latent": latent_hash,
        "noise_prediction": prediction_hash,
        "target_latent": target_hash,
    }


def publish_completion_marker(
    paths: GenerationPaths, index: object, metadata: Mapping[str, object]
) -> Path:
    """Atomically publish a marker after tensors and preview are complete."""

    original_index = safe_index(index)
    marker = paths.record_path(original_index)
    if marker.exists() or marker.is_symlink():
        raise GenerationCacheError(f"refusing to overwrite completion marker {marker}")
    values = dict(metadata)
    values["original_index"] = original_index
    values["schema_version"] = GENERATION_SCHEMA_VERSION
    values["sampler_contract_version"] = SAMPLER_CONTRACT_VERSION
    values["selection_policy"] = GENERATION_SELECTION_POLICY
    values["completed_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(marker, values)
    return marker


def estimate_disk_space(
    paths: GenerationPaths,
    *,
    remaining_records: int,
    num_seeds: int,
    num_inference_steps: int,
    latent_shape: Sequence[int],
    element_size: int,
) -> DiskEstimate:
    """Estimate raw cache bytes and inspect free space."""

    elements = _product(latent_shape)
    latent_bytes = num_seeds * (num_inference_steps + 1) * elements * element_size
    predictions = 2 * num_seeds * num_inference_steps * elements * element_size
    target = elements * 4
    per_record = latent_bytes + predictions + target
    remaining = max(0, int(remaining_records)) * per_record
    paths.run_directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(paths.run_directory).free
    margin = max(1024**3, int(remaining * 0.05))
    return DiskEstimate(max(0, int(remaining_records)), per_record, remaining, margin, free)


def quarantine_record(paths: GenerationPaths, index: object) -> tuple[Path, ...]:
    """Move one invalid record bundle into its run-local stale directory."""

    original_index = safe_index(index)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = paths.stale_directory / f"{stamp}_{original_index}"
    sources = (
        paths.latent_path(original_index),
        paths.noise_prediction_path(original_index),
        paths.target_latent_path(original_index),
        paths.image_path(original_index),
        paths.record_path(original_index),
        paths.pending_record_path(original_index),
    )
    moved: list[Path] = []
    for source in sources:
        if not source.exists() and not source.is_symlink():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        if target.exists() or target.is_symlink():
            raise GenerationCacheError(f"stale destination already exists: {target}")
        os.replace(source, target)
        moved.append(target)
    return tuple(moved)


def safe_index(value: object) -> str:
    """Validate an original index before using it as a filename component."""

    text = str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
    if not text or text in {".", ".."} or Path(text).name != text:
        raise GenerationCacheError(f"unsafe original index: {value!r}")
    if "/" in text or "\\" in text or "\x00" in text:
        raise GenerationCacheError(f"unsafe original index: {value!r}")
    return text


def _validate_record_metadata(
    metadata: Mapping[str, object], index: str, science_hash: str | None
) -> list[str]:
    errors: list[str] = []
    if metadata.get("original_index") != index:
        errors.append("marker original_index differs")
    if metadata.get("schema_version") != GENERATION_SCHEMA_VERSION:
        errors.append("generation schema differs")
    if metadata.get("sampler_contract_version") != SAMPLER_CONTRACT_VERSION:
        errors.append("sampler contract differs")
    if metadata.get("selection_policy") != GENERATION_SELECTION_POLICY:
        errors.append("generation selection policy differs")
    if science_hash is not None and metadata.get("scientific_config_hash") != science_hash:
        errors.append("scientific configuration hash differs")
    return errors


def _validate_tensor_schema(
    name: str, payload: object, metadata: Mapping[str, object]
) -> list[str]:
    shapes = metadata.get("tensor_shapes")
    dtypes = metadata.get("tensor_dtypes")
    if name == "noise_prediction":
        if not isinstance(payload, tuple) or len(payload) != 2:
            return ["noise prediction payload is not a two-tensor tuple"]
        labels = ("unconditional_noise_predictions", "conditional_noise_predictions")
        tensors = payload
    else:
        labels = (name,)
        tensors = (payload,)
    errors: list[str] = []
    for label, tensor in zip(labels, tensors, strict=True):
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"{label} is not a tensor")
            continue
        expected_shape = shapes.get(label) if isinstance(shapes, Mapping) else None
        expected_dtype = dtypes.get(label) if isinstance(dtypes, Mapping) else None
        if list(tensor.shape) != expected_shape:
            errors.append(f"{label} shape differs")
        if str(tensor.dtype).removeprefix("torch.") != expected_dtype:
            errors.append(f"{label} dtype differs")
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            errors.append(f"{label} is not a contiguous CPU tensor")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            errors.append(f"{label} contains non-finite values")
    return errors


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("latent dimensions must be positive integers")
        result *= value
    return result
