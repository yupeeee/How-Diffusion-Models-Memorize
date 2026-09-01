"""Generate and resume full trajectories for every available Webster pair."""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import sys
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from utils.common.cli import validate_seed_block
from utils.common.io import (
    atomic_torch_save,
    atomic_write_frame_csv,
    atomic_write_frame_parquet,
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
    safe_torch_load,
)
from utils.models.registry import get_model_spec

from .cache import (
    GENERATION_SCHEMA_VERSION,
    GENERATION_SELECTION_POLICY,
    SAMPLER_CONTRACT_VERSION,
    GenerationPaths,
    estimate_disk_space,
    generation_paths,
    publish_completion_marker,
    quarantine_record,
    require_generation_run,
    save_generation_tensors,
    validate_generation_record,
)

MANIFEST_COLUMNS = (
    "original_index", "record_id", "source_row_number", "prompt_raw",
    "webster_overfit_type", "recovery_status", "recovery_method",
    "target_image_sha256", "latent_path", "noise_prediction_path",
    "target_latent_path", "preview_image_path", "N", "T", "latent_shape",
    "latent_dtype", "prediction_dtype", "model_id", "model_revision",
    "vae_id", "vae_revision", "scheduler", "guidance_scale",
    "completed_or_resumed",
)


class GenerationError(RuntimeError):
    """A trajectory generation run cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Final report path, status, and record counts."""

    summary_path: Path
    completed_rows: int
    failed_rows: int

    @property
    def exit_code(self) -> int:
        return int(self.failed_rows > 0)


@dataclass(slots=True)
class _Runtime:
    components: Any
    scheduler: Any
    scheduler_description: Mapping[str, object]
    latent_shape: tuple[int, int, int]


def generate_webster_trajectories(
    project_root: str | Path,
    *,
    model_name: str = "sdv1",
    scheduler_name: str = "ddim",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_seeds: int = 20,
    seed_start: int = 0,
    downscale: int = 4,
) -> GenerationResult:
    """Generate every available prompt-target pair or reuse its valid bundle."""

    started = datetime.now(UTC)
    root = Path(project_root).expanduser().resolve()
    try:
        seed_start, num_seeds = validate_seed_block(seed_start, num_seeds)
    except ValueError as error:
        raise GenerationError(str(error)) from error
    spec = get_model_spec(model_name)
    guidance, steps, seed_count, first_seed, divisor = _validated_arguments(
        scheduler_name, guidance_scale, num_inference_steps, num_seeds,
        seed_start, downscale, spec.resolution,
    )
    seed_values = tuple(range(first_seed, first_seed + seed_count))
    paths = generation_paths(
        root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=seed_count,
        seed_start=first_seed,
    )
    paths.create()
    dataset = _load_dataset(root, spec.dataset_model)
    metadata_rows = list(dataset.iter_metadata())
    existing_config = require_generation_run(paths) if paths.run_config.exists() else None
    if existing_config is not None:
        _validate_existing_invocation(
            existing_config, model_name, scheduler_name, guidance, steps,
            seed_values,
        )
    missing_markers = sum(
        not paths.record_path(row["original_index"]).is_file()
        for row in metadata_rows
    )
    latent_side = spec.resolution // 8
    disk_estimate = estimate_disk_space(
        paths,
        remaining_records=missing_markers,
        num_seeds=seed_count,
        num_inference_steps=steps,
        latent_shape=(4, latent_side, latent_side),
        element_size=torch.tensor([], dtype=torch.float32).element_size(),
    )
    print(
        "Generation cache preflight: "
        f"{disk_estimate.remaining_records} unfinished records, "
        f"{disk_estimate.bytes_per_record} bytes/record, "
        f"{disk_estimate.free_bytes} bytes free."
    )
    if not disk_estimate.sufficient:
        raise GenerationError(
            "insufficient disk space for unfinished generation records plus "
            "the required safety margin"
        )

    runtime: _Runtime | None = None
    run_config = existing_config
    manifest_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    status_by_index: dict[str, str] = {}
    target_cache: dict[str, torch.Tensor] = {}
    record_progress = tqdm(
        enumerate(metadata_rows),
        total=len(metadata_rows),
        desc="[Generation] Records",
        unit="record",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    )
    for position, metadata in record_progress:
        index = str(metadata["original_index"])
        record_progress.set_postfix_str(f"id={index} phase=validating")
        try:
            science_hash = (
                str(run_config["scientific_config_hash"])
                if run_config is not None
                else None
            )
            validation = validate_generation_record(
                paths,
                index,
                expected_scientific_hash=science_hash,
                expected_record_identity=metadata,
                load_tensors=False,
            )
            if validation.valid:
                status_by_index[index] = "resumed"
                manifest_rows.append(
                    _manifest_row(root, paths, validation.metadata or {}, "resumed")
                )
                continue
            if _preview_only_invalid(validation.errors, validation.metadata):
                if run_config is None:
                    raise GenerationError("preview recovery has no run configuration")
                if runtime is None:
                    record_progress.set_postfix_str(
                        f"id={index} phase=loading-model"
                    )
                    runtime = _load_runtime(spec, scheduler_name, steps)
                record_progress.set_postfix_str(
                    f"id={index} phase=preview-recovery"
                )
                with tqdm(
                    total=seed_count,
                    desc=f"[Generation] Preview {index}",
                    unit="image",
                    dynamic_ncols=True,
                    leave=False,
                    disable=False,
                    position=1,
                ) as preview_progress:
                    _regenerate_preview(
                        paths, index, validation.metadata or {}, runtime,
                        spec.resolution, divisor,
                        progress_callback=preview_progress.update,
                    )
                repaired = validate_generation_record(
                    paths, index,
                    expected_scientific_hash=str(run_config["scientific_config_hash"]),
                    expected_record_identity=metadata,
                    load_tensors=False,
                )
                if not repaired.valid:
                    raise GenerationError("preview recovery did not validate")
                status_by_index[index] = "preview_regenerated"
                manifest_rows.append(
                    _manifest_row(root, paths, repaired.metadata or {}, "resumed")
                )
                continue
            if validation.metadata is not None or any(
                path.exists() or path.is_symlink()
                for path in _record_paths(paths, index)
            ):
                quarantine_record(paths, index)
            item = dataset[position]
            image = item.get("image")
            prompt = item.get("prompt")
            if not isinstance(image, Image.Image) or not isinstance(prompt, str):
                skipped_rows.append(_skip_row(metadata, "prompt or target image unavailable"))
                continue
            if runtime is None:
                record_progress.set_postfix_str(
                    f"id={index} phase=loading-model"
                )
                runtime = _load_runtime(spec, scheduler_name, steps)
            if run_config is None:
                run_config = _create_run_config(
                    root, paths, dataset, spec, runtime, scheduler_name,
                    guidance, steps, seed_values, divisor,
                )
            target_hash = str(metadata.get("target_image_sha256") or "")
            target_latent = target_cache.get(target_hash)
            if target_latent is None:
                target_latent = _encode_target(
                    image, runtime.components.vae, runtime.components.device,
                    spec.resolution,
                )
                target_cache[target_hash] = target_latent
            record_progress.set_postfix_str(f"id={index} phase=denoising")
            with tqdm(
                total=seed_count * steps,
                desc=f"[Generation] Denoising {index}",
                unit="seed-step",
                dynamic_ncols=True,
                leave=False,
                disable=False,
                position=1,
            ) as denoising_progress:
                batch = _sample(
                    prompt, runtime, scheduler_name, guidance, steps, seed_values,
                    progress_callback=denoising_progress.update,
                )
            record_progress.set_postfix_str(f"id={index} phase=saving")
            tensor_hashes = save_generation_tensors(
                paths,
                index,
                latents=batch.latents,
                unconditional_predictions=batch.unconditional_noise_predictions,
                conditional_predictions=batch.conditional_noise_predictions,
                target_latent=target_latent,
            )
            record_progress.set_postfix_str(f"id={index} phase=preview")
            with tqdm(
                total=seed_count,
                desc=f"[Generation] Preview {index}",
                unit="image",
                dynamic_ncols=True,
                leave=False,
                disable=False,
                position=1,
            ) as preview_progress:
                preview_hash = _write_preview(
                    batch.latents[:, -1], runtime.components.vae,
                    runtime.components.device, paths.image_path(index),
                    spec.resolution, divisor,
                    progress_callback=preview_progress.update,
                )
            record = _record_metadata(
                root, paths, metadata, run_config, runtime, scheduler_name,
                guidance, steps, seed_values, divisor, tensor_hashes, preview_hash,
            )
            publish_completion_marker(paths, index, record)
            record_progress.set_postfix_str(
                f"id={index} phase=validating-output"
            )
            completed = validate_generation_record(
                paths, index,
                expected_scientific_hash=str(run_config["scientific_config_hash"]),
                expected_record_identity=metadata,
                load_tensors=False,
            )
            if not completed.valid:
                raise GenerationError("newly generated record did not validate: " + "; ".join(completed.errors))
            status_by_index[index] = "generated"
            manifest_rows.append(
                _manifest_row(root, paths, completed.metadata or {}, "generated")
            )
        except Exception as error:
            trace_path = paths.traceback_directory / f"{index}.txt"
            atomic_write_text(trace_path, traceback.format_exc())
            failed_rows.append(
                {
                    "original_index": index,
                    "record_id": metadata.get("record_id"),
                    "source_row_number": metadata.get("source_row_number"),
                    "webster_overfit_type": metadata.get("overfit_type"),
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback_path": _relative(trace_path, root),
                }
            )
            tqdm.write(
                f"[Generation] Record {index} failed: "
                f"{type(error).__name__}: {error}; traceback: "
                f"{_relative(trace_path, root)}",
                file=sys.stderr,
            )
        finally:
            record_progress.set_postfix(
                completed=len(manifest_rows),
                skipped=len(skipped_rows),
                failed=len(failed_rows),
                refresh=False,
            )

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values(
        ["source_row_number", "original_index"], kind="stable"
    ).reset_index(drop=True)
    skipped_columns = (
        "original_index", "record_id", "source_row_number",
        "webster_overfit_type", "reason",
    )
    failed_columns = (
        "original_index", "record_id", "source_row_number", "webster_overfit_type",
        "exception_type", "exception_message", "traceback_path",
    )
    atomic_write_frame_parquet(manifest, paths.manifest_parquet)
    atomic_write_frame_csv(manifest, paths.manifest_csv)
    atomic_write_frame_csv(pd.DataFrame(skipped_rows, columns=skipped_columns), paths.skipped_csv)
    atomic_write_frame_csv(pd.DataFrame(failed_rows, columns=failed_columns), paths.failed_csv)
    summary = _summary(
        root, paths, dataset, manifest, skipped_rows, failed_rows,
        status_by_index, run_config, disk_estimate, started,
    )
    atomic_write_json(paths.summary_json, summary)
    return GenerationResult(paths.summary_json, len(manifest), len(failed_rows))


def _load_dataset(root: Path, dataset_model: str) -> Any:
    from utils.data.webster import WebsterDataset

    return WebsterDataset(root, dataset_model, recovered_only=True)


def _load_runtime(spec: Any, scheduler_name: str, steps: int) -> _Runtime:
    from utils.models.loading import (
        load_model_components,
        preflight_model_components,
        select_runtime,
    )
    from utils.models.sampling import latent_shape_from_unet
    from utils.models.schedulers import build_scheduler

    selected = select_runtime()
    components = load_model_components(spec, runtime=selected)
    scheduler_build = build_scheduler(components.original_scheduler, scheduler_name)
    preflight_model_components(
        components, scheduler=scheduler_build.scheduler,
        num_inference_steps=steps,
    )
    shape = latent_shape_from_unet(components.unet)
    return _Runtime(
        components, scheduler_build.scheduler,
        {
            "name": scheduler_build.name,
            "class": scheduler_build.class_name,
            "config": scheduler_build.config,
            "removed_config_keys": list(scheduler_build.removed_config_keys),
        },
        shape,
    )


def _sample(
    prompt: str, runtime: _Runtime, scheduler_name: str,
    guidance: float, steps: int, seed_values: Sequence[int],
    progress_callback: Callable[[int], None] | None = None,
) -> Any:
    from utils.models.sampling import sample_trajectory

    components = runtime.components
    return sample_trajectory(
        prompt=prompt,
        seeds=tuple(seed_values),
        tokenizer=components.tokenizer,
        text_encoder=components.text_encoder,
        unet=components.unet,
        scheduler=runtime.scheduler,
        device=components.device,
        inference_dtype=components.inference_dtype,
        guidance_scale=guidance,
        num_inference_steps=steps,
        scheduler_name=scheduler_name,
        initial_microbatch_size=4,
        progress_callback=progress_callback,
    )


def _encode_target(
    image: Image.Image, vae: Any, device: torch.device, resolution: int
) -> torch.Tensor:
    from utils.models.latent import encode_target_latent

    return encode_target_latent(image, vae, device, resolution=resolution)


def _write_preview(
    final_latents: torch.Tensor,
    vae: Any,
    device: torch.device,
    destination: Path,
    resolution: int,
    downscale: int,
    progress_callback: Callable[[int], None] | None = None,
) -> str:
    from utils.models.latent import decode_generated_latents

    images = decode_generated_latents(
        final_latents,
        vae,
        device,
        progress_callback=progress_callback,
    )
    if not images:
        raise GenerationError("preview decoder returned no images")
    tile_size = resolution // downscale
    columns = min(10, len(images))
    rows = math.ceil(len(images) / columns)
    montage = Image.new("RGB", (columns * tile_size, rows * tile_size))
    for position, image in enumerate(images):
        tile = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        montage.paste(tile, ((position % columns) * tile_size, (position // columns) * tile_size))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        montage.save(temporary, format="PNG")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(destination)


def _regenerate_preview(
    paths: GenerationPaths,
    index: str,
    metadata: Mapping[str, Any],
    runtime: _Runtime,
    resolution: int,
    downscale: int,
    progress_callback: Callable[[int], None] | None = None,
) -> None:
    trajectory = safe_torch_load(paths.latent_path(index))
    if not isinstance(trajectory, torch.Tensor):
        raise GenerationError("cached trajectory is not a tensor")
    digest = _write_preview(
        trajectory[:, -1], runtime.components.vae, runtime.components.device,
        paths.image_path(index), resolution, downscale,
        progress_callback=progress_callback,
    )
    updated = dict(metadata)
    updated["preview_image_sha256"] = digest
    updated["preview_downscale"] = downscale
    updated["preview_status"] = "complete"
    updated["preview_completed_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(paths.record_path(index), updated)


def _create_run_config(
    root: Path,
    paths: GenerationPaths,
    dataset: Any,
    spec: Any,
    runtime: _Runtime,
    scheduler_name: str,
    guidance: float,
    steps: int,
    seed_values: Sequence[int],
    downscale: int,
) -> dict[str, object]:
    from utils.models.latent import TARGET_LATENT_DEFINITION, target_preprocessing_policy
    from utils.models.prediction_conversion import (
        prediction_conversion_description,
        scheduler_prediction_type,
    )
    from utils.models.schedule_metadata import build_schedule_metadata

    components = runtime.components
    seed_count = len(seed_values)
    native_type = scheduler_prediction_type(runtime.scheduler)
    science: dict[str, object] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "sampler_contract_version": SAMPLER_CONTRACT_VERSION,
        "selection_policy": GENERATION_SELECTION_POLICY,
        "target_role": "canonical_paired_source",
        "model_cli_name": spec.cli_name,
        "dataset_model": spec.dataset_model,
        "dataset_directory": str((root / "data" / "webster").resolve()),
        "dataset_manifest_path": str(dataset.metadata_path.resolve()),
        "model_id": spec.model_id,
        "model_revision": components.model_revision,
        "vae_id": components.vae_id,
        "vae_revision": components.vae_revision,
        "resolution": spec.resolution,
        "scheduler": dict(runtime.scheduler_description),
        "native_prediction_type": native_type,
        "stored_prediction_type": "epsilon",
        "prediction_conversion": prediction_conversion_description(native_type),
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "num_seeds": seed_count,
        "seeds": list(seed_values),
        "latent_shape": list(runtime.latent_shape),
        "inference_dtype": str(components.inference_dtype).removeprefix("torch."),
        "device_metadata": dict(components.device_metadata),
        "package_versions": dict(components.package_versions),
        "trajectory_order": "noise_to_image",
        "latent_index_description": "latents[:, 0] is x_T; latents[:, T] is terminal x_0",
        "prediction_index_description": "branches[:, i] are epsilon at latents[:, i]",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
        "target_preprocessing": target_preprocessing_policy(spec.resolution),
        "scientific_tensor_storage": {
            "dtype": str(components.inference_dtype).removeprefix("torch."),
            "device": "cpu", "layout": "contiguous",
        },
    }
    preview = {
        "downscale": downscale,
        "resolution": spec.resolution,
        "tile_size": [spec.resolution // downscale] * 2,
        "samples_per_row": 10,
        "sample_order": "seed_ascending",
        "resize_resampling": "Pillow Lanczos",
        "format": "PNG",
        "scientific_metric_input": False,
    }
    configuration: dict[str, object] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "scientific_config": science,
        "scientific_config_hash": canonical_hash(science),
        "preview_config": preview,
        "preview_config_hash": canonical_hash(preview),
        "runtime_provenance": _runtime_provenance(root),
    }
    atomic_write_json(paths.run_config, configuration)
    schedule = build_schedule_metadata(
        runtime.scheduler, scheduler_name,
        num_inference_steps=steps, device=components.device,
    )
    atomic_torch_save(schedule, paths.schedule)
    return configuration


def _record_metadata(
    root: Path,
    paths: GenerationPaths,
    metadata: Mapping[str, object],
    run_config: Mapping[str, object],
    runtime: _Runtime,
    scheduler_name: str,
    guidance: float,
    steps: int,
    seed_values: Sequence[int],
    downscale: int,
    tensor_hashes: Mapping[str, str],
    preview_hash: str,
) -> dict[str, object]:
    from utils.models.latent import TARGET_LATENT_DEFINITION, target_preprocessing_policy
    from utils.models.prediction_conversion import scheduler_prediction_type

    index = str(metadata["original_index"])
    components = runtime.components
    seed_count = len(seed_values)
    target_path = (root / "data" / "webster" / str(metadata["image_path"])).resolve()
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "record_id": metadata["record_id"],
        "source_row_number": metadata["source_row_number"],
        "prompt_raw": metadata["prompt_raw"],
        "webster_overfit_type": metadata.get("overfit_type"),
        "recovery_status": metadata.get("recovery_status"),
        "recovery_method": metadata.get("recovery_method"),
        "model_cli_name": components.spec.cli_name,
        "dataset_model": components.spec.dataset_model,
        "model_id": components.spec.model_id,
        "model_revision": components.model_revision,
        "vae_id": components.vae_id,
        "vae_revision": components.vae_revision,
        "scheduler_name": scheduler_name,
        "scheduler_class": type(runtime.scheduler).__name__,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "num_seeds": seed_count,
        "seeds": list(seed_values),
        "latent_shape": list(runtime.latent_shape),
        "inference_dtype": str(components.inference_dtype).removeprefix("torch."),
        "native_prediction_type": scheduler_prediction_type(runtime.scheduler),
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
        "scientific_config_hash": run_config["scientific_config_hash"],
        "target_image_path": str(target_path),
        "target_image_sha256": metadata["target_image_sha256"],
        "target_latent_definition": TARGET_LATENT_DEFINITION,
        "target_preprocessing": target_preprocessing_policy(components.spec.resolution),
        "latent_path": f"latent/{index}.pt",
        "noise_prediction_path": f"noise_pred/{index}.pt",
        "target_latent_path": f"target_latent/{index}.pt",
        "preview_image_path": f"image/{index}.png",
        "schedule_path": "schedule.pt",
        "tensor_file_sha256": dict(tensor_hashes),
        "preview_image_sha256": preview_hash,
        "preview_downscale": downscale,
        "preview_status": "complete",
        "tensor_shapes": {
            "latent": [seed_count, steps + 1, *runtime.latent_shape],
            "unconditional_noise_predictions": [seed_count, steps, *runtime.latent_shape],
            "conditional_noise_predictions": [seed_count, steps, *runtime.latent_shape],
            "target_latent": list(runtime.latent_shape),
        },
        "tensor_dtypes": {
            "latent": str(components.inference_dtype).removeprefix("torch."),
            "unconditional_noise_predictions": str(components.inference_dtype).removeprefix("torch."),
            "conditional_noise_predictions": str(components.inference_dtype).removeprefix("torch."),
            "target_latent": "float32",
        },
        "runtime_provenance": _runtime_provenance(root),
    }


def _manifest_row(
    root: Path, paths: GenerationPaths, metadata: Mapping[str, object], status: str
) -> dict[str, object]:
    science_shape = metadata.get("latent_shape")
    dtypes = metadata.get("tensor_dtypes")
    return {
        "original_index": metadata.get("original_index"),
        "record_id": metadata.get("record_id"),
        "source_row_number": metadata.get("source_row_number"),
        "prompt_raw": metadata.get("prompt_raw"),
        "webster_overfit_type": metadata.get("webster_overfit_type"),
        "recovery_status": metadata.get("recovery_status"),
        "recovery_method": metadata.get("recovery_method"),
        "target_image_sha256": metadata.get("target_image_sha256"),
        "latent_path": metadata.get("latent_path"),
        "noise_prediction_path": metadata.get("noise_prediction_path"),
        "target_latent_path": metadata.get("target_latent_path"),
        "preview_image_path": metadata.get("preview_image_path"),
        "N": metadata.get("num_seeds"),
        "T": metadata.get("num_inference_steps"),
        "latent_shape": science_shape,
        "latent_dtype": dtypes.get("latent") if isinstance(dtypes, Mapping) else None,
        "prediction_dtype": dtypes.get("conditional_noise_predictions") if isinstance(dtypes, Mapping) else None,
        "model_id": metadata.get("model_id"),
        "model_revision": metadata.get("model_revision"),
        "vae_id": metadata.get("vae_id"),
        "vae_revision": metadata.get("vae_revision"),
        "scheduler": metadata.get("scheduler_name"),
        "guidance_scale": metadata.get("guidance_scale"),
        "completed_or_resumed": status,
    }


def _summary(
    root: Path,
    paths: GenerationPaths,
    dataset: Any,
    manifest: pd.DataFrame,
    skipped: list[dict[str, object]],
    failed: list[dict[str, object]],
    statuses: Mapping[str, str],
    run_config: Mapping[str, object] | None,
    disk_estimate: Any,
    started: datetime,
) -> dict[str, object]:
    counts = Counter(manifest.get("webster_overfit_type", pd.Series(dtype=str)).fillna("UNKNOWN"))
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": (datetime.now(UTC) - started).total_seconds(),
        "outcome": "completed" if not failed else "completed_with_failures",
        "run_directory": _relative(paths.run_directory, root),
        "model_manifest_rows": dataset.total_manifest_rows,
        "available_paired_image_rows": len(dataset),
        "selected_rows": len(dataset),
        "completed_rows": len(manifest),
        "newly_generated_rows": sum(value == "generated" for value in statuses.values()),
        "fully_resumed_rows": sum(value == "resumed" for value in statuses.values()),
        "preview_only_regenerated_rows": sum(value == "preview_regenerated" for value in statuses.values()),
        "skipped_rows": len(skipped),
        "failed_rows": len(failed),
        "counts_by_webster_label": dict(sorted(counts.items())),
        "total_trajectories": int(manifest.get("N", pd.Series(dtype=int)).fillna(0).sum()),
        "selection_policy": GENERATION_SELECTION_POLICY,
        "scientific_config_hash": None if run_config is None else run_config.get("scientific_config_hash"),
        "preview_config_hash": None if run_config is None else run_config.get("preview_config_hash"),
        "estimated_bytes_per_record": disk_estimate.bytes_per_record,
        "estimated_remaining_bytes": disk_estimate.estimated_remaining_bytes,
        "free_bytes_at_preflight": disk_estimate.free_bytes,
        "disk_safety_margin_bytes": disk_estimate.safety_margin_bytes,
    }


def _runtime_provenance(root: Path) -> dict[str, object]:
    source_paths = (
        root / "scripts" / "generate.py",
        root / "utils" / "experiments" / "generation.py",
        root / "utils" / "experiments" / "cache.py",
        root / "utils" / "models" / "sampling.py",
        root / "utils" / "models" / "latent.py",
    )
    versions: dict[str, str | None] = {}
    for package in ("torch", "diffusers", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "project_root": str(root),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "sampler_contract_version": SAMPLER_CONTRACT_VERSION,
        "source_sha256": {str(path.relative_to(root)): file_sha256(path) for path in source_paths},
        "package_versions": versions,
    }


def _validate_existing_invocation(
    configuration: Mapping[str, object], model: str, scheduler: str,
    guidance: float, steps: int, seed_values: Sequence[int],
) -> None:
    from utils.models.latent import TARGET_LATENT_DEFINITION

    science = configuration.get("scientific_config")
    if not isinstance(science, Mapping):
        raise GenerationError("scientific configuration is missing")
    spec = get_model_spec(model)
    scheduler_value = science.get("scheduler")
    observed_scheduler = scheduler_value.get("name") if isinstance(scheduler_value, Mapping) else None
    expected = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "sampler_contract_version": SAMPLER_CONTRACT_VERSION,
        "selection_policy": GENERATION_SELECTION_POLICY,
        "target_role": "canonical_paired_source",
        "model_cli_name": model,
        "dataset_model": spec.dataset_model,
        "model_id": spec.model_id,
        "vae_id": spec.vae_id or spec.model_id,
        "resolution": spec.resolution,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "num_seeds": len(seed_values),
        "seeds": list(seed_values),
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
    }
    wrong = [key for key, value in expected.items() if science.get(key) != value]
    if observed_scheduler != scheduler:
        wrong.append("scheduler")
    if science.get("native_prediction_type") not in {
        "epsilon", "v_prediction", "sample"
    }:
        wrong.append("native_prediction_type")
    if wrong:
        raise GenerationError(
            "existing generation scientific contract differs at: "
            + ", ".join(wrong)
        )


def _validated_arguments(
    scheduler: str, guidance: float, steps: int, seeds: int,
    seed_start: int, downscale: int, resolution: int,
) -> tuple[float, int, int, int, int]:
    if scheduler not in {"ddim", "ddpm"}:
        raise GenerationError("scheduler must be ddim or ddpm")
    guidance_value = float(guidance)
    if not math.isfinite(guidance_value):
        raise GenerationError("guidance must be finite")
    for name, value in (("steps", steps), ("seeds", seeds), ("downscale", downscale)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GenerationError(f"{name} must be a positive integer")
    try:
        seed_start, seeds = validate_seed_block(seed_start, seeds)
    except ValueError as error:
        raise GenerationError(str(error)) from error
    if resolution % downscale:
        raise GenerationError("resolution must be divisible by downscale")
    return guidance_value, steps, seeds, seed_start, downscale


def _preview_only_invalid(
    errors: Sequence[str], metadata: Mapping[str, object] | None
) -> bool:
    return metadata is not None and bool(errors) and all(
        error.startswith("preview image") for error in errors
    )


def _record_paths(paths: GenerationPaths, index: str) -> tuple[Path, ...]:
    return (
        paths.latent_path(index), paths.noise_prediction_path(index),
        paths.target_latent_path(index), paths.image_path(index),
        paths.record_path(index), paths.pending_record_path(index),
    )


def _skip_row(metadata: Mapping[str, object], reason: str) -> dict[str, object]:
    return {
        "original_index": metadata.get("original_index"),
        "record_id": metadata.get("record_id"),
        "source_row_number": metadata.get("source_row_number"),
        "webster_overfit_type": metadata.get("overfit_type"),
        "reason": reason,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())
