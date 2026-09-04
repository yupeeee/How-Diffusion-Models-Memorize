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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
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
    CacheIOError,
    atomic_torch_save,
    atomic_write_frame_csv,
    atomic_write_frame_parquet,
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    file_sha256,
    safe_torch_load,
)
from utils.models.devices import (
    DeviceSelectionError,
    configure_worker_cpu_threads,
    resolve_devices,
    worker_count_for_tasks,
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
    "original_index",
    "record_id",
    "source_row_number",
    "prompt_raw",
    "webster_overfit_type",
    "recovery_status",
    "recovery_method",
    "target_image_sha256",
    "latent_path",
    "noise_prediction_path",
    "target_latent_path",
    "preview_image_path",
    "N",
    "T",
    "latent_shape",
    "latent_dtype",
    "prediction_dtype",
    "model_id",
    "model_revision",
    "vae_id",
    "vae_revision",
    "scheduler",
    "guidance_scale",
    "completed_or_resumed",
)


class GenerationError(RuntimeError):
    """A trajectory generation run cannot proceed safely."""


def _install_tqdm_lock(lock: Any) -> None:
    """Install the one inter-process lock shared by all progress bars."""

    tqdm.set_lock(lock)


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
    source_provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _GenerationShardResult:
    manifest_rows: tuple[dict[str, object], ...]
    skipped_rows: tuple[dict[str, object], ...]
    failed_rows: tuple[dict[str, object], ...]
    status_by_index: dict[str, str]
    run_config: dict[str, object] | None


def generate_webster_trajectories(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
    downscale: int,
    device: str | torch.device,
) -> GenerationResult:
    """Generate every available pair, sharding whole pairs across devices."""

    started = datetime.now(UTC)
    root = Path(project_root).expanduser().resolve()
    try:
        seed_start, num_seeds = validate_seed_block(seed_start, num_seeds)
    except ValueError as error:
        raise GenerationError(str(error)) from error
    spec = get_model_spec(model_name)
    guidance, steps, seed_count, first_seed, divisor = _validated_arguments(
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
        downscale,
        spec.resolution,
    )
    try:
        devices = resolve_devices(device)
    except DeviceSelectionError as error:
        raise GenerationError(str(error)) from error
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
    config_present = paths.run_config.exists() or paths.run_config.is_symlink()
    if not config_present:
        _reject_orphaned_scientific_artifacts(paths)
    paths.create()
    dataset = _load_dataset(root, spec.dataset_model)
    metadata_rows = list(dataset.iter_metadata())
    entries = tuple(enumerate(metadata_rows))
    existing_config = require_generation_run(paths) if config_present else None
    if existing_config is not None:
        _validate_existing_invocation(
            paths,
            existing_config,
            model_name,
            scheduler_name,
            guidance,
            steps,
            seed_values,
        )
        existing_config = _update_preview_configuration(
            paths,
            existing_config,
            resolution=spec.resolution,
            downscale=divisor,
        )
    missing_markers = sum(
        not paths.record_path(row["original_index"]).is_file() for row in metadata_rows
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

    run_config = existing_config
    parent_runtime: _Runtime | None = None
    if run_config is None and len(devices) > 1:
        run_config, parent_runtime, devices = (
            _initialize_generation_configuration_with_failover(
                root=root,
                paths=paths,
                dataset=dataset,
                entries=entries,
                spec=spec,
                scheduler_name=scheduler_name,
                guidance=guidance,
                steps=steps,
                seed_values=seed_values,
                downscale=divisor,
                devices=devices,
            )
        )
        # With no runnable pair there is no GPU work and no shared contract for
        # child processes to consume. The serial path retains normal failures
        # and skipped-row reporting without loading a model.
        if run_config is None:
            devices = devices[:1]

    requested_workers = worker_count_for_tasks(devices, len(entries))
    shards = (
        ((),)
        if requested_workers == 0
        else _target_group_shards(entries, requested_workers)
    )
    worker_count = len(shards)
    active_devices = devices[:worker_count]
    shard_sizes = ", ".join(
        f"{selected}={len(shard)} records"
        for selected, shard in zip(active_devices, shards, strict=True)
    )
    print(
        f"Generation device shards: {shard_sizes}. "
        "Progress ETAs are local: Records covers that device's shard; "
        "Denoising covers only the current record."
    )
    if worker_count > 1:
        print(
            "Generation uses concurrent model replicas; prompt-target pairs are "
            "disjoint and all seeds for one pair stay together."
        )
        context = get_context("spawn")
        progress_lock = context.RLock()
        tqdm.set_lock(progress_lock)
        with ProcessPoolExecutor(
            max_workers=worker_count - 1,
            mp_context=context,
            initializer=_install_tqdm_lock,
            initargs=(progress_lock,),
        ) as executor:
            futures = [
                executor.submit(
                    _generation_subprocess,
                    root,
                    model_name,
                    scheduler_name,
                    guidance,
                    steps,
                    seed_values,
                    divisor,
                    shards[worker_index],
                    dict(run_config),
                    str(active_devices[worker_index]),
                    worker_index,
                    worker_count,
                )
                for worker_index in range(1, worker_count)
            ]
            local, _ = _run_generation_shard(
                root=root,
                paths=paths,
                dataset=dataset,
                spec=spec,
                scheduler_name=scheduler_name,
                guidance=guidance,
                steps=steps,
                seed_values=seed_values,
                downscale=divisor,
                entries=shards[0],
                run_config=run_config,
                device=active_devices[0],
                worker_index=0,
                worker_count=worker_count,
                runtime=parent_runtime,
            )
            shard_results = [local, *(future.result() for future in futures)]
    else:
        local, _ = _run_generation_shard(
            root=root,
            paths=paths,
            dataset=dataset,
            spec=spec,
            scheduler_name=scheduler_name,
            guidance=guidance,
            steps=steps,
            seed_values=seed_values,
            downscale=divisor,
            entries=shards[0],
            run_config=run_config,
            device=active_devices[0],
            worker_index=0,
            worker_count=1,
            runtime=parent_runtime,
        )
        shard_results = [local]

    device_shards = [
        {
            "device": str(selected),
            "assigned_rows": len(shard),
            "completed_rows": len(result.manifest_rows),
            "newly_generated_rows": sum(
                status == "generated" for status in result.status_by_index.values()
            ),
            "fully_resumed_rows": sum(
                status == "resumed" for status in result.status_by_index.values()
            ),
            "preview_only_regenerated_rows": sum(
                status == "preview_regenerated"
                for status in result.status_by_index.values()
            ),
            "skipped_rows": len(result.skipped_rows),
            "failed_rows": len(result.failed_rows),
        }
        for selected, shard, result in zip(
            active_devices, shards, shard_results, strict=True
        )
    ]
    manifest_rows = [row for result in shard_results for row in result.manifest_rows]
    skipped_rows = [row for result in shard_results for row in result.skipped_rows]
    failed_rows = [row for result in shard_results for row in result.failed_rows]
    status_by_index: dict[str, str] = {}
    for result in shard_results:
        overlap = status_by_index.keys() & result.status_by_index.keys()
        if overlap:
            raise GenerationError(
                "prompt-target pair was assigned to multiple devices: "
                + ", ".join(sorted(overlap))
            )
        status_by_index.update(result.status_by_index)
        if run_config is None and result.run_config is not None:
            run_config = result.run_config

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values(
        ["source_row_number", "original_index"], kind="stable"
    ).reset_index(drop=True)
    skipped_rows.sort(
        key=lambda row: (row.get("source_row_number"), str(row.get("original_index")))
    )
    failed_rows.sort(
        key=lambda row: (row.get("source_row_number"), str(row.get("original_index")))
    )
    skipped_columns = (
        "original_index",
        "record_id",
        "source_row_number",
        "webster_overfit_type",
        "reason",
    )
    failed_columns = (
        "original_index",
        "record_id",
        "source_row_number",
        "webster_overfit_type",
        "exception_type",
        "exception_message",
        "traceback_path",
    )
    atomic_write_frame_parquet(manifest, paths.manifest_parquet)
    atomic_write_frame_csv(manifest, paths.manifest_csv)
    atomic_write_frame_csv(
        pd.DataFrame(skipped_rows, columns=skipped_columns), paths.skipped_csv
    )
    atomic_write_frame_csv(
        pd.DataFrame(failed_rows, columns=failed_columns), paths.failed_csv
    )
    summary = _summary(
        root,
        paths,
        dataset,
        manifest,
        skipped_rows,
        failed_rows,
        status_by_index,
        run_config,
        disk_estimate,
        started,
    )
    summary.update(
        {
            "requested_device": str(device),
            "execution_devices": [str(selected) for selected in active_devices],
            "worker_count": worker_count,
            "device_shards": device_shards,
        }
    )
    atomic_write_json(paths.summary_json, summary)
    return GenerationResult(paths.summary_json, len(manifest), len(failed_rows))


def _target_group_shards(
    entries: Sequence[tuple[int, Mapping[str, object]]],
    requested_workers: int,
) -> tuple[tuple[tuple[int, Mapping[str, object]], ...], ...]:
    """Balance records while keeping identical target images on one device."""

    if requested_workers < 1:
        raise ValueError("requested worker count must be positive")
    groups: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
    for position, metadata in entries:
        digest = metadata.get("target_image_sha256")
        key = (
            str(digest)
            if isinstance(digest, str) and digest
            else f"unshared:{position}:{metadata.get('original_index')}"
        )
        groups.setdefault(key, []).append((position, metadata))
    worker_count = min(requested_workers, len(groups))
    if worker_count == 0:
        return ((),)
    shards: list[list[tuple[int, Mapping[str, object]]]] = [
        [] for _ in range(worker_count)
    ]
    for group in groups.values():
        worker_index = min(
            range(worker_count),
            key=lambda index: (len(shards[index]), index),
        )
        shards[worker_index].extend(group)
    return tuple(tuple(sorted(shard, key=lambda entry: entry[0])) for shard in shards)


def _reject_orphaned_scientific_artifacts(paths: GenerationPaths) -> None:
    """Reject unpublished scientific data instead of adopting it implicitly."""

    if paths.schedule.exists() or paths.schedule.is_symlink():
        raise GenerationError(
            "generation cache contains scientific artifacts without "
            f"run_config.json: {paths.schedule}"
        )
    directories = (
        paths.latent_directory,
        paths.noise_prediction_directory,
        paths.target_latent_directory,
        paths.record_directory,
    )
    for directory in directories:
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise GenerationError(
                "generation cache contains an unsafe scientific-artifact "
                f"directory without run_config.json: {directory}"
            )
        if directory.is_dir():
            artifact = next(directory.iterdir(), None)
            if artifact is not None:
                raise GenerationError(
                    "generation cache contains scientific artifacts without "
                    f"run_config.json: {artifact}"
                )


def _initialize_generation_configuration_with_failover(
    *,
    root: Path,
    paths: GenerationPaths,
    dataset: Any,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    spec: Any,
    scheduler_name: str,
    guidance: float,
    steps: int,
    seed_values: Sequence[int],
    downscale: int,
    devices: Sequence[torch.device],
) -> tuple[
    dict[str, object] | None,
    _Runtime | None,
    tuple[torch.device, ...],
]:
    """Publish a fresh shared contract on the first working device."""

    candidates = tuple(devices)
    if not candidates:
        raise GenerationError("generation initialization requires a device")
    for position, selected in enumerate(candidates):
        try:
            configuration, runtime = _initialize_generation_configuration(
                root=root,
                paths=paths,
                dataset=dataset,
                entries=entries,
                spec=spec,
                scheduler_name=scheduler_name,
                guidance=guidance,
                steps=steps,
                seed_values=seed_values,
                downscale=downscale,
                device=selected,
            )
        except Exception as error:
            tqdm.write(
                f"[Generation] Initialization failed on {selected}; "
                f"trying the next device: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            continue
        if configuration is None:
            return None, None, candidates[:1]
        if runtime is None:
            raise GenerationError(
                "generation configuration was created without its runtime"
            )
        return configuration, runtime, candidates[position:]

    tqdm.write(
        "[Generation] No device could initialize the shared model contract; "
        "recording failures in one coordinator shard.",
        file=sys.stderr,
    )
    return None, None, candidates[:1]


def _initialize_generation_configuration(
    *,
    root: Path,
    paths: GenerationPaths,
    dataset: Any,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    spec: Any,
    scheduler_name: str,
    guidance: float,
    steps: int,
    seed_values: Sequence[int],
    downscale: int,
    device: torch.device,
) -> tuple[dict[str, object] | None, _Runtime | None]:
    """Create the one shared run contract before child processes start."""

    for position, metadata in entries:
        index = str(metadata["original_index"])
        try:
            validation = validate_generation_record(
                paths,
                index,
                expected_record_identity=metadata,
                load_tensors=False,
            )
            if validation.valid:
                continue
            item = dataset[position]
            image = item.get("image")
            prompt = item.get("prompt")
            if not isinstance(image, Image.Image) or not isinstance(prompt, str):
                continue
        except Exception:
            # The normal record loop records this failure. Configuration setup
            # only needs to find one runnable pair without changing artifacts.
            continue
        runtime = _load_runtime(root, spec, scheduler_name, steps, device=device)
        configuration = _create_run_config(
            root,
            paths,
            dataset,
            spec,
            runtime,
            scheduler_name,
            guidance,
            steps,
            seed_values,
            downscale,
        )
        return configuration, runtime
    return None, None


def _generation_subprocess(
    root: Path,
    model_name: str,
    scheduler_name: str,
    guidance: float,
    steps: int,
    seed_values: Sequence[int],
    downscale: int,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    run_config: Mapping[str, object],
    device: str,
    worker_index: int,
    worker_count: int,
) -> _GenerationShardResult:
    """Load one model replica and process one disjoint device shard."""

    spec = get_model_spec(model_name)
    paths = generation_paths(
        root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=len(seed_values),
        seed_start=int(seed_values[0]),
    )
    dataset = _load_dataset(root, spec.dataset_model, True)
    result, _ = _run_generation_shard(
        root=root,
        paths=paths,
        dataset=dataset,
        spec=spec,
        scheduler_name=scheduler_name,
        guidance=guidance,
        steps=steps,
        seed_values=seed_values,
        downscale=downscale,
        entries=entries,
        run_config=run_config,
        device=torch.device(device),
        worker_index=worker_index,
        worker_count=worker_count,
    )
    return result


def _run_generation_shard(
    *,
    root: Path,
    paths: GenerationPaths,
    dataset: Any,
    spec: Any,
    scheduler_name: str,
    guidance: float,
    steps: int,
    seed_values: Sequence[int],
    downscale: int,
    entries: Sequence[tuple[int, Mapping[str, object]]],
    run_config: Mapping[str, object] | None,
    device: torch.device,
    worker_index: int,
    worker_count: int,
    runtime: _Runtime | None = None,
) -> tuple[_GenerationShardResult, _Runtime | None]:
    """Process one stable pair shard; write no run-level aggregate files."""

    configure_worker_cpu_threads(worker_count)
    seed_count = len(seed_values)
    manifest_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    status_by_index: dict[str, str] = {}
    target_cache: dict[str, torch.Tensor] = {}
    local_config = None if run_config is None else dict(run_config)
    runtime_failure: Exception | None = None

    def load_runtime_once() -> _Runtime:
        nonlocal runtime, runtime_failure
        if runtime is not None:
            return runtime
        if runtime_failure is not None:
            raise GenerationError(
                f"model runtime is unavailable on {device}: {runtime_failure}"
            ) from runtime_failure
        try:
            runtime = _load_runtime(
                root,
                spec,
                scheduler_name,
                steps,
                device=device,
                run_config=local_config,
            )
        except Exception as error:
            runtime_failure = error
            raise
        return runtime

    description = (
        "[Generation] Records (shard-local ETA)"
        if worker_count == 1
        else f"[Generation] Records {device} shard {worker_index + 1}/{worker_count} (local ETA)"
    )
    record_progress = tqdm(
        entries,
        total=len(entries),
        desc=description,
        unit="record",
        dynamic_ncols=True,
        leave=True,
        disable=False,
        position=worker_index * 2,
    )
    for position, metadata in record_progress:
        index = str(metadata["original_index"])
        record_progress.set_postfix_str(f"id={index} phase=validating")
        try:
            science_hash = (
                str(local_config["scientific_config_hash"])
                if local_config is not None
                else None
            )
            validation = validate_generation_record(
                paths,
                index,
                expected_scientific_hash=science_hash,
                expected_record_identity=metadata,
                load_tensors=False,
            )
            preview_matches = _preview_matches_downscale(validation.metadata, downscale)
            if validation.valid and preview_matches:
                status_by_index[index] = "resumed"
                manifest_rows.append(
                    _manifest_row(root, paths, validation.metadata or {}, "resumed")
                )
                continue
            if validation.valid or _preview_only_invalid(
                validation.errors, validation.metadata
            ):
                if local_config is None:
                    raise GenerationError("preview recovery has no run configuration")
                if runtime is None:
                    record_progress.set_postfix_str(f"id={index} phase=loading-model")
                    runtime = load_runtime_once()
                record_progress.set_postfix_str(f"id={index} phase=preview-recovery")
                with tqdm(
                    total=seed_count,
                    desc=f"[Generation] Preview {index}",
                    unit="image",
                    dynamic_ncols=True,
                    leave=False,
                    disable=False,
                    position=worker_index * 2 + 1,
                ) as preview_progress:
                    _regenerate_preview(
                        paths,
                        index,
                        validation.metadata or {},
                        runtime,
                        spec.resolution,
                        downscale,
                        progress_callback=preview_progress.update,
                    )
                repaired = validate_generation_record(
                    paths,
                    index,
                    expected_scientific_hash=str(
                        local_config["scientific_config_hash"]
                    ),
                    expected_record_identity=metadata,
                    load_tensors=False,
                )
                if not repaired.valid or not _preview_matches_downscale(
                    repaired.metadata, downscale
                ):
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
                skipped_rows.append(
                    _skip_row(metadata, "prompt or target image unavailable")
                )
                continue
            if runtime is None:
                record_progress.set_postfix_str(f"id={index} phase=loading-model")
                runtime = load_runtime_once()
            if local_config is None:
                local_config = _create_run_config(
                    root,
                    paths,
                    dataset,
                    spec,
                    runtime,
                    scheduler_name,
                    guidance,
                    steps,
                    seed_values,
                    downscale,
                )
            target_hash = str(metadata.get("target_image_sha256") or "")
            target_latent = target_cache.get(target_hash)
            if target_latent is None:
                target_latent = _encode_target(
                    image,
                    runtime.components.vae,
                    runtime.components.device,
                    spec.resolution,
                )
                target_cache[target_hash] = target_latent
            record_progress.set_postfix_str(f"id={index} phase=denoising")
            with tqdm(
                total=seed_count * steps,
                desc=(f"[Generation] Denoising {index} (current-record local ETA)"),
                unit="seed-step",
                dynamic_ncols=True,
                leave=False,
                disable=False,
                position=worker_index * 2 + 1,
            ) as denoising_progress:
                batch = _sample(
                    prompt,
                    runtime,
                    scheduler_name,
                    guidance,
                    steps,
                    seed_values,
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
                position=worker_index * 2 + 1,
            ) as preview_progress:
                preview_hash = _write_preview(
                    batch.latents[:, -1],
                    runtime.components.vae,
                    runtime.components.device,
                    paths.image_path(index),
                    spec.resolution,
                    downscale,
                    progress_callback=preview_progress.update,
                )
            record = _record_metadata(
                root,
                paths,
                metadata,
                local_config,
                runtime,
                scheduler_name,
                guidance,
                steps,
                seed_values,
                downscale,
                tensor_hashes,
                preview_hash,
            )
            publish_completion_marker(paths, index, record)
            record_progress.set_postfix_str(f"id={index} phase=validating-output")
            completed = validate_generation_record(
                paths,
                index,
                expected_scientific_hash=str(local_config["scientific_config_hash"]),
                expected_record_identity=metadata,
                load_tensors=False,
                verify_file_hashes=False,
            )
            if not completed.valid:
                raise GenerationError(
                    "newly generated record did not validate: "
                    + "; ".join(completed.errors)
                )
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
                f"{type(error).__name__}: {error}; device={device}; traceback: "
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

    return (
        _GenerationShardResult(
            tuple(manifest_rows),
            tuple(skipped_rows),
            tuple(failed_rows),
            status_by_index,
            local_config,
        ),
        runtime,
    )


def _load_dataset(
    root: Path,
    dataset_model: str,
    defer_image_validation: bool = False,
) -> Any:
    from utils.data.webster import WebsterDataset

    return WebsterDataset(
        root,
        dataset_model,
        recovered_only=True,
        defer_image_validation=defer_image_validation,
    )


def _load_runtime(
    root: Path,
    spec: Any,
    scheduler_name: str,
    steps: int,
    *,
    device: str | torch.device,
    run_config: Mapping[str, object] | None = None,
) -> _Runtime:
    source_provenance = _runtime_provenance(root)
    from utils.models.loading import (
        compile_loaded_unet,
        load_model_components,
        preflight_model_components,
        select_runtime,
    )
    from utils.models.sampling import latent_shape_from_unet
    from utils.models.schedulers import build_scheduler

    concrete_device = torch.device(device)
    if concrete_device.type == "cuda":
        if concrete_device.index is None:
            raise GenerationError("worker CUDA device must have a concrete index")
        torch.cuda.set_device(concrete_device)
    selected = select_runtime(concrete_device)
    resolver = _revision_resolver(run_config) if run_config is not None else None
    components = load_model_components(
        spec, runtime=selected, revision_resolver=resolver
    )
    scheduler_build = build_scheduler(components.original_scheduler, scheduler_name)
    preflight_model_components(
        components,
        scheduler=scheduler_build.scheduler,
        num_inference_steps=steps,
    )
    shape = latent_shape_from_unet(components.unet)
    runtime = _Runtime(
        components,
        scheduler_build.scheduler,
        {
            "name": scheduler_build.name,
            "class": scheduler_build.class_name,
            "config": scheduler_build.config,
            "removed_config_keys": list(scheduler_build.removed_config_keys),
        },
        shape,
        source_provenance,
    )
    if run_config is not None:
        _validate_runtime_contract(runtime, run_config)
    runtime.components = compile_loaded_unet(runtime.components)
    return runtime


def _revision_resolver(
    run_config: Mapping[str, object],
) -> Callable[[str], str]:
    science = run_config.get("scientific_config")
    if not isinstance(science, Mapping):
        raise GenerationError("generation scientific configuration is missing")
    revisions = {
        str(science.get("model_id")): science.get("model_revision"),
        str(science.get("vae_id")): science.get("vae_revision"),
    }

    def resolve(repository_id: str) -> str:
        revision = revisions.get(repository_id)
        if not isinstance(revision, str):
            raise GenerationError(
                f"generation configuration has no revision for {repository_id}"
            )
        return revision

    return resolve


def _validate_runtime_contract(
    runtime: _Runtime, run_config: Mapping[str, object]
) -> None:
    science = run_config.get("scientific_config")
    if not isinstance(science, Mapping):
        raise GenerationError("generation scientific configuration is missing")
    components = runtime.components
    observed = {
        "model_revision": components.model_revision,
        "vae_revision": components.vae_revision,
        "latent_shape": list(runtime.latent_shape),
        "inference_dtype": str(components.inference_dtype).removeprefix("torch."),
    }
    wrong = [key for key, value in observed.items() if science.get(key) != value]
    if canonical_hash(science.get("scheduler")) != canonical_hash(
        dict(runtime.scheduler_description)
    ):
        wrong.append("scheduler")
    if wrong:
        raise GenerationError(
            "loaded worker runtime differs from the generation contract at: "
            + ", ".join(wrong)
        )


def _sample(
    prompt: str,
    runtime: _Runtime,
    scheduler_name: str,
    guidance: float,
    steps: int,
    seed_values: Sequence[int],
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
        montage.paste(
            tile, ((position % columns) * tile_size, (position // columns) * tile_size)
        )
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
        trajectory[:, -1],
        runtime.components.vae,
        runtime.components.device,
        paths.image_path(index),
        resolution,
        downscale,
        progress_callback=progress_callback,
    )
    updated = dict(metadata)
    updated["preview_image_sha256"] = digest
    updated["preview_downscale"] = downscale
    updated["preview_status"] = "complete"
    updated["preview_completed_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(paths.record_path(index), updated)


def _preview_configuration(resolution: int, downscale: int) -> dict[str, object]:
    return {
        "downscale": downscale,
        "resolution": resolution,
        "tile_size": [resolution // downscale] * 2,
        "samples_per_row": 10,
        "sample_order": "seed_ascending",
        "resize_resampling": "Pillow Lanczos",
        "format": "PNG",
        "scientific_metric_input": False,
    }


def _update_preview_configuration(
    paths: GenerationPaths,
    configuration: Mapping[str, object],
    *,
    resolution: int,
    downscale: int,
) -> dict[str, object]:
    desired = _preview_configuration(resolution, downscale)
    desired_hash = canonical_hash(desired)
    updated = dict(configuration)
    if (
        updated.get("preview_config") == desired
        and updated.get("preview_config_hash") == desired_hash
    ):
        return updated
    updated["preview_config"] = desired
    updated["preview_config_hash"] = desired_hash
    atomic_write_json(paths.run_config, updated)
    return updated


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
    from utils.models.latent import (
        TARGET_LATENT_DEFINITION,
        target_preprocessing_policy,
    )
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
        "package_versions": dict(components.package_versions),
        "trajectory_order": "noise_to_image",
        "latent_index_description": "latents[:, 0] is x_T; latents[:, T] is terminal x_0",
        "prediction_index_description": "branches[:, i] are epsilon at latents[:, i]",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
        "target_preprocessing": target_preprocessing_policy(spec.resolution),
        "scientific_tensor_storage": {
            "dtype": str(components.inference_dtype).removeprefix("torch."),
            "device": "cpu",
            "layout": "contiguous",
        },
    }
    preview = _preview_configuration(spec.resolution, downscale)
    configuration: dict[str, object] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "scientific_config": science,
        "scientific_config_hash": canonical_hash(science),
        "preview_config": preview,
        "preview_config_hash": canonical_hash(preview),
        "runtime_provenance": {
            **runtime.source_provenance,
            "device_metadata": dict(components.device_metadata),
        },
    }
    schedule = build_schedule_metadata(
        runtime.scheduler,
        scheduler_name,
        num_inference_steps=steps,
        device=components.device,
    )
    atomic_torch_save(schedule, paths.schedule)
    atomic_write_json(paths.run_config, configuration)
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
    from utils.models.latent import (
        TARGET_LATENT_DEFINITION,
        target_preprocessing_policy,
    )
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
            "unconditional_noise_predictions": [
                seed_count,
                steps,
                *runtime.latent_shape,
            ],
            "conditional_noise_predictions": [seed_count, steps, *runtime.latent_shape],
            "target_latent": list(runtime.latent_shape),
        },
        "tensor_dtypes": {
            "latent": str(components.inference_dtype).removeprefix("torch."),
            "unconditional_noise_predictions": str(
                components.inference_dtype
            ).removeprefix("torch."),
            "conditional_noise_predictions": str(
                components.inference_dtype
            ).removeprefix("torch."),
            "target_latent": "float32",
        },
        "runtime_provenance": {
            **runtime.source_provenance,
            "device_metadata": dict(components.device_metadata),
        },
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
        "prediction_dtype": dtypes.get("conditional_noise_predictions")
        if isinstance(dtypes, Mapping)
        else None,
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
    counts = Counter(
        manifest.get("webster_overfit_type", pd.Series(dtype=str)).fillna("UNKNOWN")
    )
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
        "newly_generated_rows": sum(
            value == "generated" for value in statuses.values()
        ),
        "fully_resumed_rows": sum(value == "resumed" for value in statuses.values()),
        "preview_only_regenerated_rows": sum(
            value == "preview_regenerated" for value in statuses.values()
        ),
        "skipped_rows": len(skipped),
        "failed_rows": len(failed),
        "counts_by_webster_label": dict(sorted(counts.items())),
        "total_trajectories": int(
            manifest.get("N", pd.Series(dtype=int)).fillna(0).sum()
        ),
        "selection_policy": GENERATION_SELECTION_POLICY,
        "scientific_config_hash": None
        if run_config is None
        else run_config.get("scientific_config_hash"),
        "preview_config_hash": None
        if run_config is None
        else run_config.get("preview_config_hash"),
        "estimated_bytes_per_record": disk_estimate.bytes_per_record,
        "estimated_remaining_bytes": disk_estimate.estimated_remaining_bytes,
        "free_bytes_at_preflight": disk_estimate.free_bytes,
        "disk_safety_margin_bytes": disk_estimate.safety_margin_bytes,
    }


def _runtime_provenance(root: Path) -> dict[str, object]:
    source_paths = (
        root / "scripts" / "generate.py",
        root / "utils" / "common" / "io.py",
        root / "utils" / "experiments" / "generation.py",
        root / "utils" / "experiments" / "cache.py",
        root / "utils" / "models" / "devices.py",
        root / "utils" / "models" / "loading.py",
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
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path) for path in source_paths
        },
        "package_versions": versions,
    }


def _validate_existing_invocation(
    paths: GenerationPaths,
    configuration: Mapping[str, object],
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    seed_values: Sequence[int],
) -> None:
    from utils.models.latent import TARGET_LATENT_DEFINITION

    science = configuration.get("scientific_config")
    if not isinstance(science, Mapping):
        raise GenerationError("scientific configuration is missing")
    spec = get_model_spec(model)
    scheduler_value = science.get("scheduler")
    observed_scheduler = (
        scheduler_value.get("name") if isinstance(scheduler_value, Mapping) else None
    )
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
        "epsilon",
        "v_prediction",
        "sample",
    }:
        wrong.append("native_prediction_type")
    if wrong:
        raise GenerationError(
            "existing generation scientific contract differs at: " + ", ".join(wrong)
        )
    _validate_saved_schedule(paths, science, scheduler, steps)


def _validate_saved_schedule(
    paths: GenerationPaths,
    science: Mapping[str, object],
    scheduler_name: str,
    steps: int,
) -> None:
    """Validate the published CPU schedule against the run contract."""

    if not paths.schedule.is_file() or paths.schedule.is_symlink():
        raise GenerationError("saved generation schedule is missing or unsafe")
    try:
        payload = safe_torch_load(paths.schedule)
    except CacheIOError as error:
        raise GenerationError(
            f"saved generation schedule cannot be read: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise GenerationError("saved generation schedule must be a mapping")
    scheduler = science.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise GenerationError("generation scheduler contract is missing")
    if (
        payload.get("scheduler_name") != scheduler_name
        or not isinstance(scheduler.get("class"), str)
        or payload.get("scheduler_class") != scheduler.get("class")
        or not isinstance(scheduler.get("config"), Mapping)
        or not isinstance(payload.get("scheduler_config"), Mapping)
        or canonical_hash(payload.get("scheduler_config"))
        != canonical_hash(scheduler.get("config"))
    ):
        raise GenerationError("saved generation schedule differs from its contract")
    expected_metadata = {
        "native_prediction_type": science.get("native_prediction_type"),
        "stored_prediction_type": science.get("stored_prediction_type"),
        "trajectory_order": science.get("trajectory_order"),
    }
    if any(payload.get(key) != value for key, value in expected_metadata.items()):
        raise GenerationError("saved generation schedule prediction metadata differs")

    tensors: dict[str, torch.Tensor] = {}
    for name in ("timesteps", "alpha_t", "sigma_t", "alphas_cumprod_t"):
        value = payload.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or value.numel() != steps
            or value.device.type != "cpu"
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise GenerationError(f"saved generation schedule {name} is invalid")
        tensors[name] = value
    if tensors["timesteps"].dtype != torch.int64 or bool(
        (tensors["timesteps"] < 0).any()
    ):
        raise GenerationError("saved generation timesteps are invalid")
    for name in ("alpha_t", "sigma_t", "alphas_cumprod_t"):
        if tensors[name].dtype != torch.float32:
            raise GenerationError(f"saved generation schedule {name} dtype differs")
    alpha_squared = tensors["alpha_t"].square()
    sigma_squared = tensors["sigma_t"].square()
    cumulative = tensors["alphas_cumprod_t"]
    if not (
        torch.allclose(alpha_squared, cumulative, rtol=1e-5, atol=1e-6)
        and torch.allclose(
            sigma_squared,
            1.0 - cumulative,
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        raise GenerationError("saved generation schedule coefficients are inconsistent")
    try:
        init_noise_sigma = float(payload.get("init_noise_sigma"))
    except (TypeError, ValueError) as error:
        raise GenerationError(
            "saved scheduler initial-noise scale is invalid"
        ) from error
    if not math.isfinite(init_noise_sigma) or init_noise_sigma <= 0:
        raise GenerationError("saved scheduler initial-noise scale is invalid")


def _validated_arguments(
    scheduler: str,
    guidance: float,
    steps: int,
    seeds: int,
    seed_start: int,
    downscale: int,
    resolution: int,
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


def _preview_matches_downscale(
    metadata: Mapping[str, object] | None, downscale: int
) -> bool:
    if metadata is None:
        return False
    observed = metadata.get("preview_downscale")
    return (
        isinstance(observed, int)
        and not isinstance(observed, bool)
        and observed == downscale
    )


def _preview_only_invalid(
    errors: Sequence[str], metadata: Mapping[str, object] | None
) -> bool:
    return (
        metadata is not None
        and bool(errors)
        and all(error.startswith("preview image") for error in errors)
    )


def _record_paths(paths: GenerationPaths, index: str) -> tuple[Path, ...]:
    return (
        paths.latent_path(index),
        paths.noise_prediction_path(index),
        paths.target_latent_path(index),
        paths.image_path(index),
        paths.record_path(index),
        paths.pending_record_path(index),
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
