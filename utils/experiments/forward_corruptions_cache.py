"""Read-only generation/SSCD inputs for Forward Corruptions and Generated States."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch

from utils.common.cli import validate_seed_block

from utils.common.io import canonical_hash, file_sha256, read_json, safe_torch_load
from utils.experiments.cache import (
    GENERATION_SCHEMA_VERSION,
    GENERATION_SELECTION_POLICY,
    SAMPLER_CONTRACT_VERSION,
    CompletedGenerationRecord,
    GenerationPaths,
    generation_paths,
    require_generation_run,
    safe_index,
    validate_generation_record,
)
from utils.experiments.sscd import (
    SCORE_DEFINITION,
    SSCD_SCHEMA_VERSION,
    SSCD_SELECTION_POLICY,
    SSCDPaths,
    _read_cached_scores,
    sscd_configuration_hash,
)
from utils.metrics.sscd import (
    SSCD_CHECKPOINT_URL,
    SSCD_FEATURE_DIMENSION,
    SSCD_INPUT_SIZE,
    SSCD_MODEL_NAME,
    sscd_preprocessing_hash,
    sscd_preprocessing_policy,
)
from utils.models.latent import TARGET_LATENT_DEFINITION, target_preprocessing_policy
from utils.models.registry import get_model_spec
from utils.models.sampling import make_initial_noise
from utils.models.schedule_metadata import build_schedule_metadata
from utils.models.schedulers import build_scheduler_from_config

MODEL_NAME = "sdv1"
SCHEDULER_NAME = "ddim"
GUIDANCE_SCALE = 7.5
NUM_SEEDS = 20


class ForwardCorruptionsCacheError(ValueError):
    """A required published input is missing, unsafe, or scientifically incompatible."""


@dataclass(frozen=True, slots=True)
class CachedRun:
    paths: GenerationPaths
    configuration: Mapping[str, Any]
    science: Mapping[str, Any]
    seeds: tuple[int, ...]
    timesteps: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor
    schedule_payload: Mapping[str, Any]
    scheduler_final_alpha_cumprod: float
    sscd_paths: SSCDPaths
    sscd_configuration: Mapping[str, Any]

    @property
    def scientific_hash(self) -> str:
        return str(self.configuration["scientific_config_hash"])

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return tuple(self.science["latent_shape"])

    @property
    def stored_dtype(self) -> torch.dtype:
        return _storage_dtype(self.science)


def _expect(
    observed: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    if not isinstance(observed, Mapping):
        raise ForwardCorruptionsCacheError(f"{label} is not a mapping")
    wrong = [key for key, value in expected.items() if observed.get(key) != value]
    if wrong:
        raise ForwardCorruptionsCacheError(f"{label} differs at: {', '.join(wrong)}")


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ForwardCorruptionsCacheError(f"{label} is missing or unsafe: {path}")


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _storage_dtype(science: Mapping[str, Any]) -> torch.dtype:
    storage = science.get("scientific_tensor_storage")
    if not isinstance(storage, Mapping):
        raise ForwardCorruptionsCacheError("scientific tensor storage is missing")
    _expect(storage, {"device": "cpu", "layout": "contiguous"}, "tensor storage")
    allowed = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    name = storage.get("dtype")
    if (
        not isinstance(name, str)
        or name not in allowed
        or science.get("inference_dtype") != name
    ):
        raise ForwardCorruptionsCacheError("inference and stored tensor dtype differ")
    return allowed[name]


def _sscd_configuration(
    root: Path,
    paths: GenerationPaths,
    configuration: Mapping[str, Any],
    seeds: tuple[int, ...],
) -> tuple[SSCDPaths, Mapping[str, Any]]:
    score_paths = SSCDPaths(paths.run_directory)
    _regular_file(score_paths.config_json, "SSCD configuration")
    saved = read_json(score_paths.config_json)
    if saved.get("configuration_hash") != sscd_configuration_hash(saved):
        raise ForwardCorruptionsCacheError("SSCD configuration hash differs")
    science = configuration["scientific_config"]
    order = (
        "seed 0 through seed N - 1" if seeds[0] == 0 else "same order as explicit seeds"
    )
    expected = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "generation_run_path": paths.run_directory.relative_to(root).as_posix(),
        "generation_run_config_path": paths.run_config.relative_to(root).as_posix(),
        "generation_scientific_config_hash": configuration["scientific_config_hash"],
        "selection_policy": SSCD_SELECTION_POLICY,
        "num_seeds": NUM_SEEDS,
        "seeds": list(seeds),
        "terminal_latent_index": -1,
        "decode_dtype": "float32",
        "score_definition": SCORE_DEFINITION,
        "score_tensor_schema": {
            "shape": ["num_seeds"],
            "dtype": "float32",
            "device": "cpu",
            "contiguous": True,
            "order": order,
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
        **{
            key: science[key]
            for key in ("model_id", "model_revision", "vae_id", "vae_revision")
        },
    }
    _expect(saved, expected, "SSCD scientific contract")
    if not _sha256(saved.get("sscd_checkpoint_sha256")):
        raise ForwardCorruptionsCacheError("SSCD checkpoint hash is invalid")
    return score_paths, saved


def load_run(
    root: str | Path, *, num_inference_steps: int = 50, seed_start: int = 0
) -> CachedRun:
    """Read the fixed SD1.4/DDIM/g=7.5/20-seed cache; never create or repair inputs."""
    if (
        isinstance(num_inference_steps, bool)
        or not isinstance(num_inference_steps, int)
        or not 1 <= num_inference_steps <= 1000
    ):
        raise ForwardCorruptionsCacheError(
            "num_inference_steps must be an integer in [1, 1000]"
        )
    try:
        seed_start, _ = validate_seed_block(seed_start, NUM_SEEDS)
    except ValueError as error:
        raise ForwardCorruptionsCacheError(str(error)) from error
    project = Path(root).expanduser().resolve()
    seeds = tuple(range(seed_start, seed_start + NUM_SEEDS))
    paths = generation_paths(
        project,
        model_name=MODEL_NAME,
        scheduler_name=SCHEDULER_NAME,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=num_inference_steps,
        num_seeds=NUM_SEEDS,
        seed_start=seed_start,
    )
    configuration = require_generation_run(paths)
    science = configuration["scientific_config"]
    spec = get_model_spec(MODEL_NAME)
    _expect(
        science,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "sampler_contract_version": SAMPLER_CONTRACT_VERSION,
            "selection_policy": GENERATION_SELECTION_POLICY,
            "target_role": "canonical_paired_source",
            "model_cli_name": MODEL_NAME,
            "dataset_model": spec.dataset_model,
            "model_id": spec.model_id,
            "resolution": spec.resolution,
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": num_inference_steps,
            "num_seeds": NUM_SEEDS,
            "seeds": list(seeds),
            "latent_shape": [4, 64, 64],
            "native_prediction_type": "epsilon",
            "stored_prediction_type": "epsilon",
            "trajectory_order": "noise_to_image",
            "target_latent_definition": TARGET_LATENT_DEFINITION,
            "target_preprocessing": target_preprocessing_policy(spec.resolution),
        },
        "generation scientific contract",
    )
    for key in ("model_revision", "vae_id", "vae_revision"):
        if not isinstance(science.get(key), str) or not science[key]:
            raise ForwardCorruptionsCacheError(f"generation {key} is missing")
    _storage_dtype(science)
    scheduler_metadata = science.get("scheduler")
    if not isinstance(scheduler_metadata, Mapping):
        raise ForwardCorruptionsCacheError("scheduler metadata is missing")
    _expect(
        scheduler_metadata,
        {"name": SCHEDULER_NAME, "class": "DDIMScheduler"},
        "scheduler",
    )
    _regular_file(paths.schedule, "generation schedule")
    schedule_hash = file_sha256(paths.schedule)
    schedule = safe_torch_load(paths.schedule)
    if not isinstance(schedule, Mapping):
        raise ForwardCorruptionsCacheError("saved schedule is not a mapping")
    _expect(
        schedule,
        {
            "scheduler_name": SCHEDULER_NAME,
            "scheduler_class": "DDIMScheduler",
            "trajectory_order": "noise_to_image",
            "prediction_type": "epsilon",
            "native_prediction_type": "epsilon",
            "stored_prediction_type": "epsilon",
            "init_noise_sigma": 1.0,
        },
        "saved schedule",
    )
    if canonical_hash(schedule.get("scheduler_config")) != canonical_hash(
        scheduler_metadata.get("config")
    ):
        raise ForwardCorruptionsCacheError(
            "saved and configured scheduler settings differ"
        )
    scheduler = build_scheduler_from_config(
        scheduler_metadata["config"], SCHEDULER_NAME
    ).scheduler
    rebuilt = build_schedule_metadata(
        scheduler, SCHEDULER_NAME, num_inference_steps=num_inference_steps, device="cpu"
    )
    for key in ("timesteps", "alpha_t", "sigma_t", "alphas_cumprod_t"):
        value = schedule.get(key)
        expected = rebuilt[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != expected.dtype
            or value.device.type != "cpu"
            or not value.is_contiguous()
            or value.requires_grad
            or not torch.equal(value, expected)
        ):
            raise ForwardCorruptionsCacheError(
                f"saved schedule {key} differs from the native scheduler"
            )
    if not bool((schedule["timesteps"][:-1] > schedule["timesteps"][1:]).all()):
        raise ForwardCorruptionsCacheError(
            "saved timesteps must be strictly descending"
        )
    if not bool((schedule["alpha_t"] > 0).all() and (schedule["sigma_t"] > 0).all()):
        raise ForwardCorruptionsCacheError(
            "evaluated schedule coefficients must be positive"
        )
    final_alpha = float(torch.as_tensor(scheduler.final_alpha_cumprod).item())
    if not math.isfinite(final_alpha) or not 0 < final_alpha <= 1:
        raise ForwardCorruptionsCacheError("scheduler final alpha_cumprod is invalid")
    if file_sha256(paths.schedule) != schedule_hash:
        raise ForwardCorruptionsCacheError("generation schedule changed while loading")
    score_paths, score_configuration = _sscd_configuration(
        project, paths, configuration, seeds
    )
    # This is the actual scheduler boundary. An ideal forward endpoint (1, 0)
    # must be logged separately; set_alpha_to_one=False does not use that boundary.
    return CachedRun(
        paths,
        configuration,
        science,
        seeds,
        schedule["timesteps"].clone(),
        schedule["alpha_t"].clone(),
        schedule["sigma_t"].clone(),
        schedule,
        final_alpha,
        score_paths,
        score_configuration,
    )


def validate_run_pair(reference: CachedRun, evaluation: CachedRun) -> None:
    """Require independent reference seeds with otherwise identical scientific inputs."""
    for run in (reference, evaluation):
        if len(run.seeds) != NUM_SEEDS or run.seeds != tuple(
            range(run.seeds[0], run.seeds[0] + NUM_SEEDS)
        ):
            raise ForwardCorruptionsCacheError(
                "reference/evaluation runs must each use one contiguous 20-seed block"
            )
    if set(reference.seeds).intersection(evaluation.seeds):
        raise ForwardCorruptionsCacheError("reference and evaluation seeds overlap")
    for key in (
        "model_cli_name",
        "dataset_model",
        "model_id",
        "model_revision",
        "vae_id",
        "vae_revision",
        "resolution",
        "scheduler",
        "guidance_scale",
        "num_inference_steps",
        "num_seeds",
        "latent_shape",
        "inference_dtype",
        "scientific_tensor_storage",
        "native_prediction_type",
        "stored_prediction_type",
        "target_latent_definition",
        "target_preprocessing",
    ):
        if canonical_hash(reference.science.get(key)) != canonical_hash(
            evaluation.science.get(key)
        ):
            raise ForwardCorruptionsCacheError(f"reference and evaluation {key} differ")
    for key in ("timesteps", "alpha", "sigma"):
        if not torch.equal(getattr(reference, key), getattr(evaluation, key)):
            raise ForwardCorruptionsCacheError(f"reference and evaluation {key} differ")
    if (
        reference.scheduler_final_alpha_cumprod
        != evaluation.scheduler_final_alpha_cumprod
    ):
        raise ForwardCorruptionsCacheError(
            "reference and evaluation final scheduler coefficients differ"
        )
    for key in (
        "sscd_checkpoint_sha256",
        "sscd_preprocessing_hash",
        "score_definition",
    ):
        if reference.sscd_configuration.get(key) != evaluation.sscd_configuration.get(
            key
        ):
            raise ForwardCorruptionsCacheError(f"reference and evaluation {key} differ")


def _record_fields(run: CachedRun, metadata: Mapping[str, Any]) -> None:
    expected = {
        key: run.science[key]
        for key in (
            "model_cli_name",
            "dataset_model",
            "model_id",
            "model_revision",
            "vae_id",
            "vae_revision",
            "guidance_scale",
            "num_inference_steps",
            "num_seeds",
            "seeds",
            "latent_shape",
            "inference_dtype",
            "native_prediction_type",
            "stored_prediction_type",
            "trajectory_order",
            "target_latent_definition",
            "target_preprocessing",
        )
    }
    expected.update(scheduler_name=SCHEDULER_NAME, scheduler_class="DDIMScheduler")
    _expect(metadata, expected, "generation record")
    if not _sha256(metadata.get("target_image_sha256")):
        raise ForwardCorruptionsCacheError("record target image hash is invalid")
    if (
        not isinstance(metadata.get("record_id"), str)
        or not metadata["record_id"]
        or not isinstance(metadata.get("prompt_raw"), str)
    ):
        raise ForwardCorruptionsCacheError("record prompt-target identity is invalid")


def _validated_record(
    run: CachedRun,
    record: CompletedGenerationRecord,
    *,
    tensor_names: Sequence[str],
    verify_file_hashes: bool,
) -> Mapping[str, Any]:
    identity = {
        key: record.metadata.get(key)
        for key in (
            "record_id",
            "source_row_number",
            "prompt_raw",
            "target_image_sha256",
        )
    }
    validation = validate_generation_record(
        run.paths,
        record.original_index,
        expected_scientific_hash=run.scientific_hash,
        expected_record_identity=identity,
        load_tensors=False,
        tensor_names=tensor_names,
        require_preview=False,
        verify_file_hashes=verify_file_hashes,
    )
    if not validation.valid or validation.metadata is None:
        raise ForwardCorruptionsCacheError(
            f"invalid generation record {record.original_index}: {'; '.join(validation.errors)}"
        )
    _record_fields(run, validation.metadata)
    return validation.metadata


def load_scores(run: CachedRun, record: CompletedGenerationRecord) -> torch.Tensor:
    """Read exactly one published target-specific score per generation seed."""
    if (
        record.marker_path.resolve()
        != run.paths.record_path(record.original_index).resolve()
    ):
        raise ForwardCorruptionsCacheError(
            "score record belongs to another generation run"
        )
    marker = _validated_record(run, record, tensor_names=(), verify_file_hashes=False)
    if canonical_hash(marker) != canonical_hash(record.metadata):
        raise ForwardCorruptionsCacheError(
            "generation record changed before SSCD loading"
        )
    current = CompletedGenerationRecord(
        record.original_index, record.source_row_number, record.marker_path, marker
    )
    scores = _read_cached_scores(
        run.sscd_paths,
        current,
        str(run.sscd_configuration["configuration_hash"]),
        run.seeds,
    )
    if scores is None:
        raise ForwardCorruptionsCacheError(
            f"published SSCD scores are missing: {record.original_index}"
        )
    score_marker = read_json(run.sscd_paths.marker_path(record.original_index))
    _expect(
        score_marker,
        {
            "prompt_raw": marker["prompt_raw"],
            "model_id": run.science["model_id"],
            "model_revision": run.science["model_revision"],
            "vae_id": run.science["vae_id"],
            "vae_revision": run.science["vae_revision"],
            "sscd_checkpoint_sha256": run.sscd_configuration["sscd_checkpoint_sha256"],
        },
        "SSCD record",
    )
    return scores


def _tensor(
    value: object, *, shape: tuple[int, ...], dtype: torch.dtype, label: str
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
        raise ForwardCorruptionsCacheError(
            f"cached {label} tensor violates its contract"
        )
    return value


def load_states(
    run: CachedRun, reference_record: CompletedGenerationRecord
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Read selected evaluation states and the same fixed target, never predictions."""
    index = safe_index(reference_record.original_index)
    paths = (run.paths.latent_path(index), run.paths.target_latent_path(index))
    for path in paths:
        _regular_file(path, "generation tensor")
    stats = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    marker = _validated_record(
        run,
        reference_record,
        tensor_names=("latent", "target_latent"),
        verify_file_hashes=True,
    )
    hashes = marker.get("tensor_file_sha256")
    reference_hashes = reference_record.metadata.get("tensor_file_sha256")
    if (
        not isinstance(hashes, Mapping)
        or not isinstance(reference_hashes, Mapping)
        or hashes.get("target_latent") != reference_hashes.get("target_latent")
    ):
        raise ForwardCorruptionsCacheError(
            "paired reference/evaluation target latent hashes differ"
        )
    shape = run.latent_shape
    states = _tensor(
        safe_torch_load(paths[0]),
        shape=(NUM_SEEDS, len(run.timesteps) + 1, *shape),
        dtype=run.stored_dtype,
        label="generated states",
    )
    target = _tensor(
        safe_torch_load(paths[1]),
        shape=shape,
        dtype=torch.float32,
        label="paired target",
    )
    _expect(
        marker.get("tensor_shapes", {}),
        {"latent": list(states.shape), "target_latent": list(shape)},
        "record tensor shapes",
    )
    _expect(
        marker.get("tensor_dtypes", {}),
        {
            "latent": str(states.dtype).removeprefix("torch."),
            "target_latent": "float32",
        },
        "record tensor dtypes",
    )
    initial = make_initial_noise(run.seeds, shape).to(dtype=run.stored_dtype)
    if not torch.equal(states[:, 0], initial):
        raise ForwardCorruptionsCacheError(
            "cached initial states differ from the exact seeded Gaussian draws"
        )
    if stats != [
        (path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    ] or canonical_hash(read_json(run.paths.record_path(index))) != canonical_hash(
        marker
    ):
        raise ForwardCorruptionsCacheError("generation inputs changed while loading")
    return states, target, dict(marker)
