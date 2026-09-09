"""Tensor-free validation path for unconditional-baseline plot-only consumers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from utils.common.io import file_sha256

from .unconditional_baseline import (
    DEFAULT_NUM_BASELINE_SEEDS,
    UnconditionalBaselineArtifact,
    UnconditionalBaselineError,
    _SourceContract,
    _artifact_from_validated,
    _baseline_seed_values,
    _is_sha256,
    _load_reference_identity,
    _mu_hat_description,
    _normalized_scheduler_config,
    _read_artifact_metadata,
    _validate_metadata,
    unconditional_baseline_artifact_paths,
)


def load_unconditional_baseline_metadata_only(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    num_baseline_seeds: int = DEFAULT_NUM_BASELINE_SEEDS,
) -> UnconditionalBaselineArtifact:
    """Validate JSON and raw hashes without deserializing Torch payloads."""

    identity = _load_reference_identity(
        project_root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
    )
    baseline_seeds = _baseline_seed_values(num_seeds, num_baseline_seeds)
    tensor_path, metadata_path = unconditional_baseline_artifact_paths(
        identity.paths.run_directory,
        seed_start=baseline_seeds[0],
        num_baseline_seeds=len(baseline_seeds),
    )
    metadata = _read_artifact_metadata(metadata_path, tensor_path)

    scheduler = identity.science.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise UnconditionalBaselineError("generation scheduler metadata is invalid")
    scheduler_config = _normalized_scheduler_config(scheduler.get("config"))
    init_noise_sigma = _positive_finite_float(
        metadata.get("init_noise_sigma"), "scheduler initial-noise scale"
    )
    if not math.isclose(init_noise_sigma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise UnconditionalBaselineError(
            f"{identity.scheduler_name.upper()} init_noise_sigma must equal 1"
        )
    timestep = _nonnegative_integer(metadata.get("timestep"), "baseline timestep")
    alpha_t = _positive_finite_float(metadata.get("alpha_T"), "baseline alpha_T")
    sigma_t = _positive_finite_float(metadata.get("sigma_T"), "baseline sigma_T")
    if not math.isclose(alpha_t**2 + sigma_t**2, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise UnconditionalBaselineError(
            "baseline alpha_T and sigma_T violate the diffusion coefficient identity"
        )
    num_train_timesteps = _nonnegative_integer(
        scheduler_config.get("num_train_timesteps"),
        "scheduler num_train_timesteps",
    )
    if num_train_timesteps <= 0 or timestep >= num_train_timesteps:
        raise UnconditionalBaselineError(
            "baseline timestep is outside the training schedule"
        )
    contract = _SourceContract(
        identity=identity,
        schedule_payload=None,
        scheduler_config=scheduler_config,
        init_noise_sigma=init_noise_sigma,
        timestep=timestep,
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        baseline_seeds=baseline_seeds,
    )
    _validate_metadata(metadata, contract, metadata_path, tensor_path)
    description = _mu_hat_description(metadata)
    expected_hash = description.get("sha256")
    if not _is_sha256(expected_hash) or file_sha256(tensor_path) != expected_hash:
        raise UnconditionalBaselineError(
            "unconditional baseline mu_hat SHA-256 differs"
        )
    return _artifact_from_validated(
        contract,
        metadata,
        metadata_path,
        tensor_path,
        str(expected_hash),
        None,
    )


def _positive_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise UnconditionalBaselineError(f"{label} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise UnconditionalBaselineError(
            f"{label} must be positive and finite"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise UnconditionalBaselineError(f"{label} must be positive and finite")
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise UnconditionalBaselineError(f"{label} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise UnconditionalBaselineError(
            f"{label} must be a nonnegative integer"
        ) from error
    if result != value or result < 0:
        raise UnconditionalBaselineError(f"{label} must be a nonnegative integer")
    return result
