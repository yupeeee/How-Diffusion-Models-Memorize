"""Fixed, held-out reference-initial centers without additional inference."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import torch

from utils.common.io import read_json, file_sha256
from utils.models.devices import (
    configure_worker_cpu_threads,
    round_robin_shard,
    worker_count_for_tasks,
)
from .cache_reader import load_record
from .contracts import TheoryError


def clean_initial(z, epsilon, alpha, sigma):
    if not math.isfinite(alpha) or alpha <= 0:
        raise TheoryError(
            "Reference initial alpha is zero/invalid; center is unidentified"
        )
    return (z.double() - sigma * epsilon.double()) / alpha


def compare_initial(z, epsilon, canonical_z, canonical_epsilon):
    """Compare repeated seed observations before counting them only once.

    Initial RNG states must be identical. Empty outputs permit 8 storage eps
    relative to their largest absolute coordinate, allowing order-dependent
    quantization in batched inference. This is a screening tolerance, not a
    rigorous neural-network forward-error guarantee.
    """
    if not torch.equal(z, canonical_z):
        raise TheoryError("Repeated initial states differ for the same actual seed")
    eps = max(torch.finfo(epsilon.dtype).eps, torch.finfo(canonical_epsilon.dtype).eps)
    scale = max(1.0, float(canonical_epsilon.double().abs().max()))
    tolerance = 8 * eps * scale
    discrepancy = float((epsilon.double() - canonical_epsilon.double()).abs().max())
    if discrepancy > tolerance:
        raise TheoryError(
            f"Repeated initial unconditional outputs inconsistent: {discrepancy} > {tolerance}"
        )
    return discrepancy, tolerance


def _reference_initial_shard(
    paths,
    records,
    canonical_z,
    canonical_u,
    *,
    verify_hashes,
    worker_count,
    progress=None,
):
    """Check disjoint records against the coordinator's single initial anchor.

    Full trajectories stay memory mapped on CPU; workers return only scalar
    summaries and never fit a new center or write shared source metadata.
    """
    configure_worker_cpu_threads(worker_count)
    max_discrepancy, max_tolerance = 0.0, 0.0
    for position, record in enumerate(records, 1):
        other_z, other_u, _, _ = load_record(
            paths, record, initial_only=True, verify_hashes=verify_hashes
        )
        discrepancy, tolerance = compare_initial(
            other_z, other_u, canonical_z, canonical_u
        )
        max_discrepancy = max(max_discrepancy, discrepancy)
        max_tolerance = max(max_tolerance, tolerance)
        if progress and (position % 25 == 0 or position == len(records)):
            progress(position)
    return len(records), max_discrepancy, max_tolerance


def _reference_initial_subprocess(
    paths, records, anchor_z, anchor_u, z_dtype, u_dtype, **kwargs
):
    # Byte arrays preserve every storage dtype, including bfloat16, without
    # PyTorch's file-descriptor/shared-memory tensor transport.
    return _reference_initial_shard(
        paths,
        records,
        torch.from_numpy(anchor_z).view(z_dtype),
        torch.from_numpy(anchor_u).view(u_dtype),
        **kwargs,
    )


def _reference_consistency(paths, records, z, u, *, devices, verify_hashes, progress):
    worker_count = worker_count_for_tasks(devices, len(records))
    if worker_count <= 1:
        return _reference_initial_shard(
            paths,
            records,
            z,
            u,
            verify_hashes=verify_hashes,
            worker_count=1,
            progress=(
                lambda position: progress(
                    f"reference initial consistency {position + 1}/{len(records) + 1}"
                )
            )
            if progress
            else None,
        )
    shards = tuple(
        round_robin_shard(records, worker_index=index, worker_count=worker_count)
        for index in range(worker_count)
    )
    if progress:
        progress(
            f"reference initial consistency: {worker_count} CPU reader workers, "
            f"{len(records) + 1} records; one shared canonical initial anchor"
        )
    anchor_z = z.contiguous().view(torch.uint8).numpy()
    anchor_u = u.contiguous().view(torch.uint8).numpy()
    # Match generation: one local worker and one spawned process for every
    # other active device. Hash and initial-state checks need no CUDA context.
    with ProcessPoolExecutor(
        max_workers=worker_count - 1, mp_context=get_context("spawn")
    ) as executor:
        futures = [
            executor.submit(
                _reference_initial_subprocess,
                paths,
                shard,
                anchor_z,
                anchor_u,
                z.dtype,
                u.dtype,
                verify_hashes=verify_hashes,
                worker_count=worker_count,
            )
            for shard in shards[1:]
        ]
        local = _reference_initial_shard(
            paths,
            shards[0],
            z,
            u,
            verify_hashes=verify_hashes,
            worker_count=worker_count,
            progress=(
                lambda position: progress(
                    f"reference initial consistency local shard {position}/{len(shards[0])}"
                )
            )
            if progress
            else None,
        )
        results = [local, *(future.result() for future in futures)]
    count = sum(result[0] for result in results)
    if count != len(records):
        raise TheoryError("Reference initial consistency coverage differs")
    if progress:
        progress(f"reference initial consistency {count + 1}/{len(records) + 1}")
    return (
        count,
        max(result[1] for result in results),
        max(result[2] for result in results),
    )


def reference_center(
    sources, schedule, *, verify_hashes=True, progress=None, devices=None
):
    records = sorted(
        sources.records["reference"],
        key=lambda r: (r.source_row_number, r.original_index),
    )
    cumulative = float(schedule["alphas_cumprod_t"][0])
    alpha, sigma = math.sqrt(cumulative), math.sqrt(1 - cumulative)
    canonical = records[0]
    z, u, _, _ = load_record(
        sources.reference, canonical, initial_only=True, verify_hashes=verify_hashes
    )
    estimates = clean_initial(z, u, alpha, sigma)
    center = estimates.mean(dim=0)
    _, max_discrepancy, max_tolerance = _reference_consistency(
        sources.reference,
        records[1:],
        z,
        u,
        devices=(torch.device("cpu"),) if devices is None else tuple(devices),
        verify_hashes=verify_hashes,
        progress=progress,
    )
    n = estimates.shape[0]
    variance_mean = float(estimates.var(dim=0, unbiased=True).mean()) if n > 1 else None
    metadata = {
        "definition": "mean of canonical complete record initial empty-branch clean estimates, one per reference seed",
        "evidence": "finite_noise_observation",
        "center": "reference-initial",
        "known_training_distribution_mean": False,
        "canonical_original_index": canonical.original_index,
        "canonical_record_id": canonical.metadata["record_id"],
        "canonical_selection_rule": "minimum (source_row_number, original_index) over all complete reference records; no outcome filtering",
        "seeds": canonical.metadata["seeds"],
        "source_sample_count": n,
        "verified_reference_record_count": len(records),
        "initial_alpha": alpha,
        "initial_sigma": sigma,
        "initial_snr": cumulative / (1 - cumulative) if cumulative < 1 else None,
        "initial_snr_status": "valid" if cumulative < 1 else "undefined_zero_sigma",
        "center_norm_rmse": float(center.square().mean().sqrt()),
        "reference_initial_dispersion_rmse": float(
            (estimates - center).square().mean().sqrt()
        ),
        "mc_standard_error_rmse": math.sqrt(variance_mean / n)
        if variance_mean is not None
        else None,
        "mc_uncertainty_status": "seed_sample_variance"
        if n > 1
        else "unavailable_single_reference_seed",
        "mc_uncertainty_definition": "sqrt(mean coordinate unbiased sample variance / number of unique reference seeds); not a confidence radius",
        "reference_evaluation_relation": "disjoint seeds; reference seeds also inform the frozen selection",
        "repeated_epsilon_max_abs_difference": max_discrepancy,
        "repeated_epsilon_max_abs_tolerance": max_tolerance,
        "consistency_policy": "identical initial states; epsilon max-absolute difference <= 8 * storage_eps * max(1,maxabs canonical epsilon); screening only",
        "source_dtype": str(u.dtype),
    }
    return center, metadata


def choose_center(sources, schedule, **kwargs):
    reference, metadata = reference_center(sources, schedule, **kwargs)
    mode = sources.config["center"]
    if mode == "reference-initial":
        return reference, metadata
    if mode == "zero":
        metadata = {
            "center": "zero",
            "definition": "fixed zero sensitivity diagnostic; no claim of a zero population mean",
            "evidence": "finite_noise_observation",
            "reference_initial": metadata,
            "source_sample_count": 0,
            "center_norm_rmse": 0.0,
            "_reference_center": reference,
        }
        return torch.zeros_like(reference), metadata
    from utils.experiments.unconditional_baseline import load_unconditional_baseline

    args = {
        key: sources.config[key]
        for key in (
            "model_name",
            "scheduler_name",
            "guidance_scale",
            "num_inference_steps",
            "num_seeds",
        )
    }
    explicit = sources.config.get("cached_baseline")
    if explicit:
        path = Path(explicit)
        meta_path = (
            path if path.name == "metadata.json" else path.parent / "metadata.json"
        )
        baseline_metadata = read_json(meta_path)
        args["num_baseline_seeds"] = baseline_metadata["num_baseline_seeds"]
    artifact = load_unconditional_baseline(sources.root, **args)
    if explicit and artifact.metadata_path.resolve() != meta_path.resolve():
        raise TheoryError(
            "Explicit cached baseline is outside the validated baseline namespace"
        )
    sources.metadata_files[
        artifact.metadata_path.relative_to(sources.root).as_posix()
    ] = file_sha256(artifact.metadata_path)
    return artifact.mu_hat.double(), {
        "center": "cached-baseline",
        "definition": "existing independent Gaussian unconditional baseline; optional external inference source",
        "evidence": "finite_noise_observation",
        "reference_initial": metadata,
        "source_sample_count": artifact.metadata["num_baseline_seeds"],
        "source_metadata": dict(artifact.metadata),
        "source_tensor_sha256": artifact.mu_hat_sha256,
        "_reference_center": reference,
    }
