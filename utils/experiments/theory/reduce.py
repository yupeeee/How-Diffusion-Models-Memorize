"""One shared, resumable pass from protected caches to immutable scalar tables."""

from __future__ import annotations

import dataclasses
import hashlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
import subprocess
import time
from importlib.metadata import version as package_version
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.common.io import (
    atomic_write_json,
    atomic_write_frame_parquet,
    atomic_write_frame_csv,
    atomic_torch_save,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.models.devices import (
    resolve_devices,
    round_robin_shard,
    worker_count_for_tasks,
    configure_worker_cpu_threads,
)
from utils.experiments.cache import (
    GenerationPaths,
    list_completed_records,
    require_generation_run,
)
from .cache_reader import discover_sources, load_schedule, load_record, load_scores
from .centers import choose_center, compare_initial, clean_initial
from .contracts import SCHEMA_VERSION, FORMULA_VERSION, TheoryError, analysis_parent
from .metrics import (
    branch_metrics,
    clean_estimates,
    initial_distribution_diagnostics,
    NUMERICAL_POLICY,
    candidate_terminal_bound,
)
from .progress import RecordProgress, install_progress_queue, report_worker_record
from .support import FiniteSupport
from .feedback import proposition5_feedback, unavailable_feedback, INTEGRATION_POLICY
from . import summaries
from .scheduler_adapter import SchedulerAdapter
from .statements import load_statement_registry, paper_unavailable_metrics


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _science_only(value):
    if isinstance(value, dict):
        return {
            k: _science_only(v)
            for k, v in value.items()
            if k
            not in {
                "completed_at",
                "created_at",
                "created_at_utc",
                "finished_at_utc",
                "duration_seconds",
                "preview_completed_at",
            }
        }
    if isinstance(value, list):
        return [_science_only(v) for v in value]
    return value


def _scalar_rows(metrics, count):
    """Transfer scalar columns together, avoiding one CUDA sync per metric."""
    rows = [dict() for _ in range(count)]
    columns, device_groups = {}, {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            flat = value.detach().reshape(-1)
            if flat.numel() == 1:
                flat = flat.expand(count)
            if flat.numel() != count:
                raise TheoryError(
                    f"Scalar metric {key} has unexpected shape {tuple(value.shape)}"
                )
            # Group by dtype as well, preserving integer/boolean scalar types.
            device_groups.setdefault((flat.device, flat.dtype), []).append((key, flat))
        elif isinstance(value, (list, np.ndarray)) and len(value) == count:
            columns[key] = list(value)
        else:
            columns[key] = [value] * count
    for group in device_groups.values():
        host = torch.stack([flat for _, flat in group]).cpu().tolist()
        columns.update(
            (key, values) for (key, _), values in zip(group, host, strict=True)
        )
    # Retain schema order regardless of packing/device layout.
    for key in metrics:
        for row, item in zip(rows, columns[key], strict=True):
            row[key] = _json_safe(item)
    return rows


TIME_BLOCK_SIZE = 8


def build_support(
    sources, *, candidate_chunk_size=256, query_chunk_size=16, device="cpu"
):
    """Freeze every valid target in the declared experiment source run before selection."""
    science = sources.runs["experiment"]["scientific_config"]
    keys = (
        "model_id",
        "model_revision",
        "vae_id",
        "vae_revision",
        "latent_shape",
        "target_preprocessing",
        "target_latent_definition",
    )
    candidates, identities, source_rows, missing = [], {}, [], []
    # The source bank is one explicitly declared run, before retained-prompt or
    # outcome filtering. Other model-compatible folders cannot silently enlarge it.
    directories = {sources.experiment.run_directory}
    for directory in sorted(directories):
        paths = GenerationPaths(directory)
        raw_config = read_json(paths.run_config)
        candidate_science = raw_config.get("scientific_config", {})
        if not all(candidate_science.get(k) == science.get(k) for k in keys):
            continue
        run = require_generation_run(paths)
        records = list_completed_records(paths)
        if not records:
            continue
        relative = directory.relative_to(sources.root).as_posix()
        sources.metadata_files[
            paths.run_config.relative_to(sources.root).as_posix()
        ] = file_sha256(paths.run_config)
        sources.marker_inventory[f"{relative}/record"] = sorted(
            r.marker_path.name for r in records
        )
        seen = set()
        for record in records:
            metadata = record.metadata
            if metadata.get("scientific_config_hash") != run["scientific_config_hash"]:
                raise TheoryError(
                    f"Support record science differs: {record.marker_path}"
                )
            source = paths.target_latent_path(record.original_index)
            expected_hash = metadata.get("tensor_file_sha256", {}).get("target_latent")
            if (
                not source.is_file()
                or source.is_symlink()
                or file_sha256(source) != expected_hash
            ):
                raise TheoryError(f"Missing/stale complete support target: {source}")
            target = safe_torch_load(source)
            if (
                not isinstance(target, torch.Tensor)
                or list(target.shape) != science["latent_shape"]
                or not bool(torch.isfinite(target).all())
            ):
                raise TheoryError(f"Support target shape/values invalid: {source}")
            key = f"{relative}:{record.original_index}"
            candidates.append((key, target))
            identities[key] = metadata["target_image_sha256"]
            seen.add(source.name)
            sources.metadata_files[
                record.marker_path.relative_to(sources.root).as_posix()
            ] = file_sha256(record.marker_path)
            source_rows.append(
                {
                    "candidate_id": key,
                    "run_id": run["scientific_config_hash"],
                    "original_index": record.original_index,
                    "record_id": metadata["record_id"],
                    "prompt_raw": metadata["prompt_raw"],
                    "target_image_sha256": metadata["target_image_sha256"],
                    "target_latent_sha256": expected_hash,
                    "target_l2": float(target.double().norm()),
                    "target_latent_dtype": str(target.dtype),
                    "latent_dimension": target.numel(),
                    "included": directory == sources.experiment.run_directory
                    and record.original_index in sources.selection.included_indices,
                    "source_marker_sha256": file_sha256(record.marker_path),
                    "source_metadata_json": json.dumps(metadata, sort_keys=True),
                    "source_scientific_config_json": json.dumps(
                        run["scientific_config"], sort_keys=True
                    ),
                }
            )
        missing.extend(
            {
                "path": str(path.relative_to(sources.root)),
                "reason": "target lacks complete generation record; excluded from support",
            }
            for path in paths.target_latent_directory.glob("*.pt")
            if path.name not in seen
        )
    support = FiniteSupport.from_candidates(
        candidates,
        identities=identities,
        candidate_chunk=candidate_chunk_size,
        query_chunk=query_chunk_size,
    )
    if str(device) != "cpu":
        support = FiniteSupport(
            support.atoms.to(device),
            support.atom_ids,
            support.aliases,
            candidate_chunk=candidate_chunk_size,
            query_chunk=query_chunk_size,
            weights=support.weights,
        )
    metadata = support.metadata()
    metadata["missing_candidates"] = missing
    metadata["bank_policy"] = (
        "all_valid_complete_targets_from_declared_experiment_run_before_frozen_prompt_selection; equal_mass_per_exact_distinct_latent_atom"
    )
    metadata["source_run"] = sources.experiment.run_directory.relative_to(
        sources.root
    ).as_posix()
    metadata["source_run_hash"] = sources.runs["experiment"]["scientific_config_hash"]
    metadata["source_schedule_sha256"] = file_sha256(sources.experiment.schedule)
    metadata["source_qualification"] = (
        "declared_candidate_distribution_K; actual_training_law_and_frequencies_unidentified"
    )
    candidate_mean = support.weights @ support.flat
    metadata["candidate_mean_kind"] = "mu_K_mean_of_declared_candidate_atoms"
    metadata["candidate_mean_l2"] = float(candidate_mean.norm())
    metadata["candidate_mean_rmse"] = float(candidate_mean.norm()) / math.sqrt(
        support.dimension
    )
    metadata["candidate_mean_vector_sha256"] = hashlib.sha256(
        candidate_mean.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    metadata["latent_contract"] = {key: science.get(key) for key in keys}
    metadata["membership_source_count"] = len(source_rows)
    return support, metadata, pd.DataFrame(source_rows)


def _validate_shard(frame, record, count, steps, run_id):
    required = {
        "run_id",
        "original_index",
        "record_id",
        "seed",
        "step_index",
        "is_initial",
        "is_final_update",
        "unconditional_target_error_rmse",
        "conditional_target_error_rmse",
        "candidate_variation_l2",
        "candidate_condition_margin_rmse",
        "candidate_log_probability_gain",
        "feedback_eligible",
        "target_id",
        "sscd",
        "branch_status",
        "paper_terminal_bound_l2",
    }
    if not required.issubset(frame.columns):
        raise TheoryError(
            f"Missing required scalar columns for {record.original_index}: {sorted(required - set(frame.columns))}"
        )
    expected = {(seed, step) for seed in range(count) for step in range(steps)}
    keys = list(zip(frame.seed, frame.step_index))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise TheoryError(
            f"Missing/duplicate row identities for {record.original_index}"
        )
    for key, value in {
        "original_index": record.original_index,
        "record_id": record.metadata["record_id"],
        "run_id": run_id,
    }.items():
        if not frame[key].eq(value).all():
            raise TheoryError(f"Shard {key} differs for {record.original_index}")
    if (
        not frame.is_initial.eq(frame.step_index.eq(0)).all()
        or not frame.is_final_update.eq(frame.step_index.eq(steps - 1)).all()
    ):
        raise TheoryError(
            f"Initial/final-update indexing differs: {record.original_index}"
        )


def _record_metrics(
    sources,
    record,
    schedule,
    adapter,
    support,
    center,
    evaluation_initial,
    *,
    device,
    query_chunk_size,
    defer_path_integration=False,
    loaded_record=None,
):
    z, epsilon_u, epsilon_c, target = (
        load_record(sources.experiment, record)
        if loaded_record is None
        else loaded_record
    )
    scores = load_scores(sources, record)
    initial_z, initial_u = z[:, 0], epsilon_u[:, 0]
    if evaluation_initial is None:
        evaluation_initial = (initial_z.clone(), initial_u.clone())
    discrepancy, tolerance = compare_initial(initial_z, initial_u, *evaluation_initial)
    count, steps = sources.config["num_seeds"], sources.config["num_inference_steps"]
    g = sources.config["guidance_scale"]
    target, center_device = target.to(device), center.to(device)
    target_id = f"{sources.experiment.run_directory.relative_to(sources.root).as_posix()}:{record.original_index}"
    rows = []
    run_id = sources.runs["experiment"]["scientific_config_hash"]
    for time_start in range(0, steps, TIME_BLOCK_SIZE):
        time_end = min(steps, time_start + TIME_BLOCK_SIZE)
        for start in range(0, count, query_chunk_size):
            end = min(count, start + query_chunk_size)
            # Preserve the recorded dtype until each arithmetic reduction promotes
            # to float64. No full-record float64 GPU trajectory is allocated.
            states = z[start:end, time_start : time_end + 1].to(device)
            u_block = epsilon_u[start:end, time_start:time_end].to(device)
            c_block = epsilon_c[start:end, time_start:time_end].to(device)
            for k in range(time_start, time_end):
                coefficients = adapter.coefficients(k)
                local_step = k - time_start
                state, next_state = states[:, local_step], states[:, local_step + 1]
                u, c = u_block[:, local_step], c_block[:, local_step]
                metrics = branch_metrics(
                    state,
                    u,
                    c,
                    target,
                    coefficients.alpha,
                    coefficients.sigma,
                    g,
                    center_device,
                )
                mu, mc, _, _ = clean_estimates(
                    state, u, c, coefficients.alpha, coefficients.sigma, g
                )
                current_reference = support.evaluate(
                    state,
                    target_id,
                    coefficients.alpha,
                    coefficients.sigma,
                    unconditional=mu,
                )
                metrics.update(current_reference)
                update_metrics, matched, displacement = adapter.matched_update(
                    state, next_state, u, c, g, k, target=target - center_device
                )
                metrics.update(update_metrics)
                if (
                    defer_path_integration
                    and k < steps - 1
                    and coefficients.affine
                    and coefficients.B > 0
                ):
                    feedback = unavailable_feedback(
                        end - start,
                        device=device,
                        reason="path_integration_deferred_for_endpoint_stage",
                        status="not_computed",
                    )
                elif k < steps - 1 and coefficients.affine and coefficients.B > 0:
                    feedback = proposition5_feedback(
                        support,
                        state,
                        mc,
                        mu,
                        matched,
                        next_state,
                        target_id,
                        current_alpha=coefficients.alpha,
                        current_sigma=coefficients.sigma,
                        destination_alpha=coefficients.destination_alpha,
                        destination_sigma=coefficients.destination_sigma,
                        guidance=g,
                        kappa=coefficients.B,
                        source_error_l2=metrics[
                            "clean_estimate_rounding_sensitivity_rmse"
                        ]
                        * math.sqrt(target.numel()),
                    )
                else:
                    reason = (
                        "terminal_transition_not_in_Proposition5"
                        if k == steps - 1
                        else "non_affine_or_nonpositive_kappa"
                    )
                    feedback = unavailable_feedback(
                        end - start,
                        device=device,
                        reason=reason,
                        status="not_applicable",
                    )
                metrics.update(feedback)
                # Current-reference measurements remain available even when the
                # matched nonterminal feedback comparison does not apply.
                metrics["candidate_unconditional_reference_error_l2"] = (
                    current_reference["candidate_unconditional_reference_error_l2"]
                )
                metrics["candidate_conditional_reference_error_l2"] = metrics[
                    "conditional_target_error_l2"
                ]
                terminal = None
                if k == steps - 1:
                    terminal = adapter.terminal_diagnostics(
                        state, next_state, u, c, g, k, target
                    )
                    metrics.update(terminal)
                    if defer_path_integration:
                        from .paper_measurements import terminal_affine_accounting

                        metrics.update(
                            terminal_affine_accounting(
                                adapter, state, next_state, u, c, target, g, step=k
                            )
                        )
                    metrics.update(
                        candidate_terminal_bound(
                            metrics["conditional_target_error_l2"],
                            current_reference[
                                "candidate_unconditional_reference_error_l2"
                            ],
                            current_reference["candidate_radius_l2"],
                            current_reference["target_log_complement"],
                            guidance=g,
                            dimension=target.numel(),
                            clean_error_l2=terminal["terminal_clean_error_l2"],
                            endpoint_error_l2=terminal["terminal_endpoint_error_l2"],
                            terminal_clean_update=terminal["terminal_clean_applicable"],
                        )
                    )
                metrics.update(
                    paper_unavailable_metrics(
                        g,
                        terminal_clean_update=terminal["terminal_clean_structural"]
                        if terminal is not None
                        else False,
                        branch_gap_l2=metrics["branch_gap_l2"],
                        conditional_target_error_l2=metrics["e_c_l2"],
                    )
                )
                metrics.update(adapter.endpoint_bound(metrics, k, g, target.numel()))
                metrics.update(
                    {
                        "run_id": run_id,
                        "original_index": record.original_index,
                        "record_id": record.metadata["record_id"],
                        "target_id": record.metadata["tensor_file_sha256"][
                            "target_latent"
                        ],
                        "target_latent_sha256": record.metadata["tensor_file_sha256"][
                            "target_latent"
                        ],
                        "target_image_sha256": record.metadata["target_image_sha256"],
                        "support_target_id": target_id,
                        "candidate_target_atom": support.aliases.get(target_id),
                        "step_index": k,
                        "destination_step_index": k + 1,
                        "destination_snr": coefficients.destination_alpha**2
                        / coefficients.destination_sigma**2
                        if coefficients.destination_sigma > 0
                        else None,
                        "timestep": int(schedule["timesteps"][k]),
                        "destination_timestep": coefficients.destination_timestep,
                        "is_initial": k == 0,
                        "is_final_update": k == steps - 1,
                        "seed_role": "experiment",
                        "included": True,
                        "include_prompt": True,
                        "alpha": coefficients.alpha,
                        "sigma": coefficients.sigma,
                        "initial_snr": float(schedule["alphas_cumprod_t"][0])
                        / (1 - float(schedule["alphas_cumprod_t"][0]))
                        if float(schedule["alphas_cumprod_t"][0]) < 1
                        else None,
                        "same_seed_sscd_evidence": "finite_noise_observation",
                        "initial_unconditional_consistency_max_abs": discrepancy,
                        "initial_unconditional_consistency_tolerance": tolerance,
                        "terminal_alpha_condition": terminal[
                            "terminal_clean_structural"
                        ]
                        if terminal is not None
                        else False,
                    }
                )
                batch_rows = _scalar_rows(metrics, end - start)
                for offset, row in enumerate(batch_rows, start):
                    row["seed"] = record.metadata["seeds"][offset]
                    row["sscd"] = float(scores[offset])
                    row["terminal_sscd"] = float(scores[offset])
                    row["target_group"] = support.aliases[target_id]
                rows.extend(batch_rows)
    frame = (
        pd.DataFrame(rows).sort_values(["seed", "step_index"]).reset_index(drop=True)
    )
    for column in ("e_u_rmse", "e_c_rmse", "joint_rmse", "branch_gap_rmse"):
        frame["suffix_max_" + column] = frame.groupby("seed")[column].transform(
            lambda s: s.iloc[::-1].cummax().iloc[::-1]
        )
    # The endpoint is a separate state, never a nonexistent branch evaluation.
    terminal = (z[:, -1].double() - target.cpu().double()).flatten(1).norm(dim=1)
    proximity = sources.proximity[
        sources.proximity.original_index.eq(record.original_index)
    ].set_index("seed")
    endpoint = frame[frame.is_final_update].copy()
    dimension = target.numel()
    endpoint["terminal_l2"] = terminal.tolist()
    endpoint["terminal_rmse"] = (terminal / math.sqrt(dimension)).tolist()
    residuals = [
        float(terminal[i]) - float(proximity.loc[seed, "l2_norm"])
        for i, seed in enumerate(record.metadata["seeds"])
    ]
    # Protected proximity computes raw L2 in float32; this diagnostic promotes
    # before subtraction and may therefore differ at float32 reduction precision.
    budgets = [
        32 * np.finfo(np.float32).eps * max(1.0, float(value)) for value in terminal
    ]
    endpoint["protected_proximity_consistency_residual_l2"] = residuals
    endpoint["protected_proximity_consistency_budget_l2"] = budgets
    endpoint["protected_proximity_status"] = [
        "consistent_float32_reduction" if abs(r) <= b else "inconsistent"
        for r, b in zip(residuals, budgets)
    ]
    if endpoint.protected_proximity_status.eq("inconsistent").any():
        raise TheoryError(
            f"Protected proximity raw-L2 mismatch: {record.original_index}"
        )
    if "operational_bound_rmse" in endpoint:
        endpoint["operational_bound_slack_rmse"] = (
            endpoint.operational_bound_rmse - endpoint.terminal_rmse
        )
    initial = frame[frame.is_initial].copy()
    initial_coeff = adapter.coefficients(0)
    for key, value in initial_distribution_diagnostics(
        target,
        initial_coeff.alpha,
        initial_coeff.sigma,
        schedule["init_noise_sigma"],
        gaussian_initialization_recorded=sources.runs["experiment"][
            "scientific_config"
        ].get("sampler_contract_version")
        == 2,
    ).items():
        initial[key] = value
    initial["initialization_contract_source"] = (
        "preserved_sampler_contract_version_2:seeded_CPU_float32_standard_Gaussian_times_recorded_init_noise_sigma"
    )
    return frame, initial, endpoint, evaluation_initial


def _summary(initial, trajectory, endpoint, center_metadata):
    """Descriptive statistics conditional on selected targets and fixed seed bank."""

    def rates(frame, column):
        if column not in frame:
            return {"denominator": 0, "unavailable_reason": "metric not applicable"}
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values)]
        return {
            "denominator": len(values),
            "positive": int((values > 0).sum()),
            "negative": int((values < 0).sum()),
            "zero": int((values == 0).sum()),
            "median": None if values.empty else float(values.median()),
            "quantile_10": None if values.empty else float(values.quantile(0.1)),
            "quantile_90": None if values.empty else float(values.quantile(0.9)),
        }

    within = []
    for index, values in initial.groupby("original_index", sort=True):
        x, y = values.e_c_rmse, values.sscd
        correlation = (
            float(x.corr(y, method="spearman"))
            if x.nunique() > 1 and y.nunique() > 1
            else None
        )
        within.append(
            {
                "original_index": index,
                "n": len(values),
                "initial_conditional_error_terminal_sscd_spearman": correlation,
            }
        )
    result = {
        "statistical_scope": "descriptive, conditional on frozen reference selection and fixed experiment seed bank; no IID timestep inference",
        "bootstrap": {
            "method": "none; descriptive quantiles are not confidence intervals",
            "seed": None,
        },
        "selected_prompt_count": int(initial.original_index.nunique()),
        "experiment_sample_count": len(initial),
        "trajectory_row_count": len(trajectory),
        "endpoint_count": len(endpoint),
        "effective_initial_unconditional_seed_count": int(initial.seed.nunique()),
        "center": center_metadata,
        "within_prompt_associations": within,
        "initial_recovery": rates(initial, "conditional_error_reduction_rmse"),
        "posterior_feedback": summaries.feedback_counts(trajectory),
        "posterior_condition_margin": rates(
            trajectory, "candidate_condition_margin_rmse"
        ),
        "posterior_log_probability_gain": rates(
            trajectory, "candidate_log_probability_gain"
        ),
        "posterior_sign_policy": "saved gain ordering retains strict signs even when endpoint probabilities saturate; condition coverage is numerically estimated with unresolved cases retained",
        "summary_policy": summaries.SUMMARY_POLICY,
        "operational_bound_slack": rates(endpoint, "operational_bound_slack_rmse"),
        "operational_bound_magnitude": rates(endpoint, "operational_bound_rmse"),
        "terminal_error_magnitude": rates(endpoint, "terminal_rmse"),
        "terminal_zero_error_ratio_counts": {
            key: int(endpoint[key].fillna(False).astype(bool).sum())
            for key in endpoint
            if "zero_error" in key and "undefined" in key
        },
        "terminal_terms": {
            key: summaries.quantiles(endpoint[key])
            for key in (
                "terminal_A_rmse",
                "terminal_B_rmse",
                "terminal_terms_cosine",
                "terminal_cross_term_per_dimension",
                "terminal_clean_error_rmse",
                "scheduler_rho_rmse",
                "candidate_terminal_bound_rmse",
            )
            if key in endpoint
        },
        "unavailable": {
            "population_forward_loss": "independent forward-corruption evaluations unavailable",
            "population_mean": "reference-initial center is not known data mean",
            "learned_target_posterior": "finite-support surrogate only",
            "paper_terminal_bound": "training-distribution radius, reference-approximation errors and true posterior unavailable",
        },
        "metric_status_counts": {
            key: trajectory[key].fillna("null").value_counts().to_dict()
            for key in trajectory
            if key.endswith("status")
        },
    }
    if "margin_min" in trajectory and "log_odds_gain" in trajectory:
        valid = trajectory[["margin_min", "log_odds_gain"]].dropna()
        result["posterior_regime_counts"] = {
            "denominator": len(valid),
            "positive_margin_positive_gain": int(
                ((valid.margin_min > 0) & (valid.log_odds_gain > 0)).sum()
            ),
            "nonpositive_margin_positive_gain": int(
                ((valid.margin_min <= 0) & (valid.log_odds_gain > 0)).sum()
            ),
        }
    return _json_safe(result)


def _resolve_theory_devices(requested):
    """Reuse generation's visible-device semantics without changing precision."""
    devices = resolve_devices(requested)
    if any(device.type == "mps" for device in devices):
        if isinstance(requested, str) and requested.strip().lower() == "auto":
            print(
                "[theory] MPS does not support the required float64 reductions; using CPU",
                flush=True,
            )
            return (torch.device("cpu"),)
        raise TheoryError(
            "MPS cannot preserve theory float64 arithmetic; use cpu or cuda"
        )
    return tuple(devices)


@dataclasses.dataclass
class _TheoryShardResult:
    worker_index: int
    device: str
    assigned_indices: tuple[str, ...]
    status_by_index: dict
    failed_rows: list
    pid: int
    cpu_threads: int
    duration_seconds: float = 0.0


def _record_shard_paths(bundle, record):
    return (
        bundle / "trajectory_metrics" / f"part-{record.original_index}.parquet",
        bundle / "initial_shards" / f"part-{record.original_index}.parquet",
        bundle / "endpoint_shards" / f"part-{record.original_index}.parquet",
    )


def _worker_sources(sources, records):
    """Send only this shard's scalar context through the process queue."""
    indices = [record.original_index for record in records]
    return dataclasses.replace(
        sources,
        records={"experiment": list(records), "reference": []},
        selection=None,
        proximity=sources.proximity[
            sources.proximity.original_index.isin(indices)
        ].copy(),
        runs={"experiment": sources.runs["experiment"]},
        sscd_config={"experiment": sources.sscd_config["experiment"]},
        metadata_files={},
        marker_inventory={},
    )


def _shard_failure(record, error):
    return {
        "original_index": record.original_index,
        "record_id": record.metadata["record_id"],
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }


def _run_theory_shard(
    *,
    sources,
    records,
    bundle,
    analysis_hash,
    device,
    worker_index,
    worker_count,
    recompute,
    candidate_chunk_size,
    query_chunk_size,
    auxiliary_hashes,
    progress=None,
    defer_path_integration=False,
    prepare_candidate_endpoints=False,
):
    """Reduce disjoint whole records; write no shared run-level aggregates.

    Inputs are compact scalar metadata and paths, never high-dimensional IPC
    payloads. Each worker loads one immutable support/center copy and uses its
    concrete device for all assigned records, following generation's convention.
    """
    started = time.perf_counter()
    threads = configure_worker_cpu_threads(worker_count)
    concrete = torch.device(device)
    result = _TheoryShardResult(
        worker_index,
        str(concrete),
        tuple(record.original_index for record in records),
        {},
        [],
        os.getpid(),
        threads,
    )
    # A worker report also survives interruption before its result reaches the
    # coordinator. Each worker owns a different file.
    report = bundle / "worker_reports" / f"part-{worker_index}.json"
    try:
        if concrete.type == "cuda":
            if concrete.index is None:
                raise TheoryError("worker CUDA device must have a concrete index")
            torch.cuda.set_device(concrete)
        payloads = {}
        for name in (
            "center.pt",
            "support.pt",
            "evaluation_initial.pt",
            "worker_schedule.pt",
        ):
            path = bundle / name
            if (
                not path.is_file()
                or path.is_symlink()
                or file_sha256(path) != auxiliary_hashes[name]
            ):
                raise TheoryError(f"Shared worker input changed: {path}")
            payloads[name] = safe_torch_load(path)
        center = payloads["center.pt"].to(concrete)
        support_meta = read_json(bundle / "support_metadata.json")
        support = FiniteSupport(
            payloads["support.pt"].to(concrete),
            support_meta["atom_ids"],
            support_meta["aliases"],
            candidate_chunk=candidate_chunk_size,
            query_chunk=query_chunk_size,
            weights=support_meta["weights"],
        )
        evaluation_initial = payloads["evaluation_initial.pt"]
        schedule = payloads["worker_schedule.pt"]
        version = (
            sources.runs["experiment"]["scientific_config"]
            .get("package_versions", {})
            .get("diffusers")
        )
        adapter = SchedulerAdapter(schedule, recorded_diffusers_version=version)
        del payloads
        for record in records:
            paths = _record_shard_paths(bundle, record)
            stamp = paths[0].with_suffix(".json")
            try:
                resumed = False
                if (
                    not recompute
                    and stamp.exists()
                    and all(path.exists() for path in paths)
                ):
                    checkpoint = read_json(stamp)
                    if checkpoint.get(
                        "analysis_hash"
                    ) == analysis_hash and checkpoint.get("files") == {
                        path.relative_to(bundle).as_posix(): file_sha256(path)
                        for path in paths
                    }:
                        frame, initial, endpoint = [
                            pd.read_parquet(path) for path in paths
                        ]
                        _validate_shard(
                            frame,
                            record,
                            sources.config["num_seeds"],
                            sources.config["num_inference_steps"],
                            sources.runs["experiment"]["scientific_config_hash"],
                        )
                        for table, step in (
                            (initial, 0),
                            (endpoint, sources.config["num_inference_steps"] - 1),
                        ):
                            if (
                                len(table) != sources.config["num_seeds"]
                                or table.duplicated(["seed"]).any()
                                or not table.step_index.eq(step).all()
                                or set(table.seed)
                                != set(range(sources.config["num_seeds"]))
                            ):
                                raise TheoryError(
                                    f"Initial/endpoint shard row identities differ: {record.original_index}"
                                )
                        from utils.experiments.cache import validate_generation_record

                        validation = validate_generation_record(
                            sources.experiment,
                            record.original_index,
                            load_tensors=False,
                            require_preview=False,
                        )
                        if not validation.valid:
                            raise TheoryError(
                                f"Stale resumed source {record.original_index}: {validation.errors}"
                            )
                        resumed = True
                if resumed and prepare_candidate_endpoints:
                    from .candidate_reduce import shared_endpoint_record_valid

                    resumed = shared_endpoint_record_valid(bundle, record)
                if not resumed:
                    loaded_record = (
                        load_record(sources.experiment, record)
                        if prepare_candidate_endpoints
                        else None
                    )
                    with torch.inference_mode():
                        frame, initial, endpoint, _ = _record_metrics(
                            sources,
                            record,
                            schedule,
                            adapter,
                            support,
                            center,
                            evaluation_initial,
                            device=concrete,
                            query_chunk_size=query_chunk_size,
                            defer_path_integration=defer_path_integration,
                            loaded_record=loaded_record,
                        )
                        if prepare_candidate_endpoints:
                            from .candidate_reduce import stage_shared_endpoint_record

                            stage_shared_endpoint_record(
                                sources,
                                record,
                                bundle,
                                support,
                                center,
                                schedule,
                                adapter,
                                device=concrete,
                                query_chunk_size=query_chunk_size,
                                bank_hash=support_meta["tensor_sha256"],
                                loaded_record=loaded_record,
                                base_frames=(frame, initial, endpoint),
                            )
                    del loaded_record
                    _validate_shard(
                        frame,
                        record,
                        sources.config["num_seeds"],
                        sources.config["num_inference_steps"],
                        sources.runs["experiment"]["scientific_config_hash"],
                    )
                    for data, path in zip(
                        (frame, initial, endpoint), paths, strict=True
                    ):
                        atomic_write_frame_parquet(data, path)
                    atomic_write_json(
                        stamp,
                        {
                            "analysis_hash": analysis_hash,
                            "files": {
                                path.relative_to(bundle).as_posix(): file_sha256(path)
                                for path in paths
                            },
                        },
                    )
                result.status_by_index[record.original_index] = (
                    "resumed" if resumed else "reduced"
                )
                del frame, initial, endpoint
            except Exception as error:
                result.status_by_index[record.original_index] = "failed"
                result.failed_rows.append(_shard_failure(record, error))
            result.duration_seconds = time.perf_counter() - started
            atomic_write_json(report, _json_safe(dataclasses.asdict(result)))
            if progress:
                progress(
                    record.original_index,
                    result.status_by_index[record.original_index],
                    str(concrete),
                )
    except Exception as error:
        # Preserve completed shards; unfinished assignments are explicit failures.
        for record in records:
            if record.original_index not in result.status_by_index:
                result.status_by_index[record.original_index] = "failed"
                result.failed_rows.append(_shard_failure(record, error))
                if progress:
                    progress(record.original_index, "failed", str(concrete))
    result.duration_seconds = time.perf_counter() - started
    atomic_write_json(report, _json_safe(dataclasses.asdict(result)))
    return result


def _theory_subprocess(**kwargs):
    """Top-level, spawn-picklable worker entry point, as in generation."""
    return _run_theory_shard(progress=report_worker_record, **kwargs)


def _dispatch_theory_shards(
    *,
    sources,
    records,
    bundle,
    analysis_hash,
    devices,
    recompute,
    candidate_chunk_size,
    query_chunk_size,
    auxiliary_hashes,
    defer_path_integration=False,
    prepare_candidate_endpoints=False,
):
    worker_count = worker_count_for_tasks(devices, len(records))
    if worker_count == 0:
        return []
    active_devices = devices[:worker_count]
    shards = [
        round_robin_shard(records, worker_index=index, worker_count=worker_count)
        for index in range(worker_count)
    ]
    arguments = [
        dict(
            sources=_worker_sources(sources, shard),
            records=shard,
            bundle=bundle,
            analysis_hash=analysis_hash,
            device=str(device),
            worker_index=index,
            worker_count=worker_count,
            recompute=recompute,
            candidate_chunk_size=candidate_chunk_size,
            query_chunk_size=query_chunk_size,
            auxiliary_hashes=auxiliary_hashes,
            defer_path_integration=defer_path_integration,
            prepare_candidate_endpoints=prepare_candidate_endpoints,
        )
        for index, (device, shard) in enumerate(
            zip(active_devices, shards, strict=True)
        )
    ]
    context = get_context("spawn")
    with RecordProgress(
        total=len(records), devices=worker_count, context=context
    ) as progress:
        if worker_count == 1:
            results = [_run_theory_shard(progress=progress.report, **arguments[0])]
        else:
            # Keep one GPU in the coordinator, as generation does. A lightweight
            # consumer thread updates the sole bar while this local shard runs.
            with ProcessPoolExecutor(
                max_workers=worker_count - 1,
                mp_context=context,
                initializer=install_progress_queue,
                initargs=(progress.queue,),
            ) as executor:
                futures = [
                    executor.submit(_theory_subprocess, **args)
                    for args in arguments[1:]
                ]
                results = [_run_theory_shard(progress=progress.report, **arguments[0])]
                for index, future in enumerate(futures, 1):
                    try:
                        results.append(future.result())
                    except Exception as error:
                        failures = [
                            _shard_failure(record, error) for record in shards[index]
                        ]
                        results.append(
                            _TheoryShardResult(
                                index,
                                str(active_devices[index]),
                                tuple(r.original_index for r in shards[index]),
                                {r.original_index: "failed" for r in shards[index]},
                                failures,
                                -1,
                                0,
                            )
                        )
        # Reconcile missed events (for example a crashed worker) against final
        # results after all child queue feeders have finished. Record IDs dedupe
        # these updates, so reduced/resumed/failed prompts each count once.
        for result in results:
            for original_index, status in result.status_by_index.items():
                progress.report(original_index, status, result.device)
    return results


def _run_theory(
    project_root,
    *,
    recompute=False,
    device="auto",
    candidate_chunk_size=256,
    query_chunk_size=16,
    smoke_record_limit=None,
    defer_path_integration=False,
    prepare_candidate_endpoints=False,
    **config,
):
    """Reduce one configuration; never generate, fit selection, decode or score.

    ``smoke_record_limit`` is an explicitly scoped audit API, not a CLI/default
    eligibility rule. It publishes a separate bundle and never the full-run index.
    """
    if not isinstance(prepare_candidate_endpoints, bool):
        raise TheoryError("prepare_candidate_endpoints must be a boolean")
    if prepare_candidate_endpoints and not defer_path_integration:
        raise TheoryError(
            "Shared candidate endpoints require defer_path_integration=True"
        )
    if not isinstance(defer_path_integration, bool):
        raise TheoryError("defer_path_integration must be a boolean")
    devices = _resolve_theory_devices(device)
    sources = discover_sources(project_root, **config)
    schedule = load_schedule(sources)
    registry = load_statement_registry()
    manuscript = registry.get("manuscript_source", {})
    pdf = sources.root / manuscript.get("path", "revised.pdf")
    if pdf.is_file() and file_sha256(pdf) != manuscript.get("sha256"):
        raise TheoryError(
            f"Manuscript {pdf} differs from the verified statement registry; reconcile the statements before reducing"
        )
    print(
        f"[theory] validating {len(sources.selected)} selected prompts; cache-only {device}",
        flush=True,
    )
    support, support_metadata, source_records = build_support(
        sources,
        candidate_chunk_size=candidate_chunk_size,
        query_chunk_size=query_chunk_size,
        device="cpu",
    )
    package_dir = Path(__file__).parent
    code_hashes = {
        name: file_sha256(package_dir / name)
        for name in (
            "contracts.py",
            "cache_reader.py",
            "centers.py",
            "metrics.py",
            "scheduler_adapter.py",
            "support.py",
            "feedback.py",
            "summaries.py",
            "statements.py",
            "reduce.py",
        )
    }
    if prepare_candidate_endpoints:
        from .candidate_reduce import ENDPOINT_SOURCE_FILES, _source_hashes

        code_hashes.update(_source_hashes(ENDPOINT_SOURCE_FILES))
    if defer_path_integration:
        code_hashes["paper_measurements.py"] = file_sha256(
            package_dir / "paper_measurements.py"
        )
    numerical_policy = (
        {"base_policy": NUMERICAL_POLICY, "defer_path_integration": True}
        if defer_path_integration
        else NUMERICAL_POLICY
    )
    if sources.config["center"] == "cached-baseline":
        from utils.experiments.unconditional_baseline import load_unconditional_baseline

        baseline_args = {
            k: sources.config[k]
            for k in (
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
            baseline_args["num_baseline_seeds"] = read_json(meta_path)[
                "num_baseline_seeds"
            ]
        baseline = load_unconditional_baseline(
            sources.root, **baseline_args, load_tensor=False
        )
        sources.metadata_files[
            baseline.metadata_path.relative_to(sources.root).as_posix()
        ] = file_sha256(baseline.metadata_path)
    scientific_sources = {
        relative: _science_only(read_json(sources.root / relative))
        for relative in sources.metadata_files
        if relative.endswith(".json")
    }
    nonjson_sources = {
        k: v for k, v in sources.metadata_files.items() if not k.endswith(".json")
    }
    science_identity = {
        "config": sources.config,
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "reducer_sources": code_hashes,
        "source_science": scientific_sources,
        "source_scalar_hashes": nonjson_sources,
        "schedule_value_hash": canonical_hash(schedule),
        "selection_hash": sources.selection.sha256,
        "support_value_hash": support_metadata["tensor_sha256"],
        "support_aliases": support_metadata["aliases"],
        "registry": registry,
        "numerical_policy": numerical_policy,
        "defer_path_integration": defer_path_integration,
        "prepare_candidate_endpoints": prepare_candidate_endpoints,
        "path_integration_complete": not defer_path_integration,
        "integration_policy": INTEGRATION_POLICY,
        "reduction_backends": sorted({selected.type for selected in devices}),
        "time_block_size": TIME_BLOCK_SIZE,
        "chunk_sizes": [candidate_chunk_size, query_chunk_size],
        "ddpm_delta": 0.05,
        "reducer_package_versions": {
            name: package_version(name)
            for name in ("torch", "numpy", "scipy", "diffusers", "pandas", "pyarrow")
        },
        "smoke_record_limit": smoke_record_limit,
    }
    digest = canonical_hash(science_identity)
    parent = analysis_parent(sources.root, **sources.config)
    published_bundle = parent / digest
    # A forced rerun builds in a resumable sibling and leaves the current valid
    # scientific result and index intact until every required stage succeeds.
    replacing_complete = recompute and (published_bundle / "manifest.json").is_file()
    bundle = (
        parent / (".recompute-" + digest) if replacing_complete else published_bundle
    )
    bundle.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists() and not recompute:
        from .plotting import validate_bundle

        completed_manifest = validate_bundle(bundle, expected_config=sources.config)
        # Content hashes, not merely marker existence, protect completed runs.
        for role, records in (
            ("experiment", sources.selected),
            ("reference", sources.records["reference"]),
        ):
            for record in records:
                from utils.experiments.cache import validate_generation_record

                valid = validate_generation_record(
                    getattr(sources, role),
                    record.original_index,
                    load_tensors=False,
                    require_preview=False,
                )
                if not valid.valid:
                    raise TheoryError(
                        f"Stale source {record.original_index}: {valid.errors}"
                    )
        if completed_manifest["source_metadata_files"] != sources.metadata_files:
            completed_manifest.setdefault("source_metadata_history", []).append(
                completed_manifest["source_metadata_files"]
            )
            completed_manifest["source_metadata_files"] = sources.metadata_files
            atomic_write_json(manifest_path, completed_manifest)
        extension_valid = True
        if prepare_candidate_endpoints:
            from .candidate_reduce import shared_endpoint_record_valid

            expected_records = (
                sources.selected[:smoke_record_limit]
                if smoke_record_limit is not None
                else sources.selected
            )
            extension_valid = all(
                shared_endpoint_record_valid(bundle, record)
                for record in expected_records
            )
        if extension_valid:
            return bundle
    atomic_write_json(bundle / "analysis_identity.json", _json_safe(science_identity))
    failed = []
    failure_path = bundle / "failed.csv"
    failure_history = (
        pd.read_csv(failure_path, keep_default_na=False).to_dict("records")
        if failure_path.exists()
        else []
    )
    center, center_metadata = choose_center(
        sources,
        schedule,
        progress=lambda message: print(f"[theory] {message}", flush=True),
        devices=devices,
    )
    reference_center_value = center_metadata.pop("_reference_center", center)
    center_metadata.update(
        center_kind=sources.config["center"],
        evidence="model_center_diagnostic",
        vector_sha256=hashlib.sha256(
            center.detach().cpu().double().contiguous().numpy().tobytes()
        ).hexdigest(),
        source_run_hash=sources.runs["reference"]["scientific_config_hash"],
        source_schedule_sha256=file_sha256(sources.reference.schedule),
        center_norm_l2=float(center.double().norm()),
        initial_prediction_dtype=center_metadata.get(
            "source_dtype",
            center_metadata.get("reference_initial", {}).get("source_dtype"),
        ),
    )
    center_metadata["reference_check_run_hash"] = sources.runs["reference"][
        "scientific_config_hash"
    ]
    if sources.config["center"] != "reference-initial":
        independent = center_metadata.get("source_metadata", {})
        center_metadata["source_run_hash"] = independent.get("scientific_config_hash")
        center_metadata["source_schedule_sha256"] = independent.get("schedule_sha256")
        center_metadata["source_provenance_status"] = (
            "literal_zero_center_no_estimation_inputs"
            if sources.config["center"] == "zero"
            else "full_existing_independent_baseline_metadata_preserved; unavailable_hash_fields_remain_null"
        )
    center_hash = atomic_torch_save(center.cpu().contiguous(), bundle / "center.pt")
    support_hash = atomic_torch_save(
        support.atoms.cpu().contiguous(), bundle / "support.pt"
    )
    atomic_write_json(bundle / "center_metadata.json", _json_safe(center_metadata))
    atomic_write_json(bundle / "support_metadata.json", _json_safe(support_metadata))
    atomic_write_json(bundle / "statement_registry.json", registry)
    atomic_write_frame_parquet(source_records, bundle / "source_records.parquet")
    version = (
        sources.runs["experiment"]["scientific_config"]
        .get("package_versions", {})
        .get("diffusers")
    )
    adapter = SchedulerAdapter(schedule, recorded_diffusers_version=version)
    schedule_rows = [
        dataclasses.asdict(adapter.coefficients(k))
        for k in range(sources.config["num_inference_steps"])
    ]
    atomic_write_frame_csv(pd.DataFrame(schedule_rows), bundle / "schedule.csv")
    selected = sources.selected
    if smoke_record_limit is not None:
        if not isinstance(smoke_record_limit, int) or smoke_record_limit < 1:
            raise TheoryError("smoke_record_limit must be positive")
        selected = selected[:smoke_record_limit]
    # The same canonical experiment seed bank anchors every worker; no per-shard
    # refitting or deduplication against a different prompt is permitted.
    anchor_z, anchor_u, _, _ = load_record(
        sources.experiment, selected[0], initial_only=True, verify_hashes=False
    )
    evaluation_initial = (anchor_z, anchor_u)
    candidate_mean = (support.weights @ support.flat).reshape(support.latent_shape)
    auxiliary_hashes = {
        "candidate_mean.pt": atomic_torch_save(
            candidate_mean.cpu(), bundle / "candidate_mean.pt"
        ),
        "center.pt": center_hash,
        "support.pt": support_hash,
        "evaluation_initial.pt": atomic_torch_save(
            evaluation_initial, bundle / "evaluation_initial.pt"
        ),
        "worker_schedule.pt": atomic_torch_save(
            schedule, bundle / "worker_schedule.pt"
        ),
    }
    if prepare_candidate_endpoints:
        from .candidate_reduce import prepare_shared_endpoint_extension

        prepare_shared_endpoint_extension(
            bundle,
            analysis_hash=digest,
            config=sources.config,
            support_metadata=support_metadata,
            candidate_chunk_size=candidate_chunk_size,
            query_chunk_size=query_chunk_size,
            backend_types=sorted({selected.type for selected in devices}),
        )
    results = _dispatch_theory_shards(
        sources=sources,
        records=selected,
        bundle=bundle,
        analysis_hash=digest,
        devices=devices,
        recompute=False,
        candidate_chunk_size=candidate_chunk_size,
        query_chunk_size=query_chunk_size,
        auxiliary_hashes=auxiliary_hashes,
        defer_path_integration=defer_path_integration,
        prepare_candidate_endpoints=prepare_candidate_endpoints,
    )
    assignments = [index for result in results for index in result.assigned_indices]
    expected_indices = [record.original_index for record in selected]
    if len(assignments) != len(set(assignments)) or set(assignments) != set(
        expected_indices
    ):
        raise TheoryError(
            "Worker assignments do not cover every required prompt exactly once"
        )
    failed = [row for result in results for row in result.failed_rows]
    statuses = {
        index: status
        for result in results
        for index, status in result.status_by_index.items()
    }
    if set(statuses) != set(expected_indices):
        raise TheoryError(
            "Worker completion reports are missing required prompt identities"
        )
    if failed:
        atomic_write_frame_csv(pd.DataFrame(failure_history + failed), failure_path)
    if failed:
        raise TheoryError(
            f"{len(failed)} required record(s) failed; no complete manifest published. See {failure_path}"
        )
    # Preserve any previous failure audit rather than overwriting it on resume.
    if not failure_path.exists():
        atomic_write_frame_csv(
            pd.DataFrame(columns=["original_index", "record_id", "error", "traceback"]),
            failure_path,
        )
    # Workers publish their own compact shards. Only the coordinator reloads and
    # aggregates them, in canonical source order, before publishing completion.
    frames, initials, endpoints = [], [], []
    for record in selected:
        frame, initial, endpoint = [
            pd.read_parquet(path) for path in _record_shard_paths(bundle, record)
        ]
        _validate_shard(
            frame,
            record,
            sources.config["num_seeds"],
            sources.config["num_inference_steps"],
            sources.runs["experiment"]["scientific_config_hash"],
        )
        frames.append(frame)
        initials.append(initial)
        endpoints.append(endpoint)
    trajectory, initial, endpoint = [
        pd.concat(items, ignore_index=True) for items in (frames, initials, endpoints)
    ]
    expected = len(selected) * sources.config["num_seeds"]
    if (
        len(initial) != expected
        or len(endpoint) != expected
        or len(trajectory) != expected * sources.config["num_inference_steps"]
    ):
        raise TheoryError("Final scalar bundle row identities incomplete")
    # Summarize initial unconditional observations once per actual experiment seed.
    cz, cu = evaluation_initial
    first = adapter.coefficients(0)
    estimates = clean_initial(cz, cu, first.alpha, first.sigma)
    evaluation_mean = estimates.mean(0)
    center_metadata["evaluation_initial_dispersion_rmse"] = float(
        (estimates - evaluation_mean).square().mean().sqrt()
    )
    center_metadata["evaluation_reference_center_difference_rmse"] = float(
        (evaluation_mean - reference_center_value).square().mean().sqrt()
    )
    center_metadata["evaluation_unique_seed_count"] = len(estimates)
    center_metadata["evaluation_seeds"] = list(range(sources.config["num_seeds"]))
    center_metadata["evaluation_repeated_prediction_max_abs_difference"] = float(
        initial.initial_unconditional_consistency_max_abs.max()
    )
    center_metadata["evaluation_repeated_prediction_tolerance"] = float(
        initial.initial_unconditional_consistency_tolerance.max()
    )
    center_metadata["unique_evaluation_dispersion_quantiles"] = summaries.quantiles(
        (estimates - center).flatten(1).norm(dim=1).cpu().numpy()
        / math.sqrt(center.numel())
    )
    atomic_write_json(bundle / "center_metadata.json", _json_safe(center_metadata))
    atomic_write_frame_parquet(initial, bundle / "initial_metrics.parquet")
    atomic_write_frame_parquet(endpoint, bundle / "endpoint_metrics.parquet")
    summary = _summary(initial, trajectory, endpoint, center_metadata)
    summary["analysis_scope"] = (
        "complete_frozen_selection"
        if smoke_record_limit is None
        else "explicit_smoke_prefix; not_full_experiment"
    )
    summary["smoke_record_limit"] = smoke_record_limit
    summary["eligible_prompt_count"] = len(sources.selected)
    atomic_write_json(bundle / "summary.json", summary)
    statement_ids = [
        item.get("semantic_id", item.get("id")) for item in registry["statements"]
    ]
    unique_initial = (
        initial.sort_values(
            ["run_id", "record_id", "original_index", "seed"], kind="stable"
        )
        .drop_duplicates(["run_id", "seed"])
        .copy()
    )
    if len(unique_initial) != sources.config["num_seeds"]:
        raise TheoryError(
            "Initial unconditional CDF must contain one input per evaluation seed"
        )
    for semantic_id, table in {
        "initial_recovery": initial,
        "unconditional_center": unique_initial,
        "posterior_feedback": trajectory,
        "target_synchronization": trajectory,
        "target_injection": initial,
        "terminal_terms": endpoint,
    }.items():
        atomic_write_frame_parquet(
            table, bundle / "plotdata" / f"{semantic_id}.parquet"
        )
    for name, table in {
        "synchronization": summaries.synchronization_summary(trajectory),
        "paired_synchronization": summaries.paired_synchronization_summary(trajectory),
        "phases": summaries.phase_summary(
            trajectory, sources.config["num_inference_steps"]
        ),
        "initial_by_prompt": summaries.initial_prompt_summary(initial),
        "feedback_by_step": summaries.feedback_summary(trajectory),
        "feedback_by_prompt": summaries.feedback_summary(
            trajectory, summaries.PROMPT_KEYS
        ),
    }.items():
        atomic_write_frame_parquet(table, bundle / "summaries" / f"{name}.parquet")
    tolerance = sources.config.get("target_error_tolerance")
    if tolerance is not None:
        atomic_write_frame_parquet(
            summaries.tolerance_entry_summary(trajectory, tolerance),
            bundle / "summaries" / "tolerance_entries.parquet",
        )
    availability = summaries.applicability_report(
        initial, trajectory, endpoint, center_metadata, support_metadata
    )
    if tolerance is not None:
        availability["tolerance"] = {
            "status": "explicitly_supplied",
            "value": tolerance,
            "units": "raw_latent_L2",
            "source": "analysis_configuration",
            "not_inferred_from_sscd": True,
        }
    atomic_write_json(bundle / "applicability_report.json", _json_safe(availability))
    numerical_files = {
        p.relative_to(bundle).as_posix(): file_sha256(p)
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
        and p.suffix in {".json", ".csv", ".parquet"}
        and "figures" not in p.parts
        and "candidate_endpoint_extension" not in p.relative_to(bundle).parts
        and p.name != "manifest.json"
    }
    revision = subprocess.run(
        ["git", "-C", str(package_dir.parents[2]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "active_code_revision": revision.stdout.strip()
        if revision.returncode == 0
        else None,
        "reducer_source_hashes": code_hashes,
        "complete": True,
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "analysis_hash": digest,
        "config": sources.config,
        "source_root": str(sources.root),
        "source_metadata_files": sources.metadata_files,
        "source_marker_inventory": sources.marker_inventory,
        "numerical_files": numerical_files,
        "auxiliary_tensor_files": auxiliary_hashes,
        "execution": {
            "requested_device": str(device),
            "resolved_devices": [str(selected) for selected in devices],
            "worker_count": len(results),
            "device_shards": [dataclasses.asdict(result) for result in results],
            "gpu_staging_time_steps": TIME_BLOCK_SIZE,
            "scalar_host_transfer_policy": "one packed transfer per device and dtype per query batch",
        },
        "scientific_generation_hash": sources.runs["experiment"][
            "scientific_config_hash"
        ],
        "reference_hash": sources.runs["reference"]["scientific_config_hash"],
        "schedule_sha256": file_sha256(sources.experiment.schedule),
        "reference_schedule_sha256": file_sha256(sources.reference.schedule),
        "selection_hash": sources.selection.sha256,
        "sscd_configurations": sources.sscd_config,
        "expected_counts": {
            "prompts": len(selected),
            "seeds_per_prompt": sources.config["num_seeds"],
            "trajectory_rows": len(trajectory),
            "initial_rows": len(initial),
            "endpoint_rows": len(endpoint),
        },
        "statement_ids": statement_ids,
        "default_figure_ids": registry["default_figure_ids"],
        "diagnostic_figure_ids": registry["diagnostic_figure_ids"],
        "analysis_scope": summary["analysis_scope"],
        "current_failed_record_count": 0,
        "failed_csv_preserves_history": True,
        "numerical_policy": numerical_policy,
        "defer_path_integration": defer_path_integration,
        "prepare_candidate_endpoints": prepare_candidate_endpoints,
        "path_integration_complete": not defer_path_integration,
        "integration_policy": INTEGRATION_POLICY,
        "scheduler_adapter": adapter.metadata(),
        "metric_evidence": {
            "branch_geometry": "finite_noise_observation",
            "cache_identities": "algebraic_qa",
            "posterior_fields": "candidate_distribution_diagnostic",
            "center_fields": "model_center_diagnostic",
            "population_and_paper_bounds": "unavailable",
        },
    }
    if prepare_candidate_endpoints:
        from .candidate_reduce import shared_endpoint_extension_manifest

        manifest["candidate_endpoint_extension"] = shared_endpoint_extension_manifest(
            bundle, selected
        )
    # Refuse publication if a producer changed completion/scientific metadata
    # during this read-only reduction.
    for relative, expected_hash in sources.metadata_files.items():
        if file_sha256(sources.root / relative) != expected_hash:
            raise TheoryError(
                f"Source metadata changed during reduction: {sources.root / relative}"
            )
    atomic_write_json(manifest_path, _json_safe(manifest))
    if replacing_complete:
        archive = parent / "archived_analyses"
        archive.mkdir(exist_ok=True)
        previous = archive / (digest + "-" + str(time.time_ns()))
        os.replace(published_bundle, previous)
        try:
            os.replace(bundle, published_bundle)
        except BaseException:
            os.replace(previous, published_bundle)
            raise
        bundle = published_bundle
    if smoke_record_limit is None:
        index_path = parent / "index.json"
        index = read_json(index_path) if index_path.exists() else {"analyses": []}
        index["analyses"] = [
            entry
            for entry in index["analyses"]
            if entry.get("config") != sources.config
        ]
        index["analyses"].append({"config": sources.config, "analysis_hash": digest})
        atomic_write_json(index_path, index)
    return bundle


def run_theory(
    project_root,
    *,
    recompute=False,
    device="auto",
    candidate_chunk_size=256,
    query_chunk_size=16,
    smoke_record_limit=None,
    defer_path_integration=False,
    prepare_candidate_endpoints=False,
    **config,
):
    """Public reduction entry point with durable failure auditing.

    A preflight or center failure has no complete per-record bundle yet. Keep
    those failures beside the bundles; per-record failures remain in failed.csv.
    """
    try:
        return _run_theory(
            project_root,
            recompute=recompute,
            device=device,
            candidate_chunk_size=candidate_chunk_size,
            query_chunk_size=query_chunk_size,
            smoke_record_limit=smoke_record_limit,
            defer_path_integration=defer_path_integration,
            prepare_candidate_endpoints=prepare_candidate_endpoints,
            **config,
        )
    except Exception as error:
        try:
            report = analysis_parent(project_root, **config) / "preflight_failed.csv"
            history = (
                pd.read_csv(report, keep_default_na=False).to_dict("records")
                if report.exists()
                else []
            )
            history.append(
                {
                    "scope": "source_validation_or_reduction",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            atomic_write_frame_csv(pd.DataFrame(history), report)
        except (OSError, ValueError, TheoryError):
            pass  # Preserve the original, more useful source/contract error.
        raise
