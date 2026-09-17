"""Compact paper inputs from saved scalars; optional independent terminal accounting.

Plot-input helpers do no file I/O and never load or evaluate learned components.
The explicit archived-terminal supplement is analysis-only and reads a caller-selected log root.
The tensor helper accepts a caller-supplied last-update batch; all plot-input
builders operate exclusively on already validated scalar tables.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .candidate_summaries import GROUPS, RECOVERY_TOLERANCES, prompt_weights
from .summaries import PROMPT_KEYS, SAMPLE_KEYS, weighted_quantiles

PAPER_MEASUREMENT_VERSION = "paper-measurements-1"
OUTCOMES = GROUPS[1:]
WEIGHTING = (
    "Equal mass per represented complete prompt identity, divided equally over its "
    "structurally eligible seeds. Numerical missingness retains its original mass "
    "in counts; finite-value distributions are explicitly conditional on finite mass."
)
# Units are per displayed axis. Keeping every stem explicit prevents a generic
# distance label from silently propagating to ranks, coordinates, or ratios.
_DISTANCE = "latent L2 / sqrt(d), with no additional dimension normalization"
_CDF = "dimensionless empirical cumulative probability"
_SNR = "dimensionless current SNR"
FIGURE_AXIS_UNITS = {
    "initial_recovery": {"x": _DISTANCE, "y": _DISTANCE},
    "initial_unconditional_concentration": {"x": _DISTANCE, "y": _CDF},
    "target_synchronization": {"x": _SNR, "y": _DISTANCE},
    "initial_recovery_within_prompt": {
        "x": "within-prompt centered difference of latent L2 target errors / sqrt(d)",
        "y": "within-prompt centered SSCD, dimensionless",
    },
    "initial_target_retrieval_rank": {
        "x": "dimensionless strict Euclidean retrieval rank; no division by sqrt(d)",
        "y": _CDF,
    },
    "branch_gap": {"x": _SNR, "y": _DISTANCE},
    "joint_target_recovery_early": {"x": _DISTANCE, "y": _CDF},
    "joint_target_recovery_late": {"x": _DISTANCE, "y": _CDF},
    "terminal_error_terms": {"x": _DISTANCE, "y": _DISTANCE},
    "terminal_bound_tightness": {
        "x": "dimensionless bound / observed terminal error; numerator and denominator have the same latent L2 / sqrt(d) units",
        "y": _CDF,
    },
    "initial_candidate_reference_discrepancy": {"x": _DISTANCE, "y": _DISTANCE},
    "initial_reference_snr_sweep": {
        "x": "dimensionless analytical SNR on the fixed saved grid",
        "y": _DISTANCE,
    },
    "initial_injection_geometry": {
        "x": "dimensionless target-direction coefficient <J,v>/||v||^2",
        "y": "dimensionless perpendicular norm / target-direction norm",
    },
    "initial_target_coordinates": {
        "x": "dimensionless conditional target-direction coordinate <m_c-mu,v>/||v||^2",
        "y": "dimensionless guided target-direction coordinate <m_g-mu,v>/||v||^2",
    },
    "initial_injection_mismatch": {
        "x": "dimensionless same-seed terminal SSCD",
        "y": "dimensionless relative full-vector injection mismatch",
    },
    "synchronization_bound_components": {"x": _SNR, "y": _DISTANCE},
    "matched_update_residual": {
        "x": "stored update index, dimensionless",
        "y": _DISTANCE,
    },
    "last_prediction_error_terms": {"x": _DISTANCE, "y": _DISTANCE},
    "terminal_cancellation": {
        "x": "dimensionless error-guidance cosine",
        "y": "dimensionless observed error / triangle-bound ratio",
    },
}
_SCATTER_WEIGHTING = (
    "One unaggregated point per retained prompt/seed observation; no group "
    "average or weighted fit. Numerically missing rows remain in the saved table."
)
FIGURE_WEIGHTING = {name: WEIGHTING for name in FIGURE_AXIS_UNITS}
FIGURE_WEIGHTING.update(
    {
        name: _SCATTER_WEIGHTING
        for name in (
            "initial_recovery",
            "terminal_error_terms",
            "initial_injection_geometry",
            "initial_target_coordinates",
            "initial_injection_mismatch",
            "last_prediction_error_terms",
            "terminal_cancellation",
        )
    }
)
FIGURE_WEIGHTING.update(
    {
        "initial_unconditional_concentration": "Separate populations: equal weight per unique evaluation run/seed pair; saved empirical-law mass per distinct candidate atom, including duplicate source-record multiplicity.",
        "initial_candidate_reference_discrepancy": "One unaggregated observation per unique initial evaluation run/seed pair; no duplication over prompts and no cohort aggregation.",
        "initial_reference_snr_sweep": "Equal weight over the same unique evaluation run/seed pairs at every analytical SNR.",
        "initial_recovery_within_prompt": "Center both variables over the same prompt's finite paired seeds before outcome grouping. Retain degenerate rows; the saved descriptive trend uses equal prompt mass divided over finite seeds in prompts with usable within-prompt variation. Missing and degenerate rows are excluded from that fit with saved statuses.",
        "synchronization_bound_components": "The displayed metric comparison uses saved all-population curves; outcome-cohort curves also remain saved. "
        + WEIGHTING,
        "matched_update_residual": "The displayed metric comparison uses saved all-population curves; outcome-cohort curves also remain saved. "
        + WEIGHTING,
    }
)


def _numeric(frame, name):
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def _bool(frame, name, default=False):
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[name].fillna(default).astype(bool)


def _require(frame, names, context):
    missing = set(names) - set(frame)
    if missing:
        raise ValueError(f"{context}: missing saved fields {sorted(missing)}")


def _counts(frame):
    result = {"rows": len(frame)}
    for name, keys in (("prompts", PROMPT_KEYS), ("samples", SAMPLE_KEYS)):
        if set(keys) <= set(frame):
            result[name] = len(frame[list(keys)].drop_duplicates())
    if "seed" in frame:
        result["unique_seed_ids"] = int(frame.seed.nunique())
    if "candidate_target_atom_id" in frame:
        result["distinct_target_atoms"] = int(frame.candidate_target_atom_id.nunique())
    if "terminal_sscd" in frame:
        scores = _numeric(frame, "terminal_sscd")
        result["missing_sscd"] = int((~np.isfinite(scores)).sum())
        result["out_of_range_sscd"] = int(((scores < 0) | (scores > 1)).sum())
        if set(PROMPT_KEYS) <= set(frame):
            high = frame.loc[scores > 0.75, PROMPT_KEYS].drop_duplicates()
            low = frame.loc[
                np.isfinite(scores) & (scores <= 0.75), PROMPT_KEYS
            ].drop_duplicates()
            result["overlapping_outcome_prompts"] = len(high.merge(low, on=PROMPT_KEYS))
    return result


def _groups(frame):
    scores = _numeric(frame, "terminal_sscd")
    for name, mask in (
        (OUTCOMES[0], scores > 0.75),
        (OUTCOMES[1], np.isfinite(scores) & (scores <= 0.75)),
    ):
        yield name, frame.loc[mask]


def _identity(frame):
    return [
        c
        for c in (
            *SAMPLE_KEYS,
            "candidate_target_atom_id",
            "step_index",
            "timestep",
            "snr",
            "latent_dimension",
        )
        if c in frame
    ]


def _scatter(frame, x, y):
    _require(frame, [x, y], "paper scatter")
    columns = list(
        dict.fromkeys(
            [
                *_identity(frame),
                *[c for c in ("terminal_sscd", "within_prompt_status") if c in frame],
            ]
        )
    )
    result = frame[columns].copy()
    result["x"], result["y"] = _numeric(frame, x), _numeric(frame, y)
    result["numeric_status"] = np.where(
        np.isfinite(result.x) & np.isfinite(result.y),
        "available",
        "missing_numeric_value",
    )
    return result


def _finite_limits(*arrays):
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return [float(min(0.0, values.min())), float(max(0.0, values.max()))]


def exact_weighted_ecdf(
    frame, value_column, *, weighted=True, weight_column=None, group="all", branch=None, distribution=None
):
    """Exact support; missing observations never change other seeds' base mass."""
    values = _numeric(frame, value_column).to_numpy()
    finite = np.isfinite(values)
    if weight_column is not None:
        _require(frame, [weight_column], "explicit ECDF masses")
        weights = _numeric(frame, weight_column).to_numpy()
        if not (np.isfinite(weights) & (weights > 0)).all():
            raise ValueError("Explicit ECDF masses must be finite and positive")
    else:
        weights = prompt_weights(frame).to_numpy() if weighted else np.ones(len(frame))
    structural_mass = float(weights.sum())
    values, weights = values[finite], weights[finite]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    finite_mass = float(weights.sum())
    metadata = {
        **_counts(frame),
        "finite_count": len(values),
        "missing_count": int((~finite).sum()),
        "structural_weight": structural_mass,
        "finite_weight": finite_mass,
        "missing_weight": structural_mass - finite_mass,
        "status": "available"
        if len(values)
        else "empty_group"
        if frame.empty
        else "missing_numeric_values",
    }
    columns = [
        "value",
        "cdf",
        "weight",
        "group",
        "branch",
        "distribution",
        "finite_count",
        "missing_count",
        "structural_weight",
        "finite_weight",
    ]
    if not len(values):
        return pd.DataFrame(columns=columns), metadata
    cumulative = np.cumsum(weights) / finite_mass
    # Keep one support location at the right edge of each exact tie, including
    # all its mass. This is an exact ECDF, not a threshold-grid approximation.
    ends = np.r_[values[:-1] != values[1:], True]
    positions = np.flatnonzero(ends)
    starts = np.r_[0, positions[:-1] + 1]
    result = pd.DataFrame(
        {
            "value": values[positions],
            "cdf": cumulative[positions],
            "weight": [
                float(weights[a : b + 1].sum())
                for a, b in zip(starts, positions, strict=True)
            ],
            "group": group,
            "branch": branch,
            "distribution": distribution,
            "finite_count": len(values),
            "missing_count": int((~finite).sum()),
            "structural_weight": structural_mass,
            "finite_weight": finite_mass,
        }
    )
    return result[columns], metadata


def repair_current_reference(frame):
    """Restore current-state reference independently of next-state eligibility."""
    rows = frame.copy(deep=False)
    current = _numeric(rows, "candidate_current_reference_error_u_l2")
    base = _numeric(rows, "candidate_unconditional_reference_error_l2")
    current_noise = _numeric(rows, "sigma")
    if "sigma" not in rows:
        current_noise = _numeric(rows, "current_sigma")
    positive_current = np.isfinite(current_noise) & (current_noise > 0)
    if "sigma" not in rows and "current_sigma" not in rows:
        # The row-local saved support contract establishes current-noise
        # applicability when its scalar coefficients are absent.
        positive_current = (
            rows.get("support_status", pd.Series("", index=rows.index))
            .astype(str)
            .eq("valid")
        )
    primary = positive_current & np.isfinite(current) & (current >= 0)
    fallback = positive_current & current.isna() & np.isfinite(base) & (base >= 0)
    repaired = current.where(primary).where(~fallback, base)
    dimension = _numeric(rows, "latent_dimension")
    valid_dimension = np.isfinite(dimension) & (dimension > 0)
    repaired = repaired.where(valid_dimension)
    rows["paper_current_reference_error_u_l2"] = repaired
    rows["current_unconditional_candidate_reference_error_rmse"] = repaired / np.sqrt(
        dimension
    )
    rows["paper_current_reference_source"] = np.select(
        [primary & valid_dimension, fallback & valid_dimension],
        [
            "saved_candidate_endpoint_current_reference",
            "saved_base_v3_current_reference",
        ],
        default="unavailable",
    )
    rows["paper_current_reference_status"] = np.select(
        [
            ~valid_dimension,
            ~positive_current,
            current.notna() & (~np.isfinite(current) | (current < 0)),
            np.isfinite(repaired),
        ],
        [
            "invalid_latent_dimension",
            "unavailable_current_noise_contract",
            "invalid_saved_current_reference_norm",
            "available",
        ],
        default="missing_current_reference_measurement",
    )
    radius, logtail = (
        _numeric(rows, "candidate_radius_l2"),
        _numeric(rows, "target_log_complement"),
    )
    valid_tail = (
        positive_current
        & valid_dimension
        & np.isfinite(radius)
        & (radius >= 0)
        & (np.isfinite(logtail) | np.isneginf(logtail))
        & (logtail <= 0)
    )
    with np.errstate(under="ignore"):
        rows["candidate_radius_tail_rmse"] = (
            radius * np.exp(logtail) / np.sqrt(dimension)
        ).where(valid_tail)
    rows["paper_current_reference_tail_status"] = np.where(
        valid_tail, "available", "missing_or_invalid_current_reference_tail"
    )
    audit = {
        "cause": "Unavailable next-state endpoint placeholders previously shadowed the valid base-v3 current-state reference column.",
        "repair": "Per-row saved-scalar fallback; no posterior recomputation or new prediction.",
        "source_counts": rows.paper_current_reference_source.value_counts().to_dict(),
        "status_counts": rows.paper_current_reference_status.value_counts().to_dict(),
    }
    if "step_index" in rows and len(rows):
        last = rows.loc[rows.step_index == rows.step_index.max()]
        audit["last_prediction"] = {
            "step_index": int(rows.step_index.max()),
            "rows": len(last),
            "source_counts": last.paper_current_reference_source.value_counts().to_dict(),
            "status_counts": last.paper_current_reference_status.value_counts().to_dict(),
        }
    return rows, audit


def _curves(frame, metrics, *, include_all=False):
    output = []
    steps = sorted(frame.step_index.unique())
    schedules = {
        step: rows.iloc[0] for step, rows in frame.groupby("step_index", sort=True)
    }
    populations = [("all", frame)] if include_all else []
    populations.extend(_groups(frame))
    for group, cohort in populations:
        by_step = {step: rows for step, rows in cohort.groupby("step_index", sort=True)}
        for step in steps:
            rows = by_step.get(step, cohort.iloc[:0])
            weights = prompt_weights(rows).to_numpy()
            schedule = schedules[step]
            for metric, branch in metrics.items():
                values = _numeric(rows, metric).to_numpy()
                finite = np.isfinite(values)
                v, w = values[finite], weights[finite]
                quantiles = weighted_quantiles(v, w, (0.25, 0.5, 0.75))
                output.append(
                    {
                        "group": group,
                        "branch": branch,
                        "metric": metric,
                        "step_index": int(step),
                        "timestep": schedule.get("timestep", np.nan),
                        "snr": schedule.get("snr", np.nan),
                        "q25": quantiles[0],
                        "median": quantiles[1],
                        "q75": quantiles[2],
                        "minimum": float(v.min()) if len(v) else np.nan,
                        "maximum": float(v.max()) if len(v) else np.nan,
                        "n_observations": len(v),
                        "total_count": len(rows),
                        "missing_count": int((~finite).sum()),
                        "prompt_count": len(rows[PROMPT_KEYS].drop_duplicates()),
                        "seed_count": rows.seed.nunique(),
                        "weighted_denominator": float(weights.sum()),
                        "finite_weight": float(w.sum()),
                        "missing_weight": float(weights[~finite].sum()),
                        "status": "available"
                        if len(v)
                        else "empty_group"
                        if rows.empty
                        else "missing_numeric_values",
                    }
                )
    return pd.DataFrame(output)


def terminal_contract_status(endpoint):
    """Distinguish structural exclusions, missing metadata, and failed replay."""
    rows = endpoint[_identity(endpoint)].copy()
    statuses, reasons, reconstruction, accounting = [], [], [], []
    for item in endpoint.to_dict("records"):
        required = (
            "A",
            "kappa",
            "destination_alpha",
            "destination_sigma",
            "destination_noise_std",
            "affine_applicable",
            "scheduler_status",
        )
        missing = [key for key in required if key not in item or pd.isna(item[key])]
        if missing:
            statuses.append("missing_contract_metadata")
            reasons.append("missing saved scheduler fields: " + ",".join(missing))
            reconstruction.append("unavailable")
            accounting.append("missing_contract_metadata")
            continue
        affine = bool(item["affine_applicable"])
        if not affine:
            unsupported = str(item["scheduler_status"]).startswith(
                "valid_nonlinear"
            ) or str(item["scheduler_status"]).startswith(
                "unsupported_branch_dependent"
            )
            statuses.append(
                "unsupported_nonaffine_update"
                if unsupported
                else "missing_contract_metadata"
            )
            reasons.append(str(item["scheduler_status"]))
            reconstruction.append("not_applicable" if unsupported else "unavailable")
            accounting.append(
                "unsupported_nonaffine_update"
                if unsupported
                else "missing_contract_metadata"
            )
            continue
        structural = (
            item["A"] == 0
            and item["kappa"] == 1
            and item["destination_alpha"] == 1
            and item["destination_sigma"] == 0
            and item["destination_noise_std"] == 0
        )
        stochastic = item["destination_noise_std"] != 0
        residual = item.get("independent_replay_rmse", np.nan)
        tolerance = item.get("update_rounding_sensitivity_rmse", np.nan)
        residual = float(residual) if residual is not None else np.nan
        tolerance = float(tolerance) if tolerance is not None else np.nan
        independently_measured = (
            np.isfinite(residual) and np.isfinite(tolerance) and tolerance >= 0
        )
        supplied = item.get("terminal_accounting_status") == "available"
        supplied_pass = item.get("terminal_accounting_numeric_pass")
        supplied_known = (
            supplied and supplied_pass is not None and not pd.isna(supplied_pass)
        )
        failed = (independently_measured and residual > tolerance) or (
            supplied_known and not bool(supplied_pass)
        )
        measured = independently_measured or supplied_known
        if supplied_known:
            reconstruction.append("failed" if failed else "passed")
            accounting.append("independent_affine_accounting_available")
        elif stochastic:
            reconstruction.append("unavailable_independent_saved_sampling_noise")
            accounting.append("unavailable_saved_sampling_noise")
        elif independently_measured:
            reconstruction.append("failed" if failed else "passed")
            accounting.append("independent_deterministic_replay_available")
        else:
            reconstruction.append("unavailable")
            accounting.append("missing_numeric_reconstruction")
        if failed:
            statuses.append("failed_numeric_reconstruction")
            reasons.append(
                "independent affine replay/accounting exceeds saved source-precision sensitivity"
            )
        elif (supplied and not supplied_known) or (not stochastic and not measured):
            statuses.append("missing_contract_metadata")
            reasons.append(
                "independent affine replay/accounting numeric result missing"
            )
        elif structural:
            statuses.append("clean_update_verified")
            reasons.append(
                "zero independent additive/state term, unit clean coefficient, and numerical replay passed"
            )
        else:
            statuses.append("nonclean_affine_update")
            reasons.append(
                "nonzero state/additive coefficient or nonunit clean coefficient; exact clean-update prerequisite fails"
            )
    rows["paper_terminal_status"], rows["paper_terminal_reason"] = statuses, reasons
    (
        rows["paper_terminal_reconstruction_status"],
        rows["paper_terminal_accounting_status"],
    ) = reconstruction, accounting
    rows["paper_terminal_clean_verified"] = rows.paper_terminal_status.eq(
        "clean_update_verified"
    )
    blocking = rows.paper_terminal_status.isin(
        ["missing_contract_metadata", "failed_numeric_reconstruction"]
    )
    return rows, {
        "status_counts": rows.paper_terminal_status.value_counts().to_dict(),
        "accounting_status_counts": rows.paper_terminal_accounting_status.value_counts().to_dict(),
        "blocking": bool(blocking.any()),
        "blocking_rows": rows.loc[blocking].to_dict("records"),
    }


def terminal_affine_accounting(
    adapter,
    state,
    next_state,
    epsilon_u,
    epsilon_c,
    target,
    guidance,
    *,
    step,
    realized_noise=None,
    noise_provenance=None,
):
    """Independent last-update vector accounting; never fit a from x_out.

    ``realized_noise`` is the actual additive noise contribution, already scaled
    by the saved noise coefficient. It must be independently recorded, not an
    innovation recovered by subtracting a predicted drift from the endpoint.
    """
    import torch
    from .metrics import clean_estimates

    original = [
        torch.as_tensor(x) for x in (state, next_state, epsilon_u, epsilon_c, target)
    ]
    source_eps = max(torch.finfo(x.dtype).eps for x in original)
    z, out, eu, ec, target = [x.to(dtype=torch.float64) for x in original]
    coeff = adapter.coefficients(step)
    dims = tuple(range(-target.ndim, 0))
    dimension = target.numel()

    def norm(value):
        return value.square().sum(dim=dims).sqrt()

    result = {
        "terminal_accounting_kappa": coeff.B,
        "terminal_accounting_state_coefficient": coeff.A,
        "terminal_accounting_noise_std": coeff.noise_std,
        "terminal_accounting_evidence": "independent_scheduler_accounting_not_Theorem7",
    }
    if not coeff.affine:
        return result | {"terminal_accounting_status": "unsupported_nonaffine_update"}
    if coeff.noise_std != 0:
        if realized_noise is None:
            return result | {
                "terminal_accounting_status": "unavailable_saved_sampling_noise"
            }
        if noise_provenance not in {
            "independently_saved_additive_noise",
            "independently_saved_rng_replay",
        }:
            raise ValueError(
                "Terminal accounting requires independently recorded noise, never an endpoint-fitted residual"
            )
        noise = torch.as_tensor(realized_noise, dtype=torch.float64, device=z.device)
        if noise.shape != z.shape or not bool(torch.isfinite(noise).all()):
            raise ValueError("Invalid independently saved terminal noise contribution")
    else:
        noise = torch.zeros_like(z)
        noise_provenance = "structurally_zero_additive_noise"
    _, mc, delta, mg = clean_estimates(
        z, eu, ec, coeff.alpha, coeff.sigma, guidance, latent_ndim=target.ndim
    )
    additive = coeff.A * z + noise
    terms = [
        additive + (coeff.B - 1) * target,
        coeff.B * (mc - target),
        coeff.B * (float(guidance) - 1) * delta,
    ]
    predicted = additive + coeff.B * mg
    residual = norm(out - predicted) / math.sqrt(dimension)
    tolerance = (
        64
        * source_eps
        * (norm(out) + abs(coeff.A) * norm(z) + norm(noise) + abs(coeff.B) * norm(mg))
        / math.sqrt(dimension)
    )
    result.update(
        terminal_accounting_status="available",
        terminal_accounting_noise_provenance=noise_provenance,
        terminal_accounting_residual_rmse=residual,
        terminal_accounting_tolerance_rmse=tolerance,
        terminal_accounting_numeric_pass=torch.isfinite(residual)
        & (residual <= tolerance),
        terminal_accounting_additive_l2=norm(additive),
        terminal_accounting_observed_error_l2=norm(out - target),
        terminal_accounting_predicted_error_l2=norm(sum(terms)),
    )
    for i, term in enumerate(terms):
        result[f"terminal_accounting_term{i}_l2"] = norm(term)
        result[f"terminal_accounting_term{i}_rmse"] = norm(term) / math.sqrt(dimension)
        for j in range(i):
            cross = (term * terms[j]).sum(dim=dims)
            result[f"terminal_accounting_cross{j}{i}_squared_l2"] = 2 * cross
            result[f"terminal_accounting_cross{j}{i}_per_dimension"] = (
                2 * cross / dimension
            )
    result["terminal_accounting_error_squared_l2_from_terms"] = sum(
        norm(t).square() for t in terms
    ) + sum(
        result[f"terminal_accounting_cross{j}{i}_squared_l2"]
        for i in range(3)
        for j in range(i)
    )
    return result


def build_nonfeedback_plot_inputs(tables, *, config, diagnostics=False):
    """Return compact per-stem DataFrames plus saved rendering/audit metadata."""
    initial, trajectory, endpoint = (
        tables[name] for name in ("initial", "trajectory", "endpoint")
    )
    _require(
        initial,
        SAMPLE_KEYS
        + [
            "conditional_target_error_rmse",
            "unconditional_target_error_rmse",
            "terminal_sscd",
        ],
        "initial paper population",
    )
    _require(
        trajectory,
        SAMPLE_KEYS
        + [
            "step_index",
            "timestep",
            "snr",
            "conditional_target_error_rmse",
            "unconditional_target_error_rmse",
            "branch_gap_rmse",
            "terminal_sscd",
        ],
        "trajectory paper population",
    )
    if (
        initial.duplicated(SAMPLE_KEYS).any()
        or trajectory.duplicated(SAMPLE_KEYS + ["step_index"]).any()
    ):
        raise ValueError("Duplicate scientific identity in saved paper population")
    summaries = tables.get("summaries", {})
    manifest = tables.get("manifest", {})
    outputs, figures = {}, {}
    audit = {
        "blocking": False,
        "initial_population": _counts(initial),
        "measurement_version": PAPER_MEASUREMENT_VERSION,
    }

    def register(stem, data, *, status="available", reason=None, **extra):
        if not len(data.columns):
            data = pd.DataFrame(columns=["numeric_status"])
        outputs[stem] = data
        figures[stem] = {
            "status": status,
            "reason": reason,
            "counts": _counts(data),
            "population_counts": _counts(initial),
            "weighting": FIGURE_WEIGHTING[stem],
            "axis_units": FIGURE_AXIS_UNITS[stem].copy(),
            "normalization": "; ".join(
                axis + ": " + units for axis, units in FIGURE_AXIS_UNITS[stem].items()
            ),
            "actual_initial_snr": float(
                trajectory.loc[
                    trajectory.step_index == trajectory.step_index.min(), "snr"
                ].iloc[0]
            ),
            "columns": list(data.columns),
            "dtypes": {c: str(data[c].dtype) for c in data},
            **extra,
        }

    scatter = _scatter(
        initial, "unconditional_target_error_rmse", "conditional_target_error_rmse"
    )
    limits = _finite_limits(scatter.x, scatter.y)
    register(
        "initial_recovery",
        scatter,
        x_limits=limits,
        y_limits=limits,
        data_sources=["initial.parquet"],
        reference_lines=["Equal target errors"],
    )

    baseline, bank = tables["reference_initial"], tables["bank_geometry"]
    _require(
        baseline,
        ["run_id", "seed", "unconditional_center_rmse"],
        "unique initial baseline",
    )
    _require(bank, ["candidate_atom_center_distance_rmse", "weight"], "candidate bank geometry")
    if baseline.duplicated(["run_id", "seed"]).any():
        raise ValueError(
            "Initial unconditional baseline must have one observation per unique run/seed"
        )
    expected_seeds = int(config.get("num_seeds", config.get("N", len(baseline))))
    for _, rows in baseline.groupby("run_id", dropna=False):
        if set(rows.seed.astype(int)) != set(range(expected_seeds)):
            raise ValueError(
                "Initial baseline does not retain the complete evaluation seed block"
            )
    for key in ("candidate_atom_id", "atom_id"):
        if key in bank and bank[key].duplicated().any():
            raise ValueError("Candidate baseline requires exact distinct atoms")
    initial_ecdf, initial_counts = exact_weighted_ecdf(
        baseline,
        "unconditional_center_rmse",
        weighted=False,
        distribution="initial_unconditional",
    )
    atom_ecdf, atom_counts = exact_weighted_ecdf(
        bank,
        "candidate_atom_center_distance_rmse",
        weight_column="weight",
        distribution="candidate_atoms",
    )
    center = manifest.get("center_metadata", {})
    if set(center.get("seeds", [])) != set(
        range(expected_seeds, 2 * expected_seeds)
    ) or set(center.get("evaluation_seeds", [])) != set(range(expected_seeds)):
        raise ValueError(
            "Independent reference/evaluation center seed provenance is missing or inconsistent"
        )
    if not center.get("vector_sha256"):
        raise ValueError("Fixed reference-center vector identity is missing")
    discrepancy = center.get("evaluation_repeated_prediction_max_abs_difference")
    sensitivity = center.get("evaluation_repeated_prediction_tolerance")
    if (
        discrepancy is None
        or sensitivity is None
        or not (
            math.isfinite(float(discrepancy))
            and math.isfinite(float(sensitivity))
            and 0 <= float(discrepancy) <= float(sensitivity)
        )
    ):
        raise ValueError(
            "Saved same-seed initial prediction consistency check is missing or failed"
        )
    audit["initial_center"] = {
        "reference_seed_count": len(center["seeds"]),
        "evaluation_seed_count": expected_seeds,
        "distinct_candidate_count": len(bank),
        "center_vector_sha256": center["vector_sha256"],
        "same_seed_initial_consistency": "passed_saved_source_precision_screen",
        "maximum_prediction_discrepancy": discrepancy,
        "prediction_sensitivity": sensitivity,
    }
    register(
        "initial_unconditional_concentration",
        pd.concat([initial_ecdf, atom_ecdf], ignore_index=True),
        population_counts={
            "initial_unconditional": initial_counts,
            "candidate_atoms": atom_counts,
        },
        counts={
            "initial_unconditional": initial_counts,
            "candidate_atoms": atom_counts,
        },
        center_provenance=center,
        actual_initial_snr=float(
            trajectory.loc[
                trajectory.step_index == trajectory.step_index.min(), "snr"
            ].iloc[0]
        ),
        data_sources=[
            "reference_initial.parquet",
            "bank_geometry.parquet",
            "center_metadata.json",
        ],
    )

    curves = _curves(
        trajectory,
        {
            "conditional_target_error_rmse": "conditional",
            "unconditional_target_error_rmse": "unconditional",
            "branch_gap_rmse": "gap",
        },
    )
    register(
        "target_synchronization",
        curves.loc[curves.branch.isin(["conditional", "unconditional"])].reset_index(
            drop=True
        ),
        band_definition="Saved weighted interquartile range, descriptive dispersion; no confidence interval.",
        data_sources=["trajectory_metrics/*"],
    )
    register(
        "branch_gap",
        curves.loc[curves.branch == "gap"].reset_index(drop=True),
        band_definition="Weighted interquartile range, descriptive dispersion.",
        data_sources=["trajectory_metrics/*"],
    )

    within = summaries.get("within_prompt_initial")
    trend_table = summaries.get("within_prompt_initial_trend")
    if within is None or trend_table is None:
        raise ValueError(
            "IR03 requires saved within-prompt observations and saved descriptive trend"
        )
    trend_rows = trend_table.loc[trend_table.group.astype(str).str.lower() == "all"]
    if len(trend_rows) != 1:
        raise ValueError(
            "IR03 requires exactly one saved all-population weighted trend"
        )
    trend = trend_rows.iloc[0].to_dict()
    trend["status"] = (
        "available"
        if np.isfinite(float(trend.get("slope", np.nan)))
        and np.isfinite(float(trend.get("intercept", np.nan)))
        else "undefined_no_usable_seed_variation"
    )
    register(
        "initial_recovery_within_prompt",
        _scatter(within, "x_centered", "y_centered"),
        trend=trend,
        status_counts=within.within_prompt_status.value_counts().to_dict()
        if "within_prompt_status" in within
        else {},
        data_sources=[
            "summaries/within_prompt_initial.parquet",
            "summaries/within_prompt_initial_trend.parquet",
        ],
    )

    rank_tables, rank_counts = [], {}
    for group, rows in _groups(initial):
        for branch in ("conditional", "unconditional"):
            column = f"initial_{branch}_target_rank"
            _require(rows, [column], "initial Euclidean target retrieval")
            if (_numeric(rows, column).dropna() < 1).any():
                raise ValueError("Initial retrieval rank must be at least one")
            curve, counts = exact_weighted_ecdf(
                rows, column, group=group, branch=branch
            )
            counts["exact_tied_rows"] = int(
                (_numeric(rows, f"initial_{branch}_target_tie_count") > 1).sum()
            )
            rank_tables.append(curve)
            rank_counts[group + "/" + branch] = counts
    register(
        "initial_target_retrieval_rank",
        pd.concat(rank_tables, ignore_index=True),
        counts=rank_counts,
        bank_size=len(bank),
        tie_policy="One plus strictly closer exact distinct atoms; target-inclusive exact ties retained separately.",
        target_mapping_provenance="analysis_manifest.support_metadata aliases and canonical candidate_target_atom_id",
        data_sources=["initial.parquet"],
    )

    prediction_steps = sorted(trajectory.step_index.astype(int).unique())
    total = int(
        config.get("num_inference_steps", config.get("T", max(prediction_steps) + 1))
    )
    recorded = manifest.get("synchronization_snapshots", [])
    recorded = dict(recorded) if isinstance(recorded, list) else recorded
    if total == 50:
        early, late = 10, 48
    elif recorded and "early" in recorded and "late" in recorded:
        early, late = int(recorded["early"]), int(recorded["late"])
    else:
        early, late = min(max(0, math.floor(0.2 * total)), total - 1), max(0, total - 2)
    if early not in prediction_steps or late not in prediction_steps:
        raise ValueError(
            "Frozen joint-recovery snapshot missing; no data-dependent substitution"
        )
    paired = trajectory[_identity(trajectory) + ["terminal_sscd"]].copy()
    paired["joint_target_error_rmse"] = np.maximum(
        _numeric(trajectory, "conditional_target_error_rmse"),
        _numeric(trajectory, "unconditional_target_error_rmse"),
    )
    frozen = paired.loc[paired.step_index.isin([early, late])]
    joint_limits = _finite_limits(_numeric(frozen, "joint_target_error_rmse"))
    tolerance_rows = []
    for name, step in (("early", early), ("late", late)):
        source = paired.loc[paired.step_index == step]
        parts, counts = [], {}
        for group, rows in _groups(source):
            curve, count = exact_weighted_ecdf(
                rows, "joint_target_error_rmse", group=group
            )
            parts.append(curve)
            counts[group] = count
            for tolerance in RECOVERY_TOLERANCES:
                weights = prompt_weights(rows).to_numpy()
                value = _numeric(rows, "joint_target_error_rmse").to_numpy()
                denom = float(weights.sum())
                tolerance_rows.append(
                    {
                        "snapshot": name,
                        "step_index": step,
                        "group": group,
                        "tolerance": tolerance,
                        "fraction": float(
                            weights[np.isfinite(value) & (value <= tolerance)].sum()
                            / denom
                        )
                        if denom
                        else np.nan,
                        "unresolved_fraction": float(
                            weights[~np.isfinite(value)].sum() / denom
                        )
                        if denom
                        else np.nan,
                        "total_count": len(rows),
                    }
                )
        first = source.iloc[0]
        register(
            "joint_target_recovery_" + name,
            pd.concat(parts, ignore_index=True),
            status="alias" if name == "late" and early == late else "available",
            reason="Early and late deterministic snapshots coincide in this short run"
            if name == "late" and early == late
            else None,
            alias_of="joint_target_recovery_early"
            if name == "late" and early == late
            else None,
            counts=counts,
            x_limits=joint_limits,
            y_limits=[0.0, 1.0],
            snapshot={
                "step_index": step,
                "timestep": float(first.timestep),
                "snr": float(first.snr),
            },
            ecdf="Exact weighted support; per-sample max of the paired target errors.",
            data_sources=["trajectory_metrics/*"],
        )
    outputs["joint_recovery_tolerance_audit"] = pd.DataFrame(tolerance_rows)

    contract, terminal_audit = terminal_contract_status(endpoint)
    # The sampler's clean-update identity and the manuscript's g > 1 domain
    # are independent prerequisites. In particular, saved (g-1)||Delta|| is
    # signed outside this domain and must not be presented as a norm bound.
    guidance = float(config.get("guidance_scale", 7.5))
    guidance_domain = math.isfinite(guidance) and guidance > 1
    guidance_reason = "inapplicable_manuscript_guidance_domain_g_gt_1"
    verified = contract.paper_terminal_clean_verified.to_numpy()
    contract["paper_terminal_theorem_guidance_domain_satisfied"] = guidance_domain
    contract["paper_terminal_theorem_applicable"] = verified & guidance_domain
    terminal_audit["theorem_guidance_domain_satisfied"] = guidance_domain
    terminal_audit["theorem_guidance_domain_reason"] = (
        None if guidance_domain else guidance_reason
    )
    outputs["terminal_contract"] = contract
    audit["terminal"], audit["blocking"] = terminal_audit, terminal_audit["blocking"]
    terminal = endpoint.loc[verified & guidance_domain]
    failed = terminal_audit["blocking"]
    terminal_status = (
        "error" if failed else "available" if len(terminal) else "not_applicable"
    )
    reason = (
        "Missing contract data or failed independent reconstruction"
        if failed
        else guidance_reason
        if not guidance_domain
        else None
        if len(terminal)
        else "; ".join(sorted(set(contract.paper_terminal_reason)))
    )
    for stem in ("terminal_error_terms", "terminal_bound_tightness"):
        register(
            stem,
            pd.DataFrame(),
            status=terminal_status,
            reason=reason,
            status_counts=terminal_audit["status_counts"],
            data_sources=["endpoint.parquet"],
        )
    if len(terminal) and not failed:
        register(
            "terminal_error_terms",
            _scatter(terminal, "terminal_A_rmse", "terminal_B_rmse"),
            status_counts=terminal_audit["status_counts"],
            independent_latent_tolerance=config.get("target_error_tolerance"),
            latent_dimension=int(terminal.latent_dimension.iloc[0]),
            independent_tolerance_rmse=(
                float(config["target_error_tolerance"])
                / math.sqrt(float(terminal.latent_dimension.iloc[0]))
            )
            if config.get("target_error_tolerance") is not None
            else None,
            data_sources=["endpoint.parquet"],
        )
        error = _numeric(terminal, "terminal_rmse")
        if "terminal_rmse" not in terminal:
            error = _numeric(terminal, "terminal_endpoint_error_rmse")
        ratios = terminal[_identity(terminal) + ["terminal_sscd"]].copy()
        ratios["observable_triangle"] = (
            _numeric(terminal, "terminal_A_rmse")
            + _numeric(terminal, "terminal_B_rmse")
        ) / error.where(error > 0)
        candidate = _numeric(terminal, "candidate_terminal_bound_rmse")
        valid_candidate = _bool(
            terminal, "candidate_terminal_bound_applies_to_endpoint"
        ) & np.isfinite(candidate)
        ratios["candidate_reference"] = candidate.where(valid_candidate) / error.where(
            error > 0
        )
        parts, counts = [], {}
        for distribution in ("observable_triangle", "candidate_reference"):
            curve, count = exact_weighted_ecdf(
                ratios, distribution, distribution=distribution
            )
            count["undefined_zero_endpoint_error"] = int((error == 0).sum())
            parts.append(curve)
            counts[distribution] = count
        complete = bool(valid_candidate.all())
        register(
            "terminal_bound_tightness",
            pd.concat(parts, ignore_index=True),
            counts=counts,
            comparison_status="complete_two_bound_comparison"
            if complete
            else "incomplete_candidate_bound_comparison",
            reason=None
            if complete
            else "Some candidate-reference bound terms are unavailable; finite ratios and missing counts are explicit.",
            data_sources=["endpoint.parquet"],
        )

    repaired, reference_audit = repair_current_reference(trajectory)
    audit["current_reference"] = reference_audit
    columns = _identity(repaired) + [
        "paper_current_reference_error_u_l2",
        "current_unconditional_candidate_reference_error_rmse",
        "paper_current_reference_source",
        "paper_current_reference_status",
        "candidate_radius_tail_rmse",
        "paper_current_reference_tail_status",
    ]
    outputs["current_reference_status"] = repaired[columns].copy()
    if diagnostics:
        definitions = {
            "initial_candidate_reference_discrepancy": (
                baseline,
                "initial_candidate_reference_movement_rmse",
                "initial_unconditional_candidate_reference_error_rmse",
            ),
            "initial_injection_geometry": (initial, "a_parallel", "off_target"),
            "initial_target_coordinates": (
                initial,
                "initial_target_coordinate_c",
                "initial_target_coordinate_g",
            ),
            "initial_injection_mismatch": (
                initial,
                "terminal_sscd",
                "injection_relative_mismatch",
            ),
            "last_prediction_error_terms": (
                endpoint,
                "terminal_A_rmse",
                "terminal_B_rmse",
            ),
        }
        for stem, (rows, x, y) in definitions.items():
            if stem == "last_prediction_error_terms" and not guidance_domain:
                register(
                    stem,
                    pd.DataFrame(),
                    status="not_applicable",
                    reason=guidance_reason,
                    data_sources=["endpoint.parquet"],
                )
            elif x not in rows or y not in rows:
                register(
                    stem,
                    pd.DataFrame(),
                    status="error",
                    reason="Missing saved diagnostic fields",
                    population_counts=_counts(rows),
                    data_sources=[
                        "initial.parquet"
                        if rows is initial
                        else "endpoint.parquet"
                        if rows is endpoint
                        else "reference_initial.parquet"
                    ],
                )
            else:
                register(
                    stem,
                    _scatter(rows, x, y),
                    scope="last prediction geometry; not a terminal certificate"
                    if stem == "last_prediction_error_terms"
                    else "initial state",
                    population_counts=_counts(rows),
                    data_sources=[
                        "initial.parquet"
                        if rows is initial
                        else "endpoint.parquet"
                        if rows is endpoint
                        else "reference_initial.parquet"
                    ],
                )
        sweep = summaries.get("reference_sweep")
        register(
            "initial_reference_snr_sweep",
            pd.DataFrame() if sweep is None else sweep.copy(),
            status="error" if sweep is None else "available",
            reason="Missing saved analytical sweep" if sweep is None else None,
            population_counts=_counts(baseline),
            data_sources=["summaries/reference_sweep.parquet"],
        )
        components = _curves(
            repaired,
            {
                "branch_gap_rmse": "gap",
                "conditional_target_error_rmse": "conditional",
                "current_unconditional_candidate_reference_error_rmse": "unconditional_reference",
                "candidate_radius_tail_rmse": "candidate_radius_tail",
            },
            include_all=True,
        )
        register(
            "synchronization_bound_components",
            components,
            current_reference_audit=reference_audit,
            data_sources=["trajectory_metrics/*"],
        )
        replay = _curves(
            trajectory,
            {
                "independent_replay_rmse": "independent_residual",
                "update_rounding_sensitivity_rmse": "source_precision_envelope",
            },
            include_all=True,
        )
        available = np.isfinite(_numeric(trajectory, "independent_replay_rmse")).any()
        register(
            "matched_update_residual",
            replay,
            status="available" if available else "not_applicable",
            reason=None
            if available
            else "Independent replay unavailable without separately saved stochastic contribution",
            data_sources=["trajectory_metrics/*"],
        )
        if len(terminal) and not failed:
            cancellation = terminal.copy()
            cancellation["_ratio"] = error / (
                _numeric(terminal, "terminal_A_rmse")
                + _numeric(terminal, "terminal_B_rmse")
            ).where(lambda x: x > 0)
            register(
                "terminal_cancellation",
                _scatter(cancellation, "last_prediction_error_gap_cosine", "_ratio"),
                data_sources=["endpoint.parquet"],
            )
        else:
            register(
                "terminal_cancellation",
                pd.DataFrame(),
                status=terminal_status,
                reason=reason,
                data_sources=["endpoint.parquet"],
            )
    return outputs, {
        "figures": figures,
        "audit": audit,
        "auxiliary_tables": [
            "terminal_contract",
            "current_reference_status",
            "joint_recovery_tolerance_audit",
        ],
    }


def extend_terminal_accounting(
    endpoint,
    *,
    manifest,
    source_logs,
    device="cpu",
    progress=None,
    verify_payload_hashes=False,
):
    """Read only missing terminal-accounting inputs from an explicit log root.

    Large tensors are memory mapped and only the final current/next states and
    canonical predictions are cloned. Completion/run/schedule metadata hashes
    must match the already validated candidate manifest. Full payload hashing is
    optional and its scope is recorded; new slice hashes identify the exact
    observed inputs. Nothing is written to the source or candidate bundle.
    """
    from pathlib import Path
    import hashlib
    import torch
    from utils.common.io import file_sha256, read_json, safe_torch_load
    from utils.experiments.cache import (
        GenerationPaths,
        generation_log_relative_path,
        require_generation_run,
        validate_generation_record,
    )
    from .scheduler_adapter import SchedulerAdapter

    _require(
        endpoint,
        SAMPLE_KEYS + ["step_index", "destination_noise_std", "affine_applicable"],
        "terminal supplement population",
    )
    config = manifest["config"]
    args = {
        key: config[key]
        for key in (
            "model_name",
            "scheduler_name",
            "guidance_scale",
            "num_inference_steps",
            "num_seeds",
        )
    }
    if source_logs is None:
        raise ValueError(
            "Terminal supplement requires explicit source_logs; current logs are never a fallback"
        )
    requested = Path(source_logs).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError(
            "Terminal source_logs must be an existing nonsymlink directory"
        )
    root = requested.resolve()
    relative = generation_log_relative_path(**args, seed_start=0)
    paths = GenerationPaths(root / relative.relative_to("logs"))
    existing = endpoint.get(
        "terminal_accounting_status", pd.Series("", index=endpoint.index)
    ).fillna("")
    stable = existing.isin(
        [
            "available",
            "unavailable_saved_sampling_noise",
            "unsupported_nonaffine_update",
        ]
    )
    wanted = endpoint.loc[~stable]
    result = endpoint[
        list(SAMPLE_KEYS)
        + [c for c in endpoint if c.startswith("terminal_accounting_")]
    ].copy()
    if wanted.empty:
        return result, {
            "status": "reused_saved_accounting",
            "records_loaded": 0,
            "source_logs": str(root),
        }
    statuses = np.where(
        ~_bool(wanted, "affine_applicable"),
        "unsupported_nonaffine_update",
        np.where(
            _numeric(wanted, "destination_noise_std") != 0,
            "unavailable_saved_sampling_noise",
            "needs_deterministic_accounting",
        ),
    )
    additions = wanted[list(SAMPLE_KEYS)].copy()
    additions["terminal_accounting_status"] = statuses
    deterministic = wanted.loc[statuses == "needs_deterministic_accounting"]
    report = {
        "source_logs": str(root),
        "records_loaded": 0,
        "full_payload_hashes_verified": bool(verify_payload_hashes),
        "provenance_scope": "Saved candidate-manifest metadata hashes, source shapes/dtypes, immutable metadata/stat checks, and exact cloned terminal-slice hashes; full payload rehash only if explicitly enabled.",
        "records": [],
    }
    if not deterministic.empty:
        for parent in [paths.run_directory, *paths.run_directory.parents]:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError("Unsafe symlink in explicit terminal source path")
        source_hashes = manifest.get("source_metadata_files", {})

        def verify_metadata(path):
            logical = (Path("logs") / path.relative_to(root)).as_posix()
            expected = source_hashes.get(logical)
            if path == paths.schedule:
                expected = (
                    expected
                    or manifest.get("schedule_sha256")
                    or manifest.get("support_metadata", {}).get(
                        "source_schedule_sha256"
                    )
                )
            if not expected or path.is_symlink() or file_sha256(path) != expected:
                raise ValueError(
                    f"Terminal source metadata does not match saved analysis: {logical}"
                )
            return expected

        verify_metadata(paths.run_config)
        verify_metadata(paths.schedule)
        run = require_generation_run(paths)
        if set(deterministic.run_id.astype(str)) != {
            str(run["scientific_config_hash"])
        }:
            raise ValueError(
                "Terminal source run does not match saved scalar run identity"
            )
        science = run["scientific_config"]
        if science.get("stored_prediction_type") != "epsilon" or science.get(
            "seeds"
        ) != list(range(config["num_seeds"])):
            raise ValueError("Terminal source canonical-epsilon/seed contract differs")
        adapter = SchedulerAdapter(
            safe_torch_load(paths.schedule),
            recorded_diffusers_version=manifest.get("scheduler_adapter", {}).get(
                "recorded_diffusers_version"
            ),
        )
        for position, (original_index, rows) in enumerate(
            deterministic.groupby("original_index", sort=True), 1
        ):
            marker_path = paths.record_path(original_index)
            marker_hash = verify_metadata(marker_path)
            marker = read_json(marker_path)
            validation = validate_generation_record(
                paths,
                original_index,
                expected_scientific_hash=run["scientific_config_hash"],
                load_tensors=False,
                require_preview=False,
                verify_file_hashes=verify_payload_hashes,
            )
            if not validation.valid:
                raise ValueError(
                    f"Invalid terminal source record {original_index}: {validation.errors}"
                )
            tensor_paths = (
                paths.latent_path(original_index),
                paths.noise_prediction_path(original_index),
                paths.target_latent_path(original_index),
            )
            before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in tensor_paths]
            z = torch.load(
                tensor_paths[0], weights_only=True, map_location="cpu", mmap=True
            )
            predictions = torch.load(
                tensor_paths[1], weights_only=True, map_location="cpu", mmap=True
            )
            target = safe_torch_load(tensor_paths[2])
            if not isinstance(predictions, tuple) or len(predictions) != 2:
                raise ValueError(
                    "Terminal source requires stored canonical epsilon branch pair"
                )
            eu, ec = predictions
            n, steps = int(config["num_seeds"]), int(config["num_inference_steps"])
            if (
                z.ndim != 5
                or tuple(z.shape[:2]) != (n, steps + 1)
                or eu.shape != ec.shape
                or tuple(eu.shape) != (n, steps, *z.shape[2:])
                or tuple(target.shape) != tuple(z.shape[2:])
            ):
                raise ValueError(
                    "Terminal source prediction/state/target indexing mismatch"
                )
            for name, tensor in (
                ("latent", z),
                ("unconditional_noise_predictions", eu),
                ("conditional_noise_predictions", ec),
                ("target_latent", target),
            ):
                if (
                    list(tensor.shape) != marker["tensor_shapes"][name]
                    or str(tensor.dtype).removeprefix("torch.")
                    != marker["tensor_dtypes"][name]
                ):
                    raise ValueError(
                        "Terminal source tensor shape/dtype differs from pinned completion metadata"
                    )
            last = steps - 1
            if set(rows.step_index.astype(int)) != {last}:
                raise ValueError(
                    "Terminal accounting cannot synthesize predictions at the final output state"
                )
            tail = [
                z[:, last].clone(),
                z[:, last + 1].clone(),
                eu[:, last].clone(),
                ec[:, last].clone(),
                target.clone(),
            ]
            del z, eu, ec, target, predictions
            if not all(bool(torch.isfinite(x).all()) for x in tail):
                raise ValueError("Nonfinite terminal source tensors")
            slice_hashes = [
                hashlib.sha256(
                    x.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                for x in tail
            ]
            measured = terminal_affine_accounting(
                adapter,
                *[x.to(device) for x in tail],
                config["guidance_scale"],
                step=last,
            )
            seed_positions = {
                int(seed): index for index, seed in enumerate(marker["seeds"])
            }
            for index, row in rows.iterrows():
                if str(row.record_id) != str(marker.get("record_id")):
                    raise ValueError(
                        "Terminal source record identity differs from saved sample"
                    )
                seed_position = seed_positions[int(row.seed)]
                for key, value in measured.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().tolist()
                        value = (
                            value[seed_position] if isinstance(value, list) else value
                        )
                    additions.loc[index, key] = value
            after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in tensor_paths]
            if before != after or file_sha256(marker_path) != marker_hash:
                raise ValueError(
                    "Terminal source changed while its supplement was computed"
                )
            report["records"].append(
                {
                    "original_index": str(original_index),
                    "marker_sha256": marker_hash,
                    "terminal_slice_sha256": slice_hashes,
                    "rows": len(rows),
                }
            )
            report["records_loaded"] += 1
            if progress:
                progress(position, deterministic.original_index.nunique())
        verify_metadata(paths.run_config)
        verify_metadata(paths.schedule)
    if not result.empty:
        result = result.set_index(SAMPLE_KEYS)
        updates = additions.set_index(SAMPLE_KEYS)
        for column in updates:
            if column not in result:
                result[column] = pd.Series(
                    index=result.index, dtype=updates[column].dtype
                )
            result.loc[updates.index, column] = updates[column]
        result = result.reset_index()
    else:
        result = additions.reset_index(drop=True)
    report["status"] = "completed"
    report["status_counts"] = result.terminal_accounting_status.value_counts(
        dropna=False
    ).to_dict()
    return result, report
