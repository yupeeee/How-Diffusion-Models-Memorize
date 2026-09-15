"""Descriptive saved-scalar summaries; no tensors, model imports or inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

OUTCOME_THRESHOLD = 0.75
OUTCOME_GROUPS = ("SSCD > 0.75", "SSCD <= 0.75")
PROMPT_KEYS = ["run_id", "original_index", "record_id", "target_id"]
SAMPLE_KEYS = PROMPT_KEYS + ["seed"]
PHASES = ("early", "intermediate", "late")
SUMMARY_POLICY = {
    "aggregation": "Each represented prompt has equal total weight in each outcome group, divided among its member seeds; weighted quantiles use the inverse weighted empirical CDF.",
    "outcome_threshold": OUTCOME_THRESHOLD,
    "outcome_groups": list(OUTCOME_GROUPS),
    "phase_partition": "floor(3*k/T): early [0,1/3), intermediate [1/3,2/3), late [2/3,1); k is saved update index, T prediction count",
    "interpretation": "Descriptive summaries conditional on frozen prompt selection and the shared fixed seed bank; interquartile bands are not confidence intervals.",
}


def _finite(series):
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def quantiles(values, prefix=""):
    values = _finite(pd.Series(values))
    result = {prefix + "count": len(values)}
    result.update(
        {
            prefix + name: float(values.quantile(q)) if len(values) else None
            for name, q in (
                ("q10", 0.1),
                ("q25", 0.25),
                ("median", 0.5),
                ("q75", 0.75),
                ("q90", 0.9),
            )
        }
    )
    return result


def weighted_quantiles(values, weights, probabilities=(0.25, 0.5, 0.75)):
    """Inverse weighted ECDF, with deterministic value ordering and no fit."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values):
        return np.full(len(probabilities), np.nan)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return values[
        np.minimum(
            np.searchsorted(cumulative, probabilities, side="left"), len(values) - 1
        )
    ]


def outcome_groups(frame):
    """Return the prespecified cohorts without changing scientific eligibility."""
    values = pd.to_numeric(frame.terminal_sscd, errors="coerce")
    return (
        (
            OUTCOME_GROUPS[0],
            frame.loc[np.isfinite(values) & (values > OUTCOME_THRESHOLD)],
        ),
        (
            OUTCOME_GROUPS[1],
            frame.loc[np.isfinite(values) & (values <= OUTCOME_THRESHOLD)],
        ),
    )


def _prompt_weights(rows):
    keys = [key for key in PROMPT_KEYS if key in rows]
    return 1.0 / rows.groupby(keys, dropna=False)["seed"].transform("count")


def synchronization_summary(frame):
    """Four branch/cohort curves, with each prompt equally weighted per step."""
    columns = [
        "group",
        "branch",
        "step_index",
        "snr",
        "error_median",
        "error_q25",
        "error_q75",
        "prompt_count",
        "seed_count",
        "n_observations",
    ]
    output = []
    for group, cohort in outcome_groups(frame):
        for branch in ("conditional", "unconditional"):
            column = branch + "_target_error_rmse"
            for (step, snr), rows in cohort.groupby(["step_index", "snr"], sort=True):
                valid = rows.loc[
                    np.isfinite(pd.to_numeric(rows[column], errors="coerce"))
                ]
                if valid.empty:
                    continue
                lo, median, hi = weighted_quantiles(
                    valid[column], _prompt_weights(valid)
                )
                output.append(
                    dict(
                        group=group,
                        branch=branch,
                        step_index=int(step),
                        snr=float(snr),
                        error_median=median,
                        error_q25=lo,
                        error_q75=hi,
                        prompt_count=len(
                            valid[
                                [k for k in PROMPT_KEYS if k in valid]
                            ].drop_duplicates()
                        ),
                        seed_count=len(
                            valid[
                                [k for k in SAMPLE_KEYS if k in valid]
                            ].drop_duplicates()
                        ),
                        n_observations=len(valid),
                    )
                )
    return pd.DataFrame(output, columns=columns)


def paired_synchronization_summary(frame):
    output = []
    metrics = (
        "conditional_target_error_rmse",
        "unconditional_target_error_rmse",
        "joint_target_error_rmse",
        "branch_gap_rmse",
        "paired_target_error_improvement_rmse",
    )
    for group, cohort in outcome_groups(frame):
        for step, rows in cohort.groupby("step_index", sort=True):
            item = dict(
                group=group,
                step_index=int(step),
                snr=float(rows.snr.iloc[0]),
                n_observations=len(rows),
                prompt_count=len(
                    rows[[key for key in PROMPT_KEYS if key in rows]].drop_duplicates()
                ),
            )
            for metric in metrics:
                valid = rows.loc[
                    np.isfinite(pd.to_numeric(rows[metric], errors="coerce"))
                ]
                q25, median, q75 = (
                    weighted_quantiles(valid[metric], _prompt_weights(valid))
                    if len(valid)
                    else (np.nan,) * 3
                )
                item.update(
                    {
                        metric + "_" + name: value
                        for name, value in zip(
                            ("q25", "median", "q75"), (q25, median, q75), strict=True
                        )
                    }
                )
                item[metric + "_count"] = len(valid)
            output.append(item)
    return pd.DataFrame(output)


def phase_summary(frame, steps):
    """Aggregate a fixed step-fraction partition; never choose data-driven cuts."""
    frame = frame.copy()
    frame["phase"] = [
        PHASES[min(2, int(3 * int(k) // steps))] for k in frame.step_index
    ]
    output = []
    for group, cohort in outcome_groups(frame):
        for phase in PHASES:
            rows = cohort.loc[cohort.phase.eq(phase)]
            item = dict(
                group=group,
                phase=phase,
                n_observations=len(rows),
                prompt_count=len(
                    rows[[key for key in PROMPT_KEYS if key in rows]].drop_duplicates()
                ),
                sample_count=len(rows[SAMPLE_KEYS].drop_duplicates()),
            )
            for metric in (
                "conditional_target_error_rmse",
                "unconditional_target_error_rmse",
                "joint_target_error_rmse",
                "paired_target_error_improvement_rmse",
                "branch_gap_rmse",
            ):
                # Each prompt has equal total mass across its phase observations.
                valid = rows.loc[
                    np.isfinite(pd.to_numeric(rows[metric], errors="coerce"))
                ]
                q25, median, q75 = (
                    weighted_quantiles(valid[metric], _prompt_weights(valid))
                    if len(valid)
                    else (np.nan,) * 3
                )
                item.update(
                    {
                        metric + "_" + name: value
                        for name, value in zip(
                            ("q25", "median", "q75"), (q25, median, q75), strict=True
                        )
                    }
                )
            output.append(item)
    return pd.DataFrame(output)


def initial_prompt_summary(initial):
    output = []
    for identity, rows in initial.groupby(PROMPT_KEYS, sort=True, dropna=False):
        item = dict(zip(PROMPT_KEYS, identity, strict=True))
        improvement = _finite(rows.paired_target_error_improvement_rmse)
        item.update(
            sample_count=len(rows),
            improvement_denominator=len(improvement),
            improved_count=int((improvement > 0).sum()),
            fraction_improved=(
                float((improvement > 0).mean()) if len(improvement) else None
            ),
        )
        for metric in (
            "conditional_target_error_rmse",
            "unconditional_target_error_rmse",
            "paired_target_error_improvement_rmse",
            "relative_target_error_improvement",
        ):
            item.update(quantiles(rows[metric], metric + "_"))
        output.append(item)
    return pd.DataFrame(output)


def feedback_counts(frame):
    eligible = frame.feedback_eligible.fillna(False).astype(bool)
    statuses = frame.candidate_condition_status.fillna("unavailable")
    met = eligible & statuses.eq("estimated_met")
    unresolved = eligible & statuses.eq("numerically_unresolved")
    gain = frame.candidate_gain_status.fillna("unavailable")
    finite = np.isfinite(
        frame[
            ["candidate_condition_margin_rmse", "candidate_log_probability_gain"]
        ].apply(pd.to_numeric, errors="coerce")
    ).all(axis=1)
    result = dict(
        total_count=len(frame),
        eligible_count=int(eligible.sum()),
        condition_met_count=int(met.sum()),
        condition_not_met_count=int(
            (eligible & statuses.eq("estimated_not_met")).sum()
        ),
        condition_unresolved_count=int(unresolved.sum()),
        unplottable_count=int((~finite).sum()),
        condition_coverage=(
            float(met.sum() / eligible.sum()) if eligible.any() else None
        ),
        condition_coverage_kind="numerically estimated; unresolved rows retained in eligible denominator",
    )
    for sign in (
        "positive",
        "negative",
        "zero",
        "numerically_unresolved",
        "unavailable",
    ):
        result["gain_" + sign + "_count"] = int((eligible & gain.eq(sign)).sum())
        result["condition_met_gain_" + sign + "_count"] = int(
            (met & gain.eq(sign)).sum()
        )
    for flag in ("candidate_gain_saturated", "candidate_gain_underflow"):
        values = (
            frame[flag].fillna(False).astype(bool)
            if flag in frame
            else pd.Series(False, index=frame.index)
        )
        result[flag + "_count"] = int((eligible & values).sum())
        result[flag + "_fraction"] = (
            float((eligible & values).sum() / eligible.sum())
            if eligible.any()
            else None
        )
    return result


def feedback_summary(frame, grouping=("step_index",)):
    output = []
    metric_columns = (
        "candidate_conditional_reference_error_l2",
        "candidate_unconditional_reference_error_l2",
        "branch_gap_l2",
        "candidate_variation_l2",
        "candidate_variation_error_l2",
        "candidate_condition_margin_l2",
        "candidate_log_probability_gain",
        "candidate_log_odds_gain",
        "candidate_matched_log_probability",
        "candidate_guided_log_probability",
        "candidate_matched_log_complement",
        "candidate_guided_log_complement",
    )
    for keys, rows in frame.groupby(list(grouping), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = dict(zip(grouping, keys, strict=True))
        item.update(feedback_counts(rows))
        item["source_snr"] = (
            float(rows.snr.iloc[0]) if rows.snr.nunique() == 1 else None
        )
        item["destination_snr"] = (
            float(rows.destination_snr.iloc[0])
            if rows.destination_snr.nunique() == 1
            else None
        )
        item["high_sscd_negative_gain_count"] = int(
            (
                (rows.terminal_sscd > OUTCOME_THRESHOLD)
                & rows.candidate_gain_status.eq("negative")
            ).sum()
        )
        for column in metric_columns:
            if column in rows:
                item.update(quantiles(rows[column], column + "_"))
        # Complement probabilities come from their own saved stable logs, never
        # from 1 - a rounded target probability. exp(-inf) is the valid value 0.
        # Exponentiation underflow remains zero; gain-sign flags stay separate.
        for endpoint in ("matched", "guided"):
            for field in ("probability", "complement"):
                column = f"candidate_{endpoint}_log_{field}"
                if column not in rows:
                    continue
                logs = pd.to_numeric(rows[column], errors="coerce")
                valid_logs = logs.where(logs <= 0)
                with np.errstate(under="ignore"):
                    probabilities = np.exp(valid_logs)
                item.update(quantiles(probabilities, f"candidate_{endpoint}_{field}_"))
        output.append(item)
    return pd.DataFrame(output)


def tolerance_entry_summary(frame, raw_l2_tolerance):
    """Observed-grid entries with explicit non-entry and right-censoring status."""
    if raw_l2_tolerance is None:
        return pd.DataFrame()
    tolerance = float(raw_l2_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "Target-error tolerance must be a finite nonnegative raw L2 value"
        )
    output = []
    for identity, rows in frame.groupby(SAMPLE_KEYS, sort=True):
        rows = rows.sort_values("step_index")
        values = rows.joint_target_error_rmse.to_numpy(dtype=float)
        threshold = tolerance / math.sqrt(int(rows.latent_dimension.iloc[0]))
        valid = np.isfinite(values)
        inside = valid & (values <= threshold)
        first = np.flatnonzero(inside)
        sustained = np.flatnonzero(np.logical_and.accumulate(inside[::-1])[::-1])
        item = dict(zip(SAMPLE_KEYS, identity, strict=True))
        item.update(
            raw_l2_tolerance=tolerance,
            rmse_tolerance=threshold,
            first_entry_step=(
                int(rows.step_index.iloc[first[0]]) if len(first) else None
            ),
            first_entry_status="reached" if len(first) else "not_reached",
            sustained_entry_step=(
                int(rows.step_index.iloc[sustained[0]]) if len(sustained) else None
            ),
            sustained_entry_status=(
                "observed_suffix_only" if len(sustained) else "not_reached"
            ),
            right_censored=True,
            last_observed_prediction_step=int(rows.step_index.iloc[-1]),
            missing_observation_count=int((~valid).sum()),
        )
        output.append(item)
    return pd.DataFrame(output)


def applicability_report(
    initial, trajectory, endpoint, center_metadata, support_metadata
):
    structural = endpoint.terminal_clean_structural.fillna(False).astype(bool)
    applicable = endpoint.terminal_clean_applicable.fillna(False).astype(bool)
    numeric = endpoint.terminal_clean_numeric_within_sensitivity.fillna(False).astype(
        bool
    )
    feedback = feedback_counts(trajectory)
    if feedback["eligible_count"]:
        feedback_status = (
            "available"
            if feedback["unplottable_count"] < len(trajectory)
            else "numerically_unresolved"
        )
    else:
        feedback_status = (
            "not_applicable"
            if trajectory.candidate_condition_status.eq("not_applicable").all()
            else "unavailable"
        )
    terminal_status = (
        "applicable"
        if applicable.any()
        else "numerical_inconsistency"
        if structural.any()
        else "not_applicable"
    )
    return {
        "training_reference": {
            "status": "unavailable",
            "reason": "Preserved caches do not identify the training law, its mean/posterior/support radius, forward loss, or the single-target training assumption.",
        },
        "center": {
            "status": "model_center_diagnostic",
            "kind": center_metadata.get("center_kind", center_metadata.get("center")),
            "unique_reference_inputs": center_metadata.get("source_sample_count"),
            "unique_evaluation_inputs": int(initial.seed.nunique()),
        },
        "candidate_reference": {
            "status": "candidate_distribution_diagnostic",
            "support_size": support_metadata["support_size"],
            "policy": support_metadata.get("bank_policy"),
            "training_frequencies_identified": False,
        },
        "posterior_feedback": feedback
        | {
            "status": feedback_status,
            "condition_status_kind": "estimated",
            "unavailability_reasons": trajectory.feedback_status.fillna("missing")
            .value_counts()
            .to_dict(),
        },
        "terminal_clean_update": {
            "status": terminal_status,
            "applicable_count": int(applicable.sum()),
            "structural_applicable_count": int(structural.sum()),
            "numerical_check_failed_count": int((structural & ~numeric).sum()),
            "total_count": len(endpoint),
            "reasons": endpoint.terminal_clean_status.fillna("missing")
            .value_counts()
            .to_dict(),
            "scope": "Structural Eq.5 conditions precede the numerical residual check; last clean-estimate terms remain diagnostics when false.",
        },
        "matched_displacement": {
            "status": "algebraic_qa",
            "figure": None,
            "construction_is_independent_validation": False,
        },
        "tolerance": {
            "status": "not_supplied",
            "reason": "No latent tolerance is inferred from SSCD or observed outcomes.",
        },
    }
