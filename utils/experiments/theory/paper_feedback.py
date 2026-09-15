"""Paper feedback summaries and checks from saved scalars, without tensor loaders.

Arithmetic resolution, unknown source-input sensitivity, and estimated integration
uncertainty are distinct. Numerical missingness never changes a fraction's
structurally eligible, prompt-balanced denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .summaries import PROMPT_KEYS, SAMPLE_KEYS, weighted_quantiles

FORMULA_VERSION = "paper-feedback-1"
GROUPS = ("SSCD > 0.75", "SSCD <= 0.75")
ROW_KEYS = [*SAMPLE_KEYS, "step_index"]
MARGINS = ("original", "combined", "signed_error", "projected_variation")
STRUCTURAL_REASONS = {
    "missing_candidate_target",
    "no_distinct_competitor",
    "terminal_or_nonaffine_or_nonpositive_destination",
    "invalid_or_nonpositive_noise_guidance_kappa",
}
STEMS = {
    "PF02": "posterior_feedback_normalized_gain",
    "PF03": "posterior_feedback_positive_fraction",
    "PF04": "posterior_feedback_initial_comparison",
    "PF06": "posterior_feedback_initial_dose",
    "PF07": "posterior_feedback_coverage_audit",
    "PF08": "posterior_feedback_increasing_profile_fraction",
}


def _numbers(frame, column):
    if column not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _strings(frame, column, default="numerically_unresolved"):
    if column not in frame:
        return np.full(len(frame), default, dtype=object)
    return frame[column].fillna(default).astype(str).to_numpy()


def _require(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError("Missing saved feedback fields: " + ", ".join(sorted(missing)))


def structural_feedback_eligibility(frame):
    """Exclude known structural cases; retain failed/nonfinite measurements."""
    _require(frame, ["candidate_feedback_eligible", "candidate_feedback_status"])
    reasons = _strings(frame, "candidate_feedback_status")
    excluded = np.isin(reasons, list(STRUCTURAL_REASONS))
    for name in ("destination_sigma", "destination_alpha", "kappa"):
        if name in frame:
            values = _numbers(frame, name)
            excluded |= np.isfinite(values) & (values <= 0)
    reported = frame.candidate_feedback_eligible.fillna(False).to_numpy(dtype=bool)
    unknown = (
        ~reported
        & ~excluded
        & ~np.isin(
            reasons,
            [
                "invalid_nonfinite_input",
                "numerically_unresolved",
                "matched_displacement_QA_failure",
            ],
        )
    )
    if unknown.any():
        raise ValueError(
            "Unclassified feedback applicability reasons: "
            + ", ".join(sorted(set(reasons[unknown])))
        )
    return pd.Series(~excluded, index=frame.index, dtype=bool)


def prompt_balanced_weights(frame):
    """Equal prompt mass at each step, then equal eligible seed mass."""
    _require(frame, ROW_KEYS)
    if frame[ROW_KEYS].isna().any().any() or frame.duplicated(ROW_KEYS).any():
        raise ValueError("Missing or duplicate complete feedback sample identity")
    if frame.empty:
        return np.empty(0, dtype=float)
    counts = frame.groupby([*PROMPT_KEYS, "step_index"], dropna=False)[
        "seed"
    ].transform("nunique")
    return 1.0 / counts.to_numpy(dtype=float)


def _gain_classes(frame):
    gain = _numbers(frame, "candidate_log_odds_gain")
    tol = _numbers(frame, "candidate_gain_arithmetic_tolerance")
    status = _strings(frame, "candidate_gain_status")
    result = np.full(len(frame), "unresolved", dtype=object)
    valid = np.isfinite(gain) & np.isfinite(tol) & (tol >= 0)
    result[valid & (gain > tol) & (status == "positive")] = "positive"
    result[valid & (gain < -tol) & (status == "negative")] = "negative"
    result[valid & (gain == 0) & np.isin(status, ["arithmetic_zero", "zero"])] = "zero"
    return result


def _profile_classes(frame):
    c0 = _numbers(frame, "candidate_endpoint_slope_C0")
    c1 = _numbers(frame, "candidate_endpoint_slope_C1")
    tol = _numbers(frame, "candidate_endpoint_arithmetic_tolerance")
    recorded = _strings(frame, "candidate_profile_class")
    result = np.full(len(frame), "unresolved", dtype=object)
    valid = np.isfinite(c0) & np.isfinite(c1) & np.isfinite(tol) & (tol >= 0)
    valid &= (
        _strings(frame, "candidate_endpoint_bracket_status")
        != "numerical_inconsistency"
    )
    valid &= (
        _strings(frame, "candidate_feedback_status")
        != "matched_displacement_QA_failure"
    )
    result[valid & (c1 > tol)] = "increasing"
    result[valid & (c0 < -tol)] = "decreasing"
    result[valid & (c0 > tol) & (c1 < -tol)] = "interior_peak"
    result[valid & (c0 == 0) & (c1 == 0) & (recorded == "flat")] = "flat"
    return result


def audit_feedback_rows(frame):
    """Return JSON-ready audit and every offending ID, without clipping rows.

    The saved log-odds tolerance propagates through sigmoid to bound H error;
    a relative offset term and float64/subnormal allowance are also included.
    Source-input and quadrature errors cannot excuse contradictions between the
    same stored endpoints. Only explicitly certified margin claims are blockers;
    unresolved numerical integration does not block endpoint figures.
    """
    _require(frame, ROW_KEYS)
    if frame[ROW_KEYS].isna().any().any() or frame.duplicated(ROW_KEYS).any():
        raise ValueError("Missing or duplicate complete feedback sample identity")
    eligible = structural_feedback_eligibility(frame).to_numpy()
    required = [
        "candidate_log_probability_gain",
        "candidate_log_odds_gain",
        "candidate_gain_arithmetic_tolerance",
        "candidate_endpoint_slope_C0",
        "candidate_endpoint_slope_C1",
        "candidate_endpoint_arithmetic_tolerance",
        "candidate_matched_log_odds",
        "candidate_guided_log_odds",
        "candidate_gain_status",
        "candidate_profile_class",
    ]
    _require(frame, required)
    h, g, gt, c0, c1, ct, l0, l1 = (_numbers(frame, name) for name in required[:8])
    finite = eligible & np.isfinite(g) & np.isfinite(gt) & (gt >= 0)
    slope_valid = (
        finite & np.isfinite(c0) & np.isfinite(c1) & np.isfinite(ct) & (ct >= 0)
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        derivative = np.exp(-np.logaddexp(0.0, np.minimum(l0, l1)))
        h_budget = gt * derivative + np.abs(h) * (gt + 128 * np.finfo(float).eps)
    h_budget += 8 * np.nextafter(0.0, 1.0)
    h_reliable = np.isfinite(h) & np.isfinite(h_budget) & (np.abs(h) > h_budget)
    g_reliable = finite & (np.abs(g) > gt)
    offenders = []

    def finding(mask, reason):
        positions = np.flatnonzero(mask)
        if not len(positions):
            return
        selected = frame.iloc[positions][ROW_KEYS].copy()
        selected["reason"], selected["blocking"] = reason, True
        for name, values in {
            "H": h,
            "G": g,
            "C0": c0,
            "C1": c1,
            "gain_arithmetic_allowance": gt,
            "endpoint_arithmetic_allowance": ct,
            "H_arithmetic_allowance": h_budget,
        }.items():
            selected[name] = values[positions]
        offenders.append(selected)

    finding(
        g_reliable & h_reliable & (np.sign(h) != np.sign(g)),
        "reliable_H_G_sign_disagreement",
    )
    odds_allowance = gt + 64 * np.finfo(float).eps * (np.abs(l0) + np.abs(l1))
    finding(
        finite
        & np.isfinite(l0)
        & np.isfinite(l1)
        & (np.abs((l1 - l0) - g) > odds_allowance),
        "G_disagrees_with_saved_endpoint_log_odds",
    )
    if all(
        name in frame
        for name in (
            "candidate_matched_log_probability",
            "candidate_guided_log_probability",
        )
    ):
        p0 = _numbers(frame, "candidate_matched_log_probability")
        p1 = _numbers(frame, "candidate_guided_log_probability")
        logp_allowance = h_budget + 64 * np.finfo(float).eps * (np.abs(p0) + np.abs(p1))
        finding(
            eligible
            & np.isfinite(p0)
            & np.isfinite(p1)
            & np.isfinite(h)
            & (np.abs((p1 - p0) - h) > logp_allowance),
            "H_disagrees_with_saved_endpoint_log_probabilities",
        )
    finding(slope_valid & (c1 - g > ct), "C1_exceeds_G_outside_arithmetic_allowance")
    finding(slope_valid & (g - c0 > ct), "G_exceeds_C0_outside_arithmetic_allowance")
    finding(slope_valid & (c1 - c0 > ct), "C1_exceeds_C0_outside_arithmetic_allowance")
    reported = _strings(frame, "candidate_gain_status")
    finding(
        g_reliable
        & (((reported == "positive") & (g < 0)) | ((reported == "negative") & (g > 0))),
        "saved_resolved_gain_sign_disagrees_with_G",
    )
    finding(eligible & (gt < 0), "negative_saved_gain_arithmetic_allowance")
    finding(eligible & (ct < 0), "negative_saved_endpoint_arithmetic_allowance")
    if "terminal_sscd" in frame:
        changed = (
            frame.groupby(SAMPLE_KEYS, dropna=False).terminal_sscd.transform("nunique")
            > 1
        )
        finding(changed.to_numpy(), "same_seed_SSCD_changes_across_steps")
    for name in MARGINS:
        prefix = "candidate_margin_" + name
        value, error = (
            _numbers(frame, prefix + suffix) for suffix in ("_l2", "_uncertainty_l2")
        )
        if prefix + "_certified" in frame:
            certified = frame[prefix + "_certified"].fillna(False).to_numpy(dtype=bool)
            finding(
                eligible
                & certified
                & np.isfinite(error)
                & (error >= 0)
                & (value > error)
                & (g < -gt),
                "certified_strict_" + name + "_contradicts_negative_gain",
            )
    columns = [
        *ROW_KEYS,
        "reason",
        "blocking",
        "H",
        "G",
        "C0",
        "C1",
        "gain_arithmetic_allowance",
        "endpoint_arithmetic_allowance",
        "H_arithmetic_allowance",
    ]
    table = (
        pd.concat(offenders, ignore_index=True)
        if offenders
        else pd.DataFrame(columns=columns)
    )
    return {
        "status": "blocked" if len(table) else "passed",
        "formula_version": FORMULA_VERSION,
        "population_count": len(frame),
        "structurally_eligible_count": int(eligible.sum()),
        "structurally_excluded_count": int((~eligible).sum()),
        "blocking_finding_count": len(table),
        "offending_row_count": len(table[ROW_KEYS].drop_duplicates()),
        "reasons": table.reason.value_counts().to_dict(),
        "reliable_H_G_pairs": int((g_reliable & h_reliable).sum()),
        "retained_H_underflow_with_resolved_G": int((g_reliable & (h == 0)).sum()),
        "endpoint_allowance": "Saved float64 gain/slope allowances; H propagates gain tolerance through sigmoid plus relative/subnormal roundoff. No source-input or quadrature allowance is added.",
        "integration_interpretation": "Resolved margin estimates retain recorded uncertainty; they are not certified quadrature enclosures.",
    }, table


def _groups(frame, *, include_all=False):
    if include_all:
        yield "All", frame
    score = _numbers(frame, "terminal_sscd")
    yield GROUPS[0], frame.loc[np.isfinite(score) & (score > 0.75)]
    yield GROUPS[1], frame.loc[np.isfinite(score) & (score <= 0.75)]


def _population(rows):
    eligible = structural_feedback_eligibility(rows).to_numpy()
    selected, excluded = rows.loc[eligible], rows.loc[~eligible]
    weights, all_weights = (
        prompt_balanced_weights(selected),
        prompt_balanced_weights(rows),
    )
    snrs = pd.unique(_numbers(rows, "snr"))
    finite_snr = snrs[np.isfinite(snrs)]
    if len(finite_snr) > 1:
        raise ValueError("One paper curve step mixes distinct SNR values")
    return (
        selected,
        weights,
        {
            "snr": float(finite_snr[0]) if len(finite_snr) else np.nan,
            "population_count": len(rows),
            "eligible_count": len(selected),
            "excluded_count": len(excluded),
            "denominator_weight": float(weights.sum()),
            "excluded_weight": float(all_weights[~eligible].sum()),
            "eligible_prompt_count": len(selected[PROMPT_KEYS].drop_duplicates()),
            "eligible_sample_count": len(selected[SAMPLE_KEYS].drop_duplicates()),
            "excluded_prompt_count": len(excluded[PROMPT_KEYS].drop_duplicates()),
            "excluded_sample_count": len(excluded[SAMPLE_KEYS].drop_duplicates()),
        },
    )


def _mass_fields(classes, weights, names, denominator):
    result = {}
    for name in names:
        mask = classes == name
        result[name + "_count"] = int(mask.sum())
        result[name + "_weight"] = float(weights[mask].sum())
        result[name + "_fraction"] = (
            result[name + "_weight"] / denominator if denominator else np.nan
        )
    return result


def fraction_summaries(frame):
    """PF03 and PF08 share identical eligible rows and weights at every step."""
    gain_records, profile_records = [], []
    for group, subset in _groups(frame):
        for step, rows in subset.groupby("step_index", sort=True):
            selected, weights, base = _population(rows)
            denom = base["denominator_weight"]
            gain = _mass_fields(
                _gain_classes(selected),
                weights,
                ("positive", "negative", "zero", "unresolved"),
                denom,
            )
            profile = _mass_fields(
                _profile_classes(selected),
                weights,
                ("increasing", "decreasing", "interior_peak", "flat", "unresolved"),
                denom,
            )
            common = {"group": group, "step_index": int(step), **base}
            gain_records.append(
                {
                    **common,
                    **gain,
                    "fraction": gain["positive_fraction"],
                    "upper_fraction": gain["positive_fraction"]
                    + gain["unresolved_fraction"],
                }
            )
            profile.update(
                {
                    "positive_count": profile["increasing_count"],
                    "positive_weight": profile["increasing_weight"],
                    "negative_count": profile["decreasing_count"]
                    + profile["interior_peak_count"],
                    "negative_weight": profile["decreasing_weight"]
                    + profile["interior_peak_weight"],
                    "zero_count": profile["flat_count"],
                    "zero_weight": profile["flat_weight"],
                }
            )
            profile_records.append(
                {
                    **common,
                    **profile,
                    "fraction": profile["increasing_fraction"],
                    "upper_fraction": profile["increasing_fraction"]
                    + profile["unresolved_fraction"],
                }
            )
    return pd.DataFrame(gain_records), pd.DataFrame(profile_records)


def margin_coverage_audit(frame):
    """Compare raw and uncertainty-qualified margins on the same population.

    Raw floating point positivity is retained even when stable gain is negative;
    neither estimates nor estimated uncertainty envelopes are formal certificates.
    """
    records, findings = [], []
    for group, subset in _groups(frame, include_all=True):
        for step, rows in subset.groupby("step_index", sort=True):
            selected, weights, base = _population(rows)
            denominator, gain = base["denominator_weight"], _gain_classes(selected)
            for name in MARGINS:
                prefix = "candidate_margin_" + name
                value, error = (
                    _numbers(selected, prefix + suffix)
                    for suffix in ("_l2", "_uncertainty_l2")
                )
                status, finite = (
                    _strings(selected, prefix + "_status"),
                    np.isfinite(value),
                )
                masks = {
                    "estimated_strict_positive": finite & (value > 0),
                    "estimated_nonnegative": finite & (value >= 0),
                    "resolved_positive": finite
                    & np.isfinite(error)
                    & (error >= 0)
                    & (value > error)
                    & (status == "positive"),
                    "resolved_negative": finite
                    & np.isfinite(error)
                    & (error >= 0)
                    & (value < -error)
                    & (status == "negative"),
                    "stable_gain_positive": gain == "positive",
                    "stable_gain_nonnegative": np.isin(gain, ["positive", "zero"]),
                    "stable_gain_negative": gain == "negative",
                    "gain_unresolved": gain == "unresolved",
                    "zero_direction": _strings(
                        selected, "candidate_direction_normalization_status"
                    )
                    == "zero_displacement",
                    "arithmetic_unresolved_direction": _strings(
                        selected, "candidate_direction_normalization_status"
                    )
                    == "arithmetic_unresolved",
                    "margin_not_applicable": status == "not_applicable",
                }
                masks["margin_unresolved"] = ~(
                    masks["resolved_positive"] | masks["resolved_negative"]
                )
                masks["estimated_positive_negative_gain"] = (
                    masks["estimated_strict_positive"] & masks["stable_gain_negative"]
                )
                masks["resolved_positive_negative_gain"] = (
                    masks["resolved_positive"] & masks["stable_gain_negative"]
                )
                record = {
                    "group": group,
                    "step_index": int(step),
                    **base,
                    "margin": name,
                    "formal_certificate": False,
                }
                for label, mask in masks.items():
                    mass = float(weights[mask].sum())
                    record[label + "_count"], record[label + "_weight"] = (
                        int(mask.sum()),
                        mass,
                    )
                    record[label + "_fraction"] = (
                        mass / denominator if denominator else np.nan
                    )
                record["estimated_minus_stable_positive_fraction"] = (
                    record["estimated_strict_positive_fraction"]
                    - record["stable_gain_positive_fraction"]
                )
                records.append(record)
                bad = (
                    masks["estimated_positive_negative_gain"]
                    | masks["resolved_positive_negative_gain"]
                )
                if bad.any():
                    details = selected.loc[bad, ROW_KEYS].copy()
                    for key, val in {
                        "group": group,
                        "margin": name,
                        "margin_l2": value[bad],
                        "margin_uncertainty_l2": error[bad],
                        "margin_status": status[bad],
                        "sample_weight": weights[bad],
                        "resolved_margin_contradiction": masks[
                            "resolved_positive_negative_gain"
                        ][bad],
                    }.items():
                        details[key] = val
                    for column in (
                        "candidate_log_odds_gain",
                        "candidate_log_probability_gain",
                        "candidate_gain_status",
                        "candidate_gain_saturated",
                        "candidate_gain_underflow",
                        "candidate_direction_input_precision_status",
                        "candidate_integration_status",
                    ):
                        if column in selected:
                            details[column] = selected.loc[bad, column].to_numpy()
                    findings.append(details)
    columns = [
        *ROW_KEYS,
        "group",
        "margin",
        "margin_l2",
        "margin_uncertainty_l2",
        "margin_status",
        "sample_weight",
        "resolved_margin_contradiction",
    ]
    return pd.DataFrame(records), pd.concat(
        findings, ignore_index=True
    ) if findings else pd.DataFrame(columns=columns)


def _normalized_summary(frame):
    records = []
    for group, subset in _groups(frame):
        for step, rows in subset.groupby("step_index", sort=True):
            selected, weights, base = _population(rows)
            values = _numbers(selected, "candidate_normalized_log_odds_gain")
            finite = np.isfinite(values)
            q25, median, q75 = weighted_quantiles(
                values[finite], weights[finite], (0.25, 0.5, 0.75)
            )
            records.append(
                {
                    "group": group,
                    "step_index": int(step),
                    **base,
                    "q25": float(q25),
                    "median": float(median),
                    "q75": float(q75),
                    "minimum": float(values[finite].min()) if finite.any() else np.nan,
                    "maximum": float(values[finite].max()) if finite.any() else np.nan,
                    "finite_count": int(finite.sum()),
                    "unresolved_count": int((~finite).sum()),
                    "finite_weight": float(weights[finite].sum()),
                    "unresolved_weight": float(weights[~finite].sum()),
                }
            )
    return pd.DataFrame(records)


def build_feedback_plot_inputs(tables, *, config):
    """Return (stem-to-DataFrame, metadata); the caller publishes audit files."""
    frame = tables.get("trajectory", pd.DataFrame())
    if frame.empty:
        return {}, {
            "figures": {},
            "audit": {
                "status": "unavailable",
                "reason": "No saved endpoint trajectory scalars",
            },
            "auxiliary_tables": {},
        }
    audit, offenders = audit_feedback_rows(frame)
    fractions, profiles = fraction_summaries(frame)
    coverage, coverage_rows = margin_coverage_audit(frame)
    normalized = _normalized_summary(frame)
    structural = structural_feedback_eligibility(frame).to_numpy()
    initial = frame.loc[structural & frame.step_index.eq(0)].copy()
    columns = [
        *ROW_KEYS,
        "snr",
        "terminal_sscd",
        "candidate_matched_log_odds",
        "candidate_guided_log_odds",
        "candidate_log_odds_gain",
        "candidate_gain_status",
    ]
    initial = initial[[column for column in columns if column in initial]].rename(
        columns={"candidate_matched_log_odds": "x", "candidate_guided_log_odds": "y"}
    )
    frames = {
        STEMS["PF02"]: normalized,
        STEMS["PF03"]: fractions,
        STEMS["PF04"]: initial,
        STEMS["PF07"]: coverage,
        STEMS["PF08"]: profiles,
    }
    dose = tables.get("summaries", {}).get("dose", pd.DataFrame())
    frames[STEMS["PF06"]] = (
        dose.loc[dose.step_index.eq(0)].copy()
        if not dose.empty and "step_index" in dose
        else pd.DataFrame()
    )
    metadata = {}
    for code, stem in STEMS.items():
        values = frames[stem]
        unavailable = values.empty
        metadata[stem] = {
            "figure_id": code,
            "status": "blocked"
            if audit["status"] == "blocked"
            else "unavailable"
            if unavailable
            else "complete",
            "reason": "Endpoint identity audit failed; see offending IDs"
            if audit["status"] == "blocked"
            else "No eligible saved scalar observations"
            if unavailable
            else None,
            "row_count": len(values),
            "column_schema": {c: str(values[c].dtype) for c in values},
            "formula_version": FORMULA_VERSION,
            "weighting": "Equal prompt mass split among structurally eligible member seeds in each group/step; numerical resolution never alters this denominator.",
            "groups": list(GROUPS),
            "empty_groups": [
                g for g in GROUPS if "group" in values and not values.group.eq(g).any()
            ],
            "band_meaning": "Resolved positive through resolved positive plus unresolved mass; numerical ambiguity, not a confidence interval"
            if code in {"PF03", "PF08"}
            else "Descriptive interquartile range"
            if code in {"PF02", "PF06"}
            else None,
            "fixed_snapshot": "First actual cached update, step_index=0"
            if code in {"PF04", "PF06"}
            else None,
            "guidance_scale": config.get("guidance_scale"),
            "formal_quadrature_certificate": False,
            "excluded_weight_meaning": "Excluded mass uses all-group prompt weights; eligible weights are fixed separately before numerical classification.",
        }
    initial_snrs = _numbers(frame.loc[frame.step_index.eq(0)], "snr")
    finite_initial_snrs = initial_snrs[np.isfinite(initial_snrs)]
    source_counts = {
        "trajectory_rows": len(frame),
        "prompt_count": len(frame[PROMPT_KEYS].drop_duplicates()),
        "sample_count": len(frame[SAMPLE_KEYS].drop_duplicates()),
        "structurally_eligible_rows": int(structural.sum()),
        "structurally_excluded_rows": int((~structural).sum()),
        "per_group": {},
    }
    for group, group_rows in _groups(frame):
        mask = structural_feedback_eligibility(group_rows).to_numpy()
        source_counts["per_group"][group] = {
            "trajectory_rows": len(group_rows),
            "structurally_eligible_rows": int(mask.sum()),
            "structurally_excluded_rows": int((~mask).sum()),
            "prompt_count": len(group_rows[PROMPT_KEYS].drop_duplicates()),
            "sample_count": len(group_rows[SAMPLE_KEYS].drop_duplicates()),
            "gain_status_counts": pd.Series(_gain_classes(group_rows.loc[mask]))
            .value_counts()
            .to_dict(),
            "profile_status_counts": pd.Series(_profile_classes(group_rows.loc[mask]))
            .value_counts()
            .to_dict(),
        }
    for entry in metadata.values():
        entry["counts"] = source_counts
        entry["status_counts"] = {
            "gain": pd.Series(_gain_classes(frame.loc[structural]))
            .value_counts()
            .to_dict(),
            "profile": pd.Series(_profile_classes(frame.loc[structural]))
            .value_counts()
            .to_dict(),
        }
        entry["counts_scope"] = (
            "Full saved trajectory population; exact group/step weighted denominators remain in the figure input table."
        )
        entry["actual_initial_snr"] = (
            float(finite_initial_snrs[0]) if len(finite_initial_snrs) else None
        )
        entry["band_definition"] = entry["band_meaning"]
    if audit["status"] != "blocked" and not structural.any():
        reasons = sorted(set(_strings(frame, "candidate_feedback_status")))
        for entry in metadata.values():
            entry["status"] = "not_applicable"
            entry["reason"] = (
                "No structurally eligible matched feedback comparisons: "
                + "; ".join(reasons)
            )
    elif audit["status"] != "blocked":
        direction = _strings(
            frame.loc[structural], "candidate_direction_normalization_status"
        )
        if len(direction) and np.isin(direction, ["zero_displacement"]).all():
            metadata[STEMS["PF02"]]["status"] = "not_applicable"
            metadata[STEMS["PF02"]]["reason"] = (
                "Direction-normalized gain is undefined for every exactly zero displacement"
            )
        if not any("candidate_margin_" + n + "_l2" in frame for n in MARGINS):
            metadata[STEMS["PF07"]]["status"] = "unavailable"
            metadata[STEMS["PF07"]]["reason"] = (
                "Path integration has not been computed; endpoint figures remain available"
            )
    audit["coverage_audit"] = {
        "status": "available"
        if any("candidate_margin_" + n + "_l2" in frame for n in MARGINS)
        else "integration_unavailable",
        "same_population_and_weights": True,
        "raw_estimates_are_certificates": False,
        "retained_estimated_positive_negative_gain_rows": len(
            coverage_rows.loc[coverage_rows.group.eq("All")]
        )
        if len(coverage_rows)
        else 0,
        "resolved_positive_negative_gain_rows": int(
            coverage_rows.loc[
                coverage_rows.group.eq("All"), "resolved_margin_contradiction"
            ].sum()
        )
        if len(coverage_rows)
        else 0,
        "explanation": "Dotted curves are raw strictly positive estimates. Saved same-denominator differences and row IDs distinguish cancellation/saturation and unresolved source/integration sensitivity from stable gain signs. Nonnegative estimates are separate; no clipping is applied.",
    }
    audit["missing_SSCD_rows"] = int(
        (~np.isfinite(_numbers(frame, "terminal_sscd"))).sum()
    )
    return frames, {
        "figures": metadata,
        "audit": audit,
        "auxiliary_tables": {
            "posterior_feedback_endpoint_offenders": offenders,
            "posterior_feedback_coverage_disagreements": coverage_rows,
        },
    }
