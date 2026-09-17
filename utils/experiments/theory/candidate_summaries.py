"""Fixed, descriptive candidate-gallery summaries from saved scalar tables only."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from .summaries import PROMPT_KEYS, SAMPLE_KEYS, weighted_quantiles

GROUPS = ("all", "SSCD > 0.75", "SSCD <= 0.75")
SSCD_EDGES = np.linspace(0.0, 1.0, 11)
LOG_ODDS_EDGES = np.array([-np.inf, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, np.inf])
RECOVERY_TOLERANCES = (0.1, 0.2, 0.3, 0.5, 1.0)
STABILITY_SALT = "candidate-target-stability-v1"
SUMMARY_POLICY = {
    "variation_contract": "projected-gap-error-1; signed projected branch-gap reference error; positive-part signed variation with norm and absolute-projection variation diagnostics",
    "weighting": "Within each reported subset, equal total mass per complete prompt identity; equal mass per included seed within prompt; repeated observations divide that seed's mass equally.",
    "quantiles": "inverse weighted empirical CDF; IQR is descriptive dispersion, not a confidence interval",
    "outcome_groups": list(GROUPS),
    "outcome_threshold": 0.75,
    "sscd_edges": SSCD_EDGES.tolist(),
    "sscd_tails": "retain <0 and >1 as separate bins; retain missing counts",
    "log_odds_edges": ["-inf", -10, -5, -2, 0, 2, 5, 10, "inf"],
    "normalized_progress_bins": 10,
    "heatmap_low_count_threshold": 20,
    "recovery_tolerances": list(RECOVERY_TOLERANCES),
    "recovery_comparison": "strictly below",
    "joint_tolerance_comparison": "less than or equal",
    "tolerance_grid": "201 evenly spaced RMSE values from zero to all-data finite maximum; no trimming",
    "temporal_windows": "k20=ceil(.2 L), k40=ceil(.4 L), clipped to last stored prediction; exposure 0<=k<k20; response E_u[k20]-E_u[k40]",
    "lagged_analysis": "every available adjacent actual learned prediction pair; no selected optimal lag",
    "target_split": "SHA256(candidate-target-stability-v1|canonical target identity), low bit of first 8 digest bytes; descriptive explored-data stability only",
}
TIME_METRICS = (
    "independent_replay_rmse",
    "matched_construction_vector_residual_rmse",
    "clean_estimate_rounding_sensitivity_rmse",
    "update_rounding_sensitivity_rmse",
    "matched_shift_rmse",
    "candidate_log_probability_gain",
    "candidate_log_odds_gain",
    "candidate_normalized_log_probability_gain",
    "candidate_normalized_log_odds_gain",
    "candidate_current_alignment_cosine",
    "candidate_next_reference_target_contraction_rmse",
    "candidate_specificity_contrast",
    "conditional_target_error_rmse",
    "unconditional_target_error_rmse",
    "joint_target_error_rmse",
    "branch_gap_rmse",
    "current_unconditional_candidate_reference_error_rmse",
    "candidate_radius_tail_rmse",
    "candidate_current_reference_error_u_l2",
    "candidate_conditional_error_l2",
    "candidate_combined_reference_error_l2",
    "candidate_signed_error_projection_l2",
    "candidate_variation_positive_signed_integral_l2",
    "candidate_variation_norm_integral_l2",
    "candidate_variation_abs_projected_integral_l2",
    "candidate_variation_signed_integral_l2",
    *tuple(
        f"candidate_margin_{name}_l2"
        for name in ("original", "combined", "signed_error", "projected_variation")
    ),
)
MARGIN_STEMS = ("original", "combined", "signed_error", "projected_variation")


def _numeric(frame, column):
    return (
        pd.to_numeric(frame[column], errors="coerce").astype(float)
        if column in frame
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )


def _groups(frame):
    yield "all", frame
    score = _numeric(frame, "terminal_sscd")
    yield GROUPS[1], frame.loc[np.isfinite(score) & (score > 0.75)]
    yield GROUPS[2], frame.loc[np.isfinite(score) & (score <= 0.75)]


def _keys(frame, names=PROMPT_KEYS):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(
            "Complete scientific identity required: missing "
            + ", ".join(sorted(missing))
        )
    return list(names)


def prompt_weights(frame):
    """Per-seed observations share the seed's equal within-prompt mass."""
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    keys = _keys(frame)
    samples = _keys(frame, SAMPLE_KEYS)
    seed_counts = frame.groupby(keys, dropna=False)["seed"].transform("nunique")
    observations = frame.groupby(samples, dropna=False)["seed"].transform("size")
    return 1.0 / (seed_counts * observations)


def _counts(rows):
    if rows.empty:
        return dict(n_observations=0, prompt_count=0, seed_count=0, target_count=0)
    return dict(
        n_observations=len(rows),
        prompt_count=len(rows[_keys(rows)].drop_duplicates()),
        seed_count=rows.seed.nunique(),
        target_count=rows[
            next(
                (
                    c
                    for c in ("candidate_target_atom_id", "target_atom_id", "target_id")
                    if c in rows
                )
            )
        ].nunique(),
    )


class _SummaryStats:
    """Reuse subset weights only when metrics have exactly the same finite rows."""

    def __init__(self, rows):
        self.rows = rows
        self.subsets = {}

    def __call__(self, column):
        values = _numeric(self.rows, column).to_numpy()
        finite = np.isfinite(values)
        key = finite.tobytes()
        if key not in self.subsets:
            valid = self.rows.loc[finite]
            self.subsets[key] = prompt_weights(valid).to_numpy(), _counts(valid)
        weights, counts = self.subsets[key]
        # Preserve the original second coercion after filtering: mixed object
        # columns can parse differently once nonnumeric entries are removed.
        values = (
            _numeric(self.rows.loc[finite, [column]], column).to_numpy()
            if column in self.rows
            else np.array([], dtype=float)
        )
        qs = weighted_quantiles(values, weights, (0.1, 0.25, 0.5, 0.75, 0.9))
        return (
            dict(
                zip(("q10", "q25", "median", "q75", "q90"), map(float, qs), strict=True)
            )
            | counts
            | {
                "total_count": len(self.rows),
                "missing_count": len(self.rows) - len(values),
                "mean": float(np.average(values, weights=weights))
                if len(values)
                else np.nan,
                "minimum": float(values.min()) if len(values) else np.nan,
                "maximum": float(values.max()) if len(values) else np.nan,
            }
        )


def _stats(rows, column):
    return _SummaryStats(rows)(column)


def prepare_candidate_scalars(frame):
    """Small derived aliases; source fields are retained unchanged."""
    result = frame.copy(deep=False)
    d = np.sqrt(_numeric(result, "latent_dimension"))
    source = next(
        (
            c
            for c in (
                "candidate_current_reference_error_u_l2",
                "candidate_unconditional_reference_error_l2",
            )
            if c in result
        ),
        None,
    )
    if source:
        result["current_unconditional_candidate_reference_error_rmse"] = (
            _numeric(result, source) / d
        )
    if "candidate_radius_l2" in result and "target_log_complement" in result:
        with np.errstate(under="ignore"):
            result["candidate_radius_tail_rmse"] = (
                _numeric(result, "candidate_radius_l2")
                * np.exp(_numeric(result, "target_log_complement"))
                / d
            )
    return result


def fixed_snapshots(eligible_steps):
    steps = sorted(set(map(int, eligible_steps)))
    if not steps:
        return []
    positions = {0, 1, 2, 4, 9, (len(steps) - 1) // 2, len(steps) - 1}
    return [steps[i] for i in sorted(positions) if i < len(steps)]


def synchronization_snapshots(eligible_steps):
    steps = sorted(set(map(int, eligible_steps)))
    if not steps:
        return []
    positions = {
        0,
        min(math.ceil(0.2 * len(steps)), len(steps) - 1),
        (len(steps) - 1) // 2,
        len(steps) - 1,
    }
    return [steps[i] for i in sorted(positions)]


def _eligible_steps(frame):
    if "destination_sigma" in frame:
        valid = _numeric(frame, "destination_sigma") > 0
        if "kappa" in frame:
            valid &= _numeric(frame, "kappa") > 0
        return sorted(frame.loc[valid, "step_index"].dropna().astype(int).unique())
    if "candidate_gain_status" in frame:
        return sorted(
            frame.loc[
                ~frame.candidate_gain_status.isin(["not_applicable", "unavailable"]),
                "step_index",
            ]
            .dropna()
            .astype(int)
            .unique()
        )
    return sorted(frame.step_index.dropna().astype(int).unique())


def timeseries_summary(frame, metrics=TIME_METRICS):
    output = []
    metrics = [c for c in metrics if c in frame]
    steps = sorted(frame.step_index.unique())
    snrs = {
        step: float(rows.snr.iloc[0])
        for step, rows in frame.groupby("step_index", sort=True)
    }
    for group, cohort in _groups(frame):
        by_step = {k: rows for k, rows in cohort.groupby("step_index", sort=True)}
        for step in steps:
            rows = by_step.get(step, cohort.iloc[:0])
            snr = snrs.get(step, np.nan)
            stats = _SummaryStats(rows)
            for metric in metrics:
                output.append(
                    dict(group=group, metric=metric, step_index=int(step), snr=snr)
                    | stats(metric)
                )
    return pd.DataFrame(output)


def coverage_summary(frame):
    """Unresolved eligible rows remain in every positive-coverage denominator."""
    specs = {
        "independent_replay_status": (
            "saved_endpoint_vs_independent_analytic_drift_no_endpoint_fit",
            "unavailable_saved_noise_realization_recovered_innovation_is_not_independent_replay",
            "not_applicable",
        ),
        "candidate_gain_status": (
            "positive",
            "negative",
            "arithmetic_zero",
            "numerically_unresolved",
            "not_applicable",
        ),
        "candidate_profile_class": (
            "increasing",
            "decreasing",
            "interior_peak",
            "flat",
            "numerically_unresolved",
            "not_applicable",
        ),
    }
    if "candidate_gain_magnitude_status" in frame:
        specs["candidate_gain_magnitude_status"] = (
            "representable",
            "underflow",
            "numerically_unresolved",
            "not_applicable",
        )
    specs.update(
        {
            flag: ("True", "False", "not_applicable")
            for flag in ("candidate_gain_saturated", "candidate_gain_underflow")
            if flag in frame
        }
    )
    specs.update(
        {
            f"candidate_margin_{stem}_status": (
                "positive",
                "negative",
                "numerically_unresolved",
                "not_applicable",
            )
            for stem in MARGIN_STEMS
        }
    )
    specs.update(
        {
            f"candidate_margin_{stem}_estimated_sign": (
                "positive",
                "negative",
                "arithmetic_zero",
                "numerically_unresolved",
                "not_applicable",
            )
            for stem in MARGIN_STEMS
        }
    )
    output = []
    steps = sorted(frame.step_index.unique())
    snrs = {
        step: float(rows.snr.iloc[0])
        for step, rows in frame.groupby("step_index", sort=True)
    }
    for group, cohort in _groups(frame):
        by_step = {step: rows for step, rows in cohort.groupby("step_index", sort=True)}
        for step in steps:
            rows = by_step.get(step, cohort.iloc[:0])
            counts = _counts(rows)
            all_weights = prompt_weights(rows)
            population_denominator = all_weights.sum()
            eligible_weights = {}
            for metric, expected in specs.items():
                if metric not in frame:
                    continue
                statuses = rows[metric].fillna("numerically_unresolved").astype(str)
                if (
                    metric
                    in (
                        "candidate_gain_saturated",
                        "candidate_gain_underflow",
                        "candidate_gain_magnitude_status",
                    )
                    and "candidate_gain_status" in rows
                ):
                    statuses = statuses.where(
                        ~rows.candidate_gain_status.isin(
                            ["not_applicable", "unavailable"]
                        ),
                        "not_applicable",
                    )
                if metric.endswith("_estimated_sign"):
                    resolved_column = metric.removesuffix("_estimated_sign") + "_status"
                    if resolved_column in rows:
                        statuses = statuses.where(
                            ~rows[resolved_column].isin(
                                ["not_applicable", "unavailable"]
                            ),
                            "not_applicable",
                        )
                eligible = ~statuses.str.startswith(("not_applicable", "unavailable"))
                eligibility_key = eligible.to_numpy().tobytes()
                if eligibility_key not in eligible_weights:
                    eligible_weights[eligibility_key] = prompt_weights(
                        rows.loc[eligible]
                    )
                weights = eligible_weights[eligibility_key]
                denominator = float(weights.sum())
                denominator_count = int(eligible.sum())
                for status in dict.fromkeys((*expected, *statuses.unique())):
                    chosen = statuses.loc[eligible].eq(status)
                    numerator = float(weights.loc[chosen].sum())
                    population_numerator = float(
                        all_weights.loc[statuses.eq(status)].sum()
                    )
                    output.append(
                        dict(
                            group=group,
                            metric=metric,
                            step_index=int(step),
                            snr=snrs[step],
                            status=status,
                            count=int(statuses.eq(status).sum()),
                            fraction=numerator / denominator if denominator else np.nan,
                            weighted_numerator=numerator,
                            weighted_denominator=denominator,
                            population_fraction=population_numerator
                            / population_denominator
                            if len(rows)
                            else np.nan,
                            denominator_count=denominator_count,
                            total_count=len(rows),
                            denominator_kind="eligible_including_numerically_unresolved",
                        )
                        | counts
                    )
    return pd.DataFrame(output)


def fixed_sscd_bins(scores):
    values = np.asarray(scores, dtype=float)
    bins = np.searchsorted(SSCD_EDGES, values, side="right") - 1
    bins = np.where(values == 1, 9, bins)
    bins = np.where(~np.isfinite(values), -2, bins)
    return bins.astype(int)


def sscd_bin_summary(frame, metrics, *, scope="initial"):
    rows = frame.copy()
    rows["_sscd_bin"] = fixed_sscd_bins(_numeric(rows, "terminal_sscd"))
    output = []
    for b in (-2, -1, *range(10), 10):
        subset = rows.loc[rows._sscd_bin == b]
        lower = SSCD_EDGES[b] if 0 <= b <= 9 else (1.0 if b == 10 else -np.inf)
        upper = SSCD_EDGES[b + 1] if 0 <= b <= 9 else (0.0 if b == -1 else np.inf)
        stats = _SummaryStats(subset)
        sscd_median = stats("terminal_sscd")["median"]
        for metric in metrics:
            if metric in rows:
                output.append(
                    dict(
                        scope=scope,
                        metric=metric,
                        bin_index=b,
                        bin_lower=lower,
                        bin_upper=upper,
                        bin_status="missing"
                        if b == -2
                        else "lower_tail"
                        if b == -1
                        else "upper_tail"
                        if b == 10
                        else "in_range",
                        sscd_median=sscd_median,
                    )
                    | stats(metric)
                )
    return pd.DataFrame(output)


def within_prompt_center(frame, x, y):
    """Retain every row and explicitly mark prompts without usable variation."""
    rows = frame.copy()
    _keys(rows, SAMPLE_KEYS)
    rows[x] = _numeric(rows, x)
    rows[y] = _numeric(rows, y)
    finite = np.isfinite(_numeric(rows, x)) & np.isfinite(_numeric(rows, y))
    usable = rows.loc[finite]
    group = usable.groupby(PROMPT_KEYS, dropna=False)
    rows["x_centered"] = np.nan
    rows["y_centered"] = np.nan
    rows["within_prompt_status"] = "missing_pair"
    if len(usable):
        rows.loc[finite, "x_centered"] = usable[x] - group[x].transform("mean")
        rows.loc[finite, "y_centered"] = usable[y] - group[y].transform("mean")
        varying = (
            (group[x].transform("nunique") > 1)
            & (group[y].transform("nunique") > 1)
            & (group.seed.transform("nunique") > 1)
        )
        rows.loc[usable.index, "within_prompt_status"] = np.where(
            varying, "usable_seed_variation", "no_usable_seed_variation"
        )
    rows["within_prompt_x"] = x
    rows["within_prompt_y"] = y
    return rows


def descriptive_trend(frame, x="x_centered", y="y_centered"):
    output = []
    for group, rows in _groups(frame):
        valid = rows.loc[
            np.isfinite(_numeric(rows, x)) & np.isfinite(_numeric(rows, y))
        ]
        if "within_prompt_status" in valid:
            valid = valid.loc[valid.within_prompt_status == "usable_seed_variation"]
        w = prompt_weights(valid).to_numpy()
        xx, yy = _numeric(valid, x).to_numpy(), _numeric(valid, y).to_numpy()
        slope = intercept = correlation = np.nan
        if len(valid):
            xm, ym = np.average(xx, weights=w), np.average(yy, weights=w)
            vx = np.average((xx - xm) ** 2, weights=w)
            vy = np.average((yy - ym) ** 2, weights=w)
            cov = np.average((xx - xm) * (yy - ym), weights=w)
            if vx > 0:
                slope = cov / vx
                intercept = ym - slope * xm
            if vx > 0 and vy > 0:
                correlation = cov / math.sqrt(vx * vy)
        output.append(
            dict(
                group=group,
                x=x,
                y=y,
                slope=slope,
                intercept=intercept,
                correlation=correlation,
                status="descriptive_no_inference",
                excluded_count=len(rows) - len(valid),
            )
            | _counts(valid)
        )
    return pd.DataFrame(output)


def temporal_feedback_summary(frame, *, eligible_steps=None):
    steps = _eligible_steps(frame) if eligible_steps is None else list(eligible_steps)
    L = len(steps)
    if not L:
        return pd.DataFrame(), pd.DataFrame()
    last = int(frame.step_index.max())
    k20 = min(math.ceil(0.2 * L), last)
    k40 = min(math.ceil(0.4 * L), last)
    output = []
    lagged = []
    gain = "candidate_normalized_log_odds_gain"
    columns = [
        c
        for c in (
            *SAMPLE_KEYS,
            "candidate_target_atom_id",
            "target_atom_id",
            "step_index",
            "snr",
            "terminal_sscd",
            "unconditional_target_error_rmse",
            gain,
        )
        if c in frame
    ]
    frame = frame[columns].copy()
    for column in (gain, "unconditional_target_error_rmse", "snr"):
        frame[column] = _numeric(frame, column)
    for identity, rows in frame.groupby(
        _keys(frame, SAMPLE_KEYS), sort=True, dropna=False
    ):
        rows = rows.sort_values("step_index")
        if rows.step_index.duplicated().any():
            raise ValueError("Duplicate stored step in complete sample identity")
        by_step = rows.set_index("step_index")
        base = dict(zip(SAMPLE_KEYS, identity, strict=True))
        base["terminal_sscd"] = rows.terminal_sscd.iloc[0]
        for column in ("candidate_target_atom_id", "target_atom_id"):
            if column in rows:
                base[column] = rows[column].iloc[0]
        early = rows.loc[(rows.step_index >= 0) & (rows.step_index < k20)]
        values = _numeric(early, gain)
        finite = np.isfinite(values)
        response = np.nan
        if k40 > k20 and k20 in by_step.index and k40 in by_step.index:
            response = float(
                by_step.loc[k20, "unconditional_target_error_rmse"]
                - by_step.loc[k40, "unconditional_target_error_rmse"]
            )
        output.append(
            base
            | dict(
                early_normalized_gain=float(values.loc[finite].mean())
                if finite.any()
                else np.nan,
                subsequent_unconditional_improvement=response,
                window_k20=k20,
                window_k40=k40,
                eligible_transition_count=L,
                exposure_observation_count=len(early),
                exposure_finite_count=int(finite.sum()),
                exposure_missing_count=int((~finite).sum()),
                response_status="available"
                if np.isfinite(response)
                else "not_applicable_no_subsequent_interval"
                if k40 <= k20
                else "missing_prediction_endpoint",
            )
        )
        for a, b in zip(rows.iloc[:-1].itertuples(), rows.iloc[1:].itertuples()):
            if int(b.step_index) != int(a.step_index) + 1:
                continue
            lagged.append(
                base
                | dict(
                    step_index=int(a.step_index),
                    next_prediction_step=int(b.step_index),
                    snr=float(a.snr),
                    lag=1,
                    normalized_gain=float(getattr(a, gain, np.nan)),
                    next_unconditional_improvement=float(
                        a.unconditional_target_error_rmse
                        - b.unconditional_target_error_rmse
                    ),
                )
            )
    early_frame = within_prompt_center(
        pd.DataFrame(output),
        "early_normalized_gain",
        "subsequent_unconditional_improvement",
    )
    lagged_frame = pd.DataFrame(lagged)
    # Center lagged seed comparisons separately at each stored step; never pool
    # timestep effects into a purported within-prompt association.
    if len(lagged_frame):
        lagged_frame = pd.concat(
            [
                within_prompt_center(
                    rows, "normalized_gain", "next_unconditional_improvement"
                )
                for _, rows in lagged_frame.groupby("step_index", sort=True)
            ],
            ignore_index=True,
        )
    return early_frame, lagged_frame


def feedback_heatmap(frame, *, eligible_steps=None):
    steps = _eligible_steps(frame) if eligible_steps is None else list(eligible_steps)
    last = max(steps) if steps else 0
    rows = frame[
        [
            c
            for c in (
                *SAMPLE_KEYS,
                "step_index",
                "snr",
                "terminal_sscd",
                "candidate_gain_status",
                "candidate_matched_log_odds",
            )
            if c in frame
        ]
    ].copy()
    rows["_progress_bin"] = np.minimum(
        9, np.floor(10 * _numeric(rows, "step_index") / max(1, last + 1)).astype(int)
    )
    odds = _numeric(rows, "candidate_matched_log_odds")
    regimes = np.searchsorted(LOG_ODDS_EDGES, odds, side="right") - 1
    rows["_regime_bin"] = np.where(np.isnan(odds), -1, np.clip(regimes, 0, 7))
    output = []
    for group, cohort in _groups(rows):
        for progress in range(10):
            for regime in range(-1, 8):
                subset = cohort.loc[
                    (cohort._progress_bin == progress) & (cohort._regime_bin == regime)
                ]
                status = subset.get(
                    "candidate_gain_status",
                    pd.Series("not_applicable", index=subset.index),
                )
                valid = ~status.isin(["not_applicable", "unavailable"])
                eligible = subset.loc[valid]
                weights = prompt_weights(eligible)
                denom = float(weights.sum())

                def count(name):
                    return int(status.eq(name).sum())

                def fraction(name):
                    return (
                        float(weights.loc[status.loc[valid].eq(name)].sum() / denom)
                        if denom
                        else np.nan
                    )

                pos, unresolved = (
                    fraction("positive"),
                    fraction("numerically_unresolved"),
                )
                output.append(
                    dict(
                        group=group,
                        progress_bin=progress,
                        progress_lower=progress / 10,
                        progress_upper=(progress + 1) / 10,
                        regime_bin=regime,
                        regime_lower=LOG_ODDS_EDGES[regime] if regime >= 0 else np.nan,
                        regime_upper=LOG_ODDS_EDGES[regime + 1]
                        if regime >= 0
                        else np.nan,
                        positive_fraction=pos,
                        negative_fraction=fraction("negative"),
                        zero_fraction=fraction("arithmetic_zero"),
                        unresolved_fraction=unresolved,
                        positive_fraction_lower=pos,
                        positive_fraction_upper=pos + unresolved,
                        positive_count=count("positive"),
                        negative_count=count("negative"),
                        zero_count=count("arithmetic_zero"),
                        unresolved_count=count("numerically_unresolved"),
                        not_applicable_count=int((~valid).sum()),
                        denominator_count=int(valid.sum()),
                        weighted_denominator=denom,
                        status="empty"
                        if not len(subset)
                        else "missing_counterfactual_regime"
                        if regime < 0
                        else "low_count"
                        if len(eligible) < 20
                        else "descriptive",
                        uncertainty_kind="unresolved-sign envelope; no independent-row confidence interval",
                    )
                    | _counts(subset)
                )
    return pd.DataFrame(output)


def tolerance_grid(*values, count=201):
    arrays = [np.asarray(value, dtype=float).ravel() for value in values]
    finite = np.concatenate(arrays) if arrays else np.array([])
    finite = finite[np.isfinite(finite) & (finite >= 0)]
    if not len(finite):
        return np.array([], dtype=float)
    return np.unique(np.linspace(0.0, float(finite.max()), count))


def joint_tolerance_summary(frame, *, grid=None, snapshots=None):
    if grid is None:
        grid = tolerance_grid(_numeric(frame, "joint_target_error_rmse"))
    if snapshots is None:
        snapshots = synchronization_snapshots(_eligible_steps(frame))
    output = []
    for group, cohort in _groups(frame):
        for step in snapshots:
            source = frame.loc[frame.step_index == step]
            rows = cohort.loc[cohort.step_index == step]
            rows = rows.loc[np.isfinite(_numeric(rows, "joint_target_error_rmse"))]
            weights = prompt_weights(rows)
            denominator = weights.sum()
            counts = _counts(rows)
            for tolerance in grid:
                fraction = (
                    float(
                        weights.loc[rows.joint_target_error_rmse <= tolerance].sum()
                        / denominator
                    )
                    if len(rows)
                    else np.nan
                )
                output.append(
                    dict(
                        group=group,
                        step_index=int(step),
                        snr=float(source.snr.iloc[0]) if len(source) else np.nan,
                        tolerance=float(tolerance),
                        fraction=fraction,
                    )
                    | counts
                )
    return pd.DataFrame(output)


def recovery_time_summary(frame, tolerances=RECOVERY_TOLERANCES):
    output = []
    for identity, rows in frame.groupby(
        _keys(frame, SAMPLE_KEYS), sort=True, dropna=False
    ):
        rows = rows.sort_values("step_index")
        steps = rows.step_index.to_numpy(dtype=int)
        if len(np.unique(steps)) != len(steps):
            raise ValueError("Duplicate prediction steps cannot define first passage")
        base = dict(zip(SAMPLE_KEYS, identity, strict=True)) | dict(
            terminal_sscd=rows.terminal_sscd.iloc[0],
            last_observed_prediction_step=int(steps[-1]),
        )
        for tolerance in tolerances:
            item = base | dict(tolerance=float(tolerance))
            for branch in ("conditional", "unconditional"):
                values = _numeric(rows, branch + "_target_error_rmse").to_numpy()
                finite = np.isfinite(values)
                recovered = finite & (values < float(tolerance))
                reached = np.flatnonzero(recovered)
                consecutive = np.diff(steps) == 1
                recrossings = int(
                    np.sum(recovered[:-1] & finite[1:] & ~recovered[1:] & consecutive)
                )
                persistent = [
                    i
                    for i in range(max(0, len(steps) - 2))
                    if recovered[i : i + 3].all()
                    and np.all(np.diff(steps[i : i + 3]) == 1)
                ]
                item.update(
                    {
                        branch + "_first_step": int(steps[reached[0]])
                        if len(reached)
                        else np.nan,
                        branch + "_status": "reached"
                        if len(reached)
                        else "right_censored"
                        if finite.any()
                        else "unavailable",
                        branch + "_right_censored": bool(
                            not len(reached) and finite.any()
                        ),
                        branch + "_recrossing_count": recrossings,
                        branch + "_missing_count": int((~finite).sum()),
                        branch + "_three_step_first": int(steps[persistent[0]])
                        if persistent
                        else np.nan,
                        branch + "_three_step_status": "reached"
                        if persistent
                        else "not_observed",
                    }
                )
            output.append(item)
    return pd.DataFrame(output)


def terminal_measurements(endpoint):
    rows = endpoint.copy()
    clean = (
        rows.get("terminal_clean_applicable", pd.Series(False, index=rows.index))
        .fillna(False)
        .astype(bool)
    )
    error = _numeric(rows, "terminal_rmse")
    if "terminal_rmse" not in rows:
        error = _numeric(rows, "terminal_endpoint_error_rmse")
    observed = _numeric(rows, "terminal_A_rmse") + _numeric(rows, "terminal_B_rmse")
    candidate = _numeric(rows, "candidate_terminal_bound_rmse")
    rows["terminal_observed_bound_rmse"] = observed
    rows["terminal_candidate_bound_rmse"] = candidate
    rows["terminal_observed_error_rmse"] = error
    for name, bound in (("observed", observed), ("candidate", candidate)):
        applicable = clean.copy()
        if (
            name == "candidate"
            and "candidate_terminal_bound_applies_to_endpoint" in rows
        ):
            applicable &= rows.candidate_terminal_bound_applies_to_endpoint.fillna(
                False
            ).astype(bool)
        valid = (
            applicable
            & np.isfinite(bound)
            & np.isfinite(error)
            & (bound >= 0)
            & (error >= 0)
        )
        rows[f"terminal_{name}_bound_error_ratio"] = np.where(
            valid & (error > 0), bound / error, np.nan
        )
        rows[f"terminal_{name}_ratio_status"] = np.select(
            [~applicable, ~valid, error == 0],
            ["not_applicable", "unavailable", "undefined_zero_endpoint_error"],
            default="valid",
        )
    rows["terminal_error_observed_bound_ratio"] = np.where(
        clean & (observed > 0) & np.isfinite(error), error / observed, np.nan
    )
    rows["terminal_ratio_status"] = rows.terminal_observed_ratio_status
    return rows


def terminal_tolerance_summary(endpoint, *, grid=None):
    rows = terminal_measurements(endpoint)
    metrics = (
        "terminal_observed_error_rmse",
        "terminal_observed_bound_rmse",
        "terminal_candidate_bound_rmse",
    )
    if grid is None:
        grid = tolerance_grid(*[_numeric(rows, m) for m in metrics])
    output = []
    for group, cohort in _groups(rows):
        gate = (
            cohort.get(
                "terminal_clean_applicable", pd.Series(False, index=cohort.index)
            )
            .fillna(False)
            .astype(bool)
        )
        for metric in metrics:
            eligible = gate & np.isfinite(
                _numeric(cohort, "terminal_observed_error_rmse")
            )
            valid = eligible & np.isfinite(_numeric(cohort, metric))
            if (
                metric == "terminal_candidate_bound_rmse"
                and "candidate_terminal_bound_applies_to_endpoint" in cohort
            ):
                valid &= cohort.candidate_terminal_bound_applies_to_endpoint.fillna(
                    False
                ).astype(bool)
            subset = cohort.loc[eligible]
            weights = prompt_weights(subset)
            denominator = float(weights.sum())
            unresolved = (
                float(weights.loc[~valid.loc[eligible]].sum() / denominator)
                if denominator
                else np.nan
            )
            counts = _counts(subset)
            missing_count = int((eligible & ~valid).sum())
            measured_count = int(valid.sum())
            not_applicable_count = int((~gate).sum())
            status = (
                "applicable"
                if valid.any()
                else "not_applicable"
                if not gate.any()
                else "unavailable"
            )
            for tolerance in grid:
                within = valid.loc[eligible] & (subset[metric] <= tolerance)
                fraction = (
                    float(weights.loc[within].sum() / denominator)
                    if denominator
                    else np.nan
                )
                output.append(
                    dict(
                        group=group,
                        metric=metric,
                        tolerance=float(tolerance),
                        fraction=fraction,
                        fraction_upper=fraction + unresolved,
                        unresolved_fraction=unresolved,
                        missing_count=missing_count,
                        measured_count=measured_count,
                        total_count=len(cohort),
                        not_applicable_count=not_applicable_count,
                        status=status,
                    )
                    | counts
                )
    return pd.DataFrame(output)


def target_stability_split(frame):
    rows = frame.copy()
    identity = next(
        (
            column
            for column in ("candidate_target_atom_id", "target_atom_id", "target_id")
            if column in rows
        ),
        None,
    )
    if identity is None:
        raise ValueError("Stability split requires canonical target identity")
    keys = rows[identity].astype(str)
    assignment = {
        key: int.from_bytes(
            hashlib.sha256((STABILITY_SALT + "|" + key).encode()).digest()[:8], "big"
        )
        % 2
        for key in keys.unique()
    }
    rows["target_stability_split"] = keys.map(assignment).map({0: "A", 1: "B"})
    rows["target_stability_identity_field"] = identity
    return rows


def unique_initial_samples(frame, columns=()):
    """Validate rather than average prompt-independent quantities for each run/seed."""
    keys = ["run_id", "seed"]
    if not set(keys).issubset(frame):
        raise ValueError("Unique initial observations require run_id and seed")
    for column in columns:
        if column in frame:
            counts = frame.groupby(keys, dropna=False)[column].nunique(dropna=False)
            if (counts > 1).any():
                raise ValueError(
                    f"Prompt-independent initial field varies within run/seed: {column}"
                )
    order = [
        k for k in (*keys, "original_index", "record_id", "target_id") if k in frame
    ]
    return (
        frame.sort_values(order, kind="stable")
        .drop_duplicates(keys)
        .reset_index(drop=True)
    )


def dose_summary(dose, *, snapshots=None):
    if dose is None or dose.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "step_index",
                "snr",
                "lambda",
                "median",
                "q25",
                "q75",
                "n_observations",
                "prompt_count",
                "seed_count",
            ]
        )
    rows = dose.loc[dose.step_index.isin(snapshots)] if snapshots is not None else dose
    # Compact lists are expanded only for predeclared snapshots, never a full
    # trajectory-by-dose Cartesian materialization.
    compact = (
        "lambda" in rows
        and len(rows)
        and isinstance(rows["lambda"].iloc[0], (list, tuple, np.ndarray))
    )
    if "lambda" not in rows or compact:
        grid_column = next(
            (
                c
                for c in (
                    "lambda",
                    "candidate_dose_lambda",
                    "candidate_dose_lambdas",
                    "lambda_grid",
                    "dose_lambda",
                )
                if c in rows
            ),
            None,
        )
        gain_column = next(
            (
                c
                for c in (
                    "candidate_dose_log_probability_gain",
                    "candidate_dose_log_probability_gains",
                    "log_probability_gain",
                    "dose_log_probability_gain",
                )
                if c in rows
            ),
            None,
        )
        if grid_column is None or gain_column is None:
            raise ValueError(
                "Dose table needs fixed lambda and log-probability gain lists"
            )
        rows = rows.explode([grid_column, gain_column], ignore_index=True).rename(
            columns={grid_column: "lambda", gain_column: "candidate_dose_gain"}
        )
    else:
        rows = rows.copy()
        column = next(
            (
                c
                for c in (
                    "candidate_dose_gain",
                    "candidate_dose_log_probability_gain",
                    "candidate_log_probability_gain",
                    "log_probability_gain",
                )
                if c in rows
            ),
            None,
        )
        if column is None:
            raise ValueError("Long dose table is missing gain column")
        rows["candidate_dose_gain"] = rows[column]
    rows["lambda"] = pd.to_numeric(rows["lambda"], errors="coerce")
    rows["candidate_dose_gain"] = pd.to_numeric(
        rows.candidate_dose_gain, errors="coerce"
    )
    output = []
    for group, cohort in _groups(rows):
        for (step, lam), subset in cohort.groupby(["step_index", "lambda"], sort=True):
            output.append(
                dict(
                    group=group,
                    step_index=int(step),
                    snr=float(subset.snr.iloc[0]),
                    **{"lambda": float(lam)},
                )
                | _stats(subset, "candidate_dose_gain")
            )
    return pd.DataFrame(output)


def retrieval_ecdf_summary(initial):
    output = []
    for group, cohort in _groups(initial):
        for branch in ("conditional", "unconditional"):
            column = f"initial_{branch}_target_rank"
            if column not in initial:
                continue
            valid = cohort.loc[
                np.isfinite(_numeric(cohort, column)) & (_numeric(cohort, column) >= 1)
            ]
            weights = prompt_weights(valid)
            ranks = sorted(
                _numeric(initial, column).loc[_numeric(initial, column) >= 1].unique()
            )
            if not ranks:
                ranks = [np.nan]
            counts = _counts(valid)
            denominator = weights.sum()
            tie_count = int(
                (_numeric(valid, f"initial_{branch}_target_tie_count") > 1).sum()
            )
            for rank in ranks:
                output.append(
                    dict(
                        group=group,
                        branch=branch,
                        rank=float(rank),
                        fraction=float(
                            weights.loc[valid[column] <= rank].sum() / denominator
                        )
                        if len(valid)
                        else np.nan,
                        tie_count=tie_count,
                        bank_size=int(initial.initial_retrieval_bank_size.iloc[0])
                        if "initial_retrieval_bank_size" in initial and len(initial)
                        else None,
                    )
                    | counts
                )
    return pd.DataFrame(output)


def initial_error_bin_summary(initial, count=20):
    """IR02 equal-width error bins span the full observed range; no tail trimming."""
    x = _numeric(initial, "conditional_target_error_rmse")
    finite = x[np.isfinite(x)]
    output = []
    if not len(finite):
        return pd.DataFrame()
    lo, hi = float(finite.min()), float(finite.max())
    edges = np.linspace(lo, hi, count + 1) if hi > lo else np.array([lo, hi])
    bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        rows = initial.loc[np.isfinite(x) & (bins == b)]
        stats = _SummaryStats(rows)
        result = dict(
            bin_index=b,
            bin_lower=float(edges[b]),
            bin_upper=float(edges[b + 1]),
            conditional_error_median=stats("conditional_target_error_rmse")["median"],
        )
        result.update(stats("terminal_sscd"))
        output.append(result)
    return pd.DataFrame(output)


def sscd_timeseries_summary(frame):
    columns = [
        c
        for c in (*SAMPLE_KEYS, "step_index", "snr", "terminal_sscd", *TIME_METRICS)
        if c in frame
    ]
    rows = frame[list(dict.fromkeys(columns))].copy(deep=False)
    output = []
    for step, subset in rows.groupby("step_index", sort=True):
        summary = sscd_bin_summary(
            subset, [c for c in TIME_METRICS if c in subset], scope="trajectory"
        )
        summary["step_index"] = int(step)
        summary["snr"] = float(subset.snr.iloc[0])
        output.append(summary)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def reference_sweep_summary(sweep):
    output = []
    for (index, snr), rows in sweep.groupby(
        ["snr_grid_index", "analytical_snr"], sort=True
    ):
        if rows.seed.duplicated().any():
            raise ValueError(
                "Analytical sweep must have one observation per seed and SNR"
            )
        values = pd.to_numeric(
            rows.candidate_reference_movement_rmse, errors="coerce"
        ).to_numpy()
        valid = values[np.isfinite(values)]
        q = weighted_quantiles(valid, np.ones(len(valid)), (0.1, 0.25, 0.5, 0.75, 0.9))
        output.append(
            dict(
                snr_grid_index=int(index),
                analytical_snr=float(snr),
                actual_initial_snr=float(rows.actual_initial_snr.iloc[0]),
                n_observations=len(valid),
                seed_count=rows.seed.nunique(),
                missing_count=len(rows) - len(valid),
                evidence="reference_only_illustration",
            )
            | dict(
                zip(("q10", "q25", "median", "q75", "q90"), map(float, q), strict=True)
            )
        )
    return pd.DataFrame(output)


def population_summary(initial):
    output = []
    high = initial.loc[
        np.isfinite(_numeric(initial, "terminal_sscd"))
        & (_numeric(initial, "terminal_sscd") > 0.75)
    ]
    lower = initial.loc[
        np.isfinite(_numeric(initial, "terminal_sscd"))
        & (_numeric(initial, "terminal_sscd") <= 0.75)
    ]
    overlap = len(
        high[PROMPT_KEYS]
        .drop_duplicates()
        .merge(lower[PROMPT_KEYS].drop_duplicates(), on=PROMPT_KEYS)
    )
    for group, rows in _groups(initial):
        output.append(
            dict(
                group=group,
                overlapping_outcome_prompt_count=overlap,
                out_of_range_sscd_count=int(
                    (
                        (_numeric(rows, "terminal_sscd") < 0)
                        | (_numeric(rows, "terminal_sscd") > 1)
                    ).sum()
                ),
                missing_sscd_count=int(
                    (~np.isfinite(_numeric(rows, "terminal_sscd"))).sum()
                ),
            )
            | _counts(rows)
        )
    return pd.DataFrame(output)


def phase_summary(frame):
    """Fixed thirds of the full saved prediction index; phases never optimized."""
    columns = [
        c
        for c in (*SAMPLE_KEYS, "step_index", "terminal_sscd", *TIME_METRICS)
        if c in frame
    ]
    rows = frame[list(dict.fromkeys(columns))].copy()
    total = int(frame.step_index.max()) + 1
    rows["phase"] = np.asarray(["early", "middle", "late"])[
        np.minimum(2, (3 * rows.step_index.to_numpy(dtype=int)) // max(1, total))
    ]
    output = []
    for group, cohort in _groups(rows):
        for phase in ("early", "middle", "late"):
            subset = cohort.loc[cohort.phase == phase]
            stats = _SummaryStats(subset)
            for metric in TIME_METRICS:
                if metric in rows:
                    output.append(
                        dict(
                            group=group,
                            phase=phase,
                            metric=metric,
                            prediction_count=total,
                        )
                        | stats(metric)
                    )
    return pd.DataFrame(output)


def stability_summaries(frame, temporal):
    # Recompute weights inside each fixed target split. Hashing never reads a
    # gain, SSCD, timestep, or error; all negative and unresolved rows remain.
    columns = [
        c
        for c in (
            *SAMPLE_KEYS,
            "candidate_target_atom_id",
            "target_atom_id",
            "step_index",
            "snr",
            "terminal_sscd",
            *TIME_METRICS,
            "candidate_gain_status",
            "candidate_profile_class",
            "independent_replay_status",
            *(
                f"candidate_margin_{stem}_{suffix}"
                for stem in MARGIN_STEMS
                for suffix in ("status", "estimated_sign")
            ),
        )
        if c in frame
    ]
    assigned = target_stability_split(frame[list(dict.fromkeys(columns))])
    times = []
    covers = []
    trends = []
    for split in ("A", "B"):
        subset = assigned.loc[assigned.target_stability_split == split]
        if len(subset):
            times.append(
                timeseries_summary(subset).assign(target_stability_split=split)
            )
            covers.append(coverage_summary(subset).assign(target_stability_split=split))
        if temporal is not None and len(temporal):
            temporal_split = target_stability_split(temporal)
            trends.append(
                descriptive_trend(
                    temporal_split.loc[temporal_split.target_stability_split == split]
                ).assign(target_stability_split=split)
            )
    return {
        "stability_timeseries": pd.concat(times, ignore_index=True)
        if times
        else pd.DataFrame(),
        "stability_coverage": pd.concat(covers, ignore_index=True)
        if covers
        else pd.DataFrame(),
        "stability_temporal_trend": pd.concat(trends, ignore_index=True)
        if trends
        else pd.DataFrame(),
    }


def build_candidate_summaries(
    trajectory,
    initial=None,
    baseline=None,
    endpoint=None,
    dose=None,
    *,
    eligible_steps=None,
    controls=None,
    reference_sweep=None,
):
    """Build named scalar-only tables. Missing specialized metrics stay local."""
    initial = (
        trajectory.loc[trajectory.step_index == trajectory.step_index.min()].copy()
        if initial is None
        else initial.copy()
    )
    needed = (
        *SAMPLE_KEYS,
        "candidate_target_atom_id",
        "target_atom_id",
        "step_index",
        "snr",
        "terminal_sscd",
        "latent_dimension",
        "destination_sigma",
        "kappa",
        "candidate_radius_l2",
        "target_log_complement",
        "candidate_unconditional_reference_error_l2",
        "candidate_matched_log_odds",
        "candidate_gain_status",
        "candidate_gain_magnitude_status",
        "candidate_profile_class",
        "candidate_gain_saturated",
        "candidate_gain_underflow",
        "independent_replay_status",
        *TIME_METRICS,
        *(
            f"candidate_margin_{stem}_{suffix}"
            for stem in MARGIN_STEMS
            for suffix in ("status", "estimated_sign")
        ),
    )
    frame = prepare_candidate_scalars(
        trajectory[[c for c in dict.fromkeys(needed) if c in trajectory]]
    )
    steps = _eligible_steps(frame) if eligible_steps is None else list(eligible_steps)
    initial["paired_initial_improvement_rmse"] = _numeric(
        initial, "unconditional_target_error_rmse"
    ) - _numeric(initial, "conditional_target_error_rmse")
    within = within_prompt_center(
        initial, "paired_initial_improvement_rmse", "terminal_sscd"
    )
    temporal, lagged = temporal_feedback_summary(frame, eligible_steps=steps)
    snapshots = fixed_snapshots(steps)
    output = {
        "timeseries": timeseries_summary(frame),
        "coverage": coverage_summary(frame),
        "sscd_bins": sscd_bin_summary(
            initial,
            [
                "conditional_target_error_rmse",
                "unconditional_target_error_rmse",
                "paired_initial_improvement_rmse",
                "injection_relative_mismatch",
            ],
        ),
        "within_prompt_initial": within,
        "within_prompt_initial_trend": descriptive_trend(within),
        "temporal_feedback": temporal,
        "temporal_feedback_lagged": lagged,
        "temporal_feedback_trend": descriptive_trend(temporal)
        if len(temporal)
        else pd.DataFrame(),
        "feedback_heatmap": feedback_heatmap(frame, eligible_steps=steps),
        "joint_tolerance": joint_tolerance_summary(
            frame, snapshots=synchronization_snapshots(steps)
        ),
        "recovery_times": recovery_time_summary(frame),
        "dose": dose_summary(dose, snapshots=snapshots),
        "stability": target_stability_split(initial),
        "phase_summary": phase_summary(frame),
        "retrieval_ecdf": retrieval_ecdf_summary(initial),
        "initial_error_bins": initial_error_bin_summary(initial),
        "sscd_timeseries": sscd_timeseries_summary(frame),
        "population": population_summary(initial),
    }
    output.update(stability_summaries(frame, temporal))
    output["reference_sweep"] = (
        reference_sweep_summary(reference_sweep)
        if reference_sweep is not None
        else pd.DataFrame()
    )
    if baseline is not None:
        output["unique_initial"] = unique_initial_samples(
            baseline,
            columns=(
                "initial_unconditional_center_rmse",
                "unconditional_center_rmse",
                "initial_candidate_reference_movement_rmse",
                "initial_unconditional_candidate_reference_error_rmse",
            ),
        )
    if endpoint is not None:
        output["terminal_ratios"] = terminal_measurements(endpoint)
        output["terminal_tolerance"] = terminal_tolerance_summary(endpoint)
    else:
        output["terminal_ratios"] = pd.DataFrame()
        output["terminal_tolerance"] = pd.DataFrame()
    return output
