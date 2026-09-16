"""Analysis-only compact inputs for the seven direct manuscript comparisons.

This module accepts already measured scalar tables. It never reads raw tensors,
loads a model, reconstructs a vector distance from scalar centers, or performs
posterior integration. All descriptive summaries are saved by the caller before
paper_plotting is invoked. Importing the renderer does not import this module.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import TheoryError
from .summaries import weighted_quantiles

DIRECT_FIGURE_VERSION = "direct-figure-inputs-1"
RECOVERY_TOLERANCES = (0.1, 0.2, 0.3, 0.5, 1.0)
PAIR_KEYS = ["run_id", "original_index", "record_id", "target_id"]
PAIR_STEP = PAIR_KEYS + ["step_index"]
GAUSSIAN_KEYS = ["run_id", "seed", "step_index"]
LEGACY_STEMS = (
    "initial_recovery", "initial_unconditional_concentration", "posterior_feedback_positive_fraction",
    "target_synchronization", "initial_recovery_within_prompt", "initial_target_retrieval_rank",
    "posterior_feedback_initial_comparison", "posterior_feedback_normalized_gain",
    "posterior_feedback_increasing_profile_fraction", "branch_gap", "joint_target_recovery_early",
    "joint_target_recovery_late", "terminal_error_terms", "terminal_bound_tightness",
    "initial_candidate_reference_discrepancy", "initial_reference_snr_sweep", "initial_injection_geometry",
    "initial_target_coordinates", "initial_injection_mismatch", "posterior_feedback_initial_dose",
    "posterior_feedback_coverage_audit", "synchronization_bound_components", "matched_update_residual",
    "last_prediction_error_terms", "terminal_cancellation",
)
SCATTER_SPECS = {
    "corollary3_initial_cfg_amplification": ("initial", "direct_cor3_injection_bound_rmse", "direct_cor3_injection_discrepancy_rmse", "direct_cor3"),
    "corollary3_guided_estimate_error": ("initial", "direct_cor3_guided_bound_rmse", "direct_cor3_guided_discrepancy_rmse", "direct_cor3"),
    "lemma4_matched_displacement": ("matched_updates", "direct_lemma4_predicted_rmse", "direct_lemma4_displacement_rmse", "direct_lemma4"),
    "lemma4_vector_residual": ("matched_updates", "direct_lemma4_source_tolerance_rmse", "direct_lemma4_vector_residual_rmse", "direct_lemma4"),
    "proposition5_posterior_feedback": ("matched_updates", "direct_prop5_margin_rmse", "direct_prop5_log_probability_gain", "direct_prop5"),
    "lemma6_target_specific_synchronization": ("trajectory", "direct_lemma6_rhs_rmse", "direct_lemma6_gap_rmse", "direct_lemma6"),
    "lemma6_posterior_concentration": ("trajectory", "direct_radius_tail_rmse", "direct_reference_target_error_rmse", "direct_lemma6"),
    "theorem7_final_reproduction": ("terminal", "direct_theorem7_bound_rmse", "direct_theorem7_endpoint_error_rmse", "direct_theorem7"),
}
BOUND_STEMS = {
    "corollary3_initial_cfg_amplification", "corollary3_guided_estimate_error",
    "lemma2_reference_to_learned_bound", "lemma6_target_specific_synchronization",
    "lemma6_posterior_concentration", "theorem7_final_reproduction",
}


def _number(frame, column):
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _boolean(frame, column, default=False):
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    allowed = values.isna() | values.isin([True, False])
    if not allowed.all():
        raise TheoryError(f"{column} must contain measured Boolean values")
    return values.fillna(default).astype(bool)


def _require(frame, columns, table):
    missing = set(columns) - set(frame)
    if missing:
        raise TheoryError(f"Missing direct measurement columns in {table}: {sorted(missing)}; rerun direct analysis")


def _unique(frame, keys, table):
    _require(frame, keys, table)
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise TheoryError(f"Missing or duplicate complete scientific identity in {table}")


def _source(frame, expected, table):
    _require(frame, ["input_source"], table)
    if not frame.input_source.eq(expected).all():
        raise TheoryError(f"{table} mixes input laws; expected {expected}")


def _counts(frame):
    result = {"rows": int(len(frame))}
    if set(PAIR_KEYS).issubset(frame):
        result["pairs"] = int(len(frame[PAIR_KEYS].drop_duplicates()))
    if {"run_id", "seed"}.issubset(frame):
        result["unique_run_seeds"] = int(len(frame[["run_id", "seed"]].drop_duplicates()))
    return result


def _statuses(frame):
    columns = [c for c in frame if c.endswith("_status") or c == "marker_class"]
    return {column: {str(k): int(v) for k, v in frame[column].fillna("missing").value_counts(dropna=False).items()} for column in columns}


def _quantities(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    result = {prefix + "finite_count": int(len(values))}
    if not len(values):
        return result
    result.update({prefix + "minimum": float(values.min()), prefix + "maximum": float(values.max()), prefix + "mean": float(values.mean())})
    for label, q in (("q10", .1), ("q25", .25), ("median", .5), ("q75", .75), ("q90", .9)):
        result[prefix + label] = float(np.quantile(values, q))
    return result


def _scatter_summary(stem, frame):
    """Save counts and absolute scales, not only an inequality agreement rate."""
    if frame.empty:
        return []
    output = []
    groups = frame.groupby("step_index", sort=True, dropna=False) if "step_index" in frame else [(None, frame)]
    for step, rows in groups:
        x, y = _number(rows, "x"), _number(rows, "y")
        finite = np.isfinite(x) & np.isfinite(y)
        applicable = _boolean(rows, "applicable", True)
        row = {"figure": stem, "step_index": step, **_counts(rows), "finite_pairs": int(finite.sum()), "applicable_count": int(applicable.sum()), "inapplicable_count": int((~applicable).sum())}
        row.update(_quantities(x, "x_"))
        row.update(_quantities(y, "y_"))
        if stem in BOUND_STEMS:
            usable = finite & applicable
            row.update(_quantities((x - y)[usable], "bound_slack_"))
            row.update(_quantities((y / x.where(x != 0))[usable], "error_bound_ratio_"))
            row["zero_bound_count"] = int((usable & x.eq(0)).sum())
            row["zero_error_count"] = int((usable & y.eq(0)).sum())
            for rho in RECOVERY_TOLERANCES:
                row[f"bound_le_{rho:g}_count"] = int((usable & (x <= rho)).sum())
        if stem == "proposition5_posterior_feedback":
            # Preserve the upstream resolved decision, raw estimated sign and
            # nonstrict event separately; never promote quadrature to a certificate.
            for column in ("direct_prop5_condition_status", "direct_prop5_gain_status", "direct_prop5_integral_status"):
                for value, count in rows[column].fillna("missing").value_counts().items():
                    row[column + "__" + str(value)] = int(count)
            row["estimated_strict_positive_count"] = int((finite & (x > 0)).sum())
            row["estimated_nonstrict_count"] = int((finite & (x >= 0)).sum())
            resolved = rows.marker_class.eq("observed") & finite
            gain_status = rows.direct_prop5_gain_status
            negative = gain_status.eq("negative")
            exact_zero = gain_status.isin(["zero", "arithmetic_zero"])
            # H can underflow to zero while the stable log-odds comparison
            # resolves a positive gain. Preserve H for display, but use the
            # measured stable sign when checking either implication.
            row["alleged_strict_implication_violations"] = int((resolved & (x > 0) & (negative | exact_zero)).sum())
            row["alleged_nonstrict_implication_violations"] = int((resolved & (x >= 0) & negative).sum())
            row["gain_underflow_count"] = int(_boolean(rows, "direct_prop5_gain_underflow").sum())
            row["certification_status"] = "not_certified_quadrature_estimate"
        output.append(row)
    return output


def _markers(rows, prefix):
    result = pd.Series("observed", index=rows.index, dtype=object)
    applicable = _boolean(rows, prefix + "_applicable", True)
    result.loc[~applicable] = "inapplicable"
    if prefix == "direct_lemma4":
        _require(rows, ["direct_lemma4_verification_source", "direct_lemma4_independent"], "matched_updates")
        independent = _boolean(rows, "direct_lemma4_independent")
        result.loc[applicable] = "constructed"
        result.loc[applicable & independent] = "independently_checked"
    if prefix == "direct_prop5":
        for column in (prefix + "_condition_status", prefix + "_gain_status", prefix + "_integral_status"):
            _require(rows, [column], "matched_updates")
            unresolved = rows[column].fillna("unavailable").astype(str).str.contains("unresolved|exhaust|unavailable|failed|pending|not_computed|missing", regex=True)
            result.loc[applicable & unresolved] = "numerically_unresolved"
    return result, applicable


def _pair_frame(rows, x, y, prefix=None):
    result = rows.copy()
    result["x"], result["y"] = _number(rows, x), _number(rows, y)
    if prefix:
        result["marker_class"], result["applicable"] = _markers(rows, prefix)
    else:
        result["marker_class"], result["applicable"] = "observed", True
    if "terminal_sscd" not in result:
        result["terminal_sscd"] = np.nan
    return result


def _wilson(failures, count):
    """Fixed 95% binomial interval; conditional on the pair, not pooled prompts."""
    if count == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = failures / count
    denominator = 1 + z * z / count
    midpoint = (p + z * z / (2 * count)) / denominator
    width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, midpoint - width), min(1.0, midpoint + width)


def build_direct_plot_inputs(tables, config, provenance):
    """Return ``frames, metadata``; auxiliary tables are analysis-owned outputs.

    Missing required forward-loss inputs are errors, never zero estimates.
    Numerical failure/unavailability is preserved for other statement entries.
    Compatible legacy compact frames may be supplied separately by the caller.
    """
    config = config.get("scientific_config", config)
    frames, figures, summary_rows, auxiliary = {}, {}, [], {}
    provenance = dict(provenance or {})
    law_hashes = set()
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and "reference_law_hash" in table:
            law_hashes.update(str(v) for v in table.reference_law_hash.dropna().unique() if str(v) not in {"", "fixed_pair_target"})
    if len(law_hashes) > 1:
        raise TheoryError("Direct comparisons mix reference-law hashes")
    expected_law = provenance.get("reference_law_hash")
    if expected_law and law_hashes and law_hashes != {expected_law}:
        raise TheoryError("Scalar reference law differs from publication provenance")
    law_metadata = provenance.get("reference_law", {})
    reference_provenance = {
        "reference_law": config.get("reference_law", "cached-targets"),
        "reference_law_hash": expected_law or next(iter(law_hashes), None),
        "scope": law_metadata.get("reference_law_scope", law_metadata.get("scope")) if isinstance(law_metadata, dict) else None,
        "mean_definition": "Exact declared-law weighted mean of latent atoms, never a model-output center",
        "single_target_assumption": "Assumed experimental idealization",
        "authoritative_provenance": "run_config.json:provenance",
        "manuscript_sha256": provenance.get("manuscript_sha256"),
    }

    def save(stem, frame, *, metadata=None, reason=None, status=None):
        frame = frame.copy()
        if "marker_class" not in frame and "x" in frame:
            frame["marker_class"] = "observed"
        finite = (np.isfinite(_number(frame, "x")) & np.isfinite(_number(frame, "y"))) if "x" in frame else np.isfinite(_number(frame, "median"))
        actual_status = status or ("available" if finite.any() else "unavailable")
        entry = {"status": actual_status, "reason": reason if reason else (None if actual_status == "available" else "No finite direct comparison is available from the recorded measurements"), "counts": _counts(frame), "status_counts": _statuses(frame), "reference_provenance": reference_provenance, "formula_version": DIRECT_FIGURE_VERSION, "single_target_idealization": "assumed, not inferred from SSCD", "input_sources": sorted(str(v) for v in frame.get("input_source", pd.Series(dtype=str)).dropna().unique()), "quantile_policy": "linear descriptive sample quantiles; no significance test"}
        if "step_index" in frame and "snr" in frame:
            initial = _number(frame.loc[_number(frame, "step_index") == 0], "snr")
            initial = initial[np.isfinite(initial)]
            if len(initial):
                entry["actual_initial_snr"] = float(initial.iloc[0])
        entry.update(metadata or {})
        frames[stem], figures[stem] = frame, entry
        if "x" in frame:
            summary_rows.extend(_scatter_summary(stem, frame))

    loss = tables.get("forward_loss_summary", pd.DataFrame()).copy()
    gaussian = tables.get("gaussian_conditional", pd.DataFrame()).copy()
    if loss.empty or gaussian.empty:
        raise TheoryError("Missing required independent forward loss or Gaussian conditional measurements; run direct analysis with --num-loss-seeds and --loss-seed")
    _unique(loss, PAIR_STEP, "forward_loss_summary")
    _unique(gaussian, PAIR_STEP + ["seed"], "gaussian_conditional")
    _source(loss, "forward_target", "forward_loss_summary")
    _source(gaussian, "gaussian_probe", "gaussian_conditional")
    loss_columns = ["draw_count", "loss_mean_squared_l2", "loss_mc_se_squared_l2", "forward_clean_error_mean_squared_l2", "latent_dimension", "snr", "pinsker_tv_bound"]
    _require(loss, loss_columns, "forward_loss_summary")
    _require(gaussian, ["conditional_error_l2", "conditional_error_squared_l2", "conditional_error_rmse", "latent_dimension", "snr"], "gaussian_conditional")
    loss = loss.loc[_number(loss, "step_index") == 0].copy()
    gaussian = gaussian.loc[_number(gaussian, "step_index") == 0].copy()
    if loss.empty or gaussian.empty:
        raise TheoryError("Required actual-initial-timestep loss/Gaussian measurements are absent")
    loss_pairs = set(map(tuple, loss[PAIR_KEYS].astype(str).to_numpy()))
    gaussian_pairs = set(map(tuple, gaussian[PAIR_KEYS].astype(str).to_numpy()))
    if loss_pairs != gaussian_pairs:
        raise TheoryError("Initial Gaussian and independent forward-loss pair populations differ")
    expected_seeds = {str(seed) for seed in range(int(config["num_seeds"]))}
    for _, rows in gaussian.groupby(PAIR_KEYS, sort=False, dropna=False):
        if set(rows.seed.astype(str)) != expected_seeds:
            raise TheoryError("Initial Gaussian measurements do not retain every evaluation seed")
    if (loss.draw_count <= 0).any() or (~np.isfinite(_number(loss, "loss_mean_squared_l2"))).any() or (_number(loss, "loss_mean_squared_l2") < 0).any():
        raise TheoryError("Required forward loss is missing, failed, or a placeholder rather than a measured nonnegative draw mean")
    joined = gaussian.merge(loss[PAIR_STEP + loss_columns], on=PAIR_STEP, how="left", validate="many_to_one", suffixes=("", "_forward"), indicator=True)
    if not joined._merge.eq("both").all():
        raise TheoryError("Gaussian pair is missing its independent forward-loss estimate")
    joined = joined.drop(columns="_merge")
    if not np.allclose(_number(joined, "snr"), _number(joined, "snr_forward"), rtol=1e-12, atol=0) or not joined.latent_dimension.eq(joined.latent_dimension_forward).all():
        raise TheoryError("Forward/Gaussian native noise or latent dimension differs")
    d, snr = _number(joined, "latent_dimension"), _number(joined, "snr")
    if (d <= 0).any() or (snr <= 0).any() or not np.isfinite(d * snr).all():
        raise TheoryError("Initial direct measurement needs finite positive dimension and SNR")
    if "terminal_sscd" in joined:
        matched = _boolean(joined, "terminal_sscd_matched_initialization")
        joined.loc[~matched, "terminal_sscd"] = np.nan
    joined["loss_scale_mse"] = _number(joined, "loss_mean_squared_l2") / (d * snr)
    joined["loss_scale_rmse"] = np.sqrt(joined.loss_scale_mse)
    joined["gaussian_error_mse"] = _number(joined, "conditional_error_squared_l2") / d
    comparison_metadata = {"loss_statistical_unit": "Independent pair-level forward-draw mean, repeated over Gaussian seeds without increasing the loss effective sample size", "gaussian_statistical_unit": "Shared unique seed bank; prompts are not independent through shared Gaussian inputs", "forward_input_source": "forward_target", "outcome_input_source": "gaussian_probe"}
    save("theorem1_loss_recovery", _pair_frame(joined, "loss_scale_rmse", "conditional_error_rmse"), metadata=comparison_metadata)
    save("theorem1_squared_loss_recovery", _pair_frame(joined, "loss_scale_mse", "gaussian_error_mse"), metadata={**comparison_metadata, "normalization": "Squared L2/d on both axes"})
    forward = loss.copy()
    forward["forward_loss_scale_mse"] = _number(forward, "loss_mean_squared_l2") / (_number(forward, "latent_dimension") * _number(forward, "snr"))
    forward["forward_error_mse"] = _number(forward, "forward_clean_error_mean_squared_l2") / _number(forward, "latent_dimension")
    save("theorem1_forward_identity", _pair_frame(forward, "forward_loss_scale_mse", "forward_error_mse"), metadata={"normalization": "Squared L2/d; independently implemented forward loss and clean-error means", "statistical_unit": "One pair-level mean over forward draws"})
    gaussian_summary, transfer = [], []
    for keys, rows in joined.groupby(PAIR_STEP, sort=False, dropna=False):
        identity = dict(zip(PAIR_STEP, keys))
        first = rows.iloc[0]
        errors = _number(rows, "conditional_error_rmse")
        squares = _number(rows, "gaussian_error_mse")
        valid = np.isfinite(errors) & np.isfinite(squares)
        n = int(valid.sum())
        record = {**identity, "total_gaussian_count": len(rows), "finite_gaussian_count": n, "loss_draw_count": int(first.draw_count), "loss_mc_se_squared_l2": float(first.loss_mc_se_squared_l2)}
        record.update(_quantities(errors, "gaussian_error_rmse_"))
        record["gaussian_mean_squared_error_per_dimension"] = float(squares[valid].mean()) if n else np.nan
        record["gaussian_root_mean_squared_error"] = math.sqrt(record["gaussian_mean_squared_error_per_dimension"]) if n else np.nan
        gaussian_summary.append(record)
        delta = float(first.pinsker_tv_bound)
        if not math.isfinite(delta) or not 0 <= delta <= 1:
            raise TheoryError("Missing or invalid saved Pinsker bound; delta cannot be replaced by zero")
        for rho in RECOVERY_TOLERANCES:
            failures = int((errors[valid] > rho).sum())
            low, high = _wilson(failures, n)
            bound = float(first.loss_scale_mse) / rho**2 + delta
            transfer.append({**identity, "rho": rho, "raw_tolerance_l2": rho * math.sqrt(float(first.latent_dimension)), "x": min(1.0, bound), "y": failures / n if n else np.nan, "bound_unclipped": bound, "bound_clipped": min(1.0, bound), "saturated_at_one": bound >= 1, "pinsker_tv_bound": delta, "loss_mc_se_squared_l2": float(first.loss_mc_se_squared_l2), "bound_mc_se": float(first.loss_mc_se_squared_l2) / (float(first.snr) * float(first.latent_dimension) * rho**2), "frequency_low": low, "frequency_high": high, "failure_count": failures, "gaussian_count": n, "missing_gaussian_count": int(len(rows) - n), "loss_draw_count": int(first.draw_count), "input_source": "gaussian_probe", "marker_class": "observed", "applicable": True})
    transfer = pd.DataFrame(transfer)
    save("theorem1_gaussian_transfer", transfer, metadata={"tolerance_grid_rmse": list(RECOVERY_TOLERANCES), "saturation_to_one_count": int(transfer.saturated_at_one.sum()), "band_definition": "Fixed 95% Wilson interval for within-pair Gaussian frequency; shared prompts/seeds are not pooled for inference", "interpretation": "Plug-in empirical check, not a finite-sample statistical certificate"})
    auxiliary["theorem1_gaussian_summary"] = pd.DataFrame(gaussian_summary)

    reference = tables.get("gaussian_reference", pd.DataFrame()).copy()
    if reference.empty:
        raise TheoryError("Missing required unique-seed native Gaussian baseline sweep")
    _unique(reference, GAUSSIAN_KEYS, "gaussian_reference")
    _source(reference, "gaussian_probe", "gaussian_reference")
    for _, rows in reference.groupby(["run_id", "step_index"], sort=False, dropna=False):
        if set(rows.seed.astype(str)) != expected_seeds:
            raise TheoryError("Gaussian reference sweep does not retain the fixed unique evaluation seed bank")
    metrics = {"reference": "reference_mean_error_rmse", "learned": "learned_mean_error_rmse", "reference_error": "unconditional_reference_error_rmse"}
    _require(reference, [*metrics.values(), "baseline_bound_rmse", "snr"], "gaussian_reference")
    curves = []
    for step, rows in reference.groupby("step_index", sort=True):
        noises = _number(rows, "snr")
        if len(noises.unique()) != 1 or not np.isfinite(noises).all() or (noises <= 0).any():
            raise TheoryError("Gaussian baseline native timestep has inconsistent SNR")
        for metric, column in metrics.items():
            values = _number(rows, column)
            stats = _quantities(values, "")
            curves.append({"metric": metric, "step_index": step, "timestep": rows.timestep.iloc[0] if "timestep" in rows else np.nan, "snr": float(noises.iloc[0]), "median": stats.get("median", np.nan), "q25": stats.get("q25", np.nan), "q75": stats.get("q75", np.nan), "minimum": stats.get("minimum", np.nan), "maximum": stats.get("maximum", np.nan), "seed_count": len(rows), "finite_count": stats["finite_count"], "missing_count": len(rows) - stats["finite_count"], "input_source": "gaussian_probe"})
    save("lemma2_unconditional_baseline", pd.DataFrame(curves), metadata={"weighting": "Equal mass over unique run/seed probes at each native timestep; no outcome split", "band_definition": "Descriptive interquartile range of unique Gaussian seeds"})
    save("lemma2_reference_to_learned_bound", _pair_frame(reference, "baseline_bound_rmse", "learned_mean_error_rmse"))

    for stem, (table, x, y, prefix) in SCATTER_SPECS.items():
        rows = tables.get(table, pd.DataFrame()).copy()
        if rows.empty or x not in rows or y not in rows:
            reason = provenance.get("table_statuses", {}).get(table, {}).get("reason", "Required direct fields are unavailable in " + table)
            save(stem, pd.DataFrame(columns=["x", "y", "marker_class"]), reason=reason)
            continue
        _source(rows, "generated_state", table)
        _require(rows, PAIR_KEYS + ["seed", "step_index"], table)
        _unique(rows, PAIR_KEYS + ["seed", "step_index"], table)
        excluded = 0
        if prefix == "direct_prop5":
            eligible = _boolean(rows, "direct_prop5_applicable")
            excluded = int((~eligible).sum())
            rows = rows.loc[eligible].copy()
        frame = _pair_frame(rows, x, y, prefix)
        metadata = {"excluded_structural_rows": excluded}
        if prefix == "direct_prop5":
            metadata.update(symlog_linthresh=0.001, numerical_uncertainty="Original integration error, source sensitivity, sign statuses and exhausted budgets remain saved per row", certification_status="not_certified_quadrature_estimate")
        if prefix == "direct_theorem7":
            applicable = _boolean(frame, "applicable")
            metadata["clean_update_prerequisite"] = {"applicable": int(applicable.sum()), "total": len(frame), "inapplicable": int((~applicable).sum())}
            metadata["comparison_status"] = f"{int(applicable.sum())}/{len(frame)} satisfy the clean-update prerequisite"
            metadata["theorem_claim_scope"] = "Only applicable rows; no theorem violation or certification is assigned to inapplicable points"
        save(stem, frame, metadata=metadata)

    excess = tables.get("forward_unconditional_loss", pd.DataFrame()).copy()
    if not config.get("measure_unconditional_loss", config.get("loss_timesteps") == "saved"):
        save("lemma2_excess_loss_gaussian_error", pd.DataFrame(columns=["x", "y", "marker_class"]), status="not_applicable", reason="Marginal forward excess-loss measurements were not requested")
    elif excess.empty:
        save("lemma2_excess_loss_gaussian_error", pd.DataFrame(columns=["x", "y", "marker_class"]), reason="Requested direct marginal forward excess loss is unavailable")
    else:
        _source(excess, "forward_marginal", "forward_unconditional_loss")
        _require(excess, ["run_id", "step_index", "draw_index", "excess_loss_squared_l2", "snr", "latent_dimension"], "forward_unconditional_loss")
        _unique(excess, ["run_id", "step_index", "draw_index"], "forward_unconditional_loss")
        excess = excess.loc[_number(excess, "step_index") == 0]
        estimates = excess.groupby(["run_id", "step_index"], as_index=False).agg(excess_mean_squared_l2=("excess_loss_squared_l2", "mean"), excess_draw_count=("excess_loss_squared_l2", "count"))
        values = reference.loc[_number(reference, "step_index") == 0].merge(estimates, on=["run_id", "step_index"], how="left", validate="many_to_one")
        values["excess_scale_mse"] = _number(values, "excess_mean_squared_l2") / (_number(values, "latent_dimension") * _number(values, "snr"))
        values["gaussian_reference_error_mse"] = _number(values, "unconditional_reference_error_rmse") ** 2
        save("lemma2_excess_loss_gaussian_error", _pair_frame(values, "excess_scale_mse", "gaussian_reference_error_mse"), metadata={"input_sources": ["forward_marginal", "gaussian_probe"], "interpretation": "Direct excess-loss mean and Gaussian reference errors have different input laws; no equality line"})

    tolerance = config.get("target_error_tolerance")
    terminal = frames["theorem7_final_reproduction"]
    if tolerance is None:
        save("theorem7_certification", pd.DataFrame(columns=["x", "y", "marker_class"]), status="not_applicable", reason="No independent raw latent reproduction tolerance was declared")
    else:
        applicable = _boolean(terminal, "applicable")
        rows = terminal.loc[applicable]
        if rows.empty:
            save("theorem7_certification", pd.DataFrame(columns=["x", "y", "marker_class"]), status="not_applicable", reason="0 applicable rows satisfy the clean-update prerequisite; no theorem certification population")
        else:
            _require(rows, ["direct_theorem7_bound_l2", "direct_theorem7_endpoint_error_l2", "direct_theorem7_condition_certified", "direct_theorem7_actual_proximity"], "terminal")
            bound, error = _number(rows, "direct_theorem7_bound_l2"), _number(rows, "direct_theorem7_endpoint_error_l2")
            finite = np.isfinite(bound) & np.isfinite(error)
            n = int(finite.sum())
            certified = _boolean(rows, "direct_theorem7_condition_certified")
            proximity = _boolean(rows, "direct_theorem7_actual_proximity")
            cert = pd.DataFrame([{"x": float(certified[finite].mean()) if n else np.nan, "y": float(proximity[finite].mean()) if n else np.nan, "raw_tolerance_l2": tolerance, "applicable_count": len(rows), "finite_count": n, "missing_count": len(rows) - n, "certified_count": int(certified[finite].sum()), "actual_proximity_count": int(proximity[finite].sum()), "marker_class": "observed", "input_source": "generated_state"}])
            save("theorem7_certification", cert, metadata={"independent_latent_tolerance": tolerance, "denominator": "Applicable finite paired bound/endpoint observations; missing counts remain explicit"})
    # Former behavioral figures remain separately named diagnostics. Reuse
    # compatible scalars without another trajectory read or a reference-center conversion.
    for stem in LEGACY_STEMS:
        save(stem, pd.DataFrame(columns=["x", "y", "marker_class"]), reason="Legacy diagnostic inputs are not provided by this direct measurement contract; no direct statement is substituted")
    initial = tables.get("initial", pd.DataFrame())
    if {"direct_unconditional_target_error_rmse", "direct_conditional_error_rmse"}.issubset(initial):
        save("initial_recovery", _pair_frame(initial, "direct_unconditional_target_error_rmse", "direct_conditional_error_rmse"), metadata={"interpretation": "Retained behavioral target-error plane; not the independent forward-loss comparison"})
    trajectory = tables.get("trajectory", pd.DataFrame())
    if {"terminal_sscd", "direct_conditional_error_rmse", "direct_unconditional_target_error_rmse", *PAIR_KEYS, "seed", "step_index", "snr"}.issubset(trajectory):
        records = []
        for group, chosen in _outcomes(trajectory):
            for step, rows in chosen.groupby("step_index", sort=True):
                weights = 1.0 / rows.groupby(PAIR_KEYS, dropna=False).seed.transform("size")
                for branch, field in (("conditional", "direct_conditional_error_rmse"), ("unconditional", "direct_unconditional_target_error_rmse")):
                    values = _number(rows, field)
                    q25, median, q75 = weighted_quantiles(values, weights)
                    finite = values[np.isfinite(values)]
                    records.append({"group": group, "branch": branch, "step_index": step, "snr": float(rows.snr.iloc[0]), "median": median, "q25": q25, "q75": q75, "minimum": float(finite.min()) if len(finite) else np.nan, "maximum": float(finite.max()) if len(finite) else np.nan, "input_source": "generated_state", "seed_count": len(rows), "prompt_count": len(rows[PAIR_KEYS].drop_duplicates())})
        save("target_synchronization", pd.DataFrame(records), metadata={"weighting": "Each complete prompt has equal mass within the saved outcome cohort; equal mass per eligible seed", "interpretation": "Retained branch target-error trajectories, not the two sides of Lemma 6"})
    matched = tables.get("matched_updates", pd.DataFrame())
    if {"direct_prop5_applicable", "direct_prop5_gain_status", "terminal_sscd", *PAIR_KEYS, "seed", "step_index", "snr"}.issubset(matched):
        records = []
        eligible = matched.loc[_boolean(matched, "direct_prop5_applicable")]
        for group, chosen in _outcomes(eligible):
            for step, rows in chosen.groupby("step_index", sort=True):
                weights = 1.0 / rows.groupby(PAIR_KEYS, dropna=False).seed.transform("size")
                denominator = float(weights.sum())
                positive = rows.direct_prop5_gain_status.eq("positive")
                unresolved = rows.direct_prop5_gain_status.fillna("unavailable").str.contains("unresolved|unavailable|failed|missing", regex=True)
                fraction = float(weights[positive].sum()) / denominator
                records.append({"group": group, "step_index": step, "snr": float(rows.snr.iloc[0]), "fraction": fraction, "upper_fraction": fraction + float(weights[unresolved].sum()) / denominator, "input_source": "generated_state", "eligible_rows": len(rows), "denominator_weight": denominator})
        frame = pd.DataFrame(records)
        if not frame.empty:
            # save() normally recognizes scatter or curve values; fractions
            # explicitly declare availability on their validated saved population.
            save("posterior_feedback_positive_fraction", frame, status="available", metadata={"band_definition": "Unresolved gain-sign mass on the same structurally eligible weighted denominator", "interpretation": "Secondary descriptive frequency, not the Proposition 5 condition-versus-conclusion comparison"})
    auxiliary["statement_summary"] = pd.DataFrame(summary_rows)
    return frames, {"figures": figures, "audit": {"version": DIRECT_FIGURE_VERSION, "blocking": False, "reference_law_hashes": sorted(law_hashes), "gaussian_recovery_tolerances_rmse": list(RECOVERY_TOLERANCES), "single_target_assumption": "assumed experimental idealization"}, "auxiliary_tables": auxiliary}


def _outcomes(frame):
    score = _number(frame, "terminal_sscd")
    yield "SSCD > 0.75", frame.loc[np.isfinite(score) & (score > .75)]
    yield "SSCD <= 0.75", frame.loc[np.isfinite(score) & (score <= .75)]
