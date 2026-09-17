"""Analysis-only scalar assembly for the fixed four-stage mechanism suite.

No tensor, checkpoint, posterior, integration or filesystem reads occur here.
All statistical reductions are saved before the sole compact-data renderer runs.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd

from utils.common.io import json_value

from .contracts import TheoryError
from .direct_figures import PAIR_KEYS, _boolean, _counts, _number, _outcomes, _pair_frame, _require, _statuses
from .evidence_figures import (
    BOOTSTRAP_POLICY, BOOTSTRAP_SEED, DIRECTIONAL_VARIATION_DEFINITION, EVIDENCE_FORMULA_VERSION, SAMPLE_KEYS, _curve_summary, _ecdf, _interval,
    _quantile_description, _segments, _weights, build_evidence_plot_inputs,
)
from .paper_registry import GROUPS, paper_registry
from .summaries import weighted_quantiles

FOUR_STAGE_FORMULA_VERSION = "four-stage-scalar-inputs-projected-gap-error-1"
DOSE_LINEAR_THRESHOLD = 1e-3
MOTION_FIELDS = {
    "conditional": "gap_motion_conditional", "unconditional": "gap_motion_unconditional",
    "quadratic": "gap_motion_quadratic", "change": "gap_squared_change",
}


def _population(rows):
    result = _counts(rows)
    if "target_id" in rows:
        result["targets"] = int(rows.target_id.nunique(dropna=False))
    return result


def _schedule(rows, steps):
    columns = [name for name in ("step_index", "timestep", "native_timestep", "snr", "alpha", "sigma", "destination_timestep", "destination_alpha", "destination_sigma") if name in rows]
    schedule = rows[columns].drop_duplicates().sort_values("step_index")
    if schedule.step_index.duplicated().any():
        raise TheoryError("One chronological prediction index has inconsistent saved schedule metadata")
    if not schedule.step_index.isin(range(steps)).all():
        raise TheoryError("A terminal output was mislabeled as another branch prediction")
    return schedule.to_dict("records")


def _chronological(rows, fields, steps):
    summaries = []
    for group, chosen in _outcomes(rows):
        if chosen.empty:
            continue
        frame = _segments(_curve_summary(chosen, fields, group=group))
        frame["normalized_progress"] = frame.step_index / (steps - 1) if steps > 1 else 0.
        summaries.append(frame)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def _prompt_scalar_inputs(trajectory, initial, config, save, auxiliary, *,
                          stem="branch_gap_per_prompt", source_column="direct_lemma6_gap_rmse",
                          value_column="mean_gap_rmse", value_domain="nonnegative",
                          prediction_domain="pre_update_predictions", row_eligibility_column=None,
                          audit_status_columns=()):
    """Average saved per-seed scalars on a fixed full cohort for each figure."""
    probability = value_domain == "probability"
    if value_domain not in {"nonnegative", "probability"}:
        raise TheoryError("Unknown per-prompt trajectory scalar domain")
    legacy_gap = stem == "branch_gap_per_prompt"
    description = "branch-gap" if legacy_gap else stem.replace("_", "-")
    missing_source = source_column not in trajectory
    if missing_source and not legacy_gap:
        # A missing measurement column creates no plotted values. NaNs keep the
        # existing cohort audit explicit for every affected prompt-target pair.
        trajectory = trajectory.copy()
        trajectory[source_column] = np.nan
    steps, seed_count = int(config["num_inference_steps"]), int(config["num_seeds"])
    if prediction_domain not in {"pre_update_predictions", "positive_noise_transitions"}:
        raise TheoryError("Unknown per-prompt prediction domain")
    transition_domain = prediction_domain == "positive_noise_transitions"
    curve_steps = steps - int(transition_domain)
    expected_seeds, expected_steps = set(range(seed_count)), set(range(curve_steps))
    required = SAMPLE_KEYS + ["step_index", source_column, "terminal_sscd", "latent_dimension"]
    _require(trajectory, required, "per-prompt " + description + " trajectory scalars")
    _require(initial, SAMPLE_KEYS + ["terminal_sscd"], "per-prompt paired outcomes")
    trajectory, initial = trajectory.copy(), initial.copy()
    excluded_terminal_rows = 0
    if transition_domain:
        # The revised directional variation excludes t=1, even for positive terminal sampler noise.
        terminal = _number(trajectory, "step_index").eq(steps - 1)
        excluded_terminal_rows = int(terminal.sum())
        trajectory = trajectory.loc[~terminal].copy()
    for name, rows in (("trajectory", trajectory), ("initial", initial)):
        if rows[SAMPLE_KEYS].isna().any().any():
            raise TheoryError("Missing per-prompt " + description + " identity in " + name)
        for key in PAIR_KEYS:
            rows[key] = rows[key].astype(str)
        seeds = _number(rows, "seed")
        if not (np.isfinite(seeds) & seeds.eq(np.floor(seeds))).all():
            raise TheoryError("Per-prompt " + description + " seeds must be integer identities")
        rows["seed"] = seeds.astype(int)
        keys = SAMPLE_KEYS + (["step_index"] if name == "trajectory" else [])
        if rows.duplicated(keys).any():
            raise TheoryError("Duplicate per-prompt " + description + " seed/step observation in " + name)
    raw_values = _number(trajectory, source_column)
    if probability:
        valid_log_probability = raw_values.le(0) & ~raw_values.isna()
        values = pd.Series(np.nan, index=trajectory.index, dtype=float)
        with np.errstate(under="ignore"):
            values.loc[valid_log_probability] = np.exp(raw_values.loc[valid_log_probability])
        trajectory["_finite_log_underflow"] = np.isfinite(raw_values) & values.eq(0)
        trajectory["_negative_infinite_log"] = raw_values.eq(-np.inf)
    else:
        values = raw_values
    trajectory["_prompt_scalar_value"] = values
    pairs = pd.concat([initial[PAIR_KEYS], trajectory[PAIR_KEYS]], ignore_index=True).drop_duplicates().sort_values(PAIR_KEYS)
    trajectories = {key: rows for key, rows in trajectory.groupby(PAIR_KEYS, sort=False, dropna=False)}
    outcomes = {key: rows for key, rows in initial.groupby(PAIR_KEYS, sort=False, dropna=False)}
    seed_json = json.dumps(sorted(expected_seeds))
    records, audits = [], []
    for identity in pairs.itertuples(index=False, name=None):
        rows = trajectories.get(identity, trajectory.iloc[:0])
        scores = outcomes.get(identity, initial.iloc[:0])
        reasons = ["missing_source_column:" + source_column] if missing_source else []
        if set(scores.seed) != expected_seeds:
            reasons.append("incomplete_or_unexpected_terminal_seed_cohort")
        if set(rows.seed) != expected_seeds:
            reasons.append("incomplete_or_unexpected_trajectory_seed_cohort")
        indices = _number(rows, "step_index")
        valid_indices = np.isfinite(indices) & indices.eq(np.floor(indices)) & indices.isin(expected_steps)
        if not valid_indices.all():
            reasons.append("invalid_prediction_step")
        complete_grid = len(rows) == seed_count * curve_steps and all(
            set(_number(group, "step_index")) == expected_steps for _, group in rows.groupby("seed", sort=False))
        if not complete_grid:
            reasons.append("incomplete_seed_step_grid")
        invalid_measurements = int((~_boolean(rows, row_eligibility_column)).sum()) if row_eligibility_column else 0
        if invalid_measurements:
            reasons.append("inapplicable_or_unavailable_saved_measurement")
        status_audit = {column + "_counts_json": json.dumps(
            {str(value): int(count) for value, count in rows[column].value_counts(dropna=False).items()}, sort_keys=True)
            for column in audit_status_columns if column in rows}
        values, colors = _number(rows, "_prompt_scalar_value"), _number(scores, "terminal_sscd")
        invalid_values = int((~np.isfinite(values) | values.lt(0) | (values.gt(1) if probability else False)).sum())
        invalid_colors = int((~np.isfinite(colors)).sum())
        if invalid_values:
            reasons.append("invalid_saved_target_log_probability" if probability else
                           "nonfinite_or_negative_saved_gap" if legacy_gap else
                           "nonfinite_or_negative_saved_variation" if stem == "reference_variation_per_prompt" else
                           "nonfinite_or_negative_saved_error")
        if invalid_colors:
            reasons.append("nonfinite_terminal_sscd")
        dimensions = _number(rows, "latent_dimension")
        dimension_valid = (not rows.empty and np.isfinite(dimensions).all() and dimensions.gt(0).all()
                           and dimensions.eq(np.floor(dimensions)).all() and dimensions.nunique() == 1)
        if not dimension_valid:
            reasons.append("missing_or_inconsistent_latent_dimension")
        expected_color = rows.seed.map(dict(zip(scores.seed, colors)))
        saved_colors = _number(rows, "terminal_sscd")
        mismatches = int((~np.isfinite(saved_colors) | ~saved_colors.eq(expected_color)).sum())
        if mismatches:
            reasons.append("trajectory_terminal_sscd_differs_from_paired_outcome")
        mean_color = float(colors.mean()) if len(colors) == seed_count and np.isfinite(colors).all() else math.nan
        eligible = not reasons
        probability_audit = {}
        if probability:
            underflow = rows.loc[rows["_finite_log_underflow"]]
            underflow_records = [
                {"seed": int(seed), "step_index": float(step) if math.isfinite(float(step)) else None,
                 source_column: float(log_probability)}
                for seed, step, log_probability in zip(underflow.seed, _number(underflow, "step_index"),
                                                        _number(underflow, source_column), strict=True)]
            probability_audit = {
                "finite_log_probability_underflow_rows": int(len(underflow)),
                "negative_infinite_log_probability_rows": int(rows["_negative_infinite_log"].sum()),
                "finite_log_probability_underflow_observations_json": json.dumps(underflow_records, allow_nan=False),
            }
        audits.append({**dict(zip(PAIR_KEYS, identity)), "eligible": eligible,
            "reason": ";".join(reasons) if reasons else "complete_fixed_seed_cohort",
            "expected_seed_count": seed_count, "trajectory_seed_count": int(rows.seed.nunique()),
            "terminal_seed_count": int(scores.seed.nunique()), "expected_rows": seed_count * curve_steps,
            "observed_rows": len(rows),
            ("invalid_gap_rows" if legacy_gap else "invalid_value_rows"): invalid_values,
            **probability_audit, **status_audit,
            **({"inapplicable_or_unavailable_measurement_rows": invalid_measurements} if row_eligibility_column else {}),
            "invalid_terminal_sscd_rows": invalid_colors, "terminal_sscd_mismatch_rows": mismatches,
            "seed_ids_json": seed_json, "mean_terminal_sscd": mean_color})
        if not eligible:
            continue
        for step, group in rows.groupby("step_index", sort=True):
            records.append({**dict(zip(PAIR_KEYS, identity)), "step_index": int(step),
                value_column: float(_number(group, "_prompt_scalar_value").mean()),
                "mean_terminal_sscd": mean_color, "seed_count": seed_count,
                "seed_ids_json": seed_json, "latent_dimension": int(dimensions.iloc[0]),
                "cohort_complete": True})
    columns = PAIR_KEYS + ["step_index", value_column, "mean_terminal_sscd", "seed_count", "seed_ids_json", "latent_dimension", "cohort_complete"]
    frame = pd.DataFrame(records, columns=columns)
    cohort = pd.DataFrame(audits)
    auxiliary[stem + "_cohort"] = cohort
    plotted = int(cohort.eligible.sum()) if not cohort.empty else 0
    reason = None if plotted else "No prompt-target pair has the full declared seed/step cohort and valid saved measurement/outcome scalars"
    if missing_source:
        reason = "Missing saved column " + source_column + "; run --recompute-experiments to rebuild analysis scalars"
    probability_metadata = {}
    if probability:
        probability_metadata = {
            "probability_conversion": "Arithmetic mean of exp(each saved per-seed log p), never exp(mean log p); -inf is retained as p=0. Positive log p, +inf and NaN are invalid.",
            "probability_underflow": "Finite log p that exponentiates to zero is retained with its seed, step and log value in the cohort audit; this representational zero is distinct from saved -inf.",
            "finite_log_probability_underflow_rows": int(cohort.finite_log_probability_underflow_rows.sum()) if not cohort.empty else 0,
            "negative_infinite_log_probability_rows": int(cohort.negative_infinite_log_probability_rows.sum()) if not cohort.empty else 0,
        }
    save(stem, frame, status="available" if plotted else "unavailable",
         reason=reason, value_column=value_column, value_domain=value_domain,
         source_column=source_column, **probability_metadata,
         formula_version="branch-gap-per-prompt-1" if legacy_gap else "prompt-trajectory-scalars-1",
         prediction_steps=steps, prediction_domain=prediction_domain,
         **({"excluded_terminal_prediction_rows": excluded_terminal_rows} if transition_domain else {}),
         initialization_index=0, expected_seed_count=seed_count,
         expected_seed_ids=list(range(seed_count)), pair_identity=PAIR_KEYS,
         aggregation=("Arithmetic mean of each seed's exp(direct_target_log_probability); not exp of the mean log probability" if probability
                      else "Arithmetic mean of each seed's saved " + source_column + "; not the norm of a mean vector"),
         normalization=("Dimensionless per-seed probabilities averaged directly; no norm or dimension normalization" if probability
                        else "Saved L2/sqrt(d) values averaged directly; no second normalization"),
         color_population="Arithmetic mean terminal paired-target SSCD across the same complete seed cohort; fixed for every step of a curve",
         cohort_policy=("All declared seeds and all manuscript transitions t=2..T required per prompt-target pair; incomplete pairs excluded once for the entire curve and retained in the cohort audit"
                        if transition_domain else "All declared seeds and all pre-update steps required per prompt-target pair; incomplete pairs excluded once for the entire curve and retained in the cohort audit"),
         cohort_audit_table="audit_data/" + stem + "_cohort.csv",
         counts={"pairs": len(pairs), "plotted_pairs": plotted, "excluded_pairs": len(pairs) - plotted,
                 "seeds_per_pair": seed_count, "rows": len(frame)},
         exclusion_reason_counts={str(reason): int(count) for reason, count in cohort.loc[~cohort.eligible, "reason"].value_counts().items()} if not cohort.empty else {},
         terminal_output_prediction=False)


def _prompt_gap_inputs(trajectory, initial, config, save, auxiliary):
    """Retain the original branch-gap figure and complete-cohort contract."""
    _prompt_scalar_inputs(trajectory, initial, config, save, auxiliary)


def _prompt_reference_inputs(trajectory, initial, config, save, auxiliary):
    """Saved reference quantities on the actual generated trajectory states.

    The reference branch gap is bar_x_t(c) - bar_x_t(empty). Under the
    declared single-target conditional reference, bar_x_t(c) = x_star.
    The existing reference-target error stores the norm of the negative
    of that vector, so its saved L2/sqrt(d) value is exactly the needed norm.
    """
    for stem, source_column, value_column, domain in (
        ("conditional_reference_error_per_prompt", "direct_conditional_error_rmse", "mean_conditional_error_rmse", "nonnegative"),
        ("unconditional_reference_error_per_prompt", "direct_unconditional_reference_error_rmse", "mean_unconditional_reference_error_rmse", "nonnegative"),
        ("target_probability_per_prompt", "direct_target_log_probability", "mean_target_probability", "probability"),
        ("reference_branch_gap_per_prompt", "direct_reference_target_error_rmse", "mean_reference_gap_rmse", "nonnegative"),
    ):
        _prompt_scalar_inputs(trajectory, initial, config, save, auxiliary, stem=stem,
                              source_column=source_column, value_column=value_column, value_domain=domain)


def _prompt_variation_inputs(matched, initial, config, save, auxiliary):
    """Reduce saved positive parts of signed directional integrals, never norm diagnostics."""
    stem = "reference_variation_per_prompt"
    value_column, source_column = "mean_reference_variation_rmse", "direct_prop5_variation_rmse"
    steps, seed_count = int(config["num_inference_steps"]), int(config["num_seeds"])
    columns = PAIR_KEYS + ["step_index", value_column, "mean_terminal_sscd", "seed_count",
                          "seed_ids_json", "latent_dimension", "cohort_complete"]
    status_columns = ("direct_prop5_integral_status", "direct_prop5_condition_status",
                      "direct_prop5_quadrature_budget_exhausted", "condition_sign_status",
                      "condition_value_status", "quadrature_status", "numerical_stopping_reason",
                      "direct_prop5_measurement_contract", "direct_prop5_variation_definition",
                      "direct_prop5_variation_direction_status", "variation_value_identity_status")
    required = (source_column, "direct_prop5_applicable", "direct_prop5_integral_status",
                "direct_prop5_measurement_contract", "direct_prop5_variation_definition",
                "direct_prop5_variation_signed_integral_l2")
    missing = [name for name in required if name not in matched]
    source = matched.copy()
    source["_variation_measurement_eligible"] = _boolean(source, "direct_prop5_applicable")
    if "direct_prop5_integral_status" in source:
        source["_variation_measurement_eligible"] &= source.direct_prop5_integral_status.isin(
            ["estimated_converged", "converged", "numerically_unresolved", "analytic_single_atom_reference"])
    else:
        source["_variation_measurement_eligible"] = False
    for field, expected in (("direct_prop5_measurement_contract", EVIDENCE_FORMULA_VERSION),
                            ("direct_prop5_variation_definition", DIRECTIONAL_VARIATION_DEFINITION)):
        source["_variation_measurement_eligible"] &= (source[field].eq(expected).fillna(False)
                                                     if field in source else False)
    signed = _number(source, "direct_prop5_variation_signed_integral_l2")
    canonical, dimension = _number(source, source_column), _number(source, "latent_dimension")
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = signed.clip(lower=0) / np.sqrt(dimension)
    finite_identity = np.isfinite(signed) & np.isfinite(canonical) & np.isfinite(expected) & dimension.gt(0)
    identity_tolerance = 128 * np.finfo(float).eps * np.maximum(1., np.maximum(np.abs(canonical), np.abs(expected)))
    identity_consistent = finite_identity & (np.abs(canonical - expected) <= identity_tolerance)
    source["variation_value_identity_status"] = np.select(
        [~finite_identity, ~identity_consistent],
        ["unavailable_positive_part_identity", "inconsistent_positive_part_identity"],
        default="consistent_with_float64_estimate")
    source["_variation_measurement_eligible"] &= identity_consistent
    in_domain = source.loc[_number(source, "step_index").isin(range(max(0, steps - 1)))]
    status_counts = {column: {str(value): int(count) for value, count in in_domain[column].value_counts(dropna=False).items()}
                     for column in status_columns if column in in_domain}
    scope = ("Saved revised directional positive-variation point estimates: positive part after the signed "
             "unit-gap projection integral, then normalization once by sqrt(d); "
             "finite unresolved quadrature/condition-sign estimates are retained. "
             "These curves are not certified values or uncertainty bands. Missing, negative or nonfinite estimates "
             "and unavailable measurement receipts exclude the entire prompt curve; no zero filling, "
             "interpolation, sign-based substitution or refined-interval midpoint is used.")

    def save_variation(frame, **info):
        info.update(formula_version="reference-directional-variation-per-prompt-1",
                    prediction_domain="positive_noise_transitions", missing_measurement_fields=missing,
                    numerical_scope=scope, numerical_status_counts=status_counts,
                    measurement_audit_table="audit_data/feedback_endpoints.csv",
                    measurement_contract=EVIDENCE_FORMULA_VERSION,
                    variation_definition=DIRECTIONAL_VARIATION_DEFINITION,
                    variation_identity_check="Saved V_rmse equals max(0,signed_integral_l2)/sqrt(d), within 128*eps64*max(1,abs(saved),abs(expected)); numerical consistency only, not an interval certificate",
                    zero_gap_convention="mathcal{V}_t=0 when Delta_t is exactly zero; no unit direction is formed",
                    reference_definition="Current unconditional posterior clean reference bar{x}_t(empty), not selected global mu or the matched segment's left endpoint",
                    manuscript_domain="Requested directional revision: t=2,...,T; chronological steps 0,...,T-2; t=1 is excluded")
        if missing and info.get("status") != "not_applicable":
            info.update(status="unavailable", reason="Missing saved variation fields: " + ", ".join(missing)
                        + "; run --recompute-experiments to rebuild analysis scalars")
        save(stem, frame, **info)

    if steps < 2:
        auxiliary[stem + "_cohort"] = pd.DataFrame(columns=PAIR_KEYS + ["eligible", "reason"])
        save_variation(pd.DataFrame(columns=columns), status="not_applicable",
                       reason="The revised directional variation requires t=2,...,T; this configuration has no such transition",
                       prediction_steps=steps, expected_seed_count=seed_count, expected_seed_ids=list(range(seed_count)),
                       value_column=value_column, value_domain="nonnegative", source_column=source_column)
        return
    _prompt_scalar_inputs(source, initial, config,
                          lambda unused_stem, frame, **info: save_variation(frame, **info), auxiliary,
                          stem=stem, source_column=source_column, value_column=value_column,
                          prediction_domain="positive_noise_transitions",
                          row_eligibility_column="_variation_measurement_eligible", audit_status_columns=status_columns)


def _theory_mean_metadata(tables, provenance):
    """Retain the selected comparison mean separately from the finite-bank mean."""
    law = provenance.get("reference_law", {})
    receipt = copy.deepcopy(law.get("theory_mean", provenance.get("theory_mean", {})))
    rows = [tables[name] for name in ("gaussian_reference", "reference_analytical", "reference_atoms") if name in tables]
    for column, key in (("theory_mean_source", "source"), ("theory_mean_sha256", "vector_sha256")):
        saved = {str(value) for frame in rows if column in frame for value in frame[column].dropna().unique()}
        if len(saved) > 1 or (saved and receipt.get(key) is not None and saved != {str(receipt[key])}):
            raise TheoryError("Theory-mean scalar receipts disagree on " + column)
        if saved:
            receipt.setdefault(key, next(iter(saved)))
    source = receipt.get("source")
    if source is not None and source not in {"minimum_snr_unconditional_reference_monte_carlo", "initial_unconditional_reference_monte_carlo", "declared_finite_bank_mean"}:
        raise TheoryError("Unknown selected theory-mean source")
    norms = {}
    for field in ("mean_norm_rmse", "mean_norm_l2", "bank_mean_norm_rmse", "bank_mean_norm_l2", "mean_offset_rmse", "mean_offset_l2", "comparison_scale_rmse"):
        parts = [_number(frame, field) for frame in rows if field in frame]
        values = pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype=float)
        values = values.loc[np.isfinite(values)]
        if len(values):
            if values.lt(0).any() or not np.allclose(values, values.iloc[0], rtol=1e-12, atol=1e-14):
                raise TheoryError("Theory-mean comparisons have inconsistent saved " + field)
            norms[field] = float(values.iloc[0])
    if source == "minimum_snr_unconditional_reference_monte_carlo":
        definition = "mu is the saved vector average of unconditional posterior clean references at the smallest SNR of the prespecified analytical extension, evaluated on independent Gaussian inputs; the default estimator uses 10000 draws. The receipt distinguishes this estimation SNR from the native initial SNR. At a finite chosen SNR this reference estimate approximates, but need not equal, the exact finite-bank mean and is not an exact training-data mean. No learned-network evaluation is required."
    elif source == "initial_unconditional_reference_monte_carlo":
        definition = "mu is the explicitly selected legacy reference-initial vector average of unconditional posterior clean references at the first native denoising noise level on independent Gaussian inputs. This finite-initial-noise reference estimate is not an exact training-data mean and requires no learned-network evaluation; its sample count and estimation level remain recorded."
    elif source == "declared_finite_bank_mean":
        definition = "mu is the explicitly selected exact declared finite-bank weighted atom mean; this optional baseline is not identified with the full training-data mean."
    else:
        definition = "Selected theory-mean measurement receipt is unavailable; analysis is required before publishing a mu comparison."
    return {"theory_mean": receipt, "theory_mean_source": source,
        "theory_mean_sha256": receipt.get("vector_sha256"), **norms,
        "theory_mean_definition": definition,
        "theory_mean_uncertainty": "Evaluation-seed intervals condition on the saved selected mean. The independent mean estimator's Monte Carlo standard error and finite-selected-SNR bias scope are reported separately in its receipt.",
        "bank_mean_definition": "Exact declared finite-law atom-weighted mean; retained for the posterior and its lower-SNR limit independently of the selected mu.",
        "reference_mean_limit_definition": "Distance from the exact finite-bank weighted mean to selected mu, divided by sqrt(d)."}



def _guidance_fit_inputs(tables, config, mean_metadata, save, auxiliary):
    """Join genuine forward loss to a joint initial guided-estimate LS fit.

    Each seed has the same target-minus-selected-mean design vector. Therefore
    the mean of its signed scalar fits is exactly the joint equal-weight LS
    coefficient. Saved orthogonal residuals also determine the pooled residual.
    No vector direction is reconstructed from differences of scalar distances.
    """
    stem = "corollary3_guidance_scale_vs_loss"
    audit_name = stem + "_cohort"
    columns = PAIR_KEYS + ["x", "y", "mean_terminal_sscd", "seed_count", "seed_ids_json",
        "latent_dimension", "snr", "guidance_scale", "fit_residual_rmse", "direction_norm_rmse"]
    seed_count = int(config["num_seeds"])
    expected_seeds = set(range(seed_count))
    g = float(config["guidance_scale"])
    metadata = dict(
        formula_version="corollary3-guidance-fit-1", guidance_scale=g,
        expected_seed_count=seed_count, expected_seed_ids=sorted(expected_seeds),
        aggregation="joint_no_intercept_least_squares_shared_target_direction",
        fit_definition="argmin_a sum_seed ||[hat{x}_T(c;g)-mu]-a(x_star-mu)||^2; unconstrained signed scalar, no intercept",
        coefficient_definition="mean_seed <x_star-mu,hat{x}_T(c;g)-mu>/||x_star-mu||^2; shared target and selected mu",
        residual_definition="sqrt(mean_seed [r_seed^2+(g_hat_seed-g_hat)^2*||x_star-mu||^2/d]); r_seed is the per-seed orthogonal residual divided by sqrt(d)",
        normalization="x=L_T(c)/(d*SNR_T), no square root; y is dimensionless and has no dimensional normalization",
        input_source="generated_state_at_first_prediction",
        forward_input_source="forward_target",
        color_population="Mean terminal paired-target SSCD over the same complete fixed evaluation seed cohort as the joint fit",
        weighting="One point per retained prompt-target pair; equal weight for every expected evaluation seed in its joint fit",
        cohort_audit_table="audit_data/" + audit_name + ".csv",
        seed_fit_table="audit_data/initial_generated_samples.csv",
        forward_loss_table="audit_data/initial_pairs.csv",
        interpretation="Corollary 3 Eq.54 limiting guided-estimate coefficient; a descriptive finite-noise fit, not a guidance sweep or an estimate of an unknown configured sampler parameter",
    )
    required_fit = SAMPLE_KEYS + ["step_index", "snr", "latent_dimension", "terminal_sscd", "input_source",
        "theory_mean_source", "theory_mean_sha256", "cor3_guidance_fit",
        "cor3_guidance_fit_residual_rmse", "cor3_guidance_fit_direction_l2",
        "cor3_guidance_fit_direction_tolerance_l2", "cor3_guidance_fit_applicable",
        "cor3_guidance_fit_status", "cor3_guidance_fit_qa"]
    fitted = tables.get("initial_generated_samples", pd.DataFrame()).copy()
    missing = sorted(set(required_fit) - set(fitted))
    if missing:
        auxiliary[audit_name] = pd.DataFrame([{"included": False, "reason": "missing_saved_guided_fit",
                                               "missing_columns_json": json.dumps(missing)}])
        save(stem, pd.DataFrame(columns=columns), status="unavailable",
             reason="Analysis required: initial cached-vector least-squares measurements are missing; run --recompute-experiments", **metadata)
        return
    actual = tables["initial"].copy()
    loss = tables["forward_loss_summary"].copy()
    metadata["initial_gaussian_match_status_counts"] = {
        str(value): int(count) for value, count in
        fitted.get("initial_gaussian_bank_match", pd.Series("unavailable", index=fitted.index)).value_counts(dropna=False).items()
    }
    metadata["initialization_scope"] = "Saved actual first-prediction states; Gaussian-bank agreement is recorded separately and is not silently assumed for unmatched states."
    _require(actual, SAMPLE_KEYS + ["step_index", "snr", "latent_dimension", "terminal_sscd", "input_source"], "guidance-fit terminal cohort")
    _require(loss, PAIR_KEYS + ["step_index", "snr", "latent_dimension", "input_source", "loss_mean_squared_l2", "draw_count"], "guidance-fit forward loss")
    loss = loss.loc[_number(loss, "step_index").eq(0)].copy()
    for name, rows in (("fit", fitted), ("actual", actual), ("loss", loss)):
        keys = PAIR_KEYS if name == "loss" else SAMPLE_KEYS
        if rows[keys].isna().any().any():
            raise TheoryError("Guidance-fit " + name + " has missing identities")
        for key in PAIR_KEYS:
            rows[key] = rows[key].astype(str)
        if name != "loss":
            seeds = _number(rows, "seed")
            if not (np.isfinite(seeds) & seeds.eq(np.floor(seeds))).all():
                raise TheoryError("Guidance-fit seeds must have finite integer identities")
            rows["seed"] = seeds.astype(int)
        if rows.duplicated(keys).any():
            raise TheoryError("Duplicate guidance-fit " + name + " identity")
        if not _number(rows, "step_index").eq(0).all():
            raise TheoryError("Guidance-fit comparison requires first prediction only")
        if not rows.input_source.eq("forward_target" if name == "loss" else "generated_state").all():
            raise TheoryError("Guidance-fit " + name + " uses a different input law")
    for field in ("theory_mean_source", "theory_mean_sha256"):
        expected = mean_metadata.get(field)
        if expected is None or not fitted[field].eq(expected).all():
            raise TheoryError("Guidance-fit selected mean receipt differs: " + field)
    cohorts = [{key: rows.sort_values("seed") if "seed" in rows else rows
                for key, rows in frame.groupby(PAIR_KEYS, sort=False, dropna=False)}
               for frame in (fitted, actual, loss)]
    identities = sorted(set().union(*(set(cohort) for cohort in cohorts)))
    records, audits = [], []
    for identity in identities:
        fit, outcomes, forward = [cohort.get(identity, frame.iloc[:0])
                                  for cohort, frame in zip(cohorts, (fitted, actual, loss))]
        reasons = []
        if set(fit.seed) != expected_seeds:
            reasons.append("incomplete_fit_seed_cohort")
        if set(outcomes.seed) != expected_seeds:
            reasons.append("incomplete_terminal_seed_cohort")
        if len(forward) != 1:
            reasons.append("missing_pair_forward_loss")
        if not _boolean(fit, "cor3_guidance_fit_applicable").all():
            reasons.append("inapplicable_or_degenerate_target_direction")
        if not fit.cor3_guidance_fit_status.eq("measured").all():
            reasons.append("invalid_seed_fit_status")
        if not fit.cor3_guidance_fit_qa.eq("consistent_float64_estimate").all():
            reasons.append("unresolved_seed_fit_identity")
        fits, residual = _number(fit, "cor3_guidance_fit"), _number(fit, "cor3_guidance_fit_residual_rmse")
        direction, tolerance = _number(fit, "cor3_guidance_fit_direction_l2"), _number(fit, "cor3_guidance_fit_direction_tolerance_l2")
        colors = _number(outcomes, "terminal_sscd")
        if not (np.isfinite(fits) & np.isfinite(residual) & residual.ge(0)
                & np.isfinite(direction) & np.isfinite(tolerance) & tolerance.ge(0)
                & direction.gt(tolerance) & direction.gt(0)).all():
            reasons.append("invalid_or_unresolved_fit_scalars")
        if not np.isfinite(colors).all():
            reasons.append("missing_terminal_sscd")
        if not reasons:
            # The complete identity join must retain the same native level,
            # latent space and actual terminal outcomes in both saved sources.
            d, snr = _number(fit, "latent_dimension"), _number(fit, "snr")
            d0, snr0 = float(d.iloc[0]), float(snr.iloc[0])
            if (not math.isfinite(d0) or d0 <= 0 or d0 != int(d0) or not d.eq(d0).all()
                    or not math.isfinite(snr0) or snr0 <= 0 or not np.allclose(snr, snr0, rtol=1e-12, atol=0)
                    or not _number(outcomes, "latent_dimension").eq(d0).all()
                    or not np.allclose(_number(outcomes, "snr"), snr0, rtol=1e-12, atol=0)
                    or float(forward.latent_dimension.iloc[0]) != d0
                    or not math.isclose(float(forward.snr.iloc[0]), snr0, rel_tol=1e-12)):
                raise TheoryError("Guidance-fit forward/actual/fit native noise or latent dimension differs")
            if not np.allclose(_number(fit, "terminal_sscd"), colors, rtol=0, atol=0):
                raise TheoryError("Guidance-fit terminal SSCD changed for the same pair and seed")
            if not np.allclose(direction, float(direction.iloc[0]), rtol=1e-12, atol=0):
                raise TheoryError("Guidance-fit target direction changes across evaluation seeds")
            if "guidance_scale" in fit and not np.allclose(_number(fit, "guidance_scale"), g, rtol=1e-12, atol=0):
                raise TheoryError("Guidance-fit configured guidance differs")
            loss_value = float(forward.loss_mean_squared_l2.iloc[0])
            draws = float(forward.draw_count.iloc[0])
            if not math.isfinite(loss_value) or loss_value < 0 or not math.isfinite(draws) or draws <= 0 or draws != int(draws):
                raise TheoryError("Guidance-fit x requires a measured nonnegative forward-loss mean")
            denominator = d0 * snr0
            if not math.isfinite(denominator) or denominator <= 0:
                raise TheoryError("Guidance-fit loss normalization requires finite positive d*SNR_T")
            # Scaling by seed count before summation avoids needless overflow.
            coefficient = float((fits / seed_count).sum())
            direction_rmse = float(direction.iloc[0]) / math.sqrt(d0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                pooled_terms = np.hypot(residual, (fits - coefficient) * direction_rmse)
                scale = float(pooled_terms.max())
                pooled_residual = scale * float(np.sqrt(np.mean(np.square(pooled_terms / scale)))) if scale > 0 and math.isfinite(scale) else scale
                x = loss_value / denominator
            if loss_value > 0 and x == 0:
                reasons.append("positive_loss_ratio_underflow")
            elif not all(math.isfinite(value) for value in (coefficient, pooled_residual, x)):
                reasons.append("nonfinite_joint_fit_reduction")
            else:
                records.append({**dict(zip(PAIR_KEYS, identity)), "x": x, "y": coefficient,
                    "mean_terminal_sscd": float((colors / seed_count).sum()),
                    "seed_count": seed_count, "seed_ids_json": json.dumps(sorted(expected_seeds)),
                    "latent_dimension": int(d0), "snr": snr0, "guidance_scale": g,
                    "fit_residual_rmse": pooled_residual, "direction_norm_rmse": direction_rmse})
        audits.append({**dict(zip(PAIR_KEYS, identity)), "included": not reasons,
            "reason": "complete_joint_fit" if not reasons else ";".join(reasons),
            "expected_seed_ids_json": json.dumps(sorted(expected_seeds)),
            "fit_seed_ids_json": json.dumps(sorted(set(fit.seed))),
            "terminal_seed_ids_json": json.dumps(sorted(set(outcomes.seed))),
            "fit_status_counts_json": json.dumps(fit.cor3_guidance_fit_status.value_counts(dropna=False).to_dict(), sort_keys=True),
            "fit_qa_counts_json": json.dumps(fit.cor3_guidance_fit_qa.value_counts(dropna=False).to_dict(), sort_keys=True)})
    frame = pd.DataFrame(records, columns=columns)
    auxiliary[audit_name] = pd.DataFrame(audits)
    status = "available" if len(frame) else "unavailable"
    direction_only_reasons = {"inapplicable_or_degenerate_target_direction", "invalid_seed_fit_status",
                              "unresolved_seed_fit_identity", "invalid_or_unresolved_fit_scalars"}
    if (not len(frame) and len(fitted) and audits
            and fitted.cor3_guidance_fit_status.eq("exact_zero_target_direction").all()
            and all(set(row["reason"].split(";")) <= direction_only_reasons for row in audits)):
        # Only a fully observed exact-zero design is mathematically unidentified.
        # Missing cohorts and roundoff-unresolved directions require analysis.
        status = "not_applicable"
    initial_snr = float(frame.snr.iloc[0]) if len(frame) else None
    if len(frame) and not np.allclose(frame.snr, initial_snr, rtol=1e-12, atol=0):
        raise TheoryError("Guidance-fit scatter combines different initial native noise levels")
    save(stem, frame, status=status,
         reason=None if len(frame) else "No complete valid initial guided-estimate fit cohort; see saved cohort audit",
         actual_initial_snr=initial_snr, represented_pair_count=len(frame),
         excluded_pair_count=sum(not row["included"] for row in audits), **metadata)


def build_four_stage_plot_inputs(tables, config, provenance):
    """Return compact per-stem frames and the established figures/audit contract."""
    config = {**config.get("scientific_config", config), **config.get("supplemental_config", {})}
    gaussian_source = tables["gaussian_conditional"]
    _require(gaussian_source, ["gaussian_control_status"], "initial Gaussian controls")
    independent = gaussian_source.gaussian_control_status.eq("measured_independent_gaussian_control")
    if independent.any():
        _require(gaussian_source, ["gaussian_control_input_source", "gaussian_control_input_sha256", "gaussian_control_task_hash"], "independent Gaussian control provenance")
        selected = gaussian_source.loc[independent]
        if not selected.gaussian_control_input_source.eq("true_gaussian_initialization_probe").all() or selected[["gaussian_control_input_sha256", "gaussian_control_task_hash"]].isna().any().any():
            raise TheoryError("Independent initial controls require a genuine Gaussian-input identity and task receipt")
    old_frames, previous = build_evidence_plot_inputs(tables, config, provenance)
    frames = {name: frame.copy() for name, frame in old_frames.items()}
    metadata = copy.deepcopy(previous["figures"])
    auxiliary = dict(previous["auxiliary_tables"])
    audit = copy.deepcopy(previous["audit"])
    audit["four_stage_formula_version"] = FOUR_STAGE_FORMULA_VERSION
    _terminal_display_policy(tables["terminal"], metadata, audit, auxiliary)
    reference = previous["figures"]["theorem1_loss_recovery"]["reference_provenance"]
    mean_metadata = _theory_mean_metadata(tables, provenance)
    for info in metadata.values():
        info.update(mean_metadata)
    steps = int(config["num_inference_steps"])

    def save(stem, frame, *, status=None, reason=None, **info):
        if status is None:
            candidates = [name for name in ("x", "y", "median", "mean", "fraction", "cdf") if name in frame]
            finite = any(np.isfinite(_number(frame, name)).any() for name in candidates)
            if {"x", "y"}.issubset(frame):
                finite = bool((np.isfinite(_number(frame, "x")) & np.isfinite(_number(frame, "y"))).any())
            status = "available" if finite else "unavailable"
        frames[stem] = frame.copy()
        metadata[stem] = {"status": status, "reason": reason if reason else None if status == "available" else "No eligible measurements for the fixed comparison",
            "formula_version": FOUR_STAGE_FORMULA_VERSION, "reference_provenance": reference,
            "counts": _population(frame), "status_counts": _statuses(frame), **mean_metadata, **info}
        if status == "blocked":
            audit["blocking"] = True

    def reuse(stem, source, *, frame=None, **extra):
        frames[stem] = frames[source].copy() if frame is None else frame.copy()
        metadata[stem] = {**copy.deepcopy(metadata[source]), "formula_version": FOUR_STAGE_FORMULA_VERSION,
                          "source_comparison": source, **extra}

    reuse("initial_loss_recovery", "theorem1_loss_recovery",
          initial_baseline_summary="initial_baseline_summary.csv", initial_baseline_json="initial_baseline_summary.json")
    # Preserve the prior 1000-resample recipe. Reusing the same sorted seed
    # stream makes Y/U intervals paired without introducing a pooled bootstrap.
    pairs = frames["initial_loss_recovery"].copy()
    gaussian = auxiliary["theorem1_gaussian_seed_tails"]
    for identity, rows in gaussian.groupby(PAIR_KEYS, sort=False, dropna=False):
        rows = rows.sort_values("seed", key=lambda values: pd.to_numeric(values, errors="raise"))
        valid = np.isfinite(_number(rows, "conditional_error_rmse")) & np.isfinite(_number(rows, "unconditional_target_error_rmse"))
        valid &= rows.gaussian_control_status.isin(["same_saved_gaussian_input_canonical_epsilon", "measured_independent_gaussian_control"])
        low, high = _interval(np.square(_number(rows.loc[valid], "unconditional_target_error_rmse")), seed=BOOTSTRAP_SEED)
        match = pd.Series(True, index=pairs.index)
        for key, value in zip(PAIR_KEYS, identity):
            match &= pairs[key].astype(str).eq(str(value))
        pairs.loc[match, "control_y_low"], pairs.loc[match, "control_y_high"] = low, high
    actual = tables["initial"].copy()
    _require(actual, SAMPLE_KEYS + ["terminal_sscd"], "initial paired terminal outcomes")
    if actual.duplicated(SAMPLE_KEYS).any():
        raise TheoryError("Actual terminal SSCD must have one observation per retained pair and seed")
    for identity, rows in actual.groupby(PAIR_KEYS, sort=False, dropna=False):
        match = pd.Series(True, index=pairs.index)
        for key, value in zip(PAIR_KEYS, identity):
            match &= pairs[key].astype(str).eq(str(value))
        scores = _number(rows, "terminal_sscd")
        finite = np.isfinite(scores)
        pairs.loc[match, "mean_terminal_sscd"] = float(scores[finite].mean()) if finite.any() else np.nan
        pairs.loc[match, "terminal_outcome_count"] = int(finite.sum())
        pairs.loc[match, "terminal_outcome_missing_count"] = int((~finite).sum())
    metadata["initial_loss_recovery"]["color_population"] = "Mean terminal paired-target SSCD across all retained actual evaluation seeds of each pair, independent of the Gaussian recovery bank; no matched-input claim for fresh probes"
    metadata["initial_loss_recovery"]["matched_control_status_counts"] = {str(key): int(value) for key, value in gaussian_source.gaussian_control_status.value_counts(dropna=False).items()}
    frames["initial_loss_recovery"] = pairs
    complete_pairs = pairs.copy()
    complete_pairs["A_c"], complete_pairs["Y_c"], complete_pairs["U_c"] = pairs.x, pairs.y, pairs.control_y
    loss = tables["forward_loss_summary"].loc[lambda rows: _number(rows, "step_index").eq(0)].copy()
    for key in PAIR_KEYS:
        complete_pairs[key] = complete_pairs[key].astype(str)
        loss[key] = loss[key].astype(str)
    loss_fields = [name for name in loss if name not in complete_pairs or name in PAIR_KEYS]
    complete_pairs = complete_pairs.merge(loss[loss_fields], on=PAIR_KEYS, how="left", validate="one_to_one")
    auxiliary["initial_pairs"] = complete_pairs
    auxiliary["initial_samples"] = tables.get("initial_samples", gaussian).copy()
    auxiliary["initial_gaussian_samples"] = gaussian.copy()
    _baseline_inputs(tables, auxiliary, metadata["initial_loss_recovery"])
    _guidance_fit_inputs(tables, config, mean_metadata, save, auxiliary)
    reuse("initial_unconditional_mean_concentration", "lemma2_initial_concentration")
    # Merge only saved scalar families; no fresh analytical or network evaluation.
    reference_frame = pd.concat([
        frames["lemma2_unconditional_baseline"].loc[lambda rows: rows.source_range.eq("analytical")] if "source_range" in frames["lemma2_unconditional_baseline"] else pd.DataFrame(),
        frames["lemma2_native_gaussian_sweep"],
    ], ignore_index=True)
    reuse("unconditional_reference_convergence", "lemma2_unconditional_baseline", frame=reference_frame,
          input_law_separation="Analytical reference-only lower-SNR extension plus the entire available native Gaussian probe sweep. No network extrapolation or favorable time window.")
    metadata["unconditional_reference_convergence"].pop("native_sweep_figure", None)
    metadata["unconditional_reference_convergence"]["display_window_definition"] = "Entire saved analytical extension and entire genuine native Gaussian sweep; no favorable step restriction"
    native_reference = reference_frame.loc[reference_frame.source_range.eq("native")] if "source_range" in reference_frame else pd.DataFrame()
    native_available = all(not native_reference.loc[native_reference.metric.eq(metric)].empty and np.isfinite(_number(native_reference.loc[native_reference.metric.eq(metric)], "median")).any()
                           for metric in ("reference", "learned", "reference_error")) if not native_reference.empty else False
    if native_available:
        metadata["unconditional_reference_convergence"].update(status="available", reason=None)
    else:
        metadata["unconditional_reference_convergence"].update(status="unavailable", reason="A required genuine native Gaussian-probe comparison is missing; analytical reference curves cannot substitute for network observations")

    for stem in ("initial_loss_recovery", "initial_unconditional_mean_concentration", "unconditional_reference_convergence"):
        metadata[stem].update(mean_metadata,
            reference_scale_definition="Root mean squared atom deviation about the exact finite-bank mean; this bank spread is not the radial RMS about selected mu.")
        required = ("theory_mean_source", "theory_mean_sha256", "mean_norm_rmse", "bank_mean_norm_rmse", "mean_offset_rmse")
        if any(metadata[stem].get(key) is None for key in required) and metadata[stem]["status"] not in {"blocked", "not_applicable"}:
            metadata[stem].update(status="unavailable", reason="Analysis required: selected theory-mean receipt and distinct finite-bank mean/offset scalars are missing")
    metadata["unconditional_reference_convergence"]["reference_limit_rmse"] = mean_metadata.get("mean_offset_rmse")

    _response_inputs(tables.get("feedback_response", pd.DataFrame()), config, save, auxiliary)
    reuse("posterior_feedback_over_time", "proposition5_posterior_feedback")
    reuse("posterior_feedback_condition_margin", "proposition5_condition_vs_gain")
    if "refinement_applicable" in tables["matched_updates"] and not _boolean(tables["matched_updates"], "refinement_applicable").any():
        for stem in ("posterior_feedback_over_time", "posterior_feedback_condition_margin"):
            metadata[stem].update(status="not_applicable", reason="Every transition is outside the recorded manuscript/reference domain")
    for stem in ("posterior_feedback_over_time", "posterior_feedback_condition_margin"):
        metadata[stem]["numerical_resolution_table"] = "audit_data/proposition5_numerical_resolution_rows.csv"
        # Reuse the saved numerical sign assessment and its separate scope
        # receipt. Optional interval eligibility is not a display prerequisite.
        metadata[stem]["condition_display_policy"] = "Saved numerical estimates and optional enclosed signs; unresolved markers/bands follow sign uncertainty, not absence of interval certification"
    auxiliary["feedback_endpoints"] = tables["matched_updates"].copy()
    _prompt_variation_inputs(tables["matched_updates"], actual, config, save, auxiliary)

    trajectory = tables.get("trajectory_metrics", tables["trajectory"]).copy()
    _require(trajectory, SAMPLE_KEYS + ["step_index", "snr", "terminal_sscd", "direct_conditional_error_rmse", "direct_unconditional_target_error_rmse", "direct_lemma6_gap_rmse"], "four-stage trajectory metrics")
    trajectory["joint_error"] = np.maximum(_number(trajectory, "direct_conditional_error_rmse"), _number(trajectory, "direct_unconditional_target_error_rmse"))
    trajectory["normalized_progress"] = _number(trajectory, "step_index") / (steps - 1) if steps > 1 else 0.
    schedule = _schedule(trajectory, steps)
    shared = {"prediction_steps": steps, "initialization_index": 0, "chronological_schedule": schedule,
        "normalized_progress_definition": "k/(K_steps-1), or 0 for K_steps=1",
        "weighting": "Equal mass per represented complete prompt, divided over its common eligible member seeds within each outcome group and step",
        "denominator_column": "denominator_weight", "population_counts": _population(trajectory),
        "missing_outcome_row_count": int((~np.isfinite(_number(trajectory, "terminal_sscd"))).sum()),
        "empty_groups": [group for group, rows in _outcomes(trajectory) if rows.empty]}
    save("branch_gap_synchronization", _chronological(trajectory, {"gap": "direct_lemma6_gap_rmse", "joint_error": "joint_error"}, steps),
         band_definition="Descriptive Q IQR; same finite D/Q population at each step, independent of reference-bound availability", **shared)
    save("branch_target_errors", _chronological(trajectory, {"conditional": "direct_conditional_error_rmse", "unconditional": "direct_unconditional_target_error_rmse"}, steps), **shared)
    bound = _chronological(trajectory, {"gap": "direct_lemma6_gap_rmse", "joint_error": "joint_error", "bound": "direct_lemma6_rhs_rmse"}, steps)
    reuse("synchronization_bound", "lemma6_target_specific_synchronization", frame=bound, **shared)
    _prompt_gap_inputs(trajectory, actual, config, save, auxiliary)
    _prompt_reference_inputs(trajectory, actual, config, save, auxiliary)
    auxiliary["trajectory_metrics"] = trajectory
    for name in ("trajectory_shape_prompts", "trajectory_shape_mixed_prompts", "initial_baseline_disagreements", "initial_generated_samples", "reference_law", "reference_target_radii"):
        if name in tables:
            auxiliary[name] = tables[name].copy()
    _shape_inputs(tables.get("trajectory_shapes", pd.DataFrame()), steps, save, auxiliary)
    _motion_inputs(tables.get("branch_motion", pd.DataFrame()), steps, save, auxiliary)
    _terminal_inputs(tables["terminal"], frames, metadata, config, save, auxiliary)
    _terminal_grouped_coverage(tables["terminal"], frames, metadata, save, auxiliary)
    _counterfactual_inputs(tables.get("counterfactual_unconditional", pd.DataFrame()), bool(config.get("counterfactual_unconditional", False)), save, auxiliary)
    for entry in paper_registry(True, counterfactual=bool(config.get("counterfactual_unconditional", False))):
        if entry["stem"] not in frames:
            save(entry["stem"], pd.DataFrame(), reason="Optional compatible measurement was not recorded; plot mode does not backfill scientific work")
    audit["blocking"] = bool(audit.get("blocking", False) or any(value["status"] == "blocked" for value in metadata.values()))
    audit["figure_blockers"] = sorted(stem for stem, info in metadata.items() if info["status"] == "blocked")
    for stem in ("posterior_feedback_over_time", "posterior_feedback_condition_margin", "synchronization_bound",
                 "final_reproduction_bound", "terminal_bound_coverage"):
        metadata[stem].update(formula_version="projected-gap-error-1",
                              measurement_contract="projected-gap-error-1")
    return frames, {"figures": metadata, "audit": audit, "auxiliary_tables": auxiliary,
                    "initial_baseline_summary": auxiliary.get("initial_baseline_summary", pd.DataFrame()).to_dict("records")}


def _terminal_display_policy(terminal, metadata, audit, auxiliary):
    """Keep observed comparison failures visible, while retaining invalid-input gates."""
    stems = ("theorem7_final_reproduction", "theorem7_terminal_components")
    original = {stem: copy.deepcopy(metadata[stem]) for stem in stems}
    ordering = auxiliary.get("theorem7_paired_bound_ordering", pd.DataFrame())
    if ordering.empty:
        return
    invalid = pd.Series(False, index=terminal.index)
    for name in ("evidence_terminal_status", "evidence_terminal_reconstruction_status", "evidence_terminal_clean_bound_order_status"):
        if name in terminal:
            invalid |= terminal[name].fillna("").astype(str).str.contains("failed", case=False, regex=False)
    looseness = auxiliary.get("theorem7_looseness_decomposition", pd.DataFrame())
    invalid_looseness = (not looseness.empty and looseness.looseness_audit_status.eq("failed_fixed_float64_consistency").any())
    invalid_order = ordering.observable_above_reference_beyond_roundoff.any()
    observed = int(ordering.actual_above_observable_beyond_sensitivity.sum())
    audit["terminal_observed_comparison_policy"] = {
        "observed_above_observable_count": observed,
        "legacy_measurement_metadata": original,
        "comparison_table": "audit_data/theorem7_paired_bound_ordering.csv",
        "rule": "Observed endpoint-above-bound comparisons remain plotted and counted. Invalid construction, formula/identity failures and inconsistent bound ordering still block publication. Probability-bound exceedances retain their probability scope.",
    }
    if invalid.any() or invalid_looseness or invalid_order:
        return
    for stem in stems:
        metadata[stem].update(status="available", reason=None,
            observed_bound_exceedance_count=observed,
            observed_comparison_policy="All eligible actual endpoints remain visible, including actual bound exceedances; comparison visibility is not a successful theorem-validation claim",
            inherited_audit_status=original[stem]["status"])
    prior_blockers = {stem for stem, info in metadata.items() if info["status"] == "blocked"}
    legacy_blocked = audit.get("inherited_nonfigure_blocking", True)
    unexplained_blockers = set(audit.get("figure_blockers", [])) - set(stems)
    # Only the two known old terminal comparison gates have been relaxed.
    if not prior_blockers and not legacy_blocked and not unexplained_blockers and any(info["status"] == "blocked" for info in original.values()):
        audit["blocking"] = False


def _baseline_inputs(tables, auxiliary, initial_metadata):
    baseline = tables.get("initial_baseline_summary", tables.get("baseline_summary", pd.DataFrame())).copy()
    reference = tables["gaussian_reference"]
    seeds = reference.loc[_number(reference, "step_index").eq(0)].copy()
    _require(seeds, ["run_id", "seed", "learned_mean_error_rmse"], "unique initial Gaussian baseline")
    if seeds.duplicated(["run_id", "seed"]).any():
        raise TheoryError("Initial baseline must use one validated canonical observation per Gaussian seed")
    if baseline.empty:
        raise TheoryError("Missing mandatory unique-seed initial baseline report; rerun the four-stage scalar supplement")
    _require(baseline, ["mean_norm_l2", "spread_rmse", "baseline_error_rmse", "relative_baseline_error", "unique_seed_count", "disagreement_count"], "initial_baseline_summary")
    baseline["baseline_mc_low"], baseline["baseline_mc_high"] = np.nan, np.nan
    for index, row in baseline.iterrows():
        group = seeds.loc[seeds.run_id.astype(str).eq(str(row.run_id))] if "run_id" in baseline else seeds
        if len(group) != int(row.unique_seed_count):
            raise TheoryError("Unique-seed baseline receipt has a different Gaussian population")
        values = _number(group.sort_values("seed", key=lambda value: pd.to_numeric(value, errors="raise")), "learned_mean_error_rmse")
        rms = math.sqrt(float(np.square(values).mean())) if len(values) and np.isfinite(values).all() else np.nan
        if not math.isclose(rms, float(row.baseline_error_rmse), rel_tol=1e-12, abs_tol=1e-14):
            raise TheoryError("Saved baseline RMS disagrees with the validated unique-seed observations")
        scale = float(row.spread_rmse)
        if scale < 0 or not math.isfinite(scale):
            raise TheoryError("Invalid declared-law RMS spread")
        if scale == 0 and np.isfinite(float(row.relative_baseline_error)):
            raise TheoryError("Zero-spread relative baseline error must remain undefined")
        baseline.loc[index, ["baseline_mc_low", "baseline_mc_high"]] = _interval(np.square(values), seed=BOOTSTRAP_SEED)
    auxiliary["initial_baseline_summary"] = baseline
    auxiliary["initial_baseline_samples"] = seeds
    initial_metadata.update(baseline_status="available" if baseline.disagreement_count.eq(0).all() else "canonical_unconditional_disagreement_recorded",
        baseline_summary=baseline.to_dict("records"),
        baseline_uncertainty={
            **copy.deepcopy(BOOTSTRAP_POLICY),
            "scope": "Monte Carlo uncertainty over unique Gaussian evaluation seeds only; no population confidence claim",
            "seed_dependence": "Sorted unique Gaussian seed IDs use one fixed resampling stream; repeated prompts add no observations",
            "conditioning": "Conditional on the saved selected mu",
            "mean_estimation_uncertainty": "Independent mean-estimation Monte Carlo uncertainty is reported separately in theory_mean",
        },
        baseline_interpretation="Unique Gaussian evaluation seeds, not prompt-replicated estimates; the baseline measures distance to the separately selected mu and differs from the unconditional paired-target control. The denominator is the exact finite-bank centered spread; zero bank spread gives an undefined ratio.")


def _response_inputs(response, config, save, auxiliary):
    g = float(config["guidance_scale"])
    grid = sorted(set([index / 40 for index in range(41)] + ([1 / g] if g > 0 and 0 <= 1 / g <= 1 else [])))
    info = {"snapshot": {"step_index": 0, "rule": "Fixed first update for every configuration; never selected from effect size"},
        "dose_grid": grid, "guidance_scale": g, "symlog_linthresh": DOSE_LINEAR_THRESHOLD,
        "y_transform": {"scale": "symlog", "linthresh": DOSE_LINEAR_THRESHOLD, "base": 10, "units": "natural-log probability gain"},
        "cohort_policy": "One complete finite-H curve cohort per outcome group, fixed for every s; numerical uncertainty and negative responses never exclude a finite curve",
        "band_definition": "Prompt-balanced weighted median and descriptive IQR, not statistical confidence intervals"}
    if response.empty:
        save("branch_gap_posterior_response", pd.DataFrame(), reason="Missing fixed initial dose measurements; do not choose a later snapshot or substitute log odds", **info)
        return
    _require(response, SAMPLE_KEYS + ["step_index", "s", "log_probability_gain", "structural_applicable", "terminal_sscd", "numerical_status", "endpoint_status", "endpoint_contract"], "feedback_response")
    initial = response.loc[_number(response, "step_index").eq(0)].copy()
    if initial.empty:
        save("branch_gap_posterior_response", pd.DataFrame(), reason="Fixed first-update response is missing; later snapshots cannot substitute", **info)
        return
    if initial.duplicated(SAMPLE_KEYS + ["s"]).any():
        raise TheoryError("Duplicate scientific sample/dose identity")
    auxiliary["feedback_response"] = response.copy()
    complete_keys, exclusions, zero_failures = [], [], []
    for identity, curve in initial.groupby(SAMPLE_KEYS, sort=False, dropna=False):
        dose = _number(curve, "s")
        structural = _boolean(curve, "structural_applicable").all()
        finite = np.isfinite(_number(curve, "log_probability_gain")).all()
        same_grid = np.array_equal(np.sort(dose.to_numpy()), np.asarray(grid))
        reason = "structurally_inapplicable" if not structural else "missing_or_different_fixed_grid" if not same_grid else "unavailable_nonfinite_H_curve" if not finite else "complete"
        identity_record = dict(zip(SAMPLE_KEYS, identity))
        if reason == "complete":
            complete_keys.append(identity_record)
            if not _number(curve.loc[dose.eq(0)], "log_probability_gain").eq(0).all():
                zero_failures.append(identity_record)
        else:
            exclusions.append({**identity_record, "reason": reason, "observed_doses": len(curve)})
    keys = pd.DataFrame(complete_keys, columns=SAMPLE_KEYS)
    chosen = initial.merge(keys, on=SAMPLE_KEYS, how="inner", validate="many_to_one")
    records, statuses = [], []
    for group, selected in _outcomes(chosen):
        for s, rows in selected.groupby("s", sort=True):
            weights = _weights(rows)
            values = _number(rows, "log_probability_gain")
            q25, median, q75 = weighted_quantiles(values, weights)
            records.append({"group": group, "s": float(s), "median": median, "q25": q25, "q75": q75,
                "minimum": float(values.min()), "maximum": float(values.max()), "denominator_weight": float(weights.sum()), "eligible_count": len(rows),
                "prompt_count": len(rows[PAIR_KEYS].drop_duplicates()), "target_count": rows.target_id.nunique()})
            for status, group_rows in rows.groupby("numerical_status", dropna=False):
                statuses.append({"group": group, "s": float(s), "status": str(status), "count": len(group_rows),
                    "weight": float(weights.loc[group_rows.index].sum()), "denominator_weight": float(weights.sum())})
    info["missing_outcome_curve_count"] = int(initial.loc[~np.isfinite(_number(initial, "terminal_sscd")), SAMPLE_KEYS].drop_duplicates().shape[0])
    schedule_columns = [name for name in ("step_index", "timestep", "native_timestep", "destination_timestep", "snr", "alpha", "sigma", "destination_alpha", "destination_sigma") if name in initial]
    info["chronological_schedule"] = initial[schedule_columns].drop_duplicates().to_dict("records")
    info.update(counts={"samples": len(initial[SAMPLE_KEYS].drop_duplicates()), "complete_curves": len(keys), "excluded_curves": len(exclusions)},
        endpoint_status_counts={str(key): int(value) for key, value in initial.loc[_number(initial, "s").eq(1), "endpoint_status"].value_counts(dropna=False).items()},
        endpoint_contracts=sorted(initial.endpoint_contract.dropna().astype(str).unique()),
        cfg_endpoint_label="Matched actual CFG" if initial.loc[_number(initial, "s").eq(1), "endpoint_status"].eq("affine_and_saved_agree_under_declared_source_precision").all() else "Reconstructed matched CFG",
        response_status_table="audit_data/feedback_response_status.csv", excluded_response_table="audit_data/feedback_response_exclusions.csv")
    failed = bool(zero_failures) or initial.numerical_status.fillna("").astype(str).str.startswith("failed").any()
    status = "blocked" if failed else "not_applicable" if not _boolean(initial, "structural_applicable").any() else None
    save("branch_gap_posterior_response", pd.DataFrame(records), status=status,
         reason="Saved matched zero-dose comparison or numerical consistency audit failed" if failed else "Initial update is outside the recorded affine/reference domain" if status == "not_applicable" else None, **info)
    auxiliary["feedback_response_status"] = pd.DataFrame(statuses)
    auxiliary["feedback_response_exclusions"] = pd.DataFrame(exclusions)
    auxiliary["feedback_response_zero_failures"] = pd.DataFrame(zero_failures)


def _shape_inputs(shapes, steps, save, auxiliary):
    if shapes.empty:
        save("branch_gap_peak_step", pd.DataFrame(), reason="Missing complete individual-trajectory shape measurements")
        return
    _require(shapes, SAMPLE_KEYS + ["first_peak_step", "shape_status", "shape_complete", "terminal_sscd"], "trajectory_shapes")
    auxiliary["trajectory_shapes"] = shapes.copy()
    records, prompt_rows, mixed_rows, group_info = [], [], [], {}
    prompt_peak_rows = []
    shape_fields = [name for name in shapes if name.endswith("_rmse") and name not in {"terminal_sscd"}]
    for group, chosen in _outcomes(shapes):
        if chosen.empty:
            continue
        weights = _weights(chosen)
        peak = _number(chosen, "first_peak_step")
        resolved = _boolean(chosen, "shape_complete") & np.isfinite(peak)
        if (resolved & (~peak.isin(range(steps)))).any():
            raise TheoryError("A trajectory peak lies outside the actual prediction indices")
        denominator = float(weights.sum())
        for step in range(steps):
            members = resolved & peak.eq(step)
            records.append({"group": group, "step_index": step, "fraction": float(weights[members].sum()) / denominator,
                "denominator_weight": denominator, "sample_count": int(members.sum())})
        group_info[group] = {"samples": len(chosen), "resolved_peak_count": int(resolved.sum()),
            "unassigned_weight_fraction": float(weights[~resolved].sum()) / denominator,
            "status_counts": {str(key): int(value) for key, value in chosen.shape_status.value_counts(dropna=False).items()}}
        for field, name in (("peak_ties_json", "multiple_exact_peak_count"), ("near_peak_ties_json", "multiple_near_peak_count")):
            if field in chosen:
                group_info[group][name] = int(chosen[field].fillna("[]").map(lambda value: len(json.loads(value)) > 1).sum())
        for identity, rows in chosen.groupby(PAIR_KEYS, dropna=False, sort=False):
            record = {**dict(zip(PAIR_KEYS, identity)), "group": group, "sample_count": len(rows)}
            for step in range(steps):
                prompt_peak_rows.append({**dict(zip(PAIR_KEYS, identity)), "group": group, "step_index": step,
                    "fraction": float((_boolean(rows, "shape_complete") & _number(rows, "first_peak_step").eq(step)).sum()) / len(rows),
                    "sample_count": len(rows)})
            for name in shape_fields:
                finite = _number(rows, name)
                record[name + "_median"] = float(finite.median()) if finite.notna().any() else np.nan
            prompt_rows.append(record)
    prompt = pd.DataFrame(prompt_rows)
    if not prompt.empty:
        for identity, rows in prompt.groupby(PAIR_KEYS, dropna=False, sort=False):
            if set(rows.group) == set(GROUPS):
                values = rows.set_index("group")
                for name in shape_fields:
                    mixed_rows.append({**dict(zip(PAIR_KEYS, identity)), "metric": name,
                        "high_sscd_median": values.loc[GROUPS[0], name + "_median"],
                        "lower_sscd_median": values.loc[GROUPS[1], name + "_median"]})
    save("branch_gap_peak_step", pd.DataFrame(records), prediction_steps=steps, shape_group_counts=group_info,
         population_counts=_population(shapes), missing_outcome_row_count=int((~np.isfinite(_number(shapes, "terminal_sscd"))).sum()),
         shape_population="Individual complete raw trajectories; unassigned flat/incomplete/unresolved mass remains in the denominator",
         shape_rule="No smoothing, group peak alignment, monotonicity assumption or SSCD-dependent classifier threshold",
         prompt_peak_distribution="audit_data/prompt_peak_distribution.csv", shape_table="audit_data/trajectory_shapes.csv", prompt_shape_table="audit_data/prompt_trajectory_shapes.csv", mixed_prompt_shape_table="audit_data/mixed_prompt_trajectory_shapes.csv")
    auxiliary["prompt_trajectory_shapes"] = prompt
    auxiliary["prompt_peak_distribution"] = pd.DataFrame(prompt_peak_rows)
    auxiliary["mixed_prompt_trajectory_shapes"] = pd.DataFrame(mixed_rows)


def _motion_inputs(motion, steps, save, auxiliary):
    stems = dict(zip(GROUPS, ("branch_gap_motion_high_sscd", "branch_gap_motion_lower_sscd")))
    if motion.empty:
        for stem in stems.values():
            save(stem, pd.DataFrame(), status="not_applicable" if steps == 1 else "unavailable", reason="One prediction has no consecutive branch motion" if steps == 1 else "Missing consecutive-prediction vector accounting")
        return
    _require(motion, SAMPLE_KEYS + ["step_index", "terminal_sscd", *MOTION_FIELDS.values(), "identity_residual", "identity_tolerance", "identity_status"], "branch_motion")
    auxiliary["branch_motion"] = motion.copy()
    if not _number(motion, "step_index").isin(range(max(steps - 1, 0))).all():
        raise TheoryError("Branch motion includes a nonexistent consecutive-prediction pair")
    valid = pd.Series(True, index=motion.index)
    for column in MOTION_FIELDS.values():
        valid &= np.isfinite(_number(motion, column))
    failures = motion.identity_status.fillna("").astype(str).str.startswith("failed")
    failures |= valid & (abs(_number(motion, "identity_residual")) > _number(motion, "identity_tolerance"))
    for group, chosen in _outcomes(motion):
        records = []
        for step, population in chosen.groupby("step_index", sort=True):
            selected = population.loc[valid.loc[population.index]]
            weights = _weights(selected)
            denominator = float(weights.sum())
            for metric, column in MOTION_FIELDS.items():
                values = _number(selected, column)
                records.append({"group": group, "metric": metric, "step_index": step,
                    "mean": float((weights * values).sum()) / denominator if denominator else np.nan,
                    "minimum": float(values.min()) if len(values) else np.nan, "maximum": float(values.max()) if len(values) else np.nan,
                    "denominator_weight": denominator, "eligible_count": len(selected), "excluded_count": len(population)-len(selected)})
        save(stems[group], pd.DataFrame(records), status="blocked" if failures.loc[chosen.index].any() else "not_applicable" if chosen.empty else None,
             reason="Consecutive branch-motion identity failed its recorded arithmetic budget" if failures.loc[chosen.index].any() else "Empty fixed outcome group" if chosen.empty else None,
             prediction_steps=steps, group=group, motion_audit_table="audit_data/branch_motion.csv",
             weighting="Prompt-balanced weighted means on identical finite rows for C, U, quadratic and actual change; means preserve additivity",
             interpretation="Descriptive accounting including changing native noise labels; not causal or theorem-level prevalence evidence")
    if failures.any():
        auxiliary["branch_motion_failed"] = motion.loc[failures].copy()



def _terminal_all_inapplicable(terminal):
    """Only explicit saved false flags establish mathematical inapplicability."""
    column = "evidence_terminal_applicable"
    if terminal.empty or column not in terminal:
        return False
    flags = terminal[column]
    normalized = flags.astype("string").str.strip().str.lower().map({
        "false": False, "true": True, "0": False, "1": True, "0.0": False, "1.0": True,
    })
    if (flags.notna() & normalized.isna()).any():
        raise TheoryError("evidence_terminal_applicable must contain saved Boolean values or missing observations")
    return bool(flags.notna().all() and normalized.eq(False).all())


def _terminal_inputs(terminal, frames, metadata, config, save, auxiliary):
    info = copy.deepcopy(metadata["theorem7_final_reproduction"])
    info.pop("counts", None)
    for column, key in (("evidence_terminal_correction_source", "terminal_correction_sources"), ("evidence_terminal_certainty", "terminal_certainty")):
        info[key] = sorted(terminal[column].dropna().astype(str).unique()) if column in terminal else []
    status = info.pop("status")
    reason = info.pop("reason", None)
    chosen = auxiliary.get("terminal_common_population", pd.DataFrame()).copy()
    auxiliary["terminal_metrics"] = terminal.copy()
    all_inapplicable = _terminal_all_inapplicable(terminal)
    if chosen.empty and status != "blocked" and all_inapplicable:
        for legacy in ("theorem7_final_reproduction", "theorem7_terminal_components"):
            metadata[legacy].update(status="not_applicable", reason="Every saved terminal update is outside the supported correction contract")
    for stem, field, name in (("final_reproduction_bound", "evidence_terminal_corrected_ref_bound_rmse", "reference"),
                              ("terminal_observable_bound", "evidence_terminal_corrected_obs_bound_rmse", "observable")):
        if chosen.empty:
            save(stem, pd.DataFrame(), status="blocked" if status == "blocked" else "not_applicable" if all_inapplicable else "unavailable", reason=reason or "No supported terminal correction population", **info)
            continue
        frame = _pair_frame(chosen, field, "direct_theorem7_endpoint_error_rmse")
        frame["terminal_scope"] = chosen.evidence_terminal_scope
        frame["applicable"] = True
        frame["bound_slack_rmse"] = frame.x - frame.y
        frame["bound_error_ratio"] = frame.x / frame.y.where(frame.y != 0)
        frame["zero_error_ratio_undefined"] = frame.y.eq(0)
        frame["zero_bound"] = frame.x.eq(0)
        finite = np.isfinite(_number(frame, "x")) & np.isfinite(_number(frame, "y"))
        positive = np.concatenate([_number(frame.loc[finite], "x"), _number(frame.loc[finite], "y")])
        positive = positive[positive > 0]
        axis_scale = "log" if len(positive) and positive.max() / positive.min() >= 100 else "linear"
        save(stem, frame, status=status, reason=reason, bound_kind=name,
             axis_scale=axis_scale, log_policy="Log axes when common positive range spans at least two decades; exact zeros are edge markers in axes coordinates, never epsilon replacements",
             zero_counts={"x": int(frame.x.eq(0).sum()), "y": int(frame.y.eq(0).sum()), "both": int((frame.x.eq(0) & frame.y.eq(0)).sum())},
             infinite_bound_count=int(np.isposinf(_number(frame, "x")).sum()),
             tolerance_rmse=info.get("predeclared_tolerance_rmse"), **info)
        auxiliary[stem + "_sample_audit"] = frame


def _terminal_grouped_coverage(terminal, frames, metadata, save, auxiliary):
    """Split the existing valid terminal population by its same-seed SSCD."""
    source = "theorem7_final_reproduction"
    pooled = copy.deepcopy(metadata[source])
    def metadata_value(value):
        # Match publication JSON: unavailable nonfinite metadata is null;
        # exact +infinity support points remain in the separate pooled CSV.
        if isinstance(value, dict):
            return {str(key): metadata_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [metadata_value(item) for item in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value
    auxiliary["terminal_pooled_cdf"] = frames[source].copy()
    auxiliary["terminal_pooled_cdf_metadata"] = pd.DataFrame([{
        "source_comparison": source,
        "metadata_json": json.dumps(metadata_value(json_value(pooled)), sort_keys=True, allow_nan=False),
    }])
    fields = {"actual": "direct_theorem7_endpoint_error_rmse",
              "observable": "evidence_terminal_corrected_obs_bound_rmse",
              "reference": "evidence_terminal_corrected_ref_bound_rmse"}
    chosen = auxiliary.get("terminal_common_population", terminal.iloc[:0]).copy()
    if not chosen.empty:
        _require(chosen, SAMPLE_KEYS + list(fields.values()), "terminal common population")
        if chosen[SAMPLE_KEYS].isna().any().any() or chosen.duplicated(SAMPLE_KEYS).any():
            raise TheoryError("Grouped terminal coverage requires unique complete saved sample identities")
        valid = np.isfinite(_number(chosen, fields["actual"]))
        for field in fields.values():
            values = _number(chosen, field)
            valid &= ~np.isnan(values) & values.ge(0)
        if not valid.all():
            raise TheoryError("Saved terminal common population contains invalid bound/error values")
    finite_sscd = np.isfinite(_number(chosen, "terminal_sscd"))
    excluded = chosen.loc[~finite_sscd].copy()
    excluded["exclusion_reason"] = "missing_or_nonfinite_same_seed_terminal_sscd"
    auxiliary["terminal_grouped_sscd_exclusions"] = excluded
    populations, zero_mass, infinite_mass, predeclared, group_ordering = {}, {}, {}, {}, {}
    curves, receipts = [], []
    tolerance = pooled.get("predeclared_tolerance_rmse")
    ordering = auxiliary.get("theorem7_paired_bound_ordering", pd.DataFrame())
    for group, rows in _outcomes(chosen):
        rows = rows.copy()
        weights = _weights(rows) if len(rows) else pd.Series(dtype=float, index=rows.index)
        denominator = float(weights.sum())
        populations[group] = {"eligible_count": int(len(rows)),
            "prompt_count": int(len(rows[PAIR_KEYS].drop_duplicates())) if len(rows) else 0,
            "denominator_weight": denominator}
        zero_mass[group], infinite_mass[group], predeclared[group] = {}, {}, {}
        for name, field in fields.items():
            values = _number(rows, field)
            zero_mass[group][name] = float(weights[values.eq(0)].sum()) / denominator if denominator else None
            infinite_mass[group][name] = float(weights[np.isposinf(values)].sum()) / denominator if denominator else None
            if tolerance is not None:
                predeclared[group][name] = float(weights[values.le(tolerance)].sum()) / denominator if denominator else None
            if len(rows):
                curves.append(_ecdf(values, weights, group=group, distribution=name, **populations[group]))
        rows["group"], rows["prompt_weight"] = group, weights
        receipts.append(rows)
        actual, observable, reference = (_number(rows, fields[name]) for name in fields)
        group_ordering[group] = {"actual_above_observable": int((actual > observable).sum()),
            "observable_above_reference": int((observable > reference).sum())}
        if len(rows) and not ordering.empty:
            paired = rows[SAMPLE_KEYS].merge(ordering, on=SAMPLE_KEYS, how="left", validate="one_to_one")
            for column in ("actual_above_observable_beyond_sensitivity", "observable_above_reference_beyond_roundoff",
                           "permitted_noise_event", "blocking_consistency_failure"):
                if column in paired:
                    group_ordering[group][column + "_count"] = int(_boolean(paired, column).sum())
    common = pd.concat(receipts, ignore_index=True) if receipts else chosen.iloc[:0].copy()
    auxiliary["terminal_grouped_common_population"] = common
    columns = ["group", "distribution", "value", "cdf", "mass", "count", "denominator_weight", "eligible_count", "prompt_count"]
    frame = pd.concat(curves, ignore_index=True)[columns] if curves else pd.DataFrame(columns=columns)
    info = copy.deepcopy(pooled)
    status, reason = info.pop("status"), info.pop("reason", None)
    # These pooled quantities remain in the immutable audit, not as ambiguous
    # denominators or masses for the newly grouped active figure.
    for key in ("counts", "denominator_weight", "zero_mass", "infinite_mass", "predeclared_coverage"):
        info.pop(key, None)
    if frame.empty and status not in {"blocked", "not_applicable"}:
        status = "unavailable"
        reason = "No eligible terminal samples with finite same-seed SSCD in either fixed outcome group"
    info.update(formula_version="projected-gap-error-1", source_comparison=source,
        counts={"total": int(len(terminal)), "common_eligible": int(len(chosen)),
                "eligible": int(len(common)), "excluded": int(len(terminal) - len(common)),
                "excluded_before_sscd_grouping": int(len(terminal) - len(chosen)),
                "missing_or_nonfinite_sscd": int((~finite_sscd).sum())},
        group_population=populations, grouped_zero_mass=zero_mass, grouped_infinite_mass=infinite_mass,
        grouped_paired_ordering_audit=group_ordering,
        empty_groups=[group for group in GROUPS if not populations[group]["eligible_count"]],
        same_population="All three corrected terminal quantities within each SSCD group use exactly the same saved eligible sample identities and prompt weights",
        weighting="Within each same-seed SSCD group, one mass unit per represented prompt-target pair divided over its eligible member seeds; identical weights for all three curves",
        outcome_grouping="Same-seed saved terminal_sscd > 0.75 versus <= 0.75; missing or nonfinite scores are explicitly excluded without reassignment",
        denominator_column="denominator_weight",
        group_common_population_table="audit_data/terminal_grouped_common_population.csv",
        group_sscd_exclusion_table="audit_data/terminal_grouped_sscd_exclusions.csv",
        pooled_cdf_audit_table="audit_data/terminal_pooled_cdf.csv",
        pooled_metadata_audit_table="audit_data/terminal_pooled_cdf_metadata.csv")
    if tolerance is not None:
        info["grouped_predeclared_coverage"] = predeclared
    save("terminal_bound_coverage", frame, status=status, reason=reason, **info)


def _counterfactual_inputs(frame, enabled, save, auxiliary):
    stem = "counterfactual_unconditional_response"
    if frame.empty:
        save(stem, pd.DataFrame(), status="unavailable" if enabled else "not_applicable", reason="Enabled learned counterfactual measurements are missing" if enabled else "Optional learned-network counterfactual was not requested")
        return
    _require(frame, SAMPLE_KEYS + ["step_index", "terminal_sscd", "status", "counterfactual_applicable", "counterfactual_I_net", "counterfactual_target_error_rmse", "actual_next_target_error_rmse"], stem)
    chosen = frame.loc[frame.status.eq("measured") & _boolean(frame, "counterfactual_applicable")].copy()
    finite = np.isfinite(_number(chosen, "counterfactual_I_net")) & np.isfinite(_number(chosen, "counterfactual_target_error_rmse")) & np.isfinite(_number(chosen, "actual_next_target_error_rmse"))
    chosen = chosen.loc[finite]
    failures = frame.status.eq("failed")
    save(stem, _pair_frame(chosen, "counterfactual_target_error_rmse", "actual_next_target_error_rmse"),
         status="blocked" if enabled and failures.any() else "not_applicable" if not _boolean(frame, "counterfactual_applicable").any() else None,
         reason="Explicitly requested learned-counterfactual task failed" if enabled and failures.any() else None,
         counts={"total": len(frame), "measured": len(chosen), "excluded": len(frame)-len(chosen)},
         status_counts=_statuses(frame), fixed_snapshots=sorted(frame.step_index.unique().tolist()),
         optional_network_scope="Actual empty-prompt network predictions at both recorded endpoints under the same inference context; no posterior-reference substitution or rollout",
         improvement_statistics=_quantile_description(_number(chosen, "counterfactual_I_net")))
    auxiliary["counterfactual_unconditional"] = frame.copy()
