"""Analysis-only scalar reductions for the fixed seven-statement evidence suite.

No paths, tensors, model calls or posterior evaluations enter this module. The
renderer consumes these saved tables and never calls this statistical reducer.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from .contracts import TheoryError
from .direct_figures import (
    PAIR_KEYS, RECOVERY_TOLERANCES, _boolean, _counts, _number, _outcomes, _pair_frame,
    _quantities, _require, _require_projected_gap_error, _source, _statuses, _unique, build_direct_plot_inputs,
)
from .paper_registry import AUDIT_ALIASES, evidence_registry as paper_registry
from .summaries import weighted_quantiles

EVIDENCE_FORMULA_VERSION = "projected-gap-error-1"
DIRECTIONAL_VARIATION_DEFINITION = "positive_part_after_integrated_unit_gap_projection"
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = .95
LEMMA4_RELATIVE_DENOMINATOR_FLOOR_RMSE = 1e-12
SCALAR_QA_OPERATION_MULTIPLIER = 128
SCALAR_QA_SCOPE = "Fixed 128*eps_float64*max(1,sum(abs(operands))) operation-sensitivity estimate; not a rigorous arithmetic enclosure or source-perturbation certificate"
GEOMETRY_LEVELS = (.25, .5, 1.)
SAMPLE_KEYS = PAIR_KEYS + ["seed"]
BOOTSTRAP_POLICY = {
    "definition": "Monte Carlo bootstrap intervals",
    "replicates": BOOTSTRAP_REPLICATES,
    "seed": BOOTSTRAP_SEED,
    "confidence": BOOTSTRAP_CONFIDENCE,
    "percentile_method": "linear interpolation of fixed bootstrap replicate quantiles; NumPy default_rng PCG64",
    "scope": "Pair-specific Monte Carlo uncertainty only; no pooled significance or population confidence claim.",
    "seed_dependence": "Sorted Gaussian seed IDs use the same resampling stream across pairs; pair-specific forward draws have identity-derived independent streams.",
    "shared_targets": "Pair and target identities are preserved; no population bootstrap that would ignore shared-target clusters is computed.",
}
TERMINAL_SCOPES = (
    "original_clean_terminal_theorem",
    "finite_terminal_update_extension_deterministic",
    "finite_terminal_update_extension_gaussian_noise_bound",
)


def _weights(rows):
    """One unit of mass per complete prompt, divided over its member seeds."""
    return 1.0 / rows.groupby(PAIR_KEYS, dropna=False).seed.transform("size")


def _interval(values, *, seed, scale=1.0):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = np.sqrt(values[draws].mean(axis=1) / scale)
    alpha = (1 - BOOTSTRAP_CONFIDENCE) / 2
    return tuple(float(v) for v in np.quantile(means, [alpha, 1 - alpha]))


def _pair_seed(identity):
    text = "|".join(str(v) for v in identity) + f"|{BOOTSTRAP_SEED}|forward-bootstrap"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _range(values):
    unique = sorted(set(int(v) for v in values))
    return unique[0] if len(unique) == 1 else f"{unique[0]}–{unique[-1]}" if unique else 0


def _ecdf(values, weights, **identity):
    """Exact right-continuous weighted ECDF; zeros and +inf stay explicit."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    valid = ~np.isnan(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values):
        return pd.DataFrame(columns=[*identity, "value", "cdf", "mass", "count"])
    grouped = pd.DataFrame({"value": values, "weight": weights}).groupby("value", sort=True).agg(weight=("weight", "sum"), count=("weight", "size")).reset_index()
    grouped["mass"] = grouped.weight / grouped.weight.sum()
    grouped["cdf"] = grouped.mass.cumsum()
    grouped.loc[grouped.index[-1], "cdf"] = 1.0
    for key, value in identity.items():
        grouped[key] = value
    return grouped.drop(columns="weight")


def _curve_summary(rows, fields, *, group=None, source_range=None, common=True):
    """Save all observed noise rows, including explicit NaN gaps."""
    result = []
    for step, population in rows.groupby("step_index", sort=True):
        values = {metric: _number(population, field) for metric, field in fields.items()}
        valid = pd.Series(True, index=population.index)
        if common:
            for value in values.values():
                valid &= np.isfinite(value)
        selected = population.loc[valid]
        weights = _weights(selected) if set(SAMPLE_KEYS).issubset(selected) else pd.Series(1.0, index=selected.index)
        snrs = _number(population, "snr")
        if len(snrs.dropna().unique()) != 1:
            raise TheoryError("Evidence aggregation found inconsistent native SNR at one step")
        for metric, value in values.items():
            value = value.loc[selected.index]
            finite = np.isfinite(value)
            q25, median, q75 = weighted_quantiles(value, weights)
            record = {"metric": metric, "step_index": step, "snr": float(snrs.iloc[0]),
                      "median": median, "q25": q25, "q75": q75,
                      "minimum": float(value[finite].min()) if finite.any() else np.nan,
                      "maximum": float(value[finite].max()) if finite.any() else np.nan,
                      "denominator_weight": float(weights.sum()), "population_count": len(population),
                      "eligible_count": len(selected), "excluded_nonfinite_count": int((~valid).sum()),
                      "finite_count": int(finite.sum()), "missing_count": int((~finite).sum())}
            if group is not None:
                record["group"] = group
            if source_range is not None:
                record["source_range"] = source_range
            result.append(record)
    return pd.DataFrame(result)


def _segments(frame):
    frame = frame.sort_values(["metric", "step_index"]).copy()
    frame["segment_id"] = 0
    for _, rows in frame.groupby("metric", sort=False):
        steps = _number(rows, "step_index")
        # Missing entire saved labels and explicit missing values both break lines.
        boundary = steps.diff().ne(1) | ~np.isfinite(_number(rows, "median"))
        boundary |= ~np.isfinite(_number(rows, "median").shift())
        frame.loc[rows.index, "segment_id"] = boundary.cumsum().to_numpy()
    return frame


def _operation_tolerance(*values):
    """Fixed, declared scalar-operation QA; never fit to the observed residual."""
    scale = sum(abs(value) for value in values)
    return SCALAR_QA_OPERATION_MULTIPLIER * np.finfo(np.float64).eps * np.maximum(1., scale)


def _quantile_description(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"count": len(values), **{
        name: float(np.quantile(values, q)) if len(values) else None
        for name, q in (("minimum", 0), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95), ("q99", .99), ("maximum", 1))}}


def _injection_audit(initial, applicable, auxiliary):
    required = ["evidence_injection_direction_l2", "evidence_injection_conditional_target_error_l2",
                "evidence_injection_unconditional_mean_error_l2", "direct_cor3_error_mean_cross_inner_product"]
    _require(initial, required, "Corollary 3 saved premise and vector-error inner product")
    rows = initial.copy()
    direction = _number(rows, "evidence_injection_direction_l2")
    ec = _number(rows, "evidence_injection_conditional_target_error_l2")
    bu = _number(rows, "evidence_injection_unconditional_mean_error_l2")
    cross = _number(rows, "direct_cor3_error_mean_cross_inner_product")
    relative = _number(rows, "evidence_injection_relative_error")
    valid = applicable & (direction > 0) & np.isfinite(direction) & np.isfinite(ec) & np.isfinite(bu) & np.isfinite(cross) & np.isfinite(relative)
    squared_direction = np.square(direction.where(valid))
    rows["premise_ratio"] = (ec + bu) / direction.where(valid)
    rows["error_decomposition_squared"] = (np.square(ec) + np.square(bu) - 2 * cross) / squared_direction
    rows["relative_error_squared"] = np.square(relative)
    rows["error_decomposition_residual"] = rows.relative_error_squared - rows.error_decomposition_squared
    rows["error_decomposition_tolerance"] = _operation_tolerance(rows.relative_error_squared,
        np.square(ec) / squared_direction, np.square(bu) / squared_direction, 2 * cross / squared_direction)
    rows["premise_slack"] = rows.premise_ratio - relative
    rows["premise_tolerance"] = _operation_tolerance(rows.premise_ratio, relative)
    failed = valid & ((abs(rows.error_decomposition_residual) > rows.error_decomposition_tolerance) | (rows.premise_slack < -rows.premise_tolerance))
    rows["premise_audit_status"] = np.select([~applicable, ~valid, failed],
        ["not_applicable_direction_or_guidance", "unavailable_finite_saved_premise", "failed_fixed_float64_consistency"], default="consistent_with_float64_estimate")
    rows["arithmetic_error_scope"] = SCALAR_QA_SCOPE
    auxiliary["corollary3_premise_error_decomposition"] = rows
    pair_records = []
    coverage_records = []
    selected = rows.loc[applicable & np.isfinite(relative)]
    weights = _weights(selected)
    for level in GEOMETRY_LEVELS:
        within = _number(selected, "evidence_injection_relative_error") <= level
        coverage_records.append({"level": level, "population": "sample", "count": len(selected), "covered_count": int(within.sum()),
            "fraction": float(within.mean()) if len(selected) else np.nan,
            "prompt_balanced_fraction": float(weights[within].sum() / weights.sum()) if len(selected) else np.nan})
    for identity, group in selected.groupby(PAIR_KEYS, sort=False, dropna=False):
        values = _number(group, "evidence_injection_relative_error")
        record = {**dict(zip(PAIR_KEYS, identity)), **_quantile_description(values)}
        pair_records.append(record)
        for level in GEOMETRY_LEVELS:
            coverage_records.append({**dict(zip(PAIR_KEYS, identity)), "level": level, "population": "pair",
                "count": len(group), "covered_count": int((values <= level).sum()), "fraction": float((values <= level).mean())})
    pair_summary = pd.DataFrame(pair_records)
    coverage = pd.DataFrame(coverage_records)
    auxiliary["corollary3_pair_relative_error"] = pair_summary
    auxiliary["corollary3_contour_coverage"] = coverage
    info = {
        "sample_relative_error_quantiles": _quantile_description(_number(selected, "evidence_injection_relative_error")),
        "pair_relative_error_quantiles": _quantile_description(pair_summary.get("median", pd.Series(dtype=float))),
        "pair_quantile_definition": "Distribution of within-pair seed medians, with equal pair weight; every pair also has its full saved sample quantiles",
        "relative_error_quantile_method": "Fixed linear interpolation; sample quantiles weight each retained eligible seed equally",
        "geometry_contour_coverage": coverage.loc[coverage.population.eq("sample"), ["level", "count", "covered_count", "fraction", "prompt_balanced_fraction"]].to_dict("records"),
        "geometry_pair_table": "audit_data/corollary3_pair_relative_error.csv",
        "geometry_coverage_table": "audit_data/corollary3_contour_coverage.csv",
        "injection_premise_table": "audit_data/corollary3_premise_error_decomposition.csv",
        "injection_premise": {"definition": "A=(e_c+b_u)/||x_star-mu_K||; E_rel<=A and E_rel^2=(e_c^2+b_u^2-2<mc-target,mu-mu_K>)/||x_star-mu_K||^2",
            "failed_count": int(failed.sum()), "unavailable_count": int((applicable & ~valid).sum()),
            "status_counts": {str(key): int(value) for key, value in rows.premise_audit_status.value_counts().items()},
            "arithmetic_error_scope": SCALAR_QA_SCOPE,
            "interpretation": "Attribution of measured finite initial errors; the same-input bound audit is not independent empirical proof or a reproduction threshold"}}
    return info, failed


def build_evidence_plot_inputs(tables, config, provenance):
    """Return compact frames and metadata, retaining all prior direct audits."""
    config = config.get("scientific_config", config)
    old_frames, old_meta = build_direct_plot_inputs(tables, config, provenance)
    frames = {name: frame.copy() for name, frame in old_frames.items()}
    metadata = {name: dict(info) for name, info in old_meta["figures"].items()}
    auxiliary = dict(old_meta["auxiliary_tables"])
    audit = dict(old_meta["audit"])
    audit["inherited_nonfigure_blocking"] = bool(old_meta["audit"].get("blocking", False))
    audit.update(evidence_formula_version=EVIDENCE_FORMULA_VERSION, bootstrap_policy=BOOTSTRAP_POLICY)
    for original, alias in AUDIT_ALIASES.items():
        frames[alias], metadata[alias] = old_frames[original].copy(), dict(old_meta["figures"][original])
    copies = {
        "corollary3_full_guided_error": "corollary3_guided_estimate_error",
        "proposition5_condition_vs_gain": "proposition5_posterior_feedback",
        "lemma6_bound_vs_gap": "lemma6_target_specific_synchronization",
    }
    for name, original in copies.items():
        frames[name], metadata[name] = old_frames[original].copy(), dict(old_meta["figures"][original])
    # Preserve the old comparison and its descriptive alleged-violation counts.
    # The direct builder currently returns blocking=False; it does not certify
    # an implication or issue a publication blocker. Do not blanket-clear an
    # upstream blocker if that contract later changes. New P5 statuses below
    # are assigned from the independently scoped refinement fields only.
    auxiliary["proposition5_legacy_condition_gain"] = old_frames["proposition5_posterior_feedback"].copy()
    audit["proposition5_interpretation_migration"] = {
        "legacy_figure_status": old_meta["figures"]["proposition5_posterior_feedback"]["status"],
        "legacy_audit_blocking": bool(old_meta["audit"].get("blocking", False)),
        "legacy_comparison_table": "audit_data/proposition5_legacy_condition_gain.csv",
        "legacy_alleged_violation_summary": "audit_data/statement_summary.csv",
        "current_row_audit": "audit_data/proposition5_numerical_resolution_rows.csv",
        "interpretation": "Legacy source-sensitive assessments and possible mismatched-endpoint comparisons remain recorded; they are not fixed-cache certificates and are not converted to missing observations.",
        "publication_rule": "Use the new numerical_publication_blocker and justified same-endpoint implication gate. Legacy missing V or heuristic ambiguity cannot override a resolved negative precheck. Independently inherited non-P5 blockers remain intact.",
    }
    # Preserve the complete transfer audit before scatter export compaction.
    transfer_keys = PAIR_KEYS + ["step_index"]
    saved_loss = tables["forward_loss_summary"]
    _require(saved_loss, transfer_keys + ["KL_target_to_gaussian"], "forward_loss_summary transfer KL")
    _unique(saved_loss, transfer_keys, "forward_loss_summary transfer KL")
    transfer = old_frames["theorem1_gaussian_transfer"].merge(
        saved_loss[transfer_keys + ["KL_target_to_gaussian"]],
        on=transfer_keys, how="left", validate="many_to_one")
    kl = _number(transfer, "KL_target_to_gaussian")
    if kl.isna().any() or (kl < 0).any():
        raise TheoryError("Transfer audit requires the true saved nonnegative KL_target_to_gaussian; clipped TV cannot reconstruct it")
    transfer["pinsker_tv_bound_unclipped"] = np.sqrt(kl / 2)
    if not np.allclose(_number(transfer, "pinsker_tv_bound"), np.minimum(1., transfer.pinsker_tv_bound_unclipped), rtol=1e-12, atol=0):
        raise TheoryError("Saved clipped Pinsker bound is inconsistent with the true saved Gaussian KL")
    auxiliary["theorem1_gaussian_transfer_audit"] = transfer.copy()
    frames["theorem1_gaussian_transfer"] = transfer
    metadata["theorem1_gaussian_transfer"].update(
        complete_transfer_audit_table="audit_data/theorem1_gaussian_transfer_audit.csv",
        pinsker_tv_bound_unclipped_definition="sqrt(saved KL_target_to_gaussian/2), before clipping to the probability range",
        transfer_audit_scope="All pair/tolerance rows retain raw and clipped probability bounds, raw and clipped TV, Gaussian failure counts, and Monte Carlo uncertainty; plug-in rather than certified statistical bounds")
    reference = old_meta["figures"]["lemma2_unconditional_baseline"]["reference_provenance"]
    initial_snr = float(old_frames["theorem1_loss_recovery"].snr.iloc[0])

    def save(stem, frame, *, status=None, reason=None, **info):
        frames[stem] = frame.copy()
        if status is None:
            candidates = [name for name in ("x", "y", "median", "fraction", "resolved_fraction", "cdf") if name in frame]
            available = any(np.isfinite(_number(frame, name)).any() for name in candidates)
            if {"x", "y"}.issubset(frame):
                available = bool((np.isfinite(_number(frame, "x")) & np.isfinite(_number(frame, "y"))).any())
            status = "available" if available else "unavailable"
        metadata[stem] = {"status": status, "reason": reason or (None if status == "available" else "No eligible measured observations for this fixed evidence design"),
                          "counts": _counts(frame), "status_counts": _statuses(frame),
                          "formula_version": EVIDENCE_FORMULA_VERSION, "reference_provenance": reference,
                          "evidence_classification": "fixed_consequence_with_retained_direct_audit", "actual_initial_snr": initial_snr, **info}
        if status == "blocked":
            audit["blocking"] = True

    # Required independent draws are bootstrap inputs, never reconstructed from y.
    gaussian = old_frames["theorem1_loss_recovery"].copy()
    draws = tables.get("forward_loss_draws", pd.DataFrame()).copy()
    _require(draws, PAIR_KEYS + ["step_index", "draw_index", "loss_squared_l2", "input_source"], "forward_loss_draws")
    _source(draws, "forward_target", "forward_loss_draws")
    _unique(draws, PAIR_KEYS + ["step_index", "draw_index"], "forward_loss_draws")
    draws = draws.loc[_number(draws, "step_index").eq(0)]
    _require(gaussian, ["unconditional_target_error_rmse", "gaussian_control_status"], "gaussian_conditional matched control")
    pair_records, branch_records = [], []
    draw_groups = {tuple(str(v) for v in key): group for key, group in draws.groupby(PAIR_KEYS, sort=False, dropna=False)}
    for identity, rows in gaussian.groupby(PAIR_KEYS, sort=False, dropna=False):
        rows = rows.sort_values("seed", key=lambda values: pd.to_numeric(values, errors="raise"))
        first = rows.iloc[0]
        loss_draws = draw_groups.get(tuple(str(v) for v in identity), pd.DataFrame())
        if loss_draws.empty or len(loss_draws) != int(first.draw_count):
            raise TheoryError("Pair bootstrap requires every saved independent forward-loss draw")
        loss_draws = loss_draws.sort_values("draw_index", key=lambda values: pd.to_numeric(values, errors="raise"))
        losses = _number(loss_draws, "loss_squared_l2")
        if not np.isfinite(losses).all() or (losses < 0).any():
            raise TheoryError("Invalid forward-loss draw cannot be used as a bootstrap placeholder")
        c, u = _number(rows, "conditional_error_rmse"), _number(rows, "unconditional_target_error_rmse")
        valid = np.isfinite(c) & np.isfinite(u)
        valid &= rows.gaussian_control_status.isin(["same_saved_gaussian_input_canonical_epsilon", "measured_independent_gaussian_control"])
        chosen = rows.loc[valid]
        x_low, x_high = _interval(losses, seed=_pair_seed(identity), scale=float(first.latent_dimension) * float(first.snr))
        y_low, y_high = _interval(np.square(c[valid]), seed=BOOTSTRAP_SEED)
        sscd = _number(chosen, "terminal_sscd")
        score_valid = np.isfinite(sscd)
        record = {**dict(zip(PAIR_KEYS, identity)), "step_index": 0, "snr": float(first.snr),
                  "x": float(first.loss_scale_rmse), "y": math.sqrt(float(np.square(c[valid]).mean())) if valid.any() else np.nan,
                  "control_y": math.sqrt(float(np.square(u[valid]).mean())) if valid.any() else np.nan,
                  "mean_terminal_sscd": float(sscd[score_valid].mean()) if score_valid.any() else np.nan,
                  "x_low": x_low, "x_high": x_high, "y_low": y_low, "y_high": y_high,
                  "gaussian_count": int(valid.sum()), "total_gaussian_count": len(rows), "excluded_gaussian_count": int((~valid).sum()),
                  "sscd_count": int(score_valid.sum()), "forward_draw_count": len(losses), "latent_dimension": int(first.latent_dimension),
                  "status": "available" if valid.any() else "unavailable_matched_unconditional_gaussian_control",
                  "input_source": "independent_forward_and_gaussian_pair_summaries"}
        pair_records.append(record)
        branch = _pair_frame(rows, "unconditional_target_error_rmse", "conditional_error_rmse")
        branch_records.append(branch)
    pair_frame = pd.DataFrame(pair_records)
    initial_snr = float(pair_frame.snr.iloc[0])
    counts = {"pairs": len(pair_frame), "gaussian_seeds_per_pair": _range(pair_frame.gaussian_count), "forward_draws": _range(pair_frame.forward_draw_count), "excluded_gaussian_rows": int(pair_frame.excluded_gaussian_count.sum())}
    save("theorem1_loss_recovery", pair_frame, actual_initial_snr=initial_snr, counts=counts,
         bootstrap_interval_definition="Monte Carlo bootstrap intervals", bootstrap_policy=BOOTSTRAP_POLICY,
         interpretation="Pair-specific empirical finite-sample RMS summaries; convergence in probability does not imply this moment conclusion.")
    save("theorem1_initial_branch_comparison", pd.concat(branch_records, ignore_index=True), actual_initial_snr=initial_snr)
    control_statuses = gaussian.get("gaussian_control_status", pd.Series("missing_control_provenance", index=gaussian.index)).fillna("unavailable").value_counts().to_dict()
    for stem in ("theorem1_loss_recovery", "theorem1_initial_branch_comparison"):
        metadata[stem]["matched_control_status_counts"] = {str(key): int(value) for key, value in control_statuses.items()}
        metadata[stem]["incomplete_pair_count"] = int(pair_frame.excluded_gaussian_count.gt(0).sum())
        if pair_frame.excluded_gaussian_count.gt(0).any():
            metadata[stem].update(status="unavailable", reason="Required same-input Gaussian conditional/unconditional controls are incomplete; rerun evidence analysis to measure missing controls. Partial pair summaries remain saved but are not published.")
    auxiliary["theorem1_pair_rms"] = pair_frame.copy()
    auxiliary["theorem1_gaussian_seed_tails"] = gaussian.copy()
    association = gaussian.copy()
    association["branch_error_difference"] = _number(association, "unconditional_target_error_rmse") - _number(association, "conditional_error_rmse")
    association["sscd"] = _number(association, "terminal_sscd")
    common_association = np.isfinite(association.branch_error_difference) & np.isfinite(association.sscd)
    association["association_eligible"] = common_association
    association["x_centered"] = np.nan
    association["y_centered"] = np.nan
    valid_association = association.loc[common_association]
    for field, output in (("branch_error_difference", "x_centered"), ("sscd", "y_centered")):
        association.loc[common_association, output] = valid_association[field] - valid_association.groupby(PAIR_KEYS, dropna=False)[field].transform("mean")
    auxiliary["theorem1_within_prompt_associations"] = association
    audit["within_prompt_association_interpretation"] = "Descriptive paired within-prompt centered errors and same-seed SSCD; no fitted significance or independent-prompt claim"

    # The primary asks the initial high-noise question. All later native probes
    # remain a separate mandatory appendix, including unavailable-label gaps.
    analytical = tables.get("reference_analytical", pd.DataFrame()).copy()
    native = tables["gaussian_reference"].copy()
    native_curves = _segments(_curve_summary(native, {
        "reference": "reference_mean_error_rmse", "learned": "learned_mean_error_rmse",
        "reference_error": "unconditional_reference_error_rmse"}, source_range="native", common=False))
    save("lemma2_native_gaussian_sweep", native_curves,
         quantile_policy="Inverse weighted ECDF over equal unique Gaussian seed weights",
         band_definition="Descriptive IQR over unique Gaussian seeds",
         input_law_separation="Complete saved native-label Gaussian probe sweep; these are not generated-state trajectories or learned lower-SNR extrapolations.")
    if analytical.empty:
        save("lemma2_unconditional_baseline", pd.DataFrame(), reason="Missing fixed 97-point analytical reference-only sweep; run evidence analysis")
    else:
        _require(analytical, ["run_id", "seed", "grid_index", "snr", "reference_mean_error_rmse", "reference_scale_rmse"], "reference_analytical")
        _source(analytical, "analytical_reference_only", "reference_analytical")
        _unique(analytical, ["run_id", "seed", "grid_index"], "reference_analytical")
        expected_seeds = {str(seed) for seed in range(int(config["num_seeds"]))}
        if set(pd.to_numeric(analytical.grid_index, errors="raise")) != set(range(97)):
            raise TheoryError("Analytical reference sweep must retain all 97 fixed grid points")
        for _, bank_rows in analytical.groupby(["run_id", "grid_index"], sort=False):
            if set(bank_rows.seed.astype(str)) != expected_seeds:
                raise TheoryError("Analytical reference sweep changed the fixed unique Gaussian seed bank")
        analytical["step_index"] = analytical.grid_index
        reference_curves = _curve_summary(analytical, {"reference": "reference_mean_error_rmse"}, source_range="analytical", common=False)
        reference_curves["segment_id"] = 0
        initial_markers = native_curves.loc[_number(native_curves, "step_index").eq(0)].copy()
        initial_markers["source_range"] = "native_initial"
        if len(initial_markers) != 3 or not initial_markers.snr.eq(initial_snr).all():
            raise TheoryError("Initial reference comparison requires exactly three native summaries at the actual checkpoint initial SNR")
        grid = sorted(float(v) for v in analytical.snr.unique())
        expected_grid = np.geomspace(initial_snr * 10.0 ** -float(config.get("reference_snr_decades", 6.)), initial_snr, 97)
        if not np.array_equal(np.asarray(grid), expected_grid):
            raise TheoryError("Analytical reference SNR rows differ from the fixed initial-noise grid")
        scales = _number(analytical, "reference_scale_rmse").dropna().unique()
        if len(scales) != 1:
            raise TheoryError("Analytical reference scale must identify one declared finite law")
        save("lemma2_unconditional_baseline", pd.concat([reference_curves, initial_markers], ignore_index=True),
             actual_initial_snr=initial_snr, reference_scale_rmse=float(scales[0]), analytical_snr_grid=grid,
             reference_scale_definition="s_K=sqrt(sum_j pi_j ||u_j-mu_K||^2/d), retained as a caption quantity rather than a primary horizontal guide",
             quantile_policy="Inverse weighted ECDF over equal unique Gaussian seed weights",
             analytical_range_label="analytical reference only", band_definition="Descriptive IQR over unique Gaussian seeds",
             native_sweep_figure="appendix/lemma2_native_gaussian_sweep.png",
             display_window_definition="Initial-noise question: fixed analytical SNR_T*1e-6 through SNR_T; not selected from observed favorable behavior or claimed preregistered",
             input_law_separation="Only the three initial markers are native checkpoint measurements. The lower-SNR curve is analytical reference only, not a convergent learned tail. The mandatory native-sweep appendix preserves every later native label and increasing error.")
    atoms = tables.get("reference_atoms", pd.DataFrame()).copy()
    if atoms.empty:
        save("lemma2_initial_concentration", pd.DataFrame(), reason="Missing distinct-atom distances about the exact declared-law mean; no model-center substitution")
    else:
        _require(atoms, ["atom_id", "weight", "atom_center_distance_rmse"], "reference_atoms")
        _unique(atoms, ["atom_id"], "reference_atoms")
        initial_reference = native.loc[_number(native, "step_index").eq(0)]
        learned_cdf = _ecdf(_number(initial_reference, "learned_mean_error_rmse"), np.ones(len(initial_reference)), distribution="initial_unconditional")
        atom_cdf = _ecdf(_number(atoms, "atom_center_distance_rmse"), _number(atoms, "weight"), distribution="candidate_atoms")
        save("lemma2_initial_concentration", pd.concat([learned_cdf, atom_cdf], ignore_index=True), actual_initial_snr=initial_snr,
             reference_scale_rmse=metadata["lemma2_unconditional_baseline"].get("reference_scale_rmse"),
             reference_scale_definition="s_K=sqrt(sum_j pi_j ||u_j-mu_K||^2/d) for the same declared empirical law",
             counts={"unique_gaussian_seeds": len(initial_reference), "distinct_atoms": len(atoms)}, weighting="Uniform unique Gaussian seeds and declared finite-law atom masses; same mu_K.")

    initial = tables["initial"].copy()
    geometry_fields = ["evidence_injection_q", "evidence_injection_r_perp", "evidence_injection_relative_error", "evidence_injection_applicable", "evidence_injection_status", "evidence_injection_decomposition_qa"]
    if not set(geometry_fields).issubset(initial):
        save("corollary3_initial_cfg_amplification", pd.DataFrame(), reason="Missing vector-computed normalized injection geometry; scalar distance subtraction is not a valid migration")
    else:
        applicable = _boolean(initial, "evidence_injection_applicable")
        geometry_failed = applicable & initial.evidence_injection_decomposition_qa.ne("consistent_with_float64_estimate")
        premise_info, premise_failed = _injection_audit(initial, applicable, auxiliary)
        geometry_failed |= premise_failed
        geometry = _pair_frame(initial, "evidence_injection_q", "evidence_injection_r_perp")
        geometry["applicable"] = applicable
        geometry.loc[~applicable, ["x", "y"]] = np.nan
        save("corollary3_initial_cfg_amplification", geometry, actual_initial_snr=initial_snr,
             status="blocked" if geometry_failed.any() else "not_applicable" if not applicable.any() else None, reason="Applicable full-vector injection decomposition exceeded its saved numerical consistency budget" if geometry_failed.any() else "All normalized injection directions are numerically degenerate or guidance is outside the supported domain; absolute discrepancies remain saved" if not applicable.any() else None,
             decomposition_qa_counts={str(key): int(value) for key, value in initial.evidence_injection_decomposition_qa.value_counts(dropna=False).items()},
             excluded_direction_count=int((~applicable).sum()), exclusion_statuses=_statuses(initial.loc[~applicable]),
             relative_error_statistics=_quantities(_number(initial.loc[applicable], "evidence_injection_relative_error"), ""),
             branch_error_statistics={name: _quantities(_number(initial, field), "") for name, field in {"conditional": "direct_conditional_error_rmse", "unconditional": "direct_unconditional_target_error_rmse"}.items()},
             **premise_info, geometry_levels=list(GEOMETRY_LEVELS), interpretation="Distance from (1,0) is full relative injection error; visual contours are not memorization thresholds.")
        auxiliary["injection_excluded_directions"] = initial.loc[~applicable].copy()
        auxiliary["injection_failed_decomposition"] = initial.loc[geometry_failed].copy()

    matched_audit = tables["matched_updates"].copy()
    _require(matched_audit, ["direct_lemma4_applicable", "direct_lemma4_independent", "direct_lemma4_numeric_within_sensitivity"], "matched update consistency audit")
    failed_independent = _boolean(matched_audit, "direct_lemma4_applicable") & _boolean(matched_audit, "direct_lemma4_independent") & ~_boolean(matched_audit, "direct_lemma4_numeric_within_sensitivity")
    if failed_independent.any():
        metadata["lemma4_matched_displacement"].update(status="blocked", reason="Independent applicable matched-update reconstruction failed its recorded source-precision consistency check; sensitivity is numerical QA, not a formal arithmetic certificate")
        auxiliary["lemma4_failed_independent_updates"] = matched_audit.loc[failed_independent].copy()
    metadata["lemma4_matched_displacement"]["failed_independent_consistency_count"] = int(failed_independent.sum())
    residual = _number(matched_audit, "direct_lemma4_vector_residual_rmse")
    denominator = _number(matched_audit, "direct_lemma4_predicted_rmse") + _number(matched_audit, "direct_lemma4_displacement_rmse")
    relative = residual / denominator.where(denominator > LEMMA4_RELATIVE_DENOMINATOR_FLOOR_RMSE)
    residual_stats = _quantities(residual, "")
    finite_residuals = residual[np.isfinite(residual)]
    residual_stats["q99"] = float(np.quantile(finite_residuals, .99)) if len(finite_residuals) else None
    metadata["lemma4_matched_displacement"].update(vector_residual_statistics=residual_stats,
        relative_residual_statistics=_quantities(relative, ""), relative_residual_denominator_floor_rmse=LEMMA4_RELATIVE_DENOMINATOR_FLOOR_RMSE,
        relative_residual_excluded_count=int((~np.isfinite(relative)).sum()),
        source_precision_statistics=_quantities(_number(matched_audit, "direct_lemma4_source_tolerance_rmse"), ""),
        verification_source_counts={str(key): int(value) for key, value in matched_audit.direct_lemma4_verification_source.value_counts(dropna=False).items()})
    auxiliary["lemma4_vector_qa"] = matched_audit[[*SAMPLE_KEYS, "step_index", "direct_lemma4_vector_residual_rmse", "direct_lemma4_source_tolerance_rmse", "direct_lemma4_verification_source"]].copy()
    auxiliary["lemma4_vector_qa"]["relative_residual"] = relative
    auxiliary["lemma4_vector_qa"]["relative_denominator_rmse"] = denominator
    _feedback_inputs(tables["matched_updates"], save, auxiliary)
    _synchronization_inputs(tables["trajectory"], save, auxiliary)
    _terminal_inputs(tables["terminal"], config, save, auxiliary)
    for entry in paper_registry(diagnostics=True):
        if entry["stem"] not in frames:
            save(entry["stem"], pd.DataFrame(), reason="No compatible saved input for this optional audit; no proxy substitution")
    # Logical source tables retain full numerical histories. Scatter exports
    # need only plotted scalars, exact identities and their classification.
    compact_columns = set(SAMPLE_KEYS + ["step_index", "timestep", "snr", "x", "y", "marker_class", "applicable", "terminal_sscd", "input_source", "reference_law_hash", "latent_dimension", "direct_lemma4_verification_source", "direct_lemma4_vector_residual_rmse", "direct_lemma4_source_tolerance_rmse", "direct_prop5_condition_status", "direct_prop5_gain_status", "direct_prop5_integral_status", "direct_prop5_uncertainty_l2", "direct_prop5_gain_underflow", "frequency_low", "frequency_high", "fixed_cache_gain_sign", "source_robust_gain_sign", "condition_sign_status", "condition_value_status", "input_contract_status", "endpoint_construction_method", "arithmetic_status", "condition_arithmetic_status", "implication_eligible", "implication_audit_status"])
    for entry in paper_registry(diagnostics=True):
        if entry["kind"] in {"scatter", "injection_geometry"}:
            stem = entry["stem"]
            keep = compact_columns | set(entry["required_columns"])
            frames[stem] = frames[stem].loc[:, [name for name in frames[stem] if name in keep]].copy()
    audit["blocking"] = bool(audit.get("blocking", False) or any(info["status"] == "blocked" for info in metadata.values()))
    audit["figure_blockers"] = sorted(stem for stem, info in metadata.items() if info["status"] == "blocked")
    return frames, {"figures": metadata, "audit": audit, "auxiliary_tables": auxiliary}


def _feedback_inputs(matched, save, auxiliary):
    _require(matched, SAMPLE_KEYS + ["step_index", "snr", "terminal_sscd"], "matched_updates")
    required = {"refinement_applicable", "fixed_cache_gain_sign", "source_robust_gain_sign",
                "condition_sign_status", "condition_value_status", "input_contract_status",
                "endpoint_construction_method", "arithmetic_status", "arithmetic_error_method",
                "quadrature_status", "quadrature_error_scope", "source_sensitivity_status",
                "source_sensitivity_model", "numerical_publication_blocker", "implication_eligible",
                "implication_audit_status", "numerical_original_margin_rmse",
                "numerical_log_probability_gain", "numerical_log_odds_gain",
                "direct_prop5_measurement_contract", "direct_prop5_variation_definition"}
    stems = ("proposition5_posterior_feedback", "proposition5_condition_vs_gain",
             "proposition5_first_update_comparison", "proposition5_numerical_resolution")
    missing = sorted(required - set(matched))
    if missing:
        for stem in stems:
            save(stem, pd.DataFrame(), reason="Fixed-cache numerical assessments are unavailable; run theory analysis to save the gain and condition estimates. Interval certification is optional. Legacy source-sensitive classifications remain audits and are not substituted.", missing_refinement_fields=missing)
        auxiliary["proposition5_numerical_resolution_rows"] = matched.copy()
        return
    structural = _boolean(matched, "refinement_applicable")
    matching_contract = matched.direct_prop5_measurement_contract.eq(EVIDENCE_FORMULA_VERSION).fillna(False)
    matching_definition = matched.direct_prop5_variation_definition.eq(DIRECTIONAL_VARIATION_DEFINITION).fillna(False)
    incompatible = structural & ~(matching_contract & matching_definition)
    if incompatible.any():
        for stem in stems:
            save(stem, pd.DataFrame(), reason="Signed projected-gap-error or directional positive-variation receipts are incompatible; run --recompute-experiments. Historical norm-error or norm-integral values cannot be relabeled.",
                 measurement_contract=EVIDENCE_FORMULA_VERSION, incompatible_variation_contract_count=int(incompatible.sum()))
        auxiliary["proposition5_numerical_resolution_rows"] = matched.copy()
        return
    eligible = matched.loc[structural].copy()
    records, cause_records = [], []
    layers = {"feedback": "fixed_cache_gain_sign", "condition": "condition_sign_status", "source_robust_feedback": "source_robust_gain_sign"}
    sign_values = {"positive", "negative", "zero", "unresolved", "unavailable", "not_applicable"}
    for field in layers.values():
        if set(eligible[field].dropna().astype(str)) - sign_values:
            raise TheoryError("Unrecognized numerical refinement sign vocabulary in " + field)
    # The numerical audit retains every structural and excluded row. Missing V
    # cannot remove a row whose original-condition sign has a negative precheck.
    status_fields = sorted({name for name in matched if name.endswith("_status") or name in {
        *layers.values(), "input_contract_status", "endpoint_construction_method", "arithmetic_error_method",
        "quadrature_error_scope", "source_sensitivity_model", "reference_law_id", "numerical_stopping_reason",
        "numerical_certification_scope", "numerical_gain_before", "numerical_condition_before"}})
    for group, selected in [("All structural rows", eligible), ("Structurally excluded", matched.loc[~structural]), *_outcomes(eligible)]:
        for step, rows in selected.groupby("step_index", sort=True):
            weights = _weights(rows)
            denominator = float(weights.sum())
            for field in status_fields:
                for value, indexes in rows[field].fillna("missing").astype(str).groupby(rows[field].fillna("missing").astype(str)).groups.items():
                    mass = float(weights.loc[indexes].sum())
                    cause_records.append({"group": group, "step_index": step, "snr": float(rows.snr.iloc[0]),
                        "field": field, "status": value, "count": len(indexes), "weight": mass,
                        "denominator_count": len(rows), "denominator_weight": denominator, "fraction": mass / denominator})
            if group in {"All structural rows", "Structurally excluded"}:
                continue
            for metric, field in layers.items():
                status = rows[field].fillna("unavailable")
                positive, negative, zero = (status.eq(value) for value in ("positive", "negative", "zero"))
                unknown = ~(positive | negative | zero)
                positive_weight, unknown_weight = float(weights[positive].sum()), float(weights[unknown].sum())
                records.append({"group": group, "step_index": step, "snr": float(rows.snr.iloc[0]), "metric": metric,
                    "fraction": positive_weight / denominator, "upper_fraction": (positive_weight + unknown_weight) / denominator,
                    "resolved_fraction": float(weights[~unknown].sum()) / denominator, "unknown_fraction": unknown_weight / denominator,
                    "denominator_weight": denominator, "eligible_count": len(rows), "prompt_count": len(rows[PAIR_KEYS].drop_duplicates()),
                    "positive_count": int(positive.sum()), "negative_count": int(negative.sum()), "zero_count": int(zero.sum()),
                    "unresolved_count": int(unknown.sum()), "unavailable_count": int(status.eq("unavailable").sum()),
                    "positive_weight": positive_weight, "unresolved_weight": unknown_weight,
                    "nonnegative_resolved_count": int((positive | zero).sum()),
                    "condition_without_point_value_count": int((rows.condition_sign_status.eq("negative") & ~np.isfinite(_number(rows, "numerical_original_margin_rmse"))).sum()),
                    "gain_underflow_count": int(_boolean(rows, "numerical_gain_underflow").sum())})
    resolution = pd.DataFrame(records)
    curves = resolution.loc[resolution.metric.isin(["feedback", "condition"])].copy() if not resolution.empty else pd.DataFrame()
    failures = _boolean(eligible, "numerical_publication_blocker")
    for field in ("implication_audit_status", "numerical_gain_sign_audit_status", "numerical_concavity_audit_status", "numerical_integral_audit_status", "numerical_margin_ordering_audit_status"):
        if field in eligible:
            failures |= eligible[field].fillna("").astype(str).str.startswith("failed")
    contradiction = _boolean(eligible, "implication_eligible") & eligible.condition_sign_status.eq("positive") & eligible.fixed_cache_gain_sign.isin(["negative", "zero"])
    failures |= contradiction
    # A resolved numerical estimate is displayable without being an interval
    # certificate. Keep its scope separate from both sign and implication gates.
    condition_resolved = eligible.condition_sign_status.isin(["positive", "negative", "zero"])
    condition_arithmetic = eligible.get("condition_arithmetic_status", pd.Series("unavailable", index=eligible.index)).fillna("unavailable").astype(str)
    estimated = condition_resolved & condition_arithmetic.eq("float64_numerical_assessment")
    enclosed = condition_resolved & condition_arithmetic.str.startswith("certified_")
    assessment_counts = {
        "float64_estimated_sign": int(estimated.sum()),
        "enclosed_sign": int(enclosed.sum()),
        "other_resolved_sign": int((condition_resolved & ~(estimated | enclosed)).sum()),
        "unresolved_or_unavailable": int((~condition_resolved).sum()),
    }
    common = {
        "same_population": "At each step all three numerical layers share the same structurally eligible rows and prompt-balanced weights; unavailable V and unresolved signs remain in the denominator",
        "denominator_column": "denominator_weight", "excluded_structural_count": int((~structural).sum()),
        "excluded_missing_sscd_count": int((~np.isfinite(_number(eligible, "terminal_sscd"))).sum()),
        "reliable_positive_margin_negative_gain_count": int(contradiction.sum()),
        "measurement_contract": "projected-gap-error-1",
        "condition_formula": "||Delta_t||-e_t^parallel(Delta)-mathcal{V}_t; e_t^parallel(Delta)=u_t^T(Delta_t-bar_Delta_t), u_t=Delta_t/||Delta_t||; signed with exact-zero extension u=e^parallel=0",
        "variation_definition": "mathcal{V}_t=[integral_0^1 <Delta_t/||Delta_t||,bar{x}_{t-1}(z_t(s),empty)-bar{x}_t(x_t,empty)> ds]_+; positive part after signed integration",
        "zero_gap_convention": "mathcal{V}_t=0 when Delta_t is exactly zero; no unit direction is formed",
        "comparison_interpretation": "Tightened gap-error sufficient condition from the same vector decomposition; original input/reference assumptions and justified same-endpoint transfer remain required",
        "band_definition": "Saved-assessment range [positive mass, positive plus unknown mass]; point-estimate signs are numerical assessments, not interval certificates, and this is not a statistical confidence interval",
        "condition_assessment_counts": assessment_counts,
        "numerical_layers": {"feedback": "Stable fixed-cache saved-endpoint H/G sign; arithmetic assessments and scoped enclosures are recorded separately",
            "condition": "Affine-path margin ||Delta||-E_parallel-mathcal{V}, with the positive part taken after signed unit-gap projection integration and signed vector-projected E_parallel. Default signs are numerical estimates assessed against saved quadrature and float64 roundoff uncertainty; optional interval certification is recorded separately. Saved-endpoint implication requires its separately justified transfer",
            "source_robust_feedback": "Separate justified perturbation model; heuristic source-dtype sensitivity does not certify robustness",
            "reference_model_identification": "D_K is the declared empirical law, not the identified complete checkpoint training law"},
        "numerical_resolution_table": "audit_data/proposition5_numerical_resolution_rows.csv",
        "numerical_cause_counts_table": "audit_data/proposition5_numerical_cause_counts.csv",
        "numerical_coverage_table": "audit_data/proposition5_strict_coverage.csv",
        "numerical_resolution_figure": "appendix/proposition5_numerical_resolution.png",
        "status_counts": _statuses(eligible),
        "certification_status": "Numerical estimates do not require interval certification to be displayed. Per-row arithmetic_status, condition_arithmetic_status and numerical_certification_scope distinguish float64 assessments from optional scoped certificates; no blanket certificate or implication eligibility is inferred"}
    if not curves.empty:
        condition = curves.loc[curves.metric.eq("condition")]
        common["condition_positive_counts"] = {group: int(rows.positive_count.sum()) for group, rows in condition.groupby("group")}
        common["condition_unresolved_counts"] = {group: int(rows.unresolved_count.sum()) for group, rows in condition.groupby("group")}
        common["condition_zero_overlap"] = len(condition.group.unique()) == 2 and condition.fraction.eq(0).all()
        common["condition_unresolved_fractions"] = {group: {"minimum": float(rows.unknown_fraction.min()), "maximum": float(rows.unknown_fraction.max())} for group, rows in condition.groupby("group")}
        common["observed_condition_coverage"] = {group: {"minimum": float(rows.fraction.min()), "maximum": float(rows.fraction.max()), "upper_maximum": float(rows.upper_fraction.max())} for group, rows in condition.groupby("group")}
        common["feedback_unresolved_fractions"] = {group: {"minimum": float(rows.unknown_fraction.min()), "maximum": float(rows.unknown_fraction.max())} for group, rows in curves.loc[curves.metric.eq("feedback")].groupby("group")}
        common["condition_interpretation"] = "Positive fractions include clear numerical estimates and any optional certified signs; unknown mass is retained separately. Absence of interval certification alone does not make a sign unresolved. This display does not establish a certified condition-coverage or implication claim, whose original endpoint and arithmetic gates remain separate"
    status = "blocked" if failures.any() else None
    reason = "A justified same-contract implication, arithmetic identity, or enclosure audit failed; retain offender identities and repair the numerical calculation" if failures.any() else None
    save("proposition5_posterior_feedback", curves, status=status, reason=reason, **common)
    save("proposition5_numerical_resolution", resolution, status=status, reason=reason, **common)
    if failures.any():
        auxiliary["proposition5_failed_observations"] = eligible.loc[failures].copy()
    direct = _pair_frame(eligible, "numerical_original_margin_rmse", "numerical_log_probability_gain")
    direct["marker_class"] = np.where(eligible.condition_sign_status.isin(["positive", "negative", "zero"]) & eligible.fixed_cache_gain_sign.isin(["positive", "negative", "zero"]), "observed", "numerically_unresolved")
    save("proposition5_condition_vs_gain", direct, status=status, reason=reason, symlog_linthresh=.001,
         endpoint_contract="x is affine-path margin ||Delta||-E_parallel-mathcal{V}; y is saved-endpoint H. Ordinary points have resolved saved numerical signs, including float64 estimates, without claiming interval certification. Crosses retain finite coordinates whose sign assessment remains unresolved. An implication requires saved implication_eligible and justified endpoint transfer. Both affine and saved gains remain in the audit.", **common)
    first = eligible.loc[_number(eligible, "step_index").eq(0)]
    if not {"numerical_matched_log_odds", "numerical_saved_log_odds"}.issubset(first):
        save("proposition5_first_update_comparison", pd.DataFrame(), reason="Stable first-update endpoint log odds are unavailable; do not infer them from clipped probabilities")
    else:
        save("proposition5_first_update_comparison", _pair_frame(first, "numerical_matched_log_odds", "numerical_saved_log_odds"), status=status, reason=reason,
             input_source="matched_counterfactual_and_saved_guided_same_destination", **common)
    auxiliary["proposition5_strict_coverage"] = curves.copy()
    auxiliary["proposition5_resolution_rates"] = resolution.copy()
    auxiliary["proposition5_numerical_resolution_rows"] = matched.copy()
    auxiliary["proposition5_numerical_cause_counts"] = pd.DataFrame(cause_records)


def _synchronization_inputs(trajectory, save, auxiliary):
    _require(trajectory, SAMPLE_KEYS + ["step_index", "snr", "terminal_sscd", "direct_conditional_error_rmse", "direct_unconditional_target_error_rmse", "direct_lemma6_gap_rmse", "direct_lemma6_rhs_rmse", "direct_branch_gap_error_rmse", "direct_radius_tail_rmse", "direct_unconditional_reference_error_rmse"], "trajectory")
    _require_projected_gap_error(trajectory, "trajectory synchronization")
    rows = trajectory.copy()
    rows["joint_error"] = np.maximum(_number(rows, "direct_conditional_error_rmse"), _number(rows, "direct_unconditional_target_error_rmse"))
    gap, bound = _number(rows, "direct_lemma6_gap_rmse"), _number(rows, "direct_lemma6_rhs_rmse")
    common = np.isfinite(rows.joint_error) & np.isfinite(gap) & np.isfinite(bound)
    rows["gap_bound_slack_rmse"] = bound - gap
    rows["joint_bound_slack_rmse"] = bound - rows.joint_error
    rows["row_bound_tolerance_rmse"] = _operation_tolerance(rows.joint_error, gap, bound)
    gap_failed = common & (rows.gap_bound_slack_rmse < -rows.row_bound_tolerance_rmse)
    joint_failed = common & (rows.joint_bound_slack_rmse < -rows.row_bound_tolerance_rmse)
    revised_rhs = _number(rows, "direct_branch_gap_error_rmse") + _number(rows, "direct_radius_tail_rmse")
    rows["branch_gap_error_rhs_identity_residual_rmse"] = bound - revised_rhs
    # Error cancellation can make E_parallel small while both target errors are
    # large. The tightened D bound is not a bound on their paired maximum Q.
    rows["joint_target_bound_rmse"] = np.maximum(_number(rows, "direct_conditional_error_rmse"),
        _number(rows, "direct_unconditional_reference_error_rmse") + _number(rows, "direct_radius_tail_rmse"))
    rows["joint_target_bound_slack_rmse"] = rows.joint_target_bound_rmse - rows.joint_error
    joint_target_failed = common & (~np.isfinite(rows.joint_target_bound_rmse) |
        (rows.joint_target_bound_slack_rmse < -_operation_tolerance(rows.joint_target_bound_rmse, rows.joint_error)))
    failed = gap_failed | joint_target_failed | (common & (~np.isfinite(revised_rhs) |
        (abs(bound - revised_rhs) > _operation_tolerance(bound, revised_rhs))))
    rows["row_bound_audit_status"] = np.select([~common, failed],
        ["unavailable_nonfinite_bound_or_error", "failed_fixed_float64_consistency"], default="consistent_with_float64_estimate")
    rows["arithmetic_error_scope"] = SCALAR_QA_SCOPE
    primary, branches, coverage = [], [], []
    fields = {"joint_error": "joint_error", "gap": "direct_lemma6_gap_rmse", "bound": "direct_lemma6_rhs_rmse"}
    # A common finite mask precedes every quantity's weights and coverage.
    for group, chosen in _outcomes(rows):
        if chosen.empty:
            continue
        primary.append(_segments(_curve_summary(chosen, fields, group=group)))
        all_fields = {**fields, "conditional": "direct_conditional_error_rmse", "unconditional": "direct_unconditional_target_error_rmse"}
        branch = _curve_summary(chosen, all_fields, group=group)
        branch = branch.loc[branch.metric.isin(["conditional", "unconditional"])].copy()
        branch["branch"] = branch.metric
        branches.append(_segments(branch))
        for step, population in chosen.groupby("step_index", sort=True):
            selected = population.loc[common.loc[population.index]]
            weights = _weights(selected)
            denominator = float(weights.sum())
            for metric, field in (("joint_error", "joint_error"), ("bound", "direct_lemma6_rhs_rmse")):
                for tolerance in RECOVERY_TOLERANCES:
                    within = _number(selected, field) <= tolerance
                    coverage.append({"group": group, "step_index": step, "snr": float(population.snr.iloc[0]),
                        "metric": metric, "tolerance_rmse": tolerance, "fraction": float(weights[within].sum()) / denominator if denominator else np.nan,
                        "covered_count": int(within.sum()), "eligible_count": len(selected), "excluded_count": len(population) - len(selected),
                        "denominator_weight": denominator})
    info = {"denominator_column": "denominator_weight",
        "same_population": "All Q/D/S quantities share the same finite paired rows and prompt weights at each step",
        "quantile_policy": "Inverse weighted ECDF on common prompt-balanced weights",
        "row_bound_audit_table": "audit_data/lemma6_joint_sample_audit.csv",
        "absolute_tolerance_coverage_table": "audit_data/lemma6_absolute_tolerance_coverage.csv",
        "absolute_tolerance_grid_rmse": list(RECOVERY_TOLERANCES),
        "absolute_tolerance_definition": "Fixed latent L2 target-error tolerance divided by sqrt(d); unchanged across SSCD groups, steps and configurations; not fitted from outcomes",
        "row_bound_audit": {"gap_above_bound_count": int(gap_failed.sum()), "joint_above_bound_count": int(joint_failed.sum()), "joint_above_own_bound_count": int(joint_target_failed.sum()),
            "eligible_count": int(common.sum()), "unavailable_count": int((~common).sum()), "arithmetic_error_scope": SCALAR_QA_SCOPE},
        "measurement_contract": "projected-gap-error-1",
        "branch_gap_error_definition": "signed_projected_branch_gap_reference_error",
        "derived_quantity": "D<=S with S=(e_t^parallel(Delta)+R(1-p_t))/sqrt(d). Q remains the paired target-error maximum and is separately bounded by max(e_t(c), e_t(empty)+R(1-p_t))/sqrt(d); Q>S is not a failure"}
    status = "blocked" if failed.any() else None
    reason = "A saved scalar identity, tightened D bound, or separate branch target-error bound failed numerical consistency; offender rows are retained" if failed.any() else None
    frame = pd.concat(primary, ignore_index=True) if primary else pd.DataFrame()
    save("lemma6_target_specific_synchronization", frame, status=status, reason=reason,
         band_definition="Descriptive IQR of Q displayed; quantiles of D and S retained", **info)
    save("lemma6_branch_target_errors", pd.concat(branches, ignore_index=True) if branches else pd.DataFrame(), status=status, reason=reason,
         band_definition="Descriptive branch IQRs on the same Q/D/S population", **info)
    auxiliary["lemma6_joint_sample_audit"] = rows
    auxiliary["lemma6_absolute_tolerance_coverage"] = pd.DataFrame(coverage)
    if failed.any():
        auxiliary["lemma6_failed_observations"] = rows.loc[failed].copy()


def _terminal_looseness(terminal, config, auxiliary):
    fields = {"conditional": "direct_conditional_error_l2", "branch_gap_error": "direct_branch_gap_error_l2",
              "radius_tail": "direct_radius_tail_l2", "reference_target": "direct_reference_target_error_l2",
              "S_raw": "direct_lemma6_rhs_l2", "D_raw": "direct_lemma6_gap_l2",
              "B_obs": "evidence_terminal_B_obs_l2", "B_ref": "evidence_terminal_B_ref_l2"}
    _require(terminal, [*fields.values(), "latent_dimension"], "terminal exact looseness decomposition")
    _require_projected_gap_error(terminal, "terminal exact looseness decomposition")
    rows = terminal.copy()
    values = {name: _number(rows, field) for name, field in fields.items()}
    dimension = _number(rows, "latent_dimension")
    root_dimension = np.sqrt(dimension.where(dimension > 0))
    g = float(config["guidance_scale"])
    valid = _boolean(rows, "evidence_terminal_applicable") & np.isfinite(root_dimension)
    for value in values.values():
        valid &= np.isfinite(value)
    ec, gap_error, tail, center = (values[name] for name in ("conditional", "branch_gap_error", "radius_tail", "reference_target"))
    S, D, obs, ref = (values[name] for name in ("S_raw", "D_raw", "B_obs", "B_ref"))
    rows["bound_looseness_l2"] = ref - obs
    rows["slack_radius_l2"] = tail - center
    # E_parallel=D-u^T bar_Delta is signed; this is alignment slack, not a norm triangle.
    rows["slack_projection_alignment_l2"] = gap_error + center - D
    rows["lemma6_slack_cost_l2"] = (g - 1) * (S - D)
    rows["decomposed_slack_cost_l2"] = (g - 1) * (rows.slack_radius_l2 + rows.slack_projection_alignment_l2)
    rows["looseness_identity_residual_l2"] = rows.bound_looseness_l2 - rows.lemma6_slack_cost_l2
    rows["looseness_decomposition_residual_l2"] = rows.bound_looseness_l2 - rows.decomposed_slack_cost_l2
    rows["lemma6_rhs_identity_residual_l2"] = S - (gap_error + tail)
    rows["observable_formula_residual_l2"] = obs - (ec + (g - 1) * D)
    rows["reference_formula_residual_l2"] = ref - (ec + (g - 1) * (gap_error + tail))
    rows["looseness_tolerance_l2"] = _operation_tolerance(ref, obs, (g - 1) * S, (g - 1) * D,
        ec, (g - 1) * gap_error, (g - 1) * tail, (g - 1) * center)
    rows["decomposition_tolerance_l2"] = _operation_tolerance(ec, gap_error, tail, center, S, D)
    failed = valid & ((abs(rows.looseness_identity_residual_l2) > rows.looseness_tolerance_l2)
        | (abs(rows.looseness_decomposition_residual_l2) > rows.looseness_tolerance_l2)
        | (abs(rows.lemma6_rhs_identity_residual_l2) > rows.decomposition_tolerance_l2)
        | (abs(rows.observable_formula_residual_l2) > rows.looseness_tolerance_l2)
        | (abs(rows.reference_formula_residual_l2) > rows.looseness_tolerance_l2)
        | (rows.slack_radius_l2 < -rows.decomposition_tolerance_l2)
        | (rows.slack_projection_alignment_l2 < -rows.decomposition_tolerance_l2)
        | (rows.bound_looseness_l2 < -rows.looseness_tolerance_l2))
    corrected_obs = _number(rows, "evidence_terminal_corrected_obs_bound_rmse")
    corrected_ref = _number(rows, "evidence_terminal_corrected_ref_bound_rmse")
    corrected_finite = valid & np.isfinite(corrected_obs) & np.isfinite(corrected_ref)
    rows["scheduler_cancellation_residual_rmse"] = corrected_ref.where(corrected_finite) - corrected_obs.where(corrected_finite) - rows.bound_looseness_l2 / root_dimension
    rows["scheduler_cancellation_tolerance_rmse"] = _operation_tolerance(corrected_ref, corrected_obs, rows.bound_looseness_l2 / root_dimension)
    failed |= corrected_finite & (abs(rows.scheduler_cancellation_residual_rmse) > rows.scheduler_cancellation_tolerance_rmse)
    # Preserve +inf correction masses; inf-inf is not fabricated as zero.
    rows["scheduler_cancellation_status"] = np.where(corrected_finite, "finite_scalar_comparison", "unavailable_nonfinite_corrected_bounds")
    rows["looseness_audit_status"] = np.select([~valid, failed], ["unavailable_finite_scalar_decomposition", "failed_fixed_float64_consistency"], default="consistent_with_float64_estimate")
    rows["arithmetic_error_scope"] = SCALAR_QA_SCOPE
    components = ("bound_looseness", "slack_radius", "slack_projection_alignment", "lemma6_slack_cost", "decomposed_slack_cost", "looseness_identity_residual", "looseness_decomposition_residual")
    for name in components:
        rows[name + "_rmse"] = rows[name + "_l2"] / root_dimension
    auxiliary["theorem7_looseness_decomposition"] = rows
    return {
        "looseness_audit_table": "audit_data/theorem7_looseness_decomposition.csv",
        "looseness_statistics": {name: _quantile_description(_number(rows.loc[valid], name + "_rmse")) for name in components},
        "looseness_identity": "B_ref-B_obs=(g-1)(S_raw-D_raw)=(g-1)(slack_radius+slack_projection_alignment); S_raw=E_parallel+R(1-p), slack_radius=R(1-p)-||bar_Delta||, slack_projection_alignment=E_parallel+||bar_Delta||-D_raw=||bar_Delta||-u^T bar_Delta >=0; E_parallel itself may be negative. The same scheduler correction cancels.",
        "looseness_audit": {"failed_count": int(failed.sum()), "eligible_count": int(valid.sum()),
            "unavailable_count": int((~valid).sum()), "arithmetic_error_scope": SCALAR_QA_SCOPE,
            "interpretation": "Exact algebraic attribution of the tightened branch-gap-error bound, with projection-alignment/radius and independent scheduler-correction QA"}}, failed


def _terminal_inputs(terminal, config, save, auxiliary):
    fields = {"actual": "direct_theorem7_endpoint_error_rmse",
              "observable": "evidence_terminal_corrected_obs_bound_rmse",
              "reference": "evidence_terminal_corrected_ref_bound_rmse"}
    components = {"conditional": "evidence_terminal_conditional_error_rmse",
                  "guidance": "evidence_terminal_residual_guidance_rmse",
                  "scheduler": "evidence_terminal_defect_bound_rmse"}
    required = {*fields.values(), *components.values(), "evidence_terminal_scope", "evidence_terminal_status", "evidence_terminal_applicable"}
    original = _boolean(terminal, "direct_theorem7_applicable")
    original_counts = {"applicable": int(original.sum()), "total": len(terminal)}
    if not required.issubset(terminal):
        info = {"terminal_scope": "unavailable_unsupported_terminal_contract", "original_clean_counts": original_counts}
        for stem in ("theorem7_final_reproduction", "theorem7_terminal_components"):
            save(stem, pd.DataFrame(), reason="Independent predictor-side terminal correction fields are unavailable; endpoint-minus-estimate is QA only", **info)
        return
    applicable = _boolean(terminal, "evidence_terminal_applicable")
    looseness_info, looseness_failed = _terminal_looseness(terminal, config, auxiliary)
    common = applicable.copy()
    for field in [*fields.values(), *components.values()]:
        values = _number(terminal, field)
        common &= ~np.isnan(values) & (values >= 0)
    common &= np.isfinite(_number(terminal, fields["actual"]))
    chosen = terminal.loc[common].copy()
    failures = terminal.evidence_terminal_status.fillna("").astype(str).str.contains("failed", case=False, regex=False)
    failures |= looseness_failed
    if "evidence_terminal_reconstruction_status" in terminal:
        failures |= terminal.evidence_terminal_reconstruction_status.fillna("").astype(str).str.contains("failed", case=False, regex=False)
    if "evidence_terminal_clean_bound_order_status" in terminal:
        failures |= terminal.evidence_terminal_clean_bound_order_status.fillna("").astype(str).str.contains("failed", case=False, regex=False)
    scope_counts = {str(key): int(value) for key, value in chosen.evidence_terminal_scope.value_counts().items()}
    if set(scope_counts) - set(TERMINAL_SCOPES):
        raise TheoryError("Eligible terminal correction has an unknown mathematical scope")
    scope = next((value for value in reversed(TERMINAL_SCOPES) if value in scope_counts), "unavailable_unsupported_terminal_contract")
    manuscript_extension_required = scope in TERMINAL_SCOPES[1:]
    info = {"terminal_scope": scope, "scope_counts": scope_counts, "original_clean_counts": original_counts,
            "measurement_contract": "projected-gap-error-1",
            "reference_comparison_formula": "e_c+(g-1)*(E_parallel+R(1-p)) plus the unchanged scheduler correction",
            "branch_gap_error_definition": "signed_projected_branch_gap_reference_error",
            "reference_comparison_scope": "Tightened reference-based terminal bound using the signed projected branch-gap error; the same scheduler applicability and noise qualifications remain in force",
            "manuscript_extension_required": manuscript_extension_required,
            "counts": {"total": len(terminal), "eligible": len(chosen), "excluded": int((~common).sum())},
            "excluded_status_counts": _statuses(terminal.loc[~common]),
            "weighting": "One mass unit per represented prompt divided over the identical eligible member seeds for all curves; no SSCD split",
            "certainty": "Exact-arithmetic comparison plus saved numerical QA; Gaussian noise bounds retain their declared simultaneous per-run probability scope"}
    noise_columns = [name for name in ("run_id", "evidence_terminal_noise_run_alpha", "evidence_terminal_noise_update_count", "evidence_terminal_noise_run_scope", "evidence_terminal_noise_per_update_alpha", "evidence_terminal_noise_chi2_norm_quantile", "evidence_terminal_noise_probability_scope") if name in terminal]
    info.update(looseness_info)
    info["noise_probability_scope"] = terminal[noise_columns].drop_duplicates().to_dict("records") if noise_columns else []
    info["noise_bound_exceedance_count"] = int(_boolean(terminal, "evidence_terminal_noise_bound_exceeded").sum())
    if failures.any():
        auxiliary["terminal_failed_observations"] = terminal.loc[failures].copy()
    if chosen.empty:
        for stem in ("theorem7_final_reproduction", "theorem7_terminal_components"):
            save(stem, pd.DataFrame(), status="blocked" if failures.any() else "unavailable",
                 reason="Failed independent terminal reconstruction" if failures.any() else "No supported independent terminal-correction population", **info)
        return
    weights = _weights(chosen)
    denominator = float(weights.sum())
    curves = [_ecdf(_number(chosen, field), weights, distribution=name) for name, field in fields.items()]
    terms = [_ecdf(_number(chosen, field), weights, component=name) for name, field in components.items()]
    info["denominator_weight"] = denominator
    info["zero_mass"] = {name: float(weights[_number(chosen, field).eq(0)].sum()) / denominator for name, field in fields.items()}
    info["infinite_mass"] = {name: float(weights[np.isposinf(_number(chosen, field))].sum()) / denominator for name, field in fields.items()}
    actual, observable, reference = (_number(chosen, fields[name]) for name in ("actual", "observable", "reference"))
    ordering = chosen[[*SAMPLE_KEYS, "step_index", "evidence_terminal_scope"]].copy()
    ordering["actual_error_rmse"], ordering["observable_bound_rmse"], ordering["reference_bound_rmse"] = actual, observable, reference
    ordering["arithmetic_tolerance_rmse"] = _operation_tolerance(actual, observable)
    source_tolerance = pd.concat([_number(chosen, name).fillna(0).clip(lower=0) for name in
        ("evidence_terminal_reconstruction_tolerance_rmse", "direct_theorem7_numeric_tolerance_rmse")], axis=1).max(axis=1)
    ordering["inherited_source_sensitivity_rmse"] = source_tolerance
    ordering["actual_above_observable_beyond_sensitivity"] = np.isfinite(observable) & (actual - observable > ordering.arithmetic_tolerance_rmse + source_tolerance)
    ordering["observable_above_reference_beyond_roundoff"] = np.isfinite(observable) & np.isfinite(reference) & (observable - reference > _operation_tolerance(observable, reference))
    stochastic = chosen.evidence_terminal_scope.eq("finite_terminal_update_extension_gaussian_noise_bound")
    ordering["permitted_noise_event"] = stochastic & ordering.actual_above_observable_beyond_sensitivity
    ordering["actual_above_reference_beyond_sensitivity"] = np.isfinite(reference) & (actual - reference > _operation_tolerance(actual, reference) + source_tolerance)
    ordering["blocking_consistency_failure"] = ordering.observable_above_reference_beyond_roundoff | (~stochastic & ordering.actual_above_observable_beyond_sensitivity)
    failures.loc[chosen.index] |= ordering.blocking_consistency_failure
    auxiliary["theorem7_paired_bound_ordering"] = ordering
    if failures.any():
        auxiliary["terminal_failed_observations"] = terminal.loc[failures].copy()
    info["paired_ordering_audit_table"] = "audit_data/theorem7_paired_bound_ordering.csv"
    info["paired_ordering_audit"] = {"blocking_failure_count": int(ordering.blocking_consistency_failure.sum()),
        "permitted_gaussian_noise_event_count": int(ordering.permitted_noise_event.sum()),
        "arithmetic_error_scope": SCALAR_QA_SCOPE,
        "source_sensitivity_scope": "Inherited independently saved reconstruction/clean-update tolerance is a source-sensitivity screen, not an arithmetic certificate. Gaussian confidence-bound failures remain possible probability events."}
    info["raw_ordering_violations"] = {"actual_above_observable": int((actual > observable).sum()), "observable_above_reference": int((observable > reference).sum()), "actual_above_reference": int((actual > reference).sum())}
    info["ordering_interpretation"] = "Raw tightened-bound comparisons are retained without clipping; deterministic source-precision QA and permitted Gaussian noise-bound failures remain distinct"
    if config.get("target_error_tolerance") is not None:
        dimensions = _number(chosen, "latent_dimension").dropna().unique()
        if len(dimensions) != 1 or dimensions[0] <= 0:
            raise TheoryError("Terminal tolerance normalization requires one valid latent dimension")
        tolerance = float(config["target_error_tolerance"]) / math.sqrt(float(dimensions[0]))
        info["predeclared_tolerance_rmse"] = tolerance
        info["predeclared_coverage"] = {name: float(weights[_number(chosen, field) <= tolerance].sum()) / denominator for name, field in fields.items()}
    status = "blocked" if failures.any() else "available"
    reason = "Independent terminal reconstruction, paired bound ordering, or exact looseness attribution failed its recorded numerical consistency screen; do not fit corrections or tolerances to endpoints" if failures.any() else None
    save("theorem7_final_reproduction", pd.concat(curves, ignore_index=True), status=status, reason=reason, **info)
    component_info = {**info, "zero_mass": {name: float(weights[_number(chosen, field).eq(0)].sum()) / denominator for name, field in components.items()}}
    component_info["infinite_mass"] = {name: float(weights[np.isposinf(_number(chosen, field))].sum()) / denominator for name, field in components.items()}
    save("theorem7_terminal_components", pd.concat(terms, ignore_index=True), status=status, reason=reason, **component_info)
    auxiliary["terminal_common_population"] = chosen.copy()
    auxiliary["terminal_excluded_population"] = terminal.loc[~common].copy()
