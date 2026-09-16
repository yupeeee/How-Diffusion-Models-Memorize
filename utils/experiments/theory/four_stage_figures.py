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

from .contracts import TheoryError
from .direct_figures import PAIR_KEYS, _boolean, _counts, _number, _outcomes, _pair_frame, _require, _statuses
from .evidence_figures import (
    BOOTSTRAP_POLICY, BOOTSTRAP_SEED, SAMPLE_KEYS, _curve_summary, _interval,
    _quantile_description, _segments, _weights, build_evidence_plot_inputs,
)
from .paper_registry import GROUPS, paper_registry
from .summaries import weighted_quantiles

FOUR_STAGE_FORMULA_VERSION = "four-stage-scalar-inputs-1"
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
            "counts": _population(frame), "status_counts": _statuses(frame), **info}
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

    _response_inputs(tables.get("feedback_response", pd.DataFrame()), config, save, auxiliary)
    reuse("posterior_feedback_over_time", "proposition5_posterior_feedback")
    reuse("posterior_feedback_condition_margin", "proposition5_condition_vs_gain")
    if "refinement_applicable" in tables["matched_updates"] and not _boolean(tables["matched_updates"], "refinement_applicable").any():
        for stem in ("posterior_feedback_over_time", "posterior_feedback_condition_margin"):
            metadata[stem].update(status="not_applicable", reason="Every transition is outside the recorded manuscript/reference domain")
    for stem in ("posterior_feedback_over_time", "posterior_feedback_condition_margin"):
        metadata[stem]["numerical_resolution_table"] = "audit_data/proposition5_numerical_resolution_rows.csv"
    auxiliary["feedback_endpoints"] = tables["matched_updates"].copy()

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
    auxiliary["trajectory_metrics"] = trajectory
    for name in ("trajectory_shape_prompts", "trajectory_shape_mixed_prompts", "initial_baseline_disagreements", "initial_generated_samples", "reference_law", "reference_target_radii"):
        if name in tables:
            auxiliary[name] = tables[name].copy()
    _shape_inputs(tables.get("trajectory_shapes", pd.DataFrame()), steps, save, auxiliary)
    _motion_inputs(tables.get("branch_motion", pd.DataFrame()), steps, save, auxiliary)
    _terminal_inputs(tables["terminal"], frames, metadata, config, save, auxiliary)
    reuse("terminal_bound_coverage", "theorem7_final_reproduction")
    _counterfactual_inputs(tables.get("counterfactual_unconditional", pd.DataFrame()), bool(config.get("counterfactual_unconditional", False)), save, auxiliary)
    for entry in paper_registry(True, counterfactual=bool(config.get("counterfactual_unconditional", False))):
        if entry["stem"] not in frames:
            save(entry["stem"], pd.DataFrame(), reason="Optional compatible measurement was not recorded; plot mode does not backfill scientific work")
    audit["blocking"] = bool(audit.get("blocking", False) or any(value["status"] == "blocked" for value in metadata.values()))
    audit["figure_blockers"] = sorted(stem for stem, info in metadata.items() if info["status"] == "blocked")
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
        baseline_summary=baseline.to_dict("records"), baseline_uncertainty=BOOTSTRAP_POLICY,
        baseline_interpretation="Unique Gaussian seeds, not prompt-replicated estimates; B_T^K about mu_K differs from the unconditional paired-target control. S_K=0 gives an undefined ratio.")


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


def _terminal_inputs(terminal, frames, metadata, config, save, auxiliary):
    info = copy.deepcopy(metadata["theorem7_final_reproduction"])
    info.pop("counts", None)
    for column, key in (("evidence_terminal_correction_source", "terminal_correction_sources"), ("evidence_terminal_certainty", "terminal_certainty")):
        info[key] = sorted(terminal[column].dropna().astype(str).unique()) if column in terminal else []
    status = info.pop("status")
    reason = info.pop("reason", None)
    chosen = auxiliary.get("terminal_common_population", pd.DataFrame()).copy()
    auxiliary["terminal_metrics"] = terminal.copy()
    if chosen.empty and status != "blocked" and not _boolean(terminal, "evidence_terminal_applicable").any():
        for legacy in ("theorem7_final_reproduction", "theorem7_terminal_components"):
            metadata[legacy].update(status="not_applicable", reason="Every saved terminal update is outside the supported correction contract")
    for stem, field, name in (("final_reproduction_bound", "evidence_terminal_corrected_ref_bound_rmse", "reference"),
                              ("terminal_observable_bound", "evidence_terminal_corrected_obs_bound_rmse", "observable")):
        if chosen.empty:
            save(stem, pd.DataFrame(), status="blocked" if status == "blocked" else "not_applicable" if not _boolean(terminal, "evidence_terminal_applicable").any() else "unavailable", reason=reason or "No supported terminal correction population", **info)
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
