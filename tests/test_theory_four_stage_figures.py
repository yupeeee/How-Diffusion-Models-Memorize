"""Four-stage scalar contracts; authored source-only, execution left to author."""
import json

import numpy as np
import pandas as pd
import pytest

from tests.test_theory_evidence_figures import evidence_config, evidence_tables
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.four_stage_figures import build_four_stage_plot_inputs
from utils.experiments.theory.paper_registry import GROUPS, evidence_registry, paper_registry


def four_stage_config(**overrides):
    return evidence_config(**{"mean_source": "cached-targets", **overrides})


def four_stage_tables():
    tables = evidence_tables()
    initial = tables["initial"]
    tables["trajectory_metrics"] = tables["trajectory"].copy()
    tables["trajectory_metrics"]["direct_lemma6_gap_rmse"] = np.where(tables["trajectory_metrics"].step_index.eq(0), .5, .7)
    tables["trajectory_metrics"]["direct_unconditional_reference_error_rmse"] = 1. + tables["trajectory_metrics"].seed
    tables["trajectory_metrics"]["direct_target_log_probability"] = np.log(.2 + .4 * tables["trajectory_metrics"].seed)
    tables["trajectory_metrics"]["direct_reference_target_error_rmse"] = .3 + .2 * tables["trajectory_metrics"].seed + .1 * tables["trajectory_metrics"].step_index
    matched = tables["matched_updates"]
    matched["direct_prop5_variation_rmse"] = .25 + .5 * matched.seed + .2 * matched.step_index
    matched["direct_prop5_variation_l2"] = 2. * matched.direct_prop5_variation_rmse
    matched["direct_prop5_integral_status"] = np.where(matched.seed.eq(0), "estimated_converged", "numerically_unresolved")
    matched["direct_prop5_quadrature_budget_exhausted"] = matched.seed.eq(1)
    tables["initial_samples"] = initial.copy()
    fitted = initial.copy()
    fitted["cor3_guidance_fit"] = [-2., 4., 5., 9.]
    fitted["cor3_guidance_fit_residual_rmse"] = [.5, 1.5, .25, .75]
    fitted["cor3_guidance_fit_direction_l2"] = 4.
    fitted["cor3_guidance_fit_direction_tolerance_l2"] = 1e-12
    fitted["cor3_guidance_fit_applicable"] = True
    fitted["cor3_guidance_fit_status"] = "measured"
    fitted["cor3_guidance_fit_qa"] = "consistent_float64_estimate"
    fitted["theory_mean_source"] = "declared_finite_bank_mean"
    fitted["theory_mean_sha256"] = "fixture-bank-mean"
    fitted["guidance_scale"] = 2.
    tables["initial_generated_samples"] = fitted
    reference = tables["gaussian_reference"]
    reference["learned_zero_error_rmse"] = reference.learned_mean_error_rmse + .25
    reference["reference_zero_error_rmse"] = reference.reference_mean_error_rmse + .5
    analytical = tables["reference_analytical"]
    analytical["reference_zero_error_rmse"] = analytical.reference_mean_error_rmse + .5
    tables["reference_atoms"]["atom_zero_distance_rmse"] = [1.5, 3.5]
    for name in ("gaussian_reference", "reference_analytical", "reference_atoms"):
        tables[name]["mean_norm_l2"], tables[name]["mean_norm_rmse"] = 1., .5
        tables[name]["bank_mean_norm_l2"], tables[name]["bank_mean_norm_rmse"] = 1., .5
        tables[name]["mean_offset_l2"], tables[name]["mean_offset_rmse"] = 0., 0.
        tables[name]["theory_mean_source"] = "declared_finite_bank_mean"
        tables[name]["theory_mean_sha256"] = "fixture-bank-mean"
    tables["initial_baseline_summary"] = pd.DataFrame([dict(
        run_id="run", mean_norm_l2=1., spread_rmse=3.,
        baseline_error_rmse=np.sqrt(6.5), relative_baseline_error=np.sqrt(6.5)/3,
        unique_seed_count=2, disagreement_count=0,
    )])
    dose, motion, shapes = [], [], []
    for row in initial.to_dict("records"):
        for index in range(41):
            s = index / 40
            dose.append({**row, "s": s, "log_probability_gain": s if row["seed"] else -2*s,
                         "log_odds_gain": 10*s if row["seed"] else -20*s,
                         "structural_applicable": True, "numerical_status": "finite_flagged_arithmetic_assessment",
                         "endpoint_contract": "fixed_reconstructed_affine_segment",
                         "endpoint_status": "affine_and_saved_agree_under_declared_source_precision"})
        motion.append({**row, "gap_motion_conditional": .3, "gap_motion_unconditional": -.1,
                       "gap_motion_quadratic": .04, "gap_squared_change": .24,
                       "identity_residual": 0., "identity_tolerance": 1e-12, "identity_status": "consistent",
                       "conditional_target_error_squared_change": 0., "unconditional_target_error_squared_change": 0.})
        shapes.append({**row, "step_index": -1, "first_peak_step": 1., "shape_complete": True,
                       "shape_status": "resolved_peak", "peak_ties_json": "[1]", "near_peak_ties_json": "[1]",
                       "initial_gap_rmse": .5, "maximum_gap_rmse": .7, "final_gap_rmse": .7,
                       "early_gap_rmse": .5, "middle_gap_rmse": .5, "late_gap_rmse": .7,
                       "peak_rise_rmse": .2, "post_peak_decline_rmse": 0.,
                       "total_upward_variation_rmse": .2, "total_downward_variation_rmse": 0.})
    tables["feedback_response"] = pd.DataFrame(dose)
    tables["branch_motion"] = pd.DataFrame(motion)
    tables["trajectory_shapes"] = pd.DataFrame(shapes)
    return tables


def assemble(tables=None, **config):
    tables = tables or four_stage_tables()
    source = tables["gaussian_reference"].theory_mean_source.iloc[0]
    sampled = source in {"minimum_snr_unconditional_reference_monte_carlo", "initial_unconditional_reference_monte_carlo"}
    receipt = {"source": source, "vector_sha256": tables["gaussian_reference"].theory_mean_sha256.iloc[0],
               "sample_count": 10000 if sampled else None}
    if sampled:
        initial_level = {"source_range": "native", "step_index": 0, "timestep": 10, "snr": 1., "alpha": np.sqrt(.5), "sigma": np.sqrt(.5)}
        minimum = source == "minimum_snr_unconditional_reference_monte_carlo"
        estimation_snr = 1e-6 if minimum else 1.
        level = {"source_range": "analytical", "grid_index": 0, "step_index": None, "timestep": None,
                 "snr": estimation_snr, "alpha": np.sqrt(estimation_snr / (1 + estimation_snr)), "sigma": np.sqrt(1 / (1 + estimation_snr))} if minimum else initial_level
        receipt.update(mean_seed=123456, initial_snr=1., estimation_snr=estimation_snr,
                       initial_level=initial_level, level=level, reference_snr_decades=6.,
                       mean_mc_standard_error_rmse=.01, estimator_hash="fixture-independent-reference-mean")
    return build_four_stage_plot_inputs(tables, four_stage_config(**config),
        {"reference_law_hash": "law", "reference_law": {"theory_mean": receipt}})


def test_fixed_four_main_sixteen_appendix_and_legacy_diagnostics():
    entries = paper_registry()
    assert [entry["stem"] for entry in entries if entry["category"] == "main"] == [
        "initial_loss_recovery", "branch_gap_posterior_response", "branch_gap_synchronization", "terminal_bound_coverage"]
    assert sum(entry["category"] == "appendix" for entry in entries) == 16
    assert len(evidence_registry()) == 19
    optional = {entry["stem"]: entry for entry in paper_registry(True)}
    assert optional["lemma4_matched_displacement"]["category"] == "diagnostics"
    assert "counterfactual_unconditional_response" not in {entry["stem"] for entry in entries}
    assert len(paper_registry(counterfactual=True)) == 20
    assert optional["counterfactual_unconditional_response"]["category"] == "diagnostics"


def test_pair_rms_unique_seed_baseline_and_full_native_sweep():
    frames, meta = assemble()
    pairs = frames["initial_loss_recovery"]
    np.testing.assert_allclose(pairs.y, np.sqrt(10.25))
    np.testing.assert_allclose(pairs.control_y, np.sqrt(50.))
    np.testing.assert_allclose(pairs.mean_terminal_sscd, .6)
    assert "control_y_low" in pairs and len(pairs) == 2
    assert {"A_c", "Y_c", "U_c", "loss_mean_squared_l2", "loss_mc_se_squared_l2"}.issubset(meta["auxiliary_tables"]["initial_pairs"])
    assert meta["auxiliary_tables"]["initial_baseline_summary"].unique_seed_count.tolist() == [2]
    from utils.experiments.theory.evidence_figures import BOOTSTRAP_POLICY
    uncertainty = meta["figures"]["initial_loss_recovery"]["baseline_uncertainty"]
    assert isinstance(uncertainty, dict)
    for key in ("definition", "replicates", "seed", "confidence", "percentile_method"):
        assert uncertainty[key] == BOOTSTRAP_POLICY[key]
    assert "unique Gaussian evaluation seeds" in uncertainty["scope"]
    assert "repeated prompts add no observations" in uncertainty["seed_dependence"]
    assert "Pair-specific" in BOOTSTRAP_POLICY["scope"]
    assert "saved selected mu" in uncertainty["conditioning"]
    assert "reported separately in theory_mean" in uncertainty["mean_estimation_uncertainty"]
    assert json.loads(json.dumps(uncertainty, allow_nan=False)) == uncertainty
    assert "conditioning" not in BOOTSTRAP_POLICY and "mean_estimation_uncertainty" not in BOOTSTRAP_POLICY
    assert uncertainty is not BOOTSTRAP_POLICY
    sweep = frames["unconditional_reference_convergence"]
    assert set(sweep.loc[sweep.source_range.eq("native"), "step_index"]) == {0, 1}
    assert sweep.loc[sweep.source_range.eq("analytical"), "metric"].eq("reference").all()


RETIRED_ZERO_STEMS = {
    "initial_unconditional_mean_concentration_zero", "unconditional_reference_convergence_zero",
}
ZERO_MEASUREMENT_FIELDS = {
    "gaussian_reference": ["learned_zero_error_rmse", "reference_zero_error_rmse"],
    "reference_analytical": ["reference_zero_error_rmse"],
    "reference_atoms": ["atom_zero_distance_rmse"],
}


def test_active_assembly_retires_zero_figures_and_summary_without_mutating_saved_scalars():
    tables = four_stage_tables()
    before = {name: tables[name].copy(deep=True) for name in ZERO_MEASUREMENT_FIELDS}
    frames, metadata = assemble(tables)
    assert not RETIRED_ZERO_STEMS.intersection(frames)
    assert not RETIRED_ZERO_STEMS.intersection(metadata["figures"])
    assert "zero_baseline_summary" not in metadata["auxiliary_tables"]
    for stem in ("initial_loss_recovery", "initial_unconditional_mean_concentration", "unconditional_reference_convergence"):
        info = metadata["figures"][stem]
        assert info["status"] == "available"
        assert "zero_baseline_summary_table" not in info and "reference_zero_limit_definition" not in info
    for name, original in before.items():
        pd.testing.assert_frame_equal(tables[name], original)


@pytest.mark.parametrize("change", ["missing", "invalid", "different"])
def test_active_figures_do_not_depend_on_retired_zero_measurements(change):
    tables = four_stage_tables()
    original_frames, original_metadata = assemble(tables)
    for name, fields in ZERO_MEASUREMENT_FIELDS.items():
        if change == "missing":
            tables[name] = tables[name].drop(columns=fields)
        else:
            tables[name][fields] = np.nan if change == "invalid" else 12345.
    frames, metadata = assemble(tables)
    for entry in paper_registry():
        stem = entry["stem"]
        pd.testing.assert_frame_equal(frames[stem], original_frames[stem])
        assert metadata["figures"][stem]["status"] == original_metadata["figures"][stem]["status"]
    assert not RETIRED_ZERO_STEMS.intersection(frames)
    assert "zero_baseline_summary" not in metadata["auxiliary_tables"]


def test_selected_mean_consistency_remains_required_after_zero_figures_retire():
    tables = four_stage_tables()
    tables["reference_atoms"].loc[0, "mean_norm_rmse"] = .6
    with pytest.raises(TheoryError, match="inconsistent saved mean_norm_rmse"):
        assemble(tables)


def test_a_zero_valued_selected_mean_keeps_the_selected_mean_figure_contract():
    tables = four_stage_tables()
    for name in ("gaussian_reference", "reference_analytical", "reference_atoms"):
        tables[name]["mean_norm_l2"], tables[name]["mean_norm_rmse"] = 0., 0.
        tables[name]["bank_mean_norm_l2"], tables[name]["bank_mean_norm_rmse"] = 0., 0.
    tables["initial_baseline_summary"]["mean_norm_l2"] = 0.
    frames, metadata = assemble(tables)
    for stem in ("initial_unconditional_mean_concentration", "unconditional_reference_convergence"):
        info = metadata["figures"][stem]
        assert info["mean_norm_rmse"] == 0. and info["theory_mean_source"] == "declared_finite_bank_mean"
        assert info["status"] == "available"
    assert not RETIRED_ZERO_STEMS.intersection(frames)


@pytest.mark.parametrize("source, option, selected_norm, offset", [
    ("initial_unconditional_reference_monte_carlo", "reference-initial", .8, .3),
    ("minimum_snr_unconditional_reference_monte_carlo", "reference-min-snr", .500002, .000003),
])
def test_selected_reference_mean_receipt_keeps_bank_mean_and_selected_offset_separate(source, option, selected_norm, offset):
    tables = four_stage_tables()
    for name in ("gaussian_reference", "reference_analytical", "reference_atoms"):
        tables[name]["theory_mean_source"] = source
        tables[name]["theory_mean_sha256"] = "fixture-selected-reference-mean"
        tables[name]["mean_norm_l2"], tables[name]["mean_norm_rmse"] = selected_norm * 2, selected_norm
        tables[name]["mean_offset_l2"], tables[name]["mean_offset_rmse"] = offset * 2, offset
    tables["initial_baseline_summary"]["mean_norm_l2"] = selected_norm * 2
    tables["initial_generated_samples"]["theory_mean_source"] = source
    tables["initial_generated_samples"]["theory_mean_sha256"] = "fixture-selected-reference-mean"
    frames, metadata = assemble(tables, mean_source=option)
    original = metadata["figures"]["unconditional_reference_convergence"]
    assert original["reference_limit_rmse"] == offset > 0
    assert original["bank_mean_norm_rmse"] == .5
    for info in (original, metadata["figures"]["initial_unconditional_mean_concentration"],
                 metadata["figures"]["initial_loss_recovery"]):
        assert info["theory_mean_source"] == source
        assert info["theory_mean_sha256"] == "fixture-selected-reference-mean"
        receipt = info["theory_mean"]
        assert receipt["sample_count"] == 10000 and receipt["mean_mc_standard_error_rmse"] == .01
        assert receipt["initial_snr"] == 1.
        assert info["mean_norm_rmse"] == selected_norm and info["bank_mean_norm_rmse"] == .5
        if option == "reference-min-snr":
            assert receipt["estimation_snr"] == 1e-6 and receipt["reference_snr_decades"] == 6.
            assert receipt["level"]["source_range"] == "analytical" and receipt["level"]["grid_index"] == 0
            assert receipt["level"]["step_index"] is None and receipt["level"]["timestep"] is None
            assert receipt["initial_level"]["step_index"] == 0
            assert "smallest SNR" in info["theory_mean_definition"] and "need not equal" in info["theory_mean_definition"]
        else:
            assert receipt["estimation_snr"] == 1. and receipt["level"]["source_range"] == "native"
            assert "explicitly selected legacy" in info["theory_mean_definition"]
    assert original["mean_offset_rmse"] == offset
    assert "zero_baseline_summary" not in metadata["auxiliary_tables"]
    assert not RETIRED_ZERO_STEMS.intersection(frames)
    assert set(frames["unconditional_reference_convergence"].metric) == {"reference", "learned", "reference_error"}


def test_mean_scalar_receipts_cannot_silently_mix_selected_centres():
    tables = four_stage_tables()
    tables["reference_atoms"]["theory_mean_sha256"] = "different-centre"
    with pytest.raises(TheoryError, match="receipts disagree"):
        assemble(tables)


def test_fresh_gaussian_controls_keep_provenance_and_actual_pair_outcomes():
    tables = four_stage_tables()
    gaussian = tables["gaussian_conditional"]
    gaussian["gaussian_control_status"] = "measured_independent_gaussian_control"
    gaussian["gaussian_control_input_source"] = "true_gaussian_initialization_probe"
    gaussian["gaussian_control_input_sha256"] = "input-hash"
    gaussian["gaussian_control_task_hash"] = "task-hash"
    gaussian["terminal_sscd"] = np.nan
    gaussian["terminal_sscd_matched_initialization"] = False
    frames, meta = assemble(tables)
    np.testing.assert_allclose(frames["initial_loss_recovery"].mean_terminal_sscd, .6)
    assert meta["figures"]["initial_loss_recovery"]["matched_control_status_counts"] == {"measured_independent_gaussian_control": 4}


def test_dose_keeps_negative_and_unresolved_curves_fixed_cohort():
    frames, meta = assemble()
    response = frames["branch_gap_posterior_response"]
    assert response.loc[response.group.eq(GROUPS[1]) & response.s.eq(1), "median"].iloc[0] == -2
    assert all(rows.denominator_weight.nunique() == 1 for _, rows in response.groupby("group"))
    assert len(response) == 82
    assert meta["figures"]["branch_gap_posterior_response"]["dose_grid"] == [k/40 for k in range(41)]
    assert not meta["audit"]["blocking"]


def test_incomplete_curve_exclusion_does_not_change_later_dose_denominator():
    tables = four_stage_tables()
    rows = tables["feedback_response"]
    rows.loc[rows.original_index.eq("0") & rows.seed.eq(0) & rows.s.eq(.5), "log_probability_gain"] = np.nan
    frames, meta = assemble(tables)
    lower = frames["branch_gap_posterior_response"].loc[lambda frame: frame.group.eq(GROUPS[1])]
    assert lower.eligible_count.eq(1).all()
    assert meta["figures"]["branch_gap_posterior_response"]["counts"]["excluded_curves"] == 1
    assert len(meta["auxiliary_tables"]["trajectory_metrics"]) == 8


def test_wrong_target_agreement_and_missing_reference_do_not_remove_main_rows():
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    rows["direct_lemma6_gap_rmse"] = 0.
    rows.loc[rows.step_index.eq(1), "direct_lemma6_rhs_rmse"] = np.nan
    frames, _ = assemble(tables)
    main = frames["branch_gap_synchronization"]
    assert main.loc[main.metric.eq("gap"), "median"].eq(0).all()
    assert main.loc[main.metric.eq("joint_error"), "median"].eq(.7).all()
    assert set(main.step_index) == {0, 1}
    assert main.loc[main.step_index.eq(1), "denominator_weight"].gt(0).all()


def test_motion_means_are_additive_and_flat_peaks_keep_unassigned_mass():
    tables = four_stage_tables()
    shapes = tables["trajectory_shapes"]
    shapes.loc[0, ["first_peak_step", "shape_status"]] = [np.nan, "flat"]
    frames, meta = assemble(tables)
    for stem in ("branch_gap_motion_high_sscd", "branch_gap_motion_lower_sscd"):
        row = frames[stem].set_index("metric")["mean"]
        assert row["change"] == pytest.approx(row["conditional"] + row["unconditional"] + row["quadratic"])
    assert meta["figures"]["branch_gap_peak_step"]["shape_group_counts"][GROUPS[1]]["unassigned_weight_fraction"] == .5


def test_terminal_actual_exceedances_remain_visible_but_invalid_construction_blocks():
    tables = four_stage_tables()
    tables["terminal"].loc[1, "direct_theorem7_endpoint_error_rmse"] = 20.
    tables["terminal"].loc[1, "direct_theorem7_endpoint_error_l2"] = 40.
    frames, meta = assemble(tables)
    assert frames["final_reproduction_bound"].y.max() == 20.
    grouped_actual = frames["terminal_bound_coverage"].loc[lambda rows: rows.distribution.eq("actual")]
    assert grouped_actual.value.max() == 20.
    assert meta["figures"]["terminal_bound_coverage"]["status"] == "available"
    assert meta["figures"]["final_reproduction_bound"]["status"] == "available"
    assert not meta["audit"]["blocking"]
    tables["terminal"].loc[1, "evidence_terminal_reconstruction_status"] = "failed_reconstruction"
    _, blocked = assemble(tables)
    assert blocked["audit"]["blocking"]
    assert blocked["figures"]["final_reproduction_bound"]["status"] == "blocked"
    assert blocked["figures"]["terminal_bound_coverage"]["status"] == "blocked"


def test_terminal_grouped_cdfs_share_populations_and_preserve_pooled_audit():
    frames, metadata = assemble(target_error_tolerance=4.)
    frame = frames["terminal_bound_coverage"]
    info = metadata["figures"]["terminal_bound_coverage"]
    assert info["formula_version"] == "terminal-coverage-by-sscd-1"
    assert set(frame.group) == set(GROUPS)
    for group in GROUPS:
        receipt = info["group_population"][group]
        assert receipt == {"eligible_count": 2, "prompt_count": 2, "denominator_weight": 2.}
        rows = frame.loc[frame.group.eq(group)]
        assert set(rows.distribution) == {"actual", "observable", "reference"}
        assert rows.eligible_count.eq(2).all() and rows.denominator_weight.eq(2).all()
        for _, curve in rows.groupby("distribution"):
            assert curve.cdf.iloc[-1] == 1. and curve["count"].sum() == 2
        def at(distribution, tolerance):
            values = rows.loc[rows.distribution.eq(distribution) & rows.value.le(tolerance), "cdf"]
            return values.iloc[-1] if len(values) else 0.
        for tolerance in sorted(rows.value.unique()):
            assert at("reference", tolerance) <= at("observable", tolerance) <= at("actual", tolerance)
    assert info["grouped_zero_mass"][GROUPS[1]] == {"actual": .5, "observable": .5, "reference": .5}
    assert info["grouped_predeclared_coverage"][GROUPS[0]] == {"actual": .5, "observable": .5, "reference": 0.}
    assert info["grouped_predeclared_coverage"][GROUPS[1]] == {"actual": 1., "observable": .5, "reference": .5}
    auxiliary = metadata["auxiliary_tables"]
    pd.testing.assert_frame_equal(auxiliary["terminal_pooled_cdf"], frames["theorem7_final_reproduction"])
    pooled = json.loads(auxiliary["terminal_pooled_cdf_metadata"].metadata_json.iloc[0])
    assert pooled["zero_mass"] == {"actual": .25, "observable": .25, "reference": .25}
    assert pooled["weighting"].endswith("no SSCD split")
    assert "zero_mass" not in info and "denominator_weight" not in info
    for key in ("terminal_scope", "original_clean_counts", "paired_ordering_audit", "noise_probability_scope"):
        assert info[key] == pooled[key]
    common = auxiliary["terminal_grouped_common_population"]
    assert len(common) == 4 and not common.duplicated(["run_id", "original_index", "record_id", "target_id", "seed"]).any()
    for group, rows in common.groupby("group"):
        assert rows.prompt_weight.sum() == info["group_population"][group]["denominator_weight"]


def test_terminal_grouped_cdf_weights_prompts_equally_with_unequal_member_seed_counts():
    tables = four_stage_tables()
    extra = tables["terminal"].iloc[[0]].copy()
    extra["seed"], extra["terminal_sscd"] = 2, .8
    tables["terminal"] = pd.concat([tables["terminal"], extra], ignore_index=True)
    frames, metadata = assemble(tables)
    high = frames["terminal_bound_coverage"].loc[lambda rows: rows.group.eq(GROUPS[0]) & rows.distribution.eq("actual")]
    np.testing.assert_allclose(high.value, [0., 1., 3.])
    np.testing.assert_allclose(high.cdf, [.25, .5, 1.])
    assert metadata["figures"]["terminal_bound_coverage"]["group_population"][GROUPS[0]] == {
        "eligible_count": 3, "prompt_count": 2, "denominator_weight": 2.}
    common = metadata["auxiliary_tables"]["terminal_grouped_common_population"]
    high = common.loc[common.group.eq(GROUPS[0])]
    np.testing.assert_allclose(high.groupby("record_id").prompt_weight.sum(), 1.)


def test_terminal_grouped_scores_use_same_seed_threshold_and_audit_nonfinite_exclusions():
    tables = four_stage_tables()
    tables["terminal"]["terminal_sscd"] = [.75, .8, np.nan, np.inf]
    frames, metadata = assemble(tables)
    info = metadata["figures"]["terminal_bound_coverage"]
    assert info["counts"]["missing_or_nonfinite_sscd"] == 2
    assert info["counts"]["eligible"] == 2 and info["counts"]["common_eligible"] == 4
    common = metadata["auxiliary_tables"]["terminal_grouped_common_population"]
    assert common.loc[common.seed.eq(0), "group"].eq(GROUPS[1]).all()
    assert common.loc[common.seed.eq(1), "group"].eq(GROUPS[0]).all()
    excluded = metadata["auxiliary_tables"]["terminal_grouped_sscd_exclusions"]
    assert len(excluded) == 2 and excluded.original_index.eq("1").all()
    assert excluded.exclusion_reason.eq("missing_or_nonfinite_same_seed_terminal_sscd").all()
    assert frames["terminal_bound_coverage"].eligible_count.eq(1).all()
    assert metadata["figures"]["theorem7_final_reproduction"]["counts"]["eligible"] == 4


def test_terminal_grouped_population_is_common_before_grouping_any_curve():
    tables = four_stage_tables()
    tables["terminal"].loc[1, "evidence_terminal_corrected_ref_bound_rmse"] = np.nan
    frames, metadata = assemble(tables)
    info = metadata["figures"]["terminal_bound_coverage"]
    assert info["counts"]["excluded_before_sscd_grouping"] == 1
    high = frames["terminal_bound_coverage"].loc[lambda rows: rows.group.eq(GROUPS[0])]
    assert high.eligible_count.eq(1).all() and high.prompt_count.eq(1).all()
    assert high.distribution.nunique() == 3
    common = metadata["auxiliary_tables"]["terminal_grouped_common_population"]
    assert len(common) == 3
    assert not (common.original_index.eq("0") & common.seed.eq(1)).any()


def test_terminal_grouped_cdf_keeps_exact_zero_and_infinite_bound_masses():
    tables = four_stage_tables()
    for field in ("evidence_terminal_corrected_obs_bound_rmse", "evidence_terminal_corrected_ref_bound_rmse", "evidence_terminal_defect_bound_rmse"):
        tables["terminal"].loc[0, field] = np.inf
    frames, metadata = assemble(tables)
    info = metadata["figures"]["terminal_bound_coverage"]
    assert info["grouped_zero_mass"][GROUPS[1]]["actual"] == .5
    assert info["grouped_infinite_mass"][GROUPS[1]] == {"actual": 0., "observable": .5, "reference": .5}
    lower = frames["terminal_bound_coverage"].loc[lambda rows: rows.group.eq(GROUPS[1])]
    assert lower.loc[lower.distribution.eq("actual"), "value"].iloc[0] == 0.
    for distribution in ("observable", "reference"):
        curve = lower.loc[lower.distribution.eq(distribution)]
        assert np.isposinf(curve.value.iloc[-1]) and curve.cdf.iloc[-1] == 1.
        assert curve.cdf.iloc[-2] == .5


def test_terminal_grouped_empty_group_is_explicit_and_all_missing_is_unavailable():
    tables = four_stage_tables()
    tables["terminal"]["terminal_sscd"] = .5
    frames, metadata = assemble(tables)
    info = metadata["figures"]["terminal_bound_coverage"]
    assert info["status"] == "available" and info["empty_groups"] == [GROUPS[0]]
    assert info["group_population"][GROUPS[0]] == {"eligible_count": 0, "prompt_count": 0, "denominator_weight": 0.}
    assert all(value is None for value in info["grouped_zero_mass"][GROUPS[0]].values())
    assert set(frames["terminal_bound_coverage"].group) == {GROUPS[1]}
    tables["terminal"]["terminal_sscd"] = np.nan
    frames, metadata = assemble(tables)
    info = metadata["figures"]["terminal_bound_coverage"]
    assert info["status"] == "unavailable" and info["empty_groups"] == list(GROUPS)
    assert info["counts"]["missing_or_nonfinite_sscd"] == 4
    assert frames["terminal_bound_coverage"].empty
    assert metadata["figures"]["theorem7_final_reproduction"]["status"] == "available"
    tables["terminal"].loc[0, "evidence_terminal_reconstruction_status"] = "failed_reconstruction"
    _, blocked = assemble(tables)
    assert blocked["figures"]["terminal_bound_coverage"]["status"] == "blocked"
    assert blocked["audit"]["blocking"]


def test_terminal_grouped_preserves_mathematical_inapplicability():
    tables = four_stage_tables()
    tables["terminal"]["evidence_terminal_applicable"] = False
    tables["terminal"]["terminal_sscd"] = np.nan
    frames, metadata = assemble(tables)
    info = metadata["figures"]["terminal_bound_coverage"]
    assert info["status"] == "not_applicable"
    assert info["reason"] == "Every saved terminal update is outside the supported correction contract"
    assert info["counts"]["common_eligible"] == 0
    assert info["counts"]["excluded_before_sscd_grouping"] == 4
    assert info["counts"]["missing_or_nonfinite_sscd"] == 0
    assert frames["terminal_bound_coverage"].empty



@pytest.mark.parametrize("case", ["missing_column", "all_missing", "partly_missing", "empty", "explicit_false", "saved_false", "mixed_true_false"])
def test_terminal_empty_population_needs_explicit_false_flags_for_inapplicability(case):
    from utils.experiments.theory.four_stage_figures import _terminal_inputs
    terminal = pd.DataFrame({"seed": [0, 1]})
    if case == "all_missing":
        terminal["evidence_terminal_applicable"] = [None, pd.NA]
    elif case == "partly_missing":
        terminal["evidence_terminal_applicable"] = [False, np.nan]
    elif case == "empty":
        terminal = pd.DataFrame(columns=["seed", "evidence_terminal_applicable"])
    elif case == "explicit_false":
        terminal["evidence_terminal_applicable"] = [False, 0]
    elif case == "saved_false":
        terminal["evidence_terminal_applicable"] = ["False", " 0 "]
    elif case == "mixed_true_false":
        terminal["evidence_terminal_applicable"] = ["False", "True"]
    metadata = {stem: {"status": "unavailable", "reason": "No supported terminal population"}
        for stem in ("theorem7_final_reproduction", "theorem7_terminal_components")}
    saved, auxiliary = {}, {}
    def save(stem, frame, **info):
        saved[stem] = info
    _terminal_inputs(terminal, {}, metadata, {}, save, auxiliary)
    expected = "not_applicable" if case in {"explicit_false", "saved_false"} else "unavailable"
    assert all(info["status"] == expected for info in metadata.values())
    assert all(info["status"] == expected for info in saved.values())
    assert set(saved) == {"final_reproduction_bound", "terminal_observable_bound"}


def test_terminal_empty_population_rejects_unknown_applicability_flags():
    from utils.experiments.theory.four_stage_figures import _terminal_inputs
    terminal = pd.DataFrame({"evidence_terminal_applicable": [False, "unresolved"]})
    metadata = {"theorem7_final_reproduction": {"status": "unavailable", "reason": None}}
    with pytest.raises(TheoryError, match="must contain saved Boolean values"):
        _terminal_inputs(terminal, {}, metadata, {}, lambda *args, **kwargs: None, {})


def test_mandatory_baseline_and_first_snapshot_cannot_be_silently_replaced():
    tables = four_stage_tables()
    tables.pop("initial_baseline_summary")
    with pytest.raises(TheoryError, match="baseline report"):
        assemble(tables)
    tables = four_stage_tables()
    tables["feedback_response"]["step_index"] = 1
    _, meta = assemble(tables)
    assert meta["figures"]["branch_gap_posterior_response"]["status"] == "unavailable"


def test_observed_terminal_policy_never_clears_an_inherited_nonfigure_blocker():
    from utils.experiments.theory.evidence_figures import build_evidence_plot_inputs
    from utils.experiments.theory.four_stage_figures import _terminal_display_policy
    tables = four_stage_tables()
    tables["terminal"].loc[1, "direct_theorem7_endpoint_error_rmse"] = 20.
    _, metadata = build_evidence_plot_inputs(tables, four_stage_config(), {"reference_law_hash": "law"})
    metadata["audit"]["inherited_nonfigure_blocking"] = True
    _terminal_display_policy(tables["terminal"], metadata["figures"], metadata["audit"], metadata["auxiliary_tables"])
    assert metadata["audit"]["blocking"]


def test_optional_counterfactual_retains_I_net_and_does_not_block_default():
    tables = four_stage_tables()
    optional = tables["initial"].copy()
    optional["status"] = "measured"
    optional["counterfactual_applicable"] = True
    optional["counterfactual_I_net"] = .4
    optional["counterfactual_target_error_rmse"] = .7
    optional["actual_next_target_error_rmse"] = .3
    tables["counterfactual_unconditional"] = optional
    frames, _ = assemble(tables, counterfactual_unconditional=True)
    assert frames["counterfactual_unconditional_response"].counterfactual_I_net.eq(.4).all()
    tables["counterfactual_unconditional"].loc[0, "status"] = "failed"
    _, default = assemble(tables)
    assert not default["audit"]["blocking"]
    _, requested = assemble(tables, counterfactual_unconditional=True)
    assert requested["audit"]["blocking"]


def test_prompt_gap_means_saved_norms_once_with_fixed_full_seed_color():
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    rows["direct_lemma6_gap_rmse"] = 1. + 2. * rows.seed + rows.step_index
    rows["prompt_raw"] = "same raw prompt for distinct targets"
    tables["trajectory_metrics"] = rows.iloc[::-1].copy()
    frames, metadata = assemble(tables)
    curves = frames["branch_gap_per_prompt"]
    assert len(curves) == 4 and curves.target_id.nunique() == 2
    for _, curve in curves.groupby("record_id", sort=False):
        np.testing.assert_array_equal(curve.step_index, [0, 1])
        np.testing.assert_allclose(curve.mean_gap_rmse, [2., 3.])
        np.testing.assert_allclose(curve.mean_terminal_sscd, [.6, .6])
        assert curve.seed_count.eq(2).all() and curve.cohort_complete.all()
        assert curve.seed_ids_json.eq("[0, 1]").all()
    info = metadata["figures"]["branch_gap_per_prompt"]
    assert info["counts"]["plotted_pairs"] == 2 and info["counts"]["excluded_pairs"] == 0
    assert "not the norm of a mean vector" in info["aggregation"]
    assert "no second normalization" in info["normalization"]


@pytest.mark.parametrize("failure", ["missing_step", "missing_seed", "nonfinite_gap", "different_score", "different_dimension"])
def test_prompt_gap_excludes_whole_incomplete_pair_without_changing_seed_denominator(failure):
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    selected = rows.original_index.eq("0") & rows.seed.eq(0) & rows.step_index.eq(1)
    if failure == "missing_step":
        tables["trajectory_metrics"] = rows.loc[~selected].copy()
    elif failure == "missing_seed":
        tables["trajectory_metrics"] = rows.loc[~(rows.original_index.eq("0") & rows.seed.eq(0))].copy()
    elif failure == "nonfinite_gap":
        rows.loc[selected, "direct_lemma6_gap_rmse"] = np.nan
    elif failure == "different_score":
        rows.loc[selected, "terminal_sscd"] = .99
    else:
        rows.loc[selected, "latent_dimension"] = 16
    frames, metadata = assemble(tables)
    curves = frames["branch_gap_per_prompt"]
    assert curves.original_index.eq("1").all() and len(curves) == 2
    assert curves.seed_count.eq(2).all()
    audit = metadata["auxiliary_tables"]["branch_gap_per_prompt_cohort"]
    assert len(audit) == 2 and audit.eligible.sum() == 1
    rejected = audit.loc[audit.original_index.eq("0")].iloc[0]
    assert not rejected.eligible and rejected.reason != "complete_fixed_seed_cohort"
    assert metadata["figures"]["branch_gap_per_prompt"]["counts"]["excluded_pairs"] == 1


def test_prompt_gap_rejects_duplicate_seed_step_identities():
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    tables["trajectory_metrics"] = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    with pytest.raises(TheoryError, match="Duplicate per-prompt branch-gap"):
        assemble(tables)


def test_prompt_gap_all_missing_curves_remain_unavailable_with_audit():
    tables = four_stage_tables()
    tables["trajectory_metrics"]["direct_lemma6_gap_rmse"] = np.nan
    frames, metadata = assemble(tables)
    assert frames["branch_gap_per_prompt"].empty
    assert metadata["figures"]["branch_gap_per_prompt"]["status"] == "unavailable"
    assert metadata["figures"]["branch_gap_per_prompt"]["counts"]["excluded_pairs"] == 2
    assert not metadata["auxiliary_tables"]["branch_gap_per_prompt_cohort"].eligible.any()


def test_prompt_gap_supports_single_prediction_and_audits_missing_whole_terminal_seed():
    from utils.experiments.theory.four_stage_figures import _prompt_gap_inputs
    tables = four_stage_tables()
    trajectory = tables["trajectory_metrics"].loc[lambda frame: frame.step_index.eq(0)]
    initial = tables["initial"].loc[lambda frame: ~(frame.original_index.eq("0") & frame.seed.eq(0))]
    saved, auxiliary = {}, {}
    def save(stem, frame, **metadata):
        saved[stem] = (frame, metadata)
    _prompt_gap_inputs(trajectory, initial, four_stage_config(num_inference_steps=1), save, auxiliary)
    frame, info = saved["branch_gap_per_prompt"]
    assert len(frame) == 1 and frame.original_index.eq("1").all() and frame.step_index.eq(0).all()
    assert frame.mean_gap_rmse.iloc[0] == pytest.approx(.5)
    assert frame.mean_terminal_sscd.iloc[0] == pytest.approx(.6)
    assert info["prediction_steps"] == 1 and info["counts"]["excluded_pairs"] == 1
    rejected = auxiliary["branch_gap_per_prompt_cohort"].loc[lambda rows: rows.original_index.eq("0")].iloc[0]
    assert "incomplete_or_unexpected_terminal_seed_cohort" in rejected.reason


PROMPT_REFERENCE_FIELDS = {
    "conditional_reference_error_per_prompt": ("direct_conditional_error_rmse", "mean_conditional_error_rmse"),
    "unconditional_reference_error_per_prompt": ("direct_unconditional_reference_error_rmse", "mean_unconditional_reference_error_rmse"),
    "target_probability_per_prompt": ("direct_target_log_probability", "mean_target_probability"),
    "reference_branch_gap_per_prompt": ("direct_reference_target_error_rmse", "mean_reference_gap_rmse"),
}


def test_prompt_reference_errors_average_seed_norms_once_and_use_reference_not_target_error():
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    rows["direct_conditional_error_rmse"] = 1. + 2. * rows.seed + rows.step_index
    rows["direct_unconditional_reference_error_rmse"] = 4. + 4. * rows.seed + 2. * rows.step_index
    rows["direct_unconditional_target_error_rmse"] = 111.
    rows["prompt_raw"] = "same words, distinct prompt-target identities"
    tables["trajectory_metrics"] = rows.iloc[::-1].copy()
    before = tables["trajectory_metrics"].copy(deep=True)
    frames, metadata = assemble(tables)
    for stem, expected in (("conditional_reference_error_per_prompt", [2., 3.]),
                           ("unconditional_reference_error_per_prompt", [6., 8.])):
        frame = frames[stem]
        field = PROMPT_REFERENCE_FIELDS[stem][1]
        assert len(frame) == 4 and frame.target_id.nunique() == 2
        for _, curve in frame.groupby("record_id", sort=False):
            np.testing.assert_array_equal(curve.step_index, [0, 1])
            np.testing.assert_allclose(curve[field], expected)
            np.testing.assert_allclose(curve.mean_terminal_sscd, [.6, .6])
            assert curve.seed_count.eq(2).all() and curve.cohort_complete.all()
            assert curve.seed_ids_json.eq("[0, 1]").all()
        info = metadata["figures"][stem]
        assert info["formula_version"] == "prompt-trajectory-scalars-1"
        assert info["value_column"] == field and info["value_domain"] == "nonnegative"
        assert "no second normalization" in info["normalization"]
        assert info["cohort_audit_table"] == "audit_data/" + stem + "_cohort.csv"
    pd.testing.assert_frame_equal(tables["trajectory_metrics"], before)


def test_prompt_probability_is_arithmetic_mean_of_seed_probabilities_and_retains_negative_infinity():
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    rows["direct_target_log_probability"] = np.where(rows.seed.eq(0), np.log(.25), np.log(.81))
    rows.loc[rows.step_index.eq(1) & rows.seed.eq(0), "direct_target_log_probability"] = -np.inf
    rows.loc[rows.step_index.eq(1) & rows.seed.eq(1), "direct_target_log_probability"] = np.log(.6)
    frames, metadata = assemble(tables)
    frame = frames["target_probability_per_prompt"]
    for _, curve in frame.groupby("record_id", sort=False):
        np.testing.assert_allclose(curve.mean_target_probability, [.53, .3])
        assert curve.mean_target_probability.iloc[0] != pytest.approx(np.sqrt(.25 * .81))
        np.testing.assert_allclose(curve.mean_terminal_sscd, [.6, .6])
    info = metadata["figures"]["target_probability_per_prompt"]
    assert info["value_column"] == "mean_target_probability" and info["value_domain"] == "probability"
    assert info["negative_infinite_log_probability_rows"] == 2
    assert info["finite_log_probability_underflow_rows"] == 0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1e-15])
def test_prompt_probability_invalid_log_values_exclude_only_its_complete_curve(invalid):
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    selected = rows.original_index.eq("0") & rows.seed.eq(0) & rows.step_index.eq(1)
    rows.loc[selected, "direct_target_log_probability"] = invalid
    frames, metadata = assemble(tables)
    frame = frames["target_probability_per_prompt"]
    assert len(frame) == 2 and frame.original_index.eq("1").all()
    assert frame.seed_count.eq(2).all()
    for stem in ("branch_gap_per_prompt", "conditional_reference_error_per_prompt", "unconditional_reference_error_per_prompt"):
        assert len(frames[stem]) == 4
    audit = metadata["auxiliary_tables"]["target_probability_per_prompt_cohort"]
    rejected = audit.loc[audit.original_index.eq("0")].iloc[0]
    assert rejected.invalid_value_rows == 1 and "invalid_saved_target_log_probability" in rejected.reason


def test_prompt_probability_audits_finite_log_underflow_separately_from_exact_zero():
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    chosen = rows.original_index.eq("0")
    rows.loc[chosen & rows.step_index.eq(0) & rows.seed.eq(0), "direct_target_log_probability"] = -1000.
    rows.loc[chosen & rows.step_index.eq(0) & rows.seed.eq(1), "direct_target_log_probability"] = np.log(.4)
    rows.loc[chosen & rows.step_index.eq(1) & rows.seed.eq(0), "direct_target_log_probability"] = -np.inf
    rows.loc[chosen & rows.step_index.eq(1) & rows.seed.eq(1), "direct_target_log_probability"] = np.log(.8)
    frames, metadata = assemble(tables)
    curve = frames["target_probability_per_prompt"].loc[lambda frame: frame.original_index.eq("0")]
    np.testing.assert_allclose(curve.mean_target_probability, [.2, .4])
    audit = metadata["auxiliary_tables"]["target_probability_per_prompt_cohort"]
    receipt = audit.loc[audit.original_index.eq("0")].iloc[0]
    assert receipt.eligible and receipt.finite_log_probability_underflow_rows == 1
    assert receipt.negative_infinite_log_probability_rows == 1
    assert json.loads(receipt.finite_log_probability_underflow_observations_json) == [
        {"seed": 0, "step_index": 0, "direct_target_log_probability": -1000.}]
    info = metadata["figures"]["target_probability_per_prompt"]
    assert info["finite_log_probability_underflow_rows"] == 1
    assert "representational zero" in info["probability_underflow"]


@pytest.mark.parametrize("failure", ["missing_step", "missing_seed", "different_score", "different_dimension"])
def test_prompt_reference_figures_require_complete_same_seed_time_and_color_cohorts(failure):
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    selected = rows.original_index.eq("0") & rows.seed.eq(0) & rows.step_index.eq(1)
    if failure == "missing_step":
        tables["trajectory_metrics"] = rows.loc[~selected].copy()
    elif failure == "missing_seed":
        tables["trajectory_metrics"] = rows.loc[~(rows.original_index.eq("0") & rows.seed.eq(0))].copy()
    elif failure == "different_score":
        rows.loc[selected, "terminal_sscd"] = .99
    else:
        rows.loc[selected, "latent_dimension"] = 16
    frames, metadata = assemble(tables)
    for stem in PROMPT_REFERENCE_FIELDS:
        frame = frames[stem]
        assert len(frame) == 2 and frame.original_index.eq("1").all()
        assert frame.seed_count.eq(2).all()
        assert metadata["figures"][stem]["counts"]["excluded_pairs"] == 1
        audit = metadata["auxiliary_tables"][stem + "_cohort"]
        assert len(audit) == 2 and audit.eligible.sum() == 1


@pytest.mark.parametrize("stem", ["conditional_reference_error_per_prompt", "unconditional_reference_error_per_prompt", "reference_branch_gap_per_prompt"])
def test_prompt_error_nonfinite_values_do_not_remove_other_figures_cohorts(stem):
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    selected = rows.original_index.eq("0") & rows.seed.eq(0) & rows.step_index.eq(1)
    rows.loc[selected, PROMPT_REFERENCE_FIELDS[stem][0]] = np.nan
    frames, _ = assemble(tables)
    assert len(frames[stem]) == 2 and frames[stem].original_index.eq("1").all()
    for other in set(PROMPT_REFERENCE_FIELDS) - {stem}:
        assert len(frames[other]) == 4


@pytest.mark.parametrize("missing_stem", list(PROMPT_REFERENCE_FIELDS))
def test_prompt_reference_missing_source_column_is_unavailable_with_audit(missing_stem):
    from utils.experiments.theory.four_stage_figures import _prompt_reference_inputs
    tables = four_stage_tables()
    missing_column, value_column = PROMPT_REFERENCE_FIELDS[missing_stem]
    trajectory = tables["trajectory_metrics"].drop(columns=missing_column)
    saved, auxiliary = {}, {}

    def save(stem, frame, **metadata):
        saved[stem] = frame, metadata

    _prompt_reference_inputs(trajectory, tables["initial"], four_stage_config(), save, auxiliary)
    frame, info = saved[missing_stem]
    assert frame.empty and value_column in frame
    assert info["status"] == "unavailable" and "--recompute-experiments" in info["reason"]
    assert missing_column in info["reason"]
    audit = auxiliary[missing_stem + "_cohort"]
    assert len(audit) == 2 and not audit.eligible.any()
    assert audit.reason.str.contains("missing_source_column:" + missing_column).all()
    for stem in set(PROMPT_REFERENCE_FIELDS) - {missing_stem}:
        assert len(saved[stem][0]) == 4 and saved[stem][1]["status"] == "available"



def test_reference_branch_gap_averages_saved_reference_norms_not_learned_gap_or_error_difference():
    tables = four_stage_tables()
    rows = tables["trajectory_metrics"]
    rows["direct_reference_target_error_rmse"] = 2. + 6. * rows.seed + 3. * rows.step_index
    rows["direct_lemma6_gap_rmse"] = 17.
    rows["direct_conditional_error_rmse"] = 23.
    rows["direct_unconditional_reference_error_rmse"] = 9.
    rows["direct_unconditional_target_error_rmse"] = 41.
    rows["direct_radius_tail_rmse"] = 50.
    tables["trajectory_metrics"] = rows.iloc[::-1].copy()
    before = tables["trajectory_metrics"].copy(deep=True)
    frames, metadata = assemble(tables)
    stem = "reference_branch_gap_per_prompt"
    frame = frames[stem]
    assert len(frame) == 4 and frame.target_id.nunique() == 2
    for _, curve in frame.groupby("record_id", sort=False):
        np.testing.assert_array_equal(curve.step_index, [0, 1])
        np.testing.assert_allclose(curve.mean_reference_gap_rmse, [5., 8.])
        np.testing.assert_allclose(curve.mean_terminal_sscd, [.6, .6])
        assert curve.seed_count.eq(2).all() and curve.cohort_complete.all()
        assert curve.seed_ids_json.eq("[0, 1]").all()
    info = metadata["figures"][stem]
    assert info["source_column"] == "direct_reference_target_error_rmse"
    assert info["value_column"] == "mean_reference_gap_rmse" and info["value_domain"] == "nonnegative"
    assert "no second normalization" in info["normalization"]
    assert "not the norm of a mean vector" in info["aggregation"]
    assert info["counts"]["plotted_pairs"] == 2
    pd.testing.assert_frame_equal(tables["trajectory_metrics"], before)


GUIDANCE_FIT_STEM = "corollary3_guidance_scale_vs_loss"


def reduce_guidance_fit(tables=None):
    from utils.experiments.theory.four_stage_figures import _guidance_fit_inputs, _theory_mean_metadata
    tables = four_stage_tables() if tables is None else tables
    saved, auxiliary = {}, {}
    def save(stem, frame, **metadata):
        saved[stem] = (frame, metadata)
    mean = _theory_mean_metadata(tables, {})
    _guidance_fit_inputs(tables, four_stage_config(), mean, save, auxiliary)
    frame, info = saved[GUIDANCE_FIT_STEM]
    return frame, info, auxiliary


def test_guidance_fit_uses_true_loss_ratio_joint_signed_fit_and_same_cohort_sscd():
    tables = four_stage_tables()
    original = {name: frame.copy(deep=True) for name, frame in tables.items()}
    frame, info, auxiliary = reduce_guidance_fit(tables)
    np.testing.assert_allclose(frame.x, [4., 8.])  # Neither sqrt(loss/(d*SNR)) nor an error norm.
    np.testing.assert_allclose(frame.y, [1., 7.])  # Mean of signed fits; no abs/clipping/g substitution.
    np.testing.assert_allclose(frame.mean_terminal_sscd, [.6, .6])
    np.testing.assert_allclose(frame.fit_residual_rmse, [np.sqrt(37.25), np.sqrt(16.3125)])
    np.testing.assert_allclose(frame.direction_norm_rmse, [2., 2.])
    assert frame.seed_count.tolist() == [2, 2]
    assert all(json.loads(value) == [0, 1] for value in frame.seed_ids_json)
    assert info["aggregation"] == "joint_no_intercept_least_squares_shared_target_direction"
    assert info["formula_version"] == "corollary3-guidance-fit-1"
    assert info["actual_initial_snr"] == 1.
    assert auxiliary[GUIDANCE_FIT_STEM + "_cohort"].included.all()
    for name, before in original.items():
        pd.testing.assert_frame_equal(tables[name], before)


def test_guidance_fit_keeps_negative_joint_estimates_and_measured_zero_loss():
    tables = four_stage_tables()
    tables["initial_generated_samples"].loc[:1, "cor3_guidance_fit"] = [-8., 2.]
    tables["forward_loss_summary"].loc[0, "loss_mean_squared_l2"] = 0.
    frame, _, _ = reduce_guidance_fit(tables)
    assert frame.iloc[0].y == -3. and frame.iloc[0].x == 0.


@pytest.mark.parametrize("problem", ["missing_seed", "missing_sscd", "degenerate", "unresolved_qa"])
def test_guidance_fit_excludes_entire_invalid_pair_cohort_with_audit(problem):
    tables = four_stage_tables()
    fitted = tables["initial_generated_samples"]
    if problem == "missing_seed":
        tables["initial_generated_samples"] = fitted.drop(index=0)
    elif problem == "missing_sscd":
        tables["initial"].loc[0, "terminal_sscd"] = np.nan
    elif problem == "degenerate":
        fitted.loc[0, "cor3_guidance_fit_applicable"] = False
        fitted.loc[0, "cor3_guidance_fit_status"] = "exact_zero_target_direction"
        fitted.loc[0, "cor3_guidance_fit"] = np.nan
    else:
        fitted.loc[0, "cor3_guidance_fit_qa"] = "numerically_unresolved"
    frame, info, auxiliary = reduce_guidance_fit(tables)
    assert frame.original_index.tolist() == ["1"]
    assert info["excluded_pair_count"] == 1
    assert auxiliary[GUIDANCE_FIT_STEM + "_cohort"].included.tolist() == [False, True]


@pytest.mark.parametrize("problem, message", [
    ("different_mean", "selected mean receipt"),
    ("different_snr", "native noise or latent dimension"),
    ("different_dimension", "native noise or latent dimension"),
    ("different_direction", "direction changes"),
    ("different_sscd", "terminal SSCD changed"),
    ("different_guidance", "configured guidance differs"),
    ("duplicate_seed", "Duplicate guidance-fit"),
    ("wrong_input_law", "different input law"),
])
def test_guidance_fit_rejects_mixed_measurement_contracts(problem, message):
    tables = four_stage_tables()
    fitted = tables["initial_generated_samples"]
    if problem == "different_mean":
        fitted.loc[0, "theory_mean_sha256"] = "another-mean"
    elif problem == "different_snr":
        fitted.loc[0, "snr"] = .5
    elif problem == "different_dimension":
        fitted.loc[0, "latent_dimension"] = 8
    elif problem == "different_direction":
        fitted.loc[0, "cor3_guidance_fit_direction_l2"] = 5.
    elif problem == "different_sscd":
        fitted.loc[0, "terminal_sscd"] = .1
    elif problem == "different_guidance":
        fitted.loc[0, "guidance_scale"] = 4.
    elif problem == "duplicate_seed":
        tables["initial_generated_samples"] = pd.concat([fitted, fitted.iloc[[0]]], ignore_index=True)
    else:
        fitted.loc[0, "input_source"] = "gaussian_probe"
    with pytest.raises(TheoryError, match=message):
        reduce_guidance_fit(tables)


def test_guidance_fit_missing_measurement_is_unavailable_without_scalar_substitution():
    tables = four_stage_tables()
    del tables["initial_generated_samples"]
    frame, info, auxiliary = reduce_guidance_fit(tables)
    assert frame.empty and info["status"] == "unavailable"
    assert "--recompute-experiments" in info["reason"]
    assert auxiliary[GUIDANCE_FIT_STEM + "_cohort"].reason.tolist() == ["missing_saved_guided_fit"]


def test_guidance_fit_is_attached_to_active_paper_inputs_with_selected_mean_receipt():
    frames, metadata = assemble()
    frame = frames[GUIDANCE_FIT_STEM]
    info = metadata["figures"][GUIDANCE_FIT_STEM]
    assert len(frame) == 2 and info["status"] == "available"
    assert info["theory_mean_sha256"] == "fixture-bank-mean"
    assert info["theory_mean_source"] == "declared_finite_bank_mean"
    assert info["formula_version"] == "corollary3-guidance-fit-1"


@pytest.mark.parametrize("condition, expected", [
    ("complete_zero", "not_applicable"),
    ("missing_seed", "unavailable"),
    ("missing_pair", "unavailable"),
    ("roundoff_unresolved", "unavailable"),
])
def test_guidance_fit_exact_zero_status_cannot_hide_missing_or_unresolved_cohorts(condition, expected):
    tables = four_stage_tables()
    fitted = tables["initial_generated_samples"]
    fitted["cor3_guidance_fit_status"] = "exact_zero_target_direction"
    fitted["cor3_guidance_fit_applicable"] = False
    fitted["cor3_guidance_fit_qa"] = "not_applicable"
    fitted["cor3_guidance_fit_direction_l2"] = 0.
    fitted[["cor3_guidance_fit", "cor3_guidance_fit_residual_rmse"]] = np.nan
    if condition == "missing_seed":
        tables["initial_generated_samples"] = fitted.drop(index=0)
    elif condition == "missing_pair":
        tables["initial_generated_samples"] = fitted.loc[fitted.original_index.eq("0")].copy()
    elif condition == "roundoff_unresolved":
        fitted["cor3_guidance_fit_status"] = "arithmetic_unresolved_target_direction"
        fitted["cor3_guidance_fit_qa"] = "numerically_unresolved"
    frame, info, _ = reduce_guidance_fit(tables)
    assert frame.empty and info["status"] == expected



def _variation_fixture():
    tables = four_stage_tables()
    rows = tables["matched_updates"].copy()
    rows["direct_prop5_applicable"] = True
    rows["direct_prop5_variation_rmse"] = 1. + 2. * rows.seed + rows.step_index
    rows["condition_sign_status"] = "unresolved"
    rows["numerical_variation_lower_l2"] = 100.
    rows["numerical_variation_upper_l2"] = 200.
    terminal = rows.loc[rows.step_index.eq(1)].copy()
    terminal["step_index"] = 2
    terminal["direct_prop5_variation_rmse"] = 999.
    return pd.concat([rows, terminal], ignore_index=True), tables["initial"].copy()


def _reduce_variation(rows, initial, *, steps=3):
    from utils.experiments.theory.four_stage_figures import _prompt_variation_inputs
    saved, auxiliary = {}, {}
    def save(stem, frame, **metadata):
        saved[stem] = (frame, metadata)
    _prompt_variation_inputs(rows, initial, four_stage_config(num_inference_steps=steps), save, auxiliary)
    frame, metadata = saved["reference_variation_per_prompt"]
    return frame, metadata, auxiliary["reference_variation_per_prompt_cohort"]


def test_variation_is_saved_in_active_suite_with_original_numerical_statuses():
    frames, metadata = assemble()
    stem = "reference_variation_per_prompt"
    curves = frames[stem]
    assert len(curves) == 2 and curves.step_index.eq(0).all()
    np.testing.assert_allclose(curves.mean_reference_variation_rmse, .5)
    info = metadata["figures"][stem]
    assert info["formula_version"] == "reference-variation-per-prompt-1"
    assert info["prediction_domain"] == "positive_noise_transitions"
    assert info["source_column"] == "direct_prop5_variation_rmse"
    assert info["measurement_audit_table"] == "audit_data/feedback_endpoints.csv"
    assert "not certified" in info["numerical_scope"]
    assert info["numerical_status_counts"]["direct_prop5_integral_status"]["numerically_unresolved"] == 2


def test_variation_means_saved_seed_norm_integrals_once_and_omits_final_prediction():
    rows, initial = _variation_fixture()
    before = rows.copy(deep=True)
    frame, info, audit = _reduce_variation(rows.sample(frac=1, random_state=12), initial)
    for _, curve in frame.groupby("original_index"):
        assert curve.step_index.tolist() == [0, 1]
        np.testing.assert_allclose(curve.mean_reference_variation_rmse, [2., 3.])
        np.testing.assert_allclose(curve.mean_terminal_sscd, .6)
        assert curve.seed_ids_json.eq("[0, 1]").all()
    assert info["prediction_steps"] == 3 and info["excluded_terminal_prediction_rows"] == 4
    assert info["expected_seed_ids"] == [0, 1]
    assert info["counts"]["plotted_pairs"] == 2
    assert audit.eligible.all() and audit.expected_rows.eq(4).all()
    for value in audit.direct_prop5_integral_status_counts_json:
        assert json.loads(value) == {"estimated_converged": 2, "numerically_unresolved": 2}
    assert info["numerical_status_counts"]["condition_sign_status"] == {"unresolved": 8}
    pd.testing.assert_frame_equal(rows, before)


@pytest.mark.parametrize("failure", ["missing_row", "nan", "negative", "infinite", "inapplicable", "failed_integral", "mismatched_sscd"])
def test_variation_excludes_entire_pair_without_changing_seed_cohort(failure):
    rows, initial = _variation_fixture()
    chosen = rows.original_index.eq("0") & rows.seed.eq(1) & rows.step_index.eq(1)
    if failure == "missing_row":
        rows = rows.loc[~chosen].copy()
    elif failure == "inapplicable":
        rows.loc[chosen, "direct_prop5_applicable"] = False
    elif failure == "failed_integral":
        rows.loc[chosen, "direct_prop5_integral_status"] = "not_computed"
    elif failure == "mismatched_sscd":
        rows.loc[chosen, "terminal_sscd"] = .2
    else:
        rows.loc[chosen, "direct_prop5_variation_rmse"] = {"nan": np.nan, "negative": -1., "infinite": np.inf}[failure]
    frame, info, audit = _reduce_variation(rows, initial)
    assert set(frame.original_index) == {"1"} and len(frame) == 2
    np.testing.assert_allclose(frame.mean_reference_variation_rmse, [2., 3.])
    assert info["counts"]["excluded_pairs"] == 1
    assert not audit.loc[audit.original_index.eq("0"), "eligible"].item()


@pytest.mark.parametrize("field", ["direct_prop5_variation_rmse", "direct_prop5_applicable", "direct_prop5_integral_status"])
def test_variation_missing_measurement_field_cannot_be_filled_from_refined_interval(field):
    rows, initial = _variation_fixture()
    frame, info, audit = _reduce_variation(rows.drop(columns=field), initial)
    assert frame.empty and info["status"] == "unavailable"
    assert info["missing_measurement_fields"] == [field]
    assert field in info["reason"] and not audit.eligible.any()


def test_variation_single_prediction_has_no_manuscript_transition():
    rows, initial = _variation_fixture()
    rows = rows.loc[rows.step_index.eq(0)].copy()
    frame, info, audit = _reduce_variation(rows, initial, steps=1)
    assert frame.empty and audit.empty
    assert info["status"] == "not_applicable" and "no such transition" in info["reason"]


def test_variation_preserves_saved_zero_integrals_with_unresolved_condition_sign():
    rows, initial = _variation_fixture()
    rows["direct_prop5_variation_rmse"] = 0.
    frame, info, _ = _reduce_variation(rows, initial)
    assert len(frame) == 4 and frame.mean_reference_variation_rmse.eq(0).all()
    assert info["status"] == "available"


def test_variation_refuses_duplicate_sample_transition_identity():
    rows, initial = _variation_fixture()
    rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    with pytest.raises(TheoryError, match="Duplicate per-prompt"):
        _reduce_variation(rows, initial)
