"""Fixed evidence contracts authored without executing measurements or tests."""
import numpy as np
import pandas as pd
import pytest

from tests.test_theory_direct_figures import direct_config, direct_tables
from utils.experiments.theory.evidence_figures import (
    BOOTSTRAP_POLICY, build_evidence_plot_inputs,
)
from utils.experiments.theory.paper_registry import evidence_registry as paper_registry
from utils.experiments.theory.contracts import TheoryError


def evidence_config(**overrides):
    return direct_config(**overrides)


def evidence_tables():
    """Synthetic scalar fixture shared with publication/CLI tests; no file reads."""
    tables = direct_tables()
    # Preserve the historical S=.7 fixture with an independently saved E=.3.
    # Q=.7 uses its own target-error bound max(ec, eu+tail), with eu=.3.
    for name in ("initial", "trajectory", "matched_updates", "terminal"):
        rows = tables[name]
        rows["direct_branch_gap_error_rmse"] = .3
        rows["direct_branch_gap_error_l2"] = .3 * np.sqrt(rows.latent_dimension)
        rows["direct_unconditional_reference_error_rmse"] = .3
        rows["direct_unconditional_reference_error_l2"] = .3 * np.sqrt(rows.latent_dimension)
    tables["forward_loss_summary"]["KL_target_to_gaussian"] = .08
    gaussian = tables["gaussian_conditional"]
    gaussian["unconditional_target_error_rmse"] = [6., 8., 6., 8.]
    gaussian["gaussian_control_status"] = "same_saved_gaussian_input_canonical_epsilon"
    draws = []
    for row in tables["forward_loss_summary"].to_dict("records"):
        for draw in range(row["draw_count"]):
            draws.append({**row, "draw_index": draw,
                          "loss_squared_l2": row["loss_mean_squared_l2"],
                          "clean_error_squared_l2": row["forward_clean_error_mean_squared_l2"]})
    tables["forward_loss_draws"] = pd.DataFrame(draws)
    tables["reference_analytical"] = pd.DataFrame([
        {"run_id": "run", "seed": seed, "grid_index": index, "snr": snr,
         "reference_mean_error_rmse": .5 * np.sqrt(snr), "reference_scale_rmse": 3.,
         "input_source": "analytical_reference_only", "reference_law_hash": "law"}
        for index, snr in enumerate(np.geomspace(1e-6, 1., 97)) for seed in range(2)
    ])
    tables["reference_atoms"] = pd.DataFrame([
        {"atom_id": "a", "weight": .25, "atom_center_distance_rmse": 1., "reference_law_hash": "law"},
        {"atom_id": "b", "weight": .75, "atom_center_distance_rmse": 4., "reference_law_hash": "law"},
    ])
    initial = tables["initial"]
    initial["evidence_injection_q"] = [-1., 1., 2., 1.]
    initial["evidence_injection_r_perp"] = [2., 0., 1., 0.]
    initial["evidence_injection_relative_error"] = np.sqrt(
        np.square(initial.evidence_injection_q - 1) + np.square(initial.evidence_injection_r_perp))
    initial["evidence_injection_applicable"] = True
    initial["evidence_injection_status"] = "resolved_direction"
    initial["evidence_injection_decomposition_qa"] = "consistent_with_float64_estimate"
    initial["evidence_injection_absolute_error_rmse"] = initial.evidence_injection_relative_error
    initial["evidence_injection_direction_l2"] = 1.
    initial["evidence_injection_conditional_target_error_l2"] = 2.
    initial["evidence_injection_unconditional_mean_error_l2"] = 2.
    initial["direct_cor3_error_mean_cross_inner_product"] = 4. - np.square(initial.evidence_injection_relative_error) / 2
    matched = tables["matched_updates"]
    matched["direct_prop5_measurement_contract"] = "projected-gap-error-1"
    matched["direct_prop5_variation_definition"] = "positive_part_after_integrated_unit_gap_projection"
    matched["direct_prop5_zero_gap_convention"] = "V=0_when_Delta_is_exactly_zero"
    matched["direct_prop5_matched_log_odds"] = -2.
    matched["direct_prop5_guided_log_odds"] = matched.direct_prop5_log_probability_gain - 2.
    matched["direct_prop5_zero_displacement"] = False
    matched["direct_lemma4_numeric_within_sensitivity"] = True
    matched["refinement_applicable"] = matched.direct_prop5_applicable
    matched["fixed_cache_gain_sign"] = np.where(matched.refinement_applicable, matched.direct_prop5_gain_status, "not_applicable")
    matched["condition_sign_status"] = np.where(matched.refinement_applicable, np.where(matched.seed.eq(0), "negative", "unresolved"), "not_applicable")
    matched["source_robust_gain_sign"] = np.where(matched.refinement_applicable, "unresolved", "not_applicable")
    matched["condition_value_status"] = "estimated_existing_variation"
    matched["input_contract_status"] = "original_saved_inputs_available"
    matched["endpoint_construction_method"] = "saved_and_independent_affine_endpoint"
    matched["arithmetic_status"] = "certified_original_saved_inputs"
    matched["condition_arithmetic_status"] = "certified_original_saved_inputs"
    matched["arithmetic_error_method"] = "fixture_declared_interval"
    matched["quadrature_status"] = "existing_embedded_estimate"
    matched["quadrature_error_scope"] = "numerical_estimate_not_certificate"
    matched["source_sensitivity_status"] = "heuristic_only_no_justified_endpoint_radii"
    matched["source_sensitivity_model"] = "separate_dtype_sensitivity"
    matched["numerical_original_margin_rmse"] = matched.direct_prop5_margin_rmse
    matched["numerical_log_probability_gain"] = matched.direct_prop5_log_probability_gain
    matched["numerical_log_odds_gain"] = matched.direct_prop5_log_probability_gain
    matched["numerical_affine_log_probability_gain"] = matched.direct_prop5_log_probability_gain
    matched["numerical_matched_log_odds"] = matched.direct_prop5_matched_log_odds
    matched["numerical_saved_log_odds"] = matched.direct_prop5_guided_log_odds
    matched["numerical_publication_blocker"] = False
    matched["implication_eligible"] = False
    matched["implication_audit_status"] = "unavailable_positive_margin_or_endpoint_transfer_not_resolved"
    matched["numerical_stopping_reason"] = "fixed_fixture_budget"
    terminal = tables["terminal"]
    terminal["direct_theorem7_endpoint_error_rmse"] = [0., 1., 2., 3.]
    terminal["direct_theorem7_endpoint_error_l2"] = terminal.direct_theorem7_endpoint_error_rmse * 2
    terminal["evidence_terminal_corrected_obs_bound_rmse"] = [0., 2., 4., 6.]
    terminal["evidence_terminal_corrected_ref_bound_rmse"] = [0., 3., 6., 9.]
    terminal["evidence_terminal_conditional_error_rmse"] = [0., .5, 1., 1.5]
    terminal["evidence_terminal_residual_guidance_rmse"] = [0., .5, 1., 1.5]
    terminal["evidence_terminal_defect_bound_rmse"] = [0., 1., 2., 3.]
    terminal["evidence_terminal_scope"] = "finite_terminal_update_extension_deterministic"
    terminal["evidence_terminal_status"] = "available"
    terminal["evidence_terminal_applicable"] = True
    terminal["evidence_terminal_reconstruction_status"] = "consistent_with_source_precision"
    terminal["evidence_terminal_clean_bound_order_status"] = "consistent_with_float64_estimate"
    terminal["evidence_terminal_noise_bound_exceeded"] = False
    # g=2: exact scalar identities accompany the deliberately loose bounds.
    terminal["evidence_terminal_B_obs_rmse"] = terminal.evidence_terminal_corrected_obs_bound_rmse - terminal.evidence_terminal_defect_bound_rmse
    terminal["evidence_terminal_B_ref_rmse"] = terminal.evidence_terminal_corrected_ref_bound_rmse - terminal.evidence_terminal_defect_bound_rmse
    terminal["direct_conditional_error_rmse"] = terminal.evidence_terminal_conditional_error_rmse
    terminal["direct_unconditional_reference_error_rmse"] = terminal.direct_conditional_error_rmse
    terminal["direct_branch_gap_error_rmse"] = 2 * terminal.direct_conditional_error_rmse
    terminal["direct_radius_tail_rmse"] = terminal.direct_conditional_error_rmse
    terminal["direct_reference_target_error_rmse"] = terminal.direct_conditional_error_rmse
    terminal["direct_lemma6_gap_rmse"] = terminal.evidence_terminal_residual_guidance_rmse
    terminal["direct_lemma6_rhs_rmse"] = terminal.direct_conditional_error_rmse * 3
    for stem in ("direct_conditional_error", "direct_unconditional_reference_error", "direct_branch_gap_error", "direct_radius_tail",
                 "direct_reference_target_error", "direct_lemma6_gap", "direct_lemma6_rhs", "evidence_terminal_B_obs", "evidence_terminal_B_ref"):
        terminal[stem + "_l2"] = terminal[stem + "_rmse"] * 2
    terminal["evidence_terminal_reconstruction_tolerance_rmse"] = 0.
    terminal["direct_theorem7_numeric_tolerance_rmse"] = 0.
    return tables


def assemble(tables=None, **options):
    return build_evidence_plot_inputs(tables or evidence_tables(), evidence_config(**options), {"reference_law_hash": "law"})


def test_fixed_seven_primary_and_twelve_appendix_contracts():
    entries = paper_registry()
    assert [entry["result_number"] for entry in entries if entry["category"] == "main"] == list(range(1, 8))
    assert [entry["stem"] for entry in entries if entry["category"] == "appendix"] == [
        "theorem1_initial_branch_comparison", "theorem1_gaussian_transfer", "lemma2_initial_concentration", "lemma2_native_gaussian_sweep",
        "corollary3_full_guided_error", "proposition5_condition_vs_gain", "proposition5_first_update_comparison", "proposition5_numerical_resolution",
        "lemma6_branch_target_errors", "lemma6_bound_vs_gap", "lemma6_posterior_concentration", "theorem7_terminal_components",
    ]
    optional = {entry["stem"]: entry for entry in paper_registry(True)}
    assert optional["theorem7_uncorrected_bound_audit"]["category"] == "diagnostics"
    assert optional["theorem1_forward_identity"]["category"] == "diagnostics"


def test_pair_rms_uses_squared_seed_errors_and_independent_draw_bootstrap():
    frames, metadata = assemble()
    frame = frames["theorem1_loss_recovery"]
    assert len(frame) == 2
    np.testing.assert_allclose(frame.y, np.sqrt(10.25))
    np.testing.assert_allclose(frame.control_y, np.sqrt(50.))
    np.testing.assert_allclose(frame.mean_terminal_sscd, .6)
    np.testing.assert_allclose(frame.x_low, frame.x)
    np.testing.assert_allclose(frame.x_high, frame.x)
    assert frame.y_low.lt(frame.y_high).all()
    assert metadata["figures"]["theorem1_loss_recovery"]["bootstrap_policy"] == BOOTSTRAP_POLICY
    assert len(metadata["auxiliary_tables"]["theorem1_gaussian_seed_tails"]) == 4
    centered = metadata["auxiliary_tables"]["theorem1_within_prompt_associations"]
    np.testing.assert_allclose(centered.groupby("record_id").x_centered.sum(), 0)


def test_loss_intervals_require_genuine_draws_and_control_never_uses_target_proxy():
    tables = evidence_tables()
    tables["forward_loss_draws"] = tables["forward_loss_draws"].iloc[1:]
    with pytest.raises(TheoryError, match="every saved independent"):
        assemble(tables)
    tables = evidence_tables()
    tables["gaussian_conditional"]["unconditional_target_error_rmse"] = np.nan
    frames, metadata = assemble(tables)
    assert frames["theorem1_loss_recovery"].y.isna().all()
    assert metadata["figures"]["theorem1_loss_recovery"]["status"] == "unavailable"


def test_reference_native_boundary_and_atom_mean_population_remain_separate():
    frames, metadata = assemble()
    reference = frames["lemma2_unconditional_baseline"]
    low = reference.loc[reference.source_range.eq("analytical")]
    assert len(low) == 97 and set(low.metric) == {"reference"}
    native = reference.loc[reference.source_range.eq("native_initial")]
    assert native.snr.min() == 1. and set(native.metric) == {"reference", "learned", "reference_error"}
    assert metadata["figures"]["lemma2_unconditional_baseline"]["reference_scale_rmse"] == 3.
    ecdf = frames["lemma2_initial_concentration"]
    atoms = ecdf.loc[ecdf.distribution.eq("candidate_atoms")]
    assert atoms.cdf.tolist() == [.25, 1.]


def test_geometry_retains_negative_and_large_components_and_excluded_rows():
    tables = evidence_tables()
    tables["initial"].loc[0, "evidence_injection_applicable"] = False
    tables["initial"].loc[0, "evidence_injection_status"] = "zero_target_direction"
    frames, metadata = assemble(tables)
    frame = frames["corollary3_initial_cfg_amplification"]
    assert len(frame) == 4 and frame.x.isna().sum() == 1
    assert frame.x.max() == 2
    excluded = metadata["auxiliary_tables"]["injection_excluded_directions"]
    assert excluded.evidence_injection_absolute_error_rmse.iloc[0] > 0


def test_feedback_strict_fractions_keep_common_weights_and_unresolved_denominator():
    frames, metadata = assemble()
    frame = frames["proposition5_posterior_feedback"]
    paired = frame.pivot(index=["group", "step_index"], columns="metric", values="denominator_weight")
    np.testing.assert_array_equal(paired.feedback, paired.condition)
    high = frame.loc[frame.group.eq("SSCD > 0.75")]
    assert high.loc[high.metric.eq("feedback"), "fraction"].iloc[0] == 1
    assert high.loc[high.metric.eq("condition"), "fraction"].iloc[0] == 0
    assert high.loc[high.metric.eq("condition"), "upper_fraction"].iloc[0] == 1
    assert metadata["figures"]["proposition5_posterior_feedback"]["condition_zero_overlap"]



def test_float64_condition_estimates_are_displayed_without_interval_certificates():
    tables = evidence_tables()
    rows = tables["matched_updates"]
    selected = rows.refinement_applicable
    rows.loc[selected, "condition_sign_status"] = "positive"
    rows.loc[selected, "condition_arithmetic_status"] = "float64_numerical_assessment"
    rows.loc[selected, "numerical_original_margin_rmse"] = 1.
    rows.loc[selected, "fixed_cache_gain_sign"] = "negative"
    rows.loc[selected, "numerical_log_probability_gain"] = -.1
    rows.loc[selected, "implication_eligible"] = False
    frames, metadata = assemble(tables)
    condition = frames["proposition5_posterior_feedback"].query("metric == 'condition'")
    assert condition.fraction.eq(1).all() and condition.upper_fraction.eq(1).all()
    assert condition.unresolved_count.eq(0).all()
    assert frames["proposition5_condition_vs_gain"].marker_class.eq("observed").all()
    info = metadata["figures"]["proposition5_posterior_feedback"]
    assert info["condition_assessment_counts"] == {
        "float64_estimated_sign": int(selected.sum()), "enclosed_sign": 0,
        "other_resolved_sign": 0, "unresolved_or_unavailable": 0,
    }
    assert "numerical estimates" in info["condition_interpretation"]
    assert not metadata["audit"]["blocking"]
    audit = metadata["auxiliary_tables"]["proposition5_numerical_resolution_rows"]
    assert not audit.loc[selected, "implication_eligible"].any()
    # A real identity/enclosure failure must still block estimated displays.
    rows.loc[selected, "numerical_publication_blocker"] = True
    _, blocked = assemble(tables)
    assert blocked["figures"]["proposition5_posterior_feedback"]["status"] == "blocked"


def test_near_zero_float64_assessment_retains_unresolved_mass_and_marker():
    tables = evidence_tables()
    rows = tables["matched_updates"]
    selected = rows.refinement_applicable
    uncertain = selected & rows.seed.eq(1)
    rows.loc[selected, "condition_arithmetic_status"] = "float64_numerical_assessment"
    rows.loc[uncertain, "condition_sign_status"] = "unresolved"
    rows.loc[uncertain, "numerical_original_margin_rmse"] = 1e-14
    rows.loc[uncertain, "numerical_condition_estimate_status"] = "within_numerical_uncertainty"
    rows.loc[uncertain, "numerical_condition_estimate_uncertainty_l2"] = 1e-12
    frames, metadata = assemble(tables)
    condition = frames["proposition5_posterior_feedback"].query("metric == 'condition' and group == 'SSCD > 0.75'")
    assert condition.fraction.eq(0).all() and condition.upper_fraction.eq(1).all()
    assert condition.unknown_fraction.eq(1).all()
    scatter = frames["proposition5_condition_vs_gain"]
    assert scatter.loc[scatter.seed.eq(1), "marker_class"].eq("numerically_unresolved").all()
    assert scatter.loc[scatter.seed.eq(0), "marker_class"].eq("observed").all()
    counts = metadata["figures"]["proposition5_posterior_feedback"]["condition_assessment_counts"]
    assert counts["unresolved_or_unavailable"] == int(uncertain.sum())
    assert counts["float64_estimated_sign"] == int((selected & ~uncertain).sum())
    assert not metadata["audit"]["blocking"]


def test_reliable_original_condition_gain_contradiction_blocks_feedback():
    tables = evidence_tables()
    selected = tables["matched_updates"].direct_prop5_applicable
    tables["matched_updates"].loc[selected, "condition_sign_status"] = "positive"
    tables["matched_updates"].loc[selected, "fixed_cache_gain_sign"] = "negative"
    tables["matched_updates"].loc[selected, "implication_eligible"] = True
    _, metadata = assemble(tables)
    assert metadata["audit"]["blocking"]
    assert metadata["figures"]["proposition5_posterior_feedback"]["status"] == "blocked"


def _set_trajectory_branch_norms(rows, *, conditional, unconditional, error, gap, reference_gap=0., tail=0.):
    """Consistent scalar designs below use target=reference=0 and collinear vectors."""
    fields = {
        "direct_conditional_error": conditional,
        "direct_unconditional_target_error": unconditional,
        "direct_unconditional_reference_error": unconditional,
        "direct_branch_gap_error": error,
        "direct_lemma6_gap": gap,
        "direct_reference_target_error": reference_gap,
        "direct_radius_tail": tail,
        "direct_lemma6_rhs": error + tail,
    }
    for stem, value in fields.items():
        rows[stem + "_rmse"] = value
        rows[stem + "_l2"] = value * np.sqrt(rows.latent_dimension)


def test_common_mode_wrong_target_agreement_keeps_Q_above_S_without_blocking():
    tables = evidence_tables()
    rows = tables["trajectory"]
    # Both learned branches coincide away from the target/reference: their
    # vector errors cancel exactly although neither branch recovers the target.
    _set_trajectory_branch_norms(rows, conditional=4., unconditional=4., error=0., gap=0.)
    frames, metadata = assemble(tables)
    frame = frames["lemma6_target_specific_synchronization"]
    assert frame.loc[frame.metric.eq("joint_error"), "median"].eq(4).all()
    assert frame.loc[frame.metric.isin(["gap", "bound"]), "median"].eq(0).all()
    assert set(frame.step_index) == {0, 1}
    info = metadata["figures"]["lemma6_target_specific_synchronization"]
    assert info["status"] == "available" and not metadata["audit"]["blocking"]
    assert info["row_bound_audit"]["joint_above_bound_count"] == len(rows)
    assert info["row_bound_audit"]["joint_above_own_bound_count"] == 0
    audit = metadata["auxiliary_tables"]["lemma6_joint_sample_audit"]
    assert audit.joint_target_bound_rmse.eq(4.).all()
    assert audit.joint_target_bound_slack_rmse.eq(0.).all()
    assert audit.branch_gap_error_rhs_identity_residual_rmse.eq(0.).all()


def test_nonzero_collinear_branch_errors_use_signed_projection_not_norm_sum():
    tables = evidence_tables()
    rows = tables["trajectory"]
    # In d=4, learned vectors (6,0,0,0) and (4,0,0,0), with target and
    # posterior reference at zero, give E/sqrt(d)=D/sqrt(d)=1 < 3+2.
    _set_trajectory_branch_norms(rows, conditional=3., unconditional=2., error=1., gap=1.)
    frames, metadata = assemble(tables)
    frame = frames["lemma6_target_specific_synchronization"]
    assert frame.loc[frame.metric.eq("joint_error"), "median"].eq(3.).all()
    assert frame.loc[frame.metric.isin(["gap", "bound"]), "median"].eq(1.).all()
    assert not metadata["audit"]["blocking"]
    assert metadata["figures"]["lemma6_target_specific_synchronization"]["measurement_contract"] == "projected-gap-error-1"
    rows.loc[0, "direct_branch_gap_error_rmse"] = .5
    _, failed = assemble(tables)
    assert failed["figures"]["lemma6_target_specific_synchronization"]["status"] == "blocked"
    assert len(failed["auxiliary_tables"]["lemma6_failed_observations"]) == 1


@pytest.mark.parametrize("table,column", [("trajectory", "direct_branch_gap_error_rmse"),
                                             ("terminal", "direct_branch_gap_error_l2")])
def test_signed_error_cannot_be_reconstructed_from_old_individual_norms(table, column):
    tables = evidence_tables()
    tables[table] = tables[table].drop(columns=column)
    with pytest.raises(TheoryError, match=column):
        assemble(tables)


def test_terminal_exact_ecdf_zero_mass_and_ordering_preserve_clean_gate():
    frames, metadata = assemble(target_error_tolerance=4.)
    frame = frames["theorem7_final_reproduction"]
    info = metadata["figures"]["theorem7_final_reproduction"]
    assert info["original_clean_counts"] == {"applicable": 0, "total": 4}
    assert info["manuscript_extension_required"]
    assert info["zero_mass"] == {"actual": .25, "observable": .25, "reference": .25}
    assert info["predeclared_tolerance_rmse"] == 2.
    supports = sorted(frame.value.unique())
    def at(distribution, tolerance):
        values = frame.loc[frame.distribution.eq(distribution) & frame.value.le(tolerance), "cdf"]
        return values.iloc[-1] if len(values) else 0.
    for tolerance in supports:
        assert at("reference", tolerance) <= at("observable", tolerance) <= at("actual", tolerance)


def test_noise_event_is_reported_while_reconstruction_failure_blocks():
    tables = evidence_tables()
    tables["terminal"]["evidence_terminal_noise_bound_exceeded"] = True
    tables["terminal"]["evidence_terminal_scope"] = "finite_terminal_update_extension_gaussian_noise_bound"
    _, metadata = assemble(tables)
    assert not metadata["audit"]["blocking"]
    assert metadata["figures"]["theorem7_final_reproduction"]["noise_bound_exceedance_count"] == 4
    tables["terminal"].loc[0, "evidence_terminal_reconstruction_status"] = "failed_reconstruction"
    _, metadata = assemble(tables)
    assert metadata["audit"]["blocking"]
    assert metadata["figures"]["theorem7_final_reproduction"]["status"] == "blocked"


def test_every_registry_input_and_caption_contract_is_saved():
    frames, metadata = assemble()
    for entry in paper_registry(True):
        stem = entry["stem"]
        assert stem in metadata["figures"]
        if metadata["figures"][stem]["status"] == "available":
            assert set(entry["required_columns"]) <= set(frames[stem])


def test_finite_identity_failures_block_geometry_and_terminal_bounds():
    tables = evidence_tables()
    tables["initial"].loc[1, "evidence_injection_decomposition_qa"] = "numerically_unresolved"
    _, metadata = assemble(tables)
    assert metadata["figures"]["corollary3_initial_cfg_amplification"]["status"] == "blocked"
    tables = evidence_tables()
    tables["terminal"].loc[0, "evidence_terminal_clean_bound_order_status"] = "failed_clean_bound_order"
    _, metadata = assemble(tables)
    assert metadata["figures"]["theorem7_final_reproduction"]["status"] == "blocked"


def test_bootstrap_is_invariant_to_saved_draw_and_seed_row_order():
    tables = evidence_tables()
    baseline, _ = assemble(tables)
    tables["forward_loss_draws"] = tables["forward_loss_draws"].iloc[::-1]
    tables["gaussian_conditional"] = tables["gaussian_conditional"].iloc[::-1]
    reordered, _ = assemble(tables)
    columns = ["x", "y", "control_y", "x_low", "x_high", "y_low", "y_high"]
    left = baseline["theorem1_loss_recovery"].sort_values("record_id")[columns]
    right = reordered["theorem1_loss_recovery"].sort_values("record_id")[columns]
    np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())


def test_failed_independent_matched_identity_blocks_but_construction_is_distinct():
    tables = evidence_tables()
    tables["matched_updates"]["direct_lemma4_numeric_within_sensitivity"] = False
    _, metadata = assemble(tables)
    info = metadata["figures"]["lemma4_matched_displacement"]
    assert info["status"] == "blocked"
    assert info["failed_independent_consistency_count"] == 4
    tables["matched_updates"]["direct_lemma4_independent"] = False
    _, metadata = assemble(tables)
    assert metadata["figures"]["lemma4_matched_displacement"]["status"] == "available"


def test_complete_transfer_audit_preserves_unclipped_tv_and_all_uncertainty():
    tables = evidence_tables()
    for name in ("forward_loss_summary", "forward_loss_draws"):
        tables[name]["KL_target_to_gaussian"] = 8.
        tables[name]["pinsker_tv_bound"] = 1.
    frames, metadata = assemble(tables)
    audit = metadata["auxiliary_tables"]["theorem1_gaussian_transfer_audit"]
    assert len(audit) == 10
    assert audit.KL_target_to_gaussian.eq(8).all()
    assert audit.pinsker_tv_bound_unclipped.eq(2).all()
    assert audit.pinsker_tv_bound.eq(1).all()
    assert {"rho", "raw_tolerance_l2", "bound_unclipped", "bound_clipped",
            "loss_mc_se_squared_l2", "bound_mc_se", "frequency_low", "frequency_high",
            "failure_count", "gaussian_count", "missing_gaussian_count", "loss_draw_count"} <= set(audit)
    assert audit.bound_unclipped.gt(1).all()
    assert audit.bound_clipped.eq(1).all()
    # Plot compaction may discard audit-only columns, but must retain the
    # intervals that the saved-only renderer actually draws.
    assert {"frequency_low", "frequency_high"} <= set(frames["theorem1_gaussian_transfer"])
    assert metadata["figures"]["theorem1_gaussian_transfer"]["complete_transfer_audit_table"] == "audit_data/theorem1_gaussian_transfer_audit.csv"


def test_transfer_audit_requires_true_kl_instead_of_inverting_clipped_tv():
    tables = evidence_tables()
    tables["forward_loss_summary"] = tables["forward_loss_summary"].drop(columns="KL_target_to_gaussian")
    with pytest.raises(TheoryError, match="KL_target_to_gaussian"):
        assemble(tables)


def test_initial_reference_markers_do_not_hide_complete_native_sweep():
    frames, metadata = assemble()
    main = frames["lemma2_unconditional_baseline"]
    native = frames["lemma2_native_gaussian_sweep"]
    markers = main.loc[main.source_range.eq("native_initial")]
    assert len(markers) == 3 and markers.step_index.eq(0).all()
    assert markers.snr.eq(1).all()
    assert set(main.source_range) == {"analytical", "native_initial"}
    assert set(native.step_index) == {0, 1}
    assert set(native.metric) == {"reference", "learned", "reference_error"}
    assert native.source_range.eq("native").all()
    info = metadata["figures"]["lemma2_unconditional_baseline"]
    assert info["native_sweep_figure"].endswith("lemma2_native_gaussian_sweep.png")
    assert "not selected" in info["display_window_definition"]


def test_geometry_saves_sample_pair_contour_and_inner_product_audits():
    _, metadata = assemble()
    info = metadata["figures"]["corollary3_initial_cfg_amplification"]
    assert info["sample_relative_error_quantiles"]["count"] == 4
    assert info["pair_relative_error_quantiles"]["count"] == 2
    audit = metadata["auxiliary_tables"]["corollary3_premise_error_decomposition"]
    np.testing.assert_allclose(audit.error_decomposition_residual, 0, atol=1e-14)
    assert audit.premise_slack.ge(0).all()
    coverage = metadata["auxiliary_tables"]["corollary3_contour_coverage"]
    assert set(coverage.level) == {.25, .5, 1.}
    assert coverage.loc[coverage.population.eq("sample"), "fraction"].eq(.5).all()
    assert "not a rigorous" in info["injection_premise"]["arithmetic_error_scope"]
    tables = evidence_tables()
    tables["initial"].loc[0, "direct_cor3_error_mean_cross_inner_product"] += 1
    _, broken = assemble(tables)
    assert broken["figures"]["corollary3_initial_cfg_amplification"]["status"] == "blocked"


def test_negative_precheck_stays_resolved_without_variation_point_value():
    tables = evidence_tables()
    rows = tables["matched_updates"]
    selected = rows.refinement_applicable
    rows.loc[selected, "condition_sign_status"] = "negative"
    rows.loc[selected, "condition_value_status"] = "unavailable_variation_not_computed"
    rows.loc[selected, "numerical_original_margin_rmse"] = np.nan
    rows["numerical_variation_l2"] = np.nan
    rows.loc[selected, "quadrature_status"] = "not_computed"
    frames, metadata = assemble(tables)
    conditions = frames["proposition5_posterior_feedback"].query("metric == 'condition'")
    assert conditions.fraction.eq(0).all() and conditions.upper_fraction.eq(0).all()
    assert conditions.unresolved_count.eq(0).all()
    assert conditions.condition_without_point_value_count.eq(2).all()
    audit = metadata["auxiliary_tables"]["proposition5_numerical_resolution_rows"]
    assert audit.loc[selected, "numerical_original_margin_rmse"].isna().all()
    assert not metadata["audit"]["blocking"]


def test_resolution_layers_preserve_common_denominator_and_complete_causes():
    frames, metadata = assemble()
    frame = frames["proposition5_numerical_resolution"]
    assert set(frame.metric) == {"feedback", "condition", "source_robust_feedback"}
    denominator = frame.pivot(index=["group", "step_index"], columns="metric", values="denominator_weight")
    np.testing.assert_array_equal(denominator.feedback, denominator.condition)
    np.testing.assert_array_equal(denominator.feedback, denominator.source_robust_feedback)
    robustness = frame.loc[frame.metric.eq("source_robust_feedback")]
    assert robustness.resolved_fraction.eq(0).all() and robustness.unknown_fraction.eq(1).all()
    causes = metadata["auxiliary_tables"]["proposition5_numerical_cause_counts"]
    assert {"arithmetic_status", "condition_arithmetic_status", "condition_value_status",
            "endpoint_construction_method", "quadrature_status", "source_sensitivity_status",
            "numerical_stopping_reason"} <= set(causes.field)
    assert "Structurally excluded" in set(causes.group)
    assert causes["count"].ge(0).all() and causes.weight.ge(0).all()
    info = metadata["figures"]["proposition5_posterior_feedback"]
    assert info["condition_unresolved_counts"]["SSCD > 0.75"] == 2
    assert info["observed_condition_coverage"]["SSCD > 0.75"]["maximum"] == 0
    assert info["condition_unresolved_fractions"]["SSCD > 0.75"]["maximum"] == 1


def test_feedback_uses_stable_sign_with_raw_H_underflow_and_endpoint_contract_gate():
    tables = evidence_tables()
    rows = tables["matched_updates"]
    selected = rows.refinement_applicable & rows.seed.eq(1)
    rows.loc[selected, "numerical_log_probability_gain"] = 0.
    rows.loc[selected, "numerical_gain_underflow"] = True
    frames, metadata = assemble(tables)
    high = frames["proposition5_posterior_feedback"].query("group == 'SSCD > 0.75' and metric == 'feedback'")
    assert high.fraction.eq(1).all() and high.gain_underflow_count.eq(2).all()
    assert not metadata["audit"]["blocking"]
    rows.loc[selected, "condition_sign_status"] = "positive"
    rows.loc[selected, "fixed_cache_gain_sign"] = "negative"
    rows.loc[selected, "implication_eligible"] = False
    _, untransferred = assemble(tables)
    assert not untransferred["audit"]["blocking"]
    rows.loc[selected, "implication_eligible"] = True
    _, contradicted = assemble(tables)
    assert contradicted["figures"]["proposition5_posterior_feedback"]["status"] == "blocked"



@pytest.mark.parametrize("field, value", [
    ("direct_prop5_measurement_contract", "branch-gap-error-substitution-1"),
    ("direct_prop5_variation_definition", "integral_of_reference_difference_norm"),
])
def test_feedback_refuses_historical_norm_variation_receipts(field, value):
    tables = evidence_tables()
    rows = tables["matched_updates"]
    rows.loc[rows.refinement_applicable, field] = value
    frames, metadata = assemble(tables)
    for stem in ("proposition5_posterior_feedback", "proposition5_condition_vs_gain"):
        assert frames[stem].empty
        info = metadata["figures"][stem]
        assert info["status"] == "unavailable"
        assert info["incompatible_variation_contract_count"] == int(rows.refinement_applicable.sum())
        assert "--recompute-experiments" in info["reason"]
    audit = metadata["auxiliary_tables"]["proposition5_numerical_resolution_rows"]
    assert audit.loc[rows.refinement_applicable, field].eq(value).all()


def test_missing_refinement_has_explicit_status_and_never_reuses_legacy_signs():
    tables = evidence_tables()
    tables["matched_updates"] = tables["matched_updates"].drop(columns="fixed_cache_gain_sign")
    _, metadata = assemble(tables)
    for stem in ("proposition5_posterior_feedback", "proposition5_numerical_resolution"):
        assert metadata["figures"][stem]["status"] == "unavailable"
        assert "fixed_cache_gain_sign" in metadata["figures"][stem]["missing_refinement_fields"]
    assert "direct_prop5_gain_status" in metadata["auxiliary_tables"]["proposition5_numerical_resolution_rows"]


def test_joint_bound_audit_and_fixed_absolute_coverage_precede_aggregation():
    _, metadata = assemble()
    coverage = metadata["auxiliary_tables"]["lemma6_absolute_tolerance_coverage"]
    assert set(coverage.tolerance_rmse) == {.1, .2, .3, .5, 1.}
    assert set(coverage.metric) == {"joint_error", "bound"}
    assert coverage.loc[coverage.tolerance_rmse.eq(.5), "fraction"].eq(0).all()
    assert coverage.loc[coverage.tolerance_rmse.eq(1), "fraction"].eq(1).all()
    paired = coverage.pivot(index=["group", "step_index", "tolerance_rmse"], columns="metric", values="denominator_weight")
    np.testing.assert_array_equal(paired.joint_error, paired.bound)
    tables = evidence_tables()
    tables["trajectory"].loc[0, "direct_unconditional_target_error_rmse"] = 9.
    _, failed = assemble(tables)
    assert failed["figures"]["lemma6_target_specific_synchronization"]["status"] == "blocked"
    assert len(failed["auxiliary_tables"]["lemma6_failed_observations"]) == 1


def test_terminal_looseness_uses_exact_paired_cost_and_shared_correction():
    _, metadata = assemble()
    audit = metadata["auxiliary_tables"]["theorem7_looseness_decomposition"]
    np.testing.assert_allclose(audit.bound_looseness_l2, audit.lemma6_slack_cost_l2)
    np.testing.assert_allclose(audit.bound_looseness_l2, audit.decomposed_slack_cost_l2)
    np.testing.assert_allclose(audit.scheduler_cancellation_residual_rmse, 0)
    assert audit.slack_radius_l2.ge(0).all() and audit.slack_projection_alignment_l2.ge(0).all()
    info = metadata["figures"]["theorem7_terminal_components"]
    assert info["looseness_statistics"]["bound_looseness"]["count"] == 4
    tables = evidence_tables()
    tables["terminal"].loc[1, "direct_reference_target_error_l2"] = 100.
    _, failed = assemble(tables)
    assert failed["figures"]["theorem7_final_reproduction"]["status"] == "blocked"


def test_terminal_cancelling_branch_errors_tighten_reference_bound_without_changing_correction():
    tables = evidence_tables()
    rows = tables["terminal"]
    _set_trajectory_branch_norms(rows, conditional=3., unconditional=2., error=1., gap=1.)
    for stem in ("evidence_terminal_B_obs", "evidence_terminal_B_ref"):
        rows[stem + "_rmse"] = 4.
        rows[stem + "_l2"] = 8.
    rows["evidence_terminal_conditional_error_rmse"] = 3.
    rows["evidence_terminal_residual_guidance_rmse"] = 1.
    rows["evidence_terminal_corrected_obs_bound_rmse"] = 4. + rows.evidence_terminal_defect_bound_rmse
    rows["evidence_terminal_corrected_ref_bound_rmse"] = 4. + rows.evidence_terminal_defect_bound_rmse
    _, metadata = assemble(tables)
    audit = metadata["auxiliary_tables"]["theorem7_looseness_decomposition"]
    assert not metadata["audit"]["blocking"]
    np.testing.assert_allclose(audit.bound_looseness_l2, 0.)
    np.testing.assert_allclose(audit.slack_projection_alignment_l2, 0.)
    np.testing.assert_allclose(audit.reference_formula_residual_l2, 0.)
    np.testing.assert_allclose(audit.scheduler_cancellation_residual_rmse, 0.)
    assert audit.evidence_terminal_B_ref_rmse.eq(4.).all()
    assert (audit.evidence_terminal_B_ref_rmse < audit.direct_conditional_error_rmse +
            audit.direct_conditional_error_rmse + audit.direct_unconditional_reference_error_rmse).all()


def test_terminal_probabilistic_failure_does_not_become_numerical_bug():
    tables = evidence_tables()
    terminal = tables["terminal"]
    terminal.loc[0, "direct_theorem7_endpoint_error_rmse"] = 5.
    terminal.loc[0, "evidence_terminal_scope"] = "finite_terminal_update_extension_gaussian_noise_bound"
    terminal.loc[0, "evidence_terminal_noise_bound_exceeded"] = True
    _, metadata = assemble(tables)
    assert not metadata["audit"]["blocking"]
    info = metadata["figures"]["theorem7_final_reproduction"]
    assert info["paired_ordering_audit"]["permitted_gaussian_noise_event_count"] == 1
    terminal.loc[0, "evidence_terminal_scope"] = "finite_terminal_update_extension_deterministic"
    _, deterministic = assemble(tables)
    assert deterministic["figures"]["theorem7_final_reproduction"]["status"] == "blocked"


def test_primary_neutral_axes_and_lemma4_identity_has_no_outcome_color():
    entries = {entry["stem"]: entry for entry in paper_registry()}
    assert entries["theorem1_loss_recovery"]["axes"]["y"] == "Initial target error (RMS)"
    assert entries["lemma6_target_specific_synchronization"]["axes"]["y"] == "Target error, gap, or bound (RMSE)"
    assert entries["theorem7_final_reproduction"]["axes"]["y"] == "Coverage of latent-error tolerance"
    assert entries["lemma4_matched_displacement"]["sscd"] is False
    assert "direct_lemma4_verification_source" in entries["lemma4_matched_displacement"]["required_columns"]


def test_legacy_alleged_feedback_contradictions_are_retained_not_inherited_as_certificates():
    tables = evidence_tables()
    rows = tables["matched_updates"]
    old_eligible = rows.direct_prop5_applicable
    rows.loc[old_eligible, "direct_prop5_margin_rmse"] = 1.
    rows.loc[old_eligible, "direct_prop5_log_probability_gain"] = -.1
    rows.loc[old_eligible, "direct_prop5_condition_status"] = "estimated_positive"
    rows.loc[old_eligible, "direct_prop5_gain_status"] = "negative"
    frames, metadata = assemble(tables)
    assert not metadata["audit"]["blocking"]
    assert metadata["figures"]["proposition5_posterior_feedback"]["status"] == "available"
    legacy = metadata["auxiliary_tables"]["proposition5_legacy_condition_gain"]
    assert len(legacy) == 4 and legacy.x.eq(1).all() and legacy.y.eq(-.1).all()
    summary = metadata["auxiliary_tables"]["statement_summary"]
    old_summary = summary.loc[summary.figure.eq("proposition5_posterior_feedback")]
    assert old_summary.alleged_strict_implication_violations.sum() == 4
    high = frames["proposition5_posterior_feedback"].query("group == 'SSCD > 0.75' and metric == 'feedback'")
    assert high.fraction.eq(1).all()  # New saved-endpoint sign, not old overwritten classification.
    assert metadata["audit"]["proposition5_interpretation_migration"]["legacy_figure_status"] == "available"
    rows.loc[old_eligible, "numerical_publication_blocker"] = True
    _, checked = assemble(tables)
    assert checked["audit"]["blocking"]
    assert checked["figures"]["proposition5_posterior_feedback"]["status"] == "blocked"


def test_precision_only_precheck_overrides_legacy_missing_V_availability_not_evidence():
    tables = evidence_tables()
    rows = tables["matched_updates"]
    new_eligible = rows.refinement_applicable
    rows["direct_prop5_applicable"] = False
    rows["direct_prop5_margin_rmse"] = np.nan
    rows["direct_prop5_variation_l2"] = np.nan
    rows["direct_prop5_integral_status"] = "failed_or_unavailable_legacy_quadrature"
    rows.loc[new_eligible, "condition_sign_status"] = "negative"
    rows.loc[new_eligible, "condition_value_status"] = "unavailable_variation_not_computed"
    rows.loc[new_eligible, "numerical_original_margin_rmse"] = np.nan
    frames, metadata = assemble(tables)
    assert not metadata["audit"]["blocking"]
    assert metadata["audit"]["proposition5_interpretation_migration"]["legacy_figure_status"] == "unavailable"
    assert metadata["figures"]["proposition5_posterior_feedback"]["status"] == "available"
    condition = frames["proposition5_posterior_feedback"].query("metric == 'condition'")
    assert condition.eligible_count.eq(2).all() and condition.unresolved_count.eq(0).all()
    assert condition.fraction.eq(0).all() and condition.upper_fraction.eq(0).all()
    retained = metadata["auxiliary_tables"]["proposition5_numerical_resolution_rows"]
    assert retained.direct_prop5_integral_status.eq("failed_or_unavailable_legacy_quadrature").all()
    assert retained.numerical_original_margin_rmse.loc[new_eligible].isna().all()


def test_unrecognized_inherited_blockers_are_not_blanket_cleared(monkeypatch):
    from utils.experiments.theory import evidence_figures
    original = evidence_figures.build_direct_plot_inputs

    def inherited_failure(*args, **kwargs):
        frames, metadata = original(*args, **kwargs)
        metadata["audit"]["blocking"] = True
        metadata["audit"]["other_statement_failure"] = "preserve independently inherited blocker"
        return frames, metadata

    monkeypatch.setattr(evidence_figures, "build_direct_plot_inputs", inherited_failure)
    _, metadata = assemble()
    assert metadata["audit"]["blocking"]
    assert metadata["audit"]["other_statement_failure"] == "preserve independently inherited blocker"


def test_negative_projected_error_is_preserved_in_synchronization_and_terminal_audits():
    tables = evidence_tables()
    # d=4: mc=0, mu=-2, target=0, reference=-4 along one coordinate.
    # D/sqrt(d)=1, reference-gap norm/sqrt(d)=2, E_parallel/sqrt(d)=-1.
    for name in ("trajectory", "terminal"):
        rows = tables[name]
        _set_trajectory_branch_norms(rows, conditional=0., unconditional=1., error=-1.,
                                     gap=1., reference_gap=2., tail=2.)
    terminal = tables["terminal"]
    for field in ("evidence_terminal_B_obs", "evidence_terminal_B_ref"):
        terminal[field + "_rmse"] = 1.
        terminal[field + "_l2"] = 2.
    terminal["evidence_terminal_conditional_error_rmse"] = 0.
    terminal["evidence_terminal_residual_guidance_rmse"] = 1.
    terminal["evidence_terminal_corrected_obs_bound_rmse"] = 1. + terminal.evidence_terminal_defect_bound_rmse
    terminal["evidence_terminal_corrected_ref_bound_rmse"] = 1. + terminal.evidence_terminal_defect_bound_rmse
    terminal["direct_theorem7_endpoint_error_rmse"] = 0.
    frames, metadata = assemble(tables)
    assert not metadata["audit"]["blocking"]
    synchronization = metadata["auxiliary_tables"]["lemma6_joint_sample_audit"]
    assert synchronization.direct_branch_gap_error_rmse.eq(-1.).all()
    np.testing.assert_allclose(synchronization.branch_gap_error_rhs_identity_residual_rmse, 0.)
    assert frames["lemma6_target_specific_synchronization"].loc[lambda rows: rows.metric.eq("bound"), "median"].eq(1.).all()
    audit = metadata["auxiliary_tables"]["theorem7_looseness_decomposition"]
    assert audit.direct_branch_gap_error_l2.eq(-2.).all()
    np.testing.assert_allclose(audit.slack_projection_alignment_l2, 0.)
    np.testing.assert_allclose(audit.bound_looseness_l2, 0.)
    assert "slack_triangle_l2" not in audit


@pytest.mark.parametrize("table, field", [
    ("trajectory", "direct_branch_gap_error_definition"),
    ("terminal", "direct_branch_gap_error_definition"),
    ("matched_updates", "direct_prop5_branch_gap_error_definition"),
])
def test_historical_norm_error_receipts_cannot_be_relabelled_as_signed_projections(table, field):
    tables = evidence_tables()
    tables[table][field] = "norm_of_learned_minus_reference_gap"
    with pytest.raises(TheoryError, match="signed projected"):
        assemble(tables)
