"""Four-stage scalar contracts; authored source-only, execution left to author."""
import numpy as np
import pandas as pd
import pytest

from tests.test_theory_evidence_figures import evidence_config, evidence_tables
from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.four_stage_figures import build_four_stage_plot_inputs
from utils.experiments.theory.paper_registry import GROUPS, evidence_registry, paper_registry


def four_stage_config(**overrides):
    return evidence_config(**overrides)


def four_stage_tables():
    tables = evidence_tables()
    initial = tables["initial"]
    tables["trajectory_metrics"] = tables["trajectory"].copy()
    tables["trajectory_metrics"]["direct_lemma6_gap_rmse"] = np.where(tables["trajectory_metrics"].step_index.eq(0), .5, .7)
    tables["initial_samples"] = initial.copy()
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
    return build_four_stage_plot_inputs(tables or four_stage_tables(), four_stage_config(**config), {"reference_law_hash": "law"})


def test_fixed_four_main_eleven_appendix_and_legacy_diagnostics():
    entries = paper_registry()
    assert [entry["stem"] for entry in entries if entry["category"] == "main"] == [
        "initial_loss_recovery", "branch_gap_posterior_response", "branch_gap_synchronization", "final_reproduction_bound"]
    assert sum(entry["category"] == "appendix" for entry in entries) == 11
    assert len(evidence_registry()) == 19
    optional = {entry["stem"]: entry for entry in paper_registry(True)}
    assert optional["lemma4_matched_displacement"]["category"] == "diagnostics"
    assert "counterfactual_unconditional_response" not in {entry["stem"] for entry in entries}
    assert paper_registry(counterfactual=True)[-1]["category"] == "appendix"


def test_pair_rms_unique_seed_baseline_and_full_native_sweep():
    frames, meta = assemble()
    pairs = frames["initial_loss_recovery"]
    np.testing.assert_allclose(pairs.y, np.sqrt(10.25))
    np.testing.assert_allclose(pairs.control_y, np.sqrt(50.))
    np.testing.assert_allclose(pairs.mean_terminal_sscd, .6)
    assert "control_y_low" in pairs and len(pairs) == 2
    assert {"A_c", "Y_c", "U_c", "loss_mean_squared_l2", "loss_mc_se_squared_l2"}.issubset(meta["auxiliary_tables"]["initial_pairs"])
    assert meta["auxiliary_tables"]["initial_baseline_summary"].unique_seed_count.tolist() == [2]
    sweep = frames["unconditional_reference_convergence"]
    assert set(sweep.loc[sweep.source_range.eq("native"), "step_index"]) == {0, 1}
    assert sweep.loc[sweep.source_range.eq("analytical"), "metric"].eq("reference").all()


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
    assert meta["figures"]["final_reproduction_bound"]["status"] == "available"
    assert not meta["audit"]["blocking"]
    tables["terminal"].loc[1, "evidence_terminal_reconstruction_status"] = "failed_reconstruction"
    _, blocked = assemble(tables)
    assert blocked["audit"]["blocking"]
    assert blocked["figures"]["final_reproduction_bound"]["status"] == "blocked"


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
