"""Synthetic direct-statement contracts; execution is left to the author."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from utils.experiments.theory.contracts import TheoryError, numerical_config
from utils.experiments.theory.direct_figures import (
    RECOVERY_TOLERANCES,
    build_direct_plot_inputs,
)
from utils.experiments.theory.paper_registry import direct_audit_registry as paper_registry
from utils.experiments.theory import paper_plotting


def direct_config(**overrides):
    return numerical_config(
        **{**dict(model_name="sdv1", scheduler_name="ddim", num_seeds=2,
                   num_inference_steps=2, guidance_scale=2.0), **overrides}
    )


def direct_tables():
    """Complete synthetic scalar inputs; no checkpoint, tensors or file reads."""
    initial, trajectory, matched, terminal, gaussian, loss = [], [], [], [], [], []
    for pair in range(2):
        identity = dict(run_id="run", original_index=str(pair), record_id=f"record-{pair}", target_id=f"target-image-{pair}")
        loss.append({**identity, "step_index": 0, "timestep": 10, "snr": 1., "latent_dimension": 4,
                     "input_source": "forward_target", "draw_count": 64,
                     "loss_mean_squared_l2": 16. * (pair + 1), "loss_mc_se_squared_l2": 1.,
                     "forward_clean_error_mean_squared_l2": 16. * (pair + 1),
                     "pinsker_tv_bound": .2, "status": "complete"})
        for seed in range(2):
            ec = 1. + 8. * seed
            sample = {**identity, "seed": seed, "terminal_sscd": .8 if seed else .4,
                      "latent_dimension": 4, "reference_law_hash": "law",
                      "reference_law_scope": "cached-targets"}
            gaussian.append({**sample, "step_index": 0, "timestep": 10, "snr": 1.,
                             "reference_law_hash": "fixed_pair_target", "reference_law_scope": "fixed_pair_target",
                             "input_source": "gaussian_probe", "conditional_error_l2": ec,
                             "conditional_error_squared_l2": ec**2, "conditional_error_rmse": ec / 2,
                             "terminal_sscd_matched_initialization": True,
                             "initialization_status": "matched_saved_standard_gaussian_with_recorded_quantization"})
            for step in range(2):
                row = {**sample, "step_index": step, "timestep": 10 - 9 * step, "snr": 1. + step,
                       "input_source": "generated_state", "direct_conditional_error_rmse": .2,
                       "direct_unconditional_target_error_rmse": .7,
                       "direct_unconditional_reference_error_rmse": .1,
                       "direct_radius_tail_rmse": .4, "direct_reference_target_error_rmse": .3,
                       "direct_lemma6_rhs_rmse": .7, "direct_lemma6_gap_rmse": .5,
                       "direct_lemma6_applicable": True, "direct_lemma6_status": "declared_law",
                       "direct_branch_gap_error_definition": "signed_projected_branch_gap_reference_error",
                       "direct_prop5_branch_gap_error_definition": "signed_projected_branch_gap_reference_error"}
                trajectory.append(row)
                matched.append({**row, "direct_lemma4_predicted_rmse": 1., "direct_lemma4_displacement_rmse": 1.,
                                "direct_lemma4_vector_residual_rmse": 1e-8,
                                "direct_lemma4_source_tolerance_rmse": 1e-5,
                                "direct_lemma4_independent": not bool(seed),
                                "direct_lemma4_verification_source": "independent_deterministic_affine_counterfactual" if not seed else "constructed_shared_innovation_from_saved_endpoint_not_independent",
                                "direct_lemma4_applicable": True,
                                "direct_prop5_margin_rmse": -.5 if not seed else .1,
                                "direct_prop5_log_probability_gain": -.7 if not seed else .8,
                                "direct_prop5_condition_status": "estimated_negative" if not seed else "numerically_unresolved",
                                "direct_prop5_gain_status": "negative" if not seed else "positive",
                                "direct_prop5_integral_status": "converged",
                                "direct_prop5_applicable": step == 0,
                                "direct_prop5_estimated_condition_sign": "negative" if not seed else "positive",
                                "direct_prop5_nonnegative_condition": bool(seed)})
                if step == 0:
                    initial.append({**row, "direct_cor3_injection_bound_rmse": 2., "direct_cor3_injection_discrepancy_rmse": 1.,
                                    "direct_cor3_guided_bound_rmse": 1.5, "direct_cor3_guided_discrepancy_rmse": .8,
                                    "direct_cor3_applicable": True, "direct_cor3_status": "derived_triangle"})
                else:
                    terminal.append({**row, "direct_theorem7_bound_rmse": 1., "direct_theorem7_endpoint_error_rmse": 4.,
                                     "direct_theorem7_bound_l2": 2., "direct_theorem7_endpoint_error_l2": 8.,
                                     "direct_theorem7_applicable": False, "direct_theorem7_status": "inapplicable_nonzero_terminal_noise",
                                     "direct_theorem7_condition_certified": False, "direct_theorem7_actual_proximity": False})
    reference = pd.DataFrame([
        dict(run_id="run", seed=seed, step_index=step, timestep=10 - 9 * step, snr=1. + step,
             latent_dimension=4, input_source="gaussian_probe", reference_law_hash="law", reference_law_scope="cached-targets",
             learned_mean_error_rmse=2. + seed, reference_mean_error_rmse=.5,
             unconditional_reference_error_rmse=2. + seed, baseline_bound_rmse=2.5 + seed)
        for step in range(2) for seed in range(2)
    ])
    return {"initial": pd.DataFrame(initial), "trajectory": pd.DataFrame(trajectory),
            "matched_updates": pd.DataFrame(matched), "terminal": pd.DataFrame(terminal),
            "gaussian_conditional": pd.DataFrame(gaussian), "gaussian_reference": reference,
            "forward_loss_summary": pd.DataFrame(loss), "forward_unconditional_loss": pd.DataFrame()}


def assemble(tables=None, **options):
    return build_direct_plot_inputs(tables or direct_tables(), direct_config(**options), {"reference_law_hash": "law"})


def test_registry_maps_exact_seven_statements_and_keeps_behavioral_diagnostics():
    entries = paper_registry()
    main = [e for e in entries if e["category"] == "main"]
    assert [e["result_number"] for e in main] == list(range(1, 8))
    assert [e["stem"] for e in main] == [
        "theorem1_loss_recovery", "lemma2_unconditional_baseline", "corollary3_initial_cfg_amplification",
        "lemma4_matched_displacement", "proposition5_posterior_feedback",
        "lemma6_target_specific_synchronization", "theorem7_final_reproduction",
    ]
    for entry in main:
        for key in ("manuscript_label", "x_definition", "y_definition", "reference_law", "input_source",
                    "applicability_assumptions", "interpretation", "averaging_measure", "averaging_unit", "supporting_equation"):
            assert entry[key]
    assert not main[0].get("equal") and not main[4].get("equal")
    assert main[4]["zero_guides"]
    optional = {e["stem"]: e for e in paper_registry(True)}
    for stem in ("initial_recovery", "initial_unconditional_concentration", "target_synchronization", "posterior_feedback_positive_fraction"):
        assert optional[stem]["category"] == "diagnostics"
        assert not optional[stem]["direct_statement"]


def test_forward_loss_and_gaussian_outcomes_are_independent_and_normalized_once():
    frames, metadata = assemble()
    observed = frames["theorem1_loss_recovery"]
    assert observed.x.tolist() == [2., 2., np.sqrt(8.), np.sqrt(8.)]
    assert observed.y.tolist() == [.5, 4.5, .5, 4.5]
    forward = frames["theorem1_forward_identity"]
    np.testing.assert_allclose(forward.x, forward.y)
    assert len(forward) == 2 and len(observed) == 4
    summary = metadata["auxiliary_tables"]["theorem1_gaussian_summary"]
    np.testing.assert_allclose(summary.gaussian_root_mean_squared_error, np.sqrt(10.25))
    np.testing.assert_allclose(summary.gaussian_error_rmse_mean, 2.5)
    assert summary.loss_draw_count.tolist() == [64, 64]
    assert not np.allclose(observed.x, observed.y)


def test_transfer_keeps_all_tolerances_raw_bounds_and_uncertainty():
    frames, metadata = assemble()
    frame = frames["theorem1_gaussian_transfer"]
    assert set(frame.rho) == set(RECOVERY_TOLERANCES)
    assert len(frame) == 10
    np.testing.assert_allclose(frame.raw_tolerance_l2, frame.rho * 2)
    assert (frame.bound_unclipped > 1).all()
    assert frame.x.eq(1).all() and frame.saturated_at_one.all()
    assert (frame.frequency_low <= frame.y).all() and (frame.frequency_high >= frame.y).all()
    assert (frame.bound_mc_se > 0).all()
    assert metadata["figures"]["theorem1_gaussian_transfer"]["saturation_to_one_count"] == 10


def test_missing_loss_wrong_input_law_or_duplicate_gaussian_seed_is_rejected():
    for mutation, match in (("missing", "Missing required"), ("wrong_source", "mixes input laws"), ("duplicate", "duplicate")):
        tables = direct_tables()
        if mutation == "missing":
            tables["forward_loss_summary"] = pd.DataFrame()
        elif mutation == "wrong_source":
            tables["gaussian_conditional"]["input_source"] = "generated_state"
        else:
            tables["gaussian_reference"] = pd.concat([tables["gaussian_reference"], tables["gaussian_reference"].iloc[:1]])
        with pytest.raises(TheoryError, match=match):
            assemble(tables)


def test_reference_law_cannot_mix_and_unmatched_gaussian_sscd_is_removed():
    tables = direct_tables()
    tables["gaussian_reference"]["reference_law_hash"] = "other"
    with pytest.raises(TheoryError, match="mix reference-law"):
        assemble(tables)
    tables = direct_tables()
    tables["gaussian_conditional"]["terminal_sscd_matched_initialization"] = False
    frames, _ = assemble(tables)
    assert frames["theorem1_loss_recovery"].terminal_sscd.isna().all()


def test_gaussian_sweep_uses_unique_seeds_and_three_separate_metrics():
    frames, _ = assemble()
    baseline = frames["lemma2_unconditional_baseline"]
    assert len(baseline) == 6
    assert set(baseline.metric) == {"reference", "learned", "reference_error"}
    assert baseline.seed_count.eq(2).all()
    assert "group" not in baseline
    assert len(frames["lemma2_reference_to_learned_bound"]) == 4


def test_feedback_keeps_original_signed_values_unresolved_and_structural_counts():
    frames, metadata = assemble()
    frame = frames["proposition5_posterior_feedback"]
    assert len(frame) == 4
    assert frame.x.tolist() == [-.5, .1, -.5, .1]
    assert frame.y.tolist() == [-.7, .8, -.7, .8]
    assert frame.marker_class.tolist() == ["observed", "numerically_unresolved"] * 2
    assert metadata["figures"]["proposition5_posterior_feedback"]["excluded_structural_rows"] == 4
    summary = metadata["auxiliary_tables"]["statement_summary"]
    row = summary.loc[summary.figure.eq("proposition5_posterior_feedback")].iloc[0]
    assert row.estimated_strict_positive_count == 2
    assert row.alleged_strict_implication_violations == 0
    assert "not_certified" in row.certification_status


def test_all_inapplicable_terminal_pairs_remain_with_no_theorem_summary_claim():
    frames, metadata = assemble(target_error_tolerance=3.)
    frame = frames["theorem7_final_reproduction"]
    assert len(frame) == 4 and frame.marker_class.eq("inapplicable").all()
    assert frame.y.eq(4).all() and frame.x.eq(1).all()
    meta = metadata["figures"]["theorem7_final_reproduction"]
    assert meta["status"] == "available"
    assert meta["comparison_status"] == "0/4 satisfy the clean-update prerequisite"
    assert metadata["figures"]["theorem7_certification"]["status"] == "not_applicable"
    row = metadata["auxiliary_tables"]["statement_summary"].query("figure == 'theorem7_final_reproduction'").iloc[0]
    assert row.bound_slack_finite_count == 0
    assert len(frames["lemma6_target_specific_synchronization"]) == 8


def test_feedback_implication_summary_uses_stable_gain_sign_with_underflow():
    tables = direct_tables()
    rows = tables["matched_updates"]
    eligible = rows.direct_prop5_applicable
    rows.loc[eligible, "direct_prop5_margin_rmse"] = .1
    rows.loc[eligible, "direct_prop5_condition_status"] = "estimated_positive"
    rows.loc[eligible, "direct_prop5_log_probability_gain"] = 0.
    rows.loc[eligible, "direct_prop5_gain_status"] = [
        "positive", "negative", "zero", "numerically_unresolved"
    ]
    rows["direct_prop5_gain_underflow"] = False
    rows.loc[eligible, "direct_prop5_gain_underflow"] = [True, True, False, False]
    frames, metadata = assemble(tables)
    assert frames["proposition5_posterior_feedback"].y.eq(0).all()
    summary = metadata["auxiliary_tables"]["statement_summary"]
    row = summary.loc[summary.figure.eq("proposition5_posterior_feedback")].iloc[0]
    assert row.alleged_strict_implication_violations == 2
    assert row.alleged_nonstrict_implication_violations == 1
    assert row.gain_underflow_count == 2


def test_fullvector_and_independent_update_classes_are_not_projection_substitutes():
    frames, _ = assemble()
    assert frames["corollary3_initial_cfg_amplification"].x.eq(2.).all()
    assert frames["corollary3_guided_estimate_error"].x.eq(1.5).all()
    assert set(frames["lemma4_matched_displacement"].marker_class) == {"independently_checked", "constructed"}
    assert frames["lemma4_vector_residual"].y.eq(1e-8).all()


def test_renderer_uses_saved_scalars_and_correct_reference_lines(monkeypatch):
    frames, metadata = assemble()
    entries = {e["stem"]: e for e in paper_registry()}
    config = {"scientific_config": direct_config()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Renderer attempted numerical recomputation")

    import utils.experiments.theory.direct_figures as numerical
    monkeypatch.setattr(numerical, "build_direct_plot_inputs", forbidden)
    for stem in ("theorem1_loss_recovery", "proposition5_posterior_feedback", "theorem7_final_reproduction", "lemma4_matched_displacement"):
        figure, stats = paper_plotting._draw(entries[stem], frames[stem], metadata["figures"][stem], config)
        try:
            ax = figure.axes[0]
            if stem == "theorem1_loss_recovery":
                assert not ax.lines
            elif stem == "proposition5_posterior_feedback":
                assert len(ax.lines) == 2 and ax.get_yscale() == "symlog"
                assert stats["marker_counts"]["numerically_unresolved"] == 2
            elif stem == "theorem7_final_reproduction":
                assert not ax.lines
                assert any("0/4" in text.get_text() for text in ax.texts)
            else:
                assert any(line.get_label() == "Update identity" for line in ax.lines)
        finally:
            plt.close(figure)


def test_missing_gaussian_seed_does_not_silently_shrink_the_population():
    tables = direct_tables()
    tables["gaussian_conditional"] = tables["gaussian_conditional"].iloc[1:].copy()
    with pytest.raises(TheoryError, match="every evaluation seed"):
        assemble(tables)


def test_excess_loss_uses_direct_draws_without_forward_gaussian_identity():
    tables = direct_tables()
    tables["forward_unconditional_loss"] = pd.DataFrame([
        dict(run_id="run", step_index=0, draw_index=i, snr=1., latent_dimension=4,
             input_source="forward_marginal", reference_law_hash="law",
             excess_loss_squared_l2=value, total_loss_squared_l2=100., optimal_loss_squared_l2=99.)
        for i, value in enumerate([4., 12.])
    ])
    frames, _ = assemble(tables, measure_unconditional_loss=True)
    observed = frames["lemma2_excess_loss_gaussian_error"]
    assert len(observed) == 2 and observed.x.eq(2.).all()
    assert observed.y.tolist() == [4., 9.]
    entry = next(e for e in paper_registry() if e["stem"] == "lemma2_excess_loss_gaussian_error")
    assert not entry.get("equal")


def test_terminal_certification_preserves_recorded_underflow_exclusion():
    tables = direct_tables()
    rows = tables["terminal"]
    rows["direct_theorem7_applicable"] = True
    rows["direct_theorem7_condition_certified"] = False
    rows["direct_theorem7_actual_proximity"] = False
    frames, _ = assemble(tables, target_error_tolerance=3.)
    # The raw estimate 2 <= 3 is insufficient to override an upstream numerical
    # exclusion; the compact stage preserves the measured certification event.
    assert frames["theorem7_certification"].x.iloc[0] == 0


def test_compact_builder_covers_all_registry_entries_with_named_statuses():
    frames, metadata = assemble()
    assert set(metadata["figures"]) == {e["stem"] for e in paper_registry(True)}
    for stem, info in metadata["figures"].items():
        assert info["status"] in {"available", "unavailable", "not_applicable"}
        if info["status"] != "available":
            assert info["reason"]
        assert stem in frames
    for stem in ("initial_recovery", "target_synchronization", "posterior_feedback_positive_fraction"):
        assert metadata["figures"][stem]["status"] == "available"
    assert metadata["figures"]["initial_unconditional_concentration"]["status"] == "unavailable"
