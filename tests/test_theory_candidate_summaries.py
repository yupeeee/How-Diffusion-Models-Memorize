"""Fixed populations and temporal estimands, including unfavorable/censored data."""

import math

import numpy as np
import pandas as pd
import pytest

from utils.experiments.theory.candidate_summaries import (
    build_candidate_summaries,
    coverage_summary,
    dose_summary,
    feedback_heatmap,
    fixed_snapshots,
    fixed_sscd_bins,
    joint_tolerance_summary,
    prompt_weights,
    recovery_time_summary,
    sscd_bin_summary,
    synchronization_snapshots,
    target_stability_split,
    temporal_feedback_summary,
    terminal_measurements,
    terminal_tolerance_summary,
    timeseries_summary,
    tolerance_grid,
    unique_initial_samples,
    within_prompt_center,
)


def row(prompt=0, seed=0, step=0, **changes):
    result = dict(
        run_id="run",
        original_index=prompt,
        record_id=f"r{prompt}",
        target_id=f"t{prompt}",
        seed=seed,
        step_index=step,
        snr=0.01 * (step + 1),
        terminal_sscd=0.8,
        latent_dimension=4,
        conditional_target_error_rmse=1.0,
        unconditional_target_error_rmse=2.0,
        joint_target_error_rmse=2.0,
        branch_gap_rmse=1.0,
        candidate_log_probability_gain=-1.0,
        candidate_normalized_log_odds_gain=-1.0,
        candidate_matched_log_odds=-3.0,
        candidate_gain_status="negative",
        candidate_profile_class="decreasing",
    )
    result.update(changes)
    return result


def test_weights_give_equal_mass_to_prompt_and_then_seed_despite_repeated_steps():
    frame = pd.DataFrame([row(0, 0, 0), row(0, 0, 1), row(0, 1, 0), row(1, 0, 0)])
    w = prompt_weights(frame)
    assert w.tolist() == [0.25, 0.25, 0.5, 1.0]
    assert w.groupby(frame.original_index).sum().tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="Complete scientific identity"):
        prompt_weights(frame.drop(columns="run_id"))


def test_timeseries_preserves_negative_values_equal_prompt_weight_and_empty_group():
    frame = pd.DataFrame(
        [row(0, i, candidate_log_probability_gain=-10.0) for i in range(3)]
        + [row(1, 0, candidate_log_probability_gain=5.0)]
    )
    result = timeseries_summary(frame, metrics=["candidate_log_probability_gain"])
    all_rows = result.loc[result.group == "all"].iloc[0]
    assert all_rows["mean"] == -2.5
    assert all_rows.minimum == -10
    assert all_rows.maximum == 5
    lower = result.loc[result.group == "SSCD <= 0.75"].iloc[0]
    assert lower.n_observations == 0
    assert math.isnan(lower["median"])


def test_gain_coverage_keeps_unresolved_denominator_and_outcome_prompt_overlap():
    frame = pd.DataFrame(
        [
            row(0, 0, candidate_gain_status="positive"),
            row(
                0, 1, candidate_gain_status="numerically_unresolved", terminal_sscd=0.5
            ),
            row(1, 0, candidate_gain_status="negative"),
            row(2, 0, candidate_gain_status="not_applicable"),
        ]
    )
    result = coverage_summary(frame)
    selected = result.loc[
        (result.group == "all") & (result.metric == "candidate_gain_status")
    ]
    positive = selected.loc[selected.status == "positive"].iloc[0]
    assert positive.denominator_count == 3
    assert positive.fraction == pytest.approx(0.25)
    assert positive.weighted_numerator == 0.5
    assert positive.weighted_denominator == 2.0
    notapp = selected.loc[selected.status == "not_applicable"].iloc[0]
    assert notapp["count"] == 1
    assert notapp.population_fraction == pytest.approx(1 / 3)
    groups = result.loc[
        (result.metric == "candidate_gain_status") & (result.status == "positive")
    ]
    assert groups.loc[groups.group == "SSCD > 0.75", "prompt_count"].item() == 3
    assert groups.loc[groups.group == "SSCD <= 0.75", "prompt_count"].item() == 1


def test_refined_margin_categories_are_never_renamed_original():
    frame = pd.DataFrame(
        [
            row(
                candidate_margin_original_status="negative",
                candidate_margin_combined_status="positive",
            )
        ]
    )
    result = coverage_summary(frame)
    for stem, wanted in [("original", 0.0), ("combined", 1.0)]:
        subset = result.loc[
            (result.group == "all")
            & (result.metric == f"candidate_margin_{stem}_status")
            & (result.status == "positive")
        ]
        assert subset.fraction.item() == wanted


def test_fixed_score_bins_keep_both_tails_and_closed_one_boundary():
    assert fixed_sscd_bins([-0.1, 0.0, 0.1, 0.999, 1.0, 1.1, np.nan]).tolist() == [
        -1,
        0,
        1,
        9,
        9,
        10,
        -2,
    ]
    frame = pd.DataFrame(
        [
            row(i, terminal_sscd=value)
            for i, value in enumerate([-0.1, 1.0, 1.1, np.nan])
        ]
    )
    result = sscd_bin_summary(frame, ["conditional_target_error_rmse"])
    assert result.total_count.sum() == 4
    assert result.loc[result.bin_status == "lower_tail", "n_observations"].item() == 1
    assert result.loc[result.bin_status == "upper_tail", "n_observations"].item() == 1


def test_within_prompt_centering_removes_between_prompt_offsets_only():
    frame = pd.DataFrame(
        [
            row(0, 0, x=0.0, y=100.0),
            row(0, 1, x=2.0, y=102.0),
            row(1, 0, x=20.0, y=-10.0),
            row(1, 1, x=22.0, y=-8.0),
            row(2, 0, x=5.0, y=3.0),
        ]
    )
    result = within_prompt_center(frame, "x", "y")
    assert result.x_centered.tolist() == [-1.0, 1.0, -1.0, 1.0, 0.0]
    assert result.y_centered.tolist() == [-1.0, 1.0, -1.0, 1.0, 0.0]
    assert result.within_prompt_status.iloc[-1] == "no_usable_seed_variation"
    assert len(result) == 5


def test_temporal_response_uses_subsequent_window_not_exposure_window():
    frame = pd.DataFrame(
        [
            row(
                seed=seed,
                step=k,
                candidate_normalized_log_odds_gain=(-2.0 + seed if k < 2 else 1000.0),
                unconditional_target_error_rmse=20.0 - k * (seed + 1),
            )
            for seed in (0, 1)
            for k in range(11)
        ]
    )
    temporal, lagged = temporal_feedback_summary(frame, eligible_steps=list(range(10)))
    assert temporal.window_k20.tolist() == [2, 2]
    assert temporal.window_k40.tolist() == [4, 4]
    assert temporal.early_normalized_gain.tolist() == [-2.0, -1.0]
    assert temporal.subsequent_unconditional_improvement.tolist() == [2.0, 4.0]
    assert temporal.x_centered.tolist() == [-0.5, 0.5]
    assert len(lagged) == 20
    assert set(lagged.lag) == {1}
    assert lagged.next_prediction_step.max() == 10


def test_snapshots_are_fixed_by_indices_and_valid_for_short_runs():
    assert fixed_snapshots(range(49)) == [0, 1, 2, 4, 9, 24, 48]
    assert synchronization_snapshots(range(49)) == [0, 10, 24, 48]
    assert fixed_snapshots([3, 7]) == [3, 7]
    assert fixed_snapshots([]) == []


def test_heatmap_uses_counterfactual_regime_and_has_unresolved_envelope():
    frame = pd.DataFrame(
        [
            row(
                0,
                0,
                0,
                candidate_gain_status="positive",
                candidate_matched_log_odds=-3.0,
                candidate_guided_log_odds=100.0,
            ),
            row(
                1,
                0,
                0,
                candidate_gain_status="numerically_unresolved",
                candidate_matched_log_odds=-3.0,
            ),
            row(
                2,
                0,
                0,
                candidate_gain_status="negative",
                candidate_matched_log_odds=np.nan,
            ),
        ]
    )
    result = feedback_heatmap(frame, eligible_steps=list(range(10)))
    cell = result.loc[
        (result.group == "all") & (result.progress_bin == 0) & (result.regime_bin == 2)
    ].iloc[0]
    assert cell.positive_fraction == 0.5
    assert cell.positive_fraction_upper == 1
    assert cell.denominator_count == 2
    assert cell.status == "low_count"
    missing = result.loc[
        (result.group == "all") & (result.progress_bin == 0) & (result.regime_bin == -1)
    ].iloc[0]
    assert missing.negative_count == 1
    assert (
        result.loc[
            (result.group == "all")
            & (result.progress_bin == 9)
            & (result.regime_bin == 7),
            "status",
        ].item()
        == "empty"
    )


def test_joint_tolerance_is_paired_not_two_marginal_medians_and_preserves_maximum():
    frame = pd.DataFrame(
        [
            row(
                0,
                0,
                conditional_target_error_rmse=0.0,
                unconditional_target_error_rmse=10.0,
                joint_target_error_rmse=10.0,
            ),
            row(
                0,
                1,
                conditional_target_error_rmse=10.0,
                unconditional_target_error_rmse=0.0,
                joint_target_error_rmse=10.0,
            ),
            row(1, 0, joint_target_error_rmse=100000.0),
        ]
    )
    grid = tolerance_grid(frame.joint_target_error_rmse)
    assert grid[-1] == 100000
    result = joint_tolerance_summary(
        frame, grid=[0.0, 5.0, 10.0, 100000.0], snapshots=[0]
    )
    curve = result.loc[result.group == "all"]
    assert curve.fraction.tolist() == [0.0, 0.0, 0.5, 1.0]


def test_first_passage_is_strict_preserves_censoring_and_logs_recrossings():
    c = [0.2, 0.1, 0.3, 0.05, 0.04, 0.03]
    u = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    frame = pd.DataFrame(
        [
            row(
                step=k,
                conditional_target_error_rmse=a,
                unconditional_target_error_rmse=b,
            )
            for k, (a, b) in enumerate(zip(c, u))
        ]
    )
    result = recovery_time_summary(frame, tolerances=[0.1]).iloc[0]
    assert result.conditional_first_step == 3  # equality at k=1 is not below.
    assert result.conditional_three_step_first == 3
    assert result.unconditional_status == "right_censored"
    assert math.isnan(result.unconditional_first_step)
    assert result.unconditional_right_censored
    crossed = recovery_time_summary(frame, tolerances=[0.2]).iloc[0]
    assert crossed.conditional_first_step == 1
    assert crossed.conditional_recrossing_count == 1


def test_zero_endpoint_ratios_are_undefined_and_nonclean_terms_cannot_certify():
    frame = pd.DataFrame(
        [
            row(
                0,
                terminal_clean_applicable=True,
                terminal_rmse=0.0,
                terminal_A_rmse=0.1,
                terminal_B_rmse=0.2,
                candidate_terminal_bound_rmse=0.4,
            ),
            row(
                1,
                terminal_clean_applicable=False,
                terminal_rmse=0.1,
                terminal_A_rmse=0.1,
                terminal_B_rmse=0.2,
                candidate_terminal_bound_rmse=0.4,
            ),
        ]
    )
    result = terminal_measurements(frame)
    assert result.terminal_observed_ratio_status.tolist() == [
        "undefined_zero_endpoint_error",
        "not_applicable",
    ]
    assert result.terminal_observed_bound_error_ratio.isna().all()
    assert result.terminal_error_observed_bound_ratio.iloc[0] == 0
    summary = terminal_tolerance_summary(frame, grid=[0.0, 0.5])
    bound = summary.loc[
        (summary.group == "all") & (summary.metric == "terminal_observed_bound_rmse")
    ]
    assert bound.fraction.tolist() == [0.0, 1.0]
    assert bound.n_observations.tolist() == [1, 1]
    assert bound.not_applicable_count.tolist() == [1, 1]


def test_unique_initial_deduplicates_prompts_but_rejects_inconsistent_seed_values():
    frame = pd.DataFrame(
        [
            row(0, 0, initial_unconditional_center_rmse=0.2),
            row(1, 0, initial_unconditional_center_rmse=0.2),
            row(1, 1, initial_unconditional_center_rmse=0.3),
        ]
    )
    result = unique_initial_samples(frame, ["initial_unconditional_center_rmse"])
    assert len(result) == 2
    frame.loc[1, "initial_unconditional_center_rmse"] = 0.9
    with pytest.raises(ValueError, match="varies within run/seed"):
        unique_initial_samples(frame, ["initial_unconditional_center_rmse"])


def test_stability_split_depends_on_target_identity_not_outcome_or_row_order():
    frame = pd.DataFrame(
        [
            row(i, seed=s, target_id="same" if i < 2 else "other")
            for i in range(3)
            for s in range(2)
        ]
    )
    first = target_stability_split(frame)
    changed = frame.iloc[::-1].copy()
    changed["terminal_sscd"] = -123
    second = target_stability_split(changed)
    expected = first.set_index(
        ["original_index", "seed"]
    ).target_stability_split.sort_index()
    actual = second.set_index(
        ["original_index", "seed"]
    ).target_stability_split.sort_index()
    pd.testing.assert_series_equal(actual, expected)
    assert first.loc[first.target_id == "same", "target_stability_split"].nunique() == 1


def test_compact_dose_expands_only_fixed_snapshots_and_preserves_negative_gain():
    frame = pd.DataFrame(
        [
            row(
                step=k,
                **{
                    "lambda": [0.0, 0.5, 1.0],
                    "candidate_dose_log_probability_gain": [0.0, -1.0, -3.0],
                },
            )
            for k in range(3)
        ]
    )
    result = dose_summary(frame, snapshots=[0, 2])
    selected = result.loc[result.group == "all"]
    assert len(selected) == 6
    assert set(selected.step_index) == {0, 2}
    assert selected.loc[selected["lambda"] == 1, "median"].tolist() == [-3.0, -3.0]


def test_build_endpoint_stage_needs_no_integrated_margins_or_positive_findings():
    frame = pd.DataFrame([row(step=k, seed=s) for k in range(3) for s in range(2)])
    result = build_candidate_summaries(frame, eligible_steps=[0, 1])
    assert {
        "timeseries",
        "coverage",
        "feedback_heatmap",
        "joint_tolerance",
        "recovery_times",
        "stability",
    } <= set(result)
    gain = result["coverage"].query(
        "group=='all' and metric=='candidate_gain_status' and status=='positive'"
    )
    assert gain.fraction.eq(0).all()
    assert result["terminal_ratios"].empty


def test_retrieval_ecdf_counts_ties_and_reweights_each_outcome_group():
    from utils.experiments.theory.candidate_summaries import retrieval_ecdf_summary

    frame = pd.DataFrame(
        [
            row(
                0,
                i,
                initial_conditional_target_rank=1,
                initial_conditional_target_tie_count=2,
                initial_retrieval_bank_size=10,
            )
            for i in range(3)
        ]
        + [
            row(
                1,
                0,
                initial_conditional_target_rank=10,
                initial_conditional_target_tie_count=1,
                initial_retrieval_bank_size=10,
            )
        ]
    )
    result = retrieval_ecdf_summary(frame)
    all_rows = result.loc[result.group == "all"]
    assert all_rows.fraction.tolist() == [0.5, 1.0]
    assert all_rows.tie_count.tolist() == [3, 3]


def test_fixed_phase_summary_retains_adverse_observations():
    from utils.experiments.theory.candidate_summaries import phase_summary

    frame = pd.DataFrame(
        [row(step=k, candidate_log_probability_gain=float(k - 5)) for k in range(6)]
    )
    result = phase_summary(frame)
    result = result.loc[
        (result.group == "all") & (result.metric == "candidate_log_probability_gain")
    ]
    assert result.phase.tolist() == ["early", "middle", "late"]
    assert result["mean"].tolist() == [-4.5, -2.5, -0.5]
    assert result.n_observations.tolist() == [2, 2, 2]


def test_score_binned_initial_association_uses_all_data_range():
    from utils.experiments.theory.candidate_summaries import initial_error_bin_summary

    frame = pd.DataFrame(
        [
            row(0, conditional_target_error_rmse=0.0),
            row(1, conditional_target_error_rmse=10000.0),
        ]
    )
    result = initial_error_bin_summary(frame)
    assert result.bin_lower.min() == 0
    assert result.bin_upper.max() == 10000
    assert result.total_count.sum() == 2


def test_boolean_saturation_is_separate_from_positive_gain_sign():
    frame = pd.DataFrame(
        [
            row(
                0,
                candidate_gain_status="positive",
                candidate_gain_saturated=True,
                candidate_gain_underflow=True,
            ),
            row(
                1,
                candidate_gain_status="negative",
                candidate_gain_saturated=True,
                candidate_gain_underflow=False,
            ),
        ]
    )
    result = coverage_summary(frame)
    saturation = result.loc[
        (result.group == "all")
        & (result.metric == "candidate_gain_saturated")
        & (result.status == "True")
    ]
    assert saturation.fraction.item() == 1
    positive = result.loc[
        (result.group == "all")
        & (result.metric == "candidate_gain_status")
        & (result.status == "positive")
    ]
    assert positive.fraction.item() == 0.5


def test_terminal_certification_denominator_keeps_unavailable_bound_rows():
    frame = pd.DataFrame(
        [
            row(
                0,
                terminal_clean_applicable=True,
                terminal_rmse=0.1,
                terminal_A_rmse=0.1,
                terminal_B_rmse=0.1,
                candidate_terminal_bound_rmse=0.3,
            ),
            row(
                1,
                terminal_clean_applicable=True,
                terminal_rmse=0.1,
                terminal_A_rmse=0.1,
                terminal_B_rmse=0.1,
                candidate_terminal_bound_rmse=np.nan,
            ),
        ]
    )
    result = terminal_tolerance_summary(frame, grid=[1.0])
    candidate = result.loc[
        (result.group == "all") & (result.metric == "terminal_candidate_bound_rmse")
    ].iloc[0]
    assert candidate.fraction == 0.5
    assert candidate.fraction_upper == 1
    assert candidate.unresolved_fraction == 0.5
    assert candidate.n_observations == 2
    assert candidate.measured_count == 1


def test_short_temporal_window_has_explicit_unavailable_response():
    frame = pd.DataFrame([row(step=k) for k in range(2)])
    early, _ = temporal_feedback_summary(frame, eligible_steps=[0])
    assert early.response_status.item() == "not_applicable_no_subsequent_interval"
    assert np.isnan(early.subsequent_unconditional_improvement.item())


def test_estimated_positive_margin_is_not_mislabeled_resolved_positive():
    frame = pd.DataFrame(
        [
            row(
                candidate_margin_original_status="numerically_unresolved",
                candidate_margin_original_estimated_sign="positive",
            )
        ]
    )
    summary = coverage_summary(frame)
    resolved = summary.loc[
        (summary.group == "all")
        & (summary.metric == "candidate_margin_original_status")
        & (summary.status == "positive")
    ]
    estimated = summary.loc[
        (summary.group == "all")
        & (summary.metric == "candidate_margin_original_estimated_sign")
        & (summary.status == "positive")
    ]
    assert resolved.fraction.item() == 0
    assert estimated.fraction.item() == 1


def test_independent_replay_unavailable_status_is_preserved_separately_from_construction():
    status = "unavailable_saved_noise_realization_recovered_innovation_is_not_independent_replay"
    frame = pd.DataFrame(
        [
            row(
                independent_replay_status=status,
                independent_replay_rmse=np.nan,
                matched_construction_vector_residual_rmse=0.0,
                update_rounding_sensitivity_rmse=0.001,
            )
        ]
    )
    covered = coverage_summary(frame)
    observed = covered.loc[
        (covered.group == "all")
        & (covered.metric == "independent_replay_status")
        & (covered.status == status)
    ].iloc[0]
    assert observed["count"] == 1
    assert observed.denominator_count == 0
    assert observed.population_fraction == 1
    series = timeseries_summary(frame)
    replay = series.loc[
        (series.group == "all") & (series.metric == "independent_replay_rmse")
    ].iloc[0]
    assert replay.n_observations == 0
    envelope = series.loc[
        (series.group == "all") & (series.metric == "update_rounding_sensitivity_rmse")
    ].iloc[0]
    assert envelope["median"] == 0.001


def test_all_normalized_gains_unavailable_remain_missing_through_temporal_summary():
    frame = pd.DataFrame(
        [
            row(step=k, seed=s, candidate_normalized_log_odds_gain=None)
            for k in range(6)
            for s in range(2)
        ]
    )
    temporal, lagged = temporal_feedback_summary(frame, eligible_steps=list(range(5)))
    assert temporal.early_normalized_gain.isna().all()
    assert temporal.exposure_finite_count.eq(0).all()
    assert temporal.within_prompt_status.eq("missing_pair").all()
    assert lagged.normalized_gain.isna().all()
    assert lagged.x_centered.isna().all()
    assert len(lagged) == 10


def test_nullable_prediction_response_does_not_become_zero():
    frame = pd.DataFrame(
        [
            row(step=k, unconditional_target_error_rmse=None if k == 2 else 2.0)
            for k in range(6)
        ]
    )
    temporal, lagged = temporal_feedback_summary(frame, eligible_steps=list(range(5)))
    assert temporal.subsequent_unconditional_improvement.isna().all()
    assert temporal.response_status.item() == "missing_prediction_endpoint"
    assert lagged.next_unconditional_improvement.isna().sum() == 2


def test_cached_metric_weights_distinguish_equal_size_missing_populations():
    frame = pd.DataFrame(
        [row(0, 0), row(0, 0), row(0, 1), row(1, 0), row(1, 0), row(1, 1)]
    )
    columns = (
        "candidate_log_probability_gain",
        "candidate_log_odds_gain",
        "candidate_current_alignment_cosine",
    )
    frame[columns[0]] = [2.0, 4.0, None, 10.0, 14.0, 18.0]
    frame[columns[1]] = [None, 4.0, 8.0, 10.0, 14.0, 18.0]
    frame[columns[2]] = [2.0, None, 8.0, 10.0, 14.0, 18.0]
    frame.index = [0] * len(frame)
    result = timeseries_summary(frame, metrics=columns)
    all_rows = result.loc[result.group == "all"].set_index("metric")
    # Each prompt has mass one; repeated rows divide their seed's mass.
    assert all_rows.loc[list(columns), "mean"].tolist() == [9.0, 10.5, 10.0]
    assert all_rows.loc[list(columns), "median"].tolist() == [4.0, 8.0, 8.0]
    assert all_rows.n_observations.tolist() == [5, 5, 5]
    assert all_rows.prompt_count.tolist() == [2, 2, 2]
    assert all_rows.missing_count.tolist() == [1, 1, 1]


def test_cached_summaries_preserve_numeric_coercion_after_missing_values_removed():
    values = [
        "9007199254740993",
        "bad",
        "9007199254740995",
        "-9223372036854775807",
        None,
        "3",
        "4",
        "5",
    ]
    frame = pd.DataFrame(
        [
            row(seed=i, candidate_log_probability_gain=value)
            for i, value in enumerate(values)
        ]
    )
    result = timeseries_summary(frame, metrics=["candidate_log_probability_gain"])
    observed = result.loc[result.group == "all"].iloc[0]
    expected = np.array(
        [9007199254740993, 9007199254740995, -9223372036854775807, 3, 4, 5],
        dtype=np.float64,
    )
    assert observed.minimum == expected.min()
    assert observed.q10 == expected.min()
    assert observed.maximum == expected.max()
    assert observed["mean"] == np.average(expected, weights=np.full(6, 1.0 / 6))
    assert observed.n_observations == 6
    assert observed.missing_count == 2
