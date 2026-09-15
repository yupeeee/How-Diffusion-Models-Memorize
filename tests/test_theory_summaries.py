"""Descriptive summary units, identities, weighting, and adverse outcomes."""

import math

import numpy as np
import pandas as pd
import pytest

from utils.experiments.theory.summaries import (
    OUTCOME_GROUPS,
    feedback_counts,
    feedback_summary,
    initial_prompt_summary,
    paired_synchronization_summary,
    phase_summary,
    synchronization_summary,
    tolerance_entry_summary,
    weighted_quantiles,
)


def observation(
    prompt="a",
    seed=0,
    step=0,
    *,
    conditional=1.0,
    unconditional=2.0,
    sscd=0.9,
    run="run-a",
    original_index=None,
):
    return dict(
        run_id=run,
        original_index=prompt if original_index is None else original_index,
        record_id="record-" + prompt,
        target_id="target-" + prompt,
        seed=seed,
        step_index=step,
        snr=10.0**step,
        destination_snr=10.0 ** (step + 1),
        terminal_sscd=sscd,
        latent_dimension=4,
        conditional_target_error_rmse=conditional,
        unconditional_target_error_rmse=unconditional,
        joint_target_error_rmse=max(conditional, unconditional),
        branch_gap_rmse=abs(unconditional - conditional),
        paired_target_error_improvement_rmse=unconditional - conditional,
        relative_target_error_improvement=(
            (unconditional - conditional) / unconditional if unconditional else np.nan
        ),
    )


def test_prompt_balanced_curves_do_not_weight_large_seed_cohorts_as_more_prompts():
    rows = []
    for step in (0, 1):
        rows.append(
            observation("a", 0, step, conditional=1 + step, unconditional=101 + step)
        )
        for seed, value in enumerate((10.0, 11.0, 12.0)):
            rows.append(
                observation(
                    "b",
                    seed,
                    step,
                    conditional=value + step,
                    unconditional=100 + value + step,
                )
            )
        rows.append(
            observation(
                "a", 8, step, conditional=3 + step, unconditional=30 + step, sscd=0.75
            )
        )
        rows.append(
            observation(
                "b", 8, step, conditional=4 + step, unconditional=40 + step, sscd=0.2
            )
        )
    summary = synchronization_summary(pd.DataFrame(rows))
    assert len(summary) == 8  # Both branches, both fixed cohorts, both saved steps.
    high = summary.loc[summary.group.eq(OUTCOME_GROUPS[0])]
    first = high.loc[high.branch.eq("conditional") & high.step_index.eq(0)].iloc[0]
    assert first.error_median == 1  # Each prompt has half the total weight.
    assert first.error_median != np.median([1, 10, 11, 12])
    assert first.error_q25 == 1 and first.error_q75 == 11
    assert first.prompt_count == 2 and first.seed_count == 4
    assert first.n_observations == 4
    null = high.loc[high.branch.eq("unconditional") & high.step_index.eq(0)].iloc[0]
    assert null.error_median == 101
    low = summary.loc[summary.group.eq(OUTCOME_GROUPS[1])]
    assert set(low.branch) == {"conditional", "unconditional"}
    assert set(low.seed_count) == {2}


def test_empty_cohort_stays_empty_and_strict_threshold_is_not_relaxed():
    frame = pd.DataFrame([observation(sscd=0.75), observation("b", sscd=0.4)])
    summary = synchronization_summary(frame)
    assert set(summary.group) == {OUTCOME_GROUPS[1]}
    assert len(summary) == 2
    frame["terminal_sscd"] = np.nan
    summary = synchronization_summary(frame)
    assert summary.empty
    assert {"branch", "group", "error_median", "seed_count"} <= set(summary.columns)


def test_branch_gap_zero_does_not_imply_small_paired_joint_errors():
    frame = pd.DataFrame([observation(conditional=9, unconditional=9)])
    curves = synchronization_summary(frame)
    assert set(curves.error_median) == {9}
    paired = paired_synchronization_summary(frame).iloc[0]
    assert paired.branch_gap_rmse_median == 0
    assert paired.joint_target_error_rmse_median == 9
    assert paired.paired_target_error_improvement_rmse_median == 0


def test_summary_prompt_counts_use_complete_identities():
    frame = pd.DataFrame(
        [
            observation("a", run="one", original_index="same"),
            observation("a", run="two", original_index="same"),
        ]
    )
    assert set(synchronization_summary(frame).prompt_count) == {2}
    assert paired_synchronization_summary(frame).iloc[0].prompt_count == 2
    phases = phase_summary(frame, 1)
    assert (
        phases.loc[
            phases.group.eq(OUTCOME_GROUPS[0]) & phases.phase.eq("early"),
            "prompt_count",
        ].item()
        == 2
    )


def test_fixed_phase_partition_preserves_late_deterioration():
    frame = pd.DataFrame(
        [
            observation(step=k, conditional=value, unconditional=value)
            for k, value in enumerate((5.0, 4.0, 3.0, 2.0, 100.0))
        ]
    )
    phases = phase_summary(frame, 5)
    high = phases.loc[phases.group.eq(OUTCOME_GROUPS[0])].set_index("phase")
    assert high.n_observations.to_dict() == {"early": 2, "intermediate": 2, "late": 1}
    assert high.loc["late", "joint_target_error_rmse_median"] == 100
    assert set(high.sample_count) == {1}
    empty = phases.loc[phases.group.eq(OUTCOME_GROUPS[1])]
    assert set(empty.n_observations) == {0}
    assert empty.joint_target_error_rmse_median.isna().all()


def test_weighted_quantiles_ignore_unavailable_values_and_invalid_weights():
    answer = weighted_quantiles([1, 10, np.nan, np.inf, -100], [1, 3, 100, 100, 0])
    assert answer.tolist() == [1, 10, 10]
    assert np.isnan(weighted_quantiles([np.nan], [1])).all()


def feedback_frame():
    rows = []
    cases = [
        # eligible, condition status, gain sign, gain, saturation, underflow, score
        (True, "estimated_met", "positive", 0.0, True, True, 0.9),
        (True, "estimated_met", "numerically_unresolved", 1e-15, False, False, 0.2),
        (True, "estimated_not_met", "negative", -2.0, False, False, 0.95),
        (True, "estimated_not_met", "positive", 0.2, False, False, 0.3),
        (True, "numerically_unresolved", "zero", 0.0, True, False, 0.8),
        (
            True,
            "numerically_unresolved",
            "numerically_unresolved",
            np.nan,
            False,
            False,
            0.8,
        ),
        (False, "not_applicable", "unavailable", np.nan, False, False, 0.9),
    ]
    for index, (
        eligible,
        condition,
        gain_status,
        gain,
        saturated,
        underflow,
        score,
    ) in enumerate(cases):
        row = observation(str(index), sscd=score)
        row.update(
            feedback_eligible=eligible,
            candidate_condition_status=condition,
            candidate_gain_status=gain_status,
            candidate_log_probability_gain=gain,
            candidate_condition_margin_rmse=(
                1
                if condition == "estimated_met"
                else (
                    -1
                    if condition == "estimated_not_met"
                    else 0
                    if eligible
                    else np.nan
                )
            ),
            candidate_gain_saturated=saturated,
            candidate_gain_underflow=underflow,
            candidate_variation_l2=float(index) if eligible else np.nan,
            candidate_variation_error_l2=0.1 if eligible else np.nan,
            candidate_conditional_reference_error_l2=1.0,
            candidate_unconditional_reference_error_l2=2.0,
            candidate_log_odds_gain=100.0 if index == 0 else gain,
            candidate_matched_log_probability=-1.0,
            candidate_guided_log_probability=-1.0 + gain,
            feedback_status=(
                "valid" if eligible else "terminal_transition_not_in_Proposition5"
            ),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_feedback_counts_preserve_negative_saturated_and_unresolved_outcomes():
    counts = feedback_counts(feedback_frame())
    assert counts["total_count"] == 7 and counts["eligible_count"] == 6
    assert counts["condition_met_count"] == 2
    assert counts["condition_not_met_count"] == 2
    assert counts["condition_unresolved_count"] == 2
    assert counts["condition_coverage"] == pytest.approx(2 / 6)
    assert (
        counts["gain_positive_count"] == 2
    )  # Includes strict positive with underflowed H=0.
    assert counts["gain_negative_count"] == 1
    assert counts["gain_zero_count"] == 1
    assert counts["gain_numerically_unresolved_count"] == 2
    assert counts["condition_met_gain_positive_count"] == 1
    assert counts["condition_met_gain_numerically_unresolved_count"] == 1
    assert counts["candidate_gain_saturated_count"] == 2
    assert counts["candidate_gain_saturated_fraction"] == pytest.approx(2 / 6)
    assert counts["candidate_gain_underflow_count"] == 1
    assert counts["unplottable_count"] == 2


def test_feedback_timestep_summaries_record_destination_snr_and_negative_high_sscd():
    frame = feedback_frame()
    late = frame.copy()
    late["step_index"], late["snr"], late["destination_snr"] = 1, 10.0, 1000.0
    late["candidate_variation_l2"] *= 10
    summary = feedback_summary(pd.concat([frame, late], ignore_index=True)).set_index(
        "step_index"
    )
    assert summary.loc[0, "source_snr"] == 1
    assert summary.loc[0, "destination_snr"] == 10
    assert summary.loc[1, "source_snr"] == 10
    assert summary.loc[1, "destination_snr"] == 1000
    assert set(summary.high_sscd_negative_gain_count) == {1}
    assert set(summary.candidate_log_probability_gain_count) == {5}
    assert summary.loc[0, "candidate_variation_l2_median"] == pytest.approx(2.5)
    assert summary.loc[1, "candidate_variation_l2_median"] == pytest.approx(25)


def test_zero_coverage_and_no_eligible_rows_are_distinct():
    frame = feedback_frame().iloc[[2, 3]].copy()
    zero = feedback_counts(frame)
    assert zero["condition_coverage"] == 0 and zero["eligible_count"] == 2
    unavailable = feedback_counts(feedback_frame().iloc[[-1]])
    assert (
        unavailable["condition_coverage"] is None and unavailable["eligible_count"] == 0
    )


def test_prompt_initial_improvement_denominator_excludes_only_unavailable_metrics():
    frame = pd.DataFrame(
        [
            observation(seed=0, conditional=1, unconditional=2),
            observation(seed=1, conditional=3, unconditional=2),
            observation(seed=2, conditional=0, unconditional=0),
        ]
    )
    summary = initial_prompt_summary(frame).iloc[0]
    assert summary.sample_count == 3 and summary.improvement_denominator == 3
    assert summary.improved_count == 1 and summary.fraction_improved == pytest.approx(
        1 / 3
    )
    assert summary.relative_target_error_improvement_count == 2
    assert summary.paired_target_error_improvement_rmse_median == 0


def test_optional_tolerance_uses_raw_l2_and_observed_suffix_censoring():
    rows = []
    for seed, values in enumerate(
        ((2.0, 0.4, 0.8, 0.3), (2.0, 2.0, 2.0, 2.0), (0.2, np.nan, 0.3, 0.4))
    ):
        for step, value in enumerate(values):
            rows.append(
                observation(
                    seed=seed, step=step, conditional=value, unconditional=value
                )
            )
    frame = pd.DataFrame(rows)
    assert tolerance_entry_summary(frame, None).empty
    result = tolerance_entry_summary(frame, 1).set_index("seed")
    assert set(result.rmse_tolerance) == {0.5}  # d=4, so raw tau=1 maps to RMSE=.5.
    assert result.loc[0, "first_entry_step"] == 1
    assert result.loc[0, "sustained_entry_step"] == 3
    assert result.loc[1, "first_entry_status"] == "not_reached"
    assert pd.isna(result.loc[1, "first_entry_step"])
    assert result.loc[2, "first_entry_step"] == 0
    assert result.loc[2, "sustained_entry_step"] == 2
    assert result.loc[2, "missing_observation_count"] == 1
    assert result.right_censored.all()
    assert set(result.last_observed_prediction_step) == {3}


@pytest.mark.parametrize("tolerance", [-1, math.inf, math.nan])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="finite nonnegative"):
        tolerance_entry_summary(pd.DataFrame([observation()]), tolerance)


def test_feedback_probability_quantiles_keep_stable_complements_and_underflow():
    frame = feedback_frame().iloc[:4].copy()
    for endpoint in ("matched", "guided"):
        frame[f"candidate_{endpoint}_log_probability"] = [0.0, 0.0, -1000.0, np.nan]
        frame[f"candidate_{endpoint}_log_complement"] = [-100.0, -1000.0, 0.0, np.nan]
    original = frame.copy(deep=True)
    result = feedback_summary(frame).iloc[0]
    for endpoint in ("matched", "guided"):
        assert result[f"candidate_{endpoint}_probability_count"] == 3
        assert result[f"candidate_{endpoint}_complement_count"] == 3
        assert result[f"candidate_{endpoint}_probability_median"] == 1
        complement = result[f"candidate_{endpoint}_complement_median"]
        assert complement > 0  # 1 - rounded_probability would incorrectly give 0.
        assert math.isclose(complement, math.exp(-100), rel_tol=1e-14, abs_tol=0)
        assert result[f"candidate_{endpoint}_log_complement_median"] == -100
        assert result[f"candidate_{endpoint}_log_probability_count"] == 3
    pd.testing.assert_frame_equal(frame, original)
    # Both target/complement underflow can also remain exactly represented zero.
    one = feedback_summary(frame.iloc[[1]]).iloc[0]
    assert one.candidate_guided_probability_median == 1
    assert one.candidate_guided_complement_median == 0
    assert one.candidate_guided_complement_count == 1


@pytest.mark.parametrize(
    "condition,expected",
    [("not_applicable", "not_applicable"), ("unavailable", "unavailable")],
)
def test_availability_separates_missing_reference_from_mathematical_inapplicability(
    condition, expected
):
    from utils.experiments.theory.summaries import applicability_report

    trajectory = feedback_frame().copy()
    trajectory["feedback_eligible"] = False
    trajectory["candidate_condition_status"] = condition
    endpoint = pd.DataFrame(
        [
            dict(
                terminal_clean_structural=False,
                terminal_clean_applicable=False,
                terminal_clean_numeric_within_sensitivity=True,
                terminal_clean_status="nonclean_destination",
            )
        ]
    )
    report = applicability_report(
        pd.DataFrame({"seed": [0, 1]}),
        trajectory,
        endpoint,
        {"source_sample_count": 2},
        {"support_size": 2},
    )
    assert report["posterior_feedback"]["status"] == expected
    assert report["terminal_clean_update"]["status"] == "not_applicable"


def test_terminal_availability_separates_structural_and_numerical_failure():
    from utils.experiments.theory.summaries import applicability_report

    endpoint = pd.DataFrame(
        [
            dict(
                terminal_clean_structural=True,
                terminal_clean_applicable=False,
                terminal_clean_numeric_within_sensitivity=False,
                terminal_clean_status="structurally_clean_numerical_check_separate",
            )
        ]
    )
    report = applicability_report(
        pd.DataFrame({"seed": [0]}),
        feedback_frame(),
        endpoint,
        {"source_sample_count": 1},
        {"support_size": 2},
    )["terminal_clean_update"]
    assert report["status"] == "numerical_inconsistency"
    assert report["structural_applicable_count"] == 1
    assert report["numerical_check_failed_count"] == 1
    assert report["applicable_count"] == 0
