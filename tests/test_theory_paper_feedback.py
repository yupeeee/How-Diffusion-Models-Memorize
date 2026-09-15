"""Paper feedback checks use the same finite-law observations and populations."""

import numpy as np
import pandas as pd
import pytest

from utils.experiments.theory.paper_feedback import (
    STEMS,
    audit_feedback_rows,
    build_feedback_plot_inputs,
    fraction_summaries,
    margin_coverage_audit,
    prompt_balanced_weights,
    structural_feedback_eligibility,
)


def rows(gains=(1.0,), *, scores=None):
    result = []
    for seed, gain in enumerate(gains):
        result.append(
            dict(
                run_id="run",
                original_index=4,
                record_id="record",
                target_id="target",
                seed=seed,
                step_index=0,
                snr=0.01,
                terminal_sscd=0.8 if scores is None else scores[seed],
                destination_sigma=0.9,
                destination_alpha=0.1,
                kappa=0.2,
                candidate_feedback_eligible=True,
                candidate_feedback_status="matched_endpoint_available",
                candidate_log_probability_gain=float(
                    np.logaddexp(0, 0) - np.logaddexp(0, -gain)
                ),
                candidate_log_odds_gain=gain,
                candidate_normalized_log_odds_gain=gain / 2,
                candidate_matched_log_odds=0.0,
                candidate_guided_log_odds=gain,
                candidate_gain_arithmetic_tolerance=1e-10,
                candidate_endpoint_arithmetic_tolerance=2e-10,
                candidate_endpoint_slope_C0=gain + abs(gain) / 2,
                candidate_endpoint_slope_C1=gain - abs(gain) / 2,
                candidate_gain_status="positive"
                if gain > 0
                else "negative"
                if gain < 0
                else "arithmetic_zero",
                candidate_profile_class="increasing"
                if gain > 0
                else "decreasing"
                if gain < 0
                else "flat",
                candidate_endpoint_bracket_status="consistent",
                candidate_gain_saturated=False,
                candidate_gain_underflow=False,
            )
        )
    return pd.DataFrame(result)


def test_prompt_population_includes_unresolved_and_balances_unequal_seed_counts():
    frame = rows((1, 1, -1))
    frame.loc[2, ["record_id", "original_index"]] = ["second", 5]
    frame.loc[1, "candidate_gain_status"] = "numerically_unresolved"
    frame.loc[1, ["candidate_endpoint_slope_C0", "candidate_endpoint_slope_C1"]] = (
        np.nan
    )
    positive, increasing = fraction_summaries(frame)
    assert positive.iloc[0].denominator_weight == 2
    assert positive.iloc[0].fraction == 0.25
    assert positive.iloc[0].upper_fraction == 0.5
    assert positive.iloc[0].unresolved_count == 1
    pd.testing.assert_series_equal(
        positive.denominator_weight, increasing.denominator_weight
    )
    assert increasing.iloc[0].fraction == 0.25
    assert increasing.iloc[0].upper_fraction == 0.5


def test_numerical_failure_stays_eligible_but_terminal_target_and_competitor_do_not():
    frame = rows((1, 1, 1, 1))
    frame["candidate_feedback_eligible"] = False
    frame["candidate_feedback_status"] = [
        "invalid_nonfinite_input",
        "terminal_or_nonaffine_or_nonpositive_destination",
        "missing_candidate_target",
        "no_distinct_competitor",
    ]
    frame.loc[0, "candidate_gain_status"] = "not_applicable"
    frame.loc[0, "candidate_log_odds_gain"] = np.nan
    assert structural_feedback_eligibility(frame).tolist() == [
        True,
        False,
        False,
        False,
    ]
    summary, _ = fraction_summaries(frame)
    assert summary.iloc[0].eligible_count == 1
    assert summary.iloc[0].excluded_count == 3
    assert summary.iloc[0].fraction == 0
    assert summary.iloc[0].upper_fraction == 1


def test_unknown_inapplicability_and_duplicate_full_identity_fail_closed():
    frame = rows()
    frame["candidate_feedback_eligible"] = False
    frame["candidate_feedback_status"] = "unknown_reason"
    with pytest.raises(ValueError, match="Unclassified"):
        structural_feedback_eligibility(frame)
    with pytest.raises(ValueError, match="duplicate"):
        audit_feedback_rows(pd.concat([rows(), rows()]))


def test_overlapping_prompts_are_balanced_within_each_sscd_group():
    frame = rows((1, -1), scores=(0.9, 0.7))
    summary, _ = fraction_summaries(frame)
    assert summary.group.tolist() == ["SSCD > 0.75", "SSCD <= 0.75"]
    assert summary.denominator_weight.tolist() == [1, 1]
    assert summary.fraction.tolist() == [1, 0]
    assert summary.eligible_prompt_count.tolist() == [1, 1]


def test_full_prompt_identity_and_step_are_preserved_in_weights():
    frame = pd.concat([rows((1, 1)), rows((1, 1))], ignore_index=True)
    frame.loc[2:, "step_index"] = 1
    assert prompt_balanced_weights(frame).tolist() == [0.5] * 4
    frame.loc[1, "run_id"] = "other_run"
    assert prompt_balanced_weights(frame).tolist() == [1, 1, 0.5, 0.5]


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("candidate_log_probability_gain", -0.1, "reliable_H_G_sign_disagreement"),
        (
            "candidate_endpoint_slope_C1",
            2.0,
            "C1_exceeds_G_outside_arithmetic_allowance",
        ),
        (
            "candidate_endpoint_slope_C0",
            0.0,
            "G_exceeds_C0_outside_arithmetic_allowance",
        ),
        (
            "candidate_gain_status",
            "negative",
            "saved_resolved_gain_sign_disagrees_with_G",
        ),
    ],
)
def test_endpoint_contradictions_block_and_keep_offending_ids(column, value, reason):
    frame = rows()
    frame[column] = value
    frame["candidate_source_sensitivity_l2"] = 1e100
    audit, offending = audit_feedback_rows(frame)
    assert audit["status"] == "blocked"
    assert reason in offending.reason.tolist()
    assert offending.record_id.eq("record").all()
    assert offending.seed.eq(0).all()


def test_tiny_stable_H_sign_is_checked_with_propagated_not_fixed_absolute_budget():
    frame = rows()
    frame["candidate_matched_log_odds"] = 500
    frame["candidate_guided_log_odds"] = 501
    frame["candidate_log_probability_gain"] = -1e-218
    audit, offending = audit_feedback_rows(frame)
    assert audit["status"] == "blocked"
    assert "reliable_H_G_sign_disagreement" in offending.reason.tolist()


def test_H_underflow_retains_G_sign_and_roundoff_boundary_does_not_block():
    frame = rows()
    frame["candidate_log_probability_gain"] = 0
    frame["candidate_matched_log_odds"] = 1000
    frame["candidate_guided_log_odds"] = 1001
    frame["candidate_endpoint_slope_C1"] = 1 + 1e-10
    audit, offending = audit_feedback_rows(frame)
    assert audit["status"] == "passed"
    assert audit["retained_H_underflow_with_resolved_G"] == 1
    assert offending.empty
    fractions, _ = fraction_summaries(frame)
    assert fractions.iloc[0].fraction == 1


def test_flat_zero_is_not_increasing_or_ambiguous():
    fractions, profiles = fraction_summaries(rows((0,)))
    assert fractions.iloc[0].zero_count == 1
    assert profiles.iloc[0].flat_count == 1
    assert profiles.iloc[0].fraction == profiles.iloc[0].upper_fraction == 0


def test_PF08_increasing_uses_resolved_C1_instead_of_gain_or_raw_profile_name():
    frame = rows((1, 1))
    frame.loc[0, "candidate_endpoint_slope_C1"] = -0.1
    frame.loc[0, "candidate_profile_class"] = "interior_peak"
    frame.loc[1, "candidate_endpoint_slope_C1"] = 1e-11
    _, profiles = fraction_summaries(frame)
    assert profiles.iloc[0].fraction == 0
    assert profiles.iloc[0].upper_fraction == 0.5
    assert profiles.iloc[0].interior_peak_count == 1


def test_margin_strict_nonnegative_and_uncertainty_status_have_same_denominator():
    frame = rows((-1, 0, 1))
    prefix = "candidate_margin_projected_variation"
    frame[prefix + "_l2"] = [1e-12, 0, 2]
    frame[prefix + "_uncertainty_l2"] = [1, 1, 0.1]
    frame[prefix + "_status"] = [
        "numerically_unresolved",
        "numerically_unresolved",
        "positive",
    ]
    coverage, details = margin_coverage_audit(frame)
    record = coverage.loc[
        (coverage.group == "All") & (coverage.margin == "projected_variation")
    ].iloc[0]
    assert record.estimated_strict_positive_fraction == pytest.approx(2 / 3)
    assert record.estimated_nonnegative_fraction == 1
    assert record.resolved_positive_fraction == pytest.approx(1 / 3)
    assert record.stable_gain_positive_fraction == pytest.approx(1 / 3)
    assert record.estimated_positive_negative_gain_count == 1
    assert record.resolved_positive_negative_gain_count == 0
    assert details.loc[details.group.eq("All"), "seed"].tolist() == [0]
    assert not record.formal_certificate
    assert audit_feedback_rows(frame)[0]["status"] == "passed"


def test_only_an_explicit_certified_strict_margin_claim_blocks_on_negative_gain():
    frame = rows((-1,))
    prefix = "candidate_margin_original"
    frame[prefix + "_l2"], frame[prefix + "_uncertainty_l2"] = 1, 0.1
    frame[prefix + "_status"] = "positive"
    assert audit_feedback_rows(frame)[0]["status"] == "passed"
    frame[prefix + "_certified"] = True
    assert audit_feedback_rows(frame)[0]["status"] == "blocked"
    frame[prefix + "_l2"] = 0
    assert audit_feedback_rows(frame)[0]["status"] == "passed"


def test_endpoint_only_missing_integration_is_usable_and_snapshot_is_exactly_zero():
    frame = rows((1, -1))
    frames, metadata = build_feedback_plot_inputs(
        {"trajectory": frame}, config={"guidance_scale": 7.5}
    )
    assert metadata["audit"]["status"] == "passed"
    assert metadata["audit"]["coverage_audit"]["status"] == "integration_unavailable"
    assert metadata["figures"][STEMS["PF03"]]["status"] == "complete"
    assert metadata["figures"][STEMS["PF03"]]["empty_groups"] == ["SSCD <= 0.75"]
    assert frames[STEMS["PF04"]].x.tolist() == [0, 0]
    assert frames[STEMS["PF04"]].y.tolist() == [1, -1]
    frame.step_index = 1
    frames, _ = build_feedback_plot_inputs({"trajectory": frame}, config={})
    assert frames[STEMS["PF04"]].empty


def test_endpoint_failure_gates_all_feedback_figures_but_retains_measurements():
    frame = rows()
    frame["candidate_log_probability_gain"] = -0.5
    frames, metadata = build_feedback_plot_inputs({"trajectory": frame}, config={})
    assert all(item["status"] == "blocked" for item in metadata["figures"].values())
    assert len(frames[STEMS["PF04"]]) == 1
    assert (
        len(metadata["auxiliary_tables"]["posterior_feedback_endpoint_offenders"]) == 1
    )


def test_SNR_mixing_or_same_seed_SSCD_mismatch_are_audit_failures():
    frame = rows((1, 1))
    frame.loc[1, "snr"] = 0.02
    with pytest.raises(ValueError, match="SNR"):
        fraction_summaries(frame)
    frame = pd.concat([rows(), rows()], ignore_index=True)
    frame.loc[1, ["step_index", "terminal_sscd"]] = [1, 0.1]
    audit, offenders = audit_feedback_rows(frame)
    assert audit["status"] == "blocked"
    assert offenders.reason.eq("same_seed_SSCD_changes_across_steps").all()


def test_saved_endpoint_odds_and_probability_differences_are_independently_checked():
    frame = rows()
    frame["candidate_guided_log_odds"] = -1
    audit, offending = audit_feedback_rows(frame)
    assert audit["status"] == "blocked"
    assert "G_disagrees_with_saved_endpoint_log_odds" in offending.reason.tolist()
    frame = rows()
    frame["candidate_matched_log_probability"] = -1
    frame["candidate_guided_log_probability"] = -2
    audit, offending = audit_feedback_rows(frame)
    assert audit["status"] == "blocked"
    assert (
        "H_disagrees_with_saved_endpoint_log_probabilities" in offending.reason.tolist()
    )


def test_structurally_inapplicable_feedback_has_named_metadata_and_keeps_exclusions():
    frame = rows()
    frame["destination_sigma"] = 0
    frame["candidate_feedback_eligible"] = False
    frame["candidate_feedback_status"] = (
        "terminal_or_nonaffine_or_nonpositive_destination"
    )
    frames, metadata = build_feedback_plot_inputs({"trajectory": frame}, config={})
    assert all(
        entry["status"] == "not_applicable" for entry in metadata["figures"].values()
    )
    assert "terminal_or_nonaffine" in metadata["figures"][STEMS["PF03"]]["reason"]
    assert frames[STEMS["PF03"]].iloc[0].excluded_count == 1
    assert np.isnan(frames[STEMS["PF03"]].iloc[0].fraction)


def test_all_zero_displacements_only_disable_normalized_gain():
    frame = rows((0,))
    frame["candidate_direction_normalization_status"] = "zero_displacement"
    frame["candidate_normalized_log_odds_gain"] = np.nan
    _, metadata = build_feedback_plot_inputs({"trajectory": frame}, config={})
    assert metadata["figures"][STEMS["PF02"]]["status"] == "not_applicable"
    assert metadata["figures"][STEMS["PF03"]]["status"] == "complete"
