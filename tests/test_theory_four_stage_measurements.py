"""Unexecuted four-stage measurement specifications; no model is required."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.four_stage_measurements import (
    baseline_summary, branch_motion, dose_grid, prompt_shape_summaries,
    response_rows, trajectory_shape,
)


def identities(count=1):
    return pd.DataFrame([{"run_id": "run", "original_index": str(i), "record_id": str(i),
        "target_id": "target", "seed": str(i), "step_index": 0, "terminal_sscd": .9,
        "sscd": .9, "affine_endpoint_tolerance_l2": 1e-10,
        "prompt_utf8_sha256": "prompt"} for i in range(count)])


def payload(slopes, *, discrepancy=0.):
    return {"intercept": torch.zeros((1, len(slopes)), dtype=torch.float64),
            "slopes": torch.tensor([slopes], dtype=torch.float64), "target_atom": 0,
            "endpoint_difference_l2": torch.tensor([discrepancy], dtype=torch.float64),
            "endpoint_construction_method": "independent_deterministic_affine_baseline"}


def test_exact_fixed_dose_grid_has_all_declared_endpoints():
    grid = dose_grid(7.5)
    assert grid == sorted(set(grid))
    assert grid[0] == 0 and grid[-1] == 1
    assert 1 / 7.5 in grid
    assert set(j / 40 for j in range(41)).issubset(grid)
    assert len(dose_grid(2.0)) == 41


def test_large_gap_direction_can_decrease_posterior_and_remains_in_response():
    rows = response_rows(payload([0., 20.]), identities(), 7.5, device="cpu")
    assert rows.s.tolist() == dose_grid(7.5)
    assert rows.iloc[0].log_probability_gain == 0
    assert rows.iloc[-1].log_probability_gain < 0
    assert rows.iloc[-1].log_odds_gain < 0
    assert rows.structural_applicable.all()
    assert len(rows) == len(dose_grid(7.5))
    assert rows.iloc[-1].log_probability_gain != rows.iloc[-1].log_odds_gain


def test_singleton_and_zero_displacement_curves_preserve_zero_gain():
    singleton = response_rows(payload([0.]), identities(), 7.5, device="cpu")
    assert singleton.log_probability_gain.eq(0).all()
    assert singleton.log_odds_gain.isna().all()
    zero = response_rows(payload([0., 0.]), identities(), 7.5, device="cpu")
    assert zero.log_probability_gain.eq(0).all()
    assert zero.log_odds_gain.eq(0).all()


def test_saved_endpoint_disagreement_keeps_reconstructed_segment_qualification():
    rows = response_rows(payload([0., -1.], discrepancy=1.), identities(), 7.5, device="cpu")
    assert rows.endpoint_status.str.contains("discrepancy_recorded").all()
    assert rows.saved_endpoint_discrepancy_l2.eq(1.).all()
    assert rows.iloc[-1].log_probability_gain > 0


def test_branch_motion_preserves_additive_identity_and_target_error_changes():
    mu = torch.tensor([[1., 0., -1.], [2., 2., 2.]], dtype=torch.float64)
    mc = torch.tensor([[2., 3., -2.], [2., 2., 2.]], dtype=torch.float64)
    next_mu = mu + torch.tensor([[.5, -.25, 1.], [-1., 0., 1.]], dtype=torch.float64)
    next_mc = mc + torch.tensor([[-1., .5, .25], [0., -1., 0.]], dtype=torch.float64)
    result = branch_motion(mu, mc, next_mu, next_mc, torch.zeros(3))
    summed = result["gap_motion_conditional"] + result["gap_motion_unconditional"] + result["gap_motion_quadratic"]
    assert torch.allclose(summed, result["gap_squared_change"], atol=1e-14, rtol=1e-14)
    assert torch.all(result["identity_residual"].abs() <= result["identity_tolerance"])
    assert result["gap_motion_conditional"][1] == 0
    assert result["gap_motion_unconditional"][1] == 0
    # Agreement away from the target is not target recovery.
    assert torch.linalg.vector_norm(mc[1] - mu[1]) == 0
    assert torch.linalg.vector_norm(mc[1]) > 0
    weights = torch.tensor([.25, .75], dtype=torch.float64)
    assert (weights * summed).sum() == pytest.approx((weights * result["gap_squared_change"]).sum().item())


@pytest.mark.parametrize("curve, expected", [
    ([0., 0., 0.], "flat_or_numerically_unresolved_plateau"),
    ([3., 2., 1.], "nonincreasing_within_declared_resolution"),
    ([1., 2., 3.], "nondecreasing_within_declared_resolution"),
    ([1., 3., 3., 2.], "rise_then_fall_not_a_strict_unimodality_claim"),
    ([1., 3., 2., 4., 1.], "complex_multi_direction_curve"),
    ([1., math.nan, 2.], "incomplete_raw_curve"),
])
def test_unsmoothed_shapes_retain_flat_complex_and_incomplete_cases(curve, expected):
    result = trajectory_shape(curve, len(curve))
    assert result["shape_status"] == expected
    if expected in {"flat_or_numerically_unresolved_plateau", "incomplete_raw_curve"}:
        assert math.isnan(result["first_peak_step"])
    if curve == [1., 3., 3., 2.]:
        assert result["first_peak_step"] == 1
        assert result["peak_ties_json"] == "[1, 2]"
        assert result["total_upward_variation_rmse"] == 2
        assert result["total_downward_variation_rmse"] == 1


def test_missing_interior_and_single_step_rules_are_explicit():
    incomplete = trajectory_shape([1., 2.], 3)
    assert not incomplete["shape_complete"]
    assert math.isnan(incomplete["maximum_gap_rmse"])
    single = trajectory_shape([2.], 1)
    assert single["shape_complete"] and single["shape_flat"]
    assert single["total_upward_variation_rmse"] == 0
    assert math.isnan(single["first_peak_step"])


def baseline_inputs(spread=2.):
    probes = pd.DataFrame({"run_id": ["run", "run"], "seed": [0, 1], "step_index": [0, 0],
                           "learned_mean_error_rmse": [3., 4.]})
    repeats = pd.DataFrame({"run_id": ["run"] * 4, "seed": [0, 0, 1, 1], "original_index": [0, 1, 0, 1],
        "record_id": ["a", "b", "a", "b"], "initial_gaussian_bank_match": [True] * 4,
        "initial_unconditional_vector_sha256": ["same", "same", "first", "different"],
        "initial_unconditional_mean_error_rmse": [3., 3., 4., 4.25], "mean_norm_l2": [2.] * 4,
        "mean_norm_rmse": [1.] * 4})
    atoms = pd.DataFrame({"reference_scale_rmse": [spread], "reference_law_hash": ["law"]})
    return probes, repeats, atoms


def test_baseline_counts_unique_seeds_and_discloses_prompt_disagreements():
    summary, audit = baseline_summary(*baseline_inputs())
    row = summary.iloc[0]
    assert row.unique_seed_count == 2
    assert row.baseline_error_rmse == pytest.approx(math.sqrt((3**2 + 4**2) / 2))
    assert row.relative_baseline_error == pytest.approx(row.baseline_error_rmse / 2)
    assert row.disagreement_count == 1
    assert audit.canonical_record_id.tolist() == ["a", "a"]
    assert audit.unconditional_disagreement.tolist() == [False, True]


def test_zero_reference_spread_has_undefined_relative_baseline_and_duplicate_probes_fail():
    probes, repeats, atoms = baseline_inputs(0.)
    summary, _ = baseline_summary(probes, repeats, atoms)
    assert summary.relative_baseline_error.isna().all()
    assert summary.relative_status.iloc[0] == "undefined_zero_reference_spread"
    with pytest.raises(TheoryError, match="unique seed"):
        baseline_summary(pd.concat([probes, probes.iloc[:1]], ignore_index=True), repeats, atoms)


def test_mixed_outcome_prompt_remains_paired_in_shape_summaries():
    rows = []
    for index, (sscd, curve) in enumerate(((.9, [1., 3., 2.]), (.5, [3., 2., 1.]))):
        rows.append({**identities(1).iloc[0].to_dict(), "seed": str(index), "sscd": sscd,
                     **trajectory_shape(curve, 3)})
    grouped, paired = prompt_shape_summaries(pd.DataFrame(rows))
    assert len(grouped) == 2
    assert grouped.prompt_utf8_sha256.nunique() == 1
    assert len(paired) == 1
    assert paired.peak_rise_rmse_high_minus_lower.iloc[0] == 2


def test_record_reduction_keeps_last_current_prediction_and_weighted_reference_geometry():
    from types import SimpleNamespace
    from utils.experiments.theory.evidence_reduce import KEYS, _join
    from utils.experiments.theory.four_stage_measurements import measure_record
    from utils.experiments.theory.reference_law import reference_law_from_atoms
    law = reference_law_from_atoms([torch.tensor([0., 0.]), torch.tensor([2., 0.])],
                                   ["target", "other"], weights=[.25, .75])
    record = SimpleNamespace(metadata={"seeds": [0, 1], "stored_prediction_type": "epsilon"})
    states = torch.full((2, 4, 2), 2., dtype=torch.float64)
    epsilon_u = torch.zeros((2, 3, 2), dtype=torch.float64)
    epsilon_c = torch.zeros_like(epsilon_u)
    target = torch.zeros(2, dtype=torch.float64)
    def coefficients(step):
        return SimpleNamespace(alpha=.5, sigma=.75, A=.25, kappa=.5, affine=True,
            noise_std=0., timestep=9 - 3 * step, destination_timestep=6 - 3 * step,
            destination_alpha=.8, destination_sigma=.6 if step < 2 else 0., snr=.25 / .75**2)
    rows = []
    for step in range(3):
        for seed in range(2):
            row = identities(2).iloc[seed].to_dict()
            row.update(step_index=step, timestep=9 - step * 3, destination_timestep=6 - step * 3,
                       snr=.25 / .75**2, candidate_target_atom_id="target")
            rows.append(row)
    base = pd.DataFrame(rows)
    core = {"trajectory": base, "initial": base.loc[base.step_index.eq(0)].copy(), "matched_updates": base.copy()}
    reduced = measure_record((states, epsilon_u, epsilon_c, target), record, law,
        SimpleNamespace(coefficients=coefficients), {"num_inference_steps": 3, "guidance_scale": 7.5},
        core, gaussian_bank=states[:, 0], batch_size=1)
    trajectory = reduced["trajectory_four_stage"]
    assert trajectory.step_index.max() == 2
    assert len(trajectory) == 6 and len(reduced["branch_motion"]) == 4
    assert trajectory.gap_rmse.eq(0).all()
    assert trajectory.joint_target_error_rmse.gt(0).all()
    assert trajectory.normalized_progress.min() == 0 and trajectory.normalized_progress.max() == 1
    assert set(trajectory).intersection(base.columns) == set(KEYS)
    assert len(_join(base, trajectory)) == len(base)
    assert reduced["initial_four_stage"].mean_norm_l2.tolist() == pytest.approx([1.5, 1.5])
    assert reduced["initial_four_stage"].reference_spread_rmse.tolist() == pytest.approx([math.sqrt(.375)] * 2)
    assert reduced["trajectory_shapes"].shape_flat.all()
    assert reduced["feedback_response"].log_probability_gain.eq(0).all()


def test_response_saturation_does_not_send_whole_dose_grid_to_decimal(monkeypatch):
    from utils.experiments.theory import four_stage_measurements as module
    item = payload([0., -1.])
    item["intercept"] = torch.tensor([[0., -1000.]], dtype=torch.float64)
    def forbidden(*args, **kwargs):
        raise AssertionError("Saturation alone must not invoke Decimal")
    monkeypatch.setattr(module, "Directed", forbidden)
    rows = response_rows(item, identities(), 7.5, device="cpu")
    assert rows.log_probability_gain.eq(0).all()
    assert rows.log_odds_gain.iloc[-1] > 0
    assert not rows.response_fallback_selected.any()
    assert rows.response_dose_decimal_products.eq(0).all()


def test_response_zero_budget_does_not_construct_decimal_vectors(monkeypatch):
    from utils.experiments.theory import four_stage_measurements as module
    from utils.experiments.theory.numerical_refinement import NumericalPolicy
    monkeypatch.setattr(module, "_gain_retry_needed", lambda *args, **kwargs: True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero-budget response must not construct Decimal arithmetic")
    monkeypatch.setattr(module, "Directed", forbidden)
    rows = response_rows(payload([0., 1.]), identities(), 7.5, device="cpu",
                         policy=NumericalPolicy(max_decimal_products=0))
    assert rows.response_dose_decimal_products.eq(0).all()
    assert rows.response_fallback_status.eq("budget_unavailable_keep_finite_assessment").all()
    assert np.isfinite(rows.log_probability_gain).all()


def test_response_decimal_budget_is_shared_across_the_whole_seed_curve(monkeypatch):
    from utils.experiments.theory import four_stage_measurements as module
    from utils.experiments.theory.numerical_refinement import NumericalPolicy
    from utils.experiments.theory.numerical_intervals import Interval
    calls = []
    monkeypatch.setattr(module, "_gain_retry_needed", lambda *args, **kwargs: True)
    def artificial_enclosure(arithmetic, intercept, slopes, target):
        calls.append(True)
        arithmetic.charge(700)
        return {"H": Interval.point(.1), "G": Interval.point(.2), "gain_sign": "positive"}
    monkeypatch.setattr(module, "interval_gain", artificial_enclosure)
    rows = response_rows(payload([0., 1.]), identities(), 7.5, device="cpu",
                         policy=NumericalPolicy(max_decimal_products=1000))
    assert len(calls) == 1
    assert rows.response_dose_decimal_products.sum() <= 1000
    assert rows.response_curve_decimal_products.iloc[-1] == rows.response_dose_decimal_products.sum()
    assert rows.response_fallback_status.eq("budget_unavailable_keep_finite_assessment").any()
    assert rows.response_arithmetic_scope.str.contains("not_original_vector_construction").all()
    # The refined G's sign cannot be substituted for H's plotted magnitude.
    assert rows.log_probability_gain.iloc[-1] < 0


def test_response_corrects_only_its_own_contradicted_magnitude_and_retains_prior_values(monkeypatch):
    from utils.experiments.theory import four_stage_measurements as module
    from utils.experiments.theory.numerical_intervals import Interval
    from utils.experiments.theory.numerical_refinement import NumericalPolicy
    monkeypatch.setattr(module, "_gain_retry_needed", lambda *args, **kwargs: True)
    def instrumented_interval(arithmetic, intercept, slopes, target):
        arithmetic.charge(700)
        return {"H": Interval.point(.25), "G": Interval.point(2.), "gain_sign": "positive"}
    monkeypatch.setattr(module, "interval_gain", instrumented_interval)
    result = response_rows(payload([0., 1.]), identities(), 7.5, device="cpu",
                           policy=NumericalPolicy(max_decimal_products=1000))
    first = result.iloc[0]
    assert first.log_probability_gain == .25
    assert first.log_odds_gain == 2.
    assert first.log_probability_gain_float64_before == 0
    assert first.log_odds_gain_float64_before == 0
    assert first.log_probability_gain != first.log_odds_gain
    assert first.response_arithmetic_scope.endswith("not_original_vector_construction")


@pytest.mark.parametrize("guidance", [0., .5, 1.])
def test_unsupported_guidance_keeps_an_in_range_inapplicability_grid(guidance):
    from utils.experiments.theory.four_stage_measurements import unavailable_response
    rows = unavailable_response(identities(), guidance, "outside_guidance_domain")
    assert rows.s.between(0, 1).all()
    assert len(rows) == 41
    assert not rows.structural_applicable.any()
    assert rows.log_probability_gain.isna().all()


def test_four_stage_collection_identity_ignores_device_and_manifest_receipt_changes():
    from copy import deepcopy
    from utils.experiments.theory.four_stage_reduce import collection_identity
    from utils.common.io import canonical_hash
    result = {"directory": "/first/path", "files": {"trajectory": {"path": "/first/path/trajectory.parquet", "sha256": "saved-scalars", "rows": 20}},
              "logical_joins": {"trajectory_metrics": {"base": "trajectory", "supplement": "extra", "keys": ["seed"], "how": "additive"}},
              "provenance": {"analysis_hash": "fixed-base-science", "execution": {"devices": ["cuda:0", "cuda:1"]},
                             "base_manifest_sha256": "first-run-receipt"}}
    resumed = deepcopy(result)
    resumed["directory"] = "/moved/path"
    resumed["files"]["trajectory"]["path"] = "/moved/path/trajectory.parquet"
    resumed["provenance"]["execution"] = {"devices": [], "resumed": True}
    resumed["provenance"]["base_manifest_sha256"] = "resume-receipt"
    recipe, tasks = {"version": "fixed-four-stage"}, ["record-0", "record-1"]
    assert canonical_hash(collection_identity(result, recipe, tasks)) == canonical_hash(collection_identity(resumed, recipe, tasks))
    resumed["files"]["trajectory"]["sha256"] = "changed-measurements"
    assert canonical_hash(collection_identity(result, recipe, tasks)) != canonical_hash(collection_identity(resumed, recipe, tasks))
