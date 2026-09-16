"""Unexecuted four-stage measurement specifications; no model is required."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical backend required")

from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.four_stage_measurements import (
    baseline_summary, branch_motion, corollary3_guidance_fit, dose_grid, prompt_shape_summaries,
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


@requires_cuda
def test_large_gap_direction_can_decrease_posterior_and_remains_in_response():
    rows = response_rows(payload([0., 20.]), identities(), 7.5, device="cuda")
    assert rows.s.tolist() == dose_grid(7.5)
    assert rows.iloc[0].log_probability_gain == 0
    assert rows.iloc[-1].log_probability_gain < 0
    assert rows.iloc[-1].log_odds_gain < 0
    assert rows.structural_applicable.all()
    assert len(rows) == len(dose_grid(7.5))
    assert rows.iloc[-1].log_probability_gain != rows.iloc[-1].log_odds_gain


@requires_cuda
def test_singleton_and_zero_displacement_curves_preserve_zero_gain():
    singleton = response_rows(payload([0.]), identities(), 7.5, device="cuda")
    assert singleton.log_probability_gain.eq(0).all()
    assert singleton.log_odds_gain.isna().all()
    zero = response_rows(payload([0., 0.]), identities(), 7.5, device="cuda")
    assert zero.log_probability_gain.eq(0).all()
    assert zero.log_odds_gain.eq(0).all()


@requires_cuda
def test_saved_endpoint_disagreement_keeps_reconstructed_segment_qualification():
    rows = response_rows(payload([0., -1.], discrepancy=1.), identities(), 7.5, device="cuda")
    assert rows.endpoint_status.str.contains("discrepancy_recorded").all()
    assert rows.saved_endpoint_discrepancy_l2.eq(1.).all()
    assert rows.iloc[-1].log_probability_gain > 0


def test_corollary3_guidance_fit_rejects_cpu_arithmetic():
    with pytest.raises(ValueError, match="require CUDA"):
        corollary3_guidance_fit(torch.zeros(1, 2), torch.ones(1, 2), torch.ones(2), torch.zeros(2), 7.5)


@requires_cuda
def test_corollary3_guidance_fit_is_signed_and_retains_off_axis_residual():
    mean = torch.tensor([1., -2.], device="cuda", dtype=torch.float64)
    target = torch.tensor([3., -2.], device="cuda", dtype=torch.float64)
    response = torch.tensor([[4., 3.], [-2.5, -4.]], device="cuda", dtype=torch.float64)
    unconditional = mean + torch.tensor([[.75, .5], [.75, .5]], device="cuda", dtype=torch.float64)
    conditional = unconditional + (mean + response - unconditional) / 7.5
    result = corollary3_guidance_fit(unconditional, conditional, target, mean, 7.5)
    torch.testing.assert_close(result["cor3_guidance_fit"], torch.tensor([2., -1.25], device="cuda", dtype=torch.float64))
    torch.testing.assert_close(result["cor3_guidance_fit_residual_rmse"],
                               torch.tensor([3., 4.], device="cuda", dtype=torch.float64) / math.sqrt(2))
    assert result["cor3_guidance_fit_direction_l2"].item() == 2.
    assert result["cor3_guidance_fit_applicable"].all()
    assert result["cor3_guidance_fit_status"] == ["measured", "measured"]
    assert result["cor3_guidance_fit_qa"] == ["consistent_float64_estimate"] * 2
    assert (result["cor3_guidance_fit_orthogonality_l2"].abs()
            <= result["cor3_guidance_fit_orthogonality_tolerance_l2"]).all()
    assert (result["cor3_guidance_fit_pythagorean_residual_scaled"].abs()
            <= result["cor3_guidance_fit_pythagorean_tolerance_scaled"]).all()
    assert result["cor3_guidance_fit"].device.type == "cuda"
    assert result["cor3_guidance_fit"].dtype == torch.float64


@requires_cuda
def test_corollary3_guidance_fit_changes_with_selected_mean_without_zero_substitution():
    unconditional = torch.tensor([[0., 0.]], device="cuda", dtype=torch.float64)
    conditional = torch.tensor([[2., 1.]], device="cuda", dtype=torch.float64)
    target = torch.tensor([2., 0.], device="cuda", dtype=torch.float64)
    first = corollary3_guidance_fit(unconditional, conditional, target, torch.zeros_like(target), 2.)
    selected_mean = torch.tensor([1., 1.], device="cuda", dtype=torch.float64)
    second = corollary3_guidance_fit(unconditional, conditional, target, selected_mean, 2.)
    assert first["cor3_guidance_fit"].item() == pytest.approx(2.)
    assert second["cor3_guidance_fit"].item() == pytest.approx(1.)
    assert first["cor3_guidance_fit_residual_rmse"].item() == pytest.approx(math.sqrt(2))
    assert second["cor3_guidance_fit_residual_rmse"].item() == pytest.approx(2.)


@requires_cuda
def test_corollary3_full_guided_fit_is_not_injection_projection():
    mean = torch.tensor([1., 0.], device="cuda", dtype=torch.float64)
    target = torch.tensor([3., 0.], device="cuda", dtype=torch.float64)
    unconditional = torch.tensor([[4., 1.]], device="cuda", dtype=torch.float64)
    conditional = torch.tensor([[5., 2.]], device="cuda", dtype=torch.float64)
    result = corollary3_guidance_fit(unconditional, conditional, target, mean, 3.)
    assert result["cor3_guidance_fit"].item() == pytest.approx(3.)
    direction = target - mean
    injection_fit = ((3. * (conditional - unconditional)) * direction).sum(1) / direction.square().sum()
    assert injection_fit.item() == pytest.approx(1.5)
    assert not torch.allclose(result["cor3_guidance_fit"], injection_fit)


@requires_cuda
@pytest.mark.parametrize("case, status, qa", [
    ("zero", "exact_zero_target_direction", "not_applicable"),
    ("unresolved", "arithmetic_unresolved_target_direction", "numerically_unresolved"),
    ("nonfinite_mean", "nonfinite_target_or_mean", "not_applicable"),
    ("nonfinite_guidance", "nonfinite_guidance", "not_applicable"),
])
def test_corollary3_guidance_fit_preserves_explicit_degenerate_directions(case, status, qa):
    mean = torch.tensor([1., 0.], device="cuda", dtype=torch.float64)
    target = mean.clone()
    guidance = 7.5
    if case == "unresolved":
        target[0] += torch.finfo(torch.float64).eps
    elif case == "nonfinite_mean":
        mean[0] = torch.nan
    elif case == "nonfinite_guidance":
        target[0] = 2.
        guidance = math.inf
    unconditional = torch.zeros((2, 2), device="cuda", dtype=torch.float64)
    result = corollary3_guidance_fit(unconditional, unconditional + 1., target, mean, guidance)
    assert not result["cor3_guidance_fit_applicable"].any()
    assert torch.isnan(result["cor3_guidance_fit"]).all()
    assert torch.isnan(result["cor3_guidance_fit_residual_rmse"]).all()
    assert result["cor3_guidance_fit_status"] == [status] * 2
    assert result["cor3_guidance_fit_qa"] == [qa] * 2
    if case == "unresolved":
        assert 0 < result["cor3_guidance_fit_direction_l2"] <= result["cor3_guidance_fit_direction_tolerance_l2"]


@requires_cuda
def test_corollary3_guidance_fit_invalid_seed_does_not_fabricate_a_zero_coefficient():
    unconditional = torch.tensor([[0., 0.], [math.nan, 0.]], device="cuda", dtype=torch.float64)
    conditional = torch.ones_like(unconditional)
    target = torch.tensor([2., 0.], device="cuda", dtype=torch.float64)
    mean = torch.tensor([.5, 0.], device="cuda", dtype=torch.float64)
    result = corollary3_guidance_fit(unconditional, conditional, target, mean, 2.)
    assert result["cor3_guidance_fit_applicable"].tolist() == [True, False]
    assert result["cor3_guidance_fit_status"] == ["measured", "nonfinite_clean_branch"]
    assert result["cor3_guidance_fit_qa"] == ["consistent_float64_estimate", "not_applicable"]
    assert torch.isnan(result["cor3_guidance_fit"][1])


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


@requires_cuda
@pytest.mark.parametrize("use_reference_mean", [False, True])
def test_record_reduction_keeps_last_current_prediction_and_weighted_reference_geometry(use_reference_mean):
    from types import SimpleNamespace
    from utils.experiments.theory.evidence_reduce import KEYS, _join
    from utils.experiments.theory.four_stage_measurements import measure_record
    from utils.experiments.theory.reference_law import reference_law_from_atoms
    law = reference_law_from_atoms([torch.tensor([0., 0.]), torch.tensor([2., 0.])],
                                   ["target", "other"], weights=[.25, .75], device="cuda")
    selected = torch.tensor([4., 1.], dtype=torch.float64, device="cuda") if use_reference_mean else law.mean_vector
    if use_reference_mean:
        from utils.experiments.theory.reference_law import ReferenceLaw
        from utils.experiments.theory.support import _tensor_hash
        receipt = {"source": "initial_unconditional_reference_monte_carlo", "vector_sha256": _tensor_hash(selected), "sample_count": 10000}
        law = ReferenceLaw(law.support, law.metadata | {"theory_mean": receipt}, "selected-mean-fixture", selected)
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
    initial = reduced["initial_four_stage"]
    assert initial.mean_norm_l2.tolist() == pytest.approx([math.sqrt(17.) if use_reference_mean else 1.5] * 2)
    assert initial.bank_mean_norm_l2.tolist() == pytest.approx([1.5] * 2)
    assert initial.mean_offset_l2.tolist() == pytest.approx([math.sqrt(7.25) if use_reference_mean else 0.] * 2)
    assert initial.reference_spread_rmse.tolist() == pytest.approx([math.sqrt(.375)] * 2)
    assert initial.comparison_scale_rmse.tolist() == pytest.approx([2. if use_reference_mean else math.sqrt(.375)] * 2)
    assert initial.step_index.eq(0).all() and len(initial) == 2
    assert initial.cor3_guidance_fit_applicable.all()
    assert initial.cor3_guidance_fit_qa.eq("consistent_float64_estimate").all()
    direction = target.to(device="cuda") - selected
    guided = torch.full_like(selected, 4.)
    expected_fit = float(((guided - selected) * direction).sum() / direction.square().sum())
    assert initial.cor3_guidance_fit.tolist() == pytest.approx([expected_fit] * 2)
    assert "cor3_guidance_fit" not in reduced["trajectory_four_stage"]
    if use_reference_mean:
        assert initial.initial_unconditional_mean_error_l2.tolist() == pytest.approx([3.] * 2)
        assert initial.theory_mean_sha256.eq(receipt["vector_sha256"]).all()
    assert reduced["trajectory_shapes"].shape_flat.all()
    assert reduced["feedback_response"].log_probability_gain.eq(0).all()


@requires_cuda
def test_response_saturation_does_not_enclose_whole_dose_grid(monkeypatch):
    from utils.experiments.theory import four_stage_measurements as module
    item = payload([0., -1.])
    item["intercept"] = torch.tensor([[0., -1000.]], dtype=torch.float64)
    def forbidden(*args, **kwargs):
        raise AssertionError("Saturation alone must not invoke GPU intervals")
    monkeypatch.setattr(module, "GpuDirected", forbidden)
    rows = response_rows(item, identities(), 7.5, device="cuda")
    assert rows.log_probability_gain.eq(0).all()
    assert rows.log_odds_gain.iloc[-1] > 0
    assert not rows.response_fallback_selected.any()
    assert rows.response_dose_gpu_products.eq(0).all()


@requires_cuda
def test_response_zero_budget_does_not_construct_gpu_intervals(monkeypatch):
    from utils.experiments.theory import four_stage_measurements as module
    from utils.experiments.theory.numerical_refinement import NumericalPolicy
    monkeypatch.setattr(module, "_gain_retry_needed", lambda *args, **kwargs: True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero-budget response must not construct GPU intervals")
    monkeypatch.setattr(module, "GpuDirected", forbidden)
    rows = response_rows(payload([0., 1.]), identities(), 7.5, device="cuda",
                         policy=NumericalPolicy(max_decimal_products=0))
    assert rows.response_dose_gpu_products.eq(0).all()
    assert rows.response_fallback_status.eq("budget_unavailable_keep_finite_assessment").all()
    assert np.isfinite(rows.log_probability_gain).all()


@requires_cuda
def test_response_gpu_budget_is_shared_across_the_whole_seed_curve(monkeypatch):
    from utils.experiments.theory import four_stage_measurements as module
    from utils.experiments.theory.numerical_refinement import NumericalPolicy
    from utils.experiments.theory.gpu_intervals import GpuInterval
    calls = []
    monkeypatch.setattr(module, "_gain_retry_needed", lambda *args, **kwargs: True)
    def artificial_enclosure(arithmetic, intercept, slopes, target):
        calls.append(True)
        arithmetic.charge(700)
        return {"H": arithmetic.interval(.1), "G": arithmetic.interval(.2), "gain_sign": "positive"}
    monkeypatch.setattr(module, "interval_gain", artificial_enclosure)
    rows = response_rows(payload([0., 1.]), identities(), 7.5, device="cuda",
                         policy=NumericalPolicy(max_decimal_products=1000))
    assert len(calls) >= 1
    assert rows.response_dose_gpu_products.sum() <= 1000
    assert rows.response_curve_gpu_products.iloc[-1] == rows.response_dose_gpu_products.sum()
    assert rows.response_fallback_status.str.contains("budget").any()
    assert rows.response_arithmetic_scope.str.contains("not_original_vector_construction").all()
    # The refined G's sign cannot be substituted for H's plotted magnitude.
    assert rows.log_probability_gain.iloc[-1] < 0


@requires_cuda
def test_response_corrects_only_its_own_contradicted_magnitude_and_retains_prior_values(monkeypatch):
    from utils.experiments.theory import four_stage_measurements as module
    from utils.experiments.theory.gpu_intervals import GpuInterval
    from utils.experiments.theory.numerical_refinement import NumericalPolicy
    monkeypatch.setattr(module, "_gain_retry_needed", lambda *args, **kwargs: True)
    def instrumented_interval(arithmetic, intercept, slopes, target):
        arithmetic.charge(700)
        return {"H": arithmetic.interval(.25), "G": arithmetic.interval(2.), "gain_sign": "positive"}
    monkeypatch.setattr(module, "interval_gain", instrumented_interval)
    result = response_rows(payload([0., 1.]), identities(), 7.5, device="cuda",
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


def test_response_requires_cuda_instead_of_silently_falling_back():
    with pytest.raises(ValueError, match="CUDA.*CPU fallback"):
        response_rows({}, identities(), 7.5, device="cpu")


def test_baseline_summary_preserves_selected_and_bank_mean_receipts():
    probes, repeats, atoms = baseline_inputs()
    for name, value in {"bank_mean_norm_l2": 4., "bank_mean_norm_rmse": 2.,
                        "mean_offset_l2": 3., "mean_offset_rmse": 1.5,
                        "comparison_scale_rmse": 2.5,
                        "theory_mean_source": "initial_unconditional_reference_monte_carlo",
                        "theory_mean_sha256": "independent-estimate"}.items():
        repeats[name] = value
    summary, _ = baseline_summary(probes, repeats, atoms)
    row = summary.iloc[0]
    assert row.mean_norm_l2 == 2. and row.bank_mean_norm_l2 == 4.
    assert row.mean_offset_l2 == 3. and row.comparison_scale_rmse == 2.5
    assert row.theory_mean_sha256 == "independent-estimate"
    assert row.spread_rmse == 2.  # True bank spread still normalizes the ratio.
    assert row.baseline_error_rmse == pytest.approx(math.sqrt(12.5))
