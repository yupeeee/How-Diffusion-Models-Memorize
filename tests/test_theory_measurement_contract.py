"""Manuscript measurement contracts and counterexamples, without inference."""

import math

import pytest
import torch

from utils.experiments.theory.metrics import (
    branch_metrics,
    candidate_terminal_bound,
    initial_distribution_diagnostics,
)
from utils.experiments.theory.scheduler_adapter import (
    SchedulerAdapter,
    UpdateCoefficients,
    displacement_vector_residual,
)
from utils.models.schedule_metadata import build_schedule_metadata


def make_adapter(name="ddim", **kwargs):
    import diffusers
    from diffusers import DDIMScheduler, DDPMScheduler

    scheduler_type = DDIMScheduler if name == "ddim" else DDPMScheduler
    scheduler = scheduler_type(num_train_timesteps=31, clip_sample=False, **kwargs)
    schedule = build_schedule_metadata(
        scheduler, name, num_inference_steps=6, device="cpu"
    )
    return scheduler, SchedulerAdapter(
        schedule, recorded_diffusers_version=diffusers.__version__
    )


def exact_terminal_adapter(**changes):
    # Rational coefficients isolate vector arithmetic from schedule sqrt rounding.
    adapter = SchedulerAdapter.__new__(SchedulerAdapter)
    adapter.timesteps = torch.tensor([0])
    values = dict(
        step_index=0,
        timestep=0,
        destination_timestep=-1,
        alpha=0.5,
        sigma=0.5,
        snr=1,
        destination_alpha=1,
        destination_sigma=0,
        A=0,
        B=1,
        noise_std=0,
        deterministic=True,
        affine=True,
        status="valid_affine",
    )
    values.update(changes)
    adapter._coefficients = [UpdateCoefficients(**values)]
    return adapter


def test_both_branches_can_agree_and_both_be_wrong():
    z = torch.ones(2, 1, 1, 2, dtype=torch.float64)
    eps = torch.zeros_like(z)
    target = torch.zeros(1, 1, 2, dtype=torch.float64)
    values = branch_metrics(z, eps, eps, target, 0.8, 0.6, 7.5)
    assert torch.all(values["branch_gap_rmse"] == 0)
    assert torch.all(values["conditional_target_error_rmse"] == 1.25)
    assert torch.equal(
        values["joint_target_error_rmse"], values["conditional_target_error_rmse"]
    )
    assert values["conditional_target_error_reference_scope"].endswith(
        "not_verified_by_pair"
    )
    assert values["population_forward_loss"] is None


def test_zero_target_error_relative_improvement_is_undefined():
    z = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    values = branch_metrics(z, z, torch.ones_like(z), z[0], 0.8, 0.6, 7.5)
    assert values["relative_target_error_improvement_undefined_zero_denominator"].item()
    assert torch.isnan(values["relative_target_error_improvement"]).all()
    assert values["paired_target_error_improvement_rmse"].item() < 0


def test_complete_injection_mismatch_includes_wrong_parallel_magnitude_and_perpendicular():
    target = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    z = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    # m_c=(-1,2), g=2 => J=(-2,4), a=-2,r=4.
    values = branch_metrics(
        z, z, torch.tensor([[[[1.0, -2.0]]]]), target, 0.5, 0.5, 2, center=z[0]
    )
    assert values["a_parallel"].item() == pytest.approx(-2)
    assert values["off_target"].item() == pytest.approx(4)
    assert values["injection_relative_mismatch"].item() == pytest.approx(math.sqrt(8))
    direct = math.sqrt(32) / 2
    assert values["injection_relative_mismatch"].item() == pytest.approx(direct)


@pytest.mark.parametrize("offset", [0.0, 1e-7])
def test_near_degenerate_direction_is_not_filled_with_zero(offset):
    target = torch.tensor([[[1.0 + offset]]], dtype=torch.float32)
    z = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    values = branch_metrics(
        z, z, -z, target, 0.8, 0.6, 7.5, center=torch.ones_like(target)
    )
    assert values["projection_status"] == "degenerate_target_center_direction"
    assert torch.isnan(values["a_parallel"]).all()
    assert torch.isnan(values["off_target"]).all()
    assert torch.isnan(values["injection_relative_mismatch"]).all()
    assert torch.isfinite(values["conditional_target_error_rmse"]).all()


def test_eq31_uses_forward_to_initial_kl_and_preserves_unclipped_bound():
    target = torch.ones(4, dtype=torch.float64) * 20
    a, sigma = 0.3, math.sqrt(0.91)
    result = initial_distribution_diagnostics(
        target, a, sigma, 1, gaussian_initialization_recorded=True
    )
    wanted = 0.5 * math.sqrt(
        a * a * target.square().sum().item()
        + 4 * (sigma * sigma - 1 - math.log(sigma * sigma))
    )
    assert result["initial_transfer_bound_unclipped"] == pytest.approx(wanted)
    assert result["initial_transfer_bound_clipped"] == 1
    assert wanted != pytest.approx(math.sqrt(result["initial_gaussian_kl"] / 2))
    assert result["initial_gaussian_kl_direction"] == "initialization_P_to_forward_Q"


@pytest.mark.parametrize(
    "scale,recorded,alpha,sigma",
    [(2, True, 0.6, 0.8), (1, False, 0.6, 0.8), (1, True, 0.5, 0.5)],
)
def test_eq31_requires_initialization_and_vp_contract(scale, recorded, alpha, sigma):
    result = initial_distribution_diagnostics(
        torch.ones(2), alpha, sigma, scale, gaussian_initialization_recorded=recorded
    )
    assert result["initial_transfer_bound_unclipped"] is None
    assert (
        result["initial_transfer_bound_status"]
        != "applicable_intended_standard_Gaussian_initialization"
    )


def test_vector_displacement_qa_detects_equal_norm_wrong_direction():
    next_state = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
    matched = torch.zeros_like(next_state)
    delta = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    assert next_state.norm() == delta.norm()
    residual = displacement_vector_residual(next_state, matched, delta, 1, 1)
    assert residual.item() == 1


@pytest.mark.parametrize("name", ["ddim", "ddpm"])
def test_same_noise_matched_reconstruction_and_independent_replay_status(name):
    scheduler, adapter = make_adapter(name)
    z = torch.tensor([[[[1.0, -2.0]]]], dtype=torch.float64)
    u, c, g = 0.25 * z, -0.5 * z, 2
    kwargs = (
        {"eta": 0.0}
        if name == "ddim"
        else {"generator": torch.Generator().manual_seed(31)}
    )
    guided = scheduler.step(
        u + g * (c - u), scheduler.timesteps[0], z, **kwargs
    ).prev_sample
    kwargs = (
        {"eta": 0.0}
        if name == "ddim"
        else {"generator": torch.Generator().manual_seed(31)}
    )
    expected = scheduler.step(u, scheduler.timesteps[0], z, **kwargs).prev_sample
    values, matched, _ = adapter.matched_update(z, guided, u, c, g, 0)
    assert torch.allclose(matched, expected, atol=2e-6, rtol=2e-6)
    assert values["matched_construction_vector_residual_rmse"].max() < 1e-12
    assert not values["matched_construction_residual_is_independent_verification"]
    assert values["kappa"] > 0
    if name == "ddpm":
        assert values["independent_replay_status"].startswith("unavailable_saved_noise")
        assert torch.isnan(values["independent_replay_rmse"]).all()
    else:
        assert values["independent_replay_rmse"].max() < 2e-6


def test_nonpositive_kappa_rejected_for_manuscript_identity():
    _, adapter = make_adapter(set_alpha_to_one=False)
    coeff = adapter.coefficients(5)
    assert coeff.B == 0
    assert not coeff.affine
    assert coeff.status == "inapplicable_nonpositive_kappa"
    z = torch.ones(1, 1, 1, 2)
    values, matched, _ = adapter.matched_update(z, z, z, z, 7.5, 5)
    assert values["matched_identity_status"].startswith("inapplicable")
    assert torch.isnan(matched).all()


@pytest.mark.parametrize(
    "changes",
    [
        dict(destination_alpha=0.99, destination_sigma=0.1, A=0.1, B=0.9),
        dict(noise_std=1e-10, deterministic=False),
        dict(affine=False, status="valid_nonlinear_drift"),
    ],
)
def test_accidental_zero_terminal_residual_does_not_override_structure(changes):
    adapter = exact_terminal_adapter(**changes)
    z = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    values = adapter.terminal_diagnostics(z, z, z, z, 2, 0, z[0])
    assert values["scheduler_rho_rmse"].item() == 0
    assert values["terminal_clean_numeric_within_sensitivity"].item()
    assert not values["terminal_clean_structural"]
    assert not values["terminal_clean_applicable"].item()


def test_eq16_cancellation_is_not_a_small_two_term_bound():
    adapter = exact_terminal_adapter()
    z = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    # m_u=2, m_c=1, g=2 => guided=0 exactly; target=0.
    values = adapter.terminal_diagnostics(z, z, z - 2, z - 1, 2, 0, z[0])
    assert values["terminal_clean_applicable"].item()
    assert values["terminal_A_rmse"].item() == 1
    assert values["terminal_B_rmse"].item() == 1
    assert values["terminal_terms_cosine"].item() == -1
    assert values["terminal_cross_term_per_dimension"].item() == -2
    assert values["terminal_clean_error_rmse"].item() == 0
    assert values["terminal_two_term_bound_rmse"].item() == 2
    assert values["terminal_eq16_vector_residual_rmse"].item() == 0
    assert values["terminal_eq16_squared_l2_from_terms"].item() == 0
    assert values["terminal_two_term_ratio_undefined_zero_error"].item()
    assert torch.isnan(values["terminal_two_term_bound_error_ratio"]).all()


def test_candidate_bound_is_distinct_and_log_complement_stable():
    # log p rounds to zero for p=1-exp(-100); saved log complement retains mass.
    result = candidate_terminal_bound(
        2.0,
        3.0,
        4.0,
        -100.0,
        guidance=2,
        dimension=4,
        clean_error_l2=torch.tensor([0.0, 1.0]),
        endpoint_error_l2=torch.tensor([1.0, 2.0]),
        terminal_clean_update=torch.tensor([True, False]),
    )
    assert result["candidate_terminal_non_target_mass"].item() == pytest.approx(
        math.exp(-100)
    )
    assert result["candidate_terminal_concentration_term_l2"].item() == pytest.approx(
        4 * math.exp(-100)
    )
    assert result["candidate_terminal_conditional_term_l2"].item() == 2.0
    assert result["candidate_terminal_branch_gap_error_term_l2"].item() == 3.0
    assert result["candidate_terminal_measurement_contract"] == "projected-gap-error-1"
    assert result["candidate_terminal_bound_rmse"].item() == pytest.approx(2.5)
    assert result["candidate_terminal_clean_bound_slack_l2"].tolist() == [5.0, 4.0]
    assert result["candidate_terminal_bound_applies_to_endpoint"].tolist() == [
        True,
        False,
    ]
    assert torch.isnan(result["candidate_terminal_bound_clean_error_ratio"][0])
    assert torch.isnan(result["candidate_terminal_endpoint_bound_slack_l2"][1])


def test_unavailable_candidate_reference_does_not_invent_a_bound():
    result = candidate_terminal_bound(
        1,
        None,
        None,
        None,
        guidance=7.5,
        dimension=2,
        clean_error_l2=torch.ones(1),
        endpoint_error_l2=torch.ones(1),
        terminal_clean_update=True,
    )
    assert result["candidate_terminal_bound_status"] == "unavailable_reference"
    assert torch.isnan(result["candidate_terminal_bound_rmse"]).all()


def test_candidate_terminal_bound_keeps_negative_projected_error():
    result = candidate_terminal_bound(
        1.0, -1.0, 2.0, 0.0, guidance=2.0, dimension=4,
        clean_error_l2=torch.tensor([1.0]), endpoint_error_l2=torch.tensor([1.0]),
        terminal_clean_update=True,
    )
    assert result["candidate_terminal_bound_reference_valid"].item()
    assert result["candidate_terminal_branch_gap_error_term_l2"].item() == -1.0
    assert result["candidate_terminal_bound_l2"].item() == 2.0
    assert result["candidate_terminal_bound_rmse"].item() == 1.0
