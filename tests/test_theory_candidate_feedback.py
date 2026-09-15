"""Small exact/reference contracts; empirical model signs are never asserted."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
import torch

from utils.experiments.theory.candidate_feedback import (
    dose_grid,
    endpoint_metrics,
    fixed_control_atoms,
    stable_log_probability_gain,
    stable_logsum_difference,
    unavailable_endpoint,
)
from utils.experiments.theory.candidate_integration import (
    IntegrationConfig,
    integration_metrics,
    wide_partition,
)
from utils.experiments.theory.support import FiniteSupport


def support(atoms=((1.0,), (-1.0,)), weights=None):
    return FiniteSupport(
        torch.tensor(atoms, dtype=torch.float64),
        [str(i) for i in range(len(atoms))],
        {str(i): i for i in range(len(atoms))},
        weights=weights,
    )


def measure(
    bank=None,
    *,
    state=None,
    mc=None,
    mu=None,
    matched=None,
    guidance=2.0,
    kappa=0.2,
    current_alpha=0.8,
    current_sigma=0.6,
    destination_alpha=0.9,
    destination_sigma=math.sqrt(0.19),
    source_error_l2=0.0,
    target_id="0",
):
    bank = support() if bank is None else bank
    d = bank.dimension

    def vec(x, default):
        return torch.tensor([default if x is None else x], dtype=torch.float64).reshape(
            -1, d
        )

    state, mc, mu, matched = (
        vec(state, [0.0] * d),
        vec(mc, bank.flat[0].tolist()),
        vec(mu, [0.0] * d),
        vec(matched, [0.0] * d),
    )
    guided = matched + guidance * kappa * (mc - mu)
    return endpoint_metrics(
        bank,
        state,
        mc,
        mu,
        matched,
        guided,
        target_id,
        current_alpha=current_alpha,
        current_sigma=current_sigma,
        destination_alpha=destination_alpha,
        destination_sigma=destination_sigma,
        guidance=guidance,
        kappa=kappa,
        bank_hash="fixture-bank",
        source_error_l2=source_error_l2,
    )


def integrate(bank, batch, **kwargs):
    return integration_metrics(bank, batch.segment, bank_hash="fixture-bank", **kwargs)


def test_binary_nonuniform_endpoint_bracket_and_fixed_dose():
    bank = support(weights=[0.2, 0.8])
    out = measure(bank)
    s = out.scalars
    b = math.log(4)
    q = 0.9 / 0.19 * 0.4 * 2
    assert s["candidate_matched_log_odds"].item() == pytest.approx(-b)
    assert s["candidate_log_odds_gain"].item() == pytest.approx(q)
    assert s["candidate_endpoint_slope_C0"].item() == pytest.approx(q)
    assert s["candidate_endpoint_slope_C1"].item() == pytest.approx(q)
    assert s["candidate_profile_class"] == ["increasing"]
    assert s["candidate_gain_status"] == ["positive"]
    assert (
        s["candidate_endpoint_bracket_residual"].item()
        >= -s["candidate_endpoint_arithmetic_tolerance"].item()
    )
    grid = out.dose["lambda"]
    assert set([0.0, 0.5, 1.0]) <= set(grid.tolist())
    assert out.dose["candidate_dose_log_probability_gain"][0, 0].item() == 0
    assert out.dose["candidate_dose_log_probability_gain"][
        0, -1
    ].item() == pytest.approx(s["candidate_log_probability_gain"].item())
    assert out.dose["candidate_dose_curvature"].abs().max() < 1e-20


@pytest.mark.parametrize(
    "mc,profile,sign",
    [(1.0, "increasing", "positive"), (-1.0, "decreasing", "negative")],
)
def test_binary_signed_profiles(mc, profile, sign):
    out = measure(mc=[mc])
    assert out.scalars["candidate_profile_class"] == [profile]
    assert out.scalars["candidate_gain_status"] == [sign]


def test_interior_turn_has_concave_log_odds_without_assuming_net_gain():
    bank = support(((0.0,), (-1.0,), (1.0,)))
    out = measure(
        bank,
        mc=[1.0],
        mu=[0.0],
        matched=[-0.4],
        guidance=2.0,
        kappa=0.5,
        destination_alpha=1.0,
        destination_sigma=1.0,
    )
    assert out.scalars["candidate_profile_class"] == ["interior_peak"]
    slopes = out.dose["candidate_dose_slope"][0]
    assert bool((slopes[1:] <= slopes[:-1] + 1e-14).all())
    assert out.dose["candidate_dose_curvature"].max() <= 0
    assert (
        out.dose["candidate_dose_log_probability_gain"][0, -1]
        < out.dose["candidate_dose_log_probability_gain"][0].max()
    )


def test_zero_displacement_is_exact_zero_with_no_artificial_direction():
    bank = support()
    out = measure(bank, mc=[2.0], mu=[2.0])
    s = out.scalars
    assert s["candidate_gain_status"] == ["arithmetic_zero"]
    assert s["candidate_profile_class"] == ["flat"]
    assert s["candidate_log_probability_gain"].item() == 0
    assert s["candidate_direction_normalization_status"] == ["zero_displacement"]
    assert torch.isnan(s["candidate_normalized_log_odds_gain"]).all()
    integrated = integrate(bank, out)
    assert torch.isnan(integrated["candidate_exact_directional_average_l2"]).all()
    assert torch.isfinite(integrated["candidate_margin_original_l2"]).all()


@pytest.mark.parametrize(
    "matched,mc,expected",
    [(500.0, 1.0, "positive"), (500.0, -1.0, "negative"), (-500.0, 1.0, "positive")],
)
def test_extreme_endpoint_probabilities_preserve_strict_sign(matched, mc, expected):
    out = measure(
        matched=[matched], mc=[mc], destination_alpha=1.0, destination_sigma=1.0
    )
    assert out.scalars["candidate_gain_status"] == [expected]
    assert out.scalars["candidate_gain_saturated"].item()
    if matched > 0:
        assert out.scalars["candidate_gain_underflow"].item()
        assert out.scalars["candidate_gain_magnitude_status"] == ["underflow"]
    else:
        assert out.scalars["candidate_log_probability_gain"].item() > 0


def test_stable_small_softplus_and_logsum_differences():
    a = torch.tensor([400.0, -400.0], dtype=torch.float64)
    g = torch.tensor([1e-8, 1e-8], dtype=torch.float64)
    gain = stable_log_probability_gain(a, a + g, g)
    assert gain[0].item() == pytest.approx(
        math.exp(-400) * (-math.expm1(-1e-8)), rel=1e-12, abs=0
    )
    assert gain[1].item() == pytest.approx(1e-8)
    right = torch.tensor([[1e8, 1e8 - 1]], dtype=torch.float64)
    left = right + 1e-5
    assert stable_logsum_difference(left, right).item() == pytest.approx(
        (left - right)[0, 0].item(), abs=1e-14
    )


def test_large_logit_subtraction_and_tiny_direction_remain_unresolved():
    out = measure(
        matched=[1e16],
        mc=[0.1],
        destination_alpha=1.0,
        destination_sigma=1.0,
        source_error_l2=1.0,
    )
    assert out.scalars["candidate_gain_status"] == ["numerically_unresolved"]
    assert out.scalars["candidate_direction_normalization_status"] == [
        "arithmetic_unresolved"
    ]
    assert torch.isnan(out.scalars["candidate_normalized_log_odds_gain"]).all()


def test_exact_duplicate_aggregation_missing_target_and_single_atom():
    bank = FiniteSupport.from_candidates(
        [
            ("a", torch.tensor([1.0])),
            ("copy", torch.tensor([1.0])),
            ("b", torch.tensor([-1.0])),
        ]
    )
    assert bank.size == 2 and bank.aliases["copy"] == bank.aliases["a"]
    a = measure(bank, target_id="a")
    copy = measure(bank, target_id="copy")
    assert torch.equal(
        a.scalars["candidate_log_probability_gain"],
        copy.scalars["candidate_log_probability_gain"],
    )
    assert measure(bank, target_id="missing").scalars["candidate_feedback_status"] == [
        "missing_candidate_target"
    ]
    assert measure(support(((1.0,),))).scalars["candidate_feedback_status"] == [
        "no_distinct_competitor"
    ]


def test_fixed_controls_depend_only_on_bank_and_atom_identities():
    bank = support(tuple((float(i),) for i in range(30)))
    chosen = fixed_control_atoms(bank, 0, "hash-a")
    assert len(chosen) == 16 and 0 not in chosen
    assert chosen == fixed_control_atoms(bank, 0, "hash-a")
    assert chosen != fixed_control_atoms(bank, 0, "hash-b")
    a = measure(bank, mc=[1.0])
    b = measure(bank, mc=[-1.0], matched=[3.0])
    assert a.controls["control_atom_indices"] == b.controls["control_atom_indices"]
    assert a.scalars["candidate_specificity_contrast"].item() == pytest.approx(
        a.scalars["candidate_log_probability_gain"].item()
        - torch.quantile(
            a.controls["candidate_control_log_probability_gain"], 0.5, dim=1
        ).item()
    )


def test_current_alignment_and_destination_contraction_use_distinct_noise_levels():
    bank = support()
    out = measure(
        bank,
        state=[0.2],
        current_alpha=0.2,
        current_sigma=0.9,
        matched=[-0.3],
        mc=[1.0],
        destination_alpha=0.95,
        destination_sigma=0.2,
    )
    expected_current = math.tanh(0.2 * 0.2 / 0.9**2)
    assert out.scalars[
        "candidate_current_reference_error_u_l2"
    ].item() == pytest.approx(abs(expected_current))
    mean0 = math.tanh(0.95 * (-0.3) / 0.2**2)
    mean1 = math.tanh(0.95 * 0.1 / 0.2**2)
    assert out.scalars[
        "candidate_next_reference_target_contraction_rmse"
    ].item() == pytest.approx(abs(mean0 - 1) - abs(mean1 - 1))


def test_positive_target_gain_can_move_reference_farther_from_target():
    bank = support(
        ((0.0,), (-1.0,), (10.0,)), weights=[1.0, math.exp(0.5), math.exp(30)]
    )
    out = measure(
        bank,
        mc=[1.8],
        mu=[0.0],
        guidance=2.0,
        kappa=0.5,
        destination_alpha=1.0,
        destination_sigma=1.0,
    )
    assert out.scalars["candidate_log_probability_gain"].item() > 0
    assert out.scalars["candidate_next_reference_target_contraction_rmse"].item() < 0


def test_cancelled_orthogonal_errors_refine_negative_original_margin():
    bank = support(((1.0, 0.0), (-1.0, 0.0)))
    out = measure(bank, mc=[1.0, 100.0], mu=[0.0, 100.0])
    r = integrate(bank, out)
    assert out.scalars["candidate_combined_reference_error_l2"].item() == pytest.approx(
        0
    )
    assert r["candidate_margin_original_l2"].item() < 0
    assert r["candidate_margin_combined_l2"].item() > 0
    assert out.scalars["candidate_log_probability_gain"].item() > 0
    margins = [
        r[f"candidate_margin_{name}_l2"].item()
        for name in ("original", "combined", "signed_error", "projected_variation")
    ] + [r["candidate_exact_directional_average_l2"].item()]
    assert np.diff(margins).min() >= -1e-12
    assert abs(r["candidate_integral_identity_residual"].item()) < 1e-8
    assert abs(r["candidate_directional_identity_residual"].item()) < 1e-8


def test_integration_resume_uses_no_latent_queries(monkeypatch):
    bank = support()
    out = measure(bank)
    assert out.segment["intercept"].shape == (1, bank.size)
    assert not any(
        isinstance(x, torch.Tensor) and x.ndim > 2 for x in out.segment.values()
    )
    monkeypatch.setattr(
        bank,
        "_queries",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw query reload")),
    )
    result = integrate(bank, out)
    assert result["candidate_integration_status"] == ["estimated_converged"]
    with pytest.raises(ValueError, match="identity mismatch"):
        integration_metrics(bank, out.segment, bank_hash="changed-bank")


def test_sharp_transition_with_wide_tail_partition_and_independent_integral():
    bank = support()
    out = measure(
        bank,
        state=[-1.0],
        mc=[2.0],
        mu=[-1.0],
        matched=[-1.0],
        guidance=2.0,
        kappa=0.5,
        current_alpha=1.0,
        current_sigma=0.05,
        destination_alpha=1.0,
        destination_sigma=1e-4,
    )
    r = integrate(
        bank,
        out,
        config=IntegrationConfig(absolute_tolerance=1e-8, relative_tolerance=1e-8),
    )
    assert r["candidate_variation_norm_integral_l2"].item() == pytest.approx(
        4 / 3, abs=2e-7
    )
    assert r["candidate_variation_abs_projected_integral_l2"].item() == pytest.approx(
        4 / 3, abs=2e-7
    )
    assert r["candidate_variation_signed_integral_l2"].item() == pytest.approx(
        4 / 3, abs=2e-7
    )
    b = np.array([0.0, 1000.0])
    s = np.array([0.0, -2000.0])
    parts, truncated = wide_partition(b, s, IntegrationConfig())
    points = sorted({x for pair in parts for x in pair})
    assert 0.5 + 64 / 2000 in points and 0.5 - 64 / 2000 in points and not truncated
    independent = quad(
        lambda u: -(-2000.0) / (1 + math.exp(np.clip(-(1000 - 2000 * u), -700, 700))),
        0,
        1,
        points=points[1:-1],
        epsabs=1e-9,
    )[0]
    assert independent == pytest.approx(1000.0, abs=1e-8)


def test_budget_failure_does_not_erase_endpoint_gain():
    bank = support()
    out = measure(
        bank,
        matched=[-0.431],
        mc=[1.0],
        mu=[-1.0],
        guidance=2.0,
        kappa=0.5,
        destination_alpha=1.0,
        destination_sigma=0.1,
    )
    original = out.scalars["candidate_log_probability_gain"].clone()
    r = integrate(
        bank,
        out,
        config=IntegrationConfig(
            max_evaluations=15, absolute_tolerance=1e-12, relative_tolerance=1e-12
        ),
    )
    assert r["candidate_quadrature_budget_exhausted"].item()
    assert r["candidate_integration_status"] == ["numerically_unresolved"]
    assert torch.equal(original, out.scalars["candidate_log_probability_gain"])


def test_mixed_invalid_population_retained_after_integration():
    bank = support()
    state = torch.tensor([[0.0], [float("nan")], [0.1]], dtype=torch.float64)
    mc = torch.ones_like(state)
    mu = torch.zeros_like(state)
    cf = torch.zeros_like(state)
    guided = cf + 0.4 * (mc - mu)
    out = endpoint_metrics(
        bank,
        state,
        mc,
        mu,
        cf,
        guided,
        "0",
        current_alpha=0.8,
        current_sigma=0.6,
        destination_alpha=0.9,
        destination_sigma=0.5,
        guidance=2.0,
        kappa=0.2,
        bank_hash="fixture-bank",
    )
    assert out.scalars["candidate_feedback_eligible"].tolist() == [True, False, True]
    assert out.segment["population_count"] == 3 and out.segment[
        "row_indices"
    ].tolist() == [0, 2]
    result = integrate(bank, out)
    assert len(result["candidate_margin_original_status"]) == 3
    assert result["candidate_margin_original_status"][1] == "not_applicable"


def test_gram_and_vector_integrals_agree():
    bank = support(((1.0, 0.0), (-1.0, 1.0), (0.2, -1.0)), weights=[0.2, 0.3, 0.5])
    out = measure(
        bank, mc=[1.0, 0.2], mu=[-0.3, 0.1], state=[0.1, -0.1], matched=[-0.2, 0.2]
    )
    gram = integrate(bank, out)
    direct = integrate(bank, out, config=IntegrationConfig(use_gram=False))
    for name in ("norm", "abs_projected", "signed"):
        assert torch.allclose(
            gram[f"candidate_variation_{name}_integral_l2"],
            direct[f"candidate_variation_{name}_integral_l2"],
            atol=1e-9,
            rtol=1e-8,
        )


def test_status_schema_stable_for_unavailable():
    available = measure()
    unavailable = unavailable_endpoint(1)
    assert set(available.scalars) == set(unavailable.scalars)
    assert set(available.dose) == set(unavailable.dose)
    assert len(dose_grid(7.5)) == 22 and 1 / 7.5 in dose_grid(7.5).tolist()


def test_integration_chunking_never_uses_boolean_scatter(monkeypatch):
    bank = support()
    out = measure(
        bank,
        matched=[-0.274],
        mc=[2.0],
        mu=[-1.0],
        guidance=2.0,
        kappa=0.5,
        destination_alpha=1.0,
        destination_sigma=0.01,
    )
    original = torch.Tensor.scatter_reduce_
    dtypes = []

    def checked(self, dim, index, source, **kwargs):
        assert self.dtype != torch.bool and source.dtype != torch.bool
        dtypes.append(self.dtype)
        return original(self, dim, index, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "scatter_reduce_", checked)
    small = integrate(bank, out, config=IntegrationConfig(node_chunk=7))
    large = integrate(bank, out, config=IntegrationConfig(node_chunk=256))
    assert torch.int64 in dtypes
    assert torch.allclose(
        small["candidate_variation_norm_integral_l2"],
        large["candidate_variation_norm_integral_l2"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert (
        abs(small["candidate_integral_identity_residual"].item())
        <= small["candidate_integral_identity_allowance"].item()
    )


def test_wide_partition_handles_transition_just_outside_endpoint():
    cfg = IntegrationConfig()
    parts, _ = wide_partition(np.array([0.0, 1002.0]), np.array([0.0, -1000.0]), cfg)
    points = sorted({x for pair in parts for x in pair})
    # Crossing at1.002 is outside; its posterior tail still varies inside.
    assert any(0.93 < x < 1 for x in points)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable on audit host"
)
def test_cuda_endpoint_and_integration_match_cpu():
    cpu = support()
    a = measure(cpu)
    gpu = FiniteSupport(
        cpu.atoms.cuda(), cpu.atom_ids, cpu.aliases, weights=cpu.weights.cuda()
    )
    b = measure(gpu)
    for key in (
        "candidate_log_probability_gain",
        "candidate_log_odds_gain",
        "candidate_current_alignment_cosine",
    ):
        assert torch.allclose(
            a.scalars[key], b.scalars[key].cpu(), atol=1e-10, rtol=1e-10
        )
    x = integrate(cpu, a)
    y = integrate(gpu, b)
    assert torch.allclose(
        x["candidate_variation_norm_integral_l2"],
        y["candidate_variation_norm_integral_l2"].cpu(),
        atol=1e-8,
        rtol=1e-8,
    )


def test_zero_guidance_retains_zero_gain_without_dividing_by_zero():
    out = measure(guidance=0.0)
    assert out.scalars["candidate_gain_status"] == ["arithmetic_zero"]
    assert out.scalars["candidate_profile_class"] == ["flat"]
    assert out.scalars["candidate_direction_normalization_status"] == [
        "zero_displacement"
    ]
    assert torch.isnan(out.scalars["candidate_normalized_log_probability_gain"]).all()
    assert out.dose["candidate_dose_log_probability_gain"].abs().max() == 0


def test_large_source_budget_does_not_erase_arithmetically_resolved_observations():
    bank = support()
    out = measure(bank, mc=[0.1], source_error_l2=10.0)
    exact_inputs = measure(bank, mc=[0.1], source_error_l2=0.0)
    scalars = out.scalars
    assert scalars["candidate_direction_normalization_status"] == ["resolved"]
    assert scalars["candidate_direction_input_precision_status"] == [
        "source_sensitivity_at_or_above_observed_gap"
    ]
    assert scalars[
        "candidate_direction_input_sensitivity_ratio"
    ].item() == pytest.approx(100.0)
    for field in (
        "normalized_log_probability_gain",
        "normalized_log_odds_gain",
        "signed_error_projection_l2",
        "current_alignment_cosine",
    ):
        assert torch.equal(
            scalars["candidate_" + field], exact_inputs.scalars["candidate_" + field]
        )
    refined = integrate(bank, out)
    for name in ("original", "combined", "signed_error", "projected_variation"):
        assert refined[f"candidate_margin_{name}_uncertainty_l2"].item() >= 10.0


def test_real_smoke_shaped_source_uncertainty_is_separate_from_normalization():
    # The real smoke has20seeds, d16384, source envelopes larger than D, and
    # ordinary representable h. This fixture tests that measurement contract,
    # not any expected sign or coverage in empirical model data.
    dimension, count = 16384, 20
    atoms = torch.zeros((2, dimension), dtype=torch.float64)
    atoms[:, 0] = torch.tensor([1.0, -1.0])
    bank = FiniteSupport(atoms, ["0", "1"], {"0": 0, "1": 1})
    state = torch.zeros((count, dimension), dtype=torch.float64)
    mu = torch.zeros_like(state)
    mu[:, 1] = 100.0
    mc = mu.clone()
    mc[:, 0] = torch.linspace(-20.0, 20.0, count)
    cf = torch.zeros_like(state)
    g, kappa, alpha_next, sigma_next = 7.5, 0.04, 0.9, 0.5
    guided = cf + g * kappa * (mc - mu)
    out = endpoint_metrics(
        bank,
        state,
        mc,
        mu,
        cf,
        guided,
        "0",
        current_alpha=0.8,
        current_sigma=0.6,
        destination_alpha=alpha_next,
        destination_sigma=sigma_next,
        guidance=g,
        kappa=kappa,
        bank_hash="fixture-bank",
        source_error_l2=100.0,
    )
    s = out.scalars
    assert s["candidate_direction_normalization_status"] == ["resolved"] * count
    assert (
        s["candidate_direction_input_precision_status"]
        == ["source_sensitivity_at_or_above_observed_gap"] * count
    )
    denominator = (
        alpha_next / sigma_next**2 * (guided - cf).norm(dim=1) * math.sqrt(dimension)
    )
    assert torch.allclose(
        s["candidate_normalized_log_odds_gain"],
        s["candidate_log_odds_gain"] / denominator,
    )
    assert torch.isfinite(s["candidate_current_alignment_cosine"]).all()
    assert torch.isfinite(s["candidate_signed_error_projection_l2"]).all()
    integrated = integrate(bank, out)
    assert torch.isfinite(integrated["candidate_exact_directional_average_l2"]).all()
    assert (integrated["candidate_margin_signed_error_uncertainty_l2"] >= 100.0).all()
