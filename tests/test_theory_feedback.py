"""Exact low-dimensional tests for candidate-law Proposition 5 measurements."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
import torch

from utils.experiments.theory.feedback import (
    IntegrationConfig,
    mean_distance_from_weights,
    proposition5_feedback,
    unavailable_feedback,
)
from utils.experiments.theory.support import FiniteSupport


def vec(value):
    return torch.tensor(value, dtype=torch.float64).reshape(-1, 1)


def binary_support(**kwargs):
    return FiniteSupport.from_candidates(
        [("target", vec(1).flatten()), ("other", vec(-1).flatten())], **kwargs
    )


def measure(
    *,
    current=0.0,
    conditional=1.0,
    unconditional=0.0,
    matched=0.0,
    guidance=2.0,
    kappa=0.2,
    current_alpha=0.8,
    current_sigma=0.6,
    destination_alpha=0.9,
    destination_sigma=math.sqrt(0.19),
    support=None,
    target_id="target",
    source_error_l2=0.0,
    config=None
):
    bank = binary_support() if support is None else support
    guided = np.asarray(matched) + guidance * kappa * (
        np.asarray(conditional) - np.asarray(unconditional)
    )
    return proposition5_feedback(
        bank,
        vec(current),
        vec(conditional),
        vec(unconditional),
        vec(matched),
        vec(guided),
        target_id,
        current_alpha=current_alpha,
        current_sigma=current_sigma,
        destination_alpha=destination_alpha,
        destination_sigma=destination_sigma,
        guidance=guidance,
        kappa=kappa,
        source_error_l2=source_error_l2,
        config=config,
    )


def test_positive_condition_and_endpoint_integral_identity():
    result = measure()
    assert 0 < result["candidate_variation_l2"].item() < 1
    assert result["candidate_conditional_reference_error_l2"].item() == 0
    assert result["candidate_unconditional_reference_error_l2"].item() == pytest.approx(
        0, abs=1e-14
    )
    assert result["candidate_condition_status"] == ["estimated_met"]
    assert result["candidate_gain_status"] == ["positive"]
    assert result["candidate_log_probability_gain"].item() > 0
    assert result["candidate_integral_identity_residual"].abs().item() < 1e-9
    assert result["candidate_proof_lower_bound_slack"].item() >= -1e-9
    assert result["candidate_integral_qa_status"] == [
        "consistent_with_numerical_estimates"
    ]
    assert result["feedback_evidence"] == "candidate_distribution_diagnostic"


@pytest.mark.parametrize(
    "conditional,unconditional,gain_sign",
    [(-1.0, 0.0, "negative"), (0.0, -2.0, "positive")],
)
def test_unmet_condition_preserves_both_gain_signs(
    conditional, unconditional, gain_sign
):
    result = measure(conditional=conditional, unconditional=unconditional)
    assert result["candidate_condition_status"] == ["estimated_not_met"]
    assert result["candidate_gain_status"] == [gain_sign]
    expected = 1 if gain_sign == "positive" else -1
    assert expected * result["candidate_log_probability_gain"].item() > 0
    assert result["candidate_integral_identity_residual"].abs().item() < 1e-7


def test_current_reference_uses_current_noise_and_state():
    result = measure(
        current=0.4,
        current_alpha=0.3,
        current_sigma=0.95,
        matched=-0.2,
        conditional=0.8,
        unconditional=0.1,
        destination_alpha=0.95,
        destination_sigma=0.2,
        config=IntegrationConfig(absolute_tolerance=1e-8, relative_tolerance=1e-8),
    )
    current_mean = math.tanh(0.3 * 0.4 / 0.95**2)
    shift = 2 * 0.2 * (0.8 - 0.1)
    variation = quad(
        lambda s: abs(math.tanh(0.95 * (-0.2 + s * shift) / 0.2**2) - current_mean),
        0,
        1,
        epsabs=1e-11,
        points=[0.2 / shift],
    )[0]
    assert result["candidate_unconditional_reference_error_l2"].item() == pytest.approx(
        abs(0.1 - current_mean)
    )
    assert result["candidate_variation_l2"].item() == pytest.approx(variation, abs=3e-7)
    wrong_reference = math.tanh(0.95 * 0.4 / 0.2**2)
    wrong = quad(
        lambda s: abs(math.tanh(0.95 * (-0.2 + s * shift) / 0.2**2) - wrong_reference),
        0,
        1,
    )[0]
    assert abs(variation - wrong) > 0.1


def test_sharp_interior_transition_resolved_and_not_endpoint_only():
    config = IntegrationConfig(
        absolute_tolerance=1e-7,
        relative_tolerance=1e-7,
        identity_absolute_tolerance=1e-7,
        max_evaluations=4095,
    )
    # Destination posterior switches inside the interval at s=1/3, across a
    # width ~1e-8. Endpoint-only trapezoids predict 1; the true V is 4/3.
    result = measure(
        current=-1.0,
        current_alpha=1.0,
        current_sigma=0.05,
        conditional=2.0,
        unconditional=-1.0,
        matched=-1.0,
        guidance=2.0,
        kappa=0.5,
        destination_alpha=1.0,
        destination_sigma=1e-4,
        config=config,
    )
    assert result["candidate_variation_l2"].item() == pytest.approx(4 / 3, abs=2e-7)
    assert abs(result["candidate_variation_l2"].item() - 1) > 0.3
    assert result["candidate_quadrature_evaluations"].item() <= config.max_evaluations
    assert result["candidate_integral_status"] == ["estimated_converged"]
    assert result["candidate_integral_identity_residual"].abs().item() < 1e-3


def test_norm_is_inside_integral_with_interior_cancellation():
    # Current mean is zero. Symmetric segment means integrate to zero, while
    # their norm has strictly positive average.
    result = measure(
        conditional=1.0,
        unconditional=-1.0,
        matched=-1.0,
        guidance=2.0,
        kappa=0.5,
        destination_alpha=1.0,
        destination_sigma=0.2,
    )
    beta = 1 / 0.2**2
    expected = np.logaddexp(beta, -beta) / beta - math.log(2) / beta
    assert result["candidate_variation_l2"].item() == pytest.approx(expected, abs=1e-6)
    assert result["candidate_variation_l2"].item() > 0.9


def test_budget_exhaustion_retains_variation_and_unresolved_status():
    result = measure(
        conditional=1.0,
        unconditional=-1.0,
        matched=-0.431,
        guidance=2.0,
        kappa=0.5,
        destination_alpha=1.0,
        destination_sigma=0.1,
        config=IntegrationConfig(
            max_evaluations=15,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-12,
            identity_absolute_tolerance=1e-12,
        ),
    )
    assert result["candidate_variation_l2"].item() > 0
    assert result["candidate_quadrature_budget_exhausted"].item()
    assert result["candidate_condition_status"] == ["numerically_unresolved"]
    assert result["feedback_status"] == ["quadrature_budget_exhausted"]


def test_nearzero_margin_and_source_precision_are_unresolved():
    bank = FiniteSupport.from_candidates(
        [
            ("target", torch.tensor([0.0])),
            ("left", torch.tensor([-1.0])),
            ("right", torch.tensor([1.0])),
        ]
    )
    result = measure(conditional=0.0, unconditional=0.0, support=bank)
    assert result["candidate_gain_status"] == ["zero"]
    assert result["candidate_condition_margin_l2"].item() == 0
    assert result["candidate_condition_status"] == ["numerically_unresolved"]
    # A known positive condition is deliberately smaller than the supplied,
    # predeclared source-rounding sensitivity.
    result = measure(source_error_l2=1.0)
    assert result["candidate_condition_margin_l2"].item() > 0
    assert result["candidate_condition_status"] == ["numerically_unresolved"]
    assert result["candidate_gain_status"] == ["positive"]


def test_saturation_preserves_strict_sign_without_inventing_gain():
    result = measure(
        current=1.0,
        conditional=1.0,
        unconditional=0.0,
        matched=1.0,
        current_alpha=1.0,
        current_sigma=0.04,
        destination_alpha=1.0,
        destination_sigma=0.04,
    )
    assert result["candidate_matched_log_probability"].item() == 0
    assert result["candidate_guided_log_probability"].item() == 0
    assert result["candidate_log_probability_gain"].item() == 0
    assert result["candidate_log_odds_gain"].item() > 0
    assert result["candidate_gain_status"] == ["positive"]
    assert result["candidate_gain_underflow"].item()
    assert result["candidate_gain_saturated"].item()
    assert result["candidate_matched_log_complement"].item() < -1000
    negative = measure(
        current=1.0,
        conditional=0.0,
        unconditional=1.0,
        matched=2.0,
        current_alpha=1.0,
        current_sigma=0.04,
        destination_alpha=1.0,
        destination_sigma=0.04,
    )
    assert negative["candidate_log_probability_gain"].item() == 0
    assert negative["candidate_gain_status"] == ["negative"]


@pytest.mark.parametrize(
    "query_chunk,node_chunk,candidate_chunk", [(1, 1, 1), (3, 7, 2), (20, 256, 20)]
)
def test_batched_chunked_agreement_and_nonfinite_row_isolation(
    query_chunk, node_chunk, candidate_chunk
):
    bank = binary_support(query_chunk=query_chunk, candidate_chunk=candidate_chunk)
    config = IntegrationConfig(node_chunk=node_chunk)
    result = measure(
        current=[0.0, 0.0, math.nan],
        conditional=[1.0, -1.0, 1.0],
        unconditional=[0.0, 0.0, 0.0],
        matched=[0.0, 0.0, 0.0],
        support=bank,
        config=config,
    )
    assert result["candidate_condition_status"][:2] == [
        "estimated_met",
        "estimated_not_met",
    ]
    assert result["candidate_gain_status"][:2] == ["positive", "negative"]
    assert result["candidate_condition_status"][2] == "unavailable"
    assert torch.isnan(result["candidate_variation_l2"][2])
    assert result["candidate_variation_l2"][0].item() == pytest.approx(
        measure()["candidate_variation_l2"].item(), abs=1e-12
    )


def test_direct_and_gram_distances_and_integrals_agree():
    generator = torch.Generator().manual_seed(8)
    atoms = torch.randn(8, 11, dtype=torch.float64, generator=generator)
    bank = FiniteSupport.from_candidates(
        [(str(i), atom) for i, atom in enumerate(atoms)]
    )
    left = torch.randn(13, 8, dtype=torch.float64, generator=generator).softmax(1)
    right = torch.randn(13, 8, dtype=torch.float64, generator=generator).softmax(1)
    gram, allowance, invalid = mean_distance_from_weights(bank, left - right)
    direct, _, _ = mean_distance_from_weights(bank, left - right, use_gram=False)
    assert torch.allclose(gram, direct, atol=1e-12, rtol=1e-12)
    assert not invalid.any()
    assert torch.all(allowance >= 0)
    direct_result = measure(
        current=0.4, matched=-0.2, config=IntegrationConfig(use_gram=False)
    )
    gram_result = measure(current=0.4, matched=-0.2)
    assert direct_result["candidate_variation_l2"].item() == pytest.approx(
        gram_result["candidate_variation_l2"].item(), abs=1e-12
    )


def test_gram_only_clamps_small_negative_roundoff():
    bank = binary_support()
    difference = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    mean_distance_from_weights(bank, difference)
    bank._feedback_centered_gram = -torch.eye(2, dtype=torch.float64)
    bank._feedback_gram_max_abs = torch.tensor(1.0, dtype=torch.float64)
    distance, _, invalid = mean_distance_from_weights(bank, difference)
    assert invalid.item()
    assert torch.isnan(distance).item()


@pytest.mark.parametrize(
    "kind", ["missing_target", "single_atom", "zero_noise", "nonpositive_kappa"]
)
def test_unavailable_comparisons_keep_identical_scalar_schema(kind):
    arguments = {}
    if kind == "missing_target":
        arguments["target_id"] = "missing"
    elif kind == "single_atom":
        arguments["support"] = FiniteSupport.from_candidates(
            [("target", torch.tensor([1.0]))]
        )
    elif kind == "zero_noise":
        arguments["destination_sigma"] = 0.0
    else:
        arguments["kappa"] = 0.0
    result = measure(**arguments)
    assert (
        set(result)
        == set(measure())
        == set(unavailable_feedback(1, reason="terminal_update"))
    )
    assert not result["feedback_eligible"].item()
    assert torch.isnan(result["candidate_variation_l2"]).all()


def test_posterior_log_complement_declared_weights_and_target_unavailable():
    bank = FiniteSupport(
        torch.tensor([[1.0], [-1.0]]),
        ["target", "other"],
        {"target": 0, "other": 1},
        weights=[1.0, 3.0],
    )
    result = bank.evaluate(vec(0), "target", 0.8, 0.6, unconditional=vec(0))
    assert result["target_log_probability"].exp().item() == pytest.approx(0.25)
    assert result["target_log_complement"].exp().item() == pytest.approx(0.75)
    assert result["candidate_unconditional_reference_error_l2"].item() == pytest.approx(
        0.5
    )
    assert result["candidate_radius_l2"].item() == 2
    missing = bank.evaluate(vec(0), "missing", 0.8, 0.6)
    assert "absent" in missing["support_status"]
    assert torch.isnan(missing["target_log_probability"]).all()


def test_duplicate_records_do_not_add_mass_to_the_atom():
    bank = FiniteSupport.from_candidates(
        [
            ("target", torch.tensor([1.0])),
            ("alias", torch.tensor([1.0])),
            ("other", torch.tensor([-1.0])),
        ]
    )
    result = bank.evaluate(vec(0), "target", 0.8, 0.6)
    assert bank.size == 2
    assert result["target_log_probability"].exp().item() == pytest.approx(0.5)
    assert bank.aliases["target"] == bank.aliases["alias"]


def test_mismatched_segment_is_qa_failure_not_a_theorem_refutation():
    bank = binary_support()
    result = proposition5_feedback(
        bank,
        vec(0),
        vec(1),
        vec(0),
        vec(0),
        vec(-0.4),
        "target",
        current_alpha=0.8,
        current_sigma=0.6,
        destination_alpha=0.9,
        destination_sigma=0.4,
        guidance=2,
        kappa=0.2,
    )
    assert result["candidate_integral_qa_status"] == ["numerical_inconsistency"]
    assert result["candidate_condition_status"] == ["numerically_unresolved"]
    assert result["candidate_gain_status"] == ["negative"]
    assert result["feedback_status"] == [
        "numerical_inconsistency_requires_investigation"
    ]


def test_guidance_outside_paper_domain_remains_explicit():
    result = measure(guidance=0.5)
    assert result["feedback_eligible"].item()
    assert result["candidate_manuscript_guidance_status"] == [
        "outside_manuscript_domain_analytic_positive_guidance_diagnostic"
    ]


def test_feedback_avoids_boolean_scatter_and_preserves_invalid_node_flags(monkeypatch):
    """Emulate CUDA's unsupported Bool scatter on CPU, including true flags."""
    from utils.experiments.theory import feedback

    original = torch.Tensor.scatter_reduce_
    calls = []

    def cuda_compatible_scatter(self, dim, index, source, *args, **kwargs):
        if self.dtype == torch.bool or source.dtype == torch.bool:
            raise RuntimeError("scatter_reduce_cuda not implemented for Bool")
        calls.append((self.dtype, source.dtype))
        return original(self, dim, index, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "scatter_reduce_", cuda_compatible_scatter)
    result = measure()
    assert result["candidate_condition_status"] == ["estimated_met"]
    assert result["candidate_integral_qa_status"] == [
        "consistent_with_numerical_estimates"
    ]
    assert (torch.int64, torch.int64) in calls

    # A node flagged invalid once must remain invalid across later node chunks.
    actual_distances = feedback.mean_distance_from_weights
    evaluations = 0

    def invalid_first_node(*args, **kwargs):
        nonlocal evaluations
        value, allowance, invalid = actual_distances(*args, **kwargs)
        if evaluations == 0:
            invalid[0] = True
            value[0] = torch.nan
        evaluations += 1
        return value, allowance, invalid

    monkeypatch.setattr(feedback, "mean_distance_from_weights", invalid_first_node)
    result = measure(config=IntegrationConfig(node_chunk=1))
    assert evaluations > 1
    assert result["candidate_condition_status"] == ["numerically_unresolved"]
    assert result["feedback_status"] == ["gram_squared_norm_outside_roundoff_allowance"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_candidate_feedback_matches_cpu_without_boolean_scatter():
    cpu_support = binary_support()
    cuda_support = FiniteSupport(
        cpu_support.atoms.to("cuda:0"),
        cpu_support.atom_ids,
        cpu_support.aliases,
        weights=cpu_support.weights.to("cuda:0"),
    )
    cases = [
        dict(
            current=[0.0, 0.0],
            conditional=[1.0, -1.0],
            unconditional=[0.0, 0.0],
            matched=[0.0, 0.0],
        ),
        dict(
            current=1.0,
            conditional=1.0,
            unconditional=0.0,
            matched=1.0,
            current_alpha=1.0,
            current_sigma=0.04,
            destination_alpha=1.0,
            destination_sigma=0.04,
        ),
    ]
    for case in cases:
        cpu = measure(support=cpu_support, **case)
        cuda = measure(support=cuda_support, **case)
        assert cpu.keys() == cuda.keys()
        for name, expected in cpu.items():
            actual = cuda[name]
            if isinstance(expected, torch.Tensor):
                assert actual.device.type == "cuda"
                if expected.dtype == torch.bool or not expected.is_floating_point():
                    assert torch.equal(actual.cpu(), expected), name
                else:
                    torch.testing.assert_close(
                        actual.cpu(),
                        expected,
                        rtol=1e-9,
                        atol=1e-9,
                        equal_nan=True,
                        msg=name,
                    )
            else:
                assert actual == expected, name
