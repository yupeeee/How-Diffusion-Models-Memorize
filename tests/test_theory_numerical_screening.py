"""Unexecuted specifications for batched exact-input condition screening."""
from fractions import Fraction
import math

import pytest
import torch

from utils.experiments.theory.numerical_screening import screen_negative_condition


def saved(state, epsilon_u, epsilon_c, target, *, alpha=.5, sigma=.75):
    return {"state": torch.tensor(state, dtype=torch.float64),
            "epsilon_u": torch.tensor(epsilon_u, dtype=torch.float64),
            "epsilon_c": torch.tensor(epsilon_c, dtype=torch.float64),
            "target": torch.tensor(target, dtype=torch.float64),
            "alpha": alpha, "sigma": sigma, "stored_prediction_type": "epsilon"}


def exact_squared_quantities(inputs, row):
    # Fractions are an exact, independent reference for finite binary inputs.
    alpha, sigma = (Fraction.from_float(float(inputs[key])) for key in ("alpha", "sigma"))
    state, u, c = [inputs[key].reshape(len(inputs[key]), -1)[row].tolist()
                   for key in ("state", "epsilon_u", "epsilon_c")]
    target = inputs["target"].reshape(-1).tolist()
    delta, residual = Fraction(0), Fraction(0)
    for z, eu, ec, t in zip(state, u, c, target):
        z, eu, ec, t = map(Fraction.from_float, (z, eu, ec, t))
        delta += (sigma * (eu - ec)) ** 2
        residual += (z - sigma * ec - alpha * t) ** 2
    return delta, residual


def test_negative_screen_is_sufficient_only_and_does_not_fabricate_norm_margin():
    inputs = saved([[2.], [0.], [0.]], [[0.], [2.], [0.]], [[0.], [0.], [0.]], [0.])
    result = screen_negative_condition(inputs)
    assert result["certified_negative"].tolist() == [True, False, False]
    assert result["arithmetic_resolved"].all()
    assert result["squared_difference_upper"][0] < 0
    assert result["squared_difference_upper"][2] == 0
    assert not {"M", "V", "D_upper", "ec_lower"}.intersection(result)


@pytest.mark.parametrize("dimension", [1, 3, 8, 17, 65])
def test_pairwise_bounds_enclose_exact_rational_norms_for_odd_and_even_dimensions(dimension):
    state = [[((i * 17 + row * 3) % 23 - 11) / 7 for i in range(dimension)] for row in range(3)]
    u = [[((i * 7 + row) % 13 - 6) / 11 for i in range(dimension)] for row in range(3)]
    c = [[((i * 3 + row) % 17 - 8) / 13 for i in range(dimension)] for row in range(3)]
    inputs = saved(state, u, c, [((i * 5) % 7 - 3) / 19 for i in range(dimension)], alpha=.123, sigma=.789)
    result = screen_negative_condition(inputs)
    assert result["arithmetic_resolved"].all()
    for row in range(3):
        delta, residual = exact_squared_quantities(inputs, row)
        assert Fraction.from_float(result["delta_squared_lower"][row].item()) <= delta
        assert Fraction.from_float(result["delta_squared_upper"][row].item()) >= delta
        assert Fraction.from_float(result["residual_squared_lower"][row].item()) <= residual
        assert Fraction.from_float(result["residual_squared_upper"][row].item()) >= residual
        assert Fraction.from_float(result["squared_difference_lower"][row].item()) <= delta - residual
        assert Fraction.from_float(result["squared_difference_upper"][row].item()) >= delta - residual
        if result["certified_negative"][row]:
            assert delta < residual


def test_equality_is_not_a_negative_certificate_even_with_cancellation():
    inputs = saved([[0.]], [[0.]], [[2.]], [0.], alpha=.5, sigma=.75)
    result = screen_negative_condition(inputs)
    assert not result["certified_negative"].item()
    assert result["squared_difference_upper"].item() >= 0


def test_large_common_state_terms_do_not_change_direct_epsilon_difference_contract():
    inputs = saved([[2.0**50, -(2.0**50)]], [[1., -1.]], [[1. + 2.0**-40, -1.]], [0., 0.])
    result = screen_negative_condition(inputs)
    delta, residual = exact_squared_quantities(inputs, 0)
    assert result["certified_negative"].item()
    assert Fraction.from_float(result["delta_squared_upper"].item()) >= delta
    assert Fraction.from_float(result["residual_squared_lower"].item()) <= residual


@pytest.mark.parametrize("value", [math.inf, math.nan, 1e308, math.ulp(0.0), 1e-200])
def test_exceptional_or_underflowing_arithmetic_is_unresolved_per_row(value):
    inputs = saved([[value], [2.]], [[0.], [0.]], [[0.], [0.]], [0.])
    result = screen_negative_condition(inputs)
    assert result["arithmetic_resolved"].tolist() == [False, True]
    assert result["certified_negative"].tolist() == [False, True]
    assert result["squared_difference_upper"][0].isnan()


def test_zero_noise_and_latent_shapes_are_supported_without_posterior_data():
    inputs = saved([[1., 2., 3., 4.]], [[5., 6., 7., 8.]], [[9., 10., 11., 12.]], [0., 0., 0., 0.], sigma=0.)
    for key in ("state", "epsilon_u", "epsilon_c"):
        inputs[key] = inputs[key].reshape(1, 1, 2, 2)
    inputs["target"] = inputs["target"].reshape(1, 2, 2)
    assert screen_negative_condition(inputs)["certified_negative"].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_and_cuda_both_enclose_identical_exact_saved_inputs():
    inputs = saved([[2., -.25, .125], [0., 0., 0.]], [[.125, .25, -.5], [0., 0., 0.]],
                   [[-.25, .125, .5], [0., 0., 0.]], [.2, -.3, .4], alpha=.123, sigma=.789)
    for device in ("cpu", "cuda"):
        result = screen_negative_condition(inputs, device=device)
        assert result["arithmetic_resolved"].all()
        for row in range(2):
            delta, residual = exact_squared_quantities(inputs, row)
            assert Fraction.from_float(result["delta_squared_lower"][row].item()) <= delta
            assert Fraction.from_float(result["delta_squared_upper"][row].item()) >= delta
            assert Fraction.from_float(result["residual_squared_lower"][row].item()) <= residual
            assert Fraction.from_float(result["residual_squared_upper"][row].item()) >= residual
            assert Fraction.from_float(result["squared_difference_lower"][row].item()) <= delta - residual
            assert Fraction.from_float(result["squared_difference_upper"][row].item()) >= delta - residual
            if result["certified_negative"][row]:
                assert delta < residual


def test_unsupported_cpu_denormal_mode_cannot_produce_a_certificate(monkeypatch):
    from utils.experiments.theory import numerical_screening
    monkeypatch.setattr(numerical_screening, "_cpu_denormal_mode_is_supported", lambda: False)
    result = screen_negative_condition(saved([[2.]], [[0.]], [[0.]], [0.]))
    assert not result["arithmetic_resolved"].item()
    assert not result["certified_negative"].item()
    assert result["squared_difference_upper"].isnan().item()


def test_exact_conditional_clean_match_cannot_have_a_negative_certificate():
    # z = sigma*epsilon_c + alpha*target, so ec is exactly zero.
    result = screen_negative_condition(saved([[2.]], [[3.]], [[2.]], [1.], alpha=.5, sigma=.75))
    assert not result["certified_negative"].item()
    assert result["residual_squared_lower"].item() == 0


def test_positive_squared_comparison_only_rules_out_the_cheap_negative_precheck():
    result = screen_negative_condition(saved([[0.]], [[2.]], [[0.]], [0.]))
    assert result["arithmetic_resolved"].item()
    assert result["squared_difference_lower"].item() > 0
    assert not result["certified_negative"].item()
    # Neither eu nor V was supplied, so no full-condition positive certificate
    # or margin can follow from the sign of D² - ec².
    assert "certified_positive" not in result
    assert "condition_nonnegative" not in result
    assert "M" not in result
