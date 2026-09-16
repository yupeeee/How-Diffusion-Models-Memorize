"""Unexecuted regression specifications for fixed-cache numerical refinement.

Fixtures are finite laws and exact saved tensors. No model or experiment data
is needed. This file was authored without running tests or project imports.
"""
from decimal import Decimal
import math
from types import SimpleNamespace

import pytest
import torch

from utils.experiments.theory.numerical_intervals import (
    Directed, Interval, enclose_variation, interval_gain, negative_condition_precheck,
)
from utils.experiments.theory.gpu_intervals import GpuDirected, GpuInterval

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical backend required")

from utils.experiments.theory.numerical_refinement import (
    NumericalPolicy, build_refinement_payload, refine_payload,
    source_robust_gain_interval, stable_gain,
)


def support(atoms):
    flat = torch.tensor(atoms, dtype=torch.float64, device="cuda")
    return SimpleNamespace(flat=flat, weights=torch.full((len(flat),), 1 / len(flat), dtype=torch.float64, device="cuda"),
                           size=len(flat), dimension=flat.shape[1], candidate_chunk=1)


def points(values):
    return [Interval.point(value) for value in values]


def raw(*, endpoint=0.5, conditional_epsilon=-1.0, noise_std=0.0):
    return {"state": torch.tensor([[0.25]], dtype=torch.float64, device="cuda"),
            "epsilon_u": torch.tensor([[0.0]], dtype=torch.float64, device="cuda"),
            "epsilon_c": torch.tensor([[conditional_epsilon]], dtype=torch.float64, device="cuda"),
            "target": torch.tensor([0.0], dtype=torch.float64, device="cuda"),
            "saved_endpoint": torch.tensor([[endpoint]], dtype=torch.float64, device="cuda"),
            "alpha": .5, "sigma": .75, "A": .25, "kappa": .5,
            "guidance": 2.0, "destination_alpha": .8, "destination_sigma": .6,
            "noise_std": noise_std, "manuscript_domain": True}


def scalar_rows(**values):
    return [{"direct_prop5_applicable": True, "direct_prop5_variation_l2": None,
             "direct_prop5_margin_l2": None, "direct_prop5_gain_status": "numerically_unresolved",
             "direct_prop5_condition_status": "numerically_unresolved", **values}]


@requires_cuda
def test_large_absolute_logits_do_not_erase_small_net_gain():
    b = torch.tensor([[0.0, 1e16]], dtype=torch.float64, device="cuda")
    a = torch.tensor([[0.0, 1e-8]], dtype=torch.float64, device="cuda")
    result = stable_gain(b, a, 0)
    assert result["old_difference"].item() == 0
    assert result["G"].item() == pytest.approx(-1e-8, rel=1e-14)
    assert result["H"].item() == pytest.approx(-1e-8, rel=1e-14)
    assert result["flagged"].item()


@requires_cuda
def test_saturated_target_retains_zero_H_magnitude_and_nonzero_G_sign():
    result = stable_gain(torch.tensor([[0., -1000.]], dtype=torch.float64, device="cuda"),
                         torch.tensor([[0., -1.]], dtype=torch.float64, device="cuda"), 0)
    assert result["G"].item() > 0
    assert result["H"].item() == 0
    assert result["flagged"].item()
    gain = interval_gain(Directed(128), points([0., -1000.]), points([0., -1.]), 0)
    assert gain["G"].lower > 0
    assert gain["gain_sign"] == "positive"
    # H's interval may overlap zero at the declared precision; its magnitude
    # must never be replaced by the order-one G value.
    assert abs(float(gain["H"].upper)) < 1e-100


@requires_cuda
def test_H_uses_full_posterior_while_G_uses_only_non_target_normalization():
    result = stable_gain(torch.tensor([[0., 0.]], dtype=torch.float64, device="cuda"),
                         torch.tensor([[0., math.log(2)]], dtype=torch.float64, device="cuda"), 0)
    assert result["G"].item() == pytest.approx(-math.log(2))
    assert result["H"].item() == pytest.approx(math.log(2 / 3))
    assert result["G"].item() != result["H"].item()


def test_mixed_signed_small_slopes_refine_cancellation_without_clipping():
    c = Directed(128)
    gain = interval_gain(c, points([0., 0., 0.]), points([0., 1e-8, -1e-8]), 0)
    assert gain["G"].upper < 0
    # -log(cosh(a)) is negative, although the linear signed contributions cancel.
    assert gain["G"].lower < Decimal("-4.999999999e-17")
    assert gain["G"].upper > Decimal("-5.000000001e-17")
    assert gain["H"].upper < 0


@pytest.mark.parametrize("atoms", [1, 3])
@requires_cuda
def test_exact_zero_and_single_atom_have_explicit_gain_contracts(atoms):
    result = stable_gain(torch.zeros(2, atoms, dtype=torch.float64, device="cuda"),
                         torch.zeros(2, atoms, dtype=torch.float64, device="cuda"), 0)
    assert torch.equal(result["H"], torch.zeros(2, dtype=torch.float64, device="cuda"))
    assert result["zero"].all()
    if atoms == 1:
        assert result["G"].isnan().all()
    else:
        assert torch.equal(result["G"], torch.zeros(2, dtype=torch.float64, device="cuda"))
    enclosed = interval_gain(Directed(), points([0.] * atoms), points([0.] * atoms), 0)
    assert enclosed["H"].sign == "zero"
    if atoms == 1:
        assert enclosed["G"] is None
    else:
        assert enclosed["G"].sign == "zero"


def test_negative_precheck_resolves_sign_without_fabricating_V_or_M():
    result = negative_condition_precheck(Interval.point(1), Interval.point(2))
    assert result["upper"] == -1
    assert result["condition_sign_status"] == "negative"
    assert result["condition_value_status"] == "unavailable_variation_not_computed"
    assert "V" not in result and "M" not in result
    boundary = negative_condition_precheck(Interval.point(2), Interval.point(2))
    assert boundary["condition_sign_status"] == "unresolved"


def test_directed_basic_and_transcendental_bounds_contain_exact_identities():
    c = Directed(64)
    third = c.div(Interval.point(1), Interval.point(3))
    assert c.mul(third, Interval.point(3)).lower <= 1 <= c.mul(third, Interval.point(3)).upper
    square = c.square(Interval(Decimal(-2), Decimal(1)))
    assert square.lower == 0 and square.upper == 4
    root = c.sqrt(Interval.point(2))
    assert c.square(root).lower <= 2 <= c.square(root).upper
    roundtrip = c.log(c.exp(Interval.point(3)))
    assert roundtrip.lower <= 3 <= roundtrip.upper
    tiny = Interval(Decimal("1e-1000"), Decimal("2e-1000"))
    lo, hi = tiny.floats()
    assert lo == 0 and hi > 0


def test_global_enclosure_includes_sharp_interior_transition_missed_by_midpoint():
    c = Directed(80)
    atoms = [points([-1]), points([1])]
    # Posterior switches near s=.9; f(.5) is almost zero but V is about .2.
    result = enclose_variation(c, atoms, points([0, -180]), points([0, 200]),
        points([-1]), Interval.point(.1), Interval.point(0), Interval.point(0),
        max_nodes=1, absolute_width=1e-12)
    assert result["V"].lower <= Decimal("0.2") <= result["V"].upper
    assert result["M"].sign == "unresolved"
    assert result["nodes"] == 1
    assert result["stopping_reason"] == "node_budget_exhausted"
    assert result["V"].upper > 0
    # A single midpoint estimate cannot be presented as a certified zero V.
    assert "directed_decimal_nodes" in result["error_method"]


def test_variation_certifies_constant_reference_without_exceeding_even_budget():
    c = Directed(80)
    result = enclose_variation(c, [points([0]), points([2])], points([0, 0]), points([0, 0]),
        points([1]), Interval.point(1), Interval.point(0), Interval.point(0), max_nodes=2)
    assert result["V"].lower == 0
    assert result["V"].upper < Decimal("1e-60")
    assert result["M"].sign == "positive"
    assert result["nodes"] <= 2
    assert result["integrated_H"].lower <= 0 <= result["integrated_H"].upper


@requires_cuda
def test_builder_preserves_saved_and_affine_endpoints_and_cache_tensor_finiteness():
    law = support([[0.], [1.]])
    verified = raw(endpoint=-3.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    assert payload["endpoint_construction_method"] == "independent_deterministic_affine_baseline"
    assert not torch.equal(payload["slopes"], payload["saved_endpoint_slopes"])
    assert payload["endpoint_difference_l2"].item() > 0
    assert all(torch.isfinite(value).all() for value in payload.values() if isinstance(value, torch.Tensor))
    output = refine_payload(law, payload, scalar_rows(), policy=NumericalPolicy(max_decimal_products=0), verified_inputs=verified)[0]
    assert output["numerical_condition_endpoint_contract"] == "original_affine_path_with_current_reference"
    assert output["numerical_gain_endpoint_contract"] == "saved_endpoint"
    assert output["numerical_log_odds_gain"] != output["numerical_affine_log_odds_gain"]
    assert output["implication_eligible"] is False
    assert output["numerical_stopping_reason"] == "negative_condition_batched_screen"
    assert output["numerical_gpu_products"] == 0
    assert output["condition_sign_status"] == "negative"


@requires_cuda
def test_stochastic_constructed_baseline_is_not_independent_verification():
    payload = build_refinement_payload(support([[0.], [1.]]), raw(noise_std=.1), bank_hash="law", target_atom=0)
    assert "not_independent_replay" in payload["endpoint_construction_method"]
    verified = raw(noise_std=.1)
    verified["independent_innovation"] = torch.zeros(1, 1, dtype=torch.float64, device="cuda")
    verified["innovation_provenance"] = "recovered_from_endpoint"
    with pytest.raises(ValueError, match="independent"):
        build_refinement_payload(support([[0.], [1.]]), verified, bank_hash="law", target_atom=0)


@requires_cuda
def test_raw_negative_precheck_ignores_heuristic_source_envelope_and_preserves_missingness():
    law = support([[0.], [1.]])
    verified = raw(endpoint=.3125, conditional_epsilon=0.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    output = refine_payload(law, payload, scalar_rows(direct_prop5_source_sensitivity_l2=1e9),
                            verified_inputs=verified)[0]
    assert output["condition_sign_status"] == "negative"
    assert output["condition_arithmetic_status"] == "certified_original_saved_inputs"
    assert math.isnan(output["numerical_variation_l2"])
    assert math.isnan(output["numerical_original_margin_l2"])
    assert output["numerical_variation_nodes"] == 0
    assert output["source_robust_gain_sign"] == "unresolved"
    assert output["numerical_source_sensitivity_l2"] == 1e9


@requires_cuda
def test_reduced_payload_certificate_is_not_an_original_tensor_certificate():
    law = support([[0.], [1.]])
    payload = build_refinement_payload(law, raw(conditional_epsilon=0.), bank_hash="law", target_atom=0)
    output = refine_payload(law, payload, scalar_rows())[0]
    assert output["condition_sign_status"] == "negative"
    assert "reduced_scalar" in output["condition_arithmetic_status"]
    assert "original_construction_unenclosed" in output["input_contract_status"]
    assert output["implication_eligible"] is False


@requires_cuda
def test_source_robust_gain_requires_justified_radii_and_keeps_fixed_sign():
    arithmetic = GpuDirected(device="cuda")
    gain = arithmetic.interval(.9, 1.1)
    with pytest.raises(ValueError, match="justified"):
        source_robust_gain_interval(gain, beta_radius=2., endpoint_radii=[1., 1.],
                                   justification={"status": "dtype_epsilon_multiplier"}, arithmetic=arithmetic)
    robust = source_robust_gain_interval(gain, beta_radius=2., endpoint_radii=[.5, .5],
        justification={"status": "justified_bound", "scope": "endpoint_locations_only_fixed_law_and_coefficients",
                       "derivation": "Fixture independently supplies exact deterministic radius bounds"}, arithmetic=arithmetic)
    assert gain.sign == "positive" and robust.sign == "unresolved"


@requires_cuda
def test_structural_exclusion_and_budgets_are_explicit():
    law = support([[0.], [1.]])
    verified = raw()
    verified["manuscript_domain"] = False
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    output = refine_payload(law, payload, scalar_rows(), verified_inputs=verified)[0]
    assert output["fixed_cache_gain_sign"] == output["condition_sign_status"] == "not_applicable"
    assert not output["refinement_applicable"]
    assert output["numerical_gpu_products"] == 0
    verified["manuscript_domain"] = True
    verified["epsilon_c"] = torch.ones_like(verified["epsilon_c"])  # D > ec: screen is inconclusive.
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    output = refine_payload(law, payload, scalar_rows(), verified_inputs=verified,
                            policy=NumericalPolicy(max_decimal_products=1, max_variation_nodes=2))[0]
    assert output["numerical_gpu_products"] <= 1
    assert output["numerical_variation_nodes"] <= 2
    assert output["numerical_stopping_reason"] == "gpu_operation_budget_exhausted_before_raw_enclosure"
    assert math.isnan(output["numerical_variation_l2"])


@requires_cuda
def test_legacy_numerical_failure_does_not_remove_structurally_valid_denominator():
    law = support([[0.], [1.]])
    verified = raw(endpoint=.3125, conditional_epsilon=0.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    old = scalar_rows(direct_prop5_applicable=False, direct_prop5_status="failed_integration")
    output = refine_payload(law, payload, old, verified_inputs=verified)[0]
    assert output["refinement_applicable"] is True
    assert output["condition_sign_status"] == "negative"
    assert output["condition_value_status"] == "unavailable_variation_not_computed"
    assert old[0]["direct_prop5_applicable"] is False


def test_signed_gain_integral_is_invariant_to_a_common_logit_slope():
    kwargs = dict(atoms=[points([0]), points([2])], intercept=points([0, 0]),
                  current_mean=points([1]), D=Interval.point(1), ec=Interval.point(0),
                  eu=Interval.point(0), max_nodes=1)
    result = enclose_variation(Directed(80), slopes=points([7, 7]), **kwargs)
    assert result["integrated_H"].lower <= 0 <= result["integrated_H"].upper
    assert abs(float(result["integrated_H"].upper)) < 1e-60


def test_rounded_current_weights_do_not_silently_acquire_a_convex_hull_cap():
    # The supplied reference is 1.5 times the sole atom, so V=.5 exceeds the
    # support diameter zero. The explicit mass-defect term must retain it.
    result = enclose_variation(Directed(80), [points([1])], points([0]), points([0]),
        points([1.5]), Interval.point(0), Interval.point(0), Interval.point(0),
        max_nodes=1, current_in_convex_hull=False, current_mass_defect=Interval.point(.5))
    assert result["V"].lower <= Decimal("0.5") <= result["V"].upper
    assert result["M"].upper < 0
    with pytest.raises(ValueError, match="mass-defect"):
        enclose_variation(Directed(), [points([1])], points([0]), points([0]),
            points([1.5]), Interval.point(0), Interval.point(0), Interval.point(0),
            current_in_convex_hull=False)


@requires_cuda
def test_batched_negative_screen_avoids_selective_interval_work(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law, verified = support([[0.], [1.]]), raw()
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("A resolved batched row entered selective GPU enclosure")
    monkeypatch.setattr(engine, "_raw_row", forbidden)
    monkeypatch.setattr(engine, "_clean_intervals", forbidden)
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified)[0]
    assert result["condition_sign_status"] == "negative"
    assert result["condition_arithmetic_status"] == "certified_original_saved_inputs"
    assert result["arithmetic_status"] == "float64_numerical_assessment"
    assert result["numerical_condition_screen_squared_upper"] < 0
    assert math.isnan(result["numerical_margin_upper_l2"])
    assert math.isnan(result["numerical_original_margin_l2"])
    assert math.isnan(result["numerical_variation_l2"])
    assert result["numerical_gpu_products"] == result["numerical_variation_nodes"] == 0
    assert not result["numerical_gpu_refinement_selected"]
    assert not result["numerical_gpu_refinement_executed"]


@pytest.mark.parametrize("b,a", [([0., 1e16], [0., 1e-8]), ([0., -1000.], [0., -1.])])
@requires_cuda
def test_saturation_and_unstable_subtraction_do_not_trigger_gpu_enclosure(b, a):
    from utils.experiments.theory.numerical_refinement import _gain_retry_needed
    assessed = stable_gain(torch.tensor([b], dtype=torch.float64, device="cuda"), torch.tensor([a], dtype=torch.float64, device="cuda"), 0)
    values = {name: value.tolist() for name, value in assessed.items()}
    assert values["flagged"][0]
    assert not _gain_retry_needed(values, 0, original_inputs=True, singleton=False)
    assert abs(values["G"][0]) > values["tolerance"][0]


@pytest.mark.parametrize("changes", [
    {"G": [float("nan")]}, {"H": [float("inf")]}, {"G": [0.]},
    {"H": [-1.]}, {"zero": [True]},
])
def test_unresolved_or_inconsistent_stable_gain_requires_gpu_enclosure(changes):
    from utils.experiments.theory.numerical_refinement import _gain_retry_needed
    values = {"G": [1.], "H": [.5], "tolerance": [1e-10], "H_tolerance": [1e-10], "zero": [False]}
    assert _gain_retry_needed(values | changes, 0, original_inputs=True, singleton=False)


@requires_cuda
def test_budget_preflight_keeps_gpu_negative_proof_without_raw_conversion(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law, verified = support([[0.], [1.]]), raw(endpoint=.3125, conditional_epsilon=0.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("An unaffordable fallback converted a raw row")
    monkeypatch.setattr(engine, "_raw_row", forbidden)
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified,
                            policy=NumericalPolicy(max_decimal_products=1))[0]
    assert result["condition_sign_status"] == "negative"
    assert result["fixed_cache_gain_sign"] == "unresolved"  # No rounded-zero shortcut.
    assert result["numerical_gpu_refinement_selected"]
    assert not result["numerical_gpu_refinement_executed"]
    assert result["numerical_stopping_reason"] == "gpu_operation_budget_exhausted_before_raw_enclosure"
    assert result["numerical_gpu_products"] == 0
    assert math.isnan(result["numerical_variation_l2"])


@requires_cuda
def test_unresolved_condition_uses_directed_cuda_refinement(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=1.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    original, calls = engine._raw_row, []
    def track(inputs, index, **kwargs):
        calls.append(index)
        return original(inputs, index, **kwargs)
    monkeypatch.setattr(engine, "_raw_row", track)
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified)[0]
    assert result["numerical_condition_screen_status"] == "inconclusive"
    assert calls == [0]
    assert result["numerical_gpu_refinement_executed"]
    assert result["numerical_gpu_products"] > 0
    assert result["condition_arithmetic_status"] == "certified_original_saved_inputs"


@pytest.mark.parametrize("atoms", [[[0.]], [[0.], [1.]], [[0., 0.], [1., 2.], [-1., 3.]]])
@requires_cuda
def test_posterior_budget_floor_never_exceeds_actual_unavoidable_projection_charges(atoms):
    from utils.experiments.theory.numerical_refinement import _clean_intervals, _raw_segment, _raw_row, _posterior_work_floor
    dimension = len(atoms[0])
    verified = raw()
    for key in ("state", "epsilon_u", "epsilon_c", "saved_endpoint"):
        verified[key] = verified[key].repeat(1, dimension)
    verified["target"] = torch.zeros(dimension, dtype=torch.float64, device="cuda")
    witness = _raw_row(verified, 0, device="cuda")
    context = GpuDirected(2_000_000, device="cuda")
    clean = _clean_intervals(context, witness)
    before = context.products
    _raw_segment(context, witness, context.interval(atoms), context.interval([1. / len(atoms)] * len(atoms)), 0, clean)
    assert context.products - before >= _posterior_work_floor(len(atoms), dimension, original_inputs=True)


@requires_cuda
def test_proven_D_above_ec_skips_unaffordable_posterior_without_claiming_margin_sign(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=1.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Impossible posterior budget entered raw GPU enclosure")
    monkeypatch.setattr(engine, "_raw_row", forbidden)
    # Enough for clean-vector operations, insufficient for required projection.
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified,
                            policy=NumericalPolicy(max_decimal_products=45))[0]
    assert result["numerical_D_ge_ec_certified"]
    assert result["numerical_condition_screen_squared_lower"] > 0
    assert result["condition_sign_status"] == "unresolved"
    assert not result["numerical_condition_nonnegative_certified"]
    assert result["numerical_gpu_products"] == 0
    assert result["numerical_gpu_refinement_selected"] and not result["numerical_gpu_refinement_executed"]


def test_production_refinement_rejects_cpu_before_computing():
    law = SimpleNamespace(flat=torch.tensor([[0.]], device="cpu"), dimension=1, size=1)
    with pytest.raises(ValueError, match="CUDA.*CPU fallback"):
        build_refinement_payload(law, {}, bank_hash="law", target_atom=0)
    with pytest.raises(ValueError, match="CUDA.*CPU fallback"):
        refine_payload(law, {}, [])


def test_policy_reports_fixed_gpu_precision_and_no_cpu_fallback():
    identity = NumericalPolicy(decimal_precision=128).identity()
    assert identity["effective_mantissa_bits"] == 53
    assert identity["retry_precision_multiplier"] == 1
    assert identity["cpu_fallback"] is False
    assert "legacy_configuration_only" in identity["decimal_precision_role"]
