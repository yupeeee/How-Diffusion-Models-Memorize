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
        points([-1]), Interval.point(.1), Interval.point(0),
        delta=points([.1]), max_nodes=1, absolute_width=1e-12)
    assert result["V"].lower <= Decimal("0.2") <= result["V"].upper
    assert result["M"].sign == "unresolved"
    assert result["nodes"] == 1
    assert result["stopping_reason"] == "node_budget_exhausted"
    assert result["V"].upper > 0
    # A single midpoint estimate cannot be presented as a certified zero V.
    assert "directed_decimal_signed_projection_nodes" in result["error_method"]


def test_variation_certifies_constant_reference_without_exceeding_even_budget():
    c = Directed(80)
    result = enclose_variation(c, [points([0]), points([2])], points([0, 0]), points([0, 0]),
        points([1]), Interval.point(1), Interval.point(0), delta=points([1]), max_nodes=2)
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
    assert output["numerical_stopping_reason"] == "gpu_refinement_disabled_by_zero_budget"
    assert output["numerical_gpu_products"] == 0
    assert output["condition_sign_status"] == "unresolved"
    assert payload["condition_contract"] == output["numerical_condition_contract"] == "projected-gap-error-1"
    expected_reference = payload["current_weights"] @ law.flat
    difference = payload["delta"] - (law.flat[0] - expected_reference)
    expected_error = (payload["delta"] * difference).sum(-1) / payload["delta_norm"]
    torch.testing.assert_close(payload["branch_gap_error"], expected_error)
    torch.testing.assert_close(payload["branch_gap_error_norm_diagnostic"], difference.norm(dim=-1))


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
    verified = raw(endpoint=.6875, conditional_epsilon=-.25)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    output = refine_payload(law, payload, scalar_rows(direct_prop5_source_sensitivity_l2=1e9),
                            verified_inputs=verified, policy=NumericalPolicy(max_decimal_products=2_000_000))[0]
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
    payload = build_refinement_payload(law, raw(conditional_epsilon=-.25), bank_hash="law", target_atom=0)
    output = refine_payload(law, payload, scalar_rows(), policy=NumericalPolicy(max_decimal_products=2_000_000))[0]
    assert output["condition_sign_status"] == "negative"
    assert output["condition_arithmetic_status"] == "certified_reduced_payload_only"
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
    verified["epsilon_c"] = torch.ones_like(verified["epsilon_c"])  # Posterior-derived joint error is still required.
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
    verified = raw(endpoint=.6875, conditional_epsilon=-.25)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    old = scalar_rows(direct_prop5_applicable=False, direct_prop5_status="failed_integration")
    output = refine_payload(law, payload, old, verified_inputs=verified, policy=NumericalPolicy(max_decimal_products=2_000_000))[0]
    assert output["refinement_applicable"] is True
    assert output["condition_sign_status"] == "negative"
    assert output["condition_value_status"] == "unavailable_variation_not_computed"
    assert old[0]["direct_prop5_applicable"] is False


def test_signed_gain_integral_is_invariant_to_a_common_logit_slope():
    kwargs = dict(atoms=[points([0]), points([2])], intercept=points([0, 0]),
                  current_mean=points([1]), D=Interval.point(1), branch_gap_error=Interval.point(0), delta=points([1]), max_nodes=1)
    result = enclose_variation(Directed(80), slopes=points([7, 7]), **kwargs)
    assert result["integrated_H"].lower <= 0 <= result["integrated_H"].upper
    assert abs(float(result["integrated_H"].upper)) < 1e-60


def test_rounded_current_weights_do_not_silently_acquire_a_convex_hull_cap():
    # The supplied reference is 1.5 times the sole atom, so V=.5 exceeds the
    # support diameter zero. The projected range about the supplied current
    # mean must retain it instead of assuming that mean lies in the hull.
    result = enclose_variation(Directed(80), [points([1])], points([0]), points([0]),
        points([1.5]), Interval.point(1), Interval.point(1), delta=points([-1]),
        max_nodes=1, current_in_convex_hull=False, current_mass_defect=Interval.point(.5))
    assert result["V"].lower <= Decimal("0.5") <= result["V"].upper
    assert result["M"].upper < 0
    with pytest.raises(ValueError, match="mass-defect"):
        enclose_variation(Directed(), [points([1])], points([0]), points([0]),
            points([1.5]), Interval.point(1), Interval.point(1), delta=points([-1]),
            current_in_convex_hull=False)


@requires_cuda
def test_conditional_error_shortcut_cannot_override_branch_gap_error_condition(monkeypatch):
    # Large shared learned errors cancel in Delta-barDelta. D < e_c therefore
    # cannot establish the sign of the tighter branch-gap-error condition.
    law, verified = support([[0.], [1.]]), raw(endpoint=11., conditional_epsilon=1.)
    verified["state"].fill_(10.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    assert payload["branch_gap_error"].item() < payload["delta_norm"].item() < payload["conditional_error"].item()
    from utils.experiments.theory import numerical_refinement as engine
    original_screen, original_segment = engine.screen_branch_gap_condition, engine._raw_segment
    screen_calls, segment_calls = [], []
    def screen(*args, **kwargs):
        value = original_screen(*args, **kwargs)
        screen_calls.append(value)
        return value
    def segment(*args, **kwargs):
        current = kwargs.get("current_reference")
        assert current is not None
        assert all(isinstance(current[key], GpuInterval) for key in ("current_mean", "D", "branch_gap_error"))
        value = original_segment(*args, **kwargs)
        # The current posterior/error enclosure is reused, not recomputed.
        assert value["current_mean"] is current["current_mean"]
        assert value["D"] is current["D"]
        assert value["branch_gap_error"] is current["branch_gap_error"]
        segment_calls.append(current)
        return value
    monkeypatch.setattr(engine, "screen_branch_gap_condition", screen)
    monkeypatch.setattr(engine, "_raw_segment", segment)
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified, policy=NumericalPolicy(max_decimal_products=2_000_000))[0]
    assert len(screen_calls) == len(segment_calls) == 1
    assert result["numerical_endpoint_refinement_executed"]
    assert result["numerical_current_posterior_reused"]
    assert result["numerical_gpu_products"] > result["numerical_condition_screen_gpu_products"] > 0
    assert screen_calls[0]["complete"] and not bool(screen_calls[0]["certified_negative"][0])
    assert result["condition_sign_status"] == "positive"
    assert result["numerical_margin_lower_l2"] > 0.
    assert result["numerical_condition_screen_status"] == "inconclusive"
    assert result["numerical_gpu_refinement_executed"] and result["numerical_gpu_products"] > 0
    assert result["implication_eligible"]
    assert result["implication_audit_status"] == "checked_branch_gap_error_bound_with_proven_saved_endpoint_transfer"
    assert not result["numerical_publication_blocker"]


@requires_cuda
def test_old_condition_payload_cannot_be_reused_as_branch_gap_error_inputs():
    law, verified = support([[0.], [1.]]), raw()
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    payload["schema_version"] = 2
    payload.pop("condition_contract")
    payload.pop("branch_gap_error")
    with pytest.raises(ValueError, match="sufficient-input schema"):
        refine_payload(law, payload, scalar_rows(), verified_inputs=verified)


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
def test_budget_preflight_leaves_branch_gap_error_unknown_without_raw_conversion(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law, verified = support([[0.], [1.]]), raw(endpoint=.3125, conditional_epsilon=0.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("An unaffordable fallback converted a raw row")
    monkeypatch.setattr(engine, "_raw_row", forbidden)
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified,
                            policy=NumericalPolicy(max_decimal_products=1))[0]
    assert result["condition_sign_status"] == "unresolved"
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
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified, policy=NumericalPolicy(max_decimal_products=2_000_000))[0]
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
def test_branch_gap_error_requires_posterior_budget_without_claiming_margin_sign(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=1.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Impossible posterior budget entered raw GPU enclosure")
    monkeypatch.setattr(engine, "_raw_row", forbidden)
    # Enough for clean-vector operations, insufficient for required projection.
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified,
                            policy=NumericalPolicy(max_decimal_products=45))[0]
    assert not result["numerical_D_ge_ec_certified"]
    assert math.isnan(result["numerical_condition_screen_squared_lower"])
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
    assert NumericalPolicy().max_decimal_products == 0
    assert identity["version"] == "fixed-cache-numerics-cuda-8"
    assert identity["backend"] == "cuda_float64_assessment"
    assert identity["certification_enabled"] is False
    assert NumericalPolicy(max_decimal_products=2_000_000).identity()["certification_enabled"] is True
    assert identity["effective_mantissa_bits"] == 53
    assert identity["retry_precision_multiplier"] == 1
    assert identity["cpu_fallback"] is False
    assert identity["condition_contract"] == "projected-gap-error-1"
    assert "legacy_configuration_only" in identity["decimal_precision_role"]


@requires_cuda
def test_batched_negative_screen_skips_endpoint_gain_and_variation_enclosures(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law = support([[0.], [1.]])
    # Symmetric current likelihood gives reference=.5, D=.375, E=.875.
    # The actual affine endpoint is .6875; gain is nonzero and well resolved.
    verified = raw(endpoint=.6875, conditional_epsilon=-.25)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    expected = stable_gain(payload["intercept"], payload["saved_endpoint_slopes"], 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Certified negative screen entered endpoint/gain/quadrature work")
    for name in ("_raw_row", "_raw_segment", "interval_gain", "enclose_variation"):
        monkeypatch.setattr(engine, name, forbidden)
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified, policy=NumericalPolicy(max_decimal_products=2_000_000))[0]
    assert result["numerical_condition_screen_status"] == "certified_negative"
    assert result["condition_sign_status"] == "negative"
    assert result["condition_arithmetic_status"] == "certified_original_saved_inputs"
    assert result["numerical_condition_precheck_upper_l2"] < 0
    assert result["numerical_margin_upper_l2"] < 0
    assert result["numerical_stopping_reason"] == "negative_condition_precheck_batched_reference"
    assert result["numerical_gpu_refinement_selected"] and result["numerical_gpu_refinement_executed"]
    assert not result["numerical_endpoint_refinement_executed"]
    assert not result["numerical_current_posterior_reused"]
    assert result["numerical_condition_screen_batch_size"] == 1
    assert result["numerical_gpu_products"] == result["numerical_condition_screen_gpu_products"]
    assert result["condition_value_status"] == "unavailable_variation_not_computed"
    assert math.isnan(result["numerical_original_margin_l2"])
    assert math.isnan(result["numerical_variation_l2"])
    assert result["numerical_variation_nodes"] == 0
    assert result["fixed_cache_gain_sign"] == "negative"
    assert result["numerical_log_odds_gain"] == pytest.approx(expected["G"].item())
    assert result["numerical_log_probability_gain"] == pytest.approx(expected["H"].item())
    assert 0 < result["numerical_gpu_products"] <= 2_000_000
    assert not result["numerical_publication_blocker"]


@pytest.mark.parametrize("retry", ["gain_arithmetic", "source_perturbation"])
@requires_cuda
def test_negative_screen_still_honors_independent_gain_and_source_retries(monkeypatch, retry):
    from utils.experiments.theory import numerical_refinement as engine
    law = support([[0.], [1.]])
    verified = raw(endpoint=.6875, conditional_epsilon=-.25)
    if retry == "source_perturbation":
        verified["source_endpoint_radii"] = [0., 0.]
        verified["source_sensitivity_justification"] = {
            "status": "justified_bound", "scope": "endpoint_locations_only_fixed_law_and_coefficients",
            "derivation": "Exact zero endpoint radii supplied by the synthetic fixture"}
    else:
        monkeypatch.setattr(engine, "_gain_retry_needed", lambda *args, **kwargs: True)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    original_segment, original_gain = engine._raw_segment, engine.interval_gain
    segment_calls, gain_calls = [], []
    def segment(*args, **kwargs):
        assert kwargs.get("current_reference") is not None
        segment_calls.append(1)
        return original_segment(*args, **kwargs)
    def gain(*args, **kwargs):
        gain_calls.append(1)
        return original_gain(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Negative condition unnecessarily integrated variation")
    monkeypatch.setattr(engine, "_raw_segment", segment)
    monkeypatch.setattr(engine, "interval_gain", gain)
    monkeypatch.setattr(engine, "enclose_variation", forbidden)
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified, policy=NumericalPolicy(max_decimal_products=2_000_000))[0]
    assert result["numerical_condition_screen_status"] == "certified_negative"
    assert result["condition_sign_status"] == "negative"
    assert len(segment_calls) == 1 and len(gain_calls) == 2
    assert result["numerical_endpoint_refinement_executed"]
    assert result["numerical_current_posterior_reused"]
    assert result["numerical_gpu_products"] > result["numerical_condition_screen_gpu_products"] > 0
    assert retry in result["numerical_gpu_refinement_reason"]
    assert result["numerical_gpu_refinement_executed"]
    assert result["numerical_variation_nodes"] == 0
    assert result["fixed_cache_gain_sign"] == "negative"
    if retry == "source_perturbation":
        assert result["source_sensitivity_status"] == "justified_endpoint_radius_enclosure"
        assert result["source_robust_gain_sign"] == "negative"


@requires_cuda
def test_partial_condition_screen_budget_stops_without_restarting_posterior(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law = support([[0.], [1.]])
    verified = raw(endpoint=.6875, conditional_epsilon=-.25)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    # Enter the screen, but cannot finish its bounded transcendental arithmetic.
    budget = 38 * law.dimension + engine._posterior_work_floor(
        law.size, law.dimension, original_inputs=True) + 1
    def forbidden(*args, **kwargs):
        raise AssertionError("An incomplete screen restarted endpoint/posterior work")
    for name in ("_raw_row", "_raw_segment", "interval_gain", "enclose_variation"):
        monkeypatch.setattr(engine, name, forbidden)
    result = refine_payload(law, payload, scalar_rows(), verified_inputs=verified,
                            policy=NumericalPolicy(max_decimal_products=budget))[0]
    assert result["numerical_condition_screen_status"] == "budget_exhausted"
    assert result["numerical_stopping_reason"] == "gpu_operation_budget_exhausted_in_condition_screen"
    assert result["condition_sign_status"] == "unresolved"
    assert not result["numerical_condition_nonnegative_certified"]
    assert result["numerical_gpu_refinement_selected"]
    assert result["numerical_gpu_refinement_executed"] == (result["numerical_gpu_products"] > 0)
    assert not result["numerical_endpoint_refinement_executed"]
    assert not result["numerical_current_posterior_reused"]
    assert 0 <= result["numerical_gpu_products"] <= budget
    assert result["numerical_gpu_products"] == result["numerical_condition_screen_gpu_products"]
    assert math.isnan(result["numerical_variation_l2"])


def _saved_condition_rows(payload, *, variation=.1, embedded_error=1e-8, projection_roundoff=1e-9):
    margin = (payload["delta_norm"] - payload["branch_gap_error"] - variation).item()
    return scalar_rows(
        direct_prop5_margin_l2=margin, direct_prop5_variation_l2=variation,
        direct_prop5_variation_error_l2=embedded_error,
        direct_prop5_variation_signed_integral_l2=variation,
        direct_prop5_variation_signed_integral_error_l2=embedded_error,
        direct_prop5_variation_direction_status="finite_nonzero_gap_projection",
        direct_prop5_projection_roundoff_l2=projection_roundoff,
        direct_prop5_gram_roundoff_l2=1e9,
        direct_prop5_status="matched_finite_law_comparison",
        direct_prop5_integral_status="estimated_converged",
        direct_prop5_integral_reason="estimated_converged",
        direct_prop5_source_sensitivity_l2=1e9, direct_prop5_uncertainty_l2=1e9,
    )


def _forbid_intervals(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine

    def forbidden(*args, **kwargs):
        raise AssertionError("Default numerical assessment entered interval certification")

    for name in ("screen_branch_gap_condition", "_raw_row", "_raw_segment", "_clean_intervals",
                 "interval_gain", "enclose_variation", "GpuDirected"):
        monkeypatch.setattr(engine, name, forbidden)


@pytest.mark.parametrize("epsilon,variation,embedded,expected", [
    (-.25, .1, 1e-8, "negative"),
    (1., .1, 1e-8, "positive"),
    (1., .5 - 1e-10, 1e-8, "unresolved"),
    (1., .5, 0., "unresolved"),
])
@requires_cuda
def test_default_saved_margin_assessment_retains_values_without_any_interval_work(
        monkeypatch, epsilon, variation, embedded, expected):
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=epsilon)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    rows = _saved_condition_rows(payload, variation=variation, embedded_error=embedded, projection_roundoff=0.)
    expected_gain = stable_gain(payload["intercept"], payload["saved_endpoint_slopes"], 0)
    _forbid_intervals(monkeypatch)
    result = refine_payload(law, payload, rows, verified_inputs=verified)[0]
    assert result["condition_sign_status"] == result["condition_estimate_sign_status"] == expected
    assert result["condition_arithmetic_status"] == "float64_numerical_assessment"
    assert result["arithmetic_status"] == "float64_numerical_assessment"
    assert result["numerical_backend"] == "cuda_float64_assessment"
    assert result["numerical_original_margin_l2"] == rows[0]["direct_prop5_margin_l2"]
    assert result["numerical_variation_l2"] == variation
    assert result["numerical_variation_definition"] == "positive_part_after_integrated_unit_gap_projection"
    assert result["numerical_signed_projected_integral_l2"] == variation
    assert result["numerical_signed_projected_integral_error_l2"] == embedded
    assert result["numerical_condition_estimate_projection_roundoff_l2"] == 0.
    assert result["numerical_condition_estimate_gram_roundoff_l2"] == 1e9
    assert result["numerical_log_probability_gain"] == expected_gain["H"].item()
    assert result["numerical_log_odds_gain"] == expected_gain["G"].item()
    assert result["numerical_condition_estimate_uncertainty_l2"] >= embedded
    assert result["numerical_condition_estimate_uncertainty_l2"] < 1e-6
    assert result["numerical_source_sensitivity_l2"] == 1e9
    assert result["numerical_gpu_products"] == result["numerical_variation_nodes"] == 0
    assert not result["numerical_gpu_refinement_selected"]
    assert not result["numerical_gpu_refinement_executed"]
    assert not result["numerical_endpoint_refinement_executed"]
    assert not result["implication_eligible"] and not result["numerical_condition_nonnegative_certified"]
    assert result["numerical_certification_scope"] == "none_float64_assessment"
    assert result["numerical_stopping_reason"] == "gpu_refinement_disabled_by_zero_budget"
    assert all(math.isnan(result[name]) for name in (
        "numerical_margin_lower_l2", "numerical_margin_upper_l2",
        "numerical_variation_lower_l2", "numerical_variation_upper_l2"))
    assert "numerical_gain_H_lower" not in result


@pytest.mark.parametrize("missing", ["margin_l2", "variation_l2", "variation_error_l2", "projection_roundoff_l2"])
@requires_cuda
def test_default_assessment_preserves_missing_measurements_as_unresolved(monkeypatch, missing):
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=1.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    rows = _saved_condition_rows(payload)
    rows[0]["direct_prop5_" + missing] = None
    _forbid_intervals(monkeypatch)
    result = refine_payload(law, payload, rows, verified_inputs=verified)[0]
    assert result["condition_sign_status"] == result["condition_estimate_sign_status"] == "unresolved"
    assert result["numerical_condition_estimate_status"] == "missing_or_invalid_saved_measurement"
    assert math.isnan(result["numerical_condition_estimate_uncertainty_l2"])
    assert not result["implication_eligible"]
    if missing == "margin_l2":
        assert math.isnan(result["numerical_original_margin_l2"])
    if missing == "variation_l2":
        assert math.isnan(result["numerical_variation_l2"])


@pytest.mark.parametrize("changes", [
    {"direct_prop5_status": "failed_integration"},
    {"direct_prop5_integral_status": "failed"},
    {"direct_prop5_integral_reason": "invalid_norm_or_nonfinite_integrand"},
    {"direct_prop5_integral_reason": "matched_displacement_QA_failure"},
])
@requires_cuda
def test_default_assessment_does_not_hide_failed_integration(monkeypatch, changes):
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=1.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    rows = _saved_condition_rows(payload)
    rows[0].update(changes)
    _forbid_intervals(monkeypatch)
    result = refine_payload(law, payload, rows, verified_inputs=verified)[0]
    assert result["condition_sign_status"] == "unresolved"
    assert result["numerical_condition_estimate_status"] == "failed_existing_integration"
    assert result["numerical_original_margin_l2"] == rows[0]["direct_prop5_margin_l2"]
    assert result["numerical_variation_l2"] == rows[0]["direct_prop5_variation_l2"]
    assert math.isnan(result["numerical_condition_estimate_uncertainty_l2"])
    assert not result["implication_eligible"] and not result["numerical_gpu_refinement_executed"]


@pytest.mark.parametrize("field,value", [
    ("variation_l2", -1.), ("variation_error_l2", -1.),
    ("projection_roundoff_l2", -1.), ("variation_error_l2", math.inf),
])
@requires_cuda
def test_default_assessment_never_treats_invalid_error_estimates_as_zero(monkeypatch, field, value):
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=1.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    rows = _saved_condition_rows(payload)
    rows[0]["direct_prop5_" + field] = value
    _forbid_intervals(monkeypatch)
    result = refine_payload(law, payload, rows, verified_inputs=verified)[0]
    assert result["condition_sign_status"] == "unresolved"
    assert result["numerical_condition_estimate_status"] == "missing_or_invalid_saved_measurement"
    assert math.isnan(result["numerical_condition_estimate_uncertainty_l2"])


@requires_cuda
def test_singleton_margin_zero_is_an_identity_not_a_rounded_zero_shortcut(monkeypatch):
    law, verified = support([[0.]]), raw(conditional_epsilon=1.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    rows = _saved_condition_rows(payload, variation=0., embedded_error=0., projection_roundoff=0.)
    _forbid_intervals(monkeypatch)
    result = refine_payload(law, payload, rows, verified_inputs=verified)[0]
    assert result["numerical_original_margin_l2"] == 0.
    assert result["condition_sign_status"] == result["condition_estimate_sign_status"] == "zero"
    assert result["numerical_condition_estimate_status"] == "analytic_single_atom_or_zero_gap_margin_identity"
    assert result["condition_arithmetic_status"] == "float64_numerical_assessment"
    assert not result["numerical_condition_nonnegative_certified"] and not result["implication_eligible"]
    assert math.isnan(result["numerical_margin_lower_l2"])


@requires_cuda
def test_explicit_certification_keeps_assessed_sign_separate_from_unresolved_certificate(monkeypatch):
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=1.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    rows = _saved_condition_rows(payload)
    _forbid_intervals(monkeypatch)
    result = refine_payload(law, payload, rows, verified_inputs=verified,
                            policy=NumericalPolicy(max_decimal_products=1))[0]
    assert result["condition_estimate_sign_status"] == "positive"
    assert result["condition_sign_status"] == "unresolved"
    assert result["condition_arithmetic_status"] == "not_enclosed"
    assert result["numerical_gpu_refinement_selected"]
    assert not result["numerical_gpu_refinement_executed"]
    assert result["numerical_stopping_reason"] == "gpu_operation_budget_exhausted_before_raw_enclosure"
    assert not result["implication_eligible"]


@pytest.mark.parametrize("centre", [Decimal('.6'), Decimal('.4')])
def test_decimal_directional_oracle_clips_after_the_signed_integral(centre):
    result = enclose_variation(Directed(80), [points([0]), points([1])],
        points([0, -1]), points([0, 2]), points([centre]),
        Interval.point(1), Interval.point(1), delta=points([1]),
        max_nodes=65, absolute_width=1e-20)
    expected = Decimal('.5') - centre
    signed = result["signed_projected_integral"]
    assert signed.lower <= expected <= signed.upper
    if expected < 0:
        assert signed.upper < 0
        assert result["V"].lower == result["V"].upper == 0
    else:
        assert 0 < result["V"].lower <= expected <= result["V"].upper


def test_decimal_directional_oracle_distinguishes_zero_gap_from_uncertain_norm():
    kwargs = dict(atoms=[points([0]), points([1])], intercept=points([0, 0]),
                  slopes=points([0, 0]), current_mean=points([Decimal('.25')]),
                  branch_gap_error=Interval.point(0), max_nodes=1)
    result = enclose_variation(Directed(80), D=Interval.point(0), delta=points([0]), **kwargs)
    assert result["V"].lower == result["V"].upper == 0
    assert result["direction_status"] == "exact_zero_gap_no_unit_direction"
    with pytest.raises(ArithmeticError, match="Unit branch-gap direction unresolved"):
        enclose_variation(Directed(80), D=Interval(Decimal(0), Decimal(1)),
                          delta=[Interval(Decimal(0), Decimal(1))], **kwargs)


@requires_cuda
def test_directional_payload_saves_actual_delta_and_rejects_old_norm_only_contract():
    law, verified = support([[0.], [1.]]), raw()
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    assert payload["schema_version"] == 5
    assert payload["condition_contract"] == "projected-gap-error-1"
    expected = verified["sigma"] * (verified["epsilon_u"] - verified["epsilon_c"]) / verified["alpha"]
    torch.testing.assert_close(payload["delta"], expected)
    old = {**payload, "schema_version": 4, "condition_contract": "directional-positive-variation-1"}
    old.pop("delta")
    with pytest.raises(ValueError, match="sufficient-input schema"):
        refine_payload(law, old, scalar_rows(), verified_inputs=verified)


@requires_cuda
def test_reduced_direction_norm_is_enclosed_from_the_same_vector(monkeypatch):
    from utils.experiments.theory import numerical_refinement as engine
    law, verified = support([[0., 0.], [1., 1.]]), raw(conditional_epsilon=1.)
    for key in ("state", "epsilon_u", "epsilon_c", "saved_endpoint"):
        verified[key] = verified[key].repeat(1, 2)
    verified["target"] = torch.zeros(2, dtype=torch.float64, device="cuda")
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    payload["delta"] = torch.tensor([[-1., -1.]], dtype=torch.float64, device="cuda")
    payload["delta_norm"] = torch.tensor([1.], dtype=torch.float64, device="cuda")  # Deliberately rounded/invalid scalar.
    payload["branch_gap_error"] = torch.tensor([0.], dtype=torch.float64, device="cuda")
    captured = []
    def observe(c, atoms, intercept, slopes, current_mean, D, E, *, delta, **kwargs):
        captured.append((D, delta))
        assert D.lower < math.sqrt(2.) < D.upper
        assert D.lower > 1.
        raise ArithmeticError("Unit branch-gap direction unresolved: test stops before quadrature")
    monkeypatch.setattr(engine, "enclose_variation", observe)
    result = refine_payload(law, payload, scalar_rows(), policy=NumericalPolicy(max_decimal_products=2_000_000))[0]
    assert len(captured) == 1
    assert result["condition_sign_status"] == "unresolved"
    assert result["numerical_variation_direction_status"] == "unresolved_norm_contains_zero_or_nonfinite"
    assert result["numerical_stopping_reason"] == "branch_gap_direction_unresolved"
    assert math.isnan(result["numerical_variation_lower_l2"])
    assert not result["implication_eligible"]


@pytest.mark.parametrize("delta,reference,expected", [
    ([3, 4], [6, 8], -5), ([3, 4], [0, 0], 5), ([0, 0], [6, 8], 0),
])
def test_decimal_projected_error_keeps_signed_value_and_zero_convention(delta, reference, expected):
    from utils.experiments.theory.numerical_intervals import projected_branch_gap_error
    value = projected_branch_gap_error(Directed(80), points(delta), points(reference))
    assert value.lower <= expected <= value.upper
    if expected < 0:
        assert value.upper < 0
    if delta == [0, 0]:
        assert value.lower == value.upper == 0


@requires_cuda
def test_default_signed_negative_error_keeps_positive_margin_without_interval_work(monkeypatch):
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=.25)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    assert payload["branch_gap_error"].item() == pytest.approx(-.125)
    assert payload["branch_gap_error_norm_diagnostic"].item() == pytest.approx(.125)
    rows = _saved_condition_rows(payload, variation=.1, embedded_error=0., projection_roundoff=0.)
    _forbid_intervals(monkeypatch)
    result = refine_payload(law, payload, rows, verified_inputs=verified)[0]
    assert result["condition_sign_status"] == "positive"
    assert result["numerical_original_margin_l2"] == pytest.approx(.4)
    assert result["numerical_condition_estimate_subtraction_roundoff_l2"] == pytest.approx(
        128 * torch.finfo(torch.float64).eps * (.375 + .125 + .1))
    assert result["numerical_gpu_products"] == 0
    assert not result["implication_eligible"]


@requires_cuda
def test_default_exact_zero_gap_uses_zero_projected_error_and_variation(monkeypatch):
    law, verified = support([[0.], [1.]]), raw(conditional_epsilon=0.)
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    assert payload["delta_exact_zero"].item()
    assert payload["branch_gap_error"].item() == 0
    assert payload["branch_gap_error_norm_diagnostic"].item() > 0
    rows = _saved_condition_rows(payload, variation=0., embedded_error=0., projection_roundoff=0.)
    _forbid_intervals(monkeypatch)
    result = refine_payload(law, payload, rows, verified_inputs=verified)[0]
    assert result["condition_sign_status"] == "zero"
    assert result["numerical_original_margin_l2"] == result["numerical_variation_l2"] == 0
    assert not result["numerical_condition_nonnegative_certified"]


@requires_cuda
def test_nonzero_vector_cannot_use_a_false_zero_direction_witness():
    law, verified = support([[0.], [1.]]), raw()
    payload = build_refinement_payload(law, verified, bank_hash="law", target_atom=0)
    payload["delta_exact_zero"].fill_(True)
    with pytest.raises(ValueError, match="zero-direction witness"):
        refine_payload(law, payload, scalar_rows(), verified_inputs=verified)
