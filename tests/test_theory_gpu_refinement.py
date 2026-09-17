"""CUDA-only enclosure regression specifications; never project data or models."""
from decimal import Decimal

import pytest
import torch

from utils.experiments.theory.gpu_intervals import GpuDirected
from utils.experiments.theory.gpu_refinement import enclose_variation, interval_gain, raw_row
from utils.experiments.theory.numerical_intervals import Directed, Interval, interval_gain as reference_gain

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical backend required")


@pytest.mark.parametrize("intercept,slopes", [
    ([0., 0., -2.], [0., 1., -.5]),
    ([0., -1000.], [0., -1.]),
    ([0., 0., 0.], [0., 1e-8, -1e-8]),
])
def test_gpu_gain_encloses_independent_high_precision_reference(intercept, slopes):
    context = GpuDirected(20_000_000, device="cuda")
    actual = interval_gain(context, context.interval(intercept), context.interval(slopes), 0)
    reference = reference_gain(Directed(128), [Interval.point(x) for x in intercept],
                               [Interval.point(x) for x in slopes], 0)
    for name in ("H", "G", "slope0", "slope1"):
        low, high = actual[name].floats()
        assert Decimal.from_float(low) <= reference[name].lower
        assert Decimal.from_float(high) >= reference[name].upper
    if slopes == [0., 1e-8, -1e-8]:
        # More operation budget cannot produce extra binary64 precision.
        assert actual["gain_sign"] == "unresolved"


def test_gpu_variation_retains_sharp_transition_uncertainty():
    c = GpuDirected(20_000_000, device="cuda")
    result = enclose_variation(c, c.interval([[-1.], [1.]]), c.interval([0., -180.]),
                               c.interval([0., 200.]), c.interval([-1.]),
                               c.interval(.1), c.interval(0.), delta=c.interval([.1]), max_nodes=1)
    low, high = result["V"].floats()
    assert low <= .2 <= high
    assert result["M"].sign == "unresolved"
    assert result["stopping_reason"] == "node_budget_exhausted"
    assert result["nodes"] == c.evaluated_nodes == 1
    assert result["V"].lower.device.type == "cuda"


def test_original_row_keeps_saved_vector_inputs_on_cuda():
    values = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, device="cuda")
    raw = {name: values for name in ("state", "epsilon_u", "epsilon_c", "saved_endpoint")}
    raw["target"] = values[0]
    selected = raw_row(raw, 1, device="cuda")
    assert all(selected[name].device.type == "cuda" for name in raw)
    assert torch.equal(selected["state"], values[1])
    assert selected["state"].data_ptr() == values[1].data_ptr()


def test_gpu_branch_gap_error_precheck_and_margin_ignore_separate_branch_errors():
    from utils.experiments.theory.gpu_refinement import negative_condition_precheck

    c = GpuDirected(20_000_000, device="cuda")
    negative = negative_condition_precheck(c.interval(.5), c.interval(1.), arithmetic=c)
    assert negative["condition_sign_status"] == "negative"
    assert negative["upper"] < 0
    # A supplied zero joint vector error and constant posterior give M=D,
    # independently of the magnitudes of the two separate branch errors.
    result = enclose_variation(c, c.interval([[0.]]), c.interval([0.]),
                               c.interval([0.]), c.interval([0.]),
                               c.interval(1.), c.interval(0.), delta=c.interval([1.]), max_nodes=1)
    assert result["V"].lower <= 0 <= result["V"].upper
    assert result["M"].lower <= 1 <= result["M"].upper
    assert result["M"].sign == "positive"


def _raw_segment_fixture(context):
    atoms = context.interval([[0., 0.], [1., -.5], [-.25, 1.]])
    masses = context.interval([.2, .3, .5])
    def vector(value):
        return torch.tensor(value, dtype=torch.float64, device=context.device)
    raw = {"state": vector([.2, -.1]), "epsilon_u": vector([.3, .5]),
           "epsilon_c": vector([.1, -.2]), "target": vector([0., 0.]),
           "saved_endpoint": vector([.1, .2]), "alpha": .8, "sigma": .6,
           "destination_alpha": .9, "destination_sigma": .4,
           "A": .2, "kappa": .3, "guidance": 2., "noise_std": 0.}
    return raw, atoms, masses


def test_raw_segment_reuses_screened_reference_without_posterior_or_individual_errors(monkeypatch):
    from utils.experiments.theory import gpu_refinement as gpu

    original_context = GpuDirected(20_000_000, device="cuda")
    raw, atoms, masses = _raw_segment_fixture(original_context)
    clean = gpu.clean_intervals(original_context, raw)
    baseline = gpu.raw_segment(original_context, raw, atoms, masses, 0, clean, chunk=2)
    receipt = {name: baseline[name] for name in ("current_mean", "D", "branch_gap_error")}
    receipt["delta"] = clean[3]
    context = GpuDirected(20_000_000, device="cuda")
    def forbidden(*args, **kwargs):
        raise AssertionError("The current posterior must not be evaluated a second time")
    monkeypatch.setattr(gpu, "posterior_mean", forbidden)
    # Reusing delta does not need the conditional branch or an individual error.
    optimized_raw = {name: value for name, value in raw.items() if name not in {"epsilon_c", "target"}}
    norm_calls, original_norm = [], context.norm
    def norm(value, *args, **kwargs):
        norm_calls.append(value.lower.shape)
        return original_norm(value, *args, **kwargs)
    monkeypatch.setattr(context, "norm", norm)
    reused_clean = gpu.clean_intervals(context, optimized_raw, current_reference=receipt)
    assert reused_clean[2] is None and reused_clean[5] is None
    assert norm_calls == []
    result = gpu.raw_segment(context, optimized_raw, atoms, masses, 0, reused_clean,
                             chunk=2, current_reference=receipt)
    assert len(norm_calls) == 1  # the endpoint-transfer residual, not ec/eu/D/E
    assert result["ec"] is None and result["eu"] is None
    for name in ("current_mean", "D", "branch_gap_error"):
        assert result[name] is receipt[name]
    for name in ("b", "slopes", "saved_slopes", "transfer", "spatial_factor", "prefactor"):
        torch.testing.assert_close(result[name].lower, baseline[name].lower, rtol=0, atol=0)
        torch.testing.assert_close(result[name].upper, baseline[name].upper, rtol=0, atol=0)
    assert context.products < original_context.products


def test_raw_segment_builds_mass_priors_once_per_chunk_for_both_noise_levels(monkeypatch):
    from utils.experiments.theory import gpu_refinement as gpu

    context = GpuDirected(20_000_000, device="cuda")
    raw, atoms, masses = _raw_segment_fixture(context)
    clean = gpu.clean_intervals(context, raw)
    calls, original_log = [], context.log
    def log(value):
        calls.append(value.lower.shape)
        return original_log(value)
    monkeypatch.setattr(context, "log", log)
    result = gpu.raw_segment(context, raw, atoms, masses, 0, clean, chunk=2)
    assert calls == [torch.Size([2]), torch.Size([1])]
    assert result["current_mean"].lower.shape == (2,)
    assert result["ec"] is clean[5] and result["eu"] is not None


@pytest.mark.parametrize("invalid", ["point_tensor", "batch_mean", "batch_norm", "missing_error"])
def test_precomputed_reference_rejects_unenclosed_values_and_nonrow_shapes(invalid):
    from utils.experiments.theory import gpu_refinement as gpu

    context = GpuDirected(20_000_000, device="cuda")
    raw, atoms, masses = _raw_segment_fixture(context)
    receipt = {"current_mean": context.interval([0., 0.]),
               "delta": context.interval([.1, .2]),
               "D": context.interval(.3), "branch_gap_error": context.interval(.4)}
    if invalid == "point_tensor":
        receipt["current_mean"] = receipt["current_mean"].lower
    elif invalid == "batch_mean":
        receipt["current_mean"] = context.interval([[0., 0.]])
    elif invalid == "batch_norm":
        receipt["D"] = context.interval([.3])
    else:
        receipt.pop("branch_gap_error")
    with pytest.raises(ValueError, match="Precomputed current reference"):
        gpu.clean_intervals(context, raw, current_reference=receipt)


def test_precomputed_reference_can_reuse_norm_with_delta_reconstructed_from_raw():
    from utils.experiments.theory import gpu_refinement as gpu

    context = GpuDirected(20_000_000, device="cuda")
    raw, atoms, masses = _raw_segment_fixture(context)
    clean = gpu.clean_intervals(context, raw)
    baseline = gpu.raw_segment(context, raw, atoms, masses, 0, clean, chunk=2)
    receipt = {name: baseline[name] for name in ("current_mean", "D", "branch_gap_error")}
    reused = gpu.clean_intervals(context, raw, current_reference=receipt)
    assert reused[4] is receipt["D"] and reused[5] is None
    torch.testing.assert_close(reused[3].lower, clean[3].lower, rtol=0, atol=0)
    torch.testing.assert_close(reused[3].upper, clean[3].upper, rtol=0, atol=0)


@pytest.mark.parametrize("centre,sign", [(.6, "negative"), (.4, "positive")])
def test_directional_integral_retains_cancellation_before_positive_part(centre, sign):
    # mean(s)=sigmoid(2*s-1) integrates to exactly 1/2 by symmetry.
    # At centre=.6 the integrand changes sign, but its integral is -.1;
    # clipping each node before integration would incorrectly produce V>0.
    c = GpuDirected(20_000_000, device="cuda")
    result = enclose_variation(c, c.interval([[0.], [1.]]), c.interval([0., -1.]),
                               c.interval([0., 2.]), c.interval([centre]),
                               c.interval(1.), c.interval(1.), delta=c.interval([1.]),
                               max_nodes=65, absolute_width=1e-12)
    signed_lo, signed_hi = result["signed_projected_integral"].floats()
    assert signed_lo <= .5 - centre <= signed_hi
    assert result["signed_projected_integral"].sign == sign
    assert result["direction_status"] == "unit_direction_enclosed_from_actual_delta"
    low, high = result["V"].floats()
    if sign == "negative":
        assert low == high == 0.
        assert result["nodes"] >= 3
    else:
        assert 0 < low <= .5 - centre <= high
    assert "positive_part_after_integral" in result["error_method"]


def test_directional_variation_excludes_orthogonal_reference_motion():
    c = GpuDirected(20_000_000, device="cuda")
    # The next posterior mean is (0,1) and the current reference is (0,0).
    # Its displacement has norm1 but zero projection on the actual x gap.
    result = enclose_variation(c, c.interval([[0., 0.], [0., 2.]]), c.interval([0., 0.]),
                               c.interval([0., 0.]), c.interval([0., 0.]),
                               c.interval(1.), c.interval(1.), delta=c.interval([1., 0.]), max_nodes=1)
    assert result["signed_projected_integral"].floats() == (0., 0.)
    assert result["V"].floats() == (0., 0.)
    assert result["M"].sign == "zero"


def test_directional_variation_uses_current_noise_reference_baseline():
    c = GpuDirected(20_000_000, device="cuda")
    # The next-level path mean is constantly .5, different from current .25.
    # Subtracting the next-level baseline instead would incorrectly give V=0.
    result = enclose_variation(c, c.interval([[0.], [1.]]), c.interval([0., 0.]),
                               c.interval([0., 0.]), c.interval([.25]),
                               c.interval(1.), c.interval(0.), delta=c.interval([1.]), max_nodes=1)
    low, high = result["V"].floats()
    assert 0 < low <= .25 <= high
    assert result["signed_projected_integral"].lower > 0


def test_exact_zero_gap_has_zero_variation_without_constructing_unit_direction(monkeypatch):
    c = GpuDirected(20_000_000, device="cuda")
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero-gap convention attempted a division")
    monkeypatch.setattr(c, "div", forbidden)
    result = enclose_variation(c, c.interval([[0.], [1.]]), c.interval([0., 0.]),
                               c.interval([0., 0.]), c.interval([.25]),
                               c.interval(0.), c.interval(0.), delta=c.interval([0.]), max_nodes=1)
    assert result["V"].floats() == (0., 0.)
    assert result["signed_projected_integral"].floats() == (0., 0.)
    assert result["integrated_H"].floats() == (0., 0.)
    assert result["nodes"] == 0
    assert result["direction_status"] == "exact_zero_gap_no_unit_direction"
    assert result["stopping_reason"] == "exact_zero_branch_gap_convention"
    assert result["M"].sign == "zero"


def test_uncertain_gap_norm_touching_zero_never_creates_a_unit_direction():
    c = GpuDirected(20_000_000, device="cuda")
    with pytest.raises(ArithmeticError, match="Unit branch-gap direction unresolved"):
        enclose_variation(c, c.interval([[0.], [1.]]), c.interval([0., 0.]),
                          c.interval([0., 1.]), c.interval([.5]),
                          c.interval(0., 1.), c.interval(0.),
                          delta=c.interval([0.], [1.]), max_nodes=1)


@pytest.mark.parametrize("delta,reference,expected", [
    ([3., 4.], [6., 8.], -5.), ([3., 4.], [0., 0.], 5.),
    ([3., 4.], [3., 14.], -8.), ([0., 0.], [6., 8.], 0.),
])
def test_projected_error_interval_encloses_signed_dot_not_error_norm(delta, reference, expected):
    from utils.experiments.theory.gpu_refinement import projected_branch_gap_error
    c = GpuDirected(20_000_000, device="cuda")
    value = projected_branch_gap_error(c, c.interval(delta), c.interval(reference))
    lo, hi = value.floats()
    assert lo <= expected <= hi
    if expected < 0:
        assert hi < 0
    if delta == [0., 0.]:
        assert lo == hi == 0


def test_projected_error_uncertain_norm_cannot_return_finite_bound_or_sign():
    from utils.experiments.theory.gpu_refinement import projected_branch_gap_error
    c = GpuDirected(20_000_000, device="cuda")
    value = projected_branch_gap_error(c, c.interval([-1.], [1.]), c.interval([3.]), c.interval(0., 1.))
    assert value.floats() == (-float("inf"), float("inf"))
    assert value.sign == "unresolved"
