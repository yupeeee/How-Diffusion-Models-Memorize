"""Unexecuted CUDA interval regression specifications; no experiment inputs.

Decimal/Fraction are independent test oracles only. Production refinement must
never execute them or transfer raw numerical arrays back to CPU.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math

import pytest
import torch

from utils.experiments.theory.gpu_intervals import (
    ArithmeticBudgetExceeded, GpuDirected, GpuInterval,
    _LN2_LOWER, _LN2_UPPER,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA enclosure backend requires NVIDIA GPU")


def _encloses(interval, exact):
    lo, hi = interval.floats()
    assert Decimal.from_float(lo) <= exact <= Decimal.from_float(hi)


def test_ln2_constants_have_exact_rational_series_proof():
    # With y=1/3 and N=64, the geometric tail is a strict upper bound on
    # 2*sum_{k>=N} y**(2*k+1)/(2*k+1). This proof has no transcendental oracle.
    y, count = Fraction(1, 3), 64
    partial = 2 * sum(y ** (2*k + 1) / (2*k + 1) for k in range(count))
    tail = 2 * y ** (2*count + 1) / ((2*count + 1) * (1-y*y))
    assert Fraction.from_float(_LN2_LOWER) < partial
    assert partial + tail < Fraction.from_float(_LN2_UPPER)


def test_cpu_backend_is_rejected_instead_of_falling_back():
    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        GpuDirected(device="cpu")
    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        GpuInterval.point(1., device="cpu")


@requires_cuda
def test_basic_operations_enclose_cancellation_underflow_and_overflow():
    c = GpuDirected(max_products=100_000, device="cuda:0")
    third = c.div(c.point(1.), c.point(3.))
    with localcontext() as context:
        context.prec = 100
        _encloses(third, Decimal(1) / Decimal(3))
    _encloses(c.mul(third, c.point(3.)), Decimal(1))
    smallest = float.fromhex("0x0.0000000000001p-1022")
    _encloses(c.mul(c.point(smallest), c.point(.5)), Decimal.from_float(smallest) / 2)
    biggest = float.fromhex("0x1.fffffffffffffp+1023")
    overflow = c.mul(c.point(biggest), c.point(2.))
    assert overflow.lower.item() <= biggest and overflow.upper.item() == math.inf
    assert c.sub(c.point(3.), c.point(3.)).sign == "zero"
    _encloses(c.sqrt(c.point(2.)), Decimal(2).sqrt())


@requires_cuda
def test_pairwise_reduction_encloses_exact_cancellation_and_counts_all_lanes():
    c = GpuDirected(max_products=100, device="cuda:0")
    values = c.point([1e16, 1., -1e16, 3., 1e-200])
    before = c.products
    result = c.total(values)
    assert c.products - before == 4
    with localcontext() as context:
        context.prec = 300
        exact = sum(map(Decimal.from_float, [1e16, 1., -1e16, 3., 1e-200]))
        _encloses(result, exact)
    assert result.lower.device.type == result.upper.device.type == "cuda"


@requires_cuda
def test_broadcast_product_budget_cannot_count_only_the_input_shape():
    c = GpuDirected(max_products=1000, device="cuda:0")
    left, right = c.point([[1.], [2.]]), c.point([[3., 4., 5.]])
    before = c.products
    result = c.mul(left, right)
    assert result.lower.shape == (2, 3)
    assert c.products - before == 4 * 6
    c.max_products = c.products + 4 * 6 - 1
    with pytest.raises(ArithmeticBudgetExceeded, match="GPU interval"):
        c.mul(left, right)


@requires_cuda
def test_bounded_input_budget_counts_both_broadcast_endpoints():
    c = GpuDirected(max_products=100, device="cuda:0")
    before = c.products
    result = c.interval(0., [1., 2., 3.])
    assert result.lower.shape == result.upper.shape == (3,)
    assert c.products - before == 2 * 3
    c.max_products = c.products + 5
    with pytest.raises(ArithmeticBudgetExceeded, match="GPU interval"):
        c.interval([0., 1., 2.], 3.)


@requires_cuda
@pytest.mark.parametrize("value", [-1100., -745., -100., -1e-12, 0., .125, .5, 1., 700., 709.7, 1024., 1100.])
def test_exp_series_encloses_independent_high_precision_oracle(value):
    c = GpuDirected(max_products=100_000, device="cuda:0")
    interval = c.exp(c.point(value))
    with localcontext() as context:
        context.prec = 150
        _encloses(interval, Decimal.from_float(value).exp())
    assert interval.lower.item() >= 0
    if value == 0:
        assert interval.floats() == (1., 1.)


@requires_cuda
@pytest.mark.parametrize("value", [float.fromhex("0x0.0000000000001p-1022"), 1e-300, .5, .999999999999, 1., 1.000000000001, 2., 1e300, float.fromhex("0x1.fffffffffffffp+1023")])
def test_log_series_encloses_independent_high_precision_oracle(value):
    c = GpuDirected(max_products=100_000, device="cuda:0")
    interval = c.log(c.point(value))
    with localcontext() as context:
        context.prec = 150
        _encloses(interval, Decimal.from_float(value).ln())
    if value == 1:
        assert interval.floats() == (0., 0.)


@requires_cuda
def test_probability_and_norm_bounds_remain_on_device_and_enclose_oracles():
    c = GpuDirected(max_products=1_000_000, device="cuda:0")
    logits = c.point([[0., -1., 2.], [-1000., 0., 1000.]])
    weights = c.softmax(logits, dim=1)
    totals = c.total(weights, dim=1)
    assert (totals.lower <= 1).all() and (totals.upper >= 1).all()
    assert (weights.lower >= 0).all() and (weights.upper <= 1).all()
    with localcontext() as context:
        context.prec = 150
        row = [Decimal(0), Decimal(-1), Decimal(2)]
        normalizer = sum(value.exp() for value in row)
        for index, value in enumerate(row):
            _encloses(weights[0, index], value.exp() / normalizer)
        _encloses(c.logsumexp(logits, dim=1)[0], normalizer.ln())
    norm = c.norm(c.point([[3., 4.], [0., 0.]]), dim=1)
    _encloses(norm[0], Decimal(5))
    assert norm[1].sign == "zero"


@requires_cuda
def test_invalid_domains_and_nonfinite_inputs_never_give_false_signs():
    c = GpuDirected(max_products=100_000, device="cuda:0")
    assert c.point(float("nan")).sign == "unresolved"
    assert c.point(float("inf")).sign == "unresolved"
    assert c.div(c.point(1.), c.interval(-1., 1.)).sign == "unresolved"
    assert c.log(c.point(-1.)).sign == "unresolved"
    assert c.sqrt(c.point(-1.)).sign == "unresolved"


@requires_cuda
def test_softplus_matches_monotone_endpoint_oracles():
    c = GpuDirected(max_products=1_000_000, device="cuda:0")
    for value in [-1000., -30., -1., 0., 1., 30., 1000.]:
        with localcontext() as context:
            context.prec = 150
            exact = (1 + Decimal.from_float(value).exp()).ln()
            _encloses(c.softplus(c.point(value)), exact)
