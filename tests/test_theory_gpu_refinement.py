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
                               c.interval(.1), c.interval(0.), c.interval(0.), max_nodes=1)
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
