"""Batched posterior-only CUDA screening preserves outward sign certificates."""
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest
import torch

from utils.experiments.theory.numerical_screening import SCREENING_VERSION, screen_negative_condition, screen_branch_gap_condition

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA interval screening required")


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_retired_conditional_error_screen_rejects_before_any_tensor_computation(device):
    class NoTensorAccess:
        def __getitem__(self, key):
            raise AssertionError("Retired screening attempted to read numerical inputs")

    with pytest.raises(ValueError, match="Conditional-error screening is retired.*projected-gap-error-1"):
        screen_negative_condition(NoTensorAccess(), device=device)
    assert "projected-gap-error-1" in SCREENING_VERSION


def _support(atoms, weights=None, *, chunk=256):
    flat = torch.tensor(atoms, dtype=torch.float64, device="cuda")
    masses = torch.full((len(flat),), 1 / len(flat), dtype=torch.float64, device="cuda") if weights is None else torch.tensor(weights, dtype=torch.float64, device="cuda")
    return SimpleNamespace(flat=flat, weights=masses, size=len(flat), dimension=flat.shape[1], candidate_chunk=chunk)


def _saved(epsilon_c, *, state=None, alpha=.5, sigma=.75):
    epsilon = torch.tensor(epsilon_c, dtype=torch.float64, device="cuda")
    z = torch.full_like(epsilon, .25) if state is None else torch.tensor(state, dtype=torch.float64, device="cuda")
    return {"state": z, "epsilon_u": torch.zeros_like(epsilon), "epsilon_c": epsilon,
            "target": torch.zeros(epsilon.shape[1], dtype=torch.float64, device="cuda"),
            "saved_endpoint": torch.zeros_like(epsilon), "alpha": alpha, "sigma": sigma,
            "stored_prediction_type": "epsilon"}


def _contains(interval, expected):
    low, high = interval.lower.detach().cpu().reshape(-1).tolist(), interval.upper.detach().cpu().reshape(-1).tolist()
    for lo, value, hi in zip(low, expected, high, strict=True):
        assert Decimal.from_float(lo) <= value <= Decimal.from_float(hi)


@requires_cuda
def test_batch_screen_encloses_exact_posterior_error_and_only_certifies_negative_margin():
    support = _support([[0.], [1.]])
    saved = _saved([[-.25], [1.], [0.]])
    result = screen_branch_gap_condition(support, saved, max_products=2_000_000, target_atom=0)
    assert result["complete"] and result["status"] == "complete"
    assert result["certified_negative"].tolist() == [True, False, False]
    assert result["arithmetic_resolved"].all()
    _contains(result["current_mean"], [Decimal('.5')] * 3)
    _contains(result["D"], [Decimal('.375'), Decimal('1.5'), Decimal(0)])
    _contains(result["branch_gap_error"], [Decimal('.875'), Decimal(1), Decimal(0)])
    assert result["margin_upper"][0] < 0
    assert result["margin_upper"][2] == 0
    assert result["margin_upper"][1] > 0
    assert all(result[key].device.type == "cuda" for key in ("certified_negative", "margin_upper", "delta_lower", "error_upper"))
    assert result["prepared_inputs"]["state"].data_ptr() == saved["state"].data_ptr()
    assert not {"V", "H", "G", "endpoint", "slopes"}.intersection(result)
    assert result["total_operations"] <= len(saved["state"]) * result["operations_per_row"] <= 6_000_000


@requires_cuda
def test_joint_error_cancellation_does_not_use_individual_conditional_error():
    saved = _saved([[1.]], state=[[10.]])
    result = screen_branch_gap_condition(_support([[0.], [1.]]), saved, max_products=2_000_000, target_atom=0)
    assert result["complete"] and result["arithmetic_resolved"].item()
    assert not result["certified_negative"].item()
    assert result["delta_lower"].item() > result["error_upper"].item()
    # e_c=18.5 is much greater than D=1.5; that obsolete shortcut would
    # incorrectly classify this row as a negative condition.
    assert result["delta_upper"].item() < 18.5


@requires_cuda
def test_exact_D_equals_E_is_never_classified_strictly_negative():
    saved = _saved([[.25]], alpha=.5, sigma=.5)
    result = screen_branch_gap_condition(_support([[0.]]), saved, max_products=2_000_000, target_atom=0)
    assert result["complete"] and not result["certified_negative"].item()
    assert result["margin_upper"].item() >= 0
    _contains(result["D"], [Decimal('.25')])
    _contains(result["branch_gap_error"], [Decimal('.25')])


@requires_cuda
def test_nonuniform_odd_support_encloses_independent_decimal_vector_posterior():
    atoms = [[0., 0., 0.], [1., 2., 3.], [-1., -2., -3.]]
    support = _support(atoms, [.25, .5, .25], chunk=2)
    saved = _saved([[.25, -.5, .75], [-.25, .5, -.75]], state=[[0., 0., 0.]] * 2)
    result = screen_branch_gap_condition(support, saved, max_products=2_000_000, target_atom=0)
    assert result["complete"] and result["arithmetic_resolved"].all()
    with localcontext() as context:
        context.prec = 90
        alpha, sigma = Decimal('.5'), Decimal('.75')
        likelihood = (-(alpha * alpha * 14) / (2 * sigma * sigma)).exp()
        coefficient = Decimal('.25') * likelihood / (Decimal('.25') + Decimal('.75') * likelihood)
        reference = [coefficient * coordinate for coordinate in (1, 2, 3)]
        _contains(result["current_mean"], reference * 2)
        for row, epsilon in enumerate(([Decimal('.25'), Decimal('-.5'), Decimal('.75')],
                                        [Decimal('-.25'), Decimal('.5'), Decimal('-.75')])):
            delta = [-sigma * value / alpha for value in epsilon]
            D = sum(value * value for value in delta).sqrt()
            E = sum(value * (value + mean) for value, mean in zip(delta, reference, strict=True)) / D
            _contains(result["D"][row], [D])
            _contains(result["branch_gap_error"][row], [E])
            assert Decimal.from_float(result["margin_upper"][row].item()) >= D - E
            if result["certified_negative"][row].item():
                assert D < E


@requires_cuda
def test_exhausted_per_row_budget_never_exposes_partial_posterior_or_sign():
    support, saved = _support([[0.], [1.]]), _saved([[-.25], [1.]])
    full = screen_branch_gap_condition(support, saved, max_products=2_000_000, target_atom=0)
    budget = full["operations_per_row"] - 1
    partial = screen_branch_gap_condition(support, saved, max_products=budget, target_atom=0)
    assert not partial["complete"] and partial["status"] == "budget_exhausted"
    assert not partial["certified_negative"].any() and not partial["arithmetic_resolved"].any()
    assert partial["margin_upper"].isnan().all()
    assert all(partial[key] is None for key in ("current_mean", "delta", "D", "branch_gap_error"))
    assert 0 < partial["operations_per_row"] <= budget
    assert partial["total_operations"] <= len(saved["state"]) * budget
    assert partial["prepared_inputs"]["state"].device.type == "cuda"


@requires_cuda
def test_batch_budget_is_per_seed_and_flat_source_radii_remain_compatible():
    support = _support([[0.], [1.]])
    one = _saved([[-.25]])
    one["source_endpoint_radii"] = [0., 0.]
    first = screen_branch_gap_condition(support, one, max_products=2_000_000, target_atom=0)
    many = _saved([[-.25]] * 3)
    second = screen_branch_gap_condition(support, many, max_products=first["operations_per_row"], target_atom=0)
    assert first["complete"] and second["complete"]
    assert second["operations_per_row"] == first["operations_per_row"]
    assert second["certified_negative"].all()
    assert first["prepared_inputs"]["source_endpoint_radii"].shape == (1, 2)


@requires_cuda
def test_support_chunking_does_not_drop_odd_atom_or_seed_dimensions(monkeypatch):
    from utils.experiments.theory import numerical_screening as module

    monkeypatch.setattr(module, "_SCREEN_WORKING_LANES", 18)
    support = _support([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]], chunk=256)
    result = screen_branch_gap_condition(support, _saved([[0., 0., 0.]] * 3, state=[[0., 0., 0.]] * 3),
                                         max_products=2_000_000, target_atom=0)
    assert result["support_chunk_size"] == 2 and result["complete"]
    _contains(result["current_mean"], [Decimal(0)] * 9)
    assert not result["certified_negative"].any()


@requires_cuda
def test_nonfinite_seed_is_unresolved_without_hiding_other_finite_rows():
    saved = _saved([[-.25], [-.25]])
    saved["state"][1, 0] = torch.nan
    result = screen_branch_gap_condition(_support([[0.], [1.]]), saved, max_products=2_000_000, target_atom=0)
    assert result["complete"] and result["status"] == "unsupported_arithmetic_range"
    assert result["arithmetic_resolved"].tolist() == [True, False]
    assert result["certified_negative"].tolist() == [True, False]
    assert result["margin_upper"][1].isnan()


@requires_cuda
def test_screen_requires_exact_declared_target_atom():
    saved = _saved([[-.25]])
    saved["target"].fill_(.25)
    with pytest.raises(ValueError, match="exact atom"):
        screen_branch_gap_condition(_support([[0.], [1.]]), saved, max_products=2_000_000, target_atom=0)


def test_screen_rejects_cpu_before_saved_input_access():
    support = SimpleNamespace(flat=torch.zeros(1, 1, device="cpu"))
    with pytest.raises(ValueError, match="CUDA.*CPU fallback"):
        screen_branch_gap_condition(support, {}, max_products=2_000_000)


@requires_cuda
def test_signed_projected_screen_preserves_negative_error_without_absolute_value():
    # Symmetric current posterior=.5; Delta=-.375, barDelta=-.5.
    # u=-1 makes E=-.125, although ||Delta-barDelta||=.125.
    result = screen_branch_gap_condition(_support([[0.], [1.]]), _saved([[.25]]),
                                         max_products=2_000_000, target_atom=0)
    assert result["complete"] and result["arithmetic_resolved"].item()
    _contains(result["branch_gap_error"], [Decimal('-.125')])
    assert result["error_upper"].item() < 0
    assert not result["certified_negative"].item()
    assert result["margin_upper"].item() >= .5
