"""Counterexamples and analytic checks for candidate measurements; no inference."""

import math

import numpy as np
import pytest
import torch

from utils.experiments.theory.candidate_metrics import (
    bank_geometry,
    basic_step_metrics,
    candidate_posterior_mean,
    initial_reference_metrics,
    initial_retrieval_metrics,
    reference_only_sweep,
)
from utils.experiments.theory.support import FiniteSupport


def bank(atoms, weights=None):
    atoms = torch.tensor(atoms, dtype=torch.float64)
    return FiniteSupport(
        atoms,
        [str(i) for i in range(len(atoms))],
        {str(i): i for i in range(len(atoms))},
        weights=weights,
    )


def test_equal_errors_do_not_imply_small_gap_and_equal_wrong_vectors_do_not_recover():
    target = torch.zeros(2)
    result = basic_step_metrics(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]]), target, 2
    )
    assert (
        result["unconditional_target_error_rmse"].item()
        == result["conditional_target_error_rmse"].item()
    )
    assert result["delta_norm"].item() == 2
    assert result["paired_error_dot_l2_squared"].item() == -1
    wrong = basic_step_metrics(torch.ones(1, 2), torch.ones(1, 2), target, 2)
    assert wrong["delta_norm"].item() == 0
    assert wrong["joint_target_error_rmse"].item() == 1


def test_initial_coordinates_retain_perpendicular_components_and_negative_injection():
    result = basic_step_metrics(
        torch.tensor([[0.0, 2.0]]),
        torch.tensor([[-1.0, 2.0]]),
        torch.tensor([1.0, 0.0]),
        2,
        center=torch.zeros(2),
    )
    assert result["initial_target_coordinate_u"].item() == 0
    assert result["initial_target_coordinate_c"].item() == -1
    assert result["initial_target_coordinate_g"].item() == -2
    assert result["initial_perpendicular_g_rmse"].item() == pytest.approx(math.sqrt(2))
    assert result["a_parallel"].item() == -2
    assert result["injection_relative_mismatch"].item() == 2


def test_degenerate_center_has_no_invented_direction_but_keeps_errors():
    target = torch.ones(2)
    result = basic_step_metrics(
        torch.zeros(1, 2), torch.ones(1, 2), target, 7.5, center=target
    )
    assert not result["initial_target_direction_valid"].item()
    assert torch.isnan(result["initial_target_coordinate_c"]).all()
    assert result["conditional_target_error_rmse"].item() == 0


def test_source_precision_gate_is_explicit():
    result = basic_step_metrics(
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.tensor([1.0 + 1e-7], dtype=torch.float64),
        2,
        center=torch.ones(1),
        source_epsilon=torch.finfo(torch.float32).eps,
    )
    assert not result["initial_target_direction_valid"].item()


def test_large_canceling_reference_errors_preserve_signed_projection():
    # Both residuals share a large orthogonal component, while Delta targets x*.
    result = basic_step_metrics(
        torch.tensor([[0.0, 100.0]]),
        torch.tensor([[1.0, 100.0]]),
        torch.tensor([1.0, 0.0]),
        2,
        current_reference=torch.tensor([[0.0, 0.0]]),
    )
    assert result["combined_reference_error_norm"].item() == 0
    assert result["signed_error_projection"].item() == 0
    assert result["conditional_error"].item() == 100
    assert result["delta_norm"].item() == 1


def test_euclidean_retrieval_ties_ignore_nonuniform_posterior_priors():
    support = bank([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], [0.001, 0.001, 0.998])
    estimates = torch.tensor([[1.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
    result = initial_retrieval_metrics(estimates, estimates, support, "0")
    assert result["initial_conditional_target_rank"].tolist() == [1, 3]
    assert result["initial_conditional_target_tie_count"].tolist() == [2, 1]
    assert result["initial_retrieval_bank_size"] == 3


def test_exact_atom_deduplication_precedes_retrieval():
    support = FiniteSupport.from_candidates(
        [
            ("a", torch.tensor([0.0])),
            ("copy", torch.tensor([0.0])),
            ("b", torch.tensor([1.0])),
        ]
    )
    result = initial_retrieval_metrics(
        torch.tensor([[0.0]]), torch.tensor([[0.0]]), support, "copy"
    )
    assert result["initial_retrieval_bank_size"] == 2
    assert result["initial_conditional_target_tie_count"].item() == 1
    with pytest.raises(ValueError, match="absent"):
        initial_retrieval_metrics(
            torch.tensor([[0.0]]), torch.tensor([[0.0]]), support, "missing"
        )


def test_nonuniform_candidate_mean_matches_direct_gaussian_weights():
    support = bank([[-1.0], [2.0]], [0.2, 0.8])
    z = torch.tensor([[0.7]], dtype=torch.float64)
    a, s = 0.5, 0.8
    logits = support.weights.log() - ((z - support.flat.T * a) ** 2) / (2 * s * s)
    expected = torch.softmax(logits, -1) @ support.flat
    got = candidate_posterior_mean(z, support, a, s)
    assert torch.allclose(got, expected, rtol=1e-14, atol=1e-14)
    metrics = initial_reference_metrics(z, z, support, a, s)
    assert metrics["initial_candidate_reference_movement_rmse"].item() == pytest.approx(
        abs(expected.item() - 1.4)
    )


def test_reference_sweep_reuses_fixed_inputs_and_all_prespecified_snr_values():
    support = bank([[-2.0], [1.0]], [0.3, 0.7])
    z = torch.tensor([[-0.5], [2.0]], dtype=torch.float64)
    sweep = reference_only_sweep(z, [0, 1], support, 0.01)
    assert len(sweep) == 98
    assert sweep.snr_grid_index.nunique() == 49
    assert sweep.analytical_snr.max() == pytest.approx(0.01)
    assert sweep.analytical_snr.min() == pytest.approx(1e-8)
    assert set(sweep.evidence) == {"reference_only_illustration"}
    last = sweep.loc[sweep.snr_grid_index == 48].sort_values("seed")
    expected = candidate_posterior_mean(
        z, support, last.alpha.iloc[0], last.sigma.iloc[0]
    )
    expected = (expected - (support.weights @ support.flat)).abs().reshape(-1)
    assert np.allclose(last.candidate_reference_movement_rmse, expected.numpy())
    with pytest.raises(ValueError, match="distinct seed"):
        reference_only_sweep(z, [0, 0], support, 0.01)


def test_bank_geometry_preserves_distinct_population_and_weights():
    result = bank_geometry(
        bank([[0.0, 0.0], [2.0, 0.0]], [0.1, 0.9]), torch.tensor([1.0, 0.0])
    )
    assert len(result) == 2
    assert np.allclose(result.candidate_atom_center_distance_rmse, 1 / math.sqrt(2))
    assert result.weight.tolist() == pytest.approx([0.1, 0.9])


@pytest.mark.parametrize("guidance", [0.0, -2.0, float("nan")])
def test_nonpositive_or_nonfinite_guidance_does_not_create_normalized_injection(
    guidance,
):
    values = basic_step_metrics(
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.ones(2),
        guidance,
        center=torch.zeros(2),
    )
    assert torch.isnan(values["injection_relative_mismatch"]).all()
    assert (
        values["initial_injection_relative_mismatch_status"]
        == "undefined_nonpositive_or_nonfinite_guidance"
    )
    assert values["candidate_guidance_domain_status"].startswith("not_applicable")
    assert torch.isfinite(values["conditional_target_error_rmse"]).all()
