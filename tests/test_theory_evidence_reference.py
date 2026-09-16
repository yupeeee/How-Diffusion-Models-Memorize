"""Author-run regressions for the fixed analytical-only reference grid."""
import math

import numpy as np
import pytest
import torch

from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.evidence_reference import gaussian_bank, reference_atoms, reference_grid, reference_grid_rows
from utils.experiments.theory.reference_law import ReferenceLaw, reference_law_from_atoms
from utils.experiments.theory.support import _tensor_hash


def test_fixed_reference_grid_extends_only_the_posterior_and_converges(monkeypatch):
    from utils.experiments.theory import direct_probes
    monkeypatch.setattr(direct_probes, "_load_replica", lambda *a, **k: pytest.fail("analytical reference loaded a model"))
    law = reference_law_from_atoms([torch.tensor([2., 1.]), torch.tensor([6., -1.])], ["a", "b"], weights=[1, 3])
    posterior_means = []
    original_posterior_mean = law.posterior_mean

    def measured_posterior_mean(*args, **kwargs):
        result = original_posterior_mean(*args, **kwargs)
        posterior_means.append(result.clone())
        return result

    monkeypatch.setattr(law, "posterior_mean", measured_posterior_mean)
    grid = reference_grid(0.01)
    assert len(grid) == 97 and grid[-1] == 0.01
    assert grid[0] == pytest.approx(1e-8)
    assert np.all(np.diff(grid) > 0)
    bank = torch.tensor([[0.3, -0.1], [-0.8, 0.4]], dtype=torch.float64)
    rows = [reference_grid_rows(law, bank, seeds=[5, 7], run_id="run", snr=float(snr),
                               grid_index=index, initial_snr=0.01) for index, snr in enumerate(grid)]
    assert rows[0].reference_mean_error_rmse.max() < rows[-1].reference_mean_error_rmse.max()
    assert rows[0].reference_mean_error_rmse.max() < 1e-3
    assert len(posterior_means) == len(grid)  # No second posterior evaluation for zero distances.
    declared_mean_norm = math.sqrt(5.0 ** 2 + 0.5 ** 2)
    for frame, posterior in zip(rows, posterior_means, strict=True):
        np.testing.assert_allclose(frame.reference_zero_error_l2, posterior.norm(dim=1).numpy())
        np.testing.assert_allclose(frame.reference_zero_error_rmse, frame.reference_zero_error_l2 / math.sqrt(2))
        np.testing.assert_allclose(frame.mean_norm_l2, declared_mean_norm)
        np.testing.assert_allclose(frame.mean_norm_rmse, declared_mean_norm / math.sqrt(2))
    # At low SNR the posterior approaches the nonzero mean, not the origin.
    np.testing.assert_allclose(rows[0].reference_zero_error_rmse, declared_mean_norm / math.sqrt(2), atol=1e-3)
    assert rows[0].reference_zero_error_rmse.min() > 3
    assert all(frame.input_source.eq("analytical_reference_only").all() for frame in rows)
    assert all(not frame.network_evaluated.any() for frame in rows)
    assert all(frame.seed.tolist() == [5, 7] for frame in rows)
    np.testing.assert_allclose(rows[-1].alpha, math.sqrt(.01 / 1.01))
    np.testing.assert_allclose(rows[-1].sigma, 1 / math.sqrt(1.01))


def test_reference_atom_zero_distances_keep_the_weighted_mean_and_centered_scale():
    law = reference_law_from_atoms([torch.tensor([2., 1.]), torch.tensor([6., -1.])],
                                 ["a", "b"], weights=[1, 3])
    rows = reference_atoms(law)
    np.testing.assert_allclose(rows.weight, [0.25, 0.75])
    np.testing.assert_allclose(rows.atom_zero_distance_rmse, np.sqrt([5.0, 37.0]) / math.sqrt(2))
    # The declared mean is [5, -0.5], including its nonuniform atom weights.
    np.testing.assert_allclose(rows.atom_center_distance_rmse, np.sqrt([11.25, 1.25]) / math.sqrt(2))
    np.testing.assert_allclose(rows.mean_norm_l2, math.sqrt(25.25))
    np.testing.assert_allclose(rows.mean_norm_rmse, math.sqrt(25.25 / 2))
    np.testing.assert_allclose(rows.reference_scale_rmse, math.sqrt(3.75 / 2))


@pytest.mark.parametrize("sigma", [1.0, 1.7])
def test_reference_bank_preserves_the_existing_native_probe_precision(sigma):
    from utils.models.sampling import make_initial_noise
    science = {"seeds": [0, 1, 2], "latent_shape": [1, 2, 2], "scientific_tensor_storage": {"dtype": "float16"}}
    original = make_initial_noise(science["seeds"], science["latent_shape"])
    expected = original.half().double() if sigma == 1 else original.double()
    torch.testing.assert_close(gaussian_bank(science, {"init_noise_sigma": sigma}), expected, rtol=0, atol=0)


@pytest.mark.parametrize("snr,decades", [(0, 6), (-1, 6), (math.inf, 6), (.1, 0), (.1, 13)])
def test_invalid_reference_grid_does_not_create_an_alternate_design(snr, decades):
    with pytest.raises(TheoryError):
        reference_grid(snr, decades)


def test_selected_reference_mean_changes_comparisons_without_recentering_the_finite_law():
    law = reference_law_from_atoms([torch.tensor([2., 1.]), torch.tensor([6., -1.])],
                                 ["a", "b"], weights=[1, 3])
    bank = torch.tensor([[.3, -.1], [-.8, .4]], dtype=torch.float64)
    selected = torch.tensor([1., 2.], dtype=torch.float64)
    original_mean, original_weights = law.mean_vector.clone(), law.support.weights.clone()
    original_posterior = law.posterior_mean(bank, .1, math.sqrt(.99)).clone()
    receipt = {"source": "initial_unconditional_reference_monte_carlo", "vector_sha256": _tensor_hash(selected), "sample_count": 10000}
    law = ReferenceLaw(law.support, law.metadata | {"theory_mean": receipt}, "selected-mean-fixture", selected)
    atoms = reference_atoms(law)
    np.testing.assert_allclose(atoms.atom_center_distance_rmse, [1., math.sqrt(17.)])
    np.testing.assert_allclose(atoms.atom_bank_center_distance_rmse, np.sqrt([11.25, 1.25]) / math.sqrt(2))
    np.testing.assert_allclose(atoms.reference_scale_rmse, math.sqrt(3.75 / 2))
    np.testing.assert_allclose(atoms.comparison_scale_rmse, math.sqrt(13.))
    np.testing.assert_allclose(atoms.comparison_scale_rmse**2,
                               atoms.reference_scale_rmse**2 + atoms.mean_offset_rmse**2)
    np.testing.assert_allclose(atoms.mean_norm_l2, math.sqrt(5.))
    np.testing.assert_allclose(atoms.bank_mean_norm_l2, math.sqrt(25.25))
    assert atoms.theory_mean_source.eq("initial_unconditional_reference_monte_carlo").all()
    assert atoms.theory_mean_sha256.eq(receipt["vector_sha256"]).all()
    low = reference_grid_rows(law, bank, seeds=[5, 7], run_id="run", snr=1e-12,
                              grid_index=0, initial_snr=.01)
    # The analytical posterior still approaches the atom mean; its distance to
    # the independently selected centre consequently has a nonzero limit.
    np.testing.assert_allclose(low.reference_mean_error_rmse, math.sqrt(22.25 / 2), atol=1e-4)
    np.testing.assert_allclose(low.reference_zero_error_rmse, math.sqrt(25.25 / 2), atol=1e-4)
    assert low.reference_to_bank_mean_error_rmse.max() < 1e-4
    torch.testing.assert_close(law.mean_vector, original_mean, rtol=0, atol=0)
    torch.testing.assert_close(law.support.weights, original_weights, rtol=0, atol=0)
    torch.testing.assert_close(law.posterior_mean(bank, .1, math.sqrt(.99)), original_posterior, rtol=0, atol=0)
