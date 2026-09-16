"""Author-run regressions for the fixed analytical-only reference grid."""
import math

import numpy as np
import pytest
import torch

from utils.experiments.theory.contracts import TheoryError
from utils.experiments.theory.evidence_reference import gaussian_bank, reference_grid, reference_grid_rows
from utils.experiments.theory.reference_law import reference_law_from_atoms


def test_fixed_reference_grid_extends_only_the_posterior_and_converges(monkeypatch):
    from utils.experiments.theory import direct_probes
    monkeypatch.setattr(direct_probes, "_load_replica", lambda *a, **k: pytest.fail("analytical reference loaded a model"))
    law = reference_law_from_atoms([torch.tensor([2., 1.]), torch.tensor([6., -1.])], ["a", "b"], weights=[1, 3])
    grid = reference_grid(0.01)
    assert len(grid) == 97 and grid[-1] == 0.01
    assert grid[0] == pytest.approx(1e-8)
    assert np.all(np.diff(grid) > 0)
    bank = torch.tensor([[0.3, -0.1], [-0.8, 0.4]], dtype=torch.float64)
    rows = [reference_grid_rows(law, bank, seeds=[5, 7], run_id="run", snr=float(snr),
                               grid_index=index, initial_snr=0.01) for index, snr in enumerate(grid)]
    assert rows[0].reference_mean_error_rmse.max() < rows[-1].reference_mean_error_rmse.max()
    assert rows[0].reference_mean_error_rmse.max() < 1e-3
    assert all(frame.input_source.eq("analytical_reference_only").all() for frame in rows)
    assert all(not frame.network_evaluated.any() for frame in rows)
    assert all(frame.seed.tolist() == [5, 7] for frame in rows)
    np.testing.assert_allclose(rows[-1].alpha, math.sqrt(.01 / 1.01))
    np.testing.assert_allclose(rows[-1].sigma, 1 / math.sqrt(1.01))


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
