"""Fixed analytical reference evaluations, with no learned-model calls.

All functions are invoked by analysis workers. The initial Gaussian bank is the
same saved-precision bank used by the native direct probes; changing SNR changes
only the finite-law posterior, never a checkpoint timestep or the input draws.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .contracts import TheoryError

REFERENCE_SWEEP_VERSION = "fixed-reference-only-97-v1"
REFERENCE_GRID_POINTS = 97


def reference_grid(initial_snr, decades=6.0):
    initial_snr, decades = float(initial_snr), float(decades)
    if not math.isfinite(initial_snr) or initial_snr <= 0:
        raise TheoryError("Analytical reference grid requires positive finite initial SNR")
    if not math.isfinite(decades) or not 0 < decades <= 12:
        raise TheoryError("Analytical reference decades must be in (0, 12]")
    grid = np.geomspace(initial_snr * 10.0 ** -decades, initial_snr, REFERENCE_GRID_POINTS)
    grid[-1] = initial_snr
    return grid


def gaussian_bank(science, schedule):
    from utils.models.sampling import make_initial_noise
    bank = make_initial_noise(science["seeds"], science["latent_shape"])
    if float(schedule["init_noise_sigma"]) == 1.0:
        bank = bank.to(dtype=getattr(torch, science["scientific_tensor_storage"]["dtype"]))
    return bank.double()


def reference_atoms(law):
    distances = (law.support.flat - law.mean_vector.reshape(1, -1)).norm(dim=1)
    scale = float((law.support.weights * distances.square()).sum().sqrt()) / math.sqrt(law.support.dimension)
    return pd.DataFrame({
        "atom_id": law.support.atom_ids,
        "weight": law.support.weights.detach().cpu().tolist(),
        "atom_center_distance_rmse": (distances / math.sqrt(law.support.dimension)).detach().cpu().tolist(),
        "reference_scale_rmse": scale,
        "reference_law_hash": law.law_hash,
        "input_source": "declared_reference_atoms",
    })


def reference_grid_rows(law, bank, *, seeds, run_id, snr, grid_index, initial_snr, reference_scale_rmse=None):
    """Reduce one fixed grid point to one scalar per unique Gaussian seed."""
    if len(bank) != len(seeds) or len(set(seeds)) != len(seeds):
        raise TheoryError("Reference-only evaluations require the unique native Gaussian seed bank")
    snr = float(snr)
    alpha = math.sqrt(snr / (1.0 + snr))
    sigma = 1.0 / math.sqrt(1.0 + snr)
    value = bank.to(device=law.support.flat.device, dtype=torch.float64)
    means = law.posterior_mean(value, alpha, sigma)
    error = (means - law.mean_vector).flatten(1).norm(dim=1) / math.sqrt(law.support.dimension)
    if not bool(torch.isfinite(error).all()):
        raise TheoryError("Nonfinite analytical reference distance")
    scale = (float(reference_atoms(law).reference_scale_rmse.iloc[0])
             if reference_scale_rmse is None else float(reference_scale_rmse))
    if not math.isfinite(scale) or scale < 0:
        raise TheoryError("Reference scale must be the finite nonnegative declared-law RMS scale")
    return pd.DataFrame({
        "run_id": str(run_id), "seed": list(seeds), "grid_index": int(grid_index),
        "snr": snr, "alpha": alpha, "sigma": sigma,
        "initial_snr": float(initial_snr), "latent_dimension": law.support.dimension,
        "reference_mean_error_rmse": error.detach().cpu().tolist(),
        "reference_scale_rmse": scale, "reference_law_hash": law.law_hash,
        "input_source": "analytical_reference_only", "formula_version": REFERENCE_SWEEP_VERSION,
        "network_evaluated": False,
        "gaussian_bank_precision": "same_fixed_saved_precision_bank_as_native_direct_probes",
    })
