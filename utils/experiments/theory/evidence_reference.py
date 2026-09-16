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

REFERENCE_SWEEP_VERSION = "fixed-reference-only-97-v3"
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


def _mean_fields(law):
    mean, bank_mean = law.theory_mean_vector, law.mean_vector
    root_d = math.sqrt(law.support.dimension)
    mean_norm, bank_norm, offset = (float(value.norm()) for value in (mean, bank_mean, bank_mean - mean))
    if not all(math.isfinite(value) for value in (mean_norm, bank_norm, offset)):
        raise TheoryError("Nonfinite theory comparison mean or finite-bank mean")
    receipt = law.theory_mean_metadata
    return {
        "mean_norm_l2": mean_norm, "mean_norm_rmse": mean_norm / root_d,
        "bank_mean_norm_l2": bank_norm, "bank_mean_norm_rmse": bank_norm / root_d,
        "mean_offset_l2": offset, "mean_offset_rmse": offset / root_d,
        "theory_mean_source": str(receipt.get("source", "declared_finite_bank_mean")),
        "theory_mean_sha256": str(receipt.get("vector_sha256", law.metadata.get("mean_sha256", ""))),
    }


def reference_atoms(law):
    mean = law.theory_mean_vector.reshape(1, -1)
    bank_mean = law.mean_vector.reshape(1, -1)
    root_d = math.sqrt(law.support.dimension)
    distances = (law.support.flat - mean).norm(dim=1)
    bank_distances = (law.support.flat - bank_mean).norm(dim=1)
    zero_distances = law.support.flat.norm(dim=1)
    # The empirical-law variance and Eq. 23 still use its own exact mean.
    scale = float((law.support.weights * bank_distances.square()).sum().sqrt()) / root_d
    comparison_scale = float((law.support.weights * distances.square()).sum().sqrt()) / root_d
    return pd.DataFrame({
        "atom_id": law.support.atom_ids,
        "weight": law.support.weights.detach().cpu().tolist(),
        "atom_center_distance_rmse": (distances / root_d).detach().cpu().tolist(),
        "atom_bank_center_distance_rmse": (bank_distances / root_d).detach().cpu().tolist(),
        "atom_zero_distance_rmse": (zero_distances / root_d).detach().cpu().tolist(),
        **_mean_fields(law),
        "reference_scale_rmse": scale,
        "comparison_scale_rmse": comparison_scale,
        "latent_dimension": law.support.dimension,
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
    mean, bank_mean = law.theory_mean_vector, law.mean_vector
    root_d = math.sqrt(law.support.dimension)
    error = (means - mean).flatten(1).norm(dim=1) / root_d
    bank_error = (means - bank_mean).flatten(1).norm(dim=1)
    zero_error = means.flatten(1).norm(dim=1)
    if not all(bool(torch.isfinite(value).all()) for value in (error, bank_error, zero_error)):
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
        "reference_zero_error_l2": zero_error.detach().cpu().tolist(),
        "reference_zero_error_rmse": (zero_error / root_d).detach().cpu().tolist(),
        "reference_to_bank_mean_error_l2": bank_error.detach().cpu().tolist(),
        "reference_to_bank_mean_error_rmse": (bank_error / root_d).detach().cpu().tolist(),
        **_mean_fields(law),
        "reference_scale_rmse": scale, "reference_law_hash": law.law_hash,
        "input_source": "analytical_reference_only", "formula_version": REFERENCE_SWEEP_VERSION,
        "network_evaluated": False,
        "gaussian_bank_precision": "same_fixed_saved_precision_bank_as_native_direct_probes",
    })
