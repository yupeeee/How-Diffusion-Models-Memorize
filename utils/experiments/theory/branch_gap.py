"""Signed branch-gap reference errors in float64 on the input tensor device.

These tensor primitives never transfer latent vectors to the host. Production
callers supply CUDA tensors; CPU tensors are useful only for independent tests.
The unit vector has the explicit extension zero at an exactly zero branch gap.
"""
from __future__ import annotations

import torch

BRANCH_GAP_ERROR_DEFINITION = "signed_projected_branch_gap_reference_error"
BRANCH_GAP_ZERO_CONVENTION = "unit_Delta=0_and_E_parallel=0_when_Delta_is_exactly_zero"


def _components(delta, latent_ndim):
    delta = torch.as_tensor(delta).to(dtype=torch.float64)
    if not isinstance(latent_ndim, int) or not 1 <= latent_ndim <= delta.ndim:
        raise ValueError("latent_ndim must identify at least one existing latent dimension")
    dims = tuple(range(-latent_ndim, 0))
    scale = delta.abs().amax(dim=dims, keepdim=True)
    finite = torch.isfinite(delta).all(dim=dims, keepdim=True)
    zero = (delta == 0).all(dim=dims, keepdim=True)
    scaled = delta / torch.where(finite & (scale > 0), scale, torch.ones_like(scale))
    length = scaled.square().sum(dim=dims, keepdim=True).sqrt()
    return delta, dims, scale, finite, zero, scaled, length


def branch_gap_norm(delta, *, latent_ndim=1):
    """Scaled L2 norm that does not square small raw components first."""
    delta, dims, scale, finite, zero, scaled, length = _components(delta, latent_ndim)
    value = (scale * length).reshape(delta.shape[:-latent_ndim])
    return torch.where(finite.reshape(value.shape), value, torch.full_like(value, torch.nan))


def unit_branch_gap(delta, *, latent_ndim=1):
    """Normalize nonzero finite gaps; exact zero maps to zero, invalid to NaN."""
    delta, dims, scale, finite, zero, scaled, length = _components(delta, latent_ndim)
    unit = scaled / torch.where(length > 0, length, torch.ones_like(length))
    return torch.where(finite, torch.where(zero, torch.zeros_like(unit), unit), torch.full_like(unit, torch.nan))


def projected_branch_gap_error(delta, reference_delta, *, latent_ndim=1):
    """Return unit(delta) dot (delta-reference_delta), without abs or clipping.

    Leading batch dimensions are preserved. Exact-zero delta has error zero
    when the reference vector is finite; missing/nonfinite inputs remain NaN.
    The ``l2``/``rmse`` suffixes used by scalar receipts denote units only.
    """
    delta = torch.as_tensor(delta).to(dtype=torch.float64)
    reference_delta = torch.as_tensor(reference_delta, dtype=torch.float64, device=delta.device)
    delta, reference_delta = torch.broadcast_tensors(delta, reference_delta)
    unit = unit_branch_gap(delta, latent_ndim=latent_ndim)
    dims = tuple(range(-latent_ndim, 0))
    value = (unit * (delta - reference_delta)).sum(dim=dims)
    valid = torch.isfinite(delta).all(dim=dims) & torch.isfinite(reference_delta).all(dim=dims) & torch.isfinite(value)
    return torch.where(valid, value, torch.full_like(value, torch.nan))
