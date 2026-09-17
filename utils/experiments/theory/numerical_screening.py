"""Batched CUDA enclosures for the corrected branch-gap-error condition.

Only the current posterior is evaluated here. With E=<Delta/||Delta||, Delta-barDelta> and
V>=0, an outward upper bound for D-E below zero certifies D-E-V<0 without an
endpoint, gain or variation calculation. The former D<e_c shortcut is invalid
because the two learned-reference vector errors can cancel.
"""
from __future__ import annotations

import math

import torch

from .gpu_intervals import ArithmeticBudgetExceeded, GpuDirected, GpuInterval
from .gpu_refinement import projected_branch_gap_error

SCREENING_VERSION = "projected-gap-error-1-batched-posterior-screen-1"
_SCREEN_WORKING_LANES = 8 * 1024 * 1024


def screen_negative_condition(verified_inputs, *, device=None):
    """Reject the retired conditional-error proof before accessing inputs."""
    raise ValueError(
        "Conditional-error screening is retired for projected-gap-error-1; "
        "enclose the branch-gap error from the posterior before testing D-branch_gap_error"
    )


class _BatchDirected(GpuDirected):
    """Uniform batched operations, charged conservatively to each seed row.

    Shared input/geometry work uses a separate ordinary GpuDirected context and
    is charged in full to every row. All operations in this context have the
    batch axis (or are scalar metadata operations); ceil(lanes/B) bounds every
    row's lane cost. No data-dependent subset changes the batch denominator.
    """
    def __init__(self, batch_size, max_products, *, device):
        self.batch_size = batch_size
        self.scalar_lane_products = 0
        super().__init__(max_products=max_products, device=device)

    def charge(self, count=1):
        if not isinstance(count, int) or count < 0:
            raise ValueError("GPU interval work charge must be nonnegative")
        super().charge((count + self.batch_size - 1) // self.batch_size)
        self.scalar_lane_products += count


def _prepared_inputs(support, verified_inputs):
    device = torch.device(support.flat.device)
    if device.type != "cuda" or getattr(torch.version, "hip", None):
        raise ValueError("Branch-gap screening requires NVIDIA CUDA; CPU fallback is disabled")
    if verified_inputs.get("stored_prediction_type", "epsilon") != "epsilon":
        raise ValueError("Branch-gap screening requires canonical saved epsilon")
    prepared = dict(verified_inputs)
    for name in ("state", "epsilon_u", "epsilon_c", "saved_endpoint", "independent_innovation",
                 "target", "source_endpoint_radii"):
        if prepared.get(name) is not None:
            prepared[name] = torch.as_tensor(prepared[name], device=device, dtype=torch.float64)
    state = prepared["state"]
    if state.ndim < 2 or len(state) < 1:
        raise ValueError("Branch-gap screening requires a nonempty saved vector batch")
    count, dimension = len(state), int(support.dimension)
    if state.reshape(count, -1).shape[1] != dimension or dimension < 1:
        raise ValueError("Branch-gap screening state/support dimensions differ")
    if any(prepared.get(name) is None for name in ("epsilon_u", "epsilon_c")):
        raise ValueError("Branch-gap screening requires both saved epsilon branches")
    for name in ("epsilon_u", "epsilon_c", "saved_endpoint", "independent_innovation"):
        if prepared.get(name) is not None and prepared[name].shape != state.shape:
            raise ValueError("Branch-gap screening saved batch shapes differ: " + name)
    if prepared["target"].numel() != dimension:
        raise ValueError("Branch-gap screening target/support dimensions differ")
    if prepared.get("source_endpoint_radii") is not None:
        if prepared["source_endpoint_radii"].numel() != 2 * count:
            raise ValueError("Branch-gap screening endpoint radius shape differs")
        prepared["source_endpoint_radii"] = prepared["source_endpoint_radii"].reshape(count, 2)
    alpha, sigma = float(prepared["alpha"]), float(prepared["sigma"])
    if not (math.isfinite(alpha) and math.isfinite(sigma) and alpha > 0 and sigma > 0):
        raise ValueError("Branch-gap screening requires finite positive alpha and sigma")
    return prepared, count, dimension, device


def screen_branch_gap_condition(support, verified_inputs, *, max_products, target_atom=None):
    """Enclose all current-posterior D/E pairs with one CUDA seed batch.

    Returned tensors and interval objects stay on CUDA. Only compact status and
    deterministic operation counters are host values. ``operations_per_row``
    includes the full shared-geometry cost plus a conservative per-row batched
    charge; no row exceeds ``max_products``. Exhaustion never invents a sign.
    ``prepared_inputs`` can be reused by subsequent raw-row refinement without
    retransferring the full saved batch for each seed.
    """
    prepared, count, dimension, device = _prepared_inputs(support, verified_inputs)
    shared = GpuDirected(max_products=max_products, device=device)
    flat = torch.as_tensor(support.flat, device=device, dtype=torch.float64)
    masses = torch.as_tensor(support.weights, device=device, dtype=torch.float64)
    target = prepared["target"].reshape(-1)
    if flat.shape != (support.size, dimension) or masses.shape != (support.size,) or support.size < 1:
        raise ValueError("Branch-gap screening support shape differs")
    if not bool(torch.isfinite(flat).all() & torch.isfinite(masses).all() & (masses > 0).all()):
        raise ValueError("Branch-gap screening needs finite atoms and positive finite masses")
    if target_atom is None:
        matches = (flat == target).all(dim=1).nonzero().flatten()
        if len(matches) != 1:
            raise ValueError("Branch-gap screening requires one exact target atom")
        target_atom = int(matches[0])
    if type(target_atom) is not int or not 0 <= target_atom < support.size or not torch.equal(target, flat[target_atom]):
        raise ValueError("Branch-gap screening target is not the declared exact atom")
    nan = torch.full((count,), torch.nan, device=device, dtype=torch.float64)
    result = {
        "certified_negative": torch.zeros(count, device=device, dtype=torch.bool),
        "arithmetic_resolved": torch.zeros(count, device=device, dtype=torch.bool),
        "margin_upper": nan.clone(), "delta_lower": nan.clone(), "delta_upper": nan.clone(),
        "error_lower": nan.clone(), "error_upper": nan.clone(),
        "current_mean": None, "delta": None, "D": None, "branch_gap_error": None,
        "prepared_inputs": prepared, "complete": False, "status": "budget_exhausted",
        "operations_per_row": 0, "total_operations": 0,
        "method": "batched_cuda_outward_current_posterior_D_minus_signed_projected_error",
        "version": SCREENING_VERSION,
    }
    batched = None
    # Four-corner interval products dominate transient memory. Bound their
    # working lanes independently of the full support/seed/latent dimensions.
    chunk = max(1, min(int(support.candidate_chunk), _SCREEN_WORKING_LANES // (count * dimension)))
    result["support_chunk_size"] = chunk
    try:
        # Validated exact binary64 inputs need no copied point arrays.
        shared.charge(flat.numel() + masses.numel())
        atoms, mass = GpuInterval(flat, flat), GpuInterval(masses, masses)
        alpha, sigma = shared.interval(prepared["alpha"]), shared.interval(prepared["sigma"])
        sigma2 = shared.square(sigma)
        beta = shared.div(alpha, sigma2)
        gamma = shared.div(shared.square(alpha), shared.mul(shared.interval(2.), sigma2))
        scaled_target = shared.mul(alpha, atoms[target_atom])
        equal_masses = bool((masses == masses[target_atom]).all())
        geometry = []
        for start in range(0, support.size, chunk):
            stop = min(start + chunk, support.size)
            difference = shared.sub(atoms[start:stop], atoms[target_atom])
            squared_distance = shared.total(shared.square(difference), dim=-1)
            if equal_masses:
                # Exact equal positive binary64 masses give prior ratio one.
                prior = shared.interval(torch.zeros(stop - start, device=device, dtype=torch.float64))
            else:
                prior = shared.log(shared.div(mass[start:stop], mass[target_atom]))
            geometry.append((start, stop, difference, squared_distance, prior))
        batched = _BatchDirected(count, max_products - shared.products, device=device)
        state = batched.interval(prepared["state"].reshape(count, dimension))
        epsilon_u = batched.interval(prepared["epsilon_u"].reshape(count, dimension))
        epsilon_c = batched.interval(prepared["epsilon_c"].reshape(count, dimension))
        delta = batched.div(batched.mul(sigma, batched.sub(epsilon_u, epsilon_c)), alpha)
        D = batched.norm(delta)
        offset = batched.sub(state, scaled_target)
        parts = []
        for _, _, difference, squared_distance, prior in geometry:
            projected = batched.dot(offset.unsqueeze(-2), difference.unsqueeze(0), dim=-1)
            # Expand the shared terms so every charged arithmetic output in the
            # batched context has the same explicit seed axis.
            penalty = GpuInterval(squared_distance.lower.expand(count, -1), squared_distance.upper.expand(count, -1))
            log_prior = GpuInterval(prior.lower.expand(count, -1), prior.upper.expand(count, -1))
            parts.append(batched.add(batched.sub(batched.mul(beta, projected), batched.mul(gamma, penalty)), log_prior))
        lower = torch.cat([part.lower for part in parts], dim=-1)
        upper = torch.cat([part.upper for part in parts], dim=-1)
        lower[:, target_atom], upper[:, target_atom] = 0., 0.
        weights = batched.softmax(GpuInterval(lower, upper), dim=-1)
        # Target-centered accumulation retains precision near concentrated
        # posteriors; every support sum is an outward pairwise reduction.
        mean_offset = batched.interval(torch.zeros((count, dimension), device=device, dtype=torch.float64))
        for start, stop, difference, _, _ in geometry:
            piece = GpuInterval(weights.lower[:, start:stop], weights.upper[:, start:stop]).unsqueeze(-1)
            mean_offset = batched.add(mean_offset, batched.total(batched.mul(piece, difference.unsqueeze(0)), dim=-2))
        current_mean = batched.add(atoms[target_atom], mean_offset)
        # barDelta = target-current_mean = -mean_offset exactly. Avoid an
        # unnecessary subtraction of a large shared target from its mean.
        E = projected_branch_gap_error(batched, delta, batched.neg(mean_offset), D)
        margin = batched.sub(D, E)
        # Broad intervals can recover finite posterior hulls even when a raw
        # state was nonfinite; that source row must still remain unresolved.
        finite_source = (torch.isfinite(prepared["state"].reshape(count, dimension)).all(dim=-1)
                         & torch.isfinite(prepared["epsilon_u"].reshape(count, dimension)).all(dim=-1)
                         & torch.isfinite(prepared["epsilon_c"].reshape(count, dimension)).all(dim=-1))
        resolved = (finite_source & torch.isfinite(D.lower) & torch.isfinite(D.upper)
                    & torch.isfinite(E.lower) & torch.isfinite(E.upper)
                    & torch.isfinite(margin.upper)
                    & torch.isfinite(current_mean.lower).all(dim=-1)
                    & torch.isfinite(current_mean.upper).all(dim=-1)
                    & torch.isfinite(delta.lower).all(dim=-1)
                    & torch.isfinite(delta.upper).all(dim=-1))
        result.update(
            certified_negative=resolved & (margin.upper < 0), arithmetic_resolved=resolved,
            margin_upper=torch.where(resolved, margin.upper, nan),
            delta_lower=D.lower, delta_upper=D.upper, error_lower=E.lower, error_upper=E.upper,
            current_mean=current_mean, delta=delta, D=D, branch_gap_error=E,
            complete=True, status="complete" if bool(resolved.all()) else "unsupported_arithmetic_range",
        )
    except ArithmeticBudgetExceeded:
        # A partial posterior is never a posterior of a smaller reference law.
        # Preserve the spent budget, but expose no incomplete enclosure/sign.
        pass
    result["operations_per_row"] = shared.products + (0 if batched is None else batched.products)
    result["total_operations"] = shared.products + (0 if batched is None else batched.scalar_lane_products)
    return result
