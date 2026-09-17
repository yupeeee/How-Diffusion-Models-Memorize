"""CUDA binary64 enclosures for exact saved inputs and affine finite-law paths.

Only scalar bounds and control decisions leave the device. These enclosures
have fixed binary64 resolution; additional nodes/operations do not add digits.
"""
from __future__ import annotations

import torch

from .gpu_intervals import GpuInterval


def require_cuda(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Theory numerical measurements require CUDA; CPU fallback is disabled")
    return device


def _cat(values, dim=0):
    return GpuInterval(torch.cat([value.lower for value in values], dim=dim),
                       torch.cat([value.upper for value in values], dim=dim))


def _stack(values, dim=0):
    return GpuInterval(torch.stack([value.lower for value in values], dim=dim),
                       torch.stack([value.upper for value in values], dim=dim))


def posterior_mean(c, atoms, weights, *, chunk=256):
    mean = c.interval(torch.zeros(atoms.lower.shape[-1], device=c.device, dtype=torch.float64))
    for start in range(0, len(atoms.lower), chunk):
        mean = c.add(mean, c.total(c.mul(weights[start:start + chunk].unsqueeze(-1),
                                        atoms[start:start + chunk]), dim=0))
    return mean


def interval_gain(c, intercept, slopes, target_atom):
    """Gain enclosure of the supplied interval affine logits, on CUDA."""
    if intercept.lower.ndim != 1 or intercept.lower.shape != slopes.lower.shape:
        raise ValueError("Gain enclosure requires one matching logit vector")
    size = len(intercept.lower)
    if not 0 <= target_atom < size:
        raise ValueError("Gain dimensions/target differ")
    zero = c.interval(0.)
    if size == 1:
        return {"H": zero, "G": None, "gain_sign": "zero", "slope0": zero, "slope1": zero}
    keep = torch.arange(size, device=c.device) != target_atom
    b = c.sub(intercept[keep], intercept[target_atom])
    a = c.sub(slopes[keep], slopes[target_atom])
    if bool(((a.lower == 0) & (a.upper == 0)).all()):
        return {"H": zero, "G": zero, "gain_sign": "zero", "slope0": zero, "slope1": zero}
    weights = c.softmax(b)
    shift = c.interval(a.upper.amax())
    expectation = c.total(c.mul(weights, c.exp(c.sub(a, shift))))
    G = c.neg(c.add(shift, c.log(expectation)))
    end = c.add(b, a)
    H = c.sub(c.softplus(c.logsumexp(b)), c.softplus(c.logsumexp(end)))
    return {"H": H, "G": G, "gain_sign": G.sign,
            "slope0": c.neg(c.dot(weights, a)), "slope1": c.neg(c.dot(c.softmax(end), a))}



def projected_branch_gap_error(c, delta, reference_delta, D=None):
    """Outward signed <Delta/||Delta||, Delta-referenceDelta>, per CUDA row.

    Exact zero vectors use the declared zero-direction convention. A nonzero
    interval whose norm touches zero has no certified direction and yields an
    unbounded error interval; it must never inherit a norm-based certificate.
    """
    delta, reference_delta = c.interval(delta), c.interval(reference_delta)
    if delta.lower.shape != reference_delta.lower.shape or delta.lower.ndim < 1:
        raise ValueError("Projected error requires matching actual branch-gap vectors")
    D = c.norm(delta) if D is None else c.interval(D)
    if D.lower.shape != delta.lower.shape[:-1]:
        raise ValueError("Projected error direction norm has an invalid shape")
    exact_zero = ((delta.lower == 0) & (delta.upper == 0)).all(dim=-1)
    exact_zero = exact_zero & (D.lower == 0) & (D.upper == 0)
    if bool(exact_zero.all()):
        return c.interval(torch.zeros_like(D.lower))
    resolved = (D.lower > 0) & torch.isfinite(D.upper)
    denominator = GpuInterval(torch.where(exact_zero, 1., D.lower),
                              torch.where(exact_zero, 1., D.upper))
    unit = c.div(delta, denominator.unsqueeze(-1))
    signed = c.dot(unit, c.sub(delta, reference_delta))
    return GpuInterval(torch.where(exact_zero, 0., torch.where(resolved, signed.lower, -torch.inf)),
                       torch.where(exact_zero, 0., torch.where(resolved, signed.upper, torch.inf)))


def negative_condition_precheck(D, branch_gap_error, *, arithmetic):
    """V >= 0: enclose the substituted margin above by D - branch_gap_error."""
    c = arithmetic
    upper = c.sub(D, branch_gap_error).upper
    return {"upper": upper, "condition_sign_status": "negative" if bool(upper < 0) else "unresolved",
            "condition_value_status": "unavailable_variation_not_computed"}


def raw_row(verified, index, *, device):
    if verified is None:
        return None
    require_cuda(device)
    result = dict(verified)
    for key in ("state", "epsilon_u", "epsilon_c", "saved_endpoint", "independent_innovation"):
        if result.get(key) is not None:
            value = torch.as_tensor(result[key], device=device, dtype=torch.float64)
            result[key] = value.reshape(len(value), -1)[index]
    if result.get("source_endpoint_radii") is not None:
        result["source_endpoint_radii"] = torch.as_tensor(result["source_endpoint_radii"],
                                                         device=device, dtype=torch.float64).reshape(-1, 2)[index]
    result["target"] = torch.as_tensor(result["target"], device=device, dtype=torch.float64).reshape(-1)
    return result


def _current_reference(c, supplied, dimension):
    """Validate an already charged enclosure from the same raw-input screen.

    Source/row identity is the caller's verified screen receipt. This helper
    accepts intervals only: rounded posterior estimates cannot become certified
    point intervals by passing through this optimization.
    """
    required = ("current_mean", "D", "branch_gap_error")
    if not isinstance(supplied, dict) or any(name not in supplied for name in required):
        raise ValueError("Precomputed current reference requires current_mean, D and branch_gap_error intervals")
    result = {}
    for name in (*required, *(("delta",) if "delta" in supplied else ())):
        value = supplied[name]
        if not isinstance(value, GpuInterval):
            raise ValueError("Precomputed current reference must contain CUDA interval enclosures")
        value = c.interval(value)  # device validation; does not re-charge existing work
        shape = (dimension,) if name in {"current_mean", "delta"} else ()
        if tuple(value.lower.shape) != shape:
            raise ValueError("Precomputed current reference has an invalid row shape for " + name)
        result[name] = value
    return result


def clean_intervals(c, raw, *, current_reference=None):
    """Enclose clean inputs; reuse the already charged screen's delta and norm.

    With a current-reference receipt the conditional clean estimate and its
    individual reference-error norm are unused, and their tuple slots are None.
    """
    state, u = (c.interval(raw[key]) for key in ("state", "epsilon_u"))
    alpha, sigma = c.interval(raw["alpha"]), c.interval(raw["sigma"])
    mu = c.div(c.sub(state, c.mul(sigma, u)), alpha)
    if current_reference is not None:
        current = _current_reference(c, current_reference, len(state.lower))
        delta = current.get("delta")
        if delta is None:
            v = c.interval(raw["epsilon_c"])
            delta = c.div(c.mul(sigma, c.sub(u, v)), alpha)
        return state, mu, None, delta, current["D"], None
    v, target = c.interval(raw["epsilon_c"]), c.interval(raw["target"])
    mc = c.div(c.sub(state, c.mul(sigma, v)), alpha)
    delta = c.div(c.mul(sigma, c.sub(u, v)), alpha)
    return state, mu, mc, delta, c.norm(delta), c.norm(c.sub(mc, target))


def _logit_parameters(c, atoms, target_atom, query, alpha, sigma):
    alpha, sigma = c.interval(alpha), c.interval(sigma)
    sigma2 = c.square(sigma)
    beta = c.div(alpha, sigma2)
    gamma = c.div(c.square(alpha), c.mul(c.interval(2.), sigma2))
    offset = c.sub(query, c.mul(alpha, atoms[target_atom]))
    return offset, beta, gamma


def _relative_logit_chunk(c, parameters, geometry, squared_geometry, log_prior):
    offset, beta, gamma = parameters
    likelihood = c.sub(c.mul(beta, c.dot(offset, geometry)), c.mul(gamma, squared_geometry))
    return c.add(likelihood, log_prior)


def _zero_target(values, target_atom):
    logits = _cat(values)
    # The target-relative logit is analytically zero, including shared inputs.
    lo, hi = logits.lower.clone(), logits.upper.clone()
    lo[target_atom], hi[target_atom] = 0., 0.
    return GpuInterval(lo, hi)


def raw_logits(c, atoms, masses, target_atom, query, alpha, sigma, *, chunk=256):
    parameters = _logit_parameters(c, atoms, target_atom, query, alpha, sigma)
    values = []
    for start in range(0, len(atoms.lower), chunk):
        geometry = c.sub(atoms[start:start + chunk], atoms[target_atom])
        squared = c.total(c.square(geometry))
        prior = c.log(c.div(masses[start:start + chunk], masses[target_atom]))
        values.append(_relative_logit_chunk(c, parameters, geometry, squared, prior))
    return _zero_target(values, target_atom)


def raw_segment(c, raw, atoms, masses, target_atom, clean, *, chunk=256,
                current_reference=None):
    """Enclose an endpoint path, optionally reusing its screened current reference.

    Per-target geometry and log mass ratios are computed once per atom chunk
    and shared by both noise levels and the path projections. The scratch
    geometry stays chunk bounded; no cache survives this context or row.
    """
    state, mu, _mc, delta, D, ec = clean
    current = None if current_reference is None else _current_reference(
        c, current_reference, atoms.lower.shape[-1])
    if current is not None:
        D = current["D"]
        delta = current.get("delta", delta)
        ec = None
    A, kappa, g = (c.interval(raw[key]) for key in ("A", "kappa", "guidance"))
    observed = c.interval(raw["saved_endpoint"])
    h = c.mul(c.mul(g, kappa), delta)
    independent = raw.get("independent_innovation")
    if independent is not None or raw["noise_std"] == 0:
        noise = c.interval(independent if independent is not None else torch.zeros_like(state.lower))
        baseline = c.add(c.add(c.mul(A, state), c.mul(kappa, mu)), noise)
    else:
        baseline = c.sub(observed, h)
    saved_h = c.sub(observed, baseline)
    destination = _logit_parameters(c, atoms, target_atom, baseline,
                                    raw["destination_alpha"], raw["destination_sigma"])
    present = None if current is not None else _logit_parameters(
        c, atoms, target_atom, state, raw["alpha"], raw["sigma"])
    beta = destination[1]
    logits, current_logits, slopes, saved_slopes, radii = [], [], [], [], []
    for start in range(0, len(atoms.lower), chunk):
        geometry = c.sub(atoms[start:start + chunk], atoms[target_atom])
        squared = c.total(c.square(geometry))
        prior = c.log(c.div(masses[start:start + chunk], masses[target_atom]))
        logits.append(_relative_logit_chunk(c, destination, geometry, squared, prior))
        if present is not None:
            current_logits.append(_relative_logit_chunk(c, present, geometry, squared, prior))
        slopes.append(c.mul(beta, c.dot(h, geometry)))
        saved_slopes.append(c.mul(beta, c.dot(saved_h, geometry)))
        radii.append(c.sqrt(GpuInterval(squared.lower.clamp_min(0), squared.upper)).upper.amax())
    b = _zero_target(logits, target_atom)
    slopes, saved_slopes = _cat(slopes), _cat(saved_slopes)
    for value in (slopes, saved_slopes):
        value.lower[target_atom], value.upper[target_atom] = 0., 0.
    if current is None:
        current_mean = posterior_mean(c, atoms, c.softmax(_zero_target(current_logits, target_atom)), chunk=chunk)
        eu = c.norm(c.sub(mu, current_mean))
        branch_gap_error = projected_branch_gap_error(c, delta, c.sub(atoms[target_atom], current_mean), D)
    else:
        current_mean, branch_gap_error = current["current_mean"], current["branch_gap_error"]
        eu = None
    residual = c.norm(c.sub(observed, c.add(baseline, h)))
    radius = c.interval(torch.stack(radii).amax())
    spatial_factor = c.mul(beta, radius)
    return {"b": b, "slopes": slopes, "saved_slopes": saved_slopes,
            "current_mean": current_mean, "delta": delta, "D": D, "ec": ec, "eu": eu,
            "branch_gap_error": branch_gap_error,
            "transfer": c.mul(spatial_factor, residual), "beta": beta,
            "spatial_factor": spatial_factor, "prefactor": c.mul(c.mul(beta, g), kappa)}


def enclose_variation(c, atoms, intercept, slopes, current_mean, D, branch_gap_error,
                      *, delta, max_nodes=65, absolute_width=1e-6, current_in_convex_hull=True,
                      current_mass_defect=None, target_atom=0):
    """Enclose the signed directional integral, then take its positive part.

    V = [integral <delta/||delta||, posterior(s)-current_mean> ds]_+.
    Signed leaf contributions are summed BEFORE clipping. Preprojecting atoms
    makes each node a scalar posterior expectation; no node computes a norm.
    The derivative is a covariance bounded by range(projections)*range(slopes)/4.
    All arithmetic is outward CUDA; midpoint selection is only host control.
    """
    if max_nodes < 1 or absolute_width <= 0 or not 0 <= target_atom < len(slopes.lower):
        raise ValueError("Invalid variation enclosure policy")
    delta = c.interval(delta)
    if delta.lower.shape != current_mean.lower.shape or delta.lower.shape != atoms.lower.shape[1:]:
        raise ValueError("Directional variation requires the actual branch-gap vector")
    if not current_in_convex_hull and current_mass_defect is None:
        raise ValueError("Rounded current weights require an explicit mass-defect bound")
    width = c.sub(c.interval(atoms.upper.amax(0)), c.interval(atoms.lower.amin(0)))
    diameter = c.norm(width)
    exact_zero = bool((D.lower == 0) & (D.upper == 0)) and bool(((delta.lower == 0) & (delta.upper == 0)).all())
    if exact_zero:
        relative_slopes = c.sub(slopes, slopes[target_atom])
        if not bool(((relative_slopes.lower == 0) & (relative_slopes.upper == 0)).all()):
            raise ArithmeticError("Zero branch gap has inconsistent nonconstant affine slopes")
        if not bool((branch_gap_error.lower <= 0) & (branch_gap_error.upper >= 0)):
            raise ArithmeticError("Zero branch gap has inconsistent projected error")
        zero = c.interval(0.)
        return {"V": zero, "M": zero, "signed_projected_integral": zero,
                "nodes": 0, "stopping_reason": "exact_zero_branch_gap_convention",
                "direction_status": "exact_zero_gap_no_unit_direction", "lipschitz": zero,
                "diameter": diameter, "integrated_H": zero,
                "error_method": "analytic_zero_direction_convention_no_integrand_division"}
    if not bool((D.lower > 0) & torch.isfinite(D.upper)):
        raise ArithmeticError("Unit branch-gap direction unresolved: norm enclosure contains zero or is nonfinite")
    direction = c.div(delta, D)
    projections = c.dot(c.sub(atoms, current_mean), direction)
    projection_range = GpuInterval(projections.lower.amin(), projections.upper.amax())
    projection_spread = c.sub(c.interval(projection_range.upper), c.interval(projection_range.lower))
    spread = c.sub(c.interval(slopes.upper.amax()), c.interval(slopes.lower.amin()))
    lipschitz = c.div(c.mul(projection_spread, spread), c.interval(4.))
    gain_lipschitz = c.div(c.square(spread), c.interval(4.))
    nodes = 0

    def leaf(left, right):
        nonlocal nodes
        width = c.sub(c.interval(right), c.interval(left))
        midpoint = c.div(c.add(c.interval(left), c.interval(right)), c.interval(2.))
        weights = c.softmax(c.add(intercept, c.mul(midpoint, slopes)))
        f = c.dot(weights, projections)
        midpoint_integral = c.mul(width, f)
        remainder = c.div(c.mul(lipschitz, c.square(width)), c.interval(4.))
        support_range = c.mul(width, projection_range)
        lo = torch.maximum(support_range.lower, c.sub(midpoint_integral, remainder).lower)
        hi = torch.minimum(support_range.upper, c.add(midpoint_integral, remainder).upper)
        if bool(hi < lo):
            raise ArithmeticError("Inconsistent signed node enclosure and posterior support range")
        derivative = c.sub(slopes[target_atom], c.dot(weights, slopes))
        gain_midpoint = c.mul(width, derivative)
        gain_remainder = c.div(c.mul(gain_lipschitz, c.square(width)), c.interval(4.))
        gain_interval = GpuInterval(c.sub(gain_midpoint, gain_remainder).lower,
                                    c.add(gain_midpoint, gain_remainder).upper)
        nodes += 1
        c.evaluated_nodes += 1
        return left, right, GpuInterval(lo, hi), gain_interval

    leaves = [leaf(0., 1.)]
    stop = "node_budget_exhausted"
    while True:
        signed = c.total(_stack([value for _, _, value, _ in leaves]))
        signed = GpuInterval(torch.maximum(signed.lower, projection_range.lower),
                             torch.minimum(signed.upper, projection_range.upper))
        if bool(signed.upper < signed.lower):
            raise ArithmeticError("Inconsistent signed integral and posterior support range")
        integrated_H = c.total(_stack([value for _, _, _, value in leaves]))
        # Monotonicity of max(0, x) preserves the complete signed enclosure.
        V = GpuInterval(signed.lower.clamp_min(0.), signed.upper.clamp_min(0.))
        margin = c.sub(c.sub(D, branch_gap_error), V)
        width = c.sub(c.interval(V.upper), c.interval(V.lower)).upper
        if margin.sign in {"positive", "negative", "zero"}:
            stop = "condition_sign_resolved"
            break
        if bool(width <= absolute_width):
            stop = "requested_variation_width_reached"
            break
        if nodes + 2 > max_nodes:
            break
        ranks = torch.stack([value.upper - value.lower for _, _, value, _ in leaves])
        index = int(ranks.argmax())
        left, right, _, _ = leaves.pop(index)
        midpoint = (left + right) / 2.
        if midpoint in (left, right):
            stop = "binary64_subdivision_resolution_reached"
            break
        leaves.extend((leaf(left, midpoint), leaf(midpoint, right)))
    return {"V": V, "M": margin, "signed_projected_integral": signed,
            "direction_status": "unit_direction_enclosed_from_actual_delta",
            "nodes": nodes, "stopping_reason": stop, "lipschitz": lipschitz,
            "diameter": diameter, "integrated_H": integrated_H,
            "error_method": "outward_cuda_signed_projection_nodes_plus_covariance_Lipschitz_remainder_positive_part_after_integral"}


def source_robust_gain_interval(gain, *, beta_radius, endpoint_radii, justification, arithmetic):
    if (not isinstance(justification, dict) or justification.get("status") != "justified_bound"
            or justification.get("scope") != "endpoint_locations_only_fixed_law_and_coefficients"
            or not justification.get("derivation")):
        raise ValueError("Endpoint perturbation radii need an explicit justified scope and derivation")
    c = arithmetic
    radii = c.interval(endpoint_radii)
    if radii.lower.shape != (2,) or not bool(torch.isfinite(radii.upper).all()) or bool((radii.lower < 0).any()):
        raise ValueError("Endpoint radii must be finite and nonnegative")
    factor = beta_radius if isinstance(beta_radius, GpuInterval) else c.interval(beta_radius)
    if bool(factor.lower < 0):
        raise ValueError("Spatial Lipschitz factor must be nonnegative")
    error = c.mul(factor, c.total(radii))
    return GpuInterval(c.sub(gain, error).lower, c.add(gain, error).upper)
