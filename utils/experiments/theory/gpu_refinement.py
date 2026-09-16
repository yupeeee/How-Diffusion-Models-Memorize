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


def negative_condition_precheck(D, ec, eu=None, *, arithmetic):
    c = arithmetic
    upper = c.sub(c.sub(D, ec), c.interval(0.) if eu is None else eu).upper
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


def clean_intervals(c, raw):
    state, u, v = (c.interval(raw[key]) for key in ("state", "epsilon_u", "epsilon_c"))
    target = c.interval(raw["target"])
    alpha, sigma = c.interval(raw["alpha"]), c.interval(raw["sigma"])
    mu, mc = (c.div(c.sub(state, c.mul(sigma, epsilon)), alpha) for epsilon in (u, v))
    delta = c.div(c.mul(sigma, c.sub(u, v)), alpha)
    return state, mu, mc, delta, c.norm(delta), c.norm(c.sub(mc, target))


def raw_logits(c, atoms, masses, target_atom, query, alpha, sigma, *, chunk=256):
    alpha, sigma = c.interval(alpha), c.interval(sigma)
    sigma2 = c.square(sigma)
    beta = c.div(alpha, sigma2)
    gamma = c.div(c.square(alpha), c.mul(c.interval(2.), sigma2))
    target = atoms[target_atom]
    offset = c.sub(query, c.mul(alpha, target))
    values = []
    for start in range(0, len(atoms.lower), chunk):
        geometry = c.sub(atoms[start:start + chunk], target)
        likelihood = c.sub(c.mul(beta, c.dot(offset, geometry)), c.mul(gamma, c.total(c.square(geometry))))
        values.append(c.add(likelihood, c.log(c.div(masses[start:start + chunk], masses[target_atom]))))
    logits = _cat(values)
    # The target-relative logit is analytically zero, including shared inputs.
    lo, hi = logits.lower.clone(), logits.upper.clone()
    lo[target_atom], hi[target_atom] = 0., 0.
    return GpuInterval(lo, hi)


def raw_segment(c, raw, atoms, masses, target_atom, clean, *, chunk=256):
    state, mu, _mc, delta, D, ec = clean
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
    b = raw_logits(c, atoms, masses, target_atom, baseline, raw["destination_alpha"], raw["destination_sigma"], chunk=chunk)
    beta = c.div(c.interval(raw["destination_alpha"]), c.square(c.interval(raw["destination_sigma"])))
    slopes, saved_slopes, radii = [], [], []
    for start in range(0, len(atoms.lower), chunk):
        geometry = c.sub(atoms[start:start + chunk], atoms[target_atom])
        slopes.append(c.mul(beta, c.dot(h, geometry)))
        saved_slopes.append(c.mul(beta, c.dot(saved_h, geometry)))
        radii.append(c.norm(geometry).upper.amax())
    slopes, saved_slopes = _cat(slopes), _cat(saved_slopes)
    for value in (slopes, saved_slopes):
        value.lower[target_atom], value.upper[target_atom] = 0., 0.
    current_logits = raw_logits(c, atoms, masses, target_atom, state, raw["alpha"], raw["sigma"], chunk=chunk)
    current_mean = posterior_mean(c, atoms, c.softmax(current_logits), chunk=chunk)
    eu = c.norm(c.sub(mu, current_mean))
    residual = c.norm(c.sub(observed, c.add(baseline, h)))
    radius = c.interval(torch.stack(radii).amax())
    spatial_factor = c.mul(beta, radius)
    return {"b": b, "slopes": slopes, "saved_slopes": saved_slopes,
            "current_mean": current_mean, "D": D, "ec": ec, "eu": eu,
            "transfer": c.mul(spatial_factor, residual), "beta": beta,
            "spatial_factor": spatial_factor, "prefactor": c.mul(c.mul(beta, g), kappa)}


def enclose_variation(c, atoms, intercept, slopes, current_mean, D, ec, eu,
                      *, max_nodes=65, absolute_width=1e-6, current_in_convex_hull=True,
                      current_mass_defect=None, target_atom=0):
    """Adaptive midpoint integral with enclosed nodes and Lipschitz remainder.

    Node selection is host control. All support reductions, quadrature bounds,
    and margin arithmetic execute on the CUDA worker, including node batches.
    """
    if max_nodes < 1 or absolute_width <= 0 or not 0 <= target_atom < len(slopes.lower):
        raise ValueError("Invalid variation enclosure policy")
    width = c.sub(c.interval(atoms.upper.amax(0)), c.interval(atoms.lower.amin(0)))
    diameter = c.norm(width)
    spread = c.sub(c.interval(slopes.upper.amax()), c.interval(slopes.lower.amin()))
    lipschitz = c.div(c.mul(diameter, spread), c.interval(4.))
    cap = diameter.upper
    if not current_in_convex_hull:
        if current_mass_defect is None:
            raise ValueError("Rounded current weights require an explicit mass-defect bound")
        maximum_norm = c.interval(c.norm(atoms).upper.amax())
        cap = c.add(diameter, c.mul(current_mass_defect, maximum_norm)).upper
    gain_lipschitz = c.div(c.square(spread), c.interval(4.))
    nodes = 0

    def leaf(left, right):
        nonlocal nodes
        # Dyadic subdivisions are exactly representable; midpoint/width bounds
        # are still evaluated through outward device arithmetic.
        width = c.sub(c.interval(right), c.interval(left))
        midpoint = c.div(c.add(c.interval(left), c.interval(right)), c.interval(2.))
        weights = c.softmax(c.add(intercept, c.mul(midpoint, slopes)))
        mean = posterior_mean(c, atoms, weights)
        f = c.norm(c.sub(mean, current_mean))
        midpoint_integral = c.mul(width, f)
        remainder = c.div(c.mul(lipschitz, c.square(width)), c.interval(4.))
        derivative = c.sub(slopes[target_atom], c.dot(weights, slopes))
        gain_midpoint = c.mul(width, derivative)
        gain_remainder = c.div(c.mul(gain_lipschitz, c.square(width)), c.interval(4.))
        gain_interval = GpuInterval(c.sub(gain_midpoint, gain_remainder).lower,
                                    c.add(gain_midpoint, gain_remainder).upper)
        lo = c.sub(midpoint_integral, remainder).lower.clamp_min(0.)
        hi = torch.minimum(c.mul(width, c.interval(cap)).upper,
                           c.add(midpoint_integral, remainder).upper)
        if bool(hi < lo):
            raise ArithmeticError("Inconsistent node enclosure and convex-hull cap")
        nodes += 1
        c.evaluated_nodes += 1
        return left, right, GpuInterval(lo, hi), gain_interval

    leaves = [leaf(0., 1.)]
    stop = "node_budget_exhausted"
    while True:
        V = c.total(_stack([value for _, _, value, _ in leaves]))
        integrated_H = c.total(_stack([value for _, _, _, value in leaves]))
        V = GpuInterval(V.lower.clamp_min(0.), torch.minimum(cap, V.upper))
        margin = c.sub(c.sub(c.sub(D, ec), eu), V)
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
    return {"V": V, "M": margin, "nodes": nodes, "stopping_reason": stop,
            "lipschitz": lipschitz, "diameter": diameter, "integrated_H": integrated_H,
            "error_method": "outward_cuda_binary64_nodes_plus_global_lipschitz_midpoint_remainder"}


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
