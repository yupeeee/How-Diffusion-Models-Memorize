"""Pure tensor measurements for the candidate gallery; no cache or model loading."""

from __future__ import annotations

import math

import pandas as pd
import torch

FORMULA_VERSION = "candidate-measurements-1.0"


def _flat(values, target):
    values = torch.as_tensor(values, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64, device=values.device)
    if tuple(values.shape[-target.ndim :]) != tuple(target.shape):
        raise ValueError("Estimate and target latent shapes differ")
    lead = values.shape[: -target.ndim]
    return values.reshape(-1, target.numel()), target.reshape(-1), lead


def basic_step_metrics(
    clean_u,
    clean_c,
    target,
    guidance,
    *,
    center=None,
    current_reference=None,
    source_epsilon=None,
):
    """Branch geometry on one or more saved steps, with arbitrary leading axes.

    Inputs are clean estimates, already obtained from canonical epsilon. IA names
    carry ``initial_`` because only k=0 values enter those registered designs.
    The caller may reduce the returned tensor dictionary at each stored step.
    """
    mu, target, lead = _flat(clean_u, target)
    if torch.as_tensor(clean_u).shape != torch.as_tensor(clean_c).shape:
        raise ValueError("Unconditional and conditional estimate shapes differ")
    mc = torch.as_tensor(clean_c, dtype=torch.float64, device=mu.device).reshape_as(mu)
    g, d = float(guidance), target.numel()
    delta, ru, rc = mc - mu, mu - target, mc - target

    def norm(values):
        return torch.linalg.vector_norm(values, dim=-1)

    D = norm(delta)
    values = {
        "delta_norm": D,
        "paired_error_dot_l2_squared": (ru * rc).sum(-1),
        "paired_error_dot_per_dimension": (ru * rc).sum(-1) / d,
        "branch_gap_rmse": D / math.sqrt(d),
        "conditional_target_error_rmse": norm(rc) / math.sqrt(d),
        "unconditional_target_error_rmse": norm(ru) / math.sqrt(d),
        "conditional_target_error_l2": norm(rc),
        "unconditional_target_error_l2": norm(ru),
        "joint_target_error_rmse": torch.maximum(norm(rc), norm(ru)) / math.sqrt(d),
        "last_prediction_conditional_error_rmse": norm(rc) / math.sqrt(d),
        "last_prediction_residual_guidance_rmse": (g - 1) * D / math.sqrt(d),
        "last_prediction_cross_term_per_dimension": 2
        * (g - 1)
        * (rc * delta).sum(-1)
        / d,
        "last_prediction_error_gap_cosine": (rc * delta).sum(-1)
        / torch.where(norm(rc) * D > 0, norm(rc) * D, torch.nan),
    }
    if current_reference is not None:
        ref = torch.as_tensor(
            current_reference, dtype=torch.float64, device=mu.device
        ).reshape_as(mu)
        eu = mu - ref
        direction = delta / torch.where(D[:, None] > 0, D[:, None], torch.nan)
        values.update(
            current_reference_error_u=norm(eu),
            conditional_error=norm(rc),
            combined_reference_error_norm=norm(rc - eu),
            signed_error_projection=(direction * (eu - rc)).sum(-1),
        )
    for branch, estimate in (("u", mu), ("c", mc), ("g", mu + g * delta)):
        values[f"initial_target_coordinate_{branch}"] = torch.full_like(D, torch.nan)
        values[f"initial_perpendicular_{branch}_rmse"] = torch.full_like(D, torch.nan)
    values.update(
        initial_target_direction_valid=torch.zeros_like(D, dtype=torch.bool),
        a_parallel=torch.full_like(D, torch.nan),
        off_target=torch.full_like(D, torch.nan),
        injection_relative_mismatch=torch.full_like(D, torch.nan),
    )
    status = "unavailable_reference_center"
    if center is not None:
        center = torch.as_tensor(
            center, dtype=torch.float64, device=mu.device
        ).reshape_as(target)
        v = target - center
        vn = v.norm()
        eps = float(
            source_epsilon
            if source_epsilon is not None
            else torch.finfo(torch.float64).eps
        )
        allowance = 8 * eps * max(1.0, float(target.norm()), float(center.norm()))
        valid = bool(torch.isfinite(v).all()) and float(vn) > allowance
        status = (
            "valid"
            if valid
            else "degenerate_or_precision_unresolved_target_center_direction"
        )
        values["initial_target_direction_valid"] = torch.full_like(
            D, valid, dtype=torch.bool
        )
        if valid:
            for branch, estimate in (("u", mu), ("c", mc), ("g", mu + g * delta)):
                offset = estimate - center
                coordinate = (offset * v).sum(-1) / vn.square()
                values[f"initial_target_coordinate_{branch}"] = coordinate
                values[f"initial_perpendicular_{branch}_rmse"] = norm(
                    offset - coordinate[:, None] * v
                ) / math.sqrt(d)
            J = g * delta
            parallel = (J * v).sum(-1) / vn.square()
            values["a_parallel"] = parallel
            values["off_target"] = norm(J - parallel[:, None] * v) / vn
            if g > 0:
                values["injection_relative_mismatch"] = norm(J - g * v) / (g * vn)
    return {k: v.reshape(lead) for k, v in values.items()} | {
        "candidate_basic_status": "valid"
        if bool(torch.isfinite(mu).all() & torch.isfinite(mc).all())
        else "nonfinite_branch_input",
        "initial_target_direction_status": status,
        "candidate_guidance_domain_status": "valid_g_gt_one"
        if math.isfinite(g) and g > 1
        else "not_applicable_manuscript_requires_g_gt_one",
        "initial_injection_relative_mismatch_status": (
            "undefined_nonpositive_or_nonfinite_guidance"
            if not math.isfinite(g) or g <= 0
            else status
        ),
        "candidate_basic_formula_version": FORMULA_VERSION,
    }


def initial_retrieval_metrics(clean_u, clean_c, support, target_id):
    """Euclidean rank = 1 + strictly closer distinct atoms; exact ties include target.

    Stable target-relative distance differences avoid subtracting large query
    norms. Near numerical ties are recomputed with direct float64 distances.
    No candidate prior or posterior score participates in Euclidean retrieval.
    """
    target = support._target(target_id)
    geometry, squared = support.target_geometry(target)
    output = {
        "initial_retrieval_bank_size": support.size,
        "initial_retrieval_tie_policy": "one_plus_strictly_closer; tie_count_includes_target; exact_float64_distances",
        "initial_retrieval_status": "valid",
    }
    for branch, estimates in (("unconditional", clean_u), ("conditional", clean_c)):
        queries, lead = support._queries(estimates)
        rank = torch.empty(len(queries), dtype=torch.int64, device=queries.device)
        ties = torch.empty_like(rank)
        near = torch.empty_like(rank)
        for start in range(0, len(queries), support.query_chunk):
            query = queries[start : start + support.query_chunk]
            offset = query - support.flat[target]
            dot = offset @ geometry.T
            diff = squared[None, :] - 2 * dot
            allowance = (
                32
                * torch.finfo(torch.float64).eps
                * (squared[None, :] + 2 * dot.abs() + 1)
            )
            ambiguous = diff.abs() <= allowance
            # Refinement is bounded to a query chunk and ambiguous atom pairs.
            for qi, ai in ambiguous.nonzero().detach().cpu().tolist():
                diff[qi, ai] = (query[qi] - support.flat[ai]).square().sum() - offset[
                    qi
                ].square().sum()
            rank[start : start + len(query)] = 1 + (diff < 0).sum(-1)
            ties[start : start + len(query)] = (diff == 0).sum(-1)
            near[start : start + len(query)] = ambiguous.sum(-1)
        valid = torch.isfinite(queries).all(-1)
        output[f"initial_{branch}_target_rank"] = torch.where(valid, rank, -1).reshape(
            lead
        )
        output[f"initial_{branch}_target_tie_count"] = torch.where(
            valid, ties, -1
        ).reshape(lead)
        output[f"initial_{branch}_near_tie_count"] = torch.where(
            valid, near, -1
        ).reshape(lead)
        output[f"initial_{branch}_retrieval_valid"] = valid.reshape(lead)
    return output


def candidate_posterior_mean(z, support, alpha, sigma):
    """Fixed-law analytical mean, using bounded query-by-atom matrices."""
    a, s = float(alpha), float(sigma)
    if not (math.isfinite(a) and a >= 0 and math.isfinite(s) and s > 0):
        raise ValueError("Analytical candidate reference requires alpha>=0, sigma>0")
    queries, lead = support._queries(z)
    means = []
    for start in range(0, len(queries), support.query_chunk):
        query = queries[start : start + support.query_chunk]
        logits, _ = support._relative_logits(query, 0, 0, support.size, a, s)
        means.append(torch.softmax(logits, dim=-1) @ support.flat)
    return torch.cat(means).reshape(*lead, *support.latent_shape)


def initial_reference_metrics(x_initial, clean_u_initial, support, alpha, sigma):
    mean = candidate_posterior_mean(x_initial, support, alpha, sigma)
    flat, lead = support._queries(mean)
    clean, _ = support._queries(clean_u_initial)
    if clean.shape != flat.shape:
        raise ValueError(
            "Initial states and unconditional estimates have different sample shapes"
        )
    center = support.weights @ support.flat
    scale = math.sqrt(support.dimension)
    return {
        "initial_candidate_reference_movement_rmse": (
            (flat - center).norm(dim=-1) / scale
        ).reshape(lead),
        "initial_unconditional_candidate_reference_error_rmse": (
            (clean - flat).norm(dim=-1) / scale
        ).reshape(lead),
        "initial_candidate_reference_status": "valid_fixed_candidate_law",
    }


def reference_only_sweep(x_initial, seeds, support, initial_snr):
    """UB04: 49 fixed SNRs and exactly the same input samples at every SNR."""
    if not math.isfinite(float(initial_snr)) or float(initial_snr) <= 0:
        raise ValueError("UB04 requires a finite positive initial SNR")
    query, lead = support._queries(x_initial)
    seeds = list(seeds)
    if len(lead) != 1 or len(seeds) != len(query) or len(set(seeds)) != len(seeds):
        raise ValueError("UB04 requires one initial sample per distinct seed")
    mean = support.weights @ support.flat
    output = []
    for i in range(49):
        r = float(initial_snr) * 10 ** (-i / 8)
        alpha, sigma = math.sqrt(r / (1 + r)), math.sqrt(1 / (1 + r))
        reference = candidate_posterior_mean(
            x_initial, support, alpha, sigma
        ).reshape_as(query)
        distance = (reference - mean).norm(dim=-1) / math.sqrt(support.dimension)
        output.extend(
            dict(
                seed=int(seed),
                snr_grid_index=i,
                analytical_snr=r,
                alpha=alpha,
                sigma=sigma,
                actual_initial_snr=float(initial_snr),
                candidate_reference_movement_rmse=float(value),
                evidence="reference_only_illustration",
            )
            for seed, value in zip(seeds, distance.detach().cpu().tolist(), strict=True)
        )
    return pd.DataFrame(output)


def bank_geometry(support, center):
    center = torch.as_tensor(
        center, dtype=torch.float64, device=support.flat.device
    ).reshape(-1)
    if center.numel() != support.dimension:
        raise ValueError("Reference-center and candidate latent dimensions differ")
    values = (support.flat - center).norm(dim=-1) / math.sqrt(support.dimension)
    return pd.DataFrame(
        {
            "atom_index": list(range(support.size)),
            "atom_id": support.atom_ids,
            "candidate_atom_center_distance_rmse": values.detach().cpu().numpy(),
            "weight": support.weights.detach().cpu().numpy(),
            "evidence": "model_center",
        }
    )
