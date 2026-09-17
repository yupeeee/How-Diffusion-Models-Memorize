"""Endpoint-first candidate-law diagnostics, independent of path integration.

All high-dimensional operations are batched float64 tensor algebra on the
selected support device. Returned segment payloads contain O(batch * atoms)
scalars, never latent trajectories. The candidate law and controls are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .branch_gap import (BRANCH_GAP_ERROR_DEFINITION, BRANCH_GAP_ZERO_CONVENTION,
                         branch_gap_norm, projected_branch_gap_error, unit_branch_gap)
from .feedback import _logits, _posterior_fields

ENDPOINT_VERSION = "candidate-endpoints-projected-gap-error-5"
CONTROL_SALT = "candidate-feedback-specificity-controls-v1"
ENDPOINT_POLICY = {
    "version": ENDPOINT_VERSION,
    "measurement_contract": "projected-gap-error-1",
    "branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
    "branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
    "arithmetic": "float64; stable_logsum_difference_and_softplus_difference",
    "gain_sign": "arithmetic_resolved_log_odds_order; input_sensitivity_separate",
    "direction": "observed_D_and_h_exceed_float64_subtraction_scale_only",
    "direction_input_confidence": "supplied_source_sensitivity_over_D_reported_separately; no_filtering_of_arithmetically_resolved_observations",
    "dose": "sorted_unique_linspace_0_1_21_union_in_range_1_over_g",
    "control_salt": CONTROL_SALT,
    "maximum_controls": 16,
    "flat": "all_competitor_directional_increments_exactly_zero",
    "evidence": "candidate_reference",
}


@dataclass
class EndpointBatch:
    scalars: dict
    dose: dict
    controls: dict
    segment: dict


def fixed_control_atoms(support, target_atom, bank_hash, maximum=16):
    """Choose controls from immutable identities, never measured outcomes."""
    if not isinstance(bank_hash, str) or not bank_hash:
        raise ValueError("A declared candidate bank hash is required")
    if not 0 <= int(target_atom) < support.size or maximum < 0:
        raise ValueError("Invalid control selection request")
    cache = getattr(support, "_candidate_control_cache", {})
    key = (bank_hash, int(target_atom), int(maximum))
    if key in cache:
        return list(cache[key])
    exact_copies = (
        (support.flat == support.flat[target_atom]).all(dim=1).detach().cpu().tolist()
    )
    choices = []
    for index in range(support.size):
        if exact_copies[index]:
            continue
        body = json.dumps(
            [
                CONTROL_SALT,
                bank_hash,
                support.atom_ids[target_atom],
                support.atom_ids[index],
            ],
            separators=(",", ":"),
        )
        choices.append(
            (hashlib.sha256(body.encode()).hexdigest(), support.atom_ids[index], index)
        )
    selected = [entry[2] for entry in sorted(choices)[:maximum]]
    cache[key] = selected
    support._candidate_control_cache = cache
    return list(selected)


def dose_grid(guidance, *, device="cpu"):
    values = [i / 20 for i in range(21)]
    if math.isfinite(float(guidance)) and guidance >= 1:
        values.append(1 / float(guidance))
    return torch.tensor(sorted(set(values)), dtype=torch.float64, device=device)


def stable_logsum_difference(left, right):
    """LSE(left)-LSE(right), retaining small representable log-ratio changes."""
    difference = left - right
    weights = right.softmax(dim=-1)
    small = difference.abs().amax(dim=-1) <= 0.5
    # The small branch avoids subtracting two nearly equal, large log sums.
    increment = (weights * torch.expm1(difference.clamp(-0.5, 0.5))).sum(dim=-1)
    close = torch.log1p(increment)
    general = torch.logsumexp(right.log_softmax(dim=-1) + difference, dim=-1)
    return torch.where(small, close, general)


def stable_log_probability_gain(odds0, odds1, odds_gain=None):
    """Difference of log sigmoids without cancellation near saturation."""
    difference = odds1 - odds0 if odds_gain is None else odds_gain
    low = torch.minimum(odds0, odds1)
    high = torch.maximum(odds0, odds1)
    distance = difference.abs()
    # Positive-half formula is also safe for crossings no farther than -700.
    numerator = torch.exp((-low).clamp(max=700)) * (-torch.expm1(-distance))
    positive_half = torch.log1p(numerator / (1 + torch.exp((-high).clamp(max=700))))
    negative_half = distance - torch.log1p(
        torch.exp(high.clamp(max=0))
        * (-torch.expm1(-distance))
        / (1 + torch.exp(low.clamp(max=0)))
    )
    crossing = torch.nn.functional.softplus(-low) - torch.nn.functional.softplus(-high)
    magnitude = torch.where(
        high <= 0, negative_half, torch.where(low >= -700, positive_half, crossing)
    )
    return torch.sign(difference) * magnitude


def _sign(values, tolerance, *, exact_zero=None):
    v, t = values.detach().cpu().tolist(), tolerance.detach().cpu().tolist()
    zeros = (
        [False] * len(v) if exact_zero is None else exact_zero.detach().cpu().tolist()
    )
    return [
        (
            "arithmetic_zero"
            if z
            else "positive"
            if x > e
            else "negative"
            if x < -e
            else "numerically_unresolved"
        )
        if math.isfinite(x) and math.isfinite(e)
        else "not_applicable"
        for x, e, z in zip(v, t, zeros)
    ]


_FLOAT_FIELDS = (
    "delta_norm_l2",
    "displacement_norm_l2",
    "current_reference_error_u_l2",
    "conditional_error_l2",
    "reference_branch_gap_l2",
    "branch_gap_error_l2",
    "branch_gap_error_norm_diagnostic_l2",
    "combined_reference_error_l2",
    "signed_error_projection_l2",
    "current_alignment_cosine",
    "next_reference_matched_target_error_rmse",
    "next_reference_guided_target_error_rmse",
    "next_reference_target_contraction_rmse",
    "log_probability_gain",
    "log_odds_gain",
    "normalized_log_probability_gain",
    "normalized_log_odds_gain",
    "gain_arithmetic_tolerance",
    "gain_input_sensitivity",
    "source_sensitivity_l2",
    "direction_tolerance_l2",
    "displacement_arithmetic_tolerance_l2",
    "direction_input_sensitivity_ratio",
    "endpoint_slope_C0",
    "endpoint_slope_C1",
    "endpoint_curvature_0",
    "endpoint_curvature_1",
    "endpoint_bracket_lower_residual",
    "endpoint_bracket_upper_residual",
    "endpoint_bracket_residual",
    "endpoint_arithmetic_tolerance",
    "affine_log_odds_gain",
    "affine_log_probability_gain",
    "affine_endpoint_gain_residual",
    "segment_displacement_residual_l2",
    "control_gain_median",
    "specificity_contrast",
) + tuple(
    f"{endpoint}_{field}"
    for endpoint in ("matched", "guided")
    for field in ("log_probability", "log_complement", "log_odds")
)
_STATUS_FIELDS = (
    "feedback_status",
    "gain_status",
    "gain_magnitude_status",
    "profile_class",
    "direction_normalization_status",
    "direction_input_precision_status",
    "current_alignment_status",
    "endpoint_bracket_status",
    "source_precision_status",
    "specificity_status",
)


def unavailable_endpoint(count, *, device="cpu", reason="not_applicable", guidance=1.0):
    scalars = {
        "candidate_" + key: torch.full(
            (count,), torch.nan, dtype=torch.float64, device=device
        )
        for key in _FLOAT_FIELDS
    }
    scalars.update(
        {"candidate_" + key: ["not_applicable"] * count for key in _STATUS_FIELDS}
    )
    scalars["candidate_feedback_status"] = [reason] * count
    scalars["candidate_measurement_contract"] = ["projected-gap-error-1"] * count
    scalars["candidate_branch_gap_error_definition"] = [BRANCH_GAP_ERROR_DEFINITION] * count
    scalars["candidate_branch_gap_error_zero_convention"] = [BRANCH_GAP_ZERO_CONVENTION] * count
    for key in ("feedback_eligible", "gain_saturated", "gain_underflow"):
        scalars["candidate_" + key] = torch.zeros(
            count, dtype=torch.bool, device=device
        )
    grid = dose_grid(guidance, device=device)
    dose = {"lambda": grid}
    for key in (
        "log_probability_gain",
        "log_odds_gain",
        "log_probability",
        "log_complement",
        "log_odds",
        "slope",
        "curvature",
    ):
        dose["candidate_dose_" + key] = torch.full(
            (count, len(grid)), torch.nan, dtype=torch.float64, device=device
        )
    return EndpointBatch(
        scalars,
        dose,
        {
            "control_atom_indices": [],
            "control_atom_ids": [],
            "candidate_control_log_probability_gain": torch.empty(
                (count, 0), dtype=torch.float64, device=device
            ),
            "candidate_control_log_odds_gain": torch.empty(
                (count, 0), dtype=torch.float64, device=device
            ),
        },
        {},
    )


def endpoint_metrics(
    support,
    state,
    mc,
    mu,
    matched,
    guided,
    target_id,
    *,
    current_alpha,
    current_sigma,
    destination_alpha,
    destination_sigma,
    guidance,
    kappa,
    bank_hash,
    source_error_l2=0.0,
):
    """Measure direct endpoints/dose/controls and return integration sufficient data.

    Caller verifies the actual affine scheduler and nonterminal destination.
    ``source_error_l2`` is a declared input-sensitivity envelope, not a theorem
    error and not an arithmetic error estimate. Its status remains separate.
    """
    arrays = [support._queries(x)[0] for x in (state, mc, mu, matched, guided)]
    z, mc, mu, cf, observed = arrays
    if any(x.shape != z.shape for x in arrays):
        raise ValueError("Endpoint query batches must have identical latent shapes")
    n, device = len(z), z.device
    a, s, an, sn, g, kap = map(
        float,
        (
            current_alpha,
            current_sigma,
            destination_alpha,
            destination_sigma,
            guidance,
            kappa,
        ),
    )
    result = unavailable_endpoint(n, device=device, guidance=g)
    source = torch.as_tensor(
        source_error_l2, dtype=torch.float64, device=device
    ).expand(n)
    if not bool(torch.isfinite(source).all() & (source >= 0).all()):
        raise ValueError("Source-error sensitivity must be finite and nonnegative")
    if str(target_id) not in support.aliases:
        result.scalars["candidate_feedback_status"] = ["missing_candidate_target"] * n
        return result
    if support.size < 2:
        result.scalars["candidate_feedback_status"] = ["no_distinct_competitor"] * n
        return result
    if (
        not all(math.isfinite(x) and x > 0 for x in (a, s, an, sn, kap))
        or not math.isfinite(g)
        or g < 0
    ):
        result.scalars["candidate_feedback_status"] = [
            "invalid_or_nonpositive_noise_guidance_kappa"
        ] * n
        return result
    valid = torch.stack([torch.isfinite(x).all(dim=1) for x in arrays]).all(dim=0)
    if not bool(valid.all()):
        # Each input row remains in the scalar population even when unavailable.
        result.scalars["candidate_feedback_status"] = ["invalid_nonfinite_input"] * n
        if not bool(valid.any()):
            return result
        ids = valid.nonzero().flatten()
        part = endpoint_metrics(
            support,
            *(x[ids].reshape(-1, *support.latent_shape) for x in arrays),
            target_id,
            current_alpha=a,
            current_sigma=s,
            destination_alpha=an,
            destination_sigma=sn,
            guidance=g,
            kappa=kap,
            bank_hash=bank_hash,
            source_error_l2=source[ids],
        )
        for key, value in part.scalars.items():
            if isinstance(value, torch.Tensor):
                result.scalars[key][ids] = value
            else:
                for index, item in zip(ids.tolist(), value):
                    result.scalars[key][index] = item
        for key, value in part.dose.items():
            if key != "lambda":
                result.dose[key][ids] = value
        result.controls = {
            key: value
            for key, value in part.controls.items()
            if not isinstance(value, torch.Tensor)
        }
        for key, value in part.controls.items():
            if isinstance(value, torch.Tensor):
                result.controls[key] = torch.full(
                    (n, *value.shape[1:]), torch.nan, dtype=value.dtype, device=device
                )
                result.controls[key][ids] = value
        result.segment = dict(part.segment)
        result.segment["row_indices"] = ids
        result.segment["population_count"] = n
        return result
    target = support.aliases[str(target_id)]
    controls = fixed_control_atoms(support, target, bank_hash)
    if not controls:
        result.scalars["candidate_feedback_status"] = ["no_distinct_competitor"] * n
        return result
    delta = mc - mu
    D = branch_gap_norm(delta)
    shift = g * kap * delta
    hnorm = shift.norm(dim=1)
    eps = torch.finfo(torch.float64).eps
    # Resolution of the measured displacement is an arithmetic question. A
    # conservative envelope for unknown pre-storage inputs is logged separately;
    # it must not erase well-defined observations of the stored branch vectors.
    direction_tol = 128 * eps * (mc.norm(dim=1) + mu.norm(dim=1))
    displacement_tol = 128 * eps * (observed.norm(dim=1) + cf.norm(dim=1))
    direction_ok = (D > direction_tol) & (hnorm > displacement_tol)
    input_sensitivity_ratio = torch.where(
        D > 0, source / torch.where(D > 0, D, torch.ones_like(D)), torch.nan
    )
    zero_shift = (shift == 0).all(dim=1)
    direction = unit_branch_gap(delta)
    beta = an / sn**2
    differences = support.target_geometry(target)[0]
    current_logits = _logits(support, z, target, a, s)
    current_weights = current_logits.softmax(dim=1)
    current_reference = current_weights @ support.flat
    rc, ru = mc - support.flat[target], mu - current_reference
    ec, eu = rc.norm(dim=1), ru.norm(dim=1)
    combined = projected_branch_gap_error(delta, support.flat[target] - current_reference)
    error_norm_diagnostic = branch_gap_norm(delta - (support.flat[target] - current_reference))
    S = -combined
    target_direction = support.flat[target] - current_reference
    target_distance = target_direction.norm(dim=1)
    alignment_ok = direction_ok & (
        target_distance
        > 128 * eps * (support.flat[target].norm() + current_reference.norm(dim=1))
    )
    alignment = (direction * target_direction).sum(dim=1) / torch.where(
        target_distance > 0, target_distance, torch.ones_like(target_distance)
    )
    intercept = _logits(support, cf, target, an, sn)
    slopes = beta * (shift @ differences.T)
    endpoint_logits = _logits(support, observed, target, an, sn)
    competitor = torch.arange(support.size, device=device) != target
    b, q = intercept[:, competitor], -slopes[:, competitor]
    end_competitor = endpoint_logits[:, competitor]
    p0, p1 = (
        _posterior_fields(intercept, target),
        _posterior_fields(endpoint_logits, target),
    )
    G = -stable_logsum_difference(end_competitor, b)
    H = stable_log_probability_gain(p0["log_odds"], p1["log_odds"], G)
    affine_odds1 = -torch.logsumexp(b - q, dim=1)
    affine_G = -stable_logsum_difference(b - q, b)
    affine_H = stable_log_probability_gain(p0["log_odds"], affine_odds1, affine_G)
    arithmetic_tol = (
        128
        * eps
        * math.log2(support.size + 1)
        * torch.maximum(
            torch.ones_like(G),
            torch.maximum(
                intercept.abs().amax(dim=1), endpoint_logits.abs().amax(dim=1)
            ),
        )
    )
    p_comp0, p_comp1 = b.softmax(dim=1), (b - q).softmax(dim=1)
    C0, C1 = (p_comp0 * q).sum(dim=1), (p_comp1 * q).sum(dim=1)
    curvature0 = -(p_comp0 * (q - C0[:, None]).square()).sum(dim=1)
    curvature1 = -(p_comp1 * (q - C1[:, None]).square()).sum(dim=1)
    slope_tol = arithmetic_tol + 128 * eps * q.abs().amax(dim=1)
    constant = (q == 0).all(dim=1)
    bracket_lower, bracket_upper = G - C1, C0 - G
    bracket_bad = (torch.minimum(bracket_lower, bracket_upper) < -slope_tol) | (
        C1 - C0 > slope_tol
    )
    displacement_residual = (observed - cf - shift).norm(dim=1)
    reconstruction_tol = 128 * eps * (observed.norm(dim=1) + cf.norm(dim=1) + hnorm)
    reconstruction_bad = displacement_residual > reconstruction_tol
    # Zero displacement means exactly equal matched endpoints, not an artificial
    # direction. A nonidentical endpoint supplied by a caller is a QA failure.
    structural_zero = zero_shift & (observed == cf).all(dim=1)
    H, G = torch.where(structural_zero, 0.0, H), torch.where(structural_zero, 0.0, G)
    denominator = beta * hnorm * math.sqrt(support.dimension)
    safe_denominator = torch.where(
        direction_ok, denominator, torch.ones_like(denominator)
    )
    mean0 = intercept.softmax(dim=1) @ support.flat
    mean1 = endpoint_logits.softmax(dim=1) @ support.flat
    distance0 = (mean0 - support.flat[target]).norm(dim=1) / math.sqrt(
        support.dimension
    )
    distance1 = (mean1 - support.flat[target]).norm(dim=1) / math.sqrt(
        support.dimension
    )
    # The same law and the same endpoints are reused for every fixed control.
    other_indices = torch.tensor(
        [
            [index for index in range(support.size) if index != control]
            for control in controls
        ],
        dtype=torch.long,
        device=device,
    )
    cb0 = intercept[:, other_indices] - intercept[:, controls, None]
    cb1 = endpoint_logits[:, other_indices] - endpoint_logits[:, controls, None]
    control_odds0, control_odds1 = (
        -torch.logsumexp(cb0, dim=2),
        -torch.logsumexp(cb1, dim=2),
    )
    control_G = -stable_logsum_difference(cb1, cb0)
    control_H = stable_log_probability_gain(control_odds0, control_odds1, control_G)
    control_median = torch.quantile(control_H, 0.5, dim=1)
    raw = {
        "delta_norm_l2": D,
        "displacement_norm_l2": hnorm,
        "current_reference_error_u_l2": eu,
        "conditional_error_l2": ec,
        "reference_branch_gap_l2": target_distance,
        "branch_gap_error_l2": combined,
        "branch_gap_error_norm_diagnostic_l2": error_norm_diagnostic,
        "combined_reference_error_l2": combined,
        "signed_error_projection_l2": S,
        "current_alignment_cosine": torch.where(alignment_ok, alignment, torch.nan),
        "next_reference_matched_target_error_rmse": distance0,
        "next_reference_guided_target_error_rmse": distance1,
        "next_reference_target_contraction_rmse": distance0 - distance1,
        "log_probability_gain": H,
        "log_odds_gain": G,
        "normalized_log_probability_gain": torch.where(
            direction_ok, H / safe_denominator, torch.nan
        ),
        "normalized_log_odds_gain": torch.where(
            direction_ok, G / safe_denominator, torch.nan
        ),
        "gain_arithmetic_tolerance": arithmetic_tol,
        "gain_input_sensitivity": beta * differences.norm(dim=1).max() * source,
        "source_sensitivity_l2": source,
        "direction_tolerance_l2": direction_tol,
        "displacement_arithmetic_tolerance_l2": displacement_tol,
        "direction_input_sensitivity_ratio": input_sensitivity_ratio,
        "endpoint_slope_C0": C0,
        "endpoint_slope_C1": C1,
        "endpoint_curvature_0": curvature0,
        "endpoint_curvature_1": curvature1,
        "endpoint_bracket_lower_residual": bracket_lower,
        "endpoint_bracket_upper_residual": bracket_upper,
        "endpoint_bracket_residual": torch.minimum(bracket_lower, bracket_upper),
        "endpoint_arithmetic_tolerance": slope_tol,
        "affine_log_odds_gain": affine_G,
        "affine_log_probability_gain": affine_H,
        "affine_endpoint_gain_residual": H - affine_H,
        "segment_displacement_residual_l2": displacement_residual,
        "control_gain_median": control_median,
        "specificity_contrast": H - control_median,
    }
    scalars = result.scalars
    scalars.update({"candidate_" + key: value for key, value in raw.items()})
    for name, fields in (("matched", p0), ("guided", p1)):
        scalars.update(
            {f"candidate_{name}_{key}": value for key, value in fields.items()}
        )
    probability0, probability1 = (
        p0["log_probability"].exp(),
        p1["log_probability"].exp(),
    )
    saturated = (
        (probability0 == 1)
        | (probability1 == 1)
        | (probability0 == 0)
        | (probability1 == 0)
    )
    underflow = (H == 0) & (G.abs() > arithmetic_tol)
    # One compact status transfer per query batch, matching the shared reducer's
    # packed scalar convention. High-dimensional tensors remain on the device.
    decisions = (
        torch.stack(
            (
                G,
                arithmetic_tol,
                structural_zero
                | (constant & (endpoint_logits == intercept).all(dim=1)),
                C0,
                C1,
                slope_tol,
                constant,
                bracket_bad,
                reconstruction_bad,
                direction_ok,
                zero_shift,
                alignment_ok,
                underflow,
                D > 0,
                source < D,
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .tolist()
    )
    signs, profiles, status, magnitude_status = [], [], [], []
    normalization_status, alignment_status, bracket_status = [], [], []
    direction_input_status = []
    for (
        gain,
        tol,
        exact_zero,
        c0,
        c1,
        ct,
        isconstant,
        badbracket,
        badreconstruction,
        direction_valid,
        zero,
        aligned,
        tiny,
        nonzero_gap,
        source_below_gap,
    ) in decisions:
        sign = (
            "not_applicable"
            if not math.isfinite(gain)
            else "arithmetic_zero"
            if exact_zero
            else "positive"
            if gain > tol
            else "negative"
            if gain < -tol
            else "numerically_unresolved"
        )
        if badreconstruction:
            sign = "numerically_unresolved"
        signs.append(sign)
        status.append(
            "matched_endpoint_available"
            if not badreconstruction
            else "matched_displacement_QA_failure"
        )
        profiles.append(
            "numerically_unresolved"
            if badbracket or badreconstruction
            else "flat"
            if isconstant
            else "increasing"
            if c1 > ct
            else "decreasing"
            if c0 < -ct
            else "interior_peak"
            if c0 > ct and c1 < -ct
            else "numerically_unresolved"
        )
        magnitude_status.append(
            "underflow"
            if tiny
            else "represented"
            if sign not in {"numerically_unresolved", "not_applicable"}
            else sign
        )
        normalization_status.append(
            "resolved"
            if direction_valid
            else "zero_displacement"
            if zero
            else "arithmetic_unresolved"
        )
        alignment_status.append(
            "resolved" if aligned else "undefined_or_arithmetic_unresolved"
        )
        direction_input_status.append(
            "undefined_zero_branch_gap"
            if not nonzero_gap
            else "source_sensitivity_below_observed_gap"
            if source_below_gap
            else "source_sensitivity_at_or_above_observed_gap"
        )
        bracket_status.append(
            "numerical_inconsistency"
            if badbracket
            else "consistent_with_arithmetic_estimate"
        )
    scalars.update(
        {
            "candidate_feedback_eligible": torch.ones(
                n, dtype=torch.bool, device=device
            ),
            "candidate_feedback_status": status,
            "candidate_gain_status": signs,
            "candidate_gain_saturated": saturated,
            "candidate_gain_underflow": underflow,
            "candidate_gain_magnitude_status": magnitude_status,
            "candidate_profile_class": profiles,
            "candidate_direction_normalization_status": normalization_status,
            "candidate_direction_input_precision_status": direction_input_status,
            "candidate_current_alignment_status": alignment_status,
            "candidate_endpoint_bracket_status": bracket_status,
            "candidate_source_precision_status": [
                "supplied_input_sensitivity_not_certified"
            ]
            * n,
            "candidate_specificity_status": ["fixed_mismatched_controls_available"] * n,
        }
    )
    grid = result.dose["lambda"]
    # Dose uses affine ratios: no new latent dot products or network predictions.
    ratio = b[:, None, :] - grid[None, :, None] * q[:, None, :]
    odds = -torch.logsumexp(ratio, dim=2)
    doseG = -stable_logsum_difference(ratio, b[:, None, :].expand_as(ratio))
    doseH = stable_log_probability_gain(p0["log_odds"][:, None], odds, doseG)
    weights = ratio.softmax(dim=2)
    derivative = (weights * q[:, None, :]).sum(dim=2)
    curvature = -(weights * (q[:, None, :] - derivative[:, :, None]).square()).sum(
        dim=2
    )
    result.dose.update(
        {
            "candidate_dose_log_probability_gain": doseH,
            "candidate_dose_log_odds_gain": doseG,
            "candidate_dose_log_probability": -torch.nn.functional.softplus(-odds),
            "candidate_dose_log_complement": -torch.nn.functional.softplus(odds),
            "candidate_dose_log_odds": odds,
            "candidate_dose_slope": derivative,
            "candidate_dose_curvature": curvature,
        }
    )
    result.controls = {
        "control_atom_indices": controls,
        "control_atom_ids": [support.atom_ids[x] for x in controls],
        "candidate_control_log_probability_gain": control_H,
        "candidate_control_log_odds_gain": control_G,
    }
    result.segment = {
        "schema_version": 1,
        "bank_hash": bank_hash,
        "target_atom": target,
        "dimension": support.dimension,
        "beta": beta,
        "guidance": g,
        "kappa": kap,
        "population_count": n,
        "row_indices": torch.arange(n, device=device),
        "intercept": intercept,
        "slopes": slopes,
        "current_weights": current_weights,
        "delta_norm": D,
        "delta_exact_zero": (delta == 0).all(dim=1),
        "conditional_error": ec,
        "unconditional_reference_error": eu,
        "reference_branch_gap": target_distance,
        "branch_gap_error": combined,
        "branch_gap_error_norm_diagnostic": error_norm_diagnostic,
        "branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
        "branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
        "measurement_contract": "projected-gap-error-1",
        "combined_reference_error": combined,
        "signed_error_projection": S,
        "source_error_l2": source,
        "direction_resolved": direction_ok,
        "direct_log_probability_gain": H,
        "gain_arithmetic_tolerance": arithmetic_tol,
        "reconstruction_valid": ~reconstruction_bad,
    }
    return result
