"""Separately resumable directional refinements from compact endpoint payloads.

This module has no trajectory/cache reader. It integrates a saved affine
candidate-logit segment using the same declared bank, never learned predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .feedback import _X, _WG, _WK, mean_distance_from_weights


@dataclass(frozen=True)
class IntegrationConfig:
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-6
    identity_absolute_tolerance: float = 1e-8
    max_evaluations: int = 8190
    initial_intervals: int = 2
    node_chunk: int = 256
    use_gram: bool = True

    def __post_init__(self):
        if (
            any(
                not math.isfinite(x) or x <= 0
                for x in (
                    self.absolute_tolerance,
                    self.relative_tolerance,
                    self.identity_absolute_tolerance,
                )
            )
            or self.max_evaluations < 15
            or self.initial_intervals < 1
            or self.node_chunk < 1
        ):
            raise ValueError("Invalid candidate-refinement integration policy")


DEFAULT_INTEGRATION = IntegrationConfig()
WIDTHS = (0.0, -2.0, 2.0, -8.0, 8.0, -16.0, 16.0, -32.0, 32.0, -64.0, 64.0)
INTEGRATION_POLICY = {
    "version": "candidate-directional-integrals-1",
    **asdict(DEFAULT_INTEGRATION),
    "method": "adaptive_Gauss_Kronrod_15_7_joint_V_Vparallel_W_H",
    "partition_widths": list(WIDTHS),
    "boundary_tails": "include_upper_envelope_crossings_outside_segment_when_width_neighborhood_overlaps",
    "error_scope": "embedded_estimate_plus_logged_roundoff_and_source_sensitivity_not_certified",
    "endpoint_independence": "direct_endpoint_H_is_saved_before_and_independent_of_this_integration",
}
MARGINS = (
    "original",
    "combined",
    "signed_error",
    "projected_variation",
    "exact_directional_average",
)
VARIATIONS = ("norm", "abs_projected", "signed")


def wide_partition(intercept, slope, config):
    """Predeclared 64-width tail coverage, including near-boundary transitions."""
    order = sorted(
        range(len(slope)), key=lambda index: (slope[index], intercept[index])
    )
    hull, starts = [], []
    for index in order:
        if hull and slope[index] == slope[hull[-1]]:
            hull.pop()
            starts.pop()
        crossing = -math.inf
        while hull:
            old = hull[-1]
            crossing = (intercept[old] - intercept[index]) / (slope[index] - slope[old])
            if crossing > starts[-1]:
                break
            hull.pop()
            starts.pop()
        hull.append(index)
        starts.append(crossing if len(hull) > 1 else -math.inf)
    points = set(np.linspace(0.0, 1.0, config.initial_intervals + 1).tolist())
    for position in range(1, len(hull)):
        crossing = starts[position]
        steepness = abs(slope[hull[position]] - slope[hull[position - 1]])
        if not math.isfinite(crossing) or steepness == 0:
            continue
        for width in WIDTHS:
            value = crossing + width / steepness
            if 0 < value < 1:
                points.add(float(value))
    points = sorted(points)
    maximum = max(1, config.max_evaluations // 15)
    truncated = len(points) - 1 > maximum
    if truncated:
        points = [
            points[index]
            for index in np.linspace(0, len(points) - 1, maximum + 1).astype(int)
        ]
    return list(zip(points[:-1], points[1:])), truncated


class _Segment:
    def __init__(self, support, payload, config):
        self.support, self.config = support, config
        self.device = support.flat.device
        self.intercept = torch.as_tensor(
            payload["intercept"], dtype=torch.float64, device=self.device
        )
        self.slopes = torch.as_tensor(
            payload["slopes"], dtype=torch.float64, device=self.device
        )
        self.current_weights = torch.as_tensor(
            payload["current_weights"], dtype=torch.float64, device=self.device
        )
        self.D = torch.as_tensor(
            payload["delta_norm"], dtype=torch.float64, device=self.device
        )
        factor = payload["beta"] * payload["guidance"] * payload["kappa"] * self.D
        self.projected_atoms = (
            self.slopes
            / torch.where(factor > 0, factor, torch.ones_like(factor))[:, None]
        )
        self.x = torch.as_tensor(_X, dtype=torch.float64, device=self.device)
        self.wk = torch.as_tensor(_WK, dtype=torch.float64, device=self.device)
        self.wg = torch.as_tensor(_WG, dtype=torch.float64, device=self.device)
        self.roundoff = torch.zeros(
            len(self.D), dtype=torch.float64, device=self.device
        )
        self.invalid = torch.zeros(len(self.D), dtype=torch.int64, device=self.device)

    def intervals(self, intervals):
        query = torch.tensor(
            [x[0] for x in intervals], dtype=torch.long, device=self.device
        )
        left = torch.tensor(
            [x[1] for x in intervals], dtype=torch.float64, device=self.device
        )
        right = torch.tensor(
            [x[2] for x in intervals], dtype=torch.float64, device=self.device
        )
        half = (right - left) / 2
        positions = (
            (left[:, None] + right[:, None]) / 2 + half[:, None] * self.x
        ).flatten()
        queries = query.repeat_interleave(15)
        chunks = []
        for start in range(0, len(positions), self.config.node_chunk):
            ids = queries[start : start + self.config.node_chunk]
            coordinate = positions[start : start + self.config.node_chunk]
            slopes = self.slopes[ids]
            weights = (self.intercept[ids] + coordinate[:, None] * slopes).softmax(
                dim=1
            )
            difference = weights - self.current_weights[ids]
            norm, rounding, invalid = mean_distance_from_weights(
                self.support, difference, use_gram=self.config.use_gram
            )
            projected = (difference * self.projected_atoms[ids]).sum(dim=1)
            derivative = -(weights * slopes).sum(dim=1)
            chunks.append(
                torch.stack((norm, projected.abs(), projected, derivative), dim=1)
            )
            self.roundoff.scatter_reduce_(
                0, ids, rounding, reduce="amax", include_self=True
            )
            self.invalid.scatter_reduce_(
                0, ids, invalid.to(torch.int64), reduce="amax", include_self=True
            )
        values = torch.cat(chunks).reshape(len(intervals), 15, 4)
        kronrod = (values * self.wk[None, :, None]).sum(dim=1) * half[:, None]
        gauss = (values * self.wg[None, :, None]).sum(dim=1) * half[:, None]
        magnitude = (values.abs() * self.wk[None, :, None]).sum(dim=1) * half[:, None]
        error = torch.maximum(
            (kronrod - gauss).abs(), 50 * torch.finfo(torch.float64).eps * magnitude
        )
        return kronrod, error


def _adaptive(segment, config):
    count = len(segment.D)
    intercepts, slopes = (
        segment.intercept.detach().cpu().numpy(),
        segment.slopes.detach().cpu().numpy(),
    )
    pending, truncated = [], []
    for query in range(count):
        intervals, over_budget = wide_partition(
            intercepts[query], slopes[query], config
        )
        pending.extend((query, left, right) for left, right in intervals)
        truncated.append(over_budget)
    values, errors = segment.intervals(pending)
    leaves = [[] for _ in range(count)]
    evaluations, refinements = [0] * count, [0] * count
    for (query, left, right), value, error in zip(pending, values, errors):
        leaves[query].append((left, right, value, error))
        evaluations[query] += 15
    exhausted = list(truncated)
    absolute = torch.tensor(
        [config.absolute_tolerance] * 3 + [config.identity_absolute_tolerance],
        dtype=torch.float64, device=segment.device,
    )
    while True:
        pending, replacements = [], []
        for query, items in enumerate(leaves):
            total = torch.stack([leaf[2] for leaf in items]).sum(0)
            error = torch.stack([leaf[3] for leaf in items]).sum(0)
            tolerance = absolute + config.relative_tolerance * total.abs()
            if not bool(torch.isfinite(total).all() & torch.isfinite(error).all()) or bool(
                (error <= tolerance).all()
            ):
                continue
            if evaluations[query] + 30 > config.max_evaluations:
                exhausted[query] = True
                continue
            worst = int((torch.stack([item[3] for item in items]) / tolerance).amax(1).argmax())
            left, right = items[worst][:2]
            middle = (left + right) / 2
            if middle == left or middle == right:
                exhausted[query] = True
                continue
            pending.extend(((query, left, middle), (query, middle, right)))
            replacements.append((query, worst))
            evaluations[query] += 30
            refinements[query] += 1
        if not pending:
            break
        values, errors = segment.intervals(pending)
        for offset, (query, index) in enumerate(replacements):
            children = [
                (pending[j][1], pending[j][2], values[j], errors[j])
                for j in (2 * offset, 2 * offset + 1)
            ]
            leaves[query][index : index + 1] = children
    total = torch.stack([torch.stack([leaf[2] for leaf in items]).sum(0) for items in leaves])
    error = torch.stack([torch.stack([leaf[3] for leaf in items]).sum(0) for items in leaves])
    return total, error, evaluations, refinements, exhausted


def unavailable_integration(
    count, *, device="cpu", reason="endpoint_segment_unavailable"
):
    floats = []
    for name in VARIATIONS:
        floats.extend(
            (f"variation_{name}_integral_l2", f"variation_{name}_integral_error_l2")
        )
    for name in MARGINS:
        prefix = (
            "exact_directional_average"
            if name == "exact_directional_average"
            else "margin_" + name
        )
        floats.extend((prefix + "_l2", prefix + "_rmse", prefix + "_uncertainty_l2"))
    floats.extend(
        (
            "integrated_log_probability_gain",
            "integrated_log_probability_gain_error",
            "integral_identity_residual",
            "integral_identity_allowance",
            "directional_identity_residual",
            "directional_identity_allowance",
            "margin_order_minimum_gap",
            "margin_order_allowance",
            "gram_roundoff_allowance_l2",
            "quadrature_evaluations",
            "quadrature_refinements",
        )
    )
    floats.extend(f"margin_gap_{index}_{index + 1}_l2" for index in range(4))
    result = {
        "candidate_" + key: torch.full(
            (count,), torch.nan, dtype=torch.float64, device=device
        )
        for key in floats
    }
    for name in MARGINS:
        prefix = (
            "candidate_exact_directional_average"
            if name == "exact_directional_average"
            else "candidate_margin_" + name
        )
        result[prefix + "_status"] = ["not_applicable"] * count
        result[prefix + "_estimated_sign"] = ["not_applicable"] * count
    result.update(
        {
            "candidate_integration_status": ["not_applicable"] * count,
            "candidate_integration_reason": [reason] * count,
            "candidate_integration_identity_status": ["not_applicable"] * count,
            "candidate_margin_order_status": ["not_applicable"] * count,
            "candidate_quadrature_budget_exhausted": torch.zeros(
                count, dtype=torch.bool, device=device
            ),
        }
    )
    return result


def integration_metrics(support, payload, *, bank_hash, config=None):
    """Integrate only saved scalar payloads; no raw state/prediction access."""
    config = DEFAULT_INTEGRATION if config is None else config
    if not payload:
        return unavailable_integration(0, device=support.flat.device)
    if payload.get("schema_version") != 1 or payload.get("bank_hash") != bank_hash:
        raise ValueError("Integration payload schema/bank identity mismatch")
    if (
        payload.get("dimension") != support.dimension
        or not 0 <= int(payload["target_atom"]) < support.size
    ):
        raise ValueError("Integration support contract mismatch")
    segment = _Segment(support, payload, config)
    if (
        segment.intercept.shape != segment.slopes.shape
        or segment.intercept.shape != segment.current_weights.shape
        or segment.intercept.shape != (len(segment.D), support.size)
    ):
        raise ValueError("Malformed candidate-logit segment payload")
    if (
        not all(
            bool(torch.isfinite(x).all())
            for x in (segment.intercept, segment.slopes, segment.current_weights)
        )
        or not bool((segment.current_weights >= 0).all())
        or not torch.allclose(
            segment.current_weights.sum(dim=1),
            torch.ones_like(segment.D),
            atol=1e-12,
            rtol=0,
        )
    ):
        raise ValueError("Nonfinite or invalid saved candidate weights/logits")
    count, device = len(segment.D), segment.device
    total, error, evaluations, refinements, exhausted = _adaptive(segment, config)
    values = torch.as_tensor(total, dtype=torch.float64, device=device)
    uncertainty = torch.as_tensor(error, dtype=torch.float64, device=device)

    def tensor(name):
        return torch.as_tensor(payload[name], dtype=torch.float64, device=device)

    D, ec, eu, combined, S, source = (
        tensor(name)
        for name in (
            "delta_norm",
            "conditional_error",
            "unconditional_reference_error",
            "combined_reference_error",
            "signed_error_projection",
            "source_error_l2",
        )
    )
    direction = torch.as_tensor(
        payload["direction_resolved"], dtype=torch.bool, device=device
    )
    reconstruction = torch.as_tensor(
        payload["reconstruction_valid"], dtype=torch.bool, device=device
    )
    V, Vparallel, W, integrated_H = values.unbind(dim=1)
    raw_margins = torch.stack(
        (D - ec - eu - V, D - combined - V, D + S - V, D + S - Vparallel, D + S - W),
        dim=1,
    )
    projection_roundoff = (
        128
        * torch.finfo(torch.float64).eps
        * torch.maximum(torch.ones_like(D), segment.projected_atoms.abs().amax(dim=1))
        * support.size
    )
    margin_errors = torch.stack(
        (
            uncertainty[:, 0] + segment.roundoff + source,
            uncertainty[:, 0] + segment.roundoff + source,
            uncertainty[:, 0] + segment.roundoff + source + projection_roundoff,
            uncertainty[:, 1] + source + projection_roundoff,
            uncertainty[:, 2] + source + projection_roundoff,
        ),
        dim=1,
    )
    prefactor = payload["beta"] * payload["guidance"] * payload["kappa"] * D
    identity = tensor("direct_log_probability_gain") - integrated_H
    identity_allowance = uncertainty[:, 3] + tensor("gain_arithmetic_tolerance")
    directional_identity = integrated_H - prefactor * raw_margins[:, 4]
    directional_allowance = (
        uncertainty[:, 3]
        + prefactor.abs() * margin_errors[:, 4]
        + tensor("gain_arithmetic_tolerance")
    )
    gaps = raw_margins[:, 1:] - raw_margins[:, :-1]
    order_allowance = (
        margin_errors[:, 1:]
        + margin_errors[:, :-1]
        + 128
        * torch.finfo(torch.float64).eps
        * (D + ec + eu + combined + V + Vparallel + W.abs())[:, None]
    )
    order_bad = (gaps < -order_allowance).any(dim=1) & direction
    identity_bad = identity.abs() > identity_allowance
    directional_bad = (directional_identity.abs() > directional_allowance) & direction
    exhausted = torch.as_tensor(exhausted, dtype=torch.bool, device=device)
    valid = (
        torch.isfinite(values).all(dim=1)
        & torch.isfinite(uncertainty).all(dim=1)
        & ~segment.invalid.bool()
    )
    resolved = (
        valid
        & ~exhausted
        & ~identity_bad
        & ~directional_bad
        & ~order_bad
        & reconstruction
    )
    result = unavailable_integration(count, device=device)
    for index, name in enumerate(VARIATIONS):
        applicable = torch.ones_like(direction) if index == 0 else direction
        result[f"candidate_variation_{name}_integral_l2"] = torch.where(
            applicable, values[:, index], torch.nan
        )
        result[f"candidate_variation_{name}_integral_error_l2"] = torch.where(
            applicable, uncertainty[:, index], torch.nan
        )
    host_margins, host_errors = (
        raw_margins.detach().cpu().tolist(),
        margin_errors.detach().cpu().tolist(),
    )
    host_resolved, host_direction = (
        resolved.detach().cpu().tolist(),
        direction.detach().cpu().tolist(),
    )
    for index, name in enumerate(MARGINS):
        prefix = (
            "candidate_exact_directional_average"
            if name == "exact_directional_average"
            else "candidate_margin_" + name
        )
        applicable = torch.ones_like(direction) if index < 2 else direction
        result[prefix + "_l2"] = torch.where(
            applicable, raw_margins[:, index], torch.nan
        )
        result[prefix + "_rmse"] = result[prefix + "_l2"] / math.sqrt(support.dimension)
        result[prefix + "_uncertainty_l2"] = torch.where(
            applicable, margin_errors[:, index], torch.nan
        )
        estimates, statuses = [], []
        for row, errors, ok, directional in zip(
            host_margins, host_errors, host_resolved, host_direction
        ):
            if index >= 2 and not directional or not math.isfinite(row[index]):
                estimates.append("not_applicable")
                statuses.append("not_applicable")
                continue
            estimates.append(
                "positive"
                if row[index] > 0
                else "negative"
                if row[index] < 0
                else "arithmetic_zero"
            )
            statuses.append(
                "positive"
                if ok and row[index] > errors[index]
                else "negative"
                if ok and row[index] < -errors[index]
                else "numerically_unresolved"
            )
        result[prefix + "_estimated_sign"], result[prefix + "_status"] = (
            estimates,
            statuses,
        )
    result.update(
        {
            "candidate_integrated_log_probability_gain": integrated_H,
            "candidate_integrated_log_probability_gain_error": uncertainty[:, 3],
            "candidate_integral_identity_residual": identity,
            "candidate_integral_identity_allowance": identity_allowance,
            "candidate_directional_identity_residual": torch.where(
                direction, directional_identity, torch.nan
            ),
            "candidate_directional_identity_allowance": torch.where(
                direction, directional_allowance, torch.nan
            ),
            "candidate_margin_order_minimum_gap": torch.where(
                direction, gaps.min(dim=1).values, torch.nan
            ),
            "candidate_margin_order_allowance": order_allowance.max(dim=1).values,
            "candidate_gram_roundoff_allowance_l2": segment.roundoff,
            "candidate_quadrature_evaluations": torch.as_tensor(
                evaluations, device=device
            ),
            "candidate_quadrature_refinements": torch.as_tensor(
                refinements, device=device
            ),
            "candidate_quadrature_budget_exhausted": exhausted,
        }
    )
    for index in range(4):
        result[f"candidate_margin_gap_{index}_{index + 1}_l2"] = torch.where(
            direction if index > 0 else torch.ones_like(direction),
            gaps[:, index],
            torch.nan,
        )
    reasons = []
    for good, budget, bad, directional, order, recon, overall in zip(
        valid.tolist(),
        exhausted.tolist(),
        identity_bad.tolist(),
        directional_bad.tolist(),
        order_bad.tolist(),
        reconstruction.tolist(),
        resolved.tolist(),
    ):
        reasons.append(
            "invalid_norm_or_nonfinite_integrand"
            if not good
            else "quadrature_budget_exhausted"
            if budget
            else "endpoint_integral_QA_unresolved"
            if bad
            else "directional_identity_QA_unresolved"
            if directional
            else "bound_order_QA_unresolved"
            if order
            else "matched_displacement_QA_failure"
            if not recon
            else "estimated_converged"
            if overall
            else "numerically_unresolved"
        )
    result["candidate_integration_status"] = [
        "estimated_converged" if ok else "numerically_unresolved"
        for ok in host_resolved
    ]
    result["candidate_integration_reason"] = reasons
    result["candidate_integration_identity_status"] = [
        "numerically_unresolved" if bad else "consistent_with_numerical_estimate"
        for bad in (identity_bad | directional_bad).tolist()
    ]
    result["candidate_margin_order_status"] = [
        "not_applicable"
        if not directional
        else "numerically_unresolved"
        if bad
        else "consistent_with_numerical_estimate"
        for directional, bad in zip(host_direction, order_bad.tolist())
    ]
    population = int(payload.get("population_count", count))
    ids = torch.as_tensor(
        payload.get("row_indices", torch.arange(count)), dtype=torch.long, device=device
    )
    if population == count and torch.equal(ids, torch.arange(count, device=device)):
        return result
    expanded = unavailable_integration(population, device=device)
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            expanded[key][ids] = value.to(expanded[key].dtype)
        else:
            for index, entry in zip(ids.tolist(), value):
                expanded[key][index] = entry
    return expanded
