"""Positive signed-directional reference variation and feedback diagnostics.

The learned network is never evaluated here. Destination Gaussian log weights
are affine on the matched segment. The support Gram matrix is cached once on
its device; node evaluation uses O(nodes * K**2) work and bounded O(nodes * K)
scratch, avoiding query-by-node-by-candidate-by-latent materialization. Raw
latent arithmetic and quadrature use float64. Integration error estimates are
not certified bounds, and all condition decisions are explicitly estimated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .branch_gap import (BRANCH_GAP_ERROR_DEFINITION, BRANCH_GAP_ZERO_CONVENTION,
                         branch_gap_norm, projected_branch_gap_error, unit_branch_gap)
from .support import EVIDENCE


@dataclass(frozen=True)
class IntegrationConfig:
    absolute_tolerance: float = 1e-5  # raw latent L2 units for Equation 15
    relative_tolerance: float = 1e-5
    identity_absolute_tolerance: float = 1e-7  # log-probability units
    max_evaluations: int = 4095  # per query, including initial intervals
    initial_intervals: int = 2
    node_chunk: int = 256
    use_gram: bool = True

    def __post_init__(self):
        if (
            not all(
                math.isfinite(x) and x > 0
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
            raise ValueError("Invalid adaptive integration tolerances or budget")


DEFAULT_INTEGRATION = IntegrationConfig()
INTEGRATION_POLICY = {
    **asdict(DEFAULT_INTEGRATION),
    "version": "projected-gap-error-1",
    "measurement_contract": "projected-gap-error-1",
    "branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
    "branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
    "variation_definition": "positive_part_after_integrated_unit_gap_projection",
    "zero_gap_convention": "V=0_when_Delta_is_exactly_zero",
    "method": "adaptive_Gauss_Kronrod_15_7_with_affine_envelope_initial_partition",
    "error_scope": "successive_embedded_quadrature_estimate_not_certified",
    "source_error_scope": "supplied_source_precision_sensitivity_not_certified",
    "gram_roundoff_multiplier": 128,
    "initial_partition": "uniform plus dominant-logit crossings and +/-2,+/-8 slope-width points",
}

# Embedded Kronrod-15 / Gauss-7 rule, ordered from left to right.
_X = np.array(
    [
        -0.9914553711208126,
        -0.9491079123427585,
        -0.8648644233597691,
        -0.7415311855993945,
        -0.5860872354676911,
        -0.4058451513773972,
        -0.2077849550078985,
        0.0,
        0.2077849550078985,
        0.4058451513773972,
        0.5860872354676911,
        0.7415311855993945,
        0.8648644233597691,
        0.9491079123427585,
        0.9914553711208126,
    ]
)
_WK = np.array(
    [
        0.02293532201052922,
        0.06309209262997855,
        0.1047900103222502,
        0.1406532597155259,
        0.1690047266392679,
        0.1903505780647854,
        0.2044329400752989,
        0.2094821410847278,
        0.2044329400752989,
        0.1903505780647854,
        0.1690047266392679,
        0.1406532597155259,
        0.1047900103222502,
        0.06309209262997855,
        0.02293532201052922,
    ]
)
_WG = np.zeros(15)
_WG[[1, 3, 5, 7, 9, 11, 13]] = [
    0.1294849661688697,
    0.2797053914892767,
    0.3818300505051189,
    0.4179591836734694,
    0.3818300505051189,
    0.2797053914892767,
    0.1294849661688697,
]


def _logits(support, queries, target, alpha, sigma):
    chunks = []
    for start in range(0, support.size, support.candidate_chunk):
        end = min(start + support.candidate_chunk, support.size)
        chunks.append(
            support._relative_logits(queries, target, start, end, alpha, sigma)[0]
        )
    return torch.cat(chunks, dim=1)


def _posterior_fields(logits, target):
    competitors = logits.clone()
    competitors[:, target] = -torch.inf
    log_complement_odds = torch.logsumexp(competitors, dim=1)
    zero = torch.zeros_like(log_complement_odds)
    return {
        "log_probability": -torch.logaddexp(zero, log_complement_odds),
        "log_complement": -torch.logaddexp(zero, -log_complement_odds),
        "log_odds": -log_complement_odds,
    }


def _gram(support):
    if not hasattr(support, "_feedback_centered_gram"):
        centered = support.flat - support.weights @ support.flat
        support._feedback_centered_gram = centered @ centered.T
        support._feedback_gram_max_abs = support._feedback_centered_gram.abs().max()
    return support._feedback_centered_gram


def mean_distance_from_weights(support, difference, *, use_gram=True):
    """Norm of a posterior-mean difference, with explicit Gram roundoff handling.

    The two weight vectors each sum to one, so subtracting a fixed support
    center leaves their mean difference unchanged. Only negative squared norms
    within the declared float64 roundoff allowance are clamped to zero.
    """
    if not use_gram:
        return (
            (difference @ support.flat).norm(dim=1),
            torch.zeros(len(difference), dtype=torch.float64, device=difference.device),
            torch.zeros(len(difference), dtype=torch.bool, device=difference.device),
        )
    gram = _gram(support)
    squared = ((difference @ gram) * difference).sum(dim=1)
    # A standard magnitude-scaled floating-point sensitivity allowance, not a
    # certified interval for GPU BLAS. It is recorded and used conservatively.
    allowance = (
        128
        * torch.finfo(torch.float64).eps
        * support.size
        * support._feedback_gram_max_abs
        * difference.abs().sum(dim=1).square()
    )
    invalid = squared < -allowance
    distance = torch.where(invalid, torch.nan, squared.clamp_min(0).sqrt())
    return distance, allowance.sqrt(), invalid


def _partition(intercept, slope, config):
    """Resolve narrow interior softmax transitions before adaptive refinement.

    Upper-envelope line intersections locate dominant posterior transitions.
    Width neighborhoods sample the smooth transition itself. Non-dominant
    contributions remain in every softmax; no atom is removed or reweighted.
    This protects against easily missed sharp transitions, not a certified
    integration enclosure.
    """
    order = sorted(range(len(slope)), key=lambda i: (slope[i], intercept[i]))
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
        if not 0 < crossing < 1:
            continue
        points.add(float(crossing))
        steepness = abs(slope[hull[position]] - slope[hull[position - 1]])
        for amount in (-8.0, -2.0, 2.0, 8.0):
            point = crossing + amount / steepness
            if 0 < point < 1:
                points.add(float(point))
    points = sorted(points)
    maximum_intervals = max(1, config.max_evaluations // 15)
    truncated = len(points) - 1 > maximum_intervals
    if truncated:
        indexes = np.linspace(0, len(points) - 1, maximum_intervals + 1).astype(int)
        points = [points[i] for i in indexes]
    return list(zip(points[:-1], points[1:])), truncated


class _Segment:
    def __init__(self, support, intercept, slopes, current_weights, projected_atoms, config):
        self.support, self.intercept, self.slopes = support, intercept, slopes
        self.current_weights, self.config = current_weights, config
        self.projected_atoms = projected_atoms
        self.device = intercept.device
        self.x = torch.as_tensor(_X, device=self.device)
        self.wk = torch.as_tensor(_WK, device=self.device)
        self.wg = torch.as_tensor(_WG, device=self.device)
        self.roundoff = torch.zeros(
            len(intercept), dtype=torch.float64, device=self.device
        )
        # CUDA scatter_reduce_(amax) does not support Boolean accumulators.
        # Integer 0/1 maxima preserve the node-invalidity logical OR exactly.
        self.invalid = torch.zeros(
            len(intercept), dtype=torch.int64, device=self.device
        )

    def intervals(self, intervals):
        # Return one pair of integral/error estimates per (query, left, right).
        query = torch.tensor(
            [i[0] for i in intervals], dtype=torch.long, device=self.device
        )
        left = torch.tensor(
            [i[1] for i in intervals], dtype=torch.float64, device=self.device
        )
        right = torch.tensor(
            [i[2] for i in intervals], dtype=torch.float64, device=self.device
        )
        half = (right - left) / 2
        nodes = (left[:, None] + right[:, None]) / 2 + half[:, None] * self.x
        node_queries, nodes = query.repeat_interleave(15), nodes.flatten()
        values = []
        for start in range(0, len(nodes), self.config.node_chunk):
            end = min(start + self.config.node_chunk, len(nodes))
            ids, positions = node_queries[start:end], nodes[start:end]
            slopes = self.slopes[ids]
            logits = self.intercept[ids] + positions[:, None] * slopes
            weights = logits.softmax(dim=1)
            differences = weights - self.current_weights[ids]
            variation, rounding, invalid = mean_distance_from_weights(
                self.support, differences, use_gram=self.config.use_gram
            )
            # With target-relative logits, -E[slope] is exactly the Eq.69
            # integrand alpha_next*g*kappa/sigma_next^2 * Delta.(target-mean).
            derivative = -(weights * slopes).sum(dim=1)
            projected = (differences * self.projected_atoms[ids]).sum(dim=1)
            values.append(torch.stack((projected, derivative, variation), dim=1))
            self.roundoff.scatter_reduce_(
                0, ids, rounding, reduce="amax", include_self=True
            )
            self.invalid.scatter_reduce_(
                0, ids, invalid.to(dtype=torch.int64), reduce="amax", include_self=True
            )
        values = torch.cat(values).reshape(len(intervals), 15, 3)
        kronrod = (values * self.wk[None, :, None]).sum(dim=1) * half[:, None]
        gauss = (values * self.wg[None, :, None]).sum(dim=1) * half[:, None]
        magnitude = (values.abs() * self.wk[None, :, None]).sum(dim=1) * half[:, None]
        error = torch.maximum(
            (kronrod - gauss).abs(), 50 * torch.finfo(torch.float64).eps * magnitude
        )
        return kronrod, error


def _integrate(segment, config):
    count = len(segment.intercept)
    intercepts = segment.intercept.detach().cpu().numpy()
    slopes = segment.slopes.detach().cpu().numpy()
    intervals, truncated = [], []
    for query in range(count):
        partition, was_truncated = _partition(intercepts[query], slopes[query], config)
        intervals.extend((query, left, right) for left, right in partition)
        truncated.append(was_truncated)
    values, errors = segment.intervals(intervals)
    leaves = [[] for _ in range(count)]
    evaluations, refinements = [0] * count, [0] * count
    for interval, value, error in zip(intervals, values, errors):
        query, left, right = interval
        leaves[query].append((left, right, value, error))
        evaluations[query] += 15
    absolute = torch.tensor([config.absolute_tolerance, config.identity_absolute_tolerance, config.absolute_tolerance],
                            dtype=torch.float64, device=segment.intercept.device)
    budget_exhausted = list(truncated)
    while True:
        pending, replace = [], []
        for query in range(count):
            total = torch.stack([leaf[2] for leaf in leaves[query]]).sum(0)
            uncertainty = torch.stack([leaf[3] for leaf in leaves[query]]).sum(0)
            tolerance = absolute + config.relative_tolerance * total.abs()
            if not bool(torch.isfinite(total).all() & torch.isfinite(uncertainty).all()):
                continue
            if bool((uncertainty <= tolerance).all()):
                continue
            if evaluations[query] + 30 > config.max_evaluations:
                budget_exhausted[query] = True
                continue
            worst = int((torch.stack([leaf[3] for leaf in leaves[query]]) / tolerance).amax(1).argmax())
            left, right = leaves[query][worst][:2]
            middle = (left + right) / 2
            if middle == left or middle == right:
                budget_exhausted[query] = True
                continue
            pending.extend(((query, left, middle), (query, middle, right)))
            replace.append((query, worst))
            evaluations[query] += 30
            refinements[query] += 1
        if not pending:
            break
        values, errors = segment.intervals(pending)
        for offset, (query, index) in enumerate(replace):
            children = []
            for j in (2 * offset, 2 * offset + 1):
                _, left, right = pending[j]
                children.append((left, right, values[j], errors[j]))
            leaves[query][index : index + 1] = children
    totals = torch.stack([torch.stack([leaf[2] for leaf in items]).sum(0) for items in leaves])
    uncertainty = torch.stack([torch.stack([leaf[3] for leaf in items]).sum(0) for items in leaves])
    return totals, uncertainty, evaluations, refinements, budget_exhausted


def unavailable_feedback(count, *, device="cpu", reason, status="not_applicable"):
    """Stable scalar schema for valid observations without an eligible comparison."""
    keys = (
        "candidate_conditional_reference_error_l2",
        "candidate_unconditional_reference_error_l2",
        "candidate_branch_gap_error_l2",
        "candidate_branch_gap_error_norm_diagnostic_l2",
        "candidate_variation_l2",
        "candidate_variation_error_l2",
        "candidate_variation_signed_integral_l2",
        "candidate_variation_signed_integral_error_l2",
        "candidate_variation_norm_diagnostic_l2",
        "candidate_variation_norm_diagnostic_error_l2",
        "candidate_projection_roundoff_allowance_l2",
        "candidate_condition_margin_l2",
        "candidate_condition_margin_rmse",
        "candidate_condition_numerical_uncertainty_l2",
        "candidate_log_probability_gain",
        "candidate_log_odds_gain",
        "candidate_integrated_log_probability_gain",
        "candidate_integral_identity_residual",
        "candidate_integral_identity_error_estimate",
        "candidate_proof_lower_bound",
        "candidate_proof_lower_bound_slack",
        "candidate_gram_roundoff_allowance_l2",
        "candidate_source_sensitivity_l2",
        "candidate_quadrature_evaluations",
        "candidate_quadrature_refinements",
        "candidate_gain_numerical_tolerance",
        "candidate_segment_displacement_residual_l2",
    )
    result = {
        key: torch.full((count,), torch.nan, dtype=torch.float64, device=device)
        for key in keys
    }
    for endpoint in ("matched", "guided"):
        for field in ("log_probability", "log_complement", "log_odds"):
            result[f"candidate_{endpoint}_{field}"] = torch.full(
                (count,), torch.nan, dtype=torch.float64, device=device
            )
    result.update(
        {
            "feedback_evidence": EVIDENCE,
            "candidate_measurement_contract": "projected-gap-error-1",
            "candidate_branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
            "candidate_branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
            "candidate_comparison_scope": "Condition uses signed e_Delta=unit_Delta dot (Delta-bar_Delta) and V=[integral unit_Delta dot (next_reference-current_reference)]_+; the same endpoint and reference assumptions are required",
            "candidate_variation_definition": "positive_part_after_integrated_unit_gap_projection",
            "candidate_zero_gap_convention": "V=0_when_Delta_is_exactly_zero",
            "candidate_variation_direction_status": ["unavailable_projection_denominator"] * count,
            "feedback_eligible": torch.zeros(count, dtype=torch.bool, device=device),
            "feedback_status": [reason] * count,
            "candidate_condition_status": [status] * count,
            "candidate_gain_status": ["unavailable"] * count,
            "candidate_integral_status": [status] * count,
            "candidate_integral_qa_status": [status] * count,
            "candidate_manuscript_guidance_status": ["not_evaluated"] * count,
            "candidate_gain_saturated": torch.zeros(
                count, dtype=torch.bool, device=device
            ),
            "candidate_gain_underflow": torch.zeros(
                count, dtype=torch.bool, device=device
            ),
            "candidate_quadrature_budget_exhausted": torch.zeros(
                count, dtype=torch.bool, device=device
            ),
            "candidate_integration_absolute_tolerance": DEFAULT_INTEGRATION.absolute_tolerance,
            "candidate_integration_relative_tolerance": DEFAULT_INTEGRATION.relative_tolerance,
            "candidate_integration_max_evaluations": DEFAULT_INTEGRATION.max_evaluations,
        }
    )
    return result


def proposition5_feedback(
    support,
    current,
    conditional,
    unconditional,
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
    source_error_l2=0.0,
    config=None,
):
    """Compute the positive-directional variation condition and matched log-p gain.

    The caller establishes the affine scheduler and nonterminal transition.
    `source_error_l2` is a predeclared, raw-L2 source-rounding sensitivity of the
    condition components, not an inferred model error or formal certificate.
    Returned tensors are scalar per query; strings are parallel status lists.
    """
    config = DEFAULT_INTEGRATION if config is None else config
    z, lead = support._queries(current)
    mc, _ = support._queries(conditional)
    mu, _ = support._queries(unconditional)
    cf, _ = support._queries(matched)
    observed, _ = support._queries(guided)
    if any(value.shape != z.shape for value in (mc, mu, cf, observed)):
        raise ValueError("Current, branch and matched query batches differ")
    count, device = len(z), z.device
    parameters = (
        current_alpha,
        current_sigma,
        destination_alpha,
        destination_sigma,
        guidance,
        kappa,
    )
    a, s, an, sn, g, kap = map(float, parameters)
    result = unavailable_feedback(count, device=device, reason="inapplicable_feedback")
    result.update(
        {
            "candidate_integration_absolute_tolerance": config.absolute_tolerance,
            "candidate_integration_relative_tolerance": config.relative_tolerance,
            "candidate_integration_max_evaluations": config.max_evaluations,
            "candidate_manuscript_guidance_status": [
                (
                    "within_manuscript_domain_g_gt_1"
                    if g > 1
                    else "outside_manuscript_domain_analytic_positive_guidance_diagnostic"
                )
            ]
            * count,
        }
    )
    if str(target_id) not in support.aliases:
        result["feedback_status"] = [
            "target_absent_from_declared_candidate_distribution"
        ] * count
        result["candidate_condition_status"] = ["unavailable"] * count
        return result
    if support.size < 2:
        result["feedback_status"] = ["single_atom_candidate_distribution"] * count
        return result
    if (
        not all(math.isfinite(x) for x in (a, s, an, sn, g, kap))
        or min(a, an, s, sn, g, kap) <= 0
    ):
        result["feedback_status"] = [
            "nonpositive_or_invalid_noise_guidance_or_kappa"
        ] * count
        return result
    valid = torch.isfinite(torch.cat((z, mc, mu, cf, observed), dim=1)).all(dim=1)
    if not bool(valid.all()):
        # Malformed rows do not erase usable rows in the same device batch.
        for index in (~valid).nonzero().flatten().tolist():
            result["feedback_status"][index] = "invalid_nonfinite_feedback_input"
            result["candidate_condition_status"][index] = "unavailable"
        if bool(valid.any()):
            ids = valid.nonzero().flatten()
            error = torch.as_tensor(
                source_error_l2, dtype=torch.float64, device=device
            ).expand(count)[ids]
            subset = proposition5_feedback(
                support,
                *(
                    value[ids].reshape(-1, *support.latent_shape)
                    for value in (z, mc, mu, cf, observed)
                ),
                target_id,
                current_alpha=a,
                current_sigma=s,
                destination_alpha=an,
                destination_sigma=sn,
                guidance=g,
                kappa=kap,
                source_error_l2=error,
                config=config,
            )
            indices = ids.tolist()
            for key, value in subset.items():
                if isinstance(value, torch.Tensor):
                    result[key][ids] = value.reshape(-1).to(result[key].dtype)
                elif isinstance(value, list):
                    for destination, item in zip(indices, value):
                        result[key][destination] = item
                else:
                    result[key] = value
        return result
    target = support.aliases[str(target_id)]
    delta, shift = mc - mu, g * kap * (mc - mu)
    current_logits = _logits(support, z, target, a, s)
    current_weights = current_logits.softmax(dim=1)
    reference = current_weights @ support.flat
    ec = (mc - support.flat[target]).norm(dim=1)
    eu = (mu - reference).norm(dim=1)
    gap = branch_gap_norm(delta)
    gap_error = projected_branch_gap_error(delta, support.flat[target] - reference)
    error_norm_diagnostic = branch_gap_norm(delta - (support.flat[target] - reference))
    intercept = _logits(support, cf, target, an, sn)
    difference = support.target_geometry(target)[0]
    slopes = (an / (sn * sn)) * (shift @ difference.T)
    zero_gap = (delta == 0).all(dim=1)
    projection_available = torch.isfinite(gap) & (gap > 0)
    direction = unit_branch_gap(delta)
    projected_atoms = direction @ difference.T
    projection_available &= torch.isfinite(projected_atoms).all(dim=1)
    projected_atoms = torch.where(zero_gap[:, None], torch.zeros_like(projected_atoms),
        torch.where(projection_available[:, None], projected_atoms, torch.full_like(projected_atoms, torch.nan)))
    segment = _Segment(support, intercept, slopes, current_weights, projected_atoms, config)
    totals, errors, evaluations, refinements, exhausted = _integrate(segment, config)
    values = torch.as_tensor(totals, dtype=torch.float64, device=device)
    uncertainty = torch.as_tensor(errors, dtype=torch.float64, device=device)
    source_error = torch.as_tensor(
        source_error_l2, dtype=torch.float64, device=device
    ).expand(count)
    if not bool(torch.isfinite(source_error).all() & (source_error >= 0).all()):
        raise ValueError("Source-error sensitivity must be finite and nonnegative")
    signed_variation, integrated_gain = values[:, 0], values[:, 1]
    signed_variation = torch.where(zero_gap, torch.zeros_like(gap), signed_variation)
    variation = signed_variation.clamp_min(0)
    variation_error = torch.where(zero_gap, torch.zeros_like(gap), uncertainty[:, 0])
    projection_roundoff = 128 * torch.finfo(torch.float64).eps * support.size * torch.maximum(
        torch.ones_like(gap), projected_atoms.abs().amax(dim=1))
    projection_roundoff = torch.where(zero_gap, torch.zeros_like(gap), projection_roundoff)
    margin = gap - gap_error - variation
    margin_error = variation_error + projection_roundoff + source_error + 128 * torch.finfo(torch.float64).eps * (gap + gap_error.abs() + variation)
    endpoints = {
        "matched": _posterior_fields(intercept, target),
        # Compute observed endpoint logits independently of segment interpolation.
        "guided": _posterior_fields(_logits(support, observed, target, an, sn), target),
    }
    lp0, lp1 = (
        endpoints["matched"]["log_probability"],
        endpoints["guided"]["log_probability"],
    )
    odds0, odds1 = endpoints["matched"]["log_odds"], endpoints["guided"]["log_odds"]
    gain, odds_gain = lp1 - lp0, odds1 - odds0
    gain_tolerance = (
        128
        * torch.finfo(torch.float64).eps
        * torch.maximum(torch.ones_like(gain), torch.maximum(odds0.abs(), odds1.abs()))
        * max(1.0, math.log2(support.size + 1))
    )
    saturation = (lp0.exp() == 1) | (lp1.exp() == 1)
    resolved_gain = odds_gain.abs() > gain_tolerance
    underflow = (gain == 0) & resolved_gain
    prefactor = an * g * kap / (sn * sn)
    lower_bound = prefactor * gap * margin
    identity_residual = gain - integrated_gain
    qa_allowance = uncertainty[:, 1] + gain_tolerance
    source_qa_allowance = prefactor * gap * margin_error
    segment_residual = (observed - cf - shift).norm(dim=1)
    displacement_tolerance = (
        128
        * torch.finfo(torch.float64).eps
        * torch.maximum(
            torch.ones_like(gap),
            observed.norm(dim=1) + cf.norm(dim=1) + shift.norm(dim=1),
        )
    )
    result.update(
        {
            "feedback_eligible": torch.ones(count, dtype=torch.bool, device=device),
            "candidate_conditional_reference_error_l2": ec,
            "candidate_unconditional_reference_error_l2": eu,
            "candidate_branch_gap_error_l2": gap_error,
            "candidate_branch_gap_error_norm_diagnostic_l2": error_norm_diagnostic,
            "candidate_measurement_contract": "projected-gap-error-1",
            "candidate_branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
            "candidate_branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
            "candidate_comparison_scope": "Condition uses signed e_Delta=unit_Delta dot (Delta-bar_Delta) and V=[integral unit_Delta dot (next_reference-current_reference)]_+; the same endpoint and reference assumptions are required",
            "candidate_variation_l2": variation,
            "candidate_variation_error_l2": variation_error,
            "candidate_variation_signed_integral_l2": signed_variation,
            "candidate_variation_signed_integral_error_l2": variation_error,
            "candidate_variation_norm_diagnostic_l2": values[:, 2],
            "candidate_variation_norm_diagnostic_error_l2": uncertainty[:, 2],
            "candidate_projection_roundoff_allowance_l2": projection_roundoff,
            "candidate_variation_direction_status": [
                "zero_gap_convention" if zero else "finite_nonzero_gap_projection" if available
                else "unavailable_projection_denominator"
                for zero, available in zip(zero_gap.tolist(), projection_available.tolist())],
            "candidate_condition_margin_l2": margin,
            "candidate_condition_margin_rmse": margin / math.sqrt(support.dimension),
            "candidate_condition_numerical_uncertainty_l2": margin_error,
            "candidate_log_probability_gain": gain,
            "candidate_log_odds_gain": odds_gain,
            "candidate_integrated_log_probability_gain": integrated_gain,
            "candidate_integral_identity_residual": identity_residual,
            "candidate_integral_identity_error_estimate": uncertainty[:, 1],
            "candidate_proof_lower_bound": lower_bound,
            "candidate_proof_lower_bound_slack": gain - lower_bound,
            "candidate_gram_roundoff_allowance_l2": segment.roundoff,
            "candidate_source_sensitivity_l2": source_error,
            "candidate_quadrature_evaluations": torch.as_tensor(
                evaluations, device=device
            ),
            "candidate_quadrature_refinements": torch.as_tensor(
                refinements, device=device
            ),
            "candidate_quadrature_budget_exhausted": torch.as_tensor(
                exhausted, device=device
            ),
            "candidate_gain_numerical_tolerance": gain_tolerance,
            "candidate_gain_saturated": saturation,
            "candidate_gain_underflow": underflow,
            "candidate_segment_displacement_residual_l2": segment_residual,
        }
    )
    for endpoint, fields in endpoints.items():
        result.update(
            {f"candidate_{endpoint}_{name}": value for name, value in fields.items()}
        )
    # Only this compact scalar status matrix crosses devices after quadrature.
    decisions = (
        torch.stack(
            (
                margin,
                margin_error,
                odds_gain,
                gain_tolerance,
                identity_residual.abs(),
                qa_allowance,
                gain - lower_bound,
                source_qa_allowance,
                segment_residual,
                displacement_tolerance,
                segment.invalid.double(),
                (observed == cf).all(dim=1).double(),
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .tolist()
    )
    for index, (
        m,
        me,
        og,
        gt,
        residual,
        allowance,
        slack,
        slack_allowance,
        sr,
        st,
        invalid,
        identical,
    ) in enumerate(decisions):
        integration_ok = not exhausted[index] and not invalid and math.isfinite(m) and math.isfinite(me)
        qa_ok = residual <= allowance and slack >= -slack_allowance - gt and sr <= st
        result["candidate_integral_status"][index] = (
            "estimated_converged" if integration_ok else "numerically_unresolved"
        )
        result["candidate_integral_qa_status"][index] = (
            "consistent_with_numerical_estimates"
            if qa_ok
            else "numerical_inconsistency"
        )
        result["candidate_condition_status"][index] = (
            "estimated_met"
            if integration_ok and qa_ok and m > me
            else (
                "estimated_not_met"
                if integration_ok and qa_ok and m < -me
                else "numerically_unresolved"
            )
        )
        result["candidate_gain_status"][index] = (
            "positive"
            if og > gt
            else (
                "negative"
                if og < -gt
                else "zero" if identical else "numerically_unresolved"
            )
        )
        result["feedback_status"][index] = (
            "valid_candidate_distribution_comparison"
            if integration_ok and qa_ok
            else (
                "gram_squared_norm_outside_roundoff_allowance"
                if invalid
                else (
                    "quadrature_budget_exhausted"
                    if exhausted[index]
                    else "numerical_inconsistency_requires_investigation"
                )
            )
        )
    return result
