"""Shared cached-vector measurements of Corollary 3 through Theorem 7.

No learned component or raw-cache loader is called here. The orchestrator loads
one record and supplies its canonical epsilon branches. The feedback penalty is
the positive part of the integrated unit-gap projection of the cross-step
unconditional reference displacement, with clipping only after integration.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math

import pandas as pd
import torch

from .branch_gap import (BRANCH_GAP_ERROR_DEFINITION, BRANCH_GAP_ZERO_CONVENTION,
                         projected_branch_gap_error)
from .candidate_feedback import stable_log_probability_gain, stable_logsum_difference
from .candidate_integration import DEFAULT_INTEGRATION
from .direct_integration import integrate_proposition5_payload
from .feedback import _logits, _posterior_fields
from .metrics import clean_estimates

DIRECT_MEASUREMENT_VERSION = "direct-cached-vectors-projected-gap-error-1"
MEASUREMENT_CONTRACT = "projected-gap-error-1"
DIRECT_INTEGRATION_RECIPE = {
    "version": "direct-projected-gap-error-1",
    "measurement_contract": MEASUREMENT_CONTRACT,
    "branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
    "branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
    "condition": "||Delta|| - unit_Delta dot (Delta-reference_Delta) - V",
    **asdict(DEFAULT_INTEGRATION),
    "variation": "max(0, integral_0^1 (Delta/||Delta||).(posterior_mean_next(cf+s*g*kappa*Delta)-posterior_mean_current(state)) ds)",
    "positive_part_placement": "after the signed integral, before per-seed averaging",
    "zero_gap_convention": "V=0_when_Delta_is_exactly_zero",
    "directional_domain": "t=2,...,T; no final-output variation",
    "status_scope": "embedded_quadrature_and_source_sensitivity_estimates_not_certified_enclosures",
    "reuse": "candidate_integration.wide_partition_and_signed_projection_integrator; norm_integral_retained_as_separate_diagnostic",
}


def _norm(value, latent_ndim):
    """Scaled L2 norm retains small representable vectors before squaring."""
    dims = tuple(range(-latent_ndim, 0))
    scale = value.abs().amax(dim=dims, keepdim=True)
    normalized = value / torch.where(scale > 0, scale, torch.ones_like(scale))
    return normalized.square().sum(dim=dims).sqrt() * scale.reshape(value.shape[:-latent_ndim])


def _put_norm(result, stem, value, dimension):
    result[stem + "_l2"] = value
    result[stem + "_squared_l2"] = value.square()
    result[stem + "_mse"] = value.square() / dimension
    result[stem + "_squared_underflow"] = (value != 0) & (value.square() == 0)
    result[stem + "_rmse"] = value / math.sqrt(dimension)


def corollary3_metrics(mu, mc, target, mean, guidance):
    """Full-vector comparisons about one centre, with g and g-1 coefficients."""
    mu, mc, target, mean = [torch.as_tensor(x).to(dtype=torch.float64) for x in (mu, mc, target, mean)]
    ndim, d, g = target.ndim, target.numel(), float(guidance)
    rc, bu = mc - target, mu - mean
    delta = mc - mu
    injection = g * delta
    injection_limit = g * (target - mean)
    guided = mu + injection
    guided_limit = mean + injection_limit
    first = injection - injection_limit
    second = guided - guided_limit
    ec, b = _norm(rc, ndim), _norm(bu, ndim)
    result = {"direct_cor3_applicable": math.isfinite(g) and g > 1,
              "direct_cor3_status": "derived_finite_triangle_comparison" if math.isfinite(g) and g > 1 else "outside_manuscript_guidance_domain_g_gt_1"}
    for name, values in (
        ("unconditional_mean_error", b),
        ("cor3_injection_discrepancy", _norm(first, ndim)),
        ("cor3_injection_bound", g * (ec + b)),
        ("cor3_guided_discrepancy", _norm(second, ndim)),
        ("cor3_guided_bound", g * ec + (g - 1) * b),
        ("cor3_injection_vector_identity_residual", _norm(first - g * (rc - bu), ndim)),
        ("cor3_guided_vector_identity_residual", _norm(second - (g * rc - (g - 1) * bu), ndim)),
    ):
        _put_norm(result, "direct_" + name, values, d)
    dot = (rc * bu).sum(dim=tuple(range(-ndim, 0)))
    result["direct_cor3_error_mean_cross_inner_product"] = dot
    result["direct_cor3_injection_cross_term_squared_l2"] = -2 * g * g * dot
    result["direct_cor3_guided_cross_term_squared_l2"] = -2 * g * (g - 1) * dot
    result["direct_cor3_injection_slack_l2"] = result["direct_cor3_injection_bound_l2"] - result["direct_cor3_injection_discrepancy_l2"]
    result["direct_cor3_guided_slack_l2"] = result["direct_cor3_guided_bound_l2"] - result["direct_cor3_guided_discrepancy_l2"]
    return result


def current_reference_metrics(state, mu, mc, target, law, target_id, alpha, sigma):
    """One current-state posterior; its weights also feed the matched integral."""
    support = law.support
    flat, _ = support._queries(state)
    d, ndim = support.dimension, target.ndim
    anchor = support.aliases[target_id] if target_id is not None else 0
    if not (math.isfinite(float(alpha)) and float(alpha) >= 0 and math.isfinite(float(sigma)) and float(sigma) > 0):
        nan = torch.full((len(flat),), torch.nan, dtype=torch.float64, device=flat.device)
        result = {"direct_lemma6_applicable": False,
                  "direct_lemma6_status": "unavailable_current_reference_nonpositive_noise",
                  "direct_measurement_contract": MEASUREMENT_CONTRACT,
                  "direct_branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
                  "direct_branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
                  "direct_target_log_probability": nan, "direct_target_log_complement": nan,
                  "direct_target_log_odds": nan, "direct_non_target_mass": nan,
                  "direct_target_radius_l2": law.radius_for(target),
                  "direct_radius_tail_underflow": torch.zeros_like(nan, dtype=torch.bool),
                  "direct_radius_tail_status": "unavailable_current_reference",
                  "direct_reference_target_positive_mass": target_id is not None}
        for name in ("conditional_error", "unconditional_reference_error", "unconditional_target_error",
                     "reference_target_error", "reference_branch_gap", "branch_gap_error", "branch_gap_error_norm_diagnostic", "reference_mean_offset", "reference_to_bank_mean_error", "radius_tail", "branch_gap", "lemma6_rhs", "lemma6_gap"):
            value = _norm(mc - target, ndim) if name == "conditional_error" else _norm(mu - target, ndim) if name == "unconditional_target_error" else _norm(mc - mu, ndim) if name in {"branch_gap", "lemma6_gap"} else nan
            _put_norm(result, "direct_" + name, value, d)
        return result, torch.full((len(flat), support.size), torch.nan, dtype=torch.float64, device=flat.device), torch.full_like(mu, torch.nan)
    logits = _logits(support, flat, anchor, float(alpha), float(sigma))
    weights = logits.softmax(dim=1)
    geometry = support.target_geometry(anchor)[0]
    mean_flat = support.flat[anchor] + weights @ geometry
    mean = mean_flat.reshape_as(mu)
    fields = _posterior_fields(logits, anchor)
    if target_id is None:
        fields = {"log_probability": torch.full_like(fields["log_probability"], -torch.inf),
                  "log_complement": torch.zeros_like(fields["log_complement"]),
                  "log_odds": torch.full_like(fields["log_odds"], -torch.inf)}
    radius = law.radius_for(target)
    logtail = fields["log_complement"]
    tail = logtail.exp()
    radius_tail = radius * tail
    ec, eu = _norm(mc - target, ndim), _norm(mu - mean, ndim)
    gap = _norm(mc - mu, ndim)
    ref_target = _norm(mean - target, ndim)
    branch_gap_error = projected_branch_gap_error(mc - mu, target - mean, latent_ndim=ndim)
    norm_diagnostic = _norm((mc - mu) - (target - mean), ndim)
    rhs = branch_gap_error + radius_tail
    underflow = (radius > 0) & torch.isfinite(logtail) & (radius_tail == 0)
    result = {
        "direct_target_log_probability": fields["log_probability"],
        "direct_target_log_complement": logtail,
        "direct_target_log_odds": fields["log_odds"],
        "direct_non_target_mass": tail,
        "direct_target_radius_l2": radius,
        "direct_radius_tail_underflow": underflow,
        "direct_reference_target_positive_mass": target_id is not None,
        "direct_lemma6_applicable": torch.isfinite(rhs) & torch.isfinite(gap),
        "direct_lemma6_status": "branch_gap_error_refined_bound",
        "direct_measurement_contract": MEASUREMENT_CONTRACT,
        "direct_branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
        "direct_branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
        "direct_lemma6_comparison_scope": "signed_projected_gap_error_plus_radius_tail; projection_bound_under_single_target_reference",
        "direct_lemma6_numerical_status": ["nonzero_tail_underflow" if value else "finite_float64_estimate" for value in underflow.tolist()],
        "direct_lemma6_zero_rhs_is_exact": (rhs == 0) & ~underflow,
        "direct_lemma6_slack_l2": rhs - gap,
        "direct_lemma6_bound_error_ratio": rhs / torch.where(gap > 0, gap, torch.nan),
        "direct_lemma6_zero_gap_ratio_undefined": gap == 0,
        "direct_radius_tail_status": ["underflow_nonzero_tail_not_exact_zero" if value else "represented" for value in underflow.tolist()],
    }
    for name, value in (
        ("conditional_error", ec), ("unconditional_reference_error", eu),
        ("unconditional_target_error", _norm(mu - target, ndim)),
        ("reference_target_error", ref_target), ("reference_branch_gap", ref_target),
        ("branch_gap_error", branch_gap_error),
        ("branch_gap_error_norm_diagnostic", norm_diagnostic),
        ("reference_mean_offset", _norm(mean - law.theory_mean_vector, ndim)),
        ("reference_to_bank_mean_error", _norm(mean - law.mean_vector, ndim)),
        ("radius_tail", radius_tail), ("branch_gap", gap),
        ("lemma6_rhs", rhs), ("lemma6_gap", gap),
    ):
        _put_norm(result, "direct_" + name, value, d)
    result["direct_reference_concentration_slack_l2"] = radius_tail - ref_target
    result["direct_reference_concentration_roundoff_l2"] = 128 * torch.finfo(torch.float64).eps * (
        _norm(mean, ndim) + _norm(target, ndim) + radius
    )
    return result, weights, mean


def _unavailable_prop5(count, device, reason, *, status="not_applicable"):
    names = (
        "margin_l2", "margin_rmse", "variation_l2", "variation_rmse", "variation_error_l2",
        "variation_signed_integral_l2", "variation_signed_integral_error_l2",
        "variation_norm_diagnostic_l2", "variation_norm_diagnostic_error_l2", "projection_roundoff_l2",
        "log_probability_gain", "log_odds_gain", "integrated_gain", "integrated_gain_error",
        "signed_integral", "lower_bound", "lower_bound_slack", "uncertainty_l2",
        "gain_arithmetic_tolerance", "integral_identity_residual", "integral_identity_allowance",
        "gram_roundoff_l2", "source_sensitivity_l2", "quadrature_evaluations",
        "quadrature_refinements", "prefactor",
    )
    result = {"direct_prop5_" + name: torch.full((count,), torch.nan, dtype=torch.float64, device=device) for name in names}
    result.update({
        "direct_prop5_applicable": False, "direct_prop5_status": reason,
        "direct_prop5_condition_status": [status] * count,
        "direct_prop5_estimated_condition_sign": [status] * count,
        "direct_prop5_gain_status": [status] * count,
        "direct_prop5_integral_status": [status] * count,
        "direct_prop5_nonnegative_condition": [None] * count,
        "direct_prop5_strict_positive_condition": [None] * count,
        "direct_prop5_zero_displacement": [None] * count,
        "direct_prop5_certification_status": "not_certified_quadrature_estimate",
        "direct_prop5_measurement_contract": MEASUREMENT_CONTRACT,
        "direct_prop5_branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
        "direct_prop5_branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
        "direct_prop5_variation_definition": "positive_part_after_integrated_unit_gap_projection",
        "direct_prop5_zero_gap_convention": "V=0_when_Delta_is_exactly_zero",
        "direct_prop5_variation_direction_status": ["not_applicable"] * count,
        "direct_prop5_comparison_scope": "branch_gap_error_refinement; implication_requires_original_endpoint_and_reference_contracts",
        "direct_prop5_quadrature_budget_exhausted": [None] * count,
    })
    return result


def proposition5_metrics(state, mu, mc, matched, observed, target, target_id,
                         law, coeff, current_weights, reference_mean, source_error,
                         reconstruction_valid, guidance, *, config=None, integrate=True,
                         payload_callback=None):
    """Direct endpoint gain and the branch-gap-error-minus-V condition."""
    support, g = law.support, float(guidance)
    count, d = len(state), support.dimension
    result = _unavailable_prop5(count, state.device, "inapplicable_destination_or_reference")
    if not (coeff.affine and coeff.kappa > 0 and coeff.alpha > 0 and coeff.sigma > 0 and coeff.destination_sigma > 0
            and coeff.destination_alpha > 0 and g > 1 and target_id is not None):
        return result
    config = DEFAULT_INTEGRATION if config is None else config
    if support.size == 1:
        D = _norm(mc - mu, target.ndim)
        branch_gap_error = projected_branch_gap_error(mc - mu, target - reference_mean, latent_ndim=target.ndim)
        margin = D - branch_gap_error
        zero = torch.zeros_like(D)
        prefactor = coeff.destination_alpha * g * coeff.kappa / coeff.destination_sigma**2
        uncertainty = source_error + 128 * torch.finfo(torch.float64).eps * (D + branch_gap_error.abs())
        result.update(
            direct_prop5_applicable=True,
            direct_prop5_status="analytic_single_atom_reference_probability_one",
            direct_prop5_margin_l2=margin, direct_prop5_margin_rmse=margin / math.sqrt(d),
            direct_prop5_variation_l2=zero, direct_prop5_variation_rmse=zero,
            direct_prop5_variation_error_l2=zero,
            direct_prop5_variation_signed_integral_l2=zero,
            direct_prop5_variation_signed_integral_error_l2=zero,
            direct_prop5_variation_norm_diagnostic_l2=zero,
            direct_prop5_variation_norm_diagnostic_error_l2=zero,
            direct_prop5_projection_roundoff_l2=zero,
            direct_prop5_variation_direction_status=["exact_zero_gap_convention" if bool(value) else "analytic_constant_reference" for value in (mc == mu).reshape(count, -1).all(dim=1).tolist()],
            direct_prop5_log_probability_gain=zero,
            direct_prop5_gain_status=["zero"] * count,
            direct_prop5_integrated_gain=zero, direct_prop5_integrated_gain_error=zero,
            direct_prop5_signed_integral=zero, direct_prop5_integral_identity_residual=zero,
            direct_prop5_integral_identity_allowance=zero,
            direct_prop5_integral_status=["analytic_single_atom_reference"] * count,
            direct_prop5_lower_bound=prefactor * D * margin,
            direct_prop5_lower_bound_slack=-prefactor * D * margin,
            direct_prop5_uncertainty_l2=uncertainty,
            direct_prop5_source_sensitivity_l2=source_error,
            direct_prop5_gain_arithmetic_tolerance=zero,
            direct_prop5_gram_roundoff_l2=zero,
            direct_prop5_quadrature_evaluations=zero,
            direct_prop5_quadrature_refinements=zero,
            direct_prop5_quadrature_budget_exhausted=torch.zeros_like(D, dtype=torch.bool),
            direct_prop5_prefactor=prefactor,
            direct_prop5_zero_displacement=D == 0,
            direct_prop5_nonnegative_condition=margin >= 0,
            direct_prop5_strict_positive_condition=margin > 0,
            direct_prop5_condition_status=["estimated_positive" if m > e else "estimated_negative" if m < -e else "numerically_unresolved" for m, e in zip(margin.tolist(), uncertainty.tolist())],
            direct_prop5_estimated_condition_sign=["positive" if m > 0 else "negative" if m < 0 else "arithmetic_zero" for m in margin.tolist()],
            direct_prop5_certification_status="analytic_constant_reference_integral; margin_source_sensitivity_not_certified",
            direct_prop5_matched_log_probability=zero,
            direct_prop5_guided_log_probability=zero,
            direct_prop5_matched_log_complement=torch.full_like(D, -torch.inf),
            direct_prop5_guided_log_complement=torch.full_like(D, -torch.inf),
            direct_prop5_gain_saturated=torch.ones_like(D, dtype=torch.bool),
            direct_prop5_gain_underflow=torch.zeros_like(D, dtype=torch.bool),
        )
        return result
    cf, observed_flat = support._queries(matched)[0], support._queries(observed)[0]
    delta = (mc - mu).reshape(count, -1)
    D = _norm(mc - mu, target.ndim)
    index = support.aliases[target_id]
    beta = coeff.destination_alpha / coeff.destination_sigma**2
    intercept = _logits(support, cf, index, coeff.destination_alpha, coeff.destination_sigma)
    endpoint = _logits(support, observed_flat, index, coeff.destination_alpha, coeff.destination_sigma)
    slopes = beta * ((g * coeff.kappa * delta) @ support.target_geometry(index)[0].T)
    competitor = torch.arange(support.size, device=state.device) != index
    p0, p1 = _posterior_fields(intercept, index), _posterior_fields(endpoint, index)
    G = -stable_logsum_difference(endpoint[:, competitor], intercept[:, competitor])
    H = stable_log_probability_gain(p0["log_odds"], p1["log_odds"], G)
    arithmetic = 128 * torch.finfo(torch.float64).eps * math.log2(support.size + 1) * torch.maximum(
        torch.ones_like(G), torch.maximum(intercept.abs().amax(dim=1), endpoint.abs().amax(dim=1))
    )
    rc, ru = (mc - target).reshape(count, -1), (mu - reference_mean).reshape(count, -1)
    branch_gap_error = projected_branch_gap_error(delta, (target - reference_mean).reshape(count, -1))
    direction_ok = D > 128 * torch.finfo(torch.float64).eps * (
        _norm(mc, target.ndim) + _norm(mu, target.ndim)
    )
    payload = {
        "schema_version": 1, "bank_hash": law.law_hash, "target_atom": index,
        "dimension": d, "beta": beta, "guidance": g, "kappa": coeff.kappa,
        "intercept": intercept, "slopes": slopes, "current_weights": current_weights,
        "delta_norm": D, "delta_exact_zero": (delta == 0).all(dim=1),
        "conditional_error": _norm(rc, 1),
        "unconditional_reference_error": _norm(ru, 1),
        "reference_branch_gap": _norm(target - reference_mean, target.ndim),
        "branch_gap_error": branch_gap_error,
        "branch_gap_error_norm_diagnostic": _norm(delta - (target - reference_mean).reshape(count, -1), 1),
        "branch_gap_error_definition": BRANCH_GAP_ERROR_DEFINITION,
        "branch_gap_error_zero_convention": BRANCH_GAP_ZERO_CONVENTION,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "combined_reference_error": branch_gap_error,
        "signed_error_projection": -branch_gap_error,
        "source_error_l2": source_error, "direction_resolved": direction_ok,
        "direct_log_probability_gain": H, "gain_arithmetic_tolerance": arithmetic,
        "reconstruction_valid": reconstruction_valid,
    }
    exact_zero = (D == 0) & (observed_flat == cf).all(dim=1)
    gain_status = ["zero" if zero else "positive" if gain > tol else "negative" if gain < -tol else "numerically_unresolved"
                   for gain, tol, zero in zip(G.tolist(), arithmetic.tolist(), exact_zero.tolist())]
    result.update(
        direct_prop5_applicable=True, direct_prop5_status="matched_finite_law_comparison",
        direct_prop5_log_probability_gain=H, direct_prop5_log_odds_gain=G,
        direct_prop5_gain_status=gain_status, direct_prop5_gain_arithmetic_tolerance=arithmetic,
        direct_prop5_zero_displacement=D == 0,
        direct_prop5_matched_reconstruction_valid=reconstruction_valid,
        direct_prop5_gain_underflow=(H == 0) & (G.abs() > arithmetic),
        direct_prop5_gain_saturated=(p0["log_probability"].exp() == 0) | (p0["log_probability"].exp() == 1)
        | (p1["log_probability"].exp() == 0) | (p1["log_probability"].exp() == 1),
        direct_prop5_integration_absolute_tolerance=config.absolute_tolerance,
        direct_prop5_integration_relative_tolerance=config.relative_tolerance,
        direct_prop5_integration_max_evaluations=config.max_evaluations,
    )
    for prefix, fields in (("matched", p0), ("guided", p1)):
        result.update({"direct_prop5_" + prefix + "_" + key: value for key, value in fields.items()})
    if payload_callback is not None:
        try:
            payload_callback(payload, dict(result))
        except Exception as error:
            # A failed receipt must not erase endpoint observations already
            # computed from the source record. The outer audit remains failed.
            result["direct_prop5_status"] = "failed_integration"
            result["direct_prop5_integral_status"] = ["failed"] * count
            result["direct_prop5_condition_status"] = ["failed"] * count
            result["direct_prop5_failure_phase"] = "payload_persistence"
            result["direct_prop5_integration_error"] = type(error).__name__ + ": " + str(error)
            return result
    if not integrate:
        result["direct_prop5_status"] = "integration_required_not_computed"
        result["direct_prop5_integral_status"] = ["not_computed"] * count
        return result
    return integrate_proposition5_payload(support, payload, result, config=config)


def theorem7_metrics(mu, mc, target, observed, law, reference, terminal, guidance, tolerance=None):
    """Keep the raw endpoint comparison even when the clean-update gate fails."""
    g, d, ndim = float(guidance), target.numel(), target.ndim
    ec, branch_gap_error, tail = (reference[name] for name in (
        "direct_conditional_error_l2", "direct_branch_gap_error_l2", "direct_radius_tail_l2"))
    bound = ec + (g - 1) * (branch_gap_error + tail)
    error = _norm(observed - target, ndim)
    structural = bool(terminal["terminal_clean_structural"])
    numerical = torch.as_tensor(terminal["terminal_clean_numeric_within_sensitivity"], dtype=torch.bool, device=error.device)
    valid = torch.isfinite(bound) & torch.isfinite(error)
    applicable = valid & numerical & structural & (g > 1)
    result = {
        "direct_theorem7_applicable": applicable,
        "direct_theorem7_measurement_contract": MEASUREMENT_CONTRACT,
        "direct_theorem7_structural_clean": structural,
        "direct_theorem7_numeric_within_sensitivity": numerical,
        "direct_theorem7_numeric_tolerance_rmse": terminal["terminal_clean_numeric_tolerance_rmse"],
        "direct_theorem7_scheduler_residual_l2": terminal["scheduler_rho_l2"],
        "direct_theorem7_scheduler_residual_rmse": terminal["scheduler_rho_rmse"],
        "direct_theorem7_structural_reason": terminal["terminal_clean_structural_reason"],
        "direct_theorem7_status": ["applicable_branch_gap_error_refined_bound" if applies else
            "inapplicable_clean_update:" + str(terminal["terminal_clean_structural_reason"]) if not structural else
            "outside_manuscript_guidance_domain_g_gt_1" if g <= 1 else
            "missing_reference_quantity" if not finite else "numerical_clean_update_unresolved"
            for applies, finite in zip(applicable.tolist(), valid.tolist())],
        "direct_theorem7_bound_error_ratio": bound / torch.where(error > 0, error, torch.nan),
        "direct_theorem7_zero_error_ratio_undefined": error == 0,
        "direct_theorem7_observed_slack_l2": bound - error,
        "direct_theorem7_applicable_slack_l2": torch.where(applicable, bound - error, torch.nan),
        "direct_theorem7_latent_tolerance_l2": tolerance,
        "direct_theorem7_radius_tail_underflow": reference["direct_radius_tail_underflow"],
        "direct_theorem7_condition_certified": [None] * len(error) if tolerance is None else applicable & ~reference["direct_radius_tail_underflow"] & (bound <= float(tolerance)),
        "direct_theorem7_actual_proximity": [None] * len(error) if tolerance is None else error <= float(tolerance),
        "direct_theorem7_certification_scope": "branch_gap_error_refined_sufficient_event_given_applicability_and_reference_law; scalar_estimate_not_interval_certificate",
    }
    for name, value in (("bound", bound), ("endpoint_error", error), ("conditional_term", ec),
                        ("branch_gap_error_term", (g - 1) * branch_gap_error), ("concentration_term", (g - 1) * tail)):
        _put_norm(result, "direct_theorem7_" + name, value, d)
    for key in ("terminal_eq16_vector_residual_rmse", "terminal_eq16_squared_l2_from_terms",
                "terminal_cross_term_squared_l2", "terminal_cross_term_per_dimension",
                "terminal_A_l2", "terminal_B_l2", "terminal_clean_error_l2"):
        result["direct_theorem7_" + key] = terminal[key]
    return result


def _rows(metrics, identities):
    count = len(identities)
    columns = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                value = value.item()
            elif value.shape == (count,):
                value = value.tolist()
            else:
                raise ValueError(f"Nonscalar direct measurement leaked into table: {key}")
        columns[key] = value if isinstance(value, list) and len(value) == count else [value] * count
    return [{**identity, **{key: value[i] for key, value in columns.items()}} for i, identity in enumerate(identities)]


def measure_direct_record(log, record, law, config, *, adapter, terminal_sscd=None,
                          integration=True, integration_config=None, payload_callback=None):
    """Reduce a preloaded record once; failures preserve every sample identity."""
    z, u, c, target = log
    meta = record.metadata if hasattr(record, "metadata") else record
    if meta.get("stored_prediction_type", "epsilon") != "epsilon":
        raise ValueError("Direct cached measurements require already-canonical epsilon")
    seeds = list(meta["seeds"])
    steps = int(config["num_inference_steps"])
    if z.shape[:2] != (len(seeds), steps + 1) or u.shape != c.shape or u.shape[:2] != (len(seeds), steps):
        raise ValueError("Direct record requires T+1 states and T corresponding branch predictions")
    if tuple(z.shape[2:]) != tuple(target.shape) or tuple(u.shape[2:]) != tuple(target.shape):
        raise ValueError("Direct record latent dimensions differ")
    if terminal_sscd is None:
        scores = [None] * len(seeds)
    else:
        supplied_scores = terminal_sscd.detach().cpu().tolist() if isinstance(terminal_sscd, torch.Tensor) else terminal_sscd
        # SSCD is a copied observation, not an input to float32 tensor math.
        # Preserve Python/NumPy float64 values and explicit unavailable scores.
        scores = [None if value is None else float(value) for value in supplied_scores]
    if len(scores) != len(seeds):
        raise ValueError("Same-seed terminal SSCD coverage differs")
    device = law.support.flat.device
    target = target.to(device=device, dtype=torch.float64)
    target_id = law.target_id_for(target, required=False)
    g = float(config["guidance_scale"])
    mean = law.theory_mean_vector
    prompt_digest = hashlib.sha256(str(meta.get("prompt_raw", "")).encode("utf-8")).hexdigest()
    ambiguous_prompts = {
        item["prompt_utf8_sha256"]
        for item in law.metadata.get("prompt_assumption_audit", {}).get("ambiguities", [])
    }
    all_rows, failures = [], []
    pending_integrals = 0
    chunk = law.support.query_chunk
    for step in range(steps):
        coeff = adapter.coefficients(step)
        for start in range(0, len(seeds), chunk):
            stop = min(start + chunk, len(seeds))
            state, next_state, eu, ec = [value.to(device) for value in (
                z[start:stop, step], z[start:stop, step + 1], u[start:stop, step], c[start:stop, step])]
            mu, mc, _, _ = clean_estimates(state, eu, ec, coeff.alpha, coeff.sigma, g, latent_ndim=target.ndim)
            reference, weights, posterior_mean = current_reference_metrics(
                state, mu, mc, target, law, target_id, coeff.alpha, coeff.sigma)
            matched_metrics, matched, _ = adapter.direct_matched_update(
                state, next_state, eu, ec, g, step, latent_ndim=target.ndim)
            source_eps = max(torch.finfo(value.dtype).eps for value in (state, next_state, eu, ec))
            source_error = 96 * source_eps * (
                _norm(state.double(), target.ndim) + coeff.sigma * torch.maximum(
                    _norm(eu.double(), target.ndim), _norm(ec.double(), target.ndim))) / coeff.alpha
            metrics = {**reference, **matched_metrics,
                       "direct_source_sensitivity_l2": source_error,
                       "direct_measurement_version": DIRECT_MEASUREMENT_VERSION}
            def retain_payload(payload, base_metrics):
                nonlocal pending_integrals
                pending_integrals += 1
                if payload_callback is not None:
                    payload_callback(step, start, stop, payload, base_metrics)

            try:
                feedback = proposition5_metrics(
                    state, mu, mc, matched, next_state, target, target_id, law, coeff,
                    weights, posterior_mean, source_error,
                    matched_metrics["direct_lemma4_numeric_within_sensitivity"], g,
                    config=integration_config, integrate=integration, payload_callback=retain_payload)
            except Exception as error:
                feedback = _unavailable_prop5(stop - start, device, "failed_integration", status="failed")
                failures.append({"step_index": step, "seeds": seeds[start:stop],
                                 "error_type": type(error).__name__, "error": str(error)})
            if feedback.get("direct_prop5_status") == "failed_integration" and "direct_prop5_integration_error" in feedback:
                failures.append({"step_index": step, "seeds": seeds[start:stop],
                                 "error": feedback["direct_prop5_integration_error"]})
            metrics.update(feedback)
            if step == 0:
                metrics.update(corollary3_metrics(mu, mc, target, mean, g))
            if step == steps - 1:
                terminal = adapter.terminal_diagnostics(state, next_state, eu, ec, g, step, target)
                metrics.update(theorem7_metrics(mu, mc, target, next_state.double(), law, reference, terminal, g,
                                               config.get("target_error_tolerance")))
            identities = [{
                "run_id": str(meta["scientific_config_hash"]), "original_index": str(meta["original_index"]),
                "record_id": str(meta["record_id"]), "target_id": str(meta["target_image_sha256"]),
                "target_latent_sha256": meta.get("tensor_file_sha256", {}).get("target_latent"),
                "candidate_target_atom_id": None if target_id is None else law.support.atom_ids[law.support.aliases[target_id]],
                "seed": seeds[i], "step_index": step, "timestep": coeff.timestep,
                "snr": coeff.snr, "alpha": coeff.alpha, "sigma": coeff.sigma,
                "destination_timestep": coeff.destination_timestep,
                "destination_alpha": coeff.destination_alpha, "destination_sigma": coeff.destination_sigma,
                "destination_snr": coeff.destination_alpha**2 / coeff.destination_sigma**2 if coeff.destination_sigma else math.inf,
                "A": coeff.A, "kappa": coeff.kappa, "destination_noise_std": coeff.noise_std,
                "latent_dimension": target.numel(), "terminal_sscd": scores[i],
                "input_source": "generated_state", "reference_law_hash": law.law_hash,
                "stored_prediction_type": "epsilon", "reduction_dtype": "torch.float64",
                "source_state_dtype": str(z.dtype),
                "source_unconditional_dtype": str(u.dtype), "source_conditional_dtype": str(c.dtype),
                "reference_law_scope": law.metadata["reference_law_scope"],
                "conditional_reference_scope": "assumed_single_target_conditional_idealization",
                "prompt_utf8_sha256": prompt_digest,
                "conditional_reference_observed_prompt_ambiguous": prompt_digest in ambiguous_prompts,
            } for i in range(start, stop)]
            all_rows.extend(_rows(metrics, identities))
    frame = pd.DataFrame(all_rows)
    common = [name for name in frame if not name.startswith("direct_")]
    def select(prefixes):
        return list(dict.fromkeys(common + [name for name in frame if name.startswith(prefixes)]))
    return {
        "initial": frame.loc[frame.step_index == 0].copy(),
        "trajectory": frame[select(("direct_lemma6_", "direct_conditional_", "direct_unconditional_", "direct_reference_", "direct_radius_", "direct_target_", "direct_non_target_", "direct_branch_gap_", "direct_source_"))].copy(),
        "matched_updates": frame[select(("direct_lemma4_", "direct_prop5_"))].copy(),
        "terminal": frame.loc[frame.step_index == steps - 1].copy(),
        "audit": {"complete": not failures, "core_complete": not failures,
                  "integration_complete": not failures and (integration or pending_integrals == 0),
                  "no_pending_integrals": pending_integrals == 0,
                  "integral_payload_count": pending_integrals, "failed_updates": failures,
                  "rows": len(frame), "seeds": seeds, "prediction_steps": steps,
                  "reference_law_hash": law.law_hash,
                  "integration_recipe": {**DIRECT_INTEGRATION_RECIPE,
                                         **asdict(DEFAULT_INTEGRATION if integration_config is None else integration_config)},
                  "single_target_assumption": law.metadata.get("single_target_assumption"),
                  "prompt_assumption_audit": law.metadata.get("prompt_assumption_audit", {}),
                  "execution_status": "runtime_measurements_requested; code_creation_itself_performs_no_measurements"},
    }
