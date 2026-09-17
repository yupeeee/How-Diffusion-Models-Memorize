"""Float64 reductions on canonical-epsilon trajectory caches; never model inference."""

from __future__ import annotations

import math
from typing import Any

import torch

EFFECTIVE_TARGET_RESIDUAL_EVIDENCE = "exact_cache_algebra"
NUMERICAL_POLICY = (
    "promote_to_float64_before_arithmetic; storage/inference_rounding_not_reversed"
)


MEASUREMENT_FORMULA_VERSION = "manuscript-measurements-projected-gap-error-1"


def _double(value: torch.Tensor, device=None) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=torch.float64)


def _coefficient(value, reference: torch.Tensor, latent_ndim: int):
    result = _double(value, reference.device)
    if result.ndim:
        result = result.reshape(*result.shape, *([1] * latent_ndim))
    return result


def clean_estimates(z, epsilon_u, epsilon_c, alpha, sigma, guidance, *, latent_ndim=3):
    """Return m_u, m_c, D, m_g. Stored branches are ALREADY canonical epsilon."""
    z = _double(z)
    eu, ec = _double(epsilon_u, z.device), _double(epsilon_c, z.device)
    a = _coefficient(alpha, z, latent_ndim)
    s = _coefficient(sigma, z, latent_ndim)
    valid = torch.isfinite(a) & torch.isfinite(s) & (a > 0) & (s >= 0)
    safe_a = torch.where(valid, a, torch.nan)
    mu, mc = (z - s * eu) / safe_a, (z - s * ec) / safe_a
    difference = (s / safe_a) * (eu - ec)
    return mu, mc, difference, mu + float(guidance) * difference


def _norm_metrics(result, name, values, dims, dimension):
    squared = values.square().sum(dim=dims)
    result[name + "_squared_l2"] = squared
    result[name + "_l2"] = squared.sqrt()
    result[name + "_rmse"] = (squared / dimension).sqrt()


def branch_metrics(
    z, epsilon_u, epsilon_c, target, alpha, sigma, guidance, center=None
) -> dict[str, Any]:
    """Reduce arbitrary leading query dimensions followed by target.shape.

    Target errors are finite-noise observations. The effective target residual
    equality and D reconstruction are algebra QA, never a population loss.
    """
    source_dtypes = {
        str(torch.as_tensor(x).dtype) for x in (z, epsilon_u, epsilon_c, target)
    }
    source_epsilon = max(
        torch.finfo(torch.as_tensor(x).dtype).eps
        for x in (z, epsilon_u, epsilon_c, target)
    )
    z = _double(z)
    target = _double(target, z.device)
    eu, ec = _double(epsilon_u, z.device), _double(epsilon_c, z.device)
    if (
        z.shape != eu.shape
        or z.shape != ec.shape
        or tuple(z.shape[-target.ndim :]) != tuple(target.shape)
    ):
        raise ValueError(
            "Current states, canonical branches, and target shapes are incompatible"
        )
    dims = tuple(range(-target.ndim, 0))
    d = target.numel()
    a, s = _coefficient(alpha, z, target.ndim), _coefficient(sigma, z, target.ndim)
    mu, mc, difference, mg = clean_estimates(
        z, eu, ec, alpha, sigma, guidance, latent_ndim=target.ndim
    )
    error_u, error_c = mu - target, mc - target
    result: dict[str, Any] = {
        "measurement_formula_version": MEASUREMENT_FORMULA_VERSION,
        "source_dtypes": ",".join(sorted(source_dtypes)),
        "source_dtype_epsilon": source_epsilon,
        "latent_dimension": d,
        "branch_status": (
            "valid"
            if bool(torch.isfinite(mu).all() & torch.isfinite(mc).all())
            else "invalid_alpha_or_nonfinite_input"
        ),
        "branch_evidence": "finite_noise_observation",
        "effective_target_residual_evidence": EFFECTIVE_TARGET_RESIDUAL_EVIDENCE,
        "effective_target_identity_evidence": "algebraic_qa",
        "effective_target_residual_evidence_is_legacy_alias": True,
        "unconditional_center_evidence": (
            "model_center_diagnostic" if center is not None else "unavailable"
        ),
        "injection_evidence": (
            "model_center_diagnostic" if center is not None else "unavailable"
        ),
        "population_forward_loss": None,
        "population_forward_loss_status": "unavailable_from_preserved_cache: reverse_state_is_not_independent_forward_corruption",
    }
    for name, value in (
        ("state", z),
        ("state_error", z - target),
        ("e_u", error_u),
        ("e_c", error_c),
        ("e_g", mg - target),
        ("branch_gap", difference),
        ("epsilon_gap", eu - ec),
    ):
        _norm_metrics(result, name, value, dims, d)
    # e_u/e_c remain compatibility aliases, never reference errors.
    for branch, old in (
        ("unconditional", "e_u"),
        ("conditional", "e_c"),
        ("guided", "e_g"),
    ):
        for unit in ("l2", "squared_l2", "rmse"):
            result[f"{branch}_target_error_{unit}"] = result[f"{old}_{unit}"]
    result["conditional_target_error_reference_scope"] = (
        "equals_conditional_reference_error_only_under_single_target_idealization_not_verified_by_pair"
    )
    result["target_l2"] = target.norm()
    result["target_rmse"] = target.norm() / math.sqrt(d)
    result["target_squared_l2"] = target.square().sum()
    result["error_cross_inner_product_per_dimension"] = (error_u * error_c).sum(
        dim=dims
    ) / d
    result["joint_rmse"] = torch.maximum(result["e_u_rmse"], result["e_c_rmse"])
    result["conditional_error_reduction_rmse"] = result["e_u_rmse"] - result["e_c_rmse"]
    for unit in ("l2", "squared_l2", "rmse"):
        result["joint_target_error_" + unit] = torch.maximum(
            result["unconditional_target_error_" + unit],
            result["conditional_target_error_" + unit],
        )
    result["paired_target_error_improvement_rmse"] = result[
        "conditional_error_reduction_rmse"
    ]
    result["paired_target_error_improvement_l2"] = result["e_u_l2"] - result["e_c_l2"]
    positive_denominator = result["e_u_rmse"] > 0
    result["relative_target_error_improvement"] = result[
        "paired_target_error_improvement_rmse"
    ] / torch.where(positive_denominator, result["e_u_rmse"], torch.nan)
    result["relative_target_error_improvement_denominator_valid"] = positive_denominator
    result["relative_target_error_improvement_undefined_zero_denominator"] = (
        result["e_u_rmse"] == 0
    )
    result["branch_difference_identity_rmse"] = (
        (mc - mu - difference).square().sum(dim=dims) / d
    ).sqrt()
    _norm_metrics(result, "unconditional_zero_center", mu, dims, d)
    aa, ss = _double(alpha, z.device), _double(sigma, z.device)
    result["snr"] = torch.where(
        ss > 0, aa.square() / ss.square(), torch.tensor(float("inf"), device=z.device)
    )
    result["inverse_alpha"] = torch.where(aa > 0, 1 / aa, torch.nan)
    result["sigma_over_alpha"] = torch.where(aa > 0, ss / aa, torch.nan)
    residual = ec - (z - a * target) / torch.where(s > 0, s, torch.nan)
    result["effective_target_noise_residual_mse"] = residual.square().sum(dim=dims) / d
    identity_rhs = result["effective_target_noise_residual_mse"] / result["snr"]
    result["effective_target_identity_residual"] = (
        result["e_c_squared_l2"] / d - identity_rhs
    )
    result["effective_target_identity_status"] = (
        "valid" if bool(((aa > 0) & (ss > 0)).all()) else "invalid_zero_alpha_or_sigma"
    )
    # Conservative first-order sensitivity indicator, NOT a rigorously certified
    # model inference error: native predictions are no longer available.
    prediction_scale = torch.maximum(
        (eu.square().sum(dim=dims) / d).sqrt(), (ec.square().sum(dim=dims) / d).sqrt()
    )
    result["clean_estimate_rounding_sensitivity_rmse"] = (
        32 * source_epsilon * (result["state_rmse"] + ss * prediction_scale) / aa
    )
    result["numerical_budget_status"] = (
        "dtype_aware_first_order_sensitivity_not_certified_inference_error"
    )
    injection = float(guidance) * difference
    _norm_metrics(result, "injection", injection, dims, d)
    if center is None:
        for key in (
            "unconditional_center_rmse",
            "conditional_center_rmse",
            "target_center_l2",
            "target_center_rmse",
            "a_parallel",
            "off_target",
            "target_direction_cosine",
            "injection_perpendicular_rmse",
            "injection_total_residual_rmse",
            "injection_relative_mismatch",
        ):
            result[key] = torch.full_like(result["e_u_rmse"], torch.nan)
        result["projection_status"] = "unavailable_center"
        result["injection_relative_mismatch_status"] = "unavailable_center"
        return result
    center = _double(center, z.device)
    if center.shape != target.shape:
        raise ValueError("The fixed center and target must have identical shapes")
    _norm_metrics(result, "unconditional_center", mu - center, dims, d)
    _norm_metrics(result, "conditional_center", mc - center, dims, d)
    vector = target - center
    vector_sq = vector.square().sum()
    vector_norm = vector_sq.sqrt()
    # Fixed source-precision screen, not a certified inference-error bound.
    direction_scale = max(1.0, float(target.norm()), float(center.norm()))
    direction_tolerance = 8 * source_epsilon * direction_scale
    direction_valid = bool(
        torch.isfinite(vector_norm) & (vector_norm > direction_tolerance)
    )
    result["projection_direction_tolerance_l2"] = direction_tolerance
    result["projection_direction_tolerance_policy"] = (
        "8_source_dtype_eps_times_max_1_target_norm_center_norm_screening_not_certified"
    )
    result["target_center_l2"], result["target_center_rmse"] = (
        vector_norm,
        vector_norm / math.sqrt(d),
    )
    dot = (injection * vector).sum(dim=dims)
    result["injection_target_inner_product"] = dot
    result["a_parallel"] = dot / (vector_sq if direction_valid else torch.nan)
    expanded = result["a_parallel"].reshape(
        *result["a_parallel"].shape, *([1] * target.ndim)
    )
    perp_sq = (injection - expanded * vector).square().sum(dim=dims)
    result["off_target"] = perp_sq.sqrt() / torch.where(
        torch.tensor(direction_valid, device=z.device), vector_norm, torch.nan
    )
    result["injection_perpendicular_rmse"] = (perp_sq / d).sqrt()
    denominator = result["injection_l2"] * vector_norm
    result["target_direction_cosine"] = dot / torch.where(
        (denominator > 0) & direction_valid, denominator, torch.nan
    )
    result["injection_total_residual_rmse"] = (
        (injection - float(guidance) * vector).square().sum(dim=dims) / d
    ).sqrt()
    result["projection_status"] = (
        "valid" if direction_valid else "degenerate_target_center_direction"
    )
    positive_guidance = math.isfinite(float(guidance)) and float(guidance) > 0
    result["injection_relative_mismatch"] = torch.sqrt(
        (result["a_parallel"] - float(guidance)).square()
        + result["off_target"].square()
    ) / (float(guidance) if positive_guidance else torch.nan)
    result["injection_relative_mismatch_status"] = (
        "valid"
        if direction_valid and positive_guidance
        else (
            "degenerate_target_center_direction"
            if not direction_valid
            else "inapplicable_nonpositive_or_nonfinite_guidance"
        )
    )
    result["cosine_status"] = torch.where(
        (denominator > 0) & direction_valid, 0, 1
    )  # 1: zero injection or target direction
    return result


def initial_distribution_diagnostics(
    target, alpha, sigma, initial_noise_scale, *, gaussian_initialization_recorded=False
):
    """Eq.31 bound for intended Gaussians, never a measured total variation.

    ``gaussian_initialization_recorded`` asserts the preserved initialization
    contract; it must not be inferred from the observed initial tensor values.
    """
    target = _double(target)
    d, norm_sq = target.numel(), float(target.square().sum())
    a, s, s0 = float(alpha), float(sigma), float(initial_noise_scale)
    valid = (
        all(math.isfinite(v) for v in (a, s, s0, norm_sq))
        and s > 0
        and s0 > 0
        and a >= 0
    )
    result = {
        "initial_transfer_bound_unclipped": None,
        "initial_transfer_bound_clipped": None,
        "initial_transfer_bound_status": "unavailable_invalid_gaussian_parameters",
        "initial_transfer_bound_evidence": "algebraic_qa",
        "initial_transfer_bound_formula": "Appendix_D.1_Eq31:0.5*sqrt(alpha^2*target_l2^2+d*(sigma^2-1-log(sigma^2)))",
        "initial_transfer_bound_scope": "upper_bound_on_intended_Gaussian_TV_not_measured_TV_or_quantized_tensor_law",
        "initial_gaussian_initialization_recorded": bool(
            gaussian_initialization_recorded
        ),
        "initial_forward_to_initial_gaussian_kl": None,
    }
    if not valid:
        result.update(
            initial_signal_energy=None,
            initial_gaussian_kl=None,
            initial_distribution_status="invalid_gaussian_parameters",
        )
        return result
    ratio = s0 * s0 / (s * s)
    signal = a * a * norm_sq / (s * s)
    result.update(
        initial_signal_energy=signal,
        initial_gaussian_kl=0.5 * (d * (ratio - 1 - math.log(ratio)) + signal),
        initial_gaussian_kl_direction="initialization_P_to_forward_Q",
        initial_noise_scale=s0,
        initial_distribution_status="analytic_intended_gaussians_not_quantized_tensor_distribution",
        initial_distribution_evidence="algebraic_qa",
    )
    if not gaussian_initialization_recorded:
        result["initial_transfer_bound_status"] = (
            "unavailable_unverified_gaussian_initialization"
        )
    elif not math.isclose(s0, 1.0, rel_tol=0, abs_tol=1e-12):
        result["initial_transfer_bound_status"] = (
            "inapplicable_nonstandard_initialization_scale"
        )
    elif not math.isclose(a * a + s * s, 1.0, rel_tol=0, abs_tol=2e-6):
        result["initial_transfer_bound_status"] = (
            "inapplicable_non_variance_preserving_noise_contract"
        )
    else:
        variance_offset = s * s - 1
        # log1p retains the small VP variance contribution near zero SNR.
        radicand = a * a * norm_sq + d * (
            variance_offset
            - (
                math.log1p(variance_offset)
                if variance_offset > -0.5
                else 2 * math.log(s)
            )
        )
        value = 0.5 * math.sqrt(max(0.0, radicand))
        result.update(
            initial_forward_to_initial_gaussian_kl=radicand / 2,
            initial_transfer_bound_unclipped=value,
            initial_transfer_bound_clipped=min(1.0, value),
            initial_transfer_bound_status="applicable_intended_standard_Gaussian_initialization",
        )
    return result


def candidate_terminal_bound(
    conditional_error_l2,
    branch_gap_error_l2,
    radius_l2,
    target_log_complement,
    *,
    guidance,
    dimension,
    clean_error_l2,
    endpoint_error_l2,
    terminal_clean_update,
):
    """Projected branch-gap-error terminal bound, with signed raw and normalized error terms.

    No candidate input is claimed to identify the training law. The current
    conditional branch is interpreted under the single-target idealization.
    ``target_log_complement`` is computed by a stable posterior calculation,
    never by subtracting a rounded target probability from one.
    """
    if int(dimension) <= 0:
        raise ValueError("The latent dimension must be positive")
    scale = math.sqrt(int(dimension))
    template = _double(clean_error_l2)
    unavailable = torch.full_like(template, torch.nan)
    result = {
        "candidate_terminal_bound_evidence": "candidate_distribution_diagnostic",
        "candidate_terminal_measurement_contract": "projected-gap-error-1",
        "candidate_terminal_branch_gap_error_definition": "signed_projected_branch_gap_reference_error",
        "candidate_terminal_comparison_scope": "Signed projected error e_parallel=u dot (Delta-bar_Delta), u=Delta/||Delta||; zero gap uses u=0. Clean-update applicability and declared-law assumptions remain required",
        "candidate_terminal_bound_formula": "e_c+(g-1)*(e_parallel+R_K*(1-p_K))",
        "candidate_terminal_bound_reference_scope": "declared_candidate_law_and_single_target_conditional_idealization_not_identified_training_law",
        "candidate_terminal_bound_status": "unavailable_reference",
        "candidate_terminal_bound_applies_to_endpoint": False,
    }
    for stem in (
        "candidate_terminal_conditional_term",
        "candidate_terminal_branch_gap_error_term",
        "candidate_terminal_concentration_term",
        "candidate_terminal_bound",
        "candidate_terminal_clean_bound_slack",
        "candidate_terminal_endpoint_bound_slack",
    ):
        result[stem + "_l2"] = unavailable
        result[stem + "_rmse"] = unavailable
    result.update(
        candidate_terminal_bound_clean_error_ratio=unavailable,
        candidate_terminal_bound_endpoint_error_ratio=unavailable,
        candidate_terminal_bound_clean_zero_error_ratio_undefined=template == 0,
        candidate_terminal_bound_endpoint_zero_error_ratio_undefined=_double(
            endpoint_error_l2, template.device
        )
        == 0,
        candidate_terminal_non_target_mass=unavailable,
        candidate_terminal_reference_radius_l2=unavailable,
        candidate_terminal_reference_radius_rmse=unavailable,
    )
    if any(
        value is None
        for value in (
            conditional_error_l2,
            branch_gap_error_l2,
            radius_l2,
            target_log_complement,
        )
    ):
        return result
    g = float(guidance)
    if not math.isfinite(g) or g <= 1:
        result["candidate_terminal_bound_status"] = (
            "inapplicable_manuscript_guidance_domain_g_gt_1"
        )
        return result
    ec, gap_error, radius, log_complement = [
        _double(value, template.device)
        for value in (
            conditional_error_l2,
            branch_gap_error_l2,
            radius_l2,
            target_log_complement,
        )
    ]
    valid = (
        torch.isfinite(ec)
        & torch.isfinite(gap_error)
        & torch.isfinite(radius)
        & (ec >= 0)
        & (radius >= 0)
        & ~torch.isnan(log_complement)
        & (log_complement <= 0)
    )
    mass = log_complement.exp()
    terms = {
        "candidate_terminal_conditional_term": ec,
        "candidate_terminal_branch_gap_error_term": (g - 1) * gap_error,
        "candidate_terminal_concentration_term": (g - 1) * radius * mass,
    }
    bound = sum(terms.values())
    terms["candidate_terminal_bound"] = bound
    terms["candidate_terminal_clean_bound_slack"] = bound - template
    applicable = torch.as_tensor(
        terminal_clean_update, device=template.device, dtype=torch.bool
    )
    endpoint = _double(endpoint_error_l2, template.device)
    terms["candidate_terminal_endpoint_bound_slack"] = torch.where(
        applicable, bound - endpoint, torch.nan
    )
    for stem, value in terms.items():
        result[stem + "_l2"] = torch.where(valid, value, torch.nan)
        result[stem + "_rmse"] = result[stem + "_l2"] / scale
    result.update(
        candidate_terminal_non_target_mass=torch.where(valid, mass, torch.nan),
        candidate_terminal_reference_radius_l2=torch.where(valid, radius, torch.nan),
        candidate_terminal_reference_radius_rmse=torch.where(
            valid, radius / scale, torch.nan
        ),
        candidate_terminal_bound_clean_error_ratio=torch.where(
            valid & (template > 0), bound / template, torch.nan
        ),
        candidate_terminal_bound_endpoint_error_ratio=torch.where(
            valid & applicable & (endpoint > 0), bound / endpoint, torch.nan
        ),
        candidate_terminal_bound_applies_to_endpoint=valid & applicable,
        candidate_terminal_bound_reference_valid=valid,
        candidate_terminal_bound_status=(
            "candidate_law_terms_for_clean_estimate_endpoint_applicability_separate"
            if bool(valid.all())
            else "unavailable_or_partially_invalid_candidate_reference"
        ),
    )
    return result
