"""Resumable original Proposition 5 integration from saved endpoint payloads.

This stage accepts scalar logits and shared law geometry only, never raw
trajectory tensors, model loaders or denoiser calls.
"""
from __future__ import annotations

import math

import torch

from .candidate_integration import DEFAULT_INTEGRATION, integration_metrics


def integrate_proposition5_payload(support, payload, base_metrics, config=None):
    """Complete original-V observations using saved scalar/Gram inputs only.

    Learned predictions and raw latent states are unnecessary for this stage.
    Endpoint observations survive an integration failure and can be resumed.
    """
    config = DEFAULT_INTEGRATION if config is None else config
    result = dict(base_metrics)
    result.update(
        direct_prop5_status="matched_finite_law_comparison",
        direct_prop5_integration_absolute_tolerance=config.absolute_tolerance,
        direct_prop5_integration_relative_tolerance=config.relative_tolerance,
        direct_prop5_integration_max_evaluations=config.max_evaluations,
    )
    count = len(payload["delta_norm"])
    d = support.dimension
    prefactor = float(payload["beta"]) * float(payload["guidance"]) * float(payload["kappa"])
    D = torch.as_tensor(payload["delta_norm"], dtype=torch.float64, device=support.flat.device)
    H = torch.as_tensor(payload["direct_log_probability_gain"], dtype=torch.float64, device=support.flat.device)
    source_error = torch.as_tensor(payload["source_error_l2"], dtype=torch.float64, device=support.flat.device)
    try:
        integrated = integration_metrics(support, payload, bank_hash=payload["bank_hash"], config=config)
    except Exception as error:
        result["direct_prop5_status"] = "failed_integration"
        result["direct_prop5_integral_status"] = ["failed"] * count
        result["direct_prop5_condition_status"] = ["failed"] * count
        result["direct_prop5_integration_error"] = type(error).__name__ + ": " + str(error)
        return result
    mapping = {
        "margin_l2": "margin_original_l2", "margin_rmse": "margin_original_rmse",
        "variation_l2": "variation_norm_integral_l2", "variation_error_l2": "variation_norm_integral_error_l2",
        "uncertainty_l2": "margin_original_uncertainty_l2", "integrated_gain": "integrated_log_probability_gain",
        "integrated_gain_error": "integrated_log_probability_gain_error",
        "integral_identity_residual": "integral_identity_residual", "integral_identity_allowance": "integral_identity_allowance",
        "gram_roundoff_l2": "gram_roundoff_allowance_l2", "quadrature_evaluations": "quadrature_evaluations",
        "quadrature_refinements": "quadrature_refinements", "quadrature_budget_exhausted": "quadrature_budget_exhausted",
        "integral_status": "integration_status", "integral_reason": "integration_reason",
        "integral_identity_status": "integration_identity_status",
        "estimated_condition_sign": "margin_original_estimated_sign",
    }
    for name, source in mapping.items():
        result["direct_prop5_" + name] = integrated["candidate_" + source]
    margin = result["direct_prop5_margin_l2"]
    result.update(
        direct_prop5_variation_rmse=result["direct_prop5_variation_l2"] / math.sqrt(d),
        direct_prop5_condition_status=["estimated_" + status if status in {"positive", "negative"} else status
                                      for status in integrated["candidate_margin_original_status"]],
        direct_prop5_nonnegative_condition=[m >= 0 if math.isfinite(m) else None for m in margin.tolist()],
        direct_prop5_strict_positive_condition=[m > 0 if math.isfinite(m) else None for m in margin.tolist()],
        direct_prop5_prefactor=prefactor,
        direct_prop5_source_sensitivity_l2=source_error,
        direct_prop5_signed_integral=result["direct_prop5_integrated_gain"] / (prefactor),
        direct_prop5_lower_bound=prefactor * D * margin,
    )
    result["direct_prop5_lower_bound_slack"] = H - result["direct_prop5_lower_bound"]
    return result

