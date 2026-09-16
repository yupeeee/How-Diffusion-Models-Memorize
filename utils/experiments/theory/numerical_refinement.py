"""Versioned fixed-cache arithmetic refinements, separate from legacy scalars.

No model, raw-cache loader, file writer, or scheduler mutation enters this
module. GPU float64 assessments are distinct from CUDA outward binary64
enclosures. No production numerical fallback runs on CPU. A certificate over rounded sufficient inputs is explicitly NOT a
certificate of their construction from the original saved vectors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .numerical_screening import screen_negative_condition
from .gpu_intervals import ArithmeticBudgetExceeded, GpuDirected
from .gpu_refinement import (
    clean_intervals as _clean_intervals, enclose_variation, interval_gain,
    negative_condition_precheck, posterior_mean, raw_row as _raw_row,
    raw_segment as _raw_segment, require_cuda, source_robust_gain_interval,
)

NUMERICAL_VERSION = "fixed-cache-numerics-cuda-3"
PAYLOAD_VERSION = 2


@dataclass(frozen=True)
class NumericalPolicy:
    decimal_precision: int = 64
    max_decimal_products: int = 2_000_000
    max_variation_nodes: int = 65
    variation_absolute_width: float = 1e-6

    def __post_init__(self):
        if (not isinstance(self.decimal_precision, int) or isinstance(self.decimal_precision, bool) or self.decimal_precision < 32
                or not isinstance(self.max_decimal_products, int) or isinstance(self.max_decimal_products, bool) or self.max_decimal_products < 0
                or not isinstance(self.max_variation_nodes, int) or isinstance(self.max_variation_nodes, bool) or self.max_variation_nodes < 1
                or not math.isfinite(self.variation_absolute_width)
                or self.variation_absolute_width <= 0):
            raise ValueError("Invalid numerical-refinement budget")

    def identity(self):
        return {"version": NUMERICAL_VERSION, **asdict(self),
                "backend": "cuda_outward_binary64", "cpu_fallback": False,
                "effective_dtype": "float64", "effective_mantissa_bits": 53,
                "decimal_precision_role": "legacy_configuration_only_no_Decimal_execution",
                "max_decimal_products_role": "compatibility_alias_for_GPU_interval_operation_budget",
                "retry_precision_multiplier": 1,
                "screening": "batched_outward_float64_squared_negative_condition_before_GPU_enclosure",
                "gain_retry_selection": "nonfinite_or_unresolved_stable_gain_or_H_G_disagreement_or_raw_rounded_zero; saturation_and_absolute_subtraction_are_diagnostics",
                "certified_arithmetic": "CUDA outward binary64 primitives and bounded series transcendental enclosures; fixed precision",
                "budget_scope": "charged CUDA interval scalar primitives per row; device transport and shared input storage separate"}


def _flat(value, device):
    result = torch.as_tensor(value, dtype=torch.float64, device=device)
    return result.reshape(len(result), -1)


def _norm(value):
    return torch.linalg.vector_norm(value, dim=-1)


def _target_logits(support, query, target_atom, alpha, sigma):
    target = support.flat[target_atom]
    chunks = []
    for start in range(0, support.size, support.candidate_chunk):
        stop = min(start + support.candidate_chunk, support.size)
        geometry = support.flat[start:stop] - target
        prior = support.weights[start:stop].log() - support.weights[target_atom].log()
        chunks.append((alpha / sigma**2) * ((query - alpha * target) @ geometry.T)
                      - (alpha**2 / (2 * sigma**2)) * geometry.square().sum(1) + prior)
    return torch.cat(chunks, dim=1)


def build_refinement_payload(support, verified_inputs, *, bank_hash, target_atom):
    """Finite sufficient tensors for saved AND affine endpoint contracts.

    verified_inputs stores exact saved state/epsilon tensors and saved scalar
    coefficients. It is transient; callers need not duplicate raw vectors in
    the compact payload. Dtype perturbations and coefficient rederivation are
    outside this exact-input contract. Canonical epsilon is never reconverted.
    """
    x = verified_inputs
    if x.get("stored_prediction_type", "epsilon") != "epsilon":
        raise ValueError("Refinement requires already-canonical saved epsilon")
    device = require_cuda(support.flat.device)
    state, epsilon_u, epsilon_c, observed = [_flat(x[key], device) for key in (
        "state", "epsilon_u", "epsilon_c", "saved_endpoint")]
    target = torch.as_tensor(x["target"], dtype=torch.float64, device=device).reshape(-1)
    if not 0 <= int(target_atom) < support.size or target.shape != (support.dimension,):
        raise ValueError("Refinement target/support dimensions differ")
    if not torch.equal(target, support.flat[int(target_atom)]):
        raise ValueError("Refinement target is not the declared exact law atom")
    if any(value.shape != state.shape for value in (epsilon_u, epsilon_c, observed)) or state.shape[1] != support.dimension:
        raise ValueError("Refinement cached vector shapes differ")
    names = ("alpha", "sigma", "A", "kappa", "guidance", "destination_alpha", "destination_sigma", "noise_std")
    coefficients = {name: float(x[name]) for name in names}
    if not all(math.isfinite(value) for value in coefficients.values()):
        raise ValueError("Nonfinite saved scheduler coefficient")
    alpha, sigma, A, kappa, g, next_alpha, next_sigma, rho = (coefficients[name] for name in names)
    if min(alpha, sigma, kappa, next_alpha, next_sigma) <= 0 or g <= 1 or rho < 0:
        raise ValueError("Refinement is outside the manuscript affine positive-noise domain")
    mu, mc = (state - sigma * epsilon_u) / alpha, (state - sigma * epsilon_c) / alpha
    delta = mc - mu
    affine_h = g * kappa * delta
    if x.get("independent_innovation") is not None:
        if x.get("innovation_provenance") not in {"independently_saved_additive_noise", "independently_saved_rng_replay"}:
            raise ValueError("Innovation provenance must be independent of the saved endpoint")
        innovation = _flat(x["independent_innovation"], device)
        if innovation.shape != state.shape or (rho == 0 and bool((innovation != 0).any())):
            raise ValueError("Independent innovation differs from the saved noise contract")
        baseline = A * state + kappa * mu + innovation
        method = "independent_shared_innovation_affine_baseline"
    elif rho == 0:
        baseline = A * state + kappa * mu
        method = "independent_deterministic_affine_baseline"
    else:
        baseline = observed - affine_h
        method = "constructed_shared_innovation_from_saved_endpoint_not_independent_replay"
    beta = next_alpha / next_sigma**2
    current_logits = _target_logits(support, state, int(target_atom), alpha, sigma)
    current_weights = current_logits.softmax(1)
    reference = target.expand_as(state).clone()
    affine_slopes, saved_slopes = [], []
    radius = 0.0
    for start in range(0, support.size, support.candidate_chunk):
        stop = min(start + support.candidate_chunk, support.size)
        geometry = support.flat[start:stop] - target
        reference = reference + current_weights[:, start:stop] @ geometry
        affine_slopes.append(beta * (affine_h @ geometry.T))
        saved_slopes.append(beta * ((observed - baseline) @ geometry.T))
        radius = max(radius, float(_norm(geometry).max()))
    result = {"schema_version": PAYLOAD_VERSION, "bank_hash": str(bank_hash),
              "target_atom": int(target_atom), "dimension": support.dimension,
              "intercept": _target_logits(support, baseline, int(target_atom), next_alpha, next_sigma),
              "slopes": torch.cat(affine_slopes, dim=1),
              "saved_endpoint_slopes": torch.cat(saved_slopes, dim=1),
              "current_logits": current_logits, "current_weights": current_weights,
              "delta_norm": _norm(delta), "conditional_error": _norm(mc - target),
              "unconditional_reference_error": _norm(mu - reference),
              "endpoint_difference_l2": _norm(observed - (baseline + affine_h)),
              "target_radius_l2": radius,
              "beta": beta, "guidance": g, "kappa": kappa,
              "endpoint_construction_method": method,
              "coefficient_scope": "exact_saved_scalar_coefficients; no native-coefficient perturbation claim",
              "fixed_input_scope": "original_saved_state_canonical_epsilon_target_atoms_positive_masses_and_coefficients",
              "manuscript_domain": bool(x.get("manuscript_domain", True)),
              "structural_applicable": True}
    for name, value in result.items():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite sufficient tensor: " + name)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite sufficient scalar: " + name)
    return result


def stable_gain(intercept, slopes, target_atom):
    """Batched float64 assessment using centered finite-law gains.

    Small slopes use separately accumulated positive/negative expm1 terms.
    The tolerance is a flagged numerical assessment, NOT an error enclosure.
    H is saturation-aware and retains its own magnitude even when it underflows.
    """
    b = torch.as_tensor(intercept, dtype=torch.float64)
    a = torch.as_tensor(slopes, dtype=torch.float64, device=b.device)
    if b.ndim != 2 or b.shape != a.shape or not 0 <= int(target_atom) < b.shape[1]:
        raise ValueError("Invalid gain shape/target")
    if not bool(torch.isfinite(b).all() & torch.isfinite(a).all()):
        raise ValueError("Gain inputs must be finite")
    batch, size = b.shape
    if size == 1:
        zero = torch.zeros(batch, dtype=torch.float64, device=b.device)
        return {"H": zero, "G": torch.full_like(zero, torch.nan), "tolerance": zero, "H_tolerance": zero,
                "old_difference": torch.full_like(zero, torch.nan),
                "log_odds0": torch.full_like(zero, torch.inf), "log_odds1": torch.full_like(zero, torch.inf),
                "flagged": torch.zeros(batch, dtype=torch.bool, device=b.device),
                "zero": torch.ones(batch, dtype=torch.bool, device=b.device)}
    relative_b = b - b[:, int(target_atom), None]
    relative_a = a - a[:, int(target_atom), None]
    keep = torch.arange(size, device=b.device) != int(target_atom)

    def log_expectation_gain(logits, slopes):
        centered = logits - logits.amax(1, keepdim=True)
        log_weights = centered - torch.logsumexp(centered, 1, keepdim=True)
        weights = log_weights.exp()
        shift = slopes.amax(1)
        gain = -(shift + torch.logsumexp(log_weights + slopes - shift[:, None], 1))
        small = slopes.abs().amax(1) <= .5
        # The unused branch receives zero, never a clipped measured slope.
        expm1 = torch.expm1(torch.where(small[:, None], slopes, torch.zeros_like(slopes)))
        terms = weights * expm1
        positive, negative = terms.clamp_min(0).sum(1), terms.clamp_max(0).sum(1)
        gain = torch.where(small, -torch.log1p(positive + negative), gain)
        eps = torch.finfo(torch.float64).eps
        tolerance = 64 * eps * math.log2(size + 1) * (1 + slopes.abs().amax(1))
        tolerance = torch.where(small, 64 * eps * math.log2(size + 1) * (positive - negative + gain.abs()), tolerance)
        return gain, tolerance

    b, a = relative_b[:, keep], relative_a[:, keep]
    G, tolerance = log_expectation_gain(b, a)
    # Independent full-posterior normalization: H=-log E_{all atoms at z0}
    # exp(a_j), with a_target=0. It never consumes G or its sign/magnitude.
    H, H_tolerance = log_expectation_gain(relative_b, relative_a)
    c0, c1 = torch.logsumexp(b, 1), torch.logsumexp(b + a, 1)
    zero = (a == 0).all(1)
    G, H = torch.where(zero, torch.zeros_like(G), G), torch.where(zero, torch.zeros_like(H), H)
    old_difference = c0 - c1
    flagged = (~torch.isfinite(G) | ~torch.isfinite(H)
               | ((G.abs() <= tolerance) & ~zero)
               | ((old_difference - G).abs() > tolerance)
               | ((H == 0) & (G != 0))
               | ((H.abs() > H_tolerance) & (G.abs() > tolerance) & (torch.sign(H) != torch.sign(G))))
    return {"H": H, "G": G, "tolerance": tolerance, "H_tolerance": H_tolerance, "old_difference": old_difference,
            "log_odds0": -c0, "log_odds1": -c1, "flagged": flagged, "zero": zero}


def _store_interval(result, name, value):
    if value is None:
        result[name + "_lower"], result[name + "_upper"] = math.nan, math.nan
    else:
        result[name + "_lower"], result[name + "_upper"] = value.floats()


def _number(row, name, default=math.nan):
    value = row.get(name, default)
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _base_records(base_rows, count):
    if hasattr(base_rows, "to_dict"):
        rows = base_rows.to_dict("records")
    elif isinstance(base_rows, dict):
        rows = [{key: value[i] if isinstance(value, (list, tuple)) else value
                 for key, value in base_rows.items()} for i in range(count)]
    else:
        rows = list(base_rows)
    if len(rows) != count:
        raise ValueError("Refinement scalar rows do not match payload seeds")
    return rows


def _host_columns(columns):
    """Transfer only compact row diagnostics, once per device batch."""
    names = tuple(columns)
    packed = torch.stack([columns[name].to(dtype=torch.float64) for name in names], dim=1).detach().cpu().tolist()
    return {name: [row[index] for row in packed] for index, name in enumerate(names)}


def _gain_retry_needed(values, index, *, original_inputs, singleton):
    """Select inconclusive stable gains, retaining harmless cross-check flags.

    This selector does not turn the float64 assessment into a certificate.
    A zero H from saturation can keep its independently assessed G sign. The
    unstable absolute-logit subtraction is never the only gain evaluator.
    """
    if singleton:
        return False
    G, H = values["G"][index], values["H"][index]
    if not math.isfinite(G) or not math.isfinite(H):
        return True
    if original_inputs and values["zero"][index]:
        return True  # Rounded zero slopes do not prove an original-input zero.
    unresolved = not values["zero"][index] and abs(G) <= values["tolerance"][index]
    disagreement = (abs(H) > values["H_tolerance"][index] and abs(G) > values["tolerance"][index]
                    and (H > 0) != (G > 0))
    return bool(unresolved or disagreement)


def _posterior_work_floor(size, dimension, *, original_inputs):
    """Lower bound on this policy's unavoidable charged projection work.

    This is a conservative lower bound used only for work selection; actual
    CUDA interval primitives charge their own vectorized operation counts.
    """
    return 4 * size * dimension


def refine_payload(support, payload, base_rows, *, policy=None, verified_inputs=None):
    """Return only additive scalar records in original payload seed order.

    Fast signs are assessments. Directed certificates are explicitly qualified
    as original saved inputs (raw witness supplied) or reduced-payload-only.
    Heuristic source sensitivity NEVER broadens fixed-input error intervals.
    """
    policy = NumericalPolicy() if policy is None else policy
    require_cuda(support.flat.device)
    if payload.get("schema_version") not in {1, PAYLOAD_VERSION}:
        raise ValueError("Unsupported numerical sufficient-input schema")
    target = int(payload["target_atom"])
    if payload.get("dimension") != support.dimension or not 0 <= target < support.size:
        raise ValueError("Refinement payload law geometry differs")
    b = torch.as_tensor(payload["intercept"], dtype=torch.float64, device=support.flat.device)
    a = torch.as_tensor(payload["slopes"], dtype=torch.float64, device=b.device)
    actual_a = torch.as_tensor(payload.get("saved_endpoint_slopes", payload["slopes"]), dtype=torch.float64, device=b.device)
    if b.shape != a.shape or b.shape != actual_a.shape or b.ndim != 2 or b.shape[1] != support.size:
        raise ValueError("Malformed refinement segment")
    has_saved_endpoints = "saved_endpoint_slopes" in payload
    count = len(b)
    rows = _base_records(base_rows, count)
    actual, affine = stable_gain(b, actual_a, target), stable_gain(b, a, target)
    # Screen the raw fixed-input condition on the current worker device before
    # selective GPU enclosure work. Raw vectors stay on the CUDA device.
    screening = None
    if verified_inputs is not None and payload.get("manuscript_domain", True):
        screening = screen_negative_condition(verified_inputs, device=b.device)
    norm_names = ("delta_norm", "conditional_error", "unconditional_reference_error")
    screen_names = ("certified_negative", "arithmetic_resolved", "delta_squared_upper",
                    "residual_squared_lower", "squared_difference_upper", "squared_difference_lower")
    compact = {**{"actual_" + key: value for key, value in actual.items()},
               **{"affine_" + key: value for key, value in affine.items()},
               **{"norm_" + key: torch.as_tensor(payload[key], dtype=torch.float64, device=b.device)
                  for key in norm_names}}
    if screening is not None:
        compact.update({"screen_" + key: screening[key] for key in screen_names})
    host = _host_columns(compact)
    fast = {key: host["actual_" + key] for key in actual}
    affine_fast = {key: host["affine_" + key] for key in affine}
    scalars = {key: host["norm_" + key] for key in norm_names}
    screen = None if screening is None else {key: host["screen_" + key] for key in screen_names}
    for name in norm_names:
        vector = torch.as_tensor(payload[name], dtype=torch.float64, device=b.device)
        if vector.shape != (count,) or not bool(torch.isfinite(vector).all() & (vector >= 0).all()):
            raise ValueError("Invalid nonnegative norm sufficient input: " + name)
    stored_weights = torch.as_tensor(payload["current_weights"], dtype=torch.float64, device=b.device)
    if stored_weights.shape != b.shape or not bool(torch.isfinite(stored_weights).all() & (stored_weights >= 0).all()) or bool((stored_weights.sum(1) <= 0).any()):
        raise ValueError("Invalid current reference weights")
    atoms = None
    masses = None
    output = []
    for index, old in enumerate(rows):
        structural = bool(payload.get("structural_applicable", old.get("direct_prop5_applicable", False)))
        applicable = bool(payload.get("manuscript_domain", True)) and structural
        G, H, tolerance = fast["G"][index], fast["H"][index], fast["tolerance"][index]
        sign = "zero" if fast["zero"][index] else "positive" if G > tolerance else "negative" if G < -tolerance else "unresolved"
        if support.size > 1 and (not math.isfinite(G) or not math.isfinite(H)):
            sign = "unresolved"
        if abs(H) > fast["H_tolerance"][index] and abs(G) > tolerance and (H > 0) != (G > 0):
            sign = "unresolved"
        if verified_inputs is not None and fast["zero"][index] and support.size > 1:
            sign = "unresolved"  # Rounded construction alone does not prove zero displacement.
        V = _number(old, "direct_prop5_variation_l2")
        M = _number(old, "direct_prop5_margin_l2")
        result = {
            "numerical_policy_version": NUMERICAL_VERSION, "refinement_applicable": applicable,
            "input_contract_status": "original_saved_inputs_available" if verified_inputs is not None else "rounded_sufficient_inputs_only",
            "endpoint_construction_method": payload.get("endpoint_construction_method", "legacy_affine_logit_segment_saved_endpoint_not_available"),
            "fixed_cache_gain_sign": sign if applicable else "not_applicable",
            "source_robust_gain_sign": "unresolved" if applicable else "not_applicable",
            "condition_sign_status": "unresolved" if applicable else "not_applicable",
            "condition_value_status": "estimated_existing_variation" if math.isfinite(V) else "unavailable_variation_not_computed",
            "arithmetic_status": "float64_numerical_assessment", "arithmetic_error_method": "centered_logsumexp_signed_expm1_flagging_not_enclosure",
            "quadrature_status": "existing_embedded_estimate" if math.isfinite(V) else "not_computed",
            "condition_arithmetic_status": "not_enclosed",
            "numerical_gain_endpoint_contract": "saved_endpoint" if "saved_endpoint_slopes" in payload else "affine_only_saved_endpoint_unavailable",
            "numerical_condition_endpoint_contract": "original_affine_path_with_current_reference",
            "quadrature_error_scope": "existing_estimate_not_certified; source_sensitivity_separate",
            "source_sensitivity_status": "heuristic_only_no_justified_endpoint_radii",
            "source_sensitivity_model": "legacy_dtype_multiplier_retained_as_sensitivity_not_fixed_cache_error",
            "reference_law_id": str(payload.get("bank_hash", "unknown")),
            "reference_model_identification": "declared_D_K_not_identified_complete_training_distribution",
            "numerical_log_probability_gain": H if has_saved_endpoints else math.nan, "numerical_log_odds_gain": G if has_saved_endpoints else math.nan,
            "numerical_log_probability_gain_float64_before": H, "numerical_log_odds_gain_float64_before": G,
            "numerical_affine_log_probability_gain": affine_fast["H"][index],
            "numerical_affine_log_odds_gain": affine_fast["G"][index],
            "numerical_matched_log_odds": fast["log_odds0"][index], "numerical_saved_log_odds": fast["log_odds1"][index],
            "numerical_endpoint_difference_crosscheck": fast["old_difference"][index],
            "numerical_endpoint_difference_crosscheck_residual": fast["old_difference"][index] - G,
            "numerical_legacy_saved_log_probability_gain": _number(old, "direct_prop5_log_probability_gain"),
            "numerical_legacy_saved_log_odds_gain": _number(old, "direct_prop5_log_odds_gain"),
            "numerical_original_margin_l2": M, "numerical_original_margin_rmse": M / math.sqrt(support.dimension),
            "numerical_variation_l2": V, "numerical_variation_rmse": V / math.sqrt(support.dimension),
            "numerical_margin_lower_l2": math.nan, "numerical_margin_upper_l2": math.nan,
            "numerical_variation_lower_l2": math.nan, "numerical_variation_upper_l2": math.nan,
            "numerical_gain_before": str(old.get("direct_prop5_gain_status", "unavailable")),
            "numerical_condition_before": str(old.get("direct_prop5_condition_status", "unavailable")),
            "numerical_decimal_precision": 0, "numerical_decimal_products": 0, "numerical_variation_nodes": 0,
            "numerical_gpu_products": 0, "numerical_backend": "cuda_outward_binary64",
            "numerical_effective_mantissa_bits": 53,
            "numerical_gpu_refinement_selected": False, "numerical_gpu_refinement_executed": False,
            "numerical_gpu_refinement_reason": "none",
            "numerical_max_decimal_products": policy.max_decimal_products,
            "numerical_max_variation_nodes": policy.max_variation_nodes,
            "numerical_max_decimal_precision": 0,
            "numerical_max_gpu_products": policy.max_decimal_products,
            "numerical_variation_absolute_width": policy.variation_absolute_width,
            "numerical_certification_scope": "none_float64_assessment",
            "numerical_stopping_reason": "gpu_refinement_disabled_by_zero_budget" if policy.max_decimal_products == 0 else "float64_gain_assessed_condition_not_enclosed",
            "numerical_publication_blocker": False, "implication_eligible": False,
            "implication_audit_status": "unavailable_same_endpoint_and_original_input_enclosure_required",
            "numerical_gain_underflow": H == 0 and sign in {"positive", "negative"},
            "numerical_source_sensitivity_l2": _number(old, "direct_prop5_source_sensitivity_l2"),
            "numerical_integral_audit_status": "unavailable_new_affine_integral_enclosure",
            "numerical_margin_ordering_audit_status": "not_applicable_no_new_refined_margins",
            "numerical_gain_sign_audit_status": "float64_assessment_agreement_or_unresolved",
            "numerical_concavity_audit_status": "not_enclosed",
            "numerical_condition_nonnegative_certified": False,
            "numerical_condition_precheck_upper_l2": math.nan,
            "numerical_condition_screen_status": "not_available_original_inputs_required",
            "numerical_condition_screen_method": None if screening is None else screening["method"],
            "numerical_condition_screen_device": str(b.device),
            "numerical_condition_screen_squared_upper": math.nan,
            "numerical_condition_screen_squared_lower": math.nan,
            "numerical_D_ge_ec_certified": False,
            "numerical_condition_screen_delta_squared_upper": math.nan,
            "numerical_condition_screen_residual_squared_lower": math.nan,
            "numerical_gain_assessment_flagged_before": bool(fast["flagged"][index]),
            "numerical_affine_gain_assessment_flagged_before": bool(affine_fast["flagged"][index]),
            "numerical_gain_saturation_diagnostic": H == 0 and G != 0,
            "numerical_absolute_subtraction_disagreement": abs(fast["old_difference"][index] - G) > tolerance,
            "numerical_cpu_fallback_selected": False,
            "numerical_cpu_fallback_executed": False,
            "numerical_cpu_fallback_reason": "none",

        }
        if support.size == 1:
            result["arithmetic_status"] = "analytic_single_atom_exact_probability_one"
            result["arithmetic_error_method"] = "constant_log_probability_identity; log_odds_undefined"
            result["numerical_certification_scope"] = "single_atom_declared_law_log_probability_identity"
        legacy_integrated = _number(old, "direct_prop5_integrated_gain")
        legacy_integral_error = _number(old, "direct_prop5_integrated_gain_error")
        result["numerical_legacy_integrated_gain"] = legacy_integrated
        result["numerical_legacy_integrated_gain_error"] = legacy_integral_error
        result["numerical_legacy_affine_integral_residual"] = legacy_integrated - affine_fast["H"][index]
        result["numerical_legacy_integral_comparison_scope"] = "old_embedded_integral_vs_new_affine_endpoint; input_construction_rounding_not_certified"
        if math.isfinite(legacy_integrated) and math.isfinite(legacy_integral_error):
            assessed_allowance = abs(legacy_integral_error) + affine_fast["H_tolerance"][index]
            result["numerical_legacy_integral_assessed_allowance"] = assessed_allowance
            result["numerical_integral_audit_status"] = "legacy_assessment_disagreement_requires_refinement" if abs(result["numerical_legacy_affine_integral_residual"]) > assessed_allowance else "consistent_legacy_assessment_not_certificate"
        if not has_saved_endpoints:
            result["fixed_cache_gain_sign"] = "unavailable"
            result["numerical_saved_log_odds"] = math.nan
        if not applicable:
            result["numerical_stopping_reason"] = "outside_manuscript_structural_domain"
            output.append(result)
            continue
        if screen is not None:
            screened_negative = bool(screen["certified_negative"][index])
            result["numerical_condition_screen_status"] = (
                "certified_negative" if screened_negative else "inconclusive"
                if screen["arithmetic_resolved"][index] else "unsupported_arithmetic_range")
            result["numerical_condition_screen_squared_upper"] = screen["squared_difference_upper"][index]
            result["numerical_condition_screen_squared_lower"] = screen["squared_difference_lower"][index]
            result["numerical_D_ge_ec_certified"] = bool(screen["arithmetic_resolved"][index] and screen["squared_difference_lower"][index] >= 0)
            result["numerical_condition_screen_delta_squared_upper"] = screen["delta_squared_upper"][index]
            result["numerical_condition_screen_residual_squared_lower"] = screen["residual_squared_lower"][index]
            if screened_negative:
                result["condition_sign_status"] = "negative"
                result["condition_arithmetic_status"] = "certified_original_saved_inputs"
                result["numerical_certification_scope"] = "condition_only_original_saved_tensors_coefficients_outward_binary64_squared_test_eu_V_nonnegative"
                result["numerical_stopping_reason"] = "negative_condition_batched_screen"
                # The proof is in squared numerator units. Do not invent an
                # L2 margin bound, point M, or V from that sign-only proof.
        original_certificate = verified_inputs is not None
        gain_retry = (_gain_retry_needed(fast, index, original_inputs=original_certificate, singleton=support.size == 1)
                      or _gain_retry_needed(affine_fast, index, original_inputs=original_certificate, singleton=support.size == 1))
        source_retry = original_certificate and verified_inputs.get("source_endpoint_radii") is not None
        condition_retry = result["condition_sign_status"] == "unresolved"
        reasons = [name for name, selected in (("gain_arithmetic", gain_retry),
            ("condition_sign", condition_retry), ("source_perturbation", source_retry)) if selected]
        result["numerical_gpu_refinement_selected"] = bool(reasons)
        result["numerical_gpu_refinement_reason"] = "+".join(reasons) or "none"
        if not reasons:
            output.append(result)
            continue
        if policy.max_decimal_products == 0:
            result["numerical_stopping_reason"] = "gpu_refinement_disabled_by_zero_budget"
            output.append(result)
            continue
        # Avoid moving/converting a raw row when even the unavoidable clean
        # vector operations or a required support projection cannot fit.
        minimum_clean_work = 38 * support.dimension if original_certificate else 2
        minimum_work = minimum_clean_work
        if not condition_retry or result["numerical_D_ge_ec_certified"]:
            # D>=ec says only that the cheap negative test cannot succeed.
            # The original margin remains unknown without eu and V.
            minimum_work += _posterior_work_floor(support.size, support.dimension, original_inputs=original_certificate)
        if minimum_work > policy.max_decimal_products:
            result["numerical_stopping_reason"] = "gpu_operation_budget_exhausted_before_raw_enclosure"
            output.append(result)
            continue
        raw = _raw_row(verified_inputs, index, device=b.device)
        result["numerical_gpu_refinement_executed"] = True
        total_operations = 0
        saved_gain = None
        affine_gain = None
        segment = None
        variation = None
        for _attempt in range(1):  # More work cannot increase binary64 precision.
            remaining = policy.max_decimal_products - total_operations
            if remaining <= 0:
                break
            c = GpuDirected(max_products=remaining, device=b.device)
            previous_nodes = result["numerical_variation_nodes"]
            try:
                if raw is not None:
                    clean = _clean_intervals(c, raw)
                    precheck = negative_condition_precheck(clean[4], clean[5], arithmetic=c)
                else:
                    clean = None
                    precheck = negative_condition_precheck(
                        c.interval(scalars["delta_norm"][index]), c.interval(scalars["conditional_error"][index]),
                        c.interval(scalars["unconditional_reference_error"][index]), arithmetic=c)
                result["numerical_condition_precheck_upper_l2"] = float(precheck["upper"])
                if precheck["condition_sign_status"] == "negative":
                    result["condition_sign_status"] = "negative"
                    result["numerical_margin_upper_l2"] = result["numerical_condition_precheck_upper_l2"]
                    result["numerical_stopping_reason"] = "negative_condition_precheck_V_nonnegative"
                    result["condition_arithmetic_status"] = "certified_original_saved_inputs" if original_certificate else "certified_reduced_scalar_inputs_only"
                    result["numerical_certification_scope"] = "condition_only_original_saved_tensors_and_coefficients_eu_nonnegative" if original_certificate else "condition_only_rounded_cached_norm_scalars"
                # Refine posterior arithmetic only if flagged, condition sign is
                # still critical, or old/new independently computed gains disagree.
                needs_posterior = gain_retry or source_retry or result["condition_sign_status"] != "negative"
                if not needs_posterior:
                    break
                if _posterior_work_floor(support.size, support.dimension, original_inputs=original_certificate) > c.max_products - c.products:
                    raise ArithmeticBudgetExceeded("remaining budget cannot cover a full declared support projection")
                if atoms is None:
                    atoms = c.interval(support.flat)
                    masses = c.interval(support.weights)
                if raw is not None:
                    segment = _raw_segment(c, raw, atoms, masses, target, clean, chunk=support.candidate_chunk)
                else:
                    bi, ai, si = (c.interval(value[index]) for value in (b, a, actual_a))
                    weights = c.interval(stored_weights[index])
                    if "current_logits" in payload:
                        weights = c.softmax(c.interval(torch.as_tensor(payload["current_logits"], device=b.device, dtype=torch.float64)[index]))
                        convex = True
                        defect = None
                    else:
                        total = c.total(weights)
                        residual = c.sub(total, c.interval(1))
                        defect = c.interval(torch.maximum(residual.lower.abs(), residual.upper.abs()))
                        convex = False
                    segment = {"b": bi, "slopes": ai, "saved_slopes": si,
                               "current_mean": posterior_mean(c, atoms, weights),
                               "D": c.interval(scalars["delta_norm"][index]),
                               "ec": c.interval(scalars["conditional_error"][index]),
                               "eu": c.interval(scalars["unconditional_reference_error"][index]),
                               "convex": convex, "mass_defect": defect}
                saved_gain = interval_gain(c, segment["b"], segment["saved_slopes"], target)
                affine_gain = interval_gain(c, segment["b"], segment["slopes"], target)
                result["fixed_cache_gain_sign"] = saved_gain["gain_sign"] if has_saved_endpoints or original_certificate else "unavailable"
                if has_saved_endpoints or original_certificate:
                    for name, interval in (("numerical_log_probability_gain", saved_gain["H"]), ("numerical_log_odds_gain", saved_gain["G"])):
                        if interval is not None:
                            old_value = result[name]
                            if not math.isfinite(old_value) or not bool((interval.lower <= old_value) & (old_value <= interval.upper)):
                                midpoint = interval.lower / 2. + interval.upper / 2.
                                if bool(torch.isfinite(midpoint)):
                                    result[name] = float(midpoint)

                result["arithmetic_status"] = "certified_original_saved_inputs" if original_certificate else "certified_reduced_payload_only"
                result["arithmetic_error_method"] = "outward_cuda_binary64_input_to_gain_enclosure"
                result["numerical_certification_scope"] = "original_saved_tensors_atoms_masses_and_scalar_coefficients_cuda_binary64_enclosure" if original_certificate else "rounded_cached_sufficient_floats_only_not_original_tensor_construction"
                for prefix, gain in (("numerical_gain", saved_gain), ("numerical_affine_gain", affine_gain)):
                    _store_interval(result, prefix + "_H", gain["H"])
                    _store_interval(result, prefix + "_G", gain["G"])
                if original_certificate and saved_gain["G"] is not None and raw.get("source_endpoint_radii") is not None:
                    # Reuse a directly enclosed beta*R, not transfer/residual division.
                    spatial_factor = segment["spatial_factor"]
                    robust = source_robust_gain_interval(saved_gain["G"], beta_radius=spatial_factor,
                        endpoint_radii=raw["source_endpoint_radii"], justification=raw.get("source_sensitivity_justification"), arithmetic=c)
                    result["source_robust_gain_sign"] = robust.sign
                    result["source_sensitivity_status"] = "justified_endpoint_radius_enclosure"
                    result["source_sensitivity_model"] = str(raw["source_sensitivity_justification"])
                    _store_interval(result, "numerical_source_robust_G", robust)
                if saved_gain["G"] is not None:
                    if saved_gain["H"].sign in {"positive", "negative"} and saved_gain["H"].sign != saved_gain["G"].sign and saved_gain["G"].sign in {"positive", "negative"}:
                        result["numerical_publication_blocker"] = True
                        result["numerical_gain_sign_audit_status"] = "failed_certified_H_G_sign_agreement"
                    else:
                        result["numerical_gain_sign_audit_status"] = "consistent_or_H_magnitude_unresolved_sign_follows_G"
                    # Concavity bounds: slope at 1 <= G <= slope at 0.
                    violation = bool((saved_gain["G"].lower > saved_gain["slope0"].upper) | (saved_gain["G"].upper < saved_gain["slope1"].lower))
                    result["numerical_concavity_audit_status"] = "failed_certified_concavity" if violation else "consistent_interval_concavity"
                    result["numerical_publication_blocker"] |= violation
                if result["condition_sign_status"] not in {"negative", "positive", "zero"}:
                    full_pre = negative_condition_precheck(segment["D"], segment["ec"], segment["eu"], arithmetic=c)
                    if full_pre["condition_sign_status"] == "negative":
                        result["condition_sign_status"] = "negative"
                        result["numerical_margin_upper_l2"] = float(full_pre["upper"])
                        result["numerical_stopping_reason"] = "negative_condition_precheck_full_reference"
                        result["condition_arithmetic_status"] = result["arithmetic_status"]
                    else:
                        remaining_nodes = policy.max_variation_nodes - result["numerical_variation_nodes"]
                        if remaining_nodes < 1:
                            result["numerical_stopping_reason"] = "variation_node_budget_exhausted"
                            break
                        variation = enclose_variation(c, atoms, segment["b"], segment["slopes"], segment["current_mean"],
                            segment["D"], segment["ec"], segment["eu"], max_nodes=remaining_nodes,
                            absolute_width=policy.variation_absolute_width,
                            current_in_convex_hull=segment.get("convex", True), current_mass_defect=segment.get("mass_defect"), target_atom=target)
                        result["numerical_variation_lower_l2"], result["numerical_variation_upper_l2"] = variation["V"].floats()
                        result["numerical_margin_lower_l2"], result["numerical_margin_upper_l2"] = variation["M"].floats()
                        result["condition_sign_status"] = variation["M"].sign
                        result["numerical_condition_nonnegative_certified"] = bool(variation["M"].lower >= 0) and original_certificate
                        result["condition_value_status"] = "enclosed_original_margin" if original_certificate else "enclosed_reduced_payload_margin_only"
                        result["quadrature_status"] = "certified_enclosure" if original_certificate else "certified_reduced_payload_only"
                        result["quadrature_error_scope"] = variation["error_method"]
                        result["numerical_variation_nodes"] += variation["nodes"]
                        _store_interval(result, "numerical_integrated_affine_H", variation["integrated_H"])
                        disjoint = bool((variation["integrated_H"].upper < affine_gain["H"].lower) | (variation["integrated_H"].lower > affine_gain["H"].upper))
                        result["numerical_integral_identity_status"] = "failed_certified_affine_integral_identity" if disjoint else "consistent_affine_endpoint_integral_enclosures"
                        result["numerical_integral_audit_status"] = result["numerical_integral_identity_status"]
                        result["numerical_publication_blocker"] |= disjoint
                        result["numerical_stopping_reason"] = variation["stopping_reason"]
                        result["condition_arithmetic_status"] = result["arithmetic_status"]
                if original_certificate and segment is not None and variation is not None:
                    lower_bound = c.mul(c.mul(segment["prefactor"], segment["D"]), variation["M"])
                    transferred = c.sub(lower_bound, segment["transfer"])
                    result["numerical_endpoint_transfer_bound"] = segment["transfer"].floats()[1]
                    result["numerical_saved_gain_theorem_lower_bound"] = transferred.floats()[0]
                    result["implication_eligible"] = bool((variation["M"].lower > 0) & (transferred.lower > 0))
                    contradiction = result["implication_eligible"] and saved_gain["G"] is not None and bool(saved_gain["G"].upper <= 0)
                    result["implication_audit_status"] = "failed_certified_positive_margin_implication" if contradiction else "checked_original_affine_bound_with_proven_saved_endpoint_transfer" if result["implication_eligible"] else "unavailable_positive_margin_or_endpoint_transfer_not_resolved"
                    result["numerical_publication_blocker"] |= contradiction
                if result["fixed_cache_gain_sign"] != "unresolved" and result["condition_sign_status"] != "unresolved":
                    break
            except ArithmeticBudgetExceeded:
                result["numerical_stopping_reason"] = "gpu_operation_budget_exhausted"
                break
            except (ArithmeticError, ValueError, OverflowError) as error:
                result["numerical_stopping_reason"] = "arithmetic_enclosure_unavailable: " + type(error).__name__
                result["numerical_enclosure_failure"] = str(error)
                break
            finally:
                total_operations += c.products
                result["numerical_variation_nodes"] = max(result["numerical_variation_nodes"], previous_nodes + c.evaluated_nodes)
        result["numerical_gpu_products"] = total_operations
        result["numerical_gain_underflow"] = H == 0 and result["fixed_cache_gain_sign"] in {"positive", "negative"}
        if not original_certificate:
            # A certificate of rounded reduced statistics is useful diagnostic
            # information, but must not become an original-tensor certificate.
            result["input_contract_status"] = "rounded_sufficient_inputs_only_original_construction_unenclosed"
        output.append(result)
    return output
