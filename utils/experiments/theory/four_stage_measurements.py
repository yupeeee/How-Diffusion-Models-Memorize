"""Additive measurements for the fixed four-stage mechanism experiments.

This module is called only by analysis workers. No inference, rollout, image
conversion, or change to the declared reference law occurs here.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch

from .contracts import TheoryError
from .evidence_reduce import KEYS
from .evidence_measurements import _norm, _ordered_rows, _scalar_frame
from .metrics import clean_estimates
from .numerical_refinement import NumericalPolicy, _gain_retry_needed, stable_gain
from .gpu_intervals import ArithmeticBudgetExceeded, GpuDirected
from .gpu_refinement import interval_gain, require_cuda

FOUR_STAGE_VERSION = "four-stage-measurements-cuda-3"
SHAPE_ATOL = 1e-12
SHAPE_RTOL = 1e-10
SHAPE_FIXED_FRACTIONS = (.25, .5, .75)


def dose_grid(guidance):
    guidance = float(guidance)
    if not math.isfinite(guidance):
        raise TheoryError("The matched response requires a finite guidance scale")
    conditional = {1 / guidance} if guidance > 0 and 0 <= 1 / guidance <= 1 else set()
    # Unsupported guidance domains still receive the standard inapplicable
    # curve receipt. Never place an out-of-range conditional marker on [0,1].
    return sorted({j / 40 for j in range(41)} | conditional)


def corollary3_guidance_fit(mu, mc, target, mean, guidance):
    """Signed no-intercept fit of the full guided clean estimate about selected mu.

    Fit y=xhat_g-mu to a*v, where v=x_star-mu and xhat_g=xhat_u+g*(xhat_c-xhat_u).
    The unit-vector form is the one-column least-squares closed form, without
    squaring a tiny direction norm. This is not a fit of the CFG injection alone.
    """
    device = require_cuda(torch.as_tensor(mu).device)
    mu, mc, target, mean = [torch.as_tensor(value, device=device, dtype=torch.float64)
                            for value in (mu, mc, target, mean)]
    if (target.ndim < 1 or mu.ndim != target.ndim + 1 or mu.shape != mc.shape
            or tuple(mu.shape[1:]) != tuple(target.shape) or target.shape != mean.shape or len(mu) == 0):
        raise TheoryError("Corollary 3 guidance-fit latent shapes differ")
    ndim, dimension, count = target.ndim, target.numel(), len(mu)
    g = float(guidance)
    v = target - mean
    vnorm = _norm(v, ndim)
    eps = torch.finfo(torch.float64).eps
    direction_tolerance = 64 * eps * (_norm(target, ndim) + _norm(mean, ndim))
    nan = torch.full((count,), torch.nan, dtype=torch.float64, device=device)
    scalar_fields = ("cor3_guidance_fit", "cor3_guidance_fit_residual_rmse",
                     "cor3_guidance_fit_orthogonality_l2", "cor3_guidance_fit_orthogonality_tolerance_l2",
                     "cor3_guidance_fit_pythagorean_residual_scaled", "cor3_guidance_fit_pythagorean_tolerance_scaled")
    result = {name: nan.clone() for name in scalar_fields}
    result.update(cor3_guidance_fit_direction_l2=vnorm,
                  cor3_guidance_fit_direction_tolerance_l2=direction_tolerance,
                  cor3_guidance_fit_applicable=torch.zeros(count, dtype=torch.bool, device=device),
                  cor3_guidance_fit_definition="argmin_a ||(xhat_u+g*(xhat_c-xhat_u)-selected_mu)-a*(target-selected_mu)||^2; signed_unconstrained_no_intercept",
                  cor3_guidance_fit_qa_scope="dimension_scaled_float64_projection_assessment_not_certified_enclosure",
                  cor3_guidance_fit_degeneracy_rule="||v||==0 or ||v||<=64*eps_float64*(||target||+||selected_mu||)")
    finite_reference = bool(torch.isfinite(target).all() & torch.isfinite(mean).all())
    finite_direction = bool(torch.isfinite(vnorm) & torch.isfinite(direction_tolerance))
    if (not finite_reference or not finite_direction or not math.isfinite(g)
            or bool(vnorm == 0) or bool(vnorm <= direction_tolerance)):
        reason = ("nonfinite_target_or_mean" if not finite_reference else
                  "nonfinite_guidance" if not math.isfinite(g) else
                  "exact_zero_target_direction" if bool(vnorm == 0) else
                  "arithmetic_unresolved_target_direction")
        result["cor3_guidance_fit_status"] = [reason] * count
        result["cor3_guidance_fit_qa"] = ["numerically_unresolved" if reason == "arithmetic_unresolved_target_direction" else "not_applicable"] * count
        return result

    flat_mu, flat_mc = mu.flatten(1), mc.flatten(1)
    finite_inputs = torch.isfinite(flat_mu).all(1) & torch.isfinite(flat_mc).all(1)
    guided = flat_mu + g * (flat_mc - flat_mu)
    response = guided - mean.reshape(1, -1)
    unit = v.flatten() / vnorm
    projection = (response * unit).sum(1)
    fitted = projection / vnorm
    residual = response - fitted[:, None] * v.flatten()
    response_norm, residual_norm = _norm(response, 1), _norm(residual, 1)
    orthogonality = (residual * unit).sum(1)
    # These finite-precision checks assess the measured projection; they do not
    # certify a real-arithmetic solution or introduce an epsilon fit denominator.
    qa_roundoff = 256 * eps * max(1, dimension)
    orthogonal_tolerance = (qa_roundoff * response_norm + qa_roundoff * residual_norm
                            + qa_roundoff * projection.abs())
    qa_scale = torch.maximum(response_norm, torch.maximum(residual_norm, projection.abs()))
    qa_scale = torch.where(qa_scale > 0, qa_scale, torch.ones_like(qa_scale))
    response_squared = (response / qa_scale[:, None]).square().sum(1)
    residual_squared = (residual / qa_scale[:, None]).square().sum(1)
    projected_squared = (projection / qa_scale).square()
    pythagorean_residual = response_squared - residual_squared - projected_squared
    pythagorean_tolerance = qa_roundoff * (1 + response_squared + residual_squared + projected_squared)
    finite_result = (finite_inputs & torch.isfinite(response).all(1) & torch.isfinite(fitted)
                     & torch.isfinite(residual).all(1) & torch.isfinite(residual_norm)
                     & torch.isfinite(orthogonality) & torch.isfinite(orthogonal_tolerance)
                     & torch.isfinite(pythagorean_residual) & torch.isfinite(pythagorean_tolerance))
    qa_pass = (finite_result & (orthogonality.abs() <= orthogonal_tolerance)
               & (pythagorean_residual.abs() <= pythagorean_tolerance))
    values = (fitted, residual_norm / math.sqrt(dimension), orthogonality, orthogonal_tolerance,
              pythagorean_residual, pythagorean_tolerance)
    result.update({name: torch.where(finite_result, value, nan) for name, value in zip(scalar_fields, values, strict=True)})
    result["cor3_guidance_fit_applicable"] = finite_result
    result["cor3_guidance_fit_status"] = [
        "measured" if good else "numerically_unresolved_projection" if finite else
        "nonfinite_projection_arithmetic" if inputs else "nonfinite_clean_branch"
        for inputs, finite, good in zip(finite_inputs.tolist(), finite_result.tolist(), qa_pass.tolist(), strict=True)]
    result["cor3_guidance_fit_qa"] = [
        "consistent_float64_estimate" if good else "numerically_unresolved" if inputs else "not_applicable"
        for inputs, good in zip(finite_inputs.tolist(), qa_pass.tolist(), strict=True)]
    return result


def branch_motion(mu, mc, next_mu, next_mc, target):
    """Exact algebraic accounting in per-coordinate squared latent units."""
    values = [torch.as_tensor(value).double().flatten(1) for value in (mu, mc, next_mu, next_mc)]
    mu, mc, next_mu, next_mc = values
    target = torch.as_tensor(target, device=mu.device, dtype=torch.float64).reshape(1, -1)
    if any(value.shape != mu.shape for value in values) or target.shape[1] != mu.shape[1]:
        raise TheoryError("Branch-motion latent shapes differ")
    d = mu.shape[1]
    gap, vc, vu = mc - mu, next_mc - mc, next_mu - mu
    conditional = 2 * (gap * vc).sum(1) / d
    unconditional = -2 * (gap * vu).sum(1) / d
    quadratic = (vc - vu).square().sum(1) / d
    change = ((next_mc - next_mu).square().sum(1) - gap.square().sum(1)) / d
    residual = change - conditional - unconditional - quadratic
    # A declared float64 QA allowance, not a formal interval certificate.
    tolerance = 256 * torch.finfo(torch.float64).eps * (1 + change.abs() + conditional.abs() + unconditional.abs() + quadratic.abs())
    return {"gap_motion_conditional": conditional, "gap_motion_unconditional": unconditional,
            "gap_motion_quadratic": quadratic, "gap_squared_change": change,
            "identity_residual": residual, "identity_tolerance": tolerance,
            "identity_status": ["consistent_float64_estimate" if ok else "failed_float64_identity_check"
                                for ok in (residual.abs() <= tolerance).tolist()],
            "conditional_target_error_squared_change": ((next_mc - target).square().sum(1) - (mc - target).square().sum(1)) / d,
            "unconditional_target_error_squared_change": ((next_mu - target).square().sum(1) - (mu - target).square().sum(1)) / d,
            "motion_scope": "consecutive_prediction_accounting_includes_noise_label_change_not_causal_attribution"}


def trajectory_shape(values, steps, *, atol=SHAPE_ATOL, rtol=SHAPE_RTOL):
    """Predeclared raw-curve descriptors; neither smoothing nor outcome gates."""
    values = np.asarray(values, dtype=float)
    complete = len(values) == int(steps) and np.isfinite(values).all()
    result = {"shape_complete": bool(complete), "shape_observed_steps": int(np.isfinite(values).sum()),
              "shape_expected_steps": int(steps), "shape_atol_rmse": float(atol), "shape_rtol": float(rtol),
              "shape_tolerance_scope": "fixed_float64_descriptor_resolution_not_source_precision_certificate",
              "first_peak_step": math.nan, "first_peak_progress": math.nan,
              "peak_ties_json": "[]", "near_peak_ties_json": "[]", "shape_flat": False,
              "fixed_snapshot_fractions_json": json.dumps(SHAPE_FIXED_FRACTIONS)}
    scalar_names = ("initial_gap_rmse", "maximum_gap_rmse", "final_gap_rmse", "early_gap_rmse", "middle_gap_rmse", "late_gap_rmse",
                    "peak_rise_rmse", "post_peak_decline_rmse", "total_upward_variation_rmse", "total_downward_variation_rmse")
    result.update({name: math.nan for name in scalar_names})
    if not complete or int(steps) < 1:
        return {**result, "shape_status": "incomplete_raw_curve", "shape_tolerance_rmse": math.nan}
    maximum = float(values.max())
    tolerance = float(atol + rtol * np.abs(values).max())
    delta = np.diff(values)
    peak = int(np.argmax(values))
    exact = np.flatnonzero(values == maximum).tolist()
    near = np.flatnonzero(maximum - values <= tolerance).tolist()
    flat = bool(maximum - values.min() <= tolerance)
    direction = np.sign(delta[np.abs(delta) > tolerance])
    if flat:
        status = "flat_or_numerically_unresolved_plateau"
    elif np.all(direction <= 0):
        status = "nonincreasing_within_declared_resolution"
    elif np.all(direction >= 0):
        status = "nondecreasing_within_declared_resolution"
    elif np.all(np.diff(direction) <= 0):
        status = "rise_then_fall_not_a_strict_unimodality_claim"
    else:
        status = "complex_multi_direction_curve"
    result.update(shape_status=status, shape_flat=flat, shape_tolerance_rmse=tolerance,
                  first_peak_step=math.nan if flat else peak,
                  first_peak_progress=math.nan if flat else peak / (steps - 1) if steps > 1 else 0.,
                  peak_ties_json=json.dumps(exact), near_peak_ties_json=json.dumps(near),
                  initial_gap_rmse=float(values[0]), maximum_gap_rmse=maximum, final_gap_rmse=float(values[-1]),
                  peak_rise_rmse=maximum - float(values[0]), post_peak_decline_rmse=maximum - float(values[-1]),
                  total_upward_variation_rmse=float(np.maximum(delta, 0).sum()),
                  total_downward_variation_rmse=float(np.maximum(-delta, 0).sum()))
    for name, fraction in zip(("early_gap_rmse", "middle_gap_rmse", "late_gap_rmse"), SHAPE_FIXED_FRACTIONS, strict=True):
        result[name] = float(values[int(math.floor((steps - 1) * fraction + .5))])
    return result


def response_rows(payload, identities, guidance, *, device, policy=None):
    """Evaluate the fixed affine segment using compact candidate logits only."""
    device = require_cuda(device)
    b = payload["intercept"].to(device=device, dtype=torch.float64)
    slopes = payload["slopes"].to(device=device, dtype=torch.float64)
    target = int(payload["target_atom"])
    if len(b) != len(identities):
        raise TheoryError("Response payload seed population differs")
    rows = []
    policy = NumericalPolicy() if policy is None else policy
    spent = [0] * len(b)
    discrepancy = torch.as_tensor(payload["endpoint_difference_l2"]).detach().cpu().tolist()
    for dose_index, s in enumerate(dose_grid(guidance)):
        scaled_slopes = s * slopes
        gain = stable_gain(b, scaled_slopes, target)
        values = {name: gain[name].detach().cpu().tolist() for name in ("H", "G", "flagged", "zero", "H_tolerance", "tolerance", "log_odds0", "log_odds1")}
        for index, identity in enumerate(identities.to_dict("records")):
            H, G = values["H"][index], values["G"][index]
            H_before, G_before = H, G
            finite = math.isfinite(H)
            before = spent[index]
            retry = _gain_retry_needed(values, index, original_inputs=False, singleton=b.shape[1] == 1)
            interval_fields = {"response_H_interval_lower": math.nan, "response_H_interval_upper": math.nan,
                               "response_G_interval_lower": math.nan, "response_G_interval_upper": math.nan}
            fallback_status, reduced_sign = "not_selected_stable_assessment", "unresolved"
            if retry:
                fallback_status = "budget_unavailable_keep_finite_assessment"
                # Only unresolved rows enter bounded CUDA interval arithmetic.
                # Candidate logits never move to host for numerical work.
                remaining = policy.max_decimal_products - spent[index]
                if remaining > 0:
                    arithmetic = GpuDirected(max_products=remaining, device=device)
                    try:
                        enclosed = interval_gain(arithmetic, arithmetic.interval(b[index]),
                                                 arithmetic.interval(scaled_slopes[index]), target)
                        reduced_sign = enclosed["gain_sign"]
                        for quantity in ("H", "G"):
                            interval = enclosed[quantity]
                            if interval is None:
                                continue
                            interval_fields["response_" + quantity + "_interval_lower"], interval_fields["response_" + quantity + "_interval_upper"] = interval.floats()
                            estimate = H if quantity == "H" else G
                            if not math.isfinite(estimate) or not bool((interval.lower <= estimate) & (estimate <= interval.upper)):
                                midpoint = interval.lower / 2. + interval.upper / 2.
                                if bool(torch.isfinite(midpoint)):
                                    if quantity == "H":
                                        H = float(midpoint)
                                    else:
                                        G = float(midpoint)
                        fallback_status = "cuda_reduced_logit_interval_resolved" if reduced_sign != "unresolved" else "cuda_reduced_logit_interval_unresolved_fixed_binary64"
                    except (ArithmeticBudgetExceeded, ArithmeticError, OverflowError, ValueError):
                        fallback_status = "cuda_reduced_logit_interval_budget_or_arithmetic_unresolved"
                    finally:
                        spent[index] += arithmetic.products
            finite = math.isfinite(H)
            tolerance = float(identity.get("affine_endpoint_tolerance_l2", math.nan))
            resolved = math.isfinite(tolerance) and math.isfinite(discrepancy[index]) and discrepancy[index] <= tolerance
            method = payload["endpoint_construction_method"]
            independently_constructed = not method.startswith("constructed_shared_innovation")
            endpoint_status = ("affine_and_saved_agree_under_declared_source_precision"
                               if resolved and independently_constructed else "reconstructed_affine_segment_saved_endpoint_discrepancy_recorded")
            rows.append({**identity, "dose_index": dose_index, "s": s,
                "log_probability_gain": H, "log_odds_gain": G,
                "log_probability_gain_float64_before": H_before, "log_odds_gain_float64_before": G_before,
                "log_probability_arithmetic_tolerance": values["H_tolerance"][index],
                "log_odds_arithmetic_tolerance": values["tolerance"][index],
                "matched_log_odds_before": values["log_odds0"][index], "matched_log_odds_after": values["log_odds1"][index],
                "numerical_status": "unavailable_nonfinite" if not finite else "finite_estimate_reduced_logit_sign_resolved" if reduced_sign != "unresolved" else "finite_flagged_arithmetic_assessment" if values["flagged"][index] else "finite_float64_assessment",
                "response_fallback_selected": retry, "response_fallback_status": fallback_status,
                "response_reduced_logit_gain_sign": reduced_sign, **interval_fields,
                "response_arithmetic_scope": "exact_stored_scaled_logit_inputs_only_not_original_vector_construction",
                "response_dose_decimal_products": 0, "response_curve_decimal_products": 0,
                "response_curve_decimal_budget": 0,
                "response_dose_gpu_products": spent[index] - before,
                "response_curve_gpu_products": spent[index],
                "response_curve_gpu_budget": policy.max_decimal_products,
                "response_arithmetic_backend": "cuda_outward_binary64",
                "response_effective_mantissa_bits": 53, "response_cpu_fallback": False,
                "structural_applicable": True, "endpoint_contract": "fixed_reconstructed_affine_segment",
                "endpoint_status": endpoint_status, "endpoint_construction_method": method,
                "saved_endpoint_discrepancy_l2": discrepancy[index],
                "input_source": "matched_affine_segment_no_network_reevaluation",
                "curve_weight_population": "fixed_all_finite_full_curves_independent_of_sign_or_arithmetic_resolution"})
    return pd.DataFrame(rows)


def unavailable_response(identities, guidance, reason):
    return pd.DataFrame([{**identity, "dose_index": i, "s": s, "log_probability_gain": math.nan,
                          "log_odds_gain": math.nan, "structural_applicable": False,
                          "numerical_status": reason, "endpoint_contract": "unavailable", "endpoint_status": reason}
                         for identity in identities.to_dict("records") for i, s in enumerate(dose_grid(guidance))])


def measure_record(log, record, law, adapter, config, core_tables, *, gaussian_bank, gaussian_initialization_compatible=True, payload_loader=None, payload_saver=None, batch_size=16):
    """Read no files; reduce one already loaded record in bounded vector blocks."""
    from .numerical_reduce import _verified_inputs
    from .numerical_refinement import build_refinement_payload
    z, u, c, target = log
    meta = record.metadata
    seeds = list(map(str, meta["seeds"]))
    steps = int(config["num_inference_steps"])
    g = float(config["guidance_scale"])
    if z.shape[:2] != (len(seeds), steps + 1) or u.shape[:2] != (len(seeds), steps) or u.shape != c.shape:
        raise TheoryError("Four-stage measurements require T predictions and T+1 saved states")
    if meta.get("stored_prediction_type", "epsilon") != "epsilon":
        raise TheoryError("Four-stage measurements require already-canonical saved epsilon")
    device = require_cuda(law.support.flat.device)
    target = target.to(device=device, dtype=torch.float64)
    target_id = law.target_id_for(target, required=False)
    target_atom = None if target_id is None else law.support.aliases[target_id]
    dimension, root_d = target.numel(), math.sqrt(target.numel())
    geometry = getattr(law, "_four_stage_geometry", None)
    if geometry is None:
        mean = law.theory_mean_vector.to(device=device, dtype=torch.float64)
        bank_mean = law.mean_vector.to(device=device, dtype=torch.float64)
        distances = (law.support.flat - bank_mean.reshape(1, -1)).norm(dim=1)
        spread = float((law.support.weights * distances.square()).sum().sqrt()) / root_d
        comparison_distances = (law.support.flat - mean.reshape(1, -1)).norm(dim=1)
        comparison_scale = float((law.support.weights * comparison_distances.square()).sum().sqrt()) / root_d
        geometry = mean, bank_mean, spread, comparison_scale
        law._four_stage_geometry = geometry
    mean, bank_mean, spread, comparison_scale = geometry
    mean_receipt = law.theory_mean_metadata
    mean_fields = {
        "mean_norm_l2": float(mean.norm()), "mean_norm_rmse": float(mean.norm()) / root_d,
        "bank_mean_norm_l2": float(bank_mean.norm()), "bank_mean_norm_rmse": float(bank_mean.norm()) / root_d,
        "mean_offset_l2": float((bank_mean - mean).norm()), "mean_offset_rmse": float((bank_mean - mean).norm()) / root_d,
        "theory_mean_source": str(mean_receipt.get("source", "declared_finite_bank_mean")),
        "theory_mean_sha256": str(mean_receipt.get("vector_sha256", law.metadata.get("mean_sha256", ""))),
    }
    trajectory, motions, initial, responses = [], [], [], []
    initial_base = _ordered_rows(core_tables["initial"], seeds, 0)
    matched_initial = _ordered_rows(core_tables["matched_updates"], seeds, 0)
    all_keys = [*KEYS, "terminal_sscd", "prompt_utf8_sha256", "candidate_target_atom_id", "timestep", "destination_timestep", "snr"]
    for start in range(0, len(seeds), batch_size):
        stop = min(start + batch_size, len(seeds))
        previous = None
        for step in range(steps):
            coeff = adapter.coefficients(step)
            state, eu, ec = [value[start:stop, step].to(device=device) for value in (z, u, c)]
            mu, mc, _, _ = clean_estimates(state, eu, ec, coeff.alpha, coeff.sigma, g, latent_ndim=target.ndim)
            mu, mc = mu.double(), mc.double()
            base = _ordered_rows(core_tables["trajectory"], seeds, step).iloc[start:stop].reset_index(drop=True)
            identity = base[[name for name in all_keys if name in base]].copy()
            identity["sscd"] = identity.terminal_sscd
            identity["native_timestep"] = coeff.timestep
            gap = (mc - mu).flatten(1).norm(dim=1)
            conditional = (mc - target).flatten(1).norm(dim=1)
            unconditional = (mu - target).flatten(1).norm(dim=1)
            noise_gap = (ec.double() - eu.double()).flatten(1).norm(dim=1) / root_d
            values = {"k": step, "total_steps": steps, "normalized_progress": step / (steps - 1) if steps > 1 else 0.,
                "gap_l2": gap, "gap_rmse": gap / root_d,
                "conditional_target_error_l2": conditional, "conditional_target_error_rmse": conditional / root_d,
                "unconditional_target_error_l2": unconditional, "unconditional_target_error_rmse": unconditional / root_d,
                "joint_target_error_l2": torch.maximum(conditional, unconditional),
                "joint_target_error_rmse": torch.maximum(conditional, unconditional) / root_d,
                "noise_gap_rmse": noise_gap, "schedule_scale": coeff.sigma / coeff.alpha if coeff.alpha > 0 else math.nan,
                "scaling_identity_residual_rmse": gap / root_d - (coeff.sigma / coeff.alpha if coeff.alpha > 0 else math.nan) * noise_gap,
                "four_stage_numeric_scope": "saved_precision_vectors_promoted_before_arithmetic_float64_estimates"}
            trajectory.append(_scalar_frame(values, identity))
            if previous is not None:
                last_mu, last_mc, last_identity = previous
                frame = _scalar_frame(branch_motion(last_mu, last_mc, mu, mc, target), last_identity)
                for name in ("terminal_sscd", "sscd", "prompt_utf8_sha256", "native_timestep"):
                    if name in last_identity:
                        frame[name] = last_identity[name].to_numpy()
                frame["destination_prediction_index"] = step
                frame["destination_native_timestep"] = coeff.timestep
                motions.append(frame)
            previous = mu, mc, identity
            if step == 0:
                mean_error = (mu - mean).flatten(1).norm(dim=1)
                hashes = [hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for value in mu]
                matched_bank = (z[start:stop, 0].to(device=device, dtype=torch.float64) == gaussian_bank[start:stop].to(device=device, dtype=torch.float64)).flatten(1).all(1)
                if not gaussian_initialization_compatible:
                    matched_bank.fill_(False)
                first = _scalar_frame({"initial_unconditional_mean_error_l2": mean_error,
                    "initial_unconditional_mean_error_rmse": mean_error / root_d,
                    "initial_unconditional_vector_sha256": hashes,
                    "initial_gaussian_bank_match": matched_bank,
                    **mean_fields, **corollary3_guidance_fit(mu, mc, target, mean, g),
                    "reference_spread_rmse": spread,
                    "comparison_scale_rmse": comparison_scale}, initial_base.iloc[start:stop])
                initial.append(first)
                response_id = matched_initial.iloc[start:stop][[name for name in all_keys if name in matched_initial]].reset_index(drop=True)
                response_id["sscd"] = response_id.terminal_sscd
                response_id["native_timestep"] = coeff.timestep
                source_epsilon = max(torch.finfo(value.dtype).eps for value in (z, u, c))
                observed_norm = z[start:stop, 1].to(device=device, dtype=torch.float64).flatten(1).norm(dim=1)
                source_scale = (abs(coeff.A) * state.double().flatten(1).norm(dim=1)
                                + abs(coeff.kappa) * (mu.flatten(1).norm(dim=1)
                                    + abs(g) * (mc.flatten(1).norm(dim=1) + mu.flatten(1).norm(dim=1))))
                response_id["affine_endpoint_tolerance_l2"] = (128 * source_epsilon * (observed_norm + source_scale)).detach().cpu().tolist()
                response_id["endpoint_tolerance_scope"] = "declared_source_dtype_sensitivity_screen_not_certified_arithmetic_enclosure"
                response_id["direct_prop5_matched_reconstruction_valid"] = matched_initial.iloc[start:stop].get("direct_prop5_matched_reconstruction_valid", pd.Series(False, index=range(stop-start))).to_numpy()
                applicable = (target_atom is not None and coeff.affine and coeff.kappa > 0 and coeff.alpha > 0
                              and coeff.sigma > 0 and coeff.destination_alpha > 0 and coeff.destination_sigma > 0 and steps > 1 and g > 1)
                if applicable:
                    payload = payload_loader(start, stop) if payload_loader else None
                    if payload is None:
                        verified = _verified_inputs(log, coeff, indices=list(range(start, stop)), step=0, steps=steps, guidance=g)
                        payload = build_refinement_payload(law.support, verified, bank_hash=law.law_hash, target_atom=target_atom)
                        if payload_saver is not None:
                            payload_saver(start, stop, payload)
                    if payload.get("bank_hash") != law.law_hash or int(payload["target_atom"]) != target_atom:
                        raise TheoryError("First-response sufficient payload differs from fixed law/target")
                    responses.append(response_rows(payload, response_id, g, device=device,
                        policy=NumericalPolicy(decimal_precision=int(config.get("numerical_decimal_precision", 64)),
                                               max_decimal_products=int(config.get("numerical_max_decimal_products", 2_000_000)))))
                else:
                    responses.append(unavailable_response(response_id, g, "first_update_outside_nonterminal_affine_positive_noise_domain"))
    trajectory_frame = pd.concat(trajectory, ignore_index=True)
    shape_rows = []
    for seed in seeds:
        curve = trajectory_frame.loc[trajectory_frame.seed.astype(str).eq(seed)].sort_values("step_index")
        key = initial_base.loc[initial_base.seed.astype(str).eq(seed)].iloc[0]
        shape_rows.append({**{name: key[name] for name in all_keys if name in key}, "step_index": -1,
                           "sscd": key.terminal_sscd, **trajectory_shape(curve.gap_rmse.to_numpy(), steps)})
    return {"trajectory_four_stage": trajectory_frame, "branch_motion": pd.concat(motions, ignore_index=True) if motions else pd.DataFrame(columns=KEYS + ["gap_motion_conditional", "gap_motion_unconditional", "gap_motion_quadratic", "gap_squared_change", "identity_residual", "identity_tolerance", "identity_status", "sscd"]),
            "initial_four_stage": pd.concat(initial, ignore_index=True), "feedback_response": pd.concat(responses, ignore_index=True),
            "trajectory_shapes": pd.DataFrame(shape_rows)}


def baseline_summary(gaussian_reference, initial_samples, reference_atoms):
    """Use genuine unique-seed probes; repeated prompt vectors only audit parity."""
    source = gaussian_reference.loc[gaussian_reference.step_index.eq(0)].copy()
    if source.empty or source.duplicated(["run_id", "seed"]).any():
        raise TheoryError("The initial unconditional baseline requires one genuine Gaussian probe per unique seed")
    values = pd.to_numeric(source.learned_mean_error_rmse, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    complete = bool(finite.all())
    spread = float(reference_atoms.reference_scale_rmse.iloc[0])
    estimate = float(np.sqrt(np.mean(values**2))) if complete else math.nan
    low = high = math.nan
    if complete:
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(values), size=(1000, len(values)))
        low, high = np.quantile(np.sqrt(np.square(values[draws]).mean(1)), [.025, .975])
    audits = []
    for (run_id, seed), group in initial_samples.groupby(["run_id", "seed"], sort=True):
        eligible = group.loc[group.initial_gaussian_bank_match.fillna(False).astype(bool)]
        canonical = eligible.sort_values(["original_index", "record_id"]).iloc[0] if len(eligible) else None
        distinct = int(eligible.initial_unconditional_vector_sha256.nunique())
        audits.append({"run_id": run_id, "seed": seed, "record_count": len(group), "compatible_gaussian_record_count": len(eligible),
                       "distinct_unconditional_vectors": distinct, "unconditional_disagreement": distinct > 1,
                       "canonical_record_id": None if canonical is None else str(canonical.record_id),
                       "canonical_rule": "lexicographic_original_index_record_id_without_outcome_filtering",
                       "mean_distance_range_rmse": float(eligible.initial_unconditional_mean_error_rmse.max() - eligible.initial_unconditional_mean_error_rmse.min()) if len(eligible) else math.nan})
    audit = pd.DataFrame(audits)
    row = {"run_id": str(source.run_id.iloc[0]), "reference_law_hash": str(reference_atoms.reference_law_hash.iloc[0]),
           "mean_norm_l2": float(initial_samples.mean_norm_l2.iloc[0]), "mean_norm_rmse": float(initial_samples.mean_norm_rmse.iloc[0]),
           "spread_rmse": spread, "baseline_error_rmse": estimate, "baseline_error_interval_low_rmse": float(low),
           "baseline_error_interval_high_rmse": float(high), "relative_baseline_error": estimate / spread if spread > 0 else math.nan,
           "relative_status": "defined" if spread > 0 else "undefined_zero_reference_spread", "unique_seed_count": len(source),
           "finite_seed_count": int(finite.sum()), "complete": complete,
           "disagreement_count": int(audit.unconditional_disagreement.sum()) if len(audit) else 0,
           "baseline_input_source": "unique_genuine_gaussian_reference_probes_not_prompt_average",
           "bootstrap_replicates": 1000, "bootstrap_seed": 0, "bootstrap_scope": "Monte_Carlo_unique_Gaussian_seed_estimator_not_independent_prompt_inference"}
    for field in ("bank_mean_norm_l2", "bank_mean_norm_rmse", "mean_offset_l2", "mean_offset_rmse",
                  "theory_mean_source", "theory_mean_sha256", "comparison_scale_rmse"):
        if field in initial_samples:
            row[field] = initial_samples[field].iloc[0]
    row["spread_definition"] = "weighted_atom_RMS_about_exact_finite_bank_mean"
    row["comparison_definition"] = "held_out_Gaussian_estimates_about_selected_theory_mean"
    for name, field in (("initial_unconditional_reference_error", "unconditional_reference_error_rmse"),
                        ("initial_reference_to_mean_error", "reference_mean_error_rmse")):
        samples = pd.to_numeric(source[field], errors="coerce").to_numpy(dtype=float) if field in source else np.full(len(source), np.nan)
        finite_samples = np.isfinite(samples)
        row[name + "_rms_rmse"] = float(np.sqrt(np.mean(samples**2))) if finite_samples.all() else math.nan
        row[name + "_finite_seed_count"] = int(finite_samples.sum())
        row[name + "_status"] = "complete_common_unique_seed_population" if finite_samples.all() else "unavailable_incomplete_common_seed_population"
    row["reference_summary_scope"] = "root_mean_squared_per_seed_RMS_distances_on_identical_unique_Gaussian_seed_population"
    return pd.DataFrame([row]), audit


def prompt_shape_summaries(shapes):
    """Preserve overlapping prompt populations and explicitly paired groups."""
    rows = []
    grouped = shapes.assign(outcome_group=np.where(shapes.sscd > .75, "high_sscd", "lower_sscd"))
    columns = ["initial_gap_rmse", "maximum_gap_rmse", "final_gap_rmse", "peak_rise_rmse", "post_peak_decline_rmse", "total_upward_variation_rmse", "total_downward_variation_rmse"]
    for (run, prompt, group), values in grouped.groupby(["run_id", "prompt_utf8_sha256", "outcome_group"], sort=True):
        complete = values.loc[values.shape_complete]
        rows.append({"run_id": run, "prompt_utf8_sha256": prompt, "outcome_group": group, "seed_count": len(values),
                     "complete_seed_count": len(complete), **{name: float(complete[name].mean()) for name in columns},
                     "summary_scope": "within_prompt_outcome_group_mean_of_per_seed_shape_descriptors"})
    summary = pd.DataFrame(rows)
    mixed = []
    for (run, prompt), values in summary.groupby(["run_id", "prompt_utf8_sha256"], sort=True):
        if set(values.outcome_group) != {"high_sscd", "lower_sscd"}:
            continue
        lookup = values.set_index("outcome_group")
        mixed.append({"run_id": run, "prompt_utf8_sha256": prompt,
                      **{name + "_high_minus_lower": float(lookup.loc["high_sscd", name] - lookup.loc["lower_sscd", name]) for name in columns},
                      "summary_scope": "paired_within_same_mixed_outcome_prompt_descriptive_difference"})
    return summary, pd.DataFrame(mixed, columns=["run_id", "prompt_utf8_sha256", *[name + "_high_minus_lower" for name in columns], "summary_scope"])
