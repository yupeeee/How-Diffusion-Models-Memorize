"""Supplemental geometry and finite-terminal-update evidence measurements.

The terminal predictor has no observed-endpoint argument. Observed endpoints
enter only the separately named QA helper. Existing direct scalars and the
original clean-update applicability gate are never overwritten here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import pandas as pd
import torch

from .metrics import clean_estimates

EVIDENCE_MEASUREMENT_VERSION = "fixed-seven-evidence-vectors-1"
DIRECTION_ROUNDOFF_MULTIPLIER = 64
TERMINAL_QA_MULTIPLIER = 64
DEFAULT_TERMINAL_NOISE_RUN_ALPHA = 0.05
IDENTITY_KEYS = ("run_id", "original_index", "record_id", "target_id", "seed", "step_index")
INDEPENDENT_INNOVATION_SOURCES = {
    "independently_saved_additive_noise",
    "independently_saved_rng_replay",
}


def _norm(value, latent_ndim):
    dims = tuple(range(-latent_ndim, 0))
    scale = value.abs().amax(dim=dims, keepdim=True)
    normalized = value / torch.where(scale > 0, scale, torch.ones_like(scale))
    return normalized.square().sum(dim=dims).sqrt() * scale.reshape(value.shape[:-latent_ndim])


def _double_inputs(*values):
    device = torch.as_tensor(values[0]).device
    return [torch.as_tensor(value, dtype=torch.float64, device=device) for value in values]


def _units(result, stem, value, dimension):
    result[stem + "_l2"] = value
    result[stem + "_rmse"] = value / math.sqrt(dimension)


def injection_geometry(mu, mc, target, mean, guidance):
    """Complete injection discrepancy, with explicit arithmetic degeneracy.

    Direction normalization uses only a fixed float64 subtraction envelope,
    never a source-dtype or outcome-selected norm cutoff. Absolute errors are
    retained for zero and unresolved directions. The factor g cancels from
    q/r_perp/E_rel by algebra; this function performs no guidance sweep.
    """
    mu, mc, target, mean = _double_inputs(mu, mc, target, mean)
    if mu.shape != mc.shape or tuple(mu.shape[1:]) != tuple(target.shape) or target.shape != mean.shape:
        raise ValueError("Injection geometry requires matching batched branches and fixed target/mean")
    ndim, dimension, count = target.ndim, target.numel(), len(mu)
    g = float(guidance)
    v, delta = target - mean, mc - mu
    vnorm = _norm(v, ndim)
    direction_tolerance = DIRECTION_ROUNDOFF_MULTIPLIER * torch.finfo(torch.float64).eps * (
        _norm(target, ndim) + _norm(mean, ndim)
    )
    finite_rows = torch.isfinite(mu.reshape(count, -1)).all(dim=1) & torch.isfinite(mc.reshape(count, -1)).all(dim=1)
    finite_reference = bool(torch.isfinite(target).all() & torch.isfinite(mean).all())
    direction_valid = finite_reference and bool(vnorm > direction_tolerance) and bool(vnorm > 0)
    guidance_valid = math.isfinite(g) and g > 1
    absolute = abs(g) * _norm(delta - v, ndim)
    result = {
        "evidence_injection_direction_l2": float(vnorm),
        "evidence_injection_direction_tolerance_l2": float(direction_tolerance),
        "evidence_injection_degeneracy_rule": "||v||==0 or ||v||<=64*eps_float64*(||target||+||mean||)",
        "evidence_injection_guidance_cancellation": "exact_algebra_at_fixed_guidance_not_a_sweep",
        "evidence_injection_reference": "mu_K_exact_declared_atom_mean",
    }
    _units(result, "evidence_injection_absolute_error", absolute, dimension)
    _units(result, "evidence_injection_conditional_target_error", _norm(mc - target, ndim), dimension)
    _units(result, "evidence_injection_unconditional_target_error", _norm(mu - target, ndim), dimension)
    _units(result, "evidence_injection_unconditional_mean_error", _norm(mu - mean, ndim), dimension)
    if direction_valid and guidance_valid:
        unit = v / vnorm
        dims = tuple(range(-ndim, 0))
        q = ((delta / vnorm) * unit).sum(dim=dims)
        aligned = q.reshape(count, *([1] * ndim)) * v
        perpendicular = delta - aligned
        r_perp = _norm(perpendicular, ndim) / vnorm
        relative = _norm(delta - v, ndim) / vnorm
        orthogonality = ((perpendicular / vnorm) * unit).sum(dim=dims)
        # The perpendicular vector is computed directly above. Squared norms
        # are used only for the audit of the already measured decomposition.
        identity_residual = relative.square() - ((q - 1).square() + r_perp.square())
        identity_scale = relative.square() + (q - 1).square() + r_perp.square()
        identity_tolerance = 256 * torch.finfo(torch.float64).eps * torch.maximum(torch.ones_like(relative), identity_scale)
        valid = finite_rows & torch.isfinite(q) & torch.isfinite(r_perp) & torch.isfinite(relative)
        audit_finite = torch.isfinite(identity_residual) & torch.isfinite(identity_tolerance)
        qa_pass = audit_finite & (identity_residual.abs() <= identity_tolerance)
        status = ["valid" if ok else "nonfinite_normalized_geometry" for ok in valid.tolist()]
        result.update(
            evidence_injection_q=q,
            evidence_injection_r_perp=r_perp,
            evidence_injection_relative_error=relative,
            evidence_injection_perpendicular_orthogonality=orthogonality,
            evidence_injection_decomposition_squared_residual=identity_residual,
            evidence_injection_decomposition_tolerance=identity_tolerance,
            evidence_injection_decomposition_qa=["consistent_with_float64_estimate" if ok else "numerically_unresolved" for ok in qa_pass.tolist()],
        )
    else:
        valid = torch.zeros(count, dtype=torch.bool, device=mu.device)
        reason = (
            "nonfinite_target_or_mean" if not finite_reference else
            "exact_zero_target_direction" if bool(vnorm == 0) else
            "arithmetic_unresolved_target_direction" if not direction_valid else
            "outside_manuscript_guidance_domain_g_gt_1"
        )
        status = [reason] * count
        for name in ("q", "r_perp", "relative_error", "perpendicular_orthogonality",
                     "decomposition_squared_residual", "decomposition_tolerance"):
            result["evidence_injection_" + name] = torch.full((count,), torch.nan, dtype=torch.float64, device=mu.device)
        result["evidence_injection_decomposition_qa"] = ["not_applicable"] * count
    result["evidence_injection_applicable"] = valid
    result["evidence_injection_status"] = status
    return result


def paired_target_metrics(mu, mc, target, reference_rhs_l2=None):
    """Paired Q is a maximum of the two same-observation target errors."""
    mu, mc, target = _double_inputs(mu, mc, target)
    ndim, dimension = target.ndim, target.numel()
    conditional, unconditional = _norm(mc - target, ndim), _norm(mu - target, ndim)
    joint = torch.maximum(conditional, unconditional)
    result = {"evidence_joint_target_error_definition": "same_seed_maximum_before_any_aggregation",
              "evidence_joint_bound_scope": "direct_triangle_consequence_not_verbatim_Lemma6"}
    _units(result, "evidence_joint_target_error", joint, dimension)
    if reference_rhs_l2 is not None:
        rhs = torch.as_tensor(reference_rhs_l2, dtype=torch.float64, device=mu.device)
        _units(result, "evidence_joint_bound_slack", rhs - joint, dimension)
    return result


def derive_gaussian_noise_contract(adapter, record_metadata):
    """Declare only a saved compatible native DDPM Gaussian variance law.

    This identifies the distribution, not a realized innovation or RNG replay.
    The union bound does not require independence between reused seed paths.
    """
    reasons = []
    if getattr(adapter, "name", None) != "ddpm":
        reasons.append("not_native_ddpm")
    if record_metadata.get("sampler_contract_version") != 2:
        reasons.append("unverified_sampler_contract_v2")
    if getattr(adapter, "version_status", None) != "verified_same_diffusers_version":
        reasons.append("unverified_saved_diffusers_version")
    variance = getattr(adapter, "variance_type", None)
    if variance not in {"fixed_small", "fixed_small_log", "fixed_large"}:
        reasons.append("unsupported_or_branch_dependent_variance")
    if getattr(adapter, "nonlinear", True):
        reasons.append("nonlinear_clean_estimate_operation")
    return {
        "supported": not reasons,
        "reason": ";".join(reasons) or "saved_native_fixed_variance_gaussian_sampler",
        "family": "conditional_isotropic_gaussian" if not reasons else None,
        "conditional_independence_from_preupdate_history": not reasons,
        "scalar_standard_deviation": True if not reasons else False,
        "sampler_contract_version": record_metadata.get("sampler_contract_version"),
        "variance_type": variance,
        "scheduler_name": getattr(adapter, "name", None),
        "recorded_diffusers_version": getattr(adapter, "recorded_version", None),
        "installed_diffusers_version": getattr(adapter, "installed_version", None),
        "source": "sampling._noise_and_generators; sampling._sample_microbatch; native_DDPMScheduler.step",
        "law_scope": "declared_native_Gaussian_random_draw_model; source_precision_QA_separate",
        "realized_innovation_available": False,
    }


@dataclass
class TerminalPrediction:
    """Transient pre-update vectors, never persisted in a scalar table."""
    scalars: dict
    guided_clean: torch.Tensor
    deterministic_drift: torch.Tensor
    predicted_endpoint: torch.Tensor | None
    predicted_defect: torch.Tensor | None
    source_scale_l2: torch.Tensor
    latent_ndim: int


def terminal_predictor_bounds(state, mu, mc, target, coeff, reference, *, final_update,
                              run_scope, terminal_update_count,
                              noise_run_alpha=DEFAULT_TERMINAL_NOISE_RUN_ALPHA,
                              noise_contract=None, independent_innovation=None,
                              innovation_provenance=None):
    """Predictor-side defect bound: deliberately NO observed endpoint input.

    ``independent_innovation`` is an independently saved/replayed ADDITIVE
    vector, already multiplied by the scheduler's scalar noise standard
    deviation. An endpoint-recovered innovation is never accepted.
    """
    state, mu, mc, target = _double_inputs(state, mu, mc, target)
    if state.shape != mu.shape or mu.shape != mc.shape or tuple(state.shape[1:]) != tuple(target.shape):
        raise ValueError("Terminal predictor shapes differ")
    if isinstance(terminal_update_count, bool) or not isinstance(terminal_update_count, Integral) or terminal_update_count < 1:
        raise ValueError("Terminal update count m must be a positive predeclared integer")
    if not isinstance(run_scope, str) or not run_scope:
        raise ValueError("A scientific run scope is required for the terminal noise budget")
    eta = float(noise_run_alpha)
    if not math.isfinite(eta) or not 0 < eta < 1:
        raise ValueError("Terminal run-level failure budget eta must be in (0,1)")
    g = float(reference["guidance_scale"])
    ndim, dimension, count = target.ndim, target.numel(), len(state)
    delta = mc - mu
    mg = mc + (g - 1) * delta
    drift = coeff.A * state + coeff.kappa * mg
    defect_drift = coeff.A * state + (coeff.kappa - 1) * mg
    ec_vector, gap = _norm(mc - target, ndim), _norm(delta, ndim)
    ec = torch.as_tensor(reference.get("direct_conditional_error_l2", ec_vector), dtype=torch.float64, device=state.device).expand(count)
    eu = torch.as_tensor(reference.get("direct_unconditional_reference_error_l2", torch.nan), dtype=torch.float64, device=state.device).expand(count)
    tail = torch.as_tensor(reference.get("direct_radius_tail_l2", torch.nan), dtype=torch.float64, device=state.device).expand(count)
    B_obs, B_ref = ec + (g - 1) * gap, g * ec + (g - 1) * (eu + tail)
    structural_checks = {
        "final_update": bool(final_update), "affine_raw_clean": bool(coeff.affine),
        "zero_current_state_coefficient": coeff.A == 0,
        "unit_clean_coefficient": coeff.kappa == 1,
        "clean_destination_alpha": coeff.destination_alpha == 1,
        "zero_destination_sigma": coeff.destination_sigma == 0,
        "zero_additive_noise": coeff.noise_std == 0,
    }
    original_clean = all(structural_checks.values())
    contract_supported = (bool(final_update) and bool(coeff.affine) and math.isfinite(g) and g > 1
                          and math.isfinite(coeff.A) and math.isfinite(coeff.kappa) and coeff.kappa > 0
                          and math.isfinite(coeff.noise_std) and coeff.noise_std >= 0)
    nan = torch.full((count,), torch.nan, dtype=torch.float64, device=state.device)
    delta_bound = nan.clone()
    predicted_endpoint = predicted_defect = None
    correction_source = "unavailable"
    scope = "unavailable_unsupported_terminal_contract"
    reason = "unsupported_nonfinal_nonaffine_nonpositive_kappa_guidance_or_noise"
    certainty = "unavailable"
    noise_norm_bound, quantile, noise_failure = 0.0, None, None
    innovation_norm = torch.zeros_like(nan)
    if independent_innovation is not None:
        if innovation_provenance not in INDEPENDENT_INNOVATION_SOURCES:
            raise ValueError("Terminal bound forbids endpoint-recovered or unverified innovations")
        innovation = torch.as_tensor(independent_innovation, dtype=torch.float64, device=state.device)
        if innovation.shape != state.shape or not bool(torch.isfinite(innovation).all()):
            raise ValueError("Independent additive innovation has incompatible shape or values")
        if coeff.noise_std == 0 and bool((innovation != 0).any()):
            raise ValueError("Nonzero supplied innovation contradicts the deterministic scheduler contract")
    if contract_supported and coeff.noise_std == 0:
        predicted_endpoint, predicted_defect = drift, defect_drift
        delta_bound = _norm(defect_drift, ndim)
        correction_source = "preupdate_deterministic_affine_defect"
        scope = "original_clean_terminal_theorem" if original_clean else "finite_terminal_update_extension_deterministic"
        reason, certainty = "available", "deterministic_exact_arithmetic; numerical_source_QA_is_not_a_certificate"
    elif contract_supported and independent_innovation is not None:
        predicted_endpoint, predicted_defect = drift + innovation, defect_drift + innovation
        delta_bound = _norm(predicted_defect, ndim)
        innovation_norm = _norm(innovation, ndim)
        correction_source = str(innovation_provenance)
        scope = "finite_terminal_update_extension_deterministic"
        reason = "available_conditional_on_independently_supplied_additive_innovation"
        certainty = "pathwise_exact_arithmetic_given_independently_supplied_innovation"
    elif contract_supported:
        noise_contract = {} if noise_contract is None else noise_contract
        noise_valid = (noise_contract.get("supported") is True
                       and noise_contract.get("family") == "conditional_isotropic_gaussian"
                       and noise_contract.get("conditional_independence_from_preupdate_history") is True
                       and noise_contract.get("scalar_standard_deviation") is True)
        if noise_valid:
            from scipy.stats import chi2
            noise_failure = eta / int(terminal_update_count)
            quantile = math.sqrt(float(chi2.isf(noise_failure, dimension)))
            if math.isfinite(quantile):
                noise_norm_bound = coeff.noise_std * quantile
                delta_bound = _norm(defect_drift, ndim) + noise_norm_bound
                correction_source = "preupdate_affine_defect_plus_declared_Gaussian_noise_norm_bound"
                scope = "finite_terminal_update_extension_gaussian_noise_bound"
                reason = "available_with_declared_simultaneous_run_probability"
                certainty = "simultaneous_probability_at_least_1_minus_eta_under_declared_Gaussian_law"
            else:
                reason = "unrepresentable_Gaussian_noise_quantile"
        else:
            reason = "unsupported_or_unverified_terminal_Gaussian_variance_contract:" + str(noise_contract.get("reason", "missing"))
    clean_error = _norm(mg - target, ndim)
    bound_order_tolerance = 256 * torch.finfo(torch.float64).eps * (
        ec.abs() + eu.abs() + tail.abs() + B_obs.abs() + B_ref.abs() + clean_error)
    bound_order_finite = torch.isfinite(bound_order_tolerance)
    bound_order_consistent = (B_ref - B_obs >= -bound_order_tolerance) & (B_obs - clean_error >= -bound_order_tolerance)
    finite = torch.isfinite(B_obs) & torch.isfinite(B_ref) & torch.isfinite(delta_bound)
    available = finite & (scope != "unavailable_unsupported_terminal_contract")
    scalars = {
        "evidence_terminal_scope": scope,
        "evidence_terminal_status": [reason if valid else "unavailable_nonfinite_preupdate_quantity" if scope != "unavailable_unsupported_terminal_contract" else reason for valid in finite.tolist()],
        "evidence_terminal_applicable": available,
        "evidence_terminal_correction_source": correction_source,
        "evidence_terminal_certainty": certainty,
        "evidence_terminal_original_clean_structural": original_clean,
        "evidence_terminal_original_clean_structural_reason": ";".join(name for name, valid in structural_checks.items() if not valid) or None,
        "evidence_terminal_manuscript_extension_required": scope.startswith("finite_terminal_update_extension"),
        "evidence_terminal_noise_run_alpha": eta,
        "evidence_terminal_A": coeff.A, "evidence_terminal_kappa": coeff.kappa,
        "evidence_terminal_guidance_scale": g, "evidence_terminal_noise_std": coeff.noise_std,
        "evidence_terminal_latent_dimension": dimension,
        "evidence_terminal_variance_type": None if noise_contract is None else noise_contract.get("variance_type"),
        "evidence_terminal_noise_update_count": int(terminal_update_count),
        "evidence_terminal_noise_run_scope": run_scope,
        "evidence_terminal_noise_per_update_alpha": noise_failure,
        "evidence_terminal_noise_chi2_norm_quantile": quantile,
        "evidence_terminal_noise_quantile_method": "sqrt(scipy.stats.chi2.isf(eta/m,d))" if quantile is not None else None,
        "evidence_terminal_noise_probability_scope": "simultaneous_across_m_terminal_updates_in_this_scientific_run; trajectory_independence_not_required" if noise_failure is not None else None,
        "evidence_terminal_variance_contract": None if noise_contract is None else noise_contract.get("reason"),
        "evidence_terminal_input_scope": "preupdate_state_and_clean_estimates_only",
        "evidence_terminal_conditional_scalar_vector_residual_l2": ec - ec_vector,
        "evidence_terminal_clean_bound_order_slack_l2": B_ref - B_obs,
        "evidence_terminal_clean_error_l2": clean_error,
        "evidence_terminal_clean_bound_order_tolerance_l2": bound_order_tolerance,
        "evidence_terminal_clean_bound_order_status": [
            "not_applicable_guidance_domain" if not (math.isfinite(g) and g > 1) else
            "consistent_with_float64_estimate" if ok and finite_value else
            "failed_clean_bound_order" if finite_value else "unavailable_nonfinite_reference"
            for ok, finite_value in zip(bound_order_consistent.tolist(), bound_order_finite.tolist())
        ],
    }
    for name, value in (("B_obs", B_obs), ("B_ref", B_ref), ("defect_bound", delta_bound),
                        ("corrected_obs_bound", B_obs + delta_bound), ("corrected_ref_bound", B_ref + delta_bound),
                        ("conditional_error", ec), ("residual_guidance", (g - 1) * gap),
                        ("deterministic_defect", _norm(defect_drift, ndim)),
                        ("noise_norm_bound", torch.full_like(ec, noise_norm_bound))):
        _units(scalars, "evidence_terminal_" + name, value, dimension)
    source_scale = abs(coeff.A) * _norm(state, ndim) + abs(coeff.kappa) * (
        _norm(mc, ndim) + abs(g - 1) * (_norm(mc, ndim) + _norm(mu, ndim))) + innovation_norm
    return TerminalPrediction(scalars, mg, drift, predicted_endpoint, predicted_defect, source_scale, ndim)


def terminal_update_qa(observed, prediction, *, source_epsilon):
    """Observed-output QA cannot change any predictor-side correction or gate."""
    observed = torch.as_tensor(observed, dtype=torch.float64, device=prediction.guided_clean.device)
    if observed.shape != prediction.guided_clean.shape:
        raise ValueError("Terminal QA endpoint shape differs")
    eps = float(source_epsilon)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Source epsilon must describe the saved floating precision")
    ndim = prediction.latent_ndim
    dimension = math.prod(observed.shape[1:])
    tolerance = TERMINAL_QA_MULTIPLIER * eps * (_norm(observed, ndim) + prediction.source_scale_l2)
    observed_defect = _norm(observed - prediction.guided_clean, ndim)
    result = {"evidence_terminal_qa_scope": "source_dtype_sensitivity_screen_not_machine_arithmetic_certificate"}
    _units(result, "evidence_terminal_observed_defect_qa_only", observed_defect, dimension)
    _units(result, "evidence_terminal_reconstruction_tolerance", tolerance, dimension)
    if prediction.predicted_endpoint is None:
        residual = torch.full_like(tolerance, torch.nan)
        unavailable = ("unavailable_unsupported_terminal_contract"
                       if prediction.scalars["evidence_terminal_scope"] == "unavailable_unsupported_terminal_contract"
                       else "unavailable_independent_innovation_for_update_reconstruction")
        statuses = [unavailable] * len(tolerance)
    else:
        residual = _norm(observed - prediction.predicted_endpoint, ndim)
        valid = torch.isfinite(residual) & (residual <= tolerance)
        statuses = ["consistent_with_source_precision" if ok else "failed_reconstruction" for ok in valid.tolist()]
    _units(result, "evidence_terminal_reconstruction_residual", residual, dimension)
    result["evidence_terminal_reconstruction_status"] = statuses
    if prediction.scalars["evidence_terminal_scope"] == "finite_terminal_update_extension_gaussian_noise_bound":
        # This residual is explicitly observed-output QA, not the correction.
        implied_noise = _norm(observed - prediction.deterministic_drift, ndim)
        bound = prediction.scalars["evidence_terminal_noise_norm_bound_l2"]
        _units(result, "evidence_terminal_implied_noise_qa_only", implied_noise, dimension)
        result["evidence_terminal_noise_bound_exceeded"] = implied_noise > bound
        result["evidence_terminal_noise_bound_exceeded_beyond_source_sensitivity"] = implied_noise > bound + tolerance
        result["evidence_terminal_noise_bound_qa_scope"] = "a_probability_event_may_fail; not_an_integration_error_or_predictor_input"
    else:
        result["evidence_terminal_noise_bound_exceeded"] = [None] * len(tolerance)
        result["evidence_terminal_noise_bound_exceeded_beyond_source_sensitivity"] = [None] * len(tolerance)
    return result


def _scalar_frame(values, identities):
    count = len(identities)
    result = identities.loc[:, list(IDENTITY_KEYS)].copy().reset_index(drop=True)
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                value = value.item()
            elif value.shape == (count,):
                value = value.tolist()
            else:
                raise ValueError(f"Non-scalar evidence value {name}")
        if isinstance(value, list) and len(value) != count:
            raise ValueError(f"Evidence scalar count differs for {name}")
        result[name] = value
    return result


def _ordered_rows(frame, seeds, step):
    selected = frame.loc[frame.step_index.eq(step)].copy()
    if len(selected) != len(seeds) or selected.seed.astype(str).duplicated().any():
        raise ValueError("Supplement requires one saved core scalar row per seed/step")
    lookup = {str(value): index for index, value in enumerate(selected.seed)}
    if set(lookup) != set(map(str, seeds)):
        raise ValueError("Supplement seed population differs from saved core")
    return selected.iloc[[lookup[str(seed)] for seed in seeds]].reset_index(drop=True)


def measure_evidence_record(preloaded, record, law, config, adapter,
                            terminal_update_count, run_scope, core_tables):
    """One missing-vector supplement; no posterior, integrator or learned calls.

    Only the initial and final current vectors are used. Q is derived from
    already saved paired errors at every current state, including the last.
    """
    z, epsilon_u, epsilon_c, target = preloaded
    meta = record.metadata if hasattr(record, "metadata") else record
    seeds = list(meta["seeds"])
    if str(meta["scientific_config_hash"]) != run_scope:
        raise ValueError("Terminal probability scope must match this scientific run")
    steps, g = int(config["num_inference_steps"]), float(config["guidance_scale"])
    if z.shape[:2] != (len(seeds), steps + 1) or epsilon_u.shape[:2] != (len(seeds), steps) or epsilon_u.shape != epsilon_c.shape:
        raise ValueError("Evidence supplement requires T+1 states and T canonical branches")
    if meta.get("stored_prediction_type", "epsilon") != "epsilon":
        raise ValueError("Evidence supplement must not convert canonical epsilon twice")
    device = law.support.flat.device
    target = target.to(device=device, dtype=torch.float64)
    first = _ordered_rows(core_tables["initial"], seeds, 0)
    last = _ordered_rows(core_tables["terminal"], seeds, steps - 1)
    initial_coeff = adapter.coefficients(0)
    initial_mu, initial_mc, _, _ = clean_estimates(
        z[:, 0].to(device), epsilon_u[:, 0].to(device), epsilon_c[:, 0].to(device),
        initial_coeff.alpha, initial_coeff.sigma, g, latent_ndim=target.ndim,
    )
    geometry = injection_geometry(initial_mu, initial_mc, target, law.mean_vector, g)
    initial = _scalar_frame(geometry, first)
    trajectory_source = core_tables["trajectory"]
    trajectory = trajectory_source.loc[:, list(IDENTITY_KEYS)].copy()
    ec = pd.to_numeric(trajectory_source.direct_conditional_error_l2, errors="coerce")
    eu_target = pd.to_numeric(trajectory_source.direct_unconditional_target_error_l2, errors="coerce")
    joint = pd.concat([ec, eu_target], axis=1).max(axis=1, skipna=False)
    dimension = target.numel()
    trajectory["evidence_joint_target_error_l2"] = joint
    trajectory["evidence_joint_target_error_rmse"] = joint / math.sqrt(dimension)
    trajectory["evidence_joint_target_error_definition"] = "same_seed_maximum_before_any_aggregation"
    trajectory["evidence_joint_bound_scope"] = "direct_triangle_consequence_not_verbatim_Lemma6"
    trajectory["evidence_joint_bound_slack_l2"] = trajectory_source.direct_lemma6_rhs_l2 - joint
    trajectory["evidence_joint_bound_slack_rmse"] = trajectory.evidence_joint_bound_slack_l2 / math.sqrt(dimension)
    coeff = adapter.coefficients(steps - 1)
    state, u, c = [value.to(device) for value in (z[:, -2], epsilon_u[:, -1], epsilon_c[:, -1])]
    mu, mc, _, _ = clean_estimates(state, u, c, coeff.alpha, coeff.sigma, g, latent_ndim=target.ndim)
    # Pandas may expose read-only arrays; give torch writable storage to share.
    reference = {
        name: torch.as_tensor(
            pd.to_numeric(last[name], errors="coerce").to_numpy(
                dtype=float, na_value=float("nan"), copy=True,
            ),
            dtype=torch.float64, device=device,
        )
        for name in (
            "direct_conditional_error_l2",
            "direct_unconditional_reference_error_l2",
            "direct_radius_tail_l2",
        )
    }
    reference["guidance_scale"] = g
    noise_contract = derive_gaussian_noise_contract(adapter, meta)
    prediction = terminal_predictor_bounds(
        state, mu, mc, target, coeff, reference, final_update=True,
        run_scope=run_scope, terminal_update_count=terminal_update_count,
        noise_run_alpha=config.get("terminal_noise_run_alpha", DEFAULT_TERMINAL_NOISE_RUN_ALPHA),
        noise_contract=noise_contract,
    )
    source_epsilon = max(torch.finfo(value.dtype).eps for value in (z, epsilon_u, epsilon_c))
    qa = terminal_update_qa(z[:, -1], prediction, source_epsilon=source_epsilon)
    terminal = _scalar_frame({**prediction.scalars, **qa}, last)
    for frame in (initial, trajectory, terminal):
        frame["evidence_measurement_version"] = EVIDENCE_MEASUREMENT_VERSION
    return {
        "initial": initial, "trajectory": trajectory, "terminal": terminal,
        "audit": {"complete": True, "formula_version": EVIDENCE_MEASUREMENT_VERSION,
                  "scope": "additive_scalars_only; original_direct_values_and_clean_gate_unchanged",
                  "gaussian_noise_contract": noise_contract,
                  "terminal_noise_run_scope": run_scope, "terminal_noise_update_count": int(terminal_update_count),
                  "terminal_noise_run_alpha": config.get("terminal_noise_run_alpha", DEFAULT_TERMINAL_NOISE_RUN_ALPHA),
                  "reconstruction_status_counts": terminal.evidence_terminal_reconstruction_status.value_counts(dropna=False).to_dict(),
                  "injection_status_counts": initial.evidence_injection_status.value_counts(dropna=False).to_dict(),
                  "manuscript_extension_required": bool(terminal.evidence_terminal_manuscript_extension_required.any())},
    }
