"""Cache-derived same-state/same-innovation DDIM/DDPM matched updates.

Only lightweight schedulers are constructed from saved configuration. There is
no checkpoint lookup, model load, model call, sampling rollout, or new noise draw.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from .metrics import clean_estimates


def displacement_vector_residual(
    z_next, matched, delta, guidance, kappa, *, latent_ndim=3
):
    """Vector QA; equal displacement norms alone do not establish this identity."""
    z_next, matched, delta = [
        torch.as_tensor(x).to(dtype=torch.float64) for x in (z_next, matched, delta)
    ]
    dims = tuple(range(-latent_ndim, 0))
    dimension = math.prod(delta.shape[axis] for axis in dims)
    residual = (z_next - matched) - float(guidance) * float(kappa) * delta
    return (residual.square().sum(dim=dims) / dimension).sqrt()


@dataclass(frozen=True)
class UpdateCoefficients:
    step_index: int
    timestep: int
    destination_timestep: int
    alpha: float
    sigma: float
    snr: float
    destination_alpha: float
    destination_sigma: float
    A: float
    B: float
    noise_std: float
    deterministic: bool
    affine: bool
    status: str

    @property
    def kappa(self):
        return self.B

    def as_dict(self):
        result = asdict(self)
        result["kappa"] = self.B
        return result


class SchedulerAdapter:
    def __init__(
        self, schedule: dict[str, Any], *, delta=0.05, recorded_diffusers_version=None
    ):
        import diffusers
        from diffusers import DDIMScheduler, DDPMScheduler

        self.schedule = schedule
        self.name = str(schedule.get("scheduler_name", "")).lower()
        if self.name not in ("ddim", "ddpm"):
            raise ValueError(
                f"Unavailable scheduler history/unsupported scheduler {self.name!r}"
            )
        if schedule.get("stored_prediction_type", "epsilon") != "epsilon":
            raise ValueError(
                "Theory requires saved canonical epsilon; never convert branches twice"
            )
        self.config = dict(schedule["scheduler_config"])
        self.native_prediction_type = schedule.get(
            "native_prediction_type",
            schedule.get("prediction_type", self.config.get("prediction_type")),
        )
        if self.native_prediction_type not in ("epsilon", "v_prediction", "sample"):
            raise ValueError("Unrecognized checkpoint native prediction contract")
        if self.native_prediction_type != self.config.get("prediction_type", "epsilon"):
            raise ValueError(
                "Saved native prediction type differs from scheduler configuration"
            )
        self.timesteps = torch.as_tensor(schedule["timesteps"], dtype=torch.int64).cpu()
        if (
            self.timesteps.ndim != 1
            or not len(self.timesteps)
            or not bool((self.timesteps[:-1] > self.timesteps[1:]).all())
        ):
            raise ValueError(
                "Saved schedule must have strictly decreasing integer timesteps"
            )
        self.scheduler = (
            DDIMScheduler if self.name == "ddim" else DDPMScheduler
        ).from_config(self.config)
        self.scheduler.set_timesteps(len(self.timesteps))
        if not torch.equal(self.scheduler.timesteps.cpu(), self.timesteps):
            raise ValueError(
                "Saved timesteps disagree with recorded scheduler configuration; unavailable custom history"
            )
        self.installed_version = diffusers.__version__
        self.recorded_version = recorded_diffusers_version or schedule.get(
            "diffusers_version"
        )
        self.version_status = (
            "unrecorded_diffusers_version"
            if self.recorded_version is None
            else (
                "verified_same_diffusers_version"
                if self.recorded_version == self.installed_version
                else "unsupported_diffusers_version_mismatch"
            )
        )
        cumulative = self.scheduler.alphas_cumprod[self.timesteps].double()
        saved = torch.as_tensor(schedule["alphas_cumprod_t"]).double().cpu()
        if saved.shape != cumulative.shape or not torch.allclose(
            saved, cumulative, atol=1e-8, rtol=2e-6
        ):
            raise ValueError(
                "Saved cumulative alphas disagree with reconstructed scheduler configuration"
            )
        self.cumulative = saved
        # Saved alpha/sigma are float32 square roots. Compare, then use the
        # required float64 sqrt(saved cumulative alpha) throughout reductions.
        for key, wanted in (("alpha_t", saved.sqrt()), ("sigma_t", (1 - saved).sqrt())):
            if key in schedule and not torch.allclose(
                torch.as_tensor(schedule[key]).double().cpu(),
                wanted,
                atol=1e-7,
                rtol=2e-6,
            ):
                raise ValueError(f"Saved {key} is inconsistent with cumulative alphas")
        self.delta = float(delta)
        if not 0 < self.delta < 1:
            raise ValueError("Noise bound delta must be predeclared in (0,1)")
        self.nonlinear = bool(
            self.config.get("clip_sample", True)
            or self.config.get("thresholding", False)
        )
        self.variance_type = self.config.get("variance_type", "fixed_small")
        self._coefficients = [
            self._build_coefficients(k) for k in range(len(self.timesteps))
        ]

    def metadata(self):
        return {
            "scheduler_name": self.name,
            "installed_diffusers_version": self.installed_version,
            "recorded_diffusers_version": self.recorded_version,
            "scheduler_version_status": self.version_status,
            "native_prediction_type": self.native_prediction_type,
            "stored_prediction_type": "epsilon",
            "nonlinear_drift": self.nonlinear,
            "variance_type": self.variance_type,
            "ddim_eta": 0.0 if self.name == "ddim" else None,
            "noise_delta": self.delta,
            "noise_coverage_scope": "per_update; no_simultaneous_all_seed_claim",
            "reconstruction": "cache_derived_canonical_epsilon_with_dtype_sensitivity; not_bitwise_replay",
        }

    def _build_coefficients(self, step):
        t = int(self.timesteps[step])
        if self.name == "ddim":
            destination = t - int(self.scheduler.config.num_train_timesteps) // len(
                self.timesteps
            )
            cumulative_s = (
                self.scheduler.alphas_cumprod[destination]
                if destination >= 0
                else self.scheduler.final_alpha_cumprod
            )
        else:
            destination = int(self.scheduler.previous_timestep(t))
            cumulative_s = (
                self.scheduler.alphas_cumprod[destination]
                if destination >= 0
                else self.scheduler.one
            )
        at2, ass2 = float(self.cumulative[step]), float(cumulative_s)
        alpha, sigma = math.sqrt(at2), math.sqrt(1 - at2)
        alpha_s, sigma_s = math.sqrt(ass2), math.sqrt(1 - ass2)
        status = "valid_nonlinear_drift" if self.nonlinear else "valid_affine"
        if self.version_status == "unsupported_diffusers_version_mismatch":
            status = self.version_status
        if sigma == 0 or alpha == 0:
            A = B = float("nan")
            status = "invalid_zero_current_alpha_or_sigma"
        elif self.name == "ddim":
            A = sigma_s / sigma
            B = alpha_s - A * alpha
        else:
            ratio = at2 / ass2
            A = math.sqrt(ratio) * (1 - ass2) / (1 - at2)
            B = alpha_s * (1 - ratio) / (1 - at2)
        rho = 0.0
        if self.name == "ddpm":
            if self.variance_type not in (
                "fixed_small",
                "fixed_small_log",
                "fixed_large",
            ):
                status = "unsupported_branch_dependent_or_invalid_variance:" + str(
                    self.variance_type
                )
                rho = float("nan")
            elif t > 0:
                variance = float(self.scheduler._get_variance(t))
                rho = (
                    variance
                    if self.variance_type == "fixed_small_log"
                    else math.sqrt(variance)
                )
        if status == "valid_affine" and not (math.isfinite(B) and B > 0):
            status = "inapplicable_nonpositive_kappa"
        return UpdateCoefficients(
            step,
            t,
            destination,
            alpha,
            sigma,
            at2 / (1 - at2) if sigma else math.inf,
            alpha_s,
            sigma_s,
            A,
            B,
            rho,
            self.name == "ddim" or rho == 0,
            not self.nonlinear and status == "valid_affine",
            status,
        )

    def coefficients(self, step):
        return self._coefficients[int(step)]

    def _clip(self, clean):
        if self.config.get("thresholding", False):
            return self.scheduler._threshold_sample(clean)
        if self.config.get("clip_sample", True):
            clip = float(self.config.get("clip_sample_range", 1.0))
            return clean.clamp(-clip, clip)
        return clean

    def matched_update(
        self, z, z_next, epsilon_u, epsilon_c, guidance, step, target=None
    ):
        """Return (scalar diagnostics, matched next state, actual drift shift).

        ``target`` here is an optional projection direction (normally x*-center).
        DDPM's reconstructed innovation is shared, never drawn or used as the
        prospective terminal noise-bound input.
        """
        coeff = self.coefficients(step)
        source_eps = max(
            torch.finfo(torch.as_tensor(x).dtype).eps
            for x in (z, z_next, epsilon_u, epsilon_c)
        )
        z, z_next, eu, ec = [
            torch.as_tensor(x).to(dtype=torch.float64)
            for x in (z, z_next, epsilon_u, epsilon_c)
        ]
        if not (z.shape == z_next.shape == eu.shape == ec.shape):
            raise ValueError(
                "Matched update requires corresponding states and two branch predictions"
            )
        dims = tuple(range(-3, 0)) if z.ndim >= 3 else (-1,)
        d = math.prod(z.shape[i] for i in dims)

        def rmse(value):
            return (value.square().sum(dim=dims) / d).sqrt()

        mu, _, D, mg = clean_estimates(
            z, eu, ec, coeff.alpha, coeff.sigma, guidance, latent_ndim=len(dims)
        )
        diagnostics = {
            "A": coeff.A,
            "B": coeff.B,
            "kappa": coeff.B,
            "kappa_positive": math.isfinite(coeff.B) and coeff.B > 0,
            "destination_update_index": int(step) + 1,
            "destination_snr": (
                coeff.destination_alpha**2 / coeff.destination_sigma**2
                if coeff.destination_sigma > 0
                else math.inf
            ),
            "destination_alpha": coeff.destination_alpha,
            "destination_sigma": coeff.destination_sigma,
            "destination_timestep": coeff.destination_timestep,
            "destination_noise_std": coeff.noise_std,
            "scheduler_status": coeff.status,
            "scheduler_version_status": self.version_status,
            "affine_applicable": coeff.affine,
            "deterministic_update": coeff.deterministic,
            "matched_evidence": "algebraic_qa",
            "matched_identity_status": (
                "applicable_affine_positive_kappa"
                if coeff.affine
                else "inapplicable:" + coeff.status
            ),
            "matched_reconstruction_method": (
                "subtract_affine_guidance_shift_from_saved_endpoint"
                if coeff.affine
                else "nonlinear_matched_drift_diagnostic_not_manuscript_identity"
            ),
            "matched_construction_residual_is_independent_verification": False,
            "matched_source_dtype_epsilon": source_eps,
            "reconstruction_status": "cache_derived_matched_reconstruction_not_bitwise_replay",
        }
        if not coeff.status.startswith("valid"):
            invalid = torch.full_like(z, torch.nan)
            diagnostics.update(
                matched_shift_rmse=rmse(invalid),
                predicted_shift_rmse=rmse(invalid),
                innovation_rmse=rmse(invalid),
                deterministic_replay_rmse=rmse(invalid),
                matched_shift_projection=rmse(invalid),
                update_rounding_sensitivity_rmse=rmse(invalid),
                matched_displacement_identity_rmse=rmse(invalid),
                matched_construction_vector_residual_rmse=rmse(invalid),
                independent_replay_rmse=rmse(invalid),
                independent_replay_status="unavailable:" + coeff.status,
            )
            return diagnostics, invalid, invalid
        if self.name == "ddim" and self.nonlinear:
            drift_u = (
                coeff.destination_alpha * self._clip(mu) + coeff.destination_sigma * eu
            )
            drift_g = coeff.destination_alpha * self._clip(
                mg
            ) + coeff.destination_sigma * (eu + float(guidance) * (ec - eu))
        else:
            drift_u = coeff.A * z + coeff.B * self._clip(mu)
            drift_g = coeff.A * z + coeff.B * self._clip(mg)
        shift = drift_g - drift_u if self.nonlinear else coeff.B * float(guidance) * D
        matched = z_next - shift
        innovation = z_next - drift_g
        diagnostics["matched_shift_rmse"] = rmse(shift)
        diagnostics["matched_shift_l2"] = rmse(shift) * math.sqrt(d)
        diagnostics["predicted_shift_rmse"] = (
            abs(coeff.B * float(guidance)) * rmse(D)
            if coeff.affine
            else torch.full_like(rmse(D), torch.nan)
        )
        construction_residual = displacement_vector_residual(
            z_next, matched, D, guidance, coeff.B, latent_ndim=len(dims)
        )
        diagnostics["matched_displacement_identity_rmse"] = (
            construction_residual
            if coeff.affine
            else torch.full_like(construction_residual, torch.nan)
        )
        diagnostics["matched_construction_vector_residual_rmse"] = diagnostics[
            "matched_displacement_identity_rmse"
        ]
        diagnostics["innovation_rmse"] = rmse(innovation)
        diagnostics["deterministic_replay_rmse"] = (
            rmse(innovation)
            if coeff.deterministic
            else torch.full_like(rmse(innovation), torch.nan)
        )
        diagnostics["replay_status"] = (
            "deterministic_numeric_reconstruction_residual"
            if coeff.deterministic
            else "stochastic_shared_innovation_not_replay_error"
        )
        diagnostics["independent_replay_rmse"] = diagnostics[
            "deterministic_replay_rmse"
        ]
        diagnostics["independent_replay_status"] = (
            "saved_endpoint_vs_independent_analytic_drift_no_endpoint_fit"
            if coeff.deterministic
            else "unavailable_saved_noise_realization_recovered_innovation_is_not_independent_replay"
        )
        sensitivity = (
            64
            * source_eps
            * (
                rmse(z_next)
                + abs(coeff.A) * rmse(z)
                + abs(coeff.B) * (rmse(mu) + abs(float(guidance)) * rmse(D))
            )
        )
        diagnostics["update_rounding_sensitivity_rmse"] = sensitivity
        diagnostics["deterministic_replay_within_sensitivity"] = (
            (rmse(innovation) <= sensitivity) if coeff.deterministic else None
        )
        if target is not None:
            vector = torch.as_tensor(target, dtype=torch.float64, device=z.device)
            norm_sq = vector.square().sum()
            diagnostics["matched_shift_projection"] = (shift * vector).sum(
                dim=dims
            ) / torch.where(norm_sq > 0, norm_sq, torch.nan)
            diagnostics["matched_projection_status"] = (
                "valid" if bool(norm_sq > 0) else "zero_target_direction"
            )
        else:
            diagnostics["matched_shift_projection"] = torch.full_like(
                rmse(shift), torch.nan
            )
            diagnostics["matched_projection_status"] = "unavailable_target_direction"
        return diagnostics, matched, shift

    def direct_matched_update(
        self, z, z_next, epsilon_u, epsilon_c, guidance, step, *,
        latent_ndim=3, realized_noise=None, noise_provenance=None,
    ):
        """Independent deterministic counterfactual, separate constructed DDPM case.

        An optional realized_noise is the independently saved additive vector,
        already scaled by the scheduler. Endpoint-fitted noise is never accepted
        as independent verification. No new random draw is made here.
        """
        originals = [torch.as_tensor(x) for x in (z, z_next, epsilon_u, epsilon_c)]
        source_eps = max(torch.finfo(x.dtype).eps for x in originals)
        state, observed, eu, ec = [x.to(dtype=torch.float64) for x in originals]
        if not (state.shape == observed.shape == eu.shape == ec.shape):
            raise ValueError("Direct matched-update shapes differ")
        dims = tuple(range(-latent_ndim, 0))
        dimension = math.prod(state.shape[axis] for axis in dims)

        def norm(value):
            scale = value.abs().amax(dim=dims, keepdim=True)
            normalized = value / torch.where(scale > 0, scale, torch.ones_like(scale))
            return normalized.square().sum(dim=dims).sqrt() * scale.reshape(value.shape[:-latent_ndim])

        coeff = self.coefficients(step)
        g = float(guidance)
        mu, mc, _, mg = clean_estimates(
            state, eu, ec, coeff.alpha, coeff.sigma, g, latent_ndim=latent_ndim
        )
        delta = mc - mu
        shift = g * coeff.kappa * delta
        drift_u = coeff.A * state + coeff.kappa * mu
        drift_g = coeff.A * state + coeff.kappa * mg
        structural = coeff.affine and coeff.kappa > 0 and math.isfinite(g) and g >= 0
        independent = False
        if not structural:
            matched = torch.full_like(observed, torch.nan)
            verification = "inapplicable_nonaffine_nonpositive_kappa_or_negative_guidance"
        elif coeff.noise_std == 0:
            matched = drift_u
            independent = True
            verification = "independent_deterministic_affine_counterfactual"
        elif realized_noise is not None:
            if noise_provenance not in {"independently_saved_additive_noise", "independently_saved_rng_replay"}:
                raise ValueError("Independent matched verification requires verified saved/replayed noise")
            noise = torch.as_tensor(realized_noise, dtype=torch.float64, device=state.device)
            if noise.shape != state.shape or not bool(torch.isfinite(noise).all()):
                raise ValueError("Independent matched noise shape/values differ")
            matched = drift_u + noise
            independent = True
            verification = noise_provenance
        else:
            matched = observed - shift
            verification = "constructed_shared_innovation_from_saved_endpoint_not_independent"
        displacement = observed - matched
        residual = displacement - shift
        tolerance = 64 * source_eps * (
            norm(observed) + abs(coeff.A) * norm(state)
            + abs(coeff.kappa) * (norm(mu) + abs(g) * norm(delta))
        )
        denominator = norm(displacement) + norm(shift)
        diagnostics = {
            "direct_lemma4_applicable": structural,
            "direct_lemma4_independent": independent,
            "direct_lemma4_verification_source": verification,
            "direct_lemma4_status": (
                "independent_source_precision_check" if independent else verification
            ),
            "direct_lemma4_predicted_l2": g * coeff.kappa * norm(delta),
            "direct_lemma4_displacement_l2": norm(displacement),
            "direct_lemma4_vector_residual_l2": norm(residual),
            "direct_lemma4_relative_residual": norm(residual) / torch.where(denominator > 0, denominator, torch.nan),
            "direct_lemma4_source_tolerance_l2": tolerance,
            "direct_lemma4_numeric_within_sensitivity": torch.isfinite(norm(residual)) & (norm(residual) <= tolerance),
            "direct_lemma4_drift_difference_l2": norm(drift_g - drift_u),
            "direct_lemma4_drift_vector_residual_l2": norm((drift_g - drift_u) - shift),
            "direct_lemma4_source_precision_scope": "dtype_sensitivity_estimate_not_certified_inference_error",
        }
        for name, value in list(diagnostics.items()):
            if name.endswith("_l2"):
                diagnostics[name[:-3] + "_rmse"] = value / math.sqrt(dimension)
        return diagnostics, matched, shift

    def terminal_diagnostics(
        self, z, z_next, epsilon_u, epsilon_c, guidance, step, target
    ):
        """Eq.5 structural gate followed by numerical QA, and separate Eq.16 terms.

        Tiny stochastic variance still violates the exact clean-update contract.
        An accidental sample-level cancellation cannot override this gate.
        """
        coeff = self.coefficients(step)
        original = tuple(
            torch.as_tensor(x) for x in (z, z_next, epsilon_u, epsilon_c, target)
        )
        source_eps = max(torch.finfo(x.dtype).eps for x in original)
        z, endpoint, eu, ec, target = [x.to(dtype=torch.float64) for x in original]
        dims = tuple(range(-target.ndim, 0))
        d = target.numel()

        def norm(value):
            return value.square().sum(dim=dims).sqrt()

        mu, mc, delta, mg = clean_estimates(
            z, eu, ec, coeff.alpha, coeff.sigma, guidance, latent_ndim=target.ndim
        )
        g = float(guidance)
        first, second = mc - target, (g - 1) * delta
        clean_error, endpoint_error, rho = mg - target, endpoint - target, endpoint - mg
        first_norm, second_norm = norm(first), norm(second)
        cross = (first * second).sum(dim=dims)
        denominator = first_norm * second_norm
        rho_l2 = norm(rho)
        # Source precision screening is reported separately from the structural
        # exact conditions and never used to set a nonzero noise_std to zero.
        tolerance = (
            64
            * source_eps
            * (norm(endpoint) + norm(z) + norm(mu) + abs(g) * norm(delta))
            / math.sqrt(d)
        )
        structural_checks = {
            "final_update": int(step) == len(self.timesteps) - 1,
            "affine_raw_clean": coeff.affine,
            "zero_current_state_coefficient": coeff.A == 0,
            "unit_clean_coefficient": coeff.B == 1,
            "clean_destination_alpha": coeff.destination_alpha == 1,
            "zero_destination_sigma": coeff.destination_sigma == 0,
            "zero_additive_noise": coeff.noise_std == 0,
        }
        structural = all(structural_checks.values())
        residual_valid = torch.isfinite(rho_l2) & (rho_l2 / math.sqrt(d) <= tolerance)
        applicable = residual_valid & structural
        reason = ";".join(key for key, value in structural_checks.items() if not value)
        result = {
            "terminal_clean_structural": structural,
            "terminal_clean_applicable": applicable,
            "terminal_clean_status": (
                "structurally_clean_numerical_check_separate"
                if structural
                else "inapplicable:" + reason
            ),
            "terminal_clean_structural_reason": reason or None,
            "terminal_clean_numeric_within_sensitivity": residual_valid,
            "terminal_clean_numeric_tolerance_rmse": tolerance,
            "terminal_clean_evidence": "algebraic_qa",
            "terminal_terms_evidence": "finite_noise_observation",
            "terminal_terms_scope": "last_stored_guided_clean_estimate; actual_endpoint_only_if_terminal_clean_applicable",
            "terminal_guidance_domain_satisfied": math.isfinite(g) and g > 1,
            "terminal_A_l2": first_norm,
            "terminal_A_rmse": first_norm / math.sqrt(d),
            "terminal_B_l2": (g - 1) * norm(delta),
            "terminal_B_rmse": (g - 1) * norm(delta) / math.sqrt(d),
            "terminal_cross_inner_product": cross,
            "terminal_cross_term_squared_l2": 2 * cross,
            "terminal_cross_term_per_dimension": 2 * cross / d,
            "terminal_terms_cosine": cross
            / torch.where(denominator > 0, denominator, torch.nan),
            "terminal_terms_cosine_defined": denominator > 0,
            "terminal_eq16_vector_residual_rmse": norm(clean_error - first - second)
            / math.sqrt(d),
            "terminal_clean_error_l2": norm(clean_error),
            "terminal_clean_error_rmse": norm(clean_error) / math.sqrt(d),
            "terminal_endpoint_error_l2": norm(endpoint_error),
            "terminal_endpoint_error_rmse": norm(endpoint_error) / math.sqrt(d),
            "scheduler_rho_l2": rho_l2,
            "scheduler_rho_rmse": rho_l2 / math.sqrt(d),
            "scheduler_rho_evidence": "finite_noise_observation",
            "scheduler_rho_scope": "observed_endpoint_minus_last_guided_clean_estimate_not_an_ex_ante_bound",
            "terminal_clean_error_squared_l2": norm(clean_error).square(),
            "terminal_eq16_squared_l2_from_terms": first_norm.square()
            + second_norm.square()
            + 2 * cross,
        }
        if math.isfinite(g) and g > 1:
            two_term_bound = first_norm + second_norm
            error = norm(clean_error)
            result.update(
                terminal_two_term_bound_l2=two_term_bound,
                terminal_two_term_bound_rmse=two_term_bound / math.sqrt(d),
                terminal_two_term_clean_slack_rmse=(two_term_bound - error)
                / math.sqrt(d),
                terminal_two_term_bound_error_ratio=two_term_bound
                / torch.where(error > 0, error, torch.nan),
                terminal_two_term_ratio_undefined_zero_error=error == 0,
            )
        else:
            for key in (
                "terminal_two_term_bound_l2",
                "terminal_two_term_bound_rmse",
                "terminal_two_term_clean_slack_rmse",
                "terminal_two_term_bound_error_ratio",
            ):
                result[key] = torch.full_like(first_norm, torch.nan)
            result["terminal_two_term_ratio_undefined_zero_error"] = (
                norm(clean_error) == 0
            )
        return result

    def endpoint_bound(self, metrics, step, guidance, dimension):
        """Prospective affine bound from PRE-update quantities only.

        No z_next or observed innovation is accepted by this API. DDPM noise is
        bounded by the known variance and predeclared chi-square probability.
        """
        coeff = self.coefficients(step)
        result = {
            "operational_bound_evidence": "algebraic_qa",
            "paper_terminal_bound_rmse": None,
            "paper_terminal_bound_status": "unavailable_true_posterior_and_reference_approximation_errors",
            "noise_bound_delta": self.delta if not coeff.deterministic else None,
            "noise_bound_coverage_scope": (
                "per_update"
                if not coeff.deterministic
                else "deterministic_exact_arithmetic"
            ),
            "operational_bound_status": (
                "valid" if coeff.affine else "inapplicable:" + coeff.status
            ),
        }
        if not coeff.affine:
            for key in (
                "operational_bound_rmse",
                "bound_state_component",
                "bound_unconditional_component",
                "bound_conditional_component",
                "bound_target_offset_component",
                "bound_noise_component",
            ):
                result[key] = None
            return result
        rho_bound = 0.0
        if coeff.noise_std > 0:
            from scipy.stats import chi2

            rho_bound = coeff.noise_std * math.sqrt(
                float(chi2.ppf(1 - self.delta, dimension)) / dimension
            )
        result["bound_state_component"] = abs(coeff.A) * metrics["state_error_rmse"]
        result["bound_unconditional_component"] = (
            abs(coeff.B) * abs(1 - float(guidance)) * metrics["e_u_rmse"]
        )
        result["bound_conditional_component"] = (
            abs(coeff.B) * abs(float(guidance)) * metrics["e_c_rmse"]
        )
        result["bound_target_offset_component"] = (
            abs(coeff.A + coeff.B - 1) * metrics["target_rmse"]
        )
        result["bound_noise_component"] = rho_bound
        result["operational_bound_rmse"] = sum(
            result[k]
            for k in (
                "bound_state_component",
                "bound_unconditional_component",
                "bound_conditional_component",
                "bound_target_offset_component",
                "bound_noise_component",
            )
        )
        result["operational_bound_status"] = (
            "valid_exact_arithmetic_numeric_deviation_separate"
            if coeff.deterministic
            else "valid_high_probability_known_variance"
        )
        return result
