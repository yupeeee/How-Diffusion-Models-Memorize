"""Float64 scalar reductions for genuine forward and Gaussian observations.

These functions do not run a network. Forward-input identities and Gaussian
transfer quantities deliberately have separate interfaces and averaging units.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

FORMULA_VERSION = "direct-probe-math-1"
PROMPT_KEYS = ["run_id", "original_index", "record_id", "target_id"]


def _flat(value):
    tensor = torch.as_tensor(value).detach().to(dtype=torch.float64)
    if tensor.ndim < 2 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Probe batches must have finite values and a batch axis")
    return tensor.reshape(len(tensor), -1)


def _scales(alpha, sigma):
    a, s = float(alpha), float(sigma)
    if not (math.isfinite(a) and math.isfinite(s) and a > 0 and s > 0):
        raise ValueError("Learned probes require positive finite native noise scales")
    return a, s, (a / s) ** 2


def conditional_forward_metrics(target, noise, samples, prediction, alpha, sigma):
    """Measure raw coordinate-summed loss and clean error independently per draw."""
    a, s, snr = _scales(alpha, sigma)
    eps, z, predicted = map(_flat, (noise, samples, prediction))
    target = torch.as_tensor(target, dtype=torch.float64, device=z.device).reshape(1, -1)
    if eps.shape != z.shape or predicted.shape != z.shape or target.shape[1] != z.shape[1]:
        raise ValueError("Forward probe shapes do not agree")
    dimension = z.shape[1]
    noise_residual = eps - predicted
    loss = noise_residual.square().sum(1)
    recovered = (z - s * predicted) / a
    error = (recovered - target).square().sum(1)
    corruption_residual = (z - (a * target + s * eps)).norm(dim=1)
    corruption_error = corruption_residual / a
    roundoff = 128 * torch.finfo(torch.float64).eps * (1 + error + loss / snr)
    allowance = roundoff + corruption_error * (
        2 * noise_residual.norm(dim=1) / math.sqrt(snr) + corruption_error
    )
    return {
        "loss_squared_l2": loss,
        "loss_mse": loss / dimension,
        "clean_error_squared_l2": error,
        "clean_error_l2": error.sqrt(),
        "clean_error_rmse": (error / dimension).sqrt(),
        "loss_over_snr_squared_l2": loss / snr,
        "identity_residual_squared_l2": error - loss / snr,
        "identity_arithmetic_allowance_squared_l2": allowance,
        "identity_numeric_pass": (error - loss / snr).abs() <= allowance,
        "forward_corruption_residual_l2": corruption_residual,
        "latent_dimension": dimension,
    }


def conditional_gaussian_metrics(target, samples, prediction, alpha, sigma):
    """Observe Gaussian-input error, without equating it to forward expected loss."""
    a, s, _ = _scales(alpha, sigma)
    z, predicted = map(_flat, (samples, prediction))
    target = torch.as_tensor(target, dtype=torch.float64, device=z.device).reshape(1, -1)
    if z.shape != predicted.shape or target.shape[1] != z.shape[1]:
        raise ValueError("Gaussian conditional probe shapes do not agree")
    error = ((z - s * predicted) / a - target).square().sum(1)
    return {
        "conditional_error_squared_l2": error,
        "conditional_error_l2": error.sqrt(),
        "conditional_error_rmse": (error / z.shape[1]).sqrt(),
        "latent_dimension": z.shape[1],
    }


def gaussian_reference_metrics(samples, prediction, reference_mean, mean, alpha, sigma, max_atom_norm, *, bank_mean=None):
    """Comparison-centred decomposition with the finite-law offset kept explicit.

    ``mean`` is the independently estimated comparison centre when available;
    ``bank_mean`` remains the exact atom mean used by the posterior's bound.
    The Eq. 45 bank-centred bound needs their distance added when the centres
    differ. All distances reuse the same learned and posterior evaluations.
    """
    a, s, _ = _scales(alpha, sigma)
    z, predicted, posterior = map(_flat, (samples, prediction, reference_mean))
    mean = torch.as_tensor(mean, dtype=torch.float64, device=z.device).reshape(1, -1)
    bank_mean = mean if bank_mean is None else torch.as_tensor(bank_mean, dtype=torch.float64, device=z.device).reshape(1, -1)
    if (z.shape != predicted.shape or z.shape != posterior.shape
            or mean.shape[1] != z.shape[1] or bank_mean.shape != mean.shape):
        raise ValueError("Gaussian reference probe shapes do not agree")
    learned = (z - s * predicted) / a
    displacement, reference, residual = learned - mean, posterior - mean, learned - posterior
    lhs, movement, error = (v.norm(dim=1) for v in (displacement, reference, residual))
    learned_zero, reference_zero = (v.norm(dim=1) for v in (learned, posterior))
    mean_norm = mean.norm(dim=1).expand(len(z))
    bank_mean_norm = bank_mean.norm(dim=1).expand(len(z))
    mean_offset = (bank_mean - mean).norm(dim=1).expand(len(z))
    bank_movement = (posterior - bank_mean).norm(dim=1)
    cross = 2 * (reference * residual).sum(1)
    M = float(max_atom_norm)
    if not math.isfinite(M) or M < 0:
        raise ValueError("Reference maximum atom norm must be finite and nonnegative")
    B = (a * M * z.norm(dim=1) + a * a * M * M / 2) / (s * s)
    twice = 2 * B
    # log(expm1(x)) is stable both near zero and beyond exp's representable range.
    log_bound = torch.full_like(B, -math.inf)
    if M > 0:
        small = (twice > 0) & (twice <= 50)
        large = twice > 50
        log_bound[small] = math.log(M) + torch.expm1(twice[small]).log()
        log_bound[large] = math.log(M) + twice[large] + torch.log1p(-torch.exp(-twice[large]))
    bank_log_bound = log_bound
    bank_bound = bank_log_bound.exp()
    log_bound = torch.logaddexp(bank_log_bound, mean_offset.log())
    bound = bank_bound + mean_offset
    root_d = math.sqrt(z.shape[1])
    return {
        "learned_mean_error_l2": lhs,
        "learned_mean_error_squared_l2": lhs.square(),
        "learned_mean_error_rmse": lhs / root_d,
        "reference_mean_error_l2": movement,
        "reference_mean_error_rmse": movement / root_d,
        "learned_zero_error_l2": learned_zero,
        "learned_zero_error_rmse": learned_zero / root_d,
        "reference_zero_error_l2": reference_zero,
        "reference_zero_error_rmse": reference_zero / root_d,
        "mean_norm_l2": mean_norm,
        "mean_norm_rmse": mean_norm / root_d,
        "bank_mean_norm_l2": bank_mean_norm,
        "bank_mean_norm_rmse": bank_mean_norm / root_d,
        "mean_offset_l2": mean_offset,
        "mean_offset_rmse": mean_offset / root_d,
        "reference_to_bank_mean_error_l2": bank_movement,
        "reference_to_bank_mean_error_rmse": bank_movement / root_d,
        "unconditional_reference_error_l2": error,
        "unconditional_reference_error_rmse": error / root_d,
        "baseline_bound_l2": error + movement,
        "baseline_bound_rmse": (error + movement) / root_d,
        "baseline_slack_l2": error + movement - lhs,
        "baseline_cross_term_squared_l2": cross,
        "baseline_squared_identity_residual": lhs.square() - error.square() - movement.square() - cross,
        "baseline_vector_residual_l2": (displacement - residual - reference).norm(dim=1),
        "reference_log_weight_bound": B,
        "reference_bound_log_l2": log_bound,
        "reference_bound_l2": bound,
        "reference_bound_overflow": ~torch.isfinite(bound),
        "bank_reference_bound_log_l2": bank_log_bound,
        "bank_reference_bound_l2": bank_bound,
        "bank_reference_bound_overflow": ~torch.isfinite(bank_bound),
        "reference_bound_mean_offset_included": True,
        "latent_dimension": z.shape[1],
    }


def marginal_forward_metrics(noise, samples, prediction, reference_mean, alpha, sigma):
    """Direct excess loss and finite-sample cross term, never mean subtraction."""
    a, s, snr = _scales(alpha, sigma)
    eps, z, predicted, posterior = map(_flat, (noise, samples, prediction, reference_mean))
    if not (eps.shape == z.shape == predicted.shape == posterior.shape):
        raise ValueError("Marginal forward probe shapes do not agree")
    optimal_prediction = (z - a * posterior) / s
    optimal_residual = eps - optimal_prediction
    excess_residual = predicted - optimal_prediction
    total = (eps - predicted).square().sum(1)
    optimal = optimal_residual.square().sum(1)
    excess = excess_residual.square().sum(1)
    cross = -2 * (optimal_residual * excess_residual).sum(1)
    reference_error = ((z - s * predicted) / a - posterior).square().sum(1)
    allowance = 128 * torch.finfo(torch.float64).eps * (1 + reference_error + excess / snr)
    return {
        "total_loss_squared_l2": total,
        "optimal_loss_squared_l2": optimal,
        "excess_loss_squared_l2": excess,
        "cross_term_squared_l2": cross,
        "loss_decomposition_residual_squared_l2": total - optimal - excess - cross,
        "reference_error_squared_l2": reference_error,
        "excess_over_snr_squared_l2": excess / snr,
        "identity_residual_squared_l2": reference_error - excess / snr,
        "identity_arithmetic_allowance_squared_l2": allowance,
        "identity_numeric_pass": (reference_error - excess / snr).abs() <= allowance,
        "latent_dimension": z.shape[1],
    }


def pinsker_quantities(target, alpha, sigma):
    """KL(N(alpha*x,sigma² I)||N(0,I)) and Pinsker's upper bound on TV."""
    a, s, _ = _scales(alpha, sigma)
    target = torch.as_tensor(target, dtype=torch.float64)
    variance = s * s
    delta = variance - 1
    if abs(delta) < 1e-4:
        # Stable x-log(1+x), including the exactly equal-variance limit.
        variance_term = sum(((-1) ** n) * delta ** n / n for n in range(2, 10))
    else:
        variance_term = delta - math.log(variance)
    kl = 0.5 * (a * a * float(target.square().sum()) + target.numel() * variance_term)
    return {
        "KL_target_to_gaussian": kl,
        "pinsker_tv_bound": min(1.0, math.sqrt(kl / 2)),
        "pinsker_interpretation": "analytic_TV_upper_bound; empirical_loss_transfer_is_plugin_not_certificate",
    }


def summarize_forward_losses(draws):
    """One independent Monte Carlo loss estimate per pair/native timestep."""
    if draws.empty:
        return pd.DataFrame()
    rows = []
    for _, frame in draws.groupby(PROMPT_KEYS + ["step_index"], sort=True, dropna=False):
        if frame.draw_index.duplicated().any():
            raise ValueError("Duplicate conditional forward draw")
        frame = frame.sort_values("draw_index")
        loss = frame.loss_squared_l2.to_numpy(dtype=np.float64)
        if not np.isfinite(loss).all() or (loss < 0).any():
            raise ValueError("Missing or negative measured conditional loss")
        first = frame.iloc[0]
        n, d, snr = len(loss), int(first.latent_dimension), float(first.snr)
        variance = float(np.var(loss, ddof=1)) if n > 1 else math.nan
        fields = PROMPT_KEYS + [
            "timestep", "step_index", "alpha", "sigma", "snr", "latent_dimension",
            "input_source", "target_latent_sha256", "KL_target_to_gaussian", "pinsker_tv_bound", "pinsker_interpretation",
        ]
        row = {name: first[name] for name in fields}
        mean = float(np.mean(loss))
        row.update(
            draw_count=n,
            loss_sum_squared_l2=float(np.sum(loss)),
            loss_mean_squared_l2=mean,
            loss_variance_squared_l2=variance,
            loss_mc_se_squared_l2=math.sqrt(variance / n),
            loss_over_snr_mse=mean / (d * snr),
            loss_over_snr_rmse=math.sqrt(mean / (d * snr)),
            forward_clean_error_mean_squared_l2=float(frame.clean_error_squared_l2.mean()),
            identity_residual_max_abs_squared_l2=float(frame.identity_residual_squared_l2.abs().max()),
            identity_numeric_pass=bool(frame.identity_numeric_pass.all()),
            status="measured",
            averaging_unit="independent_forward_draws_per_pair_and_native_timestep",
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gaussian_conditional(observations):
    """Prompt/noise descriptive statistics; sqrt(mean(e²)) is not mean(e)."""
    if observations.empty:
        return pd.DataFrame()
    result = []
    for _, frame in observations.groupby(PROMPT_KEYS + ["step_index"], sort=True, dropna=False):
        if frame.seed.duplicated().any():
            raise ValueError("Duplicate Gaussian seed in one prompt/noise group")
        first = frame.iloc[0]
        squared = frame.conditional_error_squared_l2.to_numpy(dtype=np.float64)
        norm = frame.conditional_error_l2.to_numpy(dtype=np.float64)
        dimension = int(first.latent_dimension)
        if not np.isfinite(squared).all() or (squared < 0).any():
            raise ValueError("Invalid Gaussian conditional squared errors")
        row = {name: first[name] for name in PROMPT_KEYS + ["step_index", "timestep", "snr", "latent_dimension", "input_source"]}
        q25, median, q75 = np.quantile(norm / math.sqrt(dimension), [0.25, 0.5, 0.75])
        row.update(
            gaussian_seed_count=len(frame),
            gaussian_mean_error_l2=float(np.mean(norm)),
            gaussian_mean_squared_error_l2=float(np.mean(squared)),
            gaussian_root_mean_squared_error_l2=math.sqrt(float(np.mean(squared))),
            gaussian_mean_error_rmse=float(np.mean(norm)) / math.sqrt(dimension),
            gaussian_root_mean_squared_error_rmse=math.sqrt(float(np.mean(squared)) / dimension),
            gaussian_error_q25_rmse=float(q25), gaussian_error_median_rmse=float(median),
            gaussian_error_q75_rmse=float(q75),
            gaussian_error_minimum_rmse=float(np.min(norm)) / math.sqrt(dimension),
            gaussian_error_maximum_rmse=float(np.max(norm)) / math.sqrt(dimension),
            dependence="shared_evaluation_seeds_across_prompts; descriptive_not_independent_loss_replicates",
        )
        result.append(row)
    return pd.DataFrame(result)
