"""Batched sufficient sign certificates using only saved inputs and float64.

For positive alpha and nonnegative sigma, D < ec is equivalent to

    ||sigma * (epsilon_u - epsilon_c)||²
        < ||state - sigma * epsilon_c - alpha * target||².

This proves M = D - ec - eu - V < 0 because eu and V are nonnegative.
It proves neither M's magnitude nor a value for V. Every elementary operation
is separately evaluated in eager IEEE-754 binary64 arithmetic and expanded
outward with nextafter. Reductions use an explicit pairwise tree, never a
backend-dependent floating reduction. No division, square root, transcendental,
matrix multiplication, or approximate posterior calculation enters this proof.

The contract is eager CPU/CUDA PyTorch with correctly rounded binary64 basic
operations; this function must not be compiled with fast-math reassociation or
fusion. Nonfinite, overflow, or subnormal arithmetic conservatively falls back
to an unresolved result. Dtype/source perturbations are outside this scope.
"""
from __future__ import annotations

import math

import torch


SCREENING_VERSION = "outward-binary64-squared-precheck-2"


class _OutwardBatch:
    """Track per-row exceptional arithmetic without host synchronizations."""

    def __init__(self, batch, device):
        self.valid = torch.ones(batch, dtype=torch.bool, device=device)

    def observe(self, value):
        # Bit classification also detects subnormals when a host happens to
        # have enabled denormal flushing. No floating comparison decides this.
        bits = value.contiguous().view(torch.int64)
        exponent = bits & 0x7FF0000000000000
        fraction = bits & 0x000FFFFFFFFFFFFF
        exceptional = (exponent == 0x7FF0000000000000) | ((exponent == 0) & (fraction != 0))
        self.valid &= ~exceptional.reshape(len(self.valid), -1).any(dim=1)

    def rounded(self, operation, left, right):
        if operation == "add":
            value = left + right
            exact_zero = left == -right
        elif operation == "subtract":
            value = left - right
            exact_zero = left == right
        elif operation == "multiply":
            value = left * right
            exact_zero = (left == 0) | (right == 0)
        else:
            raise ValueError("Unsupported screening operation")
        self.observe(value)
        # A product that underflows all the way to zero must not pass merely
        # because zero has no subnormal bits. This also guards flushed results.
        unexpected_zero = (value == 0) & ~exact_zero
        self.valid &= ~unexpected_zero.reshape(len(self.valid), -1).any(dim=1)
        lower = torch.nextafter(value, torch.full_like(value, -torch.inf))
        upper = torch.nextafter(value, torch.full_like(value, torch.inf))
        # These algebraic zeros are exact for finite operands, and retaining
        # them avoids artificial subnormals and needless widening of padding.
        lower = torch.where(exact_zero, torch.zeros_like(lower), lower)
        upper = torch.where(exact_zero, torch.zeros_like(upper), upper)
        self.observe(lower)
        self.observe(upper)
        return lower, upper

    def scale_positive(self, interval, coefficient):
        lower = self.rounded("multiply", interval[0], coefficient)[0]
        upper = self.rounded("multiply", interval[1], coefficient)[1]
        return lower, upper

    def subtract(self, left, right):
        lower = self.rounded("subtract", left[0], right[1])[0]
        upper = self.rounded("subtract", left[1], right[0])[1]
        return lower, upper

    def squared_norm_bound(self, interval, *, upper):
        lower, higher = interval
        if upper:
            absolute = torch.maximum(lower.abs(), higher.abs())
            entries = self.rounded("multiply", absolute, absolute)[1]
        else:
            includes_zero = (lower <= 0) & (higher >= 0)
            absolute = torch.where(includes_zero, torch.zeros_like(lower),
                                   torch.minimum(lower.abs(), higher.abs()))
            entries = self.rounded("multiply", absolute, absolute)[0].clamp_min(0)
        while entries.shape[1] > 1:
            pairs = entries.shape[1] // 2
            combined = self.rounded("add", entries[:, :2 * pairs:2],
                                    entries[:, 1:2 * pairs:2])[int(upper)]
            # Carry an odd last entry unchanged; no fictitious summand is used.
            if entries.shape[1] % 2:
                combined = torch.cat((combined, entries[:, -1:]), dim=1)
            entries = combined
        return entries[:, 0]


def _cpu_denormal_mode_is_supported():
    """Check this thread's mode without changing process-wide arithmetic state."""
    normal64 = torch.tensor([torch.finfo(torch.float64).tiny], dtype=torch.float64, device="cpu")
    half64 = normal64 * .5
    restored64 = half64 * 2.
    normal32 = torch.tensor([torch.finfo(torch.float32).tiny], dtype=torch.float32, device="cpu")
    half32 = normal32 * .5
    restored32 = half32.to(dtype=torch.float64) * 2.
    return torch.equal(restored64, normal64) and torch.equal(restored32, normal32.to(dtype=torch.float64))


def _saved_tensor(value, *, device):
    # Python floats are binary64; do not pass them through the default float32
    # tensor dtype before widening. Existing floating tensor/array dtypes are
    # preserved through transport, then exactly promoted to binary64.
    tensor = torch.as_tensor(value, device=device, dtype=None if hasattr(value, "dtype") else torch.float64)
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Numerical screening requires saved floating tensors")
    return tensor.detach().to(dtype=torch.float64)


@torch.no_grad()
def screen_negative_condition(verified_inputs, *, device=None):
    """Return a batched original-condition sign proof, without CPU row loops.

    All numerical bounds are in squared *numerator* units (alpha² times the
    corresponding squared clean error), not L2 margin units. Strict negativity
    of squared_difference_upper certifies only M < 0. A nonnegative bound is
    inconclusive, and must never be interpreted as a nonnegative condition.
    A nonnegative squared_difference_lower establishes D >= ec: the cheap
    D - ec precheck cannot prove M < 0, but M itself may still be negative
    because eu and V have not been evaluated.
    """
    x = verified_inputs
    if x.get("stored_prediction_type", "epsilon") != "epsilon":
        raise ValueError("Numerical screening requires already-canonical epsilon")
    alpha, sigma = float(x["alpha"]), float(x["sigma"])
    if not math.isfinite(alpha) or not math.isfinite(sigma) or alpha <= 0 or sigma < 0:
        raise ValueError("Numerical screening requires finite alpha > 0 and sigma >= 0")
    state = _saved_tensor(x["state"], device=device)
    if state.ndim < 2 or state.shape[0] < 1:
        raise ValueError("Numerical screening requires a nonempty batch of vectors")
    device = state.device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Numerical screening supports only eager CPU/CUDA binary64 arithmetic")
    state = state.reshape(len(state), -1)
    if state.shape[1] < 1:
        raise ValueError("Numerical screening requires nonempty vectors")
    epsilon_u, epsilon_c = [_saved_tensor(x[name], device=device) for name in ("epsilon_u", "epsilon_c")]
    if any(value.ndim < 2 or len(value) != len(state) for value in (epsilon_u, epsilon_c)):
        raise ValueError("Numerical screening batch shapes differ")
    epsilon_u, epsilon_c = [value.reshape(len(state), -1) for value in (epsilon_u, epsilon_c)]
    target = _saved_tensor(x["target"], device=device).reshape(1, -1)
    if epsilon_u.shape != state.shape or epsilon_c.shape != state.shape or target.shape[1] != state.shape[1]:
        raise ValueError("Numerical screening vector dimensions differ")
    target = target.expand_as(state)
    arithmetic = _OutwardBatch(len(state), device)
    if device.type == "cpu" and not _cpu_denormal_mode_is_supported():
        arithmetic.valid.fill_(False)
    alpha_tensor = torch.full((len(state), 1), alpha, dtype=torch.float64, device=device)
    sigma_tensor = torch.full_like(alpha_tensor, sigma)
    for value in (state, epsilon_u, epsilon_c, target, alpha_tensor, sigma_tensor):
        arithmetic.observe(value)
    difference = arithmetic.rounded("subtract", epsilon_u, epsilon_c)
    delta = arithmetic.scale_positive(difference, sigma_tensor)
    conditional_noise = arithmetic.rounded("multiply", epsilon_c, sigma_tensor)
    scaled_target = arithmetic.rounded("multiply", target, alpha_tensor)
    residual = arithmetic.subtract(arithmetic.subtract((state, state), conditional_noise), scaled_target)
    delta_lower = arithmetic.squared_norm_bound(delta, upper=False)
    delta_upper = arithmetic.squared_norm_bound(delta, upper=True)
    residual_lower = arithmetic.squared_norm_bound(residual, upper=False)
    residual_upper = arithmetic.squared_norm_bound(residual, upper=True)
    squared_lower = arithmetic.rounded("subtract", delta_lower[:, None], residual_upper[:, None])[0][:, 0]
    squared_upper = arithmetic.rounded("subtract", delta_upper[:, None], residual_lower[:, None])[1][:, 0]
    resolved = arithmetic.valid
    nan = torch.full_like(delta_upper, torch.nan)
    return {
        "certified_negative": resolved & (squared_upper < 0),
        "arithmetic_resolved": resolved,
        "delta_squared_lower": torch.where(resolved, delta_lower, nan),
        "delta_squared_upper": torch.where(resolved, delta_upper, nan),
        "residual_squared_lower": torch.where(resolved, residual_lower, nan),
        "residual_squared_upper": torch.where(resolved, residual_upper, nan),
        "squared_difference_lower": torch.where(resolved, squared_lower, nan),
        "squared_difference_upper": torch.where(resolved, squared_upper, nan),
        "method": "batched_float64_directed_squared_negative_precheck",
        "version": SCREENING_VERSION,
        "scope": "original_saved_state_canonical_epsilon_target_and_exact_saved_alpha_sigma",
        "proof_quantity": "upper_bound_for_alpha_squared_times_D_squared_minus_ec_squared",
        "proof_quantity_lower": "lower_bound_for_alpha_squared_times_D_squared_minus_ec_squared",
    }
