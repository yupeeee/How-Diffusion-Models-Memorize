"""Convert native diffusion-model predictions to canonical epsilon values."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from typing import Any

import torch

STORED_PREDICTION_TYPE = "epsilon"
SUPPORTED_PREDICTION_TYPES = ("epsilon", "v_prediction", "sample")

_CONVERSION_DESCRIPTIONS = {
    "epsilon": "epsilon = model_output",
    "v_prediction": "epsilon = alpha_t * model_output + sigma_t * x_t",
    "sample": "epsilon = (x_t - alpha_t * model_output) / sigma_t",
}


class PredictionConversionError(ValueError):
    """Report an invalid scheduler prediction type or diffusion coefficient."""


def scheduler_prediction_type(scheduler: Any) -> str:
    """Return and validate the scheduler's native model-prediction type."""

    config = getattr(scheduler, "config", None)
    if isinstance(config, Mapping):
        value = config.get("prediction_type")
    else:
        value = getattr(config, "prediction_type", None)
    if value not in SUPPORTED_PREDICTION_TYPES:
        choices = ", ".join(SUPPORTED_PREDICTION_TYPES)
        raise PredictionConversionError(
            f"Unsupported scheduler prediction_type {value!r}; expected one of: "
            f"{choices}"
        )
    return value


def prediction_conversion_description(prediction_type: str) -> str:
    """Return the exact native-to-epsilon formula used for run metadata."""

    try:
        return _CONVERSION_DESCRIPTIONS[prediction_type]
    except (KeyError, TypeError) as error:
        choices = ", ".join(SUPPORTED_PREDICTION_TYPES)
        raise PredictionConversionError(
            f"Unsupported prediction_type {prediction_type!r}; expected one of: "
            f"{choices}"
        ) from error


def alpha_sigma_for_timestep(
    scheduler: Any,
    timestep: int | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scalar ``alpha_t`` and ``sigma_t`` for one exact timestep.

    ``alpha_t`` is ``sqrt(alphas_cumprod[t])`` and ``sigma_t`` is
    ``sqrt(1 - alphas_cumprod[t])``.  The returned tensors use the explicitly
    requested device and floating dtype so callers can either do native model
    arithmetic or save a non-half-precision schedule.
    """

    if (
        not isinstance(dtype, torch.dtype)
        or not torch.empty((), dtype=dtype).is_floating_point()
    ):
        raise PredictionConversionError("coefficient dtype must be floating point")
    index = _timestep_index(timestep)
    cumulative = getattr(scheduler, "alphas_cumprod", None)
    if cumulative is None:
        raise PredictionConversionError("scheduler has no alphas_cumprod")
    try:
        values = torch.as_tensor(cumulative)
    except (TypeError, ValueError, RuntimeError) as error:
        raise PredictionConversionError(
            "scheduler.alphas_cumprod must be a one-dimensional numeric sequence"
        ) from error
    if values.ndim != 1 or values.numel() == 0:
        raise PredictionConversionError(
            "scheduler.alphas_cumprod must be a non-empty one-dimensional sequence"
        )
    if index >= values.numel():
        raise PredictionConversionError(
            f"timestep {index} is outside scheduler.alphas_cumprod of length "
            f"{values.numel()}"
        )
    value = values[index].detach().to(device="cpu", dtype=torch.float64)
    cumulative_alpha = float(value.item())
    if not math.isfinite(cumulative_alpha) or not 0.0 <= cumulative_alpha <= 1.0:
        raise PredictionConversionError(
            f"alphas_cumprod[{index}] must be finite and in [0, 1]"
        )
    selected_device = torch.device(device)
    alpha = torch.tensor(
        math.sqrt(cumulative_alpha), device=selected_device, dtype=dtype
    )
    sigma = torch.tensor(
        math.sqrt(1.0 - cumulative_alpha), device=selected_device, dtype=dtype
    )
    return alpha, sigma


def schedule_alpha_sigma(
    scheduler: Any,
    timesteps: Sequence[int] | torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU schedule coefficients in the supplied reverse-step order.

    The tuple contains ``alpha_t``, ``sigma_t``, and ``alphas_cumprod_t``.
    This helper intentionally preserves the order of ``timesteps`` rather than
    sorting timestep indices.
    """

    if isinstance(timesteps, torch.Tensor):
        if timesteps.ndim != 1:
            raise PredictionConversionError("timesteps must be one-dimensional")
        ordered_timesteps: Sequence[int | torch.Tensor] = tuple(timesteps)
    elif isinstance(timesteps, Sequence) and not isinstance(timesteps, (str, bytes)):
        ordered_timesteps = timesteps
    else:
        raise PredictionConversionError("timesteps must be a non-empty sequence")
    if not ordered_timesteps:
        raise PredictionConversionError("timesteps must be a non-empty sequence")

    alphas: list[torch.Tensor] = []
    sigmas: list[torch.Tensor] = []
    for timestep in ordered_timesteps:
        alpha, sigma = alpha_sigma_for_timestep(
            scheduler, timestep, device="cpu", dtype=dtype
        )
        alphas.append(alpha)
        sigmas.append(sigma)
    alpha_t = torch.stack(alphas).contiguous()
    sigma_t = torch.stack(sigmas).contiguous()
    cumulative_t = torch.tensor(
        [
            float(
                torch.as_tensor(scheduler.alphas_cumprod)[_timestep_index(t)]
                .detach()
                .cpu()
            )
            for t in ordered_timesteps
        ],
        dtype=dtype,
    ).contiguous()
    return alpha_t, sigma_t, cumulative_t


def native_prediction_to_epsilon(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    timestep: int | torch.Tensor,
    scheduler: Any,
    *,
    prediction_type: str | None = None,
) -> torch.Tensor:
    """Convert one native branch output to canonical epsilon semantics.

    ``sample`` must be the unscaled scheduler state ``x_t``.  In particular,
    callers must not pass the value returned by ``scale_model_input``.
    The scheduler continues to receive its native model output; this converted
    tensor is intended only for trajectory storage and epsilon-space analysis.
    """

    _validate_prediction_tensor(model_output, "model_output")
    _validate_prediction_tensor(sample, "sample")
    if model_output.shape != sample.shape:
        raise PredictionConversionError(
            "model_output and unscaled sample must have identical shapes"
        )
    if model_output.device != sample.device:
        raise PredictionConversionError(
            "model_output and unscaled sample must be on the same device"
        )
    if model_output.dtype != sample.dtype:
        raise PredictionConversionError(
            "model_output and unscaled sample must have the same dtype"
        )

    native_type = (
        scheduler_prediction_type(scheduler)
        if prediction_type is None
        else prediction_type
    )
    # Validate an explicit override exactly as strictly as scheduler metadata.
    prediction_conversion_description(native_type)
    if native_type == "epsilon":
        epsilon = model_output
    else:
        alpha_t, sigma_t = alpha_sigma_for_timestep(
            scheduler,
            timestep,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        if native_type == "v_prediction":
            epsilon = alpha_t * model_output + sigma_t * sample
        else:
            if float(sigma_t.detach().float().cpu().item()) == 0.0:
                raise PredictionConversionError(
                    "sample prediction cannot be converted to epsilon when sigma_t is zero"
                )
            epsilon = (sample - alpha_t * model_output) / sigma_t

    if not bool(torch.isfinite(epsilon).all().item()):
        raise PredictionConversionError("converted epsilon contains NaN or infinity")
    return epsilon


def _validate_prediction_tensor(tensor: object, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise PredictionConversionError(f"{name} must be a torch tensor")
    if tensor.ndim < 1 or not tensor.is_floating_point():
        raise PredictionConversionError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(tensor).all().item()):
        raise PredictionConversionError(f"{name} contains NaN or infinity")


def _timestep_index(timestep: int | torch.Tensor) -> int:
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() != 1:
            raise PredictionConversionError("timestep must be a scalar integer")
        timestep = timestep.detach().to(device="cpu").item()
    if isinstance(timestep, bool):
        raise PredictionConversionError("timestep must be a non-negative integer")
    try:
        index = operator.index(timestep)
    except TypeError as error:
        raise PredictionConversionError(
            "timestep must be a non-negative integer"
        ) from error
    if index < 0:
        raise PredictionConversionError("timestep must be a non-negative integer")
    return index
