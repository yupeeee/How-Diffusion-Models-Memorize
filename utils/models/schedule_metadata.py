"""Build the shared, exact CPU schedule description for a trajectory run."""

from __future__ import annotations

import math
import operator
from typing import Any

import torch

from .prediction_conversion import (
    STORED_PREDICTION_TYPE,
    prediction_conversion_description,
    schedule_alpha_sigma,
    scheduler_prediction_type,
)
from .schedulers import SCHEDULER_NAMES, scheduler_config_dict


class ScheduleMetadataError(ValueError):
    """Report an invalid or incomplete scheduler trajectory description."""


def build_schedule_metadata(
    scheduler: Any,
    scheduler_name: str,
    *,
    num_inference_steps: int,
    device: str | torch.device,
) -> dict[str, object]:
    """Return the exact shared schedule payload using CPU scientific tensors."""

    if scheduler_name not in SCHEDULER_NAMES:
        choices = ", ".join(SCHEDULER_NAMES)
        raise ScheduleMetadataError(
            f"Unknown scheduler {scheduler_name!r}; expected one of: {choices}"
        )
    if isinstance(num_inference_steps, bool):
        raise ScheduleMetadataError("num_inference_steps must be a positive integer")
    try:
        steps = operator.index(num_inference_steps)
    except TypeError as error:
        raise ScheduleMetadataError(
            "num_inference_steps must be a positive integer"
        ) from error
    if steps <= 0:
        raise ScheduleMetadataError("num_inference_steps must be a positive integer")

    scheduler.set_timesteps(steps, device=torch.device(device))
    raw_timesteps = getattr(scheduler, "timesteps", None)
    if raw_timesteps is None:
        raise ScheduleMetadataError("scheduler.set_timesteps did not create timesteps")
    try:
        timestep_values = torch.as_tensor(raw_timesteps).detach().to(device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ScheduleMetadataError("scheduler.timesteps must be numeric") from error
    if timestep_values.ndim != 1 or timestep_values.numel() != steps:
        raise ScheduleMetadataError(
            f"scheduler must produce exactly {steps} one-dimensional timesteps"
        )
    if timestep_values.is_floating_point():
        if not bool(torch.isfinite(timestep_values).all().item()):
            raise ScheduleMetadataError("scheduler.timesteps contains NaN or infinity")
        if not torch.equal(timestep_values, timestep_values.round()):
            raise ScheduleMetadataError("DDIM/DDPM timesteps must be integer-valued")
    timesteps = timestep_values.to(dtype=torch.int64).contiguous()

    alpha_t, sigma_t, cumulative_t = schedule_alpha_sigma(
        scheduler, timesteps, dtype=torch.float32
    )
    if not (len(alpha_t) == len(sigma_t) == len(cumulative_t) == steps):
        raise ScheduleMetadataError("schedule coefficient lengths do not match T")
    try:
        init_noise_sigma = float(getattr(scheduler, "init_noise_sigma", None))
    except (TypeError, ValueError) as error:
        raise ScheduleMetadataError(
            "scheduler.init_noise_sigma must be numeric"
        ) from error
    if not math.isfinite(init_noise_sigma) or init_noise_sigma <= 0:
        raise ScheduleMetadataError(
            "scheduler.init_noise_sigma must be finite and positive"
        )

    native_type = scheduler_prediction_type(scheduler)
    return {
        "timesteps": timesteps,
        "alpha_t": alpha_t,
        "sigma_t": sigma_t,
        "alphas_cumprod_t": cumulative_t,
        "init_noise_sigma": init_noise_sigma,
        "prediction_type": native_type,
        "native_prediction_type": native_type,
        "stored_prediction_type": STORED_PREDICTION_TYPE,
        "prediction_conversion": prediction_conversion_description(native_type),
        "scheduler_name": scheduler_name,
        "scheduler_class": type(scheduler).__name__,
        "scheduler_config": scheduler_config_dict(scheduler),
        "trajectory_order": "noise_to_image",
        "latent_index_description": (
            "latents[:, 0] is the initial scheduler-scaled x_T; "
            "latents[:, i + 1] is the state after reverse update i; "
            "latents[:, T] is terminal x_0"
        ),
        "prediction_index_description": (
            "branch_predictions[:, i] are canonical epsilon predictions "
            "evaluated from latents[:, i] at timesteps[i]"
        ),
    }
