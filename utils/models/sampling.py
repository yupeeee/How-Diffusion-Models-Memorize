"""Reusable classifier-free-guided full-trajectory diffusion sampling."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from utils.common.cli import MAX_SEED

from .devices import DeviceNormalizationError, devices_match
from .prediction_conversion import (
    native_prediction_to_epsilon,
    scheduler_prediction_type,
)
from .schedulers import SCHEDULER_NAMES, scheduler_step_kwargs

SAMPLER_CONTRACT_VERSION = 2


class SamplingError(RuntimeError):
    """Report an invalid model interface or non-finite sampling result."""


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """Hold noise-to-image latent states and both canonical epsilon branches.

    All three tensors are detached, contiguous, and on CPU. For ``N`` seeds
    and ``T`` denoising updates, ``latents`` has shape ``[N,T+1,C,H,W]`` and
    each branch prediction has shape ``[N,T,C,H,W]``.
    """

    latents: torch.Tensor
    unconditional_noise_predictions: torch.Tensor
    conditional_noise_predictions: torch.Tensor

    def __post_init__(self) -> None:
        _validate_trajectory_batch(self)

    @property
    def num_seeds(self) -> int:
        """Return ``N``."""

        return self.latents.shape[0]

    @property
    def num_inference_steps(self) -> int:
        """Return ``T``."""

        return self.unconditional_noise_predictions.shape[1]

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        """Return ``[C,H,W]`` as a tuple."""

        values = self.latents.shape[2:]
        return values[0], values[1], values[2]

    @property
    def noise_predictions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the storage tuple in unconditional, conditional order."""

        return (
            self.unconditional_noise_predictions,
            self.conditional_noise_predictions,
        )


@dataclass(frozen=True, slots=True)
class _MicrobatchTrajectory:
    latents: torch.Tensor
    unconditional: torch.Tensor
    conditional: torch.Tensor


def seeds_from_count(count: int) -> list[int]:
    """Return the experiment's fixed seed sequence ``0`` through ``N - 1``."""

    size = _positive_integer(count, "seed count")
    return list(range(size))


def latent_shape_from_unet(unet: Any) -> tuple[int, int, int]:
    """Return ``[C,H,W]`` from a Stable Diffusion UNet configuration."""

    config = getattr(unet, "config", None)
    channels = _config_value(config, "in_channels")
    sample_size = _config_value(config, "sample_size")
    try:
        channel_count = operator.index(channels)
    except TypeError as error:
        raise SamplingError("UNet config.in_channels must be an integer") from error
    if channel_count <= 0:
        raise SamplingError("UNet config.in_channels must be positive")

    if not isinstance(sample_size, bool):
        try:
            side = operator.index(sample_size)
        except TypeError:
            side = None
        if side is not None:
            if side <= 0:
                raise SamplingError("UNet config.sample_size must be positive")
            return channel_count, side, side
    if isinstance(sample_size, Sequence) and not isinstance(sample_size, str):
        if len(sample_size) == 2:
            height = _positive_integer(sample_size[0], "UNet sample height")
            width = _positive_integer(sample_size[1], "UNet sample width")
            return channel_count, height, width
    raise SamplingError("UNet config.sample_size must be an integer or [H, W]")


def validate_latent_shape(
    unet: Any,
    expected_shape: Sequence[int] = (4, 64, 64),
) -> tuple[int, int, int]:
    """Derive and require the registered model's expected latent shape."""

    actual = latent_shape_from_unet(unet)
    expected_values = tuple(
        _positive_integer(value, "expected latent dimension")
        for value in expected_shape
    )
    if len(expected_values) != 3:
        raise SamplingError("expected latent shape must contain [C, H, W]")
    if actual != expected_values:
        raise SamplingError(
            f"UNet latent shape {actual} does not match the registered shape "
            f"{expected_values}"
        )
    return actual


def make_initial_noise(
    seeds: Sequence[int],
    latent_shape: Sequence[int],
) -> torch.Tensor:
    """Generate per-seed standard Gaussian noise on CPU in float32."""

    noise, _ = _noise_and_generators(seeds, latent_shape)
    return noise


def sample_trajectory(
    *,
    prompt: str,
    seeds: Sequence[int],
    tokenizer: Any,
    text_encoder: Any,
    unet: Any,
    scheduler: Any,
    device: str | torch.device,
    inference_dtype: torch.dtype,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler_name: str,
    initial_microbatch_size: int = 4,
    progress_callback: Callable[[int], None] | None = None,
) -> TrajectoryBatch:
    """Sample and record every latent and both CFG epsilon branches.

    The scheduler always receives the guided *native* prediction. Saved branch
    tensors are separately converted to epsilon using the unscaled current
    latent. One persistent CPU generator per seed supplies both its initial
    noise and any DDPM stochastic-step noise. The optional progress callback
    receives signed logical seed-step deltas; a failed OOM attempt is rolled
    back before retrying with a smaller microbatch.
    """

    if not isinstance(prompt, str):
        raise SamplingError("prompt must be the exact raw string")
    seed_values = _validated_seeds(seeds)
    steps = _positive_integer(num_inference_steps, "num_inference_steps")
    microbatch_size = _positive_integer(
        initial_microbatch_size, "initial_microbatch_size"
    )
    if scheduler_name not in SCHEDULER_NAMES:
        choices = ", ".join(SCHEDULER_NAMES)
        raise ValueError(
            f"Unknown scheduler {scheduler_name!r}; expected one of: {choices}"
        )
    try:
        guidance = float(guidance_scale)
    except (TypeError, ValueError) as error:
        raise SamplingError("guidance_scale must be a finite float") from error
    if not math.isfinite(guidance):
        raise SamplingError("guidance_scale must be a finite float")
    if not isinstance(inference_dtype, torch.dtype):
        raise SamplingError("inference_dtype must be a torch dtype")
    if not torch.empty((), dtype=inference_dtype).is_floating_point():
        raise SamplingError("inference_dtype must be floating point")

    execution_device, execution_dtype = _unet_execution_device_and_dtype(
        unet,
        intended_device=device,
        inference_dtype=inference_dtype,
    )
    latent_shape = latent_shape_from_unet(unet)
    native_prediction_type = scheduler_prediction_type(scheduler)
    with torch.inference_mode():
        unconditional, conditional = _encode_cfg_conditions(
            prompt,
            tokenizer,
            text_encoder,
            execution_device,
            execution_dtype,
        )
        if (
            unconditional.device != execution_device
            or conditional.device != execution_device
        ):
            raise SamplingError(
                "Conditional and unconditional embeddings must be on the concrete "
                f"UNet device {execution_device}"
            )
        parts = _sample_adaptive(
            seed_values=seed_values,
            latent_shape=latent_shape,
            unconditional=unconditional,
            conditional=conditional,
            unet=unet,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            native_prediction_type=native_prediction_type,
            device=execution_device,
            inference_dtype=execution_dtype,
            guidance_scale=guidance,
            num_inference_steps=steps,
            initial_microbatch_size=microbatch_size,
            progress_callback=progress_callback,
        )

    result = TrajectoryBatch(
        latents=torch.cat([part.latents for part in parts], dim=0).contiguous(),
        unconditional_noise_predictions=torch.cat(
            [part.unconditional for part in parts], dim=0
        ).contiguous(),
        conditional_noise_predictions=torch.cat(
            [part.conditional for part in parts], dim=0
        ).contiguous(),
    )
    expected_latents = (len(seed_values), steps + 1, *latent_shape)
    if result.latents.shape != expected_latents:
        raise SamplingError(
            f"Unexpected latent trajectory shape: {tuple(result.latents.shape)}"
        )
    return result


def _sample_adaptive(
    *,
    seed_values: list[int],
    latent_shape: tuple[int, int, int],
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    unet: Any,
    scheduler: Any,
    scheduler_name: str,
    native_prediction_type: str,
    device: torch.device,
    inference_dtype: torch.dtype,
    guidance_scale: float,
    num_inference_steps: int,
    initial_microbatch_size: int,
    progress_callback: Callable[[int], None] | None,
) -> list[_MicrobatchTrajectory]:
    outputs: list[_MicrobatchTrajectory] = []
    position = 0
    batch_size = min(initial_microbatch_size, len(seed_values))
    while position < len(seed_values):
        current_size = min(batch_size, len(seed_values) - position)
        batch_seeds = seed_values[position : position + current_size]
        retry_after_oom = False
        attempt_progress = 0

        def report_progress(delta: int) -> None:
            nonlocal attempt_progress
            attempt_progress += delta
            if progress_callback is not None:
                progress_callback(delta)

        try:
            trajectory = _sample_microbatch(
                seeds=batch_seeds,
                latent_shape=latent_shape,
                unconditional=unconditional,
                conditional=conditional,
                unet=unet,
                scheduler=scheduler,
                scheduler_name=scheduler_name,
                native_prediction_type=native_prediction_type,
                device=device,
                inference_dtype=inference_dtype,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                progress_callback=report_progress,
            )
        except RuntimeError as error:
            if not _is_cuda_out_of_memory(error, device) or current_size == 1:
                raise
            if progress_callback is not None and attempt_progress:
                progress_callback(-attempt_progress)
            batch_size = max(1, current_size // 2)
            retry_after_oom = True
        if retry_after_oom:
            # Exit the handler first so failed-forward traceback references are
            # released before asking CUDA's allocator to release cached blocks.
            torch.cuda.empty_cache()
            continue
        outputs.append(trajectory)
        position += current_size
    return outputs


def _sample_microbatch(
    *,
    seeds: Sequence[int],
    latent_shape: tuple[int, int, int],
    unconditional: torch.Tensor,
    conditional: torch.Tensor,
    unet: Any,
    scheduler: Any,
    scheduler_name: str,
    native_prediction_type: str,
    device: torch.device,
    inference_dtype: torch.dtype,
    guidance_scale: float,
    num_inference_steps: int,
    progress_callback: Callable[[int], None] | None,
) -> _MicrobatchTrajectory:
    noise, generators = _noise_and_generators(seeds, latent_shape)
    latents = noise.to(device=device, dtype=inference_dtype)
    latents = latents * _finite_noise_sigma(scheduler)
    _validate_current_latents(
        latents,
        batch_size=len(seeds),
        latent_shape=latent_shape,
        expected_device=device,
        expected_dtype=inference_dtype,
        name="initial latents",
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    raw_timesteps = getattr(scheduler, "timesteps", None)
    if raw_timesteps is None:
        raise SamplingError("scheduler.set_timesteps did not create timesteps")
    try:
        timesteps = tuple(raw_timesteps)
    except TypeError as error:
        raise SamplingError("scheduler.timesteps must be iterable") from error
    if len(timesteps) != num_inference_steps:
        raise SamplingError(
            "scheduler produced "
            f"{len(timesteps)} timesteps for {num_inference_steps} requested steps"
        )
    step_keywords = scheduler_step_kwargs(
        scheduler,
        scheduler_name,
        generators=generators if scheduler_name == "ddpm" else None,
    )

    batch_size = len(seeds)
    latent_trajectory = torch.empty(
        (batch_size, num_inference_steps + 1, *latent_shape),
        device="cpu",
        dtype=inference_dtype,
    )
    unconditional_epsilon = torch.empty(
        (batch_size, num_inference_steps, *latent_shape),
        device="cpu",
        dtype=inference_dtype,
    )
    conditional_epsilon = torch.empty_like(unconditional_epsilon)
    _copy_to_cpu(latent_trajectory[:, 0], latents)

    condition_batch = torch.cat(
        [
            unconditional.expand(batch_size, *unconditional.shape[1:]),
            conditional.expand(batch_size, *conditional.shape[1:]),
        ],
        dim=0,
    )
    expected_prediction_shape = (2 * batch_size, *latent_shape)
    for step_index, timestep in enumerate(timesteps):
        # This unscaled state is both stored latent i and conversion x_t.
        latent_input = torch.cat([latents, latents], dim=0)
        scaled_input = scheduler.scale_model_input(latent_input, timestep)
        if not isinstance(scaled_input, torch.Tensor):
            raise SamplingError("scheduler.scale_model_input must return a tensor")
        _validate_current_latents(
            scaled_input,
            batch_size=2 * batch_size,
            latent_shape=latent_shape,
            expected_device=latent_input.device,
            expected_dtype=inference_dtype,
            name=f"scheduler-scaled model input at step {step_index}",
        )
        model_output = unet(
            scaled_input,
            timestep,
            encoder_hidden_states=condition_batch,
        )
        native_prediction = _output_tensor(model_output, "sample", "UNet")
        _validate_native_prediction(
            native_prediction,
            expected_shape=expected_prediction_shape,
            expected_device=scaled_input.device,
            expected_dtype=inference_dtype,
        )
        unconditional_native, conditional_native = native_prediction.chunk(2)

        unconditional_at_t = native_prediction_to_epsilon(
            unconditional_native,
            latents,
            timestep,
            scheduler,
            prediction_type=native_prediction_type,
        )
        conditional_at_t = native_prediction_to_epsilon(
            conditional_native,
            latents,
            timestep,
            scheduler,
            prediction_type=native_prediction_type,
        )
        _copy_to_cpu(unconditional_epsilon[:, step_index], unconditional_at_t)
        _copy_to_cpu(conditional_epsilon[:, step_index], conditional_at_t)

        # Guidance remains in the checkpoint's native prediction space.
        guided_native = unconditional_native + guidance_scale * (
            conditional_native - unconditional_native
        )
        _validate_current_latents(
            guided_native,
            batch_size=batch_size,
            latent_shape=latent_shape,
            expected_device=latents.device,
            expected_dtype=inference_dtype,
            name=f"guided native prediction at step {step_index}",
        )
        scheduler_output = scheduler.step(
            guided_native,
            timestep,
            latents,
            **step_keywords,
        )
        previous_latent_device = latents.device
        latents = _output_tensor(scheduler_output, "prev_sample", "scheduler")
        _validate_current_latents(
            latents,
            batch_size=batch_size,
            latent_shape=latent_shape,
            expected_device=previous_latent_device,
            expected_dtype=inference_dtype,
            name=f"latents after reverse step {step_index}",
        )
        _copy_to_cpu(latent_trajectory[:, step_index + 1], latents)
        if progress_callback is not None:
            progress_callback(batch_size)

    return _MicrobatchTrajectory(
        latents=latent_trajectory.contiguous(),
        unconditional=unconditional_epsilon.contiguous(),
        conditional=conditional_epsilon.contiguous(),
    )


def _encode_cfg_conditions(
    prompt: str,
    tokenizer: Any,
    text_encoder: Any,
    device: torch.device,
    inference_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum_length = getattr(tokenizer, "model_max_length", None)
    try:
        maximum_length = operator.index(maximum_length)
    except TypeError as error:
        raise SamplingError("tokenizer.model_max_length must be an integer") from error
    encoded = tokenizer(
        ["", prompt],
        padding="max_length",
        max_length=maximum_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = _token_tensor(encoded, "input_ids").to(device)
    encoder_arguments: dict[str, torch.Tensor] = {}
    encoder_config = getattr(text_encoder, "config", None)
    if bool(_config_value(encoder_config, "use_attention_mask")):
        encoder_arguments["attention_mask"] = _token_tensor(
            encoded, "attention_mask"
        ).to(device)
    encoder_output = text_encoder(input_ids, **encoder_arguments)
    embeddings = _output_tensor(encoder_output, "last_hidden_state", "text encoder")
    if embeddings.ndim != 3 or embeddings.shape[0] != 2:
        raise SamplingError("Text embeddings must have shape [2, tokens, features]")
    if embeddings.device != input_ids.device:
        raise SamplingError(
            "Text embeddings are on a different device from text encoder inputs"
        )
    if not embeddings.is_floating_point():
        raise SamplingError("Text embeddings must be floating point")
    embeddings = embeddings.to(dtype=inference_dtype)
    _require_finite(embeddings, "text embeddings")
    return embeddings[0:1], embeddings[1:2]


def _noise_and_generators(
    seeds: Sequence[int], latent_shape: Sequence[int]
) -> tuple[torch.Tensor, list[torch.Generator]]:
    seed_values = _validated_seeds(seeds)
    shape = tuple(_positive_integer(size, "latent dimension") for size in latent_shape)
    if len(shape) != 3:
        raise SamplingError("latent_shape must contain [C, H, W]")
    generators: list[torch.Generator] = []
    samples: list[torch.Tensor] = []
    for seed in seed_values:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        generators.append(generator)
        samples.append(
            torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        )
    return torch.stack(samples, dim=0), generators


def _validated_seeds(seeds: Sequence[int]) -> list[int]:
    if isinstance(seeds, (str, bytes)):
        raise SamplingError("seeds must be a non-empty integer sequence")
    try:
        candidates = list(seeds)
    except TypeError as error:
        raise SamplingError("seeds must be a non-empty integer sequence") from error
    values: list[int] = []
    for seed in candidates:
        if isinstance(seed, bool):
            raise SamplingError("seeds must contain non-negative integers")
        try:
            value = operator.index(seed)
        except TypeError as error:
            raise SamplingError("seeds must contain non-negative integers") from error
        if value < 0 or value > MAX_SEED:
            raise SamplingError("seeds must be between 0 and 2**63 - 1")
        values.append(value)
    if not values:
        raise SamplingError("seeds must not be empty")
    if len(set(values)) != len(values):
        raise SamplingError("seeds must be unique")
    return values


def _unet_execution_device_and_dtype(
    unet: Any,
    *,
    intended_device: str | torch.device,
    inference_dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    """Resolve and validate the concrete UNet execution placement."""

    parameter_method = getattr(unet, "parameters", None)
    if not callable(parameter_method):
        raise SamplingError("UNet has no callable parameters interface")
    try:
        parameters = tuple(parameter_method(recurse=True))
    except TypeError:
        try:
            parameters = tuple(parameter_method())
        except (TypeError, RuntimeError) as error:
            raise SamplingError("Cannot inspect UNet parameters") from error
    except RuntimeError as error:
        raise SamplingError("Cannot inspect UNet parameters") from error
    if any(not isinstance(parameter, torch.Tensor) for parameter in parameters):
        raise SamplingError("UNet parameters interface returned a non-tensor")

    floating = tuple(
        parameter for parameter in parameters if parameter.is_floating_point()
    )
    if not floating:
        raise SamplingError(
            "UNet has no floating-point parameters; execution placement is unknown"
        )
    devices = {parameter.device for parameter in floating}
    if len(devices) != 1:
        rendered = ", ".join(sorted(str(value) for value in devices))
        raise SamplingError(
            f"UNet floating-point parameters span multiple devices: {rendered}"
        )
    dtypes = {parameter.dtype for parameter in floating}
    if len(dtypes) != 1:
        rendered = ", ".join(sorted(str(value) for value in dtypes))
        raise SamplingError(
            f"UNet floating-point parameters span multiple dtypes: {rendered}"
        )

    execution_device = next(iter(devices))
    execution_dtype = next(iter(dtypes))
    if execution_dtype != inference_dtype:
        raise SamplingError(
            "UNet floating-point parameters use "
            f"{execution_dtype}; expected inference dtype {inference_dtype}"
        )
    try:
        placement_matches = devices_match(execution_device, intended_device)
    except DeviceNormalizationError as error:
        raise SamplingError(f"Cannot resolve intended UNet device: {error}") from error
    if not placement_matches:
        raise SamplingError(
            f"UNet is on {execution_device}; intended execution device is "
            f"{torch.device(intended_device)}"
        )
    return execution_device, execution_dtype


def _validate_trajectory_batch(batch: TrajectoryBatch) -> None:
    tensors = {
        "latents": batch.latents,
        "unconditional noise predictions": batch.unconditional_noise_predictions,
        "conditional noise predictions": batch.conditional_noise_predictions,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise SamplingError(f"{name} must be a torch tensor")
        if tensor.ndim != 5:
            raise SamplingError(f"{name} must have five dimensions [N, steps, C, H, W]")
        if not tensor.is_floating_point():
            raise SamplingError(f"{name} must be floating point")
        if tensor.device.type != "cpu":
            raise SamplingError(f"{name} must be stored on CPU")
        if not tensor.is_contiguous():
            raise SamplingError(f"{name} must be contiguous")
        if tensor.requires_grad:
            raise SamplingError(f"{name} must be detached from autograd")
        if any(size <= 0 for size in tensor.shape):
            raise SamplingError(f"{name} dimensions must be positive")
        _require_finite(tensor, name)

    latents = batch.latents
    unconditional = batch.unconditional_noise_predictions
    conditional = batch.conditional_noise_predictions
    if unconditional.shape != conditional.shape:
        raise SamplingError(
            "unconditional and conditional prediction shapes must be identical"
        )
    expected_prediction_shape = (
        latents.shape[0],
        latents.shape[1] - 1,
        *latents.shape[2:],
    )
    if unconditional.shape != expected_prediction_shape:
        raise SamplingError(
            "prediction shape must be [N, T, C, H, W] when latent shape is "
            "[N, T + 1, C, H, W]"
        )
    if not (latents.dtype == unconditional.dtype == conditional.dtype):
        raise SamplingError("trajectory and branch tensors must have the same dtype")


def _validate_native_prediction(
    prediction: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    if prediction.shape != expected_shape:
        raise SamplingError(
            "UNet prediction has invalid shape: "
            f"expected {expected_shape}, got {tuple(prediction.shape)}"
        )
    if prediction.device != expected_device:
        raise SamplingError(
            "UNet prediction is on the wrong device: "
            f"expected {expected_device}, got {prediction.device}"
        )
    if prediction.dtype != expected_dtype or not prediction.is_floating_point():
        raise SamplingError("UNet prediction must use the inference dtype")
    _require_finite(prediction, "UNet prediction")


def _validate_current_latents(
    latents: torch.Tensor,
    *,
    batch_size: int,
    latent_shape: tuple[int, int, int],
    expected_device: torch.device,
    expected_dtype: torch.dtype,
    name: str,
) -> None:
    if latents.shape != (batch_size, *latent_shape):
        raise SamplingError(
            f"{name} has invalid shape: expected "
            f"{(batch_size, *latent_shape)}, got {tuple(latents.shape)}"
        )
    if latents.device != expected_device:
        raise SamplingError(
            f"{name} is on {latents.device}; expected {expected_device}"
        )
    if not latents.is_floating_point() or latents.dtype != expected_dtype:
        raise SamplingError(f"{name} must use the inference dtype")
    _require_finite(latents, name)


def _copy_to_cpu(destination: torch.Tensor, source: torch.Tensor) -> None:
    destination.copy_(source.detach().to(device="cpu", dtype=destination.dtype))


def _token_tensor(encoded: Any, name: str) -> torch.Tensor:
    if isinstance(encoded, Mapping):
        value = encoded.get(name)
    else:
        value = getattr(encoded, name, None)
    if not isinstance(value, torch.Tensor):
        raise SamplingError(f"Tokenizer result has no tensor {name!r}")
    return value


def _output_tensor(output: Any, name: str, component: str) -> torch.Tensor:
    value: object | None
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    if value is None and isinstance(output, (tuple, list)) and output:
        value = output[0]
    if not isinstance(value, torch.Tensor):
        raise SamplingError(f"{component} output has no tensor {name!r}")
    return value


def _finite_noise_sigma(scheduler: Any) -> float:
    value = getattr(scheduler, "init_noise_sigma", None)
    try:
        sigma = float(value)
    except (TypeError, ValueError) as error:
        raise SamplingError("scheduler.init_noise_sigma must be numeric") from error
    if not math.isfinite(sigma) or sigma <= 0:
        raise SamplingError("scheduler.init_noise_sigma must be finite and positive")
    return sigma


def _config_value(config: Any, key: str) -> object:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise SamplingError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise SamplingError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise SamplingError(f"{name} must be a positive integer")
    return result


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise SamplingError(f"{name} contains NaN or infinity")


def _is_cuda_out_of_memory(error: RuntimeError, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    return (
        isinstance(error, out_of_memory_type) or "out of memory" in str(error).lower()
    )
