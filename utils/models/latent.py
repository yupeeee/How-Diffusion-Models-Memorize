"""Build deterministic target latents and compute Definition-1 distances."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, ImageOps

TARGET_LATENT_DEFINITION = "deterministic_vae_posterior_mode_scaled"


class LatentSpaceError(ValueError):
    """Report invalid target preprocessing or incompatible latent shapes."""


@dataclass(frozen=True, slots=True)
class LatentDistances:
    """Hold the raw Definition-1 distance and two diagnostic quantities."""

    l2_norms: torch.Tensor
    l2_squared: torch.Tensor
    latent_rmse: torch.Tensor


def target_preprocessing_policy(resolution: int = 512) -> dict[str, object]:
    """Return the complete JSON-ready empirical-target preprocessing policy."""

    size = _positive_integer(resolution, "resolution")
    return {
        "exif_transpose": True,
        "color_mode": "RGB",
        "resize": {
            "width": size,
            "height": size,
            "resampling": "Pillow Lanczos",
            "aspect_policy": "direct resize without crop",
        },
        "random_crop": False,
        "augmentation": False,
        "pixel_range": [-1.0, 1.0],
        "vae_encoding_dtype": "float32 when supported",
        "vae_posterior": "mode",
        "latent_scaling": "multiply by vae.config.scaling_factor",
        "saved_target_dtype": "float32",
    }


def preprocess_target_image(
    image: Image.Image,
    *,
    resolution: int = 512,
) -> torch.Tensor:
    """Apply the fixed image policy and return a float32 ``[1, 3, H, W]`` tensor."""

    if not isinstance(image, Image.Image):
        raise LatentSpaceError("target image must be a Pillow Image")
    size = _positive_integer(resolution, "resolution")
    transposed = ImageOps.exif_transpose(image)
    rgb_image = transposed.convert("RGB")
    resized = rgb_image.resize((size, size), resample=Image.Resampling.LANCZOS)
    pixels = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    pixels = pixels.reshape(size, size, 3).permute(2, 0, 1).contiguous()
    return pixels.to(dtype=torch.float32).div(127.5).sub(1.0).unsqueeze(0)


def encode_target_latent(
    image: Image.Image,
    vae: Any,
    device: str | torch.device,
    *,
    resolution: int = 512,
    expected_channels: int | None = None,
    encoding_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode an image as a scaled deterministic VAE posterior mode."""

    if not isinstance(encoding_dtype, torch.dtype):
        raise LatentSpaceError("encoding_dtype must be a torch dtype")
    selected_device = torch.device(device)
    image_tensor = preprocess_target_image(image, resolution=resolution)
    if hasattr(vae, "to"):
        vae.to(device=selected_device, dtype=encoding_dtype)
    if hasattr(vae, "eval"):
        vae.eval()
    image_tensor = image_tensor.to(device=selected_device, dtype=encoding_dtype)

    scaling_factor = _vae_scaling_factor(vae)
    with torch.inference_mode():
        encoded = vae.encode(image_tensor)
        posterior = getattr(encoded, "latent_dist", None)
        if posterior is None or not callable(getattr(posterior, "mode", None)):
            raise LatentSpaceError("VAE encode result has no posterior mode")
        target = posterior.mode()

    if not isinstance(target, torch.Tensor) or target.ndim != 4 or target.shape[0] != 1:
        shape = getattr(target, "shape", None)
        raise LatentSpaceError(
            f"VAE posterior mode must have shape [1, C, H, W], got {shape}"
        )
    target = target.mul(scaling_factor).squeeze(0).float()
    if expected_channels is not None:
        channels = _positive_integer(expected_channels, "expected_channels")
        if target.shape[0] != channels:
            raise LatentSpaceError(
                f"Target has {target.shape[0]} channels but UNet expects {channels}"
            )
    _require_finite(target, "target latent")
    return target.detach().cpu().contiguous()


def decode_generated_latents(
    generated_x0: torch.Tensor,
    vae: Any,
    device: str | torch.device,
    *,
    microbatch_size: int = 4,
    progress_callback: Callable[[int], None] | None = None,
) -> list[Image.Image]:
    """Decode scaled terminal latents into deterministic RGB Pillow images.

    Stable Diffusion stores denoising latents after multiplication by the VAE
    scaling factor. Decoding therefore divides by that factor. The VAE and its
    inputs are deliberately kept in float32, matching target encoding and
    avoiding CPU float16 kernels and half-precision decode artifacts. The
    optional progress callback receives each successfully decoded batch size.
    """

    if not isinstance(generated_x0, torch.Tensor) or generated_x0.ndim != 4:
        raise LatentSpaceError("generated endpoints must have shape [N, C, H, W]")
    if generated_x0.shape[0] < 1:
        raise LatentSpaceError("at least one generated endpoint is required")
    if not generated_x0.is_floating_point():
        raise LatentSpaceError("generated endpoints must be floating point")
    _require_finite(generated_x0, "generated endpoints")
    batch_size = _positive_integer(microbatch_size, "microbatch_size")
    selected_device = torch.device(device)
    scaling_factor = _vae_scaling_factor(vae)
    if scaling_factor <= 0:
        raise LatentSpaceError("VAE scaling_factor must be positive")
    if hasattr(vae, "to"):
        vae.to(device=selected_device, dtype=torch.float32)
    if hasattr(vae, "eval"):
        vae.eval()

    images: list[Image.Image] = []
    position = 0
    batch_size = min(batch_size, generated_x0.shape[0])
    with torch.inference_mode():
        while position < generated_x0.shape[0]:
            current_size = min(batch_size, generated_x0.shape[0] - position)
            endpoints = generated_x0[position : position + current_size]
            retry_after_oom = False
            try:
                decoded_images = _decode_latent_microbatch(
                    endpoints,
                    vae,
                    selected_device,
                    scaling_factor,
                )
            except RuntimeError as error:
                if (
                    not _is_cuda_out_of_memory(error, selected_device)
                    or current_size == 1
                ):
                    raise
                batch_size = max(1, current_size // 2)
                retry_after_oom = True
            if retry_after_oom:
                # Leave the exception handler first so its traceback releases
                # failed-forward intermediates before emptying CUDA's cache.
                torch.cuda.empty_cache()
                continue
            images.extend(decoded_images)
            position += current_size
            if progress_callback is not None:
                progress_callback(current_size)
    return images


def compute_latent_distances(
    generated_x0: torch.Tensor,
    target_x_star: torch.Tensor,
) -> LatentDistances:
    """Compute raw terminal L2 norms and diagnostic squared norms and RMSE."""

    if not isinstance(generated_x0, torch.Tensor) or not isinstance(
        target_x_star, torch.Tensor
    ):
        raise LatentSpaceError("generated and target latents must be torch tensors")
    if target_x_star.ndim != 3 or generated_x0.ndim != 4:
        raise LatentSpaceError(
            "Expected target [C, H, W] and generated endpoints [N, C, H, W]"
        )
    if generated_x0.shape[0] < 1:
        raise LatentSpaceError("At least one generated endpoint is required")
    if tuple(generated_x0.shape[1:]) != tuple(target_x_star.shape):
        raise LatentSpaceError(
            "Target and generated latent channel/spatial dimensions do not match"
        )
    _require_finite(generated_x0, "generated endpoints")
    _require_finite(target_x_star, "target latent")

    target = target_x_star.to(device=generated_x0.device, dtype=torch.float32)
    difference = generated_x0.float() - target.unsqueeze(0)
    l2_norms = torch.linalg.vector_norm(difference.flatten(start_dim=1), ord=2, dim=1)
    l2_squared = l2_norms.square()
    latent_rmse = l2_norms / math.sqrt(target_x_star.numel())
    _require_finite(l2_norms, "L2 norms")
    return LatentDistances(
        l2_norms=l2_norms.detach().cpu().to(torch.float32).contiguous(),
        l2_squared=l2_squared.detach().cpu().to(torch.float32).contiguous(),
        latent_rmse=latent_rmse.detach().cpu().to(torch.float32).contiguous(),
    )


def raw_l2_norms(
    generated_x0: torch.Tensor,
    target_x_star: torch.Tensor,
) -> torch.Tensor:
    """Return only the raw, unnormalized Definition-1 L2 norm vector."""

    return compute_latent_distances(generated_x0, target_x_star).l2_norms


def _vae_scaling_factor(vae: Any) -> float:
    config = getattr(vae, "config", None)
    if config is None:
        raise LatentSpaceError("VAE config is missing scaling_factor")
    if isinstance(config, Mapping):
        value = config.get("scaling_factor")
    else:
        value = getattr(config, "scaling_factor", None)
    try:
        scaling_factor = float(value)
    except (TypeError, ValueError) as error:
        raise LatentSpaceError("VAE scaling_factor must be numeric") from error
    if not math.isfinite(scaling_factor):
        raise LatentSpaceError("VAE scaling_factor must be finite")
    return scaling_factor


def _vae_decoded_sample(decoded: object) -> torch.Tensor:
    """Extract a sample tensor from supported diffusers decode return forms."""

    sample = getattr(decoded, "sample", None)
    if sample is None and isinstance(decoded, Mapping):
        sample = decoded.get("sample")
    if sample is None and isinstance(decoded, (list, tuple)) and decoded:
        sample = decoded[0]
    if not isinstance(sample, torch.Tensor):
        raise LatentSpaceError("VAE decode result has no sample tensor")
    return sample


def _decode_latent_microbatch(
    endpoints: torch.Tensor,
    vae: Any,
    device: torch.device,
    scaling_factor: float,
) -> list[Image.Image]:
    """Decode one complete microbatch without publishing partial images."""

    scaled = endpoints.to(device=device, dtype=torch.float32)
    decoded = _vae_decoded_sample(vae.decode(scaled / scaling_factor))
    if decoded.ndim != 4 or decoded.shape[0] != scaled.shape[0]:
        raise LatentSpaceError("VAE decoded sample must have shape [N, 3, H, W]")
    if decoded.shape[1] != 3:
        raise LatentSpaceError(
            f"VAE decoded sample has {decoded.shape[1]} channels, expected RGB"
        )
    _require_finite(decoded, "decoded images")
    pixels = (
        decoded.float()
        .div(2.0)
        .add(0.5)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8, device="cpu")
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    return [
        Image.fromarray(pixels[index].numpy(), mode="RGB")
        for index in range(len(pixels))
    ]


def _is_cuda_out_of_memory(error: RuntimeError, device: torch.device) -> bool:
    """Return whether an error is specifically CUDA memory exhaustion."""

    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    if isinstance(error, out_of_memory_type):
        return True
    return device.type == "cuda" and "out of memory" in str(error).lower()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise LatentSpaceError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise LatentSpaceError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise LatentSpaceError(f"{name} must be a positive integer")
    return result


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise LatentSpaceError(f"{name} contains NaN or infinity")
