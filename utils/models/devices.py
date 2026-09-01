"""Canonical runtime-device comparisons with explicit CUDA alias handling."""

from __future__ import annotations

import operator
from typing import Callable

import torch

CurrentCudaIndex = Callable[[], int]


class DeviceNormalizationError(ValueError):
    """Report an invalid device or active CUDA index."""


def normalize_device_alias(
    device: str | torch.device,
    *,
    active_cuda_index: int | None = None,
    current_cuda_index: CurrentCudaIndex | None = None,
) -> torch.device:
    """Resolve bare ``cuda`` to the active CUDA index; preserve other devices."""

    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise DeviceNormalizationError(f"invalid torch device: {device!r}") from error
    if parsed.type != "cuda" or parsed.index is not None:
        return parsed
    index = _active_cuda_index(active_cuda_index, current_cuda_index)
    return torch.device("cuda", index)


def devices_match(
    left: str | torch.device,
    right: str | torch.device,
    *,
    active_cuda_index: int | None = None,
    current_cuda_index: CurrentCudaIndex | None = None,
) -> bool:
    """Return whether two devices identify the same concrete execution target."""

    normalized_left = normalize_device_alias(
        left,
        active_cuda_index=active_cuda_index,
        current_cuda_index=current_cuda_index,
    )
    normalized_right = normalize_device_alias(
        right,
        active_cuda_index=active_cuda_index,
        current_cuda_index=current_cuda_index,
    )
    return normalized_left == normalized_right


def _active_cuda_index(
    active_cuda_index: int | None,
    current_cuda_index: CurrentCudaIndex | None,
) -> int:
    value: object
    if active_cuda_index is None:
        resolver = current_cuda_index or torch.cuda.current_device
        try:
            value = resolver()
        except Exception as error:
            raise DeviceNormalizationError(
                "cannot resolve the active CUDA index for bare 'cuda'"
            ) from error
    else:
        value = active_cuda_index
    if isinstance(value, bool):
        raise DeviceNormalizationError("active CUDA index must be non-negative")
    try:
        index = operator.index(value)
    except TypeError as error:
        raise DeviceNormalizationError(
            "active CUDA index must be non-negative"
        ) from error
    if index < 0:
        raise DeviceNormalizationError("active CUDA index must be non-negative")
    return index
