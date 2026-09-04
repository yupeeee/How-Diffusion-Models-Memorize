"""Canonical runtime-device comparisons with explicit CUDA alias handling."""

from __future__ import annotations

import operator
import os
from collections.abc import Sequence
from typing import Callable, TypeVar

import torch

CurrentCudaIndex = Callable[[], int]
_Item = TypeVar("_Item")


class DeviceNormalizationError(ValueError):
    """Report an invalid device or active CUDA index."""


class DeviceSelectionError(ValueError):
    """Report a requested execution device that is unavailable or unsupported."""


def resolve_devices(
    requested: str | torch.device = "auto",
) -> tuple[torch.device, ...]:
    """Resolve one device request to concrete execution devices.

    ``auto`` selects every CUDA device visible to PyTorch in logical-index
    order. With no visible CUDA device it selects MPS when available and CPU
    otherwise. Every explicit request selects exactly one device.
    """

    if isinstance(requested, str) and requested.strip().casefold() == "auto":
        if _cuda_is_available():
            count = _cuda_device_count()
            if count > 0:
                return tuple(torch.device("cuda", index) for index in range(count))
        if _mps_is_available():
            return (torch.device("mps"),)
        return (torch.device("cpu"),)

    parsed = _parse_explicit_device(requested)
    if parsed.type == "cpu":
        if parsed.index is not None:
            raise DeviceSelectionError("CPU device must be specified as 'cpu'")
        return (torch.device("cpu"),)
    if parsed.type == "mps":
        if parsed.index is not None:
            raise DeviceSelectionError("MPS device must be specified as 'mps'")
        if not _mps_is_available():
            raise DeviceSelectionError("requested MPS device is unavailable")
        return (torch.device("mps"),)
    if parsed.type != "cuda":
        raise DeviceSelectionError("device must be auto, cpu, mps, cuda, or cuda:N")
    if not _cuda_is_available():
        raise DeviceSelectionError("requested CUDA device is unavailable")
    count = _cuda_device_count()
    if count < 1:
        raise DeviceSelectionError("requested CUDA device is unavailable")
    try:
        concrete = normalize_device_alias(parsed)
    except DeviceNormalizationError as error:
        raise DeviceSelectionError(str(error)) from error
    assert concrete.index is not None
    if concrete.index >= count:
        raise DeviceSelectionError(
            f"requested CUDA device index {concrete.index} is unavailable; "
            f"PyTorch exposes {count} CUDA device(s)"
        )
    return (concrete,)


def round_robin_shard(
    values: Sequence[_Item],
    *,
    worker_index: int,
    worker_count: int,
) -> tuple[_Item, ...]:
    """Return one stable, disjoint round-robin shard of ``values``."""

    count = _positive_index(worker_count, "worker count")
    index = _nonnegative_index(worker_index, "worker index")
    if index >= count:
        raise ValueError("worker index must be smaller than worker count")
    return tuple(values[index::count])


def worker_count_for_tasks(
    devices: Sequence[torch.device],
    task_count: int,
) -> int:
    """Use no more workers than there are concrete devices or tasks."""

    if not devices:
        raise ValueError("at least one execution device is required")
    count = _nonnegative_index(task_count, "task count")
    return min(len(devices), count)


def configure_worker_cpu_threads(worker_count: int) -> int:
    """Keep concurrent GPU workers within the process CPU allocation.

    Every spawned process otherwise inherits PyTorch's full intra-op thread
    pool. Dividing the available CPUs prevents N GPU workers from each
    competing for all of them. An existing tighter user/runtime limit is
    preserved.
    """

    workers = _positive_index(worker_count, "worker count")
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        available = os.cpu_count()
    else:
        available = len(affinity)
    if available is None:
        available = 1
    available_cpus = _positive_index(available, "available CPU count")
    current = _positive_index(torch.get_num_threads(), "PyTorch intra-op thread count")
    selected = min(current, max(1, available_cpus // workers))
    if selected != current:
        torch.set_num_threads(selected)
    return selected


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


def _parse_explicit_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str):
        value = device.strip().casefold()
        if not value:
            raise DeviceSelectionError("device must not be empty")
    elif isinstance(device, torch.device):
        value = device
    else:
        raise DeviceSelectionError("device must be auto, cpu, mps, cuda, or cuda:N")
    try:
        return torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise DeviceSelectionError(f"invalid device: {device!r}") from error


def _cuda_is_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception as error:
        raise DeviceSelectionError("cannot query CUDA availability") from error


def _cuda_device_count() -> int:
    try:
        count = torch.cuda.device_count()
    except Exception as error:
        raise DeviceSelectionError("cannot enumerate visible CUDA devices") from error
    return _nonnegative_index(count, "CUDA device count")


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    available = getattr(backend, "is_available", None)
    if not callable(available):
        return False
    try:
        return bool(available())
    except Exception as error:
        raise DeviceSelectionError("cannot query MPS availability") from error


def _positive_index(value: object, label: str) -> int:
    result = _nonnegative_index(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_index(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result
