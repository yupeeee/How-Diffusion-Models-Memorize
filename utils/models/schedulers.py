"""Construct DDIM and DDPM samplers from a checkpoint's native settings."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


SCHEDULER_NAMES = ("ddim", "ddpm")


class SchedulerConstructionError(RuntimeError):
    """Report an unsupported scheduler name or step interface."""


@dataclass(frozen=True, slots=True)
class SchedulerBuildResult:
    """Hold a scheduler and the exact configuration used to construct it."""

    scheduler: Any
    name: str
    class_name: str
    config: dict[str, object]
    removed_config_keys: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        """Return JSON-ready scheduler metadata."""

        return {
            "scheduler_name": self.name,
            "scheduler_class": self.class_name,
            "scheduler_config": self.config,
            "removed_config_keys": list(self.removed_config_keys),
        }


def build_scheduler(
    original_scheduler: Any,
    scheduler_name: str,
    *,
    scheduler_classes: Mapping[str, Any] | None = None,
) -> SchedulerBuildResult:
    """Build the requested scheduler from the checkpoint scheduler config."""

    original_config = getattr(original_scheduler, "config", None)
    if original_config is None:
        raise SchedulerConstructionError("Checkpoint scheduler has no config")
    return build_scheduler_from_config(
        original_config,
        scheduler_name,
        scheduler_classes=scheduler_classes,
    )


def build_scheduler_from_config(
    original_config: Any,
    scheduler_name: str,
    *,
    scheduler_classes: Mapping[str, Any] | None = None,
) -> SchedulerBuildResult:
    """Build a scheduler directly from a validated checkpoint configuration."""

    if original_config is None:
        raise SchedulerConstructionError("Checkpoint scheduler has no config")
    name = _validated_scheduler_name(scheduler_name)
    classes = scheduler_classes or _diffusers_scheduler_classes()
    try:
        scheduler_type = classes[name]
    except KeyError as error:
        raise SchedulerConstructionError(
            f"No scheduler class was supplied for {name!r}"
        ) from error
    # Diffusers ConfigMixin handles compatible extra keys itself. We do not
    # rewrite betas, prediction type, timestep spacing, or clipping settings.
    scheduler = scheduler_type.from_config(original_config)
    selected_config = scheduler_config_dict(scheduler)
    original_prediction = _config_value(original_config, "prediction_type")
    selected_prediction = selected_config.get("prediction_type")
    if original_prediction is not None and selected_prediction != original_prediction:
        raise SchedulerConstructionError(
            "Scheduler construction changed the checkpoint prediction_type"
        )
    return SchedulerBuildResult(
        scheduler=scheduler,
        name=name,
        class_name=type(scheduler).__name__,
        config=selected_config,
        removed_config_keys=(),
    )


def scheduler_config_dict(scheduler: Any) -> dict[str, object]:
    """Return a JSON-ready copy of a scheduler's final configuration."""

    config = getattr(scheduler, "config", None)
    if config is None:
        raise SchedulerConstructionError("Selected scheduler has no config")
    if isinstance(config, Mapping):
        raw = dict(config)
    elif hasattr(config, "to_dict"):
        raw = dict(config.to_dict())
    else:
        try:
            raw = dict(vars(config))
        except TypeError as error:
            raise SchedulerConstructionError(
                "Scheduler config cannot be serialized"
            ) from error
    result = {str(key): _json_value(value) for key, value in raw.items()}
    defaulted_keys = result.get("_use_default_values")
    if defaulted_keys is not None:
        if not isinstance(defaulted_keys, list) or any(
            not isinstance(key, str) for key in defaulted_keys
        ):
            raise SchedulerConstructionError(
                "Scheduler config _use_default_values must be a string sequence"
            )
        # Diffusers constructs this private bookkeeping field from a set. Its
        # iteration order is not scientific and can vary between processes.
        result["_use_default_values"] = sorted(defaulted_keys)
    return result


def scheduler_step_kwargs(
    scheduler: Any,
    scheduler_name: str,
    *,
    generators: Sequence[torch.Generator] | None = None,
) -> dict[str, object]:
    """Return deterministic keyword arguments for one scheduler step."""

    name = _validated_scheduler_name(scheduler_name)
    if name == "ddim":
        _require_step_keyword(scheduler, "eta")
        return {"eta": 0.0}
    if not generators:
        raise SchedulerConstructionError(
            "DDPM requires one persistent generator for every seed"
        )
    _require_step_keyword(scheduler, "generator")
    return {"generator": list(generators)}


def _validated_scheduler_name(name: str) -> str:
    if name not in SCHEDULER_NAMES:
        choices = ", ".join(SCHEDULER_NAMES)
        raise ValueError(f"Unknown scheduler {name!r}; expected one of: {choices}")
    return name


def _diffusers_scheduler_classes() -> dict[str, Any]:
    try:
        from diffusers import DDIMScheduler, DDPMScheduler
    except ImportError as error:  # pragma: no cover - dependency message
        raise SchedulerConstructionError(
            "diffusers is required to construct DDIM and DDPM schedulers"
        ) from error
    return {"ddim": DDIMScheduler, "ddpm": DDPMScheduler}


def _config_value(config: Any, key: str) -> object:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _require_step_keyword(scheduler: Any, keyword: str) -> None:
    try:
        parameters = inspect.signature(scheduler.step).parameters.values()
    except (TypeError, ValueError) as error:
        raise SchedulerConstructionError(
            f"Cannot inspect {type(scheduler).__name__}.step"
        ) from error
    accepts_keyword = any(
        parameter.name == keyword
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if not accepts_keyword:
        raise SchedulerConstructionError(
            f"{type(scheduler).__name__}.step does not accept {keyword!r}"
        )


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            return str(value)
    return str(value)
