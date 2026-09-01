"""Stable Diffusion checkpoint definitions used by the experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Describe one model checkpoint and its Webster dataset view."""

    cli_name: str
    dataset_model: str
    model_id: str
    vae_id: str | None
    resolution: int

    def __post_init__(self) -> None:
        """Reject incomplete registry entries early."""

        for field_name in ("cli_name", "dataset_model", "model_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ModelSpec.{field_name} must be a non-empty string")
        if self.vae_id is not None and not self.vae_id:
            raise ValueError("ModelSpec.vae_id must be non-empty when supplied")
        if isinstance(self.resolution, bool) or self.resolution <= 0:
            raise ValueError("ModelSpec.resolution must be a positive integer")


_MODEL_SPECS = {
    "sdv1": ModelSpec(
        cli_name="sdv1",
        dataset_model="sdv1",
        model_id="CompVis/stable-diffusion-v1-4",
        vae_id=None,
        resolution=512,
    ),
    "sdv2": ModelSpec(
        cli_name="sdv2",
        dataset_model="sdv2",
        model_id="Manojb/stable-diffusion-2-1-base",
        vae_id=None,
        resolution=512,
    ),
    # These are the only two lines to change if RealisticVision is pinned to
    # a different model or VAE in a later experiment.
    "realvis": ModelSpec(
        cli_name="realvis",
        dataset_model="realisticvision",
        model_id="SG161222/Realistic_Vision_V2.0",
        vae_id="stabilityai/sd-vae-ft-mse",
        resolution=512,
    ),
}

MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(_MODEL_SPECS)


def model_names() -> tuple[str, ...]:
    """Return supported CLI model names in their documented order."""

    return tuple(MODEL_SPECS)


def get_model_spec(name: str) -> ModelSpec:
    """Return the checkpoint definition for one exact CLI model name."""

    try:
        return MODEL_SPECS[name]
    except (KeyError, TypeError) as error:
        choices = ", ".join(model_names())
        raise ValueError(f"Unknown model {name!r}; expected one of: {choices}") from error
