"""Opt-in checkpoint loader for theory probes; generation loading is unchanged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .loading import (
    ModelLoadingError,
    RuntimeSelection,
    _move_for_inference,
    _validated_revision,
    configure_reproducibility,
    offline_mode_enabled,
    package_version_metadata,
)


@dataclass(frozen=True)
class LoadedDenoiserComponents:
    tokenizer: Any
    text_encoder: Any
    unet: Any
    model_id: str
    model_revision: str
    device: torch.device
    inference_dtype: torch.dtype
    device_metadata: dict
    package_versions: dict


def load_denoiser_components(*, model_id, model_revision, runtime, component_classes=None):
    """Load only tokenizer/text encoder/UNet from the preserved immutable revision.

    No VAE, image decoder, safety checker, scheduler update, or new revision
    resolution is involved. Tests may supply component classes without a Hub.
    """
    if not isinstance(runtime, RuntimeSelection):
        raise ModelLoadingError("Probe runtime must be an explicit RuntimeSelection")
    revision = _validated_revision(model_id, model_revision)
    if not isinstance(model_id, str) or not model_id:
        raise ModelLoadingError("Probe checkpoint model_id must be nonempty")
    if component_classes is None:
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        component_classes = {
            "tokenizer": CLIPTokenizer,
            "text_encoder": CLIPTextModel,
            "unet": UNet2DConditionModel,
        }
    configure_reproducibility()
    common = {"revision": revision, "local_files_only": offline_mode_enabled()}
    components = {}
    for name in ("tokenizer", "text_encoder", "unet"):
        kwargs = common | {"subfolder": name}
        if name != "tokenizer":
            kwargs["torch_dtype"] = runtime.inference_dtype
        component = component_classes[name].from_pretrained(model_id, **kwargs)
        if name != "tokenizer":
            _move_for_inference(component, runtime, runtime.inference_dtype)
            component.requires_grad_(False)
            parameters = [p for p in component.parameters() if p.is_floating_point()]
            if not parameters or any(p.dtype != runtime.inference_dtype or p.device != runtime.device for p in parameters):
                raise ModelLoadingError(f"Loaded {name} parameters differ from preserved inference device/dtype")
            if component.training:
                raise ModelLoadingError(f"Loaded {name} unexpectedly remains in training mode")
        components[name] = component
    return LoadedDenoiserComponents(
        **components,
        model_id=model_id,
        model_revision=revision,
        device=runtime.device,
        inference_dtype=runtime.inference_dtype,
        device_metadata=runtime.metadata() | {
            "parameter_dtypes": {name: sorted({str(p.dtype).removeprefix("torch.") for p in components[name].parameters() if p.is_floating_point()}) for name in ("text_encoder", "unet")},
            "component_identities": {name: {"repository": model_id, "revision": revision, "subfolder": name} for name in components},
            "gradient_state": "all_parameters_frozen_eval_mode",
        },
        package_versions=package_version_metadata(),
    )
