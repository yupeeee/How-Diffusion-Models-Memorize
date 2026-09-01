"""Load immutable model components and select reproducible runtime settings."""

from __future__ import annotations

import importlib.metadata
import math
import operator
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .registry import ModelSpec
from .devices import DeviceNormalizationError, devices_match

RevisionResolver = Callable[[str], str]
_TRUE_ENVIRONMENT_VALUES = frozenset({"1", "on", "true", "yes"})


class ModelLoadingError(RuntimeError):
    """Report an invalid revision, runtime, or checkpoint component."""


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """Hold the one device and inference dtype selected for a run."""

    device: torch.device
    inference_dtype: torch.dtype
    gpu_name: str | None

    def metadata(self) -> dict[str, str | None]:
        """Return JSON-ready runtime metadata."""

        return {
            "device": str(self.device),
            "device_type": self.device.type,
            "gpu_name": self.gpu_name,
            "inference_dtype": dtype_name(self.inference_dtype),
        }


@dataclass(frozen=True, slots=True)
class LoadedVAE:
    """Hold one pinned float32 VAE without other diffusion components."""

    vae: Any
    device: torch.device
    inference_dtype: torch.dtype
    model_id: str
    model_revision: str
    vae_id: str
    vae_revision: str
    scaling_factor: float
    device_metadata: dict[str, str | None]
    package_versions: dict[str, str | None]


@dataclass(frozen=True, slots=True)
class LoadedModelComponents:
    """Hold the components needed by latent-space inference."""

    spec: ModelSpec
    tokenizer: Any
    text_encoder: Any
    unet: Any
    vae: Any
    original_scheduler: Any
    device: torch.device
    inference_dtype: torch.dtype
    model_revision: str
    vae_id: str
    vae_revision: str
    device_metadata: dict[str, str | None]
    package_versions: dict[str, str | None]

    @property
    def scheduler(self) -> Any:
        """Return the checkpoint's native scheduler."""

        return self.original_scheduler

    @property
    def model_id(self) -> str:
        """Return the immutable model repository identifier."""

        return self.spec.model_id


def dtype_name(dtype: torch.dtype) -> str:
    """Return a stable dtype name such as ``float16``."""

    text = str(dtype)
    return text.removeprefix("torch.")


def select_runtime(*, warn_without_cuda: bool = True) -> RuntimeSelection:
    """Select CUDA, MPS, or CPU and the experiment's shared inference dtype."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        return RuntimeSelection(device, torch.float16, gpu_name)

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        runtime = RuntimeSelection(
            torch.device("mps"),
            torch.float16,
            "Apple Metal Performance Shaders (MPS)",
        )
    else:
        runtime = RuntimeSelection(torch.device("cpu"), torch.float32, None)

    if warn_without_cuda:
        print(
            "WARNING: CUDA is unavailable. The full proximity experiment will "
            "be very slow on this device.",
            file=sys.stderr,
        )
    return runtime


def configure_reproducibility() -> None:
    """Disable nondeterministic acceleration used by neither sampler."""

    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None:
        matmul = getattr(cuda_backend, "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = False
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None:
        if hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = False
        if hasattr(cudnn_backend, "benchmark"):
            cudnn_backend.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - for older supported Torch releases
        torch.use_deterministic_algorithms(True)


def package_version_metadata() -> dict[str, str | None]:
    """Return package versions needed to describe a reproducible run."""

    return {
        "torch": str(torch.__version__),
        "diffusers": _installed_version("diffusers"),
        "transformers": _installed_version("transformers"),
        "cuda": torch.version.cuda,
    }


def _installed_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def offline_mode_enabled() -> bool:
    """Return whether either supported Hugging Face stack is offline."""

    return any(
        os.environ.get(name, "").strip().casefold() in _TRUE_ENVIRONMENT_VALUES
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def resolve_hf_revision(model_id: str, *, api: Any | None = None) -> str:
    """Resolve a repository's current branch to an immutable commit SHA."""

    if offline_mode_enabled():
        return _resolve_cached_hf_revision(model_id)
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:  # pragma: no cover - dependency message
            raise ModelLoadingError(
                "huggingface_hub is required to resolve checkpoint revisions"
            ) from error
        api = HfApi()
    info = api.model_info(repo_id=model_id, revision="main")
    return _validated_revision(model_id, getattr(info, "sha", None))


def _resolve_cached_hf_revision(model_id: str) -> str:
    """Resolve ``main`` through the Hub cache without any network fallback."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency message
        raise ModelLoadingError(
            "huggingface_hub is required to resolve cached checkpoint revisions"
        ) from error
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision="main",
                local_files_only=True,
            )
        )
    except Exception as error:
        raise ModelLoadingError(
            f"Offline mode requires a cached snapshot for {model_id}; "
            "no network request was attempted"
        ) from error
    if snapshot.parent.name != "snapshots" or not snapshot.is_dir():
        raise ModelLoadingError(
            f"{model_id} resolved to an invalid local snapshot path: {snapshot}"
        )
    return _validated_revision(model_id, snapshot.name)


def _validated_revision(model_id: str, revision: object) -> str:
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is None
    ):
        raise ModelLoadingError(
            f"{model_id} did not resolve to an immutable 40-64 digit commit SHA"
        )
    return revision.lower()


def resolve_model_revisions(
    spec: ModelSpec,
    *,
    resolver: RevisionResolver | None = None,
) -> tuple[str, str]:
    """Resolve the model and any external VAE repository independently."""

    active_resolver = resolver or resolve_hf_revision
    model_revision = _validated_revision(spec.model_id, active_resolver(spec.model_id))
    if spec.vae_id is None:
        return model_revision, model_revision
    vae_revision = _validated_revision(spec.vae_id, active_resolver(spec.vae_id))
    return model_revision, vae_revision


def load_vae_from_generation_config(
    generation_config: Mapping[str, object],
    *,
    runtime: RuntimeSelection | None = None,
) -> LoadedVAE:
    """Load only the immutable VAE pinned by a generation run."""

    science = _generation_science_config(generation_config)
    model_id = _required_repository_id(science, "model_id")
    vae_id = _required_repository_id(science, "vae_id")
    model_revision = _validated_revision(model_id, science.get("model_revision"))
    vae_revision = _validated_revision(vae_id, science.get("vae_revision"))
    selected_runtime = runtime or select_runtime(warn_without_cuda=False)
    try:
        device = torch.device(selected_runtime.device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ModelLoadingError("VAE runtime device is invalid") from error

    autoencoder_type = _autoencoder_kl_class()
    factory = getattr(autoencoder_type, "from_pretrained", None)
    if not callable(factory):
        raise ModelLoadingError("AutoencoderKL has no from_pretrained interface")
    repository_id = model_id if vae_id == model_id else vae_id
    arguments: dict[str, object] = {
        "revision": vae_revision,
        "torch_dtype": torch.float32,
    }
    if vae_id == model_id:
        arguments["subfolder"] = "vae"
    if offline_mode_enabled():
        arguments["local_files_only"] = True
    try:
        vae = factory(repository_id, **arguments)
    except Exception as error:
        raise ModelLoadingError(
            f"Cannot load pinned VAE {repository_id}@{vae_revision}: {error}"
        ) from error

    scaling_factor = _loaded_vae_scaling_factor(vae)
    move = _required_module_method(vae, "to")
    evaluate = _required_module_method(vae, "eval")
    disable_gradients = _required_module_method(vae, "requires_grad_")
    try:
        move(device=device, dtype=torch.float32)
        evaluate()
        disable_gradients(False)
    except Exception as error:
        raise ModelLoadingError(
            f"Cannot prepare pinned VAE for float32 inference on {device}: {error}"
        ) from error

    device_metadata = dict(selected_runtime.metadata())
    device_metadata["inference_dtype"] = dtype_name(torch.float32)
    return LoadedVAE(
        vae=vae,
        device=device,
        inference_dtype=torch.float32,
        model_id=model_id,
        model_revision=model_revision,
        vae_id=vae_id,
        vae_revision=vae_revision,
        scaling_factor=scaling_factor,
        device_metadata=device_metadata,
        package_versions=package_version_metadata(),
    )


def _generation_science_config(
    generation_config: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(generation_config, Mapping):
        raise ModelLoadingError("generation configuration is not a mapping")
    science = generation_config.get("scientific_config")
    if not isinstance(science, Mapping):
        raise ModelLoadingError(
            "generation configuration scientific_config is not a mapping"
        )
    return science


def _required_repository_id(
    science: Mapping[str, object],
    key: str,
) -> str:
    value = science.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ModelLoadingError(f"generation scientific_config {key} is invalid")
    return value


def _autoencoder_kl_class() -> Any:
    try:
        from diffusers import AutoencoderKL
    except ImportError as error:  # pragma: no cover - dependency message
        raise ModelLoadingError(
            "diffusers is required to load the pinned VAE"
        ) from error
    return AutoencoderKL


def _required_module_method(module: Any, name: str) -> Callable[..., Any]:
    method = getattr(module, name, None)
    if not callable(method):
        raise ModelLoadingError(f"loaded VAE has no callable {name} method")
    return method


def _loaded_vae_scaling_factor(vae: Any) -> float:
    config = getattr(vae, "config", None)
    if isinstance(config, Mapping):
        value = config.get("scaling_factor")
    else:
        value = getattr(config, "scaling_factor", None)
    if isinstance(value, bool):
        raise ModelLoadingError("loaded VAE scaling_factor must be numeric")
    try:
        scaling_factor = float(value)
    except (TypeError, ValueError) as error:
        raise ModelLoadingError("loaded VAE scaling_factor must be numeric") from error
    if not math.isfinite(scaling_factor) or scaling_factor <= 0:
        raise ModelLoadingError("loaded VAE scaling_factor must be finite and positive")
    return scaling_factor


def load_model_components(
    spec: ModelSpec,
    *,
    runtime: RuntimeSelection | None = None,
    revision_resolver: RevisionResolver | None = None,
    pipeline_class: Any | None = None,
    vae_class: Any | None = None,
) -> LoadedModelComponents:
    """Resolve immutable revisions and load the components for one model."""

    configure_reproducibility()
    selected_runtime = runtime or select_runtime()
    model_revision, vae_revision = resolve_model_revisions(
        spec, resolver=revision_resolver
    )
    pipeline_type, autoencoder_type = _diffusers_classes(pipeline_class, vae_class)
    local_files_only = offline_mode_enabled()

    pipeline_arguments: dict[str, object] = {
        "revision": model_revision,
        "torch_dtype": selected_runtime.inference_dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if local_files_only:
        pipeline_arguments["local_files_only"] = True
    if spec.vae_id is not None:
        vae_arguments: dict[str, object] = {
            "revision": vae_revision,
            "torch_dtype": torch.float32,
        }
        if local_files_only:
            vae_arguments["local_files_only"] = True
        pipeline_arguments["vae"] = autoencoder_type.from_pretrained(
            spec.vae_id, **vae_arguments
        )
    pipeline = pipeline_type.from_pretrained(spec.model_id, **pipeline_arguments)
    pipeline.to(selected_runtime.device)

    tokenizer = _required_component(pipeline, "tokenizer")
    text_encoder = _required_component(pipeline, "text_encoder")
    unet = _required_component(pipeline, "unet")
    vae = _required_component(pipeline, "vae")
    scheduler = _required_component(pipeline, "scheduler")
    _move_for_inference(
        text_encoder, selected_runtime, selected_runtime.inference_dtype
    )
    _move_for_inference(unet, selected_runtime, selected_runtime.inference_dtype)
    # Target encoding uses a deterministic posterior mode in float32.
    _move_for_inference(vae, selected_runtime, torch.float32)

    return LoadedModelComponents(
        spec=spec,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        original_scheduler=scheduler,
        device=selected_runtime.device,
        inference_dtype=selected_runtime.inference_dtype,
        model_revision=model_revision,
        vae_id=spec.vae_id or spec.model_id,
        vae_revision=vae_revision,
        device_metadata=selected_runtime.metadata(),
        package_versions=package_version_metadata(),
    )


def preflight_model_components(
    components: LoadedModelComponents,
    *,
    scheduler: Any,
    num_inference_steps: int,
    active_cuda_index: int | None = None,
) -> None:
    """Validate loaded execution components once before record processing."""

    steps = _positive_inference_steps(num_inference_steps)
    try:
        intended_device = torch.device(components.device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ModelLoadingError(
            "Loaded components have an invalid runtime device"
        ) from error
    inference_dtype = components.inference_dtype
    if (
        not isinstance(inference_dtype, torch.dtype)
        or not torch.empty((), dtype=inference_dtype).is_floating_point()
    ):
        raise ModelLoadingError("Loaded components have an invalid inference dtype")

    text_device, _ = _module_device_and_parameters(
        components.text_encoder, "text encoder"
    )
    unet_device, unet_parameters = _module_device_and_parameters(
        components.unet, "UNet"
    )
    vae_device, vae_parameters = _module_device_and_parameters(components.vae, "VAE")
    _require_module_device(
        "text encoder", text_device, intended_device, active_cuda_index
    )
    _require_module_device("UNet", unet_device, intended_device, active_cuda_index)
    _require_module_device("VAE", vae_device, intended_device, active_cuda_index)
    _require_parameter_dtype("UNet", unet_parameters, inference_dtype)
    _require_parameter_dtype("VAE", vae_parameters, torch.float32)

    embeddings = _preflight_text_embeddings(
        components.tokenizer,
        components.text_encoder,
        device=text_device,
    )
    if embeddings.device != unet_device:
        raise ModelLoadingError(
            "Text embeddings are on "
            f"{embeddings.device}; UNet parameters are on {unet_device}"
        )
    if embeddings.dtype != inference_dtype:
        raise ModelLoadingError(
            "Text embeddings use "
            f"{dtype_name(embeddings.dtype)}; expected {dtype_name(inference_dtype)}"
        )
    _preflight_scheduler(scheduler, steps, unet_device)


def _module_device_and_parameters(
    module: Any,
    label: str,
) -> tuple[torch.device, tuple[torch.Tensor, ...]]:
    parameters = _module_tensors(module, "parameters", label)
    buffers = _module_tensors(module, "buffers", label)
    state = (*parameters, *buffers)
    if not state:
        raise ModelLoadingError(
            f"{label} has no parameters or buffers; placement cannot be verified"
        )
    devices = {tensor.device for tensor in state}
    if len(devices) != 1:
        rendered = ", ".join(sorted(str(device) for device in devices))
        raise ModelLoadingError(
            f"{label} parameters or buffers span unsupported devices: {rendered}"
        )
    return next(iter(devices)), parameters


def _module_tensors(
    module: Any,
    method_name: str,
    label: str,
) -> tuple[torch.Tensor, ...]:
    method = getattr(module, method_name, None)
    if method is None:
        return ()
    if not callable(method):
        raise ModelLoadingError(f"{label}.{method_name} is not callable")
    try:
        values = tuple(method(recurse=True))
    except TypeError:
        try:
            values = tuple(method())
        except (TypeError, RuntimeError) as error:
            raise ModelLoadingError(f"Cannot inspect {label} {method_name}") from error
    except RuntimeError as error:
        raise ModelLoadingError(f"Cannot inspect {label} {method_name}") from error
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ModelLoadingError(f"{label}.{method_name} returned a non-tensor")
    return values


def _require_module_device(
    label: str,
    actual: torch.device,
    intended: torch.device,
    active_cuda_index: int | None,
) -> None:
    try:
        matches = devices_match(
            actual,
            intended,
            active_cuda_index=active_cuda_index,
        )
    except DeviceNormalizationError as error:
        raise ModelLoadingError(
            f"Cannot resolve the intended device for {label}: {error}"
        ) from error
    if not matches:
        raise ModelLoadingError(
            f"{label} is on {actual}; intended execution device is {intended}"
        )


def _require_parameter_dtype(
    label: str,
    parameters: tuple[torch.Tensor, ...],
    expected_dtype: torch.dtype,
) -> None:
    floating = tuple(
        parameter for parameter in parameters if parameter.is_floating_point()
    )
    if not floating:
        raise ModelLoadingError(
            f"{label} has no floating-point parameters; dtype cannot be verified"
        )
    wrong = sorted(
        {
            dtype_name(parameter.dtype)
            for parameter in floating
            if parameter.dtype != expected_dtype
        }
    )
    if wrong:
        raise ModelLoadingError(
            f"{label} floating-point parameters use {', '.join(wrong)}; "
            f"expected {dtype_name(expected_dtype)}"
        )


def _preflight_text_embeddings(
    tokenizer: Any,
    text_encoder: Any,
    *,
    device: torch.device,
) -> torch.Tensor:
    maximum_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(maximum_length, bool):
        raise ModelLoadingError("tokenizer.model_max_length must be a positive integer")
    try:
        maximum_length = operator.index(maximum_length)
    except TypeError as error:
        raise ModelLoadingError(
            "tokenizer.model_max_length must be a positive integer"
        ) from error
    if maximum_length <= 0:
        raise ModelLoadingError("tokenizer.model_max_length must be a positive integer")
    try:
        encoded = tokenizer(
            ["", ""],
            padding="max_length",
            max_length=maximum_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = _encoded_tensor(encoded, "input_ids").to(device)
        arguments: dict[str, torch.Tensor] = {}
        config = getattr(text_encoder, "config", None)
        if bool(_config_value(config, "use_attention_mask")):
            arguments["attention_mask"] = _encoded_tensor(encoded, "attention_mask").to(
                device
            )
        with torch.inference_mode():
            output = text_encoder(input_ids, **arguments)
    except ModelLoadingError:
        raise
    except Exception as error:
        raise ModelLoadingError(f"Text-embedding preflight failed: {error}") from error
    embeddings = _component_tensor(output, "last_hidden_state", "text encoder")
    if embeddings.ndim != 3 or embeddings.shape[0] != 2:
        raise ModelLoadingError(
            "Text-embedding preflight expected shape [2, tokens, features]"
        )
    if embeddings.device != device:
        raise ModelLoadingError(
            f"Text encoder returned embeddings on {embeddings.device}; expected {device}"
        )
    if not embeddings.is_floating_point():
        raise ModelLoadingError("Text embeddings must be floating point")
    if not bool(torch.isfinite(embeddings).all().item()):
        raise ModelLoadingError("Text embeddings contain NaN or infinity")
    return embeddings


def _config_value(config: Any, key: str) -> object:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _encoded_tensor(encoded: Any, name: str) -> torch.Tensor:
    if isinstance(encoded, Mapping):
        value = encoded.get(name)
    else:
        value = getattr(encoded, name, None)
    if not isinstance(value, torch.Tensor):
        raise ModelLoadingError(f"Tokenizer result has no tensor {name!r}")
    return value


def _component_tensor(output: Any, name: str, label: str) -> torch.Tensor:
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    if value is None and isinstance(output, (tuple, list)) and output:
        value = output[0]
    if not isinstance(value, torch.Tensor):
        raise ModelLoadingError(f"{label} output has no tensor {name!r}")
    return value


def _preflight_scheduler(
    scheduler: Any,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    for method_name in ("set_timesteps", "scale_model_input", "step"):
        if not callable(getattr(scheduler, method_name, None)):
            raise ModelLoadingError(
                f"Scheduler has no callable {method_name} interface"
            )
    try:
        scheduler.set_timesteps(num_inference_steps, device=device)
    except Exception as error:
        raise ModelLoadingError(
            f"Scheduler cannot create {num_inference_steps} timesteps: {error}"
        ) from error
    raw_timesteps = getattr(scheduler, "timesteps", None)
    if raw_timesteps is None:
        raise ModelLoadingError("Scheduler did not create timesteps")
    try:
        timesteps = tuple(raw_timesteps)
    except TypeError as error:
        raise ModelLoadingError("Scheduler timesteps are not iterable") from error
    if len(timesteps) != num_inference_steps:
        raise ModelLoadingError(
            "Scheduler created "
            f"{len(timesteps)} timesteps; expected {num_inference_steps}"
        )


def _positive_inference_steps(value: object) -> int:
    if isinstance(value, bool):
        raise ModelLoadingError("num_inference_steps must be a positive integer")
    try:
        steps = operator.index(value)
    except TypeError as error:
        raise ModelLoadingError(
            "num_inference_steps must be a positive integer"
        ) from error
    if steps <= 0:
        raise ModelLoadingError("num_inference_steps must be a positive integer")
    return steps


def _diffusers_classes(
    pipeline_class: Any | None, vae_class: Any | None
) -> tuple[Any, Any]:
    if pipeline_class is not None and vae_class is not None:
        return pipeline_class, vae_class
    try:
        from diffusers import AutoencoderKL, StableDiffusionPipeline
    except ImportError as error:  # pragma: no cover - dependency message
        raise ModelLoadingError(
            "diffusers and transformers are required to load model checkpoints"
        ) from error
    return pipeline_class or StableDiffusionPipeline, vae_class or AutoencoderKL


def _required_component(pipeline: Any, name: str) -> Any:
    component = getattr(pipeline, name, None)
    if component is None:
        raise ModelLoadingError(f"Loaded pipeline has no {name} component")
    return component


def _move_for_inference(
    module: Any, runtime: RuntimeSelection, dtype: torch.dtype
) -> None:
    module.to(device=runtime.device, dtype=dtype)
    if hasattr(module, "eval"):
        module.eval()
