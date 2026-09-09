"""Small model-free tests for the single diffusion implementation."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from utils.common.io import canonical_hash
from utils.models import devices as devices_module
from utils.models import loading as loading_module
from utils.models import sampling as sampling_module
from utils.models.latent import (
    compute_latent_distances,
    decode_generated_latents,
    encode_target_latent,
    preprocess_target_image,
    target_preprocessing_policy,
)
from utils.models.registry import get_model_spec, model_names
from utils.models.sampling import (
    encode_prompt_condition,
    predict_conditional_epsilon,
    sample_trajectory,
    validate_latent_shape,
)
from utils.models.schedulers import (
    build_scheduler,
    build_scheduler_from_config,
    scheduler_config_dict,
    scheduler_step_kwargs,
)


class _Tokenizer:
    model_max_length = 1

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        assert texts == ["", "target prompt"]
        return {"input_ids": torch.tensor([[0], [1]], dtype=torch.int64)}


class _TextEncoder:
    config = {"use_attention_mask": False}

    def __call__(self, input_ids: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        return {"last_hidden_state": input_ids.float().unsqueeze(-1)}


class _UNet(torch.nn.Module):
    config = {"in_channels": 1, "sample_size": 1}

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(
        self,
        sample: torch.Tensor,
        _timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        assert return_dict is False
        values = encoder_hidden_states[:, 0, 0].reshape(-1, 1, 1, 1)
        return (values.expand_as(sample) + self.anchor,)


class _Scheduler:
    config = {"prediction_type": "epsilon"}
    init_noise_sigma = 1.0

    def __init__(self) -> None:
        self.timesteps = torch.tensor([], dtype=torch.int64)
        self.received: list[torch.Tensor] = []

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        assert steps == 1
        self.timesteps = torch.tensor([0], device=device)

    def scale_model_input(
        self, sample: torch.Tensor, _timestep: torch.Tensor
    ) -> torch.Tensor:
        return sample

    def step(
        self,
        prediction: torch.Tensor,
        _timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        eta: float,
    ) -> SimpleNamespace:
        assert eta == 0.0
        self.received.append(prediction.detach().cpu())
        return SimpleNamespace(prev_sample=sample - prediction)


def _mock_device_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
    cuda_count: int,
    cuda_current: int = 0,
    mps_available: bool = False,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: cuda_count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: cuda_current)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        monkeypatch.setattr(
            torch.backends,
            "mps",
            SimpleNamespace(is_available=lambda: mps_available),
            raising=False,
        )
    else:
        monkeypatch.setattr(
            mps_backend,
            "is_available",
            lambda: mps_available,
        )


def test_auto_device_resolution_uses_every_visible_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_device_backends(
        monkeypatch,
        cuda_available=True,
        cuda_count=3,
        mps_available=True,
    )

    assert devices_module.resolve_devices() == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
        torch.device("cuda:2"),
    )


def test_auto_device_resolution_falls_back_to_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_device_backends(
        monkeypatch,
        cuda_available=False,
        cuda_count=0,
        mps_available=True,
    )
    assert devices_module.resolve_devices("AUTO") == (torch.device("mps"),)

    _mock_device_backends(
        monkeypatch,
        cuda_available=False,
        cuda_count=0,
        mps_available=False,
    )
    assert devices_module.resolve_devices(" auto ") == (torch.device("cpu"),)


def test_explicit_device_resolution_selects_exactly_one_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_device_backends(
        monkeypatch,
        cuda_available=True,
        cuda_count=3,
        cuda_current=2,
        mps_available=True,
    )

    assert devices_module.resolve_devices("cpu") == (torch.device("cpu"),)
    assert devices_module.resolve_devices("mps") == (torch.device("mps"),)
    assert devices_module.resolve_devices("cuda") == (torch.device("cuda:2"),)
    assert devices_module.resolve_devices(torch.device("cuda:1")) == (
        torch.device("cuda:1"),
    )


def test_explicit_device_resolution_rejects_unavailable_or_invalid_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_device_backends(
        monkeypatch,
        cuda_available=False,
        cuda_count=0,
        mps_available=False,
    )
    with pytest.raises(devices_module.DeviceSelectionError, match="CUDA.*unavailable"):
        devices_module.resolve_devices("cuda:0")
    with pytest.raises(devices_module.DeviceSelectionError, match="MPS.*unavailable"):
        devices_module.resolve_devices("mps")
    with pytest.raises(devices_module.DeviceSelectionError, match="must not be empty"):
        devices_module.resolve_devices(" ")
    with pytest.raises(devices_module.DeviceSelectionError, match="must be auto"):
        devices_module.resolve_devices("meta")

    _mock_device_backends(
        monkeypatch,
        cuda_available=True,
        cuda_count=2,
    )
    with pytest.raises(
        devices_module.DeviceSelectionError,
        match="index 2.*unavailable",
    ):
        devices_module.resolve_devices("cuda:2")


def test_round_robin_shards_are_stable_disjoint_and_exhaustive() -> None:
    values = tuple("abcdefg")
    shards = tuple(
        devices_module.round_robin_shard(
            values,
            worker_index=index,
            worker_count=3,
        )
        for index in range(3)
    )

    assert shards == (("a", "d", "g"), ("b", "e"), ("c", "f"))
    assigned = [value for shard in shards for value in shard]
    assert len(assigned) == len(set(assigned)) == len(values)
    assert sorted(assigned) == sorted(values)

    with pytest.raises(ValueError, match="worker count must be positive"):
        devices_module.round_robin_shard(values, worker_index=0, worker_count=0)
    with pytest.raises(ValueError, match="smaller than worker count"):
        devices_module.round_robin_shard(values, worker_index=3, worker_count=3)


def test_worker_count_is_bounded_by_devices_and_tasks() -> None:
    devices = tuple(torch.device("cuda", index) for index in range(3))
    assert devices_module.worker_count_for_tasks(devices, 5) == 3
    assert devices_module.worker_count_for_tasks(devices, 2) == 2
    assert devices_module.worker_count_for_tasks(devices, 0) == 0
    with pytest.raises(ValueError, match="at least one execution device"):
        devices_module.worker_count_for_tasks((), 1)
    with pytest.raises(ValueError, match="task count must be a non-negative integer"):
        devices_module.worker_count_for_tasks(devices, -1)


def test_worker_cpu_threads_are_divided_without_exceeding_existing_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[int] = []
    monkeypatch.setattr(
        devices_module.os,
        "sched_getaffinity",
        lambda _pid: set(range(12)),
    )
    monkeypatch.setattr(devices_module.torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(
        devices_module.torch,
        "set_num_threads",
        lambda value: configured.append(value),
    )

    assert devices_module.configure_worker_cpu_threads(3) == 4
    assert configured == [4]

    configured.clear()
    assert devices_module.configure_worker_cpu_threads(1) == 8
    assert configured == []


def test_worker_cpu_threads_fall_back_to_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[int] = []
    monkeypatch.setattr(
        devices_module.os,
        "sched_getaffinity",
        lambda _pid: (_ for _ in ()).throw(OSError("unavailable")),
    )
    monkeypatch.setattr(devices_module.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(devices_module.torch, "get_num_threads", lambda: 12)
    monkeypatch.setattr(
        devices_module.torch,
        "set_num_threads",
        lambda value: configured.append(value),
    )

    assert devices_module.configure_worker_cpu_threads(4) == 1
    assert configured == [1]


def test_select_runtime_accepts_one_explicit_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_device_backends(
        monkeypatch,
        cuda_available=True,
        cuda_count=2,
    )
    inspected: list[torch.device] = []

    def gpu_name(device: torch.device) -> str:
        inspected.append(torch.device(device))
        return "Test GPU"

    monkeypatch.setattr(torch.cuda, "get_device_name", gpu_name)
    runtime = loading_module.select_runtime(torch.device("cuda:1"))

    assert runtime.device == torch.device("cuda:1")
    assert runtime.inference_dtype is torch.float16
    assert runtime.gpu_name == "Test GPU"
    assert inspected == [torch.device("cuda:1")]
    assert runtime.metadata() == {
        "device": "cuda:1",
        "device_type": "cuda",
        "gpu_name": "Test GPU",
        "inference_dtype": "float16",
    }


def test_select_runtime_uses_expected_non_cuda_dtypes_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mock_device_backends(
        monkeypatch,
        cuda_available=False,
        cuda_count=0,
        mps_available=True,
    )

    mps = loading_module.select_runtime("mps")
    cpu = loading_module.select_runtime("cpu")

    assert mps == loading_module.RuntimeSelection(
        torch.device("mps"),
        torch.float16,
        "Apple Metal Performance Shaders (MPS)",
    )
    assert cpu == loading_module.RuntimeSelection(
        torch.device("cpu"),
        torch.float32,
        None,
    )
    assert capsys.readouterr().err == ""


def test_select_runtime_preserves_no_argument_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_device_backends(
        monkeypatch,
        cuda_available=True,
        cuda_count=2,
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Default GPU")

    runtime = loading_module.select_runtime()

    assert runtime == loading_module.RuntimeSelection(
        torch.device("cuda"),
        torch.float16,
        "Default GPU",
    )


def test_select_runtime_rejects_multi_device_auto_request() -> None:
    with pytest.raises(loading_module.ModelLoadingError, match="resolve 'auto' first"):
        loading_module.select_runtime("auto")


def _compile_test_components(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device(device),
        unet=_UNet(),
        device_metadata={"device": device},
    )


def _call_compile_test_unet(components: SimpleNamespace) -> tuple[torch.Tensor]:
    return components.unet(
        torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        torch.tensor(0),
        encoder_hidden_states=torch.ones((1, 1, 1), dtype=torch.float32),
        return_dict=False,
    )


def _mock_public_compile_backends(
    monkeypatch: pytest.MonkeyPatch,
    *backends: str,
) -> None:
    monkeypatch.setattr(
        loading_module.torch,
        "compiler",
        SimpleNamespace(list_backends=lambda: backends),
        raising=False,
    )


def test_compile_loaded_unet_skips_non_cuda_without_calling_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _compile_test_components("cpu")
    unet = components.unet
    eager_forward = unet.forward

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("CPU inference must not invoke torch.compile")

    monkeypatch.setattr(loading_module.torch, "compile", forbidden, raising=False)
    _mock_public_compile_backends(monkeypatch, "inductor")

    observed = loading_module.compile_loaded_unet(components)

    assert observed is components
    assert observed.unet is unet
    assert observed.unet.forward == eager_forward
    assert observed.device_metadata["torch_compile"] == {
        "target": "unet.forward",
        "status": "skipped_non_cuda",
        "backend": None,
        "mode": None,
        "options": None,
        "fullgraph": None,
    }


def test_compile_loaded_unet_configures_bound_forward_and_becomes_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _compile_test_components("cuda:1")
    unet = components.unet
    eager_forward = unet.forward
    compile_calls: list[tuple[object, dict[str, object]]] = []
    compiled_calls = 0

    def fake_compile(target: object, **kwargs: object) -> object:
        compile_calls.append((target, dict(kwargs)))

        def compiled(*args: object, **call_kwargs: object) -> object:
            nonlocal compiled_calls
            compiled_calls += 1
            return target(*args, **call_kwargs)  # type: ignore[operator]

        return compiled

    monkeypatch.setattr(loading_module.torch, "compile", fake_compile, raising=False)
    _mock_public_compile_backends(monkeypatch, "eager", "inductor")

    observed = loading_module.compile_loaded_unet(components)

    assert observed is components
    assert observed.unet is unet
    assert len(compile_calls) == 1
    compiled_target, compile_kwargs = compile_calls[0]
    assert compiled_target == eager_forward
    assert compile_kwargs == {
        "backend": "inductor",
        "options": {
            "triton.cudagraphs": True,
            "triton.cudagraph_skip_dynamic_graphs": True,
        },
        "fullgraph": True,
    }
    assert observed.device_metadata["torch_compile"] == {
        "target": "unet.forward",
        "status": "configured",
        "backend": "inductor",
        "mode": None,
        "options": {
            "triton.cudagraphs": True,
            "triton.cudagraph_skip_dynamic_graphs": True,
        },
        "fullgraph": True,
    }

    first = _call_compile_test_unet(observed)
    second = _call_compile_test_unet(observed)
    torch.testing.assert_close(first[0], torch.ones((1, 1, 1, 1)))
    torch.testing.assert_close(second[0], torch.ones((1, 1, 1, 1)))
    assert compiled_calls == 2
    assert observed.device_metadata["torch_compile"]["status"] == "active"
    assert loading_module.compile_loaded_unet(observed) is observed
    assert len(compile_calls) == 1


@pytest.mark.parametrize(
    "availability",
    ("missing_compile", "noncallable_compile", "missing_inductor"),
)
def test_compile_loaded_unet_records_unavailable_environment(
    monkeypatch: pytest.MonkeyPatch,
    availability: str,
) -> None:
    components = _compile_test_components("cuda:0")
    unet = components.unet
    eager_forward = unet.forward

    if availability == "missing_compile":
        monkeypatch.delattr(loading_module.torch, "compile", raising=False)
    elif availability == "noncallable_compile":
        monkeypatch.setattr(
            loading_module.torch,
            "compile",
            None,
            raising=False,
        )
    else:

        def forbidden(*_: object, **__: object) -> object:
            raise AssertionError("missing backend must prevent torch.compile")

        monkeypatch.setattr(
            loading_module.torch,
            "compile",
            forbidden,
            raising=False,
        )
        _mock_public_compile_backends(monkeypatch, "eager")

    observed = loading_module.compile_loaded_unet(components)

    assert observed is components
    assert observed.unet is unet
    assert observed.unet.forward == eager_forward
    metadata = observed.device_metadata["torch_compile"]
    assert metadata["target"] == "unet.forward"
    assert metadata["status"] == "unavailable"
    assert metadata["backend"] is None
    assert metadata["mode"] is None
    assert metadata["options"] is None
    assert metadata["fullgraph"] is None
    assert isinstance(metadata["reason"], str)
    _call_compile_test_unet(observed)


def test_compile_loaded_unet_falls_back_after_immediate_setup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _compile_test_components("cuda:0")
    unet = components.unet
    eager_forward = unet.forward

    def failing_compile(*_: object, **__: object) -> object:
        raise RuntimeError("synthetic compile setup failure")

    monkeypatch.setattr(
        loading_module.torch,
        "compile",
        failing_compile,
        raising=False,
    )
    _mock_public_compile_backends(monkeypatch, "inductor")

    observed = loading_module.compile_loaded_unet(components)

    assert observed is components
    assert observed.unet is unet
    assert observed.unet.forward == eager_forward
    metadata = observed.device_metadata["torch_compile"]
    assert metadata == {
        "target": "unet.forward",
        "status": "eager_fallback",
        "backend": "inductor",
        "mode": None,
        "options": {
            "triton.cudagraphs": True,
            "triton.cudagraph_skip_dynamic_graphs": True,
        },
        "fullgraph": True,
        "reason": "RuntimeError: synthetic compile setup failure",
    }
    _call_compile_test_unet(observed)


def test_compile_loaded_unet_lazy_error_retries_once_then_stays_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _compile_test_components("cuda:0")
    unet = components.unet
    original_forward = unet.forward
    eager_calls = 0
    compiled_calls = 0

    def counted_eager(*args: object, **kwargs: object) -> object:
        nonlocal eager_calls
        eager_calls += 1
        return original_forward(*args, **kwargs)

    unet.forward = counted_eager

    def fake_compile(target: object, **_: object) -> object:
        assert target is counted_eager

        def failing_compiled(*_: object, **__: object) -> object:
            nonlocal compiled_calls
            compiled_calls += 1
            raise RuntimeError("synthetic lazy compiler failure")

        return failing_compiled

    monkeypatch.setattr(loading_module.torch, "compile", fake_compile, raising=False)
    _mock_public_compile_backends(monkeypatch, "inductor")
    observed = loading_module.compile_loaded_unet(components)

    first = _call_compile_test_unet(observed)
    assert compiled_calls == 1
    assert eager_calls == 1
    assert observed.unet.forward is counted_eager
    assert observed.device_metadata["torch_compile"]["status"] == "eager_fallback"
    assert "synthetic lazy compiler failure" in str(
        observed.device_metadata["torch_compile"]["reason"]
    )

    second = _call_compile_test_unet(observed)
    torch.testing.assert_close(first[0], torch.ones((1, 1, 1, 1)))
    torch.testing.assert_close(second[0], torch.ones((1, 1, 1, 1)))
    assert compiled_calls == 1
    assert eager_calls == 2


def test_compile_loaded_unet_cuda_oom_disables_compile_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _compile_test_components("cuda:0")
    unet = components.unet
    original_forward = unet.forward
    eager_calls = 0
    compiled_calls = 0

    def counted_eager(*args: object, **kwargs: object) -> object:
        nonlocal eager_calls
        eager_calls += 1
        return original_forward(*args, **kwargs)

    unet.forward = counted_eager

    def fake_compile(target: object, **_: object) -> object:
        assert target is counted_eager

        def oom_compiled(*_: object, **__: object) -> object:
            nonlocal compiled_calls
            compiled_calls += 1
            raise RuntimeError("synthetic CUDA out of memory")

        return oom_compiled

    monkeypatch.setattr(loading_module.torch, "compile", fake_compile, raising=False)
    _mock_public_compile_backends(monkeypatch, "inductor")
    observed = loading_module.compile_loaded_unet(components)

    with pytest.raises(RuntimeError, match="synthetic CUDA out of memory"):
        _call_compile_test_unet(observed)

    assert compiled_calls == 1
    assert eager_calls == 0
    assert observed.unet.forward is counted_eager
    assert observed.device_metadata["torch_compile"]["status"] == "eager_fallback"
    assert "out of memory" in str(observed.device_metadata["torch_compile"]["reason"])

    eager_result = _call_compile_test_unet(observed)
    torch.testing.assert_close(eager_result[0], torch.ones((1, 1, 1, 1)))
    assert compiled_calls == 1
    assert eager_calls == 1


def test_registry_has_one_explicit_spec_per_cli_model() -> None:
    assert model_names() == ("sdv1", "sdv2", "realvis")
    assert get_model_spec("realvis").dataset_model == "realisticvision"
    sdv2 = get_model_spec("sdv2")
    assert sdv2.model_id == "Manojb/stable-diffusion-2-1-base"
    assert sdv2.resolution == 512
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_spec("unknown")


def test_model_dependent_latent_dimensions_have_no_implicit_defaults() -> None:
    for function in (
        target_preprocessing_policy,
        preprocess_target_image,
        encode_target_latent,
    ):
        assert (
            inspect.signature(function).parameters["resolution"].default
            is inspect.Parameter.empty
        )
    assert (
        inspect.signature(validate_latent_shape).parameters["expected_shape"].default
        is inspect.Parameter.empty
    )


def test_latent_distances_are_raw_l2_squared_and_dimension_normalized() -> None:
    target = torch.zeros((1, 2, 2), dtype=torch.float32)
    generated = torch.stack((target, torch.ones_like(target)))
    distances = compute_latent_distances(generated, target)
    torch.testing.assert_close(distances.l2_norms, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(distances.l2_squared, torch.tensor([0.0, 4.0]))
    torch.testing.assert_close(distances.latent_rmse, torch.tensor([0.0, 1.0]))


def test_sampler_saves_both_cfg_branches_before_guidance() -> None:
    scheduler = _Scheduler()
    progress_updates: list[int] = []
    batch = sample_trajectory(
        prompt="target prompt",
        seeds=[3, 4],
        tokenizer=_Tokenizer(),
        text_encoder=_TextEncoder(),
        unet=_UNet(),
        scheduler=scheduler,
        device="cpu",
        inference_dtype=torch.float32,
        guidance_scale=3.0,
        num_inference_steps=1,
        scheduler_name="ddim",
        initial_microbatch_size=2,
        progress_callback=progress_updates.append,
    )

    assert batch.latents.shape == (2, 2, 1, 1, 1)
    torch.testing.assert_close(
        batch.unconditional_noise_predictions, torch.zeros((2, 1, 1, 1, 1))
    )
    torch.testing.assert_close(
        batch.conditional_noise_predictions, torch.ones((2, 1, 1, 1, 1))
    )
    assert len(scheduler.received) == 1
    torch.testing.assert_close(scheduler.received[0], torch.full((2, 1, 1, 1), 3.0))
    assert progress_updates == [2]


def test_encode_prompt_condition_never_constructs_null_branch() -> None:
    tokenized_texts: list[list[str]] = []

    class ConditionalTokenizer:
        model_max_length = 2

        def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
            tokenized_texts.append(list(texts))
            return {
                "input_ids": torch.tensor([[7, 8]], dtype=torch.int64),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
            }

    class ConditionalTextEncoder:
        config = {"use_attention_mask": True}

        def __call__(
            self,
            input_ids: torch.Tensor,
            *,
            attention_mask: torch.Tensor,
        ) -> SimpleNamespace:
            torch.testing.assert_close(attention_mask, torch.ones_like(attention_mask))
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))

    condition = encode_prompt_condition(
        "target prompt",
        ConditionalTokenizer(),
        ConditionalTextEncoder(),
        "cpu",
        torch.float64,
    )

    assert tokenized_texts == [["target prompt"]]
    assert condition.shape == (1, 2, 1)
    assert condition.dtype == torch.float64
    torch.testing.assert_close(
        condition,
        torch.tensor([[[7.0], [8.0]]], dtype=torch.float64),
    )


def test_conditional_prediction_scales_unet_input_and_converts_unscaled_state() -> None:
    class ScalingScheduler:
        config = {"prediction_type": "v_prediction"}
        alphas_cumprod = torch.tensor([0.25], dtype=torch.float32)

        def __init__(self) -> None:
            self.scale_calls = 0
            self.step_calls = 0
            self.received_unscaled: torch.Tensor | None = None

        def scale_model_input(
            self, sample: torch.Tensor, timestep: int
        ) -> torch.Tensor:
            assert timestep == 0
            self.scale_calls += 1
            self.received_unscaled = sample.detach().clone()
            return sample * 3.0

        def step(self, *_: object, **__: object) -> None:
            self.step_calls += 1
            raise AssertionError("conditional prediction must not step the scheduler")

    class RecordingUNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float16))
            self.calls = 0
            self.received_sample: torch.Tensor | None = None
            self.received_condition: torch.Tensor | None = None

        def forward(
            self,
            sample: torch.Tensor,
            timestep: int,
            *,
            encoder_hidden_states: torch.Tensor,
            return_dict: bool,
        ) -> tuple[torch.Tensor]:
            assert return_dict is False
            assert timestep == 0
            self.calls += 1
            self.received_sample = sample.detach().clone()
            self.received_condition = encoder_hidden_states.detach().clone()
            return (torch.full_like(sample, 0.5) + self.anchor,)

    model_samples = torch.tensor([[[[1.0001]]], [[[2.0002]]]], dtype=torch.float32).to(
        torch.float16
    )
    conceptual_samples = torch.tensor([[[[1.0001]]], [[[2.0002]]]], dtype=torch.float32)
    original_model_samples = model_samples.clone()
    condition = torch.tensor([[[4.0]]], dtype=torch.float16)
    scheduler = ScalingScheduler()
    unet = RecordingUNet()

    epsilon = predict_conditional_epsilon(
        model_samples,
        0,
        condition,
        unet,
        scheduler,
        conversion_sample=conceptual_samples,
    )

    assert scheduler.scale_calls == 1
    assert scheduler.step_calls == 0
    assert unet.calls == 1
    torch.testing.assert_close(model_samples, original_model_samples)
    torch.testing.assert_close(scheduler.received_unscaled, original_model_samples)
    torch.testing.assert_close(unet.received_sample, original_model_samples * 3.0)
    torch.testing.assert_close(
        unet.received_condition,
        condition.expand(model_samples.shape[0], *condition.shape[1:]),
    )
    expected = 0.5 * torch.full_like(conceptual_samples, 0.5)
    expected += torch.sqrt(torch.tensor(0.75)) * conceptual_samples
    assert epsilon.dtype == torch.float32
    torch.testing.assert_close(epsilon, expected)


def test_adaptive_sampler_rolls_back_failed_microbatch_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_updates: list[int] = []
    calls = 0

    def fake_microbatch(**keywords: Any) -> object:
        nonlocal calls
        calls += 1
        batch_size = len(keywords["seeds"])
        report = keywords["progress_callback"]
        if calls == 1:
            report(batch_size)
            raise RuntimeError("synthetic CUDA out of memory")
        for _ in range(keywords["num_inference_steps"]):
            report(batch_size)
        return SimpleNamespace()

    monkeypatch.setattr(sampling_module, "_sample_microbatch", fake_microbatch)
    monkeypatch.setattr(sampling_module, "_is_cuda_out_of_memory", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    parts = sampling_module._sample_adaptive(
        seed_values=[0, 1, 2, 3],
        latent_shape=(1, 1, 1),
        unconditional=torch.zeros((1, 1, 1)),
        conditional=torch.ones((1, 1, 1)),
        unet=object(),
        scheduler=object(),
        scheduler_name="ddim",
        native_prediction_type="epsilon",
        device=torch.device("cuda"),
        inference_dtype=torch.float32,
        guidance_scale=7.5,
        num_inference_steps=2,
        initial_microbatch_size=4,
        progress_callback=progress_updates.append,
    )

    assert len(parts) == 2
    assert progress_updates == [4, -4, 2, 2, 2, 2]
    assert sum(progress_updates) == 4 * 2


def test_preview_decode_reports_each_completed_image() -> None:
    class VAE:
        config = {"scaling_factor": 1.0}

        def to(self, **_: object) -> "VAE":
            return self

        def eval(self) -> "VAE":
            return self

        def decode(self, latents: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                sample=torch.zeros(
                    (latents.shape[0], 3, latents.shape[2], latents.shape[3]),
                    dtype=torch.float32,
                )
            )

    progress_updates: list[int] = []
    images = decode_generated_latents(
        torch.zeros((5, 1, 2, 2), dtype=torch.float32),
        VAE(),
        "cpu",
        microbatch_size=2,
        progress_callback=progress_updates.append,
    )

    assert len(images) == 5
    assert progress_updates == [2, 2, 1]


def test_scheduler_construction_preserves_native_configuration() -> None:
    class Built:
        def __init__(self, config: dict[str, object]):
            self.config = dict(config)

        @classmethod
        def from_config(cls, config: dict[str, object]) -> "Built":
            return cls(config)

        def step(self, *_: object, eta: float = 0.0) -> None:
            return None

    original = SimpleNamespace(
        config={"prediction_type": "v_prediction", "beta_start": 0.1}
    )
    result = build_scheduler(original, "ddim", scheduler_classes={"ddim": Built})
    assert result.config == original.config
    assert scheduler_step_kwargs(result.scheduler, "ddim") == {"eta": 0.0}


def test_scheduler_construction_from_config_matches_component_wrapper() -> None:
    received: list[dict[str, object]] = []

    class Built:
        def __init__(self, config: dict[str, object]) -> None:
            self.config = dict(config)

        @classmethod
        def from_config(cls, config: dict[str, object]) -> "Built":
            received.append(config)
            return cls(config)

    config = {
        "prediction_type": "v_prediction",
        "beta_start": 0.00085,
        "_use_default_values": ["thresholding", "clip_sample_range"],
    }
    direct = build_scheduler_from_config(
        config,
        "ddim",
        scheduler_classes={"ddim": Built},
    )
    wrapped = build_scheduler(
        SimpleNamespace(config=config),
        "ddim",
        scheduler_classes={"ddim": Built},
    )

    assert received == [config, config]
    assert direct.name == wrapped.name == "ddim"
    assert direct.class_name == wrapped.class_name == "Built"
    assert (
        direct.config
        == wrapped.config
        == {
            "prediction_type": "v_prediction",
            "beta_start": 0.00085,
            "_use_default_values": ["clip_sample_range", "thresholding"],
        }
    )
    assert direct.removed_config_keys == wrapped.removed_config_keys == ()
    assert direct.metadata() == wrapped.metadata()


def test_scheduler_construction_from_config_rejects_changed_prediction_type() -> None:
    class Built:
        config = {"prediction_type": "epsilon"}

        @classmethod
        def from_config(cls, _config: object) -> "Built":
            return cls()

    with pytest.raises(
        RuntimeError,
        match="changed the checkpoint prediction_type",
    ):
        build_scheduler_from_config(
            {"prediction_type": "v_prediction"},
            "ddim",
            scheduler_classes={"ddim": Built},
        )


def test_scheduler_construction_from_config_rejects_absent_configuration() -> None:
    with pytest.raises(RuntimeError, match="Checkpoint scheduler has no config"):
        build_scheduler_from_config(None, "ddim", scheduler_classes={})


def test_scheduler_default_key_order_is_canonical() -> None:
    first = SimpleNamespace(
        config={
            "prediction_type": "epsilon",
            "_use_default_values": [
                "thresholding",
                "clip_sample_range",
                "sample_max_value",
            ],
        }
    )
    second = SimpleNamespace(
        config={
            "prediction_type": "epsilon",
            "_use_default_values": [
                "sample_max_value",
                "thresholding",
                "clip_sample_range",
            ],
        }
    )

    first_config = scheduler_config_dict(first)
    second_config = scheduler_config_dict(second)
    assert first_config == second_config
    assert first_config["_use_default_values"] == [
        "clip_sample_range",
        "sample_max_value",
        "thresholding",
    ]
    assert canonical_hash(first_config) == canonical_hash(second_config)


def test_only_one_sampler_and_one_denoising_loop_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    function_locations: list[tuple[Path, str]] = []
    loop_locations: list[Path] = []
    unet_loop_locations: list[Path] = []
    for source_path in (root / "utils").rglob("*.py"):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "sample_trajectory"
            ):
                function_locations.append((source_path, node.name))
            if isinstance(node, (ast.For, ast.While)):
                calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)]
                if any(
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "scheduler"
                    and call.func.attr == "step"
                    for call in calls
                ):
                    loop_locations.append(source_path)
                if any(
                    isinstance(call.func, ast.Name) and call.func.id == "unet"
                    for call in calls
                ):
                    unet_loop_locations.append(source_path)

    expected = root / "utils/models/sampling.py"
    assert function_locations == [(expected, "sample_trajectory")]
    assert loop_locations == [expected]
    assert unet_loop_locations == [expected]
