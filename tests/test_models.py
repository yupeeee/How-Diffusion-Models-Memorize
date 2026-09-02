"""Small model-free tests for the single diffusion implementation."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from utils.common.io import canonical_hash
from utils.models import sampling as sampling_module
from utils.models.latent import compute_latent_distances, decode_generated_latents
from utils.models.registry import get_model_spec, model_names
from utils.models.sampling import (
    encode_prompt_condition,
    predict_conditional_epsilon,
    sample_trajectory,
)
from utils.models.schedulers import (
    build_scheduler,
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
    ) -> SimpleNamespace:
        values = encoder_hidden_states[:, 0, 0].reshape(-1, 1, 1, 1)
        return SimpleNamespace(sample=values.expand_as(sample) + self.anchor)


class _Scheduler:
    config = {"prediction_type": "epsilon"}
    init_noise_sigma = 1.0

    def __init__(self) -> None:
        self.timesteps = torch.tensor([], dtype=torch.int64)
        self.received: list[torch.Tensor] = []

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        assert steps == 1
        self.timesteps = torch.tensor([0], device=device)

    def scale_model_input(self, sample: torch.Tensor, _timestep: torch.Tensor) -> torch.Tensor:
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


def test_registry_has_one_explicit_spec_per_cli_model() -> None:
    assert model_names() == ("sdv1", "sdv2", "realvis")
    assert get_model_spec("realvis").dataset_model == "realisticvision"
    sdv2 = get_model_spec("sdv2")
    assert sdv2.model_id == "Manojb/stable-diffusion-2-1-base"
    assert sdv2.resolution == 512
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_spec("unknown")


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
        ) -> SimpleNamespace:
            assert timestep == 0
            self.calls += 1
            self.received_sample = sample.detach().clone()
            self.received_condition = encoder_hidden_states.detach().clone()
            return SimpleNamespace(sample=torch.full_like(sample, 0.5) + self.anchor)

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
    monkeypatch.setattr(
        sampling_module, "_is_cuda_out_of_memory", lambda *_: True
    )
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

    original = SimpleNamespace(config={"prediction_type": "v_prediction", "beta_start": 0.1})
    result = build_scheduler(original, "ddim", scheduler_classes={"ddim": Built})
    assert result.config == original.config
    assert scheduler_step_kwargs(result.scheduler, "ddim") == {"eta": 0.0}


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
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "sample_trajectory":
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
                if any(isinstance(call.func, ast.Name) and call.func.id == "unet" for call in calls):
                    unet_loop_locations.append(source_path)

    expected = root / "utils/models/sampling.py"
    assert function_locations == [(expected, "sample_trajectory")]
    assert loop_locations == [expected]
    assert unet_loop_locations == [expected]
