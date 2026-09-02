"""Focused model-free contracts for the pair-level Theorem 1 experiment."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PathCollection, QuadMesh
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import theorem1_loss_recovery as experiment


EXPECTED_COLUMNS = (
    "record_id",
    "model_name",
    "timestep",
    "alpha_t",
    "sigma_t",
    "snr_t",
    "latent_dimension",
    "num_loss_seeds",
    "loss_seed",
    "generation_seeds",
    "num_generation_seeds",
    "conditional_loss",
    "normalized_loss_mse",
    "normalized_loss_rmse",
    "recovery_mse",
    "recovery_rmse",
    "mean_target_sscd",
    "status",
    "error",
)


def _field(result: object, name: str, index: int | None = None) -> object:
    if isinstance(result, Mapping):
        return result[name]
    if hasattr(result, name):
        return getattr(result, name)
    if index is not None and isinstance(result, Sequence):
        return result[index]
    raise AssertionError(f"result has no {name!r} field: {result!r}")


def _seed_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if isinstance(value, Sequence):
        return tuple(int(item) for item in value)
    raise AssertionError(f"generation seed value is not a sequence: {value!r}")


def _prediction_samples(
    args: Sequence[object], kwargs: Mapping[str, object]
) -> torch.Tensor:
    value = kwargs.get("samples")
    if value is None and args:
        value = args[0]
    if not isinstance(value, torch.Tensor):
        raise AssertionError("conditional predictor did not receive a sample tensor")
    return value


def test_cli_schema_defaults_and_output_names_are_exact() -> None:
    parser = experiment.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"--help", "-h"}
    }
    assert options == {
        "--model",
        "--num-loss-seeds",
        "--loss-seed",
        "--generation-seeds",
        "--max-generation-seeds",
        "--sample-batch-size",
        "--max-records",
        "--output-dir",
        "--plot",
        "--plot-only",
    }

    arguments = parser.parse_args([])
    assert arguments.model == "sdv1"
    assert arguments.num_loss_seeds == 20
    assert arguments.loss_seed == 0
    assert _seed_tuple(arguments.generation_seeds) == tuple(range(20))
    assert arguments.max_generation_seeds is None
    assert arguments.sample_batch_size > 0
    assert arguments.max_records is None
    assert arguments.output_dir is None
    assert arguments.plot_only is False
    assert parser.parse_args(["--plot"]).plot_only is True
    assert parser.parse_args(["--plot-only"]).plot_only is True
    expected_output_directories = {
        model_name: (
            experiment.ROOT
            / "outputs"
            / f"{model_name}_ddim_g7.5_T50_N20"
            / "theorem1_loss_recovery"
        )
        for model_name in ("sdv1", "sdv2", "realvis")
    }
    assert {
        model_name: experiment._default_output_directory(model_name)
        for model_name in expected_output_directories
    } == expected_output_directories
    assert len(set(expected_output_directories.values())) == 3

    smoke = parser.parse_args(
        [
            "--max-records",
            "2",
            "--num-loss-seeds",
            "2",
            "--max-generation-seeds",
            "2",
        ]
    )
    assert (
        smoke.max_records,
        smoke.num_loss_seeds,
        smoke.max_generation_seeds,
    ) == (2, 2, 2)
    custom = parser.parse_args(["--generation-seeds", "2,5,11"])
    assert _seed_tuple(custom.generation_seeds) == (2, 5, 11)

    with pytest.raises(SystemExit):
        parser.parse_args(["--num-loss-seeds", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss-seed", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-generation-seeds", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-loss", "2"])

    assert experiment.CSV_COLUMNS == EXPECTED_COLUMNS
    assert experiment.CSV_NAME == "theorem1_loss_recovery.csv"
    assert experiment.FIGURE_NAME == "theorem1_loss_recovery.pdf"


def test_conditional_loss_uses_sum_then_mean_then_square_root_and_checks_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.tensor([[[1.0, -1.0], [0.5, 2.0]]], dtype=torch.float32)
    loss_noise = torch.tensor(
        [
            [[[1.0, 2.0], [0.0, 0.0]]],
            [[[-1.0, 0.0], [2.0, 2.0]]],
        ],
        dtype=torch.float32,
    )
    predictor_inputs: list[torch.Tensor] = []

    def zero_prediction(*args: object, **kwargs: object) -> torch.Tensor:
        samples = _prediction_samples(args, kwargs)
        predictor_inputs.append(samples.detach().cpu().clone())
        return torch.zeros_like(samples)

    monkeypatch.setattr(experiment, "predict_conditional_epsilon", zero_prediction)
    components = SimpleNamespace(
        device=torch.device("cpu"),
        inference_dtype=torch.float32,
        unet=object(),
    )
    alpha_t, sigma_t, snr_t = 0.6, 0.8, 0.5625
    result = experiment._measure_conditional_loss(
        target_latent=target,
        loss_noise=loss_noise,
        condition=torch.ones((1, 1, 1), dtype=torch.float32),
        timestep=876,
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        snr_t=snr_t,
        components=components,
        scheduler=object(),
        sample_batch_size=2,
    )

    expected_q = alpha_t * target.unsqueeze(0) + sigma_t * loss_noise
    assert len(predictor_inputs) == 1
    torch.testing.assert_close(predictor_inputs[0], expected_q)
    per_seed_sum = loss_noise.double().square().flatten(1).sum(1)
    assert per_seed_sum.tolist() == [5.0, 9.0]
    expected_mse = 7.0 / (target.numel() * snr_t)
    expected_rmse = math.sqrt(expected_mse)
    assert float(_field(result, "conditional_loss", 0)) == pytest.approx(7.0)
    assert float(_field(result, "normalized_loss_mse", 1)) == pytest.approx(
        expected_mse
    )
    assert float(_field(result, "normalized_loss_rmse", 2)) == pytest.approx(
        expected_rmse
    )
    mean_seedwise_rmse = float(
        torch.sqrt(per_seed_sum / (target.numel() * snr_t)).mean()
    )
    assert expected_rmse != pytest.approx(mean_seedwise_rmse)

    first_reconstruction = (
        expected_q[0] - sigma_t * torch.zeros_like(loss_noise[0])
    ) / alpha_t
    identity_lhs = float(
        (first_reconstruction - target).double().square().sum() / target.numel()
    )
    identity_rhs = float(per_seed_sum[0] / (target.numel() * snr_t))
    assert identity_lhs == pytest.approx(identity_rhs)

    with pytest.raises(
        (ValueError, RuntimeError), match="identity|SNR|snr|alpha|sigma"
    ):
        experiment._measure_conditional_loss(
            target_latent=target,
            loss_noise=loss_noise,
            condition=torch.ones((1, 1, 1), dtype=torch.float32),
            timestep=876,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            snr_t=0.5,
            components=components,
            scheduler=object(),
            sample_batch_size=2,
        )


def test_generation_initial_latents_match_canonical_sampler_and_recovery_has_no_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_noise = torch.tensor(
        [
            [[[0.25, 0.50]]],
            [[[0.50, 1.00]]],
        ],
        dtype=torch.float32,
    )
    noise_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def canonical_noise(
        seeds: Sequence[int], latent_shape: Sequence[int]
    ) -> torch.Tensor:
        noise_calls.append(
            (
                tuple(int(seed) for seed in seeds),
                tuple(int(size) for size in latent_shape),
            )
        )
        return stored_noise.clone()

    monkeypatch.setattr(experiment, "make_initial_noise", canonical_noise)
    reconstructed = experiment._reconstruct_generation_initial_latents(
        seeds=(3, 7),
        latent_shape=(1, 1, 2),
        stored_dtype=torch.float32,
        init_noise_sigma=1.0,
    )
    assert noise_calls == [((3, 7), (1, 1, 2))]
    torch.testing.assert_close(reconstructed, stored_noise)

    predictor_inputs: list[torch.Tensor] = []

    def zero_prediction(*args: object, **kwargs: object) -> torch.Tensor:
        samples = _prediction_samples(args, kwargs)
        predictor_inputs.append(samples.detach().cpu().clone())
        return torch.zeros_like(samples)

    def forbidden_step(*_: object, **__: object) -> None:
        raise AssertionError("the loss-recovery experiment must not denoise")

    monkeypatch.setattr(experiment, "predict_conditional_epsilon", zero_prediction)
    target = torch.zeros((1, 1, 2), dtype=torch.float32)
    components = SimpleNamespace(
        device=torch.device("cpu"),
        inference_dtype=torch.float32,
        unet=object(),
    )
    result = experiment._measure_recovery(
        target_latent=target,
        generation_initial_latents=reconstructed,
        condition=torch.ones((1, 1, 1), dtype=torch.float32),
        timestep=876,
        alpha_t=0.5,
        sigma_t=math.sqrt(0.75),
        components=components,
        scheduler=SimpleNamespace(step=forbidden_step),
        sample_batch_size=2,
    )

    assert len(predictor_inputs) == 1
    torch.testing.assert_close(predictor_inputs[0], stored_noise)
    per_seed_mse = torch.tensor([0.625, 2.5], dtype=torch.float64)
    expected_mse = float(per_seed_mse.mean())
    expected_rmse = math.sqrt(expected_mse)
    assert float(_field(result, "recovery_mse", 0)) == pytest.approx(expected_mse)
    assert float(_field(result, "recovery_rmse", 1)) == pytest.approx(expected_rmse)
    assert expected_rmse != pytest.approx(float(torch.sqrt(per_seed_mse).mean()))


def test_loss_noise_stream_is_reproducible_and_independent_of_generation_seeds() -> (
    None
):
    generation_seeds = tuple(range(20))
    first_seeds = experiment._loss_sample_seeds(
        0,
        3,
        forbidden_seeds=generation_seeds,
    )
    repeated_seeds = experiment._loss_sample_seeds(
        0,
        3,
        forbidden_seeds=generation_seeds,
    )
    different_seeds = experiment._loss_sample_seeds(
        1,
        3,
        forbidden_seeds=generation_seeds,
    )

    assert first_seeds == repeated_seeds
    assert len(first_seeds) == len(set(first_seeds)) == 3
    assert set(first_seeds).isdisjoint(generation_seeds)
    assert different_seeds != first_seeds
    assert set(different_seeds).isdisjoint(generation_seeds)

    first_noise = experiment._make_loss_noise(
        num_loss_seeds=3,
        latent_shape=(1, 2, 2),
        loss_seed=0,
        forbidden_seeds=generation_seeds,
    )
    repeated_noise = experiment._make_loss_noise(
        num_loss_seeds=3,
        latent_shape=(1, 2, 2),
        loss_seed=0,
        forbidden_seeds=generation_seeds,
    )
    canonical_loss_noise = experiment.make_initial_noise(first_seeds, (1, 2, 2))
    canonical_generation_noise = experiment.make_initial_noise((0, 1, 2), (1, 2, 2))
    torch.testing.assert_close(first_noise, repeated_noise)
    torch.testing.assert_close(first_noise, canonical_loss_noise)
    assert not torch.equal(first_noise, canonical_generation_noise)


class _FakeProgress:
    instances: list[_FakeProgress] = []

    def __init__(self, iterable: object = None, **kwargs: object) -> None:
        self.iterable = iterable
        self.kwargs = kwargs
        self.updated = 0
        self.postfixes: list[dict[str, object]] = []
        type(self).instances.append(self)

    def __enter__(self) -> _FakeProgress:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __iter__(self):  # type: ignore[no-untyped-def]
        if self.iterable is None:
            return
        for value in self.iterable:  # type: ignore[union-attr]
            yield value
            self.updated += 1

    def update(self, amount: int = 1) -> None:
        self.updated += amount

    def set_postfix(
        self,
        ordered_dict: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self.postfixes.append(values)

    def close(self) -> None:
        return None


class _EvalModule:
    def __init__(self, **config: object) -> None:
        self.config = SimpleNamespace(**config)
        self.training = True

    def eval(self) -> _EvalModule:
        self.training = False
        return self


def test_mocked_realvis_pair_run_keeps_alias_streams_seed_alignment_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "record_id": "realisticvision-0000",
            "model_name": "realisticvision",
            "prompt": "first prompt",
            "prompt_raw": "first prompt",
            "image": object(),
            "original_index": "100",
            "source_row_number": 10,
            "target_image_sha256": "a" * 64,
        },
        {
            "record_id": "realisticvision-0001",
            "model_name": "realisticvision",
            "prompt": "second prompt",
            "prompt_raw": "second prompt",
            "image": object(),
            "original_index": "101",
            "source_row_number": 11,
            "target_image_sha256": "b" * 64,
        },
    ]

    class FakeDataset:
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, object]:
            return dict(records[index])

        def iter_metadata(self):  # type: ignore[no-untyped-def]
            for record in records:
                metadata = dict(record)
                metadata["image_available"] = True
                metadata["image"] = None
                yield metadata

    alpha_bar = torch.tensor(
        [0.9, 0.8, 0.7, 0.36, 0.2, 0.1, 0.05, 0.01], dtype=torch.float64
    )
    scheduler = SimpleNamespace(
        alphas_cumprod=alpha_bar,
        timesteps=torch.tensor([3, 2, 1], dtype=torch.int64),
        init_noise_sigma=1.0,
        config=SimpleNamespace(num_train_timesteps=len(alpha_bar)),
        set_timesteps=lambda *_args, **_kwargs: None,
    )
    components = SimpleNamespace(
        device=torch.device("cpu"),
        inference_dtype=torch.float32,
        unet=_EvalModule(in_channels=1, sample_size=2),
        vae=_EvalModule(),
        tokenizer=object(),
        text_encoder=_EvalModule(),
        scheduler=scheduler,
    )
    generation_contract = SimpleNamespace(
        latent_shape=(1, 2, 2),
        stored_dtype=torch.float32,
        latent_dtype=torch.float32,
        init_noise_sigma=1.0,
        seeds=tuple(range(20)),
        generation_seeds=tuple(range(20)),
        scheduler_config={},
        paths=SimpleNamespace(),
        sscd_paths=SimpleNamespace(),
    )

    requested_dataset_models: list[str] = []

    def fake_dataset(_root: Path, dataset_model: str, **_kwargs: object) -> FakeDataset:
        requested_dataset_models.append(dataset_model)
        return FakeDataset()

    monkeypatch.setattr(experiment, "WebsterDataset", fake_dataset)
    monkeypatch.setattr(
        experiment,
        "_load_generation_contract",
        lambda *_args, **_kwargs: generation_contract,
    )
    monkeypatch.setattr(
        experiment,
        "_validate_active_scheduler",
        lambda *_args, **_kwargs: scheduler,
    )
    monkeypatch.setattr(
        experiment,
        "load_model_components",
        lambda *_args, **_kwargs: components,
    )
    monkeypatch.setattr(
        experiment, "preflight_model_components", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        experiment,
        "_validate_loaded_components",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        experiment,
        "encode_prompt_condition",
        lambda *_args, **_kwargs: torch.ones((1, 1, 1), dtype=torch.float32),
    )

    def encode_target(image: object, *_args: object, **_kwargs: object) -> torch.Tensor:
        if image is records[1]["image"]:
            raise RuntimeError("synthetic target failure")
        return torch.ones((1, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(experiment, "encode_target_latent", encode_target)
    _FakeProgress.instances.clear()

    loss_noises: list[torch.Tensor] = []
    recovery_latents: list[torch.Tensor] = []
    requested_recovery_seeds: list[tuple[int, ...]] = []
    requested_sscd_seeds: list[tuple[int, ...]] = []
    generation_initial = torch.full((2, 1, 2, 2), 99.0, dtype=torch.float32)

    def reconstruct(
        seeds: Sequence[int], *_args: object, **_kwargs: object
    ) -> torch.Tensor:
        requested_recovery_seeds.append(tuple(int(seed) for seed in seeds))
        return generation_initial.clone()

    monkeypatch.setattr(
        experiment, "_reconstruct_generation_initial_latents", reconstruct
    )

    def make_loss_noise(*_args: object, **kwargs: object) -> torch.Tensor:
        count = int(kwargs.get("num_loss_seeds", kwargs.get("count", 2)))
        return torch.arange(count * 4, dtype=torch.float32).reshape(count, 1, 2, 2)

    monkeypatch.setattr(experiment, "_make_loss_noise", make_loss_noise)

    def measure_loss(*_args: object, **kwargs: object) -> experiment.LossMeasurement:
        noise = kwargs.get("loss_noise")
        if not isinstance(noise, torch.Tensor):
            raise AssertionError("run did not pass pair loss-noise samples")
        loss_noises.append(noise.detach().cpu().clone())
        return experiment.LossMeasurement(8.0, 2.0, math.sqrt(2.0))

    def measure_recovery(
        *_args: object, **kwargs: object
    ) -> experiment.RecoveryMeasurement:
        latents = kwargs.get("generation_initial_latents")
        if not isinstance(latents, torch.Tensor):
            raise AssertionError("run did not pass generation initial latents")
        recovery_latents.append(latents.detach().cpu().clone())
        return experiment.RecoveryMeasurement(3.0, math.sqrt(3.0))

    monkeypatch.setattr(experiment, "_measure_conditional_loss", measure_loss)
    monkeypatch.setattr(experiment, "_measure_recovery", measure_recovery)

    def target_sscd(*args: object, **kwargs: object) -> float:
        seeds = kwargs.get(
            "generation_seeds",
            kwargs.get("selected_seeds", kwargs.get("seeds")),
        )
        if not isinstance(seeds, Sequence):
            for candidate in reversed(args):
                if isinstance(candidate, Sequence) and not isinstance(
                    candidate, (str, bytes, torch.Tensor)
                ):
                    seeds = candidate
                    break
        if not isinstance(seeds, Sequence):
            raise AssertionError("SSCD lookup did not receive generation seeds")
        requested_sscd_seeds.append(tuple(int(seed) for seed in seeds))
        return 0.4

    monkeypatch.setattr(experiment, "_load_mean_target_sscd", target_sscd)

    plotted: list[tuple[Path, Path]] = []

    def fake_plot(csv_path: Path, figure_path: Path) -> None:
        assert Path(csv_path).is_file()
        plotted.append((Path(csv_path), Path(figure_path)))
        Path(figure_path).write_bytes(b"%PDF-1.4\n%%EOF\n")

    monkeypatch.setattr(experiment, "plot_saved_results", fake_plot)
    output = tmp_path / "loss-recovery"
    experiment.run_experiment(
        model_name="realvis",
        num_loss_seeds=2,
        loss_seed=7,
        generation_seeds=(2, 5),
        max_generation_seeds=None,
        sample_batch_size=2,
        max_records=2,
        output_dir=output,
        progress_factory=_FakeProgress,
    )

    frame = pd.read_csv(output / experiment.CSV_NAME, keep_default_na=False)
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 2
    assert requested_dataset_models == ["realisticvision"]
    assert frame["record_id"].tolist() == [
        "realisticvision-0000",
        "realisticvision-0001",
    ]
    assert frame["model_name"].tolist() == ["realvis", "realvis"]
    first, second = frame.iloc[0], frame.iloc[1]
    assert first["status"] == "ok" and first["error"] == ""
    assert second["status"] == "error"
    assert "synthetic target failure" in str(second["error"])
    assert int(first["timestep"]) == 3
    assert float(first["alpha_t"]) == pytest.approx(0.6)
    assert float(first["sigma_t"]) == pytest.approx(0.8)
    assert float(first["snr_t"]) == pytest.approx(0.5625)
    assert int(first["num_loss_seeds"]) == 2
    assert int(first["loss_seed"]) == 7
    assert str(first["generation_seeds"]) == "2,5"
    assert int(first["num_generation_seeds"]) == 2
    assert float(first["conditional_loss"]) == pytest.approx(8.0)
    assert float(first["normalized_loss_mse"]) == pytest.approx(2.0)
    assert float(first["normalized_loss_rmse"]) == pytest.approx(math.sqrt(2.0))
    assert float(first["recovery_mse"]) == pytest.approx(3.0)
    assert float(first["recovery_rmse"]) == pytest.approx(math.sqrt(3.0))
    assert float(first["mean_target_sscd"]) == pytest.approx(0.4)

    assert requested_recovery_seeds == [(2, 5)]
    assert requested_sscd_seeds == [(2, 5)]
    assert len(loss_noises) == len(recovery_latents) == 1
    assert loss_noises[0].shape == recovery_latents[0].shape == (2, 1, 2, 2)
    assert not torch.equal(loss_noises[0], recovery_latents[0])
    torch.testing.assert_close(recovery_latents[0], generation_initial)

    assert len(_FakeProgress.instances) == 1
    progress = _FakeProgress.instances[0]
    assert progress.kwargs["total"] == 2
    assert progress.kwargs["desc"] == "Theorem 1 loss–recovery"
    assert progress.kwargs["dynamic_ncols"] is True
    assert progress.kwargs["leave"] is True
    assert progress.updated == 2
    assert [values.get("record_id") for values in progress.postfixes] == [
        "realisticvision-0000",
        "realisticvision-0001",
    ]
    assert plotted == [(output / experiment.CSV_NAME, output / experiment.FIGURE_NAME)]
    assert {path.name for path in output.iterdir()} == {
        experiment.CSV_NAME,
        experiment.FIGURE_NAME,
    }


def _row(
    record_id: str,
    *,
    normalized_loss_rmse: float,
    recovery_rmse: float,
    mean_target_sscd: float,
    status: str = "ok",
    error: str = "",
    num_loss_seeds: int = 2,
    generation_seeds: str = "0,1",
) -> dict[str, object]:
    alpha_t = 0.1
    sigma_t = math.sqrt(1.0 - alpha_t**2)
    snr_t = alpha_t**2 / sigma_t**2
    normalized_loss_mse = normalized_loss_rmse**2
    recovery_mse = recovery_rmse**2
    return {
        "record_id": record_id,
        "model_name": "sdv1",
        "timestep": 981,
        "alpha_t": alpha_t,
        "sigma_t": sigma_t,
        "snr_t": snr_t,
        "latent_dimension": 4,
        "num_loss_seeds": num_loss_seeds,
        "loss_seed": 123,
        "generation_seeds": generation_seeds,
        "num_generation_seeds": len(generation_seeds.split(",")),
        "conditional_loss": normalized_loss_mse * 4 * snr_t,
        "normalized_loss_mse": normalized_loss_mse,
        "normalized_loss_rmse": normalized_loss_rmse,
        "recovery_mse": recovery_mse,
        "recovery_rmse": recovery_rmse,
        "mean_target_sscd": mean_target_sscd,
        "status": status,
        "error": error,
    }


def _offset_tuples(collection: PathCollection) -> set[tuple[float, float]]:
    offsets = np.asarray(collection.get_offsets(), dtype=float)
    if offsets.size == 0:
        return set()
    return {(float(x_value), float(y_value)) for x_value, y_value in offsets}


def test_plot_reloads_csv_and_draws_pair_scatter_and_binned_median(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_rows = [
        _row(
            f"sdv1-{index:04d}",
            normalized_loss_rmse=float(index + 1),
            recovery_rmse=1.0 / float(index + 1),
            mean_target_sscd=0.1 + 0.8 * index / 19.0,
        )
        for index in range(20)
    ]
    excluded_rows = [
        _row(
            "sdv1-error",
            normalized_loss_rmse=3.0,
            recovery_rmse=0.3,
            mean_target_sscd=0.5,
            status="error",
            error="synthetic failure",
        ),
        _row(
            "sdv1-zero-x",
            normalized_loss_rmse=0.0,
            recovery_rmse=0.2,
            mean_target_sscd=0.5,
        ),
        _row(
            "sdv1-nan-y",
            normalized_loss_rmse=2.0,
            recovery_rmse=math.nan,
            mean_target_sscd=0.5,
        ),
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    figure_path = tmp_path / experiment.FIGURE_NAME
    pd.DataFrame(
        [*excluded_rows, *reversed(valid_rows)], columns=EXPECTED_COLUMNS
    ).to_csv(csv_path, index=False)

    real_read_csv = pd.read_csv
    reads: list[Path] = []

    def read_csv_spy(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(experiment.pd, "read_csv", read_csv_spy)
    real_subplots = experiment.plt.subplots
    captured: dict[str, object] = {}

    def subplots_spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        figure, axis = real_subplots(*args, **kwargs)
        real_savefig = figure.savefig
        real_tight_layout = figure.tight_layout

        def savefig_spy(*save_args: object, **save_kwargs: object) -> object:
            captured["savefig_kwargs"] = dict(save_kwargs)
            return real_savefig(*save_args, **save_kwargs)

        def tight_layout_spy(*layout_args: object, **layout_kwargs: object) -> object:
            captured["tight_layout_calls"] = int(
                captured.get("tight_layout_calls", 0)
            ) + 1
            return real_tight_layout(*layout_args, **layout_kwargs)

        monkeypatch.setattr(figure, "savefig", savefig_spy)
        monkeypatch.setattr(figure, "tight_layout", tight_layout_spy)
        captured.update(
            figure=figure,
            axis=axis,
            font_family=tuple(experiment.matplotlib.rcParams["font.family"]),
            mathtext_fontset=experiment.matplotlib.rcParams["mathtext.fontset"],
            default_font_size=experiment.matplotlib.rcParams["font.size"],
        )
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", subplots_spy)
    experiment.plot_saved_results(csv_path, figure_path)

    assert reads == [csv_path]
    assert figure_path.is_file()
    figure = captured["figure"]
    axis = captured["axis"]
    np.testing.assert_allclose(figure.get_size_inches(), [4.0, 4.0])
    assert captured["font_family"] == ("STIXGeneral",)
    assert captured["mathtext_fontset"] == "stix"
    assert captured["default_font_size"] == 15
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert axis.get_title() == ""
    assert axis.xaxis.label.get_fontsize() == 15
    assert axis.yaxis.label.get_fontsize() == 15
    assert axis.xaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
    assert all(label.get_fontsize() == 12 for label in axis.get_yticklabels())

    pair_collections = [
        collection
        for collection in axis.collections
        if isinstance(collection, PathCollection)
        and np.asarray(collection.get_offsets()).shape[0] == 20
    ]
    assert len(pair_collections) == 1
    scatter = pair_collections[0]
    assert scatter.cmap.name == "viridis"
    assert scatter.norm.vmin == pytest.approx(0.0)
    assert scatter.norm.vmax == pytest.approx(1.0)
    assert scatter.norm.clip is True
    assert scatter.get_alpha() == pytest.approx(experiment.SCATTER_ALPHA)
    assert scatter.get_alpha() != experiment.COLORBAR_ALPHA
    assert scatter.get_array() is not None
    np.testing.assert_allclose(
        np.sort(np.asarray(scatter.get_array(), dtype=float)),
        np.linspace(0.1, 0.9, 20),
    )
    observed_points = _offset_tuples(scatter)
    expected_points = {(float(index), 1.0 / index) for index in range(1, 21)}
    assert len(observed_points) == len(expected_points)
    for expected in expected_points:
        assert any(np.allclose(expected, observed) for observed in observed_points)

    trend = [line for line in axis.lines if line.get_label() == "Binned median"]
    assert len(trend) == 1
    trend_line = trend[0]
    assert str(trend_line.get_color()).lower() in {"black", "k", "#000000"}
    assert trend_line.get_marker() == "o"
    expected_x = np.arange(1.5, 20.0, 2.0)
    expected_y = np.asarray(
        [(1.0 / left + 1.0 / (left + 1)) / 2.0 for left in range(1, 21, 2)]
    )
    np.testing.assert_allclose(trend_line.get_xdata(), expected_x)
    np.testing.assert_allclose(trend_line.get_ydata(), expected_y)
    assert len(axis.lines) == 1

    assert len(axis.texts) == 0
    assert axis.get_xlabel() == (
        r"$\sqrt{\mathcal{L}^{\star}_{T,c} / "
        r"[d\,(\alpha_T^2/\sigma_T^2)]}$"
    )
    assert r"\frac" not in axis.get_xlabel()
    assert "SNR" not in axis.get_xlabel()
    assert axis.get_ylabel() == (
        r"$\sqrt{\mathbb{E}_{\mathbf{x}_T\sim"
        r"\mathcal{N}(\mathbf{0},\mathbf{I})}"
        r"[\|\hat{\mathbf{x}}_{0\mid T,c}-\mathbf{x}^{\star}\|_2^2/d]}$"
    )
    assert "recovery_rmse" not in axis.get_ylabel()

    legend = axis.get_legend()
    assert legend is not None
    assert all(item.get_fontsize() == 10 for item in legend.get_texts())

    colorbar_axes = [plot_axis for plot_axis in figure.axes if plot_axis is not axis]
    assert len(colorbar_axes) == 1
    colorbar_axis = colorbar_axes[0]
    assert colorbar_axis.get_ylabel() == "SSCD"
    np.testing.assert_allclose(colorbar_axis.get_ylim(), [0.0, 1.0])
    assert colorbar_axis.yaxis.label.get_fontsize() == 15
    assert colorbar_axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert all(label.get_fontsize() == 12 for label in colorbar_axis.get_yticklabels())
    meshes = [
        collection
        for collection in colorbar_axis.collections
        if isinstance(collection, QuadMesh)
    ]
    assert len(meshes) == 1
    colorbar_mesh = meshes[0]
    assert colorbar_mesh.get_alpha() == pytest.approx(1.0)
    figure.canvas.draw()
    np.testing.assert_allclose(colorbar_mesh.get_facecolors()[:, 3], 1.0)
    assert captured["tight_layout_calls"] == 1
    assert captured["savefig_kwargs"] == {
        "format": "pdf",
        "bbox_inches": "tight",
        "pad_inches": 0.05,
    }


def test_plot_reduces_bins_and_warns_when_two_bins_are_impossible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / experiment.CSV_NAME
    figure_path = tmp_path / experiment.FIGURE_NAME
    pd.DataFrame(
        [
            _row(
                "sdv1-0000",
                normalized_loss_rmse=1.0,
                recovery_rmse=2.0,
                mean_target_sscd=0.3,
            )
        ],
        columns=EXPECTED_COLUMNS,
    ).to_csv(csv_path, index=False)

    real_subplots = experiment.plt.subplots
    captured: dict[str, object] = {}

    def subplots_spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        figure, axis = real_subplots(*args, **kwargs)
        captured.update(figure=figure, axis=axis)
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", subplots_spy)
    experiment.plot_saved_results(csv_path, figure_path)

    axis = captured["axis"]
    assert not [line for line in axis.lines if line.get_label() == "Binned median"]
    captured_output = capsys.readouterr()
    assert "warning" in (captured_output.out + captured_output.err).lower()


@pytest.mark.parametrize("plot_flag", ("--plot", "--plot-only"))
def test_plot_only_uses_saved_csv_without_loading_diffusion_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plot_flag: str,
) -> None:
    output = tmp_path / "plot-only"
    output.mkdir()
    csv_path = output / experiment.CSV_NAME
    pd.DataFrame(
        [
            _row(
                "sdv1-0000",
                normalized_loss_rmse=1.0,
                recovery_rmse=2.0,
                mean_target_sscd=0.3,
            ),
            _row(
                "sdv1-0001",
                normalized_loss_rmse=2.0,
                recovery_rmse=1.0,
                mean_target_sscd=0.6,
            ),
        ],
        columns=EXPECTED_COLUMNS,
    ).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("plot-only must not load or run the diffusion model")

    monkeypatch.setattr(experiment, "load_model_components", forbidden, raising=False)
    monkeypatch.setattr(
        experiment, "preflight_model_components", forbidden, raising=False
    )
    monkeypatch.setattr(experiment, "WebsterDataset", forbidden, raising=False)
    monkeypatch.setattr(experiment, "run_experiment", forbidden)
    assert experiment.main([plot_flag, "--output-dir", str(output)]) == 0
    assert csv_path.read_bytes() == original_csv
    assert {path.name for path in output.iterdir()} == {
        experiment.CSV_NAME,
        experiment.FIGURE_NAME,
    }


def test_source_has_no_previous_experiment_or_reverse_trajectory_logic() -> None:
    source_path = Path(experiment.__file__)
    source = source_path.read_text(encoding="utf-8")
    lowered = source.lower()
    tree = ast.parse(source, filename=str(source_path))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    for forbidden in (
        "theorem1_finite_t_recovery",
        "theorem1_conditional_recovery",
        "failure_probability",
        "theorem_bound",
        "markov_term",
        "pinsker",
        "noise_level_index",
        "recovery_rmse_threshold",
        "num_noise_levels",
        "num_generation_samples",
        "num_loss_samples",
        "seedwise",
        "95th percentile",
        "np.percentile",
    ):
        assert forbidden not in lowered
    assert "step" not in called_attributes
    assert "fill_between" not in called_attributes
    assert "axline" not in called_attributes
    assert "polyfit" not in called_attributes
    assert "linregress" not in called_names
    assert not called_names.intersection(
        {"sample_trajectory", "run_batched_sampling", "sample_diffusion"}
    )
    assert ".png" not in lowered
    assert ".svg" not in lowered
    assert "generation_seed" not in experiment.CSV_COLUMNS
    assert "seed_sscd" not in experiment.CSV_COLUMNS
