"""Focused model-free contracts for the pair-level Theorem 1 experiment."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import (
    LineCollection,
    PathCollection,
    QuadMesh,
)
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import theorem1_loss_recovery as experiment


EXPECTED_COLUMNS = (
    "record_id",
    "selection_strategy",
    "selection_hash",
    "model_name",
    "scheduler_name",
    "guidance_scale",
    "num_inference_steps",
    "evaluation_generation_scientific_config_hash",
    "evaluation_schedule_sha256",
    "evaluation_source",
    "step_index",
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
    "trajectory_sha256",
    "status",
    "error",
)
SELECTION_HASH = "c" * 64


def _assert_primary_median(
    axis: object,
    *,
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> None:
    assert len(axis.lines) == 1
    median = axis.lines[0]
    order = np.argsort(np.asarray(x_values, dtype=float), kind="stable")
    sorted_x = np.asarray(x_values, dtype=float)[order]
    sorted_y = np.asarray(y_values, dtype=float)[order]
    bins = np.array_split(np.arange(len(sorted_x)), min(10, len(sorted_x)))
    expected_x = np.asarray([np.median(sorted_x[index]) for index in bins])
    expected_y = np.asarray([np.median(sorted_y[index]) for index in bins])
    np.testing.assert_allclose(median.get_xdata(), expected_x)
    np.testing.assert_allclose(median.get_ydata(), expected_y)
    assert median.get_color() == "black"
    assert median.get_linestyle() == "-"
    assert median.get_linewidth() == pytest.approx(1.35)
    assert matplotlib.colors.to_rgba(median.get_color(), median.get_alpha())[
        -1
    ] == pytest.approx(1.0)
    assert median.get_zorder() == 3
    assert median.get_label() == "Median"
    legend = axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Median"]
    assert all(text.get_fontsize() == 10 for text in legend.get_texts())
    assert legend.get_frame_on() is False
    assert legend._loc == 4  # Lower right.


def _assert_shared_scientific_axes(axis: object) -> None:
    for coordinate in (axis.xaxis, axis.yaxis):
        formatter = coordinate.get_major_formatter()
        assert isinstance(formatter, experiment._SharedScientificFormatter)
        formatter.set_locs(coordinate.get_majorticklocs())
        expected_offset = (
            rf"$\times 10^{{{formatter.exponent}}}$" if formatter.exponent else ""
        )
        assert formatter.get_offset() == expected_offset
        assert coordinate.get_offset_text().get_fontsize() == 10
        assert experiment.AXIS_MULTIPLIER_FONT_SIZE == 10
        assert isinstance(coordinate.get_minor_formatter(), NullFormatter)
        assert all(label.get_text() == "" for label in coordinate.get_minorticklabels())
        for label in coordinate.get_ticklabels():
            assert r"\times" not in label.get_text()
            assert "^{" not in label.get_text()
    assert not axis.texts


def _parameters(timesteps: Sequence[int]) -> tuple[experiment.TerminalParameters, ...]:
    return tuple(
        experiment.TerminalParameters(
            timestep=int(t),
            alpha_t=math.sqrt(0.36 + 0.01 * k),
            sigma_t=math.sqrt(0.64 - 0.01 * k),
            snr_t=(0.36 + 0.01 * k) / (0.64 - 0.01 * k),
        )
        for k, t in enumerate(timesteps)
    )


def _selection(
    *,
    included: Sequence[str],
    excluded: Sequence[str] = (),
    strategy: str = "gmm",
    selection_hash: str = SELECTION_HASH,
) -> SimpleNamespace:
    indices = tuple(dict.fromkeys((*included, *excluded)))
    return SimpleNamespace(
        included_indices=frozenset(included),
        excluded_indices=frozenset(excluded),
        selection_strategy=strategy,
        sha256=selection_hash,
        prompt_frame=pd.DataFrame(
            [
                {
                    "original_index": index,
                    "record_id": index,
                    "source_row_number": position,
                    "prompt": f"prompt {index}",
                    "target_image_sha256": f"{position + 1:x}" * 64,
                }
                for position, index in enumerate(indices)
            ]
        ),
    )


def _prediction_samples(
    args: Sequence[object], kwargs: Mapping[str, object]
) -> torch.Tensor:
    assert not args
    value = kwargs["samples"]
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
        "--selection-strategy",
        "--scheduler",
        "--g",
        "--T",
        "--num-loss-seeds",
        "--evaluation-source",
        "--loss-seed",
        "--N",
        "--sample-batch-size",
        "--max-records",
        "--output-dir",
        "--device",
        "--plot",
        "--overwrite",
    }

    arguments = parser.parse_args([])
    assert arguments.model == "sdv1"
    assert arguments.selection_strategy == "gmm"
    assert arguments.evaluation_source == "both"
    assert arguments.scheduler == "ddim"
    assert arguments.g == pytest.approx(7.5)
    assert arguments.num_inference_steps == 50
    assert arguments.num_loss_seeds == 20
    assert arguments.loss_seed == 0
    assert arguments.num_seeds == 20
    assert arguments.sample_batch_size > 0
    assert arguments.max_records is None
    assert arguments.output_dir is None
    assert arguments.device == "auto"
    assert arguments.plot is False
    assert arguments.overwrite is False
    assert parser.parse_args(["--plot"]).plot is True
    assert parser.parse_args(["--overwrite"]).overwrite is True
    assert (
        parser.parse_args(["--selection-strategy", "gmm"]).selection_strategy == "gmm"
    )
    assert parser.parse_args(["--device", "CUDA:2"]).device == "cuda:2"
    expected_output_directories = {
        model_name: (
            experiment.ROOT
            / "outputs"
            / f"{model_name}_ddim_g7.5_T50_N20"
            / "theorem1_loss_recovery"
            / SELECTION_HASH
            / "evaluation_both"
        )
        for model_name in ("sdv1", "sdv2", "realvis")
    }
    assert {
        model_name: experiment._default_output_directory(
            model_name, "ddim", 7.5, 50, 20, "gmm", SELECTION_HASH, "both"
        )
        for model_name in expected_output_directories
    } == expected_output_directories
    assert len(set(expected_output_directories.values())) == 3
    assert experiment._default_output_directory(
        "sdv1", "ddpm", 3.25, 12, 3, "gmm", "d" * 64, "trajectory"
    ) == (
        experiment.ROOT
        / "outputs"
        / "sdv1_ddpm_g3.25_T12_N3"
        / "theorem1_loss_recovery"
        / ("d" * 64)
        / "evaluation_trajectory"
    )
    assert experiment._default_output_directory(
        "sdv1", "ddpm", 3.25, 12, 3, "gmm", "e" * 64, "gaussian"
    ) == (
        experiment.ROOT
        / "outputs"
        / "sdv1_ddpm_g3.25_T12_N3"
        / "theorem1_loss_recovery"
        / ("e" * 64)
        / "evaluation_gaussian"
    )

    smoke = parser.parse_args(
        [
            "--max-records",
            "2",
            "--scheduler",
            "ddpm",
            "--g",
            "3.25",
            "--T",
            "12",
            "--num-loss-seeds",
            "2",
            "--N",
            "2",
        ]
    )
    assert (smoke.scheduler, smoke.g, smoke.num_inference_steps) == (
        "ddpm",
        pytest.approx(3.25),
        12,
    )
    assert (
        smoke.max_records,
        smoke.num_loss_seeds,
        smoke.num_seeds,
    ) == (2, 2, 2)
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-loss-seeds", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--scheduler", "euler"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--g", "nan"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--T", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss-seed", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--N", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-loss", "2"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--plot-only"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "cuda:-1"])
    for unsupported in ("spearman", "gmm-evidence", "all", "kmeans"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--selection-strategy", unsupported])

    assert experiment.CSV_COLUMNS == EXPECTED_COLUMNS
    assert experiment.CSV_NAME == "theorem1_loss_recovery.csv"
    assert experiment.FIGURE_FILENAMES == (
        "theorem1_loss_recovery.png",
        "theorem1_loss_recovery.pdf",
        "theorem1_loss_recovery_trajectory.png",
        "theorem1_loss_recovery_trajectory.pdf",
        "theorem1_loss_vs_timestep.png",
        "theorem1_loss_vs_timestep.pdf",
    )
    assert experiment.LOSS_TIMESTEP_FIGURE_FILENAMES == (
        "theorem1_loss_vs_timestep.png",
        "theorem1_loss_vs_timestep.pdf",
    )


def test_run_experiment_requires_explicit_device() -> None:
    assert (
        inspect.signature(experiment.run_experiment).parameters["device"].default
        is inspect.Parameter.empty
    )


def test_output_directory_accepts_only_the_csv_and_both_figure_formats(
    tmp_path: Path,
) -> None:
    output = tmp_path / "theorem1"
    output.mkdir()
    for filename in (experiment.CSV_NAME, *experiment.FIGURE_FILENAMES):
        (output / filename).write_bytes(b"existing artifact")

    assert experiment._prepare_output_directory(output) == output.resolve()

    unexpected = output / "unrelated.log"
    unexpected.write_text("unrelated", encoding="utf-8")
    with pytest.raises(
        experiment.ExperimentError,
        match="output directory contains unexpected artifacts: unrelated.log",
    ):
        experiment._prepare_output_directory(output)

    invalid_output = tmp_path / "invalid-theorem1"
    invalid_output.mkdir()
    invalid_name = invalid_output / experiment.FIGURE_FILENAMES[0]
    invalid_name.mkdir()
    with pytest.raises(
        experiment.ExperimentError,
        match=(
            "output directory contains unexpected artifacts: theorem1_loss_recovery.png"
        ),
    ):
        experiment._prepare_output_directory(invalid_output)


def test_output_preparation_removes_only_retired_noise_sweep_figures(
    tmp_path: Path,
) -> None:
    current = [
        tmp_path / name for name in (experiment.CSV_NAME, *experiment.FIGURE_FILENAMES)
    ]
    retired = [
        tmp_path / "theorem1_loss_recovery_noise_sweep.png",
        tmp_path / "theorem1_loss_recovery_noise_sweep.pdf",
    ]
    for path in (*current, *retired):
        path.write_bytes(b"existing artifact")
    assert experiment._prepare_output_directory(tmp_path) == tmp_path.resolve()
    assert all(not path.exists() for path in retired)
    assert all(path.read_bytes() == b"existing artifact" for path in current)
    assert set(tmp_path.iterdir()) == set(current)


@pytest.mark.parametrize("entrypoint", ["prepare", "cleanup"])
@pytest.mark.parametrize("unsafe_kind", ["directory", "symlink"])
def test_retired_figure_cleanup_validates_all_targets_before_removing_any(
    tmp_path: Path, entrypoint: str, unsafe_kind: str
) -> None:
    output = tmp_path / "figures"
    output.mkdir()
    retired_png = output / "theorem1_loss_recovery_noise_sweep.png"
    retired_png.write_bytes(b"preserve until all targets are validated")
    retired_pdf = output / "theorem1_loss_recovery_noise_sweep.pdf"
    target = tmp_path / "unrelated.pdf"
    target.write_bytes(b"unrelated")
    if unsafe_kind == "directory":
        retired_pdf.mkdir()
    else:
        retired_pdf.symlink_to(target)
    function = (
        experiment._prepare_output_directory
        if entrypoint == "prepare"
        else experiment._remove_noise_sweep_outputs
    )
    with pytest.raises(experiment.ExperimentError):
        function(output)
    assert retired_png.read_bytes() == b"preserve until all targets are validated"
    assert retired_pdf.exists()
    assert target.read_bytes() == b"unrelated"


def test_unknown_output_artifact_prevents_retired_figure_cleanup(
    tmp_path: Path,
) -> None:
    retired = tmp_path / "theorem1_loss_recovery_noise_sweep.png"
    retired.write_bytes(b"old figure")
    unrelated = tmp_path / "theorem1_loss_recovery_noise_sweep_notes.txt"
    unrelated.write_bytes(b"preserve")
    with pytest.raises(experiment.ExperimentError, match="unexpected artifacts"):
        experiment._prepare_output_directory(tmp_path)
    assert retired.read_bytes() == b"old figure"
    assert unrelated.read_bytes() == b"preserve"


def test_nondefault_generation_contract_uses_exact_sampler_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule_path = tmp_path / "schedule.pt"
    schedule_path.write_bytes(b"cached schedule")
    paths = SimpleNamespace(
        schedule=schedule_path,
        run_directory=tmp_path / "logs/sdv1_ddpm_g3.25_T12_N3/experiment_S0_N3",
    )
    path_requests: list[tuple[Path, dict[str, object]]] = []

    def fake_generation_paths(root: Path, **kwargs: object) -> SimpleNamespace:
        path_requests.append((root, dict(kwargs)))
        return paths

    scheduler_config = {
        "_class_name": "DDPMScheduler",
        "prediction_type": "v_prediction",
    }
    science = {
        "model_cli_name": "sdv1",
        "dataset_model": "sdv1",
        "model_id": "synthetic-model",
        "guidance_scale": 3.25,
        "num_inference_steps": 12,
        "num_seeds": 3,
        "seeds": [0, 1, 2],
        "trajectory_order": "noise_to_image",
        "stored_prediction_type": "epsilon",
        "target_latent_definition": experiment.TARGET_LATENT_DEFINITION,
        "target_preprocessing": {"policy": "test"},
        "latent_shape": [4, 8, 8],
        "scientific_tensor_storage": {
            "dtype": "float16",
            "device": "cpu",
            "layout": "contiguous",
        },
        "scheduler": {"name": "ddpm", "config": scheduler_config},
    }
    schedule_payload = {
        **{
            name: torch.tensor(
                [getattr(p, name) for p in _parameters(range(11, -1, -1))],
                dtype=torch.float32,
            )
            for name in ("alpha_t", "sigma_t")
        },
        "alphas_cumprod_t": torch.tensor(
            [p.alpha_t**2 for p in _parameters(range(11, -1, -1))], dtype=torch.float32
        ),
        "timesteps": torch.arange(11, -1, -1, dtype=torch.int64),
        "init_noise_sigma": 1.0,
        "scheduler_config": scheduler_config,
    }
    sscd_requests: list[tuple[object, Mapping[str, object], str, tuple[int, ...]]] = []

    def fake_sscd_configuration(
        sscd_paths: object,
        *,
        science: Mapping[str, object],
        scientific_hash: str,
        seeds: Sequence[int],
    ) -> tuple[None, None, str]:
        sscd_requests.append(
            (sscd_paths, science, scientific_hash, tuple(int(seed) for seed in seeds))
        )
        return None, None, "synthetic missing SSCD"

    monkeypatch.setattr(experiment, "generation_paths", fake_generation_paths)
    monkeypatch.setattr(
        experiment,
        "require_generation_run",
        lambda observed: (
            {
                "scientific_config": science,
                "scientific_config_hash": "science-hash",
            }
            if observed is paths
            else None
        ),
    )
    monkeypatch.setattr(
        experiment,
        "get_model_spec",
        lambda _name: SimpleNamespace(
            dataset_model="sdv1",
            model_id="synthetic-model",
            resolution=64,
        ),
    )
    monkeypatch.setattr(
        experiment,
        "target_preprocessing_policy",
        lambda _resolution: {"policy": "test"},
    )
    monkeypatch.setattr(
        experiment,
        "safe_torch_load",
        lambda observed: (
            schedule_payload
            if observed == schedule_path
            else pytest.fail(f"unexpected tensor load: {observed}")
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_load_sscd_configuration",
        fake_sscd_configuration,
    )

    contract = experiment._load_generation_contract(
        tmp_path,
        "sdv1",
        "ddpm",
        3.25,
        12,
        3,
    )

    assert path_requests == [
        (
            tmp_path.resolve(),
            {
                "model_name": "sdv1",
                "scheduler_name": "ddpm",
                "guidance_scale": 3.25,
                "num_inference_steps": 12,
                "num_seeds": 3,
                "seed_start": 0,
            },
        )
    ]
    assert contract.paths is paths
    assert contract.seeds == (0, 1, 2)
    assert contract.stored_dtype is torch.float16
    assert contract.scheduler_config == scheduler_config
    assert contract.sscd_error == "synthetic missing SSCD"
    assert len(sscd_requests) == 1
    assert sscd_requests[0][1:] == (
        science,
        "science-hash",
        (0, 1, 2),
    )


def test_mean_sscd_does_not_depend_on_preview_sensitive_generation_marker_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = tmp_path / "7.json"
    score_path = tmp_path / "7.pt"
    marker_path.write_text("{}", encoding="utf-8")
    score_path.write_bytes(b"fixed score tensor")
    configuration_hash = "b" * 64
    scientific_hash = "a" * 64
    latent_hash = "d" * 64
    score_hash = "e" * 64
    metadata = {
        "original_index": "7",
        "record_id": "sdv1-0007",
        "source_row_number": 8,
        "prompt_raw": "selected prompt",
        "target_image_sha256": "f" * 64,
    }
    marker = {
        **metadata,
        "model_id": "model",
        "model_revision": "revision",
        "num_seeds": 2,
        "seeds": [0, 1],
        "similarity": experiment.SCORE_DEFINITION,
        "sscd_configuration_hash": configuration_hash,
        "generation_scientific_config_hash": scientific_hash,
        "generation_latent_sha256": latent_hash,
        "generation_record_sha256": "outdated-preview-audit-hash",
        "score_sha256": score_hash,
    }
    contract = SimpleNamespace(
        sscd_error=None,
        sscd_configuration={"seeds": [0, 1]},
        sscd_configuration_hash=configuration_hash,
        scientific_hash=scientific_hash,
        science={"model_id": "model", "model_revision": "revision"},
        sscd_paths=SimpleNamespace(
            marker_path=lambda index: (
                marker_path if index == "7" else pytest.fail("wrong marker index")
            ),
            score_path=lambda index: (
                score_path if index == "7" else pytest.fail("wrong score index")
            ),
        ),
        paths=SimpleNamespace(
            record_path=lambda _index: pytest.fail(
                "preview-sensitive generation record must not be hashed"
            )
        ),
    )
    monkeypatch.setattr(experiment, "read_json", lambda path: marker)
    monkeypatch.setattr(
        experiment,
        "file_sha256",
        lambda path: (
            score_hash
            if path == score_path
            else pytest.fail(f"unexpected hash request: {path}")
        ),
    )
    monkeypatch.setattr(
        experiment,
        "safe_torch_load",
        lambda path: (
            torch.tensor([0.2, 0.6], dtype=torch.float32)
            if path == score_path
            else pytest.fail(f"unexpected tensor load: {path}")
        ),
    )

    result = experiment._load_mean_target_sscd(
        contract=contract,
        metadata=metadata,
        generation_seeds=(0, 1),
        generation_marker={"tensor_file_sha256": {"latent": latent_hash}},
    )

    assert result == pytest.approx(0.4)


def test_active_scheduler_uses_requested_name_and_inference_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_timesteps = torch.arange(11, -1, -1, dtype=torch.int64)
    timestep_calls: list[tuple[int, torch.device]] = []

    def set_timesteps(count: int, *, device: torch.device) -> None:
        timestep_calls.append((count, device))
        scheduler.timesteps = expected_timesteps.to(device=device)

    scheduler = SimpleNamespace(
        timesteps=torch.empty(0, dtype=torch.int64),
        alphas_cumprod=torch.tensor(
            [p.alpha_t**2 for p in reversed(_parameters(expected_timesteps))],
            dtype=torch.float64,
        ),
        init_noise_sigma=1.0,
        set_timesteps=set_timesteps,
    )
    scheduler_config = {
        "_class_name": "DDPMScheduler",
        "prediction_type": "sample",
    }
    original_scheduler = object()
    build_requests: list[tuple[object, str]] = []

    def fake_build_scheduler(original: object, name: str) -> SimpleNamespace:
        build_requests.append((original, name))
        return SimpleNamespace(scheduler=scheduler, config=scheduler_config)

    monkeypatch.setattr(experiment, "build_scheduler", fake_build_scheduler)
    contract = SimpleNamespace(
        schedule_payload={"timesteps": expected_timesteps},
        parameters=_parameters(expected_timesteps),
        scheduler_config=scheduler_config,
        init_noise_sigma=1.0,
    )
    components = SimpleNamespace(
        original_scheduler=original_scheduler,
        device=torch.device("cpu"),
    )

    active = experiment._validate_active_scheduler(
        components,
        contract,
        "ddpm",
        12,
    )

    assert active is scheduler
    assert build_requests == [(original_scheduler, "ddpm")]
    assert timestep_calls == [(12, torch.device("cpu"))]


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
    assert result.conditional_loss == pytest.approx(7.0)
    assert result.normalized_loss_mse == pytest.approx(expected_mse)
    assert result.normalized_loss_rmse == pytest.approx(expected_rmse)
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
    assert result.recovery_mse == pytest.approx(expected_mse)
    assert result.recovery_rmse == pytest.approx(expected_rmse)
    assert expected_rmse != pytest.approx(float(torch.sqrt(per_seed_mse).mean()))


def test_cuda_oom_halving_preserves_loss_and_recovery_sample_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.zeros((1, 1, 1), dtype=torch.float32)
    samples = torch.arange(1, 6, dtype=torch.float32).reshape(5, 1, 1, 1)
    components = SimpleNamespace(
        device=torch.device("cuda:0"),
        inference_dtype=torch.float32,
    )
    attempted_batches: list[torch.Tensor] = []
    successful_samples: list[torch.Tensor] = []
    cache_releases: list[None] = []

    def oom_above_two(
        conceptual_samples: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        attempted_batches.append(conceptual_samples.detach().clone())
        if conceptual_samples.shape[0] > 2:
            raise RuntimeError("synthetic CUDA out of memory")
        successful_samples.append(conceptual_samples.detach().clone())
        return torch.zeros_like(conceptual_samples)

    monkeypatch.setattr(experiment, "_prediction_batch", oom_above_two)
    monkeypatch.setattr(
        experiment.torch.cuda,
        "empty_cache",
        lambda: cache_releases.append(None),
    )

    loss = experiment._measure_conditional_loss(
        target_latent=target,
        loss_noise=samples,
        condition=torch.ones((1, 1, 1), dtype=torch.float32),
        timestep=876,
        alpha_t=0.6,
        sigma_t=0.8,
        snr_t=0.5625,
        components=components,
        scheduler=object(),
        sample_batch_size=4,
    )

    assert [batch.shape[0] for batch in attempted_batches] == [4, 2, 2, 1]
    torch.testing.assert_close(
        torch.cat(successful_samples),
        0.8 * samples,
    )
    assert loss.conditional_loss == pytest.approx(11.0)
    assert loss.normalized_loss_mse == pytest.approx(11.0 / 0.5625)
    assert loss.normalized_loss_rmse == pytest.approx(math.sqrt(11.0 / 0.5625))

    attempted_batches.clear()
    successful_samples.clear()
    recovery = experiment._measure_recovery(
        target_latent=target,
        generation_initial_latents=samples,
        condition=torch.ones((1, 1, 1), dtype=torch.float32),
        timestep=876,
        alpha_t=0.6,
        sigma_t=0.8,
        components=components,
        scheduler=object(),
        sample_batch_size=4,
    )

    assert [batch.shape[0] for batch in attempted_batches] == [4, 2, 2, 1]
    torch.testing.assert_close(torch.cat(successful_samples), samples)
    assert recovery.recovery_mse == pytest.approx(11.0 / 0.36)
    assert recovery.recovery_rmse == pytest.approx(math.sqrt(11.0 / 0.36))
    assert len(cache_releases) == 2


def test_non_cuda_out_of_memory_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def fail(samples: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        attempts.append(samples.shape[0])
        raise RuntimeError("synthetic out of memory")

    monkeypatch.setattr(experiment, "_prediction_batch", fail)
    monkeypatch.setattr(
        experiment.torch.cuda,
        "empty_cache",
        lambda: pytest.fail("CPU failure must not touch the CUDA allocator"),
    )
    components = SimpleNamespace(device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="synthetic out of memory"):
        tuple(
            experiment._prediction_batches(
                torch.zeros((3, 1, 1, 1)),
                condition=torch.zeros((1, 1, 1)),
                timestep=1,
                components=components,
                scheduler=object(),
                initial_batch_size=3,
            )
        )
    assert attempts == [3]


def test_unused_vae_is_offloaded_after_validation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfers: list[dict[str, object]] = []
    cache_releases: list[None] = []

    class FakeVAE:
        def to(self, **kwargs: object) -> FakeVAE:
            transfers.append(dict(kwargs))
            return self

    components = SimpleNamespace(
        device=torch.device("cuda:2"),
        vae=FakeVAE(),
    )
    monkeypatch.setattr(
        experiment.torch.cuda,
        "empty_cache",
        lambda: cache_releases.append(None),
    )

    experiment._offload_unused_vae(components)

    assert transfers == [{"device": torch.device("cpu"), "dtype": torch.float32}]
    assert cache_releases == [None]


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
            "record_id": "realisticvision-discarded",
            "model_name": "realisticvision",
            "prompt": "discarded prompt",
            "prompt_raw": "discarded prompt",
            "image": object(),
            "original_index": "101",
            "source_row_number": 11,
            "target_image_sha256": "b" * 64,
        },
        {
            "record_id": "realisticvision-0001",
            "model_name": "realisticvision",
            "prompt": "third prompt",
            "prompt_raw": "third prompt",
            "image": object(),
            "original_index": "102",
            "source_row_number": 12,
            "target_image_sha256": "c" * 64,
        },
    ]
    accessed_dataset_positions: list[int] = []

    class FakeDataset:
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, object]:
            accessed_dataset_positions.append(index)
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
        device_metadata={},
    )
    generation_contract = SimpleNamespace(
        parameters=_parameters((3, 2, 1)),
        scientific_hash="a" * 64,
        schedule_sha256="b" * 64,
        sscd_configuration_hash="e" * 64,
        latent_shape=(1, 2, 2),
        stored_dtype=torch.float32,
        init_noise_sigma=1.0,
        seeds=tuple(range(3)),
        scheduler_config={},
        science={
            "model_id": "SG161222/Realistic_Vision_V2.0",
            "model_revision": "a" * 40,
            "vae_id": "stabilityai/sd-vae-ft-mse",
            "vae_revision": "b" * 40,
        },
        paths=SimpleNamespace(run_directory=tmp_path / "generation-realvis"),
        sscd_paths=SimpleNamespace(),
    )

    requested_dataset_models: list[str] = []

    def fake_dataset(_root: Path, dataset_model: str, **_kwargs: object) -> FakeDataset:
        requested_dataset_models.append(dataset_model)
        return FakeDataset()

    monkeypatch.setattr(experiment, "WebsterDataset", fake_dataset)
    selection_requests: list[dict[str, object]] = []

    def load_selection(_root: Path, **kwargs: object) -> SimpleNamespace:
        selection_requests.append(dict(kwargs))
        return _selection(
            included=("100", "102"),
            excluded=("101",),
            strategy="gmm",
        )

    monkeypatch.setattr(experiment, "_load_frozen_selection", load_selection)
    loaded_contract_arguments: list[tuple[object, ...]] = []

    def load_contract(*args: object, **_kwargs: object) -> object:
        loaded_contract_arguments.append(args)
        return generation_contract

    monkeypatch.setattr(experiment, "_load_generation_contract", load_contract)
    monkeypatch.setattr(
        experiment, "_pair_cache_fingerprint", lambda *_args, **_kwargs: None
    )
    active_scheduler_arguments: list[tuple[object, ...]] = []

    def validate_scheduler(*args: object, **_kwargs: object) -> object:
        active_scheduler_arguments.append(args)
        return scheduler

    monkeypatch.setattr(experiment, "_validate_active_scheduler", validate_scheduler)
    loaded_runtimes: list[object] = []

    def load_components(*_args: object, **kwargs: object) -> object:
        loaded_runtimes.append(kwargs["runtime"])
        resolver = kwargs["revision_resolver"]
        assert callable(resolver)
        assert resolver("SG161222/Realistic_Vision_V2.0") == "a" * 40
        return components

    monkeypatch.setattr(experiment, "load_model_components", load_components)
    preflight_steps: list[int] = []
    monkeypatch.setattr(
        experiment,
        "preflight_model_components",
        lambda *_args, **kwargs: preflight_steps.append(
            int(kwargs["num_inference_steps"])
        ),
    )
    component_lifecycle: list[str] = []
    monkeypatch.setattr(
        experiment,
        "_validate_loaded_components",
        lambda *_args, **_kwargs: component_lifecycle.append("validated"),
    )
    monkeypatch.setattr(
        experiment,
        "_offload_unused_vae",
        lambda *_args, **_kwargs: component_lifecycle.append("offloaded"),
    )
    monkeypatch.setattr(
        experiment,
        "encode_prompt_condition",
        lambda *_args, **_kwargs: torch.ones((1, 1, 1), dtype=torch.float32),
    )

    validated_pairs: list[str] = []

    def validate_pair(
        contract: object,
        metadata: Mapping[str, object],
    ) -> Mapping[str, object]:
        assert contract is generation_contract
        record_id = str(metadata["record_id"])
        validated_pairs.append(record_id)
        return {"validated_record_id": record_id}

    monkeypatch.setattr(experiment, "_validate_generation_pair", validate_pair)

    def load_target(
        *,
        metadata: Mapping[str, object],
        contract: object,
        generation_marker: Mapping[str, object],
    ) -> torch.Tensor:
        assert contract is generation_contract
        assert generation_marker["validated_record_id"] == metadata["record_id"]
        if metadata["record_id"] == records[2]["record_id"]:
            raise RuntimeError("synthetic target failure")
        return torch.ones((1, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(experiment, "_load_target_latent", load_target)
    _FakeProgress.instances.clear()

    loss_noises: list[torch.Tensor] = []
    recovery_latents: list[torch.Tensor] = []
    requested_recovery_seeds: list[tuple[int, ...]] = []
    requested_sscd_seeds: list[tuple[int, ...]] = []
    generation_initial = torch.full((3, 1, 2, 2), 99.0, dtype=torch.float32)

    def reconstruct(
        seeds: Sequence[int], *_args: object, **_kwargs: object
    ) -> torch.Tensor:
        requested_recovery_seeds.append(tuple(int(seed) for seed in seeds))
        return generation_initial.clone()

    monkeypatch.setattr(experiment, "_make_gaussian_recovery_samples", reconstruct)

    def make_loss_noise(*_args: object, **kwargs: object) -> torch.Tensor:
        count = int(kwargs["num_loss_seeds"])
        return torch.arange(count * 4, dtype=torch.float32).reshape(count, 1, 2, 2)

    monkeypatch.setattr(experiment, "_make_loss_noise", make_loss_noise)

    def measure_loss(*_args: object, **kwargs: object) -> experiment.LossMeasurement:
        noise = kwargs["loss_noise"]
        if not isinstance(noise, torch.Tensor):
            raise AssertionError("run did not pass pair loss-noise samples")
        loss_noises.append(noise.detach().cpu().clone())
        return experiment.LossMeasurement(8.0, 2.0, math.sqrt(2.0))

    def measure_recovery(
        *_args: object, **kwargs: object
    ) -> experiment.RecoveryMeasurement:
        latents = kwargs["generation_initial_latents"]
        if not isinstance(latents, torch.Tensor):
            raise AssertionError("run did not pass generation initial latents")
        recovery_latents.append(latents.detach().cpu().clone())
        return experiment.RecoveryMeasurement(3.0, math.sqrt(3.0))

    monkeypatch.setattr(experiment, "_measure_conditional_loss", measure_loss)
    monkeypatch.setattr(experiment, "_measure_recovery", measure_recovery)

    def target_sscd(
        *,
        contract: object,
        metadata: Mapping[str, object],
        generation_seeds: Sequence[int],
        generation_marker: Mapping[str, object],
    ) -> float:
        assert contract is generation_contract
        assert generation_marker["validated_record_id"] == metadata["record_id"]
        requested_sscd_seeds.append(tuple(int(seed) for seed in generation_seeds))
        return 0.4

    monkeypatch.setattr(experiment, "_load_mean_target_sscd", target_sscd)

    plotted: list[tuple[Path, Path]] = []

    def fake_plot(csv_path: Path, output_directory: Path, **_: object) -> None:
        assert Path(csv_path).is_file()
        output_path = Path(output_directory)
        plotted.append((Path(csv_path), output_path))
        for figure_path in experiment._figure_paths(output_path):
            figure_path.write_bytes(b"synthetic figure")

    monkeypatch.setattr(experiment, "plot_saved_results", fake_plot)
    output = tmp_path / "loss-recovery"
    experiment.run_experiment(
        model_name="realvis",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=3,
        evaluation_source="gaussian",
        num_loss_seeds=2,
        loss_seed=7,
        num_seeds=3,
        selection_strategy="gmm",
        sample_batch_size=2,
        max_records=2,
        output_dir=output,
        device="cpu",
        progress_factory=_FakeProgress,
    )

    frame = pd.read_csv(output / experiment.CSV_NAME, keep_default_na=False)
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 6
    assert requested_dataset_models == ["realisticvision"]
    assert accessed_dataset_positions == [0, 2]
    assert selection_requests == [
        {
            "model_name": "realvis",
            "scheduler_name": "ddpm",
            "guidance_scale": 3.25,
            "num_inference_steps": 3,
            "num_seeds": 3,
            "selection_strategy": "gmm",
        }
    ]
    assert loaded_contract_arguments == [
        (experiment.ROOT, "realvis", "ddpm", 3.25, 3, 3)
    ]
    assert active_scheduler_arguments == [(components, generation_contract, "ddpm", 3)]
    assert preflight_steps == [3]
    assert len(loaded_runtimes) == 1
    assert loaded_runtimes[0].device == torch.device("cpu")
    assert loaded_runtimes[0].inference_dtype is torch.float32
    assert component_lifecycle == ["validated", "offloaded"]
    assert validated_pairs == [
        "realisticvision-0000",
        "realisticvision-0001",
    ]
    assert (
        frame["record_id"].tolist()
        == ["realisticvision-0000"] * 3 + ["realisticvision-0001"] * 3
    )
    assert frame["model_name"].eq("realvis").all()
    assert frame["selection_strategy"].eq("gmm").all()
    assert frame["selection_hash"].eq(SELECTION_HASH).all()
    assert frame["scheduler_name"].eq("ddpm").all()
    assert frame["guidance_scale"].eq(3.25).all()
    assert frame["num_inference_steps"].eq(3).all()
    assert frame["step_index"].tolist() == [0, 1, 2] * 2
    assert frame["timestep"].tolist() == [3, 2, 1] * 2
    assert frame["evaluation_source"].eq("gaussian").all()
    first, second = frame.iloc[0], frame.iloc[3]
    assert first["status"] == "ok" and first["error"] == ""
    assert second["status"] == "error"
    assert "synthetic target failure" in str(second["error"])
    assert int(first["timestep"]) == 3
    assert float(first["alpha_t"]) == pytest.approx(0.6)
    assert float(first["sigma_t"]) == pytest.approx(0.8)
    assert float(first["snr_t"]) == pytest.approx(0.5625)
    assert int(first["num_loss_seeds"]) == 2
    assert int(first["loss_seed"]) == 7
    assert str(first["generation_seeds"]) == "0,1,2"
    assert int(first["num_generation_seeds"]) == 3
    assert float(first["conditional_loss"]) == pytest.approx(8.0)
    assert float(first["normalized_loss_mse"]) == pytest.approx(2.0)
    assert float(first["normalized_loss_rmse"]) == pytest.approx(math.sqrt(2.0))
    assert float(first["recovery_mse"]) == pytest.approx(3.0)
    assert float(first["recovery_rmse"]) == pytest.approx(math.sqrt(3.0))
    assert float(first["mean_target_sscd"]) == pytest.approx(0.4)

    assert requested_recovery_seeds == [(0, 1, 2)]
    assert requested_sscd_seeds == [(0, 1, 2)]
    assert len(loss_noises) == len(recovery_latents) == 3
    assert all(torch.equal(value, loss_noises[0]) for value in loss_noises)
    assert all(torch.equal(value, recovery_latents[0]) for value in recovery_latents)
    assert loss_noises[0].shape == (2, 1, 2, 2)
    assert recovery_latents[0].shape == (3, 1, 2, 2)
    assert not torch.equal(loss_noises[0], recovery_latents[0])
    torch.testing.assert_close(recovery_latents[0], generation_initial)

    assert len(_FakeProgress.instances) == 1
    progress = _FakeProgress.instances[0]
    assert progress.kwargs["total"] == 6
    assert progress.kwargs["desc"] == "Theorem 1 loss–recovery"
    assert progress.kwargs["unit"] == "observation"
    assert progress.kwargs["dynamic_ncols"] is True
    assert progress.kwargs["leave"] is True
    assert progress.updated == 6
    completed_postfixes = [
        values for values in progress.postfixes if "status" in values
    ]
    assert [values["record_id"] for values in completed_postfixes] == [
        "realisticvision-0000"
    ] * 3 + ["realisticvision-0001"] * 3
    assert [values["status"] for values in completed_postfixes] == ["ok"] * 3 + [
        "error"
    ] * 3
    assert {
        values.get("phase") for values in progress.postfixes if "phase" in values
    } == {
        "loading-pair",
        "conditional-loss",
        "recovery",
    }
    log_csv = (
        experiment._default_log_directory(
            generation_contract, SELECTION_HASH, 2, 7, "gaussian"
        )
        / experiment.CSV_NAME
    )
    assert plotted == [(log_csv, output)]
    assert {path.name for path in output.iterdir()} == {
        experiment.CSV_NAME,
        *experiment.FIGURE_FILENAMES,
    }


def test_worker_setup_failure_still_writes_all_pair_rows_and_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        {"record_id": "sdv1-0000", "original_index": "10"},
        {"record_id": "sdv1-0001", "original_index": "11"},
    )

    class FakeDataset:
        def iter_metadata(self):  # type: ignore[no-untyped-def]
            yield from records

    contract = SimpleNamespace(
        sscd_configuration_hash="e" * 64,
        seeds=tuple(range(2)),
        parameters=_parameters(range(50, 0, -1)),
        scientific_hash="a" * 64,
        schedule_sha256="b" * 64,
        latent_shape=(4, 8, 8),
        science={
            "model_id": "model",
            "model_revision": "a" * 40,
            "vae_id": "vae",
            "vae_revision": "b" * 40,
        },
        paths=SimpleNamespace(run_directory=tmp_path / "generation-setup-failure"),
        sscd_paths=SimpleNamespace(),
    )
    monkeypatch.setattr(
        experiment, "WebsterDataset", lambda *_args, **_kwargs: FakeDataset()
    )
    monkeypatch.setattr(
        experiment,
        "_load_frozen_selection",
        lambda *_args, **_kwargs: _selection(included=("10", "11")),
    )
    monkeypatch.setattr(
        experiment,
        "_load_generation_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        experiment, "_pair_cache_fingerprint", lambda *_args, **_kwargs: None
    )

    def fail_model_load(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic model initialization failure")

    monkeypatch.setattr(experiment, "load_model_components", fail_model_load)
    plotted: list[tuple[Path, Path]] = []

    def fake_plot(csv_path: Path, output_directory: Path, **_: object) -> None:
        output_path = Path(output_directory)
        plotted.append((Path(csv_path), output_path))
        for figure_path in experiment._figure_paths(output_path):
            figure_path.write_bytes(b"synthetic figure")

    monkeypatch.setattr(experiment, "plot_saved_results", fake_plot)
    _FakeProgress.instances.clear()
    output = tmp_path / "setup-failure"

    exit_code = experiment.run_experiment(
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        evaluation_source="both",
        num_loss_seeds=3,
        loss_seed=17,
        num_seeds=2,
        selection_strategy="gmm",
        sample_batch_size=2,
        max_records=None,
        output_dir=output,
        device="cpu",
        progress_factory=_FakeProgress,
    )

    assert exit_code == 1
    frame = pd.read_csv(output / experiment.CSV_NAME)
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 200
    assert frame["record_id"].tolist() == ["sdv1-0000"] * 100 + ["sdv1-0001"] * 100
    assert frame["selection_strategy"].eq("gmm").all()
    assert frame["selection_hash"].eq(SELECTION_HASH).all()
    assert frame["scheduler_name"].eq("ddim").all()
    assert frame["guidance_scale"].eq(7.5).all()
    assert frame["num_inference_steps"].eq(50).all()
    assert frame["status"].eq("error").all()
    assert frame["latent_dimension"].eq(256).all()
    assert frame["generation_seeds"].astype(str).eq("0,1").all()
    assert (
        frame["timestep"].tolist()
        == [t for t in range(50, 0, -1) for _ in range(2)] * 2
    )
    assert frame["evaluation_source"].tolist() == ["gaussian", "trajectory"] * 100
    assert frame["conditional_loss"].isna().all()
    assert all(
        "synthetic model initialization failure" in message
        for message in frame["error"].astype(str)
    )
    log_csv = (
        experiment._default_log_directory(contract, SELECTION_HASH, 3, 17, "both")
        / experiment.CSV_NAME
    )
    assert plotted == [(log_csv, output)]
    assert {path.name for path in output.iterdir()} == {
        experiment.CSV_NAME,
        *experiment.FIGURE_FILENAMES,
    }
    assert len(_FakeProgress.instances) == 1
    progress = _FakeProgress.instances[0]
    assert progress.updated == 200
    assert [postfix["record_id"] for postfix in progress.postfixes] == [
        "sdv1-0000"
    ] * 100 + ["sdv1-0001"] * 100
    assert all(postfix["status"] == "error" for postfix in progress.postfixes)
    assert all(postfix["phase"] == "worker-failure" for postfix in progress.postfixes)


def test_multi_gpu_shards_whole_pairs_and_restores_metadata_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = tuple(
        (position, {"record_id": f"pair-{position}"}) for position in range(5)
    )
    devices = tuple(torch.device(f"cuda:{index}") for index in range(3))
    progress_lock = experiment.tqdm.get_lock()
    spawn_context = SimpleNamespace(RLock=lambda: progress_lock)
    dataset = object()
    contract = SimpleNamespace(parameters=_parameters(range(12, 0, -1)))
    progress_factory = object()
    observed: list[
        tuple[
            int, str, str, float, int, tuple[int, ...], tuple[int, ...], int, int, bool
        ]
    ] = []

    class ImmediateFuture:
        def __init__(self, value: object) -> None:
            self.value = value

        def result(self) -> object:
            return self.value

    class ImmediateExecutor:
        def __init__(
            self,
            *,
            max_workers: int,
            mp_context: object,
            initializer: object,
            initargs: tuple[object, ...],
        ) -> None:
            assert max_workers == 2
            assert mp_context is spawn_context
            assert initializer is experiment._install_tqdm_lock
            assert initargs == (progress_lock,)

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *_arguments: object) -> None:
            return None

        def submit(
            self, function: object, *arguments: object, **kwargs: object
        ) -> ImmediateFuture:
            assert function is fake_pair_shard
            assert not arguments
            assert "dataset" not in kwargs
            assert kwargs["contract"] is contract
            assert "progress_factory" not in kwargs
            return ImmediateFuture(fake_pair_shard(**kwargs))

    def fake_pair_shard(**kwargs: object) -> experiment._PairShardResult:
        worker_index = int(kwargs["worker_index"])
        selected = kwargs["device"]
        shard = kwargs["entries"]
        seeds = kwargs["generation_seeds"]
        assert isinstance(selected, torch.device)
        assert isinstance(shard, Sequence)
        assert isinstance(seeds, tuple)
        positions = tuple(int(entry[0]) for entry in shard)
        is_local = "dataset" in kwargs
        if is_local:
            assert kwargs["dataset"] is dataset
            assert kwargs["contract"] is contract
            assert kwargs["progress_factory"] is progress_factory
        observed.append(
            (
                worker_index,
                str(selected),
                str(kwargs["scheduler_name"]),
                float(kwargs["guidance_scale"]),
                int(kwargs["num_inference_steps"]),
                positions,
                tuple(int(seed) for seed in seeds),
                int(kwargs["num_loss_seeds"]),
                int(kwargs["loss_seed"]),
                is_local,
            )
        )
        rows = tuple(
            (
                position * 12 + step,
                {
                    "record_id": f"pair-{position}",
                    "status": "error" if position == 3 else "ok",
                },
            )
            for position in positions
            for step in range(12)
        )
        return experiment._PairShardResult(rows, 12 * int(3 in positions))

    monkeypatch.setattr(
        experiment.multiprocessing,
        "get_context",
        lambda method: spawn_context if method == "spawn" else None,
    )
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(experiment, "_run_pair_shard", fake_pair_shard)

    result = experiment._run_pair_shards(
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=12,
        num_loss_seeds=7,
        evaluation_source="gaussian",
        loss_seed=31,
        generation_seeds=(2, 5, 11),
        sample_batch_size=4,
        entries=entries,
        devices=devices,
        dataset=dataset,
        contract=contract,
        progress_factory=progress_factory,
    )

    assert observed == [
        (1, "cuda:1", "ddpm", 3.25, 12, (1, 4), (2, 5, 11), 7, 31, False),
        (2, "cuda:2", "ddpm", 3.25, 12, (2,), (2, 5, 11), 7, 31, False),
        (0, "cuda:0", "ddpm", 3.25, 12, (0, 3), (2, 5, 11), 7, 31, True),
    ]
    assert [position for position, _ in result.indexed_rows] == list(range(60))
    assert [row["record_id"] for _, row in result.indexed_rows] == [
        f"pair-{position}" for position in range(5) for _ in range(12)
    ]
    assert result.failed_count == 12


def test_multi_gpu_worker_crash_becomes_source_ordered_error_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_lock = experiment.tqdm.get_lock()
    spawn_context = SimpleNamespace(RLock=lambda: progress_lock)

    class FailedFuture:
        def result(self) -> object:
            raise RuntimeError("synthetic worker crash")

    class ImmediateExecutor:
        def __init__(
            self,
            *,
            max_workers: int,
            mp_context: object,
            initializer: object,
            initargs: tuple[object, ...],
        ) -> None:
            assert max_workers == 1
            assert mp_context is spawn_context
            assert initializer is experiment._install_tqdm_lock
            assert initargs == (progress_lock,)

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *_arguments: object) -> None:
            return None

        def submit(self, *_arguments: object, **_kwargs: object) -> FailedFuture:
            return FailedFuture()

    def local_pair_shard(**kwargs: object) -> experiment._PairShardResult:
        assert kwargs["worker_index"] == 0
        return experiment._PairShardResult(
            tuple((k, {"record_id": "pair-0", "status": "ok"}) for k in range(12)),
            0,
        )

    monkeypatch.setattr(
        experiment.multiprocessing,
        "get_context",
        lambda method: spawn_context if method == "spawn" else None,
    )
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(experiment, "_run_pair_shard", local_pair_shard)
    _FakeProgress.instances.clear()

    result = experiment._run_pair_shards(
        project_root=tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=12,
        num_loss_seeds=2,
        loss_seed=0,
        evaluation_source="gaussian",
        generation_seeds=(0, 1),
        sample_batch_size=2,
        entries=(
            (0, {"record_id": "pair-0"}),
            (1, {"record_id": "pair-1"}),
        ),
        devices=(torch.device("cuda:0"), torch.device("cuda:1")),
        dataset=object(),
        contract=SimpleNamespace(
            latent_shape=(4, 8, 8),
            parameters=_parameters(range(12, 0, -1)),
            scientific_hash="a" * 64,
            schedule_sha256="b" * 64,
        ),
        progress_factory=_FakeProgress,
    )

    assert [position for position, _row in result.indexed_rows] == list(range(24))
    assert result.indexed_rows[0][1] == {
        "record_id": "pair-0",
        "status": "ok",
    }
    failed_row = result.indexed_rows[12][1]
    assert failed_row["record_id"] == "pair-1"
    assert failed_row["status"] == "error"
    assert failed_row["latent_dimension"] == 256
    assert failed_row["scheduler_name"] == "ddpm"
    assert failed_row["guidance_scale"] == pytest.approx(3.25)
    assert failed_row["num_inference_steps"] == 12
    assert "worker on cuda:1 failed" in str(failed_row["error"])
    assert "synthetic worker crash" in str(failed_row["error"])
    assert result.failed_count == 12
    assert len(_FakeProgress.instances) == 1
    progress = _FakeProgress.instances[0]
    assert progress.updated == 12
    assert progress.kwargs["total"] == 12
    assert progress.kwargs["desc"] == (
        experiment.PROGRESS_DESCRIPTION + " [cuda:1 shard 2/2]"
    )
    assert progress.postfixes[-1]["phase"] == "worker-failure"
    assert progress.postfixes[-1]["status"] == "error"


def test_merge_rejects_missing_or_duplicate_pair_positions() -> None:
    first = experiment._PairShardResult(((0, {"status": "ok"}),), 0)
    duplicate = experiment._PairShardResult(((0, {"status": "ok"}),), 0)
    with pytest.raises(experiment.ExperimentError, match="duplicate or missing"):
        experiment._merge_pair_shard_results((first, duplicate), (0, 2))

    noncontiguous = experiment._merge_pair_shard_results(
        (
            experiment._PairShardResult(((4, {"record_id": "second"}),), 0),
            experiment._PairShardResult(((1, {"record_id": "first"}),), 0),
        ),
        (1, 4),
    )
    assert [position for position, _row in noncontiguous.indexed_rows] == [1, 4]


def test_gaussian_probes_keep_raw_noise_without_stored_dtype_rounding() -> None:
    seeds = (0, 1, 2)
    shape = (1, 2, 2)
    expected = experiment.make_initial_noise(seeds, shape)
    probes = experiment._make_gaussian_recovery_samples(seeds, shape)
    assert probes.dtype == torch.float32
    assert probes.device.type == "cpu" and probes.is_contiguous()
    assert torch.equal(probes, expected)
    assert not torch.equal(probes, expected.half().float())


def test_cached_recovery_uses_matching_latent_prediction_at_every_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("cached recovery must not sample or run model inference")

    monkeypatch.setattr(experiment, "predict_conditional_epsilon", forbidden)
    monkeypatch.setattr(
        experiment, "_reconstruct_generation_initial_latents", forbidden
    )
    latents = torch.arange(32, dtype=torch.float32).reshape(2, 4, 1, 2, 2)
    predictions = torch.arange(24, dtype=torch.float32).reshape(2, 3, 1, 2, 2) / 10
    target = torch.ones((1, 2, 2), dtype=torch.float32)
    for step, parameters in enumerate(_parameters((981, 501, 1))):
        measured = experiment._measure_trajectory_recovery(
            target_latent=target,
            latent_trajectory=latents,
            conditional_epsilon=predictions,
            step_index=step,
            alpha_t=parameters.alpha_t,
            sigma_t=parameters.sigma_t,
        )
        expected = (
            (
                (
                    latents[:, step].double()
                    - parameters.sigma_t * predictions[:, step].double()
                )
                / parameters.alpha_t
                - target.double()
            )
            .square()
            .mean()
            .item()
        )
        assert measured.recovery_mse == pytest.approx(expected)
        assert measured.recovery_rmse == pytest.approx(math.sqrt(expected))


@pytest.mark.parametrize(
    "damage",
    [
        "missing_step",
        "duplicate",
        "missing_source",
        "bad_ratio",
        "missing_columns",
        "bad_provenance",
        "failed",
    ],
)
def test_plot_rejects_incomplete_or_invalid_noise_sweeps(
    tmp_path: Path, damage: str
) -> None:
    rows = [
        _row(
            record,
            normalized_loss_rmse=1.0,
            recovery_rmse=2.0,
            mean_target_sscd=0.4,
            step_index=step,
            num_inference_steps=3,
            evaluation_source=source,
        )
        for record in ("one", "two")
        for step in range(3)
        for source in ("gaussian", "trajectory")
    ]
    if damage == "missing_step":
        rows.pop()
    elif damage == "duplicate":
        rows.append(dict(rows[-1]))
    elif damage == "missing_source":
        rows = [
            row
            for row in rows
            if not (
                row["record_id"] == "two" and row["evaluation_source"] == "trajectory"
            )
        ]
    elif damage == "bad_ratio":
        rows[0]["snr_t"] = 9.0
    elif damage == "bad_provenance":
        rows[0]["trajectory_sha256"] = "d" * 64
    elif damage == "failed":
        rows[0].update(status="error", error="synthetic timestep failure")
    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    if damage == "missing_columns":
        frame = frame.drop(columns=["evaluation_source", "step_index"])
    csv_path = tmp_path / experiment.CSV_NAME
    frame.to_csv(csv_path, index=False)
    with pytest.raises(experiment.ExperimentError):
        experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="both")
    assert list(tmp_path.iterdir()) == [csv_path]


@pytest.mark.parametrize(
    "sscd_score",
    [-1.0, -0.0136833919212222, 0.0, 1.0, -1.000005, 1.000005],
)
def test_plot_keeps_gaussian_and_trajectory_populations_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sscd_score: float,
) -> None:
    rows = [
        _row(
            "one",
            normalized_loss_rmse=1.0,
            recovery_rmse=value,
            mean_target_sscd=sscd_score,
            step_index=step,
            num_inference_steps=3,
            evaluation_source=source,
        )
        for source, value in (("gaussian", 2.0), ("trajectory", 20.0))
        for step in range(3)
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()
    captured: dict[str, pd.DataFrame] = {}
    routed_destinations: dict[str, tuple[Path, ...]] = {}

    def capture(
        *,
        valid: pd.DataFrame,
        evaluation_source: str,
        destinations: object,
        initial: pd.DataFrame | None = None,
    ) -> None:
        assert initial is not None
        captured[evaluation_source] = valid.copy()
        captured["overlay"] = initial.copy()
        routed_destinations[evaluation_source] = tuple(destinations)

    def capture_initial(*, valid: pd.DataFrame, destinations: object) -> None:
        captured["initial"] = valid.copy()
        routed_destinations["initial"] = tuple(destinations)

    monkeypatch.setattr(experiment, "_render_pair_figure", capture)
    monkeypatch.setattr(experiment, "_render_initial_loss_recovery", capture_initial)
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="both")
    assert set(captured) == {"trajectory", "initial", "overlay"}
    pd.testing.assert_frame_equal(captured["initial"], captured["overlay"])
    assert captured["initial"]["evaluation_source"].eq("gaussian").all()
    assert captured["initial"]["step_index"].eq(0).all()
    assert captured["initial"]["recovery_rmse"].eq(2.0).all()
    assert len(captured["initial"]) == 1
    np.testing.assert_allclose(captured["initial"]["mean_target_sscd"], sscd_score)
    assert routed_destinations["initial"] == experiment._figure_paths(
        tmp_path, "gaussian"
    )
    assert routed_destinations["trajectory"] == experiment._figure_paths(
        tmp_path, "trajectory"
    )
    assert csv_path.read_bytes() == original_csv
    assert captured["trajectory"]["evaluation_source"].eq("trajectory").all()
    assert captured["trajectory"]["recovery_rmse"].eq(20.0).all()
    assert set(captured["trajectory"]["step_index"]) == {0, 1, 2}
    np.testing.assert_allclose(captured["trajectory"]["mean_target_sscd"], sscd_score)


@pytest.mark.parametrize("score", [-1.01, 1.01, math.nan, math.inf, -math.inf])
def test_plot_rejects_invalid_sscd_with_record_diagnostics(
    tmp_path: Path,
    score: float,
) -> None:
    rows = [
        _row(
            "invalid-sscd",
            normalized_loss_rmse=1.0,
            recovery_rmse=2.0,
            mean_target_sscd=score,
            step_index=step,
            num_inference_steps=3,
        )
        for step in range(3)
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    with pytest.raises(experiment.ExperimentError) as caught:
        experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="gaussian")
    message = str(caught.value)
    assert "invalid gaussian plotting measurements" in message
    assert "mean_target_sscd: 3 invalid rows" in message
    assert "record_id=invalid-sscd" in message and "timestep=981" in message
    assert list(tmp_path.iterdir()) == [csv_path]


@pytest.mark.parametrize(
    "losses,recoveries,xscale,yscale",
    [
        ([1.0, 2.0], [3.0, 4.0], "log", "log"),
        ([0.0, 2.0], [3.0, 4.0], "symlog", "log"),
        ([1.0, 2.0], [0.0, 4.0], "log", "symlog"),
        ([0.0, 2.0], [0.0, 4.0], "symlog", "symlog"),
        ([0.0, 0.0], [3.0, 4.0], "linear", "log"),
        ([1.0, 2.0], [0.0, 0.0], "log", "linear"),
        ([0.0, 0.0], [0.0, 0.0], "linear", "linear"),
    ],
)
def test_trajectory_sweep_keeps_exact_zero_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    losses: list[float],
    recoveries: list[float],
    xscale: str,
    yscale: str,
) -> None:
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(
        [
            _row(
                "one",
                normalized_loss_rmse=loss,
                recovery_rmse=recovery,
                mean_target_sscd=0.4,
                step_index=step,
                num_inference_steps=len(losses),
                evaluation_source="trajectory",
            )
            for step, (loss, recovery) in enumerate(
                zip(losses, recoveries, strict=True)
            )
        ],
        columns=EXPECTED_COLUMNS,
    ).to_csv(csv_path, index=False)
    real_subplots = experiment.plt.subplots
    axes = []

    def capture(*args: object, **kwargs: object):
        figure, axis = real_subplots(*args, **kwargs)
        axes.append(axis)
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", capture)
    monkeypatch.setattr(
        experiment, "_render_initial_loss_recovery", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        experiment, "_render_loss_timestep_figure", lambda **_kwargs: None
    )
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="trajectory")
    assert len(axes) == 1
    axis = axes[0]
    assert axis.get_xscale() == xscale
    assert axis.get_yscale() == yscale
    _assert_shared_scientific_axes(axis)
    assert not axis.lines
    assert axis.get_legend() is None
    assert len(axis.collections) == 2
    observations = next(
        item for item in axis.collections if isinstance(item, LineCollection)
    )
    initial_points = next(
        item for item in axis.collections if isinstance(item, PathCollection)
    )
    np.testing.assert_array_equal(
        initial_points.get_offsets(), [[losses[0], recoveries[0]]]
    )
    assert initial_points.get_alpha() == 1.0
    assert initial_points.get_zorder() > observations.get_zorder()
    segments = observations.get_segments()
    assert len(segments) == 1
    np.testing.assert_array_equal(segments[0][:, 0], losses)
    np.testing.assert_array_equal(segments[0][:, 1], recoveries)
    assert axis.get_xlim()[0] <= min(losses)
    assert axis.get_xlim()[1] >= max(losses)
    if 0.0 in losses:
        assert axis.get_xlim()[0] <= 0.0 <= axis.get_xlim()[1]
    if 0.0 in recoveries:
        assert axis.get_ylim()[0] <= 0.0 <= axis.get_ylim()[1]


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
    step_index: int = 0,
    num_inference_steps: int = 1,
    evaluation_source: str = "gaussian",
) -> dict[str, object]:
    alpha_t = 0.1 * (step_index + 1)
    sigma_t = math.sqrt(1.0 - alpha_t**2)
    snr_t = alpha_t**2 / sigma_t**2
    normalized_loss_mse = normalized_loss_rmse**2
    recovery_mse = recovery_rmse**2
    return {
        "record_id": record_id,
        "selection_strategy": "gmm",
        "selection_hash": SELECTION_HASH,
        "model_name": "sdv1",
        "scheduler_name": "ddim",
        "guidance_scale": 7.5,
        "num_inference_steps": num_inference_steps,
        "evaluation_generation_scientific_config_hash": "a" * 64,
        "evaluation_schedule_sha256": "b" * 64,
        "evaluation_source": evaluation_source,
        "step_index": step_index,
        "timestep": 981 - 20 * step_index,
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
        "trajectory_sha256": "d" * 64 if evaluation_source == "trajectory" else "",
        "status": status,
        "error": error,
    }


def test_initial_loss_recovery_scatter_reloads_only_gaussian_initial_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row(
            f"record-{index}",
            normalized_loss_rmse=loss + step * 100.0,
            recovery_rmse=recovery + step * 200.0 + source_offset,
            mean_target_sscd=sscd,
            step_index=step,
            num_inference_steps=3,
            evaluation_source=source,
        )
        for index, (loss, recovery, sscd) in enumerate(
            [(0.5, 3.0, -0.1), (2.0, 0.25, 0.5), (8.0, 1.5, 0.9)]
        )
        for step in range(3)
        for source, source_offset in (("gaussian", 0.0), ("trajectory", 1000.0))
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows[::-1], columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()
    captured: dict[str, object] = {}
    read_paths: list[Path] = []
    real_read_csv = experiment.pd.read_csv
    real_subplots = experiment.plt.subplots

    def read_csv_spy(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    def subplots_spy(*args: object, **kwargs: object):
        figure, axis = real_subplots(*args, **kwargs)
        captured.update(
            figure=figure,
            axis=axis,
            font_family=tuple(experiment.matplotlib.rcParams["font.family"]),
            mathtext_fontset=experiment.matplotlib.rcParams["mathtext.fontset"],
        )
        return figure, axis

    monkeypatch.setattr(experiment.pd, "read_csv", read_csv_spy)
    monkeypatch.setattr(experiment.plt, "subplots", subplots_spy)
    monkeypatch.setattr(experiment, "_render_pair_figure", lambda **_kwargs: None)
    monkeypatch.setattr(
        experiment, "_render_loss_timestep_figure", lambda **_kwargs: None
    )
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="both")

    assert read_paths == [csv_path]
    assert csv_path.read_bytes() == original_csv
    png_path, pdf_path = experiment._figure_paths(tmp_path, "gaussian")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF-")
    figure = captured["figure"]
    axis = captured["axis"]
    assert len(figure.axes) == 2  # One scientific panel and its SSCD colorbar.
    np.testing.assert_allclose(figure.get_size_inches(), [4.0, 4.0])
    assert captured["font_family"] == ("STIXGeneral",)
    assert captured["mathtext_fontset"] == "stix"
    assert axis.get_title() == ""
    _assert_primary_median(
        axis,
        x_values=[0.5, 2.0, 8.0],
        y_values=[3.0, 0.25, 1.5],
    )
    assert not np.allclose(axis.lines[0].get_xdata(), 1.0 / math.sqrt(4))
    assert not axis.texts
    _assert_shared_scientific_axes(axis)
    assert axis.get_xscale() == axis.get_yscale() == "log"
    for coordinate in (axis.xaxis, axis.yaxis):
        locator = coordinate.get_major_locator()
        formatter = coordinate.get_major_formatter()
        assert isinstance(locator, LogLocator)
        lower, upper = coordinate.get_view_interval()
        expected_subs = [1.0] if math.log10(upper / lower) > 1.0 else [1.0, 2.0, 5.0]
        np.testing.assert_array_equal(locator._subs, expected_subs)
        assert locator.numticks == 7
        assert isinstance(formatter, experiment._SharedScientificFormatter)
        assert coordinate.get_offset_text().get_fontsize() == 10
        assert isinstance(coordinate.get_minor_formatter(), NullFormatter)
        assert all(label.get_text() == "" for label in coordinate.get_minorticklabels())
    assert axis.xaxis.label.get_fontsize() == 15
    assert axis.yaxis.label.get_fontsize() == 15
    assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
    assert all(label.get_fontsize() == 12 for label in axis.get_yticklabels())
    assert axis.get_xlabel() == (r"$\sqrt{\mathcal{L}_T(c)/[d\,\mathrm{SNR}_T]}$")
    assert axis.get_ylabel() == (
        r"$\sqrt{\mathbb{E}_{\mathbf{x}_T,\boldsymbol{\xi}}"
        r"[\|\widehat{\mathbf{x}}_{0\mid T,c}(\mathbf{x}_T)-"
        r"\mathbf{x}^{\star}\|^2]/d}$"
    )
    assert r"\frac" not in axis.get_xlabel()
    assert r"\|_2" not in axis.get_ylabel()

    assert len(axis.collections) == 1  # No percentile bands on the initial scatter.
    observations = axis.collections[0]
    assert isinstance(observations, PathCollection)
    offsets = np.asarray(observations.get_offsets(), dtype=float)
    colors = np.asarray(observations.get_array(), dtype=float)
    order = np.argsort(offsets[:, 0])
    np.testing.assert_allclose(offsets[order], [[0.5, 3.0], [2.0, 0.25], [8.0, 1.5]])
    np.testing.assert_allclose(colors[order], [-0.1, 0.5, 0.9])
    assert observations.cmap.name == "viridis"
    assert observations.norm.vmin == 0.0
    assert observations.norm.vmax == 1.0
    assert observations.norm.clip is True
    assert observations.get_alpha() == pytest.approx(experiment.OBSERVATION_LINE_ALPHA)
    np.testing.assert_allclose(observations.get_sizes(), [18.0])
    np.testing.assert_allclose(observations.get_linewidths(), [0.0])

    colorbar = next(item for item in figure.axes if item is not axis)
    assert colorbar.get_ylabel() == "SSCD"
    np.testing.assert_allclose(colorbar.get_ylim(), [0.0, 1.0])
    meshes = [item for item in colorbar.collections if isinstance(item, QuadMesh)]
    assert len(meshes) == 1
    assert meshes[0].get_alpha() == 1.0


@pytest.mark.parametrize(
    ("ticks", "limits", "visible_values", "coefficients", "exponent"),
    [
        (
            [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            (0.19, 1.01),
            [0.2, 0.5, 1.0],
            ["2", "5", "10"],
            -1,
        ),
        (
            [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
            (1e-4, 100.0),
            [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
            ["1", "10", "100", "1000", "10000", "100000", "1000000"],
            -4,
        ),
        ([1.0, 2.0, 5.0], (0.9, 5.1), [1.0, 2.0, 5.0], ["1", "2", "5"], 0),
        ([0.0], (-1.0, 1.0), [0.0], ["0"], 0),
        (
            [0.0, 0.2, 0.5, 1.0],
            (0.0, 1.0),
            [0.0, 0.2, 0.5, 1.0],
            ["0", "2", "5", "10"],
            -1,
        ),
        (
            [np.nextafter(0.1, 0.0), 0.2, 0.5],
            (0.09, 0.51),
            [np.nextafter(0.1, 0.0), 0.2, 0.5],
            ["1", "2", "5"],
            -1,
        ),
        (
            [100.0, 200.0, 500.0],
            (99.0, 501.0),
            [100.0, 200.0, 500.0],
            ["1", "2", "5"],
            2,
        ),
        (
            [0.00012, 0.00456, 100.0],
            (0.0001, 100.0),
            [0.00012, 0.00456, 100.0],
            ["1.2", "45.6", "1000000"],
            -4,
        ),
        (
            [math.nan, -math.inf, 0.0, 0.2, 0.5, 1.0, math.inf],
            (0.0, 1.0),
            [0.0, 0.2, 0.5, 1.0],
            ["0", "2", "5", "10"],
            -1,
        ),
        ([], (-1.0, 1.0), [0.0], ["0"], 0),
    ],
    ids=[
        "visible-ticks-ignore-off-view-padding",
        "six-decades-never-round-nonzero-to-zero",
        "no-multiplier-for-exponent-zero",
        "all-zero",
        "zero-and-positive",
        "near-power-roundoff",
        "positive-exponent",
        "fractional-coefficients-across-decades",
        "nonfinite-and-zero-do-not-select-exponent",
        "empty-locations",
    ],
)
def test_shared_scientific_formatter_uses_one_visible_tick_exponent(
    ticks: list[float],
    limits: tuple[float, float],
    visible_values: list[float],
    coefficients: list[str],
    exponent: int,
) -> None:
    figure, axis = experiment.plt.subplots()
    try:
        axis.set_xlim(*limits)
        formatter = experiment._SharedScientificFormatter()
        axis.xaxis.set_major_formatter(formatter)
        formatter.set_locs(np.asarray(ticks, dtype=float))
        assert formatter.exponent == exponent
        expected_offset = rf"$\times 10^{{{exponent}}}$" if exponent else ""
        assert formatter.get_offset() == expected_offset
        assert [formatter(value) for value in visible_values] == [
            rf"$\mathdefault{{{coefficient}}}$" for coefficient in coefficients
        ]
        assert all(
            formatter(value) != r"$\mathdefault{0}$"
            for value in visible_values
            if value != 0.0
        )
        assert not axis.texts
    finally:
        experiment.plt.close(figure)


def test_shared_scientific_formatter_refreshes_exponent_when_view_changes() -> None:
    figure, axis = experiment.plt.subplots()
    try:
        ticks = np.asarray([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        formatter = experiment._SharedScientificFormatter()
        axis.xaxis.set_major_formatter(formatter)
        axis.set_xlim(0.19, 1.01)
        formatter.set_locs(ticks)
        assert formatter.exponent == -1
        assert formatter(0.5) == r"$\mathdefault{5}$"

        axis.set_xlim(0.9, 5.1)
        formatter.set_locs(ticks)
        assert formatter.exponent == 0
        assert formatter.get_offset() == ""
        assert formatter(1.0) == r"$\mathdefault{1}$"

        axis.set_xlim(1.01, 0.19)
        formatter.set_locs(ticks)
        assert formatter.exponent == -1
        assert formatter.get_offset() == r"$\times 10^{-1}$"
    finally:
        experiment.plt.close(figure)


@pytest.mark.parametrize(
    ("values", "limits", "ticks", "expected_scale", "expected_offset"),
    [
        (
            [0.2, 0.5, 1.0],
            (0.19, 1.01),
            [0.1, 0.2, 0.5, 1.0, 2.0],
            "log",
            r"$\times 10^{-1}$",
        ),
        (
            [0.0, 0.2, 0.5, 1.0],
            (0.0, 1.01),
            [0.0, 0.2, 0.5, 1.0],
            "symlog",
            r"$\times 10^{-1}$",
        ),
        ([0.0, 0.0], (-1.0, 1.0), [0.0], "linear", ""),
    ],
)
def test_shared_scientific_format_preserves_axes_data_locators_and_native_offsets(
    values: list[float],
    limits: tuple[float, float],
    ticks: list[float],
    expected_scale: str,
    expected_offset: str,
) -> None:
    with experiment.matplotlib.rc_context(experiment.PLOT_STYLE):
        figure, axis = experiment.plt.subplots()
        try:
            points = np.asarray(values, dtype=float)
            line = axis.plot(points, points)[0]
            experiment._set_recovery_axis_scale(axis, points, coordinate="x")
            experiment._set_recovery_axis_scale(axis, points)
            axis.set_xlim(*limits)
            axis.set_ylim(*limits)
            x_locator = FixedLocator(ticks)
            y_locator = FixedLocator(ticks)
            axis.xaxis.set_major_locator(x_locator)
            axis.yaxis.set_major_locator(y_locator)
            experiment._set_shared_scientific_format(axis)
            figure.canvas.draw()

            assert axis.get_xscale() == axis.get_yscale() == expected_scale
            assert axis.xaxis.get_major_locator() is x_locator
            assert axis.yaxis.get_major_locator() is y_locator
            np.testing.assert_array_equal(axis.get_xlim(), limits)
            np.testing.assert_array_equal(axis.get_ylim(), limits)
            np.testing.assert_array_equal(line.get_xdata(), points)
            np.testing.assert_array_equal(line.get_ydata(), points)
            _assert_shared_scientific_axes(axis)
            for coordinate in (axis.xaxis, axis.yaxis):
                assert coordinate.get_offset_text().get_text() == expected_offset
                assert coordinate.get_offset_text().get_fontfamily() == ["STIXGeneral"]
                assert coordinate.get_offset_text().get_fontsize() == 10
            assert len(figure.axes) == 1
            assert not axis.texts
            assert axis.get_legend() is None
        finally:
            experiment.plt.close(figure)


def test_binned_medians_use_x_sorted_equal_count_prompt_bins() -> None:
    x_values = np.arange(12.0, 0.0, -1.0)
    y_values = 100.0 - x_values

    median_x, median_y = experiment._binned_medians(x_values, y_values)

    np.testing.assert_array_equal(
        median_x,
        [1.5, 3.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    np.testing.assert_array_equal(
        median_y,
        [98.5, 96.5, 95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0],
    )


@pytest.mark.parametrize(
    "losses,recoveries,xscale,yscale",
    [
        ([1.0, 2.0], [3.0, 4.0], "log", "log"),
        ([0.0, 2.0], [3.0, 4.0], "symlog", "log"),
        ([1.0, 2.0], [0.0, 4.0], "log", "symlog"),
        ([0.0, 2.0], [0.0, 4.0], "symlog", "symlog"),
        ([0.0, 0.0], [3.0, 4.0], "linear", "log"),
        ([1.0, 2.0], [0.0, 0.0], "log", "linear"),
        ([0.0, 0.0], [0.0, 0.0], "linear", "linear"),
    ],
)
def test_initial_loss_recovery_keeps_exact_zero_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    losses: list[float],
    recoveries: list[float],
    xscale: str,
    yscale: str,
) -> None:
    frame = pd.DataFrame(
        [
            _row(
                f"record-{index}",
                normalized_loss_rmse=loss,
                recovery_rmse=recovery,
                mean_target_sscd=0.5,
            )
            for index, (loss, recovery) in enumerate(
                zip(losses, recoveries, strict=True)
            )
        ],
        columns=EXPECTED_COLUMNS,
    )
    real_subplots = experiment.plt.subplots
    captured = {}

    def capture(*args: object, **kwargs: object):
        figure, axis = real_subplots(*args, **kwargs)
        captured["axis"] = axis
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", capture)
    monkeypatch.setattr(experiment, "_atomic_save_figures", lambda *_args: None)
    experiment._render_initial_loss_recovery(
        valid=frame, destinations=experiment._figure_paths(tmp_path, "gaussian")
    )
    axis = captured["axis"]
    assert axis.get_xscale() == xscale
    assert axis.get_yscale() == yscale
    _assert_shared_scientific_axes(axis)
    _assert_primary_median(axis, x_values=losses, y_values=recoveries)
    offsets = np.asarray(axis.collections[0].get_offsets(), dtype=float)
    np.testing.assert_array_equal(offsets[:, 0], losses)
    np.testing.assert_array_equal(offsets[:, 1], recoveries)
    assert axis.get_xlim()[0] <= min(losses)
    assert axis.get_xlim()[1] >= max(losses)
    if 0.0 in losses:
        assert axis.get_xlim()[0] <= 0.0 <= axis.get_xlim()[1]
    if 0.0 in recoveries:
        assert axis.get_ylim()[0] <= 0.0 <= axis.get_ylim()[1]


@pytest.mark.parametrize("latent_dimension", [16384, 4])
def test_initial_loss_recovery_limits_do_not_expand_to_latent_dimension_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    latent_dimension: int,
) -> None:
    frame = pd.DataFrame(
        [
            _row(
                f"record-{index}",
                normalized_loss_rmse=loss,
                recovery_rmse=recovery,
                mean_target_sscd=score,
            )
            for index, (loss, recovery, score) in enumerate(
                [(1.0, 3.0, 0.2), (2.0, 4.0, 0.8)]
            )
        ],
        columns=EXPECTED_COLUMNS,
    )
    frame["latent_dimension"] = latent_dimension
    frame["conditional_loss"] = (
        frame["normalized_loss_mse"] * latent_dimension * frame["snr_t"]
    )
    original = frame.copy(deep=True)
    captured = {}
    real_subplots = experiment.plt.subplots

    def capture(*args: object, **kwargs: object):
        figure, axis = real_subplots(*args, **kwargs)
        captured["axis"] = axis
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", capture)
    monkeypatch.setattr(experiment, "_atomic_save_figures", lambda *_args: None)
    experiment._render_initial_loss_recovery(
        valid=frame, destinations=experiment._figure_paths(tmp_path, "gaussian")
    )

    pd.testing.assert_frame_equal(frame, original)
    axis = captured["axis"]
    _assert_primary_median(
        axis,
        x_values=[1.0, 2.0],
        y_values=[3.0, 4.0],
    )
    assert axis.get_xscale() == axis.get_yscale() == "log"
    np.testing.assert_array_equal(axis.xaxis.get_major_locator()._subs, [1.0, 2.0, 5.0])
    assert axis.xaxis.get_major_locator().numticks == 7
    assert isinstance(axis.xaxis.get_minor_formatter(), NullFormatter)
    assert all(label.get_text() == "" for label in axis.xaxis.get_minorticklabels())
    np.testing.assert_array_equal(axis.yaxis.get_major_locator()._subs, [1.0, 2.0, 5.0])
    assert axis.get_xlim()[0] > 1.0 / math.sqrt(latent_dimension)
    assert axis.get_xlim()[1] >= 2.0
    np.testing.assert_array_equal(
        axis.collections[0].get_offsets(), [[1.0, 3.0], [2.0, 4.0]]
    )
    assert math.log10(axis.get_xlim()[1] / axis.get_xlim()[0]) <= 1.0


def test_trajectory_only_plot_never_creates_initial_loss_recovery_scatter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(
        [
            _row(
                "one",
                normalized_loss_rmse=1.0,
                recovery_rmse=2.0,
                mean_target_sscd=0.5,
                step_index=step,
                num_inference_steps=3,
                evaluation_source="trajectory",
            )
            for step in range(3)
        ],
        columns=EXPECTED_COLUMNS,
    ).to_csv(csv_path, index=False)
    captured = []

    def forbidden(**_kwargs: object) -> None:
        pytest.fail("trajectory diagnostics cannot use the Gaussian theorem panel")

    def capture(
        *,
        valid: pd.DataFrame,
        evaluation_source: str,
        destinations: object,
        initial: pd.DataFrame | None = None,
    ) -> None:
        assert initial is None
        captured.append((evaluation_source, valid.copy(), tuple(destinations)))

    monkeypatch.setattr(experiment, "_render_initial_loss_recovery", forbidden)
    monkeypatch.setattr(experiment, "_render_pair_figure", capture)
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="trajectory")
    assert len(captured) == 1
    source, frame, paths = captured[0]
    assert source == "trajectory"
    assert set(frame["step_index"]) == {0, 1, 2}
    assert paths == experiment._figure_paths(tmp_path, "trajectory")


def test_plot_reloads_csv_and_draws_trajectory_with_matching_primary_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_frames = []

    def capture_primary(*, valid: pd.DataFrame, destinations: object) -> None:
        primary_frames.append(valid.copy())

    monkeypatch.setattr(experiment, "_render_initial_loss_recovery", capture_primary)
    valid_rows = [
        _row(
            f"sdv1-{index:04d}",
            normalized_loss_rmse=float(index + 1) * [1.0, 3.0, 2.0][step],
            recovery_rmse=1.0 / float(index + 1)
            + (50.0 if source == "gaussian" else 0.0),
            mean_target_sscd=-0.1 + index / 19.0,
            step_index=step,
            num_inference_steps=3,
            evaluation_source=source,
        )
        for index in range(20)
        for step in range(3)
        for source in ("gaussian", "trajectory")
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(list(reversed(valid_rows)), columns=EXPECTED_COLUMNS).to_csv(
        csv_path, index=False
    )
    original_csv = csv_path.read_bytes()

    real_read_csv = pd.read_csv
    reads: list[Path] = []

    def read_csv_spy(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(experiment.pd, "read_csv", read_csv_spy)
    real_subplots = experiment.plt.subplots
    captured: dict[str, object] = {}
    savefig_calls: list[dict[str, object]] = []

    def subplots_spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        figure, axis = real_subplots(*args, **kwargs)
        real_savefig = figure.savefig
        real_tight_layout = figure.tight_layout

        def savefig_spy(*save_args: object, **save_kwargs: object) -> object:
            savefig_calls.append(dict(save_kwargs))
            return real_savefig(*save_args, **save_kwargs)

        def tight_layout_spy(*layout_args: object, **layout_kwargs: object) -> object:
            captured["tight_layout_calls"] = (
                int(captured.get("tight_layout_calls", 0)) + 1
            )
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
    monkeypatch.setattr(
        experiment, "_render_loss_timestep_figure", lambda **_kwargs: None
    )
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="both")

    assert reads == [csv_path]
    assert csv_path.read_bytes() == original_csv
    assert len(primary_frames) == 1
    png_path, pdf_path = experiment._figure_paths(tmp_path, "trajectory")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF-")
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

    observation_collections = [
        collection
        for collection in axis.collections
        if isinstance(collection, LineCollection)
    ]
    assert len(observation_collections) == 1
    observations = observation_collections[0]
    initial_points = [
        item for item in axis.collections if isinstance(item, PathCollection)
    ]
    assert len(initial_points) == 1
    overlay = initial_points[0]
    assert overlay.get_alpha() == 1.0
    assert overlay.get_zorder() > observations.get_zorder()
    assert overlay.cmap.name == "viridis"
    assert overlay.norm.vmin == 0.0
    assert overlay.norm.vmax == 1.0
    assert overlay.norm.clip is True
    np.testing.assert_allclose(overlay.get_sizes(), [18.0])
    overlay_xy = np.asarray(overlay.get_offsets(), dtype=float)
    overlay_scores = np.asarray(overlay.get_array(), dtype=float)
    order = np.argsort(overlay_xy[:, 0])
    primary = primary_frames[0].sort_values("normalized_loss_rmse")
    assert len(primary) == len(overlay_xy) == 20
    assert set(primary["record_id"]) == {f"sdv1-{index:04d}" for index in range(20)}
    np.testing.assert_allclose(
        overlay_xy[order], primary[["normalized_loss_rmse", "recovery_rmse"]]
    )
    np.testing.assert_allclose(overlay_scores[order], primary["mean_target_sscd"])
    assert observations.cmap.name == "viridis"
    assert observations.norm.vmin == pytest.approx(0.0)
    assert observations.norm.vmax == pytest.approx(1.0)
    assert observations.norm.clip is True
    assert observations.get_alpha() == pytest.approx(experiment.OBSERVATION_LINE_ALPHA)
    assert observations.get_alpha() != experiment.COLORBAR_ALPHA
    assert observations.get_array() is not None
    np.testing.assert_allclose(
        np.sort(np.asarray(observations.get_array(), dtype=float)),
        np.linspace(-0.1, 0.9, 20),
    )
    assert float(np.min(observations.get_array())) < 0.0
    assert float(observations.norm(-0.1)) == 0.0
    np.testing.assert_allclose(
        observations.cmap(observations.norm(-0.1)), observations.cmap(0.0)
    )
    segments = observations.get_segments()
    assert len(segments) == 20
    for index in range(20):
        matching = [
            segment
            for segment in segments
            if np.allclose(segment[:, 1], 1.0 / float(index + 1))
        ]
        assert len(matching) == 1
        expected_x = float(index + 1) * np.asarray([1.0, 3.0, 2.0])
        np.testing.assert_allclose(matching[0][:, 0], expected_x)
        assert np.diff(matching[0][:, 0])[0] > 0.0
        assert np.diff(matching[0][:, 0])[1] < 0.0
    plotted_x = np.concatenate([segment[:, 0] for segment in segments])
    assert axis.get_xlim()[0] <= plotted_x.min()
    assert axis.get_xlim()[1] >= plotted_x.max()
    assert not axis.lines
    assert axis.get_legend() is None
    assert len(axis.collections) == 2  # Prompt curves and initial points, no bands.
    assert axis.get_ylim()[1] >= overlay_xy[:, 1].max()

    assert len(axis.texts) == 0
    _assert_shared_scientific_axes(axis)
    assert axis.get_xlabel() == (r"$\sqrt{\mathcal{L}_t(c)/[d\,\mathrm{SNR}_t]}$")
    assert r"\frac" not in axis.get_xlabel()
    assert r"\mathrm{SNR}_t" in axis.get_xlabel()
    assert axis.get_ylabel() == experiment._source_ylabel("trajectory")
    for source in ("gaussian", "trajectory"):
        label = experiment._source_ylabel(source)
        assert r"\|_2" not in label
        assert r"\|^2]/d" in label
        assert r"\mathbb{E}_{\mathbf{x}_T,\boldsymbol{\xi}}" in label
        assert r"\mathbb{E}_{s}" not in label
        assert "^{(s)}" not in label
    assert experiment._source_ylabel("gaussian") == (
        r"$\sqrt{\mathbb{E}_{\mathbf{x}_T,\boldsymbol{\xi}}"
        r"[\|\widehat{\mathbf{x}}_{0\mid t,c}(\mathbf{x}_T)-"
        r"\mathbf{x}^{\star}\|^2]/d}$"
    )
    assert experiment._source_ylabel("trajectory") == (
        r"$\sqrt{\mathbb{E}_{\mathbf{x}_T,\boldsymbol{\xi}}"
        r"[\|\widehat{\mathbf{x}}_{0\mid t,c}-"
        r"\mathbf{x}^{\star}\|^2]/d}$"
    )
    assert "recovery_rmse" not in axis.get_ylabel()

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
    assert savefig_calls == [
        {
            "format": "png",
            "bbox_inches": "tight",
            "pad_inches": 0.05,
            "dpi": 150,
        },
        {
            "format": "pdf",
            "bbox_inches": "tight",
            "pad_inches": 0.05,
            "dpi": 150,
        },
    ]


@pytest.mark.parametrize(
    "damage", ["missing_prompt", "extra_prompt", "duplicate_prompt", "later_step"]
)
def test_trajectory_overlay_rejects_mismatched_selected_prompt_coverage(
    tmp_path: Path, damage: str
) -> None:
    trajectories = pd.DataFrame(
        [
            _row(
                record,
                normalized_loss_rmse=1.0,
                recovery_rmse=2.0,
                mean_target_sscd=0.5,
                step_index=step,
                num_inference_steps=2,
                evaluation_source="trajectory",
            )
            for record in ("selected-one", "selected-two")
            for step in range(2)
        ],
        columns=EXPECTED_COLUMNS,
    )
    initial = trajectories.loc[trajectories["step_index"].eq(0)].copy()
    initial["evaluation_source"] = "gaussian"
    if damage == "missing_prompt":
        initial = initial.iloc[:1]
    elif damage == "extra_prompt":
        extra = initial.iloc[:1].copy()
        extra["record_id"] = "unselected"
        initial = pd.concat([initial, extra], ignore_index=True)
    elif damage == "duplicate_prompt":
        initial = pd.concat([initial, initial.iloc[:1]], ignore_index=True)
    else:
        initial["step_index"] = 1
    with pytest.raises(experiment.ExperimentError, match="selected trajectory prompts"):
        experiment._render_pair_figure(
            valid=trajectories,
            evaluation_source="trajectory",
            destinations=experiment._figure_paths(tmp_path, "trajectory"),
            initial=initial,
        )
    assert not list(tmp_path.iterdir())


def test_gaussian_only_plot_creates_no_sweep_and_removes_only_retired_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(
        [
            _row(
                "selected",
                normalized_loss_rmse=1.0 + step,
                recovery_rmse=2.0 + step,
                mean_target_sscd=0.5,
                step_index=step,
                num_inference_steps=2,
            )
            for step in range(2)
        ],
        columns=EXPECTED_COLUMNS,
    ).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()
    retired = [
        tmp_path / "theorem1_loss_recovery_noise_sweep.png",
        tmp_path / "theorem1_loss_recovery_noise_sweep.pdf",
    ]
    for path in retired:
        path.write_bytes(b"retired")
    unrelated = tmp_path / "theorem1_loss_recovery_noise_sweep_notes.txt"
    unrelated.write_bytes(b"unrelated")
    calls = []

    def capture_primary(*, valid: pd.DataFrame, destinations: object) -> None:
        calls.append((valid.copy(), tuple(destinations)))

    def forbidden(**_kwargs: object) -> None:
        pytest.fail("Gaussian-only plotting must not render any sweep")

    monkeypatch.setattr(experiment, "_render_initial_loss_recovery", capture_primary)
    monkeypatch.setattr(experiment, "_render_pair_figure", forbidden)
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="gaussian")
    assert len(calls) == 1
    frame, destinations = calls[0]
    assert frame["record_id"].tolist() == ["selected"]
    assert frame["step_index"].tolist() == [0]
    assert destinations == experiment._figure_paths(tmp_path, "gaussian")
    assert all(not path.exists() for path in retired)
    assert unrelated.read_bytes() == b"unrelated"
    assert csv_path.read_bytes() == original_csv


def test_plot_single_timestep_keeps_one_unjoined_observation_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment, "_render_initial_loss_recovery", lambda **_kwargs: None
    )
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(
        [
            _row(
                "sdv1-0000",
                normalized_loss_rmse=1.0,
                recovery_rmse=2.0,
                mean_target_sscd=0.3,
                evaluation_source="trajectory",
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
    monkeypatch.setattr(
        experiment, "_render_loss_timestep_figure", lambda **_kwargs: None
    )
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="trajectory")

    axis = captured["axis"]
    assert not axis.lines
    assert axis.get_legend() is None
    observations = [
        item for item in axis.collections if isinstance(item, LineCollection)
    ]
    assert len(observations) == 1
    assert len(observations[0].get_segments()) == 1
    assert observations[0].get_segments()[0].shape == (1, 2)


@pytest.mark.parametrize(
    "source_mode,expected_source",
    [("gaussian", "gaussian"), ("trajectory", "trajectory"), ("both", "gaussian")],
)
def test_loss_timestep_plot_routes_full_grid_once_without_source_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_mode: str,
    expected_source: str,
) -> None:
    sources = ("gaussian", "trajectory") if source_mode == "both" else (source_mode,)
    rows = [
        _row(
            record,
            normalized_loss_rmse=1.0 + step,
            recovery_rmse=2.0 if source == "gaussian" else 20.0,
            mean_target_sscd=score,
            step_index=step,
            num_inference_steps=3,
            evaluation_source=source,
        )
        for record, score in (("one", 0.1), ("two", 0.9))
        for step in range(3)
        for source in sources
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows[::-1], columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()
    captured = []
    reads = []
    real_read_csv = experiment.pd.read_csv

    def read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    def capture(*, valid: pd.DataFrame, destinations: object) -> None:
        captured.append((valid.copy(), tuple(destinations)))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("loss-timestep plotting must use saved losses, not model inference")

    monkeypatch.setattr(experiment.pd, "read_csv", read_csv)
    monkeypatch.setattr(
        experiment, "_render_initial_loss_recovery", lambda **_kwargs: None
    )
    monkeypatch.setattr(experiment, "_render_pair_figure", lambda **_kwargs: None)
    monkeypatch.setattr(experiment, "_render_loss_timestep_figure", capture)
    monkeypatch.setattr(experiment, "load_model_components", forbidden)
    monkeypatch.setattr(experiment, "preflight_model_components", forbidden)
    monkeypatch.setattr(experiment, "_measure_conditional_loss", forbidden)
    monkeypatch.setattr(
        experiment, "_reconstruct_generation_initial_latents", forbidden
    )
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source=source_mode)

    assert reads == [csv_path]
    assert csv_path.read_bytes() == original_csv
    assert len(captured) == 1
    frame, destinations = captured[0]
    assert len(frame) == 6
    assert set(frame["record_id"]) == {"one", "two"}
    assert set(frame["step_index"]) == {0, 1, 2}
    assert frame["evaluation_source"].eq(expected_source).all()
    assert not frame.duplicated(["record_id", "timestep"]).any()
    assert destinations == tuple(
        tmp_path / name for name in experiment.LOSS_TIMESTEP_FIGURE_FILENAMES
    )


def test_loss_timestep_plot_reloads_raw_losses_and_actual_scheduler_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timesteps = (981, 417, 7)
    prompt_losses = {
        "selected-low": (0.012, 0.38, 2.0),
        "selected-high": (0.004, 0.16, 0.7),
        "selected-negative-sscd": (0.2, 0.8, 4.0),
    }
    prompt_scores = {
        "selected-low": 0.1,
        "selected-high": 0.95,
        "selected-negative-sscd": -0.05,
    }
    rows = []
    for record, losses in prompt_losses.items():
        for step, (timestep, loss) in enumerate(zip(timesteps, losses, strict=True)):
            for source in ("gaussian", "trajectory"):
                row = _row(
                    record,
                    normalized_loss_rmse=1.0,
                    recovery_rmse=9.0 if source == "gaussian" else 90.0,
                    mean_target_sscd=prompt_scores[record],
                    step_index=step,
                    num_inference_steps=3,
                    evaluation_source=source,
                )
                row["timestep"] = timestep
                row["conditional_loss"] = loss
                row["normalized_loss_mse"] = loss / (
                    float(row["latent_dimension"]) * float(row["snr_t"])
                )
                row["normalized_loss_rmse"] = math.sqrt(
                    float(row["normalized_loss_mse"])
                )
                rows.append(row)
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).sample(
        frac=1.0, random_state=19
    ).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()
    captured = {}
    savefig_calls = []
    real_subplots = experiment.plt.subplots

    def capture(*args: object, **kwargs: object):
        figure, axis = real_subplots(*args, **kwargs)
        real_savefig = figure.savefig

        def savefig(*save_args: object, **save_kwargs: object) -> object:
            savefig_calls.append(dict(save_kwargs))
            return real_savefig(*save_args, **save_kwargs)

        monkeypatch.setattr(figure, "savefig", savefig)
        captured.update(
            figure=figure,
            axis=axis,
            font_family=tuple(experiment.matplotlib.rcParams["font.family"]),
            mathtext_fontset=experiment.matplotlib.rcParams["mathtext.fontset"],
        )
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", capture)
    monkeypatch.setattr(
        experiment, "_render_initial_loss_recovery", lambda **_kwargs: None
    )
    monkeypatch.setattr(experiment, "_render_pair_figure", lambda **_kwargs: None)
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="both")

    assert csv_path.read_bytes() == original_csv
    png_path, pdf_path = tuple(
        tmp_path / name for name in experiment.LOSS_TIMESTEP_FIGURE_FILENAMES
    )
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF-")
    assert set(tmp_path.iterdir()) == {csv_path, png_path, pdf_path}
    figure, axis = captured["figure"], captured["axis"]
    assert len(figure.axes) == 2  # One scientific panel plus SSCD colorbar.
    np.testing.assert_allclose(figure.get_size_inches(), [4.0, 4.0])
    assert captured["font_family"] == ("STIXGeneral",)
    assert captured["mathtext_fontset"] == "stix"
    assert axis.get_xlabel() == r"$t$"
    assert axis.get_ylabel() == r"$\mathcal{L}_t(c)$"
    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "log"
    assert not axis.xaxis_inverted()
    assert axis.get_title() == ""
    assert not axis.texts
    assert axis.xaxis.label.get_fontsize() == 15
    assert axis.yaxis.label.get_fontsize() == 15
    assert axis.xaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert axis.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
    assert all(label.get_fontsize() == 12 for label in axis.get_yticklabels())

    assert len(axis.collections) == 1  # Prompt lines, no scatter or bands.
    observations = axis.collections[0]
    assert isinstance(observations, LineCollection)
    assert observations.cmap.name == "viridis"
    assert observations.norm.vmin == 0.0
    assert observations.norm.vmax == 1.0
    assert observations.norm.clip is True
    assert observations.get_alpha() == pytest.approx(
        experiment.LOSS_TIMESTEP_LINE_ALPHA
    )
    assert observations.get_alpha() == pytest.approx(0.35)
    assert observations.get_alpha() < experiment.OBSERVATION_LINE_ALPHA
    np.testing.assert_allclose(
        observations.get_linewidths(), [experiment.OBSERVATION_LINE_WIDTH]
    )
    segments = observations.get_segments()
    colors = np.asarray(observations.get_array(), dtype=float)
    assert len(segments) == len(colors) == len(prompt_losses)
    np.testing.assert_allclose(np.sort(colors), sorted(prompt_scores.values()))
    for record, losses in prompt_losses.items():
        index = int(np.flatnonzero(np.isclose(colors, prompt_scores[record]))[0])
        np.testing.assert_allclose(segments[index][:, 0], timesteps[::-1])
        np.testing.assert_allclose(segments[index][:, 1], losses[::-1])
        assert np.all(np.diff(segments[index][:, 0]) > 0)
    np.testing.assert_allclose(axis.get_xlim(), [0.0, max(timesteps)])
    np.testing.assert_allclose(axis.get_xticks(), [0.0, max(timesteps)])
    assert [label.get_text() for label in axis.get_xticklabels()] == [r"$0$", r"$T$"]
    assert len(axis.get_xticks(minor=True)) == 0

    assert (
        len(axis.lines) == 1
    )  # Scheduler ratio reference, never fitted or normalized.
    reference = axis.lines[0]
    np.testing.assert_allclose(reference.get_xdata(), timesteps[::-1])
    expected_snr = [
        float(row["snr_t"])
        for row in rows
        if row["record_id"] == "selected-low" and row["evaluation_source"] == "gaussian"
    ][::-1]
    np.testing.assert_allclose(reference.get_ydata(), expected_snr)
    assert reference.get_color() == "black"
    assert reference.get_linestyle() == "--"
    assert reference.get_linewidth() == 1.0
    assert reference.get_alpha() == 1.0
    label = r"$\mathrm{SNR}_t$"
    assert reference.get_label() == label
    legend = axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [label]
    assert all(text.get_fontsize() == 10 for text in legend.get_texts())
    assert legend.get_frame_on() is False
    assert legend._loc == 3
    plotted_values = [
        *expected_snr,
        *(value for values in prompt_losses.values() for value in values),
    ]
    assert axis.get_ylim()[0] <= min(plotted_values)
    assert axis.get_ylim()[1] >= max(plotted_values)

    colorbar = next(item for item in figure.axes if item is not axis)
    assert colorbar.get_ylabel() == "SSCD"
    np.testing.assert_allclose(colorbar.get_ylim(), [0.0, 1.0])
    assert colorbar.yaxis.label.get_fontsize() == 15
    assert colorbar.yaxis.label.get_fontfamily() == ["STIXGeneral"]
    assert all(label.get_fontsize() == 12 for label in colorbar.get_yticklabels())
    meshes = [item for item in colorbar.collections if isinstance(item, QuadMesh)]
    assert len(meshes) == 1
    assert meshes[0].get_alpha() == 1.0
    figure.canvas.draw()
    np.testing.assert_allclose(meshes[0].get_facecolors()[:, 3], 1.0)
    assert savefig_calls == [
        {"format": extension, "bbox_inches": "tight", "pad_inches": 0.05, "dpi": 150}
        for extension in ("png", "pdf")
    ]


@pytest.mark.parametrize("loss", [math.nan, math.inf, -math.inf, -0.1])
def test_loss_timestep_plot_rejects_invalid_raw_loss(
    tmp_path: Path,
    loss: float,
) -> None:
    row = _row(
        "invalid-loss",
        normalized_loss_rmse=1.0,
        recovery_rmse=2.0,
        mean_target_sscd=0.5,
    )
    row["conditional_loss"] = loss
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame([row], columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    with pytest.raises(experiment.ExperimentError, match="conditional_loss"):
        experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="gaussian")
    assert list(tmp_path.iterdir()) == [csv_path]


@pytest.mark.parametrize("source_mode", ["gaussian", "trajectory", "both"])
def test_loss_timestep_plot_rejects_prompt_color_changing_between_timesteps(
    tmp_path: Path,
    source_mode: str,
) -> None:
    sources = ("gaussian", "trajectory") if source_mode == "both" else (source_mode,)
    rows = [
        _row(
            "changing-sscd",
            normalized_loss_rmse=1.0,
            recovery_rmse=2.0,
            mean_target_sscd=score,
            step_index=step,
            num_inference_steps=2,
            evaluation_source=source,
        )
        for source in sources
        for step, score in enumerate((0.2, 0.8))
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()
    with pytest.raises(experiment.ExperimentError, match="SSCD|sscd"):
        experiment.plot_saved_results(csv_path, tmp_path, evaluation_source=source_mode)
    assert csv_path.read_bytes() == original_csv


@pytest.mark.parametrize(
    "losses,expected_scale",
    [([0.0, 2.0], "symlog"), ([0.0, 0.0], "symlog"), ([0.0], "symlog"), ([2.0], "log")],
)
def test_loss_timestep_plot_preserves_zero_losses_and_single_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    losses: list[float],
    expected_scale: str,
) -> None:
    rows = [
        _row(
            "selected",
            normalized_loss_rmse=loss,
            recovery_rmse=2.0,
            mean_target_sscd=0.5,
            step_index=step,
            num_inference_steps=len(losses),
            evaluation_source="trajectory",
        )
        for step, loss in enumerate(losses)
    ]
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame(rows[::-1], columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    captured = {}
    real_subplots = experiment.plt.subplots

    def capture(*args: object, **kwargs: object):
        figure, axis = real_subplots(*args, **kwargs)
        captured["axis"] = axis
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", capture)
    monkeypatch.setattr(
        experiment, "_render_initial_loss_recovery", lambda **_kwargs: None
    )
    monkeypatch.setattr(experiment, "_render_pair_figure", lambda **_kwargs: None)
    monkeypatch.setattr(experiment, "_atomic_save_figures", lambda *_args: None)
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="trajectory")
    axis = captured["axis"]
    terminal_timestep = max(float(row["timestep"]) for row in rows)
    np.testing.assert_allclose(axis.get_xlim(), [0.0, terminal_timestep])
    np.testing.assert_allclose(axis.get_xticks(), [0.0, terminal_timestep])
    assert [label.get_text() for label in axis.get_xticklabels()] == [r"$0$", r"$T$"]
    assert len(axis.get_xticks(minor=True)) == 0
    assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
    assert axis.get_yscale() == expected_scale
    assert axis.get_xscale() == "linear"
    observations = [
        item for item in axis.collections if isinstance(item, LineCollection)
    ]
    assert len(observations) == 1
    segments = observations[0].get_segments()
    assert len(segments) == 1
    expected = [
        [float(row["timestep"]), float(row["conditional_loss"])] for row in rows[::-1]
    ]
    np.testing.assert_allclose(segments[0], expected)
    np.testing.assert_allclose(
        axis.lines[0].get_ydata(), [float(row["snr_t"]) for row in rows[::-1]]
    )
    if 0.0 in losses:
        assert axis.get_ylim()[0] <= 0.0 <= axis.get_ylim()[1]
    assert np.isfinite(axis.get_xlim()).all()
    assert np.isfinite(axis.get_ylim()).all()


def test_loss_timestep_plot_with_only_zero_timestep_does_not_invent_terminal_tick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row(
        "selected",
        normalized_loss_rmse=1.0,
        recovery_rmse=2.0,
        mean_target_sscd=0.5,
        evaluation_source="trajectory",
    )
    row["timestep"] = 0
    csv_path = tmp_path / experiment.CSV_NAME
    pd.DataFrame([row], columns=EXPECTED_COLUMNS).to_csv(csv_path, index=False)
    original_csv = csv_path.read_bytes()
    captured = {}
    real_subplots = experiment.plt.subplots

    def capture(*args: object, **kwargs: object):
        figure, axis = real_subplots(*args, **kwargs)
        captured["axis"] = axis
        return figure, axis

    monkeypatch.setattr(experiment.plt, "subplots", capture)
    monkeypatch.setattr(experiment, "_render_pair_figure", lambda **_kwargs: None)
    monkeypatch.setattr(experiment, "_atomic_save_figures", lambda *_args: None)
    experiment.plot_saved_results(csv_path, tmp_path, evaluation_source="trajectory")

    assert csv_path.read_bytes() == original_csv
    axis = captured["axis"]
    np.testing.assert_allclose(axis.get_xticks(), [0.0])
    assert [label.get_text() for label in axis.get_xticklabels()] == [r"$0$"]
    assert all(label.get_fontsize() == 12 for label in axis.get_xticklabels())
    assert len(axis.get_xticks(minor=True)) == 0
    assert np.isfinite(axis.get_xlim()).all()
    assert axis.get_xlim()[0] < axis.get_xlim()[1]
    assert axis.get_xlim()[0] <= 0.0 <= axis.get_xlim()[1]
    observations = [
        item for item in axis.collections if isinstance(item, LineCollection)
    ]
    assert len(observations) == 1
    np.testing.assert_allclose(
        observations[0].get_segments()[0], [[0.0, float(row["conditional_loss"])]]
    )
    assert len(axis.lines) == 1
    np.testing.assert_allclose(axis.lines[0].get_xdata(), [0.0])
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [float(row["snr_t"])])


def test_failed_figure_staging_preserves_both_outputs_and_removes_staged_files(
    tmp_path: Path,
) -> None:
    destinations = experiment._figure_paths(tmp_path, "gaussian")
    original_contents = (b"original PNG", b"original PDF")
    for destination, contents in zip(destinations, original_contents, strict=True):
        destination.write_bytes(contents)

    def fail_on_pdf(destination: Path, **options: object) -> None:
        Path(destination).write_bytes(b"staged figure")
        if options["format"] == "pdf":
            raise KeyboardInterrupt("synthetic PDF failure")

    figure = SimpleNamespace(savefig=fail_on_pdf)
    with pytest.raises(KeyboardInterrupt, match="synthetic PDF failure"):
        experiment._atomic_save_figures(figure, destinations)

    assert tuple(destination.read_bytes() for destination in destinations) == (
        original_contents
    )
    assert set(tmp_path.iterdir()) == set(destinations)


def test_partial_figure_install_rolls_back_both_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destinations = experiment._figure_paths(tmp_path, "gaussian")
    original_contents = (b"original PNG", b"original PDF")
    for destination, contents in zip(destinations, original_contents, strict=True):
        destination.write_bytes(contents)

    def save_figure(destination: Path, **options: object) -> None:
        Path(destination).write_bytes(f"new {options['format']}".encode())

    real_replace = experiment.os.replace
    failed = False

    def fail_second_install(source: Path, destination: Path) -> None:
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not failed
            and destination_path == destinations[1]
            and source_path.name.startswith(f".{destinations[1].name}.")
        ):
            failed = True
            raise OSError("synthetic second-install failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(experiment.os, "replace", fail_second_install)
    figure = SimpleNamespace(savefig=save_figure)
    with pytest.raises(OSError, match="synthetic second-install failure"):
        experiment._atomic_save_figures(figure, destinations)

    assert tuple(destination.read_bytes() for destination in destinations) == (
        original_contents
    )
    assert set(tmp_path.iterdir()) == set(destinations)


def test_plot_uses_saved_csv_without_loading_diffusion_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "plot"
    output.mkdir()
    contract = SimpleNamespace(
        scientific_hash="a" * 64,
        schedule_sha256="b" * 64,
        paths=SimpleNamespace(run_directory=tmp_path / "generation"),
    )
    log_directory = experiment._default_log_directory(
        contract, SELECTION_HASH, 2, 123, "gaussian"
    )
    log_directory.mkdir(parents=True)
    csv_path = log_directory / experiment.CSV_NAME
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
        raise AssertionError("plot mode must not load or run the diffusion model")

    monkeypatch.setattr(experiment, "load_model_components", forbidden)
    monkeypatch.setattr(experiment, "preflight_model_components", forbidden)
    monkeypatch.setattr(experiment, "WebsterDataset", forbidden)
    monkeypatch.setattr(experiment, "run_experiment", forbidden)
    monkeypatch.setattr(experiment, "resolve_devices", forbidden)
    monkeypatch.setattr(
        experiment,
        "_load_generation_contract",
        lambda *_args, **_kwargs: contract,
    )
    selection_hash = SELECTION_HASH
    monkeypatch.setattr(
        experiment,
        "_load_frozen_selection",
        lambda *_args, **kwargs: _selection(
            included=("0", "1"),
            strategy=str(kwargs["selection_strategy"]),
            selection_hash=selection_hash,
        ),
    )
    with pytest.raises(
        experiment.ExperimentError,
        match="saved Theorem 1 log CSV is missing or unsafe",
    ):
        experiment.main(
            [
                "--plot",
                "--T",
                "1",
                "--evaluation-source",
                "gaussian",
                "--selection-strategy",
                "gmm",
                "--N",
                "2",
                "--num-loss-seeds",
                "2",
                "--loss-seed",
                "124",
                "--device",
                "cuda:999",
                "--output-dir",
                str(output),
            ]
        )
    selection_hash = "d" * 64
    with pytest.raises(
        experiment.ExperimentError,
        match="saved Theorem 1 log CSV is missing or unsafe",
    ):
        experiment.main(
            [
                "--plot",
                "--T",
                "1",
                "--evaluation-source",
                "gaussian",
                "--selection-strategy",
                "gmm",
                "--N",
                "2",
                "--num-loss-seeds",
                "2",
                "--loss-seed",
                "123",
                "--output-dir",
                str(output),
            ]
        )
    selection_hash = SELECTION_HASH
    validated_logs: list[Path] = []

    def validate_log(
        directory: Path, *_args: object, **_kwargs: object
    ) -> pd.DataFrame:
        validated_logs.append(Path(directory))
        return pd.read_csv(
            Path(directory) / experiment.CSV_NAME,
            keep_default_na=False,
            float_precision="round_trip",
        )

    monkeypatch.setattr(experiment, "_validated_plot_log_rows", validate_log)
    assert (
        experiment.main(
            [
                "--plot",
                "--T",
                "1",
                "--evaluation-source",
                "gaussian",
                "--selection-strategy",
                "gmm",
                "--N",
                "2",
                "--num-loss-seeds",
                "2",
                "--loss-seed",
                "123",
                "--device",
                "cuda:999",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert validated_logs == [log_directory]
    assert csv_path.read_bytes() == original_csv
    pd.testing.assert_frame_equal(
        pd.read_csv(output / experiment.CSV_NAME, keep_default_na=False),
        pd.read_csv(csv_path, keep_default_na=False),
        check_dtype=False,
    )
    assert {path.name for path in output.iterdir()} == {
        experiment.CSV_NAME,
        *(path.name for path in experiment._figure_paths(output, "gaussian")),
        *experiment.LOSS_TIMESTEP_FIGURE_FILENAMES,
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
    assert ".svg" not in lowered
    assert "generation_seed" not in experiment.CSV_COLUMNS
    assert "seed_sscd" not in experiment.CSV_COLUMNS


def _measurement_cache_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, dict[str, object], tuple[tuple[int, dict[str, object]], ...]]:
    template = _row(
        "unused",
        normalized_loss_rmse=1.0,
        recovery_rmse=2.0,
        mean_target_sscd=0.25,
    )
    contract = SimpleNamespace(
        paths=SimpleNamespace(run_directory=tmp_path / "generation"),
        parameters=(
            experiment.TerminalParameters(
                timestep=int(template["timestep"]),
                alpha_t=float(template["alpha_t"]),
                sigma_t=float(template["sigma_t"]),
                snr_t=float(template["snr_t"]),
            ),
        ),
        latent_shape=(1, 2, 2),
        sscd_configuration_hash="e" * 64,
    )
    expected = experiment._expected_csv_configuration(
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=1,
        num_loss_seeds=2,
        loss_seed=123,
        num_seeds=2,
        selection_strategy="gmm",
        selection_hash=SELECTION_HASH,
        evaluation_source="gaussian",
        scientific_hash="a" * 64,
        schedule_sha256="b" * 64,
    )
    entries = tuple(
        (
            position,
            {
                "original_index": str(10 + position),
                "record_id": f"sdv1-000{position}",
                "source_row_number": 10 + position,
                "prompt": f"prompt {position}",
                "prompt_raw": f"prompt {position}",
                "target_image_sha256": f"{position + 1}" * 64,
            },
        )
        for position in range(2)
    )
    monkeypatch.setattr(
        experiment,
        "_pair_cache_fingerprint",
        lambda _contract, metadata: str(metadata["original_index"]) * 64,
    )
    return contract, expected, entries


def _successful_cache_rows(record_id: str) -> list[dict[str, object]]:
    return [
        _row(
            record_id,
            normalized_loss_rmse=1.0,
            recovery_rmse=2.0,
            mean_target_sscd=0.25,
        )
    ]


def test_measurement_cache_resumes_only_complete_prompt_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, expected, entries = _measurement_cache_inputs(tmp_path, monkeypatch)
    cache, is_new = experiment._prepare_measurement_cache(
        contract, expected, entries, overwrite=False
    )
    assert is_new is True
    first, second = entries[0][1], entries[1][1]
    experiment._save_pair_checkpoint(
        cache, first, _successful_cache_rows(str(first["record_id"]))
    )
    failed = _successful_cache_rows(str(second["record_id"]))
    failed[0].update(status="error", error="synthetic interruption")
    experiment._save_pair_checkpoint(cache, second, failed)

    resumed = experiment._load_pair_checkpoint(cache, first)
    assert resumed is not None
    assert len(resumed) == 1
    assert resumed[0]["record_id"] == first["record_id"]
    assert resumed[0]["status"] == "ok"
    assert experiment._load_pair_checkpoint(cache, second) is None
    assert (cache.directory / "record/10.json").is_file()
    assert not (cache.directory / "record/11.json").exists()


def test_measurement_cache_overwrite_removes_records_and_aggregate_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, expected, entries = _measurement_cache_inputs(tmp_path, monkeypatch)
    cache, _ = experiment._prepare_measurement_cache(
        contract, expected, entries, overwrite=False
    )
    for _, metadata in entries:
        experiment._save_pair_checkpoint(
            cache, metadata, _successful_cache_rows(str(metadata["record_id"]))
        )
    aggregate = cache.directory / experiment.CSV_NAME
    aggregate.write_text("old aggregate", encoding="utf-8")

    reused, is_new = experiment._prepare_measurement_cache(
        contract, expected, entries, overwrite=False
    )
    assert is_new is False
    assert all(
        experiment._load_pair_checkpoint(reused, metadata) is not None
        for _, metadata in entries
    )

    overwritten, is_new = experiment._prepare_measurement_cache(
        contract, expected, entries, overwrite=True
    )
    assert is_new is False
    assert overwritten.directory == cache.directory
    assert not aggregate.exists()
    assert not list((cache.directory / "record").glob("*.json"))
    assert (cache.directory / "run_config.json").is_file()


def test_default_measurement_log_directory_isolated_by_every_cache_axis(
    tmp_path: Path,
) -> None:
    contract = SimpleNamespace(
        paths=SimpleNamespace(run_directory=tmp_path / "generation")
    )
    baseline = experiment._default_log_directory(
        contract, SELECTION_HASH, 20, 0, "both"
    )
    assert baseline == (
        tmp_path
        / "generation"
        / "theorem1_loss_recovery"
        / SELECTION_HASH
        / "loss_S0_N20"
        / "evaluation_both"
    )
    variants = {
        experiment._default_log_directory(contract, "d" * 64, 20, 0, "both"),
        experiment._default_log_directory(contract, SELECTION_HASH, 21, 0, "both"),
        experiment._default_log_directory(contract, SELECTION_HASH, 20, 1, "both"),
        experiment._default_log_directory(contract, SELECTION_HASH, 20, 0, "gaussian"),
        experiment._default_log_directory(
            contract, SELECTION_HASH, 20, 0, "trajectory"
        ),
    }
    assert len(variants) == 5
    assert baseline not in variants
