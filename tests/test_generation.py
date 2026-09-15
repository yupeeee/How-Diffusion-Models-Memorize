"""Offline generation-cache and flattened-architecture tests."""

from __future__ import annotations

import argparse
import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image
from scripts import generate as generate_script

from utils.common.cli import (
    MAX_SEED,
    add_generation_arguments,
    add_run_arguments,
    generation_cache_namespace,
    generation_cache_parent_name,
    generation_run_name,
)
from utils.common.io import (
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.experiments import cache as cache_module
from utils.experiments import generation as generation_module
from utils.experiments.cache import (
    GENERATION_SELECTION_POLICY,
    GENERATION_SCHEMA_VERSION,
    SAMPLER_CONTRACT_VERSION,
    CacheValidation,
    DiskEstimate,
    GenerationCacheError,
    GenerationPaths,
    generation_log_relative_path,
    generation_paths,
    list_completed_records,
    publish_completion_marker,
    require_generation_run,
    save_generation_tensors,
    validate_generation_record,
)
from utils.models.latent import TARGET_LATENT_DEFINITION
from utils.models.registry import get_model_spec


ROOT = Path(__file__).resolve().parents[1]


def test_core_generation_apis_require_every_run_setting() -> None:
    functions = (
        generation_module.generate_webster_trajectories,
        generation_run_name,
        generation_cache_parent_name,
        generation_cache_namespace,
        generation_log_relative_path,
        generation_paths,
    )

    for function in functions:
        parameters = inspect.signature(function).parameters
        assert parameters
        assert all(
            parameter.default is inspect.Parameter.empty
            for parameter in parameters.values()
        ), function.__name__


def test_shared_cli_owns_generation_downscale_parsing() -> None:
    generation_parser = add_generation_arguments(
        argparse.ArgumentParser(allow_abbrev=False)
    )
    defaults = generation_parser.parse_args([])
    assert vars(defaults) == {
        "model": "sdv1",
        "scheduler": "ddim",
        "g": 7.5,
        "T": 50,
        "N": 20,
        "seed_start": 0,
        "downscale": 4,
        "device": "auto",
        "overwrite": False,
    }
    assert generation_parser.parse_args(["--overwrite"]).overwrite is True
    assert generation_parser.parse_args(["--downscale", "8"]).downscale == 8
    assert generation_parser.parse_args(["--seed-start", "20"]).seed_start == 20
    assert generation_parser.parse_args(["--device", "cuda:1"]).device == "cuda:1"
    with pytest.raises(SystemExit):
        generation_parser.parse_args(["--downscale", "0"])
    with pytest.raises(SystemExit):
        generation_parser.parse_args(["--seed-start", "-1"])
    with pytest.raises(SystemExit):
        generation_parser.parse_args(["--seed-start", str(MAX_SEED + 1)])

    analysis_parser = add_run_arguments(argparse.ArgumentParser(allow_abbrev=False))
    assert not hasattr(analysis_parser.parse_args([]), "downscale")
    cache_only_parser = add_run_arguments(
        argparse.ArgumentParser(allow_abbrev=False), include_device=False
    )
    assert not hasattr(cache_only_parser.parse_args([]), "device")


def test_generation_shards_keep_duplicate_target_images_on_one_device() -> None:
    entries = tuple(
        (
            position,
            {
                "original_index": str(position),
                "target_image_sha256": digest,
            },
        )
        for position, digest in enumerate(("a", "b", "a", "c", "b", "d"))
    )

    shards = generation_module._target_group_shards(entries, 3)

    assert len(shards) == 3
    assert sorted(position for shard in shards for position, _ in shard) == list(
        range(len(entries))
    )
    owners = {
        digest: {
            worker_index
            for worker_index, shard in enumerate(shards)
            for _, metadata in shard
            if metadata["target_image_sha256"] == digest
        }
        for digest in {"a", "b", "c", "d"}
    }
    assert all(len(worker_owners) == 1 for worker_owners in owners.values())
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.parametrize("overwrite", (False, True))
def test_generation_uses_one_disjoint_shard_per_auto_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    overwrite: bool,
) -> None:
    rows = tuple(
        {
            "original_index": str(100 + position),
            "record_id": f"sdv1-{position:04d}",
            "source_row_number": 4 - position,
            "prompt_raw": f"prompt {position}",
            "target_image_sha256": f"{position:064x}",
            "overfit_type": "TV" if position % 2 == 0 else "N",
        }
        for position in range(5)
    )

    class Dataset:
        total_manifest_rows = len(rows)

        def __len__(self) -> int:
            return len(rows)

        def iter_metadata(self):
            yield from (dict(row) for row in rows)

    configuration = {
        "scientific_config_hash": "a" * 64,
        "preview_config_hash": "b" * 64,
    }
    parent_runtime = object()
    progress_lock = generation_module.tqdm.get_lock()
    spawn_context = SimpleNamespace(RLock=lambda: progress_lock)
    events: list[tuple[object, ...]] = []
    observed: list[tuple[str, tuple[int, ...], int]] = []

    def shard_result(
        entries: tuple[tuple[int, dict[str, object]], ...],
        selected: str,
        worker_index: int,
    ) -> generation_module._GenerationShardResult:
        positions = tuple(position for position, _ in entries)
        observed.append((selected, positions, worker_index))
        manifest_rows = tuple(
            {
                "original_index": metadata["original_index"],
                "source_row_number": metadata["source_row_number"],
                "webster_overfit_type": metadata["overfit_type"],
            }
            for _, metadata in entries
        )
        statuses = {
            str(metadata["original_index"]): "generated" for _, metadata in entries
        }
        return generation_module._GenerationShardResult(
            manifest_rows,
            (),
            (),
            statuses,
            dict(configuration),
        )

    def initialize(**_: object) -> tuple[dict[str, object], object]:
        events.append(("initialize",))
        return dict(configuration), parent_runtime

    def local_shard(**kwargs: object):
        entries = kwargs["entries"]
        selected = kwargs["device"]
        worker_index = kwargs["worker_index"]
        assert kwargs["runtime"] is parent_runtime
        assert isinstance(entries, tuple)
        assert isinstance(selected, torch.device)
        assert isinstance(worker_index, int)
        events.append(("local", str(selected)))
        return shard_result(entries, str(selected), worker_index), parent_runtime

    def subprocess_shard(*arguments: object):
        entries = arguments[7]
        selected = arguments[9]
        worker_index = arguments[10]
        overwrite = arguments[12]
        assert isinstance(entries, tuple)
        assert isinstance(selected, str)
        assert isinstance(worker_index, int)
        assert overwrite is expected_overwrite
        events.append(("child", selected))
        return shard_result(entries, selected, worker_index)

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
            assert initializer is generation_module._install_tqdm_lock
            assert initargs == (progress_lock,)
            events.append(("executor", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, *arguments: object) -> None:
            del arguments

        def submit(self, function: object, *arguments: object) -> ImmediateFuture:
            assert function is subprocess_shard
            return ImmediateFuture(subprocess_shard(*arguments))

    expected_overwrite = overwrite
    monkeypatch.setattr(generation_module, "_load_dataset", lambda *_: Dataset())
    monkeypatch.setattr(
        generation_module,
        "resolve_devices",
        lambda _: (
            torch.device("cuda:0"),
            torch.device("cuda:1"),
            torch.device("cuda:2"),
        ),
    )
    monkeypatch.setattr(
        generation_module,
        "estimate_disk_space",
        lambda *_, **__: DiskEstimate(5, 1, 5, 0, 10**12),
    )
    monkeypatch.setattr(
        generation_module,
        "_initialize_generation_configuration",
        initialize,
    )
    monkeypatch.setattr(generation_module, "_run_generation_shard", local_shard)
    monkeypatch.setattr(generation_module, "_generation_subprocess", subprocess_shard)
    monkeypatch.setattr(generation_module, "get_context", lambda _: spawn_context)
    monkeypatch.setattr(generation_module, "ProcessPoolExecutor", ImmediateExecutor)

    result = generation_module.generate_webster_trajectories(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
        downscale=4,
        device="auto",
        overwrite=overwrite,
    )

    assert result.exit_code == 0
    assert result.completed_rows == 5
    assert events == [
        ("initialize",),
        ("executor", 2),
        ("child", "cuda:1"),
        ("child", "cuda:2"),
        ("local", "cuda:0"),
    ]
    assert observed == [
        ("cuda:1", (1, 4), 1),
        ("cuda:2", (2,), 2),
        ("cuda:0", (0, 3), 0),
    ]
    paths = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
    )
    manifest = pd.read_parquet(paths.manifest_parquet)
    assert manifest["source_row_number"].tolist() == [0, 1, 2, 3, 4]
    summary = read_json(paths.summary_json)
    assert summary["execution_devices"] == ["cuda:0", "cuda:1", "cuda:2"]
    assert summary["worker_count"] == 3
    assert summary["overwrite"] is overwrite
    assert summary["device_shards"] == [
        {
            "device": "cuda:0",
            "assigned_rows": 2,
            "completed_rows": 2,
            "newly_generated_rows": 2,
            "fully_resumed_rows": 0,
            "preview_only_regenerated_rows": 0,
            "skipped_rows": 0,
            "failed_rows": 0,
        },
        {
            "device": "cuda:1",
            "assigned_rows": 2,
            "completed_rows": 2,
            "newly_generated_rows": 2,
            "fully_resumed_rows": 0,
            "preview_only_regenerated_rows": 0,
            "skipped_rows": 0,
            "failed_rows": 0,
        },
        {
            "device": "cuda:2",
            "assigned_rows": 1,
            "completed_rows": 1,
            "newly_generated_rows": 1,
            "fully_resumed_rows": 0,
            "preview_only_regenerated_rows": 0,
            "skipped_rows": 0,
            "failed_rows": 0,
        },
    ]
    startup = capsys.readouterr().out
    assert "cuda:0=2 records, cuda:1=2 records, cuda:2=1 records" in startup
    assert "Denoising covers only the current record" in startup


def test_fresh_generation_configuration_skips_a_failed_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    devices = tuple(torch.device(f"cuda:{index}") for index in range(3))
    calls: list[torch.device] = []
    configuration = {"scientific_config_hash": "a" * 64}
    runtime = object()

    def initialize(**kwargs: object):
        selected = kwargs["device"]
        assert isinstance(selected, torch.device)
        calls.append(selected)
        if selected == devices[0]:
            raise RuntimeError("synthetic CUDA initialization failure")
        return dict(configuration), runtime

    monkeypatch.setattr(
        generation_module,
        "_initialize_generation_configuration",
        initialize,
    )

    observed_config, observed_runtime, active_devices = (
        generation_module._initialize_generation_configuration_with_failover(
            root=tmp_path,
            paths=GenerationPaths(tmp_path / "logs" / "synthetic"),
            dataset=object(),
            entries=(),
            spec=object(),
            scheduler_name="ddim",
            guidance=7.5,
            steps=50,
            seed_values=tuple(range(20)),
            downscale=4,
            devices=devices,
        )
    )

    assert observed_config == configuration
    assert observed_runtime is runtime
    assert calls == [devices[0], devices[1]]
    assert active_devices == devices[1:]
    assert "Initialization failed on cuda:0" in capsys.readouterr().err


def test_seed_blocks_have_collision_free_shared_log_namespaces(
    tmp_path: Path,
) -> None:
    common = ("sdv1", "ddim", 7.5, 50, 4)
    assert generation_run_name(*common, seed_start=0) == "sdv1_ddim_g7.5_T50_N4"
    assert generation_run_name(*common, seed_start=4) == "sdv1_ddim_g7.5_T50_S4_N4"
    zero = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=4,
        seed_start=0,
    )
    reference = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=4,
        seed_start=4,
    )
    parent = tmp_path.resolve() / "logs/sdv1_ddim_g7.5_T50_N4"
    assert zero.run_directory == parent / "experiment_S0_N4"
    assert reference.run_directory == parent / "reference_S4_N4"
    assert zero.run_directory.parent == reference.run_directory.parent
    assert zero.run_directory != reference.run_directory
    noncanonical_reference = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=12,
        num_seeds=4,
        seed_start=4,
    )
    assert noncanonical_reference.run_directory == (
        tmp_path.resolve() / "logs/sdv1_ddpm_g3.25_T12_N4/reference_S4_N4"
    )
    generic_seed_block = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=3.25,
        num_inference_steps=12,
        num_seeds=4,
        seed_start=2,
    )
    assert generic_seed_block.run_directory == (
        tmp_path.resolve() / "logs/sdv1_ddpm_g3.25_T12_N4/seed_S2_N4"
    )
    with pytest.raises(ValueError, match="seed block must end"):
        generation_run_name(*common[:-1], 2, seed_start=MAX_SEED)


@pytest.mark.parametrize("script_name", ["generate.py", "sscd.py"])
def test_seed_block_cli_rejects_an_out_of_range_terminal_seed(
    script_name: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script_name),
            "--N",
            "2",
            "--seed-start",
            str(MAX_SEED),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "seed block must end at or before" in result.stderr
    assert "Traceback" not in result.stderr


def test_generation_rejects_an_out_of_range_seed_block_before_writing(
    tmp_path: Path,
) -> None:
    with pytest.raises(generation_module.GenerationError, match="seed block must end"):
        generation_module.generate_webster_trajectories(
            tmp_path,
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=50,
            num_seeds=2,
            seed_start=MAX_SEED,
            downscale=4,
            device="cpu",
            overwrite=False,
        )

    assert not (tmp_path / "logs").exists()


def _small_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = torch.arange(24, dtype=torch.float32).reshape(2, 3, 1, 2, 2)
    unconditional = torch.zeros((2, 2, 1, 2, 2), dtype=torch.float32)
    conditional = torch.ones((2, 2, 1, 2, 2), dtype=torch.float32)
    target = torch.full((1, 2, 2), 0.5, dtype=torch.float32)
    return latent, unconditional, conditional, target


def _write_cached_statistics_artifacts(
    project: Path,
    run_directory: Path,
    *,
    science_hash: str,
    rows: int,
    marker_fingerprint: str,
) -> tuple[Path, Path, Path]:
    directory = run_directory / "target_latent_statistics"
    mean_path = directory / "mean.pt"
    std_path = directory / "population_std.pt"
    mean_hash = atomic_torch_save(
        torch.zeros((1, 1, 1), dtype=torch.float64), mean_path
    )
    std_hash = atomic_torch_save(torch.ones((1, 1, 1), dtype=torch.float64), std_path)
    report_path = directory / "report.json"
    atomic_write_json(
        report_path,
        {
            "schema_version": 1,
            "artifact": "target_latent_statistics",
            "completed_target_latent_rows": rows,
            "scientific_config_hash": science_hash,
            "generation_target_latent_marker_fingerprint_sha256": (marker_fingerprint),
            "latent_shape": [1, 1, 1],
            "coordinate_artifacts": {
                "mean": {
                    "path": mean_path.relative_to(project).as_posix(),
                    "sha256": mean_hash,
                },
                "population_std": {
                    "path": std_path.relative_to(project).as_posix(),
                    "sha256": std_hash,
                },
            },
        },
    )
    return report_path, mean_path, std_path


def test_generation_entrypoint_skips_only_complete_cached_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_directory = tmp_path / "logs" / "synthetic"
    run_directory.mkdir(parents=True)
    science_hash = "a" * 64
    atomic_write_json(
        run_directory / "run_config.json",
        {"scientific_config_hash": science_hash},
    )
    paths = GenerationPaths(run_directory)
    paths.create()
    target_hash = atomic_torch_save(
        torch.zeros((1, 1, 1), dtype=torch.float32),
        paths.target_latent_path("1"),
    )
    publish_completion_marker(
        paths,
        "1",
        {
            "source_row_number": 0,
            "target_image_sha256": "b" * 64,
            "tensor_file_sha256": {"target_latent": target_hash},
        },
    )
    from utils.experiments.latent_statistics import target_latent_marker_fingerprint

    marker_fingerprint = target_latent_marker_fingerprint(list_completed_records(paths))
    _, _, std_path = _write_cached_statistics_artifacts(
        tmp_path,
        run_directory,
        science_hash=science_hash,
        rows=1,
        marker_fingerprint=marker_fingerprint,
    )
    result = generation_module.GenerationResult(
        run_directory / "summary.json", 1, 0, True
    )
    calls: list[bool] = []

    def generate(*_args: object, **kwargs: object) -> object:
        calls.append(bool(kwargs["overwrite"]))
        return result

    monkeypatch.setattr(generate_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generation_module, "generate_webster_trajectories", generate)
    from utils.experiments import latent_statistics

    monkeypatch.setattr(
        latent_statistics,
        "compute_target_latent_statistics",
        lambda *_: pytest.fail("complete cached statistics must not be recomputed"),
    )
    assert generate_script.main([]) == 0
    assert calls == [False]
    assert "Target latent statistics skipped" in capsys.readouterr().out

    marker_path = paths.record_path("1")
    marker = read_json(marker_path)
    marker["preview_downscale"] = 8
    atomic_write_json(marker_path, marker)
    assert generate_script.main([]) == 0
    assert calls == [False, False]
    assert "Target latent statistics skipped" in capsys.readouterr().out

    target_path = paths.target_latent_path("1")
    original_target_bytes = target_path.read_bytes()
    target_path.write_bytes(b"corrupt target latent")
    assert generate_script._cached_statistics_report(run_directory, 1) is None
    target_path.write_bytes(original_target_bytes)
    assert (
        generate_script._cached_statistics_report(run_directory, 1)
        == run_directory / "target_latent_statistics" / "report.json"
    )

    replacement_hash = atomic_torch_save(
        torch.ones((1, 1, 1), dtype=torch.float32),
        target_path,
    )
    marker_hashes = marker["tensor_file_sha256"]
    assert isinstance(marker_hashes, dict)
    marker_hashes["target_latent"] = replacement_hash
    atomic_write_json(marker_path, marker)
    computed: list[bool] = []
    rebuilt = SimpleNamespace(report={}, report_path=std_path.parent / "report.json")
    monkeypatch.setattr(
        latent_statistics,
        "compute_target_latent_statistics",
        lambda *_: computed.append(True) or rebuilt,
    )
    monkeypatch.setattr(
        latent_statistics,
        "format_target_latent_statistics",
        lambda _: "rebuilt statistics",
    )
    assert generate_script.main([]) == 0
    assert computed == [True]
    assert calls == [False, False, False]
    assert "report is absent; computing it now" in capsys.readouterr().out


def _science(
    *, steps: int = 50, seeds: int = 20, seed_start: int = 0
) -> dict[str, object]:
    spec = get_model_spec("sdv1")
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "sampler_contract_version": SAMPLER_CONTRACT_VERSION,
        "selection_policy": GENERATION_SELECTION_POLICY,
        "target_role": "canonical_paired_source",
        "model_cli_name": "sdv1",
        "dataset_model": "sdv1",
        "model_id": spec.model_id,
        "model_revision": "a" * 40,
        "vae_id": spec.vae_id or spec.model_id,
        "vae_revision": "b" * 40,
        "resolution": spec.resolution,
        "scheduler": {
            "name": "ddim",
            "class": "SyntheticScheduler",
            "config": {"prediction_type": "epsilon"},
        },
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "guidance_scale": 7.5,
        "num_inference_steps": steps,
        "num_seeds": seeds,
        "seeds": list(range(seed_start, seed_start + seeds)),
        "latent_shape": [1, 1, 1],
        "inference_dtype": "float32",
        "trajectory_order": "noise_to_image",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
        "target_preprocessing": {"policy": "synthetic"},
    }


def _schedule(steps: int = 50) -> dict[str, object]:
    cumulative = torch.linspace(0.99, 0.50, steps, dtype=torch.float32)
    return {
        "timesteps": torch.arange(steps - 1, -1, -1, dtype=torch.int64),
        "alpha_t": cumulative.sqrt(),
        "sigma_t": (1.0 - cumulative).sqrt(),
        "alphas_cumprod_t": cumulative,
        "init_noise_sigma": 1.0,
        "prediction_type": "epsilon",
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "scheduler_name": "ddim",
        "scheduler_class": "SyntheticScheduler",
        "scheduler_config": {"prediction_type": "epsilon"},
        "trajectory_order": "noise_to_image",
    }


def _publish_cached_record(
    paths: GenerationPaths,
    identity: dict[str, object],
    science_hash: str,
    *,
    steps: int = 50,
    seeds: int = 20,
    seed_start: int = 0,
    downscale: int = 4,
) -> None:
    index = str(identity["original_index"])
    latent = torch.zeros((seeds, steps + 1, 1, 1, 1), dtype=torch.float32)
    unconditional = torch.zeros((seeds, steps, 1, 1, 1), dtype=torch.float32)
    conditional = torch.ones_like(unconditional)
    target = torch.zeros((1, 1, 1), dtype=torch.float32)
    hashes = save_generation_tensors(
        paths,
        index,
        latents=latent,
        unconditional_predictions=unconditional,
        conditional_predictions=conditional,
        target_latent=target,
    )
    paths.image_path(index).write_bytes(b"cached preview")
    publish_completion_marker(
        paths,
        index,
        {
            **identity,
            "created_at": "2026-01-01T00:00:00+00:00",
            "webster_overfit_type": identity["overfit_type"],
            "recovery_status": "recovered_exact_url",
            "recovery_method": "direct",
            "model_cli_name": "sdv1",
            "dataset_model": "sdv1",
            "model_id": get_model_spec("sdv1").model_id,
            "model_revision": "a" * 40,
            "vae_id": get_model_spec("sdv1").vae_id or get_model_spec("sdv1").model_id,
            "vae_revision": "b" * 40,
            "scientific_config_hash": science_hash,
            "scheduler_name": "ddim",
            "scheduler_class": "SyntheticScheduler",
            "guidance_scale": 7.5,
            "num_inference_steps": steps,
            "num_seeds": seeds,
            "seeds": list(range(seed_start, seed_start + seeds)),
            "latent_shape": [1, 1, 1],
            "inference_dtype": "float32",
            "native_prediction_type": "epsilon",
            "stored_prediction_type": "epsilon",
            "trajectory_order": "noise_to_image",
            "target_latent_definition": TARGET_LATENT_DEFINITION,
            "target_preprocessing": {"policy": "synthetic"},
            "latent_path": f"latent/{index}.pt",
            "noise_prediction_path": f"noise_pred/{index}.pt",
            "target_latent_path": f"target_latent/{index}.pt",
            "preview_image_path": f"image/{index}.png",
            "schedule_path": "schedule.pt",
            "tensor_file_sha256": hashes,
            "preview_image_sha256": file_sha256(paths.image_path(index)),
            "preview_downscale": downscale,
            "preview_status": "complete",
            "tensor_shapes": {
                "latent": list(latent.shape),
                "unconditional_noise_predictions": list(unconditional.shape),
                "conditional_noise_predictions": list(conditional.shape),
                "target_latent": list(target.shape),
            },
            "tensor_dtypes": {
                "latent": "float32",
                "unconditional_noise_predictions": "float32",
                "conditional_noise_predictions": "float32",
                "target_latent": "float32",
            },
        },
    )


def test_scientific_tensors_are_write_once_and_branch_order_is_explicit(
    tmp_path: Path,
) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    latent, unconditional, conditional, target = _small_tensors()
    hashes = save_generation_tensors(
        paths,
        "7",
        latents=latent,
        unconditional_predictions=unconditional,
        conditional_predictions=conditional,
        target_latent=target,
    )
    stored_branches = safe_torch_load(paths.noise_prediction_path("7"))
    assert isinstance(stored_branches, tuple) and len(stored_branches) == 2
    torch.testing.assert_close(stored_branches[0], unconditional)
    torch.testing.assert_close(stored_branches[1], conditional)
    before = {
        name: file_sha256(path)
        for name, path in {
            "latent": paths.latent_path("7"),
            "noise_prediction": paths.noise_prediction_path("7"),
            "target_latent": paths.target_latent_path("7"),
        }.items()
    }

    with pytest.raises(GenerationCacheError, match="refusing to overwrite"):
        save_generation_tensors(
            paths,
            "7",
            latents=latent + 1,
            unconditional_predictions=unconditional + 1,
            conditional_predictions=conditional + 1,
            target_latent=target + 1,
        )
    assert hashes == before
    assert before == {
        name: file_sha256(path)
        for name, path in {
            "latent": paths.latent_path("7"),
            "noise_prediction": paths.noise_prediction_path("7"),
            "target_latent": paths.target_latent_path("7"),
        }.items()
    }


def test_explicit_overwrite_atomically_replaces_tensors_and_marker(
    tmp_path: Path,
) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    latent, unconditional, conditional, target = _small_tensors()
    save_generation_tensors(
        paths,
        "7",
        latents=latent,
        unconditional_predictions=unconditional,
        conditional_predictions=conditional,
        target_latent=target,
    )
    publish_completion_marker(paths, "7", {"revision": "original"})

    replacements = save_generation_tensors(
        paths,
        "7",
        latents=latent + 1,
        unconditional_predictions=unconditional + 2,
        conditional_predictions=conditional + 3,
        target_latent=target + 4,
        overwrite=True,
    )
    publish_completion_marker(
        paths,
        "7",
        {"revision": "replacement", "tensor_file_sha256": replacements},
        overwrite=True,
    )

    torch.testing.assert_close(safe_torch_load(paths.latent_path("7")), latent + 1)
    stored_predictions = safe_torch_load(paths.noise_prediction_path("7"))
    assert isinstance(stored_predictions, tuple)
    torch.testing.assert_close(stored_predictions[0], unconditional + 2)
    torch.testing.assert_close(stored_predictions[1], conditional + 3)
    torch.testing.assert_close(
        safe_torch_load(paths.target_latent_path("7")), target + 4
    )
    assert read_json(paths.record_path("7"))["revision"] == "replacement"
    assert not paths.stale_directory.exists()


def test_completion_marker_validates_full_tensor_contract(tmp_path: Path) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    latent, unconditional, conditional, target = _small_tensors()
    hashes = save_generation_tensors(
        paths,
        7,
        latents=latent,
        unconditional_predictions=unconditional,
        conditional_predictions=conditional,
        target_latent=target,
    )
    paths.image_path(7).write_bytes(b"preview bytes")
    science_hash = "a" * 64
    metadata = {
        "record_id": "sdv1-0000",
        "source_row_number": 0,
        "prompt_raw": "exact prompt",
        "target_image_sha256": "b" * 64,
        "scientific_config_hash": science_hash,
        "tensor_file_sha256": hashes,
        "preview_image_sha256": file_sha256(paths.image_path(7)),
        "tensor_shapes": {
            "latent": list(latent.shape),
            "unconditional_noise_predictions": list(unconditional.shape),
            "conditional_noise_predictions": list(conditional.shape),
            "target_latent": list(target.shape),
        },
        "tensor_dtypes": {
            "latent": "float32",
            "unconditional_noise_predictions": "float32",
            "conditional_noise_predictions": "float32",
            "target_latent": "float32",
        },
    }
    publish_completion_marker(paths, 7, metadata)
    validation = validate_generation_record(
        paths,
        7,
        expected_scientific_hash=science_hash,
        expected_record_identity={
            "record_id": "sdv1-0000",
            "source_row_number": 0,
            "prompt_raw": "exact prompt",
            "target_image_sha256": "b" * 64,
        },
        load_tensors=True,
    )
    assert validation.valid, validation.errors
    assert validation.metadata is not None
    assert (
        GENERATION_SELECTION_POLICY == "all_available_canonical_pairs_label_independent"
    )
    assert validation.metadata["selection_policy"] == GENERATION_SELECTION_POLICY
    wrong_identity = validate_generation_record(
        paths,
        7,
        expected_record_identity={
            "record_id": "sdv1-0000",
            "source_row_number": 0,
            "prompt_raw": "changed prompt",
            "target_image_sha256": "b" * 64,
        },
        load_tensors=False,
    )
    assert not wrong_identity.valid
    assert (
        "marker prompt_raw differs from current Webster metadata"
        in wrong_identity.errors
    )
    paths.latent_path(7).write_bytes(b"tampered latent")
    corrupted = validate_generation_record(paths, 7, load_tensors=False)
    assert not corrupted.valid
    assert "latent SHA-256 differs" in corrupted.errors
    with pytest.raises(GenerationCacheError, match="refusing to overwrite"):
        publish_completion_marker(paths, 7, metadata)


def test_record_validation_can_select_artifacts_and_skip_physical_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    identity = {
        "original_index": "7",
        "record_id": "sdv1-0000",
        "source_row_number": 0,
        "prompt_raw": "exact prompt",
        "target_image_sha256": "b" * 64,
        "overfit_type": "N",
    }
    _publish_cached_record(paths, identity, "a" * 64, steps=2, seeds=2)
    paths.noise_prediction_path(7).unlink()
    paths.target_latent_path(7).unlink()
    paths.image_path(7).unlink()

    def forbidden_hash(_path: Path) -> str:
        raise AssertionError("verify_file_hashes=False must not read an artifact")

    monkeypatch.setattr(cache_module, "file_sha256", forbidden_hash)
    validation = validate_generation_record(
        paths,
        7,
        load_tensors=False,
        tensor_names=("latent",),
        require_preview=False,
        verify_file_hashes=False,
    )

    assert validation.valid, validation.errors


def test_hash_fields_remain_required_when_physical_hashing_is_skipped(
    tmp_path: Path,
) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    identity = {
        "original_index": "7",
        "record_id": "sdv1-0000",
        "source_row_number": 0,
        "prompt_raw": "exact prompt",
        "target_image_sha256": "b" * 64,
        "overfit_type": "N",
    }
    _publish_cached_record(paths, identity, "a" * 64, steps=2, seeds=2)
    marker = read_json(paths.record_path(7))
    marker["tensor_file_sha256"]["latent"] = "not-a-sha256"
    atomic_write_json(paths.record_path(7), marker)

    validation = validate_generation_record(
        paths,
        7,
        load_tensors=False,
        tensor_names=("latent",),
        require_preview=False,
        verify_file_hashes=False,
    )

    assert not validation.valid
    assert validation.errors == ("latent SHA-256 differs",)


@pytest.mark.parametrize(
    ("tensor_names", "message"),
    [
        (("latent", "unknown"), "unknown generation tensor"),
        (("latent", "latent"), "duplicate generation tensor"),
        ("latent", "must be a sequence"),
    ],
)
def test_record_validation_rejects_invalid_tensor_selections(
    tmp_path: Path,
    tensor_names: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_generation_record(
            GenerationPaths(tmp_path),
            7,
            tensor_names=tensor_names,  # type: ignore[arg-type]
        )


def test_generation_run_rejects_scientific_hash_corruption(tmp_path: Path) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    science = _science()
    configuration = {
        "scientific_config": science,
        "scientific_config_hash": canonical_hash(science),
    }
    atomic_write_json(paths.run_config, configuration)
    assert require_generation_run(paths) == configuration
    configuration["scientific_config"]["guidance_scale"] = 8.0  # type: ignore[index]
    atomic_write_json(paths.run_config, configuration)
    with pytest.raises(GenerationCacheError, match="scientific_config_hash differs"):
        require_generation_run(paths)


def test_generation_rejects_orphaned_science_without_a_run_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
    )
    paths.latent_directory.mkdir(parents=True)
    paths.latent_path("7").write_bytes(b"orphaned scientific tensor")
    monkeypatch.setattr(
        generation_module,
        "_load_dataset",
        lambda *_: pytest.fail("orphan rejection must precede dataset loading"),
    )

    with pytest.raises(
        generation_module.GenerationError,
        match="scientific artifacts without run_config.json",
    ):
        generation_module.generate_webster_trajectories(
            tmp_path,
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=50,
            num_seeds=20,
            seed_start=0,
            downscale=4,
            device="cpu",
            overwrite=False,
        )


def test_saved_generation_schedule_requires_consistent_coefficients(
    tmp_path: Path,
) -> None:
    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    science = _science(steps=2)
    payload = _schedule(steps=2)
    atomic_torch_save(payload, paths.schedule)
    generation_module._validate_saved_schedule(paths, science, "ddim", 2)

    payload["sigma_t"] = torch.zeros(2, dtype=torch.float32)
    atomic_torch_save(payload, paths.schedule)
    with pytest.raises(
        generation_module.GenerationError,
        match="coefficients are inconsistent",
    ):
        generation_module._validate_saved_schedule(paths, science, "ddim", 2)


def test_generation_device_provenance_is_not_part_of_the_scientific_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from utils.models import schedule_metadata as schedule_module

    class Scheduler:
        config = {"prediction_type": "epsilon"}

    spec = get_model_spec("sdv1")
    components = SimpleNamespace(
        spec=spec,
        model_revision="a" * 40,
        vae_id="synthetic/vae",
        vae_revision="b" * 40,
        inference_dtype=torch.float16,
        package_versions={"torch": torch.__version__},
        device_metadata={"device": "cuda:0", "gpu_name": "Synthetic GPU"},
        device=torch.device("cpu"),
    )
    runtime = generation_module._Runtime(
        components=components,
        scheduler=Scheduler(),
        scheduler_description={
            "name": "ddim",
            "class": "SyntheticScheduler",
            "config": {"prediction_type": "epsilon"},
            "removed_config_keys": [],
        },
        latent_shape=(4, 64, 64),
        source_provenance={"snapshot": "captured-once"},
    )
    dataset = SimpleNamespace(metadata_path=tmp_path / "metadata.parquet")
    paths = GenerationPaths(tmp_path / "logs" / "successful")
    paths.create()

    def forbidden_provenance_reread(_root: Path) -> dict[str, object]:
        raise AssertionError("runtime provenance must be captured only once")

    monkeypatch.setattr(
        generation_module, "_runtime_provenance", forbidden_provenance_reread
    )
    monkeypatch.setattr(
        schedule_module,
        "build_schedule_metadata",
        lambda *_, **__: _schedule(steps=2),
    )

    configuration = generation_module._create_run_config(
        tmp_path,
        paths,
        dataset,
        spec,
        runtime,
        "ddim",
        7.5,
        2,
        (0, 1),
        4,
    )

    science = configuration["scientific_config"]
    assert isinstance(science, dict)
    assert "device_metadata" not in science
    assert configuration["scientific_config_hash"] == canonical_hash(science)
    assert configuration["runtime_provenance"] == {
        "snapshot": "captured-once",
        "device_metadata": components.device_metadata,
    }

    record = generation_module._record_metadata(
        tmp_path,
        paths,
        {
            "original_index": "7",
            "record_id": "sdv1-0007",
            "source_row_number": 7,
            "prompt_raw": "prompt",
            "overfit_type": "N",
            "recovery_status": "available",
            "recovery_method": "fixture",
            "image_path": "sdv1/images/7.png",
            "target_image_sha256": "7" * 64,
        },
        configuration,
        runtime,
        "ddim",
        7.5,
        2,
        (0, 1),
        4,
        {
            "latent": "a" * 64,
            "noise_prediction": "b" * 64,
            "target_latent": "c" * 64,
        },
        "d" * 64,
    )
    assert record["runtime_provenance"] == configuration["runtime_provenance"]
    assert paths.schedule.is_file()
    assert paths.run_config.is_file()

    unpublished = GenerationPaths(tmp_path / "logs" / "unpublished")
    unpublished.create()

    def fail_schedule(*_: object, **__: object) -> object:
        raise RuntimeError("synthetic schedule failure")

    monkeypatch.setattr(schedule_module, "build_schedule_metadata", fail_schedule)
    with pytest.raises(RuntimeError, match="synthetic schedule failure"):
        generation_module._create_run_config(
            tmp_path,
            unpublished,
            dataset,
            get_model_spec("sdv1"),
            runtime,
            "ddim",
            7.5,
            2,
            (0, 1),
            4,
        )
    assert not unpublished.run_config.exists()


@pytest.mark.parametrize("seed_start", [0, 20])
def test_generation_resumes_every_cached_prompt_without_loading_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seed_start: int,
) -> None:
    rows = [
        {
            "original_index": "101",
            "record_id": "sdv1-0000",
            "source_row_number": 0,
            "prompt_raw": "TV prompt",
            "image_path": "sdv1/images/101.png",
            "target_image_path": str(
                (tmp_path / "data/webster/sdv1/images/101.png").resolve()
            ),
            "target_image_sha256": "1" * 64,
            "overfit_type": "TV",
        },
        {
            "original_index": "102",
            "record_id": "sdv1-0001",
            "source_row_number": 1,
            "prompt_raw": "non-TV prompt",
            "image_path": "sdv1/images/102.png",
            "target_image_path": str(
                (tmp_path / "data/webster/sdv1/images/102.png").resolve()
            ),
            "target_image_sha256": "2" * 64,
            "overfit_type": "N",
        },
    ]

    class CachedDataset:
        total_manifest_rows = len(rows)

        def __len__(self) -> int:
            return len(rows)

        def iter_metadata(self):
            yield from (dict(row) for row in rows)

        def __getitem__(self, _index: int) -> object:
            raise AssertionError("valid cached records must not load source images")

    science = _science(seed_start=seed_start)
    science_hash = canonical_hash(science)
    paths = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=seed_start,
    )
    paths.create()
    preview_config = generation_module._preview_configuration(512, 4)
    atomic_write_json(
        paths.run_config,
        {
            "scientific_config": science,
            "scientific_config_hash": science_hash,
            "preview_config": preview_config,
            "preview_config_hash": canonical_hash(preview_config),
        },
    )
    atomic_torch_save(_schedule(), paths.schedule)
    for row in rows:
        _publish_cached_record(paths, row, science_hash, seed_start=seed_start)
    tensor_hashes = {
        path: file_sha256(path)
        for row in rows
        for path in (
            paths.latent_path(row["original_index"]),
            paths.noise_prediction_path(row["original_index"]),
            paths.target_latent_path(row["original_index"]),
        )
    }
    runtime_calls: list[object] = []

    def forbidden_runtime(*arguments: object, **keywords: object) -> object:
        runtime_calls.append((arguments, keywords))
        raise RuntimeError("real model loading is forbidden")

    monkeypatch.setattr(generation_module, "_load_dataset", lambda *_: CachedDataset())
    monkeypatch.setattr(generation_module, "_load_runtime", forbidden_runtime)
    monkeypatch.setattr(
        generation_module,
        "estimate_disk_space",
        lambda *_, **__: DiskEstimate(0, 0, 0, 0, 10**12),
    )
    result = generation_module.generate_webster_trajectories(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=seed_start,
        downscale=4,
        device="cpu",
        overwrite=False,
    )

    assert result.exit_code == 0
    assert result.completed_rows == 2
    assert not runtime_calls
    assert tensor_hashes == {path: file_sha256(path) for path in tensor_hashes}
    manifest = pd.read_parquet(paths.manifest_parquet)
    assert manifest["webster_overfit_type"].tolist() == ["TV", "N"]
    assert manifest["completed_or_resumed"].eq("resumed").all()
    summary = read_json(paths.summary_json)
    assert summary["available_paired_image_rows"] == 2
    assert summary["selected_rows"] == 2
    assert summary["fully_resumed_rows"] == 2
    assert summary["selection_policy"] == GENERATION_SELECTION_POLICY
    progress_output = capsys.readouterr().err
    assert "[Generation] Records" in progress_output
    assert "2/2" in progress_output

    with monkeypatch.context() as cache_hit_patch:
        cache_hit_patch.setattr(
            generation_module,
            "resolve_devices",
            lambda *_: pytest.fail("complete cache must return before device setup"),
        )
        cache_hit_patch.setattr(
            cache_module,
            "file_sha256",
            lambda *_: pytest.fail("complete cache must not physically hash tensors"),
        )
        cached = generation_module.generate_webster_trajectories(
            tmp_path,
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=50,
            num_seeds=20,
            seed_start=seed_start,
            downscale=4,
            device="cuda:999",
            overwrite=False,
        )

    assert cached.complete_cache_hit is True
    assert cached.completed_rows == len(rows)

    configuration = read_json(paths.run_config)
    marker_path = paths.record_path(rows[0]["original_index"])
    complete_marker = read_json(marker_path)
    for required_field in (
        "num_inference_steps",
        "num_seeds",
        "seeds",
        "latent_shape",
        "inference_dtype",
        "scheduler_class",
        "target_image_path",
        "target_image_sha256",
        "schedule_path",
        "tensor_file_sha256",
        "tensor_shapes",
        "tensor_dtypes",
        "preview_image_sha256",
        "preview_downscale",
        "preview_status",
    ):
        incomplete_marker = dict(complete_marker)
        incomplete_marker.pop(required_field)
        atomic_write_json(marker_path, incomplete_marker)
        atomic_write_json(paths.summary_json, summary)
        assert (
            generation_module._complete_cache_result(
                paths,
                CachedDataset(),
                rows,
                configuration,
                project_root=tmp_path,
                resolution=512,
                downscale=4,
                num_seeds=20,
            )
            is None
        )
        atomic_write_json(marker_path, complete_marker)
        atomic_write_json(paths.summary_json, summary)

    atomic_write_json(marker_path, complete_marker)
    assert (
        generation_module._complete_cache_result(
            paths,
            CachedDataset(),
            rows,
            configuration,
            project_root=tmp_path,
            resolution=512,
            downscale=4,
            num_seeds=20,
        )
        is None
    )
    atomic_write_json(paths.summary_json, summary)

    failed_contents = paths.failed_csv.read_bytes()
    paths.failed_csv.write_bytes(b"")
    atomic_write_json(paths.summary_json, summary)
    assert (
        generation_module._complete_cache_result(
            paths,
            CachedDataset(),
            rows,
            configuration,
            project_root=tmp_path,
            resolution=512,
            downscale=4,
            num_seeds=20,
        )
        is None
    )
    paths.failed_csv.write_bytes(failed_contents)
    atomic_write_json(paths.summary_json, summary)

    schedule_contents = paths.schedule.read_bytes()
    paths.schedule.write_bytes(b"")
    atomic_write_json(paths.summary_json, summary)
    assert (
        generation_module._complete_cache_result(
            paths,
            CachedDataset(),
            rows,
            configuration,
            project_root=tmp_path,
            resolution=512,
            downscale=4,
            num_seeds=20,
        )
        is None
    )
    paths.schedule.write_bytes(schedule_contents)
    atomic_write_json(paths.summary_json, summary)

    marker_stat = marker_path.stat()
    changed_path = paths.latent_path(rows[0]["original_index"])
    original_stat = changed_path.stat()
    replacement = changed_path.with_name(f".{changed_path.name}.replacement")
    replacement.write_bytes(changed_path.read_bytes())
    os.utime(
        replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    os.replace(replacement, changed_path)
    replaced_stat = changed_path.stat()
    assert replaced_stat.st_mtime_ns == original_stat.st_mtime_ns
    assert replaced_stat.st_ctime_ns > marker_stat.st_ctime_ns
    checked = generation_module.generate_webster_trajectories(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=seed_start,
        downscale=4,
        device="cpu",
        overwrite=False,
    )
    assert checked.complete_cache_hit is False
    assert checked.completed_rows == len(rows)


def test_generation_changed_downscale_regenerates_only_cached_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "original_index": "101",
        "record_id": "sdv1-0000",
        "source_row_number": 0,
        "prompt_raw": "cached prompt",
        "target_image_sha256": "1" * 64,
        "overfit_type": "TV",
    }

    class CachedDataset:
        total_manifest_rows = 1

        def __len__(self) -> int:
            return 1

        def iter_metadata(self):
            yield dict(row)

        def __getitem__(self, _index: int) -> object:
            raise AssertionError("preview-only recovery must not load source images")

    spec = get_model_spec("sdv1")
    science = _science()
    science_hash = canonical_hash(science)
    paths = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
    )
    paths.create()
    original_preview_config = generation_module._preview_configuration(
        spec.resolution, 4
    )
    atomic_write_json(
        paths.run_config,
        {
            "scientific_config": science,
            "scientific_config_hash": science_hash,
            "preview_config": original_preview_config,
            "preview_config_hash": canonical_hash(original_preview_config),
        },
    )
    atomic_torch_save(_schedule(), paths.schedule)
    _publish_cached_record(paths, row, science_hash, downscale=4)
    atomic_write_json(
        paths.summary_json,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "outcome": "completed",
            "model_manifest_rows": 1,
            "available_paired_image_rows": 1,
            "selected_rows": 1,
            "completed_rows": 1,
            "skipped_rows": 0,
            "failed_rows": 0,
            "total_trajectories": 20,
            "selection_policy": GENERATION_SELECTION_POLICY,
            "scientific_config_hash": science_hash,
            "preview_config_hash": canonical_hash(original_preview_config),
        },
    )
    for aggregate in (
        paths.manifest_csv,
        paths.manifest_parquet,
        paths.skipped_csv,
        paths.failed_csv,
    ):
        aggregate.write_bytes(b"published aggregate")

    scientific_paths = (
        paths.latent_path("101"),
        paths.noise_prediction_path("101"),
        paths.target_latent_path("101"),
    )
    scientific_hashes = {path: file_sha256(path) for path in scientific_paths}
    cached_trajectory = safe_torch_load(paths.latent_path("101"))
    assert isinstance(cached_trajectory, torch.Tensor)
    original_run_config = read_json(paths.run_config)
    original_marker = read_json(paths.record_path("101"))

    vae = object()
    runtime = generation_module._Runtime(
        components=SimpleNamespace(vae=vae, device=torch.device("cpu")),
        scheduler=object(),
        scheduler_description={},
        latent_shape=(1, 1, 1),
        source_provenance={},
    )
    preview_calls: list[tuple[int, int]] = []

    def write_preview(
        terminal_latents: torch.Tensor,
        observed_vae: object,
        observed_device: torch.device,
        destination: Path,
        resolution: int,
        downscale: int,
        progress_callback: object = None,
    ) -> str:
        assert observed_vae is vae
        assert observed_device == torch.device("cpu")
        torch.testing.assert_close(terminal_latents, cached_trajectory[:, -1])
        preview_calls.append((resolution, downscale))
        destination.write_bytes(b"preview regenerated at downscale 8")
        if callable(progress_callback):
            progress_callback(len(terminal_latents))
        return file_sha256(destination)

    monkeypatch.setattr(generation_module, "_load_dataset", lambda *_: CachedDataset())
    monkeypatch.setattr(generation_module, "_load_runtime", lambda *_, **__: runtime)
    monkeypatch.setattr(generation_module, "_write_preview", write_preview)
    monkeypatch.setattr(
        generation_module,
        "_sample",
        lambda *_, **__: pytest.fail("preview recovery must not denoise"),
    )
    monkeypatch.setattr(
        generation_module,
        "save_generation_tensors",
        lambda *_, **__: pytest.fail("preview recovery must not rewrite tensors"),
    )
    monkeypatch.setattr(
        generation_module,
        "estimate_disk_space",
        lambda *_, **__: DiskEstimate(0, 0, 0, 0, 10**12),
    )

    result = generation_module.generate_webster_trajectories(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
        downscale=8,
        device="cpu",
        overwrite=False,
    )

    assert result.exit_code == 0
    assert result.completed_rows == 1
    assert preview_calls == [(spec.resolution, 8)]
    assert scientific_hashes == {path: file_sha256(path) for path in scientific_paths}
    updated_run_config = read_json(paths.run_config)
    expected_preview_config = generation_module._preview_configuration(
        spec.resolution, 8
    )
    assert (
        updated_run_config["scientific_config"]
        == original_run_config["scientific_config"]
    )
    assert updated_run_config["scientific_config_hash"] == science_hash
    assert updated_run_config["preview_config"] == expected_preview_config
    assert updated_run_config["preview_config_hash"] == canonical_hash(
        expected_preview_config
    )
    updated_marker = read_json(paths.record_path("101"))
    assert updated_marker["preview_downscale"] == 8
    assert updated_marker["preview_image_sha256"] == file_sha256(
        paths.image_path("101")
    )
    assert (
        updated_marker["preview_image_sha256"]
        != original_marker["preview_image_sha256"]
    )
    summary = read_json(paths.summary_json)
    assert summary["newly_generated_rows"] == 0
    assert summary["fully_resumed_rows"] == 0
    assert summary["preview_only_regenerated_rows"] == 1


def test_generation_loads_a_failed_worker_runtime_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tuple(
        (
            position,
            {
                "original_index": str(position),
                "record_id": f"sdv1-{position:04d}",
                "source_row_number": position,
                "prompt_raw": f"prompt {position}",
                "target_image_sha256": str(position) * 64,
                "overfit_type": "N",
            },
        )
        for position in (1, 2)
    )

    class Dataset:
        def __getitem__(self, position: int) -> dict[str, object]:
            return {
                "image": Image.new("RGB", (1, 1)),
                "prompt": rows[position - 1][1]["prompt_raw"],
            }

    class InvalidRecord:
        valid = False
        errors = ("completion marker is missing",)
        metadata = None

    runtime_calls: list[torch.device] = []

    def fail_runtime(*_: object, device: torch.device, **__: object) -> object:
        runtime_calls.append(device)
        raise RuntimeError("synthetic model load failure")

    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    monkeypatch.setattr(
        generation_module,
        "validate_generation_record",
        lambda *_, **__: InvalidRecord(),
    )
    monkeypatch.setattr(generation_module, "_load_runtime", fail_runtime)

    result, runtime = generation_module._run_generation_shard(
        root=tmp_path,
        paths=paths,
        dataset=Dataset(),
        spec=get_model_spec("sdv1"),
        scheduler_name="ddim",
        guidance=7.5,
        steps=50,
        seed_values=tuple(range(20)),
        downscale=4,
        entries=rows,
        run_config={"scientific_config_hash": "f" * 64},
        device=torch.device("cpu"),
        worker_index=0,
        worker_count=1,
        overwrite=False,
    )

    assert runtime is None
    assert len(result.failed_rows) == 2
    assert runtime_calls == [torch.device("cpu")]
    assert "synthetic model load failure" in result.failed_rows[1]["exception_message"]


def test_generation_reports_record_failure_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = {
        "original_index": "77",
        "record_id": "sdv1-0077",
        "source_row_number": 77,
        "prompt_raw": "failing prompt",
        "target_image_sha256": "7" * 64,
        "overfit_type": "N",
    }

    class FailingDataset:
        total_manifest_rows = 1

        def __len__(self) -> int:
            return 1

        def iter_metadata(self):
            yield dict(row)

    def fail_validation(*_: object, **__: object) -> object:
        raise RuntimeError("synthetic validation failure")

    monkeypatch.setattr(generation_module, "_load_dataset", lambda *_: FailingDataset())
    monkeypatch.setattr(
        generation_module, "validate_generation_record", fail_validation
    )
    monkeypatch.setattr(
        generation_module,
        "estimate_disk_space",
        lambda *_, **__: DiskEstimate(1, 1, 1, 0, 10**12),
    )

    result = generation_module.generate_webster_trajectories(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
        downscale=4,
        device="cpu",
        overwrite=False,
    )

    assert result.exit_code == 1
    assert result.failed_rows == 1
    progress_output = capsys.readouterr().err
    assert (
        "[Generation] Record 77 failed: RuntimeError: synthetic validation failure"
    ) in progress_output
    assert "traceback:" in progress_output
    assert "1/1" in progress_output


def test_overwrite_unavailable_item_quarantines_old_record_and_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "original_index": "77",
        "record_id": "sdv1-0077",
        "source_row_number": 0,
        "prompt_raw": "temporarily unavailable prompt",
        "target_image_sha256": "7" * 64,
        "overfit_type": "N",
    }

    class UnavailableDataset:
        total_manifest_rows = 1

        def __len__(self) -> int:
            return 1

        def iter_metadata(self):
            yield dict(row)

        def __getitem__(self, position: int) -> dict[str, object]:
            assert position == 0
            return {"image": None, "prompt": row["prompt_raw"]}

    paths = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
    )
    paths.create()
    science = _science()
    science_hash = canonical_hash(science)
    atomic_write_json(
        paths.run_config,
        {
            "scientific_config": science,
            "scientific_config_hash": science_hash,
        },
    )
    atomic_torch_save(_schedule(), paths.schedule)
    _publish_cached_record(paths, row, science_hash)

    monkeypatch.setattr(
        generation_module, "_load_dataset", lambda *_: UnavailableDataset()
    )
    monkeypatch.setattr(
        generation_module,
        "_load_runtime",
        lambda *_args, **_kwargs: pytest.fail(
            "unavailable input must not load a model"
        ),
    )
    monkeypatch.setattr(
        generation_module,
        "estimate_disk_space",
        lambda *_, **__: DiskEstimate(1, 1, 1, 0, 10**12),
    )

    result = generation_module.generate_webster_trajectories(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=0,
        downscale=4,
        device="cpu",
        overwrite=True,
    )

    assert result.exit_code == 1
    assert result.completed_rows == 0
    assert result.failed_rows == 1
    assert not paths.record_path("77").exists()
    assert list_completed_records(paths) == []
    stale_bundles = list(paths.stale_directory.iterdir())
    assert len(stale_bundles) == 1
    assert (stale_bundles[0] / "record" / "77.json").is_file()
    for path in (
        paths.latent_path("77"),
        paths.noise_prediction_path("77"),
        paths.target_latent_path("77"),
        paths.image_path("77"),
    ):
        assert not path.exists()


@pytest.mark.parametrize(
    ("initial_validation", "overwrite"),
    ((False, False), (True, True)),
)
def test_generation_writes_new_or_explicitly_overwritten_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    initial_validation: bool,
    overwrite: bool,
) -> None:
    row = {
        "original_index": "7",
        "record_id": "sdv1-0007",
        "source_row_number": 7,
        "prompt_raw": "fresh prompt",
        "target_image_sha256": "7" * 64,
        "overfit_type": "N",
    }

    class Dataset:
        def __getitem__(self, position: int) -> dict[str, object]:
            assert position == 0
            return {
                "image": Image.new("RGB", (1, 1)),
                "prompt": row["prompt_raw"],
            }

    paths = GenerationPaths(tmp_path / "logs" / "synthetic")
    paths.create()
    components = SimpleNamespace(vae=object(), device=torch.device("cpu"))
    runtime = generation_module._Runtime(
        components=components,
        scheduler=object(),
        scheduler_description={},
        latent_shape=(1, 1, 1),
        source_provenance={},
    )
    validation_keywords: list[dict[str, object]] = []

    def validate(*_arguments: object, **keywords: object) -> CacheValidation:
        validation_keywords.append(dict(keywords))
        if len(validation_keywords) == 1:
            if initial_validation:
                return CacheValidation(True, (), dict(row))
            return CacheValidation(False, ("completion marker is missing",))
        return CacheValidation(True, (), read_json(paths.record_path(7)))

    def sample(*_arguments: object, **keywords: object) -> object:
        callback = keywords.get("progress_callback")
        if callable(callback):
            callback(1)
        predictions = torch.zeros((1, 1, 1, 1, 1))
        return SimpleNamespace(
            latents=torch.zeros((1, 2, 1, 1, 1)),
            unconditional_noise_predictions=predictions,
            conditional_noise_predictions=predictions,
        )

    def write_preview(
        _latents: torch.Tensor,
        _vae: object,
        _device: torch.device,
        destination: Path,
        _resolution: int,
        _downscale: int,
        progress_callback: object = None,
    ) -> str:
        destination.write_bytes(b"preview")
        if callable(progress_callback):
            progress_callback(1)
        return "e" * 64

    monkeypatch.setattr(generation_module, "validate_generation_record", validate)
    monkeypatch.setattr(
        generation_module,
        "_encode_target",
        lambda *_args, **_kwargs: torch.zeros((1, 1, 1)),
    )
    monkeypatch.setattr(generation_module, "_sample", sample)
    save_overwrite: list[bool] = []

    def save(*_args: object, **kwargs: object) -> dict[str, str]:
        save_overwrite.append(bool(kwargs.get("overwrite")))
        return {
            "latent": "a" * 64,
            "noise_prediction": "b" * 64,
            "target_latent": "c" * 64,
        }

    monkeypatch.setattr(generation_module, "save_generation_tensors", save)
    monkeypatch.setattr(generation_module, "_write_preview", write_preview)
    monkeypatch.setattr(
        generation_module,
        "_record_metadata",
        lambda *_args, **_kwargs: dict(row),
    )
    monkeypatch.setattr(
        generation_module,
        "_manifest_row",
        lambda *_args, **_kwargs: {
            "original_index": "7",
            "source_row_number": 7,
        },
    )
    if overwrite:
        atomic_write_json(paths.record_path("7"), {"old": True})

    result, observed_runtime = generation_module._run_generation_shard(
        root=tmp_path,
        paths=paths,
        dataset=Dataset(),
        spec=get_model_spec("sdv1"),
        scheduler_name="ddim",
        guidance=7.5,
        steps=1,
        seed_values=(0,),
        downscale=4,
        entries=((0, row),),
        run_config={"scientific_config_hash": "f" * 64},
        device=torch.device("cpu"),
        worker_index=0,
        worker_count=1,
        overwrite=overwrite,
        runtime=runtime,
    )

    assert observed_runtime is runtime
    assert len(result.manifest_rows) == 1
    assert save_overwrite == [overwrite]
    assert len(validation_keywords) == 2
    assert validation_keywords[0]["verify_file_hashes"] is (not overwrite)
    assert validation_keywords[1]["verify_file_hashes"] is False
    assert "current-record local ETA" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("wrapper", "entrypoint"),
    [
        ("download_webster.sh", "scripts/download_webster.py"),
        ("generate.sh", "scripts/generate.py"),
        ("sscd.sh", "scripts/sscd.py"),
        ("compute_proximity.sh", "scripts/compute_proximity.py"),
        (
            "theorem1_loss_recovery.sh",
            "scripts/theorem1_loss_recovery.py",
        ),
        (
            "unconditional_baseline.sh",
            "scripts/unconditional_baseline.py",
        ),
        (
            "lemma2_mean_convergence.sh",
            "scripts/lemma2_mean_convergence.py",
        ),
        (
            "corollary3_cfg_amplification.sh",
            "scripts/corollary3_cfg_amplification.py",
        ),
    ],
)
def test_shell_wrappers_resolve_the_project_root_from_any_directory(
    wrapper: str, entrypoint: str, tmp_path: Path
) -> None:
    source = (ROOT / wrapper).read_text(encoding="utf-8")
    assert "BASH_SOURCE[0]" in source
    assert "export PYTHONUNBUFFERED=1" in source
    assert f'"$PROJECT_ROOT/{entrypoint}"' in source
    environment = dict(os.environ, PYTHON=sys.executable)
    result = subprocess.run(
        ["bash", str(ROOT / wrapper), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.casefold()


def test_download_cli_rejects_per_host_limit_above_worker_count(
    tmp_path: Path,
) -> None:
    environment = dict(
        os.environ,
        PYTHON=sys.executable,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "download_webster.sh"),
            "--direct-workers",
            "2",
            "--per-host-concurrency",
            "3",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot exceed" in result.stderr


# Root orchestration contracts moved to test_theory_plot_cli.py with cache-only theory.


def _imports(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.add(node.module)
    return values


def test_lemma2_trajectory_reducers_do_not_run_model_inference() -> None:
    path = ROOT / "scripts/lemma2_mean_convergence.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    reducers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_load_cached_trajectory", "_centered_trajectory_distances"}
    ]
    assert len(reducers) == 2
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for reducer in reducers
        for node in ast.walk(reducer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert calls.isdisjoint(
        {
            "build_scheduler",
            "compile_loaded_unet",
            "encode_prompt_condition",
            "load_model_components",
            "make_initial_noise",
            "predict_conditional_epsilon",
            "preflight_model_components",
            "randn",
            "randn_like",
        }
    )
    for snippet in (
        "selection.prompt_frame",
        'tensor_names=("latent", "noise_prediction")',
        "prediction_value[0]",
        "latents[:, step_index]",
        "unconditional_epsilon[:, step_index]",
        "canonical prompt-major, seed-major, timestep-major CSV grid",
        '"evaluation_schedule_sha256": contract.identity.schedule_sha256',
        '"evaluation_schedule_sha256": identity.schedule_sha256',
        "baseline.source_schedule_sha256 != contract.identity.schedule_sha256",
        "verify_file_hashes=False",
    ):
        assert snippet in source


def test_readme_documents_cache_only_theory_and_provenance() -> None:
    import json

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    registry = json.loads((ROOT / "docs/theory/statement_registry.json").read_text())
    assert len(registry["statements"]) == 7
    documented_figures = {
        entry[field]["id"]
        for entry in registry["statements"]
        for field in ("main_figure", "diagnostic_figure")
        if entry.get(field) is not None
    }
    assert documented_figures == {
        "initial_recovery", "unconditional_center", "posterior_feedback",
        "target_synchronization", "target_injection", "terminal_terms",
    }
    assert all(figure in readme for figure in documented_figures)
    for required in (
        "--recompute-experiments", "--bundle", "--plot", "--overwrite",
        "--center reference-initial", "--cached-baseline PATH", "--use-mu",
        "reference seeds `N..2N-1`", "including unsuccessful reproductions",
        "not the known data mean",
        "four default figures", "figure tables", "legacy utilities",
        "source identities/hashes", "applicability_report.json", "PYTHON",
        "--include-diagnostics", "cache-theory-3.0-proposition5",
        "never reads raw tensors", "preserving the previous",
    ):
        assert required in readme
    assert (ROOT / "docs/theory/quantity_dictionary.md").is_file()


def test_prompt_selection_is_owned_by_proximity_not_generation_or_sscd() -> None:
    stage_imports: dict[str, set[str]] = {}
    stage_calls: dict[str, set[str]] = {}
    for stage in ("generation", "sscd", "proximity"):
        path = ROOT / f"utils/experiments/{stage}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        stage_imports[stage] = _imports(tree)
        stage_calls[stage] = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
    for stage in ("generation", "sscd"):
        assert "utils.data.selection" not in stage_imports[stage]
        assert not stage_calls[stage].intersection(
            {"load_target_pair_selection", "ensure_reference_target_pair_selection"}
        )
    assert "utils.data.selection" in stage_imports["proximity"]


def test_source_tree_has_only_the_current_modules_and_imports() -> None:
    allowed = {
        "utils/common": {"__init__.py", "cli.py", "io.py"},
        "utils/data": {
            "__init__.py",
            "arquivo.py",
            "commoncrawl.py",
            "images.py",
            "mirror.py",
            "official.py",
            "proximity_gmm.py",
            "recovery.py",
            "selection.py",
            "state.py",
            "wayback.py",
            "webster.py",
        },
        "utils/experiments": {
            "__init__.py",
            "cache.py",
            "forward_corruptions_cache.py",
            "forward_corruptions_plotting.py",
            "generation.py",
            "plotting.py",
            "latent_statistics.py",
            "_unconditional_baseline_metadata.py",
            "unconditional_baseline.py",
            "proximity.py",
            "sscd.py",
        },
        "utils/experiments/theory": {
            "__init__.py", "contracts.py", "cache_reader.py", "centers.py",
            "support.py", "scheduler_adapter.py", "metrics.py", "statements.py",
            "reduce.py", "plotting.py", "progress.py", "feedback.py", "summaries.py",
            "candidate_contracts.py", "candidate_feedback.py", "candidate_integration.py",
            "candidate_metrics.py", "candidate_summaries.py", "candidate_reduce.py",
            "candidate_registry.py", "candidate_plotting.py",
        },
        "utils/metrics": {"__init__.py", "sscd.py"},
        "utils/models": {
            "__init__.py",
            "devices.py",
            "latent.py",
            "loading.py",
            "prediction_conversion.py",
            "registry.py",
            "sampling.py",
            "schedule_metadata.py",
            "schedulers.py",
        },
    }
    for relative, permitted in allowed.items():
        observed = {path.name for path in (ROOT / relative).glob("*.py")}
        assert not observed - permitted, (
            f"unexpected Python files in {relative}: {sorted(observed - permitted)}"
        )
    script_files = {
        path.relative_to(ROOT / "scripts").as_posix()
        for path in (ROOT / "scripts").rglob("*.py")
    }
    assert script_files == {
        "check_proximity_clusters.py",
        "check_proximity_gmm.py",
        "compute_proximity.py",
        "download_webster.py",
        "forward_corruptions_generated_states.py",
        "generate.py",
        "sscd.py",
        "theorem1_loss_recovery.py",
        "theory_validation.py",
        "unconditional_baseline.py",
        "lemma2_mean_convergence.py",
        "corollary3_cfg_amplification.py",
    }
    assert not (ROOT / "utils/runtime_provenance.py").exists()

    forbidden_import_fragments = (
        "utils.data.curation",
        "utils.data.webster_dataset",
        "utils.experiments.definition1",
        "utils.experiments.generation_",
        "utils.experiments.completed_records",
        "utils.experiments.sscd_evaluation",
        "utils.experiments.proximity_sscd_plot",
        "utils.models.latent_space",
        "utils.runtime_provenance",
        "scripts.data",
        "scripts.experiments",
    )
    for source_path in [
        *(ROOT / "utils").rglob("*.py"),
        *(ROOT / "scripts").rglob("*.py"),
    ]:
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
        imports = _imports(tree)
        assert not {
            name
            for name in imports
            if any(fragment in name for fragment in forbidden_import_fragments)
        }, source_path
        assert not any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr == "modules"
            for node in ast.walk(tree)
        ), source_path
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            for node in ast.walk(tree)
        ), source_path
        if source_path.name == "__init__.py":
            assert not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__getattr__"
                for node in ast.walk(tree)
            ), source_path


def test_cluster_diagnostic_writes_three_single_panel_line_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    import pandas as pd
    from matplotlib.collections import LineCollection, PathCollection
    from scripts import check_proximity_clusters as diagnostic

    frame = pd.DataFrame(
        [
            {
                "original_index": str(prompt),
                "seed": seed,
                "l2_norm": float(3 - seed + prompt * 4),
                "sscd": (0.1, 0.8, 0.2)[seed],
                "kind": ("MV", "N")[prompt],
                "kmeans_cluster": ("low_sscd_mode", "high_sscd_mode", "low_sscd_mode")[
                    seed
                ],
                "prompt_rule": ("rho < 0", "rho >= 0")[prompt],
            }
            for prompt in range(2)
            for seed in range(3)
        ]
    )
    captured = []
    original_publish = diagnostic._publish_figures

    def publish(output, figures):
        captured.extend(figures)
        original_publish(output, figures)

    monkeypatch.setattr(diagnostic, "_publish_figures", publish)
    paths = diagnostic.plot_assignments(frame, tmp_path / "kmeans_k2.png", cutoff=0.2)
    assert {path.name for path in paths} == {
        f"kmeans_k2{suffix}.{extension}"
        for suffix in ("", "_spearman", "_kind")
        for extension in ("png", "pdf")
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert len(captured) == 3
    for figure, _filenames in captured:
        np.testing.assert_allclose(figure.get_size_inches(), (4.0, 4.0))
        assert len(figure.axes) == 2  # One data panel and its colorbar.
        axis, colorbar_axis = figure.axes
        assert not axis.get_title() and figure._suptitle is None
        assert not any(isinstance(item, PathCollection) for item in axis.collections)
        curves = [item for item in axis.collections if isinstance(item, LineCollection)]
        assert len(curves) == 1 and len(curves[0].get_segments()) == 4
        assert curves[0].cmap.name == "viridis"
        assert (curves[0].norm.vmin, curves[0].norm.vmax) == (0.0, 1.0)
        assert colorbar_axis.get_ylabel() == "SSCD"
        assert colorbar_axis.collections[-1].get_alpha() == 1.0
        assert axis.xaxis.label.get_fontsize() == 15
        assert all(
            handle.get_alpha() == 1.0 for handle in axis.get_legend().legend_handles
        )


def test_repository_hygiene_excludes_generated_and_macos_metadata() -> None:
    ignore_rules = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {
        "__pycache__/",
        "*.py[cod]",
        ".pytest_cache/",
        ".DS_Store",
        "__MACOSX/",
    } <= ignore_rules

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    tracked_paths = [
        Path(value.decode("utf-8")) for value in result.stdout.split(b"\0") if value
    ]
    generated = [
        path
        for path in tracked_paths
        if "__pycache__" in path.parts
        or "__MACOSX" in path.parts
        or path.name == ".DS_Store"
        or path.suffix in {".pyc", ".pyo"}
    ]
    assert not generated, f"tracked generated metadata: {generated}"
