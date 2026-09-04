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
    }
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


def test_generation_uses_one_disjoint_shard_per_auto_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        assert isinstance(entries, tuple)
        assert isinstance(selected, str)
        assert isinstance(worker_index, int)
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
        )

    assert not (tmp_path / "logs").exists()


def _small_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latent = torch.arange(24, dtype=torch.float32).reshape(2, 3, 1, 2, 2)
    unconditional = torch.zeros((2, 2, 1, 2, 2), dtype=torch.float32)
    conditional = torch.ones((2, 2, 1, 2, 2), dtype=torch.float32)
    target = torch.full((1, 2, 2), 0.5, dtype=torch.float32)
    return latent, unconditional, conditional, target


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
        "vae_id": spec.vae_id or spec.model_id,
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
        "trajectory_order": "noise_to_image",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
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
            "webster_overfit_type": identity["overfit_type"],
            "recovery_status": "recovered_exact_url",
            "recovery_method": "direct",
            "scientific_config_hash": science_hash,
            "scheduler_name": "ddim",
            "guidance_scale": 7.5,
            "num_inference_steps": steps,
            "num_seeds": seeds,
            "seeds": list(range(seed_start, seed_start + seeds)),
            "latent_shape": [1, 1, 1],
            "latent_path": f"latent/{index}.pt",
            "noise_prediction_path": f"noise_pred/{index}.pt",
            "target_latent_path": f"target_latent/{index}.pt",
            "preview_image_path": f"image/{index}.png",
            "tensor_file_sha256": hashes,
            "preview_image_sha256": file_sha256(paths.image_path(index)),
            "preview_downscale": downscale,
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
            "target_image_sha256": "1" * 64,
            "overfit_type": "TV",
        },
        {
            "original_index": "102",
            "record_id": "sdv1-0001",
            "source_row_number": 1,
            "prompt_raw": "non-TV prompt",
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
    )

    assert result.exit_code == 1
    assert result.failed_rows == 1
    progress_output = capsys.readouterr().err
    assert (
        "[Generation] Record 77 failed: RuntimeError: synthetic validation failure"
    ) in progress_output
    assert "traceback:" in progress_output
    assert "1/1" in progress_output


def test_new_generation_record_skips_only_the_immediate_physical_hash_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch.setattr(
        generation_module,
        "save_generation_tensors",
        lambda *_args, **_kwargs: {
            "latent": "a" * 64,
            "noise_prediction": "b" * 64,
            "target_latent": "c" * 64,
        },
    )
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
        runtime=runtime,
    )

    assert observed_runtime is runtime
    assert len(result.manifest_rows) == 1
    assert len(validation_keywords) == 2
    assert "verify_file_hashes" not in validation_keywords[0]
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


def test_run_all_orders_reference_and_experiment_seed_pools(tmp_path: Path) -> None:
    project = tmp_path / "pipeline"
    project.mkdir()
    run_all = project / "run_all.sh"
    run_all.write_bytes((ROOT / "run_all.sh").read_bytes())
    run_all.chmod(0o755)

    fake_wrapper = """#!/usr/bin/env bash
set -Eeuo pipefail
wrapper_name="$(basename -- "$0")"
{
    printf '%s' "$wrapper_name"
    if (($# > 0)); then
        printf '\t%s' "$@"
    fi
    printf '\n'
} >> "$RUN_ALL_LOG"
if [[ "$wrapper_name" == "download_webster.sh" ]]; then
    printf 'download progress marker\n' >&2
    exit "${DOWNLOAD_EXIT_CODE:-0}"
fi
if [[ -n "${FAIL_WRAPPER:-}" \
    && "$wrapper_name" == "$FAIL_WRAPPER" \
    && " $* " == *" --model ${FAIL_MODEL:-} "* ]]; then
    printf 'injected failure for %s\n' "${FAIL_MODEL:-}" >&2
    exit "${FAIL_EXIT_CODE:-17}"
fi
"""
    wrappers = (
        "download_webster.sh",
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "theorem1_loss_recovery.sh",
    )
    for wrapper in wrappers:
        path = project / wrapper
        path.write_text(fake_wrapper, encoding="utf-8")
        path.chmod(0o755)

    log = tmp_path / "run-all.log"
    environment = dict(
        os.environ,
        RUN_ALL_LOG=str(log),
        DOWNLOAD_EXIT_CODE="2",
    )
    result = subprocess.run(
        [
            "bash",
            str(run_all),
            "--download",
            "--model=sdv2",
            "--scheduler",
            "ddpm",
            "--g=3.25",
            "--T",
            "12",
            "--N=4",
            "--num-loss-seeds=7",
            "--loss-seed",
            "11",
            "--downscale",
            "8",
            "--direct-workers=12",
            "--direct-attempts",
            "1",
            "--per-host-concurrency=3",
            "--device=cuda:2",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    reference = (
        "\t--model\tsdv2\t--scheduler\tddpm\t--g\t3.25"
        "\t--T\t12\t--N\t4\t--seed-start\t4"
    )
    experiment = (
        "\t--model\tsdv2\t--scheduler\tddpm\t--g\t3.25"
        "\t--T\t12\t--N\t4\t--seed-start\t0"
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        (
            "download_webster.sh\t--direct-workers\t12"
            "\t--direct-attempts\t1\t--per-host-concurrency\t3"
        ),
        f"generate.sh{reference}\t--device\tcuda:2\t--downscale\t8",
        f"sscd.sh{reference}\t--device\tcuda:2",
        f"compute_proximity.sh{reference}",
        f"generate.sh{experiment}\t--device\tcuda:2\t--downscale\t8",
        f"sscd.sh{experiment}\t--device\tcuda:2",
        f"compute_proximity.sh{experiment}",
        (
            "theorem1_loss_recovery.sh\t--model\tsdv2\t--scheduler\tddpm"
            "\t--g\t3.25\t--T\t12\t--N\t4\t--num-loss-seeds\t7"
            "\t--loss-seed\t11\t--device\tcuda:2"
        ),
    ]
    stage_offsets = [result.stdout.index(f"[{stage}/8]") for stage in range(1, 9)]
    assert stage_offsets == sorted(stage_offsets)
    assert "retryable records" not in result.stderr
    assert "download progress marker" in result.stderr
    assert "reference (seeds 4-7)" in result.stdout

    log.unlink()
    environment["DOWNLOAD_EXIT_CODE"] = "1"
    failed = subprocess.run(
        ["bash", str(run_all), "--download"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert failed.returncode == 1
    assert log.read_text(encoding="utf-8").splitlines() == [
        (
            "download_webster.sh\t--direct-workers\t24"
            "\t--direct-attempts\t2\t--per-host-concurrency\t4"
        )
    ]

    log.unlink()
    skipped = subprocess.run(
        [
            "bash",
            str(run_all),
            "--scheduler",
            "ddpm",
            "--g",
            "3.25",
            "--T",
            "12",
            "--N",
            "1",
            "--downscale",
            "8",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    skipped_lines = log.read_text(encoding="utf-8").splitlines()
    assert skipped.returncode == 0, skipped.stderr
    expected_stage_wrappers = [
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "theorem1_loss_recovery.sh",
    ]
    expected_skipped_lines = []
    for model in ("sdv1", "sdv2", "realvis"):
        model_reference = (
            f"\t--model\t{model}\t--scheduler\tddpm\t--g\t3.25"
            "\t--T\t12\t--N\t1\t--seed-start\t1"
        )
        model_experiment = (
            f"\t--model\t{model}\t--scheduler\tddpm\t--g\t3.25"
            "\t--T\t12\t--N\t1\t--seed-start\t0"
        )
        expected_skipped_lines.extend(
            [
                f"generate.sh{model_reference}\t--device\tauto\t--downscale\t8",
                f"sscd.sh{model_reference}\t--device\tauto",
                f"compute_proximity.sh{model_reference}",
                f"generate.sh{model_experiment}\t--device\tauto\t--downscale\t8",
                f"sscd.sh{model_experiment}\t--device\tauto",
                f"compute_proximity.sh{model_experiment}",
                (
                    f"theorem1_loss_recovery.sh\t--model\t{model}"
                    "\t--scheduler\tddpm\t--g\t3.25\t--T\t12\t--N\t1"
                    "\t--num-loss-seeds\t20\t--loss-seed\t0"
                    "\t--device\tauto"
                ),
            ]
        )
    assert skipped_lines == expected_skipped_lines
    assert all(not line.startswith("download_webster.sh") for line in skipped_lines)
    assert all(skipped.stdout.count(f"[{stage}/7]") == 3 for stage in range(1, 8))
    assert skipped.stdout.count("Webster data preparation skipped") == 3
    assert "download progress marker" not in skipped.stderr
    assert skipped.stdout.count("Running Theorem 1 loss–recovery experiment") == 3
    assert "All-model pipeline complete." in skipped.stdout

    log.unlink()
    default_run = subprocess.run(
        ["bash", str(run_all), "--model", "realvis"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    default_lines = log.read_text(encoding="utf-8").splitlines()
    assert default_run.returncode == 0, default_run.stderr
    assert [line.split("\t", 1)[0] for line in default_lines] == [
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "theorem1_loss_recovery.sh",
    ]
    assert default_lines[-1] == (
        "theorem1_loss_recovery.sh\t--model\trealvis\t--scheduler\tddim"
        "\t--g\t7.5\t--T\t50\t--N\t20\t--num-loss-seeds\t20"
        "\t--loss-seed\t0\t--device\tauto"
    )
    stage_offsets = [default_run.stdout.index(f"[{stage}/7]") for stage in range(1, 8)]
    assert stage_offsets == sorted(stage_offsets)

    frozen = project / (
        "data/webster/selection/realisticvision/"
        "realvis_ddpm_g3.25_T12_N4/reference_S4_N4"
    )
    frozen.mkdir(parents=True)
    log.unlink()
    reused = subprocess.run(
        [
            "bash",
            str(run_all),
            "--model",
            "realvis",
            "--scheduler",
            "ddpm",
            "--g",
            "+3.250e0",
            "--T",
            "12",
            "--N",
            "4",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    reused_lines = log.read_text(encoding="utf-8").splitlines()
    assert reused.returncode == 0, reused.stderr
    normalized_reference = (
        "\t--model\trealvis\t--scheduler\tddpm\t--g\t3.25"
        "\t--T\t12\t--N\t4\t--seed-start\t4"
    )
    normalized_experiment = (
        "\t--model\trealvis\t--scheduler\tddpm\t--g\t3.25"
        "\t--T\t12\t--N\t4\t--seed-start\t0"
    )
    assert reused_lines == [
        f"generate.sh{normalized_reference}\t--device\tauto\t--downscale\t4",
        f"compute_proximity.sh{normalized_reference}",
        f"generate.sh{normalized_experiment}\t--device\tauto\t--downscale\t4",
        f"sscd.sh{normalized_experiment}\t--device\tauto",
        f"compute_proximity.sh{normalized_experiment}",
        (
            "theorem1_loss_recovery.sh\t--model\trealvis"
            "\t--scheduler\tddpm\t--g\t3.25\t--T\t12\t--N\t4"
            "\t--num-loss-seeds\t20\t--loss-seed\t0"
            "\t--device\tauto"
        ),
    ]
    stage_offsets = [reused.stdout.index(f"[{stage}/6]") for stage in range(1, 7)]
    assert stage_offsets == sorted(stage_offsets)
    assert "Resuming cached proximity-selection reference previews" in reused.stdout
    assert "Validating and reusing frozen prompt selection" in reused.stdout
    frozen.rmdir()

    log.unlink()
    plot_only = subprocess.run(
        [
            "bash",
            str(run_all),
            "--plot",
            "--model",
            "sdv2",
            "--scheduler",
            "ddpm",
            "--g",
            "+3.250e0",
            "--T",
            "12",
            "--N",
            "4",
            "--num-loss-seeds",
            "6",
            "--loss-seed=13",
            "--device",
            "  CUDA:2  ",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert plot_only.returncode == 0, plot_only.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "theorem1_loss_recovery.sh\t--model\tsdv2\t--scheduler\tddpm"
        "\t--g\t3.25\t--T\t12\t--N\t4\t--num-loss-seeds\t6"
        "\t--loss-seed\t13\t--device\tcuda:2\t--plot"
    ]
    assert "[1/1]" in plot_only.stdout

    log.unlink()
    all_plots = subprocess.run(
        ["bash", str(run_all), "--plot"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert all_plots.returncode == 0, all_plots.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        (
            "theorem1_loss_recovery.sh\t--model\tsdv1\t--scheduler\tddim"
            "\t--g\t7.5\t--T\t50\t--N\t20\t--num-loss-seeds\t20"
            "\t--loss-seed\t0\t--device\tauto\t--plot"
        ),
        (
            "theorem1_loss_recovery.sh\t--model\tsdv2\t--scheduler\tddim"
            "\t--g\t7.5\t--T\t50\t--N\t20\t--num-loss-seeds\t20"
            "\t--loss-seed\t0\t--device\tauto\t--plot"
        ),
        (
            "theorem1_loss_recovery.sh\t--model\trealvis\t--scheduler\tddim"
            "\t--g\t7.5\t--T\t50\t--N\t20\t--num-loss-seeds\t20"
            "\t--loss-seed\t0\t--device\tauto\t--plot"
        ),
    ]
    assert all_plots.stdout.count("[1/1]") == 3
    assert "All-model pipeline complete." in all_plots.stdout

    log.unlink()
    all_defaults = subprocess.run(
        ["bash", str(run_all)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert all_defaults.returncode == 0, all_defaults.stderr
    all_default_lines = log.read_text(encoding="utf-8").splitlines()
    expected_default_wrappers = expected_stage_wrappers
    assert [line.split("\t", 1)[0] for line in all_default_lines] == (
        expected_default_wrappers * 3
    )
    assert [line.split("\t")[2] for line in all_default_lines] == (
        ["sdv1"] * 7 + ["sdv2"] * 7 + ["realvis"] * 7
    )

    log.unlink()
    failure_environment = dict(
        environment,
        FAIL_WRAPPER="generate.sh",
        FAIL_MODEL="sdv2",
        FAIL_EXIT_CODE="17",
    )
    failed_model = subprocess.run(
        ["bash", str(run_all), "--N", "1"],
        cwd=tmp_path,
        env=failure_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    failed_model_lines = log.read_text(encoding="utf-8").splitlines()
    assert failed_model.returncode == 17
    assert [line.split("\t", 1)[0] for line in failed_model_lines] == [
        *expected_stage_wrappers,
        "generate.sh",
    ]
    assert [line.split("\t")[2] for line in failed_model_lines] == (
        ["sdv1"] * 7 + ["sdv2"]
    )
    assert "pipeline failed for model sdv2 (exit 17)" in failed_model.stderr
    assert "\t--model\trealvis\t" not in log.read_text(encoding="utf-8")

    log.unlink()
    environment["DOWNLOAD_EXIT_CODE"] = "0"
    downloaded = subprocess.run(
        ["bash", str(run_all), "--download", "--N", "1"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    downloaded_lines = log.read_text(encoding="utf-8").splitlines()
    assert downloaded.returncode == 0, downloaded.stderr
    assert [line.split("\t", 1)[0] for line in downloaded_lines].count(
        "download_webster.sh"
    ) == 1
    assert [line.split("\t")[2] for line in downloaded_lines[1:]] == (
        ["sdv1"] * 7 + ["sdv2"] * 7 + ["realvis"] * 7
    )

    log.unlink()
    large_seed_count = subprocess.run(
        [
            "bash",
            str(run_all),
            "--model",
            "sdv1",
            "--scheduler",
            "ddpm",
            "--N",
            "21",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert large_seed_count.returncode == 0, large_seed_count.stderr
    large_seed_count_lines = log.read_text(encoding="utf-8").splitlines()
    assert "\t--N\t21\t--seed-start\t21" in large_seed_count_lines[0]
    assert "\t--N\t21\t--seed-start\t0" in large_seed_count_lines[3]
    assert large_seed_count_lines[-1] == (
        "theorem1_loss_recovery.sh\t--model\tsdv1\t--scheduler\tddpm"
        "\t--g\t7.5\t--T\t50\t--N\t21\t--num-loss-seeds\t20"
        "\t--loss-seed\t0\t--device\tauto"
    )

    log.unlink()
    invalid_device = subprocess.run(
        ["bash", str(run_all), "--device", "cuda:all"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert invalid_device.returncode == 2
    assert not log.exists(), "invalid device must fail before any stage"
    assert "invalid --device" in invalid_device.stderr

    for invalid_n in ("0", "not-an-integer", "4611686018427387905"):
        invalid = subprocess.run(
            ["bash", str(run_all), "--N", invalid_n],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert invalid.returncode == 2
        assert not log.exists(), "invalid N must fail before any stage"
        assert "invalid --N" in invalid.stderr

    for invalid_guidance in ("nan", "inf", "1e999999"):
        invalid = subprocess.run(
            ["bash", str(run_all), "--g", invalid_guidance],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert invalid.returncode == 2
        assert not log.exists(), "invalid guidance must fail before any stage"
        assert "invalid --g" in invalid.stderr

    invalid_theorem_values = (
        ("--num-loss-seeds", "0"),
        ("--num-loss-seeds", "not-an-integer"),
        ("--loss-seed", "-1"),
        ("--loss-seed", "not-an-integer"),
        ("--loss-seed", "9223372036854775808"),
    )
    for option, value in invalid_theorem_values:
        invalid = subprocess.run(
            ["bash", str(run_all), option, value],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert invalid.returncode == 2
        assert not log.exists(), f"invalid {option} must fail before any stage"
        assert f"invalid {option}" in invalid.stderr


def _imports(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.add(node.module)
    return values


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
            "recovery.py",
            "selection.py",
            "state.py",
            "wayback.py",
            "webster.py",
        },
        "utils/experiments": {
            "__init__.py",
            "cache.py",
            "generation.py",
            "plotting.py",
            "latent_statistics.py",
            "proximity.py",
            "sscd.py",
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
        "compute_proximity.py",
        "download_webster.py",
        "generate.py",
        "sscd.py",
        "theorem1_loss_recovery.py",
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
