"""Offline generation-cache and flattened-architecture tests."""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from utils.common.cli import (
    MAX_SEED,
    add_generation_arguments,
    add_run_arguments,
    generation_run_name,
)
from utils.common.io import (
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.experiments import generation as generation_module
from utils.experiments.cache import (
    GENERATION_SELECTION_POLICY,
    GENERATION_SCHEMA_VERSION,
    SAMPLER_CONTRACT_VERSION,
    DiskEstimate,
    GenerationCacheError,
    GenerationPaths,
    generation_paths,
    publish_completion_marker,
    require_generation_run,
    save_generation_tensors,
    validate_generation_record,
)
from utils.models.latent import TARGET_LATENT_DEFINITION
from utils.models.registry import get_model_spec


ROOT = Path(__file__).resolve().parents[1]


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
    }
    assert generation_parser.parse_args(["--downscale", "8"]).downscale == 8
    assert generation_parser.parse_args(["--seed-start", "20"]).seed_start == 20
    with pytest.raises(SystemExit):
        generation_parser.parse_args(["--downscale", "0"])
    with pytest.raises(SystemExit):
        generation_parser.parse_args(["--seed-start", "-1"])
    with pytest.raises(SystemExit):
        generation_parser.parse_args(["--seed-start", str(MAX_SEED + 1)])

    analysis_parser = add_run_arguments(argparse.ArgumentParser(allow_abbrev=False))
    assert not hasattr(analysis_parser.parse_args([]), "downscale")


def test_seed_blocks_have_collision_free_shared_log_namespaces(
    tmp_path: Path,
) -> None:
    common = ("sdv1", "ddim", 7.5, 50, 20)
    assert generation_run_name(*common) == "sdv1_ddim_g7.5_T50_N20"
    assert generation_run_name(*common, seed_start=0) == "sdv1_ddim_g7.5_T50_N20"
    assert (
        generation_run_name(*common, seed_start=20)
        == "sdv1_ddim_g7.5_T50_S20_N20"
    )
    zero = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
    )
    reference = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=20,
    )
    parent = tmp_path.resolve() / "logs/sdv1_ddim_g7.5_T50_N20"
    assert zero.run_directory == parent / "experiment_S0_N20"
    assert reference.run_directory == parent / "reference_S20_N20"
    assert zero.run_directory.parent == reference.run_directory.parent
    assert zero.run_directory != reference.run_directory
    generic = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddpm",
        guidance_scale=7.5,
        num_inference_steps=50,
        num_seeds=20,
        seed_start=20,
    )
    assert generic.run_directory == (
        tmp_path.resolve()
        / "logs/sdv1_ddpm_g7.5_T50_N20/seed_S20_N20"
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
            num_seeds=2,
            seed_start=MAX_SEED,
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
        "scheduler": {"name": "ddim"},
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


def _publish_cached_record(
    paths: GenerationPaths,
    identity: dict[str, object],
    science_hash: str,
    *,
    steps: int = 50,
    seeds: int = 20,
    seed_start: int = 0,
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
    before = {name: file_sha256(path) for name, path in {
        "latent": paths.latent_path("7"),
        "noise_prediction": paths.noise_prediction_path("7"),
        "target_latent": paths.target_latent_path("7"),
    }.items()}

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
    assert before == {name: file_sha256(path) for name, path in {
        "latent": paths.latent_path("7"),
        "noise_prediction": paths.noise_prediction_path("7"),
        "target_latent": paths.target_latent_path("7"),
    }.items()}


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
    assert GENERATION_SELECTION_POLICY == "all_available_canonical_pairs_label_independent"
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
    assert "marker prompt_raw differs from current Webster metadata" in wrong_identity.errors
    paths.latent_path(7).write_bytes(b"tampered latent")
    corrupted = validate_generation_record(paths, 7, load_tensors=False)
    assert not corrupted.valid
    assert "latent SHA-256 differs" in corrupted.errors
    with pytest.raises(GenerationCacheError, match="refusing to overwrite"):
        publish_completion_marker(paths, 7, metadata)


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
    atomic_write_json(
        paths.run_config,
        {
            "scientific_config": science,
            "scientific_config_hash": science_hash,
            "preview_config_hash": "f" * 64,
        },
    )
    for row in rows:
        _publish_cached_record(
            paths, row, science_hash, seed_start=seed_start
        )
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
        tmp_path, seed_start=seed_start
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
    monkeypatch.setattr(generation_module, "validate_generation_record", fail_validation)
    monkeypatch.setattr(
        generation_module,
        "estimate_disk_space",
        lambda *_, **__: DiskEstimate(1, 1, 1, 0, 10**12),
    )

    result = generation_module.generate_webster_trajectories(tmp_path)

    assert result.exit_code == 1
    assert result.failed_rows == 1
    progress_output = capsys.readouterr().err
    assert (
        "[Generation] Record 77 failed: RuntimeError: "
        "synthetic validation failure"
    ) in progress_output
    assert "traceback:" in progress_output
    assert "1/1" in progress_output


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
    assert 'BASH_SOURCE[0]' in source
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
            "--per-host",
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
            "bash", str(run_all), "--download", "--model=sdv2",
            "--scheduler", "ddpm",
            "--g=3.25", "--T", "12", "--N=4", "--downscale", "8",
            "--direct-workers=12", "--direct-attempts", "1", "--per-host=3",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    reference = (
        "\t--model\tsdv2\t--scheduler\tddim\t--g\t7.5"
        "\t--T\t50\t--N\t20\t--seed-start\t20"
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
        f"generate.sh{reference}\t--downscale\t8",
        f"sscd.sh{reference}",
        f"compute_proximity.sh{reference}",
        f"generate.sh{experiment}\t--downscale\t8",
        f"sscd.sh{experiment}",
        f"compute_proximity.sh{experiment}",
    ]
    stage_offsets = [
        result.stdout.index(f"[{stage}/7]")
        for stage in range(1, 8)
    ]
    assert stage_offsets == sorted(stage_offsets)
    assert "retryable records" not in result.stderr
    assert "download progress marker" in result.stderr

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
        ["bash", str(run_all), "--N", "1"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    skipped_lines = log.read_text(encoding="utf-8").splitlines()
    assert skipped.returncode == 0, skipped.stderr
    assert [line.split("\t", 1)[0] for line in skipped_lines] == [
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
    ]
    assert all(
        not line.startswith("download_webster.sh") for line in skipped_lines
    )
    stage_offsets = [
        skipped.stdout.index(f"[{stage}/6]")
        for stage in range(1, 7)
    ]
    assert stage_offsets == sorted(stage_offsets)
    assert "Webster data preparation skipped" in skipped.stdout
    assert "download progress marker" not in skipped.stderr
    assert "Theorem 1 loss–recovery skipped" in skipped.stdout

    log.unlink()
    canonical = subprocess.run(
        ["bash", str(run_all), "--model", "realvis"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    canonical_lines = log.read_text(encoding="utf-8").splitlines()
    assert canonical.returncode == 0, canonical.stderr
    assert [line.split("\t", 1)[0] for line in canonical_lines] == [
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "generate.sh",
        "sscd.sh",
        "compute_proximity.sh",
        "theorem1_loss_recovery.sh",
    ]
    assert canonical_lines[-1] == (
        "theorem1_loss_recovery.sh\t--model\trealvis"
    )
    stage_offsets = [
        canonical.stdout.index(f"[{stage}/7]")
        for stage in range(1, 8)
    ]
    assert stage_offsets == sorted(stage_offsets)

    log.unlink()
    plot_only = subprocess.run(
        ["bash", str(run_all), "--plot", "--model", "sdv2"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert plot_only.returncode == 0, plot_only.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "theorem1_loss_recovery.sh\t--model\tsdv2\t--plot"
    ]
    assert "[1/1]" in plot_only.stdout

    log.unlink()
    environment["DOWNLOAD_EXIT_CODE"] = "0"
    for invalid_n in ("0", "21"):
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
        assert not log.exists(), "invalid experiment N must fail before any stage"
        assert "invalid --N" in invalid.stderr



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


def test_source_tree_has_no_legacy_modules_imports_or_dynamic_aliases() -> None:
    allowed = {
        "utils/common": {"__init__.py", "cli.py", "io.py"},
        "utils/data": {
            "__init__.py", "arquivo.py", "commoncrawl.py", "images.py",
            "mirror.py", "official.py", "recovery.py", "selection.py", "state.py",
            "wayback.py", "webster.py",
        },
            "utils/experiments": {
                "__init__.py", "cache.py", "generation.py", "plotting.py",
                "held_out.py", "latent_statistics.py", "proximity.py", "sscd.py",
            },
        "utils/metrics": {"__init__.py", "sscd.py"},
        "utils/models": {
            "__init__.py", "devices.py", "latent.py", "loading.py",
            "prediction_conversion.py", "registry.py", "sampling.py",
            "schedule_metadata.py", "schedulers.py",
        },
    }
    for relative, permitted in allowed.items():
        observed = {path.name for path in (ROOT / relative).glob("*.py")}
        assert not observed - permitted, f"legacy Python files in {relative}: {sorted(observed - permitted)}"
    script_files = {
        path.relative_to(ROOT / "scripts").as_posix()
        for path in (ROOT / "scripts").rglob("*.py")
    }
    assert script_files == {
        "compute_proximity.py", "download_webster.py", "generate.py", "sscd.py",
        "theorem1_loss_recovery.py",
    }
    assert not (ROOT / "utils/runtime_provenance.py").exists()

    forbidden_import_fragments = (
        "utils.data.curation", "utils.data.webster_dataset",
        "utils.experiments.definition1", "utils.experiments.generation_",
        "utils.experiments.completed_records", "utils.experiments.sscd_evaluation",
        "utils.experiments.proximity_sscd_plot", "utils.models.latent_space",
        "utils.runtime_provenance", "scripts.data", "scripts.experiments",
    )
    for source_path in [*(ROOT / "utils").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        imports = _imports(tree)
        assert not {
            name for name in imports
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
    assert {"__pycache__/", "*.py[cod]", ".pytest_cache/", ".DS_Store", "__MACOSX/"} <= ignore_rules

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    tracked_paths = [
        Path(value.decode("utf-8"))
        for value in result.stdout.split(b"\0")
        if value
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
