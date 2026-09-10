"""Offline provenance and seed-alignment tests for the standalone cache reader."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from utils.common.io import (
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    read_json,
)
from utils.experiments import forward_corruptions_cache as cache
from utils.experiments.cache import (
    CompletedGenerationRecord,
    GenerationCacheError,
    generation_paths,
    publish_completion_marker,
)
from utils.experiments import sscd
from utils.models.latent import TARGET_LATENT_DEFINITION, target_preprocessing_policy
from utils.models.schedule_metadata import build_schedule_metadata
from utils.models.schedulers import build_scheduler_from_config, scheduler_config_dict


def _make_run(root: Path, *, seed_start: int = 0) -> cache.CachedRun:
    paths = generation_paths(
        root,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=2,
        num_seeds=20,
        seed_start=seed_start,
    )
    paths.create()
    scheduler = build_scheduler_from_config(
        {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "clip_sample": False,
            "set_alpha_to_one": False,
            "steps_offset": 1,
            "prediction_type": "epsilon",
            "timestep_spacing": "leading",
        },
        "ddim",
    ).scheduler
    schedule = build_schedule_metadata(
        scheduler, "ddim", num_inference_steps=2, device="cpu"
    )
    seeds = list(range(seed_start, seed_start + 20))
    science = {
        "schema_version": 1,
        "sampler_contract_version": 2,
        "selection_policy": "all_available_canonical_pairs_label_independent",
        "target_role": "canonical_paired_source",
        "model_cli_name": "sdv1",
        "dataset_model": "sdv1",
        "model_id": "CompVis/stable-diffusion-v1-4",
        "model_revision": "a" * 40,
        "vae_id": "CompVis/stable-diffusion-v1-4",
        "vae_revision": "a" * 40,
        "resolution": 512,
        "scheduler": {
            "name": "ddim",
            "class": "DDIMScheduler",
            "config": scheduler_config_dict(scheduler),
        },
        "guidance_scale": 7.5,
        "num_inference_steps": 2,
        "num_seeds": 20,
        "seeds": seeds,
        "latent_shape": [4, 64, 64],
        "inference_dtype": "float16",
        "scientific_tensor_storage": {
            "device": "cpu",
            "dtype": "float16",
            "layout": "contiguous",
        },
        "native_prediction_type": "epsilon",
        "stored_prediction_type": "epsilon",
        "trajectory_order": "noise_to_image",
        "target_latent_definition": TARGET_LATENT_DEFINITION,
        "target_preprocessing": target_preprocessing_policy(512),
    }
    configuration = {
        "scientific_config": science,
        "scientific_config_hash": canonical_hash(science),
    }
    atomic_write_json(paths.run_config, configuration)
    atomic_torch_save(schedule, paths.schedule)
    scores = sscd.SSCDPaths(paths.run_directory)
    scores.create()
    score_configuration = {
        "schema_version": 1,
        "generation_run_path": paths.run_directory.relative_to(root).as_posix(),
        "generation_run_config_path": paths.run_config.relative_to(root).as_posix(),
        "generation_scientific_config_hash": configuration["scientific_config_hash"],
        "selection_policy": sscd.SSCD_SELECTION_POLICY,
        "num_seeds": 20,
        "seeds": seeds,
        "terminal_latent_index": -1,
        "decode_dtype": "float32",
        "score_definition": sscd.SCORE_DEFINITION,
        "score_tensor_schema": {
            "shape": ["num_seeds"],
            "dtype": "float32",
            "device": "cpu",
            "contiguous": True,
            "order": "seed 0 through seed N - 1"
            if seed_start == 0
            else "same order as explicit seeds",
        },
        "duplicates_images_or_features": False,
        "feature_normalization": "explicit_l2_p2_dim1",
        "sscd_model_name": sscd.SSCD_MODEL_NAME,
        "sscd_checkpoint_path": "checkpoints/sscd/sscd_disc_large.torchscript.pt",
        "sscd_checkpoint_url": sscd.SSCD_CHECKPOINT_URL,
        "sscd_checkpoint_sha256": "b" * 64,
        "sscd_feature_dimension": sscd.SSCD_FEATURE_DIMENSION,
        "sscd_input_size": sscd.SSCD_INPUT_SIZE,
        "sscd_preprocessing": sscd.sscd_preprocessing_policy(),
        "sscd_preprocessing_hash": sscd.sscd_preprocessing_hash(),
        **{
            key: science[key]
            for key in ("model_id", "model_revision", "vae_id", "vae_revision")
        },
    }
    score_configuration["configuration_hash"] = sscd.sscd_configuration_hash(
        score_configuration
    )
    atomic_write_json(scores.config_json, score_configuration)
    return cache.load_run(root, num_inference_steps=2, seed_start=seed_start)


def _record(
    run: cache.CachedRun, *, index: str = "example"
) -> CompletedGenerationRecord:
    noise = cache.make_initial_noise(run.seeds, run.latent_shape).to(run.stored_dtype)
    states = torch.stack([noise, noise + 0.125, noise + 0.25], dim=1).contiguous()
    target = torch.linspace(-1.0, 1.0, 4 * 64 * 64, dtype=torch.float32).reshape(
        4, 64, 64
    )
    hashes = {
        "latent": atomic_torch_save(states, run.paths.latent_path(index)),
        "target_latent": atomic_torch_save(target, run.paths.target_latent_path(index)),
        "noise_prediction": "c" * 64,  # Not needed and intentionally absent.
    }
    marker = {
        **dict(run.science),
        "scientific_config_hash": run.scientific_hash,
        "record_id": "sdv1-0001",
        "source_row_number": 1,
        "prompt_raw": "fixed exact prompt",
        "target_image_sha256": "d" * 64,
        "scheduler_name": "ddim",
        "scheduler_class": "DDIMScheduler",
        "tensor_file_sha256": hashes,
        "tensor_shapes": {
            "latent": list(states.shape),
            "target_latent": list(target.shape),
        },
        "tensor_dtypes": {"latent": "float16", "target_latent": "float32"},
    }
    marker_path = publish_completion_marker(run.paths, index, marker)
    metadata = read_json(marker_path)
    return CompletedGenerationRecord(index, 1, marker_path, metadata)


def _scores(run: cache.CachedRun, record: CompletedGenerationRecord) -> torch.Tensor:
    scores = torch.linspace(-0.1, 0.95, 20, dtype=torch.float32)
    score_hash = atomic_torch_save(
        scores, run.sscd_paths.score_path(record.original_index)
    )
    marker = {
        "schema_version": 1,
        "original_index": record.original_index,
        "record_id": record.metadata["record_id"],
        "source_row_number": record.source_row_number,
        "prompt_raw": record.metadata["prompt_raw"],
        "num_seeds": 20,
        "seeds": list(run.seeds),
        "sscd_configuration_hash": run.sscd_configuration["configuration_hash"],
        "generation_latent_sha256": record.metadata["tensor_file_sha256"]["latent"],
        "generation_scientific_config_hash": run.scientific_hash,
        "target_image_sha256": record.metadata["target_image_sha256"],
        "score_shape": [20],
        "score_dtype": "float32",
        "terminal_latent_index": -1,
        "similarity": sscd.SCORE_DEFINITION,
        "score_sha256": score_hash,
        "sscd_checkpoint_sha256": run.sscd_configuration["sscd_checkpoint_sha256"],
        **{
            key: run.science[key]
            for key in ("model_id", "model_revision", "vae_id", "vae_revision")
        },
    }
    atomic_write_json(run.sscd_paths.marker_path(record.original_index), marker)
    return scores


def test_load_run_uses_native_schedule_and_records_nonideal_final_boundary(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path)
    assert run.timesteps.tolist() == [501, 1]
    assert run.alpha.dtype == run.sigma.dtype == torch.float32
    assert run.seeds == tuple(range(20))
    assert run.stored_dtype == torch.float16
    assert run.latent_shape == (4, 64, 64)
    assert run.scheduler_final_alpha_cumprod == pytest.approx(0.99915)
    assert run.scheduler_final_alpha_cumprod != 1.0
    assert len(run.alpha) == len(run.sigma) == 2


@pytest.mark.parametrize("seed_start", [-1, True, 1.5, 2**63 - 10])
def test_load_run_rejects_invalid_seed_blocks_before_reading(
    tmp_path: Path, seed_start: object
) -> None:
    with pytest.raises(ValueError):
        cache.load_run(tmp_path, seed_start=seed_start)
    assert not list(tmp_path.iterdir())


def test_reference_and_evaluation_may_use_any_disjoint_seed_blocks(
    tmp_path: Path,
) -> None:
    evaluation = _make_run(tmp_path, seed_start=40)
    reference = _make_run(tmp_path, seed_start=80)
    cache.validate_run_pair(reference, evaluation)
    assert evaluation.seeds == tuple(range(40, 60))
    assert reference.seeds == tuple(range(80, 100))


def test_run_pair_rejects_overlap_and_model_or_score_changes(tmp_path: Path) -> None:
    evaluation = _make_run(tmp_path)
    reference = _make_run(tmp_path, seed_start=20)
    cache.validate_run_pair(reference, evaluation)
    with pytest.raises(cache.ForwardCorruptionsCacheError, match="overlap"):
        cache.validate_run_pair(
            replace(reference, seeds=tuple(range(10, 30))), evaluation
        )
    with pytest.raises(cache.ForwardCorruptionsCacheError, match="model_revision"):
        cache.validate_run_pair(
            replace(
                reference, science={**reference.science, "model_revision": "changed"}
            ),
            evaluation,
        )
    with pytest.raises(cache.ForwardCorruptionsCacheError, match="sscd_checkpoint"):
        cache.validate_run_pair(
            replace(
                reference,
                sscd_configuration={
                    **reference.sscd_configuration,
                    "sscd_checkpoint_sha256": "f" * 64,
                },
            ),
            evaluation,
        )


@pytest.mark.parametrize(
    "damage",
    ["science_hash", "schedule_values", "init_scale", "sscd_hash", "sscd_seeds"],
)
def test_load_run_rejects_corrupt_scientific_inputs_without_repair(
    tmp_path: Path, damage: str
) -> None:
    run = _make_run(tmp_path)
    if damage == "science_hash":
        atomic_write_json(
            run.paths.run_config,
            {**run.configuration, "scientific_config_hash": "f" * 64},
        )
    elif damage in ("schedule_values", "init_scale"):
        schedule = dict(run.schedule_payload)
        if damage == "schedule_values":
            schedule["alpha_t"] = schedule["alpha_t"] + 0.01
        else:
            schedule["init_noise_sigma"] = 2.0
        atomic_torch_save(schedule, run.paths.schedule)
    else:
        saved = dict(run.sscd_configuration)
        if damage == "sscd_hash":
            saved["configuration_hash"] = "f" * 64
        else:
            saved["seeds"] = list(range(20, 40))
            saved["configuration_hash"] = sscd.sscd_configuration_hash(saved)
        atomic_write_json(run.sscd_paths.config_json, saved)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, GenerationCacheError)):
        cache.load_run(tmp_path, num_inference_steps=2)
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before


def test_load_scores_is_read_only_and_preserves_raw_negative_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _make_run(tmp_path)
    record = _record(run)
    expected = _scores(run, record)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("cache reading must not create, quarantine, or compute SSCD")

    monkeypatch.setattr(sscd, "_load_cached_scores", forbidden)
    monkeypatch.setattr(sscd, "_quarantine_score_artifacts", forbidden)
    monkeypatch.setattr(sscd, "ensure_sscd_checkpoint", forbidden)
    torch.testing.assert_close(cache.load_scores(run, record), expected, rtol=0, atol=0)
    assert float(cache.load_scores(run, record)[0]) < 0
    assert not run.paths.noise_prediction_path(record.original_index).exists()


@pytest.mark.parametrize(
    "damage", ["target", "seed_order", "score_hash", "prompt", "missing"]
)
def test_load_scores_rejects_stale_or_missing_cache_without_mutation(
    tmp_path: Path, damage: str
) -> None:
    run = _make_run(tmp_path)
    record = _record(run)
    _scores(run, record)
    path = run.sscd_paths.marker_path(record.original_index)
    marker = read_json(path)
    if damage == "target":
        marker["target_image_sha256"] = "f" * 64
    elif damage == "seed_order":
        marker["seeds"] = list(reversed(marker["seeds"]))
    elif damage == "score_hash":
        marker["score_sha256"] = "f" * 64
    elif damage == "prompt":
        marker["prompt_raw"] = "different prompt"
    else:
        run.sscd_paths.score_path(record.original_index).unlink()
    atomic_write_json(path, marker)
    before = {
        item: item.read_bytes() for item in run.sscd_paths.record_directory.iterdir()
    }
    with pytest.raises((ValueError, sscd.SSCDEvaluationError)):
        cache.load_scores(run, record)
    assert {
        item: item.read_bytes() for item in run.sscd_paths.record_directory.iterdir()
    } == before
    assert not run.sscd_paths.stale_directory.exists()


def test_load_states_reads_only_latent_and_fixed_target_with_exact_seed_rounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = _make_run(tmp_path)
    reference = _make_run(tmp_path, seed_start=20)
    chosen = _record(reference)
    evaluated = _record(evaluation)
    observed_paths = []
    original = cache.safe_torch_load

    def capture(path: Path) -> object:
        observed_paths.append(path)
        return original(path)

    monkeypatch.setattr(cache, "safe_torch_load", capture)
    states, target, marker = cache.load_states(evaluation, chosen)
    assert observed_paths == [
        evaluation.paths.latent_path(chosen.original_index),
        evaluation.paths.target_latent_path(chosen.original_index),
    ]
    assert states.shape == (20, 3, 4, 64, 64)
    assert target.shape == (4, 64, 64)
    assert target.dtype == torch.float32
    assert marker == evaluated.metadata
    raw = cache.make_initial_noise(evaluation.seeds, evaluation.latent_shape)
    assert torch.equal(states[:, 0], raw.half())
    assert not torch.equal(states[:, 0].float(), raw)


@pytest.mark.parametrize(
    "damage", ["seed_initial", "nonfinite", "target_shape", "paired_target", "prompt"]
)
def test_load_states_rejects_misaligned_or_corrupt_inputs(
    tmp_path: Path, damage: str
) -> None:
    run = _make_run(tmp_path)
    record = _record(run)
    metadata = dict(record.metadata)
    hashes = dict(metadata["tensor_file_sha256"])
    if damage in ("seed_initial", "nonfinite"):
        states = cache.safe_torch_load(run.paths.latent_path(record.original_index))
        if damage == "seed_initial":
            states[0, 0, 0, 0, 0] += 1
        else:
            # Construct a malformed decoded tensor without publishing it.
            states[0, 1, 0, 0, 0] = float("nan")
        if damage == "nonfinite":
            # atomic_torch_save intentionally rejects nonfinite values; test the loader
            # with a valid file hash and an injected malformed decoded tensor instead.
            with pytest.MonkeyPatch.context() as patch:
                original = cache.safe_torch_load
                patch.setattr(
                    cache,
                    "safe_torch_load",
                    lambda path: (
                        states
                        if Path(path) == run.paths.latent_path(record.original_index)
                        else original(path)
                    ),
                )
                with pytest.raises(cache.ForwardCorruptionsCacheError, match="tensor"):
                    cache.load_states(run, record)
            return
        hashes["latent"] = atomic_torch_save(
            states, run.paths.latent_path(record.original_index)
        )
    elif damage == "target_shape":
        hashes["target_latent"] = atomic_torch_save(
            torch.zeros(1), run.paths.target_latent_path(record.original_index)
        )
    elif damage == "paired_target":
        hashes["target_latent"] = "f" * 64
        changed = replace(record, metadata={**metadata, "tensor_file_sha256": hashes})
        with pytest.raises(
            cache.ForwardCorruptionsCacheError, match="target latent hashes"
        ):
            cache.load_states(run, changed)
        return
    else:
        metadata["prompt_raw"] = "different paired prompt"
    metadata["tensor_file_sha256"] = hashes
    atomic_write_json(record.marker_path, metadata)
    reference = record if damage == "prompt" else replace(record, metadata=metadata)
    with pytest.raises(cache.ForwardCorruptionsCacheError):
        cache.load_states(run, reference)
