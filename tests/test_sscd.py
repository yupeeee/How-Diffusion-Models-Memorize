"""Model-free tests for SSCD preprocessing and cache reuse."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from utils.common.cli import MAX_SEED
from utils.common.io import (
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
)
from utils.experiments import sscd as sscd_module
from utils.experiments.cache import (
    CompletedGenerationRecord,
    GenerationPaths,
    generation_paths,
    publish_completion_marker,
    save_generation_tensors,
)
from utils.experiments.sscd import (
    SCORE_DEFINITION,
    SSCD_SELECTION_POLICY,
    SSCD_SCHEMA_VERSION,
    SSCDPaths,
    SSCDEvaluationError,
    _load_cached_scores,
    sscd_configuration_hash,
)
from utils.metrics import sscd as sscd_metrics_module
from utils.metrics.sscd import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SSCD_FEATURE_DIMENSION,
    SSCD_INPUT_SIZE,
    SSCD_CHECKPOINT_URL,
    SSCD_MODEL_NAME,
    SSCDMetricError,
    cosine_similarity_scores,
    preprocess_sscd_images,
    sscd_preprocessing_hash,
    sscd_preprocessing_policy,
    validate_sscd_scores,
)


def test_sscd_preprocessing_is_fixed_full_resolution_float32() -> None:
    batch = preprocess_sscd_images([Image.new("RGB", (3, 5), (0, 0, 0))])
    assert batch.shape == (1, 3, SSCD_INPUT_SIZE, SSCD_INPUT_SIZE)
    assert batch.dtype is torch.float32
    assert batch.device.type == "cpu"
    expected = torch.tensor(
        [-mean / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)]
    )
    torch.testing.assert_close(batch[0, :, 0, 0], expected)
    assert sscd_preprocessing_policy()["augmentation"] is False
    assert len(sscd_preprocessing_hash()) == 64
    assert SSCD_SELECTION_POLICY == "all_completed_generation_records"


def test_checkpoint_download_reports_streamed_byte_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = b"synthetic SSCD checkpoint"

    class Response:
        headers = {"content-length": str(len(payload))}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *arguments: object) -> None:
            del arguments

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, *, chunk_size: int):
            assert chunk_size == sscd_metrics_module._DOWNLOAD_CHUNK_SIZE
            yield payload[:7]
            yield b""
            yield payload[7:]

    class Client:
        def stream(
            self,
            method: str,
            url: str,
            *,
            follow_redirects: bool,
        ) -> Response:
            assert method == "GET"
            assert url == SSCD_CHECKPOINT_URL
            assert follow_redirects
            return Response()

    monkeypatch.setattr(
        sscd_metrics_module, "_validate_torchscript", lambda _path: None
    )
    destination = tmp_path / "checkpoint.pt"
    expected_hash = hashlib.sha256(payload).hexdigest()
    observed_hash = sscd_metrics_module._download_checkpoint(
        destination,
        client=Client(),  # type: ignore[arg-type]
        expected_hash=expected_hash,
    )

    assert observed_hash == expected_hash
    assert destination.read_bytes() == payload
    progress_output = capsys.readouterr().err
    assert "[SSCD] Checkpoint download" in progress_output
    assert "100%" in progress_output


def test_sscd_cosine_scores_preserve_generated_order() -> None:
    target = torch.zeros((1, SSCD_FEATURE_DIMENSION), dtype=torch.float32)
    target[0, 0] = 1.0
    generated = torch.zeros((2, SSCD_FEATURE_DIMENSION), dtype=torch.float32)
    generated[0, 0] = 4.0
    generated[1, 1] = 2.0
    scores = cosine_similarity_scores(target, generated)
    torch.testing.assert_close(scores, torch.tensor([1.0, 0.0]))
    with pytest.raises(SSCDMetricError, match="outside"):
        validate_sscd_scores(torch.tensor([1.1], dtype=torch.float32), 1)


def test_sscd_rejects_an_out_of_range_seed_block_before_cache_lookup(
    tmp_path: Path,
) -> None:
    with pytest.raises(SSCDEvaluationError, match="seed block must end"):
        sscd_module.run_sscd(
            tmp_path,
            model_name="sdv1",
            scheduler_name="ddim",
            guidance_scale=7.5,
            num_inference_steps=50,
            num_seeds=2,
            seed_start=MAX_SEED,
        )

    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("legacy_run", "nested_run"),
    [
        (
            "logs/sdv1_ddim_g7.5_T50_N20",
            "logs/sdv1_ddim_g7.5_T50_N20/experiment_S0_N20",
        ),
        (
            "logs/sdv1_ddim_g7.5_T50_S20_N20",
            "logs/sdv1_ddim_g7.5_T50_N20/reference_S20_N20",
        ),
    ],
)
def test_sscd_configuration_hash_is_stable_across_cache_relocation(
    legacy_run: str,
    nested_run: str,
) -> None:
    legacy = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "generation_run_path": legacy_run,
        "generation_run_config_path": f"{legacy_run}/run_config.json",
        "num_seeds": 20,
    }
    nested = {
        **legacy,
        "generation_run_path": nested_run,
        "generation_run_config_path": f"{nested_run}/run_config.json",
    }

    assert sscd_configuration_hash(legacy) == canonical_hash(legacy)
    assert sscd_configuration_hash(nested) == canonical_hash(legacy)

    unrelated = {**nested, "num_seeds": 19}
    assert sscd_configuration_hash(unrelated) != canonical_hash(legacy)


def test_cached_scores_are_reused_only_with_matching_provenance(tmp_path: Path) -> None:
    generation = GenerationPaths(tmp_path / "logs" / "synthetic")
    generation.create()
    atomic_torch_save(torch.zeros((2, 2, 1, 1, 1)), generation.latent_path("9"))
    atomic_write_json(generation.record_path("9"), {"record": "stable"})
    completed = CompletedGenerationRecord(
        "9",
        4,
        generation.record_path("9"),
        {"record_id": "sdv1-0004"},
    )
    paths = SSCDPaths(generation.run_directory)
    paths.create()
    scores = torch.tensor([0.2, 0.8], dtype=torch.float32)
    score_hash = atomic_torch_save(scores, paths.score_path("9"))
    configuration_hash = "c" * 64
    marker = {
        "schema_version": 1,
        "original_index": "9",
        "record_id": "sdv1-0004",
        "source_row_number": 4,
        "num_seeds": 2,
        "seeds": [0, 1],
        "sscd_configuration_hash": configuration_hash,
        "generation_record_sha256": file_sha256(generation.record_path("9")),
        "generation_latent_sha256": file_sha256(generation.latent_path("9")),
        "score_sha256": score_hash,
    }
    atomic_write_json(paths.marker_path("9"), marker)
    seed_values = (0, 1)
    reused = _load_cached_scores(
        paths, generation, completed, configuration_hash, seed_values
    )
    torch.testing.assert_close(reused, scores)

    generation_marker_bytes = generation.record_path("9").read_bytes()
    atomic_write_json(generation.record_path("9"), {"record": "changed"})
    with pytest.raises(SSCDEvaluationError, match="generation_record_sha256 differs"):
        _load_cached_scores(
            paths, generation, completed, configuration_hash, seed_values
        )
    generation.record_path("9").write_bytes(generation_marker_bytes)

    score_bytes = paths.score_path("9").read_bytes()
    paths.score_path("9").write_bytes(b"tampered score")
    with pytest.raises(SSCDEvaluationError, match="cached SSCD hash differs"):
        _load_cached_scores(
            paths, generation, completed, configuration_hash, seed_values
        )
    paths.score_path("9").write_bytes(score_bytes)

    marker["sscd_configuration_hash"] = "d" * 64
    atomic_write_json(paths.marker_path("9"), marker)
    with pytest.raises(SSCDEvaluationError, match="configuration_hash differs"):
        _load_cached_scores(
            paths, generation, completed, configuration_hash, seed_values
        )


@pytest.mark.parametrize("seed_start", [0, 20])
def test_sscd_resumes_every_completed_prompt_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seed_start: int,
) -> None:
    rows = (
        ("201", "sdv1-0000", 0, "TV prompt", "TV", "1" * 64),
        ("202", "sdv1-0001", 1, "non-TV prompt", "N", "2" * 64),
    )
    seed_values = [seed_start, seed_start + 1]
    generation = generation_paths(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=1,
        num_seeds=2,
        seed_start=seed_start,
    )
    generation.create()
    science = {
        "model_cli_name": "sdv1",
        "model_id": "synthetic/model",
        "model_revision": "a" * 40,
        "vae_id": "synthetic/vae",
        "vae_revision": "c" * 40,
        "scheduler": {"name": "ddim"},
        "guidance_scale": 7.5,
        "num_inference_steps": 1,
        "num_seeds": 2,
        "seeds": seed_values,
    }
    science_hash = canonical_hash(science)
    atomic_write_json(
        generation.run_config,
        {
            "scientific_config": science,
            "scientific_config_hash": science_hash,
        },
    )
    for index, record_id, source_row, prompt, label, target_hash in rows:
        latent = torch.zeros((2, 2, 1, 1, 1), dtype=torch.float32)
        unconditional = torch.zeros((2, 1, 1, 1, 1), dtype=torch.float32)
        conditional = torch.ones_like(unconditional)
        target = torch.zeros((1, 1, 1), dtype=torch.float32)
        hashes = save_generation_tensors(
            generation,
            index,
            latents=latent,
            unconditional_predictions=unconditional,
            conditional_predictions=conditional,
            target_latent=target,
        )
        generation.image_path(index).write_bytes(b"cached preview")
        publish_completion_marker(
            generation,
            index,
            {
                "record_id": record_id,
                "source_row_number": source_row,
                "prompt_raw": prompt,
                "webster_overfit_type": label,
                "target_image_sha256": target_hash,
                "scientific_config_hash": science_hash,
                "num_seeds": 2,
                "seeds": seed_values,
                "tensor_file_sha256": hashes,
                "preview_image_sha256": file_sha256(generation.image_path(index)),
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

    paths = SSCDPaths(generation.run_directory)
    paths.create()
    configuration = {
        "schema_version": SSCD_SCHEMA_VERSION,
        "generation_run_path": generation.run_directory.relative_to(tmp_path).as_posix(),
        "generation_run_config_path": generation.run_config.relative_to(tmp_path).as_posix(),
        "generation_scientific_config_hash": science_hash,
        "selection_policy": "completed_generation_cache_records",
        "num_seeds": 2,
        "seeds": seed_values,
        "terminal_latent_index": -1,
        "model_id": science["model_id"],
        "model_revision": science["model_revision"],
        "vae_id": science["vae_id"],
        "vae_revision": science["vae_revision"],
        "decode_dtype": "float32",
        "score_definition": SCORE_DEFINITION,
        "score_tensor_schema": {
            "shape": ["num_seeds"],
            "dtype": "float32",
            "device": "cpu",
            "contiguous": True,
            "order": (
                "seed 0 through seed N - 1"
                if seed_start == 0
                else "same order as explicit seeds"
            ),
        },
        "duplicates_images_or_features": False,
        "feature_normalization": "explicit_l2_p2_dim1",
        "sscd_model_name": SSCD_MODEL_NAME,
        "sscd_checkpoint_path": "checkpoints/sscd/sscd_disc_large.torchscript.pt",
        "sscd_checkpoint_url": SSCD_CHECKPOINT_URL,
        "sscd_checkpoint_sha256": "b" * 64,
        "sscd_feature_dimension": SSCD_FEATURE_DIMENSION,
        "sscd_input_size": SSCD_INPUT_SIZE,
        "sscd_preprocessing": sscd_preprocessing_policy(),
        "sscd_preprocessing_hash": sscd_preprocessing_hash(),
    }
    configuration["configuration_hash"] = sscd_configuration_hash(configuration)
    atomic_write_json(paths.config_json, configuration)
    for position, (index, record_id, source_row, *_rest) in enumerate(rows):
        offset = 0.1 * position
        scores = torch.tensor([0.1 + offset, 0.2 + offset], dtype=torch.float32)
        score_hash = atomic_torch_save(scores, paths.score_path(index))
        atomic_write_json(
            paths.marker_path(index),
            {
                "schema_version": SSCD_SCHEMA_VERSION,
                "original_index": index,
                "record_id": record_id,
                "source_row_number": source_row,
                "num_seeds": 2,
                "seeds": seed_values,
                "sscd_configuration_hash": configuration["configuration_hash"],
                "generation_record_sha256": file_sha256(
                    generation.record_path(index)
                ),
                "generation_latent_sha256": file_sha256(
                    generation.latent_path(index)
                ),
                "score_sha256": score_hash,
            },
        )

    runtime_calls: list[object] = []

    def forbidden_runtime(*arguments: object, **keywords: object) -> object:
        runtime_calls.append((arguments, keywords))
        raise RuntimeError("real SSCD or VAE loading is forbidden")

    monkeypatch.setattr(sscd_module, "_load_runtime", forbidden_runtime)
    result = sscd_module.run_sscd(
        tmp_path,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=1,
        num_seeds=2,
        seed_start=seed_start,
    )

    assert result.exit_code == 0
    assert result.completed_prompt_count == 2
    assert not runtime_calls
    manifest = pd.read_parquet(paths.manifest_parquet)
    assert manifest["webster_overfit_type"].tolist() == ["TV", "N"]
    assert manifest["completed_or_resumed"].eq("resumed").all()
    summary = read_json(paths.summary_json)
    assert summary["generation_record_count"] == 2
    assert summary["selected_prompt_count"] == 2
    assert summary["resumed_prompt_count"] == 2
    assert summary["total_sscd_scores"] == 4
    assert summary["selection_policy"] == SSCD_SELECTION_POLICY
    assert read_json(paths.config_json)["seeds"] == seed_values
    progress_output = capsys.readouterr().err
    assert "[SSCD] Scoring prompts" in progress_output
    assert "2/2" in progress_output


def test_sscd_experiment_has_no_unet_tokenizer_or_sampler_dependency() -> None:
    path = Path(__file__).resolve().parents[1] / "utils/experiments/sscd.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    identifiers = {
        node.id.casefold() for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "unet" not in identifiers
    assert "tokenizer" not in identifiers
    assert not any(module.endswith("models.sampling") for module in imports)
