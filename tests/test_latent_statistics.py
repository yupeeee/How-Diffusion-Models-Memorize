"""Offline contracts for streaming target-latent standardization reports."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from utils.common.io import (
    atomic_torch_save,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.experiments.cache import GenerationPaths, publish_completion_marker
from utils.experiments.latent_statistics import (
    TargetLatentStatisticsError,
    compute_target_latent_statistics,
    format_target_latent_statistics,
)
from utils.models.latent import TARGET_LATENT_DEFINITION


LATENT_SHAPE = (4, 2, 2)


def _create_generation_run(
    root: Path,
    *,
    latent_shape: tuple[int, int, int] = LATENT_SHAPE,
) -> tuple[GenerationPaths, str]:
    paths = GenerationPaths(root / "logs" / "synthetic")
    paths.create()
    science = {
        "latent_shape": list(latent_shape),
        "target_latent_definition": TARGET_LATENT_DEFINITION,
    }
    science_hash = canonical_hash(science)
    atomic_write_json(
        paths.run_config,
        {
            "scientific_config": science,
            "scientific_config_hash": science_hash,
        },
    )
    return paths, science_hash


def _publish_target(
    paths: GenerationPaths,
    science_hash: str,
    *,
    index: str,
    source_row: int,
    image_hash: str,
    latent: torch.Tensor,
    marker_file_hash: str | None = None,
    unsafe_save: bool = False,
) -> str:
    target_path = paths.target_latent_path(index)
    if unsafe_save:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent, target_path)
        actual_hash = file_sha256(target_path)
    else:
        actual_hash = atomic_torch_save(latent, target_path)
    tensor_dtype = str(latent.dtype).removeprefix("torch.")
    publish_completion_marker(
        paths,
        index,
        {
            "record_id": f"sdv1-{source_row:04d}",
            "source_row_number": source_row,
            "scientific_config_hash": science_hash,
            "target_image_sha256": image_hash,
            "target_latent_definition": TARGET_LATENT_DEFINITION,
            "latent_shape": list(latent.shape),
            "target_latent_path": f"target_latent/{index}.pt",
            "tensor_file_sha256": {
                "target_latent": marker_file_hash or actual_hash,
            },
            "tensor_shapes": {"target_latent": list(latent.shape)},
            "tensor_dtypes": {"target_latent": tensor_dtype},
        },
    )
    return actual_hash


def _statistic(report: Mapping[str, object], name: str) -> Mapping[str, Any]:
    value = report[name]
    assert isinstance(value, Mapping)
    return value


def _two_known_latents() -> tuple[torch.Tensor, torch.Tensor]:
    first = (
        torch.arange(4, dtype=torch.float32)
        .reshape(4, 1, 1)
        .expand(LATENT_SHAPE)
        .contiguous()
    )
    return first, first + 2.0


def test_exact_population_formulas_and_coordinate_artifacts(tmp_path: Path) -> None:
    paths, science_hash = _create_generation_run(tmp_path)
    first, second = _two_known_latents()
    _publish_target(
        paths,
        science_hash,
        index="10",
        source_row=0,
        image_hash="1" * 64,
        latent=first,
    )
    _publish_target(
        paths,
        science_hash,
        index="11",
        source_row=1,
        image_hash="2" * 64,
        latent=second,
    )
    source_hashes = {
        path: file_sha256(path)
        for path in (paths.target_latent_path("10"), paths.target_latent_path("11"))
    }

    result = compute_target_latent_statistics(tmp_path, paths.run_directory)

    report = result.report
    assert report["schema_version"] == 1
    assert report["latent_shape"] == [4, 2, 2]
    assert report["dimensions_per_latent"] == 16
    assert report["accumulation_dtype"] == "float64"
    paired = _statistic(report, "paired_record_weighted")
    assert paired["record_count"] == 2
    assert paired["element_mean"] == pytest.approx(2.5)
    assert paired["element_population_std"] == pytest.approx(1.5)
    assert paired["mean_squared_l2_per_dimension"] == pytest.approx(8.5)
    assert paired["root_mean_squared_l2_per_dimension"] == pytest.approx(8.5**0.5)
    assert paired["mean_vector_rms"] == pytest.approx(7.5**0.5)
    assert paired["mean_vector_max_abs"] == pytest.approx(4.0)
    assert paired["mean_variance_per_dimension"] == pytest.approx(1.0)
    assert paired["root_mean_variance_per_dimension"] == pytest.approx(1.0)

    assert paired["per_channel_element_mean"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert paired["per_channel_element_population_std"] == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )
    assert paired["per_channel_mean_squared_l2_per_dimension"] == pytest.approx(
        [2.0, 5.0, 10.0, 17.0]
    )

    # This report treats the finite recovered rows as its empirical population.
    # A correction=1 sample standard deviation would be sqrt(72 / 31), not 1.5.
    flattened = torch.stack((first, second)).flatten().double()
    assert paired["element_population_std"] == pytest.approx(
        flattened.std(correction=0).item()
    )
    assert paired["element_population_std"] != pytest.approx(
        flattened.std(correction=1).item()
    )

    artifact_directory = paths.run_directory / "target_latent_statistics"
    mean_path = artifact_directory / "mean.pt"
    population_std_path = artifact_directory / "population_std.pt"
    assert result.report_path == artifact_directory / "report.json"
    assert read_json(result.report_path) == report
    mean = safe_torch_load(mean_path)
    population_std = safe_torch_load(population_std_path)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(population_std, torch.Tensor)
    assert mean.dtype == torch.float64
    assert population_std.dtype == torch.float64
    torch.testing.assert_close(mean, (first.double() + second.double()) / 2)
    torch.testing.assert_close(
        population_std, torch.ones(LATENT_SHAPE, dtype=torch.float64)
    )

    artifacts = report["coordinate_artifacts"]
    assert isinstance(artifacts, Mapping)
    for name, path in (("mean", mean_path), ("population_std", population_std_path)):
        metadata = artifacts[name]
        assert isinstance(metadata, Mapping)
        assert Path(str(metadata["path"])).name == path.name
        assert metadata["sha256"] == file_sha256(path)
        assert metadata["shape"] == [4, 2, 2]
        assert metadata["dtype"] == "float64"

    assert source_hashes == {path: file_sha256(path) for path in source_hashes}


def test_pair_and_unique_image_weighting_are_both_explicit(tmp_path: Path) -> None:
    paths, science_hash = _create_generation_run(tmp_path)
    zero = torch.zeros(LATENT_SHAPE, dtype=torch.float32)
    three = torch.full(LATENT_SHAPE, 3.0, dtype=torch.float32)
    for index, row in (("20", 0), ("21", 1)):
        _publish_target(
            paths,
            science_hash,
            index=index,
            source_row=row,
            image_hash="a" * 64,
            latent=zero,
        )
    _publish_target(
        paths,
        science_hash,
        index="22",
        source_row=2,
        image_hash="b" * 64,
        latent=three,
    )

    report = compute_target_latent_statistics(tmp_path, paths.run_directory).report
    paired = _statistic(report, "paired_record_weighted")
    unique = _statistic(report, "unique_target_image_weighted")

    assert paired["record_count"] == 3
    assert paired["element_mean"] == pytest.approx(1.0)
    assert paired["element_population_std"] == pytest.approx(2.0**0.5)
    assert paired["mean_squared_l2_per_dimension"] == pytest.approx(3.0)
    assert unique["record_count"] == 2
    assert unique["element_mean"] == pytest.approx(1.5)
    assert unique["element_population_std"] == pytest.approx(1.5)
    assert unique["mean_squared_l2_per_dimension"] == pytest.approx(4.5)


def test_duplicate_image_with_different_target_latent_is_rejected(
    tmp_path: Path,
) -> None:
    paths, science_hash = _create_generation_run(tmp_path)
    _publish_target(
        paths,
        science_hash,
        index="30",
        source_row=0,
        image_hash="c" * 64,
        latent=torch.zeros(LATENT_SHAPE, dtype=torch.float32),
    )
    _publish_target(
        paths,
        science_hash,
        index="31",
        source_row=1,
        image_hash="c" * 64,
        latent=torch.ones(LATENT_SHAPE, dtype=torch.float32),
    )

    with pytest.raises(
        TargetLatentStatisticsError,
        match="(?i)(duplicate|same target image)",
    ):
        compute_target_latent_statistics(tmp_path, paths.run_directory)


def test_target_latent_file_hash_is_verified(tmp_path: Path) -> None:
    paths, science_hash = _create_generation_run(tmp_path)
    _publish_target(
        paths,
        science_hash,
        index="40",
        source_row=0,
        image_hash="d" * 64,
        latent=torch.zeros(LATENT_SHAPE, dtype=torch.float32),
        marker_file_hash="f" * 64,
    )

    with pytest.raises(TargetLatentStatisticsError, match="(?i)(sha-256|hash)"):
        compute_target_latent_statistics(tmp_path, paths.run_directory)


@pytest.mark.parametrize(
    ("latent", "message"),
    [
        (torch.zeros(LATENT_SHAPE, dtype=torch.float64), "dtype|float32"),
        (torch.zeros((4, 2, 1), dtype=torch.float32), "shape"),
        (
            torch.full(LATENT_SHAPE, float("nan"), dtype=torch.float32),
            "finite",
        ),
        (
            torch.full(LATENT_SHAPE, float("inf"), dtype=torch.float32),
            "finite",
        ),
    ],
    ids=("wrong-dtype", "wrong-shape", "nan", "infinity"),
)
def test_invalid_target_latent_payload_is_rejected(
    tmp_path: Path,
    latent: torch.Tensor,
    message: str,
) -> None:
    paths, science_hash = _create_generation_run(tmp_path)
    _publish_target(
        paths,
        science_hash,
        index="50",
        source_row=0,
        image_hash="e" * 64,
        latent=latent,
        unsafe_save=not bool(torch.isfinite(latent).all()),
    )

    with pytest.raises(TargetLatentStatisticsError, match=f"(?i)({message})"):
        compute_target_latent_statistics(tmp_path, paths.run_directory)


def test_symlinked_target_latent_is_rejected(tmp_path: Path) -> None:
    paths, science_hash = _create_generation_run(tmp_path)
    latent = torch.zeros(LATENT_SHAPE, dtype=torch.float32)
    backing_path = paths.run_directory / "backing_target.pt"
    backing_hash = atomic_torch_save(latent, backing_path)
    target_path = paths.target_latent_path("60")
    target_path.symlink_to(backing_path)
    publish_completion_marker(
        paths,
        "60",
        {
            "record_id": "sdv1-0000",
            "source_row_number": 0,
            "scientific_config_hash": science_hash,
            "target_image_sha256": "6" * 64,
            "target_latent_definition": TARGET_LATENT_DEFINITION,
            "latent_shape": list(LATENT_SHAPE),
            "target_latent_path": "target_latent/60.pt",
            "tensor_file_sha256": {"target_latent": backing_hash},
            "tensor_shapes": {"target_latent": list(LATENT_SHAPE)},
            "tensor_dtypes": {"target_latent": "float32"},
        },
    )

    with pytest.raises(
        TargetLatentStatisticsError,
        match="(?i)(symlink|unsafe|regular)",
    ):
        compute_target_latent_statistics(tmp_path, paths.run_directory)


def test_formatter_reports_population_contract_concisely(tmp_path: Path) -> None:
    paths, science_hash = _create_generation_run(tmp_path)
    first, second = _two_known_latents()
    _publish_target(
        paths,
        science_hash,
        index="70",
        source_row=0,
        image_hash="7" * 64,
        latent=first,
    )
    _publish_target(
        paths,
        science_hash,
        index="71",
        source_row=1,
        image_hash="8" * 64,
        latent=second,
    )
    report = compute_target_latent_statistics(tmp_path, paths.run_directory).report

    formatted = format_target_latent_statistics(report)

    assert formatted == format_target_latent_statistics(report)
    assert "Target latent population:" in formatted
    assert "2/? recovered/completed rows" in formatted
    assert "2 unique images" in formatted
    assert "shape=(4, 2, 2)" in formatted
    assert "population correction=0" in formatted
    assert "Target latent moments: mean=" in formatted
    assert "std=" in formatted
    assert "E[||x_0||^2]/d=" in formatted
    assert "Target latent mean vector: RMS=" in formatted
    assert "Target latent channels: mean=" in formatted
