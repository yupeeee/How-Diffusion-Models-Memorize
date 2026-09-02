"""Offline tests for the consolidated Webster data surface."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from utils.common.io import CacheIOError, atomic_copy
from utils.data.images import normalized_png_bytes, validate_image_bytes
from utils.data.webster import (
    DownloadError,
    EXPECTED_ROWS_PER_MODEL,
    OriginalIndexError,
    WebsterDataset,
    WebsterImageError,
    WebsterManifestError,
    WebsterPaths,
    _resolve_huggingface_cache_file,
    canonicalize_original_index,
    normalize_webster_type,
)


def _png_bytes(color: tuple[int, int, int] = (20, 80, 140)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="PNG")
    return output.getvalue()


def _write_synthetic_model_view(root: Path) -> tuple[WebsterPaths, str]:
    paths = WebsterPaths.from_root(root)
    paths.create()
    model_directory = paths.model_directory("sdv1")
    image_directory = paths.model_images("sdv1")
    image_directory.mkdir(parents=True, exist_ok=True)
    image_bytes = _png_bytes()
    image_path = image_directory / "1000.png"
    image_path.write_bytes(image_bytes)
    digest = hashlib.sha256(image_bytes).hexdigest()

    rows: list[dict[str, object]] = []
    for source_row in range(EXPECTED_ROWS_PER_MODEL):
        original = 1000 + source_row
        available = source_row == 0
        rows.append(
            {
                "record_id": f"sdv1-{source_row:04d}",
                "model_name": "sdv1",
                "source_row_number": source_row,
                "laion_or_sample_index": original,
                "original_index_raw": original,
                "original_index": str(original),
                "image_filename": f"{original}.png" if available else None,
                "prompt_raw": "" if available else f"prompt {source_row}",
                "overfit_type": "TV" if available else "N",
                "recovery_status": (
                    "recovered_exact_url" if available else "unresolved"
                ),
                "recovery_method": "direct" if available else None,
                "image_available": available,
                "image_path": "sdv1/images/1000.png" if available else None,
                "sha256": digest if available else None,
                "target_image_sha256": digest if available else None,
                "original_url": f"https://example.invalid/{original}.png",
                "auxiliary_asset_paths": "[]",
            }
        )
    model_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(paths.model_metadata_parquet("sdv1"), index=False)
    return paths, digest


def test_original_indices_are_stable_filename_components() -> None:
    assert canonicalize_original_index(42) == "42"
    assert canonicalize_original_index(42.0) == "42"
    assert canonicalize_original_index(" source/index ") == "source%2Findex"
    with pytest.raises(OriginalIndexError):
        canonicalize_original_index(None)
    with pytest.raises(OriginalIndexError):
        canonicalize_original_index(float("nan"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("template-verbatim", "TV"), (" memorized ", "MV"), ("normal", "N"), (None, "UNKNOWN")],
)
def test_webster_labels_are_normalized_without_selection(
    raw: object, expected: str
) -> None:
    assert normalize_webster_type(raw) == expected


def test_image_validation_and_normalization_are_deterministic() -> None:
    source = _png_bytes()
    metadata = validate_image_bytes(source)
    first = normalized_png_bytes(source)
    second = normalized_png_bytes(source)

    assert (metadata.width, metadata.height, metadata.image_format) == (8, 6, "PNG")
    assert metadata.sha256 == hashlib.sha256(source).hexdigest()
    assert first == second
    normalized = validate_image_bytes(first)
    assert (normalized.width, normalized.height) == (8, 6)


def test_huggingface_snapshot_symlink_resolves_only_inside_cache(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "huggingface_cache"
    blob = cache / "datasets--owner--dataset/blobs/content-hash"
    snapshot = (
        cache
        / "datasets--owner--dataset/snapshots/revision/groundtruth_parquets/data.parquet"
    )
    blob.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    blob.write_bytes(b"parquet bytes")
    snapshot.symlink_to(os.path.relpath(blob, snapshot.parent))

    assert _resolve_huggingface_cache_file(snapshot, cache) == blob.resolve()

    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"not in cache")
    escaping = snapshot.with_name("escaping.parquet")
    escaping.symlink_to(os.path.relpath(outside, escaping.parent))
    with pytest.raises(DownloadError, match="unsafe cached file"):
        _resolve_huggingface_cache_file(escaping, cache)

    broken = snapshot.with_name("broken.parquet")
    broken.symlink_to("missing.parquet")
    with pytest.raises(DownloadError, match="unsafe cached file"):
        _resolve_huggingface_cache_file(broken, cache)

    outside_entry = tmp_path / "outside-snapshot.parquet"
    outside_entry.symlink_to(os.path.relpath(blob, outside_entry.parent))
    with pytest.raises(DownloadError, match="unsafe cached file"):
        _resolve_huggingface_cache_file(outside_entry, cache)

    directory_entry = snapshot.with_name("directory.parquet")
    directory_entry.symlink_to(os.path.relpath(blob.parent, directory_entry.parent))
    with pytest.raises(DownloadError, match="not a regular file"):
        _resolve_huggingface_cache_file(directory_entry, cache)


def test_atomic_copy_remains_strict_after_huggingface_resolution(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"parquet bytes")
    destination = tmp_path / "published.parquet"

    atomic_copy(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert not destination.is_symlink()

    source_link = tmp_path / "source-link.parquet"
    source_link.symlink_to(source.name)
    with pytest.raises(CacheIOError, match="copy source is not a regular file"):
        atomic_copy(source_link, tmp_path / "must-not-exist.parquet")


def test_dataset_exposes_every_local_pair_including_tv_and_empty_prompt(
    tmp_path: Path,
) -> None:
    paths, digest = _write_synthetic_model_view(tmp_path)
    recovered = WebsterDataset(tmp_path, "sdv1")
    complete_view = WebsterDataset(tmp_path, "sdv1", recovered_only=False)

    assert recovered.metadata_path == paths.model_metadata_parquet("sdv1")
    assert recovered.total_manifest_rows == EXPECTED_ROWS_PER_MODEL
    assert len(recovered) == 1
    assert len(complete_view) == EXPECTED_ROWS_PER_MODEL
    item = recovered[0]
    assert item["prompt"] == ""
    assert item["webster_overfit_type_normalized"] == "TV"
    assert item["target_image_sha256"] == digest
    assert item["image"].mode == "RGB"
    assert complete_view[-1]["image"] is None


def test_deferred_image_validation_reports_bad_target_when_item_is_loaded(
    tmp_path: Path,
) -> None:
    paths, _ = _write_synthetic_model_view(tmp_path)
    paths.model_images("sdv1").joinpath("1000.png").write_bytes(
        _png_bytes((220, 10, 30))
    )

    with pytest.raises(WebsterManifestError, match="target image SHA-256 differs"):
        WebsterDataset(tmp_path, "sdv1")

    deferred = WebsterDataset(tmp_path, "sdv1", defer_image_validation=True)
    assert len(list(deferred.iter_metadata())) == 1
    with pytest.raises(WebsterImageError, match="target image SHA-256 differs"):
        deferred[0]
