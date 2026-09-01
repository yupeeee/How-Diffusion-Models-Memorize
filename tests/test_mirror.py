"""Offline provenance and integrity tests for the Webster ground-truth mirror."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image

from utils.data import images as images_module
from utils.data import mirror as mirror_module
from utils.data.state import RecoveryState
from utils.data.webster import (
    ImageMetadata,
    RecoveryStatus,
    RetryableStageError,
    StageOutcome,
    WebsterPaths,
)


Identity = tuple[str, str, str]


def _manifest(identities: Sequence[Identity]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": f"sdv1-{position:04d}",
                "model_name": "sdv1",
                "source_row_number": position,
                "prompt_raw": prompt,
                "overfit_type": "TV",
                "laion_or_sample_index": index,
                "original_url": url,
                "normalized_url": url,
                "source_page_url": None,
            }
            for position, (index, prompt, url) in enumerate(identities)
        ]
    )


def _png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _recover(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    data: bytes,
    status: RecoveryStatus = RecoveryStatus.RECOVERED_EXACT_URL,
) -> None:
    metadata, raw_path, normalized_path = images_module.store_image(
        paths,
        state,
        data,
        original_url=str(record["original_url"]),
        final_url=str(record["original_url"]),
    )
    images_module.apply_image_resolution(
        paths,
        state,
        record,
        metadata,
        raw_path,
        normalized_path,
        status=status,
        method="synthetic",
        resolved_url=str(record["original_url"]),
    )


def _patch_identity_pin(
    monkeypatch: pytest.MonkeyPatch,
    mirror_identities: Sequence[Identity],
    source_identities: Sequence[Identity],
) -> tuple[mirror_module.MirrorShard, dict[str, pa.Table]]:
    shard = mirror_module.MirrorShard("fixture.parquet", 1, "a" * 64)
    intersection = set(mirror_identities) & set(source_identities)
    monkeypatch.setattr(mirror_module, "MIRROR_SHARDS", (shard,))
    monkeypatch.setattr(
        mirror_module, "MIRROR_ROWS_PER_SHARD", len(mirror_identities)
    )
    monkeypatch.setattr(
        mirror_module, "MIRROR_EXPECTED_ROWS", len(mirror_identities)
    )
    monkeypatch.setattr(
        mirror_module, "MIRROR_EXPECTED_INTERSECTION", len(intersection)
    )
    monkeypatch.setattr(
        mirror_module,
        "MIRROR_IDENTITY_SHA256",
        mirror_module.identity_sha256(mirror_identities),
    )
    monkeypatch.setattr(
        mirror_module,
        "SDV1_IDENTITY_SHA256",
        mirror_module.identity_sha256(source_identities),
    )
    monkeypatch.setattr(
        mirror_module,
        "MIRROR_INTERSECTION_SHA256",
        mirror_module.identity_sha256(intersection),
    )
    table = pa.Table.from_pylist(
        [
            {"index": index, "raw_prompt": prompt, "url": url}
            for index, prompt, url in mirror_identities
        ]
    )
    return shard, {shard.path: table}


def test_identity_join_accepts_only_the_exact_pinned_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = ("1", "shared prompt", "https://example.test/shared")
    mirror_only = ("2", "mirror prompt", "https://example.test/mirror")
    source_only = ("3", "source prompt", "https://example.test/source")
    shard, tables = _patch_identity_pin(
        monkeypatch, (shared, mirror_only), (shared, source_only)
    )

    selected, mirror_hash, source_hash, intersection_hash = (
        mirror_module.validate_mirror_identity(
            tables, _manifest((shared, source_only))
        )
    )

    assert selected == {shared: (shard.path, 0, 0)}
    assert mirror_hash == mirror_module.identity_sha256((shared, mirror_only))
    assert source_hash == mirror_module.identity_sha256((shared, source_only))
    assert intersection_hash == mirror_module.identity_sha256((shared,))


@pytest.mark.parametrize(
    ("source_only", "signal"),
    [
        (("2", "source prompt", "https://example.test/source"), "index"),
        (("3", "mirror prompt", "https://example.test/source"), "prompt"),
        (("3", "source prompt", "https://example.test/mirror"), "URL"),
    ],
)
def test_identity_join_rejects_partial_signal_collisions(
    monkeypatch: pytest.MonkeyPatch,
    source_only: Identity,
    signal: str,
) -> None:
    shared = ("1", "shared prompt", "https://example.test/shared")
    mirror_only = ("2", "mirror prompt", "https://example.test/mirror")
    _, tables = _patch_identity_pin(
        monkeypatch, (shared, mirror_only), (shared, source_only)
    )

    with pytest.raises(
        mirror_module.MirrorIntegrityError,
        match=rf"partial {signal} collision",
    ):
        mirror_module.validate_mirror_identity(
            tables, _manifest((shared, source_only))
        )


def test_ground_truth_projection_never_requests_generated_columns_and_cannot_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = ("7", "ground-truth prompt", "https://example.test/7")
    item: dict[str, object] = {
        "index": identity[0],
        "raw_prompt": identity[1],
        "ground_truth": {"bytes": _png((1, 2, 3)), "path": "ignored.jpg"},
        "url": identity[2],
        "gen_images_sd1_4": {"bytes": _png((200, 100, 50))},
    }

    class Column:
        path_in_schema = "ground_truth.bytes"
        total_compressed_size = 123

    class RowGroup:
        num_columns = 1

        @staticmethod
        def column(unused_number: int) -> Column:
            return Column()

    class Metadata:
        num_row_groups = 1

        @staticmethod
        def row_group(unused_group: int) -> RowGroup:
            return RowGroup()

    class Parquet:
        metadata = Metadata()

        def __init__(self) -> None:
            self.requests: list[tuple[str, ...]] = []

        def read_row_group(
            self, unused_group: int, *, columns: list[str]
        ) -> pa.Table:
            self.requests.append(tuple(columns))
            return pa.Table.from_pylist(
                [{column: item[column] for column in columns}]
            )

    class Stream:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Progress:
        def __init__(self) -> None:
            self.updates: list[int] = []

        def update(self, amount: int) -> None:
            self.updates.append(amount)

    parquet = Parquet()
    stream = Stream()
    progress = Progress()
    monkeypatch.setattr(
        mirror_module,
        "_open_parquet",
        lambda unused_filesystem, unused_shard: (stream, parquet),
    )

    table, projected_bytes = mirror_module._load_remote_ground_truth(
        object(),
        mirror_module.MirrorShard("fixture.parquet", 1, "b" * 64),
        {identity},
        progress,  # type: ignore[arg-type]
    )

    assert parquet.requests == [
        mirror_module.IDENTITY_COLUMNS,
        mirror_module.MIRROR_COLUMNS,
    ]
    assert "gen_images_sd1_4" not in table.column_names
    assert table.column_names == list(mirror_module.MIRROR_COLUMNS)
    assert projected_bytes == 123
    assert progress.updates == [123]
    assert stream.closed

    with pytest.raises(
        mirror_module.MirrorIntegrityError, match="ground_truth is null"
    ):
        mirror_module._mirror_row_from_item(
            {**item, "ground_truth": None}, ("fixture.parquet", 0, 0)
        )


def test_loader_reuses_valid_projected_cache_without_remote_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = ("1", "shared prompt", "https://example.test/shared")
    mirror_only = ("2", "mirror prompt", "https://example.test/mirror")
    source_only = ("3", "source prompt", "https://example.test/source")
    shard, identity_tables = _patch_identity_pin(
        monkeypatch, (shared, mirror_only), (shared, source_only)
    )
    paths = WebsterPaths.from_root(tmp_path)
    manifest = _manifest((shared, source_only))

    class ForbiddenRemote:
        def dataset_info(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("valid projected cache contacted the Hub API")

        def open(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("valid projected cache opened a remote shard")

    ground_truth = pa.Table.from_pylist(
        [
            {
                "index": shared[0],
                "raw_prompt": shared[1],
                "ground_truth": {"bytes": _png((1, 2, 3)), "path": "ignored"},
                "url": shared[2],
            }
        ]
    ).select(list(mirror_module.MIRROR_COLUMNS))
    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        mirror_module._write_cached_table(
            paths,
            shard,
            "identities",
            identity_tables[shard.path],
            mirror_module.IDENTITY_COLUMNS,
        )
        mirror_module._write_cached_table(
            paths,
            shard,
            "ground_truth",
            ground_truth,
            mirror_module.MIRROR_COLUMNS,
            source_projected_bytes=123,
        )

        catalog = mirror_module.load_pinned_mirror_rows(
            paths,
            state,
            manifest,
            api=ForbiddenRemote(),
            filesystem=ForbiddenRemote(),
        )

        assert [row.identity for row in catalog.rows] == [shared]
        assert catalog.rows[0].image_bytes == _png((1, 2, 3))
        cached_path, _ = mirror_module._cache_paths(
            paths, shard, "ground_truth"
        )
        cached_path.write_bytes(b"corrupt")
        assert mirror_module._read_cached_table(
            paths,
            shard,
            "ground_truth",
            mirror_module.MIRROR_COLUMNS,
            None,
        ) is None


def test_phash_distance_is_exact_and_rejects_invalid_inputs() -> None:
    assert mirror_module._phash_distance("00", "03") == 2
    assert mirror_module._phash_distance("ffff", "fffe") == 1
    with pytest.raises(mirror_module.MirrorIntegrityError, match="incompatible"):
        mirror_module._phash_distance("0", "00")
    with pytest.raises(mirror_module.MirrorIntegrityError, match="invalid"):
        mirror_module._phash_distance("zz", "00")


def _metadata(data: bytes, perceptual_hash: str) -> ImageMetadata:
    return ImageMetadata(
        hashlib.sha256(data).hexdigest(),
        hashlib.sha1(data).hexdigest(),
        perceptual_hash,
        8,
        8,
        "RGB",
        "PNG",
        "png",
        len(data),
    )


def _threshold_audit(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    minimum_agreement: float,
) -> mirror_module.MirrorAudit:
    identities = (
        ("10", "first", "https://example.test/10"),
        ("11", "second", "https://example.test/11"),
    )
    paths = WebsterPaths.from_root(root)
    paths.create()
    donor_bytes = (_png((10, 20, 30)), _png((40, 50, 60)))
    mirror_bytes = (b"synthetic-mirror-one", b"synthetic-mirror-two")
    hashes = {
        mirror_bytes[0]: _metadata(mirror_bytes[0], "0000000000000000"),
        donor_bytes[0]: _metadata(donor_bytes[0], "0000000000000001"),
        mirror_bytes[1]: _metadata(mirror_bytes[1], "0000000000000000"),
        donor_bytes[1]: _metadata(donor_bytes[1], "0000000000000003"),
    }

    def fake_validate(data: bytes, **unused_kwargs: object) -> ImageMetadata:
        return hashes[data]

    monkeypatch.setattr(mirror_module, "validate_image_bytes", fake_validate)
    monkeypatch.setattr(mirror_module, "MIRROR_EXPECTED_INTERSECTION", 2)
    monkeypatch.setattr(mirror_module, "MIN_INDEPENDENT_AUDIT", 2)
    monkeypatch.setattr(mirror_module, "MAX_PHASH_DISTANCE", 1)
    monkeypatch.setattr(
        mirror_module, "MIN_AGREEMENT_FRACTION", minimum_agreement
    )
    rows = tuple(
        mirror_module.MirrorRow(
            position,
            "fixture.parquet",
            position,
            *identity,
            mirror_bytes[position],
        )
        for position, identity in enumerate(identities)
    )
    catalog = mirror_module.MirrorCatalog(rows, "m", "s", "i", {"f": "h"})

    with RecoveryState(paths) as state:
        state.initialize_records(_manifest(identities))
        for position, record in enumerate(state.records()):
            _recover(paths, state, record, donor_bytes[position])
        return mirror_module.audit_mirror(paths, state, catalog)


def test_audit_applies_perceptual_distance_and_agreement_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _threshold_audit(
        tmp_path / "passing", monkeypatch, minimum_agreement=0.50
    )
    assert (audit.evidence_count, audit.agreement_count) == (2, 1)
    assert audit.agreement_fraction == 0.5
    assert audit.mismatch_count == 1

    with pytest.raises(
        mirror_module.MirrorIntegrityError, match=r"agreement is 1/2"
    ):
        _threshold_audit(
            tmp_path / "failing", monkeypatch, minimum_agreement=0.75
        )


def test_audit_excludes_mirror_and_verified_duplicate_circular_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = (
        ("20", "mirror donor", "https://example.test/20"),
        ("21", "duplicate donor", "https://example.test/21"),
    )
    images = (_png((1, 2, 3)), _png((4, 5, 6)))
    paths = WebsterPaths.from_root(tmp_path)
    paths.create()
    monkeypatch.setattr(mirror_module, "MIRROR_EXPECTED_INTERSECTION", 2)
    monkeypatch.setattr(mirror_module, "MIN_INDEPENDENT_AUDIT", 1)
    rows = tuple(
        mirror_module.MirrorRow(
            position,
            "fixture.parquet",
            position,
            *identity,
            images[position],
        )
        for position, identity in enumerate(identities)
    )
    catalog = mirror_module.MirrorCatalog(rows, "m", "s", "i", {"f": "h"})

    with RecoveryState(paths) as state:
        state.initialize_records(_manifest(identities))
        records = state.records()
        _recover(
            paths,
            state,
            records[0],
            images[0],
            RecoveryStatus.RECOVERED_GROUND_TRUTH_MIRROR,
        )
        _recover(
            paths,
            state,
            records[1],
            images[1],
            RecoveryStatus.RECOVERED_VERIFIED_DUPLICATE,
        )
        with pytest.raises(
            RetryableStageError, match=r"only 0/1 independent targets"
        ):
            mirror_module.audit_mirror(paths, state, catalog)


def test_recover_mirror_row_records_audited_ground_truth_provenance(
    tmp_path: Path,
) -> None:
    identity = ("30", "recovered prompt", "https://example.test/30")
    paths = WebsterPaths.from_root(tmp_path)
    paths.create()
    row = mirror_module.MirrorRow(
        0, "fixture.parquet", 4, *identity, _png((11, 22, 33))
    )

    with RecoveryState(paths) as state:
        state.initialize_records(_manifest((identity,)))
        record = state.get_record("sdv1-0000")
        assert record is not None

        result = mirror_module.recover_mirror_row(paths, state, record, row)

        recovered = state.get_record("sdv1-0000")
        attempts = state.attempts("sdv1-0000")
        assert recovered is not None
        assert result.outcome is StageOutcome.RECOVERED
        assert (
            recovered["recovery_status"]
            == RecoveryStatus.RECOVERED_GROUND_TRUTH_MIRROR.value
        )
        assert recovered["recovery_method"] == "hf_ground_truth_mirror"
        assert recovered["resolved_url"] == row.locator
        assert len(attempts) == 1
        assert attempts[0]["strategy"] == "hf_ground_truth_mirror"
        assert attempts[0]["attempted_url"] == row.locator
        assert attempts[0]["validation_result"] == "valid_audited_ground_truth"
        assert attempts[0]["candidate_sha256"] == recovered["sha256"]
        assert attempts[0]["response_size"] == len(row.image_bytes)
