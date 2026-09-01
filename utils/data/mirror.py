"""Pinned, audited recovery from a ground-truth-only Webster mirror.

The mirror is deliberately treated as an untrusted third-party source.  Its
immutable Hub revision, LFS objects, schema, and complete identity set are
pinned here.  Only rows that exactly intersect the official SD1 metadata are
read from the embedded ``ground_truth.bytes`` Parquet column.  Generated-image
columns are never read; the untrusted ``ground_truth.path`` value is never
followed or used.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from utils.common.io import (
    atomic_write_json,
    canonical_hash,
    file_sha256,
    utc_now,
)

from .images import (
    apply_image_resolution,
    recovery_artifacts_available,
    store_image,
    validate_image_bytes,
    verify_existing_blob,
)
from .webster import (
    RECOVERED_STATUS_VALUES,
    ImageMetadata,
    RecoveryStatus,
    RetryableStageError,
    StageOutcome,
    StageResult,
    StateConsistencyError,
    WebsterPaths,
)

if TYPE_CHECKING:
    from .state import RecoveryState


MIRROR_REPOSITORY = "gdhanuka/memorization_data2"
MIRROR_REVISION = "327530503d2c6c16b87691e0fb57c08973a5fc47"
MIRROR_STRATEGY_VERSION = 1
MIRROR_EXPECTED_ROWS = 500
MIRROR_EXPECTED_INTERSECTION = 352
MIRROR_ROWS_PER_SHARD = 125
MIRROR_ROW_GROUP_ROWS = (100, 25)
MIRROR_IDENTITY_SHA256 = (
    "c810e9a8b4bfa34d90723f049be1bca17db89a59d01206aced1eb79a91c6c127"
)
SDV1_IDENTITY_SHA256 = (
    "aa3662ddf68831832565609689fb8ad39acd021b13c3cc6600610e975a167c75"
)
MIRROR_INTERSECTION_SHA256 = (
    "f5fbaab4b5f788ef2361e792df22cb55a895180c8b580a9db1d73df9154cf80c"
)
MIN_INDEPENDENT_AUDIT = 50
MAX_PHASH_DISTANCE = 4
MIN_AGREEMENT_FRACTION = 0.90
MIRROR_PROJECTED_GROUND_TRUTH_BYTES = 211_290_210
IDENTITY_COLUMNS = ("index", "raw_prompt", "url")
MIRROR_COLUMNS = ("index", "raw_prompt", "ground_truth", "url")


@dataclass(frozen=True, slots=True)
class MirrorShard:
    path: str
    size: int
    lfs_sha256: str


MIRROR_SHARDS = (
    MirrorShard(
        "data/train-00000-of-00004-62fbf8499ec78dd2.parquet",
        394_035_108,
        "dd3859226a011587466eb9c01c15d1d65da9679b79073fdac8243ad1b108364a",
    ),
    MirrorShard(
        "data/train-00001-of-00004-5cb5e8740c5cf8ff.parquet",
        407_816_287,
        "7776658504e4e2335c3046b9f2829ff9e6f14ffe3be081dcb20126dfd2a8e87d",
    ),
    MirrorShard(
        "data/train-00002-of-00004-838c24c28411bba7.parquet",
        389_460_052,
        "483a70fc420222855186319ecb54c22c119b7e14de76e2f7a5ad8d9d10594ff1",
    ),
    MirrorShard(
        "data/train-00003-of-00004-76f6c4852f38573b.parquet",
        368_926_927,
        "c4fbb7d576728d637f26acbab7960ea91ca6d9047a1e84e6534db7d1b3d118e2",
    ),
)

INDEPENDENT_AUDIT_STATUSES = frozenset(
    {
        RecoveryStatus.RECOVERED_EXACT_URL.value,
        RecoveryStatus.RECOVERED_OFFICIAL_CACHE.value,
        RecoveryStatus.RECOVERED_WAYBACK_EXACT.value,
        RecoveryStatus.RECOVERED_ARQUIVO_EXACT.value,
        RecoveryStatus.RECOVERED_COMMONCRAWL_EXACT.value,
    }
)

Identity = tuple[str, str, str]


class MirrorIntegrityError(StateConsistencyError):
    """The pinned mirror violates a provenance or identity invariant."""


@dataclass(frozen=True, slots=True)
class MirrorRow:
    ordinal: int
    shard: str
    row_in_shard: int
    index: str
    raw_prompt: str
    url: str
    image_bytes: bytes

    @property
    def identity(self) -> Identity:
        return (self.index, self.raw_prompt, self.url)

    @property
    def locator(self) -> str:
        return (
            f"hf://datasets/{MIRROR_REPOSITORY}@{MIRROR_REVISION}/"
            f"{self.shard}#row={self.row_in_shard}:ground_truth"
        )


@dataclass(frozen=True, slots=True)
class MirrorCatalog:
    rows: tuple[MirrorRow, ...]
    identity_sha256: str
    source_identity_sha256: str
    intersection_sha256: str
    cache_sha256: Mapping[str, str]

    def by_identity(self) -> dict[Identity, MirrorRow]:
        return {row.identity: row for row in self.rows}


@dataclass(frozen=True, slots=True)
class MirrorAudit:
    evidence_count: int
    agreement_count: int
    agreement_fraction: float
    mismatch_count: int
    evidence_signature: str
    reused_attestation: bool


@dataclass(frozen=True, slots=True)
class _CachedTable:
    table: pa.Table
    metadata: Mapping[str, object]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def identity_from_values(index: object, prompt: object, url: object) -> Identity:
    """Return the exact source identity, normalizing only the scalar index."""

    if _is_missing(index) or not isinstance(prompt, str) or not isinstance(url, str):
        raise MirrorIntegrityError("mirror identity contains a null or non-string field")
    if not prompt or not url:
        raise MirrorIntegrityError("mirror identity contains an empty prompt or URL")
    return (str(index), prompt, url)


def record_identity(record: Mapping[str, object]) -> Identity:
    return identity_from_values(
        record.get("laion_or_sample_index"),
        record.get("prompt_raw"),
        record.get("original_url"),
    )


def identity_sha256(identities: Sequence[Identity] | set[Identity]) -> str:
    """Hash sorted compact-JSON identity triples with a trailing newline."""

    payload = "".join(
        json.dumps(
            list(identity),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for identity in sorted(set(identities))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_paths(paths: WebsterPaths, shard: MirrorShard, kind: str) -> tuple[Path, Path]:
    directory = paths.ground_truth_mirror_cache / MIRROR_REVISION
    stem = Path(shard.path).name.removesuffix(".parquet")
    parquet_path = directory / f"{stem}.{kind}.parquet"
    return parquet_path, parquet_path.with_suffix(".manifest.json")


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _read_cached_table(
    paths: WebsterPaths,
    shard: MirrorShard,
    kind: str,
    columns: Sequence[str],
    expected_rows: int | None,
) -> _CachedTable | None:
    parquet_path, manifest_path = _cache_paths(paths, shard, kind)
    if not _regular_file(parquet_path) or not _regular_file(manifest_path):
        return None
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return None
        if any(
            (
                metadata.get("schema") != "webster_ground_truth_mirror_cache_v1",
                metadata.get("repository") != MIRROR_REPOSITORY,
                metadata.get("revision") != MIRROR_REVISION,
                metadata.get("source_shard") != shard.path,
                metadata.get("source_lfs_sha256") != shard.lfs_sha256,
                metadata.get("kind") != kind,
                metadata.get("columns") != list(columns),
            )
        ):
            return None
        observed_hash = file_sha256(parquet_path)
        if metadata.get("parquet_sha256") != observed_hash:
            return None
        table = pq.read_table(parquet_path, columns=list(columns))
        if table.column_names != list(columns):
            return None
        if expected_rows is not None and table.num_rows != expected_rows:
            return None
        if int(metadata.get("rows", -1)) != table.num_rows:
            return None
        return _CachedTable(table.combine_chunks(), metadata)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, pa.ArrowException):
        return None


def _quarantine_owned_path(paths: WebsterPaths, path: Path) -> None:
    if not path.is_symlink() and not path.exists():
        return
    quarantine = paths.ground_truth_mirror_cache / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    os.replace(path, quarantine / f"{path.name}.{uuid.uuid4().hex}")


def _write_cached_table(
    paths: WebsterPaths,
    shard: MirrorShard,
    kind: str,
    table: pa.Table,
    columns: Sequence[str],
    *,
    source_projected_bytes: int = 0,
) -> _CachedTable:
    parquet_path, manifest_path = _cache_paths(paths, shard, kind)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = parquet_path.with_name(f".{parquet_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _quarantine_owned_path(paths, parquet_path)
        os.replace(temporary, parquet_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata: dict[str, object] = {
        "schema": "webster_ground_truth_mirror_cache_v1",
        "repository": MIRROR_REPOSITORY,
        "revision": MIRROR_REVISION,
        "source_shard": shard.path,
        "source_size": shard.size,
        "source_lfs_sha256": shard.lfs_sha256,
        "kind": kind,
        "columns": list(columns),
        "rows": table.num_rows,
        "source_projected_bytes": source_projected_bytes,
        "parquet_sha256": file_sha256(parquet_path),
        "created_at": utc_now(),
    }
    _quarantine_owned_path(paths, manifest_path)
    atomic_write_json(manifest_path, metadata)
    return _CachedTable(table.combine_chunks(), metadata)


def _member_value(item: object, name: str) -> object:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _new_hub_clients() -> tuple[object, object]:
    try:
        from huggingface_hub import HfApi, HfFileSystem
    except ImportError as error:
        raise MirrorIntegrityError(
            "huggingface_hub is required for ground-truth mirror recovery"
        ) from error
    return HfApi(token=False), HfFileSystem(token=False)


def _validate_remote_attestation(api: object) -> None:
    try:
        info = api.dataset_info(  # type: ignore[attr-defined]
            MIRROR_REPOSITORY,
            revision=MIRROR_REVISION,
            files_metadata=True,
        )
    except Exception as error:
        name = type(error).__name__
        if name in {
            "RevisionNotFoundError",
            "EntryNotFoundError",
            "RemoteEntryNotFoundError",
            "RepositoryNotFoundError",
        }:
            raise MirrorIntegrityError(
                f"pinned ground-truth mirror is unavailable: {name}: {error}"
            ) from error
        raise RetryableStageError(
            f"cannot attest pinned ground-truth mirror: {name}: {error}"
        ) from error
    if str(getattr(info, "sha", "")) != MIRROR_REVISION:
        raise MirrorIntegrityError("Hugging Face returned a different mirror revision")
    siblings = list(getattr(info, "siblings", ()) or ())
    by_path = {
        str(_member_value(sibling, "rfilename")): sibling for sibling in siblings
    }
    expected_paths = {shard.path for shard in MIRROR_SHARDS}
    observed_parquets = {
        path
        for path in by_path
        if path.startswith("data/") and path.endswith(".parquet")
    }
    if observed_parquets != expected_paths:
        raise MirrorIntegrityError("pinned mirror Parquet shard set changed")
    for shard in MIRROR_SHARDS:
        sibling = by_path.get(shard.path)
        lfs = _member_value(sibling, "lfs") if sibling is not None else None
        observed_size = _member_value(lfs, "size") or _member_value(sibling, "size")
        observed_sha = _member_value(lfs, "sha256")
        if int(observed_size or -1) != shard.size or str(observed_sha) != shard.lfs_sha256:
            raise MirrorIntegrityError(
                f"pinned mirror LFS attestation changed for {shard.path}"
            )


def _remote_path(shard: MirrorShard) -> str:
    return f"datasets/{MIRROR_REPOSITORY}@{MIRROR_REVISION}/{shard.path}"


def _validate_parquet_file(parquet: pq.ParquetFile, shard: MirrorShard) -> None:
    metadata = parquet.metadata
    if metadata.num_rows != MIRROR_ROWS_PER_SHARD:
        raise MirrorIntegrityError(f"unexpected row count in {shard.path}")
    if metadata.num_row_groups != len(MIRROR_ROW_GROUP_ROWS):
        raise MirrorIntegrityError(f"unexpected row-group count in {shard.path}")
    if tuple(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)) != MIRROR_ROW_GROUP_ROWS:
        raise MirrorIntegrityError(f"unexpected row-group layout in {shard.path}")
    names = set(parquet.schema_arrow.names)
    if not set(MIRROR_COLUMNS).issubset(names):
        raise MirrorIntegrityError(f"required mirror columns are absent in {shard.path}")
    ground_truth = parquet.schema_arrow.field("ground_truth").type
    if not pa.types.is_struct(ground_truth):
        raise MirrorIntegrityError("ground_truth is not an embedded image struct")
    fields = {field.name: field.type for field in ground_truth}
    if "bytes" not in fields or not (
        pa.types.is_binary(fields["bytes"]) or pa.types.is_large_binary(fields["bytes"])
    ):
        raise MirrorIntegrityError("ground_truth.bytes is not an embedded binary column")


def _open_parquet(filesystem: object, shard: MirrorShard) -> tuple[object, pq.ParquetFile]:
    try:
        stream = filesystem.open(  # type: ignore[attr-defined]
            _remote_path(shard),
            "rb",
            block_size=1 << 20,
            cache_type="none",
        )
        parquet = pq.ParquetFile(stream)
        _validate_parquet_file(parquet, shard)
        return stream, parquet
    except MirrorIntegrityError:
        raise
    except Exception as error:
        raise RetryableStageError(
            f"cannot range-read pinned mirror shard {shard.path}: "
            f"{type(error).__name__}: {error}"
        ) from error


def _load_remote_identities(filesystem: object, shard: MirrorShard) -> pa.Table:
    stream, parquet = _open_parquet(filesystem, shard)
    try:
        tables = [
            parquet.read_row_group(group, columns=list(IDENTITY_COLUMNS))
            for group in range(parquet.metadata.num_row_groups)
        ]
        table = pa.concat_tables(tables).combine_chunks()
    except Exception as error:
        raise RetryableStageError(
            f"cannot read mirror identities from {shard.path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        stream.close()  # type: ignore[attr-defined]
    if table.column_names != list(IDENTITY_COLUMNS) or table.num_rows != MIRROR_ROWS_PER_SHARD:
        raise MirrorIntegrityError(f"invalid identity projection in {shard.path}")
    return table


def validate_mirror_identity(
    identity_tables: Mapping[str, pa.Table],
    source_manifest: pd.DataFrame,
) -> tuple[dict[Identity, tuple[str, int, int]], str, str, str]:
    """Fail closed unless the full pinned catalog and exact SD1 join match."""

    ordered: list[tuple[Identity, str, int, int]] = []
    ordinal = 0
    for shard in MIRROR_SHARDS:
        table = identity_tables.get(shard.path)
        if table is None or table.num_rows != MIRROR_ROWS_PER_SHARD:
            raise MirrorIntegrityError(f"identity cache is incomplete for {shard.path}")
        for row_in_shard, row in enumerate(table.to_pylist()):
            identity = identity_from_values(row["index"], row["raw_prompt"], row["url"])
            ordered.append((identity, shard.path, row_in_shard, ordinal))
            ordinal += 1
    mirror_identities = [item[0] for item in ordered]
    if len(mirror_identities) != MIRROR_EXPECTED_ROWS or len(set(mirror_identities)) != MIRROR_EXPECTED_ROWS:
        raise MirrorIntegrityError("mirror identities are missing or duplicated")
    for position in range(3):
        if len({identity[position] for identity in mirror_identities}) != MIRROR_EXPECTED_ROWS:
            raise MirrorIntegrityError("a mirror identity signal is duplicated")
    mirror_digest = identity_sha256(mirror_identities)
    if mirror_digest != MIRROR_IDENTITY_SHA256:
        raise MirrorIntegrityError("full mirror identity digest does not match the pin")

    sdv1 = source_manifest.loc[source_manifest["model_name"].astype(str) == "sdv1"]
    if len(sdv1) != MIRROR_EXPECTED_ROWS:
        raise MirrorIntegrityError("official SD1 source manifest is not 500 rows")
    source_by_identity: dict[Identity, str] = {}
    for record in sdv1.to_dict(orient="records"):
        identity = record_identity(record)
        if identity in source_by_identity:
            raise MirrorIntegrityError("official SD1 source identities are duplicated")
        source_by_identity[identity] = str(record["record_id"])
    source_digest = identity_sha256(set(source_by_identity))
    if source_digest != SDV1_IDENTITY_SHA256:
        raise MirrorIntegrityError("official SD1 identity digest does not match the pin")

    mirror_set = set(mirror_identities)
    source_set = set(source_by_identity)
    intersection = mirror_set & source_set
    if len(intersection) != MIRROR_EXPECTED_INTERSECTION:
        raise MirrorIntegrityError("mirror does not have the pinned 352-row intersection")
    intersection_digest = identity_sha256(intersection)
    if intersection_digest != MIRROR_INTERSECTION_SHA256:
        raise MirrorIntegrityError("mirror intersection identity digest does not match the pin")
    for position, label in enumerate(("index", "prompt", "URL")):
        mirror_values = {identity[position] for identity in mirror_set}
        source_values = {identity[position] for identity in source_set}
        if len(mirror_values & source_values) != MIRROR_EXPECTED_INTERSECTION:
            raise MirrorIntegrityError(f"mirror contains a partial {label} collision")

    selected = {
        identity: (shard, row_in_shard, ordinal_value)
        for identity, shard, row_in_shard, ordinal_value in ordered
        if identity in intersection
    }
    return selected, mirror_digest, source_digest, intersection_digest


def _ground_truth_compressed_bytes(parquet: pq.ParquetFile, group: int) -> int:
    row_group = parquet.metadata.row_group(group)
    total = 0
    for column_number in range(row_group.num_columns):
        column = row_group.column(column_number)
        if str(column.path_in_schema).split(".", 1)[0] == "ground_truth":
            total += int(column.total_compressed_size)
    return total


def _load_remote_ground_truth(
    filesystem: object,
    shard: MirrorShard,
    accepted: set[Identity],
    progress: tqdm,
) -> tuple[pa.Table, int]:
    stream, parquet = _open_parquet(filesystem, shard)
    selected_tables: list[pa.Table] = []
    projected_bytes = 0
    try:
        for group in range(parquet.metadata.num_row_groups):
            identities = parquet.read_row_group(group, columns=list(IDENTITY_COLUMNS)).to_pylist()
            positions = [
                position
                for position, item in enumerate(identities)
                if identity_from_values(item["index"], item["raw_prompt"], item["url"])
                in accepted
            ]
            if not positions:
                continue
            table = parquet.read_row_group(group, columns=list(MIRROR_COLUMNS))
            size = _ground_truth_compressed_bytes(parquet, group)
            projected_bytes += size
            progress.update(size)
            selected_tables.append(table.take(pa.array(positions, type=pa.int64())))
    except MirrorIntegrityError:
        raise
    except Exception as error:
        raise RetryableStageError(
            f"cannot read ground_truth from {shard.path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        stream.close()  # type: ignore[attr-defined]
    if not selected_tables:
        return pa.table({name: [] for name in MIRROR_COLUMNS}), projected_bytes
    return pa.concat_tables(selected_tables).combine_chunks(), projected_bytes


def _mirror_row_from_item(
    item: Mapping[str, object],
    location: tuple[str, int, int],
) -> MirrorRow:
    identity = identity_from_values(item["index"], item["raw_prompt"], item["url"])
    embedded = item.get("ground_truth")
    if not isinstance(embedded, Mapping):
        raise MirrorIntegrityError(f"ground_truth is null for mirror index {identity[0]}")
    image = embedded.get("bytes")
    if not isinstance(image, (bytes, bytearray, memoryview)) or not image:
        raise MirrorIntegrityError(
            f"ground_truth.bytes is missing for mirror index {identity[0]}"
        )
    shard, row_in_shard, ordinal = location
    return MirrorRow(
        ordinal,
        shard,
        row_in_shard,
        identity[0],
        identity[1],
        identity[2],
        bytes(image),
    )


def load_pinned_mirror_rows(
    paths: WebsterPaths,
    state: RecoveryState,
    source_manifest: pd.DataFrame,
    *,
    show_progress: bool = False,
    progress_postfix: Callable[[], str] | None = None,
    api: object | None = None,
    filesystem: object | None = None,
) -> MirrorCatalog:
    """Load only exact ground-truth rows through resumable projected caches."""

    del state
    paths.create()
    identity_tables: dict[str, pa.Table] = {}
    identity_cache: dict[str, _CachedTable] = {}
    missing_identity: list[MirrorShard] = []
    for shard in MIRROR_SHARDS:
        cached = _read_cached_table(
            paths, shard, "identities", IDENTITY_COLUMNS, MIRROR_ROWS_PER_SHARD
        )
        if cached is None:
            missing_identity.append(shard)
        else:
            identity_cache[shard.path] = cached
            identity_tables[shard.path] = cached.table

    ground_truth_cache: dict[str, _CachedTable] = {}
    for shard in MIRROR_SHARDS:
        cached = _read_cached_table(paths, shard, "ground_truth", MIRROR_COLUMNS, None)
        if cached is not None:
            ground_truth_cache[shard.path] = cached

    needs_remote = bool(missing_identity) or len(ground_truth_cache) != len(MIRROR_SHARDS)
    if needs_remote:
        if api is None or filesystem is None:
            created_api, created_filesystem = _new_hub_clients()
            api = api or created_api
            filesystem = filesystem or created_filesystem
        _validate_remote_attestation(api)

    metadata_progress = tqdm(
        MIRROR_SHARDS,
        total=len(MIRROR_SHARDS),
        desc="[Webster 4/10] Mirror metadata",
        unit="shard",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    if progress_postfix is not None:
        metadata_progress.set_postfix_str(progress_postfix(), refresh=False)
    for shard in metadata_progress:
        if shard.path in identity_tables:
            continue
        assert filesystem is not None
        table = _load_remote_identities(filesystem, shard)
        cached = _write_cached_table(
            paths, shard, "identities", table, IDENTITY_COLUMNS
        )
        identity_cache[shard.path] = cached
        identity_tables[shard.path] = cached.table
    metadata_progress.close()

    selected, mirror_digest, source_digest, intersection_digest = validate_mirror_identity(
        identity_tables, source_manifest
    )
    accepted_by_shard: dict[str, set[Identity]] = defaultdict(set)
    for identity, (shard_path, _row, _ordinal) in selected.items():
        accepted_by_shard[shard_path].add(identity)

    image_progress = tqdm(
        total=MIRROR_PROJECTED_GROUND_TRUTH_BYTES,
        desc="[Webster 4/10] Mirror ground truth",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    if progress_postfix is not None:
        image_progress.set_postfix_str(progress_postfix(), refresh=False)
    for shard in MIRROR_SHARDS:
        cached = ground_truth_cache.get(shard.path)
        if cached is not None:
            image_progress.update(int(cached.metadata.get("source_projected_bytes", 0)))
            continue
        assert filesystem is not None
        table, projected_bytes = _load_remote_ground_truth(
            filesystem, shard, accepted_by_shard[shard.path], image_progress
        )
        expected = len(accepted_by_shard[shard.path])
        if table.num_rows != expected:
            raise MirrorIntegrityError(
                f"ground-truth projection for {shard.path} has "
                f"{table.num_rows} rows; expected {expected}"
            )
        ground_truth_cache[shard.path] = _write_cached_table(
            paths,
            shard,
            "ground_truth",
            table,
            MIRROR_COLUMNS,
            source_projected_bytes=projected_bytes,
        )
    image_progress.close()

    rows_by_identity: dict[Identity, MirrorRow] = {}
    cache_hashes: dict[str, str] = {}
    for shard in MIRROR_SHARDS:
        cached = ground_truth_cache[shard.path]
        cache_hashes[shard.path] = str(cached.metadata["parquet_sha256"])
        for item in cached.table.to_pylist():
            identity = identity_from_values(item["index"], item["raw_prompt"], item["url"])
            location = selected.get(identity)
            if location is None or location[0] != shard.path:
                raise MirrorIntegrityError("cached ground truth is not in the pinned intersection")
            if identity in rows_by_identity:
                raise MirrorIntegrityError("cached ground truth contains a duplicate identity")
            rows_by_identity[identity] = _mirror_row_from_item(item, location)
    if set(rows_by_identity) != set(selected):
        raise MirrorIntegrityError("ground-truth cache does not cover the pinned intersection")
    rows = tuple(sorted(rows_by_identity.values(), key=lambda item: item.ordinal))
    return MirrorCatalog(
        rows,
        mirror_digest,
        source_digest,
        intersection_digest,
        cache_hashes,
    )


def _phash_distance(left: str, right: str) -> int:
    if len(left) != len(right) or not left:
        raise MirrorIntegrityError("incompatible perceptual hashes in mirror audit")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as error:
        raise MirrorIntegrityError("invalid perceptual hash in mirror audit") from error


def _audit_report_matches(
    report: object,
    catalog_signature: str,
    evidence_signature: str,
) -> bool:
    return bool(
        isinstance(report, Mapping)
        and report.get("schema") == "webster_ground_truth_mirror_audit_v1"
        and report.get("repository") == MIRROR_REPOSITORY
        and report.get("revision") == MIRROR_REVISION
        and report.get("strategy_version") == MIRROR_STRATEGY_VERSION
        and report.get("catalog_signature") == catalog_signature
        and report.get("evidence_signature") == evidence_signature
        and report.get("validated_image_count") == MIRROR_EXPECTED_INTERSECTION
        and report.get("passed") is True
    )


def audit_mirror(
    paths: WebsterPaths,
    state: RecoveryState,
    catalog: MirrorCatalog,
    *,
    show_progress: bool = False,
    progress_postfix: Callable[[], str] | None = None,
) -> MirrorAudit:
    """Attest the collection against unique, non-circular exact recoveries."""

    by_identity = catalog.by_identity()
    donors: dict[Identity, list[Mapping[str, object]]] = defaultdict(list)
    for record in state.records(statuses=RECOVERED_STATUS_VALUES):
        if str(record.get("recovery_status") or "") not in INDEPENDENT_AUDIT_STATUSES:
            continue
        try:
            identity = record_identity(record)
        except MirrorIntegrityError:
            continue
        if identity not in by_identity or not recovery_artifacts_available(paths, record):
            continue
        raw_path = Path(str(record.get("local_raw_path") or ""))
        expected_sha = str(record.get("sha256") or "")
        if not verify_existing_blob(raw_path, expected_sha):
            continue
        donors[identity].append(record)

    evidence_payload: list[dict[str, object]] = []
    for identity in sorted(donors):
        records = donors[identity]
        hashes = {str(record.get("sha256") or "") for record in records}
        if len(hashes) != 1:
            raise MirrorIntegrityError(
                f"independent donors disagree for source index {identity[0]}"
            )
        evidence_payload.append(
            {
                "identity": list(identity),
                "sha256": next(iter(hashes)),
                "record_ids": sorted(str(record["record_id"]) for record in records),
                "statuses": sorted({str(record["recovery_status"]) for record in records}),
            }
        )
    evidence_signature = canonical_hash(evidence_payload)
    catalog_signature = canonical_hash(
        {
            "identity_sha256": catalog.identity_sha256,
            "source_identity_sha256": catalog.source_identity_sha256,
            "intersection_sha256": catalog.intersection_sha256,
            "cache_sha256": dict(catalog.cache_sha256),
        }
    )
    try:
        prior_report: object = json.loads(
            paths.ground_truth_mirror_audit.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        prior_report = None
    if _audit_report_matches(prior_report, catalog_signature, evidence_signature):
        assert isinstance(prior_report, Mapping)
        evidence_count = int(prior_report["evidence_count"])
        agreement_count = int(prior_report["agreement_count"])
        return MirrorAudit(
            evidence_count,
            agreement_count,
            float(prior_report["agreement_fraction"]),
            int(prior_report["mismatch_count"]),
            evidence_signature,
            True,
        )

    validation_progress = tqdm(
        catalog.rows,
        total=len(catalog.rows),
        desc="[Webster 4/10] Validate mirror images",
        unit="image",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    if progress_postfix is not None:
        validation_progress.set_postfix_str(progress_postfix(), refresh=False)
    mirror_metadata_by_identity: dict[Identity, ImageMetadata] = {}
    mirror_images: list[dict[str, object]] = []
    for mirror in validation_progress:
        try:
            metadata = validate_image_bytes(
                mirror.image_bytes,
                original_url=mirror.url,
                final_url=mirror.locator,
            )
        except Exception as error:
            raise MirrorIntegrityError(
                f"invalid ground_truth for source index {mirror.index}: "
                f"{type(error).__name__}: {error}"
            ) from error
        mirror_metadata_by_identity[mirror.identity] = metadata
        mirror_images.append(
            {
                "index": mirror.index,
                "locator": mirror.locator,
                "sha256": metadata.sha256,
                "perceptual_hash": metadata.perceptual_hash,
                "width": metadata.width,
                "height": metadata.height,
            }
        )
    validation_progress.close()
    if len(mirror_metadata_by_identity) != MIRROR_EXPECTED_INTERSECTION:
        raise MirrorIntegrityError("not every accepted mirror image was validated")

    audit_progress = tqdm(
        sorted(donors.items()),
        total=len(donors),
        desc="[Webster 4/10] Audit mirror overlaps",
        unit="target",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    if progress_postfix is not None:
        audit_progress.set_postfix_str(progress_postfix(), refresh=False)
    evidence: list[dict[str, object]] = []
    agreement_count = 0
    for identity, records in audit_progress:
        mirror = by_identity[identity]
        mirror_metadata = mirror_metadata_by_identity[identity]
        try:
            donor_path = Path(str(records[0]["local_raw_path"]))
            donor_metadata = validate_image_bytes(
                donor_path.read_bytes(),
                original_url=str(records[0].get("original_url") or ""),
                final_url=str(records[0].get("resolved_url") or ""),
            )
        except Exception as error:
            raise MirrorIntegrityError(
                f"cannot decode mirror audit pair for source index {identity[0]}: "
                f"{type(error).__name__}: {error}"
            ) from error
        distance = _phash_distance(
            mirror_metadata.perceptual_hash, donor_metadata.perceptual_hash
        )
        agrees = distance <= MAX_PHASH_DISTANCE
        agreement_count += int(agrees)
        evidence.append(
            {
                "identity": list(identity),
                "donor_record_ids": sorted(str(record["record_id"]) for record in records),
                "donor_sha256": donor_metadata.sha256,
                "donor_perceptual_hash": donor_metadata.perceptual_hash,
                "mirror_sha256": mirror_metadata.sha256,
                "mirror_perceptual_hash": mirror_metadata.perceptual_hash,
                "perceptual_distance": distance,
                "same_dimensions": (
                    mirror_metadata.width == donor_metadata.width
                    and mirror_metadata.height == donor_metadata.height
                ),
                "agrees": agrees,
                "mirror_locator": mirror.locator,
            }
        )
    audit_progress.close()

    evidence_count = len(evidence)
    agreement_fraction = (
        agreement_count / evidence_count if evidence_count else 0.0
    )
    enough_evidence = evidence_count >= MIN_INDEPENDENT_AUDIT
    passed = enough_evidence and agreement_fraction >= MIN_AGREEMENT_FRACTION
    report = {
        "schema": "webster_ground_truth_mirror_audit_v1",
        "repository": MIRROR_REPOSITORY,
        "revision": MIRROR_REVISION,
        "strategy_version": MIRROR_STRATEGY_VERSION,
        "shards": [
            {"path": shard.path, "size": shard.size, "lfs_sha256": shard.lfs_sha256}
            for shard in MIRROR_SHARDS
        ],
        "identity_sha256": catalog.identity_sha256,
        "source_identity_sha256": catalog.source_identity_sha256,
        "intersection_sha256": catalog.intersection_sha256,
        "intersection_count": len(catalog.rows),
        "catalog_signature": catalog_signature,
        "evidence_signature": evidence_signature,
        "minimum_evidence": MIN_INDEPENDENT_AUDIT,
        "maximum_perceptual_distance": MAX_PHASH_DISTANCE,
        "minimum_agreement_fraction": MIN_AGREEMENT_FRACTION,
        "evidence_count": evidence_count,
        "agreement_count": agreement_count,
        "agreement_fraction": agreement_fraction,
        "mismatch_count": evidence_count - agreement_count,
        "validated_image_count": len(mirror_images),
        "passed": passed,
        "mirror_images": mirror_images,
        "evidence": evidence,
        "audited_at": utc_now(),
    }
    atomic_write_json(paths.ground_truth_mirror_audit, report)
    state.set_run_metadata(
        "ground_truth_mirror_audit",
        {
            key: value
            for key, value in report.items()
            if key not in {"evidence", "mirror_images"}
        },
    )
    if not enough_evidence:
        raise RetryableStageError(
            "ground-truth mirror audit has only "
            f"{evidence_count}/{MIN_INDEPENDENT_AUDIT} independent targets"
        )
    if not passed:
        raise MirrorIntegrityError(
            "ground-truth mirror audit agreement is "
            f"{agreement_count}/{evidence_count} ({agreement_fraction:.1%}); "
            f"required {MIN_AGREEMENT_FRACTION:.0%}"
        )
    return MirrorAudit(
        evidence_count,
        agreement_count,
        agreement_fraction,
        evidence_count - agreement_count,
        evidence_signature,
        False,
    )


def prepare_verified_mirror(
    paths: WebsterPaths,
    state: RecoveryState,
    source_manifest: pd.DataFrame,
    *,
    show_progress: bool = False,
    progress_postfix: Callable[[], str] | None = None,
) -> tuple[MirrorCatalog, MirrorAudit]:
    catalog = load_pinned_mirror_rows(
        paths,
        state,
        source_manifest,
        show_progress=show_progress,
        progress_postfix=progress_postfix,
    )
    audit = audit_mirror(
        paths,
        state,
        catalog,
        show_progress=show_progress,
        progress_postfix=progress_postfix,
    )
    state.set_run_metadata_batch(
        {
            "ground_truth_mirror_strategy_version": MIRROR_STRATEGY_VERSION,
            "ground_truth_mirror_source": {
                "repository": MIRROR_REPOSITORY,
                "revision": MIRROR_REVISION,
                "columns": list(MIRROR_COLUMNS),
                "intersection_count": len(catalog.rows),
                "identity_sha256": catalog.identity_sha256,
                "source_identity_sha256": catalog.source_identity_sha256,
                "intersection_sha256": catalog.intersection_sha256,
                "cache_sha256": dict(catalog.cache_sha256),
            },
        }
    )
    return catalog, audit


def recover_mirror_row(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    row: MirrorRow,
) -> StageResult:
    """Import one globally audited ground-truth image into canonical storage."""

    if record_identity(record) != row.identity:
        raise MirrorIntegrityError("mirror row does not exactly identify the recovery record")
    try:
        metadata, raw_path, normalized_path = store_image(
            paths,
            state,
            row.image_bytes,
            original_url=row.url,
            final_url=row.locator,
        )
    except Exception as error:
        raise MirrorIntegrityError(
            f"audited mirror image became invalid for source index {row.index}: "
            f"{type(error).__name__}: {error}"
        ) from error
    attempt_id = state.log_attempt(
        str(record["record_id"]),
        "hf_ground_truth_mirror",
        attempted_url=row.locator,
        response_size=len(row.image_bytes),
        response_mime=f"image/{metadata.image_format.casefold()}",
        validation_result="valid_audited_ground_truth",
        candidate_sha256=metadata.sha256,
    )
    apply_image_resolution(
        paths,
        state,
        record,
        metadata,
        raw_path,
        normalized_path,
        status=RecoveryStatus.RECOVERED_GROUND_TRUTH_MIRROR,
        method="hf_ground_truth_mirror",
        resolved_url=row.locator,
    )
    state.update_attempt(
        attempt_id,
        validation_result="valid_audited_ground_truth",
        candidate_sha256=metadata.sha256,
    )
    return StageResult(StageOutcome.RECOVERED, "audited ground-truth mirror")
