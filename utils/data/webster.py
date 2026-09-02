"""Webster metadata, organized model views, dataset access, and inspection.

This module is deliberately independent of recovery clients.  Importing
``WebsterDataset`` opens only local Parquet and image files; it never imports
archive or diffusion code and never performs network access.
"""

from __future__ import annotations

import json
import math
import operator
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypedDict
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from utils.common.io import (
    atomic_copy,
    atomic_write_frame_csv,
    atomic_write_frame_parquet,
    atomic_write_json,
    file_sha256,
    utc_now,
)

if TYPE_CHECKING:
    from .state import RecoveryState


EXPECTED_ROWS_PER_MODEL = 500
EXPECTED_TOTAL_ROWS = 1_500
HF_REPOSITORY = "fraisdufour/templates-verbs"
MODEL_FILES: dict[str, str] = {
    "sdv1": "groundtruth_parquets/sdv1_bb_edge_groundtruth.parquet",
    "sdv2": "groundtruth_parquets/sdv2_bb_edge_groundtruth.parquet",
    "realisticvision": (
        "groundtruth_parquets/realistic_vision_sdv1_edge_groundtruth.parquet"
    ),
}


class WebsterModel(str, Enum):
    """One of the three model-specific Webster metadata tables."""

    SDV1 = "sdv1"
    SDV2 = "sdv2"
    REALISTIC_VISION = "realisticvision"


class RecoveryStatus(str, Enum):
    """Durable record-level recovery status."""

    PENDING = "pending"
    RETRYABLE_ERROR = "retryable_error"
    RECOVERED_EXACT_URL = "recovered_exact_url"
    RECOVERED_OFFICIAL_CACHE = "recovered_official_cache"
    RECOVERED_GROUND_TRUTH_MIRROR = "recovered_ground_truth_mirror"
    RECOVERED_WAYBACK_EXACT = "recovered_wayback_exact"
    RECOVERED_ARQUIVO_EXACT = "recovered_arquivo_exact"
    RECOVERED_COMMONCRAWL_EXACT = "recovered_commoncrawl_exact"
    RECOVERED_SOURCE_PAGE = "recovered_source_page"
    RECOVERED_VERIFIED_DUPLICATE = "recovered_verified_duplicate"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    INVALID_METADATA = "invalid_metadata"


class RecoveryStage(str, Enum):
    """A resumable recovery stage, in execution order."""

    EXACT_REUSE = "exact_reuse"
    OFFICIAL_ASSETS = "official_assets"
    DIRECT = "direct"
    GROUND_TRUTH_MIRROR = "ground_truth_mirror"
    WAYBACK = "wayback"
    ARQUIVO = "arquivo"
    COMMON_CRAWL = "common_crawl"
    FINAL_RESOLUTION = "final_resolution"


class StageOutcome(str, Enum):
    """The durable result of one recovery stage."""

    RUNNING = "running"
    RECOVERED = "recovered"
    MISS = "miss"
    AMBIGUOUS = "ambiguous"
    RETRYABLE_ERROR = "retryable_error"
    INVALID_METADATA = "invalid_metadata"
    ALREADY_RECOVERED = "already_recovered"
    NOT_APPLICABLE = "not_applicable"


class DownloadExitCode(IntEnum):
    COMPLETE = 0
    FATAL_ERROR = 1
    RETRYABLE_RECORDS = 2


RECOVERED_STATUSES = frozenset(
    {
        RecoveryStatus.RECOVERED_EXACT_URL,
        RecoveryStatus.RECOVERED_OFFICIAL_CACHE,
        RecoveryStatus.RECOVERED_GROUND_TRUTH_MIRROR,
        RecoveryStatus.RECOVERED_WAYBACK_EXACT,
        RecoveryStatus.RECOVERED_ARQUIVO_EXACT,
        RecoveryStatus.RECOVERED_COMMONCRAWL_EXACT,
        RecoveryStatus.RECOVERED_SOURCE_PAGE,
        RecoveryStatus.RECOVERED_VERIFIED_DUPLICATE,
    }
)
RECOVERED_STATUS_VALUES = frozenset(item.value for item in RECOVERED_STATUSES)
TERMINAL_STATUSES = frozenset(
    set(RECOVERED_STATUSES)
    | {
        RecoveryStatus.AMBIGUOUS,
        RecoveryStatus.UNRESOLVED,
        RecoveryStatus.INVALID_METADATA,
    }
)
TERMINAL_STATUS_VALUES = frozenset(item.value for item in TERMINAL_STATUSES)
ACTIVE_STATUS_VALUES = frozenset(
    {RecoveryStatus.PENDING.value, RecoveryStatus.RETRYABLE_ERROR.value}
)
ALL_STATUS_VALUES = frozenset(item.value for item in RecoveryStatus)
STAGE_ORDER = tuple(RecoveryStage)


@dataclass(frozen=True, slots=True)
class StageResult:
    outcome: StageOutcome
    message: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if self.retryable != (self.outcome is StageOutcome.RETRYABLE_ERROR):
            raise ValueError(
                "retryable must be true exactly when outcome is retryable_error"
            )

    @classmethod
    def retryable_error(cls, message: str) -> StageResult:
        return cls(StageOutcome.RETRYABLE_ERROR, message, True)


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    root: Path
    total_records: int
    status_counts: Mapping[str, int]
    retryable_record_ids: tuple[str, ...]
    manifest_parquet: Path
    manifest_csv: Path
    summary_json: Path
    retryable_csv: Path
    exit_code: int

    @property
    def retryable_count(self) -> int:
        return len(self.retryable_record_ids)


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    sha256: str
    sha1: str
    perceptual_hash: str
    width: int
    height: int
    mode: str
    image_format: str
    extension: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class HTTPResult:
    data: bytes
    status_code: int
    content_type: str | None
    final_url: str
    attempt_id: int | None
    request_key: str
    from_cache: bool


@dataclass(frozen=True, slots=True)
class ArchiveCandidate:
    record_id: str
    sha256: str
    perceptual_hash: str
    raw_path: str
    normalized_path: str
    candidate_path: str
    strategy: str
    resolved_url: str
    archive_timestamp: str | None
    archive_digest: str | None
    source_page: bool
    response_mime: str | None


@dataclass(frozen=True, slots=True)
class WarcPayload:
    record_type: str
    http_status: str | None
    decoded_body: bytes | None
    body_too_large: bool
    content_type: str | None
    target_uri: str | None
    reported_payload_digest: str | None
    computed_payload_digest: str | None
    payload_digest_verified: bool
    warc_record_id: str | None
    warc_date: str | None
    refers_to: str | None
    refers_to_target_uri: str | None
    refers_to_date: str | None


class WebsterDataError(RuntimeError):
    """Base class for expected Webster data failures."""


class DownloadError(WebsterDataError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        transient: bool = False,
        permanent_host_failure: bool = False,
        retry_after: float = 0.0,
        service_failure_key: str | None = None,
        deferred_by_circuit: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient
        self.permanent_host_failure = permanent_host_failure
        self.retry_after = retry_after
        self.service_failure_key = service_failure_key
        self.deferred_by_circuit = deferred_by_circuit


class RetryableStageError(WebsterDataError):
    """A stage failure which may succeed on a later invocation."""


class StateSchemaError(WebsterDataError):
    """The SQLite file is not the current recovery schema."""


class StateConsistencyError(WebsterDataError):
    """Persisted recovery facts contradict one another."""


class ImageValidationError(ValueError):
    """Bytes are not an acceptable source image."""


class DatasetIntegrityError(WebsterDataError):
    """Metadata and organized files are inconsistent."""


class OriginalIndexError(ValueError):
    """Source metadata lacks a usable original index."""


class WebsterDatasetError(DatasetIntegrityError):
    """Base error for malformed Webster manifests and images."""


class WebsterManifestError(WebsterDatasetError):
    """A model metadata file violates a dataset invariant."""


class WebsterImageError(WebsterDatasetError):
    def __init__(self, record_id: str, image_path: Path, kind: str, message: str):
        self.record_id = record_id
        self.image_path = image_path
        self.kind = kind
        super().__init__(f"{record_id}: {message}: {image_path}")


@dataclass(frozen=True, slots=True)
class WebsterPaths:
    """All paths owned beneath ``ROOT/data/webster``."""

    base: Path

    @classmethod
    def from_root(cls, root: str | Path) -> WebsterPaths:
        return cls(Path(root).expanduser().resolve() / "data" / "webster")

    @property
    def root(self) -> Path:
        return self.base.parent.parent

    @property
    def source_parquets(self) -> Path:
        return self.base / "source_parquets"

    @property
    def manifests(self) -> Path:
        return self.base / "manifests"

    @property
    def raw(self) -> Path:
        return self.base / "images" / "raw"

    @property
    def normalized(self) -> Path:
        return self.base / "images" / "normalized"

    @property
    def auxiliary(self) -> Path:
        return self.base / "auxiliary"

    @property
    def candidates(self) -> Path:
        return self.base / "candidates"

    @property
    def logs(self) -> Path:
        return self.base / "logs"

    @property
    def state(self) -> Path:
        return self.base / "state"

    @property
    def database(self) -> Path:
        return self.state / "recovery.sqlite"

    @property
    def hub_cache(self) -> Path:
        return self.state / "huggingface_cache"

    @property
    def http_cache(self) -> Path:
        return self.state / "http_cache"

    @property
    def ground_truth_mirror_cache(self) -> Path:
        return self.state / "ground_truth_mirror"

    @property
    def ground_truth_mirror_audit(self) -> Path:
        return self.state / "ground_truth_mirror_audit.json"

    @property
    def source_records(self) -> Path:
        return self.state / "source_records.parquet"

    @property
    def model_views_state(self) -> Path:
        return self.state / "model_views.json"

    @property
    def pipeline_lock(self) -> Path:
        return self.state / "webster-pipeline.lock"

    def model_directory(self, model: WebsterModel | str) -> Path:
        return self.base / coerce_model(model).value

    def model_images(self, model: WebsterModel | str) -> Path:
        return self.model_directory(model) / "images"

    def model_metadata_parquet(self, model: WebsterModel | str) -> Path:
        return self.model_directory(model) / "metadata.parquet"

    def model_metadata_csv(self, model: WebsterModel | str) -> Path:
        return self.model_directory(model) / "metadata.csv"

    def create(self) -> None:
        for directory in (
            self.source_parquets,
            self.manifests,
            self.raw,
            self.normalized,
            self.auxiliary / "retrieved_targets",
            self.auxiliary / "templates",
            self.auxiliary / "masks",
            self.auxiliary / "paired_sources",
            self.candidates,
            self.logs,
            self.state,
            self.hub_cache,
            self.http_cache,
        ):
            self.ground_truth_mirror_cache,
            directory.mkdir(parents=True, exist_ok=True)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


class OriginalIndexFields(TypedDict):
    original_index_raw: object
    original_index: str
    image_filename: str


_SAFE_FILENAME_CHARACTERS = "-._~"
_WINDOWS_RESERVED_STEMS = {
    "aux", "clock$", "con", "nul", "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def canonicalize_original_index(value: object) -> str:
    """Return the untouched source index as a safe deterministic file stem."""

    if value is None or isinstance(value, (bool, bytes, bytearray, memoryview)):
        raise OriginalIndexError("original index is null or has an unusable type")
    try:
        integer = operator.index(value)
    except TypeError:
        integer = None
    if integer is not None:
        text = str(integer)
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise OriginalIndexError("original index is non-finite")
        text = str(int(value)) if value == value.to_integral_value() else str(value)
    elif isinstance(value, float) or _is_real_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise OriginalIndexError("original index is non-finite")
        text = str(int(number)) if number.is_integer() else format(number, ".17g")
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise OriginalIndexError(
            f"original index has an unusable type: {type(value).__name__}"
        )
    if not text or text in {".", ".."} or "\x00" in text:
        raise OriginalIndexError("original index is empty or unsafe")
    encoded = quote(text, safe=_SAFE_FILENAME_CHARACTERS)
    if encoded.casefold().split(".", 1)[0] in _WINDOWS_RESERVED_STEMS:
        encoded = "%" + encoded
    if len(encoded.encode("utf-8")) > 251:
        raise OriginalIndexError("original index is too long for an image filename")
    return encoded


def original_index_fields(value: object) -> OriginalIndexFields:
    original_index = canonicalize_original_index(value)
    return {
        "original_index_raw": value,
        "original_index": original_index,
        "image_filename": f"{original_index}.png",
    }


def _is_real_number(value: object) -> bool:
    try:
        import numbers

        return isinstance(value, numbers.Real)
    except TypeError:
        return False


def normalize_webster_type(value: object) -> str:
    """Normalize Webster's MV/RV/TV/N labels without selecting prompts."""

    if _is_missing(value):
        return "UNKNOWN"
    text = str(value).strip()
    folded = re.sub(r"[\s_-]+", "", text).casefold()
    aliases = {
        "mv": "MV", "memorizedverbatim": "MV", "memorized": "MV",
        "rv": "RV", "retrievalverbatim": "RV", "retrieval": "RV",
        "tv": "TV", "templateverbatim": "TV", "template": "TV",
        "n": "N", "normal": "N", "nonmemorized": "N",
    }
    return aliases.get(folded, text.upper() if text.upper() in {"MV", "RV", "TV", "N"} else "UNKNOWN")


def normalize_url(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if parts.scheme.casefold() not in {"http", "https"} or not parts.netloc:
        return None
    return urlunsplit(
        (parts.scheme.casefold(), parts.netloc.casefold(), parts.path, parts.query, "")
    )


PROMPT_ALIASES = ("prompt", "caption", "text", "TEXT", "prompt_raw")
URL_ALIASES = ("url", "URL", "image_url", "original_url")
INDEX_ALIASES = ("index", "id", "key", "laion_index", "sample_id", "laion_or_sample_index")
OVERFIT_ALIASES = ("overfit_type", "memorization_type", "category", "label")
SOURCE_PAGE_ALIASES = ("source_page_url", "page_url", "webpage_url", "source_url", "document_url")


def find_alias(columns: Sequence[str], aliases: Sequence[str], required: bool) -> str | None:
    exact = {column: column for column in columns}
    folded = {column.casefold(): column for column in columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    if required:
        raise DatasetIntegrityError(
            f"metadata lacks required column; expected one of {tuple(aliases)}"
        )
    return None


def resolve_schema(columns: Sequence[str]) -> dict[str, str | None]:
    return {
        "prompt": find_alias(columns, PROMPT_ALIASES, True),
        "url": find_alias(columns, URL_ALIASES, True),
        "index": find_alias(columns, INDEX_ALIASES, True),
        "overfit_type": find_alias(columns, OVERFIT_ALIASES, False),
        "source_page_url": find_alias(columns, SOURCE_PAGE_ALIASES, False),
    }


def validate_source_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "record_id", "model_name", "source_row_number", "prompt_raw",
        "laion_or_sample_index", "original_url", "normalized_url",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise DatasetIntegrityError(f"source manifest lacks columns: {', '.join(missing)}")
    if len(manifest) != EXPECTED_TOTAL_ROWS:
        raise DatasetIntegrityError(
            f"source manifest has {len(manifest)} rows; expected {EXPECTED_TOTAL_ROWS}"
        )
    if manifest["record_id"].isna().any() or manifest["record_id"].duplicated().any():
        raise DatasetIntegrityError("record IDs must be present and unique")
    for model in WebsterModel:
        rows = manifest.loc[manifest["model_name"].astype(str) == model.value]
        if len(rows) != EXPECTED_ROWS_PER_MODEL:
            raise DatasetIntegrityError(
                f"{model.value} has {len(rows)} rows; expected {EXPECTED_ROWS_PER_MODEL}"
            )
        source_rows = [operator.index(value) for value in rows["source_row_number"]]
        if source_rows != list(range(EXPECTED_ROWS_PER_MODEL)):
            raise DatasetIntegrityError(f"{model.value} source order is not 0 through 499")
        expected_ids = [f"{model.value}-{index:04d}" for index in source_rows]
        if rows["record_id"].astype(str).tolist() != expected_ids:
            raise DatasetIntegrityError(f"{model.value} record IDs do not preserve source order")


def enrich_source_frame(model: WebsterModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable recovery identity columns without changing source columns."""

    if len(frame) != EXPECTED_ROWS_PER_MODEL:
        raise DatasetIntegrityError(
            f"{model.value} contains {len(frame)} rows; expected {EXPECTED_ROWS_PER_MODEL}"
        )
    binding = resolve_schema([str(column) for column in frame.columns])
    output = frame.copy(deep=True)
    output["model_name"] = model.value
    output["source_row_number"] = range(len(output))
    output["record_id"] = [f"{model.value}-{row:04d}" for row in range(len(output))]
    output["prompt_raw"] = output[str(binding["prompt"])]
    output["overfit_type"] = (
        output[str(binding["overfit_type"])] if binding["overfit_type"] else None
    )
    output["laion_or_sample_index"] = output[str(binding["index"])]
    output["original_url"] = output[str(binding["url"])]
    output["normalized_url"] = output["original_url"].map(normalize_url)
    output["source_page_url"] = (
        output[str(binding["source_page_url"])] if binding["source_page_url"] else None
    )
    return output


def load_source_manifest(paths: WebsterPaths) -> pd.DataFrame:
    """Load validated local source Parquets; no download is attempted."""

    frames: list[pd.DataFrame] = []
    for model in WebsterModel:
        path = paths.source_parquets / Path(MODEL_FILES[model.value]).name
        try:
            frame = pd.read_parquet(path)
        except (FileNotFoundError, ImportError, OSError, ValueError) as error:
            raise DatasetIntegrityError(f"cannot read source metadata {path}: {error}") from error
        frames.append(enrich_source_frame(model, frame))
    manifest = pd.concat(frames, ignore_index=True, sort=False)
    validate_source_manifest(manifest)
    return manifest


def _resolve_huggingface_cache_file(
    downloaded: str | Path, cache_directory: Path,
) -> Path:
    """Resolve a Hub snapshot symlink to a regular blob inside its cache."""

    source = Path(downloaded)
    try:
        cache_location = Path(os.path.abspath(cache_directory))
        source_location = Path(os.path.abspath(source))
        source_location.relative_to(cache_location)
        cache_root = cache_directory.resolve(strict=True)
        resolved = source.resolve(strict=True)
        resolved.relative_to(cache_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise DownloadError(
            f"Hugging Face returned an unsafe cached file: {source}"
        ) from error
    if not resolved.is_file() or resolved.is_symlink():
        raise DownloadError(
            f"Hugging Face cached metadata is not a regular file: {resolved}"
        )
    return resolved


def acquire_and_build_manifest(paths: WebsterPaths, state: RecoveryState) -> pd.DataFrame:
    """Reuse local metadata or acquire only missing pinned source Parquets."""

    paths.create()
    revision = state.get_run_metadata("huggingface_revision")
    missing = [
        model
        for model in WebsterModel
        if not (paths.source_parquets / Path(MODEL_FILES[model.value]).name).is_file()
    ]
    if missing:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as error:
            raise DownloadError("huggingface_hub is required to acquire Webster metadata") from error
    models = list(WebsterModel)
    missing_set = set(missing)
    with tqdm(
        models,
        total=len(models),
        desc="[Webster metadata] Source files",
        unit="file",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    ) as progress:
        if missing and (
            not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)
        ):
            info = HfApi().repo_info(
                repo_id=HF_REPOSITORY, repo_type="dataset"
            )
            revision = str(info.sha)
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise DownloadError(
                    "Hugging Face did not return an immutable revision"
                )
            state.set_run_metadata("huggingface_revision", revision)
        for model in progress:
            if model not in missing_set:
                continue
            cached = hf_hub_download(
                repo_id=HF_REPOSITORY,
                repo_type="dataset",
                filename=MODEL_FILES[model.value],
                revision=revision,
                cache_dir=paths.hub_cache,
            )
            source = _resolve_huggingface_cache_file(
                cached, paths.hub_cache
            )
            atomic_copy(
                source,
                paths.source_parquets / Path(MODEL_FILES[model.value]).name,
            )
    manifest = load_source_manifest(paths)
    atomic_write_frame_parquet(manifest, paths.source_records)
    state.initialize_records(manifest)
    state.set_run_metadata_batch(
        {
            "source_manifest_created_at": utc_now(),
            "source_parquet_sha256": {
                model.value: file_sha256(
                    paths.source_parquets / Path(MODEL_FILES[model.value]).name
                )
                for model in WebsterModel
            },
        }
    )
    return manifest


@dataclass(frozen=True, slots=True)
class ModelOrganizationSummary:
    model_name: str
    rows: int
    available_images: int
    unavailable_images: int
    metadata_parquet: Path
    metadata_csv: Path


@dataclass(frozen=True, slots=True)
class OrganizationSummary:
    models: tuple[ModelOrganizationSummary, ...]

    @property
    def rebuilt_models(self) -> tuple[str, ...]:
        return tuple(model.model_name for model in self.models)


class OrganizationError(DatasetIntegrityError):
    """The combined recovery manifest cannot form safe model views."""


def organize_model_views(
    paths: WebsterPaths,
    manifest: pd.DataFrame,
    *,
    show_progress: bool = False,
    progress_postfix: Callable[[], str] | None = None,
) -> OrganizationSummary:
    """Atomically publish flat model views using original-index filenames."""

    validate_source_manifest(manifest)
    summaries: list[ModelOrganizationSummary] = []
    models = tqdm(
        list(WebsterModel),
        total=len(WebsterModel),
        desc="[Webster 9/10] Publish model views",
        unit="model",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    if progress_postfix is not None:
        models.set_postfix_str(progress_postfix(), refresh=False)
    for model in models:
        rows = manifest.loc[manifest["model_name"].astype(str) == model.value]
        rows = rows.sort_values("source_row_number", kind="stable").reset_index(drop=True)
        prepared = _prepare_model_rows(paths, model, rows)
        images = paths.model_images(model)
        images.mkdir(parents=True, exist_ok=True)
        expected_names = set(prepared["image_filename"].dropna().astype(str))
        for entry in images.iterdir():
            if entry.is_file() and entry.name not in expected_names:
                entry.unlink()
        for row in prepared.loc[prepared["image_available"]].to_dict(orient="records"):
            source = Path(str(row["local_normalized_path"]))
            _publish_reference(source, images / str(row["image_filename"]))
        atomic_write_frame_parquet(prepared, paths.model_metadata_parquet(model))
        atomic_write_frame_csv(prepared, paths.model_metadata_csv(model))
        available = int(prepared["image_available"].sum())
        summaries.append(
            ModelOrganizationSummary(
                model.value,
                len(prepared),
                available,
                len(prepared) - available,
                paths.model_metadata_parquet(model),
                paths.model_metadata_csv(model),
            )
        )
        if progress_postfix is not None:
            models.set_postfix_str(progress_postfix(), refresh=False)
    atomic_write_json(
        paths.model_views_state,
        {
            "schema": "webster_model_views_v2",
            "models": {
                item.model_name: {
                    "rows": item.rows,
                    "available_images": item.available_images,
                    "metadata_sha256": file_sha256(item.metadata_parquet),
                }
                for item in summaries
            },
        },
    )
    return OrganizationSummary(tuple(summaries))


def _prepare_model_rows(
    paths: WebsterPaths, model: WebsterModel, rows: pd.DataFrame
) -> pd.DataFrame:
    prepared = rows.copy(deep=True)
    fields: list[OriginalIndexFields] = []
    seen: dict[str, str] = {}
    for row in prepared.to_dict(orient="records"):
        try:
            item = original_index_fields(row.get("laion_or_sample_index"))
        except OriginalIndexError as error:
            raise OrganizationError(f"{row['record_id']}: {error}") from error
        conflict = seen.get(item["original_index"])
        if conflict is not None:
            raise OrganizationError(
                f"{model.value} original index {item['original_index']} is shared by "
                f"{conflict} and {row['record_id']}"
            )
        seen[item["original_index"]] = str(row["record_id"])
        fields.append(item)
    prepared["original_index_raw"] = [item["original_index_raw"] for item in fields]
    prepared["original_index"] = [item["original_index"] for item in fields]
    recovered = prepared["recovery_status"].astype(str).isin(RECOVERED_STATUS_VALUES)
    prepared["image_available"] = recovered
    prepared["image_filename"] = [
        item["image_filename"] if available else None
        for item, available in zip(fields, recovered, strict=True)
    ]
    prepared["image_path"] = [
        f"{model.value}/images/{item['image_filename']}" if available else None
        for item, available in zip(fields, recovered, strict=True)
    ]
    for row in prepared.loc[recovered].to_dict(orient="records"):
        source = Path(str(row.get("local_normalized_path")))
        if not source.is_file() or source.is_symlink() or not path_is_within(source, paths.normalized):
            raise OrganizationError(
                f"{row['record_id']} has no safe normalized recovery image: {source}"
            )
        try:
            with Image.open(source) as image:
                image.verify()
        except (OSError, SyntaxError, ValueError) as error:
            raise OrganizationError(f"{row['record_id']} normalized image is corrupt") from error
    return prepared


def _publish_reference(source: Path, destination: Path) -> None:
    if destination.is_file() and not destination.is_symlink():
        try:
            if os.path.samefile(source, destination) or file_sha256(source) == file_sha256(destination):
                return
        except OSError:
            destination.unlink(missing_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


ImageTransform: TypeAlias = Callable[[Image.Image], object]


class WebsterDataset(torch.utils.data.Dataset):
    """Load every completed paired target, or the full raw 500-row view.

    ``recovered_only=True`` is the experiment-facing all-completed view.
    ``False`` is the unfiltered raw metadata view.  Prompt selection is a
    separate concern handled by :mod:`utils.data.selection`.
    """

    REQUIRED_COLUMNS = {
        "record_id", "model_name", "source_row_number", "laion_or_sample_index",
        "original_index_raw", "original_index", "image_filename", "prompt_raw",
        "overfit_type", "recovery_status", "recovery_method", "image_available",
        "image_path", "sha256", "original_url", "auxiliary_asset_paths",
    }

    def __init__(
        self,
        root: str | Path,
        model: str | WebsterModel,
        transform: ImageTransform | None = None,
        recovered_only: bool = True,
        categories: str | None | Iterable[str | None] = None,
        statuses: str | RecoveryStatus | Iterable[str | RecoveryStatus] | None = None,
        defer_image_validation: bool = False,
    ) -> None:
        if not isinstance(defer_image_validation, bool):
            raise TypeError("defer_image_validation must be a bool")
        self.paths = WebsterPaths.from_root(root)
        self.model = coerce_model(model)
        self.transform = transform
        self.defer_image_validation = defer_image_validation
        metadata_path = self.paths.model_metadata_parquet(self.model)
        try:
            manifest = pd.read_parquet(metadata_path)
        except (FileNotFoundError, ImportError, OSError, TypeError, ValueError) as error:
            raise WebsterManifestError(f"model metadata cannot be read: {metadata_path}") from error
        self._manifest = self._validate_manifest(manifest)
        rows = self._manifest
        category_filter = _category_filter(categories)
        if category_filter is not None:
            rows = rows.loc[rows["webster_overfit_type_normalized"].isin(category_filter)]
        status_filter = _status_filter(statuses)
        if status_filter is not None:
            rows = rows.loc[rows["recovery_status"].astype(str).isin(status_filter)]
        if recovered_only:
            rows = rows.loc[rows["image_available"]]
        self._rows = rows.reset_index(drop=True)

    def _validate_manifest(self, manifest: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(self.REQUIRED_COLUMNS - set(manifest.columns))
        if missing:
            raise WebsterManifestError(f"{self.model.value} metadata is missing: {', '.join(missing)}")
        if len(manifest) != EXPECTED_ROWS_PER_MODEL:
            raise WebsterManifestError(
                f"{self.model.value} metadata has {len(manifest)} rows; expected 500"
            )
        expected_rows = list(range(EXPECTED_ROWS_PER_MODEL))
        try:
            source_rows = [operator.index(value) for value in manifest["source_row_number"]]
        except TypeError as error:
            raise WebsterManifestError("source row numbers must be integers") from error
        if source_rows != expected_rows:
            raise WebsterManifestError("metadata must preserve source rows 0 through 499")
        expected_ids = [f"{self.model.value}-{row:04d}" for row in expected_rows]
        if manifest["record_id"].astype(str).tolist() != expected_ids:
            raise WebsterManifestError("record IDs do not match source-row order")
        result = manifest.copy(deep=True)
        normalized_labels: list[str] = []
        target_hashes: list[str | None] = []
        seen: set[str] = set()
        for row in result.to_dict(orient="records"):
            record_id = str(row["record_id"])
            fields = original_index_fields(row.get("laion_or_sample_index"))
            if not _values_equal(row.get("original_index_raw"), fields["original_index_raw"]):
                raise WebsterManifestError(f"{record_id} does not preserve original_index_raw")
            if row.get("original_index") != fields["original_index"]:
                raise WebsterManifestError(f"{record_id} has a noncanonical original_index")
            if fields["original_index"] in seen:
                raise WebsterManifestError(f"duplicate original index {fields['original_index']}")
            seen.add(fields["original_index"])
            available = _strict_bool(row.get("image_available"), record_id)
            status = str(row.get("recovery_status"))
            if available != (status in RECOVERED_STATUS_VALUES):
                raise WebsterManifestError(f"{record_id} image availability contradicts recovery status")
            normalized_labels.append(normalize_webster_type(row.get("overfit_type")))
            if available:
                expected_filename = fields["image_filename"]
                if row.get("image_filename") != expected_filename:
                    raise WebsterManifestError(f"{record_id} image filename must be {expected_filename}")
                image_path = self._resolve_image_path(record_id, expected_filename, row.get("image_path"))
                claimed = row.get("target_image_sha256")
                if self.defer_image_validation:
                    target_hashes.append(None if _is_missing(claimed) else str(claimed))
                else:
                    observed_hash = file_sha256(image_path)
                    if not _is_missing(claimed) and str(claimed) != observed_hash:
                        raise WebsterManifestError(f"{record_id} target image SHA-256 differs")
                    target_hashes.append(observed_hash)
            else:
                if not _is_missing(row.get("image_filename")) or not _is_missing(row.get("image_path")):
                    raise WebsterManifestError(f"{record_id} unavailable row names an image")
                target_hashes.append(None)
        result["webster_overfit_type_raw"] = result["overfit_type"]
        result["webster_overfit_type_normalized"] = normalized_labels
        result["target_image_sha256"] = target_hashes
        return result

    def _resolve_image_path(self, record_id: str, filename: str, value: object) -> Path:
        expected = Path(self.model.value) / "images" / filename
        if _is_missing(value) or Path(str(value)) != expected:
            raise WebsterManifestError(f"{record_id} image path must be {expected.as_posix()}")
        resolved = (self.paths.base / expected).resolve(strict=False)
        if resolved.parent != self.paths.model_images(self.model).resolve(strict=False):
            raise WebsterManifestError(f"{record_id} image path leaves its model directory")
        return resolved

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        try:
            metadata = _row_metadata(self._rows.iloc[index])
        except IndexError as error:
            raise IndexError(f"WebsterDataset index is out of range: {index}") from error
        record_id = str(metadata["record_id"])
        image: object | None = None
        if bool(metadata["image_available"]):
            path = self._resolve_image_path(
                record_id, str(metadata["image_filename"]), metadata.get("image_path")
            )
            image = self._load_image(record_id, path)
            if self.defer_image_validation:
                observed_hash = file_sha256(path)
                claimed_hash = metadata.get("target_image_sha256")
                if (
                    not _is_missing(claimed_hash)
                    and str(claimed_hash) != observed_hash
                ):
                    raise WebsterImageError(
                        record_id,
                        path,
                        "hash_mismatch",
                        "target image SHA-256 differs",
                    )
                metadata["target_image_sha256"] = observed_hash
            if self.transform is not None:
                image = self.transform(image)
        auxiliary = _decode_auxiliary_paths(metadata.get("auxiliary_asset_paths"))
        metadata["auxiliary_asset_paths"] = auxiliary
        return {
            "image": image,
            "prompt": metadata.get("prompt_raw"),
            "record_id": record_id,
            "original_index_raw": metadata.get("original_index_raw"),
            "original_index": metadata.get("original_index"),
            "image_filename": metadata.get("image_filename"),
            "model_name": metadata.get("model_name"),
            "source_row_number": metadata.get("source_row_number"),
            "overfit_type": metadata.get("overfit_type"),
            "webster_overfit_type_raw": metadata.get("webster_overfit_type_raw"),
            "webster_overfit_type_normalized": metadata.get("webster_overfit_type_normalized"),
            "recovery_status": metadata.get("recovery_status"),
            "image_path": metadata.get("image_path"),
            "sha256": metadata.get("sha256"),
            "target_image_sha256": metadata.get("target_image_sha256"),
            "original_url": metadata.get("original_url"),
            "auxiliary_asset_paths": auxiliary,
            "metadata": metadata,
        }

    def _load_image(self, record_id: str, path: Path) -> Image.Image:
        if not path.is_file():
            raise WebsterImageError(record_id, path, "missing", "referenced image is missing")
        try:
            with Image.open(path) as image:
                image.load()
                return image.convert("RGB").copy()
        except (EOFError, Image.DecompressionBombError, OSError, SyntaxError, UnidentifiedImageError, ValueError) as error:
            raise WebsterImageError(record_id, path, "corrupt", "referenced image is corrupt") from error

    @property
    def total_manifest_rows(self) -> int:
        return len(self._manifest)

    @property
    def metadata_path(self) -> Path:
        """Return the exact Parquet manifest backing this model view."""

        return self.paths.model_metadata_parquet(self.model)

    @property
    def available_count(self) -> int:
        return int(self._rows["image_available"].sum())

    @property
    def unavailable_count(self) -> int:
        return len(self._rows) - self.available_count

    def iter_metadata(self) -> Iterator[dict[str, object]]:
        for _, row in self._rows.iterrows():
            metadata = _row_metadata(row)
            metadata["auxiliary_asset_paths"] = _decode_auxiliary_paths(
                metadata.get("auxiliary_asset_paths")
            )
            yield metadata

    def statistics(self) -> dict[str, object]:
        return {
            "total_manifest_rows": self.total_manifest_rows,
            "selected_rows": len(self),
            "by_recovery_status": dict(sorted(Counter(self._rows["recovery_status"].astype(str)).items())),
            "by_webster_category": dict(sorted(Counter(self._rows["webster_overfit_type_normalized"]).items())),
            "by_recovery_method": dict(sorted(Counter(_display(value, "none") for value in self._rows["recovery_method"]).items())),
            "by_image_availability": {
                "available": self.available_count,
                "unavailable": self.unavailable_count,
            },
        }


@dataclass(frozen=True, slots=True)
class DuplicateDigest:
    sha256: str
    record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelInspection:
    model_name: str
    manifest_rows: int
    recovered_image_records: int
    unique_image_files: int
    unresolved_records: int
    ambiguous_records: int
    invalid_metadata_records: int
    pending_records: int
    retryable_error_records: int
    loadable_images: int
    corrupt_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    orphan_files: tuple[str, ...]
    unexpected_entries: tuple[str, ...]
    duplicate_digests: tuple[DuplicateDigest, ...]
    status_counts: dict[str, int]
    category_counts: dict[str, int]
    recovery_method_counts: dict[str, int]
    examples: tuple[str, ...]
    metadata_errors: tuple[str, ...]

    @property
    def has_integrity_errors(self) -> bool:
        return bool(self.metadata_errors or self.corrupt_files or self.missing_files or self.orphan_files or self.unexpected_entries)

    @property
    def has_retryable_records(self) -> bool:
        return self.pending_records > 0 or self.retryable_error_records > 0


@dataclass(frozen=True, slots=True)
class InspectionSummary:
    models: tuple[ModelInspection, ...]

    @property
    def exit_code(self) -> int:
        if any(model.has_integrity_errors for model in self.models):
            return 1
        return 2 if any(model.has_retryable_records for model in self.models) else 0


def inspect_model(
    root: str | Path, model: str | WebsterModel, *, example_count: int = 3
) -> ModelInspection:
    model_value = coerce_model(model)
    all_rows = WebsterDataset(root, model_value, recovered_only=False)
    available = WebsterDataset(root, model_value, recovered_only=True)
    metadata = tuple(all_rows.iter_metadata())
    statistics = all_rows.statistics()
    missing: set[str] = set()
    corrupt: set[str] = set()
    loadable = 0
    for index in range(len(available)):
        try:
            available[index]
            loadable += 1
        except WebsterImageError as error:
            (missing if error.kind == "missing" else corrupt).add(str(error.image_path))
    paths = WebsterPaths.from_root(root)
    directory = paths.model_images(model_value)
    referenced = {
        str(paths.base / str(row["image_path"]))
        for row in metadata
        if row.get("image_path") is not None
    }
    disk_files: set[str] = set()
    unexpected: set[str] = set()
    if directory.is_dir():
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix.casefold() == ".png":
                disk_files.add(str(entry))
            else:
                unexpected.add(str(entry))
    missing.update(referenced - disk_files)
    orphans = disk_files - referenced
    digest_records: dict[str, list[str]] = defaultdict(list)
    for row in metadata:
        if not _is_missing(row.get("sha256")):
            digest_records[str(row["sha256"])].append(str(row["record_id"]))
    duplicates = tuple(
        DuplicateDigest(digest, tuple(records))
        for digest, records in sorted(digest_records.items())
        if len(records) > 1
    )
    examples = tuple(
        f"{row['record_id']} original index {row['original_index']}: "
        f"{str(row.get('prompt_raw') or '')[:120]}" + ("..." if len(str(row.get("prompt_raw") or "")) > 120 else "")
        for row in metadata[:example_count]
    )
    status_counts = {str(key): int(value) for key, value in dict(statistics["by_recovery_status"]).items()}
    return ModelInspection(
        model_value.value,
        len(all_rows),
        len(available),
        len(disk_files),
        status_counts.get(RecoveryStatus.UNRESOLVED.value, 0),
        status_counts.get(RecoveryStatus.AMBIGUOUS.value, 0),
        status_counts.get(RecoveryStatus.INVALID_METADATA.value, 0),
        status_counts.get(RecoveryStatus.PENDING.value, 0),
        status_counts.get(RecoveryStatus.RETRYABLE_ERROR.value, 0),
        loadable,
        tuple(sorted(corrupt)),
        tuple(sorted(missing)),
        tuple(sorted(orphans)),
        tuple(sorted(unexpected)),
        duplicates,
        status_counts,
        {str(k): int(v) for k, v in dict(statistics["by_webster_category"]).items()},
        {str(k): int(v) for k, v in dict(statistics["by_recovery_method"]).items()},
        examples,
        (),
    )


def inspect_webster_dataset(
    root: str | Path,
    *,
    show_progress: bool = False,
    progress_postfix: Callable[[], str] | None = None,
) -> InspectionSummary:
    results: list[ModelInspection] = []
    models = tqdm(
        list(WebsterModel),
        total=len(WebsterModel),
        desc="[Webster 10/10] Validate model views",
        unit="model",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    if progress_postfix is not None:
        models.set_postfix_str(progress_postfix(), refresh=False)
    for model in models:
        try:
            results.append(inspect_model(root, model))
        except (WebsterDatasetError, OSError) as error:
            results.append(
                ModelInspection(
                    model.value, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    (), (), (), (), (), {}, {}, {}, (), (str(error),),
                )
            )
        if progress_postfix is not None:
            models.set_postfix_str(progress_postfix(), refresh=False)
    return InspectionSummary(tuple(results))


def format_inspection(summary: InspectionSummary) -> str:
    lines: list[str] = []
    for model in summary.models:
        lines.extend(
            [
                f"{model.model_name}:",
                f"  manifest rows: {model.manifest_rows}",
                f"  recovered image records: {model.recovered_image_records}",
                f"  unique image files: {model.unique_image_files}",
                f"  loadable images: {model.loadable_images}",
                f"  missing files: {len(model.missing_files)}",
                f"  corrupt images: {len(model.corrupt_files)}",
                f"  orphan files: {len(model.orphan_files)}",
                f"  metadata errors: {len(model.metadata_errors)}",
            ]
        )
    lines.append(f"exit code: {summary.exit_code}")
    return "\n".join(lines)


def coerce_model(model: str | WebsterModel) -> WebsterModel:
    if isinstance(model, WebsterModel):
        return model
    try:
        return WebsterModel(str(model))
    except ValueError as error:
        raise WebsterManifestError(f"unknown Webster model: {model!r}") from error


def _category_filter(values: str | None | Iterable[str | None]) -> frozenset[str] | None:
    if values is None:
        return None
    source: Iterable[str | None] = (values,) if isinstance(values, str) else values
    normalized = frozenset(normalize_webster_type(value) for value in source)
    if "UNKNOWN" in normalized and not any(_is_missing(value) or str(value).strip() in {"", "UNKNOWN"} for value in source):
        raise WebsterManifestError("unknown Webster category filter")
    return normalized


def _status_filter(
    values: str | RecoveryStatus | Iterable[str | RecoveryStatus] | None,
) -> frozenset[str] | None:
    if values is None:
        return None
    source: Iterable[str | RecoveryStatus] = (values,) if isinstance(values, (str, RecoveryStatus)) else values
    selected: set[str] = set()
    for value in source:
        try:
            selected.add(value.value if isinstance(value, RecoveryStatus) else RecoveryStatus(str(value)).value)
        except ValueError as error:
            raise WebsterManifestError(f"unknown recovery status filter: {value!r}") from error
    return frozenset(selected)


def _strict_bool(value: object, record_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if type(value).__name__ == "bool_":
        return bool(value)
    raise WebsterManifestError(f"{record_id} image_available must be true or false")


def _decode_auxiliary_paths(value: object) -> list[str]:
    if _is_missing(value):
        return []
    decoded = json.loads(value) if isinstance(value, str) else value
    if hasattr(decoded, "tolist"):
        decoded = decoded.tolist()
    if not isinstance(decoded, (list, tuple)):
        raise WebsterManifestError("auxiliary_asset_paths must contain a list")
    return [str(item) for item in decoded if not _is_missing(item)]


def _row_metadata(row: pd.Series) -> dict[str, object]:
    return {str(column): _python_value(value) for column, value in row.items()}


def _python_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(item) for item in value]
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _display(value: object, missing: str) -> str:
    return missing if _is_missing(value) else str(value)


def _values_equal(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "ACTIVE_STATUS_VALUES", "ALL_STATUS_VALUES", "ArchiveCandidate",
    "DatasetIntegrityError", "DownloadError", "DownloadExitCode", "DownloadSummary",
    "DuplicateDigest", "EXPECTED_ROWS_PER_MODEL", "EXPECTED_TOTAL_ROWS", "HTTPResult",
    "ImageMetadata", "ImageValidationError", "InspectionSummary", "ModelInspection",
    "ModelOrganizationSummary", "OrganizationError", "OrganizationSummary",
    "RECOVERED_STATUSES", "RECOVERED_STATUS_VALUES", "RecoveryStage", "RecoveryStatus",
    "RetryableStageError", "STAGE_ORDER", "StageOutcome", "StageResult",
    "StateConsistencyError", "StateSchemaError", "TERMINAL_STATUS_VALUES", "WarcPayload",
    "WebsterDataError", "WebsterDataset", "WebsterDatasetError", "WebsterImageError",
    "WebsterManifestError", "WebsterModel", "WebsterPaths", "acquire_and_build_manifest",
    "canonicalize_original_index", "coerce_model", "enrich_source_frame", "format_inspection",
    "inspect_model", "inspect_webster_dataset", "load_source_manifest",
    "normalize_url", "normalize_webster_type", "organize_model_views",
    "original_index_fields", "path_is_within", "resolve_schema", "validate_source_manifest",
]
