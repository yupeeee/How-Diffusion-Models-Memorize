"""Official repository metadata, explicit asset mapping, and cached recovery."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol

import pandas as pd
from tqdm import tqdm

from utils.common.io import atomic_write_json

from .images import (
    apply_image_resolution,
    publish_verified_blob,
    store_image,
    validate_image_bytes,
    verify_existing_blob,
)
from .state import RecoveryState
from .webster import (
    DownloadError,
    HTTPResult,
    ImageValidationError,
    RecoveryStatus,
    StageOutcome,
    StageResult,
    WebsterPaths,
    path_is_within,
)


HF_REPOSITORY = "fraisdufour/templates-verbs"
GITHUB_REPOSITORY = "ryanwebster90/onestep-extraction"
MAX_IMAGE_BYTES = 100 * 1024 * 1024
IMAGE_SUFFIXES = frozenset(
    {
        ".avif", ".bmp", ".gif", ".heic", ".heif", ".ico", ".jp2",
        ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
    }
)
TABLE_SUFFIXES = frozenset(
    {".csv", ".json", ".jsonl", ".ndjson", ".parquet", ".tsv"}
)


class Fetcher(Protocol):
    get: Callable[..., HTTPResult]


def _safe_repository_path(value: object) -> str | None:
    text = str(value or "").replace("\\", "/").strip("/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        return None
    return path.as_posix()


def infer_official_role(
    repo_path: str,
    column_name: str | None = None,
) -> str:
    """Classify an explicitly referenced official asset by its semantics."""

    text = f"{column_name or ''} {repo_path}".casefold()
    suffix = PurePosixPath(repo_path).suffix.casefold()
    if suffix in TABLE_SUFFIXES:
        return "metadata_table"
    if any(token in text for token in ("mask", "segmentation")):
        return "mask"
    if any(token in text for token in ("template", "template_indices")):
        return "template"
    if any(
        token in text
        for token in ("retrieved", "retrieval", "nearest", "candidate")
    ):
        return "retrieved_target"
    if suffix in IMAGE_SUFFIXES and any(
        token in text for token in ("source", "target", "groundtruth", "image")
    ):
        return "paired_source"
    return "auxiliary"


def role_directory(paths: WebsterPaths, role: str) -> Path:
    directories = {
        "paired_source": paths.auxiliary / "paired_sources",
        "retrieved_target": paths.auxiliary / "retrieved_targets",
        "template": paths.auxiliary / "templates",
        "mask": paths.auxiliary / "masks",
        "metadata_table": paths.auxiliary / "metadata_tables",
        "auxiliary": paths.auxiliary / "other",
    }
    try:
        return directories[role]
    except KeyError as error:
        raise ValueError(f"unknown official asset role: {role!r}") from error


def identifier_token(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")
    return text or None


def add_official_selection(
    selections: dict[tuple[str, str], dict[str, object]],
    *,
    repo_path: str,
    role: str,
    source: str,
    source_url: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Add one unambiguous explicit mapping and reject role conflicts."""

    safe_path = _safe_repository_path(repo_path)
    if safe_path is None:
        raise ValueError(f"unsafe official repository path: {repo_path!r}")
    key = (role, safe_path)
    value: dict[str, object] = {
        "repo_path": safe_path,
        "role": role,
        "mapping_source": source,
        "source_url": source_url,
        **dict(metadata or {}),
    }
    prior = selections.get(key)
    if prior is not None and prior != value:
        raise ValueError(
            f"conflicting official mapping for {role}/{safe_path}"
        )
    selections[key] = value


def inspect_official_tree(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
) -> list[dict[str, object]]:
    """Return a pinned official file catalog, reusing persisted local metadata."""

    cached = state.get_run_metadata("official_catalog")
    if isinstance(cached, list) and all(
        isinstance(item, dict) for item in cached
    ):
        return [dict(item) for item in cached]
    records = state.records()
    if not records:
        raise DownloadError(
            "official catalog lookup requires initialized recovery records"
        )
    audit_record_id = str(records[0]["record_id"])
    entries: list[dict[str, object]] = []
    catalog_errors: list[str] = []
    try:
        from huggingface_hub import HfApi

        revision = state.get_run_metadata("huggingface_revision")
        tree = HfApi().list_repo_tree(
            repo_id=HF_REPOSITORY,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=True,
        )
        for item in tree:
            repo_path = _safe_repository_path(getattr(item, "path", None))
            if repo_path is None or getattr(item, "type", "file") != "file":
                continue
            entries.append(
                {
                    "repository": "huggingface",
                    "repo_path": repo_path,
                    "revision": revision,
                    "size": getattr(item, "size", None),
                    "source_url": (
                        f"https://huggingface.co/datasets/{HF_REPOSITORY}"
                        f"/resolve/{revision}/{repo_path}"
                    ),
                }
            )
    except Exception as error:
        catalog_errors.append(
            f"Hugging Face: {type(error).__name__}: {error}"
        )
    try:
        result = fetcher.get(
            audit_record_id,
            "official_github_tree",
            (
                "https://api.github.com/repos/"
                f"{GITHUB_REPOSITORY}/git/trees/HEAD"
            ),
            params={"recursive": "1"},
            cache=True,
        )
        decoded = json.loads(result.data)
        for item in decoded.get("tree", []):
            if not isinstance(item, dict) or item.get("type") != "blob":
                continue
            repo_path = _safe_repository_path(item.get("path"))
            if repo_path is None:
                continue
            entries.append(
                {
                    "repository": "github",
                    "repo_path": repo_path,
                    "revision": decoded.get("sha"),
                    "size": item.get("size"),
                    "source_url": (
                        f"https://raw.githubusercontent.com/"
                        f"{GITHUB_REPOSITORY}/{decoded.get('sha')}/{repo_path}"
                    ),
                }
            )
    except (DownloadError, json.JSONDecodeError, TypeError, ValueError) as error:
        catalog_errors.append(
            f"GitHub: {type(error).__name__}: {error}"
        )
    if not entries:
        raise DownloadError(
            "could not enumerate either official repository: "
            + " | ".join(catalog_errors),
            transient=True,
        )
    entries.sort(
        key=lambda item: (
            str(item["repository"]),
            str(item["repo_path"]),
        )
    )
    state.set_run_metadata("official_catalog", entries)
    atomic_write_json(paths.state / "official_catalog.json", {"files": entries})
    return entries


def _flatten_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return _flatten_values(json.loads(stripped))
            except json.JSONDecodeError:
                return [value]
        return [value]
    if isinstance(value, Mapping):
        flattened: list[object] = []
        for item in value.values():
            flattened.extend(_flatten_values(item))
        return flattened
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        flattened = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return flattened
    if hasattr(value, "tolist"):
        return _flatten_values(value.tolist())
    return [value]


def discover_official_selections(
    record: Mapping[str, object],
    catalog: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Map only literal metadata references or unique identifier matches."""

    catalog_by_path = {
        str(item["repo_path"]): item for item in catalog
    }
    selections: dict[tuple[str, str], dict[str, object]] = {}
    metadata = record
    if isinstance(record.get("metadata_json"), str):
        try:
            decoded = json.loads(str(record["metadata_json"]))
            if isinstance(decoded, dict):
                metadata = {**decoded, **record}
        except json.JSONDecodeError:
            metadata = record
    for column, value in metadata.items():
        role_hint = infer_official_role("", str(column))
        for item in _flatten_values(value):
            candidate = _safe_repository_path(item)
            if candidate is None or candidate not in catalog_by_path:
                continue
            entry = catalog_by_path[candidate]
            add_official_selection(
                selections,
                repo_path=candidate,
                role=(
                    infer_official_role(candidate, str(column))
                    if role_hint == "auxiliary"
                    else role_hint
                ),
                source=f"metadata:{column}",
                source_url=str(entry.get("source_url") or "") or None,
                metadata={"repository": entry.get("repository")},
            )
    tokens = {
        token
        for token in (
            identifier_token(record.get("record_id")),
            identifier_token(record.get("laion_or_sample_index")),
        )
        if token is not None
    }
    for token in tokens:
        matches = []
        for item in catalog:
            path_token = identifier_token(item.get("repo_path"))
            if (
                path_token is not None
                and token in path_token.split("-")
                and PurePosixPath(
                    str(item["repo_path"])
                ).suffix.casefold() in IMAGE_SUFFIXES
            ):
                matches.append(item)
        if len(matches) == 1:
            entry = matches[0]
            repo_path = str(entry["repo_path"])
            add_official_selection(
                selections,
                repo_path=repo_path,
                role=infer_official_role(repo_path),
                source=f"unique_identifier:{token}",
                source_url=str(entry.get("source_url") or "") or None,
                metadata={"repository": entry.get("repository")},
            )
    return [
        selections[key]
        for key in sorted(selections)
    ]


def cache_catalog_asset(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
    record_id: str,
    selection: Mapping[str, object],
) -> dict[str, object]:
    source_url = str(selection.get("source_url") or "")
    if not source_url:
        raise DownloadError(
            f"official asset has no pinned source URL: {selection['repo_path']}"
        )
    result = fetcher.get(
        record_id,
        "official_asset",
        source_url,
        maximum_bytes=MAX_IMAGE_BYTES,
        cache=True,
    )
    digest = hashlib.sha256(result.data).hexdigest()
    role = str(selection["role"])
    suffix = PurePosixPath(str(selection["repo_path"])).suffix or ".bin"
    destination = role_directory(paths, role) / f"{digest}{suffix.casefold()}"
    publish_verified_blob(
        destination, result.data, paths.state / "quarantine"
    )
    if suffix.casefold() in IMAGE_SUFFIXES:
        validate_image_bytes(
            result.data,
            original_url=source_url,
            final_url=result.final_url,
        )
    asset = {
        **dict(selection),
        "local_path": str(destination),
        "sha256": digest,
        "response_url": result.final_url,
    }
    state.add_official_asset(record_id, asset)
    state.update_attempt(
        result.attempt_id,
        validation_result="valid_official_asset",
        candidate_sha256=digest,
    )
    return asset


def official_asset_artifacts_valid(
    paths: WebsterPaths,
    asset: Mapping[str, object],
) -> bool:
    try:
        local_path = Path(str(asset["local_path"]))
        role = str(asset["role"])
        if (
            not path_is_within(local_path, role_directory(paths, role))
            or not verify_existing_blob(local_path, str(asset["sha256"]))
        ):
            return False
        if local_path.suffix.casefold() in IMAGE_SUFFIXES:
            validate_image_bytes(local_path.read_bytes())
        return True
    except (
        ImageValidationError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def validated_official_assets(
    paths: WebsterPaths,
    state: RecoveryState,
    record_id: str,
) -> list[dict[str, object]]:
    valid: list[dict[str, object]] = []
    for asset in state.official_assets(record_id):
        if official_asset_artifacts_valid(paths, asset):
            valid.append(asset)
        else:
            state.delete_official_asset(
                record_id,
                str(asset["role"]),
                str(asset["repo_path"]),
            )
    return valid


def map_and_cache_official_assets(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
    manifest: pd.DataFrame,
    catalog: Sequence[Mapping[str, object]] | None = None,
    *,
    show_progress: bool = False,
    progress_postfix: Callable[[], str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    entries = (
        list(catalog)
        if catalog is not None
        else inspect_official_tree(paths, state, fetcher)
    )
    mapped: dict[str, list[dict[str, object]]] = {}
    records = manifest.to_dict(orient="records")
    iterator = tqdm(
        records,
        total=len(records),
        desc="[Webster 2/10] Official downloads",
        unit="record",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    if progress_postfix is not None:
        iterator.set_postfix_str(progress_postfix(), refresh=False)
    for record in iterator:
        record_id = str(record["record_id"])
        assets = validated_official_assets(paths, state, record_id)
        existing = {
            (str(asset["role"]), str(asset["repo_path"]))
            for asset in assets
        }
        for selection in discover_official_selections(record, entries):
            key = (
                str(selection["role"]),
                str(selection["repo_path"]),
            )
            if key not in existing:
                assets.append(
                    cache_catalog_asset(
                        paths, state, fetcher, record_id, selection
                    )
                )
        mapped[record_id] = assets
        state.update_record(
            record_id,
            auxiliary_asset_paths=[
                str(asset["local_path"])
                for asset in assets
                if str(asset["role"]) != "paired_source"
            ],
        )
        if progress_postfix is not None:
            iterator.set_postfix_str(progress_postfix(), refresh=False)
    return mapped


def recover_official(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
) -> StageResult:
    """Recover from exactly one explicitly mapped paired-source asset."""

    assets = [
        asset
        for asset in validated_official_assets(
            paths, state, str(record["record_id"])
        )
        if str(asset["role"]) == "paired_source"
    ]
    if not assets:
        return StageResult(
            StageOutcome.NOT_APPLICABLE,
            "no explicitly mapped official paired source",
        )
    hashes = {str(asset["sha256"]) for asset in assets}
    if len(hashes) != 1:
        state.set_status(
            str(record["record_id"]),
            RecoveryStatus.AMBIGUOUS,
            reason="official paired-source mappings disagree",
        )
        return StageResult(
            StageOutcome.AMBIGUOUS,
            "official paired-source mappings disagree",
        )
    chosen = assets[0]
    data = Path(str(chosen["local_path"])).read_bytes()
    metadata, raw_path, normalized_path = store_image(
        paths,
        state,
        data,
        original_url=str(record.get("original_url") or ""),
        final_url=str(chosen.get("source_url") or ""),
    )
    apply_image_resolution(
        paths,
        state,
        record,
        metadata,
        raw_path,
        normalized_path,
        status=RecoveryStatus.RECOVERED_OFFICIAL_CACHE,
        method="official_explicit_mapping",
        resolved_url=str(chosen.get("source_url") or chosen["repo_path"]),
    )
    return StageResult(
        StageOutcome.RECOVERED,
        f"recovered official asset {chosen['repo_path']}",
    )


__all__ = [
    "add_official_selection",
    "cache_catalog_asset",
    "discover_official_selections",
    "infer_official_role",
    "inspect_official_tree",
    "map_and_cache_official_assets",
    "official_asset_artifacts_valid",
    "recover_official",
    "role_directory",
    "validated_official_assets",
]
