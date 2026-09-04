"""Exact-URL image recovery from the Portuguese Arquivo.pt archive."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from collections.abc import Mapping
from typing import Callable, Protocol

from .images import preserve_archive_candidate
from .state import RecoveryState
from .wayback import (
    archive_sha1,
    encoded_path_variant,
    is_valid_http_url,
    normalize_url,
    safe_url_variants,
)
from .webster import (
    ArchiveCandidate,
    DownloadError,
    HTTPResult,
    ImageValidationError,
    StageOutcome,
    StageResult,
    WebsterPaths,
)

ARQUIVO_SEARCH_URL = "https://arquivo.pt/textsearch"
ARQUIVO_STRATEGY_VERSION = 2
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_METADATA_BYTES = 32 * 1024 * 1024
ARQUIVO_PAGE_SIZE = 50
MAX_ARQUIVO_PAGES = 4
MAX_ARQUIVO_REPLAYS = 12


class Fetcher(Protocol):
    state: RecoveryState
    get: Callable[..., HTTPResult]


def arquivo_digest_options(value: object) -> tuple[str, ...]:
    """Return every supported algorithm-qualified interpretation."""

    if value is None:
        return ()
    text = str(value).strip().casefold()
    if text in {"", "-"}:
        return ()
    if text.startswith("md5:"):
        key = text.removeprefix("md5:")
        return (f"md5:{key}",) if re.fullmatch(r"[0-9a-f]{32}", key) else ()
    if text.startswith("sha1:"):
        key = text.removeprefix("sha1:")
        return (f"sha1:{key}",) if re.fullmatch(r"[a-z2-7]{32}", key) else ()
    options: list[str] = []
    if re.fullmatch(r"[0-9a-f]{32}", text):
        options.append(f"md5:{text}")
    if re.fullmatch(r"[a-z2-7]{32}", text):
        options.append(f"sha1:{text}")
    return tuple(options)


def verify_arquivo_digest(
    value: object,
    data: bytes,
) -> str | None:
    """Verify SHA-1 or MD5 digests reported by Arquivo.pt."""

    md5_hex: str | None = None
    sha1_base32: str | None = None
    for option in arquivo_digest_options(value):
        algorithm, key = option.split(":", 1)
        if algorithm == "md5":
            if md5_hex is None:
                md5_hex = hashlib.md5(data, usedforsecurity=False).hexdigest()
            if md5_hex == key:
                return option
        elif algorithm == "sha1":
            if sha1_base32 is None:
                sha1_base32 = archive_sha1(data)
            if sha1_base32 == key:
                return option
    return None


def parse_arquivo_response(
    data: bytes | str,
) -> tuple[list[dict[str, str]], int | None, int, int]:
    """Parse one Arquivo.pt version-history response."""

    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Arquivo.pt returned invalid JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError("Arquivo.pt response is not an object")
    raw_items = decoded.get("response_items")
    if not isinstance(raw_items, list):
        raise ValueError("Arquivo.pt response_items is not an array")

    estimated_value = decoded.get("estimated_nr_results")
    estimated: int | None
    if estimated_value is None:
        estimated = None
    else:
        try:
            estimated = int(estimated_value)
        except (TypeError, ValueError) as error:
            raise ValueError("Arquivo.pt estimated_nr_results is invalid") from error
        if estimated < 0:
            raise ValueError("Arquivo.pt estimated_nr_results is negative")

    captures: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    malformed_digests = 0
    for position, value in enumerate(raw_items):
        if not isinstance(value, dict):
            raise ValueError(f"Arquivo.pt response item {position} is not an object")
        status = str(value.get("statusCode") or "")
        if status != "200":
            continue
        original = str(value.get("originalURL") or "").strip()
        timestamp = str(value.get("tstamp") or "").strip()
        if not is_valid_http_url(original):
            raise ValueError(f"Arquivo.pt response item {position} has an invalid URL")
        if re.fullmatch(r"\d{14}", timestamp) is None:
            raise ValueError(
                f"Arquivo.pt response item {position} has an invalid timestamp"
            )
        reported_digest = str(value.get("digest") or "").strip()
        if reported_digest not in {"", "-"} and not arquivo_digest_options(
            reported_digest
        ):
            malformed_digests += 1
            continue
        replay_url = str(value.get("linkToOriginalFile") or "").strip()
        capture = {
            "original": original,
            "timestamp": timestamp,
            "digest": ("" if reported_digest == "-" else reported_digest),
            "reported_digest": reported_digest,
            "mimetype": str(value.get("mimeType") or "").strip(),
            "replay_url": replay_url,
            "collection": str(value.get("collection") or "").strip(),
            "filename": str(value.get("fileName") or "").strip(),
            "offset": str(
                "" if value.get("offset") is None else value.get("offset")
            ).strip(),
            "content_length": str(
                "" if value.get("contentLength") is None else value.get("contentLength")
            ).strip(),
        }
        identity = (
            original,
            timestamp,
            reported_digest,
            replay_url,
        )
        if identity not in seen:
            captures.append(capture)
            seen.add(identity)
    return (
        captures,
        estimated,
        len(raw_items),
        malformed_digests,
    )


def arquivo_history_offsets(
    estimated: int | None,
    *,
    page_size: int = ARQUIVO_PAGE_SIZE,
    maximum_pages: int = MAX_ARQUIVO_PAGES,
) -> list[int]:
    """Sample metadata pages across the complete newest-to-oldest history."""

    if page_size <= 0 or maximum_pages <= 0:
        raise ValueError("page_size and maximum_pages must be positive")
    if estimated is None:
        return [page_size * index for index in range(maximum_pages)]
    if estimated <= page_size:
        return [0]
    last_offset = max(0, estimated - page_size)
    if maximum_pages == 1:
        return [0]
    return sorted(
        {
            round(position * last_offset / (maximum_pages - 1))
            for position in range(maximum_pages)
        }
    )


def _arquivo_page(
    fetcher: Fetcher,
    record_id: str,
    target_url: str,
    offset: int,
    page_number: int,
) -> tuple[list[dict[str, str]], int | None, int]:
    try:
        result = fetcher.get(
            record_id,
            "arquivo_version_history",
            ARQUIVO_SEARCH_URL,
            params={
                "versionHistory": target_url,
                "maxItems": ARQUIVO_PAGE_SIZE,
                "offset": offset,
            },
            maximum_bytes=MAX_METADATA_BYTES,
            cache=True,
            archive_collection="arquivo.pt",
            attempts=1,
            follow_redirects=False,
        )
    except DownloadError as error:
        if error.status_code == 404 and page_number == 1:
            return [], 0, 0
        raise
    try:
        page, estimated, returned, malformed = parse_arquivo_response(result.data)
    except ValueError as error:
        fetcher.state.invalidate_http_cache(result.request_key)
        fetcher.state.update_attempt(
            result.attempt_id,
            validation_result="invalid_arquivo_response",
            exception=f"{type(error).__name__}: {error}",
        )
        raise DownloadError(
            f"invalid Arquivo.pt response: {error}",
            transient=True,
        ) from error
    fetcher.state.update_attempt(
        result.attempt_id,
        validation_result=(
            f"arquivo_page:{page_number}:offset:{offset}:"
            f"returned:{returned}:malformed_digest:{malformed}"
        ),
    )
    return page, estimated, returned


def arquivo_captures(
    fetcher: Fetcher,
    record_id: str,
    target_url: str,
) -> list[dict[str, str]]:
    """Query cached pages sampled across one exact URL's full history."""

    expected = normalize_url(target_url)
    if expected is None:
        return []
    first, estimated, returned = _arquivo_page(fetcher, record_id, target_url, 0, 1)
    exact_first = [
        capture for capture in first if normalize_url(capture["original"]) == expected
    ]
    if not exact_first or returned == 0:
        return []

    captures: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def extend(
        values: list[dict[str, str]],
        history_offset: int,
    ) -> None:
        for supplied in values:
            if normalize_url(supplied["original"]) != expected:
                continue
            capture = {
                **supplied,
                "queried_url": target_url,
                "history_offset": str(history_offset),
            }
            identity = (
                capture["original"],
                capture["timestamp"],
                capture["reported_digest"],
                capture["replay_url"],
            )
            if identity not in seen:
                captures.append(capture)
                seen.add(identity)

    extend(first, 0)
    offsets = arquivo_history_offsets(estimated)
    if estimated is None and returned < ARQUIVO_PAGE_SIZE:
        offsets = [0]
    for page_number, offset in enumerate(offsets[1:], 2):
        page, _, page_returned = _arquivo_page(
            fetcher,
            record_id,
            target_url,
            offset,
            page_number,
        )
        extend(page, offset)
        if estimated is None and page_returned < ARQUIVO_PAGE_SIZE:
            break
    return captures


def build_arquivo_replay_url(
    timestamp: str,
    original_url: str,
) -> str:
    if re.fullmatch(r"\d{14}", timestamp) is None:
        raise ValueError(f"invalid Arquivo.pt timestamp: {timestamp!r}")
    archived = encoded_path_variant(urllib.parse.urldefrag(original_url).url)
    return f"https://arquivo.pt/noFrame/replay/" f"{timestamp}id_/{archived}"


def arquivo_replay_matches(
    final_url: str,
    timestamp: str,
    original_url: str,
) -> bool:
    """Require an exact HTTPS Arquivo.pt raw replay identity."""

    try:
        parsed = urllib.parse.urlsplit(final_url)
        canonical_origin = (
            parsed.scheme.casefold() == "https"
            and (parsed.hostname or "").casefold() == "arquivo.pt"
            and parsed.port in {None, 443}
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
        )
    except (UnicodeError, ValueError):
        return False
    if not canonical_origin:
        return False
    match = re.fullmatch(
        r"/noFrame/replay/(\d{14})(?:id_/|/id_/)(.*)",
        parsed.path,
        re.IGNORECASE,
    )
    if match is None or match.group(1) != timestamp:
        return False
    embedded = match.group(2) + ("?" + parsed.query if parsed.query else "")
    expected = normalize_url(original_url)
    if expected is None:
        return False
    raw = normalize_url(embedded)
    if raw == expected:
        return True
    return normalize_url(urllib.parse.unquote(embedded)) == expected


def _evenly_spaced(
    captures: list[dict[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    ordered = sorted(
        captures,
        key=lambda capture: capture["timestamp"],
        reverse=True,
    )
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return ordered[:1]
    indexes = {
        round(position * (len(ordered) - 1) / (limit - 1)) for position in range(limit)
    }
    return [ordered[index] for index in sorted(indexes)]


def select_arquivo_captures(
    captures: list[dict[str, str]],
    *,
    limit: int = MAX_ARQUIVO_REPLAYS,
) -> list[dict[str, str]]:
    """Apply one replay budget after target-wide capture deduplication."""

    if limit <= 0:
        return []
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for capture in captures:
        identity = (
            capture["original"],
            capture["timestamp"],
            capture["reported_digest"],
            capture["replay_url"],
        )
        if identity not in seen:
            unique.append(capture)
            seen.add(identity)
    images = [
        capture
        for capture in unique
        if capture["mimetype"].casefold().startswith("image/")
    ]
    image_ids = {id(capture) for capture in images}
    other = [capture for capture in unique if id(capture) not in image_ids]
    if not images:
        return _evenly_spaced(other, limit)
    image_limit = limit if not other else max(1, limit - 2)
    selected = _evenly_spaced(images, image_limit)
    selected.extend(_evenly_spaced(other, limit - len(selected)))
    return selected[:limit]


def _validated_arquivo_replay(
    fetcher: Fetcher,
    record_id: str,
    capture: Mapping[str, str],
) -> tuple[HTTPResult, str] | None:
    timestamp = capture["timestamp"]
    original = capture["original"]
    supplied = str(capture.get("replay_url") or "")
    requested = (
        supplied
        if arquivo_replay_matches(supplied, timestamp, original)
        else build_arquivo_replay_url(timestamp, original)
    )
    try:
        result = fetcher.get(
            record_id,
            "arquivo_replay",
            requested,
            headers={"Accept-Encoding": "identity"},
            maximum_bytes=MAX_IMAGE_BYTES,
            cache=True,
            archive_collection=(str(capture.get("collection") or "") or "arquivo.pt"),
            archive_timestamp=timestamp,
            raw_stream=True,
            attempts=1,
            follow_redirects=False,
        )
    except DownloadError as error:
        if error.transient:
            raise
        return None
    if arquivo_replay_matches(result.final_url, timestamp, original):
        return result, requested
    fetcher.state.update_attempt(
        result.attempt_id,
        validation_result="rejected_arquivo_semantic_redirect",
        exception=("replay final URL changed timestamp or original URL"),
    )
    fetcher.state.invalidate_http_cache(result.request_key)
    if result.from_cache:
        raise DownloadError(
            "cached Arquivo.pt replay has the wrong identity",
            transient=True,
        )
    return None


def collect_arquivo_candidates(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
    record: Mapping[str, object],
    captures: list[dict[str, str]],
) -> list[ArchiveCandidate]:
    """Replay one globally bounded, target-deduplicated capture set."""

    record_id = str(record["record_id"])
    recovered: list[ArchiveCandidate] = []
    for capture in select_arquivo_captures(captures):
        replay = _validated_arquivo_replay(fetcher, record_id, capture)
        if replay is None:
            continue
        result, requested_replay_url = replay
        reported_digest = str(capture.get("digest") or "")
        verified_digest = verify_arquivo_digest(reported_digest, result.data)
        if reported_digest and verified_digest is None:
            message = (
                f"Arquivo.pt digest {reported_digest!r} differs from "
                f"replay md5:{hashlib.md5(result.data, usedforsecurity=False).hexdigest()} "
                f"and sha1:{archive_sha1(result.data)}"
            )
            state.update_attempt(
                result.attempt_id,
                validation_result="arquivo_archive_digest_mismatch",
                exception=message,
            )
            state.invalidate_http_cache(result.request_key)
            if result.from_cache:
                raise DownloadError(
                    "cached Arquivo.pt replay failed its digest",
                    transient=True,
                )
            continue
        if verified_digest is not None:
            state.update_attempt(
                result.attempt_id,
                validation_result=(
                    "arquivo_payload_digest_verified:"
                    + verified_digest.split(":", 1)[0]
                ),
            )
        try:
            recovered.append(
                preserve_archive_candidate(
                    paths,
                    state,
                    record,
                    result,
                    strategy="arquivo_exact_image",
                    resolved_url=capture["original"],
                    archive_timestamp_value=capture["timestamp"],
                    archive_digest=verified_digest,
                    verified_archive_digest=verified_digest,
                    source_page=False,
                    provenance={
                        "capture": dict(capture),
                        "queried_url": capture.get("queried_url"),
                        "supplied_replay_url": capture.get("replay_url"),
                        "requested_replay_url": requested_replay_url,
                        "replay_final_url": result.final_url,
                        "replay_request_key": result.request_key,
                        "arquivo_reported_digest": reported_digest,
                        "verified_digest_algorithm": (
                            verified_digest.split(":", 1)[0]
                            if verified_digest
                            else None
                        ),
                    },
                )
            )
        except ImageValidationError as error:
            state.update_attempt(
                result.attempt_id,
                validation_result="invalid_image",
                exception=str(error),
            )
    return recovered


def recover_arquivo(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
    record: Mapping[str, object],
) -> StageResult:
    variants = safe_url_variants(record.get("original_url"))
    if not variants:
        return StageResult(
            StageOutcome.NOT_APPLICABLE,
            "no exact image URL for Arquivo.pt",
        )
    captures: list[dict[str, str]] = []
    visited: set[str] = set()
    for variant in variants:
        normalized = normalize_url(variant)
        if normalized is None or normalized in visited:
            continue
        visited.add(normalized)
        captures = arquivo_captures(
            fetcher,
            str(record["record_id"]),
            variant,
        )
        if captures:
            break
    candidates = collect_arquivo_candidates(paths, state, fetcher, record, captures)
    if candidates:
        return StageResult(
            StageOutcome.RECOVERED,
            f"collected {len(candidates)} exact Arquivo.pt candidates",
        )
    return StageResult(
        StageOutcome.MISS,
        "all exact Arquivo.pt captures were exhausted",
    )


__all__ = [
    "ARQUIVO_PAGE_SIZE",
    "ARQUIVO_SEARCH_URL",
    "ARQUIVO_STRATEGY_VERSION",
    "MAX_ARQUIVO_PAGES",
    "MAX_ARQUIVO_REPLAYS",
    "arquivo_captures",
    "arquivo_digest_options",
    "arquivo_history_offsets",
    "arquivo_replay_matches",
    "build_arquivo_replay_url",
    "collect_arquivo_candidates",
    "parse_arquivo_response",
    "recover_arquivo",
    "select_arquivo_captures",
    "verify_arquivo_digest",
]
