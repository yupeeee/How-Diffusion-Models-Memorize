"""Common Crawl capture traversal, WARC recovery, and revisit handling."""

from __future__ import annotations

import io
import json
import re
import urllib.parse
import zlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol

from warcio.archiveiterator import ArchiveIterator
from warcio.exceptions import ArchiveLoadFailed

from .images import (
    candidate_artifacts_valid,
    create_reference,
    preserve_archive_candidate,
    verify_existing_blob,
)
from .state import RecoveryState
from .wayback import (
    archive_digest_key,
    archive_sha1,
    extract_explicit_image_urls,
    is_valid_http_url,
    looks_like_html,
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
    WarcPayload,
    WebsterPaths,
)


MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_METADATA_BYTES = 32 * 1024 * 1024


class Fetcher(Protocol):
    state: RecoveryState
    get: Callable[..., HTTPResult]


def archive_timestamp(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{14}", text):
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def parse_commoncrawl_indexes(data: bytes | str) -> list[dict[str, str]]:
    text = (
        data.decode("utf-8", errors="replace")
        if isinstance(data, bytes)
        else data
    )
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Common Crawl collection list is invalid JSON: {error}"
        ) from error
    if not isinstance(decoded, list):
        raise ValueError("Common Crawl collection list is not an array")
    indexes: list[dict[str, str]] = []
    seen: set[str] = set()
    for position, value in enumerate(decoded):
        if not isinstance(value, dict):
            raise ValueError(
                f"collection entry {position} is not an object"
            )
        identifier = str(value.get("id") or "")
        endpoint = str(
            value.get("cdx-api")
            or (
                f"https://index.commoncrawl.org/{identifier}-index"
                if identifier
                else ""
            )
        )
        if not identifier or not is_valid_http_url(endpoint):
            raise ValueError(f"collection entry {position} is malformed")
        if identifier in seen:
            raise ValueError(
                f"duplicated collection identifier: {identifier}"
            )
        indexes.append({"id": identifier, "cdx_api": endpoint})
        seen.add(identifier)
    return indexes


def parse_commoncrawl_records(
    data: bytes | str,
) -> list[dict[str, object]]:
    text = (
        data.decode("utf-8", errors="replace")
        if isinstance(data, bytes)
        else data
    )
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"CDX line {line_number} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"CDX line {line_number} is not an object"
            )
        if "error" in value or (
            "message" in value and "filename" not in value
        ):
            raise ValueError(f"CDX error response: {value}")
        status = str(value.get("status") or value.get("statuscode") or "")
        if not status:
            raise ValueError(f"CDX line {line_number} has no status")
        if status != "200":
            continue
        if not is_valid_http_url(value.get("url")):
            raise ValueError(
                f"CDX line {line_number} has an invalid URL"
            )
        if archive_timestamp(value.get("timestamp")) is None:
            raise ValueError(
                f"CDX line {line_number} has an invalid timestamp"
            )
        digest = str(value.get("digest") or "").strip()
        if digest not in {"", "-"} and archive_digest_key(digest) is None:
            raise ValueError(
                f"CDX line {line_number} has an invalid digest"
            )
        try:
            offset = int(value["offset"])
            length = int(value["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"CDX line {line_number} has invalid range metadata"
            ) from error
        filename = str(value.get("filename") or "")
        safe_filename = (
            filename.startswith("crawl-data/")
            and ".." not in Path(filename).parts
            and "://" not in filename
        )
        if offset < 0 or length <= 0 or not safe_filename:
            raise ValueError(
                f"CDX line {line_number} has an unsafe WARC range"
            )
        item: dict[str, object] = dict(value)
        item.update(
            {
                "offset": offset,
                "length": length,
                "filename": filename,
                "range_too_large": (
                    length > MAX_IMAGE_BYTES + 1024 * 1024
                ),
            }
        )
        records.append(item)
    return records


def commoncrawl_indexes(
    fetcher: Fetcher,
    state: RecoveryState,
    record_id: str,
) -> list[dict[str, str]]:
    result = fetcher.get(
        record_id,
        "commoncrawl_index_list",
        "https://index.commoncrawl.org/collinfo.json",
        maximum_bytes=MAX_METADATA_BYTES,
        cache=True,
        archive_collection="commoncrawl",
        attempts=1,
    )
    try:
        indexes = parse_commoncrawl_indexes(result.data)
    except ValueError as error:
        state.invalidate_http_cache(result.request_key)
        raise DownloadError(
            f"invalid Common Crawl collection list: {error}",
            transient=True,
        ) from error
    if not indexes:
        state.invalidate_http_cache(result.request_key)
        raise DownloadError(
            "Common Crawl returned no collection indexes",
            transient=True,
        )
    state.update_attempt(
        result.attempt_id,
        validation_result=f"commoncrawl_indexes:{len(indexes)}",
    )
    return indexes


def _page_count(
    fetcher: Fetcher,
    record_id: str,
    index: Mapping[str, str],
    parameters: Mapping[str, object],
) -> int | None:
    try:
        result = fetcher.get(
            record_id,
            "commoncrawl_cdx_pages",
            index["cdx_api"],
            params={**parameters, "showNumPages": "true"},
            maximum_bytes=MAX_METADATA_BYTES,
            cache=True,
            archive_collection=index["id"],
            attempts=1,
        )
    except DownloadError as error:
        if error.status_code == 404:
            return None
        raise
    try:
        metadata = json.loads(
            result.data.decode("utf-8", errors="replace")
        )
        if not isinstance(metadata, dict) or "pages" not in metadata:
            raise ValueError("page-count response lacks pages")
        count = int(metadata["pages"])
        if count < 0:
            raise ValueError("page count is negative")
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        fetcher.state.invalidate_http_cache(result.request_key)
        fetcher.state.update_attempt(
            result.attempt_id,
            validation_result="invalid_commoncrawl_page_count",
            exception=f"{type(error).__name__}: {error}",
        )
        raise DownloadError(
            f"invalid page count for {index['id']}: {error}",
            transient=True,
        ) from error
    fetcher.state.update_attempt(
        result.attempt_id,
        validation_result=f"commoncrawl_pages:{count}",
    )
    return count


def query_commoncrawl_url(
    fetcher: Fetcher,
    record_id: str,
    target_url: str,
    indexes: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for index in indexes:
        base_parameters: dict[str, object] = {
            "url": target_url,
            "output": "json",
            "matchType": "exact",
            "filter": "status:200",
        }
        count = _page_count(
            fetcher, record_id, index, base_parameters
        )
        if count is None:
            continue
        for page_number in range(count):
            parameters = dict(base_parameters)
            if count > 1:
                parameters["page"] = page_number
            result = fetcher.get(
                record_id,
                "commoncrawl_cdx",
                index["cdx_api"],
                params=parameters,
                maximum_bytes=MAX_METADATA_BYTES,
                cache=True,
                archive_collection=index["id"],
                attempts=1,
            )
            try:
                captures = parse_commoncrawl_records(result.data)
            except ValueError as error:
                fetcher.state.invalidate_http_cache(result.request_key)
                raise DownloadError(
                    f"invalid Common Crawl CDX response: {error}",
                    transient=True,
                ) from error
            fetcher.state.update_attempt(
                result.attempt_id,
                validation_result=(
                    f"commoncrawl_records:{len(captures)}"
                ),
            )
            for capture in captures:
                identity = (
                    str(capture["filename"]),
                    int(capture["offset"]),
                    int(capture["length"]),
                )
                if identity in seen:
                    continue
                enriched = dict(capture)
                enriched["collection"] = index["id"]
                enriched["queried_url"] = target_url
                records.append(enriched)
                seen.add(identity)
    return records


def extract_warc_payloads(segment: bytes) -> list[WarcPayload]:
    """Parse raw and decoded WARC phases and retain digest provenance."""

    computed_digests: list[str | None] = []
    extracted: list[WarcPayload] = []
    try:
        for raw_record in ArchiveIterator(
            io.BytesIO(segment), arc2warc=True
        ):
            computed = None
            if (
                str(raw_record.rec_type or "") == "response"
                and raw_record.http_headers is not None
                and str(
                    raw_record.http_headers.get_statuscode() or ""
                ) == "200"
            ):
                raw_body = raw_record.raw_stream.read(
                    MAX_IMAGE_BYTES + 1
                )
                if len(raw_body) <= MAX_IMAGE_BYTES:
                    computed = archive_sha1(raw_body)
            computed_digests.append(computed)
        for record_index, record in enumerate(
            ArchiveIterator(io.BytesIO(segment), arc2warc=True)
        ):
            if record_index >= len(computed_digests):
                raise ValueError(
                    "decoded WARC phase produced additional records"
                )
            record_type = str(record.rec_type or "")
            http_status: str | None = None
            content_type: str | None = None
            decoded_body: bytes | None = None
            body_too_large = False
            if (
                record_type == "response"
                and record.http_headers is not None
            ):
                http_status = str(
                    record.http_headers.get_statuscode() or ""
                )
                content_type = record.http_headers.get_header(
                    "Content-Type"
                )
                if http_status == "200":
                    body = record.content_stream().read(
                        MAX_IMAGE_BYTES + 1
                    )
                    if len(body) <= MAX_IMAGE_BYTES:
                        decoded_body = body
                    else:
                        body_too_large = True
            reported = record.rec_headers.get_header(
                "WARC-Payload-Digest"
            )
            computed = computed_digests[record_index]
            reported_key = archive_digest_key(reported)
            extracted.append(
                WarcPayload(
                    record_type=record_type,
                    http_status=http_status,
                    decoded_body=decoded_body,
                    body_too_large=body_too_large,
                    content_type=content_type,
                    target_uri=record.rec_headers.get_header(
                        "WARC-Target-URI"
                    ),
                    reported_payload_digest=reported,
                    computed_payload_digest=computed,
                    payload_digest_verified=(
                        reported_key is not None
                        and reported_key == computed
                    ),
                    warc_record_id=record.rec_headers.get_header(
                        "WARC-Record-ID"
                    ),
                    warc_date=record.rec_headers.get_header("WARC-Date"),
                    refers_to=record.rec_headers.get_header(
                        "WARC-Refers-To"
                    ),
                    refers_to_target_uri=record.rec_headers.get_header(
                        "WARC-Refers-To-Target-URI"
                    ),
                    refers_to_date=record.rec_headers.get_header(
                        "WARC-Refers-To-Date"
                    ),
                )
            )
    except (
        ArchiveLoadFailed,
        EOFError,
        OSError,
        TypeError,
        ValueError,
        zlib.error,
    ) as error:
        raise ValueError(
            f"WARC parsing failed: {type(error).__name__}: {error}"
        ) from error
    if len(extracted) != len(computed_digests):
        raise ValueError(
            "raw and decoded WARC phases produced different record counts"
        )
    return extracted


def reject_warc_semantics(
    state: RecoveryState,
    result: HTTPResult,
    validation_result: str,
    message: str,
) -> None:
    state.update_attempt(
        result.attempt_id,
        validation_result=validation_result,
        exception=message,
    )
    state.invalidate_http_cache(result.request_key)
    if result.from_cache:
        raise DownloadError(
            f"cached WARC range failed {validation_result}: {message}",
            transient=True,
        )


def _canonical_commoncrawl_object(
    final_url: str, filename: str
) -> bool:
    try:
        parsed = urllib.parse.urlsplit(final_url)
        return (
            parsed.scheme.casefold() == "https"
            and (parsed.hostname or "").casefold()
            == "data.commoncrawl.org"
            and parsed.port in {None, 443}
            and parsed.username is None
            and parsed.password is None
            and parsed.path == "/" + filename
            and not parsed.query
            and not parsed.fragment
        )
    except (UnicodeError, ValueError):
        return False


def fetch_commoncrawl_warc(
    fetcher: Fetcher,
    record_id: str,
    capture: Mapping[str, object],
) -> tuple[HTTPResult, list[WarcPayload]] | None:
    filename = str(capture["filename"])
    offset = int(capture["offset"])
    length = int(capture["length"])
    end = offset + length - 1
    try:
        result = fetcher.get(
            record_id,
            "commoncrawl_warc_range",
            "https://data.commoncrawl.org/" + filename,
            headers={
                "Range": f"bytes={offset}-{end}",
                "Accept-Encoding": "identity",
            },
            maximum_bytes=length,
            cache=True,
            archive_collection=str(capture.get("collection") or ""),
            archive_timestamp=str(capture.get("timestamp") or ""),
            warc_filename=filename,
            warc_offset=offset,
            warc_length=length,
            raw_stream=True,
            required_status=206,
            required_length=length,
        )
    except DownloadError as error:
        if error.status_code in {404, 416}:
            return None
        raise
    if not _canonical_commoncrawl_object(result.final_url, filename):
        fetcher.state.update_attempt(
            result.attempt_id,
            validation_result="rejected_commoncrawl_redirect",
            exception=(
                "range response did not end at the canonical WARC object"
            ),
        )
        fetcher.state.invalidate_http_cache(result.request_key)
        if result.from_cache:
            raise DownloadError(
                "cached WARC object identity is invalid",
                transient=True,
            )
        return None
    try:
        payloads = extract_warc_payloads(result.data)
    except ValueError as error:
        fetcher.state.update_attempt(
            result.attempt_id,
            validation_result="warc_parse_error",
            exception=str(error),
        )
        fetcher.state.invalidate_http_cache(result.request_key)
        raise DownloadError(str(error), transient=True) from error
    fetcher.state.update_attempt(
        result.attempt_id,
        validation_result=f"warc_records:{len(payloads)}",
    )
    return result, payloads


def _required_digest(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold().removeprefix("sha1:")
    return archive_digest_key(normalized)


def _payload_identity_is_valid(
    state: RecoveryState,
    result: HTTPResult,
    capture: Mapping[str, object],
    payload: WarcPayload,
    *,
    required_digest: str | None,
    required_timestamp: str | None,
    required_record_id: str | None,
) -> tuple[bool, str | None]:
    expected_urls = {
        normalize_url(capture.get("url")),
        normalize_url(capture.get("queried_url")),
    }
    expected_urls.discard(None)
    payload_url = normalize_url(payload.target_uri)
    if payload_url is None or not expected_urls or payload_url not in expected_urls:
        reject_warc_semantics(
            state,
            result,
            "warc_target_mismatch",
            f"WARC target {payload.target_uri!r} does not match capture identity",
        )
        return False, None
    capture_time = archive_timestamp(capture.get("timestamp"))
    payload_time = archive_timestamp(payload.warc_date)
    if capture_time is None or payload_time != capture_time:
        reject_warc_semantics(
            state,
            result,
            "warc_capture_date_mismatch",
            f"WARC date {payload.warc_date!r} does not match CDX timestamp",
        )
        return False, None
    warc_digest = archive_digest_key(payload.reported_payload_digest)
    cdx_digest = archive_digest_key(capture.get("digest"))
    if warc_digest is not None and cdx_digest is not None and warc_digest != cdx_digest:
        reject_warc_semantics(
            state,
            result,
            "warc_reported_digest_conflict",
            "WARC and CDX reported different payload digests",
        )
        return False, None
    reported_digest = warc_digest or cdx_digest
    if required_digest is not None and reported_digest != required_digest:
        reject_warc_semantics(
            state,
            result,
            "revisit_donor_digest_identity_mismatch",
            "reported digest does not match the revisit constraint",
        )
        return False, None
    if required_record_id is not None and payload.warc_record_id != required_record_id:
        reject_warc_semantics(
            state,
            result,
            "revisit_donor_record_mismatch",
            "WARC record ID does not match the revisit constraint",
        )
        return False, None
    if required_timestamp is not None and payload_time != required_timestamp:
        reject_warc_semantics(
            state,
            result,
            "revisit_donor_date_mismatch",
            "WARC date does not match the revisit constraint",
        )
        return False, None
    return True, reported_digest


def _validated_payload_digest(
    state: RecoveryState,
    result: HTTPResult,
    payload: WarcPayload,
    reported_digest: str | None,
    required_digest: str | None,
) -> str | None:
    if reported_digest is None:
        if required_digest is None:
            return None
        reject_warc_semantics(
            state,
            result,
            "revisit_donor_digest_missing",
            "revisit donor did not report a payload digest",
        )
        return None
    if payload.computed_payload_digest != reported_digest:
        reject_warc_semantics(
            state,
            result,
            "archive_digest_mismatch",
            "reported digest differs from the raw WARC payload digest",
        )
        return None
    if required_digest is not None and reported_digest != required_digest:
        reject_warc_semantics(
            state,
            result,
            "revisit_donor_digest_mismatch",
            "verified payload digest does not match the revisit constraint",
        )
        return None
    return reported_digest


def _persist_payload_candidate(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    result: HTTPResult,
    capture: Mapping[str, object],
    payload: WarcPayload,
    *,
    target_url: str,
    source_page: bool,
    parent_page: Mapping[str, object] | None,
    reported_digest: str | None,
    verified_digest: str | None,
    required_digest: str | None,
    required_date: str | None,
    required_record_id: str | None,
) -> ArchiveCandidate:
    if payload.decoded_body is None:
        raise ImageValidationError("WARC response contains no decoded payload")
    payload_result = HTTPResult(
        data=payload.decoded_body,
        status_code=200,
        content_type=payload.content_type,
        final_url=payload.target_uri or str(capture.get("url") or target_url),
        attempt_id=result.attempt_id,
        request_key=result.request_key,
        from_cache=result.from_cache,
    )
    return preserve_archive_candidate(
        paths,
        state,
        record,
        payload_result,
        strategy="commoncrawl_source_page" if source_page else "commoncrawl_exact_image",
        resolved_url=payload_result.final_url,
        archive_timestamp_value=str(capture.get("timestamp") or "") or None,
        archive_digest=(f"sha1:{reported_digest}" if reported_digest else None),
        verified_archive_digest=verified_digest,
        source_page=source_page,
        provenance={
            "capture": dict(capture),
            "warc_record_type": payload.record_type,
            "warc_reported_payload_digest": payload.reported_payload_digest,
            "computed_raw_payload_digest": payload.computed_payload_digest,
            "warc_payload_digest_verified": payload.payload_digest_verified,
            "source_page_provenance": parent_page,
            "required_revisit_digest": required_digest,
            "required_revisit_date": required_date,
            "required_revisit_record_id": required_record_id,
        },
    )


def _candidate_for_digest(
    state: RecoveryState, digest: str
) -> dict[str, object] | None:
    matches = [
        candidate
        for candidate in state.all_candidates()
        if archive_digest_key(candidate.get("archive_digest")) == digest
        and candidate_artifacts_valid(candidate)
    ]
    hashes = {str(candidate["sha256"]) for candidate in matches}
    return matches[0] if len(hashes) == 1 and matches else None


def _reuse_revisit_candidate(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    capture: Mapping[str, object],
    revisit: WarcPayload,
    donor: Mapping[str, object],
    *,
    source_page: bool,
    target_url: str,
    digest: str,
) -> ArchiveCandidate | None:
    raw_path = Path(str(donor["raw_path"]))
    normalized_path = Path(str(donor["normalized_path"]))
    if not verify_existing_blob(raw_path, str(donor["sha256"])) or not normalized_path.is_file():
        return None
    reference = create_reference(
        raw_path,
        paths.candidates
        / str(record["record_id"])
        / f"{donor['sha256']}{raw_path.suffix.casefold()}",
    )
    candidate = ArchiveCandidate(
        record_id=str(record["record_id"]),
        sha256=str(donor["sha256"]),
        perceptual_hash=str(donor["perceptual_hash"]),
        raw_path=str(raw_path),
        normalized_path=str(normalized_path),
        candidate_path=str(reference),
        strategy=(
            "commoncrawl_source_page_revisit"
            if source_page
            else "commoncrawl_exact_image_revisit"
        ),
        resolved_url=revisit.target_uri or str(capture.get("url") or target_url),
        archive_timestamp=str(capture.get("timestamp") or "") or None,
        archive_digest=f"sha1:{digest}",
        source_page=source_page,
        response_mime=revisit.content_type,
    )
    state.add_candidate(
        candidate,
        {
            "capture": dict(capture),
            "revisit_resolved_by_digest": digest,
            "refers_to": revisit.refers_to,
            "refers_to_target_uri": revisit.refers_to_target_uri,
            "refers_to_date": revisit.refers_to_date,
        },
    )
    return candidate


def collect_commoncrawl_url(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
    record: Mapping[str, object],
    target_url: str,
    indexes: Sequence[Mapping[str, str]],
    *,
    source_page: bool,
    permit_page_expansion: bool,
    visited: set[str],
    parent_page: Mapping[str, object] | None = None,
    required_digest: str | None = None,
    required_date: str | None = None,
    required_record_id: str | None = None,
) -> list[ArchiveCandidate]:
    """Collect all exact Common Crawl candidates for one URL."""
    digest_constraint = _required_digest(required_digest)
    if required_digest is not None and digest_constraint is None:
        state.log_attempt(
            str(record["record_id"]),
            "commoncrawl_revisit_referral",
            attempted_url=target_url,
            exception=f"unusable revisit digest {required_digest!r}",
            validation_result="revisit_digest_unusable",
        )
        return []
    timestamp_constraint = archive_timestamp(required_date)
    query_key = json.dumps(
        {
            "url": target_url,
            "digest": digest_constraint,
            "date": timestamp_constraint,
            "record_id": required_record_id,
        },
        sort_keys=True,
    )
    if not is_valid_http_url(target_url) or query_key in visited:
        return []
    visited.add(query_key)
    captures = query_commoncrawl_url(
        fetcher, str(record["record_id"]), target_url, indexes
    )
    if digest_constraint is not None:
        captures = [
            capture
            for capture in captures
            if archive_digest_key(capture.get("digest")) == digest_constraint
        ]
    if timestamp_constraint is not None:
        captures = [
            capture
            for capture in captures
            if archive_timestamp(capture.get("timestamp")) == timestamp_constraint
        ]
    captures.sort(
        key=lambda capture: (
            not str(capture.get("mime") or capture.get("mime-detected") or "")
            .casefold()
            .startswith("image/"),
            str(capture.get("timestamp") or ""),
        )
    )
    recovered: list[ArchiveCandidate] = []
    embedded: list[tuple[str, dict[str, object]]] = []
    revisits: list[tuple[Mapping[str, object], WarcPayload]] = []
    for capture in captures:
        if bool(capture.get("range_too_large")):
            state.log_attempt(
                str(record["record_id"]),
                "commoncrawl_warc_range_skipped",
                attempted_url="https://data.commoncrawl.org/" + str(capture["filename"]),
                response_size=int(capture["length"]),
                archive_collection=str(capture.get("collection") or ""),
                archive_timestamp=str(capture.get("timestamp") or ""),
                warc_filename=str(capture["filename"]),
                warc_offset=int(capture["offset"]),
                warc_length=int(capture["length"]),
                validation_result="warc_range_exceeds_safe_size_limit",
            )
            continue
        fetched = fetch_commoncrawl_warc(fetcher, str(record["record_id"]), capture)
        if fetched is None:
            continue
        result, payloads = fetched
        for payload in payloads:
            valid, reported_digest = _payload_identity_is_valid(
                state,
                result,
                capture,
                payload,
                required_digest=digest_constraint,
                required_timestamp=timestamp_constraint,
                required_record_id=required_record_id,
            )
            if not valid:
                continue
            if payload.record_type == "revisit":
                revisits.append((capture, payload))
                continue
            if payload.record_type != "response" or payload.http_status != "200":
                reject_warc_semantics(
                    state,
                    result,
                    "unsupported_warc_response",
                    f"record type/status was {payload.record_type!r}/{payload.http_status!r}",
                )
                continue
            if payload.body_too_large:
                reject_warc_semantics(
                    state,
                    result,
                    "warc_payload_too_large",
                    f"decoded payload exceeds {MAX_IMAGE_BYTES} bytes",
                )
                continue
            if payload.decoded_body is None:
                reject_warc_semantics(
                    state,
                    result,
                    "warc_response_payload_missing",
                    "ordinary response has no decoded payload",
                )
                continue
            verified_digest = _validated_payload_digest(
                state, result, payload, reported_digest, digest_constraint
            )
            if reported_digest is not None and verified_digest is None:
                continue
            try:
                recovered.append(
                    _persist_payload_candidate(
                        paths,
                        state,
                        record,
                        result,
                        capture,
                        payload,
                        target_url=target_url,
                        source_page=source_page,
                        parent_page=parent_page,
                        reported_digest=reported_digest,
                        verified_digest=verified_digest,
                        required_digest=digest_constraint,
                        required_date=required_date,
                        required_record_id=required_record_id,
                    )
                )
            except ImageValidationError as error:
                if permit_page_expansion and looks_like_html(
                    payload.decoded_body, payload.content_type
                ):
                    state.update_attempt(
                        result.attempt_id,
                        validation_result="html_source_page",
                        exception=str(error),
                    )
                    page_url = payload.target_uri or str(capture.get("url") or target_url)
                    for image_url in extract_explicit_image_urls(payload.decoded_body, page_url):
                        embedded.append(
                            (
                                image_url,
                                {
                                    "page_capture": dict(capture),
                                    "page_target_uri": payload.target_uri,
                                },
                            )
                        )
                else:
                    state.update_attempt(
                        result.attempt_id,
                        validation_result="invalid_warc_payload",
                        exception=str(error),
                    )
    for capture, revisit in revisits:
        digest = archive_digest_key(revisit.reported_payload_digest) or archive_digest_key(
            capture.get("digest")
        )
        if digest is None:
            continue
        donor = _candidate_for_digest(state, digest)
        if donor is not None:
            candidate = _reuse_revisit_candidate(
                paths,
                state,
                record,
                capture,
                revisit,
                donor,
                source_page=source_page,
                target_url=target_url,
                digest=digest,
            )
            if candidate is not None:
                recovered.append(candidate)
                continue
        referral_url = next(
            (
                str(value)
                for value in (
                    revisit.refers_to_target_uri,
                    revisit.target_uri,
                    capture.get("url"),
                    target_url,
                )
                if is_valid_http_url(value)
            ),
            None,
        )
        if referral_url is not None:
            recovered.extend(
                collect_commoncrawl_url(
                    paths,
                    state,
                    fetcher,
                    record,
                    referral_url,
                    indexes,
                    source_page=source_page,
                    permit_page_expansion=True,
                    visited=visited,
                    parent_page={"revisit_capture": dict(capture)},
                    required_digest=digest,
                    required_date=revisit.refers_to_date,
                    required_record_id=revisit.refers_to,
                )
            )
    for image_url, provenance in embedded:
        recovered.extend(
            collect_commoncrawl_url(
                paths,
                state,
                fetcher,
                record,
                image_url,
                indexes,
                source_page=True,
                permit_page_expansion=False,
                visited=visited,
                parent_page=provenance,
            )
        )
    return recovered


def recover_commoncrawl(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
    record: Mapping[str, object],
    *,
    indexes: Sequence[Mapping[str, str]] | None = None,
) -> StageResult:
    """Collect Common Crawl candidates independently of every other source."""
    original_variants = safe_url_variants(record.get("original_url"))
    source_page_url = record.get("source_page_url")
    if not original_variants and not is_valid_http_url(source_page_url):
        return StageResult(StageOutcome.NOT_APPLICABLE, "no archivable URL")
    active_indexes = (
        list(indexes)
        if indexes is not None
        else commoncrawl_indexes(
            fetcher, state, str(record["record_id"])
        )
    )
    visited: set[str] = set()
    recovered: list[ArchiveCandidate] = []
    for variant in original_variants:
        recovered.extend(
            collect_commoncrawl_url(
                paths,
                state,
                fetcher,
                record,
                variant,
                active_indexes,
                source_page=False,
                permit_page_expansion=True,
                visited=visited,
            )
        )
    if is_valid_http_url(source_page_url):
        recovered.extend(
            collect_commoncrawl_url(
                paths,
                state,
                fetcher,
                record,
                str(source_page_url),
                active_indexes,
                source_page=True,
                permit_page_expansion=True,
                visited=visited,
            )
        )
    if recovered:
        return StageResult(StageOutcome.RECOVERED, f"collected {len(recovered)} candidates")
    return StageResult(StageOutcome.MISS, "all Common Crawl captures were exhausted")
