"""Wayback CDX lookup, replay validation, and archived-image recovery."""

from __future__ import annotations

import base64
import hashlib
import html
import ipaddress
import json
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Callable, Protocol

from bs4 import BeautifulSoup

from .images import preserve_archive_candidate
from .state import RecoveryState
from .webster import (
    ArchiveCandidate,
    DownloadError,
    HTTPResult,
    ImageValidationError,
    StageOutcome,
    StageResult,
    WebsterPaths,
)


MAX_METADATA_BYTES = 32 * 1024 * 1024
WAYBACK_CDX_PAGE_SIZE = 1_000
MAX_WAYBACK_CDX_PAGES = 10_000


class Fetcher(Protocol):
    state: RecoveryState
    get: Callable[..., HTTPResult]


def archive_digest_key(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().removeprefix("sha1:")
    return (
        normalized
        if re.fullmatch(r"[a-z2-7]{32}", normalized)
        else None
    )


def archive_sha1(data: bytes) -> str:
    return (
        base64.b32encode(hashlib.sha1(data).digest())
        .decode("ascii")
        .rstrip("=")
        .lower()
    )


def archive_digest_matches(digest: str, data: bytes) -> bool:
    normalized = archive_digest_key(digest)
    return normalized is not None and archive_sha1(data) == normalized


def is_valid_http_url(value: object) -> bool:
    if value is None or not str(value).strip():
        return False
    try:
        parsed = urllib.parse.urlsplit(str(value).strip())
    except (UnicodeError, ValueError):
        return False
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and bool(parsed.hostname)
    )


def encoded_path_variant(url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return url
    escaped = re.sub(r"%(?![0-9A-Fa-f]{2})", "%25", parsed.path)
    encoded = urllib.parse.quote(
        escaped,
        safe="/%:@!$&'()*+,;=-._~",
        encoding="utf-8",
        errors="strict",
    )
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            encoded,
            parsed.query,
            parsed.fragment,
        )
    )


def normalize_url(raw_url: object) -> str | None:
    if raw_url is None or not str(raw_url).strip():
        return None
    decoded = html.unescape(str(raw_url))
    encoded = encoded_path_variant(urllib.parse.urldefrag(decoded).url)
    try:
        parsed = urllib.parse.urlsplit(encoded)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or parsed.hostname is None
        ):
            return None
        scheme = parsed.scheme.casefold()
        port = parsed.port
        if (scheme, port) in {("http", 80), ("https", 443)}:
            port = None
        host = parsed.hostname.casefold()
        netloc = host if port is None else f"{host}:{port}"
        return urllib.parse.urlunsplit(
            (scheme, netloc, parsed.path or "/", parsed.query, "")
        )
    except (UnicodeError, ValueError):
        return None


def _toggle_www(url: str) -> str | None:
    try:
        parsed = urllib.parse.urlsplit(url)
        hostname = parsed.hostname
        if hostname is None or "." not in hostname:
            return None
        try:
            ipaddress.ip_address(hostname)
            return None
        except ValueError:
            if hostname.casefold().startswith("www."):
                replacement = hostname[4:]
            elif len(hostname.split(".")) == 2:
                replacement = "www." + hostname
            else:
                return None
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urllib.parse.urlunsplit(
            (
                parsed.scheme,
                replacement + port,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    except (UnicodeError, ValueError):
        return None


def safe_url_variants(raw_url: object) -> list[str]:
    if raw_url is None or not str(raw_url).strip():
        return []
    original = str(raw_url)
    decoded = html.unescape(original)
    defragmented = urllib.parse.urldefrag(decoded).url
    encoded = encoded_path_variant(defragmented)
    candidates = [original, decoded, defragmented, encoded]
    try:
        parsed = urllib.parse.urlsplit(encoded)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.scheme.casefold() in {"http", "https"}:
        alternate = (
            "https" if parsed.scheme.casefold() == "http" else "http"
        )
        candidates.append(
            urllib.parse.urlunsplit(
                (
                    alternate,
                    parsed.netloc,
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
        )
    www = _toggle_www(encoded)
    if www is not None:
        candidates.append(www)
    result: list[str] = []
    for candidate in candidates:
        if is_valid_http_url(candidate) and candidate not in result:
            result.append(candidate)
    return result


def looks_like_html(data: bytes, content_type: str | None) -> bool:
    prefix = data[:2048].lstrip().lower()
    mime = (content_type or "").casefold()
    return (
        "text/html" in mime
        or "application/xhtml" in mime
        or prefix.startswith((b"<!doctype html", b"<html"))
        or b"<html" in prefix
    )


def extract_explicit_image_urls(
    page: bytes | str, original_page_url: str
) -> list[str]:
    text = (
        page.decode("utf-8", errors="replace")
        if isinstance(page, bytes)
        else page
    )
    soup = BeautifulSoup(text, "lxml")
    references: list[str] = []
    selectors: Sequence[tuple[str, str]] = (
        ('meta[property="og:image:secure_url"]', "content"),
        ('meta[property="og:image"]', "content"),
        ('meta[property="og:image:url"]', "content"),
        ('meta[name="twitter:image"]', "content"),
        ('meta[name="twitter:image:src"]', "content"),
        ('meta[itemprop="image"]', "content"),
        ('link[rel="image_src"]', "href"),
    )
    for selector, attribute in selectors:
        for tag in soup.select(selector):
            value = tag.get(attribute)
            if value is not None and str(value).strip():
                references.append(str(value).strip())
    if not references:
        image_tags: list[str] = []
        for tag in soup.find_all("img"):
            source = tag.get("src") or tag.get("data-src")
            if source is None or not str(source).strip():
                continue
            lowered = str(source).casefold()
            furniture = (
                "logo", "avatar", "icon", "sprite", "pixel", "spacer", "advert"
            )
            if (
                lowered.startswith("data:")
                or any(token in lowered for token in furniture)
            ):
                continue
            image_tags.append(str(source).strip())
        if len(image_tags) == 1:
            references.extend(image_tags)
    base_url = original_page_url
    base_tag = soup.find("base", href=True)
    if base_tag is not None:
        proposed = urllib.parse.urljoin(
            original_page_url, str(base_tag["href"])
        )
        if is_valid_http_url(proposed):
            base_url = proposed
    resolved: list[str] = []
    for reference in references:
        wrapper = re.search(
            r"/web/\d{1,14}(?:id_|im_|if_)?/(https?://.+)$",
            html.unescape(reference),
        )
        unwrapped = wrapper.group(1) if wrapper else html.unescape(reference)
        absolute = urllib.parse.urljoin(base_url, unwrapped)
        if is_valid_http_url(absolute) and absolute not in resolved:
            resolved.append(absolute)
    return resolved


def parse_cdx_response(data: bytes | str) -> list[dict[str, str]]:
    """Parse and validate one complete Wayback CDX JSON array."""

    text = (
        data.decode("utf-8", errors="replace")
        if isinstance(data, bytes)
        else data
    )
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Wayback CDX returned invalid JSON: {error}") from error
    if not isinstance(decoded, list):
        raise ValueError("Wayback CDX response is not a JSON array")
    if not decoded:
        return []
    if not isinstance(decoded[0], list):
        raise ValueError("Wayback CDX response has no header row")
    fields = [str(value) for value in decoded[0]]
    required = {
        "timestamp", "original", "statuscode", "mimetype", "digest", "length"
    }
    if len(fields) != len(set(fields)) or not required.issubset(fields):
        raise ValueError(
            f"Wayback CDX header is incomplete or duplicated: {fields}"
        )
    captures: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row_number, raw_row in enumerate(decoded[1:], 1):
        if not isinstance(raw_row, list) or len(raw_row) != len(fields):
            raise ValueError(
                f"Wayback CDX row {row_number} has the wrong shape"
            )
        values = ["" if value is None else str(value) for value in raw_row]
        capture = dict(zip(fields, values, strict=True))
        if capture["statuscode"] != "200":
            continue
        if re.fullmatch(r"\d{14}", capture["timestamp"]) is None:
            raise ValueError(
                f"Wayback CDX row {row_number} has an invalid timestamp"
            )
        if not is_valid_http_url(capture["original"]):
            raise ValueError(
                f"Wayback CDX row {row_number} has an invalid URL"
            )
        digest = capture["digest"].strip()
        if digest not in {"", "-"} and archive_digest_key(digest) is None:
            raise ValueError(
                f"Wayback CDX row {row_number} has an invalid digest"
            )
        length = capture["length"].strip()
        if length not in {"", "-"}:
            try:
                if int(length) < 0:
                    raise ValueError
            except ValueError as error:
                raise ValueError(
                    f"Wayback CDX row {row_number} has an invalid length"
                ) from error
        identity = tuple(capture.get(field, "") for field in fields)
        if identity not in seen:
            captures.append(capture)
            seen.add(identity)
    return captures


def parse_wayback_cdx_page(
    data: bytes | str,
) -> tuple[list[dict[str, str]], str | None]:
    text = (
        data.decode("utf-8", errors="replace")
        if isinstance(data, bytes)
        else data
    )
    stripped = text.lstrip()
    try:
        decoded, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as error:
        raise ValueError(f"Wayback CDX returned invalid JSON: {error}") from error
    if not isinstance(decoded, list):
        raise ValueError("Wayback CDX response is not a JSON array")
    rows = list(decoded)
    resume_key: str | None = None
    if rows and isinstance(rows[-1], dict) and "resumeKey" in rows[-1]:
        resume_key = str(rows.pop()["resumeKey"]).strip()
    elif (
        rows
        and isinstance(rows[-1], list)
        and len(rows[-1]) == 2
        and str(rows[-1][0]).casefold() == "resumekey"
    ):
        resume_key = str(rows.pop()[1]).strip()
    remainder = stripped[end:].strip()
    if remainder:
        trailing = _trailing_resume_key(remainder)
        if resume_key is not None and trailing != resume_key:
            raise ValueError("Wayback CDX returned conflicting resume keys")
        resume_key = trailing
    if resume_key == "":
        raise ValueError("Wayback CDX returned an empty resume key")
    return parse_cdx_response(json.dumps(rows)), resume_key


def _trailing_resume_key(remainder: str) -> str:
    if remainder.casefold().startswith("resumekey:"):
        return remainder.split(":", 1)[1].strip()
    try:
        trailing = json.loads(remainder)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Wayback CDX has an unrecognized trailing payload"
        ) from error
    if isinstance(trailing, dict) and "resumeKey" in trailing:
        return str(trailing["resumeKey"]).strip()
    if (
        isinstance(trailing, list)
        and len(trailing) == 2
        and str(trailing[0]).casefold() == "resumekey"
    ):
        return str(trailing[1]).strip()
    raise ValueError("Wayback CDX has an unrecognized trailing payload")


def wayback_cdx_captures(
    fetcher: Fetcher, record_id: str, target_url: str
) -> list[dict[str, str]]:
    base_parameters: dict[str, object] = {
        "url": target_url,
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest,length",
        "filter": "statuscode:200",
        "matchType": "exact",
        "showResumeKey": "true",
        "limit": WAYBACK_CDX_PAGE_SIZE,
    }
    captures: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    seen_keys: set[str] = set()
    resume_key: str | None = None
    for page_number in range(1, MAX_WAYBACK_CDX_PAGES + 1):
        parameters = dict(base_parameters)
        if resume_key is not None:
            parameters["resumeKey"] = resume_key
        try:
            result = fetcher.get(
                record_id,
                "wayback_cdx",
                "https://web.archive.org/cdx/search/cdx",
                params=parameters,
                maximum_bytes=MAX_METADATA_BYTES,
                cache=True,
                archive_collection="wayback",
            )
        except DownloadError as error:
            if error.status_code == 404 and page_number == 1:
                return []
            raise
        try:
            page, next_key = parse_wayback_cdx_page(result.data)
        except ValueError as error:
            fetcher.state.invalidate_http_cache(result.request_key)
            raise DownloadError(
                f"invalid Wayback CDX response: {error}", transient=True
            ) from error
        fetcher.state.update_attempt(
            result.attempt_id,
            validation_result=(
                f"cdx_page:{page_number}:captures:{len(page)}"
            ),
        )
        for capture in page:
            identity = tuple(
                capture.get(field, "")
                for field in (
                    "timestamp", "original", "statuscode",
                    "mimetype", "digest", "length",
                )
            )
            if identity not in seen:
                captures.append(capture)
                seen.add(identity)
        if next_key is None:
            return captures
        if next_key == resume_key or next_key in seen_keys:
            fetcher.state.invalidate_http_cache(result.request_key)
            raise DownloadError("Wayback CDX resume-key loop", transient=True)
        seen_keys.add(next_key)
        resume_key = next_key
    raise DownloadError(
        f"Wayback CDX exceeded {MAX_WAYBACK_CDX_PAGES} pages",
        transient=True,
    )


def build_wayback_replay_url(timestamp: str, original_url: str) -> str:
    if re.fullmatch(r"\d{14}", timestamp) is None:
        raise ValueError(f"invalid Wayback timestamp: {timestamp!r}")
    archived = encoded_path_variant(
        urllib.parse.urldefrag(original_url).url
    )
    return f"https://web.archive.org/web/{timestamp}id_/{archived}"


def wayback_replay_matches(
    final_url: str, timestamp: str, original_url: str
) -> bool:
    try:
        parsed = urllib.parse.urlsplit(final_url)
        canonical_origin = (
            parsed.scheme.casefold() == "https"
            and (parsed.hostname or "").casefold() == "web.archive.org"
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
        r"/web/(\d{14})([a-z]+_)/(.*)", parsed.path, re.IGNORECASE
    )
    if (
        match is None
        or match.group(1) != timestamp
        or match.group(2).casefold() != "id_"
    ):
        return False
    embedded = match.group(3) + (
        "?" + parsed.query if parsed.query else ""
    )
    expected = normalize_url(original_url)
    if expected is None:
        return False
    raw = normalize_url(embedded)
    if raw == expected:
        return True
    wrapped = (
        raw is None
        or re.search(r"%25[0-9a-f]{2}", embedded, re.I) is not None
    )
    return (
        wrapped
        and normalize_url(urllib.parse.unquote(embedded)) == expected
    )


def _validated_replay(
    fetcher: Fetcher,
    record_id: str,
    replay_url: str,
    timestamp: str,
    archived_original: str,
) -> HTTPResult | None:
    try:
        result = fetcher.get(
            record_id,
            "wayback_replay",
            replay_url,
            headers={"Accept-Encoding": "identity"},
            cache=True,
            archive_collection="wayback",
            archive_timestamp=timestamp,
        )
    except DownloadError as error:
        if error.transient:
            raise
        return None
    if wayback_replay_matches(
        result.final_url, timestamp, archived_original
    ):
        return result
    fetcher.state.update_attempt(
        result.attempt_id,
        validation_result="rejected_wayback_semantic_redirect",
        exception=(
            "replay final URL changed timestamp, modifier, or original URL"
        ),
    )
    fetcher.state.invalidate_http_cache(result.request_key)
    if result.from_cache:
        raise DownloadError(
            "cached Wayback replay has the wrong identity",
            transient=True,
        )
    return None


def _verify_replay_digest(
    fetcher: Fetcher,
    record_id: str,
    replay_url: str,
    timestamp: str,
    archived_original: str,
    decoded_result: HTTPResult,
    digest: str | None,
) -> tuple[str | None, HTTPResult | None]:
    if digest is None:
        return None, None
    try:
        raw = fetcher.get(
            record_id,
            "wayback_replay_raw_digest",
            replay_url,
            headers={"Accept-Encoding": "identity"},
            cache=True,
            archive_collection="wayback",
            archive_timestamp=timestamp,
            raw_stream=True,
        )
    except DownloadError as error:
        if error.transient:
            raise
        return None, None
    if not wayback_replay_matches(
        raw.final_url, timestamp, archived_original
    ):
        fetcher.state.invalidate_http_cache(raw.request_key)
        fetcher.state.update_attempt(
            raw.attempt_id,
            validation_result="rejected_wayback_raw_semantic_redirect",
            exception="raw replay final URL changed capture identity",
        )
        if raw.from_cache:
            raise DownloadError(
                "cached raw replay has the wrong identity", transient=True
            )
        return None, None
    if raw.data != decoded_result.data:
        message = (
            "identity-encoded raw and decoded Wayback responses "
            "returned different bytes"
        )
        for result in (decoded_result, raw):
            fetcher.state.update_attempt(
                result.attempt_id,
                validation_result="wayback_candidate_payload_unbound",
                exception=message,
            )
            fetcher.state.invalidate_http_cache(result.request_key)
        raise DownloadError(message, transient=True)
    if not archive_digest_matches(digest, raw.data):
        fetcher.state.update_attempt(
            raw.attempt_id,
            validation_result="wayback_archive_digest_mismatch",
            exception=(
                f"CDX digest {digest} differs from "
                f"replay sha1:{archive_sha1(raw.data)}"
            ),
        )
        fetcher.state.invalidate_http_cache(raw.request_key)
        if raw.from_cache:
            raise DownloadError(
                "cached replay failed its CDX digest", transient=True
            )
        return None, None
    fetcher.state.update_attempt(
        raw.attempt_id,
        validation_result="wayback_raw_payload_digest_verified",
    )
    return digest, raw


def collect_wayback_url(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
    record: Mapping[str, object],
    target_url: str,
    *,
    source_page: bool,
    permit_page_expansion: bool,
    visited: set[str],
    parent_page: Mapping[str, object] | None = None,
) -> list[ArchiveCandidate]:
    if not is_valid_http_url(target_url) or target_url in visited:
        return []
    visited.add(target_url)
    captures = wayback_cdx_captures(
        fetcher, str(record["record_id"]), target_url
    )
    captures.sort(
        key=lambda capture: (
            capture.get("original")
            != str(record.get("original_url") or ""),
            not capture.get("mimetype", "").casefold().startswith("image/"),
            capture.get("timestamp", ""),
        )
    )
    recovered: list[ArchiveCandidate] = []
    embedded: list[tuple[str, dict[str, object]]] = []
    seen_replays: set[tuple[str, str, str]] = set()
    for capture in captures:
        timestamp = capture["timestamp"]
        archived_original = capture.get("original") or target_url
        digest_key = archive_digest_key(capture.get("digest"))
        digest = f"sha1:{digest_key}" if digest_key is not None else None
        identity = (timestamp, archived_original, digest or "")
        if identity in seen_replays:
            continue
        seen_replays.add(identity)
        replay_url = build_wayback_replay_url(
            timestamp, archived_original
        )
        result = _validated_replay(
            fetcher,
            str(record["record_id"]),
            replay_url,
            timestamp,
            archived_original,
        )
        if result is None:
            continue
        verified, raw = _verify_replay_digest(
            fetcher,
            str(record["record_id"]),
            replay_url,
            timestamp,
            archived_original,
            result,
            digest,
        )
        if digest is not None and verified is None:
            continue
        try:
            recovered.append(
                preserve_archive_candidate(
                    paths,
                    state,
                    record,
                    result,
                    strategy=(
                        "wayback_source_page"
                        if source_page
                        else "wayback_exact_image"
                    ),
                    resolved_url=archived_original,
                    archive_timestamp_value=timestamp,
                    archive_digest=digest,
                    verified_archive_digest=verified,
                    source_page=source_page,
                    provenance={
                        "capture": capture,
                        "replay_url": replay_url,
                        "decoded_replay_final_url": result.final_url,
                        "raw_replay_final_url": (
                            raw.final_url if raw else None
                        ),
                        "raw_replay_request_key": (
                            raw.request_key if raw else None
                        ),
                        "queried_url": target_url,
                        "source_page_provenance": parent_page,
                    },
                )
            )
        except ImageValidationError as error:
            if (
                permit_page_expansion
                and looks_like_html(result.data, result.content_type)
            ):
                state.update_attempt(
                    result.attempt_id,
                    validation_result="html_source_page",
                    exception=str(error),
                )
                for image_url in extract_explicit_image_urls(
                    result.data, archived_original
                ):
                    embedded.append(
                        (
                            image_url,
                            {
                                "page_capture": capture,
                                "page_replay_url": replay_url,
                                "page_original_url": archived_original,
                            },
                        )
                    )
            else:
                state.update_attempt(
                    result.attempt_id,
                    validation_result="invalid_image",
                    exception=str(error),
                )
    for image_url, provenance in embedded:
        recovered.extend(
            collect_wayback_url(
                paths,
                state,
                fetcher,
                record,
                image_url,
                source_page=True,
                permit_page_expansion=False,
                visited=visited,
                parent_page=provenance,
            )
        )
    return recovered


def recover_wayback(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: Fetcher,
    record: Mapping[str, object],
) -> StageResult:
    original_variants = safe_url_variants(record.get("original_url"))
    source_page_url = record.get("source_page_url")
    if not original_variants and not is_valid_http_url(source_page_url):
        return StageResult(
            StageOutcome.NOT_APPLICABLE, "no archivable URL"
        )
    visited: set[str] = set()
    candidates: list[ArchiveCandidate] = []
    for variant in original_variants:
        candidates.extend(
            collect_wayback_url(
                paths,
                state,
                fetcher,
                record,
                variant,
                source_page=False,
                permit_page_expansion=True,
                visited=visited,
            )
        )
    if is_valid_http_url(source_page_url):
        candidates.extend(
            collect_wayback_url(
                paths,
                state,
                fetcher,
                record,
                str(source_page_url),
                source_page=True,
                permit_page_expansion=True,
                visited=visited,
            )
        )
    if candidates:
        return StageResult(
            StageOutcome.RECOVERED,
            f"collected {len(candidates)} candidates",
        )
    return StageResult(
        StageOutcome.MISS, "all Wayback captures were exhausted"
    )


__all__ = [
    "archive_digest_key",
    "archive_digest_matches",
    "archive_sha1",
    "build_wayback_replay_url",
    "collect_wayback_url",
    "encoded_path_variant",
    "extract_explicit_image_urls",
    "is_valid_http_url",
    "looks_like_html",
    "normalize_url",
    "parse_cdx_response",
    "parse_wayback_cdx_page",
    "recover_wayback",
    "safe_url_variants",
    "wayback_cdx_captures",
    "wayback_replay_matches",
]
