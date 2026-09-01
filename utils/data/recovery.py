"""High-level resumable Webster recovery and bounded HTTP access."""

from __future__ import annotations

import concurrent.futures
import email.utils
import errno
import hashlib
import json
import random
import re
import socket
import ssl
import threading
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import httpx
import pandas as pd
from tqdm import tqdm

from utils.common.io import (
    atomic_write_bytes,
    atomic_write_frame_csv,
    atomic_write_frame_parquet,
    atomic_write_json,
)

from .arquivo import (
    ARQUIVO_STRATEGY_VERSION,
    recover_arquivo,
)
from .commoncrawl import commoncrawl_indexes, recover_commoncrawl
from .images import (
    apply_image_resolution,
    candidate_artifacts_valid,
    create_reference,
    recovery_artifacts_available,
    recovery_artifacts_valid,
    store_image,
)
from .mirror import (
    MIRROR_STRATEGY_VERSION,
    prepare_verified_mirror,
    record_identity as mirror_record_identity,
    recover_mirror_row,
)
from .official import (
    inspect_official_tree,
    map_and_cache_official_assets,
    recover_official,
)
from .state import RecoveryState
from .wayback import (
    is_valid_http_url,
    recover_wayback,
    safe_url_variants,
)
from .webster import (
    ACTIVE_STATUS_VALUES,
    ALL_STATUS_VALUES,
    RECOVERED_STATUS_VALUES,
    ArchiveCandidate,
    DownloadError,
    DownloadSummary,
    HTTPResult,
    ImageValidationError,
    InspectionSummary,
    OrganizationSummary,
    RecoveryStage,
    RecoveryStatus,
    RetryableStageError,
    StageOutcome,
    StageResult,
    StateConsistencyError,
    WebsterPaths,
    acquire_and_build_manifest,
    inspect_webster_dataset,
    normalize_webster_type,
    organize_model_views,
)


MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_REDIRECTS = 10
DEFAULT_HTTP_ATTEMPTS = 5
MAX_DIRECT_ATTEMPTS = DEFAULT_HTTP_ATTEMPTS
DEFAULT_DIRECT_ATTEMPTS = 2
DEFAULT_DIRECT_WORKERS = 24
MAX_DIRECT_WORKERS = 64
DEFAULT_PER_HOST_CONCURRENCY = 4
MAX_PER_HOST_CONCURRENCY = 8
ARQUIVO_CIRCUIT_BREAKER_THRESHOLD = 3
COMMONCRAWL_CIRCUIT_BREAKER_THRESHOLD = 3
BACKOFF_CAP_SECONDS = 60.0
USER_AGENT = (
    "WebsterBenchmarkSourceRecovery/2.0 "
    "(academic reproducibility)"
)
_T = TypeVar("_T")
_WEBSTER_CATEGORIES = ("MV", "TV", "RV", "N")


def _progress(
    values: Iterable[_T],
    *,
    total: int,
    description: str,
    unit: str = "record",
    enabled: bool = True,
    after: Callable[[_T], None] | None = None,
    postfix: Callable[[], str] | None = None,
) -> Iterable[_T]:
    """Show durable progress and refresh its compact recovery counters."""

    if not enabled:
        return values

    def iterate() -> Iterable[_T]:
        with tqdm(
            total=total,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
            leave=True,
            disable=False,
        ) as progress:
            if postfix is not None:
                progress.set_postfix_str(postfix(), refresh=False)
            for value in values:
                yield value
                if after is not None:
                    after(value)
                progress.update(1)
                if postfix is not None:
                    progress.set_postfix_str(postfix(), refresh=False)

    return iterate()


class _RecoveryCounts:
    """Maintain cumulative recovered/total prompt counts for tqdm postfixes."""

    def __init__(
        self,
        state: RecoveryState,
        source_manifest: pd.DataFrame,
    ) -> None:
        self._state = state
        self._lock = threading.Lock()
        self._category_by_id: dict[str, str] = {}
        self._totals: Counter[str] = Counter()
        for record in source_manifest.to_dict(orient="records"):
            record_id = str(record["record_id"])
            category = normalize_webster_type(record.get("overfit_type"))
            self._category_by_id[record_id] = category
            self._totals[category] += 1
        self._recovered_ids: set[str] = set()
        self._recovered: Counter[str] = Counter()
        for record in state.records(statuses=RECOVERED_STATUS_VALUES):
            self._synchronize(record)

    def _synchronize(self, record: Mapping[str, object]) -> None:
        record_id = str(record["record_id"])
        category = self._category_by_id.get(record_id)
        if category is None:
            category = normalize_webster_type(record.get("overfit_type"))
            self._category_by_id[record_id] = category
            self._totals[category] += 1
        status = str(record.get("recovery_status") or "")
        is_recovered = status in RECOVERED_STATUS_VALUES
        with self._lock:
            was_recovered = record_id in self._recovered_ids
            if is_recovered and not was_recovered:
                self._recovered_ids.add(record_id)
                self._recovered[category] += 1
            elif was_recovered and not is_recovered:
                self._recovered_ids.remove(record_id)
                self._recovered[category] -= 1

    def refresh(self, record: Mapping[str, object]) -> None:
        current = self._state.get_record(str(record["record_id"]))
        self._synchronize(current or record)

    def format(self) -> str:
        with self._lock:
            parts = [
                f"{category} {self._recovered[category]}/"
                f"{self._totals[category]}"
                for category in _WEBSTER_CATEGORIES
            ]
            recovered_total = len(self._recovered_ids)
            total = sum(self._totals.values())
        return " | ".join([*parts, f"Total {recovered_total}/{total}"])


def _manifest_progress_postfix(manifest: pd.DataFrame) -> str:
    categories = manifest["overfit_type"].map(normalize_webster_type)
    recovered_mask = manifest["recovery_status"].astype(str).isin(
        RECOVERED_STATUS_VALUES
    )
    totals = Counter(categories)
    recovered = Counter(categories.loc[recovered_mask])
    parts = [
        f"{category} {recovered[category]}/{totals[category]}"
        for category in _WEBSTER_CATEGORIES
    ]
    return " | ".join(
        [
            *parts,
            f"Total {int(recovered_mask.sum())}/{len(manifest)}",
        ]
    )


def _permanent_dns_failure(error: BaseException) -> bool:
    """Identify DNS answers that retries cannot change during this run."""

    current: BaseException | None = error
    visited: set[int] = set()
    permanent_codes = {socket.EAI_NONAME}
    if hasattr(socket, "EAI_NODATA"):
        permanent_codes.add(socket.EAI_NODATA)
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, socket.gaierror) and current.errno in permanent_codes:
            return True
        current = current.__cause__ or current.__context__
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "name or service not known",
            "nodename nor servname provided",
            "no address associated with hostname",
        )
    )


def _permanent_tls_failure(error: BaseException) -> bool:
    """Identify certificate errors that retrying the same URL cannot fix."""

    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        current = current.__cause__ or current.__context__
    message = str(error).casefold()
    return "certificate verify failed" in message or "hostname mismatch" in message


def _transport_failure_key(error: BaseException, url: str) -> str:
    """Return a stable service-level signature for transient HTTP failures."""

    hostname = (
        urllib.parse.urlsplit(url).hostname or "unknown-host"
    ).casefold()
    current: BaseException | None = error
    visited: set[int] = set()
    kind: str | None = None
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            kind = "tls_verification_failure"
            break
        if isinstance(current, socket.gaierror) and current.errno in {
            socket.EAI_NONAME,
            getattr(socket, "EAI_NODATA", socket.EAI_NONAME),
        }:
            kind = "permanent_dns_failure"
            break
        if isinstance(current, (httpx.TimeoutException, TimeoutError)):
            kind = "timeout"
            break
        if (
            isinstance(current, ConnectionRefusedError)
            or getattr(current, "errno", None) == errno.ECONNREFUSED
        ):
            kind = "connection_refused"
            break
        if (
            isinstance(current, socket.gaierror)
            and current.errno == socket.EAI_AGAIN
        ):
            kind = "temporary_dns_failure"
            break
        current = current.__cause__ or current.__context__
    if kind is None:
        message = str(error).casefold()
        if (
            "certificate verify failed" in message
            or "hostname mismatch" in message
        ):
            kind = "tls_verification_failure"
        elif (
            "name or service not known" in message
            or "nodename nor servname provided" in message
            or "no address associated with hostname" in message
        ):
            kind = "permanent_dns_failure"
        elif "connection refused" in message:
            kind = "connection_refused"
        elif "timed out" in message or "timeout" in message:
            kind = "timeout"
        elif "temporary failure in name resolution" in message:
            kind = "temporary_dns_failure"
        else:
            kind = "transport_error"
    return f"{hostname}:{kind}"


def _interleave_direct_records(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Round-robin hosts so blocked same-host requests cannot starve the pool."""

    occurrences: Counter[str] = Counter()
    ranked: list[tuple[int, int, dict[str, object]]] = []
    for position, record in enumerate(records):
        try:
            hostname = urllib.parse.urlsplit(
                str(record.get("original_url") or "")
            ).hostname
        except (UnicodeError, ValueError):
            hostname = None
        host_key = (hostname or f"__missing_url_{position}").casefold()
        rank = occurrences[host_key]
        occurrences[host_key] += 1
        ranked.append((rank, position, record))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in ranked]


def _identity_text(value: object) -> str | None:
    """Normalize persisted identity scalars without grouping missing values."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        return None
    return text


def _target_identity(record: Mapping[str, object]) -> tuple[str, ...]:
    """Build a conservative exact key for failed-target deduplication.

    Every available identity signal participates in the key.  Records that
    share an index but disagree on their URL (or vice versa) are intentionally
    kept separate so conflicting metadata cannot suppress a recovery attempt.
    """

    sample_index = _identity_text(record.get("laion_or_sample_index"))
    normalized_url = _identity_text(record.get("normalized_url"))
    original_url = _identity_text(record.get("original_url"))
    source_page_url = _identity_text(record.get("source_page_url"))
    parts: list[str] = []
    if sample_index is not None:
        parts.extend(("index", sample_index))
    if normalized_url is not None:
        parts.extend(("url", normalized_url))
    elif original_url is not None:
        parts.extend(("original", original_url))
    if source_page_url is not None:
        parts.extend(("source", source_page_url))
    if parts:
        return tuple(parts)
    return ("record", str(record["record_id"]))


def _group_target_records(
    records: Sequence[dict[str, object]],
) -> list[tuple[dict[str, object], ...]]:
    """Group records with the same complete, non-conflicting target identity."""

    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(_target_identity(record), []).append(record)
    return [tuple(group) for group in grouped.values()]


def _validate_direct_tuning(
    direct_workers: int,
    direct_attempts: int,
    per_host_concurrency: int,
) -> None:
    if not 1 <= direct_workers <= MAX_DIRECT_WORKERS:
        raise ValueError(
            f"direct_workers must be between 1 and {MAX_DIRECT_WORKERS}"
        )
    if not 1 <= direct_attempts <= MAX_DIRECT_ATTEMPTS:
        raise ValueError(
            f"direct_attempts must be between 1 and {MAX_DIRECT_ATTEMPTS}"
        )
    if not 1 <= per_host_concurrency <= MAX_PER_HOST_CONCURRENCY:
        raise ValueError(
            "per_host_concurrency must be between 1 and "
            f"{MAX_PER_HOST_CONCURRENCY}"
        )
    if per_host_concurrency > direct_workers:
        raise ValueError("per_host_concurrency cannot exceed direct_workers")


class RateLimiter:
    def __init__(self, minimum_interval: float):
        self.minimum_interval = minimum_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            delay = max(0.0, self._next_allowed - time.monotonic())
            if delay:
                time.sleep(delay)
            self._next_allowed = (
                time.monotonic() + self.minimum_interval
            )


class HTTPFetcher:
    """Stream bounded GETs with retry, durable cache, and attempt audit."""

    transient_statuses = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        paths: WebsterPaths,
        state: RecoveryState,
        *,
        client: httpx.Client | None = None,
        direct_workers: int = DEFAULT_DIRECT_WORKERS,
        direct_attempts: int = DEFAULT_DIRECT_ATTEMPTS,
        per_host_concurrency: int = DEFAULT_PER_HOST_CONCURRENCY,
    ):
        _validate_direct_tuning(
            direct_workers, direct_attempts, per_host_concurrency
        )
        self.paths = paths
        self.state = state
        self.direct_attempts = direct_attempts
        self.per_host_concurrency = per_host_concurrency
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(60, connect=20),
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            limits=httpx.Limits(
                max_connections=direct_workers * 2,
                max_keepalive_connections=direct_workers,
            ),
        )
        self._lock = threading.Lock()
        self._semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._wayback = RateLimiter(1.0)
        self._arquivo = RateLimiter(0.5)
        self._commoncrawl = RateLimiter(1.0)
        self._arquivo_open_circuits: dict[str, str] = {}
        self._arquivo_activity: Counter[str] = Counter()
        self._commoncrawl_open_circuits: dict[str, str] = {}
        self._commoncrawl_activity: Counter[str] = Counter()

    def __enter__(self) -> HTTPFetcher:
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    @staticmethod
    def request_identity(
        url: str,
        params: Mapping[str, object] | None,
        headers: Mapping[str, str] | None,
        raw_stream: bool,
        follow_redirects: bool,
    ) -> tuple[str, dict[str, str], str]:
        request_url = str(
            httpx.URL(url)
            if params is None
            else httpx.URL(url, params=params)
        )
        request_headers = dict(headers or {})
        signature = json.dumps(
            {
                "url": request_url,
                "headers": sorted(request_headers.items()),
                "raw_stream": raw_stream,
                "follow_redirects": follow_redirects,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        return request_url, request_headers, key

    def _semaphore(self, hostname: str) -> threading.BoundedSemaphore:
        with self._lock:
            return self._semaphores.setdefault(
                hostname.casefold(),
                threading.BoundedSemaphore(self.per_host_concurrency),
            )

    def _rate_limit(self, hostname: str) -> None:
        lowered = hostname.casefold()
        if lowered == "web.archive.org" or lowered.endswith(".archive.org"):
            self._wayback.wait()
        elif lowered == "arquivo.pt" or lowered.endswith(".arquivo.pt"):
            self._arquivo.wait()
        elif lowered.endswith("commoncrawl.org"):
            self._commoncrawl.wait()

    def _record_commoncrawl_activity(
        self, request_url: str, kind: str
    ) -> None:
        try:
            hostname = (
                urllib.parse.urlsplit(request_url).hostname or ""
            ).casefold()
        except (UnicodeError, ValueError):
            return
        if not (
            hostname == "commoncrawl.org"
            or hostname.endswith(".commoncrawl.org")
        ):
            return
        with self._lock:
            self._commoncrawl_activity[kind] += 1

    def _record_arquivo_activity(
        self, request_url: str, kind: str
    ) -> None:
        try:
            hostname = (
                urllib.parse.urlsplit(request_url).hostname or ""
            ).casefold()
        except (UnicodeError, ValueError):
            return
        if not (
            hostname == "arquivo.pt"
            or hostname.endswith(".arquivo.pt")
        ):
            return
        with self._lock:
            self._arquivo_activity[kind] += 1

    def reset_arquivo_activity(self) -> None:
        with self._lock:
            self._arquivo_activity.clear()

    def arquivo_activity(self) -> dict[str, int]:
        with self._lock:
            return dict(self._arquivo_activity)

    def reset_arquivo_circuits(self) -> None:
        """Start a fresh Arquivo.pt outage probe for this recovery run."""

        with self._lock:
            self._arquivo_open_circuits.clear()

    def open_arquivo_circuit(self, failure_key: str) -> str | None:
        """Defer uncached requests to a failed Arquivo.pt service host."""

        hostname, separator, _kind = failure_key.partition(":")
        hostname = hostname.casefold()
        if (
            not separator
            or not hostname
            or not (
                hostname == "arquivo.pt"
                or hostname.endswith(".arquivo.pt")
            )
        ):
            return None
        with self._lock:
            self._arquivo_open_circuits[hostname] = failure_key
        return hostname

    def _arquivo_circuit_error(
        self, request_url: str
    ) -> DownloadError | None:
        try:
            hostname = (
                urllib.parse.urlsplit(request_url).hostname or ""
            ).casefold()
        except (UnicodeError, ValueError):
            return None
        with self._lock:
            failure_key = self._arquivo_open_circuits.get(hostname)
        if failure_key is None:
            return None
        return DownloadError(
            f"Arquivo.pt circuit is open for {hostname}; "
            "uncached request deferred until the next run",
            transient=True,
            service_failure_key=failure_key,
            deferred_by_circuit=True,
        )

    def reset_commoncrawl_activity(self) -> None:
        with self._lock:
            self._commoncrawl_activity.clear()

    def commoncrawl_activity(self) -> dict[str, int]:
        with self._lock:
            return dict(self._commoncrawl_activity)

    def reset_commoncrawl_circuits(self) -> None:
        """Start a fresh Common Crawl outage probe for this recovery run."""

        with self._lock:
            self._commoncrawl_open_circuits.clear()

    def open_commoncrawl_circuit(self, failure_key: str) -> str | None:
        """Defer uncached requests to one failed Common Crawl service host."""

        hostname, separator, _kind = failure_key.partition(":")
        hostname = hostname.casefold()
        if (
            not separator
            or not hostname
            or not (
                hostname == "commoncrawl.org"
                or hostname.endswith(".commoncrawl.org")
            )
        ):
            return None
        with self._lock:
            self._commoncrawl_open_circuits[hostname] = failure_key
        return hostname

    def _commoncrawl_circuit_error(
        self, request_url: str
    ) -> DownloadError | None:
        try:
            hostname = (
                urllib.parse.urlsplit(request_url).hostname or ""
            ).casefold()
        except (UnicodeError, ValueError):
            return None
        with self._lock:
            failure_key = self._commoncrawl_open_circuits.get(hostname)
        if failure_key is None:
            return None
        return DownloadError(
            f"Common Crawl circuit is open for {hostname}; "
            "uncached request deferred until the next run",
            transient=True,
            service_failure_key=failure_key,
            deferred_by_circuit=True,
        )

    def _cached_result(
        self,
        record_id: str,
        strategy: str,
        request_url: str,
        request_key: str,
        maximum_bytes: int,
        required_status: int | None,
        required_length: int | None,
        audit: Mapping[str, object],
    ) -> HTTPResult | None:
        cached = self.state.get_http_cache(request_key)
        if cached is None:
            return None
        try:
            body_path = Path(str(cached["body_path"]))
            if (
                body_path.parent.resolve()
                != self.paths.http_cache.resolve()
            ):
                raise ValueError("cached path leaves the HTTP cache")
            body = body_path.read_bytes()
            status = int(cached["status_code"])
            digest = hashlib.sha256(body).hexdigest()
            if (
                len(body) != int(cached["response_size"])
                or digest != str(cached["body_sha256"])
                or len(body) > maximum_bytes
                or required_status is not None
                and status != required_status
                or required_length is not None
                and len(body) != required_length
            ):
                raise ValueError("cached response metadata is inconsistent")
        except (KeyError, OSError, TypeError, ValueError) as error:
            self.state.invalidate_http_cache(request_key)
            self.state.log_attempt(
                record_id,
                strategy + "_cached_response",
                attempted_url=request_url,
                exception=f"{type(error).__name__}: {error}",
                validation_result="cache_invalid",
                request_key=request_key,
                **audit,
            )
            return None
        attempt_id = self.state.log_attempt(
            record_id,
            strategy + "_cached_response",
            attempted_url=request_url,
            http_status=status,
            response_mime=cached.get("content_type"),
            response_size=len(body),
            validation_result="cache_hit",
            request_key=request_key,
            **audit,
        )
        self._record_commoncrawl_activity(request_url, "cache_hits")
        self._record_arquivo_activity(request_url, "cache_hits")
        return HTTPResult(
            body,
            status,
            (
                None
                if cached.get("content_type") is None
                else str(cached["content_type"])
            ),
            str(cached["final_url"]),
            attempt_id,
            request_key,
            True,
        )

    @staticmethod
    def _retry_after(value: str | None) -> float:
        if not value:
            return 0.0
        try:
            delay = float(value)
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
                delay = parsed.timestamp() - time.time()
            except (OverflowError, TypeError, ValueError):
                return 0.0
        return min(BACKOFF_CAP_SECONDS, max(0.0, delay))

    @staticmethod
    def _validate_range(
        response: httpx.Response,
        offset: int | None,
        required_length: int,
    ) -> None:
        value = response.headers.get("content-range", "")
        match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value)
        expected_end = (
            offset + required_length - 1
            if offset is not None
            else None
        )
        if (
            match is None
            or offset is None
            or int(match.group(1)) != offset
            or int(match.group(2)) != expected_end
        ):
            raise DownloadError(
                f"invalid Content-Range: {value!r}", transient=True
            )

    def _request(
        self,
        url: str,
        headers: Mapping[str, str],
        *,
        maximum_bytes: int,
        raw_stream: bool,
        required_status: int | None,
        required_length: int | None,
        warc_offset: int | None,
        follow_redirects: bool,
    ) -> tuple[bytes, int, str | None, str, float]:
        hostname = urllib.parse.urlsplit(url).hostname or ""
        if not hostname:
            raise DownloadError(f"unsafe request URL: {url!r}")
        with self._semaphore(hostname):
            self._rate_limit(hostname)
            with self.client.stream(
                "GET",
                url,
                headers=headers,
                follow_redirects=follow_redirects,
            ) as response:
                status = response.status_code
                retry_after = self._retry_after(
                    response.headers.get("retry-after")
                )
                if status < 200 or status >= 300:
                    raise DownloadError(
                        f"HTTP {status}",
                        status_code=status,
                        transient=status in self.transient_statuses,
                        retry_after=retry_after,
                        service_failure_key=(
                            f"{hostname.casefold()}:http_{status}"
                            if status in self.transient_statuses
                            else None
                        ),
                    )
                if required_status is not None and status != required_status:
                    raise DownloadError(
                        f"required HTTP {required_status}, received {status}",
                        transient=True,
                        retry_after=retry_after,
                    )
                if required_length is not None:
                    self._validate_range(
                        response, warc_offset, required_length
                    )
                chunks: list[bytes] = []
                size = 0
                iterator = (
                    response.iter_raw()
                    if raw_stream
                    else response.iter_bytes()
                )
                for chunk in iterator:
                    size += len(chunk)
                    if size > maximum_bytes:
                        raise DownloadError(
                            f"response exceeds {maximum_bytes} bytes"
                        )
                    chunks.append(chunk)
                body = b"".join(chunks)
                if (
                    required_length is not None
                    and len(body) != required_length
                ):
                    raise DownloadError(
                        "range response has the wrong length",
                        transient=True,
                    )
                return (
                    body,
                    status,
                    response.headers.get("content-type"),
                    str(response.url),
                    retry_after,
                )

    def get(
        self,
        record_id: str,
        strategy: str,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        maximum_bytes: int = MAX_IMAGE_BYTES,
        cache: bool = False,
        archive_collection: str | None = None,
        archive_timestamp: str | None = None,
        warc_filename: str | None = None,
        warc_offset: int | None = None,
        warc_length: int | None = None,
        raw_stream: bool = False,
        required_status: int | None = None,
        required_length: int | None = None,
        attempts: int | None = None,
        follow_redirects: bool = True,
    ) -> HTTPResult:
        attempt_limit = DEFAULT_HTTP_ATTEMPTS if attempts is None else attempts
        if not 1 <= attempt_limit <= DEFAULT_HTTP_ATTEMPTS:
            raise ValueError(
                f"attempts must be between 1 and {DEFAULT_HTTP_ATTEMPTS}"
            )
        request_url, request_headers, request_key = self.request_identity(
            url,
            params,
            headers,
            raw_stream,
            follow_redirects,
        )
        audit = {
            "archive_collection": archive_collection,
            "archive_timestamp": archive_timestamp,
            "warc_filename": warc_filename,
            "warc_offset": warc_offset,
            "warc_length": warc_length,
        }
        if cache:
            cached = self._cached_result(
                record_id,
                strategy,
                request_url,
                request_key,
                maximum_bytes,
                required_status,
                required_length,
                audit,
            )
            if cached is not None:
                return cached
        circuit_error = self._commoncrawl_circuit_error(request_url)
        if circuit_error is not None:
            self._record_commoncrawl_activity(
                request_url, "circuit_deferred_requests"
            )
            raise circuit_error
        circuit_error = self._arquivo_circuit_error(request_url)
        if circuit_error is not None:
            self._record_arquivo_activity(
                request_url, "circuit_deferred_requests"
            )
            raise circuit_error
        last_error = DownloadError(
            "request did not start", transient=True
        )
        for attempt_number in range(1, attempt_limit + 1):
            try:
                self._record_commoncrawl_activity(
                    request_url, "network_requests"
                )
                self._record_arquivo_activity(
                    request_url, "network_requests"
                )
                body, status, mime, final_url, retry_after = self._request(
                    request_url,
                    request_headers,
                    maximum_bytes=maximum_bytes,
                    raw_stream=raw_stream,
                    required_status=required_status,
                    required_length=required_length,
                    warc_offset=warc_offset,
                    follow_redirects=follow_redirects,
                )
            except (httpx.TransportError, httpx.TimeoutException) as error:
                permanent_dns = _permanent_dns_failure(error)
                permanent_tls = _permanent_tls_failure(error)
                last_error = DownloadError(
                    str(error),
                    transient=not permanent_dns and not permanent_tls,
                    permanent_host_failure=permanent_dns,
                    service_failure_key=_transport_failure_key(
                        error, request_url
                    ),
                )
                retry_after = 0.0
            except DownloadError as error:
                last_error = error
                retry_after = error.retry_after
            else:
                attempt_id = self.state.log_attempt(
                    record_id,
                    strategy,
                    attempted_url=final_url,
                    http_status=status,
                    response_mime=mime,
                    response_size=len(body),
                    validation_result="downloaded_unvalidated",
                    request_key=request_key,
                    **audit,
                )
                if cache:
                    body_path = (
                        self.paths.http_cache / f"{request_key}.bin"
                    )
                    atomic_write_bytes(body_path, body)
                    self.state.put_http_cache(
                        request_key,
                        status,
                        final_url,
                        mime,
                        len(body),
                        hashlib.sha256(body).hexdigest(),
                        body_path,
                    )
                return HTTPResult(
                    body,
                    status,
                    mime,
                    final_url,
                    attempt_id,
                    request_key,
                    False,
                )
            self.state.log_attempt(
                record_id,
                strategy,
                attempted_url=request_url,
                http_status=last_error.status_code,
                exception=f"{type(last_error).__name__}: {last_error}",
                validation_result="request_failed",
                request_key=request_key,
                **audit,
            )
            if (
                not last_error.transient
                or attempt_number == attempt_limit
            ):
                break
            delay = max(
                retry_after,
                min(BACKOFF_CAP_SECONDS, 2 ** (attempt_number - 1))
                + random.uniform(0.0, 1.0),
            )
            time.sleep(delay)
        raise last_error


def _matching_donor(
    state: RecoveryState, record: Mapping[str, object]
) -> dict[str, object] | None:
    for field in (
        "laion_or_sample_index",
        "original_url",
        "normalized_url",
    ):
        value = _identity_text(record.get(field))
        if value is None:
            continue
        matches = [
            candidate
            for candidate in state.recovered_donors(
                field,
                value,
                exclude_record_id=str(record["record_id"]),
            )
            if recovery_artifacts_valid(state.paths, candidate)
        ]
        hashes = {str(candidate["sha256"]) for candidate in matches}
        if len(hashes) == 1:
            return matches[0]
    return None


def reuse_exact_group_duplicate(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    donor: Mapping[str, object],
    strategy: str,
) -> bool:
    """Reuse one named donor only after the complete target identity matches."""

    if _target_identity(record) != _target_identity(donor):
        return False
    if (
        str(donor.get("recovery_status") or "") not in RECOVERED_STATUS_VALUES
        or not recovery_artifacts_valid(paths, donor)
    ):
        return False
    fields = {
        key: donor.get(key)
        for key in (
            "resolved_url", "archive_timestamp", "archive_digest",
            "local_raw_path", "local_normalized_path", "sha256", "sha1",
            "perceptual_hash", "width", "height", "image_format",
        )
    }
    donor_id = str(donor["record_id"])
    state.mark_recovered(
        str(record["record_id"]),
        RecoveryStatus.RECOVERED_VERIFIED_DUPLICATE,
        recovery_method=f"verified_duplicate:donor={donor_id}",
        ambiguity_reason=None,
        **fields,
    )
    state.log_attempt(
        str(record["record_id"]),
        strategy,
        attempted_url=str(record.get("original_url") or ""),
        validation_result=f"verified_duplicate:donor={donor_id}",
        candidate_sha256=donor["sha256"],
    )
    return True


def reuse_duplicate(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    strategy: str,
) -> bool:
    donor = _matching_donor(state, record)
    if donor is None:
        return False
    fields = {
        key: donor.get(key)
        for key in (
            "resolved_url", "archive_timestamp", "archive_digest",
            "local_raw_path", "local_normalized_path", "sha256", "sha1",
            "perceptual_hash", "width", "height", "image_format",
        )
    }
    state.mark_recovered(
        str(record["record_id"]),
        RecoveryStatus.RECOVERED_VERIFIED_DUPLICATE,
        recovery_method=(
            f"verified_duplicate:donor={donor['record_id']}"
        ),
        ambiguity_reason=None,
        **fields,
    )
    state.log_attempt(
        str(record["record_id"]),
        strategy,
        attempted_url=str(record.get("original_url") or ""),
        validation_result=(
            f"verified_duplicate:donor={donor['record_id']}"
        ),
        candidate_sha256=donor["sha256"],
    )
    return True


def recover_direct(
    paths: WebsterPaths,
    state: RecoveryState,
    fetcher: HTTPFetcher,
    record: Mapping[str, object],
) -> StageResult:
    if reuse_duplicate(paths, state, record, "direct_duplicate_reuse"):
        return StageResult(
            StageOutcome.RECOVERED, "verified duplicate"
        )
    record_id = str(record["record_id"])
    variants = safe_url_variants(record.get("original_url"))
    if not variants:
        state.log_attempt(
            record_id,
            "direct_url",
            attempted_url=str(record.get("original_url") or ""),
            validation_result="invalid_or_missing_url",
        )
        return StageResult(
            StageOutcome.NOT_APPLICABLE, "no usable direct URL"
        )
    failures: list[str] = []
    retryable: list[str] = []
    unavailable_hosts: set[str] = set()
    for position, variant in enumerate(variants):
        hostname = (
            urllib.parse.urlsplit(variant).hostname or ""
        ).casefold()
        if hostname in unavailable_hosts:
            continue
        try:
            result = fetcher.get(
                record_id,
                "direct_url",
                variant,
                cache=True,
                attempts=fetcher.direct_attempts,
            )
            metadata, raw_path, normalized_path = store_image(
                paths,
                state,
                result.data,
                original_url=str(record.get("original_url") or variant),
                final_url=result.final_url,
            )
        except DownloadError as error:
            failures.append(f"{variant}: {error}")
            if error.transient:
                retryable.append(failures[-1])
            if error.permanent_host_failure:
                unavailable_hosts.add(hostname)
            continue
        except ImageValidationError as error:
            failures.append(f"{variant}: {error}")
            state.update_attempt(
                result.attempt_id,
                validation_result="invalid_image",
                exception=str(error),
            )
            continue
        method = (
            "direct_original"
            if position == 0
            else f"direct_safe_variant_{position}"
        )
        apply_image_resolution(
            paths,
            state,
            record,
            metadata,
            raw_path,
            normalized_path,
            status=RecoveryStatus.RECOVERED_EXACT_URL,
            method=method,
            resolved_url=result.final_url,
        )
        state.update_attempt(
            result.attempt_id,
            validation_result="valid_image",
            candidate_sha256=metadata.sha256,
        )
        return StageResult(StageOutcome.RECOVERED, method)
    if retryable:
        return StageResult.retryable_error(" | ".join(retryable)[-4000:])
    return StageResult(
        StageOutcome.MISS,
        json.dumps(failures, ensure_ascii=False)[-4000:],
    )


def resolve_candidates(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
) -> StageResult:
    del paths
    valid = [
        candidate
        for candidate in candidates
        if candidate_artifacts_valid(candidate)
    ]
    exact = [
        candidate for candidate in valid
        if not bool(candidate["source_page"])
    ]
    considered = exact or valid
    hashes = {str(candidate["sha256"]) for candidate in considered}
    if not considered:
        return StageResult(StageOutcome.MISS, "no validated candidates")
    if len(hashes) != 1:
        message = (
            f"{len(considered)} candidates disagree on "
            f"{len(hashes)} image hashes"
        )
        state.set_status(
            str(record["record_id"]),
            RecoveryStatus.AMBIGUOUS,
            reason=message,
        )
        return StageResult(StageOutcome.AMBIGUOUS, message)
    selected = sorted(
        considered,
        key=lambda item: (
            str(item.get("archive_timestamp") or ""),
            str(item.get("strategy") or ""),
        ),
    )[0]
    asset = state.get_asset(str(selected["sha256"]))
    if asset is None:
        raise StateConsistencyError(
            f"candidate asset is absent: {selected['sha256']}"
        )
    strategy = str(selected["strategy"])
    if strategy.startswith("wayback"):
        status = RecoveryStatus.RECOVERED_WAYBACK_EXACT
    elif strategy.startswith("arquivo"):
        status = RecoveryStatus.RECOVERED_ARQUIVO_EXACT
    elif strategy.startswith("commoncrawl"):
        status = RecoveryStatus.RECOVERED_COMMONCRAWL_EXACT
    elif strategy.startswith("official"):
        status = RecoveryStatus.RECOVERED_OFFICIAL_CACHE
    else:
        status = RecoveryStatus.RECOVERED_EXACT_URL
    state.mark_recovered(
        str(record["record_id"]),
        status,
        recovery_method=strategy,
        resolved_url=selected.get("resolved_url"),
        archive_timestamp=selected.get("archive_timestamp"),
        archive_digest=selected.get("archive_digest"),
        local_raw_path=asset["local_raw_path"],
        local_normalized_path=asset["local_normalized_path"],
        sha256=asset["sha256"],
        sha1=asset["sha1"],
        perceptual_hash=asset["perceptual_hash"],
        width=asset["width"],
        height=asset["height"],
        image_format=asset["image_format"],
        ambiguity_reason=None,
    )
    return StageResult(StageOutcome.RECOVERED, status.value)


StageFunction = Callable[
    [WebsterPaths, RecoveryState, HTTPFetcher, Mapping[str, object]],
    StageResult,
]


class _CommonCrawlRecoverySession:
    """Fetch and parse the crawl catalog once, then reuse it for all targets."""

    def __init__(self) -> None:
        self._indexes: list[dict[str, str]] | None = None

    def __call__(
        self,
        paths: WebsterPaths,
        state: RecoveryState,
        fetcher: HTTPFetcher,
        record: Mapping[str, object],
    ) -> StageResult:
        if self._indexes is None:
            self._indexes = commoncrawl_indexes(
                fetcher, state, str(record["record_id"])
            )
        return recover_commoncrawl(
            paths,
            state,
            fetcher,
            record,
            indexes=self._indexes,
        )


class RecoveryEngine:
    """Run each independent source once and persist every outcome."""

    def __init__(
        self,
        paths: WebsterPaths,
        state: RecoveryState,
        source_manifest: pd.DataFrame,
        fetcher: HTTPFetcher,
        *,
        direct_stage: StageFunction = recover_direct,
        mirror_stage: StageFunction | None = None,
        wayback_stage: StageFunction = recover_wayback,
        arquivo_stage: StageFunction = recover_arquivo,
        commoncrawl_stage: StageFunction = recover_commoncrawl,
        show_progress: bool = True,
        direct_workers: int = DEFAULT_DIRECT_WORKERS,
    ) -> None:
        self.paths = paths
        self.state = state
        self.source_manifest = source_manifest
        self.fetcher = fetcher
        self.direct_stage = direct_stage
        self.mirror_stage = mirror_stage
        self.wayback_stage = wayback_stage
        self.arquivo_stage = arquivo_stage
        self.commoncrawl_stage = (
            _CommonCrawlRecoverySession()
            if commoncrawl_stage is recover_commoncrawl
            else commoncrawl_stage
        )
        self.show_progress = show_progress
        self._counts = _RecoveryCounts(state, source_manifest)
        self._service_failures: dict[
            tuple[str, RecoveryStage], tuple[str, bool]
        ] = {}
        if not 1 <= direct_workers <= MAX_DIRECT_WORKERS:
            raise ValueError(
                f"direct_workers must be between 1 and {MAX_DIRECT_WORKERS}"
            )
        self.direct_workers = direct_workers

    def _verify_downloaded_artifacts(self) -> None:
        records = self.state.records(statuses=RECOVERED_STATUS_VALUES)
        reason = (
            "saved recovered image artifacts are missing, corrupt, "
            "or inconsistent"
        )
        for record in _progress(
            records,
            total=len(records),
            description="[Webster 1/10] Verify downloaded files",
            enabled=self.show_progress,
            after=self._counts.refresh,
            postfix=self._counts.format,
        ):
            if recovery_artifacts_available(self.paths, record):
                continue
            self.state.reactivate_invalid_recovered_artifacts(
                str(record["record_id"]),
                reason=reason,
            )

    def _reactivate_legacy_unresolved(self) -> None:
        metadata_getter = getattr(self.state, "get_run_metadata", None)
        strategies = (
            (
                "ground_truth_mirror_strategy_version",
                MIRROR_STRATEGY_VERSION,
                RecoveryStage.GROUND_TRUTH_MIRROR,
                "ground-truth mirror",
            ),
            (
                "arquivo_strategy_version",
                ARQUIVO_STRATEGY_VERSION,
                RecoveryStage.ARQUIVO,
                "Arquivo.pt recovery",
            ),
        )
        for metadata_key, current_version, stage, label in strategies:
            stored_value = (
                metadata_getter(metadata_key, 0)
                if callable(metadata_getter)
                else 0
            )
            try:
                stored_version = int(stored_value)
            except (TypeError, ValueError) as error:
                raise StateConsistencyError(
                    f"{metadata_key} metadata is invalid"
                ) from error
            if stored_version >= current_version:
                continue
            reason = (
                f"{label} strategy upgraded from v{stored_version} "
                f"to v{current_version}"
            )
            for record in self.state.records(
                statuses={RecoveryStatus.UNRESOLVED}
            ):
                reactivated = self.state.reactivate_unresolved_for_stage(
                    str(record["record_id"]),
                    stage,
                    reason=reason,
                    strategy_version=current_version,
                )
                if reactivated:
                    self._counts.refresh(record)

    def _should_run(
        self, record_id: str, stage: RecoveryStage
    ) -> bool:
        result = self.state.stage_result(record_id, stage)
        return result is None or result.outcome in {
            StageOutcome.RUNNING,
            StageOutcome.RETRYABLE_ERROR,
        }

    def _active_records_for(
        self, stage: RecoveryStage
    ) -> list[dict[str, object]]:
        return [
            record
            for record in self.state.records(
                statuses=ACTIVE_STATUS_VALUES
            )
            if self._should_run(str(record["record_id"]), stage)
        ]

    def _active_target_groups_for(
        self, stage: RecoveryStage
    ) -> list[tuple[dict[str, object], ...]]:
        return _group_target_records(self._active_records_for(stage))

    def _refresh_group(
        self, group: Sequence[Mapping[str, object]]
    ) -> None:
        for record in group:
            self._counts.refresh(record)

    @staticmethod
    def _shared_stage_result(
        result: StageResult, representative_id: str
    ) -> StageResult:
        detail = result.message or result.outcome.value
        message = (
            f"shared exact-target result from {representative_id}: {detail}"
        )[-4000:]
        return StageResult(result.outcome, message, result.retryable)

    def _share_archive_candidates(
        self,
        representative_id: str,
        alias: Mapping[str, object],
        stage: RecoveryStage,
    ) -> int:
        prefixes = {
            RecoveryStage.WAYBACK: "wayback",
            RecoveryStage.ARQUIVO: "arquivo",
            RecoveryStage.COMMON_CRAWL: "commoncrawl",
        }
        prefix = prefixes.get(stage)
        if prefix is None:
            return 0
        candidates = [
            candidate
            for candidate in self.state.candidates(
                representative_id, prefix=prefix
            )
            if candidate_artifacts_valid(candidate)
        ]
        alias_id = str(alias["record_id"])
        copied = 0
        for source in candidates:
            asset = self.state.get_asset(str(source["sha256"]))
            if asset is None:
                continue
            destination = (
                self.paths.candidates
                / alias_id
                / (
                    f"{source['sha256']}."
                    f"{asset['extension']}"
                )
            )
            reference = create_reference(
                Path(str(source["raw_path"])), destination
            )
            try:
                source_metadata = json.loads(
                    str(source.get("metadata_json") or "{}")
                )
            except json.JSONDecodeError:
                source_metadata = {}
            candidate = ArchiveCandidate(
                record_id=alias_id,
                sha256=str(source["sha256"]),
                perceptual_hash=str(source["perceptual_hash"]),
                raw_path=str(source["raw_path"]),
                normalized_path=str(source["normalized_path"]),
                candidate_path=str(reference),
                strategy=str(source["strategy"]),
                resolved_url=str(source.get("resolved_url") or ""),
                archive_timestamp=(
                    str(source["archive_timestamp"])
                    if source.get("archive_timestamp")
                    else None
                ),
                archive_digest=(
                    str(source["archive_digest"])
                    if source.get("archive_digest")
                    else None
                ),
                source_page=bool(source["source_page"]),
                response_mime=(
                    str(source["response_mime"])
                    if source.get("response_mime")
                    else None
                ),
            )
            self.state.add_candidate(
                candidate,
                {
                    "shared_exact_target_record_id": representative_id,
                    "shared_source_candidate_id": source["candidate_id"],
                    "source_candidate_metadata": source_metadata,
                },
            )
            copied += 1
        return copied

    def _resolve_exact_archive_group(
        self,
        group: Sequence[Mapping[str, object]],
    ) -> None:
        for supplied in group:
            record_id = str(supplied["record_id"])
            current = self.state.get_record(record_id) or dict(supplied)
            if (
                str(current.get("recovery_status") or "")
                not in ACTIVE_STATUS_VALUES
            ):
                continue
            candidates = [
                candidate
                for candidate in self.state.candidates(record_id)
                if candidate_artifacts_valid(candidate)
            ]
            if not any(
                not bool(candidate["source_page"])
                for candidate in candidates
            ):
                continue
            resolve_candidates(
                self.paths, self.state, current, candidates
            )

    def _fan_out_group_result(
        self,
        group: Sequence[dict[str, object]],
        stage: RecoveryStage,
        function: StageFunction,
        representative_result: StageResult,
    ) -> None:
        if len(group) < 2:
            return
        representative_id = str(group[0]["record_id"])
        representative = (
            self.state.get_record(representative_id) or group[0]
        )
        representative_recovered = (
            str(representative.get("recovery_status") or "")
            in RECOVERED_STATUS_VALUES
        )
        shareable = representative_result.outcome in {
            StageOutcome.MISS,
            StageOutcome.NOT_APPLICABLE,
            StageOutcome.RETRYABLE_ERROR,
        }
        for alias in group[1:]:
            alias_id = str(alias["record_id"])
            current = self.state.get_record(alias_id) or alias
            if (
                str(current.get("recovery_status") or "")
                not in ACTIVE_STATUS_VALUES
            ):
                continue
            if representative_recovered:
                self.state.begin_stage(alias_id, stage)
                reused = reuse_exact_group_duplicate(
                    self.paths,
                    self.state,
                    current,
                    representative,
                    f"{stage.value}_group_duplicate_reuse",
                )
                if reused:
                    self.state.finish_stage(
                        alias_id,
                        stage,
                        StageResult(
                            StageOutcome.RECOVERED,
                            (
                                "verified exact-target donor "
                                f"{representative_id}"
                            ),
                        ),
                    )
                    continue
                self.state.finish_stage(
                    alias_id,
                    stage,
                    StageResult.retryable_error(
                        "exact-target representative recovered but "
                        "its artifacts were unavailable for reuse"
                    ),
                )
            elif (
                representative_result.outcome is StageOutcome.RECOVERED
                and stage
                in {
                    RecoveryStage.WAYBACK,
                    RecoveryStage.ARQUIVO,
                    RecoveryStage.COMMON_CRAWL,
                }
            ):
                self.state.begin_stage(alias_id, stage)
                copied = self._share_archive_candidates(
                    representative_id, current, stage
                )
                if copied:
                    shared = StageResult(
                        StageOutcome.RECOVERED,
                        (
                            f"shared {copied} exact-target candidates "
                            f"from {representative_id}"
                        ),
                    )
                else:
                    shared = StageResult.retryable_error(
                        "exact-target representative collected candidates "
                        "but none could be verified for reuse"
                    )
                self.state.finish_stage(alias_id, stage, shared)
                continue
            elif shareable:
                self.state.begin_stage(alias_id, stage)
                self.state.finish_stage(
                    alias_id,
                    stage,
                    self._shared_stage_result(
                        representative_result, representative_id
                    ),
                )
                continue
            self._run_independent_stage(
                current, stage, function
            )

    def _run_independent_stage(
        self,
        record: Mapping[str, object],
        stage: RecoveryStage,
        function: StageFunction,
    ) -> StageResult:
        record_id = str(record["record_id"])
        self._service_failures.pop((record_id, stage), None)
        current = self.state.get_record(record_id) or dict(record)
        if str(current["recovery_status"]) not in ACTIVE_STATUS_VALUES:
            return StageResult(
                StageOutcome.ALREADY_RECOVERED, "record is terminal"
            )
        self.state.begin_stage(record_id, stage)
        try:
            result = function(
                self.paths, self.state, self.fetcher, current
            )
        except DownloadError as error:
            if (
                stage in {
                    RecoveryStage.ARQUIVO,
                    RecoveryStage.COMMON_CRAWL,
                }
                and error.service_failure_key is not None
            ):
                self._service_failures[(record_id, stage)] = (
                    error.service_failure_key,
                    error.deferred_by_circuit,
                )
            message = f"{type(error).__name__}: {error}"
            if error.transient or stage is RecoveryStage.COMMON_CRAWL:
                result = StageResult.retryable_error(message)
            else:
                result = StageResult(StageOutcome.MISS, message)
        except RetryableStageError as error:
            result = StageResult.retryable_error(
                f"{type(error).__name__}: {error}"
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.state.log_attempt(
                record_id,
                f"{stage.value}_programming_error",
                exception=message,
                validation_result="fatal_programming_error",
            )
            self.state.finish_stage(
                record_id, stage, StageResult.retryable_error(message)
            )
            raise
        self.state.finish_stage(record_id, stage, result)
        return result

    def stage_exact_reuse(self) -> None:
        records = self._active_records_for(RecoveryStage.EXACT_REUSE)
        for record in _progress(
            records,
            total=len(records),
            description="[Webster 1/10] Local cache reuse",
            enabled=self.show_progress,
            after=self._counts.refresh,
            postfix=self._counts.format,
        ):
            record_id = str(record["record_id"])
            self.state.begin_stage(
                record_id, RecoveryStage.EXACT_REUSE
            )
            recovered = reuse_duplicate(
                self.paths,
                self.state,
                record,
                "exact_duplicate_reuse",
            )
            self.state.finish_stage(
                record_id,
                RecoveryStage.EXACT_REUSE,
                StageResult(
                    (
                        StageOutcome.RECOVERED
                        if recovered
                        else StageOutcome.MISS
                    ),
                    (
                        "verified duplicate"
                        if recovered
                        else "no verified donor"
                    ),
                ),
            )

    def stage_official_assets(self) -> None:
        records = self._active_records_for(
            RecoveryStage.OFFICIAL_ASSETS
        )
        if not records:
            for _ in _progress(
                records,
                total=0,
                description="[Webster 2/10] Official assets",
                enabled=self.show_progress,
                postfix=self._counts.format,
            ):
                pass
            return
        for record in records:
            self.state.begin_stage(
                str(record["record_id"]),
                RecoveryStage.OFFICIAL_ASSETS,
            )
        try:
            catalog = inspect_official_tree(
                self.paths, self.state, self.fetcher
            )
            active_ids = {str(record["record_id"]) for record in records}
            active_manifest = self.source_manifest.loc[
                self.source_manifest["record_id"].astype(str).isin(active_ids)
            ]
            map_and_cache_official_assets(
                self.paths,
                self.state,
                self.fetcher,
                active_manifest,
                catalog,
                show_progress=self.show_progress,
                progress_postfix=self._counts.format,
            )
        except (DownloadError, RetryableStageError) as error:
            result = StageResult.retryable_error(
                f"{type(error).__name__}: {error}"
            )
            for record in records:
                self.state.finish_stage(
                    str(record["record_id"]),
                    RecoveryStage.OFFICIAL_ASSETS,
                    result,
                )
            return
        for supplied in _progress(
            records,
            total=len(records),
            description="[Webster 2/10] Official resolution",
            enabled=self.show_progress,
            after=self._counts.refresh,
            postfix=self._counts.format,
        ):
            record_id = str(supplied["record_id"])
            current = self.state.get_record(record_id) or supplied
            try:
                result = recover_official(
                    self.paths, self.state, current
                )
            except ImageValidationError as error:
                result = StageResult(
                    StageOutcome.MISS,
                    f"official paired source is invalid: {error}",
                )
            self.state.finish_stage(
                record_id, RecoveryStage.OFFICIAL_ASSETS, result
            )

    def stage_direct(self) -> None:
        groups = self._active_target_groups_for(RecoveryStage.DIRECT)
        by_representative = {
            str(group[0]["record_id"]): group for group in groups
        }
        representatives = _interleave_direct_records(
            [group[0] for group in groups]
        )
        ordered_groups = [
            by_representative[str(record["record_id"])]
            for record in representatives
        ]
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.direct_workers
        )
        futures: list[concurrent.futures.Future[StageResult]] = []
        future_groups: dict[
            concurrent.futures.Future[StageResult],
            tuple[dict[str, object], ...],
        ] = {}
        try:
            future_groups = {
                executor.submit(
                    self._run_independent_stage,
                    group[0],
                    RecoveryStage.DIRECT,
                    self.direct_stage,
                ): group
                for group in ordered_groups
            }
            futures = list(future_groups)
            completed = concurrent.futures.as_completed(futures)
            for future in _progress(
                completed,
                total=len(futures),
                description="[Webster 3/10] Direct URLs (failed only)",
                unit="target",
                enabled=self.show_progress,
                after=lambda item: self._refresh_group(
                    future_groups[item]
                ),
                postfix=self._counts.format,
            ):
                group = future_groups[future]
                result = future.result()
                self._fan_out_group_result(
                    group,
                    RecoveryStage.DIRECT,
                    self.direct_stage,
                    result,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    def stage_ground_truth_mirror(self) -> None:
        """Audit the pinned mirror once, then import only failed exact targets."""

        stage = RecoveryStage.GROUND_TRUTH_MIRROR
        groups = self._active_target_groups_for(stage)
        if not groups:
            for _ in _progress(
                groups,
                total=0,
                description="[Webster 4/10] Verified mirror (failed only)",
                unit="target",
                enabled=self.show_progress,
                postfix=self._counts.format,
            ):
                pass
            return

        function = self.mirror_stage
        if function is None:
            try:
                catalog, _audit = prepare_verified_mirror(
                    self.paths,
                    self.state,
                    self.source_manifest,
                    show_progress=self.show_progress,
                    progress_postfix=self._counts.format,
                )
            except RetryableStageError as error:
                failure = StageResult.retryable_error(
                    f"{type(error).__name__}: {error}"
                )

                def unavailable(
                    unused_paths: WebsterPaths,
                    unused_state: RecoveryState,
                    unused_fetcher: HTTPFetcher,
                    unused_record: Mapping[str, object],
                ) -> StageResult:
                    del unused_paths, unused_state, unused_fetcher, unused_record
                    return failure

                for group in _progress(
                    groups,
                    total=len(groups),
                    description="[Webster 4/10] Verified mirror (failed only)",
                    unit="target",
                    enabled=self.show_progress,
                    after=self._refresh_group,
                    postfix=self._counts.format,
                ):
                    representative = group[0]
                    record_id = str(representative["record_id"])
                    self.state.begin_stage(record_id, stage)
                    self.state.finish_stage(record_id, stage, failure)
                    self._fan_out_group_result(
                        group, stage, unavailable, failure
                    )
                return
            rows_by_identity = catalog.by_identity()

            def audited_recovery(
                supplied_paths: WebsterPaths,
                supplied_state: RecoveryState,
                unused_fetcher: HTTPFetcher,
                record: Mapping[str, object],
            ) -> StageResult:
                del unused_fetcher
                try:
                    identity = mirror_record_identity(record)
                except StateConsistencyError:
                    return StageResult(
                        StageOutcome.NOT_APPLICABLE,
                        "record has no exact mirror identity",
                    )
                row = rows_by_identity.get(identity)
                if row is None:
                    return StageResult(
                        StageOutcome.NOT_APPLICABLE,
                        "record is outside the pinned mirror intersection",
                    )
                return recover_mirror_row(
                    supplied_paths, supplied_state, record, row
                )

            function = audited_recovery

        retryable = False
        for group in _progress(
            groups,
            total=len(groups),
            description="[Webster 4/10] Verified mirror (failed only)",
            unit="target",
            enabled=self.show_progress,
            after=self._refresh_group,
            postfix=self._counts.format,
        ):
            result = self._run_independent_stage(
                group[0], stage, function
            )
            retryable = retryable or result.retryable
            self._fan_out_group_result(group, stage, function, result)
        metadata_setter = getattr(self.state, "set_run_metadata", None)
        if not retryable and callable(metadata_setter):
            metadata_setter(
                "ground_truth_mirror_strategy_version",
                MIRROR_STRATEGY_VERSION,
            )

    def stage_archive(
        self,
        stage: RecoveryStage,
        function: StageFunction,
        description: str,
    ) -> None:
        groups = self._active_target_groups_for(stage)
        for group in _progress(
            groups,
            total=len(groups),
            description=f"{description} (failed only)",
            unit="target",
            enabled=self.show_progress,
            after=self._refresh_group,
            postfix=self._counts.format,
        ):
            result = self._run_independent_stage(
                group[0], stage, function
            )
            self._fan_out_group_result(
                group, stage, function, result
            )
            self._resolve_exact_archive_group(group)

    def stage_arquivo(self) -> None:
        """Try cached and live Arquivo.pt data for unique failed targets."""

        stage = RecoveryStage.ARQUIVO
        groups = self._active_target_groups_for(stage)
        reset_circuits = getattr(
            self.fetcher, "reset_arquivo_circuits", None
        )
        if callable(reset_circuits):
            reset_circuits()
        reset_activity = getattr(
            self.fetcher, "reset_arquivo_activity", None
        )
        if callable(reset_activity):
            reset_activity()
        last_failure: str | None = None
        consecutive_failures = 0
        open_hosts: set[str] = set()
        deferred_targets = 0
        duplicate_rows = sum(len(group) - 1 for group in groups)

        def progress_postfix() -> str:
            base = self._counts.format()
            activity_getter = getattr(
                self.fetcher, "arquivo_activity", None
            )
            activity = (
                activity_getter() if callable(activity_getter) else {}
            )
            network = int(activity.get("network_requests", 0))
            cache_hits = int(activity.get("cache_hits", 0))
            details = (
                f"{base} | Arquivo net {network} | cache {cache_hits} | "
                f"deferred targets {deferred_targets}"
            )
            if duplicate_rows:
                details += f" | dedup rows {duplicate_rows}"
            if open_hosts:
                details += f" | circuit open {len(open_hosts)}"
            return details

        for group in _progress(
            groups,
            total=len(groups),
            description="[Webster 6/10] Arquivo.pt (failed only)",
            unit="target",
            enabled=self.show_progress,
            after=self._refresh_group,
            postfix=progress_postfix,
        ):
            representative = group[0]
            record_id = str(representative["record_id"])
            result = self._run_independent_stage(
                representative, stage, self.arquivo_stage
            )
            self._fan_out_group_result(
                group, stage, self.arquivo_stage, result
            )
            self._resolve_exact_archive_group(group)
            failure_info = self._service_failures.pop(
                (record_id, stage), None
            )
            if failure_info is None:
                last_failure = None
                consecutive_failures = 0
                continue
            failure, deferred = failure_info
            if deferred:
                deferred_targets += 1
                continue
            if failure == last_failure:
                consecutive_failures += 1
            else:
                last_failure = failure
                consecutive_failures = 1
            if (
                consecutive_failures
                >= ARQUIVO_CIRCUIT_BREAKER_THRESHOLD
            ):
                open_circuit = getattr(
                    self.fetcher, "open_arquivo_circuit", None
                )
                if callable(open_circuit):
                    opened_host = open_circuit(failure)
                    if opened_host is not None:
                        open_hosts.add(str(opened_host))
        metadata_setter = getattr(
            self.state, "set_run_metadata", None
        )
        if callable(metadata_setter):
            metadata_setter(
                "arquivo_strategy_version",
                ARQUIVO_STRATEGY_VERSION,
            )

    def stage_commoncrawl(self) -> None:
        """Retry only unique failed targets and expose actual HTTP activity."""

        stage = RecoveryStage.COMMON_CRAWL
        groups = self._active_target_groups_for(stage)
        reset_circuits = getattr(
            self.fetcher, "reset_commoncrawl_circuits", None
        )
        if callable(reset_circuits):
            reset_circuits()
        reset_activity = getattr(
            self.fetcher, "reset_commoncrawl_activity", None
        )
        if callable(reset_activity):
            reset_activity()
        last_failure: str | None = None
        consecutive_failures = 0
        open_hosts: set[str] = set()
        deferred_targets = 0
        duplicate_rows = sum(len(group) - 1 for group in groups)

        def progress_postfix() -> str:
            base = self._counts.format()
            activity_getter = getattr(
                self.fetcher, "commoncrawl_activity", None
            )
            activity = (
                activity_getter() if callable(activity_getter) else {}
            )
            network = int(activity.get("network_requests", 0))
            cache_hits = int(activity.get("cache_hits", 0))
            details = (
                f"{base} | CC net {network} | cache {cache_hits} | "
                f"deferred targets {deferred_targets}"
            )
            if duplicate_rows:
                details += f" | dedup rows {duplicate_rows}"
            if open_hosts:
                details += f" | circuit open {len(open_hosts)}"
            return details

        for group in _progress(
            groups,
            total=len(groups),
            description="[Webster 7/10] Common Crawl (failed only)",
            unit="target",
            enabled=self.show_progress,
            after=self._refresh_group,
            postfix=progress_postfix,
        ):
            representative = group[0]
            record_id = str(representative["record_id"])
            result = self._run_independent_stage(
                representative, stage, self.commoncrawl_stage
            )
            self._fan_out_group_result(
                group, stage, self.commoncrawl_stage, result
            )
            self._resolve_exact_archive_group(group)
            failure_info = self._service_failures.pop(
                (record_id, stage), None
            )
            if failure_info is None:
                last_failure = None
                consecutive_failures = 0
                continue
            failure, deferred = failure_info
            if deferred:
                deferred_targets += 1
                continue
            if failure == last_failure:
                consecutive_failures += 1
            else:
                last_failure = failure
                consecutive_failures = 1
            if (
                consecutive_failures
                >= COMMONCRAWL_CIRCUIT_BREAKER_THRESHOLD
            ):
                open_circuit = getattr(
                    self.fetcher, "open_commoncrawl_circuit", None
                )
                if callable(open_circuit):
                    opened_host = open_circuit(failure)
                    if opened_host is not None:
                        open_hosts.add(str(opened_host))

    def _final_result(
        self, record: Mapping[str, object]
    ) -> StageResult:
        if reuse_duplicate(
            self.paths,
            self.state,
            record,
            "final_duplicate_reuse",
        ):
            return StageResult(
                StageOutcome.RECOVERED, "verified duplicate"
            )
        record_id = str(record["record_id"])
        results = self.state.stage_results(record_id)
        candidates = self.state.candidates(record_id)
        source_stages = (
            RecoveryStage.EXACT_REUSE,
            RecoveryStage.OFFICIAL_ASSETS,
            RecoveryStage.DIRECT,
            RecoveryStage.GROUND_TRUTH_MIRROR,
            RecoveryStage.WAYBACK,
            RecoveryStage.ARQUIVO,
            RecoveryStage.COMMON_CRAWL,
        )
        retryable = any(
            stage not in results
            or results[stage].outcome
            in {StageOutcome.RUNNING, StageOutcome.RETRYABLE_ERROR}
            for stage in source_stages
        )
        if candidates:
            resolved = resolve_candidates(
                self.paths, self.state, record, candidates
            )
            if (
                resolved.outcome is StageOutcome.RECOVERED
                or not retryable
            ):
                return resolved
        if retryable:
            return StageResult.retryable_error(
                "one or more recovery sources must be retried"
            )
        has_provenance = (
            is_valid_http_url(record.get("original_url"))
            or is_valid_http_url(record.get("source_page_url"))
            or bool(str(record.get("laion_or_sample_index") or ""))
        )
        if not has_provenance:
            message = "no usable source URL or sample identifier"
            self.state.set_status(
                record_id,
                RecoveryStatus.INVALID_METADATA,
                reason=message,
            )
            return StageResult(
                StageOutcome.INVALID_METADATA, message
            )
        message = "all official, direct, audited-mirror, and archive sources missed"
        self.state.set_status(
            record_id, RecoveryStatus.UNRESOLVED, reason=message
        )
        return StageResult(StageOutcome.MISS, message)

    def stage_final_resolution(self) -> None:
        records = self._active_records_for(RecoveryStage.FINAL_RESOLUTION)
        for supplied in _progress(
            records,
            total=len(records),
            description="[Webster 8/10] Final resolution",
            enabled=self.show_progress,
            after=self._counts.refresh,
            postfix=self._counts.format,
        ):
            record_id = str(supplied["record_id"])
            self.state.begin_stage(
                record_id, RecoveryStage.FINAL_RESOLUTION
            )
            record = self.state.get_record(record_id) or supplied
            result = self._final_result(record)
            self.state.finish_stage(
                record_id, RecoveryStage.FINAL_RESOLUTION, result
            )

    def run(self) -> None:
        self._verify_downloaded_artifacts()
        self._reactivate_legacy_unresolved()
        self.stage_exact_reuse()
        self.stage_official_assets()
        self.stage_direct()
        self.stage_ground_truth_mirror()
        self.stage_archive(
            RecoveryStage.WAYBACK,
            self.wayback_stage,
            "[Webster 5/10] Wayback",
        )
        self.stage_arquivo()
        # Common Crawl asks clients not to issue concurrent CDX requests from
        # one IP. Keep this stage sequential and use a resumable outage breaker
        # instead of multiplying load while the public index is unavailable.
        self.stage_commoncrawl()
        self.stage_final_resolution()
        pending = self.state.records(
            statuses={RecoveryStatus.PENDING}
        )
        if pending:
            raise StateConsistencyError(
                "recovery left pending records: "
                + ", ".join(
                    str(record["record_id"]) for record in pending
                )
            )


def current_manifest(
    source_manifest: pd.DataFrame, state: RecoveryState
) -> pd.DataFrame:
    recovery = state.records_frame().set_index("record_id")
    output = source_manifest.copy(deep=True).set_index("record_id")
    fields = [
        column
        for column in recovery.columns
        if column not in {"model_name", "source_row_number", "metadata_json"}
    ]
    for field in fields:
        output[field] = recovery[field]
    return output.reset_index()


def write_current_reports(
    paths: WebsterPaths,
    state: RecoveryState,
    source_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, DownloadSummary]:
    manifest = current_manifest(source_manifest, state)
    manifest_parquet = paths.manifests / "webster_all.parquet"
    manifest_csv = paths.manifests / "webster_all.csv"
    retryable_csv = paths.manifests / "retryable.csv"
    summary_json = paths.manifests / "recovery_summary.json"
    atomic_write_frame_parquet(manifest, manifest_parquet)
    atomic_write_frame_csv(manifest, manifest_csv)
    atomic_write_frame_parquet(
        state.attempts_frame(),
        paths.manifests / "recovery_attempts.parquet",
    )
    atomic_write_frame_csv(
        state.attempts_frame(),
        paths.manifests / "recovery_attempts.csv",
    )
    atomic_write_frame_parquet(
        state.stage_state_frame(),
        paths.manifests / "recovery_stages.parquet",
    )
    retryable = manifest.loc[
        manifest["recovery_status"].astype(str).isin(ACTIVE_STATUS_VALUES)
    ]
    atomic_write_frame_csv(retryable, retryable_csv)
    counts = {
        status: int(count)
        for status, count in manifest["recovery_status"]
        .astype(str)
        .value_counts()
        .sort_index()
        .items()
    }
    unknown = set(counts) - ALL_STATUS_VALUES
    if unknown:
        raise StateConsistencyError(
            f"report contains unknown statuses: {sorted(unknown)}"
        )
    retryable_ids = tuple(retryable["record_id"].astype(str))
    exit_code = 2 if retryable_ids else 0
    atomic_write_json(
        summary_json,
        {
            "schema": "webster_recovery_summary_v2",
            "total_records": len(manifest),
            "status_counts": counts,
            "retryable_record_ids": retryable_ids,
            "exit_code": exit_code,
        },
    )
    return manifest, DownloadSummary(
        root=paths.root,
        total_records=len(manifest),
        status_counts=counts,
        retryable_record_ids=retryable_ids,
        manifest_parquet=manifest_parquet,
        manifest_csv=manifest_csv,
        summary_json=summary_json,
        retryable_csv=retryable_csv,
        exit_code=exit_code,
    )


def recover_webster_dataset(
    root: str | Path,
    *,
    direct_workers: int = DEFAULT_DIRECT_WORKERS,
    direct_attempts: int = DEFAULT_DIRECT_ATTEMPTS,
    per_host_concurrency: int = DEFAULT_PER_HOST_CONCURRENCY,
) -> DownloadSummary:
    _validate_direct_tuning(
        direct_workers, direct_attempts, per_host_concurrency
    )
    paths = WebsterPaths.from_root(root)
    paths.create()
    with RecoveryState(paths) as state:
        manifest = acquire_and_build_manifest(paths, state)
        with HTTPFetcher(
            paths,
            state,
            direct_workers=direct_workers,
            direct_attempts=direct_attempts,
            per_host_concurrency=per_host_concurrency,
        ) as fetcher:
            RecoveryEngine(
                paths,
                state,
                manifest,
                fetcher,
                direct_workers=direct_workers,
            ).run()
        _, summary = write_current_reports(paths, state, manifest)
    return summary


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    recovery: DownloadSummary
    organization: OrganizationSummary
    inspection: InspectionSummary

    @property
    def exit_code(self) -> int:
        if self.inspection.exit_code == 1:
            return 1
        return max(self.recovery.exit_code, self.inspection.exit_code)


def prepare_webster_dataset(
    root: str | Path,
    *,
    direct_workers: int = DEFAULT_DIRECT_WORKERS,
    direct_attempts: int = DEFAULT_DIRECT_ATTEMPTS,
    per_host_concurrency: int = DEFAULT_PER_HOST_CONCURRENCY,
) -> PreparationSummary:
    paths = WebsterPaths.from_root(root)
    recovery = recover_webster_dataset(
        paths.root,
        direct_workers=direct_workers,
        direct_attempts=direct_attempts,
        per_host_concurrency=per_host_concurrency,
    )
    manifest = pd.read_parquet(recovery.manifest_parquet)
    postfix = _manifest_progress_postfix(manifest)
    organization = organize_model_views(
        paths,
        manifest,
        show_progress=True,
        progress_postfix=lambda: postfix,
    )
    inspection = inspect_webster_dataset(
        paths.root,
        show_progress=True,
        progress_postfix=lambda: postfix,
    )
    return PreparationSummary(recovery, organization, inspection)


__all__ = [
    "COMMONCRAWL_CIRCUIT_BREAKER_THRESHOLD",
    "DEFAULT_DIRECT_ATTEMPTS",
    "DEFAULT_DIRECT_WORKERS",
    "DEFAULT_PER_HOST_CONCURRENCY",
    "MAX_DIRECT_ATTEMPTS",
    "MAX_DIRECT_WORKERS",
    "MAX_PER_HOST_CONCURRENCY",
    "HTTPFetcher",
    "PreparationSummary",
    "RateLimiter",
    "RecoveryEngine",
    "current_manifest",
    "prepare_webster_dataset",
    "recover_direct",
    "recover_webster_dataset",
    "resolve_candidates",
    "reuse_duplicate",
    "write_current_reports",
]
