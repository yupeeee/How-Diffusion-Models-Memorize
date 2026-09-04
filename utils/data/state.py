"""Current-schema-only SQLite state for resumable Webster recovery.

New databases are created with the complete recovery schema. Existing databases
are validated before use. Extra tables or columns are left untouched, while an
incomplete schema is rejected without writes.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from utils.common.io import canonical_json, json_value, utc_now

from .webster import (
    ACTIVE_STATUS_VALUES,
    ALL_STATUS_VALUES,
    RECOVERED_STATUS_VALUES,
    STAGE_ORDER,
    TERMINAL_STATUS_VALUES,
    ArchiveCandidate,
    ImageMetadata,
    RecoveryStage,
    RecoveryStatus,
    StageOutcome,
    StageResult,
    StateConsistencyError,
    StateSchemaError,
    WebsterPaths,
)


_SCHEMA = (
    """CREATE TABLE records (
        record_id TEXT PRIMARY KEY, model_name TEXT NOT NULL,
        source_row_number INTEGER NOT NULL, prompt_raw TEXT, overfit_type TEXT,
        laion_or_sample_index TEXT, original_url TEXT, normalized_url TEXT,
        source_page_url TEXT, recovery_status TEXT NOT NULL DEFAULT 'pending',
        recovery_method TEXT, resolved_url TEXT, archive_timestamp TEXT,
        archive_digest TEXT, local_raw_path TEXT, local_normalized_path TEXT,
        sha256 TEXT, sha1 TEXT, perceptual_hash TEXT, width INTEGER,
        height INTEGER, image_format TEXT,
        auxiliary_asset_paths TEXT NOT NULL DEFAULT '[]',
        number_of_attempts INTEGER NOT NULL DEFAULT 0, attempt_log_path TEXT,
        ambiguity_reason TEXT, failure_reason TEXT, metadata_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "CREATE INDEX records_status_idx ON records(recovery_status)",
    "CREATE INDEX records_index_idx ON records(laion_or_sample_index)",
    "CREATE INDEX records_url_idx ON records(original_url)",
    "CREATE INDEX records_normalized_url_idx ON records(normalized_url)",
    "CREATE INDEX records_digest_idx ON records(archive_digest)",
    "CREATE INDEX records_sha_idx ON records(sha256)",
    """CREATE TABLE attempts (
        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id TEXT NOT NULL, strategy TEXT NOT NULL, attempted_url TEXT,
        timestamp TEXT NOT NULL, http_status INTEGER, exception TEXT,
        response_mime TEXT, response_size INTEGER, archive_collection TEXT,
        archive_timestamp TEXT, warc_filename TEXT, warc_offset INTEGER,
        warc_length INTEGER, validation_result TEXT, candidate_sha256 TEXT,
        request_key TEXT, FOREIGN KEY(record_id) REFERENCES records(record_id)
    )""",
    "CREATE INDEX attempts_record_idx ON attempts(record_id)",
    "CREATE INDEX attempts_request_idx ON attempts(request_key)",
    """CREATE TABLE assets (
        sha256 TEXT PRIMARY KEY, sha1 TEXT NOT NULL,
        perceptual_hash TEXT NOT NULL, width INTEGER NOT NULL,
        height INTEGER NOT NULL, mode TEXT NOT NULL, image_format TEXT NOT NULL,
        extension TEXT NOT NULL, byte_size INTEGER NOT NULL,
        local_raw_path TEXT NOT NULL, local_normalized_path TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE candidates (
        candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id TEXT NOT NULL, sha256 TEXT NOT NULL,
        perceptual_hash TEXT NOT NULL, strategy TEXT NOT NULL,
        resolved_url TEXT, archive_timestamp TEXT, archive_digest TEXT,
        source_page INTEGER NOT NULL DEFAULT 0, raw_path TEXT NOT NULL,
        normalized_path TEXT NOT NULL, candidate_path TEXT NOT NULL,
        response_mime TEXT, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL,
        UNIQUE(record_id, sha256, strategy, resolved_url, archive_timestamp),
        FOREIGN KEY(record_id) REFERENCES records(record_id),
        FOREIGN KEY(sha256) REFERENCES assets(sha256)
    )""",
    """CREATE TABLE official_assets (
        record_id TEXT NOT NULL, role TEXT NOT NULL, repo_path TEXT NOT NULL,
        local_path TEXT NOT NULL, source_url TEXT, sha256 TEXT,
        metadata_json TEXT NOT NULL, PRIMARY KEY(record_id, role, repo_path),
        FOREIGN KEY(record_id) REFERENCES records(record_id)
    )""",
    """CREATE TABLE http_cache (
        request_key TEXT PRIMARY KEY, status_code INTEGER NOT NULL,
        final_url TEXT NOT NULL, content_type TEXT, response_size INTEGER NOT NULL,
        body_sha256 TEXT, body_path TEXT NOT NULL, created_at TEXT NOT NULL
    )""",
    """CREATE TABLE run_metadata (
        key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE recovery_stage_state (
        record_id TEXT NOT NULL, stage_name TEXT NOT NULL, outcome TEXT NOT NULL,
        message TEXT, retryable INTEGER NOT NULL DEFAULT 0,
        started_at TEXT NOT NULL, finished_at TEXT, run_id TEXT NOT NULL,
        attempt_number INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY(record_id, stage_name),
        FOREIGN KEY(record_id) REFERENCES records(record_id)
    )""",
    "CREATE INDEX recovery_stage_outcome_idx ON recovery_stage_state(outcome)",
    """CREATE TABLE recovery_stage_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id TEXT NOT NULL, stage_name TEXT NOT NULL, outcome TEXT NOT NULL,
        message TEXT, retryable INTEGER NOT NULL, run_id TEXT NOT NULL,
        attempt_number INTEGER NOT NULL, started_at TEXT NOT NULL,
        finished_at TEXT, FOREIGN KEY(record_id) REFERENCES records(record_id)
    )""",
    "CREATE INDEX recovery_stage_events_record_idx ON recovery_stage_events(record_id, event_id)",
)

_REQUIRED_COLUMNS = {
    "records": frozenset(
        """record_id model_name source_row_number prompt_raw overfit_type
        laion_or_sample_index original_url normalized_url source_page_url
        recovery_status recovery_method resolved_url archive_timestamp
        archive_digest local_raw_path local_normalized_path sha256 sha1
        perceptual_hash width height image_format auxiliary_asset_paths
        number_of_attempts attempt_log_path ambiguity_reason failure_reason
        metadata_json updated_at""".split()
    ),
    "attempts": frozenset(
        """attempt_id record_id strategy attempted_url timestamp http_status
        exception response_mime response_size archive_collection
        archive_timestamp warc_filename warc_offset warc_length
        validation_result candidate_sha256 request_key""".split()
    ),
    "assets": frozenset(
        """sha256 sha1 perceptual_hash width height mode image_format extension
        byte_size local_raw_path local_normalized_path created_at""".split()
    ),
    "candidates": frozenset(
        """candidate_id record_id sha256 perceptual_hash strategy resolved_url
        archive_timestamp archive_digest source_page raw_path normalized_path
        candidate_path response_mime metadata_json created_at""".split()
    ),
    "official_assets": frozenset(
        """record_id role repo_path local_path source_url sha256 metadata_json""".split()
    ),
    "http_cache": frozenset(
        """request_key status_code final_url content_type response_size
        body_sha256 body_path created_at""".split()
    ),
    "run_metadata": frozenset("key value_json updated_at".split()),
    "recovery_stage_state": frozenset(
        """record_id stage_name outcome message retryable started_at finished_at
        run_id attempt_number""".split()
    ),
    "recovery_stage_events": frozenset(
        """event_id record_id stage_name outcome message retryable run_id
        attempt_number started_at finished_at""".split()
    ),
}
_REQUIRED_TABLES = frozenset(_REQUIRED_COLUMNS)


def _create_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _validate_schema(connection: sqlite3.Connection, database: Path) -> None:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    missing = sorted(_REQUIRED_TABLES - tables)
    if missing:
        raise StateSchemaError(
            f"current recovery schema is incomplete in {database}; "
            f"missing: {', '.join(missing)}"
        )
    for table, required in _REQUIRED_COLUMNS.items():
        columns = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        missing_columns = sorted(required - columns)
        if missing_columns:
            raise StateSchemaError(
                f"current recovery schema in {database} has incomplete {table}; "
                f"missing columns: {', '.join(missing_columns)}"
            )


def status_value(status: RecoveryStatus | str) -> str:
    try:
        return (
            status.value
            if isinstance(status, RecoveryStatus)
            else RecoveryStatus(str(status)).value
        )
    except ValueError as error:
        raise StateConsistencyError(
            f"unknown recovery status: {status!r}"
        ) from error


def stage_value(stage: RecoveryStage | str) -> str:
    try:
        return (
            stage.value
            if isinstance(stage, RecoveryStage)
            else RecoveryStage(str(stage)).value
        )
    except ValueError as error:
        raise StateConsistencyError(
            f"unknown recovery stage: {stage!r}"
        ) from error


def _scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, set)):
        return value
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


class RecoveryState:
    """Typed record, artifact, attempt, and stage-state operations."""

    RECORD_UPDATE_FIELDS = frozenset(
        {
            "recovery_status", "recovery_method", "resolved_url",
            "archive_timestamp", "archive_digest", "local_raw_path",
            "local_normalized_path", "sha256", "sha1", "perceptual_hash",
            "width", "height", "image_format", "auxiliary_asset_paths",
            "attempt_log_path", "ambiguity_reason", "failure_reason",
        }
    )
    ATTEMPT_FIELDS = frozenset(
        {
            "attempted_url", "timestamp", "http_status", "exception",
            "response_mime", "response_size", "archive_collection",
            "archive_timestamp", "warc_filename", "warc_offset", "warc_length",
            "validation_result", "candidate_sha256", "request_key",
        }
    )
    DONOR_MATCH_FIELDS = frozenset(
        {
            "laion_or_sample_index",
            "original_url",
            "normalized_url",
        }
    )

    def __init__(self, paths: WebsterPaths):
        self.paths = paths
        self.paths.state.mkdir(parents=True, exist_ok=True)
        is_new = (
            not paths.database.exists()
            or paths.database.stat().st_size == 0
        )
        self.lock = threading.RLock()
        self.run_id = uuid.uuid4().hex
        self.connection = sqlite3.connect(
            paths.database, timeout=60, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        try:
            if is_new:
                _create_schema(self.connection)
            else:
                _validate_schema(self.connection, paths.database)
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.reconcile_interrupted_stages()
        except BaseException:
            self.connection.close()
            raise

    def __enter__(self) -> RecoveryState:
        return self

    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        self.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            try:
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def initialize_records(self, manifest: pd.DataFrame) -> None:
        now = utc_now()
        values: list[tuple[Any, ...]] = []
        for raw_row in manifest.to_dict(orient="records"):
            # Concatenating model-specific source tables introduces pandas NaN
            # values for columns that are absent from one table.  They mean
            # "missing" here, so persist them as JSON null without weakening
            # canonical JSON's rejection of genuine non-finite numerics.
            row = {str(key): _scalar(value) for key, value in raw_row.items()}
            record_id = str(row["record_id"])
            sample_index = _scalar(row.get("laion_or_sample_index"))
            values.append(
                (
                    record_id,
                    str(row["model_name"]),
                    int(row["source_row_number"]),
                    _scalar(row.get("prompt_raw")),
                    _scalar(row.get("overfit_type")),
                    None if sample_index is None else str(sample_index),
                    _scalar(row.get("original_url")),
                    _scalar(row.get("normalized_url")),
                    _scalar(row.get("source_page_url")),
                    str(self.paths.logs / f"{record_id}.jsonl"),
                    canonical_json(row),
                    now,
                )
            )
        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO records(
                    record_id, model_name, source_row_number, prompt_raw,
                    overfit_type, laion_or_sample_index, original_url,
                    normalized_url, source_page_url, attempt_log_path,
                    metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    model_name=excluded.model_name,
                    source_row_number=excluded.source_row_number,
                    prompt_raw=excluded.prompt_raw,
                    overfit_type=excluded.overfit_type,
                    laion_or_sample_index=excluded.laion_or_sample_index,
                    original_url=excluded.original_url,
                    normalized_url=excluded.normalized_url,
                    source_page_url=excluded.source_page_url,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at""",
                values,
            )

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM records WHERE record_id=?", (record_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def records(
        self,
        *,
        statuses: Iterable[RecoveryStatus | str] | None = None,
    ) -> list[dict[str, Any]]:
        parameters: list[str] = []
        where = ""
        if statuses is not None:
            parameters = sorted({status_value(status) for status in statuses})
            if not parameters:
                return []
            placeholders = ",".join("?" for _ in parameters)
            where = f" WHERE recovery_status IN ({placeholders})"
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM records"
                + where
                + " ORDER BY model_name, source_row_number",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def active_records(self) -> list[dict[str, Any]]:
        return self.records(statuses=ACTIVE_STATUS_VALUES)

    def recovered_donors(
        self,
        field: str,
        value: object,
        *,
        exclude_record_id: str,
    ) -> list[dict[str, Any]]:
        """Return indexed exact-match donors without scanning every record."""

        if field not in self.DONOR_MATCH_FIELDS:
            raise ValueError(f"unsupported donor match field: {field}")
        statuses = sorted(RECOVERED_STATUS_VALUES)
        placeholders = ",".join("?" for _ in statuses)
        with self.lock:
            rows = self.connection.execute(
                f"""SELECT * FROM records
                    WHERE recovery_status IN ({placeholders})
                      AND {field}=?
                      AND record_id<>?
                    ORDER BY model_name, source_row_number""",
                (*statuses, str(value), exclude_record_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_record(self, record_id: str, **fields: Any) -> None:
        invalid = set(fields) - self.RECORD_UPDATE_FIELDS
        if invalid:
            raise ValueError(f"unsupported record fields: {sorted(invalid)}")
        if not fields:
            return
        existing = self.get_record(record_id)
        if existing is None:
            raise StateConsistencyError(
                f"recovery record does not exist: {record_id}"
            )
        if "recovery_status" in fields:
            target = status_value(fields["recovery_status"])
            self._validate_status_transition(
                str(existing["recovery_status"]), target
            )
            fields["recovery_status"] = target
        values = {
            key: (
                canonical_json(value)
                if key == "auxiliary_asset_paths"
                and not isinstance(value, str)
                else _scalar(value)
            )
            for key, value in fields.items()
        }
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE records SET {assignments} WHERE record_id=?",
                (*values.values(), record_id),
            )
            if cursor.rowcount != 1:
                raise StateConsistencyError(
                    f"recovery record does not exist: {record_id}"
                )

    def set_status(
        self,
        record_id: str,
        status: RecoveryStatus | str,
        *,
        reason: str | None = None,
    ) -> None:
        self.update_record(
            record_id, recovery_status=status, failure_reason=reason
        )

    def reactivate_unresolved_for_stage(
        self,
        record_id: str,
        stage: RecoveryStage | str,
        *,
        reason: str,
        strategy_version: int | None = None,
    ) -> bool:
        """Audit and narrowly reopen unresolved data for a versioned source."""

        name = stage_value(stage)
        allowed = {
            RecoveryStage.GROUND_TRUTH_MIRROR.value,
            RecoveryStage.ARQUIVO.value,
        }
        if name not in allowed:
            raise ValueError(
                "terminal unresolved records may only be reopened for "
                "an explicitly versioned recovery stage"
            )
        if strategy_version is not None and strategy_version <= 0:
            raise ValueError("strategy_version must be positive")
        validation_result = f"reactivated_for_{name}"
        if strategy_version is not None:
            validation_result += f"_v{strategy_version}"
        timestamp = utc_now()
        attempt_id: int | None = None
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT recovery_status FROM records WHERE record_id=?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise StateConsistencyError(
                    f"recovery record does not exist: {record_id}"
                )
            if str(row[0]) != RecoveryStatus.UNRESOLVED.value:
                return False
            stage_rows = connection.execute(
                """SELECT stage_name, outcome, retryable,
                    started_at, attempt_number
                FROM recovery_stage_state WHERE record_id=?""",
                (record_id,),
            ).fetchall()
            stage_results = {
                str(item[0]): (
                    str(item[1]),
                    bool(item[2]),
                    str(item[3]),
                    int(item[4]),
                )
                for item in stage_rows
            }
            # Strategy updates are independent, so a mirror result is not required
            # when deciding whether a mirror or Arquivo retry is eligible.
            excluded = {name, RecoveryStage.GROUND_TRUTH_MIRROR.value}
            required_stages = {
                item.value for item in STAGE_ORDER if item.value not in excluded
            }
            if any(
                required not in stage_results
                or stage_results[required][0]
                == StageOutcome.RUNNING.value
                or stage_results[required][1]
                for required in required_stages
            ):
                return False
            prior = stage_results.get(name)
            if prior is not None and strategy_version is None:
                return False
            cursor = connection.execute(
                """INSERT INTO attempts(
                    record_id, strategy, timestamp, exception,
                    validation_result
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    record_id,
                    "state_reactivation",
                    timestamp,
                    reason,
                    validation_result,
                ),
            )
            attempt_id = int(cursor.lastrowid)
            connection.execute(
                """UPDATE records
                SET recovery_status=?, failure_reason=?,
                    number_of_attempts=number_of_attempts+1, updated_at=?
                WHERE record_id=?""",
                (
                    RecoveryStatus.RETRYABLE_ERROR.value,
                    reason,
                    timestamp,
                    record_id,
                ),
            )
            if prior is not None:
                self._write_retryable_stage_locked(
                    connection,
                    record_id,
                    name,
                    reason,
                    prior[2],
                    timestamp,
                    self.run_id,
                    prior[3],
                )
            self._invalidate_final_resolution_locked(
                connection, record_id, reason, timestamp
            )
        self._append_attempt_log(
            record_id,
            {
                "event": "state_reactivation",
                "attempt_id": attempt_id,
                "strategy": "state_reactivation",
                "timestamp": timestamp,
                "exception": reason,
                "validation_result": validation_result,
            },
        )
        return True

    def reactivate_invalid_recovered_artifacts(
        self,
        record_id: str,
        *,
        reason: str,
    ) -> bool:
        """Audit a broken recovered record and make only it retryable."""

        timestamp = utc_now()
        attempt_id: int | None = None
        with self.transaction() as connection:
            row = connection.execute(
                """SELECT recovery_status, recovery_method
                FROM records WHERE record_id=?""",
                (record_id,),
            ).fetchone()
            if row is None:
                raise StateConsistencyError(
                    f"recovery record does not exist: {record_id}"
                )
            status = str(row[0])
            if status not in RECOVERED_STATUS_VALUES:
                return False
            stage_by_status = {
                RecoveryStatus.RECOVERED_EXACT_URL.value:
                    RecoveryStage.DIRECT,
                RecoveryStatus.RECOVERED_OFFICIAL_CACHE.value:
                    RecoveryStage.OFFICIAL_ASSETS,
                RecoveryStatus.RECOVERED_GROUND_TRUTH_MIRROR.value:
                    RecoveryStage.GROUND_TRUTH_MIRROR,
                RecoveryStatus.RECOVERED_WAYBACK_EXACT.value:
                    RecoveryStage.WAYBACK,
                RecoveryStatus.RECOVERED_ARQUIVO_EXACT.value:
                    RecoveryStage.ARQUIVO,
                RecoveryStatus.RECOVERED_COMMONCRAWL_EXACT.value:
                    RecoveryStage.COMMON_CRAWL,
                RecoveryStatus.RECOVERED_VERIFIED_DUPLICATE.value:
                    RecoveryStage.EXACT_REUSE,
            }
            source_stage = stage_by_status.get(status)
            if source_stage is None:
                method = str(row[1] or "").casefold()
                if method.startswith("wayback"):
                    source_stage = RecoveryStage.WAYBACK
                elif method.startswith("arquivo"):
                    source_stage = RecoveryStage.ARQUIVO
                elif method.startswith("commoncrawl"):
                    source_stage = RecoveryStage.COMMON_CRAWL
                elif method.startswith("hf_ground_truth_mirror"):
                    source_stage = RecoveryStage.GROUND_TRUTH_MIRROR
                elif method.startswith("official"):
                    source_stage = RecoveryStage.OFFICIAL_ASSETS
                else:
                    source_stage = RecoveryStage.EXACT_REUSE
            cursor = connection.execute(
                """INSERT INTO attempts(
                    record_id, strategy, timestamp, exception,
                    validation_result
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    record_id,
                    "recovered_artifact_validation",
                    timestamp,
                    reason,
                    "recovered_artifacts_invalid",
                ),
            )
            attempt_id = int(cursor.lastrowid)
            connection.execute(
                """UPDATE records
                SET recovery_status=?, recovery_method=NULL,
                    resolved_url=NULL, archive_timestamp=NULL,
                    archive_digest=NULL, local_raw_path=NULL,
                    local_normalized_path=NULL, sha256=NULL, sha1=NULL,
                    perceptual_hash=NULL, width=NULL, height=NULL,
                    image_format=NULL, auxiliary_asset_paths='[]',
                    ambiguity_reason=NULL, failure_reason=?,
                    number_of_attempts=number_of_attempts+1, updated_at=?
                WHERE record_id=?""",
                (
                    RecoveryStatus.RETRYABLE_ERROR.value,
                    reason,
                    timestamp,
                    record_id,
                ),
            )
            stages = connection.execute(
                """SELECT stage_name, started_at, attempt_number
                FROM recovery_stage_state
                WHERE record_id=? AND stage_name IN (?, ?)""",
                (
                    record_id,
                    source_stage.value,
                    RecoveryStage.FINAL_RESOLUTION.value,
                ),
            ).fetchall()
            for stage_row in stages:
                self._write_retryable_stage_locked(
                    connection,
                    record_id,
                    str(stage_row[0]),
                    reason,
                    str(stage_row[1]),
                    timestamp,
                    self.run_id,
                    int(stage_row[2]),
                )
        self._append_attempt_log(
            record_id,
            {
                "event": "recovered_artifact_validation",
                "attempt_id": attempt_id,
                "strategy": "recovered_artifact_validation",
                "timestamp": timestamp,
                "exception": reason,
                "validation_result": "recovered_artifacts_invalid",
            },
        )
        return True

    def mark_recovered(
        self,
        record_id: str,
        status: RecoveryStatus | str,
        **fields: Any,
    ) -> None:
        target = status_value(status)
        if target not in RECOVERED_STATUS_VALUES:
            raise ValueError(f"unsupported recovered status: {target}")
        self.update_record(
            record_id,
            recovery_status=target,
            failure_reason=None,
            **fields,
        )

    @staticmethod
    def _validate_status_transition(current: str, target: str) -> None:
        if current not in ALL_STATUS_VALUES or target not in ALL_STATUS_VALUES:
            raise StateConsistencyError(
                f"unknown recovery transition {current!r} -> {target!r}"
            )
        if current in TERMINAL_STATUS_VALUES and target != current:
            raise StateConsistencyError(
                f"automatic transition from terminal {current} to {target} "
                "is forbidden"
            )

    def log_attempt(
        self, record_id: str, strategy: str, **fields: Any
    ) -> int:
        invalid = set(fields) - self.ATTEMPT_FIELDS
        if invalid:
            raise ValueError(
                f"unsupported attempt fields: {sorted(invalid)}"
            )
        values = {"record_id": record_id, "strategy": strategy, **fields}
        values.setdefault("timestamp", utc_now())
        columns = list(values)
        with self.transaction() as connection:
            cursor = connection.execute(
                f"INSERT INTO attempts ({', '.join(columns)}) VALUES "
                f"({', '.join('?' for _ in columns)})",
                tuple(_scalar(values[column]) for column in columns),
            )
            connection.execute(
                """UPDATE records
                SET number_of_attempts=number_of_attempts+1, updated_at=?
                WHERE record_id=?""",
                (utc_now(), record_id),
            )
            attempt_id = int(cursor.lastrowid)
        self._append_attempt_log(
            record_id, {"attempt_id": attempt_id, **values}
        )
        return attempt_id

    def update_attempt(
        self, attempt_id: int | None, **fields: Any
    ) -> None:
        if attempt_id is None or not fields:
            return
        invalid = set(fields) - {
            "validation_result", "candidate_sha256", "exception"
        }
        if invalid:
            raise ValueError(
                f"unsupported attempt update fields: {sorted(invalid)}"
            )
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT record_id FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise StateConsistencyError(
                    f"attempt does not exist: {attempt_id}"
                )
            assignments = ", ".join(f"{name}=?" for name in fields)
            connection.execute(
                f"UPDATE attempts SET {assignments} WHERE attempt_id=?",
                (*fields.values(), attempt_id),
            )
            record_id = str(row[0])
        self._append_attempt_log(
            record_id,
            {
                "event": "attempt_update",
                "attempt_id": attempt_id,
                "timestamp": utc_now(),
                **fields,
            },
        )

    def attempts(
        self, record_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM attempts"
        parameters: tuple[Any, ...] = ()
        if record_id is not None:
            query += " WHERE record_id=?"
            parameters = (record_id,)
        return self._rows(query + " ORDER BY attempt_id", parameters)

    def _append_attempt_log(
        self, record_id: str, payload: Mapping[str, Any]
    ) -> None:
        record = self.get_record(record_id)
        if record is None or record.get("attempt_log_path") is None:
            return
        path = Path(str(record["attempt_log_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(json_value(payload), ensure_ascii=False) + "\n"
            )

    def set_run_metadata(self, key: str, value: Any) -> None:
        self.set_run_metadata_batch({key: value})

    def set_run_metadata_batch(
        self, values: Mapping[str, Any]
    ) -> None:
        if not values:
            return
        timestamp = utc_now()
        rows = [
            (str(key), canonical_json(value), timestamp)
            for key, value in values.items()
        ]
        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO run_metadata(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at""",
                rows,
            )

    def get_run_metadata(
        self, key: str, default: Any = None
    ) -> Any:
        with self.lock:
            row = self.connection.execute(
                "SELECT value_json FROM run_metadata WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(str(row[0]))
        except json.JSONDecodeError as error:
            raise StateConsistencyError(
                f"run metadata is invalid JSON: {key}"
            ) from error

    def run_metadata(self) -> dict[str, Any]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT key, value_json FROM run_metadata ORDER BY key"
            ).fetchall()
        try:
            return {
                str(row[0]): json.loads(str(row[1])) for row in rows
            }
        except json.JSONDecodeError as error:
            raise StateConsistencyError(
                "run metadata contains invalid JSON"
            ) from error

    def begin_stage(
        self,
        record_id: str,
        stage: RecoveryStage | str,
        *,
        message: str | None = None,
    ) -> int:
        name = stage_value(stage)
        started = utc_now()
        with self.transaction() as connection:
            record = connection.execute(
                "SELECT recovery_status FROM records WHERE record_id=?",
                (record_id,),
            ).fetchone()
            if record is None:
                raise StateConsistencyError(
                    f"recovery record does not exist: {record_id}"
                )
            if str(record[0]) in TERMINAL_STATUS_VALUES:
                raise StateConsistencyError(
                    f"cannot begin {name} for terminal record {record_id}"
                )
            previous = connection.execute(
                """SELECT outcome, attempt_number
                FROM recovery_stage_state
                WHERE record_id=? AND stage_name=?""",
                (record_id, name),
            ).fetchone()
            rerunnable = {
                StageOutcome.RUNNING.value,
                StageOutcome.RETRYABLE_ERROR.value,
            }
            if previous is not None and str(previous[0]) not in rerunnable:
                raise StateConsistencyError(
                    f"completed stage {name} must be invalidated before rerun"
                )
            attempt = (
                int(previous[1]) + 1 if previous is not None else 1
            )
            connection.execute(
                """INSERT INTO recovery_stage_state(
                    record_id, stage_name, outcome, message, retryable,
                    started_at, finished_at, run_id, attempt_number
                ) VALUES (?, ?, ?, ?, 0, ?, NULL, ?, ?)
                ON CONFLICT(record_id, stage_name) DO UPDATE SET
                    outcome=excluded.outcome,
                    message=excluded.message,
                    retryable=0,
                    started_at=excluded.started_at,
                    finished_at=NULL,
                    run_id=excluded.run_id,
                    attempt_number=excluded.attempt_number""",
                (
                    record_id,
                    name,
                    StageOutcome.RUNNING.value,
                    message,
                    started,
                    self.run_id,
                    attempt,
                ),
            )
            self._insert_stage_event(
                connection,
                record_id,
                name,
                StageOutcome.RUNNING.value,
                message,
                False,
                self.run_id,
                attempt,
                started,
                None,
            )
        return attempt

    def finish_stage(
        self,
        record_id: str,
        stage: RecoveryStage | str,
        result: StageResult,
    ) -> None:
        if result.outcome is StageOutcome.RUNNING:
            raise ValueError("finish_stage requires a completed outcome")
        name = stage_value(stage)
        finished = utc_now()
        with self.transaction() as connection:
            current = connection.execute(
                """SELECT started_at, run_id, attempt_number
                FROM recovery_stage_state
                WHERE record_id=? AND stage_name=?""",
                (record_id, name),
            ).fetchone()
            started, run_id, attempt = (
                (finished, self.run_id, 1)
                if current is None
                else (
                    str(current[0]),
                    str(current[1]),
                    int(current[2]),
                )
            )
            connection.execute(
                """INSERT INTO recovery_stage_state(
                    record_id, stage_name, outcome, message, retryable,
                    started_at, finished_at, run_id, attempt_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id, stage_name) DO UPDATE SET
                    outcome=excluded.outcome,
                    message=excluded.message,
                    retryable=excluded.retryable,
                    finished_at=excluded.finished_at,
                    run_id=excluded.run_id,
                    attempt_number=excluded.attempt_number""",
                (
                    record_id,
                    name,
                    result.outcome.value,
                    result.message,
                    int(result.retryable),
                    started,
                    finished,
                    run_id,
                    attempt,
                ),
            )
            self._insert_stage_event(
                connection,
                record_id,
                name,
                result.outcome.value,
                result.message,
                result.retryable,
                run_id,
                attempt,
                started,
                finished,
            )
            if result.retryable:
                connection.execute(
                    """UPDATE records
                    SET recovery_status=?, failure_reason=?, updated_at=?
                    WHERE record_id=? AND recovery_status IN (?, ?)""",
                    (
                        RecoveryStatus.RETRYABLE_ERROR.value,
                        result.message,
                        finished,
                        record_id,
                        RecoveryStatus.PENDING.value,
                        RecoveryStatus.RETRYABLE_ERROR.value,
                    ),
                )
            elif name != RecoveryStage.FINAL_RESOLUTION.value:
                self._invalidate_final_resolution_locked(
                    connection,
                    record_id,
                    f"{name} received a newer result",
                    finished,
                )

    def stage_result(
        self, record_id: str, stage: RecoveryStage | str
    ) -> StageResult | None:
        name = stage_value(stage)
        with self.lock:
            row = self.connection.execute(
                """SELECT outcome, message, retryable
                FROM recovery_stage_state
                WHERE record_id=? AND stage_name=?""",
                (record_id, name),
            ).fetchone()
        if row is None:
            return None
        try:
            return StageResult(
                StageOutcome(str(row[0])),
                None if row[1] is None else str(row[1]),
                bool(row[2]),
            )
        except ValueError as error:
            raise StateConsistencyError(
                f"invalid stage state for {record_id}/{name}"
            ) from error

    def stage_results(
        self, record_id: str
    ) -> dict[RecoveryStage, StageResult]:
        with self.lock:
            rows = self.connection.execute(
                """SELECT stage_name, outcome, message, retryable
                FROM recovery_stage_state WHERE record_id=?""",
                (record_id,),
            ).fetchall()
        try:
            return {
                RecoveryStage(str(row[0])): StageResult(
                    StageOutcome(str(row[1])),
                    None if row[2] is None else str(row[2]),
                    bool(row[3]),
                )
                for row in rows
            }
        except ValueError as error:
            raise StateConsistencyError(
                f"invalid persisted stage result for {record_id}"
            ) from error

    def stage_state_rows(
        self, record_id: str
    ) -> list[dict[str, Any]]:
        return self._rows(
            """SELECT * FROM recovery_stage_state
            WHERE record_id=? ORDER BY rowid""",
            (record_id,),
        )

    def stage_events(self, record_id: str) -> list[dict[str, Any]]:
        return self._rows(
            """SELECT * FROM recovery_stage_events
            WHERE record_id=? ORDER BY event_id""",
            (record_id,),
        )

    def stages_to_retry(self, record_id: str) -> list[RecoveryStage]:
        results = self.stage_results(record_id)
        retryable = {
            StageOutcome.RUNNING,
            StageOutcome.RETRYABLE_ERROR,
        }
        return [
            stage
            for stage in STAGE_ORDER
            if stage not in results or results[stage].outcome in retryable
        ]

    def invalidate_stage(
        self,
        record_id: str,
        stage: RecoveryStage | str,
        reason: str,
    ) -> None:
        name = stage_value(stage)
        timestamp = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                """SELECT started_at, attempt_number
                FROM recovery_stage_state
                WHERE record_id=? AND stage_name=?""",
                (record_id, name),
            ).fetchone()
            self._write_retryable_stage_locked(
                connection,
                record_id,
                name,
                reason,
                timestamp if row is None else str(row[0]),
                timestamp,
                self.run_id,
                1 if row is None else int(row[1]),
            )

    def reconcile_interrupted_stages(self) -> list[str]:
        timestamp = utc_now()
        affected: set[str] = set()
        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT s.record_id, s.stage_name, s.started_at,
                          s.run_id, s.attempt_number
                FROM recovery_stage_state AS s
                JOIN records AS r ON r.record_id=s.record_id
                WHERE s.outcome=? AND r.recovery_status IN (?, ?)""",
                (
                    StageOutcome.RUNNING.value,
                    RecoveryStatus.PENDING.value,
                    RecoveryStatus.RETRYABLE_ERROR.value,
                ),
            ).fetchall()
            for row in rows:
                record_id = str(row[0])
                name = str(row[1])
                reason = (
                    f"interrupted {name} stage from prior run {row[3]}"
                )
                self._write_retryable_stage_locked(
                    connection,
                    record_id,
                    name,
                    reason,
                    str(row[2]),
                    timestamp,
                    str(row[3]),
                    int(row[4]),
                )
                affected.add(record_id)
        return sorted(affected)

    def _write_retryable_stage_locked(
        self,
        connection: sqlite3.Connection,
        record_id: str,
        name: str,
        reason: str,
        started: str,
        finished: str,
        run_id: str,
        attempt: int,
    ) -> None:
        connection.execute(
            """INSERT INTO recovery_stage_state(
                record_id, stage_name, outcome, message, retryable,
                started_at, finished_at, run_id, attempt_number
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(record_id, stage_name) DO UPDATE SET
                outcome=excluded.outcome,
                message=excluded.message,
                retryable=1,
                finished_at=excluded.finished_at,
                run_id=excluded.run_id,
                attempt_number=excluded.attempt_number""",
            (
                record_id,
                name,
                StageOutcome.RETRYABLE_ERROR.value,
                reason,
                started,
                finished,
                run_id,
                attempt,
            ),
        )
        self._insert_stage_event(
            connection,
            record_id,
            name,
            StageOutcome.RETRYABLE_ERROR.value,
            reason,
            True,
            run_id,
            attempt,
            started,
            finished,
        )
        connection.execute(
            """UPDATE records
            SET recovery_status=?, failure_reason=?, updated_at=?
            WHERE record_id=? AND recovery_status IN (?, ?)""",
            (
                RecoveryStatus.RETRYABLE_ERROR.value,
                reason,
                finished,
                record_id,
                RecoveryStatus.PENDING.value,
                RecoveryStatus.RETRYABLE_ERROR.value,
            ),
        )

    def _invalidate_final_resolution_locked(
        self,
        connection: sqlite3.Connection,
        record_id: str,
        reason: str,
        timestamp: str,
    ) -> None:
        row = connection.execute(
            """SELECT started_at, attempt_number
            FROM recovery_stage_state
            WHERE record_id=? AND stage_name=?""",
            (record_id, RecoveryStage.FINAL_RESOLUTION.value),
        ).fetchone()
        if row is not None:
            self._write_retryable_stage_locked(
                connection,
                record_id,
                RecoveryStage.FINAL_RESOLUTION.value,
                reason,
                str(row[0]),
                timestamp,
                self.run_id,
                int(row[1]),
            )

    @staticmethod
    def _insert_stage_event(
        connection: sqlite3.Connection,
        record_id: str,
        name: str,
        outcome: str,
        message: str | None,
        retryable: bool,
        run_id: str,
        attempt: int,
        started: str,
        finished: str | None,
    ) -> None:
        connection.execute(
            """INSERT INTO recovery_stage_events(
                record_id, stage_name, outcome, message, retryable,
                run_id, attempt_number, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record_id,
                name,
                outcome,
                message,
                int(retryable),
                run_id,
                attempt,
                started,
                finished,
            ),
        )

    def add_asset(
        self,
        metadata: ImageMetadata,
        raw_path: Path,
        normalized_path: Path,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO assets(
                    sha256, sha1, perceptual_hash, width, height, mode,
                    image_format, extension, byte_size, local_raw_path,
                    local_normalized_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO UPDATE SET
                    sha1=excluded.sha1,
                    perceptual_hash=excluded.perceptual_hash,
                    width=excluded.width,
                    height=excluded.height,
                    mode=excluded.mode,
                    image_format=excluded.image_format,
                    extension=excluded.extension,
                    byte_size=excluded.byte_size,
                    local_raw_path=excluded.local_raw_path,
                    local_normalized_path=excluded.local_normalized_path""",
                (
                    metadata.sha256,
                    metadata.sha1,
                    metadata.perceptual_hash,
                    metadata.width,
                    metadata.height,
                    metadata.mode,
                    metadata.image_format,
                    metadata.extension,
                    metadata.byte_size,
                    str(raw_path),
                    str(normalized_path),
                    utc_now(),
                ),
            )

    def get_asset(self, sha256: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM assets WHERE sha256=?", (sha256,)
            ).fetchone()
        return dict(row) if row is not None else None

    def assets(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM assets ORDER BY sha256")

    def add_candidate(
        self,
        candidate: ArchiveCandidate,
        metadata: Mapping[str, Any],
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO candidates(
                    record_id, sha256, perceptual_hash, strategy,
                    resolved_url, archive_timestamp, archive_digest,
                    source_page, raw_path, normalized_path, candidate_path,
                    response_mime, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.record_id,
                    candidate.sha256,
                    candidate.perceptual_hash,
                    candidate.strategy,
                    candidate.resolved_url or "",
                    candidate.archive_timestamp or "",
                    candidate.archive_digest,
                    int(candidate.source_page),
                    candidate.raw_path,
                    candidate.normalized_path,
                    candidate.candidate_path,
                    candidate.response_mime,
                    canonical_json(metadata),
                    utc_now(),
                ),
            )

    def candidates(
        self, record_id: str, prefix: str = ""
    ) -> list[dict[str, Any]]:
        return self._rows(
            """SELECT * FROM candidates
            WHERE record_id=? AND strategy LIKE ?
            ORDER BY archive_timestamp, candidate_id""",
            (record_id, prefix + "%"),
        )

    def all_candidates(self) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM candidates ORDER BY record_id, candidate_id"
        )

    def candidates_for_archive_digest(
        self, digest: str
    ) -> list[dict[str, Any]]:
        normalized = str(digest).strip().lower().removeprefix("sha1:")
        return self._rows(
            """SELECT * FROM candidates
            WHERE LOWER(
                REPLACE(COALESCE(archive_digest, ''), 'sha1:', '')
            )=?
            ORDER BY candidate_id""",
            (normalized,),
        )

    def update_candidate_path(
        self, candidate_id: int, candidate_path: Path
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE candidates SET candidate_path=? "
                "WHERE candidate_id=?",
                (str(candidate_path), candidate_id),
            )

    def delete_candidate(self, candidate_id: int) -> None:
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM candidates WHERE candidate_id=?",
                (candidate_id,),
            )

    def add_official_asset(
        self, record_id: str, asset: Mapping[str, Any]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO official_assets(
                    record_id, role, repo_path, local_path, source_url,
                    sha256, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id,
                    str(asset["role"]),
                    str(asset["repo_path"]),
                    str(asset["local_path"]),
                    _scalar(asset.get("source_url")),
                    _scalar(asset.get("sha256")),
                    canonical_json(asset),
                ),
            )

    def official_assets(
        self, record_id: str | None = None
    ) -> list[dict[str, Any]]:
        if record_id is None:
            return self._rows(
                """SELECT * FROM official_assets
                ORDER BY record_id, role, repo_path"""
            )
        return self._rows(
            """SELECT * FROM official_assets
            WHERE record_id=? ORDER BY role, repo_path""",
            (record_id,),
        )

    def delete_official_asset(
        self, record_id: str, role: str, repo_path: str
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """DELETE FROM official_assets
                WHERE record_id=? AND role=? AND repo_path=?""",
                (record_id, role, repo_path),
            )

    def clear_official_assets(self, record_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM official_assets WHERE record_id=?",
                (record_id,),
            )

    def get_http_cache(
        self, request_key: str
    ) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM http_cache WHERE request_key=?",
                (request_key,),
            ).fetchone()
        return dict(row) if row is not None else None

    def put_http_cache(
        self,
        request_key: str,
        status_code: int,
        final_url: str,
        content_type: str | None,
        response_size: int,
        body_sha256: str,
        body_path: Path,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO http_cache(
                    request_key, status_code, final_url, content_type,
                    response_size, body_sha256, body_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_key,
                    status_code,
                    final_url,
                    content_type,
                    response_size,
                    body_sha256,
                    str(body_path),
                    utc_now(),
                ),
            )

    def invalidate_http_cache(self, request_key: str) -> Path | None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT body_path FROM http_cache WHERE request_key=?",
                (request_key,),
            ).fetchone()
            connection.execute(
                "DELETE FROM http_cache WHERE request_key=?",
                (request_key,),
            )
        return None if row is None else Path(str(row[0]))

    def records_frame(self) -> pd.DataFrame:
        return self._frame(
            "SELECT * FROM records ORDER BY model_name, source_row_number"
        )

    def attempts_frame(self) -> pd.DataFrame:
        return self._frame(
            "SELECT * FROM attempts ORDER BY attempt_id"
        )

    def candidates_frame(self) -> pd.DataFrame:
        return self._frame(
            "SELECT * FROM candidates ORDER BY record_id, candidate_id"
        )

    def official_assets_frame(self) -> pd.DataFrame:
        return self._frame(
            """SELECT * FROM official_assets
            ORDER BY record_id, role, repo_path"""
        )

    def stage_state_frame(self) -> pd.DataFrame:
        return self._frame(
            """SELECT * FROM recovery_stage_state
            ORDER BY record_id, rowid"""
        )

    def _rows(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def _frame(self, query: str) -> pd.DataFrame:
        with self.lock:
            return pd.read_sql_query(query, self.connection)


__all__ = ["RecoveryState", "stage_value", "status_value"]
