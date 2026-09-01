"""Offline tests for current-schema Webster recovery state."""

from __future__ import annotations

import ast
import concurrent.futures
import io
import json
import socket
import sqlite3
from pathlib import Path

import httpx
import pandas as pd
import pytest
from PIL import Image

from utils.data import commoncrawl as commoncrawl_module
from utils.data import images as images_module
from utils.data import recovery as recovery_module
from utils.data import webster as webster_module
from utils.data.state import RecoveryState
from utils.data.webster import (
    DownloadError,
    HTTPResult,
    RecoveryStage,
    RecoveryStatus,
    StageOutcome,
    StageResult,
    StateSchemaError,
    WebsterPaths,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "sdv1-0000",
                "model_name": "sdv1",
                "source_row_number": 0,
                "prompt_raw": "exact prompt",
                "overfit_type": "TV",
                "laion_or_sample_index": 123,
                "original_url": "https://example.invalid/source.png",
                "normalized_url": "https://example.invalid/source.png",
                "source_page_url": None,
            }
        ]
    )


def test_recovery_state_persists_records_attempts_metadata_and_stages(
    tmp_path: Path,
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    with RecoveryState(paths) as state:
        state.initialize_records(_manifest())
        state.set_run_metadata("source_revision", "a" * 40)
        attempt = state.log_attempt(
            "sdv1-0000",
            "direct",
            attempted_url="https://example.invalid/source.png",
            http_status=404,
            validation_result="miss",
        )
        assert attempt == 1
        assert state.begin_stage("sdv1-0000", RecoveryStage.DIRECT) == 1
        state.finish_stage(
            "sdv1-0000",
            RecoveryStage.DIRECT,
            StageResult(StageOutcome.MISS, "not found"),
        )

        record = state.get_record("sdv1-0000")
        assert record is not None
        assert record["number_of_attempts"] == 1
        assert state.get_run_metadata("source_revision") == "a" * 40
        assert state.attempts("sdv1-0000")[0]["http_status"] == 404
        assert state.stage_result("sdv1-0000", RecoveryStage.DIRECT) == StageResult(
            StageOutcome.MISS, "not found"
        )
        assert len(state.stage_events("sdv1-0000")) == 2

    assert paths.database.is_file()
    with sqlite3.connect(paths.database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_initialize_records_serializes_dataframe_missing_values_as_null(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["template_indices"] = float("nan")
    paths = WebsterPaths.from_root(tmp_path)

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        record = state.get_record("sdv1-0000")

    assert record is not None
    metadata_json = str(record["metadata_json"])
    assert "NaN" not in metadata_json
    assert json.loads(metadata_json)["template_indices"] is None


def test_progress_remains_visible_when_stderr_is_captured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[int] = []
    completed = list(
        recovery_module._progress(
            range(2),
            total=2,
            description="Webster progress regression",
            after=seen.append,
            postfix=lambda: f"Total {len(seen)}/2",
        )
    )

    assert completed == [0, 1]
    output = capsys.readouterr().err
    assert "Webster progress regression" in output
    assert "2/2" in output
    assert "Total 2/2" in output


def test_recovery_engine_reports_all_recovery_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptions: list[tuple[str, int]] = []

    def fake_progress(
        values: object,
        *,
        total: int,
        description: str,
        unit: str = "record",
        enabled: bool = True,
        after: object = None,
        postfix: object = None,
    ) -> object:
        assert enabled
        assert unit in {"record", "target"}
        assert after is None or callable(after)
        assert callable(postfix)
        descriptions.append((description, total))
        return values

    class EmptyState:
        def records(self, *, statuses: object) -> list[dict[str, object]]:
            del statuses
            return []

    monkeypatch.setattr(recovery_module, "_progress", fake_progress)
    engine = recovery_module.RecoveryEngine(
        WebsterPaths.from_root(tmp_path),
        EmptyState(),  # type: ignore[arg-type]
        pd.DataFrame(),
        object(),  # type: ignore[arg-type]
    )

    engine.run()

    assert descriptions == [
        ("[Webster 1/10] Verify downloaded files", 0),
        ("[Webster 1/10] Local cache reuse", 0),
        ("[Webster 2/10] Official assets", 0),
        ("[Webster 3/10] Direct URLs (failed only)", 0),
        ("[Webster 4/10] Verified mirror (failed only)", 0),
        ("[Webster 5/10] Wayback (failed only)", 0),
        ("[Webster 6/10] Arquivo.pt (failed only)", 0),
        ("[Webster 7/10] Common Crawl (failed only)", 0),
        ("[Webster 8/10] Final resolution", 0),
    ]


def _category_manifest() -> pd.DataFrame:
    base = _manifest().iloc[0].to_dict()
    rows: list[dict[str, object]] = []
    for index, category in enumerate((" memorized ", "TV", "retrieval", "N")):
        rows.append(
            {
                **base,
                "record_id": f"sdv1-{index:04d}",
                "source_row_number": index,
                "overfit_type": category,
                "laion_or_sample_index": index,
                "original_url": f"https://example.test/{index}.png",
                "normalized_url": f"https://example.test/{index}.png",
            }
        )
    return pd.DataFrame(rows)


def test_recovery_category_counts_include_resumed_and_new_records(
    tmp_path: Path,
) -> None:
    manifest = _category_manifest()
    paths = WebsterPaths.from_root(tmp_path)
    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        state.mark_recovered(
            "sdv1-0000", RecoveryStatus.RECOVERED_EXACT_URL
        )
        counts = recovery_module._RecoveryCounts(state, manifest)

        assert counts.format() == (
            "MV 1/1 | TV 0/1 | RV 0/1 | N 0/1 | Total 1/4"
        )

        state.mark_recovered(
            "sdv1-0001", RecoveryStatus.RECOVERED_OFFICIAL_CACHE
        )
        counts.refresh(manifest.iloc[1].to_dict())
        counts.refresh(manifest.iloc[1].to_dict())

        assert counts.format() == (
            "MV 1/1 | TV 1/1 | RV 0/1 | N 0/1 | Total 2/4"
        )


def test_threaded_direct_progress_reports_each_category_once(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _category_manifest()
    paths = WebsterPaths.from_root(tmp_path)

    def recover(
        unused_paths: WebsterPaths,
        state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_paths, unused_fetcher
        state.mark_recovered(
            str(record["record_id"]),
            RecoveryStatus.RECOVERED_EXACT_URL,
        )
        return StageResult(StageOutcome.RECOVERED, "synthetic")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        engine = recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            direct_stage=recover,  # type: ignore[arg-type]
            direct_workers=4,
        )
        engine.stage_direct()

    output = capsys.readouterr().err
    assert "[Webster 3/10] Direct URLs" in output
    assert "MV 1/1 | TV 1/1 | RV 1/1 | N 1/1 | Total 4/4" in output


def _duplicate_target_manifest() -> pd.DataFrame:
    base = _manifest().iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                **base,
                "record_id": "sdv1-0000",
                "model_name": "sdv1",
                "source_row_number": 0,
                "overfit_type": "MV",
            },
            {
                **base,
                "record_id": "realisticvision-0000",
                "model_name": "realisticvision",
                "source_row_number": 0,
                "overfit_type": "TV",
            },
        ]
    )


def _tiny_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(output, format="PNG")
    return output.getvalue()


def _recover_synthetic_image(
    paths: WebsterPaths,
    state: RecoveryState,
    record: dict[str, object],
    *,
    status: RecoveryStatus = RecoveryStatus.RECOVERED_EXACT_URL,
) -> None:
    metadata, raw_path, normalized_path = images_module.store_image(
        paths,
        state,
        _tiny_png(),
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
        method="synthetic_exact",
        resolved_url=str(record["original_url"]),
    )


def test_target_grouping_deduplicates_only_complete_consistent_identities() -> None:
    duplicate = _duplicate_target_manifest().to_dict(orient="records")
    groups = recovery_module._group_target_records(duplicate)
    assert len(groups) == 1
    assert len(groups[0]) == 2

    conflicting = [dict(item) for item in duplicate]
    conflicting[1]["normalized_url"] = "https://example.invalid/other.png"
    assert len(recovery_module._group_target_records(conflicting)) == 2

    missing = [dict(item) for item in duplicate]
    for index, item in enumerate(missing):
        item["record_id"] = f"missing-{index}"
        item["laion_or_sample_index"] = None
        item["normalized_url"] = None
        item["original_url"] = None
    assert len(recovery_module._group_target_records(missing)) == 2


def test_direct_success_runs_once_per_target_and_fans_out_verified_donor(
    tmp_path: Path,
) -> None:
    manifest = _duplicate_target_manifest()
    paths = WebsterPaths.from_root(tmp_path)
    calls: list[str] = []

    def recover(
        supplied_paths: WebsterPaths,
        state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_fetcher
        calls.append(str(record["record_id"]))
        _recover_synthetic_image(supplied_paths, state, record)
        return StageResult(StageOutcome.RECOVERED, "synthetic exact")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            direct_stage=recover,  # type: ignore[arg-type]
            direct_workers=2,
            show_progress=False,
        ).stage_direct()

        assert calls == ["realisticvision-0000"]
        records = {
            str(item["record_id"]): item for item in state.records()
        }
        assert all(
            str(item["recovery_status"]) in webster_module.RECOVERED_STATUS_VALUES
            for item in records.values()
        )
        assert len({str(item["sha256"]) for item in records.values()}) == 1
        assert state.stage_result(
            "sdv1-0000", RecoveryStage.DIRECT
        ).outcome is StageOutcome.RECOVERED
        assert state.stage_result(
            "realisticvision-0000", RecoveryStage.DIRECT
        ).outcome is StageOutcome.RECOVERED


def test_retryable_duplicate_target_is_queried_once_and_fanned_out(
    tmp_path: Path,
) -> None:
    manifest = _duplicate_target_manifest()
    paths = WebsterPaths.from_root(tmp_path)
    calls: list[str] = []

    def unavailable(
        unused_paths: WebsterPaths,
        unused_state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_paths, unused_state, unused_fetcher
        calls.append(str(record["record_id"]))
        return StageResult.retryable_error("synthetic outage")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            commoncrawl_stage=unavailable,  # type: ignore[arg-type]
            show_progress=False,
        ).stage_commoncrawl()

        assert calls == ["realisticvision-0000"]
        for record_id in manifest["record_id"].astype(str):
            result = state.stage_result(
                record_id, RecoveryStage.COMMON_CRAWL
            )
            assert result is not None
            assert result.outcome is StageOutcome.RETRYABLE_ERROR


def test_valid_recovered_target_is_skipped_by_every_recovery_stage(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    paths = WebsterPaths.from_root(tmp_path)
    calls: list[str] = []

    def forbidden(*arguments: object, **values: object) -> StageResult:
        del arguments, values
        calls.append("called")
        raise AssertionError("recovered target was scheduled again")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        record = state.get_record("sdv1-0000")
        assert record is not None
        _recover_synthetic_image(paths, state, record)
        recovered = state.get_record("sdv1-0000")
        assert recovered is not None
        normalized = Path(str(recovered["local_normalized_path"]))
        before = (normalized.stat().st_mtime_ns, normalized.read_bytes())

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            direct_stage=forbidden,  # type: ignore[arg-type]
            mirror_stage=forbidden,  # type: ignore[arg-type]
            wayback_stage=forbidden,  # type: ignore[arg-type]
            commoncrawl_stage=forbidden,  # type: ignore[arg-type]
            show_progress=False,
        ).run()

        assert calls == []
        assert state.stage_events("sdv1-0000") == []
        assert (normalized.stat().st_mtime_ns, normalized.read_bytes()) == before


def test_corrupt_recovered_artifact_retries_only_that_record(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    paths = WebsterPaths.from_root(tmp_path)
    direct_calls: list[str] = []
    forbidden_calls: list[str] = []

    def direct_repair(
        supplied_paths: WebsterPaths,
        state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_fetcher
        direct_calls.append(str(record["record_id"]))
        _recover_synthetic_image(supplied_paths, state, record)
        return StageResult(StageOutcome.RECOVERED, "repaired")

    def forbidden(*arguments: object, **values: object) -> StageResult:
        del arguments, values
        forbidden_calls.append("called")
        raise AssertionError("unrelated archive stage was scheduled")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        for stage in (
            RecoveryStage.EXACT_REUSE,
            RecoveryStage.OFFICIAL_ASSETS,
        ):
            state.begin_stage("sdv1-0000", stage)
            state.finish_stage(
                "sdv1-0000",
                stage,
                StageResult(StageOutcome.MISS, "prior miss"),
            )
        record = state.get_record("sdv1-0000")
        assert record is not None
        _recover_synthetic_image(paths, state, record)
        recovered = state.get_record("sdv1-0000")
        assert recovered is not None
        Path(str(recovered["local_normalized_path"])).write_bytes(
            b"corrupt"
        )

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            direct_stage=direct_repair,  # type: ignore[arg-type]
            mirror_stage=forbidden,  # type: ignore[arg-type]
            wayback_stage=forbidden,  # type: ignore[arg-type]
            arquivo_stage=forbidden,  # type: ignore[arg-type]
            commoncrawl_stage=forbidden,  # type: ignore[arg-type]
            show_progress=False,
        ).run()

        assert direct_calls == ["sdv1-0000"]
        assert forbidden_calls == []
        current = state.get_record("sdv1-0000")
        assert current is not None
        assert (
            current["recovery_status"]
            == RecoveryStatus.RECOVERED_EXACT_URL.value
        )
        assert images_module.recovery_artifacts_valid(paths, current)
        validations = [
            attempt["validation_result"]
            for attempt in state.attempts("sdv1-0000")
        ]
        assert "recovered_artifacts_invalid" in validations


def test_corrupt_mirror_artifact_reactivates_only_mirror_stage(
    tmp_path: Path,
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    with RecoveryState(paths) as state:
        state.initialize_records(_manifest())
        record = state.get_record("sdv1-0000")
        assert record is not None
        state.begin_stage(
            "sdv1-0000", RecoveryStage.GROUND_TRUTH_MIRROR
        )
        _recover_synthetic_image(
            paths,
            state,
            record,
            status=RecoveryStatus.RECOVERED_GROUND_TRUTH_MIRROR,
        )
        state.finish_stage(
            "sdv1-0000",
            RecoveryStage.GROUND_TRUTH_MIRROR,
            StageResult(StageOutcome.RECOVERED, "synthetic mirror"),
        )
        recovered = state.get_record("sdv1-0000")
        assert recovered is not None
        Path(str(recovered["local_normalized_path"])).write_bytes(b"corrupt")

        assert state.reactivate_invalid_recovered_artifacts(
            "sdv1-0000", reason="corrupt mirror artifact"
        )
        mirror_result = state.stage_result(
            "sdv1-0000", RecoveryStage.GROUND_TRUTH_MIRROR
        )
        assert mirror_result is not None
        assert mirror_result.outcome is StageOutcome.RETRYABLE_ERROR
        assert state.stage_result("sdv1-0000", RecoveryStage.DIRECT) is None


def test_named_group_donor_rejects_partial_identity_matches(
    tmp_path: Path,
) -> None:
    manifest = _duplicate_target_manifest()
    manifest.loc[1, "original_url"] = "https://example.invalid/other.png"
    manifest.loc[1, "normalized_url"] = "https://example.invalid/other.png"
    paths = WebsterPaths.from_root(tmp_path)
    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        donor = state.get_record("sdv1-0000")
        alias = state.get_record("realisticvision-0000")
        assert donor is not None and alias is not None
        _recover_synthetic_image(paths, state, donor)
        recovered_donor = state.get_record("sdv1-0000")
        assert recovered_donor is not None

        assert not recovery_module.reuse_exact_group_duplicate(
            paths,
            state,
            alias,
            recovered_donor,
            "synthetic_group_reuse",
        )
        assert (
            state.get_record("realisticvision-0000")["recovery_status"]
            == RecoveryStatus.PENDING.value
        )


def test_nontransient_wayback_failure_becomes_durable_miss(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    paths = WebsterPaths.from_root(tmp_path)
    calls: list[str] = []

    def forbidden(
        unused_paths: WebsterPaths,
        unused_state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_paths, unused_state, unused_fetcher
        calls.append(str(record["record_id"]))
        raise DownloadError("HTTP 403", status_code=403, transient=False)

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        engine = recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            wayback_stage=forbidden,  # type: ignore[arg-type]
            show_progress=False,
        )
        engine.stage_archive(
            RecoveryStage.WAYBACK,
            forbidden,  # type: ignore[arg-type]
            "[Webster 5/10] Wayback",
        )
        assert calls == ["sdv1-0000"]
        result = state.stage_result(
            "sdv1-0000", RecoveryStage.WAYBACK
        )
        assert result is not None
        assert result.outcome is StageOutcome.MISS

        engine.stage_archive(
            RecoveryStage.WAYBACK,
            forbidden,  # type: ignore[arg-type]
            "[Webster 5/10] Wayback",
        )
        assert calls == ["sdv1-0000"]


def test_commoncrawl_outage_breaker_probes_three_then_advances_all_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    threshold = recovery_module.COMMONCRAWL_CIRCUIT_BREAKER_THRESHOLD
    total = threshold + 3
    base = _manifest().iloc[0].to_dict()
    manifest = pd.DataFrame(
        [
            {
                **base,
                "record_id": f"sdv1-{index:04d}",
                "source_row_number": index,
                "laion_or_sample_index": index,
                "original_url": f"https://example.test/{index}.png",
                "normalized_url": f"https://example.test/{index}.png",
            }
            for index in range(total)
        ]
    )
    stage_calls: list[str] = []
    network_probes: list[str] = []

    class CircuitFetcher:
        circuit_open = False

        def reset_commoncrawl_circuits(self) -> None:
            self.circuit_open = False

        def open_commoncrawl_circuit(
            self, failure_key: str
        ) -> str:
            assert failure_key == (
                "index.commoncrawl.org:connection_refused"
            )
            self.circuit_open = True
            return "index.commoncrawl.org"
    fetcher = CircuitFetcher()

    def unavailable(
        unused_paths: WebsterPaths,
        unused_state: RecoveryState,
        supplied_fetcher: CircuitFetcher,
        record: dict[str, object],
    ) -> StageResult:
        del unused_paths, unused_state
        record_id = str(record["record_id"])
        stage_calls.append(record_id)
        if supplied_fetcher.circuit_open:
            raise DownloadError(
                "Common Crawl circuit is open",
                transient=True,
                service_failure_key=(
                    "index.commoncrawl.org:connection_refused"
                ),
                deferred_by_circuit=True,
            )
        network_probes.append(record_id)
        raise DownloadError(
            "[Errno 111] Connection refused",
            transient=True,
            service_failure_key=(
                "index.commoncrawl.org:connection_refused"
            ),
        )

    with RecoveryState(WebsterPaths.from_root(tmp_path)) as state:
        state.initialize_records(manifest)
        engine = recovery_module.RecoveryEngine(
            state.paths,
            state,
            manifest,
            fetcher,  # type: ignore[arg-type]
            commoncrawl_stage=unavailable,  # type: ignore[arg-type]
        )

        engine.stage_commoncrawl()

        assert stage_calls == [
            f"sdv1-{index:04d}" for index in range(total)
        ]
        assert network_probes == stage_calls[:threshold]
        results = [
            state.stage_result(
                f"sdv1-{index:04d}", RecoveryStage.COMMON_CRAWL
            )
            for index in range(total)
        ]
        assert all(
            result is not None
            and result.outcome is StageOutcome.RETRYABLE_ERROR
            for result in results
        )
        for result in results[threshold:]:
            assert result is not None
            assert "circuit is open" in str(result.message)

        resumed: list[str] = []

        def available(
            unused_paths: WebsterPaths,
            unused_state: RecoveryState,
            unused_fetcher: object,
            record: dict[str, object],
        ) -> StageResult:
            del unused_paths, unused_state, unused_fetcher
            resumed.append(str(record["record_id"]))
            return StageResult(StageOutcome.MISS, "synthetic miss")

        recovery_module.RecoveryEngine(
            state.paths,
            state,
            manifest,
            CircuitFetcher(),  # type: ignore[arg-type]
            commoncrawl_stage=available,  # type: ignore[arg-type]
            show_progress=False,
        ).stage_commoncrawl()
        assert resumed == [f"sdv1-{index:04d}" for index in range(total)]
        assert all(
            state.stage_result(
                f"sdv1-{index:04d}", RecoveryStage.COMMON_CRAWL
            )
            == StageResult(StageOutcome.MISS, "synthetic miss")
            for index in range(total)
        )
        assert state.connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

    output = capsys.readouterr().err
    assert "[Webster 7/10] Common Crawl" in output
    assert f"{total}/{total}" in output
    assert "circuit open 1" in output
    assert f"deferred targets {total - threshold}" in output


def test_commoncrawl_open_circuit_uses_cache_and_only_blocks_failed_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    cached_url = "https://index.commoncrawl.org/cached-index"
    failed_url = "https://index.commoncrawl.org/uncached-index"
    warc_url = "https://data.commoncrawl.org/crawl-data/test.warc.gz"
    network_calls: list[str] = []

    with RecoveryState(paths) as state:
        state.initialize_records(_manifest())
        fetcher = recovery_module.HTTPFetcher(
            paths,
            state,
            client=object(),  # type: ignore[arg-type]
        )

        def request(
            url: str,
            unused_headers: object,
            **unused_options: object,
        ) -> tuple[bytes, int, str, str, float]:
            del unused_headers, unused_options
            network_calls.append(url)
            return b"cached metadata", 200, "application/json", url, 0.0

        monkeypatch.setattr(fetcher, "_request", request)
        initial = fetcher.get(
            "sdv1-0000",
            "precache",
            cached_url,
            cache=True,
            attempts=1,
        )
        assert not initial.from_cache
        network_calls.clear()

        assert fetcher.open_commoncrawl_circuit(
            "index.commoncrawl.org:connection_refused"
        ) == "index.commoncrawl.org"
        reused = fetcher.get(
            "sdv1-0000",
            "cached_index",
            cached_url,
            cache=True,
            attempts=1,
        )
        assert reused.from_cache
        assert reused.data == b"cached metadata"

        fetched = fetcher.get(
            "sdv1-0000",
            "commoncrawl_warc_range",
            warc_url,
            attempts=1,
        )
        assert fetched.data == b"cached metadata"

        with pytest.raises(DownloadError) as captured:
            fetcher.get(
                "sdv1-0000",
                "uncached_index",
                failed_url,
                attempts=1,
            )
        assert captured.value.deferred_by_circuit
        assert captured.value.service_failure_key == (
            "index.commoncrawl.org:connection_refused"
        )
        assert network_calls == [warc_url]
        assert fetcher.commoncrawl_activity() == {
            "network_requests": 2,
            "cache_hits": 1,
            "circuit_deferred_requests": 1,
        }


@pytest.mark.parametrize(
    ("message", "failure_kind"),
    [
        (
            "[Errno -2] Name or service not known",
            "permanent_dns_failure",
        ),
        (
            "certificate verify failed: hostname mismatch",
            "tls_verification_failure",
        ),
    ],
)
def test_commoncrawl_permanent_service_failures_trip_breaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    failure_kind: str,
) -> None:
    manifest = _category_manifest()
    paths = WebsterPaths.from_root(tmp_path)
    index_url = "https://index.commoncrawl.org/test-index"
    network_calls: list[str] = []
    stage_calls: list[str] = []

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        fetcher = recovery_module.HTTPFetcher(
            paths,
            state,
            client=object(),  # type: ignore[arg-type]
        )

        def fail_request(
            url: str,
            unused_headers: object,
            **unused_options: object,
        ) -> object:
            del unused_headers, unused_options
            network_calls.append(url)
            raise httpx.ConnectError(message)

        monkeypatch.setattr(fetcher, "_request", fail_request)

        def unavailable(
            unused_paths: WebsterPaths,
            unused_state: RecoveryState,
            supplied_fetcher: recovery_module.HTTPFetcher,
            record: dict[str, object],
        ) -> StageResult:
            del unused_paths, unused_state
            record_id = str(record["record_id"])
            stage_calls.append(record_id)
            supplied_fetcher.get(
                record_id,
                "commoncrawl_cdx_pages",
                index_url,
                attempts=1,
            )
            raise AssertionError("unreachable")

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            fetcher,
            commoncrawl_stage=unavailable,  # type: ignore[arg-type]
            show_progress=False,
        ).stage_commoncrawl()

        threshold = (
            recovery_module.COMMONCRAWL_CIRCUIT_BREAKER_THRESHOLD
        )
        assert len(stage_calls) == len(manifest)
        assert network_calls == [index_url] * threshold
        assert all(
            state.stage_result(
                str(record_id), RecoveryStage.COMMON_CRAWL
            ).outcome
            is StageOutcome.RETRYABLE_ERROR
            for record_id in manifest["record_id"]
        )
        assert recovery_module._transport_failure_key(
            httpx.ConnectError(message), index_url
        ) == f"index.commoncrawl.org:{failure_kind}"


def test_commoncrawl_warc_keeps_bounded_retries_until_host_circuit_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    calls: list[str] = []
    capture = {
        "filename": "crawl-data/test.warc.gz",
        "offset": 0,
        "length": 10,
        "collection": "CC-MAIN-TEST",
        "timestamp": "20200101000000",
    }

    with RecoveryState(paths) as state:
        state.initialize_records(_manifest())
        fetcher = recovery_module.HTTPFetcher(
            paths,
            state,
            client=object(),  # type: ignore[arg-type]
        )

        def fail_request(
            url: str,
            unused_headers: object,
            **unused_options: object,
        ) -> object:
            del unused_headers, unused_options
            calls.append(url)
            raise httpx.ConnectError("[Errno 111] Connection refused")

        monkeypatch.setattr(fetcher, "_request", fail_request)
        monkeypatch.setattr(recovery_module.time, "sleep", lambda _: None)

        with pytest.raises(DownloadError):
            commoncrawl_module.fetch_commoncrawl_warc(
                fetcher, "sdv1-0000", capture
            )
        assert len(calls) == recovery_module.DEFAULT_HTTP_ATTEMPTS

        assert fetcher.open_commoncrawl_circuit(
            "data.commoncrawl.org:connection_refused"
        ) == "data.commoncrawl.org"
        with pytest.raises(DownloadError) as captured:
            commoncrawl_module.fetch_commoncrawl_warc(
                fetcher, "sdv1-0000", capture
            )
        assert captured.value.deferred_by_circuit
        assert len(calls) == recovery_module.DEFAULT_HTTP_ATTEMPTS



def test_commoncrawl_cdx_metadata_requests_are_single_attempt() -> None:
    calls: list[tuple[str, int]] = []

    class FakeState:
        def update_attempt(self, *arguments: object, **values: object) -> None:
            del arguments, values

        def invalidate_http_cache(self, request_key: str) -> None:
            raise AssertionError(f"unexpected invalid cache: {request_key}")

    class Result:
        def __init__(self, data: bytes, request_key: str) -> None:
            self.data = data
            self.request_key = request_key
            self.attempt_id = 1

    class FakeFetcher:
        state = FakeState()

        def get(
            self,
            record_id: str,
            strategy: str,
            url: str,
            **options: object,
        ) -> Result:
            del record_id, url
            calls.append((strategy, int(options["attempts"])))
            if strategy == "commoncrawl_index_list":
                return Result(
                    b'[{"id":"CC-MAIN-TEST","cdx-api":'
                    b'"https://index.commoncrawl.org/test-index"}]',
                    "indexes",
                )
            if strategy == "commoncrawl_cdx_pages":
                return Result(b'{"pages":1}', "pages")
            assert strategy == "commoncrawl_cdx"
            return Result(b"", "records")

    fetcher = FakeFetcher()
    indexes = commoncrawl_module.commoncrawl_indexes(
        fetcher, fetcher.state, "sdv1-0000"  # type: ignore[arg-type]
    )
    assert commoncrawl_module.query_commoncrawl_url(
        fetcher,  # type: ignore[arg-type]
        "sdv1-0000",
        "https://example.test/image.png",
        indexes,
    ) == []
    assert calls == [
        ("commoncrawl_index_list", 1),
        ("commoncrawl_cdx_pages", 1),
        ("commoncrawl_cdx", 1),
    ]


def test_metadata_acquisition_uses_one_tqdm_without_plain_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    paths.create()
    for filename in webster_module.MODEL_FILES.values():
        (paths.source_parquets / Path(filename).name).write_bytes(b"cached")
    manifest = _category_manifest()

    class MetadataState:
        def get_run_metadata(self, key: str) -> None:
            del key
            return None

        def initialize_records(self, value: pd.DataFrame) -> None:
            assert value is manifest

        def set_run_metadata_batch(self, values: object) -> None:
            assert values

    monkeypatch.setattr(
        webster_module, "load_source_manifest", lambda unused: manifest
    )
    monkeypatch.setattr(
        webster_module,
        "atomic_write_frame_parquet",
        lambda *arguments: None,
    )

    result = webster_module.acquire_and_build_manifest(
        paths,
        MetadataState(),  # type: ignore[arg-type]
    )

    assert result is manifest
    output = capsys.readouterr().err
    assert "[Webster metadata] Source files" in output
    assert "3/3" in output
    assert "Fetching" not in output
    assert "Reusing" not in output


def test_publish_and_validation_progress_stays_visible_with_postfix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    manifest = pd.DataFrame(
        {
            "model_name": [model.value for model in webster_module.WebsterModel],
            "source_row_number": [0, 0, 0],
        }
    )
    postfix = "MV 1/1 | TV 2/2 | RV 3/3 | N 4/4 | Total 10/10"
    prepared = pd.DataFrame(
        {"image_filename": [None], "image_available": [False]}
    )
    monkeypatch.setattr(
        webster_module, "validate_source_manifest", lambda value: None
    )
    monkeypatch.setattr(
        webster_module,
        "_prepare_model_rows",
        lambda *arguments: prepared,
    )
    monkeypatch.setattr(
        webster_module,
        "atomic_write_frame_parquet",
        lambda *arguments: None,
    )
    monkeypatch.setattr(
        webster_module,
        "atomic_write_frame_csv",
        lambda *arguments: None,
    )
    monkeypatch.setattr(
        webster_module, "atomic_write_json", lambda *arguments: None
    )
    monkeypatch.setattr(
        webster_module, "file_sha256", lambda unused: "0" * 64
    )

    organization = webster_module.organize_model_views(
        paths,
        manifest,
        show_progress=True,
        progress_postfix=lambda: postfix,
    )
    publish_output = capsys.readouterr().err

    assert len(organization.models) == 3
    assert "[Webster 9/10] Publish model views" in publish_output
    assert "3/3" in publish_output
    assert postfix in publish_output

    monkeypatch.setattr(
        webster_module,
        "inspect_model",
        lambda *arguments: object(),
    )
    inspection = webster_module.inspect_webster_dataset(
        paths.root,
        show_progress=True,
        progress_postfix=lambda: postfix,
    )
    validation_output = capsys.readouterr().err

    assert len(inspection.models) == 3
    assert "[Webster 10/10] Validate model views" in validation_output
    assert "3/3" in validation_output
    assert postfix in validation_output


def test_direct_records_are_interleaved_across_hosts() -> None:
    records = [
        {"record_id": "a1", "original_url": "https://a.test/1"},
        {"record_id": "a2", "original_url": "https://a.test/2"},
        {"record_id": "a3", "original_url": "https://a.test/3"},
        {"record_id": "b1", "original_url": "https://b.test/1"},
        {"record_id": "b2", "original_url": "https://b.test/2"},
        {"record_id": "c1", "original_url": "https://c.test/1"},
        {"record_id": "bad", "original_url": "https://[malformed/"},
    ]

    ordered = recovery_module._interleave_direct_records(records)

    assert [record["record_id"] for record in ordered] == [
        "a1", "b1", "c1", "bad", "a2", "b2", "a3",
    ]


def test_direct_stage_uses_configured_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_workers: list[int] = []
    record = {
        "record_id": "sdv1-0000",
        "recovery_status": RecoveryStatus.PENDING.value,
        "original_url": "https://example.test/image.png",
    }

    class OneRecordState:
        def records(self, *, statuses: object) -> list[dict[str, object]]:
            del statuses
            return [record]

        def stage_result(self, record_id: str, stage: object) -> None:
            del record_id, stage
            return None

    class ImmediateExecutor:
        def __init__(self, *, max_workers: int) -> None:
            observed_workers.append(max_workers)

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *arguments: object) -> None:
            del arguments

        def submit(self, function: object, *arguments: object) -> concurrent.futures.Future[object]:
            del function, arguments
            future: concurrent.futures.Future[object] = concurrent.futures.Future()
            future.set_result(StageResult(StageOutcome.MISS, "synthetic"))
            return future

        def shutdown(
            self, *, wait: bool, cancel_futures: bool = False,
        ) -> None:
            del wait, cancel_futures

    monkeypatch.setattr(
        recovery_module.concurrent.futures,
        "ThreadPoolExecutor",
        ImmediateExecutor,
    )
    engine = recovery_module.RecoveryEngine(
        WebsterPaths.from_root(tmp_path),
        OneRecordState(),  # type: ignore[arg-type]
        pd.DataFrame(),
        object(),  # type: ignore[arg-type]
        show_progress=False,
        direct_workers=17,
    )

    engine.stage_direct()

    assert observed_workers == [17]


def test_permanent_dns_fails_once_even_with_larger_attempt_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    class AttemptState:
        def log_attempt(self, record_id: str, strategy: str, **values: object) -> int:
            del record_id, strategy, values
            attempts.append("failed")
            return len(attempts)

    fetcher = recovery_module.HTTPFetcher(
        WebsterPaths.from_root(tmp_path),
        AttemptState(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
    )

    def fail_request(*arguments: object, **keywords: object) -> object:
        del arguments, keywords
        raise httpx.ConnectError("[Errno -2] Name or service not known")

    monkeypatch.setattr(fetcher, "_request", fail_request)
    with pytest.raises(DownloadError) as captured:
        fetcher.get(
            "sdv1-0000",
            "direct_url",
            "https://dead.test/image.png",
            attempts=5,
        )

    assert captured.value.permanent_host_failure
    assert not captured.value.transient
    assert attempts == ["failed"]
    assert recovery_module._permanent_dns_failure(
        socket.gaierror(socket.EAI_NONAME, "unknown host")
    )
    assert not recovery_module._permanent_dns_failure(
        socket.gaierror(socket.EAI_AGAIN, "temporary failure")
    )


def test_temporary_dns_retries_and_permanent_tls_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    class AttemptState:
        def log_attempt(self, *arguments: object, **values: object) -> int:
            del arguments, values
            attempts.append("failed")
            return len(attempts)

    fetcher = recovery_module.HTTPFetcher(
        WebsterPaths.from_root(tmp_path),
        AttemptState(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
    )

    def temporary_dns(*arguments: object, **keywords: object) -> object:
        del arguments, keywords
        raise httpx.ConnectError("[Errno -3] Temporary failure in name resolution")

    monkeypatch.setattr(fetcher, "_request", temporary_dns)
    monkeypatch.setattr(recovery_module.time, "sleep", sleeps.append)
    with pytest.raises(DownloadError) as captured:
        fetcher.get(
            "sdv1-0000",
            "direct_url",
            "https://temporary.test/image.png",
            attempts=2,
        )

    assert captured.value.transient
    assert attempts == ["failed", "failed"]
    assert len(sleeps) == 1
    assert recovery_module._permanent_tls_failure(
        httpx.ConnectError("certificate verify failed: hostname mismatch")
    )


def test_per_host_concurrency_is_bounded(tmp_path: Path) -> None:
    fetcher = recovery_module.HTTPFetcher(
        WebsterPaths.from_root(tmp_path),
        object(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        direct_workers=3,
        per_host_concurrency=3,
    )
    semaphore = fetcher._semaphore("EXAMPLE.test")

    assert fetcher._semaphore("example.TEST") is semaphore
    assert [semaphore.acquire(blocking=False) for _ in range(3)] == [
        True, True, True,
    ]
    assert not semaphore.acquire(blocking=False)
    for _ in range(3):
        semaphore.release()


def test_recovery_state_supports_24_concurrent_attempt_writers(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    base = _manifest().iloc[0].to_dict()
    for index in range(48):
        rows.append(
            {
                **base,
                "record_id": f"sdv1-{index:04d}",
                "source_row_number": index,
                "laion_or_sample_index": index,
                "original_url": f"https://example.test/{index}.png",
                "normalized_url": f"https://example.test/{index}.png",
            }
        )
    paths = WebsterPaths.from_root(tmp_path)
    with RecoveryState(paths) as state:
        state.initialize_records(pd.DataFrame(rows))

        def write_attempt(index: int) -> int:
            return state.log_attempt(
                f"sdv1-{index:04d}",
                "direct_url",
                attempted_url=f"https://example.test/{index}.png",
                validation_result="synthetic",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
            attempt_ids = list(executor.map(write_attempt, range(48)))

        assert len(set(attempt_ids)) == 48
        assert len(state.attempts()) == 48
        assert state.connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"


def test_direct_dns_failure_skips_same_host_variants_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    class FailingFetcher:
        direct_attempts = 2

        def get(self, record_id: str, strategy: str, url: str, **values: object) -> None:
            del record_id, strategy
            calls.append((url, int(values["attempts"])))
            raise DownloadError(
                "host does not exist",
                permanent_host_failure=True,
            )

    monkeypatch.setattr(recovery_module, "reuse_duplicate", lambda *args: False)
    monkeypatch.setattr(
        recovery_module,
        "safe_url_variants",
        lambda value: [
            "https://dead.test/image.png",
            "http://dead.test/image.png",
            "https://www.dead.test/image.png",
        ],
    )

    result = recovery_module.recover_direct(
        WebsterPaths.from_root(tmp_path),
        object(),  # type: ignore[arg-type]
        FailingFetcher(),  # type: ignore[arg-type]
        {"record_id": "sdv1-0000", "original_url": "https://dead.test/image.png"},
    )

    assert result.outcome is StageOutcome.MISS
    assert calls == [
        ("https://dead.test/image.png", 2),
        ("https://www.dead.test/image.png", 2),
    ]


def test_terminal_status_cannot_be_silently_reopened(tmp_path: Path) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    with RecoveryState(paths) as state:
        state.initialize_records(_manifest())
        state.set_status("sdv1-0000", RecoveryStatus.UNRESOLVED, reason="exhausted")
        with pytest.raises(Exception, match="terminal"):
            state.set_status("sdv1-0000", RecoveryStatus.PENDING)


def test_old_or_partial_database_is_rejected_without_migration(tmp_path: Path) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    paths.state.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(paths.database) as connection:
        connection.execute("CREATE TABLE records(record_id TEXT PRIMARY KEY)")
    with pytest.raises(StateSchemaError, match="current schema v2"):
        RecoveryState(paths)


def test_final_state_module_defines_no_migration_function() -> None:
    source_path = Path(__file__).resolve().parents[1] / "utils/data/state.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    names = {
        node.name.casefold()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not {name for name in names if "migrat" in name}


def test_archive_candidate_success_runs_once_per_exact_target(
    tmp_path: Path,
) -> None:
    manifest = _duplicate_target_manifest()
    paths = WebsterPaths.from_root(tmp_path)
    calls: list[str] = []

    def recover(
        supplied_paths: WebsterPaths,
        state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_fetcher
        record_id = str(record["record_id"])
        calls.append(record_id)
        attempt_id = state.log_attempt(
            record_id,
            "synthetic_wayback",
            attempted_url=str(record["original_url"]),
            validation_result="downloaded_unvalidated",
        )
        result = HTTPResult(
            _tiny_png(),
            200,
            "image/png",
            str(record["original_url"]),
            attempt_id,
            f"synthetic-{record_id}",
            False,
        )
        images_module.preserve_archive_candidate(
            supplied_paths,
            state,
            record,
            result,
            strategy="wayback_exact_image",
            resolved_url=str(record["original_url"]),
            archive_timestamp_value="20240102030405",
            archive_digest=None,
            verified_archive_digest=None,
            source_page=False,
            provenance={"synthetic": True},
        )
        return StageResult(StageOutcome.RECOVERED, "synthetic candidate")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        engine = recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            wayback_stage=recover,  # type: ignore[arg-type]
            show_progress=False,
        )
        engine.stage_archive(
            RecoveryStage.WAYBACK,
            recover,  # type: ignore[arg-type]
            "[Webster 5/10] Wayback",
        )

        assert len(calls) == 1
        candidates = {
            record_id: state.candidates(record_id, prefix="wayback")
            for record_id in manifest["record_id"].astype(str)
        }
        assert all(len(items) == 1 for items in candidates.values())
        assert len(
            {
                str(items[0]["sha256"])
                for items in candidates.values()
            }
        ) == 1
        assert all(
            state.stage_result(record_id, RecoveryStage.WAYBACK).outcome
            is StageOutcome.RECOVERED
            for record_id in candidates
        )
        assert all(
            state.get_record(record_id)["recovery_status"]
            == RecoveryStatus.RECOVERED_WAYBACK_EXACT.value
            for record_id in candidates
        )
        assert (
            engine._active_records_for(RecoveryStage.GROUND_TRUTH_MIRROR) == []
        )
        assert (
            engine._active_records_for(RecoveryStage.ARQUIVO) == []
        )
        assert (
            engine._active_records_for(RecoveryStage.COMMON_CRAWL) == []
        )


def test_arquivo_outage_breaker_is_cache_first_and_defers_after_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threshold = recovery_module.ARQUIVO_CIRCUIT_BREAKER_THRESHOLD
    total = threshold + 3
    base = _manifest().iloc[0].to_dict()
    manifest = pd.DataFrame(
        [
            {
                **base,
                "record_id": f"sdv1-{index:04d}",
                "source_row_number": index,
                "laion_or_sample_index": index,
                "original_url": f"https://example.test/{index}.png",
                "normalized_url": f"https://example.test/{index}.png",
            }
            for index in range(total)
        ]
    )
    paths = WebsterPaths.from_root(tmp_path)
    network_calls: list[str] = []
    stage_calls: list[str] = []
    endpoint = "https://arquivo.pt/textsearch"

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        fetcher = recovery_module.HTTPFetcher(
            paths,
            state,
            client=object(),  # type: ignore[arg-type]
        )

        def fail_request(
            url: str,
            unused_headers: object,
            **unused_options: object,
        ) -> object:
            del unused_headers, unused_options
            network_calls.append(url)
            raise httpx.ConnectError("[Errno 111] Connection refused")

        monkeypatch.setattr(fetcher, "_request", fail_request)
        monkeypatch.setattr(recovery_module.time, "sleep", lambda _: None)

        def unavailable(
            unused_paths: WebsterPaths,
            unused_state: RecoveryState,
            supplied_fetcher: recovery_module.HTTPFetcher,
            record: dict[str, object],
        ) -> StageResult:
            del unused_paths, unused_state
            record_id = str(record["record_id"])
            stage_calls.append(record_id)
            supplied_fetcher.get(
                record_id,
                "arquivo_version_history",
                endpoint,
                cache=True,
                attempts=1,
            )
            raise AssertionError("unreachable")

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            fetcher,
            arquivo_stage=unavailable,  # type: ignore[arg-type]
            show_progress=False,
        ).stage_arquivo()

        assert len(stage_calls) == total
        assert network_calls == [endpoint] * threshold
        assert fetcher.arquivo_activity() == {
            "network_requests": threshold,
            "circuit_deferred_requests": total - threshold,
        }
        assert all(
            state.stage_result(
                record_id, RecoveryStage.ARQUIVO
            ).outcome
            is StageOutcome.RETRYABLE_ERROR
            for record_id in manifest["record_id"].astype(str)
        )


def test_legacy_unresolved_record_reopens_once_for_new_mirror_stage(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    paths = WebsterPaths.from_root(tmp_path)
    old_stages = (
        RecoveryStage.EXACT_REUSE,
        RecoveryStage.OFFICIAL_ASSETS,
        RecoveryStage.DIRECT,
        RecoveryStage.WAYBACK,
        RecoveryStage.ARQUIVO,
        RecoveryStage.COMMON_CRAWL,
    )
    mirror_calls: list[str] = []
    arquivo_calls: list[str] = []

    def mirror_miss(
        unused_paths: WebsterPaths,
        unused_state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_paths, unused_state, unused_fetcher
        mirror_calls.append(str(record["record_id"]))
        return StageResult(StageOutcome.MISS, "synthetic mirror miss")

    def arquivo_miss(
        unused_paths: WebsterPaths,
        unused_state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_paths, unused_state, unused_fetcher
        arquivo_calls.append(str(record["record_id"]))
        return StageResult(StageOutcome.MISS, "synthetic Arquivo miss")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        for stage in old_stages:
            state.begin_stage("sdv1-0000", stage)
            state.finish_stage(
                "sdv1-0000",
                stage,
                StageResult(StageOutcome.MISS, "legacy miss"),
            )
        state.begin_stage(
            "sdv1-0000", RecoveryStage.FINAL_RESOLUTION
        )
        state.set_status(
            "sdv1-0000",
            RecoveryStatus.UNRESOLVED,
            reason="legacy sources exhausted",
        )
        state.finish_stage(
            "sdv1-0000",
            RecoveryStage.FINAL_RESOLUTION,
            StageResult(StageOutcome.MISS, "legacy sources exhausted"),
        )

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            mirror_stage=mirror_miss,  # type: ignore[arg-type]
            arquivo_stage=arquivo_miss,  # type: ignore[arg-type]
            show_progress=False,
        ).run()

        assert mirror_calls == ["sdv1-0000"]
        assert arquivo_calls == []
        assert (
            state.get_record("sdv1-0000")["recovery_status"]
            == RecoveryStatus.UNRESOLVED.value
        )
        assert [
            attempt["validation_result"]
            for attempt in state.attempts("sdv1-0000")
            if attempt["strategy"] == "state_reactivation"
        ] == [
            "reactivated_for_ground_truth_mirror_v"
            f"{recovery_module.MIRROR_STRATEGY_VERSION}"
        ]

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            mirror_stage=mirror_miss,  # type: ignore[arg-type]
            arquivo_stage=arquivo_miss,  # type: ignore[arg-type]
            show_progress=False,
        ).run()
        assert mirror_calls == ["sdv1-0000"]
        assert arquivo_calls == []


def test_legacy_unresolved_reactivation_rejects_partial_or_wrong_stage(
    tmp_path: Path,
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    with RecoveryState(paths) as state:
        state.initialize_records(_manifest())
        state.set_status(
            "sdv1-0000",
            RecoveryStatus.UNRESOLVED,
            reason="partial legacy state",
        )

        assert not state.reactivate_unresolved_for_stage(
            "sdv1-0000",
            RecoveryStage.ARQUIVO,
            reason="new strategy",
        )
        with pytest.raises(ValueError, match="only be reopened"):
            state.reactivate_unresolved_for_stage(
                "sdv1-0000",
                RecoveryStage.COMMON_CRAWL,
                reason="unsafe source",
            )
        assert (
            state.get_record("sdv1-0000")["recovery_status"]
            == RecoveryStatus.UNRESOLVED.value
        )
        assert state.attempts("sdv1-0000") == []


def test_arquivo_strategy_version_bump_retries_only_prior_unresolved_miss(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    paths = WebsterPaths.from_root(tmp_path)
    old_source_stages = (
        RecoveryStage.EXACT_REUSE,
        RecoveryStage.OFFICIAL_ASSETS,
        RecoveryStage.DIRECT,
        RecoveryStage.GROUND_TRUTH_MIRROR,
        RecoveryStage.WAYBACK,
        RecoveryStage.COMMON_CRAWL,
    )
    calls: list[str] = []

    def upgraded_miss(
        unused_paths: WebsterPaths,
        unused_state: RecoveryState,
        unused_fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        del unused_paths, unused_state, unused_fetcher
        calls.append(str(record["record_id"]))
        return StageResult(StageOutcome.MISS, "upgraded miss")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        for stage in old_source_stages:
            state.begin_stage("sdv1-0000", stage)
            state.finish_stage(
                "sdv1-0000",
                stage,
                StageResult(StageOutcome.MISS, "old miss"),
            )
        state.begin_stage("sdv1-0000", RecoveryStage.ARQUIVO)
        state.finish_stage(
            "sdv1-0000",
            RecoveryStage.ARQUIVO,
            StageResult(StageOutcome.MISS, "strategy v1 miss"),
        )
        state.begin_stage(
            "sdv1-0000", RecoveryStage.FINAL_RESOLUTION
        )
        state.set_status(
            "sdv1-0000",
            RecoveryStatus.UNRESOLVED,
            reason="all v1 sources exhausted",
        )
        state.finish_stage(
            "sdv1-0000",
            RecoveryStage.FINAL_RESOLUTION,
            StageResult(StageOutcome.MISS, "all v1 sources exhausted"),
        )
        state.set_run_metadata("arquivo_strategy_version", 1)
        state.set_run_metadata(
            "ground_truth_mirror_strategy_version",
            recovery_module.MIRROR_STRATEGY_VERSION,
        )

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            arquivo_stage=upgraded_miss,  # type: ignore[arg-type]
            show_progress=False,
        ).run()

        assert calls == ["sdv1-0000"]
        assert (
            state.get_run_metadata("arquivo_strategy_version")
            == recovery_module.ARQUIVO_STRATEGY_VERSION
        )
        assert all(
            state.stage_result("sdv1-0000", stage).message
            == "old miss"
            for stage in old_source_stages
        )
        assert (
            state.stage_result(
                "sdv1-0000", RecoveryStage.ARQUIVO
            ).message
            == "upgraded miss"
        )

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            object(),  # type: ignore[arg-type]
            arquivo_stage=upgraded_miss,  # type: ignore[arg-type]
            show_progress=False,
        ).run()
        assert calls == ["sdv1-0000"]


def test_arquivo_open_circuit_still_reuses_cached_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    cached_url = "https://arquivo.pt/textsearch?cached=1"
    uncached_url = "https://arquivo.pt/textsearch?cached=0"
    network_calls: list[str] = []

    with RecoveryState(paths) as state:
        state.initialize_records(_manifest())
        fetcher = recovery_module.HTTPFetcher(
            paths,
            state,
            client=object(),  # type: ignore[arg-type]
        )

        def request(
            url: str,
            unused_headers: object,
            **unused_options: object,
        ) -> tuple[bytes, int, str, str, float]:
            del unused_headers, unused_options
            network_calls.append(url)
            return b"cached metadata", 200, "application/json", url, 0.0

        monkeypatch.setattr(fetcher, "_request", request)
        fetcher.get(
            "sdv1-0000",
            "precache",
            cached_url,
            cache=True,
            attempts=1,
            follow_redirects=False,
        )
        network_calls.clear()
        fetcher.reset_arquivo_activity()

        assert fetcher.open_arquivo_circuit(
            "arquivo.pt:connection_refused"
        ) == "arquivo.pt"
        reused = fetcher.get(
            "sdv1-0000",
            "cached_arquivo",
            cached_url,
            cache=True,
            attempts=1,
            follow_redirects=False,
        )
        assert reused.from_cache

        with pytest.raises(DownloadError) as captured:
            fetcher.get(
                "sdv1-0000",
                "uncached_arquivo",
                uncached_url,
                cache=True,
                attempts=1,
                follow_redirects=False,
            )
        assert captured.value.deferred_by_circuit
        assert network_calls == []
        assert fetcher.arquivo_activity() == {
            "cache_hits": 1,
            "circuit_deferred_requests": 1,
        }
