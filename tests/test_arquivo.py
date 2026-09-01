"""Offline tests for exact-URL Arquivo.pt recovery."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from utils.data import arquivo as arquivo_module
from utils.data import recovery as recovery_module
from utils.data.state import RecoveryState
from utils.data.wayback import archive_sha1
from utils.data.webster import (
    HTTPResult,
    RecoveryStatus,
    StageOutcome,
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
                "original_url": "https://example.test/source.png",
                "normalized_url": "https://example.test/source.png",
                "source_page_url": None,
            }
        ]
    )


def _tiny_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(output, format="PNG")
    return output.getvalue()


def _response(
    *,
    original: str = "https://example.test/source.png",
    timestamp: str = "20240102030405",
    digest: str = "A" * 32,
    estimated: int = 1,
    replay_url: str | None = None,
) -> bytes:
    return json.dumps(
        {
            "estimated_nr_results": estimated,
            "response_items": [
                {
                    "originalURL": original,
                    "tstamp": timestamp,
                    "digest": digest,
                    "mimeType": "image/png",
                    "statusCode": 200,
                    "collection": "TEST",
                    "fileName": "test.warc.gz",
                    "offset": 123,
                    "linkToOriginalFile": (
                        replay_url
                        if replay_url is not None
                        else (
                            "https://arquivo.pt/noFrame/replay/"
                            f"{timestamp}id_/{original}"
                        )
                    ),
                }
            ],
        }
    ).encode("utf-8")


def test_parse_arquivo_response_validates_current_api_shape() -> None:
    captures, estimated, returned, malformed = arquivo_module.parse_arquivo_response(
        _response()
    )

    assert estimated == 1
    assert returned == 1
    assert malformed == 0
    assert captures == [
        {
            "original": "https://example.test/source.png",
            "timestamp": "20240102030405",
            "digest": "A" * 32,
            "reported_digest": "A" * 32,
            "mimetype": "image/png",
            "replay_url": (
                "https://arquivo.pt/noFrame/replay/"
                "20240102030405id_/"
                "https://example.test/source.png"
            ),
            "collection": "TEST",
            "filename": "test.warc.gz",
            "offset": "123",
            "content_length": "",
        }
    ]

    unsupported, _, _, malformed = arquivo_module.parse_arquivo_response(
        _response(digest="not-a-supported-digest")
    )
    assert unsupported == []
    assert malformed == 1


@pytest.mark.parametrize(
    "url",
    [
        (
            "https://evil.test/noFrame/replay/"
            "20240102030405id_/https://example.test/source.png"
        ),
        (
            "https://arquivo.pt/noFrame/replay/"
            "20240102030406id_/https://example.test/source.png"
        ),
        (
            "https://arquivo.pt/noFrame/replay/"
            "20240102030405im_/https://example.test/source.png"
        ),
        (
            "https://arquivo.pt/noFrame/replay/"
            "20240102030405id_/https://example.test/other.png"
        ),
    ],
)
def test_arquivo_replay_identity_rejects_semantic_changes(
    url: str,
) -> None:
    assert not arquivo_module.arquivo_replay_matches(
        url,
        "20240102030405",
        "https://example.test/source.png",
    )


def test_arquivo_replay_identity_accepts_both_raw_replay_forms() -> None:
    original = "https://example.test/a path/source.png?x=1"
    encoded = "https://example.test/a%20path/source.png?x=1"
    assert arquivo_module.arquivo_replay_matches(
        ("https://arquivo.pt/noFrame/replay/" f"20240102030405id_/{encoded}"),
        "20240102030405",
        original,
    )
    assert arquivo_module.arquivo_replay_matches(
        ("https://arquivo.pt/noFrame/replay/" f"20240102030405/id_/{encoded}"),
        "20240102030405",
        original,
    )


@pytest.mark.parametrize("digest_algorithm", ["sha1", "md5"])
def test_exact_arquivo_recovery_preserves_verified_candidate(
    tmp_path: Path,
    digest_algorithm: str,
) -> None:
    manifest = _manifest()
    paths = WebsterPaths.from_root(tmp_path)
    image = _tiny_png()
    digest = (
        archive_sha1(image).upper()
        if digest_algorithm == "sha1"
        else hashlib.md5(image, usedforsecurity=False).hexdigest()
    )
    expected_digest = f"{digest_algorithm}:{digest.casefold()}"
    original = str(manifest.iloc[0]["original_url"])
    timestamp = "20240102030405"
    replay_url = arquivo_module.build_arquivo_replay_url(timestamp, original)

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)

        class Fetcher:
            def __init__(self) -> None:
                self.state = state
                self.calls: list[tuple[str, str]] = []

            def get(
                self,
                record_id: str,
                strategy: str,
                url: str,
                **options: object,
            ) -> HTTPResult:
                self.calls.append((strategy, url))
                attempt_id = state.log_attempt(
                    record_id,
                    strategy,
                    attempted_url=url,
                    validation_result="downloaded_unvalidated",
                )
                if strategy == "arquivo_version_history":
                    queried = str(
                        (options.get("params") or {}).get("versionHistory", "")
                    )
                    data = (
                        _response(
                            original=original,
                            timestamp=timestamp,
                            digest=digest,
                            replay_url="https://invalid.test/replay",
                        )
                        if queried == original
                        else json.dumps(
                            {
                                "estimated_nr_results": 0,
                                "response_items": [],
                            }
                        ).encode("utf-8")
                    )
                    return HTTPResult(
                        data,
                        200,
                        "application/json",
                        url,
                        attempt_id,
                        f"metadata-{len(self.calls)}",
                        False,
                    )
                assert strategy == "arquivo_replay"
                return HTTPResult(
                    image,
                    200,
                    "image/png",
                    replay_url,
                    attempt_id,
                    "replay",
                    False,
                )

        fetcher = Fetcher()
        record = state.get_record("sdv1-0000")
        assert record is not None
        result = arquivo_module.recover_arquivo(
            paths,
            state,
            fetcher,  # type: ignore[arg-type]
            record,
        )

        assert result.outcome is StageOutcome.RECOVERED
        candidates = state.candidates("sdv1-0000", prefix="arquivo")
        assert len(candidates) == 1
        assert candidates[0]["archive_digest"] == expected_digest
        provenance = json.loads(candidates[0]["metadata_json"])
        assert provenance["supplied_replay_url"] == "https://invalid.test/replay"
        assert provenance["requested_replay_url"] == replay_url
        assert provenance["replay_final_url"] == replay_url
        assert provenance["verified_digest_algorithm"] == digest_algorithm
        resolved = recovery_module.resolve_candidates(paths, state, record, candidates)
        assert resolved.outcome is StageOutcome.RECOVERED
        assert (
            state.get_record("sdv1-0000")["recovery_status"]
            == RecoveryStatus.RECOVERED_ARQUIVO_EXACT.value
        )


def test_arquivo_digest_options_cover_legacy_modern_and_ambiguous() -> None:
    assert arquivo_module.arquivo_digest_options(
        "0123456789abcdef0123456789abcdef"
    ) == ("md5:0123456789abcdef0123456789abcdef",)
    assert arquivo_module.arquivo_digest_options("A" * 32) == (
        "md5:" + "a" * 32,
        "sha1:" + "a" * 32,
    )
    assert arquivo_module.arquivo_digest_options(
        "FRQI6VBULMPYF2VMWAK74VFEERNGCYLJ"
    ) == ("sha1:frqi6vbulmpyf2vmwak74vfeerngcylj",)


def test_history_offsets_span_oldest_and_newest_captures() -> None:
    offsets = arquivo_module.arquivo_history_offsets(1_000)
    assert offsets == [0, 317, 633, 950]


def test_target_wide_capture_selection_has_one_global_replay_budget() -> None:
    captures = [
        {
            "original": f"https://example.test/{index}.png",
            "timestamp": f"2024{index + 1:010d}"[-14:],
            "digest": "",
            "reported_digest": "",
            "mimetype": "image/png",
            "replay_url": f"https://arquivo.pt/replay/{index}",
        }
        for index in range(30)
    ]
    selected = arquivo_module.select_arquivo_captures(captures)
    assert len(selected) == arquivo_module.MAX_ARQUIVO_REPLAYS
    assert (
        len(
            {
                (
                    item["original"],
                    item["timestamp"],
                    item["replay_url"],
                )
                for item in selected
            }
        )
        == arquivo_module.MAX_ARQUIVO_REPLAYS
    )


def test_arquivo_captures_queries_offsets_across_full_history() -> None:
    original = "https://example.test/source.png"
    offsets: list[int] = []

    class State:
        def update_attempt(self, *arguments: object, **values: object) -> None:
            del arguments, values

        def invalidate_http_cache(self, request_key: str) -> None:
            raise AssertionError(f"unexpected invalid cache: {request_key}")

    class Fetcher:
        state = State()

        def get(
            self,
            record_id: str,
            strategy: str,
            url: str,
            **options: object,
        ) -> HTTPResult:
            del record_id
            assert strategy == "arquivo_version_history"
            assert url == arquivo_module.ARQUIVO_SEARCH_URL
            assert options["follow_redirects"] is False
            parameters = options["params"]
            assert isinstance(parameters, dict)
            offset = int(parameters["offset"])
            offsets.append(offset)
            timestamp = f"{20000101000000 + offset:014d}"
            return HTTPResult(
                _response(
                    original=original,
                    timestamp=timestamp,
                    digest="",
                    estimated=1_000,
                ),
                200,
                "application/json",
                url,
                offset + 1,
                f"page-{offset}",
                False,
            )

    captures = arquivo_module.arquivo_captures(
        Fetcher(),  # type: ignore[arg-type]
        "sdv1-0000",
        original,
    )

    assert offsets == [0, 317, 633, 950]
    assert len(captures) == 4
    assert {item["history_offset"] for item in captures} == {"0", "317", "633", "950"}
