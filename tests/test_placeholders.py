"""Offline regressions for rejecting missing-image substitutes during recovery."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tests.test_recovery import _manifest, _recover_synthetic_image
from utils.data import images as images_module
from utils.data import recovery as recovery_module
from utils.data import webster as webster_module
from utils.data.state import RecoveryState
from utils.data.webster import (
    HTTPResult,
    ImageValidationError,
    RecoveryStage,
    RecoveryStatus,
    StageOutcome,
    StageResult,
    WebsterPaths,
)


@pytest.fixture
def placeholder_bytes() -> bytes:
    fixture = Path(__file__).parent / "fixtures" / "no_image_available.png.b64"
    return base64.b64decode(fixture.read_bytes())


@pytest.mark.parametrize("url", (None, "https://shop.example/products/123.jpg"))
@pytest.mark.parametrize("image_format", ("PNG", "JPEG"))
@pytest.mark.parametrize("size", ((96, 60), (384, 240), (1200, 750)))
def test_no_image_available_is_rejected_after_resizing_and_reencoding(
    placeholder_bytes: bytes,
    url: str | None,
    image_format: str,
    size: tuple[int, int],
) -> None:
    with Image.open(io.BytesIO(placeholder_bytes)) as source:
        image = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format=image_format, quality=85)

    with pytest.raises(ImageValidationError, match="placeholder"):
        images_module.validate_image_bytes(
            output.getvalue(), original_url=url, final_url=url
        )


@pytest.mark.parametrize("kind", ("white", "solid", "graphic"))
def test_legitimate_simple_images_are_not_rejected(kind: str) -> None:
    image = Image.new("RGB", (384, 240), "white" if kind == "white" else "gray")
    if kind == "graphic":
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((32, 40, 176, 160), fill="navy")
        drawing.ellipse((160, 64, 332, 220), fill="gold", outline="black", width=3)
        drawing.text((40, 200), "Product photo", fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")

    metadata = images_module.validate_image_bytes(output.getvalue())

    assert (metadata.width, metadata.height) == image.size


class _PlaceholderFetcher:
    direct_attempts = 1

    def __init__(
        self, state: RecoveryState, payload: bytes, *, from_cache: bool = False
    ) -> None:
        self.state = state
        self.payload = payload
        self.from_cache = from_cache

    def get(
        self, record_id: str, strategy: str, url: str, **_kwargs: object
    ) -> HTTPResult:
        attempt_id = self.state.log_attempt(
            record_id, strategy, attempted_url=url, http_status=200
        )
        return HTTPResult(
            data=self.payload,
            status_code=200,
            content_type="image/png",
            final_url=url,
            attempt_id=attempt_id,
            request_key=url,
            from_cache=self.from_cache,
        )


@pytest.mark.parametrize("from_cache", (False, True))
def test_direct_placeholder_is_audited_without_publishing_an_asset(
    tmp_path: Path, placeholder_bytes: bytes, from_cache: bool
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    with RecoveryState(paths) as state:
        state.initialize_records(_manifest())
        record = state.get_record("sdv1-0000")
        assert record is not None

        result = recovery_module.recover_direct(
            paths,
            state,
            _PlaceholderFetcher(state, placeholder_bytes, from_cache=from_cache),
            record,
        )

        assert result.outcome is StageOutcome.MISS
        attempts = state.attempts("sdv1-0000")
        assert attempts
        assert all(attempt["validation_result"] == "invalid_image" for attempt in attempts)
        assert all("placeholder" in str(attempt["exception"]) for attempt in attempts)
        assert state.get_asset(hashlib.sha256(placeholder_bytes).hexdigest()) is None
        assert not tuple(paths.raw.glob("*"))
        assert not tuple(paths.normalized.glob("*"))
        assert state.get_record("sdv1-0000")["recovery_status"] == "pending"


@pytest.mark.parametrize("replacement_available", (False, True))
def test_resume_rejects_previously_accepted_placeholder_and_tries_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    placeholder_bytes: bytes,
    replacement_available: bool,
) -> None:
    paths = WebsterPaths.from_root(tmp_path)
    manifest = _manifest()
    fallback_calls: list[str] = []

    def mirror(
        supplied_paths: WebsterPaths,
        state: RecoveryState,
        _fetcher: object,
        record: dict[str, object],
    ) -> StageResult:
        fallback_calls.append(str(record["record_id"]))
        if replacement_available:
            _recover_synthetic_image(
                supplied_paths,
                state,
                record,
                status=RecoveryStatus.RECOVERED_GROUND_TRUTH_MIRROR,
            )
            return StageResult(StageOutcome.RECOVERED, "replacement image")
        return StageResult(StageOutcome.MISS, "no replacement image")

    def miss(*_args: object, **_kwargs: object) -> StageResult:
        return StageResult(StageOutcome.MISS, "offline miss")

    with RecoveryState(paths) as state:
        state.initialize_records(manifest)
        for stage in (RecoveryStage.EXACT_REUSE, RecoveryStage.OFFICIAL_ASSETS):
            state.begin_stage("sdv1-0000", stage)
            state.finish_stage("sdv1-0000", stage, StageResult(StageOutcome.MISS))
        record = state.get_record("sdv1-0000")
        assert record is not None
        state.begin_stage("sdv1-0000", RecoveryStage.DIRECT)
        with monkeypatch.context() as previous_validation:
            previous_validation.setattr(
                images_module, "_known_placeholder_reason", lambda *_args: None
            )
            metadata, raw_path, normalized_path = images_module.store_image(
                paths,
                state,
                placeholder_bytes,
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
                status=RecoveryStatus.RECOVERED_EXACT_URL,
                method="direct_original",
                resolved_url=str(record["original_url"]),
            )
        state.finish_stage(
            "sdv1-0000", RecoveryStage.DIRECT, StageResult(StageOutcome.RECOVERED)
        )
        recovered = state.get_record("sdv1-0000")
        assert recovered is not None
        assert not images_module.recovery_artifacts_available(paths, recovered)
        assert not images_module.recovery_artifacts_valid(paths, recovered)

        recovery_module.RecoveryEngine(
            paths,
            state,
            manifest,
            _PlaceholderFetcher(state, placeholder_bytes, from_cache=True),
            mirror_stage=mirror,
            wayback_stage=miss,
            arquivo_stage=miss,
            commoncrawl_stage=miss,
            direct_workers=1,
            show_progress=False,
        ).run()

        assert fallback_calls == ["sdv1-0000"]
        current = state.get_record("sdv1-0000")
        assert current is not None
        assert raw_path.read_bytes() == placeholder_bytes
        validations = {attempt["validation_result"] for attempt in state.attempts("sdv1-0000")}
        assert {"recovered_artifacts_invalid", "invalid_image"} <= validations
        if replacement_available:
            assert current["recovery_status"] == "recovered_ground_truth_mirror"
            assert current["sha256"] != metadata.sha256
            assert images_module.recovery_artifacts_valid(paths, current)
        else:
            assert current["recovery_status"] == "unresolved"
            assert current["local_normalized_path"] is None
            assert current["sha256"] is None
        prepared = webster_module._prepare_model_rows(
            paths,
            webster_module.WebsterModel.SDV1,
            recovery_module.current_manifest(manifest, state),
        )
        assert bool(prepared.iloc[0]["image_available"]) is replacement_available
