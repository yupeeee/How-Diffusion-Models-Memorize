"""Image validation, normalization, hashes, and content-addressed storage."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import filetype
import imagehash
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from utils.common.io import atomic_write_json

from .webster import (
    ArchiveCandidate,
    HTTPResult,
    ImageMetadata,
    ImageValidationError,
    RecoveryStatus,
    WebsterPaths,
    path_is_within,
)

if TYPE_CHECKING:
    from .state import RecoveryState


FORMAT_EXTENSIONS = {
    "JPEG": "jpg",
    "PNG": "png",
    "GIF": "gif",
    "WEBP": "webp",
    "TIFF": "tif",
    "BMP": "bmp",
    "ICO": "ico",
    "AVIF": "avif",
    "HEIF": "heif",
    "JPEG2000": "jp2",
    "PPM": "ppm",
}
_TRACKING_PIXEL = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff"
    b"!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01"
    b"\x00\x00\x02\x02D\x01\x00;"
)
KNOWN_PLACEHOLDER_SHA256 = {hashlib.sha256(_TRACKING_PIXEL).hexdigest()}
# Camera icon above "No Image Available", returned at a Wayfair product URL.
# Fingerprints of the decoded sample (raw SHA256 below) survive resizing and
# re-encoding; requiring both hashes avoids discarding unrelated simple images.
# e1d6caae9082aa1deb5d18b4c9ec58d7f70540c881a216d566445fc30dedde5a
KNOWN_PLACEHOLDER_VISUAL_HASHES = (
    ("No Image Available (camera icon)", "ec343131d3c6e1ce", "9e3b2723273bd599"),
)
PLACEHOLDER_MAX_PHASH_DISTANCE = 4
PLACEHOLDER_MAX_DHASH_DISTANCE = 8


def has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or (
        image.mode == "P" and "transparency" in image.info
    )


def visual_rgb_frame(image: Image.Image) -> Image.Image:
    """Return an EXIF-corrected RGB frame on a white alpha background."""

    visual = ImageOps.exif_transpose(image.copy())
    if not has_alpha(visual):
        return visual.convert("RGB")
    rgba = visual.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _known_placeholder_reason(
    visual: Image.Image, perceptual_hash: imagehash.ImageHash
) -> str | None:
    """Match known replacement graphics independently of their response URLs."""

    for name, phash, dhash in KNOWN_PLACEHOLDER_VISUAL_HASHES:
        if perceptual_hash - imagehash.hex_to_hash(phash) > PLACEHOLDER_MAX_PHASH_DISTANCE:
            continue
        if imagehash.dhash(visual) - imagehash.hex_to_hash(dhash) <= PLACEHOLDER_MAX_DHASH_DISTANCE:
            return f"known placeholder image: {name}"
    return None


def _url_path(value: str | None) -> str:
    return urlsplit(value or "").path.casefold()


def _substitute_reason(
    image: Image.Image,
    sha256: str,
    original_url: str | None,
    final_url: str | None,
) -> str | None:
    if image.width * image.height <= 1:
        return "one-pixel tracking image"
    if min(image.size) <= 2 and max(image.size) <= 64:
        return "tracking-sized image"
    if sha256 in KNOWN_PLACEHOLDER_SHA256:
        return "known tracking or placeholder image"
    visible = False
    for frame_number in range(int(getattr(image, "n_frames", 1))):
        image.seek(frame_number)
        image.load()
        if (
            not has_alpha(image)
            or image.convert("RGBA").getchannel("A").getbbox() is not None
        ):
            visible = True
            break
    image.seek(0)
    if not visible:
        return "fully transparent placeholder"
    original_path = _url_path(original_url)
    final_path = _url_path(final_url)
    suspicious = (
        "placeholder", "missing-image", "no_image", "noimage", "spacer", "pixel"
    )
    if any(token in final_path for token in suspicious) and max(image.size) <= 512:
        return "generic placeholder image"
    if (
        original_url
        and final_url
        and final_url != original_url
        and "logo" in final_path
        and "logo" not in original_path
        and max(image.size) <= 512
    ):
        return "redirected to a site logo"
    return None


def validate_image_bytes(
    image_bytes: bytes,
    *,
    original_url: str | None = None,
    final_url: str | None = None,
) -> ImageMetadata:
    """Fully decode image bytes and reject documents, corruption, and placeholders."""

    if not image_bytes:
        raise ImageValidationError("empty response")
    prefix = image_bytes[:1024].lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<html" in prefix:
        raise ImageValidationError("HTML or XML response")
    kind = filetype.guess(image_bytes)
    if kind is None or not str(kind.mime).casefold().startswith("image/"):
        raise ImageValidationError("unrecognized image signature")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            detected_format = (probe.format or "").upper()
            probe.verify()
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            detected_format = (decoded.format or detected_format).upper()
            width, height = decoded.size
            mode = decoded.mode
            for frame_number in range(int(getattr(decoded, "n_frames", 1))):
                decoded.seek(frame_number)
                decoded.load()
            decoded.seek(0)
            sha256 = hashlib.sha256(image_bytes).hexdigest()
            reason = _substitute_reason(
                decoded, sha256, original_url, final_url
            )
            if reason is not None:
                raise ImageValidationError(reason)
            visual = visual_rgb_frame(decoded)
            visual_hash = imagehash.phash(visual)
            reason = _known_placeholder_reason(visual, visual_hash)
            if reason is not None:
                raise ImageValidationError(reason)
            perceptual_hash = str(visual_hash)
    except ImageValidationError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise ImageValidationError(
            f"Pillow rejected image: {type(error).__name__}: {error}"
        ) from error
    extension = FORMAT_EXTENSIONS.get(detected_format)
    if width <= 0 or height <= 0:
        raise ImageValidationError("invalid image dimensions")
    if extension is None:
        raise ImageValidationError(
            f"unsupported decoded image format {detected_format!r}"
        )
    return ImageMetadata(
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        sha1=hashlib.sha1(image_bytes).hexdigest(),
        perceptual_hash=perceptual_hash,
        width=int(width),
        height=int(height),
        mode=str(mode),
        image_format=detected_format,
        extension=extension,
        byte_size=len(image_bytes),
    )


def normalized_png_bytes(image_bytes: bytes) -> bytes:
    """Return deterministic lossless PNG bytes, preserving animation frames."""

    with Image.open(io.BytesIO(image_bytes)) as source:
        frame_count = int(getattr(source, "n_frames", 1))
        loop = int(source.info.get("loop", 0) or 0)
        default_image = bool(
            source.format == "PNG" and source.info.get("default_image", False)
        )
        frames: list[Image.Image] = []
        durations: list[int] = []
        for frame_number in range(frame_count):
            source.seek(frame_number)
            source.load()
            frames.append(ImageOps.exif_transpose(source.copy()))
            durations.append(int(source.info.get("duration", 0) or 0))
        mode = "RGBA" if any(has_alpha(frame) for frame in frames) else "RGB"
        converted = [frame.convert(mode) for frame in frames]
        output = io.BytesIO()
        arguments: dict[str, object] = {
            "format": "PNG",
            "compress_level": 9,
            "optimize": False,
        }
        if len(converted) > 1:
            animation_start = 1 if default_image else 0
            arguments.update(
                {
                    "save_all": True,
                    "append_images": converted[1:],
                    "duration": durations[animation_start:],
                    "loop": loop,
                    "disposal": [0] * (len(converted) - animation_start),
                    "blend": [0] * (len(converted) - animation_start),
                }
            )
            if default_image:
                arguments["default_image"] = True
        converted[0].save(output, **arguments)
        return output.getvalue()


def verify_existing_blob(path: Path, expected_sha256: str) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == expected_sha256


def _publish_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != data:
            raise RuntimeError(f"immutable path collision: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def publish_verified_blob(
    path: Path, data: bytes, quarantine_directory: Path
) -> None:
    """Publish immutable bytes, quarantining any conflicting owned path."""

    if path.is_symlink() or path.exists() and not path.is_file():
        quarantine_directory.mkdir(parents=True, exist_ok=True)
        os.replace(
            path,
            quarantine_directory / f"{path.name}.unsafe.{uuid.uuid4().hex}",
        )
    elif path.exists():
        if path.read_bytes() == data:
            return
        quarantine_directory.mkdir(parents=True, exist_ok=True)
        os.replace(
            path,
            quarantine_directory / f"{path.name}.corrupt.{uuid.uuid4().hex}",
        )
    _publish_immutable(path, data)


def store_image(
    paths: WebsterPaths,
    state: RecoveryState,
    image_bytes: bytes,
    *,
    original_url: str | None,
    final_url: str | None,
) -> tuple[ImageMetadata, Path, Path]:
    metadata = validate_image_bytes(
        image_bytes, original_url=original_url, final_url=final_url
    )
    raw_path = paths.raw / f"{metadata.sha256}.{metadata.extension}"
    normalized_path = paths.normalized / f"{metadata.sha256}.png"
    quarantine = paths.state / "quarantine"
    publish_verified_blob(raw_path, image_bytes, quarantine)
    publish_verified_blob(
        normalized_path, normalized_png_bytes(image_bytes), quarantine
    )
    state.add_asset(metadata, raw_path, normalized_path)
    return metadata, raw_path, normalized_path


def pointer_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + ".pointer.json")


def reference_matches(source: Path, destination: Path) -> bool:
    if destination.is_file():
        try:
            return os.path.samefile(source, destination) or (
                source.stat().st_size == destination.stat().st_size
                and hashlib.sha256(source.read_bytes()).digest()
                == hashlib.sha256(destination.read_bytes()).digest()
            )
        except OSError:
            return False
    pointer = pointer_path(destination)
    if not pointer.is_file():
        return False
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        target = Path(str(payload["target"]))
        if not target.is_absolute():
            target = pointer.parent / target
        return target.resolve(strict=True) == source.resolve(strict=True)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def remove_stale_reference(destination: Path) -> None:
    for path in (destination, pointer_path(destination)):
        if path.is_symlink() or path.exists():
            if path.is_dir() and not path.is_symlink():
                raise RuntimeError(f"reference path is a directory: {path}")
            path.unlink()


def create_reference(source: Path, destination: Path) -> Path:
    """Create a hard link, relative symbolic link, or explicit pointer."""

    if not source.is_file():
        raise RuntimeError(f"reference source does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if reference_matches(source, destination):
        return (
            destination if destination.is_file() else pointer_path(destination)
        )
    remove_stale_reference(destination)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
        return destination
    except OSError:
        temporary.unlink(missing_ok=True)
    try:
        temporary.symlink_to(os.path.relpath(source, destination.parent))
        os.replace(temporary, destination)
        return destination
    except OSError:
        temporary.unlink(missing_ok=True)
    pointer = pointer_path(destination)
    atomic_write_json(pointer, {"target": str(source)})
    return pointer


def recovery_artifacts_available(
    paths: WebsterPaths, record: Mapping[str, object]
) -> bool:
    """Check paths, image headers, and cached placeholder content before reuse."""

    try:
        raw_path = Path(str(record["local_raw_path"]))
        normalized_path = Path(str(record["local_normalized_path"]))
        if not path_is_within(raw_path, paths.raw) or not path_is_within(
            normalized_path, paths.normalized
        ):
            return False
        if (
            not raw_path.is_file()
            or not normalized_path.is_file()
            or raw_path.stat().st_size <= 0
            or normalized_path.stat().st_size <= 8
        ):
            return False
        with Image.open(raw_path) as raw:
            raw_size = raw.size
            raw_format = str(raw.format or "")
        with Image.open(normalized_path) as normalized:
            normalized_size = normalized.size
            normalized_format = str(normalized.format or "")
            visual = visual_rgb_frame(normalized)
            if _known_placeholder_reason(visual, imagehash.phash(visual)) is not None:
                return False
        expected_width = int(record["width"])
        expected_height = int(record["height"])
        expected_format = str(record["image_format"] or "")
        return (
            raw_size == (expected_width, expected_height)
            and normalized_size == raw_size
            and raw_format.casefold() == expected_format.casefold()
            and normalized_format.casefold() == "png"
            and re.fullmatch(
                r"[0-9a-f]{64}", str(record["sha256"] or "").casefold()
            )
            is not None
        )
    except (
        KeyError,
        OSError,
        TypeError,
        UnidentifiedImageError,
        ValueError,
    ):
        return False


def recovery_artifacts_valid(
    paths: WebsterPaths, record: Mapping[str, object]
) -> bool:
    try:
        raw_path = Path(str(record["local_raw_path"]))
        normalized_path = Path(str(record["local_normalized_path"]))
        if not path_is_within(raw_path, paths.raw) or not path_is_within(
            normalized_path, paths.normalized
        ):
            return False
        if not verify_existing_blob(raw_path, str(record["sha256"])):
            return False
        normalized = normalized_path.read_bytes()
        if normalized != normalized_png_bytes(raw_path.read_bytes()):
            return False
        validate_image_bytes(normalized)
        return True
    except (
        ImageValidationError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


def candidate_reference_destination(candidate_path: object) -> Path:
    text = str(candidate_path)
    suffix = ".pointer.json"
    return Path(text[:-len(suffix)] if text.endswith(suffix) else text)


def candidate_artifacts_valid(candidate: Mapping[str, object]) -> bool:
    try:
        raw_path = Path(str(candidate["raw_path"]))
        normalized_path = Path(str(candidate["normalized_path"]))
        if not verify_existing_blob(raw_path, str(candidate["sha256"])):
            return False
        normalized = normalized_path.read_bytes()
        if normalized != normalized_png_bytes(raw_path.read_bytes()):
            return False
        validate_image_bytes(normalized)
        return reference_matches(
            raw_path,
            candidate_reference_destination(candidate["candidate_path"]),
        )
    except (
        ImageValidationError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


def apply_image_resolution(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    metadata: ImageMetadata,
    raw_path: Path,
    normalized_path: Path,
    *,
    status: RecoveryStatus,
    method: str,
    resolved_url: str,
    archive_timestamp: str | None = None,
    archive_digest: str | None = None,
) -> None:
    """Commit one validated canonical paired-source image to its record."""

    del paths
    state.mark_recovered(
        str(record["record_id"]),
        status,
        recovery_method=method,
        resolved_url=resolved_url,
        archive_timestamp=archive_timestamp,
        archive_digest=archive_digest,
        local_raw_path=str(raw_path),
        local_normalized_path=str(normalized_path),
        sha256=metadata.sha256,
        sha1=metadata.sha1,
        perceptual_hash=metadata.perceptual_hash,
        width=metadata.width,
        height=metadata.height,
        image_format=metadata.image_format,
        ambiguity_reason=None,
    )


def _normalized_archive_digest(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    if text.startswith("sha1:"):
        key = text.removeprefix("sha1:")
        return (
            f"sha1:{key}"
            if re.fullmatch(r"[a-z2-7]{32}", key)
            else None
        )
    if text.startswith("md5:"):
        key = text.removeprefix("md5:")
        return (
            f"md5:{key}"
            if re.fullmatch(r"[0-9a-f]{32}", key)
            else None
        )
    if re.fullmatch(r"[0-9a-f]{32}", text):
        return f"md5:{text}"
    if re.fullmatch(r"[a-z2-7]{32}", text):
        return f"sha1:{text}"
    return None


def preserve_archive_candidate(
    paths: WebsterPaths,
    state: RecoveryState,
    record: Mapping[str, object],
    result: HTTPResult,
    *,
    strategy: str,
    resolved_url: str,
    archive_timestamp_value: str | None,
    archive_digest: str | None,
    verified_archive_digest: str | None,
    source_page: bool,
    provenance: Mapping[str, object],
) -> ArchiveCandidate:
    """Validate and preserve one archive response with complete provenance."""

    reported_key = _normalized_archive_digest(archive_digest)
    verified_key = _normalized_archive_digest(verified_archive_digest)
    if verified_archive_digest is not None and verified_key is None:
        state.update_attempt(
            result.attempt_id,
            validation_result="archive_digest_unusable",
            exception=(
                f"unusable verified digest {verified_archive_digest!r}"
            ),
        )
        raise ImageValidationError("verified archive digest is malformed")
    if verified_key is not None and reported_key != verified_key:
        state.update_attempt(
            result.attempt_id,
            validation_result="archive_digest_mismatch",
            exception="reported and verified archive digests differ",
        )
        raise ImageValidationError(
            "reported archive digest failed verification"
        )
    metadata, raw_path, normalized_path = store_image(
        paths,
        state,
        result.data,
        original_url=str(record.get("original_url") or resolved_url),
        final_url=result.final_url,
    )
    destination = (
        paths.candidates
        / str(record["record_id"])
        / f"{metadata.sha256}.{metadata.extension}"
    )
    preserved = create_reference(raw_path, destination)
    candidate = ArchiveCandidate(
        record_id=str(record["record_id"]),
        sha256=metadata.sha256,
        perceptual_hash=metadata.perceptual_hash,
        raw_path=str(raw_path),
        normalized_path=str(normalized_path),
        candidate_path=str(preserved),
        strategy=strategy,
        resolved_url=resolved_url,
        archive_timestamp=archive_timestamp_value,
        archive_digest=verified_key,
        source_page=source_page,
        response_mime=result.content_type,
    )
    state.add_candidate(
        candidate,
        {
            **dict(provenance),
            "reported_archive_digest": archive_digest,
            "verified_archive_digest": verified_key,
            "archive_digest_verified": verified_key is not None,
        },
    )
    state.update_attempt(
        result.attempt_id,
        validation_result="valid_image",
        candidate_sha256=metadata.sha256,
    )
    return candidate


__all__ = [
    "apply_image_resolution",
    "candidate_artifacts_valid",
    "candidate_reference_destination",
    "create_reference",
    "has_alpha",
    "normalized_png_bytes",
    "pointer_path",
    "preserve_archive_candidate",
    "publish_verified_blob",
    "recovery_artifacts_available",
    "recovery_artifacts_valid",
    "reference_matches",
    "store_image",
    "validate_image_bytes",
    "verify_existing_blob",
    "visual_rgb_frame",
]
