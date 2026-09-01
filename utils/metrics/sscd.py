"""Pinned SSCD checkpoint handling, preprocessing, and cosine scoring.

The module has no import-time network or model side effects.  Checkpoint
acquisition is explicit, fixed to the official standalone ``sscd_disc_large``
artifact, and records the observed hash instead of inventing an expected one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from tqdm import tqdm

from utils.common.io import (
    atomic_write_json,
    canonical_hash,
    file_sha256,
)

SSCD_MODEL_NAME = "sscd_disc_large"
SSCD_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/sscd-copy-detection/"
    "sscd_disc_large.torchscript.pt"
)
SSCD_FEATURE_DIMENSION = 1024
SSCD_INPUT_SIZE = 288
SSCD_FEATURE_BATCH_SIZE = 8
SSCD_SCORE_RANGE_TOLERANCE = 1e-5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CHECKPOINT_RELATIVE_PATH = Path("checkpoints/sscd/sscd_disc_large.torchscript.pt")
_MANIFEST_RELATIVE_PATH = Path("checkpoints/sscd/manifest.json")
_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SSCD_CHECKPOINT_URL",
    "SSCD_FEATURE_BATCH_SIZE",
    "SSCD_FEATURE_DIMENSION",
    "SSCD_INPUT_SIZE",
    "SSCD_MODEL_NAME",
    "SSCDCheckpoint",
    "SSCDMetricError",
    "compute_sscd_scores",
    "cosine_similarity_scores",
    "ensure_sscd_checkpoint",
    "extract_sscd_features",
    "load_sscd_model",
    "preprocess_sscd_images",
    "sscd_preprocessing_hash",
    "sscd_preprocessing_policy",
    "validate_sscd_scores",
]


class SSCDMetricError(RuntimeError):
    """Report an invalid SSCD checkpoint, descriptor, or score."""


@dataclass(frozen=True, slots=True)
class SSCDCheckpoint:
    """The fixed SSCD checkpoint identity observed on local storage."""

    model_name: str
    url: str
    path: Path
    sha256: str


def sscd_preprocessing_policy() -> dict[str, object]:
    """Return the complete JSON-ready SSCD image preprocessing policy."""

    return {
        "exif_transpose": True,
        "color_mode": "RGB",
        "resize": {
            "width": SSCD_INPUT_SIZE,
            "height": SSCD_INPUT_SIZE,
            "resampling": "Pillow bilinear",
            "aspect_policy": "direct resize without crop",
        },
        "pixel_range": [0.0, 1.0],
        "normalization": {
            "name": "ImageNet",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "random_crop": False,
        "augmentation": False,
        "descriptor_centering": False,
        "descriptor_whitening": False,
        "score_normalization": False,
        "output_shape": ["B", 3, SSCD_INPUT_SIZE, SSCD_INPUT_SIZE],
        "output_dtype": "float32",
    }


def sscd_preprocessing_hash() -> str:
    """Hash the fixed preprocessing policy deterministically."""

    return canonical_hash(sscd_preprocessing_policy())


def preprocess_sscd_images(images: Sequence[Image.Image]) -> torch.Tensor:
    """Apply fixed SSCD preprocessing and return ``[B,3,288,288]`` float32."""

    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise SSCDMetricError("SSCD images must be a sequence of Pillow images")
    if not images:
        raise SSCDMetricError("SSCD preprocessing requires at least one image")

    processed: list[torch.Tensor] = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise SSCDMetricError("Every SSCD input must be a Pillow image")
        transposed = ImageOps.exif_transpose(image)
        rgb = transposed.convert("RGB")
        resized = rgb.resize(
            (SSCD_INPUT_SIZE, SSCD_INPUT_SIZE),
            resample=Image.Resampling.BILINEAR,
        )
        pixels = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
        pixels = (
            pixels.reshape(SSCD_INPUT_SIZE, SSCD_INPUT_SIZE, 3)
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .div(255.0)
        )
        processed.append(pixels)

    batch = torch.stack(processed, dim=0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    batch = batch.sub(mean).div(std).contiguous()
    if batch.shape != (len(images), 3, SSCD_INPUT_SIZE, SSCD_INPUT_SIZE):
        raise SSCDMetricError("SSCD preprocessing produced an invalid tensor shape")
    if batch.dtype is not torch.float32 or batch.device.type != "cpu":
        raise SSCDMetricError("SSCD preprocessing must produce CPU float32 tensors")
    if not bool(torch.isfinite(batch).all().item()):
        raise SSCDMetricError("SSCD preprocessing produced NaN or infinity")
    return batch


def ensure_sscd_checkpoint(
    project_root: str | Path,
    *,
    client: httpx.Client | None = None,
) -> SSCDCheckpoint:
    """Reuse or atomically acquire the fixed official SSCD checkpoint.

    A persisted manifest is a trust-on-first-observation integrity lock.  A
    later byte mismatch is rejected and never silently replaced.
    """

    root = Path(project_root).expanduser().resolve()
    checkpoint_path = root / _CHECKPOINT_RELATIVE_PATH
    manifest_path = root / _MANIFEST_RELATIVE_PATH
    manifest = _read_checkpoint_manifest(manifest_path)

    if checkpoint_path.exists():
        if not checkpoint_path.is_file():
            raise SSCDMetricError(
                f"SSCD checkpoint path is not a regular file: {checkpoint_path}"
            )
        observed_hash = _checkpoint_hash(checkpoint_path)
        if manifest is not None:
            expected_hash = str(manifest["checkpoint_sha256"])
            if observed_hash != expected_hash:
                raise SSCDMetricError(
                    "SSCD checkpoint SHA-256 differs from the persisted manifest"
                )
        else:
            _validate_torchscript(checkpoint_path)
            _write_checkpoint_manifest(manifest_path, observed_hash)
        return SSCDCheckpoint(
            model_name=SSCD_MODEL_NAME,
            url=SSCD_CHECKPOINT_URL,
            path=checkpoint_path,
            sha256=observed_hash,
        )

    expected_hash = str(manifest["checkpoint_sha256"]) if manifest is not None else None
    observed_hash = _download_checkpoint(
        checkpoint_path,
        client=client,
        expected_hash=expected_hash,
    )
    if manifest is None:
        _write_checkpoint_manifest(manifest_path, observed_hash)
    return SSCDCheckpoint(
        model_name=SSCD_MODEL_NAME,
        url=SSCD_CHECKPOINT_URL,
        path=checkpoint_path,
        sha256=observed_hash,
    )


def load_sscd_model(
    checkpoint: SSCDCheckpoint,
    device: str | torch.device,
) -> Any:
    """Hash-check and load the pinned TorchScript descriptor in evaluation mode."""

    _validate_checkpoint_identity(checkpoint)
    observed_hash = _checkpoint_hash(checkpoint.path)
    if observed_hash != checkpoint.sha256:
        raise SSCDMetricError(
            "SSCD checkpoint SHA-256 changed after its identity was established"
        )
    selected_device = torch.device(device)
    try:
        model = torch.jit.load(checkpoint.path, map_location=selected_device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    except Exception as error:
        raise SSCDMetricError(
            f"Cannot load SSCD TorchScript checkpoint {checkpoint.path}: "
            f"{type(error).__name__}: {error}"
        ) from error
    return model


def extract_sscd_features(
    images: Sequence[Image.Image],
    model: Any,
    device: str | torch.device,
    *,
    initial_batch_size: int = SSCD_FEATURE_BATCH_SIZE,
) -> torch.Tensor:
    """Extract normalized descriptors in order with adaptive CUDA-OOM batches."""

    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise SSCDMetricError("SSCD images must be a sequence of Pillow images")
    if not images:
        raise SSCDMetricError("SSCD feature extraction requires at least one image")
    batch_size = _positive_integer(initial_batch_size, "initial_batch_size")
    selected_device = torch.device(device)
    outputs: list[torch.Tensor] = []
    position = 0
    batch_size = min(batch_size, len(images))

    with torch.inference_mode():
        while position < len(images):
            current_size = min(batch_size, len(images) - position)
            cpu_inputs = preprocess_sscd_images(
                images[position : position + current_size]
            )
            inputs: torch.Tensor | None = None
            raw_features: object | None = None
            normalized: torch.Tensor | None = None
            retry_after_oom = False
            try:
                inputs = cpu_inputs.to(device=selected_device, dtype=torch.float32)
                raw_features = model(inputs)
                normalized = _normalized_features(
                    raw_features,
                    expected_batch=current_size,
                    label="SSCD model output",
                )
            except RuntimeError as error:
                if (
                    not _is_cuda_out_of_memory(error, selected_device)
                    or current_size == 1
                ):
                    raise
                batch_size = max(1, current_size // 2)
                retry_after_oom = True
            if retry_after_oom:
                del cpu_inputs, inputs, raw_features, normalized
                torch.cuda.empty_cache()
                continue
            assert isinstance(normalized, torch.Tensor)
            outputs.append(normalized)
            position += current_size
            del cpu_inputs, inputs, raw_features, normalized

    features = torch.cat(outputs, dim=0).to(dtype=torch.float32).contiguous()
    if features.shape != (len(images), SSCD_FEATURE_DIMENSION):
        raise SSCDMetricError("Combined SSCD feature tensor has an invalid shape")
    return features


def cosine_similarity_scores(
    target_feature: torch.Tensor,
    generated_features: torch.Tensor,
) -> torch.Tensor:
    """Return one target-versus-generated cosine score in generated order."""

    target = _normalized_features(
        target_feature,
        expected_batch=1,
        label="target SSCD feature",
    )
    if not isinstance(generated_features, torch.Tensor) or generated_features.ndim != 2:
        raise SSCDMetricError("generated SSCD features must have shape [N,1024]")
    generated_count = generated_features.shape[0]
    if generated_count < 1:
        raise SSCDMetricError("at least one generated SSCD feature is required")
    generated = _normalized_features(
        generated_features,
        expected_batch=generated_count,
        label="generated SSCD features",
    )
    scores = (generated @ target.T)[:, 0]
    scores = scores.detach().cpu().to(dtype=torch.float32).contiguous()
    return validate_sscd_scores(scores, generated_count)


def compute_sscd_scores(
    target_image: Image.Image,
    generated_images: Sequence[Image.Image],
    model: Any,
    device: str | torch.device,
    *,
    initial_batch_size: int = SSCD_FEATURE_BATCH_SIZE,
) -> torch.Tensor:
    """Compare one paired source image with generated images in seed order."""

    if not isinstance(target_image, Image.Image):
        raise SSCDMetricError("SSCD target must be a Pillow image")
    target_feature = extract_sscd_features(
        [target_image],
        model,
        device,
        initial_batch_size=1,
    )
    generated_features = extract_sscd_features(
        generated_images,
        model,
        device,
        initial_batch_size=initial_batch_size,
    )
    return cosine_similarity_scores(target_feature, generated_features)


def validate_sscd_scores(
    scores: object,
    expected_count: int,
) -> torch.Tensor:
    """Validate the exact plain-tensor SSCD score persistence contract."""

    count = _positive_integer(expected_count, "expected_count")
    if not isinstance(scores, torch.Tensor):
        raise SSCDMetricError("SSCD scores must be a torch tensor")
    if scores.shape != (count,):
        raise SSCDMetricError(f"SSCD scores must have shape [{count}]")
    if scores.dtype is not torch.float32:
        raise SSCDMetricError("SSCD scores must have dtype float32")
    if scores.device.type != "cpu":
        raise SSCDMetricError("SSCD scores must be stored on CPU")
    if not scores.is_contiguous():
        raise SSCDMetricError("SSCD scores must be contiguous")
    if scores.requires_grad:
        raise SSCDMetricError("SSCD scores must not require gradients")
    if not bool(torch.isfinite(scores).all().item()):
        raise SSCDMetricError("SSCD scores contain NaN or infinity")
    lower = -1.0 - SSCD_SCORE_RANGE_TOLERANCE
    upper = 1.0 + SSCD_SCORE_RANGE_TOLERANCE
    if bool(((scores < lower) | (scores > upper)).any().item()):
        raise SSCDMetricError("SSCD cosine scores are outside [-1, 1]")
    return scores


def _normalized_features(
    value: object,
    *,
    expected_batch: int,
    label: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SSCDMetricError(f"{label} is not a tensor")
    if value.shape != (expected_batch, SSCD_FEATURE_DIMENSION):
        raise SSCDMetricError(
            f"{label} must have shape [{expected_batch},{SSCD_FEATURE_DIMENSION}]"
        )
    if not value.is_floating_point():
        raise SSCDMetricError(f"{label} must be floating point")
    if not bool(torch.isfinite(value).all().item()):
        raise SSCDMetricError(f"{label} contains NaN or infinity")
    features = value.float()
    norms = torch.linalg.vector_norm(features, ord=2, dim=1)
    if bool((norms <= 0).any().item()):
        raise SSCDMetricError(f"{label} contains a zero-length descriptor")
    features = functional.normalize(features, p=2, dim=1)
    if not bool(torch.isfinite(features).all().item()):
        raise SSCDMetricError(f"{label} normalization produced NaN or infinity")
    return features.detach().cpu().to(dtype=torch.float32).contiguous()


def _read_checkpoint_manifest(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise SSCDMetricError(f"SSCD manifest path is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SSCDMetricError(
            f"Cannot read SSCD checkpoint manifest {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise SSCDMetricError("SSCD checkpoint manifest is not a JSON object")
    expected = {
        "schema_version": _CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "model_name": SSCD_MODEL_NAME,
        "checkpoint_url": SSCD_CHECKPOINT_URL,
        "checkpoint_path": _CHECKPOINT_RELATIVE_PATH.as_posix(),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise SSCDMetricError(f"SSCD checkpoint manifest {key} differs")
    digest = value.get("checkpoint_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise SSCDMetricError("SSCD checkpoint manifest SHA-256 is invalid")
    return value


def _write_checkpoint_manifest(path: Path, digest: str) -> None:
    if _SHA256.fullmatch(digest) is None:
        raise SSCDMetricError("Refusing to persist an invalid checkpoint SHA-256")
    atomic_write_json(
        path,
        {
            "schema_version": _CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "model_name": SSCD_MODEL_NAME,
            "checkpoint_url": SSCD_CHECKPOINT_URL,
            "checkpoint_path": _CHECKPOINT_RELATIVE_PATH.as_posix(),
            "checkpoint_sha256": digest,
        },
    )


def _download_checkpoint(
    destination: Path,
    *,
    client: httpx.Client | None,
    expected_hash: str | None,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    owns_client = client is None
    active_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        try:
            with active_client.stream(
                "GET", SSCD_CHECKPOINT_URL, follow_redirects=True
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                try:
                    total_bytes = (
                        int(content_length) if content_length is not None else None
                    )
                except (TypeError, ValueError):
                    total_bytes = None
                if total_bytes is not None and total_bytes <= 0:
                    total_bytes = None
                with temporary.open("xb") as handle:
                    with tqdm(
                        total=total_bytes,
                        desc="[SSCD] Checkpoint download",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        dynamic_ncols=True,
                        leave=True,
                        disable=False,
                    ) as progress:
                        for chunk in response.iter_bytes(
                            chunk_size=_DOWNLOAD_CHUNK_SIZE
                        ):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            digest.update(chunk)
                            byte_count += len(chunk)
                            progress.update(len(chunk))
                    handle.flush()
                    os.fsync(handle.fileno())
        except (httpx.HTTPError, OSError) as error:
            raise SSCDMetricError(
                f"Cannot download SSCD checkpoint from {SSCD_CHECKPOINT_URL}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if byte_count == 0:
            raise SSCDMetricError("Downloaded SSCD checkpoint is empty")
        observed_hash = digest.hexdigest()
        if expected_hash is not None and observed_hash != expected_hash:
            raise SSCDMetricError(
                "Downloaded SSCD checkpoint SHA-256 differs from the persisted manifest"
            )
        _validate_torchscript(temporary)
        os.replace(temporary, destination)
        return observed_hash
    finally:
        temporary.unlink(missing_ok=True)
        if owns_client:
            active_client.close()


def _validate_torchscript(path: Path) -> None:
    try:
        model = torch.jit.load(path, map_location=torch.device("cpu"))
        model.eval()
        del model
    except Exception as error:
        raise SSCDMetricError(
            f"SSCD checkpoint is not valid TorchScript: "
            f"{type(error).__name__}: {error}"
        ) from error


def _checkpoint_hash(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise SSCDMetricError(
            f"Cannot hash SSCD checkpoint {path}: {type(error).__name__}: {error}"
        ) from error


def _validate_checkpoint_identity(checkpoint: SSCDCheckpoint) -> None:
    if not isinstance(checkpoint, SSCDCheckpoint):
        raise SSCDMetricError("checkpoint must be an SSCDCheckpoint")
    if checkpoint.model_name != SSCD_MODEL_NAME:
        raise SSCDMetricError("SSCD checkpoint model name differs")
    if checkpoint.url != SSCD_CHECKPOINT_URL:
        raise SSCDMetricError("SSCD checkpoint URL differs")
    if _SHA256.fullmatch(checkpoint.sha256) is None:
        raise SSCDMetricError("SSCD checkpoint SHA-256 is invalid")
    if not checkpoint.path.is_file():
        raise SSCDMetricError(f"SSCD checkpoint is missing: {checkpoint.path}")


def _is_cuda_out_of_memory(error: RuntimeError, device: torch.device) -> bool:
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", ())
    if isinstance(error, out_of_memory_type):
        return True
    return device.type == "cuda" and "out of memory" in str(error).lower()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SSCDMetricError(f"{name} must be a positive integer")
    return value
