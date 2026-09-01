"""Atomic persistence, canonical hashes, and safe local cache loading."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import torch


class CacheIOError(RuntimeError):
    """A local cache artifact is unsafe, corrupt, or cannot be persisted."""


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""

    return datetime.now(UTC).isoformat()


def canonical_json(value: object) -> str:
    """Return the single canonical JSON representation used by all hashes."""

    return json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    """Return SHA-256 over :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_value(value: object) -> Any:
    """Convert project values into deterministic JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=repr)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if hasattr(value, "item"):
        return json_value(value.item())
    raise TypeError(f"cannot convert {type(value).__name__} to JSON")


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(destination: str | Path, content: bytes) -> None:
    """Atomically publish a byte string through a synced sibling file."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(destination: str | Path, content: str) -> None:
    """Atomically publish UTF-8 text."""

    atomic_write_bytes(destination, content.encode("utf-8"))


def atomic_write_json(destination: str | Path, value: object) -> None:
    """Atomically write readable deterministic JSON."""

    content = json.dumps(
        json_value(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(destination, content + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level values."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheIOError(f"cannot read JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise CacheIOError(f"JSON root must be an object: {source}")
    return value


def atomic_write_csv(
    destination: str | Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """Write a row sequence as CSV with stable column order."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode nested cells deterministically for a CSV representation."""

    result = frame.copy(deep=True)
    for column in result.columns:
        result[column] = result[column].map(
            lambda value: canonical_json(value)
            if isinstance(value, (list, tuple, dict, set, frozenset))
            or hasattr(value, "tolist")
            else value
        )
    return result


def atomic_write_frame_csv(frame: pd.DataFrame, destination: str | Path) -> None:
    """Atomically write a data frame as UTF-8 CSV."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        csv_safe_frame(frame).to_csv(
            temporary, index=False, encoding="utf-8", lineterminator="\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_frame_parquet(frame: pd.DataFrame, destination: str | Path) -> None:
    """Atomically write a data frame as Parquet."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    """Atomically copy one regular file."""

    source_path = Path(source)
    if not source_path.is_file() or source_path.is_symlink():
        raise CacheIOError(f"copy source is not a regular file: {source_path}")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        shutil.copyfile(source_path, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(value: object, destination: str | Path) -> str:
    """Validate and atomically save one tensor-only payload, returning its hash."""

    _validate_torch_payload(value)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def safe_torch_load(path: str | Path) -> object:
    """Load a tensor-only artifact onto CPU with restricted unpickling."""

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise CacheIOError(f"cannot safely load Torch cache {path}: {error}") from error


def _validate_torch_payload(value: object) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or not value.is_contiguous():
            raise CacheIOError("Torch cache tensors must be contiguous CPU tensors")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise CacheIOError("Torch cache tensors must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise CacheIOError("Torch cache mapping key is unsafe")
            _validate_torch_payload(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_torch_payload(item)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise CacheIOError("Torch cache scalar must be finite")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise CacheIOError(f"unsupported Torch cache value: {type(value).__name__}")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
