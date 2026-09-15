"""Scalar-only bundle identities and discovery; no scientific tensor reads."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from utils.common.cli import generation_cache_parent_name

SCHEMA_VERSION = 3
FORMULA_VERSION = "cache-theory-3.0-proposition5"
EVIDENCE_LEVELS = (
    "finite_noise_observation",
    "model_center_diagnostic",
    "candidate_distribution_diagnostic",
    "identified_training_reference",
    "algebraic_qa",
    "unavailable",
    "not_applicable",
)


class TheoryError(RuntimeError):
    """A required source or scalar result violates the analysis contract."""


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise TheoryError(
            f"Missing/incompatible artifact {path}: {error}. Rebuild with --recompute-experiments."
        ) from error
    if not isinstance(value, dict):
        raise TheoryError(f"Expected JSON object: {path}")
    return value


def numerical_config(
    *,
    model_name,
    scheduler_name,
    guidance_scale=7.5,
    num_inference_steps=50,
    num_seeds=20,
    center="reference-initial",
    cached_baseline=None,
    target_error_tolerance=None,
    **extra,
):
    if (model_name, scheduler_name) not in {
        ("sdv1", "ddim"),
        ("sdv1", "ddpm"),
        ("sdv2", "ddim"),
        ("realvis", "ddim"),
    }:
        raise TheoryError(f"Unsupported model/scheduler: {model_name}/{scheduler_name}")
    if center not in {"reference-initial", "zero", "cached-baseline"}:
        raise TheoryError(f"Unknown center: {center}")
    result = dict(
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        num_seeds=int(num_seeds),
        center=center,
    )
    # Use the existing helper for validation and all float-dependent paths.
    generation_cache_parent_name(
        model_name,
        scheduler_name,
        float(guidance_scale),
        num_inference_steps,
        num_seeds,
    )
    if target_error_tolerance is not None:
        tolerance = float(target_error_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise TheoryError(
                "Target-error tolerance must be finite, nonnegative and in raw latent L2 units"
            )
        result["target_error_tolerance"] = tolerance
    if cached_baseline is not None:
        result["cached_baseline"] = str(Path(cached_baseline).resolve())
    return result


def analysis_parent(project_root, **config) -> Path:
    c = numerical_config(**config)
    run = generation_cache_parent_name(
        c["model_name"],
        c["scheduler_name"],
        c["guidance_scale"],
        c["num_inference_steps"],
        c["num_seeds"],
    )
    return Path(project_root).resolve() / "outputs" / run / "theory_v2"


def find_analysis_bundle(project_root, **config) -> Path:
    """Resolve the explicitly published configuration, never a different/latest run."""
    c = numerical_config(**config)
    parent = analysis_parent(project_root, **c)
    index = read_object(parent / "index.json")
    matches = [entry for entry in index.get("analyses", []) if entry.get("config") == c]
    if len(matches) != 1:
        raise TheoryError(
            f"Missing/ambiguous scalar analysis for {c}: {parent}/index.json. Run --recompute-experiments."
        )
    digest = matches[0].get("analysis_hash", "")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise TheoryError(f"Unsafe analysis hash in {parent}/index.json")
    bundle = parent / digest
    manifest = read_object(bundle / "manifest.json")
    if (
        manifest.get("analysis_hash") != digest
        or manifest.get("config") != c
        or not manifest.get("complete")
    ):
        raise TheoryError(
            f"Incomplete/incompatible scalar bundle {bundle}; run --recompute-experiments."
        )
    return bundle


def validate_source_metadata(bundle, project_root=None) -> None:
    """Root plotting can detect changed source metadata without reading .pt files.

    Standalone bundle plotting deliberately does not call this function; copied
    bundles have all required scalar provenance internally.
    """
    manifest = read_object(Path(bundle) / "manifest.json")
    root = Path(project_root or manifest["source_root"])
    for relative, expected in manifest.get("source_metadata_files", {}).items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or digest_file(path) != expected:
            raise TheoryError(
                f"Stale/missing source metadata {path}; run --recompute-experiments."
            )
    for relative, expected_names in manifest.get("source_marker_inventory", {}).items():
        directory = root / relative
        names = sorted(p.name for p in directory.glob("*.json"))
        if names != expected_names:
            raise TheoryError(
                f"Source completion inventory changed: {directory}; run --recompute-experiments."
            )
