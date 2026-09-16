"""Scalar-only bundle identities and discovery; no scientific tensor reads."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from pathlib import Path

from utils.common.cli import MAX_SEED, generation_cache_parent_name

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


DIRECT_FORMULA_VERSION = "direct-seven-statements-1"
EVIDENCE_FORMULA_VERSION = "fixed-seven-statement-evidence-1"
DIRECT_SCIENCE_DEFAULTS = {
    "num_loss_seeds": 64,
    "loss_seed": 0,
    "loss_timesteps": "initial",
    "num_unconditional_loss_seeds": 256,
    "reference_law": "cached-targets",
    "reference_snr_decades": 6.0,
    "terminal_noise_run_alpha": 0.05,
}
NUMERICAL_DEFAULTS = {
    "numerical_decimal_precision": 64,
    "numerical_max_decimal_products": 2_000_000,
    "numerical_max_variation_nodes": 65,
    "numerical_variation_absolute_width": 1e-6,
}
NUMERICAL_KEYS = tuple(NUMERICAL_DEFAULTS)
# Optional new measurements are separated from unchanged upstream identities.
FOUR_STAGE_OPTION_KEYS = ("counterfactual_unconditional", "counterfactual_steps")
DIRECT_SCIENCE_KEYS = (*DIRECT_SCIENCE_DEFAULTS, "measure_unconditional_loss", "reference_manifest", *NUMERICAL_KEYS)


def _integer_option(value, name, *, minimum):
    if isinstance(value, bool):
        raise TheoryError(f"{name} must be an integer >= {minimum}")
    try:
        number = operator.index(value)
    except TypeError as error:
        raise TheoryError(f"{name} must be an integer >= {minimum}") from error
    if number < minimum:
        raise TheoryError(f"{name} must be an integer >= {minimum}")
    return number


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
    num_loss_seeds=64,
    loss_seed=0,
    loss_timesteps="initial",
    num_unconditional_loss_seeds=256,
    measure_unconditional_loss=None,
    reference_law="cached-targets",
    reference_manifest=None,
    reference_snr_decades=6.0,
    terminal_noise_run_alpha=0.05,
    numerical_decimal_precision=64,
    numerical_max_decimal_products=2_000_000,
    numerical_max_variation_nodes=65,
    numerical_variation_absolute_width=1e-6,
    counterfactual_unconditional=False,
    counterfactual_steps=None,
    **extra,
):
    if extra:
        raise TheoryError("Unknown scientific configuration options: " + ", ".join(sorted(extra)))
    if isinstance(reference_snr_decades, bool) or not math.isfinite(float(reference_snr_decades)) or not 0 < float(reference_snr_decades) <= 12:
        raise TheoryError("reference_snr_decades must be finite in (0, 12]")
    if isinstance(terminal_noise_run_alpha, bool) or not math.isfinite(float(terminal_noise_run_alpha)) or not 0 < float(terminal_noise_run_alpha) < 1:
        raise TheoryError("terminal_noise_run_alpha must be finite in (0, 1)")
    numerical = {
        "numerical_decimal_precision": _integer_option(numerical_decimal_precision, "numerical_decimal_precision", minimum=32),
        "numerical_max_decimal_products": _integer_option(numerical_max_decimal_products, "numerical_max_decimal_products", minimum=0),
        "numerical_max_variation_nodes": _integer_option(numerical_max_variation_nodes, "numerical_max_variation_nodes", minimum=1),
    }
    if isinstance(numerical_variation_absolute_width, bool) or not math.isfinite(float(numerical_variation_absolute_width)) or float(numerical_variation_absolute_width) <= 0:
        raise TheoryError("numerical_variation_absolute_width must be finite and positive")
    numerical["numerical_variation_absolute_width"] = float(numerical_variation_absolute_width)
    if loss_timesteps not in {"initial", "saved"}:
        raise TheoryError("loss_timesteps must be initial or saved")
    if reference_law not in {"cached-targets", "manifest"}:
        raise TheoryError("reference_law must be cached-targets or manifest")
    if reference_law == "manifest" and reference_manifest is None:
        raise TheoryError("--reference-manifest is required with --reference-law manifest")
    if reference_law != "manifest" and reference_manifest is not None:
        raise TheoryError("--reference-manifest requires --reference-law manifest")
    if measure_unconditional_loss is None:
        measure_unconditional_loss = loss_timesteps == "saved"
    if not isinstance(measure_unconditional_loss, bool):
        raise TheoryError("measure_unconditional_loss must be boolean")
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
        num_loss_seeds=_integer_option(num_loss_seeds, "num_loss_seeds", minimum=1),
        loss_seed=_integer_option(loss_seed, "loss_seed", minimum=0),
        loss_timesteps=loss_timesteps,
        num_unconditional_loss_seeds=_integer_option(num_unconditional_loss_seeds, "num_unconditional_loss_seeds", minimum=1),
        measure_unconditional_loss=measure_unconditional_loss,
        reference_law=reference_law,
        reference_snr_decades=float(reference_snr_decades),
        terminal_noise_run_alpha=float(terminal_noise_run_alpha),
    )
    result.update(numerical)
    if not isinstance(counterfactual_unconditional, bool):
        raise TheoryError("counterfactual_unconditional must be boolean")
    if counterfactual_steps is not None and not counterfactual_unconditional:
        raise TheoryError("counterfactual_steps requires --counterfactual-unconditional")
    if counterfactual_unconditional:
        steps = [0] if counterfactual_steps is None else counterfactual_steps
        if isinstance(steps, str):
            try:
                steps = [int(value.strip()) for value in steps.split(",")]
            except ValueError as error:
                raise TheoryError("counterfactual_steps must be comma-separated chronological integers") from error
        if not isinstance(steps, (list, tuple)) or not steps:
            raise TheoryError("counterfactual_steps must be a nonempty fixed step list")
        steps = sorted(set(_integer_option(step, "counterfactual_steps", minimum=0) for step in steps))
        if any(step >= result["num_inference_steps"] - 1 for step in steps):
            raise TheoryError("Counterfactual snapshots require a defined next prediction: 0 <= k < K_steps-1")
        result.update(counterfactual_unconditional=True, counterfactual_steps=steps)
    if result["loss_seed"] > MAX_SEED:
        raise TheoryError(f"loss_seed must not exceed {MAX_SEED}")
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
    if reference_manifest is not None:
        result["reference_manifest"] = str(Path(reference_manifest).absolute())
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
    return Path(project_root).resolve() / "outputs" / run / "theory_measurements"


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
