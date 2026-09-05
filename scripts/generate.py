#!/usr/bin/env python3
"""Generate or resume reusable full diffusion trajectories."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.cli import add_generation_arguments, validate_seed_block  # noqa: E402


def _cached_statistics_report(
    generation_directory: Path, completed_rows: int
) -> Path | None:
    """Return an existing structurally valid target-statistics report."""

    from utils.common.io import file_sha256, read_json, safe_torch_load
    from utils.experiments.cache import GenerationPaths, list_completed_records
    from utils.experiments.latent_statistics import (
        STATISTICS_DIRECTORY_NAME,
        STATISTICS_SCHEMA_VERSION,
        TARGET_LATENT_MARKER_FINGERPRINT_FIELD,
        target_latent_marker_fingerprint,
    )

    statistics_directory = generation_directory / STATISTICS_DIRECTORY_NAME
    report_path = statistics_directory / "report.json"
    run_config_path = generation_directory / "run_config.json"
    artifacts = {
        "mean": statistics_directory / "mean.pt",
        "population_std": statistics_directory / "population_std.pt",
    }
    guarded_paths = (report_path, run_config_path, *artifacts.values())
    if any(not path.is_file() or path.is_symlink() for path in guarded_paths):
        return None
    try:
        report = read_json(report_path)
        run_config = read_json(run_config_path)
        paths = GenerationPaths(generation_directory)
        records = list_completed_records(paths)
        if len(records) != completed_rows:
            return None
        marker_fingerprint = target_latent_marker_fingerprint(records)
        if report.get(TARGET_LATENT_MARKER_FINGERPRINT_FIELD) != marker_fingerprint:
            return None
        for record in records:
            hashes = record.metadata.get("tensor_file_sha256")
            expected_hash = (
                hashes.get("target_latent") if isinstance(hashes, Mapping) else None
            )
            target_path = paths.target_latent_path(record.original_index)
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or target_path.is_symlink()
                or not target_path.is_file()
                or target_path.stat().st_size <= 0
                or file_sha256(target_path) != expected_hash
            ):
                return None
        latent_shape = report.get("latent_shape")
        if (
            report.get("schema_version") != STATISTICS_SCHEMA_VERSION
            or report.get("artifact") != "target_latent_statistics"
            or report.get("completed_target_latent_rows") != completed_rows
            or report.get("scientific_config_hash")
            != run_config.get("scientific_config_hash")
            or isinstance(latent_shape, (str, bytes))
            or not isinstance(latent_shape, Sequence)
        ):
            return None
        shape = tuple(latent_shape)
        if len(shape) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape
        ):
            return None
        coordinate_artifacts = report.get("coordinate_artifacts")
        if not isinstance(coordinate_artifacts, Mapping):
            return None
        for name, path in artifacts.items():
            description = coordinate_artifacts.get(name)
            if not isinstance(description, Mapping):
                return None
            expected_path = description.get("path")
            expected_hash = description.get("sha256")
            if (
                not isinstance(expected_path, str)
                or (PROJECT_ROOT / expected_path).resolve() != path.resolve()
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or file_sha256(path) != expected_hash
            ):
                return None
            tensor = safe_torch_load(path)
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.dtype != torch.float64
                or tensor.device.type != "cpu"
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all())
            ):
                return None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate every available Webster prompt-target pair.",
        allow_abbrev=False,
    )
    return add_generation_arguments(parser)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        validate_seed_block(arguments.seed_start, arguments.N)
    except ValueError as error:
        parser.error(str(error))
    from utils.experiments.generation import generate_webster_trajectories

    result = generate_webster_trajectories(
        PROJECT_ROOT,
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.T,
        num_seeds=arguments.N,
        seed_start=arguments.seed_start,
        downscale=arguments.downscale,
        device=arguments.device,
        overwrite=arguments.overwrite,
    )
    print(f"Summary: {result.summary_path}")
    if result.completed_rows < 1:
        print(
            "Target latent statistics failed: no completed target latents",
            file=sys.stderr,
        )
        return 1
    if result.complete_cache_hit:
        print(
            "Generation and denoising were skipped because every expected "
            "trajectory is complete; pass --overwrite to regenerate them."
        )
        statistics_report = _cached_statistics_report(
            result.summary_path.parent, result.completed_rows
        )
        if statistics_report is not None:
            print(
                "Target latent statistics skipped: existing report: "
                f"{statistics_report}"
            )
            return int(result.exit_code)
        print("Cached target latent statistics report is absent; computing it now.")
    try:
        from utils.experiments.latent_statistics import (
            compute_target_latent_statistics,
            format_target_latent_statistics,
        )

        statistics = compute_target_latent_statistics(
            PROJECT_ROOT, result.summary_path.parent
        )
    except Exception as error:
        print(
            f"Target latent statistics failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(format_target_latent_statistics(statistics.report))
    print(f"Target latent statistics: {statistics.report_path}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
