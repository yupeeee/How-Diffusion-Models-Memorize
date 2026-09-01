#!/usr/bin/env python3
"""Generate or resume reusable full diffusion trajectories."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.cli import add_generation_arguments, validate_seed_block  # noqa: E402


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
    )
    print(f"Summary: {result.summary_path}")
    if result.completed_rows < 1:
        print(
            "Target latent statistics failed: no completed target latents",
            file=sys.stderr,
        )
        return 1
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
