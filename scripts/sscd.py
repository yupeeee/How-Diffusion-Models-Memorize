#!/usr/bin/env python3
"""Score every completed generation record with SSCD."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.cli import add_run_arguments, validate_seed_block  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute SSCD from cached full-resolution endpoints.",
        allow_abbrev=False,
    )
    return add_run_arguments(parser)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        validate_seed_block(arguments.seed_start, arguments.N)
    except ValueError as error:
        parser.error(str(error))
    from utils.experiments.sscd import run_sscd

    result = run_sscd(
        PROJECT_ROOT,
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.T,
        num_seeds=arguments.N,
        seed_start=arguments.seed_start,
        device=arguments.device,
    )
    print(f"Summary: {result.summary_path}")
    return int(result.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
