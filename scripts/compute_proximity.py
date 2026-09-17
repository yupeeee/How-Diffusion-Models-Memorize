#!/usr/bin/env python3
"""Command-line entry point for cache-only proximity and plotting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.cli import add_run_arguments  # noqa: E402
from utils.data.selection import (  # noqa: E402
    DEFAULT_SELECTION_STRATEGY,
    SELECTION_STRATEGIES,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the shared cached-analysis parser."""

    parser = argparse.ArgumentParser(
        description="Compute terminal latent proximity from local caches only.",
        allow_abbrev=False,
    )
    add_run_arguments(parser, include_device=False)
    parser.add_argument(
        "--selection-strategy",
        choices=SELECTION_STRATEGIES,
        default=DEFAULT_SELECTION_STRATEGY,
        help="GMM-only prompt selection (default: gmm)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute and replace this strategy's cached proximity outputs",
    )
    mode.add_argument(
        "--plot",
        action="store_true",
        help="save PDFs under project-root figures/<experiment>/proximity/<seed-role>/ from saved scalars and cached retained/discarded images",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested analysis and return its process exit status."""

    arguments = build_parser().parse_args(argv)
    from utils.experiments.proximity import plot_proximity, run_proximity

    common = {
        "model_name": arguments.model,
        "scheduler_name": arguments.scheduler,
        "guidance_scale": arguments.g,
        "num_inference_steps": arguments.T,
        "num_seeds": arguments.N,
        "seed_start": arguments.seed_start,
        "selection_strategy": arguments.selection_strategy,
    }
    if arguments.plot:
        summary = plot_proximity(PROJECT_ROOT, **common)
    else:
        summary = run_proximity(
            PROJECT_ROOT,
            overwrite=arguments.overwrite,
            **common,
        )
    print(f"Summary: {summary.paths.summary_json}")
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
