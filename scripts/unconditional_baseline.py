#!/usr/bin/env python3
"""Build or validate the shared model-implied unconditional baseline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.cli import (  # noqa: E402
    SCHEDULER_CHOICES,
    device_argument,
    finite_float,
    positive_integer,
    validate_seed_block,
)
from utils.experiments.unconditional_baseline import (  # noqa: E402
    DEFAULT_NUM_BASELINE_SEEDS,
    UnconditionalBaselineError,
    compute_unconditional_baseline,
    load_unconditional_baseline,
)
from utils.models.registry import model_names  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the model-implied unconditional baseline by direct "
            "empty-condition inference over a dedicated Gaussian seed pool."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model", choices=model_names(), default="sdv1")
    parser.add_argument("--scheduler", choices=SCHEDULER_CHOICES, default="ddim")
    parser.add_argument("--g", type=finite_float, default=7.5, metavar="SCALE")
    parser.add_argument("--T", type=positive_integer, default=50, metavar="STEPS")
    parser.add_argument("--N", type=positive_integer, default=20, metavar="SEEDS")
    parser.add_argument(
        "--num-baseline-seeds",
        type=positive_integer,
        default=DEFAULT_NUM_BASELINE_SEEDS,
        metavar="BASELINE_SEEDS",
        help=(
            "number of dedicated baseline seeds beginning at N "
            f"(default: {DEFAULT_NUM_BASELINE_SEEDS})"
        ),
    )
    parser.add_argument(
        "--device",
        type=device_argument,
        default="auto",
        metavar="DEVICE",
        help=(
            "auto uses every CUDA device visible to PyTorch; cpu, mps, "
            "cuda, or cuda:N selects one device (default: auto)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute and atomically replace the saved baseline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        validate_seed_block(arguments.N, arguments.N)
        validate_seed_block(arguments.N, arguments.num_baseline_seeds)
    except ValueError as error:
        parser.error(str(error))
    if not arguments.overwrite:
        try:
            cached = load_unconditional_baseline(
                PROJECT_ROOT,
                model_name=arguments.model,
                scheduler_name=arguments.scheduler,
                guidance_scale=arguments.g,
                num_inference_steps=arguments.T,
                num_seeds=arguments.N,
                num_baseline_seeds=arguments.num_baseline_seeds,
                load_tensor=True,
            )
        except UnconditionalBaselineError:
            pass
        else:
            print(
                "Unconditional baseline skipped: existing validated artifact: "
                f"{cached.metadata_path}"
            )
            return 0
    try:
        artifact = compute_unconditional_baseline(
            PROJECT_ROOT,
            model_name=arguments.model,
            scheduler_name=arguments.scheduler,
            guidance_scale=arguments.g,
            num_inference_steps=arguments.T,
            num_seeds=arguments.N,
            num_baseline_seeds=arguments.num_baseline_seeds,
            device=arguments.device,
        )
    except UnconditionalBaselineError as error:
        print(f"Unconditional baseline failed: {error}", file=sys.stderr)
        return 1
    print(f"Unconditional baseline tensor: {artifact.tensor_path}")
    print(f"Unconditional baseline metadata: {artifact.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
