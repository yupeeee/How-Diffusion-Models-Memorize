"""Command-line parsing for one reproducible generation-cache run."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

MODEL_CHOICES = ("sdv1", "sdv2", "realvis")
SCHEDULER_CHOICES = ("ddim", "ddpm")
MAX_SEED = 2**63 - 1


class CLIValueError(argparse.ArgumentTypeError):
    """A public command-line value is malformed."""


def finite_float(value: str) -> float:
    """Parse one finite floating-point value."""

    try:
        number = float(value)
    except ValueError as error:
        raise CLIValueError("must be a finite float") from error
    if not math.isfinite(number):
        raise CLIValueError("must be a finite float")
    return number


def positive_integer(value: str) -> int:
    """Parse one integer greater than zero."""

    try:
        number = int(value)
    except ValueError as error:
        raise CLIValueError("must be a positive integer") from error
    if number <= 0:
        raise CLIValueError("must be a positive integer")
    return number


def nonnegative_integer(value: str) -> int:
    """Parse one integer in the supported random-seed domain."""

    try:
        number = int(value)
    except ValueError as error:
        raise CLIValueError("must be a nonnegative integer") from error
    if number < 0 or number > MAX_SEED:
        raise CLIValueError(f"must be an integer between 0 and {MAX_SEED}")
    return number


def validate_seed_block(seed_start: object, num_seeds: object) -> tuple[int, int]:
    """Validate one contiguous seed block without materializing it."""

    if (
        isinstance(seed_start, bool)
        or not isinstance(seed_start, int)
        or seed_start < 0
        or seed_start > MAX_SEED
    ):
        raise ValueError(f"seed start must be an integer between 0 and {MAX_SEED}")
    if isinstance(num_seeds, bool) or not isinstance(num_seeds, int) or num_seeds <= 0:
        raise ValueError("number of seeds must be a positive integer")
    if seed_start + num_seeds - 1 > MAX_SEED:
        raise ValueError(f"seed block must end at or before {MAX_SEED}")
    return seed_start, num_seeds


def add_run_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model, scheduler, guidance, step, and seed-block arguments."""

    parser.add_argument("--model", choices=MODEL_CHOICES, default="sdv1")
    parser.add_argument("--scheduler", choices=SCHEDULER_CHOICES, default="ddim")
    parser.add_argument("--g", type=finite_float, default=7.5, metavar="SCALE")
    parser.add_argument("--T", type=positive_integer, default=50, metavar="STEPS")
    parser.add_argument("--N", type=positive_integer, default=20, metavar="SEEDS")
    parser.add_argument(
        "--seed-start",
        type=nonnegative_integer,
        default=0,
        metavar="SEED",
        help="first seed in the contiguous N-seed block (default: 0)",
    )
    return parser


def add_generation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared run arguments and generation-preview downscale."""

    add_run_arguments(parser)
    parser.add_argument(
        "--downscale",
        type=positive_integer,
        default=4,
        metavar="FACTOR",
    )
    return parser


def stable_float(value: float) -> str:
    """Format a finite float consistently for run names."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("guidance scale must be finite")
    return "0" if number == 0 else format(number, ".15g")


def generation_run_name(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int = 0,
) -> str:
    """Return the only supported generation run-directory name."""

    if model_name not in MODEL_CHOICES:
        raise ValueError(f"unsupported model: {model_name}")
    if scheduler_name not in SCHEDULER_CHOICES:
        raise ValueError(f"unsupported scheduler: {scheduler_name}")
    if num_inference_steps <= 0:
        raise ValueError("steps and seeds must be positive")
    seed_start, num_seeds = validate_seed_block(seed_start, num_seeds)
    seed_identity = (
        f"_N{num_seeds}"
        if seed_start == 0
        else f"_S{seed_start}_N{num_seeds}"
    )
    return (
        f"{model_name}_{scheduler_name}_g{stable_float(guidance_scale)}"
        f"_T{num_inference_steps}{seed_identity}"
    )


@dataclass(frozen=True, slots=True)
class RunArguments:
    """The scientific identity shared by generation and cached analyses."""

    model_name: str
    scheduler_name: str
    guidance_scale: float
    num_inference_steps: int
    num_seeds: int
    seed_start: int = 0

    @property
    def name(self) -> str:
        return generation_run_name(
            self.model_name,
            self.scheduler_name,
            self.guidance_scale,
            self.num_inference_steps,
            self.num_seeds,
            self.seed_start,
        )
