#!/usr/bin/env python3
"""Recover, organize, and inspect the Webster benchmark."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data.recovery import (  # noqa: E402
    DEFAULT_DIRECT_ATTEMPTS,
    DEFAULT_DIRECT_WORKERS,
    DEFAULT_PER_HOST_CONCURRENCY,
    MAX_DIRECT_ATTEMPTS,
    MAX_DIRECT_WORKERS,
    MAX_PER_HOST_CONCURRENCY,
    prepare_webster_dataset,
)


def _bounded_integer(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume Webster recovery and validate all local model views.",
        allow_abbrev=False,
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--direct-workers",
        type=_bounded_integer(1, MAX_DIRECT_WORKERS),
        default=DEFAULT_DIRECT_WORKERS,
        help=f"parallel direct-URL workers (default: {DEFAULT_DIRECT_WORKERS})",
    )
    parser.add_argument(
        "--direct-attempts",
        type=_bounded_integer(1, MAX_DIRECT_ATTEMPTS),
        default=DEFAULT_DIRECT_ATTEMPTS,
        help=f"attempts per direct URL (default: {DEFAULT_DIRECT_ATTEMPTS})",
    )
    parser.add_argument(
        "--per-host-concurrency",
        "--per-host",
        type=_bounded_integer(1, MAX_PER_HOST_CONCURRENCY),
        default=DEFAULT_PER_HOST_CONCURRENCY,
        help=(
            "simultaneous requests to one host "
            f"(default: {DEFAULT_PER_HOST_CONCURRENCY})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.per_host_concurrency > arguments.direct_workers:
        parser.error("--per-host-concurrency cannot exceed --direct-workers")
    try:
        result = prepare_webster_dataset(
            arguments.root,
            direct_workers=arguments.direct_workers,
            direct_attempts=arguments.direct_attempts,
            per_host_concurrency=arguments.per_host_concurrency,
        )
    except Exception as error:
        print(f"Webster recovery failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return int(getattr(result, "exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
