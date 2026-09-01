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


def build_parser() -> argparse.ArgumentParser:
    """Build the shared cached-analysis parser."""

    parser = argparse.ArgumentParser(
        description="Compute terminal latent proximity from local caches only.",
        allow_abbrev=False,
    )
    return add_run_arguments(parser)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested analysis and return its process exit status."""

    arguments = build_parser().parse_args(argv)
    from utils.experiments.proximity import run_proximity

    summary = run_proximity(
        PROJECT_ROOT,
        model_name=arguments.model,
        scheduler_name=arguments.scheduler,
        guidance_scale=arguments.g,
        num_inference_steps=arguments.T,
        num_seeds=arguments.N,
        seed_start=arguments.seed_start,
    )
    print(f"Summary: {summary.paths.summary_json}")
    if summary.exit_code == 0:
        from utils.experiments.held_out import (
            HeldOutTVExportError,
            export_held_out_tv_results,
        )

        try:
            held_out = export_held_out_tv_results(
                PROJECT_ROOT,
                model_name=arguments.model,
                generation_run=summary.paths.generation_run,
                output_directory=summary.paths.output_directory,
            )
        except HeldOutTVExportError as error:
            print(
                "Held-out TV gallery failed after scientific proximity "
                f"completed: {error}",
                file=sys.stderr,
            )
            return 1
        action = "Reused" if held_out.reused else "Saved"
        print(
            f"{action} {held_out.prompt_count}/{held_out.total_prompt_count} "
            "held-out TV prompt previews: "
            f"{held_out.gallery_path}"
        )
        if held_out.missing_prompt_count:
            print(
                f"Warning: {held_out.missing_prompt_count} frozen held-out TV "
                "prompt(s) had no generation in this run; see held_out_tv/config.json.",
                file=sys.stderr,
            )
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
