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
            HeldOutNExportError,
            HeldOutTVExportError,
            export_held_out_n_results,
            export_held_out_tv_results,
        )

        exporters = (
            ("TV", "held_out_tv", export_held_out_tv_results),
            ("N", "held_out_n", export_held_out_n_results),
        )
        gallery_failed = False
        for category, directory_name, export_results in exporters:
            try:
                held_out = export_results(
                    PROJECT_ROOT,
                    model_name=arguments.model,
                    generation_run=summary.paths.generation_run,
                    output_directory=summary.paths.output_directory,
                )
            except (HeldOutTVExportError, HeldOutNExportError) as error:
                print(
                    f"Held-out {category} gallery failed after scientific "
                    f"proximity completed: {error}",
                    file=sys.stderr,
                )
                gallery_failed = True
                continue
            action = "Reused" if held_out.reused else "Saved"
            print(
                f"{action} {held_out.prompt_count}/{held_out.total_prompt_count} "
                f"held-out {category} prompt previews: {held_out.gallery_path}"
            )
            if held_out.missing_prompt_count:
                print(
                    f"Warning: {held_out.missing_prompt_count} frozen held-out "
                    f"{category} prompt(s) had no generation in this run; see "
                    f"{directory_name}/config.json.",
                    file=sys.stderr,
                )
        if gallery_failed:
            return 1
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
