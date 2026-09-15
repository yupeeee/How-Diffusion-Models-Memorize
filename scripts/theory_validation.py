#!/usr/bin/env python3
"""Reduce protected trajectory caches, or render an independent scalar bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.cli import (  # noqa: E402
    device_argument,
    finite_float,
    positive_integer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=("sdv1", "sdv2", "realvis"), default="sdv1")
    parser.add_argument("--scheduler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--g", type=finite_float, default=7.5)
    parser.add_argument("--T", type=positive_integer, default=50)
    parser.add_argument("--N", type=positive_integer, default=20)
    centers = parser.add_mutually_exclusive_group()
    centers.add_argument(
        "--center",
        choices=("reference-initial", "zero", "cached-baseline"),
        default="reference-initial",
    )
    centers.add_argument(
        "--use-mu",
        dest="center",
        action="store_const",
        const="cached-baseline",
        help="legacy existing independent baseline semantics; requires --cached-baseline",
    )
    centers.add_argument("--no-mu", dest="center", action="store_const", const="zero")
    parser.add_argument("--cached-baseline", type=Path)
    parser.add_argument(
        "--device",
        type=device_argument,
        default="auto",
        metavar="DEVICE",
        help=(
            "auto uses every CUDA device visible to PyTorch (CPU fallback); "
            "cpu, cuda, or cuda:N selects one device (default: auto); "
            "MPS cannot preserve the required float64 arithmetic; "
            "plotting never initializes devices"
        ),
    )
    parser.add_argument("--selection-strategy", choices=("gmm",), default="gmm")
    parser.add_argument(
        "--figure-suite",
        choices=("paper",),
        default="paper",
        help="fixed main and appendix paper suite (default); discovery modes are retired",
    )
    parser.add_argument(
        "--diagnostics",
        "--include-diagnostics",
        dest="diagnostics",
        action="store_true",
        help="also render saved optional diagnostic figures",
    )
    parser.add_argument(
        "--target-error-tolerance",
        type=finite_float,
        help="independently supplied raw latent L2 tolerance; never inferred from SSCD",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot", action="store_true", help="render saved scalars only")
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all scalar inputs without writing figures",
    )
    mode.add_argument(
        "--recompute-experiments",
        action="store_true",
        help="rebuild only derived theory from existing caches",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="copied scalar bundle; requires --plot or --validate-only",
    )
    parser.add_argument(
        "--validate-proximity",
        action="store_true",
        help="root plot preflight: validate protected saved proximity and source metadata",
    )
    parser.add_argument(
        "--source-analysis",
        type=Path,
        help="explicit validated candidate scalar bundle for analysis migration",
    )
    parser.add_argument(
        "--source-logs",
        type=Path,
        help="explicit matching archived logs for missing terminal accounting during migration",
    )
    parser.add_argument("--candidate-chunk-size", type=positive_integer, default=256)
    parser.add_argument("--query-chunk-size", type=positive_integer, default=16)
    for old in (
        "--evaluation-source",
        "--num-loss-seeds",
        "--loss-seed",
        "--num-baseline-seeds",
    ):
        parser.add_argument(old, help=argparse.SUPPRESS)
    return parser


def validate_saved_proximity(project_root: Path, config: dict) -> None:
    """Reuse protected validators, without invoking any write or tensor loader."""
    from utils.common.cli import generation_run_name
    from utils.data.selection import load_target_pair_selection
    from utils.experiments.cache import generation_log_relative_path
    from utils.experiments.proximity import (
        ProximityPaths,
        _load_saved_analysis,
        _load_saved_analysis_configuration,
        _validate_saved_analysis,
        _validate_saved_analysis_configuration,
    )
    from utils.experiments.plotting import (
        _fully_plottable_prompt_rows,
        _validate_inclusion,
        _validate_measurements,
        _validate_prompt_spearman,
        EXPERIMENT_SPEARMAN_COLUMN,
    )

    identity = {
        key: config[key]
        for key in (
            "model_name",
            "scheduler_name",
            "guidance_scale",
            "num_inference_steps",
            "num_seeds",
        )
    }
    selection = load_target_pair_selection(
        project_root, **identity, selection_strategy="gmm"
    )
    _validate_inclusion(selection.frame)
    for reference in (
        _fully_plottable_prompt_rows(selection.frame),
        selection.frame.loc[selection.frame.include_prompt],
    ):
        _validate_measurements(reference)
        _validate_prompt_spearman(reference, column="prompt_spearman")
    generation = project_root / generation_log_relative_path(**identity, seed_start=0)
    name = generation_run_name(**identity, seed_start=0)
    paths = ProximityPaths.build(
        project_root,
        generation,
        output_run_name=name,
        role="experiment",
        seed_start=0,
        num_seeds=config["num_seeds"],
    )
    saved = _load_saved_analysis_configuration(paths.run_config_json)
    _validate_saved_analysis_configuration(
        saved,
        paths=paths,
        run_name=name,
        **identity,
        selection=selection,
        selection_strategy="gmm",
    )
    frame = _load_saved_analysis(paths.output_directory / "proximity.csv")
    _validate_saved_analysis(
        frame,
        selection=selection,
        model_name=config["model_name"],
        num_seeds=config["num_seeds"],
        selection_strategy="gmm",
    )
    _validate_prompt_spearman(frame, column=EXPERIMENT_SPEARMAN_COLUMN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for old in (
        "evaluation_source",
        "num_loss_seeds",
        "loss_seed",
        "num_baseline_seeds",
    ):
        if getattr(args, old) is not None:
            parser.error(
                f"--{old.replace('_', '-')} is removed from cache-only theory; invoke the explicit legacy wrapper for independent inference"
            )
    if args.target_error_tolerance is not None and args.target_error_tolerance < 0:
        parser.error(
            "--target-error-tolerance must be nonnegative in raw latent L2 units"
        )
    if args.bundle and args.target_error_tolerance is not None:
        parser.error(
            "A copied bundle uses its saved tolerance; recompute analysis to change it"
        )
    if args.bundle and not (args.plot or args.validate_only):
        parser.error("--bundle requires --plot or --validate-only")
    if args.validate_proximity and (not args.validate_only or args.bundle):
        parser.error(
            "--validate-proximity requires --validate-only for a repository run"
        )
    if (
        not args.bundle
        and args.center == "cached-baseline"
        and args.cached_baseline is None
    ):
        parser.error(
            "--center cached-baseline requires --cached-baseline with an existing independent baseline"
        )
    if args.cached_baseline and args.center != "cached-baseline":
        parser.error("--cached-baseline requires --center cached-baseline")
    if (args.source_analysis or args.source_logs) and (
        args.plot or args.validate_only or args.bundle
    ):
        parser.error(
            "Source migration is analysis-only; plotting uses saved compact paper inputs"
        )
    if args.source_logs and not args.source_analysis:
        parser.error("--source-logs requires an explicit --source-analysis")
    from utils.experiments.theory.contracts import TheoryError, numerical_config
    from utils.experiments.theory.paper_contracts import (
        PaperPaths,
        render_saved_paper,
        validate_paper_bundle,
        recompute_command,
    )

    config = numerical_config(
        model_name=args.model,
        scheduler_name=args.scheduler,
        guidance_scale=args.g,
        num_inference_steps=args.T,
        num_seeds=args.N,
        center=args.center,
        cached_baseline=args.cached_baseline,
        target_error_tolerance=args.target_error_tolerance,
    )
    try:
        bundle = (
            args.bundle.absolute()
            if args.bundle
            else PaperPaths.build(PROJECT_ROOT, **config).output_directory
        )
        if args.plot or args.validate_only:
            validate_paper_bundle(
                bundle,
                expected_config=None if args.bundle else config,
                diagnostics=args.diagnostics,
            )
            if args.validate_proximity:
                try:
                    validate_saved_proximity(PROJECT_ROOT, config)
                except (TheoryError, OSError, ValueError, RuntimeError) as error:
                    normal_command = recompute_command(config).replace(
                        " --recompute-experiments", ""
                    )
                    print(
                        f"Saved proximity preflight failed: {error}. "
                        f"Run the normal pipeline to align proximity with the current frozen selection: {normal_command}",
                        file=sys.stderr,
                    )
                    return 1
            if args.plot:
                render_saved_paper(
                    bundle,
                    expected_config=None if args.bundle else config,
                    diagnostics=args.diagnostics,
                )
        else:
            from utils.experiments.theory.paper_reduce import run_paper

            bundle = run_paper(
                PROJECT_ROOT,
                **config,
                recompute=args.recompute_experiments,
                diagnostics=args.diagnostics,
                device=args.device,
                candidate_chunk_size=args.candidate_chunk_size,
                query_chunk_size=args.query_chunk_size,
                source_analysis=args.source_analysis,
                source_logs=args.source_logs,
            )
        print(
            f"{'Validated' if args.validate_only else 'Theory outputs'} (paper): {bundle}"
        )
        return 0
    except (TheoryError, OSError, ValueError, RuntimeError) as error:
        print(
            f"theory_validation.py: {error}\nRebuild saved analysis with {recompute_command(config)}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
