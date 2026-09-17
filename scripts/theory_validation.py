#!/usr/bin/env python3
"""Measure the four-stage mechanism suite, or render a saved scalar bundle."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common.cli import (  # noqa: E402
    device_argument,
    finite_float,
    positive_integer,
)


def _configure_dependency_warnings():
    # Diffusers ConfigMixin injects this obsolete option internally, and Hub
    # removes it before downloading. Limit the exception to that exact warning;
    # do not change dependencies, model loading, or scientific cache identities.
    warnings.filterwarnings(
        "ignore",
        message=(
            r"\AThe `local_dir_use_symlinks` argument is deprecated and ignored in "
            r"`hf_hub_download`\. Downloading to a local directory does not use symlinks anymore\.\Z"
        ),
        category=UserWarning,
        module=r"\Ahuggingface_hub\.utils\._validators\Z",
    )


def nonnegative_integer(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from error
    if number < 0 or str(value).strip() != str(number):
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


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
        choices=("reference-initial", "cached-baseline"),
        default="reference-initial",
    )
    centers.add_argument(
        "--use-mu",
        dest="center",
        action="store_const",
        const="cached-baseline",
        help="legacy existing independent baseline semantics; requires --cached-baseline",
    )
    centers.add_argument("--no-mu", dest="center", action="store_const", const="zero",
                         help=argparse.SUPPRESS)
    parser.add_argument("--cached-baseline", type=Path)
    parser.add_argument(
        "--device",
        type=device_argument,
        default="auto",
        metavar="DEVICE",
        help=(
            "auto uses every visible CUDA device; cuda or cuda:N selects one device. "
            "Theory computation requires CUDA and never falls back to CPU/MPS; "
            "plotting never initializes devices"
        ),
    )
    parser.add_argument("--selection-strategy", choices=("gmm",), default="gmm")
    parser.add_argument(
        "--figure-suite",
        choices=("paper",),
        default="paper",
        help="six selected paper figures (default); discovery modes are retired",
    )
    parser.add_argument(
        "--diagnostics",
        "--include-diagnostics",
        dest="diagnostics",
        action="store_true",
        help="compatibility flag; exports the same six figures and retains diagnostic scalar measurements",
    )
    parser.add_argument(
        "--target-error-tolerance",
        type=finite_float,
        help="independently supplied raw latent L2 tolerance; never inferred from SSCD",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot", action="store_true", help="render saved scalars only")
    parser.add_argument("--pdf-only", action="store_true",
                        help="compatibility flag: publication already exports PDFs only and preserves cached images")
    mode.add_argument("--estimate-mean-only", action="store_true",
                      help="estimate/cache the independent initial unconditional mean only; no theory reduction or figures")
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all scalar inputs without writing figures",
    )
    mode.add_argument(
        "--refine-numerics", action="store_true",
        help="opt in to interval refinement of saved posterior/condition numerics; uses the saved positive budget or 2000000 when omitted; never runs learned probes or upstream stages",
    )
    mode.add_argument(
        "--recompute-experiments",
        action="store_true",
        help="resume direct theory measurements, including missing denoiser probes; preserve upstream caches",
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
        help="retired candidate migration flag; direct analysis requires measured losses",
    )
    parser.add_argument(
        "--source-logs",
        type=Path,
        help="retired archived-log migration flag; restore declared protected sources for direct analysis",
    )
    parser.add_argument("--candidate-chunk-size", type=positive_integer, default=256)
    parser.add_argument("--query-chunk-size", type=positive_integer, default=16)
    parser.add_argument("--mean-source", choices=("reference-min-snr", "reference-initial", "cached-targets"), default=None,
                        help="theory mu: minimum analytical SNR average (default), first-step average, or exact cached-atom mean")
    parser.add_argument("--num-mean-samples", type=positive_integer, default=None,
                        help="independent analytical reference estimates for mu (default: 10000; at least 2)")
    parser.add_argument("--mean-seed", type=nonnegative_integer, default=None,
                        help="dedicated independent mean-estimation RNG root (default: 0)")
    parser.add_argument("--num-loss-seeds", type=positive_integer, default=None,
                        help="independent forward-target draws (analysis default: 64)")
    parser.add_argument("--loss-seed", type=nonnegative_integer, default=None,
                        help="domain-separated probe RNG root (analysis default: 0)")
    parser.add_argument("--loss-timesteps", choices=("initial", "saved"), default=None,
                        help="forward losses at initial or all saved inputs (default: initial)")
    parser.add_argument("--num-unconditional-loss-seeds", type=positive_integer, default=None,
                        help="forward-marginal draws (default: 256; explicit count enables stage)")
    marginal = parser.add_mutually_exclusive_group()
    marginal.add_argument("--unconditional-loss", dest="measure_unconditional_loss",
                          action="store_true", default=None)
    marginal.add_argument("--no-unconditional-loss", dest="measure_unconditional_loss",
                          action="store_false")
    parser.add_argument("--counterfactual-unconditional", action="store_true", default=None,
                        help="opt in to learned empty-prompt probes at matched next inputs; no rollout")
    parser.add_argument("--counterfactual-steps", default=None,
                        help="fixed comma-separated chronological nonterminal updates (default: 0)")
    parser.add_argument("--probe-batch-size", type=positive_integer, default=8,
                        help="execution-only denoiser batch size; default: 8")
    parser.add_argument("--reference-law", choices=("cached-targets", "manifest"), default=None)
    parser.add_argument("--reference-manifest", type=Path, default=None,
                        help="explicit validated atoms/weights manifest for --reference-law manifest")
    parser.add_argument("--reference-snr-decades", type=finite_float, default=None,
                        help="fixed 97-point analytical-only grid spans this many decades below initial SNR (default: 6)")
    parser.add_argument("--terminal-noise-run-alpha", type=finite_float, default=None,
                        help="predeclared simultaneous terminal Gaussian noise failure budget per run (default: 0.05)")
    parser.add_argument("--numerical-decimal-precision", type=positive_integer, default=None,
                        help="legacy saved-policy field; computation uses CUDA binary64 enclosures and rejects this override")
    parser.add_argument("--numerical-max-products", "--numerical-max-decimal-products",
                        dest="numerical_max_decimal_products", type=nonnegative_integer, default=None,
                        help="per-row CUDA interval-operation budget; positive values opt in (normal/recompute default: 0, numerical estimates only); --refine-numerics uses a saved positive budget or 2000000 when omitted")
    parser.add_argument("--numerical-max-variation-nodes", type=positive_integer, default=None,
                        help="fixed per-row variation enclosure node budget (default: 65)")
    parser.add_argument("--numerical-variation-absolute-width", type=finite_float, default=None,
                        help="requested variation enclosure width, raw latent L2 (default: 1e-6)")
    for old in ("--evaluation-source", "--num-baseline-seeds"):
        parser.add_argument(old, help=argparse.SUPPRESS)
    return parser


def validate_saved_proximity(project_root: Path, config: dict) -> None:
    """Validate saved scalars and cached example images without writes or tensor reads."""
    from utils.common.cli import generation_run_name
    from utils.data.selection import load_target_pair_selection, target_pair_selection_directory
    from utils.experiments.cache import generation_log_relative_path
    from utils.experiments.proximity import (
        ProximityPaths,
        _reference_output_paths,
        _write_examples,
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

    reference_generation = project_root / generation_log_relative_path(
        **identity, seed_start=config["num_seeds"])
    reference_paths = _reference_output_paths(
        project_root, reference_generation, **identity, selection_strategy="gmm")
    selection_directory = target_pair_selection_directory(
        project_root, **identity, selection_strategy="gmm")
    _write_examples(reference_paths, table_path=selection_directory / "selection.csv",
                    num_seeds=config["num_seeds"], validate_only=True)
    _write_examples(paths, table_path=paths.output_directory / "proximity.csv",
                    num_seeds=config["num_seeds"], validate_only=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(tokens)
    explicit_options = {token.split("=", 1)[0] for token in tokens if token.startswith("--")}
    if args.center == "zero":
        parser.error("--no-mu is retired; use the saved reference mean for mu")
    for old in ("evaluation_source", "num_baseline_seeds"):
        if getattr(args, old) is not None:
            parser.error(
                f"--{old.replace('_', '-')} is unsupported; use the direct probe measurement options"
            )
    if args.target_error_tolerance is not None and args.target_error_tolerance < 0:
        parser.error(
            "--target-error-tolerance must be nonnegative in raw latent L2 units"
        )
    if args.pdf_only and not args.plot:
        parser.error("--pdf-only requires --plot")
    if args.bundle and not (args.plot or args.validate_only):
        parser.error("--bundle requires --plot or --validate-only")
    if args.validate_proximity and (not args.validate_only or args.bundle):
        parser.error(
            "--validate-proximity requires --validate-only for a repository run"
        )
    if (
        not (args.plot or args.validate_only or args.refine_numerics)
        and args.center == "cached-baseline"
        and args.cached_baseline is None
    ):
        parser.error(
            "--center cached-baseline requires --cached-baseline with an existing independent baseline"
        )
    if args.cached_baseline and args.center != "cached-baseline" and not (args.plot or args.validate_only or args.refine_numerics):
        parser.error("--cached-baseline requires --center cached-baseline")
    if (args.source_analysis or args.source_logs) and (
        args.plot or args.validate_only or args.bundle
    ):
        parser.error(
            "Source migration is analysis-only; plotting uses saved compact paper inputs"
        )
    if args.source_logs and not args.source_analysis:
        parser.error("--source-logs requires an explicit --source-analysis")
    from utils.experiments.theory.contracts import TheoryError, numerical_config, OPT_IN_REFINEMENT_MAX_PRODUCTS
    from utils.experiments.theory.paper_contracts import (
        PaperPaths,
        render_saved_paper,
        validate_paper_bundle,
        recompute_command,
        saved_plot_configuration,
        saved_scientific_configuration,
    )

    if args.numerical_decimal_precision is not None and not (args.plot or args.validate_only):
        parser.error("--numerical-decimal-precision is unavailable for GPU computation: "
                     "the CUDA backend uses outward binary64 enclosures, not arbitrary Decimal digits. "
                     "Use --numerical-max-products and --numerical-max-variation-nodes to increase work.")

    base = dict(
        model_name=args.model, scheduler_name=args.scheduler,
        guidance_scale=args.g, num_inference_steps=args.T, num_seeds=args.N,
        center=args.center, cached_baseline=args.cached_baseline,
        target_error_tolerance=args.target_error_tolerance,
    )
    science_keys = ("mean_source", "num_mean_samples", "mean_seed", "num_loss_seeds", "loss_seed", "loss_timesteps",
                    "num_unconditional_loss_seeds", "measure_unconditional_loss",
                    "reference_law", "reference_manifest",
                    "reference_snr_decades", "terminal_noise_run_alpha",
                    "numerical_decimal_precision", "numerical_max_decimal_products",
                    "numerical_max_variation_nodes", "numerical_variation_absolute_width",
                    "counterfactual_unconditional", "counterfactual_steps")
    science = {key: getattr(args, key) for key in science_keys if getattr(args, key) is not None}
    config = dict(base)
    try:
        config = numerical_config(**base)
        bundle = args.bundle.absolute() if args.bundle else PaperPaths.build(
            PROJECT_ROOT, **config
        ).output_directory
        if args.plot or args.validate_only or args.refine_numerics:
            flag_keys = {
                "--model": "model_name", "--scheduler": "scheduler_name",
                "--g": "guidance_scale", "--T": "num_inference_steps", "--N": "num_seeds",
                "--center": "center", "--use-mu": "center", "--no-mu": "center",
                "--cached-baseline": "cached_baseline",
                "--target-error-tolerance": "target_error_tolerance",
                **{"--" + key.replace("_", "-"): key for key in science_keys},
                "--numerical-max-products": "numerical_max_decimal_products",
                "--unconditional-loss": "measure_unconditional_loss",
                "--no-unconditional-loss": "measure_unconditional_loss",
            }
            requested = {**base, **science}
            if args.refine_numerics:
                from utils.experiments.theory.contracts import NUMERICAL_KEYS
                requested = {key: value for key, value in requested.items() if key not in NUMERICAL_KEYS}
            config = saved_plot_configuration(
                bundle, requested=requested,
                explicit_keys={key for flag, key in flag_keys.items() if flag in explicit_options
                               and (not args.refine_numerics or key not in NUMERICAL_KEYS)},
                portable=bool(args.bundle),
            )
            if args.refine_numerics:
                overrides = {key: value for key, value in science.items() if key in NUMERICAL_KEYS}
                if args.numerical_max_decimal_products is None:
                    saved_budget = config["numerical_max_decimal_products"]
                    overrides["numerical_max_decimal_products"] = (
                        saved_budget if saved_budget > 0 else OPT_IN_REFINEMENT_MAX_PRODUCTS)
                config = numerical_config(**(config | overrides))
        else:
            # Preserve recorded draw counts/streams on derived recomputation.
            # Explicit options still replace them; absent optional counterfactual
            # flags do not opt a fresh measurement run into new inference.
            # Interval certification is likewise opt-in on every normal or
            # recompute invocation; a previous positive budget is not inherited.
            saved_path = bundle / "run_config.json"
            if saved_path.is_file():
                from utils.experiments.theory.contracts import read_object, FOUR_STAGE_OPTION_KEYS
                saved_configuration = read_object(saved_path)
                inherited = saved_scientific_configuration(saved_configuration)
                # Zero-centred diagnostics are historical only. New analysis uses
                # the normal reference centre; plot/validate retain saved identity.
                if inherited.get("center") == "zero":
                    inherited.pop("center")
                # New analysis upgrades the previous default first-step estimator.
                # Explicit source options and saved exact-bank choices remain valid.
                # Plot/refine modes above preserve their recorded mean source.
                if "mean_source" not in saved_configuration["scientific_config"]:
                    for key in ("mean_source", "num_mean_samples", "mean_seed"):
                        inherited.pop(key, None)
                elif args.mean_source is None and inherited.get("mean_source") == "reference-initial":
                    inherited.pop("mean_source", None)
                science = {**{key: inherited[key] for key in science_keys
                              if key in inherited and key not in FOUR_STAGE_OPTION_KEYS
                              and key != "numerical_max_decimal_products"}, **science}
                if args.reference_law == "cached-targets" and args.reference_manifest is None:
                    science.pop("reference_manifest", None)
                for key, flags in (("center", {"--center", "--use-mu", "--no-mu"}),
                                   ("cached_baseline", {"--cached-baseline"}),
                                   ("target_error_tolerance", {"--target-error-tolerance"})):
                    if not flags.intersection(explicit_options) and key in inherited:
                        base[key] = inherited[key]
                if (base["center"] != "cached-baseline"
                        and {"--center", "--use-mu", "--no-mu"}.intersection(explicit_options)
                        and "--cached-baseline" not in explicit_options):
                    base["cached_baseline"] = None
            if args.num_unconditional_loss_seeds is not None and args.measure_unconditional_loss is None:
                science["measure_unconditional_loss"] = True
            config = numerical_config(**base, **science)
        if args.estimate_mean_only:
            if config["mean_source"] not in {"reference-min-snr", "reference-initial"}:
                raise TheoryError("--estimate-mean-only requires --mean-source reference-min-snr or reference-initial")
            from utils.experiments.theory.reduce import _resolve_theory_devices
            devices = _resolve_theory_devices(args.device)
            from utils.experiments.theory.cache_reader import discover_sources
            from utils.experiments.theory.reference_law import build_reference_law
            sources = discover_sources(PROJECT_ROOT, **config)
            law = build_reference_law(
                sources, {**config, "candidate_chunk_size": args.candidate_chunk_size,
                          "query_chunk_size": args.query_chunk_size}, device=devices[0],
                allow_mean_compute=True, mean_device=args.device, mean_batch_size=args.probe_batch_size,
            )
            estimate = law.theory_mean_metadata
            print(f"[Theory] Reference mean ({config['mean_source']}, SNR={estimate['level']['snr']:.8g}): "
                  f"{estimate['sample_count']:,} samples; "
                  f"||mu||/sqrt(d)={estimate['mean_norm_rmse']:.6g}; "
                  f"Monte Carlo RMS standard error={estimate['mean_mc_standard_error_rmse']:.6g}")
            mean_file = (PaperPaths.build(PROJECT_ROOT, **config).output_directory.parent.parent
                         / "theory_measurements" / "reference_mean" / "collections"
                         / estimate["estimator_hash"] / "mean.pt")
            print(f"[Theory] Saved reference mean: {mean_file}")
            return 0
        if args.plot or args.validate_only:
            validate_paper_bundle(
                bundle,
                expected_config=config,
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
                from utils.experiments.figure_paths import publication_directory
                destination = publication_directory(PaperPaths.build(PROJECT_ROOT, **config).output_directory)
                render_saved_paper(
                    bundle, expected_config=config, diagnostics=args.diagnostics,
                    formats=("pdf",), figure_directory=destination,
                )
                print(f"Paper PDFs: {destination}")
        else:
            from utils.experiments.theory.paper_reduce import run_paper

            bundle = run_paper(
                PROJECT_ROOT,
                **config,
                recompute=args.recompute_experiments or args.refine_numerics,
                refine_numerics=args.refine_numerics,
                diagnostics=args.diagnostics,
                device=args.device,
                candidate_chunk_size=args.candidate_chunk_size,
                query_chunk_size=args.query_chunk_size,
                probe_batch_size=args.probe_batch_size,
                source_analysis=args.source_analysis,
                source_logs=args.source_logs,
            )
        print(
            f"{'Validated' if args.validate_only else 'Theory outputs'} (paper): {bundle}"
        )
        return 0
    except (TheoryError, OSError, ValueError, RuntimeError) as error:
        command = recompute_command(config)
        action = "Rebuild saved analysis with "
        if args.estimate_mean_only:
            command = command.replace("./run_all.sh", "./theory_validation.sh", 1).replace(
                " --recompute-experiments", " --estimate-mean-only", 1)
            action = "Retry mean estimation with "
        print(f"theory_validation.py: {error}\n{action}{command}", file=sys.stderr)
        return 1


# Spawned workers re-import the entry point as __mp_main__. Apply the same
# presentation-only warning policy there without importing any learned model.
if __name__ in {"__main__", "__mp_main__"}:
    _configure_dependency_warnings()

if __name__ == "__main__":
    raise SystemExit(main())
