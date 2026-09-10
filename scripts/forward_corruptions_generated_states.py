#!/usr/bin/env python3
"""Reference-fixed cached-state comparisons with reusable decoded VAE galleries."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common.cli import (  # noqa: E402
    nonnegative_integer,
    positive_integer,
    validate_seed_block,
)
from utils.common.io import (  # noqa: E402
    CacheIOError,
    atomic_torch_save,
    atomic_write_frame_csv,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
    safe_torch_load,
)
from utils.experiments.cache import (  # noqa: E402
    CompletedGenerationRecord,
    GenerationCacheError,
    generation_paths,
    list_completed_records,
)
from utils.experiments.forward_corruptions_cache import (  # noqa: E402
    load_run,
    load_scores,
    load_states,
    validate_run_pair,
)
from utils.experiments.forward_corruptions_plotting import (  # noqa: E402
    plot_saved_results,
    plot_saved_decoded_states,
    summarize_observations,
    validate_observations,
)
from utils.models.sampling import make_initial_noise  # noqa: E402
from utils.models.latent import decode_generated_latents  # noqa: E402
from utils.models.loading import (  # noqa: E402
    load_vae_from_generation_config,
    select_runtime,
)

EXPERIMENT = "forward_corruptions_generated_states"
PANELS = ("memorized", "normal")
ARTIFACTS = (
    "run_config.json",
    "observations.csv",
    "summary.csv",
    "final_samples.csv",
    "target_latents.pt",
    "interpretation.json",
)
SELECTION_POLICY = (
    "reference_only: memorized=max(min_sscd, mean_sscd); "
    "normal=min(max_sscd, mean_sscd); final ties by record_id"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=("sdv1",), default="sdv1")
    parser.add_argument("--scheduler", choices=("ddim",), default="ddim")
    parser.add_argument("--g", type=float, choices=(7.5,), default=7.5)
    parser.add_argument("--T", type=positive_integer, default=50)
    parser.add_argument("--N", type=int, choices=(20,), default=20)
    parser.add_argument("--seed-start", type=nonnegative_integer, default=0)
    parser.add_argument("--reference-seed-start", type=nonnegative_integer, default=20)
    parser.add_argument("--memorized-record-id")
    parser.add_argument("--normal-record-id")
    parser.add_argument(
        "--device", default=None, help="VAE gallery decode device (default: automatic)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plot", action="store_true", help="only reload saved logs and plot"
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute this new experiment from caches",
    )
    return parser


def requested_settings(arguments: argparse.Namespace) -> dict[str, object]:
    gallery_step_indices(arguments.T)
    validate_seed_block(arguments.seed_start, 20)
    validate_seed_block(arguments.reference_seed_start, 20)
    seeds = list(range(arguments.seed_start, arguments.seed_start + 20))
    reference = list(
        range(arguments.reference_seed_start, arguments.reference_seed_start + 20)
    )
    if set(seeds) & set(reference):
        raise ValueError("evaluation and reference seed blocks must be disjoint")
    identifiers = (arguments.memorized_record_id, arguments.normal_record_id)
    if any(identifiers) and not all(identifiers):
        raise ValueError("supply both --memorized-record-id and --normal-record-id")
    if all(identifiers) and identifiers[0] == identifiers[1]:
        raise ValueError("the two prompt-target records must be distinct")
    return {
        "schema_version": 1,
        "model_name": "sdv1",
        "scheduler_name": "ddim",
        "eta": 0.0,
        "guidance_scale": 7.5,
        "num_inference_steps": arguments.T,
        "num_seeds": 20,
        "evaluation_seeds": seeds,
        "reference_seeds": reference,
        "requested_record_ids": dict(zip(PANELS, identifiers, strict=True)),
    }


def experiment_paths(root: Path, settings: dict[str, object]) -> tuple[Path, Path]:
    evaluation_seeds = settings["evaluation_seeds"]
    reference_seeds = settings["reference_seeds"]
    cache = generation_paths(
        root,
        model_name="sdv1",
        scheduler_name="ddim",
        guidance_scale=7.5,
        num_inference_steps=settings["num_inference_steps"],
        num_seeds=20,
        seed_start=evaluation_seeds[0],
    )
    reference_namespace = f"reference_S{reference_seeds[0]}_N20"
    identifiers = settings["requested_record_ids"]
    mode = (
        "reference_ranked"
        if not any(identifiers.values())
        else "pairs_" + canonical_hash(identifiers)[:16]
    )
    suffix = Path(reference_namespace) / mode
    logs = cache.run_directory / EXPERIMENT / suffix
    output = (
        root
        / "outputs"
        / cache.run_directory.parent.name
        / EXPERIMENT
        / cache.run_directory.name
        / suffix
    )
    return logs, output


def rank_reference_pairs(
    candidates: Sequence[dict[str, object]],
    requested: dict[str, str | None],
) -> dict[str, dict[str, object]]:
    """Choose before loading any evaluation score, latent, or availability flag."""
    if len(candidates) < 2:
        raise ValueError("need at least two complete independent reference records")
    by_id = {row["record_id"]: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("reference record identifiers are duplicated")
    if any(requested.values()):
        if not all(requested.values()):
            raise ValueError("both reference record identifiers are required")
        missing = set(requested.values()) - set(by_id)
        if missing:
            raise ValueError(
                f"requested records lack complete reference evidence: {sorted(missing)}"
            )
        selected = {panel: by_id[requested[panel]] for panel in PANELS}
    else:
        selected = {
            "memorized": min(
                candidates,
                key=lambda row: (-row["sscd_min"], -row["sscd_mean"], row["record_id"]),
            ),
            "normal": min(
                candidates,
                key=lambda row: (row["sscd_max"], row["sscd_mean"], row["record_id"]),
            ),
        }
    if selected["memorized"]["record_id"] == selected["normal"]["record_id"]:
        raise ValueError("reference rankings do not identify two distinct examples")
    return selected


def measure_states(
    *,
    panel: str,
    record_id: str,
    original_index: str,
    seeds: Sequence[int],
    timesteps: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    states: torch.Tensor,
    progress: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """Raw L2 to a fixed target, never a paired generated-minus-forward distance."""
    n_steps = len(timesteps)
    shape = tuple(target.shape)
    if len(shape) != 3 or tuple(states.shape) != (len(seeds), n_steps + 1, *shape):
        raise ValueError(
            "states must include every pre-step input and the final output"
        )
    if alpha.shape != timesteps.shape or sigma.shape != timesteps.shape:
        raise ValueError("scheduler coefficients must align with actual timesteps")
    if not torch.isfinite(states).all() or not torch.isfinite(target).all():
        raise ValueError(
            "non-finite states or target; no evaluation seeds may be dropped"
        )
    noise = make_initial_noise(seeds, shape)
    if not torch.equal(states[:, 0], noise.to(dtype=states.dtype)):
        raise ValueError(
            "generation must start from the seed Gaussian, not a corrupted target"
        )
    fixed = target.double()
    epsilon = noise.double()
    rows = []
    for step in range(n_steps + 1):
        final = step == n_steps
        a = 1.0 if final else float(alpha[step])
        b = 0.0 if final else float(sigma[step])
        forward_delta = (a - 1.0) * fixed + b * epsilon
        forward = forward_delta.flatten(1).norm(dim=1)
        generated = (states[:, step].double() - fixed).flatten(1).norm(dim=1)
        for position, seed in enumerate(seeds):
            rows.append(
                {
                    "panel": panel,
                    "record_id": record_id,
                    "original_index": original_index,
                    "generation_seed": seed,
                    "step_index": step,
                    "timestep": -1 if final else int(timesteps[step]),
                    "is_final_clean_state": final,
                    "denoising_progress": step / n_steps,
                    "alpha_t": a,
                    "sigma_t": b,
                    "forward_distance": float(forward[position]),
                    "generated_distance": float(generated[position]),
                }
            )
        if progress:
            progress(len(seeds))
    return pd.DataFrame(rows)


def summarize_interpretation(summary: pd.DataFrame) -> dict[str, object]:
    reports = {}
    for panel in PANELS:
        steps = summary.loc[summary["panel"].eq(panel)].sort_values("step_index")
        interior = steps.iloc[1:-1]
        overlap = (interior["forward_max"] >= interior["generated_min"]) & (
            interior["generated_max"] >= interior["forward_min"]
        )
        reports[panel] = {
            "intermediate_steps": len(interior),
            "overlapping_observed_ranges": int(overlap.sum()),
            "overlapping_step_indices": interior.loc[overlap, "step_index"]
            .astype(int)
            .tolist(),
            "nonoverlapping_step_indices": interior.loc[~overlap, "step_index"]
            .astype(int)
            .tolist(),
            "result": (
                f"Observed seed ranges overlap at {int(overlap.sum())} of "
                f"{len(interior)} intermediate denoising steps."
            ),
        }
    return {
        "panels": reports,
        "bands": "Observed minimum-to-maximum across all 20 seeds, not confidence intervals.",
        "interpretation": (
            "This compares two distance-to-fixed-target ensembles at matched timesteps. "
            "It is an empirical illustration, not a formal distributional test. "
            "Overlap of this one statistic cannot establish equality of full latent-state "
            "distributions; observed non-overlap is not a formal distributional rejection. "
            "No paired x_t-minus-y_t distance is used. Normal means low reproduction "
            "of this paired target in independent reference results, not all training images."
        ),
    }


def _validate_completed(logs: Path, settings: dict[str, object]) -> None:
    completion = read_json(logs / "completion.json")
    if completion.get("settings_hash") != canonical_hash(settings):
        raise ValueError("saved experiment settings differ")
    if set(completion.get("artifacts", {})) != set(ARTIFACTS):
        raise ValueError("saved experiment is incomplete; rerun with --overwrite")
    for name, digest in completion["artifacts"].items():
        path = logs / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"saved artifact is missing or changed: {path}")


def gallery_step_indices(num_inference_steps: int) -> list[int]:
    """Ten fixed snapshots spanning initialization through the final output."""
    if (
        isinstance(num_inference_steps, bool)
        or not isinstance(num_inference_steps, int)
        or num_inference_steps < 9
    ):
        raise ValueError("ten distinct gallery states require --T >= 9")
    return np.rint(np.linspace(0, num_inference_steps, 10)).astype(int).tolist()


def gallery_latents(
    target: torch.Tensor,
    states: torch.Tensor,
    *,
    seed: int,
    step_indices: Sequence[int],
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recreate forward inputs and select cached reverse inputs, never xhat_0."""
    steps = len(alpha)
    if (
        target.ndim != 3
        or tuple(states.shape) != (steps + 1, *target.shape)
        or alpha.shape != sigma.shape
        or alpha.ndim != 1
    ):
        raise ValueError("gallery states and scheduler coefficients do not align")
    if list(step_indices) != gallery_step_indices(steps):
        raise ValueError("gallery snapshots must use the ten fixed matched steps")
    if not all(torch.isfinite(value).all() for value in (target, states, alpha, sigma)):
        raise ValueError("gallery inputs must be finite")
    noise = make_initial_noise([seed], tuple(target.shape))[0]
    if not torch.equal(states[0], noise.to(dtype=states.dtype)):
        raise ValueError("gallery initial state differs from its fixed seed Gaussian")
    forward = torch.stack(
        [
            target.double()
            if step == steps
            else alpha[step].double() * target.double()
            + sigma[step].double() * noise.double()
            for step in step_indices
        ]
    ).float()
    generated = states[list(step_indices)].float()
    return forward, generated


def ensure_decoded_states(
    root: Path,
    logs: Path,
    *,
    plot_only: bool = False,
    overwrite: bool = False,
    device: str | None = None,
) -> Path:
    """Decode once, retaining reusable RGB previews independently of measurements."""
    cache_path = logs / "decoded_states.pt"
    marker_path = logs / "decoded_states.json"
    for path in (cache_path, marker_path):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"unsafe decoded gallery artifact: {path}")
    completion_hash = file_sha256(logs / "completion.json")
    if cache_path.is_file() and marker_path.is_file() and not overwrite:
        marker = read_json(marker_path)
        if (
            marker.get("schema_version") != 1
            or marker.get("measurement_completion_sha256") != completion_hash
            or marker.get("payload_sha256") != file_sha256(cache_path)
        ):
            raise ValueError("decoded gallery cache differs; rerun with --overwrite")
        return cache_path
    if plot_only:
        raise ValueError(
            "decoded gallery logs are missing; run without --plot once to decode "
            "the cached states (no trajectory generation)"
        )
    configuration = read_json(logs / "run_config.json")
    settings = configuration["settings"]
    indices = gallery_step_indices(settings["num_inference_steps"])
    seed = settings["evaluation_seeds"][0]
    evaluation = load_run(
        root,
        num_inference_steps=settings["num_inference_steps"],
        seed_start=seed,
    )
    if (
        canonical_hash(evaluation.configuration)
        != canonical_hash(configuration["evaluation_generation_configuration"])
        or file_sha256(evaluation.paths.schedule)
        != configuration["evaluation_schedule_sha256"]
    ):
        raise ValueError("generation configuration or schedule changed since measurements")
    fixed_targets = safe_torch_load(logs / "target_latents.pt")
    batches = {}
    panels = {}
    for panel in PANELS:
        pair = configuration["pairs"][panel]
        expected_marker = pair["evaluation_generation_marker"]
        record = CompletedGenerationRecord(
            pair["original_index"],
            pair["source_row_number"],
            evaluation.paths.record_path(pair["original_index"]),
            expected_marker,
        )
        states, target, marker = load_states(evaluation, record)
        if (
            canonical_hash(marker) != canonical_hash(expected_marker)
            or not torch.equal(target, fixed_targets[panel])
        ):
            raise ValueError(f"{panel} generation states or fixed target changed")
        forward, generated = gallery_latents(
            target,
            states[evaluation.seeds.index(seed)],
            seed=seed,
            step_indices=indices,
            alpha=evaluation.alpha,
            sigma=evaluation.sigma,
        )
        batches[panel] = torch.cat((forward, generated))
        panels[panel] = {
            "record_id": pair["record_id"],
            "original_index": pair["original_index"],
            "source_latent_sha256": marker["tensor_file_sha256"]["latent"],
        }
        del states
    # Only the pinned VAE is loaded. No UNet, sampling, or SSCD evaluation.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    loaded = load_vae_from_generation_config(
        evaluation.configuration,
        runtime=select_runtime(device, warn_without_cuda=False),
    )
    try:
        with tqdm(total=40, desc="Decoded forward / generated states", unit="image") as bar:
            for panel in PANELS:
                images = decode_generated_latents(
                    batches[panel],
                    loaded.vae,
                    loaded.device,
                    microbatch_size=4,
                    progress_callback=bar.update,
                )
                if len(images) != 20:
                    raise ValueError("decoder did not return all twenty paired snapshots")
                rgb = torch.stack(
                    [
                        torch.from_numpy(
                            np.array(
                                image.resize((256, 256), Image.Resampling.LANCZOS),
                                dtype=np.uint8,
                                copy=True,
                            )
                        )
                        for image in images
                    ]
                )
                if tuple(rgb.shape) != (20, 256, 256, 3):
                    raise ValueError("gallery decoder must produce RGB images")
                panels[panel]["forward_images"] = rgb[:10].contiguous()
                panels[panel]["generated_images"] = rgb[10:].contiguous()
    finally:
        del loaded
    payload = {
        "schema_version": 1,
        "measurement_completion_sha256": completion_hash,
        "num_inference_steps": settings["num_inference_steps"],
        "generation_seed": seed,
        "seed_policy": "first evaluation seed, identical for both prompts; no outcome filtering",
        "step_indices": indices,
        "display_timesteps": [settings["num_inference_steps"] - step for step in indices],
        "scheduler_timesteps": [
            -1 if step == len(evaluation.timesteps) else int(evaluation.timesteps[step])
            for step in indices
        ],
        "alpha_t": [
            1.0 if step == len(evaluation.alpha) else float(evaluation.alpha[step])
            for step in indices
        ],
        "sigma_t": [
            0.0 if step == len(evaluation.sigma) else float(evaluation.sigma[step])
            for step in indices
        ],
        "vae": {
            key: evaluation.configuration["scientific_config"][key]
            for key in ("model_id", "model_revision", "vae_id", "vae_revision")
        },
        "decode_dtype": "float32",
        "preview_resolution": 256,
        "preview_resize": "PIL Lanczos after standard RGB decode",
        "difference": "absolute RGB pixel difference of displayed uint8 previews; fixed 0..255 scale, no contrast normalization",
        "interpretation": "paired-state visualization only; not evidence of distributional mismatch",
        "panels": panels,
    }
    if file_sha256(logs / "completion.json") != completion_hash:
        raise ValueError("measurement logs changed while decoding galleries")
    marker_path.unlink(missing_ok=True)
    atomic_torch_save(payload, cache_path)
    atomic_write_json(
        marker_path,
        {
            "schema_version": 1,
            "measurement_completion_sha256": completion_hash,
            "payload_sha256": file_sha256(cache_path),
        },
    )
    return cache_path


def _plot_experiment(
    root: Path, logs: Path, output: Path, arguments: argparse.Namespace
) -> None:
    plot_saved_results(logs / "observations.csv", output)
    decoded_path = ensure_decoded_states(
        root,
        logs,
        plot_only=arguments.plot,
        overwrite=arguments.overwrite,
        device=arguments.device,
    )
    plot_saved_decoded_states(decoded_path, output)


def run_experiment(root: Path, arguments: argparse.Namespace) -> tuple[Path, Path]:
    settings = requested_settings(arguments)
    logs, output = experiment_paths(root, settings)
    if arguments.plot or (
        (logs / "completion.json").exists() and not arguments.overwrite
    ):
        _validate_completed(logs, settings)
        _plot_experiment(root, logs, output, arguments)
        return logs, output
    torch.set_num_threads(min(4, torch.get_num_threads()))
    reference = load_run(
        root, num_inference_steps=arguments.T, seed_start=arguments.reference_seed_start
    )
    evaluation = load_run(
        root, num_inference_steps=arguments.T, seed_start=arguments.seed_start
    )
    validate_run_pair(reference, evaluation)
    records = list_completed_records(reference.paths)
    by_index = {record.original_index: record for record in records}
    candidates = []
    invalid_reference = []
    frames = []
    targets = {}
    final_rows = []
    pair_metadata = {}
    total = len(records) + 2 * 20 * (arguments.T + 1)
    with tqdm(
        total=total, desc="Forward corruptions / generated states", unit="item"
    ) as bar:
        for record in records:
            try:
                scores = load_scores(reference, record).double()
                candidates.append(
                    {
                        "record_id": record.metadata["record_id"],
                        "original_index": record.original_index,
                        "sscd_min": float(scores.min()),
                        "sscd_max": float(scores.max()),
                        "sscd_mean": float(scores.mean()),
                        "reference_scores": scores.tolist(),
                    }
                )
            except (
                CacheIOError,
                GenerationCacheError,
                ValueError,
                RuntimeError,
                OSError,
            ) as error:
                invalid_reference.append(
                    {"record_id": record.metadata.get("record_id"), "error": str(error)}
                )
            bar.update(1)
        chosen = rank_reference_pairs(candidates, settings["requested_record_ids"])
        # Missing evaluation data aborts: never substitute another pair or drop seeds.
        for panel in PANELS:
            selected = chosen[panel]
            record = by_index[selected["original_index"]]
            states, target, marker = load_states(evaluation, record)
            evaluation_record = CompletedGenerationRecord(
                record.original_index,
                record.source_row_number,
                evaluation.paths.record_path(record.original_index),
                marker,
            )
            scores = load_scores(evaluation, evaluation_record)
            observed = measure_states(
                panel=panel,
                record_id=selected["record_id"],
                original_index=record.original_index,
                seeds=evaluation.seeds,
                timesteps=evaluation.timesteps,
                alpha=evaluation.alpha,
                sigma=evaluation.sigma,
                target=target,
                states=states,
                progress=bar.update,
            )
            frames.append(observed)
            targets[panel] = target.detach().cpu().clone()
            final = observed.loc[observed["is_final_clean_state"]]
            for position, (_, row) in enumerate(final.iterrows()):
                final_rows.append(
                    {
                        "panel": panel,
                        "record_id": selected["record_id"],
                        "original_index": record.original_index,
                        "generation_seed": int(row["generation_seed"]),
                        "target_sscd": float(scores[position]),
                        "generated_distance": float(row["generated_distance"]),
                        "target_image_sha256": marker["target_image_sha256"],
                        "status": "complete",
                        "error": "",
                    }
                )
            pair_metadata[panel] = {
                **selected,
                "prompt": marker["prompt_raw"],
                "source_row_number": record.source_row_number,
                "target_image_path": marker["target_image_path"],
                "target_image_sha256": marker["target_image_sha256"],
                "reference_seeds": list(reference.seeds),
                "reference_generation_marker": record.metadata,
                "evaluation_generation_marker": marker,
                "reference_sscd_marker_sha256": file_sha256(
                    reference.sscd_paths.marker_path(record.original_index)
                ),
                "evaluation_sscd_marker_sha256": file_sha256(
                    evaluation.sscd_paths.marker_path(record.original_index)
                ),
            }
            del states
    observations = validate_observations(pd.concat(frames, ignore_index=True))
    summary = summarize_observations(observations)
    interpretation = summarize_interpretation(summary)
    configuration = {
        "settings": settings,
        "selection_policy": SELECTION_POLICY
        if not arguments.memorized_record_id
        else "explicit_reference_backed_record_ids",
        "num_valid_reference_candidates": len(candidates),
        "invalid_reference_records": invalid_reference,
        "pairs": pair_metadata,
        "reference_generation_configuration": reference.configuration,
        "evaluation_generation_configuration": evaluation.configuration,
        "reference_schedule_sha256": file_sha256(reference.paths.schedule),
        "evaluation_schedule_sha256": file_sha256(evaluation.paths.schedule),
        "reference_sscd_configuration": reference.sscd_configuration,
        "evaluation_sscd_configuration": evaluation.sscd_configuration,
        "distance_definition": "raw Euclidean latent distance to one fixed cached deterministic VAE target; no dimension normalization",
        "noise_definition": "make_initial_noise: CPU float32 standard Gaussian reused across timesteps and prompts; cache initial state verified after storage-dtype casting",
        "final_endpoint": {
            "timestep": -1,
            "alpha_t": 1.0,
            "sigma_t": 0.0,
            "meaning": "explicit ideal forward-clean boundary y=x_star compared with the actual cached final post-update generated output",
            "not_a_denoiser_evaluation": True,
            "scheduler_final_alpha_cumprod": evaluation.scheduler_final_alpha_cumprod,
            "caveat": "The saved scheduler can have set_alpha_to_one=False; endpoint alpha=1,sigma=0 define the forward boundary, not an additional DDIM update.",
        },
        "evaluation_seed_filtering": False,
        "all_evaluation_samples_required": True,
    }
    logs.mkdir(parents=True, exist_ok=True)
    for name in (*ARTIFACTS, "completion.json"):
        destination = logs / name
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise ValueError(f"unsafe output artifact: {destination}")
    (logs / "completion.json").unlink(missing_ok=True)
    atomic_write_json(logs / "run_config.json", configuration)
    atomic_write_frame_csv(observations, logs / "observations.csv")
    atomic_write_frame_csv(summary, logs / "summary.csv")
    atomic_write_frame_csv(pd.DataFrame(final_rows), logs / "final_samples.csv")
    atomic_torch_save(targets, logs / "target_latents.pt")
    atomic_write_json(logs / "interpretation.json", interpretation)
    atomic_write_json(
        logs / "completion.json",
        {
            "settings_hash": canonical_hash(settings),
            "artifacts": {name: file_sha256(logs / name) for name in ARTIFACTS},
        },
    )
    _plot_experiment(root, logs, output, arguments)
    for panel, report in interpretation["panels"].items():
        print(f"{panel}: {report['result']}")
    return logs, output


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        logs, output = run_experiment(ROOT, arguments)
    except (
        CacheIOError,
        GenerationCacheError,
        ValueError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Logs: {logs}")
    print(f"Figures (PDF/PNG): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
