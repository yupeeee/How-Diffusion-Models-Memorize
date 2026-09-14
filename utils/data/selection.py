"""Freeze whole-prompt GMM selection from terminal-L2/SSCD reference evidence.

Selection uses all reference seeds N--2N-1, where N is the experiment seed
count. The full-covariance GMM fits every valid reference observation and
assigns each to exactly one of two components. A complete prompt is discarded
only if strictly more than half its reference seeds belong to the low-SSCD
component and none has SSCD strictly above 0.75. An exact half/half split is
kept. Posteriors are not averaged into a second selection gate. Webster kind
and within-prompt Spearman correlation are descriptive audit metadata and
never affect the selection.
"""

from __future__ import annotations

import math
import operator
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.common.cli import (
    generation_cache_namespace,
    generation_cache_parent_name,
    generation_run_name,
    stable_float,
    validate_seed_block,
)
from utils.common.io import (
    atomic_write_frame_csv,
    atomic_write_json,
    canonical_hash,
    file_sha256,
    read_json,
)
from utils.data.proximity_gmm import (
    COMPONENT_NAMES,
    DEFAULT_REG_COVAR,
    INITIALIZATION_NAME,
    LIKELIHOOD_DECREASE_TOLERANCE,
    MAX_ITERATIONS,
    TOLERANCE,
    expectation,
    fit_gaussian_mixture,
    standardize_features,
)
from utils.data.webster import normalize_webster_type
from utils.experiments.cache import generation_log_relative_path

DEFAULT_SELECTION_STRATEGY = "gmm"
SELECTION_STRATEGIES = ("gmm",)
HIGH_SSCD_RETENTION_THRESHOLD = 0.75
SELECTION_POLICIES = {
    "gmm": "prompt_gmm_low_component_majority_sscd_override_full_covariance",
}
INCLUDED_PROXIMITY_RULE = "included_proximity_rule"
DISCARDED_PROXIMITY_RULE = "discarded_proximity_rule"
UNUSABLE_REFERENCE_OBSERVATIONS = "unusable_reference_observations"

SELECTION_COLUMNS = tuple(
    "model_name original_index record_id source_row_number seed prompt kind "
    "target_image_sha256 generated_image_path generated_image_tile_index "
    "l2_norm sscd observation_status observation_error selection_strategy "
    "prompt_spearman gmm_component gmm_low_mode_probability "
    "include_prompt "
    "selection_status selection_reason reference_run_name "
    "reference_generation_hash reference_sscd_hash selection_policy selection_hash".split()
)
_IDENTITY_COLUMNS = tuple(
    "original_index record_id source_row_number prompt kind target_image_sha256".split()
)
_MODELS = {"sdv1": "sdv1", "sdv2": "sdv2", "realvis": "realisticvision"}
_HASH = re.compile(r"[0-9a-f]{64}")
_FILES = ("selection.csv", "config.json", "summary.json")
_REFERENCE_MARKER_DIRECTORIES = (
    ("generation", "record"),
    ("sscd", "sscd_record"),
)


class TargetPairSelectionError(RuntimeError):
    """Selection evidence or a frozen selection is invalid."""


class TargetPairSelectionMissingError(TargetPairSelectionError):
    """The required frozen selection does not exist."""


class FrozenTargetPairSelectionError(TargetPairSelectionError):
    """New evidence conflicts with an existing frozen selection."""


@dataclass(frozen=True, slots=True)
class TargetPairSelection:
    """A validated seed-level selection audit."""

    root: Path
    model_name: str
    scheduler_name: str
    guidance_scale: float
    num_inference_steps: int
    num_seeds: int
    selection_strategy: str
    frame: pd.DataFrame
    configuration: Mapping[str, object]
    sha256: str

    @property
    def prompt_frame(self) -> pd.DataFrame:
        return (
            self.frame.drop_duplicates("original_index").reset_index(drop=True).copy()
        )

    @property
    def included_indices(self) -> frozenset[str]:
        rows = self.prompt_frame
        return frozenset(rows.loc[rows["include_prompt"], "original_index"].astype(str))

    @property
    def excluded_indices(self) -> frozenset[str]:
        rows = self.prompt_frame
        return frozenset(
            rows.loc[~rows["include_prompt"], "original_index"].astype(str)
        )


FrameInput = pd.DataFrame | str | Path
ConfigurationInput = Mapping[str, object] | str | Path


def normalize_selection_strategy(value: object) -> str:
    """Return one supported prompt-selection strategy."""

    if not isinstance(value, str) or value not in SELECTION_STRATEGIES:
        raise TargetPairSelectionError(
            "selection_strategy must be one of: " + ", ".join(SELECTION_STRATEGIES)
        )
    return value


def selection_policy(selection_strategy: object) -> str:
    """Return the immutable policy identifier for a strategy."""

    return SELECTION_POLICIES[normalize_selection_strategy(selection_strategy)]


def is_reference_configuration(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
) -> bool:
    """Return whether CLI values identify the N-seed reference run."""

    try:
        _, _, _, _, count = _reference_values(
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
        )
        first_seed = operator.index(seed_start)
    except (TargetPairSelectionError, TypeError, ValueError, OverflowError):
        return False
    return not isinstance(seed_start, bool) and first_seed == count


def reference_run_name(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> str:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    return generation_run_name(
        model,
        scheduler,
        guidance,
        steps,
        count,
        seed_start=count,
    )


def reference_run_path(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
) -> Path:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    return generation_log_relative_path(
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        seed_start=count,
    )


def reference_selection_command(
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
) -> str:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    common = (
        f"--model {model} --scheduler {scheduler} --g {stable_float(guidance)} "
        f"--T {steps} "
        f"--N {count} --seed-start {count}"
    )
    strategy = normalize_selection_strategy(selection_strategy)
    return (
        f"./generate.sh {common}\n./sscd.sh {common}\n"
        f"./compute_proximity.sh {common} --selection-strategy {strategy}"
    )


def target_pair_selection_directory(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
) -> Path:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    parent = generation_cache_parent_name(model, scheduler, guidance, steps, count)
    namespace = generation_cache_namespace(
        model, scheduler, guidance, steps, count, seed_start=count
    )
    normalize_selection_strategy(selection_strategy)
    return (
        Path(root).expanduser().resolve()
        / "data/webster/selection"
        / _MODELS[model]
        / parent
        / namespace
    )


def reference_completion_fingerprint(
    reference_run: str | Path,
) -> dict[str, object]:
    """Fingerprint the exact generation and SSCD completion-marker JSON files."""

    run = Path(reference_run).expanduser().resolve()
    return {
        label: _completion_marker_directory_fingerprint(run / relative, label)
        for label, relative in _REFERENCE_MARKER_DIRECTORIES
    }


def build_target_pair_selection(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    overwrite: bool,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    paired_frame: FrameInput,
    reference_run_config: ConfigurationInput | None = None,
    sscd_config: ConfigurationInput | None = None,
    records_frame: FrameInput | None = None,
) -> TargetPairSelection:
    """Build and atomically freeze the seed-level audit."""

    project_root = Path(root).expanduser().resolve()
    if not isinstance(overwrite, bool):
        raise TargetPairSelectionError("overwrite must be a boolean")
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    strategy = normalize_selection_strategy(selection_strategy)
    run = project_root / reference_run_path(model, scheduler, guidance, steps, count)
    generation = _read_config(
        reference_run_config, run / "run_config.json", "generation config"
    )
    sscd = _read_config(sscd_config, run / "sscd_config.json", "SSCD config")
    generation_hash = _validate_generation(
        generation, model, scheduler, guidance, steps, count
    )
    sscd_hash = _validate_sscd(sscd, generation_hash, count)
    completion_fingerprint = reference_completion_fingerprint(run)

    paired = _read_frame(paired_frame, "paired reference observations")
    source = paired if records_frame is None else _read_frame(records_frame, "records")
    records = _prompt_records(source, derived=records_frame is None, model=model)
    observations = _observations(
        paired, set(records["original_index"]), num_seeds=count
    )
    manifest, gmm_fit = _manifest(
        records,
        observations,
        model,
        scheduler,
        guidance,
        steps,
        generation_hash,
        sscd_hash,
        count,
        strategy,
    )
    digest = compute_target_pair_selection_hash(
        manifest,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        reference_generation_hash=generation_hash,
        reference_sscd_hash=sscd_hash,
        reference_completion_fingerprint=completion_fingerprint,
        selection_strategy=strategy,
        gmm_fit=gmm_fit,
    )
    manifest["selection_hash"] = digest
    config = _selection_config(
        model,
        scheduler,
        guidance,
        steps,
        count,
        generation_hash,
        sscd_hash,
        sscd["sscd_checkpoint_sha256"],
        sscd["sscd_preprocessing_hash"],
        completion_fingerprint,
        digest,
        strategy,
        gmm_fit,
    )
    selection = TargetPairSelection(
        root=project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        selection_strategy=strategy,
        frame=manifest,
        configuration=config,
        sha256=digest,
    )
    validate_target_pair_selection(selection)

    destination = target_pair_selection_directory(
        project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        selection_strategy=strategy,
    )
    if (destination.exists() or destination.is_symlink()) and not overwrite:
        frozen = load_target_pair_selection(
            project_root,
            model_name=model,
            scheduler_name=scheduler,
            guidance_scale=guidance,
            num_inference_steps=steps,
            num_seeds=count,
            selection_strategy=strategy,
        )
        if frozen.sha256 != digest:
            raise FrozenTargetPairSelectionError(
                f"Reference evidence would change frozen selection {destination}: "
                f"{frozen.sha256} != {digest}"
            )
        return frozen
    _write_selection(selection, destination, overwrite=overwrite)
    return load_target_pair_selection(
        project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        selection_strategy=strategy,
    )


def load_target_pair_selection(
    root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
) -> TargetPairSelection:
    """Load and fully validate the three frozen artifacts."""

    project_root = Path(root).expanduser().resolve()
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    strategy = normalize_selection_strategy(selection_strategy)
    directory = target_pair_selection_directory(
        project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        selection_strategy=strategy,
    )
    if not directory.exists():
        _raise_missing(project_root, model, scheduler, guidance, steps, count, strategy)
    if not directory.is_dir() or directory.is_symlink():
        raise TargetPairSelectionError(f"Invalid selection directory: {directory}")
    missing = [name for name in _FILES if not (directory / name).is_file()]
    if missing:
        raise TargetPairSelectionError(
            f"Incomplete frozen selection {directory}; missing: {', '.join(missing)}"
        )
    if any((directory / name).is_symlink() for name in _FILES):
        raise TargetPairSelectionError(
            "Frozen selection artifacts must not be symlinks"
        )
    try:
        frame = pd.read_csv(
            directory / "selection.csv",
            dtype={
                "original_index": str,
                "record_id": str,
                "prompt": str,
                "kind": str,
                "target_image_sha256": str,
                "generated_image_path": str,
            },
            keep_default_na=False,
            float_precision="round_trip",
        )
        config = read_json(directory / "config.json")
        summary = read_json(directory / "summary.json")
    except (OSError, RuntimeError, ValueError) as error:
        raise TargetPairSelectionError(
            f"Cannot read selection {directory}: {error}"
        ) from error
    frame = _normalize_csv(frame)
    digest = config.get("selection_hash")
    if not _is_hash(digest):
        raise TargetPairSelectionError("config.json selection_hash is invalid")
    selection = TargetPairSelection(
        root=project_root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        selection_strategy=strategy,
        frame=frame,
        configuration=config,
        sha256=str(digest),
    )
    validate_target_pair_selection(selection)
    if summary != _summary(selection):
        raise TargetPairSelectionError("Frozen selection summary is inconsistent")
    return selection


def validate_target_pair_selection(selection: TargetPairSelection) -> None:
    """Validate fixed provenance, seed groups, decisions, and content hash."""

    model, scheduler, guidance, steps, count = _reference_values(
        selection.model_name,
        selection.scheduler_name,
        selection.guidance_scale,
        selection.num_inference_steps,
        selection.num_seeds,
    )
    strategy = normalize_selection_strategy(selection.selection_strategy)
    frame, config = selection.frame, selection.configuration
    if tuple(frame.columns) != SELECTION_COLUMNS or frame.empty:
        raise TargetPairSelectionError("selection.csv has an invalid schema")
    if not pd.api.types.is_bool_dtype(frame["include_prompt"]):
        raise TargetPairSelectionError("include_prompt must contain booleans")
    gmm_fit = config.get("gmm_fit")
    completion_fingerprint = _normalize_completion_fingerprint(
        config.get("reference_completion_fingerprint")
    )
    expected = _selection_config(
        model,
        scheduler,
        guidance,
        steps,
        count,
        str(config.get("reference_generation_hash")),
        str(config.get("reference_sscd_hash")),
        config.get("sscd_checkpoint_sha256"),
        config.get("sscd_preprocessing_hash"),
        completion_fingerprint,
        selection.sha256,
        strategy,
        gmm_fit if isinstance(gmm_fit, Mapping) else None,
    )
    if dict(config) != expected:
        raise TargetPairSelectionError(
            "config.json does not match the selection contract"
        )
    if any(
        not _is_hash(config[key])
        for key in (
            "reference_generation_hash",
            "reference_sscd_hash",
            "sscd_checkpoint_sha256",
            "sscd_preprocessing_hash",
            "selection_hash",
        )
    ):
        raise TargetPairSelectionError("Selection provenance hash is invalid")
    fixed = {
        "model_name": model,
        "reference_run_name": reference_run_name(
            model, scheduler, guidance, steps, count
        ),
        "reference_generation_hash": config["reference_generation_hash"],
        "reference_sscd_hash": config["reference_sscd_hash"],
        "selection_strategy": strategy,
        "selection_policy": selection_policy(strategy),
        "selection_hash": selection.sha256,
    }
    if any(not frame[column].eq(value).all() for column, value in fixed.items()):
        raise TargetPairSelectionError("Selection row provenance is inconsistent")
    expected_order = frame.sort_values(
        ["source_row_number", "original_index", "seed"], kind="stable"
    ).reset_index(drop=True)
    if not frame.reset_index(drop=True).equals(expected_order):
        raise TargetPairSelectionError("selection.csv rows are not canonically ordered")
    for index, group in frame.groupby("original_index", sort=False):
        _validate_prompt_group(
            str(index), group, model, scheduler, guidance, steps, count, strategy
        )
    assert isinstance(gmm_fit, Mapping)
    _validate_gmm_decisions(frame, gmm_fit, strategy)
    digest = compute_target_pair_selection_hash(
        frame,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        reference_generation_hash=str(config["reference_generation_hash"]),
        reference_sscd_hash=str(config["reference_sscd_hash"]),
        reference_completion_fingerprint=completion_fingerprint,
        selection_strategy=strategy,
        gmm_fit=gmm_fit if isinstance(gmm_fit, Mapping) else None,
    )
    if digest != selection.sha256:
        raise TargetPairSelectionError(
            f"Selection hash mismatch: {selection.sha256} != {digest}"
        )


def apply_target_pair_selection(
    frame: pd.DataFrame, selection: TargetPairSelection
) -> pd.DataFrame:
    """Keep every input row belonging to an included prompt."""

    if "original_index" not in frame:
        raise TargetPairSelectionError("Table is missing original_index")
    result = frame.copy(deep=True)
    indices = result["original_index"].map(_index)
    known = set(selection.prompt_frame["original_index"])
    unknown = sorted(set(indices) - known)
    if unknown:
        raise TargetPairSelectionError(
            "Prompts absent from selection: " + ", ".join(unknown[:5])
        )
    return result.loc[indices.isin(selection.included_indices)].reset_index(drop=True)


def compute_target_pair_selection_hash(
    frame: pd.DataFrame,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    reference_generation_hash: str,
    reference_sscd_hash: str,
    reference_completion_fingerprint: Mapping[str, object],
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    gmm_fit: Mapping[str, object] | None = None,
) -> str:
    """Hash every seed observation and deterministic prompt decision."""

    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    reference_seeds = _reference_seeds(count)
    strategy = normalize_selection_strategy(selection_strategy)
    contract = _strategy_contract(strategy)
    completion_fingerprint = _normalize_completion_fingerprint(
        reference_completion_fingerprint
    )
    if not isinstance(gmm_fit, Mapping):
        raise TargetPairSelectionError("GMM selection fit is missing")
    if not _is_hash(reference_generation_hash) or not _is_hash(reference_sscd_hash):
        raise TargetPairSelectionError("Selection provenance hash is invalid")
    columns = tuple(
        column for column in SELECTION_COLUMNS if column != "selection_hash"
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise TargetPairSelectionError("Cannot hash without: " + ", ".join(missing))
    rows = [
        {column: _json(row[column]) for column in columns}
        for row in frame.sort_values(
            ["source_row_number", "original_index", "seed"], kind="stable"
        )
        .loc[:, columns]
        .to_dict(orient="records")
    ]
    return canonical_hash(
        {
            "selection_strategy": strategy,
            "selection_policy": selection_policy(strategy),
            "model_name": model,
            "dataset_model_name": _MODELS[model],
            "reference_scheduler": scheduler,
            "reference_guidance_scale": guidance,
            "reference_num_inference_steps": steps,
            "reference_run_name": reference_run_name(
                model, scheduler, guidance, steps, count
            ),
            "reference_generation_hash": reference_generation_hash,
            "reference_sscd_hash": reference_sscd_hash,
            "reference_completion_fingerprint": completion_fingerprint,
            "reference_seeds": list(reference_seeds),
            **contract,
            "gmm_fit": _json(gmm_fit),
            "rows": rows,
        }
    )


def _strategy_contract(selection_strategy: str) -> dict[str, object]:
    normalize_selection_strategy(selection_strategy)
    return {
        "selection_metric": "two_component_full_covariance_gmm(l2_norm,sscd)",
        "fit_population": "all_complete_reference_observations",
        "prompt_reduction": "count(gmm_component == low_sscd_mode)",
        "sscd_retention_threshold": HIGH_SSCD_RETENTION_THRESHOLD,
        "include_when": (
            "any(sscd > sscd_retention_threshold) or "
            "2 * count(gmm_low_mode_probability >= 0.5) <= num_reference_seeds"
        ),
    }


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Spearman rho as Pearson correlation of average ranks."""

    if len(left) != len(right) or len(left) < 2:
        return math.nan
    values = [float(value) for value in (*left, *right)]
    if not all(math.isfinite(value) for value in values):
        return math.nan
    left_rank = pd.Series(left, dtype="float64").rank(method="average").tolist()
    right_rank = pd.Series(right, dtype="float64").rank(method="average").tolist()
    left_mean = math.fsum(left_rank) / len(left_rank)
    right_mean = math.fsum(right_rank) / len(right_rank)
    left_centered = [value - left_mean for value in left_rank]
    right_centered = [value - right_mean for value in right_rank]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return math.nan
    return (
        math.fsum(x * y for x, y in zip(left_centered, right_centered, strict=True))
        / denominator
    )


def _manifest(
    records: pd.DataFrame,
    observations: Mapping[str, Mapping[int, Mapping[str, object]]],
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    generation_hash: str,
    sscd_hash: str,
    num_seeds: int,
    selection_strategy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    count = _num_seeds(num_seeds)
    strategy = normalize_selection_strategy(selection_strategy)
    reference_seeds = _reference_seeds(count)
    rows: list[dict[str, object]] = []
    for record in records.to_dict(orient="records"):
        index = str(record["original_index"])
        per_seed = []
        for tile, seed in enumerate(reference_seeds):
            observed = observations.get(index, {}).get(
                seed,
                {
                    "l2_norm": math.nan,
                    "sscd": math.nan,
                    "observation_status": "missing",
                    "observation_error": "reference observation is missing",
                },
            )
            per_seed.append(
                {"seed": seed, "generated_image_tile_index": tile, **observed}
            )
        rho = _prompt_spearman(per_seed)
        issues = _observation_issues(per_seed)
        if issues:
            include, status, reason = (
                False,
                UNUSABLE_REFERENCE_OBSERVATIONS,
                "invalid_reference_observations:" + ",".join(issues),
            )
        else:
            include, status, reason = False, "", ""
        common = {
            "model_name": model,
            **{column: record[column] for column in _IDENTITY_COLUMNS},
            "generated_image_path": (
                reference_run_path(model, scheduler, guidance, steps, count)
                / "image"
                / f"{index}.png"
            ).as_posix(),
            "selection_strategy": strategy,
            "prompt_spearman": rho,
            "gmm_component": "",
            "gmm_low_mode_probability": math.nan,
            "include_prompt": include,
            "selection_status": status,
            "selection_reason": reason,
            "reference_run_name": reference_run_name(
                model, scheduler, guidance, steps, count
            ),
            "reference_generation_hash": generation_hash,
            "reference_sscd_hash": sscd_hash,
            "selection_policy": selection_policy(strategy),
            "selection_hash": "",
        }
        rows.extend({**common, **observed} for observed in per_seed)
    frame = pd.DataFrame(rows, columns=SELECTION_COLUMNS).sort_values(
        ["source_row_number", "original_index", "seed"],
        kind="stable",
        ignore_index=True,
    )
    gmm_fit = _apply_gmm_decisions(frame, strategy)
    return frame, gmm_fit


def _observation_issues(
    observations: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(row["observation_status"])
                for row in observations
                if row["observation_status"] != "complete"
            }
        )
    )


def _prompt_spearman(observations: Sequence[Mapping[str, object]]) -> float:
    if _observation_issues(observations):
        return math.nan
    return spearman_correlation(
        [float(row["l2_norm"]) for row in observations],
        [float(row["sscd"]) for row in observations],
    )


def _gmm_decision(
    low_mode_probabilities: Sequence[float],
    sscd_scores: Sequence[float],
) -> tuple[bool, str, str]:
    probabilities = [float(value) for value in low_mode_probabilities]
    if not probabilities or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities
    ):
        raise TargetPairSelectionError("GMM low-mode probabilities are invalid")
    scores = [float(value) for value in sscd_scores]
    if len(scores) != len(probabilities) or any(
        not math.isfinite(value) for value in scores
    ):
        raise TargetPairSelectionError("GMM SSCD scores are invalid or misaligned")
    if any(value > HIGH_SSCD_RETENTION_THRESHOLD for value in scores):
        return (
            True,
            INCLUDED_PROXIMITY_RULE,
            "has_reference_seed_sscd_above_retention_threshold",
        )
    # A posterior tie assigns that seed to low; a prompt-level tie is retained.
    low_seed_count = sum(value >= 0.5 for value in probabilities)
    if 2 * low_seed_count <= len(probabilities):
        return (
            True,
            INCLUDED_PROXIMITY_RULE,
            "no_low_sscd_mode_majority",
        )
    return (
        False,
        DISCARDED_PROXIMITY_RULE,
        "majority_reference_seeds_in_low_sscd_mode",
    )


def _gmm_fit_positions(frame: pd.DataFrame) -> list[int]:
    """Fit every valid seed, including valid siblings of an incomplete seed."""

    return [
        int(position)
        for position in frame.index[frame["observation_status"].eq("complete")]
    ]


def _apply_gmm_decisions(
    frame: pd.DataFrame, selection_strategy: str
) -> dict[str, object]:
    normalize_selection_strategy(selection_strategy)
    usable_positions = _gmm_fit_positions(frame)
    usable_prompts = [
        str(index)
        for index, group in frame.groupby("original_index", sort=False)
        if group["observation_status"].eq("complete").all()
    ]
    if not usable_positions:
        raise TargetPairSelectionError(
            "GMM selection has no complete reference observations"
        )
    raw = frame.loc[usable_positions, ["l2_norm", "sscd"]].to_numpy(dtype=np.float64)
    try:
        standardized, feature_mean, feature_scale = standardize_features(raw)
        fit = fit_gaussian_mixture(standardized, reg_covar=DEFAULT_REG_COVAR)
    except ValueError as error:
        raise TargetPairSelectionError(f"Cannot fit selection GMM: {error}") from error

    low_probability = pd.Series(
        fit.responsibilities[:, 0], index=usable_positions, dtype="float64"
    )
    hard_component = np.asarray(COMPONENT_NAMES)[fit.responsibilities.argmax(axis=1)]
    frame.loc[usable_positions, "gmm_low_mode_probability"] = low_probability
    frame.loc[usable_positions, "gmm_component"] = hard_component
    for index in usable_prompts:
        positions = frame.index[frame["original_index"].eq(index)].tolist()
        include, status, reason = _gmm_decision(
            [float(low_probability.loc[position]) for position in positions],
            frame.loc[positions, "sscd"].tolist(),
        )
        frame.loc[positions, "include_prompt"] = include
        frame.loc[positions, "selection_status"] = status
        frame.loc[positions, "selection_reason"] = reason

    return {
        "feature_names": ["l2_norm", "sscd"],
        "standardization": "population_zscore",
        "standardization_ddof": 0,
        "feature_mean": [float(value) for value in feature_mean],
        "feature_scale": [float(value) for value in feature_scale],
        "component_names": list(COMPONENT_NAMES),
        "num_components": 2,
        "covariance_type": "full",
        "initialization": INITIALIZATION_NAME,
        "reg_covar": DEFAULT_REG_COVAR,
        "max_iterations": MAX_ITERATIONS,
        "tolerance": TOLERANCE,
        "likelihood_decrease_tolerance": LIKELIHOOD_DECREASE_TOLERANCE,
        "weights": [float(value) for value in fit.weights],
        "means_standardized": fit.means.astype(float).tolist(),
        "covariances_standardized": fit.covariances.astype(float).tolist(),
        "log_likelihood": float(fit.log_likelihood),
        "iterations": int(fit.iterations),
        "usable_prompt_count": int(
            frame.loc[usable_positions, "original_index"].nunique()
        ),
        "usable_observation_count": len(usable_positions),
    }


def _prompt_records(source: pd.DataFrame, *, derived: bool, model: str) -> pd.DataFrame:
    missing = sorted(set(_IDENTITY_COLUMNS) - set(source.columns))
    if missing:
        suffix = " or provide records_frame" if derived else ""
        raise TargetPairSelectionError(
            "Prompt identity lacks: " + ", ".join(missing) + suffix
        )
    if "model_name" in source:
        observed = {str(value) for value in source["model_name"] if not _missing(value)}
        if not observed or not observed.issubset({model, _MODELS[model]}):
            raise TargetPairSelectionError(f"Prompt records do not belong to {model}")
    frame = source.loc[:, _IDENTITY_COLUMNS].copy()
    frame["original_index"] = frame["original_index"].map(_index)
    if derived:
        rows = []
        for index, group in frame.groupby("original_index", sort=False):
            row = {"original_index": index}
            for column in _IDENTITY_COLUMNS[1:]:
                values = [_scalar(value) for value in group[column]]
                if len({repr(value) for value in values}) != 1:
                    raise TargetPairSelectionError(
                        f"Prompt identity differs for {index}: {column}"
                    )
                row[column] = values[0]
            rows.append(row)
        frame = pd.DataFrame(rows, columns=_IDENTITY_COLUMNS)
    elif frame["original_index"].duplicated().any():
        raise TargetPairSelectionError("records_frame must have one row per prompt")
    for row in frame.to_dict(orient="records"):
        if not isinstance(row["record_id"], str) or not row["record_id"].strip():
            raise TargetPairSelectionError("record_id must be a non-empty string")
        if not isinstance(row["prompt"], str):
            raise TargetPairSelectionError("prompt must be a string")
        if _integer(row["source_row_number"], "source_row_number") < 0:
            raise TargetPairSelectionError("source_row_number must be non-negative")
        if not _is_hash(row["target_image_sha256"]):
            raise TargetPairSelectionError("target_image_sha256 is invalid")
    frame["source_row_number"] = frame["source_row_number"].map(
        lambda value: _integer(value, "source_row_number")
    )
    frame["kind"] = frame["kind"].map(normalize_webster_type)
    return frame.sort_values(
        ["source_row_number", "original_index"], kind="stable", ignore_index=True
    )


def _observations(
    paired: pd.DataFrame,
    known: set[str],
    *,
    num_seeds: int,
) -> dict[str, dict[int, dict[str, object]]]:
    reference_seeds = frozenset(_reference_seeds(num_seeds))
    required = {
        "original_index",
        "seed",
        "l2_norm",
        "sscd",
        "observation_status",
        "observation_error",
    }
    missing = sorted(required - set(paired.columns))
    if missing:
        raise TargetPairSelectionError(
            "Paired reference observations lack: " + ", ".join(missing)
        )
    grouped: dict[str, dict[int, list[Mapping[str, object]]]] = {}
    for row in paired.loc[:, sorted(required)].to_dict(orient="records"):
        index = _index(row["original_index"])
        if index not in known:
            raise TargetPairSelectionError(f"Unknown paired prompt {index}")
        seed = _integer(row["seed"], "seed")
        if seed not in reference_seeds:
            first, last = min(reference_seeds), max(reference_seeds)
            raise TargetPairSelectionError(
                f"Reference seed outside {first}--{last}: {seed}"
            )
        grouped.setdefault(index, {}).setdefault(seed, []).append(row)
    result: dict[str, dict[int, dict[str, object]]] = {}
    for index, seeds in grouped.items():
        result[index] = {}
        for seed, rows in seeds.items():
            if len(rows) != 1:
                result[index][seed] = {
                    "l2_norm": math.nan,
                    "sscd": math.nan,
                    "observation_status": "duplicate",
                    "observation_error": (
                        f"duplicate reference observations for prompt {index}, seed {seed}"
                    ),
                }
                continue
            row = rows[0]
            source_status = str(row["observation_status"]).strip()
            source_error = str(row["observation_error"]).strip()
            if source_status != "complete":
                result[index][seed] = {
                    "l2_norm": math.nan,
                    "sscd": math.nan,
                    "observation_status": source_status or "failed",
                    "observation_error": source_error or "reference observation failed",
                }
                continue
            l2, l2_issue = _metric(row["l2_norm"], "l2_norm")
            sscd, sscd_issue = _metric(row["sscd"], "sscd")
            issues = [issue for issue in (l2_issue, sscd_issue) if issue]
            result[index][seed] = {
                "l2_norm": l2,
                "sscd": sscd,
                "observation_status": "complete" if not issues else "+".join(issues),
                "observation_error": "" if not issues else ",".join(issues),
            }
    return result


def _metric(value: object, name: str) -> tuple[float, str | None]:
    if _missing(value) or value == "":
        return math.nan, f"missing_{name}"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan, f"nonnumeric_{name}"
    if not math.isfinite(number):
        return math.nan, f"nonfinite_{name}"
    if name == "l2_norm" and number < 0:
        return math.nan, "negative_l2_norm"
    if name == "sscd" and not -1.00001 <= number <= 1.00001:
        return math.nan, "out_of_range_sscd"
    return number, None


def _validate_prompt_group(
    index: str,
    group: pd.DataFrame,
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    num_seeds: int,
    selection_strategy: str,
) -> None:
    count = _num_seeds(num_seeds)
    normalize_selection_strategy(selection_strategy)
    if len(group) != count:
        raise TargetPairSelectionError(
            f"Prompt {index} must contain exactly {count} rows"
        )
    seeds = tuple(_integer(value, "seed") for value in group["seed"])
    tiles = tuple(
        _integer(value, "tile") for value in group["generated_image_tile_index"]
    )
    if seeds != _reference_seeds(count) or tiles != tuple(range(count)):
        raise TargetPairSelectionError(
            f"Prompt {index} seed or tile sequence is invalid"
        )
    image = (
        reference_run_path(model, scheduler, guidance, steps, count)
        / "image"
        / f"{index}.png"
    ).as_posix()
    if not group["generated_image_path"].eq(image).all():
        raise TargetPairSelectionError(f"Prompt {index} image path is invalid")
    constant = (
        "record_id",
        "source_row_number",
        "prompt",
        "kind",
        "target_image_sha256",
        "selection_strategy",
        "prompt_spearman",
        "include_prompt",
        "selection_status",
        "selection_reason",
    )
    for column in constant:
        values = [_json(value) for value in group[column]]
        if any(value != values[0] for value in values[1:]):
            raise TargetPairSelectionError(f"Prompt {index} has inconsistent {column}")
    if group.iloc[0]["kind"] not in {"MV", "RV", "TV", "N", "UNKNOWN"}:
        raise TargetPairSelectionError(f"Prompt {index} kind is invalid")
    observations = []
    for row in group.to_dict(orient="records"):
        status = str(row["observation_status"])
        error = str(row["observation_error"])
        l2 = _optional_float(row["l2_norm"], "l2_norm")
        sscd = _optional_float(row["sscd"], "sscd")
        if status == "complete" and (math.isnan(l2) or math.isnan(sscd) or error):
            raise TargetPairSelectionError(
                f"Prompt {index} has an invalid complete row"
            )
        if status != "complete" and not error:
            raise TargetPairSelectionError(
                f"Prompt {index} has an unexplained failed row"
            )
        observations.append({"l2_norm": l2, "sscd": sscd, "observation_status": status})
    rho = _prompt_spearman(observations)
    stored_rho = _optional_float(group.iloc[0]["prompt_spearman"], "prompt_spearman")
    first = group.iloc[0]
    if not _same_float(rho, stored_rho):
        raise TargetPairSelectionError(f"Prompt {index} Spearman value is inconsistent")
    low_probability = [
        _optional_float(value, "gmm_low_mode_probability")
        for value in group["gmm_low_mode_probability"]
    ]
    components = [str(value) for value in group["gmm_component"]]
    issues = _observation_issues(observations)
    if issues:
        _require_prompt_decision(
            index,
            first,
            False,
            UNUSABLE_REFERENCE_OBSERVATIONS,
            "invalid_reference_observations:" + ",".join(issues),
        )
        for observation, component, probability in zip(
            observations, components, low_probability, strict=True
        ):
            if observation["observation_status"] == "complete":
                if component not in COMPONENT_NAMES or not (
                    math.isfinite(probability) and 0.0 <= probability <= 1.0
                ):
                    raise TargetPairSelectionError(
                        f"Valid observation in prompt {index} lacks a GMM assignment"
                    )
            elif component or not math.isnan(probability):
                raise TargetPairSelectionError(
                    f"Unusable observation in prompt {index} has seed-level GMM fields"
                )
        return
    if any(component not in COMPONENT_NAMES for component in components):
        raise TargetPairSelectionError(f"Prompt {index} has an invalid GMM component")
    if any(
        math.isnan(value) or value < 0.0 or value > 1.0 for value in low_probability
    ):
        raise TargetPairSelectionError(f"Prompt {index} has an invalid GMM posterior")
    _require_prompt_decision(
        index,
        first,
        *_gmm_decision(
            low_probability, [float(row["sscd"]) for row in observations]
        ),
    )


def _require_prompt_decision(
    index: str,
    first: pd.Series,
    include: bool,
    status: str,
    reason: str,
) -> None:
    if (
        bool(first["include_prompt"]) != include
        or first["selection_status"] != status
        or first["selection_reason"] != reason
    ):
        raise TargetPairSelectionError(f"Prompt {index} decision is inconsistent")


_GMM_FIT_KEYS = {
    "feature_names",
    "standardization",
    "standardization_ddof",
    "feature_mean",
    "feature_scale",
    "component_names",
    "num_components",
    "covariance_type",
    "initialization",
    "reg_covar",
    "max_iterations",
    "tolerance",
    "likelihood_decrease_tolerance",
    "weights",
    "means_standardized",
    "covariances_standardized",
    "log_likelihood",
    "iterations",
    "usable_prompt_count",
    "usable_observation_count",
}


def _validate_gmm_decisions(
    frame: pd.DataFrame,
    gmm_fit: Mapping[str, object],
    selection_strategy: str,
) -> None:
    normalize_selection_strategy(selection_strategy)
    if set(gmm_fit) != _GMM_FIT_KEYS:
        raise TargetPairSelectionError("GMM fit has an invalid schema")
    fixed = {
        "feature_names": ["l2_norm", "sscd"],
        "standardization": "population_zscore",
        "standardization_ddof": 0,
        "component_names": list(COMPONENT_NAMES),
        "num_components": 2,
        "covariance_type": "full",
        "initialization": INITIALIZATION_NAME,
        "reg_covar": DEFAULT_REG_COVAR,
        "max_iterations": MAX_ITERATIONS,
        "tolerance": TOLERANCE,
        "likelihood_decrease_tolerance": LIKELIHOOD_DECREASE_TOLERANCE,
    }
    if any(
        canonical_hash(gmm_fit.get(key)) != canonical_hash(value)
        for key, value in fixed.items()
    ):
        raise TargetPairSelectionError("GMM fit settings are inconsistent")

    usable_positions = _gmm_fit_positions(frame)
    usable_prompt_count = int(frame.loc[usable_positions, "original_index"].nunique())
    if _integer(
        gmm_fit.get("usable_prompt_count"), "usable_prompt_count"
    ) != usable_prompt_count or _integer(
        gmm_fit.get("usable_observation_count"), "usable_observation_count"
    ) != len(usable_positions):
        raise TargetPairSelectionError("GMM fit counts are inconsistent")
    if not usable_positions:
        raise TargetPairSelectionError("GMM fit has no usable observations")

    try:
        feature_mean = np.asarray(gmm_fit["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(gmm_fit["feature_scale"], dtype=np.float64)
        weights = np.asarray(gmm_fit["weights"], dtype=np.float64)
        means = np.asarray(gmm_fit["means_standardized"], dtype=np.float64)
        covariances = np.asarray(gmm_fit["covariances_standardized"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TargetPairSelectionError("GMM fit arrays are invalid") from error
    if (
        feature_mean.shape != (2,)
        or feature_scale.shape != (2,)
        or weights.shape != (2,)
        or means.shape != (2, 2)
        or covariances.shape != (2, 2, 2)
        or not all(
            np.all(np.isfinite(values))
            for values in (feature_mean, feature_scale, weights, means, covariances)
        )
        or np.any(feature_scale <= 0.0)
        or np.any(weights <= 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)
        or not means[0, 1] < means[1, 1]
    ):
        raise TargetPairSelectionError("GMM fit parameters are invalid")
    for covariance in covariances:
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise TargetPairSelectionError("GMM covariance is not symmetric")
        try:
            eigenvalues = np.linalg.eigvalsh(covariance)
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise TargetPairSelectionError(
                "GMM covariance is not positive definite"
            ) from error
        if np.any(eigenvalues < DEFAULT_REG_COVAR * (1.0 - 1e-10)):
            raise TargetPairSelectionError(
                "GMM covariance is below its eigenvalue floor"
            )

    raw = frame.loc[usable_positions, ["l2_norm", "sscd"]].to_numpy(dtype=np.float64)
    try:
        standardized, observed_mean, observed_scale = standardize_features(raw)
    except ValueError as error:
        raise TargetPairSelectionError(
            f"GMM observations cannot be standardized: {error}"
        ) from error
    if not np.allclose(feature_mean, observed_mean, rtol=1e-12, atol=1e-12):
        raise TargetPairSelectionError("GMM feature mean differs from observations")
    if not np.allclose(feature_scale, observed_scale, rtol=1e-12, atol=1e-12):
        raise TargetPairSelectionError("GMM feature scale differs from observations")
    try:
        responsibilities, log_likelihood = expectation(
            standardized, weights, means, covariances
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        raise TargetPairSelectionError("GMM posterior computation failed") from error
    stored_log_likelihood = _optional_float(
        gmm_fit.get("log_likelihood"), "log_likelihood"
    )
    if math.isnan(stored_log_likelihood) or not math.isclose(
        log_likelihood,
        stored_log_likelihood,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise TargetPairSelectionError("GMM log likelihood is inconsistent")
    iterations = _integer(gmm_fit.get("iterations"), "iterations")
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise TargetPairSelectionError("GMM iteration count is invalid")

    expected_low = pd.Series(
        responsibilities[:, 0], index=usable_positions, dtype="float64"
    )
    expected_component = pd.Series(
        np.asarray(COMPONENT_NAMES)[responsibilities.argmax(axis=1)],
        index=usable_positions,
        dtype="object",
    )
    for position in usable_positions:
        observed_probability = _optional_float(
            frame.at[position, "gmm_low_mode_probability"],
            "gmm_low_mode_probability",
        )
        if (
            not math.isclose(
                observed_probability,
                float(expected_low.loc[position]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or frame.at[position, "gmm_component"] != expected_component.loc[position]
        ):
            raise TargetPairSelectionError("GMM seed assignment is inconsistent")

    for index, group in frame.groupby("original_index", sort=False):
        if not group["observation_status"].eq("complete").all():
            continue
        positions = [int(position) for position in group.index]
        include, status, reason = _gmm_decision(
            [float(expected_low.loc[position]) for position in positions],
            group["sscd"].tolist(),
        )
        _require_prompt_decision(str(index), group.iloc[0], include, status, reason)


def _completion_marker_directory_fingerprint(
    directory: Path, label: str
) -> dict[str, object]:
    if directory.is_symlink() or not directory.is_dir():
        raise TargetPairSelectionError(
            f"Reference {label} completion marker directory is missing or unsafe: "
            f"{directory}"
        )
    try:
        markers = sorted(
            (path for path in directory.iterdir() if path.suffix == ".json"),
            key=lambda path: path.name,
        )
    except OSError as error:
        raise TargetPairSelectionError(
            f"Cannot list reference {label} completion markers: {directory}"
        ) from error

    entries: list[dict[str, str]] = []
    for marker in markers:
        if marker.is_symlink() or not marker.is_file():
            raise TargetPairSelectionError(
                f"Reference {label} completion marker is unsafe: {marker}"
            )
        try:
            digest = file_sha256(marker)
        except OSError as error:
            raise TargetPairSelectionError(
                f"Cannot hash reference {label} completion marker: {marker}"
            ) from error
        entries.append({"name": marker.name, "sha256": digest})
    return {"count": len(entries), "sha256": canonical_hash(entries)}


def _normalize_completion_fingerprint(value: object) -> dict[str, object]:
    labels = tuple(label for label, _ in _REFERENCE_MARKER_DIRECTORIES)
    if not isinstance(value, Mapping) or set(value) != set(labels):
        raise TargetPairSelectionError(
            "Reference completion fingerprint has an invalid schema"
        )
    result: dict[str, object] = {}
    for label in labels:
        item = value.get(label)
        if not isinstance(item, Mapping) or set(item) != {"count", "sha256"}:
            raise TargetPairSelectionError(
                "Reference completion fingerprint has an invalid schema"
            )
        count = _integer(item.get("count"), f"{label} marker count")
        digest = item.get("sha256")
        if count < 0 or not _is_hash(digest):
            raise TargetPairSelectionError(
                "Reference completion fingerprint is invalid"
            )
        result[label] = {"count": count, "sha256": str(digest)}
    return result


def _selection_config(
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    num_seeds: int,
    generation_hash: str,
    sscd_hash: str,
    checkpoint_hash: object,
    preprocessing_hash: object,
    completion_fingerprint: Mapping[str, object],
    selection_hash: str,
    selection_strategy: str,
    gmm_fit: Mapping[str, object] | None,
) -> dict[str, object]:
    count = _num_seeds(num_seeds)
    reference_seeds = _reference_seeds(count)
    strategy = normalize_selection_strategy(selection_strategy)
    fingerprint = _normalize_completion_fingerprint(completion_fingerprint)
    if not isinstance(gmm_fit, Mapping):
        raise TargetPairSelectionError("GMM selection fit is missing")
    result: dict[str, object] = {
        "selection_strategy": strategy,
        "selection_policy": selection_policy(strategy),
        "model_name": model,
        "dataset_model_name": _MODELS[model],
        "reference_run_name": reference_run_name(
            model, scheduler, guidance, steps, count
        ),
        "reference_run_path": reference_run_path(
            model, scheduler, guidance, steps, count
        ).as_posix(),
        "reference_scheduler": scheduler,
        "reference_guidance_scale": guidance,
        "reference_num_inference_steps": steps,
        "reference_seed_start": count,
        "reference_num_seeds": count,
        "reference_seeds": list(reference_seeds),
        **_strategy_contract(strategy),
        "decision_scope": "whole_prompt",
        "kind_affects_selection": False,
        "unusable_evidence_is_included": False,
        "generated_image_path_base": "project_root",
        "generated_image_tile_order": "zero_based_row_major_reference_seed_order",
        "reference_generation_hash": generation_hash,
        "reference_sscd_hash": sscd_hash,
        "reference_completion_fingerprint": fingerprint,
        "sscd_checkpoint_sha256": checkpoint_hash,
        "sscd_preprocessing_hash": preprocessing_hash,
        "selection_hash": selection_hash,
    }
    result["gmm_fit"] = dict(gmm_fit)
    return result


def _validate_generation(
    config: Mapping[str, object],
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    num_seeds: int,
) -> str:
    count = _num_seeds(num_seeds)
    science, digest = (
        config.get("scientific_config"),
        config.get("scientific_config_hash"),
    )
    if (
        not isinstance(science, Mapping)
        or not _is_hash(digest)
        or canonical_hash(science) != digest
    ):
        raise TargetPairSelectionError(
            "Reference generation scientific hash is invalid"
        )
    scheduler_metadata = science.get("scheduler")
    scheduler_name = (
        scheduler_metadata.get("name")
        if isinstance(scheduler_metadata, Mapping)
        else scheduler_metadata
    )
    expected = {
        "model_cli_name": model,
        "dataset_model": _MODELS[model],
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "num_seeds": count,
        "seeds": list(_reference_seeds(count)),
    }
    wrong = [key for key, value in expected.items() if science.get(key) != value]
    if scheduler_name != scheduler:
        wrong.append("scheduler")
    if wrong:
        raise TargetPairSelectionError(
            "Reference generation differs at: " + ", ".join(wrong)
        )
    return str(digest)


def _validate_sscd(
    config: Mapping[str, object], generation_hash: str, num_seeds: int
) -> str:
    from utils.experiments.sscd import sscd_configuration_hash

    count = _num_seeds(num_seeds)

    digest = config.get("configuration_hash")
    if not _is_hash(digest) or sscd_configuration_hash(config) != digest:
        raise TargetPairSelectionError("Reference SSCD configuration hash is invalid")
    expected = {
        "generation_scientific_config_hash": generation_hash,
        "num_seeds": count,
        "seeds": list(_reference_seeds(count)),
    }
    wrong = [key for key, value in expected.items() if config.get(key) != value]
    if wrong:
        raise TargetPairSelectionError("Reference SSCD differs at: " + ", ".join(wrong))
    if any(
        not _is_hash(config.get(key))
        for key in ("sscd_checkpoint_sha256", "sscd_preprocessing_hash")
    ):
        raise TargetPairSelectionError("Reference SSCD provenance hash is invalid")
    return str(digest)


def _write_selection(
    selection: TargetPairSelection, directory: Path, *, overwrite: bool
) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    destination_exists = directory.exists() or directory.is_symlink()
    if destination_exists and (directory.is_symlink() or not directory.is_dir()):
        raise TargetPairSelectionError(
            f"Cannot overwrite unsafe selection directory: {directory}"
        )
    if destination_exists and not overwrite:
        raise FrozenTargetPairSelectionError(
            f"Frozen selection already exists: {directory}"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.", suffix=".tmp", dir=directory.parent
        )
    )
    try:
        atomic_write_frame_csv(selection.frame, temporary / "selection.csv")
        atomic_write_json(temporary / "config.json", selection.configuration)
        atomic_write_json(temporary / "summary.json", _summary(selection))
        _fsync_directory(temporary)
        if destination_exists:
            _replace_selection_directory(temporary, directory)
        else:
            os.replace(temporary, directory)
            _fsync_directory(directory.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _replace_selection_directory(temporary: Path, directory: Path) -> None:
    """Install a complete replacement while retaining rollback state."""

    rollback_root = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.", suffix=".rollback", dir=directory.parent
        )
    )
    previous = rollback_root / "previous"
    installed = rollback_root / "replacement"
    try:
        os.replace(directory, previous)
    except BaseException:
        shutil.rmtree(rollback_root, ignore_errors=True)
        raise
    try:
        os.replace(temporary, directory)
        _fsync_directory(directory.parent)
    except BaseException:
        try:
            if directory.exists() or directory.is_symlink():
                os.replace(directory, installed)
            os.replace(previous, directory)
            _fsync_directory(directory.parent)
        except BaseException as rollback_error:
            raise TargetPairSelectionError(
                "Frozen selection replacement failed and rollback state was "
                f"preserved at {rollback_root}"
            ) from rollback_error
        shutil.rmtree(rollback_root, ignore_errors=True)
        raise
    shutil.rmtree(rollback_root)
    _fsync_directory(directory.parent)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _summary(selection: TargetPairSelection) -> dict[str, object]:
    prompts = selection.prompt_frame
    included = prompts["include_prompt"]
    unusable = prompts["selection_status"].eq(UNUSABLE_REFERENCE_OBSERVATIONS)
    return {
        "complete": True,
        "selection_strategy": selection.selection_strategy,
        "selection_policy": selection_policy(selection.selection_strategy),
        "model_name": selection.model_name,
        "selection_hash": selection.sha256,
        "reference_generation_hash": selection.configuration[
            "reference_generation_hash"
        ],
        "reference_sscd_hash": selection.configuration["reference_sscd_hash"],
        "reference_observation_count": len(selection.frame),
        "total_prompt_count": len(prompts),
        "included_prompt_count": int(included.sum()),
        "discarded_prompt_count": int((~included & ~unusable).sum()),
        "unusable_prompt_count": int(unusable.sum()),
    }


def _read_config(
    source: ConfigurationInput | None, default: Path, label: str
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = default if source is None else Path(source)
    if not path.is_file():
        raise TargetPairSelectionError(f"Missing {label}: {path}")
    try:
        return read_json(path)
    except RuntimeError as error:
        raise TargetPairSelectionError(f"Invalid {label}: {error}") from error


def _read_frame(source: FrameInput, label: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    path = Path(source)
    if not path.is_file() or path.suffix.casefold() != ".csv":
        raise TargetPairSelectionError(f"{label} must be a DataFrame or CSV")
    try:
        return pd.read_csv(
            path,
            dtype={"original_index": str},
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, ValueError) as error:
        raise TargetPairSelectionError(
            f"Cannot read {label} {path}: {error}"
        ) from error


def _normalize_csv(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != SELECTION_COLUMNS:
        raise TargetPairSelectionError("selection.csv has an invalid schema")
    result = frame.copy()
    for column in ("source_row_number", "seed", "generated_image_tile_index"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    for column in (
        "l2_norm",
        "sscd",
        "prompt_spearman",
        "gmm_low_mode_probability",
    ):
        result[column] = result[column].map(
            lambda value, name=column: _optional_float(value, name)
        )
    if not pd.api.types.is_bool_dtype(result["include_prompt"]):
        values = {"True": True, "False": False}
        if not result["include_prompt"].isin(values).all():
            raise TargetPairSelectionError("include_prompt must contain booleans")
        result["include_prompt"] = result["include_prompt"].map(values).astype(bool)
    return result.loc[:, SELECTION_COLUMNS]


def _optional_float(value: object, label: str) -> float:
    if _missing(value) or value == "":
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetPairSelectionError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise TargetPairSelectionError(f"{label} must be finite or blank")
    return number


def _same_float(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or left == right


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TargetPairSelectionError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TargetPairSelectionError(f"{label} must be an integer") from error
        if not math.isfinite(number) or not number.is_integer():
            raise TargetPairSelectionError(f"{label} must be an integer")
        return int(number)


def _index(value: object) -> str:
    if _missing(value):
        raise TargetPairSelectionError("original_index is missing")
    text = str(value).strip()
    if not text or text in {".", ".."} or Path(text).name != text or "\\" in text:
        raise TargetPairSelectionError(f"Unsafe original_index: {value!r}")
    return text


def _num_seeds(value: object) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        count = operator.index(value)
        validate_seed_block(count, count)
    except (TypeError, ValueError) as error:
        raise TargetPairSelectionError(
            "num_seeds must define a valid experiment and reference seed block"
        ) from error
    return count


def _reference_values(
    model_name: object,
    scheduler_name: object,
    guidance_scale: object,
    num_inference_steps: object,
    num_seeds: object,
) -> tuple[str, str, float, int, int]:
    model = _model(model_name)
    count = _num_seeds(num_seeds)
    try:
        if not isinstance(scheduler_name, str):
            raise TypeError
        if isinstance(guidance_scale, bool):
            raise TypeError
        guidance = float(guidance_scale)
        if not math.isfinite(guidance):
            raise ValueError
        if isinstance(num_inference_steps, bool):
            raise TypeError
        steps = operator.index(num_inference_steps)
        generation_run_name(
            model,
            scheduler_name,
            guidance,
            steps,
            count,
            seed_start=count,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetPairSelectionError(
            "model, scheduler, guidance scale, steps, and N must define a valid "
            "reference run"
        ) from error
    return model, scheduler_name, guidance, steps, count


def _reference_seeds(num_seeds: int) -> tuple[int, ...]:
    count = _num_seeds(num_seeds)
    return tuple(range(count, 2 * count))


def _model(value: object) -> str:
    if not isinstance(value, str) or value not in _MODELS:
        raise TargetPairSelectionError(
            f"Unknown model {value!r}; expected one of: {', '.join(_MODELS)}"
        )
    return value


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _scalar(value: object) -> object:
    if _missing(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _json(value: object) -> object:
    if _missing(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json(item) for item in value]
    return value


def _raise_missing(
    root: str | Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str,
) -> None:
    model, scheduler, guidance, steps, count = _reference_values(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
    )
    strategy = normalize_selection_strategy(selection_strategy)
    directory = target_pair_selection_directory(
        root,
        model_name=model,
        scheduler_name=scheduler,
        guidance_scale=guidance,
        num_inference_steps=steps,
        num_seeds=count,
        selection_strategy=strategy,
    )
    raise TargetPairSelectionMissingError(
        f"Frozen target-pair selection is missing: {directory}\n"
        f"Required reference run: "
        f"{reference_run_path(model, scheduler, guidance, steps, count).as_posix()}\n"
        f"Create it with:\n"
        f"{
            reference_selection_command(
                model, scheduler, guidance, steps, count, strategy
            )
        }"
    )
