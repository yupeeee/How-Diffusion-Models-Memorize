"""Fixed, frozen, model-specific Webster target-pair selection.

Ordinary experiments use seeds 0--19. Selection is built from a distinct
reference run: seeds 20--29 affect prompt inclusion, while seeds 30--39 are a
reference-only validation half. Experimental scores, validation scores,
latent proximity, and correlations never affect inclusion.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import json
import math
import operator
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

import pandas as pd

from utils.common.cli import generation_run_name
from utils.common.io import (
    atomic_write_frame_csv,
    atomic_write_frame_parquet,
    atomic_write_json,
    canonical_hash,
    read_json,
)
from utils.experiments.cache import generation_log_relative_path

REFERENCE_SSCD_BOUNDARY = 0.25
# Retain the old public name for callers that use the TV-specific spelling.
TV_MEAN_SSCD_THRESHOLD = REFERENCE_SSCD_BOUNDARY
REFERENCE_SCHEDULER, REFERENCE_GUIDANCE_SCALE = "ddim", 7.5
REFERENCE_NUM_INFERENCE_STEPS, REFERENCE_SEED_START = 50, 20
REFERENCE_NUM_SEEDS = 20
SELECTION_SEEDS = tuple(range(20, 30))
REFERENCE_VALIDATION_SEEDS = tuple(range(30, 40))
REFERENCE_SEEDS = SELECTION_SEEDS + REFERENCE_VALIDATION_SEEDS
SELECTION_POLICY = "target_pair_selection_tv_ge_0_25_n_ge_0_25"
SELECTION_SCHEMA_VERSION = 4
THRESHOLD_SENSITIVITY = (0.15, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40)
INCLUDED_NON_TV, INCLUDED_TV_TARGET_SUPPORTED = "included_non_tv", "included_tv_target_supported"
EXCLUDED_TV_TARGET_UNSUPPORTED = "excluded_tv_target_unsupported"
INCLUDED_N_TARGET_SUPPORTED = "included_n_target_supported"
EXCLUDED_N_TARGET_UNSUPPORTED = "excluded_n_target_unsupported"
MISSING_REFERENCE_SSCD, INVALID_RECORD = "missing_reference_sscd", "invalid_record"

_MODEL_DATASET = {
    "sdv1": "sdv1", "sdv2": "sdv2", "realvis": "realisticvision"
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITY_COLUMNS = tuple(
    "original_index record_id source_row_number prompt_raw "
    "webster_overfit_type_raw target_image_sha256".split()
)
SELECTION_COLUMNS = tuple(
    "model_name original_index record_id source_row_number prompt_raw "
    "target_image_sha256 webster_overfit_type_raw "
    "webster_overfit_type_normalized reference_run_name "
    "reference_generation_hash reference_sscd_hash selection_seed_values "
    "reference_validation_seed_values selection_mean_sscd selection_median_sscd "
    "selection_min_sscd selection_max_sscd selection_fraction_ge_0_25 "
    "reference_validation_mean_sscd reference_validation_median_sscd "
    "comparison_operator selection_rule_passes reference_validation_rule_passes "
    "selection_validation_agree include_target_pair "
    "selection_status selection_reason target_semantics threshold "
    "selection_policy selection_hash".split()
)
DIAGNOSTIC_COLUMNS = tuple(
    "category comparison_operator threshold included_prompt_count "
    "excluded_prompt_count scored_prompt_count missing_prompt_count "
    "validation_scored_prompt_count validation_missing_prompt_count "
    "selection_validation_agreement_count "
    "selection_validation_disagreement_count is_fixed_threshold "
    "otsu_threshold otsu_lower_neighbor otsu_upper_neighbor "
    "otsu_lower_count otsu_upper_count".split()
)
_FILES = tuple(
    "selection.parquet selection.csv selected_tv.csv excluded_tv.csv "
    "selected_n.csv excluded_n.csv "
    "threshold_diagnostics.csv config.json summary.json".split()
)
_FROZEN_COMPATIBILITY_CONFIG_FIELDS = tuple(
    "schema_version selection_policy model_name dataset_model_name "
    "reference_run_name reference_scheduler reference_guidance_scale "
    "reference_num_inference_steps reference_seed_start reference_num_seeds "
    "reference_seeds selection_seeds reference_validation_seeds boundary "
    "threshold category_rules "
    "decision_scope selection_metric reference_validation_values_affect_selection "
    "proximity_affects_selection correlation_affects_selection "
    "sscd_checkpoint_sha256 sscd_preprocessing_hash".split()
)

class TargetPairSelectionError(RuntimeError):
    """Invalid selection input or artifact."""


class TargetPairSelectionMissingError(TargetPairSelectionError):
    """Missing frozen selection."""


class FrozenTargetPairSelectionError(TargetPairSelectionError):
    """Frozen evidence mismatch."""


class StaleTargetPairSelectionError(FrozenTargetPairSelectionError):
    """A frozen artifact implements an older scientific selection policy."""


@dataclass(frozen=True, slots=True)
class OtsuDiagnostic:
    """The exact best split between neighboring sorted prompt means."""
    threshold: float
    lower_neighbor: float
    upper_neighbor: float
    lower_count: int
    upper_count: int
    between_class_variance: float
@dataclass(frozen=True, slots=True)
class TargetPairSelection:
    """One validated frozen prompt selection and its audit values."""
    root: Path
    model_name: str
    frame: pd.DataFrame
    configuration: Mapping[str, object]
    sha256: str
    diagnostics: pd.DataFrame
    @property
    def included_indices(self) -> frozenset[str]:
        return _indices(self.frame, self.frame["include_target_pair"].astype(bool))
    @property
    def excluded_indices(self) -> frozenset[str]:
        return _indices(self.frame, ~self.frame["include_target_pair"].astype(bool))
    @property
    def selected_tv_indices(self) -> frozenset[str]:
        mask = self.frame["include_target_pair"].astype(bool) & self.frame[
            "webster_overfit_type_normalized"
        ].eq("TV")
        return _indices(self.frame, mask)

    @property
    def excluded_tv_indices(self) -> frozenset[str]:
        mask = ~self.frame["include_target_pair"].astype(bool) & self.frame[
            "webster_overfit_type_normalized"
        ].eq("TV")
        return _indices(self.frame, mask)

    @property
    def selected_n_indices(self) -> frozenset[str]:
        mask = self.frame["include_target_pair"].astype(bool) & self.frame[
            "webster_overfit_type_normalized"
        ].eq("N")
        return _indices(self.frame, mask)

    @property
    def excluded_n_indices(self) -> frozenset[str]:
        mask = ~self.frame["include_target_pair"].astype(bool) & self.frame[
            "webster_overfit_type_normalized"
        ].eq("N")
        return _indices(self.frame, mask)

FrameInput = pd.DataFrame | str | Path
ConfigurationInput = Mapping[str, object] | str | Path
def is_reference_configuration(
    *, model_name: str, scheduler_name: str, guidance_scale: float,
    num_inference_steps: int, num_seeds: int, seed_start: int = 0,
) -> bool:
    """Return whether CLI values identify the sole selection-building run."""
    try:
        guidance = float(guidance_scale)
        steps = operator.index(num_inference_steps)
        seeds = operator.index(num_seeds)
        first_seed = operator.index(seed_start)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        model_name in _MODEL_DATASET
        and scheduler_name == REFERENCE_SCHEDULER
        and math.isfinite(guidance)
        and guidance == REFERENCE_GUIDANCE_SCALE
        and not isinstance(num_inference_steps, bool)
        and not isinstance(num_seeds, bool)
        and not isinstance(seed_start, bool)
        and steps == REFERENCE_NUM_INFERENCE_STEPS
        and seeds == REFERENCE_NUM_SEEDS
        and first_seed == REFERENCE_SEED_START
    )
def reference_run_name(model_name: str) -> str:
    return generation_run_name(
        _model(model_name),
        REFERENCE_SCHEDULER,
        REFERENCE_GUIDANCE_SCALE,
        REFERENCE_NUM_INFERENCE_STEPS,
        REFERENCE_NUM_SEEDS,
        seed_start=REFERENCE_SEED_START,
    )
def reference_run_path(model_name: str) -> Path:
    return generation_log_relative_path(
        model_name=_model(model_name),
        scheduler_name=REFERENCE_SCHEDULER,
        guidance_scale=REFERENCE_GUIDANCE_SCALE,
        num_inference_steps=REFERENCE_NUM_INFERENCE_STEPS,
        num_seeds=REFERENCE_NUM_SEEDS,
        seed_start=REFERENCE_SEED_START,
    )
def reference_selection_command(model_name: str) -> str:
    model = _model(model_name)
    return (
        f"./generate.sh --model {model} --scheduler ddim --g 7.5 "
        "--T 50 --N 20 --seed-start 20 --downscale 4\n"
        f"./sscd.sh --model {model} --scheduler ddim --g 7.5 "
        "--T 50 --N 20 --seed-start 20\n"
        f"./compute_proximity.sh --model {model} --scheduler ddim --g 7.5 "
        "--T 50 --N 20 --seed-start 20"
    )
def target_pair_selection_directory(root: str | Path, *, model_name: str) -> Path:
    return (
        Path(root).expanduser().resolve()
        / "data/webster/selection"
        / _MODEL_DATASET[_model(model_name)]
        / "reference_S20_N20"
    )
def build_target_pair_selection(
    root: str | Path,
    *,
    model_name: str,
    paired_frame: FrameInput,
    reference_run_config: ConfigurationInput | None = None,
    sscd_config: ConfigurationInput | None = None,
    records_frame: FrameInput | None = None,
) -> TargetPairSelection:
    """Build and freeze a selection from local reference SSCD rows."""
    project_root = Path(root).expanduser().resolve()
    model = _model(model_name)
    run_directory = project_root / reference_run_path(model)
    run_config = _configuration(
        reference_run_config, run_directory / "run_config.json", "generation config"
    )
    score_config = _configuration(
        sscd_config, run_directory / "sscd_config.json", "SSCD config"
    )
    generation_hash = _validate_reference_run(run_config, model)
    sscd_hash = _validate_reference_sscd(score_config, generation_hash)
    paired = _frame(paired_frame, "paired SSCD")
    records = _prompt_records(paired, records_frame, model)
    manifest = _manifest(
        records, _score_map(paired, set(records["original_index"].astype(str))),
        model, generation_hash, sscd_hash,
    )
    config = _selection_config(model, generation_hash, sscd_hash, score_config)
    digest = compute_target_pair_selection_hash(
        manifest, model_name=model, reference_generation_hash=generation_hash,
        reference_sscd_hash=sscd_hash,
    )
    manifest["selection_hash"] = digest
    config["selection_hash"] = digest
    labels = manifest["webster_overfit_type_normalized"]
    tv = labels.eq("TV")
    normal = labels.eq("N")
    selection = TargetPairSelection(
        root=project_root,
        model_name=model,
        frame=manifest.loc[:, SELECTION_COLUMNS].reset_index(drop=True),
        configuration=config,
        sha256=digest,
        diagnostics=build_threshold_diagnostics(
            manifest.loc[tv, "selection_mean_sscd"],
            total_tv_count=int(tv.sum()),
            tv_validation_means=manifest.loc[
                tv, "reference_validation_mean_sscd"
            ],
            n_selection_means=manifest.loc[normal, "selection_mean_sscd"],
            total_n_count=int(normal.sum()),
            n_validation_means=manifest.loc[
                normal, "reference_validation_mean_sscd"
            ],
        ),
    )
    validate_target_pair_selection(selection)
    destination = target_pair_selection_directory(project_root, model_name=model)
    if destination.exists():
        frozen = load_target_pair_selection(project_root, model_name=model)
        if frozen.sha256 != digest and not _matches_frozen_selection_evidence(
            frozen, selection
        ):
            raise FrozenTargetPairSelectionError(
                f"Reference evidence would change frozen selection {destination}: "
                f"{frozen.sha256} != {digest}"
            )
        return frozen
    _write_selection(selection, destination)
    return load_target_pair_selection(project_root, model_name=model)


def _matches_frozen_selection_evidence(
    frozen: TargetPairSelection,
    candidate: TargetPairSelection,
) -> bool:
    """Accept aggregate-hash drift only when frozen decision evidence is exact."""

    if any(
        frozen.configuration.get(key) != candidate.configuration.get(key)
        for key in _FROZEN_COMPATIBILITY_CONFIG_FIELDS
    ):
        return False
    generation_hash = frozen.configuration.get("reference_generation_hash")
    sscd_hash = frozen.configuration.get("reference_sscd_hash")
    if not _is_hash(generation_hash) or not _is_hash(sscd_hash):
        return False
    compatible_digest = compute_target_pair_selection_hash(
        candidate.frame,
        model_name=candidate.model_name,
        reference_generation_hash=str(generation_hash),
        reference_sscd_hash=str(sscd_hash),
    )
    return compatible_digest == frozen.sha256
def ensure_reference_target_pair_selection(
    root: str | Path,
    *,
    model_name: str,
    paired_frame: FrameInput | None = None,
    reference_run_config: ConfigurationInput | None = None,
    sscd_config: ConfigurationInput | None = None,
    records_frame: FrameInput | None = None,
) -> TargetPairSelection:
    """Load a frozen selection, building only when reference rows are supplied."""
    if paired_frame is not None:
        return build_target_pair_selection(
            root, model_name=model_name, paired_frame=paired_frame,
            reference_run_config=reference_run_config, sscd_config=sscd_config,
            records_frame=records_frame,
        )
    if any(item is not None for item in (reference_run_config, sscd_config, records_frame)):
        raise TargetPairSelectionError("paired_frame is required with reference inputs")
    if target_pair_selection_directory(root, model_name=model_name).exists():
        return load_target_pair_selection(root, model_name=model_name)
    _raise_missing(root, model_name)
def load_target_pair_selection(
    root: str | Path, *, model_name: str,
) -> TargetPairSelection:
    """Strictly load one frozen model-specific selection."""
    project_root = Path(root).expanduser().resolve()
    model = _model(model_name)
    directory = target_pair_selection_directory(project_root, model_name=model)
    if not directory.exists():
        _raise_missing(project_root, model)
    _raise_if_stale_selection(directory)
    missing = [name for name in _FILES if not (directory / name).is_file()]
    if missing:
        raise TargetPairSelectionError(
            f"Incomplete frozen selection {directory}; missing: {', '.join(missing)}"
        )
    try:
        frame = pd.read_parquet(directory / "selection.parquet")
        selection_csv = pd.read_csv(
            directory / "selection.csv", dtype={"original_index": str}
        )
        selected_tv_csv = pd.read_csv(
            directory / "selected_tv.csv", dtype={"original_index": str}
        )
        excluded_tv_csv = pd.read_csv(
            directory / "excluded_tv.csv", dtype={"original_index": str}
        )
        selected_n_csv = pd.read_csv(
            directory / "selected_n.csv", dtype={"original_index": str}
        )
        excluded_n_csv = pd.read_csv(
            directory / "excluded_n.csv", dtype={"original_index": str}
        )
        diagnostics = pd.read_csv(directory / "threshold_diagnostics.csv")
        config = read_json(directory / "config.json")
        summary = read_json(directory / "summary.json")
    except (OSError, ValueError, RuntimeError) as error:
        raise TargetPairSelectionError(f"Cannot read selection {directory}: {error}") from error
    digest = config.get("selection_hash")
    if not _is_hash(digest):
        raise TargetPairSelectionError("config.json selection_hash is invalid")
    selection = TargetPairSelection(
        project_root, model, frame, config, str(digest), diagnostics
    )
    validate_target_pair_selection(selection)
    _validate_selection_sidecars(
        frame,
        selection_csv,
        selected_tv_csv,
        excluded_tv_csv,
        selected_n_csv,
        excluded_n_csv,
    )
    if summary != _summary(selection):
        raise TargetPairSelectionError("Frozen selection summary is inconsistent")
    return selection
def validate_target_pair_selection(selection: TargetPairSelection) -> None:
    """Validate the canonical manifest, fixed configuration, and digest."""
    model = _model(selection.model_name)
    config = selection.configuration
    expected = _selection_config(
        model, str(config.get("reference_generation_hash")),
        str(config.get("reference_sscd_hash")), config,
    )
    expected["selection_hash"] = selection.sha256
    missing_config_keys = sorted(set(expected) - set(config))
    extra_config_keys = sorted(set(config) - set(expected))
    if missing_config_keys or extra_config_keys:
        details = []
        if missing_config_keys:
            details.append("missing: " + ", ".join(missing_config_keys))
        if extra_config_keys:
            details.append("unexpected: " + ", ".join(extra_config_keys))
        raise TargetPairSelectionError(
            "Invalid selection config keys; " + "; ".join(details)
        )
    wrong = [key for key, value in expected.items() if config.get(key) != value]
    if wrong:
        raise TargetPairSelectionError("Invalid selection config fields: " + ", ".join(wrong))
    required_hashes = tuple(
        "reference_generation_hash reference_sscd_hash sscd_checkpoint_sha256 "
        "sscd_preprocessing_hash".split()
    )
    if any(not _is_hash(config.get(key)) for key in required_hashes):
        raise TargetPairSelectionError("Selection config contains an invalid provenance hash")
    frame = selection.frame
    missing = sorted(set(SELECTION_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        detail = ", ".join(missing) if missing else "empty manifest"
        raise TargetPairSelectionError(f"Invalid selection schema: {detail}")
    if frame["original_index"].astype(str).duplicated().any():
        raise TargetPairSelectionError("Selection repeats original_index")
    if (
        not pd.api.types.is_bool_dtype(frame["include_target_pair"])
        or frame["include_target_pair"].isna().any()
    ):
        raise TargetPairSelectionError("include_target_pair must contain booleans")
    scalar_expectations = {
        "model_name": model,
        "reference_run_name": reference_run_name(model),
        "reference_generation_hash": config["reference_generation_hash"],
        "reference_sscd_hash": config["reference_sscd_hash"],
        "threshold": TV_MEAN_SSCD_THRESHOLD,
        "selection_policy": SELECTION_POLICY,
        "selection_hash": selection.sha256,
    }
    if any(not frame[key].eq(value).all() for key, value in scalar_expectations.items()):
        raise TargetPairSelectionError("Manifest row provenance is inconsistent")
    normalized = frame["webster_overfit_type_raw"].map(normalize_webster_overfit_type)
    if not normalized.eq(frame["webster_overfit_type_normalized"]).all():
        raise TargetPairSelectionError("Manifest contains an invalid normalized label")
    score_map: dict[str, dict[int, float | None]] = {}
    for row in frame.to_dict(orient="records"):
        index = _index(row["original_index"])
        selection_values = _stored_scores(
            row["selection_seed_values"], "selection_seed_values"
        )
        reference_validation_values = _stored_scores(
            row["reference_validation_seed_values"], "reference_validation_seed_values"
        )
        score_map[index] = {
            **dict(zip(SELECTION_SEEDS, selection_values, strict=True)),
            **dict(zip(REFERENCE_VALIDATION_SEEDS, reference_validation_values, strict=True)),
        }
    rebuilt = _manifest(
        frame.loc[:, _IDENTITY_COLUMNS],
        score_map,
        model,
        str(config["reference_generation_hash"]),
        str(config["reference_sscd_hash"]),
    )
    decision_columns = [
        "original_index", "selection_seed_values", "reference_validation_seed_values",
        "selection_mean_sscd", "selection_median_sscd", "selection_min_sscd",
        "selection_max_sscd", "selection_fraction_ge_0_25",
        "reference_validation_mean_sscd", "reference_validation_median_sscd",
        "comparison_operator", "selection_rule_passes",
        "reference_validation_rule_passes", "selection_validation_agree",
        "include_target_pair", "selection_status", "selection_reason",
        "target_semantics",
    ]
    actual_decisions = frame.loc[:, decision_columns].reset_index(drop=True).copy()
    rebuilt_decisions = rebuilt.loc[:, decision_columns].reset_index(drop=True).copy()
    for column in ("selection_seed_values", "reference_validation_seed_values"):
        actual_decisions[column] = actual_decisions[column].map(
            lambda value, label=column: _stored_scores(value, label)
        )
        rebuilt_decisions[column] = rebuilt_decisions[column].map(
            lambda value, label=column: _stored_scores(value, label)
        )
    try:
        pd.testing.assert_frame_equal(
            actual_decisions,
            rebuilt_decisions,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise TargetPairSelectionError(
            "Manifest decisions or SSCD statistics are inconsistent"
        ) from error
    labels = frame["webster_overfit_type_normalized"]
    tv = labels.eq("TV")
    normal = labels.eq("N")
    expected_diagnostics = build_threshold_diagnostics(
        frame.loc[tv, "selection_mean_sscd"],
        total_tv_count=int(tv.sum()),
        tv_validation_means=frame.loc[tv, "reference_validation_mean_sscd"],
        n_selection_means=frame.loc[normal, "selection_mean_sscd"],
        total_n_count=int(normal.sum()),
        n_validation_means=frame.loc[
            normal, "reference_validation_mean_sscd"
        ],
    )
    actual_diagnostics = selection.diagnostics.reset_index(drop=True).copy()
    rebuilt_diagnostics = expected_diagnostics.copy()
    for column in DIAGNOSTIC_COLUMNS:
        actual_diagnostics[column] = actual_diagnostics[column].map(
            lambda value: math.nan if _missing(value) else value
        )
        rebuilt_diagnostics[column] = rebuilt_diagnostics[column].map(
            lambda value: math.nan if _missing(value) else value
        )
    try:
        pd.testing.assert_frame_equal(
            actual_diagnostics,
            rebuilt_diagnostics,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-15,
        )
    except AssertionError as error:
        raise TargetPairSelectionError(
            "Threshold diagnostics are inconsistent"
        ) from error
    digest = compute_target_pair_selection_hash(
        frame, model_name=model,
        reference_generation_hash=str(config["reference_generation_hash"]),
        reference_sscd_hash=str(config["reference_sscd_hash"]),
    )
    if digest != selection.sha256:
        raise TargetPairSelectionError(f"Selection hash mismatch: {selection.sha256} != {digest}")
def apply_target_pair_selection(
    frame: pd.DataFrame, selection: TargetPairSelection,
) -> pd.DataFrame:
    """Keep every row for included prompts and exclude whole rejected prompts."""
    if "original_index" not in frame:
        raise TargetPairSelectionError("Table is missing original_index")
    result = frame.copy(deep=True)
    indices = result["original_index"].map(_index)
    known = set(selection.frame["original_index"].astype(str))
    unknown = sorted(set(indices) - known)
    if unknown:
        raise TargetPairSelectionError(f"Prompts absent from selection: {', '.join(unknown[:5])}")
    mask = indices.isin(selection.included_indices)
    return result.loc[mask].reset_index(drop=True)
def exact_two_class_otsu(values: Iterable[object]) -> OtsuDiagnostic | None:
    """Compute exact two-class Otsu over sorted values, without histogram bins."""
    ordered = sorted(_finite_values(values))
    if len(ordered) < 2 or ordered[0] == ordered[-1]:
        return None
    total, prefix, best = sum(ordered), 0.0, None
    for position, (lower, upper) in enumerate(zip(ordered, ordered[1:])):
        prefix += lower
        if lower == upper:
            continue
        left, right = position + 1, len(ordered) - position - 1
        score = left * right * (prefix / left - (total - prefix) / right) ** 2
        if best is None or score > best[0]:
            best = score, position
    assert best is not None
    score, position = best
    lower, upper = ordered[position : position + 2]
    return OtsuDiagnostic(
        (lower + upper) / 2, lower, upper, position + 1,
        len(ordered) - position - 1, score / len(ordered) ** 2,
    )
def build_threshold_diagnostics(
    tv_selection_means: Iterable[object],
    *,
    total_tv_count: int | None = None,
    tv_validation_means: Iterable[object] | None = None,
    n_selection_means: Iterable[object] | None = None,
    total_n_count: int | None = None,
    n_validation_means: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Return threshold sensitivity for both category-specific directions."""

    rows = _category_threshold_diagnostics(
        "TV",
        tv_selection_means,
        tv_validation_means,
        total_count=total_tv_count,
    )
    if n_selection_means is not None:
        rows.extend(
            _category_threshold_diagnostics(
                "N",
                n_selection_means,
                n_validation_means,
                total_count=total_n_count,
            )
        )
    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def _category_threshold_diagnostics(
    category: str,
    selection_means: Iterable[object],
    validation_means: Iterable[object] | None,
    *,
    total_count: int | None,
) -> list[dict[str, object]]:
    comparison_operator = _comparison_operator(category)
    if comparison_operator is None:
        raise TargetPairSelectionError(f"No threshold rule for category {category}")
    selection_values = [_number(value) for value in selection_means]
    observed_count = len(selection_values)
    total = observed_count if total_count is None else int(total_count)
    if total < observed_count or total < 0:
        raise TargetPairSelectionError(
            f"total_{category.casefold()}_count is smaller than prompt values"
        )
    selection_values.extend([None] * (total - observed_count))
    if validation_means is None:
        validation_values: list[float | None] = [None] * total
    else:
        validation_values = [_number(value) for value in validation_means]
        if len(validation_values) != observed_count:
            raise TargetPairSelectionError(
                f"{category} validation values do not align with selection values"
            )
        validation_values.extend([None] * (total - observed_count))
    finite_selection = [value for value in selection_values if value is not None]
    finite_validation = [value for value in validation_values if value is not None]
    otsu = exact_two_class_otsu(finite_selection)
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLD_SENSITIVITY:
        selection_passes = [
            _score_passes(category, value, threshold=threshold)
            for value in selection_values
        ]
        validation_passes = [
            _score_passes(category, value, threshold=threshold)
            for value in validation_values
        ]
        comparable = [
            (selection_pass, validation_pass)
            for selection_pass, validation_pass in zip(
                selection_passes, validation_passes, strict=True
            )
            if selection_pass is not None and validation_pass is not None
        ]
        included = sum(value is True for value in selection_passes)
        rows.append(
            {
                "category": category,
                "comparison_operator": comparison_operator,
                "threshold": threshold,
                "included_prompt_count": included,
                "excluded_prompt_count": total - included,
                "scored_prompt_count": len(finite_selection),
                "missing_prompt_count": total - len(finite_selection),
                "validation_scored_prompt_count": len(finite_validation),
                "validation_missing_prompt_count": total - len(finite_validation),
                "selection_validation_agreement_count": sum(
                    left == right for left, right in comparable
                ),
                "selection_validation_disagreement_count": sum(
                    left != right for left, right in comparable
                ),
                "is_fixed_threshold": threshold == REFERENCE_SSCD_BOUNDARY,
                "otsu_threshold": None if otsu is None else otsu.threshold,
                "otsu_lower_neighbor": None if otsu is None else otsu.lower_neighbor,
                "otsu_upper_neighbor": None if otsu is None else otsu.upper_neighbor,
                "otsu_lower_count": None if otsu is None else otsu.lower_count,
                "otsu_upper_count": None if otsu is None else otsu.upper_count,
            }
        )
    return rows
def compute_target_pair_selection_hash(
    frame: pd.DataFrame,
    *,
    model_name: str,
    reference_generation_hash: str,
    reference_sscd_hash: str,
    threshold: float = TV_MEAN_SSCD_THRESHOLD,
) -> str:
    """Hash policy inputs and selection-half evidence, never outcome metrics."""
    model = _model(model_name)
    if not _is_hash(reference_generation_hash) or not _is_hash(reference_sscd_hash):
        raise TargetPairSelectionError("Selection provenance hash is invalid")
    fields = (
        "original_index", "record_id", "source_row_number", "prompt_raw",
        "target_image_sha256", "webster_overfit_type_raw",
        "webster_overfit_type_normalized", "selection_seed_values",
        "comparison_operator", "selection_rule_passes",
        "include_target_pair", "selection_status", "selection_reason", "target_semantics",
    )
    missing = sorted(set(fields) - set(frame.columns))
    if missing:
        raise TargetPairSelectionError("Cannot hash without: " + ", ".join(missing))
    rows = [
        {field: _json(row.get(field)) for field in fields}
        for row in frame.sort_values(
            ["source_row_number", "original_index"], kind="stable"
        ).to_dict(orient="records")
    ]
    return canonical_hash({
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_policy": SELECTION_POLICY,
        "model_name": model,
        "dataset_model_name": _MODEL_DATASET[model],
        "reference_run_name": reference_run_name(model),
        "reference_generation_hash": reference_generation_hash,
        "reference_sscd_hash": reference_sscd_hash,
        "boundary": float(threshold),
        "threshold": float(threshold),
        "category_rules": _category_rules(boundary=float(threshold)),
        "selection_seeds": list(SELECTION_SEEDS),
        "reference_validation_seeds": list(REFERENCE_VALIDATION_SEEDS),
        "rows": rows,
    })
def normalize_webster_overfit_type(value: object) -> str:
    """Apply the repository's canonical Webster category normalization."""

    # Keep the selection API stable while avoiding a module-load dependency
    # between the dataset and frozen-selection implementations.
    from utils.data.webster import normalize_webster_type

    return normalize_webster_type(value)


def _category_rules(
    *, boundary: float = REFERENCE_SSCD_BOUNDARY,
) -> dict[str, dict[str, object]]:
    return {
        "TV": {
            "selection_metric": "mean_reference_sscd",
            "comparison_operator": ">=",
            "boundary": boundary,
            "include_when": "mean_reference_sscd >= boundary",
        },
        "N": {
            "selection_metric": "mean_reference_sscd",
            "comparison_operator": ">=",
            "boundary": boundary,
            "include_when": "mean_reference_sscd >= boundary",
        },
        "MV": {"decision": "unconditional_include_preserved"},
        "RV": {"decision": "unconditional_include_preserved"},
        "UNKNOWN": {"decision": "unconditional_include_preserved"},
    }


def _comparison_operator(label: str) -> str | None:
    return {"TV": ">=", "N": ">="}.get(label)


def _score_passes(
    label: str,
    value: float | None,
    *,
    threshold: float = REFERENCE_SSCD_BOUNDARY,
) -> bool | None:
    if value is None:
        return None
    if label in {"TV", "N"}:
        return value >= threshold
    return None


def _manifest(
    records: pd.DataFrame, scores: Mapping[str, Mapping[int, float | None]], model: str,
    generation_hash: str, sscd_hash: str,
) -> pd.DataFrame:
    rows = []
    for record in records.to_dict(orient="records"):
        index = _index(record["original_index"])
        raw = _scalar(record["webster_overfit_type_raw"])
        label = normalize_webster_overfit_type(raw)
        by_seed = scores.get(index, {})
        selection_values = [_number(by_seed.get(seed)) for seed in SELECTION_SEEDS]
        reference_validation_values = [_number(by_seed.get(seed)) for seed in REFERENCE_VALIDATION_SEEDS]
        selection_stats, reference_validation_stats = _stats(selection_values), _stats(reference_validation_values)
        comparison_operator = _comparison_operator(label)
        selection_rule_passes = _score_passes(
            label, None if selection_stats is None else selection_stats[0]
        )
        reference_validation_rule_passes = _score_passes(
            label,
            None if reference_validation_stats is None else reference_validation_stats[0],
        )
        selection_validation_agree = (
            None
            if selection_rule_passes is None
            or reference_validation_rule_passes is None
            else selection_rule_passes == reference_validation_rule_passes
        )
        decision = _decision(_valid_record(record), label, selection_stats)
        rows.append({
            "model_name": model, "original_index": index,
            "record_id": _scalar(record["record_id"]), "source_row_number": _scalar(record["source_row_number"]),
            "prompt_raw": _scalar(record["prompt_raw"]),
            "target_image_sha256": _scalar(record["target_image_sha256"]),
            "webster_overfit_type_raw": raw, "webster_overfit_type_normalized": label,
            "reference_run_name": reference_run_name(model),
            "reference_generation_hash": generation_hash, "reference_sscd_hash": sscd_hash,
            "selection_seed_values": selection_values, "reference_validation_seed_values": reference_validation_values,
            "selection_mean_sscd": _stat(selection_stats, 0), "selection_median_sscd": _stat(selection_stats, 1),
            "selection_min_sscd": _stat(selection_stats, 2), "selection_max_sscd": _stat(selection_stats, 3),
            "selection_fraction_ge_0_25": (
                math.nan if selection_stats is None else
                sum(value >= TV_MEAN_SSCD_THRESHOLD for value in selection_values) / 10
            ),
            "reference_validation_mean_sscd": _stat(reference_validation_stats, 0), "reference_validation_median_sscd": _stat(reference_validation_stats, 1),
            "comparison_operator": comparison_operator,
            "selection_rule_passes": selection_rule_passes,
            "reference_validation_rule_passes": reference_validation_rule_passes,
            "selection_validation_agree": selection_validation_agree,
            "include_target_pair": decision[0], "selection_status": decision[1],
            "selection_reason": decision[2], "target_semantics": decision[3],
            "threshold": TV_MEAN_SSCD_THRESHOLD, "selection_policy": SELECTION_POLICY,
            "selection_hash": "",
        })
    frame = pd.DataFrame(rows, columns=SELECTION_COLUMNS).sort_values(
        ["source_row_number", "original_index"], kind="stable", ignore_index=True
    )
    for column in (
        "selection_rule_passes",
        "reference_validation_rule_passes",
        "selection_validation_agree",
    ):
        frame[column] = frame[column].astype("boolean")
    return frame
def _decision(valid: bool, label: str, stats: tuple[float, float, float, float] | None,
              ) -> tuple[bool, str, str, str | None]:
    if not valid:
        return False, INVALID_RECORD, "invalid_prompt_or_target", None
    if label not in {"TV", "N"}:
        return True, INCLUDED_NON_TV, "non_tv_current_policy", "current_non_tv_policy"
    if stats is None:
        semantics = (
            "unsupported_template_target"
            if label == "TV"
            else "n_target_support_unresolved"
        )
        return False, MISSING_REFERENCE_SSCD, "reference_scores_unavailable", semantics
    if label == "TV" and stats[0] >= REFERENCE_SSCD_BOUNDARY:
        return True, INCLUDED_TV_TARGET_SUPPORTED, "tv_selection_mean_sscd_ge_0_25", "empirically_singleton_compatible_tv"
    if label == "TV":
        return False, EXCLUDED_TV_TARGET_UNSUPPORTED, "tv_selection_mean_sscd_lt_0_25", "unsupported_template_target"
    if stats[0] >= REFERENCE_SSCD_BOUNDARY:
        return True, INCLUDED_N_TARGET_SUPPORTED, "n_selection_mean_sscd_ge_0_25", "n_target_supported_under_frozen_reference_criterion"
    return (
        False,
        EXCLUDED_N_TARGET_UNSUPPORTED,
        "n_selection_mean_sscd_lt_0_25",
        "n_target_unsupported_under_frozen_reference_criterion",
    )
def _prompt_records(paired: pd.DataFrame, source: FrameInput | None,
                    model: str) -> pd.DataFrame:
    frame = paired.copy(deep=True) if source is None else _frame(source, "records")
    if "webster_overfit_type_raw" not in frame:
        if "webster_overfit_type" not in frame:
            raise TargetPairSelectionError("Prompt identity lacks webster_overfit_type_raw")
        frame["webster_overfit_type_raw"] = frame["webster_overfit_type"]
    missing = sorted(set(_IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        suffix = " or provide records_frame" if source is None else ""
        raise TargetPairSelectionError("Prompt identity lacks: " + ", ".join(missing) + suffix)
    if "model_name" in frame:
        observed = {str(value) for value in frame["model_name"] if not _missing(value)}
        if not observed or not observed.issubset({model, _MODEL_DATASET[model]}):
            raise TargetPairSelectionError(f"Prompt records do not belong to {model}")
    frame = frame.loc[:, _IDENTITY_COLUMNS].copy()
    frame["original_index"] = frame["original_index"].map(_index)
    if source is None:
        rows = []
        for index, group in frame.groupby("original_index", sort=False):
            row = {"original_index": index}
            for field in _IDENTITY_COLUMNS[1:]:
                values = [_scalar(value) for value in group[field]]
                if len({repr(value) for value in values}) != 1:
                    raise TargetPairSelectionError(f"Identity differs for {index}: {field}")
                row[field] = values[0]
            rows.append(row)
        frame = pd.DataFrame(rows, columns=_IDENTITY_COLUMNS)
    elif frame["original_index"].duplicated().any():
        raise TargetPairSelectionError("records_frame must have one row per prompt")
    return frame.sort_values(
        ["source_row_number", "original_index"], kind="stable", ignore_index=True
    )
def _score_map(paired: pd.DataFrame,
               known: set[str]) -> dict[str, dict[int, float | None]]:
    required = {"original_index", "seed", "sscd_cosine_similarity"}
    missing = sorted(required - set(paired.columns))
    if missing:
        raise TargetPairSelectionError("paired SSCD lacks: " + ", ".join(missing))
    result: dict[str, dict[int, float | None]] = {}
    for row in paired.loc[:, list(required)].to_dict(orient="records"):
        index, seed = _index(row["original_index"]), _int(row["seed"], "seed")
        if index not in known:
            raise TargetPairSelectionError(f"Unknown paired prompt {index}")
        if seed not in REFERENCE_SEEDS:
            raise TargetPairSelectionError(f"Reference seed outside 20--39: {seed}")
        if seed in result.setdefault(index, {}):
            raise TargetPairSelectionError(f"Duplicate score for {index}, seed {seed}")
        result[index][seed] = _number(row["sscd_cosine_similarity"])
    return result
def _selection_config(model: str, generation_hash: str, sscd_hash: str,
                      sscd: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": SELECTION_SCHEMA_VERSION, "selection_policy": SELECTION_POLICY,
        "model_name": model, "dataset_model_name": _MODEL_DATASET[model],
        "reference_run_name": reference_run_name(model),
        "reference_run_path": reference_run_path(model).as_posix(),
        "reference_scheduler": REFERENCE_SCHEDULER, "reference_guidance_scale": REFERENCE_GUIDANCE_SCALE,
        "reference_num_inference_steps": REFERENCE_NUM_INFERENCE_STEPS,
        "reference_seed_start": REFERENCE_SEED_START,
        "reference_num_seeds": REFERENCE_NUM_SEEDS,
        "reference_seeds": list(REFERENCE_SEEDS),
        "selection_seeds": list(SELECTION_SEEDS),
        "reference_validation_seeds": list(REFERENCE_VALIDATION_SEEDS),
        "boundary": REFERENCE_SSCD_BOUNDARY,
        "threshold": REFERENCE_SSCD_BOUNDARY,
        "category_rules": _category_rules(),
        "decision_scope": "prompt_all_experiment_seeds",
        "selection_metric": "mean_sscd_over_selection_seeds",
        "reference_validation_values_affect_selection": False, "proximity_affects_selection": False,
        "correlation_affects_selection": False,
        "reference_generation_hash": generation_hash, "reference_sscd_hash": sscd_hash,
        "sscd_checkpoint_sha256": sscd.get("sscd_checkpoint_sha256"), "sscd_preprocessing_hash": sscd.get("sscd_preprocessing_hash"),
        "selection_hash": "",
    }
def _validate_reference_run(config: Mapping[str, object], model: str) -> str:
    science, digest = config.get("scientific_config"), config.get("scientific_config_hash")
    if not isinstance(science, Mapping) or not _is_hash(digest) or canonical_hash(science) != digest:
        raise TargetPairSelectionError("Reference generation scientific hash is invalid")
    scheduler = science.get("scheduler")
    scheduler = scheduler.get("name") if isinstance(scheduler, Mapping) else scheduler
    expected = {
        "model_cli_name": model,
        "dataset_model": _MODEL_DATASET[model],
        "guidance_scale": REFERENCE_GUIDANCE_SCALE,
        "num_inference_steps": REFERENCE_NUM_INFERENCE_STEPS,
        "num_seeds": REFERENCE_NUM_SEEDS,
        "seeds": list(REFERENCE_SEEDS),
    }
    wrong = [key for key, value in expected.items() if science.get(key) != value]
    if scheduler != REFERENCE_SCHEDULER:
        wrong.append("scheduler")
    if wrong:
        raise TargetPairSelectionError("Reference generation differs at: " + ", ".join(wrong))
    return str(digest)
def _validate_reference_sscd(config: Mapping[str, object], generation_hash: str) -> str:
    from utils.experiments.sscd import sscd_configuration_hash

    digest = config.get("configuration_hash")
    if not _is_hash(digest) or sscd_configuration_hash(config) != digest:
        raise TargetPairSelectionError("Reference SSCD configuration hash is invalid")
    expected = {
        "generation_scientific_config_hash": generation_hash,
        "num_seeds": REFERENCE_NUM_SEEDS,
        "seeds": list(REFERENCE_SEEDS),
    }
    wrong = [key for key, value in expected.items() if config.get(key) != value]
    if wrong:
        raise TargetPairSelectionError("Reference SSCD differs at: " + ", ".join(wrong))
    if any(not _is_hash(config.get(key)) for key in (
        "sscd_checkpoint_sha256", "sscd_preprocessing_hash"
    )):
        raise TargetPairSelectionError("Reference SSCD provenance hash is invalid")
    return str(digest)
def _write_selection(selection: TargetPairSelection, directory: Path) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists() or directory.is_symlink():
        raise FrozenTargetPairSelectionError(
            f"Refusing to replace frozen selection directory: {directory}"
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.",
            suffix=".tmp",
            dir=directory.parent,
        )
    )
    try:
        frame = selection.frame
        tv = frame["webster_overfit_type_normalized"].eq("TV")
        normal = frame["webster_overfit_type_normalized"].eq("N")
        included = frame["include_target_pair"].astype(bool)
        atomic_write_frame_parquet(frame, temporary / "selection.parquet")
        atomic_write_frame_csv(frame, temporary / "selection.csv")
        atomic_write_frame_csv(frame.loc[tv & included], temporary / "selected_tv.csv")
        atomic_write_frame_csv(frame.loc[tv & ~included], temporary / "excluded_tv.csv")
        atomic_write_frame_csv(frame.loc[normal & included], temporary / "selected_n.csv")
        atomic_write_frame_csv(frame.loc[normal & ~included], temporary / "excluded_n.csv")
        atomic_write_frame_csv(
            selection.diagnostics, temporary / "threshold_diagnostics.csv"
        )
        atomic_write_json(temporary / "summary.json", _summary(selection))
        atomic_write_json(temporary / "config.json", selection.configuration)
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, directory)
        parent_descriptor = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
def _summary(selection: TargetPairSelection) -> dict[str, object]:
    frame = selection.frame
    tv = frame["webster_overfit_type_normalized"].eq("TV")
    normal = frame["webster_overfit_type_normalized"].eq("N")
    included = frame["include_target_pair"].astype(bool)
    selected_tv = frame.loc[tv & included]
    selected_n = frame.loc[normal & included]
    tv_validation = pd.to_numeric(
        selected_tv["reference_validation_mean_sscd"], errors="coerce"
    )
    n_validation = pd.to_numeric(
        selected_n["reference_validation_mean_sscd"], errors="coerce"
    )
    otsu = exact_two_class_otsu(frame.loc[tv, "selection_mean_sscd"])
    return {
        "schema_version": SELECTION_SCHEMA_VERSION, "selection_policy": SELECTION_POLICY,
        "model_name": selection.model_name, "selection_hash": selection.sha256,
        "boundary": REFERENCE_SSCD_BOUNDARY,
        "threshold": REFERENCE_SSCD_BOUNDARY, "total_prompt_count": len(frame),
        "included_prompt_count": int(included.sum()), "excluded_prompt_count": int((~included).sum()),
        "non_tv_prompt_count": int((~tv).sum()),
        "unconditional_category_prompt_count": int((~tv & ~normal).sum()),
        "unconditionally_included_category_prompt_count": int(
            (~tv & ~normal & included).sum()
        ),
        "tv_prompt_count": int(tv.sum()), "n_prompt_count": int(normal.sum()),
        "included_tv_prompt_count": len(selected_tv),
        "selected_tv_prompt_count": len(selected_tv),
        "excluded_tv_prompt_count": int((tv & ~included).sum()),
        "included_n_prompt_count": len(selected_n),
        "selected_n_prompt_count": len(selected_n),
        "excluded_n_prompt_count": int((normal & ~included).sum()),
        "tv_selection_validation_agreement_count": int(
            frame.loc[tv, "selection_validation_agree"].eq(True).sum()
        ),
        "tv_selection_validation_disagreement_count": int(
            frame.loc[tv, "selection_validation_agree"].eq(False).sum()
        ),
        "n_selection_validation_agreement_count": int(
            frame.loc[normal, "selection_validation_agree"].eq(True).sum()
        ),
        "n_selection_validation_disagreement_count": int(
            frame.loc[normal, "selection_validation_agree"].eq(False).sum()
        ),
        "status_counts": {
            str(key): int(value) for key, value in
            frame["selection_status"].value_counts().sort_index().items()
        },
        "otsu_threshold": None if otsu is None else otsu.threshold,
        "otsu_lower_neighbor": None if otsu is None else otsu.lower_neighbor, "otsu_upper_neighbor": None if otsu is None else otsu.upper_neighbor,
        "selected_tv_complete_reference_validation_count": int(tv_validation.notna().sum()),
        "selected_tv_reference_validation_mean_ge_0_25_count": int(tv_validation.ge(0.25).sum()),
        "all_selected_tv_reference_validation_mean_ge_0_25": bool(
            len(selected_tv) == tv_validation.notna().sum() and tv_validation.ge(0.25).all()
        ),
        "selected_n_complete_reference_validation_count": int(n_validation.notna().sum()),
        "selected_n_reference_validation_mean_ge_0_25_count": int(n_validation.ge(0.25).sum()),
        "all_selected_n_reference_validation_mean_ge_0_25": bool(
            len(selected_n) == n_validation.notna().sum() and n_validation.ge(0.25).all()
        ),
    }
def _configuration(source: ConfigurationInput | None, default: Path,
                   label: str) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = default if source is None else Path(source)
    if not path.is_file():
        raise TargetPairSelectionError(f"Missing {label}: {path}")
    try:
        return read_json(path)
    except RuntimeError as error:
        raise TargetPairSelectionError(f"Invalid {label}: {error}") from error
def _frame(source: FrameInput, label: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    path = Path(source)
    if not path.is_file():
        raise TargetPairSelectionError(f"Missing {label}: {path}")
    try:
        if path.suffix.casefold() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.casefold() == ".csv":
            return pd.read_csv(path, dtype={"original_index": str})
    except (OSError, ValueError) as error:
        raise TargetPairSelectionError(f"Cannot read {label} {path}: {error}") from error
    raise TargetPairSelectionError(f"{label} must be a DataFrame, .csv, or .parquet")
def _stats(values: Sequence[float | None]) -> tuple[float, float, float, float] | None:
    if len(values) != 10 or any(value is None for value in values):
        return None
    numbers = [float(value) for value in values if value is not None]
    return fmean(numbers), median(numbers), min(numbers), max(numbers)
def _stored_scores(value: object, label: str) -> list[float | None]:
    values = _json(value)
    if not isinstance(values, list) or len(values) != 10:
        raise TargetPairSelectionError(f"{label} must contain exactly 10 scores")
    return [_number(item) for item in values]
def _validate_selection_sidecars(
    frame: pd.DataFrame,
    selection_csv: pd.DataFrame,
    selected_tv_csv: pd.DataFrame,
    excluded_tv_csv: pd.DataFrame,
    selected_n_csv: pd.DataFrame,
    excluded_n_csv: pd.DataFrame,
) -> None:
    tv = frame["webster_overfit_type_normalized"].eq("TV")
    normal = frame["webster_overfit_type_normalized"].eq("N")
    included = frame["include_target_pair"]
    comparisons = (
        (selection_csv, frame, "selection.csv"),
        (selected_tv_csv, frame.loc[tv & included], "selected_tv.csv"),
        (excluded_tv_csv, frame.loc[tv & ~included], "excluded_tv.csv"),
        (selected_n_csv, frame.loc[normal & included], "selected_n.csv"),
        (excluded_n_csv, frame.loc[normal & ~included], "excluded_n.csv"),
    )
    for observed, expected, label in comparisons:
        _assert_selection_sidecar(observed, expected, label)


def _assert_selection_sidecar(
    observed: pd.DataFrame, expected: pd.DataFrame, label: str,
) -> None:
    if tuple(observed.columns) != SELECTION_COLUMNS:
        raise TargetPairSelectionError(f"{label} schema differs from selection.parquet")
    actual = _normalize_selection_sidecar(observed, label)
    canonical = _normalize_selection_sidecar(expected, "selection.parquet")
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            canonical.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-14,
            atol=1e-15,
        )
    except AssertionError as error:
        raise TargetPairSelectionError(
            f"{label} differs from selection.parquet"
        ) from error


def _normalize_selection_sidecar(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.loc[:, SELECTION_COLUMNS].copy()
    for column in ("selection_seed_values", "reference_validation_seed_values"):
        def parse_scores(value: object, *, field: str = column) -> list[float | None]:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as error:
                    raise TargetPairSelectionError(
                        f"{label} {field} is not valid JSON"
                    ) from error
            return _stored_scores(value, field)

        result[column] = result[column].map(parse_scores)
    if len(result) and not pd.api.types.is_bool_dtype(result["include_target_pair"]):
        raise TargetPairSelectionError(
            f"{label} include_target_pair must contain booleans"
        )
    result["include_target_pair"] = result["include_target_pair"].astype(bool)
    numeric = {
        "source_row_number", "selection_mean_sscd", "selection_median_sscd",
        "selection_min_sscd", "selection_max_sscd",
        "selection_fraction_ge_0_25", "reference_validation_mean_sscd",
        "reference_validation_median_sscd", "threshold",
    }
    for column in numeric:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise TargetPairSelectionError(
                f"{label} {column} must be numeric"
            ) from error
    for column in set(SELECTION_COLUMNS) - numeric - {
        "selection_seed_values", "reference_validation_seed_values", "include_target_pair",
    }:
        result[column] = result[column].map(
            lambda value: "" if _missing(value) else str(value)
        )
    return result
def _valid_record(row: Mapping[str, object]) -> bool:
    try:
        source_row = _int(row.get("source_row_number"), "source_row_number")
    except TargetPairSelectionError:
        return False
    return (
        source_row >= 0
        and isinstance(row.get("record_id"), str) and bool(str(row["record_id"]).strip())
        and isinstance(row.get("prompt_raw"), str)
        and _is_hash(row.get("target_image_sha256"))
    )
def _finite_values(values: Iterable[object]) -> list[float]:
    return [number for number in (_number(value) for value in values) if number is not None]
def _number(value: object) -> float | None:
    if _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetPairSelectionError(f"SSCD score is not numeric: {value!r}") from error
    return number if math.isfinite(number) else None
def _stat(stats: tuple[float, ...] | None, position: int) -> float:
    return math.nan if stats is None else stats[position]
def _int(value: object, label: str) -> int:
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
def _model(value: object) -> str:
    if not isinstance(value, str) or value not in _MODEL_DATASET:
        raise TargetPairSelectionError(
            f"Unknown model {value!r}; expected one of: {', '.join(_MODEL_DATASET)}"
        )
    return value
def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None
def _indices(frame: pd.DataFrame, mask: pd.Series) -> frozenset[str]:
    return frozenset(frame.loc[mask, "original_index"].astype(str))
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
    if hasattr(value, "item"):
        return value.item()
    return value
def _json(value: object) -> object:
    value = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json(item) for item in value]
    return _scalar(value)
def _raise_missing(root: str | Path, model_name: str) -> None:
    model = _model(model_name)
    directory = target_pair_selection_directory(root, model_name=model)
    legacy_directory = directory.parent
    legacy_note = ""
    if (legacy_directory / "config.json").is_file():
        legacy_note = (
            f"\nPreserved schema-1 selection is incompatible: {legacy_directory}"
        )
    raise TargetPairSelectionMissingError(
        f"Frozen schema-{SELECTION_SCHEMA_VERSION} target-pair selection is missing: {directory}\n"
        f"Required reference run: {reference_run_path(model).as_posix()}\n"
        f"Create it with:\n{reference_selection_command(model)}"
        f"{legacy_note}"
    )


def _raise_if_stale_selection(directory: Path) -> None:
    """Reject prior scientific policies before reporting current files missing."""

    config_path = directory / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        return
    try:
        config = read_json(config_path)
    except RuntimeError:
        return
    if not isinstance(config, Mapping):
        return
    stale_schemas = {2, 3}
    stale_policies = {
        "target_pair_selection",
        "target_pair_selection_tv_ge_0_25_n_lt_0_25",
    }
    if not (
        config.get("schema_version") in stale_schemas
        or config.get("selection_policy") in stale_policies
    ):
        return
    raise StaleTargetPairSelectionError(
        "Stale prior-policy frozen target-pair selection is incompatible with "
        f"schema {SELECTION_SCHEMA_VERSION} and policy {SELECTION_POLICY}.\n"
        "Archive or remove only this exact derived selection directory before "
        f"rebuilding:\n- {directory}\n"
        "Preserve generation trajectories, noise predictions, target latents, "
        "generated previews, and SSCD tensors."
    )
