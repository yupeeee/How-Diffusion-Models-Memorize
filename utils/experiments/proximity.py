"""Build or apply frozen prompt selection from generation and SSCD caches."""

from __future__ import annotations

import hashlib
import io
import math
import os
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from utils.common.cli import generation_run_name, stable_float
from utils.common.io import (
    CacheIOError,
    atomic_write_bytes,
    atomic_write_frame_csv,
    atomic_write_json,
    canonical_json,
    file_sha256,
    read_json,
    safe_torch_load,
    utc_now,
)
from utils.data.selection import (
    DEFAULT_SELECTION_STRATEGY,
    DISCARDED_PROXIMITY_RULE,
    INCLUDED_PROXIMITY_RULE,
    TargetPairSelectionError,
    normalize_selection_strategy,
    reference_completion_fingerprint,
)
from utils.data.webster import normalize_webster_type
from utils.models.latent import compute_latent_distances

from .cache import (
    CompletedGenerationRecord,
    GenerationCacheError,
    GenerationPaths,
    generation_log_relative_path,
    generation_paths,
    list_completed_records,
    require_generation_run,
    validate_generation_record,
)
from .figure_paths import publication_directory
from .plotting import (
    AnalysisStatistics,
    EXPERIMENT_SPEARMAN_COLUMN,
    FIGURE_DPI,
    PROXIMITY_FIGURES,
    PROXIMITY_PDF_FIGURES,
    PROXIMITY_PDF_FILENAMES,
    write_analysis_outputs,
    write_gmm_fit_figure,
    write_saved_analysis_figures,
    write_selection_figure,
)

OBSERVATION_COLUMNS = tuple(
    "model_name original_index record_id source_row_number seed prompt kind "
    "target_image_sha256 generated_image_path generated_image_tile_index "
    "l2_norm sscd observation_status observation_error".split()
)
ANALYSIS_COLUMNS = OBSERVATION_COLUMNS + tuple(
    "selection_strategy prompt_spearman include_prompt selection_status "
    "selection_reason".split()
)
FAILED_COLUMNS = tuple(
    "original_index record_id source_row_number issue_type reason "
    "generation_record_path sscd_record_path".split()
)


class ProximityError(RuntimeError):
    """The local generation, SSCD, or frozen selection cache is unusable."""


@dataclass(frozen=True, slots=True)
class ProximityPaths:
    project_root: Path
    generation_run: Path
    output_directory: Path
    configuration_filename: str = "run_config.json"

    @classmethod
    def build(
        cls,
        root: Path,
        generation_run: str | Path,
        *,
        output_run_name: str | None = None,
        role: str,
        seed_start: int,
        num_seeds: int,
        selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
    ) -> "ProximityPaths":
        generation = Path(generation_run)
        if not generation.is_absolute():
            generation = root / "logs" / generation
        if output_run_name is None:
            try:
                output_run_name = generation.relative_to(root / "logs").parts[0]
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "output_run_name is required outside the logs directory"
                ) from error
        namespace = f"{role}_S{seed_start}_N{num_seeds}"
        normalize_selection_strategy(selection_strategy)
        output = root / "outputs" / output_run_name / "proximity" / namespace
        return cls(root, generation, output)

    @classmethod
    def frozen_selection(
        cls, root: Path, generation_run: Path, selection_directory: Path
    ) -> "ProximityPaths":
        return cls(root, generation_run, selection_directory, "config.json")

    @property
    def figure_directory(self) -> Path:
        return publication_directory(self.output_directory)

    @property
    def failed_csv(self) -> Path:
        return self.output_directory / "failed.csv"

    @property
    def run_config_json(self) -> Path:
        return self.output_directory / self.configuration_filename

    @property
    def summary_json(self) -> Path:
        return self.output_directory / "summary.json"


@dataclass(frozen=True, slots=True)
class ProximitySummary:
    paths: ProximityPaths
    values: Mapping[str, object]

    @property
    def exit_code(self) -> int:
        return 0 if bool(self.values.get("complete")) else 1


@dataclass(frozen=True, slots=True)
class _GenerationRecord:
    metadata: Mapping[str, object]
    completed: CompletedGenerationRecord
    paths: GenerationPaths
    generated_image_path: str


def _record_progress(values: Sequence[Any]) -> Iterable[Any]:
    return tqdm(
        values,
        total=len(values),
        desc="[Proximity] Records",
        unit="record",
        dynamic_ncols=True,
        leave=True,
        disable=False,
    )


def _set_record_progress(
    progress: Any,
    completed: CompletedGenerationRecord,
    status: str,
) -> None:
    record_id = completed.metadata.get("record_id") or completed.original_index
    progress.set_postfix(record=record_id, status=status, refresh=True)


def _seed_role(
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
) -> str:
    from utils.data.selection import is_reference_configuration

    if is_reference_configuration(
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        seed_start=seed_start,
    ):
        return "reference"
    if (
        isinstance(num_seeds, int)
        and not isinstance(num_seeds, bool)
        and num_seeds > 0
        and seed_start == 0
    ):
        return "experiment"
    raise ProximityError(
        "seed range must be the first N seeds (--seed-start 0) for the "
        "experiment, or the next N seeds (--seed-start N) for the matching "
        "selection reference; N must be positive"
    )


def run_proximity(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
    overwrite: bool,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
) -> ProximitySummary:
    """Build on reference seeds N..2N-1 or report a frozen experiment."""

    started = time.monotonic()
    root = Path(project_root).expanduser().resolve()
    if not isinstance(overwrite, bool):
        raise ProximityError("overwrite must be a boolean")
    strategy = normalize_selection_strategy(selection_strategy)
    arguments = {
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "seed_start": seed_start,
    }
    role = _seed_role(**arguments)
    cache = generation_paths(root, **arguments)
    run_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
    )
    parent_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start=0,
    )
    paths = ProximityPaths.build(
        root,
        cache.run_directory,
        output_run_name=parent_name,
        role=role,
        seed_start=seed_start,
        num_seeds=num_seeds,
        selection_strategy=strategy,
    )

    if role == "reference" and not overwrite:
        frozen = _frozen_reference_result(
            root,
            cache.run_directory,
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
            strategy,
        )
        if frozen is not None:
            return frozen

    generation_command = _generation_command(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
    )
    if not cache.run_config.is_file() or cache.run_config.is_symlink():
        raise ProximityError(
            f"required generation configuration is absent: {cache.run_config}\n"
            f"Create it with:\n{generation_command}"
        )
    generation = require_generation_run(cache)
    _validate_generation_invocation(generation, **arguments)
    sscd_path = cache.run_directory / "sscd_config.json"
    if not sscd_path.is_file() or sscd_path.is_symlink():
        raise ProximityError(
            f"required cached SSCD configuration is absent: {sscd_path}\n"
            f"Create it with:\n{
                _sscd_command(
                    model_name,
                    scheduler_name,
                    guidance_scale,
                    num_inference_steps,
                    num_seeds,
                    seed_start,
                )
            }"
        )
    sscd = _load_sscd_config(cache.run_directory, generation)

    selection = None
    if role == "experiment":
        from utils.data.selection import (
            TargetPairSelectionMissingError,
            load_target_pair_selection,
            target_pair_selection_directory,
        )

        try:
            selection = load_target_pair_selection(
                root,
                model_name=model_name,
                scheduler_name=scheduler_name,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_seeds=num_seeds,
                selection_strategy=strategy,
            )
        except TargetPairSelectionMissingError:
            raise
        except TargetPairSelectionError as error:
            directory = target_pair_selection_directory(
                root,
                model_name=model_name,
                scheduler_name=scheduler_name,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_seeds=num_seeds,
                selection_strategy=strategy,
            )
            raise _stale_selection_error(
                directory,
                model_name,
                scheduler_name,
                guidance_scale,
                num_inference_steps,
                num_seeds,
                strategy,
                str(error),
            ) from error

    records = list_completed_records(cache)
    if not records:
        raise ProximityError(
            f"no completed generation records found in {cache.record_directory}\n"
            f"Create them with:\n{generation_command}"
        )
    _require_generation_coverage(cache, records)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    progress = _record_progress(records)
    for completed in progress:
        _set_record_progress(progress, completed, "processing")
        try:
            record = _validated_record(root, cache, completed, generation)
            l2 = _terminal_l2(record)
            scores = _sscd_scores(root, record, generation, sscd)
            rows.extend(_paired_rows(record, l2, scores, model_name=model_name))
            _set_record_progress(progress, completed, "complete")
        except (
            CacheIOError,
            GenerationCacheError,
            KeyError,
            OSError,
            ProximityError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            if role == "reference":
                try:
                    failed_rows = _failed_reference_rows(
                        root,
                        cache,
                        completed,
                        generation,
                        error,
                        model_name=model_name,
                    )
                except Exception:
                    _set_record_progress(progress, completed, "failed")
                    raise
                rows.extend(failed_rows)
                _set_record_progress(progress, completed, "unusable")
            else:
                failures.append(_failure_row(completed, error, cache.run_directory))
                _set_record_progress(progress, completed, "failed")

    observations = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).sort_values(
        ["source_row_number", "original_index", "seed"],
        kind="stable",
        ignore_index=True,
    )
    if role == "reference":
        from utils.data.selection import build_target_pair_selection

        build_target_pair_selection(
            root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            overwrite=overwrite,
            selection_strategy=strategy,
            paired_frame=observations,
            records_frame=_reference_records_frame(records, model_name),
            reference_run_config=generation,
            sscd_config=sscd,
        )
        result = _frozen_reference_result(
            root,
            cache.run_directory,
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
            strategy,
        )
        assert result is not None
        return result

    assert selection is not None
    analysis = _annotate_selection(
        observations, selection, require_complete=not failures
    )
    configuration = _analysis_configuration(
        paths, run_name, generation, sscd, selection
    )
    _write_configuration(paths.run_config_json, configuration, overwrite=overwrite)
    if failures:
        _remove_figure_outputs(paths.output_directory, figure_directory=paths.figure_directory)
        atomic_write_frame_csv(analysis, paths.output_directory / "proximity.csv")
        atomic_write_frame_csv(
            pd.DataFrame(failures, columns=FAILED_COLUMNS), paths.failed_csv
        )
        values = _summary(
            paths,
            run_name,
            generation,
            sscd,
            selection,
            analysis,
            duration=time.monotonic() - started,
            failures=failures,
        )
    else:
        _remove_optional(paths.failed_csv)
        statistics = write_analysis_outputs(
            paths.output_directory, analysis=analysis, figure_directory=paths.figure_directory)
        _write_examples(
            paths,
            table_path=paths.output_directory / "proximity.csv",
            num_seeds=num_seeds,
        )
        values = _summary(
            paths,
            run_name,
            generation,
            sscd,
            selection,
            analysis,
            duration=time.monotonic() - started,
            statistics=statistics,
        )
    atomic_write_json(paths.summary_json, values)
    return ProximitySummary(paths, values)


def plot_proximity(
    project_root: str | Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    seed_start: int,
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY,
) -> ProximitySummary:
    """Publish scalar plots and cached-image examples without inference or tensor reads."""

    from utils.data.selection import (
        load_target_pair_selection,
        target_pair_selection_directory,
    )

    root = Path(project_root).expanduser().resolve()
    strategy = normalize_selection_strategy(selection_strategy)
    arguments = {
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "seed_start": seed_start,
    }
    role = _seed_role(**arguments)
    selection_identity = {
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "selection_strategy": strategy,
    }
    selection = load_target_pair_selection(root, **selection_identity)
    generation_run = root / generation_log_relative_path(**arguments)

    if role == "reference":
        directory = target_pair_selection_directory(root, **selection_identity)
        paths = ProximityPaths.frozen_selection(root, generation_run, directory)
        outputs = _reference_output_paths(
            root,
            generation_run,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            selection_strategy=strategy,
        )
        _write_examples(outputs, table_path=directory / "selection.csv",
                        num_seeds=num_seeds, validate_only=True)
        write_selection_figure(
            directory,
            output_directory=outputs.output_directory,
            figure_directory=outputs.figure_directory,
        )
        write_gmm_fit_figure(
            outputs.output_directory,
            figure_directory=outputs.figure_directory,
            frame=selection.frame,
            configuration=selection.configuration,
        )
        _write_examples(outputs, table_path=directory / "selection.csv",
                        num_seeds=num_seeds)
        return ProximitySummary(
            paths,
            {
                "complete": True,
                "plot_only": True,
                "selection_hash": selection.sha256,
            },
        )

    run_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start,
    )
    parent_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start=0,
    )
    paths = ProximityPaths.build(
        root,
        generation_run,
        output_run_name=parent_name,
        role=role,
        seed_start=seed_start,
        num_seeds=num_seeds,
        selection_strategy=strategy,
    )
    configuration = _load_saved_analysis_configuration(paths.run_config_json)
    _validate_saved_analysis_configuration(
        configuration,
        paths=paths,
        run_name=run_name,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        selection=selection,
        selection_strategy=strategy,
    )
    analysis = _load_saved_analysis(paths.output_directory / "proximity.csv")
    _validate_saved_analysis(
        analysis,
        selection=selection,
        model_name=model_name,
        num_seeds=num_seeds,
        selection_strategy=strategy,
    )
    _write_examples(paths, table_path=paths.output_directory / "proximity.csv",
                    num_seeds=num_seeds, validate_only=True)
    statistics = write_saved_analysis_figures(
        paths.output_directory, figure_directory=paths.figure_directory)
    _write_examples(paths, table_path=paths.output_directory / "proximity.csv",
                    num_seeds=num_seeds)
    return ProximitySummary(
        paths,
        {
            "complete": True,
            "plot_only": True,
            "selection_hash": selection.sha256,
            "prompt_spearman_summary": statistics.as_dict(),
        },
    )


def _reference_output_paths(
    root: Path,
    generation_run: Path,
    *,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str,
) -> ProximityPaths:
    parent_name = generation_run_name(
        model_name,
        scheduler_name,
        guidance_scale,
        num_inference_steps,
        num_seeds,
        seed_start=0,
    )
    return ProximityPaths.build(
        root,
        generation_run,
        output_run_name=parent_name,
        role="reference",
        seed_start=num_seeds,
        num_seeds=num_seeds,
        selection_strategy=selection_strategy,
    )


def _load_saved_analysis_configuration(path: Path) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ProximityError(f"saved proximity configuration is missing or unsafe: {path}")
    try:
        return read_json(path)
    except (OSError, RuntimeError, ValueError) as error:
        raise ProximityError(
            f"cannot read saved proximity configuration {path}: {error}"
        ) from error


def _validate_saved_analysis_configuration(
    configuration: Mapping[str, object],
    *,
    paths: ProximityPaths,
    run_name: str,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection: Any,
    selection_strategy: str,
) -> None:
    expected = {
        "analysis": "proximity",
        "source": "validated_generation_and_sscd_caches_only",
        "run_name": run_name,
        "generation_run_path": _display_path(
            paths.generation_run, paths.project_root
        ),
        "selection_policy": selection.configuration["selection_policy"],
        "selection_strategy": selection_strategy,
        "selection_hash": selection.sha256,
        "model_name": model_name,
        "scheduler_name": scheduler_name,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "num_seeds": num_seeds,
        "seeds": list(range(num_seeds)),
        "terminal_latent_index": -1,
        "distance": "euclidean_l2",
        "outputs": {
            "table": "proximity.csv",
            "figures": _figure_filename_catalog(),
        },
    }
    hash_fields = (
        "generation_scientific_config_hash",
        "sscd_configuration_hash",
    )
    expected_keys = {*expected, *hash_fields}
    if set(configuration) != expected_keys:
        raise ProximityError("saved proximity configuration has an invalid contract")
    wrong = [
        key
        for key, value in expected.items()
        if canonical_json(configuration.get(key)) != canonical_json(value)
    ]
    if wrong:
        raise ProximityError(
            "saved proximity configuration differs at: " + ", ".join(wrong)
        )
    if any(not _sha256(configuration.get(key)) for key in hash_fields):
        raise ProximityError("saved proximity configuration hash is invalid")


def _load_saved_analysis(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise ProximityError(f"saved proximity table is missing or unsafe: {path}")
    try:
        frame = pd.read_csv(
            path,
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
    except (OSError, TypeError, ValueError) as error:
        raise ProximityError(f"cannot read saved proximity table {path}: {error}") from error
    expected = (*ANALYSIS_COLUMNS, EXPERIMENT_SPEARMAN_COLUMN)
    if tuple(frame.columns) != expected:
        raise ProximityError("saved proximity table has an invalid schema")
    return frame


def _validate_saved_analysis(
    frame: pd.DataFrame,
    *,
    selection: Any,
    model_name: str,
    num_seeds: int,
    selection_strategy: str,
) -> None:
    if frame.empty:
        raise ProximityError("saved proximity table is empty")
    if not frame["model_name"].eq(model_name).all():
        raise ProximityError("saved proximity table model differs")
    if not frame["selection_strategy"].eq(selection_strategy).all():
        raise ProximityError("saved proximity table selection strategy differs")
    expected_seeds = list(range(num_seeds))
    for index, rows in frame.groupby("original_index", sort=False, dropna=False):
        seeds = pd.to_numeric(rows["seed"], errors="coerce").tolist()
        if seeds != expected_seeds:
            raise ProximityError(
                f"saved proximity seeds are incomplete or unordered: {index}"
            )
    annotated = _annotate_selection(
        frame.loc[:, OBSERVATION_COLUMNS],
        selection,
        require_complete=True,
    )
    saved = frame.loc[:, ANALYSIS_COLUMNS].reset_index(drop=True)
    if not saved.equals(annotated):
        raise ProximityError("saved proximity table differs from frozen selection")


def _frozen_reference_result(
    root: Path,
    generation_run: Path,
    model_name: str,
    scheduler_name: str,
    guidance_scale: float,
    num_inference_steps: int,
    num_seeds: int,
    selection_strategy: str,
) -> ProximitySummary | None:
    from utils.data.selection import (
        load_target_pair_selection,
        target_pair_selection_directory,
    )

    directory = target_pair_selection_directory(
        root,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        selection_strategy=selection_strategy,
    )
    if not directory.exists() and not directory.is_symlink():
        return None
    try:
        selection = load_target_pair_selection(
            root,
            model_name=model_name,
            scheduler_name=scheduler_name,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_seeds=num_seeds,
            selection_strategy=selection_strategy,
        )
    except TargetPairSelectionError as error:
        raise _stale_selection_error(
            directory,
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
            selection_strategy,
            str(error),
        ) from error
    try:
        observed_fingerprint = reference_completion_fingerprint(generation_run)
    except TargetPairSelectionError as error:
        raise _stale_selection_error(
            directory,
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
            selection_strategy,
            str(error),
        ) from error
    expected_fingerprint = selection.configuration["reference_completion_fingerprint"]
    if canonical_json(observed_fingerprint) != canonical_json(expected_fingerprint):
        changed = ", ".join(
            label
            for label in ("generation", "sscd")
            if canonical_json(observed_fingerprint[label])
            != canonical_json(expected_fingerprint[label])
        )
        raise _stale_selection_error(
            directory,
            model_name,
            scheduler_name,
            guidance_scale,
            num_inference_steps,
            num_seeds,
            selection_strategy,
            f"changed marker groups: {changed}",
        )
    paths = ProximityPaths.frozen_selection(root, generation_run, directory)
    summary = read_json(paths.summary_json)
    if summary.get("selection_hash") != selection.sha256:
        raise ProximityError("frozen selection summary hash differs")
    outputs = _reference_output_paths(
        root,
        generation_run,
        model_name=model_name,
        scheduler_name=scheduler_name,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_seeds=num_seeds,
        selection_strategy=selection_strategy,
    )
    write_selection_figure(
        directory,
        output_directory=outputs.output_directory,
        figure_directory=outputs.figure_directory,
    )
    write_gmm_fit_figure(
        outputs.output_directory,
        figure_directory=outputs.figure_directory,
        frame=selection.frame,
        configuration=selection.configuration,
    )
    _write_examples(
        outputs,
        table_path=directory / "selection.csv",
        num_seeds=num_seeds,
    )
    return ProximitySummary(paths, summary)


def _validate_generation_invocation(
    config: Mapping[str, object], **expected: object
) -> None:
    science = config.get("scientific_config")
    if not isinstance(science, Mapping):
        raise ProximityError("generation scientific_config is missing")
    seeds = science.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ProximityError("generation seeds are missing")
    observed = {
        "model_name": science.get("model_cli_name"),
        "scheduler_name": _scheduler_name(science),
        "guidance_scale": science.get("guidance_scale"),
        "num_inference_steps": science.get("num_inference_steps"),
        "num_seeds": science.get("num_seeds"),
        "seed_start": seeds[0],
    }
    wrong = [
        key
        for key, value in expected.items()
        if canonical_json(observed.get(key)) != canonical_json(value)
    ]
    if wrong:
        raise ProximityError("generation configuration differs at: " + ", ".join(wrong))
    start, count = int(expected["seed_start"]), int(expected["num_seeds"])
    if seeds != list(range(start, start + count)):
        raise ProximityError("generation seeds differ from seed-start and N")
    _latent_shape(science.get("latent_shape"))


def _load_sscd_config(run: Path, generation: Mapping[str, object]) -> dict[str, Any]:
    from utils.experiments.sscd import sscd_configuration_hash

    config = read_json(run / "sscd_config.json")
    digest = config.get("configuration_hash")
    if not _sha256(digest) or sscd_configuration_hash(config) != digest:
        raise ProximityError("SSCD configuration hash is invalid")
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    expected = {
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
    }
    if any(
        canonical_json(config.get(key)) != canonical_json(value)
        for key, value in expected.items()
    ):
        raise ProximityError("SSCD configuration differs from generation")
    return config


def _require_generation_coverage(
    paths: GenerationPaths, records: Sequence[CompletedGenerationRecord]
) -> None:
    path = paths.summary_json
    if not path.is_file() or path.is_symlink():
        raise ProximityError(f"required generation summary is absent: {path}")
    summary = read_json(path)
    if (
        summary.get("failed_rows") != 0
        or summary.get("completed_rows") != len(records)
        or summary.get("selected_rows") != len(records)
    ):
        raise ProximityError(
            "generation did not complete every selected prompt; rerun generation"
        )


def _validated_record(
    root: Path,
    paths: GenerationPaths,
    completed: CompletedGenerationRecord,
    config: Mapping[str, object],
) -> _GenerationRecord:
    metadata = completed.metadata
    index = completed.original_index
    science = config["scientific_config"]
    assert isinstance(science, Mapping)
    expected = {
        "scientific_config_hash": config["scientific_config_hash"],
        "model_cli_name": science["model_cli_name"],
        "scheduler_name": _scheduler_name(science),
        "guidance_scale": science["guidance_scale"],
        "num_inference_steps": science["num_inference_steps"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
        "latent_shape": science["latent_shape"],
        "preview_image_path": f"image/{index}.png",
    }
    wrong = [
        key
        for key, value in expected.items()
        if canonical_json(metadata.get(key)) != canonical_json(value)
    ]
    if wrong:
        raise ProximityError("generation marker differs at: " + ", ".join(wrong))
    hashes = metadata.get("tensor_file_sha256")
    if not isinstance(hashes, Mapping):
        raise ProximityError("generation tensor hashes are missing")
    _require_cached_file(paths.latent_path(index), hashes.get("latent"), "latent")
    _require_cached_file(
        paths.target_latent_path(index),
        hashes.get("target_latent"),
        "target latent",
    )
    _require_cached_file(
        paths.image_path(index),
        metadata.get("preview_image_sha256"),
        "generated montage",
    )
    return _GenerationRecord(
        metadata,
        completed,
        paths,
        _display_path(paths.image_path(index), root),
    )


def _terminal_l2(record: _GenerationRecord) -> torch.Tensor:
    trajectory = safe_torch_load(
        record.paths.latent_path(record.completed.original_index)
    )
    target = safe_torch_load(
        record.paths.target_latent_path(record.completed.original_index)
    )
    if not isinstance(trajectory, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise ProximityError("generation proximity inputs must be tensors")
    shape = _latent_shape(record.metadata["latent_shape"])
    expected = (
        int(record.metadata["num_seeds"]),
        int(record.metadata["num_inference_steps"]) + 1,
        *shape,
    )
    if tuple(trajectory.shape) != expected or tuple(target.shape) != shape:
        raise ProximityError("generation latent shape differs from marker")
    return compute_latent_distances(trajectory[:, -1], target).l2_norms


def _sscd_scores(
    root: Path,
    record: _GenerationRecord,
    generation: Mapping[str, object],
    config: Mapping[str, object],
) -> torch.Tensor:
    index = record.completed.original_index
    run = record.paths.run_directory
    marker_path = run / "sscd_record" / f"{index}.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ProximityError(f"cached SSCD marker is absent: {marker_path}")
    marker = read_json(marker_path)
    expected = {
        "original_index": index,
        "record_id": record.metadata["record_id"],
        "source_row_number": record.completed.source_row_number,
        "prompt_raw": record.metadata["prompt_raw"],
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "generation_latent_sha256": record.metadata["tensor_file_sha256"]["latent"],
        "target_image_sha256": record.metadata.get("target_image_sha256"),
        "sscd_configuration_hash": config["configuration_hash"],
        "num_seeds": record.metadata["num_seeds"],
        "seeds": record.metadata["seeds"],
        "score_shape": [record.metadata["num_seeds"]],
        "score_dtype": "float32",
    }
    wrong = [
        key
        for key, value in expected.items()
        if canonical_json(marker.get(key)) != canonical_json(value)
    ]
    if wrong:
        raise ProximityError("SSCD marker differs at: " + ", ".join(wrong))
    score_path = _canonical_score_path(
        root, run, marker.get("score_path"), run / "sscd" / f"{index}.pt"
    )
    if (
        not _sha256(marker.get("score_sha256"))
        or not score_path.is_file()
        or score_path.is_symlink()
        or file_sha256(score_path) != marker["score_sha256"]
    ):
        raise ProximityError("cached SSCD score file is invalid")
    scores = safe_torch_load(score_path)
    count = int(record.metadata["num_seeds"])
    if (
        not isinstance(scores, torch.Tensor)
        or scores.dtype != torch.float32
        or scores.device.type != "cpu"
        or not scores.is_contiguous()
        or tuple(scores.shape) != (count,)
        or not bool(torch.isfinite(scores).all())
        or bool((scores < -1.00001).any())
        or bool((scores > 1.00001).any())
    ):
        raise ProximityError("cached SSCD score tensor is invalid")
    return scores


def _paired_rows(
    record: _GenerationRecord,
    l2: torch.Tensor,
    scores: torch.Tensor,
    *,
    model_name: str,
) -> list[dict[str, object]]:
    count = int(record.metadata["num_seeds"])
    if (
        tuple(l2.shape) != (count,)
        or not bool(torch.isfinite(l2).all())
        or tuple(scores.shape) != (count,)
    ):
        raise ProximityError("L2 and SSCD values do not align with seeds")
    common = {
        "model_name": model_name,
        "original_index": record.completed.original_index,
        "record_id": record.metadata["record_id"],
        "source_row_number": record.completed.source_row_number,
        "prompt": record.metadata["prompt_raw"],
        "kind": normalize_webster_type(record.metadata.get("webster_overfit_type")),
        "target_image_sha256": record.metadata.get("target_image_sha256"),
        "generated_image_path": record.generated_image_path,
    }
    return [
        {
            **common,
            "seed": int(seed),
            "generated_image_tile_index": position,
            "l2_norm": float(l2[position]),
            "sscd": float(scores[position]),
            "observation_status": "complete",
            "observation_error": "",
        }
        for position, seed in enumerate(record.metadata["seeds"])
    ]


def _reference_records_frame(
    records: Sequence[CompletedGenerationRecord], model_name: str
) -> pd.DataFrame:
    rows = []
    for completed in records:
        metadata = completed.metadata
        row = {
            "model_name": model_name,
            "original_index": completed.original_index,
            "record_id": metadata.get("record_id"),
            "source_row_number": completed.source_row_number,
            "prompt": metadata.get("prompt_raw"),
            "kind": normalize_webster_type(metadata.get("webster_overfit_type")),
            "target_image_sha256": metadata.get("target_image_sha256"),
        }
        if (
            not isinstance(row["record_id"], str)
            or not row["record_id"]
            or not isinstance(row["prompt"], str)
            or not _sha256(row["target_image_sha256"])
        ):
            raise ProximityError(
                "cannot establish prompt identity for reference record "
                f"{completed.original_index}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _failed_reference_rows(
    root: Path,
    paths: GenerationPaths,
    completed: CompletedGenerationRecord,
    generation: Mapping[str, object],
    error: Exception,
    *,
    model_name: str,
) -> list[dict[str, object]]:
    identity = _reference_records_frame([completed], model_name).iloc[0].to_dict()
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    common = {
        **identity,
        "generated_image_path": _display_path(
            paths.image_path(completed.original_index), root
        ),
    }
    message = f"{type(error).__name__}: {error}"
    return [
        {
            **common,
            "seed": int(seed),
            "generated_image_tile_index": position,
            "l2_norm": float("nan"),
            "sscd": float("nan"),
            "observation_status": "cache_error",
            "observation_error": message,
        }
        for position, seed in enumerate(science["seeds"])
    ]


def _annotate_selection(
    frame: pd.DataFrame,
    selection: Any,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    prompts = selection.prompt_frame.copy(deep=True)
    prompts["original_index"] = prompts["original_index"].astype(str)
    observations = frame.copy(deep=True)
    observations["original_index"] = observations["original_index"].astype(str)
    identity = (
        "model_name",
        "record_id",
        "source_row_number",
        "prompt",
        "kind",
        "target_image_sha256",
    )
    decisions = (
        "selection_strategy",
        "prompt_spearman",
        "include_prompt",
        "selection_status",
        "selection_reason",
    )
    required = {"original_index", *identity, *decisions}
    missing = sorted(required - set(prompts))
    if missing:
        raise ProximityError(
            "frozen selection prompt table is missing: " + ", ".join(missing)
        )
    annotated = observations.merge(
        prompts.loc[:, ["original_index", *identity, *decisions]],
        on="original_index",
        how="left",
        suffixes=("", "_selection"),
        validate="many_to_one",
    )
    if annotated["include_prompt"].isna().any():
        raise ProximityError("experiment contains a prompt absent from selection")
    observed, frozen = (
        set(observations["original_index"]),
        set(prompts["original_index"]),
    )
    if require_complete and observed != frozen:
        raise ProximityError("experiment does not contain every frozen prompt")
    for field in identity:
        other = f"{field}_selection"
        if any(
            canonical_json(left) != canonical_json(right)
            for left, right in zip(annotated[field], annotated[other], strict=True)
        ):
            raise ProximityError(f"experiment and frozen selection differ at {field}")
        annotated = annotated.drop(columns=other)
    annotated["include_prompt"] = annotated["include_prompt"].astype(bool)
    return annotated.loc[:, ANALYSIS_COLUMNS].sort_values(
        ["source_row_number", "original_index", "seed"],
        kind="stable",
        ignore_index=True,
    )


def _analysis_configuration(
    paths: ProximityPaths,
    run_name: str,
    generation: Mapping[str, object],
    sscd: Mapping[str, object],
    selection: Any,
) -> dict[str, object]:
    science = generation["scientific_config"]
    assert isinstance(science, Mapping)
    return {
        "analysis": "proximity",
        "source": "validated_generation_and_sscd_caches_only",
        "run_name": run_name,
        "generation_run_path": _display_path(paths.generation_run, paths.project_root),
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "sscd_configuration_hash": sscd["configuration_hash"],
        "selection_policy": selection.configuration["selection_policy"],
        "selection_strategy": selection.configuration["selection_strategy"],
        "selection_hash": selection.sha256,
        "model_name": science["model_cli_name"],
        "scheduler_name": _scheduler_name(science),
        "guidance_scale": science["guidance_scale"],
        "num_inference_steps": science["num_inference_steps"],
        "num_seeds": science["num_seeds"],
        "seeds": science["seeds"],
        "terminal_latent_index": -1,
        "distance": "euclidean_l2",
        "outputs": {
            "table": "proximity.csv",
            "figures": _figure_filename_catalog(),
        },
    }


def _write_configuration(
    path: Path, desired: Mapping[str, object], *, overwrite: bool = False
) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ProximityError(f"unsafe existing analysis configuration: {path}")
        existing = read_json(path)
        if canonical_json(existing) == canonical_json(desired) and not overwrite:
            return
        if not overwrite:
            if canonical_json(_without_derived_outputs(existing)) == canonical_json(
                _without_derived_outputs(desired)
            ):
                atomic_write_json(path, desired)
                return
            raise ProximityError(
                f"existing analysis configuration is incompatible: {path}"
            )
    atomic_write_json(path, desired)


def _without_derived_outputs(value: object) -> object:
    """Exclude the rebuildable output catalog from analysis identity checks."""

    if not isinstance(value, Mapping):
        return value
    result = dict(value)
    result.pop("outputs", None)
    return result


def _figure_filename_catalog() -> dict[str, dict[str, str]]:
    """Keep historical config receipts stable; active PDF paths live in summaries."""
    return {
        scope: {file_format: filename for file_format, filename in formats.items()}
        for scope, formats in PROXIMITY_FIGURES.items()
    }


def _summary(
    paths: ProximityPaths,
    run_name: str,
    generation: Mapping[str, object],
    sscd: Mapping[str, object],
    selection: Any,
    analysis: pd.DataFrame,
    *,
    duration: float,
    statistics: AnalysisStatistics | None = None,
    failures: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    prompts = selection.prompt_frame
    included = prompts["include_prompt"].astype(bool)
    unusable = prompts["selection_status"].eq("unusable_reference_observations")
    values: dict[str, object] = {
        "complete": not failures,
        "run_name": run_name,
        "generation_scientific_config_hash": generation["scientific_config_hash"],
        "sscd_configuration_hash": sscd["configuration_hash"],
        "selection_policy": selection.configuration["selection_policy"],
        "selection_strategy": selection.configuration["selection_strategy"],
        "selection_hash": selection.sha256,
        "total_prompt_count": len(prompts),
        "included_prompt_count": int(included.sum()),
        "discarded_prompt_count": int((~included & ~unusable).sum()),
        "unusable_prompt_count": int(unusable.sum()),
        "observation_count": len(analysis),
        "included_observation_count": int(analysis["include_prompt"].sum()),
        "failed_rows": len(failures),
        "run_duration_seconds": duration,
        "finished_at_utc": utc_now(),
        "proximity_csv": _display_path(
            paths.output_directory / "proximity.csv", paths.project_root
        ),
    }
    if failures:
        values["failed_csv"] = _display_path(paths.failed_csv, paths.project_root)
    else:
        assert statistics is not None
        values["prompt_spearman_summary"] = statistics.as_dict()
        values["figures"] = {
            scope: {
                file_format: _display_path(
                    paths.figure_directory / filename,
                    paths.project_root,
                )
                for file_format, filename in formats.items()
            }
            for scope, formats in PROXIMITY_PDF_FIGURES.items()
        }
    return values


def _example_prompts(
    analysis: pd.DataFrame, *, num_seeds: int
) -> list[dict[str, object]]:
    """Choose actual prompt medians and extremes by mean terminal L2."""
    if isinstance(num_seeds, bool) or not isinstance(num_seeds, int) or num_seeds <= 0:
        raise ProximityError("example num_seeds must be a positive integer")
    required = {
        "original_index",
        "record_id",
        "source_row_number",
        "prompt",
        "seed",
        "l2_norm",
        "sscd",
        "include_prompt",
        "selection_status",
        "observation_status",
    }
    missing = sorted(required - set(analysis))
    if missing:
        raise ProximityError("example table is missing: " + ", ".join(missing))
    if not analysis.empty and not pd.api.types.is_bool_dtype(
        analysis["include_prompt"]
    ):
        raise ProximityError("example include_prompt must contain booleans")
    prompts: dict[str, list[dict[str, object]]] = {"retained": [], "discarded": []}
    for index, rows in analysis.groupby("original_index", sort=False):
        if (
            len(rows) != num_seeds
            or not rows["observation_status"].eq("complete").all()
        ):
            continue
        identity = (
            "record_id",
            "source_row_number",
            "prompt",
            "include_prompt",
            "selection_status",
        )
        if any(rows[column].nunique(dropna=False) != 1 for column in identity):
            raise ProximityError(
                f"example prompt identity differs across seeds: {index}"
            )
        first = rows.iloc[0]
        included = bool(first["include_prompt"])
        expected_status = (
            INCLUDED_PROXIMITY_RULE if included else DISCARDED_PROXIMITY_RULE
        )
        if first["selection_status"] != expected_status:
            continue
        l2 = pd.to_numeric(rows["l2_norm"], errors="coerce").tolist()
        seeds = pd.to_numeric(rows["seed"], errors="coerce").tolist()
        if (
            any(not math.isfinite(value) or value < 0 for value in l2)
            or any(
                not math.isfinite(value) or value < 0 or value != int(value)
                for value in seeds
            )
            or len(set(seeds)) != num_seeds
        ):
            continue
        scores = pd.to_numeric(rows["sscd"], errors="coerce").tolist()
        if any(not math.isfinite(value) for value in scores):
            raise ProximityError(f"example SSCD scores are invalid: {index}")
        group = "retained" if included else "discarded"
        prompts[group].append(
            {
                "group": group,
                "original_index": str(index),
                "record_id": str(first["record_id"]),
                "source_row_number": int(first["source_row_number"]),
                "prompt": str(first["prompt"]),
                "mean_l2_norm": math.fsum(l2) / num_seeds,
                "mean_sscd": math.fsum(scores) / num_seeds,
                "l2_norms": [
                    float(value) for _, value in sorted(zip(seeds, l2, strict=True))
                ],
                "seeds": sorted(int(seed) for seed in seeds),
            }
        )
    chosen: list[dict[str, object]] = []
    for candidates in prompts.values():
        candidates.sort(
            key=lambda row: (
                row["mean_l2_norm"],
                row["source_row_number"],
                row["original_index"],
            )
        )
        count = len(candidates)
        if not count:
            continue
        ranks = (
            [("median", 0)]
            if count == 1
            else [("highest", count - 1), ("lowest", 0)]
            if count == 2
            else [("highest", count - 1), ("median", (count - 1) // 2), ("lowest", 0)]
        )
        chosen.extend(
            {**candidates[position], "rank": rank} for rank, position in ranks
        )
    return chosen


def _example_output_files(
    output_directory: Path, *, figure_directory: Path | None = None
) -> list[Path]:
    """List current PDF exports and their cache receipt; never legacy PNGs."""
    metadata_directory = output_directory / "examples"
    image_directory = (figure_directory if figure_directory is not None else output_directory) / "examples"
    owned = [metadata_directory / "manifest.json"]
    for group in ("retained", "discarded"):
        for rank in ("highest", "median", "lowest"):
            owned.extend(image_directory / group / f"{rank}_l2_{kind}.pdf"
                         for kind in ("generated", "training"))
    for path in owned:
        _validate_example_destination(path)
    return owned


def _validate_example_destination(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ProximityError(f"unsafe example output: {path}")
    for parent in path.absolute().parents:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ProximityError(f"unsafe example directory: {parent}")


_EXAMPLE_GENERATED_SCALE = 0.75
_EXAMPLE_TRAINING_MAX_EDGE = 256


def _example_png(
    content: bytes, *, scale: float, max_edge: int | None = None
) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Downscale a display-only copy without cropping or upsampling."""
    try:
        with Image.open(io.BytesIO(content)) as original:
            source_size = original.size
            factor = min(1.0, scale)
            if max_edge is not None:
                factor = min(factor, max_edge / max(source_size))
            size = tuple(max(1, round(edge * factor)) for edge in source_size)
            with original.resize(size, Image.Resampling.LANCZOS) as preview:
                buffer = io.BytesIO()
                preview.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), source_size, size
    except (OSError, ValueError) as error:
        raise ProximityError(f"cannot encode proximity example PNG: {error}") from error


def _example_decoded_size(content: bytes) -> tuple[int, int]:
    """Validate cached image decodability without encoding a new image."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            return image.size
    except (OSError, ValueError) as error:
        raise ProximityError(
            "cannot decode cached proximity example image; restore the cached "
            "montage or normalized PNG (no regeneration is performed): " + str(error)
        ) from error


def _example_pdf(content: bytes) -> bytes:
    """Put the entire display-sized cached image on one aspect-preserving PDF page."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with Image.open(io.BytesIO(content)) as source:
        source.load()
        width, height = source.size
        figure = Figure(figsize=(width / FIGURE_DPI, height / FIGURE_DPI), dpi=FIGURE_DPI, facecolor="white")
        FigureCanvasAgg(figure)
        try:
            axes = figure.add_axes((0., 0., 1., 1.))
            axes.imshow(source, interpolation="none", aspect="equal")
            axes.set_axis_off()
            buffer = io.BytesIO()
            figure.savefig(buffer, format="pdf", dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=.05,
                           facecolor="white", metadata={"CreationDate": None, "ModDate": None})
            return buffer.getvalue()
        finally:
            figure.clear()


def _publish_example_payloads(payloads: Sequence[tuple[Path, bytes]], *, stale: Sequence[Path] = ()) -> None:
    """Stage PDFs and the cache manifest before replacing any member of the set."""
    current = [destination for destination, _content in payloads]
    if len(set(current)) != len(current):
        raise ProximityError("duplicate proximity example destinations")
    stale = tuple(path for path in stale if path not in current and path.exists())
    for path in (*current, *stale):
        _validate_example_destination(path)
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path | None, Path]] = []
    committed = False
    try:
        for destination, content in payloads:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
            staged.append((temporary, destination))
            atomic_write_bytes(temporary, content)
        # Stale PDFs are part of the same rollback set; the manifest is the
        # final staged payload and is installed only after all image changes.
        actions = [(None, path) for path in stale] + staged
        for temporary, destination in actions:
            _validate_example_destination(destination)
            backup = None
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
                os.replace(destination, backup)
            backups.append((backup, destination))
            if temporary is not None:
                os.replace(temporary, destination)
        committed = True
    except BaseException:
        for backup, destination in reversed(backups):
            if destination.exists() or destination.is_symlink():
                if not destination.is_file() or destination.is_symlink():
                    raise ProximityError(f"unsafe example destination during rollback: {destination}")
                destination.unlink()
            if backup is not None:
                os.replace(backup, destination)
        raise
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)
        if committed:
            for backup, _destination in backups:
                if backup is not None:
                    backup.unlink(missing_ok=True)


def _write_examples(
    paths: ProximityPaths, *, table_path: Path, num_seeds: int, validate_only: bool = False
) -> None:
    """Validate cached pairs, then publish PDF montages/targets without inference."""
    try:
        analysis = pd.read_csv(
            table_path,
            dtype={"original_index": str},
            keep_default_na=False,
            float_precision="round_trip",
        )
    except (OSError, ValueError) as error:
        raise ProximityError(
            f"cannot read example table {table_path}: {error}"
        ) from error
    examples = _example_prompts(analysis, num_seeds=num_seeds)
    output = paths.figure_directory / "examples"
    manifest_path = paths.output_directory / "examples" / "manifest.json"
    previous = _example_output_files(paths.output_directory, figure_directory=paths.figure_directory)
    cache = GenerationPaths(paths.generation_run)
    previews: list[tuple[Path, bytes, dict[str, object], str]] = []
    entries: list[dict[str, object]] = []
    if examples:
        generation = require_generation_run(cache)
        science = generation["scientific_config"]
        if science.get("num_seeds") != num_seeds:
            raise ProximityError("example seed count differs from generation")
    for example in examples:
        index = str(example["original_index"])
        rows = analysis.loc[analysis["original_index"].eq(index)].sort_values("seed")
        for field in ("generated_image_path", "target_image_sha256"):
            if field not in rows or rows[field].nunique(dropna=False) != 1:
                raise ProximityError(f"example {field} differs across seeds: {index}")
        if (
            example["seeds"] != science.get("seeds")
            or "generated_image_tile_index" not in rows
            or rows["generated_image_tile_index"].tolist() != list(range(num_seeds))
        ):
            raise ProximityError(f"example seeds or montage tile order differ: {index}")
        first = rows.iloc[0]
        generated = cache.image_path(index)
        recorded = Path(str(first["generated_image_path"]))
        if not recorded.is_absolute():
            recorded = paths.project_root / recorded
        if recorded.resolve() != generated.resolve():
            raise ProximityError(
                f"example montage path differs from generation: {index}"
            )
        validation = validate_generation_record(
            cache,
            index,
            expected_scientific_hash=generation["scientific_config_hash"],
            expected_record_identity={
                "record_id": example["record_id"],
                "source_row_number": example["source_row_number"],
                "prompt_raw": example["prompt"],
                "target_image_sha256": first["target_image_sha256"],
            },
            load_tensors=False,
            tensor_names=(),
            require_preview=True,
            verify_file_hashes=True,
        )
        if not validation.valid or validation.metadata is None:
            raise ProximityError(
                f"example generation cache is invalid for {index}: "
                + "; ".join(validation.errors)
            )
        metadata = validation.metadata
        if (
            metadata.get("seeds") != example["seeds"]
            or metadata.get("num_seeds") != num_seeds
        ):
            raise ProximityError(f"example marker seeds differ: {index}")
        target_value = metadata.get("target_image_path")
        if not isinstance(target_value, str) or not target_value:
            raise ProximityError(f"example paired training path is missing: {index}")
        target = Path(target_value)
        if not target.is_absolute():
            target = paths.project_root / target
        if target.suffix.lower() != ".png":
            raise ProximityError(
                f"example paired training image must be the cached normalized PNG: {index}"
            )
        _require_cached_file(
            target, first["target_image_sha256"], "paired training image"
        )
        prefix = Path(str(example["group"])) / f"{example['rank']}_l2"
        generated_relative = Path(f"{prefix}_generated.pdf")
        training_relative = Path(f"{prefix}_training.pdf")
        # Read only images; the cached montage already contains every generation seed.
        generated_bytes, training_bytes = generated.read_bytes(), target.read_bytes()
        if (
            hashlib.sha256(generated_bytes).hexdigest()
            != metadata["preview_image_sha256"]
            or hashlib.sha256(training_bytes).hexdigest()
            != first["target_image_sha256"]
        ):
            raise ProximityError(f"example image changed during export: {index}")
        _example_decoded_size(generated_bytes)
        _example_decoded_size(training_bytes)
        if validate_only:
            continue
        generated_png, generated_source_size, generated_size = _example_png(
            generated_bytes, scale=_EXAMPLE_GENERATED_SCALE
        )
        training_png, training_source_size, training_size = _example_png(
            training_bytes, scale=1.0, max_edge=_EXAMPLE_TRAINING_MAX_EDGE
        )
        if training_size == training_source_size and len(training_png) >= len(
            training_bytes
        ):
            # Keep already-small targets from growing through PNG re-encoding.
            training_png = training_bytes
        entries.append(
            {
                **example,
                "generated_source_path": _display_path(generated, paths.project_root),
                "training_source_path": _display_path(target, paths.project_root),
                "generated_image_path": generated_relative.as_posix(),
                "training_image_path": training_relative.as_posix(),
                "target_image_sha256": first["target_image_sha256"],
                "generated_source_sha256": metadata["preview_image_sha256"],
                "training_source_sha256": first["target_image_sha256"],
                "generated_source_size": list(generated_source_size),
                "training_source_size": list(training_source_size),
                "generated_image_size": list(generated_size),
                "training_image_size": list(training_size),
            }
        )
        previews.extend([
            (output / generated_relative, generated_png, entries[-1], "generated_image_sha256"),
            (output / training_relative, training_png, entries[-1], "training_image_sha256"),
        ])
    if validate_only:
        return
    # Every source is validated before PDF rendering or publication starts.
    # A failed late image/PDF leaves the previous exports and receipt intact.
    payloads: list[tuple[Path, bytes]] = []
    for destination, preview, entry, hash_field in previews:
        content = _example_pdf(preview)
        entry[hash_field] = hashlib.sha256(content).hexdigest()
        payloads.append((destination, content))
    manifest = {
        "ranking": "mean_terminal_l2_across_seeds",
        "median_rule": "lower_middle_prompt",
        "format": "pdf",
        "figure_directory": _display_path(output, paths.project_root),
        "image_export": {
            "format": "pdf", "pdf_dpi": FIGURE_DPI, "pad_inches": .05,
            "generated_scale": _EXAMPLE_GENERATED_SCALE,
            "training_max_edge": _EXAMPLE_TRAINING_MAX_EDGE,
            "resampling": "lanczos", "png_optimize": True,
        },
        "num_seeds": num_seeds,
        "source_csv": _display_path(table_path, paths.project_root),
        "examples": entries,
    }
    payloads.append((manifest_path, (canonical_json(manifest) + "\n").encode("utf-8")))
    current = {destination for destination, _content in payloads}
    _publish_example_payloads(payloads, stale=[path for path in previous if path not in current])
    print(f"[Proximity] Saved example PDFs: {output}", flush=True)


def _canonical_score_path(root: Path, run: Path, value: object, expected: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ProximityError("cached score path is invalid")
    given = Path(value)
    candidates = (
        [given.resolve()]
        if given.is_absolute()
        else [(run / given).resolve(), (root / given).resolve()]
    )
    target = expected.resolve()
    if target not in candidates:
        raise ProximityError(f"cached score path is noncanonical: {value}")
    return target


def _require_cached_file(path: Path, digest: object, label: str) -> None:
    if (
        not _sha256(digest)
        or not path.is_file()
        or path.is_symlink()
        or file_sha256(path) != digest
    ):
        raise ProximityError(f"cached {label} file is invalid")


def _failure_row(
    completed: CompletedGenerationRecord, error: Exception, run: Path
) -> dict[str, object]:
    return {
        "original_index": completed.original_index,
        "record_id": completed.metadata.get("record_id"),
        "source_row_number": completed.metadata.get("source_row_number"),
        "issue_type": type(error).__name__,
        "reason": str(error),
        "generation_record_path": str(completed.marker_path),
        "sscd_record_path": str(
            run / "sscd_record" / f"{completed.original_index}.json"
        ),
    }


def _remove_optional(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise ProximityError(f"derived output is not a regular file: {path}")


def _remove_figure_outputs(output_directory: Path, *, figure_directory: Path | None = None) -> None:
    """Remove current derived PDFs after a failed run; retain historical images."""

    figures = figure_directory if figure_directory is not None else output_directory
    for filename in PROXIMITY_PDF_FILENAMES:
        _remove_optional(figures / filename)
    for path in _example_output_files(output_directory, figure_directory=figures):
        _remove_optional(path)


def _latent_shape(value: object) -> tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProximityError("latent_shape is invalid")
    shape = tuple(value)
    if len(shape) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in shape
    ):
        raise ProximityError("latent_shape is invalid")
    return shape  # type: ignore[return-value]


def _scheduler_name(science: Mapping[str, object]) -> str:
    scheduler = science.get("scheduler")
    name = scheduler.get("name") if isinstance(scheduler, Mapping) else None
    if not isinstance(name, str) or not name:
        raise ProximityError("generation scheduler name is missing")
    return name


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sscd_command(
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    seeds: int,
    seed_start: int,
) -> str:
    return (
        f"./sscd.sh --model {model} --scheduler {scheduler} "
        f"--g {stable_float(guidance)} --T {steps} --N {seeds} "
        f"--seed-start {seed_start}"
    )


def _generation_command(
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    seeds: int,
    seed_start: int,
) -> str:
    return (
        f"./generate.sh --model {model} --scheduler {scheduler} "
        f"--g {stable_float(guidance)} --T {steps} --N {seeds} "
        f"--seed-start {seed_start}"
    )


def _stale_selection_error(
    directory: Path,
    model: str,
    scheduler: str,
    guidance: float,
    steps: int,
    seeds: int,
    selection_strategy: str,
    detail: str,
) -> ProximityError:
    command = (
        f"./compute_proximity.sh --model {model} --scheduler {scheduler} "
        f"--g {stable_float(guidance)} --T {steps} --N {seeds} "
        f"--seed-start {seeds} --selection-strategy {selection_strategy} "
        "--overwrite"
    )
    pipeline_command = (
        f"./run_all.sh --model {model} --scheduler {scheduler} "
        f"--g {stable_float(guidance)} --T {steps} --N {seeds} "
        f"--selection-strategy {selection_strategy}"
    )
    return ProximityError(
        f"frozen selection cannot be reused: {directory}\n{detail}\n"
        "Ensure the reference generation and SSCD marker caches are complete, "
        f"then rebuild only this selection with:\n{command}\n"
        "Or rerun the pipeline while rebuilding only selection/proximity "
        "artifacts (without regenerating trajectories or SSCD):\n"
        f"{pipeline_command}"
    )
