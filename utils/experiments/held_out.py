"""Publish a browsable view of TV prompts excluded by frozen selection.

This is a diagnostic export only. It copies the already-generated preview
montages and never reads or changes scientific tensors, selection decisions,
or proximity results.
"""

from __future__ import annotations

import html
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from utils.common.io import (
    atomic_copy,
    atomic_write_frame_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    canonical_json,
    csv_safe_frame,
    file_sha256,
    read_json,
)

from .cache import (
    GenerationPaths,
    list_completed_records,
    require_generation_run,
    safe_index,
)

HELD_OUT_TV_SCHEMA_VERSION = 1
HELD_OUT_TV_DIRECTORY_NAME = "held_out_tv"

MANIFEST_COLUMNS = tuple(
    "original_index record_id source_row_number prompt_raw "
    "webster_overfit_type_normalized selection_status selection_reason "
    "target_semantics threshold selection_mean_sscd "
    "reference_validation_mean_sscd seed_start num_seeds seeds "
    "generated_preview_path prompt_text_path source_preview_path "
    "preview_image_sha256 preview_size_bytes target_image_sha256 "
    "generation_scientific_config_hash selection_hash".split()
)


class HeldOutTVExportError(RuntimeError):
    """A held-out TV preview export cannot be published safely."""


@dataclass(frozen=True, slots=True)
class HeldOutTVExportResult:
    """Paths and counts for one completed diagnostic export."""

    directory: Path
    manifest_path: Path
    gallery_path: Path
    prompt_count: int
    total_prompt_count: int
    missing_prompt_count: int
    reused: bool


def export_held_out_tv_results(
    project_root: str | Path,
    *,
    model_name: str,
    generation_run: str | Path,
    output_directory: str | Path,
) -> HeldOutTVExportResult:
    """Copy excluded-TV preview montages into a prompt-labeled gallery.

    ``held_out_tv`` means whole TV prompts excluded by the frozen target-pair
    selection. It does not mean the reference-validation seed half 30--39.
    The output directory is already scoped by run role, so a reference export
    contains that run's seeds 20--39 and an experiment export contains its own
    seed block (normally 0--19).
    """

    root = Path(project_root).expanduser().resolve()
    run = Path(generation_run).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    _require_descendant(run, root / "logs", "generation run")
    _require_descendant(output, root / "outputs", "proximity output")
    _require_regular_directory(run, "generation run")
    _require_regular_directory(output, "proximity output")

    paths = GenerationPaths(run)
    generation = require_generation_run(paths)
    science = generation.get("scientific_config")
    if not isinstance(science, Mapping):
        raise HeldOutTVExportError("generation scientific_config is missing")
    generation_hash = generation.get("scientific_config_hash")
    if not _sha256(generation_hash):
        raise HeldOutTVExportError("generation scientific hash is invalid")
    seeds = _seed_values(science.get("seeds"))

    from utils.data.selection import load_target_pair_selection

    selection = load_target_pair_selection(root, model_name=model_name)
    if selection.model_name != model_name:
        raise HeldOutTVExportError("frozen selection model differs")
    selection_hash = selection.sha256
    if not _sha256(selection_hash):
        raise HeldOutTVExportError("frozen selection hash is invalid")
    _validate_analysis_output(
        output,
        generation_hash=str(generation_hash),
        selection_hash=selection_hash,
    )

    analysis_indices = _analysis_indices(output / "paired_all.csv")
    completed = {
        record.original_index: record for record in list_completed_records(paths)
    }
    selection_frame = _held_out_tv_frame(selection.frame)
    absent_indices = tuple(
        index
        for index in selection_frame["original_index"].astype(str)
        if index not in analysis_indices
    )
    current = selection_frame.loc[
        selection_frame["original_index"].astype(str).isin(analysis_indices)
    ].copy()

    destination = output / HELD_OUT_TV_DIRECTORY_NAME
    preexisting = destination.exists() or destination.is_symlink()
    temporary: Path | None = None
    if not preexisting:
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{HELD_OUT_TV_DIRECTORY_NAME}.",
                suffix=".tmp",
                dir=output,
            )
        )

    rows: list[dict[str, object]] = []
    total_preview_bytes = 0
    try:
        progress = tqdm(
            current.to_dict(orient="records"),
            total=len(current),
            desc="[Held-out TV] Preview gallery",
            unit="prompt",
            dynamic_ncols=True,
            leave=True,
            disable=False,
        )
        for selection_row in progress:
            row, source = _manifest_row(
                root,
                paths,
                selection_row,
                completed,
                expected_seeds=seeds,
                generation_hash=str(generation_hash),
                selection_hash=selection_hash,
            )
            rows.append(row)
            total_preview_bytes += int(row["preview_size_bytes"])
            if temporary is not None:
                index = str(row["original_index"])
                prompt_directory = temporary / "prompts" / index
                copied_preview = prompt_directory / "generated.png"
                atomic_copy(source, copied_preview)
                if (
                    copied_preview.stat().st_size != row["preview_size_bytes"]
                    or file_sha256(copied_preview) != row["preview_image_sha256"]
                ):
                    raise HeldOutTVExportError(
                        f"copied held-out TV preview differs for {index}"
                    )
                atomic_write_text(
                    prompt_directory / "prompt.txt",
                    str(row["prompt_raw"]),
                )
            progress.set_postfix_str(f"id={row['original_index']}", refresh=False)

        manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        gallery = _gallery_html(
            model_name=model_name,
            generation_run=run.name,
            seeds=seeds,
            rows=rows,
        )
        config = {
            "schema_version": HELD_OUT_TV_SCHEMA_VERSION,
            "artifact": "held_out_tv_generation_previews",
            "definition": (
                "TV prompts excluded as whole prompts by the frozen "
                "target-pair selection"
            ),
            "reference_validation_seed_half_is_not_a_prompt_holdout": True,
            "model_name": model_name,
            "generation_run_path": _relative(run, root),
            "proximity_output_path": _relative(output, root),
            "generation_scientific_config_hash": str(generation_hash),
            "selection_hash": selection_hash,
            "seeds": list(seeds),
            "seed_start": seeds[0],
            "num_seeds": len(seeds),
            "frozen_held_out_tv_prompt_count": len(selection_frame),
            "exported_prompt_count": len(rows),
            "not_present_in_current_run_indices": list(absent_indices),
            "manifest_content_hash": canonical_hash(rows),
            "preview_format": "seed-ascending generation montage",
            "scientific_metric_input": False,
            "affects_selection_or_analysis": False,
        }
        summary = {
            "schema_version": HELD_OUT_TV_SCHEMA_VERSION,
            "complete": True,
            "held_out_tv_prompt_count": len(rows),
            "copied_preview_count": len(rows),
            "not_present_in_current_run_count": len(absent_indices),
            "total_preview_bytes": total_preview_bytes,
            "manifest": "manifest.csv",
            "gallery": "gallery.html",
            "prompt_directory": "prompts",
        }

        if preexisting:
            _validate_existing_export(
                destination,
                manifest=manifest,
                gallery=gallery,
                config=config,
                summary=summary,
                rows=rows,
            )
            return _result(
                destination,
                len(rows),
                total=len(selection_frame),
                reused=True,
            )

        assert temporary is not None
        atomic_write_frame_csv(manifest, temporary / "manifest.csv")
        atomic_write_text(temporary / "gallery.html", gallery)
        atomic_write_json(temporary / "config.json", config)
        atomic_write_json(temporary / "summary.json", summary)
        _fsync_file(temporary / "manifest.csv")
        _fsync_directory_tree(temporary)
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(output)
        return _result(
            destination,
            len(rows),
            total=len(selection_frame),
            reused=False,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, HeldOutTVExportError):
            raise
        raise HeldOutTVExportError(str(error)) from error
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _held_out_tv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "original_index",
        "record_id",
        "source_row_number",
        "prompt_raw",
        "target_image_sha256",
        "webster_overfit_type_normalized",
        "include_target_pair",
        "selection_status",
        "selection_reason",
        "target_semantics",
        "threshold",
        "selection_mean_sscd",
        "reference_validation_mean_sscd",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HeldOutTVExportError(
            "frozen selection is missing columns: " + ", ".join(missing)
        )
    values = frame.copy(deep=True)
    values["original_index"] = values["original_index"].astype(str)
    if values["original_index"].duplicated().any():
        raise HeldOutTVExportError("frozen selection has duplicate prompt indices")
    if not pd.api.types.is_bool_dtype(values["include_target_pair"]):
        raise HeldOutTVExportError("frozen selection inclusion column is not boolean")
    mask = (
        values["webster_overfit_type_normalized"].eq("TV")
        & ~values["include_target_pair"]
    )
    return values.loc[mask].sort_values(
        ["source_row_number", "original_index"], kind="stable", ignore_index=True
    )


def _manifest_row(
    root: Path,
    paths: GenerationPaths,
    selection_row: Mapping[str, object],
    completed: Mapping[str, Any],
    *,
    expected_seeds: tuple[int, ...],
    generation_hash: str,
    selection_hash: str,
) -> tuple[dict[str, object], Path]:
    index = safe_index(selection_row["original_index"])
    record = completed.get(index)
    if record is None:
        raise HeldOutTVExportError(
            f"analysis includes held-out TV prompt {index} without a completion marker"
        )
    metadata = record.metadata
    identity = {
        "record_id": selection_row["record_id"],
        "source_row_number": _integer(
            selection_row["source_row_number"], "selection source_row_number"
        ),
        "prompt_raw": selection_row["prompt_raw"],
        "target_image_sha256": selection_row["target_image_sha256"],
    }
    for field, expected in identity.items():
        if metadata.get(field) != expected:
            raise HeldOutTVExportError(
                f"generation marker {field} differs for held-out TV prompt {index}"
            )
    if metadata.get("scientific_config_hash") != generation_hash:
        raise HeldOutTVExportError(
            f"generation marker scientific hash differs for {index}"
        )
    if _seed_values(metadata.get("seeds")) != expected_seeds:
        raise HeldOutTVExportError(f"generation marker seeds differ for {index}")
    expected_relative = f"image/{index}.png"
    if metadata.get("preview_image_path") != expected_relative:
        raise HeldOutTVExportError(
            f"generation preview path is noncanonical for {index}"
        )
    preview_hash = metadata.get("preview_image_sha256")
    if not _sha256(preview_hash):
        raise HeldOutTVExportError(f"generation preview hash is invalid for {index}")
    source = paths.image_path(index)
    if not source.is_file() or source.is_symlink():
        raise HeldOutTVExportError(
            f"generation preview is not a regular file for {index}: {source}"
        )
    if file_sha256(source) != preview_hash:
        raise HeldOutTVExportError(f"generation preview hash differs for {index}")
    size = source.stat().st_size
    if size <= 0:
        raise HeldOutTVExportError(f"generation preview is empty for {index}")
    prompt = selection_row["prompt_raw"]
    if not isinstance(prompt, str):
        raise HeldOutTVExportError(f"prompt text is invalid for {index}")
    return (
        {
            "original_index": index,
            "record_id": str(selection_row["record_id"]),
            "source_row_number": identity["source_row_number"],
            "prompt_raw": prompt,
            "webster_overfit_type_normalized": "TV",
            "selection_status": str(selection_row["selection_status"]),
            "selection_reason": str(selection_row["selection_reason"]),
            "target_semantics": str(selection_row["target_semantics"]),
            "threshold": _optional_number(selection_row["threshold"]),
            "selection_mean_sscd": _optional_number(
                selection_row["selection_mean_sscd"]
            ),
            "reference_validation_mean_sscd": _optional_number(
                selection_row["reference_validation_mean_sscd"]
            ),
            "seed_start": expected_seeds[0],
            "num_seeds": len(expected_seeds),
            "seeds": list(expected_seeds),
            "generated_preview_path": f"prompts/{index}/generated.png",
            "prompt_text_path": f"prompts/{index}/prompt.txt",
            "source_preview_path": _relative(source, root),
            "preview_image_sha256": str(preview_hash),
            "preview_size_bytes": size,
            "target_image_sha256": str(selection_row["target_image_sha256"]),
            "generation_scientific_config_hash": generation_hash,
            "selection_hash": selection_hash,
        },
        source,
    )


def _validate_analysis_output(
    output: Path, *, generation_hash: str, selection_hash: str
) -> None:
    config_path = output / "run_config.json"
    summary_path = output / "summary.json"
    for path, label in ((config_path, "run configuration"), (summary_path, "summary")):
        if not path.is_file() or path.is_symlink():
            raise HeldOutTVExportError(
                f"completed proximity {label} is not a regular file: {path}"
            )
    config = read_json(config_path)
    summary = read_json(summary_path)
    expected = {
        "generation_scientific_config_hash": generation_hash,
        "selection_hash": selection_hash,
    }
    for key, value in expected.items():
        if config.get(key) != value or summary.get(key) != value:
            raise HeldOutTVExportError(f"proximity {key} differs")
    if summary.get("complete") is not True:
        raise HeldOutTVExportError("proximity analysis is not complete")


def _analysis_indices(path: Path) -> frozenset[str]:
    if not path.is_file() or path.is_symlink():
        raise HeldOutTVExportError(f"paired_all.csv is not a regular file: {path}")
    try:
        values = pd.read_csv(path, usecols=["original_index"], dtype=str)
    except (OSError, ValueError) as error:
        raise HeldOutTVExportError(f"cannot read paired_all.csv: {error}") from error
    indices = frozenset(values["original_index"].astype(str))
    if not indices:
        raise HeldOutTVExportError("paired_all.csv contains no prompt indices")
    return indices


def _validate_existing_export(
    directory: Path,
    *,
    manifest: pd.DataFrame,
    gallery: str,
    config: Mapping[str, object],
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise HeldOutTVExportError(f"unsafe existing held-out TV export: {directory}")
    expected_files = {
        "manifest.csv",
        "gallery.html",
        "config.json",
        "summary.json",
    }
    for row in rows:
        expected_files.add(str(row["generated_preview_path"]))
        expected_files.add(str(row["prompt_text_path"]))
    actual_files: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise HeldOutTVExportError(f"held-out TV export contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(directory).as_posix())
        elif not path.is_dir():
            raise HeldOutTVExportError(
                f"held-out TV export contains an unsafe entry: {path}"
            )
    if actual_files != expected_files:
        raise HeldOutTVExportError(
            "existing held-out TV export file set differs; remove only this "
            f"derived directory and rerun: {directory}"
        )
    if (directory / "manifest.csv").read_text(encoding="utf-8") != _csv_text(manifest):
        raise HeldOutTVExportError("existing held-out TV manifest differs")
    if (directory / "gallery.html").read_text(encoding="utf-8") != gallery:
        raise HeldOutTVExportError("existing held-out TV gallery differs")
    if canonical_json(read_json(directory / "config.json")) != canonical_json(config):
        raise HeldOutTVExportError("existing held-out TV configuration differs")
    if canonical_json(read_json(directory / "summary.json")) != canonical_json(summary):
        raise HeldOutTVExportError("existing held-out TV summary differs")
    for row in rows:
        preview = directory / str(row["generated_preview_path"])
        prompt = directory / str(row["prompt_text_path"])
        if file_sha256(preview) != row["preview_image_sha256"]:
            raise HeldOutTVExportError(
                f"existing held-out TV preview differs for {row['original_index']}"
            )
        if prompt.read_text(encoding="utf-8") != row["prompt_raw"]:
            raise HeldOutTVExportError(
                f"existing held-out TV prompt text differs for {row['original_index']}"
            )


def _gallery_html(
    *,
    model_name: str,
    generation_run: str,
    seeds: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> str:
    cards = []
    for row in rows:
        index = str(row["original_index"])
        prompt = html.escape(str(row["prompt_raw"]))
        image_path = html.escape(str(row["generated_preview_path"]), quote=True)
        cards.append(
            "\n".join(
                (
                    f'<article id="prompt-{html.escape(index, quote=True)}">',
                    f"  <h2>Prompt <code>{html.escape(index)}</code></h2>",
                    '  <p class="metrics">'
                    f"selection mean SSCD: {_score(row['selection_mean_sscd'])} · "
                    "reference-validation mean SSCD: "
                    f"{_score(row['reference_validation_mean_sscd'])}</p>",
                    f"  <pre>{prompt}</pre>",
                    f'  <a href="{image_path}"><img loading="lazy" '
                    f'src="{image_path}" alt="Generated montage for prompt '
                    f'{html.escape(index, quote=True)}"></a>',
                    "</article>",
                )
            )
        )
    seed_label = f"{seeds[0]}–{seeds[-1]}" if len(seeds) > 1 else str(seeds[0])
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>Held-out TV prompts — {html.escape(model_name)}</title>\n"
        "<style>body{font:16px system-ui,sans-serif;margin:2rem;background:#f5f5f5;color:#171717}"
        "header{max-width:1000px;margin:auto auto 2rem}main{display:grid;gap:1.5rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr))}"
        "article{background:white;padding:1rem;border-radius:.6rem;box-shadow:0 1px 5px #0002}"
        "h1,h2{margin:.2rem 0 .7rem}h2{font-size:1rem}.metrics{color:#555;font-size:.9rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#f1f3f5;padding:.8rem;border-radius:.3rem}"
        "img{display:block;width:100%;height:auto;background:#ddd}code{font-family:ui-monospace,monospace}</style>\n"
        "</head>\n<body>\n<header>\n"
        "<h1>Held-out TV prompt generations</h1>\n"
        f"<p>Model <code>{html.escape(model_name)}</code>; run "
        f"<code>{html.escape(generation_run)}</code>; seed-ascending montages "
        f"for seeds {html.escape(seed_label)}. Here “held out” means a TV prompt "
        "excluded by the frozen SSCD threshold, not the validation seed half.</p>\n"
        f"<p>{len(rows)} prompt(s). Full provenance is in "
        '<a href="manifest.csv">manifest.csv</a>.</p>\n'
        "</header>\n<main>\n" + "\n".join(cards) + "\n</main>\n</body>\n</html>\n"
    )


def _result(
    directory: Path, count: int, *, total: int, reused: bool
) -> HeldOutTVExportResult:
    return HeldOutTVExportResult(
        directory=directory,
        manifest_path=directory / "manifest.csv",
        gallery_path=directory / "gallery.html",
        prompt_count=count,
        total_prompt_count=total,
        missing_prompt_count=total - count,
        reused=reused,
    )


def _csv_text(frame: pd.DataFrame) -> str:
    return csv_safe_frame(frame).to_csv(index=False, lineterminator="\n")


def _seed_values(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise HeldOutTVExportError("generation seeds are missing")
    seeds = tuple(_integer(seed, "seed") for seed in value)
    if seeds != tuple(range(seeds[0], seeds[0] + len(seeds))):
        raise HeldOutTVExportError("generation seeds are not contiguous")
    return seeds


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise HeldOutTVExportError(f"{label} must be an integer")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise HeldOutTVExportError(f"{label} must be an integer") from error
    try:
        exact = float(value) == number  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact:
        raise HeldOutTVExportError(f"{label} must be an integer")
    return number


def _optional_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HeldOutTVExportError(
            f"selection score is not numeric: {value!r}"
        ) from error
    if not math.isfinite(number):
        return None
    return number


def _score(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.6f}"


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_descendant(path: Path, parent: Path, label: str) -> None:
    try:
        path.relative_to(parent.resolve())
    except ValueError as error:
        raise HeldOutTVExportError(f"{label} is outside {parent}") from error


def _require_regular_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise HeldOutTVExportError(f"{label} is not a regular directory: {path}")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
