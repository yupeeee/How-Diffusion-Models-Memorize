"""Canonical paper paths and lightweight, scalar-only publication contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import fcntl
import os
import shutil
import shlex
import tempfile
import uuid

import pandas as pd

from utils.common.cli import generation_cache_namespace, generation_cache_parent_name
from utils.common.io import atomic_write_json, canonical_hash, file_sha256
from .contracts import TheoryError, numerical_config, read_object

METRIC_SCHEMA_VERSION = "paper-measurements-1"
BUNDLE_SCHEMA_VERSION = 1
ID_COLUMNS = {
    "original_index",
    "record_id",
    "target_id",
    "run_id",
    "seed",
    "target_atom_id",
    "atom_id",
}


@dataclass(frozen=True)
class PaperPaths:
    project_root: Path
    output_directory: Path

    @classmethod
    def build(cls, project_root, *, seed_start=0, **config):
        c = numerical_config(**config)
        identity = {
            key: c[key]
            for key in (
                "model_name",
                "scheduler_name",
                "guidance_scale",
                "num_inference_steps",
                "num_seeds",
            )
        }
        namespace = generation_cache_namespace(**identity, seed_start=seed_start)
        if namespace != f"experiment_S0_N{c['num_seeds']}":
            raise TheoryError(
                "Paper theory measurements require evaluation seeds 0..N-1"
            )
        root = Path(project_root).absolute()
        path = (
            root
            / "outputs"
            / generation_cache_parent_name(**identity)
            / "theory"
            / namespace
        )
        reject_symlinks(path)
        return cls(root, path)


def reject_symlinks(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise TheoryError(f"Unsafe symbolic-link path: {part}")
    return path


def contained_path(root, relative):
    root = reject_symlinks(root)
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise TheoryError(f"Unsafe paper artifact path: {relative}")
    destination = reject_symlinks(root / name)
    if not destination.is_relative_to(root):
        raise TheoryError(f"Paper artifact escapes its bundle: {relative}")
    return destination


def recompute_command(config):
    c = config.get("scientific_config", config)
    command = (
        "./run_all.sh --model "
        + c["model_name"]
        + " --scheduler "
        + c["scheduler_name"]
        + f" --g {c['guidance_scale']:g} --T {c['num_inference_steps']} --N {c['num_seeds']}"
        + " --center "
        + c.get("center", "reference-initial")
        + " --recompute-experiments"
    )

    if c.get("cached_baseline"):
        command += " --cached-baseline " + shlex.quote(c["cached_baseline"])
    if c.get("target_error_tolerance") is not None:
        command += " --target-error-tolerance " + str(c["target_error_tolerance"])
    return command


def measurement_sources():
    base = Path(__file__).parent
    return {
        name: file_sha256(base / name)
        for name in (
            "paper_measurements.py",
            "paper_feedback.py",
            "candidate_summaries.py",
            "summaries.py",
        )
    }


def frame_schema(frame):
    schema = {}
    for name in frame:
        values = frame[name]
        if name in ID_COLUMNS or name.endswith("_id"):
            schema[name] = "string"
        elif pd.api.types.is_bool_dtype(values.dtype):
            schema[name] = "boolean"
        elif pd.api.types.is_integer_dtype(values.dtype):
            schema[name] = "integer"
        elif pd.api.types.is_numeric_dtype(values.dtype):
            schema[name] = "float"
        else:
            schema[name] = "string"
    return schema


def write_plot_table(frame, path):
    path = reject_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = frame_schema(frame)
    normalized = frame.copy()
    for name, kind in schema.items():
        if kind == "string":
            normalized[name] = normalized[name].map(
                lambda v: "" if pd.isna(v) else str(v)
            )
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        normalized.to_csv(temporary, index=False, float_format="%.17g", na_rep="")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"schema": schema, "rows": len(frame), "sha256": file_sha256(path)}


def read_plot_table(path, specification):
    schema = specification["schema"]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(frame.columns) != set(schema) or len(frame) != specification["rows"]:
        raise TheoryError(f"Paper CSV columns/row count differ: {path}")
    floats = [name for name, kind in schema.items() if kind == "float"]
    numeric = (
        pd.read_csv(
            path,
            usecols=floats,
            dtype={name: float for name in floats},
            keep_default_na=False,
            na_values={name: [""] for name in floats},
            float_precision="round_trip",
        )
        if floats
        else None
    )
    for name, kind in schema.items():
        values = frame[name]
        if kind == "integer":
            frame[name] = pd.array(
                [int(v) if v else pd.NA for v in values], dtype="Int64"
            )
        elif kind == "float":
            frame[name] = numeric[name]
        elif kind == "boolean":
            if not values.isin(["True", "False", ""]).all():
                raise TheoryError(f"Invalid Boolean value in {path}:{name}")
            frame[name] = values.map({"True": True, "False": False, "": pd.NA}).astype(
                "boolean"
            )
        elif kind != "string":
            raise TheoryError(f"Unsupported paper CSV dtype {kind}: {path}:{name}")
    return frame


def load_paper_inputs(
    bundle, *, expected_config=None, diagnostics=False, check_recipe=True
):
    """Read compact CSVs and stored metadata only; never open backing shards."""
    bundle = reject_symlinks(bundle)
    config = read_object(contained_path(bundle, "run_config.json"))
    command = recompute_command(config)
    try:
        if (
            config.get("schema_version") != BUNDLE_SCHEMA_VERSION
            or config.get("metric_schema_version") != METRIC_SCHEMA_VERSION
        ):
            raise TheoryError("Obsolete paper measurement schema")
        if expected_config is not None and config[
            "scientific_config"
        ] != numerical_config(**expected_config):
            raise TheoryError(
                "Paper scientific configuration differs (including center/tolerance)"
            )
        identity = config["scientific_identity"]
        if (
            canonical_hash(identity) != config["scientific_hash"]
            or identity["config"] != config["scientific_config"]
        ):
            raise TheoryError("Paper scientific identity/hash differs")
        if check_recipe and identity["measurement_sources"] != measurement_sources():
            raise TheoryError(
                "Paper measurement definitions changed; analysis migration is required"
            )
        summary = read_object(contained_path(bundle, "summary.json"))
        audit = read_object(contained_path(bundle, "audit.json"))
        if not summary.get("complete") or audit.get("blocking", True):
            raise TheoryError(
                "Paper numerical publication is incomplete or failed its correctness audit"
            )
        if (
            summary.get("scientific_hash") != config["scientific_hash"]
            or audit.get("scientific_hash") != config["scientific_hash"]
        ):
            raise TheoryError("Paper summary/audit scientific identity differs")
        for name, expected in summary["numerical_files"].items():
            # Analysis audit tables can be large; plot preflight checks only its
            # small contracts and requested compact inputs below.
            if name not in {"run_config.json", "audit.json"}:
                continue
            path = contained_path(bundle, name)
            if (
                path.suffix not in {".csv", ".json", ".md"}
                or not path.is_file()
                or file_sha256(path) != expected
            ):
                raise TheoryError(f"Missing/stale compact paper input: {name}")
        from .paper_registry import paper_registry

        entries = paper_registry(diagnostics=diagnostics)
        if isinstance(entries, dict):
            entries = entries.get("figures", entries.get("entries", []))
        frames = {}
        for entry in entries:
            stem = entry["stem"]
            metadata = summary["figures"].get(stem)
            if metadata is None:
                raise TheoryError(f"Missing applicability metadata for {stem}")
            status = metadata.get("status")
            if status in {"error", "blocked", "missing"}:
                raise TheoryError(f"{stem}: {metadata.get('reason', status)}")
            if status == "unavailable" and entry["category"] != "diagnostics":
                raise TheoryError(
                    f"Required paper figure {stem} is unavailable: {metadata.get('reason')}"
                )
            if status in {"not_applicable", "alias", "unavailable"}:
                if not metadata.get("reason"):
                    raise TheoryError(
                        f"Missing explicit applicability reason for {stem}"
                    )
                continue
            if status not in {"available", "complete"}:
                raise TheoryError(f"Unknown paper figure status for {stem}: {status}")
            spec = summary["plot_data"].get(stem)
            if not spec:
                raise TheoryError(f"Missing compact figure table for {stem}")
            if spec["path"] != entry["source_table"]:
                raise TheoryError(f"Noncanonical compact paper input for {stem}")
            path = contained_path(bundle, spec["path"])
            if not path.is_file() or file_sha256(path) != spec["sha256"]:
                raise TheoryError(f"Missing/stale plot table for {stem}")
            frames[stem] = read_plot_table(path, spec)
            missing = set(entry["required_columns"]) - set(frames[stem])
            if missing:
                raise TheoryError(
                    f"Missing required paper columns for {stem}: {sorted(missing)}"
                )
        return config, summary, audit, frames
    except (KeyError, ValueError, OSError, TheoryError) as error:
        raise TheoryError(f"{error}. Run {command}") from error


def validate_paper_bundle(bundle, expected_config=None, *, diagnostics=False):
    return load_paper_inputs(
        bundle, expected_config=expected_config, diagnostics=diagnostics
    )[0]


def audit_paper_integrity(bundle):
    """Explicit full scalar integrity audit, separate from lightweight plotting."""
    load_paper_inputs(bundle)
    summary = read_object(contained_path(bundle, "summary.json"))
    for name, digest in summary["numerical_files"].items():
        path = contained_path(bundle, name)
        if (
            path.suffix not in {".csv", ".json", ".md"}
            or not path.is_file()
            or file_sha256(path) != digest
        ):
            raise TheoryError(f"Missing/stale paper analysis artifact: {name}")
    return {"checked_files": len(summary["numerical_files"]), "complete": True}


def _copy_to_stage(source, destination):
    # Numerical CSVs are immutable and every analysis writer replaces atomically.
    # Linking avoids reading/copying large audit tables during a plot-only swap.
    if Path(source).suffix == ".csv":
        os.link(source, destination)
        return destination
    return shutil.copy2(source, destination)


@contextlib.contextmanager
def publication_lock(output):
    """Serialize an active role, including directory-swap recovery after a crash."""
    output = reject_symlinks(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = reject_symlinks(output.parent / ("." + output.name + ".lock"))
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TheoryError(
                f"Another theory invocation is publishing {output}"
            ) from error
        try:
            marker = output.parent / ("." + output.name + ".transaction.json")
            if marker.exists():
                state = read_object(marker)
                if not str(state["backup"]).startswith(
                    "." + output.name + ".backup-"
                ) or not str(state["stage"]).startswith("." + output.name + ".stage-"):
                    raise TheoryError("Unsafe theory transaction marker")
                backup = contained_path(output.parent, state["backup"])
                stage = contained_path(output.parent, state["stage"])
                if not output.exists() and backup.exists():
                    os.replace(backup, output)
                if backup.exists() and output.exists():
                    _finish_backup(output, backup, state.get("archive_previous", False))
                if stage.exists():
                    shutil.rmtree(stage)
                marker.unlink()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def staged_publication(output, *, archive_previous=False):
    """Stage a compact role; shared figure publisher handles all image exports."""
    output = reject_symlinks(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix="." + output.name + ".stage-", dir=output.parent)
    )
    backup = output.parent / ("." + output.name + ".backup-" + uuid.uuid4().hex)
    marker = output.parent / ("." + output.name + ".transaction.json")
    installed = False
    atomic_write_json(
        marker,
        {
            "schema_version": 1,
            "phase": "staging",
            "stage": stage.name,
            "backup": backup.name,
            "archive_previous": archive_previous,
        },
    )
    try:
        if output.exists():
            for path in output.rglob("*"):
                reject_symlinks(path)
            shutil.copytree(
                output, stage, dirs_exist_ok=True, copy_function=_copy_to_stage
            )
        yield stage
        atomic_write_json(
            marker,
            {
                "schema_version": 1,
                "stage": stage.name,
                "backup": backup.name,
                "archive_previous": archive_previous,
            },
        )
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except BaseException:
            if backup.exists():
                os.replace(backup, output)
            raise
        installed = True
        if backup.exists():
            _finish_backup(output, backup, archive_previous)
        marker.unlink(missing_ok=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if not installed and backup.exists() and not output.exists():
            os.replace(backup, output)
        if not backup.exists():
            marker.unlink(missing_ok=True)


def _finish_backup(output, backup, archive_previous):
    if not archive_previous:
        shutil.rmtree(backup)
        return
    previous = read_object(backup / "run_config.json")
    archive = contained_path(
        output.parent,
        ".archives/"
        + output.name
        + "/"
        + previous["scientific_hash"]
        + "-"
        + backup.name.rsplit("-", 1)[-1],
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        backup / "archive_receipt.json",
        {
            "original_path": str(output),
            "scientific_hash": previous["scientific_hash"],
            "files": {
                p.relative_to(backup).as_posix(): file_sha256(p)
                for p in backup.rglob("*")
                if p.is_file()
            },
        },
    )
    os.replace(backup, archive)


def retire_obsolete_figures(stage, previous):
    """Retire only recorded, unchanged outputs absent from the new manifest."""
    current = read_object(stage / "figure_manifest.json")
    obsolete = {
        name: digest
        for name, digest in previous.get("files", {}).items()
        if name not in current["files"] and Path(name).suffix in {".png", ".pdf"}
    }
    if not obsolete:
        return
    archive = contained_path(stage, "archive/retired-" + canonical_hash(obsolete)[:16])
    for name, digest in obsolete.items():
        source = contained_path(stage, name)
        if not source.exists():
            continue
        if file_sha256(source) != digest:
            raise TheoryError(f"Refusing to retire modified owned figure: {name}")
        target = contained_path(archive, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    atomic_write_json(
        archive / "manifest.json", {"original_paths_and_hashes": obsolete}
    )


def render_saved_paper(bundle, *, expected_config=None, diagnostics=False):
    from .paper_plotting import render_paper

    bundle = reject_symlinks(bundle)
    with publication_lock(bundle):
        load_paper_inputs(
            bundle, expected_config=expected_config, diagnostics=diagnostics
        )
        previous = (
            read_object(bundle / "figure_manifest.json")
            if (bundle / "figure_manifest.json").exists()
            else {}
        )
        with staged_publication(bundle) as stage:
            result = render_paper(stage, diagnostics=diagnostics)
            retire_obsolete_figures(stage, previous)
    return result


def retire_legacy_figures(project_root, output, source_bundle, source_manifest):
    """Archive only hash-verified images owned by the selected previous renderers."""
    root = reject_symlinks(project_root)
    source = reject_symlinks(source_bundle)
    base_run = output.parent.parent
    if not source.is_relative_to(base_run) or source.parent.name != "theory_candidates":
        return {
            "status": "external_source_preserved",
            "archived": [],
            "unrecognized": [],
        }
    owners = []
    candidate_owner = source / "candidate_figure_manifest.json"
    if candidate_owner.exists():
        owner = read_object(candidate_owner)
        owners.append((source, owner.get("files", {})))
    base = Path(source_manifest.get("base_bundle", ""))
    if (
        base.is_absolute()
        and base.is_relative_to(base_run)
        and base.name == source_manifest.get("base_analysis_hash")
    ):
        old_owner = base / "figures" / "manifest.json"
        if old_owner.is_file():
            owner = read_object(old_owner)
            images = {
                item["path"]: item["sha256"]
                for figure in owner.get("figures", {}).values()
                for item in figure.get("formats", {}).values()
            }
            owners.append((old_owner.parent, images))
    planned = {}
    unrecognized = []
    for owner_root, files in owners:
        for name, digest in files.items():
            if Path(name).suffix not in {".png", ".pdf"}:
                continue
            path = contained_path(owner_root, name)
            if not path.exists():
                continue
            relative = path.relative_to(root).as_posix()
            if file_sha256(path) != digest:
                unrecognized.append(relative)
                continue
            planned[relative] = digest
    archive = contained_path(
        output.parent, "archive/legacy-" + source_manifest["analysis_hash"]
    )
    receipt_path = archive / "manifest.json"
    prior = (
        read_object(receipt_path)
        if receipt_path.exists()
        else {"original_paths_and_hashes": {}}
    )
    if planned:
        archive.mkdir(parents=True, exist_ok=True)
        # Persist ownership before any move so interrupted retirement is resumable.
        prior["original_paths_and_hashes"].update(planned)
        prior.update(status="retiring", unrecognized_modified_files=unrecognized)
        atomic_write_json(receipt_path, prior)
        for relative, digest in planned.items():
            original = contained_path(root, relative)
            target = contained_path(archive, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and file_sha256(target) != digest:
                raise TheoryError(f"Conflicting owned figure archive: {target}")
            os.replace(original, target)
        prior["status"] = "complete"
        atomic_write_json(receipt_path, prior)
    return {
        "status": "complete",
        "archive": str(archive) if receipt_path.exists() else None,
        "archived_count": len(prior["original_paths_and_hashes"]),
        "unrecognized_modified_files": unrecognized,
    }
