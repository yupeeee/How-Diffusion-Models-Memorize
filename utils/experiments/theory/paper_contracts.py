"""Canonical paper paths and lightweight, scalar-only publication contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import fcntl
import os
import re
import shutil
import shlex
import stat
import tempfile
import uuid

import numpy as np
import pandas as pd

from utils.common.cli import generation_cache_namespace, generation_cache_parent_name
from utils.common.io import atomic_write_json, canonical_hash, file_sha256
from .contracts import FOUR_STAGE_OPTION_KEYS, TheoryError, numerical_config, read_object

METRIC_SCHEMA_VERSION = "four-stage-evidence-projected-gap-error-1"
BUNDLE_SCHEMA_VERSION = 5
ID_COLUMNS = {
    "original_index",
    "record_id",
    "target_id",
    "run_id",
    "seed",
    "target_atom_id",
    "atom_id",
    "draw_id",
    "noise_draw_id",
    "probe_id",
    "input_law_id",
    "reference_law_id",
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
    c = saved_scientific_configuration(config) if "scientific_config" in config else config
    # Historical zero-centred bundles remain readable, but rebuilding upgrades
    # their retired diagnostic centre without mutating the saved configuration.
    center = c.get("center", "reference-initial")
    if center == "zero":
        center = "reference-initial"
    command = (
        "./run_all.sh --model "
        + c["model_name"]
        + " --scheduler "
        + c["scheduler_name"]
        + f" --g {c['guidance_scale']:g} --T {c['num_inference_steps']} --N {c['num_seeds']}"
        + " --center "
        + center
        + " --recompute-experiments"
    )

    if c.get("cached_baseline"):
        command += " --cached-baseline " + shlex.quote(str(c["cached_baseline"]))
    if c.get("target_error_tolerance") is not None:
        command += " --target-error-tolerance " + str(c["target_error_tolerance"])
    # Decimal precision remains in historical receipts, but GPU computation uses
    # binary64 and rejects that option. Preserve the operation budget under its
    # current CLI spelling so the suggested command is executable.
    for key in ("mean_source", "num_mean_samples", "mean_seed", "num_loss_seeds", "loss_seed", "loss_timesteps", "num_unconditional_loss_seeds", "reference_law", "reference_manifest", "reference_snr_decades", "terminal_noise_run_alpha", "numerical_max_decimal_products", "numerical_max_variation_nodes", "numerical_variation_absolute_width"):
        if c.get(key) is not None:
            option = "numerical-max-products" if key == "numerical_max_decimal_products" else key.replace("_", "-")
            command += " --" + option + " " + shlex.quote(str(c[key]))
    if c.get("counterfactual_unconditional"):
        command += " --counterfactual-unconditional --counterfactual-steps " + shlex.quote(",".join(map(str, c.get("counterfactual_steps", [0]))))
    if "measure_unconditional_loss" in c:
        command += " --unconditional-loss" if c["measure_unconditional_loss"] else " --no-unconditional-loss"
    return command



def saved_scientific_configuration(saved):
    """Merge optional supplements only at the paper boundary.

    The base scientific_config remains readable by the unchanged numerical
    cache loader, so an optional denoiser supplement cannot invalidate probes.
    """
    if not isinstance(saved.get("scientific_config"), dict):
        raise TheoryError("Missing paper scientific configuration")
    optional = saved.get("supplemental_config", {})
    if not isinstance(optional, dict) or set(optional) - set(FOUR_STAGE_OPTION_KEYS):
        raise TheoryError("Unknown paper supplemental configuration")
    science = dict(saved["scientific_config"])
    science.setdefault("mean_source", "cached-targets")
    return numerical_config(**(science | optional))


def saved_plot_configuration(bundle, *, requested, explicit_keys=(), portable=False):
    """Inherit saved measurement settings without opening sources or initializing models.

    Repository model/scheduler/g/T/N remain mandatory identity checks. A copied
    bundle supplies those too unless explicitly overridden by the caller.
    """
    saved = saved_scientific_configuration(read_object(contained_path(bundle, "run_config.json")))
    keys = set(explicit_keys)
    if not portable:
        keys.update(("model_name", "scheduler_name", "guidance_scale",
                     "num_inference_steps", "num_seeds"))
    candidate = dict(saved)
    candidate.update({key: requested[key] for key in keys if key in requested})
    candidate = numerical_config(**candidate)
    # Resolve manifest paths as identity strings only. Plotting never opens them.
    if candidate != saved:
        conflicts = sorted(key for key in candidate if candidate[key] != saved.get(key))
        raise TheoryError(
            "Explicit plot settings conflict with saved measurements: "
            + ", ".join(conflicts) + ". Recompute with " + recompute_command(candidate)
        )
    return candidate


def measurement_sources():
    base = Path(__file__).parent
    return {
        name: file_sha256(base / name)
        for name in (
            "reference_mean.py",
            "four_stage_figures.py",
            "four_stage_measurements.py",
            "four_stage_reduce.py",
            "counterfactual_probes.py",
            "gaussian_controls.py",
            "supplemental_cache.py",
            "direct_figures.py",
            "evidence_figures.py",
            "evidence_measurements.py",
            "evidence_reduce.py",
            "evidence_reference.py",
            "numerical_reduce.py",
            "numerical_refinement.py",
            "numerical_intervals.py",
            "numerical_screening.py",
            "direct_measurements.py",
            "direct_integration.py",
            "reference_law.py",
            "direct_probes.py",
            "direct_probe_math.py",
            "direct_probe_cache.py",
            "direct_reduce.py",
            "support.py",
            "branch_gap.py",
            "scheduler_adapter.py",
            "candidate_integration.py",
            "candidate_feedback.py",
            "feedback.py",
            "metrics.py",
            "summaries.py",
            "cache_reader.py",
            "../../models/probe_loading.py",
            "../../models/sampling.py",
            "../../models/prediction_conversion.py",
            "../../models/schedulers.py",
        )
    }


def frame_schema(frame):
    schema = {}
    for name in frame:
        values = frame[name]
        if name in ID_COLUMNS or name.endswith("_id"):
            schema[name] = "string"
        elif pd.api.types.is_bool_dtype(values.dtype) or (
            len(values.dropna()) > 0
            and all(isinstance(value, (bool, np.bool_)) for value in values.dropna())
        ):
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
    combined_config = saved_scientific_configuration(config)
    command = recompute_command(combined_config)
    try:
        if (
            config.get("schema_version") != BUNDLE_SCHEMA_VERSION
            or config.get("metric_schema_version") != METRIC_SCHEMA_VERSION
        ):
            raise TheoryError("Obsolete paper measurement schema")
        if expected_config is not None and combined_config != numerical_config(**expected_config):
            raise TheoryError(
                "Paper scientific configuration differs (including center/tolerance)"
            )
        identity = config["scientific_identity"]
        if (
            canonical_hash(identity) != config["scientific_hash"]
            or identity["config"] != combined_config
        ):
            raise TheoryError("Paper scientific identity/hash differs")
        if check_recipe:
            saved_sources = identity["measurement_sources"]
            current_sources = measurement_sources()
            if saved_sources != current_sources:
                if isinstance(saved_sources, dict):
                    differing_sources = ", ".join(
                        name for name in sorted(set(saved_sources) | set(current_sources))
                        if saved_sources.get(name) != current_sources.get(name)
                    )
                else:
                    differing_sources = "invalid saved measurement source map"
                raise TheoryError(
                    "Paper measurement definitions changed; analysis migration is required"
                    f" for bundle {bundle}; differing source files: {differing_sources}"
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

        entries = paper_registry(diagnostics=diagnostics, counterfactual=combined_config.get("counterfactual_unconditional", False))
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
            if status == "unavailable" and not entry.get("allow_unavailable", entry["category"] == "diagnostics"):
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
        peak = summary["figures"].get("branch_gap_peak_step", {})
        if peak.get("status") in {"available", "complete"} and not peak.get("trajectory_peak_summary"):
            # Presentation disclosure only: no statistics are computed and no
            # saved numerical metadata is rewritten during compatible plotting.
            peak["peak_summary_status"] = "not_saved_in_this_compatible_bundle"
            peak["peak_summary_details"] = (
                "Saved weighted bins and shape/status counts are retained. Additional descriptive "
                "peak quartiles/initialization-mass summary was not saved; plotting does not reduce "
                "statistics. To save it through the explicit analysis-only scalar reduction, run " + command
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
            # Preserve unrelated user links without following them. Individual
            # scientific inputs and renderer destinations still reject symlinks.
            shutil.copytree(
                output, stage, dirs_exist_ok=True, copy_function=_copy_to_stage,
                symlinks=True,
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
                if not p.is_symlink() and p.is_file()
            },
            "preserved_symlinks": {
                p.relative_to(backup).as_posix(): os.readlink(p)
                for p in backup.rglob("*") if p.is_symlink()
            },
        },
    )
    os.replace(backup, archive)


def retire_obsolete_figures(stage, previous):
    """Apply the explicit presentation migration inside the role transaction.

    Only known renderer image paths with matching ownership hashes may be
    unlinked in the stage. Historical per-step exports are included; arbitrary
    images and every saved measurement remain outside the deletion allowlist.
    The original bundle remains the rollback copy until the directory swap.
    """
    from .paper_registry import (FIGURE_RETIREMENTS, REGISTRY_VERSION,
                                 measurement_registry, paper_registry, previous_paper_registry)

    stage = reject_symlinks(stage)
    manifest_path = contained_path(stage, "figure_manifest.json")
    current = read_object(manifest_path)
    if not current.get("complete"):
        raise TheoryError("Cannot retire figures before replacement publication completes")
    owned = {**previous.get("preserved_files", {}), **previous.get("files", {})}
    active = current.get("files", {})
    ledger_path = contained_path(stage, "figure_retirement.json")
    prior = read_object(ledger_path) if ledger_path.is_file() else {}
    if ledger_path.exists() and (
        prior.get("kind") != "theory_figure_retirement"
        or prior.get("schema_version") != 1
        or not isinstance(prior.get("records"), list)
        or any(not isinstance(row, dict) or not isinstance(row.get("old_path"), str)
               for row in prior.get("records", []))
    ):
        raise TheoryError("Refusing to replace unrecognized figure_retirement.json; existing file preserved")
    prior_records = {row["old_path"]: row for row in prior.get("records", [])}
    records, completed = {}, set()

    def regular_image(relative):
        path = contained_path(stage, relative)
        if not path.exists():
            return path, None
        if not stat.S_ISREG(path.lstat().st_mode):
            raise TheoryError("Not a regular renderer image")
        return path, file_sha256(path)

    selected = {entry["stem"]: entry for entry in paper_registry(diagnostics=True, counterfactual=True)}
    historical = [*previous_paper_registry(diagnostics=True, counterfactual=True),
                  *measurement_registry(diagnostics=True, counterfactual=True)]
    known_stems = {entry["stem"] for entry in historical} | set(selected)
    known_stems.update(spec["stem"] for spec in FIGURE_RETIREMENTS)
    aliases = {entry.get("output_stem", stem): stem for stem, entry in selected.items()}
    aliases.update({stem: stem for stem in known_stems})
    timestep_stems = {entry["stem"] for entry in historical if entry.get("per_timestep_exports")}
    categories = {"main", "appendix", "diagnostics", "figures"}

    def specification_for(name):
        # Ownership of a user comparison or a cached image does not authorize deletion.
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix not in {".png", ".pdf"}:
            return None
        parts = relative.parts
        if len(parts) == 2 and parts[0] in categories and relative.stem in aliases:
            stem = aliases[relative.stem]
            replacement = selected.get(stem, {}).get("outputs", {})
            return {"stem": stem, "replacement": replacement,
                    "reason": ("Selected publication moved to figures; saved measurements retained"
                               if replacement else "Figure omitted from the six selected publications; saved measurements retained")}
        if (len(parts) == 3 and parts[0] in categories and parts[1] in timestep_stems
                and re.fullmatch(r"step_[0-9]{3,}\.(png|pdf)", parts[2])):
            return {"stem": parts[1], "replacement": {},
                    "reason": "Per-timestep exports retired; selected combined figure and saved measurements retained"}
        return None

    def replacement_ready(specification):
        if specification["replacement"] and set(specification["replacement"]) != {"png", "pdf"}:
            return False
        for name in specification["replacement"].values():
            try:
                _path, digest = regular_image(name)
            except (OSError, TheoryError):
                return False
            if digest is None or active.get(name) != digest:
                return False
        return True

    # Known routes without ownership are reported, never silently removed.
    historical_paths = {name for entry in historical for name in entry["outputs"].values()}
    historical_paths.update(f"{spec['old_category']}/{spec['stem']}.{suffix}"
                            for spec in FIGURE_RETIREMENTS for suffix in ("png", "pdf"))
    for entry in previous.get("timestep_figures", []):
        if isinstance(entry, dict) and isinstance(entry.get("outputs"), dict):
            historical_paths.update(name for name in entry["outputs"].values() if isinstance(name, str))
    candidates = historical_paths | set(owned) | set(prior_records)
    allowed = {name: spec for name in sorted(candidates)
               if name not in active and (spec := specification_for(name)) is not None}
    removed_parents = set()
    for name, specification in allowed.items():
        destination = specification["replacement"].get(Path(name).suffix.lstrip("."))
        row = {"old_path": name, "stable_stem": specification["stem"],
               "sha256": owned.get(name), "destination": destination,
               "reason": specification["reason"]}
        try:
            path, digest = regular_image(name)
        except (OSError, TheoryError) as error:
            row.update(status="conflict", decision="preserved_unsafe_path", detail=str(error).replace(str(stage), "<active_bundle>"))
        else:
            if digest is None:
                if prior_records.get(name, {}).get("status") in {"removed", "already_absent"}:
                    row = prior_records[name]
                else:
                    row.update(status="already_absent", decision="no_file_to_retire")
                completed.add(name)
            elif name not in owned:
                row.update(status="conflict", decision="preserved_unowned", observed_sha256=digest)
            elif digest != owned[name]:
                row.update(status="conflict", decision="preserved_modified", observed_sha256=digest)
            elif not replacement_ready(specification):
                row.update(status="conflict", decision="preserved_missing_replacement_pair")
            else:
                path.unlink()
                completed.add(name)
                removed_parents.add(path.parent)
                row.update(status="removed", decision="moved" if destination else "retired")
        records[name] = row
    # Only now-empty image folders are removed. Preserved user/scalar files
    # prevent rmdir and remain untouched; this never walks arbitrary output trees.
    for parent in sorted(removed_parents, key=lambda path: len(path.parts), reverse=True):
        while parent != stage and parent.is_relative_to(stage):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    # Invalid historical ownership paths grant no authority, even if their
    # basename resembles an authorized retired figure. Never open those paths.
    for name, digest in owned.items():
        if name in allowed or name in active or Path(name).suffix not in {".png", ".pdf"}:
            continue
        try:
            contained_path(stage, name)
        except (OSError, TheoryError) as error:
            records[name] = {"old_path": name, "stable_stem": Path(name).stem,
                             "sha256": digest, "destination": None, "status": "conflict",
                             "decision": "preserved_unsafe_path", "reason": "Outside the authorized migration",
                             "detail": str(error).replace(str(stage), "<active_bundle>")}
    preserved = {name: digest for name, digest in owned.items()
                 if name not in active and name not in completed}
    current["preserved_files"] = preserved
    atomic_write_json(manifest_path, current)
    ledger = {"kind": "theory_figure_retirement", "schema_version": 1, "registry_version": REGISTRY_VERSION,
              "scope": "Six selected publications; ownership-checked renderer migration in this active bundle only",
              "records": [records[name] for name in sorted(records)],
              "preserved_other_owned_paths": sorted(name for name in preserved if name not in records),
              "other_files_policy": "All non-enumerated files and scientific artifacts are preserved"}
    # Deterministic content: a second plot does not create another receipt/archive.
    if ledger != prior:
        atomic_write_json(ledger_path, ledger)
    conflicts = [row for row in records.values() if row["status"] == "conflict"]
    for row in conflicts:
        print(f"[Theory] Figure migration conflict: {row['old_path']} ({row['decision']}); preserved", flush=True)
    return ledger


def _validate_presentation_registry(bundle):
    """A known presentation registry may migrate; unrelated user JSON may not."""
    from .paper_registry import COMPATIBLE_PRESENTATION_REGISTRY_VERSIONS

    path = contained_path(bundle, "registry.json")
    if not path.exists():
        return
    previous = read_object(path)
    if (previous.get("version") not in COMPATIBLE_PRESENTATION_REGISTRY_VERSIONS
            or not isinstance(previous.get("figures"), list)
            or any(not isinstance(entry, dict) or not isinstance(entry.get("stem"), str)
                   for entry in previous.get("figures", []))):
        raise TheoryError("Refusing to replace unrecognized registry.json; existing file preserved")


def render_saved_paper(bundle, *, expected_config=None, diagnostics=False):
    from .paper_plotting import render_paper
    from .progress import StageProgress

    bundle = reject_symlinks(bundle)
    with publication_lock(bundle):
        load_paper_inputs(
            bundle, expected_config=expected_config, diagnostics=diagnostics
        )
        _validate_presentation_registry(bundle)
        previous = (
            read_object(bundle / "figure_manifest.json")
            if (bundle / "figure_manifest.json").exists()
            else {}
        )
        with StageProgress("Publishing saved paper bundle"), staged_publication(bundle) as stage:
            render_paper(stage, diagnostics=diagnostics)
            with StageProgress("Applying verified figure placement and retirement"):
                retire_obsolete_figures(stage, previous)
            # registry.json is presentation metadata; immutable scientific
            # run_config/summary and plot_data remain byte-for-byte unchanged.
            from .paper_registry import REGISTRY_VERSION, measurement_registry, paper_registry
            saved = read_object(contained_path(stage, "run_config.json"))
            counterfactual = saved_scientific_configuration(saved).get("counterfactual_unconditional", False)
            atomic_write_json(contained_path(stage, "registry.json"), {
                "version": REGISTRY_VERSION,
                "figures": paper_registry(diagnostics=True, counterfactual=counterfactual),
                "measurement_inventory": measurement_registry(diagnostics=True, counterfactual=counterfactual),
            })
            result = read_object(contained_path(stage, "figure_manifest.json"))
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
