"""Prepare the fixed paper suite from shared, validated candidate measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.common.io import atomic_write_json, canonical_hash, file_sha256
from .contracts import TheoryError, numerical_config, read_object
from .paper_contracts import (
    BUNDLE_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    PaperPaths,
    contained_path,
    load_paper_inputs,
    measurement_sources,
    publication_lock,
    reject_symlinks,
    retire_obsolete_figures,
    retire_legacy_figures,
    staged_publication,
    write_plot_table,
)
from .paper_registry import REGISTRY_VERSION, paper_registry


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


ENDPOINT_SOURCES = frozenset(
    {
        "candidate_feedback.py",
        "candidate_metrics.py",
        "candidate_reduce.py",
        "cache_reader.py",
        "metrics.py",
        "scheduler_adapter.py",
        "support.py",
        "feedback.py",
    }
)


def _source_recipe_current(manifest, *, diagnostics=False):
    sources = manifest.get("source_code")
    required = ENDPOINT_SOURCES | (
        {"candidate_integration.py"} if diagnostics else set()
    )
    return (
        isinstance(sources, dict)
        and required <= sources.keys()
        and all(
            file_sha256(Path(__file__).parent / name) == sources[name]
            for name in required
        )
    )


def _source_bundle(root, config, *, source_analysis, diagnostics, device, **options):
    from .candidate_contracts import (
        find_candidate_bundle,
        validate_candidate_bundle,
        validate_candidate_sources,
    )

    if source_analysis is not None:
        bundle = reject_symlinks(Path(source_analysis).absolute())
        manifest = validate_candidate_bundle(bundle, expected_config=config)
        return bundle, manifest, "explicit_saved_analysis"
    try:
        bundle = find_candidate_bundle(root, **config)
        manifest = validate_candidate_bundle(bundle, expected_config=config)
        validate_candidate_sources(bundle, root)
        if not _source_recipe_current(manifest, diagnostics=diagnostics):
            raise TheoryError(
                "Candidate endpoint/integration definitions changed or lack source provenance"
            )
        if diagnostics and not manifest.get("complete"):
            raise TheoryError("Requested integration diagnostics are incomplete")
        return bundle, manifest, "validated_current_analysis"
    except (TheoryError, OSError):
        from .reduce import run_theory
        from .candidate_reduce import run_candidates

        base = run_theory(
            root,
            **config,
            **options,
            device=device,
            defer_path_integration=True,
            prepare_candidate_endpoints=True,
        )
        bundle = run_candidates(
            root,
            **config,
            **options,
            device=device,
            base_bundle=base,
            render=False,
            endpoint_only=not diagnostics,
        )
        return (
            bundle,
            validate_candidate_bundle(bundle, expected_config=config),
            "protected_cache_analysis",
        )


def _auxiliary(frames, metadata):
    auxiliary = metadata.pop("auxiliary_tables", {})
    if isinstance(auxiliary, list):
        auxiliary = {name: frames.pop(name) for name in auxiliary if name in frames}
    return auxiliary


def _save_table(stage, relative, frame, numerical_files):
    if len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["status"])
    specification = write_plot_table(frame, contained_path(stage, relative))
    numerical_files[relative] = specification["sha256"]
    return {"path": relative, **specification}


def run_paper(
    project_root,
    *,
    source_analysis=None,
    source_logs=None,
    diagnostics=False,
    recompute=False,
    device="auto",
    candidate_chunk_size=256,
    query_chunk_size=16,
    **configuration,
):
    """Analysis only: preserve source shards and publish compact paper inputs."""
    from .candidate_contracts import read_tables
    from .paper_measurements import build_nonfeedback_plot_inputs
    from .paper_feedback import build_feedback_plot_inputs
    from .paper_plotting import render_paper

    root = Path(project_root).absolute()
    config = numerical_config(**configuration)
    output = PaperPaths.build(root, **config).output_directory
    with publication_lock(output):
        previous_config = (
            read_object(output / "run_config.json")
            if (output / "run_config.json").exists()
            else None
        )
        if (
            previous_config
            and previous_config["scientific_config"] != config
            and not recompute
        ):
            raise TheoryError(
                "Active paper center/tolerance/configuration differs; use explicit --recompute-experiments to archive and reconfigure it"
            )
        if (
            previous_config
            and not recompute
            and source_analysis is None
            and source_logs is None
        ):
            try:
                from .candidate_contracts import validate_candidate_sources

                source = Path(previous_config["source_analysis"]["path"])
                validate_candidate_sources(source, root)
                if (
                    file_sha256(source / "analysis_manifest.json")
                    != previous_config["scientific_identity"]["source_manifest_sha256"]
                ):
                    raise TheoryError("Source analysis changed")
                source_manifest = read_object(source / "analysis_manifest.json")
                if not _source_recipe_current(source_manifest, diagnostics=diagnostics):
                    raise TheoryError(
                        "Candidate measurement definitions changed or lack source provenance"
                    )
                if diagnostics and not source_manifest.get("complete"):
                    raise TheoryError("Integrated diagnostics need analysis")
                load_paper_inputs(
                    output, expected_config=config, diagnostics=diagnostics
                )
            except (TheoryError, OSError, KeyError):
                pass
            else:
                previous = read_object(output / "figure_manifest.json")
                with staged_publication(output) as stage:
                    render_paper(stage, diagnostics=diagnostics)
                    retire_obsolete_figures(stage, previous)
                return output
        source, manifest, mode = _source_bundle(
            root,
            config,
            source_analysis=source_analysis,
            diagnostics=diagnostics,
            device=device,
            candidate_chunk_size=candidate_chunk_size,
            query_chunk_size=query_chunk_size,
        )
        print(f"Paper measurements: {source}", flush=True)
        tables = read_tables(source, include_dose=True, include_controls=False)
        supplement_receipt = {"status": "saved_scalar_contracts"}
        if source_logs is not None:
            from .paper_measurements import extend_terminal_accounting, SAMPLE_KEYS

            # A compact paper refresh may reuse independently audited terminal
            # terms when their source identity and measurement implementation agree.
            prior_terminal = output / "terminal.csv"
            prior_identity = (
                previous_config.get("scientific_identity", {})
                if previous_config
                else {}
            )
            reuse_terminal = (
                prior_terminal.is_file()
                and prior_identity.get("source_manifest_sha256")
                == file_sha256(source / "analysis_manifest.json")
                and prior_identity.get("measurement_sources", {}).get(
                    "paper_measurements.py"
                )
                == measurement_sources()["paper_measurements.py"]
                and prior_identity.get("terminal_supplement", {}).get("source_logs")
                == str(Path(source_logs).resolve())
            )
            if reuse_terminal:
                previous_summary = read_object(output / "summary.json")
                if (
                    file_sha256(prior_terminal)
                    != previous_summary["numerical_files"]["terminal.csv"]
                ):
                    raise TheoryError("Saved terminal accounting cache was modified")
                saved = pd.read_csv(
                    prior_terminal,
                    dtype={k: str for k in SAMPLE_KEYS},
                    keep_default_na=False,
                    float_precision="round_trip",
                )
                columns = [
                    name for name in saved if name.startswith("terminal_accounting_")
                ]
                left = tables["endpoint"].copy()
                for key in SAMPLE_KEYS:
                    left[key] = left[key].astype(str)
                left = left.drop(columns=[name for name in columns if name in left])
                tables["endpoint"] = left.merge(
                    saved[list(SAMPLE_KEYS) + columns],
                    on=list(SAMPLE_KEYS),
                    how="left",
                    validate="one_to_one",
                )
            from .reduce import _resolve_theory_devices
            from tqdm.auto import tqdm

            supplement_device = _resolve_theory_devices(device)[0]
            with tqdm(
                total=0,
                desc="[Theory] Terminal records",
                unit="record",
            ) as progress:

                def advance(completed, total):
                    progress.total = total
                    progress.update(completed - progress.n)

                supplement, supplement_receipt = extend_terminal_accounting(
                    tables["endpoint"],
                    manifest=manifest,
                    source_logs=Path(source_logs),
                    device=supplement_device,
                    progress=advance,
                )
            if reuse_terminal:
                supplement_receipt = prior_identity["terminal_supplement"]
            if not supplement.empty:
                keys = list(SAMPLE_KEYS)
                overlap = (set(supplement) & set(tables["endpoint"])) - set(keys)
                tables["endpoint"] = (
                    tables["endpoint"]
                    .drop(columns=list(overlap))
                    .merge(supplement, on=keys, how="left", validate="one_to_one")
                )
        # Always save available diagnostic inputs, independently of figure selection.
        frames, nonfeedback = build_nonfeedback_plot_inputs(
            tables, config=config, diagnostics=True
        )
        auxiliary = _auxiliary(frames, nonfeedback)
        feedback_frames, feedback = build_feedback_plot_inputs(tables, config=config)
        auxiliary.update(_auxiliary(feedback_frames, feedback))
        frames.update(feedback_frames)
        figures = {**nonfeedback["figures"], **feedback["figures"]}
        identity = {
            "config": config,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "measurement_sources": measurement_sources(),
            "source_analysis_hash": manifest["analysis_hash"],
            "source_manifest_sha256": file_sha256(source / "analysis_manifest.json"),
            "terminal_supplement": supplement_receipt,
        }
        scientific_hash = canonical_hash(_clean(identity))
        audit = _clean(
            {
                "schema_version": 1,
                "audit_version": "paper-audit-1",
                "scientific_hash": scientific_hash,
                "nonfeedback": nonfeedback["audit"],
                "feedback": feedback["audit"],
                "terminal_supplement": supplement_receipt,
                "blocking": bool(
                    nonfeedback["audit"].get("blocking", False)
                    or feedback["audit"].get("blocking", False)
                ),
            }
        )
        failed_figures = {
            name: m
            for name, m in figures.items()
            if m["status"] in {"error", "blocked", "missing"}
        }
        audit["blocking"] = audit["blocking"] or bool(failed_figures)
        if audit["blocking"]:
            failed = output.parent / ".failed-attempts" / output.name / scientific_hash
            failed.mkdir(parents=True, exist_ok=True)
            atomic_write_json(failed / "audit.json", audit)
            atomic_write_json(failed / "figure_status.json", _clean(figures))
            for name, frame in auxiliary.items():
                _save_table(failed, name + ".csv", frame, {})
            raise TheoryError(
                f"Paper correctness audit blocked publication; offending identities and reasons: {failed}"
            )
        previous = (
            read_object(output / "figure_manifest.json")
            if (output / "figure_manifest.json").exists()
            else {}
        )
        run_config = _clean(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "metric_schema_version": METRIC_SCHEMA_VERSION,
                "scientific_config": config,
                "scientific_identity": identity,
                "scientific_hash": scientific_hash,
                "registry_version": REGISTRY_VERSION,
                "source_analysis": {
                    "path": str(source),
                    "analysis_hash": manifest["analysis_hash"],
                    "mode": mode,
                },
                "provenance": {
                    "source_manifest_sha256": identity["source_manifest_sha256"],
                    "manuscript_sha256": manifest.get(
                        "manuscript_sha256",
                        manifest.get("endpoint_identity", {}).get("manuscript_sha256"),
                    ),
                    "matching_latex_source": "not available in checkout; PDF mapping retained",
                    "center_metadata": manifest.get("center_metadata"),
                    "support_metadata": manifest.get("support_metadata"),
                    "scheduler_adapter": manifest.get("scheduler_adapter"),
                    "terminal_supplement": supplement_receipt,
                },
            }
        )
        archive_previous = (
            previous_config is not None
            and previous_config["scientific_hash"] != scientific_hash
        )
        with staged_publication(output, archive_previous=archive_previous) as stage:
            numerical_files = {}
            plot_data = {}
            for entry in paper_registry(diagnostics=True):
                name = entry["stem"]
                if name not in figures:
                    raise TheoryError(f"Missing paper measurement contract: {name}")
                if name in frames and len(frames[name].columns):
                    plot_data[name] = _save_table(
                        stage, f"plot_data/{name}.csv", frames[name], numerical_files
                    )
            for name, frame in auxiliary.items():
                _save_table(stage, f"audit_data/{name}.csv", frame, numerical_files)
            for name, table in (
                ("initial", tables["initial"]),
                ("terminal", tables["endpoint"]),
            ):
                _save_table(stage, name + ".csv", table, numerical_files)
            _save_table(
                stage,
                "failed.csv",
                pd.DataFrame(columns=["record_id", "reason"]),
                numerical_files,
            )
            logical = {
                "schema_version": 1,
                "source_analysis": str(source),
                "source_manifest_sha256": identity["source_manifest_sha256"],
                "tables": {
                    name: {"path": str(source / name), "sha256": digest}
                    for name, digest in manifest["numerical_files"].items()
                    if name.startswith(
                        (
                            "trajectory_metrics/",
                            "integration_shards/",
                            "dose_shards/",
                            "controls_shards/",
                        )
                    )
                },
                "reading_policy": "Authoritative scalar shards; analysis only. Plotting reads compact plot_data CSVs.",
            }
            for name, value in (
                ("run_config.json", run_config),
                ("audit.json", audit),
                ("logical_tables.json", logical),
            ):
                atomic_write_json(stage / name, value)
                numerical_files[name] = file_sha256(stage / name)
            atomic_write_json(
                stage / "registry.json",
                {
                    "version": REGISTRY_VERSION,
                    "figures": paper_registry(diagnostics=True),
                },
            )
            summary = _clean(
                {
                    "schema_version": 1,
                    "complete": True,
                    "scientific_hash": scientific_hash,
                    "figures": figures,
                    "plot_data": plot_data,
                    "numerical_files": numerical_files,
                    "counts": {
                        "initial_rows": len(tables["initial"]),
                        "terminal_rows": len(tables["endpoint"]),
                        "trajectory_rows": len(tables["trajectory"]),
                    },
                }
            )
            atomic_write_json(stage / "summary.json", summary)
            load_paper_inputs(stage, expected_config=config, diagnostics=diagnostics)
            render_paper(stage, diagnostics=diagnostics)
            retire_obsolete_figures(stage, previous)
        retirement = retire_legacy_figures(root, output, source, manifest)
        atomic_write_json(output / "migration.json", retirement)
        print(f"Paper outputs: {output}", flush=True)
    return output
