"""Scalar-only discovery-suite identity, saved-suite selection, and table access."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.common.io import atomic_write_json
from .contracts import (
    TheoryError,
    analysis_parent,
    digest_file,
    numerical_config,
    read_object,
)

SCHEMA_VERSION = 1
FORMULA_VERSION = "theory-candidates-1.0"
ROW_KEYS = ["run_id", "original_index", "target_id", "seed", "step_index"]


def candidate_parent(project_root, **config):
    return analysis_parent(project_root, **config).parent / "theory_candidates"


def recompute_command(config):
    return (
        "./run_all.sh --recompute-experiments --figure-suite candidates "
        f"--model {config['model_name']} --scheduler {config['scheduler_name']} "
        f"--g {config['guidance_scale']:g} --T {config['num_inference_steps']} --N {config['num_seeds']}"
    )


def _hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _relative(bundle, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise TheoryError(f"Unsafe candidate table path: {name}")
    result = bundle / path
    if result.is_symlink() or not result.resolve().is_relative_to(bundle.resolve()):
        raise TheoryError(f"Unsafe candidate table: {result}")
    return result


def validate_candidate_bundle(bundle, expected_config=None, *, require_complete=False):
    """Validate only declared scalar files; never read integration payloads/tensors."""
    bundle = Path(bundle).resolve()
    manifest = read_object(bundle / "analysis_manifest.json")
    config = manifest.get("config", {})
    command = (
        recompute_command(config)
        if config
        else "./run_all.sh --recompute-experiments --figure-suite candidates"
    )
    if manifest.get("schema_version") != SCHEMA_VERSION or not manifest.get(
        "endpoint_complete"
    ):
        raise TheoryError(
            f"Missing/incompatible candidate endpoint measurements: {bundle}. Run {command}"
        )
    if require_complete and not manifest.get("complete"):
        raise TheoryError(
            f"Candidate integration stage is incomplete: {bundle}. Resume {command}"
        )
    if expected_config is not None and config != numerical_config(**expected_config):
        raise TheoryError(f"Candidate scientific configuration differs: {bundle}")
    files = manifest.get("numerical_files")
    if not isinstance(files, dict) or not files:
        raise TheoryError(f"Missing candidate numerical file contract. Run {command}")
    for name, expected in files.items():
        path = _relative(bundle, name)
        if path.suffix not in {".parquet", ".json", ".csv"}:
            raise TheoryError(f"Non-scalar file in plot contract: {name}")
        if not path.is_file() or digest_file(path) != expected:
            raise TheoryError(
                f"Missing/stale candidate measurement {path}. Run {command}"
            )
    required = {
        "initial.parquet",
        "endpoint.parquet",
        "reference_initial.parquet",
        "reference_snr.parquet",
        "bank_geometry.parquet",
        "registry.json",
    }
    if not required <= files.keys():
        raise TheoryError(
            f"Missing candidate tables {sorted(required - files.keys())}. Run {command}"
        )
    return manifest


def find_candidate_bundle(project_root, **config):
    config = numerical_config(**config)
    parent = candidate_parent(project_root, **config)
    index = read_object(parent / "index.json")
    rows = [row for row in index.get("analyses", []) if row.get("config") == config]
    if len(rows) != 1 or not _hash(rows[0].get("analysis_hash")):
        raise TheoryError(
            f"Missing/ambiguous candidate suite for this scientific run. Run {recompute_command(config)}"
        )
    bundle = parent / rows[0]["analysis_hash"]
    manifest = read_object(bundle / "analysis_manifest.json")
    if (
        not manifest.get("complete")
        or manifest.get("config") != config
        or manifest.get("analysis_hash") != bundle.name
    ):
        raise TheoryError(
            f"Incomplete candidate publication {bundle}. Run {recompute_command(config)}"
        )
    return bundle


def selected_suite(project_root, config, explicit=None):
    if explicit is not None:
        if explicit not in {"main", "candidates"}:
            raise TheoryError(f"Unknown figure suite: {explicit}")
        return explicit
    config = numerical_config(**config)
    path = analysis_parent(project_root, **config).parent / "theory_suite.json"
    if not path.exists():
        return "main"
    entries = [
        row for row in read_object(path).get("runs", []) if row.get("config") == config
    ]
    if not entries:
        return "main"
    if len(entries) != 1 or entries[0].get("figure_suite") not in {
        "main",
        "candidates",
    }:
        raise TheoryError(f"Ambiguous saved theory suite: {path}")
    return entries[0]["figure_suite"]


def remember_suite(project_root, config, suite, bundle):
    config = numerical_config(**config)
    path = analysis_parent(project_root, **config).parent / "theory_suite.json"
    value = read_object(path) if path.exists() else {"schema_version": 1, "runs": []}
    value["runs"] = [row for row in value["runs"] if row.get("config") != config]
    value["runs"].append(
        {
            "config": config,
            "figure_suite": suite,
            "analysis_hash": Path(bundle).name,
            "bundle": Path(bundle)
            .resolve()
            .relative_to(Path(project_root).resolve())
            .as_posix(),
        }
    )
    atomic_write_json(path, value)


def publish_candidate_index(project_root, config, bundle):
    config = numerical_config(**config)
    path = candidate_parent(project_root, **config) / "index.json"
    index = read_object(path) if path.exists() else {"analyses": []}
    index["analyses"] = [
        row for row in index["analyses"] if row.get("config") != config
    ]
    index["analyses"].append(
        {
            "config": config,
            "analysis_hash": Path(bundle).name,
            "figure_suite": "candidates",
        }
    )
    atomic_write_json(path, index)
    remember_suite(project_root, config, "candidates", bundle)


def read_tables(bundle, *, include_dose=True, include_controls=True):
    """Read a declared scalar bundle, joining integration strictly by sample keys."""
    bundle = Path(bundle)
    manifest = read_object(bundle / "analysis_manifest.json")
    files = manifest["numerical_files"]
    tables = {}
    for name in (
        "initial",
        "endpoint",
        "reference_initial",
        "reference_snr",
        "bank_geometry",
    ):
        relative = name + ".parquet"
        if relative in files:
            tables[name] = pd.read_parquet(_relative(bundle, relative))
    for name, directory in (
        ("trajectory", "trajectory_metrics"),
        ("integration", "integration_shards"),
        ("dose", "dose_shards"),
        ("controls", "controls_shards"),
    ):
        if (name == "dose" and not include_dose) or (
            name == "controls" and not include_controls
        ):
            continue
        parts = [
            pd.read_parquet(_relative(bundle, relative))
            for relative in sorted(files)
            if relative.startswith(directory + "/") and relative.endswith(".parquet")
        ]
        tables[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not tables.get("integration", pd.DataFrame()).empty:
        left, right = tables["trajectory"], tables["integration"]
        if left.duplicated(ROW_KEYS).any() or right.duplicated(ROW_KEYS).any():
            raise TheoryError(
                "Duplicate candidate sample identities during integration join"
            )
        overlap = (set(left.columns) & set(right.columns)) - set(ROW_KEYS)
        inherited_qa = {
            "candidate_gram_roundoff_allowance_l2",
            "candidate_integral_identity_residual",
            "candidate_integrated_log_probability_gain",
            "candidate_quadrature_budget_exhausted",
            "candidate_quadrature_evaluations",
            "candidate_quadrature_refinements",
        }
        if overlap - inherited_qa:
            raise TheoryError(
                f"Integration overwrites endpoint measurements: {sorted(overlap - inherited_qa)}"
            )
        # Keep the original schema-3 QA beside the new integration estimates.
        left = left.rename(columns={name: "base_v3_" + name for name in overlap})
        membership = right[ROW_KEYS].merge(
            left[ROW_KEYS], on=ROW_KEYS, how="left", indicator=True
        )
        if membership["_merge"].eq("left_only").any():
            raise TheoryError("Integration contains orphan sample identities")
        merged = left.merge(
            right,
            on=ROW_KEYS,
            how="left",
            validate="one_to_one",
            indicator="_integration_join",
        )
        absent = merged["_integration_join"].eq("left_only")
        for name in right.columns:
            if name.endswith(("_status", "_estimated_sign")):
                merged.loc[absent, name] = "not_applicable"
            elif name == "candidate_integration_reason":
                merged.loc[absent, name] = (
                    "endpoint_segment_not_applicable_or_unavailable"
                )
        tables["trajectory"] = merged.drop(columns="_integration_join")
    summaries = {}
    for relative in sorted(files):
        if relative.startswith("summaries/") and relative.endswith(".parquet"):
            summaries[Path(relative).stem] = pd.read_parquet(
                _relative(bundle, relative)
            )
    tables["summaries"] = summaries
    tables["manifest"] = manifest
    return tables


def validate_candidate_sources(bundle, project_root):
    manifest = read_object(Path(bundle) / "analysis_manifest.json")
    root = Path(project_root)
    for relative, expected in manifest.get("source_metadata_files", {}).items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or digest_file(path) != expected:
            raise TheoryError(
                f"Stale candidate source metadata: {path}. Run {recompute_command(manifest['config'])}"
            )
    for relative, expected in manifest.get("source_marker_inventory", {}).items():
        if sorted(path.name for path in (root / relative).glob("*.json")) != expected:
            raise TheoryError(
                f"Candidate source completion inventory changed: {relative}"
            )


def all_candidate_bundles(project_root):
    """Resolve published discovery bundles, never pool their scientific laws."""
    bundles = []
    for path in sorted(
        (Path(project_root) / "outputs").glob("*/theory_candidates/index.json")
    ):
        for entry in read_object(path).get("analyses", []):
            if not _hash(entry.get("analysis_hash")):
                raise TheoryError(f"Unsafe candidate index hash: {path}")
            bundle = path.parent / entry["analysis_hash"]
            manifest = read_object(bundle / "analysis_manifest.json")
            if manifest.get("complete") and manifest.get("config") == entry.get(
                "config"
            ):
                bundles.append(bundle)
    return bundles
