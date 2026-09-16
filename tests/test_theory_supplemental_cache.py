"""Unexecuted specifications for stable, immutable supplemental collections."""
from copy import deepcopy

import pandas as pd

from utils.common.io import atomic_write_frame_parquet, atomic_write_json, canonical_hash, file_sha256
from utils.experiments.theory.supplemental_cache import existing_collection, scientific_base_receipt


def test_scientific_base_receipt_ignores_runtime_and_paths_but_pins_scalars():
    result = {"directory": "/first", "provenance": {"analysis_hash": "science", "execution": {"workers": 4}},
              "files": {"table": {"path": "/first/table", "sha256": "data", "rows": 3}},
              "logical_joins": {"view": {"base": "table", "how": "exact_one_to_one_additive"}}}
    moved = deepcopy(result)
    moved["directory"] = "/second"
    moved["files"]["table"]["path"] = "/second/table"
    moved["provenance"]["execution"] = {"workers": 0, "resumed": True}
    assert scientific_base_receipt(result) == scientific_base_receipt(moved)
    moved["files"]["table"]["sha256"] = "different-science"
    assert scientific_base_receipt(result) != scientific_base_receipt(moved)


def saved_collection(tmp_path):
    data = tmp_path / "scalar.parquet"
    atomic_write_frame_parquet(pd.DataFrame({"value": [1., 2.]}), data)
    identity = {"policy": "fixed", "base_scientific": "science"}
    manifest = {"complete": True, "kind": "test", "identity": identity,
                "files": {"scalar": {"path": str(data), "sha256": file_sha256(data), "rows": 2}},
                "provenance": {"analysis_hash": canonical_hash(identity), "execution": {"workers": 4}}}
    atomic_write_json(tmp_path / "manifest.json", manifest)
    return identity, manifest


def test_completed_collection_reuse_preserves_creation_manifest(tmp_path):
    identity, manifest = saved_collection(tmp_path)
    before = (tmp_path / "manifest.json").read_bytes()
    assert existing_collection(tmp_path, identity=identity, kind="test") == manifest
    assert (tmp_path / "manifest.json").read_bytes() == before
    assert existing_collection(tmp_path, identity={"policy": "changed"}, kind="test") is None
    assert existing_collection(tmp_path, identity=identity, kind="other") is None


def test_collection_reuse_checks_scalar_hash_and_metadata_rows(tmp_path):
    identity, manifest = saved_collection(tmp_path)
    manifest["files"]["scalar"]["rows"] = 3
    atomic_write_json(tmp_path / "manifest.json", manifest)
    assert existing_collection(tmp_path, identity=identity, kind="test") is None
    manifest["files"]["scalar"]["rows"] = 2
    manifest["files"]["scalar"]["sha256"] = "wrong"
    atomic_write_json(tmp_path / "manifest.json", manifest)
    assert existing_collection(tmp_path, identity=identity, kind="test") is None


def test_incomplete_or_wrong_analysis_receipt_is_never_reused(tmp_path):
    identity, manifest = saved_collection(tmp_path)
    manifest["complete"] = False
    atomic_write_json(tmp_path / "manifest.json", manifest)
    assert existing_collection(tmp_path, identity=identity, kind="test") is None
    manifest["complete"] = True
    manifest["provenance"]["analysis_hash"] = "unrelated"
    atomic_write_json(tmp_path / "manifest.json", manifest)
    assert existing_collection(tmp_path, identity=identity, kind="test") is None
