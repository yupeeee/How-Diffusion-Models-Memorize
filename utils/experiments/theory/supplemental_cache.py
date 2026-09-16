"""Stable scientific receipts and immutable completed supplemental collections.

Execution placement and mutable upstream manifest formatting are provenance,
not measurement identity. Existing collections are reused only after their
scientific identity, scalar hashes, and Parquet row receipts are verified.
"""
from pathlib import Path

from utils.common.io import canonical_hash, file_sha256
from .contracts import TheoryError, read_object
from .paper_contracts import reject_symlinks


def scientific_base_receipt(result):
    """Describe measured scalar contents without paths or execution metadata."""
    return {
        "analysis_hash": result["provenance"]["analysis_hash"],
        "scalar_sources": {
            name: {"sha256": spec["sha256"], "rows": spec["rows"]}
            for name, spec in result["files"].items()
        },
        "logical_joins": result.get("logical_joins", {}),
    }


def existing_collection(directory, *, identity, kind):
    """Return a verified completed manifest; invalid/missing receipts need rebuild.

    Parquet metadata verifies row counts without loading the scalar columns.
    Reusing the original manifest also preserves its artifact-creation execution
    provenance: a resumed run must not rewrite a measurement's identity receipt.
    """
    path = reject_symlinks(Path(directory) / "manifest.json")
    if not path.is_file():
        return None
    try:
        saved = read_object(path)
        digest = canonical_hash(identity)
        if (saved.get("complete") is not True or saved.get("kind") != kind
                or canonical_hash(saved["identity"]) != digest
                or saved["provenance"]["analysis_hash"] != digest
                or not isinstance(saved["files"], dict) or not saved["files"]):
            return None
        from pyarrow.parquet import ParquetFile
        for spec in saved["files"].values():
            scalar = reject_symlinks(spec["path"])
            rows = spec["rows"]
            if (isinstance(rows, bool) or not isinstance(rows, int) or rows < 0
                    or not scalar.is_file() or file_sha256(scalar) != spec["sha256"]
                    or ParquetFile(scalar).metadata.num_rows != rows):
                return None
        return saved
    except (TheoryError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
